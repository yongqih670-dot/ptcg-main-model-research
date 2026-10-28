"""One finite, project-owned resource audit endpoint; no model or GPU jobs."""
import base64
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import sys
import time


ROOT=Path.home().resolve()/'ptcg-hybrid-v3'
SOURCES={'resource_worker.py','payload_worker.py','ptcg_lab/__init__.py','ptcg_lab/io.py',
         'ptcg_lab/server_workspace.py','ptcg_lab/systemd_job.py','ptcg_lab/unit_io_guard.py',
         'ptcg_lab/process_guardian.py','ptcg_lab/server_resource_worker.py','ptcg_lab/io_pacing.py','ptcg_lab/host_block_io.py','ptcg_lab/bounded_io.py','ptcg_lab/bounded_read.py'}


def path_inside(path):
    path=Path(path)
    if ROOT.is_symlink() or ROOT.stat().st_uid!=os.getuid() or ROOT.stat().st_mode&0o077:raise ValueError('Unowned/nonprivate project root')
    parts=path.relative_to(ROOT).parts;current=ROOT
    for part in parts:
        if part in ('.','..'):raise ValueError('Unsafe owned path')
        current=current/part
        if current.is_symlink() or (current.exists() and current.stat().st_uid!=os.getuid()):raise ValueError('Owned path identity differs')
    if ROOT.resolve() not in path.resolve().parents:raise ValueError('Write escaped canonical project')
    return path


def save(path,value):
    path=path_inside(path);temp=path_inside(path.with_name(path.name+'.'+str(os.getpid())+'.tmp'))
    with temp.open('x') as output:json.dump(value,output,indent=2);output.write('\n');output.flush();os.fsync(output.fileno())
    temp.replace(path)


class Endpoint:
    def __init__(self):self.job=None;self.handle=None;self.systemd=None
    def initialize(self,p):
        if self.job:raise RuntimeError('Audit endpoint already prepared')
        expected={'project_id':p['project_id'],'uid':os.getuid(),'root':str(ROOT)}
        marker=path_inside(ROOT/'.ptcg-owner.json')
        if json.loads(marker.read_text())!=expected:raise ValueError('Project owner differs')
        if not re.fullmatch('[A-Za-z0-9_-]{12,100}',p['run_id']):raise ValueError('Invalid audit identity')
        if set(p['files'])!=SOURCES:raise ValueError('Unexpected source bundle')
        if p['profile'] not in ['io_stop','heartbeat_loss','guardian_fault','disk_guard','normal_completion']:raise ValueError('Unknown bounded probe')
        if abs(p['client_unix']-time.time())>5 or not 90<p['deadline_unix']-time.time()<300:raise ValueError('Invalid audit time boundary')
        if p['agent_sha256']!=globals().get('PTCG_AGENT_SHA256'):raise ValueError('Endpoint source identity differs')
        self.job=path_inside(ROOT/'jobs'/p['run_id']);self.job.mkdir(mode=0o700)
        for directory in ['control','control/ptcg_lab','inputs']:(self.job/directory).mkdir(mode=0o700)
        hashes={}
        for name,record in p['files'].items():
            data=base64.b64decode(record['base64'],validate=True)
            if hashlib.sha256(data).hexdigest()!=record['sha256']:raise ValueError('Frozen source hash differs')
            with path_inside(self.job/'control'/name).open('xb') as output:output.write(data);output.flush();os.fsync(output.fileno())
            hashes[name]=record['sha256']
        with path_inside(self.job/'control/endpoint-agent.py').open('xb') as output:
            output.write(globals()['_ptcg_code']);output.flush();os.fsync(output.fileno())
        hashes['endpoint-agent.py']=p['agent_sha256'];save(self.job/'source-hashes.json',hashes)
        sys.path.insert(0,str(self.job/'control'))
        from ptcg_lab import systemd_job
        from ptcg_lab.server_workspace import owned_root,admit
        owned_root(ROOT,p['project_id']);self.systemd=systemd_job
        admission=admit(ROOT,64*2**20)
        self.ticket={'kind':'resource_audit','project_root':str(ROOT),'project_id':p['project_id'],
            'job_directory':str(self.job),'control_root':str(self.job/'control'),'ticket_path':str(self.job/'ticket.json'),
            'run_id':p['run_id'],'unit':p['unit'],'lease_epoch':p['lease_epoch'],'deadline_unix':p['deadline_unix'],
            'cpu_quota_cores':2,'memory_max_bytes':128*2**20,'workspace_hard_max_bytes':10*2**30,'stop_grace_seconds':2,
            'audit_profile':p['profile'],'resource_profile':{'read_bps':4*2**20,'write_bps':4*2**20,
            'burst_bytes':2*2**20,'stop_debt_bytes':8*2**20,'max_gap_seconds':1}}
        if p['profile']=='normal_completion':
            inputs=p['input_wheels']
            if set(inputs)!={'pip','attrs'}:raise ValueError('Only fixed tiny positive probe inputs allowed')
            if inputs['pip']['version']!='24.2' or inputs['attrs']['version']!='26.1.0':raise ValueError('Positive probe versions changed')
            for row in inputs.values():
                if Path(row['file']).name!=row['file'] or ':' in row['file'] or not 0<row['bytes']<=2*2**20:raise ValueError('Unsafe small probe input')
            self.ticket['input_wheels']=inputs
        save(self.job/'ticket.json',self.ticket);signal.alarm(max(1,int(p['deadline_unix']-time.time()+5)))
        return {'prepared':True,'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                'capacity':admission,'source_hashes':hashes,'ticket':self.ticket}
    def put_input(self,name,data):
        from ptcg_lab.io_pacing import BytePacer
        row=self.ticket['input_wheels'][name]
        if len(data)!=row['bytes'] or hashlib.sha256(data).hexdigest()!=row['sha256']:raise ValueError('Probe input hash differs')
        pacer=BytePacer(2*2**20,self.ticket['deadline_unix']);path=path_inside(self.job/'inputs'/row['file'])
        with path.open('xb') as output:
            for offset in range(0,len(data),256*1024):
                block=data[offset:offset+256*1024];pacer.consume(len(block));output.write(block)
            output.flush();os.fsync(output.fileno())
        return {'stored':True,'sha256':row['sha256']}
    def pace(self,data):
        from ptcg_lab.io_pacing import BytePacer,paced_sha256
        if len(data)!=4*2**20:raise ValueError('Pacing probe has a fixed payload')
        path=path_inside(self.job/'control-paced.bin');deadline=self.ticket['deadline_unix'];rate=2*2**20
        pacer=BytePacer(rate,deadline);started=time.monotonic()
        with path.open('xb') as output:
            for offset in range(0,len(data),256*1024):
                block=data[offset:offset+256*1024];pacer.consume(len(block));output.write(block)
            output.flush();os.fsync(output.fileno())
        write_seconds=time.monotonic()-started
        reader=BytePacer(rate,deadline);started=time.monotonic();digest=paced_sha256(path,reader)
        result={'bytes':len(data),'write_seconds':write_seconds,'read_seconds':time.monotonic()-started,
                'bytes_per_second':rate,'sha256':digest};save(self.job/'control-pacing-proof.json',result);return result
    def start(self):
        if self.handle:raise RuntimeError('Audit service cannot relaunch')
        save(self.job/'control-heartbeat.json',{'at_unix':time.time()})
        with (ROOT/'control/fence.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);fence=ROOT/'control/lease.json';prior=json.loads(fence.read_text())
            if self.ticket['lease_epoch']<=prior['lease_epoch']:raise RuntimeError('Stale audit lease')
            if not self.systemd.terminal(self.systemd.inspect_service(prior['unit'])):raise RuntimeError('Prior job still unresolved')
            save(fence,{'unit':self.ticket['unit'],'lease_epoch':self.ticket['lease_epoch'],'run_id':self.ticket['run_id']})
            self.handle,_=self.systemd.start_verified(self.ticket);save(self.job/'identity.json',self.handle)
        return self.status(True)
    def status(self,heartbeat=True):
        if heartbeat:save(self.job/'control-heartbeat.json',{'at_unix':time.time()})
        props=self.systemd.inspect_service(self.ticket['unit']);boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        if self.systemd.identity(props,boot)!=self.handle:raise RuntimeError('Resource audit invocation changed')
        result={'identity':self.handle,'host_boot_id':boot,'server_unix':time.time(),'service_properties':props,
                'terminal':self.systemd.terminal(props),'artifacts':{}}
        for name in ['guardian-start.json','guardian-progress.json','guardian-result.json','guardian-ready.json',
                     'payload-started.json','child-ready.json','normal-done.json','checkpoint-marker.json','disk-refusal.json']:
            path=self.job/name
            if path.exists():result['artifacts'][name]=json.loads(path.read_text())
        for name in ['positive-pip-result.json','payload-exit-ready.json','guardian-exit-ack.json']:
            path=self.job/name
            if path.exists():result['artifacts'][name]=json.loads(path.read_text())
        short=self.job/'short-child-accounting.json'
        if short.exists():result['artifacts']['short-child-accounting.json']=json.loads(short.read_text())
        if result['terminal']:
            result['logs']={}
            for name in ['stdout.log','stderr.log']:
                path=self.job/name
                if path.exists():
                    if path.stat().st_size>256*1024:raise ValueError('Audit log frame bound')
                    data=path.read_bytes();result['logs'][name]={'base64':base64.b64encode(data).decode(),
                                    'sha256':hashlib.sha256(data).hexdigest()}
            pids={}
            for name in ['guardian-start.json','payload-started.json','child-ready.json']:
                record=result['artifacts'].get(name)
                if not record:continue
                proc=Path('/proc')/str(record['pid'])
                if proc.exists():
                    fields=(proc/'stat').read_text().rsplit(')',1)[1].split();pids[name]={'pid':record['pid'],'alive':fields[0]!='Z'}
                else:pids[name]={'pid':record['pid'],'alive':False}
            result['processes_after_terminal']=pids
        return result
    def close(self):
        if self.handle and not self.systemd.terminal(self.systemd.inspect_service(self.ticket['unit'])):
            self.systemd.stop_verified(self.handle)


def main():
    os.umask(0o077);endpoint=Endpoint()
    def stop(*_):raise TimeoutError('Resource control endpoint stopped')
    signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGHUP,stop)
    try:
        while True:
            line=sys.stdin.buffer.readline(1024**2+1)
            if not line:break
            if len(line)>1024**2:raise ValueError('Control header too large')
            p=json.loads(line);op=p['operation'];seq=p['sequence']
            try:
                if op=='initialize':result=endpoint.initialize(p)
                elif op in ('pace','put_input'):
                    count=p['binary_bytes']
                    if (op=='pace' and count!=4*2**20) or not 0<count<=4*2**20:raise ValueError('Unexpected bounded payload')
                    data=b''
                    while len(data)<count:
                        block=sys.stdin.buffer.read(count-len(data))
                        if not block:raise EOFError('Pacing payload incomplete')
                        data+=block
                    result=endpoint.pace(data) if op=='pace' else endpoint.put_input(p['name'],data)
                elif op=='start':result=endpoint.start()
                elif op=='status':result=endpoint.status(p.get('heartbeat',True))
                elif op=='close':result={'closed':True}
                else:raise ValueError('Unreviewed resource operation')
                print(json.dumps({'sequence':seq,'ok':True,'result':result}),flush=True)
                if op=='close':break
            except Exception as exc:
                print(json.dumps({'sequence':seq,'ok':False,'error':type(exc).__name__+': '+str(exc)}),flush=True);raise
    finally:endpoint.close()


if __name__=='__main__':main()
