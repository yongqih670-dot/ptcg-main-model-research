"""Framed, finite control/transfer endpoint for one owned environment service.

Executed as a standalone program by the user's SSH process. Only project
sources, one staged wheel, immutable install receipts and ACKs are writable.
"""
import base64
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


ROOT=Path.home().resolve()/'ptcg-hybrid-v3'
ALLOWED_FILES={'environment_worker.py','ptcg_lab/__init__.py','ptcg_lab/io.py',
               'ptcg_lab/systemd_job.py','ptcg_lab/server_workspace.py','ptcg_lab/server_environment_worker.py',
               'ptcg_lab/bounded_io.py','ptcg_lab/io_pacing.py','ptcg_lab/bounded_read.py'}


def atomic(path,value):
    path=write_path(path);temp=write_path(path.with_name(path.name+'.'+str(os.getpid())+'.tmp'))
    with temp.open('x',encoding='utf-8') as output:
        json.dump(value,output,ensure_ascii=False,indent=2);output.write('\n');output.flush();os.fsync(output.fileno())
    temp.replace(path)


def write_path(path):
    path=Path(path);uid=os.getuid() if hasattr(os,'getuid') else ROOT.stat().st_uid
    if ROOT.is_symlink() or ROOT.stat().st_uid!=uid:raise ValueError('Project directory ownership differs')
    try:parts=path.relative_to(ROOT).parts
    except ValueError:raise ValueError('Write outside dedicated project')
    current=ROOT
    for part in parts:
        if part in ('.','..'):raise ValueError('Unsafe write path')
        current=current/part
        if current.is_symlink():raise ValueError('Write traverses a symlink')
        if current.exists() and current.stat().st_uid!=uid:raise ValueError('Write target UID differs')
    if ROOT.resolve() not in path.resolve().parents:raise ValueError('Write realpath escaped project')
    return path


def digest(path):
    result=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):result.update(block)
    return result.hexdigest()


def exact(stream,size):
    chunks=[];remaining=size
    while remaining:
        block=stream.read(remaining)
        if not block:raise EOFError('Truncated control frame')
        chunks.append(block);remaining-=len(block)
    return b''.join(chunks)


class Agent:
    def __init__(self):self.job=None;self.ticket=None;self.handle=None;self.staged=None;self.started=False;self.clock_checked_at=None
    def check_clock(self,p):
        if self.job is not None:raise RuntimeError('Clock handshake must precede initialization')
        now=time.time()
        if abs(now-p['client_unix'])>5:raise ValueError('Clock skew exceeds deadline tolerance')
        self.clock_checked_at=now
        return {'server_unix':now,'clock_tolerance_seconds':5}
    def initialize(self,p):
        if self.job is not None:raise RuntimeError('Control already initialized')
        if ROOT.is_symlink():raise ValueError('Symlink workspace')
        owner=ROOT/'.ptcg-owner.json'
        expected={'project_id':p['project_id'],'uid':os.getuid(),'root':str(ROOT)}
        if (ROOT.stat().st_uid!=os.getuid() or ROOT.stat().st_mode&0o077 or owner.is_symlink()
                or owner.stat().st_uid!=os.getuid() or json.loads(owner.read_text())!=expected):raise ValueError('Unowned project')
        if not re.fullmatch(r'[A-Za-z0-9_-]{12,100}',p['run_id']):raise ValueError('Invalid run identity')
        # Measure clocks with a small frame; a larger manifest can take seconds
        # in transit and must not be mistaken for actual host clock skew.
        if self.clock_checked_at is None or not 0<=time.time()-self.clock_checked_at<=60:
            raise ValueError('Fresh bounded clock handshake is required')
        if not 100<p['deadline_unix']-time.time()<=7200:raise ValueError('Invalid finite control deadline')
        staged=p.get('mode') in ('staged_direct_recovery','staged_range_repair')
        allowed=ALLOWED_FILES|({'ptcg_lab/staged_environment.py'} if staged else set())
        if p.get('mode')=='staged_range_repair':allowed.add('ptcg_lab/range_artifacts.py')
        if set(p['files'])!=allowed:raise ValueError('Unexpected control sources')
        self.job=write_path(ROOT/'jobs'/p['run_id']);self.job.mkdir(mode=0o700)
        for name in ['control','control/ptcg_lab','stage','requests','acks','evictions','receipts','logs']:
            write_path(self.job/name).mkdir(mode=0o700,exist_ok=True)
        source_hashes={}
        if globals().get('PTCG_AGENT_SHA256')!=p['agent_sha256']:raise ValueError('Executing endpoint source differs')
        with write_path(self.job/'control/endpoint-agent.py').open('xb') as output:
            output.write(globals()['_ptcg_code']);output.flush();os.fsync(output.fileno())
        source_hashes['endpoint-agent.py']=p['agent_sha256']
        for name,item in p['files'].items():
            data=base64.b64decode(item['base64'],validate=True)
            if hashlib.sha256(data).hexdigest()!=item['sha256']:raise ValueError('Control source changed')
            target=write_path(self.job/'control'/name)
            with target.open('xb') as output:output.write(data);output.flush();os.fsync(output.fileno())
            source_hashes[name]=item['sha256']
        data=base64.b64decode(p['manifest_base64'],validate=True)
        if hashlib.sha256(data).hexdigest()!=p['manifest_sha256']:raise ValueError('Manifest changed')
        manifest_path=write_path(self.job/'wheel-manifest.json')
        with manifest_path.open('xb') as output:output.write(data);output.flush();os.fsync(output.fileno())
        self.manifest=json.loads(data)
        if len(self.manifest['files'])!=44:raise ValueError('Wrong environment wheel count')
        self.reserved_future=self.manifest['future_training_capsule_bytes']+self.manifest['runtime_buffer_bytes']+128*1024**2
        for item in self.manifest['files']:
            if (Path(item['file']).name!=item['file'] or chr(92) in item['file'] or ':' in item['file']
                    or not item['file'].endswith('.whl') or not re.fullmatch('[0-9a-f]{64}',item['sha256'])):
                raise ValueError('Unsafe staged wheel identity')
        sys.path.insert(0,str(self.job/'control'))
        from ptcg_lab import systemd_job
        from ptcg_lab.server_workspace import admit
        self.systemd=systemd_job;self.admit=admit
        if staged:
            from ptcg_lab.staged_environment import validate_recovery_owner
            if self.manifest.get('mode')!=p['mode']:raise ValueError('Staged mode differs')
            prior=validate_recovery_owner(ROOT,self.manifest)
            for item in self.manifest['files']:
                fragment=item.get('range_repair',{}).get('resume')
                if fragment:
                    path=write_path(ROOT/fragment['relative_path'])
                    if not path.is_file() or path.stat().st_size!=fragment['bytes']:
                        raise ValueError('Registered old fragment path/size differs before service launch')
            atomic(self.job/'prior-environment-owner.json',prior)
            self.admit(ROOT,self.manifest['additional_peak_bytes'])
        else:
            self.admit(ROOT,self.manifest['estimated_peak_bytes'])
            if (ROOT/'.venv').exists():raise RuntimeError('Existing environment needs explicit recovery')
        self.ticket={'kind':'environment_setup','project_root':str(ROOT),'project_id':p['project_id'],
                     'job_directory':str(self.job),'control_root':str(self.job/'control'),
                     'ticket_path':str(self.job/'ticket.json'),'unit':p['unit'],'run_id':p['run_id'],
                     'lease_epoch':p['lease_epoch'],'deadline_unix':p['deadline_unix'],
                     'memory_max_bytes':4*2**30,'workspace_hard_max_bytes':10*2**30,
                     'manifest_sha256':p['manifest_sha256']}
        if staged:self.ticket.update(python_kind='venv',cpu_quota_cores=2,mode=p['mode'])
        atomic(self.job/'ticket.json',self.ticket);atomic(self.job/'source-hashes.json',source_hashes)
        signal.alarm(max(1,int(p['deadline_unix']-time.time())))
        gpu=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu',
                            '--format=csv,noheader,nounits'],text=True,capture_output=True,timeout=10)
        probe={'at_unix':time.time(),'returncode':gpu.returncode,'audit_uses_gpu':False,
               'preferred_gpu_rows':[line for line in gpu.stdout.splitlines() if line.split(',')[0].strip() in ('6','7')]}
        atomic(self.job/'gpu-probe.json',probe)
        return {'prepared':True,'host_boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                'root':str(ROOT),'capacity':self.admit(ROOT,0),'gpu_probe':probe,'source_hashes':source_hashes}
    def start(self):
        if self.ticket is None or self.started:raise RuntimeError('Service cannot be relaunched')
        fence=ROOT/'control/lease.json'
        with (ROOT/'control/fence.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            previous=json.loads(fence.read_text())
            if self.ticket['lease_epoch']<=previous['lease_epoch']:raise ValueError('Stale server lease epoch')
            if not self.systemd.terminal(self.systemd.inspect_service(previous['unit'])):
                raise RuntimeError('Prior service unresolved')
            atomic(fence,{'unit':self.ticket['unit'],'lease_epoch':self.ticket['lease_epoch'],'run_id':self.ticket['run_id']})
            self.handle,_=self.systemd.start_verified(self.ticket);self.started=True
            atomic(self.job/'identity.json',self.handle)
        return self.status()
    def status(self,index=None):
        props=self.systemd.inspect_service(self.ticket['unit']);boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        if self.handle and self.systemd.identity(props,boot)!=self.handle:raise RuntimeError('Invocation changed')
        try:capacity=self.admit(ROOT,0)
        except OSError as exc:capacity={'error':str(exc)}
        result={'identity':self.handle,'host_boot_id':boot,'server_unix':time.time(),'service_properties':props,
                'terminal':self.systemd.terminal(props),'capacity':capacity}
        group=props.get('ControlGroup')
        if group:
            cg=Path('/sys/fs/cgroup')/group.lstrip('/')
            result['kernel_limits']={name:(cg/name).read_text().strip() for name in ['cpu.max','memory.max','memory.swap.max','pids.max'] if (cg/name).is_file()}
        for name in ['progress.json','failure.json','environment-metrics.json','hard-protection.json','resources.json']:
            path=self.job/name
            if path.exists():result[name]=json.loads(path.read_text())
        if index is not None:
            receipt=self.job/'receipts'/f'{index:03d}.json'
            if receipt.exists():
                result['receipt']=json.loads(receipt.read_text());result['receipt_sha256']=digest(receipt)
                result['receipt_base64']=base64.b64encode(receipt.read_bytes()).decode()
                result['logs']={}
                for suffix in ['.log','-version.log']:
                    path=self.job/'logs'/f'{index:03d}{suffix}'
                    if path.stat().st_size>512*1024:raise ValueError('Install log archive frame limit')
                    data=path.read_bytes();result['logs'][path.name]={'base64':base64.b64encode(data).decode(),
                                                                   'sha256':hashlib.sha256(data).hexdigest()}
            result['evicted']=(self.job/'evictions'/f'{index:03d}.json').exists()
        return result
    def begin(self,index):
        if not self.started or self.staged is not None:raise RuntimeError('Overlapping staged artifact')
        if index and not (self.job/'evictions'/f'{index-1:03d}.json').exists():raise RuntimeError('Prior replica still active')
        item=self.manifest['files'][index];self.admit(ROOT,item['bytes']+item['expanded_bytes']+self.reserved_future)
        path=write_path(self.job/'stage'/(item['file']+'.partial'))
        stream=path.open('xb');self.staged={'index':index,'path':path,'stream':stream,'bytes':0,'hash':hashlib.sha256()}
        return {'index':index,'offset':0}
    def chunk(self,p,data):
        active=self.staged
        if active is None or p['index']!=active['index'] or p['offset']!=active['bytes']:
            raise ValueError('Transfer cursor differs')
        item=self.manifest['files'][p['index']]
        if not 0<len(data)<=4*1024**2 or active['bytes']+len(data)>item['bytes']:raise ValueError('Transfer size exceeds manifest')
        if hashlib.sha256(data).hexdigest()!=p['sha256']:raise ValueError('Transfer chunk hash differs')
        self.admit(ROOT,item['bytes']-active['bytes']+item['expanded_bytes']+self.reserved_future)
        active['stream'].write(data);active['stream'].flush();os.fsync(active['stream'].fileno())
        active['hash'].update(data);active['bytes']+=len(data)
        return {'index':p['index'],'offset':active['bytes']}
    def commit(self,index):
        active=self.staged
        if active is None or index!=active['index']:raise ValueError('Unknown staged artifact')
        item=self.manifest['files'][index];active['stream'].close()
        if active['bytes']!=item['bytes'] or active['hash'].hexdigest()!=item['sha256'] or digest(active['path'])!=item['sha256']:
            raise ValueError('Full staged artifact differs from official SHA')
        active['path'].replace(write_path(self.job/'stage'/item['file']));self.staged=None
        atomic(self.job/'requests'/f'{index:03d}.json',{'index':index,'manifest_sha256':self.ticket['manifest_sha256'],'sha256':item['sha256']})
        return {'committed':True,'index':index,'sha256':item['sha256']}
    def ack(self,p):
        index=p['index'];item=self.manifest['files'][index]
        receipt=self.job/'receipts'/f'{index:03d}.json';record=json.loads(receipt.read_text())
        if p['receipt_sha256']!=digest(receipt) or p['source_sha256']!=item['sha256'] or p['log_sha256']!=record['log_sha256']:
            raise ValueError('Archive ACK identity differs')
        if not re.fullmatch('[0-9a-f]{64}',p.get('local_commit_sha256','')):raise ValueError('Missing durable local commit')
        atomic(self.job/'acks'/f'{index:03d}.json',{k:p[k] for k in ['index','receipt_sha256','source_sha256','log_sha256','local_commit_sha256']})
        return {'ack_durable':True,'index':index}
    def archive_final(self):
        paths=[self.job/name for name in ['logs/smoke.log','stdout.log','stderr.log','hard-protection.json','resources.json','prior-environment-owner.json']]
        progress=self.job/'progress.json'
        if progress.exists():
            index=json.loads(progress.read_text()).get('index')
            if type(index) is int:
                paths.extend(self.job/'logs'/f'{index:03d}{suffix}' for suffix in ['.log','-version.log'])
        result={}
        for path in paths:
            if path.exists():
                if path.stat().st_size>512*1024:raise ValueError('Final log frame limit')
                data=path.read_bytes();result[path.name]={'base64':base64.b64encode(data).decode(),'sha256':hashlib.sha256(data).hexdigest()}
        return result
    def close(self):
        if self.staged is not None:self.staged['stream'].close()
        if self.started and self.handle:
            props=self.systemd.inspect_service(self.ticket['unit'])
            if not self.systemd.terminal(props):self.systemd.stop_verified(self.handle)


def main():
    os.umask(0o077)
    agent=Agent()
    def stop(*_):raise TimeoutError('Finite SSH control deadline or session stop')
    signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGHUP,stop)
    try:
        while True:
            line=sys.stdin.buffer.readline(1024**2+1)
            if not line:break
            if len(line)>1024**2:raise ValueError('Oversized control frame')
            p=json.loads(line);sequence=p['sequence'];operation=p['operation']
            data=None
            if p.get('binary_bytes') is not None:
                if operation!='chunk' or not 0<p['binary_bytes']<=4*1024**2:raise ValueError('Invalid binary frame')
                data=exact(sys.stdin.buffer,p['binary_bytes'])
            try:
                if operation=='clock':result=agent.check_clock(p)
                elif operation=='initialize':result=agent.initialize(p)
                elif operation=='start':result=agent.start()
                elif operation=='status':result=agent.status(p.get('index'))
                elif operation=='begin':result=agent.begin(p['index'])
                elif operation=='chunk':result=agent.chunk(p,data)
                elif operation=='commit':result=agent.commit(p['index'])
                elif operation=='ack':result=agent.ack(p)
                elif operation=='archive_final':result=agent.archive_final()
                elif operation=='close':result={'closing':True}
                else:raise ValueError('Unreviewed control operation')
                print(json.dumps({'sequence':sequence,'ok':True,'result':result}),flush=True)
                if operation=='close':break
            except Exception as exc:
                print(json.dumps({'sequence':sequence,'ok':False,'error':type(exc).__name__+': '+str(exc)}),flush=True)
                raise
    finally:agent.close()


if __name__=='__main__':main()
