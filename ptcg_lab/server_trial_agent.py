"""Versioned short training/check endpoint; only owned job files and services."""
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


def validate_blob(item):
    if (item['file']!=item['sha256']+'.blob' or not re.fullmatch('[0-9a-f]{64}',item['sha256'])
            or type(item['bytes']) is not int or not 0<=item['bytes']<128*2**20
            or item['bytes']==0 and item['sha256']!=hashlib.sha256(b'').hexdigest()):
        raise ValueError('Invalid input blob')


def choose_idle_gpu(rows,apps):
    parsed={}
    for line in rows:
        row=[x.strip() for x in line.split(',')]
        if len(row)!=4:raise ValueError('Malformed GPU observation')
        index=int(row[0])
        if index in (4,5,6,7) and int(row[2])<=64 and int(row[3])<=1 and row[1] not in apps:
            if not re.fullmatch(r'GPU-[0-9a-f-]{36}',row[1]):raise ValueError('GPU UUID differs')
            parsed[index]=row
    return next((parsed[i] for i in [6,7,4,5] if i in parsed),None)


def trial_seconds_maximum(mode, authorized_numeric=False, authorized_training=False, authorized_TR12=False):
    if authorized_TR12:
        if mode!='derived_scientific' or authorized_numeric or authorized_training:raise ValueError('TR12 runtime cannot borrow another scope')
        return 28800
    if authorized_training:
        if mode!='derived_scientific' or authorized_numeric:raise ValueError('Extended scientific window cannot change another role')
        return 30000
    if authorized_numeric:
        if mode!='gpu_existing_restore':raise ValueError('Authorized numerical window cannot change another role')
        return 600
    return 7200 if mode in ('fresh_scientific','derived_scientific') else 300 if mode=='gpu_existing_restore' else 1200 if mode=='compact_cpu_existing' else 900 if mode in ('journal_io_check','cpu_sampler_check') else 3600 if mode in ('compact_cpu_restore','scientific_continue') else 1800


def make_agent(space):
    root=space['ROOT'];write_path=space['write_path'];atomic=space['atomic'];digest=space['digest']
    class TrialAgent(space['Agent']):
        def initialize(self,p):
            if self.job is not None:raise RuntimeError('Already prepared')
            if self.clock_checked_at is None or time.time()-self.clock_checked_at>60:raise ValueError('Clock handshake required')
            marker=root/'.ptcg-owner.json'
            if (root.is_symlink() or root.stat().st_uid!=os.getuid() or root.stat().st_mode&0o077
                    or marker.is_symlink() or marker.stat().st_uid!=os.getuid()
                    or json.loads(marker.read_text())!={'project_id':p['project_id'],'uid':os.getuid(),'root':str(root)}):
                raise ValueError('Project ownership differs')
            if not re.fullmatch(r'[A-Za-z0-9_-]{12,100}',p['run_id']):raise ValueError('Invalid run ID')
            if p['mode'] not in ('basic_cpu_check','short_gpu_trial','compact_cpu_restore','compact_cpu_existing','gpu_existing_restore','live_archive_check','scientific_continue','fresh_scientific','derived_scientific','journal_io_check','cpu_sampler_check'):raise ValueError('Unknown trial role')
            authorized_numeric=p.get('prepared_numeric_window_seconds')==600
            authorized_training=p.get('TR11_authorized_training_seconds')==30000
            authorized_TR12=p.get('TR12_authorized_training_seconds')==28800
            TR12_acceptance=False
            if authorized_TR12:
                preview_raw=base64.b64decode(p['manifest_base64'],validate=True)
                if hashlib.sha256(preview_raw).hexdigest()!=p['manifest_sha256']:raise ValueError('TR12 runtime manifest changed')
                preview=json.loads(preview_raw);scope=preview.get('TR12_runtime_scope',{})
                TR12_acceptance=preview.get('parent_acceptance_only') is True
                if (p['mode']!='derived_scientific' or preview.get('new_diagnostic_identity')!='TR12'
                        or preview.get('engineering_only') is not TR12_acceptance or scope.get('seconds_max')!=28800
                        or scope.get('authorization_sha256')!='971709f6510a8c83527dd42175a9459e12861c00544f489067765dc6d2412ff4'
                        or scope.get('CPU_quota_cores')!=3 or preview['server_resource_profile']['cpu_quota_cores']!=3
                        or preview['training_config']['target_decisions']!=369941):
                    raise ValueError('Only exact TR12 full-state segment scope is admitted')
            if authorized_training:
                raw_preview=base64.b64decode(p['manifest_base64'],validate=True)
                if hashlib.sha256(raw_preview).hexdigest()!=p['manifest_sha256']:raise ValueError('TR11 runtime manifest changed')
                preview=json.loads(raw_preview);scope=preview.get('TR11_runtime_scope',{})
                if (p['mode']!='derived_scientific' or preview.get('new_diagnostic_identity')!='TR11'
                        or preview.get('engineering_only') is not False or scope.get('seconds_max')!=30000
                        or scope.get('CPU_quota_cores')!=3 or preview['server_resource_profile']['cpu_quota_cores']!=3
                        or preview['training_config']['target_decisions']!=49166
                        or not re.fullmatch('[0-9a-f]{64}',scope.get('authorization_sha256',''))):
                    raise ValueError('Only exact forward TR11 runtime/resource allowance is admitted')
            if authorized_numeric:
                preview_raw=base64.b64decode(p['manifest_base64'],validate=True)
                if hashlib.sha256(preview_raw).hexdigest()!=p['manifest_sha256']:raise ValueError('Authorized window manifest changed')
                preview=json.loads(preview_raw);scope=preview.get('authorized_numeric_window',{})
                if (not preview.get('prepared_parent_verify') or not preview.get('engineering_only')
                        or scope.get('seconds_max')!=600 or scope.get('maximum_new_launches')!=1
                        or not re.fullmatch('[0-9a-f]{64}',scope.get('authorization_sha256',''))):
                    raise ValueError('New numerical window has no exact scoped authorization')
            maximum=trial_seconds_maximum(p['mode'],authorized_numeric,authorized_training,authorized_TR12)
            if TR12_acceptance:maximum=1800
            active_scope=preview.get('active_clock_scope') if authorized_TR12 else None
            if active_scope:
                active_spec={'schema':1,'authorization':{'path':'research/autonomy/authorizations/TR12-shared-wait-exemption-20261028.json',
                    'sha256':'1f4a7e4c79782ec85001a43e268957e9e57969f2be50cbd329d17047dd8627a4'},
                    'hard_safety_window_seconds':86400,'basis':'gross_wall_minus_verified_shared_resource_wait'}
                if (active_scope.get('spec')!=active_spec or type(active_scope.get('execution_limit_seconds')) not in (int,float)
                        or not 0<active_scope['execution_limit_seconds']<=maximum
                        or active_scope.get('hard_deadline_unix')!=p['deadline_unix']
                        or not active_scope['execution_limit_seconds']<=active_scope.get('hard_window_seconds',0)<=86400
                        or abs(p['deadline_unix']-active_scope['job_started_unix']-active_scope['hard_window_seconds'])>.01
                        or active_scope.get('startup_wait_seconds')!=0):
                    raise ValueError('Active-clock scope changed')
                maximum=active_scope['hard_window_seconds']
            if not 100<p['deadline_unix']-time.time()<=maximum:raise ValueError('Trial deadline outside scope')
            if globals().get('PTCG_AGENT_SHA256')!=p['agent_sha256']:raise ValueError('Endpoint source differs')
            self.job=write_path(root/'jobs'/p['run_id']);self.job.mkdir(mode=0o700)
            for name in ['control','control/ptcg_lab','control/ptcg_lab/public_cg','stage','requests','acks','evictions','receipts','logs']:
                write_path(self.job/name).mkdir(mode=0o700,exist_ok=True)
            hashes={}
            for name,item in p['files'].items():
                if not (name in ('training_check.py','training_worker.py') or name.startswith('ptcg_lab/') and name.endswith('.py')):
                    raise ValueError('Unreviewed control source')
                data=base64.b64decode(item['base64'],validate=True)
                if hashlib.sha256(data).hexdigest()!=item['sha256']:raise ValueError('Control source hash differs')
                target=write_path(self.job/'control'/name);target.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
                with target.open('xb') as output:output.write(data);output.flush();os.fsync(output.fileno())
                hashes[name]=item['sha256']
            raw=base64.b64decode(p['manifest_base64'],validate=True)
            if hashlib.sha256(raw).hexdigest()!=p['manifest_sha256']:raise ValueError('Trial manifest changed')
            self.manifest=json.loads(raw)
            if self.manifest['mode']!=p['mode']:raise ValueError('Manifest role differs')
            for item in self.manifest['files']:
                validate_blob(item)
            atomic(self.job/'training-manifest.json',self.manifest)
            # Atomic uses a pretty representation; preserve original manifest bytes too.
            with write_path(self.job/'training-manifest.raw.json').open('xb') as output:
                output.write(raw);output.flush();os.fsync(output.fileno())
            atomic(self.job/'source-hashes.json',hashes)
            sys.path.insert(0,str(self.job/'control'))
            from ptcg_lab import systemd_job
            from ptcg_lab.server_workspace import admit
            self.systemd=systemd_job;self.admit=admit;self.reserved_future=512*2**20
            if p['mode']=='gpu_existing_restore':self.reserved_future=256*2**20
            if p['mode']=='live_archive_check':self.reserved_future=192*2**20
            if p['mode']=='live_archive_check' and self.manifest.get('ack_lag_barrier_check'):
                if (self.manifest.get('reserved_future_bytes'),self.manifest.get('post_retirement_forecast_bytes'))!=(16*2**20,384*2**20) or not self.manifest.get('no_training_updates'):
                    raise ValueError('ACK barrier staged capacity scope differs')
                if p.get('gpu_index') is not None or p.get('gpu_uuid') is not None:raise ValueError('ACK barrier acceptance must hide GPU')
                self.reserved_future=16*2**20
            if p['mode']=='journal_io_check':
                if self.manifest.get('reserved_future_bytes')!=256*2**20 or not self.manifest.get('no_training_updates'):raise ValueError('CPU journal/IO check scope differs')
                self.reserved_future=256*2**20
            if p['mode']=='cpu_sampler_check':
                if self.manifest.get('reserved_future_bytes')!=16*2**20 or self.manifest.get('post_retirement_forecast_bytes')!=256*2**20 or not self.manifest.get('no_training_updates') or p.get('gpu_index') is not None or p.get('gpu_uuid') is not None:
                    raise ValueError('CPU sampler acceptance resource role differs')
                self.reserved_future=16*2**20
            if p['mode']=='scientific_continue':
                if self.manifest.get('reserved_future_bytes')!=16*2**20 or not 384*2**20<=self.manifest.get('post_cache_retirement_forecast_bytes',0)<=768*2**20:raise ValueError('Scientific phased capacity forecast outside reviewed scope')
                self.reserved_future=self.manifest['reserved_future_bytes']
            if p['mode']=='fresh_scientific':
                if self.manifest.get('reserved_future_bytes')!=16*2**20 or not 768*2**20<=self.manifest.get('post_retirement_forecast_bytes',0)<=1024*2**20:
                    raise ValueError('Fresh scientific phased capacity forecast differs')
                self.reserved_future=16*2**20
            if p['mode']=='derived_scientific':
                if not 512*2**20<=self.manifest.get('reserved_future_bytes',0)<=1024*2**20:raise ValueError('Derived training capacity forecast differs')
                self.reserved_future=self.manifest['reserved_future_bytes']
            if p['mode']=='compact_cpu_restore':
                capsule=self.manifest['restore_capsule']
                # The endpoint uses system Python; this validation does not import Torch.
                expanded=sum(x['raw_bytes'] for x in capsule['components'])+sum(x['raw_bytes'] for x in capsule['history']['members'])
                if expanded!=capsule['expanded_bytes'] or len(capsule['history']['members'])!=31 or capsule['expected_update_input']['decisions']!=5430:
                    raise ValueError('Full expanded original TR04 scope differs')
                expected=expanded+384*2**20
                if self.manifest.get('reserved_future_bytes')!=expected:raise ValueError('Expanded restore capacity forecast differs')
                self.reserved_future=expected
            self.admit(root,sum(x['bytes'] for x in self.manifest['files'])+self.reserved_future)
            self.ticket={'kind':'train_ppo' if p['mode'] in ('short_gpu_trial','gpu_existing_restore','scientific_continue','fresh_scientific','derived_scientific') else 'training_check',
                'project_root':str(root),'project_id':p['project_id'],'job_directory':str(self.job),
                'control_root':str(self.job/'control'),'ticket_path':str(self.job/'ticket.json'),
                'unit':p['unit'],'run_id':p['run_id'],'lease_epoch':p['lease_epoch'],
                'deadline_unix':p['deadline_unix'],'memory_max_bytes':4*2**30,'workspace_hard_max_bytes':10*2**30,
                'python_kind':'venv','cpu_quota_cores':2,'mode':p['mode'],'manifest_sha256':p['manifest_sha256'],
                'gpu_index':p.get('gpu_index'),'gpu_uuid':p.get('gpu_uuid')}
            if str(root)=='SERVER_PROJECT_ROOT':
                from ptcg_lab.storage_profile import load
                profile=load(root)
                if profile is None:raise ValueError('Authorized HDD2 storage profile required')
                self.ticket.update(storage_profile=profile,workspace_hard_max_bytes=profile['hard_max_bytes'])
            if self.manifest.get('server_resource_profile') is not None:
                from ptcg_lab.resource_profile import limits,verify_actual_machine
                profile=self.manifest['server_resource_profile'];cpu,memory=limits(profile)
                actual=verify_actual_machine(profile)
                self.ticket.update(cpu_quota_cores=cpu,memory_max_bytes=memory,server_resource_profile=profile)
                atomic(self.job/'actual-resource-share.json',actual)
            if authorized_training:
                self.ticket.update(TR11_runtime_scope=self.manifest['TR11_runtime_scope'],new_diagnostic_identity='TR11')
            if authorized_TR12:
                self.ticket.update(TR12_runtime_scope=self.manifest['TR12_runtime_scope'],new_diagnostic_identity='TR12')
            self.clock_wait_started=None;self.clock_wait_seconds=0.
            if active_scope:self.ticket['active_clock_scope']=dict(active_scope)
            atomic(self.job/'ticket.json',self.ticket)
            signal.alarm(max(1,int(p['deadline_unix']-time.time())))
            return {'prepared':True,'root':str(root),'capacity':self.admit(root,0),
                    'host_boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
        def begin(self,index):
            if self.started or self.staged is not None:raise RuntimeError('Inputs must precede service start')
            item=self.manifest['files'][index];self.admit(root,item['bytes']+self.reserved_future)
            path=write_path(self.job/'stage'/(item['file']+'.partial'));stream=path.open('xb')
            self.staged={'index':index,'path':path,'stream':stream,'bytes':0,'hash':hashlib.sha256()}
            return {'index':index,'offset':0}
        def start(self):
            if self.staged is not None:raise RuntimeError('Uncommitted input remains')
            for item in self.manifest['files']:
                path=write_path(self.job/'stage'/item['file'])
                if not path.is_file() or path.stat().st_size!=item['bytes']:raise ValueError('Input missing')
            if self.ticket['mode'] in ('short_gpu_trial','gpu_existing_restore','scientific_continue','fresh_scientific','derived_scientific'):
                rows=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=10).stdout.splitlines()
                apps=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],capture_output=True,text=True,check=True,timeout=10).stdout
                if self.ticket['mode'] in ('fresh_scientific','derived_scientific'):
                    row=choose_idle_gpu(rows,apps)
                    if row is None:
                        if self.ticket.get('active_clock_scope') and self.clock_wait_started is None:self.clock_wait_started=time.monotonic()
                        elapsed=self.clock_wait_seconds+(time.monotonic()-self.clock_wait_started if self.clock_wait_started is not None else 0)
                        return {'waiting_for_authorized_idle_gpu':True,'no_service_started':True,'startup_wait_seconds':elapsed}
                    if self.ticket.get('active_clock_scope'):
                        if self.clock_wait_started is not None:
                            self.clock_wait_seconds+=time.monotonic()-self.clock_wait_started;self.clock_wait_started=None
                        self.ticket['active_clock_scope']['startup_wait_seconds']=self.clock_wait_seconds
                    self.ticket.update(gpu_index=int(row[0]),gpu_uuid=row[1])
                    atomic(self.job/'ticket.json',self.ticket)
                selected=[line.split(',') for line in rows if int(line.split(',')[0])==self.ticket['gpu_index']]
                if self.ticket['gpu_index'] not in (4,5,6,7) or len(selected)!=1:raise ValueError('Unauthorized GPU')
                row=[x.strip() for x in selected[0]]
                if row[1]!=self.ticket['gpu_uuid'] or int(row[2])>64 or int(row[3])>1 or row[1] in apps:
                    raise RuntimeError('Authorized GPU ceased to be idle; no launch')
                atomic(self.job/'gpu-start-probe.json',{'selected_row':row,'no_existing_compute_users':True,'at_unix':time.time(),
                    'selection_once_before_service_start':True,'authorized_indices':[4,5,6,7]})
            result=super().start()
            if self.ticket.get('active_clock_scope'):result['startup_wait_seconds']=self.clock_wait_seconds
            return result
        def status(self,index=None):
            result=super().status()
            for name in ['trial-metrics.json','audit.json','gpu-start-probe.json','active-clock-server.json']:
                path=self.job/name
                if path.exists():result[name]=json.loads(path.read_text())
            return result
        def read_file(self,p):
            relative=Path(p['path'])
            if relative.is_absolute() or '..' in relative.parts:raise ValueError('Artifact read escaped this job')
            path=write_path(self.job/relative)
            if p.get('optional') is True and not path.exists():return {'exists':False}
            size=path.stat().st_size
            if not 0<=p['offset']<=size or not 0<p['count']<=256*1024:raise ValueError('Artifact read envelope')
            with path.open('rb') as stream:stream.seek(p['offset']);data=stream.read(min(p['count'],size-p['offset']))
            return {'path':p['path'],'offset':p['offset'],'size':size,'base64':base64.b64encode(data).decode(),
                    'sha256':hashlib.sha256(data).hexdigest()}
        def training_archive(self,operation,p):
            if self.manifest.get('online_archive_enabled') is not True:raise ValueError('Live archive was not explicitly admitted for this job')
            from ptcg_lab.archive_outbox import ArchiveOutbox
            outbox=ArchiveOutbox(self.job,self.ticket['run_id'],self.ticket['lease_epoch'])
            if operation=='archive_pending':return outbox.pending(p.get('limit',16))
            if operation=='archive_chunk':return outbox.chunk(p['object_sequence'],p['offset'],p['count'])
            if operation=='archive_ack':return outbox.accept_ack(p['ack'])
            raise ValueError('Unknown training archive operation')
    return TrialAgent()


def main():
    os.umask(0o077);agent=None;space=None
    try:
        while True:
            line=sys.stdin.buffer.readline(1024**2+1)
            if not line:break
            if len(line)>1024**2:raise ValueError('Oversized control frame')
            p=json.loads(line);op=p['operation'];data=None
            if p.get('binary_bytes') is not None:
                if op!='chunk' or not 0<p['binary_bytes']<=4*2**20:raise ValueError('Binary frame envelope')
                data=space['exact'](sys.stdin.buffer,p['binary_bytes'])
            try:
                if op=='load_base':
                    if agent is not None:raise RuntimeError('Base already loaded')
                    code=base64.b64decode(p['base_source'],validate=True)
                    if hashlib.sha256(code).hexdigest()!=p['base_sha256']:raise ValueError('Endpoint library changed')
                    space={'__name__':'ptcg_trial_endpoint_library'};exec(compile(code,'reviewed_endpoint_library','exec'),space)
                    if p.get('project_root'):
                        target=Path(p['project_root'])
                        if str(target)!='SERVER_PROJECT_ROOT' or target.is_symlink() or target.resolve()!=target:
                            raise ValueError('Unapproved relocated project root')
                        marker=target/'.ptcg-storage.json'
                        if marker.is_symlink() or marker.stat().st_uid!=os.getuid() or json.loads(marker.read_text())!=p['storage_profile']:
                            raise ValueError('Relocated storage authority differs')
                        space['ROOT']=target
                    agent=make_agent(space);result={'loaded':True,'base_sha256':p['base_sha256']}
                elif op=='clock':result=agent.check_clock(p)
                elif op=='initialize':result=agent.initialize(p)
                elif op=='begin':result=agent.begin(p['index'])
                elif op=='chunk':result=agent.chunk(p,data)
                elif op=='commit':result=agent.commit(p['index'])
                elif op=='start':result=agent.start()
                elif op=='status':result=agent.status()
                elif op=='read_file':result=agent.read_file(p)
                elif op in ('archive_pending','archive_chunk','archive_ack'):result=agent.training_archive(op,p)
                elif op=='archive_final':result=agent.archive_final()
                elif op=='close':result={'closing':True}
                else:raise ValueError('Unreviewed operation')
                print(json.dumps({'sequence':p['sequence'],'ok':True,'result':result}),flush=True)
                if op=='close':break
            except Exception as exc:
                print(json.dumps({'sequence':p['sequence'],'ok':False,'error':type(exc).__name__+': '+str(exc)}),flush=True)
                if isinstance(exc,ValueError) and op in ('archive_pending','archive_chunk','archive_ack'):continue
                raise
    finally:
        if agent is not None:agent.close()


if __name__=='__main__':main()
