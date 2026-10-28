"""One small owned CPU service proves real notify extension and hard expiry."""
import base64,json,os,subprocess,time
from pathlib import Path
from .io import read_json,atomic_json,sha256,utcnow
from .active_clock import AUTH
from .server_boundary_audit import ssh_command

WORKER=r'''
import argparse,os,time
from pathlib import Path
from ptcg_lab.io import read_json,atomic_json
from ptcg_lab.active_clock import ServiceClock,WaitRecorder,process_ticks
p=argparse.ArgumentParser();p.add_argument('--ticket');a=p.parse_args();t=read_json(a.ticket);j=Path(t['job_directory'])
assert os.environ['CUDA_VISIBLE_DEVICES']==''
c=ServiceClock(j,t['run_id'],t['lease_epoch'],t['active_clock_scope'],reserve=1)
c.register(os.getpid(),process_ticks(os.getpid()));c.start()
time.sleep(2)
w=WaitRecorder(j,t['run_id'],t['lease_epoch']);began=last=time.monotonic()
while time.monotonic()-began<8:
 time.sleep(.1);now=time.monotonic();w.add(now-last);last=now
w.add(0,force=True)
atomic_json(j/'wait-completed.json',{'wait_seconds':w.total,'no_more_wait_credit_after_this_point':True})
# Deliberately remain alive: systemd must expire after the finite earned wait.
time.sleep(60)
raise RuntimeError('Finite active timeout was not enforced')
'''

REMOTE=r'''
import os,pathlib,json,sys,time,hashlib,base64,subprocess
root=pathlib.Path('SERVER_PROJECT_ROOT');lease=P['lease']
for p in [root,root/'jobs']:
 if p.is_symlink() or p.resolve()!=p or p.stat().st_uid!=os.getuid():raise ValueError('Root identity differs')
if json.loads((root/'.ptcg-owner.json').read_text())!={'project_id':P['project_id'],'uid':os.getuid(),'root':str(root)}:raise ValueError('Owner marker differs')
job=root/'jobs'/lease['run_id'];job.mkdir(mode=0o700);control=job/'control';control.mkdir(mode=0o700)
for name,item in P['files'].items():
 path=control/name
 if path.is_absolute() and (not path.is_relative_to(control) or '..' in pathlib.Path(name).parts):raise ValueError('Control path escaped')
 raw=base64.b64decode(item['base64'],validate=True)
 if len(raw)>256*1024 or hashlib.sha256(raw).hexdigest()!=item['sha256']:raise ValueError('Control source changed')
 path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
(control/'training_check.py').write_text(P['worker'])
sys.path.insert(0,str(control))
from ptcg_lab import systemd_job
from ptcg_lab.active_clock import SPEC,AUTH
from ptcg_lab.io import atomic_json,read_json
from ptcg_lab.server_workspace import owned_root,admit
from ptcg_lab.storage_profile import load
owned_root(root,P['project_id']);capacity=admit(root,16*2**20);profile=load(root)
started=time.time();scope={'spec':SPEC,'execution_limit_seconds':12,'job_started_unix':started,
 'hard_deadline_unix':lease['deadline_unix'],'hard_window_seconds':lease['deadline_unix']-started,'startup_wait_seconds':0}
t={'kind':'training_check','project_root':str(root),'project_id':P['project_id'],'job_directory':str(job),
 'control_root':str(control),'ticket_path':str(job/'ticket.json'),'unit':lease['unit'],'run_id':lease['run_id'],
 'lease_epoch':lease['lease_epoch'],'deadline_unix':lease['deadline_unix'],'memory_max_bytes':256*2**20,
 'workspace_hard_max_bytes':profile['hard_max_bytes'],'python_kind':'venv','cpu_quota_cores':1,'mode':'active_clock_acceptance',
 'storage_profile':profile,'active_clock_scope':scope,'active_clock_acceptance_authorization':AUTH,'stop_grace_seconds':1}
atomic_json(job/'ticket.json',t);handle=None;error=None
try:
 handle,initial=systemd_job.start_verified(t);atomic_json(job/'identity.json',handle)
 print(json.dumps({'type':'started','identity':handle,'ticket':t}),flush=True)
 while True:
  props=systemd_job.inspect_service(lease['unit'])
  if systemd_job.terminal(props):break
  if time.time()>lease['deadline_unix']-10:raise TimeoutError('Engineering hard safety deadline')
  time.sleep(.5)
except Exception as exc:error=type(exc).__name__+': '+str(exc)
finally:
 if handle:
  props=systemd_job.inspect_service(lease['unit'])
  if not systemd_job.terminal(props):props=systemd_job.stop_verified(handle)
  boot=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
  if systemd_job.identity(props,boot)!=handle:raise ValueError('Terminal invocation changed')
  files={}
  for path in [job/'active-clock-server.json',job/'wait-completed.json',*sorted((job/'resource-wait').glob('*.json'))]:
   if path.exists():
    raw=path.read_bytes()
    if len(raw)>16384:raise ValueError('Oversized proof')
    files[path.relative_to(job).as_posix()]={'sha256':hashlib.sha256(raw).hexdigest(),'base64':base64.b64encode(raw).decode()}
  tail=(job/'stderr.log').read_text(errors='replace')[-4096:] if (job/'stderr.log').exists() else ''
  print(json.dumps({'type':'terminal','identity':handle,'service_properties':props,'host_boot_id':boot,
   'server_unix':time.time(),'files':files,'error':error,'stderr_tail':tail,'initial_properties':initial}),flush=True)
if error:raise RuntimeError(error)
'''

def run(config,output):
    output=Path(output);root=Path(config['project_path']);source=Path(__file__).resolve().parents[1]
    if os.environ.get('PTCG_JOB_RUN_ID')!=output.name:raise ValueError('Controlled runner required')
    if config.get('wait_authorization')!=AUTH or sha256(root/AUTH['path'])!=AUTH['sha256']:
        raise ValueError('Explicit wait authority required')
    if config.get('source_GPU_acceptance') is None:raise ValueError('Existing actual GPU proof required')
    proof=root/config['source_GPU_acceptance']['path']
    if sha256(proof)!=config['source_GPU_acceptance']['sha256'] or not read_json(proof)['complete']:
        raise ValueError('Current endpoint GPU proof changed')
    epoch=read_json(output/'experiment.json')['remote_lease_epoch']
    lease={'run_id':output.name,'attempt':1,'lease_epoch':epoch,'host':config['host'],
        'unit':'ptcg-hybrid-clock-'+output.name.rsplit('_',1)[1]+'-'+str(epoch)+'.service',
        'deadline_unix':time.time()+110,'identity':None}
    from .server_trial import remote_source_files
    files=remote_source_files(source)
    payload={n:{'base64':base64.b64encode(raw).decode(),'sha256':__import__('hashlib').sha256(raw).hexdigest()} for n,raw in files.items()}
    params={'project_id':config['project_id'],'lease':lease,'files':payload,'worker':WORKER}
    request='P='+repr(params)+'\n'+REMOTE
    if len(request.encode())>2*2**20:raise ValueError('Bounded engineering input exceeded')
    atomic_json(output/'remote-job.json',lease)
    atomic_json(output/'active-clock-tested-sources.json',{n:v['sha256'] for n,v in payload.items()})
    with (output/'ssh-stderr.log').open('w',encoding='utf-8') as errors:
        proc=subprocess.Popen(ssh_command(config),stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=errors,
                              text=True,encoding='utf-8')
        proc.stdin.write(request);proc.stdin.close();terminal=None
        for line in proc.stdout:
            if len(line)>512*1024:raise ValueError('Engineering response exceeded envelope')
            row=json.loads(line)
            if row['type']=='started':
                lease['identity']=row['identity'];atomic_json(output/'remote-job.json',lease)
                atomic_json(output/'remote-ticket.json',row['ticket'])
            elif row['type']=='terminal':terminal=row
            else:raise ValueError('Unexpected engineering response')
        status=proc.wait(timeout=10)
    if terminal is None:raise RuntimeError('Missing terminal proof; reconcile before any new job')
    receipt={**lease,**{k:terminal[k] for k in ('service_properties','host_boot_id','server_unix')},'at':utcnow()}
    atomic_json(output/'remote-terminal.json',receipt)
    lease.update(terminal_receipt_path=(output/'remote-terminal.json').relative_to(root).as_posix(),
                 terminal_receipt_sha256=sha256(output/'remote-terminal.json'))
    atomic_json(output/'remote-job.json',lease)
    for name,item in terminal['files'].items():
        raw=base64.b64decode(item['base64'],validate=True)
        if __import__('hashlib').sha256(raw).hexdigest()!=item['sha256']:raise ValueError('Proof bytes changed')
        path=output/'remote-results'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    atomic_json(output/'clock-test-observation.json',terminal)
    if status or terminal['error']:raise RuntimeError('Clock acceptance failed: '+str(terminal['error']))
    clock=read_json(output/'remote-results/active-clock-server.json')
    completed=read_json(output/'remote-results/wait-completed.json')
    props=terminal['service_properties'];elapsed=(int(props['ExecMainExitTimestampMonotonic'])-int(props['ExecMainStartTimestampMonotonic']))/1e6
    initial_runtime=terminal['initial_properties']['RuntimeMaxUSec']
    if (props['Result']!='timeout' or not 7.9<=completed['wait_seconds']<=10
            or not elapsed>12 or not elapsed<25 or clock['counted_execution_seconds']>12.5
            or clock['verified_wait_seconds']<7.9):
        raise ValueError('Finite runtime extension/expiry was not demonstrated')
    atomic_json(output/'metrics.json',{'at':utcnow(),'complete':True,'engineering_only':True,
        'actual_notify_extension_and_expiry_passed':True,'actual_service_elapsed_seconds':elapsed,
        'initial_RuntimeMaxUSec':initial_runtime,'active_execution_seconds_observed':clock['counted_execution_seconds'],
        'verified_test_wait_seconds':clock['verified_wait_seconds'],'expected_terminal_result':'timeout',
        'no_GPU_or_model_execution':True,'new_scientific_samples':0,'optimizer_steps_executed':0,
        'sources':{'path':(output/'active-clock-tested-sources.json').relative_to(root).as_posix(),
                   'sha256':sha256(output/'active-clock-tested-sources.json')}})
