"""One finite training-supervisor audit on the user's server; no ML training."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

from .io import atomic_json,read_json,sha256,utcnow
from .remote_lease import validate_no_unreconciled_remote


WORKER=r'''
import argparse,json,os,pathlib,signal,subprocess,sys,time
p=argparse.ArgumentParser();p.add_argument('--ticket',required=True);a=p.parse_args()
t=json.loads(pathlib.Path(a.ticket).read_text());job=pathlib.Path(t['job_directory'])
if t['kind']!='supervisor_audit' or t['run_id']!=os.environ['PTCG_JOB_RUN_ID']:raise RuntimeError('Audit ticket differs')
signal.signal(signal.SIGTERM,signal.SIG_IGN)
child=subprocess.Popen([sys.executable,'-B','-c','import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(180)'])
def ticks(pid):return pathlib.Path('/proc').joinpath(str(pid),'stat').read_text().rsplit(')',1)[1].split()[19]
record={'run_id':t['run_id'],'lease_epoch':t['lease_epoch'],'parent_pid':os.getpid(),'parent_start_ticks':ticks(os.getpid()),'child_pid':child.pid,'child_start_ticks':ticks(child.pid),'status':'synthetic_recovery_marker_not_PPO_checkpoint'}
tmp=job/'checkpoint.tmp'
with tmp.open('w') as f:json.dump(record,f);f.flush();os.fsync(f.fileno())
tmp.replace(job/'checkpoint.json')
while True:time.sleep(.2)
'''


REMOTE=r'''
import base64,fcntl,hashlib,json,os,pathlib,shutil,subprocess,sys,time
P=json.loads(PARAMETERS)
root=pathlib.Path.home()/'ptcg-hybrid-v3'
if root.is_symlink():raise RuntimeError('Symlink project root')
root=root.resolve();expected=(pathlib.Path.home().resolve()/'ptcg-hybrid-v3')
if root!=expected:raise RuntimeError('Project outside own home')
marker=root/'.ptcg-owner.json';owner={'project_id':P['project_id'],'uid':os.getuid(),'root':str(root)}
def write(path,value):
 tmp=path.with_name(path.name+'.tmp')
 with tmp.open('w') as f:json.dump(value,f);f.flush();os.fsync(f.fileno())
 tmp.replace(path)
if P['action']=='start':
 if not root.exists():
  if shutil.disk_usage(root.parent).free<20*2**30+1024**2:raise RuntimeError('Free reserve')
  root.mkdir(mode=0o700);write(marker,owner)
 if not marker.is_file() or marker.is_symlink() or json.loads(marker.read_text())!=owner:raise RuntimeError('Unowned existing project')
 for name in ['control','jobs','tmp']:
  d=root/name
  if d.is_symlink():raise RuntimeError('Symlink control directory')
  d.mkdir(exist_ok=True)
 package=root/'control/ptcg_lab';package.mkdir(exist_ok=True)
 if package.is_symlink():raise RuntimeError('Symlink package directory')
 for name,item in P['files'].items():
  target=root/'control'/name
  if target.is_symlink() or '..' in pathlib.Path(name).parts:raise RuntimeError('Unsafe source path')
  data=base64.b64decode(item['base64'])
  if hashlib.sha256(data).hexdigest()!=item['sha256']:raise RuntimeError('Control source hash differs')
  if target.exists():
   if target.read_bytes()!=data:raise RuntimeError('Existing control source differs; migration required')
  else:
   with target.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
if not marker.is_file() or json.loads(marker.read_text())!=owner:raise RuntimeError('Workspace owner differs')
sys.path.insert(0,str(root/'control'))
from ptcg_lab.systemd_job import start_verified,inspect_service,identity,stop_verified,terminal
from ptcg_lab.server_workspace import admit
job=root/'jobs'/P['run_id'];unit=P['unit'];boot=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
if P['action']=='start':
 if abs(time.time()-P['client_unix'])>5:raise RuntimeError('Server clock differs from local deadline')
 admit(root,1024**2)
 probe=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],
                      text=True,capture_output=True,timeout=5)
 write(root/'jobs/gpu-probe.json',{'at_unix':time.time(),'returncode':probe.returncode,
       'preferred_gpu_rows':[line for line in probe.stdout.splitlines() if line.split(',')[0].strip() in ('6','7')],
       'audit_uses_gpu':False})
 with (root/'control/fence.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  fence=root/'control/lease.json'
  if fence.exists():
   old=json.loads(fence.read_text())
   if P['lease_epoch']<=old['lease_epoch']:raise RuntimeError('Stale lease epoch')
   if not terminal(inspect_service(old['unit'])):raise RuntimeError('Previous remote service unresolved')
  job.mkdir()
  ticket={'kind':'supervisor_audit','unit':unit,'project_root':str(root),'job_directory':str(job),'ticket_path':str(job/'ticket.json'),
          'run_id':P['run_id'],'lease_epoch':P['lease_epoch'],'deadline_unix':P['deadline_unix'],
          'memory_max_bytes':64*1024**2,'workspace_hard_max_bytes':10*2**30,'stop_grace_seconds':2}
  write(job/'ticket.json',ticket);write(fence,{'unit':unit,'lease_epoch':P['lease_epoch'],'run_id':P['run_id']})
  handle,props=start_verified(ticket);write(job/'identity.json',handle)
  negative=dict(handle,invocation_id='0'*32)
  try:stop_verified(negative)
  except RuntimeError as exc:
   negative_rejected='identity changed' in str(exc)
   if not negative_rejected:raise
  else:raise RuntimeError('Wrong invocation accepted')
  write(job/'negative-identity.json',{'rejected':negative_rejected,'same_unit_still_present':inspect_service(unit)['InvocationID']==handle['invocation_id']})
ticket=json.loads((job/'ticket.json').read_text())
if any(ticket[k]!=P[k] for k in ['run_id','unit','lease_epoch','deadline_unix']):raise RuntimeError('Persisted ticket differs')
props=inspect_service(unit)
handle=json.loads((job/'identity.json').read_text()) if (job/'identity.json').exists() else identity(props,boot)
if handle['host_boot_id']!=boot:raise RuntimeError('Host reboot; reconcile before continuing')
if props.get('LoadState')=='loaded' and identity(props,boot)!=handle:raise RuntimeError('Invocation changed')
cp=json.loads((job/'checkpoint.json').read_text()) if (job/'checkpoint.json').exists() else None
def same_process(pid,ticks):
 path=pathlib.Path('/proc')/str(pid)/'stat'
 try:fields=path.read_text().rsplit(')',1)[1].split();return fields[19]==ticks and fields[0]!='Z'
 except FileNotFoundError:return False
live=None if cp is None else {'parent':same_process(cp['parent_pid'],cp['parent_start_ticks']),'child':same_process(cp['child_pid'],cp['child_start_ticks'])}
negative=json.loads((job/'negative-identity.json').read_text()) if (job/'negative-identity.json').exists() else None
cg=pathlib.Path('/sys/fs/cgroup')/props.get('ControlGroup','').lstrip('/')
kernel={}
if props.get('ControlGroup'):
 for name in ['memory.max','memory.swap.max','pids.max','cgroup.procs']:
  path=cg/name
  if path.is_file():kernel[name]=path.read_text().strip()
try:capacity=admit(root,0)
except OSError as exc:capacity={'error':str(exc)}
print(json.dumps({'identity':handle,'host_boot_id':boot,'server_unix':time.time(),'service_properties':props,'checkpoint':cp,
 'negative_identity':negative,'live_owned_processes':live,'capacity':capacity,'root':str(root),'terminal':terminal(props),
 'kernel_cgroup_files':kernel}))
'''


def duration_seconds(value):
    if value.isdigit():return int(value)/1000000
    factors={'us':.000001,'ms':.001,'s':1,'min':60,'h':3600,'d':86400}
    tokens=value.split()
    if not tokens:raise ValueError('Missing systemd duration')
    total=0.
    for token in tokens:
        match=re.fullmatch(r'(\d+(?:\.\d+)?)(us|ms|s|min|h|d)',token)
        if not match:raise ValueError('Unrecognized systemd duration')
        total+=float(match[1])*factors[match[2]]
    return total


def ssh_command(config):
    if (config['host'],config['port'],config['user'])!=('TRAINING_HOST',15134,'TRAINING_USER'):
        raise ValueError('Unapproved server target')
    key=Path(config['identity_file'])
    if str(key)!=r'SSH_IDENTITY_FILE' or not key.is_file():raise ValueError('Expected SSH identity unavailable')
    return [shutil.which('ssh') or 'ssh','-T','-i',str(key),'-p','15134','-o','IdentitiesOnly=yes','-o','BatchMode=yes',
            '-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=10','TRAINING_USER@TRAINING_HOST','/usr/bin/python3','-B','-']


def run(config,output):
    output=Path(output).resolve();deadline=float(os.environ['PTCG_JOB_DEADLINE_UNIX'])
    if os.environ.get('PTCG_JOB_RUN_ID')!=output.name:raise RuntimeError('Audit requires controlled runner')
    if (output/'remote-job.json').exists():raise RuntimeError('Audit cannot relaunch; reconcile existing service')
    if deadline-time.time()<100:raise RuntimeError('Insufficient audit job time')
    source=Path(__file__).resolve().parent;experiment=read_json(output/'experiment.json')
    epoch=experiment['remote_lease_epoch'];unit='ptcg-hybrid-boundary-'+output.name.rsplit('_',1)[1]+'-'+str(epoch)+'.service'
    parameters={'project_id':config['project_id'],'run_id':output.name,'unit':unit,'lease_epoch':epoch,
                'deadline_unix':time.time()+65,'client_unix':time.time()}
    files={'boundary_worker.py':WORKER.encode(),'ptcg_lab/__init__.py':b'',
           'ptcg_lab/systemd_job.py':(source/'systemd_job.py').read_bytes(),
           'ptcg_lab/server_workspace.py':(source/'server_workspace.py').read_bytes()}
    payload={name:{'base64':base64.b64encode(data).decode(),'sha256':hashlib.sha256(data).hexdigest()} for name,data in files.items()}
    atomic_json(output/'remote-source.json',{name:item['sha256'] for name,item in payload.items()})
    lease={'run_id':output.name,'attempt':1,'lease_epoch':epoch,'host':config['host'],'unit':unit,
           'deadline_unix':parameters['deadline_unix'],'identity':None}
    atomic_json(output/'remote-job.json',lease)
    def rpc(action):
        p=dict(parameters,action=action)
        if action=='start':p['files']=payload
        script=REMOTE.replace('PARAMETERS',repr(json.dumps(p)))
        result=subprocess.run(ssh_command(config),input=script,text=True,encoding='utf-8',capture_output=True,
                              timeout=min(25,max(1,deadline-time.time()-10)))
        if result.returncode:
            with (output/'ssh-errors.log').open('a',encoding='utf-8') as log:log.write(result.stderr[-12000:])
            raise RuntimeError('Owned server audit RPC failed; inspect saved stderr, do not relaunch')
        return json.loads(result.stdout)
    failure=None;observation=None
    try:
        observation=rpc('start');atomic_json(output/'server-start-observation.json',observation)
        lease['identity']=observation['identity'];atomic_json(output/'remote-job.json',lease)
        while not observation['terminal']:
            if time.time()>deadline-30:raise TimeoutError('Audit observation deadline; independent service limit remains')
            atomic_json(output/'progress.json',{'at':utcnow(),'stage':'observing_owned_service','identity':lease['identity'],
                                               'service':observation['service_properties']})
            time.sleep(2);observation=rpc('query')
    except Exception as exc:failure=exc
    finally:
        try:
            final=rpc('query');observation=final
            if lease['identity'] is None:lease['identity']=final['identity']
            if final['identity']!=lease['identity']:raise RuntimeError('Final identity differs')
            atomic_json(output/'server-observation.json',final)
            if final['terminal']:
                receipt={**lease,'host_boot_id':final['host_boot_id'],'server_unix':final['server_unix'],
                         'service_properties':final['service_properties'],'at':utcnow()}
                receipt_path=output/'remote-terminal.json';atomic_json(receipt_path,receipt)
                project=Path(config['project_path']);lease.update(terminal_receipt_path=receipt_path.relative_to(project).as_posix(),
                                                                terminal_receipt_sha256=sha256(receipt_path))
            atomic_json(output/'remote-job.json',lease)
        except Exception as exc:
            atomic_json(output/'reconciliation-needed.json',{'at':utcnow(),'reason':type(exc).__name__,
                                                          'instruction':'Query this same owned service; never restart on observation failure'})
            if failure is None:failure=exc
    if failure:raise failure
    validate_no_unreconciled_remote(Path(config['project_path']),{'remote_job':lease})
    props=observation['service_properties'];live=observation['live_owned_processes'];negative=observation['negative_identity']
    start=read_json(output/'server-start-observation.json');kernel=start['kernel_cgroup_files']
    runtime=(int(props['ExecMainExitTimestampMonotonic'])-int(props['ExecMainStartTimestampMonotonic']))/1000000
    passed=(props.get('Result')=='timeout' and int(props.get('MemoryMax','0'))==64*1024**2
            and 0<duration_seconds(props.get('RuntimeMaxUSec',''))<=15 and observation['checkpoint'] is not None
            and 0<runtime<=22 and kernel.get('memory.max')==str(64*1024**2) and kernel.get('memory.swap.max')=='0'
            and live=={'parent':False,'child':False} and negative and negative['rejected'] and negative['same_unit_still_present'])
    atomic_json(output/'metrics.json',{'at':utcnow(),'complete':bool(passed),'engineering_only':True,
                'runtime_and_process_group_kill_verified':bool(passed),'wrong_invocation_stop_rejected':bool(negative and negative['rejected']),
                'synthetic_checkpoint_retained':observation['checkpoint'] is not None,'actual_ppo_checkpoint_recovery_verified':False,
                'memory_kernel_limit_verified':kernel.get('memory.max')==str(64*1024**2),'memory_oom_exercised':False,
                'observed_worker_lifetime_seconds':runtime,
                'server_training_started':False,'promotion_eligible':False,'observation_sha256':sha256(output/'server-observation.json')})
    if not passed:raise RuntimeError('Server boundary evidence did not meet preregistered checks')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();run(read_json(args.config),args.output)
