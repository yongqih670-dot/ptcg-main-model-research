"""Offline per-wheel installation in the owned private Linux environment."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import venv

from .io import atomic_json,read_json,sha256,utcnow
from .server_workspace import admit,inside,owned_root


def run(ticket_path):
    ticket=read_json(ticket_path);root=owned_root(ticket['project_root'],ticket['project_id'])
    job=inside(root,ticket['job_directory']);manifest_path=job/'wheel-manifest.json'
    if ticket['kind']!='environment_setup' or os.environ.get('PTCG_JOB_RUN_ID')!=ticket['run_id']:
        raise ValueError('Environment job identity differs')
    if sha256(manifest_path)!=ticket['manifest_sha256']:raise ValueError('Environment manifest changed')
    manifest=read_json(manifest_path);deadline=ticket['deadline_unix'];stop=False;child=None
    peak=0;min_free=None;installed=[];last_resources=0
    staged=manifest.get('mode') in ('staged_direct_recovery','staged_range_repair')
    if staged:
        from .staged_environment import validate_recovery_owner,kernel_snapshot,verify_prefix_item,download_wheel
        validate_recovery_owner(root,manifest)
        if Path(os.sys.prefix).resolve()!=root/'.venv':raise RuntimeError('Recovery interpreter is not the private environment')
        protection=kernel_snapshot(ticket)
        atomic_json(job/'hard-protection.json',{'at':utcnow(),'kernel':protection,
                    'run_id':ticket['run_id'],'unit':ticket['unit'],'hostwide_io_is_background_only':True,
                    'download_bytes_per_second':4*2**20,'concurrent_artifacts':1,'gpu_visible':os.environ.get('CUDA_VISIBLE_DEVICES')})
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('Environment preparation must not use a GPU')
    def stopped(*_):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGINT,stopped)
    def capacity(growth=0):
        nonlocal peak,min_free,last_resources
        record=admit(root,growth);peak=max(peak,record['workspace_bytes'])
        min_free=record['free_bytes'] if min_free is None else min(min_free,record['free_bytes'])
        if staged and time.monotonic()-last_resources>=10:
            atomic_json(job/'resources.json',{'at':utcnow(),'capacity':record,'kernel':kernel_snapshot(ticket)})
            last_resources=time.monotonic()
        return record
    def check():
        if stop or time.time()>deadline-45:raise TimeoutError('Environment checkpointed at stop/deadline')
    def progress(stage,**extra):
        atomic_json(job/'progress.json',{'at':utcnow(),'stage':stage,'installed':len(installed),
                    'peak_workspace_bytes':peak,'minimum_free_bytes':min_free,**extra})
    def execute(command,log_path,limit=300):
        nonlocal child
        check();started=time.monotonic()
        with log_path.open('wb') as log:
            child=subprocess.Popen(command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
            try:
                while child.poll() is None:
                    check();capacity()
                    if time.monotonic()-started>limit:raise TimeoutError('Single environment command limit')
                    time.sleep(.5)
                if child.returncode:raise RuntimeError('Offline environment command failed: '+log_path.name)
            finally:
                if child.poll() is None:
                    child.terminate()
                    try:child.wait(timeout=5)
                    except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)
        child=None
    try:
        capacity();environment=root/'.venv'
        if not staged:
            if environment.exists():raise RuntimeError('Existing environment requires explicit recovery; do not overwrite')
            capacity(128*1024**2)
            atomic_json(root/'environment-owner.json',{'project_id':ticket['project_id'],'creator_run_id':ticket['run_id'],
                        'manifest_sha256':ticket['manifest_sha256'],'status':'creating','at':utcnow()})
            venv.EnvBuilder(with_pip=False,symlinks=True).create(environment)
        python=str(environment/'bin/python');progress('waiting_first_wheel')
        for index,item in enumerate(manifest['files']):
            reused=staged and index<18;wheel=None;wheel_identity=None
            log=job/'logs'/f'{index:03d}.log';version_log=job/'logs'/f'{index:03d}-version.log'
            if reused:
                progress('verifying_installed_prefix',index=index,package=item['name'])
                verified=verify_prefix_item(root,manifest['recovery']['prefix'][index],check)
                log.write_text(json.dumps({'verified_files':verified,'original_receipt_sha256':manifest['recovery']['prefix'][index]['receipt_sha256']})+'\n',encoding='utf-8')
            else:
                if staged:
                    download=download_wheel
                    if manifest.get('mode')=='staged_range_repair':
                        from .range_artifacts import download
                    capacity(item['bytes']+item['expanded_bytes'])
                    wheel=download(root,job,item,deadline,check,capacity,
                         lambda stage,**extra:progress(stage,index=index,package=item['name'],**extra))
                else:
                    request_path=job/'requests'/f'{index:03d}.json'
                    while not request_path.exists():check();capacity();time.sleep(.25)
                    request=read_json(request_path)
                    if request!={'index':index,'manifest_sha256':ticket['manifest_sha256'],'sha256':item['sha256']}:
                        raise ValueError('Install request differs from immutable manifest')
                    wheel=inside(root,job/'stage'/item['file'])
                    if wheel.stat().st_size!=item['bytes'] or sha256(wheel)!=item['sha256']:raise ValueError('Staged wheel hash differs')
                st=wheel.stat();wheel_identity=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)
                capacity(item['expanded_bytes']+128*1024**2)
                progress('installing',index=index,package=item['name'])
                options=['install','--no-index','--no-deps','--no-cache-dir','--no-compile','--no-warn-script-location',str(wheel)]
                if index==0:
                    if item['name'].lower()!='pip':raise ValueError('Offline pip bootstrap must be first')
                    code='import sys;sys.path.insert(0,sys.argv.pop(1));from pip._internal.cli.main import main;raise SystemExit(main(sys.argv[1:]))'
                    command=[python,'-B','-c',code,str(wheel),*options]
                else:command=[python,'-B','-m','pip',*options]
                execute(command,log)
            execute([python,'-B','-c','import importlib.metadata,sys;print(importlib.metadata.version(sys.argv[1]))',item['name']],version_log,limit=30)
            if version_log.read_text().strip()!=item['version']:raise ValueError('Installed distribution version differs')
            record={'at':utcnow(),'index':index,'name':item['name'],'version':item['version'],
                    'source_sha256':item['sha256'],'source_bytes':item['bytes'],'manifest_sha256':ticket['manifest_sha256'],
                    'log_sha256':sha256(log),'version_log_sha256':sha256(version_log),'capacity':capacity(),
                    'peak_workspace_bytes':peak,'minimum_free_bytes':min_free,'reused_verified_prefix':reused}
            receipt=job/'receipts'/f'{index:03d}.json';atomic_json(receipt,record);installed.append(record)
            progress('waiting_local_archive_ack',index=index,package=item['name'])
            ack_path=job/'acks'/f'{index:03d}.json'
            while not ack_path.exists():check();capacity();time.sleep(.25)
            ack=read_json(ack_path)
            if (ack.get('index')!=index or ack.get('receipt_sha256')!=sha256(receipt)
                    or ack.get('log_sha256')!=record['log_sha256'] or ack.get('source_sha256')!=item['sha256']):
                raise ValueError('Install archive ACK differs')
            if wheel is not None:
                wheel=inside(root,wheel);st=wheel.stat()
                if (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)!=wheel_identity:raise ValueError('Eviction source identity changed')
                wheel.unlink()  # Only the identity-verified, acknowledged replica in this new job.
            atomic_json(job/'evictions'/f'{index:03d}.json',{'at':utcnow(),'index':index,
                        'source_sha256':item['sha256'],'ack_sha256':sha256(ack_path),'staged_replica_removed':not reused,'reused':reused})
            progress('installed_and_archived',index=index,package=item['name'])
        smoke=r'''
import importlib.metadata,json,os,pathlib,platform,sys,hashlib
import numpy,torch
from kaggle_environments import make
from kaggle_environments.envs.cabt.cg import sim
env=make('cabt',configuration={'version':2,'bo':3})
sim.get_lib()
base=pathlib.Path(sim.__file__).parent
files={name:hashlib.sha256((base/name).read_bytes()).hexdigest() for name in ['libcg.so','sim.py','game.py']}
assert torch.__version__=='2.11.0+cu128'
assert numpy.__version__=='2.5.4'
assert importlib.metadata.version('kaggle-environments')=='1.33.2'
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_available()
x=torch.arange(8,dtype=torch.float32,requires_grad=True);loss=x.square().mean();loss.backward()
assert torch.isfinite(loss) and torch.isfinite(x.grad).all()
print('PTCG_ENV_RESULT='+json.dumps({'python':sys.version,'prefix':sys.prefix,'base_prefix':sys.base_prefix,
 'torch':torch.__version__,'numpy':numpy.__version__,'kaggle_environments':importlib.metadata.version('kaggle-environments'),
 'platform':platform.platform(),'native_cg_sha256':files,'cuda_hidden_for_cpu_worker':True}))
'''
        smoke_log=job/'logs/smoke.log';progress('official_cg_cpu_environment_smoke')
        execute([python,'-B','-c',smoke],smoke_log,limit=90)
        lines=[x for x in smoke_log.read_text(encoding='utf-8').splitlines() if x.startswith('PTCG_ENV_RESULT=')]
        if len(lines)!=1:raise ValueError('Environment smoke result missing')
        result=json.loads(lines[0].split('=',1)[1]);capacity()
        if Path(result['prefix']).resolve()!=environment.resolve():raise ValueError('Interpreter is outside project environment')
        metrics={'at':utcnow(),'complete':True,'engineering_only':True,'installed_wheels':len(installed),
                 'manifest_sha256':ticket['manifest_sha256'],'peak_workspace_bytes':peak,'minimum_free_bytes':min_free,
                 'smoke':result,'smoke_log_sha256':sha256(smoke_log),'actual_ppo_restore_verified':False,
                 'gpu_training_verified':False,'server_training_started':False,'promotion_eligible':False,
                 'reused_wheels':18 if staged else 0,'staged_installation':staged}
        atomic_json(job/'environment-metrics.json',metrics)
        atomic_json(root/'environment-owner.json',{'project_id':ticket['project_id'],'creator_run_id':ticket['run_id'],
                    'manifest_sha256':ticket['manifest_sha256'],'status':'installed_and_cpu_smoke_verified','at':utcnow()})
        progress('complete')
    except Exception as exc:
        atomic_json(job/'failure.json',{'at':utcnow(),'error':type(exc).__name__+': '+str(exc),
                    'installed':len(installed),'peak_workspace_bytes':peak,'minimum_free_bytes':min_free})
        raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--ticket',required=True)
    run(parser.parse_args().ticket)


if __name__=='__main__':main()
