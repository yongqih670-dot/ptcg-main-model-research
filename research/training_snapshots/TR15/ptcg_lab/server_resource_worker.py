"""Small fixed-payload resource probes; no GPU, model, or external data."""
import argparse
import contextlib
import importlib
import importlib.metadata
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import venv

from .io import atomic_json,read_json,utcnow
from .server_workspace import owned_root,inside,admit


def wait(path,deadline):
    while not path.exists():
        if time.time()>deadline-10:raise TimeoutError('Audit barrier deadline')
        time.sleep(.02)


def run(ticket_path,child=False,quick=False):
    t=read_json(ticket_path);root=owned_root(t['project_root'],t['project_id']);job=inside(root,t['job_directory']);deadline=t['deadline_unix']
    if quick:
        with inside(root,job/'quick-child.bin').open('xb') as output:
            output.write(b'q'*2**20);output.flush();os.fsync(output.fileno())
        time.sleep(.1);return
    if child:
        atomic_json(job/'child-ready.json',{'pid':os.getpid(),'at':utcnow()});wait(job/'child-go.json',deadline)
        started=time.monotonic();path=inside(root,job/'normal.bin')
        with path.open('xb') as output:
            for _ in range(96):
                output.write(b'n'*(256*1024));output.flush();os.fsync(output.fileno());time.sleep(.035)
            os.fsync(output.fileno())
        atomic_json(job/'normal-done.json',{'at':utcnow(),'pid':os.getpid(),'bytes':24*2**20,'seconds':time.monotonic()-started})
        wait(job/'burst-go.json',deadline)
        with inside(root,job/'burst.bin').open('xb') as output:
            output.write(b'b'*(16*2**20));output.flush();os.fsync(output.fileno())
        while True:time.sleep(.05)
    wait(job/'guardian-ready.json',deadline)
    escape=False
    try:inside(root,root.parent/'forbidden-never-created')
    except ValueError:escape=True
    if not escape:raise RuntimeError('Directory escape was not refused')
    atomic_json(job/'payload-started.json',{'at':utcnow(),'pid':os.getpid(),'directory_escape_refused':escape,
                'root_uid':root.stat().st_uid,'canonical_root':str(root)})
    if t['audit_profile']=='normal_completion':
        environment=inside(root,job/'probe-env');stage=job/'positive-exec-state.json'
        def counters():return {k:int(v) for k,v in (line.split(':',1) for line in Path('/proc/self/io').read_text().splitlines())}
        if not stage.exists():
            venv.EnvBuilder(with_pip=False,symlinks=True).create(environment)
            atomic_json(stage,{'at':utcnow(),'pid':os.getpid(),'io_before_exec':counters(),'prefix_before_exec':sys.prefix})
            python=str(environment/'bin/python')
            os.execv(python,[python,'-B',str(job/'control/payload_worker.py'),'--ticket',str(ticket_path)])
        previous=read_json(stage);after=counters()
        if Path(sys.prefix).resolve()!=environment.resolve() or previous['pid']!=os.getpid():raise RuntimeError('Private environment exec identity differs')
        if any(after[name]<previous['io_before_exec'][name] for name in ['rchar','wchar','read_bytes','write_bytes']):
            raise RuntimeError('Process I/O counters reset across exec')
        from .io import sha256
        inputs=t['input_wheels'];pip_path=inside(root,job/'inputs'/inputs['pip']['file']);attrs_path=inside(root,job/'inputs'/inputs['attrs']['file'])
        for name,path in [('pip',pip_path),('attrs',attrs_path)]:
            if sha256(path)!=inputs[name]['sha256']:raise RuntimeError('Positive probe wheel changed')
        sys.path.insert(0,str(pip_path));from pip._internal.cli.main import main as pip_main
        with inside(root,job/'positive-pip.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            code=pip_main(['install','--no-index','--no-deps','--no-cache-dir','--no-compile',str(attrs_path)])
        if code:raise RuntimeError('Positive isolated in-process pip install failed')
        importlib.invalidate_caches();import attrs
        if importlib.metadata.version('attrs')!=inputs['attrs']['version'] or not Path(attrs.__file__).resolve().is_relative_to(environment):
            raise RuntimeError('Positive installed package escaped private environment')
        atomic_json(job/'positive-pip-result.json',{'at':utcnow(),'pid_before_exec':previous['pid'],'pid_after_exec':os.getpid(),
                    'private_prefix':sys.prefix,'installed_attrs_version':importlib.metadata.version('attrs'),
                    'attrs_module':attrs.__file__,'io_before_exec':previous['io_before_exec'],'io_after_exec':after,
                    'pip_source_sha256':inputs['pip']['sha256'],'attrs_source_sha256':inputs['attrs']['sha256']})
        atomic_json(job/'payload-exit-ready.json',{'at':utcnow(),'pid':os.getpid(),'run_id':t['run_id']})
        wait(job/'guardian-exit-ack.json',deadline);return
    if t['audit_profile']=='disk_guard':
        tests={}
        for name,kwargs in [('workset',{'growth':11*2**30}),('free_reserve',{'growth':0,'free_reserve':shutil.disk_usage(root).free+1})]:
            try:admit(root,**kwargs)
            except OSError as exc:tests[name]={'refused':True,'error':str(exc)}
            else:raise RuntimeError('Disk negative admission unexpectedly passed')
        atomic_json(job/'disk-refusal.json',{'at':utcnow(),'tests':tests,'no_payload_allocation':True})
    elif t['audit_profile']=='io_stop':
        from .host_block_io import HostBlockCounters
        device=str(os.major(root.stat().st_dev))+':'+str(os.minor(root.stat().st_dev))
        block_counter=HostBlockCounters(device)
        before=Path('/proc/self/io').read_text()
        quick_process=subprocess.Popen([sys.executable,'-B',str(job/'control/payload_worker.py'),'--ticket',str(ticket_path),'--quick'],cwd=job/'control')
        # Deliberately expose a dead, unreaped child to the independent kernel
        # accounting guard, rather than relying on a lucky sample schedule.
        until=time.monotonic()+15
        while True:
            fields=Path('/proc',str(quick_process.pid),'stat').read_text().rsplit(')',1)[1].split()
            if fields[0]=='Z':break
            if time.monotonic()>until:raise TimeoutError('Bounded child-exit barrier')
            time.sleep(.02)
        wait(job/'kernel-zombie-covered.json',deadline)
        quick_process.wait(timeout=15)
        after=Path('/proc/self/io').read_text()
        atomic_json(job/'short-child-accounting.json',{'pid':quick_process.pid,'returncode':quick_process.returncode,
                    'bytes':2**20,'parent_io_before':before,'parent_io_after_wait':after,'block_io_after_wait':block_counter.sample()})
        process=subprocess.Popen([sys.executable,'-B',str(job/'control/payload_worker.py'),'--ticket',str(ticket_path),'--child'],cwd=job/'control')
        wait(job/'child-ready.json',deadline);time.sleep(.2)
        atomic_json(job/'child-go.json',{'at':utcnow()});wait(job/'normal-done.json',deadline)
        # Let the guardian sample a stable child before the intentional burst.
        time.sleep(.3);atomic_json(job/'checkpoint-marker.json',{'at':utcnow(),'stage':'normal_io_complete_before_intentional_burst'})
        atomic_json(job/'burst-go.json',{'at':utcnow()})
    elif t['audit_profile'] not in ('heartbeat_loss','guardian_fault'):raise ValueError('Unregistered resource probe')
    while True:
        if time.time()>deadline-10:raise TimeoutError('Probe independently reached deadline')
        time.sleep(.05)


def main():
    p=argparse.ArgumentParser();p.add_argument('--ticket',required=True);p.add_argument('--child',action='store_true');p.add_argument('--quick',action='store_true')
    a=p.parse_args();run(a.ticket,a.child,a.quick)
