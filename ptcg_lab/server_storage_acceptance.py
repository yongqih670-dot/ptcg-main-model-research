"""Bounded fresh-process acceptance of an archived, fully copied project."""
import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import time
import zipfile

from .io import atomic_json,read_json,sha256,utcnow
from .ssh_stream import SSHStream


def run(config,output):
    output=Path(output).resolve();project=Path(config['project_path']);package=Path(__file__).resolve().parent
    if os.environ.get('PTCG_JOB_RUN_ID')!=output.name:raise RuntimeError('Controlled acceptance required')
    if (output/'remote-job.json').exists():raise RuntimeError('Existing acceptance must be reconciled')
    deadline=min(float(os.environ['PTCG_JOB_DEADLINE_UNIX']),config.get('original_deadline_unix',float('inf')));epoch=read_json(output/'experiment.json')['remote_lease_epoch']
    copyrun=project/'runs'/config['copy_run_id'];copy_metrics=read_json(copyrun/'metrics.json')
    if (not copy_metrics.get('complete') or sha256(copyrun/'full-copy-manifest.json.gz')!=config['copy_manifest_sha256']):
        raise ValueError('Full cold copy and local evidence archive must precede acceptance')
    profile=read_json(copyrun/'target-actual-initialization.json')['ticket']['storage_profile']
    packed=io.BytesIO()
    with zipfile.ZipFile(packed,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for path in package.rglob('*.py'):
            if '__pycache__' not in path.parts:z.writestr('ptcg_lab/'+path.relative_to(package).as_posix(),path.read_bytes())
        z.writestr('acceptance_worker.py',b'from ptcg_lab.storage_acceptance_worker import main\nmain()\n')
    code=packed.getvalue();atomic_json(output/'acceptance-source-zip.json',{'sha256':hashlib.sha256(code).hexdigest(),'bytes':len(code)})
    unit='ptcg-hybrid-accept-'+output.name.rsplit('_',1)[1]+'-'+str(epoch)+'.service'
    lease={'run_id':output.name,'attempt':1,'lease_epoch':epoch,'host':config['host'],'unit':unit,'deadline_unix':deadline,'identity':None}
    stream=SSHStream(config,package/'storage_migration_agent.py',output/'ssh-errors.log',deadline)
    try:
        initial=stream.request('acceptance_initialize',{'run_id':output.name,'client_unix':time.time(),'deadline_unix':deadline,
            'lease_epoch':epoch,'unit':unit,'code_zip_base64':base64.b64encode(code).decode(),'code_zip_sha256':hashlib.sha256(code).hexdigest(),
            'storage_profile':profile,'server_resource_profile':config['server_resource_profile'],
            'gpu_restore_required':config.get('gpu_restore_required',True),
            'cpu_throughput_probe':config.get('cpu_throughput_probe',False),
            'previous_engineering_run_ids':config.get('previous_engineering_run_ids',[]),
            'public_metadata_sha256':sha256(package/'public_cg/metadata.json'),
            'copy_run_id':copyrun.name,'copy_manifest_sha256':config['copy_manifest_sha256'],
            'copy_terminal_receipt':read_json(copyrun/'remote-terminal.json')},timeout=90)
        atomic_json(output/'target-acceptance-initialization.json',initial);atomic_json(output/'remote-job.json',lease)
        started=stream.request('start',timeout=35);lease['identity']=started['identity'];atomic_json(output/'remote-job.json',lease)
        atomic_json(output/'remote-start.json',started)
        while True:
            last=stream.request('status',timeout=25);atomic_json(output/'progress.json',last)
            if last.get('foreign_GPU_processes'):
                atomic_json(output/'foreign-GPU-observation.json',last)
                last=stream.request('stop',timeout=55);break
            if last.get('acceptance-metrics.json',{}).get('complete'):
                last=stream.request('stop',timeout=55);atomic_json(output/'actual-verified-stop.json',last);break
            if last['terminal']:break
            if time.time()>deadline-70:last=stream.request('stop',timeout=55);break
            time.sleep(10)
        atomic_json(output/'remote-terminal.json',{**lease,**last,'observed_at':utcnow()})
        lease.update(terminal_receipt_path=(output/'remote-terminal.json').relative_to(project).as_posix(),
            terminal_receipt_sha256=sha256(output/'remote-terminal.json'));atomic_json(output/'remote-job.json',lease)
        names=['acceptance-metrics.json','full-restore.json','fresh-process-restore.json','simulator-check.json','hard-protection.json',
            'environment-entrypoint-originals.json','environment-adaptation.json','engine-manifest.json','engine.log','gpu-start-probe.json','copy-terminal-handoff-proof.json','stdout.log','stderr.log',
            'CPU-throughput-benchmark.json','cpu-throughput-2.log','cpu-throughput-4.log','engineering-selfplay-threads2.pt','engineering-selfplay-threads4.pt','environment-adaptation-chain.json','restored-training-state.pt']
        for name in names:
            offset=0;target=output/name
            while True:
                frame=stream.request('evidence',{'name':name,'offset':offset},timeout=30)
                if frame.get('absent'):break
                data=base64.b64decode(frame['base64'],validate=True)
                if frame['offset']!=offset or hashlib.sha256(data).hexdigest()!=frame['chunk_sha256']:raise ValueError('Acceptance evidence chunk digest differs')
                with target.open('xb' if offset==0 else 'ab') as f:f.write(data);f.flush();os.fsync(f.fileno())
                offset+=len(data)
                if offset==frame['bytes']:
                    if sha256(target)!=frame['file_sha256']:raise ValueError('Full acceptance evidence digest differs')
                    break
                if not data:raise EOFError('Truncated acceptance evidence')
        props=last['service_properties'];m=last.get('acceptance-metrics.json',{})
        positive_stop=(props.get('LoadState')=='loaded' and props.get('Result')=='success' and props.get('ExecMainStatus')=='0'
            or props.get('LoadState')=='not-found' and last.get('terminal_proof_kind')=='exact_invocation_systemd_journal'
            and last.get('same_invocation_stop_issued') is True)
        good=(positive_stop
            and m.get('complete') is True and (output/'actual-verified-stop.json').exists())
        atomic_json(output/'metrics.json',{'complete':bool(good),'actual_HDD2_full_recovery_acceptance':bool(good) and config.get('gpu_restore_required',True),
            'actual_HDD2_CPU_recovery_acceptance':bool(good),'CUDA_restore_pending':not config.get('gpu_restore_required',True),
            'actual_verified_own_invocation_stopped':(output/'actual-verified-stop.json').exists(),
            'training_updates':0,'migration_switched':False,'acceptance':m})
        if not good:raise RuntimeError('Actual relocated full recovery acceptance failed; no switch or training')
    finally:stream.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(read_json(a.config),a.output)


if __name__=='__main__':main()
