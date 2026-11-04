"""Local controlled runner for one server-side cold migration, without training."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import time

from .io import atomic_json,read_json,sha256,utcnow
from .ssh_stream import SSHStream
from .storage_profile import OLD,NEW,PROJECT


def run(config,output):
    output=Path(output).resolve();project=Path(config['project_path']);package=Path(__file__).resolve().parent
    if os.environ.get('PTCG_JOB_RUN_ID')!=output.name:raise RuntimeError('Migration requires controlled runner')
    if (output/'remote-job.json').exists():raise RuntimeError('Existing copy must be reconciled, never relaunched')
    deadline=float(os.environ['PTCG_JOB_DEADLINE_UNIX']);epoch=read_json(output/'experiment.json')['remote_lease_epoch']
    authority=read_json(project/config['migration_authority']['path'])
    if sha256(project/config['migration_authority']['path'])!=config['migration_authority']['sha256']:raise ValueError('Migration authority changed')
    profile={'schema':'ptcg_HDD2_storage_v1','project_id':PROJECT,'root':NEW,'hard_max_bytes':50*2**30,
        'warning_bytes':40*2**30,'free_reserve_bytes':20*2**30,'retained_roots':[OLD],'authority':authority,
        'authority_canonical_sha256':hashlib.sha256(json.dumps(authority,sort_keys=True,separators=(',',':')).encode()).hexdigest()}
    payload={}
    for name in ['io.py','systemd_job.py','resource_profile.py','storage_profile.py','server_workspace.py','storage_copy_worker.py']:
        data=(package/name).read_bytes();payload['ptcg_lab/'+name]={'base64':base64.b64encode(data).decode(),'sha256':hashlib.sha256(data).hexdigest()}
    for name,data in {'ptcg_lab/__init__.py':b'','migration_worker.py':b'from ptcg_lab.storage_copy_worker import main\nmain()\n'}.items():
        payload[name]={'base64':base64.b64encode(data).decode(),'sha256':hashlib.sha256(data).hexdigest()}
    unit='ptcg-hybrid-migrate-'+output.name.rsplit('_',1)[1]+'-'+str(epoch)+'.service'
    lease={'run_id':output.name,'attempt':1,'lease_epoch':epoch,'host':config['host'],'unit':unit,'deadline_unix':deadline,'identity':None}
    stream=SSHStream(config,package/'storage_migration_agent.py',output/'ssh-errors.log',deadline)
    last=None
    try:
        initial=stream.request('initialize',{'run_id':output.name,'client_unix':time.time(),'deadline_unix':deadline,
            'lease_epoch':epoch,'unit':unit,'files':payload,'storage_profile':profile,
            'server_resource_profile':config['server_resource_profile'],'io_bytes_per_second':config['io_bytes_per_second']},timeout=60)
        atomic_json(output/'target-actual-initialization.json',initial)
        atomic_json(output/'remote-job.json',lease)
        started=stream.request('start',timeout=35);lease['identity']=started['identity'];atomic_json(output/'remote-job.json',lease)
        atomic_json(output/'remote-start.json',started)
        while True:
            if time.time()>deadline-65:
                last=stream.request('stop',timeout=55);break
            last=stream.request('status',timeout=25);atomic_json(output/'progress.json',last)
            if last['terminal']:break
            time.sleep(10)
        atomic_json(output/'remote-terminal.json',{**lease,**last,'observed_at':utcnow()})
        lease.update(terminal_receipt_path=(output/'remote-terminal.json').relative_to(project).as_posix(),
            terminal_receipt_sha256=sha256(output/'remote-terminal.json'));atomic_json(output/'remote-job.json',lease)
        for name in ['full-copy-manifest.json.gz','actual-resource-limits.json','copy-progress.json','stdout.log','stderr.log','copy-failure.json']:
            offset=0;target=output/name
            while True:
                frame=stream.request('evidence',{'name':name,'offset':offset},timeout=30)
                if frame.get('absent'):break
                data=base64.b64decode(frame['base64'],validate=True)
                if frame['offset']!=offset or hashlib.sha256(data).hexdigest()!=frame['chunk_sha256']:raise ValueError('Evidence transport digest differs')
                with target.open('xb' if offset==0 else 'ab') as f:f.write(data);f.flush();os.fsync(f.fileno())
                offset+=len(data)
                if offset==frame['bytes']:
                    if sha256(target)!=frame['file_sha256']:raise ValueError('Full evidence archive differs')
                    break
                if not data:raise EOFError('Truncated migration evidence')
        props=last['service_properties'];progress=last.get('copy-progress.json',{})
        good=props.get('LoadState')=='loaded' and props.get('Result')=='success' and props.get('ExecMainStatus')=='0' and progress.get('complete') is True
        atomic_json(output/'metrics.json',{'complete':bool(good),'cold_copy_only':True,'migration_switched':False,
            'training_started':False,'source_root':OLD,'target_root':NEW,'progress':progress,
            'no_old_attempt_cost_stop_or_budget_reset':True})
        if not good:raise RuntimeError('Cold copy did not complete; source and partial target preserved')
    finally:stream.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    run(read_json(a.config),a.output)


if __name__=='__main__':main()
