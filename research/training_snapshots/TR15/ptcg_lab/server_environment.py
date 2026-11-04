"""Controlled local orchestration of one offline Linux environment install."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import time

from .io import atomic_json,read_json,sha256,utcnow
from .remote_lease import validate_no_unreconciled_remote
from .ssh_stream import SSHStream


def durable_bytes(path,data,expected=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha256(data).hexdigest()
    if expected is not None and digest!=expected:raise ValueError('Archive data hash differs')
    if path.exists():
        if sha256(path)!=digest:raise ValueError('Existing archive evidence differs')
        return digest
    temporary=path.with_suffix(path.suffix+'.incoming')
    with temporary.open('xb') as output:output.write(data);output.flush();os.fsync(output.fileno())
    temporary.replace(path)
    return digest


def archive_receipt(directory,response,item,index):
    receipt=response['receipt']
    if (receipt['index']!=index or receipt['source_sha256']!=item['sha256'] or receipt['name']!=item['name']
            or receipt['source_bytes']!=item['bytes'] or receipt['version']!=item['version']):
        raise ValueError('Install receipt identity differs')
    data=base64.b64decode(response['receipt_base64'],validate=True)
    receipt_hash=durable_bytes(directory/'receipt.json',data,response['receipt_sha256'])
    if json.loads(data)!=receipt:raise ValueError('Raw and parsed receipt differ')
    log_hashes={}
    for name,record in response['logs'].items():
        if Path(name).name!=name:raise ValueError('Unsafe archive log path')
        log_hashes[name]=durable_bytes(directory/name,base64.b64decode(record['base64'],validate=True),record['sha256'])
    if log_hashes.get(f'{index:03d}.log')!=receipt['log_sha256'] or log_hashes.get(f'{index:03d}-version.log')!=receipt['version_log_sha256']:
        raise ValueError('Required install log archive missing')
    commit={'at':utcnow(),'index':index,'receipt_sha256':receipt_hash,'source_sha256':item['sha256'],'logs':log_hashes}
    atomic_json(directory/'commit.json',commit)
    return {'index':index,'receipt_sha256':receipt_hash,'source_sha256':item['sha256'],
            'log_sha256':receipt['log_sha256'],'local_commit_sha256':sha256(directory/'commit.json')}


def run(config,output):
    output=Path(output).resolve();project=Path(config['project_path']);source=Path(__file__).resolve().parent
    if os.environ.get('PTCG_JOB_RUN_ID')!=output.name:raise RuntimeError('Environment setup requires controlled runner')
    if (output/'remote-job.json').exists():raise RuntimeError('Existing remote intent must be reconciled, never relaunched')
    deadline=float(os.environ['PTCG_JOB_DEADLINE_UNIX'])
    mode=config.get('preparation_mode');staged=mode in ('staged_direct_recovery','staged_range_repair')
    if staged:
        spent=sum(a['seconds'] for run_id in config.get('installation_scope_run_ids',[config['recovery_parent_run_id']])
                  for a in read_json(project/'runs'/run_id/'attempts.json'))
        ceiling=config.get('forward_cumulative_installation_seconds_max',config['installation_scope_seconds_max'])
        if abs(spent-config['prior_installation_scope_seconds'])>.01 or spent+deadline-time.time()>ceiling+.5:
            raise RuntimeError('Original installation scope is exhausted or misaccounted')
    manifest_path=project/config['wheelhouse_manifest_path']
    if sha256(manifest_path)!=config['wheelhouse_manifest_sha256']:raise ValueError('Verified wheelhouse manifest changed')
    manifest=read_json(manifest_path)
    if not manifest['linux_dependency_closure_verified'] or manifest['estimated_peak_with_one_staged_wheel']>10*2**30:
        raise ValueError('Linux environment has not passed capacity/dependency admission')
    items=sorted(manifest['files'],key=lambda x:(x['name'].lower()!='pip',x['name'].lower()=='torch',x['name'].lower()))
    if len(items)!=44:raise ValueError('Exact environment closure required')
    for item in items:
        path=Path(item['source_path'])
        if project.resolve() not in path.resolve().parents or path.stat().st_size!=item['bytes'] or sha256(path)!=item['sha256']:
            raise ValueError('Local immutable wheel changed')
    remote_manifest={'schema_version':1,'source_manifest_sha256':config['wheelhouse_manifest_sha256'],
        'files':[{k:v for k,v in item.items() if k!='source_path'} for item in items],
        'estimated_peak_bytes':manifest['estimated_peak_with_one_staged_wheel'],
        'future_training_capsule_bytes':config.get('future_training_capsule_bytes',965000000),'runtime_buffer_bytes':2**30}
    if staged:
        from .staged_environment import recovery_descriptor,official_url
        import zlib
        remote_manifest['mode']=mode
        proof=recovery_descriptor(project,config,items)
        remote_manifest['recovery_zlib_base64']=base64.b64encode(zlib.compress(json.dumps(proof,separators=(',',':')).encode())).decode()
        selections={row['file']:row for row in manifest['source_selections']}
        for item in remote_manifest['files']:
            selection=selections[item['file']]
            if selection['sha256']!=item['sha256']:raise ValueError('Official URL selection hash differs')
            item['url']=official_url(selection['url'])
            if config.get('pypi_exact_byte_mirror') and item['url'].startswith('https://files.pythonhosted.org/packages/'):
                if config['pypi_exact_byte_mirror']!='https://mirrors.aliyun.com/pypi':raise ValueError('Unreviewed mirror base')
                item['original_official_url']=item['url']
                item['url']=official_url(config['pypi_exact_byte_mirror']+item['url'].removeprefix('https://files.pythonhosted.org'))
            if mode=='staged_range_repair':
                item['range_repair']={'workers':config.get('artifact_download_connections',2),'range_bytes':1024**2}
                if item['name'] in config.get('verified_fragment_candidates',{}):
                    fragment=dict(config['verified_fragment_candidates'][item['name']])
                    prefix=hashlib.sha256()
                    original=next(x for x in items if x['name']==item['name'])
                    left=fragment['bytes']
                    with Path(original['source_path']).open('rb') as stream:
                        while left:
                            block=stream.read(min(left,1024**2))
                            if not block:raise EOFError('Local source shorter than old fragment')
                            prefix.update(block);left-=len(block)
                    fragment['prefix_sha256']=prefix.hexdigest();item['range_repair']['resume']=fragment
        expanded=0;peak=0
        for item in items[18:]:
            expanded+=item['expanded_bytes'];peak=max(peak,expanded+item['bytes'])
        remote_manifest['additional_peak_bytes']=peak+remote_manifest['future_training_capsule_bytes']+2**30+128*2**20
    remote_manifest_path=output/'remote-wheel-manifest.json';atomic_json(remote_manifest_path,remote_manifest)
    files={name:(source/relative).read_bytes() for name,relative in {
        'ptcg_lab/io.py':'io.py','ptcg_lab/systemd_job.py':'systemd_job.py',
        'ptcg_lab/server_workspace.py':'server_workspace.py','ptcg_lab/server_environment_worker.py':'server_environment_worker.py',
        'ptcg_lab/bounded_io.py':'bounded_io.py','ptcg_lab/io_pacing.py':'io_pacing.py','ptcg_lab/bounded_read.py':'bounded_read.py'}.items()}
    files.update({'ptcg_lab/__init__.py':b'',
                  'environment_worker.py':b'from ptcg_lab.server_environment_worker import main\nmain()\n'})
    if staged:files['ptcg_lab/staged_environment.py']=(source/'staged_environment.py').read_bytes()
    if mode=='staged_range_repair':files['ptcg_lab/range_artifacts.py']=(source/'range_artifacts.py').read_bytes()
    payload={name:{'base64':base64.b64encode(data).decode(),'sha256':hashlib.sha256(data).hexdigest()} for name,data in files.items()}
    atomic_json(output/'remote-source.json',{**{name:item['sha256'] for name,item in payload.items()},
                                            'endpoint-agent.py':sha256(source/'server_environment_agent.py')})
    epoch=read_json(output/'experiment.json')['remote_lease_epoch'];unit='ptcg-hybrid-env-'+output.name.rsplit('_',1)[1]+'-'+str(epoch)+'.service'
    lease={'run_id':output.name,'attempt':1,'lease_epoch':epoch,'host':config['host'],'unit':unit,
           'deadline_unix':deadline-20,'identity':None}
    stream=SSHStream(config,source/'server_environment_agent.py',output/'ssh-stream-errors.log',deadline)
    observation=None;failure=None;completed=0
    def remaining():
        if time.time()>deadline-35:raise TimeoutError('Environment controller soft deadline')
    def progress(stage,**extra):
        atomic_json(output/'progress.json',{'at':utcnow(),'stage':stage,'completed_wheels':completed,'total_wheels':len(items),**extra})
    try:
        progress('initializing_persistent_control')
        clock_started=time.time()
        clock_result=stream.request('clock',{'client_unix':clock_started},timeout=15)
        clock_finished=time.time()
        if clock_finished-clock_started>10 or abs((clock_started+clock_finished)/2-clock_result['server_unix'])>5:
            raise RuntimeError('Clock handshake latency or skew exceeds bound')
        atomic_json(output/'clock-handshake.json',{**clock_result,'client_send_unix':clock_started,'client_receive_unix':clock_finished})
        prepared=stream.request('initialize',{'project_id':config['project_id'],'run_id':output.name,'unit':unit,
                 'mode':mode if staged else 'offline_upload',
                 'lease_epoch':epoch,'deadline_unix':lease['deadline_unix'],'client_unix':time.time(),'files':payload,
                 'agent_sha256':sha256(source/'server_environment_agent.py'),
                 'manifest_base64':base64.b64encode(remote_manifest_path.read_bytes()).decode(),
                 'manifest_sha256':sha256(remote_manifest_path)})
        atomic_json(output/'server-preparation.json',prepared)
        atomic_json(output/'remote-job.json',lease)
        observation=stream.request('start');lease['identity']=observation['identity']
        atomic_json(output/'remote-job.json',lease);atomic_json(output/'server-start-observation.json',observation)
        for index,item in enumerate(items):
            remaining()
            if not staged:
                progress('uploading_verified_wheel',index=index,package=item['name'],bytes_done=0,bytes_total=item['bytes'])
                stream.request('begin',{'index':index});offset=0;last_progress=0;computed=hashlib.sha256()
                with Path(item['source_path']).open('rb') as local:
                    while block:=local.read(4*1024**2):
                        remaining();digest=hashlib.sha256(block).hexdigest()
                        response=stream.request('chunk',{'index':index,'offset':offset,'sha256':digest},data=block,timeout=45)
                        offset+=len(block);computed.update(block)
                        if response!={'index':index,'offset':offset}:raise ValueError('Remote transfer cursor differs')
                        if time.monotonic()-last_progress>=5 or offset==item['bytes']:
                            progress('uploading_verified_wheel',index=index,package=item['name'],bytes_done=offset,bytes_total=item['bytes']);last_progress=time.monotonic()
                if offset!=item['bytes'] or computed.hexdigest()!=item['sha256']:raise ValueError('Local source changed during transfer')
                stream.request('commit',{'index':index},timeout=45)
            progress('waiting_install_receipt',index=index,package=item['name'])
            while True:
                remaining();observation=stream.request('status',{'index':index})
                progress('waiting_install_receipt',index=index,package=item['name'],remote_progress=observation.get('progress.json'),resources=observation.get('resources.json'))
                if observation.get('failure.json'):raise RuntimeError('Server installation failed; retain logs and staged source')
                if observation.get('receipt'):break
                if observation['terminal']:raise RuntimeError('Server ended before install receipt')
                time.sleep(3 if staged else 1)
            ack=archive_receipt(output/'remote-archive'/f'{index:03d}',observation,item,index)
            stream.request('ack',ack)
            while True:
                remaining();observation=stream.request('status',{'index':index})
                if observation.get('failure.json'):raise RuntimeError('Server failed during archive ACK commit')
                if observation['evicted']:break
                time.sleep(.5)
            completed+=1;progress('installed_archived_and_staged_replica_released',index=index,package=item['name'])
        while True:
            remaining();observation=stream.request('status')
            if observation['terminal']:break
            progress('official_cpu_environment_smoke',remote_progress=observation.get('progress.json'));time.sleep(1)
    except Exception as exc:failure=exc
    finally:
        try:
            if lease['identity']:
                final=stream.request('status');observation=final;atomic_json(output/'server-final-observation.json',final)
                for name,record in stream.request('archive_final').items():
                    durable_bytes(output/'remote-final-logs'/name,base64.b64decode(record['base64'],validate=True),record['sha256'])
                if final['terminal']:
                    if final['identity']!=lease['identity']:raise ValueError('Final service identity differs')
                    receipt={**lease,'host_boot_id':final['host_boot_id'],'server_unix':final['server_unix'],
                             'service_properties':final['service_properties'],'at':utcnow()}
                    path=output/'remote-terminal.json';atomic_json(path,receipt)
                    lease.update(terminal_receipt_path=path.relative_to(project).as_posix(),terminal_receipt_sha256=sha256(path))
                atomic_json(output/'remote-job.json',lease)
            stream.request('close')
        except Exception as exc:
            atomic_json(output/'reconciliation-needed.json',{'at':utcnow(),'error':type(exc).__name__+': '+str(exc),
                        'action':'Query same unit and versioned ticket; no relaunch on observation failure'})
            if failure is None:failure=exc
        finally:stream.close()
    if failure:raise failure
    validate_no_unreconciled_remote(project,{'remote_job':lease})
    metrics=observation.get('environment-metrics.json');properties=observation['service_properties']
    if (not metrics or not metrics['complete'] or properties.get('Result')!='success'
            or properties.get('ExecMainStatus')!='0' or completed!=44):raise RuntimeError('Environment service did not complete all checks')
    if metrics['peak_workspace_bytes']>=10*2**30 or metrics['minimum_free_bytes']<20*2**30:
        raise RuntimeError('Actual environment storage boundary failed')
    atomic_json(output/'metrics.json',{**metrics,'remote_environment_installed_and_cpu_smoke_verified':True,
                'final_service_identity':lease['identity'],'controller_protocol':'one_persistent_ssh_binary_framed_stream',
                'all_install_receipts_archived_locally_before_replica_eviction':True,
                'environment_manifest_sha256':sha256(remote_manifest_path),'actual_checkpoint_restore_verified':False})
    progress('environment_verified')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();run(read_json(args.config),args.output)
