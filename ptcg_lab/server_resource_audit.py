"""Controlled no-GPU probes of real shared-host protection behavior."""
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


PROFILES=('io_stop','heartbeat_loss','guardian_fault','disk_guard','normal_completion')


def audit_scope_seconds(project,exclude=None):
    """All guard revisions share the original scope; task aliases grant no time."""
    spent=0.
    for directory in (Path(project)/'runs').iterdir():
        cp=directory/'config.json';ap=directory/'attempts.json'
        if directory!=exclude and cp.exists() and read_json(cp).get('job_kind')=='server_resource_audit' and ap.exists():
            spent+=sum(a['seconds'] for a in read_json(ap))
    state_path=Path(project)/'research/autonomy/state.json'
    if state_path.exists():
        spent+=sum(row['seconds'] for row in read_json(state_path).get('environment_compute_charges',[])
                   if row.get('resource_audit_scope') is True)
    return spent


def evaluate(profile,prepared,final,pacing=None):
    artifacts=final['artifacts'];started=artifacts.get('guardian-start.json',{});result=artifacts.get('guardian-result.json',{})
    payload=artifacts.get('payload-started.json',{});kernel=started.get('kernel',{});checks={}
    try:
        quota,period=kernel['cpu.max'].split();checks['cpu_kernel_limit']=quota!='max' and int(quota)/int(period)<=2
    except (KeyError,ValueError):checks['cpu_kernel_limit']=False
    checks['memory_kernel_limit']=kernel.get('memory.max')==str(128*2**20) and kernel.get('memory.swap.max')=='0'
    checks['directory_ownership_and_realpath']=(payload.get('directory_escape_refused') is True
        and payload.get('root_uid')==1005 and payload.get('canonical_root')=='OLD_READONLY_ROOT/ptcg-hybrid-v3')
    expected_result='success' if profile=='normal_completion' else 'exit-code'
    checks['owned_unit_terminal']=final['terminal'] and final['service_properties'].get('Result')==expected_result
    processes=final.get('processes_after_terminal',{})
    checks['owned_processes_terminated']=bool(processes) and all(not x['alive'] for x in processes.values())
    reason=result.get('reason','');checks['guardian_result_present']=bool(result)
    block=started.get('kernel_block_io',{})
    checks['kernel_block_io_upper_bound_present']=(block.get('scope')=='whole_host_upper_bound'
        and block.get('source')=='/sys/fs/cgroup/io.stat' and block.get('project_device')=='8:0'
        and block.get('project_device') in block.get('devices',{}))
    if profile=='io_stop':
        events=result.get('events',[]);freezes=[e for e in events if e['action']=='freeze'];thaws=[e for e in events if e['action']=='thaw']
        checks['io_throttle_exercised']=bool(freezes and thaws) and all(all(p['state'] in ('T','t','Z') for p in e['members']) for e in freezes)
        normal=artifacts.get('normal-done.json',{});child=artifacts.get('child-ready.json',{})
        checks['paced_payload_duration']=normal.get('bytes')==24*2**20 and normal.get('seconds',0)>=5
        seen={r['pid'] for sample in result.get('recent_samples',[]) for r in sample.get('members',[])}
        checks['payload_child_accounted']=child.get('pid') in seen
        short=artifacts.get('short-child-accounting.json',{})
        def io(text):return {k:int(v) for k,v in (line.split(':',1) for line in text.splitlines())}
        try:checks['waited_child_io_accounted']=short['returncode']==0 and io(short['parent_io_after_wait'])['wchar']-io(short['parent_io_before'])['wchar']>=short['bytes']
        except (KeyError,ValueError):checks['waited_child_io_accounted']=False
        checks['exited_child_kernel_block_io_retained']=(short.get('block_io_after_wait',{}).get('project_device_delta',[0,0])[1]>=2**20)
        checks['zombie_has_independent_kernel_coverage']=any(
            event.get('action')=='kernel_zombie_coverage' and event.get('kernel_block_io',{}).get('scope')=='whole_host_upper_bound'
            and any(row.get('state')=='Z' for row in event.get('members',[])) for event in result.get('events',[]))
        checks['io_excess_stops_owned_unit']=reason=='io_burst_exceeds_stop_boundary'
        checks['checkpoint_marker_retained']='checkpoint-marker.json' in artifacts
        checks['control_transfer_io_paced']=(bool(pacing) and pacing.get('bytes')==4*2**20
            and pacing.get('write_seconds',0)>=1.9 and pacing.get('read_seconds',0)>=1.9
            and pacing.get('sha256')==hashlib.sha256(b'c'*(4*2**20)).hexdigest())
    elif profile=='heartbeat_loss':checks['monitor_loss_stops_owned_unit']=reason=='control_heartbeat_lost'
    elif profile=='guardian_fault':checks['guard_failure_stops_owned_unit']=reason.startswith('guardian_failure:RuntimeError:Injected guardian failure')
    elif profile=='disk_guard':
        tests=artifacts.get('disk-refusal.json',{}).get('tests',{})
        checks['disk_reserve_and_workset_stop']=(reason=='disk_admission_refused'
            and tests.get('workset',{}).get('refused') is True and tests.get('free_reserve',{}).get('refused') is True)
    elif profile=='normal_completion':
        positive=artifacts.get('positive-pip-result.json',{})
        checks['clean_payload_completion']=(reason=='payload_completed_after_exit_barrier' and result.get('clean_completion') is True)
        checks['private_inprocess_pip_install']=(positive.get('installed_attrs_version')=='26.1.0'
            and '/probe-env/' in positive.get('attrs_module','') and positive.get('private_prefix','').endswith('/probe-env'))
        checks['same_pid_exec_io_accounting']=(positive.get('pid_before_exec')==positive.get('pid_after_exec')
            and bool(positive.get('io_after_exec')) and all(positive['io_after_exec'][k]>=positive['io_before_exec'][k]
                for k in ['rchar','wchar','read_bytes','write_bytes']))
        checks['exit_barrier_acknowledged']='guardian-exit-ack.json' in artifacts
    else:raise ValueError('Unregistered audit profile')
    return {'checks':checks,'complete':all(checks.values()),'reason':reason,'profile':profile}


def run(config,output):
    output=Path(output).resolve();project=Path(config['project_path']);source=Path(__file__).resolve().parent
    if os.environ.get('PTCG_JOB_RUN_ID')!=output.name:raise RuntimeError('Resource probe requires controlled runner')
    profile=config['audit_profile']
    if profile not in PROFILES or config['audit_total_io_bytes_max']>64*2**20:raise ValueError('Unreviewed resource payload')
    expected=config.get('expected_guard_source_sha256')
    if expected and sha256(source/'process_guardian.py')!=expected:
        raise RuntimeError('Preregistered guard revision changed')
    if (output/'remote-job.json').exists():raise RuntimeError('Existing resource job must be reconciled; do not relaunch')
    deadline=float(os.environ['PTCG_JOB_DEADLINE_UNIX']);spent=audit_scope_seconds(project,output)
    if spent+deadline-time.time()>min(600,config['audit_scope_seconds_max'])+.5:
        raise RuntimeError('Cumulative resource-audit scope cap exhausted')
    names=['io.py','server_workspace.py','systemd_job.py','unit_io_guard.py','process_guardian.py','server_resource_worker.py','io_pacing.py','host_block_io.py','bounded_io.py','bounded_read.py']
    files={'ptcg_lab/'+name:(source/name).read_bytes() for name in names}
    files.update({'ptcg_lab/__init__.py':b'','resource_worker.py':b'from ptcg_lab.process_guardian import main\nmain()\n',
                  'payload_worker.py':b'from ptcg_lab.server_resource_worker import main\nmain()\n'})
    payload={name:{'base64':base64.b64encode(data).decode(),'sha256':hashlib.sha256(data).hexdigest()} for name,data in files.items()}
    agent_path=source/'server_resource_agent.py'
    atomic_json(output/'remote-source.json',{**{name:x['sha256'] for name,x in payload.items()},'endpoint-agent.py':sha256(agent_path)})
    epoch=read_json(output/'experiment.json')['remote_lease_epoch'];unit=f"ptcg-hybrid-resource-{output.name.rsplit('_',1)[1]}-{epoch}.service"
    lease={'run_id':output.name,'attempt':1,'lease_epoch':epoch,'host':config['host'],'unit':unit,'deadline_unix':deadline-15,'identity':None}
    stream=SSHStream(config,agent_path,output/'ssh-stream-errors.log',deadline);observation=None;pacing=None;failure=None
    try:
        input_metadata={};input_paths={}
        if profile=='normal_completion':
            manifest_path=project/config['positive_wheel_manifest_path']
            if sha256(manifest_path)!=config['positive_wheel_manifest_sha256']:raise ValueError('Verified positive wheel manifest changed')
            for row in read_json(manifest_path)['files']:
                if row['name'].lower() in ('pip','attrs'):
                    name=row['name'].lower();path=Path(row['source_path'])
                    if sha256(path)!=row['sha256']:raise ValueError('Positive wheel bytes changed')
                    input_metadata[name]={k:row[k] for k in ['file','version','bytes','sha256']};input_paths[name]=path
        prepared=stream.request('initialize',{'project_id':config['project_id'],'run_id':output.name,'unit':unit,'lease_epoch':epoch,
                    'deadline_unix':lease['deadline_unix'],'client_unix':time.time(),'files':payload,'agent_sha256':sha256(agent_path),
                    'profile':profile,'input_wheels':input_metadata})
        atomic_json(output/'server-prepared.json',prepared)
        for name,path in input_paths.items():stream.request('put_input',{'name':name},data=path.read_bytes(),timeout=45)
        if profile=='io_stop':pacing=stream.request('pace',data=b'c'*(4*2**20),timeout=45);atomic_json(output/'control-pacing-proof.json',pacing)
        atomic_json(output/'remote-job.json',lease);observation=stream.request('start');lease['identity']=observation['identity']
        atomic_json(output/'remote-job.json',lease);atomic_json(output/'server-start-observation.json',observation)
        started=time.monotonic()
        while not observation['terminal']:
            if time.time()>deadline-20:raise TimeoutError('Bounded audit observation deadline')
            atomic_json(output/'progress.json',{'at':utcnow(),'profile':profile,'stage':'observing_resource_guard',
                        'seconds':time.monotonic()-started,'remote':observation['artifacts'].get('guardian-progress.json')})
            time.sleep(.5)
            heartbeat=profile!='heartbeat_loss' or time.monotonic()-started<1
            observation=stream.request('status',{'heartbeat':heartbeat})
    except Exception as exc:failure=exc
    finally:
        try:
            if lease['identity']:
                final=stream.request('status',{'heartbeat':False});observation=final;atomic_json(output/'server-final-observation.json',final)
                if final['terminal']:
                    if final['identity']!=lease['identity']:raise RuntimeError('Final resource instance changed')
                    rp=output/'remote-terminal.json';atomic_json(rp,{**lease,'at':utcnow(),'host_boot_id':final['host_boot_id'],
                                'server_unix':final['server_unix'],'service_properties':final['service_properties']})
                    lease.update(terminal_receipt_path=rp.relative_to(project).as_posix(),terminal_receipt_sha256=sha256(rp))
                atomic_json(output/'remote-job.json',lease)
            stream.request('close')
        except Exception as exc:
            atomic_json(output/'reconciliation-needed.json',{'at':utcnow(),'error':type(exc).__name__+': '+str(exc),'action':'Query same resource unit; never restart it'})
            if failure is None:failure=exc
        finally:stream.close()
    if failure:raise failure
    validate_no_unreconciled_remote(project,{'remote_job':lease})
    result=evaluate(profile,prepared,observation,pacing)
    atomic_json(output/'metrics.json',{**result,'at':utcnow(),'engineering_only':True,'server_training_started':False,
                'promotion_eligible':False,'observation_sha256':sha256(output/'server-final-observation.json'),
                'io_mode':'mandatory whole-host kernel block-I/O upper bound plus auxiliary owned-process early pacing; own-session signals only; no kernel io.max'})
    if not result['complete']:raise RuntimeError('Real shared-host resource checks did not pass: '+str(result['checks']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(read_json(a.config),a.output)
