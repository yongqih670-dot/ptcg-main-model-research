"""Snapshot + bounded official BO3 job + durable cumulative CPU accounting."""
import argparse
import json
import os
import shutil
import stat
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ptcg_lab.io import atomic_json, read_json, read_series, sha256, utcnow
from ptcg_lab.evaluate import summarize, recover_rows
from ptcg_lab.processes import KillOnCloseJob
from ptcg_lab.stage_control import is_v3, validate_stage_job, register_stage_run
from ptcg_lab.remote_lease import validate_no_unreconciled_remote,sync_remote_intent
from ptcg_lab.shared_server_safety import validate_shared_server_job

JOB_MODULES={'evaluation':'ptcg_lab.evaluate','prepare_bc':'ptcg_lab.prepare_bc','train_catboost':'ptcg_lab.train_catboost',
             'prepare_tokens':'ptcg_lab.prepare_tokens','train_transformer':'ptcg_lab.train_transformer',
             'tr00_audit':'ptcg_lab.rl_audit','train_ppo':'ptcg_lab.train_ppo',
             'package_probe':'ptcg_lab.package_probe','linux_probe':'ptcg_lab.linux_probe',
             'server_handoff':'ptcg_lab.server_handoff','compact_audit':'ptcg_lab.compact_audit',
             'wheelhouse_prepare':'ptcg_lab.wheelhouse_prepare','ranged_wheelhouse':'ptcg_lab.ranged_wheelhouse',
             'server_boundary_audit':'ptcg_lab.server_boundary_audit','server_environment':'ptcg_lab.server_environment',
             'server_resource_audit':'ptcg_lab.server_resource_audit',
             'remote_training_check':'ptcg_lab.server_trial','remote_train_ppo':'ptcg_lab.server_trial',
             'server_artifact_archive':'ptcg_lab.server_archive',
             'server_storage_migration':'ptcg_lab.server_storage_migration',
             'server_storage_acceptance':'ptcg_lab.server_storage_acceptance',
             'local_cache_retirement':'ptcg_lab.local_cache_retirement',
             'local_cache_finalize':'ptcg_lab.local_cache_finalize'}


def compute_kind(config):
    if config.get('job_kind')=='server_storage_acceptance' and config.get('gpu_restore_required') is False:return 'cpu'
    return 'gpu' if config.get('job_kind') in ('train_transformer','tr00_audit','train_ppo','remote_train_ppo','server_storage_acceptance') else 'cpu'


def runtime_manifest(config):
    return ROOT/'research/environment'/('gpu-manifest.json' if compute_kind(config)=='gpu' else 'manifest.json')


def hypothesis_seconds(config,state):
    identity=config.get('budget_hypothesis_id',config.get('hypothesis_id'))
    if not identity:return 0.
    total=0.
    for run in (ROOT/'runs').iterdir():
        if not (run/'config.json').exists():continue
        previous_config=read_json(run/'config.json')
        if previous_config.get('budget_hypothesis_id',previous_config.get('hypothesis_id'))!=identity:continue
        attempts=read_json(run/'attempts.json') if (run/'attempts.json').exists() else []
        total+=sum(charged_attempt_seconds(run.name,a,state) for a in attempts)
        recorded={a['attempt'] for a in attempts}
        for recovery in state.get('recovery_accounting',[]):
            previous=recovery['previous']
            if previous['run_id']==run.name and previous['attempt'] not in recorded:
                total+=previous['limit_seconds'];recorded.add(previous['attempt'])
    return total


def charged_attempt_seconds(run_id,attempt,state):
    gross=attempt['seconds']
    if 'counted_execution_seconds' in attempt:
        counted=attempt['counted_execution_seconds']
        if not 0<=counted<=gross:raise ValueError('Attempt clock charge differs')
        return counted
    rows=[r for r in state.get('resource_wait_accounting',[]) if r['run_id']==run_id and r.get('attempt',1)==attempt['attempt']]
    if not rows:return gross
    if len(rows)!=1:raise ValueError('Duplicate shared wait credit')
    row=rows[0];proof=ROOT/row['evidence']['path']
    if sha256(proof)!=row['evidence']['sha256']:raise ValueError('Shared wait evidence changed')
    credit=row['exempt_shared_wait_seconds']
    if not 0<=credit<=gross:raise ValueError('Wait credit exceeds attempt')
    return gross-credit


def validate_plateau(config,policy,state,research,final):
    if is_v3(policy):return  # checked by validate_stage_job, including resume
    if not research or final:return
    stop=policy['stopping']
    if state.get('consecutive_valid_runs_without_promotion',0)>=stop['consecutive_valid_experiments_without_promotion']:
        if not config.get('low_cost_pivot'):
            raise RuntimeError('Plateau reached: only predeclared low-cost pivots remain')
        if state.get('started_pivot_runs',state.get('completed_pivot_runs',0))>=stop['additional_low_cost_pivot_experiments_max']:
            raise RuntimeError('Plateau pivot allowance exhausted')


def validate_cycle_open(state):
    validate_no_unreconciled_remote(ROOT,state)
    if state.get('status') in ('plateau','budget_exhausted','stopped_by_user','submitted_verified','activating'):
        raise RuntimeError('Research cycle is terminal: '+state['status']+'; new computation requires an authorized policy/state change')


def validate_execution_target(config,state):
    validate_shared_server_job(ROOT,config,state)
    target=state.get('execution_target',{})
    if target.get('required')=='hybrid':
        preparation={'server_handoff','compact_audit','wheelhouse_prepare','ranged_wheelhouse','server_boundary_audit','server_environment','server_resource_audit'}
        preparation|={'remote_training_check','remote_train_ppo'}  # Still gated by the exact staged route below.
        preparation.add('server_artifact_archive')
        preparation.add('server_storage_migration')
        preparation.add('server_storage_acceptance')
        active_local=preparation|{'evaluation','prepare_bc','prepare_tokens','package_probe','linux_probe','local_cache_retirement','local_cache_finalize'}
        allowed=active_local if target.get('status')=='active' else preparation
        if config.get('job_kind','evaluation') not in allowed:
            raise RuntimeError('Hybrid execution gate: local PPO training is forbidden; evaluation requires activated hybrid routing')
        return
    if target.get('required')=='remote_server' and config.get('job_kind')!='server_handoff':
        raise RuntimeError('Server migration gate: this local runner permits only hash-only handoff preparation; server jobs require a verified remote supervisor')


def collect_metrics(run, config):
    if config.get('job_kind','evaluation')=='evaluation':
        rows=recover_rows(run)
        metrics=summarize(rows)
        metrics['complete']=len(rows)==config['series_count']
        atomic_json(run/'metrics.json',metrics)
        return metrics
    return read_json(run/'metrics.json') if (run/'metrics.json').exists() else {'complete':False}


def is_same_process(pid, created):
    try:
        return abs(psutil.Process(pid).create_time()-created)<.01
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def acquire_lock(path):
    metadata={'pid':os.getpid(),'created':psutil.Process().create_time(),'at':utcnow()}
    if path.exists():
        existing=read_json(path)
        if is_same_process(existing['pid'],existing['created']):
            raise RuntimeError('Another research supervisor is running')
        child=existing.get('child')
        if child and is_same_process(child['pid'],child['created']):
            raise RuntimeError('Orphan child still alive: reconcile before starting another job')
        path.unlink()
    with path.open('x',encoding='utf-8') as f:
        json.dump(metadata,f)
    return metadata


def allowed_seconds(policy, state, final, requested, kind='cpu', config=None):
    b=policy['budget']
    ceiling=b[kind+'_job_wall_hours_max']-(0 if final else b[kind+'_job_wall_hours_reserved_final'])
    remaining=(ceiling-state['budget_used'][kind+'_job_wall_hours'])*3600
    job_limit=policy['execution']['job_wall_hours_max']*3600
    if config is not None:
        from ptcg_lab.volume_scope import limits
        forward=limits(ROOT,config,kind,policy,state)
        if forward is not None:
            return max(0,min(requested,forward['job_seconds_max'],remaining,forward['phase_remaining_seconds']))
        from ptcg_lab.continuous_runtime import local_limit
        authorized=local_limit(ROOT,config)
        if authorized is not None:job_limit=authorized
    return max(0,min(requested,job_limit,remaining))


def artifact_gib():
    # Existing official inputs are excluded. These directories were created by
    # this autonomous cycle, including environments, downloads and every run.
    directories=('.venv','.venv-gpu','agents','configs','ptcg_lab','scripts','tests','runs','releases',
                 'research/autonomy','research/environment','research/public_sources')
    total=0
    def scan_error(error):
        if not isinstance(error,FileNotFoundError):raise error
    for directory in directories:
        base=ROOT/directory
        if os.name=='nt':
            from ptcg_lab.archive_ack import publication_path
            base=Path(publication_path(base))
        try:base.stat()
        except FileNotFoundError:continue
        for folder,_,names in os.walk(base,onerror=scan_error,followlinks=False):
            for name in names:
                try:
                    info=os.stat(os.path.join(folder,name),follow_symlinks=False)
                    if stat.S_ISREG(info.st_mode):total+=info.st_size
                except FileNotFoundError:pass
    return total/2**30


def validate_config_access(config,policy,final):
    if config.get('job_kind','evaluation') not in JOB_MODULES:
        raise ValueError('Unreviewed job kind')
    workers=config.get('workers',1)
    if type(workers) is not int or not 1<=workers<=policy['execution']['cpu_workers_max']:
        raise ValueError('Worker count exceeds project policy')
    def inspect(value):
        if isinstance(value,dict):
            for key,item in value.items():
                if isinstance(item,str) and ('path' in key or 'pool' in key or 'bundle' in key):
                    parts=item.replace('\\','/').lower().split('/')
                    if not final and ('holdout' in parts or any('final_holdout' in p for p in parts)):
                        raise ValueError('Untouched final holdout cannot be used by a training/data job')
                inspect(item)
        elif isinstance(value,list):
            for item in value:inspect(item)
    inspect(config)
    for cell in config.get('matchups',[config]):
        for side in ('candidate','reference'):
            spec=cell.get(side)
            if isinstance(spec,dict) and 'holdout' in Path(spec.get('path','')).parts and not final:
                raise ValueError('Untouched final holdout cannot be used for development or teacher generation')


def snapshot(run):
    destination=run/'source'
    config=read_json(run/'config.json')
    maintenance=(config.get('job_kind') in ('server_artifact_archive','server_storage_migration','server_storage_acceptance')
                 or (config.get('job_kind')=='compact_audit' and config.get('audit_variant') in ('continuous_archive_inventory','continuous_league_readiness','continuous_result_audit','volume_source_audit','volume_results','volume_interrupted_full_state'))
                 or (config.get('job_kind')=='compact_audit' and config.get('audit_variant') in ('continuous_parent_readiness','actual_unique_PPO_volume','tr10_clean_checkpoint_recovery','tr07_failed_training_analysis','tr08_recovery_analysis','tr08_failed_training_analysis','tr09_clean_checkpoint_recovery','tr09_verified_migration_report','completed_hybrid_model_materialization','completed_hybrid_credit_analysis'))
                 or (config.get('job_kind')=='local_cache_retirement'
                 and config.get('retirement_variant') in ('public_download_prefixes','forensic_replicas','consumed_trajectory_gzip')))
    directories=('ptcg_lab','scripts','tests') if maintenance else ('ptcg_lab','scripts','tests','agents')
    for directory in directories:
        if not (ROOT/directory).exists():
            continue
        shutil.copytree(ROOT/directory,destination/directory,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    files={str(p.relative_to(destination)):sha256(p) for p in destination.rglob('*') if p.is_file()}
    atomic_json(run/'source-manifest.json',files)
    if maintenance:
        atomic_json(run/'source-snapshot-scope.json',{'complete_executed_controller_code_snapshot':True,
            'directories':list(directories),'unused_agent_model_assets_not_copied':True,
            'scope':'Archive, forensic analysis, exact replica maintenance or completed model byte reconstruction. Inputs are hash-bound retained originals; research/evaluation snapshots unchanged.'})
    return files


def verify_snapshot(run):
    expected=read_json(run/'source-manifest.json')
    actual={str(p.relative_to(run/'source')):sha256(p) for p in (run/'source').rglob('*')
            if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
    if actual!=expected:
        raise RuntimeError('Frozen code snapshot changed')
    if sha256(run/'config.json')!=read_json(run/'experiment.json')['config_sha256']:
        raise RuntimeError('Frozen config changed')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--label',default='bo3-smoke')
    parser.add_argument('--hypothesis',default='Official BO3 completes with valid actions and durable results')
    parser.add_argument('--wall-seconds',type=float,default=120)
    parser.add_argument('--stop-after-new-series',type=int)
    parser.add_argument('--research',action='store_true')
    parser.add_argument('--final-evaluation',action='store_true')
    args=parser.parse_args()
    state_path=ROOT/'research/autonomy/state.json'
    policy=read_json(ROOT/'research/autonomy/policy.json')
    lock=ROOT/'research/autonomy/job.lock'
    owner=acquire_lock(lock)
    process=None; job=None; run=None
    try:
        state=read_json(state_path)
        state['budget_used']['new_workspace_artifacts_gib']=artifact_gib()
        if state['budget_used']['new_workspace_artifacts_gib']>=policy['budget']['new_workspace_artifacts_gib_max']:
            raise RuntimeError('Artifact storage budget exhausted')
        if state.get('active_job'):
            previous=state['active_job']
            if is_same_process(previous['pid'],previous['created']):
                raise RuntimeError('Recorded child still alive')
            # Conservative recovery: reserve the unaccounted part of its authorized cap.
            elapsed=max(0,previous['limit_seconds']-previous.get('accounted_seconds',0))
            state['budget_used'][previous.get('compute_kind','cpu')+'_job_wall_hours']+=elapsed/3600
            state.setdefault('recovery_accounting',[]).append({'at':utcnow(),'previous':previous,'charged_extra_seconds':elapsed})
            state['active_job']=None
            sync_remote_intent(ROOT,state,previous['run_id'])
            atomic_json(state_path,state)
        elif state.get('last_run_id'):
            sync_remote_intent(ROOT,state,state['last_run_id'])
        if args.resume:
            run=args.resume.resolve()
            if run.parent!=ROOT/'runs':
                raise ValueError('Resume directory must be a direct child of workspace runs/')
            verify_snapshot(run)
            experiment=read_json(run/'experiment.json')
            config=read_json(run/'config.json')
            current_env=read_json(runtime_manifest(config))
            saved_env=read_json(run/'environment.json')
            if any(current_env[k]!=saved_env[k] for k in ('files','packages')):
                raise RuntimeError('Engine/environment manifest changed; create a new experiment')
            config=read_json(run/'config.json')
            validate_config_access(config,policy,args.final_evaluation)
            validate_execution_target(config,state)
            if collect_metrics(run,config).get('complete'):
                print(json.dumps({'run_id':run.name,'status':'already_complete','additional_compute_seconds':0}))
                return
            validate_cycle_open(state)
            validate_stage_job(ROOT,config,policy,state,experiment['kind']=='research',args.final_evaluation,new=False,run_id=run.name)
        else:
            validate_cycle_open(state)
            if not args.config:
                parser.error('--config required for a new run')
            config=read_json(args.config)
            validate_config_access(config,policy,args.final_evaluation)
            validate_execution_target(config,state)
            validate_plateau(config,policy,state,args.research,args.final_evaluation)
            validate_stage_job(ROOT,config,policy,state,args.research,args.final_evaluation)
            if args.research and not config.get('hypothesis_id'):
                raise ValueError('New research runs require a stable hypothesis_id for cumulative compute limits')
            if args.research and state['budget_used']['executed_research_experiments']>=policy['budget']['executed_research_experiments_max']:
                raise RuntimeError('Research experiment count exhausted')
            label=''.join(c if c.isalnum() or c=='-' else '-' for c in args.label)[:48]
            name=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+label+'_'+uuid.uuid4().hex[:6]
            run=ROOT/'runs'/name; run.mkdir()
            atomic_json(run/'config.json',read_json(args.config))
            config=read_json(args.config)
            source=snapshot(run)
            shutil.copy2(runtime_manifest(config),run/'environment.json')
            atomic_json(run/'policy.json',policy)
            from ptcg_lab.volume_scope import limits as phase_limits
            phase=phase_limits(ROOT,config,compute_kind(config),policy,state)
            hypothesis_cap=phase['hypothesis_seconds_max'] if phase else policy['execution']['hypothesis_compute_hours_max']*3600
            experiment={'run_id':name,'created_at':utcnow(),'kind':'research' if args.research else 'engineering',
                        'phase':state['current_phase'],'hypothesis':args.hypothesis,'hypothesis_id':config.get('hypothesis_id'),'compute_kind':compute_kind(config),
                        'primary_change':config.get('primary_change','official BO3 runtime harness'),
                        'success_criteria':config.get('success_criteria','All scheduled BO3 finish without invalid actions, errors or timeouts'),
                        'config_sha256':sha256(run/'config.json'),
                        'source_files':len(source),'parent_run_id':state.get('champion_run_id'),
                        'requested_wall_seconds':args.wall_seconds,'max_compute_seconds':hypothesis_cap,
                        'policy_version':policy['policy_version'],'policy_sha256':sha256(run/'policy.json')}
            if config.get('job_kind') in ('server_boundary_audit','server_environment','server_resource_audit','remote_training_check','remote_train_ppo','server_storage_migration','server_storage_acceptance') or (config.get('job_kind')=='compact_audit' and config.get('audit_variant')=='active_clock_acceptance'):
                experiment['remote_lease_epoch']=state.get('remote_lease_epoch',0)+1
                state['remote_lease_epoch']=experiment['remote_lease_epoch']
            atomic_json(run/'experiment.json',experiment)
            register_stage_run(config,policy,state,args.research,args.final_evaluation,run.name)
            if args.research:
                state['budget_used']['executed_research_experiments']+=1
                if config.get('low_cost_pivot'):
                    state['started_pivot_runs']=state.get('started_pivot_runs',state.get('completed_pivot_runs',0))+1
            # Persist reservations before launching; a crash must not reset run limits.
            atomic_json(state_path,state)
        kind=compute_kind(config);budget_key=kind+'_job_wall_hours'
        seconds=allowed_seconds(policy,state,args.final_evaluation,args.wall_seconds,kind,config=config)
        if seconds<=0:raise RuntimeError('No '+kind+' budget remaining')
        attempts=read_json(run/'attempts.json') if (run/'attempts.json').exists() else []
        if args.resume and config.get('job_kind')=='server_storage_acceptance' and attempts:
            from ptcg_lab.engineering_retry import admit_connection_resume
            same_deadline=admit_connection_resume(ROOT,run,config,attempts,state)
            seconds=min(seconds,same_deadline-time.time())
            if seconds<=60:raise RuntimeError('Original connection retry deadline exhausted')
        seconds=min(seconds,experiment['max_compute_seconds']-sum(charged_attempt_seconds(run.name,a,state) for a in attempts))
        from ptcg_lab.active_clock import validate_local as validate_active_clock,local_credit
        use_active_clock=validate_active_clock(ROOT,config,state)
        if use_active_clock:
            seconds=min(seconds,config['training_wall_seconds']-sum(charged_attempt_seconds(run.name,a,state) for a in attempts))
        if config.get('hypothesis_id'):
            from ptcg_lab.volume_scope import limits as phase_limits
            phase=phase_limits(ROOT,config,kind,policy,state)
            hypothesis_cap=phase['hypothesis_seconds_max'] if phase else policy['execution']['hypothesis_compute_hours_max']*3600
            seconds=min(seconds,hypothesis_cap-hypothesis_seconds(config,state))
        if config.get('TR11_attempt_authorization'):
            # New attempts share the original absolute deadline as well as
            # cumulative charges; a new wrapper cannot restart the clock.
            seconds=min(seconds,config['TR11_original_remote_deadline_unix']-time.time())
        if seconds<=0:
            raise RuntimeError('Hypothesis budget exhausted')
        if shutil.disk_usage(ROOT).free<policy['budget']['disk_free_gib_min']*2**30:
            raise RuntimeError('Insufficient free disk reserve')
        attempt_id=len(attempts)+1
        from ptcg_lab.volume_scope import scoped
        if scoped(config) and attempt_id>config['maximum_attempts']:
            raise RuntimeError('TR12 registered single-raw attempt cap exhausted')
        module=JOB_MODULES[config.get('job_kind','evaluation')]
        interpreter=ROOT/('.venv-gpu' if kind=='gpu' else '.venv')/'Scripts/python.exe'
        command=[str(interpreter),'-m',module,'--config',str(run/'config.json'),'--output',str(run)]
        if args.stop_after_new_series:
            command+=['--stop-after-new-series',str(args.stop_after_new_series)]
        atomic_json(run/f'command-{attempt_id:02d}.json',{'argv':command,'cwd':str(run/'source')})
        state.update(status='running',active_run_id=run.name,updated_at=utcnow())
        atomic_json(state_path,state)
        base_hours=state['budget_used'][budget_key]
        start=time.monotonic(); reason='completed'; storage_checked=start
        print(json.dumps({'run_id':run.name,'limit_seconds':seconds,'attempt':attempt_id}),flush=True)
        with (run/f'stdout-{attempt_id:02d}.log').open('w',encoding='utf-8') as stdout, (run/f'stderr-{attempt_id:02d}.log').open('w',encoding='utf-8') as stderr:
            job=KillOnCloseJob()
            wall_ceiling=86400 if use_active_clock else seconds
            started_unix=time.time()
            job_env=dict(os.environ,PTCG_JOB_DEADLINE_UNIX=str(started_unix+wall_ceiling),PTCG_JOB_RUN_ID=run.name)
            if use_active_clock:
                job_env.update(PTCG_ACTIVE_STARTED_UNIX=str(started_unix),PTCG_ACTIVE_LIMIT_SECONDS=str(seconds),PTCG_JOB_ATTEMPT=str(attempt_id))
            process=subprocess.Popen(command,cwd=run/'source',stdout=stdout,stderr=stderr,env=job_env)
            try:
                job.assign(process.pid)
                child={'pid':process.pid,'created':psutil.Process(process.pid).create_time()}
                owner['child']=child; atomic_json(lock,owner)
                state['active_job']={**child,'run_id':run.name,'limit_seconds':seconds,'accounted_seconds':0,'attempt':attempt_id,'compute_kind':kind}
                if use_active_clock:state['active_job'].update(time_basis='execution_excluding_proven_shared_wait',wall_ceiling_seconds=wall_ceiling)
                atomic_json(state_path,state)
                while process.poll() is None:
                    sync_remote_intent(ROOT,state,run.name)
                    elapsed=time.monotonic()-start
                    credit=local_credit(run,run.name,attempt_id,elapsed) if use_active_clock else 0.
                    charged=elapsed-credit
                    if charged>=seconds or elapsed>=wall_ceiling:
                        reason='wall_timeout'; job.close()
                        if process.poll() is None: process.kill()
                        break
                    if time.monotonic()-storage_checked>=30:
                        storage_checked=time.monotonic()
                        state['budget_used']['new_workspace_artifacts_gib']=artifact_gib()
                        if (state['budget_used']['new_workspace_artifacts_gib']>=policy['budget']['new_workspace_artifacts_gib_max']
                                or shutil.disk_usage(ROOT).free<policy['budget']['disk_free_gib_min']*2**30):
                            reason='storage_limit'; job.close()
                            if process.poll() is None: process.kill()
                            break
                    try:
                        parent=psutil.Process(process.pid)
                        rss=sum(p.memory_info().rss for p in [parent]+parent.children(recursive=True) if p.is_running())
                    except psutil.NoSuchProcess:
                        rss=0
                    if rss>policy['budget']['ram_working_target_gib']*2**30:
                        reason='memory_limit'; job.close()
                        if process.poll() is None: process.kill()
                        break
                    state['budget_used'][budget_key]=base_hours+charged/3600
                    state['active_job']['accounted_seconds']=charged
                    if use_active_clock:state['active_job'].update(gross_wall_seconds=elapsed,verified_shared_wait_seconds=credit)
                    state['updated_at']=utcnow()
                    atomic_json(state_path,state)
                    time.sleep(.25)
                process.wait(timeout=10)
            finally:
                job.close()
                if process.poll() is None:
                    process.kill(); process.wait(timeout=10)
        elapsed=time.monotonic()-start
        credit=local_credit(run,run.name,attempt_id,elapsed) if use_active_clock else 0.
        charged=elapsed-credit
        # A hard stop may leave metrics.json stale. Recompute from durable series rows.
        metrics=collect_metrics(run,config)
        sync_remote_intent(ROOT,state,run.name)
        if process.returncode and reason=='completed': reason='failed'
        if reason=='completed' and not metrics.get('complete'): reason='checkpointed'
        attempts.append({'attempt':attempt_id,'at':utcnow(),'seconds':elapsed,'exit_code':process.returncode,'status':reason,'compute_kind':kind})
        if use_active_clock:attempts[-1].update(counted_execution_seconds=charged,verified_shared_wait_seconds=credit,
            time_basis='execution_excluding_proven_shared_wait',fixed_hard_safety_window_seconds=wall_ceiling)
        atomic_json(run/'attempts.json',attempts)
        state['budget_used'][budget_key]=base_hours+charged/3600
        state['budget_used']['new_workspace_artifacts_gib']=artifact_gib()
        state.update(active_run_id=None,active_job=None,updated_at=utcnow(),last_completed_action=f'{run.name}: {reason}',last_run_id=run.name)
        if reason=='completed':
            key='completed_research_runs' if experiment['kind']=='research' else 'completed_engineering_runs'
            if run.name not in state.setdefault(key,[]): state[key].append(run.name)
        atomic_json(state_path,state)
        print(json.dumps({'run_id':run.name,'status':reason,'seconds':elapsed,'metrics':metrics}),flush=True)
        if reason=='failed': raise SystemExit(process.returncode or 1)
    finally:
        if job: job.close()
        if process and process.poll() is None:
            process.kill(); process.wait(timeout=10)
        if run and (run/'remote-job.json').exists():
            latest=read_json(state_path);sync_remote_intent(ROOT,latest,run.name);atomic_json(state_path,latest)
        if lock.exists() and read_json(lock).get('pid')==os.getpid():
            lock.unlink()


if __name__=='__main__':
    main()
