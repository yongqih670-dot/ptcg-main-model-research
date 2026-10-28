"""Controlled basic Linux acceptance and bounded engineering PPO trial."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import time

from .io import atomic_json,read_json,sha256,utcnow
from .remote_lease import validate_no_unreconciled_remote
from .server_environment import durable_bytes
from .ssh_stream import SSHStream,open_readonly_stream
from .io_pacing import BytePacer
from . import phase_timing


def remote_source_files(source):
    # Windows-only cache maintenance and the local Linux VM are never server jobs.
    excluded={'active_clock_acceptance.py','linux_probe.py','local_cache_retirement.py','local_cache_finalize.py','windows_archive_check.py',
              'local_fragment_retirement.py','local_forensic_retirement.py','evidence_tiering.py',
              'server_archive.py','server_trial.py','server_environment.py',
              'server_storage_migration.py','server_storage_acceptance.py','storage_migration_agent.py',
              'storage_copy_worker.py','storage_acceptance_worker.py','engineering_retry.py',
              'training_failure_analysis.py','tr05_failure_analysis.py','tr07_failure_analysis.py',
              'tr08_recovery_analysis.py','tr08_failure_analysis.py','tr09_recovery_analysis.py','tr10_recovery_analysis.py',
              'incremental_hash_cache.py','ppo_counter_compatibility.py','ppo_volume_audit.py','local_trajectory_archive.py','terminal_cache_plans.py','continuous_parent.py','prepared_parent_verify.py','continuous_archive_inventory.py','continuous_archive_accept.py','archive_transfer_measurement.py','continuous_league_readiness.py',
              'continuous_result_audit.py','continuous_diagnostic_stats.py','volume_source_audit.py','volume_boundary.py','volume_results.py','volume_statistics.py','volume_interrupted_audit.py'}
    return {p.relative_to(source).as_posix():p.read_bytes() for p in (source/'ptcg_lab').rglob('*.py')
            if '__pycache__' not in p.parts and p.name not in excluded}


def trial_source_files(source,project,config):
    files=remote_source_files(source)
    if config.get('check_variant') in ('continuous_mainline','volume_extension'):
        # These stand-alone engineering workers are never imported or executed
        # by the derived TR11 payload. Keep its control frame below1MiB.
        for name in ['boundary_worker.py','server_resource_worker.py','server_environment_worker.py']:
            files.pop('ptcg_lab/'+name,None)
    if config.get('check_variant')=='volume_extension':
        # This admission gate runs in the local controller only. No server
        # payload imports it; removing its duplicate leaves all checks active.
        files.pop('ptcg_lab/shared_server_safety.py',None)
    if config.get('check_variant')=='continuous_archive_reuse':
        files['ptcg_lab/continuous_archive_accept.py']=(source/'ptcg_lab/continuous_archive_accept.py').read_bytes()
    if config.get('check_variant')=='continuous_prepared_verify':
        files['ptcg_lab/prepared_parent_verify.py']=(source/'ptcg_lab/prepared_parent_verify.py').read_bytes()
    overrides=config.get('prototype_source_overrides',[])
    if config.get('check_variant')=='cpu_sampler' and not overrides:
        raise ValueError('CPU acceptance requires its exact reviewed prototype sources')
    if overrides:
        if config.get('check_variant')!='cpu_sampler' or config.get('engineering_only') is not True:
            raise ValueError('Prototype override is limited to CPU-only engineering acceptance')
        allowed={'ptcg_lab/training_control.py','ptcg_lab/rl_sampling.py','ptcg_lab/rl_league.py',
                 'ptcg_lab/rl_collector.py','ptcg_lab/rl_resume.py'}
        if {r['target'] for r in overrides}!=allowed or len(overrides)!=len(allowed):
            raise ValueError('CPU acceptance prototype targets differ')
        for row in overrides:
            path=(Path(project)/row['path']).resolve()
            if Path(project).resolve() not in path.parents or path.is_symlink() or sha256(path)!=row['sha256']:
                raise ValueError('CPU acceptance prototype source changed')
            files[row['target']]=path.read_bytes()
    return files


def validate_restore_repair(project,config,deadline,now=None):
    now=time.time() if now is None else now
    parents=config.get('compact_restore_repair_parent_run_ids',[])
    if not parents:return
    spent=sum(a['seconds'] for run_id in parents for a in read_json(Path(project)/'runs'/run_id/'attempts.json'))
    if abs(spent-config['prior_compact_restore_seconds'])>.01 or spent+deadline-now>3600.5:
        raise RuntimeError('Compact restore repair cannot regain original scope')


def input_objects(project,source,config):
    originals={};mapping=[]
    for field in ('training_config_template','audit_config_template'):
        if config.get(field+'_sha256') and sha256(project/config[field])!=config[field+'_sha256']:
            raise ValueError('Input configuration template changed')
    def add(path,relative):
        path=Path(path);digest=sha256(path);originals.setdefault(digest,path)
        mapping.append({'path':relative.replace('\\','/'),'sha256':digest,'bytes':path.stat().st_size})
    training=read_json(project/config['training_config_template'])
    if config.get('protected_actor_source_manifest_path'):
        path=project/config['protected_actor_source_manifest_path']
        if sha256(path)!=config['protected_actor_source_manifest_sha256']:raise ValueError('Original actor source manifest changed')
        protected={key.replace('\\','/'):value for key,value in read_json(path).items()}
        prefix=training['bundle_path']+'/'
        for relative,digest in protected.items():
            if relative.startswith(prefix) and sha256(source/relative)!=digest:
                raise ValueError('Full inherited actor source differs from original frozen TR04 input')
    for spec in training['league']['fixed_opponents']:
        for name,digest in spec['files'].items():
            if sha256(source/spec['path']/name.replace('\\','/'))!=digest:raise ValueError('Training opponent input changed')
    folders={training['bundle_path']}|{x['path'] for x in training['league']['fixed_opponents']}
    for folder in sorted(folders):
        base=source/folder
        if source not in base.resolve().parents or 'holdout' in folder.lower():raise ValueError('Training input scope escaped')
        for path in sorted(base.rglob('*')):
            if path.is_file() and not path.is_symlink() and '__pycache__' not in path.parts and path.suffix!='.pyc':
                add(path,path.relative_to(source).as_posix())
    audit=read_json(project/config['audit_config_template'])
    add(source/'ptcg_lab/public_cg/metadata.json','ptcg_lab/public_cg/metadata.json')
    replay=project/audit['replay_path']
    if sha256(replay)!=audit['replay_sha256']:raise ValueError('Basic acceptance replay changed')
    add(replay,'inputs/parity-replay.json.gz')
    return originals,mapping,training,audit


def fresh_manifest(project,source,config):
    _,mapping,training,_=input_objects(project,source,config)
    reference=read_json(project/config['matched_reference_path'])
    if sha256(project/config['matched_reference_path'])!=config['matched_reference_sha256']:raise ValueError('Matched scientific reference file changed')
    from .archive_prefix import canonical_hash
    return {'schema_version':1,'mode':'fresh_scientific','files':[],'file_map':mapping,'training_config':training,
        'input_capsule':config['input_capsule'],'matched_training_reference':reference,
        'matched_reference_data_sha256':canonical_hash(reference),'online_archive_enabled':True,
        'archive_storage_codec':'gzip','cooperative_resource_control':True,'cooperative_pause_probe_required':True,
        'reserved_future_bytes':16*2**20,'post_retirement_forecast_bytes':config['post_retirement_forecast_bytes'],
        'terminal_cache_retirement':config['terminal_cache_retirement'],'engineering_only':False,
        **({'sampling_device':'cpu','CPU_sampler_acceptance_sha256':config['CPU_sampler_acceptance']['sha256']}
           if config.get('sampling_device')=='cpu' else {}),
        **({'consumed_reference_barrier_required':True,'ACK_lag_barrier_acceptance_sha256':config['ACK_lag_barrier_acceptance']['sha256']}
           if config.get('consumed_reference_barrier_required') else {})}


def bind_clean_recovery_inputs(project,config,manifest,originals):
    """One exact supplemental fixture in the new job's stage, outside file_map."""
    if (config.get('hypothesis_id'),config.get('check_variant'))==('TR10','derived_scientific'):
        from .ppo_accounting import ACCOUNTING
        if (not config.get('consumed_reference_barrier_required') or not config.get('source_clean_checkpoint_recovery')
                or config.get('actual_decision_accounting')!=ACCOUNTING or config.get('numeric_reference')):
            raise ValueError('TR10 recovery uses its real pending tail, unchanged ACK barrier and actual-count correction')
        manifest.update(source_clean_checkpoint_recovery=config['source_clean_checkpoint_recovery'],
            consumed_reference_barrier_required=True,ACK_lag_barrier_acceptance_sha256=config['ACK_lag_barrier_acceptance']['sha256'],
            actual_decision_accounting=ACCOUNTING)
        return
    if (config.get('hypothesis_id'),config.get('check_variant'))!=('TR09','derived_scientific'):
        raise ValueError('Supplemental numeric reference only belongs to original second TR09')
    if not config.get('consumed_reference_barrier_required') or not config.get('source_clean_checkpoint_recovery'):
        raise ValueError('Clean reference cannot disable the original archive barrier')
    numeric=config['numeric_reference'];path=(project/numeric['path']).resolve()
    if (project.resolve() not in path.parents or path.name!='numeric-reference.pt'
            or type(numeric['bytes']) is not int or not 0<numeric['bytes']<=64*2**20
            or path.stat().st_size!=numeric['bytes'] or sha256(path)!=numeric['sha256']):
        raise ValueError('Exact local recovery-only reference identity differs')
    manifest.update(source_clean_checkpoint_recovery=config['source_clean_checkpoint_recovery'],
        consumed_reference_barrier_required=True,ACK_lag_barrier_acceptance_sha256=config['ACK_lag_barrier_acceptance']['sha256'])
    originals[numeric['sha256']]=path
    manifest['numeric_reference']={k:numeric[k] for k in ['sha256','bytes']}
    manifest['files']=[{'file':numeric['sha256']+'.blob','sha256':numeric['sha256'],'bytes':numeric['bytes'],'expanded_bytes':0}]


def trial_outcome(mode,metrics,props):
    """A published scientific checkpoint is incomplete, with a clean exit."""
    if (not metrics or props.get('LoadState')!='loaded' or props.get('Result')!='success'
            or props.get('ExecMainStatus')!='0' or props.get('ControlGroup')!=''):
        raise RuntimeError('Actual server check/trial failed; inspect archived evidence')
    if metrics.get('complete') is True:return 'complete'
    phases={'checkpointed_for_control','checkpointed_at_soft_deadline','checkpointed_before_update_at_soft_deadline',
            'partial_update_abandoned_at_deadline','partial_update_abandoned_for_control'}
    progress=metrics.get('last_training_progress',{});cfg=metrics.get('actual_config',{})
    target=cfg.get('target_decisions')
    trained=progress.get('unique_PPO_trained_decisions',progress.get('trained_decisions'));pending=progress.get('pending_decisions')
    if (mode in ('fresh_scientific','derived_scientific','scientific_continue') and metrics.get('complete') is False
            and metrics.get('training_only') is True and metrics.get('engineering_only') is False
            and progress.get('stage') in phases and type(target) is int and type(trained) is int
            and type(pending) is int and 0<=trained<target and pending>=0):
        return 'checkpointed'
    raise RuntimeError('Actual server check/trial failed; inspect archived evidence')


def run(config,output):
    output=Path(output).resolve();project=Path(config['project_path']);source=Path(__file__).resolve().parents[1]
    if os.environ.get('PTCG_JOB_RUN_ID')!=output.name or (output/'remote-job.json').exists():
        raise RuntimeError('Trial requires a fresh controlled runner and a reconciled lease')
    mode='cpu_sampler_check' if config.get('check_variant')=='cpu_sampler' else 'journal_io_check' if config.get('check_variant')=='journal_io' else 'derived_scientific' if config.get('check_variant') in ('derived_scientific','continuous_mainline','volume_extension') else 'fresh_scientific' if config.get('check_variant')=='fresh_scientific' else 'scientific_continue' if config.get('check_variant')=='scientific_continue' else 'live_archive_check' if config.get('check_variant') in ('live_archive','continuous_archive_reuse') else 'gpu_existing_restore' if config.get('check_variant') in ('gpu_existing','continuous_prepared_verify') else 'compact_cpu_existing' if config.get('check_variant')=='compact_existing' else 'compact_cpu_restore' if config.get('check_variant')=='compact_restore' else 'basic_cpu_check' if config['job_kind']=='remote_training_check' else 'short_gpu_trial'
    deadline=float(os.environ['PTCG_JOB_DEADLINE_UNIX'])
    authorized_numeric=config.get('check_variant')=='continuous_prepared_verify' and config.get('engineering_window_authorization') is not None
    from .continuous_runtime import local_limit
    authorized_training_seconds=local_limit(project,config)
    authorized_TR12=config.get('check_variant')=='volume_extension'
    from .active_clock import validate_local as validate_active_clock
    active_clock=validate_active_clock(project,config)
    maximum=600.5 if authorized_numeric else 7200.5 if mode in ('fresh_scientific','derived_scientific') else 3600.5 if mode=='scientific_continue' else 300.5 if mode=='gpu_existing_restore' else 1200.5 if mode=='compact_cpu_existing' else 3600.5 if mode=='compact_cpu_restore' else 900.5 if mode in ('basic_cpu_check','live_archive_check','journal_io_check','cpu_sampler_check') else 600.5
    if authorized_training_seconds:maximum=authorized_training_seconds+.5
    if authorized_TR12:maximum=(1800 if config.get('parent_acceptance_only') else 28800)+.5
    if active_clock:maximum=86400.5
    if deadline-time.time()>maximum:raise RuntimeError('Staged job wall limit exceeded')
    if config.get('check_variant')=='continuous_archive_reuse':
        from .continuous_archive_accept import validate_local
        validate_local(project,config)
    elif mode=='cpu_sampler_check':
        from .cpu_sampling_scope import validate
        validate(project,config,deadline)
    if config.get('pre_ssh_startup_failure_run_id'):
        from .shared_server_safety import validate_pre_ssh_startup_failure
        spent=validate_pre_ssh_startup_failure(project,config)
        if spent+deadline-time.time()>14400.5:raise RuntimeError('Original TR06 total training scope exhausted')
    if mode=='compact_cpu_restore':validate_restore_repair(project,config,deadline)
    if mode=='live_archive_check' and config.get('live_archive_repair_parent_run_ids'):
        spent=sum(a['seconds'] for run in config['live_archive_repair_parent_run_ids'] for a in read_json(project/'runs'/run/'attempts.json'))
        if abs(spent-config['prior_live_archive_seconds'])>.01 or spent+deadline-time.time()>900.5:
            raise RuntimeError('Live archive repair cannot regain original900s scope')
    if mode=='basic_cpu_check' and config.get('basic_repair_parent_run_ids'):
        spent=sum(a['seconds'] for run_id in config['basic_repair_parent_run_ids'] for a in read_json(project/'runs'/run_id/'attempts.json'))
        if abs(spent-config['prior_basic_scope_seconds'])>.01 or spent+deadline-time.time()>900.5:
            raise RuntimeError('CPU basic repair cannot regain already spent scope')
    originals={}
    if config.get('check_variant')=='continuous_archive_reuse':
        manifest={'schema_version':1,'mode':mode,'files':[],'file_map':[],
            'continuous_archive_reuse':True,'engineering_only':True,'model_selection_forbidden':True,
            'no_training_updates':True,'online_archive_enabled':True,'archive_storage_codec':'gzip',
            **{k:config[k] for k in ['existing_restore_source','prior_numeric_source','prior_numeric_restore_sha256',
                'numeric_source_hashes','expected_parent_objects']}}
    elif mode=='cpu_sampler_check':
        _,mapping,training,_=input_objects(project,source,config)
        manifest={'schema_version':1,'mode':mode,'files':[],'file_map':mapping,'training_config':training,
            'input_capsule':config['input_capsule'],'existing_restore_source':config['existing_restore_source'],
            'source_league_sha256':config['source_league_sha256'],'no_training_updates':True,
            'source_postcommit_recovery':config.get('source_postcommit_recovery'),
            'engineering_only':True,'model_selection_forbidden':True,'reserved_future_bytes':16*2**20,
            'public_fragment_retirement':config.get('public_fragment_retirement',[]),
            'terminal_cache_retirement':config.get('terminal_cache_retirement',[]),
            'source_consumed_cache_retirement_acks':config.get('source_consumed_cache_retirement_acks',[]),
            'post_retirement_forecast_bytes':256*2**20,
            'online_archive_enabled':True,'archive_storage_codec':'gzip'}
    elif mode=='journal_io_check':
        manifest={'schema_version':1,'mode':mode,'files':[],'file_map':[],
            'existing_restore_source':config['existing_restore_source'],'engineering_only':True,
            'model_selection_forbidden':True,'no_training_updates':True,'reserved_future_bytes':256*2**20}
    elif mode=='derived_scientific':
        if authorized_TR12:
            from .volume_extension import validate_local
            validate_local(project,config,deadline,current_run_id=output.name)
        elif config.get('check_variant')=='continuous_mainline':
            from .continuous_mainline import validate_local
            validate_local(project,config,deadline,current_run_id=output.name)
        else:
            from .scientific_resume import validate_scope
            validate_scope(project,config,deadline,current_run_id=output.name)
        _,mapping,training,_=input_objects(project,source,config)
        manifest={'schema_version':1,'mode':mode,'files':[],'file_map':mapping,'training_config':training,
            'input_capsule':config['input_capsule'],'existing_restore_source':config['existing_restore_source'],
            'source_league_sha256':config['source_league_sha256'],'source_inflight':config.get('source_inflight'),
            'source_postcommit_recovery':config.get('source_postcommit_recovery'),
            'online_archive_enabled':True,'archive_storage_codec':'gzip','cooperative_resource_control':True,
            'reserved_future_bytes':config['reserved_future_bytes'],'engineering_only':False}
        if config.get('source_transport_recovery'):
            manifest.update(source_transport_recovery=config['source_transport_recovery'],additional_complete_tail=config['additional_complete_tail'])
        if config.get('terminal_cache_retirement'):
            manifest.update(terminal_cache_retirement=config['terminal_cache_retirement'],post_retirement_forecast_bytes=config['post_retirement_forecast_bytes'])
        if config.get('sampling_device')=='cpu':
            manifest.update(sampling_device='cpu',CPU_sampler_acceptance_sha256=config['CPU_sampler_acceptance']['sha256'])
            if config.get('CPU_backend_source'):
                manifest.update(CPU_backend_source=config['CPU_backend_source'],CPU_backend_fixture_sha256=config['CPU_backend_fixture_sha256'])
        if config.get('check_variant') in ('continuous_mainline','volume_extension'):
            numeric=config['numeric_reference'];path=project/numeric['path']
            if path.stat().st_size!=numeric['bytes'] or sha256(path)!=numeric['sha256']:
                raise ValueError('New diagnostic recovery-only numerical input changed')
            originals[numeric['sha256']]=path
            manifest.update(new_diagnostic_identity='TR12' if authorized_TR12 else 'TR11',parent_acceptance_only=config.get('parent_acceptance_only') is True,
                actual_decision_accounting=config['actual_decision_accounting'],
                numeric_reference={k:numeric[k] for k in ['sha256','bytes']},
                consumed_reference_barrier_required=True,
                ACK_lag_barrier_acceptance_sha256=config['ACK_lag_barrier_acceptance']['sha256'])
            manifest['files']=[{'file':numeric['sha256']+'.blob','sha256':numeric['sha256'],'bytes':numeric['bytes'],'expanded_bytes':0}]
            if authorized_TR12:
                manifest.update(source_point=config['source_point'],source_remote_evidence=config['source_remote_evidence'],
                    engineering_only=config.get('parent_acceptance_only') is True,
                    complete_postcheckpoint_tail=config.get('complete_postcheckpoint_tail'),phase_timing=config.get('phase_timing'))
        elif config.get('source_clean_checkpoint_recovery'):
            bind_clean_recovery_inputs(project,config,manifest,originals)
    elif mode=='fresh_scientific':
        manifest=fresh_manifest(project,source,config)
    elif mode=='scientific_continue':
        from .hybrid_continue import validate_original_attempt_scope
        validate_original_attempt_scope(project,config,deadline)
        originals,mapping,training,audit=input_objects(project,source,config)
        manifest={'schema_version':1,'mode':mode,'files':[],'file_map':mapping,'training_config':training,
            'input_capsule':config['input_capsule'],'existing_restore_source':config['existing_restore_source'],
            'continuation_point':config['continuation_point'],'online_archive_enabled':True,'archive_storage_codec':'gzip',
            'reserved_future_bytes':config['reserved_future_bytes'],
            'post_cache_retirement_forecast_bytes':config['post_cache_retirement_forecast_bytes'],
            'redundant_transport_cache':config['redundant_transport_cache'],
            'retired_fixture_checkpoint_caches':config['retired_fixture_checkpoint_caches'],
            'engineering_only':False,'model_selection_forbidden':False}
    elif mode in ('compact_cpu_existing','gpu_existing_restore','live_archive_check'):
        manifest={'schema_version':1,'mode':mode,'files':[],'file_map':[],
            'existing_restore_source':config['existing_restore_source'],'engineering_only':True,'model_selection_forbidden':True}
        if mode=='gpu_existing_restore':
            manifest['original_checkpoint_sha256']=config['original_checkpoint_sha256']
            if config.get('gradient_diagnostic'):manifest.update(gradient_diagnostic=True,cooperative_resource_control=True)
            if config.get('check_variant')=='continuous_prepared_verify':
                from .prepared_parent_verify import validate_local
                validate_local(project,config)
                manifest.update(prepared_parent_verify=True,prepared_payload_config_sha256=config['prepared_payload_config_sha256'],
                    online_archive_enabled=True,archive_storage_codec='gzip',cooperative_resource_control=True)
                if authorized_numeric:
                    manifest['authorized_numeric_window']={'seconds_max':600,'maximum_new_launches':1,
                        'authorization_sha256':config['engineering_window_authorization']['sha256']}
        if mode=='live_archive_check':manifest.update(archive_fixture=config['archive_fixture'],online_archive_enabled=True,archive_storage_codec='gzip')
        if mode=='live_archive_check' and config.get('ack_lag_barrier_check'):
            manifest.update(ack_lag_barrier_check=True,no_training_updates=True,reserved_future_bytes=16*2**20,
                post_retirement_forecast_bytes=384*2**20,terminal_cache_retirement=config['terminal_cache_retirement'])
    elif mode in ('basic_cpu_check','compact_cpu_restore'):
        originals,mapping,training,audit=input_objects(project,source,config)
        manifest={'schema_version':1,'mode':mode,'file_map':mapping,'training_config':training,'audit_config':audit,
            'files':[{'file':digest+'.blob','sha256':digest,'bytes':path.stat().st_size,'expanded_bytes':0} for digest,path in originals.items()]}
        if config.get('input_capsule_run_id'):
            manifest['input_capsule']={'run_id':config['input_capsule_run_id'],'manifest_sha256':config['input_capsule_manifest_sha256'],
                                      'source_inputs_only_not_an_acceptance_claim':True}
            manifest['files']=[]
        if mode=='compact_cpu_restore':
            capsule_path=project/config['restore_capsule_path']
            if sha256(capsule_path)!=config['restore_capsule_sha256']:raise ValueError('Frozen restore capsule changed')
            capsule=read_json(capsule_path);manifest['restore_capsule']=capsule
            for item in capsule['files']:
                path=(project/item['local_path']).resolve()
                if project.resolve() not in path.parents or path.stat().st_size!=item['bytes'] or sha256(path)!=item['sha256']:
                    raise ValueError('Lossless restore transport object changed')
                originals[item['sha256']]=path
            unique={x['sha256']:x for x in capsule['files']}
            manifest['files']=[{'file':digest+'.blob','sha256':digest,'bytes':item['bytes'],'expanded_bytes':0} for digest,item in unique.items()]
            # Full expanded members plus three checkpoint objects/hot replacement
            # and bounded-write admission headroom; no short-trial forecast reuse.
            from .restore_capsule import restore_capacity
            manifest['reserved_future_bytes']=restore_capacity(capsule)
    else:
        parent=project/'runs'/config['basic_check_run_id'];metrics=read_json(parent/'metrics.json')
        if not metrics.get('basic_cpu_correctness_passed') or sha256(parent/'remote-training-manifest.json')!=config['basic_manifest_sha256']:
            raise ValueError('Actual basic CPU acceptance is missing or changed')
        manifest={'schema_version':1,'mode':mode,'files':[],'parent_capsule':{'run_id':parent.name,
            'manifest_sha256':config['basic_manifest_sha256'],'local_basic_metrics_sha256':sha256(parent/'metrics.json'),
            'remote_basic_metrics_sha256':sha256(parent/'remote-results/trial-metrics.json')},
            'basic_fixture_sha256':metrics['actor_fixture_sha256'],'parent_model_sha256':metrics['parent_model_sha256'],
            'engineering_only':True,'model_selection_forbidden':True}
    from .resource_profile import bind_manifest
    manifest=bind_manifest(project,config,manifest)
    if authorized_TR12:
        manifest['TR12_runtime_scope']={'seconds_max':28800,'CPU_quota_cores':3,
            'authorization_sha256':config['TR12_authorization']['sha256'],'absolute_unique_target':369941,
            'phase_GPU_hours_max':84,'phase_CPU_hours_max':12}
        if active_clock:
            start_unix=float(os.environ['PTCG_ACTIVE_STARTED_UNIX'])
            manifest['active_clock_scope']={'spec':config['active_clock'],
                'execution_limit_seconds':float(os.environ['PTCG_ACTIVE_LIMIT_SECONDS']),
                'job_started_unix':start_unix,'hard_deadline_unix':deadline-20,
                'hard_window_seconds':deadline-20-start_unix,'startup_wait_seconds':0}
            from .active_clock import validate_scope
            validate_scope(manifest['active_clock_scope'])
    if authorized_training_seconds:
        manifest['TR11_runtime_scope']={'seconds_max':authorized_training_seconds,'CPU_quota_cores':3,
            'authorization_sha256':config['TR11_runtime_authorization']['sha256']}
    if config.get('TR11_attempt_authorization'):
        manifest['TR11_attempt_scope']={'attempt_index':config['training_attempt_index'],'attempts_max':3,
            'authorization_sha256':config['TR11_attempt_authorization']['sha256'],
            'prior_training_runs':config['TR11_prior_training_runs'],
            'original_remote_deadline_unix':config['TR11_original_remote_deadline_unix']}
    manifest_path=output/'remote-training-manifest.json';atomic_json(manifest_path,manifest)
    existing_archive_sources={}
    for item in config.get('local_archive_sources',[]):
        if item.get('storage')=='gzip_ack':
            from .archive_compressed import verify_archive
            old_root=(project/item['archive_root']).resolve()
            if project.resolve() not in old_root.parents:raise ValueError('Compressed source escaped project')
            ack=item['ack'];verify_archive(old_root,ack)
            if ack['sha256']!=item['sha256']:raise ValueError('Compressed source identity differs')
            existing_archive_sources[item['sha256']]={'archive_root':old_root,'ack':ack}
            continue
        path=(project/item['path']).resolve()
        if project.resolve() not in path.parents or path.is_symlink():raise ValueError('Local raw archive source escaped project')
        previous=existing_archive_sources.get(item['sha256'])
        if previous is not None and previous!=path:raise ValueError('Conflicting local archive source identity')
        existing_archive_sources[item['sha256']]=path
    package=source/'ptcg_lab';files=trial_source_files(source,project,config)
    wrapper=b'from ptcg_lab.server_trial_worker import main\nmain()\n'
    files.update({'training_check.py':wrapper,'training_worker.py':wrapper})
    payload={name:{'base64':base64.b64encode(data).decode(),'sha256':hashlib.sha256(data).hexdigest()} for name,data in files.items()}
    epoch=read_json(output/'experiment.json')['remote_lease_epoch'];unit='ptcg-hybrid-trial-'+output.name.rsplit('_',1)[1]+'-'+str(epoch)+'.service'
    lease={'run_id':output.name,'attempt':1,'lease_epoch':epoch,'host':config['host'],'unit':unit,'deadline_unix':deadline-20,'identity':None}
    if active_clock:lease['attempt']=int(os.environ['PTCG_JOB_ATTEMPT'])
    atomic_json(output/'remote-source.json',{k:v['sha256'] for k,v in payload.items()})
    stream=open_readonly_stream(config,package,output,deadline)
    if manifest.get('continuous_archive_reuse'):
        from .archive_transfer_measurement import MeasuredStream
        stream=MeasuredStream(stream)
    observation=None;failure=None
    wait_seen=0.;last_clock_poll=time.monotonic()
    def record_wait(observed):
        nonlocal wait_seen,last_clock_poll
        if not active_clock:return
        from .active_clock import finite
        remote=observed.get('active-clock-server.json')
        if remote:
            if (remote['run_id'],remote['lease_epoch'])!=(output.name,epoch) or remote.get('failure'):
                raise ValueError('Remote active clock identity/health differs')
            value=finite(remote['verified_wait_seconds'])
        else:value=finite(observed.get('startup_wait_seconds',wait_seen))
        if value<wait_seen:raise ValueError('Proven wait regressed')
        wait_seen=value
        last_clock_poll=time.monotonic()
        atomic_json(output/f"active-clock-local-{lease['attempt']:02d}.json",{
            'run_id':output.name,'attempt':lease['attempt'],'lease_epoch':epoch,
            'verified_wait_seconds':value,'remote_clock':remote,'at':utcnow()})
    transfer_pacer=BytePacer(4*2**20,deadline-60)
    def check():
        if time.time()>deadline-25:raise TimeoutError('Trial controller soft deadline; reconcile same unit')
    def progress(stage,**extra):
        if active_clock and stage=='archive_receiving' and time.monotonic()-last_clock_poll>=5:
            # Large checkpoint transfers must not hide ongoing resource waits
            # from the canonical runner's active-time guard.
            record_wait(stream.request('status'))
        phase_timing.publish(stage=stage)
        atomic_json(output/'progress.json',{'at':utcnow(),'stage':stage,'mode':mode,**extra})
    def archive_small(relative,local_name,maximum=512*1024,optional=False):
        offset=0;parts=[]
        while True:
            result=stream.request('read_file',{'path':relative,'offset':offset,'count':256*1024,'optional':optional})
            if result.get('exists') is False:return None
            if result['size']>maximum:raise ValueError('Metadata archive too large')
            data=base64.b64decode(result['base64'],validate=True)
            if hashlib.sha256(data).hexdigest()!=result['sha256']:raise ValueError('Metadata chunk changed')
            transfer_pacer.consume(len(data))
            parts.append(data);offset+=len(data)
            if offset==result['size']:break
            if not data:raise EOFError('Metadata read stalled')
        return durable_bytes(output/'remote-results'/local_name,b''.join(parts))
    try:
        prepared=stream.request('initialize',{'mode':mode,'project_id':config['project_id'],'run_id':output.name,
            **({'TR12_authorized_training_seconds':28800} if authorized_TR12 else {}),
            **({'TR11_authorized_training_seconds':authorized_training_seconds} if authorized_training_seconds else {}),
            **({'prepared_numeric_window_seconds':600} if authorized_numeric else {}),
            'unit':unit,'lease_epoch':epoch,'deadline_unix':lease['deadline_unix'],'files':payload,
            'agent_sha256':sha256(package/'server_trial_agent.py'),'manifest_sha256':sha256(manifest_path),
            'manifest_base64':base64.b64encode(manifest_path.read_bytes()).decode(),
            'gpu_index':config.get('gpu_index'),'gpu_uuid':config.get('gpu_uuid')},timeout=45)
        atomic_json(output/'server-preparation.json',prepared)
        for index,item in enumerate(manifest['files']):
            check();stream.request('begin',{'index':index});offset=0
            with originals[item['sha256']].open('rb') as local:
                while data:=local.read(4*2**20):
                    check();transfer_pacer.consume(len(data));stream.request('chunk',{'index':index,'offset':offset,'sha256':hashlib.sha256(data).hexdigest()},data=data,timeout=45)
                    offset+=len(data);progress('uploading_training_inputs',input_index=index,bytes_done=offset,bytes_total=item['bytes'])
            stream.request('commit',{'index':index})
        # This intent remains if observing the launch fails. Never infer a lost
        # SSH connection means the service ended or blindly start another one.
        atomic_json(output/'remote-job.json',lease)
        observation=stream.request('start')
        while observation.get('waiting_for_authorized_idle_gpu'):
            record_wait(observation)
            check();progress('waiting_for_one_authorized_idle_gpu');time.sleep(5);observation=stream.request('start')
        record_wait(observation)
        lease['identity']=observation['identity'];lease['actual_gpu_start_probe']=observation.get('gpu-start-probe.json')
        atomic_json(output/'remote-job.json',lease);atomic_json(output/'server-start-observation.json',observation)
        if mode=='live_archive_check' and not manifest.get('continuous_archive_reuse'):
            from .archive_live_check import controller_faults
            startup=min(480,max(1,deadline-time.time()-300)) if manifest.get('ack_lag_barrier_check') else 120
            faults=controller_faults(stream,output/'online-archive',output.name,epoch,progress,startup_timeout_seconds=startup)
            atomic_json(output/'archive-controller-audit.json',faults)
        while True:
            check();observation=stream.request('status')
            record_wait(observation)
            from .storage_cadence import capacity_signals
            progress('running',remote_progress=observation.get('progress.json'),resources=observation.get('resources.json'),
                storage_capacity=observation.get('capacity'),storage_signals=capacity_signals(observation.get('capacity'),
                    config.get('storage_profile') if authorized_TR12 else None))
            archives_pending=0
            if manifest.get('online_archive_enabled'):
                from .archive_outbox import drain_available
                archived=drain_available(stream,output/'online-archive',output.name,epoch,existing_archive_sources,
                    progress=lambda row,done:progress('archive_receiving',object=row['object'],bytes_done=done,bytes_total=row['bytes']),
                    storage_codec=manifest.get('archive_storage_codec','raw'))
                archives_pending=archived['pending_objects']
            if observation['terminal'] and not archives_pending:break
            time.sleep(5)
    except Exception as exc:failure=exc
    finally:
        try:
            if lease['identity']:
                observation=stream.request('status');atomic_json(output/'server-final-observation.json',observation)
                record_wait(observation)
                if not observation['terminal']:stream.request('close');raise RuntimeError('Same remote unit needs terminal reconciliation')
                receipt={**lease,'host_boot_id':observation['host_boot_id'],'server_unix':observation['server_unix'],
                    'service_properties':observation['service_properties'],'at':utcnow()}
                terminal_path=output/'remote-terminal.json';atomic_json(terminal_path,receipt)
                lease.update(terminal_receipt_path=terminal_path.relative_to(project).as_posix(),terminal_receipt_sha256=sha256(terminal_path))
                atomic_json(output/'remote-job.json',lease)
                for name,record in stream.request('archive_final').items():
                    durable_bytes(output/'remote-final-logs'/name,base64.b64decode(record['base64'],validate=True),record['sha256'])
                metrics=observation.get('trial-metrics.json')
                if active_clock:
                    archive_small('active-clock-server.json','active-clock-server.json')
                    clock_record=read_json(output/'remote-results/active-clock-server.json')
                    for item in clock_record.get('wait_records',[]):
                        archive_small(item['path'],item['path'],maximum=16384)
                        if sha256(output/'remote-results'/item['path'])!=item['sha256']:raise ValueError('Final wait evidence changed')
                if metrics:
                    archive_small('trial-metrics.json','trial-metrics.json')
                    if mode=='basic_cpu_check':archive_small('audit.json','audit.json')
                    if mode in ('compact_cpu_restore','compact_cpu_existing'):archive_small('restore-audit.json','restore-audit.json')
                    if mode=='gpu_existing_restore':archive_small('gpu-restore-audit.json','gpu-restore-audit.json')
                    if manifest.get('prepared_parent_verify'):
                        for name in ['prepared-parent-audit.json','prepared-parent-fresh.json','derived-gpu-bootstrap.json','derived-gpu-restore.json','milestones/32782/manifest.json','milestones/32782/league.json']:
                            archive_small(name,name)
                    if manifest.get('gradient_diagnostic'):archive_small('critic-gradient-diagnostic.json','critic-gradient-diagnostic.json')
                    if manifest.get('continuous_archive_reuse'):archive_small('archive-reuse-audit.json','archive-reuse-audit.json')
                    elif mode=='live_archive_check':archive_small('archive-live-audit.json','archive-live-audit.json')
                    if mode=='journal_io_check':
                        for name in ['journal-io-audit.json','resource-journal-profile.json']:
                            archive_small(name,name)
                    if mode=='cpu_sampler_check':
                        for name in ['CPU-sampling-audit.json','CPU-sampling-fresh.json']:
                            archive_small(name,name)
                    if mode=='fresh_scientific':
                        for name in ['fresh-experiment-inheritance.json','fresh-compact-origin.json','fresh-gpu-bootstrap.json','fresh-gpu-restore.json','cooperative-pause-acceptance.json','terminal-cache-retirement.json','payload-config.json','training-updates.json','league.json']:
                            archive_small(name,name,optional=not metrics.get('complete'))
                    if mode=='derived_scientific':
                        for name in ['derived-inheritance.json','derived-gpu-bootstrap.json','derived-gpu-restore.json','payload-config.json','training-updates.json','league.json']:
                            archive_small(name,name,maximum=8*2**20 if authorized_TR12 and name=='training-updates.json' else 512*1024,
                                optional=not metrics.get('complete') or bool(manifest.get('parent_acceptance_only')))
                        if manifest.get('new_diagnostic_identity')=='TR12':
                            for name in ['checkpoint-commits/index.json','progress.json','archive-retirement-barriers.json','inflight.json','phase-timing-train.json']:
                                archive_small(name,name,optional=name in ('inflight.json','archive-retirement-barriers.json','phase-timing-train.json'))
                        if manifest.get('new_diagnostic_identity') in ('TR11','TR12') and not (authorized_TR12 and manifest.get('parent_acceptance_only')):
                            targets=[32782] if manifest.get('parent_acceptance_only') else manifest['training_config']['decision_milestones']
                            for target in targets:
                                for name in ['manifest.json','league.json']:
                                    path=f'milestones/{target}/{name}'
                                    archive_small(path,path,optional=not metrics.get('complete'))
                archive_small('logs/payload.log','payload.log',optional=not bool(metrics))
            stream.request('close')
        except Exception as exc:
            atomic_json(output/'reconciliation-needed.json',{'at':utcnow(),'error':type(exc).__name__+': '+str(exc),'no_relaunch':True})
            if failure is None:failure=exc
        finally:stream.close()
    if failure:raise failure
    if manifest.get('continuous_archive_reuse'):
        atomic_json(output/'archive-transfer-measurement.json',stream.measurement)
        if stream.measurement['archive_chunk_requests']!=0:raise ValueError('Verified complete originals must be reused without retransmission')
        if stream.measurement['archive_ACK_requests']!=8:raise ValueError('All8 actual live ACKs are required')
    validate_no_unreconciled_remote(project,{'remote_job':lease})
    metrics=observation.get('trial-metrics.json');props=observation['service_properties']
    outcome=trial_outcome(mode,metrics,props)
    atomic_json(output/'metrics.json',{**metrics,'remote_manifest_sha256':sha256(manifest_path),
        'final_service_identity':lease['identity'],'promotion_eligible':False,'controlled_outcome':outcome,
        'engineering_only':bool(manifest.get('parent_acceptance_only')) or mode not in ('scientific_continue','fresh_scientific','derived_scientific')})
    progress(outcome)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();config=read_json(args.config)
    phase_timing.configure(args.output,config.get('phase_timing'),'controller')
    phase_timing.base('remote_job_control_and_wait')
    try:run(config,args.output)
    finally:phase_timing.publish(force=True,process_finalizing=True)
