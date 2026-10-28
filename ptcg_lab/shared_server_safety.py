"""Mandatory admission for shared-host jobs, independent of research budgets."""
from pathlib import Path
from .io import read_json,sha256


REMOTE_KINDS={'server_environment','server_boundary_audit','server_resource_audit','remote_train_ppo','remote_training_check','server_artifact_archive','server_storage_migration','server_storage_acceptance'}
ENTRYPOINTS={
    'server_environment':'ptcg_lab/server_environment.py',
    'server_boundary_audit':'ptcg_lab/server_boundary_audit.py',
    'remote_train_ppo':'ptcg_lab/server_trial.py',
    'remote_training_check':'ptcg_lab/server_trial.py',
}


def _evidence(root,spec):
    root=Path(root).resolve();path=(root/spec['path']).resolve()
    if root not in path.parents or sha256(path)!=spec['sha256']:raise RuntimeError('Shared-server safety evidence changed')
    return read_json(path)


def validate_pre_ssh_startup_failure(root,config):
    """A consumed first attempt can only repair a proved zero-bootstrap failure."""
    run=Path(root)/'runs'/config['pre_ssh_startup_failure_run_id']
    if run.resolve().parent!=(Path(root)/'runs').resolve():raise RuntimeError('Startup failure path escaped')
    attempts=read_json(run/'attempts.json')
    if len(attempts)!=1 or attempts[0]['status']!='failed' or not 0<attempts[0]['seconds']<2:
        raise RuntimeError('First attempt is not the bounded preSSH failure')
    if any((run/name).exists() for name in ['remote-job.json','remote-terminal.json','remote-results','progress.json']):
        raise RuntimeError('Remote bootstrap may have occurred; use checkpoint continuation instead')
    text=(run/'stderr-01.log').read_text(encoding='utf-8')
    if "NameError: name 'package' is not defined" not in text:raise RuntimeError('Different startup failure')
    if abs(config['prior_training_seconds']-attempts[0]['seconds'])>.001:
        raise RuntimeError('Prior startup expense changed')
    if read_json(run/'config.json').get('hypothesis_id')!='TR06':raise RuntimeError('Different scientific parent')
    return attempts[0]['seconds']


def validate_shared_server_job(root,config,state):
    kind=config.get('job_kind')
    if kind not in REMOTE_KINDS:return
    gate=state.get('shared_server_safety',{})
    if not gate.get('authorization'):raise RuntimeError('Shared-server resource authorization is missing')
    authorization=_evidence(root,gate['authorization'])
    if authorization.get('allowed_gpu_indices')!=[4,5,6,7] or authorization.get('maximum_concurrent_gpus')!=1:
        raise RuntimeError('Shared-server authorization differs from latest user scope')
    migration=state.get('storage_migration',{})
    if kind=='remote_train_ppo' and migration.get('all_training_blocked_until_acceptance'):
        raise RuntimeError('HDD2 migration full recovery/stop acceptance is still pending')
    if migration.get('status')=='complete' and kind not in ('server_artifact_archive','server_storage_migration','server_storage_acceptance'):
        from .storage_profile import NEW,validate
        if config.get('server_project_root')!=NEW:
            raise RuntimeError('After accepted migration, all future server writes must explicitly use the HDD2 project root')
        validate(config.get('storage_profile',{}),NEW)
        if kind in ('server_environment','server_boundary_audit','server_resource_audit'):
            raise RuntimeError('This legacy home-root writer is historical; use/adapt a controlled HDD2-specific maintenance entrypoint before writing')
    if kind in ('server_storage_migration','server_storage_acceptance'):
        migration=state.get('storage_migration',{})
        authority=_evidence(root,config['migration_authority'])
        if (migration.get('authority')!=config['migration_authority']
                or migration.get('status')!='preparation' or state.get('active_job')
                or not migration.get('all_training_blocked_until_acceptance')
                or config.get('workers')!=1 or config.get('maximum_attempts')!=1
                or config.get('source_root')!=authority['source_root_read_only']
                or config.get('target_root')!=authority['authorized_target_root']
                or not config.get('engineering_only') or not config.get('no_training_updates')):
            raise RuntimeError('Migration must be exactly the forward-authorized controlled copy')
        from .resource_profile import validate_local_authority
        validate_local_authority(root,config['server_resource_profile'],state)
        return
    if kind=='server_resource_audit':
        if not 0<config.get('audit_total_io_bytes_max',0)<=64*2**20:
            raise RuntimeError('Resource audit exceeds its bounded I/O payload')
        return
    if kind=='server_artifact_archive':
        if config.get('workers')!=1 or config.get('maximum_attempts')!=1 or config.get('server_read_only') is not True:
            raise RuntimeError('Artifact return must be single controlled read-only job')
        if sha256(Path(root)/'ptcg_lab/server_archive.py')!=config.get('archive_source_sha256'):
            raise RuntimeError('Read-only archive execution source changed')
        receipt=_evidence(root,config['source_terminal_evidence'])
        if receipt.get('run_id')!=config.get('source_run_id') or not receipt.get('identity'):
            raise RuntimeError('Archive source termination identity missing')
        from .remote_lease import validate_no_unreconciled_remote
        directory=Path(root)/'runs'/config['source_run_id']
        validate_no_unreconciled_remote(root,{'remote_job':read_json(directory/'remote-job.json')})
        return
    if kind in ('remote_training_check','remote_train_ppo'):
        if config.get('server_resource_profile') is not None:
            from .resource_profile import validate_local_authority
            validate_local_authority(root,config['server_resource_profile'],state)
        if kind=='remote_train_ppo' and config.get('engineering_only') is False and gate.get('future_long_training_requires_journal_capacity_and_IO_acceptance'):
            raise RuntimeError('Future scientific training requires actual journal-capacity and checkpoint-IO acceptance after TR05 failure')
        if config.get('check_variant')=='cpu_sampler':
            if (kind,config.get('workers'),config.get('maximum_attempts'))!=('remote_training_check',1,1):
                raise RuntimeError('CPU sampler acceptance requires one controlled engineering payload')
            from .cpu_sampling_scope import validate
            validate(root,config)
            if not config.get('engineering_only') or not config.get('model_selection_forbidden') or not config.get('no_training_updates') or config.get('gpu_index') is not None or config.get('gpu_uuid') is not None:
                raise RuntimeError('CPU sampler route cannot train or use a GPU')
            review=_evidence(root,gate['CPU_sampler_entrypoint_review'])
            _evidence(root,config['parent_terminal_evidence'])
            environment=_evidence(root,review['actual_environment_evidence'])
            if not environment.get('complete') or not environment.get('remote_environment_installed_and_cpu_smoke_verified'):
                raise RuntimeError('Actual complete private server environment is required')
            forensic=_evidence(root,review['actual_postcommit_forensics'])
            if not forensic.get('both_checkpoints_full_model_optimizer_all_rng_and_updates_exact') or forensic.get('prior_attempt_status_preserved')!='failed':
                raise RuntimeError('Exact full postcommit forensic proof and failed-attempt preservation required')
            for name,digest in review['sources'].items():
                if sha256(Path(root)/name)!=digest:raise RuntimeError('CPU sampler check source changed')
            from .server_trial import trial_source_files
            trial_source_files(Path(root),Path(root),config)
            for row in config['prototype_source_overrides']:
                if row not in review['prototype_source_overrides']:raise RuntimeError('Unreviewed CPU sampler override')
            from .public_fragment_retirement import validate_local_plans
            validate_local_plans(root,config.get('public_fragment_retirement',[]))
            from .archive_compressed import verify_archive
            original=Path(root)/'runs'/config['existing_restore_source']['run_id']
            for ack in config.get('source_consumed_cache_retirement_acks',[]):
                if ack['run_id']!=original.name or ack['lease_epoch']!=config['existing_restore_source']['lease_epoch']:
                    raise RuntimeError('Original consumed-cache archive owner differs')
                verify_archive(original/'online-archive',ack)
            for plan in config.get('terminal_cache_retirement',[]):
                archive=Path(root)/'runs'/plan['durable_local_archive']['run_id']
                if sha256(archive/'file-receipts.json')!=plan['durable_local_archive']['receipts_sha256'] or not read_json(archive/'metrics.json').get('complete'):
                    raise RuntimeError('Engineering fixture lacks a complete durable archive')
                rows={r['path']:r for r in read_json(archive/'file-receipts.json')}
                for obj in plan['objects']:
                    local=archive/'archive'/obj['path']
                    if obj!= {k:rows[obj['path']][k] for k in obj} or local.stat().st_size!=obj['bytes'] or sha256(local)!=obj['sha256']:
                        raise RuntimeError('Retired fixture full local original bytes differ')
            from .remote_lease import validate_no_unreconciled_remote
            parent=Path(root)/'runs'/config['existing_restore_source']['run_id']
            validate_no_unreconciled_remote(root,{'remote_job':read_json(parent/'remote-job.json')})
            return
        if config.get('check_variant')=='journal_io':
            if kind!='remote_training_check' or config.get('engineering_only') is not True or config.get('no_training_updates') is not True:
                raise RuntimeError('Journal/IO acceptance is engineering-only without model updates')
            if (config.get('workers'),config.get('maximum_attempts'),config.get('check_wall_seconds'))!=(1,1,900) or config.get('gpu_index') is not None or config.get('gpu_uuid') is not None:
                raise RuntimeError('Journal/IO check CPU/time/attempt scope differs')
            review=_evidence(root,gate['journal_io_entrypoint_review'])
            actual=_evidence(root,review['actual_TR05_forensic_evidence'])
            if actual.get('classification')!='invalid_incomplete_training_target_and_attempts_exhausted' or not actual.get('complete'):
                raise RuntimeError('Prior full failure evidence missing')
            environment=_evidence(root,review['actual_environment_evidence'])
            if not environment.get('complete'):raise RuntimeError('Private actual Linux environment is required')
            for relative,digest in review['sources'].items():
                if sha256(Path(root)/relative)!=digest:raise RuntimeError('Journal/IO check source changed after review')
            _evidence(root,config['parent_terminal_evidence'])
            return
        if config.get('check_variant')=='continuous_archive_reuse':
            review=_evidence(root,gate['continuous_mainline_entrypoint_review'])
            for relative,digest in review['sources'].items():
                if sha256(Path(root)/relative)!=digest:raise RuntimeError('Archive reuse source changed after review')
            from .continuous_archive_accept import validate_local
            validate_local(root,config)
            return
        if config.get('check_variant')=='continuous_prepared_verify':
            if (kind!='remote_train_ppo' or config.get('sampling_device')!='cpu' or config.get('workers')!=1
                    or config.get('gpu_index') not in authorization['allowed_gpu_indices']):
                raise RuntimeError('Prepared-parent inspection remains one bounded GPU engineering job')
            review=_evidence(root,gate['continuous_mainline_entrypoint_review'])
            for relative,digest in review['sources'].items():
                if sha256(Path(root)/relative)!=digest:raise RuntimeError('Prepared inspection source changed after review')
            from .prepared_parent_verify import validate_local
            validate_local(root,config)
            return
        if config.get('check_variant')=='volume_extension':
            acceptance=config.get('parent_acceptance_only') is True
            if (kind!='remote_train_ppo' or config.get('engineering_only') is not acceptance
                    or state.get('execution_target',{}).get('status')!='active'
                    or not state.get('hybrid_execution',{}).get('full_hybrid_route_activated')):
                raise RuntimeError('TR12 requires the complete guarded hybrid route')
            review=_evidence(root,gate['TR12_volume_entrypoint_review'])
            for relative,digest in review['sources'].items():
                if sha256(Path(root)/relative)!=digest:raise RuntimeError('Reviewed TR12 continuation source changed')
            cpu=_evidence(root,config['CPU_sampler_acceptance']);barrier=_evidence(root,config['ACK_lag_barrier_acceptance'])
            if (not cpu.get('CPU_sampler_actual_acceptance_passed') or barrier.get('status')!='actual_ACK_lag_barrier_acceptance_passed'
                    or not config.get('consumed_reference_barrier_required') or config.get('gpu_index') not in authorization['allowed_gpu_indices']):
                raise RuntimeError('TR12 retains all CPU/ACK/one-GPU protections')
            from .volume_extension import validate_local
            import time
            if state.get('forward_phases',{}).get('TR12',{}).get('active_clock_authorization'):
                from .active_clock import validate_local as validate_clock
                if not validate_clock(root,config,state):raise RuntimeError('Current TR12 authorization requires the wait-exempt clock')
                clock=_evidence(root,gate['TR12_active_clock_actual_acceptance'])
                if not clock.get('complete') or not clock.get('actual_notify_extension_and_expiry_passed'):
                    raise RuntimeError('Real active-clock deadline difference acceptance is missing')
                tested=_evidence(root,clock['sources'])
                for name in ['active_clock','training_control','systemd_job','resource_journal','server_trial_worker','server_trial_agent']:
                    path=f'ptcg_lab/{name}.py'
                    if sha256(Path(root)/path)!=tested[path]:raise RuntimeError('Actually tested clock/control source changed')
            validate_local(root,config,time.time()+(1799 if acceptance else 28799))
            if not acceptance:
                actual=_evidence(root,gate['TR12_volume_actual_acceptance'])
                if not actual.get('complete') or not actual.get('engineering_only') or actual.get('optimizer_steps_executed')!=0:
                    raise RuntimeError('TR12 zero-update full-state/runtime/archive acceptance must pass first')
                if config.get('complete_postcheckpoint_tail'):
                    if (actual.get('derived_inheritance',{}).get('source_point')!=config['source_point']
                            or actual.get('pending_decisions_preserved')!=config['complete_postcheckpoint_tail']['pending_decisions']):
                        raise RuntimeError('Interrupted full-tail GPU acceptance must cover the exact current source')
            return
        if config.get('check_variant')=='continuous_mainline':
            acceptance=config.get('parent_acceptance_only') is True
            if (kind!='remote_train_ppo' or config.get('engineering_only') is not acceptance
                    or config.get('sampling_device')!='cpu' or config.get('workers')!=1
                    or not state.get('hybrid_execution',{}).get('full_hybrid_route_activated')):
                raise RuntimeError('New diagnostic requires the complete guarded hybrid route')
            review=_evidence(root,gate['continuous_mainline_entrypoint_review'])
            for relative,digest in review['sources'].items():
                if sha256(Path(root)/relative)!=digest:raise RuntimeError('Reviewed new diagnostic source changed')
            cpu=_evidence(root,config['CPU_sampler_acceptance'])
            barrier=_evidence(root,config['ACK_lag_barrier_acceptance'])
            if (not cpu.get('CPU_sampler_actual_acceptance_passed')
                    or barrier.get('status')!='actual_ACK_lag_barrier_acceptance_passed'
                    or not config.get('consumed_reference_barrier_required')
                    or not review.get('same_GPU_full_restore_before_any_optimizer')
                    or not review.get('CPU_GPU_parity_before_each_optimizer')
                    or config.get('gpu_index') not in authorization['allowed_gpu_indices']):
                raise RuntimeError('Original CPU/parity/ACK/one-GPU protections remain mandatory')
            from .continuous_mainline import validate_local
            import time
            validate_local(root,config,time.time()+(299 if acceptance else 7199))
            if not acceptance:
                actual=_evidence(root,config['actual_server_parent_acceptance'])
                if not actual.get('parent_acceptance_passed') or not actual.get('engineering_only') or actual.get('optimizer_steps_executed')!=0:
                    raise RuntimeError('Actual new-route full GPU/CPU restore and archive acceptance is required')
            return
        if config.get('check_variant')=='derived_scientific':
            if kind!='remote_train_ppo' or state.get('execution_target',{}).get('status')!='active' or not state.get('hybrid_execution',{}).get('full_hybrid_route_activated'):
                raise RuntimeError('Derived PPO requires actual full hybrid activation')
            review=_evidence(root,gate['derived_training_entrypoint_review'])
            for key in ['actual_live_archive_evidence','actual_first_training_stop_evidence']:_evidence(root,review[key])
            for relative,digest in review['sources'].items():
                if sha256(Path(root)/relative)!=digest:raise RuntimeError('Derived scientific route changed after review')
            hypothesis=config.get('hypothesis_id')
            if hypothesis not in ('TR05','TR07','TR08','TR09','TR10') or (config.get('workers'),config.get('maximum_attempts'),config.get('training_attempt_index'),config.get('training_wall_seconds'))!=(1,1,2,7200):
                raise RuntimeError('Only the reviewed original second7200s attempt is admitted')
            if hypothesis in ('TR07','TR08','TR09','TR10'):
                if review.get('hypothesis_id')!=hypothesis or config.get('sampling_device')!='cpu':
                    raise RuntimeError('CPU sampler continuation requires its exact reviewed route')
                actual=_evidence(root,config['CPU_sampler_acceptance'])
                if not actual.get('complete') or not actual.get('CPU_sampler_actual_acceptance_passed'):
                    raise RuntimeError('Actual server CPU completeBO3/recovery acceptance is required')
                _evidence(root,review['CPU_runtime_activation'])
                if not review.get('actual_GPU_CPU_parity_and_fresh_restore_required_before_optimizer'):
                    raise RuntimeError('Actual sameGPU/CPU numerical acceptance remains mandatory')
                for old in ['TR04','TR05','TR06']+(['TR07'] if hypothesis in ('TR08','TR09','TR10') else [])+(['TR08'] if hypothesis in ('TR09','TR10') else []):
                    if state['research_stage']['hypotheses'][old].get('remaining_original_training_attempts')!=0:
                        raise RuntimeError('Old attempts must stay exhausted')
            if hypothesis in ('TR09','TR10'):
                actual=_evidence(root,review['actual_clean_full_recovery_audit'])
                barrier=_evidence(root,config['ACK_lag_barrier_acceptance'])
                if (actual.get('classification')!=hypothesis+'_original_checkpoint_full_recovery_audited'
                        or not actual.get('complete') or actual.get('remaining_original_attempts')!=1
                        or barrier.get('status')!='actual_ACK_lag_barrier_acceptance_passed'
                        or not config.get('consumed_reference_barrier_required') or not config.get('server_resource_profile')
                        or not review.get('actual_CPU_rollout_GPU_parity_required_before_each_optimizer')):
                    raise RuntimeError('Only the fully audited original second with all actual guards may run')
                for relative,digest in barrier['tested_production_core'].items():
                    if sha256(Path(root)/relative)!=digest:
                        if hypothesis=='TR10' and relative=='ptcg_lab/train_ppo.py' and config.get('archive_counter_compatibility'):
                            from .ppo_counter_compatibility import validate
                            validate(root,config['archive_counter_compatibility'],digest)
                        else:raise RuntimeError('Actually tested archive barrier core changed')
                if hypothesis=='TR10':
                    from .ppo_accounting import ACCOUNTING
                    stage=state['research_stage'];h=stage['hypotheses']['TR10']
                    if (not h.get('diagnostic') or stage['hypotheses']['TR09'].get('status')!='completed'
                            or stage['consecutive_valid_hypotheses_without_promotion']!=4
                            or stage['diagnostic_hypotheses_started']!=1
                            or config.get('actual_decision_accounting')!=ACCOUNTING):
                        raise RuntimeError('Original TR10 diagnostic boundary and actual counting remain binding')
            if hypothesis=='TR08':
                from .archive_compressed import verify_archive
                from .remote_lease import validate_no_unreconciled_remote
                if not review.get('actual_CPU_rollout_GPU_parity_required_before_each_optimizer') or not config.get('server_resource_profile'):
                    raise RuntimeError('TR08 per-update parity and explicit future CPU/RAM profile required')
                for plan in config.get('terminal_cache_retirement',[]):
                    name=plan['source']['run_id'];original=Path(root)/'runs'/name
                    if name not in ('20261025T212721Z_tr07-v01_2f8e6f','20261025T233324Z_tr07-CPU-sampler-startup-repair_bb99ae'):
                        raise RuntimeError('Unreviewed obsolete terminal cache owner')
                    validate_no_unreconciled_remote(root,{'remote_job':read_json(original/'remote-job.json')})
                    for obj in plan['objects']:
                        ack=obj['retained_local_ACK'];verify_archive(original/'online-archive',ack)
                        if (ack['run_id'],ack['bytes'],ack['sha256'])!=(name,obj['bytes'],obj['sha256']):
                            raise RuntimeError('Obsolete tensor full primary archive differs')
            if config.get('engineering_only') is not False or config.get('gpu_index') not in authorization['allowed_gpu_indices']:
                raise RuntimeError('Derived scientific role/GPU scope differs')
            if state['research_stage']['hypotheses']['TR04'].get('remaining_original_training_attempts')!=0:
                raise RuntimeError('TR04 remains closed')
            from .scientific_resume import validate_scope
            import time
            validate_scope(root,config,time.time()+7199)
            _evidence(root,config['parent_terminal_evidence'])
            return
        if config.get('check_variant')=='fresh_scientific':
            if kind!='remote_train_ppo' or state.get('execution_target',{}).get('status')!='active' or not state.get('hybrid_execution',{}).get('full_hybrid_route_activated'):
                raise RuntimeError('Fresh PPO requires actual full hybrid activation')
            review=_evidence(root,gate['fresh_training_entrypoint_review'])
            for key in ['actual_live_archive_evidence','actual_gradient_diagnostic_evidence']:_evidence(root,review[key])
            for relative,digest in review['sources'].items():
                if sha256(Path(root)/relative)!=digest:raise RuntimeError('Fresh scientific execution source changed after review')
            identity=config.get('hypothesis_id')
            attempt=config.get('training_attempt_index')
            if identity not in ('TR05','TR06','TR07','TR08','TR09','TR10') or (config.get('workers'),config.get('maximum_attempts'),config.get('training_wall_seconds'))!=(1,1,7200) or attempt not in (1,2):
                raise RuntimeError('Only reviewed original first7200s fresh scientific attempt is admitted')
            if identity in ('TR06','TR07','TR08','TR09','TR10'):
                if review.get('hypothesis_id')!=identity:raise RuntimeError('Exact fresh scientific route review is required')
                actual=_evidence(root,review['actual_journal_IO_acceptance'])
                if actual.get('status')!='actual_CPU_repair_acceptance_passed':raise RuntimeError('Actual journal/IO repair required')
                for old in ['TR04','TR05']+(['TR06'] if identity in ('TR07','TR08','TR09','TR10') else [])+(['TR07'] if identity in ('TR08','TR09','TR10') else [])+(['TR08'] if identity in ('TR09','TR10') else []):
                    row=state['research_stage']['hypotheses'][old]
                    if row['status']!='invalid' or row.get('remaining_original_training_attempts')!=0:
                        raise RuntimeError('Old scientific attempts must remain exhausted')
                if identity in ('TR07','TR08','TR09','TR10'):
                    accepted=_evidence(root,review['actual_Windows_archive_repair'])
                    if not accepted.get('passed') or not accepted.get('complete') or len(accepted.get('actual_Windows_long_path_RAW_and_gzip_roundtrips',[]))!=4:
                        raise RuntimeError('Actual real-data Windows archive acceptance required')
                if identity in ('TR08','TR09','TR10'):
                    cpu=_evidence(root,config['CPU_sampler_acceptance'])
                    if (config.get('sampling_device')!='cpu' or not cpu.get('CPU_sampler_actual_acceptance_passed')
                            or not cpu.get('complete') or cpu.get('training_updates_executed')!=0
                            or not review.get('actual_CPU_rollout_GPU_parity_required_before_each_optimizer')):
                        raise RuntimeError('Actual CPU sampler acceptance and production CPU/GPU batch parity required')
                if identity in ('TR09','TR10'):
                    accepted=_evidence(root,config['ACK_lag_barrier_acceptance'])
                    if (accepted.get('status')!='actual_ACK_lag_barrier_acceptance_passed'
                            or config.get('consumed_reference_barrier_required') is not True
                            or not config.get('server_resource_profile')
                            or review.get('actual_ACK_lag_barrier_acceptance')!=config['ACK_lag_barrier_acceptance']):
                        raise RuntimeError('CPU PPO requires the actual delayed ACK barrier and resource profile')
                    for relative,digest in accepted['tested_production_core'].items():
                        if sha256(Path(root)/relative)!=digest:raise RuntimeError('Actually tested ACK barrier core changed')
                if identity=='TR10':
                    stage=state['research_stage'];row=stage['hypotheses'].get(identity,{})
                    credit=_evidence(root,review['actual_credit_analysis'])
                    if (row.get('diagnostic') is not True or stage['consecutive_valid_hypotheses_without_promotion']<4
                            or stage['hypotheses']['TR09'].get('status')!='completed'
                            or credit.get('new_scientific_hypothesis_registered') is not False
                            or credit.get('sample_complete_BO3')!=16
                            or review.get('single_primary_variable')!='gae_lambda:0.95->1.0'):
                        raise RuntimeError('TR10 requires the distinct registered Monte Carlo diagnostic and terminal TR09')
            if config.get('engineering_only') is not False or config.get('gpu_index') not in authorization['allowed_gpu_indices']:
                raise RuntimeError('Fresh scientific role/GPU authorization differs')
            if state['research_stage']['hypotheses']['TR04']['status']!='invalid' or state['research_stage']['hypotheses']['TR04'].get('remaining_original_training_attempts')!=0:
                raise RuntimeError('Original TR04 must stay closed with exhausted attempts')
            runs=state['research_stage']['hypotheses'].get(identity,{}).get('run_ids',[])
            if attempt==2:
                if identity!='TR06' or len(runs)!=1 or config.get('pre_ssh_startup_failure_run_id')!=runs[0]:
                    raise RuntimeError('Second bootstrap requires the exact first preSSH startup failure')
                validate_pre_ssh_startup_failure(root,config)
                actual=_evidence(root,review['actual_pre_SSH_startup_failure'])
                if actual.get('first_run_id')!=runs[0] or actual.get('no_remote_bootstrap_occurred') is not True:
                    raise RuntimeError('Actual zero-bootstrap failure proof missing')
            elif runs:
                raise RuntimeError('Fresh scientific initializer already has a raw run; no reset or repeated first attempt')
            return
        if config.get('check_variant')=='scientific_continue':
            if kind!='remote_train_ppo' or state.get('execution_target',{}).get('status')!='active' or not state.get('hybrid_execution',{}).get('full_hybrid_route_activated'):
                raise RuntimeError('Scientific PPO requires actual full hybrid activation')
            review=_evidence(root,gate['full_training_entrypoint_review'])
            acceptance=_evidence(root,review['actual_live_archive_evidence'])
            if acceptance.get('classification')!='actual_live_archive_faults_and_two_point_retention_passed':raise RuntimeError('Actual live archive acceptance missing')
            for relative,digest in review['sources'].items():
                if sha256(Path(root)/relative)!=digest:raise RuntimeError('Scientific training route changed after review')
            if (config.get('hypothesis_id'),config.get('workers'),config.get('maximum_attempts'),config.get('original_attempt_number'),config.get('original_training_run_id'))!=('TR04',1,1,2,'20261023T182702Z_tr04-ppo-lambda99_f9bdaf'):
                raise RuntimeError('Only original TR04 second attempt is admitted')
            if config.get('engineering_only') is not False or config.get('continuation_wall_seconds')!=3600 or config.get('gpu_index') not in authorization['allowed_gpu_indices']:
                raise RuntimeError('Scientific continuation role/time/GPU scope differs')
            from .hybrid_continue import validate_original_attempt_scope
            import time
            validate_original_attempt_scope(root,config,time.time()+3599)
            _evidence(root,config['parent_terminal_evidence'])
            for run in (Path(root)/'runs').iterdir():
                path=run/'config.json'
                if path.exists() and read_json(path).get('check_variant')=='scientific_continue':
                    raise RuntimeError('Original second continuation already has a raw run; reconcile it, no third attempt')
            return
        spec=gate.get('live_archive_entrypoint_review') if config.get('check_variant')=='live_archive' else gate.get('cuda_restore_entrypoint_review') if config.get('check_variant')=='gpu_existing' else gate.get('compact_restore_entrypoint_review') if config.get('check_variant') in ('compact_restore','compact_existing') else gate.get('short_training_entrypoint_review')
        if not spec or not gate.get('staged_authorization'):raise RuntimeError('Staged training entrypoint not reviewed')
        review=_evidence(root,spec);_evidence(root,gate['staged_authorization'])
        if review.get('authorization_sha256')!=gate['staged_authorization']['sha256']:
            raise RuntimeError('Staged training authorization binding changed')
        environment=_evidence(root,review['actual_environment_evidence'])
        if not environment.get('complete') or not environment.get('remote_environment_installed_and_cpu_smoke_verified'):
            raise RuntimeError('Actual complete private Linux environment is required')
        sources=review.get('sources',{})
        required={'ptcg_lab/server_trial.py','ptcg_lab/server_trial_agent.py','ptcg_lab/server_trial_worker.py',
                  'ptcg_lab/server_environment_agent.py','ptcg_lab/staged_environment.py','ptcg_lab/systemd_job.py',
                  'ptcg_lab/server_workspace.py','ptcg_lab/ssh_stream.py','ptcg_lab/rl_audit.py','ptcg_lab/train_ppo.py',
                  'ptcg_lab/rl_core.py','ptcg_lab/rl_collector.py','ptcg_lab/rl_engine_worker.py','scripts/run_experiment.py'}
        if not required.issubset(sources):raise RuntimeError('Short training execution dependencies not bound')
        if config.get('check_variant')=='live_archive':
            if kind!='remote_training_check' or config.get('live_archive_wall_seconds')!=900 or config.get('no_training_updates') is not True:
                raise RuntimeError('Live archive acceptance only permits900s CPU with no updates')
            if not {'ptcg_lab/archive_live_check.py','ptcg_lab/archive_compressed.py','ptcg_lab/archive_outbox.py','ptcg_lab/checkpoint_refs.py'}.issubset(sources):raise RuntimeError('Live archive dependencies not bound')
            _evidence(root,config['parent_terminal_evidence'])
            actual=_evidence(root,review['actual_cuda_restore_evidence'])
            if not actual.get('report',{}).get('passed'):raise RuntimeError('Actual original CUDA restore acceptance missing')
            if config.get('ack_lag_barrier_check'):
                if config.get('gpu_index') is not None or config.get('gpu_uuid') is not None or not config.get('server_resource_profile'):
                    raise RuntimeError('ACK barrier check requires CPU-only explicit authorized profile')
                closed=state['research_stage']['hypotheses']['TR08']
                if closed.get('status')!='invalid' or closed.get('remaining_original_training_attempts')!=0:
                    raise RuntimeError('TR08 must remain closed before its operational repair check')
                full=_evidence(root,review['actual_TR08_failure_analysis'])
                if not full.get('complete') or full.get('trained_decisions')!=24837:raise RuntimeError('Full real failure evidence required')
                if config.get('hdd2_no_retirement_acceptance'):
                    # Migration has room for a tiny private fixture. Reusing the
                    # old cleanup inode plan would target already-retired files.
                    actual=_evidence(root,config['HDD2_migration_acceptance'])
                    if (migration.get('status')!='complete'
                            or config['HDD2_migration_acceptance']['path']!=migration.get('activation_receipt')
                            or actual.get('status')!='complete'
                            or not actual.get('source_model_and_environment_migration_accepted')
                            or config.get('terminal_cache_retirement')!=[]
                            or review.get('HDD2_migration_acceptance')!=config['HDD2_migration_acceptance']):
                        raise RuntimeError('No-retirement HDD2 check requires actual activated full migration')
                else:
                    from .terminal_cache_plans import validate_local_plans
                    validate_local_plans(root,config['terminal_cache_retirement'],set(closed['run_ids']))
        if config.get('check_variant')=='gpu_existing':
            if kind!='remote_train_ppo' or config.get('restore_wall_seconds')!=300 or config.get('gpu_index') not in authorization['allowed_gpu_indices']:
                raise RuntimeError('Original GPU restore check requires one authorized GPU and300s cap')
            if not {'ptcg_lab/gpu_restore.py','ptcg_lab/retained_source.py'}.issubset(sources):raise RuntimeError('GPU restore source not bound')
            actual=_evidence(root,review['actual_linux_restore_evidence'])
            if actual.get('classification')!='actual_linux_compact_restore_passed_engineering_only' or not actual['report'].get('unpublished_hot_checkpoint_ignored'):
                raise RuntimeError('Actual full Linux checkpoint restore proof missing')
            _evidence(root,config['parent_terminal_evidence'])
            from .remote_lease import validate_no_unreconciled_remote
            source=Path(root)/'runs'/config['existing_restore_source']['run_id']
            validate_no_unreconciled_remote(root,{'remote_job':read_json(source/'remote-job.json')})
        if config.get('check_variant') in ('compact_restore','compact_existing'):
            maximum=1200 if config['check_variant']=='compact_existing' else 3600
            if kind!='remote_training_check' or config.get('compact_restore_wall_seconds')!=maximum:
                raise RuntimeError('Compact restore only admits bounded CPU checks')
            if not {'ptcg_lab/restore_capsule.py','ptcg_lab/history_delta.py','ptcg_lab/rl_resume.py','ptcg_lab/training_checkpoint.py','ptcg_lab/checkpoint_refs.py'}.issubset(sources):
                raise RuntimeError('Compact restore dependencies not bound')
            actual=_evidence(root,review['actual_short_trial_evidence'])
            if not actual.get('short_gpu_trial_passed'):raise RuntimeError('Prior actual short trial evidence is missing')
            capsule=_evidence(root,review['actual_restore_capsule_evidence'])
            if not capsule.get('complete') or capsule.get('full_history_members')!=31 or capsule.get('tail_shards')!=35 or capsule.get('pending_decisions')!=5430:
                raise RuntimeError('Actual complete original TR04 restore capsule is missing')
            if config['check_variant']=='compact_existing':
                _evidence(root,config['parent_terminal_evidence'])
                from .remote_lease import validate_no_unreconciled_remote
                source=Path(root)/'runs'/config['existing_restore_source']['run_id']
                validate_no_unreconciled_remote(root,{'remote_job':read_json(source/'remote-job.json')})
        for relative,digest in sources.items():
            path=(Path(root)/relative).resolve()
            if Path(root).resolve() not in path.parents or sha256(path)!=digest:
                raise RuntimeError('Staged training source changed after review')
        if config.get('workers')!=1 or config.get('maximum_attempts')!=1 or not config.get('engineering_only') or not config.get('model_selection_forbidden'):
            raise RuntimeError('This route only permits one bounded engineering check/trial')
        if kind=='remote_train_ppo' and config.get('check_variant')!='gpu_existing':
            if config.get('gpu_index') not in authorization['allowed_gpu_indices'] or config.get('trial_wall_seconds')!=600:
                raise RuntimeError('Short trial must use one authorized GPU for at most600s')
            if not config.get('basic_correctness_evidence'):raise RuntimeError('Actual CPU actor/cg/restart acceptance is missing')
            basic=_evidence(root,config['basic_correctness_evidence'])
            if not basic.get('complete') or not basic.get('basic_cpu_correctness_passed'):
                raise RuntimeError('CPU basic correctness did not pass')
        return
    if kind=='server_environment' and config.get('preparation_mode') in ('staged_direct_recovery','staged_range_repair'):
        # Latest explicit user staging replaces only the old all-I/O prerequisite.
        # Kernel limits and project ownership are checked in the service before pip.
        if not gate.get('staged_authorization') or not gate.get('environment_entrypoint_acceptance'):
            raise RuntimeError('Staged installation authorization/review is missing')
        staged=_evidence(root,gate['staged_authorization'])
        review=_evidence(root,gate['environment_entrypoint_acceptance'])
        if (staged.get('preserve_all_old_costs_run_counts_stopping_decisions_and_final_reserves') is not True
                or review.get('authorization_sha256')!=gate['staged_authorization']['sha256']):
            raise RuntimeError('Staged authorization binding differs')
        policy=read_json(Path(root)/'research/autonomy/policy.json')
        safety=policy['execution']['hybrid']['shared_server_safety']
        if not safety.get('staged_execution') or not safety.get('hostwide_io_is_background_only'):
            raise RuntimeError('Effective policy has not activated staged installation')
        boundary=_evidence(root,review['prior_boundary_evidence'])
        if not all(boundary.get(k) is True for k in ['actual_server_boundary_checks_passed',
                   'wrong_invocation_stop_rejected','same_parent_and_child_identities_terminated',
                   'actual_unreconciled_lease_gate_rejected_new_computation']):
            raise RuntimeError('Required actual stop/identity evidence is missing')
        sources=review.get('sources',{})
        required={'ptcg_lab/server_environment.py','ptcg_lab/server_environment_agent.py',
                  'ptcg_lab/server_environment_worker.py','ptcg_lab/staged_environment.py',
                  'ptcg_lab/server_workspace.py','ptcg_lab/systemd_job.py','ptcg_lab/ssh_stream.py',
                  'ptcg_lab/io.py','ptcg_lab/bounded_io.py','ptcg_lab/bounded_read.py','ptcg_lab/io_pacing.py'}
        if not required.issubset(sources):raise RuntimeError('Staged entrypoint dependencies not bound')
        for relative,digest in sources.items():
            path=(Path(root)/relative).resolve()
            if Path(root).resolve() not in path.parents or sha256(path)!=digest:
                raise RuntimeError('Staged installation source changed after review')
        if config.get('workers')!=1 or config.get('maximum_attempts')!=1:
            raise RuntimeError('Staged installation must be single worker/attempt')
        if config.get('installation_scope_seconds_max')!=3600 or config.get('prior_installation_scope_seconds',0)<=0:
            raise RuntimeError('Original installation scope must remain charged')
        if config.get('preparation_mode')=='staged_range_repair':
            spec=config.get('forward_repair_contract')
            if not spec:raise RuntimeError('Forward repair has no immutable preregistration')
            contract=_evidence(root,spec)
            if (contract.get('old_attempts_and_stops_preserved') is not True
                    or config.get('forward_cumulative_installation_seconds_max')!=contract.get('cumulative_installation_seconds_ceiling')
                    or contract.get('new_repair_wall_seconds_max',0)>3600
                    or 'ptcg_lab/range_artifacts.py' not in sources):
                raise RuntimeError('Forward engineering repair scope differs')
        return
    if not gate.get('acceptance'):
        raise RuntimeError('Shared-server CPU/memory/I/O and isolation acceptance is pending')
    acceptance=_evidence(root,gate['acceptance'])
    required=['directory_ownership_and_realpath','cpu_kernel_limit','memory_kernel_limit',
              'io_all_descendants_covered','io_throttle_exercised','io_excess_stops_owned_unit',
              'guard_failure_stops_owned_unit','disk_reserve_and_workset_stop','control_transfer_io_paced',
              'kernel_block_io_upper_bound_present','exited_child_kernel_block_io_retained',
              'zombie_has_independent_kernel_coverage','monitor_loss_stops_owned_unit',
              'clean_payload_completion','private_inprocess_pip_install','same_pid_exec_io_accounting','exit_barrier_acknowledged']
    if any(acceptance.get('checks',{}).get(name) is not True for name in required):
        raise RuntimeError('Shared-server resource acceptance is incomplete')
    if acceptance.get('authorization_sha256')!=gate['authorization']['sha256']:
        raise RuntimeError('Shared-server acceptance used another authorization')
    if not config.get('shared_guard_profile_sha256') or config['shared_guard_profile_sha256']!=acceptance.get('profile_sha256'):
        raise RuntimeError('Job does not use the verified shared-server resource profile')
    # A passed synthetic guard must not silently authorize old unguarded
    # installation/training entrypoints merely because a config copies its hash.
    route=acceptance.get('approved_entrypoints',{}).get(kind)
    entrypoint=ENTRYPOINTS.get(kind)
    if not route or route.get('entrypoint')!=entrypoint:
        raise RuntimeError('Shared-server execution entrypoint has no integrated guard acceptance')
    sources=route.get('sources',{})
    required_sources={entrypoint,'ptcg_lab/process_guardian.py','ptcg_lab/systemd_job.py',
                      'ptcg_lab/server_workspace.py','ptcg_lab/unit_io_guard.py','ptcg_lab/host_block_io.py',
                      'ptcg_lab/io.py','ptcg_lab/io_pacing.py','ptcg_lab/ssh_stream.py','ptcg_lab/bounded_io.py','ptcg_lab/tensor_io.py','ptcg_lab/bounded_read.py'}
    if kind=='server_environment':
        required_sources.update({'ptcg_lab/server_environment_agent.py','ptcg_lab/server_environment_worker.py'})
    if not required_sources.issubset(sources):
        raise RuntimeError('Guard acceptance does not bind all execution dependencies')
    for relative,digest in sources.items():
        path=(Path(root)/relative).resolve()
        if Path(root).resolve() not in path.parents or not path.is_file() or sha256(path)!=digest:
            raise RuntimeError('Guarded execution source changed after acceptance')


def validate_gpu_choice(authorization,device,compute_processes):
    if device['index'] not in authorization['allowed_gpu_indices']:
        raise RuntimeError('GPU is outside administrator-authorized range')
    if authorization['maximum_concurrent_gpus']!=1:raise RuntimeError('Only one authorized GPU may be selected')
    if any(row['gpu_uuid']==device['uuid'] for row in compute_processes):raise RuntimeError('Authorized GPU has existing compute users')
    if device['memory_used_mib']>64 or device['utilization_percent']>1:raise RuntimeError('Authorized GPU is not idle')
    return device['uuid']
