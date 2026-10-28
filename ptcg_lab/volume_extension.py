"""Full-state continuation segments for the single frozen TR12 endpoint."""
import copy
import os
from pathlib import Path
import time
from .io import read_json,atomic_json,sha256,utcnow
from .ppo_accounting import ACCOUNTING,checkpoint_decisions
from .volume_scope import validate as validate_phase,AUTH,BASE,TARGET,PARENT,PARENT_CP


def validate_training(training,previous):
    if training.get('hypothesis_id')!='TR12' or training.get('target_decisions')!=TARGET or training.get('decision_milestones')!=[TARGET]:
        raise ValueError('The only TR12 endpoint is frozen before training')
    ignored={'hypothesis_id','target_decisions','decision_milestones','primary_change','success_criteria'}
    if {k:v for k,v in training.items() if k not in ignored}!={k:v for k,v in previous.items() if k not in ignored}:
        raise ValueError('Training-volume continuation cannot alter scientific settings')


def validate_local(root,config,deadline,current_run_id=None):
    root=Path(root).resolve();state=read_json(root/'research/autonomy/state.json')
    phase=validate_phase(root,config,state=state)
    acceptance=config.get('parent_acceptance_only') is True
    identity=config.get('acceptance_identity','TR12_PARENT_ACCEPTANCE') if acceptance else 'TR12'
    if acceptance and identity not in ('TR12_PARENT_ACCEPTANCE','TR12_RECOVERY_ACCEPTANCE_002'):
        raise ValueError('Unregistered TR12 acceptance identity')
    maximum=1800 if acceptance else 28800
    if (config.get('check_variant'),config.get('hypothesis_id'),config.get('job_kind'),config.get('engineering_only'),
            config.get('workers'),config.get('training_wall_seconds'))!=('volume_extension',identity,'remote_train_ppo',acceptance,1,maximum):
        raise ValueError('TR12 requires its explicitly bounded scientific continuation route')
    from .active_clock import validate_local as validate_clock
    clock=validate_clock(root,config,state)
    if not 0<deadline-time.time()<=(86400 if clock else maximum)+.5:raise ValueError('TR12 segment exceeds its hard time boundary')
    profile=config['server_resource_profile']
    if profile['cpu_quota_cores']!=3 or profile['memory_max_bytes']!=32*2**30:
        raise ValueError('TR12 initial host quota differs')
    from .remote_lease import validate_no_unreconciled_remote
    prior=[r for r in phase['training_run_ids'] if r!=current_run_id]
    expected=prior[-1] if prior else PARENT
    source=root/'runs'/expected
    if config['existing_restore_source']['run_id']!=expected or config['source_run_id']!=expected:
        raise ValueError('TR12 must continue its latest full state, never restart an older parent')
    if not prior and config['source_point']['checkpoint']['sha256']!=PARENT_CP:
        raise ValueError('The first TR12 segment must use the exact authorized full TR11 checkpoint')
    validate_no_unreconciled_remote(root,{'remote_job':read_json(source/'remote-job.json')})
    for name,digest in config['source_evidence'].items():
        if sha256(source/name)!=digest:raise ValueError('Immutable source evidence changed')
    proof=config['source_state_audit'];path=(root/proof['path']).resolve()
    if root not in path.parents or sha256(path)!=proof['sha256']:raise ValueError('Full-state audit changed')
    audit=read_json(path)
    if (not audit['complete'] or audit['source_run_id']!=expected or not audit['all_full_state_and_pending_archives_verified']
            or not audit['reference_fixture_never_enters_training'] or audit['optimizer_steps_executed']!=0
            or audit['source_point']!=config['source_point'] or audit['numeric_reference']!=config['numeric_reference']):
        raise ValueError('TR12 full source state or recovery-only fixture is not accepted')
    actual=audit['actual_unique_PPO_decisions']
    from .volume_tail import validate_spec
    tail=config.get('complete_postcheckpoint_tail')
    if tail!=audit.get('complete_postcheckpoint_tail'):raise ValueError('Audited complete tail omitted or changed')
    validate_spec(tail,expected,config['source_point'])
    from .phase_timing import SPEC
    if config.get('phase_timing') is not None and config['phase_timing']!=SPEC:raise ValueError('Unreviewed timing options')
    if not BASE<=actual<TARGET or (not prior and actual!=BASE):raise ValueError('Source actual unique cursor differs')
    protocol_path=root/phase['protocol']['path']
    if sha256(protocol_path)!=phase['protocol']['sha256']:raise ValueError('Predeclared TR12 protocol changed')
    protocol=read_json(protocol_path)
    if protocol['training']!={'path':config['training_config_template'],'sha256':config['training_config_template_sha256']}:
        raise ValueError('TR12 scientific target/settings binding changed')
    training_path=root/config['training_config_template']
    if sha256(training_path)!=config['training_config_template_sha256']:raise ValueError('Scientific config changed')
    previous=read_json(source/'remote-training-manifest.json')['training_config']
    validate_training(read_json(training_path),previous)
    numeric=root/config['numeric_reference']['path']
    if sha256(numeric)!=config['numeric_reference']['sha256'] or numeric.stat().st_size!=config['numeric_reference']['bytes']:
        raise ValueError('Exact current-model numeric reference changed')
    if config.get('actual_decision_accounting')!=ACCOUNTING or not config.get('consumed_reference_barrier_required'):
        raise ValueError('Actual first-epoch coverage and archive retirement barrier remain mandatory')


def prepare(root,job,ticket,manifest,engine_hash):
    from .retained_source import resolve,checkpoint_reference
    from .server_workspace import inside,admit
    from .checkpoint_refs import commit_checkpoint,_copy_immutable
    from .training_checkpoint import load_training_checkpoint
    from .tensor_io import load_torch
    from .continuous_mainline import normalize_fixed_file_paths
    from .volume_scope import validate_remote
    validate_remote(manifest['TR12_runtime_scope'],manifest['mode'],manifest['new_diagnostic_identity'])
    source,previous=resolve(root,ticket['project_id'],manifest['existing_restore_source'])
    validate_training(manifest['training_config'],previous['training_config'])
    point=checkpoint_reference(source,manifest['existing_restore_source'])
    if point!=manifest['source_point']:raise ValueError('Latest full-state recovery point changed')
    for name,digest in manifest['source_remote_evidence'].items():
        if sha256(inside(root,source/name))!=digest:raise ValueError('Source terminal state changed')
    saved=load_torch(source/point['checkpoint']['path'],map_location='cpu',weights_only=False)
    actual=checkpoint_decisions(saved)
    if not BASE<=actual<TARGET or len(saved['model'])!=127 or len(saved['optimizer']['state'])!=97:
        raise ValueError('Full TR12 parent state differs')
    if source.name==PARENT and (actual!=BASE or saved['version']!=8 or point['pending'] or point['checkpoint']['sha256']!=PARENT_CP):
        raise ValueError('The initial TR12 source must be the complete frozen TR11 endpoint')
    if source.name!=PARENT and previous['training_config']['hypothesis_id']!='TR12':
        raise ValueError('Continuation cannot borrow a different experiment')
    admit(root,manifest['reserved_future_bytes'])
    for ref,relative in [(point['checkpoint'],'checkpoint.pt'),(point['prefix'],'archive-prefix.json'),
            *[(r,r['path']) for r in point['pending']]]:
        src=inside(root,source/ref['path'])
        if src.stat().st_size!=ref['bytes'] or sha256(src)!=ref['sha256']:raise ValueError('Full checkpoint/pending bytes changed')
        dst=inside(root,job/relative);dst.parent.mkdir(parents=True,exist_ok=True);os.link(src,dst)
    league=read_json(source/'league.json');runtime=read_json(source/'payload-config.json')
    retained=runtime['league'].get('retained_history')
    for row in league['snapshots']:
        base=inside(root,retained['root']) if retained and row['path'] in retained['members'] else source
        src=inside(root,base/row['path'])
        if sha256(src)!=row['sha256']:raise ValueError('Frozen league member changed')
        dst=inside(root,job/row['path']);dst.parent.mkdir(parents=True,exist_ok=True);os.link(src,dst)
    atomic_json(job/'league.json',league)
    if manifest.get('source_inflight'):
        reference=manifest['source_inflight'];src=inside(root,source/'inflight.json')
        if sha256(src)!=reference['sha256'] or src.stat().st_size!=reference['bytes']:raise ValueError('Interrupted BO3 evidence changed')
        _copy_immutable(src,inside(root,job/'inflight.json'))
    from .volume_tail import validate_spec,verify_journal
    tail=manifest.get('complete_postcheckpoint_tail');validate_spec(tail,source.name,point)
    if tail:
        for ref in tail['references']:
            src=inside(root,source/ref['path']);dst=inside(root,job/ref['path'])
            if src.stat().st_size!=ref['bytes'] or sha256(src)!=ref['sha256']:raise ValueError('New tail source changed')
            if dst.exists():raise ValueError('Post-checkpoint tail duplicates checkpoint references')
            dst.parent.mkdir(parents=True,exist_ok=True);os.link(src,dst)
    prefix=read_json(job/'archive-prefix.json')
    spec={'prefix_path':'archive-prefix.json','prefix_sha256':point['prefix']['sha256'],
        'initial_checkpoint_sha256':prefix['checkpoint']['sha256']}
    commit_checkpoint(job,job.name,ticket['lease_epoch'],0,'checkpoint.pt','archive-prefix.json',
        [r['path'] for r in point['pending']],point['cursor'])
    load_training_checkpoint(job,spec,ticket['lease_epoch'])
    if tail:verify_journal(job,saved,spec,point['checkpoint']['sha256'],tail)
    numeric=manifest['numeric_reference'];fixture=inside(root,job/'stage'/(numeric['sha256']+'.blob'))
    if sha256(fixture)!=numeric['sha256'] or fixture.stat().st_size!=numeric['bytes']:raise ValueError('Current-model recovery fixture changed')
    config=normalize_fixed_file_paths(copy.deepcopy(manifest['training_config']))
    config.update(project_path=str(root),engine_manifest_sha256=engine_hash,archive_restore=spec,
        checkpoint_interval_seconds=600,cooperative_resource_control=True,derived_resume_runtime_audit=True,
        online_archive={'maximum_pending_bytes':512*2**20},sampling_device='cpu',checkpoint_transaction_hash_reuse=True,
        CPU_GPU_batch_parity_required=True,CPU_rollout_value_parity_required=True,CPU_backend_fixture=str(fixture),
        consumed_reference_barrier_required=True,actual_decision_accounting=ACCOUNTING,
        continuous_parent_actual_decisions=BASE,continuous_parent_PPO_updates=8,
        segment_start_actual_unique=actual,segment_start_PPO_updates=saved['version'],segment_source_run_id=source.name,
        volume_parent_acceptance_only=manifest.get('parent_acceptance_only') is True,
        phase_timing=manifest.get('phase_timing'),complete_postcheckpoint_tail=tail)
    atomic_json(job/'payload-config.json',config)
    atomic_json(job/'derived-inheritance.json',{'at':utcnow(),'hypothesis_id':'TR12','source_run_id':source.name,
        'source_point':point,'actual_unique_at_segment_start':actual,'frozen_absolute_endpoint':TARGET,
        'all_model_optimizer_RNG_CPU_sampler_pending_and_league_inherited':True,'no_value_or_optimizer_reset':True,
        'all_prior_charges_preserved':True,'old_absolute_deadlines_unchanged':True,'interrupted_BO3_excluded':bool(manifest.get('source_inflight'))})
    return config
