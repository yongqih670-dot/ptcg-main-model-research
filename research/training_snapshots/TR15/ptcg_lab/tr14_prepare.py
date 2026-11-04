"""Stage exact TR12-v15 full state for the independent authorized TR14 phase."""
import copy
import os
from pathlib import Path
import time
from .io import read_json,atomic_json,sha256,utcnow
from .ppo_accounting import ACCOUNTING,checkpoint_decisions
from .tr14_scope import AUTH,BASE,TARGET,validate as validate_phase
PARENT='historical-parent-never-reopened'
PARENT_CP=None

def validate_training(training,previous):
    if training.get('hypothesis_id')!='TR14' or training.get('target_decisions')!=TARGET or training.get('decision_milestones')!=[TARGET]:
        raise ValueError('Only the directly authorized 100k endpoint is permitted')
    ignored={'hypothesis_id','target_decisions','decision_milestones','primary_change','success_criteria'}
    if {k:v for k,v in training.items() if k not in ignored}!={k:v for k,v in previous.items() if k not in ignored}:
        raise ValueError('TR14 cannot alter inherited PPO, opponent, seed or actor settings')

def validate_local(root,config,deadline=None,current_run_id=None,state=None):
    root=Path(root).resolve();state=state or read_json(root/'research/autonomy/state.json')
    phase=validate_phase(root,config,state)
    targeted=config.get('TR14_timeout_acceptance')
    expected_source=phase['recovery_required_from_positive_updates']['run_id'] if targeted else phase['latest_full_state_source']
    if config['source_run_id']!=expected_source:
        raise ValueError('TR14 must continue its latest verified complete state')
    if targeted:
        from .tr14_acceptance_reuse import validate_targeted_difference
        if not config.get('engineering_only'):raise ValueError('Targeted timing acceptance must not train')
        validate_targeted_difference(root,targeted['basis_acceptance'],targeted['difference_reference'])
        proof=read_json(root/config['source_state_audit']['path'])
        if not proof.get('local_full_state_recovery_verified') or proof['source_run_id']!=expected_source:
            raise ValueError('Latest failed full state must pass local byte/semantic recovery before GPU checks')
    if config['job_kind']!='remote_train_ppo' or config['check_variant']!='tr14_native':
        raise ValueError('TR14 scientific route differs')
    if config['server_project_root']!='/mnt/HDD2_3TB/PROJECT_OWNER/ptcg-hybrid-v3':
        raise ValueError('TR14 writable root differs')
    maximum=1800 if config['engineering_only'] else 28800
    if config['training_wall_seconds']>maximum:raise ValueError('TR14 raw execution window differs')
    if deadline is not None and not 0<deadline-time.time()<=86400.5:
        raise ValueError('TR14 hard safety window differs')
    source=root/'runs'/config['source_run_id']
    if config.get('source_v2'):
        candidate=config['source_v2'];index_path=source/'remote-results/recovery-v2.json'
        if sha256(index_path)!=candidate['index_sha256'] or read_json(index_path)['points'][-1]!=candidate['point']:
            raise ValueError('TR14 must use the latest fully archived v2 source point')
    from .remote_lease import validate_no_unreconciled_remote
    validate_no_unreconciled_remote(root,{'remote_job':read_json(source/'remote-job.json')})
    for name,digest in config['source_evidence'].items():
        if sha256(source/name)!=digest:raise ValueError('Immutable parent terminal evidence changed')
    proof=config['source_state_audit']
    if sha256(root/proof['path'])!=proof['sha256']:raise ValueError('Parent full-state audit changed')
    numeric=config['numeric_reference'];path=root/numeric['path']
    if sha256(path)!=numeric['sha256'] or path.stat().st_size!=numeric['bytes']:
        raise ValueError('Exact current model numerical reference changed')
    if sha256(root/config['training_config_template'])!=config['training_config_template_sha256']:
        raise ValueError('TR14 frozen training config changed')
    validate_training(read_json(root/config['training_config_template']),read_json(source/'remote-training-manifest.json')['training_config'])
    from .framework_v2_checkpoint_cadence import load_profile
    if config['framework_v2_checkpoint_cadence']!=load_profile(root):raise ValueError('TR14 checkpoint profile differs')

def prepare(root,job,ticket,manifest,engine_hash):
    if manifest.get('retained_numeric_reference'):
        from .server_workspace import inside
        numeric=manifest['numeric_reference'];item=manifest['retained_numeric_reference']
        src=inside(root,root/item['path']);dst=inside(root,job/'stage'/(numeric['sha256']+'.blob'))
        if (item['sha256'],item['bytes'])!=(numeric['sha256'],numeric['bytes']) or sha256(src)!=item['sha256'] or src.stat().st_size!=item['bytes']:
            raise ValueError('Retained numerical reference bytes differ')
        os.link(src,dst)
    if manifest.get('source_v2'):
        from .tr14_source import prepare as prepare_v2
        return prepare_v2(root,job,ticket,manifest,engine_hash)
    from .retained_source import resolve,checkpoint_reference
    from .server_workspace import inside,admit
    from .checkpoint_refs import commit_checkpoint,_copy_immutable
    from .training_checkpoint import load_training_checkpoint
    from .tensor_io import load_torch
    from .continuous_mainline import normalize_fixed_file_paths
    from .tr14_scope import validate_remote
    validate_remote(manifest['TR13_runtime_scope'],manifest['mode'],manifest['new_diagnostic_identity'])
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
    if source.name!=PARENT and previous['training_config']['hypothesis_id'] not in ('TR12','TR13','TR14'):
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
        checkpoint_interval_seconds=740,cooperative_resource_control=True,derived_resume_runtime_audit=True,
        online_archive={'maximum_pending_bytes':512*2**20},sampling_device='cpu',checkpoint_transaction_hash_reuse=True,
        CPU_GPU_batch_parity_required=True,CPU_rollout_value_parity_required=True,CPU_backend_fixture=str(fixture),
        consumed_reference_barrier_required=True,actual_decision_accounting=ACCOUNTING,
        continuous_parent_actual_decisions=BASE,continuous_parent_PPO_updates=15,
        segment_start_actual_unique=actual,segment_start_PPO_updates=saved['version'],segment_source_run_id=source.name,
        volume_parent_acceptance_only=manifest.get('parent_acceptance_only') is True,
        phase_timing=manifest.get('phase_timing'),complete_postcheckpoint_tail=tail)
    config['framework_v2_checkpoint_cadence']=manifest['framework_v2_checkpoint_cadence']
    if manifest.get('shared_capacity'):config['shared_capacity']=manifest['shared_capacity']
    config['decision_milestones']=[]  # endpoint package is produced after full two-point acceptance
    config['retire_consumed_with_second_checkpoint']=True
    config['segment_completion_reserve_seconds']=1200
    readonly=Path('/mnt/HDD1_3TB/PROJECT_OWNER/ptcg-hybrid-v3')
    config['readonly_retained_bytes']=sum(p.lstat().st_size for p in readonly.rglob('*') if p.is_file()) if readonly.exists() else 0
    atomic_json(job/'payload-config.json',config)
    atomic_json(job/'derived-inheritance.json',{'at':utcnow(),'hypothesis_id':'TR14','source_run_id':source.name,
        'source_point':point,'actual_unique_at_segment_start':actual,'frozen_absolute_endpoint':TARGET,
        'all_model_optimizer_RNG_CPU_sampler_pending_and_league_inherited':True,'no_value_or_optimizer_reset':True,
        'all_prior_charges_preserved':True,'old_absolute_deadlines_unchanged':True,'interrupted_BO3_excluded':bool(manifest.get('source_inflight'))})
    return config
