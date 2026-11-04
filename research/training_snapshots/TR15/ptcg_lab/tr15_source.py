"""Read-only complete v2 source and exact canonical staging for the next segment."""
import copy
import os
from pathlib import Path
from types import SimpleNamespace
import torch
from .io import read_json,atomic_json,sha256,utcnow
from .fo_v2.bridge import DriverCheckpoints
from .fo_v2.durable import HashEvidence
from .fo_v2.io_budget import ReadEnvelope
from .retained_source import resolve
from .server_workspace import inside,admit
from .tensor_io import atomic_torch_save,load_torch
from .checkpoint_refs import commit_checkpoint
from .training_checkpoint import pending_references,load_training_checkpoint
from .ppo_accounting import checkpoint_decisions,ACCOUNTING


def read_full(source,run_id,epoch,index_sha256):
    if sha256(source/'recovery-v2.json')!=index_sha256:raise ValueError('Source v2 recovery index changed')
    reader=ReadEnvelope(4*2**20)
    objects=SimpleNamespace(root=source,reader=reader,evidence=HashEvidence(source,reader))
    bridge=DriverCheckpoints(objects,run_id,epoch)
    value,point=bridge.restore(materialize=False)
    if point['model'].get('state_revision')!=value.get('model_state_revision'):
        raise ValueError('Source model/tail revision differs')
    from .framework_v2_training import semantic_digest
    if semantic_digest({k:v for k,v in value.items() if k!='state_semantic_sha256'})!=value['state_semantic_sha256']:
        raise ValueError('Source complete semantic state differs')
    return value,point,bridge.reference_map


def prepare(root,job,ticket,manifest,engine_hash):
    from .tr15_prepare import validate_training
    from .tr15_scope import BASE,TARGET,validate_remote
    from .continuous_mainline import normalize_fixed_file_paths
    validate_remote(manifest['TR13_runtime_scope'],manifest['mode'],manifest['new_diagnostic_identity'])
    source,previous=resolve(root,ticket['project_id'],manifest['existing_restore_source'])
    engineering_recovery=manifest.get('TR15_engineering_recovery')
    if previous.get('engineering_only'):
        if not manifest.get('engineering_only') or not manifest.get('TR15_native_acceptance') or not engineering_recovery or engineering_recovery['source_run_id']!=source.name:
            raise ValueError('Disposable engineering state cannot parent scientific training')
        # Only already completed zero-update evidence is reused; every new
        # partial parameter, optimizer, RNG and cursor is checked again below.
        for name,digest in engineering_recovery['reused_reports'].items():
            src=inside(root,source/name)
            if src.stat().st_size>2*2**20 or sha256(src)!=digest:raise ValueError('Reused engineering report changed')
            raw=src.read_bytes()
            if __import__('hashlib').sha256(raw).hexdigest()!=digest:raise ValueError('Reused report changed during read')
            with (job/name).open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    validate_training(manifest['training_config'],previous['training_config'])
    for name,digest in manifest['source_remote_evidence'].items():
        if sha256(source/name)!=digest:raise ValueError('Terminal v2 source evidence changed')
    spec=manifest['source_v2'];value,point,mapping=read_full(source,source.name,manifest['existing_restore_source']['lease_epoch'],spec['index_sha256'])
    if point!=spec['point']:raise ValueError('Continuation cannot select an earlier v2 checkpoint')
    actual=checkpoint_decisions(value)
    if manifest.get('acceptance_reuse'):
        from .tr15_acceptance_reuse import check_state
        check_state(value,manifest['acceptance_reuse'])
    if not BASE<=actual<TARGET:raise ValueError('TR15 source has already reached endpoint or regressed')
    admit(root,manifest['reserved_future_bytes'])
    for relative,ref in mapping.items():
        src=inside(root,source/ref['path']);dst=inside(root,job/relative)
        if sha256(src)!=ref['sha256'] or src.stat().st_size!=ref['bytes']:raise ValueError('Source canonical immutable reference differs')
        dst.parent.mkdir(parents=True,exist_ok=True);os.link(src,dst)
    for ref in {r['path']:r for r in mapping.values()}.values():
        src=inside(root,source/ref['path']);dst=inside(root,job/ref['path'])
        dst.parent.mkdir(parents=True,exist_ok=True)
        if not dst.exists():os.link(src,dst)
    complete_tail=manifest.get('complete_postcheckpoint_tail')
    if complete_tail:
        if value.get('partial_ppo') is not None:raise ValueError('A partial PPO source cannot append a complete tail')
        for ref in complete_tail['references']:
            src=inside(root,source/ref['path']);dst=inside(root,job/ref['path'])
            if sha256(src)!=ref['sha256'] or src.stat().st_size!=ref['bytes']:raise ValueError('Complete postcheckpoint tail changed')
            dst.parent.mkdir(parents=True,exist_ok=True)
            if dst.exists():
                if sha256(dst)!=ref['sha256']:raise ValueError('Staged tail conflicts with checkpoint')
            else:os.link(src,dst)
        if complete_tail.get('inflight'):
            if read_json(source/'inflight.json')!=complete_tail['inflight']:raise ValueError('Original interrupted BO3 identity changed')
            atomic_json(job/'inflight.json',complete_tail['inflight'])
    atomic_json(job/'league.json',value['league'])
    value=dict(value);value.pop('state_semantic_sha256',None)
    value['staged_v2_reference_map']=mapping
    atomic_torch_save(job/'checkpoint.pt',value,replace=False)
    prefix_names=[name for name,ref in mapping.items() if ref['sha256']==value['archive_origin_prefix_sha256']]
    if len(prefix_names)!=1:raise ValueError('Origin prefix is ambiguous')
    prefix_path=prefix_names[0];prefix=read_json(job/prefix_path)
    archive={'prefix_path':prefix_path,'prefix_sha256':value['archive_origin_prefix_sha256'],
             'initial_checkpoint_sha256':prefix['checkpoint']['sha256']}
    cursor={k:value[k] for k in ('version','trained_decisions','next_series_id')}
    commit_checkpoint(job,job.name,ticket['lease_epoch'],0,'checkpoint.pt',prefix_path,pending_references(value['archive_state']),cursor)
    load_training_checkpoint(job,archive,ticket['lease_epoch'])
    config=normalize_fixed_file_paths(copy.deepcopy(manifest['training_config']))
    config.update(project_path=str(root),engine_manifest_sha256=engine_hash,archive_restore=archive,
        framework_v2_checkpoint_cadence=manifest['framework_v2_checkpoint_cadence'],checkpoint_interval_seconds=740,
        cooperative_resource_control=True,derived_resume_runtime_audit=True,
        online_archive={'maximum_pending_bytes':512*2**20},sampling_device='cpu',
        CPU_GPU_batch_parity_required=True,CPU_rollout_value_parity_required=True,
        consumed_reference_barrier_required=True,actual_decision_accounting=ACCOUNTING,
        continuous_parent_actual_decisions=BASE,continuous_parent_PPO_updates=41,
        segment_start_actual_unique=actual,segment_start_PPO_updates=value['version'],segment_source_run_id=source.name,
        volume_parent_acceptance_only=manifest.get('parent_acceptance_only') is True,phase_timing=manifest.get('phase_timing'),
        complete_postcheckpoint_tail=complete_tail,decision_milestones=[],retire_consumed_with_second_checkpoint=True)
    config['segment_completion_reserve_seconds']=1200
    if manifest.get('shared_capacity'):config['shared_capacity']=manifest['shared_capacity']
    config['staged_v2_bootstrap']=True
    config['legacy_bootstrap_index_role']='staging cache; recovery-v2.json is authoritative after the first native commit'
    readonly=Path('/mnt/HDD1_3TB/PROJECT_OWNER/ptcg-hybrid-v3')
    config['readonly_retained_bytes']=sum(p.lstat().st_size for p in readonly.rglob('*') if p.is_file())
    numeric=manifest['numeric_reference'];fixture=inside(root,job/'stage'/(numeric['sha256']+'.blob'))
    if manifest.get('acceptance_reuse'):
        config['acceptance_reuse']=manifest['acceptance_reuse']
        config['derived_resume_runtime_audit']=False
    else:current_fixture(job,config,value,fixture)
    config['require_exact_restored_GPU_state']=manifest.get('require_exact_restored_GPU_state',False)
    from .tr15_framework import payload_fields
    config.update(payload_fields(manifest))
    atomic_json(job/'payload-config.json',config)
    atomic_json(job/'derived-inheritance.json',{'at':utcnow(),'hypothesis_id':'TR15','source_run_id':source.name,
        'source_v2_point':point,'actual_unique_at_segment_start':actual,'frozen_absolute_endpoint':TARGET,
        'all_model_optimizer_RNG_CPU_sampler_pending_and_league_inherited':True,
        'partial_PPO_preserved':value.get('partial_ppo') is not None,'no_value_or_optimizer_reset':True})
    return config


def current_fixture(job,config,value,fixture):
    """Re-evaluate only recovery queries for the exact current state; never train on them."""
    from .rl_core import load_full_actor,evaluate_decisions,rng_state,restore_rng
    reference=load_torch(fixture,map_location='cpu',weights_only=False)
    state=rng_state()
    try:
        model,features,_=load_full_actor(job/'control'/config['bundle_path'],config['parent_model_sha256'],'cpu')
        model.load_state_dict(value['model']);queries=copy.deepcopy(reference['fixture'])
        if len(queries)!=48:raise ValueError('Recovery query fixture must contain exactly48 legal decisions')
        with torch.no_grad():logp,values,_=evaluate_decisions(model,queries,features.b.STOP)
        for row,l,v in zip(queries,logp,values):row.update(old_logp=float(l),old_value=float(v))
        dest=job/'current-model-recovery-reference.pt'
        atomic_torch_save(dest,{'role':'current_model_recovery_queries_only_never_training',
            'model':value['model'],'fixture':queries,'fixture_logp':logp,'fixture_values':values},replace=False)
        config['CPU_backend_fixture']=str(dest)
    finally:restore_rng(state)
