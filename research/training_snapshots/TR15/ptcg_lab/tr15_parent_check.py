"""Latest v28 state only; reuse the already accepted complete archive catalog."""
from pathlib import Path
import io,os,shutil
import torch
from .io import read_json,atomic_json,sha256,utcnow
from .archive_compressed import verify_archive
from .framework_v2_training import semantic_digest
from .ppo_accounting import checkpoint_decisions
from .tr15_acceptance_reuse import check_state,validate_local

def run(config,output):
    root=Path(config['project_path']);output=Path(output)
    if os.environ.get('PTCG_JOB_RUN_ID')!=output.name:raise ValueError('Controlled latest parent check required')
    for ref in [config['previous_full_audit'],config['previous_catalog']]:
        if sha256(root/ref['path'])!=ref['sha256']:raise ValueError('Previously accepted parent archive evidence changed')
    # Prospective new-backend gate: do not claim acceptance before actual GPU proof.
    from .tr15_framework import validate_manifest_local
    validate_manifest_local(root,config)
    old=read_json(root/config['previous_full_audit']['path']);source=root/'runs'/old['source_run_id']
    assert old['actual_unique']==307037 and old['latest_pending_decisions']==0 and old['local_and_server_full_recovery_verified']
    index=source/'remote-results/recovery-v2.json';assert sha256(index)==old['source_v2_index_sha256']
    point=read_json(index)['points'][-1];assert point==old['latest_point'] and point['cursor']['version']==41
    archive=source/'online-archive';acks={read_json(p)['object']:read_json(p) for p in (archive/source.name/'acks').glob('*.json')}
    def payload(ref):
        ack=acks[ref['path']]
        if (ack['sha256'],ack['bytes'])!=(ref['sha256'],ref['bytes']):raise ValueError('Latest state archive binding changed')
        return verify_archive(archive,ack,collect=True)
    model=torch.load(io.BytesIO(payload(point['model']['object'])),map_location='cpu',weights_only=False)
    tail=torch.load(io.BytesIO(payload(point['tail'])),map_location='cpu',weights_only=False)
    state={**model,**tail};mapping=state.pop('v2_reference_map');expected=state.pop('state_semantic_sha256')
    assert semantic_digest(state)==expected and checkpoint_decisions(state)==307037 and len(state['updates'])==41
    assert state.get('partial_ppo') is None and state['version']==41 and state.get('inherited_pending_remaining',0)==0
    from .tr15_acceptance_reuse import format_signature,canonical_hash
    if canonical_hash(format_signature(state))!=config['parent_model_format_sha256']:raise ValueError('Inherited model format changed')
    assert all(state[k]==point['cursor'][k] for k in point['cursor'])
    catalog=read_json(root/config['previous_catalog']['path']);known={x['sha256'] for x in catalog['objects']}
    assert all(ref['sha256'] in known for ref in point['references'])
    shutil.copyfile(root/config['previous_catalog']['path'],output/'verified-archive-catalog.json')
    proof=dict(old,at=utcnow(),new_unique=0,new_PPO_updates=0,baseline_update_digest=semantic_digest(state['updates']),
        inherited_pending_first_epoch_coverage=0,new_sampling_first_epoch_coverage=0,endpoint_model=None,
        local_and_server_full_recovery_verified=False,local_full_state_recovery_verified=True,server_fresh_GPU_recovery_pending=True,
        scientific_parent_unique=307037,scientific_parent_updates=41,latest_only=True,
        reused_full_archive_evidence=config['previous_full_audit'],only_latest_model_tail_original_bytes_reverified=True,
        all_prior_archive_raw_bytes_not_rehashed_again=True,optimizer_steps_executed=0,
        latest_parent_semantic_sha256=expected)
    atomic_json(output/'metrics.json',proof)
