"""Audit only new TR11 optimizer coverage; never recount its inherited parent."""
import copy
import io
import math
import os
from pathlib import Path
import time
from .io import read_json,atomic_json,sha256,utcnow
from .ppo_accounting import checkpoint_decisions,update_decisions,ACCOUNTING


def run(config,output):
    import torch
    from .archive_compressed import verify_archive
    root=Path(config['project_path']).resolve();output=Path(output).resolve()
    if os.environ.get('PTCG_JOB_RUN_ID')!=output.name or output.parent!=root/'runs':raise RuntimeError('Controlled result audit required')
    deadline=float(os.environ['PTCG_JOB_DEADLINE_UNIX'])
    if not 0<deadline-time.time()<=180.5:raise ValueError('One bounded180s CPU accounting audit')
    torch.set_num_threads(2)
    for path,digest in config['input_evidence'].items():
        if sha256(root/path)!=digest:raise ValueError('Immutable result audit evidence changed')
    source=root/'runs'/config['training_run_id'];m=read_json(source/'metrics.json')
    if not m['complete'] or m['inherited_actual_unique_PPO_decisions']!=32782:raise ValueError('Complete inherited TR11 target required')
    milestone=read_json(source/'remote-results/milestones/49166/manifest.json')
    digest=milestone['full_state']['checkpoint']['sha256'];archive=source/'online-archive'
    acks=[read_json(p) for p in (archive/source.name/'acks').glob('*.json')]
    cps=[a for a in acks if a['object']==f'checkpoint-commits/objects/cp-{digest}.pt']
    models=[a for a in acks if a['object']=='model.pt']
    if len(cps)!=1 or len(models)!=1 or models[0]['sha256']!=m['model_sha256']:raise ValueError('Exact endpoint checkpoint/model ACK required')
    saved=torch.load(io.BytesIO(verify_archive(archive,cps[0],collect=True)),map_location='cpu',weights_only=False)
    model=torch.load(io.BytesIO(verify_archive(archive,models[0],collect=True)),map_location='cpu',weights_only=True)
    total=checkpoint_decisions(saved);rows=saved['updates'];new=rows[5:]
    parent=read_json(root/'runs/20261027T094058Z_tr10-original-second-actual-counter_2063d8/remote-results/training-updates.json')
    if rows[:5]!=parent or sum(update_decisions(r) for r in parent)!=32782:raise ValueError('Inherited optimizer history changed')
    increment=sum(update_decisions(r) for r in new)
    if (total!=m['unique_PPO_trained_decisions'] or total-32782!=increment
            or increment!=m['new_actual_unique_PPO_decisions'] or increment<16384
            or len(new)!=m['new_PPO_updates'] or saved['version']!=m['updates']):
        raise ValueError('Unique new optimizer coverage differs from terminal checkpoint/metrics')
    ids=[i for r in new for i in r['series_ids']]
    if not ids or min(ids)<222 or len(ids)!=len(set(ids)):raise ValueError('New training reuses inherited or duplicate BO3 identities')
    if len(saved['model'])!=127 or len(saved['optimizer']['state'])!=97 or set(model)!=set(saved['model']):raise ValueError('Full endpoint state missing')
    for state in saved['optimizer']['state'].values():
        if any(torch.is_tensor(value) and not torch.isfinite(value).all() for value in state.values()):
            raise ValueError('Nonfinite committed optimizer state')
    if any(not torch.equal(v,model[k]) or not torch.isfinite(v).all() for k,v in saved['model'].items()):raise ValueError('Endpoint model differs from full committed state')
    if set(saved['rng'])!={'python','numpy','torch','cuda'} or len(saved['rng']['cuda'])!=1 or 'cpu_sampler_rng' not in saved:
        raise ValueError('Full global/CUDA/CPU sampler RNG state missing')
    gradients=[];steps=0
    for row in new:
        successful=[st for st in row['steps'] if not st['skipped_for_kl']];steps+=len(successful)
        if successful:
            g=successful[0].get('policy_gradient_l2')
            gradients.append({'version':row['version'],'policy_gradient_l2':g,
                'passed':g is not None and math.isfinite(g) and g>0})
            if any(not math.isfinite(st['grad_norm']) for st in successful):raise ValueError('Nonfinite optimizer gradient')
    old=read_json(root/config['ledger_path']);assert not any(r['hypothesis_id']=='TR11' for r in old['families'])
    proposal=copy.deepcopy(old);family={'hypothesis_id':'TR11','canonical_training_family':config['training_family_members'][0],
        'training_family_members':config['training_family_members'],'scientific_status':'pending_predeclared_diagnostic',
        'trained_unique_decisions':increment,'PPO_updates':len(new),'nonzero_coverage_updates':sum(update_decisions(r)>0 for r in new),
        'successful_optimizer_steps':steps,'inherited_parent_decisions_not_counted_again':32782,
        'legacy_consumed_decisions':saved['trained_decisions']-35422,
        'unused_by_optimizer_decisions':saved['trained_decisions']-35422-increment,
        'actual_counter_accounting':ACCOUNTING,'source_run':source.name,'endpoint_checkpoint_sha256':digest,
        'new_complete_BO3_this_attempt':m['new_complete_BO3'],'new_complete_games_this_attempt':m['new_complete_games'],
        'new_sampled_decisions_this_attempt':m['new_sampled_decisions'],'promotion_eligible':False}
    proposal['families'].append(family)
    proposal.update(at=utcnow(),actual_unique_training_decisions=old['actual_unique_training_decisions']+increment,
        remaining_unique_training_decisions=max(0,old['goal_unique_actual_PPO_training_decisions']-old['actual_unique_training_decisions']-increment),
        total_committed_PPO_updates=old['total_committed_PPO_updates']+len(new),
        known_PPO_updates_lower_bound=old['total_committed_PPO_updates']+len(new),
        scope='Canonical actual committed unique PPO decisions throughTR11; only the new successful first-epoch coverage is added.',
        decision_accounting=ACCOUNTING,training_amount_is_not_strength_evidence=True)
    atomic_json(output/'CURRENT_VOLUME.proposed.json',proposal)
    atomic_json(output/'metrics.json',{'at':utcnow(),'complete':True,'engineering_only':True,
        'training_run_id':source.name,'parent_decisions_excluded':32782,'new_actual_unique_PPO_decisions':increment,
        'new_PPO_updates':len(new),'successful_optimizer_steps':steps,'new_update_gradients':gradients,
        'policy_gradient_health_passed':bool(gradients) and all(g['passed'] for g in gradients),
        'full_model_optimizer_global_CUDA_CPU_sampler_RNG_present':True,'model_bitwise_matches_checkpoint':True,
        'total_project_actual_unique_PPO_decisions':proposal['actual_unique_training_decisions'],
        'ledger_before':{'path':config['ledger_path'],'sha256':config['input_evidence'][config['ledger_path']]},
        'proposal':{'path':(output/'CURRENT_VOLUME.proposed.json').relative_to(root).as_posix(),'sha256':sha256(output/'CURRENT_VOLUME.proposed.json')},
        'new_training_during_audit':False,'promotion_eligible':False})
