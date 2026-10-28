"""Deduplicate the whole TR12 lineage and reconstruct only its fixed endpoint."""
import copy
import io
import json
import math
import os
from pathlib import Path
import shutil
import time
from .io import read_json,atomic_json,sha256,utcnow
from .ppo_accounting import checkpoint_decisions,update_decisions,ACCOUNTING
from .volume_scope import PARENT,BASE,TARGET

def run(config,output):
    import torch
    from .archive_compressed import verify_archive
    from .remote_lease import validate_no_unreconciled_remote
    root=Path(config['project_path']).resolve();output=Path(output).resolve()
    if os.environ.get('PTCG_JOB_RUN_ID')!=output.name or output.parent!=root/'runs':raise RuntimeError('Controlled audit required')
    deadline=float(os.environ['PTCG_JOB_DEADLINE_UNIX'])
    if not 0<deadline-time.time()<=1800.5:raise ValueError('Bounded1800s TR12 full archive audit required')
    torch.set_num_threads(2)
    for path,digest in config['input_evidence'].items():
        if sha256(root/path)!=digest:raise ValueError('Frozen result evidence changed')
    members=config['training_family_members'];source=root/'runs'/members[-1]
    state=read_json(root/'research/autonomy/state.json')
    if members!=state['forward_phases']['TR12']['training_run_ids']:raise ValueError('Must audit all original segments')
    m=read_json(source/'metrics.json');milestone=read_json(source/'remote-results/milestones/369941/manifest.json')
    if not m['complete'] or m['unique_PPO_trained_decisions']<TARGET:raise ValueError('Fixed endpoint not complete')
    digest=milestone['full_state']['checkpoint']['sha256'];catalog={};series={};receipts=0;model_ack=None
    def check():
        if time.time()>deadline-15:raise TimeoutError('Full archive audit deadline')
    for rid in members:
        run=root/'runs'/rid;lease=read_json(run/'remote-job.json');c=read_json(run/'config.json')
        validate_no_unreconciled_remote(root,{'remote_job':lease})
        if c['hypothesis_id']!='TR12' or c['check_variant']!='volume_extension':raise ValueError('Different family')
        archive=run/'online-archive'
        for path in sorted((archive/rid/'acks').glob('*.json')):
            check();ack=read_json(path);receipts+=1
            if ack['run_id']!=rid or ack['lease_epoch']!=lease['lease_epoch']:raise ValueError('Archive owner differs')
            is_series=ack['object'].startswith('series/') and ack['object'].endswith('.json')
            if ack['sha256'] not in catalog:
                raw=verify_archive(archive,ack,collect=is_series)
                catalog[ack['sha256']]={'archive':archive,'ack':ack,'raw':raw if is_series else None}
            else:
                previous=catalog[ack['sha256']]['ack']
                if ack['bytes']!=previous['bytes']:raise ValueError('Conflicting identical raw digest')
            if is_series:
                row=json.loads(catalog[ack['sha256']]['raw']);sid=row['series_id']
                if (row['event']!='series_end' or row['statuses']!=['DONE','DONE'] or row['rounds'] not in (2,3)
                    or ack['object']!=f'series/{sid:06d}.json' or sid<330):raise ValueError('Nonnew or incomplete BO3')
                if sid in series and series[sid]['sha256']!=ack['sha256']:raise ValueError('Conflicting duplicate BO3')
                series[sid]={'series_id':sid,'sha256':ack['sha256'],'games':row['rounds'],'sampled_decisions':row['learner_decisions']}
            if rid==source.name and ack['object']=='model.pt':
                if model_ack is not None:raise ValueError('Multiple endpoint models')
                model_ack=ack
        atomic_json(output/'progress.json',{'at':utcnow(),'stage':'full_archive_verified','source':rid,'unique_objects':len(catalog),'receipts':receipts})
    def payload(d):
        item=catalog[d];check();return verify_archive(item['archive'],item['ack'],collect=True)
    saved=torch.load(io.BytesIO(payload(digest)),map_location='cpu',weights_only=False)
    if model_ack is None or model_ack['sha256']!=m['model_sha256']:raise ValueError('Endpoint model ACK differs')
    model_bytes=payload(model_ack['sha256']);model=torch.load(io.BytesIO(model_bytes),map_location='cpu',weights_only=True)
    rows=saved['updates'];old_rows=read_json(root/'runs'/PARENT/'remote-results/training-updates.json')
    if rows[:8]!=old_rows or sum(update_decisions(x) for x in old_rows)!=BASE:raise ValueError('Inherited8updates changed')
    new=rows[8:];increment=sum(update_decisions(x) for x in new)
    if checkpoint_decisions(saved)!=BASE+increment or BASE+increment!=m['unique_PPO_trained_decisions'] or increment<320768:
        raise ValueError('New actual unique coverage differs')
    ids=[i for r in new for i in r['series_ids']]
    if not ids or min(ids)<330 or len(ids)!=len(set(ids)) or not set(ids)<=set(series):raise ValueError('Inherited or duplicate optimizer series')
    if len(saved['model'])!=127 or len(saved['optimizer']['state'])!=97 or set(model)!=set(saved['model']):raise ValueError('Incomplete full state')
    if any(not torch.equal(v,model[k]) or not torch.isfinite(v).all() for k,v in saved['model'].items()):raise ValueError('Model not exact')
    for slot in saved['optimizer']['state'].values():
        if any(torch.is_tensor(v) and not torch.isfinite(v).all() for v in slot.values()):raise ValueError('Nonfinite optimizer')
    if set(saved['rng'])!={'python','numpy','torch','cuda'} or len(saved['rng']['cuda'])!=1 or 'cpu_sampler_rng' not in saved:raise ValueError('RNG missing')
    gradients=[];steps=0
    for row in new:
        successful=[x for x in row['steps'] if not x['skipped_for_kl']];steps+=len(successful)
        if successful:
            g=successful[0].get('policy_gradient_l2');gradients.append({'version':row['version'],'policy_gradient_l2':g,
                'passed':g is not None and math.isfinite(g) and g>0})
            if any(not math.isfinite(x['grad_norm']) for x in successful):raise ValueError('Nonfinite gradient')
    check();policy=read_json(output/'policy.json')['budget']
    if shutil.disk_usage(root).free-len(model_bytes)<policy['disk_free_gib_min']*2**30:raise OSError('Physical reserve protected')
    target=source/'model.pt'
    if target.exists():
        if sha256(target)!=model_ack['sha256']:raise ValueError('Never overwrite model')
    else:
        temp=output/'model-verified.partial'
        with temp.open('xb') as f:f.write(model_bytes);f.flush();os.fsync(f.fileno())
        os.link(temp,target);temp.unlink()
    proof={'at':utcnow(),'raw_model_sha256':sha256(target),'model_archive_ack':model_ack,
        'all_archive_receipts_reconstructed_and_verified':receipts,'unique_full_hash_objects_verified':len(catalog),
        'training_family_members':members,'controlled_materialization_run_id':output.name,
        'no_model_training_or_weight_reencoding':True,'promotion_eligible':False}
    destination=source/'hybrid-materialization.json'
    if destination.exists():raise ValueError('Materialization already recorded')
    atomic_json(destination,proof)
    old=read_json(root/config['ledger_path'])
    if any(f['hypothesis_id']=='TR12' for f in old['families']):raise ValueError('TR12 already counted')
    proposal=copy.deepcopy(old);family={'hypothesis_id':'TR12','canonical_training_family':members[0],
        'training_family_members':members,'scientific_status':'pending_frozen_independent_evaluation',
        'trained_unique_decisions':increment,'PPO_updates':len(new),'successful_optimizer_steps':steps,
        'inherited_parent_decisions_not_counted_again':BASE,'actual_counter_accounting':ACCOUNTING,
        'source_run':source.name,'endpoint_checkpoint_sha256':digest,'promotion_eligible':False,
        'new_complete_BO3':len(series),'new_complete_games':sum(x['games'] for x in series.values()),
        'new_sampled_decisions':sum(x['sampled_decisions'] for x in series.values())}
    proposal['families'].append(family)
    proposal.update(at=utcnow(),actual_unique_training_decisions=old['actual_unique_training_decisions']+increment,
        remaining_unique_training_decisions=max(0,old['goal_unique_actual_PPO_training_decisions']-old['actual_unique_training_decisions']-increment),
        total_committed_PPO_updates=old['total_committed_PPO_updates']+len(new),
        known_PPO_updates_lower_bound=old['total_committed_PPO_updates']+len(new),
        scope='ThroughTR12: only new successful first-epoch coverage; inherited history and copied pending series deduplicated.',
        decision_accounting=ACCOUNTING,training_amount_is_not_strength_evidence=True)
    atomic_json(output/'CURRENT_VOLUME.proposed.json',proposal);atomic_json(output/'deduplicated-series.json',list(series.values()))
    atomic_json(output/'metrics.json',{'at':utcnow(),'complete':True,'engineering_only':True,
        'training_run_id':source.name,'training_family_members':members,'new_actual_unique_PPO_decisions':increment,
        'parent_decisions_excluded':BASE,'new_PPO_updates':len(new),'successful_optimizer_steps':steps,
        'new_update_gradients':gradients,'policy_gradient_health_passed':bool(gradients) and all(x['passed'] for x in gradients),
        'full_model_optimizer_global_CUDA_CPU_sampler_RNG_present':True,'model_bitwise_matches_checkpoint':True,
        'new_complete_BO3':family['new_complete_BO3'],'new_complete_games':family['new_complete_games'],
        'new_sampled_decisions':family['new_sampled_decisions'],'verified_unique_objects':len(catalog),
        'ledger_before':{'path':config['ledger_path'],'sha256':config['input_evidence'][config['ledger_path']]},
        'proposal':{'path':(output/'CURRENT_VOLUME.proposed.json').relative_to(root).as_posix(),'sha256':sha256(output/'CURRENT_VOLUME.proposed.json')},
        'new_training_during_audit':False,'promotion_eligible':False})
