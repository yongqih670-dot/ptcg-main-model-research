"""CPU-only full-state and archive audit before each authorized TR12 segment."""
import copy
import io
import os
from pathlib import Path
import time
from .io import read_json,atomic_json,sha256,utcnow
from .ppo_accounting import checkpoint_decisions,update_decisions
from .archive_prefix import canonical_hash
from .volume_scope import BASE,TARGET,PARENT,PARENT_CP


def run(config,output):
    import numpy as np
    import torch
    from .archive_compressed import verify_archive
    from .rl_core import load_full_actor,evaluate_decisions
    from .tensor_io import atomic_torch_save
    from .volume_boundary import inspect
    root=Path(config['project_path']).resolve();output=Path(output).resolve()
    if os.environ.get('PTCG_JOB_RUN_ID')!=output.name or output.parent!=root/'runs':raise ValueError('Controlled CPU source audit required')
    deadline=float(os.environ['PTCG_JOB_DEADLINE_UNIX'])
    if not 0<deadline-time.time()<=1800.5 or torch.cuda.is_available():raise ValueError('Bounded CPU-only audit required')
    boundary=inspect(config,output)
    paths={k:root/v['path'] for k,v in boundary['source_files'].items()}
    index=read_json(paths['checkpoint-commits/index.json']);point=index['commits'][-1]
    if index['run_id']!=config['source_run_id'] or canonical_hash({k:v for k,v in point.items() if k!='commit_id'})!=point['commit_id']:
        raise ValueError('Full checkpoint owner or commit differs')
    if len(index['commits'])!=2:raise ValueError('Source must retain two committed recovery points')
    catalogue={}
    for rid in config['archive_source_run_ids']:
        source=root/'runs'/rid;archive=source/'online-archive'
        if not (archive/rid/'acks').exists():continue
        for path in sorted((archive/rid/'acks').glob('*.json')):
            ack=read_json(path)
            if ack['run_id']!=rid:raise ValueError('Archive receipt ownership differs')
            catalogue.setdefault(ack['sha256'],(archive,ack,path))
    selected={}
    def original(digest,collect=False):
        if time.time()>deadline-30:raise TimeoutError('Full-state source audit deadline')
        if digest not in catalogue:raise ValueError('Required original bytes are not fully archived: '+digest)
        archive,ack,path=catalogue[digest]
        selected[digest]={'archive_root':archive.relative_to(root).as_posix(),'ack':{'path':path.relative_to(root).as_posix(),'sha256':sha256(path)},
            'object_sha256':digest,'raw_bytes':ack['bytes'],'archive_relative':ack['archive_relative']}
        return verify_archive(archive,ack,collect=collect)
    saved=torch.load(io.BytesIO(original(point['checkpoint']['sha256'],True)),map_location='cpu',weights_only=False)
    original(point['prefix']['sha256'])
    actual=checkpoint_decisions(saved)
    if not BASE<=actual<TARGET or len(saved['model'])!=127 or len(saved['optimizer']['state'])!=97:
        raise ValueError('Source full state is outside the frozen unfinished target')
    if {k:saved[k] for k in point['cursor']}!=point['cursor']:raise ValueError('Checkpoint cursor differs')
    if config['source_run_id']==PARENT and (actual!=BASE or saved['version']!=8 or point['pending'] or point['checkpoint']['sha256']!=PARENT_CP):
        raise ValueError('Initial source is not the complete authorized TR11 endpoint')
    prior=read_json(root/'runs'/PARENT/'remote-results/training-updates.json')
    if saved['updates'][:8]!=prior:raise ValueError('Inherited eight PPO updates changed')
    if any(not torch.isfinite(v).all() for v in saved['model'].values()):raise ValueError('Nonfinite source model')
    for state in saved['optimizer']['state'].values():
        if any(torch.is_tensor(v) and not torch.isfinite(v).all() for v in state.values()):raise ValueError('Nonfinite optimizer state')
    if set(saved['rng'])!={'python','numpy','torch','cuda'} or len(saved['rng']['cuda'])!=1 or 'cpu_sampler_rng' not in saved:
        raise ValueError('Full global,CUDA and CPU actor RNG required')
    pending_decisions=0
    for reference in point['pending']:
        data=original(reference['sha256'],True)
        if len(data)!=reference['bytes']:raise ValueError('Pending raw bytes differ')
        if reference['path'].startswith('rollouts/'):
            shard=torch.load(io.BytesIO(data),map_location='cpu',weights_only=False)
            if shard['summary']['statuses']!=['DONE','DONE']:raise ValueError('Partial BO3 cannot enter pending optimizer data')
            pending_decisions+=len(shard['decisions'])
    progress=read_json(paths['progress.json']);tail=None
    if config.get('interrupted_recovery_report'):
        proof=config['interrupted_recovery_report'];rp=root/proof['path']
        if sha256(rp)!=proof['sha256']:raise ValueError('Interrupted recovery proof changed')
        report=read_json(rp)
        if (not report['complete'] or report['source_run_id']!=config['source_run_id'] or report['source_point']!=point
                or report['unique_PPO_trained_decisions']!=actual or report['optimizer_steps_executed']!=0):
            raise ValueError('Full interrupted recovery audit is not this exact source')
        raw=root/report['recovery_source_root']
        for name,item in boundary['source_files'].items():
            if sha256(raw/name)!=item['sha256']:raise ValueError('Terminal source changed since full forensic recovery')
        refs=[]
        for name in [f"rollouts/v{saved['version']:04d}-s{report['complete_tail_series_id']:06d}.pt",f"series/{report['complete_tail_series_id']:06d}.json"]:
            p=raw/name;digest=sha256(p);original(digest)
            refs.append({'path':name,'sha256':digest,'bytes':p.stat().st_size})
        tail={'source_run_id':config['source_run_id'],'checkpoint_sha256':point['checkpoint']['sha256'],
            'references':refs,'series_ids':[report['complete_tail_series_id']],'pending_decisions':report['pending_decisions'],
            'journal_next_series_id':report['complete_tail_series_id']+1,
            'next_series_id_after_interruption_exclusion':report['next_series_id_after_interruption_exclusion'],
            'inflight_sha256':boundary['source_files']['inflight.json']['sha256'],'recovery_report':proof}
        from .volume_tail import validate_spec,verify_journal
        validate_spec(tail,config['source_run_id'],point)
        verify_journal(raw,saved,dict(prefix_path=point['prefix']['path'],prefix_sha256=point['prefix']['sha256'],
            initial_checkpoint_sha256=read_json(raw/point['prefix']['path'])['checkpoint']['sha256']),point['checkpoint']['sha256'],tail)
        pending_decisions=tail['pending_decisions']
    expected_cursor={**point['cursor'],'next_series_id':tail['journal_next_series_id']} if tail else point['cursor']
    if any(progress.get(k)!=v for k,v in expected_cursor.items()) or progress.get('pending_decisions')!=pending_decisions:
        raise ValueError('Terminal progress differs from latest complete checkpoint; forensic tail reconciliation required')
    if progress.get('control_stop_reason')=='two_consecutive_zero_coverage_updates':
        raise ValueError('Scientific two-zero-coverage stopping rule reached; additional volume cannot override it')
    league=read_json(paths['league.json'])
    for row in league['snapshots']:original(row['sha256'])
    fixture_ref=config['recovery_fixture_source'];fixture_path=root/fixture_ref['path']
    if sha256(fixture_path)!=fixture_ref['sha256']:raise ValueError('Fixed recovery-only observation fixture changed')
    source_fixture=torch.load(fixture_path,map_location='cpu',weights_only=False)
    fixture=copy.deepcopy(source_fixture['fixture']);assert len(fixture)==48
    torch.set_num_threads(2)
    model,features,_=load_full_actor(root/'agents/frozen/ppo-zero-anchor',config['origin_actor_sha256'],'cpu',reset_value=False)
    model.load_state_dict(saved['model'],strict=True)
    lo,hi=features.b.OFFSETS['oracle']
    if any(step['rows'][:,lo:hi].any() for d in fixture for step in d['steps']):raise ValueError('Hidden fixture features')
    with torch.no_grad():logp,values,_=evaluate_decisions(model,fixture,features.b.STOP)
    if not torch.isfinite(logp).all() or not torch.isfinite(values).all():raise ValueError('Nonfinite current-model fixture')
    for d,p,v in zip(fixture,logp.tolist(),values.tolist()):
        d.update(old_logp=p,old_value=float(np.asarray(v).reshape(-1)[0]),policy_version=saved['version'],recovery_fixture_only=True)
    target=output/'current-model-numeric-reference.pt'
    atomic_torch_save(target,{'model':saved['model'],'fixture':fixture,'fixture_logp':logp.cpu(),'fixture_values':values.cpu(),
        'not_training_data':True,'source_run_id':config['source_run_id'],'source_checkpoint_sha256':point['checkpoint']['sha256']},replace=False)
    atomic_json(output/'verified-archive-inventory.json',{'at':utcnow(),'objects':list(selected.values()),'only_relevant_full_hashes_verified':True})
    atomic_json(output/'metrics.json',{'at':utcnow(),'complete':True,'engineering_only':True,'source_run_id':config['source_run_id'],
        'source_point':point,'source_remote_evidence':{k:v['sha256'] for k,v in boundary['source_files'].items()},
        'source_league_sha256':boundary['source_files']['league.json']['sha256'],
        'source_inflight':({k:boundary['source_files']['inflight.json'][k] for k in ('sha256','bytes')} if 'inflight.json' in boundary['source_files'] else None),
        'actual_unique_PPO_decisions':actual,'PPO_updates':saved['version'],'pending_decisions':pending_decisions,
        'complete_postcheckpoint_tail':tail,
        'all_full_state_and_pending_archives_verified':True,'reference_fixture_never_enters_training':True,
        'optimizer_steps_executed':0,'new_sampled_decisions':0,'new_unique_training_decisions':0,
        'numeric_reference':{'path':target.relative_to(root).as_posix(),'sha256':sha256(target),'bytes':target.stat().st_size},
        'archive_inventory':{'path':(output/'verified-archive-inventory.json').relative_to(root).as_posix(),'sha256':sha256(output/'verified-archive-inventory.json')},
        'boundary_receipt':{'path':(output/'boundary-and-source-receipt.json').relative_to(root).as_posix(),'sha256':sha256(output/'boundary-and-source-receipt.json')},
        'previous_updates_and_actual_volume_not_recounted':True})
