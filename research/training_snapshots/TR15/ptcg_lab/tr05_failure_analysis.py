"""Read-only full-data analysis of the exhausted first TR05 experiment."""
from collections import Counter
import math
import os
from pathlib import Path
import time
import numpy as np
import torch
from .io import read_json,atomic_json,sha256,utcnow
from .checkpoint_refs import read_index
from .archive_prefix import load_prefix,compact_record
from .archive_state import validate_state
from .resource_journal import decode


def run(config,output):
    output=Path(output);root=Path(config['project_path']).resolve()
    if os.environ.get('PTCG_JOB_RUN_ID')!=output.name:raise RuntimeError('Forensics requires controlled runner')
    torch.set_num_threads(2);deadline=float(os.environ['PTCG_JOB_DEADLINE_UNIX'])
    archive=root/'runs'/config['archive_run_id'];source=root/'runs'/config['source_run_id'];raw=archive/'archive'
    if archive.parent!=root/'runs' or source.parent!=root/'runs':raise ValueError('Forensic source escaped')
    if sha256(archive/'file-receipts.json')!=config['archive_receipts_sha256']:raise ValueError('Archive receipts changed')
    metrics=read_json(archive/'metrics.json');receipts=read_json(archive/'file-receipts.json')
    if not metrics.get('complete') or metrics['source_run_id']!=source.name:raise ValueError('Complete source archive required')
    for row in receipts:
        path=raw/row['path']
        if raw not in path.resolve().parents or path.stat().st_size!=row['bytes'] or sha256(path)!=row['sha256']:
            raise ValueError('Full raw forensic archive changed')
        if time.time()>deadline-20:raise TimeoutError('Forensic verification deadline')
    lease=read_json(source/'remote-job.json');index=read_index(raw,source.name,lease['lease_epoch']);point=index['commits'][-1]
    if point['checkpoint']['sha256']!=config['published_checkpoint_sha256'] or point['cursor']!=config['expected_cursor']:
        raise ValueError('Published source checkpoint differs; reinspect instead of guessing')
    saved=torch.load(raw/point['checkpoint']['path'],map_location='cpu',weights_only=False)
    if {k:saved[k] for k in point['cursor']}!=point['cursor']:raise ValueError('Payload cursor differs from published point')
    if len(saved['model'])!=127 or len(saved['optimizer']['state'])!=97 or set(saved['rng'])!={'python','numpy','torch','cuda'} or len(saved['rng']['cuda'])!=1:
        raise ValueError('Full trained model/optimizer/RNG coverage differs')
    for value in saved['model'].values():
        if not torch.isfinite(value).all():raise ValueError('Nonfinite committed model')
    for slots in saved['optimizer']['state'].values():
        for value in slots.values():
            if torch.is_tensor(value) and not torch.isfinite(value).all():raise ValueError('Nonfinite committed optimizer')
    prefix=load_prefix(raw/point['prefix']['path'],point['prefix']['sha256'])
    state=validate_state(saved['archive_state'],prefix,point['prefix']['sha256'])
    parent=root/'agents/frozen/ppo-zero-anchor/model.pt'
    if sha256(parent)!=config['parent_model_sha256']:raise ValueError('Full original parent changed')
    original={k.removeprefix('module.'):v for k,v in torch.load(parent,map_location='cpu',weights_only=True).items()}
    if set(original)!=set(saved['model']):raise ValueError('Full model state keys changed')
    changed=[k for k,v in saved['model'].items() if not torch.equal(v,original[k])]
    from .rl_core import UNUSED
    if any(k.startswith(UNUSED) for k in changed):raise ValueError('Frozen hidden/auxiliary model branch changed')
    from .rl_core import RLFeatures
    features=RLFeatures(parent.parent);lo,hi=features.b.OFFSETS['oracle']
    summaries=[];decisions=0;terminal_transitions=0
    for path in sorted((raw/'series').glob('[0-9]'*6+'.json')):
        summary=read_json(path);sid=int(path.stem)
        if summary['series_id']!=sid or summary['statuses']!=['DONE','DONE'] or summary['rounds'] not in (2,3):
            raise ValueError('Only complete official BO3 may enter forensic sample counts')
        trajectory=raw/summary['trajectory_path']
        if sha256(trajectory)!=summary['trajectory_sha256']:raise ValueError('Summary trajectory hash differs')
        data=torch.load(trajectory,map_location='cpu',weights_only=False)
        if data['series_id']!=sid or data['policy_version']!=0:raise ValueError('Original TR05 sample identity changed')
        for d in data['decisions']:
            if d['series_id']!=sid or int(d['policy_version'])!=0 or not math.isfinite(d['old_logp']) or not math.isfinite(d['old_value']):
                raise ValueError('Mixed/nonfinite training decision')
            for step in d['steps']:
                if step['rows'][:,lo:hi].any():raise ValueError('Hidden oracle values in actor features')
                if not np.isfinite(step['rows']).all():raise ValueError('Nonfinite actor features')
        decisions+=len(data['decisions']);terminal_transitions+=sum(bool(d['terminal']) for d in data['decisions'])
        summaries.append(compact_record(summary,sha256(path)))
    if summaries!=state['archived']+state['live_expected']:raise ValueError('Complete BO3 history differs from checkpoint archive state')
    if decisions!=saved['trained_decisions'] or len(saved['updates'])!=saved['version']:raise ValueError('Actual PPO input/update accounting differs')
    update=saved['updates'][-1]
    if update['decisions']!=decisions or update['series_ids']!=[r['series_id'] for r in summaries] or update['terminal_rounds']!=terminal_transitions:
        raise ValueError('Published PPO update omitted/duplicated complete series')
    steps=update['steps'];numeric={}
    for key in ['loss','policy_loss','value_loss','entropy','approx_kl','clip_fraction','grad_norm']:
        values=[float(x[key]) for x in steps if key in x]
        if not all(math.isfinite(x) for x in values):raise ValueError('Nonfinite PPO update metric')
        numeric[key]={'count':len(values),'min':min(values),'mean':float(np.mean(values)),'max':max(values)} if values else None
    samples=[decode(line) for line in (raw/'resource-samples.jsonl').read_text().splitlines() if line]
    failure=read_json(raw/'failure.json')
    if failure['error']!='OSError: Bounded resource log full':raise ValueError('Unexpected failure requires a new causal analysis')
    first=root/'runs'/read_json(source/'config.json')['original_training_run_id']
    attempts=[*read_json(first/'attempts.json'),*read_json(source/'attempts.json')]
    if len(attempts)!=2:raise ValueError('Reinspect original attempt count')
    bounds={'RAM_within4GiB':all(int(x['kernel']['memory.peak'])<=4*2**30 for x in samples),
        'workspace_within10GiB':failure['peak_workspace_bytes']<=10*2**30,
        'free_reserve_at_least20GiB':failure['minimum_free_bytes']>=20*2**30}
    result={'at':utcnow(),'complete':True,'engineering_only':True,'source_run_id':source.name,
        'classification':'invalid_incomplete_training_target_and_attempts_exhausted',
        'full_archive_files_verified':len(receipts),'full_archive_raw_bytes':sum(r['bytes'] for r in receipts),
        'committed_model_cursor':point['cursor'],'published_checkpoint_sha256':point['checkpoint']['sha256'],
        'unpublished_hot_or_partial_checkpoint_not_adopted':True,'target_decisions':32768,
        'missing_trained_decisions':32768-saved['trained_decisions'],'committed_PPO_updates':saved['version'],
        'complete_BO3':len(summaries),'trained_decisions':decisions,'terminal_rounds':terminal_transitions,
        'full_model_tensors':len(saved['model']),'optimizer_states':len(saved['optimizer']['state']),
        'all_committed_model_and_optimizer_tensors_finite':True,'full_RNG_fields_retained':True,
        'changed_inherited_actor_tensor_names':[k for k in changed if not k.startswith('value_mlp.')],
        'changed_value_head_tensor_names':[k for k in changed if k.startswith('value_mlp.')],
        'value_comparison_includes_declared_initial_last_layer_zero_reset':True,
        'actor_oracle_checks_passed':True,'PPO_update_metrics':numeric,'PPO_minibatch_rows':len(steps),
        'KL_skipped_rows':sum(bool(x.get('skipped_for_kl')) for x in steps),
        'attempts':{'count':2,'records':attempts,'total_seconds':sum(x['seconds'] for x in attempts),'remaining':0},
        'failure':failure,'resource_samples':len(samples),'actual_resource_log_bytes':(raw/'resource-samples.jsonl').stat().st_size,
        'old_resource_log_byte_cap':512*1024,'resource_bounds_observed':bounds,
        'cgroup_peak_bytes':max(int(x['kernel']['memory.peak']) for x in samples),
        'own_GPU_peak_MiB':max(x.get('gpu',{}).get('own_service_process_gpu_memory_mib') or 0 for x in samples),
        'failure_causality':'Observed supervisor resource journal reached512KiB and raised before append. CPU/RAM/disk facts above are distinct from that journal cap; intermittent foreign GPU workloads caused cooperative waits, not the terminal exception. Original code, fees and failed attempts remain immutable.',
        'value_gradient_route_competitive_effect':None,'promotion_eligible':False,'original_raw_sources_unchanged':True,
        'next_action':'Close TR05 invalid without changing valid-hypothesis streak3. Verify future two-hour-sized journal and actual lower-IO checkpoint path before a genuinely distinct scientific hypothesis. No third/relabelled TR05 attempt.'}
    atomic_json(output/'analysis.json',result);atomic_json(output/'metrics.json',result)
