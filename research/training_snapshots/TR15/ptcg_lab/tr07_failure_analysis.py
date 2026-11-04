"""Read-only audit of the terminal TR07 CPU sampling attempt and committed state."""
import math
import os
from pathlib import Path
import time
import numpy as np
import torch
from .io import atomic_json,read_json,sha256,utcnow
from .checkpoint_refs import read_index
from .archive_prefix import load_prefix,compact_record
from .archive_state import validate_state
from .resource_journal import decode
from .tensor_io import load_torch


def run(config,output):
    output=Path(output);root=Path(config['project_path']).resolve()
    if os.environ.get('PTCG_JOB_RUN_ID')!=output.name:raise RuntimeError('Forensics requires controlled runner')
    torch.set_num_threads(2);deadline=float(os.environ['PTCG_JOB_DEADLINE_UNIX'])
    archive=root/'runs'/config['archive_run_id'];source=root/'runs'/config['source_run_id'];raw=archive/'archive'
    if archive.parent!=root/'runs' or source.parent!=root/'runs':raise ValueError('Forensic source escaped')
    if sha256(archive/'file-receipts.json')!=config['archive_receipts_sha256']:raise ValueError('Archive receipts changed')
    metrics=read_json(archive/'metrics.json');receipts=read_json(archive/'file-receipts.json')
    if not metrics.get('complete') or not metrics.get('all_files_hash_verified_and_fsynced_locally') or metrics['source_run_id']!=source.name:
        raise ValueError('Complete immutable raw archive required')
    for row in receipts:
        path=raw/row['path']
        if raw not in path.resolve().parents or path.stat().st_size!=row['bytes'] or sha256(path)!=row['sha256']:
            raise ValueError('Full raw forensic archive changed')
        if time.time()>deadline-20:raise TimeoutError('Forensic verification deadline')
    lease=read_json(source/'remote-job.json');terminal=read_json(source/'remote-terminal.json')
    if terminal.get('terminal_proof_kind')!='exact_invocation_systemd_journal' or terminal['journal_terminal_proof']['Result']!='exit-code':
        raise ValueError('Exact positive failed invocation evidence required')
    index=read_index(raw,source.name,lease['lease_epoch']);point=index['commits'][-1]
    if point['checkpoint']['sha256']!=config['published_checkpoint_sha256'] or point['cursor']!=config['expected_cursor']:
        raise ValueError('Published checkpoint differs; inspect rather than infer')
    saved=load_torch(raw/point['checkpoint']['path'],map_location='cpu',weights_only=False)
    if {k:saved[k] for k in point['cursor']}!=point['cursor'] or len(saved['model'])!=127 or len(saved['optimizer']['state'])!=97:
        raise ValueError('Full committed model/optimizer/cursor differs')
    if set(saved['rng'])!={'python','numpy','torch','cuda'} or len(saved['rng']['cuda'])!=1:
        raise ValueError('Missing original complete RNG')
    for value in saved['model'].values():
        if not torch.isfinite(value).all():raise ValueError('Nonfinite committed model')
    for slots in saved['optimizer']['state'].values():
        for value in slots.values():
            if torch.is_tensor(value) and not torch.isfinite(value).all():raise ValueError('Nonfinite optimizer state')
    prefix=load_prefix(raw/point['prefix']['path'],point['prefix']['sha256'])
    state=validate_state(saved['archive_state'],prefix,point['prefix']['sha256'])
    parent=root/'agents/frozen/ppo-zero-anchor/model.pt'
    if sha256(parent)!=config['parent_model_sha256']:raise ValueError('Original full H002 parent changed')
    original={k.removeprefix('module.'):v for k,v in load_torch(parent,map_location='cpu',weights_only=True).items()}
    if set(original)!=set(saved['model']):raise ValueError('Full model keys changed')
    changed=[k for k,v in saved['model'].items() if not torch.equal(v,original[k])]
    from .rl_core import UNUSED,RLFeatures
    if any(k.startswith(UNUSED) for k in changed):raise ValueError('Frozen hidden/auxiliary model branch changed')
    features=RLFeatures(parent.parent);lo,hi=features.b.OFFSETS['oracle']
    summaries=[];decisions=0;terminals=0;stream_count=0;published_pending_decisions=0
    published_pending_ids={r['series_id'] for r in state['live_expected']}
    for path in sorted((raw/'series').glob('[0-9]'*6+'.json')):
        summary=read_json(path);sid=int(path.stem)
        if summary['series_id']!=sid or summary['statuses']!=['DONE','DONE'] or summary['rounds'] not in (2,3):
            raise ValueError('Only complete official BO3 accepted')
        trajectory=raw/summary['trajectory_path']
        if sha256(trajectory)!=summary['trajectory_sha256']:raise ValueError('Summary/trajectory hash differs')
        data=load_torch(trajectory,map_location='cpu',weights_only=False)
        if data['series_id']!=sid or data['policy_version']!=saved['version']:raise ValueError('Wrong sampling identity/version')
        for d in data['decisions']:
            if d['series_id']!=sid or int(d['policy_version'])!=saved['version'] or not math.isfinite(d['old_logp']) or not math.isfinite(d['old_value']):
                raise ValueError('Mixed or nonfinite actor decisions')
            for step in d['steps']:
                if step['rows'][:,lo:hi].any() or not np.isfinite(step['rows']).all():raise ValueError('Actor oracle leakage/nonfinite features')
        stream=data.get('cpu_sampler_rng_after')
        if stream is None or stream.dtype!=torch.uint8 or stream.ndim!=1:raise ValueError('CPU stream missing from complete trajectory')
        torch.Generator(device='cpu').set_state(stream)
        stream_count+=1;decisions+=len(data['decisions']);terminals+=sum(bool(d['terminal']) for d in data['decisions'])
        if sid in published_pending_ids:published_pending_decisions+=len(data['decisions'])
        summaries.append(compact_record(summary,sha256(path)))
        if time.time()>deadline-20:raise TimeoutError('Forensic trajectory deadline')
    by_id={r['series_id']:r for r in summaries}
    if any(by_id.get(r['series_id'])!=r for r in state['live_expected']):raise ValueError('Published pending history differs')
    if sum(x['decisions'] for x in saved['updates'])!=saved['trained_decisions'] or len(saved['updates'])!=saved['version']:
        raise ValueError('Committed PPO accounting differs')
    progress=read_json(raw/'progress.json')
    if progress.get('trained_decisions')!=saved['trained_decisions'] or progress.get('pending_decisions')!=decisions:
        raise ValueError('Actual progress differs from audited complete decision input')
    if point['cursor']['trained_decisions']>=32768:raise ValueError('Unexpected completed training must be independently evaluated')
    attempts=[a for rid in config['original_training_run_ids'] for a in read_json(root/'runs'/rid/'attempts.json')]
    if len(attempts)!=2 or not all(a['status']=='failed' for a in attempts):raise ValueError('Both original attempts must remain consumed')
    samples=[decode(line) for line in (raw/'resource-samples.jsonl').read_text().splitlines() if line]
    tensors=[]
    for row in receipts:
        if row['path']=='checkpoint.pt' or row['path'].startswith('checkpoint-commits/objects/') and row['path'].endswith('.pt'):
            cp=load_torch(raw/row['path'],map_location='cpu',weights_only=False)
            tensors.append({'path':row['path'],'sha256':row['sha256'],'cursor':{k:cp[k] for k in ['version','trained_decisions','next_series_id']},
                            'model_tensors':len(cp['model']),'optimizer_states':len(cp['optimizer']['state'])})
            if cp['trained_decisions']!=saved['trained_decisions'] or cp['version']!=saved['version']:
                raise ValueError('A different hot model exists; inspect rather than omit it')
    result={'at':utcnow(),'complete':True,'engineering_only':True,'hypothesis_id':'TR07','source_run_id':source.name,
        'classification':'invalid_incomplete_training_target_and_attempts_exhausted','promotion_eligible':False,
        'full_archive_files_verified':len(receipts),'full_archive_raw_bytes':sum(r['bytes'] for r in receipts),
        'committed_model_cursor':point['cursor'],'published_checkpoint_sha256':point['checkpoint']['sha256'],
        'target_decisions':32768,'trained_decisions':saved['trained_decisions'],'committed_PPO_updates':saved['version'],
        'new_complete_CPU_sampled_BO3':len(summaries),'new_untrained_complete_decisions':decisions,
        'new_terminal_round_transitions':terminals,'CPU_sampler_streams_verified':stream_count,
        'published_pending_BO3':len(published_pending_ids),'published_pending_decisions':published_pending_decisions,
        'post_checkpoint_complete_BO3':len(summaries)-len(published_pending_ids),
        'post_checkpoint_complete_decisions':decisions-published_pending_decisions,
        'inherited_complete_BO3_history':len(prefix['consumed'])+len(prefix['tail']),
        'full_model_tensors':127,'optimizer_states':97,'all_committed_model_and_optimizer_tensors_finite':True,
        'complete_RNG_fields_and_own_CPU_stream_retained':True,'actor_information_isolation_passed':True,
        'changed_inherited_tensor_names':changed,'all_checkpoint_payloads':tensors,
        'unpublished_hot_checkpoint_not_adopted':True,'strength_effect':None,
        'attempts':{'count':2,'records':attempts,'total_accounted_seconds':sum(a['seconds'] for a in attempts),'remaining':0,
                    'second_basis':'Full7200s authorizedcap conservatively charged after local controller loss; not actual measured runtime.'},
        'local_failure_evidence_sha256':sha256(source/'controller-failure-local.json'),
        'exact_terminal_journal_proof':terminal['journal_terminal_proof'],
        'cgroup_peak_bytes':max(int(x['kernel']['memory.peak']) for x in samples),
        'resource_samples':len(samples),'resource_observations_within_RAM4GiB':all(int(x['kernel']['memory.peak'])<=4*2**30 for x in samples),
        'original_raw_sources_unchanged':True,'next_action':'Close invalidTR07; no third or renamed continuation. Preserve all costs, rawdata, stops and validstreak3. Only a genuinely distinct preregistered hypothesis may train again.'}
    atomic_json(output/'analysis.json',result);atomic_json(output/'metrics.json',result)
