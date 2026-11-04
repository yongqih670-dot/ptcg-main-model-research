"""Read-only full evidence audit before TR08's one original continuation."""
import math
import os
from pathlib import Path
import time

from .io import atomic_json,read_json,sha256,utcnow


def run(config,output):
    import numpy as np
    import torch
    from .archive_prefix import load_prefix,compact_record
    from .archive_state import validate_state
    from .checkpoint_refs import read_index
    from .rl_core import RLFeatures
    from .tensor_io import load_torch
    root=Path(config['project_path']).resolve();output=Path(output)
    if os.environ.get('PTCG_JOB_RUN_ID')!=output.name:raise RuntimeError('Controlled local audit required')
    torch.set_num_threads(2);deadline=float(os.environ['PTCG_JOB_DEADLINE_UNIX'])
    if deadline-time.time()+config.get('prior_forensic_seconds',0)>240.5:raise ValueError('Original240s forensic scope exceeded')
    archive=root/'runs'/config['archive_run_id'];raw=archive/'archive'
    source=root/'runs'/'20261026T085757Z_tr08-actor-lr3e6-CPU-sampling_868ff0'
    if archive.resolve().parent!=root/'runs':raise ValueError('Archive scope escaped')
    if sha256(archive/'file-receipts.json')!=config['archive_receipts_sha256']:raise ValueError('Archive receipt changed')
    metrics=read_json(archive/'metrics.json');rows=read_json(archive/'file-receipts.json')
    if not metrics.get('complete') or not metrics.get('all_files_hash_verified_and_fsynced_locally') or metrics['source_run_id']!=source.name:
        raise ValueError('Complete terminal archive required')
    receipts={}
    for row in rows:
        path=raw/row['path']
        if raw.resolve() not in path.resolve().parents or path.stat().st_size!=row['bytes'] or sha256(path)!=row['sha256']:
            raise ValueError('Archived original bytes differ')
        receipts[row['path']]={k:row[k] for k in ['path','bytes','sha256']}
        if time.time()>deadline-15:raise TimeoutError('Full archive audit deadline')
    terminal=read_json(source/'remote-terminal.json');lease=read_json(source/'remote-job.json')
    if terminal.get('terminal_proof_kind')!='exact_invocation_systemd_journal' or terminal['journal_terminal_proof']['Result']!='exit-code':
        raise ValueError('Positive original failed invocation proof required')
    attempts=read_json(source/'attempts.json')
    if len(attempts)!=1 or attempts[0]['status']!='failed' or abs(attempts[0]['seconds']-5330.86)>.01:
        raise ValueError('Original first attempt cost changed')
    error=(source/'stderr-01.log').read_text(encoding='utf-8')
    if 'SSH response deadline for archive_chunk' not in error:raise ValueError('Different first failure')
    point=read_index(raw,source.name,lease['lease_epoch'])['commits'][-1]
    saved=load_torch(raw/point['checkpoint']['path'],map_location='cpu',weights_only=False)
    expected={'version':2,'trained_decisions':16523,'next_series_id':132}
    if point['cursor']!=expected or {k:saved[k] for k in expected}!=expected:raise ValueError('Published cursor differs')
    if len(saved['model'])!=127 or len(saved['optimizer']['state'])!=97 or set(saved['rng'])!={'python','numpy','torch','cuda'} or len(saved['rng']['cuda'])!=1:
        raise ValueError('Full model/optimizer/RNG coverage differs')
    for tensor in [*saved['model'].values(),*[v for slots in saved['optimizer']['state'].values() for v in slots.values() if torch.is_tensor(v)]]:
        if not torch.isfinite(tensor).all():raise ValueError('Nonfinite full state')
    if len(saved['updates'])!=2 or sum(u['decisions'] for u in saved['updates'])!=16523:raise ValueError('Committed update accounting differs')
    prefix=load_prefix(raw/point['prefix']['path'],point['prefix']['sha256'])
    state=validate_state(saved['archive_state'],prefix,point['prefix']['sha256'])
    features=RLFeatures(root/'agents/frozen/ppo-zero-anchor');lo,hi=features.b.OFFSETS['oracle']
    records=[];decisions=0;published_decisions=0;terminal_count=0;tail=[]
    for path in sorted((raw/'series').glob('[0-9]'*6+'.json')):
        summary=read_json(path)
        if int(summary['policy_version'])!=2:continue
        sid=summary['series_id'];trajectory=raw/summary['trajectory_path']
        if summary['statuses']!=['DONE','DONE'] or summary['rounds'] not in (2,3) or sha256(trajectory)!=summary['trajectory_sha256']:
            raise ValueError('Incomplete or changed BO3')
        data=load_torch(trajectory,map_location='cpu',weights_only=False)
        if data['series_id']!=sid or data['policy_version']!=2:raise ValueError('Current policy identity differs')
        for d in data['decisions']:
            if d['series_id']!=sid or int(d['policy_version'])!=2 or not math.isfinite(d['old_logp']) or not math.isfinite(d['old_value']):
                raise ValueError('Nonfinite or mixed decisions')
            for step in d['steps']:
                if step['rows'][:,lo:hi].any() or not np.isfinite(step['rows']).all():raise ValueError('Actor information isolation failed')
        stream=data.get('cpu_sampler_rng_after')
        if stream is None or stream.dtype!=torch.uint8 or stream.ndim!=1:raise ValueError('CPU sampler RNG absent')
        torch.Generator(device='cpu').set_state(stream)
        n=len(data['decisions']);decisions+=n;terminal_count+=sum(bool(d['terminal']) for d in data['decisions'])
        records.append(compact_record(summary,sha256(path)))
        if sid<132:published_decisions+=n
        else:tail.extend([receipts[summary['trajectory_path']],receipts[path.relative_to(raw).as_posix()]])
        if time.time()>deadline-15:raise TimeoutError('Complete BO3 audit deadline')
    if [r['series_id'] for r in records]!=list(range(104,141)) or decisions!=6015 or len(tail)!=18:
        raise ValueError('Actual complete current policy tail differs')
    by_id={r['series_id']:r for r in records}
    if any(by_id.get(r['series_id'])!=r for r in state['live_expected']):raise ValueError('Published pending history differs')
    progress=read_json(raw/'progress.json')
    if (progress.get('stage'),progress.get('trained_decisions'),progress.get('pending_decisions'),progress.get('next_series_id'))!=('collecting',16523,6015,141):
        raise ValueError('Terminal collecting state differs')
    checkpoints=[]
    for row in rows:
        if row['path']=='checkpoint.pt' or row['path'].startswith('checkpoint-commits/objects/cp-'):
            cp=load_torch(raw/row['path'],map_location='cpu',weights_only=False)
            if cp['version']!=2 or cp['trained_decisions']!=16523:raise ValueError('Different hot model needs review')
            checkpoints.append({**receipts[row['path']],'cursor':{k:cp[k] for k in expected}})
    league=read_json(raw/'league.json')
    for member in league['snapshots']:
        if sha256(raw/member['path'])!=member['sha256']:raise ValueError('Frozen historical opponent changed')
    result={'at':utcnow(),'complete':True,'engineering_only':True,'hypothesis_id':'TR08',
        'classification':'original_first_transport_failure_recoverable_second_attempt_only','promotion_eligible':False,
        'source_run_id':source.name,'archive_run_id':archive.name,'full_archive_files_verified':len(rows),
        'full_archive_raw_bytes':sum(r['bytes'] for r in rows),'source_point':point,
        'full_model_tensors':127,'optimizer_states':97,'trained_decisions':16523,'PPO_updates':2,
        'current_policy_complete_BO3':37,'pending_decisions':6015,'published_pending_decisions':published_decisions,
        'post_checkpoint_complete_BO3':9,'post_checkpoint_complete_decisions':decisions-published_decisions,
        'additional_complete_tail':tail,'complete_terminal_transitions':terminal_count,
        'CPU_sampler_streams_verified':37,'actor_information_isolation_passed':True,
        'all_full_model_optimizer_RNG_checkpoint_payloads_verified':True,'all_checkpoint_payloads':checkpoints,
        'all_frozen_historical_members_verified':league['snapshots'],
        'source_manifest_sha256':sha256(raw/'training-manifest.raw.json'),'source_league_sha256':sha256(raw/'league.json'),
        'source_progress_sha256':sha256(raw/'progress.json'),'source_payload_config_sha256':sha256(raw/'payload-config.json'),
        'source_index_sha256':sha256(raw/'checkpoint-commits/index.json'),
        'first_error_sha256':sha256(source/'stderr-01.log'),'first_attempt_seconds':attempts[0]['seconds'],
        'positive_terminal_proof':terminal['journal_terminal_proof'],'remaining_original_attempts':1,
        'strength_effect':None,'source_modified':False,'incomplete_BO3_excluded':True}
    atomic_json(output/'analysis.json',result);atomic_json(output/'metrics.json',result)
