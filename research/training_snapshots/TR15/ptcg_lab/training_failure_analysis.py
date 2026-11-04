"""Read-only CPU analysis of terminal training, including an orphan BO3 shard."""
from collections import Counter
from datetime import datetime
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import torch
from .io import atomic_json,read_json,sha256,utcnow


def run(config,output):
    output=Path(output);root=Path(config['project_path']).resolve()
    if os.environ.get('PTCG_JOB_RUN_ID')!=output.name:raise RuntimeError('Failure analysis requires controlled runner')
    torch.set_num_threads(2);deadline=float(os.environ['PTCG_JOB_DEADLINE_UNIX'])
    archive=root/'runs'/config['archive_run_id'];source=root/'runs'/config['source_run_id']
    if archive.parent!=root/'runs' or source.parent!=root/'runs':raise ValueError('Analysis source escaped')
    if sha256(archive/'file-receipts.json')!=config['archive_receipts_sha256']:raise ValueError('Archive receipts changed')
    m=read_json(archive/'metrics.json')
    if not m.get('complete') or m['source_run_id']!=source.name:raise ValueError('Full failure archive missing')
    receipts=read_json(archive/'file-receipts.json');raw=archive/'archive'
    for row in receipts:
        path=raw/row['path']
        if raw not in path.resolve().parents or path.stat().st_size!=row['bytes'] or sha256(path)!=row['sha256']:
            raise ValueError('Durable raw failure archive changed')
        if time.time()>deadline-20:raise TimeoutError('Failure archive verification deadline')
    cp=read_json(raw/'checkpoint-commits/index.json')['commits'][-1]
    if cp['checkpoint']['sha256']!=config['original_checkpoint_sha256'] or cp['cursor']['trained_decisions']!=248128:
        raise ValueError('Unexpected newer model; inspect instead of invalidating blindly')
    from .rl_core import RLFeatures
    features=RLFeatures(root/'agents/frozen/ppo-zero-anchor');lo,hi=features.b.OFFSETS['oracle']
    rows=[];recovered=[];old_tail=0
    for path in sorted((raw/'rollouts').glob('v0030-s*.pt')):
        data=torch.load(path,map_location='cpu',weights_only=False);sid=data['series_id']
        if data['policy_version']!=30 or path.name!=f'v0030-s{sid:06d}.pt':raise ValueError('Raw trajectory identity differs')
        if sid<1578:old_tail+=len(data['decisions']);continue
        summary_path=raw/'series'/f'{sid:06d}.json';summary=read_json(summary_path) if summary_path.exists() else data['summary']
        if summary['statuses']!=['DONE','DONE'] or summary['series_id']!=sid or summary['rounds'] not in (2,3):
            raise ValueError('New raw shard does not contain a normal complete BO3')
        if summary_path.exists() and summary['trajectory_sha256']!=sha256(path):raise ValueError('Summary/shard hash differs')
        for d in data['decisions']:
            if d['series_id']!=sid or int(d['policy_version'])!=30 or not math.isfinite(d['old_logp']) or not math.isfinite(d['old_value']):raise ValueError('Mixed/nonfinite trajectory')
            for step in d['steps']:
                if step['rows'][:,lo:hi].any():raise ValueError('Oracle values present in archived actor features')
        if not summary_path.exists():
            repaired={**summary,'trajectory_path':path.relative_to(raw).as_posix(),'trajectory_sha256':sha256(path)}
            atomic_json(output/'recovered-summaries'/f'{sid:06d}.json',repaired);recovered.append(sid)
        rows.append({'series_id':sid,'learner_seat':summary['learner_seat'],'decisions':len(data['decisions']),
            'rounds':summary['rounds'],'rewards':summary['rewards'],'assignment':summary['assignment'],
            'source_raw_sha256':sha256(path),'summary_already_published':summary_path.exists(),
            'terminal_transitions':sum(bool(d['terminal']) for d in data['decisions'])})
    if old_tail!=5430:raise ValueError('Original pending tail was lost/changed')
    current=read_json(raw/'progress.json');failure=read_json(raw/'failure.json')
    attempts=read_json(source/'attempts.json');origin=root/'runs'/read_json(source/'config.json')['original_training_run_id']
    original_attempts=read_json(origin/'attempts.json')
    if len(attempts)!=1 or len(original_attempts)!=1:raise ValueError('Reinspect changed attempt history')
    total=sum(x['seconds'] for x in [*original_attempts,*attempts])
    from .resource_journal import decode
    samples=[decode(line) for line in (raw/'resource-samples.jsonl').read_text(encoding='utf-8').splitlines() if line]
    control=[]
    for path in sorted((source/'control-history').glob('*.json')):
        row=read_json(path)
        if row.get('result',{}).get('operation') in ('freeze','thaw'):control.append({'at':row['at'],'action':row['action'],'evidence':path.relative_to(root).as_posix()})
    control=sorted({(x['at'],x['action']):x for x in control}.values(),key=lambda x:x['at']);frozen_seconds=0;pending=None
    for row in control:
        if row['action']=='freeze':pending=datetime.fromisoformat(row['at'])
        elif pending is not None:frozen_seconds+=(datetime.fromisoformat(row['at'])-pending).total_seconds();pending=None
    result={'at':utcnow(),'complete':True,'engineering_only':True,'source_run_id':source.name,
        'classification':'invalid_incomplete_training_target_and_attempts_exhausted',
        'full_archive_files_verified':len(receipts),'full_archive_raw_bytes':sum(x['bytes'] for x in receipts),
        'committed_model_cursor':cp['cursor'],'original_checkpoint_sha256':cp['checkpoint']['sha256'],
        'target_decisions':262144,'missing_trained_decisions':262144-cp['cursor']['trained_decisions'],
        'new_committed_PPO_updates':0,'new_complete_BO3':len(rows),'new_decisions':sum(x['decisions'] for x in rows),
        'original_pending_decisions':old_tail,'all_recoverable_pending_decisions':old_tail+sum(x['decisions'] for x in rows),
        'new_complete_BO3_rows':rows,'orphan_complete_shards_recovered_to_separate_analysis_outputs':recovered,
        'original_raw_archive_and_failed_run_unchanged':True,'oracle_feature_checks_passed':True,
        'attempts':{'original':original_attempts,'derived_original_second':attempts,'total_seconds':total,
            'attempt_count':2,'further_TR04_attempts_allowed':0,'unused_wall_time_not_a_new_attempt_allowance':True},
        'actual_failure':failure,'control_history':control,'frozen_seconds_from_confirmed_receipts':frozen_seconds,
        'failure_causality':'Observed supervisor nvidia-smi8s TimeoutExpired on thaw. Whole-cgroup freeze can expire subprocess monotonic deadlines; exact query launch timestamp was not recorded, so this is a supported timing-risk explanation, not proof of GPU/driver failure.',
        'resources':{'samples':len(samples),'peak_workspace_bytes':max(x['capacity']['workspace_bytes'] for x in samples),
            'minimum_free_bytes':min(x['capacity']['free_bytes'] for x in samples),
            'own_NVML_peak_mib':max(x.get('gpu',{}).get('own_service_process_gpu_memory_mib',0) for x in samples)},
        'scientific_lambda_strength_conclusion':None,'promotion_eligible':False,
        'next_action':'Close TR04 invalid without changing valid hypothesis/no-promotion counts. Repair FUTURE cooperative resource pause/telemetry handling and storage IO overhead, then preregister a distinct evidence-based hypothesis from full H002; no TR04 restart, weight substitution or final-gate relaxation.'}
    atomic_json(output/'analysis.json',result);atomic_json(output/'metrics.json',result)
