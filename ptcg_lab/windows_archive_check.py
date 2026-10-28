"""Actual local long-path publication using immutable real checkpoint bytes."""
import os
from pathlib import Path
import time
import math
from .io import read_json,atomic_json,sha256,utcnow
from .archive_ack import receive_object
from .archive_compressed import receive_compressed_object,verify_archive
from .local_cache_retirement import LocalReadPacer


def run(config,output):
    output=Path(output).resolve();root=Path(config['project_path']).resolve()
    if os.name!='nt' or os.environ.get('PTCG_JOB_RUN_ID')!=output.name or output.parent!=root/'runs':
        raise RuntimeError('Controlled local Windows job required')
    deadline=float(os.environ['PTCG_JOB_DEADLINE_UNIX'])
    if deadline-time.time()>180.5:raise ValueError('Actual archive repair check is at most180s')
    archive=root/'runs'/config['source_archive_run_id'];metrics=read_json(archive/'metrics.json')
    if sha256(archive/'file-receipts.json')!=config['source_archive_receipts_sha256'] or not metrics.get('all_files_hash_verified_and_fsynced_locally'):
        raise ValueError('Full original raw archive proof missing')
    import torch
    import numpy as np
    from .checkpoint_refs import read_index
    from .rl_core import RLFeatures
    from .resource_journal import decode
    torch.set_num_threads(2);raw=archive/'archive'
    receipts=read_json(archive/'file-receipts.json')
    for row in receipts:
        path=raw/row['path']
        if raw not in path.resolve().parents or path.stat().st_size!=row['bytes'] or sha256(path)!=row['sha256']:
            raise ValueError('Complete original raw archive differs')
    source=root/'runs'/metrics['source_run_id'];lease=read_json(source/'remote-job.json')
    index=read_index(raw,source.name,lease['lease_epoch']);point=index['commits'][-1]
    saved=torch.load(raw/point['checkpoint']['path'],map_location='cpu',weights_only=False)
    if point['cursor']!={'version':0,'trained_decisions':0,'next_series_id':0} or len(saved['model'])!=127 or saved['optimizer']['state'] or saved['updates']:
        raise ValueError('Expected actual initial-only checkpoint; reinspect changed progress')
    parent=root/'agents/frozen/ppo-zero-anchor/model.pt'
    if sha256(parent)!='959b220daf6777f45ad5d6aa14c35d3274bf3958ed502c7551679453e5f4aaea':raise ValueError('Original H002 differs')
    original={k.removeprefix('module.'):v for k,v in torch.load(parent,map_location='cpu',weights_only=True).items()}
    if set(original)!=set(saved['model']):raise ValueError('Incomplete actor state')
    for name,value in saved['model'].items():
        if not torch.isfinite(value).all():raise ValueError('Nonfinite original model')
        if name.startswith('value_mlp.2.'):
            if value.count_nonzero():raise ValueError('Declared blind last-value reset differs')
        elif not torch.equal(value,original[name]):raise ValueError('Unexpected trained/changed actor tensor')
    features=RLFeatures(parent.parent);lo,hi=features.b.OFFSETS['oracle'];series=0;decisions=0;terminal_rounds=0
    for path in sorted((raw/'series').glob('[0-9]'*6+'.json')):
        summary=read_json(path);sid=int(path.stem)
        if summary['series_id']!=sid or summary['statuses']!=['DONE','DONE'] or summary['rounds'] not in (2,3):raise ValueError('Incomplete BO3 counted')
        trajectory=raw/summary['trajectory_path']
        if sha256(trajectory)!=summary['trajectory_sha256']:raise ValueError('Real trajectory hash differs')
        data=torch.load(trajectory,map_location='cpu',weights_only=False)
        if data['series_id']!=sid or data['policy_version']!=0:raise ValueError('Mixed trajectory identity')
        for d in data['decisions']:
            if d['series_id']!=sid or int(d['policy_version'])!=0 or not math.isfinite(d['old_logp']) or not math.isfinite(d['old_value']):raise ValueError('Invalid untrained decision')
            for step in d['steps']:
                if step['rows'][:,lo:hi].any() or not np.isfinite(step['rows']).all():raise ValueError('Oracle or nonfinite actor feature')
        series+=1;decisions+=len(data['decisions']);terminal_rounds+=sum(bool(d['terminal']) for d in data['decisions'])
    resources=[decode(line) for line in (raw/'resource-samples.jsonl').read_text().splitlines() if line]
    progress=read_json(raw/'progress.json');failure=read_json(raw/'failure.json')
    if decisions!=progress['pending_decisions'] or series!=progress['next_series_id']:raise ValueError('Real complete BO3 counts differ')
    attempts=[*read_json(root/'runs/20261025T205445Z_tr06-entropy001_df970d/attempts.json'),*read_json(source/'attempts.json')]
    if len(attempts)!=2:raise ValueError('Original attempt accounting differs')
    analysis={'at':utcnow(),'complete':True,'classification':'invalid_incomplete_training_target_and_attempts_exhausted',
        'source_run_id':source.name,'full_archive_files':len(receipts),'full_archive_raw_bytes':sum(r['bytes'] for r in receipts),
        'published_checkpoint':point,'full_model_tensors':127,'optimizer_states':0,'PPO_updates':0,'trained_decisions':0,
        'target_decisions':32768,'complete_BO3':series,'untrained_recoverable_decisions':decisions,'terminal_rounds':terminal_rounds,
        'full_original_actor_except_declared_blind_output_reset_exact':True,'actor_oracle_checks_passed':True,
        'attempts':{'count':2,'records':attempts,'total_seconds':sum(a['seconds'] for a in attempts),'remaining':0},
        'local_error_evidence':{'path':(source/'stderr-01.log').relative_to(root).as_posix(),'sha256':sha256(source/'stderr-01.log')},
        'terminal_causality':'Observed local CreateHardLinkW path exceeded Win32 length limit; controller closed its identity-verified service. No optimization occurred, so entropy effect cannot be estimated.',
        'resource_samples':len(resources),'memory_peak_bytes':max(int(r['kernel']['memory.peak']) for r in resources),
        'own_GPU_peak_MiB':max(r.get('gpu',{}).get('own_service_process_gpu_memory_mib') or 0 for r in resources),
        'workspace_peak_bytes':failure['peak_workspace_bytes'],'minimum_free_bytes':failure['minimum_free_bytes'],
        'promotion_eligible':False,'valid_hypothesis_streak_not_incremented':True,'source_unchanged':True}
    atomic_json(output/'analysis.json',analysis)
    rows=[];pacer=LocalReadPacer(16*2**20,deadline-15)
    namespace=output/('verified-real-archive-namespace-'+'p'*95)
    sequence=0
    for ref in config['published_recovery_objects']:
        source=archive/'archive'/ref['path']
        before=sha256(source)
        if before!=ref['sha256'] or source.stat().st_size!=ref['bytes']:raise ValueError('Published real original bytes changed')
        for receiver in [receive_object,receive_compressed_object]:
            if time.time()>deadline-15:raise TimeoutError('Actual long-path check deadline')
            def chunks():
                with source.open('rb') as stream:
                    while block:=stream.read(1024*1024):pacer.consume(len(block));yield block
            ack=receiver(namespace,output.name,1,sequence,ref['path'],ref['sha256'],ref['bytes'],chunks())
            if not verify_archive(namespace,ack):raise ValueError('Real original reconstruction failed')
            target=namespace/ack['archive_relative']
            if len(str(target))<300:raise ValueError('Actual path does not cover the failing Windows namespace')
            replay=receiver(namespace,output.name,1,sequence,ref['path'],ref['sha256'],ref['bytes'],[])
            if replay!=ack:raise ValueError('Same ACK replay differs')
            rejected=False
            try:receiver(namespace,output.name,1,sequence,ref['path'],'0'*64,ref['bytes'],[])
            except ValueError:rejected=True
            if not rejected:raise ValueError('Conflicting ACK was accepted')
            rows.append({'object':ref,'codec':ack.get('storage',{}).get('codec','raw'),
                'actual_path_characters':len(str(target)),'ACK_commit_id':ack['commit_id'],
                'fresh_full_original_SHA_and_size_readback_passed':True,'idempotent_replay':True,'conflicting_ACK_rejected':True})
            sequence+=1
        if sha256(source)!=before:raise ValueError('Immutable source changed')
    atomic_json(output/'metrics.json',{'at':utcnow(),'complete':True,'engineering_only':True,'passed':True,
        'actual_Windows_long_path_RAW_and_gzip_roundtrips':rows,'real_source_archive':archive.name,
        'original_raw_artifacts_unchanged':True,'system_or_registry_changes':False,
        'GPU_used':False,'training_updates':0,'TR06_analysis':analysis,'promotion_eligible':False})
