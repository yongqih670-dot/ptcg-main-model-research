"""Archive completion and CPU cold recovery of the interrupted first TR12 segment.

No optimizer steps, server mutations, changed checkpoint bytes or new samples.
The complete post-checkpoint BO3 is recovered with the production journal.
"""
import copy,io,json,os,random,time
from pathlib import Path
from .io import read_json,atomic_json,sha256,utcnow
from .volume_scope import PARENT,BASE,PARENT_CP

SOURCE='20261028T060043Z_tr12-volume-segment-001_a709d8'


def equal(a,b):
    import numpy as np
    import torch
    if torch.is_tensor(a):return torch.is_tensor(b) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and a.dtype==b.dtype and np.array_equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(v,b[k]) for k,v in a.items())
    if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def validate_rng(rng,sampler):
    import numpy as np
    import torch
    if set(rng)!={'python','numpy','torch','cuda'} or len(rng['cuda'])!=1:raise ValueError('Full RNG schema differs')
    py=random.Random();py.setstate(rng['python'])
    nr=np.random.RandomState();nr.set_state(rng['numpy'])
    tr=torch.Generator();tr.set_state(rng['torch'])
    actor=torch.Generator();actor.set_state(sampler)
    if not equal(py.getstate(),rng['python']) or not equal(nr.get_state(),rng['numpy']) or not torch.equal(tr.get_state(),rng['torch']) or not torch.equal(actor.get_state(),sampler):
        raise ValueError('CPU/Python/NumPy/actor RNG roundtrip differs')
    if any(x.dtype!=torch.uint8 or x.ndim!=1 or not x.numel() for x in rng['cuda']):raise ValueError('CUDA RNG bytes incomplete')


def run(config,output):
    import numpy as np
    import torch
    from .archive_ack import safe_target,validate_ack
    from .archive_compressed import verify_archive,receive_compressed_object
    from .archive_prefix import compact_record,canonical_hash
    from .checkpoint_refs import read_index
    from .ppo_accounting import checkpoint_decisions,update_decisions,ACCOUNTING
    from .remote_lease import validate_no_unreconciled_remote
    from .training_checkpoint import load_training_checkpoint
    from .rl_resume import resume_journal
    from .rl_core import load_full_actor,optimizer_for,evaluate_decisions
    from .sampling_parity import validate_outputs
    from .tensor_io import load_torch,atomic_torch_save
    root=Path(config['project_path']).resolve();output=Path(output).resolve();started=time.perf_counter()
    deadline=float(os.environ['PTCG_JOB_DEADLINE_UNIX'])
    if os.environ.get('PTCG_JOB_RUN_ID')!=output.name or output.parent!=root/'runs' or torch.cuda.is_available() or not 0<deadline-time.time()<=1800.5:
        raise RuntimeError('Require one bounded controlled CPU audit')
    torch.set_num_threads(2)
    def bounded():
        if time.time()>deadline-25:raise TimeoutError('Original recovery audit deadline')
    for path,digest in config['input_evidence'].items():
        if sha256(root/path)!=digest:raise ValueError('Frozen audit input differs: '+path)
    if config['source_run_id']!=SOURCE:raise ValueError('This audit is scoped to the interrupted first segment')
    source=root/'runs'/SOURCE;lease=read_json(source/'remote-job.json')
    validate_no_unreconciled_remote(root,{'remote_job':lease})
    prior_cost=read_json(source/'attempts.json')
    if len(prior_cost)!=1 or prior_cost[0]['status']!='failed' or prior_cost[0]['seconds']!=28800:raise ValueError('Original failed cost/attempt not preserved')
    archived=root/'runs'/config['terminal_archive_run_id'];m=read_json(archived/'metrics.json')
    receipts=read_json(archived/'file-receipts.json')
    if not m['complete'] or m['source_run_id']!=SOURCE or len(receipts)!=951 or sum(x['bytes'] for x in receipts)!=380708061:raise ValueError('Terminal archive incomplete')
    raw=output/'recovery-source'/SOURCE;raw.mkdir(parents=True)
    bypath={};byhash={}
    for row in receipts:
        bounded();old=safe_target(archived/'archive',row['path']);dest=safe_target(raw,row['path'])
        if old.stat().st_size!=row['bytes'] or sha256(old)!=row['sha256']:raise ValueError('Terminal original hash differs')
        dest.parent.mkdir(parents=True,exist_ok=True);os.link(old,dest)
        bypath[row['path']]=row;byhash.setdefault(row['sha256'],old)
    archive=source/'online-archive';acks={};new_acks=[];verified=[]
    outbox=sorted((raw/'archive-outbox').glob('*.json'))
    if len(outbox)!=302:raise ValueError('Original terminal outbox changed')
    for seq,path in enumerate(outbox):
        bounded();row=read_json(path)
        if (row['run_id'],row['lease_epoch'],row['sequence'])!=(SOURCE,lease['lease_epoch'],seq):raise ValueError('Outbox fencing differs')
        ap=archive/SOURCE/'acks'/f'{seq:08d}.json'
        if ap.exists():
            ack=validate_ack(read_json(ap),SOURCE,lease['lease_epoch'],row['object'],row['sha256'],row['bytes'],seq)
            verify_archive(archive,ack)
        else:
            item=bypath.get(row['object'])
            if item is None or (item['sha256'],item['bytes'])!=(row['sha256'],row['bytes']):raise ValueError('Missing unique original cannot be acknowledged')
            with safe_target(raw,row['object']).open('rb') as stream:
                def chunks():
                    while block:=stream.read(1024*1024):bounded();yield block
                ack=receive_compressed_object(archive,SOURCE,lease['lease_epoch'],seq,row['object'],row['sha256'],row['bytes'],chunks())
            new_acks.append({'sequence':seq,'object':row['object'],'sha256':row['sha256'],'bytes':row['bytes']})
        acks[row['object']]=ack
        verified.append({'sequence':seq,'object':row['object'],'raw_sha256':row['sha256'],'raw_bytes':row['bytes'],
            'ACK_path':ap.relative_to(root).as_posix(),'ACK_sha256':sha256(ap)})
        if seq%50==0:atomic_json(output/'progress.json',{'at':utcnow(),'stage':'full_hash_archive_verification','verified':seq+1,'total':len(outbox)})
    atomic_json(output/'completed-archive-index.json',{'at':utcnow(),'source_run_id':SOURCE,'objects':verified,
        'new_ACKs':new_acks,'remote_ACKs_not_written':True,'remote_originals_not_deleted':True})
    index=read_index(raw,SOURCE,lease['lease_epoch']);points=index['commits']
    if [x['generation'] for x in points]!=[11,12]:raise ValueError('Expected both actual retained recovery points')
    actual=read_json(raw/'payload-config.json');progress=read_json(raw/'progress.json');league=read_json(raw/'league.json')
    parent_config=read_json(source/'config.json');parent_ack=next(x for x in parent_config['local_archive_sources'] if x['sha256']==PARENT_CP)
    parent=torch.load(io.BytesIO(verify_archive(root/parent_ack['archive_root'],parent_ack['ack'],collect=True)),map_location='cpu',weights_only=False)
    if checkpoint_decisions(parent)!=BASE or parent['version']!=8:raise ValueError('TR11 parent differs')
    parent_records={r['series_id']:r for r in parent['archive_state']['archived']+parent['archive_state']['live_expected']}
    checkpoint_reports=[];payloads=[]
    model,features,_=load_full_actor(root/actual['bundle_path'],actual['parent_model_sha256'],'cpu',reset_value=False)
    model.eval()
    for point in points:
        bounded();saved=load_torch(raw/point['checkpoint']['path'],map_location='cpu',weights_only=False)
        if {k:saved[k] for k in point['cursor']}!=point['cursor'] or len(saved['model'])!=127 or len(saved['optimizer']['state'])!=97:
            raise ValueError('Full checkpoint tensor/slot/cursor differs')
        if saved['updates'][:8]!=parent['updates'] or checkpoint_decisions(saved)!=73900 or saved['version']!=11:
            raise ValueError('Inherited updates or true unique volume differs')
        for tensor in [*saved['model'].values(),*[v for slot in saved['optimizer']['state'].values() for v in slot.values() if torch.is_tensor(v)]]:
            if not torch.isfinite(tensor).all():raise ValueError('Nonfinite full model or optimizer')
        validate_rng(saved['rng'],saved['cpu_sampler_rng'])
        model.load_state_dict(saved['model'],strict=True)
        optimizer=optimizer_for(model,actual['ppo']['actor_learning_rate'],actual['ppo']['value_head_learning_rate'])
        optimizer.load_state_dict(saved['optimizer'])
        if not equal(model.state_dict(),saved['model']) or not equal(optimizer.state_dict(),saved['optimizer']):raise ValueError('CPU cold model/optimizer restore differs')
        journal=resume_journal(raw,saved,dict(actual['archive_restore'],prefix_path=point['prefix']['path']),point['checkpoint']['sha256'])
        if len(journal['decisions'])!=104 or journal['next_series_id']!=472 or journal['shards']!=['rollouts/v0011-s000471.pt']:
            raise ValueError('Complete post-checkpoint tail recovery differs')
        validate_rng(journal['rng_after'],journal['cpu_sampler_rng_after'])
        records={r['series_id']:r for r in saved['archive_state']['archived']+saved['archive_state']['live_expected']}
        if any(records.get(sid)!=r for sid,r in parent_records.items()):raise ValueError('Inherited archived history changed')
        checkpoint_reports.append({'generation':point['generation'],'checkpoint':point['checkpoint'],'cursor':point['cursor'],
            'full_model_tensor_count':127,'optimizer_state_count':97,'CPU_restore_bitwise_equal':True,'all_RNG_payloads_present':True,
            'CPU_Python_NumPy_sampler_RNG_roundtrip_equal':True,'CUDA_RNG_bytes':int(saved['rng']['cuda'][0].numel()),
            'pending_decisions_after_journal_recovery':104,'journal_next_series_id':472})
        payloads.append(saved)
    saved=payloads[-1];point=points[-1]
    if not equal(payloads[0]['model'],saved['model']) or not equal(payloads[0]['optimizer'],saved['optimizer']):raise ValueError('Same-update recovery points disagree on learned state')
    loaded=load_training_checkpoint(raw,actual['archive_restore'],lease['lease_epoch'])
    if loaded['digest']!=point['checkpoint']['sha256']:raise ValueError('Production loader did not select latest committed checkpoint')
    journal=resume_journal(raw,loaded['saved'],loaded['archive_spec'],loaded['digest'])
    decisions=journal['decisions'];shard=load_torch(raw/journal['shards'][0],map_location='cpu',weights_only=False)
    tail=read_json(raw/'series/000471.json')
    if tail['statuses']!=['DONE','DONE'] or tail['learner_decisions']!=104 or tail['rounds'] not in (2,3) or len(shard['decisions'])!=104:
        raise ValueError('Incomplete terminal BO3')
    if not equal(journal['rng_after'],shard['rng_after']) or not equal(journal['cpu_sampler_rng_after'],shard['cpu_sampler_rng_after']):raise ValueError('Journal failed to select latest complete BO3 RNG')
    inflight=read_json(raw/'inflight.json');sid=inflight['series_id']
    if sid!=472 or (raw/f'series/{sid:06d}.json').exists() or any(d['series_id']==sid for d in decisions):raise ValueError('Unfinished BO3 entered recovery data')
    effective_next=max(journal['next_series_id'],sid+1)
    if progress['pending_decisions']!=104 or progress['next_series_id']!=472 or progress['unique_PPO_trained_decisions']!=73900:raise ValueError('Actual terminal progress differs')
    if journal['fixed_statistics']!=league['fixed_training_results']:raise ValueError('League statistics need unproven correction')
    for member in league['snapshots']:
        bounded()
        if sha256(raw/member['path'])!=member['sha256']:raise ValueError('Frozen league member changed')
    frozen=load_torch(raw/league['snapshots'][-1]['path'],map_location='cpu',weights_only=True)
    if not equal(frozen,saved['model']):raise ValueError('Latest frozen actor/critic differs from saved full model')
    for opponent in actual['league']['fixed_opponents']:
        for name,digest in opponent['files'].items():
            if sha256(root/opponent['path']/name)!=digest:raise ValueError('Fixed legal opponent input changed')
    from .continuous_mainline import normalize_fixed_file_paths
    template=normalize_fixed_file_paths(read_json(root/'configs/training/tr12_volume_320768.json'))
    for key in ('seed','ppo','league'):
        if actual[key]!=template[key]:raise ValueError('Frozen scientific setting differs: '+key)
    lo,hi=features.b.OFFSETS['oracle']
    for decision in decisions:
        if decision['series_id']!=471 or int(decision['policy_version'])!=11:raise ValueError('Tail decision identity mixed')
        if any(step['rows'][:,lo:hi].any() or not np.isfinite(step['rows']).all() for step in decision['steps']):raise ValueError('Hidden or nonfinite actor feature')
    with torch.no_grad():logp,values,_=evaluate_decisions(model,decisions[:48],features.b.STOP)
    parity=validate_outputs(decisions[:48],logp,values)
    new_updates=saved['updates'][8:];trained_ids=[sid for row in new_updates for sid in row['series_ids']]
    if len(trained_ids)!=len(set(trained_ids)) or min(trained_ids)<330 or 471 in trained_ids or 472 in trained_ids:raise ValueError('Duplicate/inherited/untrained series in new updates')
    summaries=[]
    for name,ack in acks.items():
        if name.startswith('series/'):
            row=json.loads(verify_archive(archive,ack,collect=True));trajectory=acks[row['trajectory_path']]
            if row['trajectory_sha256']!=trajectory['sha256'] or row['statuses']!=['DONE','DONE'] or row['rounds'] not in (2,3):raise ValueError('Complete BO3 archive mismatch')
            summaries.append(row)
    complete_ids={r['series_id'] for r in summaries}
    if not set(trained_ids)<=complete_ids or sorted(complete_ids)!=list(range(330,472)):raise ValueError('New series archive gap or duplicate')
    records={r['series_id']:r for r in saved['archive_state']['archived']+saved['archive_state']['live_expected']}
    for row in summaries:
        sid=row['series_id'];compact=compact_record(row,acks[f'series/{sid:06d}.json']['sha256'])
        if sid<471 and records.get(sid)!=compact:raise ValueError('Checkpoint archive record differs from actual summary')
    increment=sum(update_decisions(r) for r in new_updates)
    if increment!=24727 or BASE+increment!=checkpoint_decisions(saved):raise ValueError('First-epoch unique accounting mismatch')
    # A separate recovery-only fixture may be used by a later bounded GPU check.
    fixture=output/'current-endpoint-numeric-reference.pt'
    atomic_torch_save(fixture,{'model':saved['model'],'fixture':copy.deepcopy(decisions[:48]),'fixture_logp':logp.cpu(),
        'fixture_values':values.cpu(),'not_training_data':True,'source_checkpoint_sha256':point['checkpoint']['sha256']},replace=False)
    bounded()
    report={'at':utcnow(),'complete':True,'engineering_only':True,'classification':'TR12_interrupted_full_CPU_state_and_complete_tail_verified',
        'source_run_id':SOURCE,'terminal_archive_run_id':archived.name,'recovery_source_root':raw.relative_to(root).as_posix(),
        'terminal_file_count':len(receipts),'terminal_original_bytes':sum(x['bytes'] for x in receipts),
        'original_ACKs_reused':len(outbox)-len(new_acks),'new_ACKs':new_acks,'total_complete_outbox_ACKs':len(outbox),
        'both_committed_recovery_points':checkpoint_reports,'source_point':point,
        'checkpoint_cursor':point['cursor'],'complete_tail_series_id':471,'pending_decisions':104,
        'unfinished_BO3_preserved_and_excluded':inflight,'next_series_id_after_interruption_exclusion':effective_next,
        'RNG_source_for_continuation':'complete series471 rng_after and cpu_sampler_rng_after; checkpoint model/optimizer unchanged',
        'all_12_frozen_league_hashes_verified':league['snapshots'],'fixed_training_statistics':journal['fixed_statistics'],
        'pending_CPU_probability_value_parity':parity,'CUDA_RNG_bytes_preserved':True,'new_GPU_runtime_restore_test_performed':False,
        'new_complete_BO3':len(summaries),'new_complete_games':sum(r['rounds'] for r in summaries),
        'new_sampled_decisions':sum(r['learner_decisions'] for r in summaries),
        'unique_PPO_trained_decisions':73900,'TR12_new_unique_PPO_decisions':increment,'TR12_new_PPO_updates':3,
        'decision_accounting':ACCOUNTING,'new_update_coverage':[{'version':r['version'],'consumed':r['decisions'],'actual_unique':update_decisions(r)} for r in new_updates],
        'inherited_8_updates_49173_decisions_unchanged':True,'optimizer_steps_executed':0,'new_sampled_decisions_in_audit':0,
        'all_source_checkpoint_and_terminal_bytes_unchanged':True,'remote_writes_or_evictions':False,
        'numerical_reference':{'path':fixture.relative_to(root).as_posix(),'sha256':sha256(fixture),'bytes':fixture.stat().st_size},
        'original_deadline_unix':lease['deadline_unix'],'prior_failed_attempt_seconds':28800,'promotion_eligible':False,
        'training_route_ready':False,'remaining_training_gate':'Bind complete post-checkpoint tail and interrupted cursor in future segment handoff; current endpoint GPU zero-update difference acceptance required.',
        'seconds':time.perf_counter()-started}
    atomic_json(output/'verified-recovery-report.json',report);atomic_json(output/'metrics.json',report)
    return report
