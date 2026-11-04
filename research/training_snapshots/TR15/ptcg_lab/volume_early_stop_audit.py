"""Full, zero-update cold recovery and volume audit of the user's final TR12 segment."""
import copy,io,json,math,os,time
from pathlib import Path
from .io import read_json,atomic_json,sha256,utcnow
from .volume_scope import PARENT,BASE,TARGET

def run(config,output):
    import numpy as np
    import torch
    from .archive_ack import safe_target,validate_ack
    from .archive_compressed import verify_archive,receive_compressed_object
    from .archive_prefix import canonical_hash,compact_record
    from .checkpoint_refs import read_index
    from .ppo_accounting import checkpoint_decisions,update_decisions,ACCOUNTING
    from .training_checkpoint import load_training_checkpoint
    from .rl_resume import resume_journal
    from .rl_core import load_full_actor,optimizer_for,evaluate_decisions
    from .volume_interrupted_audit import equal,validate_rng
    from .volume_boundary import inspect
    from .sampling_parity import validate_outputs
    from .tensor_io import load_torch,atomic_torch_save
    root=Path(config['project_path']).resolve();output=Path(output).resolve();started=time.monotonic()
    deadline=min(float(os.environ['PTCG_JOB_DEADLINE_UNIX']),config.get('original_audit_deadline_unix',float('inf')))
    if os.environ.get('PTCG_JOB_RUN_ID')!=output.name or output.parent!=root/'runs' or torch.cuda.is_available() or not 0<deadline-time.time()<=1800.5:
        raise ValueError('Require controlled CPU-only1800s audit')
    torch.set_num_threads(2)
    def bounded():
        if time.time()>deadline-30:raise TimeoutError('Full recovery audit deadline')
    for name,digest in config['input_evidence'].items():
        if sha256(root/name)!=digest:raise ValueError('Frozen input changed: '+name)
    stop=read_json(root/'research/autonomy/TR12_TRAINING_STOPPED_BY_USER.json')
    rid=config['source_run_id'];assert stop['source_run_id']==rid and stop['training_stopped_by_user']
    boundary=inspect(config,output)
    source=root/'runs'/rid;lease=read_json(source/'remote-job.json')
    archive_run=root/'runs'/config['terminal_archive_run_id']
    receipts=read_json(archive_run/'file-receipts.json');inventory=read_json(archive_run/'inventory.json')
    archive_attempt=read_json(archive_run/'attempts.json')
    assert inventory['run_id']==rid and inventory['identity']==lease['identity']
    assert len(archive_attempt)==1 and archive_attempt[0]['status']=='failed'
    assert 'Required verified artifact missing' in (archive_run/'stderr-01.log').read_text(encoding='utf-8')
    assert len(receipts)==1480 and sum(x['bytes'] for x in receipts)==inventory['bytes']==885907082
    measured=read_json(root/read_json(archive_run/'config.json')['source_inventory_evidence']['path'])['actual']['rows']
    assert sorted((x['path'],x['bytes'],x['file_identity']) for x in receipts)==sorted((x['path'],x['bytes'],x['identity']) for x in measured)
    raw=output/'recovery-source'/rid;raw.mkdir(parents=True)
    for row in receipts:
        bounded();old=safe_target(archive_run/'archive',row['path']);dest=safe_target(raw,row['path'])
        assert old.stat().st_size==row['bytes'] and sha256(old)==row['sha256']
        dest.parent.mkdir(parents=True,exist_ok=True);os.link(old,dest)
    for name,row in boundary['source_files'].items():assert sha256(raw/name)==row['sha256']
    archive=source/'online-archive';new_acks=[];outbox=sorted((raw/'archive-outbox').glob('*.json'))
    for seq,path in enumerate(outbox):
        bounded();row=read_json(path)
        assert (row['run_id'],row['lease_epoch'],row['sequence'])==(rid,lease['lease_epoch'],seq)
        ap=archive/rid/'acks'/f'{seq:08d}.json'
        if ap.exists():
            ack=validate_ack(read_json(ap),rid,lease['lease_epoch'],row['object'],row['sha256'],row['bytes'],seq)
            verify_archive(archive,ack)
        else:
            original=safe_target(raw,row['object']);assert sha256(original)==row['sha256']
            with original.open('rb') as stream:
                def chunks():
                    while block:=stream.read(1024*1024):bounded();yield block
                ack=receive_compressed_object(archive,rid,lease['lease_epoch'],seq,row['object'],row['sha256'],row['bytes'],chunks())
            new_acks.append({'sequence':seq,'object':row['object'],'sha256':row['sha256'],'bytes':row['bytes']})
    catalog={};series={};selected={};receipts_count=0
    members=read_json(root/'research/autonomy/state.json')['forward_phases']['TR12']['training_run_ids']
    for member in config['archive_source_run_ids']:
        ar=root/'runs'/member/'online-archive'
        for ap in sorted((ar/member/'acks').glob('*.json')):
            bounded();ack=read_json(ap);assert ack['run_id']==member
            catalog.setdefault(ack['sha256'],(ar,ack,ap))
            if member not in members:continue
            receipts_count+=1
            payload=verify_archive(ar,ack,collect=ack['object'].startswith('series/'))
            if ack['object'].startswith('series/'):
                row=json.loads(payload);sid=row['series_id'];assert row['statuses']==['DONE','DONE'] and row['rounds'] in (2,3)
                if sid in series:assert series[sid]['trajectory_sha256']==row['trajectory_sha256']
                else:series[sid]=row
    def original(digest,collect=False):
        bounded();ar,ack,ap=catalog[digest]
        selected[digest]={'archive_root':ar.relative_to(root).as_posix(),'ack':{'path':ap.relative_to(root).as_posix(),'sha256':sha256(ap)},
                          'object_sha256':digest,'raw_bytes':ack['bytes'],'archive_relative':ack['archive_relative']}
        return verify_archive(ar,ack,collect=collect)
    required=read_json(archive_run/'config.json')['required_hashes'];received={r['path']:r for r in receipts}
    archived_evicted=[]
    for name,digest in required.items():
        if name in received:assert received[name]['sha256']==digest
        else:
            original(digest)
            archived_evicted.append({'path':name,'sha256':digest,'original_archived_not_on_server':True})
    index=read_index(raw,rid,lease['lease_epoch']);points=index['commits'];assert [p['generation'] for p in points]==[19,20]
    actual=read_json(raw/'payload-config.json');progress=read_json(raw/'progress.json');league=read_json(raw/'league.json')
    prior=read_json(root/'runs'/PARENT/'remote-results/training-updates.json')
    source_updates=read_json(raw/'training-updates.json')
    prior_recovery=read_json(root/'research/autonomy/state.json')['forward_phases']['TR12']['interruption_recovery']['report']
    assert sha256(root/prior_recovery['path'])==prior_recovery['sha256']
    prior_report=read_json(root/prior_recovery['path'])
    assert prior_report['complete'] and prior_report['source_run_id']==members[0]
    prior_segment=read_json(root/prior_report['recovery_source_root']/'training-updates.json')
    model,features,_=load_full_actor(root/actual['bundle_path'],actual['parent_model_sha256'],'cpu',reset_value=False)
    model.eval();checkpoints=[];saved_states=[];journals=[]
    for point in points:
        bounded()
        for item in [point['checkpoint'],point['prefix'],*point['pending']]:
            assert sha256(raw/item['path'])==item['sha256'];original(item['sha256'])
        saved=load_torch(raw/point['checkpoint']['path'],map_location='cpu',weights_only=False)
        assert saved['updates']==source_updates and saved['updates'][:8]==prior and saved['updates'][:11]==prior_segment
        assert checkpoint_decisions(saved)==106935 and saved['version']==15
        assert {k:saved[k] for k in point['cursor']}==point['cursor']
        assert len(saved['model'])==127 and len(saved['optimizer']['state'])==97
        for t in [*saved['model'].values(),*[v for slot in saved['optimizer']['state'].values() for v in slot.values() if torch.is_tensor(v)]]:assert torch.isfinite(t).all()
        validate_rng(saved['rng'],saved['cpu_sampler_rng'])
        model.load_state_dict(saved['model'],strict=True)
        optimizer=optimizer_for(model,actual['ppo']['actor_learning_rate'],actual['ppo']['value_head_learning_rate'])
        optimizer.load_state_dict(saved['optimizer'])
        assert equal(model.state_dict(),saved['model']) and equal(optimizer.state_dict(),saved['optimizer'])
        spec=dict(actual['archive_restore'],prefix_path=point['prefix']['path'])
        journal=resume_journal(raw,saved,spec,point['checkpoint']['sha256'])
        assert len(journal['decisions'])==6807 and journal['next_series_id']==697
        assert journal['fixed_statistics']==league['fixed_training_results']
        validate_rng(journal['rng_after'],journal['cpu_sampler_rng_after'])
        checkpoints.append({'generation':point['generation'],'checkpoint':point['checkpoint'],'cursor':point['cursor'],
            'model_tensors':127,'optimizer_states':97,'CPU_model_optimizer_bitwise_restore':True,
            'all_RNG_roundtrip':True,'pending_decisions_after_recovery':len(journal['decisions']),
            'next_series_id_after_recovery':journal['next_series_id']})
        saved_states.append(saved);journals.append(journal)
    saved=saved_states[-1];point=points[-1];journal=journals[-1]
    assert equal(saved_states[0]['model'],saved['model']) and equal(saved_states[0]['optimizer'],saved['optimizer'])
    assert equal(journals[0]['rng_after'],journal['rng_after']) and equal(journals[0]['cpu_sampler_rng_after'],journal['cpu_sampler_rng_after'])
    loaded=load_training_checkpoint(raw,actual['archive_restore'],lease['lease_epoch']);assert loaded['digest']==point['checkpoint']['sha256']
    assert not (raw/'inflight.json').exists() and progress['pending_decisions']==6807 and progress['next_series_id']==697
    assert progress['unique_PPO_trained_decisions']==106935 and progress['version']==15
    for row in league['snapshots']:original(row['sha256'])
    frozen=league['snapshots'][-1];model_bytes=original(frozen['sha256'],True)
    assert equal(torch.load(io.BytesIO(model_bytes),map_location='cpu',weights_only=True),saved['model'])
    for opponent in actual['league']['fixed_opponents']:
        for name,digest in opponent['files'].items():assert sha256(root/opponent['path']/name)==digest
    from .continuous_mainline import normalize_fixed_file_paths
    template=normalize_fixed_file_paths(read_json(root/'configs/training/tr12_volume_320768.json'))
    for key in ('seed','ppo','league'):assert actual[key]==template[key],key
    pending=journal['decisions'];lo,hi=features.b.OFFSETS['oracle']
    for decision in pending:
        assert int(decision['policy_version'])==15 and 660<=decision['series_id']<697
        assert all(not step['rows'][:,lo:hi].any() and np.isfinite(step['rows']).all() for step in decision['steps'])
    with torch.no_grad():logp,values,_=evaluate_decisions(model,pending[:48],features.b.STOP)
    parity=validate_outputs(pending[:48],logp,values)
    # Match every complete summary to its immutable trajectory and journal record.
    for sid,row in series.items():
        original(row['trajectory_sha256']);assert row['learner_decisions']>0
    expected=set(range(330,697))-{472};assert set(series)==expected
    new=saved['updates'][8:];ids=[sid for u in new for sid in u['series_ids']]
    assert len(ids)==len(set(ids)) and set(ids)<=expected and not (set(ids)&{d['series_id'] for d in pending})
    increment=sum(update_decisions(u) for u in new);segment_increment=sum(update_decisions(u) for u in new[3:])
    assert increment==57762 and segment_increment==33035 and BASE+increment==checkpoint_decisions(saved)
    assert sum(row['learner_decisions'] for row in series.values())==sum(u['decisions'] for u in new)+6807
    gradients=[]
    for u in new:
        steps=[x for x in u['steps'] if not x['skipped_for_kl']]
        g=steps[0].get('policy_gradient_l2') if steps else None
        gradients.append({'version':u['version'],'policy_gradient_l2':g,'passed':g is not None and math.isfinite(g) and g>0})
    fixture=output/'current-model-numeric-reference.pt'
    fixture_rows=copy.deepcopy(pending[:48])
    for d,p,v in zip(fixture_rows,logp.tolist(),values.tolist()):d.update(old_logp=p,old_value=float(np.asarray(v).reshape(-1)[0]),recovery_fixture_only=True)
    atomic_torch_save(fixture,{'model':saved['model'],'fixture':fixture_rows,'fixture_logp':logp.cpu(),'fixture_values':values.cpu(),
        'not_training_data':True,'source_run_id':rid,'source_checkpoint_sha256':point['checkpoint']['sha256']},replace=False)
    endpoint=output/'endpoint-model.pt'
    with endpoint.open('xb') as f:f.write(model_bytes);f.flush();os.fsync(f.fileno())
    assert sha256(endpoint)==frozen['sha256']
    atomic_json(output/'verified-archive-inventory.json',{'at':utcnow(),'objects':list(selected.values()),'only_relevant_full_hashes_verified':True})
    report={'at':utcnow(),'complete':True,'engineering_only':True,'source_run_id':rid,'terminal_archive_run_id':archive_run.name,
        'source_point':point,'both_committed_recovery_points':checkpoints,'retained_generations':[19,20],
        'recovery_source_root':raw.relative_to(root).as_posix(),'source_remote_evidence':{k:v['sha256'] for k,v in boundary['source_files'].items()},
        'source_league_sha256':boundary['source_files']['league.json']['sha256'],'source_inflight':None,'complete_postcheckpoint_tail':None,
        'actual_unique_PPO_decisions':106935,'unique_PPO_trained_decisions':106935,'PPO_updates':15,
        'TR12_new_unique_PPO_decisions':increment,'segment002_new_unique_PPO_decisions':segment_increment,'TR12_new_PPO_updates':len(new),
        'new_complete_BO3':len(series),'new_complete_games':sum(r['rounds'] for r in series.values()),
        'new_sampled_decisions':sum(r['learner_decisions'] for r in series.values()),
        'pending_decisions':6807,'next_series_id':697,'inflight_BO3':None,'prior_incomplete_series472_excluded':True,
        'terminal_file_count':len(receipts),'terminal_original_bytes':sum(x['bytes'] for x in receipts),
        'terminal_archive_independent_reverification_passed':True,
        'archive_transport_attempt_status_preserved':archive_attempt,
        'archive_final_validation_failure':'Required list included already-evicted but fully archived paths; all1480 actual terminal files transferred successfully before the final membership check.',
        'archived_evicted_paths_verified_separately':archived_evicted,
        'original_ACKs_reused':len(outbox)-len(new_acks),'new_ACKs':new_acks,'total_complete_outbox_ACKs':len(outbox),
        'all_full_state_and_pending_archives_verified':True,'reference_fixture_never_enters_training':True,
        'CPU_cold_restore_passed':True,'all_RNG_roundtrip':True,'CUDA_RNG_bytes_preserved':True,
        'GPU_difference_acceptance_pending':True,'pending_CPU_probability_value_parity':parity,
        'optimizer_steps_executed':0,'new_sampled_decisions_in_audit':0,'new_unique_training_decisions':0,
        'new_update_coverage':[{'version':u['version'],'consumed':u['decisions'],'actual_unique':update_decisions(u)} for u in new],
        'new_update_gradients':gradients,'policy_gradient_health_passed':all(g['passed'] for g in gradients),
        'endpoint_model':{'path':endpoint.relative_to(root).as_posix(),'sha256':sha256(endpoint),'bytes':endpoint.stat().st_size},
        'numeric_reference':{'path':fixture.relative_to(root).as_posix(),'sha256':sha256(fixture),'bytes':fixture.stat().st_size},
        'archive_inventory':{'path':(output/'verified-archive-inventory.json').relative_to(root).as_posix(),'sha256':sha256(output/'verified-archive-inventory.json')},
        'boundary_receipt':{'path':(output/'boundary-and-source-receipt.json').relative_to(root).as_posix(),'sha256':sha256(output/'boundary-and-source-receipt.json')},
        'original_training_target_complete':False,'user_early_stop':True,'model_selected_by_win_rate':False,
        'decision_accounting':ACCOUNTING,'remote_deletion_or_ACK_writes':False,'promotion_eligible':False,'seconds':time.monotonic()-started}
    atomic_json(output/'verified-recovery-report.json',report);atomic_json(output/'metrics.json',report)
    return report
