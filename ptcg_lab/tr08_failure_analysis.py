"""Read-only full TR08 terminal evidence, including a committed orphan tensor."""
import math,os,time
from pathlib import Path
from .io import read_json,atomic_json,sha256,utcnow


def same_tree(a,b):
    import numpy as np
    import torch
    if torch.is_tensor(a):return torch.is_tensor(b) and torch.equal(a,b)
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and set(a)==set(b) and all(same_tree(v,b[k]) for k,v in a.items())
    if isinstance(a,(list,tuple)):return type(a) is type(b) and len(a)==len(b) and all(same_tree(x,y) for x,y in zip(a,b))
    return type(a) is type(b) and a==b


def run(config,output):
    import numpy as np
    import torch
    from .tensor_io import load_torch
    from .checkpoint_refs import read_index
    from .archive_prefix import load_prefix,compact_record
    from .archive_state import validate_state
    from .rl_core import RLFeatures,UNUSED,load_full_actor,evaluate_decisions
    p=Path(config['project_path']).resolve();output=Path(output)
    if os.environ.get('PTCG_JOB_RUN_ID')!=output.name:raise RuntimeError('Controlled local forensic job required')
    deadline=float(os.environ['PTCG_JOB_DEADLINE_UNIX']);torch.set_num_threads(2)
    source=p/'runs'/'20261026T111415Z_tr08-original-second-lr3e6-CPU32_8de8af';archive=p/'runs'/config['archive_run_id'];raw=archive/'archive'
    if archive.resolve().parent!=p/'runs' or sha256(archive/'file-receipts.json')!=config['archive_receipts_sha256']:raise ValueError('Full archive scope/receipt differs')
    m=read_json(archive/'metrics.json');receipts=read_json(archive/'file-receipts.json')
    if not m.get('complete') or not m.get('all_files_hash_verified_and_fsynced_locally') or m.get('source_run_id')!=source.name or len(receipts)!=432:
        raise ValueError('Complete actual432file cold archive required')
    for row in receipts:
        path=raw/row['path']
        if raw.resolve() not in path.resolve().parents or path.stat().st_size!=row['bytes'] or sha256(path)!=row['sha256']:raise ValueError('Full original raw archive changed')
        if time.time()>deadline-15:raise TimeoutError('Cold forensic deadline')
    lease=read_json(source/'remote-job.json');index=read_index(raw,source.name,lease['lease_epoch']);points=index['commits'];point=points[-1]
    cursor={'version':3,'trained_decisions':24837,'next_series_id':155}
    if [(x['generation'],len(x['pending'])) for x in points]!=[(2,102),(3,0)] or any(x['cursor']!=cursor for x in points):
        raise ValueError('Actual retained reference/cursor mechanism differs')
    saved=load_torch(raw/point['checkpoint']['path'],map_location='cpu',weights_only=False)
    if {k:saved[k] for k in cursor}!=cursor or len(saved['model'])!=127 or len(saved['optimizer']['state'])!=97 or set(saved['rng'])!={'python','numpy','torch','cuda'} or len(saved['rng']['cuda'])!=1:
        raise ValueError('Full committed state coverage differs')
    for value in [*saved['model'].values(),*[v for slots in saved['optimizer']['state'].values() for v in slots.values() if torch.is_tensor(v)]]:
        if not torch.isfinite(value).all():raise ValueError('Nonfinite committed model/optimizer')
    prefix=load_prefix(raw/point['prefix']['path'],point['prefix']['sha256']);state=validate_state(saved['archive_state'],prefix,point['prefix']['sha256'])
    if state['live_expected'] or len(state['archived'])!=155:raise ValueError('Latest consumed full BO3 history differs')
    updates=saved['updates']
    if len(updates)!=3 or [u['decisions'] for u in updates]!=[8223,8300,8314] or sum(u['decisions'] for u in updates)!=24837:
        raise ValueError('Actual committed PPO accounting differs')
    summaries={};current=[];orphan=[];consumed=[];pending_decisions=0;orphan_decisions=0;all_streams=0
    features=RLFeatures(p/'agents/frozen/ppo-zero-anchor');lo,hi=features.b.OFFSETS['oracle'];fixture=[]
    for path in sorted((raw/'rollouts').glob('*.pt')):
        data=load_torch(path,map_location='cpu',weights_only=False);sid=data['series_id'];version=int(data['policy_version'])
        if path.name!=f'v{version:04d}-s{sid:06d}.pt' or version not in (2,3):raise ValueError('Cold shard identity differs')
        summary_path=raw/'series'/f'{sid:06d}.json';is_orphan=not summary_path.exists()
        summary=read_json(summary_path) if not is_orphan else {**data['summary'],'trajectory_path':path.relative_to(raw).as_posix(),'trajectory_sha256':sha256(path)}
        if summary['statuses']!=['DONE','DONE'] or summary['rounds'] not in (2,3) or summary['series_id']!=sid or int(summary['policy_version'])!=version or summary['trajectory_sha256']!=sha256(path):
            raise ValueError('Only complete hash-verified BO3 data accepted')
        for d in data['decisions']:
            if d['series_id']!=sid or int(d['policy_version'])!=version or not math.isfinite(d['old_logp']) or not math.isfinite(d['old_value']):raise ValueError('Mixed/nonfinite actor decisions')
            for step in d['steps']:
                if step['rows'][:,lo:hi].any() or not np.isfinite(step['rows']).all():raise ValueError('Actor information isolation failed')
        stream=data.get('cpu_sampler_rng_after')
        if stream is None or stream.dtype!=torch.uint8 or stream.ndim!=1:raise ValueError('Complete CPU actor stream absent')
        torch.Generator(device='cpu').set_state(stream);all_streams+=1
        if is_orphan:
            if (version,sid)!=(3,167):raise ValueError('Unexpected orphan needs separate inspection')
            atomic_json(output/'reconstructed-orphan-167.json',{'original_summary_missing':True,'original_trajectory_sha256':sha256(path),'reconstructed_complete_BO3_summary':summary,'not_new_training_data':True})
            orphan.append(sid);orphan_decisions+=len(data['decisions'])
        elif version==3:
            current.append(sid);pending_decisions+=len(data['decisions']);fixture.extend(data['decisions'][:max(0,48-len(fixture))])
        else:consumed.append(sid)
        if time.time()>deadline-15:raise TimeoutError('Full trajectory forensic deadline')
    if consumed!=list(range(104,155)) or current!=list(range(155,167)) or orphan!=[167] or pending_decisions!=1706:
        raise ValueError('Actual consumed/current/orphan trajectory coverage differs')
    progress=read_json(raw/'progress.json')
    if (progress['trained_decisions'],progress['pending_decisions'],progress['next_series_id'])!=(24837,1706,167):raise ValueError('Actual last progress differs')
    configs=read_json(raw/'payload-config.json');model,features,_=load_full_actor(p/'agents/frozen/ppo-zero-anchor',configs['parent_model_sha256'],'cpu');model.load_state_dict(saved['model'],strict=True)
    with torch.no_grad():logp,values,_=evaluate_decisions(model,fixture,features.b.STOP)
    logp_error=float(np.max(np.abs(logp.numpy()-np.asarray([d['old_logp'] for d in fixture]))));value_error=float(np.max(np.abs(values.numpy()-np.asarray([d['old_value'] for d in fixture]))))
    if len(fixture)!=48 or not math.isfinite(logp_error) or not math.isfinite(value_error) or logp_error>1e-4 or value_error>2e-4:raise ValueError('Actual untrained currentCPU trajectory parity differs')
    cp_equivalence=[]
    for row in receipts:
        if row['path']=='checkpoint.pt' or row['path'].startswith('checkpoint-commits/objects/cp-'):
            cp=load_torch(raw/row['path'],map_location='cpu',weights_only=False)
            if any(not same_tree(cp[k],saved[k]) for k in ['model','optimizer','rng','cpu_sampler_rng','updates']):raise ValueError('Published/hot full learning states differ')
            cp_equivalence.append({k:row[k] for k in ['path','sha256','bytes']})
    attempts=[a for rid in config['original_training_run_ids'] for a in read_json(p/'runs'/rid/'attempts.json')]
    if len(attempts)!=2 or not all(a['status']=='failed' for a in attempts):raise ValueError('Both original attempts must remain consumed and failed')
    log=(raw/'logs/payload.log').read_text(encoding='utf-8')
    if 'OSError: Hybrid workspace capacity reached' not in log or "atomic_json(output/'series'" not in log:raise ValueError('Different direct failure')
    result={'at':utcnow(),'complete':True,'engineering_only':True,'hypothesis_id':'TR08','classification':'invalid_incomplete_training_target_and_attempts_exhausted','promotion_eligible':False,
        'source_run_id':source.name,'archive_run_id':archive.name,'full_archive_files_verified':len(receipts),'full_archive_raw_bytes':sum(r['bytes'] for r in receipts),
        'source_point':point,'trained_decisions':24837,'target_decisions':32768,'committed_PPO_updates':3,'full_model_tensors':127,'optimizer_states':97,
        'all_committed_model_optimizer_and_RNG_payloads_exact_and_finite':True,'all_checkpoint_payloads':cp_equivalence,'consumed_v2_BO3_retained':51,
        'current_committed_untrained_BO3':12,'current_committed_untrained_decisions':1706,'orphan_complete_BO3':1,'orphan_complete_series_id':167,'orphan_complete_untrained_decisions':orphan_decisions,
        'all_untrained_complete_BO3':13,'all_untrained_complete_decisions':1706+orphan_decisions,'orphan_not_in_original_committed_progress_or_training':True,
        'CPU_sampler_streams_verified':all_streams,'actor_information_isolation_passed':True,'CPU_current48_logp_max_error':logp_error,'CPU_current48_value_max_error':value_error,
        'old_previous_point_consumed_refs':102,'latest_point_pending_refs':0,'retained_consumed_refs_not_released_before_new_sampling':True,
        'direct_failure':'unchanged10GiB working-set admission while writingBO3summary167','payload_log_sha256':sha256(raw/'logs/payload.log'),
        'attempts':attempts,'original_total_training_attempt_seconds':sum(a['seconds'] for a in attempts),'remaining_original_attempts':0,'strength_effect':None,
        'old_costs_attempts_rawruns_stops_champion_finalpool_unchanged':True,'source_archives_not_modified':True,
        'next_action':'CloseinvalidTR08; no third or renamedcontinuation. ActualACKlagretention barrier repair andcontrolledacceptance before a truly distinct freshly initialized H002 hypothesis.'}
    atomic_json(output/'analysis.json',result);atomic_json(output/'metrics.json',result)
