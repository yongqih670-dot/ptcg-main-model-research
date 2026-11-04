"""Local full cold audit for TR09's sole original second attempt.

This module never admits a failed attempt or changes scientific data. A
separate numeric reference can use old visible inputs when the live tail is
empty. That reference is explicitly excluded from subsequent training.
"""
import copy
import gzip
import io
import json
import math
import os
import random
from pathlib import Path
import time
from .io import read_json,atomic_json,sha256,utcnow

FIRST='20261026T131453Z_tr09-epochs4-CPU32_382e9e'
CLEAN={'checkpointed_for_control','checkpointed_at_soft_deadline','checkpointed_before_update_at_soft_deadline',
       'partial_update_abandoned_at_deadline','partial_update_abandoned_for_control'}


def canonical_policy_version(value,expected):
    if type(value) not in (int,str) or type(value) is str and not value.isascii() or str(value)!=str(expected):
        raise ValueError('Pending decision policy version differs from canonical integer/string journal representation')
    return expected


def validate_clean_contract(source_id,config,attempts,metrics,progress,cursor,*,allow_proved_wrapper_failure=False):
    if source_id!=FIRST or (config.get('hypothesis_id'),config.get('check_variant'),config.get('training_attempt_index'))!=('TR09','fresh_scientific',1):
        raise ValueError('Only the exact original first TR09 may enter this clean recovery audit')
    normal=len(attempts)==1 and attempts[0].get('status')=='checkpointed' and attempts[0].get('exit_code')==0
    wrapper=(allow_proved_wrapper_failure and len(attempts)==1 and attempts[0].get('status')=='failed' and attempts[0].get('exit_code')==1)
    if not (normal or wrapper) or not 0<attempts[0]['seconds']<=7201:
        raise ValueError('One clean terminal first attempt and its full cost are required')
    if metrics.get('complete') or progress.get('stage') not in CLEAN:
        raise ValueError('Completed, failed or unexplained sources cannot use clean continuation')
    if any(progress.get(k)!=v for k,v in cursor.items()) or not 0<=cursor['trained_decisions']<32768:
        raise ValueError('Published cursor differs from the actual terminal progress')
    # A cooperative job checkpoint-stop uses this same phase and generic
    # message. Goal/user cancellation and supervisor stop locks are enforced
    # by the calling controller; a payload message cannot infer human intent.


def proved_wrapper_failure(source,raw):
    """The actual first controller failed at its final complete=False check.

    Preserve its failed status/exit1/cost. This specific proof only establishes
    a recoverable payload checkpoint, never a successful research result.
    """
    source=Path(source);raw=Path(raw)
    attempts=read_json(source/'attempts.json')
    if len(attempts)==1 and attempts[0].get('status')=='checkpointed':return None
    if (source.name!=FIRST or len(attempts)!=1 or attempts[0].get('status')!='failed'
            or attempts[0].get('exit_code')!=1 or abs(attempts[0]['seconds']-6278.281)>.01):
        raise ValueError('Different original controller failure requires its own actual analysis')
    expected={'stderr-01.log':'89063e54479c26b238b1c268fc0ea8f22ba1e88105b2fc55896845930f1f015e',
        'source/ptcg_lab/server_trial.py':'99c42f05df71d880e461512e855e33a35840ff097cfa58f6e2b804a368726eed',
        'remote-terminal.json':'a9941617dd612a5145e0ccefd9fffd33891c13a5380db98ae494e66e12786567',
        'remote-results/trial-metrics.json':'04e454a837a17b8beb8e330ef9f7208783d62a477a6c5fab37135db36ec5cf77'}
    if any(sha256(source/name)!=digest for name,digest in expected.items()):raise ValueError('Exact actual wrapper failure evidence changed')
    if sha256(raw/'trial-metrics.json')!=expected['remote-results/trial-metrics.json']:
        raise ValueError('Full cold payload metrics differ from the originally returned bytes')
    terminal=read_json(source/'remote-terminal.json');props=terminal['service_properties'];lease=read_json(source/'remote-job.json')
    if (terminal['identity']!=lease['identity'] or props.get('LoadState')!='loaded' or props.get('Result')!='success'
            or props.get('ExecMainStatus')!='0' or props.get('ControlGroup')!=''
            or props.get('InvocationID')!=lease['identity']['invocation_id']):
        raise ValueError('Positive original exit0 emptycgroup proof required, not default collected-unit fields')
    metrics=read_json(raw/'trial-metrics.json');progress=metrics.get('last_training_progress',{})
    if (metrics.get('complete') is not False or metrics.get('engineering_only') is not False
            or metrics.get('training_only') is not True or metrics.get('scientific_training') is not True
            or progress.get('stage')!='checkpointed_for_control'
            or tuple(progress.get(k) for k in ['version','trained_decisions','pending_decisions','next_series_id'])!=(2,16679,4112,130)):
        raise ValueError('Actual clean payload checkpoint fingerprint differs')
    return {'classification':'exact_original_final_wrapper_complete_false_rejection_after_payload_exit0',
        'actual_controller_status':'failed','actual_controller_exit_code':1,'actual_controller_seconds':6278.281,
        'actual_payload_exit_code':0,'actual_checkpoint_progress':progress,'original_evidence_sha256':expected,
        'no_old_status_cost_attempt_or_success_reinterpretation':True}


def run(config,output):
    import numpy as np
    import torch
    from .archive_ack import safe_target,archive_root
    from .archive_compressed import verify_archive
    from .archive_prefix import compact_record,load_prefix
    from .archive_state import validate_state
    from .checkpoint_refs import read_index
    from .remote_lease import validate_no_unreconciled_remote
    from .rl_core import load_full_actor,evaluate_decisions
    from .sampling_parity import validate_outputs
    from .tensor_io import load_torch,atomic_torch_save
    root=Path(config['project_path']).resolve();output=Path(output)
    if os.environ.get('PTCG_JOB_RUN_ID')!=output.name:raise RuntimeError('Controlled local audit required')
    deadline=float(os.environ['PTCG_JOB_DEADLINE_UNIX'])
    if not 0<deadline-time.time()<=240.5:raise ValueError('One bounded240s audit only')
    if config.get('local_audit_prior_run_ids'):
        spent=sum(a['seconds'] for rid in config['local_audit_prior_run_ids'] for a in read_json(root/'runs'/rid/'attempts.json'))
        if spent+deadline-time.time()>240.5:raise ValueError('Cumulative original local read-only audit budget exhausted')
    source=root/'runs'/FIRST;archive=root/'runs'/config['archive_run_id'];raw=archive/'archive'
    if archive.resolve().parent!=root/'runs' or sha256(archive/'file-receipts.json')!=config['archive_receipts_sha256']:
        raise ValueError('Full cold archive identity changed')
    meta=read_json(archive/'metrics.json');rows=read_json(archive/'file-receipts.json')
    if not meta.get('complete') or not meta.get('all_files_hash_verified_and_fsynced_locally') or meta['source_run_id']!=FIRST:
        raise ValueError('Complete original cold archive required')
    by_path={}
    for row in rows:
        path=safe_target(raw,row['path'])
        if path.stat().st_size!=row['bytes'] or sha256(path)!=row['sha256']:raise ValueError('Full cold original bytes changed')
        by_path[row['path']]={k:row[k] for k in ['path','bytes','sha256']}
        if time.time()>deadline-30:raise TimeoutError('Full source audit deadline')
    lease=read_json(source/'remote-job.json');validate_no_unreconciled_remote(root,{'remote_job':lease})
    point=read_index(raw,FIRST,lease['lease_epoch'])['commits'][-1];cursor=point['cursor']
    progress=read_json(raw/'progress.json');attempts=read_json(source/'attempts.json')
    original_config=read_json(source/'config.json')
    wrapper_proof=proved_wrapper_failure(source,raw)
    local_metrics=read_json(source/'metrics.json') if (source/'metrics.json').exists() else None
    if local_metrics is None:
        if wrapper_proof is None:raise ValueError('Missing controller metrics without the exact proved post-checkpoint wrapper failure')
        local_metrics=read_json(raw/'trial-metrics.json')
    validate_clean_contract(FIRST,original_config,attempts,local_metrics,progress,cursor,
        allow_proved_wrapper_failure=wrapper_proof is not None)
    protected=root/original_config['protected_actor_source_manifest_path']
    if root not in protected.resolve().parents or sha256(protected)!=original_config['protected_actor_source_manifest_sha256']:
        raise ValueError('Original full actor source manifest changed')
    bundle=read_json(raw/'training-manifest.raw.json')['training_config']['bundle_path'].replace('\\','/')+'/'
    for name,digest in read_json(protected).items():
        if name.replace('\\','/').startswith(bundle) and sha256(root/name)!=digest:
            raise ValueError('Original complete actor/encoder source changed')
    reference=root/original_config['matched_reference_path']
    if root not in reference.resolve().parents or sha256(reference)!=original_config['matched_reference_sha256']:
        raise ValueError('Original matched scientific reference changed')
    from .fresh_experiment import validate_single_change
    original_manifest=read_json(raw/'training-manifest.raw.json')
    if original_manifest['mode']!='fresh_scientific' or validate_single_change(original_manifest['training_config'],read_json(reference))!='TR09':
        raise ValueError('Original one-variable TR09 scientific settings differ')
    if sha256(raw/'checkpoint.pt')!=point['checkpoint']['sha256']:raise ValueError('Hot checkpoint differs from the published full point')
    saved=load_torch(raw/point['checkpoint']['path'],map_location='cpu',weights_only=False)
    if {k:saved[k] for k in cursor}!=cursor or len(saved['model'])!=127 or len(saved['optimizer']['state'])!=(97 if cursor['version'] else 0):
        raise ValueError('Full model/optimizer/cursor coverage differs')
    if set(saved['rng'])!={'python','numpy','torch','cuda'} or len(saved['rng']['cuda'])!=1 or 'cpu_sampler_rng' not in saved:
        raise ValueError('Complete global and CPU sampling RNG required')
    for tensor in [*saved['model'].values(),*[v for slots in saved['optimizer']['state'].values() for v in slots.values() if torch.is_tensor(v)]]:
        if not torch.isfinite(tensor).all():raise ValueError('Nonfinite full model/optimizer state')
    torch.Generator(device='cpu').set_state(saved['cpu_sampler_rng'])
    torch.Generator(device='cpu').set_state(saved['rng']['torch'])
    random.Random().setstate(saved['rng']['python']);np.random.RandomState().set_state(saved['rng']['numpy'])
    if any(s.dtype!=torch.uint8 or s.ndim!=1 or not s.numel() for s in saved['rng']['cuda']):
        raise ValueError('Incomplete CUDA RNG byte payload')
    if len(saved['updates'])!=cursor['version'] or sum(u['decisions'] for u in saved['updates'])!=cursor['trained_decisions']:
        raise ValueError('Committed update accounting changed')
    prefix=load_prefix(raw/point['prefix']['path'],point['prefix']['sha256'])
    state=validate_state(saved['archive_state'],prefix,point['prefix']['sha256'])
    # Verify the durable originals of every consumed record, including files
    # already retired on the server. No missing file is treated as disposable.
    ack_by_path={read_json(f)['object']:read_json(f) for f in (source/'online-archive').rglob('acks/*.json')}
    checked=set();all_summaries=[]
    for record in state['archived']:
        for name,digest in [(record['trajectory_path'],record['trajectory_sha256']),
                            (f"series/{record['series_id']:06d}.json",record['summary_sha256'])]:
            ack=ack_by_path.get(name)
            if not ack or ack['sha256']!=digest:raise ValueError('Consumed original lacks exact durable archive')
            if name.startswith('series/'):
                summary=json.loads(verify_archive(source/'online-archive',ack,collect=True))
                if summary['series_id']!=record['series_id'] or summary['rounds'] not in (2,3) or summary['statuses']!=['DONE','DONE']:
                    raise ValueError('Consumed complete BO3 summary differs')
                all_summaries.append(summary)
            else:verify_archive(source/'online-archive',ack)
            checked.add(name)
    current=[];records=[];streams=0;pending_paths=set()
    for path in sorted((raw/'series').glob('[0-9]'*6+'.json')):
        summary=read_json(path)
        if int(summary['policy_version'])!=cursor['version']:continue
        row=compact_record(summary,sha256(path));trajectory=raw/row['trajectory_path']
        if summary['rounds'] not in (2,3) or sha256(trajectory)!=row['trajectory_sha256']:raise ValueError('Current complete BO3 differs')
        data=load_torch(trajectory,map_location='cpu',weights_only=False)
        if data['series_id']!=row['series_id'] or data['policy_version']!=cursor['version']:raise ValueError('Mixed current policy source')
        torch.Generator(device='cpu').set_state(data['cpu_sampler_rng_after']);streams+=1
        for decision in data['decisions']:
            canonical_policy_version(decision['policy_version'],cursor['version'])
            if decision['series_id']!=row['series_id'] or not all(math.isfinite(decision[k]) for k in ['old_logp','old_value']):
                raise ValueError('Changed or nonfinite pending decision')
        current.extend(data['decisions']);records.append(row)
        all_summaries.append(summary)
        pending_paths.update([row['trajectory_path'],path.relative_to(raw).as_posix()])
    if records!=state['live_expected'] or pending_paths!={r['path'] for r in point['pending']} or len(current)!=progress['pending_decisions']:
        raise ValueError('Complete pending records, cursor or recovery reference coverage differs')
    if (len(all_summaries)!=cursor['next_series_id'] or len({r['series_id'] for r in all_summaries})!=len(all_summaries)
            or sum(r['learner_decisions'] for r in all_summaries)!=cursor['trained_decisions']+len(current)):
        raise ValueError('Unique original sampled/trained/BO3 accounting differs')
    league=read_json(raw/'league.json')
    for member in league['snapshots']:
        if sha256(raw/member['path'])!=member['sha256']:raise ValueError('Frozen history changed')
    # Numeric inputs are engineering evidence only. An empty pending tail must
    # not create a new BO3 or quietly inherit old off-policy training samples.
    fixture=copy.deepcopy(current[:48]);fixture_from_current=len(fixture)==48
    if not fixture_from_current:
        fixture=[]
        for record in state['archived']:
            ack=ack_by_path[record['trajectory_path']];packed=safe_target(archive_root(source/'online-archive'),ack['archive_relative'])
            with gzip.open(packed,'rb') if ack.get('storage') else packed.open('rb') as stream:
                data=torch.load(io.BytesIO(stream.read()),map_location='cpu',weights_only=False)
            fixture.extend(copy.deepcopy(data['decisions'][:48-len(fixture)]))
            if len(fixture)==48:break
    if len(fixture)!=48:raise ValueError('48 real source decisions required for a separate numeric reference')
    torch.set_num_threads(2)
    with torch.random.fork_rng(devices=[]):
        model,features,_=load_full_actor(root/'agents/frozen/ppo-zero-anchor',read_json(raw/'payload-config.json')['parent_model_sha256'],'cpu',reset_value=False)
        model.load_state_dict(saved['model'],strict=True);model.eval()
        lo,hi=features.b.OFFSETS['oracle']
        for decision in fixture+current:
            if any(step['rows'][:,lo:hi].any() or not np.isfinite(step['rows']).all() for step in decision['steps']):
                raise ValueError('Actor information isolation failed')
        with torch.no_grad():logp,values,_=evaluate_decisions(model,fixture,features.b.STOP)
    original_parity=validate_outputs(fixture,logp,values) if fixture_from_current else None
    for i,decision in enumerate(fixture):
        decision['old_logp']=float(logp[i]);decision['old_value']=float(values[i]);decision['engineering_reference_only_not_training']=True
    reference=output/'numeric-reference.pt'
    atomic_torch_save(reference,{'role':'TR09_engineering_recovery_reference_only_not_training','source_run_id':FIRST,
        'source_checkpoint_sha256':point['checkpoint']['sha256'],'model':saved['model'],'fixture':fixture,
        'fixture_logp':logp.cpu(),'fixture_values':values.cpu(),'original_pending_data_changed':False})
    inflight=by_path.get('inflight.json')
    result={'at':utcnow(),'complete':True,'engineering_only':True,'classification':'TR09_original_checkpoint_full_recovery_audited',
        'hypothesis_id':'TR09','source_run_id':FIRST,'archive_run_id':archive.name,'source_point':point,
        'full_archive_files_verified':len(rows),'full_archive_raw_bytes':sum(r['bytes'] for r in rows),
        'consumed_archived_originals_full_SHA_verified':len(checked),'trained_decisions':cursor['trained_decisions'],
        'PPO_updates':cursor['version'],'pending_decisions':len(current),'current_policy_complete_BO3':len(records),
        'complete_BO3':len(all_summaries),'individual_completed_games':sum(r['rounds'] for r in all_summaries),
        'sampled_unique_decisions':sum(r['learner_decisions'] for r in all_summaries),
        'unique_actual_PPO_training_decisions':cursor['trained_decisions'],
        'counting_family_root':FIRST,'epochs_resumes_migration_and_engineering_fixtures_not_new_samples':True,
        'full_model_tensors':127,'optimizer_states':len(saved['optimizer']['state']),'CPU_sampler_streams_verified':streams,
        'all_full_model_optimizer_RNG_checkpoint_payloads_verified':True,'actor_information_isolation_passed':True,
        'all_frozen_historical_members_verified':league['snapshots'],'additional_complete_tail':[],
        'source_manifest_sha256':sha256(raw/'training-manifest.raw.json'),'source_progress_sha256':sha256(raw/'progress.json'),
        'source_index_sha256':sha256(raw/'checkpoint-commits/index.json'),'source_payload_config_sha256':sha256(raw/'payload-config.json'),
        'source_league_sha256':sha256(raw/'league.json'),'source_inflight':inflight,'first_attempt_seconds':attempts[0]['seconds'],
        'numeric_reference':{'path':reference.relative_to(root).as_posix(),'sha256':sha256(reference),'bytes':reference.stat().st_size},
        'numeric_reference_from_current_pending':fixture_from_current,'actual_current_CPU_rollout_parity':original_parity,
        'reference_fixture_never_enters_training':True,'remaining_original_attempts':1,'source_modified':False,'promotion_eligible':False}
    result.update(original_controller_status=attempts[0]['status'],original_controller_exit_code=attempts[0]['exit_code'],
        controller_post_checkpoint_failure_proof=wrapper_proof)
    atomic_json(output/'analysis.json',result)
    from .scale_accounting import build
    build(root,output,result)
    atomic_json(output/'metrics.json',result)
