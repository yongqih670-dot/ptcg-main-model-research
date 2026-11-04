"""Finite execution of the one directly authorized +100k-and-evaluate protocol.

This launches existing bounded project runners sequentially. It creates no Goal,
chat, agent, timer, scientific branch, or open-ended search loop.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import psutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ptcg_lab.io import read_json,atomic_json,sha256,utcnow
from ptcg_lab.tr13_scope import AUTH,BASE,TARGET
from ptcg_lab.remote_lease import validate_no_unreconciled_remote

DEST=ROOT/'research/autonomy/activations/TR13-volume-100000-20261030'
ATTEMPT_GROSS_SECONDS=0.
EXECUTOR_MANIFEST_SHA=None


def verify_sources():
    path=DEST/'executor-source-manifest.json'
    if EXECUTOR_MANIFEST_SHA is not None and sha256(path)!=EXECUTOR_MANIFEST_SHA:
        raise ValueError('Finite executor source manifest changed')
    manifest=read_json(path)
    for name,digest in manifest['files'].items():
        if sha256(ROOT/name)!=digest:raise ValueError('Reviewed executor source changed: '+name)


def reference(path):
    p=Path(path)
    if not p.is_absolute():p=ROOT/p
    return {'path':p.relative_to(ROOT).as_posix(),'sha256':sha256(p)}


def idle_state():
    state=read_json(ROOT/'research/autonomy/state.json')
    if state.get('active_job') or state.get('active_run_id'):raise RuntimeError('Another controlled job is active')
    validate_no_unreconciled_remote(ROOT,state)
    phase=state['forward_phases']['TR13']
    if phase.get('user_stop') or state.get('current_user_pause') or (ROOT/'research/autonomy/STOP').exists():
        raise RuntimeError('User stop/pause applies')
    return state


def register_config(config,label):
    phase=idle_state()['forward_phases']['TR13']
    if phase['authority']!=AUTH:raise ValueError('Authority changed')
    path=DEST/'configs'/(label+'.json')
    if path.exists():raise FileExistsError(path)
    worker_count=config.get('workers',1) if config.get('job_kind')=='evaluation' else 1
    config=dict(config,project_path=str(ROOT),forward_phase='TR13',TR13_authorization=AUTH,
                workers=worker_count,maximum_attempts=1)
    atomic_json(path,config)
    key=hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    registry=read_json(ROOT/'research/autonomy/TR13-approved-configurations.json')
    registry['configs'][key]={'path':path.relative_to(ROOT).as_posix(),'role':'engineering' if config['engineering_only'] else 'research','registered_at':utcnow()}
    atomic_json(ROOT/'research/autonomy/TR13-approved-configurations.json',registry)
    return path


def publish(stage,**fields):
    atomic_json(DEST/'executor-progress.json',{'at':utcnow(),'pid':os.getpid(),'stage':stage,**fields})
    print(json.dumps({'at':utcnow(),'stage':stage,**fields}),flush=True)


def run_job(config,label,research,seconds,resume_run_id=None):
    global ATTEMPT_GROSS_SECONDS
    verify_sources()
    if resume_run_id:
        path=ROOT/'runs'/resume_run_id/'config.json'
        if read_json(path)!=config:raise ValueError('Reserved raw configuration changed')
    else:path=register_config(config,label)
    publish('launching_registered_job',label=label,research=research,configuration=reference(path))
    log=DEST/'executor-logs'/(label+'.log');log.parent.mkdir(parents=True,exist_ok=True)
    command=[str(ROOT/'.venv/Scripts/python.exe'),'-B',str(ROOT/'scripts/run_experiment.py'),
             '--config',str(path),'--label',label,'--wall-seconds',str(seconds),
             '--hypothesis','TR13 frozen independent +100000 valid unique decisions and endpoint evaluation']
    if research:command.append('--research')
    if resume_run_id:command+=['--resume',str(ROOT/'runs'/resume_run_id)]
    with log.open('xb') as stream:
        result=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
    state=idle_state();rid=resume_run_id or state['last_run_id'];run=ROOT/'runs'/rid
    if result.returncode:raise RuntimeError(f'{label} runner failed; preserve registered raw {rid}; see {log}')
    if read_json(run/'config.json')!=read_json(path):raise ValueError('Runner result belongs to a different configuration')
    attempts=read_json(run/'attempts.json')
    ATTEMPT_GROSS_SECONDS+=sum(row['seconds'] for row in attempts)
    if result.returncode or attempts[-1]['status'] not in ('completed','checkpointed'):
        raise RuntimeError(f'{label} did not complete safely; retained run {rid}; see {log}')
    publish('registered_job_terminal',run_id=rid,status=attempts[-1]['status'])
    return rid,read_json(run/'metrics.json')


def audit_source(rid,label):
    state=idle_state();phase=state['forward_phases']['TR13']
    cfg={'job_kind':'compact_audit','audit_variant':'tr13_full_archive','hypothesis_id':'TR13_ARCHIVE_ACCEPTANCE',
        'budget_hypothesis_id':'TR13','engineering_only':True,'training_wall_seconds':1800,
        'source_run_id':rid,'baseline_update_digest':phase.get('baseline_update_digest'),
        'primary_change':'Verify received original bytes and both full recovery points; never train or reencode model.',
        'success_criteria':'Full raw/encoded SHA256, two server/local semantic recovery points, exact inherited history and unique first-epoch coverage.'}
    if phase.get('acceptance_reuse'):
        cfg['acceptance_reuse']=phase['acceptance_reuse']
        cfg['primary_change']='Reuse accepted implementation; check latest complete state and new original-byte archives only.'
        cfg['success_criteria']='Latest state/cursor/model/optimizer/RNG, full archive ACKs and unique coverage valid; retain two recovery points.'
    audit_id,proof=run_job(cfg,label,False,1800)
    if not proof.get('complete') or not proof.get('local_and_server_full_recovery_verified'):raise ValueError('Full archive audit failed')
    state=idle_state();phase=state['forward_phases']['TR13']
    phase.update(latest_full_state_source=rid,latest_full_state_audit=reference('runs/'+audit_id+'/metrics.json'),
        latest_archive_catalog=reference('runs/'+audit_id+'/verified-archive-catalog.json'),
        baseline_update_digest=proof['baseline_update_digest'],actual_unique=proof['actual_unique'],
        actual_additional_unique=proof['new_unique'],latest_pending_decisions=proof['latest_pending_decisions'])
    phase.setdefault('accepted_sources',[]).append({'run_id':rid,'audit_run_id':audit_id,'proof':phase['latest_full_state_audit']})
    state['updated_at']=utcnow();atomic_json(ROOT/'research/autonomy/state.json',state)
    if phase.get('continuation_complete_tail'):
        phase.pop('continuation_complete_tail');atomic_json(ROOT/'research/autonomy/state.json',state)
    return audit_id,proof


def make_training(segment,proof):
    state=idle_state();phase=state['forward_phases']['TR13'];rid=phase['latest_full_state_source'];source=ROOT/'runs'/rid
    template=phase.get('remote_configuration_template',{'path':'research/autonomy/TR13-native-GPU-acceptance.config.json'})
    if template.get('sha256') and sha256(ROOT/template['path'])!=template['sha256']:raise ValueError('TR13 resource template changed')
    config=read_json(ROOT/template['path'])
    lease=read_json(source/'remote-job.json');manifest=source/'remote-training-manifest.json'
    config.update(hypothesis_id='TR13',budget_hypothesis_id='TR13',engineering_only=False,parent_acceptance_only=False,
        no_training_updates=False,training_wall_seconds=28800,source_run_id=rid,
        source_v2={'index_sha256':proof['source_v2_index_sha256'],'point':proof['latest_point']},
        source_point=proof['latest_point'],source_state_audit=phase['latest_full_state_audit'],
        existing_restore_source={'run_id':rid,'manifest_sha256':sha256(manifest),'identity':lease['identity'],
                                 'lease_epoch':lease['lease_epoch'],'cursor':proof['latest_point']['cursor']},
        source_remote_evidence={n:sha256(source/'remote-results'/n) for n in ['recovery-v2.json','payload-config.json','league.json','framework-v2-cold-recovery.json'] if (source/'remote-results'/n).exists()},
        source_evidence={n:sha256(source/n) for n in ['metrics.json','attempts.json','remote-training-manifest.json','remote-job.json','remote-terminal.json'] if (source/n).exists()},
        source_inflight=None,complete_postcheckpoint_tail=None,GPU_cursor_difference_required=segment==1,
        evaluation_protocol=reference('research/autonomy/TR13_EVALUATION_FROZEN.json'),
        primary_change='Only +100000 successful unique first-epoch decisions from full v15 state; frozen single endpoint206935.',
        success_criteria='Stop at the fixed whole-update endpoint; full state/archives before independent frozen evaluation.')
    config.pop('acceptance_identity',None)
    if phase.get('acceptance_reuse'):
        config['acceptance_reuse']=phase['acceptance_reuse']
        config['GPU_cursor_difference_required']=False
    if phase.get('require_exact_restored_GPU_state'):config['require_exact_restored_GPU_state']=True
    if phase.get('continuation_complete_tail'):
        tail=phase['continuation_complete_tail']
        if sha256(ROOT/tail['path'])!=tail['sha256']:raise ValueError('Accepted complete tail changed')
        config['complete_postcheckpoint_tail']=read_json(ROOT/tail['path'])
    numeric=config['numeric_reference'];zero=phase['native_zero_run_id']
    config['retained_numeric_reference']={'path':f"jobs/{zero}/stage/{numeric['sha256']}.blob",'sha256':numeric['sha256'],'bytes':numeric['bytes']}
    existing={x['sha256']:x for x in config.get('local_archive_sources',[])}
    for accepted in phase['accepted_sources']:
        cat=read_json(ROOT/'runs'/accepted['audit_run_id']/'verified-archive-catalog.json')
        for item in cat['objects']:existing[item['sha256']]=item
    config['local_archive_sources']=list(existing.values())
    return config


def publish_volume(proof):
    state=idle_state();phase=state['forward_phases']['TR13'];path=ROOT/'research/autonomy/CURRENT_VOLUME.json'
    old=read_json(path);backup=DEST/'before/research/autonomy/CURRENT_VOLUME.json'
    if not backup.exists():backup.parent.mkdir(parents=True,exist_ok=True);backup.write_bytes(path.read_bytes())
    baseline=read_json(backup)
    families=[f for f in old['families'] if f['hypothesis_id']!='TR13']
    family={'hypothesis_id':'TR13','canonical_training_family':phase['training_run_ids'][0],
        'training_family_members':phase['training_run_ids'],'scientific_status':'pending_frozen_independent_endpoint_evaluation',
        'trained_unique_decisions':proof['new_unique'],'PPO_updates':proof['new_PPO_updates'],
        'inherited_parent_decisions_not_counted_again':BASE,
        'inherited_pending_first_epoch_coverage':proof['inherited_pending_first_epoch_coverage'],
        'new_sampling_first_epoch_coverage':proof['new_sampling_first_epoch_coverage'],
        'full_state_audit':phase['latest_full_state_audit'],'promotion_eligible':False}
    families.append(family);old['families']=families
    old.update(at=utcnow(),actual_unique_training_decisions=baseline['actual_unique_training_decisions']+proof['new_unique'],
        remaining_unique_training_decisions=max(0,old['goal_unique_actual_PPO_training_decisions']-baseline['actual_unique_training_decisions']-proof['new_unique']),
        total_committed_PPO_updates=baseline['total_committed_PPO_updates']+proof['new_PPO_updates'],
        known_PPO_updates_lower_bound=baseline['total_committed_PPO_updates']+proof['new_PPO_updates'],
        TR13_incremental_audit=phase['latest_full_state_audit'],pre_TR13_ledger_preserved=reference(backup),
        scope='Through TR13 latest fully archived committed endpoint; inherited history, repeated epochs and unoptimized pending excluded.')
    atomic_json(path,old)


def materialize_agent(proof):
    endpoint=proof['endpoint_model']
    if not endpoint:raise ValueError('No fixed endpoint model')
    source=ROOT/'agents/learned/tr12-early-stop-v15';target=ROOT/'agents/learned/tr13-volume-100000-endpoint'
    if target.exists():raise FileExistsError(target)
    shutil.copytree(source,target,ignore=shutil.ignore_patterns('model.pt','spec.json','__pycache__','*.pyc'))
    shutil.copyfile(ROOT/endpoint['path'],target/'model.pt')
    if sha256(target/'model.pt')!=endpoint['sha256']:raise ValueError('Endpoint byte copy differs')
    spec=read_json(source/'spec.json');spec.update(id='tr13-volume-100000-endpoint',path=target.relative_to(ROOT).as_posix(),
        model_sha256=endpoint['sha256'],training_run_id=idle_state()['forward_phases']['TR13']['latest_full_state_source'],
        training_config_sha256=sha256(ROOT/'research/autonomy/TR13-training-frozen.json'))
    atomic_json(target/'spec.json',spec);return spec


def evaluate_endpoint(spec):
    frozen=read_json(ROOT/'research/autonomy/TR13_EVALUATION_FROZEN.json');results={}
    common={'job_kind':'evaluation','hypothesis_id':'TR13','budget_hypothesis_id':'TR13','engineering_only':False,
        'save_all_replays':True,'stop_on_internal_fallback':True,'policy_seed':frozen['policy_seed'],
        'bootstrap_samples':20000,'bootstrap_seed':frozen['bootstrap_seed'],'no_posthoc_extension':True,
        'native_environment_seed_not_exposed':True,'endpoint_selection':'last fixed-volume committed endpoint',
        'evaluation_protocol':reference('research/autonomy/TR13_EVALUATION_FROZEN.json')}
    primary=copy.deepcopy(frozen['endpoint_primary_cell']);primary['candidate']=spec
    for label,count,cells in [('reliability',50,[dict(primary,cell_id='reliability-vs-ppo',series_count=50)]),
                              ('primary',1600,[primary]),
                              ('development',800,[dict(c,candidate=spec) for c in frozen['endpoint_development_cells']])]:
        available=psutil.virtual_memory().available
        workers=min(8,os.cpu_count() or 1,max(1,int(available//(2*2**30))))
        cfg=dict(common,series_count=count,matchups=cells,evaluation_stage='TR13_'+label,workers=workers)
        rid,metrics=run_job(cfg,'tr13-'+label,True,21600)
        if not metrics.get('complete') or metrics.get('series')!=count:
            # Exact durable row count is authoritative for the evaluator schema.
            from ptcg_lab.io import read_series
            if not metrics.get('complete') or len(read_series(ROOT/'runs'/rid/'series.jsonl'))!=count:
                raise ValueError('Frozen endpoint evaluation incomplete')
        results[label]=rid
    return results


def activate_reuse(first_run):
    """The real first CUDA interruption proof is the sole remaining prerequisite."""
    state=idle_state();phase=state['forward_phases']['TR13'];run=ROOT/'runs'/first_run
    path=run/'remote-results/framework-v2-GPU-cursor-difference.json';proof=read_json(path)
    required=['complete','fresh_CUDA_process_after_positive_step','original_PPO_function_used']
    if not all(proof.get(k) for k in required) or proof.get('scientific_decision_credit')!=0:
        raise ValueError('Necessary first actual CUDA partial-PPO acceptance is incomplete')
    if not (proof.get('model_optimizer_all_RNG_permutation_logs_exact') or
            (proof.get('checkpoint_model_optimizer_all_RNG_permutation_exact') and proof.get('RNG_permutation_input_counters_exact') and proof.get('PPO_continuation_within_measured_GPU_repeatability'))):
        raise ValueError('Exact restored state or bounded CUDA repeatability proof is missing')
    from ptcg_lab.tr13_acceptance_reuse import AUTH as REUSE_AUTH,envelope,validate_local
    doc=read_json(DEST/'reuse-candidate.json')
    actual_source=read_json(run/'remote-source.json')
    if any(actual_source.get(k)!=v for k,v in doc['scientific_sources'].items()):
        raise ValueError('First scientific implementation differs from accepted implementation')
    doc.update(complete=True,completed_at=utcnow());doc.pop('pending_necessary_item',None)
    doc['accepted_evidence']['first_real_CUDA_partial_PPO']=reference(path)
    target=DEST/'SEGMENT_ACCEPTANCE_REUSE.json'
    if target.exists():raise FileExistsError('Reuse certificate already exists; reconcile before reactivation')
    atomic_json(target,doc);scope=envelope(doc,reference(target));validate_local(ROOT,scope)
    phase.update(acceptance_reuse=scope,GPU_cursor_difference=reference(path))
    charge_path=DEST/'reuse-preparation-CPU-charge.json'
    if charge_path.exists() and not phase.get('reuse_preparation_CPU_charge'):
        charge=read_json(charge_path)
        if charge['authority']!=REUSE_AUTH or charge['budget_upper_bound_seconds']!=300:raise ValueError('Reuse preparation charge differs')
        state['budget_used']['cpu_job_wall_hours']+=charge['budget_upper_bound_seconds']/3600
        phase['reuse_preparation_CPU_charge']=reference(charge_path)
    policy_path=ROOT/'research/autonomy/policy.json';policy=read_json(policy_path)
    policy['forward_authorizations']['TR13_segment_acceptance_reuse']=REUSE_AUTH
    atomic_json(policy_path,policy);atomic_json(ROOT/'research/autonomy/state.json',state)
    publish('prior_acceptance_reused_future_segments',certificate=scope['reference'],latest_recovery_only=True)


def wait_adopted_run(rid):
    publish('waiting_existing_first_scientific_run',run_id=rid)
    start=time.monotonic()
    while True:
        state=read_json(ROOT/'research/autonomy/state.json')
        if state.get('current_user_pause') or state['forward_phases']['TR13'].get('user_stop') or (ROOT/'research/autonomy/STOP').exists():
            raise InterruptedError('User stopped the authorized stage')
        active=state.get('active_run_id')
        if active is None:break
        if active!=rid:raise ValueError('Unexpected active raw during coordinator handoff')
        if time.monotonic()-start>86400:raise TimeoutError('Existing first-raw safety window elapsed')
        time.sleep(5)
    state=idle_state();phase=state['forward_phases']['TR13']
    adopted_config=read_json(ROOT/'runs'/rid/'config.json')
    repair=adopted_config.get('GPU_cursor_repair_only') or adopted_config.get('TR13_timeout_acceptance')
    if repair:
        if rid not in phase['engineering_run_ids']:raise ValueError('Unregistered remaining CUDA repair')
    elif phase['training_run_ids']!=[rid]:raise ValueError('Handoff does not identify the first registered scientific raw')
    attempts=read_json(ROOT/'runs'/rid/'attempts.json')
    if attempts[-1]['status'] not in ('completed','checkpointed') or attempts[-1]['exit_code']!=0:
        raise RuntimeError('Existing first raw did not terminate safely; no blind restart')
    return time.monotonic()


def activate_timeout_recovery(rid):
    state=idle_state();phase=state['forward_phases']['TR13'];run=ROOT/'runs'/rid
    cfg=read_json(run/'config.json');test=cfg['TR13_timeout_acceptance']
    report_path=run/'remote-results/TR13-timeout-recovery-acceptance.json';report=read_json(report_path)
    if not report.get('complete') or not report.get('official_action_and_total_run_timeouts_unchanged') or not report['latest_GPU_recovery']['full_CPU_and_GPU_values_RNG_exact']:
        raise ValueError('Latest GPU recovery/official timing difference did not pass')
    local=test['local_recovery_audit'];proof=read_json(ROOT/local['path'])
    if proof['actual_unique']!=report['actual_unique'] or proof['latest_point']!=report['latest_point'] or proof['source_run_id']!=report['source_run_id']:
        raise ValueError('Local and fresh GPU recovery points differ')
    combined=dict(proof,local_and_server_full_recovery_verified=True,server_fresh_GPU_recovery_pending=False,
        latest_GPU_and_timing_proof=reference(report_path),original_local_CPU_proof=local)
    path=DEST/'timeout-repair/FULL_v17_RECOVERY_ACCEPTED.json';atomic_json(path,combined)
    from ptcg_lab.tr13_acceptance_reuse import envelope,validate_local,scientific_sources
    basis=phase['acceptance_reuse'];doc=copy.deepcopy(basis['document'])
    difference=read_json(ROOT/test['difference_reference']['path'])
    doc.update(at=utcnow(),previous_certificate=basis['reference'],targeted_difference=test['difference_reference'],
        scientific_sources=scientific_sources(ROOT),collector_runtime_sources=difference['new_runtime_sources'])
    doc['accepted_evidence']['latest_v17_recovery_and_official_timing']=reference(report_path)
    cert=DEST/'timeout-repair/SEGMENT_ACCEPTANCE_REUSE_v2.json';atomic_json(cert,doc)
    scope=envelope(doc,reference(cert));validate_local(ROOT,scope)
    phase.update(acceptance_reuse=scope,latest_full_state_source=report['source_run_id'],latest_full_state_audit=reference(path),
        latest_archive_catalog=reference(Path(local['path']).parent/'verified-archive-catalog.json'),
        actual_unique=proof['actual_unique'],actual_additional_unique=proof['new_unique'],latest_pending_decisions=proof['latest_pending_decisions'])
    audit_id=Path(local['path']).parent.name
    phase.setdefault('accepted_sources',[]).append({'run_id':report['source_run_id'],'audit_run_id':audit_id,'proof':reference(path)})
    phase['recovery_required_from_positive_updates']['status']='verified'
    phase['timeout_recovery']=reference(report_path)
    state['current_phase']='TR13_full_archive_and_training';state['status']='running'
    atomic_json(ROOT/'research/autonomy/state.json',state)
    publish('latest_positive_state_recovered',actual_unique=proof['actual_unique'],source_run=report['source_run_id'],latest_generation=proof['latest_point']['generation'])
    return combined


def main():
    global EXECUTOR_MANIFEST_SHA
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--zero-run');g.add_argument('--adopt-run');g.add_argument('--resume-source');g.add_argument('--resume-registered-run')
    p.add_argument('--wait-for-zero-run',action='store_true');a=p.parse_args();selected=a.resume_registered_run or a.resume_source or a.adopt_run or a.zero_run
    if not 12<=len(selected)<=110 or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in selected):
        raise ValueError('Zero-update run id escaped its scope')
    EXECUTOR_MANIFEST_SHA=sha256(DEST/'executor-source-manifest.json');verify_sources()
    lock=DEST/'executor.lock'
    if lock.exists():raise RuntimeError('Executor lock exists; verify process identity before any continuation')
    with lock.open('x',encoding='utf-8') as stream:json.dump({'pid':os.getpid(),'created':psutil.Process().create_time(),'at':utcnow()},stream)
    executor_started=time.monotonic()
    try:
        if a.wait_for_zero_run and not a.adopt_run and not a.resume_source:
            publish('waiting_existing_zero_update_job',run_id=a.zero_run)
            waiting_started=time.monotonic()
            while True:
                current=read_json(ROOT/'research/autonomy/state.json');active=current.get('active_run_id')
                if active is None:break
                if active!=a.zero_run:raise ValueError('Unexpected concurrent job; no second controller admitted')
                if current.get('current_user_pause') or current['forward_phases']['TR13'].get('user_stop') or (ROOT/'research/autonomy/STOP').exists():
                    raise InterruptedError('User stopped the authorized stage')
                if time.monotonic()-waiting_started>86400:raise TimeoutError('Original zero-update safety window expired')
                time.sleep(5)
            executor_started=time.monotonic()  # Waiting for the already billed job is not a new CPU charge.
        verify_sources()
        reserved=None
        if a.resume_registered_run:
            state=idle_state();phase=state['forward_phases']['TR13'];reserved=a.resume_registered_run
            original=read_json(ROOT/'runs'/reserved/'config.json')
            if phase['training_run_ids'][-1]!=reserved or len(phase['training_run_ids'])!=4:
                raise ValueError('Only the final already reserved raw can be resumed here')
            if (ROOT/'runs'/reserved/'attempts.json').exists() or (ROOT/'runs'/reserved/'remote-job.json').exists():
                raise ValueError('Reserved raw already attempted; no blind retry')
            if original['source_run_id']!=phase['latest_full_state_source'] or not original.get('require_exact_restored_GPU_state'):
                raise ValueError('Reserved raw differs from the accepted latest state')
            proof=read_json(ROOT/phase['latest_full_state_audit']['path']);first_segment=4
        elif a.resume_source:
            state=idle_state();phase=state['forward_phases']['TR13']
            if phase['latest_full_state_source']!=a.resume_source or not phase.get('require_exact_restored_GPU_state'):
                raise ValueError('Resume source is not the accepted latest complete state')
            proof=read_json(ROOT/phase['latest_full_state_audit']['path'])
            publish_volume(proof);first_segment=len(phase['training_run_ids'])+1
        elif a.adopt_run:
            executor_started=wait_adopted_run(a.adopt_run)
            adopted=read_json(ROOT/'runs'/a.adopt_run/'config.json')
            if adopted.get('TR13_timeout_acceptance'):
                proof=activate_timeout_recovery(a.adopt_run);publish_volume(proof)
                first_segment=len(idle_state()['forward_phases']['TR13']['training_run_ids'])+1
            elif adopted.get('GPU_cursor_repair_only'):
                activate_reuse(a.adopt_run)
                phase=idle_state()['forward_phases']['TR13'];proof=read_json(ROOT/phase['latest_full_state_audit']['path'])
                first_segment=len(phase['training_run_ids'])+1
            else:
                activate_reuse(a.adopt_run)
                _,proof=audit_source(a.adopt_run,'tr13-segment-001-archive')
                publish_volume(proof);first_segment=2
        else:
            state=idle_state();zero=read_json(ROOT/'runs'/a.zero_run/'metrics.json')
            if not zero.get('complete') or not zero.get('parent_acceptance_passed') or zero.get('optimizer_steps_executed')!=0:
                raise ValueError('Real native/GPU zero-update acceptance must finish first')
            charge_path=DEST/'local-preflight-CPU-budget-charge.json';phase=state['forward_phases']['TR13']
            if not phase.get('local_preflight_CPU_budget_charge'):
                charge=read_json(charge_path)
                if charge['authority']!=AUTH or charge['budget_upper_bound_seconds']!=300:raise ValueError('Local preflight conservative charge differs')
                state['budget_used']['cpu_job_wall_hours']+=charge['budget_upper_bound_seconds']/3600
                phase['local_preflight_CPU_budget_charge']=reference(charge_path)
            phase.update(native_GPU_acceptance=reference('runs/'+a.zero_run+'/metrics.json'),
                native_zero_run_id=a.zero_run,evaluation_protocol=reference('research/autonomy/TR13_EVALUATION_FROZEN.json'))
            state['current_phase']='TR13_full_archive_and_training';atomic_json(ROOT/'research/autonomy/state.json',state)
            _,proof=audit_source(a.zero_run,'tr13-v15-full-archive');first_segment=1
        for segment in range(first_segment,5):
            if proof['actual_unique']>=TARGET:break
            config=read_json(ROOT/'runs'/reserved/'config.json') if reserved else make_training(segment,proof)
            label=f'tr13-volume-segment-{segment:03d}'+('-reserved-resume-002' if reserved else '')
            rid,_=run_job(config,label,True,28800,resume_run_id=reserved)
            reserved=None
            if segment==1:activate_reuse(rid)
            _,proof=audit_source(rid,f'tr13-segment-{segment:03d}-archive')
            publish_volume(proof)
            if segment==1:
                cursor=ROOT/'runs'/rid/'remote-results/framework-v2-GPU-cursor-difference.json'
                if not cursor.exists() or not read_json(cursor).get('complete'):raise ValueError('Actual GPU cursor proof missing')
                state=idle_state();state['forward_phases']['TR13']['GPU_cursor_difference']=reference(cursor)
                atomic_json(ROOT/'research/autonomy/state.json',state)
        if proof['actual_unique']<TARGET:raise RuntimeError('TR13 bounded segment allowance ended before target; full state retained')
        state=idle_state();state['current_phase']='TR13_fixed_endpoint_evaluation'
        state['forward_phases']['TR13'].update(training_finished=True,training_target_reached=proof['actual_unique'],
            actual_target_overshoot=proof['actual_unique']-TARGET,no_further_training=True)
        atomic_json(ROOT/'research/autonomy/state.json',state)
        spec=materialize_agent(proof);evaluations=evaluate_endpoint(spec)
        verify_sources()
        from ptcg_lab.tr13_analysis import finish
        finish(ROOT,proof,evaluations)
        overhead=max(0,time.monotonic()-executor_started-ATTEMPT_GROSS_SECONDS)
        state=idle_state();phase=state['forward_phases']['TR13']
        state['budget_used']['cpu_job_wall_hours']+=overhead/3600
        atomic_json(DEST/'executor-CPU-accounting.json',{'at':utcnow(),'CPU_job_seconds_charged':overhead,
            'method':'executor gross wall minus all controlled-attempt gross walls; excludes their shared waits and avoids duplicate charges',
            'already_booked_attempt_gross_seconds':ATTEMPT_GROSS_SECONDS})
        phase['executor_CPU_accounting']=reference(DEST/'executor-CPU-accounting.json')
        atomic_json(ROOT/'research/autonomy/state.json',state)
        continuation=read_json(ROOT/'research/autonomy/latest-continuation.json')
        continuation.update(at=utcnow(),budget_snapshot=state['budget_used'],TR13=phase)
        atomic_json(ROOT/'research/autonomy/latest-continuation.json',continuation)
        publish('complete',actual_additional_unique=proof['new_unique'],endpoint_evaluations=evaluations)
    except BaseException as exc:
        publish('needs_controller_attention',error=type(exc).__name__+': '+str(exc),no_blind_restart=True)
        raise
    finally:
        if lock.exists() and read_json(lock).get('pid')==os.getpid():lock.unlink()


if __name__=='__main__':main()
