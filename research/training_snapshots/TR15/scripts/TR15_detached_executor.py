"""One finite coordinator: required differences, then authorized TR15 and endpoint."""
from pathlib import Path
import copy,json,os,sys,time
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R));sys.path.insert(0,str(R/'scripts'))
import execute_TR15 as e
from ptcg_lab.io import read_json,atomic_json,sha256,utcnow
from ptcg_lab.tr15_framework import validate_report,validate_manifest_local
from ptcg_lab.tr15_acceptance_reuse import envelope,scientific_sources,validate_local
import psutil
D=e.DEST

def latest_parent():
    phase=e.idle_state()['forward_phases']['TR15'];template=read_json(D/'native-template.json')
    cfg={'job_kind':'compact_audit','audit_variant':'tr15_latest_parent','hypothesis_id':'TR15_LATEST_PARENT',
        'budget_hypothesis_id':'TR15','engineering_only':True,'training_wall_seconds':1800,
        'previous_full_audit':phase['latest_full_state_audit'],'previous_catalog':phase['latest_archive_catalog'],
        'framework_v3_manifest':template['framework_v3_manifest'],
        'parent_model_format_sha256':read_json(D/'parent-format-reference.json')['model_format_sha256'],
        'primary_change':'Latest complete v41 model, optimizer, all RNG, cursor; reuse the existing full archive catalog.',
        'success_criteria':'Original full307037/41updates/pending0 unchanged; actual new-backend native gate remains pending.'}
    rid,proof=e.run_job(cfg,'tr15-v41-latest-parent',False,1800)
    state=e.idle_state();phase=state['forward_phases']['TR15']
    phase.update(latest_full_state_audit=e.reference('runs/'+rid+'/metrics.json'),
        latest_archive_catalog=e.reference('runs/'+rid+'/verified-archive-catalog.json'),baseline_update_digest=proof['baseline_update_digest'])
    atomic_json(R/'research/autonomy/state.json',state)
    return proof

def engineering(proof):
    cfg=e.make_training(0,proof)
    cfg.update(engineering_only=True,TR15_native_acceptance=True,training_wall_seconds=5400,reserved_future_bytes=2*2**30)
    cfg.pop('framework_v3_native_acceptance',None);cfg.pop('acceptance_reuse',None)
    recovery=e.idle_state()['forward_phases']['TR15'].get('engineering_completion_configuration')
    if recovery:
        if sha256(R/recovery['path'])!=recovery['sha256']:raise ValueError('Engineering completion configuration changed')
        replacement=read_json(R/recovery['path'])
        cfg.update(replacement)
    cfg['primary_change']='Disposable actual native/GPU/selfplay interrupted partial PPO recovery; science credit zero.'
    cfg['success_criteria']='Exact saved-state CUDA cold restore, bounded continuation, fixed-self BO3, full-update refresh, complete archive ACKs.'
    repair=e.idle_state()['forward_phases']['TR15'].get('framework_startup_repair',{}).get('sequence',0)
    label='tr15-native-v41-selfplay-difference'+(f'-repair-{repair:03d}' if repair else '')
    rid,metrics=e.run_job(cfg,label,False,5400)
    report_path=R/'runs'/rid/'remote-results/TR15-native-selfplay-acceptance.json'
    report=validate_report(read_json(report_path))
    if report['framework_v3_manifest_sha256']!=cfg['framework_v3_manifest']['sha256']:raise ValueError('Acceptance runtime changed')
    audit_cfg={'job_kind':'compact_audit','audit_variant':'tr15_full_archive','hypothesis_id':'TR15_ENGINEERING_ARCHIVE',
        'budget_hypothesis_id':'TR15','engineering_only':True,'training_wall_seconds':1800,'source_run_id':rid,
        'baseline_update_digest':proof['baseline_update_digest'],
        'primary_change':'Verify actual new-backend disposable archive full original hashes and both cold points; never adopt its model.',
        'success_criteria':'Complete raw/encoded SHA256, two cold semantic points, original41-update history unchanged.'}
    audit_cfg['complete_prior_terminal_archives']=[cfg['TR15_engineering_recovery']['source_run_id']] if cfg.get('TR15_engineering_recovery') else []
    aid,archive=e.run_job(audit_cfg,'tr15-native-v41-archive',False,1800)
    if not archive.get('local_and_server_full_recovery_verified'):raise ValueError('Actual native archive cold verification failed')
    manifest=read_json(R/'runs'/rid/'remote-training-manifest.json');bound=validate_manifest_local(R,cfg)
    refs={'native_GPU':e.reference(report_path),'first_real_CUDA_partial_PPO':e.reference(R/'runs'/rid/'remote-results/TR15-cursor-comparison.json'),
          'initial_full_archive':e.reference(R/'runs'/aid/'metrics.json'),
          'initial_two_cold_points':e.reference(R/'runs'/rid/'remote-results/framework-v2-cold-recovery.json')}
    doc={'at':utcnow(),'authority':e.AUTH,'complete':True,'recovery_index_schema':3,'latest_only':True,'two_recovery_points_retained':True,
         'scientific_sources':scientific_sources(R),'phase_runtime_sources':bound['runtime_sources'],
         'runtime_environment_sha256':report['runtime_environment_sha256'],'native_engine_files':{k:v['sha256'] for k,v in metrics['engine']['files'].items()},
         'model_format_sha256':report['model_format_sha256'],'training_config':manifest['training_config'],
         'source_bundle_inputs':{row['path']:row['sha256'] for row in manifest['file_map']},'accepted_evidence':refs,
         'framework_v3_manifest_sha256':cfg['framework_v3_manifest']['sha256'],'science_credit_from_engineering':0,
         'scientific_parent_remains_original_TR14_v41':True}
    path=D/'ACCEPTANCE_REUSE.json';atomic_json(path,doc);scope=envelope(doc,e.reference(path));validate_local(R,scope)
    lease=read_json(R/'runs'/rid/'remote-job.json')
    actual={'source':{'run_id':rid,'manifest_sha256':sha256(R/'runs'/rid/'remote-training-manifest.json'),
                    'identity':lease['identity'],'lease_epoch':lease['lease_epoch']},'sha256':sha256(report_path)}
    state=e.idle_state();phase=state['forward_phases']['TR15']
    if phase['latest_full_state_source']!='20261031T111127Z_tr14-volume-segment-001_e112d1':raise ValueError('Scientific parent changed during isolated engineering')
    phase.update(native_GPU_acceptance=e.reference(report_path),framework_v3_native_acceptance=actual,
        framework_v3_acceptance_pending=False,acceptance_reuse=scope,GPU_cursor_difference=refs['first_real_CUDA_partial_PPO'],
        engineering_disposable_source=rid,engineering_archive_evidence=e.reference(R/'runs'/aid/'metrics.json'))
    state['current_phase']='TR15_scientific_training';atomic_json(R/'research/autonomy/state.json',state)
    e.publish('native_GPU_selfplay_acceptance_passed',run_id=rid,scientific_parent_unique=307037,science_credit_from_engineering=0)

def main():
    state=e.idle_state();phase=state['forward_phases']['TR15']
    if phase['authority']!=e.AUTH or phase['training_run_ids'] or phase.get('native_GPU_acceptance'):
        raise ValueError('Fresh finite executor only; no blind restart of an existing phase')
    e.EXECUTOR_MANIFEST_SHA=sha256(D/'executor-source-manifest.json');e.verify_sources()
    lock=D/'executor.lock'
    with lock.open('x',encoding='utf-8') as stream:json.dump({'pid':os.getpid(),'created':psutil.Process().create_time(),'at':utcnow()},stream)
    began=time.monotonic()
    try:
        parent_ref=phase['latest_full_state_audit']
        candidate=read_json(R/parent_ref['path'])
        if candidate.get('scientific_parent_unique')==307037 and candidate.get('local_full_state_recovery_verified') and candidate.get('latest_only'):
            if sha256(R/parent_ref['path'])!=parent_ref['sha256'] or candidate['source_run_id']!=phase['latest_full_state_source']:
                raise ValueError('Reusable latest parent proof changed')
            proof=candidate
            e.publish('latest_parent_proof_reused',reference=parent_ref,no_scientific_input_changes=True)
        else:proof=latest_parent()
        engineering(proof)
        for segment in (1,2):
            if proof['actual_unique']>=e.TARGET:break
            cfg=e.make_training(segment,proof)
            cfg['reserved_future_bytes']=16*2**30
            cfg['engineering_only']=False;cfg['TR15_native_acceptance']=False
            rid,_=e.run_job(cfg,f'tr15-selfplay-segment-{segment:03d}',True,28800)
            _,proof=e.audit_source(rid,f'tr15-selfplay-segment-{segment:03d}-archive');e.publish_volume(proof)
        if proof['actual_unique']<e.TARGET:raise RuntimeError('Finite authorized training raw allowance ended; retain latest complete positive state')
        state=e.idle_state();phase=state['forward_phases']['TR15']
        phase.update(training_finished=True,training_target_reached=proof['actual_unique'],actual_target_overshoot=proof['actual_unique']-e.TARGET,no_further_training=True)
        state['current_phase']='TR15_fixed_endpoint_evaluation';atomic_json(R/'research/autonomy/state.json',state)
        spec=e.materialize_agent(proof);evaluations=e.evaluate_endpoint(spec)
        from ptcg_lab.tr15_analysis import finish
        finish(R,proof,evaluations)
        e.publish('complete',actual_additional_unique=proof['new_unique'],endpoint_evaluations=evaluations)
    except BaseException as exc:
        e.publish('needs_controller_attention',error=type(exc).__name__+': '+str(exc),no_blind_restart=True);raise
    finally:
        overhead=max(0,time.monotonic()-began-e.ATTEMPT_GROSS_SECONDS)
        state=read_json(R/'research/autonomy/state.json')
        charge={'at':utcnow(),'CPU_job_seconds':overhead,'method':'finite coordinator gross minus exact controlled attempt gross',
                'not_charged_twice':True,'controller_pid':os.getpid()}
        if not state.get('active_job') and not state.get('active_run_id'):
            state['budget_used']['cpu_job_wall_hours']+=overhead/3600
            atomic_json(D/'executor-CPU-accounting.json',dict(charge,charged=True))
            state['forward_phases']['TR15']['executor_CPU_accounting']=e.reference(D/'executor-CPU-accounting.json')
            atomic_json(R/'research/autonomy/state.json',state)
            continuation=read_json(R/'research/autonomy/latest-continuation.json')
            continuation.update(at=utcnow(),TR15=state['forward_phases']['TR15'],budget_snapshot=state['budget_used'],current_phase=state['current_phase'])
            atomic_json(R/'research/autonomy/latest-continuation.json',continuation)
        else:atomic_json(D/'executor-CPU-accounting-pending.json',dict(charge,charged=False))
        if lock.exists() and read_json(lock).get('pid')==os.getpid():lock.unlink()

if __name__=='__main__':main()
