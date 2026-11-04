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
from ptcg_lab.tr15_scope import AUTH,BASE,TARGET
from ptcg_lab.remote_lease import validate_no_unreconciled_remote

DEST=ROOT/'research/autonomy/activations/TR15-selfplay-100000-20261101'
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
    phase=state['forward_phases']['TR15']
    if phase.get('user_stop') or state.get('current_user_pause') or (ROOT/'research/autonomy/STOP').exists():
        raise RuntimeError('User stop/pause applies')
    return state


def register_config(config,label):
    phase=idle_state()['forward_phases']['TR15']
    if phase['authority']!=AUTH:raise ValueError('Authority changed')
    path=DEST/'configs'/(label+'.json')
    if path.exists():raise FileExistsError(path)
    worker_count=config.get('workers',1) if config.get('job_kind')=='evaluation' else 1
    config=dict(config,project_path=str(ROOT),forward_phase='TR15',TR15_authorization=AUTH,
                workers=worker_count,maximum_attempts=1)
    atomic_json(path,config)
    key=hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    registry=read_json(ROOT/'research/autonomy/TR15-approved-configurations.json')
    registry['configs'][key]={'path':path.relative_to(ROOT).as_posix(),'role':'engineering' if config['engineering_only'] else 'research','registered_at':utcnow()}
    atomic_json(ROOT/'research/autonomy/TR15-approved-configurations.json',registry)
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
             '--hypothesis','TR15 frozen independent +100000 valid unique decisions and endpoint evaluation']
    if research:command.append('--research')
    if resume_run_id:command+=['--resume',str(ROOT/'runs'/resume_run_id)]
    with log.open('xb') as stream:
        result=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
    state=read_json(ROOT/'research/autonomy/state.json');rid=resume_run_id or state['last_run_id'];run=ROOT/'runs'/rid
    if read_json(run/'config.json')!=read_json(path):raise ValueError('Runner result belongs to a different configuration')
    attempts=read_json(run/'attempts.json')
    ATTEMPT_GROSS_SECONDS+=sum(row['seconds'] for row in attempts)
    if result.returncode or attempts[-1]['status'] not in ('completed','checkpointed'):
        raise RuntimeError(f'{label} did not complete safely; retained run {rid}; see {log}')
    idle_state()  # Charge the controlled interval even when reconciliation fails.
    publish('registered_job_terminal',run_id=rid,status=attempts[-1]['status'])
    return rid,read_json(run/'metrics.json')


def audit_source(rid,label):
    state=idle_state();phase=state['forward_phases']['TR15']
    cfg={'job_kind':'compact_audit','audit_variant':'tr15_full_archive','hypothesis_id':'TR15_ARCHIVE_ACCEPTANCE',
        'budget_hypothesis_id':'TR15','engineering_only':True,'training_wall_seconds':1800,
        'source_run_id':rid,'baseline_update_digest':phase.get('baseline_update_digest'),
        'primary_change':'Verify received original bytes and both full recovery points; never train or reencode model.',
        'success_criteria':'Full raw/encoded SHA256, two server/local semantic recovery points, exact inherited history and unique first-epoch coverage.'}
    if phase.get('acceptance_reuse'):
        cfg['acceptance_reuse']=phase['acceptance_reuse']
        cfg['primary_change']='Reuse accepted implementation; check latest complete state and new original-byte archives only.'
        cfg['success_criteria']='Latest state/cursor/model/optimizer/RNG, full archive ACKs and unique coverage valid; retain two recovery points.'
    receipt_path=ROOT/'runs'/rid/'postflight-receipt.json'
    if receipt_path.exists():
        receipt=read_json(receipt_path)
        if receipt['run_id']!=rid or receipt['remote_terminal_sha256']!=sha256(ROOT/'runs'/rid/'remote-terminal.json'):
            raise ValueError('Inline complete archive proof belongs to another terminal state')
        proof_ref=receipt['proof'];catalog_ref=receipt['catalog']
        if any(sha256(ROOT/ref['path'])!=ref['sha256'] for ref in [proof_ref,catalog_ref]):raise ValueError('Inline archive evidence changed')
        proof=read_json(ROOT/proof_ref['path']);audit_id=rid
    else:
        audit_id,proof=run_job(cfg,label,False,1800)
        proof_ref=reference('runs/'+audit_id+'/metrics.json');catalog_ref=reference('runs/'+audit_id+'/verified-archive-catalog.json')
    if not proof.get('complete') or not proof.get('local_and_server_full_recovery_verified'):raise ValueError('Full archive audit failed')
    state=idle_state();phase=state['forward_phases']['TR15']
    phase.update(latest_full_state_source=rid,latest_full_state_audit=proof_ref,
        latest_archive_catalog=catalog_ref,
        baseline_update_digest=proof['baseline_update_digest'],actual_unique=proof['actual_unique'],
        actual_additional_unique=proof['new_unique'],latest_pending_decisions=proof['latest_pending_decisions'])
    phase.setdefault('accepted_sources',[]).append({'run_id':rid,'audit_run_id':audit_id,'proof':proof_ref,'catalog_reference':catalog_ref})
    state['updated_at']=utcnow();atomic_json(ROOT/'research/autonomy/state.json',state)
    if phase.get('continuation_complete_tail'):
        phase.pop('continuation_complete_tail');atomic_json(ROOT/'research/autonomy/state.json',state)
    return audit_id,proof


def make_training(segment,proof):
    state=idle_state();phase=state['forward_phases']['TR15'];rid=phase['latest_full_state_source'];source=ROOT/'runs'/rid
    template=phase.get('remote_configuration_template',{'path':'research/autonomy/activations/TR15-selfplay-100000-20261101/native-template.json'})
    if template.get('sha256') and sha256(ROOT/template['path'])!=template['sha256']:raise ValueError('TR15 resource template changed')
    config=read_json(ROOT/template['path'])
    lease=read_json(source/'remote-job.json');manifest=source/'remote-training-manifest.json'
    config.update(hypothesis_id='TR15',budget_hypothesis_id='TR15',engineering_only=False,parent_acceptance_only=False,
        no_training_updates=False,training_wall_seconds=28800,source_run_id=rid,
        source_v2={'index_sha256':proof['source_v2_index_sha256'],'point':proof['latest_point']},
        source_point=proof['latest_point'],source_state_audit=phase['latest_full_state_audit'],
        existing_restore_source={'run_id':rid,'manifest_sha256':sha256(manifest),'identity':lease['identity'],
                                 'lease_epoch':lease['lease_epoch'],'cursor':proof['latest_point']['cursor']},
        source_remote_evidence={n:sha256(source/'remote-results'/n) for n in ['recovery-v2.json','payload-config.json','league.json','framework-v2-cold-recovery.json'] if (source/'remote-results'/n).exists()},
        source_evidence={n:sha256(source/n) for n in ['metrics.json','attempts.json','remote-training-manifest.json','remote-job.json','remote-terminal.json'] if (source/n).exists()},
        source_inflight=None,complete_postcheckpoint_tail=None,GPU_cursor_difference_required=False,
        evaluation_protocol=reference('research/autonomy/TR15_EVALUATION_FROZEN.json'),
        primary_change='Latest frozen self refreshed only after full PPO commits; add100000 effective unique from v41 to407037; unchanged PPO/precision.',
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
        catalog_ref=accepted.get('catalog_reference')
        if catalog_ref and sha256(ROOT/catalog_ref['path'])!=catalog_ref['sha256']:raise ValueError('Accepted latest catalog changed')
        cat=read_json(ROOT/catalog_ref['path']) if catalog_ref else read_json(ROOT/'runs'/accepted['audit_run_id']/'verified-archive-catalog.json')
        for item in cat['objects']:existing[item['sha256']]=item
    point=proof['latest_point']
    required={row['sha256'] for row in [*point['references'],point['model']['object'],point['tail'],point['manifest']]}
    config['local_archive_sources']=[item for digest,item in existing.items() if digest in required]
    from ptcg_lab.tr15_framework import validate_manifest_local
    config['framework_v3_native_acceptance']=phase.get('framework_v3_native_acceptance')
    config['TR15_native_acceptance']=False
    config['inline_archive_postflight']=True
    config['baseline_update_digest']=phase['baseline_update_digest']
    validate_manifest_local(ROOT,config)
    return config


def publish_volume(proof):
    state=idle_state();phase=state['forward_phases']['TR15'];path=ROOT/'research/autonomy/CURRENT_VOLUME.json'
    old=read_json(path);backup=DEST/'before/research/autonomy/CURRENT_VOLUME.json'
    if not backup.exists():backup.parent.mkdir(parents=True,exist_ok=True);backup.write_bytes(path.read_bytes())
    baseline=read_json(backup)
    families=[f for f in old['families'] if f['hypothesis_id']!='TR15']
    family={'hypothesis_id':'TR15','canonical_training_family':phase['training_run_ids'][0],
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
        TR15_incremental_audit=phase['latest_full_state_audit'],pre_TR15_ledger_preserved=reference(backup),
        scope='Through TR15 latest fully archived committed endpoint; inherited history, repeated epochs and unoptimized pending excluded.')
    atomic_json(path,old)


def materialize_agent(proof):
    endpoint=proof['endpoint_model']
    if not endpoint:raise ValueError('No fixed endpoint model')
    source=ROOT/'agents/learned/tr14-volume-100000-endpoint';target=ROOT/'agents/learned/tr15-volume-100000-endpoint'
    if target.exists():raise FileExistsError(target)
    shutil.copytree(source,target,ignore=shutil.ignore_patterns('model.pt','spec.json','__pycache__','*.pyc'))
    shutil.copyfile(ROOT/endpoint['path'],target/'model.pt')
    if sha256(target/'model.pt')!=endpoint['sha256']:raise ValueError('Endpoint byte copy differs')
    spec=read_json(source/'spec.json');spec.update(id='tr15-volume-100000-endpoint',path=target.relative_to(ROOT).as_posix(),
        model_sha256=endpoint['sha256'],training_run_id=idle_state()['forward_phases']['TR15']['latest_full_state_source'],
        training_config_sha256=sha256(ROOT/'research/autonomy/TR15-training-frozen.json'))
    atomic_json(target/'spec.json',spec);return spec


def evaluate_endpoint(spec):
    frozen=read_json(ROOT/'research/autonomy/TR15_EVALUATION_FROZEN.json');results={}
    common={'job_kind':'evaluation','hypothesis_id':'TR15','budget_hypothesis_id':'TR15','engineering_only':False,
        'save_all_replays':True,'stop_on_internal_fallback':True,'policy_seed':frozen['policy_seed'],
        'bootstrap_samples':20000,'bootstrap_seed':frozen['bootstrap_seed'],'no_posthoc_extension':True,
        'native_environment_seed_not_exposed':True,'endpoint_selection':'last fixed-volume committed endpoint',
        'evaluation_protocol':reference('research/autonomy/TR15_EVALUATION_FROZEN.json')}
    primary=copy.deepcopy(frozen['endpoint_primary_cell']);primary['candidate']=spec
    for label,count,cells in [('reliability',50,[dict(primary,cell_id='reliability-vs-ppo',series_count=50)]),
                              ('primary',1600,[primary]),
                              ('development',800,[dict(c,candidate=spec) for c in frozen['endpoint_development_cells']])]:
        available=psutil.virtual_memory().available
        workers=min(8,os.cpu_count() or 1,max(1,int(available//(2*2**30))))
        cfg=dict(common,series_count=count,matchups=cells,evaluation_stage='TR15_'+label,workers=workers)
        rid,metrics=run_job(cfg,'tr15-'+label,True,21600)
        if not metrics.get('complete') or metrics.get('series')!=count:
            # Exact durable row count is authoritative for the evaluator schema.
            from ptcg_lab.io import read_series
            if not metrics.get('complete') or len(read_series(ROOT/'runs'/rid/'series.jsonl'))!=count:
                raise ValueError('Frozen endpoint evaluation incomplete')
        results[label]=rid
    return results

