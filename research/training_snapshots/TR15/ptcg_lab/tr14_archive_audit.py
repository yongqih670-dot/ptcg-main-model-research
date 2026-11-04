"""Controlled local full-byte archive, semantic recovery and endpoint audit."""
import hashlib
import io
import json
import os
from pathlib import Path
import torch
from .io import read_json,atomic_json,sha256,utcnow
from .archive_compressed import verify_archive
from .archive_ack import validate_ack
from .remote_lease import validate_no_unreconciled_remote
from .fo_v2.durable import canonical
from .framework_v2_training import semantic_digest,pending_digests
from .ppo_accounting import checkpoint_decisions,update_decisions


def run(config,output):
    output=Path(output);root=Path(config['project_path']);source=root/'runs'/config['source_run_id']
    if os.environ.get('PTCG_JOB_RUN_ID')!=output.name:raise ValueError('Controlled local archive audit required')
    if config.get('complete_missing_terminal_archive'):
        from .tr14_terminal_archive import complete
        complete(root,config,output)
    lease=read_json(source/'remote-job.json');validate_no_unreconciled_remote(root,{'remote_job':lease})
    failure_recovery=config.get('failure_recovery')
    metrics=read_json(source/'metrics.json') if (source/'metrics.json').exists() else {}
    if not metrics and not failure_recovery:raise ValueError('Missing terminal scientific metrics')
    index=read_json(source/'remote-results/recovery-v2.json')
    source_config=read_json(source/'config.json')
    if source_config.get('shared_capacity') and source_config.get('parent_acceptance_only'):
        quota=read_json(source/'remote-results/shared-capacity-CPU-acceptance.json')
        if not quota.get('complete') or quota.get('observed_CPU_quotas')!=[2,1,4] or not quota.get('only_exact_owned_service_modified'):
            raise ValueError('Measured owned CPU4/2/1 acceptance is incomplete')
    cold=read_json(source/'remote-results/framework-v2-cold-recovery.json') if (source/'remote-results/framework-v2-cold-recovery.json').exists() else {}
    reuse=config.get('acceptance_reuse')
    if reuse:
        from .tr14_acceptance_reuse import validate_local as validate_reuse,validate_targeted_difference
        if failure_recovery:
            validate_targeted_difference(root,reuse,failure_recovery)
            registered=read_json(root/failure_recovery['path'])
            if registered['source_run_id']!=source.name or registered['recovery_index_sha256']!=sha256(source/'remote-results/recovery-v2.json'):
                raise ValueError('Failed positive-update source changed')
        else:validate_reuse(root,reuse)
    checked=cold.get('checked_points',cold.get('two_points',[]))
    if (not cold.get('complete') and not failure_recovery) or len(index['points'])!=2:
        raise ValueError('Reliable recovery points and fresh server proof required')
    if not reuse and len(checked)!=2:raise ValueError('Initial two-point recovery acceptance required')
    if reuse and not failure_recovery and not any(row['generation']==index['points'][-1]['generation'] for row in checked):
        raise ValueError('Latest point cold proof missing')
    cadence=[json.loads(line) for line in (source/'remote-results/framework-v2-cadence.jsonl').read_text(encoding='utf-8').splitlines() if line]
    gaps=[row['reliable_commit_gap_seconds'] for row in cadence if row.get('reliable_commit_gap_seconds') is not None]
    if not gaps or any(gap>800 for gap in gaps):raise ValueError('Native reliable commit gap was not within800 seconds')
    if (index['run_id'],index['epoch'])!=(source.name,lease['lease_epoch']):raise ValueError('Recovery owner differs')
    archive=source/'online-archive';catalog={};by_object={};count=0
    for path in sorted((archive/source.name/'acks').glob('*.json')):
        ack=read_json(path)
        validate_ack(ack,source.name,lease['lease_epoch'],ack['object'],ack['sha256'],ack['bytes'],ack['sequence'])
        if ack['object'] in by_object:raise ValueError('Duplicate archive object')
        by_object[ack['object']]=ack;count+=1
        if ack['sha256'] not in catalog:
            verify_archive(archive,ack)
            catalog[ack['sha256']]={'storage':'gzip_ack','sha256':ack['sha256'],
                'archive_root':archive.relative_to(root).as_posix(),'ack':ack}
        elif catalog[ack['sha256']]['ack']['bytes']!=ack['bytes']:raise ValueError('Conflicting identical raw SHA')
    if not count:raise ValueError('No original-byte archive ACKs')
    cache={}
    def payload(ref):
        ack=by_object.get(ref['path'])
        if not ack or (ack['sha256'],ack['bytes'])!=(ref['sha256'],ref['bytes']):
            raise ValueError('Recovery graph omitted a complete original-byte archive ACK: '+ref['path'])
        if ref['sha256'] not in cache:cache[ref['sha256']]=verify_archive(archive,ack,collect=True)
        return cache[ref['sha256']]
    recovered=[];last=None;previous=None
    points=index['points'][-1:] if reuse else index['points']
    if len(index['points'])!=2 or index['points'][1]['generation']!=index['points'][0]['generation']+1:
        raise ValueError('Two consecutive reliable points must remain retained')
    for point in points:
        body={k:v for k,v in point.items() if k not in ('commit_id','manifest')}
        if hashlib.sha256(canonical(body)).hexdigest()!=point['commit_id']:raise ValueError('Recovery point digest differs')
        if previous and point['generation']!=previous['generation']+1:raise ValueError('Recovery generations differ')
        previous=point
        descriptor=json.loads(payload(point['manifest']))
        if descriptor!={'schema':3,'run_id':source.name,'epoch':lease['lease_epoch'],
                        'point':{k:v for k,v in point.items() if k!='manifest'}}:raise ValueError('Immutable descriptor differs')
        model=torch.load(io.BytesIO(payload(point['model']['object'])),map_location='cpu',weights_only=False)
        tail=torch.load(io.BytesIO(payload(point['tail'])),map_location='cpu',weights_only=False)
        value={**model,**tail};mapping=value.pop('v2_reference_map');expected=value.pop('state_semantic_sha256')
        if reuse:
            from .tr14_acceptance_reuse import check_state
            check_state(value,reuse)
        if point['model'].get('state_revision')!=value.get('model_state_revision'):raise ValueError('Partial model/tail revision differs')
        if semantic_digest(value)!=expected:raise ValueError('Local full state semantic digest differs')
        if any(value[k]!=point['cursor'][k] for k in point['cursor']):raise ValueError('Local cursor differs')
        for ref in point['references']:
            ack=by_object.get(ref['path'])
            if not ack or (ack['sha256'],ack['bytes'])!=(ref['sha256'],ref['bytes']):raise ValueError('Missing full archive reference')
        required=pending_digests(value['archive_state'])
        required.update({r['path']:r['sha256'] for r in value['league']['snapshots']})
        if not set(required)<=set(mapping):raise ValueError('Full pending or league reference map omitted')
        for name,digest in required.items():
            if mapping[name]['sha256']!=digest:raise ValueError('Canonical reference binding differs')
        pending=0
        for row in value['archive_state']['live_expected']:
            if row['policy_version']!=value['version']:continue
            ref=mapping[f"series/{row['series_id']:06d}.json"]
            summary=json.loads(payload(ref))
            if summary['statuses']!=['DONE','DONE'] or summary['trajectory_sha256']!=row['trajectory_sha256']:
                raise ValueError('Incomplete or changed pending BO3')
            pending+=summary['learner_decisions']
        actual=checkpoint_decisions(value)
        if config.get('baseline_update_digest') and semantic_digest(value['updates'][:28])!=config['baseline_update_digest']:
            raise ValueError('Historical28 PPO updates changed')
        if len(value['model'])!=127 or len(value['optimizer']['state'])!=97 or set(value['rng'])!={'python','numpy','torch','cuda'} or 'cpu_sampler_rng' not in value:
            raise ValueError('Incomplete inherited model/optimizer/RNG')
        if not failure_recovery and not any(p['generation']==point['generation'] and p['full_CPU_and_GPU_values_RNG_exact'] for p in checked):
            raise ValueError('Cold server proof belongs to a different generation')
        recovered.append({'generation':point['generation'],'cursor':point['cursor'],
            'actual_unique':actual,'pending_decisions':pending,'partial_PPO':value.get('partial_ppo') is not None,
            'semantic_sha256':expected})
        last=value
    if last is None:raise ValueError('No complete recovery state')
    new=last['updates'][28:];ids=[sid for r in new for sid in r['series_ids']]
    if len(ids)!=len(set(ids)):raise ValueError('A complete BO3 was optimized twice across TR14 updates')
    actual=checkpoint_decisions(last)
    if actual!=206957+sum(update_decisions(row) for row in new):raise ValueError('Incremental decision accounting differs')
    inherited=sum(r.get('inherited_pending_first_epoch_coverage',0) for r in new)
    sampled=sum(r.get('new_sampling_first_epoch_coverage',0) for r in new)
    if new and (not 0<=inherited<=0 or inherited+sampled!=actual-206957):raise ValueError('Inherited/new coverage split differs')
    endpoint=None
    if metrics.get('complete') and not metrics.get('engineering_only'):
        if actual<306957 or last.get('partial_ppo') is not None:raise ValueError('Fixed endpoint has not completed')
        ack=by_object.get('model.pt')
        if not ack or ack['sha256']!=metrics['model_sha256']:raise ValueError('Endpoint model original ACK differs')
        raw=verify_archive(archive,ack,collect=True)
        weights=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
        if semantic_digest(weights)!=semantic_digest(last['model']):raise ValueError('Endpoint weights differ from committed state')
        target=source/'model.pt'
        if target.exists():
            if sha256(target)!=ack['sha256']:raise ValueError('Never overwrite a different endpoint')
        else:
            with target.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
        endpoint={'path':target.relative_to(root).as_posix(),'sha256':ack['sha256'],'bytes':len(raw)}
    atomic_json(output/'verified-archive-catalog.json',{'source_run_id':source.name,'objects':list(catalog.values()),
        'all_full_raw_and_encoded_SHA256_verified':True})
    atomic_json(output/'metrics.json',{'at':utcnow(),'complete':True,'engineering_only':True,
        'source_run_id':source.name,'checked_points':recovered,'latest_only':bool(reuse),
        'two_recovery_points_retained':True,'reused_acceptance':reuse['reference'] if reuse else None,
        'archive_ACKs':count,'unique_raw_objects':len(catalog),
        'local_and_server_full_recovery_verified':not bool(failure_recovery),
        'local_full_state_recovery_verified':True,'server_fresh_GPU_recovery_pending':bool(failure_recovery),
        'all_full_original_hashes_verified':True,
        'actual_unique':actual,'new_unique':actual-206957,'new_PPO_updates':len(new),
        'baseline_update_digest':semantic_digest(last['updates'][:28]),
        'inherited_pending_first_epoch_coverage':inherited,'new_sampling_first_epoch_coverage':sampled,
        'latest_pending_decisions':recovered[-1]['pending_decisions'],'source_v2_index_sha256':sha256(source/'remote-results/recovery-v2.json'),
        'latest_point':index['points'][-1],'endpoint_model':endpoint,'optimizer_steps_executed':0,
        'maximum_reliable_commit_gap_seconds':max(gaps),'within_800_seconds':True,'within_600_seconds':max(gaps)<=600,
        'all_data_retained_no_GitHub_upload':True,'promotion_eligible':False})
