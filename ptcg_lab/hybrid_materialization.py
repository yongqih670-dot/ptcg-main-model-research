"""CPU-accounted reconstruction of a completed hybrid candidate, without Torch."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import psutil
from .archive_ack import safe_target
from .archive_compressed import verify_archive
from .io import atomic_json, read_json, sha256, utcnow
from .remote_lease import validate_no_unreconciled_remote


def owned_runner_process(owner, run_id):
    """Windows venv redirects to a child; retain exact launcher identity."""
    if owner.get('run_id') != run_id:
        return False
    try:
        anchor = psutil.Process(owner['pid'])
        if abs(anchor.create_time() - owner['created']) >= .01:
            return False
        current = psutil.Process()
        return any(p.pid == anchor.pid for p in [current, *current.parents()])
    except (KeyError, psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def reconstruct(source, output, metrics, lease, deadline, *, free_reserve, growth_allowance):
    source, output = Path(source), Path(output)
    def check():
        if time.time() >= deadline-10:
            raise TimeoutError('Controlled model reconstruction deadline')
    check()
    archive = source/'online-archive'
    paths = sorted((archive/source.name/'acks').glob('*.json'))
    if not paths or len(paths)>4096:
        raise ValueError('Missing or excessive archive receipts')
    rows = [read_json(p) for p in paths]
    if any(r['run_id']!=source.name or r['lease_epoch']!=lease['lease_epoch'] for r in rows):
        raise ValueError('Archive receipt belongs to a different training job')
    if sum(r['bytes'] for r in rows)>4*2**30:
        raise ValueError('Reconstruction exceeds registered four GiB verification bound')
    models = [r for r in rows if r['object']=='model.pt']
    if len(models)!=1 or models[0]['sha256']!=metrics['model_sha256']:
        raise ValueError('Completed model and archive receipt differ')
    ack=models[0]
    required=ack['bytes']+16*2**20
    if required>growth_allowance or shutil.disk_usage(output).free-required<free_reserve:
        raise OSError('Model reconstruction would consume protected local storage')
    series = []
    for row in rows:
        check()
        is_series = row['object'].startswith('series/') and row['object'].endswith('.json')
        data = verify_archive(archive, row, collect=is_series)
        if is_series:
            item = json.loads(data)
            if (item.get('event') != 'series_end' or item.get('statuses') != ['DONE', 'DONE']
                    or type(item.get('rounds')) is not int or item['rounds'] not in (2, 3)
                    or type(item.get('learner_decisions')) is not int or item['learner_decisions'] < 0
                    or row['object'] != f"series/{item['series_id']:06d}.json"):
                raise ValueError('Archived BO3 summary is incomplete or inconsistent')
            series.append({'series_id': item['series_id'], 'complete_games': item['rounds'],
                           'sampled_learner_decisions': item['learner_decisions'],
                           'sha256': row['sha256']})
        check()
    if len({s['series_id'] for s in series}) != len(series):
        raise ValueError('Duplicate archived series identity')
    data=verify_archive(archive,ack,collect=True);check()
    if hashlib.sha256(data).hexdigest()!=metrics['model_sha256']:
        raise ValueError('Raw model reconstruction differs')
    target=safe_target(source,'model.pt')
    if target.exists():
        if sha256(target)!=ack['sha256']:
            raise ValueError('Existing candidate conflicts; never overwrite')
    else:
        temporary=output/'verified-model.partial'
        with temporary.open('xb') as stream:
            stream.write(data);stream.flush();os.fsync(stream.fileno())
        check()
        os.link(temporary,target);temporary.unlink()
    proof={'at':utcnow(),'model_archive_ack':ack,'raw_model_sha256':sha256(target),
           'all_archive_receipts_reconstructed_and_verified':len(rows),
           'no_model_training_or_weight_reencoding':True,'promotion_eligible':False,
           'controlled_materialization_run_id':output.name}
    if series:
        proof['archived_series_accounting'] = {
            'complete_BO3': len(series), 'complete_games': sum(s['complete_games'] for s in series),
            'sampled_learner_decisions': sum(s['sampled_learner_decisions'] for s in series),
            'series': sorted(series, key=lambda s:s['series_id']),
            'scope': 'Only summaries archived by this run, including inherited pending BO3; reconcile original prefix separately. Never count files remaining after retirement as total BO3.'}
    destination=safe_target(source,'hybrid-materialization.json')
    if destination.exists():
        previous=read_json(destination)
        if (previous.get('model_archive_ack')!=ack or previous.get('raw_model_sha256')!=proof['raw_model_sha256']
                or previous.get('all_archive_receipts_reconstructed_and_verified')!=len(rows)):
            raise ValueError('Existing materialization proof conflicts')
    else:
        atomic_json(destination,proof)
    return proof


def run(config,output):
    root=Path(config['project_path']).resolve();output=Path(output).resolve()
    if os.environ.get('PTCG_JOB_RUN_ID')!=output.name or output.parent!=root/'runs':
        raise RuntimeError('Controlled local runner required')
    deadline=float(os.environ['PTCG_JOB_DEADLINE_UNIX'])
    if config.get('original_materialization_deadline_unix'):
        deadline=min(deadline,float(config['original_materialization_deadline_unix']))
    if deadline-time.time()>600.5 or config['maximum_attempts']!=1:
        raise ValueError('Materialization is one bounded CPU engineering job')
    state=read_json(root/'research/autonomy/state.json')
    # The runner publishes active_run_id before spawning, then publishes the
    # assigned child identity. Allow only that short publication window.
    publication_deadline=min(deadline-10,time.time()+1)
    while (state.get('active_run_id')==output.name and not state.get('active_job')
           and time.time()<publication_deadline):
        time.sleep(.05);state=read_json(root/'research/autonomy/state.json')
    owner=state.get('active_job') or {}
    if state.get('active_run_id')!=output.name or not owned_runner_process(owner,output.name):
        raise RuntimeError('Another controller owns the active job')
    name=config['training_run_id']
    if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name):
        raise ValueError('Unsafe source training identity')
    source=safe_target(root,'runs/'+name)
    if source==output or source.resolve()!=source:
        raise ValueError('Training source escaped or equals engineering output')
    for filename in ['metrics.json','config.json','remote-job.json']:
        if sha256(source/filename)!=config['source_evidence_sha256'][filename]:
            raise ValueError('Completed training evidence changed: '+filename)
    metrics=read_json(source/'metrics.json');training=read_json(source/'config.json')
    if (metrics.get('complete') is not True or training.get('job_kind')!='remote_train_ppo' or training.get('check_variant') not in
            ('scientific_continue','fresh_scientific','derived_scientific','continuous_mainline') or
            not (metrics.get('scientific_continuation') or metrics.get('scientific_training'))):
        raise ValueError('Only completed scientific hybrid training can be reconstructed')
    if training.get('check_variant')=='continuous_mainline' and training.get('hypothesis_id')!='TR11':
        raise ValueError('Continuous materialization is limited to the authorized TR11 diagnostic')
    lease=read_json(source/'remote-job.json')
    if lease.get('run_id')!=source.name:
        raise ValueError('Terminal lease belongs to a different training run')
    validate_no_unreconciled_remote(root,{'remote_job':lease})
    policy=read_json(output/'policy.json')['budget']
    allowance=(policy['new_workspace_artifacts_gib_max']-state['budget_used']['new_workspace_artifacts_gib'])*2**30
    proof=reconstruct(source,output,metrics,lease,deadline,
        free_reserve=policy['disk_free_gib_min']*2**30,growth_allowance=allowance)
    atomic_json(output/'metrics.json',{'at':utcnow(),'complete':True,'engineering_only':True,
        'training_run_id':name,'raw_model_sha256':proof['raw_model_sha256'],
        'verified_archive_receipts':proof['all_archive_receipts_reconstructed_and_verified'],
        'materialization_proof_path':(source/'hybrid-materialization.json').relative_to(root).as_posix(),
        'materialization_proof_sha256':sha256(source/'hybrid-materialization.json'),
        'training_evidence_not_rewritten':True,'promotion_eligible':False})
