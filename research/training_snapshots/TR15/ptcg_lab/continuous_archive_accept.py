"""CPU-only live ACK acceptance using unchanged, fully verified parent bytes."""
import os
from pathlib import Path
import time
from .io import read_json,atomic_json,sha256,utcnow

PREPARED='20261027T124022Z_continuous-server-parent-acceptance_073bbe'
NUMERIC='20261027T132356Z_authorized-parent-numeric-600s_b612a4'
TASK='CONTINUOUS_PARENT_ARCHIVE_REUSE_ACCEPTANCE'


def validate_local(root,config):
    root=Path(root).resolve()
    if (config.get('hypothesis_id'),config.get('job_kind'),config.get('check_variant'),config.get('workers'),
            config.get('maximum_attempts'),config.get('engineering_only'),config.get('no_training_updates'),
            config.get('gpu_index'),config.get('gpu_uuid'))!=(TASK,'remote_training_check','continuous_archive_reuse',1,1,True,True,None,None):
        raise ValueError('Only one CPU zero-update archive acceptance is admitted')
    if not 180<=config['training_wall_seconds']<=600:raise ValueError('Measured bounded archive window required')
    ref=config['verified_archive_inventory'];path=(root/ref['path']).resolve()
    if root not in path.parents or sha256(path)!=ref['sha256']:raise ValueError('Full local hash inventory changed')
    inventory=read_json(path)
    if not inventory['complete'] or inventory['missing'] or len(inventory['verified'])!=8:
        raise ValueError('All8 original parent objects must have verified archive sources')
    if config['local_archive_sources']!=inventory['local_archive_sources']:raise ValueError('Reuse cannot silently omit verified originals')
    if config['expected_parent_objects']!=inventory['required_objects']:raise ValueError('Frozen parent object set changed')
    approval=config['archive_reuse_authorization'];ap=(root/approval['path']).resolve()
    if root not in ap.parents or sha256(ap)!=approval['sha256'] or not read_json(ap)['archive_reuse_and_measured_acceptance_authorized']:
        raise ValueError('New explicit archive-reuse authorization required')


def run(ticket_path):
    from .server_workspace import owned_root,inside
    from .retained_source import resolve,checkpoint_reference
    from .archive_outbox import ArchiveOutbox
    ticket=read_json(ticket_path);root=owned_root(ticket['project_root'],ticket['project_id']);job=inside(root,ticket['job_directory'])
    if (os.environ.get('PTCG_JOB_RUN_ID'),os.environ.get('CUDA_VISIBLE_DEVICES'))!=(job.name,''):
        raise ValueError('CPU-only identity required')
    manifest=read_json(job/'training-manifest.raw.json');source,original=resolve(root,ticket['project_id'],manifest['existing_restore_source'])
    numeric,nm=resolve(root,ticket['project_id'],manifest['prior_numeric_source'])
    if source.name!=PREPARED or numeric.name!=NUMERIC or not original.get('parent_acceptance_only') or not nm.get('prepared_parent_verify'):
        raise ValueError('Exact previously verified unchanged parent only')
    if nm['existing_restore_source']!=manifest['existing_restore_source']:raise ValueError('Numeric proof used a different prepared parent')
    point=checkpoint_reference(source,manifest['existing_restore_source'])
    if point['cursor']!={'version':5,'trained_decisions':35422,'next_series_id':222} or point['pending']:raise ValueError('Parent state changed')
    proof_path=inside(root,numeric/'derived-gpu-restore.json')
    if sha256(proof_path)!=manifest['prior_numeric_restore_sha256']:raise ValueError('Prior fresh GPU proof changed')
    proof=read_json(proof_path)
    if not proof['complete'] or not proof['same_GPU_fresh_process_rng_and_outputs_exact'] or proof['optimizer_steps_executed']!=0:
        raise ValueError('Prior numerical restoration did not pass')
    for name,digest in manifest['numeric_source_hashes'].items():
        if sha256(inside(root,numeric/'control'/name))!=digest or sha256(inside(root,job/'control'/name))!=digest:
            raise ValueError('Numerical source changed; prior proof cannot certify it')
    rows=manifest['expected_parent_objects'];box=ArchiveOutbox(job,job.name,ticket['lease_epoch']);started=time.monotonic()
    for row in rows:
        src=inside(root,source/row['object']);dst=inside(root,job/row['object'])
        if src.stat().st_size!=row['bytes'] or sha256(src)!=row['sha256']:raise ValueError('Remote parent bytes differ from verified local originals')
        dst.parent.mkdir(parents=True,exist_ok=True);os.link(src,dst);box.queue(row['object'],row['sha256'])
    deadline=float(os.environ['PTCG_JOB_DEADLINE_UNIX'])
    while box.pending()['pending_objects']:
        if time.time()>deadline-45:raise TimeoutError('Archive reuse ACK deadline')
        atomic_json(job/'progress.json',{'at':utcnow(),'stage':'waiting_for_reused_archive_ACKs',**box.pending(1)})
        time.sleep(.2)
    atomic_json(job/'archive-reuse-audit.json',{'at':utcnow(),'complete':True,'engineering_only':True,
        'parent_acceptance_passed':True,'full_parent_archive_ACKs_verified':True,'acknowledged_objects':len(rows),
        'acknowledged_original_bytes':sum(r['bytes'] for r in rows),'archive_acceptance_seconds':time.monotonic()-started,
        'prior_GPU_numerical_restore_reused':True,'prior_GPU_numerical_restore':proof,
        'optimizer_steps_executed':0,'new_unique_PPO_trained_decisions':0,'new_games':0,
        'old_failed_jobs_still_failed':True,'source_state_unchanged':True})


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--ticket',required=True);run(p.parse_args().ticket)
