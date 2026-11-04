"""Read-only numerical inspection of the exact completed preparation objects.

The earlier 300s preparation job remains failed. This separate bounded stage
does not rerun preparation, resume its service, sample games, or optimize.
"""
import argparse
import os
from pathlib import Path
from types import SimpleNamespace

from .io import read_json, atomic_json, sha256, utcnow

PREPARED='20261027T124022Z_continuous-server-parent-acceptance_073bbe'


def materialize_catalog(root, job, source, original):
    """Expose the exact frozen public card catalogue to this new control tree."""
    from .server_workspace import inside
    relative='ptcg_lab/public_cg/metadata.json'
    rows=[r for r in original['file_map'] if r['path']==relative]
    if len(rows)!=1:raise ValueError('One pinned public card catalogue is required')
    row=rows[0];src=inside(root,source/'control'/relative);dst=inside(root,job/'control'/relative)
    if sha256(src)!=row['sha256'] or src.stat().st_size!=row['bytes']:
        raise ValueError('Public card catalogue changed')
    dst.parent.mkdir(parents=True,exist_ok=True)
    if not dst.exists():os.link(src,dst)
    if sha256(dst)!=row['sha256']:raise ValueError('Linked public catalogue differs')
    return dst


def validate_local(root, config):
    root=Path(root)
    authorized=config.get('hypothesis_id')=='CONTINUOUS_PREPARED_PARENT_NUMERIC_VERIFY_AUTHORIZED_WINDOW'
    identity='CONTINUOUS_PREPARED_PARENT_NUMERIC_VERIFY_AUTHORIZED_WINDOW' if authorized else 'CONTINUOUS_PREPARED_PARENT_NUMERIC_VERIFY'
    seconds=600 if authorized else 300
    if (config.get('hypothesis_id'),config.get('engineering_only'),config.get('no_training_updates'),
            config.get('maximum_attempts'),config.get('training_wall_seconds')) != (identity,True,True,1,seconds):
        raise ValueError('One distinct zero-update prepared-object inspection only')
    if authorized:
        ref=config.get('engineering_window_authorization',{});path=(root/ref.get('path','')).resolve()
        if root.resolve() not in path.parents or sha256(path)!=ref.get('sha256'):
            raise ValueError('Explicit new engineering-window authorization is required')
        approval=read_json(path)
        request=root/'research/proposals/continuous_mainline_v1/ENGINEERING_WINDOW_REQUEST.json'
        if (approval.get('user_text')!='允许' or approval.get('request_sha256')!=sha256(request)
                or approval.get('GPU_job_seconds_max')!=600 or approval.get('maximum_new_engineering_launches')!=1
                or approval.get('research_raw_increment')!=0 or approval.get('optimizer_steps')!=0):
            raise ValueError('New window exceeds explicit human authorization')
    if config['existing_restore_source']['run_id']!=PREPARED:
        raise ValueError('Only the exact already prepared terminal capsule is admitted')
    run=root/'runs'/PREPARED
    from .remote_lease import validate_no_unreconciled_remote
    validate_no_unreconciled_remote(root,{'remote_job':read_json(run/'remote-job.json')})
    attempts=read_json(run/'attempts.json')
    if len(attempts)!=1 or attempts[0]['status']!='failed' or not 0<attempts[0]['seconds']<=300:
        raise ValueError('Original failed preparation expenditure must remain recorded')
    for relative,digest in config['prepared_evidence'].items():
        if sha256(run/relative)!=digest:raise ValueError('Prepared source proof changed')


def run(ticket_path, fresh=False):
    import torch
    from .server_workspace import owned_root,inside
    from .retained_source import resolve,checkpoint_reference
    from .training_control import TrainingControl
    from .rl_core import load_full_actor,optimizer_for,restore_rng
    from .tensor_io import load_torch
    from .scientific_resume import audit_runtime
    from .continuous_mainline import preserve_milestones
    from .archive_outbox import ArchiveOutbox
    ticket=read_json(ticket_path);root=owned_root(ticket['project_root'],ticket['project_id'])
    job=inside(root,ticket['job_directory']);manifest=read_json(job/'training-manifest.raw.json')
    if os.environ.get('PTCG_JOB_RUN_ID')!=job.name or os.environ.get('CUDA_VISIBLE_DEVICES')!=ticket['gpu_uuid']:
        raise ValueError('Exact inspection identity/GPU required')
    source,original=resolve(root,ticket['project_id'],manifest['existing_restore_source'])
    if source.name!=PREPARED or not original.get('parent_acceptance_only'):
        raise ValueError('This is only an inspection of the already prepared zero-update state')
    materialize_catalog(root,job,source,original)
    point=checkpoint_reference(source,manifest['existing_restore_source'])
    if point['cursor']!={'version':5,'trained_decisions':35422,'next_series_id':222} or point['pending']:
        raise ValueError('Prepared state must still be the complete untrained parent')
    config_path=inside(root,source/'payload-config.json')
    if sha256(config_path)!=manifest['prepared_payload_config_sha256']:
        raise ValueError('Prepared configuration changed')
    config=read_json(config_path)
    checkpoint=inside(root,source/point['checkpoint']['path'])
    if sha256(checkpoint)!=point['checkpoint']['sha256']:raise ValueError('Prepared checkpoint changed')
    saved=load_torch(checkpoint,map_location='cpu',weights_only=False)
    control=TrainingControl(job,job.name,ticket['lease_epoch'],float(os.environ['PTCG_JOB_DEADLINE_UNIX']),
        lambda stage,**kw:atomic_json(job/'progress.json',{'at':utcnow(),'stage':stage,**kw}))
    control.wait(45);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    model,features,_=load_full_actor(source/'control'/config['bundle_path'],config['parent_model_sha256'],
        'cuda',device_guard=lambda:control.wait(45))
    optimizer=optimizer_for(model,config['ppo']['actor_learning_rate'],config['ppo']['value_head_learning_rate'])
    model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer']);restore_rng(saved['rng'])
    audit_runtime(job,model,optimizer,saved,[],features,ticket['gpu_uuid'],reference=not fresh,
        fixture_reference=config['CPU_backend_fixture'],cpu_rollout_values=True)
    if not fresh:
        league=read_json(source/'league.json')
        for ref in [point['checkpoint'],point['prefix'],*league['snapshots']]:
            src=inside(root,source/ref['path']);dst=inside(root,job/ref['path'])
            if sha256(src)!=ref['sha256']:raise ValueError('Prepared immutable parent object changed')
            dst.parent.mkdir(parents=True,exist_ok=True);os.link(src,dst)
        box=ArchiveOutbox(job,job.name,ticket['lease_epoch'])
        league_object=SimpleNamespace(entries=league['snapshots'],stats=league['fixed_training_results'])
        preserve_milestones(job,{**config,'decision_milestones':[32782]},32782,point,league_object,box)
    atomic_json(job/('prepared-parent-fresh.json' if fresh else 'prepared-parent-audit.json'),
        {'at':utcnow(),'complete':True,'fresh_process':fresh,'parent_run_id':source.name,
         'original_preparation_still_failed':True,'source_state_unchanged':True,
         'optimizer_steps_executed':0,'new_unique_PPO_trained_decisions':0})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--ticket',required=True);parser.add_argument('--fresh',action='store_true')
    args=parser.parse_args();run(args.ticket,args.fresh)
