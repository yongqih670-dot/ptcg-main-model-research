"""Read-only original-TR04 CUDA restore, inference and fresh-process RNG proof."""
import argparse
import hashlib
import os
from pathlib import Path
import random
import numpy as np
import torch
from .io import read_json,atomic_json,sha256,utcnow
from .server_workspace import owned_root,inside
from .retained_source import resolve,checkpoint_reference
from .tensor_io import load_torch,atomic_torch_save
from .rl_core import load_full_actor,optimizer_for,evaluate_decisions,restore_rng


def fixture_rows(capsule):
    rows=[row for row in capsule['components'] if row['path'].startswith('rollouts/')]
    if len(rows)!=35 or capsule['expected_update_input']['decisions']!=5430:raise ValueError('Original TR04 pending input scope differs')
    rows=sorted(rows,key=lambda row:row['path'])
    return rows[0],rows[-1]


def exact_restored(model,optimizer,saved):
    if len(saved['model'])!=127 or len(saved['optimizer']['state'])!=97:raise ValueError('Original full model/optimizer coverage differs')
    current=model.state_dict()
    if set(current)!=set(saved['model']):raise ValueError('Model state keys differ')
    for key,value in current.items():
        if not torch.equal(value.cpu(),saved['model'][key]) or not torch.isfinite(value).all():raise ValueError('GPU model state differs or is nonfinite')
    restored=optimizer.state_dict()
    if restored['param_groups']!=saved['optimizer']['param_groups'] or set(restored['state'])!=set(saved['optimizer']['state']):
        raise ValueError('GPU optimizer groups/parameter identities differ')
    for key,values in restored['state'].items():
        original=saved['optimizer']['state'][key]
        if set(values)!=set(original):raise ValueError('GPU optimizer state keys differ')
        for name,value in values.items():
            if torch.is_tensor(value):
                if not torch.equal(value.cpu(),original[name]) or not torch.isfinite(value).all():raise ValueError('GPU optimizer tensor differs')
            elif value!=original[name]:raise ValueError('GPU optimizer scalar differs')


def run(ticket_path,fresh=False):
    ticket=read_json(ticket_path);root=owned_root(ticket['project_root'],ticket['project_id']);job=inside(root,ticket['job_directory'])
    if os.environ.get('PTCG_JOB_RUN_ID')!=job.name or os.environ.get('CUDA_VISIBLE_DEVICES')!=ticket['gpu_uuid'] or torch.cuda.device_count()!=1:
        raise ValueError('GPU restore job or selected device identity differs')
    manifest=read_json(job/'training-manifest.raw.json');spec=manifest['existing_restore_source']
    control=None
    if manifest.get('cooperative_resource_control'):
        from .training_control import TrainingControl
        controller=TrainingControl(job,job.name,ticket['lease_epoch'],float(os.environ['PTCG_JOB_DEADLINE_UNIX']),
            lambda stage,**kw:atomic_json(job/'progress.json',{'at':utcnow(),'stage':stage,**kw}))
        control=lambda:controller.wait(90)
        control()
    source,original=resolve(root,ticket['project_id'],spec);point=checkpoint_reference(source,spec);capsule=original['restore_capsule']
    if capsule['origin_checkpoint_sha256']!=manifest['original_checkpoint_sha256']:raise ValueError('Original scientific checkpoint origin changed')
    path=inside(root,source/point['checkpoint']['path'])
    if sha256(path)!=point['checkpoint']['sha256']:raise ValueError('Published original checkpoint bytes changed')
    saved=load_torch(path,map_location='cpu',weights_only=False)
    if {key:saved[key] for key in point['cursor']}!=point['cursor']:raise ValueError('Checkpoint payload cursor differs')
    from .bounded_read import environment_read_context
    from .bounded_io import environment_context
    metadata=next(row for row in original['file_map'] if row['path']=='ptcg_lab/public_cg/metadata.json')
    target=job/'control'/metadata['path']
    if not target.exists():
        with environment_read_context().open(source/'control'/metadata['path']) as stream:data=stream.read()
        if len(data)!=metadata['bytes'] or hashlib.sha256(data).hexdigest()!=metadata['sha256']:raise ValueError('Original public metadata differs')
        environment_context().atomic_write(target,lambda stream:stream.write(data),replace=False)
    elif sha256(target)!=metadata['sha256']:raise ValueError('Copied public metadata differs')
    first,last=fixture_rows(capsule)
    inputs=[]
    for row in [first,last]:
        path=inside(root,source/row['path'])
        if sha256(path)!=row['raw_sha256']:raise ValueError('Original on-policy input bytes changed')
        data=load_torch(path,map_location='cpu',weights_only=False)
        if data['policy_version']!=saved['version'] or any(int(d['policy_version'])!=saved['version'] for d in data['decisions']):
            raise ValueError('GPU restore fixture is not original on-policy data')
        inputs.append(data)
    decisions=inputs[0]['decisions'][:48];rng=inputs[-1]['rng_after']
    if len(decisions)!=48 or len(rng['cuda'])!=1 or len(saved['rng']['cuda'])!=1:raise ValueError('Full48-decision/single-CUDA RNG fixture missing')
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    if control:control()
    model,features,_=load_full_actor(source/'control/agents/frozen/ppo-zero-anchor',original['training_config']['parent_model_sha256'],'cuda')
    model.load_state_dict(saved['model'],strict=True);optimizer=optimizer_for(model);optimizer.load_state_dict(saved['optimizer'])
    exact_restored(model,optimizer,saved)
    def draws(state):
        restore_rng(state)
        return {'python':random.random(),'numpy':np.random.randint(0,1000000,8),'torch':torch.rand(8),'cuda':torch.rand(16,device='cuda').cpu()}
    def equal_draws(a,b):return a['python']==b['python'] and np.array_equal(a['numpy'],b['numpy']) and torch.equal(a['torch'],b['torch']) and torch.equal(a['cuda'],b['cuda'])
    checkpoint_a=draws(saved['rng']);checkpoint_b=draws(saved['rng'])
    a=draws(rng);b=draws(rng)
    if not equal_draws(checkpoint_a,checkpoint_b) or not equal_draws(a,b):raise ValueError('Original checkpoint/tail CPU-CUDA RNG sequence not reproduced')
    if control:control()
    with torch.no_grad():logp,values,_=evaluate_decisions(model,decisions,features.b.STOP)
    error=float(np.max(np.abs(logp.cpu().numpy()-np.asarray([d['old_logp'] for d in decisions]))))
    if error>1e-4 or not torch.isfinite(logp).all() or not torch.isfinite(values).all():raise ValueError('Original strict GPU conditional log-prob parity failed')
    exact_restored(model,optimizer,saved)
    diagnostic=None
    if manifest.get('gradient_diagnostic'):
        if not fresh:
            from .gradient_diagnostic import measure
            diagnostic=measure(model,features,inputs,original['training_config']['ppo'],control)
            diagnostic.update(source_checkpoint_sha256=point['checkpoint']['sha256'],source_run_id=source.name,
                GPU_uuid=ticket['gpu_uuid'],settings=original['training_config']['ppo'])
            exact_restored(model,optimizer,saved)
            atomic_json(job/'critic-gradient-diagnostic.json',diagnostic)
        else:
            diagnostic=read_json(job/'critic-gradient-diagnostic.json')
            if diagnostic['source_checkpoint_sha256']!=point['checkpoint']['sha256'] or diagnostic['optimizer_steps']!=0:
                raise ValueError('Prior gradient evidence identity differs')
    result={'checkpoint_draws':checkpoint_a,'tail_draws':a,'logp':logp.cpu(),'values':values.cpu(),
        'source_checkpoint_sha256':point['checkpoint']['sha256'],'gpu_uuid':ticket['gpu_uuid']}
    if not fresh:atomic_torch_save(job/'gpu-restore-reference.pt',result,replace=False)
    else:
        previous=load_torch(job/'gpu-restore-reference.pt',map_location='cpu',weights_only=False)
        if not equal_draws(previous['checkpoint_draws'],checkpoint_a) or not equal_draws(previous['tail_draws'],a) or not torch.equal(previous['logp'],result['logp']) or not torch.equal(previous['values'],result['values']):
            raise ValueError('Fresh same-GPU process restore differs')
        if previous['source_checkpoint_sha256']!=result['source_checkpoint_sha256'] or previous['gpu_uuid']!=result['gpu_uuid']:raise ValueError('Fresh process source/device differs')
    atomic_json(job/('gpu-restore-audit.json' if fresh else 'gpu-restore-stage.json'),{'at':utcnow(),'complete':True,'passed':True,
        'fresh_process_verified':fresh,'original_TR04_cuda_rng_application_verified':True,'checkpoint_and_last_pending_rng_reproduced':True,
        'full_model_tensors':127,'optimizer_states':97,'cursor':point['cursor'],'source_checkpoint_sha256':point['checkpoint']['sha256'],
        'original_checkpoint_sha256':capsule['origin_checkpoint_sha256'],'source_run_id':source.name,'source_bytes_unchanged':True,
        'conditional_logp_checks':48,'conditional_logp_max_error':error,'conditional_logp_tolerance':1e-4,
        'training_updates_executed':0,'gpu_uuid':ticket['gpu_uuid'],'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
        'same_GPU_outputs_and_rng_exact':fresh,'model_selection_forbidden':True,'long_unattended_training_admitted':False,
        'gradient_diagnostic':diagnostic})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--ticket',required=True);parser.add_argument('--fresh',action='store_true');args=parser.parse_args();run(args.ticket,args.fresh)
