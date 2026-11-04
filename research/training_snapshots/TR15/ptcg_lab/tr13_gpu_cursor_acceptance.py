"""One remaining CUDA cursor check, separating exact restore from GPU arithmetic.

Compare an uninterrupted repeat as a numerical control. Training code/settings
are never changed. Restore, RNG, input, permutation and counters remain exact.
"""
import argparse
import copy
import os
from pathlib import Path
import time
import numpy as np
import torch
from .io import read_json,atomic_json,utcnow,sha256
from .tensor_io import atomic_torch_save,load_torch
from .training_checkpoint import load_training_checkpoint
from .rl_resume import resume_journal
from .rl_core import load_full_actor,optimizer_for,rng_state,restore_rng,gae,ppo_step
from .framework_v2_training import semantic_digest
from .training_control import TrainingControl
from .server_workspace import owned_root,inside
from .retained_source import resolve
from . import ppo_transaction as tx


def compare_tensors(reference,actual,repeat):
    rows=[];okay=True
    def visit(a,b,c,path):
        nonlocal okay
        if torch.is_tensor(a):
            a=a.detach().cpu();b=b.detach().cpu();c=c.detach().cpu()
            if a.shape!=b.shape or a.dtype!=b.dtype:raise ValueError('GPU output tensor format changed')
            if not a.is_floating_point():
                if not torch.equal(a,b):okay=False
                return
            if not torch.isfinite(a).all() or not torch.isfinite(b).all() or not torch.isfinite(c).all():
                okay=False;return
            difference=float((a.double()-b.double()).abs().max()) if a.numel() else 0
            control=float((a.double()-c.double()).abs().max()) if a.numel() else 0
            scale=max(1.,float(a.abs().max())) if a.numel() else 1.
            allowed=min(5e-6, max(8*control,8*torch.finfo(a.dtype).eps*scale))
            if difference>allowed:okay=False
            if difference or control:rows.append({'path':path,'resume_max_abs':difference,'repeat_max_abs':control,'allowed_abs':allowed})
        elif isinstance(a,dict):
            if set(a)!=set(b) or set(a)!=set(c):raise ValueError('GPU output dictionary changed')
            for k in a:visit(a[k],b[k],c[k],path+'/'+str(k))
        elif isinstance(a,(list,tuple)):
            if len(a)!=len(b) or len(a)!=len(c):raise ValueError('GPU output list changed')
            for i,(x,y,z) in enumerate(zip(a,b,c)):visit(x,y,z,path+'/'+str(i))
        elif a!=b or a!=c:okay=False
    visit(reference,actual,repeat,'')
    return {'passed':okay,'changed_tensors':len(rows),'maximum_resume_abs':max([r['resume_max_abs'] for r in rows],default=0.),
            'maximum_repeat_abs':max([r['repeat_max_abs'] for r in rows],default=0.),'tensor_differences':rows}


def main():
    p=argparse.ArgumentParser();p.add_argument('--ticket',required=True);p.add_argument('--cold',action='store_true');args=p.parse_args()
    ticket=read_json(args.ticket);root=owned_root(ticket['project_root'],ticket['project_id']);job=inside(root,ticket['job_directory'])
    manifest=read_json(job/'training-manifest.raw.json')
    if not manifest.get('GPU_cursor_repair_only') or not manifest.get('engineering_only') or ticket['new_diagnostic_identity']!='TR13':
        raise ValueError('Only the registered disposable remaining CUDA acceptance is allowed')
    control=TrainingControl(job,job.name,ticket['lease_epoch'],float(os.environ['PTCG_JOB_DEADLINE_UNIX']),
        lambda stage,**kw:atomic_json(job/'progress.json',{'at':utcnow(),'stage':'remaining_CUDA_'+stage,**kw}))
    torch.set_num_threads(2);control.wait(90);torch.backends.cuda.matmul.allow_tf32=False
    folder=job/'engineering-gpu-cursor';folder.mkdir(exist_ok=True)
    source,previous=resolve(root,ticket['project_id'],manifest['existing_restore_source'])
    config=read_json(source/'payload-config.json');settings=config['ppo']
    model,features,_=load_full_actor(job/'control'/config['bundle_path'],config['parent_model_sha256'],'cuda')
    opt=optimizer_for(model,settings['actor_learning_rate'],settings['value_head_learning_rate'])
    if args.cold:
        cut=load_torch(folder/'cut.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(cut['model']);opt.load_state_dict(cut['optimizer']);restore_rng(cut['rng']);partial=cut['partial']
        current={'model':model.state_dict(),'optimizer':opt.state_dict(),'rng':rng_state(),'partial':partial}
        expected_cut={k:cut[k] for k in current}
        if semantic_digest(current)!=semantic_digest(expected_cut):raise ValueError('Latest cut parameters/optimizer/allRNG/cursor did not restore exactly')
        decisions=None;selected=cut['next_decisions'];selected_indices=cut['next_indices']
    else:
        loaded=load_training_checkpoint(source,config['archive_restore'],manifest['existing_restore_source']['lease_epoch'])
        parent=loaded['saved'];journal=resume_journal(source,parent,loaded['archive_spec'],loaded['digest']);decisions=journal['decisions']
        if len(decisions)!=6807 or parent['version']!=15:raise ValueError('Remaining test input is not the unchanged full v15 pending state')
        initial_rng=journal['rng_after'] if tx.has_postcheckpoint_tail(parent,journal) else parent['rng']
        advantages,returns=gae(decisions,settings['gamma'],settings['gae_lambda']);advantages=(advantages-advantages.mean())/(advantages.std()+1e-8)
    def one():
        control.wait(90);chosen=tx.next_batch(partial,settings['epochs']);began=time.monotonic()
        if chosen is None:raise ValueError('Two real original PPO calls required')
        if decisions is None:
            if not np.array_equal(chosen,selected_indices):raise ValueError('Cold minibatch permutation/input differs')
            batch=selected
        else:batch=[decisions[i] for i in chosen]
        metrics=ppo_step(model,opt,batch,partial['advantages'][chosen],partial['returns'][chosen],features.b.STOP,
            clip=settings['clip'],entropy_coef=settings['entropy_coefficient'],value_coef=settings['value_loss_coefficient'],
            max_grad_norm=settings['max_grad_norm'],target_kl=settings['target_kl'],normalize_advantages=False,
            critic_updates_shared_trunk=settings.get('critic_updates_shared_trunk',True),
            record_gradient_components=partial['epoch']==0 and partial['offset']==0)
        tx.acknowledge(partial,chosen,metrics,settings['epochs'],6807);control.after_gpu_work(time.monotonic()-began)
    def value():return {'model':model.state_dict(),'optimizer':opt.state_dict(),'rng':rng_state(),'partial':partial}
    if not args.cold:
        for repeat in (False,True):
            model.load_state_dict(parent['model']);opt=optimizer_for(model,settings['actor_learning_rate'],settings['value_head_learning_rate'])
            opt.load_state_dict(copy.deepcopy(parent['optimizer']));restore_rng(initial_rng)
            partial=tx.begin(decisions,advantages,returns,settings['minibatch_initial']);one()
            if partial['positive_optimizer_steps']!=1:raise ValueError('The first real optimizer step must be positive')
            if not repeat:
                indices=tx.next_batch(partial,settings['epochs'])
                atomic_torch_save(folder/'cut.pt',{**value(),'next_indices':indices,'next_decisions':[decisions[i] for i in indices]},replace=False)
            one();atomic_torch_save(folder/('repeat.pt' if repeat else 'expected.pt'),value(),replace=False)
        atomic_json(job/'remaining-CUDA-warm.json',{'complete':True,'pending_decisions':len(decisions),'scientific_decision_credit':0})
    else:
        tx.validate(partial,6807,settings['epochs']);one();actual=value()
        expected=load_torch(folder/'expected.pt',map_location='cpu',weights_only=False);repeat=load_torch(folder/'repeat.pt',map_location='cpu',weights_only=False)
        discrete=lambda x:{'rng':x['rng'],'partial':{k:v for k,v in x['partial'].items() if k!='logs'}}
        discrete_exact=semantic_digest(discrete(actual))==semantic_digest(discrete(expected))==semantic_digest(discrete(repeat))
        comparison=compare_tensors({k:expected[k] for k in ['model','optimizer']},{k:actual[k] for k in ['model','optimizer']},{k:repeat[k] for k in ['model','optimizer']})
        logs_ok=True;max_log_difference=0.
        for a,b,c in zip(expected['partial']['logs'],actual['partial']['logs'],repeat['partial']['logs']):
            if set(a)!=set(b) or set(a)!=set(c):logs_ok=False;continue
            for k in a:
                if isinstance(a[k],float):
                    delta=abs(a[k]-b[k]);max_log_difference=max(max_log_difference,delta)
                    bound=min(1e-4*max(1.,abs(a[k])),max(8*abs(a[k]-c[k]),1e-7*max(1.,abs(a[k]))))
                    if not np.isfinite(delta) or delta>bound:logs_ok=False
                elif a[k]!=b[k] or a[k]!=c[k]:logs_ok=False
        exact=semantic_digest(actual)==semantic_digest(expected)
        report={'at':utcnow(),'complete':bool(discrete_exact and comparison['passed'] and logs_ok),
            'fresh_CUDA_process_after_positive_step':True,'checkpoint_model_optimizer_all_RNG_permutation_exact':True,
            'model_optimizer_all_RNG_permutation_logs_exact':exact,'RNG_permutation_input_counters_exact':discrete_exact,
            'PPO_continuation_within_measured_GPU_repeatability':comparison['passed'] and logs_ok,
            'uninterrupted_repeat_bitwise_exact':semantic_digest(repeat)==semantic_digest(expected),
            'original_PPO_function_used':True,'scientific_decision_credit':0,'scientific_parent_model_unchanged':True,
            'actual_pending_decisions':6807,'positive_optimizer_steps_in_disposable_copy':partial['positive_optimizer_steps'],
            'numerical_comparison':comparison,'maximum_log_abs_difference':max_log_difference,
            'criterion':'Exact saved state/RNG/cursor restoration; unchanged GPU PPO arithmetic compared with an uninterrupted repeat, capped5e-6 tensor absolute error. Historical bitwise continuation failure retained.'}
        atomic_json(job/'framework-v2-GPU-cursor-difference.json',report)
        if not report['complete']:raise ValueError('Remaining CUDA restoration/repeatability check failed; retain diagnostics')


if __name__=='__main__':main()
