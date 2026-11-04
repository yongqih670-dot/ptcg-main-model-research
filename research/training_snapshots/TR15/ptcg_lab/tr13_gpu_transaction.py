"""Two original PPO minibatches in disposable copies: continuous vs cold resume.

No native games, scientific updates, or decision credit are generated here.
The unmodified ppo_step, inherited optimizer and actual v15 pending data are used.
"""
import argparse
import copy
import os
from pathlib import Path
import time
import torch
from .io import read_json,atomic_json,utcnow
from .rl_core import load_full_actor,optimizer_for,rng_state,restore_rng,gae,ppo_step
from .rl_resume import resume_journal
from .training_checkpoint import load_training_checkpoint
from .tensor_io import atomic_torch_save,load_torch
from .framework_v2_training import semantic_digest
from . import ppo_transaction as tx
from .training_control import TrainingControl


def main():
    p=argparse.ArgumentParser();p.add_argument('--ticket',required=True);p.add_argument('--cold',action='store_true');a=p.parse_args()
    ticket=read_json(a.ticket);job=Path(ticket['job_directory']);config=read_json(job/'payload-config.json')
    if ticket.get('new_diagnostic_identity')!='TR13' or os.environ.get('PTCG_JOB_RUN_ID')!=job.name:
        raise ValueError('TR13 GPU cursor proof identity differs')
    control=TrainingControl(job,job.name,ticket['lease_epoch'],float(os.environ['PTCG_JOB_DEADLINE_UNIX']),
        lambda stage,**kw:atomic_json(job/'progress.json',{'at':utcnow(),'stage':'cursor_difference_'+stage,**kw}))
    torch.set_num_threads(2);control.wait(90);torch.backends.cuda.matmul.allow_tf32=False
    model,features,_=load_full_actor(job/'control'/config['bundle_path'],config['parent_model_sha256'],'cuda')
    settings=config['ppo'];opt=optimizer_for(model,settings['actor_learning_rate'],settings['value_head_learning_rate'])
    loaded=load_training_checkpoint(job,config['archive_restore'],ticket['lease_epoch'])
    parent=loaded['saved'];journal=resume_journal(job,parent,loaded['archive_spec'],loaded['digest'])
    decisions=journal['decisions']
    if len(decisions)<512:raise ValueError('GPU cursor proof requires an actual on-policy pending input')
    folder=job/'engineering-gpu-cursor';folder.mkdir(exist_ok=True)
    if a.cold:
        cut=load_torch(folder/'cut.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(cut['model']);opt.load_state_dict(cut['optimizer']);restore_rng(cut['rng']);partial=cut['partial']
    else:
        model.load_state_dict(parent['model']);opt.load_state_dict(copy.deepcopy(parent['optimizer']))
        restore_rng(journal['rng_after'] if journal['rng_after'] is not None else parent['rng'])
        advantages,returns=gae(decisions,settings['gamma'],settings['gae_lambda'])
        advantages=(advantages-advantages.mean())/(advantages.std()+1e-8)
        partial=tx.begin(decisions,advantages,returns,settings['minibatch_initial'])
    def one():
        control.wait(90)
        chosen=tx.next_batch(partial,settings['epochs'])
        if chosen is None:raise ValueError('GPU proof requires two minibatch calls')
        began=__import__('time').monotonic()
        metrics=ppo_step(model,opt,[decisions[i] for i in chosen],partial['advantages'][chosen],partial['returns'][chosen],features.b.STOP,
            clip=settings['clip'],entropy_coef=settings['entropy_coefficient'],value_coef=settings['value_loss_coefficient'],
            max_grad_norm=settings['max_grad_norm'],target_kl=settings['target_kl'],normalize_advantages=False,
            critic_updates_shared_trunk=settings.get('critic_updates_shared_trunk',True),
            record_gradient_components=partial['epoch']==0 and partial['offset']==0)
        tx.acknowledge(partial,chosen,metrics,settings['epochs'],min(6807,len(decisions)))
        control.after_gpu_work(__import__('time').monotonic()-began)
    def value():return {'model':model.state_dict(),'optimizer':opt.state_dict(),'rng':rng_state(),'partial':partial}
    if not a.cold:
        one()
        if partial['positive_optimizer_steps']!=1:raise ValueError('GPU difference proof did not execute a positive optimizer step')
        atomic_torch_save(folder/'cut.pt',value(),replace=False)
        one()
        atomic_json(folder/'expected.json',{'semantic_sha256':semantic_digest(value()),
            'positive_optimizer_steps':partial['positive_optimizer_steps'],'pending_decisions':len(decisions)})
    else:
        tx.validate(partial,len(decisions),settings['epochs']);one()
        expected=read_json(folder/'expected.json')
        if semantic_digest(value())!=expected['semantic_sha256']:
            raise ValueError('Real GPU interrupted PPO differs from continuous original PPO')
        atomic_json(job/'framework-v2-GPU-cursor-difference.json',{'at':utcnow(),'complete':True,
            'fresh_CUDA_process_after_positive_step':True,'model_optimizer_all_RNG_permutation_logs_exact':True,
            'original_PPO_function_used':True,'positive_optimizer_steps_in_disposable_copy':partial['positive_optimizer_steps'],
            'actual_pending_decisions':len(decisions),'scientific_decision_credit':0,
            'scientific_parent_model_unchanged':True})


if __name__=='__main__':main()
