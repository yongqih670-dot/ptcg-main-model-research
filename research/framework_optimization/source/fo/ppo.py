"""Isolated PPO adapter; source baseline and generated diff are in the manifest."""
import torch
from ptcg_lab.rl_core import UNUSED, evaluate_decisions

def ppo_step(model,optimizer,decisions,advantages,returns,stop,clip=.2,entropy_coef=.01,value_coef=.5,
             max_grad_norm=.5,target_kl=None,normalize_advantages=True,critic_updates_shared_trunk=True,
             record_gradient_components=False, *, evaluator=evaluate_decisions, precision=None):
    if getattr(model,'_frozen_tables',None) is not None:raise ValueError('Learner has stale frozen tables')
    device=next(model.parameters()).device
    old=torch.tensor([d['old_logp'] for d in decisions],device=device,dtype=torch.float64)
    adv=torch.as_tensor(advantages,device=device);ret=torch.as_tensor(returns,device=device)
    if normalize_advantages:adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
    logp,values,entropy=evaluator(model,decisions,stop)
    ratio=(logp-old).exp()
    policy_loss=-torch.minimum(ratio*adv,ratio.clamp(1-clip,1+clip)*adv).mean()
    value_loss=(values-ret).square().mean()
    loss=policy_loss+value_coef*value_loss-entropy_coef*entropy.mean()
    if not torch.isfinite(loss):raise ValueError('Nonfinite PPO loss')
    kl=float(((ratio-1)-(logp-old)).mean().detach())
    if target_kl is not None and kl>target_kl:
        return {'skipped_for_kl':True,'approx_kl':kl,'entropy':float(entropy.mean().detach())}
    optimizer.zero_grad(set_to_none=True)
    diagnostics={}
    if record_gradient_components:
        shared=[p for name,p in model.named_parameters() if p.requires_grad and not name.startswith('value_mlp.')]
        for name,component in [('policy_gradient_l2',policy_loss),('weighted_value_shared_gradient_l2',value_coef*value_loss)]:
            gradients=torch.autograd.grad(component,shared,retain_graph=True,allow_unused=True)
            squared=sum(g.detach().double().square().sum() for g in gradients if g is not None)
            diagnostics[name]=float(squared.sqrt()) if torch.is_tensor(squared) else 0.
    if precision is not None and precision.mode!='fp32' and not critic_updates_shared_trunk:
        raise ValueError('AMP candidate does not support detached critic route')
    if critic_updates_shared_trunk:
        if precision is None:loss.backward()
        else:precision.backward(loss,optimizer)
    else:
        # Equivalent value-feature stop-gradient: preserve identical forward
        # values and fit the full blind head, while policy/entropy alone update
        # the shared Transformer. No value gradient is routed into the actor.
        head=[p for name,p in model.named_parameters() if p.requires_grad and name.startswith('value_mlp.')]
        if not head:raise ValueError('Blind value head is missing')
        head_gradients=torch.autograd.grad(value_coef*value_loss,head,retain_graph=True,allow_unused=True)
        (policy_loss-entropy_coef*entropy.mean()).backward()
        for parameter,gradient in zip(head,head_gradients):
            if parameter.grad is not None and torch.count_nonzero(parameter.grad):raise ValueError('Policy unexpectedly depends on value head')
            parameter.grad=gradient.detach() if gradient is not None else None
    if any(p.grad is not None for n,p in model.named_parameters() if n.startswith(UNUSED)):
        raise ValueError('Hidden/auxiliary branch received gradient')
    norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],max_grad_norm,error_if_nonfinite=True)
    if precision is None:optimizer.step()
    else:precision.step(optimizer)
    return {'loss':float(loss.detach()),'policy_loss':float(policy_loss.detach()),'value_loss':float(value_loss.detach()),
        'entropy':float(entropy.mean().detach()),'approx_kl':float(((ratio-1)-(logp-old)).mean().detach()),
        'clip_fraction':float(((ratio-1).abs()>clip).float().mean().detach()),'grad_norm':float(norm),'skipped_for_kl':False,**diagnostics}

