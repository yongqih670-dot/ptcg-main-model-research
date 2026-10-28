"""Full inherited Transformer, blind values and conditional-decision PPO."""
import copy
import os
import random
from pathlib import Path
import numpy as np
import torch
from .io import sha256
from .token_features import TokenFeatures, forced_selection
from .replay_data import actor_observation
from .tensor_io import load_torch
from .phase_timing import timed

UNUSED=('oracle_proj.','critic_readout','critic_blocks.','oracle_value_mlp.','aux_hand.','aux_prize.')


class RLFeatures(TokenFeatures):
    """Preserve parent equivalence mask for supported sizes; cover overflow."""
    def make(self,obs,tracker,forced_run,picked):
        obs=actor_observation(obs)
        n=len(obs['select']['option']);b=self.b
        if n>b.STOP:
            return super().make(obs,tracker,forced_run,picked)
        row=np.zeros(b.OBS_SIZE,np.float32)
        self.encode(row,obs,picked,len(picked)>=obs['select']['minCount'],forced_run,tracker=tracker)
        self.field(row,'global_f')[b.GlobF.ACTOR_SEAT]=0
        mapping=np.full(b.MAX_OPTIONS,-1,np.int32);mapping[:n]=range(n);mapping[b.STOP]=n
        if self.field(row,'oracle').any():raise ValueError('Oracle feature present')
        return row[None],mapping[None]


def load_full_actor(bundle,parent_hash,device='cpu',reset_value=True,device_guard=None):
    features=RLFeatures(bundle)
    from ptcg.rl.model import PTCGTransformer,arch_from_state_dict
    path=Path(bundle)/'model.pt'
    if sha256(path)!=parent_hash:raise ValueError('Frozen parent hash mismatch')
    state={k.removeprefix('module.'):v for k,v in load_torch(path,map_location='cpu',weights_only=True).items()}
    arch=arch_from_state_dict(state,heads=8)
    model=PTCGTransformer(**arch,critic='blind')
    model.load_state_dict(state,strict=True)
    if reset_value:
        torch.nn.init.zeros_(model.value_mlp[-1].weight)
        torch.nn.init.zeros_(model.value_mlp[-1].bias)
    for name,p in model.named_parameters():
        if name.startswith(UNUSED):p.requires_grad_(False)
    mismatches=[k for k,v in model.state_dict().items() if not k.startswith('value_mlp.2.') and not torch.equal(v,state[k])]
    if mismatches:raise ValueError('Incomplete actor inheritance: '+str(mismatches))
    if device_guard:device_guard()
    model.to(device);model.eval()
    return model,features,{'parent_sha256':parent_hash,'architecture':arch,'all_inherited_tensor_count':len(state),
        'non_value_output_tensor_mismatches':mismatches,'reset_keys':['value_mlp.2.weight','value_mlp.2.bias'] if reset_value else [],
        'parameters':sum(p.numel() for p in model.parameters()),'critic':'blind','learner_table_cache':False,
        'mask_contract':'Parent class mask for <=63 options; complete raw-index global support for overflow, one STOP'}


def merged_scores(scores,mapping,n,stop):
    # Differentiable gather; no chunk-local normalization. STOP represented once.
    indices={int(t):(i,j) for i,row in enumerate(mapping) for j,t in enumerate(row) if 0<=t<n}
    if len(indices)!=n:raise ValueError('Missing raw candidate')
    if scores.shape[0]>1 and not torch.allclose(scores[:,stop],scores[0,stop].expand(scores.shape[0]),atol=1e-4,rtol=1e-4):
        raise ValueError('Chunk STOP scores disagree')
    return torch.stack([scores[indices[j]] for j in range(n)]+[scores[0,stop]])


def choose_decision(model,features,obs,tracker,forced_run,greedy=False):
    sel=obs['select'];n=len(sel['option'])
    if forced_selection(sel):
        return list(range(n)) if sel['maxCount'] else [],None
    picked=[];steps=[];value=None;logp=0.
    with torch.no_grad():
        while len(picked)<sel['maxCount']:
            rows,mapping=features.make(obs,tracker,forced_run,picked)
            tensor=torch.from_numpy(rows).to(next(model.parameters()).device)
            scores,values=model.forward_eval(tensor)
            logits=merged_scores(scores,mapping,n,features.b.STOP)
            dist=torch.distributions.Categorical(logits=logits.double())
            target=logits.argmax() if greedy else dist.sample()
            prob=dist.log_prob(target)
            if not torch.isfinite(prob):raise ValueError('Nonfinite conditional probability')
            if value is None:value=float(values[0,0])
            steps.append({'rows':rows,'mapping':mapping,'n':n,'target':int(target)})
            logp+=float(prob)
            if int(target)==n:break
            if int(target) in picked:raise ValueError('Duplicate pick')
            picked.append(int(target))
    if not sel['minCount']<=len(picked)<=sel['maxCount']:raise ValueError('Invalid sequence')
    return picked,{'steps':steps,'old_logp':logp,'old_value':value,'reward':0.,'terminal':False}


def evaluate_decisions(model,decisions,stop):
    rows=np.concatenate([s['rows'] for d in decisions for s in d['steps']])
    scores,values=model.forward_eval(torch.from_numpy(rows).to(next(model.parameters()).device))
    offset=0;logps=[];entropies=[];vs=[]
    for d in decisions:
        terms=[];entropy=[];vs.append(values[offset,0])
        for s in d['steps']:
            count=len(s['rows']);logits=merged_scores(scores[offset:offset+count],s['mapping'],s['n'],stop)
            distribution=torch.distributions.Categorical(logits=logits.double())
            terms.append(distribution.log_prob(torch.as_tensor(s['target'],device=scores.device)))
            entropy.append(distribution.entropy());offset+=count
        logps.append(torch.stack(terms).sum());entropies.append(torch.stack(entropy).sum())
    return torch.stack(logps),torch.stack(vs),torch.stack(entropies)


@timed('advantage_and_return_preparation')
def gae(decisions,gamma=1.,lam=.95,bootstrap=0.):
    advantages=np.zeros(len(decisions),np.float32);last=0.;next_value=bootstrap
    for i in reversed(range(len(decisions))):
        d=decisions[i];cont=0. if d['terminal'] else 1.
        delta=d['reward']+gamma*cont*next_value-d['old_value']
        last=delta+gamma*lam*cont*last;advantages[i]=last;next_value=d['old_value']
    returns=advantages+np.array([d['old_value'] for d in decisions],np.float32)
    return advantages,returns


def optimizer_for(model,actor_lr=3e-5,value_lr=1e-4):
    actor=[];value=[]
    for name,p in model.named_parameters():
        if p.requires_grad:(value if name.startswith('value_mlp.') else actor).append(p)
    assert len({id(p) for p in actor+value})==len(actor+value)
    return torch.optim.Adam([{'params':actor,'lr':actor_lr},{'params':value,'lr':value_lr}],eps=1e-5)


@timed('PPO_forward_backward_step_host_elapsed')
def ppo_step(model,optimizer,decisions,advantages,returns,stop,clip=.2,entropy_coef=.01,value_coef=.5,
             max_grad_norm=.5,target_kl=None,normalize_advantages=True,critic_updates_shared_trunk=True,
             record_gradient_components=False):
    if getattr(model,'_frozen_tables',None) is not None:raise ValueError('Learner has stale frozen tables')
    device=next(model.parameters()).device
    old=torch.tensor([d['old_logp'] for d in decisions],device=device,dtype=torch.float64)
    adv=torch.as_tensor(advantages,device=device);ret=torch.as_tensor(returns,device=device)
    if normalize_advantages:adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
    logp,values,entropy=evaluate_decisions(model,decisions,stop)
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
    if critic_updates_shared_trunk:loss.backward()
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
    optimizer.step()
    return {'loss':float(loss.detach()),'policy_loss':float(policy_loss.detach()),'value_loss':float(value_loss.detach()),
        'entropy':float(entropy.mean().detach()),'approx_kl':float(((ratio-1)-(logp-old)).mean().detach()),
        'clip_fraction':float(((ratio-1).abs()>clip).float().mean().detach()),'grad_norm':float(norm),'skipped_for_kl':False,**diagnostics}


def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
        'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(rng):
    random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.set_rng_state(rng['torch'])
    if rng['cuda']:torch.cuda.set_rng_state_all(rng['cuda'])


def checkpoint(path,model,optimizer,**extra):
    from .tensor_io import atomic_torch_save
    return atomic_torch_save(path,{'model':model.state_dict(),'optimizer':optimizer.state_dict(),'rng':rng_state(),**extra})
