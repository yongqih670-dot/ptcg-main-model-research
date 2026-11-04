"""Descriptive old on-policy gradient components; never an optimizer step."""
import numpy as np
import torch
from .rl_core import evaluate_decisions,gae


def measure(model,features,inputs,settings,control=None):
    parameters=[p for name,p in model.named_parameters() if p.requires_grad and not name.startswith('value_mlp.')]
    result=[];lo,hi=features.b.OFFSETS['oracle']
    for data in inputs:
        decisions=data['decisions'];summary=data['summary']
        if summary['statuses']!=['DONE','DONE'] or summary['rounds'] not in (2,3):raise ValueError('Gradient diagnostic requires complete BO3')
        if any(step['rows'][:,lo:hi].any() for d in decisions for step in d['steps']):raise ValueError('Hidden actor data in gradient diagnostic')
        advantages,returns=gae(decisions,settings['gamma'],settings['gae_lambda'])
        advantages=(advantages-advantages.mean())/(advantages.std()+1e-8)
        components=[[torch.zeros_like(p) for p in parameters] for _ in range(3)]
        values_all=[]
        for offset in range(0,len(decisions),64):
            if control:control()
            batch=decisions[offset:offset+64];logp,values,entropy=evaluate_decisions(model,batch,features.b.STOP)
            old=torch.tensor([d['old_logp'] for d in batch],device=logp.device,dtype=torch.float64)
            adv=torch.as_tensor(advantages[offset:offset+len(batch)],device=logp.device)
            ret=torch.as_tensor(returns[offset:offset+len(batch)],device=logp.device)
            ratio=(logp-old).exp();clip=settings['clip']
            policy=-torch.minimum(ratio*adv,ratio.clamp(1-clip,1+clip)*adv).mean()
            entropy_loss=-settings['entropy_coefficient']*entropy.mean()
            value_loss=settings['value_loss_coefficient']*(values-ret).square().mean()
            for number,loss in enumerate([policy,entropy_loss,value_loss]):
                gradients=torch.autograd.grad(loss,parameters,retain_graph=number<2,allow_unused=True)
                for aggregate,gradient in zip(components[number],gradients):
                    if gradient is not None:aggregate.add_(gradient.detach(),alpha=len(batch)/len(decisions))
            values_all.extend(values.detach().cpu().tolist())
        vectors=[torch.cat([p.flatten() for p in component]).double() for component in components]
        if any(not torch.isfinite(v).all() for v in vectors):raise ValueError('Nonfinite diagnostic gradient')
        norms=[float(v.norm().cpu()) for v in vectors];combined=vectors[0]+vectors[1];combined_norm=float(combined.norm().cpu())
        cosine=lambda a,b:float(torch.dot(a,b).cpu()/(a.norm().cpu()*b.norm().cpu())) if float(a.norm())>0 and float(b.norm())>0 else None
        result.append({'series_id':data['series_id'],'policy_version':data['policy_version'],'decisions':len(decisions),
            'rounds':summary['rounds'],'shared_trainable_parameters':sum(p.numel() for p in parameters),
            'policy_gradient_l2':norms[0],'weighted_entropy_gradient_l2':norms[1],
            'weighted_value_shared_gradient_l2':norms[2],'policy_plus_entropy_gradient_l2':combined_norm,
            'value_to_policy_entropy_l2_ratio':norms[2]/combined_norm if combined_norm else None,
            'entropy_to_policy_l2_ratio':norms[1]/norms[0] if norms[0] else None,
            'value_vs_policy_entropy_cosine':cosine(vectors[2],combined),
            'value_mse':float(np.mean((np.asarray(values_all).reshape(-1)-returns)**2))})
        if any(p.grad is not None for p in model.parameters()):raise ValueError('Diagnostic unexpectedly populated optimizer gradients')
    return {'complete':True,'complete_BO3':len(result),'rows':result,'optimizer_steps':0,
        'old_complete_training_BO3_only':True,'selection_rule':'First and last of35 frozen pending TR04 BO3, chosen before gradient observation',
        'limits':'Within-BO3 standardized advantages and64-decision exact weighted accumulation; two training BO3 are descriptive. Raw preclip gradients do not prove Adam step dominance, competitive strength, or causality.',
        'no_model_optimizer_or_source_bytes_changed':True,'promotion_eligible':False}
