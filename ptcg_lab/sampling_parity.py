"""Check real CPU rollout probabilities against the learner before its update."""
import math
import numpy as np
import torch
from .io import atomic_json,utcnow


def validate_outputs(decisions,logp,values):
    if len(decisions)!=48 or logp.numel()!=48 or values.numel()!=48:
        raise ValueError('Exactly48 real current-policy decisions required')
    expected_logp=np.asarray([d['old_logp'] for d in decisions],dtype=np.float64)
    expected_values=np.asarray([d['old_value'] for d in decisions],dtype=np.float64)
    observed_logp=logp.detach().cpu().numpy().reshape(-1)
    observed_values=values.detach().cpu().numpy().reshape(-1)
    if not all(np.isfinite(v).all() for v in [expected_logp,expected_values,observed_logp,observed_values]):
        raise ValueError('Nonfinite CPU/GPU rollout parity values')
    error=float(np.max(np.abs(observed_logp-expected_logp)))
    value_error=float(np.max(np.abs(observed_values-expected_values)))
    if error>1e-4 or value_error>2e-4:raise ValueError('CPU/GPU conditional probability/value parity failed before optimizer')
    return {'conditional_logp_max_error':error,'conditional_logp_tolerance':1e-4,
            'value_max_error':value_error,'value_tolerance':2e-4,'real_current_policy_decisions':48}


def before_optimizer(model,decisions,stop,output,version):
    from .rl_core import evaluate_decisions,rng_state
    if next(model.parameters()).device.type!='cuda':raise ValueError('PPO learner must remain on CUDA')
    fixture=decisions[:48]
    if len(fixture)!=48 or any(int(d['policy_version'])!=version for d in fixture):
        raise ValueError('Mixed or missing current policy decisions')
    previous=rng_state()
    with torch.no_grad():logp,values,_=evaluate_decisions(model,fixture,stop)
    result=validate_outputs(fixture,logp,values)
    current=rng_state()
    if (previous['python']!=current['python'] or previous['numpy'][0]!=current['numpy'][0]
            or not np.array_equal(previous['numpy'][1],current['numpy'][1]) or previous['numpy'][2:]!=current['numpy'][2:]
            or not torch.equal(previous['torch'],current['torch']) or len(previous['cuda'])!=len(current['cuda'])
            or any(not torch.equal(a,b) for a,b in zip(previous['cuda'],current['cuda']))):
        raise ValueError('Parity check changed sampling or PPO RNG')
    atomic_json(output/f'CPU-GPU-parity-v{version:04d}.json',{'at':utcnow(),'passed':True,'policy_version':version,
        'before_optimizer':True,'sampling_device':'cpu','PPO_device':'cuda','no_RNG_draws':True,**result})
