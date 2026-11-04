"""Frozen CPU rollout replica; PPO parameters and optimizer stay on CUDA."""
import random
import numpy as np
import torch
from .phase_timing import timed


class CPUActorRNG:
    """A learner-only stream, independent of opponent model initialization."""
    def __init__(self,state):
        self.generator=torch.Generator(device='cpu');self.generator.set_state(state.cpu())
    def state(self):return self.generator.get_state().clone()
    @timed('learner_features_and_inference')
    def choose(self,model,features,observation,tracker,forced_run,greedy=False):
        if next(model.parameters()).device.type!='cpu':raise ValueError('CPU stream cannot invoke CUDA')
        from .rl_core import choose_decision
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(self.generator.get_state())
            result=choose_decision(model,features,observation,tracker,forced_run,greedy)
            self.generator.set_state(torch.get_rng_state())
        return result


def cpu_replica(learner):
    from ptcg.rl.model import PTCGTransformer,arch_from_state_dict
    before=torch.get_rng_state().clone();python_before=random.getstate();numpy_before=np.random.get_state()
    state={name:value.detach().cpu() for name,value in learner.state_dict().items()}
    # Construct directly on CPU, avoiding a deep copy that would allocate and
    # clone the learner on the GPU. Parameter initialization cannot advance
    # the CPU sampling stream; CUDA RNG is never queried or changed here.
    with torch.random.fork_rng(devices=[]),torch.device('cpu'):
        replica=PTCGTransformer(**arch_from_state_dict(state,heads=8),critic='blind')
        replica.load_state_dict(state,strict=True)
    replica.eval()
    for parameter in replica.parameters():parameter.requires_grad_(False)
    actual=replica.state_dict()
    if set(actual)!=set(state) or any(not torch.equal(actual[k],v) for k,v in state.items()):
        raise ValueError('Incomplete frozen CPU sampler state')
    now=np.random.get_state()
    if (not torch.equal(torch.get_rng_state(),before) or random.getstate()!=python_before
            or now[0]!=numpy_before[0] or not np.array_equal(now[1],numpy_before[1]) or now[2:]!=numpy_before[2:]):
        raise ValueError('Creating CPU sampler changed sampling RNG')
    return replica
