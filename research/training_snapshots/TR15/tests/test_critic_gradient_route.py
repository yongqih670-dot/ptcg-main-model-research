import copy
import unittest
import numpy as np
import torch
from tests.test_gradient_diagnostic import TinyActor
from ptcg_lab.rl_core import ppo_step


class CriticGradientRouteTests(unittest.TestCase):
    def fixture(self,model):
        rows=np.asarray([[1.,0.]],np.float32)
        with torch.no_grad():scores,_=model.forward_eval(torch.from_numpy(rows));old=float(scores.log_softmax(-1)[0,0])
        return [{'steps':[{'rows':rows,'mapping':np.asarray([[0,1,2]]),'n':2,'target':0}],
                 'old_logp':old,'old_value':.5,'reward':1.,'terminal':True}]
    def test_value_head_still_learns_while_value_alone_cannot_move_shared_actor(self):
        baseline=TinyActor();detached=copy.deepcopy(baseline);before=copy.deepcopy(baseline.state_dict())
        for model,route in [(baseline,True),(detached,False)]:
            optimizer=torch.optim.SGD(model.parameters(),lr=.1)
            ppo_step(model,optimizer,self.fixture(model),np.asarray([0.],np.float32),np.asarray([1.],np.float32),2,
                     entropy_coef=0,value_coef=.5,max_grad_norm=1e9,normalize_advantages=False,critic_updates_shared_trunk=route)
        self.assertFalse(torch.equal(baseline.trunk.weight,before['trunk.weight']))
        self.assertTrue(torch.equal(detached.trunk.weight,before['trunk.weight']))
        self.assertFalse(torch.equal(detached.value_mlp.weight,before['value_mlp.weight']))
        self.assertTrue(torch.equal(detached.value_mlp.weight,baseline.value_mlp.weight))
        self.assertTrue(torch.equal(detached.value_mlp.bias,baseline.value_mlp.bias))
    def test_policy_gradient_keeps_training_transformer_when_value_route_is_disabled(self):
        model=TinyActor();before=copy.deepcopy(model.state_dict());optimizer=torch.optim.SGD(model.parameters(),lr=.1)
        ppo_step(model,optimizer,self.fixture(model),np.asarray([1.],np.float32),np.asarray([1.],np.float32),2,
                 entropy_coef=0,value_coef=0,max_grad_norm=1e9,normalize_advantages=False,critic_updates_shared_trunk=False)
        self.assertFalse(torch.equal(model.trunk.weight,before['trunk.weight']))
        self.assertTrue(torch.equal(model.value_mlp.weight,before['value_mlp.weight']))
