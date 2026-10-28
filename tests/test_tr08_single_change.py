import copy
from pathlib import Path
import unittest
import numpy as np
import torch
from ptcg_lab.fresh_experiment import validate_single_change
from ptcg_lab.io import read_json
from ptcg_lab.sampling_parity import validate_outputs


class TR08Tests(unittest.TestCase):
    def test_exact_original_reference_and_only_actor_rate(self):
        root=Path(__file__).resolve().parents[1]
        ref=read_json(root/'runs/20261023T144339Z_tr02-ppo-lr1e5_9a0e14/config.json')
        c=copy.deepcopy(ref);c['hypothesis_id']='TR08';c['ppo']['actor_learning_rate']=3e-6
        self.assertEqual(validate_single_change(c,ref),'TR08')
        for field,value in [('value_loss_coefficient',.1),('entropy_coefficient',.001),('gae_lambda',.99),('critic_updates_shared_trunk',False),('actor_learning_rate',1e-6)]:
            altered=copy.deepcopy(c);altered['ppo'][field]=value
            with self.assertRaises(ValueError):validate_single_change(altered,ref)
        altered=copy.deepcopy(c);altered['target_decisions']=262144
        with self.assertRaises(ValueError):validate_single_change(altered,ref)

    def test_real_conditional_action_and_value_tolerances_never_relaxed(self):
        d=[{'old_logp':-.5,'old_value':.2} for _ in range(48)]
        result=validate_outputs(d,torch.full((48,),-.5),torch.full((48,),.2))
        self.assertEqual(result['conditional_logp_tolerance'],1e-4)
        for logp,value in [(-.501,.2),(-.5,.201),(float('nan'),.2),(-.5,float('inf'))]:
            with self.assertRaises(ValueError):validate_outputs(d,torch.full((48,),logp),torch.full((48,),value))
        with self.assertRaises(ValueError):validate_outputs(d[:47],torch.zeros(48),torch.zeros(48))


if __name__=='__main__':unittest.main()
