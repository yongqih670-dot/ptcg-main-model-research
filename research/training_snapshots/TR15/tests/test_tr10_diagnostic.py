import copy
import unittest
from pathlib import Path
from ptcg_lab.fresh_experiment import validate_single_change
from ptcg_lab.io import read_json
from ptcg_lab.rl_core import gae

ROOT=Path(__file__).resolve().parents[1]


class MonteCarloDiagnosticTests(unittest.TestCase):
    def test_exact_reference_allows_only_registered_lambda_change(self):
        reference=read_json(ROOT/'runs/20261023T144339Z_tr02-ppo-lr1e5_9a0e14/config.json')
        candidate=copy.deepcopy(reference)
        candidate['hypothesis_id']='TR10';candidate['ppo']['gae_lambda']=1.
        self.assertEqual(validate_single_change(candidate,reference),'TR10')
        for key,value in [('epochs',4),('actor_learning_rate',3e-6),('entropy_coefficient',.001)]:
            changed=copy.deepcopy(candidate);changed['ppo'][key]=value
            with self.assertRaises(ValueError):validate_single_change(changed,reference)
        candidate['ppo']['gae_lambda']=.99
        with self.assertRaises(ValueError):validate_single_change(candidate,reference)

    def test_production_gae_lambda_one_uses_separate_round_outcomes(self):
        decisions=[{'old_value':v,'reward':r,'terminal':terminal} for v,r,terminal in
                   [(.3,0.,False),(-.2,1.,True),(.7,0.,False),(.1,-1.,True)]]
        advantage,returns=gae(decisions,gamma=1.,lam=1.)
        for actual,expected in zip(returns,[1.,1.,-1.,-1.]):self.assertAlmostEqual(float(actual),expected,places=6)
        self.assertAlmostEqual(float(advantage[0]),.7,places=6)
