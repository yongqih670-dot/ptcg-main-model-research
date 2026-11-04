import copy
from pathlib import Path
import unittest
from ptcg_lab.fresh_experiment import validate_single_change
from ptcg_lab.io import read_json


class TR09Tests(unittest.TestCase):
    def test_only_epochs_change_and_old_learning_rate_is_restored(self):
        root=Path(__file__).resolve().parents[1]
        ref=read_json(root/'runs/20261023T144339Z_tr02-ppo-lr1e5_9a0e14/config.json')
        value=copy.deepcopy(ref);value['hypothesis_id']='TR09';value['ppo']['epochs']=4
        self.assertEqual(validate_single_change(value,ref),'TR09')
        for key,bad in [('epochs',2),('epochs',8),('actor_learning_rate',3e-6),('value_loss_coefficient',.1),('entropy_coefficient',.001),('gae_lambda',.99)]:
            changed=copy.deepcopy(value);changed['ppo'][key]=bad
            with self.assertRaises(ValueError):validate_single_change(changed,ref)
        for key,bad in [('parent_model_sha256','partial'),('target_decisions',262144)]:
            changed=copy.deepcopy(value);changed[key]=bad
            with self.assertRaises(ValueError):validate_single_change(changed,ref)


if __name__=='__main__':unittest.main()
