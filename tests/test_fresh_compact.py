from pathlib import Path
import random
import tempfile
import unittest
import numpy as np
import torch
from ptcg_lab.fresh_compact import initialize
from ptcg_lab.io import read_json
from ptcg_lab.rl_resume import resume_journal
from ptcg_lab.training_checkpoint import load_training_checkpoint,initialize_recovery


class FreshCompactTests(unittest.TestCase):
    def config(self):return {'hypothesis_id':'TR05','parent_model_sha256':'1'*64,'seed':1,'target_decisions':32768,
        'ppo':{'critic_updates_shared_trunk':False},'league':{},'source_manifest_sha256':'2'*64}
    def test_zero_history_full_checkpoint_roundtrip_preserves_rng_and_recovery(self):
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/'fresh-run';output.mkdir();model=torch.nn.Linear(2,1);optimizer=torch.optim.Adam(model.parameters())
            a=random.getstate();b=np.random.get_state();c=torch.get_rng_state().clone()
            spec=initialize(output,self.config(),model,optimizer)
            self.assertEqual(random.getstate(),a);self.assertTrue(np.array_equal(np.random.get_state()[1],b[1]));self.assertTrue(torch.equal(torch.get_rng_state(),c))
            loaded=load_training_checkpoint(output,spec,3);journal=resume_journal(output,loaded['saved'],spec,loaded['digest'])
            self.assertEqual(journal['decisions'],[]);self.assertEqual(journal['fixed_statistics'],{})
            point=initialize_recovery(output,loaded['saved'],spec,journal['archive_state'],3)
            self.assertEqual(point['pending'],[]);self.assertEqual(point['cursor']['trained_decisions'],0)
            self.assertEqual(initialize(output,self.config(),model,optimizer),spec)
            self.assertEqual(read_json(output/'archive-prefix.json')['origin_source_manifest_sha256'],'2'*64)
    def test_fresh_origin_cannot_adopt_partial_weight_or_changed_settings(self):
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/'fresh-run';output.mkdir();model=torch.nn.Linear(2,1);optimizer=torch.optim.Adam(model.parameters())
            spec=initialize(output,self.config(),model,optimizer);changed=self.config();changed['ppo']['critic_updates_shared_trunk']=True
            with self.assertRaises(ValueError):initialize(output,changed,model,optimizer)
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/'fresh-run';output.mkdir();(output/'checkpoint.pt').write_bytes(b'unregistered')
            with self.assertRaises(ValueError):initialize(output,self.config(),model,optimizer)
