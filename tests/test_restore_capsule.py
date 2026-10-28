import copy
import hashlib
import unittest
import zlib
import tempfile
from pathlib import Path
from ptcg_lab.restore_capsule import unpack,restore_capacity,reference
from ptcg_lab.server_trial import validate_restore_repair
from ptcg_lab.io import atomic_json


class RestoreCapsuleTests(unittest.TestCase):
    def test_connect_repair_keeps_spent_seconds_in_original_total(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);atomic_json(root/'runs/old/attempts.json',[{'seconds':12.39}])
            config={'compact_restore_repair_parent_run_ids':['old'],'prior_compact_restore_seconds':12.39}
            validate_restore_repair(root,config,3687,now=100)
            with self.assertRaises(RuntimeError):validate_restore_repair(root,config,3700,now=100)
            with self.assertRaises(RuntimeError):validate_restore_repair(root,{**config,'prior_compact_restore_seconds':0},3687,now=100)
    def test_component_exact_external_hash_and_no_trailing_data(self):
        raw=b'original optimizer and RNG bytes'*1000;packed=zlib.compress(raw,1)
        record={'raw_bytes':len(raw),'raw_sha256':hashlib.sha256(raw).hexdigest(),
            'packed_bytes':len(packed),'packed_sha256':hashlib.sha256(packed).hexdigest()}
        self.assertEqual(unpack(packed,record),raw)
        bad=packed+b'second stream';other={**record,'packed_bytes':len(bad),'packed_sha256':hashlib.sha256(bad).hexdigest()}
        with self.assertRaises(ValueError):unpack(bad,other)
        with self.assertRaises(ValueError):unpack(packed,{**record,'raw_sha256':'0'*64})
    def capsule(self):
        rows=[{'path':x,'raw_bytes':2} for x in ['checkpoint.pt','archive-prefix.json','league.json']+
            [f'rollouts/v0030-s{x:06d}.pt' for x in range(35)]+[f'series/{x:06d}.json' for x in range(35)]]
        return {'components':rows,'history':{'members':[{'raw_bytes':3} for _ in range(31)],'raw_bytes':93},
                'expanded_bytes':239,'expected_update_input':{'decisions':5430}}
    def test_capacity_includes_full_history_tail_and_checkpoint_headroom(self):
        capsule=self.capsule();self.assertEqual(restore_capacity(capsule),239+384*2**20)
        bad=copy.deepcopy(capsule);bad['expanded_bytes']=64*2**20
        with self.assertRaises(ValueError):restore_capacity(bad)
        bad=copy.deepcopy(capsule);bad['components'][-1]['path']='control/unreviewed.py'
        with self.assertRaises(ValueError):restore_capacity(bad)
        bad=copy.deepcopy(capsule);bad['history']['members'].pop()
        with self.assertRaises(ValueError):restore_capacity(bad)
    def test_update_input_fingerprint_changes_if_reward_or_order_changes(self):
        decisions=[{'series_id':4,'reward':0.,'old_value':.2,'terminal':False,'old_logp':-.1},
                   {'series_id':5,'reward':1.,'old_value':-.1,'terminal':True,'old_logp':-.2}]
        before=reference(decisions,{'gamma':1.,'gae_lambda':.99})
        changed=copy.deepcopy(decisions);changed[-1]['reward']=-1.
        self.assertNotEqual(before['returns_sha256'],reference(changed,{'gamma':1.,'gae_lambda':.99})['returns_sha256'])
        self.assertNotEqual(before['old_logp_sha256'],reference(list(reversed(decisions)),{'gamma':1.,'gae_lambda':.99})['old_logp_sha256'])
