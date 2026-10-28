import os
import random
import unittest
from unittest.mock import patch
import numpy as np
import torch
import tests.test_training_checkpoint as fixtures
from ptcg_lab.io import sha256
from ptcg_lab.rl_resume import advance_resume_state
from ptcg_lab.training_checkpoint import publish_training_checkpoint
from ptcg_lab.checkpoint_refs import protected_hashes


class CheckpointTransactionTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.TrainingCheckpointTests('test_two_published_points_pin_both_trajectory_and_complete_summary')
        self.fixture.setUp();self.fixture.initial()
        self.root=self.fixture.root;self.shard=self.root/'rollouts/v0001-s000000.pt'

    def tearDown(self):self.fixture.tearDown()

    def advance(self,verified):
        f=self.fixture
        return advance_resume_state(self.root,f.state,f.prefix,f.spec,
            {k:f.saved[k] for k in ['version','trained_decisions','next_series_id']},self.root.name,1,_verified=verified)

    def test_one_safe_point_reuses_only_duplicate_hashes_and_keeps_payload_and_rng(self):
        f=self.fixture;counts=[]
        with patch('ptcg_lab.checkpoint_refs.sha256',wraps=sha256) as hashing:
            state=self.advance(None);value={**f.value(1),'archive_state':state}
            plain=publish_training_checkpoint(self.root,value,f.spec,1)['recovery']
            protected_hashes(self.root,self.root.name,1)
            counts.append(sum(str(c.args[0])==str(self.shard) for c in hashing.call_args_list))
        before=(random.getstate(),np.random.get_state(),torch.get_rng_state().clone())
        verified={}
        with patch('ptcg_lab.checkpoint_refs.sha256',wraps=sha256) as hashing:
            self.assertEqual(self.advance(verified),state)
            cached=publish_training_checkpoint(self.root,value,f.spec,1,_verified=verified)['recovery']
            actual=protected_hashes(self.root,self.root.name,1,_verified=verified)
            counts.append(sum(str(c.args[0])==str(self.shard) for c in hashing.call_args_list))
        self.assertGreater(counts[0],1);self.assertEqual(counts[1],1)
        self.assertEqual(cached['checkpoint'],plain['checkpoint'])
        self.assertEqual(actual,protected_hashes(self.root,self.root.name,1))
        self.assertEqual(before[0],random.getstate());self.assertTrue(np.array_equal(before[1][1],np.random.get_state()[1]))
        self.assertTrue(torch.equal(before[2],torch.get_rng_state()))

    def test_changed_inode_metadata_between_journal_and_publish_fails_closed(self):
        f=self.fixture;verified={};self.advance(verified)
        old=self.shard.stat();data=self.shard.read_bytes();self.shard.write_bytes(b'x'*len(data))
        os.utime(self.shard,ns=(old.st_atime_ns,old.st_mtime_ns+10000000))
        with self.assertRaises(ValueError):publish_training_checkpoint(self.root,f.value(1),f.spec,1,_verified=verified)

    def test_next_safe_point_rehashes_every_unique_reference(self):
        for _ in range(2):
            with patch('ptcg_lab.checkpoint_refs.sha256',wraps=sha256) as hashing:
                protected_hashes(self.root,self.root.name,1,_verified={})
                self.assertEqual(sum(str(c.args[0])==str(self.shard) for c in hashing.call_args_list),1)


if __name__=='__main__':unittest.main()
