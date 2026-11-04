from pathlib import Path
import os
import tempfile
import unittest
from unittest.mock import patch
from ptcg_lab.archive_ack import receive_object
from ptcg_lab.checkpoint_refs import commit_checkpoint,read_index,protected_hashes,evict_archived_replica,checked_reference
from ptcg_lab.io import sha256


class CheckpointReferenceTests(unittest.TestCase):
    def test_unique_reference_hashed_once_per_operation_and_again_on_next_operation(self):
        self.commit(0,['rollout.pt']);self.commit(1,['rollout.pt'])
        from ptcg_lab import checkpoint_refs
        real=checkpoint_refs.sha256
        with patch.object(checkpoint_refs,'sha256',wraps=real) as digest:
            read_index(self.remote,'run',1)
            names=[str(call.args[0]) for call in digest.call_args_list]
            self.assertEqual(len(names),len(set(names)));first=len(names)
            read_index(self.remote,'run',1);self.assertEqual(digest.call_count,first*2)
        path=self.remote/'rollout.pt';old=path.stat();path.write_bytes(b'X'*old.st_size)
        os.utime(path,ns=(old.st_atime_ns,old.st_mtime_ns))
        # Even a same-size change with preserved mtime is detected at the next
        # boundary: no digest memo survives between operations/processes.
        with self.assertRaises(ValueError):read_index(self.remote,'run',1)
    def test_changed_identity_invalidates_memo_inside_one_writer_transaction(self):
        cache={};first=checked_reference(self.remote,'rollout.pt',cache)
        (self.remote/'rollout.pt').write_bytes(b'changed-longer-data')
        second=checked_reference(self.remote,'rollout.pt',cache)
        self.assertNotEqual(first['sha256'],second['sha256'])
    def test_corrupt_recovery_copy_cannot_publish_an_index(self):
        from ptcg_lab import checkpoint_refs
        real=checkpoint_refs._copy_immutable
        def corrupt(source,destination,*args):
            real(source,destination,*args)
            if str(destination).endswith('.pt'):destination.write_bytes(b'corrupted-copy')
        with patch.object(checkpoint_refs,'_copy_immutable',side_effect=corrupt):
            with self.assertRaises(ValueError):self.commit(0,['rollout.pt'])
        self.assertFalse((self.remote/'checkpoint-commits/index.json').exists())
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.remote=self.root/'remote';self.remote.mkdir()
        (self.remote/'checkpoint.pt').write_bytes(b'full-model-optimizer-rng');(self.remote/'prefix.json').write_bytes(b'prefix')
        (self.remote/'rollout.pt').write_bytes(b'pending-rollout')
        self.cursor={'version':1,'trained_decisions':10,'next_series_id':2}
    def tearDown(self):self.tmp.cleanup()
    def commit(self,generation,pending):
        return commit_checkpoint(self.remote,'run',1,generation,'checkpoint.pt','prefix.json',pending,self.cursor)
    def ack(self,relative):
        p=self.remote/relative;return receive_object(self.root/'archive','run',1,0,relative,sha256(p),p.stat().st_size,[p.read_bytes()])
    def test_previous_checkpoint_retains_pending_rollout_until_both_references_retire(self):
        self.commit(0,['rollout.pt']);ack=self.ack('rollout.pt');self.commit(1,[])
        self.assertFalse(evict_archived_replica(self.remote,'run',1,'rollout.pt',ack)['evicted'])
        self.commit(2,[])
        self.assertTrue(evict_archived_replica(self.remote,'run',1,'rollout.pt',ack)['evicted'])
        self.assertTrue((self.root/'archive/run/objects/rollout.pt').exists())
    def test_active_opponent_reference_prevents_eviction_after_checkpoint_retirement(self):
        self.commit(0,[]);ack=self.ack('rollout.pt')
        self.assertFalse(evict_archived_replica(self.remote,'run',1,'rollout.pt',ack,[ack['sha256']])['evicted'])
    def test_mutating_hot_checkpoint_does_not_modify_immutable_recovery_copy(self):
        c=self.commit(0,[]);(self.remote/'checkpoint.pt').write_bytes(b'new-hot-version')
        self.assertEqual((self.remote/c['checkpoint']['path']).read_bytes(),b'full-model-optimizer-rng')
        read_index(self.remote,'run',1)
    def test_failed_index_publication_keeps_prior_recoverable_index(self):
        old=self.commit(0,['rollout.pt'])
        with patch('ptcg_lab.checkpoint_refs.atomic_json',side_effect=OSError('simulated-full-disk')):
            with self.assertRaises(OSError):self.commit(1,[])
        self.assertEqual(read_index(self.remote,'run',1)['commits'][0]['commit_id'],old['commit_id'])
    def test_wrong_epoch_and_regressed_cursor_cannot_publish_or_reclaim(self):
        self.commit(0,[])
        with self.assertRaises(ValueError):read_index(self.remote,'run',2)
        self.cursor['trained_decisions']=9
        with self.assertRaisesRegex(ValueError,'regressed'):self.commit(1,[])
    def test_changed_referenced_bytes_fail_closed(self):
        self.commit(0,['rollout.pt']);(self.remote/'rollout.pt').write_bytes(b'changed')
        with self.assertRaises(ValueError):protected_hashes(self.remote,'run',1)
    def test_identical_recovery_blobs_are_not_duplicated_per_generation(self):
        self.commit(0,[]);self.commit(1,[]);self.commit(2,[])
        self.assertEqual(len(list((self.remote/'checkpoint-commits/objects').iterdir())),2)


if __name__=='__main__':unittest.main()
