from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ptcg_lab.archive_outbox import ArchiveOutbox,receive_one,drain_once
from ptcg_lab.archive_ack import receive_object
from ptcg_lab.archive_prefix import compact_record,canonical_hash,fixed_statistics
from ptcg_lab.checkpoint_refs import commit_checkpoint,evict_archived_replicas
from ptcg_lab.io import atomic_json,read_json,sha256


class Stream:
    def __init__(self,outbox):self.outbox=outbox;self.reads=0;self.lose_ack=False;self.corrupt=False
    def request(self,operation,data,**kwargs):
        if operation=='archive_pending':return self.outbox.pending(data['limit'])
        if operation=='archive_chunk':
            self.reads+=1;result=self.outbox.chunk(data['object_sequence'],data['offset'],data['count'])
            if self.corrupt:result['raw_sha256']='0'*64
            return result
        if operation=='archive_ack':
            if self.lose_ack:self.lose_ack=False;raise TimeoutError('injected-lost-ack')
            return self.outbox.accept_ack(data['ack'])
        raise AssertionError(operation)


class ArchiveOutboxTests(unittest.TestCase):
    def test_chunk_reads_only_its_validated_row_and_rejects_identity_tampering(self):
        with patch.object(self.box,'records',side_effect=AssertionError('full scan during immutable chunk')):
            self.assertGreater(self.box.chunk(0,0,1024)['raw_bytes'],0)
        path=self.root/'archive-outbox/00000000.json';row=read_json(path)
        for field,value in [('run_id','other'),('lease_epoch',2),('sequence',1),('sha256','x'*64),('object','../foreign')]:
            atomic_json(path,{**row,field:value})
            with self.assertRaises(ValueError):self.box.chunk(0,0,1024)
        atomic_json(path,row)
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.project=Path(self.tmp.name);self.root=self.project/'run';self.root.mkdir()
        path=self.root/'rollouts/v0000-s000000.pt';path.parent.mkdir();path.write_bytes(b'trajectory-padding\0'*150000)
        self.summary={'statuses':['DONE','DONE'],'series_id':0,'trajectory_path':path.relative_to(self.root).as_posix(),
            'trajectory_sha256':sha256(path),'policy_version':'0','learner_seat':0,'rewards':[-1,1],
            'assignment':{'fixed':True,'opponent_id':'frozen'}}
        atomic_json(self.root/'series/000000.json',self.summary)
        self.box=ArchiveOutbox(self.root,'run',1);self.rows=self.box.queue_series(self.summary);self.stream=Stream(self.box)
        self.archive=self.project/'archive'
    def tearDown(self):self.tmp.cleanup()
    def test_complete_binary_multi_chunk_roundtrip_and_backpressure(self):
        self.assertTrue(self.box.backpressured(1)[0])
        while self.box.pending()['objects']:drain_once(self.stream,self.archive,'run',1)
        self.assertFalse(self.box.backpressured(1)[0]);self.assertGreater(self.stream.reads,1)
        original=self.root/self.summary['trajectory_path'];returned=self.archive/'run/objects'/self.summary['trajectory_path']
        self.assertEqual(sha256(original),sha256(returned))
        self.assertEqual(self.box.queue_series(self.summary),self.rows)
    def test_lost_ack_reuses_fsynced_local_file_without_reading_remote_again(self):
        self.stream.lose_ack=True
        with self.assertRaises(TimeoutError):receive_one(self.stream,self.archive,self.rows[0])
        before=self.stream.reads;ack=receive_one(self.stream,self.archive,self.rows[0])
        self.assertEqual(before,self.stream.reads);self.assertTrue(self.box.accept_ack(ack)['replayed'])
        with self.assertRaises(ValueError):self.box.accept_ack({**ack,'lease_epoch':2})
    def test_wire_corruption_never_acks_or_publishes(self):
        self.stream.corrupt=True
        with self.assertRaises(ValueError):receive_one(self.stream,self.archive,self.rows[0])
        self.assertFalse((self.root/'archive-acks/00000000.json').exists())
        self.assertFalse((self.archive/'run/objects'/self.rows[0]['object']).exists())
    def test_local_disk_failure_keeps_remote_unacknowledged(self):
        with patch('ptcg_lab.archive_ack.shutil.disk_usage') as usage:
            usage.return_value.free=0
            with self.assertRaises(OSError):receive_one(self.stream,self.archive,self.rows[0])
        self.assertEqual(self.box.pending()['pending_objects'],2)
    def test_existing_raw_archive_can_ack_only_identical_original_bytes(self):
        source=self.root/self.rows[0]['object'];ack=receive_one(self.stream,self.archive,self.rows[0],source)
        self.assertEqual(self.stream.reads,0);self.assertEqual(ack['sha256'],sha256(source))
        with self.assertRaises(ValueError):receive_one(self.stream,self.project/'other',self.rows[1],source)
    def test_verified_compressed_original_avoids_wire_and_rejects_wrong_hash(self):
        from ptcg_lab.archive_compressed import receive_compressed_object,verify_archive
        row=self.rows[0];data=(self.root/row['object']).read_bytes();prior=self.project/'old-archive'
        ack=receive_compressed_object(prior,'old',2,0,row['object'],row['sha256'],row['bytes'],[data])
        source={'archive_root':prior,'ack':ack}
        received=receive_one(self.stream,self.archive,row,source,storage_codec='gzip')
        self.assertEqual(self.stream.reads,0);self.assertTrue(verify_archive(self.archive,received))
        with self.assertRaises(ValueError):receive_one(self.stream,self.project/'bad',self.rows[1],source,storage_codec='gzip')
    def test_unfinished_or_changed_artifact_and_oversized_read_fail(self):
        with self.assertRaises(ValueError):self.box.queue_series({**self.summary,'statuses':['ACTIVE','ACTIVE']})
        with self.assertRaises(ValueError):self.box.chunk(0,0,2*1024**2)
        (self.root/self.rows[0]['object']).write_bytes(b'changed')
        with self.assertRaises(ValueError):self.box.chunk(0,0,1024)
        with self.assertRaises(ValueError):self.box.queue(self.rows[0]['object'])
    def test_previous_recovery_and_active_hashes_block_eviction(self):
        for row in self.rows:receive_one(self.stream,self.archive,row)
        (self.root/'checkpoint.pt').write_bytes(b'full-checkpoint');(self.root/'prefix.json').write_bytes(b'prefix')
        cursor={'version':1,'trained_decisions':10,'next_series_id':1};paths=[r['object'] for r in self.rows]
        commit_checkpoint(self.root,'run',1,0,'checkpoint.pt','prefix.json',paths,cursor)
        row=compact_record(self.summary,sha256(self.root/'series/000000.json'));state={'archived':[row]}
        self.assertTrue(all(not x['evicted'] for x in self.box.evict_consumed(state)))
        commit_checkpoint(self.root,'run',1,1,'checkpoint.pt','prefix.json',[],cursor)
        self.assertTrue(all(not x['evicted'] for x in self.box.evict_consumed(state)))
        commit_checkpoint(self.root,'run',1,2,'checkpoint.pt','prefix.json',[],cursor)
        blocked=self.box.evict_consumed(state,[self.rows[0]['sha256']])
        self.assertFalse(blocked[0]['evicted']);self.assertTrue(blocked[1]['evicted'])
        result=self.box.evict_consumed(state);self.assertTrue(result[0]['evicted'])
        self.assertTrue((self.archive/'run/objects'/self.rows[0]['object']).exists())
    def test_index_changed_during_batch_validation_prevents_deletion(self):
        (self.root/'checkpoint.pt').write_bytes(b'checkpoint');(self.root/'prefix.json').write_bytes(b'prefix')
        commit_checkpoint(self.root,'run',1,0,'checkpoint.pt','prefix.json',[],{'version':1,'trained_decisions':1,'next_series_id':1})
        ack=receive_one(self.stream,self.archive,self.rows[0])
        from ptcg_lab import checkpoint_refs
        original=checkpoint_refs.protected_hashes
        def mutate(*args,**kwargs):
            result=original(*args,**kwargs);p=self.root/'checkpoint-commits/index.json';atomic_json(p,read_json(p));return result
        with patch.object(checkpoint_refs,'protected_hashes',side_effect=mutate):
            with self.assertRaises(RuntimeError):evict_archived_replicas(self.root,'run',1,[ack])
        self.assertTrue((self.root/self.rows[0]['object']).exists())
