import tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
from ptcg_lab.archive_outbox import ArchiveOutbox,receive_one
from ptcg_lab.archive_barrier import retire_before_sampling
from ptcg_lab.archive_prefix import compact_record
from ptcg_lab.checkpoint_refs import commit_checkpoint,read_index
from ptcg_lab.io import atomic_json,sha256,read_json
from ptcg_lab.training_control import ControlledTrainingStop


class Wire:
    def __init__(self,box):self.box=box
    def request(self,op,body,**kwargs):
        if op=='archive_chunk':return self.box.chunk(body['object_sequence'],body['offset'],body['count'])
        if op=='archive_ack':return self.box.accept_ack(body['ack'])
        raise AssertionError(op)


class ArchiveBarrierTests(unittest.TestCase):
    def fixture(self,root):
        job=root/'job';job.mkdir();trajectory=job/'rollouts/v0000-s000000.pt';trajectory.parent.mkdir();trajectory.write_bytes(b'complete BO3 payload'*100)
        summary={'series_id':0,'policy_version':'0','statuses':['DONE','DONE'],'learner_seat':0,'rewards':[1,-1],
            'assignment':{'fixed':True,'opponent_id':'fixed'},'trajectory_path':'rollouts/v0000-s000000.pt','trajectory_sha256':sha256(trajectory)}
        atomic_json(job/'series/000000.json',summary);(job/'checkpoint.pt').write_bytes(b'full-checkpoint');(job/'prefix.json').write_bytes(b'prefix')
        cursor={'version':1,'trained_decisions':3,'next_series_id':1};pair=['rollouts/v0000-s000000.pt','series/000000.json']
        commit_checkpoint(job,'job',1,0,'checkpoint.pt','prefix.json',pair,cursor)
        point=commit_checkpoint(job,'job',1,1,'checkpoint.pt','prefix.json',pair,cursor)
        box=ArchiveOutbox(job,'job',1);rows=box.queue_series(summary)
        state={'archived':[compact_record(summary,sha256(job/'series/000000.json'))]}
        return job,box,point,cursor,rows,state
    def test_delayed_ack_requires_two_rotations_and_no_early_eviction(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);job,box,point,cursor,rows,state=self.fixture(root);events=[];rotations=[];generation=1
            def ack_after_wait(_):
                self.assertEqual(rotations,[])
                for row in rows:receive_one(Wire(box),root/'archive',row,storage_codec='gzip')
            def save():
                nonlocal generation
                generation+=1;commit_checkpoint(job,'job',1,generation,'checkpoint.pt','prefix.json',[],cursor)
                result=box.evict_consumed(state);rotations.append([r['evicted'] for r in result])
            with patch('ptcg_lab.archive_barrier.time.sleep',side_effect=ack_after_wait):
                result=retire_before_sampling(box,point['pending'],cursor,save,time.time()+60,lambda:1,lambda *a,**k:events.append(a[0]))
            self.assertEqual(result['post_ACK_writer_rotations'],2);self.assertTrue(all(not v for v in rotations[0]));self.assertTrue(all(rotations[1]))
            self.assertTrue(result['ACK_wait_samples']>=1);self.assertTrue(all(not (job/r['object']).exists() for r in rows))
            self.assertEqual(len(read_index(job,'job',1)['commits']),2)
    def test_original_deadline_and_bad_ack_never_advance_or_delete(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);job,box,point,cursor,rows,state=self.fixture(root);saves=[]
            with self.assertRaises(ControlledTrainingStop):retire_before_sampling(box,point['pending'],cursor,lambda:saves.append(1),time.time(),lambda:1,lambda *a,**k:None)
            self.assertEqual(saves,[]);self.assertTrue(all((job/r['object']).exists() for r in rows))
            atomic_json(job/'archive-acks/00000000.json',{'foreign':True})
            with self.assertRaises((KeyError,ValueError)):retire_before_sampling(box,point['pending'],cursor,lambda:saves.append(1),time.time()+60,lambda:1,lambda *a,**k:None)
            self.assertEqual(saves,[])
    def test_callback_that_leaves_references_pinned_fails_before_new_sampling(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);job,box,point,cursor,rows,state=self.fixture(root)
            for row in rows:receive_one(Wire(box),root/'archive',row)
            with self.assertRaises(RuntimeError):retire_before_sampling(box,point['pending'],cursor,lambda:None,time.time()+60,lambda:1,lambda *a,**k:None)
            self.assertTrue(all((job/r['object']).exists() for r in rows))
