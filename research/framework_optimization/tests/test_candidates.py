import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
import numpy as np
import torch

WORK=Path(__file__).resolve().parents[1];ROOT=WORK.parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(WORK/'source'))
sys.path.insert(0,str(WORK/'benchmarks'))
from fo.packed import Packed,evaluate_decisions
from fo.ppo import ppo_step
from fo.precision import Precision,parity
from fo.timing import Trace,union_seconds
from fo.archive_queue import TransferQueue,freeze_state
from fo.coordinator import Coordinator
from ptcg_lab.rl_core import evaluate_decisions as original,ppo_step as original_step
from ptcg_lab.archive_ack import receive_object
from ptcg_lab.archive_outbox import ArchiveOutbox
from common import real_fixture
from tests.test_batched_actor_prototype import Features,Model,request
from fo.conditional_batch import choose_requests

torch.set_num_threads(2)


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__();self.trunk=torch.nn.Linear(4,4);self.value_mlp=torch.nn.Linear(4,1)
    def forward_eval(self,x):
        # STOP identical across chunks, as required by the production model.
        h=self.trunk(x);return torch.cat((h[:,:3],h[:,3:4]*0),1),self.value_mlp(h)


def fixture():
    rng=np.random.default_rng(8);rows=rng.normal(size=(3,4)).astype(np.float32)
    return [dict(steps=[dict(rows=rows,mapping=np.array([[0,1,2,7],[3,4,5,7],[6,-1,-1,7]]),n=7,target=6),
                       dict(rows=rows[:1],mapping=np.array([[0,1,2,3]]),n=3,target=3)],
                 old_logp=-3.,old_value=0.,reward=1.,terminal=True),
            dict(steps=[dict(rows=rows[1:2],mapping=np.array([[0,1,-1,2]]),n=2,target=1)],
                 old_logp=-1.,old_value=0.,reward=-1.,terminal=True)]


class PackedTests(unittest.TestCase):
    def test_130_raw_candidates_and_padding_match_complete_support(self):
        class Large(torch.nn.Module):
            def __init__(self):super().__init__();self.scale=torch.nn.Parameter(torch.tensor(.7))
            def forward_eval(self,x):return x[:,:64]*self.scale,x[:,64:65]*self.scale
        rng=np.random.default_rng(31);decisions=[]
        for n,targets in [(130,[129,130]),(64,[63]),(17,[17]),(1,[0])]:
            steps=[]
            for target in targets:
                count=(n+62)//63;r=rng.normal(size=(count,65)).astype(np.float32);r[:,63]=.5
                mapping=np.full((count,64),-1,np.int32)
                for j in range(n):mapping[j//63,j%63]=j
                mapping[:,63]=n
                steps.append(dict(rows=r,mapping=mapping,n=n,target=target))
            decisions.append(dict(steps=steps))
        a=Large();b=copy.deepcopy(a)
        x=original(a,decisions,63);y=evaluate_decisions(b,decisions,63)
        for i,j in zip(x,y):torch.testing.assert_close(i,j,atol=1e-12,rtol=0)
        sum(t.sum() for t in x).backward();sum(t.sum() for t in y).backward()
        torch.testing.assert_close(a.scale.grad,b.scale.grad,atol=2e-6,rtol=0)

    def test_overflow_stop_variable_steps_outputs_and_gradients(self):
        a=Toy();b=copy.deepcopy(a);d=fixture()
        x=original(a,d,3);y=evaluate_decisions(b,d,3)
        for i,j in zip(x,y):torch.testing.assert_close(i,j,atol=1e-12,rtol=0)
        sum(t.sum() for t in x).backward();sum(t.sum() for t in y).backward()
        for i,j in zip(a.parameters(),b.parameters()):torch.testing.assert_close(i.grad,j.grad,atol=2e-6,rtol=0)

    def test_missing_raw_mapping_and_stop_disagreement_rejected(self):
        d=fixture();d[0]['steps'][0]['mapping'][2,0]=-1
        with self.assertRaises(ValueError):Packed.build(d,3)
        class Bad(Toy):
            def forward_eval(self,x):return self.trunk(x),self.value_mlp(self.trunk(x))
        for fn in (original,evaluate_decisions):
            with self.assertRaises(ValueError):fn(Bad(),fixture(),3)

    def test_one_step_optimizer_and_KL_gate_match(self):
        for route in (True,False):
            a=Toy();b=copy.deepcopy(a);d=fixture()
            with torch.no_grad():lp,_,_=original(a,d,3)
            for row,old in zip(d,lp):row['old_logp']=float(old)
            oa=torch.optim.Adam(a.parameters(),lr=1e-4);ob=torch.optim.Adam(b.parameters(),lr=1e-4)
            kw=dict(normalize_advantages=False,critic_updates_shared_trunk=route,record_gradient_components=True)
            ma=original_step(a,oa,d,np.array([1.,-1.],np.float32),np.array([.5,-.5],np.float32),3,**kw)
            mb=ppo_step(b,ob,d,np.array([1.,-1.],np.float32),np.array([.5,-.5],np.float32),3,evaluator=evaluate_decisions,**kw)
            self.assertEqual(ma['skipped_for_kl'],mb['skipped_for_kl'])
            for key in ma:self.assertAlmostEqual(ma[key],mb[key],places=6)
            for i,j in zip(a.parameters(),b.parameters()):torch.testing.assert_close(i,j,atol=2e-7,rtol=0)
            for i,j in zip(oa.state.values(),ob.state.values()):
                for key in i:torch.testing.assert_close(i[key],j[key],atol=2e-7,rtol=0)
            for row in d:row['old_logp']-=5
            before=copy.deepcopy(b.state_dict());step=ppo_step(b,ob,d,np.ones(2,np.float32),np.zeros(2,np.float32),3,
                evaluator=evaluate_decisions,target_kl=.01)
            self.assertTrue(step['skipped_for_kl'])
            for k,v in b.state_dict().items():self.assertTrue(torch.equal(v,before[k]))

    def test_real_fixture_forward_and_gradient_parity(self):
        a,features,d,_=real_fixture();b=copy.deepcopy(a)
        x=original(a,d,features.b.STOP);y=evaluate_decisions(b,d,features.b.STOP)
        result=parity(x,y);self.assertTrue(result['passed'])
        # Engineering derivative probe, no scientific PPO or optimizer update.
        sum(t.double().mean() for t in x).backward();sum(t.double().mean() for t in y).backward()
        error=0.
        for (name,i),(_,j) in zip(a.named_parameters(),b.named_parameters()):
            self.assertEqual(i.grad is None,j.grad is None,name)
            if i.grad is not None:
                error=max(error,float((i.grad-j.grad).abs().max()))
                torch.testing.assert_close(i.grad,j.grad,atol=2e-6,rtol=1e-5,msg=name)
        (WORK/'reports/real_gradient_parity.json').write_text(json.dumps(dict(result,gradient_max_error=error,
            gradient_atol=2e-6,gradient_rtol=1e-5,optimizer_steps=0,new_scientific_decisions=0),indent=2))


class RecoveryTests(unittest.TestCase):
    def test_frozen_state_preserves_storage_alias_stride_offset_and_model_metadata(self):
        from ptcg_lab.bounded_checkpoint_audit import exact_state
        x=torch.arange(20.).reshape(4,5);state={'a':x,'b':x[1:3,::2],'model':Toy().state_dict()}
        frozen=freeze_state(state);exact_state(state,frozen)
        self.assertNotEqual(x.untyped_storage().data_ptr(),frozen['a'].untyped_storage().data_ptr())
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'state.pt';torch.save(frozen,path)
            exact_state(frozen,torch.load(path,weights_only=False))

    def test_hardlink_publication_survives_atomic_hot_checkpoint_replacement(self):
        from fo.checkpoint_refs import commit_checkpoint,read_index
        from ptcg_lab.tensor_io import atomic_torch_save
        from unittest.mock import patch
        from ptcg_lab.io import sha256
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'prefix.json').write_text('{}')
            atomic_torch_save(root/'checkpoint.pt',{'tensor':torch.tensor([1.])})
            first=commit_checkpoint(root,'run',1,0,'checkpoint.pt','prefix.json',[],
                dict(version=0,trained_decisions=0,next_series_id=0),immutable_hardlink=True)
            stored=root/first['checkpoint']['path'];first_hash=sha256(stored)
            atomic_torch_save(root/'checkpoint.pt',{'tensor':torch.tensor([2.])})
            second=commit_checkpoint(root,'run',1,1,'checkpoint.pt','prefix.json',[],
                dict(version=1,trained_decisions=1,next_series_id=1),immutable_hardlink=True)
            self.assertEqual(sha256(stored),first_hash)
            self.assertEqual(len(read_index(root,'run',1)['commits']),2)
            self.assertTrue(torch.equal(torch.load(stored,weights_only=True)['tensor'],torch.tensor([1.])))
            # A writer violating atomic replacement is detected by full hashes.
            (root/'checkpoint.pt').write_bytes(b'corruption')
            with self.assertRaises(ValueError):read_index(root,'run',1)

    def test_hardlink_unsupported_falls_back_to_verified_copy(self):
        import errno
        from unittest.mock import patch
        from fo.immutable_link import publish_link
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';source.write_bytes(b'data')
            with patch('fo.immutable_link.os.link',side_effect=OSError(errno.EXDEV,'cross-device')):
                self.assertFalse(publish_link(source,root/'target',root))
            from fo.checkpoint_refs import commit_checkpoint,read_index
            (root/'prefix.json').write_text('{}')
            with patch('fo.immutable_link.publish_link',return_value=False):
                receipt=commit_checkpoint(root,'run',1,0,'source','prefix.json',[],
                    dict(version=0,trained_decisions=0,next_series_id=0),immutable_hardlink=True)
            self.assertEqual(read_index(root,'run',1)['commits'][0],receipt)
            self.assertEqual((root/receipt['checkpoint']['path']).read_bytes(),b'data')

    def test_queue_capacity_does_not_drop_unsubmitted_rows(self):
        import threading
        event=threading.Event()
        row={'run_id':'run','lease_epoch':1,'sequence':0,'object':'model.pt','bytes':8,'sha256':'a'*64}
        def fail_after_release(r,deadline):event.wait(2);raise ConnectionError('injected disconnect')
        q=TransferQueue('run',1,fail_after_release,lambda ack:None,max_objects=1,max_bytes=8,deadline=time.time()+5)
        try:
            self.assertEqual(q.submit_batch([row,dict(row,sequence=1)]),1)
            self.assertEqual(list(q.pending),[0])
        finally:
            event.set()
            with self.assertRaises(ConnectionError):q.close()

    def test_freeze_detaches_model_optimizer_rng_pending_and_precision(self):
        model=Toy();opt=torch.optim.Adam(model.parameters());sum(p.sum() for p in model.parameters()).backward();opt.step()
        precision=Precision();state={'model':model.state_dict(),'optimizer':opt.state_dict(),'rng':torch.get_rng_state(),
            'pending':fixture(),'league':{'cursor':4},'precision':precision.state_dict()}
        frozen=freeze_state(state)
        with torch.no_grad():next(model.parameters()).add_(1)
        state['pending'][0]['steps'][0]['rows'][:]=0
        self.assertFalse(torch.equal(next(model.parameters()),next(iter(frozen['model'].values()))))
        self.assertTrue(frozen['pending'][0]['steps'][0]['rows'].any())
        restored=Toy();restored.load_state_dict(frozen['model']);opt2=torch.optim.Adam(restored.parameters());opt2.load_state_dict(frozen['optimizer'])
        Precision().load_state_dict(frozen['precision'])
        with self.assertRaises(ValueError):Precision().load_state_dict(dict(frozen['precision'],mode='fp16'))

    def test_partial_transfer_lost_ack_duplicate_ack_and_backpressure(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'run';source.mkdir();(source/'model.pt').write_bytes(b'12345678')
            outbox=ArchiveOutbox(source,'run',1);row=outbox.queue('model.pt')
            def partial(r,deadline):
                return receive_object(root/'archive','run',1,0,r['object'],r['sha256'],r['bytes'],[b'12'])
            queue=TransferQueue('run',1,partial,outbox.accept_ack,deadline=time.time()+20)
            queue.submit_batch([row])
            with self.assertRaises(ValueError):queue.close()
            self.assertEqual(outbox.pending()['pending_objects'],1)
            self.assertFalse((root/'archive/run/acks/00000000.json').exists())
            def transfer(r,deadline):
                return receive_object(root/'archive','run',1,0,r['object'],r['sha256'],r['bytes'],[b'12345678'])
            # Receive completed but sender crashed before ACK: retry verifies
            # the same complete object and cannot create a second logical row.
            transfer(row,time.time()+20)
            with TransferQueue('run',1,transfer,outbox.accept_ack,max_objects=1,max_bytes=8,deadline=time.time()+20) as queue:
                self.assertEqual(queue.submit_batch([row]),1)
            self.assertEqual(outbox.pending()['pending_objects'],0)
            ack=transfer(row,time.time()+20);self.assertTrue(outbox.accept_ack(ack)['replayed'])
            bad=dict(ack,lease_epoch=2)
            with self.assertRaises(ValueError):outbox.accept_ack(bad)
            self.assertTrue((source/'model.pt').exists())  # no eviction method


class CoordinatorTests(unittest.TestCase):
    class Tracker:
        def __init__(self,deck):pass
        def update(self,obs):pass
    class Opponent:
        deck=[];identity='fixed'
        def close(self):pass
    class Engine:
        def __init__(self,seat):self.seat=seat;self.i=0;self.closed=False
        def send(self,message):pass
        def read(self):
            self.i+=1
            if self.i<=2:
                obs=request('x',3,1,1)['observation']
                obs['current']={'round':self.i,'yourIndex':self.seat,'players':[{'win':1},{'win':0}],'draw':0}
                return {'event':'decision','observation':obs,'seat':self.seat}
            return {'event':'series_end','statuses':['DONE','DONE'],'scores':[2,0,0],'rewards':[1,-1]}
        def close(self):self.closed=True

    def test_one_two_three_workers_full_bo3_rng_pending_and_version_barrier(self):
        for workers in (1,2,3):
            f=Features();f.Tracker=self.Tracker;c=Coordinator(workers=workers,version=7,next_series_id=10)
            result=c.collect_wave(Model(),f,[],lambda sid,seat:(self.Engine(seat),self.Opponent(),{'opponent_id':'fixed'}))
            self.assertEqual(result['complete_BO3'],workers)
            self.assertEqual([x['series_id'] for x in c.pending],list(range(10,10+workers)))
            for entry in c.pending:
                self.assertEqual(len(entry['trajectories']),2)
                self.assertTrue(all(t['decisions'][-1]['terminal'] for t in entry['trajectories']))
            with self.assertRaises(ValueError):c.advance_policy(8)
            state=c.state_dict();d=Coordinator(workers=workers);d.load_state_dict(state)
            self.assertEqual(d.next_series_id,c.next_series_id)
            self.assertTrue(all(torch.equal(a.get_state(),b.get_state()) for a,b in zip(c.rng,d.rng)))
            ids=[x['series_id'] for x in d.pending];count=sum(len(t['decisions']) for x in d.pending for t in x['trajectories'])
            d.acknowledge_pending(ids)
            with self.assertRaises(ValueError):d.advance_policy(8)
            covered=d.advance_policy(8,dict(version=8,series_ids=ids,decisions=count,
                steps=[dict(epoch=0,offset=0,batch=count,skipped_for_kl=False)]))
            self.assertEqual(covered,count)

    def test_interrupted_native_bo3_excluded_and_cursor_not_reused(self):
        c=Coordinator(workers=2);f=Features();f.Tracker=self.Tracker
        def guard():raise RuntimeError('hard boundary')
        with self.assertRaises(RuntimeError):c.collect_wave(Model(),f,[],lambda sid,seat:(self.Engine(seat),self.Opponent(),{}),guard)
        self.assertEqual([x['series_id'] for x in c.excluded],[0,1]);self.assertEqual(c.next_series_id,2)
        self.assertEqual(c.pending,[]);self.assertFalse(c.running)


class TimingTests(unittest.TestCase):
    def test_overlapping_nested_spans_are_union_not_sum(self):
        self.assertEqual(union_seconds([(0,5),(1,2),(3,8),(10,11)]),9)
        t=Trace(True)
        with t.span('outer'):
            with t.span('inner'):pass
        r=t.report();self.assertLessEqual(r['union_covered_seconds'],r['wall_seconds'])
        self.assertGreater(sum(r['phases_union_seconds'].values()),r['union_covered_seconds'])


if __name__=='__main__':unittest.main()
