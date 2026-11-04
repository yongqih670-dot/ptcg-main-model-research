from types import SimpleNamespace
import unittest
import numpy as np
import torch
from research.prototypes.batched_actor_v1.conditional_batch import choose_requests
from ptcg_lab.rl_core import choose_decision


class Features:
    b=SimpleNamespace(STOP=3)
    def field(self,row,name):return row[-1:]
    def make(self,obs,tracker,forced_run,picked):
        sel=obs['select'];n=len(sel['option']);rows=[];maps=[]
        for start in range(0,max(1,n),3):
            row=np.full(7,-1e9,np.float32);mapping=np.full(4,-1,np.int32)
            for j,k in enumerate(range(start,min(start+3,n))):
                mapping[j]=k;row[j]=obs['scores'][k] if k not in picked else -1e9
            row[3]=obs.get('STOP_score',0.) if len(picked)>=sel['minCount'] else -1e9
            mapping[3]=n;row[4]=obs['value'];row[5]=obs['tag'];row[6]=obs.get('oracle',0.)
            rows.append(row);maps.append(mapping)
        return np.stack(rows),np.stack(maps)


class Model(torch.nn.Module):
    def __init__(self):super().__init__();self.parameter=torch.nn.Parameter(torch.zeros(1));self.calls=0;self.eval()
    def forward_eval(self,x):self.calls+=1;return x[:,:4],x[:,4:5]


def request(key,n,lo,hi,seed=12):
    return {'key':key,'observation':{'select':{'option':list(range(n)),'minCount':lo,'maxCount':hi},
        'scores':[float(i+1) for i in range(n)],'value':float(n)/10,'tag':float(seed)},
        'tracker':None,'forced_run':0,'rng':torch.Generator().manual_seed(seed)}


class BatchedPrototypeTests(unittest.TestCase):
    def test_mixed_overflow_multiselect_stop_and_forced_equal_serial(self):
        requests=[request('overflow',7,2,4),request('single',3,1,1),request('forced',2,2,2),request('stop',5,0,3)]
        requests[-1]['observation']['STOP_score']=20.
        model=Model();features=Features();actual,stats=choose_requests(model,features,requests,greedy=True)
        for r in requests:
            action,decision=choose_decision(Model(),features,r['observation'],None,0,True)
            self.assertEqual(actual[r['key']][0],action)
            if decision is None:self.assertIsNone(actual[r['key']][1]);continue
            self.assertEqual(actual[r['key']][1]['old_value'],decision['old_value'])
            self.assertAlmostEqual(actual[r['key']][1]['old_logp'],decision['old_logp'])
            self.assertEqual([x['target'] for x in actual[r['key']][1]['steps']],[x['target'] for x in decision['steps']])
        self.assertLess(stats['forward_calls'],sum(len(x[1]['steps']) for x in actual.values() if x[1]))
        self.assertEqual(actual['overflow'][0],[6,5,4,3]);self.assertEqual(actual['stop'][0],[])

    def test_streams_independent_of_batch_order_and_other_environment(self):
        first=[request('a',7,1,4,4),request('b',4,0,3,9)]
        second=[request('b',4,0,3,9),request('a',7,1,4,4)]
        a,_=choose_requests(Model(),Features(),first);b,_=choose_requests(Model(),Features(),second)
        for i,j in [(0,1),(1,0)]:
            key=first[i]['key'];self.assertEqual(a[key][0],b[key][0]);self.assertEqual(a[key][1]['old_logp'],b[key][1]['old_logp'])
            self.assertTrue(torch.equal(first[i]['rng'].get_state(),second[j]['rng'].get_state()))
        for request_ in first:
            single=request(request_['key'],len(request_['observation']['scores']),request_['observation']['select']['minCount'],request_['observation']['select']['maxCount'],4 if request_['key']=='a' else 9)
            result,_=choose_requests(Model(),Features(),[single]);self.assertEqual(result[request_['key']][0],a[request_['key']][0])

    def test_sampled_actions_and_logprob_equal_serial_with_same_environment_rng(self):
        requests=[request('a',7,1,4,4),request('b',4,0,3,9)]
        global_before=torch.get_rng_state().clone()
        expected={};states={}
        for r in requests:
            with torch.random.fork_rng(devices=[]):
                torch.set_rng_state(r['rng'].get_state())
                expected[r['key']]=choose_decision(Model(),Features(),r['observation'],None,0,False)
                states[r['key']]=torch.get_rng_state().clone()
        actual,_=choose_requests(Model(),Features(),requests)
        self.assertTrue(torch.equal(torch.get_rng_state(),global_before))
        for r in requests:
            self.assertEqual(actual[r['key']][0],expected[r['key']][0])
            self.assertEqual(actual[r['key']][1]['old_logp'],expected[r['key']][1]['old_logp'])
            self.assertTrue(torch.equal(r['rng'].get_state(),states[r['key']]))

    def test_oracle_and_capacity_fail_before_any_forward_and_stream_advance(self):
        for limit,oracle in [(1,0.),(128,1.)]:
            r=request('overflow',7,1,2);r['observation']['oracle']=oracle;state=r['rng'].get_state().clone();model=Model()
            with self.assertRaises(ValueError):choose_requests(model,Features(),[r],max_rows=limit)
            self.assertEqual(model.calls,0);self.assertTrue(torch.equal(r['rng'].get_state(),state))

    def test_partial_sampling_failure_restores_all_streams(self):
        requests=[request('a',7,1,4,4),request('b',4,1,3,9)];before=[x['rng'].get_state().clone() for x in requests]
        calls=[]
        def guard():
            calls.append(1)
            if len(calls)==2:raise RuntimeError('Original deadline or GPU pause')
        with self.assertRaises(RuntimeError):choose_requests(Model(),Features(),requests,before_forward=guard)
        self.assertEqual(len(calls),2)
        self.assertTrue(all(torch.equal(r['rng'].get_state(),s) for r,s in zip(requests,before)))

    def test_no_shared_environment_rng_or_request_identity(self):
        a=request('a',3,1,2);b=request('a',3,1,2)
        with self.assertRaises(ValueError):choose_requests(Model(),Features(),[a,b])
        with self.assertRaises(ValueError):choose_requests(Model().train(),Features(),[request('c',3,1,2)])
        b['key']='b';b['rng']=a['rng']
        with self.assertRaises(ValueError):choose_requests(Model(),Features(),[a,b])


if __name__=='__main__':unittest.main()
