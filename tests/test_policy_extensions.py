"""CPU tests on the actual inherited model, with synthetic text channels/rows.

No private weights, original text vectors, native battles or training budgets
are consumed. Tests do not establish battle strength or engine acceptance.
"""
import copy
import random
import tempfile
import os
import time
import json
import shutil
import subprocess
import sys
import importlib.util
import types
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from ptcg_lab.rl_core import (RLFeatures, choose_decision, evaluate_decisions,
    optimizer_for, ppo_step, audit_recurrent_rollout, load_full_actor)
from ptcg_lab.policy_extensions import (ZoneMemoryPolicy, complete_sequences,
    sequence_order, sequence_batch_size, validate_options)
from ptcg_lab.rl_league import League, choose_bucket, FrozenOpponent
from ptcg_lab.rl_sampling import cpu_replica
from ptcg_lab.io import sha256


class Features:
    def __init__(self,b): self.b=b
    def make(self,obs,tracker,forced,picked):
        b=self.b;n=len(obs['select']['option']);rows=[];maps=[]
        for start in range(0,n,b.STOP):
            row=np.zeros(b.OBS_SIZE,np.float32)
            ti=row[slice(*b.OFFSETS['tok_int'])].reshape(b.MAX_TOKENS,b.TOK_INT)
            ti[:4,0]=[1,2,3,4];ti[:4,1]=[1,3,5,7];ti[:4,2]=[1,2,1,1]
            oi=row[slice(*b.OFFSETS['opt_int'])].reshape(b.MAX_OPTIONS,b.OPT_INT)
            mask=row[slice(*b.OFFSETS['opt_mask'])]
            count=min(b.STOP,n-start);oi[:count,3]=np.arange(start+1,start+count+1)
            oi[:count,1]=1;oi[:count,2]=2
            mask[:count]=[start+j not in picked for j in range(count)]
            mask[b.STOP]=len(picked)>=obs['select']['minCount']
            mapping=np.full(b.MAX_OPTIONS,-1,np.int32)
            mapping[:count]=range(start,start+count);mapping[b.STOP]=n
            rows.append(row);maps.append(mapping)
        return np.stack(rows),np.stack(maps)

    class Tracker:
        def __init__(self,deck):pass
        def update(self,obs):pass


class EngineFixture:
    """Two complete synthetic games with two learner decisions each."""
    def __init__(self,*args):self.i=0
    def send(self,request):pass
    def close(self):pass
    def read(self):
        self.i+=1
        if self.i==5:return {'event':'series_end','statuses':['DONE','DONE'],'scores':[2,0,0],'rewards':[1,-1]}
        rd=(self.i-1)//2+1
        return {'event':'decision','seat':0,'observation':{'current':{'round':rd,'yourIndex':0,
            'players':[{'win':rd-1},{'win':0}],'draw':0},
            'select':{'option':[0,1,2],'minCount':1,'maxCount':2}}}


class PolicyExtensionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.bundle=Path(__file__).resolve().parents[1]/'agents/frozen/ppo-zero-anchor'
        cls.features_base=RLFeatures(cls.bundle)
        from ptcg.rl import model,cards
        cls.module=model;cls.b=cls.features_base.b
        cls.text=patch.object(model,'load_text_tables',return_value=(
            np.zeros((cards.N_CARDS,128),np.float32),np.zeros((cards.N_ATTACKS,64),np.float32),
            np.zeros((cards.N_SKILLS,64),np.float32)))
        cls.text.start()
    @classmethod
    def tearDownClass(cls):cls.text.stop()
    def setUp(self):
        torch.manual_seed(41)
        self.base=self.module.PTCGTransformer(d=16,layers=1,heads=8,ffn=32,critic='blind').eval()
        # Match the production trainable scope, not unused oracle auxiliaries.
        from ptcg_lab.rl_core import UNUSED
        for name,p in self.base.named_parameters():
            if name.startswith(UNUSED):p.requires_grad_(False)
        self.model=ZoneMemoryPolicy(self.base,self.b,dict(zone_readout=True,memory_size=8,heads=4)).eval()
        self.features=Features(self.b)
        self.obs={'select':{'option':[0,1,2],'minCount':1,'maxCount':2}}
    def rows(self):return self.features.make(self.obs,None,0,[])[0]
    def trajectory(self,model=None):
        model=model or self.model;tracker=types.SimpleNamespace();result=[]
        for j in range(2):
            _,d=choose_decision(model,self.features,self.obs,tracker,0,greedy=True)
            d.update(series_id=0,round=1,learner_seat=0,policy_version='0',terminal=j==1,reward=float(j==1))
            result.append(d)
        return result
    def test_initial_logits_and_values_exactly_preserve_parent(self):
        rows=torch.from_numpy(self.rows())
        expected=self.base.forward_eval(rows)
        actual=self.model.forward_step(rows)
        for x,y in zip(expected,actual[:2]):self.assertTrue(torch.equal(x,y))
    def test_stateless_zone_ablation_and_empty_groups_are_finite(self):
        m=ZoneMemoryPolicy(copy.deepcopy(self.base),self.b,dict(zone_readout=True))
        rows=torch.from_numpy(self.rows());rows[:,slice(*self.b.OFFSETS['tok_int'])]=0
        a,v=m.forward_eval(rows)
        self.assertTrue(torch.isfinite(a).all() and torch.isfinite(v).all())
    def test_hidden_input_and_implicit_memory_are_rejected(self):
        rows=torch.from_numpy(self.rows());rows[:,self.b.OFFSETS['oracle'][0]]=1
        with self.assertRaises(ValueError):self.model.forward_step(rows)
        with self.assertRaises(ValueError):self.model.forward_eval(rows)
        with self.assertRaises(ValueError):validate_options({'memory_size':-1})
    def test_rollout_and_full_sequence_replay_match(self):
        # Exercise learned memory, not only the zero-output compatibility path.
        torch.nn.init.normal_(self.model.policy_residual[-1].weight,std=.1)
        torch.nn.init.normal_(self.model.value_residual.weight,std=.1)
        decisions=self.trajectory()
        audit_recurrent_rollout(self.model,decisions,self.b.STOP,1e-5)
        logp,values,_=evaluate_decisions(self.model,decisions,self.b.STOP)
        np.testing.assert_allclose(logp.detach(),[d['old_logp'] for d in decisions],atol=1e-5)
        np.testing.assert_allclose(values.detach(),[d['old_value'] for d in decisions],atol=1e-5)
    def test_multiselect_advances_memory_once_and_forced_choice_does_not(self):
        tracker=types.SimpleNamespace()
        with patch.object(self.model.memory,'forward',wraps=self.model.memory.forward) as call:
            _,d=choose_decision(self.model,self.features,self.obs,tracker,0,greedy=True)
            self.assertEqual(call.call_count,1)
        before=tracker._policy_memory.clone()
        obs={'select':{'option':[0],'minCount':1,'maxCount':1}}
        action,d=choose_decision(self.model,self.features,obs,tracker,0)
        self.assertEqual(action,[0]);self.assertIsNone(d)
        self.assertTrue(torch.equal(before,tracker._policy_memory))
    def test_overflow_chunks_have_shared_stop_and_one_memory_transition(self):
        obs={'select':{'option':list(range(65)),'minCount':1,'maxCount':2}}
        with patch.object(self.model.memory,'forward',wraps=self.model.memory.forward) as call:
            action,_=choose_decision(self.model,self.features,obs,types.SimpleNamespace(),0,greedy=True)
            self.assertEqual(call.call_count,1)
        self.assertTrue(1<=len(action)<=2 and len(set(action))==len(action))
    def test_audit_detects_unselected_distribution_corruption(self):
        decisions=self.trajectory();decisions[0]['steps'][0]['old_full_logp'][1]+=.5
        with self.assertRaises(ValueError):audit_recurrent_rollout(self.model,decisions,self.b.STOP)
    def test_sequence_validation_and_packing_reject_partial_or_mixed_games(self):
        decisions=self.trajectory();second=copy.deepcopy(decisions)
        for d in second:d['series_id']=1
        data=decisions+second
        indices=sequence_order(data,np.random.RandomState(7))
        complete_sequences([data[i] for i in indices])
        self.assertEqual(sequence_batch_size(data,indices,1),2)
        for bad in [data[1:],data[:-1],list(reversed(decisions)),decisions+decisions]:
            with self.assertRaises(ValueError):complete_sequences(bad)
        decisions[1]['policy_version']='1'
        with self.assertRaises(ValueError):complete_sequences(decisions)
    def test_memory_receives_gradient_across_decisions(self):
        nn=torch.nn
        nn.init.normal_(self.model.policy_residual[-1].weight,std=.1)
        rows=torch.from_numpy(self.rows())
        _,_,memory=self.model.forward_step(rows);memory.retain_grad()
        logits,_,_=self.model.forward_step(rows,memory)
        logits[0,0].backward()
        self.assertGreater(float(memory.grad.abs().sum()),0)
        self.assertGreater(float(self.model.memory.weight_hh.grad.abs().sum()),0)
    def test_sequence_ppo_update_and_resume_next_step_are_exact(self):
        m=self.model;opt=optimizer_for(m)
        decisions=self.trajectory();adv=np.array([1.,-.5],np.float32);ret=np.ones(2,np.float32)
        ppo_step(m,opt,decisions,adv,ret,self.b.STOP,normalize_advantages=False)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'state.pt';torch.save({'model':m.state_dict(),'optimizer':opt.state_dict()},path)
            saved=torch.load(path,weights_only=True)
            other=copy.deepcopy(m);other.load_state_dict(saved['model']);op=optimizer_for(other);op.load_state_dict(saved['optimizer'])
            d=self.trajectory(m)
            for model,optimizer in [(m,opt),(other,op)]:
                ppo_step(model,optimizer,d,adv,ret,self.b.STOP,normalize_advantages=False)
            for k,v in m.state_dict().items():self.assertTrue(torch.equal(v,other.state_dict()[k]),k)
    def test_cpu_replica_preserves_rng_and_all_extension_tensors(self):
        before=torch.get_rng_state().clone();replica=cpu_replica(self.model)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        for k,v in self.model.state_dict().items():self.assertTrue(torch.equal(v,replica.state_dict()[k]))
    def test_league_freezes_loads_history_and_never_mutates_learner(self):
        config={'weights':{'fixed_diverse':0.,'historical_learner':1.,'latest_frozen_learner':0.,'weakness_training_opponents':0.},
                'fixed_opponents':[],'history_window':1,'history_startup':'latest_frozen_learner'}
        with tempfile.TemporaryDirectory() as tmp:
            league=League(Path(tmp),Path(tmp),config,self.model,self.features,list(range(60)))
            league.freeze(0)
            opponent,assignment=league.select(0,1)
            self.assertEqual(assignment['bucket'],'latest_frozen_learner')
            with torch.no_grad():self.model.value_residual.bias.add_(1)
            league.freeze(1)
            opponent,assignment=league.select(0,2)
            self.assertEqual(assignment['bucket'],'historical_learner')
            self.assertEqual(opponent.identity['version'],0)
            self.assertEqual(float(opponent.model.value_residual.bias),0)
            self.assertEqual(float(self.model.value_residual.bias.detach()),1)
            # Snapshot identity is rechecked even when a model is cached.
            (Path(tmp)/'league/learner-0000.pt').write_bytes(b'corrupt')
            with self.assertRaises(ValueError):league.select(0,3)
    def test_mixture_is_reproducible_and_invalid_weights_fail(self):
        weights=dict(fixed_diverse=0.,historical_learner=.2,latest_frozen_learner=.8,weakness_training_opponents=0.)
        rng=random.Random(10);buckets=[choose_bucket(rng,True,weights) for _ in range(2000)]
        self.assertTrue(300<buckets.count('historical_learner')<500)
        with self.assertRaises(ValueError):choose_bucket(rng,True,{**weights,'historical_learner':-.2})
    def test_loader_strictly_roundtrips_extended_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.pt';torch.save(self.model.state_dict(),path)
            with patch('ptcg_lab.rl_core.RLFeatures',return_value=self.features):
                restored,_,_=load_full_actor(tmp,sha256(path),reset_value=False,model_extensions=self.model.options)
                for k,v in self.model.state_dict().items():self.assertTrue(torch.equal(v,restored.state_dict()[k]))
                with self.assertRaises(ValueError):load_full_actor(tmp,sha256(path))
                with self.assertRaises(ValueError):load_full_actor(tmp,'incorrect',model_extensions=self.model.options)
    def test_actual_training_loop_collects_sequences_updates_saves_and_resumes(self):
        from ptcg_lab import train_ppo
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)
            config={'project_path':str(root),'bundle_path':'agents/frozen/ppo-zero-anchor',
                'parent_model_sha256':'synthetic','seed':71,'sampling_device':'cpu','engine_manifest_sha256':'synthetic',
                'target_decisions':4,'model_extensions':self.model.options,
                'league':{'weights':dict(fixed_diverse=0.,historical_learner=.2,latest_frozen_learner=.8,weakness_training_opponents=0.),
                    'history_startup':'latest_frozen_learner','fixed_opponents':[]},
                'ppo':{'actor_learning_rate':1e-5,'value_head_learning_rate':1e-4,'gamma':1.,'gae_lambda':1.,
                    'rollout_complete_learner_decisions':4,'minibatch_initial':2,'epochs':2,'minibatch_memory_fallbacks':[1],
                    'clip':.2,'entropy_coefficient':.01,'value_loss_coefficient':.5,'max_grad_norm':.5,'target_kl':.02}}
            env={'PTCG_JOB_RUN_ID':output.name,'PTCG_JOB_DEADLINE_UNIX':str(time.time()+60)}
            with patch.dict(os.environ,env),patch.object(train_ppo,'load_full_actor',return_value=(self.model,self.features,{})),patch.object(train_ppo,'Engine',EngineFixture):
                train_ppo.run(config,output)
                from ptcg_lab.io import read_json
                self.assertTrue(read_json(output/'metrics.json')['complete'])
                self.assertEqual(read_json(output/'metrics.json')['unique_PPO_trained_decisions'],4)
                checkpoint=torch.load(output/'checkpoint.pt',weights_only=False)
                self.assertEqual(checkpoint['model_extensions'],self.model.options)
                self.assertTrue((output/'recurrent-replay-v0000.json').exists())
                rollout=torch.load(next((output/'rollouts').glob('*.pt')),weights_only=False)['decisions']
                self.assertEqual([d['sequence_start'] for d in rollout],[True,False,True,False])
                before={k:v.clone() for k,v in self.model.state_dict().items()}
                # Completed resume is a no-update operation; no duplicate data.
                train_ppo.run(config,output)
                for k,v in self.model.state_dict().items():self.assertTrue(torch.equal(v,before[k]))
                with self.assertRaises(ValueError):train_ppo.run({**config,'model_extensions':{**self.model.options,'heads':2}},output)
    def test_packaged_extended_model_loads_in_isolated_process(self):
        root=Path(__file__).resolve().parents[1]
        spec=importlib.util.spec_from_file_location('package_extension_test',root/'scripts/package_rl_agent.py')
        script=importlib.util.module_from_spec(spec);spec.loader.exec_module(script)
        from ptcg.rl import cards
        with tempfile.TemporaryDirectory() as tmp:
            project=Path(tmp);folder=project/'agents/test'
            shutil.copytree(self.bundle,folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
            shutil.copytree(root/'ptcg_lab',project/'ptcg_lab',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
            np.savez(folder/'data/embed_pca.npz',cards=np.zeros((cards.N_CARDS,128),np.float32),
                attacks=np.zeros((cards.N_ATTACKS,64),np.float32),skills=np.zeros((cards.N_SKILLS,64),np.float32))
            torch.save(self.model.state_dict(),folder/'model.pt')
            payload=dict(id='synthetic',kind='zone_memory_rl',path='agents/test',model_sha256=sha256(folder/'model.pt'),
                model_extensions=self.model.options,training_run_id='synthetic-only')
            path=project/'spec.json';path.write_text(json.dumps(payload),encoding='utf-8')
            script.ROOT=project
            with patch.object(sys,'argv',['package','--spec',str(path),'--name','synthetic']):script.main()
            stage=project/'releases/validation/synthetic/contents'
            code="import main; from ptcg_lab.rl_core import load_full_actor; load_full_actor('bundle',main._SPEC['model_sha256'],reset_value=False,model_extensions=main._SPEC['model_extensions']); print('extended-load-ok')"
            result=subprocess.run([sys.executable,'-B','-c',code],cwd=stage,capture_output=True,text=True,timeout=45)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('extended-load-ok',result.stdout)


if __name__=='__main__':unittest.main()
