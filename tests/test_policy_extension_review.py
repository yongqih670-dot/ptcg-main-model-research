"""Adversarial regression cases found during the implementation review."""
import copy
import tempfile
import types
import unittest
from pathlib import Path
import torch
from unittest.mock import patch
from tests.test_policy_extensions import PolicyExtensionFixture
from ptcg_lab.policy_extensions import ZoneMemoryPolicy,complete_sequences
from ptcg_lab.rl_core import choose_decision,audit_recurrent_rollout
from ptcg_lab.rl_league import League


def encoded_observation(count=3):
    """Synthetic addressed observation exercising the real public encoder."""
    player=dict(active=[],bench=[],hand=[],discard=[],prize=[None]*6,deckCount=54,handCount=0,benchMax=5,
        poisoned=False,burned=False,asleep=False,paralyzed=False,confused=False)
    return dict(logs=[],current=dict(players=[copy.deepcopy(player),copy.deepcopy(player)],yourIndex=0,round=1,
        turn=1,firstPlayer=0,turnActionCount=0,supporterPlayed=False,stadiumPlayed=False,energyAttached=False,
        retreated=False,stadium=[],looking=[]),select=dict(type=0,context=0,
        option=[dict(type=0,number=i) for i in range(count)],minCount=1,maxCount=2,
        remainDamageCounter=0,remainEnergyCost=0))


class ExtensionReviewTests(PolicyExtensionFixture,unittest.TestCase):
    def test_memory_can_change_relative_preferences_for_identical_pointer_options(self):
        torch.nn.init.normal_(self.model.policy_residual[-1].weight,std=.2)
        rows=torch.from_numpy(self.rows())
        a,_,_=self.model.forward_step(rows,torch.zeros(1,8),advance=False)
        b,_,_=self.model.forward_step(rows,torch.ones(1,8),advance=False)
        # Three legal options point at the same pair but carry different cards.
        delta=(a[:,:3].softmax(-1)-b[:,:3].softmax(-1)).abs().max()
        self.assertGreater(float(delta.detach()),1e-7)

    def test_same_shape_attention_head_change_must_not_load(self):
        altered=ZoneMemoryPolicy(copy.deepcopy(self.base),self.b,
            {**self.model.options,'heads':2})
        with self.assertRaises(RuntimeError):altered.load_state_dict(self.model.state_dict())

    def test_middle_swap_or_drop_is_rejected_before_any_network_call(self):
        tracker=types.SimpleNamespace();data=[]
        for i in range(4):
            _,d=choose_decision(self.model,self.features,self.obs,tracker,0,greedy=True)
            d.update(series_id=0,round=1,learner_seat=0,policy_version='0',terminal=i==3)
            data.append(d)
        for bad in ([data[0],data[2],data[1],data[3]],[data[0],data[2],data[3]]):
            with self.assertRaises(ValueError):complete_sequences(bad)

    def test_nonfinite_old_value_cannot_pass_replay_audit(self):
        data=self.trajectory();data[0]['old_value']=float('nan')
        with self.assertRaises(ValueError):audit_recurrent_rollout(self.model,data,self.b.STOP)

    def test_resume_cannot_silently_change_history_sampling_distribution(self):
        config={'weights':dict(fixed_diverse=0.,historical_learner=.2,latest_frozen_learner=.8,weakness_training_opponents=0.),
                'fixed_opponents':[],'history_startup':'latest_frozen_learner','history_window':8}
        with tempfile.TemporaryDirectory() as tmp:
            League(Path(tmp),Path(tmp),config,self.model,self.features,list(range(60))).freeze(0)
            with self.assertRaises(ValueError):
                League(Path(tmp),Path(tmp),{**config,'history_window':1},self.model,self.features,list(range(60)))

    def test_fresh_compact_origin_binds_and_restores_extensions(self):
        from ptcg_lab.fresh_compact import initialize,settings_identity
        from ptcg_lab.training_checkpoint import load_training_checkpoint
        from ptcg_lab.rl_core import optimizer_for
        config=dict(hypothesis_id='new-research',parent_model_sha256='1'*64,source_manifest_sha256='2'*64,
            seed=1,target_decisions=8,ppo={},league={},model_extensions=self.model.options)
        other={**config,'model_extensions':{**self.model.options,'heads':2}}
        self.assertNotEqual(settings_identity(config),settings_identity(other))
        with tempfile.TemporaryDirectory() as tmp:
            spec=initialize(tmp,config,self.model,optimizer_for(self.model))
            saved=load_training_checkpoint(tmp,spec,1)['saved']
            self.assertEqual(saved['model_extensions'],config['model_extensions'])

    def test_real_encoder_and_tracker_overflow_replay(self):
        obs=encoded_observation(65);tracker=self.features_base.Tracker([1]*60)
        torch.nn.init.normal_(self.model.policy_residual[-1].weight,std=.1)
        data=[]
        for i in range(3):
            tracker.update(obs)
            _,d=choose_decision(self.model,self.features_base,obs,tracker,0,greedy=True)
            d.update(series_id=0,round=1,learner_seat=0,policy_version='0',terminal=i==2)
            data.append(d)
        audit_recurrent_rollout(self.model,data,self.b.STOP,1e-5)

    def test_compact_bootstrap_dynamic_coverage_and_changed_restore_rejection(self):
        from ptcg_lab.fresh_compact import initialize,audit_gpu_restart
        from ptcg_lab.training_checkpoint import load_training_checkpoint
        from ptcg_lab.rl_core import optimizer_for
        from ptcg_lab.io import read_json
        import numpy as np
        config=dict(hypothesis_id='new-research',parent_model_sha256='1'*64,source_manifest_sha256='2'*64,
            seed=1,target_decisions=8,ppo={},league={},model_extensions=self.model.options)
        draws=dict(python=.5,numpy=np.arange(8),torch=torch.zeros(8),cuda=torch.zeros(16))
        with tempfile.TemporaryDirectory() as tmp:
            opt=optimizer_for(self.model);spec=initialize(tmp,config,self.model,opt)
            saved=load_training_checkpoint(tmp,spec,1)['saved']
            with patch('ptcg_lab.fresh_compact._draws',return_value=draws):
                audit_gpu_restart(tmp,self.model,opt,saved,'synthetic-device',True)
                audit_gpu_restart(tmp,self.model,opt,saved,'synthetic-device',False)
                self.assertEqual(read_json(Path(tmp)/'fresh-gpu-restore.json')['model_tensors'],len(self.model.state_dict()))
                bad=copy.deepcopy(saved);del bad['model']['value_residual.bias']
                with self.assertRaises(ValueError):audit_gpu_restart(tmp,self.model,opt,bad,'synthetic-device',False)

    def test_default_width_and_memory_have_finite_forward_and_gradients(self):
        base=self.module.PTCGTransformer(critic='blind').eval()
        model=ZoneMemoryPolicy(base,self.b,dict(zone_readout=True,memory_size=128,heads=4)).eval()
        rows=torch.from_numpy(self.rows())
        with torch.no_grad():
            original=base.forward_eval(rows);scores,value,memory=model.forward_step(rows)
            self.assertTrue(torch.equal(scores,original[0]) and torch.equal(value,original[1]))
            self.assertEqual(tuple(memory.shape),(1,128))
        torch.nn.init.normal_(model.policy_residual[-1].weight,std=.01)
        scores,_,memory=model.forward_step(rows)
        scores,_,_=model.forward_step(rows,memory)
        scores[0,0].backward()
        self.assertTrue(torch.isfinite(model.memory.weight_hh.grad).all())

    def test_nonfinite_logp_and_malformed_boundary_flags_are_rejected(self):
        for field,value in [('old_logp',float('nan')),('old_value',float('inf'))]:
            data=self.trajectory();data[0][field]=value
            with self.assertRaises(ValueError):audit_recurrent_rollout(self.model,data,self.b.STOP)
        for field in ('sequence_start','terminal'):
            data=self.trajectory();data[0][field]=1
            with self.assertRaises(ValueError):complete_sequences(data)

    def test_compact_accepts_explicit_features_with_omitted_default_options(self):
        from ptcg_lab.fresh_compact import initialize,audit_gpu_restart
        from ptcg_lab.training_checkpoint import load_training_checkpoint
        from ptcg_lab.rl_core import optimizer_for
        import numpy as np
        model=ZoneMemoryPolicy(copy.deepcopy(self.base),self.b,dict(zone_readout=True))
        config=dict(hypothesis_id='new-research',parent_model_sha256='1'*64,source_manifest_sha256='2'*64,
            seed=1,target_decisions=8,ppo={},league={},model_extensions={'zone_readout':True})
        draws=dict(python=.5,numpy=np.arange(8),torch=torch.zeros(8),cuda=torch.zeros(16))
        with tempfile.TemporaryDirectory() as tmp:
            opt=optimizer_for(model);spec=initialize(tmp,config,model,opt)
            saved=load_training_checkpoint(tmp,spec,1)['saved']
            with patch('ptcg_lab.fresh_compact._draws',return_value=draws):
                audit_gpu_restart(tmp,model,opt,saved,'synthetic-device',True)


if __name__=='__main__':unittest.main()
