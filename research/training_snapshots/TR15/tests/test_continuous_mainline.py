import copy
import json
from pathlib import Path
import unittest
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from ptcg_lab.continuous_mainline import validate_training, BASE, ADDITIONAL
from ptcg_lab.rl_core import ppo_step
from tests.test_gradient_diagnostic import TinyActor


class ContinuousContractTests(unittest.TestCase):
    def test_linux_runtime_normalizes_every_fixed_opponent_file_without_changing_hashes(self):
        import copy
        from pathlib import Path,PurePosixPath
        from ptcg_lab.continuous_mainline import normalize_fixed_file_paths
        from ptcg_lab.io import read_json,sha256
        root=Path(__file__).resolve().parents[1]
        original=read_json(root/'configs/training/tr11_continuous_basic_learning.json')
        runtime=normalize_fixed_file_paths(copy.deepcopy(original))
        self.assertEqual(runtime['ppo'],original['ppo'])
        for fixed,old in zip(runtime['league']['fixed_opponents'],original['league']['fixed_opponents']):
            self.assertEqual(set(fixed['files'].values()),set(old['files'].values()))
            for name,digest in fixed['files'].items():
                self.assertNotIn('\\',str(PurePosixPath(name)))
                self.assertEqual(sha256(root/fixed['path']/name),digest)
        self.assertIn('data/embed_pca.npz',runtime['league']['fixed_opponents'][0]['files'])

    def test_runtime_path_conversion_rejects_collisions_and_path_escape(self):
        from ptcg_lab.continuous_mainline import normalize_fixed_file_paths
        for files in [{'a\\b':'a','a/b':'a'},{'../outside':'a'}]:
            with self.assertRaises(ValueError):normalize_fixed_file_paths({'league':{'fixed_opponents':[{'files':files}]}})
    def test_authorized_600s_numeric_window_does_not_expand_legacy_or_scientific_roles(self):
        from ptcg_lab.server_trial_agent import trial_seconds_maximum
        self.assertEqual(trial_seconds_maximum('gpu_existing_restore'),300)
        self.assertEqual(trial_seconds_maximum('gpu_existing_restore',True),600)
        self.assertEqual(trial_seconds_maximum('derived_scientific'),7200)
        with self.assertRaises(ValueError):trial_seconds_maximum('derived_scientific',True)

    def setUp(self):
        root=Path(__file__).resolve().parents[1]
        self.parent=json.loads((root/'configs/training/tr10_ppo.json').read_bytes())
        self.training=copy.deepcopy(self.parent)
        self.training.update(hypothesis_id='TR11',target_decisions=BASE+ADDITIONAL,
                             decision_milestones=[BASE+8192,BASE+ADDITIONAL])
        self.training['league']['fixed_opponents']=[x for x in self.parent['league']['fixed_opponents'] if x['id']=='ppo']
        self.training['league']['weights']={'fixed_diverse':1.,'historical_learner':0.,'latest_frozen_learner':0.,'weakness_training_opponents':0.}

    def test_exact_finite_new_question_is_admitted(self):
        validate_training(self.training,self.parent)

    def test_old_attempt_cannot_be_renamed_or_given_a_larger_target(self):
        for key,value in [('hypothesis_id','TR10'),('target_decisions',BASE+ADDITIONAL+1)]:
            changed=copy.deepcopy(self.training);changed[key]=value
            with self.assertRaises(ValueError):validate_training(changed,self.parent)

    def test_algorithm_seed_and_milestone_changes_are_rejected(self):
        for change in ['learning_rate','seed','milestone']:
            changed=copy.deepcopy(self.training)
            if change=='learning_rate':changed['ppo']['actor_learning_rate']=3e-5
            elif change=='seed':changed['seed']+=1
            else:changed['decision_milestones']=[BASE+4096,BASE+ADDITIONAL]
            with self.assertRaises(ValueError):validate_training(changed,self.parent)

    def test_training_opponent_versions_and_weights_cannot_drift(self):
        for change in ['weight','opponent']:
            changed=copy.deepcopy(self.training)
            if change=='weight':changed['league']['weights']['fixed_diverse']=.9
            else:changed['league']['fixed_opponents'][0]['files']['model.pt']='0'*64
            with self.assertRaises(ValueError):validate_training(changed,self.parent)

    def test_gradient_observation_preserves_exact_optimizer_update(self):
        torch.manual_seed(3)
        baseline=TinyActor();observed=copy.deepcopy(baseline)
        rows=np.asarray([[1.,0.]],np.float32)
        with torch.no_grad():scores,_=baseline.forward_eval(torch.from_numpy(rows));old=float(scores.log_softmax(-1)[0,0])
        fixture=[{'steps':[{'rows':rows,'mapping':np.asarray([[0,1,2]]),'n':2,'target':0}],
                  'old_logp':old,'old_value':.5,'reward':1.,'terminal':True}]
        states=[]
        for model,enabled in [(baseline,False),(observed,True)]:
            optimizer=torch.optim.Adam(model.parameters(),lr=.001)
            before=torch.get_rng_state().clone()
            metrics=ppo_step(model,optimizer,fixture,np.asarray([1.],np.float32),np.asarray([1.],np.float32),2,
                normalize_advantages=False,record_gradient_components=enabled)
            self.assertTrue(torch.equal(before,torch.get_rng_state()))
            states.append((copy.deepcopy(model.state_dict()),optimizer.state_dict()))
            if enabled:self.assertGreater(metrics['policy_gradient_l2'],0)
        for key,value in states[0][0].items():self.assertTrue(torch.equal(value,states[1][0][key]))
        for key,value in states[0][1]['state'].items():
            for name,slot in value.items():self.assertTrue(torch.equal(slot,states[1][1]['state'][key][name]))

    def test_milestone_keeps_full_state_after_hot_name_replacement_and_uses_existing_outbox_namespace(self):
        from ptcg_lab.continuous_mainline import preserve_milestones
        from ptcg_lab.archive_outbox import ArchiveOutbox
        from ptcg_lab.io import sha256,read_json
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);output=root/'owned-run';output.mkdir()
            checkpoint=output/'checkpoint-commits/objects'/('cp-'+'a'*64+'.pt')
            prefix=checkpoint.with_name('prefix-'+'b'*64+'.json')
            checkpoint.parent.mkdir(parents=True);checkpoint.write_bytes(b'complete-model-optimizer-all-RNG')
            prefix.write_bytes(b'prefix')
            frozen=output/'league/learner-0006.pt';frozen.parent.mkdir();frozen.write_bytes(b'actor')
            ref=lambda p:{'path':p.relative_to(output).as_posix(),'sha256':sha256(p),'bytes':p.stat().st_size}
            recovery={'checkpoint':ref(checkpoint),'prefix':ref(prefix),'pending':[],
                      'cursor':{'version':6,'trained_decisions':44000,'next_series_id':280}}
            league=SimpleNamespace(entries=[{'version':6,**ref(frozen)}],stats={})
            config={'project_path':str(root),'hypothesis_id':'TR11','decision_milestones':[BASE+8192],
                    'league':{'weights':self.training['league']['weights']}}
            box=ArchiveOutbox(output,output.name,1)
            # Exercise the real pin/outbox/content behavior, with the Linux
            # ownership helper replaced only for this Windows temporary root.
            def contained(r,p):
                path=Path(p).resolve()
                self.assertIn(Path(r).resolve(),path.parents)
                return path
            with patch('ptcg_lab.server_workspace.inside',side_effect=contained):
                preserve_milestones(output,config,BASE+8200,recovery,league,box)
            marker=output/'milestones'/str(BASE+8192)/'manifest.json'
            record=read_json(marker)
            self.assertEqual(record['new_unique_since_parent'],8200)
            self.assertEqual({r['object'] for r in box.records()},
                {recovery['checkpoint']['path'],recovery['prefix']['path'],frozen.relative_to(output).as_posix()})
            pin=output/record['full_state']['checkpoint']['path']
            checkpoint.unlink();checkpoint.write_bytes(b'new hot content')
            self.assertEqual(pin.read_bytes(),b'complete-model-optimizer-all-RNG')

    def test_prepared_inspection_materializes_exact_missing_public_metadata_before_actor_load(self):
        from ptcg_lab.prepared_parent_verify import materialize_catalog
        from ptcg_lab.io import sha256
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'prepared';job=root/'new-inspection'
            relative='ptcg_lab/public_cg/metadata.json'
            original=source/'control'/relative;original.parent.mkdir(parents=True)
            original.write_bytes(b'{"public_catalogue":true}')
            manifest={'file_map':[{'path':relative,'sha256':sha256(original),'bytes':original.stat().st_size}]}
            with patch('ptcg_lab.server_workspace.inside',side_effect=lambda r,p:Path(p)):
                linked=materialize_catalog(root,job,source,manifest)
                self.assertEqual(linked.read_bytes(),original.read_bytes())
                self.assertEqual(original.stat().st_ino,linked.stat().st_ino)
                original.write_bytes(b'changed public asset')
                with self.assertRaises(ValueError):materialize_catalog(root,job,source,manifest)


if __name__=='__main__':unittest.main()
