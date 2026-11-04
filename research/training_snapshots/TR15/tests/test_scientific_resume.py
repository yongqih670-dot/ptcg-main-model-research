import copy
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import torch
from ptcg_lab.io import atomic_json,read_json,sha256
from ptcg_lab.fresh_compact import initialize
from ptcg_lab.archive_state import initial_state,advance_state
from ptcg_lab.archive_prefix import compact_record
from ptcg_lab.checkpoint_refs import commit_checkpoint,read_index
from ptcg_lab.tensor_io import load_torch,atomic_torch_save
from ptcg_lab.scientific_resume import prepare,validate_scope,validate_postcommit_recovery


class ScientificResumeTests(unittest.TestCase):
    def test_postcommit_recovery_rejects_partial_update_or_wrong_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'20261025T212721Z_tr07-v01_2f8e6f';log=source/'logs/payload.log'
            log.parent.mkdir(parents=True);log.write_text('outbox.evict_consumed(\nOSError: Hybrid filesystem reserve reached\n')
            cursor={'version':2,'trained_decisions':16646,'next_series_id':102}
            points=[{'checkpoint':{'sha256':str(i)*64},'cursor':cursor,'pending':[]} for i in [1,2]]
            path=source/'checkpoint-commits/index.json';atomic_json(path,{'commits':points})
            proof={'payload_log_sha256':sha256(log),'checkpoint_hashes':['1'*64,'2'*64],'cursor':cursor}
            manifest={'source_postcommit_recovery':proof}
            self.assertEqual(validate_postcommit_recovery(source,manifest),proof)
            wrong=copy.deepcopy(points);wrong[1]['cursor']['trained_decisions']=16647
            atomic_json(path,{'commits':wrong})
            with self.assertRaises(ValueError):validate_postcommit_recovery(source,manifest)
            atomic_json(path,{'commits':points[:1]})
            with self.assertRaises(ValueError):validate_postcommit_recovery(source,manifest)
            log.write_text('a different payload failure')
            with self.assertRaises(ValueError):validate_postcommit_recovery(source,manifest)

    def test_TR07_second_keeps_time_and_ignores_unrelated_old_TR05_derived(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);rid='20261025T212721Z_tr07-v01_2f8e6f';source=root/'runs'/rid
            atomic_json(source/'config.json',{'check_variant':'fresh_scientific','hypothesis_id':'TR07'})
            atomic_json(source/'attempts.json',[{'seconds':7100}]);atomic_json(source/'remote-job.json',{})
            atomic_json(root/'runs/old-TR05/config.json',{'check_variant':'derived_scientific','hypothesis_id':'TR05'})
            config={'original_training_run_id':rid,'prior_training_seconds':7100,'hypothesis_id':'TR07',
                'training_attempt_index':2,'maximum_attempts':1,'check_variant':'derived_scientific'}
            with patch('ptcg_lab.remote_lease.validate_no_unreconciled_remote'):
                validate_scope(root,config,time.time()+7199)
                with self.assertRaises(ValueError):validate_scope(root,dict(config,training_attempt_index=3),time.time()+7199)
                atomic_json(root/'runs/TR07-second/config.json',config)
                with self.assertRaises(ValueError):validate_scope(root,config,time.time()+7199)

    def test_second_attempt_does_not_regain_time_or_third_attempt(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);rid='20261025T174630Z_tr05-value-head-only_0d9af6';source=root/'runs'/rid
            atomic_json(source/'config.json',{'check_variant':'fresh_scientific','hypothesis_id':'TR05'})
            atomic_json(source/'attempts.json',[{'seconds':1984.891}]);atomic_json(source/'remote-job.json',{})
            config={'original_training_run_id':rid,'prior_training_seconds':1984.891,
                    'hypothesis_id':'TR05','training_attempt_index':2,'maximum_attempts':1,'check_variant':'derived_scientific'}
            with patch('ptcg_lab.remote_lease.validate_no_unreconciled_remote'):
                validate_scope(root,config,time.time()+7199)
                for field,value in [('training_attempt_index',3),('prior_training_seconds',0),('maximum_attempts',2)]:
                    with self.assertRaises(ValueError):validate_scope(root,dict(config,**{field:value}),time.time()+7199)
                with self.assertRaises(ValueError):validate_scope(root,config,time.time()+7201)
                atomic_json(root/'runs'/'derived'/'config.json',config)
                with self.assertRaises(ValueError):validate_scope(root,config,time.time()+7199)
                validate_scope(root,config,time.time()+7199,current_run_id='derived')
                with self.assertRaises(ValueError):validate_scope(root,dict(config,maximum_attempts=2),time.time()+7199,current_run_id='derived')

    def test_derived_owner_preserves_raw_checkpoint_rng_prefix_and_frozen_history(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'source';source.mkdir();job=root/'derived';job.mkdir()
            model=torch.nn.Linear(2,1);optimizer=torch.optim.Adam(model.parameters())
            config={'hypothesis_id':'TR05','parent_model_sha256':'1'*64,'seed':1,'target_decisions':32768,
                    'ppo':{'critic_updates_shared_trunk':False},'league':{'fixed_opponents':[]},'source_manifest_sha256':'2'*64}
            spec=initialize(source,config,model,optimizer);saved=load_torch(source/'checkpoint.pt',map_location='cpu',weights_only=False)
            prefix=read_json(source/'archive-prefix.json');state=initial_state(prefix,spec['prefix_sha256'])
            shard=source/'rollouts/v0000-s000000.pt';shard.parent.mkdir();shard.write_bytes(b'exact complete pending BO3 bytes')
            summary={'series_id':0,'policy_version':'0','learner_seat':0,'rewards':[1,-1],
                'assignment':{'fixed':True,'opponent_id':'fixed'},'statuses':['DONE','DONE'],
                'trajectory_path':'rollouts/v0000-s000000.pt','trajectory_sha256':sha256(shard)}
            record_path=source/'series/000000.json';atomic_json(record_path,summary)
            record=compact_record(summary,sha256(record_path));cursor={'version':0,'trained_decisions':0,'next_series_id':1}
            state=advance_state(state,prefix,spec['prefix_sha256'],cursor,[record],{},source.name,1)
            saved.update(**cursor,archive_state=state,archive_origin_prefix_sha256=spec['prefix_sha256'])
            atomic_torch_save(source/'checkpoint.pt',saved)
            point=commit_checkpoint(source,source.name,1,0,'checkpoint.pt','archive-prefix.json',
                ['rollouts/v0000-s000000.pt','series/000000.json'],cursor)
            actor=source/'league/learner-0000.pt';actor.parent.mkdir();actor.write_bytes(b'unchanged frozen history')
            league={'snapshots':[{'path':'league/learner-0000.pt','sha256':sha256(actor),'version':0}],'fixed_training_results':{}}
            atomic_json(source/'league.json',league);atomic_json(source/'progress.json',{'stage':'checkpointed_for_control'})
            manifest={'existing_restore_source':{},'training_config':config,'reserved_future_bytes':768*2**20,
                      'source_league_sha256':sha256(source/'league.json')}
            ticket={'project_id':'test','lease_epoch':2};original={'mode':'fresh_scientific','training_config':copy.deepcopy(config)}
            before=sha256(source/point['checkpoint']['path'])
            with patch('ptcg_lab.retained_source.resolve',return_value=(source,original)),patch('ptcg_lab.retained_source.checkpoint_reference',return_value=point),patch('ptcg_lab.server_workspace.admit'):
                result=prepare(root,job,ticket,manifest,'3'*64)
            derived=read_index(job,job.name,2)['commits'][-1]
            self.assertEqual(derived['checkpoint']['sha256'],before);self.assertEqual(derived['prefix']['sha256'],point['prefix']['sha256'])
            self.assertEqual(derived['pending'],point['pending'])
            self.assertEqual(sha256(job/'rollouts/v0000-s000000.pt'),sha256(shard))
            self.assertEqual(sha256(source/point['checkpoint']['path']),before)
            self.assertEqual(result['archive_restore']['initial_checkpoint_sha256'],prefix['checkpoint']['sha256'])
            self.assertEqual(result['league']['retained_history']['members'],{'league/learner-0000.pt':sha256(actor)})
            self.assertNotIn('fresh_compact_bootstrap',result)
            self.assertEqual(os.stat(job/'checkpoint.pt').st_ino,os.stat(source/point['checkpoint']['path']).st_ino)
            # Atomic replacement of the derived hot name must preserve parent bytes.
            atomic_torch_save(job/'checkpoint.pt',dict(saved,test_marker=True))
            self.assertEqual(sha256(source/point['checkpoint']['path']),before)
            with self.assertRaises(ValueError):read_index(job,job.name,1)


if __name__=='__main__':unittest.main()
