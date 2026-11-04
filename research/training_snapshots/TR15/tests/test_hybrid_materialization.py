import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
import psutil
from unittest.mock import patch
from ptcg_lab.archive_compressed import receive_compressed_object
from ptcg_lab.hybrid_materialization import reconstruct,run,owned_runner_process


class HybridMaterializationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.source=self.root/'training';self.source.mkdir()
        self.output=self.root/'audit';self.output.mkdir()
        self.data=b'exact original model bytes'*1024
        self.digest=hashlib.sha256(self.data).hexdigest()
        self.archive=self.source/'online-archive'
        self.ack=receive_compressed_object(self.archive,'training',55,0,'model.pt',self.digest,
            len(self.data),[self.data],minimum_free_bytes=0)
        self.metrics={'model_sha256':self.digest};self.lease={'lease_epoch':55}
    def tearDown(self):self.temp.cleanup()
    def materialize(self,**kwargs):
        return reconstruct(self.source,self.output,self.metrics,self.lease,time.time()+60,
            free_reserve=0,growth_allowance=kwargs.get('growth',32*2**20))
    def test_original_bytes_reconstructed_with_all_receipts_and_idempotent_proof(self):
        proof=self.materialize();self.assertEqual((self.source/'model.pt').read_bytes(),self.data)
        path=self.source/'hybrid-materialization.json';original=path.read_bytes()
        self.materialize();self.assertEqual(path.read_bytes(),original)
        self.assertEqual(proof['all_archive_receipts_reconstructed_and_verified'],1)
    def test_corrupted_archive_cannot_publish_a_candidate(self):
        (self.archive/self.ack['archive_relative']).write_bytes(b'corrupt')
        with self.assertRaises(ValueError):self.materialize()
        self.assertFalse((self.source/'model.pt').exists())
    def test_foreign_lease_and_storage_exhaustion_are_rejected(self):
        self.lease['lease_epoch']=56
        with self.assertRaises(ValueError):self.materialize()
        self.lease['lease_epoch']=55
        with self.assertRaises(OSError):self.materialize(growth=0)
        self.assertFalse((self.source/'model.pt').exists())
    def test_conflicting_model_never_overwritten(self):
        (self.source/'model.pt').write_bytes(b'other')
        with self.assertRaises(ValueError):self.materialize()
        self.assertEqual((self.source/'model.pt').read_bytes(),b'other')
    def archive_series(self, identity, rounds, statuses=None):
        data=json.dumps({'series_id':identity,'event':'series_end',
            'statuses':statuses or ['DONE','DONE'],'rounds':rounds,
            'learner_decisions':100+identity}).encode()
        receive_compressed_object(self.archive,'training',55,identity+1,f'series/{identity:06d}.json',
            hashlib.sha256(data).hexdigest(),len(data),[data],minimum_free_bytes=0)
    def test_retired_raw_files_do_not_erase_archived_bo3_or_game_counts(self):
        self.archive_series(7,2);self.archive_series(8,3)
        self.assertFalse((self.source/'series').exists())
        counts=self.materialize()['archived_series_accounting']
        self.assertEqual((counts['complete_BO3'],counts['complete_games'],counts['sampled_learner_decisions']), (2,5,215))
    def test_unfinished_series_does_not_count_as_complete(self):
        self.archive_series(7,2,['DONE','ERROR'])
        with self.assertRaisesRegex(ValueError,'BO3 summary'):
            self.materialize()
        self.assertFalse((self.source/'model.pt').exists())
    def test_own_launcher_ancestor_allowed_but_reused_pid_or_foreign_job_rejected(self):
        parent=psutil.Process().parent()
        owner={'pid':parent.pid,'created':parent.create_time(),'run_id':'own'}
        self.assertTrue(owned_runner_process(owner,'own'))
        self.assertFalse(owned_runner_process(owner,'foreign'))
        owner['created']-=1
        self.assertFalse(owned_runner_process(owner,'own'))
    def test_expired_deadline_and_uncontrolled_entry_rejected(self):
        with self.assertRaises(TimeoutError):
            reconstruct(self.source,self.output,self.metrics,self.lease,time.time(),free_reserve=0,growth_allowance=32*2**20)
        with self.assertRaises(RuntimeError):run({'project_path':str(self.root)},self.output)

    def test_incomplete_training_rejected_by_controlled_entry(self):
        runs=self.root/'runs';runs.mkdir();output=runs/'controlled';output.mkdir()
        source=runs/'candidate';source.mkdir()
        autonomy=self.root/'research/autonomy';autonomy.mkdir(parents=True)
        (autonomy/'state.json').write_text(json.dumps({'active_run_id':'controlled',
            'active_job':{'run_id':'controlled','pid':os.getpid(),'created':psutil.Process().create_time()}}),encoding='utf-8')
        records={'metrics.json':{'complete':False,'scientific_training':True},
                 'config.json':{'job_kind':'remote_train_ppo','check_variant':'derived_scientific'},
                 'remote-job.json':{'run_id':'candidate','lease_epoch':55}}
        hashes={}
        for name,value in records.items():
            path=source/name;path.write_text(json.dumps(value),encoding='utf-8')
            hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
        config={'project_path':str(self.root),'maximum_attempts':1,'training_run_id':'candidate',
                'source_evidence_sha256':hashes}
        with patch.dict(os.environ,{'PTCG_JOB_RUN_ID':'controlled','PTCG_JOB_DEADLINE_UNIX':str(time.time()+60)}):
            with self.assertRaisesRegex(ValueError,'Only completed'):
                run(config,output)
        self.assertFalse((source/'model.pt').exists())


if __name__=='__main__':unittest.main()
