from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from ptcg_lab.archive_prefix import compact_record,canonical_hash,fixed_statistics
from ptcg_lab.archive_state import initial_state
from ptcg_lab.checkpoint_refs import read_index
from ptcg_lab.io import atomic_json,read_json,sha256
from ptcg_lab.training_checkpoint import initialize_recovery,publish_training_checkpoint,load_training_checkpoint


class TrainingCheckpointTests(unittest.TestCase):
    def test_self_consistent_index_cannot_omit_payload_pending_references(self):
        self.initial();publish_training_checkpoint(self.root,self.value(1),self.spec,1)
        path=self.root/'checkpoint-commits/index.json';index=read_json(path);latest=index['commits'][-1]
        latest['pending']=[];latest['commit_id']=canonical_hash({k:v for k,v in latest.items() if k!='commit_id'})
        atomic_json(path,index)
        with self.assertRaisesRegex(ValueError,'live-tail reference'):load_training_checkpoint(self.root,self.spec,1)
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)/'run';self.root.mkdir()
        self.saved={'model':{'actor':torch.arange(8)},'optimizer':{'state':{}},'rng':{'test_rng':'unchanged'},
                    'version':1,'trained_decisions':10,'next_series_id':1,'updates':[]}
        torch.save(self.saved,self.root/'checkpoint.pt')
        path=self.root/'rollouts/v0001-s000000.pt';path.parent.mkdir();path.write_bytes(b'complete-committed-shard')
        summary={'series_id':0,'policy_version':'1','learner_seat':0,'rewards':[0,0],
                 'assignment':{'fixed':True,'opponent_id':'frozen'},'statuses':['DONE','DONE'],
                 'trajectory_path':path.relative_to(self.root).as_posix(),'trajectory_sha256':sha256(path)}
        s=self.root/'series/000000.json';atomic_json(s,summary);row=compact_record(summary,sha256(s))
        self.prefix={'schema_version':1,'checkpoint':{**{k:self.saved[k] for k in ['version','trained_decisions','next_series_id']},'sha256':sha256(self.root/'checkpoint.pt')},
                     'consumed':[],'tail':[row],'consumed_records_root':canonical_hash([]),
                     'consumed_fixed_statistics':{},'full_fixed_statistics':fixed_statistics([row])}
        atomic_json(self.root/'prefix.json',self.prefix)
        self.spec={'prefix_path':'prefix.json','prefix_sha256':sha256(self.root/'prefix.json'),'initial_checkpoint_sha256':sha256(self.root/'checkpoint.pt')}
        self.state=initial_state(self.prefix,self.spec['prefix_sha256'])
    def tearDown(self):self.temp.cleanup()
    def initial(self):return initialize_recovery(self.root,self.saved,self.spec,self.state,1)
    def value(self,marker):return {**self.saved,'archive_state':self.state,'archive_origin_prefix_sha256':self.spec['prefix_sha256'],'test_marker':marker}
    def test_two_published_points_pin_both_trajectory_and_complete_summary(self):
        self.initial();publish_training_checkpoint(self.root,self.value(1),self.spec,1)
        publish_training_checkpoint(self.root,self.value(2),self.spec,1)
        index=read_index(self.root,'run',1);self.assertEqual([c['generation'] for c in index['commits']],[1,2])
        for commit in index['commits']:
            self.assertEqual({r['path'] for r in commit['pending']},{'rollouts/v0001-s000000.pt','series/000000.json'})
        loaded=load_training_checkpoint(self.root,self.spec,1)
        self.assertEqual(loaded['saved']['test_marker'],2)
        self.assertTrue(loaded['archive_spec']['prefix_path'].startswith('checkpoint-commits/objects/'))
    def test_failed_index_commit_cannot_make_new_hot_payload_authoritative(self):
        self.initial();publish_training_checkpoint(self.root,self.value(1),self.spec,1)
        with patch('ptcg_lab.checkpoint_refs.atomic_json',side_effect=OSError('injected-index-write-failure')):
            with self.assertRaises(OSError):publish_training_checkpoint(self.root,self.value(2),self.spec,1)
        self.assertEqual(torch.load(self.root/'checkpoint.pt',weights_only=False)['test_marker'],2)
        loaded=load_training_checkpoint(self.root,self.spec,1)
        self.assertEqual(loaded['saved']['test_marker'],1);self.assertEqual(loaded['recovery']['generation'],1)
        self.assertTrue(torch.equal(loaded['saved']['model']['actor'],self.saved['model']['actor']))
    def test_epoch_cursor_or_changed_pending_content_fails_closed(self):
        self.initial()
        with self.assertRaises(ValueError):load_training_checkpoint(self.root,self.spec,2)
        bad=self.value(1);bad['next_series_id']=2
        with self.assertRaises(ValueError):publish_training_checkpoint(self.root,bad,self.spec,1)
        (self.root/'rollouts/v0001-s000000.pt').write_bytes(b'changed')
        with self.assertRaises(ValueError):load_training_checkpoint(self.root,self.spec,1)
    def test_compact_hot_checkpoint_without_published_index_is_rejected(self):
        torch.save(self.value(0),self.root/'checkpoint.pt')
        with self.assertRaisesRegex(ValueError,'no published recovery index'):load_training_checkpoint(self.root,self.spec,1)
    def test_initial_bootstrap_cannot_silently_replace_frozen_parent(self):
        (self.root/'checkpoint.pt').write_bytes(b'wrong-parent')
        with self.assertRaises(ValueError):self.initial()
