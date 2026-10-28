from pathlib import Path
import tempfile
import unittest
import torch
from ptcg_lab.archive_prefix import compact_record,canonical_hash,fixed_statistics
from ptcg_lab.io import atomic_json,sha256,read_json
from ptcg_lab.rl_resume import resume_journal,advance_resume_state


class CompactPPOResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        old={'series_id':0,'policy_version':0,'learner_seat':0,'rewards':[-1,1],
             'assignment':{'fixed':True,'opponent_id':'fixed'},'trajectory_path':'rollouts/v0000-s000000.pt',
             'trajectory_sha256':'a'*64,'summary_sha256':'b'*64}
        tail=self.row(1,1,1)
        prefix={'schema_version':1,'checkpoint':{'version':1,'trained_decisions':10,'next_series_id':2,'sha256':'c'*64},
                'consumed':[old],'tail':[tail],'consumed_records_root':canonical_hash([old]),
                'consumed_fixed_statistics':fixed_statistics([old]),'full_fixed_statistics':fixed_statistics([old,tail])}
        atomic_json(self.root/'prefix.json',prefix)
        self.spec={'prefix_path':'prefix.json','prefix_sha256':sha256(self.root/'prefix.json'),'initial_checkpoint_sha256':'c'*64}
        self.saved={'version':1,'trained_decisions':10,'next_series_id':2}
    def tearDown(self):self.temp.cleanup()
    def row(self,sid,version,reward):
        base={'series_id':sid,'policy_version':str(version),'learner_seat':0,'rewards':[reward,-reward],
              'assignment':{'fixed':True,'opponent_id':'fixed'},'statuses':['DONE','DONE']}
        path=self.root/f'rollouts/v{version:04d}-s{sid:06d}.pt';path.parent.mkdir(exist_ok=True)
        torch.save({'series_id':sid,'policy_version':version,'decisions':[{'series_id':sid,'policy_version':str(version)}],
                    'summary':base,'rng_after':{'marker':sid}},path)
        summary=self.root/f'series/{sid:06d}.json'
        atomic_json(summary,{**base,'trajectory_sha256':sha256(path),'trajectory_path':path.relative_to(self.root).as_posix()})
        return compact_record(read_json(summary),sha256(summary))
    def initial(self):return resume_journal(self.root,self.saved,self.spec,'c'*64)
    def test_initial_compact_startup_never_reads_absent_consumed_shards(self):
        result=self.initial()
        self.assertEqual(result['shards'],['rollouts/v0001-s000001.pt'])
        self.assertEqual(result['fixed_statistics']['fixed'],{'n':2,'losses':1.})
        self.assertEqual(result['rng_after'],{'marker':1})
        self.assertFalse((self.root/'rollouts/v0000-s000000.pt').exists())
    def test_missing_durable_summary_is_recovered_only_from_complete_current_bo3(self):
        summary=self.root/'series/000001.json';before=sha256(summary);summary.unlink()
        self.assertEqual(len(self.initial()['decisions']),1);self.assertEqual(sha256(summary),before)
    def test_origin_checksum_checkpoint_and_saved_cursor_are_all_bound(self):
        with self.assertRaises(ValueError):resume_journal(self.root,self.saved,self.spec,'wrong')
        with self.assertRaises(ValueError):resume_journal(self.root,self.saved,dict(self.spec,prefix_sha256='wrong'),'c'*64)
        with self.assertRaises(ValueError):resume_journal(self.root,dict(self.saved,next_series_id=3),self.spec,'c'*64)
    def test_unacknowledged_consumed_policy_cannot_be_retrained_after_later_update(self):
        start=self.initial();prefix=start['origin_prefix']
        state=advance_resume_state(self.root,start['archive_state'],prefix,self.spec,
            {'version':2,'trained_decisions':11,'next_series_id':2},'run',1)
        self.row(2,2,-1)
        state=advance_resume_state(self.root,state,prefix,self.spec,
            {'version':3,'trained_decisions':12,'next_series_id':3},'run',1)
        self.assertEqual(state['archive_cut_version'],2)
        self.row(3,3,1)
        saved={'version':3,'trained_decisions':12,'next_series_id':3,'archive_state':state,
               'archive_origin_prefix_sha256':self.spec['prefix_sha256']}
        result=resume_journal(self.root,saved,self.spec,'later-checkpoint')
        self.assertEqual(result['shards'],['rollouts/v0003-s000003.pt'])
        self.assertEqual(result['next_series_id'],4)
        self.assertEqual(result['fixed_statistics']['fixed']['n'],4)
        self.assertTrue((self.root/'rollouts/v0002-s000002.pt').exists())
        with self.assertRaises(ValueError):resume_journal(self.root,saved,None)
    def test_decision_identity_and_pending_bytes_cannot_be_silently_changed(self):
        path=self.root/'rollouts/v0001-s000001.pt';data=torch.load(path,weights_only=False)
        data['decisions'][0]['series_id']=999;torch.save(data,path)
        with self.assertRaises(ValueError):self.initial()
    def test_wrong_lease_ack_cannot_advance_compact_history(self):
        result=self.initial();self.row(2,1,0)
        atomic_json(self.root/'archive-acks/wrong.json',{'run_id':'other','lease_epoch':9,'sequence':0,
            'object':'rollouts/v0001-s000002.pt','sha256':'x','bytes':1,'archive_relative':'other','commit_id':'forged'})
        with self.assertRaises(ValueError):advance_resume_state(self.root,result['archive_state'],result['origin_prefix'],self.spec,
            {'version':2,'trained_decisions':12,'next_series_id':3},'run',1)
