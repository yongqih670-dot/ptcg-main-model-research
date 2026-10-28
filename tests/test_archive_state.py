from pathlib import Path
import tempfile
import unittest
from ptcg_lab.archive_state import initial_state,advance_state,restore_live_state
from ptcg_lab.archive_prefix import canonical_hash,compact_record,fixed_statistics
from ptcg_lab.archive_ack import receive_object
from ptcg_lab.io import atomic_json,sha256


class RollingArchiveTests(unittest.TestCase):
    def test_archived_summary_after_partial_replica_removal_can_still_checkpoint(self):
        from ptcg_lab.rl_resume import advance_resume_state
        (self.remote/self.a['trajectory_path']).unlink()
        updated=advance_resume_state(self.remote,self.state,self.origin,{'prefix_sha256':'origin-hash'},
            self.state['checkpoint_cursor'],'run',1)
        self.assertEqual(updated['fixed_statistics'],self.state['fixed_statistics'])
        path=self.remote/'series/000000.json';changed=__import__('ptcg_lab.io',fromlist=['read_json']).read_json(path)
        changed['rewards']=[1,-1];atomic_json(path,changed)
        with self.assertRaises(ValueError):advance_resume_state(self.remote,self.state,self.origin,{'prefix_sha256':'origin-hash'},
            self.state['checkpoint_cursor'],'run',1)
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.remote=self.root/'remote';self.remote.mkdir()
        self.a=self.row(0,0,-1);self.b=self.row(1,1,1)
        self.origin={'schema_version':1,'checkpoint':{'version':1,'trained_decisions':10,'next_series_id':2},'consumed':[self.a],'tail':[self.b],
                     'consumed_records_root':canonical_hash([self.a]),'consumed_fixed_statistics':fixed_statistics([self.a]),'full_fixed_statistics':fixed_statistics([self.a,self.b])}
        self.state=initial_state(self.origin,'origin-hash')
    def tearDown(self):self.tmp.cleanup()
    def row(self,sid,version,reward):
        p=self.remote/f'rollouts/v{version:04d}-s{sid:06d}.pt';p.parent.mkdir(exist_ok=True);p.write_bytes(f'update-data-{sid}'.encode())
        row={'series_id':sid,'policy_version':str(version),'statuses':['DONE','DONE'],'learner_seat':0,'rewards':[reward,-reward],
             'assignment':{'fixed':True,'opponent_id':'frozen'},'trajectory_path':p.relative_to(self.remote).as_posix(),'trajectory_sha256':sha256(p)}
        s=self.remote/f'series/{sid:06d}.json';atomic_json(s,row);return compact_record(row,sha256(s))
    def ack(self,row):
        result={}
        for sequence,name in enumerate([row['trajectory_path'],f"series/{row['series_id']:06d}.json"]):
            p=self.remote/name;result[name]=receive_object(self.root/'localarchive','run',1,row['series_id']*2+sequence,name,sha256(p),p.stat().st_size,[p.read_bytes()])
        return result
    def advance(self,state,version,nextid,rows,acks):
        return advance_state(state,self.origin,'origin-hash',{'version':version,'trained_decisions':version*10,'next_series_id':nextid},rows,acks,'run',1)
    def test_slow_ack_can_lag_model_but_old_policy_is_never_retrained(self):
        c=self.row(2,2,0);d=self.row(3,3,-1)
        s=self.advance(self.state,2,3,[c],{})
        s=self.advance(s,3,4,[d],{})
        self.assertEqual(s['archive_cut_version'],2)
        restored=restore_live_state(self.remote,s,self.origin,'origin-hash')
        self.assertEqual(restored['on_policy_shards'],[d['trajectory_path']])
        self.assertIn(c['trajectory_path'],restored['journal_reference_paths'])
        s=self.advance(s,3,4,[],self.ack(c))
        self.assertEqual(s['archive_cut_version'],3)
        self.assertEqual(restore_live_state(self.remote,s,self.origin,'origin-hash')['fixed_statistics'],fixed_statistics([self.a,self.b,c,d]))
    def test_trajectory_ack_alone_does_not_advance_past_unarchived_summary(self):
        c=self.row(2,2,0);d=self.row(3,3,1);s=self.advance(self.state,2,3,[c],{})
        acks=self.ack(c);del acks[f"series/{c['series_id']:06d}.json"]
        s=self.advance(s,3,4,[d],acks);self.assertEqual(s['archive_cut_version'],2)
    def test_changed_epoch_conflicting_row_or_missing_tail_fail_closed(self):
        c=self.row(2,2,0);s=self.advance(self.state,2,3,[c],{});acks=self.ack(c)
        for ack in acks.values():ack['lease_epoch']=2
        with self.assertRaises(ValueError):self.advance(s,3,3,[],acks)
        changed=dict(c,rewards=[1,-1])
        with self.assertRaises(ValueError):self.advance(s,2,3,[changed],{})
        (self.remote/f"series/{c['series_id']:06d}.json").unlink()
        with self.assertRaises(ValueError):restore_live_state(self.remote,s,self.origin,'origin-hash')
    def test_origin_records_cannot_be_removed_to_improve_statistics(self):
        bad=dict(self.state,archived=[],state_id='forged')
        with self.assertRaises(ValueError):self.advance(bad,2,2,[],{})
