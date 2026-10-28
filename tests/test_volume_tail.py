import copy,tempfile,unittest
from pathlib import Path
import torch
from ptcg_lab.io import atomic_json,sha256
from ptcg_lab.archive_prefix import canonical_hash
from ptcg_lab.archive_state import initial_state
from ptcg_lab.volume_tail import validate_spec,verify_journal

class VolumeTailTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        prefix={'schema_version':1,'checkpoint':{'version':0,'trained_decisions':0,'next_series_id':0,'sha256':'c'*64},
            'consumed':[],'tail':[],'consumed_records_root':canonical_hash([]),'consumed_fixed_statistics':{},'full_fixed_statistics':{}}
        atomic_json(self.root/'prefix.json',prefix);ph=sha256(self.root/'prefix.json')
        self.arc={'prefix_path':'prefix.json','prefix_sha256':ph,'initial_checkpoint_sha256':'c'*64}
        self.saved={'version':0,'trained_decisions':0,'next_series_id':0,'archive_state':initial_state(prefix,ph),'archive_origin_prefix_sha256':ph}
        summary={'series_id':0,'policy_version':'0','learner_seat':0,'rewards':[1,-1],'assignment':{'fixed':True,'opponent_id':'fixed'},'statuses':['DONE','DONE']}
        shard=self.root/'rollouts/v0000-s000000.pt';shard.parent.mkdir()
        torch.save({'series_id':0,'policy_version':0,'decisions':[{'series_id':0,'policy_version':'0'}],
            'summary':summary,'rng_after':{'marker':5},'cpu_sampler_rng_after':torch.Generator().get_state()},shard)
        atomic_json(self.root/'series/000000.json',{**summary,'trajectory_path':shard.relative_to(self.root).as_posix(),'trajectory_sha256':sha256(shard)})
        atomic_json(self.root/'inflight.json',{'series_id':1})
        self.point={'checkpoint':{'sha256':'c'*64},'cursor':{'version':0,'trained_decisions':0,'next_series_id':0}}
        self.spec={'source_run_id':'source','checkpoint_sha256':'c'*64,'series_ids':[0],'pending_decisions':1,
            'journal_next_series_id':1,'next_series_id_after_interruption_exclusion':2,'inflight_sha256':sha256(self.root/'inflight.json'),
            'references':[{'path':p,'sha256':sha256(self.root/p),'bytes':(self.root/p).stat().st_size} for p in ('rollouts/v0000-s000000.pt','series/000000.json')]}
    def tearDown(self):self.temp.cleanup()
    def test_post_checkpoint_tail_rng_and_interrupted_cursor_preserved(self):
        validate_spec(self.spec,'source',self.point)
        result=verify_journal(self.root,self.saved,self.arc,'c'*64,self.spec)
        self.assertEqual(result['rng_after'],{'marker':5});self.assertEqual(len(result['decisions']),1)
        self.assertTrue((self.root/'inflight.json').exists());self.assertEqual(self.saved['next_series_id'],0)
    def test_missing_pair_wrong_source_and_checkpoint_rejected(self):
        for spec in ({**self.spec,'references':self.spec['references'][:1]}, {**self.spec,'source_run_id':'other'}, {**self.spec,'checkpoint_sha256':'d'*64}):
            with self.assertRaises(ValueError):validate_spec(spec,'source',self.point)
    def test_tail_bytes_or_effective_cursor_cannot_silently_change(self):
        with self.assertRaises(ValueError):verify_journal(self.root,self.saved,self.arc,'c'*64,{**self.spec,'pending_decisions':2})
        with self.assertRaises(ValueError):verify_journal(self.root,self.saved,self.arc,'c'*64,{**self.spec,'next_series_id_after_interruption_exclusion':1})
        (self.root/'rollouts/v0000-s000000.pt').write_bytes(b'corrupt')
        with self.assertRaises(ValueError):verify_journal(self.root,self.saved,self.arc,'c'*64,self.spec)
