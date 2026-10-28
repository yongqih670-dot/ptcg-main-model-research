import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from ptcg_lab.archive_ack import receive_object,may_evict,validate_ack
from ptcg_lab.archive_prefix import canonical_hash,compact_record,fixed_statistics,validate_prefix,restored_statistics
from ptcg_lab.io import atomic_json,sha256


class ArchiveProtocolTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def series(self,sid,version,reward):
        shard=self.root/f'rollouts/v{version:04d}-s{sid:06d}.pt';shard.parent.mkdir(exist_ok=True);shard.write_bytes(f'shard{sid}'.encode())
        row={'series_id':sid,'policy_version':str(version),'learner_seat':0,'rewards':[reward,-reward],
             'statuses':['DONE','DONE'],'assignment':{'fixed':True,'opponent_id':'fixed'},
             'trajectory_path':shard.relative_to(self.root).as_posix(),'trajectory_sha256':sha256(shard)}
        path=self.root/f'series/{sid:06d}.json';atomic_json(path,row)
        return compact_record(row,sha256(path)),path,shard
    def prefix(self,consumed,tail):
        return {'schema_version':1,'checkpoint':{'version':2,'next_series_id':3},'consumed':consumed,'tail':tail,
                'consumed_records_root':canonical_hash(consumed),'consumed_fixed_statistics':fixed_statistics(consumed),
                'full_fixed_statistics':fixed_statistics(consumed+tail)}
    def test_consumed_prefix_plus_live_tail_matches_full_result_without_old_shards(self):
        a,pa,sa=self.series(0,0,-1);b,pb,sb=self.series(1,1,0);c,pc,sc=self.series(2,2,1)
        p=self.prefix([a,b],[c]);expected=fixed_statistics([a,b,c])
        # Simulated server lacks old bytes; local full archive would retain them.
        pa.unlink();sa.unlink();pb.unlink();sb.unlink()
        self.assertEqual(restored_statistics(self.root,p),expected)
        self.assertEqual(expected,{'fixed':{'n':3,'losses':1.5}})
    def test_tail_loss_corruption_and_double_count_fail_closed(self):
        a,pa,sa=self.series(0,0,-1);c,pc,sc=self.series(2,2,1);p=self.prefix([a],[c])
        with self.assertRaisesRegex(ValueError,'duplicated'):restored_statistics(self.root,p)
        pa.unlink();sa.unlink();sc.write_bytes(b'corrupted')
        with self.assertRaisesRegex(ValueError,'Live trajectory changed'):restored_statistics(self.root,p)
        pc.unlink()
        with self.assertRaisesRegex(ValueError,'Missing unconsumed'):restored_statistics(self.root,p)
    def test_reclassified_unconsumed_data_and_changed_prefix_rejected(self):
        a,_,_=self.series(0,2,1)
        with self.assertRaisesRegex(ValueError,'Unconsumed'):validate_prefix(self.prefix([a],[]))
        a['policy_version']=0;p=self.prefix([a],[]);p['consumed'][0]['rewards']=[-1,1]
        with self.assertRaises(ValueError):validate_prefix(p)
    def test_receive_is_durable_idempotent_and_bound_to_epoch_sequence_and_hash(self):
        data=b'checkpoint';digest=hashlib.sha256(data).hexdigest()
        ack=receive_object(self.root,'run',3,1,'checkpoint.pt',digest,len(data),[data[:3],data[3:]])
        self.assertEqual(receive_object(self.root,'run',3,1,'checkpoint.pt',digest,len(data),[]),ack)
        self.assertEqual((self.root/ack['archive_relative']).read_bytes(),data)
        for epoch,name,sha in [(4,'checkpoint.pt',digest),(3,'other.pt',digest),(3,'checkpoint.pt','0'*64)]:
            with self.assertRaises(ValueError):validate_ack(ack,'run',epoch,name,sha,len(data))
        with self.assertRaises(ValueError):receive_object(self.root,'run',3,1,'other.pt',digest,len(data),[data])
    def test_interrupted_or_corrupt_transfer_gets_no_ack(self):
        digest=hashlib.sha256(b'correct').hexdigest()
        with self.assertRaises(ValueError):receive_object(self.root,'run',1,0,'tail.pt',digest,7,[b'bad'])
        self.assertFalse((self.root/'run/acks/00000000.json').exists())
        self.assertFalse((self.root/'run/objects/tail.pt').exists())
        self.assertFalse(list(self.root.rglob('.incoming-*')))
    def test_ack_does_not_allow_evicting_any_live_checkpoint_or_opponent_reference(self):
        data=b'rollout';digest=hashlib.sha256(data).hexdigest()
        ack=receive_object(self.root,'run',1,0,'tail.pt',digest,len(data),[data])
        self.assertFalse(may_evict(ack,'run',1,'tail.pt',digest,len(data),[digest]))
        self.assertTrue(may_evict(ack,'run',1,'tail.pt',digest,len(data),[]))
        with self.assertRaises(ValueError):may_evict(ack,'run',2,'tail.pt',digest,len(data),[])
    def test_archive_traversal_cannot_publish_or_ack(self):
        with self.assertRaises(ValueError):receive_object(self.root,'run',1,0,'../outside','0'*64,0,[])
    def test_low_disk_and_relocated_receipt_cannot_be_acknowledged(self):
        data=b'data';digest=hashlib.sha256(data).hexdigest()
        with patch('ptcg_lab.archive_ack.shutil.disk_usage',return_value=SimpleNamespace(free=10)):
            with self.assertRaises(OSError):receive_object(self.root,'run',1,0,'x',digest,len(data),[data],minimum_free_bytes=8)
        self.assertFalse(list(self.root.rglob('*.json')))
        ack=receive_object(self.root,'run',1,0,'x',digest,len(data),[data])
        atomic_json(self.root/'run/acks/00000001.json',ack)
        with self.assertRaisesRegex(ValueError,'sequence'):receive_object(self.root,'run',1,1,'x',digest,len(data),[])


if __name__=='__main__':unittest.main()
