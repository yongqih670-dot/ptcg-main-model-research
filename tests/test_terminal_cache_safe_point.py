from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from ptcg_lab.io import atomic_json,sha256,read_json
from ptcg_lab.archive_ack import receive_object
from ptcg_lab.checkpoint_refs import commit_checkpoint
from ptcg_lab.public_fragment_retirement import finish_terminal_consumed_retirement


class TerminalCacheSafePointTests(unittest.TestCase):
    def test_original_ACK_and_both_published_points_remain_required(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'source';source.mkdir();job=root/'check';job.mkdir()
            (source/'checkpoint.pt').write_bytes(b'exact complete checkpoint')
            (source/'prefix.json').write_bytes(b'exact original prefix')
            cursor={'version':2,'trained_decisions':16646,'next_series_id':102}
            for generation in [0,1]:commit_checkpoint(source,source.name,40,generation,'checkpoint.pt','prefix.json',[],cursor)
            target=source/'rollouts/v0001-s000061.pt';target.parent.mkdir();target.write_bytes(b'exact completed consumed trajectory')
            ack=receive_object(root/'archives',source.name,40,0,'rollouts/v0001-s000061.pt',sha256(target),target.stat().st_size,[target.read_bytes()])
            path=source/'archive-acks/00000000.json';atomic_json(path,ack)
            before=sha256(source/'checkpoint-commits/index.json')
            spec={'lease_epoch':40};proof={}
            with patch('ptcg_lab.retained_source.resolve',return_value=(source,{})),patch('ptcg_lab.scientific_resume.validate_postcommit_recovery'):
                bad={**ack,'commit_id':'0'*64}
                with self.assertRaises(ValueError):finish_terminal_consumed_retirement(root,job,'owned',spec,proof,[bad])
                self.assertTrue(target.exists())
                finish_terminal_consumed_retirement(root,job,'owned',spec,proof,[ack])
            self.assertFalse(target.exists());self.assertEqual(sha256(source/'checkpoint-commits/index.json'),before)
            self.assertEqual(len(read_json(source/'checkpoint-commits/index.json')['commits']),2)
            self.assertTrue((source/'checkpoint.pt').exists());self.assertTrue((source/'prefix.json').exists())


if __name__=='__main__':unittest.main()
