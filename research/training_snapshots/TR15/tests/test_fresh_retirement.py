from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from ptcg_lab.fresh_experiment import retire_caches
from ptcg_lab.io import read_json,sha256


class FreshRetirementTests(unittest.TestCase):
    def fixture(self,root):
        source=root/'old';source.mkdir();job=root/'new';job.mkdir();objects=[]
        for name in ['rollouts/v0000-s000001.pt','checkpoint.pt']:
            path=source/name;path.parent.mkdir(exist_ok=True);path.write_bytes(b'durable-original')
            st=path.stat();objects.append({'path':name,'sha256':sha256(path),'bytes':st.st_size,
                'file_identity':[st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns]})
        (source/'metadata.json').write_text('{}');return source,job,[{'source':{'run_id':'old'},'objects':objects}]
    def test_only_hash_identity_verified_explicit_terminal_replicas_retire(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source,job,plan=self.fixture(root)
            with patch('ptcg_lab.retained_source.resolve',return_value=(source,{})):retire_caches(root,job,'project',plan)
            self.assertTrue((source/'metadata.json').exists());self.assertFalse((source/'checkpoint.pt').exists())
            self.assertEqual(read_json(job/'terminal-cache-retirement.json')['status'],'retired')
    def test_any_mismatch_or_escaped_target_prevents_all_deletion(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source,job,plan=self.fixture(root);plan[0]['objects'][1]['sha256']='0'*64
            with patch('ptcg_lab.retained_source.resolve',return_value=(source,{})):
                with self.assertRaises(ValueError):retire_caches(root,job,'project',plan)
            self.assertTrue((source/'checkpoint.pt').exists());self.assertTrue((source/'rollouts/v0000-s000001.pt').exists())
