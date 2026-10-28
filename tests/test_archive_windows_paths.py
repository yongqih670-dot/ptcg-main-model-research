import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from ptcg_lab.archive_ack import receive_object,publication_path
from ptcg_lab.archive_compressed import receive_compressed_object,verify_archive


@unittest.skipUnless(os.name=='nt','Actual Windows file publication')
class WindowsArchivePaths(unittest.TestCase):
    def test_long_files_remain_in_the_storage_budget(self):
        import scripts.run_experiment as runner
        with tempfile.TemporaryDirectory(dir=publication_path(tempfile.gettempdir())) as folder:
            root=Path(folder);path=root/'runs'/('p'*140)/('file-'+'a'*95)
            path.parent.mkdir(parents=True);path.write_bytes(b'full original bytes'*1000)
            ordinary=Path(str(root).removeprefix('\\\\?\\'))
            with patch.object(runner,'ROOT',ordinary):
                self.assertEqual(runner.artifact_gib(),path.stat().st_size/2**30)

    def test_real_long_RAW_and_gzip_paths_preserve_hash_receipt_and_idempotence(self):
        with tempfile.TemporaryDirectory(dir=publication_path(tempfile.gettempdir())) as folder:
            root=Path(folder)/('r'*95);root.mkdir()
            data=b'complete original trajectory bytes'*1000;digest=hashlib.sha256(data).hexdigest()
            obj='checkpoint-commits/objects/prefix-'+('a'*64)+'.json'
            for i,receiver in enumerate([receive_object,receive_compressed_object]):
                rid='controlled-scientific-run-'+str(i)*35
                target=root/rid/'objects'/obj
                self.assertGreater(len(str(target)),280)
                ack=receiver(root,rid,1,i,obj,digest,len(data),[data],minimum_free_bytes=0)
                self.assertTrue(verify_archive(root,ack))
                self.assertEqual(receiver(root,rid,1,i,obj,digest,len(data),[],minimum_free_bytes=0),ack)
                self.assertTrue(str(publication_path(root)).startswith('\\\\?\\'))
