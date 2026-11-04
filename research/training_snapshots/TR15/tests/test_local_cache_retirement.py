import os
from pathlib import Path
import tempfile
import time
import unittest
import zipfile
import hashlib
from ptcg_lab.local_cache_retirement import windows_security,owned,verify_members,LocalReadPacer
from ptcg_lab.io_pacing import BytePacer


@unittest.skipUnless(os.name=='nt','Actual Windows OS owner checks')
class LocalCacheRetirementTests(unittest.TestCase):
    def test_local_pacer_is_bounded_and_does_not_change_server_allowance(self):
        LocalReadPacer(16*2**20,time.time()+5).consume(1)
        with self.assertRaises(ValueError):LocalReadPacer(32*2**20,time.time()+5)
        from ptcg_lab.io_pacing import BytePacer
        with self.assertRaises(ValueError):BytePacer(16*2**20,time.time()+5)
    def test_actual_os_ownership_and_full_wheel_members_before_any_unlink(self):
        sid,owner=windows_security()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();cache=root/'cache';cache.mkdir();(cache/'lib.so').write_bytes(b'original tensor library')
            wheel=root/'package.whl'
            with zipfile.ZipFile(wheel,'w') as z:z.writestr('lib.so',b'original tensor library')
            result=verify_members(root,cache,wheel,sid,owner,BytePacer(1024**2,time.time()+10))
            self.assertEqual(len(result),1);self.assertEqual(result[0]['bytes'],23)
            self.assertTrue((cache/'lib.so').exists())
            (cache/'lib.so').write_bytes(b'corrupted tensor data!!')
            with self.assertRaises(ValueError):verify_members(root,cache,wheel,sid,owner,BytePacer(1024**2,time.time()+10))
            self.assertTrue((cache/'lib.so').exists())

    def test_owner_escape_and_extra_file_fail_without_deletion(self):
        sid,owner=windows_security()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();cache=root/'cache';cache.mkdir();wheel=root/'package.whl'
            with zipfile.ZipFile(wheel,'w') as z:z.writestr('a.txt',b'a')
            (cache/'a.txt').write_bytes(b'a');(cache/'other.txt').write_bytes(b'not a package member')
            with self.assertRaises(ValueError):verify_members(root,cache,wheel,sid,owner,BytePacer(1024**2,time.time()+10))
            with self.assertRaises(ValueError):owned(root,root/'../foreign',sid,owner)
            with self.assertRaises(ValueError):owned(root,cache,sid,lambda _:sid+'-not-owner')
            self.assertEqual(len(list(cache.iterdir())),2)

    def test_only_exact_registered_Codex_runs_container_owner_may_differ(self):
        sid,owner=windows_security()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();target=root/'runs/cache';target.mkdir(parents=True)
            def known(path):return 'known-codex-token' if path==root/'runs' else owner(path)
            with self.assertRaises(ValueError):owned(root,target,sid,known)
            with self.assertRaises(ValueError):owned(root,target,sid,known,'0'*64)
            expected=hashlib.sha256(b'known-codex-token').hexdigest()
            self.assertEqual(owned(root,target,sid,known,expected),target)


if __name__=='__main__':unittest.main()
