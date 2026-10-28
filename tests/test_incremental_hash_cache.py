import hashlib
import os
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

from ptcg_lab.checkpoint_refs import checked_reference
from ptcg_lab import incremental_hash_cache as module
from ptcg_lab.io import sha256


class IncrementalHashCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name).resolve()
        self.support=patch.object(module,'_linux_runtime_supported',return_value=True)
        self.uid=patch.object(module,'_getuid',return_value=self.root.stat().st_uid)
        self.support.start();self.uid.start()

    def tearDown(self):
        self.uid.stop();self.support.stop();self.temp.cleanup()

    def test_same_immutable_reference_reuses_full_hash_without_exposing_mutable_dict(self):
        (self.root/'a').write_bytes(b'original');cache=module.ImmutableReferenceHashCache(self.root)
        with patch('ptcg_lab.checkpoint_refs.sha256',wraps=sha256) as full:
            a=checked_reference(self.root,'a',cache);a['sha256']='changed return value'
            b=checked_reference(self.root,'a',cache)
            self.assertEqual(full.call_count,1)
            self.assertEqual(b['sha256'],hashlib.sha256(b'original').hexdigest())

    def test_change_time_invalidates_same_inode_size_and_restored_mtime(self):
        path=self.root/'a';path.write_bytes(b'aaaa');old=path.stat()
        cache=module.ImmutableReferenceHashCache(self.root);counter=[0];actual=module._fingerprint
        # Windows birth time is not Linux ctime. Emulate only the ctime event;
        # the production route still requires actual Linux acceptance.
        def observed(p):return (*actual(p)[:6],counter[0])
        with patch.object(module,'_fingerprint',side_effect=observed),patch('ptcg_lab.checkpoint_refs.sha256',wraps=sha256) as full:
            first=checked_reference(self.root,'a',cache)
            path.write_bytes(b'bbbb');os.utime(path,ns=(old.st_atime_ns,old.st_mtime_ns));counter[0]+=1
            second=checked_reference(self.root,'a',cache)
            self.assertNotEqual(first['sha256'],second['sha256']);self.assertEqual(full.call_count,2)

    def test_same_metadata_write_during_full_hash_is_rejected(self):
        path=self.root/'a';path.write_bytes(b'aaaa');old=path.stat();counter=[0];actual=module._fingerprint
        cache=module.ImmutableReferenceHashCache(self.root)
        def changed(p):
            digest=sha256(p);path.write_bytes(b'bbbb');os.utime(path,ns=(old.st_atime_ns,old.st_mtime_ns))
            counter[0]+=1
            return digest
        with patch.object(module,'_fingerprint',side_effect=lambda p:(*actual(p)[:6],counter[0])),patch('ptcg_lab.checkpoint_refs.sha256',side_effect=changed):
            with self.assertRaisesRegex(ValueError,'changed during complete hash'):
                checked_reference(self.root,'a',cache)
        self.assertEqual(len(cache),0)

    def test_cache_is_bounded_root_bound_process_bound_and_not_serializable(self):
        cache=module.ImmutableReferenceHashCache(self.root,maximum_entries=2)
        for name in ['a','b','c']:
            (self.root/name).write_bytes(name.encode());checked_reference(self.root,name,cache)
        self.assertEqual(len(cache),2)
        with self.assertRaises(ValueError):cache.get(('other-root','a'))
        with patch.object(module.os,'getpid',return_value=cache.pid+1):
            with self.assertRaises(RuntimeError):checked_reference(self.root,'a',cache)
        with self.assertRaises(TypeError):pickle.dumps(cache)
