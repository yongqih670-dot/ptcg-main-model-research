import json,os,tempfile,unittest
from pathlib import Path
from ptcg_lab.framework_v3_runtime import DirectoryEvidence,JsonEvidence
class DirectoryTests(unittest.TestCase):
    def test_empty_poll_new_row_and_replacement_are_seen_immediately(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);folder=root/'archive-outbox';folder.mkdir();cache=DirectoryEvidence(root)
            self.assertEqual(cache.paths(folder,'*.json'),[]);self.assertEqual(cache.paths(folder,'*.json'),[]);self.assertEqual(cache.hits,1)
            stamp=folder.stat();(folder/'00000000.json').write_text('{}');os.utime(folder,ns=(stamp.st_atime_ns,stamp.st_mtime_ns))
            self.assertEqual(len(cache.paths(folder,'*.json')),1)
            (folder/'00000000.json').unlink();folder.rmdir();self.assertEqual(cache.paths(folder,'*.json'),[])
            folder.mkdir();(folder/'00000001.json').write_text('{}');self.assertEqual(cache.paths(folder,'*.json')[0].name,'00000001.json')
    def test_cached_enumeration_does_not_hide_existing_row_edit(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);folder=root/'queue';folder.mkdir();path=folder/'row.json';path.write_text('{"sequence":0}')
            dirs=DirectoryEvidence(root);rows=JsonEvidence(root,lambda p:json.loads(p.read_text()))
            self.assertEqual(rows.get(dirs.paths(folder,'*.json')[0])['sequence'],0)
            stamp=path.stat();path.write_text('{"sequence":9}');os.utime(path,ns=(stamp.st_atime_ns,stamp.st_mtime_ns))
            self.assertEqual(rows.get(dirs.paths(folder,'*.json')[0])['sequence'],9);self.assertEqual(dirs.hits,1)
