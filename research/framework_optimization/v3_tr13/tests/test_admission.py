import os,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
from ptcg_lab.framework_v3_admission import install
from ptcg_lab.fo_v2.durable import Capacity
from ptcg_lab.bounded_io import WriteContext
from ptcg_lab.io import atomic_json
class AdmissionTests(unittest.TestCase):
    def test_legacy_writes_replacements_failures_and_external_growth(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);job=root/'job';job.mkdir();atomic_json(root/'.ptcg-owner.json',dict(project_id='v3-admission',uid=os.getuid() if hasattr(os,'getuid') else 0,root=str(root.resolve())))
            cap=Capacity(root,2**30,free_reserve=0,metadata_reserve=0);backend=SimpleNamespace(root=job,capacity=cap,write_admission=None)
            scope=install(backend,root);context=WriteContext(root,'v3-admission',time.time()+60)
            try:
                before=cap.used;context.atomic_write(job/'progress.json',lambda s:s.write(b'abc'),expected_bytes=3)
                self.assertEqual(cap.used,before+3)
                context.atomic_write(job/'progress.json',lambda s:s.write(b'ab'),expected_bytes=2);self.assertEqual(cap.used,before+2)
                def broken(s):s.write(b'failed bytes');s.flush();raise OSError('injected')
                with self.assertRaises(OSError):context.atomic_write(job/'uncommitted',broken,expected_bytes=12)
                self.assertGreaterEqual(cap.used,before+2+12);self.assertEqual(cap.reservations,{})
                external=root/'outside-controlled-writer';external.write_bytes(b'X'*1024);cap.at-=61;cap.reserve(1)
                self.assertGreaterEqual(cap.used,before+2+12+1024)
                with self.assertRaises(ValueError):
                    with scope.publication(root.parent/'escape',10):pass
            finally:scope.close()
    def test_registered_writer_scope_reuses_only_fresh_full_project_snapshot(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);job=root/'job';job.mkdir();cap=Capacity(root,2**30,free_reserve=0,metadata_reserve=0);backend=SimpleNamespace(root=job,capacity=cap,write_admission=None)
            scope=install(backend,root)
            try:
                scans=cap.scans
                with scope.publication(job/'known',10):(job/'known').write_bytes(b'123')
                self.assertEqual(cap.scans,scans);self.assertEqual(cap.reservations,{})
                cap.disk_free=lambda:0
                with self.assertRaises(OSError):
                    with scope.publication(job/'denied',10):pass
                self.assertFalse((job/'denied').exists())
            finally:scope.close()
    def test_immutable_temporary_and_final_logical_names_both_reserved(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);job=root/'job';job.mkdir();atomic_json(root/'.ptcg-owner.json',dict(project_id='v3-double-path',uid=os.getuid() if hasattr(os,'getuid') else 0,root=str(root.resolve())))
            cap=Capacity(root,2**30,free_reserve=0,metadata_reserve=0);backend=SimpleNamespace(root=job,capacity=cap,write_admission=None);scope=install(backend,root);context=WriteContext(root,'v3-double-path',time.time()+60)
            try:
                cap.hard=cap.used+64
                with self.assertRaises(OSError):context.atomic_write(job/'immutable',lambda s:s.write(b'x'*40),replace=False,expected_bytes=40)
                self.assertFalse((job/'immutable').exists());self.assertEqual(cap.reservations,{})
            finally:scope.close()
