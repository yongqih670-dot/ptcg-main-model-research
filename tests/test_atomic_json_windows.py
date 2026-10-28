import ctypes,os,tempfile,threading,time
from pathlib import Path
import unittest
from ptcg_lab.io import atomic_json,read_json,_local_reader


@unittest.skipUnless(os.name=='nt','Real Windows sharing semantics')
class AtomicJSONWindowsTests(unittest.TestCase):
    def hold_without_delete_sharing(self,path):
        from ctypes import wintypes as w
        k=ctypes.WinDLL('kernel32',use_last_error=True)
        k.CreateFileW.argtypes=[w.LPCWSTR,w.DWORD,w.DWORD,ctypes.c_void_p,w.DWORD,w.DWORD,w.HANDLE]
        k.CreateFileW.restype=w.HANDLE;k.CloseHandle.argtypes=[w.HANDLE]
        handle=k.CreateFileW(str(path),0x80000000,3,None,3,0x80,None)
        if handle==ctypes.c_void_p(-1).value:raise ctypes.WinError(ctypes.get_last_error())
        return k,handle

    def test_actual_transient_reader_allows_bounded_atomic_replacement_after_release(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'state.json';atomic_json(p,{'version':0});k,h=self.hold_without_delete_sharing(p)
            release=threading.Thread(target=lambda:(time.sleep(.12),k.CloseHandle(h)));release.start()
            try:atomic_json(p,{'version':1})
            finally:release.join()
            self.assertEqual(read_json(p),{'version':1})

    def test_concurrent_project_reads_and_publications_never_observe_partial_json(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'state.json';atomic_json(p,{'version':0})
            stop=threading.Event();errors=[];seen=[]
            def reading():
                try:
                    while not stop.is_set():seen.append(read_json(p)['version'])
                except BaseException as exc:errors.append(exc)
            reader=threading.Thread(target=reading);reader.start()
            try:
                for version in range(1,101):atomic_json(p,{'version':version})
            finally:stop.set();reader.join()
            self.assertFalse(errors);self.assertTrue(seen)
            self.assertTrue(all(0<=x<=100 for x in seen));self.assertEqual(read_json(p),{'version':100})

    def test_persistent_conflict_keeps_old_json_and_unique_failed_temps(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'state.json';atomic_json(p,{'version':0});k,h=self.hold_without_delete_sharing(p)
            try:
                for version in [1,2]:
                    with self.assertRaises(PermissionError):atomic_json(p,{'version':version},replace_timeout_seconds=.03)
            finally:k.CloseHandle(h)
            self.assertEqual(read_json(p),{'version':0})
            failed=list(Path(folder).glob('state.json.*.tmp'));self.assertEqual(len(failed),2)
            self.assertEqual({read_json(f)['version'] for f in failed},{1,2})
            atomic_json(p,{'version':3});self.assertEqual(read_json(p),{'version':3})


if __name__=='__main__':unittest.main()
