import hashlib
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from ptcg_lab.range_fetch import fetch_verified


class Response:
    status=206
    def __init__(self,body,headers):self.body=body;self.headers=headers
    def read(self,n):return self.body[:n]
    def __enter__(self):return self
    def __exit__(self,*args):pass


class RangeFetchTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.path=Path(self.tmp.name)/'artifact.whl';self.data=b'0123456789abcdef'*1000;self.digest=hashlib.sha256(self.data).hexdigest();self.calls=[]
    def tearDown(self):self.tmp.cleanup()
    def serve(self,request,**kwargs):
        start,end=map(int,request.headers['Range'].split('=')[1].split('-'));self.calls.append(start)
        return Response(self.data[start:end+1],{'Content-Range':f'bytes {start}-{end}/{len(self.data)}','ETag':'stable'})
    def fetch(self,progress=None):return fetch_verified('https://example.invalid/artifact',self.path,len(self.data),self.digest,time.time()+60,progress,chunk_bytes=1024)
    def test_interrupted_transfer_resumes_after_verified_chunks(self):
        def interrupt(done,total):
            if done>=2048:raise RuntimeError('simulated interruption')
        with patch('urllib.request.urlopen',side_effect=self.serve):
            with self.assertRaises(RuntimeError):self.fetch(interrupt)
            self.calls=[];self.fetch()
        self.assertEqual(self.calls[0],2048);self.assertEqual(self.path.read_bytes(),self.data)
    def test_wrong_range_or_wrong_full_hash_never_publishes(self):
        with patch('urllib.request.urlopen',return_value=Response(b'x'*1024,{'Content-Range':'bytes 1-1024/16000','ETag':'stable'})):
            with self.assertRaises(ValueError):self.fetch()
        self.assertFalse(self.path.exists())
    def test_modified_committed_partial_is_rejected(self):
        def stop(done,total):raise RuntimeError('stop')
        with patch('urllib.request.urlopen',side_effect=self.serve):
            with self.assertRaises(RuntimeError):self.fetch(stop)
        self.path.with_suffix('.whl.partial').write_bytes(b'x'*1024)
        with self.assertRaisesRegex(ValueError,'chunk changed'):self.fetch()
    def test_final_hash_mismatch_preserves_partial_but_not_valid_artifact(self):
        with patch('urllib.request.urlopen',side_effect=self.serve):
            with self.assertRaisesRegex(ValueError,'official SHA256'):
                fetch_verified('https://example.invalid/artifact',self.path,len(self.data),'0'*64,time.time()+60,chunk_bytes=1024)
        self.assertFalse(self.path.exists());self.assertTrue(self.path.with_suffix('.whl.partial').exists())
    def test_changing_etags_only_quarantine_until_official_digest_matches(self):
        def serve(request,**kwargs):
            result=self.serve(request,**kwargs);result.headers['ETag']=str(len(self.calls));return result
        with patch('urllib.request.urlopen',side_effect=serve):
            fetch_verified('https://example.invalid/artifact',self.path,len(self.data),self.digest,time.time()+60,
                           chunk_bytes=1024,entity_policy='official_digest_quarantine')
        self.assertEqual(self.path.read_bytes(),self.data)
    def test_quarantined_ranges_with_bad_content_never_publish(self):
        def serve(request,**kwargs):
            result=self.serve(request,**kwargs);result.headers['ETag']=str(len(self.calls));result.body=b'x'*len(result.body);return result
        with patch('urllib.request.urlopen',side_effect=serve):
            with self.assertRaisesRegex(ValueError,'official SHA256'):
                fetch_verified('https://example.invalid/artifact',self.path,len(self.data),self.digest,time.time()+60,
                               chunk_bytes=1024,entity_policy='official_digest_quarantine')
        self.assertFalse(self.path.exists())
    def test_seed_is_copy_only_and_final_full_artifact_still_required(self):
        seed=self.path.parent/'old-failed.partial';seed.write_bytes(self.data[:2048])
        record={'path':str(seed),'bytes':2048,'sha256':hashlib.sha256(seed.read_bytes()).hexdigest()}
        with patch('urllib.request.urlopen',side_effect=self.serve):
            fetch_verified('https://example.invalid/artifact',self.path,len(self.data),self.digest,time.time()+60,
                           chunk_bytes=1024,entity_policy='official_digest_quarantine',seed=record)
        self.assertEqual(self.calls[0],2048);self.assertEqual(seed.read_bytes(),self.data[:2048])
        self.assertEqual(self.path.read_bytes(),self.data)
    def test_closed_strict_journal_cannot_be_reinterpreted_as_digest_mode(self):
        with patch('urllib.request.urlopen',side_effect=self.serve):
            with self.assertRaises(RuntimeError):self.fetch(lambda *_:(_ for _ in ()).throw(RuntimeError('stop')))
        with self.assertRaisesRegex(ValueError,'reinterpret'):
            fetch_verified('https://example.invalid/artifact',self.path,len(self.data),self.digest,time.time()+60,
                           entity_policy='official_digest_quarantine')
