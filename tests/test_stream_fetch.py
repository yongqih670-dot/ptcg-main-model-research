import hashlib
import io
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from ptcg_lab.stream_fetch import fetch_single_response


class Response:
    def __init__(self,data,status=200,length=None):
        self.body=io.BytesIO(data);self.status=status
        self.headers={'Content-Length':str(len(data) if length is None else length)}
    def read(self,size):return self.body.read(size)
    def __enter__(self):return self
    def __exit__(self,*args):pass


class SingleResponseTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.path=Path(self.tmp.name)/'artifact.whl'
        self.data=b'correct-content'*1000;self.digest=hashlib.sha256(self.data).hexdigest()
    def tearDown(self):self.tmp.cleanup()
    def fetch(self):
        return fetch_single_response('https://example.invalid/artifact',self.path,len(self.data),self.digest,time.time()+60)
    def test_one_response_and_full_digest_required(self):
        with patch('urllib.request.urlopen',return_value=Response(self.data)) as opened:
            self.fetch();self.assertEqual(opened.call_count,1)
        self.assertEqual(self.path.read_bytes(),self.data)
    def test_partial_http_response_cannot_be_published(self):
        with patch('urllib.request.urlopen',return_value=Response(self.data,status=206)):
            with self.assertRaises(ValueError):self.fetch()
        self.assertFalse(self.path.exists())
    def test_wrong_size_truncation_and_wrong_digest_fail_closed(self):
        for response in [Response(self.data,length=1),Response(self.data[:-1],length=len(self.data)),Response(b'x'*len(self.data))]:
            with self.subTest(response=response):
                self.path.with_suffix('.whl.stream.partial').unlink(missing_ok=True)
                with patch('urllib.request.urlopen',return_value=response):
                    with self.assertRaises((ValueError,OSError)):self.fetch()
                self.assertFalse(self.path.exists())
    def test_failed_response_is_preserved_and_cannot_be_resumed(self):
        with patch('urllib.request.urlopen',return_value=Response(self.data[:-1],length=len(self.data))):
            with self.assertRaises(OSError):self.fetch()
        partial=self.path.with_suffix('.whl.stream.partial');self.assertTrue(partial.exists())
        with self.assertRaises(RuntimeError):self.fetch()
        self.assertEqual(partial.read_bytes(),self.data[:-1])
