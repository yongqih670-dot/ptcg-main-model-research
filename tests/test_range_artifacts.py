import hashlib
import io
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from ptcg_lab.range_artifacts import download


class Response(io.BytesIO):
    status=206
    url='https://files.pythonhosted.org/pinned.whl'
    def __init__(self,body,start,end,total):
        super().__init__(body);self.headers={'Content-Range':f'bytes {start}-{end}/{total}'}


class FastPacer:
    def __init__(self,*args):pass
    def consume(self,count):assert 0<count<=256*1024


class RangeArtifactTests(unittest.TestCase):
    def exercise(self,resume=False,corrupt=False,bad_range=False):
        body=bytes(range(251))*10000
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();(root/'stage').mkdir()
            spec={'workers':2,'range_bytes':1024**2}
            old=None
            if resume:
                old=root/'old.partial';old.write_bytes(body[:174321])
                spec['resume']={'relative_path':'old.partial','bytes':old.stat().st_size,
                                'prefix_sha256':('0'*64 if corrupt else hashlib.sha256(old.read_bytes()).hexdigest())}
            item={'file':'pinned.whl','url':Response.url,'bytes':len(body),
                  'sha256':hashlib.sha256(body).hexdigest(),'range_repair':spec}
            def fetch(req,timeout):
                start,end=map(int,req.headers['Range'].split('=')[1].split('-'))
                return Response(body[start:end+1],start,end,len(body)+(1 if bad_range else 0))
            with patch('ptcg_lab.range_artifacts.urlopen',fetch),patch('ptcg_lab.range_artifacts.BytePacer',FastPacer):
                if corrupt or bad_range:
                    with self.assertRaises(ValueError):download(root,root,item,time.time()+120,lambda:None,lambda:None,lambda *a,**kw:None)
                    self.assertFalse((root/'stage/pinned.whl').exists())
                else:
                    path=download(root,root,item,time.time()+120,lambda:None,lambda:None,lambda *a,**kw:None)
                    self.assertEqual(path.read_bytes(),body)
            if old:self.assertEqual(old.read_bytes(),body[:174321])
    def test_two_readers_publish_only_complete_original_hash(self):self.exercise()
    def test_verified_fragment_reused_without_touching_source(self):self.exercise(resume=True)
    def test_wrong_fragment_never_publishes(self):self.exercise(resume=True,corrupt=True)
    def test_wrong_range_total_never_publishes(self):self.exercise(bad_range=True)
