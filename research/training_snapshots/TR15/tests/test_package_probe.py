import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from ptcg_lab.io import sha256
from ptcg_lab.package_probe import extract_verified


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def package(self,extra=None,wrong_file=False):
        files={'main.py':b'def agent(obs): return []\n','deck.csv':b'1\n'*60,
               'agent-spec.json':b'{}','bundle/model.pt':b'model-fixture'}
        manifest={'files':{n:hashlib.sha256(v).hexdigest() for n,v in files.items()},
                  'model_sha256':hashlib.sha256(files['bundle/model.pt']).hexdigest()}
        if wrong_file:files['main.py']=b'changed'
        files['package-manifest.json']=json.dumps(manifest).encode()
        archive=self.root/'submission.tar.gz'
        with tarfile.open(archive,'w:gz') as tf:
            for name,body in files.items():
                info=tarfile.TarInfo(name);info.size=len(body);tf.addfile(info,io.BytesIO(body))
            if extra:
                info=tarfile.TarInfo(extra)
                if extra=='link':info.type=tarfile.SYMTYPE;info.linkname='../outside'
                tf.addfile(info,io.BytesIO())
        return archive
    def test_verified_package_extracts_exact_files(self):
        path=self.package();dest=self.root/'out'
        manifest=extract_verified(path,dest,sha256(path))
        self.assertEqual((dest/'bundle/model.pt').read_bytes(),b'model-fixture')
        self.assertEqual(set(manifest['files']),{'main.py','deck.csv','agent-spec.json','bundle/model.pt'})
    def test_external_paths_links_and_duplicates_fail_closed(self):
        for index,name in enumerate(['../outside','/outside','C:/outside','dir\\outside','main.py','link']):
            path=self.package(extra=name)
            with self.assertRaisesRegex(ValueError,'Unsafe or duplicate'):
                extract_verified(path,self.root/f'out-{index}',sha256(path))
        self.assertFalse((self.root.parent/'outside').exists())
    def test_changed_archive_and_changed_content_fail_closed(self):
        path=self.package()
        with self.assertRaisesRegex(ValueError,'Archive changed'):
            extract_verified(path,self.root/'out-a','0'*64)
        path=self.package(wrong_file=True)
        with self.assertRaisesRegex(ValueError,'file hash mismatch'):
            extract_verified(path,self.root/'out-b',sha256(path))
    def test_unlisted_member_fails_coverage(self):
        path=self.package(extra='unlisted.py')
        with self.assertRaisesRegex(ValueError,'coverage mismatch'):
            extract_verified(path,self.root/'out',sha256(path))


if __name__=='__main__':unittest.main()
