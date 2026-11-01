import gzip,hashlib,io,json,sys,tarfile,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
import compact
class RetentionTests(unittest.TestCase):
    def test_case_local_gzip_original_restores_exact_bytes_and_rejects_wrong_proof(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);retained=root/'retained';retained.mkdir();raw=b'original tensor/container bytes'*1024;encoded=gzip.compress(raw,mtime=0);blob=retained/'toy.tar.gz'
            with tarfile.open(blob,'w:gz') as archive:
                info=tarfile.TarInfo('archive/original.gz');info.size=len(encoded);archive.addfile(info,io.BytesIO(encoded))
            h=lambda value:hashlib.sha256(value).hexdigest()
            receipt=dict(bundle_sha256=h(blob.read_bytes()),files=[dict(path='archive/original.gz',sha256=h(encoded),bytes=len(encoded)),dict(path='job/model.bin',sha256=h(raw),bytes=len(raw),gzip_dependency='archive/original.gz')]);(retained/'toy.json').write_text(json.dumps(receipt))
            with patch.object(compact,'V3',root):
                output=compact.reconstruct('toy',root/'restored/good');self.assertEqual((output/'job/model.bin').read_bytes(),raw)
                receipt['files'][1]['sha256']='f'*64;(retained/'toy.json').write_text(json.dumps(receipt))
                with self.assertRaises(ValueError):compact.reconstruct('toy',root/'restored/rejected')
