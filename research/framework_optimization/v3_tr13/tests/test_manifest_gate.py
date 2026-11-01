import hashlib,json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from ptcg_lab.framework_v3_runtime import from_config
from ptcg_lab.archive_outbox import ArchiveOutbox
from ptcg_lab.io import sha256
V3=Path(__file__).resolve().parents[1]
class ManifestGateTests(unittest.TestCase):
    def test_full_source_binding_and_manifest_edit_invalidation(self):
        with tempfile.TemporaryDirectory() as directory:
            job=Path(directory)/'job';job.mkdir();path=job/'framework-v3-manifest.json'
            source='ptcg_lab/framework_v3_runtime.py';path.write_text(json.dumps({'runtime_sources':{source:sha256(V3/'candidate'/source)}}))
            bound=sha256(path);switches={'outbox_cache':True}
            # Inert unit fixture only: no launcher, controlled job, gradients,
            # server registration or original project state is created.
            ticket=dict(run_id='job',lease_epoch=1,engineering_scope='framework_efficiency_v3',framework_v3_approved=True,framework_v3_manifest_sha256=bound)
            (job/'ticket.json').write_text(json.dumps(ticket));config=dict(engineering_only=True,new_scientific_training=False,framework_v3_optimizations=switches,framework_v3_manifest_sha256=bound)
            with patch.dict(os.environ,PTCG_JOB_RUN_ID='job',PTCG_JOB_LEASE_EPOCH='1'):
                self.assertTrue(from_config(config,job)['outbox_cache']);out=ArchiveOutbox(job,'job',1,optimizations=switches)
                self.assertTrue(out._allowed('objects/model-'+'a'*64+'.bin'))
                stamp=path.stat();path.write_text(path.read_text().replace('runtime_sources','runtime_sourceX'));os.utime(path,ns=(stamp.st_atime_ns,stamp.st_mtime_ns))
                with self.assertRaises(ValueError):out._allowed('objects/model-'+'b'*64+'.bin')
    def test_empty_or_outside_source_catalog_cannot_be_admitted(self):
        with tempfile.TemporaryDirectory() as directory:
            job=Path(directory)/'job';job.mkdir();path=job/'framework-v3-manifest.json';path.write_text(json.dumps({'runtime_sources':{'../outside':'a'*64}}));bound=sha256(path)
            (job/'ticket.json').write_text(json.dumps(dict(run_id='job',lease_epoch=1,engineering_scope='framework_efficiency_v3',framework_v3_approved=True,framework_v3_manifest_sha256=bound)))
            config=dict(engineering_only=True,new_scientific_training=False,framework_v3_optimizations={},framework_v3_manifest_sha256=bound)
            with patch.dict(os.environ,PTCG_JOB_RUN_ID='job',PTCG_JOB_LEASE_EPOCH='1'):
                with self.assertRaises(ValueError):from_config(config,job)
