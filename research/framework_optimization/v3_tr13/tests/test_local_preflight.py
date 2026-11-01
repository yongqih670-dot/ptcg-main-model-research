import json,os,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
V3=Path(__file__).resolve().parents[1];ROOT=V3.parents[2]
sys.path.insert(0,str(V3/'benchmarks'))
from local_adapter import install_archive_publication_adapter
from ptcg_lab.archive_compressed import receive_compressed_object,verify_archive
from ptcg_lab.io import atomic_json,sha256
from ptcg_lab.rl_core import load_full_actor,optimizer_for
class LocalPreflightTests(unittest.TestCase):
    def test_native_complete_commit_full_outbox_and_ACK_including_prefix_alias(self):
        import torch,random,numpy as np,time
        from ptcg_lab.framework_v2_training import NativeCheckpoints
        from ptcg_lab.archive_state import initial_state
        from ptcg_lab.archive_prefix import canonical_hash
        from ptcg_lab.archive_outbox import receive_one
        from local_adapter import local_outbox
        with tempfile.TemporaryDirectory(dir=V3/'work') as directory:
            root=Path(directory);job=root/'job';job.mkdir();atomic_json(root/'.ptcg-owner.json',dict(project_id='v3-combined-preflight',uid=os.getuid() if hasattr(os,'getuid') else 0,root=str(root.resolve())))
            with patch.dict(os.environ,PTCG_PROJECT_ROOT=str(root),PTCG_PROJECT_ID='v3-combined-preflight',PTCG_JOB_DEADLINE_UNIX=str(time.time()+60),V3_TOKEN='inert-unit-fixture'):
                prefix=dict(schema_version=1,checkpoint=dict(sha256='a'*64,version=1,trained_decisions=10,next_series_id=0),consumed=[],tail=[],consumed_records_root=canonical_hash([]),consumed_fixed_statistics={},full_fixed_statistics={})
                atomic_json(job/'prefix.json',prefix);digest=sha256(job/'prefix.json');atomic_json(job/'local-engineering-owner.json',dict(parent_token='inert-unit-fixture',scope='disposable fixed replay; no scientific authority'))
                out=local_outbox(job,1,optimizations={'outbox_cache':True},prefix_sha256=digest);install_archive_publication_adapter()
                from common import profile
                backend=NativeCheckpoints(job,1,profile(),hard_bytes=2*2**30,rate=2**40)
                try:
                    value=dict(model={'actor':torch.ones(2)},optimizer={'state':{},'param_groups':[]},rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=[]),version=1,trained_decisions=10,next_series_id=0,archive_state=initial_state(prefix,digest))
                    commit=backend.save(value,dict(prefix_path='prefix.json',prefix_sha256=digest),{'snapshots':[]},model_revision=0);out.queue_recovery(commit)
                    class Stream:
                        def request(self,op,payload,**kwargs):
                            if op=='archive_ack':return out.accept_ack(payload['ack'])
                            return out.chunk(payload['object_sequence'],payload['offset'],payload['count'])
                    sources={}
                    for row in out.records():
                        ack=receive_one(Stream(),root/'archive',row,existing_source=sources.get(row['sha256']),storage_codec='gzip');sources[row['sha256']]={'archive_root':root/'archive','ack':ack}
                    self.assertEqual(out.pending()['pending_objects'],0);self.assertIn('prefix.json',[r['object'] for r in out.records()])
                finally:backend.close()
    def test_real_actor_features_and_full_optimizer_load(self):
        import torch
        spec=json.loads((V3/'reports/INPUTS.json').read_text());bundle=ROOT/spec['current_bundle']
        model,features,_=load_full_actor(bundle,sha256(bundle/'model.pt'),reset_value=False)
        parent=torch.load(ROOT/spec['parent']['path'],map_location='cpu',weights_only=False)
        model.load_state_dict(parent['model'],strict=True);opt=optimizer_for(model);opt.load_state_dict(parent['optimizer'])
        self.assertGreater(len(opt.state_dict()['state']),0);self.assertEqual(features.b.STOP,63)
    def test_extended_archive_ACK_publication_stays_in_owned_root(self):
        with tempfile.TemporaryDirectory(dir=V3/'work') as d:
            root=Path(d);atomic_json(root/'.ptcg-owner.json',dict(project_id='v3-preflight',uid=os.getuid() if hasattr(os,'getuid') else 0,root=str(root.resolve())))
            import time,hashlib
            with patch.dict(os.environ,PTCG_PROJECT_ROOT=str(root),PTCG_PROJECT_ID='v3-preflight',PTCG_JOB_DEADLINE_UNIX=str(time.time()+60)):
                install_archive_publication_adapter();data=b'full lossless local object';digest=hashlib.sha256(data).hexdigest()
                ack=receive_compressed_object(root/'archive','job',1,0,'model.pt',digest,len(data),[data]);self.assertTrue(verify_archive(root/'archive',ack))
                from ptcg_lab.archive_compressed import atomic_json as publish
                with self.assertRaises(ValueError):publish(root.parent/'escaped.json',{'must':'fail'})
