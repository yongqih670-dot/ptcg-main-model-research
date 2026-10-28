import hashlib,io,os,tempfile,time
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
from ptcg_lab.io import atomic_json,read_json,sha256
from ptcg_lab.archive_compressed import receive_compressed_object
from ptcg_lab.local_cache_retirement import windows_security,identity
from ptcg_lab.evidence_tiering import ARCHIVES,read_missing,owned_path,original_bytes
from ptcg_lab.tensor_io import load_torch


@unittest.skipUnless(os.name=='nt','Actual local Windows identities and read-through route')
class EvidenceTieringTests(unittest.TestCase):
    def fixture(self,root):
        sid,owner=windows_security();fp=hashlib.sha256(sid.encode()).hexdigest();tokens={'desktop':fp,'current_Codex_sandbox':fp}
        stream=io.BytesIO();torch.save({'model':{'actor':torch.arange(11)},'optimizer':{'state':{1:{'step':torch.tensor(7)}}}},stream)
        data=stream.getvalue();digest=hashlib.sha256(data).hexdigest();rid=sorted(ARCHIVES)[0]
        relative=f'runs/{rid}/archive/checkpoint.pt';target=root/relative;target.parent.mkdir(parents=True);target.write_bytes(data)
        primary=root/'runs/retained-primary/checkpoint.pt';primary.parent.mkdir();primary.write_bytes(data)
        raw={'kind':'retained_primary_raw','path':primary.relative_to(root).as_posix(),'sha256':digest,'bytes':len(data),'file_identity':identity(primary)}
        ack=receive_compressed_object(root/'durable','source',3,0,'model.pt',digest,len(data),[data])
        packed={'kind':'durable_gzip_ACK','archive_root':'durable','ACK':ack}
        row={'target':relative,'sha256':digest,'bytes':len(data),'file_identity':identity(target),
            'owner_token_sha256':fp,'retained_source':packed}
        return row,raw,tokens,owner,data,target

    def catalog(self,root,row,tokens):
        manifest=root/'runs/tiered/tiered-evidence-manifest.json';atomic_json(manifest,{'rows':[row]})
        atomic_json(root/'research/autonomy/evidence-locations.json',{'project_path':str(root),'trusted_execution_tokens':tokens,
            'objects':{row['target']:{'manifest':{'path':manifest.relative_to(root).as_posix(),'sha256':sha256(manifest)}}}})

    def test_missing_forensic_tensor_restores_all_original_bytes_and_torch_payload(self):
        with tempfile.TemporaryDirectory() as f:
            root=Path(f).resolve();row,raw,tokens,owner,data,target=self.fixture(root)
            self.assertEqual(original_bytes(root,row,tokens,owner),data)
            self.assertEqual(original_bytes(root,{**row,'retained_source':raw},tokens,owner),data)
            self.catalog(root,row,tokens);target.unlink()
            self.assertEqual(read_missing(target),data)
            restored=load_torch(target,map_location='cpu',weights_only=False)
            self.assertTrue(torch.equal(restored['model']['actor'],torch.arange(11)))
            self.assertEqual(int(restored['optimizer']['state'][1]['step']),7)
            self.assertFalse(target.exists())  # Read-through does not consume extra artifact space.

    def test_corrupt_gzip_or_changed_primary_never_returns_a_plausible_model(self):
        with tempfile.TemporaryDirectory() as f:
            root=Path(f).resolve();row,raw,tokens,owner,data,target=self.fixture(root)
            packed=root/'durable'/row['retained_source']['ACK']['archive_relative'];packed.write_bytes(b'corrupted')
            with self.assertRaises(ValueError):original_bytes(root,row,tokens,owner)
            primary=root/raw['path'];primary.write_bytes(b'corrupted')
            with self.assertRaises(ValueError):original_bytes(root,{**row,'retained_source':raw},tokens,owner)
            self.assertTrue(target.exists())

    def test_catalog_manifest_mutation_and_unknown_paths_fail_closed(self):
        with tempfile.TemporaryDirectory() as f:
            root=Path(f).resolve();row,_,tokens,_,_,target=self.fixture(root);self.catalog(root,row,tokens);target.unlink()
            self.assertIsNone(read_missing(root/'agents/missing-model.pt'))
            manifest=root/'runs/tiered/tiered-evidence-manifest.json';manifest.write_text('{}')
            with self.assertRaises(ValueError):read_missing(target)

    def test_foreign_owner_and_primary_dependency_cycle_rejected(self):
        with tempfile.TemporaryDirectory() as f:
            root=Path(f).resolve();row,raw,tokens,owner,_,target=self.fixture(root)
            with self.assertRaises(ValueError):owned_path(root,target,tokens,lambda _:'foreign-token')
            cycle={**raw,'path':row['target']}
            with self.assertRaises(ValueError):original_bytes(root,{**row,'retained_source':cycle},tokens,owner)

    def test_real_delete_access_probe_does_not_delete_or_modify_the_target(self):
        from ptcg_lab.local_forensic_retirement import delete_access
        with tempfile.TemporaryDirectory() as f:
            root=Path(f).resolve();_,_,_,_,data,target=self.fixture(root);before=identity(target)
            delete_access(target);self.assertEqual(identity(target),before);self.assertEqual(target.read_bytes(),data)

    def test_full_transaction_keeps_metadata_and_reads_retired_tensors_through_catalog(self):
        import ptcg_lab.local_forensic_retirement as module
        with tempfile.TemporaryDirectory() as f:
            root=Path(f).resolve();row,_,tokens,_,data,target=self.fixture(root);out=root/'runs/maintenance';out.mkdir()
            evidence={}
            for rid in ARCHIVES:
                p=root/'runs'/rid;p.mkdir(exist_ok=True);atomic_json(p/'metrics.json',{'complete':True})
                evidence[rid]={'metrics.json':sha256(p/'metrics.json')}
            state={'research_stage':{'hypotheses':{n:{'status':'invalid','remaining_original_training_attempts':0} for n in ['TR04','TR05','TR06']}}}
            atomic_json(root/'research/autonomy/state.json',state)
            inspection=root/'inspection.json';atomic_json(inspection,{'trusted_execution_tokens':tokens,'rows':[row],'archive_evidence':evidence})
            config={'project_path':str(root),'job_kind':'local_cache_retirement','retirement_variant':'forensic_replicas','maximum_attempts':1,
                'retirement_module_sha256':sha256(module.__file__),'inspection_evidence':{'path':'inspection.json','sha256':sha256(inspection)}}
            with patch.dict(os.environ,{'PTCG_JOB_RUN_ID':out.name,'PTCG_JOB_DEADLINE_UNIX':str(time.time()+120)}):module.run(config,out)
            self.assertFalse(target.exists());self.assertEqual(read_missing(target),data)
            self.assertTrue(read_json(out/'metrics.json')['all_original_bytes_retained_in_primary_raw_or_verified_gzip'])
            for rid,values in evidence.items():self.assertEqual(sha256(root/'runs'/rid/'metrics.json'),values['metrics.json'])

    def test_closed_TR07_scope_preserves_its_exact_gzip_original_and_metadata(self):
        import ptcg_lab.local_forensic_retirement as module
        from ptcg_lab.evidence_tiering import TR07_ARCHIVE
        with tempfile.TemporaryDirectory() as f:
            root=Path(f).resolve();row,_,tokens,owner,data,_=self.fixture(root)
            target=root/'runs'/TR07_ARCHIVE/'archive/checkpoint.pt';target.parent.mkdir(parents=True);target.write_bytes(data)
            row.update(target=target.relative_to(root).as_posix(),file_identity=identity(target))
            metadata=root/'runs'/TR07_ARCHIVE/'metrics.json';atomic_json(metadata,{'complete':True})
            before=sha256(metadata);out=root/'runs/tr07-maintenance';out.mkdir()
            atomic_json(root/'research/autonomy/state.json',{'research_stage':{'hypotheses':{'TR07':{'status':'invalid','remaining_original_training_attempts':0}}}})
            inspection=root/'inspection.json';atomic_json(inspection,{'trusted_execution_tokens':tokens,'rows':[row],'archive_evidence':{TR07_ARCHIVE:{'metrics.json':before}}})
            config={'project_path':str(root),'job_kind':'local_cache_retirement','retirement_variant':'forensic_replicas','maximum_attempts':1,
                'retirement_module_sha256':sha256(module.__file__),'inspection_evidence':{'path':'inspection.json','sha256':sha256(inspection)}}
            with patch.dict(os.environ,{'PTCG_JOB_RUN_ID':out.name,'PTCG_JOB_DEADLINE_UNIX':str(time.time()+120)}):module.run(config,out)
            self.assertEqual(sha256(metadata),before);self.assertFalse(target.exists());self.assertEqual(read_missing(target),data)


if __name__=='__main__':unittest.main()
