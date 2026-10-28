import copy,gzip,hashlib,io,os,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
from ptcg_lab.io import atomic_json,read_json,sha256
from ptcg_lab.local_cache_retirement import identity,windows_security
from ptcg_lab.evidence_tiering import TR03_SOURCE,original_bytes,read_missing,target_relative
from ptcg_lab.local_trajectory_archive import pack_original,check_scope,run
import ptcg_lab.local_trajectory_archive as archive_module


class NoWait:
    def __init__(self,*args):pass
    def consume(self,count):pass


class LocalTrajectoryArchiveTests(unittest.TestCase):
    def fixture(self,root):
        sid,_=windows_security();token=hashlib.sha256(sid.encode()).hexdigest();tokens={'desktop':token,'current_Codex_sandbox':token}
        source=root/'runs'/TR03_SOURCE;path=source/'rollouts/v0000-s000000.pt';path.parent.mkdir(parents=True)
        data=b'exact complete original trajectory\0'*1000;path.write_bytes(data)
        summary=source/'series/000000.json';atomic_json(summary,{'statuses':['DONE','DONE'],'series_id':0,'policy_version':'0','trajectory_path':'rollouts/v0000-s000000.pt','trajectory_sha256':sha256(path)})
        atomic_json(source/'metrics.json',{'complete':True,'updates':32,'trained_decisions':262165})
        atomic_json(root/'research/autonomy/state.json',{'research_stage':{'hypotheses':{'TR03':{'status':'completed','classification':'inconclusive','run_ids':[TR03_SOURCE]}}}})
        atomic_json(root/'research/autonomy/evidence-locations.json',{'project_path':str(root),'trusted_execution_tokens':tokens,'objects':{}})
        row={'target':path.relative_to(root).as_posix(),'bytes':len(data),'sha256':sha256(path),'file_identity':identity(path),'owner_token_sha256':token,'summary_sha256':sha256(summary)}
        proof={'trusted_execution_tokens':tokens,'rows':[row],'protected_evidence':{(source/'metrics.json').relative_to(root).as_posix():sha256(source/'metrics.json')}}
        config={'job_kind':'local_cache_retirement','retirement_variant':'consumed_trajectory_gzip','maximum_attempts':1,'source_run_id':TR03_SOURCE,'project_path':str(root),'retirement_module_sha256':sha256(archive_module.__file__)}
        return source,path,data,row,proof,config

    def test_full_transaction_preserves_exact_original_readthrough_and_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();source,path,data,row,proof,c=self.fixture(root);out=root/'runs/archival';out.mkdir()
            inspection=root/'inspection.json';atomic_json(inspection,proof);c['inspection_evidence']={'path':'inspection.json','sha256':sha256(inspection)}
            with patch.dict(os.environ,{'PTCG_JOB_RUN_ID':out.name,'PTCG_JOB_DEADLINE_UNIX':str(time.time()+120)}),patch('ptcg_lab.local_trajectory_archive.LocalReadPacer',NoWait):run(c,out)
            self.assertFalse(path.exists());self.assertEqual(read_missing(path),data)
            self.assertTrue(read_json(out/'metrics.json')['actual_production_read_through_all_files_passed'])
            self.assertEqual(sha256(source/'metrics.json'),proof['protected_evidence'][(source/'metrics.json').relative_to(root).as_posix()])
    def test_wrong_original_SHA_never_publishes_or_unlinks_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'original.pt';source.write_bytes(b'original');packed=root/'bad.gz'
            with self.assertRaises(ValueError):pack_original(source,packed,'0'*64,8,NoWait())
            self.assertEqual(source.read_bytes(),b'original');self.assertFalse(packed.exists())
    def test_active_hypothesis_wrong_version_or_protected_path_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();source,path,_,row,proof,c=self.fixture(root)
            self.assertIsNone(check_scope(root,c,proof))
            bad=copy.deepcopy(proof);bad['rows'][0]['target']=f'runs/{TR03_SOURCE}/rollouts/v0032-s000000.pt'
            with self.assertRaises(ValueError):check_scope(root,c,bad)
            bad=copy.deepcopy(proof);bad['protected_evidence']={'../outside':'0'*64}
            with self.assertRaises(ValueError):check_scope(root,c,bad)
            atomic_json(root/'research/autonomy/state.json',{'research_stage':{'hypotheses':{'TR03':{'status':'started'}}}})
            with self.assertRaises(ValueError):check_scope(root,c,proof)
    def test_corrupt_encoded_bytes_and_unknown_target_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();_,path,data,row,proof,_=self.fixture(root);packed=root/'runs/archival/lossless-gzip'/(row['sha256']+'.pt.gz');packed.parent.mkdir(parents=True)
            ref=pack_original(path,packed,row['sha256'],row['bytes'],NoWait());ref['path']=packed.relative_to(root).as_posix()
            sid,_=windows_security();owner=lambda _:sid;tokens=proof['trusted_execution_tokens'];archived={**row,'retained_source':ref}
            self.assertEqual(original_bytes(root,archived,tokens,owner),data)
            packed.write_bytes(b'broken')
            with self.assertRaises(ValueError):original_bytes(root,archived,tokens,owner)
            with self.assertRaises(ValueError):target_relative(f'runs/{TR03_SOURCE}/checkpoint.pt')
    def test_real_torch_payload_is_readable_without_restoring_a_raw_replica(self):
        import torch
        from ptcg_lab.tensor_io import load_torch
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();_,path,_,row,proof,_=self.fixture(root)
            payload={'decisions':[{'old_logp':-1.25,'rows':torch.arange(12).reshape(3,4)}],'rng_after':{'torch':torch.get_rng_state()}}
            buffer=io.BytesIO();torch.save(payload,buffer);data=buffer.getvalue();path.write_bytes(data);row.update(bytes=len(data),sha256=sha256(path),file_identity=identity(path))
            packed=root/'runs/archival/lossless-gzip'/(row['sha256']+'.pt.gz');packed.parent.mkdir(parents=True)
            ref=pack_original(path,packed,row['sha256'],len(data),NoWait());ref['path']=packed.relative_to(root).as_posix()
            entry=root/'runs/archival/entry.json';atomic_json(entry,{'rows':[{**row,'retained_source':ref}]})
            catalog=read_json(root/'research/autonomy/evidence-locations.json');catalog['objects'][row['target']]={'manifest':{'path':entry.relative_to(root).as_posix(),'sha256':sha256(entry)}};atomic_json(root/'research/autonomy/evidence-locations.json',catalog)
            path.unlink();restored=load_torch(path,map_location='cpu',weights_only=False)
            self.assertTrue(torch.equal(restored['decisions'][0]['rows'],payload['decisions'][0]['rows']))
            self.assertTrue(torch.equal(restored['rng_after']['torch'],payload['rng_after']['torch']));self.assertFalse(path.exists())
