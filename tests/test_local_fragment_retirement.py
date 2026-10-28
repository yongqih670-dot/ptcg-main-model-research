import hashlib,os,time,tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from ptcg_lab.io import atomic_json,read_json,sha256
from ptcg_lab.local_cache_retirement import windows_security,identity
from ptcg_lab.local_fragment_retirement import PAIRS,run,verify_pair


@unittest.skipUnless(os.name=='nt','Real Windows ownership checks')
class PrefixRetirementTests(unittest.TestCase):
    def fixture(self,root,corrupt_second=False):
        rows=[];sid,_=windows_security()
        for i,(part,wheel) in enumerate(PAIRS.items()):
            a=root/part;b=root/wheel;a.parent.mkdir(parents=True,exist_ok=True);b.parent.mkdir(parents=True,exist_ok=True)
            full=(b'exact original public wheel bytes'+bytes([i]))*5;b.write_bytes(full);a.write_bytes(full[:37])
            attempts=root/'runs'/Path(part).parts[1]/'attempts.json';atomic_json(attempts,[{'status':'failed','seconds':1}])
            rows.append({'partial':part,'wheel':wheel,'bytes':37,'wheel_bytes':len(full),
                'sha256':sha256(a),'wheel_sha256':sha256(b),'file_identity':identity(a),'wheel_identity':identity(b),
                'source_run_id':Path(part).parts[1],'source_attempts_sha256':sha256(attempts)})
        if corrupt_second:
            a=root/rows[1]['partial'];a.write_bytes(b'x'*37);rows[1].update(sha256=sha256(a),file_identity=identity(a))
        inspection=root/'inspection.json';atomic_json(inspection,{'owner_sid_sha256':hashlib.sha256(sid.encode()).hexdigest(),
            'runs_container_owner_sid_sha256':None,'rows':rows})
        import ptcg_lab.local_fragment_retirement as module
        config={'project_path':str(root),'job_kind':'local_cache_retirement','retirement_variant':'public_download_prefixes','maximum_attempts':1,
            'retirement_module_sha256':sha256(module.__file__),'inspection_evidence':{'path':'inspection.json','sha256':sha256(inspection)}}
        output=root/'runs'/'prefix-check';output.mkdir();return config,output,rows

    def test_two_exact_prefixes_can_be_reconstructed_after_controlled_single_file_retirement(self):
        with tempfile.TemporaryDirectory() as f:
            root=Path(f).resolve();c,out,rows=self.fixture(root)
            with patch.dict(os.environ,{'PTCG_JOB_RUN_ID':out.name,'PTCG_JOB_DEADLINE_UNIX':str(time.time()+120)}):run(c,out)
            self.assertEqual(read_json(out/'metrics.json')['retired_logical_bytes'],74)
            for r in rows:
                self.assertFalse((root/r['partial']).exists())
                restored=(root/r['wheel']).read_bytes()[:r['bytes']]
                self.assertEqual(hashlib.sha256(restored).hexdigest(),r['sha256'])
                self.assertEqual(sha256(root/r['wheel']),r['wheel_sha256'])

    def test_rejected_second_prefix_preserves_both_and_never_publishes_unlink_intent(self):
        with tempfile.TemporaryDirectory() as f:
            root=Path(f).resolve();c,out,rows=self.fixture(root,corrupt_second=True)
            with patch.dict(os.environ,{'PTCG_JOB_RUN_ID':out.name,'PTCG_JOB_DEADLINE_UNIX':str(time.time()+120)}):
                with self.assertRaises(ValueError):run(c,out)
            self.assertTrue(all((root/r['partial']).exists() for r in rows))
            self.assertFalse((out/'retirement-intent.json').exists())

    def test_unregistered_target_and_changed_retained_wheel_rejected(self):
        with tempfile.TemporaryDirectory() as f:
            root=Path(f).resolve();_,_,rows=self.fixture(root);sid,owner=windows_security()
            with self.assertRaises(ValueError):verify_pair(root,{**rows[0],'partial':'../foreign'},sid,owner,None,None)
            b=root/rows[0]['wheel'];b.write_bytes(b'changed retained original')
            with self.assertRaises(ValueError):verify_pair(root,rows[0],sid,owner,None,None)


class SnapshotScopeTests(unittest.TestCase):
    def test_only_exact_prefix_maintenance_omits_unused_model_assets(self):
        from scripts import run_experiment as runner
        with tempfile.TemporaryDirectory() as f:
            root=Path(f)
            for name in ['ptcg_lab','scripts','tests','agents']:
                d=root/name;d.mkdir();(d/('model.pt' if name=='agents' else 'module.py')).write_bytes(b'complete original')
            for label,config,keeps_models in [
                ('maintenance',{'job_kind':'local_cache_retirement','retirement_variant':'public_download_prefixes'},False),
                ('read-only-archive',{'job_kind':'server_artifact_archive'},False),
                ('read-only-tr07-forensics',{'job_kind':'compact_audit','audit_variant':'tr07_failed_training_analysis'},False),
                ('other-audit',{'job_kind':'compact_audit'},True),
                ('research',{'job_kind':'remote_train_ppo'},True),
                ('other-maintenance',{'job_kind':'local_cache_retirement'},True)]:
                run=root/label;run.mkdir();atomic_json(run/'config.json',config)
                with patch.object(runner,'ROOT',root):runner.snapshot(run)
                manifest=read_json(run/'source-manifest.json')
                self.assertEqual((run/'source/agents/model.pt').exists(),keeps_models)
                self.assertTrue((run/'source/ptcg_lab/module.py').exists())
                self.assertTrue(any('module.py' in n for n in manifest))


if __name__=='__main__':unittest.main()
