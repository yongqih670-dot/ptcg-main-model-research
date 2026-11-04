import tempfile
from pathlib import Path
import unittest
from ptcg_lab.io import sha256
from ptcg_lab.server_handoff import check_asset,inventory_paths,relative_name,verify_inventory
from scripts.run_experiment import validate_execution_target


class ServerHandoffTests(unittest.TestCase):
    def test_private_inputs_and_traversal_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for name in ['.env','kaggle.json','id_ed25519','key.pem','.ssh/config']:
                p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('not a real secret')
                with self.assertRaises(ValueError):check_asset(root,p)
        for name in ['../escape','/absolute','a/../b','C:/outside','a//b']:
            with self.assertRaises(ValueError):relative_name(name)

    def test_inventory_uses_sealed_ledger_and_excludes_environment_and_own_output(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for name in ['research/autonomy/state.json','research/autonomy/job.lock','research/autonomy/sealed.json',
                         'runs/old/checkpoint.pt','runs/new/progress.json','.venv/secret.txt','scripts/a.py']:
                p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(name.encode())
            sealed=root/'research/autonomy/sealed.json';files=inventory_paths(root,root/'runs/new',sealed)
            self.assertEqual(files['research/autonomy/state.json'],sealed)
            self.assertIn('runs/old/checkpoint.pt',files)
            self.assertNotIn('runs/new/progress.json',files)
            self.assertNotIn('research/autonomy/job.lock',files)
            self.assertNotIn('.venv/secret.txt',files)

    def test_received_checkpoint_corruption_is_detected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'checkpoint.pt';p.write_bytes(b'original')
            m={'files':{'checkpoint.pt':{'sha256':sha256(p),'bytes':8}},'total_bytes':8}
            self.assertEqual(verify_inventory(root,m)['verified_files'],1)
            p.write_bytes(b'corrupt!')
            with self.assertRaises(ValueError):verify_inventory(root,m)

    def test_local_research_and_resume_are_blocked_after_server_target_change(self):
        state={'execution_target':{'required':'remote_server'}}
        for kind in ['train_ppo','evaluation','linux_probe','prepare_tokens','train_transformer']:
            with self.assertRaisesRegex(RuntimeError,'Server migration gate'):
                validate_execution_target({'job_kind':kind},state)
        validate_execution_target({'job_kind':'server_handoff'},state)
        validate_execution_target({'job_kind':'train_ppo'},{})

    def test_hybrid_preparation_does_not_enable_local_training_or_evaluation(self):
        state={'execution_target':{'required':'hybrid','status':'preparing'}}
        for kind in ['train_ppo','evaluation','tr00_audit','train_transformer','train_catboost']:
            with self.assertRaisesRegex(RuntimeError,'Hybrid execution gate'):validate_execution_target({'job_kind':kind},state)
        validate_execution_target({'job_kind':'compact_audit'},state)
        state['execution_target']['status']='active'
        for kind in ['evaluation','package_probe','linux_probe']:validate_execution_target({'job_kind':kind},state)
        with self.assertRaises(RuntimeError):validate_execution_target({'job_kind':'train_ppo'},state)


if __name__=='__main__':unittest.main()
