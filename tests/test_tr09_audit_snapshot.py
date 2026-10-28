import tempfile
from pathlib import Path
from unittest.mock import patch
import unittest
from scripts import run_experiment as runner
from ptcg_lab.io import atomic_json,sha256


class AuditSnapshotTests(unittest.TestCase):
    def test_only_local_readonly_audit_omits_unused_assets(self):
        for kind,variant,agents in [('compact_audit','tr09_clean_checkpoint_recovery',False),('remote_train_ppo','',True),('evaluation','',True)]:
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);run=root/'runs/test';run.mkdir(parents=True)
                for name in ['agents','ptcg_lab','scripts','tests']:
                    (root/name).mkdir();(root/name/'retained.txt').write_text(name)
                atomic_json(run/'config.json',{'job_kind':kind,'audit_variant':variant})
                atomic_json(run/'experiment.json',{'config_sha256':sha256(run/'config.json')})
                with patch.object(runner,'ROOT',root):
                    runner.snapshot(run);runner.verify_snapshot(run)
                self.assertEqual((run/'source/agents/retained.txt').exists(),agents)
                self.assertTrue((run/'source/ptcg_lab/retained.txt').exists())


if __name__=='__main__':unittest.main()
