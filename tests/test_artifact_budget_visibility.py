import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts import run_experiment as runner


class ArtifactBudgetVisibilityTests(unittest.TestCase):
    def test_all_regular_names_including_hardlink_aliases_keep_original_byte_accounting(self):
        with tempfile.TemporaryDirectory() as f:
            root=Path(f)
            for directory,data in [('.venv',b'abc'),('agents',b'abcd'),('runs/one',b'abcde'),('official',b'excluded original')]:
                path=root/directory/'file';path.parent.mkdir(parents=True);path.write_bytes(data)
            os.link(root/'runs/one/file',root/'runs/one/retained-alias')
            try:os.symlink(root/'official/file',root/'runs/one/symlink')
            except OSError:pass
            with patch.object(runner,'ROOT',root):self.assertEqual(runner.artifact_gib()*2**30,17)

    def test_inaccessible_subtree_never_silently_frees_workspace_budget(self):
        with tempfile.TemporaryDirectory() as f:
            root=Path(f);(root/'.venv').mkdir()
            def denied(folder,**kwargs):
                kwargs['onerror'](PermissionError(5,'Observed project-owned old directory is unreadable'))
                return iter(())
            with patch.object(runner,'ROOT',root),patch.object(runner.os,'walk',side_effect=denied):
                with self.assertRaises(PermissionError):runner.artifact_gib()


if __name__=='__main__':unittest.main()
