import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from ptcg_lab.io import atomic_json, sha256
from ptcg_lab.rl_collector import Engine


class EnginePlatformTests(unittest.TestCase):
    def test_private_interpreter_and_cpu_only_child_for_both_platforms(self):
        for platform, relative in [('linux','.venv/bin/python'), ('win32','.venv/Scripts/python.exe')]:
            with self.subTest(platform=platform), tempfile.TemporaryDirectory() as directory:
                root=Path(directory); artifact=root/'native'; artifact.write_bytes(b'pinned engine')
                manifest=root/'research/environment/manifest.json'
                atomic_json(manifest,{'files':{'native':{'path':'native','sha256':sha256(artifact)}}})
                child=MagicMock(); child.stdout=io.StringIO('{"event":"ready"}\n');child.poll.return_value=0
                with patch('ptcg_lab.rl_collector.subprocess.Popen',return_value=child) as popen, \
                     patch('ptcg_lab.rl_collector.sys.platform',platform), \
                     patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'GPU-parent'}):
                    engine=Engine(root,root,root/'engine.log',sha256(manifest));engine.close()
                args,kwargs=popen.call_args
                self.assertEqual(args[0][0],str(root/relative))
                self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'],'')
                self.assertEqual(kwargs['env']['PYTHONDONTWRITEBYTECODE'],'1')

    def test_changed_runtime_is_rejected_before_process_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); artifact=root/'native';artifact.write_bytes(b'changed')
            manifest=root/'research/environment/manifest.json'
            atomic_json(manifest,{'files':{'native':{'path':'native','sha256':'0'*64}}})
            with patch('ptcg_lab.rl_collector.subprocess.Popen') as popen:
                with self.assertRaisesRegex(ValueError,'Official runtime file changed'):
                    Engine(root,root,root/'engine.log',sha256(manifest))
                popen.assert_not_called()
