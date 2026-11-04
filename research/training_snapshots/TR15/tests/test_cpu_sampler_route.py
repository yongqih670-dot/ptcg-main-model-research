from pathlib import Path
import tempfile
import unittest
from ptcg_lab.server_trial import trial_source_files
from ptcg_lab.io import sha256


class CPUSamplerRouteTests(unittest.TestCase):
    def test_only_complete_reviewed_CPU_prototype_can_override_payload(self):
        root=Path(__file__).resolve().parents[1]
        staged=root/'research/proposals/transformer_rl_v3/cpu_sampling_runtime/source'
        names=['training_control.py','rl_sampling.py','rl_league.py','rl_collector.py','rl_resume.py']
        rows=[{'target':'ptcg_lab/'+n,'path':(staged/'ptcg_lab'/n).relative_to(root).as_posix(),
               'sha256':sha256(staged/'ptcg_lab'/n)} for n in names]
        config={'check_variant':'cpu_sampler','engineering_only':True,'prototype_source_overrides':rows}
        files=trial_source_files(root,root,config)
        self.assertEqual(files['ptcg_lab/rl_sampling.py'],(staged/'ptcg_lab/rl_sampling.py').read_bytes())
        for altered in [dict(config,engineering_only=False),dict(config,check_variant='fresh_scientific'),
                        dict(config,prototype_source_overrides=rows[:-1]),
                        dict(config,prototype_source_overrides=[{**rows[0],'sha256':'0'*64}]+rows[1:]),
                        {'check_variant':'cpu_sampler','engineering_only':True}]:
            with self.assertRaises(ValueError):trial_source_files(root,root,altered)
        self.assertEqual((root/'ptcg_lab/rl_core.py').read_bytes(),files['ptcg_lab/rl_core.py'])

