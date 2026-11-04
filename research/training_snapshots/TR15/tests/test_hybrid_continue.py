import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from ptcg_lab.io import atomic_json,sha256
from ptcg_lab.hybrid_continue import validate_original_attempt_scope
from ptcg_lab.rl_league import League


class HybridContinuationTests(unittest.TestCase):
    def test_original_attempt_limit_never_resets_on_new_host(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);atomic_json(root/'runs/original/attempts.json',[{'seconds':7160.515}])
            config={'original_training_run_id':'original','original_attempt_number':2,'hypothesis_id':'TR04','maximum_attempts':1}
            validate_original_attempt_scope(root,config,time.time()+3599)
            with self.assertRaises(ValueError):validate_original_attempt_scope(root,config,time.time()+3602)
            with self.assertRaises(ValueError):validate_original_attempt_scope(root,{**config,'maximum_attempts':2},time.time()+100)
            atomic_json(root/'runs/original/attempts.json',[{'seconds':7160.515},{'seconds':100}])
            with self.assertRaises(ValueError):validate_original_attempt_scope(root,config,time.time()+100)
    def test_full_readonly_history_catalog_and_hashes_are_required(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'source';source.mkdir();output=root/'new';output.mkdir();old=root/'old';old.mkdir()
            uid=os.getuid() if hasattr(os,'getuid') else 1
            root.chmod(0o700);atomic_json(root/'.ptcg-owner.json',{'root':str(root.resolve()),'project_id':'test','uid':uid})
            entries=[]
            for version in range(3):
                path=old/f'league/learner-{version:04d}.pt';path.parent.mkdir(exist_ok=True);path.write_bytes(bytes([version])*100)
                entries.append({'id':f'learner-{version:04d}','version':version,'path':path.relative_to(old).as_posix(),'sha256':sha256(path)})
            atomic_json(output/'league.json',{'snapshots':entries,'fixed_training_results':{}})
            config={'fixed_opponents':[],'weights':{'historical_learner':.3,'fixed_diverse':.4,'latest_frozen_learner':.2,'weakness_training_opponents':.1},
                'retained_history':{'root':str(old),'members':{x['path']:x['sha256'] for x in entries}}}
            with patch.dict(os.environ,{'PTCG_PROJECT_ROOT':str(root),'PTCG_PROJECT_ID':'test'}):
                # The real bounded read context expects server limits; use its
                # normal local JSON/hash reader for this catalog-only fixture.
                with patch('ptcg_lab.rl_league.read_json',side_effect=lambda p:__import__('json').loads(Path(p).read_text(encoding='utf-8'))),patch('ptcg_lab.rl_league.sha256',side_effect=lambda p:__import__('hashlib').sha256(Path(p).read_bytes()).hexdigest()):
                    league=League(source,output,config,None,None,[])
                    self.assertEqual(len(league.entries),3)
                    self.assertEqual(league.member_path(entries[0]),old/entries[0]['path'])
                    self.assertEqual(league.member_path({'path':'league/learner-0003.pt','sha256':'new'}),output/'league/learner-0003.pt')
                    (old/entries[1]['path']).write_bytes(b'changed')
                    with self.assertRaises(ValueError):League(source,output,config,None,None,[])
