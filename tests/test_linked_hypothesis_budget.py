from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts import run_experiment as supervisor
from ptcg_lab.io import atomic_json


class LinkedHypothesisBudgetTests(unittest.TestCase):
    def test_new_linked_smoke_counts_with_parent_without_retagging_legacy_costs(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            rows=[('main',{'hypothesis_id':'TR05'},10),
                  ('smoke',{'hypothesis_id':'TR05_SMOKE','budget_hypothesis_id':'TR05'},3),
                  ('legacy',{'hypothesis_id':'TR04_SMOKE'},7)]
            for name,config,seconds in rows:
                atomic_json(root/'runs'/name/'config.json',config)
                atomic_json(root/'runs'/name/'attempts.json',[{'attempt':1,'seconds':seconds}])
            with patch.object(supervisor,'ROOT',root):
                self.assertEqual(supervisor.hypothesis_seconds({'hypothesis_id':'TR05'},{}),13)
                self.assertEqual(supervisor.hypothesis_seconds({'hypothesis_id':'TR05_SMOKE','budget_hypothesis_id':'TR05'},{}),13)
                self.assertEqual(supervisor.hypothesis_seconds({'hypothesis_id':'TR04'},{}),0)
                self.assertEqual(supervisor.hypothesis_seconds({'hypothesis_id':'TR04_SMOKE'},{}),7)
