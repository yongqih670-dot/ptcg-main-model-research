import unittest
from scripts.analyze_panel import panel_conclusions


class OriginalPanelGateTests(unittest.TestCase):
    def panel(self,original_result):
        groups={};contrasts=[]
        for i in range(8):
            groups[f'a{i}']=[{'result':original_result if i<4 else 'win'}]*30
            groups[f'b{i}']=[{'result':'loss'}]*30
            contrasts.append({'candidate_cell':f'a{i}','anchor_cell':f'b{i}','weight':.125})
        original=[{**c,'weight':.25} for c in contrasts[:4]]
        return groups,contrasts,original

    def test_added_easy_opponents_cannot_hide_unchanged_original_panel(self):
        report=panel_conclusions(*self.panel('loss'))
        self.assertGreater(report['contrast']['ci95'][0],0)
        self.assertEqual(report['original_primary_contrast']['gain'],0)
        self.assertEqual(report['classification'],'inconclusive')

    def test_both_panels_must_show_positive_gain(self):
        report=panel_conclusions(*self.panel('win'))
        self.assertTrue(report['both_preregistered_panels_positive'])
        self.assertEqual(report['classification'],'improved')

    def test_original_regression_remains_regression_despite_full_panel_gain(self):
        groups,contrasts,original=self.panel('loss')
        for i in range(4):groups[f'b{i}']=[{'result':'draw'}]*30
        report=panel_conclusions(groups,contrasts,original)
        self.assertGreater(report['contrast']['ci95'][0],0)
        self.assertEqual(report['classification'],'regressed')
