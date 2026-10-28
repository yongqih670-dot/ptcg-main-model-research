import unittest
from ptcg_lab.io import atomic_json
from ptcg_lab.stage_control import close_hypothesis
from tests.test_stage_control import StageTests


class ClosureTests(StageTests):
    def setup_evaluation(self,classification='inconclusive',complete=True):
        row=self.s['research_stage']['hypotheses']['TR01'];row.update(status='started',run_ids=['train','screen'])
        self.s['research_stage'].update(completed_valid_hypotheses=0,diagnostic_hypotheses_completed=0,events=[])
        atomic_json(self.root/'runs/screen/config.json',{'hypothesis_id':'TR01','job_kind':'evaluation'})
        atomic_json(self.root/'runs/screen/metrics.json',{'complete':complete})
        atomic_json(self.root/'runs/screen/panel-analysis.json',{'complete':complete,'classification':classification})
    def test_two_raw_jobs_count_one_valid_hypothesis(self):
        self.setup_evaluation();close_hypothesis(self.root,self.p,self.s,'TR01','screen')
        stage=self.s['research_stage'];self.assertEqual(stage['completed_valid_hypotheses'],1)
        self.assertEqual(stage['consecutive_valid_hypotheses_without_promotion'],1)
        self.assertEqual(self.s['consecutive_valid_runs_without_promotion'],7)
        with self.assertRaises(RuntimeError):close_hypothesis(self.root,self.p,self.s,'TR01','screen')
    def test_positive_screen_does_not_promote_or_reset(self):
        self.setup_evaluation('improved');self.s['research_stage']['consecutive_valid_hypotheses_without_promotion']=3
        row=close_hypothesis(self.root,self.p,self.s,'TR01','screen')
        self.assertFalse(row['promotion_eligible']);self.assertEqual(row['classification'],'inconclusive')
        self.assertEqual(self.s['research_stage']['consecutive_valid_hypotheses_without_promotion'],4)
    def test_incomplete_or_unrelated_evaluation_cannot_close(self):
        self.setup_evaluation(complete=False)
        with self.assertRaises(RuntimeError):close_hypothesis(self.root,self.p,self.s,'TR01','screen')
        with self.assertRaises(RuntimeError):close_hypothesis(self.root,self.p,self.s,'TR01','another')
    def test_invalid_run_counts_compute_but_not_valid_streak(self):
        self.setup_evaluation('invalid');close_hypothesis(self.root,self.p,self.s,'TR01','screen')
        self.assertEqual(self.s['research_stage']['completed_valid_hypotheses'],0)
        self.assertEqual(self.s['budget_used']['executed_research_experiments'],10)


if __name__=='__main__':unittest.main()
