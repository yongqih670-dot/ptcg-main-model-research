import copy
import unittest
from ptcg_lab.server_trial import trial_outcome


class TrialOutcomeTests(unittest.TestCase):
    def setUp(self):
        self.props={'LoadState':'loaded','Result':'success','ExecMainStatus':'0','ControlGroup':''}
        self.metrics={'complete':False,'training_only':True,'engineering_only':False,
            'actual_config':{'target_decisions':32768},'last_training_progress':{'stage':'checkpointed_for_control','trained_decisions':16679,'pending_decisions':4112}}
    def test_actual_normal_checkpoint_stays_incomplete_and_cannot_promote(self):
        self.assertEqual(trial_outcome('fresh_scientific',self.metrics,self.props),'checkpointed')
        self.assertIs(self.metrics['complete'],False)
        self.assertEqual(trial_outcome('derived_scientific',self.metrics,self.props),'checkpointed')
    def test_complete_and_partial_are_not_interchangeable(self):
        self.assertEqual(trial_outcome('fresh_scientific',{'complete':True},self.props),'complete')
        for mode in ['live_archive_check','short_gpu_trial','basic_cpu_check']:
            with self.assertRaises(RuntimeError):trial_outcome(mode,self.metrics,self.props)
        for stage in ['collecting','optimizing','paused_for_resource','complete']:
            m=copy.deepcopy(self.metrics);m['last_training_progress']['stage']=stage
            with self.assertRaises(RuntimeError):trial_outcome('fresh_scientific',m,self.props)
    def test_nonzero_unclosed_and_collected_service_cannot_be_success(self):
        for field,bad in [('LoadState','not-found'),('Result','exit-code'),('ExecMainStatus','1'),('ControlGroup','/still-live')]:
            p={**self.props,field:bad}
            with self.assertRaises(RuntimeError):trial_outcome('fresh_scientific',self.metrics,p)
        m=copy.deepcopy(self.metrics);m['last_training_progress']['trained_decisions']=32768
        with self.assertRaises(RuntimeError):trial_outcome('fresh_scientific',m,self.props)

    def test_unique_target_can_still_be_incomplete_after_consumed_cursor_passes_target(self):
        m=copy.deepcopy(self.metrics)
        m['last_training_progress'].update(trained_decisions=33295,unique_PPO_trained_decisions=30655)
        self.assertEqual(trial_outcome('derived_scientific',m,self.props),'checkpointed')
        m['last_training_progress']['unique_PPO_trained_decisions']=32768
        with self.assertRaises(RuntimeError):trial_outcome('derived_scientific',m,self.props)


if __name__=='__main__':unittest.main()
