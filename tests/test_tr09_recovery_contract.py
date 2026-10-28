import copy
import unittest
from ptcg_lab.tr09_recovery_analysis import FIRST,validate_clean_contract


class TR09RecoveryContractTests(unittest.TestCase):
    def setUp(self):
        self.config={'hypothesis_id':'TR09','check_variant':'fresh_scientific','training_attempt_index':1}
        self.attempts=[{'status':'checkpointed','exit_code':0,'seconds':6600.}]
        self.cursor={'version':2,'trained_decisions':16543,'next_series_id':121}
        self.progress={'stage':'checkpointed_at_soft_deadline',**self.cursor,'pending_decisions':8134}
    def test_only_original_clean_published_cursor(self):
        validate_clean_contract(FIRST,self.config,self.attempts,{'complete':False},self.progress,self.cursor)
        for phase in ['partial_update_abandoned_at_deadline','checkpointed_for_control']:
            p={**self.progress,'stage':phase};validate_clean_contract(FIRST,self.config,self.attempts,{},p,self.cursor)
        p={**self.progress,'stage':'checkpointed_for_control','control_stop_reason':'Verified experiment stop request'}
        validate_clean_contract(FIRST,self.config,self.attempts,{},p,self.cursor)
    def test_never_old_failed_or_renamed_attempts(self):
        for name in ['TR08','TR07','TR10']:
            c={**self.config,'hypothesis_id':name}
            with self.assertRaises(ValueError):validate_clean_contract(FIRST,c,self.attempts,{},self.progress,self.cursor)
        with self.assertRaises(ValueError):validate_clean_contract('renamed',self.config,self.attempts,{},self.progress,self.cursor)
        for status,code in [('failed',1),('failed',0),('wall_timeout',1),('completed',0)]:
            a=[{**self.attempts[0],'status':status,'exit_code':code}]
            with self.assertRaises(ValueError):validate_clean_contract(FIRST,self.config,a,{},self.progress,self.cursor)
    def test_cost_completed_target_and_missing_cursor_are_not_reinterpreted(self):
        for seconds in [0,14400]:
            with self.assertRaises(ValueError):validate_clean_contract(FIRST,self.config,[{**self.attempts[0],'seconds':seconds}],{},self.progress,self.cursor)
        for progress,metrics in [({**self.progress,'version':3},{}),(self.progress,{'complete':True}),({**self.progress,'stage':'collecting'}, {})]:
            with self.assertRaises(ValueError):validate_clean_contract(FIRST,self.config,self.attempts,metrics,progress,self.cursor)
        with self.assertRaises(ValueError):validate_clean_contract(FIRST,self.config,self.attempts*2,{},self.progress,self.cursor)


if __name__=='__main__':unittest.main()
