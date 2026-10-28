import unittest
from packaging.requirements import Requirement
from ptcg_lab.ranged_wheelhouse import supported_tags,select_wheel,validate_closure,check_scope_budget


class RangedWheelhouseTests(unittest.TestCase):
    def test_windows_or_wrong_hash_wheel_cannot_enter_linux_capsule(self):
        def item(filename,digest):return {'filename':filename,'digests':{'sha256':digest},'requires_python':'>=3.10','yanked':False}
        m={'urls':[item('fixture-1-cp312-cp312-win_amd64.whl','a'),item('fixture-1-cp312-cp312-manylinux_2_28_x86_64.whl','b')]}
        r=select_wheel(m,Requirement('fixture==1'),{'b'},supported_tags(39));self.assertIn('manylinux',r['filename'])
        with self.assertRaises(ValueError):select_wheel(m,Requirement('fixture==1'),{'a'},supported_tags(39))
    def test_requested_cuda_extras_are_checked_against_actual_wheel_metadata(self):
        records=[{'name':'torch','version':'2.11.0','requires_dist':['cuda-toolkit[cublas]==12.8.1']},
                 {'name':'cuda-toolkit','version':'12.8.1','requires_dist':['nvidia-cublas-cu12==12.8.4.1; extra == "cublas"']}]
        with self.assertRaises(ValueError):validate_closure(records)
        records.append({'name':'nvidia-cublas-cu12','version':'12.8.4.1','requires_dist':[]})
        self.assertTrue(validate_closure(records))
    def test_wrong_dependency_version_cannot_be_treated_as_complete(self):
        with self.assertRaises(ValueError):validate_closure([{'name':'a','version':'1','requires_dist':['b==2']},{'name':'b','version':'1','requires_dist':[]}])
    def test_parent_cost_and_closed_attempts_cannot_be_reset(self):
        check_scope_budget(460.562,1800,2400,[])
        with self.assertRaises(RuntimeError):check_scope_budget(460.562,2000,2400,[])
        with self.assertRaises(RuntimeError):check_scope_budget(460.562,1200,2400,[{'status':'failed'}])
