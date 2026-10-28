import unittest
from ptcg_lab.tr09_recovery_analysis import canonical_policy_version


class JournalPolicyVersionTests(unittest.TestCase):
    def test_production_string_and_numeric_metadata_describe_same_policy(self):
        self.assertEqual(canonical_policy_version('2',2),2)
        self.assertEqual(canonical_policy_version(2,2),2)
    def test_no_different_boolean_fraction_or_noncanonical_version_is_accepted(self):
        for value in [True,2.0,'02','2.0','2 ','３',3,'3',None]:
            with self.assertRaises(ValueError):canonical_policy_version(value,2)


if __name__=='__main__':unittest.main()
