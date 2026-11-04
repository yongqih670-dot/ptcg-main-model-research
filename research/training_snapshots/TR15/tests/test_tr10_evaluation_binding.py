import unittest
from scripts.prepare_ppo_evaluation import bind_candidate,smoke_allocation


class EightOpponentBindingTests(unittest.TestCase):
    def test_all_new_candidate_cells_bind_without_changing_anchor(self):
        template={'matchups':[{'candidate':{'bind':'completed_TR10_model_spec'},'reference':{'id':str(i)}} for i in range(8)]}
        template['matchups'].append({'candidate':{'id':'anchor'},'reference':{'id':'0'}})
        result=bind_candidate(template,{'id':'completed','model_sha256':'a'*64},'TR10')
        self.assertTrue(all(m['candidate']['id']=='completed' for m in result['matchups'][:8]))
        self.assertEqual(result['matchups'][-1]['candidate'],{'id':'anchor'})
        self.assertIn('bind',template['matchups'][0]['candidate'])

    def test_fifty_smoke_series_cover_every_opponent_with_balanced_seats(self):
        counts=smoke_allocation(8)
        self.assertEqual(len(counts),8);self.assertEqual(sum(counts),50)
        self.assertTrue(all(n>0 and n%2==0 for n in counts))
        self.assertEqual(smoke_allocation(4),[14,12,12,12])
