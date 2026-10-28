import unittest
from ptcg_lab.public_agent import PublicRuleAgent, install_public_api
from ptcg_lab.evaluate import series_config


class PublicAgentTests(unittest.TestCase):
    def test_no_native_or_hidden_state_api(self):
        install_public_api()
        from cg import api, sim
        self.assertEqual(len(api.all_card_data()), 1431)
        self.assertFalse(hasattr(api, 'search_begin'))
        self.assertFalse(hasattr(sim.lib, 'GetBattleData'))

    def test_decks_and_distinct_per_seat_modules(self):
        for name in ('lucario','alakazam','archaludon'):
            with self.subTest(name=name):
                spec={'id':name,'path':'agents/public/'+name,'lineage':'cggonzalezsc_lucario' if name=='lucario' else name}
                a,b=PublicRuleAgent(spec,0,13),PublicRuleAgent(spec,1,13)
                try:
                    obs={'select':None,'current':None,'logs':[],'search_begin_input':None}
                    self.assertEqual(a(obs),a.deck)
                    self.assertEqual(b(obs),b.deck)
                    self.assertIsNot(a.module,b.module)
                finally:
                    a.close(); b.close()

    def test_panel_cannot_drop_or_add_series(self):
        c={'series_count':4,'matchups':[{'cell_id':'a','series_count':2},{'cell_id':'b','series_count':2}]}
        self.assertEqual([series_config(c,i)['cell_id'] for i in range(4)],['a','a','b','b'])
        c['series_count']=5
        with self.assertRaises(ValueError): series_config(c,0)


if __name__=='__main__': unittest.main()
