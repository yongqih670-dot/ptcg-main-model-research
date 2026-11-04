import copy
import unittest
from ptcg_lab.replay_data import extract_episode,actor_observation


class ReplayTests(unittest.TestCase):
    def episode(self):
        empty=lambda:{'status':'ACTIVE','observation':{'select':None},'action':[]}
        first=[empty(),empty()]
        second=[{'status':'ACTIVE' if i==0 else 'INACTIVE','action':[1]*60,
            'observation':{'current':{'yourIndex':i,'round':1},'select':{'minCount':1,'maxCount':1,'option':[{},{}],'context':0},'logs':[]}} for i in range(2)]
        third=[{'status':'DONE','action':[1] if i==0 else [],'observation':{}} for i in range(2)]
        return {'name':'cabt','configuration':{'version':2,'bo':3},'statuses':['DONE','DONE'],'steps':[first,second,third]}

    def test_labels_use_next_step_only_and_private_data_cannot_enter(self):
        e=self.episode();rows,_=extract_episode(e,'one-bo3')
        self.assertEqual(rows[0]['action'],[1])
        self.assertEqual(rows[0]['own_deck'],[1]*60)
        e['steps'][0][0]['visualize']={'future':9}
        e['steps'][1][0]['observation']['search_begin_input']='hidden'
        e['steps'][1][0]['observation']['hidden_truth']=[2]*60
        self.assertEqual(extract_episode(e,'one-bo3')[0],rows)
        e['steps'][2][0]['action']=[7]
        with self.assertRaises(ValueError):extract_episode(e,'one-bo3')

    def test_complete_series_and_actual_actor_required(self):
        e=self.episode();e['statuses']=['ERROR','DONE']
        with self.assertRaises(ValueError):extract_episode(e,'bad')
        e=self.episode();e['steps'][1][0]['observation']['current']['yourIndex']=1
        with self.assertRaises(ValueError):extract_episode(e,'bad')


if __name__=='__main__':unittest.main()
