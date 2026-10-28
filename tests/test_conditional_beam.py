import gzip
import json
import unittest
from pathlib import Path
import numpy as np
from ptcg_lab.conditional_beam import beam_answer,install_beam
from ptcg_lab.public_agent import PublicRuleAgent
from ptcg_lab.replay_data import extract_episode
ROOT=Path(__file__).resolve().parents[1]


class BeamTests(unittest.TestCase):
    def test_joint_sequence_can_beat_greedy_without_length_bonus(self):
        def logits(prefixes):
            table={():[.6,.4,0,0],(0,):[0,.5,.5,0],(1,):[0,0,1,0]}
            return np.array([[np.log(v) if v else -1e9 for v in table[tuple(p)]] for p in prefixes])
        self.assertEqual(beam_answer(3,2,2,1,logits),[0,1])
        self.assertEqual(beam_answer(3,2,2,2,logits),[1,2])

    def test_stop_and_zero_max(self):
        self.assertEqual(beam_answer(2,0,2,4,lambda ps:np.tile([-3.,-4.,0.],(len(ps),1))),[])
        self.assertEqual(beam_answer(2,0,0,4,lambda ps:None),[])

    def test_width_one_matches_public_greedy_on_real_multiselects(self):
        spec={'id':'test','kind':'public_rule','path':'agents/frozen/ppo-zero-anchor','lineage':'dipam_unown_gradiant_ppo','parameters':{'_actor_seat_mode':'zero'}}
        agent=PublicRuleAgent(spec,0,123);agent.load();sa=agent.module.SA;original=sa._decompose
        install_beam(agent.module,1,agent.folder)
        with gzip.open(ROOT/'runs/20261023T111657Z_h002-seat-confirmation_c49f14/episode-00000.json.gz','rt',encoding='utf-8') as f:e=json.load(f)
        records,_=extract_episode(e,'beam-test');checked=0;tracker=None;rd=None;forced=0
        from ptcg.tracker import InfoTracker
        for r in records:
            if r['seat']!=0:continue
            obs=r['observation'];sel=obs['select']
            if r['round']!=rd:tracker=InfoTracker(r['own_deck']);rd=r['round'];forced=0
            tracker.update(obs)
            if sa._forced(sel) is not None:forced+=1;continue
            if sel['maxCount']>1:
                self.assertEqual(original(obs,tracker,forced,sample=False),sa._decompose(obs,tracker,forced,sample=False));checked+=1
            forced=0
        agent.close();self.assertGreater(checked,5)
