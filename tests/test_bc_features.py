import copy
import gzip
import json
import unittest
from pathlib import Path
import numpy as np
from ptcg_lab.bc_features import CandidateFeatures,supported
from ptcg_lab.replay_data import extract_episode,actor_observation

ROOT=Path(__file__).resolve().parents[1]


class BCFeaturesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=ROOT/'runs/20261023T111657Z_h002-seat-confirmation_c49f14/episode-00000.json.gz'
        with gzip.open(path,'rt',encoding='utf-8') as f:episode=json.load(f)
        cls.records,_=extract_episode(episode,'feature-test-series')

    def test_tail_options_are_represented_and_private_fields_ignored(self):
        feature=CandidateFeatures(ROOT/'agents/frozen/ppo-zero-anchor')
        decision=next(r for r in self.records if supported(r['observation']['select']))
        obs=copy.deepcopy(decision['observation']);tracker=feature.Tracker(decision['own_deck']);tracker.update(obs)
        # A synthetic long number choice isolates capacity from game strategy.
        obs['select']['option']=[{'type':0,'number':i} for i in range(140)]
        x=feature.make(obs,tracker,0)
        self.assertEqual(x.shape,(140,224))
        self.assertFalse(np.array_equal(x[0],x[-1]))
        obs['visualize']={'opponent_hand':[9999]};obs['search_begin_input']='HIDDEN';obs['future_draw']=[8888]
        np.testing.assert_array_equal(x,feature.make(obs,tracker,0))

    def test_grouped_extraction_and_runtime_feature_path_agree(self):
        f=CandidateFeatures(ROOT/'agents/frozen/ppo-zero-anchor');seat=0
        selected=[r for r in self.records if r['seat']==seat]
        tracker=None;round_id=None;forced=0;checked=0
        for r in selected:
            obs=r['observation'];sel=obs['select'];n=len(sel['option'])
            if round_id!=r['round']:tracker=f.Tracker(r['own_deck']);round_id=r['round'];forced=0
            tracker.update(obs)
            if sel['minCount']==sel['maxCount']==n or n==1 and sel['minCount']>=1:forced+=1;continue
            if supported(sel):
                a=f.make(obs,tracker,forced)
                augmented={**obs,'remainingOverageTime':300,'search_begin_input':'OPAQUE'}
                np.testing.assert_array_equal(a,f.make(augmented,tracker,forced));checked+=1
            forced=0
        self.assertGreater(checked,10)


if __name__=='__main__':unittest.main()
