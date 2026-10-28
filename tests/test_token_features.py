import copy
import gzip
import json
import unittest
from pathlib import Path
import numpy as np
from ptcg_lab.token_features import TokenFeatures,conditional_targets,merge_logits
from ptcg_lab.replay_data import extract_episode

ROOT=Path(__file__).resolve().parents[1]


class TokenTests(unittest.TestCase):
    def test_conditional_stop_and_required_counts(self):
        sel={'option':[{}, {}, {}],'minCount':1,'maxCount':3}
        self.assertEqual(conditional_targets(sel,[2,0]),[([],2),([2],0),([2,0],3)])
        self.assertEqual(conditional_targets(sel,[2,0,1]),[([],2),([2],0),([2,0],1)])
        with self.assertRaises(ValueError):conditional_targets(sel,[0,0])
        with self.assertRaises(ValueError):conditional_targets(sel,[])
        self.assertEqual(conditional_targets({**sel,'minCount':0},[]),[([],3)])

    def test_large_option_conditional_mask_and_hidden_invariance(self):
        with gzip.open(ROOT/'runs/20261023T111657Z_h002-seat-confirmation_c49f14/episode-00000.json.gz','rt',encoding='utf-8') as f:episode=json.load(f)
        records,_=extract_episode(episode,'test')
        d=next(r for r in records if len(r['observation']['select']['option'])>1)
        obs=copy.deepcopy(d['observation']);f=TokenFeatures(ROOT/'agents/frozen/ppo-zero-anchor');b=f.b
        obs['select'].update(option=[{'type':0,'number':i} for i in range(140)],minCount=2,maxCount=4)
        tracker=f.Tracker(d['own_deck']);tracker.update(obs)
        rows,mapping=f.make(obs,tracker,0,[2,133])
        self.assertEqual(len(rows),3)
        masks=np.stack([f.field(row,'opt_mask') for row in rows])
        scores=merge_logits(masks,mapping,140,b.STOP)
        self.assertEqual(set(np.where(scores==1)[0]),set(range(141))-{2,133})
        for row in rows:
            self.assertFalse(f.field(row,'oracle').any())
            self.assertAlmostEqual(f.field(row,'dec_float')[b.DecF.PICKED],2/5,places=6)
        obs.update(visualize={'private':[1,2]},search_begin_input='PRIVATE',future_draw=[4])
        np.testing.assert_array_equal(rows,f.make(obs,tracker,0,[2,133])[0])
        # Tail option remains selectable after merging chunk scores.
        logits=np.full_like(masks,-1e9);logits[2,13]=42
        self.assertEqual(int(merge_logits(logits,mapping,140,b.STOP).argmax()),139)

    def test_actor_has_no_oracle_gradient_or_logit_path(self):
        import torch
        from ptcg_lab.transformer_model import build_model
        torch.set_num_threads(2)
        f=TokenFeatures(ROOT/'agents/frozen/ppo-zero-anchor')
        model,_=build_model(ROOT/'agents/frozen/ppo-zero-anchor',{'d':32,'layers':1,'heads':4,'ffn':64})
        x=torch.zeros((2,f.b.OBS_SIZE));a,z=f.b.OFFSETS['opt_mask'];x[:,a:a+2]=1
        a,z=f.b.OFFSETS['oracle'];changed=x.clone();changed[:,a:z]=123
        model.eval()
        torch.testing.assert_close(model.forward_policy(x),model.forward_policy(changed),rtol=0,atol=0)
        model.forward_policy(changed)[:,0].sum().backward()
        self.assertTrue(all(p.grad is None for n,p in model.named_parameters() if 'oracle' in n or 'critic' in n or n.startswith('aux_')))
