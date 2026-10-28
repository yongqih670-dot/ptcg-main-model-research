"""Small inference/actor-information-flow check; not a strength benchmark."""
import json
import sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ptcg_lab.public_agent import PublicRuleAgent
from ptcg_lab.io import read_json,atomic_json,utcnow

episode=read_json(ROOT/'runs/20261023T104927Z_public-rule-smoke_debc0c/episode-00000.json')
observations=[p['observation'] for step in episode['steps'] for p in step
              if p['status']=='ACTIVE' and p['observation'].get('select')
              and len(p['observation']['select']['option'])>1]
obs=observations[0]
spec={'id':'ppo','kind':'public_rule','path':'agents/public/ppo','lineage':'dipam_unown_gradiant_ppo'}
agent=PublicRuleAgent(spec,obs['current']['yourIndex'],20261023)
try:
    deck=agent({'select':None,'current':None,'logs':[],'search_begin_input':None})
    action=agent(obs)
    from ptcg.rl.buffers import OFFSETS
    from ptcg.rl.encoder import encode
    sa=agent.module.SA
    sa.tracker.update(obs)
    encode(sa.row[0],obs,[],False,0,tracker=sa.tracker)
    base=torch.from_numpy(sa.row.copy())
    start,end=OFFSETS['oracle']
    changed=base.clone()
    generator=torch.Generator().manual_seed(20261023)
    changed[:,start:end]=torch.randn(1,end-start,generator=generator)*10
    with torch.no_grad():
        left=sa.policy.forward_policy(base)
        right=sa.policy.forward_policy(changed)
    assert torch.equal(left,right), 'Actor logits changed when only oracle block changed'
    assert torch.isfinite(left).all()
    assert agent.module.GUARD['panic']==0 and agent.module.random.fallback_calls==0
    report={'at':utcnow(),'torch':torch.__version__,'device':'cpu','deck_count':len(deck),
            'legal_action':action,'finite_logits':True,'oracle_perturbation_max_abs_delta':float((left-right).abs().max()),
            'scope':'one observed multi-option state from a different deck: ABI and structural check only, not competitive evidence',
            'BO3_validation':'pending'}
    atomic_json(ROOT/'research/environment/ppo-inference-smoke.json',report)
    print(json.dumps(report))
finally:
    agent.close()
