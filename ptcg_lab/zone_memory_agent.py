"""Explicit-state CPU evaluation adapter for opt-in zone/GRU checkpoints."""
from pathlib import Path
import torch
from .agents import ProtocolAgent
from .io import sha256
from .replay_data import actor_observation
from .rl_core import load_full_actor,choose_decision


class ZoneMemoryAgent(ProtocolAgent):
    def __init__(self,spec,seat,seed):
        root=Path(__file__).resolve().parents[1]
        self.folder=(root/spec['path']).resolve()
        if root not in self.folder.parents:raise ValueError('Agent escaped source root')
        deck=list(map(int,(self.folder/'deck.csv').read_text(encoding='utf-8-sig').split()))
        super().__init__('first',deck,seat,seed)
        self.spec=spec;self.name=spec['id'];self.model=None
        self.tracker=None;self.round=None;self.forced=0;self.diagnostics=[]
        self.identity={'id':self.name,'model_sha256':spec['model_sha256'],
                       'model_extensions':spec['model_extensions'],'adapter':'zone-gru-v2'}

    def choose(self,obs):
        if obs.get('select') is None:
            self.tracker=None;self.round=None;return list(self.deck)
        if self.model is None:
            with torch.random.fork_rng(devices=[]):
                self.model,self.features,_=load_full_actor(self.folder,self.spec['model_sha256'],reset_value=False,
                    model_extensions=self.spec['model_extensions'])
            self.model.freeze_tables()
        obs=actor_observation(obs)
        rd=obs['current']['round']
        if rd!=self.round:
            self.tracker=self.features.Tracker(self.deck);self.round=rd;self.forced=0
        self.tracker.update(obs)
        action,decision=choose_decision(self.model,self.features,obs,self.tracker,self.forced,greedy=True)
        self.forced=self.forced+1 if decision is None else 0
        return action

    def close(self):
        self.tracker=None;self.round=None
