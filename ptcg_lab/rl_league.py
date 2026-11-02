"""Training-only mixture with immutable, hashed learner history."""
import copy
import random
import os
from pathlib import Path
import torch
from .io import sha256,atomic_json,read_json
from .rl_core import choose_decision,load_full_actor
from .public_agent import PublicRuleAgent
from .tensor_io import atomic_torch_save,load_torch


class FrozenOpponent:
    def __init__(self,model,features,deck,identity):
        self.model=model;self.features=features;self.deck=list(deck);self.identity=identity
        self.tracker=None;self.round=None;self.forced=0
    def choose(self,obs):
        rd=obs['current']['round']
        if rd!=self.round:self.round=rd;self.tracker=self.features.Tracker(self.deck);self.forced=0
        self.tracker.update(obs)
        action,d=choose_decision(self.model,self.features,obs,self.tracker,self.forced,greedy=True)
        self.forced=self.forced+1 if d is None else 0
        return action
    def close(self):pass


def choose_bucket(rng,history_available,weights,history_startup='fixed_diverse'):
    weights=dict(weights)
    allowed={'fixed_diverse','historical_learner','latest_frozen_learner','weakness_training_opponents'}
    if (set(weights)!=allowed or any(type(w) not in (int,float) or not 0<=w<=1 for w in weights.values())
            or abs(sum(weights.values())-1)>1e-9 or history_startup not in ('fixed_diverse','latest_frozen_learner')):
        raise ValueError('Invalid league mixture')
    if not history_available:
        weights[history_startup]+=weights['historical_learner'];weights['historical_learner']=0.
    return rng.choices(list(weights),weights=list(weights.values()))[0]


class League:
    def __init__(self,source,output,config,learner,features,deck,opponent_template=None):
        self.source=Path(source);self.output=Path(output);self.config=config
        self.features=features;self.deck=deck;self.learner=learner;self.cache=None;self.cache_id=None
        self.opponent_template=opponent_template
        self.history_window=config.get('history_window')
        if self.history_window is not None and (type(self.history_window) is not int or self.history_window<1):
            raise ValueError('History window must be positive')
        self.manifest=self.output/'league.json'
        self.sampling={'history_window':self.history_window,'history_startup':config.get('history_startup','fixed_diverse')}
        if self.manifest.exists():
            previous=read_json(self.manifest)
            old_sampling=previous.get('sampling',{'history_window':None,'history_startup':'fixed_diverse'})
            if previous['weights']!=config['weights'] or old_sampling!=self.sampling:
                raise ValueError('Resume cannot change the league sampling distribution')
        self.policy_schema=learner.metadata() if hasattr(learner,'options') else None
        if self.manifest.exists() and read_json(self.manifest).get('policy_schema')!=self.policy_schema:
            raise ValueError('League model semantics changed; start a new league')
        self.entries=read_json(self.manifest)['snapshots'] if self.manifest.exists() else []
        self.stats=read_json(self.manifest).get('fixed_training_results',{}) if self.manifest.exists() else {}
        self.retained=config.get('retained_history')
        if self.retained:
            from .server_workspace import owned_root,inside
            from .archive_prefix import relative_path
            root=owned_root(os.environ['PTCG_PROJECT_ROOT'],os.environ['PTCG_PROJECT_ID'])
            self.retained_root=inside(root,self.retained['root'])
            members=self.retained['members']
            if not members or any(relative_path(path)!=path or not path.startswith('league/') for path in members):raise ValueError('Retained history catalog escaped')
            if {e['path']:e['sha256'] for e in self.entries if e['path'] in members}!=members:
                raise ValueError('Retained history catalog does not match full original league')
        for entry in self.entries:
            if sha256(self.member_path(entry))!=entry['sha256']:raise ValueError('Frozen league member changed')
        for spec in config['fixed_opponents']:
            for name,digest in spec['files'].items():
                if sha256(self.source/spec['path']/name)!=digest:raise ValueError('Fixed training opponent changed')
    def save(self):
        atomic_json(self.manifest,{'role':'training_only','weights':self.config['weights'],'snapshots':self.entries,
            'fixed_training_results':self.stats,'no_final_holdout':True,
            **({'sampling':self.sampling} if self.history_window is not None or self.sampling['history_startup']!='fixed_diverse' else {}),
            **({'policy_schema':self.policy_schema} if self.policy_schema else {})})
    def member_path(self,entry):
        if self.retained and entry['path'] in self.retained['members']:
            from .server_workspace import inside
            if self.retained['members'][entry['path']]!=entry['sha256']:raise ValueError('Retained history SHA differs')
            return inside(self.retained_root,self.retained_root/entry['path'])
        return self.output/entry['path']
    def freeze(self,version):
        name=f'league/learner-{version:04d}.pt';path=self.output/name
        if any(e['version']==version for e in self.entries):return
        path.parent.mkdir(exist_ok=True)
        state={k:v.detach().cpu() for k,v in self.learner.state_dict().items()}
        if path.exists():
            previous=load_torch(path,map_location='cpu',weights_only=True)
            if set(previous)!=set(state) or any(not torch.equal(previous[k],state[k]) for k in state):
                raise ValueError('Uncommitted league snapshot does not match restored learner')
        else:
            atomic_torch_save(path,state,replace=False)
        self.entries.append({'id':f'learner-{version:04d}','version':version,'path':name,'sha256':sha256(path)})
        self.save()
    def select(self,seat,seed):
        bucket=choose_bucket(random,len(self.entries)>1,self.config['weights'],self.config.get('history_startup','fixed_diverse'))
        if bucket in ('fixed_diverse','weakness_training_opponents'):
            pool=self.config['fixed_opponents']
            if bucket=='weakness_training_opponents':
                # Laplace-smoothed training losses only; evaluation weights never change.
                weights=[(self.stats.get(s['id'],{}).get('losses',0)+1)/(self.stats.get(s['id'],{}).get('n',0)+2) for s in pool]
            else:weights=[s['weight'] for s in pool]
            spec=random.choices(pool,weights=weights)[0]
            opponent=PublicRuleAgent(spec,seat,seed)
            opponent.identity.update(model_sha256=spec.get('files',{}).get('model.pt'),lineage=spec['lineage'])
            return opponent,{'bucket':bucket,'opponent_id':spec['id'],'fixed':True}
        history=self.entries[:-1]
        if self.history_window:history=history[-self.history_window:]
        entry=self.entries[-1] if bucket=='latest_frozen_learner' else random.choice(history)
        path=self.member_path(entry)
        if sha256(path)!=entry['sha256']:raise ValueError('League member changed before BO3')
        if self.cache_id!=entry['id']:
            template=self.opponent_template if self.opponent_template is not None else self.learner
            self.cache=copy.deepcopy(template).cpu().eval()
            self.cache.load_state_dict(load_torch(path,map_location='cpu',weights_only=True),strict=True)
            for p in self.cache.parameters():p.requires_grad_(False)
            self.cache.freeze_tables();self.cache_id=entry['id']
        return FrozenOpponent(self.cache,self.features,self.deck,dict(entry)),{'bucket':bucket,'opponent_id':entry['id'],'fixed':False}
    def observe(self,assignment,reward):
        if assignment['fixed']:
            row=self.stats.setdefault(assignment['opponent_id'],{'n':0,'losses':0.})
            row['n']+=1;row['losses']+=1 if reward<0 else .5 if reward==0 else 0
            self.save()
