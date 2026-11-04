"""Latest frozen self opponent, with update-boundary and restore provenance."""
from pathlib import Path
from .rl_league import League
from .io import read_json,atomic_json,sha256,utcnow
from .tensor_io import load_torch
from .framework_v2_training import semantic_digest

WEIGHTS={'fixed_diverse':0.,'historical_learner':0.,'latest_frozen_learner':1.,'weakness_training_opponents':0.}

class SelfPlayLeague(League):
    def __init__(self,*args,partial_in_progress=False,**kwargs):
        self.partial_on_adoption=partial_in_progress;self.adopted=False;self.verified_version=None;self.events=[]
        super().__init__(*args,**kwargs)
        if self.config['weights']!=WEIGHTS or len(self.entries)<2:raise ValueError('TR15 permits latest frozen self only, without fixed fallback')

    def freeze(self,version):
        if self.verified_version==version:return
        before=self.entries[-1]['version']
        if version not in (before,before+1):raise ValueError('Snapshot refresh must follow exactly one full version commit')
        existing=any(e['version']==version for e in self.entries)
        if existing and self.adopted:raise ValueError('A committed new version cannot adopt a stale snapshot')
        super().freeze(version)
        entry=self.entries[-1]
        if entry['version']!=version or sha256(self.member_path(entry))!=entry['sha256']:raise ValueError('Frozen self snapshot identity differs')
        # A restored partial update deliberately retains its pre-update frozen
        # opponent, while the live actor already contains partial optimizer work.
        partial_adoption=existing and not self.adopted and self.partial_on_adoption
        if not partial_adoption:
            saved=load_torch(self.member_path(entry),map_location='cpu',weights_only=True)
            if semantic_digest(saved)!=semantic_digest(self.learner.state_dict()):raise ValueError('Frozen snapshot does not match fully committed live actor')
        self.save()  # Forward metadata records the newly authorized mixture.
        self.events.append({'at':utcnow(),'version':version,'snapshot_sha256':entry['sha256'],
            'event':'partial_restore_adopt_preupdate_opponent' if partial_adoption else 'complete_parent_adoption' if existing else 'full_PPO_commit_snapshot_refresh',
            'opponent_actions':'existing_FrozenOpponent_greedy','snapshot_matches_committed_actor':not partial_adoption})
        atomic_json(self.output/'selfplay-snapshot-audit.json',{'schema':1,'weights':WEIGHTS,'latest_version':version,'events':self.events,
            'opponent_fixed_throughout_BO3':True,'refresh_only_after_full_PPO_commit':True})
        self.verified_version=version;self.adopted=True

    def select(self,seat,seed):
        opponent,assignment=super().select(seat,seed);entry=self.entries[-1]
        if assignment['bucket']!='latest_frozen_learner' or assignment['fixed'] or self.cache_id!=entry['id']:
            raise ValueError('TR15 selected anything other than latest frozen self')
        if opponent.identity['sha256']!=entry['sha256'] or entry['version']!=self.verified_version:
            raise ValueError('Loaded opponent differs from adopted snapshot')
        assignment.update(selfplay=True,opponent_policy_version=entry['version'],opponent_snapshot_sha256=entry['sha256'],
            opponent_actions='existing_FrozenOpponent_greedy',snapshot_fixed_for_whole_BO3=True)
        return opponent,assignment
