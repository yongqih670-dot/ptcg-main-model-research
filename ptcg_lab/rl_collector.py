"""One learner seat; transitions connect that seat across opponent decisions."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from .io import read_json,sha256
from .rl_core import choose_decision
from .agents import validate_action
from .token_features import forced_selection
from .phase_timing import timed,span


class Engine:
    def __init__(self,project,source,log,manifest_hash):
        project=Path(project);manifest=Path(os.environ.get('PTCG_ENGINE_MANIFEST_PATH',str(project/'research/environment/manifest.json')))
        if os.environ.get('PTCG_PROJECT_ROOT'):
            from .server_workspace import inside
            inside(Path(os.environ['PTCG_PROJECT_ROOT']),manifest)
        if sha256(manifest)!=manifest_hash:raise ValueError('Official engine manifest changed')
        for item in read_json(manifest)['files'].values():
            path=Path(item['path'])
            if not path.is_absolute():path=project/path
            if sha256(path)!=item['sha256']:raise ValueError('Official runtime file changed')
        if os.environ.get('PTCG_PROJECT_ROOT'):
            from .server_workspace import owned_root,inside
            root=owned_root(os.environ['PTCG_PROJECT_ROOT'],os.environ['PTCG_PROJECT_ID'])
            inside(root,log)
        self.log=open(log,'a',encoding='utf-8')
        interpreter=project/('.venv/bin/python' if sys.platform=='linux' else '.venv/Scripts/python.exe')
        self.process=subprocess.Popen([str(interpreter),'-B','-m','ptcg_lab.rl_engine_worker'],
            cwd=source,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,encoding='utf-8',
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1'))
        if self.read()!= {'event':'ready'}:raise ValueError('Engine startup failed')
    @timed('environment_IPC_send')
    def send(self,obj):self.process.stdin.write(json.dumps(obj)+'\n');self.process.stdin.flush()
    @timed('environment_IPC_read_wait')
    def read(self):
        line=self.process.stdout.readline()
        if not line:raise RuntimeError('Official engine worker exited; inspect log')
        return json.loads(line)
    def close(self):
        if self.process.poll() is None:
            self.send({'command':'close'})
            try:self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:self.process.kill();self.process.wait(timeout=5)
        self.log.close()


@timed('complete_BO3_collection')
def collect_series(engine,model,features,deck,opponent,learner_seat,series_id,policy_version,progress=None,greedy=False,control=None,actor_chooser=None):
    decks=[None,None];decks[learner_seat]=deck;decks[1-learner_seat]=opponent.deck
    engine.send({'command':'bo3','decks':decks})
    trajectories=[];current=[];round_id=None;tracker=None;forced=0;last_scores=[0,0,0]
    counts={'engine_decisions':0,'learner_decisions':0,'forced':0,'opponent_decisions':0,'conditional_picks':0}
    start=time.monotonic();inference=0.;sample_observations=[]
    def finish(scores):
        nonlocal current,last_scores
        delta=[scores[i]-last_scores[i] for i in range(3)]
        if sum(delta)!=1:raise ValueError('Round reward boundary ambiguous')
        reward=float(delta[learner_seat]-delta[1-learner_seat])
        if current:current[-1].update(reward=reward,terminal=True)
        trajectories.append({'round':round_id,'reward':reward,'decisions':current})
        current=[];last_scores=list(scores)
    while True:
        if control:control()
        message=engine.read()
        if message['event']=='series_end':
            if message['statuses']!=['DONE','DONE']:raise ValueError('Official BO3 failed: '+str(message))
            if round_id is not None:finish(message['scores'])
            opponent.close()
            return trajectories,{**counts,**message,'series_id':series_id,'learner_seat':learner_seat,
                'policy_version':policy_version,'opponent_identity':opponent.identity,'seconds':time.monotonic()-start,
                'opponent_diagnostics':getattr(opponent,'diagnostics',[]),
                'learner_inference_seconds':inference,'rounds':len(trajectories),'sample_observations':sample_observations}
        obs=message['observation'];counts['engine_decisions']+=1
        new_round=obs['current']['round']
        if new_round!=round_id:
            if round_id is not None:
                cur=obs['current'];finish([cur['players'][0]['win'],cur['players'][1]['win'],cur['draw']])
            round_id=new_round;tracker=features.Tracker(deck);forced=0
        if message['seat']==learner_seat:
            if obs['current']['yourIndex']!=learner_seat:raise ValueError('Learner viewpoint mismatch')
            with span('feature_tracking'):tracker.update(obs)
            t=time.monotonic()
            action,decision=(actor_chooser or choose_decision)(model,features,obs,tracker,forced,greedy)
            inference+=time.monotonic()-t
            if decision is None:counts['forced']+=1;forced+=1
            else:
                decision.update(series_id=series_id,round=round_id,learner_seat=learner_seat,policy_version=policy_version)
                current.append(decision);forced=0;counts['learner_decisions']+=1;counts['conditional_picks']+=len(decision['steps'])
                if len(sample_observations)<8:sample_observations.append(obs)
        else:
            with span('opponent_features_and_inference'):action=opponent.choose(obs)
            counts['opponent_decisions']+=1
        validate_action(obs,action);engine.send({'action':action})
        if progress and counts['engine_decisions']%50==0:progress(counts)
