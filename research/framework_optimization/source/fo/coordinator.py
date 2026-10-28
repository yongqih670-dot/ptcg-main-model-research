"""Opt-in synchronous policy-version waves with concurrent engine reads.

Only native engine IPC waits overlap. Opponent actions and conditional inference
run on the owner thread. No actor/learner overlap. Native games cannot be
serialized: an interrupted unfinished BO3 is excluded with its identity, while
complete BO3s remain pending. Resume consumes pending before new assignments.
"""
import copy
from concurrent.futures import ThreadPoolExecutor,wait
import time
import torch
from ptcg_lab.agents import validate_action
from .conditional_batch import choose_requests


class Series:
    def __init__(self,engine,features,deck,opponent,seat,sid,version,assignment):
        self.engine=engine;self.features=features;self.opponent=opponent
        self.seat=seat;self.sid=sid;self.version=version;self.assignment=assignment
        self.round=None;self.tracker=None;self.forced=0;self.current=[];self.trajectories=[];self.scores=[0,0,0]
        self.counts={'engine_decisions':0,'learner_decisions':0,'forced':0,'opponent_decisions':0,'conditional_picks':0}
        self.deck=deck;self.begin=time.monotonic();self.done=False;self.result=None

    def start(self):
        decks=[None,None];decks[self.seat]=self.deck;decks[1-self.seat]=self.opponent.deck
        self.engine.send({'command':'bo3','decks':decks})

    def finish_round(self,scores):
        delta=[scores[i]-self.scores[i] for i in range(3)]
        if sum(delta)!=1:raise ValueError('Ambiguous complete round reward')
        reward=float(delta[self.seat]-delta[1-self.seat])
        if self.current:self.current[-1].update(reward=reward,terminal=True)
        self.trajectories.append({'round':self.round,'reward':reward,'decisions':self.current})
        self.current=[];self.scores=list(scores)

    def observe(self,message,rng):
        if message['event']=='series_end':
            if message['statuses']!=['DONE','DONE']:raise ValueError('Incomplete official BO3')
            if self.round is not None:self.finish_round(message['scores'])
            if not 2<=len(self.trajectories)<=3:raise ValueError('Complete BO3 requires 2 or 3 finished games')
            self.result={'series_id':self.sid,'policy_version':self.version,
                'trajectories':self.trajectories,'summary':dict(message,**self.counts,
                    series_id=self.sid,policy_version=str(self.version),learner_seat=self.seat,
                    assignment=self.assignment,opponent_identity=self.opponent.identity,
                    rounds=len(self.trajectories),seconds=time.monotonic()-self.begin)}
            self.done=True
            return None
        obs=message['observation'];self.counts['engine_decisions']+=1
        if obs['current']['round']!=self.round:
            if self.round is not None:
                cur=obs['current'];self.finish_round([cur['players'][0]['win'],cur['players'][1]['win'],cur['draw']])
            self.round=obs['current']['round'];self.tracker=self.features.Tracker(self.deck);self.forced=0
        if message['seat']!=self.seat:
            action=self.opponent.choose(obs);validate_action(obs,action)
            self.engine.send({'action':action});self.counts['opponent_decisions']+=1
            return None
        if obs['current']['yourIndex']!=self.seat:raise ValueError('Learner viewpoint mismatch')
        self.tracker.update(obs);self.observation=obs
        return {'key':str(self.sid),'observation':obs,'tracker':self.tracker,
                'forced_run':self.forced,'rng':rng}

    def accept(self,action,decision):
        validate_action(self.observation,action)
        if decision is None:self.forced+=1;self.counts['forced']+=1
        else:
            decision.update(series_id=self.sid,round=self.round,learner_seat=self.seat,policy_version=str(self.version))
            self.current.append(decision);self.forced=0;self.counts['learner_decisions']+=1
            self.counts['conditional_picks']+=len(decision['steps'])
        self.engine.send({'action':action})


class Coordinator:
    def __init__(self,workers=1,version=0,next_series_id=0,seeds=(1,2,3),wave_seconds=300):
        if workers not in (1,2,3) or not 0<wave_seconds<=540:raise ValueError('Bounded worker/wave envelope')
        self.workers=workers;self.version=version;self.next_series_id=next_series_id;self.wave_seconds=wave_seconds
        self.rng=[torch.Generator(device='cpu').manual_seed(seeds[i]) for i in range(workers)]
        self.pending=[];self.excluded=[];self.durable=[];self.running=False

    def collect_wave(self,model,features,deck,factory,guard=lambda:None,greedy=False):
        if self.running or self.pending:raise ValueError('Drain durable pending records before next wave')
        if model.training or next(model.parameters()).device.type!='cpu':
            raise ValueError('Frozen CPU sampler required')
        self.running=True;series=[];pool=ThreadPoolExecutor(max_workers=self.workers)
        start=time.monotonic();calls=0
        try:
            for i in range(self.workers):
                sid=self.next_series_id;self.next_series_id+=1
                # Factory creates separately owned engines/opponents; no global
                # RNG work occurs on IPC reader threads.
                try:engine,opponent,assignment=factory(sid,sid%2)
                except BaseException:
                    self.excluded.append({'series_id':sid,'policy_version':self.version,'reason':'factory_failed'})
                    raise
                s=Series(engine,features,deck,opponent,sid%2,sid,self.version,assignment)
                series.append(s)
                if len({id(x.engine) for x in series})!=len(series) or len({id(x.opponent) for x in series})!=len(series):
                    raise ValueError('Workers cannot share engine or opponent mutable state')
                s.start()
            while any(not s.done for s in series):
                guard()
                futures={i:pool.submit(s.engine.read) for i,s in enumerate(series) if not s.done}
                unfinished=set(futures.values())
                while unfinished:
                    guard()
                    if time.monotonic()-start>=self.wave_seconds:raise TimeoutError('BO3 wave bound; unfinished IDs excluded')
                    _,unfinished=wait(unfinished,timeout=.1)
                requests=[]
                for i,f in sorted(futures.items()):
                    request=series[i].observe(f.result(),self.rng[i])
                    if request:requests.append(request)
                if requests:
                    result,stats=choose_requests(model,features,requests,greedy=greedy,before_forward=guard)
                    calls+=stats['forward_calls']
                    for s in series:
                        if str(s.sid) in result:s.accept(*result[str(s.sid)])
            return {'complete_BO3':len(series),'forward_calls':calls,'policy_version':self.version,
                    'seconds':time.monotonic()-start,'actual_unique_optimized_decisions':0}
        finally:
            # Preserve complete records even if another environment failed.
            # Every assigned unfinished native series is explicitly excluded.
            for i,s in enumerate(series):
                if s.done:
                    self.pending.append(dict(s.result,worker_slot=i,sampler_rng_after=self.rng[i].get_state().clone()))
                else:self.excluded.append({'series_id':s.sid,'policy_version':self.version,'reason':'unfinished_native_BO3'})
            close_errors=[]
            try:
                for s in series:
                    for close in (s.engine.close,s.opponent.close):
                        try:close()
                        except Exception as exc:close_errors.append(repr(exc))
            finally:
                pool.shutdown(wait=True,cancel_futures=True);self.running=False
            self.pending.sort(key=lambda x:x['series_id'])
            if close_errors:raise RuntimeError('Owned worker cleanup failed: '+str(close_errors))

    def acknowledge_pending(self,series_ids):
        # Caller must durably write full trajectories/RNG plus summary first.
        # This acknowledgement removes only in-memory coordinator copies.
        expected=[x['series_id'] for x in self.pending]
        if list(series_ids)!=expected:raise ValueError('Complete sorted pending identity differs')
        self.durable.extend({'series_id':x['series_id'],'decisions':sum(len(t['decisions']) for t in x['trajectories'])}
                            for x in self.pending)
        self.pending=[]

    def advance_policy(self,version,update=None):
        if self.running or self.pending or version!=self.version+1:
            raise ValueError('Policy barrier: complete pending wave must be durable and consumed')
        if update is None or update['version']!=version or update['series_ids']!=[x['series_id'] for x in self.durable]:
            raise ValueError('Policy barrier requires exact complete-series update identity')
        from ptcg_lab.ppo_accounting import update_decisions
        if update['decisions']!=sum(x['decisions'] for x in self.durable):raise ValueError('Update input coverage differs')
        covered=update_decisions(update)
        self.version=version;self.durable=[]
        return covered

    def state_dict(self):
        if self.running:raise ValueError('Native in-flight state cannot be serialized')
        return copy.deepcopy({'schema':1,'workers':self.workers,'version':self.version,
            'next_series_id':self.next_series_id,'worker_rng':[g.get_state() for g in self.rng],
            'pending':self.pending,'excluded':self.excluded,'durable':self.durable})

    def load_state_dict(self,state):
        if self.running or state['schema']!=1 or state['workers']!=self.workers:
            raise ValueError('Resume worker layout differs')
        if len(state['worker_rng'])!=self.workers:raise ValueError('Missing worker RNG')
        ids=[x['series_id'] for name in ('pending','excluded','durable') for x in state[name]]
        if len(set(ids))!=len(ids) or any(type(sid) is not int or not 0<=sid<state['next_series_id'] for sid in ids):
            raise ValueError('Restored complete/excluded identities overlap or exceed cursor')
        if any(x['policy_version']!=state['version'] for x in state['pending']):raise ValueError('Restored pending policy differs')
        state=copy.deepcopy(state)
        for g,s in zip(self.rng,state['worker_rng']):g.set_state(s)
        self.version=state['version'];self.next_series_id=state['next_series_id']
        self.pending=state['pending'];self.excluded=state['excluded'];self.durable=state['durable']
