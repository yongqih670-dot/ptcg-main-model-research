"""Actor-only replay extraction. Full-series grouping, no visualization reads."""
import copy
import hashlib
import json
from collections import Counter
from .agents import validate_action

ACTOR_KEYS=('current','select','logs')


def group_split(group_id):
    bucket=int(hashlib.sha256(group_id.encode()).hexdigest()[:8],16)%100
    return 'train' if bucket<80 else 'validation' if bucket<90 else 'test'


def actor_observation(observation):
    # No opaque search serialization, visualization, rewards, future data,
    # opponent submitted deck or timing features. These exact allowed fields
    # are supplied to the actor by the official engine at decision time.
    return {k:copy.deepcopy(observation.get(k)) for k in ACTOR_KEYS}


def extract_episode(episode,group_id):
    if episode.get('name')!='cabt' or episode.get('configuration',{}).get('version')!=2:
        raise ValueError('Only verified CABT v2 replay schema is supported')
    if episode['configuration'].get('bo')!=3 or episode['statuses']!=['DONE','DONE']:
        raise ValueError('Require a normally completed full BO3')
    steps=episode['steps'];records=[];decks={};contexts=Counter();invalid=[]
    for step_id,step in enumerate(steps[:-1]):
        for seat,item in enumerate(step):
            if item['status']!='ACTIVE':continue
            obs=item['observation'];action=steps[step_id+1][seat]['action']
            if obs.get('select') is None:
                validate_action(obs,action)
                if seat in decks and decks[seat]!=action:raise ValueError('Deck changed inside series')
                decks[seat]=action
                continue
            if obs['current']['yourIndex']!=seat:raise ValueError('Actor seat mismatch')
            try:validate_action(obs,action)
            except ValueError as exc:
                invalid.append({'step':step_id,'seat':seat,'error':str(exc)});continue
            if seat not in decks:raise ValueError('Own submitted deck missing')
            visible=actor_observation(obs)
            fingerprint=hashlib.sha256(json.dumps(visible,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            context=obs['select']['context'];contexts[str(context)]+=1
            records.append({'group_id':group_id,'decision_id':f'{group_id}:{step_id}:{seat}',
                'split':group_split(group_id),'seat':seat,'round':obs['current']['round'],
                'own_deck':decks[seat],'observation':visible,'action':list(action),
                'observation_sha256':fingerprint,'context':context})
    if set(decks)!={0,1}:raise ValueError('Both submitted decks must be available')
    if invalid:raise ValueError(f'Action/observation alignment or schema failures: {invalid[:3]}')
    return records,{'decision_count':len(records),'contexts':dict(contexts),'group_id':group_id,
                    'split':group_split(group_id),'own_decks':decks,
                    'rounds':sorted({r['round'] for r in records}),'invalid_actions':len(invalid)}
