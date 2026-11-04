"""Read-only server episode evidence; never substitutes for local release gates.

Kaggle SDK 2.2.3's live agent states can be UNSPECIFIED even after completion.
Use the actual replay's BO3 configuration, identity, outcomes and statuses.
"""
from collections import Counter
from datetime import datetime, timezone
import math


PUBLIC='EPISODE_TYPE_PUBLIC'
VALIDATION='EPISODE_TYPE_VALIDATION'
AGENT_ERROR_PREFIX='EPISODE_AGENT_STATE_ERROR_'
BAD_STATUSES={'ERROR','INVALID','TIMEOUT'}


def utc(value):
    parsed=datetime.fromisoformat(value.replace('Z','+00:00'))
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def replay_checks(episode,replay):
    if replay is None:return {'replay_available':False}
    config=replay.get('configuration',{});steps=replay.get('steps',[])
    last=steps[-1] if steps else []
    agents=sorted(episode.get('agents',[]),key=lambda a:a['index'])
    terminal=[]
    if steps and steps[0]:
        for frame in steps[0][0].get('visualize',[]):
            result=frame.get('current',{}).get('result',-1)
            if result in (0,1,2):terminal.append(result)
    wins=Counter(terminal);n=len(terminal)
    try:
        utc(episode['end_time']);valid_end_time=True
    except (KeyError,ValueError,TypeError,AttributeError):valid_end_time=False
    expected=[1,-1] if wins[0]>wins[1] else [-1,1] if wins[1]>wins[0] else [0,0]
    series_complete=(n==3 or (n==2 and abs(wins[0]-wins[1])>3-n))
    return {
        'replay_available':True,
        'episode_identity':replay.get('info',{}).get('EpisodeId')==episode['id'],
        'official_cabt_v2_bo3':replay.get('name')=='cabt' and config.get('version')==2 and config.get('bo')==3,
        'metadata_complete':episode.get('state')=='COMPLETED' and valid_end_time,
        'two_agents_identified':len(agents)==2 and [a['index'] for a in agents]==[0,1],
        'no_metadata_agent_error':all(not a.get('state','').startswith(AGENT_ERROR_PREFIX) for a in agents),
        'terminal_done':replay.get('statuses')==['DONE','DONE'] and len(last)==2 and [a.get('status') for a in last]==['DONE','DONE'],
        'no_replay_faults':all(a.get('status') not in BAD_STATUSES for step in steps for a in step),
        'complete_bo3_outcomes':series_complete,
        'reward_consistency':replay.get('rewards')==expected and [a.get('reward') for a in last]==expected and [a.get('reward') for a in agents]==expected,
    }


def assess_feedback(submission_id,episodes,replays,score_snapshot,minimum_normal=20):
    """Assess normalized API records plus replay bodies keyed by episode ID.

    Callers must separately bind this submission ID to their authenticated own
    upload receipt and release hash. No state is changed by this function.
    """
    if type(submission_id) is not int or submission_id<=0:raise ValueError('Invalid submission identity')
    if minimum_normal<20:raise ValueError('Cannot reduce the required normal BO3 count')
    unique={};duplicates=0;conflicts=[]
    for episode in episodes:
        eid=episode['id']
        if eid in unique:
            duplicates+=1
            if unique[eid]!=episode:conflicts.append(eid)
        else:unique[eid]=episode
    normal=[];validation=[];details=[];explicit_errors=[];latest=None
    for eid,episode in sorted(unique.items()):
        agents=episode.get('agents',[])
        own=[a for a in agents if a.get('submission_id')==submission_id]
        if not own:continue
        ids=[a.get('submission_id') for a in agents];teams=[a.get('team_id') for a in agents]
        normal_identity=(len(own)==1 and len(agents)==2 and len(set(ids))==2 and
                         all(type(t) is int and t>0 for t in teams) and len(set(teams))==2)
        self_copy=len(agents)==2 and len(own)==2
        role='normal' if episode.get('type')==PUBLIC and normal_identity else (
            'validation' if episode.get('type')==VALIDATION and self_copy else 'excluded')
        checks=replay_checks(episode,replays.get(eid))
        verified=all(checks.values())
        if role=='normal' and verified:normal.append(eid)
        if role=='validation' and verified:validation.append(eid)
        bad=episode.get('state')=='ERRORED' or any(a.get('state','').startswith(AGENT_ERROR_PREFIX) for a in agents)
        replay=replays.get(eid)
        if replay:bad=bad or any(a.get('status') in BAD_STATUSES for step in replay.get('steps',[]) for a in step)
        if bad:explicit_errors.append(eid)
        if role!='excluded':
            if verified:
                timestamp=utc(episode['end_time']);latest=max(latest,timestamp) if latest else timestamp
        details.append({'episode_id':eid,'role':role,'verified':verified,'checks':checks})
    score_ok=False
    if score_snapshot:
        try:
            score_ok=(score_snapshot.get('submission_id')==submission_id and bool(score_snapshot.get('source'))
                      and math.isfinite(float(score_snapshot['score'])) and latest is not None
                      and utc(score_snapshot['captured_at'])>=latest)
        except (KeyError,ValueError,TypeError):pass
    checks={'server_self_validation':bool(validation),'normal_nonself_bo3_minimum':len(normal)>=minimum_normal,
            'no_observed_server_faults':not explicit_errors,'no_conflicting_duplicate_episodes':not conflicts,
            'current_submission_score_snapshot':score_ok}
    return {'submission_id':submission_id,'checks':checks,'server_feedback_verified':all(checks.values()),
            'normal_verified_count':len(normal),'normal_episode_ids':normal,'validation_episode_ids':validation,
            'normal_episodes_observed':sum(d['role']=='normal' for d in details),
            'explicit_failed_episode_ids':explicit_errors,'duplicate_records':duplicates,'conflicting_episode_ids':conflicts,
            'episodes_observed':len(unique),'episode_details':details,'score_snapshot':score_snapshot,
            'requires_authenticated_own_upload_binding':True,'local_release_gates_checked':False,'goal_complete':False,
            'scope':'Server feedback only. Includes failure/unknown/excluded records; not a model-selection data source or standalone full-goal completion proof.'}
