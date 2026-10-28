"""Check an independently preregistered BO3 confirmation against its screen.

This is a development gate, never a final-submission gate. It does not mutate
the champion, release anchor, opponent manifests, or research state.
"""
import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ptcg_lab.io import read_json,read_series,atomic_json
from scripts.analyze_panel import analyze


def diagnostics(rows):
    cells={}
    for name in sorted({r['cell_id'] for r in rows}):
        group=[r for r in rows if r['cell_id']==name]
        seat=defaultdict(Counter); rounds=defaultdict(Counter); first=defaultdict(Counter)
        calls=[]; times=[]
        for r in group:
            me=r['candidate_seat'];seat[str(me)][r['result']]+=1
            calls.append(r['policy_latency_max_seconds'][me])
            times.extend(r['policy_round_seconds'][me].values())
            for g in r.get('games',[]):
                result='win' if g['winner']==me else 'loss' if g['winner']==1-me else 'draw'
                rounds[str(g['round'])][result]+=1
                key='first' if g.get('first_player')==me else 'second' if g.get('first_player')==1-me else 'unknown'
                first[key][result]+=1
        cells[name]={'series_by_seat':dict(seat),'games_by_round':dict(rounds),
            'games_by_turn_order':dict(first),'candidate_max_call_seconds':max(calls),
            'candidate_mean_series_policy_seconds':float(np.mean([
                sum(r['policy_round_seconds'][r['candidate_seat']].values()) for r in group])),
            'candidate_max_round_policy_seconds':max(times)}
    return cells


def validate(run):
    config=read_json(run/'config.json');policy=read_json(run/'policy.json')
    rows=read_series(run/'series.jsonl');panel=analyze(run)
    screen=ROOT/'runs'/config['screening_run_id']
    previous=read_json(screen/'config.json')
    checks={}
    checks['explicit_confirmation']=config.get('evaluation_stage')=='independent_confirmation'
    checks['fresh_job']=run!=screen and read_json(run/'experiment.json')['created_at']>max(
        r['finished_at'] for r in read_series(screen/'series.jsonl'))
    def specifications(c):
        return [{k:v for k,v in x.items() if k!='series_count'} for x in c['matchups']]
    checks['same_strategies_and_weights']=(specifications(config)==specifications(previous)
        and config['contrasts']==previous['contrasts']
        and config.get('original_primary_contrasts')==previous.get('original_primary_contrasts'))
    relevant=lambda m:{k:v for k,v in m.items() if k.replace('\\','/').split('/')[0] in ('ptcg_lab','agents')}
    checks['same_agent_and_runner_code']=(relevant(read_json(run/'source-manifest.json'))==
                                         relevant(read_json(screen/'source-manifest.json')))
    checks['same_engine_and_dependencies']=all(read_json(run/'environment.json')[k]==
        read_json(screen/'environment.json')[k] for k in ('files','packages'))
    counts=Counter(r['cell_id'] for r in rows)
    checks['fixed_sample_complete']=panel['complete'] and all(
        counts[x['cell_id']]==x['series_count']>=policy['evaluation']['confirmation_series_per_opponent']
        for x in config['matchups'])
    checks['unique_series_ids']=len(rows)==len({r['series_id'] for r in rows})
    checks['balanced_seats']=all(abs(sum(r['candidate_seat']==0 for r in rows if r['cell_id']==x)-
        sum(r['candidate_seat']==1 for r in rows if r['cell_id']==x))<=1 for x in counts)
    checks['reliability']=all(r['result']!='error' and r['statuses']==['DONE','DONE'] and
        r['policy_invalid_count']==0 and r.get('internal_random_fallback_count',0)==0 for r in rows)
    checks['gain_ci_pass']=panel.get('contrast',{}).get('ci95',[-1])[0]>policy['evaluation']['promotion_delta_ci_lower_must_exceed']
    if config.get('original_primary_contrasts'):
        checks['original_primary_gain_ci_pass']=panel.get('original_primary_contrast',{}).get('ci95',[-1])[0]>policy['evaluation']['promotion_delta_ci_lower_must_exceed']
        checks['minimum_distinct_opponents']=len({m['reference']['id'] for m in config['matchups']})>=policy['evaluation']['opponents_min']
    detail=diagnostics(rows)
    candidate_time=sum(c['weight']*detail[c['candidate_cell']]['candidate_mean_series_policy_seconds'] for c in config['contrasts'])
    anchor_time=sum(c['weight']*detail[c['anchor_cell']]['candidate_mean_series_policy_seconds'] for c in config['contrasts'])
    slower=candidate_time>anchor_time
    checks['slower_practical_gain_pass']=not slower or panel['contrast']['gain']>=policy['evaluation']['slower_candidate_min_gain']
    report={'run_id':run.name,'screening_run_id':screen.name,'checks':checks,
            'development_confirmation_passed':all(checks.values()),'classification':panel.get('classification','inconclusive'),
            'contrast':panel.get('contrast'),'diagnostics':detail,
            'candidate_seconds_per_series':candidate_time,'anchor_seconds_per_series':anchor_time,
            'candidate_slower':slower,'submission_qualified':False,
            'scope':'Independent development confirmation only; strongest baseline selection and all final gates remain separate'}
    atomic_json(run/'confirmation.json',report)
    return report


if __name__=='__main__':
    import json
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);args=p.parse_args()
    r=validate(args.run.resolve());print(json.dumps({k:v for k,v in r.items() if k!='diagnostics'}))
