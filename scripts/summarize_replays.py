"""Post-match diagnostic summaries. Never a training-data exporter.

Official visualizations contain hidden truth. This report selects only public
attacks, HP, board identities, zone counts and terminal result/reason. Actor
training must instead use the original acting-seat observation and group by BO3.
"""
import argparse
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ptcg_lab.io import read_series,atomic_json


def diagnose(run, maximum=20):
    report=[]
    groups=defaultdict(list)
    for row in read_series(run/'series.jsonl'):
        if row['result'] in ('loss','error'): groups[row.get('cell_id','default')].append(row)
    selected=[]
    while any(groups.values()) and len(selected)<maximum:
        for values in groups.values():
            if values and len(selected)<maximum: selected.append(values.pop(0))
    for row in selected:
        if len(report)>=maximum: break
        path=run/f"episode-{row['series_id']:05d}.json.gz"
        if path.exists():
            with gzip.open(path,'rt',encoding='utf-8') as f: e=json.load(f)
        else:
            path=path.with_suffix('')
            if not path.exists(): continue
            e=json.loads(path.read_text(encoding='utf-8'))
        games=[];attacks=Counter();knockouts=[]
        for step in e['steps'][0][0].get('visualize',[]):
            cur=step['current']
            for log in step.get('logs',[]):
                if log['type']=='Attack':
                    attacks[(log['playerIndex'],log['cardId'],log['attackId'])]+=1
                # Zero HP on public boards is observed before the engine removes a KO.
            for seat, player in enumerate(cur['players']):
                for p in player['active']+player['bench']:
                    if p and p['hp']<=0:
                        key=(seat,p['serial'])
                        if key not in [tuple(k['key']) for k in knockouts]:
                            knockouts.append({'key':list(key),'turn':cur['turn'],'card_id':p['id']})
            if cur['result']>=0:
                reasons=[log.get('reason') for log in step.get('logs',[]) if log['type']=='Result']
                games.append({'round':len(games)+1,'turn':cur['turn'],'winner':cur['result'],
                    'reason':reasons[-1] if reasons else None,
                    'candidate_won':cur['result']==row['candidate_seat'],
                    'remaining_prizes':[len(p['prize']) for p in cur['players']],
                    'deck_counts':[p['deckCount'] for p in cur['players']],
                    'attacks':[{'seat':s,'card_id':c,'attack_id':a,'count':n} for (s,c,a),n in attacks.items()],
                    'public_zero_hp_events':knockouts})
                attacks=Counter();knockouts=[]
        report.append({'series_id':row['series_id'],'cell_id':row.get('cell_id'),
                       'candidate_seat':row['candidate_seat'],'games':games})
    out={'scope':'post-match public-event diagnosis only; not training data',
         'requested_maximum':maximum,'reviewed_series':len(report),'series':report}
    atomic_json(run/'failure-review.json',out)
    print(json.dumps({'reviewed_series':len(report),'path':str(run/'failure-review.json')}))
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--maximum',type=int,default=20)
    a=p.parse_args();diagnose(a.run,a.maximum)
