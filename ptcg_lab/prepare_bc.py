"""Checkpointed, same-engine teacher extraction; execute via run_experiment."""
import argparse
import gzip
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
import numpy as np
from .io import read_json,read_series,atomic_json,sha256,utcnow
from .replay_data import extract_episode,group_split
from .bc_features import CandidateFeatures,supported,FEATURE_VERSION,FEATURE_COUNT


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stop-after-new-series',type=int);args=p.parse_args();cfg=read_json(args.config);out=args.output
    source=Path(cfg['source_run'])
    for name,expected in cfg['source_hashes'].items():
        if sha256(source/name)!=expected:raise ValueError('Source run identity changed: '+name)
    source_cfg=read_json(source/'config.json')
    if 'holdout' in json.dumps(source_cfg).lower():raise ValueError('Final holdout must never supply teachers')
    rows=sorted((r for r in read_series(source/'series.jsonl') if r['cell_id'].startswith(cfg['teacher_cell_prefix'])),key=lambda r:r['series_id'])
    if len(rows)!=cfg['expected_series']:raise ValueError('Teacher series count changed')
    bundle=Path(cfg['feature_bundle']);features=CandidateFeatures(bundle)
    shards=out/'dataset';shards.mkdir(exist_ok=True);new=0
    records=[]
    for row in rows:
        sid=row['series_id'];key=f'{sid:05d}';meta_path=shards/(key+'.json');npz=shards/(key+'.npz')
        if meta_path.exists():
            m=read_json(meta_path)
            if not npz.exists() or sha256(npz)!=m['shard_sha256']:raise ValueError('Dataset checkpoint changed')
            if sha256(source/m['source_file'])!=m['source_sha256']:raise ValueError('Source replay changed after checkpoint')
            records.append(m);continue
        if args.stop_after_new_series and new>=args.stop_after_new_series:break
        if row['result']=='error' or row['policy_invalid_count']:raise ValueError('Invalid teacher series is not silently discarded')
        me=row['candidate_seat'];identity=row['identities'][me]
        if identity['parameters']!=cfg['teacher_parameters']:raise ValueError('Unexpected teacher parameter identity')
        replay=source/f'episode-{sid:05d}.json.gz'
        with gzip.open(replay,'rt',encoding='utf-8') as f:episode=json.load(f)
        decisions,summary=extract_episode(episode,source.name+'/'+key)
        chosen=[r for r in decisions if r['seat']==me]
        matrices=[];labels=[];offsets=[0];contexts=[];fingerprints=[];stats=Counter();tracker=None;round_id=None;forced=0
        for decision in chosen:
            obs=decision['observation'];sel=obs['select'];n=len(sel['option'])
            if decision['round']!=round_id:
                tracker=features.Tracker(decision['own_deck']);round_id=decision['round'];forced=0
            tracker.update(obs);stats['decisions']+=1;stats['options_over_63']+=n>63
            stats['max_options']=max(stats['max_options'],n)
            is_forced=(sel['minCount']==sel['maxCount']==n or n==1 and sel['minCount']>=1)
            if is_forced:forced+=1;stats['forced']+=1;continue
            if supported(sel):
                matrix=features.make(obs,tracker,forced)
                target=np.zeros(n,dtype=np.int8);target[decision['action'][0]]=1
                matrices.append(matrix);labels.append(target);offsets.append(offsets[-1]+n)
                contexts.append(decision['context']);fingerprints.append(decision['observation_sha256']);stats['learnable']+=1
            else:stats['unsupported_nonforced']+=1
            forced=0
        X=np.concatenate(matrices) if matrices else np.empty((0,FEATURE_COUNT),dtype=np.float32)
        y=np.concatenate(labels) if labels else np.empty(0,dtype=np.int8)
        temporary=npz.with_suffix('.npz.tmp')
        with temporary.open('wb') as f:
            np.savez_compressed(f,X=X,y=y,offsets=np.array(offsets,dtype=np.int64),contexts=np.array(contexts,dtype=np.int16),fingerprints=np.array(fingerprints,dtype='U64'))
            f.flush();os.fsync(f.fileno())
        temporary.replace(npz)
        m={'series_id':sid,'group_id':summary['group_id'],'split':summary['split'],'teacher_seat':me,
            'teacher_identity':identity,'teacher_result':row['result'],'stats':dict(stats),
            'source_file':replay.name,'source_sha256':sha256(replay),'shard':npz.name,'shard_sha256':sha256(npz),
            'candidate_rows':len(y),'feature_version':FEATURE_VERSION}
        atomic_json(meta_path,m);records.append(m);new+=1
        atomic_json(out/'progress.json',{'at':utcnow(),'completed_series':len(records),'target_series':len(rows),'last_series_id':sid})
    totals=Counter();split_series=Counter();split_decisions=Counter();max_options=0
    for m in records:
        totals.update({k:v for k,v in m['stats'].items() if k!='max_options'});max_options=max(max_options,m['stats']['max_options'])
        split_series[m['split']]+=1;split_decisions[m['split']]+=m['stats']['learnable']
    complete=len(records)==len(rows)
    manifest={'at':utcnow(),'complete':complete,'source_run':str(source),'source_hashes':cfg['source_hashes'],
        'feature_version':FEATURE_VERSION,'feature_count':FEATURE_COUNT,'grouping':'complete BO3; all rounds together; candidate teacher only; no outcome filtering',
        'split_series':dict(split_series),'split_decisions':dict(split_decisions),'stats':dict(totals),'max_options':max_options,'shards':records}
    atomic_json(out/'dataset-manifest.json',manifest)
    metrics={k:v for k,v in manifest.items() if k!='shards'};atomic_json(out/'metrics.json',metrics);print(json.dumps(metrics))


if __name__=='__main__':main()
