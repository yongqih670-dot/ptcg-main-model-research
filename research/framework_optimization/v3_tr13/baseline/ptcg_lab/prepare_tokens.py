"""Full BO3 teacher extraction with checkpointed conditional-token shards."""
import argparse
import gzip
import json
import os
from collections import Counter
from pathlib import Path
import numpy as np
from .io import read_json,read_series,atomic_json,sha256,utcnow
from .replay_data import extract_episode
from .token_features import TokenFeatures,conditional_targets,forced_selection,FEATURE_VERSION


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stop-after-new-series',type=int);args=p.parse_args();cfg=read_json(args.config);out=args.output
    source=Path(cfg['source_run'])
    for name,expected in cfg['source_hashes'].items():
        if sha256(source/name)!=expected:raise ValueError('Source identity changed: '+name)
    if 'holdout' in json.dumps(read_json(source/'config.json')).lower():raise ValueError('Final holdout cannot supply teachers')
    rows=sorted((r for r in read_series(source/'series.jsonl') if r['cell_id'].startswith(cfg['teacher_cell_prefix'])),key=lambda r:r['series_id'])
    if len(rows)!=cfg['expected_series']:raise ValueError('Teacher count changed')
    features=TokenFeatures(cfg['feature_bundle']);b=features.b
    shards=out/'dataset';shards.mkdir(exist_ok=True);new=0;records=[]
    for row in rows:
        sid=row['series_id'];key=f'{sid:05d}';meta_path=shards/(key+'.json');npz=shards/(key+'.npz')
        if meta_path.exists():
            m=read_json(meta_path)
            if sha256(npz)!=m['shard_sha256'] or sha256(source/m['source_file'])!=m['source_sha256']:raise ValueError('Shard/source checkpoint changed')
            records.append(m);continue
        if args.stop_after_new_series and new>=args.stop_after_new_series:break
        me=row['candidate_seat'];identity=row['identities'][me]
        if row['result']=='error' or row['policy_invalid_count'] or identity['parameters']!=cfg['teacher_parameters']:raise ValueError('Invalid teacher series')
        replay=source/f'episode-{sid:05d}.json.gz'
        with gzip.open(replay,'rt',encoding='utf-8') as f:episode=json.load(f)
        decisions,summary=extract_episode(episode,source.name+'/'+key)
        X=[];y=[];contexts=[];multi=[];decision_ids=[];fingerprints=[];weights=[];stats=Counter();tracker=None;round_id=None;forced=0
        for decision in decisions:
            if decision['seat']!=me:continue
            obs=decision['observation'];sel=obs['select'];n=len(sel['option'])
            if decision['round']!=round_id:
                tracker=features.Tracker(decision['own_deck']);round_id=decision['round'];forced=0
            tracker.update(obs);stats['teacher_decisions']+=1;stats['max_options']=max(stats['max_options'],n)
            if forced_selection(sel):forced+=1;stats['forced']+=1;continue
            if n>b.STOP:raise ValueError('Training shard needs ragged chunk loss; do not truncate or drop teacher')
            steps=conditional_targets(sel,decision['action']);decision_id=stats['learnable_decisions'];stats['learnable_decisions']+=1
            for picked,target in steps:
                inputs,mapping=features.make(obs,tracker,forced,picked);label=b.STOP if target==n else target
                if inputs.shape[0]!=1 or features.field(inputs[0],'opt_mask')[label]!=1:raise ValueError('Teacher label masked')
                X.append(inputs[0]);y.append(label);contexts.append(decision['context']);multi.append(sel['maxCount']>1 or sel['minCount']==0)
                decision_ids.append(decision_id);fingerprints.append(decision['observation_sha256']);weights.append(1/len(steps))
                stats['conditional_steps']+=1;stats['stop_labels']+=target==n
            forced=0
        temp=npz.with_suffix('.npz.tmp')
        with temp.open('wb') as f:
            np.savez_compressed(f,X=np.asarray(X,np.float32).reshape(-1,b.OBS_SIZE),y=np.asarray(y,np.int64),contexts=np.asarray(contexts,np.int16),
                multi=np.asarray(multi,bool),decision_ids=np.asarray(decision_ids,np.int32),fingerprints=np.asarray(fingerprints,dtype='U64'),weights=np.asarray(weights,np.float32))
            f.flush();os.fsync(f.fileno())
        temp.replace(npz)
        m={'series_id':sid,'group_id':summary['group_id'],'split':summary['split'],'teacher_seat':me,'teacher_identity':identity,
           'teacher_result':row['result'],'stats':dict(stats),'source_file':replay.name,'source_sha256':sha256(replay),'shard':npz.name,'shard_sha256':sha256(npz)}
        atomic_json(meta_path,m);records.append(m);new+=1
        atomic_json(out/'progress.json',{'at':utcnow(),'completed_series':len(records),'target_series':len(rows),'last_series_id':sid})
    totals=Counter();splits=Counter()
    for m in records:totals.update({k:v for k,v in m['stats'].items() if k!='max_options'});splits[m['split']]+=1
    manifest={'at':utcnow(),'complete':len(records)==len(rows),'source_run':str(source),'source_hashes':cfg['source_hashes'],'feature_version':FEATURE_VERSION,
        'feature_count':b.OBS_SIZE,'grouping':'complete BO3; candidate teacher; all outcomes; same hash split as H003','split_series':dict(splits),
        'stats':dict(totals),'max_options':max(m['stats']['max_options'] for m in records),'shards':records}
    atomic_json(out/'dataset-manifest.json',manifest);metrics={k:v for k,v in manifest.items() if k!='shards'};atomic_json(out/'metrics.json',metrics);print(json.dumps(metrics))


if __name__=='__main__':main()
