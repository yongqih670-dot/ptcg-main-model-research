"""CPU candidate-scoring BC with full-BO3 splits and native CatBoost snapshot."""
import argparse
import json
import time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from .io import read_json,atomic_json,sha256,utcnow
from .bc_features import FEATURE_VERSION,FEATURE_COUNT


def load_dataset(folder,manifest):
    all_fingerprints=defaultdict(set);items=[]
    for entry in manifest['shards']:
        path=folder/'dataset'/entry['shard']
        if sha256(path)!=entry['shard_sha256']:raise ValueError('Training shard identity changed')
        with np.load(path,allow_pickle=False) as z:data={k:z[k] for k in z.files}
        for fp in data['fingerprints']:all_fingerprints[str(fp)].add(entry['split'])
        items.append((entry,data))
    # Exclude exact visible-observation repeats across splits from the lower
    # priority split. This uses fingerprints, never held-out labels/metrics.
    priority={'train':0,'validation':1,'test':2}
    owner={fp:max(splits,key=priority.get) for fp,splits in all_fingerprints.items()}
    parts={s:{'X':[],'y':[],'weight':[],'offsets':[0],'contexts':[],'groups':[]} for s in priority}
    excluded=Counter();seen={s:set() for s in priority}
    for entry,d in items:
        s=entry['split'];part=parts[s]
        for i,fp in enumerate(d['fingerprints']):
            fp=str(fp)
            if owner[fp]!=s or fp in seen[s]:excluded[s]+=1;continue
            seen[s].add(fp);a,b=d['offsets'][i:i+2];n=int(b-a)
            if n<2 or d['y'][a:b].sum()!=1:raise ValueError('Malformed decision group')
            part['X'].append(d['X'][a:b]);part['y'].append(d['y'][a:b]);part['weight'].append(np.full(n,1/n,np.float32))
            part['offsets'].append(part['offsets'][-1]+n);part['contexts'].append(int(d['contexts'][i]));part['groups'].append(entry['group_id'])
    data={}
    for s,part in parts.items():
        if not part['X']:raise ValueError('Empty grouped data split: '+s)
        data[s]={'X':np.concatenate(part['X']),'y':np.concatenate(part['y']),'weight':np.concatenate(part['weight']),
            'offsets':np.asarray(part['offsets']),'contexts':np.asarray(part['contexts']),'groups':part['groups']}
    split_groups={s:set(x['groups']) for s,x in data.items()}
    if any(split_groups[a]&split_groups[b] for a,b in [('train','validation'),('train','test'),('validation','test')]):
        raise ValueError('Full BO3 crossed data splits')
    return data,{'removed_exact_repeated_decisions':dict(excluded),'cross_split_unique_visible_observations':sum(len(v)>1 for v in all_fingerprints.values()),
        'method':'test > validation > train ownership by observation hash; within split retain first exact state only',
        'groups_per_split':{s:len(g) for s,g in split_groups.items()}}


def score_decisions(model,data):
    logits=model.predict(data['X'],prediction_type='RawFormulaVal',thread_count=1)
    by=defaultdict(lambda:Counter());total=Counter();nll=[]
    for i,(a,b) in enumerate(zip(data['offsets'][:-1],data['offsets'][1:])):
        label=int(np.argmax(data['y'][a:b]));pred=int(np.argmax(logits[a:b]));correct=int(pred==label)
        for group in (total,by[str(data['contexts'][i])]):group['decisions']+=1;group['correct']+=correct
        scores=logits[a:b];stable=scores-scores.max();nll.append(float(np.log(np.exp(stable).sum())-stable[label]))
    def result(c):return {**c,'accuracy':c['correct']/c['decisions']}
    return {**result(total),'choice_softmax_nll':float(np.mean(nll)),'by_context':{k:result(v) for k,v in by.items()}}


class Progress:
    def __init__(self,out):self.out=out;self.last=0
    def after_iteration(self,info):
        now=time.monotonic()
        if now-self.last>=5:
            self.last=now
            atomic_json(self.out/'progress.json',{'at':utcnow(),'training_iteration':info.iteration,
                'latest_metrics':{group:{k:float(v[-1]) for k,v in values.items()} for group,values in info.metrics.items()}})
        return True


def main():
    from catboost import CatBoostClassifier,Pool
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();cfg=read_json(a.config);out=a.output;source=Path(cfg['dataset_run'])
    if sha256(source/'dataset-manifest.json')!=cfg['dataset_manifest_sha256']:raise ValueError('Dataset manifest changed')
    manifest=read_json(source/'dataset-manifest.json')
    if not manifest['complete'] or manifest['feature_version']!=FEATURE_VERSION:raise ValueError('Dataset incomplete or feature mismatch')
    data,audit=load_dataset(source,manifest);atomic_json(out/'training-data-audit.json',audit)
    if cfg.get('require_all_splits',True) and any(len(d['groups'])<10 for d in data.values()):raise ValueError('Insufficient validation decisions')
    model=CatBoostClassifier(iterations=cfg['iterations'],depth=cfg['depth'],learning_rate=cfg['learning_rate'],
        loss_function='Logloss',eval_metric='Logloss',random_seed=cfg['training_seed'],thread_count=cfg['workers'],
        task_type='CPU',train_dir=str(out/'catboost_info'),allow_writing_files=True)
    pool=lambda d:Pool(d['X'],d['y'],weight=d['weight'])
    model.fit(pool(data['train']),eval_set=pool(data['validation']),early_stopping_rounds=cfg['early_stopping_rounds'],
        use_best_model=True,verbose=50,save_snapshot=True,snapshot_file=str(out/'checkpoint.cbsnapshot'),snapshot_interval=30,
        callbacks=[Progress(out)])
    target=out/'model.cbm';temporary=out/'model.cbm.tmp';model.save_model(str(temporary));temporary.replace(target)
    report={'complete':True,'feature_version':FEATURE_VERSION,'feature_count':FEATURE_COUNT,'training_seed':cfg['training_seed'],
        'dataset_manifest_sha256':cfg['dataset_manifest_sha256'],'model_sha256':sha256(target),'trees':model.tree_count_,
        'best_iteration':model.get_best_iteration(),'data_audit':audit,
        'offline_metrics':{s:score_decisions(model,d) for s,d in data.items() if s!='test' or cfg.get('evaluate_test',False)},
        'test_labels_scored':cfg.get('evaluate_test',False),
        'candidate_rows':{s:len(d['y']) for s,d in data.items()},'scope':'Imitation only; no BO3 strength or promotion claim'}
    atomic_json(out/'training-curves.json',model.get_evals_result());atomic_json(out/'metrics.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='offline_metrics'}))


if __name__=='__main__':main()
