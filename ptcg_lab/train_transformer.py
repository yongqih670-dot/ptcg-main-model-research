"""Bounded GPU BC, complete-BO3 split, conditional masks, resumable optimizer/RNG."""
import argparse
import json
import os
import random
import time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import torch
from .io import read_json,atomic_json,sha256,utcnow
from .token_features import FEATURE_VERSION
from .transformer_model import build_model


def atomic_torch(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('wb') as f:torch.save(value,f);f.flush();os.fsync(f.fileno())
    temp.replace(path)


def load_dataset(source,manifest,limit_shards=None):
    entries=manifest['shards'];owners=defaultdict(set);items=[];priority={'train':0,'validation':1,'test':2}
    groups={s:set() for s in priority}
    # Read fingerprints first. Held-out features/labels are never loaded into
    # training tensors; only identity hashes participate in repeat ownership.
    for entry in entries:
        path=source/'dataset'/entry['shard']
        if sha256(path)!=entry['shard_sha256']:raise ValueError('Dataset shard changed')
        groups[entry['split']].add(entry['group_id'])
        with np.load(path,allow_pickle=False) as z:
            fps=z['fingerprints'];ids=z['decision_ids']
        for fp in fps:owners[str(fp)].add(entry['split'])
        items.append((entry,fps,ids))
    if any(groups[a]&groups[b] for a,b in [('train','validation'),('train','test'),('validation','test')]):raise ValueError('BO3 split overlap')
    owner={fp:max(s,key=priority.get) for fp,s in owners.items()};seen={s:set() for s in priority}
    parts={s:defaultdict(list) for s in ('train','validation')};removed=Counter();loaded=Counter();decision_serial=Counter()
    for entry,fps,ids in items:
        split=entry['split']
        if split=='test' or limit_shards and loaded[split]>=limit_shards:continue
        keep=np.zeros(len(fps),bool);serial=np.zeros(len(fps),np.int64)
        for did in np.unique(ids):
            positions=np.where(ids==did)[0];fp=str(fps[positions[0]])
            if owner[fp]!=split or fp in seen[split]:removed[split]+=1;continue
            seen[split].add(fp);keep[positions]=True;serial[positions]=decision_serial[split];decision_serial[split]+=1
        if not keep.any():continue
        with np.load(source/'dataset'/entry['shard'],allow_pickle=False) as z:
            for key in ('X','y','contexts','multi','weights'):parts[split][key].append(z[key][keep])
        parts[split]['decision_ids'].append(serial[keep]);loaded[split]+=1
    data={s:{k:np.concatenate(v) for k,v in p.items()} for s,p in parts.items()}
    if any(not d or len(d['y'])==0 for d in data.values()):raise ValueError('Empty training/validation')
    return data,{'groups_per_split':{s:len(g) for s,g in groups.items()},'loaded_shards':dict(loaded),
        'removed_exact_repeated_decisions':dict(removed),'decisions':dict(decision_serial),'test_labels_scored':False,
        'dedup':'complete decisions; test > validation > train by visible-observation hash; first retained within split'}


@torch.no_grad()
def score(model,data,device,batch_size):
    model.eval();by=defaultdict(Counter);total=Counter();decisions={};nll_sum=0.;weight_sum=0.
    for start in range(0,len(data['y']),batch_size):
        stop=start+batch_size;x=torch.from_numpy(data['X'][start:stop]).to(device);y=torch.from_numpy(data['y'][start:stop]).to(device)
        logits=model.forward_policy(x)
        if not torch.isfinite(logits).all():raise ValueError('Nonfinite validation logits')
        pred=logits.argmax(-1);correct=(pred==y).cpu().numpy();nll=torch.nn.functional.cross_entropy(logits,y,reduction='none').cpu().numpy()
        weights=data['weights'][start:stop];nll_sum+=float((nll*weights).sum());weight_sum+=float(weights.sum())
        for offset,ok in enumerate(correct):
            i=start+offset;context=str(int(data['contexts'][i]));kind='multi_or_optional' if data['multi'][i] else 'mandatory_single'
            for c in (total,by['context_'+context],by[kind]):c['steps']+=1;c['correct']+=int(ok)
            did=int(data['decision_ids'][i]);decisions[did]=decisions.get(did,True) and bool(ok)
    def output(c):return {**c,'accuracy':c['correct']/c['steps']}
    return {**output(total),'decision_weighted_nll':nll_sum/weight_sum,'teacher_forced_exact_decisions':sum(decisions.values()),
        'decisions':len(decisions),'teacher_forced_exact_accuracy':sum(decisions.values())/len(decisions),'by_context_and_kind':{k:output(v) for k,v in by.items()},
        'scope':'teacher-forced conditional accuracy; free-running legality and strength require official BO3'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();cfg=read_json(a.config);out=a.output;policy=read_json(out/'policy.json')
    if not torch.cuda.is_available():raise RuntimeError('GPU job cannot silently run on CPU')
    torch.set_num_threads(cfg['workers']);torch.manual_seed(cfg['training_seed']);np.random.seed(cfg['training_seed']%2**32);random.seed(cfg['training_seed'])
    torch.cuda.manual_seed_all(cfg['training_seed']);device=torch.device('cuda')
    cap=policy['budget']['vram_working_target_gib']*2**30;total=torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(cap/total,1.),0)
    source=Path(cfg['dataset_run'])
    if sha256(source/'dataset-manifest.json')!=cfg['dataset_manifest_sha256']:raise ValueError('Dataset identity changed')
    manifest=read_json(source/'dataset-manifest.json')
    if not manifest['complete'] or manifest['feature_version']!=FEATURE_VERSION:raise ValueError('Incomplete/incompatible dataset')
    data,audit=load_dataset(source,manifest,cfg.get('smoke_shards_per_split'));atomic_json(out/'training-data-audit.json',audit)
    model,initialization=build_model(cfg['feature_bundle'],cfg['architecture'],cfg['parent_model_sha256']);model.to(device)
    atomic_json(out/'initialization.json',initialization)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    rng=np.random.default_rng(cfg['training_seed']);train=data['train'];batch=cfg['batch_size'];epoch=0;cursor=0;steps=0;permutation=None;curves=[];best=float('inf')
    identity={'config_sha256':sha256(a.config),'dataset_sha256':cfg['dataset_manifest_sha256'],'source_sha256':sha256(out/'source-manifest.json')}
    checkpoint=out/'checkpoint.pt'
    if checkpoint.exists():
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if saved['identity']!=identity:raise ValueError('Training checkpoint identity mismatch')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);epoch=saved['epoch'];cursor=saved['cursor'];steps=saved['steps']
        permutation=saved['permutation'].numpy() if saved['permutation'] is not None else None
        curves=saved['curves'];best=saved['best'];rng.bit_generator.state=saved['numpy_rng'];random.setstate(saved['python_rng'])
        torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
    last_save=time.monotonic();last_progress=0;start=time.monotonic()
    def save():
        atomic_torch(checkpoint,{'identity':identity,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'cursor':cursor,'steps':steps,
            'permutation':None if permutation is None else torch.from_numpy(permutation),'curves':curves,'best':best,'numpy_rng':rng.bit_generator.state,
            'python_rng':random.getstate(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()})
    while epoch<cfg['epochs']:
        if permutation is None:permutation=rng.permutation(len(train['y']));cursor=0
        model.train()
        while cursor<len(permutation):
            idx=permutation[cursor:cursor+batch];x=torch.from_numpy(train['X'][idx]).to(device);y=torch.from_numpy(train['y'][idx]).to(device)
            weights=torch.from_numpy(train['weights'][idx]).to(device)
            optimizer.zero_grad(set_to_none=True);logits=model.forward_policy(x)
            loss=(torch.nn.functional.cross_entropy(logits,y,reduction='none')*weights).sum()/weights.sum()
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm):raise ValueError('Nonfinite gradient')
            optimizer.step();cursor+=len(idx);steps+=1
            if torch.cuda.max_memory_reserved()>cap:raise RuntimeError('VRAM target exceeded')
            now=time.monotonic()
            if now-last_progress>=10:
                atomic_json(out/'progress.json',{'at':utcnow(),'epoch':epoch+1,'cursor':cursor,'total_steps_in_epoch':len(permutation),
                    'optimizer_steps':steps,'loss':float(loss.detach()),'peak_vram_gib':torch.cuda.max_memory_reserved()/2**30});last_progress=now
            if now-last_save>=30:save();last_save=now
            if cfg.get('checkpoint_after_steps')==steps:
                save();atomic_json(out/'metrics.json',{'complete':False,'checkpointed_steps':steps});return
        validation=score(model,data['validation'],device,batch)
        curves.append({'epoch':epoch+1,'optimizer_steps':steps,'validation':validation});atomic_json(out/'training-curves.json',curves)
        if validation['decision_weighted_nll']<best:
            best=validation['decision_weighted_nll'];atomic_torch(out/'model.pt',{k:v.detach().cpu() for k,v in model.state_dict().items()})
        epoch+=1;cursor=0;permutation=None;save();last_save=time.monotonic()
    model.load_state_dict(torch.load(out/'model.pt',map_location='cpu',weights_only=True))
    metrics={s:score(model,d,device,batch) for s,d in data.items()}
    # Same saved tensors, independent CPU forward, no test-set access.
    x=torch.from_numpy(data['validation']['X'][:16]);model.eval()
    with torch.no_grad():gpu_logits=model.forward_policy(x.to(device)).cpu()
    cpu_model,_=build_model(cfg['feature_bundle'],cfg['architecture']);cpu_model.load_state_dict(torch.load(out/'model.pt',map_location='cpu',weights_only=True));cpu_model.eval();cpu_model.freeze_tables()
    with torch.no_grad():cpu_logits=cpu_model.forward_policy(x)
    delta=float((gpu_logits-cpu_logits).abs().max());same=bool(torch.equal(gpu_logits.argmax(-1),cpu_logits.argmax(-1)))
    export={'samples':len(x),'maximum_logit_absolute_difference':delta,'identical_argmax':same,'passed':same and delta<1e-3}
    atomic_json(out/'cpu-export-check.json',export)
    if not export['passed']:raise ValueError('CPU export equivalence failed')
    report={'complete':True,'feature_version':FEATURE_VERSION,'training_seed':cfg['training_seed'],'epochs':epoch,'optimizer_steps':steps,
        'model_sha256':sha256(out/'model.pt'),'data_audit':audit,'initialization':initialization,'offline_metrics':metrics,'test_labels_scored':False,
        'best_validation_nll':best,'peak_vram_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'peak_vram_reserved_gib':torch.cuda.max_memory_reserved()/2**30,
        'cpu_export_check':export,'scope':'Imitation only; not BO3 promotion evidence'}
    atomic_json(out/'metrics.json',report);print(json.dumps({k:v for k,v in report.items() if k not in ('offline_metrics','initialization')}))


if __name__=='__main__':main()
