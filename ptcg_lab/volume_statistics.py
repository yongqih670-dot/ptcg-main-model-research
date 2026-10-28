"""Frozen unpaired complete-BO3 inference; no model selection or sample extension."""
import numpy as np

def bootstrap_cells(groups,counts,seed):
    if set(groups)!=set(counts):raise ValueError('Frozen evaluation cells differ')
    rng=np.random.default_rng(seed);means={};samples={}
    for key in counts:
        x=np.asarray(groups[key],dtype=float);n=counts[key]
        if x.shape!=(n,) or not np.isin(x,[0.,.5,1.]).all():raise ValueError('Fixed complete-BO3 scores required')
        means[key]=float(x.mean());draws=[]
        for start in range(0,20000,250):draws.append(rng.choice(x,(min(250,20000-start),n)).mean(axis=1))
        samples[key]=np.concatenate(draws)
    return means,samples

def primary(groups):
    names={'TR11-vs-ppo':1600,'endpoint-vs-ppo':1600}
    means,samples=bootstrap_cells(groups,names,2026102801)
    gain=float((sum(groups['endpoint-vs-ppo'])-sum(groups['TR11-vs-ppo']))/1600)
    ci=np.quantile(samples['endpoint-vs-ppo']-samples['TR11-vs-ppo'],[.025,.975]).tolist()
    return {'gain':gain,'ci95':ci,'scores':means,'learning_signal_passed':bool(gain>=.05 and ci[0]>0),
        'bootstrap_samples':20000,'bootstrap_seed':2026102801,'method':'independent completeBO3 resampling'}

def secondary(groups,opponents):
    counts={f'{role}-vs-{o}':100 for o in opponents for role in ['endpoint','TR11','anchor']}
    means,samples=bootstrap_cells(groups,counts,2026102802);result={}
    for panel,selected in [('full_eight',opponents),('original_four',opponents[:4])]:
        weights=np.full(len(selected),1/len(selected));contrasts={}
        for left,right in [('endpoint','TR11'),('endpoint','anchor'),('TR11','anchor')]:
            gain=sum(w*(means[f'{left}-vs-{o}']-means[f'{right}-vs-{o}']) for w,o in zip(weights,selected))
            draws=sum(w*(samples[f'{left}-vs-{o}']-samples[f'{right}-vs-{o}']) for w,o in zip(weights,selected))
            contrasts[f'{left}-minus-{right}']={'gain':float(gain),'ci95':np.quantile(draws,[.025,.975]).tolist(),
                'descriptive_only_unadjusted_multiple_contrasts':True}
        result[panel]={'weights':dict(zip(selected,weights.tolist())),'contrasts':contrasts,
            'scores':{role:sum(w*means[f'{role}-vs-{o}'] for w,o in zip(weights,selected)) for role in ['endpoint','TR11','anchor']}}
    return {'panels':result,'cells':means,'bootstrap_samples':20000,'bootstrap_seed':2026102802,
        'method':'independent completeBO3 resampling within each frozen opponent and strategy cell',
        'no_automatic_promotion':True}
