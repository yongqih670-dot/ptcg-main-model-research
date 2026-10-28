"""Experimental bounded conditional inference; not an active training backend.

Each request owns a CPU sampling stream. Batching changes forward scheduling,
not raw action support, STOP conditioning or cross-environment RNG ownership.
Real actor/engine/GPU acceptance and full training recovery remain unverified.
"""
import numpy as np
import torch
from ptcg_lab.rl_core import merged_scores
from ptcg_lab.token_features import forced_selection


def choose_requests(model,features,requests,*,greedy=False,before_forward=None,max_rows=128):
    if model.training:raise ValueError('Batch inference requires eval mode; dropout changes sampling semantics')
    if not 1<=len(requests)<=8 or type(max_rows) is not int or not 1<=max_rows<=128:
        raise ValueError('Experimental inference batch exceeds bounded capacity')
    keys=[r['key'] for r in requests]
    if any(not isinstance(k,str) or not k for k in keys) or len(set(keys))!=len(keys):
        raise ValueError('Every environment needs one stable unique request identity')
    streams=[r['rng'] for r in requests]
    if len({id(g) for g in streams})!=len(streams) or any(not isinstance(g,torch.Generator) or g.device.type!='cpu' for g in streams):
        raise ValueError('Every environment must own an independent CPU sampling stream')
    original_rng=[g.get_state().clone() for g in streams]
    states=[];forwards=0;forward_rows=0
    for request in requests:
        observation=request['observation'];selection=observation['select'];n=len(selection['option'])
        lo,hi=selection['minCount'],selection['maxCount']
        if type(lo) is not int or type(hi) is not int or not 0<=lo<=hi<=n:
            raise ValueError('Invalid conditional action bounds')
        forced=forced_selection(selection)
        states.append({'n':n,'lo':lo,'hi':hi,'picked':list(range(n)) if forced and hi else [],
            'steps':[],'value':None,'logp':0.,'done':forced,'forced':forced})
    try:
        with torch.no_grad():
            while any(not s['done'] for s in states):
                prepared=[];row_count=0
                for index,(request,state) in enumerate(zip(requests,states)):
                    if state['done']:continue
                    rows,mapping=features.make(request['observation'],request['tracker'],request['forced_run'],state['picked'])
                    rows=np.array(rows,dtype=np.float32,copy=True);mapping=np.array(mapping,copy=True)
                    if rows.ndim!=2 or mapping.ndim!=2 or not len(rows) or len(mapping)!=len(rows) or not np.isfinite(rows).all():
                        raise ValueError('Invalid conditional actor feature batch')
                    if any(features.field(row,'oracle').any() for row in rows):raise ValueError('Hidden truth cannot enter the actor')
                    row_count+=len(rows)
                    if row_count>max_rows:raise ValueError('Refuse full overflow support above row cap; never truncate candidates')
                    prepared.append((index,rows,mapping))
                if before_forward:before_forward()
                device=next(model.parameters()).device
                scores,values=model.forward_eval(torch.from_numpy(np.concatenate([r for _,r,_ in prepared])).to(device))
                if len(scores)!=row_count or len(values)!=row_count or not torch.isfinite(values).all():
                    raise ValueError('Actor forward output shape/value differs')
                forwards+=1;forward_rows+=row_count;offset=0
                for index,rows,mapping in prepared:
                    state=states[index];count=len(rows)
                    logits=merged_scores(scores[offset:offset+count],mapping,state['n'],features.b.STOP).double().cpu()
                    if not torch.isfinite(logits).all():raise ValueError('Nonfinite actor logits')
                    distribution=torch.distributions.Categorical(logits=logits)
                    target=int(logits.argmax()) if greedy else int(torch.multinomial(distribution.probs,1,replacement=True,generator=streams[index]))
                    logp=float(distribution.log_prob(torch.tensor(target)))
                    if not np.isfinite(logp):raise ValueError('Nonfinite conditional probability')
                    if state['value'] is None:state['value']=float(values[offset,0])
                    state['steps'].append({'rows':rows,'mapping':mapping,'n':state['n'],'target':target})
                    state['logp']+=logp;offset+=count
                    if target==state['n']:
                        if len(state['picked'])<state['lo']:raise ValueError('STOP before conditional minimum')
                        state['done']=True
                    else:
                        if target in state['picked']:raise ValueError('Duplicate conditional action')
                        state['picked'].append(target);state['done']=len(state['picked'])==state['hi']
        results={}
        for request,state in zip(requests,states):
            if not state['lo']<=len(state['picked'])<=state['hi']:raise ValueError('Incomplete conditional sequence')
            decision=None if state['forced'] else {'steps':state['steps'],'old_logp':state['logp'],
                'old_value':state['value'],'reward':0.,'terminal':False}
            results[request['key']]=(state['picked'],decision)
        return results,{'requests':len(requests),'forward_calls':forwards,'forward_rows':forward_rows,
            'new_samples_forced':sum(s['forced'] for s in states),'not_an_actual_GPU_or_engine_acceptance':True}
    except BaseException:
        # No action is returned on failure. The caller can retry from its exact
        # environment state without advancing any request's sampling stream.
        for stream,state in zip(streams,original_rng):stream.set_state(state)
        raise
