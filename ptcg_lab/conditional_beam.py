"""Bounded beam over the frozen actor's ordered multi-choice distribution."""
import types
from collections import Counter
import numpy as np


def beam_answer(n,minimum,maximum,width,logits_for_prefixes):
    # (ordered prefix, log probability, completed). Finished paths compete
    # directly with unfinished paths; no length bonus or reward tuning.
    beam=[((),0.,maximum==0)]
    for depth in range(maximum+1):
        open_nodes=[node for node in beam if not node[2]]
        if not open_nodes:break
        logits=np.asarray(logits_for_prefixes([list(node[0]) for node in open_nodes]),np.float64)
        if logits.shape!=(len(open_nodes),n+1) or not np.isfinite(logits).all():raise ValueError('Malformed beam logits')
        expanded=[node for node in beam if node[2]]
        for (prefix,score,_),row in zip(open_nodes,logits):
            legal=[j for j in range(n+1) if row[j]>-1e8 and (j==n and len(prefix)>=minimum or j<n and j not in prefix)]
            if not legal:raise ValueError('Beam has no legal continuation')
            maximum_logit=max(row[j] for j in legal);logz=maximum_logit+np.log(sum(np.exp(row[j]-maximum_logit) for j in legal))
            # Stable index order preserves greedy argmax ties at width=1.
            for target in sorted(legal,key=lambda j:(-row[j],j))[:width]:
                picked=prefix if target==n else prefix+(target,)
                expanded.append((picked,score+row[target]-logz,target==n or len(picked)>=maximum))
        expanded.sort(key=lambda node:-node[1]);beam=expanded[:width]
    if not beam or not beam[0][2] or not minimum<=len(beam[0][0])<=maximum:raise ValueError('Unfinished beam')
    return list(beam[0][0])


def install_beam(module,width,bundle):
    import torch
    from .token_features import TokenFeatures,merge_logits
    if type(width) is not int or not 1<=width<=4:raise ValueError('Unreviewed beam width')
    features=TokenFeatures(bundle);b=features.b;original=module.SA._decompose
    if module.SA.cfg is not None or module.SA.temp!=0:raise ValueError('Beam experiment requires frozen greedy policy-only baseline')
    stats=Counter();module.BEAM_STATS=stats
    def decompose(sa,obs,tracker,forced_run,sample=True,temp=None):
        sel=obs['select'];n=len(sel['option'])
        if sel['maxCount']<=1:
            stats['unchanged_single_choices']+=1
            return original(obs,tracker,forced_run,sample=sample,temp=temp)
        stats['beam_choices']+=1;stats['large_option_choices']+=n>b.STOP
        def logits(prefixes):
            rows=[];maps=[];counts=[]
            for prefix in prefixes:
                if n<=b.STOP:
                    row=np.zeros(b.OBS_SIZE,np.float32)
                    features.encode(row,obs,prefix,len(prefix)>=sel['minCount'],forced_run,tracker=tracker)
                    # Preserve the public encoder's equivalence-class mask.
                    rows.append(row[None,:]);maps.append(None);counts.append(1)
                else:
                    chunk,mapping=features.make(obs,tracker,forced_run,prefix)
                    rows.append(chunk);maps.append(mapping);counts.append(len(chunk))
            with torch.no_grad():scores=sa.policy.forward_policy(torch.from_numpy(np.concatenate(rows))).cpu().numpy()
            result=[];offset=0
            for count,mapping in zip(counts,maps):
                block=scores[offset:offset+count];offset+=count
                if mapping is None:result.append(np.concatenate([block[0,:n],block[0,b.STOP:b.STOP+1]]))
                else:result.append(merge_logits(block,mapping,n,b.STOP))
            stats['evaluated_prefixes']+=len(prefixes);stats['encoded_rows']+=len(scores);sa.evals+=len(scores)
            return np.stack(result)
        return beam_answer(n,sel['minCount'],sel['maxCount'],width,logits)
    module.SA._decompose=types.MethodType(decompose,module.SA)
