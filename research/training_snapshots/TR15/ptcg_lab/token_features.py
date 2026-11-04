"""Blind full-token inputs and conditional raw-index choices, without truncation."""
import numpy as np
from .bc_features import load_encoder

FEATURE_VERSION='full-token-conditional-v1'


def forced_selection(sel):
    n=len(sel['option'])
    return sel['maxCount']==0 or sel['minCount']==sel['maxCount']==n or n==1 and sel['minCount']>=1


def conditional_targets(sel,action):
    n=len(sel['option']);lo=sel['minCount'];hi=sel['maxCount']
    if not lo<=len(action)<=hi or len(set(action))!=len(action) or any(type(a) is not int or not 0<=a<n for a in action):
        raise ValueError('Invalid teacher conditional action')
    if forced_selection(sel):return []
    steps=[];picked=[]
    for target in action:
        steps.append((picked.copy(),target));picked.append(target)
    if len(picked)<hi:steps.append((picked.copy(),n))  # global STOP is n
    return steps


class TokenFeatures:
    def __init__(self,bundle):
        self.encode,self.b,self.Tracker=load_encoder(bundle)

    def field(self,row,name):
        a,z=self.b.OFFSETS[name];return row[a:z]

    def make(self,obs,tracker,forced_run,picked):
        b=self.b;sel=obs['select'];n=len(sel['option']);picked=set(picked)
        if not picked.issubset(range(n)) or len(picked)>=sel['maxCount']:
            raise ValueError('No remaining conditional decision')
        stop_ok=len(picked)>=sel['minCount'];rows=[];mappings=[]
        for start in range(0,max(1,n),b.STOP):
            options=sel['option'][start:start+b.STOP]
            local_picked=[j-start for j in sorted(picked) if start<=j<start+len(options)]
            row=np.zeros(b.OBS_SIZE,np.float32)
            self.encode(row,{**obs,'select':{**sel,'option':options}},local_picked,stop_ok,forced_run,tracker=tracker)
            gf=self.field(row,'global_f');gf[b.GlobF.ACTOR_SEAT]=0
            gf[b.GlobF.PICKS_MIN_LEFT]=max(0,sel['minCount']-len(picked))/5.
            gf[b.GlobF.PICKS_MAX_LEFT]=max(0,sel['maxCount']-len(picked))/10.
            df=self.field(row,'dec_float');df[b.DecF.N_OPTIONS]=n/30.;df[b.DecF.PICKED]=len(picked)/5.
            mask=self.field(row,'opt_mask');mask[:]=0
            mask[:len(options)]=[start+j not in picked for j in range(len(options))]
            mask[b.STOP]=stop_ok
            mapping=np.full(b.MAX_OPTIONS,-1,np.int32);mapping[:len(options)]=range(start,start+len(options));mapping[b.STOP]=n
            if self.field(row,'oracle').any() or not np.isfinite(row).all():raise ValueError('Non-blind or nonfinite encoder row')
            rows.append(row);mappings.append(mapping)
        # Chunk-local references/picks change token flags. Union them so EVERY
        # chunk has exactly the same board/global/decision representation.
        token_arrays=[self.field(row,'tok_float').reshape(b.MAX_TOKENS,b.TOK_F) for row in rows]
        for column in (b.TokF.PICKED,b.TokF.REFERENCED):
            shared=np.maximum.reduce([a[:,column] for a in token_arrays])
            for a in token_arrays:a[:,column]=shared
        if len(rows)>1:
            for name in ('tok_int','tok_float','global_f','dec_int','dec_float'):
                if any(not np.array_equal(self.field(rows[0],name),self.field(row,name)) for row in rows[1:]):
                    raise ValueError('Chunk-dependent state representation: '+name)
        return np.stack(rows),np.stack(mappings)


def merge_logits(logits,mappings,n,stop):
    result=np.full(n+1,-np.inf,np.float32)
    for scores,mapping in zip(logits,mappings):
        for j,target in enumerate(mapping):
            if target>=0 and j!=stop:result[target]=scores[j]
    # Shared state and identical STOP features imply identical STOP logits.
    if not np.allclose(logits[:,stop],logits[0,stop],rtol=1e-4,atol=1e-4):
        raise ValueError('Chunk STOP scores disagree')
    result[n]=logits[0,stop]
    return result
