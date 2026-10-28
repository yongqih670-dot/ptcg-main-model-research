"""FP32 forward / FP64 conditional reduction with reusable CPU layout.

No policy changes: preserve raw candidates, one STOP, decision order and value
at each decision's first row. Duplicate raw mappings follow baseline last-win
semantics; padding is never a selectable candidate.
"""
from dataclasses import dataclass
from contextlib import nullcontext
import numpy as np
import torch


@dataclass(frozen=True)
class Packed:
    rows: np.ndarray
    gather: np.ndarray
    valid: np.ndarray
    targets: np.ndarray
    decision_ids: np.ndarray
    first_rows: np.ndarray
    stop_rows: np.ndarray
    stop_references: np.ndarray
    stop: int

    @classmethod
    def build(cls, decisions, stop, max_rows=32768, max_candidates=4096):
        if not decisions or type(stop) is not int or stop < 0:
            raise ValueError('Empty decisions or invalid STOP')
        rows=[]; indices=[]; targets=[]; owners=[]; first=[]; checks=[]; refs=[]; offset=0
        width=stop+1
        for di, d in enumerate(decisions):
            if not d['steps']:
                raise ValueError('Forced decisions must not enter PPO')
            first.append(offset)
            for s in d['steps']:
                r=np.asarray(s['rows']); m=np.asarray(s['mapping']); n=s['n']; target=s['target']
                if (r.ndim!=2 or not len(r) or m.shape!=(len(r),width)
                        or r.dtype!=np.float32 or not np.isfinite(r).all()
                        or not np.issubdtype(m.dtype,np.integer)
                        or type(n) is not int or not 0<=n<max_candidates
                        or not 0<=target<=n or int(target)!=target):
                    raise ValueError('Invalid conditional layout')
                table={int(t):(offset+i)*width+j for i,row in enumerate(m) for j,t in enumerate(row) if 0<=t<n}
                if len(table)!=n:
                    raise ValueError('Missing raw candidate')
                indices.append([table[j] for j in range(n)]+[offset*width+stop])
                rows.append(r); targets.append(target); owners.append(di)
                if len(r)>1:
                    checks.extend(range(offset,offset+len(r)));refs.extend([offset]*len(r))
                offset+=len(r)
                if offset>max_rows:
                    raise ValueError('Packed row cap exceeded; never truncate support')
        count=max(map(len,indices))
        gather=np.zeros((len(indices),count),np.int64);valid=np.zeros_like(gather,dtype=bool)
        for i,idx in enumerate(indices):
            gather[i,:len(idx)]=idx;valid[i,:len(idx)]=True
        arrays=[np.concatenate(rows),gather,valid,np.asarray(targets,np.int64),np.asarray(owners,np.int64),
                np.asarray(first,np.int64),np.asarray(checks,np.int64),np.asarray(refs,np.int64)]
        for a in arrays:a.setflags(write=False)
        return cls(*arrays,stop)

    def tensors(self, device):
        # Explicit copies own their lifetime. Pinned/nonblocking disabled until
        # controlled hardware acceptance; no persistent GPU rollout cache.
        return tuple(torch.tensor(a,device=device) for a in
                     (self.rows,self.gather,self.valid,self.targets,self.decision_ids,
                      self.first_rows,self.stop_rows,self.stop_references))


def reduce_outputs(scores, values, tensors, stop):
    _,gather,valid,targets,owners,first,checks,refs=tensors
    if scores.ndim!=2 or scores.shape[1]!=stop+1 or values.shape!=(scores.shape[0],1):
        raise ValueError('Forward output shape differs')
    if checks.numel() and not torch.allclose(scores[checks,stop],scores[refs,stop],atol=1e-4,rtol=1e-4):
        raise ValueError('Chunk STOP scores disagree')
    logits=scores.reshape(-1)[gather].double().masked_fill(~valid,float('-inf'))
    distribution=torch.distributions.Categorical(logits=logits)
    terms=distribution.log_prob(targets); entropy=distribution.entropy()
    logp=terms.new_zeros(len(first)).scatter_add(0,owners,terms)
    ent=entropy.new_zeros(len(first)).scatter_add(0,owners,entropy)
    return logp, values[first,0], ent


def evaluate_decisions(model, decisions, stop, *, trace=None, precision=None):
    span=trace.span if trace else lambda *a,**k:nullcontext()
    with span('data.pack'):
        batch=Packed.build(decisions,stop)
    return evaluate_packed(model,batch,trace=trace,precision=precision)


def evaluate_packed(model,batch,*,trace=None,precision=None):
    """Evaluate only this immutable snapshot; no second decisions list to mismatch."""
    span=trace.span if trace else lambda *a,**k:nullcontext()
    with span('data.H2D'):
        tensors=batch.tensors(next(model.parameters()).device)
    with span('policy.forward'):
        with precision.autocast() if precision else nullcontext():
            scores,values=model.forward_eval(tensors[0])
    with span('policy.conditional_reduce'):
        # AMP output precision is lifted before sensitive probability arithmetic.
        return reduce_outputs(scores.float(),values.float(),tensors,batch.stop)
