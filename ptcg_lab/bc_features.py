"""224 public state/action features, with coverage of EVERY legal option.

The reviewed MIT encoder provides static card/attack calculations and the blind
tracker. Its 63-option capacity is handled by chunks, never by truncation.
Only pointer-target tokens are gathered; the oracle buffer is never read.
"""
import sys
from pathlib import Path
import numpy as np
from .public_agent import install_public_api,package_identity

FEATURE_VERSION='candidate224-v1'
FEATURE_COUNT=224


def load_encoder(bundle):
    bundle=Path(bundle).resolve();install_public_api()
    if 'ptcg' in sys.modules:
        existing=Path(sys.modules['ptcg'].__file__).resolve().parents[1]
        if package_identity(existing)!=package_identity(bundle):raise RuntimeError('Incompatible feature package already imported')
    else:sys.path.insert(0,str(bundle))
    from ptcg.rl.encoder import encode
    from ptcg.rl import buffers
    from ptcg.tracker import InfoTracker
    return encode,buffers,InfoTracker


class CandidateFeatures:
    def __init__(self,bundle):
        self.encode,self.b,self.Tracker=load_encoder(bundle)
        self.row=np.zeros(self.b.OBS_SIZE,dtype=np.float32)

    def slice(self,name):
        a,b=self.b.OFFSETS[name];return self.row[a:b]

    def make(self,obs,tracker,forced_run):
        sel=obs['select'];options=sel['option'];parts=[];b=self.b
        if not(sel['minCount']==sel['maxCount']==1):raise ValueError('Only mandatory single-choice branch supported')
        for start in range(0,len(options),b.MAX_OPTIONS-1):
            chunk=options[start:start+b.MAX_OPTIONS-1]
            local={**obs,'select':{**sel,'option':chunk}}
            self.encode(self.row,local,[],False,forced_run,tracker=tracker)
            self.slice('global_f')[b.GlobF.ACTOR_SEAT]=0
            self.slice('dec_float')[b.DecF.N_OPTIONS]=len(options)/30.0
            common=np.concatenate([self.slice('global_f'),self.slice('dec_int'),self.slice('dec_float')])
            oi=self.slice('opt_int').reshape(b.MAX_OPTIONS,b.OPT_INT)[:len(chunk)]
            of=self.slice('opt_float').reshape(b.MAX_OPTIONS,b.OPT_F)[:len(chunk)]
            tokens=np.concatenate([self.slice('tok_int').reshape(b.MAX_TOKENS,b.TOK_INT),
                self.slice('tok_float').reshape(b.MAX_TOKENS,b.TOK_F)],axis=1)
            targets=[]
            for column in (1,2):
                indices=oi[:,column].astype(int)-1
                values=tokens[np.maximum(indices,0)].copy();values[indices<0]=0;targets.append(values)
            parts.append(np.concatenate([np.tile(common,(len(chunk),1)),oi,of,*targets],axis=1))
        result=np.concatenate(parts) if parts else np.empty((0,FEATURE_COUNT),dtype=np.float32)
        if result.shape!=(len(options),FEATURE_COUNT) or not np.isfinite(result).all():raise ValueError('Invalid candidate feature matrix')
        return result


def supported(sel):
    return sel['minCount']==sel['maxCount']==1 and len(sel['option'])>1
