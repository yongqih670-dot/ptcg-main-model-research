"""BC single-choice branch; audited frozen PPO handles all other contexts."""
import types
from collections import Counter
from pathlib import Path
from .public_agent import PublicRuleAgent,ROOT
from .bc_features import CandidateFeatures,supported,FEATURE_VERSION
from .io import sha256,read_json


class BCSingleAgent(PublicRuleAgent):
    def __init__(self,spec,seat,seed):
        super().__init__(spec,seat,seed)
        path=(ROOT/spec['model_path']).resolve()
        if ROOT not in path.parents:raise ValueError('Model must be in frozen source snapshot')
        from catboost import CatBoostClassifier
        self.model=CatBoostClassifier();self.model.load_model(str(path))
        if sha256(path)!=spec['model_sha256']:raise ValueError('BC model hash differs')
        self.features=None;self.bc_stats=Counter()
        self.learned_contexts=set(spec['learned_contexts'])
        self.identity.update(model_sha256=spec['model_sha256'],feature_version=FEATURE_VERSION)
        self.identity['learned_contexts']=sorted(self.learned_contexts)

    def load(self,round_id=None):
        super().load(round_id)
        self.features=CandidateFeatures(self.folder)
        original=self.module.SA._decompose
        def decompose(sa,obs,tracker,forced_run,sample=True,temp=None):
            if supported(obs['select']) and obs['select']['context'] in self.learned_contexts:
                matrix=self.features.make(obs,tracker,forced_run)
                score=self.model.predict(matrix,prediction_type='RawFormulaVal',thread_count=1)
                self.bc_stats['learned_single_choices']+=1
                self.bc_stats['options_over_63']+=len(matrix)>63
                return [int(score.argmax())]
            self.bc_stats['declared_ppo_fallback_choices']+=1
            return original(obs,tracker,forced_run,sample=sample,temp=temp)
        self.module.SA._decompose=types.MethodType(decompose,self.module.SA)

    def close(self):
        if self.module is not None:
            self.diagnostics.append({'round':self.current_round,'BC':dict(self.bc_stats)})
        super().close();self.bc_stats=Counter()
