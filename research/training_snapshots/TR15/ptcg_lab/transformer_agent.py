"""Full-token learned conditional policy inside the reviewed public guard."""
import types
from collections import Counter
import numpy as np
import torch
from .public_agent import PublicRuleAgent,ROOT
from .token_features import TokenFeatures,merge_logits,FEATURE_VERSION
from .transformer_model import build_model
from .io import sha256


class TransformerAgent(PublicRuleAgent):
    def __init__(self,spec,seat,seed):
        super().__init__(spec,seat,seed);self.learning_stats=Counter()
        self.model_path=(ROOT/spec['model_path']).resolve()
        if ROOT not in self.model_path.parents or sha256(self.model_path)!=spec['model_sha256']:raise ValueError('Learned model identity changed')
        self.learned_contexts=set(spec['learned_contexts'])
        self.identity.update(model_sha256=spec['model_sha256'],feature_version=FEATURE_VERSION,architecture=spec['architecture'],learned_contexts=spec['learned_contexts'])

    def load(self,round_id=None):
        super().load(round_id)
        features=TokenFeatures(self.folder);model,_=build_model(self.folder,self.spec['architecture'])
        model.load_state_dict(torch.load(self.model_path,map_location='cpu',weights_only=True));model.eval();model.freeze_tables()
        parent_model=self.module.SA.policy;original=self.module.SA._decompose
        def decompose(sa,obs,tracker,forced_run,sample=True,temp=None):
            sel=obs['select'];n=len(sel['option']);picked=[]
            learned=sel['context'] in self.learned_contexts
            if not learned:
                self.learning_stats['declared_unseen_context_ppo_fallback']+=1
                if n<=features.b.STOP:return original(obs,tracker,forced_run,sample=sample,temp=temp)
            active_model=model if learned else parent_model
            self.learning_stats['learned_choices']+=1;self.learning_stats['options_over_63']+=n>features.b.STOP
            self.learning_stats['multi_or_optional']+=sel['maxCount']>1 or sel['minCount']==0
            while len(picked)<sel['maxCount']:
                rows,mapping=features.make(obs,tracker,forced_run,picked)
                with torch.no_grad():logits=active_model.forward_policy(torch.from_numpy(rows)).numpy()
                scores=merge_logits(logits,mapping,n,features.b.STOP);target=int(scores.argmax());sa.evals+=len(rows)
                self.learning_stats['conditional_forwards']+=len(rows)
                if not np.isfinite(scores[target]) or scores[target]<-1e8:raise ValueError('No unmasked action')
                if target==n:
                    if len(picked)<sel['minCount']:raise ValueError('Premature STOP')
                    return picked
                if target in picked or not 0<=target<n:raise ValueError('Invalid conditional index')
                picked.append(target)
            return picked
        self.module.SA._decompose=types.MethodType(decompose,self.module.SA)

    def close(self):
        if self.module is not None:self.diagnostics.append({'round':self.current_round,'Transformer':dict(self.learning_stats)})
        super().close();self.learning_stats=Counter()
