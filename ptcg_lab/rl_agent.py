"""Greedy release adapter preserving the parent's rejection/time guards."""
from .public_agent import PublicRuleAgent
from .io import sha256


class TransformerRLAgent(PublicRuleAgent):
    def __init__(self,spec,seat,seed):
        super().__init__(spec,seat,seed)
        digest=sha256(self.folder/'model.pt')
        if digest!=spec['model_sha256']:raise ValueError('PPO release weights changed')
        self.identity.update(model_sha256=digest,feature_schema='full-inherited-blind-conditional-v3')
    def load(self,round_id=None):
        super().load(round_id)
        import types
        from .rl_core import RLFeatures,choose_decision
        features=RLFeatures(self.folder);sa=self.module.SA;original=sa._decompose
        sa.policy.critic_mode='blind'
        def decompose(agent,obs,tracker,forced_run,sample=True,temp=None):
            if len(obs['select']['option'])<=features.b.STOP:
                return original(obs,tracker,forced_run,sample=sample,temp=temp)
            return choose_decision(agent.policy,features,obs,tracker,forced_run,greedy=True)[0]
        sa._decompose=types.MethodType(decompose,sa)
