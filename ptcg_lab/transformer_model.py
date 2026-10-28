"""Reviewed MIT actor architecture; only shape-compatible semantic warm start."""
from pathlib import Path
from .bc_features import load_encoder
from .io import sha256

SEMANTIC_PREFIXES=('card_id_emb.','attack_id_emb.','skill_id_emb.','attack_mlp.','skill_mlp.','card_mlp.',
    'zone_emb.','owner_emb.','global_mlp.','sel_type_emb.','ctx_emb.','dec_mlp.','opt_type_emb.','opt_card.','opt_attack.')
UNUSED_PREFIXES=('value_mlp.','oracle_proj.','critic_readout','critic_blocks.','oracle_value_mlp.','aux_hand.','aux_prize.')


def build_model(bundle,architecture,parent_hash=None):
    import torch
    load_encoder(bundle)
    from ptcg.rl.model import PTCGTransformer
    model=PTCGTransformer(**architecture,critic='blind')
    copied=[]
    if parent_hash:
        path=Path(bundle)/'model.pt'
        if sha256(path)!=parent_hash:raise ValueError('Parent checkpoint changed')
        parent=torch.load(path,map_location='cpu',weights_only=True);state=model.state_dict()
        for key,value in parent.items():
            key=key.removeprefix('module.')
            if key.startswith(SEMANTIC_PREFIXES) and key in state and value.shape==state[key].shape:
                state[key]=value;copied.append(key)
        model.load_state_dict(state)
    for name,p in model.named_parameters():
        if name.startswith(UNUSED_PREFIXES):p.requires_grad_(False)
    report={'parent_model_sha256':parent_hash,'copied_semantic_keys':copied,'architecture':architecture,
        'parameters_total':sum(p.numel() for p in model.parameters()),'parameters_trainable':sum(p.numel() for p in model.parameters() if p.requires_grad),
        'actor_path':'forward_policy only; oracle/value/belief modules not trained; oracle rows zero',
        'limitations':'Upstream static embeddings map cards >1267 to UNKNOWN and attacks >1556 to zero'}
    return model,report
