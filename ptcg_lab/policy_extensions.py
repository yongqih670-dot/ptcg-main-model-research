"""Opt-in blind zone readouts and decision-clock recurrent policy residuals.

The inherited actor is unmodified. Zero output projections preserve its initial
logits/value; hidden state belongs to one player and one game, never the model.
"""
import torch
from torch import nn
from types import SimpleNamespace


VERSION = 'zone-gru-v2'
# Stable v4 buffer zone IDs: own board, opposing board, own hand, resources.
ZONE_GROUPS = ((1, 2), (3, 4), (5,), (7, 8, 9, 10, 11, 12, 13))


def validate_options(options):
    options = dict(options or {})
    if set(options) - {'zone_readout', 'memory_size', 'heads'}:
        raise ValueError('Unknown policy extension option')
    result = dict(zone_readout=options.get('zone_readout', False),
                  memory_size=options.get('memory_size', 0), heads=options.get('heads', 4))
    if (type(result['zone_readout']) is not bool or type(result['memory_size']) is not int
            or result['memory_size'] < 0 or type(result['heads']) is not int or result['heads'] < 1):
        raise ValueError('Invalid policy extension options')
    if not result['zone_readout'] and not result['memory_size']:
        raise ValueError('Omit extensions for the unchanged history-only policy')
    return result


class ZoneMemoryPolicy(nn.Module):
    def __init__(self, base, buffers, options):
        super().__init__()
        if base.critic_mode != 'blind':
            raise ValueError('Extensions require a blind critic')
        self.base = base
        # Keep a serializable schema, not an imported Python module: frozen
        # opponents and CPU replicas must be safely deepcopy-able.
        self.b = SimpleNamespace(**{name:getattr(buffers,name) for name in
            ('OFFSETS','MAX_OPTIONS','OPT_INT','OPT_F','MAX_TOKENS')})
        self.options = validate_options(options)
        self.recurrent = bool(self.options['memory_size'])
        self.register_buffer('_extension_schema',torch.tensor([
            2,int(self.options['zone_readout']),self.options['memory_size'],self.options['heads']],dtype=torch.int64))
        width = base.d
        if width % self.options['heads']:
            raise ValueError('Readout heads must divide inherited width')
        if self.options['zone_readout']:
            self.zone_attention = nn.ModuleList([
                nn.MultiheadAttention(width, self.options['heads'], dropout=0, batch_first=True)
                for _ in ZONE_GROUPS])
            self.zone_projection = nn.Linear(len(ZONE_GROUPS) * width, width, bias=False)
        else:
            self.zone_attention = nn.ModuleList()
        if self.recurrent:
            self.memory = nn.GRUCell(width, self.options['memory_size'])
            self.memory_projection = nn.Linear(self.options['memory_size'], width, bias=False)
        self.option_projection = nn.Linear(base.opt_type_emb.embedding_dim+base.d_c+
            base.attack_id_emb.embedding_dim+self.b.OPT_F,width)
        self.policy_residual = nn.Sequential(nn.Linear(3 * width, width), nn.GELU(), nn.Linear(width, 1))
        self.value_residual = nn.Linear(width, 1)
        nn.init.zeros_(self.policy_residual[-1].weight)
        nn.init.zeros_(self.policy_residual[-1].bias)
        nn.init.zeros_(self.value_residual.weight)
        nn.init.zeros_(self.value_residual.bias)

    @property
    def _frozen_tables(self):
        return getattr(self.base, '_frozen_tables', None)

    def freeze_tables(self):
        self.base.freeze_tables()

    def metadata(self):
        return {'version': VERSION, 'options': dict(self.options), 'memory_clock': 'nonforced-decision',
                'reset': 'each-player-each-game', 'critic': 'blind'}

    def _load_from_state_dict(self,state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs):
        schema=state_dict.get(prefix+'_extension_schema')
        if not isinstance(schema,torch.Tensor) or not torch.equal(schema.cpu(),self._extension_schema.cpu()):
            error_msgs.append('Policy extension schema/attention heads differ from checkpoint')
        super()._load_from_state_dict(state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs)

    def forward_step(self, rows, memory=None, *, advance=True):
        start, end = self.b.OFFSETS['oracle']
        if torch.count_nonzero(rows[:, start:end]):
            raise ValueError('Hidden oracle input forbidden')
        c = self.base._core(rows)
        r = c['r']
        context = r
        if self.zone_attention:
            tokens = c['h'][:, 1:]
            zones = c['tok_i'][:, :tokens.shape[1], 1]
            # A real zero sentinel avoids all-masked attention for empty zones.
            bank = torch.cat([tokens, torch.zeros_like(r[:, None])], dim=1)
            groups = []
            for ids, attention in zip(ZONE_GROUPS, self.zone_attention):
                keep = torch.zeros_like(zones, dtype=torch.bool)
                for zone in ids:
                    keep |= zones == zone
                keep = torch.cat([keep, ~keep.any(1, keepdim=True)], dim=1)
                value, _ = attention(r[:, None], bank, bank, key_padding_mask=~keep, need_weights=False)
                groups.append(value[:, 0])
            context = context + self.zone_projection(torch.cat(groups, dim=-1))
        if self.recurrent:
            if memory is None:
                memory = context.new_zeros(1, self.options['memory_size'])
            # All overflow chunks describe the same state. Advance only once,
            # on the first conditional pick, and use it for every later pick.
            if advance:
                memory = self.memory(context[:1], memory)
            context = context + self.memory_projection(memory).expand_as(context)
        oi = rows[:, slice(*self.b.OFFSETS['opt_int'])].reshape(-1, self.b.MAX_OPTIONS, self.b.OPT_INT).long()
        bank = c['h'].new_zeros(rows.shape[0], self.b.MAX_TOKENS + 1, self.base.d)
        bank[:, 1:c['h'].shape[1]] = c['h'][:, 1:]
        refs = [bank.gather(1, oi[:, :, j].clamp(0, self.b.MAX_TOKENS).unsqueeze(-1).expand(-1, -1, self.base.d)) for j in (1, 2)]
        # Entity pointers alone cannot distinguish two attacks/cards on the
        # same source/target. Let memory interact with the actual candidate.
        of=rows[:,slice(*self.b.OFFSETS['opt_float'])].reshape(-1,self.b.MAX_OPTIONS,self.b.OPT_F)
        option=self.option_projection(torch.cat([
            self.base.opt_type_emb(oi[:,:,0].clamp(0,self.base.opt_type_emb.num_embeddings-1)),
            c['card_vec'][oi[:,:,3].clamp(0,c['card_vec'].shape[0]-1)],
            self.base.attack_id_emb(oi[:,:,4].clamp(0,self.base.attack_id_emb.num_embeddings-1)),of],-1))
        delta = self.policy_residual(torch.cat([context[:, None].expand(-1, self.b.MAX_OPTIONS, -1), refs[0]+option,refs[1]], -1)).squeeze(-1)
        mask = rows[:, slice(*self.b.OFFSETS['opt_mask'])] != 0
        logits = (c['logits'] + delta).masked_fill(~mask, -1e9)
        value = self.base.value_mlp(torch.cat([r, c['g']], -1)) + self.value_residual(context)
        return logits, value, memory

    def forward_eval(self, rows, state=None):
        if self.recurrent:
            raise ValueError('Recurrent model requires explicit forward_step/sequence replay')
        logits, value, _ = self.forward_step(rows)
        return logits, value


def value_parameter(name):
    return name.startswith(('value_mlp.', 'base.value_mlp.', 'value_residual.'))


def complete_sequences(decisions):
    """Reject shuffled, partial, duplicated or mixed-version recurrent games."""
    sequences, current, seen = [], [], set()
    identity = None
    policy_version = None
    for i, decision in enumerate(decisions):
        key = tuple(decision.get(k) for k in ('series_id', 'round', 'learner_seat', 'policy_version'))
        if any(v is None for v in key):
            raise ValueError('Recurrent decisions need complete game/policy identity')
        if policy_version is None:policy_version=key[-1]
        if key[-1]!=policy_version:raise ValueError('Mixed recurrent policy versions')
        if type(decision.get('sequence_start')) is not bool or type(decision.get('terminal')) is not bool:
            raise ValueError('Sequence boundary flags must be booleans')
        if type(decision.get('sequence_index')) is not int or decision['sequence_index']!=len(current):
            raise ValueError('Missing, duplicated or reordered recurrent decision')
        if not isinstance(decision.get('steps'),list) or not decision['steps']:
            raise ValueError('Recurrent decision has no conditional steps')
        if not current:
            if not decision.get('sequence_start') or key in seen:
                raise ValueError('Missing sequence start or duplicate game')
            identity = key
            seen.add(key)
        elif decision.get('sequence_start') or key != identity:
            raise ValueError('Discontinuous recurrent game')
        current.append(i)
        if decision['terminal']:
            sequences.append(current)
            current = []
    if current or not sequences:
        raise ValueError('Only complete games may enter recurrent PPO')
    return sequences


def sequence_order(decisions, rng):
    sequences = complete_sequences(decisions)
    return [i for j in rng.permutation(len(sequences)) for i in sequences[j]]


def sequence_batch_size(decisions, ordered_indices, budget):
    """Pack whole games; one game may exceed the nominal decision budget."""
    count = 0
    for index in ordered_indices:
        count += 1
        if decisions[index]['terminal'] and count >= budget:
            return count
    return count
