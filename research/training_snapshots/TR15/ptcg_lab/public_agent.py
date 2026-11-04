"""Reviewed rule notebook adapter, deliberately without native search access.

This is not a sandbox for arbitrary Python. Only audited sources are admitted.
The single-threaded BO3 runner restores cwd/sys.path after each call.
"""
import atexit
import contextlib
import importlib.util
import os
import random
import sys
import types
from pathlib import Path

from .agents import ProtocolAgent
from .io import sha256

ROOT = Path(__file__).resolve().parents[1]


def package_identity(folder):
    """Cached Python package globals may be shared only for identical code/data."""
    return {str(p.relative_to(folder)): sha256(p)
            for child in ('ptcg', 'data') for p in (folder / child).rglob('*')
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}


def map_actor_seat_tensor(observation, seat_column, mode, first_decided_column, i_am_first_column):
    if mode == 'physical': return observation
    changed=observation.clone()
    if mode == 'zero': changed[:,seat_column]=0
    elif mode == 'turn_order':
        changed[:,seat_column]=(1-changed[:,i_am_first_column])*changed[:,first_decided_column]
    else: raise ValueError('Unknown actor seat feature mode')
    return changed


class ObservableRandom(random.Random):
    def __init__(self, seed):
        super().__init__(seed)
        self.fallback_calls = 0

    def sample(self, *args, **kwargs):
        self.fallback_calls += 1
        return super().sample(*args, **kwargs)


@contextlib.contextmanager
def working_directory(path):
    old, old_path = Path.cwd(), sys.path.copy()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)
        sys.path[:] = old_path


def install_public_api():
    from . import public_cg
    from .public_cg import api, sim, utils
    for name, module in [('cg', public_cg), ('cg.api', api), ('cg.sim', sim), ('cg.utils', utils)]:
        existing = sys.modules.get(name)
        if existing is not None and existing is not module:
            raise RuntimeError('Conflicting cg import: a second native runtime is not permitted')
        sys.modules[name] = module


class PublicRuleAgent(ProtocolAgent):
    def __init__(self, spec, seat, seed):
        self.spec = spec
        self.folder = (ROOT / spec['path']).resolve()
        if ROOT not in self.folder.parents:
            raise ValueError('Agent path must be inside frozen source root')
        deck = list(map(int, (self.folder / 'deck.csv').read_text(encoding='utf-8-sig').split()))
        super().__init__('first', deck, seat, seed)
        self.name = spec['id']
        self.module = None
        self.current_round = None
        self.seed = seed
        self.diagnostics = []
        self.identity = {'id': self.name, 'code_sha256': sha256(self.folder / 'main.py'),
                         'deck_sha256': sha256(self.folder / 'deck.csv'),
                         'adapter': 'public_metadata_only_round_reset_v1',
                         'parameters': spec.get('parameters', {})}

    def load(self, round_id=None):
        self.close()
        if round_id is not None:
            self.current_round = round_id
        if self.spec.get('lineage') == 'cggonzalezsc_lucario' and any(
                os.environ.get(k) for k in ('QCOL_OUT', 'QCOL_RIVDECK', 'Q_STATS_FILE')):
            raise RuntimeError('Offline oracle/collection environment is forbidden')
        install_public_api()
        module_spec = importlib.util.spec_from_file_location(f'ptcg_policy_seat_{self.seat}', self.folder / 'main.py')
        module = importlib.util.module_from_spec(module_spec)
        if self.spec.get('lineage') == 'dipam_unown_gradiant_ppo' and 'ptcg' in sys.modules:
            imported_root = Path(sys.modules['ptcg'].__file__).resolve().parents[1]
            if imported_root != self.folder and package_identity(imported_root) != package_identity(self.folder):
                raise RuntimeError('Conflicting PPO package versions must use separate worker processes')
        with working_directory(self.folder):
            module_spec.loader.exec_module(module)
        self.module = module
        # The Lucario runtime ships disabled collection/search code. Disallow its
        # opt-in environment switches instead of ever exposing hidden truth.
        if self.spec.get('lineage') == 'cggonzalezsc_lucario':
            if module._Q_COLLECT or module._Q_MODE != 'policy':
                raise RuntimeError('Only the public observation policy mode is permitted')
        if getattr(module, 'random', None) is random:
            module.random = ObservableRandom(self.seed + (self.current_round or 0))
        for key, value in self.spec.get('parameters', {}).items():
            if key == '_conditional_beam_width':
                if self.spec.get('lineage') != 'dipam_unown_gradiant_ppo':raise ValueError('Beam applies only to reviewed PPO')
                from .conditional_beam import install_beam
                install_beam(module,value,self.folder)
                continue
            if key == '_actor_seat_mode':
                if self.spec.get('lineage') != 'dipam_unown_gradiant_ppo':
                    raise ValueError('Actor feature mapping applies only to the reviewed PPO')
                if value not in ('physical','zero','turn_order'): raise ValueError('Unknown actor seat mode')
                from ptcg.rl.buffers import SEAT_COL,OFFSETS,GlobF
                old_core=module.SA.policy._core
                def mapped_core(model, observation, mode=value, original=old_core):
                    row=map_actor_seat_tensor(observation,SEAT_COL,mode,
                        OFFSETS['global_f'][0]+GlobF.FIRST_DECIDED,
                        OFFSETS['global_f'][0]+GlobF.I_AM_FIRST)
                    return original(row)
                module.SA.policy._core=types.MethodType(mapped_core,module.SA.policy)
                continue
            if key not in ('_Q_ON', '_Q_MARGIN', '_R19_B_ON') or not hasattr(module, key):
                raise ValueError(f'Unreviewed policy parameter: {key}')
            setattr(module, key, value)

    def choose(self, obs):
        round_id = (obs.get('current') or {}).get('round')
        if self.module is None:
            self.load()
        elif round_id is not None and self.current_round is not None and round_id != self.current_round:
            self.load(round_id)
        if round_id is not None:
            self.current_round = round_id
        # Opaque engine serialization is unnecessary for these rules and is not exposed.
        public = dict(obs)
        public['search_begin_input'] = None
        with working_directory(self.folder):
            result = self.module.agent(public)
        if obs.get('select') is None and result != self.deck:
            raise ValueError('Policy deck differs from frozen deck.csv')
        return result

    def close(self):
        if self.module is not None:
            self.diagnostics.append({'round': self.current_round,
                'random_fallback_calls': getattr(getattr(self.module, 'random', None), 'fallback_calls', 0),
                'QSTATS': dict(getattr(self.module, 'QSTATS', {})),
                'BEAM': dict(getattr(self.module, 'BEAM_STATS', {})),
                'GUARD': dict(getattr(self.module, 'GUARD', {}))})
        if self.module is not None and hasattr(self.module, '_flush'):
            atexit.unregister(self.module._flush)
        self.module = None


def create_agent(spec, fallback_deck, seat, seed):
    if isinstance(spec, str):
        return ProtocolAgent(spec, fallback_deck, seat, seed)
    if spec['kind'] == 'public_rule':
        return PublicRuleAgent(spec, seat, seed)
    if spec['kind'] == 'catboost_single':
        from .bc_agent import BCSingleAgent
        return BCSingleAgent(spec,seat,seed)
    if spec['kind'] == 'transformer_bc':
        from .transformer_agent import TransformerAgent
        return TransformerAgent(spec,seat,seed)
    if spec['kind'] == 'transformer_rl':
        from .rl_agent import TransformerRLAgent
        return TransformerRLAgent(spec,seat,seed)
    raise ValueError(f'Unknown policy kind: {spec["kind"]}')
