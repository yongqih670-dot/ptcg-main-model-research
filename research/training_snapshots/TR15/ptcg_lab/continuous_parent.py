"""CPU-accounted full-parent inspection and recovery-only numerical fixture."""
import copy
import io
import os
from pathlib import Path
import time

from .io import atomic_json, read_json, sha256, utcnow


def run(config, output):
    import numpy as np
    import torch
    from .archive_compressed import verify_archive
    from .ppo_accounting import checkpoint_decisions
    from .rl_core import load_full_actor, evaluate_decisions
    from .tensor_io import atomic_torch_save
    root = Path(config['project_path']).resolve()
    output = Path(output).resolve()
    if os.environ.get('PTCG_JOB_RUN_ID') != output.name or output.parent != root/'runs':
        raise RuntimeError('Controlled parent inspection required')
    if torch.cuda.is_available() or config.get('no_training_updates') is not True:
        raise ValueError('Parent inspection is CPU-only and executes no optimizer')
    deadline = float(os.environ['PTCG_JOB_DEADLINE_UNIX'])
    source = root/'runs'/config['parent_run_id']
    if source.resolve().parent != root/'runs':
        raise ValueError('Parent escaped')
    for relative, digest in config['parent_evidence'].items():
        if sha256(source/relative) != digest:
            raise ValueError('Terminal parent evidence changed')
    archive = source/'online-archive'
    saved = torch.load(io.BytesIO(verify_archive(archive, config['checkpoint_ACK'], collect=True)),
                       map_location='cpu', weights_only=False)
    if len(saved['model']) != 127 or len(saved['optimizer']['state']) != 97:
        raise ValueError('Full parent model/optimizer is missing')
    if checkpoint_decisions(saved) != 32782 or saved['version'] != 5:
        raise ValueError('Parent actual PPO coverage differs')
    if set(saved['rng']) != {'python', 'numpy', 'torch', 'cuda'} or len(saved['rng']['cuda']) != 1:
        raise ValueError('Full original RNG coverage differs')
    if 'cpu_sampler_rng' not in saved:
        raise ValueError('CPU actor RNG is missing')
    for value in saved['model'].values():
        if not torch.isfinite(value).all():
            raise ValueError('Parent model nonfinite')
    for slots in saved['optimizer']['state'].values():
        for value in slots.values():
            if torch.is_tensor(value) and not torch.isfinite(value).all():
                raise ValueError('Parent optimizer nonfinite')
    torch.set_num_threads(2)
    model, features, _ = load_full_actor(root/'agents/frozen/ppo-zero-anchor',
        config['origin_actor_sha256'], 'cpu', reset_value=False)
    model.load_state_dict(saved['model'], strict=True)
    fixture = []
    for ack in config['fixture_rollout_ACKs']:
        if time.time() > deadline-20:
            raise TimeoutError('Parent fixture deadline')
        shard = torch.load(io.BytesIO(verify_archive(archive, ack, collect=True)),
                           map_location='cpu', weights_only=False)
        if shard['summary']['statuses'] != ['DONE', 'DONE']:
            raise ValueError('Fixture source must be a complete real BO3')
        fixture.extend(copy.deepcopy(shard['decisions']))
        if len(fixture) >= 48:
            break
    fixture = fixture[:48]
    if len(fixture) != 48:
        raise ValueError('48 real legal observation fixtures required')
    lo, hi = features.b.OFFSETS['oracle']
    if any(step['rows'][:, lo:hi].any() for d in fixture for step in d['steps']):
        raise ValueError('Hidden fixture input')
    with torch.no_grad():
        logp, values, _ = evaluate_decisions(model, fixture, features.b.STOP)
    if not torch.isfinite(logp).all() or not torch.isfinite(values).all():
        raise ValueError('Nonfinite parent output')
    for d, prob, value in zip(fixture, logp.tolist(), values.tolist()):
        d.update(old_logp=prob, old_value=float(np.asarray(value).reshape(-1)[0]),
                 policy_version=saved['version'], recovery_fixture_only=True)
    target = output/'parent-numeric-reference.pt'
    atomic_torch_save(target, {'model':saved['model'], 'fixture':fixture,
                              'fixture_logp':logp.cpu(), 'fixture_values':values.cpu(),
                              'not_training_data':True}, replace=False)
    atomic_json(output/'metrics.json', {'at':utcnow(), 'complete':True, 'engineering_only':True,
        'classification':'complete_continuous_parent_verified', 'source_run_id':source.name,
        'parent_checkpoint_sha256':config['checkpoint_ACK']['sha256'],
        'model_tensors':127, 'optimizer_states':97, 'all_RNG_and_CPU_actor_RNG_present':True,
        'actual_unique_PPO_decisions':32782, 'PPO_updates':5,
        'numeric_reference':{'path':target.relative_to(root).as_posix(), 'sha256':sha256(target),
                             'bytes':target.stat().st_size},
        'fixture_legal_states':48, 'fixture_probabilities_recomputed_with_exact_parent':True,
        'fixture_must_never_enter_training':True, 'optimizer_steps_executed':0,
        'source_state_not_modified':True, 'promotion_eligible':False})
