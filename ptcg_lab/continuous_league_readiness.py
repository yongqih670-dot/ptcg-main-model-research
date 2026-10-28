"""Full CPU construction of the exact TR11 league before another GPU launch."""
import copy
import hashlib
import io
import os
from pathlib import Path
import time
from .io import read_json,atomic_json,sha256,utcnow


def run(config,output):
    import torch
    from .continuous_mainline import normalize_fixed_file_paths
    from .archive_compressed import verify_archive
    from .rl_core import load_full_actor
    from .rl_league import League
    root=Path(config['project_path']).resolve();output=Path(output).resolve()
    if os.environ.get('PTCG_JOB_RUN_ID')!=output.name or output.parent!=root/'runs':raise RuntimeError('Controlled CPU check required')
    deadline=float(os.environ['PTCG_JOB_DEADLINE_UNIX'])
    if not 0<deadline-time.time()<=180.5:raise ValueError('One bounded180s CPU readiness check only')
    started=time.monotonic();torch.set_num_threads(2)
    for name,digest in config['input_evidence'].items():
        if sha256(root/name)!=digest:raise ValueError('Pinned readiness input changed')
    inventory=read_json(root/config['inventory_path'])
    if inventory['missing']:raise ValueError('Full parent archives required')
    probe=output/'probe-parent';probe.mkdir();checkpoint=None
    for item in inventory['verified']:
        if time.time()>deadline-20:raise TimeoutError('CPU readiness bound reached')
        required=item['required'];src=item['local_archive_source'];payload=verify_archive(root/src['archive_root'],src['ack'],collect=True)
        if hashlib.sha256(payload).hexdigest()!=required['sha256']:raise ValueError('Original parent reconstruction differs')
        target=probe/required['object'];target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as stream:stream.write(payload);stream.flush();os.fsync(stream.fileno())
        if required['object'].startswith('checkpoint-commits/objects/cp-'):checkpoint=target
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    training=normalize_fixed_file_paths(copy.deepcopy(read_json(root/config['training_template'])))
    model,features,_=load_full_actor(root/training['bundle_path'],training['parent_model_sha256'],'cpu')
    model.load_state_dict(saved['model'],strict=True)
    if len(saved['model'])!=127 or len(saved['optimizer']['state'])!=97:raise ValueError('Incomplete full parent')
    league=read_json(root/config['parent_league_path']);atomic_json(probe/'league.json',league)
    deck=list(map(int,(root/training['bundle_path']/'deck.csv').read_text(encoding='utf-8-sig').split()))
    constructed=League(root,probe,training['league'],model,features,deck,opponent_template=model)
    constructed.freeze(5)
    # This local audit deliberately snapshots code only. Read the already
    # hash-protected original agent assets instead of an absent snapshot copy.
    from . import public_agent
    previous_root=public_agent.ROOT;public_agent.ROOT=root
    try:opponent,selection=constructed.select(0,231199);opponent.close()
    finally:public_agent.ROOT=previous_root
    if selection['opponent_id']!='ppo' or not selection['fixed'] or len(constructed.entries)!=6:
        raise ValueError('Wrong fixed opponent or inherited history')
    atomic_json(output/'metrics.json',{'at':utcnow(),'complete':True,'engineering_only':True,
        'full_CPU_League_and_PublicRuleAgent_constructed':True,'frozen_history_members':6,
        'fixed_opponent_files_verified':len(training['league']['fixed_opponents'][0]['files']),
        'original_checkpoint_sha256':sha256(checkpoint),'model_tensors':127,'optimizer_states_preserved':97,
        'selected_legal_fixed_opponent':selection,'seconds':time.monotonic()-started,
        'GPU_used':False,'new_games':0,'optimizer_steps_executed':0,'new_unique_PPO_trained_decisions':0,
        'source_parent_and_archives_unchanged':True,'promotion_eligible':False})
