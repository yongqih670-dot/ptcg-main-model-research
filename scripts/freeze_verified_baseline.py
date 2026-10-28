"""Freeze the strongest verified baseline after independent development evidence.

The final pool is versioned before any outcomes are observed. Strengthen its PPO
seat mapping consistently; never adjust weights or select variants by results.
"""
import argparse
import copy
import shutil
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ptcg_lab.io import read_json,atomic_json,sha256,utcnow,read_series
from scripts.freeze_opponent_pool import identity
from scripts.validate_confirmation import validate


def freeze(run):
    state_path=ROOT/'research/autonomy/state.json';state=read_json(state_path)
    if state.get('active_job'):raise RuntimeError('Wait for active job and budget reconciliation')
    if state.get('release_anchor_run_id'):raise RuntimeError('Release anchor already frozen')
    result=validate(run)
    if not result['development_confirmation_passed']:raise RuntimeError('Independent confirmation not passed')
    config=read_json(run/'config.json')
    candidate_specs=[c['candidate'] for c in config['matchups'] if c['cell_id'].startswith('zero-vs-')]
    if len(candidate_specs)!=4 or any(s!=candidate_specs[0] for s in candidate_specs):
        raise RuntimeError('Unexpected hypothesis or candidate identity')
    # Every independently implemented rule must be beaten directly, not merely
    # a pooled mean dominated by a single favorable opponent.
    panel=read_json(run/'panel-analysis.json')
    for name in ('lucario','alakazam','archaludon'):
        if panel['cells']['zero-vs-'+name]['score_with_errors_counted_as_losses']<=.5:
            raise RuntimeError('PPO not established as strongest of tested baseline families')
    # No final variants may have been used before this prospective versioning.
    for cfg in (ROOT/'runs').glob('*/config.json'):
        c=read_json(cfg)
        if any('/holdout/' in str(x.get(side,{})).replace('\\','/')
               for x in c.get('matchups',[c]) for side in ('candidate','reference')):
            raise RuntimeError('Final-pool exposure found; cannot silently refreeze')
    out=ROOT/'research/opponents'
    if (out/'final_holdout_v2.json').exists():raise RuntimeError('Final v2 already exists')
    target=ROOT/'agents/frozen/ppo-zero-anchor'
    source=run/'source'/candidate_specs[0]['path']
    shutil.copytree(source,target,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    spec={**candidate_specs[0],'id':'release-anchor-ppo-zero','path':'agents/frozen/ppo-zero-anchor'}
    at=utcnow()
    manifest={'created_at':at,'role':'release_anchor_and_initial_champion','run_id':run.name,
        'reason':'Strongest verified baseline among the four available families; independent feature confirmation and direct wins over all three rule baselines',
        'spec':spec,'files':identity(target),
        'adapter_snapshot':str((run/'source/ptcg_lab/public_agent.py').relative_to(ROOT)),
        'adapter_sha256':sha256(run/'source/ptcg_lab/public_agent.py'),
        'statistics':result['contrast'],'final_score_not_measured':True,
        'training_seed_check':'Not applicable: no stochastic training; independent fresh BO3 batch used',
        'limits':'Strongest among tested public baselines, not a claim about the Kaggle ladder. Final improvement must be measured against this corrected baseline, not the weaker physical-seat version.'}
    atomic_json(out/'release_anchor.json',manifest)
    for role in ('development','final_holdout'):
        previous=out/f'{role}_v1.json';pool=copy.deepcopy(read_json(previous))
        pool.update(version=2,created_at=at,previous_manifest_sha256=sha256(previous),
            version_reason='Prospective PPO ACTOR_SEAT correction after independent development confirmation; no final-pool BO3 outcomes observed; same decks and weights')
        for opponent in pool['opponents']:
            if opponent['lineage']=='dipam_unown_gradiant_ppo':
                opponent['parameters']={'_actor_seat_mode':'zero'}
        atomic_json(out/f'{role}_v2.json',pool)
    state.update(updated_at=at,current_phase='P3',champion_run_id=run.name,release_anchor_run_id=run.name,
        champion_manifest='research/opponents/release_anchor.json',opponent_pool_manifest='research/opponents/development_v2.json',
        final_holdout_manifest='research/opponents/final_holdout_v2.json',consecutive_valid_runs_without_promotion=0,
        next_action='P3: finish public replay compatibility audit, collect actor-only local teacher data, build CatBoost then compact Transformer BC with grouped validation and fresh BO3 tests')
    state['phase_evidence']['P2'].update(status='passed_baseline_frozen',public_rule_lineages_runtime_verified=4,
        archetypes_runtime_verified=4,ppo_bo3='20261023T110316Z_ppo-bo3-smoke_b81e8b',
        parallel_recovery='20261023T110736Z_parallel-bo3-smoke_a6dd95',seat_feature_screen=config['screening_run_id'],
        independent_confirmation=run.name,champion='research/opponents/release_anchor.json',tests_passed=16)
    state['unresolved']=[x for x in state['unresolved'] if not x.startswith('Three public rule origins') and not x.startswith('H002 seat feature')]
    atomic_json(state_path,state)
    print('Frozen corrected PPO as initial champion and release anchor; prospective opponent manifests v2; no submission qualification.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();freeze(a.run.resolve())
