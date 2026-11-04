"""Fail-closed v3 activation and prospective hypothesis/run limits.

Historical counters are never reset. Preregistrations bind permitted jobs to
exact configuration hashes; --engineering/--final cannot change their role.
"""
from pathlib import Path
from .io import read_json, sha256, utcnow


def is_v3(policy):
    return policy.get('stopping', {}).get('mode') == 'bounded_hypothesis_stage'


def safe_path(root, relative):
    path = (Path(root) / relative).resolve()
    if Path(root).resolve() not in path.parents:
        raise RuntimeError('Contract path must stay inside project')
    return path


def activation_protected_hashes(root,receipt_path):
    """Follow hash-bound amendment receipts back to the original protections."""
    seen=set();cycle=None
    while True:
        path=safe_path(root,receipt_path)
        if path in seen:raise RuntimeError('Activation receipt cycle')
        seen.add(path);receipt=read_json(path)
        if receipt.get('status')!='complete':raise RuntimeError('Incomplete activation receipt')
        if cycle is None:cycle=receipt.get('cycle_id')
        if receipt.get('cycle_id')!=cycle:raise RuntimeError('Activation amendment changed research cycle')
        if 'before_sha256' in receipt:return receipt['before_sha256']
        parent=receipt.get('parent_activation_receipt');expected=receipt.get('parent_activation_receipt_sha256')
        if not parent or not expected or sha256(safe_path(root,parent))!=expected:
            raise RuntimeError('Activation amendment parent changed or missing')
        receipt_path=parent


def validate_activation(root, policy, state):
    if not is_v3(policy):
        return
    stage = state.get('research_stage', {})
    if policy.get('status') != 'active' or not policy.get('activation', {}).get('executable'):
        raise RuntimeError('v3 proposal is not active')
    if stage.get('status') != 'active':
        raise RuntimeError('v3 stage is not active')
    receipt = read_json(safe_path(root, stage['activation_receipt']))
    if (receipt.get('status') != 'complete' or receipt.get('cycle_id') != stage.get('cycle_id')
            or receipt.get('policy_sha256') != sha256(Path(root)/'research/autonomy/policy.json')):
        raise RuntimeError('Activation transaction is incomplete or policy changed')
    if state['budget_used']['executed_research_experiments'] < stage['baseline_raw_runs']:
        raise RuntimeError('Historical budget counter decreased')


def validate_stage_job(root, config, policy, state, research, final, *, new=True, run_id=None):
    if not is_v3(policy):
        return
    validate_activation(root, policy, state)
    from .volume_scope import validate_raws
    validate_raws(root,config,policy,state,research,new)
    stop = policy['stopping']; stage = state['research_stage']
    ledger = stage['hypotheses'] if research else stage['engineering_tasks']
    identity = config.get('hypothesis_id')
    if identity not in ledger:
        raise RuntimeError('Job requires a registered stage hypothesis/task')
    row = ledger[identity]
    if row.get('status') not in ('registered', 'started'):
        raise RuntimeError('Hypothesis/task is closed')
    plan_path = safe_path(root, row['preregistration_path'])
    if sha256(plan_path) != row['preregistration_sha256']:
        raise RuntimeError('Preregistration changed')
    plan = read_json(plan_path)
    if plan['hypothesis_id'] != identity or bool(plan['research']) != research or bool(plan['final']) != final:
        raise RuntimeError('Preregistered research/final role mismatch')
    if bool(plan.get('diagnostic')) != bool(row.get('diagnostic')):
        raise RuntimeError('Preregistered diagnostic role mismatch')
    # Hash normalized config data, not source path or user-provided hash fields.
    import hashlib, json
    config_hash = hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if config_hash not in plan['config_hashes']:
        raise RuntimeError('Job config is not preregistered')
    runs = row.get('run_ids', [])
    if not new and run_id not in runs:
        raise RuntimeError('Resume run is outside current stage contract')
    if new and len(runs) >= plan['max_raw_runs']:
        raise RuntimeError('Hypothesis/task raw run cap exhausted')
    if not research:
        if final or config.get('job_kind') not in plan['allowed_job_kinds']:
            raise RuntimeError('Engineering task cannot bypass research/final gates')
        return
    used = state['budget_used']['executed_research_experiments'] - stage['baseline_raw_runs']
    if new and (used >= stop['new_raw_research_runs_max'] or
                state['budget_used']['executed_research_experiments'] >= policy['budget']['executed_research_experiments_max']):
        raise RuntimeError('Stage/global research run cap exhausted')
    if new and not final and stage['regular_raw_runs'] >= stop['new_raw_research_runs_regular_max']:
        raise RuntimeError('Final research run reserve is protected')
    if final:
        qualification = plan.get('development_qualification')
        if not qualification:
            raise RuntimeError('Final work requires hashed development qualification')
        path = safe_path(root, qualification['path'])
        if sha256(path) != qualification['sha256']:
            raise RuntimeError('Development qualification changed')
        evidence = read_json(path)
        if not evidence.get('promotion_eligible') or evidence.get('classification') != 'improved':
            raise RuntimeError('Candidate is not independently development-qualified')
    started = [h for h in stage['hypotheses'].values() if h.get('run_ids')]
    if not runs and len(started) >= stop['new_hypotheses_max']:
        raise RuntimeError('New hypothesis allowance exhausted')
    others = [h for key,h in stage['hypotheses'].items() if key != identity and h.get('status') == 'started']
    if others:
        raise RuntimeError('Close and analyze current hypothesis before starting another')
    if not final and stage['consecutive_valid_hypotheses_without_promotion'] >= stop['consecutive_valid_hypotheses_without_promotion']:
        if not plan.get('diagnostic'):
            raise RuntimeError('Plateau: only preregistered diagnostic hypotheses remain')
    if plan.get('diagnostic') and not runs and stage['diagnostic_hypotheses_started'] >= stop['additional_diagnostic_hypotheses_max_total']:
        raise RuntimeError('Diagnostic hypothesis allowance exhausted')


def register_stage_run(config, policy, state, research, final, run_id):
    if not is_v3(policy):
        return
    stage = state['research_stage']
    row = (stage['hypotheses'] if research else stage['engineering_tasks'])[config['hypothesis_id']]
    first = not row['run_ids']
    row['run_ids'].append(run_id); row['status'] = 'started'
    from .volume_scope import register
    register(config,state,research,run_id)
    if research:
        stage['final_raw_runs' if final else 'regular_raw_runs'] += 1
        if first and row.get('diagnostic'):
            stage['diagnostic_hypotheses_started'] += 1


def close_hypothesis(root,policy,state,identity,evaluation_run_id,confirmation_path=None):
    """Close one complete preregistered hypothesis, never increment per raw job.

    Caller must recompute panel-analysis using analyze_panel before this call.
    Promotion needs independent confirmation; a positive screen is insufficient.
    """
    validate_activation(root,policy,state)
    if state.get('active_job'):raise RuntimeError('Cannot close a running hypothesis')
    stage=state['research_stage'];row=stage['hypotheses'][identity]
    if row.get('status')!='started':raise RuntimeError('Hypothesis must be started and not closed')
    if evaluation_run_id not in row['run_ids']:raise RuntimeError('Evaluation is outside hypothesis')
    run=safe_path(root,'runs/'+evaluation_run_id);cfg=read_json(run/'config.json')
    panel=read_json(run/'panel-analysis.json');metrics=read_json(run/'metrics.json')
    if cfg.get('job_kind','evaluation')!='evaluation' or cfg.get('hypothesis_id')!=identity:
        raise RuntimeError('Completed hypothesis requires its planned BO3 evaluation')
    if not panel.get('complete') or not metrics.get('complete'):
        raise RuntimeError('Incomplete evaluation cannot count as completed valid hypothesis')
    classification=panel.get('classification','inconclusive')
    valid=classification in ('improved','regressed','inconclusive')
    promoted=False;evidence=None
    if confirmation_path:
        path=safe_path(root,confirmation_path);evidence=read_json(path)
        if evidence['run_id']!=evaluation_run_id:raise RuntimeError('Confirmation belongs to another run')
        promoted=(valid and classification=='improved' and evidence.get('development_confirmation_passed') is True
                  and all(evidence.get('checks',{}).values()) and bool(evidence.get('checks')))
    if classification=='improved' and not promoted:classification='inconclusive'
    row.update(status='completed' if valid else 'invalid',classification=classification,promotion_eligible=promoted,
        completed_at=utcnow(),evaluation_run_id=evaluation_run_id,
        evidence_sha256={'panel':sha256(run/'panel-analysis.json'),'metrics':sha256(run/'metrics.json')})
    if confirmation_path:row['evidence_sha256']['confirmation']=sha256(safe_path(root,confirmation_path))
    if valid:
        stage['completed_valid_hypotheses']+=1
        stage['consecutive_valid_hypotheses_without_promotion']=0 if promoted else stage['consecutive_valid_hypotheses_without_promotion']+1
    if row.get('diagnostic'):stage['diagnostic_hypotheses_completed']+=1
    stage['events'].append({'event':'hypothesis_closed','at':utcnow(),'hypothesis_id':identity,
        'classification':classification,'valid':valid,'promoted':promoted,'evaluation_run_id':evaluation_run_id})
    return row
