"""Sole-controller registered CPU engineering scope; old runs stay unchanged.

Install as ptcg_lab/framework_v2_scope.py only with the generated route patch.
The engineering candidate itself cannot create the required state/receipt.
"""
import hashlib
import json
import math
from pathlib import Path

def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def _inside(root,name):
    root=Path(root).resolve();relative=Path(name)
    if relative.is_absolute() or '..' in relative.parts:raise ValueError('Absolute/traversing scope input')
    p=root
    for part in relative.parts:
        p=p/part
        if p.is_symlink() or getattr(p,'is_junction',lambda:False)():raise ValueError('Linked scope input')
    if not p.resolve().is_relative_to(root):raise ValueError('Scope input escaped project')
    return p

def admit(root,config,state):
    spec=config.get('framework_v2_acceptance')
    if spec is None:return False
    if config.get('job_kind')!='remote_training_check' or config.get('check_variant')!='journal_io':raise ValueError('V2 route is CPU checkpoint engineering only')
    if config.get('hypothesis_id') or config.get('TR11_attempt_authorization') or config.get('active_clock'):raise ValueError('Independent engineering cannot borrow a scientific phase')
    source=config.get('existing_restore_source',{}).get('checkpoint_reference',{})
    if spec.get('kind')=='checkpoint_difference' and (spec.get('full_parent_sha256'),spec.get('full_parent_bytes'))!=(source.get('sha256'),source.get('bytes')):raise ValueError('Full parent differs from the accepted original recovery source')
    if spec.get('engineering_only') is not True or spec.get('kind') not in ('checkpoint_difference','fixed_replay_comparison') or spec.get('write_read_MiB_per_second')!=4:raise ValueError('Unreviewed engineering kind/rate')
    registry=state.get('engineering_scopes',{}).get('framework_efficiency_v2')
    if not registry or registry.get('active') is not True:raise ValueError('Sole controller has not activated v2 engineering scope')
    if registry.get('user_stop') or registry.get('completed') or state.get('current_user_pause') or (Path(root)/'research/autonomy/STOP').exists():raise ValueError('User stop/pause or engineering goal completed')
    path=_inside(root,registry['receipt_path'])
    if _sha(path)!=registry['receipt_sha256']:raise ValueError('Registered engineering receipt changed')
    receipt=_read(path)
    if receipt.get('engineering_only') is not True or receipt.get('TR12_resume') is not False or receipt.get('new_scientific_training') is not False:raise ValueError('Scope would reopen scientific training')
    if type(spec.get('engineering_window_seconds')) is not int or spec['engineering_window_seconds']<=100 or receipt.get('engineering_window_seconds')!=spec['engineering_window_seconds']:
        raise ValueError('Controller engineering measurement window changed')
    if spec['kind']=='fixed_replay_comparison':
        proof=receipt['CPU_acceptance_evidence'];p=_inside(root,proof['path'])
        if _sha(p)!=proof['sha256']:raise ValueError('Passed CPU gate changed')
        accepted=_read(p)
        if accepted.get('full_state_exact') is not True or accepted.get('actual_exit_code')!=0 or accepted.get('pending_decisions')!=104:
            raise ValueError('Complete real CPU recovery gate absent')
        if spec.get('case') not in ('baseline4','candidate4') or spec.get('repeat') not in (0,1):raise ValueError('Unregistered performance case')
        if config.get('no_training_updates') is not False or config.get('new_scientific_training') is not False or config.get('disposable_engineering_replay') is not True:
            raise ValueError('Replay optimizer updates must be honestly classified as engineering')
    if receipt.get('candidate_manifest_sha256')!=spec.get('candidate_manifest_sha256'):raise ValueError('Candidate differs from controller registration')
    manifest_path=_inside(root,spec['candidate_manifest_path'])
    if _sha(manifest_path)!=spec['candidate_manifest_sha256']:raise ValueError('Candidate delivery changed')
    manifest=_read(manifest_path)
    targets=manifest.get('integration_targets',{})
    expected={'scripts/run_experiment.py','ptcg_lab/server_trial.py','ptcg_lab/server_trial_agent.py','ptcg_lab/server_trial_worker.py','ptcg_lab/shared_server_safety.py','ptcg_lab/framework_v2_scope.py'}
    if set(targets)!=expected:raise ValueError('Complete integration target hashes missing')
    for name,digest in targets.items():
        if _sha(_inside(root,name))!=digest:raise ValueError('Installed route differs from reviewed patch')
    for name,ref in manifest['source_files'].items():
        if _sha(_inside(root,name))!=ref:raise ValueError('Registered source changed: '+name)
    if receipt.get('workload_sha256')!=hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':')).encode()).hexdigest():raise ValueError('Exact finite engineering workload changed')
    return receipt

def validate_stage(root,config,state,research,final,new=True,run_id=None):
    receipt=admit(root,config,state)
    if not receipt:return False
    if research or final:raise ValueError('Engineering scope cannot claim scientific or final-evaluation volume')
    if not new and run_id:
        path=Path(root)/'runs'/run_id/'attempts.json'
        attempts=_read(path) if path.exists() else []
        if attempts and attempts[-1]['exit_code']:
            mitigation=state['engineering_scopes']['framework_efficiency_v2'].get('mitigations',{}).get(run_id)
            if not mitigation or mitigation['prior_attempt']!=attempts[-1]['attempt']:raise ValueError('Failure diagnosis/mitigation must precede retry')
            if _sha(_inside(root,mitigation['path']))!=mitigation['sha256']:raise ValueError('Mitigation evidence changed')
    return True

def record_registration(root,config,state,research,final,run_id):
    if not validate_stage(root,config,state,research,final,True,run_id):return False
    row=state['engineering_scopes']['framework_efficiency_v2'];runs=row.setdefault('run_ids',[])
    if run_id not in runs:runs.append(run_id)
    return True

def seconds(root,config,state,requested):
    if not admit(root,config,state):return None
    if not isinstance(requested,(int,float)) or not math.isfinite(requested) or requested<=0:raise ValueError('Finite positive engineering measurement window required')
    # No historical cumulative-hour, phase-hour, raw or attempt ceiling is
    # subtracted. This finite measurement endpoint still bounds this payload.
    if requested>config['framework_v2_acceptance']['engineering_window_seconds']:raise ValueError('Requested window exceeds bound workload receipt')
    return float(requested)


def upload_inputs(root,config,state):
    receipt=admit(root,config,state);spec=config['framework_v2_acceptance']
    if not receipt or spec['kind']!='fixed_replay_comparison':raise ValueError('Exact replay admission absent')
    if spec.get('input_source_job'):return {},[]
    p=_inside(root,spec['upload_list_path'])
    if _sha(p)!=spec['upload_list_sha256']:raise ValueError('Pinned missing-only upload list changed')
    sources={};mapping=[]
    for row in _read(p):
        path=_inside(root,row['local_path'])
        if path.stat().st_size!=row['bytes'] or _sha(path)!=row['sha256']:raise ValueError('Recovered raw input differs')
        sources[row['sha256']]=path
        mapping.append({k:row[k] for k in ['path','sha256','bytes']})
    return sources,mapping
