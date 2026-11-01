"""Aggregate already completed checks; never rerun models or rewrite failures."""
import os
from pathlib import Path
import shutil
import time
from .io import read_json,atomic_json,sha256,utcnow


def run(config,output):
    root=Path(config['project_path']).resolve();output=Path(output)
    if output.name!=os.environ.get('PTCG_JOB_RUN_ID'):raise RuntimeError('Controlled report required')
    source=root/'runs'/config['core_audit_run_id'];path=source/'analysis.json'
    if sha256(path)!=config['core_analysis_sha256']:raise ValueError('Real completed core analysis changed')
    core=read_json(path);a=read_json(source/'attempts.json')
    if (not core.get('complete') or core.get('classification')!='TR09_original_checkpoint_full_recovery_audited'
            or a[0]['status']!='failed' or 'TypeError' not in (source/'stderr-01.log').read_text(encoding='utf-8')
            or not core.get('all_full_model_optimizer_RNG_checkpoint_payloads_verified')):
        raise ValueError('Require actual completed recovery checks followed by the proved optional volume summary failure')
    fixture=root/core['numeric_reference']['path']
    if sha256(fixture)!=core['numeric_reference']['sha256']:raise ValueError('Actual numeric reference changed')
    prior=sum(x['seconds'] for rid in config['related_local_audit_run_ids'] for x in read_json(root/'runs'/rid/'attempts.json'))
    if prior+float(os.environ['PTCG_JOB_DEADLINE_UNIX'])-time.time()>240.5:raise ValueError('Original local read-only execution budget exceeded')
    # Exact source analysis, not newly executed model work. Original failed
    # attempts remain immutable and explicitly linked in the finished report.
    shutil.copyfile(path,output/'analysis.json')
    from .scale_accounting import build
    ledger=build(root,output,core)
    result={**core,'report_at':utcnow(),'full_recovery_checks_executed_in_run_id':source.name,
        'core_analysis_sha256':sha256(path),'no_reexecution_or_original_status_reinterpretation':True,
        'actual_post_checks_reporting_failure':{'run_id':source.name,'attempts':a,'stderr_sha256':sha256(source/'stderr-01.log')},
        'scale_ledger_sha256':sha256(output/'scale-ledger.json'),'cumulative_unique_training_decisions':ledger['actual_unique_training_decisions']}
    atomic_json(output/'verified-recovery-report.json',result);atomic_json(output/'metrics.json',result)
