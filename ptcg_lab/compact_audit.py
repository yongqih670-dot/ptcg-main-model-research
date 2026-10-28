"""Controlled local archive preparation using real, immutable TR04 evidence."""
import argparse
import os
from pathlib import Path
import time
from .archive_prefix import build_prefix,restored_statistics,load_prefix
from .io import atomic_json,read_json,sha256,utcnow


def run(config,output):
    if config.get('audit_variant')=='active_clock_acceptance':
        from .active_clock_acceptance import run as audit
        return audit(config,output)
    if config.get('audit_variant')=='volume_interrupted_full_state':
        from .volume_interrupted_audit import run as audit
        return audit(config,output)
    if config.get('audit_variant')=='volume_results':
        from .volume_results import run as audit
        return audit(config,output)
    if config.get('audit_variant')=='volume_source_audit':
        from .volume_source_audit import run as audit
        return audit(config,output)
    if config.get('audit_variant')=='continuous_result_audit':
        from .continuous_result_audit import run as audit
        return audit(config,output)
    if config.get('audit_variant')=='continuous_league_readiness':
        from .continuous_league_readiness import run as audit
        return audit(config,output)
    if config.get('audit_variant')=='continuous_archive_inventory':
        from .continuous_archive_inventory import run as audit
        return audit(config,output)
    if config.get('audit_variant')=='continuous_parent_readiness':
        from .continuous_parent import run as audit
        return audit(config,output)
    if config.get('audit_variant')=='actual_unique_PPO_volume':
        from .ppo_volume_audit import run as audit
        return audit(config,output)
    if config.get('audit_variant')=='tr10_clean_checkpoint_recovery':
        from .tr10_recovery_analysis import run as audit
        return audit(config,output)
    if config.get('audit_variant')=='completed_hybrid_credit_analysis':
        from .hybrid_credit_analysis import run as analyze
        return analyze(config,output)
    if config.get('audit_variant')=='completed_hybrid_model_materialization':
        from .hybrid_materialization import run as materialize
        return materialize(config,output)
    if config.get('audit_variant')=='tr09_verified_migration_report':
        from .migration_summary import run as report
        return report(config,output)
    if config.get('audit_variant')=='tr09_clean_checkpoint_recovery':
        from .tr09_recovery_analysis import run as analyze
        return analyze(config,output)
    if config.get('audit_variant')=='tr08_failed_training_analysis':
        from .tr08_failure_analysis import run as analyze
        return analyze(config,output)
    if config.get('audit_variant')=='tr08_recovery_analysis':
        from .tr08_recovery_analysis import run as analyze
        return analyze(config,output)
    if config.get('audit_variant')=='tr07_failed_training_analysis':
        from .tr07_failure_analysis import run as analyze
        return analyze(config,output)
    if config.get('audit_variant')=='windows_archive_paths':
        from .windows_archive_check import run as check
        return check(config,output)
    if config.get('audit_variant')=='tr05_failed_training_analysis':
        from .tr05_failure_analysis import run as analyze
        return analyze(config,output)
    if config.get('audit_variant')=='failed_training_analysis':
        from .training_failure_analysis import run as analyze
        return analyze(config,output)
    if config.get('audit_variant')=='lossless_restore_capsule':
        from .restore_capsule import prepare
        return prepare(config,output)
    if config.get('audit_variant')=='lossless_history_pack':
        from .history_delta import run as history_run
        return history_run(config,output)
    if config.get('audit_variant')=='bounded_checkpoint':
        from .bounded_checkpoint_audit import run as bounded_run
        return bounded_run(config,output)
    output=Path(output);root=Path(config['project_path']).resolve()
    if os.environ.get('PTCG_JOB_RUN_ID')!=output.name:raise RuntimeError('Compact audit requires controlled runner')
    deadline=float(os.environ['PTCG_JOB_DEADLINE_UNIX'])
    prefix=build_prefix(root,root/'runs'/config['origin_run_id'],root/config['archive_inventory_path'],
                        config['archive_inventory_sha256'],config['origin_checkpoint_sha256'])
    if time.time()>deadline-30:raise TimeoutError('Compact audit deadline')
    path=output/'archive-prefix.json';atomic_json(path,prefix);digest=sha256(path);prefix=load_prefix(path,digest)
    origin=root/'runs'/config['origin_run_id']
    series_paths=[origin/'series'/f"{r['series_id']:06d}.json" for r in prefix['tail']]
    actual=restored_statistics(origin,prefix,live_series_paths=series_paths)
    if actual!=read_json(origin/'league.json')['fixed_training_results']:raise ValueError('Real-data restoration statistics differ')
    import torch
    decisions=0
    for row in prefix['tail']:
        if time.time()>deadline-20:raise TimeoutError('Compact tail audit deadline')
        data=torch.load(origin/row['trajectory_path'],map_location='cpu',weights_only=False)
        if data['series_id']!=row['series_id'] or data['policy_version']!=prefix['checkpoint']['version']:
            raise ValueError('Live PPO input identity mismatch')
        decisions+=len(data['decisions'])
    progress=read_json(origin/'progress.json')
    if decisions!=progress['pending_decisions']:raise ValueError('Pending update input length differs')
    atomic_json(output/'metrics.json',{'at':utcnow(),'complete':True,'engineering_only':True,
                'origin_run_id':config['origin_run_id'],'archive_prefix_sha256':digest,
                'consumed_records':len(prefix['consumed']),'tail_records':len(prefix['tail']),
                'tail_decisions':decisions,'statistics_match_original_league':True,
                'checkpoint_cursor':prefix['checkpoint'],'prefix_bytes':path.stat().st_size,
                'linux_restore_verified':False,'ppo_update_parity_verified':False,
                'streaming_training_integrated':False,'source_run_modified':False,'promotion_eligible':False})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(read_json(a.config),a.output)
