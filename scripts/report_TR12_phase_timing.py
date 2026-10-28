"""Summarize one terminal TR12 segment without adding overlapping process clocks."""
import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ptcg_lab.io import read_json, atomic_json, sha256, utcnow
from ptcg_lab.remote_lease import validate_no_unreconciled_remote


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    started = time.monotonic()
    sp = ROOT / 'research/autonomy/state.json'
    state = read_json(sp)
    assert not state.get('active_run_id') and not state.get('active_job')
    assert not (ROOT / 'research/autonomy/job.lock').exists()
    run = ROOT / 'runs' / args.run
    assert run.resolve().parent == ROOT / 'runs'
    cfg = read_json(run / 'config.json')
    assert cfg['check_variant'] == 'volume_extension' and cfg.get('phase_timing')
    lease = read_json(run / 'remote-job.json')
    validate_no_unreconciled_remote(ROOT, {'remote_job': lease})
    attempts = read_json(run / 'attempts.json')
    assert len(attempts) == 1 and attempts[0]['status'] in ('completed', 'checkpointed', 'failed', 'timeout')
    m = read_json(run / 'metrics.json')
    wall = attempts[0]['seconds']
    assert wall > 0
    source = read_json(ROOT / cfg['source_state_audit']['path'])['actual_unique_PPO_decisions']
    final = m.get('unique_PPO_trained_decisions', m.get('last_training_progress', {}).get('unique_PPO_trained_decisions'))
    known_completed_clock = attempts[0]['exit_code'] == 0 and attempts[0]['status'] in ('completed', 'checkpointed')
    increment = None if final is None else final - source
    assert increment is None or increment >= 0
    processes = {}
    for lane, name in (('train', 'remote-results/phase-timing-train.json'), ('controller', 'phase-timing-controller.json')):
        path = run / name
        if not path.exists():
            processes[lane] = {'available': False}
            continue
        data = read_json(path)
        assert not data['CUDA_kernel_timing_claimed']
        residual = data['wall_seconds'] - sum(data['exclusive_seconds'].values())
        assert abs(residual) < .001
        processes[lane] = {'available': True, 'evidence': {'path': path.relative_to(ROOT).as_posix(), 'sha256': sha256(path)},
            'process_observed_seconds': data['wall_seconds'], 'final_publication_observed': bool(data['metadata'].get('process_finalizing')),
            'exclusive_partition_residual_seconds': residual,
            'exclusive_seconds_largest_first': dict(sorted(data['exclusive_seconds'].items(), key=lambda x: -x[1])),
            'inclusive_seconds_NONADDITIVE': data['inclusive_seconds_nonadditive'], 'calls': data['calls'],
            'unattributed_other_thread_calls': data['other_thread_calls_unattributed']}
    report = {'at': utcnow(), 'run_id': args.run, 'terminal_proven': True, 'attempt_status': attempts[0]['status'],
        'accounted_job_wall_seconds': wall, 'actual_completed_clock_available': known_completed_clock,
        'source_unique_successful_decisions': source, 'final_observed_unique_successful_decisions': final,
        'new_unique_successful_decisions': increment, 'processes': processes,
        'unique_successful_decisions_per_job_second': increment / wall if known_completed_clock and increment is not None else None,
        'job_seconds_per_10000_unique_successful_decisions': 10000 * wall / increment if known_completed_clock and increment else None,
        'timing_scope_notes': [
            'Only exclusive categories within one process are additive. Train and controller clocks overlap.',
            'PPO timing is host elapsed, not CUDA kernel time. No profiler or CUDA synchronization was added.',
            'Initialization/restore excludes Python imports before timer configuration; outer residual remains unknown.',
            'Checkpoint inclusive spans include nested I/O, hash and validation; never sum them with those categories.',
            'Bounded I/O costs are global across all writers; not every such cost belongs to checkpoints.',
            'Archive receive/compress/verify is one combined measured category; its internal attribution remains unknown.',
            'Snapshot publication omits its own final write duration; original evidence and unknown residual are retained.'
        ], 'optimization_applied_by_report': False, 'promotion_claim': False}
    counted=attempts[0].get('counted_execution_seconds',wall)
    credit=attempts[0].get('verified_shared_wait_seconds',0.)
    if (run/'verified-shared-wait-accounting.json').exists():
        accounting=read_json(run/'verified-shared-wait-accounting.json')
        counted=accounting['counted_execution_seconds'];credit=accounting['verified_shared_wait_seconds']
    report.update(counted_execution_seconds=counted,verified_shared_wait_seconds=credit,
        unique_successful_decisions_per_counted_second=increment/counted if known_completed_clock and increment is not None and counted else None,
        counted_seconds_per_10000_unique_successful_decisions=10000*counted/increment if known_completed_clock and increment else None)
    if processes['controller'].get('available'):
        report['outside_controller_observed_seconds_UNATTRIBUTED'] = wall - processes['controller']['process_observed_seconds']
    if processes['train'].get('available'):
        report['outside_training_process_observed_seconds_NONADDITIVE'] = wall - processes['train']['process_observed_seconds']
    path = run / 'phase-timing-report.json'
    assert not path.exists(), 'Do not overwrite an earlier timing report'
    atomic_json(path, report)
    elapsed = time.monotonic() - started
    state['budget_used']['cpu_job_wall_hours'] += elapsed / 3600
    state.setdefault('engineering_events', []).append({'at': utcnow(), 'event': 'TR12_PHASE_TIMING_REPORT',
        'budget_hypothesis_id': 'TR12', 'CPU_seconds': elapsed,
        'evidence': {'path': path.relative_to(ROOT).as_posix(), 'sha256': sha256(path)}})
    state['updated_at'] = utcnow()
    atomic_json(sp, state)
    print({'report': str(path), 'new_unique': increment, 'wall_seconds': wall,
           'seconds_per_10000': report['job_seconds_per_10000_unique_successful_decisions']})


if __name__ == '__main__':
    main()
