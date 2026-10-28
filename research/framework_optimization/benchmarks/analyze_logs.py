"""Read-only historical analysis; gaps remain unknown, not attributed to I/O."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def analyze(run):
    metrics=read(run/'metrics.json'); attempts=read(run/'attempts.json')
    wall=sum(x['seconds'] for x in attempts)
    collect=metrics.get('collect_seconds_this_attempt');update=metrics.get('update_seconds_this_attempt')
    # Metrics timers refer to one attempt. Never subtract these from a multi-
    # attempt total and silently treat the difference as a measured phase.
    residual=wall-collect-update if len(attempts)==1 and collect is not None and update is not None else None
    observations=[];sources=[run/'metrics.json',run/'attempts.json'];barrier_observations={}
    for p in sorted((run/'remote-observations').glob('*.json')):
        v=read(p);sources.append(p)
        for name,entry in v.get('files',{}).items():
            data=entry.get('data',{})
            if isinstance(data,dict) and 'stage' in data:
                observations.append({'observed_at':v['at'],'source_at':data.get('at'),
                    'stage':data['stage'],'source_file':name,
                    'last_checkpoint_seconds':data.get('last_checkpoint_seconds')})
                if data['stage']=='consumed_archive_barrier_passed' and 'total_barrier_seconds' in data:
                    barrier_observations[str(data.get('cursor'))]={k:data.get(k) for k in (
                        'at','cursor','total_barrier_seconds','ACK_wait_seconds','post_ACK_writer_rotations',
                        'last_checkpoint_seconds','consumed_files','consumed_logical_bytes')}
    barriers=[]
    for p in (run/'remote-results').glob('*barrier*.json'):
        barriers.extend(read(p));sources.append(p)
    unique=metrics.get('new_actual_unique_PPO_decisions')
    return {'run_id':run.name,'complete':metrics.get('complete'), 'attempts':len(attempts),
        'job_wall_seconds':wall,'collect_seconds_last_attempt':collect,'update_seconds_last_attempt':update,
        'residual_seconds':residual,'residual_fraction':residual/wall if residual is not None else None,
        'new_actual_unique_PPO_decisions':unique,'new_sampled_decisions':metrics.get('new_sampled_decisions'),
        'new_complete_BO3':metrics.get('new_complete_BO3'),
        'unique_optimized_decisions_per_job_second':unique/wall if unique is not None else None,
        'job_seconds_per_10000_unique_optimized':wall*10000/unique if unique else None,
        'observed_stage_counts':dict(Counter(x['stage'] for x in observations)),
        'point_observations_not_durations':observations,
        'barrier_records':barriers or None,
        'partial_observed_barriers':list(barrier_observations.values()),
        'barrier_observations_are_partial_not_full_phase_totals':True,
        'unknown_phases':['initialization_restore','environment_advance','feature_encoding','IPC',
            'data_pack_H2D','PPO_forward_backward_step','checkpoint_serialization_write_hash',
            'compression_transfer_ACK','shared_GPU_wait','backpressure','finalization'],
        'limitations':['Collect includes environment, actor, opponent and control waits.',
            'Update timer is not CUDA kernel time.',
            'Sparse progress observations do not provide phase durations.',
            'Missing barrier log is not zero barrier time.',
            'No inference of GPU idle time or archive share from residual.'],
        'inputs':[{'path':p.as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sources]}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=analyze(args.run);args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('inputs','point_observations_not_durations','limitations','unknown_phases')},indent=2))
