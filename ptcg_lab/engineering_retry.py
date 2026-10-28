"""Only bounded pre-initialization connection retries; no science recovery grant."""
from datetime import datetime
from pathlib import Path
from .io import read_json


def admit_connection_resume(root,run,config,attempts,state):
    root=Path(root);run=Path(run)
    if config.get('job_kind')!='server_storage_acceptance' or not config.get('engineering_only'):
        raise RuntimeError('Additional attempt requires its exact existing scientific/engineering route')
    row=state['research_stage']['engineering_tasks'][config['hypothesis_id']]
    plan=read_json(root/row['preregistration_path'])
    if (plan.get('connection_retry_source_run_id')!=run.name
            or plan.get('read_only_preinitialize_connection_attempts_max')!=3 or not 1<=len(attempts)<3):
        raise RuntimeError('Bounded read-only connection retry not registered')
    for name in ['remote-job.json','progress.json','target-acceptance-initialization.json','remote-start.json']:
        if (run/name).exists():raise RuntimeError('Remote initialization may have occurred; a connection resend is forbidden')
    for a in attempts:
        n=a['attempt'];err=(run/f'stderr-{n:02d}.log').read_text(encoding='utf-8')
        if (a['status']!='failed' or not 0<a['seconds']<10 or 'BrokenPipeError' not in err
                or 'self.start_agent(agent_path)' not in err):raise RuntimeError('Different startup failure cannot use connection retries')
    if 'Connection closed by TRAINING_HOST port 15134' not in (run/'ssh-errors.log').read_text(encoding='utf-8'):
        raise RuntimeError('Actual transient transport close not proved')
    original=read_json(run/'experiment.json');deadline=datetime.fromisoformat(original['created_at']).timestamp()+original['requested_wall_seconds']
    return deadline
