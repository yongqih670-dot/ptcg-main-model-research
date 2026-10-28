"""One CPU entrypoint wrapper repair inside the original total900s scope."""
from pathlib import Path
import copy,time
from .io import read_json
from .remote_lease import validate_no_unreconciled_remote


def validate(project,config,deadline=None):
    parents=config.get('CPU_sampler_repair_parent_run_ids',[])
    wall=config['check_wall_seconds']
    if not parents:
        if wall!=900:raise ValueError('Initial CPU acceptance has exactly900s scope')
        return
    if parents!=['20261025T232927Z_tr07-CPU-sampler-acceptance_9c3c7c'] or wall!=891:
        raise ValueError('Only the exact unstarted CPU wrapper can use the original remaining891s')
    parent=Path(project)/'runs'/parents[0];attempts=read_json(parent/'attempts.json')
    if len(attempts)!=1 or attempts[0]['status']!='failed' or abs(attempts[0]['seconds']-config['prior_CPU_sampler_seconds'])>.001:
        raise ValueError('Original CPU wrapper failure or expenditure differs')
    if not 0<attempts[0]['seconds']<10 or attempts[0]['seconds']+wall>900:
        raise ValueError('CPU repair cannot regain spent scope')
    validate_no_unreconciled_remote(project,{'remote_job':read_json(parent/'remote-job.json')})
    log=(parent/'remote-final-logs/stderr.log').read_text(encoding='utf-8')
    if "UnboundLocalError: cannot access local variable 'admit'" not in log or 'initial_capacity=admit(root,0)' not in log:
        raise ValueError('A different or potentially started payload cannot use this wrapper repair')
    observed=read_json(parent/'server-final-observation.json')
    if observed.get('progress.json') or observed.get('trial-metrics.json') or observed.get('failure.json'):
        raise ValueError('CPU workload might already have started')
    old=read_json(parent/'config.json');same=copy.deepcopy(config)
    for key in ['CPU_sampler_repair_parent_run_ids','prior_CPU_sampler_seconds']:same.pop(key,None)
    same['check_wall_seconds']=900
    if same!=old:raise ValueError('CPU wrapper repair changed its workload or retirement plan')
    if deadline is not None and attempts[0]['seconds']+deadline-time.time()>900.5:
        raise ValueError('Actual CPU repair deadline exceeds the original remaining scope')
