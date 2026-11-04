"""Official BO3 engine in isolated CPU runtime; public-view JSON IPC only."""
import contextlib
import io
import json
import sys
from .replay_data import actor_observation


def emit(value):
    # Kaggle captures ordinary sys.stdout while invoking a Python agent.
    # The worker's dedicated protocol pipe must bypass that per-agent log.
    print(json.dumps(value,separators=(',',':')),file=sys.__stdout__,flush=True)


def main():
    with contextlib.redirect_stdout(sys.stderr):
        from kaggle_environments import make
    emit({'event':'ready'})
    for line in sys.stdin:
        request=json.loads(line)
        if request['command']=='close':break
        if request['command']!='bo3':raise ValueError('Unknown IPC command')
        def player(seat):
            def act(obs,configuration=None):
                if obs.get('select') is None:return request['decks'][seat]
                emit({'event':'decision','seat':seat,'observation':actor_observation(obs)})
                return json.loads(sys.stdin.readline())['action']
            return act
        env=make('cabt',configuration={'version':2,'bo':3})
        diagnostics={'maintenance_seconds':0.,'maintenance_windows':0}
        if request.get('maintenance_protocol')=='step_boundary_v1':
            from .engine_maintenance import install_step_windows,MaintenanceClosed
            try:
                install_step_windows(env,emit,lambda:json.loads(sys.stdin.readline()),diagnostics)
                env.run([player(0),player(1)])
            except MaintenanceClosed:break
        else:env.run([player(0),player(1)])
        durations=[[],[]]
        for row in env.logs:
            for seat,log in enumerate(row):
                if isinstance(log,dict) and 'duration' in log:durations[seat].append(log['duration'])
        diagnostics.update(configuration={k:env.configuration[k] for k in ['actTimeout','runTimeout','bo','version']},
            maximum_action_seconds=[max(row,default=0.) for row in durations],
            total_action_seconds=[sum(row) for row in durations])
        emit({'event':'series_end','statuses':[p.status for p in env.state],
              'rewards':[p.reward for p in env.state],'scores':getattr(env,'result',None),'timing_diagnostics':diagnostics})


if __name__=='__main__':
    import os
    if os.environ.get('PTCG_TR15_V3_ACTIVE')=='1':
        from .tr15_framework import observe
        from .framework_v3_exit import ObservedCall
        with ObservedCall(observe('engine',os.environ['PTCG_TR15_JOB_DIRECTORY'])):main()
    else:main()
