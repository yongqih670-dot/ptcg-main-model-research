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
        env.run([player(0),player(1)])
        emit({'event':'series_end','statuses':[p.status for p in env.state],
              'rewards':[p.reward for p in env.state],'scores':getattr(env,'result',None)})


if __name__=='__main__':main()
