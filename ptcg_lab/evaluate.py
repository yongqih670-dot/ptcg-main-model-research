"""Official Kaggle CABT v2 BO3, one in-flight series per process.

The official interpreter is used unchanged, including draws and round resets.
Timing here is Python-callable timing, not a submitted Linux container benchmark.
"""
import argparse
import contextlib
import io
import gzip
import json
import math
import multiprocessing
import os
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

from .public_agent import create_agent
from .io import append_series, atomic_json, read_json, read_series, utcnow


def quantile(values, q):
    if not values:
        return None
    v = sorted(values)
    return v[min(len(v) - 1, math.ceil(q * len(v)) - 1)]


def load_official(output, suffix=''):
    diagnostics = io.StringIO()
    with contextlib.redirect_stdout(diagnostics):
        from kaggle_environments import make
    Path(output, f"optional-environment-imports{suffix}.log").write_text(
        diagnostics.getvalue(), encoding="utf-8")
    return make


def play_series(make, config, series_id, output):
    seat = series_id % 2
    specs = [None, None]
    specs[seat], specs[1 - seat] = config["candidate"], config["reference"]
    policies = [create_agent(spec, config.get("deck"), i,
                             config["policy_seed"] + series_id * 2 + i)
                for i, spec in enumerate(specs)]
    names = [p.name for p in policies]
    started = time.perf_counter()
    env = make("cabt", configuration={"version": 2, "bo": 3})
    error = None
    try:
        env.run(policies)
    except Exception:
        error = traceback.format_exc()
    statuses = [s.status for s in env.state]
    rewards = [s.reward for s in env.state]
    for policy in policies:
        policy.close()
    fallback_count = sum(d.get('random_fallback_calls', 0)
                         for p in policies for d in getattr(p, 'diagnostics', []))
    if error or statuses != ["DONE", "DONE"]:
        result = "error"
    else:
        result = "win" if rewards[seat] == 1 else "loss" if rewards[seat] == -1 else "draw"
    # Framework timing includes its Agent wrapper, used separately from raw policy time.
    framework_usage = {}
    initial_budget = 600.0
    for step in env.steps:
        for i, player in enumerate(step):
            obs = player.observation
            if obs.get("current") and "round" in obs.current:
                key = f"{i}:{obs.current['round']}"
                used = initial_budget - obs.remainingOverageTime
                framework_usage[key] = max(framework_usage.get(key, 0.0), used)
    row = {
        "series_id": series_id, "finished_at": utcnow(), "candidate_seat": seat,
        "worker_pid": os.getpid(),
        "cell_id": config.get('cell_id', 'default'),
        "names": names, "result": result, "statuses": statuses, "rewards": rewards,
        "identities": [getattr(p, 'identity', {'id': p.name}) for p in policies],
        "policy_diagnostics": [getattr(p, 'diagnostics', []) for p in policies],
        "internal_random_fallback_count": fallback_count,
        "round_results": getattr(env, "result", None), "steps": len(env.steps),
        "seconds": time.perf_counter() - started,
        "seat_round_calls": [dict(p.round_calls) for p in policies],
        "seat_round_starts": [p.round_starts for p in policies],
        "contexts": [dict(p.contexts) for p in policies],
        "policy_invalid_count": sum(p.invalid for p in policies),
        "policy_latency_p99_seconds": [quantile(p.durations, .99) for p in policies],
        "policy_latency_max_seconds": [max(p.durations, default=0) for p in policies],
        "policy_round_seconds": [dict(p.round_seconds) for p in policies],
        "framework_round_seconds_max_observed": framework_usage,
        "error": error,
    }
    games=[]
    for step in env.steps[0][0].get('visualize', []):
        cur=step['current']
        if cur['result']>=0:
            reasons=[log.get('reason') for log in step.get('logs', []) if log['type']=='Result']
            games.append({'round':len(games)+1,'winner':cur['result'],'reason':reasons[-1] if reasons else None,
                          'turn':cur['turn'],'first_player':cur.get('firstPlayer'),
                          'remaining_prizes':[len(p['prize']) for p in cur['players']]})
    row['games']=games
    if (series_id < config.get("save_replays_first", 2) or result == "error"
            or config.get('save_all_replays', False)):
        replay = Path(output, f"episode-{series_id:05d}.json.gz")
        temporary = replay.with_suffix('.tmp')
        with gzip.open(temporary, 'wt', encoding='utf-8', compresslevel=3) as stream:
            json.dump(env.toJSON(), stream, ensure_ascii=False, separators=(',', ':'))
        temporary.replace(replay)
        atomic_json(Path(output, f"agent-logs-{series_id:05d}.json"), env.logs)
    # Interpreter frees the battle on normal finish; only explicitly free after exception.
    if error:
        from kaggle_environments.envs.cabt.cg.game import battle_finish
        from kaggle_environments.envs.cabt.cg.sim import Battle
        if Battle.battle_ptr:
            battle_finish()
            Battle.battle_ptr = None
    # Completed work is durable even if the parent is interrupted before JSONL commit.
    atomic_json(Path(output, 'results', f'{series_id:05d}.json'), row)
    return row


def summarize(rows):
    counts = Counter(r["result"] for r in rows)
    n = len(rows)
    return {
        "series_count": n, "wins": counts["win"], "draws": counts["draw"],
        "losses": counts["loss"], "errors": counts["error"],
        "score_with_errors_counted_as_losses": (counts["win"] + .5 * counts["draw"]) / n if n else None,
        "seats": dict(Counter(r["candidate_seat"] for r in rows)),
        "series_seconds_total": sum(r["seconds"] for r in rows),
        "invalid_actions": sum(r["policy_invalid_count"] for r in rows),
        "internal_random_fallbacks": sum(r.get('internal_random_fallback_count', 0) for r in rows),
        "timeouts": sum("TIMEOUT" in r["statuses"] for r in rows),
        "round_count_histogram": dict(Counter(sum(r["round_results"]) for r in rows if r["round_results"])),
        "scope": "official Windows BO3 callable harness; local comparison, not final strength or Linux acceptance",
    }


def series_config(config, series_id):
    if 'matchups' not in config:
        return config
    if sum(c['series_count'] for c in config['matchups']) != config['series_count']:
        raise ValueError('Panel cell sizes do not sum to series_count')
    offset = 0
    for cell in config['matchups']:
        if offset <= series_id < offset + cell['series_count']:
            return {**config, **cell}
        offset += cell['series_count']
    raise ValueError('Series outside panel')


def recover_rows(output):
    rows=read_series(output/'series.jsonl',recover_tail=True)
    by_id={r['series_id']:r for r in rows}
    for path in sorted((output/'results').glob('*.json')):
        row=read_json(path); key=row['series_id']
        if key in by_id:
            if row!=by_id[key]: raise ValueError('Staged result differs from committed series')
        else:
            append_series(output/'series.jsonl',row);rows.append(row);by_id[key]=row
    return rows


_WORKER_MAKE=None


def worker_play(config,series_id,output):
    global _WORKER_MAKE
    if _WORKER_MAKE is None: _WORKER_MAKE=load_official(output,f'-worker-{os.getpid()}')
    return play_series(_WORKER_MAKE,series_config(config,series_id),series_id,output)


def run_rows(config, ids, output):
    workers=config.get('workers',1)
    if workers==1:
        make=load_official(output)
        for series_id in ids:
            yield play_series(make,series_config(config,series_id),series_id,output)
        return
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        iterator=iter(ids);pending=set()
        for _ in range(workers):
            key=next(iterator,None)
            if key is not None: pending.add(pool.submit(worker_play,config,key,output))
        try:
            while pending:
                finished,pending=wait(pending,return_when=FIRST_COMPLETED)
                for future in finished:
                    yield future.result()
                    key=next(iterator,None)
                    if key is not None: pending.add(pool.submit(worker_play,config,key,output))
        finally:
            for future in pending: future.cancel()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--stop-after-new-series", type=int)
    args = p.parse_args()
    config = read_json(args.config)
    args.output.mkdir(exist_ok=True, parents=True)
    rows = recover_rows(args.output)
    completed = {r["series_id"] for r in rows}
    if any(i < 0 or i >= config["series_count"] for i in completed):
        raise ValueError("Existing series IDs do not match the frozen config")
    remaining=[i for i in range(config['series_count']) if i not in completed]
    if args.stop_after_new_series: remaining=remaining[:args.stop_after_new_series]
    for row in run_rows(config,remaining,args.output):
        series_id=row['series_id']
        append_series(args.output / "series.jsonl", row)
        rows.append(row)
        atomic_json(args.output / "progress.json", {
            "at": utcnow(), "completed_series": len(rows), "target_series": config["series_count"],
            "last_series_id": series_id})
        if row["result"] == "error":
            atomic_json(args.output / "metrics.json", summarize(rows))
            raise RuntimeError(f"Series {series_id} failed; inspect saved episode/log")
        if config.get('stop_on_internal_fallback') and row['internal_random_fallback_count']:
            atomic_json(args.output / 'metrics.json', summarize(rows))
            raise RuntimeError(f'Series {series_id} used an internal random fallback; inspect before continuing')
    metrics = summarize(rows)
    metrics["complete"] = len(rows) == config["series_count"]
    atomic_json(args.output / "metrics.json", metrics)
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()

