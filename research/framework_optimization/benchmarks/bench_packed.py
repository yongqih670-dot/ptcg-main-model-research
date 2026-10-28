"""CPU replay only: real frozen fixture, no optimizer, no scientific samples."""
import argparse
import functools
import json
import statistics
import time
from common import WORK,check_baseline,real_fixture,digest
import torch
from ptcg_lab.rl_core import evaluate_decisions as baseline
from fo.packed import Packed,evaluate_decisions,evaluate_packed,reduce_outputs
from fo.precision import parity
from fo.timing import Trace
from fo.profile_window import profile_window


def measure(function,repeats):
    values=[]
    for _ in range(repeats):
        begin=time.perf_counter();function();values.append(time.perf_counter()-begin)
    return dict(n=len(values),median_seconds=statistics.median(values),
                minimum_seconds=min(values),maximum_seconds=max(values),samples_seconds=values)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--repeats',type=int,default=7)
    args=parser.parse_args()
    if not 3<=args.repeats<=20:raise ValueError('Lightweight bounded repeats')
    check_baseline();torch.set_num_threads(2);torch.manual_seed(20261028)
    model,features,decisions,ref=real_fixture();stop=features.b.STOP
    begin=time.perf_counter();packed=Packed.build(decisions,stop);pack_seconds=time.perf_counter()-begin
    trace=Trace(enabled=True)
    with torch.no_grad():
        reference=baseline(model,decisions,stop);candidate=evaluate_packed(model,packed,trace=trace)
        comparison=parity(reference,candidate)
        funcs={'baseline_full_forward_reduce':lambda:baseline(model,decisions,stop),
            'candidate_pack_forward_reduce':lambda:evaluate_decisions(model,decisions,stop),
            'candidate_cached_forward_reduce':lambda:evaluate_packed(model,packed)}
        for fn in funcs.values():fn()
        # Rotate order to reduce systematic warm/cache/frequency bias.
        samples={name:[] for name in funcs};names=list(funcs)
        for repeat in range(args.repeats):
            for name in names[repeat%3:]+names[:repeat%3]:
                samples[name].append(measure(funcs[name],1)['median_seconds'])
        timings={k:dict(n=len(v),median_seconds=statistics.median(v),minimum_seconds=min(v),maximum_seconds=max(v),samples_seconds=v) for k,v in samples.items()}
        with profile_window(trace,seconds=30):profiled=measure(funcs[names[0]],3)
    result={'type':'local_CPU_real_fixture_forward_microbenchmark','threads':2,'fixture':ref,
        'parity':comparison,'one_time_pack_seconds':pack_seconds,'timings':timings,
        'profiled_baseline':profiled,'trace_path':'reports/phase_trace.json',
        'new_scientific_decisions':0,'optimizer_steps':0,'end_to_end_speedup':None,
        'limitations':['48 decisions, single real tail fixture; no full BO3 collection or archive measured.',
            'CPU timing only; hardware GPU acceptance pending.','Profiler overhead estimate is noisy at this small sample size.']}
    result['source_sha256']={p.name:digest(p) for p in (WORK/'source/fo').glob('*.py')}
    for name in names[1:]:
        timings[name]['local_speedup_vs_baseline']=timings[names[0]]['median_seconds']/timings[name]['median_seconds']
    (WORK/'reports/packed_benchmark.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    trace.write(WORK/'reports/phase_trace.json')
    print(json.dumps(result,indent=2))
    if not comparison['passed']:raise SystemExit('Parity failed')


if __name__=='__main__':main()
