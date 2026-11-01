"""Freeze only after correctness, before any four-case performance result."""
import time
from common import *
def main():
    qualification=read(V3/'reports/QUALIFICATION.json')
    if not qualification['passed']:raise ValueError('Full correctness gate first')
    actual_qualification=read(V3/'reports'/(qualification['candidate_label']+'-final.json'))
    if actual_qualification['first_epoch_credit']!=319 or actual_qualification['decisions']!=319 or actual_qualification['complete_BO3']!=2 or actual_qualification['scientific_credit']!=0:raise ValueError('Real qualification counts differ from required full BO3 first-epoch coverage')
    path=V3/'reports/PERFORMANCE_PROTOCOL.json'
    if path.exists() or list((V3/'reports').glob('D-*-perf.json')):raise ValueError('No post-result protocol replacement')
    # Correct the preparation description before freezing; retain old bytes.
    original=V3/'reports/INPUTS.json';old=original.read_bytes();repair=V3/'repairs/input-description-01';repair.mkdir(parents=True,exist_ok=False);(repair/'INPUTS.json').write_bytes(old)
    inputs=read(original);inputs['input_ready_archive_set']='same pre-frozen original source pool; every new case starts with an empty case archive, mints fresh own ACKs during the timed window; no previous case outputs reused';write(original,inputs)
    source={p.relative_to(V3/'candidate').as_posix():sha(p) for p in (V3/'candidate').rglob('*') if p.is_file() and p.suffix in ('.py','.json')}
    manifest=dict(schema=1,scope='framework_efficiency_v3; engineering only; no scientific training',runtime_sources=source,at=time.time());write(V3/'reports/framework-v3-manifest.json',manifest)
    bench={p.relative_to(V3).as_posix():sha(p) for p in (V3/'benchmarks').glob('*.py')}
    snapshot=V3/'source-snapshots/performance-freeze'
    for relative in bench:
        target=snapshot/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((V3/relative).read_bytes())
    baseline=read(V3/'reports/BASELINE.json');actual=read(ROOT/baseline['actual_source_run']/'source-manifest.json');assets={}
    for relative,digest in actual.items():
        relative=relative.replace('\\','/')
        if relative.startswith(inputs['current_bundle']+'/'):
            if sha(ROOT/relative)!=digest:raise ValueError('Actual frozen actor/encoder input changed')
            assets[relative]=digest
    protocol=dict(frozen_at=time.time(),correctness_sha256=sha(V3/'reports/QUALIFICATION.json'),inputs_sha256=sha(original),candidate_manifest_sha256=sha(V3/'reports/framework-v3-manifest.json'),benchmarks=bench,
        external_actor_assets=assets,baseline_runtime_sources=baseline['runtime_sources'],runtime_input_assets=read(V3/'reports/RUNTIME_INPUTS.json'),candidate_switches=read(V3/'reports/SELECTED_PROFILE.json')['switches'],
        order=[dict(label='D-baseline-0',variant='baseline'),dict(label='D-candidate-0',variant='candidate'),dict(label='D-candidate-1',variant='candidate'),dict(label='D-baseline-1',variant='baseline')],
        per_arm_repeats=2,whole_BO3=48,decisions=8203,control_checks=17516,progress_every_engine_decisions=50,expected_collecting_progress_messages=329,
        parent_sha256=inputs['parent']['sha256'],ppo=inputs['ppo'],fixed_order=[r['series_id'] for r in inputs['trajectories']],
        checkpoint_schedule='every8 wholeBO3, once before PPO, first successful optimizer step, twice after committed update; no sample between final2',
        rate_bytes_per_second=4*2**20,CPU_affinity_cores=3,torch_threads=2,system_RAM_reserve_bytes=8*2**30,tree_RSS_max_bytes=8*2**30,project_hard_bytes=120*2**30,physical_free_reserve_bytes=20*2**30,
        CPU_deterministic_algorithms=True,CPU_determinism_reason='Same baseline repeated first backward differed without deterministic gather/index accumulation; both arms use same original PPO with deterministic CPU execution, no dtype/model/AMP change',
        rate_scope='unchanged original paced object reads/writes, canonical tape staging and metadata publication; original full hashes and original gzip receiver code retain their original semantics; no claim of kernel-wide physical I/O cap',
        initial_archive='pre-frozen original input catalog ready, each case archive empty; fresh own fenced full ACKs in window',
        primary_seconds='whole perf Popen process including preparation+replay+all controls+PPO+checkpoint+archive, plus separate two-point cold process, plus verified lossless retention and temporary release',
        acceptance=dict(full_logical_state_exact_across_four_cases=True,first_epoch_credit=8203,two_points_cold_exact=True,restore_next_step_B_gate_required=True,median_reduction_at_least=.15,no_substantive_recovery_or_resource_degradation=True,
            resource_limits=dict(candidate_median_peak_RSS_at_most_baseline_ratio=1.10,RSS_additive_bytes=128*2**20,candidate_median_cold_seconds_at_most_baseline_ratio=1.10,cold_additive_seconds=5,candidate_median_bundle_bytes_at_most_baseline_ratio=1.05,bundle_additive_bytes=2**20)),
        applicability='Windows CPU finite fixed-tape engineering workflow only, actual TR13 source components plus equal LOCAL bookkeeping/admission adapters; not native inference, IPC transport, Linux/GPU or live training throughput',
        statistical_claim='two repeats per arm are descriptive; no significance or generalization claim; no extra cases chasing result',
        scientific_credit=0,server_scope_active=False,checkpoint_profile=profile(),local_CPU_job_cap_seconds=read(V3/'reports/LOCAL_BUDGET.json')['local_cap_seconds'])
    write(path,protocol);print('Frozen before any D case: '+sha(path))
if __name__=='__main__':main()
