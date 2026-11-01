"""Truthful independent final receipts; no historical policy/ledger rewrites."""
import statistics,time
from common import *

def final():
    import psutil
    baseline=read(V3/'reports/BASELINE.json');mismatches=[]
    for name,digest in baseline['protected_files'].items():
        if sha(ROOT/name)!=digest:mismatches.append(name)
    runtime_changed=[]
    for name,row in baseline['runtime_sources'].items():
        p=ROOT/name
        if p.exists() and sha(p)!=row['sha256']:runtime_changed.append(name)
        if sha(V3/'baseline'/name)!=row['sha256']:raise ValueError('Frozen baseline changed')
    if mismatches or runtime_changed:raise ValueError('Protected original changed: '+repr(mismatches+runtime_changed))
    jobs=read(V3/'reports/JOB_LEDGER.json');probe=read(V3/'reports/COMPONENT_PROBE.json');support=read(V3/'reports/SUPPORT_LEDGER.json')
    original_initial=support['records'][0]['wall_seconds'];controller_window=time.time()-read(V3/'AUTHORITY.json')['at'];charged=controller_window+original_initial;cap=read(V3/'reports/LOCAL_BUDGET.json')['local_cap_seconds']
    support.update(finished_at=time.time(),component_probe_subprocess_seconds=sum(r['whole_subprocess_seconds'] for r in probe['rows']),finite_job_seconds=jobs['CPU_job_seconds'],
        conservative_charged_CPU_job_seconds=charged,charge_method='whole continuous controller window since authority registration plus original pre-registration initialization; includes preparation, failures, guards, tests, recovery, retention and unclassified time; no waiting deduction; constituent rows are not added twice',
        local_cap_seconds=cap,cap_satisfied=charged<=cap,GPU_seconds=0,historical_ledgers_modified=False,remaining_existing_ordinary_CPU_hours=baseline['project_CPU_available_before_final_reserve_hours']-charged/3600)
    write(V3/'reports/SUPPORT_LEDGER.json',support)
    performance=None
    if (V3/'reports/PERFORMANCE_PROTOCOL.json').exists():
        protocol=read(V3/'reports/PERFORMANCE_PROTOCOL.json');cases=[];digests={};model_digests=[]
        for case in protocol['order']:
            label=case['label'];warm=read(V3/'reports'/f'{label}-final.json');cold=read(V3/'reports'/f'{label}-replay-cold.json');retention=read(V3/'retained'/f'{label}.json')
            cadence=read(V3/'reports'/f'{label}-cadence.json')
            if not cadence['passed'] or any(e.get('within_maximum_gap') is False for e in cadence['events']):raise ValueError('Reliable800s checkpoint gate failed')
            processes=[read(V3/'reports'/f'{label}-{mode}-process.json') for mode in ('perf','replay-cold')]
            if any(r['returncode']!=0 or r['exitcode_source']!='Popen.wait' or r['status']!='terminal' for r in processes):raise ValueError('Cannot infer process success from metrics')
            if not cold['passed'] or warm['first_epoch_credit']!=8203 or warm['complete_BO3']!=48:raise ValueError('Correctness/whole BO3 coverage failed')
            for point in cold['points']:
                digests.setdefault(str(point['generation']),[]).append(point['semantic_sha256']);model_digests.append(point['model_optimizer_RNG_sha256'])
            total=sum(r['seconds'] for r in processes)+retention['seconds']
            cases.append(dict(**case,end_to_end_seconds=total,worker_seconds=processes[0]['seconds'],cold_seconds=processes[1]['seconds'],retention_seconds=retention['seconds'],bundle_bytes=retention['bundle_bytes'],peak_RSS_bytes=max(r['owned_tree_peak_RSS_observed_bytes'] for r in processes),trace=warm['trace']))
        if any(len(set(values))!=1 or len(values)!=4 for values in digests.values()) or len(set(model_digests))!=1:raise ValueError('Four full logical states are not exact by generation')
        medians={arm:statistics.median(r['end_to_end_seconds'] for r in cases if r['variant']==arm) for arm in ('baseline','candidate')};gain=1-medians['candidate']/medians['baseline']
        median_field=lambda arm,field:statistics.median(r[field] for r in cases if r['variant']==arm)
        resource_pass=(median_field('candidate','peak_RSS_bytes')<=median_field('baseline','peak_RSS_bytes')*1.10+128*2**20 and median_field('candidate','cold_seconds')<=median_field('baseline','cold_seconds')*1.10+5 and median_field('candidate','bundle_bytes')<=median_field('baseline','bundle_bytes')*1.05+2**20)
        latest=max(digests,key=int)
        performance=dict(cases=cases,median_seconds=medians,relative_median_reduction=gain,full_logical_state_semantic_sha256=digests[latest][0],per_generation_full_state_hashes={g:values[0] for g,values in digests.items()},all_8_cold_points_exact_to_producer=True,all_8_model_optimizer_RNG_equal=True,descriptive_only=True,resource_and_recovery_gate_passed=resource_pass,local_threshold_met=gain>=.15 and resource_pass,
                         applicability=protocol['applicability'],server_or_scientific_adoption=False,native_GPU_gate='pending original sole controller registration and ticket')
        write(V3/'reports/PERFORMANCE.json',performance)
    report=dict(at=time.time(),status='local_engineering_complete_server_gate_pending',correctness_A=read(V3/'reports/A-real-parent-cold.json'),correctness_B=read(V3/'reports/QUALIFICATION.json'),targeted_tests=read(V3/'reports/tests.json'),
                performance=performance,protected_originals_unchanged=True,protected_file_count=len(baseline['protected_files']),root_runtime_unchanged=True,science_credit=0,CPU_job_hours_conservatively_charged=charged/3600,GPU_hours=0,
                server_scope='closed, no network/SSH/GPU/native execution performed',old_disconnect_initiating_cause='unproven; EOF owned-close behavior preserved',available_RAM_bytes=psutil.virtual_memory().available)
    write(V3/'reports/FINAL.json',report);write(V3/'reports/CONTINUATION.json',dict(at=time.time(),phase=report['status'],active_local_job=None,active_scientific_job=None,science_credit=0,next='Original sole controller registers a NEW v3 finite engineering scope/ticket for native/GPU/fault gates; no automatic launch',historical_data_retained=True))
    return report
if __name__=='__main__':
    report=final();print({k:v for k,v in report.items() if k in ('status','CPU_job_hours_conservatively_charged','protected_originals_unchanged')})
