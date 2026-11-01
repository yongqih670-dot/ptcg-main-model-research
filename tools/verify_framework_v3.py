"""Check published source hashes and frozen result arithmetic; no model/data load."""
import hashlib,json,math,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];V3=ROOT/'research/framework_optimization/v3_tr13'
def read(p):return json.loads(p.read_bytes().decode('utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def close(a,b):return math.isclose(a,b,rel_tol=0,abs_tol=1e-8)
def main():
    manifest=read(V3/'PUBLICATION_FILES.json')
    for row in manifest['files']:
        p=(ROOT/row['path']).resolve()
        if not p.is_relative_to(ROOT) or not p.is_file() or p.stat().st_size!=row['bytes'] or sha(p)!=row['sha256']:raise ValueError('Published file differs: '+row['path'])
    for name,digest in read(V3/'reports/framework-v3-manifest.json')['runtime_sources'].items():
        if sha(V3/'candidate'/name)!=digest:raise ValueError('Candidate source differs: '+name)
    protocol=read(V3/'reports/PERFORMANCE_PROTOCOL.json');reported=read(V3/'reports/PERFORMANCE.json');states={};models=[];cases=[]
    if [x['label'] for x in reported['cases']]!=[x['label'] for x in protocol['order']]:raise ValueError('Frozen case order differs')
    for case in reported['cases']:
        label=case['label'];warm=read(V3/'reports'/f'{label}-final.json');cold=read(V3/'reports'/f'{label}-replay-cold.json');cadence=read(V3/'reports'/f'{label}-cadence.json')
        if (warm['complete_BO3'],warm['decisions'],warm['first_epoch_credit'],warm['positive_optimizer_steps'],warm['scientific_credit'])!=(48,8203,8203,66,0):raise ValueError('Full input coverage differs')
        if not cold['passed'] or len(cold['points'])!=2 or len(cadence['events'])!=12 or any((e.get('reliable_commit_gap_seconds') or 0)>800 for e in cadence['events']):raise ValueError('Recovery/cadence evidence differs')
        if not close(sum(case[k] for k in ['worker_seconds','cold_seconds','retention_seconds']),case['end_to_end_seconds']):raise ValueError('End-to-end sum differs')
        for point in cold['points']:states.setdefault(point['generation'],[]).append(point['semantic_sha256']);models.append(point['model_optimizer_RNG_sha256'])
        cases.append(case)
    if any(len(x)!=4 or len(set(x))!=1 for x in states.values()) or len(models)!=8 or len(set(models))!=1:raise ValueError('Cross-case logical recovery states differ')
    median=lambda arm,key:statistics.median(x[key] for x in cases if x['variant']==arm)
    b=median('baseline','end_to_end_seconds');c=median('candidate','end_to_end_seconds');reduction=1-c/b;limits=protocol['acceptance']['resource_limits']
    gates=dict(RSS=median('candidate','peak_RSS_bytes')<=median('baseline','peak_RSS_bytes')*limits['candidate_median_peak_RSS_at_most_baseline_ratio']+limits['RSS_additive_bytes'],cold=median('candidate','cold_seconds')<=median('baseline','cold_seconds')*limits['candidate_median_cold_seconds_at_most_baseline_ratio']+limits['cold_additive_seconds'],bundle=median('candidate','bundle_bytes')<=median('baseline','bundle_bytes')*limits['candidate_median_bundle_bytes_at_most_baseline_ratio']+limits['bundle_additive_bytes'])
    met=reduction>=protocol['acceptance']['median_reduction_at_least'] and all(gates.values())
    if not close(reduction,reported['relative_median_reduction']) or gates!=reported['resource_subgates'] or met!=reported['local_threshold_met']:raise ValueError('Published numerical verdict differs')
    fees=read(V3/'reports/SUPPORT_LEDGER.json')
    if fees['conservative_charged_CPU_job_seconds']>fees['local_cap_seconds'] or fees['GPU_seconds']!=0:raise ValueError('Original engineering accounting differs')
    print(json.dumps(dict(published_files_verified=len(manifest['files']),baseline_median_seconds=b,candidate_median_seconds=c,reduction=reduction,threshold_met=met,resource_gates=gates,all8_cold_state_receipts_consistent=True,science_credit=0,raw_training_data_loaded=False,verification_scope='Published byte hashes and evidence arithmetic only; full numerical replay requires private inputs'),ensure_ascii=False))
if __name__=='__main__':main()
