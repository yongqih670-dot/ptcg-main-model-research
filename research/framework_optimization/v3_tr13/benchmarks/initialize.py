"""Read-only TR13 provenance, small isolated code copies; never activates jobs."""
import hashlib,json,os,shutil,sys,time
from pathlib import Path
V3=Path(__file__).resolve().parents[1];ROOT=V3.parents[2]
sys.path.insert(0,str(ROOT))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()
def write(p,value):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
    import psutil
    began=time.monotonic();cpu_began=sum(psutil.Process().cpu_times()[:2]);run=ROOT/'runs/20261030T142634Z_tr13-volume-segment-004_53eca8'
    for n in ('baseline','candidate','tests','reports','patches','repairs','work'): (V3/n).mkdir(exist_ok=True)
    if (V3/'reports/BASELINE.json').exists():raise ValueError('Do not overwrite original v3 registration')
    state=json.loads((ROOT/'research/autonomy/state.json').read_text(encoding='utf-8'));policy=json.loads((ROOT/'research/autonomy/policy.json').read_text(encoding='utf-8'))
    if state.get('active_job') or state.get('active_run_id'):raise ValueError('Existing sole controller has an active job; engineering yields')
    from ptcg_lab.server_trial import trial_source_files
    config=json.loads((run/'config.json').read_text(encoding='utf-8'));files=trial_source_files(run/'source',ROOT,config)
    wrapper=b'from ptcg_lab.server_trial_worker import main\nmain()\n';files.update({'training_worker.py':wrapper,'training_check.py':wrapper})
    actual=json.loads((run/'remote-source.json').read_text(encoding='utf-8'));sources={};differences=[]
    for name,digest in actual.items():
        if name not in files or hashlib.sha256(files[name]).hexdigest()!=digest:raise ValueError('Cannot reconstruct actual last TR13 runtime: '+name)
        for folder in ('baseline','candidate'):
            p=V3/folder/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(files[name])
        local=ROOT/name;sources[name]={'sha256':digest,'bytes':len(files[name])}
        if local.exists() and sha(local)!=digest:differences.append({'path':name,'runtime':digest,'current':sha(local)})
    protected={}
    for n in ('AGENTS.md','research/autonomy/CURRENT_GOAL.md','research/autonomy/policy.json','research/autonomy/state.json','research/autonomy/latest-continuation.json',
        'research/autonomy/TR12-result.json','research/autonomy/TR13-result.json','research/autonomy/TR13_ANALYSIS.md',
        'research/framework_optimization/reports/delivery_manifest.json','research/framework_optimization/v2/reports/DELIVERY_INVENTORY.json',
        'experiments/local_modeling_20261029/LM02_fieldwise/reports/DELIVERY_MANIFEST.json',
        'research/autonomy/engineering/framework-v2-server-20261029/FINAL_ENGINEERING_REPORT.json',
        'research/autonomy/engineering/framework-v2-checkpoint-800-20261030/SUPPLEMENTAL_ACCEPTANCE_800.json'):
        p=ROOT/n
        if p.exists():protected[n]=sha(p)
    for p in (ROOT/'research').rglob('*NV01*'):
        if p.is_file():protected[str(p.relative_to(ROOT)).replace('\\','/')]=sha(p)
    for name in ('attempts.json','metrics.json','config.json','remote-source.json','remote-terminal.json','source-manifest.json','phase-timing-controller.json','remote-results/phase-timing-train.json'):
        p=run/name;protected[str(p.relative_to(ROOT)).replace('\\','/')]=sha(p)
    from ptcg_lab.server_workspace import logical_bytes
    working=logical_bytes(ROOT);free=shutil.disk_usage(ROOT).free
    if working+1536*2**20>120*2**30 or free-1536*2**20<20*2**30:raise OSError('Initial full-workspace growth admission insufficient; keep originals and do not launch')
    manifest={'at':time.time(),'actual_source_run':str(run.relative_to(ROOT)).replace('\\','/'),'runtime_sources':sources,'current_runtime_differences':differences,
        'protected_files':protected,'project_budget_snapshot':state['budget_used'],'project_CPU_available_before_final_reserve_hours':policy['budget']['cpu_job_wall_hours_max']-policy['budget']['cpu_job_wall_hours_reserved_final']-state['budget_used']['cpu_job_wall_hours'],
        'project_GPU_available_before_final_reserve_hours':policy['budget']['gpu_job_wall_hours_max']-policy['budget']['gpu_job_wall_hours_reserved_final']-state['budget_used']['gpu_job_wall_hours'],
        'actual_local_working_set_bytes':working,'local_physical_free_bytes':free,'RAM_available_bytes':psutil.virtual_memory().available,
        'server_scope':'closed; no new CPU/GPU/native ticket registered by this chat','scientific_training_authorized':False,'scientific_decisions':0}
    write(V3/'reports/BASELINE.json',manifest)
    write(V3/'AUTHORITY.json',{'at':time.time(),'provenance':'Codex delegation from01a19033-1f4a-7ac2-9c0c-b3de79b1c9d1, reporting explicit human allow; no independent source-chat read',
        'scope':'local engineering implementation/fixed-input PPO replay only; server activation stays sole original controller',
        'no_scientific_training_no_new_agents_Goal_chat_automation_messages':True,'existing_project_caps_and_reserves_unchanged':True})
    write(V3/'reports/SUPPORT_LEDGER.json',{'records':[{'kind':'read_only_initialization','wall_seconds':time.monotonic()-began,'own_CPU_seconds':sum(psutil.Process().cpu_times()[:2])-cpu_began,'GPU_seconds':0}],'historical_ledgers_modified':False})
    print(json.dumps({k:manifest[k] for k in ('actual_source_run','current_runtime_differences','project_CPU_available_before_final_reserve_hours','actual_local_working_set_bytes','RAM_available_bytes','server_scope')}))
if __name__=='__main__':main()
