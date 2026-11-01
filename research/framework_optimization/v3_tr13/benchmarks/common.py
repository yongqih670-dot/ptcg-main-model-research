"""Shared fixture, strict state checks and finite worker admission; no SSH."""
import ast,copy,gzip,hashlib,io,json,os,random,shutil,sys,time
from pathlib import Path
V3=Path(__file__).resolve().parents[1];ROOT=V3.parents[2]
_resource_checked=0.
_budget_deadline=None
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.partial');tmp.write_text(json.dumps(x,indent=2),encoding='utf-8');os.replace(tmp,p)
def guard():
    import psutil
    global _resource_checked,_budget_deadline
    if _budget_deadline is None:
        cap=read(V3/'reports/LOCAL_BUDGET.json')['local_cap_seconds']
        initial=read(V3/'reports/SUPPORT_LEDGER.json')['records'][0]['wall_seconds']
        _budget_deadline=read(V3/'AUTHORITY.json')['at']+cap-initial
    if time.time()>=_budget_deadline:raise TimeoutError('Cumulative local CPU reservation exhausted; all earlier attempts retained')
    if (V3/'STOP').exists():raise InterruptedError('Explicit isolated engineering stop')
    if os.environ.get('V3_DEADLINE') and time.time()>=float(os.environ['V3_DEADLINE']):raise TimeoutError('Finite engineering endpoint window ended')
    now=time.monotonic()
    if now-_resource_checked<1:return
    _resource_checked=now
    if psutil.virtual_memory().available<8*2**30 or psutil.Process().memory_info().rss>8*2**30:raise MemoryError('Unchanged local RAM reserve')
    if shutil.disk_usage(ROOT).free<20*2**30:raise OSError('Physical reserve')
    if read(ROOT/'research/autonomy/state.json').get('active_job'):raise RuntimeError('Sole controller active; engineering yields')
def admitted_worker():
    import psutil
    o=read(V3/'reports/owner.json');me=psutil.Process()
    if o['phase']!='running' or o['token']!=os.environ.get('V3_TOKEN') or o['pid'] not in {p.pid for p in me.parents()} or psutil.Process(o['pid']).create_time()!=o['created']:raise ValueError('Original finite local controller missing')
    guard();return o
def strict_state(left,right):
    # Reuse the frozen reviewed strict tensor/type/order/storage-alias checker,
    # without importing another ptcg_lab package or executing its audit driver.
    import numpy as np,torch
    text=(ROOT/'ptcg_lab/bounded_checkpoint_audit.py').read_text(encoding='utf-8');tree=ast.parse(text);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='exact_state')
    ns={'torch':torch,'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),'strict_original_state_audit','exec'),ns);return ns['exact_state'](left,right)
def cpu_rng_restore(row):
    import numpy as np,torch
    random.setstate(row['python']);np.random.set_state(row['numpy']);torch.set_rng_state(row['torch'])
def full_rng(parent):
    from ptcg_lab.rl_core import rng_state
    value=rng_state();value['cuda']=copy.deepcopy(parent['rng']['cuda']);return value
def profile():return dict(schema='framework_v2_reliable_checkpoint_v1',maximum_reliable_commit_gap_seconds=800,commit_reserve_seconds=60,trigger_after_seconds=740,semantics='completed_commit_to_completed_commit')
def job_path(label):return V3/'work'/label/('job-'+label if label.startswith('D-') else 'job')
def prepare_job(label,*,all_tapes=False):
    from ptcg_lab.io_pacing import BytePacer
    import psutil
    guard();p=job_path(label)
    if p.exists():raise ValueError('Existing job namespace is retained; no blind overwrite')
    p.mkdir(parents=True);root=p.parent;project='framework-v3-isolated-engineering'
    write(root/'.ptcg-owner.json',dict(project_id=project,uid=os.getuid() if hasattr(os,'getuid') else 0,root=str(root.resolve())))
    spec=read(V3/'reports/INPUTS.json');pacer=BytePacer(4*2**20,time.time()+1800)
    def stage(source,dest,digest):
        source=ROOT/source
        if sha(source)!=digest:raise ValueError('Frozen input changed')
        dest.parent.mkdir(parents=True,exist_ok=True)
        with source.open('rb') as src,dest.open('xb') as dst:
            while block:=src.read(2**20):guard();pacer.consume(len(block));dst.write(block)
            dst.flush();os.fsync(dst.fileno())
        if sha(dest)!=digest:raise ValueError('Case preparation changed original bytes')
    for row in spec['canonical_refs']:stage(row['path'],p/row['canonical'],row['sha256'])
    first=spec['trajectories'][0]
    for key in ('trajectory','summary'):
        name=first['summary_value']['trajectory_path'] if key=='trajectory' else f"series/{first['series_id']:06d}.json";stage(first[key]['path'],p/name,first[key]['sha256'])
    write(p/'local-engineering-owner.json',dict(run_id=p.name,lease_epoch=1,scope='disposable fixed replay; no scientific authority',parent_token=os.environ['V3_TOKEN']))
    return p,spec
def setup_environment(job):
    os.environ.update(PTCG_PROJECT_ROOT=str(job.parent.resolve()),PTCG_PROJECT_ID='framework-v3-isolated-engineering',PTCG_JOB_DEADLINE_UNIX=os.environ['V3_DEADLINE'])
def native_backend(job,variant,*,instrumentation=True):
    import ptcg_lab.framework_v2_training as native
    from ptcg_lab.fo_v2.durable import atomic_json
    from ptcg_lab.fo_v2.io_budget import ReadEnvelope
    # Adapt only LOCAL bookkeeping destination, equally on both arms. The real
    # project is scanned read-only; no production root capacity.json is written.
    def persist(cap):atomic_json(job/'project-capacity-ledger.json',dict(used=cap.used,reservations=cap.reservations,hard=cap.hard,external=cap.external,scans=cap.scans))
    native.ProjectCapacity._persist=persist
    native.ReadEnvelope=lambda rate:ReadEnvelope(rate,guard)
    flags=dict(control_cache=True,journal_cache=True,outbox_cache=True,capacity_transaction=True,progress_coalescing=True,exit_evidence=True)
    if variant=='candidate' and job.parent.name.startswith('D-'):
        flags=read(V3/'reports/PERFORMANCE_PROTOCOL.json')['candidate_switches']
    kw={'optimizations':flags} if variant=='candidate' else {}
    backend=native.NativeCheckpoints(job,1,profile(),capacity_root=ROOT,hard_bytes=120*2**30,**kw);backend.objects.stop=guard
    if variant=='candidate' and flags['capacity_transaction']:
        from ptcg_lab.framework_v3_admission import install
        install(backend,job.parent)
    return backend,flags
def expected_value(parent,league,revision=0):
    from ptcg_lab.framework_v2_training import semantic_digest
    value=dict(parent,league=league,model_state_revision=revision);value.pop('state_semantic_sha256',None);value['state_semantic_sha256']=semantic_digest(value);return value
def legacy_parent(spec):
    import torch
    from ptcg_lab.fo_v2.io_budget import ReadEnvelope
    row=spec['parent'];path=ROOT/row['path']
    if sha(path)!=row['sha256']:raise ValueError('Full accepted parent bytes changed')
    with ReadEnvelope(4*2**20,guard).open(path) as f:return torch.load(f,map_location='cpu',weights_only=False)
def cold_points(job,variant):
    import torch
    from ptcg_lab.framework_v2_training import semantic_digest
    setup_environment(job);backend,_=native_backend(job,variant)
    try:
        from ptcg_lab.fo_v2.io_budget import ReadEnvelope
        with ReadEnvelope(4*2**20,guard).open(job/'expected.pt') as stream:expected=torch.load(stream,map_location='cpu',weights_only=False)
        records=[]
        for which in (0,1):
            restored,point=backend.restore(which);per_point=expected
            if (job/'expected-tails.json').exists():
                ref=read(job/'expected-tails.json')[str(point['generation'])]
                if sha(job/ref['path'])!=ref['sha256']:raise ValueError('Producer per-generation expectation changed')
                tail=torch.load(job/ref['path'],map_location='cpu',weights_only=False)
                per_point={k:expected[k] if k in ('model','optimizer') else tail[k] for k in expected}
            counts=strict_state(per_point,restored);records.append(dict(generation=point['generation'],counts=counts,semantic_sha256=semantic_digest(restored),
                model_optimizer_RNG_sha256=semantic_digest({k:restored[k] for k in ('model','optimizer','rng')})))
        return dict(passed=True,fresh_process=True,points=records,full_RNG_optimizer_league_cursor_pending=True,scientific_decisions=0)
    finally:backend.close()
