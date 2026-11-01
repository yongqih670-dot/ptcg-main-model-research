"""Finite identical hot-control probe; descriptive only, not PPO performance."""
import argparse,json,os,statistics,subprocess,sys,tempfile,time
from pathlib import Path
V3=Path(__file__).resolve().parents[1];ROOT=V3.parents[2]
def child(variant,instrumented,repeat,calls,tag):
    sys.path.insert(0,str(V3/variant))
    import psutil,torch
    psutil.Process().cpu_affinity(psutil.Process().cpu_affinity()[-3:]);torch.set_num_threads(2)
    from ptcg_lab.io import atomic_json,read_json
    from ptcg_lab.training_control import TrainingControl
    from ptcg_lab.shared_capacity import AUTH,PROFILE
    from ptcg_lab.fo_v2.durable import identity
    from trace import Trace,install
    with tempfile.TemporaryDirectory(dir=V3/'work',prefix='probe-') as d:
        root=Path(d);project='v3-component';atomic_json(root/'.ptcg-owner.json',dict(project_id=project,uid=os.getuid() if hasattr(os,'getuid') else 0,root=str(root)))
        atomic_json(root/'ticket.json',{'shared_capacity':PROFILE,'run_id':root.name,'lease_epoch':1,'gpu_uuid':None})
        atomic_json(root/'resource-training-control.json',dict(run_id=root.name,lease_epoch=1,sequence=0,command='run',reason='fixed sufficient CPU capacity',authority=AUTH,torch_threads=2,GPU_co_tenant_mode=False))
        os.environ.update(PTCG_PROJECT_ROOT=str(root),PTCG_PROJECT_ID=project,PTCG_JOB_DEADLINE_UNIX=str(time.time()+300))
        tracer=Trace()
        if instrumented:install(tracer)
        kw={'optimizations':{'control_cache':True}} if variant=='candidate' else {}
        control=TrainingControl(root,root.name,1,time.time()+300,lambda *a,**k:None,**kw)
        start=time.monotonic()
        for _ in range(calls):control.wait(gpu_required=False)
        elapsed=time.monotonic()-start
        result=dict(variant=variant,instrumented=instrumented,repeat=repeat,calls=calls,body_seconds=elapsed,scientific_decisions=0,
            cache_hits=getattr(getattr(control,'_json_evidence',None),'hits',0),trace=tracer.report() if instrumented else None)
        prefix='' if tag=='v1' else tag+'-';path=V3/'reports'/f'probe-{prefix}{variant}-{int(instrumented)}-{repeat}.json';path.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k!='trace'}))
def main():
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=['baseline','candidate']);p.add_argument('--instrumented',type=int,choices=[0,1],default=1);p.add_argument('--repeat',type=int,default=0);p.add_argument('--calls',type=int,default=1500);p.add_argument('--tag',choices=['v1','v2'],default='v1');a=p.parse_args()
    if not 100<=a.calls<=1500:raise ValueError('Finite calibration call bound')
    if a.variant:return child(a.variant,bool(a.instrumented),a.repeat,a.calls,a.tag)
    rows=[]
    for variant,instrumented,repeat in [('baseline',0,0),('baseline',1,0),('baseline',1,1),('baseline',0,1),('candidate',0,0),('candidate',1,0),('candidate',1,1),('candidate',0,1)]:
        began=time.monotonic();r=subprocess.run([sys.executable,'-B',str(Path(__file__)),'--variant',variant,'--instrumented',str(instrumented),'--repeat',str(repeat),'--calls',str(a.calls),'--tag',a.tag],capture_output=True,text=True,timeout=300)
        if r.returncode:print(r.stderr);raise RuntimeError('Component probe failed; retain diagnostics before another attempt')
        prefix='' if a.tag=='v1' else a.tag+'-';row=json.loads((V3/'reports'/f'probe-{prefix}{variant}-{instrumented}-{repeat}.json').read_text(encoding='utf-8'));row['whole_subprocess_seconds']=time.monotonic()-began;rows.append(row)
    summary={}
    for variant in ('baseline','candidate'):
        off=statistics.median(r['body_seconds'] for r in rows if r['variant']==variant and not r['instrumented']);on=statistics.median(r['body_seconds'] for r in rows if r['variant']==variant and r['instrumented'])
        summary[variant]=dict(uninstrumented_median_seconds=off,instrumented_median_seconds=on,observed_instrumentation_relative_overhead=on/off-1)
    target=V3/'reports'/('COMPONENT_PROBE.json' if a.tag=='v1' else 'COMPONENT_PROBE_'+a.tag+'.json');target.write_text(json.dumps(dict(summary=summary,rows=rows,description=f'Same{a.calls} real CPU control wait calls; tiny fixed files, no sampling/optimizer/native; two repeats descriptive; v2 includes original production spans'),indent=2),encoding='utf-8');print(json.dumps(summary))
if __name__=='__main__':main()
