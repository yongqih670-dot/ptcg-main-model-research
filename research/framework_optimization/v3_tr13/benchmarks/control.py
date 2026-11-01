"""One owning process, finite children, append-only costs; no scientific dispatch."""
import argparse,contextlib,json,os,secrets,sys,time
from pathlib import Path
from common import *

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['zero','qual','qual-recover','qual-cold-repair','qual-deterministic','qual-det-native2','perf','seal-perf-failure'],required=True);p.add_argument('--qualification-attempt',type=int,default=1);p.add_argument('--cold-attempt',type=int,default=1);a=p.parse_args()
    sys.path.insert(0,str(V3/'baseline'))
    import msvcrt,psutil
    psutil.Process().cpu_affinity(psutil.Process().cpu_affinity()[-3:])
    from supervisor import run
    lock=(V3/'controller.lock').open('a+b');lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    me=psutil.Process();token=secrets.token_hex(24);owner=dict(phase='running',token=token,pid=os.getpid(),created=me.create_time(),stage=a.stage,started=time.time())
    os.environ['V3_TOKEN']=token;write(V3/'reports/owner.json',owner)
    ledger=V3/'reports/JOB_LEDGER.json';rows=read(ledger)['rows'] if ledger.exists() else [];completed=False
    local_cap=read(V3/'reports/LOCAL_BUDGET.json')['local_cap_seconds'] if (V3/'reports/LOCAL_BUDGET.json').exists() else 8*3600
    try:
        guard()
        if a.stage=='zero':jobs=[('zero','A-real-parent','candidate'),('cold','A-real-parent','candidate')]
        elif a.stage=='seal-perf-failure':jobs=[('seal-failed','D-baseline-0','baseline'),('replay-cold','D-baseline-0','baseline')]
        elif a.stage in ('qual','qual-recover','qual-cold-repair'):
            if not read(V3/'reports/A-real-parent-cold.json')['passed']:raise ValueError('Full zero-update recovery gate first')
            prefix='B' if a.qualification_attempt==1 else 'B'+str(a.qualification_attempt)
            jobs=[('qual',prefix+'-baseline','baseline'),('replay-cold',prefix+'-baseline','baseline'),('interrupt',prefix+'-candidate','candidate'),('resume',prefix+'-candidate','candidate'),('replay-cold',prefix+'-candidate','candidate')]
            if a.stage=='qual-recover':jobs[0]=('recover-qual',prefix+'-baseline','baseline')
            if a.stage=='qual-cold-repair':jobs=jobs[1:];jobs[0]=('replay-cold-repair',prefix+'-baseline','baseline')
        elif a.stage=='qual-deterministic':jobs=[('det-reference','B4-reference','baseline'),('det-native','B4-native','candidate'),('replay-cold','B4-native','candidate')]
        elif a.stage=='qual-det-native2':jobs=[('det-native','B4-native2','candidate'),('replay-cold','B4-native2','candidate')]
        else:
            if not read(V3/'reports/QUALIFICATION.json')['passed']:raise ValueError('Correctness before performance')
            protocol=read(V3/'reports/PERFORMANCE_PROTOCOL.json')
            jobs=[item for case in protocol['order'] for item in [('perf',case['label'],case['variant']),('replay-cold',case['label'],case['variant'])]]
        for mode,label,variant in jobs:
            limit=protocol.get('warm_job_maximum_seconds',3600) if mode=='perf' else 1800
            conservative=time.time()-read(V3/'AUTHORITY.json')['at']+read(V3/'reports/SUPPORT_LEDGER.json')['records'][0]['wall_seconds']
            if max(conservative,sum(r['CPU_job_seconds'] for r in rows))+limit>local_cap:raise RuntimeError('Finite local engineering budget reserve exceeded')
            os.environ['V3_DEADLINE']=str(time.time()+limit-35)
            evidence_mode=mode+(f'-attempt{a.cold_attempt}' if mode=='replay-cold-repair' and a.cold_attempt>1 else '')
            receipt=V3/'reports'/f'{label}-{evidence_mode}-process.json'
            if receipt.exists():raise ValueError('Existing evidence retained: explicit diagnosis required before a new attempt')
            write(V3/'reports/CONTINUATION.json',dict(at=time.time(),phase=a.stage,active_local_controller=dict(pid=owner['pid'],created=owner['created']),active_finite_job_receipt=str(receipt.relative_to(V3)),active_scientific_job=None,science_credit=0,server_scope='unregistered; no server launch'))
            if a.stage=='perf':
                if sha(V3/'reports/INPUTS.json')!=protocol['inputs_sha256'] or sha(V3/'reports/framework-v3-manifest.json')!=protocol['candidate_manifest_sha256']:raise ValueError('Frozen input/manifest bytes changed')
                for source,digest in protocol['benchmarks'].items():
                    if sha(V3/source)!=digest:raise ValueError('Frozen performance harness changed')
                for source,digest in read(V3/'reports/framework-v3-manifest.json')['runtime_sources'].items():
                    if sha(V3/'candidate'/source)!=digest:raise ValueError('Frozen candidate source changed')
                for source,row in protocol['baseline_runtime_sources'].items():
                    if sha(V3/'baseline'/source)!=row['sha256']:raise ValueError('Frozen actual baseline source changed')
                for source,digest in protocol['external_actor_assets'].items():
                    if sha(ROOT/source)!=digest:raise ValueError('Frozen actor/encoder asset changed')
            row=run([sys.executable,'-B',str(V3/'benchmarks/worker.py'),mode,label,'--variant',variant],cwd=str(ROOT),console=V3/'reports'/f'{label}-{evidence_mode}-console.txt',receipt=receipt,progress=job_path(label)/'progress.json',stop_file=V3/'STOP',context=dict(scientific_credit=0,stage=mode,variant=variant,worker_sha256=sha(V3/'benchmarks/worker.py'),replay_sha256=sha(V3/'benchmarks/replay.py')),maximum_seconds=limit)
            rows.append(dict(path=str(receipt.relative_to(V3)),CPU_job_seconds=row['seconds'],returncode=row['returncode'],GPU_seconds=0))
            write(V3/'reports/JOB_LEDGER.json',dict(rows=rows,CPU_job_seconds=sum(r['CPU_job_seconds'] for r in rows),cap_seconds=local_cap,historical_ledgers_modified=False))
            if row['returncode']!=0:raise RuntimeError('Finite worker failed; no blind retry')
            if a.stage in ('perf','seal-perf-failure') and mode=='replay-cold':
                if a.stage=='perf':
                    events=[__import__('json').loads(line) for line in (job_path(label)/'framework-v2-cadence.jsonl').read_text(encoding='utf-8').splitlines()]
                    expected_count=len(protocol['ingest_checkpoint_BO3'])+4
                    if len(events)!=expected_count or any(event.get('within_maximum_gap') is False for event in events):raise ValueError('Checkpoint cadence acceptance failed')
                    write(V3/'reports'/f'{label}-cadence.json',dict(passed=True,events=events,maximum_seconds=800,expected_commits=expected_count))
                conservative=time.time()-read(V3/'AUTHORITY.json')['at']+read(V3/'reports/SUPPORT_LEDGER.json')['records'][0]['wall_seconds']
                if max(conservative,sum(r['CPU_job_seconds'] for r in rows))+1800>local_cap:raise RuntimeError('Reserve full retention endpoint before creating a new archive')
                from compact import compact
                begin=time.monotonic();os.environ['V3_DEADLINE']=str(time.time()+1800)
                compact(label);rows.append(dict(path=f'retained/{label}.json',CPU_job_seconds=time.monotonic()-begin,returncode=0,GPU_seconds=0))
                write(ledger,dict(rows=rows,CPU_job_seconds=sum(r['CPU_job_seconds'] for r in rows),cap_seconds=local_cap,historical_ledgers_modified=False))
        if a.stage in ('qual','qual-recover','qual-cold-repair'):
            import torch
            import ptcg_lab.framework_v2_training as native
            base=V3/'work'/(prefix+'-baseline')/'job';candidate=V3/'work'/(prefix+'-candidate')/'job'
            whole=strict_state(torch.load(base/'expected.pt',weights_only=False),torch.load(candidate/'expected.pt',weights_only=False))
            next_step=strict_state(torch.load(base/'next-step.pt',weights_only=False),torch.load(candidate/'next-step.pt',weights_only=False))
            final=read(V3/'reports'/(prefix+'-candidate-final.json'))
            write(V3/'reports/QUALIFICATION.json',dict(passed=True,baseline_label=prefix+'-baseline',candidate_label=prefix+'-candidate',whole_logical_state=whole,restored_next_optimizer_step=next_step,BO3=final['complete_BO3'],decisions=final['decisions'],first_epoch_credit=final['first_epoch_credit'],scientific_credit=0,fresh_process_resume=True))
        if a.stage in ('qual-deterministic','qual-det-native2'):
            import torch
            torch.set_num_threads(2)
            native_label='B4-native2' if a.stage=='qual-det-native2' else 'B4-native';direct=job_path('B4-reference');native=job_path(native_label)
            whole=strict_state(torch.load(direct/'expected.pt',weights_only=False),torch.load(native/'expected.pt',weights_only=False));step=strict_state(torch.load(direct/'next-step.pt',weights_only=False),torch.load(native/'next-step.pt',weights_only=False))
            write(V3/'reports/QUALIFICATION.json',dict(passed=True,baseline_label='B4-reference',candidate_label=native_label,whole_logical_state=whole,restored_next_optimizer_step=step,BO3=2,decisions=319,first_epoch_credit=319,
                scientific_credit=0,fresh_process_resume=True,CPU_deterministic_algorithms=True,full_input='same actual producer-bound committed revision1 partial PPO; direct continuation vs fresh Native.restore; all scientific counters deferred until commit',
                prior_nondeterminism_observations_preserved=True))
        completed=True
    finally:
        owner.update(phase='terminal',finished=time.time());write(V3/'reports/owner.json',owner)
        write(V3/'reports/CONTINUATION.json',dict(at=time.time(),phase=a.stage+('_completed' if completed else '_stopped_with_evidence'),active_local_controller=None,active_scientific_job=None,science_credit=0,server_scope='unregistered; no server launch'))
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()
if __name__=='__main__':main()
