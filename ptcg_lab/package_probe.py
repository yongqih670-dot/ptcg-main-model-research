"""Run an extracted submission entry in isolation on real official BO3.

This is engineering evidence, never a strength screen or a submission gate by
itself. Linux and verified server quotas remain explicit independent checks.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path, PurePosixPath
import platform
import queue
import subprocess
import sys
import tarfile
import threading
import time
import traceback
import psutil
from .agents import ProtocolAgent
from .evaluate import load_official, quantile
from .io import read_json, atomic_json, sha256, utcnow
from .public_agent import create_agent


CHILD = r'''
import contextlib,json,os,platform,runpy,sys,time
from pathlib import Path
import psutil
root=Path(sys.argv[1]).resolve()
os.chdir(root);sys.path.insert(0,str(root))
started=time.perf_counter()
with contextlib.redirect_stdout(sys.stderr):
    namespace=runpy.run_path(str(root/'main.py'),run_name='packaged_ptcg_entry')
fn=namespace['agent'];process=psutil.Process()
def peak():
    memory=process.memory_info()
    value=getattr(memory,'peak_wset',memory.rss)
    if platform.system()=='Linux':
        import resource
        value=max(value,resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    return value
print(json.dumps({'event':'ready','import_seconds':time.perf_counter()-started,'peak_rss_bytes':peak()}),flush=True)
for line in sys.stdin:
    message=json.loads(line)
    if message.get('command')=='close':break
    t=time.perf_counter()
    with contextlib.redirect_stdout(sys.stderr):
        action=fn(message['observation'])
    policy=namespace['_POLICY'].module.SA.policy
    devices=sorted({str(p.device) for p in policy.parameters()})
    print(json.dumps({'event':'action','action':action,'seconds':time.perf_counter()-t,'peak_rss_bytes':peak(),'parameter_devices':devices}),flush=True)
'''


def extract_verified(archive, destination, expected_sha256):
    archive=Path(archive);destination=Path(destination)
    if sha256(archive)!=expected_sha256:raise ValueError('Archive changed')
    destination.mkdir(parents=True,exist_ok=False)
    with tarfile.open(archive,'r:gz') as tf:
        members=tf.getmembers();names=set()
        if sum(m.size for m in members)>512*1024**2:raise ValueError('Unexpected expanded package size')
        for member in members:
            path=PurePosixPath(member.name)
            if (not member.isfile() or path.is_absolute() or '..' in path.parts or '\\' in member.name
                    or ':' in member.name or member.name in names or str(path)!=member.name):
                raise ValueError('Unsafe or duplicate archive member')
            names.add(member.name)
        if not {'main.py','deck.csv','package-manifest.json','agent-spec.json'}<=names:
            raise ValueError('Missing package entry or provenance')
        for member in members:
            target=destination/member.name;target.parent.mkdir(parents=True,exist_ok=True)
            with tf.extractfile(member) as src,target.open('xb') as dst:
                import shutil
                shutil.copyfileobj(src,dst)
    manifest=read_json(destination/'package-manifest.json')
    if names!=set(manifest['files'])|{'package-manifest.json'}:raise ValueError('Package manifest coverage mismatch')
    for name,digest in manifest['files'].items():
        if sha256(destination/name)!=digest:raise ValueError('Packaged file hash mismatch: '+name)
    if sha256(destination/'bundle/model.pt')!=manifest['model_sha256']:raise ValueError('Packaged model mismatch')
    return manifest


class EntryProcess:
    def __init__(self,folder,log,deadline):
        self.deadline=deadline;self.log=Path(log).open('w',encoding='utf-8')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
        self.process=subprocess.Popen([sys.executable,'-I','-B','-c',CHILD,str(folder)],cwd=folder,
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,encoding='utf-8',env=env)
        self.created=psutil.Process(self.process.pid).create_time();self.messages=queue.Queue()
        def read():
            for line in self.process.stdout:self.messages.put(line)
            self.messages.put(None)
        threading.Thread(target=read,daemon=True).start()
        self.ready=self.receive()
        if self.ready.get('event')!='ready':raise ValueError('Package entry not ready')
    def receive(self):
        remaining=min(60,self.deadline-time.time()-5)
        if remaining<=0:raise TimeoutError('Package probe deadline reached')
        line=self.messages.get(timeout=remaining)
        if line is None:raise RuntimeError('Isolated package entry exited')
        return json.loads(line)
    def action(self,observation):
        self.process.stdin.write(json.dumps({'observation':observation})+'\n');self.process.stdin.flush()
        response=self.receive()
        if response.get('event')!='action':raise ValueError('Invalid package response')
        return response
    def close(self):
        try:
            if self.process.poll() is None:
                self.process.stdin.write('{"command":"close"}\n');self.process.stdin.flush()
                try:self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    # Popen owns this child handle; identity is verified before killing.
                    if abs(psutil.Process(self.process.pid).create_time()-self.created)>.01:raise RuntimeError('Child identity changed')
                    self.process.kill();self.process.wait(timeout=3)
        finally:self.log.close()


class PackagedPolicy(ProtocolAgent):
    def __init__(self,entry,baseline,seat):
        super().__init__('first',baseline.deck,seat,0)
        self.entry=entry;self.baseline=baseline;self.parity_calls=0;self.mismatches=0
        self.package_seconds=[];self.package_round_seconds=Counter();self.peak_rss=entry.ready['peak_rss_bytes']
        self.max_candidates=0;self.cpu_observed=True
    def choose(self,observation):
        public=dict(observation);public['search_begin_input']=None
        response=self.entry.action(public);expected=self.baseline.choose(public)
        self.cpu_observed=self.cpu_observed and response['parameter_devices']==['cpu']
        if not self.cpu_observed:raise ValueError('Packaged model must execute on CPU')
        self.parity_calls+=1;action=response['action']
        if action!=expected:
            self.mismatches+=1;raise ValueError('Packaged action differs from frozen source policy')
        self.package_seconds.append(response['seconds']);self.peak_rss=max(self.peak_rss,response['peak_rss_bytes'])
        rd=(public.get('current') or {}).get('round',0)
        self.package_round_seconds[rd]+=response['seconds']
        self.max_candidates=max(self.max_candidates,len((public.get('select') or {}).get('option',[])))
        return action
    def close(self):self.baseline.close();self.entry.close()


def run(config,output):
    output=Path(output);source=Path(__file__).resolve().parents[1]
    if os.environ.get('PTCG_JOB_RUN_ID')!=output.name:raise RuntimeError('Package probe requires controlled runner')
    deadline=float(os.environ['PTCG_JOB_DEADLINE_UNIX'])
    archive=Path(config['archive_path']);folder=output/'extracted-package'
    manifest=extract_verified(archive,folder,config['archive_sha256'])
    spec=config['candidate'];packaged=read_json(folder/'agent-spec.json')
    if spec['model_sha256']!=packaged['model_sha256']:raise ValueError('Source/package weights differ')
    if sha256(source/spec['path']/'model.pt')!=spec['model_sha256']:raise ValueError('Source weights changed')
    # Check the implementation actually used by both sides of the parity test.
    for name,digest in manifest['files'].items():
        if name.startswith('ptcg_lab/') and sha256(source/name)!=digest:
            raise ValueError('Source/package inference implementation differs: '+name)
    make=load_official(output);rows=[]
    for sid in range(config['series_count']):
        if time.time()>deadline-60:break
        seat=sid%2;baseline=create_agent(spec,None,seat,0)
        entry=EntryProcess(folder,output/f'package-{sid:03d}.log',deadline)
        candidate=PackagedPolicy(entry,baseline,seat)
        reference=create_agent(config['reference'],None,1-seat,config['policy_seed']+sid)
        policies=[None,None];policies[seat]=candidate;policies[1-seat]=reference
        env=make('cabt',configuration={'version':2,'bo':3});error=None
        try:env.run(policies)
        except Exception:error=traceback.format_exc()
        finally:candidate.close();reference.close()
        statuses=[p.status for p in env.state]
        row={'series_id':sid,'seat':seat,'statuses':statuses,'error':error,'parity_calls':candidate.parity_calls,
             'action_mismatches':candidate.mismatches,'invalid_actions':candidate.invalid,
             'import_seconds':entry.ready['import_seconds'],'first_call_seconds':candidate.package_seconds[0] if candidate.package_seconds else None,
             'call_seconds':candidate.package_seconds,'round_seconds':dict(candidate.package_round_seconds),
             'peak_rss_bytes':candidate.peak_rss,'max_candidates':candidate.max_candidates,'contexts':dict(candidate.contexts),
             'cpu_only_observed':candidate.cpu_observed and candidate.parity_calls>0}
        rows.append(row);atomic_json(output/f'package-series-{sid:03d}.json',row)
        atomic_json(output/'progress.json',{'at':utcnow(),'completed_series':len(rows),'target_series':config['series_count']})
        if error or statuses!=['DONE','DONE']:break
    calls=[v for r in rows for v in r['call_seconds']]
    clean=all(r['statuses']==['DONE','DONE'] and not r['error'] and not r['action_mismatches'] and not r['invalid_actions'] for r in rows)
    complete=len(rows)==config['series_count'] and clean
    atomic_json(output/'metrics.json',{'at':utcnow(),'complete':complete,'engineering_only':True,'archive_sha256':sha256(archive),
        'model_sha256':spec['model_sha256'],'platform':platform.platform(),'python':sys.version,'series_count':len(rows),
        'parity_calls':sum(r['parity_calls'] for r in rows),'all_observed_actions_match_source':clean,
        'cpu_only_requested':True,'linux_execution_observed':platform.system()=='Linux',
        'cpu_only_observed':bool(rows) and all(r['cpu_only_observed'] for r in rows),
        'peak_rss_bytes':max((r['peak_rss_bytes'] for r in rows),default=0),
        'call_seconds':{'p50':quantile(calls,.5),'p95':quantile(calls,.95),'p99':quantile(calls,.99),'max':max(calls,default=0)},
        'max_cumulative_round_seconds':max((v for r in rows for v in r['round_seconds'].values()),default=0),
        'max_candidates_observed':max((r['max_candidates'] for r in rows),default=0),
        'server_resource_limits_verified':False,'final_reliability_qualified':False,'promotion_eligible':False,
        'scope':'Isolated archive entry on official BO3 with source-action parity; separate Linux, stress and actual server-limit evidence still required.'})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();run(read_json(args.config),args.output)
