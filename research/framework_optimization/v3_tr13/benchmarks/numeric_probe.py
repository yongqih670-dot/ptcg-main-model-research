"""Finite first-batch CPU determinism probe; disposable, no native engine."""
import argparse,os,subprocess,sys,time
from common import *
def child(deterministic,repeat):
    sys.path.insert(0,str(V3/'baseline'));import torch,psutil,numpy as np
    psutil.Process().cpu_affinity(psutil.Process().cpu_affinity()[-3:]);torch.set_num_threads(2);torch.use_deterministic_algorithms(deterministic)
    from ptcg_lab.rl_core import load_full_actor,optimizer_for,gae,ppo_step
    from ptcg_lab.framework_v2_training import semantic_digest
    from ptcg_lab.ppo_transaction import begin,next_batch
    spec=read(V3/'reports/INPUTS.json');job=job_path('B3-baseline');point=next(read(p)['point'] for p in (job/'objects').glob('manifest-*.bin') if read(p)['point']['generation']==0)
    model_part=torch.load(job/point['model']['object']['path'],map_location='cpu',weights_only=False);tail=torch.load(job/point['tail']['path'],map_location='cpu',weights_only=False)
    bundle=ROOT/spec['current_bundle'];model,features,_=load_full_actor(bundle,sha(bundle/'model.pt'),reset_value=False);opt=optimizer_for(model);model.load_state_dict(model_part['model'],strict=True);opt.load_state_dict(model_part['optimizer']);cpu_rng_restore(tail['rng'])
    decisions=[]
    for row in spec['trajectories'][:2]:decisions.extend(torch.load(ROOT/row['trajectory']['path'],map_location='cpu',weights_only=False)['decisions'])
    settings=spec['ppo'];adv,returns=gae(decisions,settings['gamma'],settings['gae_lambda']);adv=(adv-adv.mean())/(adv.std()+1e-8);partial=begin(decisions,adv,returns,256);chosen=next_batch(partial,2)
    started=time.monotonic();metrics=ppo_step(model,opt,[decisions[i] for i in chosen],adv[chosen],returns[chosen],features.b.STOP,clip=settings['clip'],entropy_coef=settings['entropy_coefficient'],value_coef=settings['value_loss_coefficient'],max_grad_norm=settings['max_grad_norm'],target_kl=settings['target_kl'],normalize_advantages=False,record_gradient_components=True)
    result=dict(deterministic=deterministic,repeat=repeat,model_sha256=semantic_digest(model.state_dict()),optimizer_sha256=semantic_digest(opt.state_dict()),metrics=metrics,seconds=time.monotonic()-started,scientific_credit=0)
    write(V3/'reports'/f'numeric-probe-{int(deterministic)}-{repeat}.json',result);print(json.dumps(result))
def main():
    p=argparse.ArgumentParser();p.add_argument('--deterministic',type=int);p.add_argument('--repeat',type=int,default=0);a=p.parse_args()
    if a.deterministic is not None:return child(bool(a.deterministic),a.repeat)
    rows=[]
    for flag,repeat in [(0,0),(0,1),(1,0),(1,1)]:
        began=time.monotonic();process=subprocess.run([sys.executable,'-B',str(Path(__file__)),'--deterministic',str(flag),'--repeat',str(repeat)],capture_output=True,text=True,timeout=120)
        row=dict(deterministic=bool(flag),repeat=repeat,returncode=process.returncode,seconds=time.monotonic()-began,exitcode_source='Popen.wait');rows.append(row)
        write(V3/'reports/NUMERIC_PROBE_PROCESSES.json',rows)
        if process.returncode:print(process.stderr);raise RuntimeError('Numerical probe failed; preserve diagnostics')
    print('Finite numeric probes complete')
if __name__=='__main__':main()
