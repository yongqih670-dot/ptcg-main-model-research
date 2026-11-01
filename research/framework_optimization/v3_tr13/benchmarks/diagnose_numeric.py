"""Read-only numerical bisect of retained actual checkpoints; no optimizer."""
import sys,time
from common import *
def main():
    sys.path.insert(0,str(V3/'baseline'));import torch,psutil,numpy as np
    psutil.Process().cpu_affinity(psutil.Process().cpu_affinity()[-3:]);torch.set_num_threads(2)
    from ptcg_lab.framework_v2_training import semantic_digest
    def points(label):
        job=job_path(label);result={}
        for path in (job/'objects').glob('manifest-*.bin'):
            v=read(path)['point'];result[v['generation']]=(job,v)
        return result
    def compare_tensors(a,b):
        rows=[]
        for k in a:
            if torch.is_tensor(a[k]) and not torch.equal(a[k],b[k]):rows.append(dict(key=k,maximum_absolute_delta=float((a[k]-b[k]).abs().max()),unequal=int(torch.count_nonzero(a[k]!=b[k]))))
        return rows
    report={};left=points('B3-baseline');right=points('B3-candidate')
    for generation in (0,1,2,3):
        def load(side):
            job,point=side[generation];model=torch.load(job/point['model']['object']['path'],map_location='cpu',weights_only=False);tail=torch.load(job/point['tail']['path'],map_location='cpu',weights_only=False);return model,tail
        a,ta=load(left);b,tb=load(right);pa=ta.get('partial_ppo');pb=tb.get('partial_ppo')
        report[str(generation)]=dict(model_deltas=compare_tensors(a['model'],b['model']),optimizer_equal=semantic_digest(a['optimizer'])==semantic_digest(b['optimizer']),
             rng_equal=semantic_digest(ta['rng'])==semantic_digest(tb['rng']),RNG_components_equal={k:semantic_digest(ta['rng'][k])==semantic_digest(tb['rng'][k]) for k in ta['rng']},
             partial_permutation_equal=np.array_equal(pa['permutation'],pb['permutation']) if pa and pb else None,
             partial_first_metrics=[pa['logs'][0],pb['logs'][0]] if pa and pb else None)
    a=torch.load(job_path('B3-baseline')/'next-step.pt',weights_only=False);b=torch.load(job_path('B3-candidate')/'next-step.pt',weights_only=False)
    report['next_step']=dict(chosen_equal=np.array_equal(a['chosen'],b['chosen']),model_deltas=compare_tensors(a['model'],b['model']),partial_logs=[a['partial']['logs'],b['partial']['logs']])
    write(V3/'reports/NUMERIC_BISECT.json',report)
    print({k:dict(model_unequal_tensors=len(v['model_deltas']),optimizer_equal=v.get('optimizer_equal'),rng_equal=v.get('rng_equal'),permutation_equal=v.get('partial_permutation_equal'),chosen_equal=v.get('chosen_equal')) for k,v in report.items()})
if __name__=='__main__':main()
