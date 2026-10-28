"""Descriptive scalar-log plots; no simulator, model loading, or selection."""
import argparse
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ptcg_lab.io import atomic_json,read_json,sha256,utcnow


def summarize(run):
    run=Path(run); updates=read_json(run/'training-updates.json')
    versions=[u['version'] for u in updates]
    if versions!=list(range(1,len(updates)+1)):raise ValueError('Missing/duplicate update versions')
    rows=[];decisions=0
    for update in updates:
        decisions+=update['decisions']
        steps=[s for s in update['steps'] if not s.get('skipped_for_kl')]
        total=sum(s['batch'] for s in steps)
        if total<=0:raise ValueError('No applied optimizer steps')
        def mean(key):
            values=[s[key] for s in steps]
            if not all(math.isfinite(v) for v in values):raise ValueError('Nonfinite '+key)
            return sum(s[key]*s['batch'] for s in steps)/total
        row={'version':update['version'],'decisions':decisions,'rollout_decisions':update['decisions'],
             'sample_presentations':total,'optimizer_steps':len(steps),
             'policy_loss':mean('policy_loss'),'value_loss':mean('value_loss'),
             'entropy':mean('entropy'),'clip_fraction':mean('clip_fraction'),
             'preclip_grad_norm':mean('grad_norm'),'max_observed_kl':max(s['approx_kl'] for s in update['steps']),
             'kl_early_stop':any(s.get('skipped_for_kl') for s in update['steps'])}
        rows.append(row)
    return {'run_id':run.name,'source_path':str((run/'training-updates.json').resolve()),
            'source_sha256':sha256(run/'training-updates.json'),'rows':rows}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--runs',nargs='+',required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();reports=[summarize(ROOT/'runs'/r) for r in args.runs]
    # Immutable completed runs only: a moving training trace should not be
    # mistaken for a frozen comparison or a completed experiment.
    for r in args.runs:
        metrics=read_json(ROOT/'runs'/r/'metrics.json')
        if not metrics.get('complete'):raise ValueError('Only completed training logs may be plotted')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(12,10),layout='constrained')
    fields=[('policy_loss','Policy loss (weighted mean)'),('value_loss','Value loss (weighted mean)'),
            ('entropy','Conditional sequence entropy'),('clip_fraction','PPO clipped fraction'),
            ('max_observed_kl','Maximum minibatch KL (including stop)'),('preclip_grad_norm','Gradient norm before clipping')]
    colors=['#2369a0','#db7c28','#26836a','#b44560']
    for index,report in enumerate(reports):
        label=report['run_id'].split('_',1)[1].rsplit('_',1)[0]
        rows=report['rows'];x=[r['decisions']/1000 for r in rows]
        for axis,(field,title) in zip(axes.flat,fields):
            axis.plot(x,[r[field] for r in rows],marker='o',markersize=3,linewidth=1.5,label=label,color=colors[index%len(colors)])
            axis.set_title(title,fontsize=11);axis.set_xlabel('Learner decisions (thousands)');axis.grid(alpha=.2)
    axes[2,0].axhline(.02,color='#666666',linestyle='--',linewidth=1,label='KL stop threshold')
    axes[2,1].axhline(.5,color='#666666',linestyle='--',linewidth=1,label='Clipping threshold')
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Completed PPO training traces: TR01–TR03',fontsize=16)
    fig.text(.5,-.015,'Descriptive training metrics only. Samples, opponents and policies change; these curves do not establish BO3 strength.',ha='center',fontsize=9)
    args.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(args.output,dpi=170,bbox_inches='tight');plt.close(fig)
    atomic_json(args.output.with_suffix('.json'),{'at':utcnow(),'source':'actual immutable completed training logs',
                 'aggregation':'batch-size weighted applied-step means; maximum KL includes skipped stop minibatch',
                 'selection_or_promotion_evidence':False,'figure_sha256':sha256(args.output),'reports':reports})
    print({'figure':str(args.output),'run_count':len(reports),'updates':[len(r['rows']) for r in reports]})


if __name__=='__main__':main()
