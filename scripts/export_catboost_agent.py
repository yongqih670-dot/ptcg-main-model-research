"""Create an immutable development agent spec from a completed learning run."""
import argparse
import shutil
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ptcg_lab.io import read_json,sha256,atomic_json,utcnow


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--name',required=True);a=p.parse_args()
    run=a.run.resolve();metrics=read_json(run/'metrics.json')
    if not metrics['complete'] or sha256(run/'model.cbm')!=metrics['model_sha256']:raise ValueError('Training model incomplete or changed')
    if not all(c.isalnum() or c=='-' for c in a.name):raise ValueError('Unsafe agent name')
    folder=ROOT/'agents/learned'/a.name
    if folder.exists():raise RuntimeError('Do not overwrite a learned agent')
    folder.mkdir(parents=True);shutil.copy2(run/'model.cbm',folder/'model.cbm')
    anchor=read_json(ROOT/'research/opponents/release_anchor.json')
    contexts=sorted(int(k) for k,v in metrics['offline_metrics']['train']['by_context'].items() if v['decisions']>0)
    spec={**anchor['spec'],'id':a.name,'kind':'catboost_single','model_path':str((folder/'model.cbm').relative_to(ROOT)).replace('\\','/'),
        'model_sha256':metrics['model_sha256'],'learned_contexts':contexts}
    atomic_json(folder/'spec.json',spec)
    atomic_json(folder/'provenance.json',{'at':utcnow(),'training_run':run.name,'model_sha256':metrics['model_sha256'],
        'training_config_sha256':sha256(run/'config.json'),'feature_version':metrics['feature_version'],
        'data_manifest_sha256':metrics['dataset_manifest_sha256'],'baseline_manifest_sha256':sha256(ROOT/'research/opponents/release_anchor.json'),
        'inference_rule':'Learned scoring only for mandatory nonforced single-choice contexts with >=1 retained training decision; otherwise frozen PPO. Gate uses training coverage only, no BO3 results.',
        'test_scored':metrics['test_labels_scored'],'submitted':False})
    print(str(folder/'spec.json'))


if __name__=='__main__':main()
