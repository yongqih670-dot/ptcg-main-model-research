"""Freeze a completed Transformer learning run for official CPU BO3 evaluation."""
import argparse
import shutil
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ptcg_lab.io import read_json,sha256,atomic_json,utcnow


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--name',required=True);a=p.parse_args();run=a.run.resolve()
    metrics=read_json(run/'metrics.json');cfg=read_json(run/'config.json')
    if not metrics['complete'] or sha256(run/'model.pt')!=metrics['model_sha256'] or not metrics['cpu_export_check']['passed']:raise ValueError('Training/export incomplete')
    if not all(c.isalnum() or c=='-' for c in a.name):raise ValueError('Unsafe name')
    folder=ROOT/'agents/learned'/a.name
    if folder.exists():raise ValueError('Do not overwrite a learned agent')
    folder.mkdir(parents=True);shutil.copy2(run/'model.pt',folder/'model.pt')
    anchor=read_json(ROOT/'research/opponents/release_anchor.json')
    spec={**anchor['spec'],'id':a.name,'kind':'transformer_bc','model_path':(folder/'model.pt').relative_to(ROOT).as_posix(),
        'model_sha256':metrics['model_sha256'],'architecture':cfg['architecture'],
        'learned_contexts':sorted(int(k.removeprefix('context_')) for k,v in metrics['offline_metrics']['train']['by_context_and_kind'].items() if k.startswith('context_') and v['steps']>0)}
    atomic_json(folder/'spec.json',spec)
    atomic_json(folder/'provenance.json',{'at':utcnow(),'training_run':run.name,'training_config_sha256':sha256(run/'config.json'),
        'source_manifest_sha256':sha256(run/'source-manifest.json'),'dataset_manifest_sha256':cfg['dataset_manifest_sha256'],
        'model_sha256':metrics['model_sha256'],'feature_version':metrics['feature_version'],
        'initialization':metrics['initialization'],'test_scored':False,'submitted':False,
        'coverage_rule':'Use learned full conditional policy only in contexts with retained train labels; otherwise frozen PPO. Derived before BO3 from training coverage only. Large-option branches always use full chunks.'})
    print(str(folder/'spec.json'))


if __name__=='__main__':main()
