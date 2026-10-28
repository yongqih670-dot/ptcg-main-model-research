"""Deterministic inference-only package for deployment validation, never upload."""
import argparse
import gzip
import json
from pathlib import Path
import shutil
import sys
import tarfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ptcg_lab.io import read_json,atomic_json,sha256,utcnow

ENTRY='''"""PTCG actor entry: fixed weights, own observation only, no networking."""
import json
import sys
from pathlib import Path
try:
    ROOT=Path(__file__).resolve().parent
except NameError:
    ROOT=Path('/kaggle_simulations/agent')
if not (ROOT/'agent-spec.json').exists():
    ROOT=Path.cwd()
sys.path.insert(0,str(ROOT))
from ptcg_lab.rl_agent import TransformerRLAgent
with (ROOT/'agent-spec.json').open(encoding='utf-8') as f:
    _SPEC=json.load(f)
_POLICY=TransformerRLAgent(_SPEC,0,0)
def agent(observation):
    return _POLICY.choose(observation)
'''


def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--name',required=True);a=p.parse_args()
    if not all(c.isalnum() or c=='-' for c in a.name):raise ValueError('Unsafe package name')
    spec=read_json(a.spec);folder=ROOT/spec['path']
    if spec['kind']!='transformer_rl' or sha256(folder/'model.pt')!=spec['model_sha256']:raise ValueError('Candidate mismatch')
    destination=ROOT/'releases/validation'/a.name
    destination.mkdir(parents=True,exist_ok=False);stage=destination/'contents';stage.mkdir()
    shutil.copytree(folder,stage/'bundle',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    (stage/'main.py').write_text(ENTRY,encoding='utf-8')
    shutil.copy2(folder/'deck.csv',stage/'deck.csv')
    for name in ('__init__.py','agents.py','public_agent.py','rl_agent.py','rl_core.py','token_features.py','bc_features.py','replay_data.py','io.py'):
        dest=stage/'ptcg_lab'/name;dest.parent.mkdir(exist_ok=True);shutil.copy2(ROOT/'ptcg_lab'/name,dest)
    shutil.copytree(ROOT/'ptcg_lab/public_cg',stage/'ptcg_lab/public_cg',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    packaged=dict(spec,path='bundle');atomic_json(stage/'agent-spec.json',packaged)
    hashes={p.relative_to(stage).as_posix():sha256(p) for p in sorted(stage.rglob('*')) if p.is_file()}
    atomic_json(stage/'package-manifest.json',{'format':'ptcg-transformer-v3','model_sha256':spec['model_sha256'],
        'source_spec_sha256':sha256(a.spec),'files':hashes,'requires':['numpy','torch'],'actor_runtime':'blind public metadata only',
        'scope':'Deployment validation artifact; no submission authorization implied by packaging'})
    archive=destination/'submission.tar.gz'
    with archive.open('wb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',mtime=0,filename='') as gz,tarfile.open(fileobj=gz,mode='w') as tf:
        for path in sorted(stage.rglob('*')):
            if not path.is_file():continue
            if path.is_symlink():raise ValueError('Symlinks forbidden')
            info=tf.gettarinfo(str(path),arcname=path.relative_to(stage).as_posix())
            info.mtime=0;info.uid=info.gid=0;info.uname=info.gname='';info.mode=0o644
            with path.open('rb') as f:tf.addfile(info,f)
    atomic_json(destination/'release-candidate.json',{'created_at':utcnow(),'archive_sha256':sha256(archive),
        'archive_bytes':archive.stat().st_size,'model_sha256':spec['model_sha256'],'training_run_id':spec['training_run_id'],
        'manifest_sha256':sha256(stage/'package-manifest.json'),'competition_submit_permitted':False,
        'linux_cpu_verified':False,'final_strength_verified':False,'qualification':'unqualified_validation_only'})
    print(archive)


if __name__=='__main__':main()
