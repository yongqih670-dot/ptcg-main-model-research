"""Prepare a reviewed MIT public baseline; preserve original dataset bytes."""
import shutil
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ptcg_lab.io import sha256,atomic_json,utcnow

source=ROOT/'research/public_sources/ppo/original'
target=ROOT/'agents/public/ppo'
if target.exists():
    raise RuntimeError('PPO destination already exists; do not overwrite frozen candidate')
shutil.copytree(source,target)
path=target/'ptcg/rl/search_agent.py'
text=path.read_text(encoding='utf-8')
old='torch.load(model_path, map_location="cpu")'
assert text.count(old)==1
path.write_text(text.replace(old,'torch.load(model_path, map_location="cpu", weights_only=True)'),encoding='utf-8')
path=target/'ptcg/effects.py'
text=path.read_text(encoding='utf-8')
assert text.count('json.load(open(_JSON))')==1
path.write_text(text.replace('json.load(open(_JSON))','json.load(open(_JSON, encoding="utf-8"))'),encoding='utf-8')
atomic_json(target/'provenance.json',{
    'id':'ppo','lineage':'dipam_unown_gradiant_ppo','at':utcnow(),
    'source_url':'https://www.kaggle.com/datasets/cristobaltudela/ptcg-playground-public-baseline',
    'upstream':'https://github.com/dipamc/kaggle-ptcg-ai-battle','license':'MIT (LICENSE retained)',
    'changes':['explicit torch.load weights_only=True','UTF-8 JSON encoding for Windows'],
    'adapter':'public catalog only, no search; independent trackers; per-game module reset; original seat preserved',
    'original_files':{str(p.relative_to(source)):sha256(p) for p in source.rglob('*') if p.is_file()},
    'prepared_files':{str(p.relative_to(target)):sha256(p) for p in target.rglob('*') if p.is_file()},
    'status':'not_yet_runtime_validated',
})
print('Prepared public PPO with original weights and documented runtime-only patches')
