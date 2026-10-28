"""Write a repeatable, credential-free environment and engine manifest."""
import contextlib
import importlib.metadata
import io
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ptcg_lab.io import atomic_json, sha256, utcnow

stream = io.StringIO()
with contextlib.redirect_stdout(stream):
    import kaggle_environments
    from kaggle_environments.envs.cabt import cabt

sdk = ROOT / 'data/competition/sample_submission/sample_submission/sample_submission'
installed = Path(cabt.__file__).parent
paths = {
    'official_package_wrapper': installed / 'cabt.py',
    'official_package_spec': installed / 'cabt.json',
    'installed_windows_engine': installed / 'cg/cg.dll',
    'downloaded_windows_engine': sdk / 'cg/cg.dll',
    'downloaded_linux_engine': sdk / 'cg/libcg.so',
    'competition_card_table': ROOT / 'data/competition/EN_Card_Data_R2_full.csv',
}
# Resolve the exact card-table location without assuming archive nesting.
if not paths['competition_card_table'].exists():
    paths['competition_card_table'] = next((ROOT/'data/competition').rglob('EN_Card_Data_R2_full.csv'))
hashes = {k: {'path': str(v), 'sha256': sha256(v)} for k,v in paths.items()}
packages = sorted((d.metadata['Name'], d.version) for d in importlib.metadata.distributions())
report = {
    'at': utcnow(), 'python': sys.executable, 'python_version': sys.version,
    'platform': platform.platform(), 'packages': dict(packages), 'files': hashes,
    'engines_identical': hashes['installed_windows_engine']['sha256'] == hashes['downloaded_windows_engine']['sha256'],
    'cabt_specification': cabt.specification,
    'dependency_scope': 'CABT-only minimal dependencies; unrelated optional game environments are intentionally absent',
    'kaggle_auth_read_check': '2026-10-23 global CLI successfully listed target competition data; not proof of submission/rules acceptance',
    'global_cuda_smoke': '2026-10-23 global torch 2.11.0+cu128 finite CUDA matmul on RTX 5070 Ti Laptop; isolated GPU runtime remains pending',
    'isolated_torch_package': dict(packages).get('torch'),
    'linux_validation': 'pending: WSL unavailable and Docker not found',
}
out = ROOT / 'research/environment'
atomic_json(out/'manifest.json', report)
(out/'requirements-cabt.lock').write_text('\n'.join(f'{name}=={version}' for name,version in packages)+'\n',encoding='utf-8')
(out/'import-diagnostics.txt').write_text(stream.getvalue(),encoding='utf-8')
print(json.dumps({'manifest':str(out/'manifest.json'), 'engines_identical':report['engines_identical'],'packages':len(packages)}))

