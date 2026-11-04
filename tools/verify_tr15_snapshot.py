"""Verify selected frozen TR15 source bytes and syntax without executing them."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / 'manifests/tr14-tr15-publication.json').read_text(encoding='utf-8'))
prefix = manifest['public_snapshot'] + '/'
checked = python = substituted = 0
for source, row in manifest['source_files'].items():
    if not row['public_path'].startswith(prefix):
        continue
    path = (ROOT / row['public_path']).resolve()
    assert ROOT in path.parents and path.is_file()
    data = path.read_bytes()
    assert len(data) == row['public_bytes']
    assert hashlib.sha256(data).hexdigest() == row['public_sha256']
    if path.suffix == '.py':
        ast.parse(data, filename=row['public_path'])
        python += 1
    substituted += row['transformation'] != 'byte-identical'
    checked += 1
assert checked and python
print(json.dumps({'verified_snapshot_files': checked, 'python_syntax_checked': python,
                  'placeholder_substitutions_disclosed': substituted,
                  'source_code_executed': False, 'models_loaded': False}))
