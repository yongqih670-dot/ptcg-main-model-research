"""Verify the curated public files without downloading or loading training data."""
from pathlib import Path
import json,hashlib
root=Path(__file__).resolve().parents[1]
rows=json.loads((root/'manifests/curated-files.json').read_text(encoding='utf-8'))
total=0
for row in rows:
    p=(root/row['path']).resolve()
    if root not in p.parents or not p.is_file():raise SystemExit('Missing/unsafe path: '+row['path'])
    data=p.read_bytes()
    if len(data)!=row['bytes'] or hashlib.sha256(data).hexdigest()!=row['sha256']:raise SystemExit('Hash mismatch: '+row['path'])
    total+=len(data)
print(json.dumps({'verified_files':len(rows),'verified_bytes':total,'training_data_downloaded':False}))
