"""One-time export of public official card/attack metadata, not battle data."""
import contextlib
import ctypes
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ptcg_lab.io import atomic_json, sha256

with contextlib.redirect_stdout(io.StringIO()):
    from kaggle_environments.envs.cabt.cg.sim import get_lib
lib = get_lib(2)
lib.AllCard.restype = ctypes.c_char_p
lib.AllAttack.restype = ctypes.c_char_p
data = {'cards': json.loads(lib.AllCard()), 'attacks': json.loads(lib.AllAttack())}
path = ROOT / 'ptcg_lab/public_cg/metadata.json'
atomic_json(path, data)
print(json.dumps({'cards': len(data['cards']), 'attacks': len(data['attacks']), 'sha256': sha256(path),
                  'card_example': data['cards'][0]}))
