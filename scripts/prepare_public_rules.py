"""Extract reviewed literal code cells; never execute downloaded notebooks."""
import ast
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ptcg_lab.io import atomic_json, sha256, utcnow


def cells(name):
    path = next((ROOT / 'research/public_sources' / name).glob('*.ipynb'))
    return path, [''.join(c['source']) for c in json.loads(path.read_text(encoding='utf-8'))['cells']]


def literals(source):
    result = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                result[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    return result


def main():
    entries = []
    for name, source_cell, deck_cell, origin in [
        ('lucario', 11, 13, 'cggonzalezsc/blast-radius-mega-lucario-agent'),
        ('alakazam', 6, 6, 'woominyo/ptcg-alakazam-dusknoir-ready-to-run-agent'),
        ('archaludon', 1, 2, 'masamikobayashi/a-sample-archaludon-75-wr-vs-my-1300-starmie'),
    ]:
        notebook, content = cells(name)
        if name == 'alakazam':
            values = literals(content[6])
            source, deck = values['MAIN_SOURCE'], values['DECK_BYTES']
            assert hashlib.sha256(source.encode()).hexdigest() == values['SOURCE_SHA']
            assert hashlib.sha256(deck).hexdigest() == values['DECK_SHA']
        else:
            assert content[source_cell].startswith('%%writefile main.py\n')
            source = content[source_cell].split('\n', 1)[1]
            if name == 'lucario':
                deck = ('\n'.join(map(str, literals(content[deck_cell])['DECK'])) + '\n').encode()
            else:
                assert content[deck_cell].startswith('%%writefile deck.csv\n')
                deck = content[deck_cell].split('\n', 1)[1].encode()
        ast.parse(source)
        assert len(deck.split()) == 60
        folder = ROOT / 'agents/public' / name
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'main.py').write_bytes(source.encode())
        (folder / 'deck.csv').write_bytes(deck)
        entry = {
            'id': name, 'source_url': 'https://www.kaggle.com/code/' + origin,
            'notebook_sha256': sha256(notebook), 'main_sha256': sha256(folder / 'main.py'),
            'deck_sha256': sha256(folder / 'deck.csv'), 'extracted_at': utcnow(),
            'source_changes': 'none; notebook magic excluded',
            'runtime_adapter': 'static card API; isolated module per seat; round reset; no native search',
            'license_status': 'Public notebook explicitly offers reuse; no explicit license in CLI metadata; verify before redistribution',
            'validation': 'pending',
        }
        atomic_json(folder / 'provenance.json', entry)
        entries.append(entry)
    # Only public dataclasses, conversion and metadata functions. No search ABI.
    sdk = ROOT / 'data/competition/sample_submission/sample_submission/sample_submission/cg'
    adapter = ROOT / 'ptcg_lab/public_cg'
    adapter.mkdir(exist_ok=True)
    api = (sdk / 'api.py').read_text(encoding='utf-8').split('def search_begin(')[0]
    # The original SDK keeps an agent_ptr declaration immediately before search_begin.
    (adapter / 'api.py').write_text(api, encoding='utf-8')
    shutil.copy2(sdk / 'utils.py', adapter / 'utils.py')
    (adapter / '__init__.py').write_text('', encoding='utf-8')
    atomic_json(ROOT / 'research/public_sources/extraction-manifest.json', entries)
    print(json.dumps([{'id': x['id'], 'source': x['main_sha256']} for x in entries]))


if __name__ == '__main__':
    main()
