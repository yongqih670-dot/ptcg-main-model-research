"""Static standard deck checks; engine acceptance remains separately required."""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cards = {c['cardId']: c for c in json.loads((ROOT/'ptcg_lab/public_cg/metadata.json').read_text(encoding='utf-8'))['cards']}
report = []
for folder in (ROOT/'agents/public').iterdir():
    deck = list(map(int, (folder/'deck.csv').read_text(encoding='utf-8-sig').split()))
    failures = []
    if len(deck) != 60: failures.append('not 60 cards')
    unknown = sorted(set(deck)-cards.keys())
    if unknown: failures.append(f'unknown card IDs: {unknown}')
    if unknown: raise ValueError(failures)
    names = Counter(cards[c]['name'] for c in deck if cards[c]['cardType'] != 5)
    if any(v > 4 for v in names.values()): failures.append('more than 4 non-basic-energy same-name cards')
    if sum(cards[c]['aceSpec'] for c in deck) > 1: failures.append('more than 1 ACE SPEC')
    if not any(cards[c]['basic'] and cards[c]['cardType'] == 0 for c in deck): failures.append('no basic Pokemon')
    report.append({'agent': folder.name, 'cards': len(deck), 'failures': failures,
                   'composition': [{'id': c, 'name': cards[c]['name'], 'count': n} for c,n in Counter(deck).items()]})
out = ROOT/'research/public_sources/deck-audit.json'
out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps([{'agent':r['agent'], 'failures':r['failures']} for r in report]))
assert not any(r['failures'] for r in report)
