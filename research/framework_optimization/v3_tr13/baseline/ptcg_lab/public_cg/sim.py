"""Public card catalog only. No battle pointer, hidden state or search API."""
from pathlib import Path
import json


class Catalog:
    def __init__(self):
        self.data = json.loads(Path(__file__).with_name('metadata.json').read_text(encoding='utf-8'))

    def AllCard(self):
        return json.dumps(self.data['cards']).encode()

    def AllAttack(self):
        return json.dumps(self.data['attacks']).encode()


lib = Catalog()
