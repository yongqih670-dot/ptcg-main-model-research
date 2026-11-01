"""Equal local-only Win32 spelling adapter; keeps original owned-path guard."""
import os
from pathlib import Path
def normal_path(path):
    text=os.fspath(path)
    if os.name=='nt':
        if text.startswith('\\\\?\\UNC\\'):text='\\\\'+text[8:]
        elif text.startswith('\\\\?\\'):text=text[4:]
    return Path(text)
def install_archive_publication_adapter():
    import ptcg_lab.archive_compressed as compressed
    import ptcg_lab.archive_ack as raw
    from ptcg_lab.io import atomic_json
    def publish(path,value):return atomic_json(normal_path(path),value)
    compressed.atomic_json=publish;raw.atomic_json=publish

def local_outbox(job,epoch,*,optimizations=None,prefix_sha256):
    import json,re,hashlib
    from ptcg_lab.archive_outbox import ArchiveOutbox,IMMUTABLE
    class LocalOutbox(ArchiveOutbox):
        def _allowed(self,relative):
            owner=json.loads((job/'local-engineering-owner.json').read_text(encoding='utf-8'))
            if owner['parent_token']!=os.environ['V3_TOKEN'] or owner['scope']!='disposable fixed replay; no scientific authority':raise ValueError('Local fixture owner changed')
            if relative=='prefix.json':return hashlib.sha256((job/relative).read_bytes()).hexdigest()==prefix_sha256
            return bool(re.fullmatch(r'objects/(model|tail|data|manifest)-[0-9a-f]{64}\.bin',relative)) or bool(IMMUTABLE.fullmatch(relative))
    kw={} if optimizations is None else {'optimizations':optimizations}
    return LocalOutbox(job,job.name,epoch,**kw)
