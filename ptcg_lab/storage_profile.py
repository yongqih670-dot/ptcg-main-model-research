"""Forward-authorized HDD2 storage; old immutable tickets keep their old caps."""
import hashlib
import json
import os
from pathlib import Path

OLD='OLD_READONLY_ROOT/ptcg-hybrid-v3'
NEW='SERVER_PROJECT_ROOT'
PROJECT='PROJECT_ID'
GIB=2**30


def validate(value,root):
    if value.get('schema')=='ptcg_HDD2_TR12_storage_v2':
        authority=value.get('authority',{})
        digest=hashlib.sha256(json.dumps(authority,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if (str(root)!=NEW or value.get('project_id')!=PROJECT or value.get('root')!=NEW
                or value.get('hard_max_bytes')!=80*GIB or value.get('warning_bytes')!=64*GIB
                or value.get('free_reserve_bytes')!=20*GIB or value.get('retained_roots')!=[OLD]
                or value.get('authority_file_sha256')!='971709f6510a8c83527dd42175a9459e12861c00544f489067765dc6d2412ff4'
                or digest!='c28ba45a6e163ce620bc6399e833cd5bfc8165b211a5e359b8e16f054077d6eb'
                or value.get('authority_canonical_sha256')!=digest):
            raise ValueError('TR12 storage authority or ownership scope differs')
        return value
    if (str(root)!=NEW or value.get('schema')!='ptcg_HDD2_storage_v1'
            or value.get('project_id')!=PROJECT or value.get('root')!=NEW
            or value.get('hard_max_bytes')!=50*GIB or value.get('free_reserve_bytes')!=20*GIB
            or value.get('warning_bytes')!=40*GIB or value.get('retained_roots')!=[OLD]):
        raise ValueError('Unrecognized authorized storage profile')
    authority=value['authority'];encoded=json.dumps(authority,sort_keys=True,separators=(',',':')).encode()
    if hashlib.sha256(encoded).hexdigest()!=value.get('authority_canonical_sha256'):
        raise ValueError('Storage authority digest differs')
    if (authority.get('authorized_target_root')!=NEW or authority.get('source_root_read_only')!=OLD
            or authority.get('server_total_project_working_set_hard_GiB')!=50
            or authority.get('server_free_disk_reserve_GiB')!=20
            or authority.get('old_copy_preserved_until_verified_and_evidence_archived') is not True):
        raise ValueError('Storage authority scope differs')
    return value


def load(root):
    root=Path(root);path=root/'.ptcg-storage.json'
    if not path.exists():return None
    if path.is_symlink() or path.resolve()!=path or path.stat().st_uid!=os.getuid():
        raise ValueError('Storage profile ownership differs')
    return validate(json.loads(path.read_text()),root)


def ticket_ceiling(ticket):
    profile=ticket.get('storage_profile')
    if profile is None:return 10*GIB
    validate(profile,ticket['project_root'])
    return profile['hard_max_bytes']
