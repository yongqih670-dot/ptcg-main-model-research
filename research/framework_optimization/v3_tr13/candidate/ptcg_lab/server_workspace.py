"""Owned hybrid workspace admission; no traversal or cleanup of external data."""
import json
import os
from pathlib import Path
import shutil
import time

_capacity_cache={}


def observed_bytes(root,profile):
    """10s metadata window, with256MiB headroom for bounded concurrent writers.

    Future HDD2 jobs admit one heavy writer (<=4MiB/s), a <=4MiB/s transfer,
    and bounded control/log writers.256MiB exceeds their10s growth envelope.
    Migration separately reserves its entire immutable source before copying.
    Legacy jobs retain their prior exact scan behavior.
    """
    if profile is None:return logical_bytes(root),0,False
    key=(os.getpid(),str(root),profile['authority_canonical_sha256']);now=time.monotonic()
    cached=_capacity_cache.get(key)
    if cached and 0<=now-cached['at']<10:
        return cached['used']+256*2**20,cached['retained'],True
    retained=sum(logical_bytes(owned_root(previous,profile['project_id'])) for previous in profile['retained_roots'])
    used=logical_bytes(root)+retained
    _capacity_cache[key]={'at':now,'used':used,'retained':retained}
    return used+256*2**20,retained,False


def owned_root(root,project_id):
    root=Path(root)
    if root.is_symlink():raise ValueError('Workspace root cannot be a symlink')
    root=root.resolve()
    if hasattr(os,'getuid') and root.stat().st_uid!=os.getuid():raise ValueError('Workspace directory UID differs')
    if os.name=='posix' and root.stat().st_mode&0o077:raise ValueError('Workspace root is not private to its owner')
    marker=root/'.ptcg-owner.json'
    if marker.is_symlink():raise ValueError('Workspace owner marker cannot be a symlink')
    if hasattr(os,'getuid') and marker.stat().st_uid!=os.getuid():raise ValueError('Workspace marker UID differs')
    owner=json.loads(marker.read_text(encoding='utf-8'))
    uid=os.getuid() if hasattr(os,'getuid') else owner['uid']
    if owner!={'project_id':project_id,'uid':uid,'root':str(root)}:raise ValueError('Workspace ownership differs')
    return root


def inside(root,path):
    root=Path(root).resolve();path=Path(path)
    if '..' in path.parts or path.is_symlink():raise ValueError('Unsafe workspace path')
    if not path.is_absolute():path=root/path
    try:parts=path.relative_to(root).parts
    except ValueError:raise ValueError('Path is outside owned workspace')
    current=root
    for part in parts:
        current=current/part
        if current.is_symlink():raise ValueError('Writable path traverses a symlink')
        if current.exists() and hasattr(os,'getuid') and current.stat().st_uid!=os.getuid():
            raise ValueError('Writable path UID differs')
    path=path.resolve()
    if root not in path.parents:raise ValueError('Path is outside owned workspace')
    return path


def logical_bytes(root):
    # Do not follow Python venv interpreter symlinks into the system runtime.
    total=0
    for directory,dirs,files in os.walk(root,followlinks=False):
        for name in files+dirs:
            path=Path(directory)/name
            try:
                if path.is_symlink() or path.is_file():total+=path.lstat().st_size
            except FileNotFoundError:pass
    return total


def admit(root,growth,hard_max=None,free_reserve=20*2**30):
    if type(growth) is not int or growth<0:raise ValueError('Invalid growth forecast')
    from .storage_profile import load
    profile=load(root);ceiling=profile['hard_max_bytes'] if profile else 10*2**30
    hard_max=ceiling if hard_max is None else hard_max
    if not 0<hard_max<=ceiling or free_reserve<20*2**30:raise ValueError('Hybrid capacity boundary relaxed')
    used,retained,cached=observed_bytes(root,profile);free=shutil.disk_usage(root).free
    if used+growth>hard_max:raise OSError('Hybrid workspace capacity reached')
    if free-growth<free_reserve:raise OSError('Hybrid filesystem reserve reached')
    return {'workspace_bytes':used,'free_bytes':free,'forecast_growth_bytes':growth,
            'hard_max_bytes':hard_max,'free_reserve_bytes':free_reserve,'retained_old_bytes':retained,
            'metadata_cache_used':cached,'conservative_growth_headroom_bytes':256*2**20 if profile else 0}
