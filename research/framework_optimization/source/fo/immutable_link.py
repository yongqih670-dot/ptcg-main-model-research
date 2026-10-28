"""Optional same-filesystem publication for an atomic-replace-only writer.

Does not skip full source/destination hashes, ACK validation, retention or fresh
pre-unlink verification. Unsupported filesystems fall back to the original
paced copy. Never use for producers that modify published files in place.
"""
import errno
import os
from pathlib import Path
from ptcg_lab.archive_ack import safe_target,publication_path,sync_directory


def publish_link(source,destination,root):
    if root is None:raise ValueError('Hardlink candidate requires an owned writer root')
    root=Path(root).resolve()
    source=safe_target(root,Path(source).relative_to(root).as_posix())
    destination=safe_target(root,Path(destination).relative_to(root).as_posix())
    before=source.stat()
    if hasattr(os,'getuid') and before.st_uid!=os.getuid():raise ValueError('Source UID differs')
    try:os.link(publication_path(source),publication_path(destination))
    except OSError as exc:
        if exc.errno in (errno.EXDEV,errno.EPERM,errno.ENOTSUP,errno.EACCES):return False
        raise
    after=source.stat();stored=destination.stat()
    identity=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_uid,s.st_mode)
    if identity(before)!=identity(after) or identity(after)!=identity(stored):
        raise ValueError('Atomic-replace writer violated publication safe point')
    sync_directory(destination.parent)
    return True
