"""Unactivated candidate for same-process immutable-reference hash reuse.

Production admission must first verify Linux change-time behavior and actual
checkpoint/RNG/ACK equivalence. Cold restores and pre-unlink full hashes must
continue to use fresh verification. This module changes no default route.
"""
from collections import OrderedDict
from collections.abc import MutableMapping
import os
from pathlib import Path
import sys

from .archive_ack import safe_target


def _getuid():
    return os.getuid() if hasattr(os,'getuid') else None


def _linux_runtime_supported():
    return sys.platform=='linux' and _getuid() is not None


def _fingerprint(path):
    s=path.stat()
    return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_uid,s.st_mode,s.st_ctime_ns)


class ImmutableReferenceHashCache(MutableMapping):
    """Existing checked_reference mapping API, with a change-time guard."""
    def __init__(self,root,maximum_entries=2048):
        if not _linux_runtime_supported():
            raise RuntimeError('Persistent reference reuse requires validated Linux change time')
        self.root=Path(root).absolute()
        if self.root.is_symlink() or self.root.resolve()!=self.root or self.root.stat().st_uid!=_getuid():
            raise ValueError('Cache root ownership or real path differs')
        if type(maximum_entries) is not int or not 1<=maximum_entries<=8192:
            raise ValueError('Reference cache capacity is outside its bounded envelope')
        self.maximum_entries=maximum_entries;self.pid=os.getpid()
        self.entries=OrderedDict();self.pending={}

    def _observe(self,key):
        if os.getpid()!=self.pid:raise RuntimeError('Hash cache cannot cross process identity')
        if not isinstance(key,tuple) or len(key)!=2 or key[0]!=str(self.root):
            raise ValueError('Reference cache key is outside its one owned root')
        path=safe_target(self.root,key[1]);fingerprint=_fingerprint(path)
        if fingerprint[4]!=_getuid():raise ValueError('Cached reference owner changed')
        return fingerprint

    def get(self,key,default=None):
        current=self._observe(key);old=self.entries.get(key)
        if old is not None and old[0]==current:
            self.entries.move_to_end(key)
            return (old[1][0],dict(old[1][1]))
        self.entries.pop(key,None)
        self.pending[key]=current
        while len(self.pending)>self.maximum_entries:self.pending.pop(next(iter(self.pending)))
        return default

    def __setitem__(self,key,value):
        current=self._observe(key)
        # get() runs before the full hash. Check ctime across the hash as well
        # as the existing six-field identity, including a forged old mtime.
        if self.pending.pop(key,None)!=current or value[0]!=current[:6]:
            raise ValueError('Reference changed during complete hash verification')
        self.entries[key]=(current,(value[0],dict(value[1])))
        self.entries.move_to_end(key)
        while len(self.entries)>self.maximum_entries:self.entries.popitem(last=False)

    def __getitem__(self,key):
        sentinel=object();value=self.get(key,sentinel)
        if value is sentinel:raise KeyError(key)
        return value

    def __delitem__(self,key):
        self.pending.pop(key,None);del self.entries[key]

    def __iter__(self):return iter(self.entries)
    def __len__(self):return len(self.entries)

    def __reduce__(self):
        raise TypeError('Reference cache is ephemeral and must never enter a checkpoint')
