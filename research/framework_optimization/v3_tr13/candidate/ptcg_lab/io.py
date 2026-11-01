import hashlib
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


def _local_reader(path):
    if os.name!='nt':return Path(path).open('rb')
    # Request delete sharing and close promptly. Windows filter drivers or
    # other readers can still defer replacement; the publisher retries below.
    import ctypes,msvcrt
    from ctypes import wintypes as w
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.CreateFileW.argtypes=[w.LPCWSTR,w.DWORD,w.DWORD,ctypes.c_void_p,w.DWORD,w.DWORD,w.HANDLE]
    kernel.CreateFileW.restype=w.HANDLE;kernel.CloseHandle.argtypes=[w.HANDLE]
    handle=kernel.CreateFileW(os.fspath(path),0x80000000,7,None,3,0x80,None)
    if handle==ctypes.c_void_p(-1).value:raise ctypes.WinError(ctypes.get_last_error())
    try:descriptor=msvcrt.open_osfhandle(handle,os.O_RDONLY|os.O_BINARY)
    except BaseException:kernel.CloseHandle(handle);raise
    try:return os.fdopen(descriptor,'rb')
    except BaseException:os.close(descriptor);raise


def _replace_json(temporary,path,timeout):
    if not 0<=timeout<=2:raise ValueError('Local JSON replacement retry exceeds its bounded scope')
    deadline=time.monotonic()+timeout;delay=.01
    while True:
        try:os.replace(temporary,path);return
        except PermissionError:
            # Windows editors/readers can briefly omit FILE_SHARE_DELETE.
            # Never change ACLs, delete the destination, or publish partial JSON.
            if os.name!='nt' or time.monotonic()>=deadline:raise
            time.sleep(min(delay,max(0,deadline-time.monotonic())));delay=min(.1,delay*2)


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    if os.environ.get('PTCG_PROJECT_ROOT'):
        from .bounded_read import environment_read_context
        with environment_read_context().open(path) as stream:return json.loads(stream.read().decode('utf-8'))
    with _local_reader(path) as stream:raw=stream.read()
    return json.loads(raw.decode('utf-8'))


def atomic_json(path, value, *, replace_timeout_seconds=2.0):
    path = Path(path)
    if os.environ.get('PTCG_PROJECT_ROOT'):
        from .bounded_io import environment_context
        context=environment_context()
        encoder=json.JSONEncoder(ensure_ascii=False,indent=2,allow_nan=False)
        blocks=[];size=1
        for block in encoder.iterencode(value):
            data=block.encode('utf-8');size+=len(data)
            if size>context.max_file_bytes:raise ValueError('JSON exceeds the unchanged bounded file limit')
            blocks.append(data)
        payload=b''.join([*blocks,b'\n'])
        return context.atomic_write(path,lambda stream:stream.write(payload),expected_bytes=size)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    _replace_json(temporary,path,replace_timeout_seconds)


def sha256(path):
    if os.environ.get('PTCG_PROJECT_ROOT'):
        from .bounded_read import environment_read_context
        return environment_read_context().sha256(path)
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_series(path, recover_tail=False):
    """Recover only an unfinished last append, never hide invalid complete rows."""
    path = Path(path)
    if not path.exists():
        return []
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        if not recover_tail:
            raise ValueError("Incomplete final JSONL append")
        end = data.rfind(b"\n") + 1
        atomic_json(path.with_suffix(".recovery.json"), {
            "at": utcnow(), "discarded_uncommitted_bytes": len(data) - end})
        with path.open("r+b") as f:
            f.truncate(end)
        data = data[:end]
    rows = [json.loads(line) for line in data.splitlines()]
    ids = [r["series_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate series IDs: do not silently double-count")
    return rows


def append_series(path, value):
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())

