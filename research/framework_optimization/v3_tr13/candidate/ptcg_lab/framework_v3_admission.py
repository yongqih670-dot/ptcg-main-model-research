"""Account legacy publications and receiver bytes in the Native capacity ledger.

Original owned-path, file-size, I/O pacing, physical-free and legacy admission
checks still execute. No cache is authorized merely by a time/rate assumption.
"""
import contextlib,functools,os
from pathlib import Path
from .fo_v2.durable import safe
from .server_workspace import inside

class WriteAdmission:
    def __init__(self,backend,root):
        import ptcg_lab.bounded_io as writer
        import ptcg_lab.archive_compressed as archive
        self.backend=backend;self.capacity=backend.capacity;self.root=Path(root).resolve();self.writer=writer;self.archive=archive;self.closed=False
        if not backend.root.is_relative_to(self.root) or not self.root.is_relative_to(self.capacity.root):raise ValueError('Capacity accounting owner/root differs')
        self.original_write=writer.WriteContext.atomic_write;self.original_receive=archive.receive_compressed_object;self.original_reuse=archive.reuse_compressed_object
        old_write=self.original_write;old_receive=self.original_receive;old_reuse=self.original_reuse
        @functools.wraps(old_write)
        def write(context,path,serialize,replace=True,expected_bytes=None):
            checked=inside(context.root,path)
            if not checked.is_relative_to(self.root):return old_write(context,path,serialize,replace,expected_bytes)
            maximum=context.max_file_bytes if expected_bytes is None else expected_bytes
            # Immutable publication briefly has temporary and final logical
            # names, exactly as TransactionalObjects' original double reserve.
            with self.publication(checked,maximum if replace else 2*maximum):return old_write(context,path,serialize,replace,expected_bytes)
        @functools.wraps(old_receive)
        def receive(root,run_id,epoch,sequence,relative,digest,size,chunks,**kw):
            from .archive_ack import safe_target
            target=safe_target(Path(root).resolve(),f'{run_id}/objects/{relative}.gz')
            # Windows extended spelling is normalized only for ledger scope;
            # the original receiver still performs all its own safe_target checks.
            text=str(target)
            if os.name=='nt' and text.startswith('\\\\?\\'):target=Path(text[4:])
            if not target.resolve().is_relative_to(self.root):return old_receive(root,run_id,epoch,sequence,relative,digest,size,chunks,**kw)
            with self.publication(target,2*(size+2**20)):return old_receive(root,run_id,epoch,sequence,relative,digest,size,chunks,**kw)
        @functools.wraps(old_reuse)
        def reuse(root,row,source_root,source_ack):
            from .archive_ack import safe_target
            target=safe_target(Path(root).resolve(),f"{row['run_id']}/objects/{row['object']}.gz")
            text=str(target)
            if os.name=='nt' and text.startswith('\\\\?\\'):target=Path(text[4:])
            if not target.resolve().is_relative_to(self.root):return old_reuse(root,row,source_root,source_ack)
            with self.publication(target,source_ack['storage']['bytes']):return old_reuse(root,row,source_root,source_ack)
        self.write=write;self.receive=receive;self.reuse=reuse;writer.WriteContext.atomic_write=write;archive.receive_compressed_object=receive;archive.reuse_compressed_object=reuse

    @contextlib.contextmanager
    def publication(self,path,maximum):
        path=Path(path).resolve()
        if not path.is_relative_to(self.root):raise ValueError('Tracked write escaped owner')
        before=path.stat().st_size if path.exists() else 0;token=self.capacity.reserve(maximum)
        try:yield
        except BaseException:
            # Original atomic writer intentionally retains failed partials.
            # A fresh full scan counts them before any next admission.
            self.capacity.finish(token,0);self.capacity.reconcile();raise
        else:
            actual=path.stat().st_size if path.exists() else 0
            self.capacity.finish(token,actual)
            if before:self.capacity.released(before)

    def close(self):
        if self.closed:return
        if self.writer.WriteContext.atomic_write is self.write:self.writer.WriteContext.atomic_write=self.original_write
        if self.archive.receive_compressed_object is self.receive:self.archive.receive_compressed_object=self.original_receive
        if self.archive.reuse_compressed_object is self.reuse:self.archive.reuse_compressed_object=self.original_reuse
        self.closed=True

def install(backend,root):
    if backend.write_admission is not None:raise ValueError('Capacity writer scope already installed')
    backend.write_admission=WriteAdmission(backend,root);return backend.write_admission
