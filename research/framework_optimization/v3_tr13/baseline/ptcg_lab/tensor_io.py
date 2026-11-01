"""One atomic serialization route for learner, rollout and league tensors."""
import os
from pathlib import Path
import uuid
from .phase_timing import timed
from .bounded_io import environment_context
from .bounded_read import environment_read_context


@timed('tensor_serialization_and_publication')
def atomic_torch_save(path,value,*,write_context=None,replace=True):
    import torch
    context=write_context or environment_context()
    if context is not None:return context.atomic_write(path,lambda stream:torch.save(value,stream),replace=replace)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.partial')
    with temporary.open('xb') as stream:
        torch.save(value,stream);stream.flush();os.fsync(stream.fileno())
    if replace:os.replace(temporary,path)
    else:os.link(temporary,path);temporary.unlink()
    return {'path':str(path),'atomic_publication':True,'application_paced':False}


@timed('tensor_loading')
def load_torch(path,*,read_context=None,**options):
    import torch
    context=read_context or environment_read_context()
    if context is None:
        if os.name=='nt' and isinstance(path,(str,os.PathLike)) and not Path(path).exists():
            from .evidence_tiering import read_missing
            original=read_missing(path)
            if original is not None:
                import io
                if options.get('mmap'):raise ValueError('Tiered original evidence cannot use a native mmap')
                return torch.load(io.BytesIO(original),**options)
        return torch.load(path,**options)
    if options.get('mmap'):raise ValueError('Native mmap would bypass paced checkpoint reads')
    with context.open(path) as stream:return torch.load(stream,**options)
