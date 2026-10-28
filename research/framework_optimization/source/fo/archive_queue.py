"""One bounded transfer worker on already immutable objects, no eviction API.

Production outbox remains the durable source of truth. After a crash, enumerate
its unacknowledged rows again. A partial transfer cannot fabricate an ACK.
Only callers at a writer safe point may freeze tensors; do not pass live model
state to a background serializer. Transport callbacks must enforce deadlines.
"""
import copy
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import numpy as np
import torch
from ptcg_lab.archive_ack import validate_ack


def freeze_state(value):
    # Copy each storage once, retaining aliases, strides and offsets. A plain
    # per-tensor clone silently changes tied-state and OrderedDict metadata.
    memo={};storages={}
    def freeze(obj):
        if id(obj) in memo:return memo[id(obj)]
        if isinstance(obj,torch.Tensor):
            if obj.layout!=torch.strided or obj.is_conj() or obj.is_neg():
                raise ValueError('Unsupported state tensor layout')
            source=obj.untyped_storage();key=(str(obj.device),source.data_ptr(),source.nbytes())
            if key not in storages:
                storages[key]=source.clone() if obj.device.type=='cpu' else source.cpu()
            result=torch.empty(0,dtype=obj.dtype,device='cpu').set_(
                storages[key],obj.storage_offset(),obj.size(),obj.stride())
        elif isinstance(obj,np.ndarray):result=obj.copy()
        elif isinstance(obj,dict):
            result=copy.copy(obj);result.clear();memo[id(obj)]=result
            result.update((k,freeze(v)) for k,v in obj.items())
            if hasattr(obj,'_metadata'):result._metadata=freeze(obj._metadata)
        elif isinstance(obj,list):
            result=[];memo[id(obj)]=result;result.extend(freeze(v) for v in obj)
        elif isinstance(obj,tuple):result=tuple(freeze(v) for v in obj)
        else:result=copy.deepcopy(obj)
        memo[id(obj)]=result
        return result
    return freeze(value)


class TransferQueue:
    def __init__(self, run_id, epoch, transfer, accept_ack, *, max_objects=8,
                 max_bytes=512*2**20, deadline, trace=None):
        if not 1<=max_objects<=32 or not 0<max_bytes<=512*2**20:
            raise ValueError('Transfer capacity exceeds candidate default envelope')
        self.run_id=run_id;self.epoch=epoch;self.transfer=transfer;self.accept_ack=accept_ack
        self.max_objects=max_objects;self.max_bytes=max_bytes;self.deadline=deadline;self.trace=trace
        self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='immutable-archive')
        self.pending={};self.closed=False;self.lock=threading.Lock()

    def reap(self):
        # ACK acceptance is performed by the owner thread, never the producer.
        done=[]
        with self.lock:
            for key,(row,future) in list(self.pending.items()):
                if not future.done():continue
                ack=future.result()  # retain failed entry and expose failure
                validate_ack(ack,self.run_id,self.epoch,row['object'],row['sha256'],row['bytes'],row['sequence'])
                self.accept_ack(ack)
                del self.pending[key];done.append(ack)
        return done

    def submit_batch(self,rows):
        self.reap()
        accepted=0
        with self.lock:
            for source in rows:
                if self.closed or time.time()>=self.deadline:raise TimeoutError('Queue closed/deadline')
                row=copy.deepcopy(source);key=row['sequence']
                if row['run_id']!=self.run_id or row['lease_epoch']!=self.epoch:
                    raise ValueError('Transfer run/lease differs')
                if type(row['bytes']) is not int or not 0<=row['bytes']<=128*2**20:
                    raise ValueError('Transfer object size differs')
                if key in self.pending:
                    if self.pending[key][0]!=row:raise ValueError('Conflicting duplicate transfer')
                    continue
                if (len(self.pending)>=self.max_objects or
                        sum(r['bytes'] for r,f in self.pending.values())+row['bytes']>self.max_bytes):break
                self.pending[key]=(row,self.pool.submit(self._transfer,row));accepted+=1
        return accepted

    def _transfer(self,row):
        if time.time()>=self.deadline:raise TimeoutError('Transfer deadline')
        if self.trace:
            with self.trace.span('archive.transfer_ACK',bytes=row['bytes']):
                return self.transfer(row,self.deadline)
        return self.transfer(row,self.deadline)

    def close(self):
        self.closed=True
        self.pool.shutdown(wait=True,cancel_futures=True)
        return self.reap()

    def __enter__(self):return self
    def __exit__(self,*exc):self.close()
