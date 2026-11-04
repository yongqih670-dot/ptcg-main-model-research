"""Durable async outbox. ACK gates deletion, not the next PPO update.

    queue state is transactional; retries never fabricate ACKs. Transports must
    be cooperative and bounded by no-progress/identity checks supplied by the
    controlled runner, since Python cannot kill a blocked transport thread.
"""
from concurrent.futures import ThreadPoolExecutor
import copy
from pathlib import Path
import time
from .durable import read,atomic_json,canonical


def watermarks(*,space_headroom,reserved_next_commit,production_Bps,consumption_Bps,
               pending_bytes,oldest_age_seconds,lag_limit_seconds=600,minimum_growth_window=30):
    available=max(0,space_headroom-reserved_next_commit)
    growth=max(0,production_Bps-consumption_Bps)
    hard=available;soft=max(0,hard-max(reserved_next_commit,growth*minimum_growth_window))
    # An old lag or projected capacity exhaustion is a safe-point pause, never
    # a generic failure, and cannot be cured by increasing a fixed byte limit.
    action='pause_and_drain' if pending_bytes>=soft or oldest_age_seconds>=lag_limit_seconds else 'continue'
    if pending_bytes>hard:action='checkpointed_backpressure'
    return dict(soft_bytes=soft,hard_bytes=hard,action=action,
                estimated_seconds_to_capacity=None if growth==0 else max(0,(hard-pending_bytes)/growth),
                asynchronous_backup_lag_seconds=oldest_age_seconds)


class AsyncArchive:
    def __init__(self,path,run_id,epoch,transfer_batch,validate_ack,*,stop=lambda:None,
                 batch_bytes=64*2**20,batch_objects=16):
        self.path=Path(path);self.run_id=run_id;self.epoch=epoch;self.transfer=transfer_batch
        self.validate=validate_ack;self.stop=stop;self.batch_bytes=batch_bytes;self.batch_objects=batch_objects
        if not self.path.exists():atomic_json(self.path,dict(schema=2,run_id=run_id,epoch=epoch,rows=[]))
        self.rows=read(self.path)
        if (self.rows['run_id'],self.rows['epoch'])!=(run_id,epoch):raise ValueError('Outbox owner differs')
        names=set()
        for sequence,row in enumerate(self.rows['rows']):
            if row['sequence']!=sequence or row['run_id']!=run_id or row['lease_epoch']!=epoch or row['object'] in names:raise ValueError('Corrupt outbox sequence/identity')
            names.add(row['object'])
            if row['ack'] is not None:self.validate(row['ack'],row)
        self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='archive-v2');self.future=None;self.inflight=[]
        self.last_error=None
    def queue(self,refs):
        self.stop();by_path={r['object']:r for r in self.rows['rows']}
        for ref in refs:
            old=by_path.get(ref['path'])
            if old:
                if (old['sha256'],old['bytes'])!=(ref['sha256'],ref['bytes']):raise ValueError('Immutable queued object changed')
                continue
            row=dict(run_id=self.run_id,lease_epoch=self.epoch,sequence=len(self.rows['rows']),
                     object=ref['path'],sha256=ref['sha256'],bytes=ref['bytes'],queued_unix=time.time(),ack=None)
            self.rows['rows'].append(row);by_path[row['object']]=row
        atomic_json(self.path,self.rows)
    def reap(self):
        if not self.future or not self.future.done():return 0
        future=self.future;batch=self.inflight;self.future=None;self.inflight=[]
        try:acks=future.result()
        except Exception as exc:self.last_error=repr(exc);raise
        expected={r['sequence']:r for r in batch};accepted={}
        for ack in acks:
            seq=ack.get('sequence')
            if seq not in expected or seq in accepted:raise ValueError('Partial/duplicate/foreign batch ACK')
            row=expected[seq];self.validate(ack,row);accepted[seq]=ack
        if set(accepted)!=set(expected):raise ValueError('Batch ACK set incomplete')
        # A crash before this publication retries the same immutable data.
        for seq,ack in accepted.items():self.rows['rows'][seq]['ack']=ack
        atomic_json(self.path,self.rows);self.last_error=None;return len(accepted)
    def pump(self):
        self.stop();self.reap()
        if self.future:return False
        batch=[];size=0
        for row in self.rows['rows']:
            if row['ack'] is not None:continue
            if batch and (len(batch)>=self.batch_objects or size+row['bytes']>self.batch_bytes):break
            batch.append(copy.deepcopy(row));size+=row['bytes']
        if not batch:return False
        self.inflight=batch;self.future=self.pool.submit(self.transfer,batch,self.stop);return True
    def pending(self):
        rows=[r for r in self.rows['rows'] if r['ack'] is None]
        return dict(objects=len(rows),bytes=sum(r['bytes'] for r in rows),
                    oldest_age_seconds=max([time.time()-r['queued_unix'] for r in rows] or [0]))
    def drain(self,*,maximum_wait=600):
        started=time.monotonic()
        while self.pending()['objects']:
            if time.monotonic()-started>maximum_wait:raise TimeoutError('Bounded drain: checkpointed_backpressure')
            self.pump();self.stop();time.sleep(.005)
    def close(self):self.pool.shutdown(wait=True,cancel_futures=True)
