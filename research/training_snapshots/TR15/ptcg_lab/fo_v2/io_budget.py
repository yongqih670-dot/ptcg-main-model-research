"""One aggregate read budget shared by owner and archive threads."""
import hashlib
from pathlib import Path
import threading
import time


class ReadEnvelope:
    def __init__(self,rate,stop=lambda:None):
        if rate<=0:raise ValueError('Positive I/O rate required')
        self.rate=rate;self.stop=stop;self.next=time.monotonic();self.total=0;self.lock=threading.Lock()
    def consume(self,count):
        if type(count) is not int or count<0:raise ValueError('Invalid byte count')
        with self.lock:
            self.total+=count;target=max(self.next,time.monotonic())+count/self.rate;self.next=target
        while (left:=target-time.monotonic())>0:self.stop();time.sleep(min(.05,left))
    def open(self,path):
        from ptcg_lab.bounded_read import PacedReader
        # This is a bounded local read operation, not a training job deadline.
        return PacedReader(Path(path).open('rb',buffering=0),self,time.time()+1800)
    def sha256(self,path):
        h=hashlib.sha256()
        with self.open(path) as stream:
            for data in iter(lambda:stream.read(2**20),b''):h.update(data)
        return h.hexdigest()
