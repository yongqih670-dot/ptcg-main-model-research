"""Cooperative pacing for the control channel's explicit file operations."""
import hashlib
import time


class BytePacer:
    def __init__(self,bytes_per_second,deadline,clock=time.monotonic,sleep=time.sleep):
        if not 0<bytes_per_second<=4*2**20:raise ValueError('Control I/O rate exceeds conservative allowance')
        self.rate=bytes_per_second;self.deadline=deadline;self.clock=clock;self.sleep=sleep
        self.started=clock();self.next_deadline=self.started;self.total=0
    def consume(self,count):
        if type(count) is not int or count<0:raise ValueError('Invalid paced byte count')
        if time.time()>=self.deadline:raise TimeoutError('Control I/O pacing deadline')
        # An idle interval grants no banked bandwidth for a later large burst.
        self.total+=count;target=max(self.next_deadline,self.clock())+count/self.rate
        self.next_deadline=target
        while True:
            delay=target-self.clock()
            if delay<=0:break
            if time.time()>self.deadline:raise TimeoutError('Control I/O pacing deadline')
            self.sleep(min(.05,delay))
        if time.time()>=self.deadline:raise TimeoutError('Control I/O pacing deadline')


def paced_sha256(path,pacer):
    digest=hashlib.sha256()
    with open(path,'rb') as stream:
        while block:=stream.read(256*1024):pacer.consume(len(block));digest.update(block)
    return digest.hexdigest()
