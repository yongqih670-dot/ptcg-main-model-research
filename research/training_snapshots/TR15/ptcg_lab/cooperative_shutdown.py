"""Turn an owned process SIGTERM into a checked checkpoint stop, not an async exception."""
import signal
from .training_control import ControlledTrainingStop


class CooperativeShutdown:
    def __init__(self):
        self.requested=False;self.previous={}
        for name in (signal.SIGTERM,signal.SIGINT):
            self.previous[name]=signal.getsignal(name);signal.signal(name,self.request)
    def request(self,*_):self.requested=True
    def check(self):
        if self.requested:raise ControlledTrainingStop('Owned process termination requested; preserve complete checkpoint')
    def restore(self):
        for name,handler in self.previous.items():signal.signal(name,handler)
