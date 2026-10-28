"""Default-off AMP experiment; this module grants no training authorization."""
from contextlib import nullcontext
import torch

LOGP_ATOL=1e-4
VALUE_ATOL=2e-4


class Precision:
    def __init__(self, mode='fp32', device='cpu'):
        if mode not in ('fp32','bf16','fp16'):
            raise ValueError('Unknown precision')
        self.mode=mode; self.device=torch.device(device)
        if mode!='fp32' and self.device.type!='cuda':
            raise ValueError('AMP candidate requires controlled CUDA acceptance')
        if mode=='bf16' and not torch.cuda.is_bf16_supported():
            raise ValueError('BF16 unsupported')
        self.scaler=torch.amp.GradScaler('cuda',enabled=mode=='fp16')

    def autocast(self):
        if self.mode=='fp32':return nullcontext()
        return torch.autocast('cuda',dtype=torch.bfloat16 if self.mode=='bf16' else torch.float16)

    def backward(self,loss,optimizer):
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(optimizer)

    def step(self,optimizer):
        # Caller already checks finite unscaled gradients before this point.
        self.scaler.step(optimizer);self.scaler.update()

    def state_dict(self):
        return {'schema':1,'mode':self.mode,'scaler':self.scaler.state_dict()}

    def load_state_dict(self,state):
        if state['schema']!=1 or state['mode']!=self.mode:
            raise ValueError('Precision state cannot silently change on resume')
        self.scaler.load_state_dict(state['scaler'])


def parity(reference, candidate):
    finite=all(torch.isfinite(t).all().item() for t in (*reference,*candidate))
    if len(reference)!=3 or len(candidate)!=3 or any(a.shape!=b.shape for a,b in zip(reference,candidate)):
        raise ValueError('Parity shape differs')
    errors=[float((a.detach().double()-b.detach().double()).abs().max()) for a,b in zip(reference,candidate)]
    return {'passed':finite and errors[0]<=LOGP_ATOL and errors[1]<=VALUE_ATOL,
            'finite':finite,'logp_max_error':errors[0],'value_max_error':errors[1],
            'entropy_max_error':errors[2], 'logp_tolerance':LOGP_ATOL,'value_tolerance':VALUE_ATOL}
