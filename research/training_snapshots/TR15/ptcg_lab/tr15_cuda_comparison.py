import torch
def compare_tensors(reference,actual,repeat):
    rows=[];okay=True
    def visit(a,b,c,path):
        nonlocal okay
        if torch.is_tensor(a):
            a=a.detach().cpu();b=b.detach().cpu();c=c.detach().cpu()
            if a.shape!=b.shape or a.dtype!=b.dtype:raise ValueError('GPU output tensor format changed')
            if not a.is_floating_point():
                if not torch.equal(a,b):okay=False
                return
            if not torch.isfinite(a).all() or not torch.isfinite(b).all() or not torch.isfinite(c).all():
                okay=False;return
            difference=float((a.double()-b.double()).abs().max()) if a.numel() else 0
            control=float((a.double()-c.double()).abs().max()) if a.numel() else 0
            scale=max(1.,float(a.abs().max())) if a.numel() else 1.
            allowed=min(5e-6, max(8*control,8*torch.finfo(a.dtype).eps*scale))
            if difference>allowed:okay=False
            if difference or control:rows.append({'path':path,'resume_max_abs':difference,'repeat_max_abs':control,'allowed_abs':allowed})
        elif isinstance(a,dict):
            if set(a)!=set(b) or set(a)!=set(c):raise ValueError('GPU output dictionary changed')
            for k in a:visit(a[k],b[k],c[k],path+'/'+str(k))
        elif isinstance(a,(list,tuple)):
            if len(a)!=len(b) or len(a)!=len(c):raise ValueError('GPU output list changed')
            for i,(x,y,z) in enumerate(zip(a,b,c)):visit(x,y,z,path+'/'+str(i))
        elif a!=b or a!=c:okay=False
    visit(reference,actual,repeat,'')
    return {'passed':okay,'changed_tensors':len(rows),'maximum_resume_abs':max([r['resume_max_abs'] for r in rows],default=0.),
            'maximum_repeat_abs':max([r['repeat_max_abs'] for r in rows],default=0.),'tensor_differences':rows}
