import sys, numpy as np
sys.path.insert(0, '/private/tmp/claude-501/-Users-wdetmold-MIT-Dropbox-William-Detmold-2026-problems/3e8ea67f-c989-4b31-8f6e-316caac734df/scratchpad/fft_recon/sandbox/workdir')
import base, solution

def blocks(out, Bs):
    SIZES=(6,8,13,17,23,36,45,64)
    res={}; off=0
    for L in SIZES:
        n=Bs[L]*L**3
        res[(L,'one')]=out[off:off+n]; off+=n
        res[(L,'fin')]=out[off:off+n]; off+=n
    return res

def check(seed, Bs, ms):
    args=[seed]+[Bs[L] for L in (6,8,13,17,23,36,45,64)]+[ms[L] for L in (6,8,13,17,23,36,45,64)]
    o1=base.transform(*args)
    o2=solution.transform(*args)
    assert o1.shape==o2.shape and o2.dtype==np.complex128, (o1.shape,o2.shape,o2.dtype)
    b1=blocks(o1,Bs); b2=blocks(o2,Bs)
    ok=True
    for k in b1:
        x,y=b1[k],b2[k]
        if x.size==0: continue
        e=np.linalg.norm(y-x)/max(np.linalg.norm(x),1e-300)
        lim = 2e-15 if k[1]=='one' else 1e-4
        flag = "OK " if e<lim else "FAIL"
        if e>=lim: ok=False
        print(f"  {k}: rel={e:.3e} {flag}")
    return ok

Bs={6:3,8:3,13:2,17:2,23:2,36:2,45:2,64:2}
ms={6:1,8:1,13:1,17:1,23:1,36:1,45:1,64:1}
print("m=1:"); ok1=check(12345,Bs,ms)
ms={L:7 for L in Bs}
print("m=7:"); ok2=check(999,Bs,ms)
print("ALL OK" if (ok1 and ok2) else "FAILURES PRESENT")
