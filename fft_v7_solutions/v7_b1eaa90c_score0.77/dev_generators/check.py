import sys, time, numpy as np
sys.path.insert(0, '/workdir')
import importlib.util
spec = importlib.util.spec_from_file_location("base", "/workdir/base.py")
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
import solution

SIZES=(6,8,13,17,23,36,45,64)
def blocks(out, Bs):
    res={}; off=0
    for L in SIZES:
        n=Bs[L]*L**3
        res[L]=(out[off:off+n], out[off+n:off+2*n]); off+=2*n
    return res

def check(seed, Bl, ml, verbose=True):
    Bs=dict(zip(SIZES,Bl)); ms=dict(zip(SIZES,ml))
    t0=time.perf_counter(); got=solution.transform(seed,*Bl,*ml); t1=time.perf_counter()
    ref=base.transform(seed,*Bl,*ml); t2=time.perf_counter()
    gb, rb = blocks(got,Bs), blocks(ref,Bs)
    gates={6:1e-4,8:3e-6,13:1e-9,17:1e-10,23:1e-10,36:1e-10,45:1e-10,64:1e-10}
    allok=True
    for L in SIZES:
        if Bs[L]==0: continue
        e1=np.linalg.norm(gb[L][0]-rb[L][0])/max(np.linalg.norm(rb[L][0]),1e-300)
        em=np.linalg.norm(gb[L][1]-rb[L][1])/max(np.linalg.norm(rb[L][1]),1e-300)
        ok = e1<1e-14 and em<gates[L]
        allok &= ok
        if verbose: print(f"  L={L:2d} one-step {e1:.2e} (<1e-14) m-step {em:.2e} (<{gates[L]:.0e}) {'OK' if ok else 'FAIL'}")
    print(f"solution {t1-t0:.3f}s base {t2-t1:.3f}s ratio {(t1-t0)/(t2-t1):.4f} {'ALL OK' if allok else 'FAILURES'}")
    return allok

if __name__=="__main__":
    check(7, (8,8,8,8,4,2,2,1), (5,5,5,5,5,5,5,5))
