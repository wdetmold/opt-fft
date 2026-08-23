import sys, time, numpy as np
sys.path.insert(0, '/private/tmp/claude-501/-Users-wdetmold-MIT-Dropbox-William-Detmold-2026-problems/3e8ea67f-c989-4b31-8f6e-316caac734df/scratchpad/fft_recon/sandbox/workdir')
import solution

SIZES=(6,8,13,17,23,36,45,64)
Bs={6:8,8:8,13:4,17:4,23:2,36:2,45:1,64:1}
m=30
res={}
for rep in range(5):
    for L in SIZES:
        B=Bs[L]
        rng=np.random.default_rng(7+L)
        x0=(rng.standard_normal((B,L,L,L))+1j*rng.standard_normal((B,L,L,L)))
        c=0.1*(rng.standard_normal((B,L,L,L))+1j*rng.standard_normal((B,L,L,L)))
        solution._run(L,B,2,x0,c)
        t0=time.perf_counter()
        solution._run(L,B,m,x0,c)
        t1=time.perf_counter()
        per=(t1-t0)/m
        res.setdefault(L,[]).append(per)
print(f"{'L':>4} {'B':>3} {'min us/step':>12} {'ns/elem':>9}  all")
for L in SIZES:
    B=Bs[L]; per=min(res[L])
    print(f"{L:>4} {B:>3} {per*1e6:>12.2f} {per/(B*L**3)*1e9:>9.2f}  {[round(x*1e6,1) for x in res[L]]}")
