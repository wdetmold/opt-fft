import sys, time, numpy as np, ctypes
sys.path.insert(0, '/private/tmp/claude-501/-Users-wdetmold-MIT-Dropbox-William-Detmold-2026-problems/3e8ea67f-c989-4b31-8f6e-316caac734df/scratchpad/fft_recon/sandbox/workdir')
import solution

SIZES=(6,8,13,17,23,36,45,64)
Bs={6:8,8:8,13:4,17:4,23:2,36:2,45:1,64:1}
m=50
print(f"{'L':>4} {'B':>3} {'ours/step(us)':>14} {'ns/elem':>9}")
for L in SIZES:
    B=Bs[L]
    rng=np.random.default_rng(7+L)
    x0=(rng.standard_normal((B,L,L,L))+1j*rng.standard_normal((B,L,L,L)))
    c=0.1*(rng.standard_normal((B,L,L,L))+1j*rng.standard_normal((B,L,L,L)))
    # warm
    solution._run(L,B,3,x0,c)
    reps = max(1, int(0.6/ (m*B*L**3*6e-9) ))
    t0=time.perf_counter()
    for _ in range(reps):
        solution._run(L,B,m,x0,c)
    t1=time.perf_counter()
    per=(t1-t0)/reps/m
    print(f"{L:>4} {B:>3} {per*1e6:>14.2f} {per/(B*L**3)*1e9:>9.2f}")
