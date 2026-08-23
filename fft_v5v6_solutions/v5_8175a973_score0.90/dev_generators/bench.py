import os, time
os.sched_setaffinity(0, {0})
import numpy as np
from harness import run, ref_np

def bench(L, B, m, mode=None, reps=3):
    rng = np.random.default_rng(5)
    x0 = rng.standard_normal((B,L,L,L)) + 1j*rng.standard_normal((B,L,L,L))
    c = 0.1*(rng.standard_normal((B,L,L,L)) + 1j*rng.standard_normal((B,L,L,L)))
    best = 1e9
    for _ in range(reps):
        t0 = time.perf_counter(); run(L, B, m, x0, c, mode=mode); t1 = time.perf_counter()
        best = min(best, t1-t0)
    bestn = 1e9
    for _ in range(reps):
        t0 = time.perf_counter(); ref_np(L, B, m, x0, c); t1 = time.perf_counter()
        bestn = min(bestn, t1-t0)
    per = best/(B*m)*1e6
    pern = bestn/(B*m)*1e6
    print(f"L={L:3d} B={B:5d} m={m:3d} mode={mode} mine={best*1e3:9.2f}ms np={bestn*1e3:9.2f}ms per_vol_iter={per:9.2f}us np={pern:9.2f}us speedup={bestn/best:6.2f}x")
    return best, bestn

for L, B in ((6, 4096), (8, 2048), (13, 512), (17, 256), (23, 128), (36, 32), (45, 16), (64, 8)):
    bench(L, B, 10)
print()
print("modes for 36/45:")
bench(36, 32, 10, mode=0); bench(36, 32, 10, mode=1)
bench(45, 16, 10, mode=0); bench(45, 16, 10, mode=1)
