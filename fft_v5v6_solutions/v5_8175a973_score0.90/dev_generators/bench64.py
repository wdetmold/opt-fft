import os, time
os.sched_setaffinity(0, {0})
import numpy as np
from harness import run
rng = np.random.default_rng(5)
L, B, m = 64, 8, 20
x0 = rng.standard_normal((B,L,L,L)) + 1j*rng.standard_normal((B,L,L,L))
c = 0.1*(rng.standard_normal((B,L,L,L)) + 1j*rng.standard_normal((B,L,L,L)))
run(L, B, 2, x0, c)  # warmup
t0 = time.perf_counter(); run(L, B, m, x0, c); t1 = time.perf_counter()
print(f"per_vol_iter {(t1-t0)/(B*m)*1e6:.1f} us")
