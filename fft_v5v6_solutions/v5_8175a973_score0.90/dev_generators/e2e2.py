import os, time, sys
import numpy as np
sys.path.insert(0, '/workdir')
import solution, base

def timeit(fn, params, reps):
    best = 1e9
    for _ in range(reps):
        t0 = time.perf_counter(); fn(*params); t1 = time.perf_counter()
        best = min(best, t1-t0)
    return best

cfgs = [
    ("balanced m=10", (7, 64, 48, 24, 16, 8, 4, 2, 1) + (10,)*8),
    ("m=30", (7, 64, 48, 24, 16, 8, 4, 2, 1) + (30,)*8),
    ("bigger batches m=10", (3, 512, 256, 96, 48, 24, 12, 6, 3) + (10,)*8),
]
for name, p in cfgs:
    tm = timeit(solution.transform, p, 3)
    tb = timeit(base.transform, p, 3)
    print(f"{name}: mine={tm:.3f}s base={tb:.3f}s ratio={tm/tb:.3f} speedup={tb/tm:.2f}x")
