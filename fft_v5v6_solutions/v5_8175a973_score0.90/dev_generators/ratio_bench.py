import os, time, sys
import numpy as np
sys.path.insert(0, '/workdir')
import solution, base

def best_of(fn, params, reps=4):
    b = 1e9
    for _ in range(reps):
        t0 = time.perf_counter(); fn(*params); t1 = time.perf_counter()
        b = min(b, t1-t0)
    return b

cfgs = [
    ("A m=10 small", (7, 64, 48, 24, 16, 8, 4, 2, 1) + (10,)*8),
    ("B m=30 small", (7, 64, 48, 24, 16, 8, 4, 2, 1) + (30,)*8),
    ("C m=10 big",   (3, 512, 256, 96, 48, 24, 12, 6, 3) + (10,)*8),
    ("D m=5  big",   (11, 256, 128, 64, 32, 16, 8, 4, 2) + (5,)*8),
    ("E m=50 tiny",  (5, 16, 12, 8, 6, 4, 2, 1, 1) + (50,)*8),
    ("F m=3",        (9, 128, 96, 48, 24, 12, 6, 3, 2) + (3,)*8),
]
for name, p in cfgs:
    tm = best_of(solution.transform, p)
    tb = best_of(base.transform, p)
    r = tm/tb
    score = 0.0 if r >= 1 else min(1.0, 0.1 + 0.9*(1-r)/(1-1/6.0))
    print(f"{name}: mine={tm:.3f}s base={tb:.3f}s ratio={r:.3f} speedup={1/r:.2f}x score~{score:.2f}")
