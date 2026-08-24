import sys, time, os
sys.path.insert(0, '/private/tmp/claude-501/-Users-wdetmold-MIT-Dropbox-William-Detmold-2026-problems/3e8ea67f-c989-4b31-8f6e-316caac734df/scratchpad/replay/workdir/dev'); sys.path.insert(0, '/private/tmp/claude-501/-Users-wdetmold-MIT-Dropbox-William-Detmold-2026-problems/3e8ea67f-c989-4b31-8f6e-316caac734df/scratchpad/replay/workdir')
import numpy as np
import solution, base

args = dict(seed=123, B6=3000, B8=1300, B13=300, B17=140, B23=55, B36=15, B45=8, B64=3,
            m6=30, m8=30, m13=30, m17=30, m23=30, m36=30, m45=30, m64=30)

# correctness spot check on small variant
small = dict(seed=7, B6=2, B8=2, B13=2, B17=2, B23=2, B36=2, B45=1, B64=1,
             m6=3, m8=4, m13=5, m23=2, m17=6, m36=2, m45=3, m64=2)
g = solution.transform(**small); r = base.transform(**small)
print("small check rel err:", np.linalg.norm(g-r)/np.linalg.norm(r))

for name, fn in (("mine", solution.transform), ("base", base.transform)):
    best = 9e9
    for _ in range(3 if name == "mine" else 2):
        t0 = time.perf_counter(); out = fn(**args); t1 = time.perf_counter()
        best = min(best, t1-t0)
    print(f"{name}: {best*1e3:9.1f} ms")
