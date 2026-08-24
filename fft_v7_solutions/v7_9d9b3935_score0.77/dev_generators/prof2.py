import sys, time
sys.path.insert(0, '/private/tmp/claude-501/-Users-wdetmold-MIT-Dropbox-William-Detmold-2026-problems/3e8ea67f-c989-4b31-8f6e-316caac734df/scratchpad/replay/workdir/dev')
import numpy as np
import solution
from solution import _RUNS, _pp, _PD

Bs = {6:3000, 8:1300, 13:300, 17:140, 23:55, 36:15, 45:8, 64:3}
ms = {L:30 for L in Bs}
seed = 123

def timed_transform():
    t_rng = t_alloc = t_c = t_cat = 0.0
    outs = []
    for L in (6,8,13,17,23,36,45,64):
        B = Bs[L]
        t0 = time.perf_counter()
        rng_x = np.random.default_rng(seed + L)
        x0r = rng_x.standard_normal((B, L, L, L)); x0i = rng_x.standard_normal((B, L, L, L))
        rng_c = np.random.default_rng(1000 + L)
        cr = rng_c.standard_normal((B, L, L, L)); ci = rng_c.standard_normal((B, L, L, L))
        t1 = time.perf_counter()
        n = B * L**3
        one = np.empty(n, dtype=np.complex128); fin = np.empty(n, dtype=np.complex128)
        t2 = time.perf_counter()
        _RUNS[L](B, ms[L], _pp(x0r), _pp(x0i), _pp(cr), _pp(ci), one.ctypes.data_as(_PD), fin.ctypes.data_as(_PD))
        t3 = time.perf_counter()
        outs += [one, fin]
        t_rng += t1-t0; t_alloc += t2-t1; t_c += t3-t2
    t0 = time.perf_counter()
    res = np.concatenate(outs)
    t_cat = time.perf_counter() - t0
    print(f"rng {t_rng*1e3:7.1f}  alloc {t_alloc*1e3:6.1f}  c-run {t_c*1e3:7.1f}  concat {t_cat*1e3:6.1f}  TOTAL {(t_rng+t_alloc+t_c+t_cat)*1e3:7.1f}")
    return res

for _ in range(3):
    timed_transform()
