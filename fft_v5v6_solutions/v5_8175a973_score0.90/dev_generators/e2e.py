import os, time, sys
import numpy as np
sys.path.insert(0, '/workdir')
import solution, base

def check(params, name):
    t0 = time.perf_counter(); got = solution.transform(*params); t1 = time.perf_counter()
    t2 = time.perf_counter(); ref = base.transform(*params); t3 = time.perf_counter()
    assert got.shape == ref.shape and got.dtype == ref.dtype, (got.shape, ref.shape)
    # per-block check
    _SIZES = (6, 8, 13, 17, 23, 36, 45, 64)
    Bs = params[1:9]
    off = 0
    ok = True
    worst1 = worstm = 0
    for i, L in enumerate(_SIZES):
        n = Bs[i]*L**3
        for which in range(2):
            g = got[off:off+n]; r = ref[off:off+n]
            e = np.linalg.norm(g-r)/max(np.linalg.norm(r), 1e-300) if n else 0
            lim = 1e-14 if which == 0 else 1e-3
            if which == 0: worst1 = max(worst1, e)
            else: worstm = max(worstm, e)
            if e >= lim: ok = False; print(f"  FAIL L={L} block{which} err={e:.2e}")
            off += n
    print(f"{name}: mine={t1-t0:.3f}s base={t3-t2:.3f}s speedup={(t3-t2)/(t1-t0):.2f}x worst1={worst1:.2e} worstm={worstm:.2e} {'OK' if ok else 'FAIL'}")

# balanced-ish: comparable numpy time per size
check((7, 64, 48, 24, 16, 8, 4, 2, 1) + (10,)*8, "balanced m=10")
check((123, 8, 8, 8, 8, 8, 8, 8, 8) + (5,)*8, "uniform B=8 m=5")
check((5, 200, 100, 40, 20, 10, 3, 2, 1) + (30, 25, 20, 15, 12, 10, 8, 6), "mixed")
check((99, 1, 1, 1, 1, 1, 1, 1, 1) + (1,)*8, "B=1 m=1")
check((31, 9, 7, 5, 3, 2, 1, 1, 1) + (2, 3, 1, 2, 4, 1, 2, 3), "small odd")
