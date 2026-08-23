import sys, numpy as np
sys.path.insert(0, '/workdir')
import solution, base
rng = np.random.default_rng(0)
_SIZES = (6, 8, 13, 17, 23, 36, 45, 64)
fails = 0
for trial in range(12):
    seed = int(rng.integers(0, 10000))
    Bs = [int(rng.integers(1, hi)) for hi in (40, 30, 20, 14, 10, 9, 9, 4)]
    ms = [int(rng.integers(1, 12)) for _ in range(8)]
    p = (seed, *Bs, *ms)
    got = solution.transform(*p)
    ref = base.transform(*p)
    off = 0; worst1 = worstm = 0
    for i, L in enumerate(_SIZES):
        n = Bs[i]*L**3
        for which in range(2):
            g = got[off:off+n]; r = ref[off:off+n]
            e = np.linalg.norm(g-r)/max(np.linalg.norm(r),1e-300)
            if which==0: worst1 = max(worst1, e)
            else: worstm = max(worstm, e)
            lim = 1e-14 if which==0 else 1e-3
            if e >= lim:
                fails += 1
                print(f"trial {trial} L={L} blk{which} B={Bs[i]} m={ms[i]} err={e:.2e} FAIL")
            off += n
    print(f"trial {trial}: Bs={Bs} ms={ms} worst1={worst1:.2e} worstm={worstm:.2e}")
print("fails:", fails)
