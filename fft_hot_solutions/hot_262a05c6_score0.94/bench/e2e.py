import sys, time, os
sys.path.insert(0, sys.argv[1])
from wl import WL
args = [12345] + [WL[L][0] for L in (6,8,13,17,23,36,45,64)] + [WL[L][1] for L in (6,8,13,17,23,36,45,64)]
if sys.argv[2] == 'mkl':
    sys.path.insert(0,'/tmp/bench')
    import numpy as np
    from sotah import sota_run
    from harness import gen_inputs
    def transform(*a):
        outs=[]
        for i,L in enumerate((6,8,13,17,23,36,45,64)):
            B,m = a[1+i], a[9+i]
            x0,c = gen_inputs(a[0], L, B)
            one, fin = sota_run(L,B,m,x0,c)
            outs += [one, fin]
        return np.concatenate(outs)
else:
    import solution
    transform = solution.transform
best=9e9
for r in range(int(sys.argv[3]) if len(sys.argv)>3 else 3):
    t=time.perf_counter(); out = transform(*args); dt=time.perf_counter()-t
    best=min(best,dt)
    print(f"  run {r}: {dt:.3f}s")
print(f"BEST {sys.argv[2]}: {best:.3f}s  (checksum {complex(out.sum())})")
