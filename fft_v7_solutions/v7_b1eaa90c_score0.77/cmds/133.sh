cd /workdir && python3 - <<'EOF'
# Final multi-scenario comparison vs base
import sys, time, numpy as np
sys.path.insert(0,'/workdir')
import importlib.util
spec = importlib.util.spec_from_file_location("base", "/workdir/base.py")
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
import solution
SIZES=(6,8,13,17,23,36,45,64)
gates={6:1e-4,8:3e-6,13:1e-9,17:1e-10,23:1e-10,36:1e-10,45:1e-10,64:1e-10}
scenarios = [
  ("balanced m~1500", (150,64,16,8,4,2,1,1), (1500,1500,1500,1500,1500,1500,1500,1500)),
  ("equal-elems m=400", (2315,977,171,60,24,11,5,2), (400,)*8),
  ("big-B small-m", (512,256,64,32,16,8,4,4), (60,60,50,50,40,40,30,30)),
]
for name, Bs, ms in scenarios:
    args=(7,)+tuple(Bs)+tuple(ms)
    best=1e9
    for _ in range(2):
        t0=time.perf_counter(); g=solution.transform(*args); t1=time.perf_counter()
        best=min(best,t1-t0)
    t2=time.perf_counter(); r=base.transform(*args); t3=time.perf_counter()
    # correctness
    off=0; ok=True
    for L,B in zip(SIZES,Bs):
        n=B*L**3
        e1=np.linalg.norm(g[off:off+n]-r[off:off+n])/np.linalg.norm(r[off:off+n])
        em=np.linalg.norm(g[off+n:off+2*n]-r[off+n:off+2*n])/np.linalg.norm(r[off+n:off+2*n])
        ok &= e1<1e-14 and em<gates[L]
        off+=2*n
    print(f"{name:22s} sol {best:6.2f}s base {t3-t2:7.2f}s ratio {best/(t3-t2):.4f} {'OK' if ok else 'FAIL'}")
EOF