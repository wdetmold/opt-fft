cd /workdir && python3 - <<'EOF'
# re-run the deep-chain scenario (small B, huge m) that exposed the problem
import sys, time, numpy as np
sys.path.insert(0,'/workdir')
import importlib.util
spec = importlib.util.spec_from_file_location("base", "/workdir/base.py")
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
import solution
SIZES=(6,8,13,17,23,36,45,64)
gates={6:1e-4,8:3e-6,13:1e-9,17:1e-10,23:1e-10,36:1e-10,45:1e-10,64:1e-10}
Bs=(2,2,1,1,1,1,1,1)
ms=(10000,10000,4000,2500,2500,3000,2000,2500)
args=(314,)+Bs+ms
best=1e9
for _ in range(2):
    t0=time.perf_counter(); g=solution.transform(*args); t1=time.perf_counter()
    best=min(best,t1-t0)
t2=time.perf_counter(); r=base.transform(*args); t3=time.perf_counter()
off=0; ok=True
for L,B in zip(SIZES,Bs):
    n=B*L**3
    e1=np.linalg.norm(g[off:off+n]-r[off:off+n])/np.linalg.norm(r[off:off+n])
    em=np.linalg.norm(g[off+n:off+2*n]-r[off+n:off+2*n])/np.linalg.norm(r[off+n:off+2*n])
    ok &= e1<1e-14 and em<gates[L]
    off+=2*n
print(f"deep-chain small-B: sol {best:.2f}s base {t3-t2:.2f}s ratio {best/(t3-t2):.4f} {'ALL OK' if ok else 'FAIL'}")
EOF