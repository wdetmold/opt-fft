cd /workdir && python3 dev/gen.py 2>/dev/null; cd /workdir && rm -f implementation.so && gcc -O3 -march=native -shared -fPIC implementation.c -o implementation.so -lm && python3 - <<'EOF'
import sys, time, numpy as np
sys.path.insert(0,'/workdir')
import importlib.util
spec = importlib.util.spec_from_file_location("base", "/workdir/base.py")
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
import solution
Bs=(32,16,8,8,4,2,2,1)
ms=(4000,4000,2000,1200,1200,1000,800,1000)
args=(7,)+Bs+ms
tot = sum(b*l**3*m for b,l,m in zip(Bs,(6,8,13,17,23,36,45,64),ms))
print(f"total elem-iters: {tot/1e9:.2f}G")
t0=time.perf_counter(); g=solution.transform(*args); t1=time.perf_counter()
print(f"solution {t1-t0:.2f}s")
t2=time.perf_counter(); r=base.transform(*args); t3=time.perf_counter()
print(f"base {t3-t2:.2f}s ratio {(t1-t0)/(t3-t2):.4f}")
# block errors
SIZES=(6,8,13,17,23,36,45,64)
gates={6:1e-4,8:3e-6,13:1e-9,17:1e-10,23:1e-10,36:1e-10,45:1e-10,64:1e-10}
off=0; ok=True
for L,B in zip(SIZES,Bs):
    n=B*L**3
    e1=np.linalg.norm(g[off:off+n]-r[off:off+n])/np.linalg.norm(r[off:off+n])
    em=np.linalg.norm(g[off+n:off+2*n]-r[off+n:off+2*n])/np.linalg.norm(r[off+n:off+2*n])
    ok &= e1<1e-14 and em<gates[L]
    print(f"L={L:2d} one={e1:.2e} final={em:.2e} {'OK' if e1<1e-14 and em<gates[L] else 'FAIL'}")
    off+=2*n
print("ALL OK" if ok else "FAIL")
EOF