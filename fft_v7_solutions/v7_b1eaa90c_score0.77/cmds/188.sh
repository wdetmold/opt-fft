cd /tmp && rm -rf gradesim && mkdir gradesim && cp /workdir/solution.py /workdir/implementation.c /workdir/base.py gradesim/ && cd gradesim && python3 - <<'EOF'
# simulate grading: fresh dir WITHOUT prebuilt .so -> import must compile
import sys, time, os
sys.path.insert(0, '/tmp/gradesim')
t0=time.time()
import solution
print(f"import+compile: {time.time()-t0:.1f}s")
import importlib.util, numpy as np
spec = importlib.util.spec_from_file_location("base", "/tmp/gradesim/base.py")
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
args=(8675309, 24,12,6,4,3,2,1,1, 150,150,120,100,100,90,80,80)
best=1e9
for _ in range(3):
    t0=time.perf_counter(); g=solution.transform(*args); t1=time.perf_counter()
    best=min(best,t1-t0)
t2=time.perf_counter(); r=base.transform(*args); t3=time.perf_counter()
SIZES=(6,8,13,17,23,36,45,64); Bs=args[1:9]
gates={6:1e-4,8:3e-6,13:1e-9,17:1e-10,23:1e-10,36:1e-10,45:1e-10,64:1e-10}
off=0; ok=True
for L,B in zip(SIZES,Bs):
    n=B*L**3
    e1=np.linalg.norm(g[off:off+n]-r[off:off+n])/np.linalg.norm(r[off:off+n])
    em=np.linalg.norm(g[off+n:off+2*n]-r[off+n:off+2*n])/np.linalg.norm(r[off+n:off+2*n])
    ok &= e1<1e-14 and em<gates[L]
    off+=2*n
print(f"GRADESIM: {'OK' if ok else 'FAIL'} | sol best {best:.3f}s base {t3-t2:.3f}s ratio {best/(t3-t2):.4f}")
EOF
cd /workdir && gcc -O3 -march=native -shared -fPIC implementation.c -o /tmp/rebuild.so -lm && cmp /tmp/rebuild.so /tmp/gradesim/implementation.so && echo "BUILD DETERMINISTIC" ; rm -rf /tmp/gradesim /tmp/rebuild.so