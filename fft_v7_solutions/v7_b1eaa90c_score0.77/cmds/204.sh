cd /workdir && rm -rf __pycache__ dev/__pycache__ && md5sum solution.py implementation.c implementation.so && du -sh /workdir && echo "=== final sanity: fresh process, pristine import, timed flow ===" && python3 - <<'EOF'
import sys, time, numpy as np
sys.path.insert(0,'/workdir')
import importlib.util
spec = importlib.util.spec_from_file_location("base", "/workdir/base.py")
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
import solution
SIZES=(6,8,13,17,23,36,45,64)
gates={6:1e-4,8:3e-6,13:1e-9,17:1e-10,23:1e-10,36:1e-10,45:1e-10,64:1e-10}
# grading-like: several reps, best-of
args=(20250823, 32,16,8,6,4,2,2,1, 600,600,400,300,300,250,250,250)
ts=[]
for _ in range(3):
    t0=time.perf_counter(); g=solution.transform(*args); t1=time.perf_counter(); ts.append(t1-t0)
tb=[]
for _ in range(2):
    t0=time.perf_counter(); r=base.transform(*args); t1=time.perf_counter(); tb.append(t1-t0)
off=0; ok=True
for L,B in zip(SIZES,args[1:9]):
    n=B*L**3
    e1=np.linalg.norm(g[off:off+n]-r[off:off+n])/np.linalg.norm(r[off:off+n])
    em=np.linalg.norm(g[off+n:off+2*n]-r[off+n:off+2*n])/np.linalg.norm(r[off+n:off+2*n])
    ok &= e1<1e-14 and em<gates[L]
    off+=2*n
print(f"times: sol {min(ts):.3f}s (runs {['%.3f'%t for t in ts]}) base {min(tb):.3f}s")
print(f"ratio {min(ts)/min(tb):.4f}  correctness: {'ALL OK' if ok else 'FAIL'}")
EOF