cd /workdir && rm -f implementation.so && time gcc -O3 -march=native -shared -fPIC implementation.c -o implementation.so -lm && python3 - <<'EOF'
# Final realistic heavy benchmark
import sys, time, numpy as np
sys.path.insert(0,'/workdir')
import importlib.util
spec = importlib.util.spec_from_file_location("base", "/workdir/base.py")
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
import solution
scenarios = [
  ("equal-base-time-ish", (40,20,8,5,3,2,1,1), (2000,2000,1500,1200,1200,1200,1200,1200)),
]
for name, Bs, ms in scenarios:
    args=(2025,)+Bs+ms
    best=1e9
    for _ in range(2):
        t0=time.perf_counter(); g=solution.transform(*args); t1=time.perf_counter()
        best=min(best,t1-t0)
    t2=time.perf_counter(); r=base.transform(*args); t3=time.perf_counter()
    err=np.linalg.norm(g-r)/np.linalg.norm(r)
    print(f"{name}: sol {best:.2f}s base {t3-t2:.2f}s ratio {best/(t3-t2):.4f} (err {err:.1e})")
EOF