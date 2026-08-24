cd /workdir && timeout 600 python3 - <<'EOF'
import sys, time, numpy as np
sys.path.insert(0,'/workdir')
import importlib.util
spec = importlib.util.spec_from_file_location("base", "/workdir/base.py")
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
import solution
# large-B test for big sizes
for Bs, ms, tag in [
    ((0,0,0,0,0,0,0,8), (1,1,1,1,1,1,1,60), "B64=8 m=60"),
    ((0,0,0,0,0,16,8,0), (1,1,1,1,1,60,60,1), "B36=16,B45=8 m=60"),
    ((0,2048,0,0,0,0,0,0), (1,400,1,1,1,1,1,1), "B8=2048 m=400"),
]:
    args=(5,)+Bs+ms
    best=1e9
    for _ in range(2):
        t0=time.perf_counter(); g=solution.transform(*args); t1=time.perf_counter()
        best=min(best,t1-t0)
    t2=time.perf_counter(); r=base.transform(*args); t3=time.perf_counter()
    err=np.linalg.norm(g-r)/np.linalg.norm(r)
    print(f"{tag:22s} sol {best:6.2f}s base {t3-t2:6.2f}s ratio {best/(t3-t2):.4f} err {err:.1e}")
EOF