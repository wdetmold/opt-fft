cd /workdir && python3 - <<'EOF'
import time, numpy as np
import importlib.util
spec = importlib.util.spec_from_file_location("base", "/workdir/base.py")
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)

# per-size cost measurement: time one size at a time with B volumes, m iters
def time_size(L, B, m):
    idx = {6:0, 8:1, 13:2, 17:3, 23:4, 36:5, 45:6, 64:7}[L]
    Bs=[0]*8; ms=[1]*8
    Bs[idx]=B; ms[idx]=m
    t0=time.perf_counter()
    r = base.transform(7, *Bs, *ms)
    t1=time.perf_counter()
    return t1-t0

for L,B,m in [(6,64,50),(8,64,50),(13,16,50),(17,8,50),(23,4,50),(36,2,20),(45,1,20),(64,1,10)]:
    # warm
    t=min(time_size(L,B,m) for _ in range(2))
    pv = t/(B*m)
    pe = pv/ (L**3) * 1e9
    print(f"L={L:2d} B={B:3d} m={m:3d}: {t:7.3f}s  per vol-iter {pv*1e3:8.4f} ms  per elem-iter {pe:7.2f} ns")
EOF