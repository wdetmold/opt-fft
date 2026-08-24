cd /workdir && python3 - <<'EOF'
import time, numpy as np
import importlib.util
spec = importlib.util.spec_from_file_location("base", "/workdir/base.py")
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)

def time_size(L, B, m, reps=2):
    idx = {6:0, 8:1, 13:2, 17:3, 23:4, 36:5, 45:6, 64:7}[L]
    Bs=[0]*8; ms=[1]*8
    Bs[idx]=B; ms[idx]=m
    best=1e9
    for _ in range(reps):
        t0=time.perf_counter(); r = base.transform(7, *Bs, *ms); t1=time.perf_counter()
        best=min(best,t1-t0)
    return best

for L,B,m in [(6,512,30),(6,4096,10),(8,512,30),(8,2048,10),(13,256,20),(13,1024,10),(17,128,20),(23,64,20),(36,16,10),(45,8,10),(64,4,10),(64,16,5)]:
    t=time_size(L,B,m)
    pe = t/(B*m)/(L**3)*1e9
    print(f"L={L:2d} B={B:5d} m={m:3d}: {t:7.3f}s  per elem-iter {pe:7.2f} ns  (batch={B*L**3*16/1e6:.1f} MB)")
EOF