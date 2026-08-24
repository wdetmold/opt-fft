cd /workdir && python3 - <<'EOF'
# ================= FINAL COMPREHENSIVE BATTERY =================
import sys, time, numpy as np
sys.path.insert(0,'/workdir')
import importlib.util
spec = importlib.util.spec_from_file_location("base", "/workdir/base.py")
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
import solution
SIZES=(6,8,13,17,23,36,45,64)
gates={6:1e-4,8:3e-6,13:1e-9,17:1e-10,23:1e-10,36:1e-10,45:1e-10,64:1e-10}

def verify(seed, Bs, ms, time_it=False):
    args=(seed,)+tuple(Bs)+tuple(ms)
    ts=1e9
    for _ in range(2 if time_it else 1):
        t0=time.perf_counter(); g=solution.transform(*args); t1=time.perf_counter()
        ts=min(ts,t1-t0)
    t2=time.perf_counter(); r=base.transform(*args); t3=time.perf_counter()
    assert g.dtype==np.complex128 and g.shape==r.shape
    off=0; ok=True; worst=0
    for L,B in zip(SIZES,Bs):
        n=B*L**3
        if n:
            e1=np.linalg.norm(g[off:off+n]-r[off:off+n])/max(np.linalg.norm(r[off:off+n]),1e-300)
            em=np.linalg.norm(g[off+n:off+2*n]-r[off+n:off+2*n])/max(np.linalg.norm(r[off+n:off+2*n]),1e-300)
            ok &= e1<1e-14 and em<gates[L]
        off+=2*n
    return ok, ts, t3-t2

tests = [
  (1,  (1,1,1,1,1,1,1,1), (1,1,1,1,1,1,1,1)),
  (2,  (2,2,2,2,2,2,2,2), (2,2,2,2,2,2,2,2)),
  (3,  (3,5,7,2,4,3,2,3), (3,4,5,6,7,8,9,10)),
  (4,  (0,1,0,1,0,1,0,1), (5,5,5,5,5,5,5,5)),
  (5,  (16,8,9,10,11,2,3,2), (11,12,13,14,15,16,17,18)),
  (6,  (100,40,20,12,6,4,2,2), (20,)*8),
  (77, (7,6,5,4,3,2,1,2), (1,2,3,4,5,6,7,8)),
]
allok=True
for seed,Bs,ms in tests:
    ok,ts,tb = verify(seed,Bs,ms)
    allok &= ok
    print(f"seed={seed:3d} {'OK' if ok else 'FAIL'} ({ts:.3f}s vs {tb:.3f}s)")
print("="*40)
print("FINAL BATTERY:", "ALL OK" if allok else "FAILURES PRESENT")
EOF