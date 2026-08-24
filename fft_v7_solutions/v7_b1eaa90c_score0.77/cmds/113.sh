cd /workdir && python3 - <<'EOF'
# per-size long-chain gate check at m near plausible graded values, vs base
import sys, numpy as np, time
sys.path.insert(0,'/workdir')
import importlib.util
spec = importlib.util.spec_from_file_location("base", "/workdir/base.py")
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
import solution
SIZES=(6,8,13,17,23,36,45,64)
gates={6:1e-4,8:3e-6,13:1e-9,17:1e-10,23:1e-10,36:1e-10,45:1e-10,64:1e-10}
Bs=(4,4,2,2,2,1,1,1)
ms=(400,400,250,180,180,160,150,140)
args=(123,)+Bs+ms
t0=time.perf_counter(); got=solution.transform(*args); t1=time.perf_counter()
ref=base.transform(*args); t2=time.perf_counter()
off=0; allok=True
for L,B in zip(SIZES,Bs):
    n=B*L**3
    for tag, sl in (("one",slice(off,off+n)), ("final",slice(off+n,off+2*n))):
        e=np.linalg.norm(got[sl]-ref[sl])/np.linalg.norm(ref[sl])
        lim = 1e-14 if tag=="one" else gates[L]
        ok = e < lim
        allok &= ok
        if tag=="final" or not ok:
            print(f"L={L:2d} {tag:5s} rel={e:.3e} lim={lim:.0e} {'OK' if ok else 'FAIL'}")
    off += 2*n
print("LONG-CHAIN", "ALL OK" if allok else "FAIL", f"sol {t1-t0:.2f}s base {t2-t1:.2f}s")
EOF