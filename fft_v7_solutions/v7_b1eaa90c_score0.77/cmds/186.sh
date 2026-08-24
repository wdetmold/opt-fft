cd /workdir && python3 - <<'EOF'
import sys, numpy as np
sys.path.insert(0,'/workdir')
import importlib.util
spec = importlib.util.spec_from_file_location("base", "/workdir/base.py")
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
import solution
SIZES=(6,8,13,17,23,36,45,64)
gates={6:1e-4,8:3e-6,13:1e-9,17:1e-10,23:1e-10,36:1e-10,45:1e-10,64:1e-10}
def verify(seed, Bs, ms):
    args=(seed,)+tuple(Bs)+tuple(ms)
    g=solution.transform(*args); r=base.transform(*args)
    off=0; ok=True
    for L,B in zip(SIZES,Bs):
        n=B*L**3
        if n:
            e1=np.linalg.norm(g[off:off+n]-r[off:off+n])/np.linalg.norm(r[off:off+n])
            em=np.linalg.norm(g[off+n:off+2*n]-r[off+n:off+2*n])/np.linalg.norm(r[off+n:off+2*n])
            ok &= e1<1e-14 and em<gates[L]
        off+=2*n
    return ok
allok=True
for seed,Bs,ms in [
  (1,(1,1,1,1,1,1,1,1),(1,)*8),(2,(2,2,2,2,2,2,2,2),(2,)*8),
  (3,(3,5,7,2,4,3,2,3),(3,4,5,6,7,8,9,10)),(4,(0,1,0,1,0,1,0,1),(5,)*8),
  (5,(16,8,9,10,11,2,3,2),(11,12,13,14,15,16,17,18)),
  (6,(100,40,20,12,6,4,2,2),(20,)*8),(7,(7,6,5,4,3,2,1,2),(1,2,3,4,5,6,7,8)),
  (8,(4,4,4,4,4,4,4,4),(50,)*8),(9,(1,2,3,4,5,6,7,8),(33,32,31,30,29,28,27,26)),
  (10,(2,2,1,1,1,1,1,1),(300,300,200,150,150,150,120,120)),
]:
    ok = verify(seed,Bs,ms); allok &= ok
    if not ok: print("FAIL at seed", seed)
print("DEFINITIVE BATTERY:", "ALL OK" if allok else "FAIL")
EOF