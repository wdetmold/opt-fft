cd /workdir && ls -la && echo "=== FINAL ACCEPTANCE RUN ===" && python3 - <<'EOF'
import sys, time, numpy as np
sys.path.insert(0,'/workdir')
import importlib.util
spec = importlib.util.spec_from_file_location("base", "/workdir/base.py")
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
import solution
SIZES=(6,8,13,17,23,36,45,64)
gates={6:1e-4,8:3e-6,13:1e-9,17:1e-10,23:1e-10,36:1e-10,45:1e-10,64:1e-10}

def full_verify(args):
    g=solution.transform(*args); r=base.transform(*args)
    assert g.dtype == np.complex128
    Bs=args[1:9]
    exp = 2*sum(b*l**3 for b,l in zip(Bs,SIZES))
    assert g.shape == (exp,), (g.shape, exp)
    off=0; ok=True
    for L,B in zip(SIZES,Bs):
        n=B*L**3
        if n:
            e1=np.linalg.norm(g[off:off+n]-r[off:off+n])/np.linalg.norm(r[off:off+n])
            em=np.linalg.norm(g[off+n:off+2*n]-r[off+n:off+2*n])/np.linalg.norm(r[off+n:off+2*n])
            ok &= e1<1e-14 and em<gates[L]
        off+=2*n
    return ok

cases = [
    (0,(1,1,1,1,1,1,1,1),(1,)*8),
    (1,(8,8,8,8,8,8,8,8),(10,)*8),
    (2,(3,1,4,1,5,2,1,2),(9,2,6,5,3,5,8,9)),
    (3,(0,0,1,0,0,0,0,1),(7,7,7,7,7,7,7,7)),
    (42,(17,13,11,7,5,3,2,1),(25,24,23,22,21,20,19,18)),
    (99,(2,2,2,2,2,2,2,2),(100,100,80,70,70,60,60,60)),
]
allok = all(full_verify((s,)+b+m) for s,b,m in cases)
print("ACCEPTANCE:", "ALL PASS" if allok else "FAIL")
# determinism
a=solution.transform(5, 3,3,3,3,3,1,1,1, 8,8,8,8,8,8,8,8)
b=solution.transform(5, 3,3,3,3,3,1,1,1, 8,8,8,8,8,8,8,8)
print("bitwise deterministic:", np.array_equal(a,b))
EOF