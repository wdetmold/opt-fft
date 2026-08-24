cd /workdir && python3 - <<'EOF'
import sys, numpy as np
sys.path.insert(0,'/workdir')
import importlib.util
spec = importlib.util.spec_from_file_location("base", "/workdir/base.py")
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
import solution
# larger-B slab sizes
args=(31337, 2,2,2,2,2,9,12,6, 3,3,3,3,3,3,3,3)
g=solution.transform(*args); r=base.transform(*args)
SIZES=(6,8,13,17,23,36,45,64)
gates={6:1e-4,8:3e-6,13:1e-9,17:1e-10,23:1e-10,36:1e-10,45:1e-10,64:1e-10}
off=0; ok=True
for L,B in zip(SIZES,args[1:9]):
    n=B*L**3
    e1=np.linalg.norm(g[off:off+n]-r[off:off+n])/np.linalg.norm(r[off:off+n])
    em=np.linalg.norm(g[off+n:off+2*n]-r[off+n:off+2*n])/np.linalg.norm(r[off+n:off+2*n])
    ok &= e1<1e-14 and em<gates[L]
    off+=2*n
print("larger-B slab sizes:", "ALL OK" if ok else "FAIL")
EOF