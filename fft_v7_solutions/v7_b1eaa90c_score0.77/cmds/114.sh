cd /workdir && python3 - <<'EOF'
import sys, numpy as np
sys.path.insert(0,'/workdir')
import importlib.util
spec = importlib.util.spec_from_file_location("base", "/workdir/base.py")
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
import solution
SIZES=(6,8,13,17,23,36,45,64)
IDX={L:i for i,L in enumerate(SIZES)}
gates={6:1e-4,8:3e-6,13:1e-9,17:1e-10,23:1e-10,36:1e-10,45:1e-10,64:1e-10}
import math
for L in SIZES:
    B=1
    rels=[]
    for m in (200, 800, 3200):
        Bs=[0]*8; ms=[1]*8; Bs[IDX[L]]=B; ms[IDX[L]]=m
        g=solution.transform(77,*Bs,*ms); r=base.transform(77,*Bs,*ms)
        n=L**3
        e=np.linalg.norm(g[n:2*n]-r[n:2*n])/np.linalg.norm(r[n:2*n])
        rels.append(e)
    # exp fit between m=800 and 3200
    q = (math.log(rels[2])-math.log(rels[1]))/(3200-800)
    # m* such that divergence = gate/10^2.75
    target = gates[L]/10**2.75
    mstar = 3200 + (math.log(target)-math.log(rels[2]))/q if q>0 else float('nan')
    print(f"L={L:2d} div@200={rels[0]:.2e} @800={rels[1]:.2e} @3200={rels[2]:.2e} growth/step={math.exp(q):.5f} -> est m*≈{mstar:,.0f}")
EOF