import sys, time, numpy as np
sys.path.insert(0, '/workdir')
import importlib.util
spec = importlib.util.spec_from_file_location("base", "/workdir/base.py")
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
import solution

SIZES=(6,8,13,17,23,36,45,64)
IDX={L:i for i,L in enumerate(SIZES)}

def t_one(mod, L, B, m, reps=3):
    Bl=[0]*8; ml=[1]*8; Bl[IDX[L]]=B; ml[IDX[L]]=m
    best=1e18
    for _ in range(reps):
        t0=time.perf_counter(); mod.transform(7,*Bl,*ml); t1=time.perf_counter()
        best=min(best,t1-t0)
    return best

# measure with two m values to separate per-iter cost from fixed cost
params={6:(64,),8:(64,),13:(32,),17:(16,),23:(8,),36:(4,),45:(2,),64:(2,)}
print(f"{'L':>3} {'B':>4} | per-elem-iter ns:  sol   base   ratio | fixed-cost sol/base (s)")
for L in SIZES:
    B=params[L][0]
    m1,m2=10,60
    s1=t_one(solution,L,B,m1); s2=t_one(solution,L,B,m2)
    b1=t_one(base,L,B,m1); b2=t_one(base,L,B,m2)
    per_s=(s2-s1)/(m2-m1)/(B*L**3)*1e9
    per_b=(b2-b1)/(m2-m1)/(B*L**3)*1e9
    fix_s=s1-per_s*m1*B*L**3/1e9; fix_b=b1-per_b*m1*B*L**3/1e9
    print(f"{L:>3} {B:>4} | {per_s:7.3f} {per_b:7.2f} {per_s/per_b:6.3f} | {fix_s:.3f}/{fix_b:.3f}")
