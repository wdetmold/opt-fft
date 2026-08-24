cd /workdir/dev && python3 - <<'EOF'
import tune
# THRESH=1: remainder>=1 -> SoA padded (always SoA); THRESH=9: always slab for remainder
A = tune.build({'soathresh':{str(L):'1' for L in (6,8,13,17,23)}}, 'thA')
B = tune.build({'soathresh':{str(L):'9' for L in (6,8,13,17,23)}}, 'thB')
params={L:(1,400) for L in (6,8,13,17,23)}
params[6]=(1,2000); params[8]=(1,1500); params[13]=(1,600)
import numpy as np
for i in range(2):
    ra = tune.bench(A, Ls=(6,8,13,17,23), params={**{L:(1, {6:2000,8:1500,13:600,17:400,23:300}[L]) for L in (6,8,13,17,23)}, }, reps=4)
    rb = tune.bench(B, Ls=(6,8,13,17,23), params={**{L:(1, {6:2000,8:1500,13:600,17:400,23:300}[L]) for L in (6,8,13,17,23)}, }, reps=4)
    print("B=1 soa-padded(eff ns/elem):", {L: round(v,2) for L,v in ra.items()})
    print("B=1 slab-one  (ns/elem):   ", {L: round(v,2) for L,v in rb.items()})
EOF