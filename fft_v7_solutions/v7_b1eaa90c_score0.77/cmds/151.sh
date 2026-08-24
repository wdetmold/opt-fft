cd /workdir/dev && python3 -c "
import tune
so='/workdir/implementation.so'
allr={}
for i in range(4):
    r=tune.bench(so, reps=4)
    for L,v in r.items(): allr.setdefault(L,[]).append(v)
import numpy as np
print('FINAL:', {L: round(min(v),2) for L,v in allr.items()})
base = {6:42, 8:34, 13:40, 17:40, 23:43, 36:27, 45:26, 64:26}
r = {L: round(min(v)/base[L],3) for L,v in allr.items()}
print('ratios vs cache-friendly base:', r)
print('mean ratio:', round(sum(r.values())/8, 4))
"