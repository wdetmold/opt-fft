cd /workdir/dev && python3 -c "
import tune
so='/workdir/implementation.so'
allr={}
for i in range(5):
    r=tune.bench(so, reps=4)
    for L,v in r.items(): allr.setdefault(L,[]).append(v)
final = {L: round(min(v),2) for L,v in allr.items()}
base = {6:42, 8:34, 13:40, 17:40, 23:43, 36:27, 45:26, 64:26}
print('===== FINAL STATE =====')
print('ns/elem-iter:', final)
print('speedup vs base (cache-friendly regime):', {L: round(base[L]/final[L],1) for L in final})
import numpy as np
print('mean per-size time-ratio:', round(np.mean([final[L]/base[L] for L in final]),4))
" && cd /workdir && cat solution.py