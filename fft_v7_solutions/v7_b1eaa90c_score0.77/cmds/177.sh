cd /workdir/dev && python3 -c "
import tune
so='/workdir/implementation.so'
allr={}
for i in range(5):
    r=tune.bench(so, reps=4)
    for L,v in r.items(): allr.setdefault(L,[]).append(v)
final = {L: round(min(v),2) for L,v in allr.items()}
print('FINAL ns/elem-iter:', final)
base = {6:42, 8:34, 13:40, 17:40, 23:43, 36:27, 45:26, 64:26}
print('per-size speedup: ', {L: round(base[L]/final[L],1) for L in final})
"