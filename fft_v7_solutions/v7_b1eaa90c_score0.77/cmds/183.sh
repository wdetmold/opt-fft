cd /workdir/dev && python3 -c "
import tune
so='/workdir/implementation.so'
allr={}
for i in range(4):
    r=tune.bench(so, Ls=(64,), reps=5, params={64:(2,20)})
    for L,v in r.items(): allr.setdefault(L,[]).append(v)
print('64 fused-s2z:', {L: [round(x,2) for x in v] for L,v in allr.items()})
" && python3 -c "
import sys; sys.path.insert(0,'/workdir/dev'); sys.path.insert(0,'/workdir')
import check
ok = True
for m in (1,2,3,4):
    ok &= check.check(400+m, (1,1,1,1,1,1,1,2), (m,)*8, verbose=False)
print('64 s2z edges:', 'OK' if ok else 'FAIL')
"