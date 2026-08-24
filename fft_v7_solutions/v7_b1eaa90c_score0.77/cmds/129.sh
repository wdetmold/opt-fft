cd /workdir && grep -c "a245" implementation.c && cd dev && python3 -c "
import tune
so='/workdir/implementation.so'
for i in range(4):
    r=tune.bench(so, Ls=(45,8), reps=5)
    print({L: round(v,2) for L,v in r.items()})
"