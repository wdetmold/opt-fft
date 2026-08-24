cd /workdir && rm -f implementation.so && gcc -O3 -march=native -shared -fPIC implementation.c -o implementation.so -lm && python3 dev/check.py && cd dev && python3 -c "
import tune
so='/workdir/implementation.so'
for i in range(3):
    r=tune.bench(so, Ls=(36,64), reps=4)
    print({L: round(v,2) for L,v in r.items()})
"