cd /workdir/dev && sed -i "s/45:'apfa', 64:'sq'}/45:'apfa2', 64:'sq'}/" gen.py && python3 gen.py && cd /workdir && rm -f implementation.so && gcc -O3 -march=native -shared -fPIC implementation.c -o implementation.so -lm && grep -c a245 implementation.c && python3 dev/check.py | tail -1 && cd dev && python3 -c "
import tune
so='/workdir/implementation.so'
allr={}
for i in range(4):
    r=tune.bench(so, reps=4)
    for L,v in r.items(): allr.setdefault(L,[]).append(v)
print('FINAL:', {L: round(min(v),2) for L,v in allr.items()})
"