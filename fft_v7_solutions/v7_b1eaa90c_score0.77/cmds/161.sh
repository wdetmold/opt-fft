cd /workdir/dev && python3 - <<'EOF'
src = open('gen.py').read()
src = src.replace("'8': 'sqrtnr'", "'8': 'alt'")
open('gen.py','w').write(src)
EOF
python3 gen.py && cd /workdir && rm -f implementation.so && gcc -O3 -march=native -shared -fPIC implementation.c -o implementation.so -lm && python3 dev/check.py | tail -1 && cd dev && python3 -c "
import tune
so='/workdir/implementation.so'
allr={}
for i in range(4):
    r=tune.bench(so, reps=4)
    for L,v in r.items(): allr.setdefault(L,[]).append(v)
print('FINAL:', {L: round(min(v),2) for L,v in allr.items()})
"