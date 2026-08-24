cd /workdir/dev && python3 - <<'EOF'
src = open('gen.py').read()
src = src.replace("if (t > 0) {{{{\n          f64_s2z(", "if ({FZ64} && t > 0) {{{{\n          f64_s2z(")
src = src.replace("def gen_sq(L, G):\n    PAR", "def gen_sq(L, G):\n    FZ64 = int(CONFIG.get('fz64', '0'))\n    PAR")
open('gen.py','w').write(src)
EOF
python3 gen.py && gcc -O3 -march=native -shared -fPIC ../implementation.c -o /workdir/implementation.so -lm && python3 -c "
import tune
so='/workdir/implementation.so'
allr={}
for i in range(4):
    r=tune.bench(so, Ls=(64,), reps=5, params={64:(2,20)})
    for L,v in r.items(): allr.setdefault(L,[]).append(v)
print('64 unfused again:', {L: [round(x,2) for x in v] for L,v in allr.items()})
" && python3 /workdir/dev/check.py | tail -1