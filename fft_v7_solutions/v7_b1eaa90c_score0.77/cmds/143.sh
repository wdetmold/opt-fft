cd /workdir/dev && python3 - <<'EOF'
src = open('gen.py').read()
src = src.replace("'sqpw': {}}", "'sqpw': {}, 'sqpar': {'36': '1'}}")
open('gen.py','w').write(src)
EOF
python3 gen.py && cd /workdir && rm -f implementation.so && gcc -O3 -march=native -shared -fPIC implementation.c -o implementation.so -lm && python3 - <<'EOF'
import sys
sys.path.insert(0,'/workdir/dev'); sys.path.insert(0,'/workdir')
import check
ok=True
for m in (1,2,3,4,5):
    ok &= check.check(60+m, (1,1,1,1,1,3,1,1), (m,)*8, verbose=False)
print("36-parity edges:", "OK" if ok else "FAIL")
EOF