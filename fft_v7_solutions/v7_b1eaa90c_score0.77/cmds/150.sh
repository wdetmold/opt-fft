cd /workdir/dev && python3 - <<'EOF'
src = open('gen.py').read()
src = src.replace("'sqpar': {'36': '1'}}", "'sqpar': {'36': '1', '45': '1'}}")
open('gen.py','w').write(src)
EOF
python3 gen.py && cd /workdir && rm -f implementation.so && gcc -O3 -march=native -shared -fPIC implementation.c -o implementation.so -lm && python3 - <<'EOF'
import sys
sys.path.insert(0,'/workdir/dev'); sys.path.insert(0,'/workdir')
import check
ok=True
for m in (1,2,3,4,5):
    ok &= check.check(70+m, (2,1,1,1,1,2,2,1), (m,m+1,m,m+2,m,m,m,m), verbose=False)
print("45/36 parity edges:", "OK" if ok else "FAIL")
EOF