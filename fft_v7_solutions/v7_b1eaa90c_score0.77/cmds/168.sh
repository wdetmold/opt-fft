cd /workdir/dev && python3 - <<'EOF'
src = open('gen.py').read()
src = src.replace("'sqpar': {'36': '1', '45': '1'}}",
                  "'sqpar': {'36': '1', '45': '1'}, 'soathresh': {'6': '3', '8': '5', '13': '5', '17': '5', '23': '8'}}")
open('gen.py','w').write(src)
EOF
python3 gen.py && cd /workdir && rm -f implementation.so && gcc -O3 -march=native -shared -fPIC implementation.c -o implementation.so -lm && python3 - <<'EOF'
import sys, time, numpy as np
sys.path.insert(0,'/workdir/dev'); sys.path.insert(0,'/workdir')
import check
ok=True
for B in (1,2,3,4,5,6,7):
    ok &= check.check(300+B, (B,B,B,B,B,1,1,1), (4,5,6,7,4,5,6,7), verbose=False)
print("THRESH BATTERY:", "OK" if ok else "FAIL")
EOF