cd /workdir/dev && python3 gen.py && cd /workdir && rm -f implementation.so && gcc -O3 -march=native -shared -fPIC implementation.c -o implementation.so -lm && python3 - <<'EOF'
import sys
sys.path.insert(0,'/workdir/dev'); sys.path.insert(0,'/workdir')
import check
# edge shapes: B=1, B=0 mixed, odd B, m=1..3, bigger B small L
ok = True
ok &= check.check(11, (1,1,1,1,1,1,1,1), (1,1,1,1,1,1,1,1), verbose=False)
ok &= check.check(12, (0,3,0,2,1,0,2,1), (2,2,2,2,2,2,2,2), verbose=False)
ok &= check.check(13, (9,7,5,3,2,3,2,3), (3,1,2,5,4,3,2,1), verbose=False)
ok &= check.check(14, (17,9,11,6,5,4,3,2), (7,6,5,4,3,2,4,5), verbose=False)
ok &= check.check(999999, (33,16,8,4,2,2,1,1), (10,10,10,10,10,10,10,10), verbose=False)
print("EDGE OK" if ok else "EDGE FAIL")
EOF