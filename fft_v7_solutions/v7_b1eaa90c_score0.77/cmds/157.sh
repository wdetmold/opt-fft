cd /workdir/dev && rm -f prof*.c pwtest pwtest.c test_cores.c core64only.h && cd /workdir && rm -rf __pycache__ dev/__pycache__ && python3 dev/gen.py && rm -f implementation.so && gcc -O3 -march=native -shared -fPIC implementation.c -o implementation.so -lm && echo "=== full battery ===" && python3 dev/check.py && python3 - <<'EOF'
import sys
sys.path.insert(0,'/workdir/dev'); sys.path.insert(0,'/workdir')
import check
ok=True
ok &= check.check(11, (1,1,1,1,1,1,1,1), (1,1,1,1,1,1,1,1), verbose=False)
ok &= check.check(12, (0,3,0,2,1,0,2,1), (2,3,2,3,2,3,2,3), verbose=False)
ok &= check.check(13, (9,7,5,3,2,3,2,3), (3,1,2,5,4,3,2,1), verbose=False)
ok &= check.check(999999, (33,16,8,4,2,2,1,1), (10,11,12,13,14,15,16,17), verbose=False)
ok &= check.check(0, (8,8,8,8,8,2,2,2), (6,6,6,6,6,6,6,6), verbose=False)
print("EDGE BATTERY:", "OK" if ok else "FAIL")
EOF