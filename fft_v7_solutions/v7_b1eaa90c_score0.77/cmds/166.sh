cd /workdir && rm -f implementation.so && gcc -O3 -march=native -shared -fPIC implementation.c -o implementation.so -lm && python3 - <<'EOF'
import sys
sys.path.insert(0,'/workdir/dev'); sys.path.insert(0,'/workdir')
import check
ok=True
# exercise slab fallbacks: B in 1..4 (< default threshold 5) for the SoA sizes
for B in (1,2,3,4):
    ok &= check.check(100+B, (B,B,B,B,B,1,1,1), (5,6,7,8,9,4,5,6), verbose=False)
# mixed: remainders above threshold (soa-padded)
ok &= check.check(200, (14,11,10,9,13,2,2,2), (4,)*8, verbose=False)
ok &= check.check(201, (8,16,9,7,6,1,1,1), (3,)*8, verbose=False)
print("FALLBACK BATTERY:", "OK" if ok else "FAIL")
EOF