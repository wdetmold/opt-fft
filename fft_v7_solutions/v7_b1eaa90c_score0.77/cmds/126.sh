cd /workdir && python3 - <<'EOF'
import sys
sys.path.insert(0,'/workdir/dev'); sys.path.insert(0,'/workdir')
import check
ok=True
for m in (1,2,3,4,5):
    ok &= check.check(50+m, (1,1,1,1,1,1,3,1), (m,)*8, verbose=False)
print("45-parity edge:", "OK" if ok else "FAIL")
EOF