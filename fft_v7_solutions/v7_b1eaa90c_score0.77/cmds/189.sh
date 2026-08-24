cd /workdir && python3 - <<'EOF'
src = open('solution.py').read()
old = """def _run(L, B, m, x0, c):
    n = int(B) * L * L * L
    one = np.empty(n, dtype=np.complex128)
    final = np.empty(n, dtype=np.complex128)"""
new = """_POOL = {}

def _run(L, B, m, x0, c):
    n = int(B) * L * L * L
    key = (L, n)
    bufs = _POOL.get(key)
    if bufs is None:
        bufs = (np.empty(n, dtype=np.complex128), np.empty(n, dtype=np.complex128))
        _POOL[key] = bufs
    one, final = bufs"""
assert old in src
src = src.replace(old, new)
open('solution.py','w').write(src)
print("pool added")
EOF
python3 dev/check.py | tail -1 && python3 - <<'EOF'
import sys, time
sys.path.insert(0,'/workdir')
import solution
args=(7, 64,32,16,8,4,2,2,2, 1,1,1,1,1,1,1,1)
for i in range(4):
    t0=time.perf_counter(); solution.transform(*args); t1=time.perf_counter()
    print(f"m=1 call {i}: {t1-t0:.3f}s")
EOF