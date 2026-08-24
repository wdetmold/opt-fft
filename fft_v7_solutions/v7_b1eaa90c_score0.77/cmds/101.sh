cd /workdir && rm -f implementation.so && gcc -O3 -march=native -shared -fPIC implementation.c -o implementation.so -lm && python3 dev/check.py && python3 - <<'EOF'
import sys, time
sys.path.insert(0,'/workdir')
import importlib.util
spec = importlib.util.spec_from_file_location("base", "/workdir/base.py")
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
import solution
Bs=(2315,977,171,60,24,11,5,2)
ms=(290,280,180,130,120,110,100,90)
args=(7,)+Bs+ms
best=1e9
for _ in range(2):
    t0=time.perf_counter(); solution.transform(*args); t1=time.perf_counter()
    best=min(best,t1-t0)
t4=time.perf_counter(); base.transform(*args); t5=time.perf_counter()
print(f"solution {best:.2f}s base {t5-t4:.2f}s ratio {best/(t5-t4):.4f}")
EOF