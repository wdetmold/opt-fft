cd /workdir/dev && python3 - <<'EOF'
import subprocess, ctypes, time, os
import numpy as np
os.sched_setaffinity(0, {0})
# build with -fprofile-generate
subprocess.run(['gcc','-O3','-march=native','-fprofile-generate','-shared','-fPIC','../implementation.c','-o','/tmp/impl_pgo_gen.so','-lm'], check=True, cwd='/tmp')
lib = ctypes.CDLL('/tmp/impl_pgo_gen.so')
SIZES=(6,8,13,17,23,36,45,64)
for L in SIZES:
    B=2; n=B*L**3
    rng=np.random.default_rng(1)
    x0=rng.standard_normal(2*n); c=0.1*rng.standard_normal(2*n)
    o1=np.empty(2*n); om=np.empty(2*n)
    f=getattr(lib,f'run{L}')
    f.argtypes=[ctypes.c_void_p]*2+[ctypes.c_long]*2+[ctypes.c_void_p]*2
    f(x0.ctypes.data,c.ctypes.data,B,6,o1.ctypes.data,om.ctypes.data)
del lib
# rebuild with -fprofile-use
subprocess.run(['gcc','-O3','-march=native','-fprofile-use','-fprofile-correction','-shared','-fPIC','../implementation.c','-o','/tmp/impl_pgo_use.so','-lm'], check=True, cwd='/tmp')
print("pgo build ok")
EOF
python3 -c "
import tune
A='/workdir/implementation.so'
B='/tmp/impl_pgo_use.so'
for i in range(3):
    ra=tune.bench(A, reps=3); rb=tune.bench(B, reps=3)
    print('std', {L: round(v,2) for L,v in ra.items()})
    print('pgo', {L: round(v,2) for L,v in rb.items()})
"