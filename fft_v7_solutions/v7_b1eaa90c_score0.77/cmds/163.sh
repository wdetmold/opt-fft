cd /workdir/dev && python3 - <<'EOF'
import subprocess, ctypes, os
import numpy as np
os.sched_setaffinity(0, {0})
os.makedirs('/tmp/pgo', exist_ok=True)
subprocess.run(['gcc','-O3','-march=native','-fprofile-generate','-shared','-fPIC','/workdir/implementation.c','-o','/tmp/pgo/impl_gen.so','-lm'], check=True, cwd='/tmp/pgo')
lib = ctypes.CDLL('/tmp/pgo/impl_gen.so')
for L in (6,8,13,17,23,36,45,64):
    B=2; n=B*L**3
    rng=np.random.default_rng(1)
    x0=rng.standard_normal(2*n); c=0.1*rng.standard_normal(2*n)
    o1=np.empty(2*n); om=np.empty(2*n)
    f=getattr(lib,f'run{L}')
    f.argtypes=[ctypes.c_void_p]*2+[ctypes.c_long]*2+[ctypes.c_void_p]*2
    f(x0.ctypes.data,c.ctypes.data,B,6,o1.ctypes.data,om.ctypes.data)
import _ctypes
_ctypes.dlclose(lib._handle)   # flush gcda on dlclose? profile written at dlclose/exit
del lib
EOF
ls /tmp/pgo/ && cd /tmp/pgo && gcc -O3 -march=native -fprofile-use -fprofile-correction -shared -fPIC /workdir/implementation.c -o /tmp/pgo/impl_use.so -lm 2>&1 | head -3 && cd /workdir/dev && python3 -c "
import tune
for i in range(3):
    ra=tune.bench('/workdir/implementation.so', reps=3); rb=tune.bench('/tmp/pgo/impl_use.so', reps=3)
    print('std', {L: round(v,2) for L,v in ra.items()})
    print('pgo', {L: round(v,2) for L,v in rb.items()})
"