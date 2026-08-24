cd /workdir && cat > /tmp/asan_test.py <<'EOF'
import ctypes, numpy as np, random
lib = ctypes.CDLL('/tmp/impl_asan.so')
SIZES=(6,8,13,17,23,36,45,64)
FUNS={}
for L in SIZES:
    f=getattr(lib,f'run{L}')
    f.restype=None
    f.argtypes=[ctypes.c_void_p]*2+[ctypes.c_long]*2+[ctypes.c_void_p]*2
    FUNS[L]=f
rnd = random.Random(777)
for trial in range(12):
    for L in SIZES:
        B = rnd.choice([1,2,3,4,5,8,9])
        m = rnd.choice([1,2,3,5,10])
        n=B*L**3
        rng=np.random.default_rng(trial*100+L)
        x0=(rng.standard_normal(n)+1j*rng.standard_normal(n)).astype(np.complex128)
        c=(0.1*(rng.standard_normal(n)+1j*rng.standard_normal(n))).astype(np.complex128)
        o1=np.empty(n,dtype=np.complex128); om=np.empty(n,dtype=np.complex128)
        FUNS[L](x0.ctypes.data,c.ctypes.data,B,m,o1.ctypes.data,om.ctypes.data)
    print("asan trial", trial, "ok", flush=True)
print("ASAN CLEAN")
EOF
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libasan.so.8 ASAN_OPTIONS=detect_leaks=0 python3 /tmp/asan_test.py 2>&1 | tail -20