import ctypes, numpy as np, time, os, sys
os.sched_setaffinity(0,{0})
lib = ctypes.CDLL(sys.argv[1] if len(sys.argv)>1 else '/tmp/g/impl_fg.so')
cases = [(36, 8, 60), (45, 6, 40)]
for N,B,m in cases:
    try: getattr(lib, f'init_{N}')()
    except AttributeError: pass
    f = getattr(lib, f'run_{N}')
    f.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_long]*2
    rng = np.random.default_rng(1234+N)
    x0 = rng.standard_normal((B,N,N,N)) + 1j*rng.standard_normal((B,N,N,N))
    c = 0.1*(rng.standard_normal((B,N,N,N)) + 1j*rng.standard_normal((B,N,N,N)))
    one = np.zeros(B*N**3, complex); fin = np.zeros(B*N**3, complex)
    best=1e9
    for r in range(5):
        t0=time.perf_counter()
        f(x0.ctypes.data, c.ctypes.data, one.ctypes.data, fin.ctypes.data, B, m)
        t1=time.perf_counter()
        best=min(best, t1-t0)
    es=B*m*N**3
    print(f'N={N} B={B} m={m}: {best:.4f}s  ns/el-step={1e9*best/es:.3f}')
