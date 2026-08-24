import ctypes, numpy as np, time, os, sys
os.sched_setaffinity(0,{0})
lib = ctypes.CDLL(sys.argv[1])
res={}
cases = [(36, 8, 60), (45, 6, 40)]
data={}
for N,B,m in cases:
    getattr(lib, f'init_{N}')()
    rng = np.random.default_rng(1234+N)
    x0 = rng.standard_normal((B,N,N,N)) + 1j*rng.standard_normal((B,N,N,N))
    c = 0.1*(rng.standard_normal((B,N,N,N)) + 1j*rng.standard_normal((B,N,N,N)))
    one = np.zeros(B*N**3, complex); fin = np.zeros(B*N**3, complex)
    data[N]=(x0,c,one,fin,B,m)
for rep in range(9):
    for N,B,m in cases:
        x0,c,one,fin,B,m = data[N]
        f = getattr(lib, f'run_{N}'); f.argtypes=[ctypes.c_void_p]*4+[ctypes.c_long]*2
        t0=time.perf_counter(); f(x0.ctypes.data, c.ctypes.data, one.ctypes.data, fin.ctypes.data, B, m); t1=time.perf_counter()
        res.setdefault(N,[]).append(t1-t0)
for N,B,m in cases:
    ts=sorted(res[N])[:3]
    es=B*m*N**3
    print(f'N={N}: best3 {" ".join(f"{1e9*t/es:.3f}" for t in ts)} ns/el-step')
