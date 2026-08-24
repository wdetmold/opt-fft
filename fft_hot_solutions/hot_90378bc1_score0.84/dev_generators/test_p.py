import ctypes, numpy as np, os, time, sys
os.sched_setaffinity(0,{0})
lib = ctypes.CDLL(sys.argv[1] if len(sys.argv)>1 else '/tmp/g/impl_p.so')
for L,B,m in ((13,24,200),(17,16,150),(23,12,100)):
    getattr(lib,f'init_{L}')()
    f = getattr(lib,f'run_{L}'); f.argtypes=[ctypes.c_void_p]*4+[ctypes.c_long]*2
    rng = np.random.default_rng(5+L)
    x0 = rng.standard_normal((B,L,L,L)) + 1j*rng.standard_normal((B,L,L,L))
    c = 0.1*(rng.standard_normal((B,L,L,L)) + 1j*rng.standard_normal((B,L,L,L)))
    one = np.zeros(B*L**3, complex); fin = np.zeros(B*L**3, complex)
    # correctness (small m)
    f(x0.ctypes.data, c.ctypes.data, one.ctypes.data, fin.ctypes.data, B, 3)
    x = x0.copy(); on=None
    for k in range(3):
        z = np.fft.fftn(x, axes=(1,2,3)) + c
        x = z/(1+np.abs(z))
        if k==0: on=x.copy()
    e1 = np.linalg.norm(one-on.ravel())/np.linalg.norm(on)
    e2 = np.linalg.norm(fin-x.ravel())/np.linalg.norm(x)
    # timing
    best=9e9
    for r in range(5):
        t0=time.perf_counter(); f(x0.ctypes.data, c.ctypes.data, one.ctypes.data, fin.ctypes.data, B, m); t1=time.perf_counter()
        best=min(best,t1-t0)
    print(f'L={L}: one={e1:.2e} fin3={e2:.2e}  B={B},m={m}: {best:.4f}s = {1e9*best/(B*m*L**3):.3f} ns/el-step')
