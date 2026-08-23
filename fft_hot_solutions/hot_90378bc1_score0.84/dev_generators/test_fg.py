import ctypes, numpy as np, time, os
os.sched_setaffinity(0,{0})
lib = ctypes.CDLL('/tmp/g/impl_fg.so')
for N in (36,45):
    getattr(lib, f'init_{N}')()
    f = getattr(lib, f'run_{N}')
    f.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_long]*2

def ref_steps(x, c, m):
    one=None
    for k in range(m):
        z = np.fft.fftn(x, axes=(1,2,3)) + c
        x = z/(1.0+np.abs(z))
        if k==0: one=x.copy()
    return one, x

for N in (36, 45):
    B, m = 2, 3
    rng = np.random.default_rng(42+N)
    x0 = rng.standard_normal((B,N,N,N)) + 1j*rng.standard_normal((B,N,N,N))
    c = 0.1*(rng.standard_normal((B,N,N,N)) + 1j*rng.standard_normal((B,N,N,N)))
    one = np.zeros(B*N**3, complex); fin = np.zeros(B*N**3, complex)
    getattr(lib, f'run_{N}')(x0.ctypes.data, c.ctypes.data, one.ctypes.data, fin.ctypes.data, B, m)
    r1, rm = ref_steps(x0, c, m)
    e1 = np.linalg.norm(one - r1.ravel())/np.linalg.norm(r1)
    em_ = np.linalg.norm(fin - rm.ravel())/np.linalg.norm(rm)
    print(f'N={N}: one rel={e1:.3e}  final rel={em_:.3e}')
