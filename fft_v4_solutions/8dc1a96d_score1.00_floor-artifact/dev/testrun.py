import ctypes, numpy as np
lib = ctypes.CDLL("/workdir/dev/implementation.so")

def ref_iter(x0, c, m, L):
    x = x0.copy()
    one = None
    for k in range(m):
        z = np.fft.fftn(x, axes=(1,2,3)) + c
        x = z / (1.0 + np.abs(z))
        if k == 0: one = x.copy()
    return one, x

def run(L, B, m, x0, c):
    out1 = np.empty(B*L**3, np.complex128)
    outm = np.empty(B*L**3, np.complex128)
    fn = getattr(lib, f"run{L}")
    fn(ctypes.c_int64(B), ctypes.c_int64(m),
       ctypes.c_void_p(x0.ctypes.data), ctypes.c_void_p(c.ctypes.data),
       ctypes.c_void_p(out1.ctypes.data), ctypes.c_void_p(outm.ctypes.data))
    return out1, outm

rng = np.random.default_rng(7)
for L in (6, 8, 13, 17, 23, 36, 45, 64):
    for B, m in ((1,1),(2,3),(3,5),(2,8)):
        x0 = rng.standard_normal((B,L,L,L)) + 1j*rng.standard_normal((B,L,L,L))
        c  = 0.1*(rng.standard_normal((B,L,L,L)) + 1j*rng.standard_normal((B,L,L,L)))
        r1, rm = ref_iter(x0, c, m, L)
        g1, gm = run(L, B, m, x0, c)
        e1 = np.linalg.norm(g1 - r1.ravel())/np.linalg.norm(r1)
        em = np.linalg.norm(gm - rm.ravel())/np.linalg.norm(rm)
        st = "OK" if (e1 < 1e-14 and em < 1e-12) else "FAIL"
        print(f"L={L:3d} B={B} m={m}: e1={e1:.3e} em={em:.3e} {st}")
