import ctypes, numpy as np, time
lib = ctypes.CDLL("/workdir/dev/implementation.so")
rng = np.random.default_rng(3)
def timed(L, B, m, x0, c, o1, om):
    fn = getattr(lib, f"run{L}")
    a = (ctypes.c_int64(B), ctypes.c_int64(m), ctypes.c_void_p(x0.ctypes.data),
         ctypes.c_void_p(c.ctypes.data), ctypes.c_void_p(o1.ctypes.data), ctypes.c_void_p(om.ctypes.data))
    fn(*a)
    best = 9e9
    for _ in range(5):
        t0=time.perf_counter(); fn(*a); t1=time.perf_counter()
        best=min(best,t1-t0)
    return best
for L in (6,8,13,17,23,36,45,64):
    B = max(1, 2_000_000 // L**3)
    N = B*L**3
    x0 = rng.standard_normal(2*N).view(np.complex128); c = (0.1*rng.standard_normal(2*N)).view(np.complex128)
    o1 = np.empty(N, np.complex128); om = np.empty(N, np.complex128)
    m1, m2 = 3, 23
    t1 = timed(L,B,m1,x0,c,o1,om); t2 = timed(L,B,m2,x0,c,o1,om)
    per_iter = (t2-t1)/(m2-m1)/N*1e9
    fixed = (t1 - per_iter*m1*N/1e9)/N*1e9
    print(f"L={L:3d} B={B:5d}: marginal {per_iter:7.3f} ns/el/iter, fixed-per-call {fixed:7.3f} ns/el")
