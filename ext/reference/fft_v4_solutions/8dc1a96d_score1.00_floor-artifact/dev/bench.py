import ctypes, numpy as np, time
lib = ctypes.CDLL("/workdir/dev/implementation.so")

def run(L, B, m, x0, c, out1, outm):
    fn = getattr(lib, f"run{L}")
    fn(ctypes.c_int64(B), ctypes.c_int64(m),
       ctypes.c_void_p(x0.ctypes.data), ctypes.c_void_p(c.ctypes.data),
       ctypes.c_void_p(out1.ctypes.data), ctypes.c_void_p(outm.ctypes.data))

rng = np.random.default_rng(7)
print(f"{'L':>3} {'B':>5} {'m':>4} {'time':>9} {'ns/el/iter':>11} {'cyc/el@2.6G':>11}")
for L, B, m in ((6,512,20),(8,256,20),(13,64,20),(17,32,20),(23,16,20),(36,8,20),(45,4,20),(64,2,20),
                (64,8,10),(36,24,10),(6,4096,5)):
    N = B*L**3
    x0 = (rng.standard_normal(2*N)).view(np.complex128)
    c  = (0.1*rng.standard_normal(2*N)).view(np.complex128)
    out1 = np.empty(N, np.complex128); outm = np.empty(N, np.complex128)
    run(L, B, 1, x0, c, out1, outm)  # warm
    best = 9e9
    for _ in range(3):
        t0 = time.perf_counter(); run(L, B, m, x0, c, out1, outm); t1 = time.perf_counter()
        best = min(best, t1-t0)
    nsel = best/ (N*m) * 1e9
    print(f"{L:>3} {B:>5} {m:>4} {best:9.4f} {nsel:11.3f} {nsel*2.6:11.2f}")
