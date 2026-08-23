import ctypes, numpy as np, time, sys
so = sys.argv[1] if len(sys.argv) > 1 else "/workdir/dev/implementation.so"
lib = ctypes.CDLL(so)
rng = np.random.default_rng(3)
print(f"{'L':>3} {'pz':>7} {'py':>7} {'px':>7} {'tot':>7}  (real values)")
for L in (6, 8, 13, 17, 23, 36, 45, 64):
    N = L**3
    x0 = rng.standard_normal(2*N).view(np.complex128)
    c = (0.1*rng.standard_normal(2*N)).view(np.complex128)
    px0 = ctypes.c_void_p(x0.ctypes.data); pc = ctypes.c_void_p(c.ctypes.data)
    seedfn = getattr(lib, f"dseed{L}")
    res = []
    for p in ("dpz","dpy","dpx"):
        fn = getattr(lib, f"{p}{L}")
        seedfn(px0, pc); fn(ctypes.c_long(max(1, 2_000_000//N)))  # ramp
        ts=[]
        K = ctypes.c_long(4)
        for _ in range(30):
            seedfn(px0, pc)
            fn(ctypes.c_long(1))
            t0=time.perf_counter(); fn(K); t1=time.perf_counter()
            ts.append((t1-t0)/4/N*1e9)
        res.append(np.median(ts))
    # full triple
    fn = getattr(lib, f"dtr{L}")
    seedfn(px0, pc); fn(ctypes.c_long(max(1, 2_000_000//N)))
    ts=[]
    for _ in range(30):
        seedfn(px0, pc); fn(ctypes.c_long(1))
        t0=time.perf_counter(); fn(ctypes.c_long(4)); t1=time.perf_counter()
        ts.append((t1-t0)/4/N*1e9)
    tri = np.median(ts)
    print(f"{L:>3} {res[0]:7.3f} {res[1]:7.3f} {res[2]:7.3f} {sum(res):7.3f}  triple={tri:7.3f}")
