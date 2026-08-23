import ctypes, numpy as np, time, sys
so = sys.argv[1] if len(sys.argv) > 1 else "/workdir/dev/implementation.so"
lib = ctypes.CDLL(so)
rng = np.random.default_rng(3)
print(f"{'L':>3} {'pz':>7} {'py':>7} {'px':>7} {'tot':>7}   ns/el steady")
tots = {}
for L in (6, 8, 13, 17, 23, 36, 45, 64):
    N = L**3
    x0 = rng.standard_normal(2*N).view(np.complex128)
    c = (0.1*rng.standard_normal(2*N)).view(np.complex128)
    getattr(lib, f"dseed{L}")(ctypes.c_void_p(x0.ctypes.data), ctypes.c_void_p(c.ctypes.data))
    res = []
    for p in ("dpz","dpy","dpx"):
        fn = getattr(lib, f"{p}{L}")
        reps = max(1, 25_000_000 // N)
        fn(ctypes.c_long(reps))  # ramp + warm
        best = 9e9
        for _ in range(5):
            t0=time.perf_counter(); fn(ctypes.c_long(reps)); t1=time.perf_counter()
            best = min(best, t1-t0)
        res.append(best/reps/N*1e9)
    tots[L] = sum(res)
    print(f"{L:>3} {res[0]:7.3f} {res[1]:7.3f} {res[2]:7.3f} {sum(res):7.3f}")
print("TOTALS:", " ".join(f"{L}:{t:.2f}" for L,t in tots.items()))
