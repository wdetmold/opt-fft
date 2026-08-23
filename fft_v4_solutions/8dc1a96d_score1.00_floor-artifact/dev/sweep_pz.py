import subprocess, ctypes, numpy as np, time, itertools, importlib, sys

def build(modes, out):
    import gen3
    importlib.reload(gen3)
    for L, m in modes.items():
        gen3.PZ_MODE[L] = m
    gen3.main()
    subprocess.run(["gcc","-O3","-march=native","-shared","-fPIC","implementation.c","-o",out,"-lm"], check=True)

def meas(so, Ls):
    lib = ctypes.CDLL(so)
    rng = np.random.default_rng(3)
    out = {}
    for L in Ls:
        N = L**3
        x0 = rng.standard_normal(2*N).view(np.complex128)
        c = (0.1*rng.standard_normal(2*N)).view(np.complex128)
        getattr(lib, f"dseed{L}")(ctypes.c_void_p(x0.ctypes.data), ctypes.c_void_p(c.ctypes.data))
        fn = getattr(lib, f"dpz{L}")
        reps = max(1, 25_000_000 // N)
        fn(ctypes.c_long(reps))
        best = 9e9
        for _ in range(5):
            t0=time.perf_counter(); fn(ctypes.c_long(reps)); t1=time.perf_counter()
            best = min(best, t1-t0)
        out[L] = best/reps/N*1e9
    return out

Ls = (6, 13, 17, 23, 45)
for mode in ("table", "dump"):
    build({L: mode for L in Ls}, f"/tmp/pz_{mode}.so")
    r = meas(f"/tmp/pz_{mode}.so", Ls)
    print(mode, {k: round(v,3) for k,v in r.items()})
