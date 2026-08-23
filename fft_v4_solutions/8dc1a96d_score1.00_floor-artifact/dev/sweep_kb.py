import subprocess, ctypes, numpy as np, time, importlib
import gen2

def build(kbdual, out):
    import gen3
    gen2.DUAL_KB = kbdual
    # patch prime_kernel default table
    importlib.reload(gen3)
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
        r = []
        for p in ("dpy","dpx"):
            fn = getattr(lib, f"{p}{L}")
            reps = max(1, 25_000_000 // N)
            fn(ctypes.c_long(reps))
            best = 9e9
            for _ in range(5):
                t0=time.perf_counter(); fn(ctypes.c_long(reps)); t1=time.perf_counter()
                best = min(best, t1-t0)
            r.append(best/reps/N*1e9)
        out[L] = r
    return out

import sys
for kb in ({13:3,17:4,23:3}, {13:2,17:2,23:2}, {13:3,17:3,23:3}, {13:6,17:4,23:6}):
    build(kb, "/tmp/kb.so")
    r = meas("/tmp/kb.so", (13,17,23))
    print(kb, {k: [round(x,3) for x in v] for k,v in r.items()}, flush=True)
