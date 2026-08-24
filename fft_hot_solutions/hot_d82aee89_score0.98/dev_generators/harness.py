"""Dev harness: build a C file exposing run_L(x0,c,out1,outm,B,m); check vs numpy ref; time steady-state."""
import sys, os, time, subprocess, ctypes
import numpy as np

_SIZES = (6, 8, 13, 17, 23, 36, 45, 64)

def ref_transform(L, B, m, x0, c):
    # numpy reference (pocketfft ok for dev), returns (one, final)
    x = x0.copy()
    one = None
    for k in range(m):
        z = np.fft.fftn(x, axes=(1,2,3)) + c
        x = z / (1.0 + np.abs(z))
        if k == 0: one = x.copy()
    return one, x

def build(cfile, sofile, flags=None):
    flags = flags or ['-O3','-march=native','-funroll-loops']
    cmd = ['gcc'] + flags + ['-shared','-fPIC',cfile,'-o',sofile,'-lm']
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[:4000]); raise SystemExit(1)
    return ctypes.CDLL(os.path.abspath(sofile))

def bind(lib, L):
    f = getattr(lib, f'run_{L}')
    f.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_long]*2
    f.restype = None
    return f

def check(lib, L, B, m, seed=7, verbose=True):
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal((B,L,L,L)) + 1j*rng.standard_normal((B,L,L,L))
    c  = 0.1*(rng.standard_normal((B,L,L,L)) + 1j*rng.standard_normal((B,L,L,L)))
    n = B*L**3
    one = np.zeros(n, np.complex128); fin = np.zeros(n, np.complex128)
    f = bind(lib, L)
    f(x0.ctypes.data, c.ctypes.data, one.ctypes.data, fin.ctypes.data, B, m)
    r1, rm = ref_transform(L, B, m, x0, c)
    e1 = np.linalg.norm(one - r1.ravel())/np.linalg.norm(r1)
    em = np.linalg.norm(fin - rm.ravel())/np.linalg.norm(rm)
    if verbose: print(f"L={L} B={B} m={m}: err1={e1:.2e} errm={em:.2e}")
    return e1, em

def timeit(lib, L, B, m, reps=3, seed=7):
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal((B,L,L,L)) + 1j*rng.standard_normal((B,L,L,L))
    c  = 0.1*(rng.standard_normal((B,L,L,L)) + 1j*rng.standard_normal((B,L,L,L)))
    n = B*L**3
    one = np.zeros(n, np.complex128); fin = np.zeros(n, np.complex128)
    f = bind(lib, L)
    f(x0.ctypes.data, c.ctypes.data, one.ctypes.data, fin.ctypes.data, B, min(m,2))  # warm
    best = 1e18
    for _ in range(reps):
        t0 = time.perf_counter()
        f(x0.ctypes.data, c.ctypes.data, one.ctypes.data, fin.ctypes.data, B, m)
        dt = time.perf_counter() - t0
        best = min(best, dt)
    perpt = best/(B*m*L**3)
    print(f"L={L} B={B} m={m}: {best*1e3:8.3f} ms total, {perpt*1e9:7.3f} ns/pt-step ({perpt*2.6e9:6.2f} tsc/pt)")
    return perpt
