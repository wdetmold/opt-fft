import os, subprocess, ctypes
import numpy as np
try:
    os.sched_setaffinity(0, {0})   # pin to one core: timing fairness
except Exception:
    pass

_here = os.path.dirname(os.path.abspath(__file__))
_c    = os.path.join(_here, 'implementation.c')   # <- your C file, written by you
_so   = os.path.join(_here, 'implementation.so')
if not os.path.exists(_so):
    subprocess.run(['gcc', '-O3', '-march=native', '-funroll-loops',
                    '-fno-math-errno', '-fno-trapping-math', '-shared', '-fPIC',
                    _c, '-o', _so, '-lm'], check=True)   # add flags if needed
_lib = ctypes.CDLL(_so)

# ---- bindings and per-L create/plan calls (import time) ----
# keep numpy's large per-call buffers on the reusable sbrk heap: avoids
# mmap/munmap page-fault churn on every transform() call.
try:
    _libc = ctypes.CDLL("libc.so.6", use_errno=True)
    _libc.mallopt(ctypes.c_int(-3), ctypes.c_int(1 << 30))   # M_MMAP_THRESHOLD
    _libc.mallopt(ctypes.c_int(-1), ctypes.c_int(1 << 30))   # M_TRIM_THRESHOLD
except Exception:
    pass

_lib.init_tables.restype = None
_lib.init_tables()
_RUNS = {}
for _L in (6, 8, 13, 17, 23, 36, 45, 64):
    _f = getattr(_lib, 'run%d' % _L)
    _f.restype = None
    _f.argtypes = [ctypes.c_longlong, ctypes.c_longlong,
                   ctypes.c_void_p, ctypes.c_void_p,
                   ctypes.c_void_p, ctypes.c_void_p]
    _RUNS[_L] = _f

_pool = {}
def _buf(key, n):
    a = _pool.get(key)
    if a is None or a.shape[0] < n:
        a = np.empty(n, dtype=np.complex128)
        _pool[key] = a
    return a[:n]

def _run(L, B, m, x0, c):
    B = int(B); m = int(m)
    n = B * L * L * L
    one = _buf((L, 0), n)
    fin = _buf((L, 1), n)
    if B > 0:
        if not x0.flags['C_CONTIGUOUS']:
            x0 = np.ascontiguousarray(x0)
        if not c.flags['C_CONTIGUOUS']:
            c = np.ascontiguousarray(c)
        _RUNS[L](B, m,
                 x0.ctypes.data, c.ctypes.data,
                 one.ctypes.data, fin.ctypes.data)
    return one, fin

def _warmup():
    # fault in internal arenas and exercise every dispatch path once so the
    # timed calls never pay one-time setup costs.
    for L in (6, 8, 13, 17, 23, 36, 45, 64):
        for B in (1, 9):
            x0 = np.full(B * L**3, 0.3 + 0.1j, dtype=np.complex128)
            c = np.full(B * L**3, 0.05 - 0.02j, dtype=np.complex128)
            _run(L, B, 2, x0, c)
_warmup()
# ---- end bindings ----

_SIZES = (6, 8, 13, 17, 23, 36, 45, 64)

def transform(seed, B6, B8, B13, B17, B23, B36, B45, B64, m6, m8, m13, m17, m23, m36, m45, m64):
    Bs = {6: B6, 8: B8, 13: B13, 17: B17, 23: B23, 36: B36, 45: B45, 64: B64}
    ms = {6: m6, 8: m8, 13: m13, 17: m17, 23: m23, 36: m36, 45: m45, 64: m64}
    outs = []
    for L in _SIZES:
        B = Bs[L]
        rng_x = np.random.default_rng(seed + L)
        x0 = (rng_x.standard_normal((B, L, L, L))
              + 1j * rng_x.standard_normal((B, L, L, L)))
        rng_c = np.random.default_rng(1000 + L)
        c = 0.1 * (rng_c.standard_normal((B, L, L, L))
              + 1j * rng_c.standard_normal((B, L, L, L)))
        one, final = _run(L, B, ms[L], x0, c)   # <- your call into C
        outs += [one.ravel(), final.ravel()]
    return np.concatenate(outs)
