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
    subprocess.run(['gcc', '-O3', '-march=native', '-shared', '-fPIC',
                    '-ffp-contract=fast', '-fno-math-errno', '-funroll-loops',
                    _c, '-o', _so, '-lm'], check=True)   # add flags if needed

# ---- bindings and per-L create/plan calls (import time) ----
# Keep big numpy allocations (inputs generated per call, concatenated output)
# on the reusable heap instead of fresh mmaps, so repeated calls do not pay
# page-fault storms.  (M_MMAP_THRESHOLD = -3, M_TRIM_THRESHOLD = -1)
try:
    _libc = ctypes.CDLL(None)
    _libc.mallopt(ctypes.c_int(-3), ctypes.c_int(1 << 30))
    _libc.mallopt(ctypes.c_int(-1), ctypes.c_int(1 << 30))
except Exception:
    pass

_lib = ctypes.CDLL(_so)
_lib.init_all()          # allocate internal padded/split-format state buffers
_lib.run.argtypes = [ctypes.c_int, ctypes.c_long, ctypes.c_long,
                     ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
_lib.run.restype = None

_SIZES = (6, 8, 13, 17, 23, 36, 45, 64)
_pool = {}

def _get_buf(key, n):
    a = _pool.get(key)
    if a is None or a.size < n:
        a = np.empty(n, np.complex128)
        a[:] = 0.0          # touch pages once; reused across calls
        _pool[key] = a
    return a[:n]

def _run(L, B, m, x0, c):
    n = B * L * L * L
    one = _get_buf(('one', L), n)
    fin = _get_buf(('fin', L), n)
    if n:
        if m < 1:
            m = 1
        _lib.run(_SIZES.index(L), B, m,
                 x0.ctypes.data, c.ctypes.data, one.ctypes.data, fin.ctypes.data)
    return one, fin
# ---- end bindings ----

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
