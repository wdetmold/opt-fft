import os, subprocess, ctypes
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_c    = os.path.join(_here, 'implementation.c')   # <- your C file, written by you
_so   = os.path.join(_here, 'implementation.so')
if not os.path.exists(_so):
    subprocess.run(['gcc', '-O3', '-march=native', '-shared', '-fPIC',
                    _c, '-o', _so, '-lm'], check=False)   # add flags if needed

# ---- bindings and per-size setup (import time) ----
def _load(path):
    lib = ctypes.CDLL(path)
    lib.initlib.restype = None
    lib.initlib.argtypes = []
    lib.run_size.restype = None
    lib.run_size.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
                             ctypes.c_void_p, ctypes.c_void_p,
                             ctypes.c_void_p, ctypes.c_void_p]
    lib.initlib()
    return lib

try:
    _lib = _load(_so)
except OSError:
    # fallback: compile to a temp location (e.g. read-only source dir)
    import tempfile
    _tmp = os.path.join(tempfile.mkdtemp(prefix='impl'), 'implementation.so')
    subprocess.run(['gcc', '-O3', '-march=native', '-shared', '-fPIC',
                    _c, '-o', _tmp, '-lm'], check=True)
    _lib = _load(_tmp)

_pool = {}

def _outbuf(L, which, n):
    a = _pool.get((L, which))
    if a is None or a.shape[0] < n:
        a = np.empty(n, dtype=np.complex128)
        _pool[(L, which)] = a
    return a[:n]

def _run(L, B, m, x0, c):
    n = B * L * L * L
    out1 = _outbuf(L, 0, n)
    outm = _outbuf(L, 1, n)
    if B > 0:
        _lib.run_size(L, B, m,
                      x0.ctypes.data, c.ctypes.data,
                      out1.ctypes.data, outm.ctypes.data)
    return out1, outm

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
