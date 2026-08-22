import os, subprocess, ctypes
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_c    = os.path.join(_here, 'implementation.c')   # <- your C file, written by you
_so   = os.path.join(_here, 'implementation.so')
if not os.path.exists(_so):
    subprocess.run(['gcc', '-O3', '-march=native', '-shared', '-fPIC',
                    _c, '-o', _so, '-lm',
                    '-ffp-contract=fast', '-fno-math-errno', '-fno-trapping-math'], check=True)
_lib = ctypes.CDLL(_so)
# --- ctypes bindings and per-L create/plan calls (import time) ---
_lib.setup()
_PD = ctypes.POINTER(ctypes.c_double)
_lib.run_size.argtypes = [ctypes.c_long, ctypes.c_long, ctypes.c_long, _PD, _PD, _PD, _PD]
_lib.run_size.restype = None

_out_cache = {}

def _outbuf(L, B, which):
    key = (L, B, which)
    buf = _out_cache.get(key)
    if buf is None:
        buf = np.empty((B, L, L, L), dtype=np.complex128)
        _out_cache[key] = buf
    return buf

def _run(L, B, m, x0, c):
    out1 = _outbuf(L, B, 0)
    outm = _outbuf(L, B, 1)
    if B > 0 and m > 0:
        _lib.run_size(ctypes.c_long(L), ctypes.c_long(B), ctypes.c_long(m),
                      x0.ctypes.data_as(_PD), c.ctypes.data_as(_PD),
                      out1.ctypes.data_as(_PD), outm.ctypes.data_as(_PD))
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
        one, final = _run(L, B, ms[L], x0, c)   # <- call into C
        outs += [one.ravel(), final.ravel()]
    return np.concatenate(outs)
