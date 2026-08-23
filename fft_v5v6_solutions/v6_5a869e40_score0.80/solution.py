import os, subprocess, ctypes
import numpy as np
try:
    os.sched_setaffinity(0, {0})   # pin to one core: timing fairness
except Exception:
    pass

_here = os.path.dirname(os.path.abspath(__file__))
_c    = os.path.join(_here, 'implementation.c')   # <- your C file, written by you
_so   = os.path.join(_here, 'implementation.so')
def _build():
    subprocess.run(['gcc', '-O3', '-falign-loops=32', '-march=native', '-shared', '-fPIC',
                    _c, '-o', _so, '-lm'], check=True)   # add flags if needed
if not os.path.exists(_so):
    _build()
try:
    _lib = ctypes.CDLL(_so)
except OSError:
    _build()
    _lib = ctypes.CDLL(_so)
_lib.init_mem()
for _L in (6, 8, 13, 17, 23, 36, 45, 64):
    for _nm in ([f'run_{_L}'] + ([f'run4_{_L}', f'run2_{_L}'] if _L <= 45 else [])):
        _f = getattr(_lib, _nm)
        _f.restype = None
        _f.argtypes = [ctypes.c_long, ctypes.c_long,
                       ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]

_I4 = {6, 8, 13, 17, 23, 45}

_cache = {}

def _buf(key, n):
    a = _cache.get(key)
    if a is None or a.size < n:
        a = np.empty(n, dtype=np.complex128)
        a.fill(0)          # fault pages once
        _cache[key] = a
    return a[:n]

def _run(L, B, m, x0, c):
    B = int(B); m = int(m)
    n = B * L * L * L
    one = _buf(('o', L), n)
    final = _buf(('f', L), n)
    if n:
        xp, cp = x0.ctypes.data, c.ctypes.data
        op, fp = one.ctypes.data, final.ctypes.data
        vb = L * L * L * 16
        done = 0
        if L in _I4:
            G = B // 4
            if G:
                getattr(_lib, f'run4_{L}')(G, m, xp, cp, op, fp)
                done = 4 * G
            if B - done >= 2:
                off = done * vb
                getattr(_lib, f'run2_{L}')(1, m, xp + off, cp + off, op + off, fp + off)
                done += 2
        r = B - done
        if r:
            off = done * vb
            getattr(_lib, f'run_{L}')(r, m, xp + off, cp + off, op + off, fp + off)
    return one, final

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
