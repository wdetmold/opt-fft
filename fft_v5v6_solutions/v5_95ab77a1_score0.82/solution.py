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
                    _c, '-o', _so, '-lm'], check=True)   # add flags if needed
_lib = ctypes.CDLL(_so)
# ... your ctypes bindings and per-L create/plan calls here (import time) ...
_lib.ensure_arena.restype = ctypes.c_void_p
_lib.ensure_arena.argtypes = [ctypes.c_long, ctypes.c_int]
for _L in (6, 8, 13, 17, 23, 36, 45, 64):
    _f = getattr(_lib, f'run{_L}')
    _f.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long, ctypes.c_long,
                   ctypes.c_void_p, ctypes.c_void_p]
    _g = getattr(_lib, f'convert_all{_L}')
    _g.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long]

_lib.vs_doubles.restype = ctypes.c_long
_lib.vs_doubles.argtypes = [ctypes.c_long]
_lib.bps_doubles.restype = ctypes.c_long
_lib.bps_doubles.argtypes = [ctypes.c_long]
for _L in (6, 8, 13, 17):
    _f = getattr(_lib, f'runb{_L}')
    _f.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long, ctypes.c_long,
                   ctypes.c_void_p, ctypes.c_void_p]
    _g = getattr(_lib, f'convball{_L}')
    _g.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long]

def _vs(L):
    return _lib.vs_doubles(L)

_outcache = {}

def _aligned_empty(n, key):
    got = _outcache.get(key)
    if got is not None and got.shape[0] >= n:
        return got[:n]
    raw = np.empty(n + 4, dtype=np.complex128)
    off = (-raw.ctypes.data // 16) % 4
    v = raw[off:off+n]
    _outcache[key] = v
    return v

def _run(L, B, m, x0, c):
    L3 = L**3
    vs = _vs(L)
    one = _aligned_empty(B*L3, (L, 0))
    fin = _aligned_empty(B*L3, (L, 1))
    if B == 0:
        return one, fin
    nbytes = (B*2*vs + 1024)*8
    if L in (6, 8, 13, 17):
        PS = _lib.bps_doubles(L)
        G, rem = divmod(B, 8)
        nbytes = (G*2*PS + rem*2*vs + 1024)*8
    st = _lib.ensure_arena(nbytes, 0)
    cst = _lib.ensure_arena(nbytes + 4096, 1)
    if not st or not cst:
        raise MemoryError("arena allocation failed")
    cst += 1088
    if not x0.flags['C_CONTIGUOUS']:
        x0 = np.ascontiguousarray(x0)
    if not c.flags['C_CONTIGUOUS']:
        c = np.ascontiguousarray(c)
    G = rem = 0
    if L in (6, 8, 13, 17):
        G, rem = divmod(B, 8)
    if G:
        PS = _lib.bps_doubles(L)
        getattr(_lib, f'convball{L}')(x0.ctypes.data, st, G)
        getattr(_lib, f'convball{L}')(c.ctypes.data, cst, G)
        getattr(_lib, f'runb{L}')(st, cst, G, m, one.ctypes.data, fin.ctypes.data)
        if rem:
            d = G*8*L3*16
            s8 = G*2*PS*8
            getattr(_lib, f'convert_all{L}')(x0.ctypes.data + d, st + s8, rem)
            getattr(_lib, f'convert_all{L}')(c.ctypes.data + d, cst + s8, rem)
            getattr(_lib, f'run{L}')(st + s8, cst + s8, rem, m,
                                     one.ctypes.data + d, fin.ctypes.data + d)
    else:
        getattr(_lib, f'convert_all{L}')(x0.ctypes.data, st, B)
        getattr(_lib, f'convert_all{L}')(c.ctypes.data, cst, B)
        getattr(_lib, f'run{L}')(st, cst, B, m, one.ctypes.data, fin.ctypes.data)
    return one, fin

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
