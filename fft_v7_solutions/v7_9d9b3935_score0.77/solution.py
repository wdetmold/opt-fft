import os, subprocess, ctypes
import numpy as np
try:
    os.sched_setaffinity(0, {0})   # pin to one core: timing fairness
except Exception:
    pass

_here = os.path.dirname(os.path.abspath(__file__))
_c    = os.path.join(_here, 'implementation.c')
_so   = os.path.join(_here, 'implementation.so')
if not os.path.exists(_so):
    subprocess.run(['gcc', '-O3', '-march=native', '-ffp-contract=fast',
                    '-fno-math-errno', '-fno-stack-protector', '-shared', '-fPIC',
                    _c, '-o', _so, '-lm'], check=True)
_lib = ctypes.CDLL(_so)
_lib.setup()
_PD = ctypes.POINTER(ctypes.c_double)
_PU = ctypes.POINTER(ctypes.c_uint64)
_SIZES = (6, 8, 13, 17, 23, 36, 45, 64)
_RUNS, _GENS = {}, {}
for _L in _SIZES:
    _f = getattr(_lib, f'run_{_L}')
    _f.argtypes = [ctypes.c_long, ctypes.c_long] + [_PD]*6
    _RUNS[_L] = _f
    _g = getattr(_lib, f'rungen_{_L}')
    _g.argtypes = [ctypes.c_long, ctypes.c_long, _PU, _PU, _PD, _PD]
    _GENS[_L] = _g
_lib.fill_normals_fast.argtypes = [_PU, ctypes.c_long, _PD]

def _pp(a):
    return a.ctypes.data_as(_PD)

def _words(seed):
    return np.random.SeedSequence(seed).generate_state(4, np.uint64)

# import-time self check: C normal generator must reproduce numpy bit-exactly
def _rng_selfcheck():
    try:
        for seed in (0, 1, 12345, 10**12 + 7):
            w = _words(seed)
            n = 200001
            got = np.empty(n)
            _lib.fill_normals_fast(w.ctypes.data_as(_PU), n, _pp(got))
            ref = np.random.default_rng(seed).standard_normal(n)
            if not np.array_equal(got, ref):
                return False
        return True
    except Exception:
        return False

_FAST_RNG = _rng_selfcheck()

def transform(seed, B6, B8, B13, B17, B23, B36, B45, B64, m6, m8, m13, m17, m23, m36, m45, m64):
    seed = int(seed)
    Bs = {6: int(B6), 8: int(B8), 13: int(B13), 17: int(B17), 23: int(B23), 36: int(B36), 45: int(B45), 64: int(B64)}
    ms = {6: int(m6), 8: int(m8), 13: int(m13), 17: int(m17), 23: int(m23), 36: int(m36), 45: int(m45), 64: int(m64)}
    total = 2 * sum(Bs[L] * L**3 for L in _SIZES)
    res = np.empty(total, dtype=np.complex128)
    base_ptr = res.ctypes.data
    off = 0
    for L in _SIZES:
        B = Bs[L]
        n = B * L**3
        p_one = ctypes.cast(base_ptr + 16 * off, _PD)
        p_fin = ctypes.cast(base_ptr + 16 * (off + n), _PD)
        if _FAST_RNG:
            wx = _words(seed + L)
            wc = _words(1000 + L)
            _GENS[L](B, ms[L], wx.ctypes.data_as(_PU), wc.ctypes.data_as(_PU), p_one, p_fin)
        else:
            rng_x = np.random.default_rng(seed + L)
            x0r = rng_x.standard_normal((B, L, L, L))
            x0i = rng_x.standard_normal((B, L, L, L))
            rng_c = np.random.default_rng(1000 + L)
            cr = rng_c.standard_normal((B, L, L, L))
            ci = rng_c.standard_normal((B, L, L, L))
            _RUNS[L](B, ms[L], _pp(x0r), _pp(x0i), _pp(cr), _pp(ci), p_one, p_fin)
        off += 2 * n
    return res
