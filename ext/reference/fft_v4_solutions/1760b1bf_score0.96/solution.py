import os, subprocess, ctypes
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_c    = os.path.join(_here, 'implementation.c')   # <- your C file, written by you
_so   = os.path.join(_here, 'implementation.so')
if not os.path.exists(_so):
    _tmp = _so + '.tmp.%d' % os.getpid()
    subprocess.run(['gcc', '-O3', '-march=native', '-ffp-contract=fast', '-DPW_STYLE=2',
                    '-shared', '-fPIC', _c, '-o', _tmp, '-lm'], check=True)   # add flags if needed
    os.replace(_tmp, _so)
_lib = ctypes.CDLL(_so)
# --- ctypes bindings and per-L setup (import time) ---
_lib.setup.restype = None
_lib.setup()
_SIZES = (6, 8, 13, 17, 23, 36, 45, 64)
_MODE  = {6:'D', 8:'B', 13:'A', 17:'A', 23:'A', 36:'B', 45:'E', 64:'E'}
_GMODE = {6:2, 8:3, 13:3, 17:1, 23:3}      # sizes using the 4-volume batched path
_RUN, _RUNG = {}, {}
for _L in _SIZES:
    _f = getattr(_lib, 'run%d_%s' % (_L, _MODE[_L]))
    _f.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_long]*2
    _f.restype = None
    _RUN[_L] = _f
    _g = getattr(_lib, 'rung%d' % _L)
    _g.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_long]*2 + [ctypes.c_int]
    _g.restype = None
    _RUNG[_L] = _g

def _run(L, B, m, x0, c):
    n = B * L * L * L
    one = np.empty(n, dtype=np.complex128)
    final = np.empty(n, dtype=np.complex128)
    if B > 0:
        xp, cp = x0.ctypes.data, c.ctypes.data
        op, fp = one.ctypes.data, final.ctypes.data
        gm = _GMODE.get(L)
        NG = B // 4 if gm is not None else 0
        if NG > 0:
            _RUNG[L](xp, cp, op, fp, NG, m, gm)
        rem = B - 4*NG
        if rem > 0:
            off = 4*NG * L*L*L * 16
            _RUN[L](xp + off, cp + off, op + off, fp + off, rem, m)
    return one, final

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
