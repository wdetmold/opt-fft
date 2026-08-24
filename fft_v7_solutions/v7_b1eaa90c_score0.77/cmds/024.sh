cd /workdir && cat > solution.py <<'PYEOF'
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

_SIZES = (6, 8, 13, 17, 23, 36, 45, 64)
_FUNS = {}
for _L in _SIZES:
    _f = getattr(_lib, f'run{_L}')
    _f.restype = None
    _f.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long, ctypes.c_long,
                   ctypes.c_void_p, ctypes.c_void_p]
    _FUNS[_L] = _f

def _run(L, B, m, x0, c):
    n = B * L * L * L
    one = np.empty(n, dtype=np.complex128)
    final = np.empty(n, dtype=np.complex128)
    if B > 0:
        x0 = np.ascontiguousarray(x0)
        c = np.ascontiguousarray(c)
        _FUNS[L](x0.ctypes.data, c.ctypes.data, B, m,
                 one.ctypes.data, final.ctypes.data)
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
        one, final = _run(L, B, ms[L], x0, c)   # <- your call into C
        outs += [one.ravel(), final.ravel()]
    return np.concatenate(outs)
PYEOF
time python3 -c "import sys; sys.path.insert(0,'/workdir'); import solution" && ls -la /workdir/implementation.so