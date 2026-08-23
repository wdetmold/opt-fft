import os, subprocess, ctypes
import numpy as np
try:
    os.sched_setaffinity(0, {0})   # pin to one core: timing fairness
except Exception:
    pass

_here = os.path.dirname(os.path.abspath(__file__))

def _build(cname, soname, flags):
    _c = os.path.join(_here, cname)
    _so = os.path.join(_here, soname)
    if not os.path.exists(_so):
        subprocess.run(['gcc'] + flags + ['-shared', '-fPIC', _c, '-o', _so, '-lm'],
                       check=True)
    return ctypes.CDLL(_so)

_libB = _build('impl_s81.c', 'impl_s81.so',
               ['-O3', '-march=native', '-ffp-contract=fast', '-fno-math-errno'])
_libC = _build('impl_3907.c', 'impl_3907.so',
               ['-O3', '-march=native', '-ffp-contract=fast', '-fno-math-errno', '-funroll-loops'])
_libD = _build('impl_mine.c', 'impl_mine.so',
               ['-O3', '-march=native', '-ffp-contract=fast', '-fno-math-errno'])
_libE = _build('impl_3f30.c', 'impl_3f30.so',
               ['-O3', '-march=native', '-funroll-loops', '-fno-math-errno', '-fno-trapping-math'])

try:
    _libc = ctypes.CDLL(None)
    _libc.mallopt(ctypes.c_int(-3), ctypes.c_int(1 << 30))
    _libc.mallopt(ctypes.c_int(-1), ctypes.c_int(1 << 30))
except Exception:
    pass

_SIZES = (6, 8, 13, 17, 23, 36, 45, 64)

# --- bindings ---
_libB.run_size.argtypes = [ctypes.c_long]*3 + [ctypes.c_void_p]*4 + [ctypes.c_long]
_libB.run_size.restype = None
_libC.init_all()
_libC.run.argtypes = [ctypes.c_int, ctypes.c_long, ctypes.c_long] + [ctypes.c_void_p]*4
_libC.run.restype = None
for _L in (13, 17, 23, 36, 45):
    _f = getattr(_libD, 'run_%d' % _L)
    _f.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_long]*2
    _f.restype = None
_libE.init_tables.restype = None
_libE.init_tables()
_libE.run64.restype = None
_libE.run64.argtypes = [ctypes.c_longlong, ctypes.c_longlong] + [ctypes.c_void_p]*4

# route per size (may depend on B): A=f40, B=s81, C=3907, D=mine
def _route(L, B):
    if L == 6: return 'C'
    if L == 8: return 'C'
    if L == 13: return 'D' if B >= 6 else 'C'
    if L == 17: return 'D' if B >= 6 else 'C'
    if L == 23: return 'D' if B >= 7 else 'B'
    if L == 36: return 'B' if B >= 7 else 'C'
    if L == 45: return 'B' if B >= 8 else 'C'
    return 'E'

_pool = {}
def _buf(key, n):
    a = _pool.get(key)
    if a is None or a.size < n:
        raw = np.empty(n + 4, np.complex128)
        off = (-raw.ctypes.data % 64) // 16
        a = raw[off:off+n]
        a[:] = 0.0
        _pool[key] = a
    return a[:n]

def _dispatch(L, B, m, px, pc, p1, pm):
    r = _route(L, B)
    if r == 'B':
        _libB.run_size(L, B, m, px, pc, p1, pm, 0)
    elif r == 'D':
        getattr(_libD, 'run_%d' % L)(px, pc, p1, pm, B, m)
    elif r == 'E':
        _libE.run64(B, m, px, pc, p1, pm)
    else:
        _libC.run(_SIZES.index(L), B, m, px, pc, p1, pm)

# tail-split: when the main route is the 8-lane-group engine (D), route the
# sub-group remainder to the per-volume-friendly engine instead of padding.
_TAILMAX = {13: 5, 17: 3, 23: 5}

def _run(L, B, m, x0, c):
    B = int(B); m = int(m); L = int(L)
    n = B * L * L * L
    one = _buf(('one', L), n)
    fin = _buf(('fin', L), n)
    if n:
        if m < 1:
            m = 1
        px, pc = x0.ctypes.data, c.ctypes.data
        p1, pm = one.ctypes.data, fin.ctypes.data
        rem = B % 8
        if (L in _TAILMAX and B >= 8 and 1 <= rem <= _TAILMAX[L]
                and _route(L, B) == 'D'):
            Bm = B - rem
            off = Bm * L * L * L * 16
            _dispatch(L, Bm, m, px, pc, p1, pm)
            _dispatch(L, rem, m, px + off, pc + off, p1 + off, pm + off)
        else:
            _dispatch(L, B, m, px, pc, p1, pm)
    return one, fin

def _warmup():
    for L in _SIZES:
        for B in (1, 9):
            x0 = np.full(B * L**3, 0.3 + 0.1j, dtype=np.complex128)
            c = np.full(B * L**3, 0.05 - 0.02j, dtype=np.complex128)
            _run(L, B, 2, x0, c)
_warmup()

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
