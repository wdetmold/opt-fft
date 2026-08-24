# Iterated batched 3D complex FFTs, eight fixed cube sizes (single core, no FFT libraries).
# Engine lineup (per-size/per-batch routed, all hand-written AVX-512 DFT code):
#   - engines/impl_a.c   : warm-start engine (00291a90 family): SoA-8 batch lanes for 6/8/13/17/23
#                          (PFA(2x3)/radix-2 DFT8/symmetric-folded prime DFTs in inline asm),
#                          two-stage PFA/CT within-volume engines for 36/45/64, fused rsqrt14/rcp14 maps.
#   - engines/impl_s81.c : PFA batched-group engine (v5_8175a973): best for 36/45 at full 8-groups.
#   - engines/impl_b64.c : L=64 chain-resident engine (v6_3f30d81f): lanes = low x-bits, best for 64.
#   - engines/impl_v8p.c : this round's engine: ping-pong SoA-8 prime pipeline, best for 13 groups.
#   - engines/impl_v8b.c : this round's engine: within-volume L2-resident pipelined PFA(4x9),
#                          best for 36 at small batch (B<8).
import os, subprocess, ctypes
import numpy as np
try:
    os.sched_setaffinity(0, {0})   # pin to one core: timing fairness
except Exception:
    pass

# keep glibc from mmap()ing numpy's large per-call buffers: reuse heap pages
try:
    _libc = ctypes.CDLL("libc.so.6", use_errno=True)
    _libc.mallopt(ctypes.c_int(-3), ctypes.c_int(1 << 30))   # M_MMAP_THRESHOLD
    _libc.mallopt(ctypes.c_int(-1), ctypes.c_int(1 << 30))   # M_TRIM_THRESHOLD
except Exception:
    pass

_here = os.path.dirname(os.path.abspath(__file__))

def _build(cname, soname, flags):
    c = os.path.join(_here, cname)
    so = os.path.join(_here, soname)
    if not os.path.exists(so):
        subprocess.run(['gcc'] + flags + ['-shared', '-fPIC', c, '-o', so, '-lm'],
                       check=True)
    return ctypes.CDLL(so)

_FLAGS_A = ['-O3', '-march=native', '-funroll-loops', '-fschedule-insns', '-fsched-pressure']
_FLAGS_B = ['-O3', '-march=native', '-funroll-loops', '-fno-math-errno', '-fno-trapping-math']
_FLAGS_S = ['-O3', '-march=native', '-ffp-contract=fast', '-fno-math-errno']
_FLAGS_V = ['-O3', '-march=native', '-fno-math-errno', '-fno-trapping-math',
            '-fschedule-insns', '-fsched-pressure']

_libA = _build(os.path.join('engines', 'impl_a.c'), os.path.join('engines', 'impl_a.so'), _FLAGS_A)
_libB = _build(os.path.join('engines', 'impl_b64.c'), os.path.join('engines', 'impl_b64.so'), _FLAGS_B)
_libS = _build(os.path.join('engines', 'impl_s81.c'), os.path.join('engines', 'impl_s81.so'), _FLAGS_S)
_libV = _build(os.path.join('engines', 'impl_v8p.c'), os.path.join('engines', 'impl_v8p.so'), _FLAGS_V)
_libW = _build(os.path.join('engines', 'impl_v8b.c'), os.path.join('engines', 'impl_v8b.so'), _FLAGS_V)

_SIZES = (6, 8, 13, 17, 23, 36, 45, 64)
for _L in _SIZES:
    _f = getattr(_libA, f'run_{_L}')
    _f.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_long]*2
    _f.restype = None
_libB.init_tables.restype = None
_libB.init_tables()
_libB.run64.restype = None
_libB.run64.argtypes = [ctypes.c_longlong, ctypes.c_longlong] + [ctypes.c_void_p]*4
_libS.run_size.argtypes = [ctypes.c_long]*3 + [ctypes.c_void_p]*4 + [ctypes.c_long]
_libS.run_size.restype = None
for _L in (13, 17, 23):
    _f = getattr(_libV, f'run2_{_L}')
    _f.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_long]*2
    _f.restype = None
for _L in (36, 45, 64):
    _f = getattr(_libW, f'run_{_L}_g')
    _f.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_long]*2
    _f.restype = None

_pool = {}

def _buf(key, n):
    a = _pool.get(key)
    if a is None or a.shape[0] < n:
        a = np.empty(max(n, 1), dtype=np.complex128)
        _pool[key] = a
    return a[:n]

_VOL = {L: L*L*L for L in _SIZES}

def _padbuf(key, n):
    a = _pool.get(key)
    if a is None or a.shape[0] < n:
        a = np.zeros(max(n, 1), dtype=np.complex128)
        _pool[key] = a
    return a

def _run_padded8(L, rem, m, x0v, cv, onev, finalv, eoff, fn):
    # process rem (<8) trailing volumes as one zero-padded 8-group via fn
    V = _VOL[L]
    xin = _padbuf((L, 'px'), 8*V); cin = _padbuf((L, 'pc'), 8*V)
    o1 = _padbuf((L, 'p1'), 8*V);  om = _padbuf((L, 'pm'), 8*V)
    xin[:rem*V] = x0v.reshape(-1)[eoff:eoff + rem*V]
    xin[rem*V:8*V] = 0.0
    cin[:rem*V] = cv.reshape(-1)[eoff:eoff + rem*V]
    cin[rem*V:8*V] = 0.0
    fn(xin.ctypes.data, cin.ctypes.data, o1.ctypes.data, om.ctypes.data, 8, int(m))
    onev[eoff:eoff + rem*V] = o1[:rem*V]
    finalv[eoff:eoff + rem*V] = om[:rem*V]

def _run(L, B, m, x0, c):
    n = B * _VOL[L]
    one = _buf((L, 0), n)
    final = _buf((L, 1), n)
    if B > 0:
        x0 = np.ascontiguousarray(x0)
        c = np.ascontiguousarray(c)
        px, pc = x0.ctypes.data, c.ctypes.data
        p1, pm = one.ctypes.data, final.ctypes.data
        B, m = int(B), int(m)
        g = (B // 8) * 8
        off = g * _VOL[L] * 16
        if L == 64:
            _libB.run64(B, m, px, pc, p1, pm)
        elif L == 13:
            if g:
                _libV.run2_13(px, pc, p1, pm, g, m)
            rem = B - g
            if rem:
                if rem >= 6:
                    _run_padded8(13, rem, m, x0, c, one, final, g * _VOL[13], _libV.run2_13)
                else:
                    _libA.run_13(px + off, pc + off, p1 + off, pm + off, rem, m)
        elif L == 36:
            if g:
                _libS.run_size(36, g, m, px, pc, p1, pm, 0)
            if B - g:
                _libW.run_36_g(px + off, pc + off, p1 + off, pm + off, B - g, m)
        elif L == 45:
            if g:
                _libS.run_size(45, g, m, px, pc, p1, pm, 0)
            if B - g:
                _libA.run_45(px + off, pc + off, p1 + off, pm + off, B - g, m)
        elif L == 17:
            rem = B % 8
            if rem >= 5:
                if g:
                    _libA.run_17(px, pc, p1, pm, g, m)
                _run_padded8(17, rem, m, x0, c, one, final, g * _VOL[17], _libA.run_17)
            else:
                _libA.run_17(px, pc, p1, pm, B, m)
        else:
            getattr(_libA, f'run_{L}')(px, pc, p1, pm, B, m)
    return one, final

# import-time warmup: fault in arenas, exercise all dispatch paths
def _warmup():
    for L in _SIZES:
        for B, m in ((1, 2), (9, 3), (7, 2), (2, 1)):
            rng = np.random.default_rng(12345 + L + B)
            x0 = rng.standard_normal(B * L**3) + 1j * rng.standard_normal(B * L**3)
            cc = 0.1 * (rng.standard_normal(B * L**3) + 1j * rng.standard_normal(B * L**3))
            _run(L, B, m, x0.reshape(B, L, L, L), cc.reshape(B, L, L, L))
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
        one, final = _run(L, B, ms[L], x0, c)
        outs += [one.ravel(), final.ravel()]
    return np.concatenate(outs)
