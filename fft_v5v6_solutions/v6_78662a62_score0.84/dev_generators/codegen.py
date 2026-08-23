#!/usr/bin/env python3
"""Development-time code generator that emits implementation.c:
fully-unrolled AVX-512 FFT codelets for L in (6,8,13,17,23,36,45,64),
fused with the elementwise map  x <- z/(1+|z|),  z = FFT3(x)+c.
The generated C file is the deliverable; this script is only a dev tool.
"""
import numpy as np

LD = np.longdouble
PI = LD('3.14159265358979323846264338327950288')
SIZES = (6, 8, 13, 17, 23, 36, 45, 64)
# padded slab pitch (in complex) for the scratch volume: chosen so the x-pass
# row stride maps to well-spread L1/L2 sets and rows stay 64B-aligned
PITCH = {6: 36, 8: 64, 13: 172, 17: 292, 23: 532, 36: 1300, 45: 2028, 64: 4116}

def tw(t, N):
    """w_N^t = exp(-2*pi*i*t/N) -> (re, im) as correctly-rounded doubles."""
    t = t % N
    ang = LD(-2) * PI * LD(t) / LD(N)
    return (float(np.cos(ang)), float(np.sin(ang)))

def flit(x):
    if x == int(x) and abs(x) < 1e15:
        return repr(float(x))
    return repr(x)

# ----------------------------------------------------------------------------
# Emitter: SSA-style straight-line vector code per width
# ----------------------------------------------------------------------------
class E:
    def __init__(self, w):
        self.w = w             # 1, 2, 4 complex lanes
        self.vt = {1: '__m128d', 2: '__m256d', 4: '__m512d'}[w]
        self.p = {1: '_mm', 2: '_mm256', 4: '_mm512'}[w]
        self.lines = []
        self.n = 0
        self.consts = {}

    def t(self):
        self.n += 1
        return 'v%d' % self.n

    def raw(self, line):
        self.lines.append(line)

    def emit(self, expr):
        v = self.t()
        self.lines.append('%s %s = %s;' % (self.vt, v, expr))
        return v

    # constants (cached per function)
    def k(self, val):
        key = repr(val)
        if key in self.consts:
            return self.consts[key]
        v = self.emit('%s_set1_pd(%s)' % (self.p, flit(val)))
        self.consts[key] = v
        return v

    def kpat(self, pat):  # patterned constant, pat = list of w*2 doubles (lane0 first)
        key = 'pat' + repr(pat)
        if key in self.consts:
            return self.consts[key]
        rev = ', '.join(flit(x) for x in reversed(pat))
        v = self.emit('%s_set_pd(%s)' % (self.p, rev))
        self.consts[key] = v
        return v

    # memory
    def load(self, addr):
        return self.emit('%s_loadu_pd(%s)' % (self.p, addr))

    def store(self, addr, v):
        self.raw('%s_storeu_pd(%s, %s);' % (self.p, addr, v))

    # arithmetic
    def add(self, a, b): return self.emit('%s_add_pd(%s, %s)' % (self.p, a, b))
    def sub(self, a, b): return self.emit('%s_sub_pd(%s, %s)' % (self.p, a, b))
    def mul(self, a, b): return self.emit('%s_mul_pd(%s, %s)' % (self.p, a, b))
    def fmadd(self, a, b, c): return self.emit('%s_fmadd_pd(%s, %s, %s)' % (self.p, a, b, c))
    def fmsub(self, a, b, c): return self.emit('%s_fmsub_pd(%s, %s, %s)' % (self.p, a, b, c))
    def fnmadd(self, a, b, c): return self.emit('%s_fnmadd_pd(%s, %s, %s)' % (self.p, a, b, c))
    def fmaddsub(self, a, b, c): return self.emit('%s_fmaddsub_pd(%s, %s, %s)' % (self.p, a, b, c))
    def maxv(self, a, b): return self.emit('%s_max_pd(%s, %s)' % (self.p, a, b))

    def swap(self, a):   # swap re/im in each complex
        imm = {1: '1', 2: '0x5', 4: '0x55'}[self.w]
        return self.emit('%s_permute_pd(%s, %s)' % (self.p, a, imm))

    def addsub(self, a, b):  # even: a-b, odd: a+b
        if self.w == 4:
            return self.fmaddsub(a, self.k(1.0), b)
        return self.emit('%s_addsub_pd(%s, %s)' % (self.p, a, b))

    # complex helpers ---------------------------------------------------------
    def cmul(self, z, wr, wi):
        """z * (wr + i*wi), constants."""
        s = self.swap(z)
        t = self.mul(s, self.k(wi))
        return self.fmaddsub(z, self.k(wr), t)

    def cmul_t(self, z, t):
        return self.cmul(z, t[0], t[1])

    def muli(self, a):   # i*a = [-im, re]
        s = self.swap(a)
        pat = [-1.0, 1.0] * self.w
        return self.mul(s, self.kpat(pat))

    def mulni(self, a):  # -i*a = [im, -re]
        s = self.swap(a)
        pat = [1.0, -1.0] * self.w
        return self.mul(s, self.kpat(pat))

    def add_i(self, a, b):   # a + i*b  = addsub(a, swap(b))
        return self.addsub(a, self.swap(b))

    def code(self):
        return '\n    '.join(self.lines)

# ----------------------------------------------------------------------------
# DFT building blocks on SSA values
# ----------------------------------------------------------------------------
def dft2(E, x):
    return [E.add(x[0], x[1]), E.sub(x[0], x[1])]

def dft4(E, x):
    s0 = E.add(x[0], x[2]); s1 = E.sub(x[0], x[2])
    s2 = E.add(x[1], x[3]); s3 = E.sub(x[1], x[3])
    X0 = E.add(s0, s2); X2 = E.sub(s0, s2)
    X3 = E.add_i(s1, s3)                      # s1 + i*s3
    X1 = E.fmsub(s1, E.k(2.0), X3)            # s1 - i*s3
    return [X0, X1, X2, X3]

def dft_prime(E, x, P):
    """odd prime (or odd) P via cos/sin symmetric pairs. x: list of P values."""
    h = (P - 1) // 2
    s = [None] * (h + 1); d = [None] * (h + 1)
    for j in range(1, h + 1):
        s[j] = E.add(x[j], x[P - j])
        d[j] = E.sub(x[j], x[P - j])
    # X0
    acc = E.add(x[0], s[1])
    for j in range(2, h + 1):
        acc = E.add(acc, s[j])
    out = [None] * P
    out[0] = acc
    for k in range(1, h + 1):
        re1, im1 = tw(k, P)
        A = E.fmadd(s[1], E.k(re1), x[0])
        O = E.mul(d[1], E.k(im1))
        for j in range(2, h + 1):
            re, im = tw(j * k, P)
            A = E.fmadd(s[j], E.k(re), A)
            O = E.fmadd(d[j], E.k(im), O)
        Xk = E.add_i(A, O)                    # A + i*O
        XPk = E.fmsub(A, E.k(2.0), Xk)        # A - i*O
        out[k] = Xk
        out[P - k] = XPk
    return out

def dft3(E, x):
    return dft_prime(E, x, 3)

def dft5(E, x):
    return dft_prime(E, x, 5)

def dft6(E, x):
    Ev = dft3(E, [x[0], x[2], x[4]])
    Od = dft3(E, [x[1], x[3], x[5]])
    T = [Od[0], E.cmul_t(Od[1], tw(1, 6)), E.cmul_t(Od[2], tw(2, 6))]
    out = [None] * 6
    for k in range(3):
        out[k] = E.add(Ev[k], T[k])
        out[k + 3] = E.sub(Ev[k], T[k])
    return out

def dft8(E, x):
    Ev = dft4(E, [x[0], x[2], x[4], x[6]])
    Od = dft4(E, [x[1], x[3], x[5], x[7]])
    T = [Od[0], E.cmul_t(Od[1], tw(1, 8)), E.mulni(Od[2]), E.cmul_t(Od[3], tw(3, 8))]
    out = [None] * 8
    for k in range(4):
        out[k] = E.add(Ev[k], T[k])
        out[k + 4] = E.sub(Ev[k], T[k])
    return out

def dft9(E, x):
    A = [dft3(E, [x[j2], x[3 + j2], x[6 + j2]]) for j2 in range(3)]
    # twiddle A[j2][k1] by w9^(j2*k1)
    B = [[None] * 3 for _ in range(3)]
    for j2 in range(3):
        for k1 in range(3):
            t = (j2 * k1) % 9
            B[j2][k1] = A[j2][k1] if t == 0 else E.cmul_t(A[j2][k1], tw(t, 9))
    out = [None] * 9
    for k1 in range(3):
        Y = dft3(E, [B[0][k1], B[1][k1], B[2][k1]])
        for k2 in range(3):
            out[k1 + 3 * k2] = Y[k2]
    return out

# ----------------------------------------------------------------------------
# Size-level codelets: produce (k, value) store sequence through env
# ----------------------------------------------------------------------------
def fft_size(E, L, ld, st):
    """ld(j)->value, st(k, value). Emits a full L-point DFT."""
    if L == 6:
        x = [ld(j) for j in range(6)]
        out = dft6(E, x)
        order = [0, 1, 2, 3, 4, 5]
        for k in order:
            st(k, out[k])
    elif L == 8:
        x = [ld(j) for j in range(8)]
        out = dft8(E, x)
        for k in range(8):
            st(k, out[k])
    elif L in (13, 17, 23):
        P = L; h = (P - 1) // 2
        x = [ld(j) for j in range(P)]
        # emit in pair order so the fused map pairs nicely
        s = [None] * (h + 1); d = [None] * (h + 1)
        for j in range(1, h + 1):
            s[j] = E.add(x[j], x[P - j])
            d[j] = E.sub(x[j], x[P - j])
        acc = E.add(x[0], s[1])
        for j in range(2, h + 1):
            acc = E.add(acc, s[j])
        st(0, acc)
        for k in range(1, h + 1):
            re1, im1 = tw(k, P)
            A = E.fmadd(s[1], E.k(re1), x[0])
            O = E.mul(d[1], E.k(im1))
            for j in range(2, h + 1):
                re, im = tw(j * k, P)
                A = E.fmadd(s[j], E.k(re), A)
                O = E.fmadd(d[j], E.k(im), O)
            Xk = E.add_i(A, O)
            XPk = E.fmsub(A, E.k(2.0), Xk)
            st(k, Xk)
            st(P - k, XPk)
    elif L == 36:
        # PFA: N1=4, N2=9 ; j=(9a+4b)%36 ; k=(9k1+28k2)%36
        S = []
        for a in range(4):
            xs = [ld((9 * a + 4 * b) % 36) for b in range(9)]
            S.append(dft9(E, xs))
        for k2 in range(9):
            Y = dft4(E, [S[a][k2] for a in range(4)])
            for k1 in range(4):
                st((9 * k1 + 28 * k2) % 36, Y[k1])
    elif L == 45:
        # PFA: N1=9, N2=5 ; j=(5a+9b)%45 ; k=(10k1+36k2)%45
        S = []
        for a in range(9):
            xs = [ld((5 * a + 9 * b) % 45) for b in range(5)]
            S.append(dft5(E, xs))
        for k2 in range(5):
            Y = dft9(E, [S[a][k2] for a in range(9)])
            for k1 in range(9):
                st((10 * k1 + 36 * k2) % 45, Y[k1])
    elif L == 64:
        # CT: 8 x 8 with twiddles
        A = [None] * 8
        for j2 in range(8):
            xs = [ld(8 * j1 + j2) for j1 in range(8)]
            F = dft8(E, xs)
            A[j2] = [None] * 8
            for k1 in range(8):
                t = (j2 * k1) % 64
                A[j2][k1] = F[k1] if t == 0 else E.cmul_t(F[k1], tw(t, 64))
        for k1 in range(8):
            Y = dft8(E, [A[j2][k1] for j2 in range(8)])
            for k2 in range(8):
                st(k1 + 8 * k2, Y[k2])
    else:
        raise ValueError(L)

# ----------------------------------------------------------------------------
# map emission: out = z / (1 + |z|) on pairs of vectors
# ----------------------------------------------------------------------------
def emit_map_flush(E, items, newton=True):
    """items: list of (k_store_expr_offset, value). Emits batched map with
    stage-interleaved chains for ILP, returns list of (koff, outvalue)."""
    n = len(items)
    pairs = []
    i = 0
    while i + 2 <= n:
        pairs.append((items[i], items[i + 1])); i += 2
    odd = items[i] if i < n else None
    mags = []
    # stage: magnitudes
    for (ka, va), (kb, vb) in pairs:
        pa = E.mul(va, va)
        pb = E.mul(vb, vb)
        u = E.emit('%s_unpacklo_pd(%s, %s)' % (E.p, pa, pb))
        v = E.emit('%s_unpackhi_pd(%s, %s)' % (E.p, pa, pb))
        s = E.add(u, v)
        mags.append(E.maxv(s, E.k(1e-280)))
    if odd is not None:
        pa = E.mul(odd[1], odd[1])
        ps = E.swap(pa)
        s = E.add(pa, ps)
        mags.append(E.maxv(s, E.k(1e-280)))
    ds = emit_inv1psqrt_batch(E, mags, newton)
    outs = []
    ilo = {1: '0', 2: '0x0', 4: '0x00'}[E.w]
    ihi = {1: '3', 2: '0xF', 4: '0xFF'}[E.w]
    for idx, ((ka, va), (kb, vb)) in enumerate(pairs):
        d = ds[idx]
        da = E.emit('%s_permute_pd(%s, %s)' % (E.p, d, ilo))
        db = E.emit('%s_permute_pd(%s, %s)' % (E.p, d, ihi))
        outs.append((ka, E.mul(va, da)))
        outs.append((kb, E.mul(vb, db)))
    if odd is not None:
        outs.append((odd[0], E.mul(odd[1], ds[-1])))
    return outs

def emit_inv1psqrt_batch(E, ss, newton):
    """batched 1/(1+sqrt(s)) with stage-interleaved emission across chains."""
    n = len(ss)
    if not newton:
        ts = [E.emit('%s_sqrt_pd(%s)' % (E.p, s)) for s in ss]
        ws = [E.add(t, E.k(1.0)) for t in ts]
        one = E.k(1.0)
        return [E.emit('%s_div_pd(%s, %s)' % (E.p, one, w)) for w in ws]
    half = E.k(0.5); c15 = E.k(1.5); one = E.k(1.0); two = E.k(2.0)
    rs = [E.emit('%s_rsqrt14_pd(%s)' % (E.p, s)) for s in ss]
    hs = [E.mul(s, half) for s in ss]
    rrs = [E.mul(r, r) for r in rs]
    es = [E.fnmadd(hs[i], rrs[i], c15) for i in range(n)]
    rs = [E.mul(rs[i], es[i]) for i in range(n)]
    t0s = [E.mul(ss[i], rs[i]) for i in range(n)]
    rhs = [E.mul(rs[i], half) for i in range(n)]
    dds = [E.fnmadd(t0s[i], t0s[i], ss[i]) for i in range(n)]
    ts = [E.fmadd(rhs[i], dds[i], t0s[i]) for i in range(n)]
    ws = [E.add(ts[i], one) for i in range(n)]
    qs = [E.emit('%s_rcp14_pd(%s)' % (E.p, w)) for w in ws]
    e1s = [E.fnmadd(ws[i], qs[i], two) for i in range(n)]
    qs = [E.mul(qs[i], e1s[i]) for i in range(n)]
    e2s = [E.fnmadd(ws[i], qs[i], two) for i in range(n)]
    qs = [E.mul(qs[i], e2s[i]) for i in range(n)]
    return qs

def emit_map_pair(E, va, vb, newton=True):
    """returns (outa, outb); va/vb each hold w complex (z already includes +c)."""
    pa = E.mul(va, va)
    pb = E.mul(vb, vb)
    u = E.emit('%s_unpacklo_pd(%s, %s)' % (E.p, pa, pb))
    v = E.emit('%s_unpackhi_pd(%s, %s)' % (E.p, pa, pb))
    s = E.add(u, v)                       # [ma0, mb0, ma1, mb1, ...]
    s = E.maxv(s, E.k(1e-280))
    d = emit_inv1psqrt(E, s, newton)
    da = E.emit('%s_permute_pd(%s, %s)' % (E.p, d, {1: '0', 2: '0x0', 4: '0x00'}[E.w]))
    db = E.emit('%s_permute_pd(%s, %s)' % (E.p, d, {1: '3', 2: '0xF', 4: '0xFF'}[E.w]))
    return E.mul(va, da), E.mul(vb, db)

def emit_map_single(E, va, newton=True):
    pa = E.mul(va, va)
    ps = E.swap(pa)
    s = E.add(pa, ps)                     # [m, m] duplicated per complex
    s = E.maxv(s, E.k(1e-280))
    d = emit_inv1psqrt(E, s, newton)
    return E.mul(va, d)

def emit_inv1psqrt(E, s, newton):
    """1/(1+sqrt(s)) elementwise on all lanes."""
    if not newton:
        t = E.emit('%s_sqrt_pd(%s)' % (E.p, s))
        w1 = E.add(t, E.k(1.0))
        return E.emit('%s_div_pd(%s, %s)' % (E.p, E.k(1.0), w1))
    half = E.k(0.5)
    r = E.emit('%s_rsqrt14_pd(%s)' % (E.p, s))
    # one Newton iteration: r = r*(1.5 - 0.5*s*r^2)
    h = E.mul(s, half)
    rr = E.mul(r, r)
    e = E.fnmadd(h, rr, E.k(1.5))
    r = E.mul(r, e)
    # t = s*r ; correction t = t + 0.5*r*(s - t*t)
    t0 = E.mul(s, r)
    rh = E.mul(r, half)
    dd = E.fnmadd(t0, t0, s)
    t = E.fmadd(rh, dd, t0)
    # w = 1 + t ; reciprocal
    w1 = E.add(t, E.k(1.0))
    import codegen_split as _cs
    if _cs.MAP_RCP_DIV:
        return E.emit('%s_div_pd(%s, %s)' % (E.p, E.k(1.0), w1))
    q = E.emit('%s_rcp14_pd(%s)' % (E.p, w1))
    e1 = E.fnmadd(w1, q, E.k(2.0))
    q = E.mul(q, e1)
    e2 = E.fnmadd(w1, q, E.k(2.0))
    q = E.mul(q, e2)
    return q

# ----------------------------------------------------------------------------
# codelet function generators
# ----------------------------------------------------------------------------
def gen_C_codelet(L, w, ES, fused, newton=True, name=None, prefetch=0):
    e = E(w)
    if name is None:
        name = 'f%d%s_w%d' % (L, 'x' if fused else 'y', w)
    if prefetch:
        for j in range(L):
            e.raw('_mm_prefetch((const char*)(p+%d), _MM_HINT_T0);' % (2 * j * ES + prefetch))
        if fused:
            for j in range(L):
                e.raw('_mm_prefetch((const char*)(c+%d), _MM_HINT_T0);' % (2 * j * ES))
    def ld(j):
        return e.load('p+%d' % (2 * j * ES))
    if not fused:
        def st(k, v):
            e.store('p+%d' % (2 * k * ES), v)
        fft_size(e, L, ld, st)
        return ('static inline __attribute__((always_inline)) void %s(double* restrict p){\n    %s\n}\n'
                % (name, e.code()))
    # fused: buffer outputs, flush as a batched map with 1-group lag (pipelined)
    pend = []
    GROUP = 8
    def flush(n):
        if not pend:
            return
        part = pend[:n]
        del pend[:n]
        outs = emit_map_flush(e, part, newton)
        for (ko, ov) in outs:
            e.store('p+%d' % (2 * ko * ES), ov)
    def st(k, v):
        cv = e.load('c+%d' % (2 * k * ES))
        zv = e.add(v, cv)
        pend.append((k, zv))
        if len(pend) > GROUP:
            flush(GROUP)
    fft_size(e, L, ld, st)
    while pend:
        flush(GROUP)
    return ('static inline __attribute__((always_inline)) void %s(double* restrict p, const double* restrict c){\n    %s\n}\n'
            % (name, e.code()))

def gen_S_codelet(L, w, name=None):
    e = E(w)
    if name is None:
        name = 'f%dz_w%d' % (L, w)
    def ld(j):
        offs = [2 * (j + r * L) for r in range(w)]
        if w == 4:
            return e.emit('ld4s(p+%d, p+%d, p+%d, p+%d)' % tuple(offs))
        elif w == 2:
            return e.emit('ld2s(p+%d, p+%d)' % tuple(offs))
        else:
            return e.load('p+%d' % offs[0])
    def st(k, v):
        offs = [2 * (k + r * L) for r in range(w)]
        if w == 4:
            e.raw('st4s(p+%d, p+%d, p+%d, p+%d, %s);' % (offs[0], offs[1], offs[2], offs[3], v))
        elif w == 2:
            e.raw('st2s(p+%d, p+%d, %s);' % (offs[0], offs[1], v))
        else:
            e.store('p+%d' % offs[0], v)
    fft_size(e, L, ld, st)
    return ('static inline __attribute__((always_inline)) void %s(double* restrict p){\n    %s\n}\n'
            % (name, e.code()))

# ----------------------------------------------------------------------------
# drivers
# ----------------------------------------------------------------------------
HEADER = r'''// Auto-generated specialized batched 3D FFT + nonlinear map iteration.
// Sizes: 6, 8, 13, 17, 23, 36, 45, 64. AVX-512 required. Single-threaded.
#include <immintrin.h>
#include <string.h>
#include <stdint.h>

static inline __attribute__((always_inline)) __m512d ld4s(const double* p0, const double* p1, const double* p2, const double* p3){
    __m256d a = _mm256_insertf128_pd(_mm256_castpd128_pd256(_mm_loadu_pd(p0)), _mm_loadu_pd(p1), 1);
    __m256d b = _mm256_insertf128_pd(_mm256_castpd128_pd256(_mm_loadu_pd(p2)), _mm_loadu_pd(p3), 1);
    return _mm512_insertf64x4(_mm512_castpd256_pd512(a), b, 1);
}
static inline __attribute__((always_inline)) void st4s(double* p0, double* p1, double* p2, double* p3, __m512d v){
    __m256d a = _mm512_castpd512_pd256(v);
    __m256d b = _mm512_extractf64x4_pd(v, 1);
    _mm_storeu_pd(p0, _mm256_castpd256_pd128(a));
    _mm_storeu_pd(p1, _mm256_extractf128_pd(a, 1));
    _mm_storeu_pd(p2, _mm256_castpd256_pd128(b));
    _mm_storeu_pd(p3, _mm256_extractf128_pd(b, 1));
}
static inline __attribute__((always_inline)) __m256d ld2s(const double* p0, const double* p1){
    return _mm256_insertf128_pd(_mm256_castpd128_pd256(_mm_loadu_pd(p0)), _mm_loadu_pd(p1), 1);
}
static inline __attribute__((always_inline)) void st2s(double* p0, double* p1, __m256d v){
    _mm_storeu_pd(p0, _mm256_castpd256_pd128(v));
    _mm_storeu_pd(p1, _mm256_extractf128_pd(v, 1));
}

#include <sys/mman.h>
static double* hp_alloc(size_t bytes){
    size_t two_mb = (size_t)1 << 21;
    size_t sz = (bytes + two_mb - 1) & ~(two_mb - 1);
    void* p = mmap(0, sz + two_mb, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return 0;
    uintptr_t a = ((uintptr_t)p + (two_mb - 1)) & ~(uintptr_t)(two_mb - 1);
    madvise((void*)a, sz, MADV_HUGEPAGE);
    memset((void*)a, 0, sz);
    return (double*)a;
}
'''

def tail_plan(n, maxw):
    """split n into chunks of width 4/2/1 (<= maxw)."""
    plan = []
    i = 0
    while i < n:
        for w in (4, 2, 1):
            if w <= maxw and i + w <= n:
                plan.append((i, w))
                i += w
                break
    return plan

def gen_size(L, newton=True):
    out = []
    L2 = L * L
    PIT = PITCH[L]
    widths_needed = set(w for _, w in tail_plan(L, 4))
    # z codelets (flavor S; w1 z == contiguous C with ES=1)
    for w in sorted(widths_needed):
        if w == 1:
            out.append(gen_C_codelet(L, 1, 1, False, name='f%dz_w1' % L))
        else:
            out.append(gen_S_codelet(L, w))
    # y codelets (flavor C, ES=L)
    for w in sorted(widths_needed):
        out.append(gen_C_codelet(L, w, L, False, name='f%dy_w%d' % (L, w)))
    # x fused codelets (flavor C, ES=PIT)
    xw = set(w for _, w in tail_plan(L2, 4))
    pfd = 32 if L >= 36 else 0   # prefetch 32 doubles (=4 codelet-calls) ahead
    for w in sorted(xw):
        out.append(gen_C_codelet(L, w, PIT, True, newton=newton, name='f%dx_w%d' % (L, w),
                                 prefetch=(pfd if w == 4 else 0)))
    # plain x codelet for debug fft3d
    out.append(gen_C_codelet(L, 4, PIT, False, name='f%dxp_w4' % L))
    if 1 in xw:
        out.append(gen_C_codelet(L, 1, PIT, False, name='f%dxp_w1' % L))

    # scratch buffers + init
    out.append('static double* xb%d; static double* cb%d;\n'
               'void init%d(void){ if (!xb%d){ xb%d = hp_alloc(%d); cb%d = hp_alloc(%d); } }\n'
               % (L, L, L, L, L, 16 * L * PIT, L, 16 * L * PIT))
    # step driver: w4 calls in rolled loops (pragma unroll 1), tails explicit
    n4z = (L // 4) * 4
    zcalls = []
    if n4z:
        zcalls.append('_Pragma("GCC unroll 1") for (int r = 0; r < %d; r += 4) f%dz_w4(p + r * %d);' % (n4z, L, 2 * L))
    for off, w in tail_plan(L, 4):
        if off >= n4z:
            zcalls.append('f%dz_w%d(p + %d);' % (L, w, off * 2 * L))
    ycalls = []
    if n4z:
        ycalls.append('_Pragma("GCC unroll 1") for (int q = 0; q < %d; q += 4) f%dy_w4(p + q * 2);' % (n4z, L))
    for off, w in tail_plan(L, 4):
        if off >= n4z:
            ycalls.append('f%dy_w%d(p + %d);' % (L, w, off * 2))

    step = []
    step.append('static void step%d(double* restrict x, const double* restrict c){' % L)
    step.append('  for (int s = 0; s < %d; s++){' % L)
    step.append('    double* p = x + (long)s * %d;' % (2 * PIT))
    for call in zcalls:
        step.append('    ' + call)
    for call in ycalls:
        step.append('    ' + call)
    step.append('  }')
    step.append('  {')
    # x pass loop: group w4 calls into a loop
    n4 = (L2 // 4) * 4
    step.append('    _Pragma("GCC unroll 1") for (long q = 0; q < %d; q += 4){ f%dx_w4(x + q*2, c + q*2); }' % (n4, L))
    rem = L2 - n4
    off = n4
    while rem > 0:
        w = 2 if rem >= 2 else 1
        if w == 2 and rem >= 2:
            step.append('    f%dx_w2(x + %d, c + %d);' % (L, off * 2, off * 2))
            off += 2; rem -= 2
        else:
            step.append('    f%dx_w1(x + %d, c + %d);' % (L, off * 2, off * 2))
            off += 1; rem -= 1
    step.append('  }')
    step.append('}')
    out.append('\n'.join(step) + '\n')

    # debug fft3d (no map, no c); copies through the padded scratch
    dbg = []
    dbg.append('void fft3d_%d(double* restrict x){' % L)
    dbg.append('  init%d();' % L)
    dbg.append('  double* restrict xb = xb%d;' % L)
    dbg.append('  for (int s = 0; s < %d; s++) memcpy(xb + (long)s * %d, x + (long)s * %d, %d);'
               % (L, 2 * PIT, 2 * L2, 16 * L2))
    dbg.append('  for (int s = 0; s < %d; s++){' % L)
    dbg.append('    double* p = xb + (long)s * %d;' % (2 * PIT))
    for call in zcalls:
        dbg.append('    ' + call)
    for call in ycalls:
        dbg.append('    ' + call)
    dbg.append('  }')
    n4 = (L2 // 4) * 4
    dbg.append('  _Pragma("GCC unroll 1") for (long q = 0; q < %d; q += 4){ f%dxp_w4(xb + q*2); }' % (n4, L))
    for i in range(n4, L2):
        dbg.append('  f%dxp_w1(xb + %d);' % (L, i * 2))
    dbg.append('  for (int s = 0; s < %d; s++) memcpy(x + (long)s * %d, xb + (long)s * %d, %d);'
               % (L, 2 * L2, 2 * PIT, 16 * L2))
    dbg.append('}')
    out.append('\n'.join(dbg) + '\n')

    # run entry (buffered in huge-page padded scratch, volume-resident iteration)
    run = []
    run.append('void run%d(double* restrict x, const double* restrict c, double* restrict o1, long B, long m){' % L)
    run.append('  const long vol2 = %d;' % (2 * L * L2))
    run.append('  init%d();' % L)
    run.append('  double* restrict xb = xb%d; double* restrict cb = cb%d;' % (L, L))
    run.append('  for (long v = 0; v < B; v++){')
    run.append('    for (int s = 0; s < %d; s++){' % L)
    run.append('      memcpy(xb + (long)s * %d, x + v * vol2 + (long)s * %d, %d);' % (2 * PIT, 2 * L2, 16 * L2))
    run.append('      memcpy(cb + (long)s * %d, c + v * vol2 + (long)s * %d, %d);' % (2 * PIT, 2 * L2, 16 * L2))
    run.append('    }')
    run.append('    for (long t = 0; t < m; t++){')
    run.append('      step%d(xb, cb);' % L)
    run.append('      if (t == 0) for (int s = 0; s < %d; s++) memcpy(o1 + v * vol2 + (long)s * %d, xb + (long)s * %d, %d);'
               % (L, 2 * L2, 2 * PIT, 16 * L2))
    run.append('    }')
    run.append('    for (int s = 0; s < %d; s++) memcpy(x + v * vol2 + (long)s * %d, xb + (long)s * %d, %d);'
               % (L, 2 * L2, 2 * PIT, 16 * L2))
    run.append('  }')
    run.append('}')
    out.append('\n'.join(run) + '\n')
    return '\n'.join(out)

def main(newton=True, path='implementation.c'):
    parts = [HEADER]
    for L in SIZES:
        parts.append('// ======================= L = %d =======================' % L)
        parts.append(gen_size(L, newton=newton))
    src = '\n'.join(parts)
    with open(path, 'w') as f:
        f.write(src)
    print('wrote %s: %d lines' % (path, src.count('\n')))

if __name__ == '__main__':
    import sys
    newton = '--sqrt' not in sys.argv
    path = 'implementation.c'
    for a in sys.argv[1:]:
        if a.startswith('--out='):
            path = a[6:]
    main(newton=newton, path=path)
