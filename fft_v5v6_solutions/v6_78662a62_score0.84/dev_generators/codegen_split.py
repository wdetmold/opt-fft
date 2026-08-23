#!/usr/bin/env python3
"""Code generator v2: split re/im plane layout, pure-zmm codelets.
Volume scratch layout (doubles):
  re plane: slab s at s*SP, row y at +y*RP, z contiguous [0..L) (+row pad to RP)
  im plane: same offsets + IMOFF
  row-group pad rows (up to ceil8(L)) and row pads are zero and stay zero.
"""
import numpy as np

LD = np.longdouble
PI = LD('3.14159265358979323846264338327950288')
SIZES = (6, 8, 13, 17, 23, 36, 45, 64)

def ceil8(n):
    return (n + 7) // 8 * 8

GEOM = {}
for L in SIZES:
    RP = ceil8(L)
    RG = ceil8(L)          # rows covered by z row-groups
    S0 = RG * RP
    pad = 8
    while (((S0 + pad) // 8) % 2) == 0:
        pad += 8
    SP = S0 + pad
    IMOFF = L * SP + 72
    BUFD = 2 * IMOFF + 16   # doubles per volume buffer (re+im planes)
    GEOM[L] = dict(RP=RP, RG=RG, SP=SP, IMOFF=IMOFF, BUFD=BUFD)

GEOM2 = {}
for L in (6, 8, 13):
    SP2 = L * L * 8
    if (SP2 // 8) % 8 == 0:   # avoid 4K-aliased slab stride
        SP2 += 8
    IM2 = L * SP2 + 64
    GEOM2[L] = dict(SP2=SP2, IM2=IM2, BUF2=2 * IM2 + 16)

def tw(t, N):
    t = t % N
    ang = LD(-2) * PI * LD(t) / LD(N)
    return (float(np.cos(ang)), float(np.sin(ang)))

def flit(x):
    return repr(float(x))

# ---------------------------------------------------------------------------
class E:
    VT = '__m512d'
    P = '_mm512'
    def __init__(self):
        self.lines = []
        self.pre = []
        self.n = 0
        self.consts = {}
    def t(self):
        self.n += 1
        return 'v%d' % self.n
    def raw(self, line):
        self.lines.append(line)
    def emit(self, expr):
        v = self.t()
        self.lines.append('%s %s = %s;' % (self.VT, v, expr))
        return v
    nocache = False
    def k(self, val):
        if self.nocache:
            return '_mm512_set1_pd(%s)' % flit(val)
        key = repr(val)
        if key not in self.consts:
            self.consts[key] = self.emit('_mm512_set1_pd(%s)' % flit(val))
        return self.consts[key]
    def kn(self, val):   # non-cached constant: folds into embedded broadcast
        return '_mm512_set1_pd(%s)' % flit(val)
    def zero(self):
        if 'zero' not in self.consts:
            self.consts['zero'] = self.emit('_mm512_setzero_pd()')
        return self.consts['zero']
    def load(self, addr):
        return self.emit('_mm512_loadu_pd(%s)' % addr)
    def store(self, addr, v):
        self.raw('_mm512_storeu_pd(%s, %s);' % (addr, v))
    def add(self, a, b): return self.emit('_mm512_add_pd(%s, %s)' % (a, b))
    def sub(self, a, b): return self.emit('_mm512_sub_pd(%s, %s)' % (a, b))
    def mul(self, a, b): return self.emit('_mm512_mul_pd(%s, %s)' % (a, b))
    def fmadd(self, a, b, c): return self.emit('_mm512_fmadd_pd(%s, %s, %s)' % (a, b, c))
    def fmsub(self, a, b, c): return self.emit('_mm512_fmsub_pd(%s, %s, %s)' % (a, b, c))
    def fnmadd(self, a, b, c): return self.emit('_mm512_fnmadd_pd(%s, %s, %s)' % (a, b, c))
    def maxv(self, a, b): return self.emit('_mm512_max_pd(%s, %s)' % (a, b))
    def code(self):
        return '\n    '.join(self.pre + self.lines)
    def scratch(self, name, nzmm):
        line = 'double %s[%d] __attribute__((aligned(64)));' % (name, nzmm * 8)
        if line not in self.pre:
            self.pre.append(line)

# complex pair ops ----------------------------------------------------------
def cadd(E, a, b): return (E.add(a[0], b[0]), E.add(a[1], b[1]))
def csub(E, a, b): return (E.sub(a[0], b[0]), E.sub(a[1], b[1]))
def cadd_i(E, a, b):   # a + i*b
    return (E.sub(a[0], b[1]), E.add(a[1], b[0]))
def csub_i(E, a, b):   # a - i*b
    return (E.add(a[0], b[1]), E.sub(a[1], b[0]))
def cmulw(E, a, w):    # a * (wr + i*wi) ; w = (wr, wi) floats
    wr, wi = w
    kr, ki = E.k(wr), E.k(wi)
    re = E.fmsub(a[0], kr, E.mul(a[1], ki))
    im = E.fmadd(a[0], ki, E.mul(a[1], kr))
    return (re, im)
def crmul(E, a, r):    # a * r (real const)
    kr = E.k(r)
    return (E.mul(a[0], kr), E.mul(a[1], kr))

# DFT blocks on pair values ---------------------------------------------------
def dft2(E, x):
    return [cadd(E, x[0], x[1]), csub(E, x[0], x[1])]

def dft4(E, x):
    s0 = cadd(E, x[0], x[2]); s1 = csub(E, x[0], x[2])
    s2 = cadd(E, x[1], x[3]); s3 = csub(E, x[1], x[3])
    return [cadd(E, s0, s2), csub_i(E, s1, s3), csub(E, s0, s2), cadd_i(E, s1, s3)]

def dft_prime(E, x, P):
    h = (P - 1) // 2
    s = [None]*(h+1); d = [None]*(h+1)
    for j in range(1, h+1):
        s[j] = cadd(E, x[j], x[P-j])
        d[j] = csub(E, x[j], x[P-j])
    acc = cadd(E, x[0], s[1])
    for j in range(2, h+1):
        acc = cadd(E, acc, s[j])
    out = [None]*P
    out[0] = acc
    for k in range(1, h+1):
        re1, im1 = tw(k, P)
        Ar = E.fmadd(s[1][0], E.k(re1), x[0][0]); Ai = E.fmadd(s[1][1], E.k(re1), x[0][1])
        Or = E.mul(d[1][0], E.k(im1)); Oi = E.mul(d[1][1], E.k(im1))
        for j in range(2, h+1):
            re, im = tw(j*k, P)
            Ar = E.fmadd(s[j][0], E.k(re), Ar); Ai = E.fmadd(s[j][1], E.k(re), Ai)
            Or = E.fmadd(d[j][0], E.k(im), Or); Oi = E.fmadd(d[j][1], E.k(im), Oi)
        A = (Ar, Ai); O = (Or, Oi)
        out[k] = cadd_i(E, A, O)      # A + i*O
        out[P-k] = csub_i(E, A, O)    # A - i*O
    return out

def dft3(E, x): return dft_prime(E, x, 3)
def dft5(E, x): return dft_prime(E, x, 5)

def dft6(E, x):
    # PFA 2x3: j=(3a+2b)%6, k=(3k1+4k2)%6 ; no twiddles
    S = []
    for a in range(2):
        S.append(dft3(E, [x[(3*a+2*b) % 6] for b in range(3)]))
    out = [None]*6
    for k2 in range(3):
        Y = dft2(E, [S[0][k2], S[1][k2]])
        for k1 in range(2):
            out[(3*k1+4*k2) % 6] = Y[k1]
    return out

def dft8(E, x):
    Ev = dft4(E, [x[0], x[2], x[4], x[6]])
    Od = dft4(E, [x[1], x[3], x[5], x[7]])
    T1 = cmulw(E, Od[1], tw(1,8))
    T3 = cmulw(E, Od[3], tw(3,8))
    out = [None]*8
    out[0] = cadd(E, Ev[0], Od[0]); out[4] = csub(E, Ev[0], Od[0])
    out[1] = cadd(E, Ev[1], T1);    out[5] = csub(E, Ev[1], T1)
    out[2] = csub_i(E, Ev[2], Od[2]); out[6] = cadd_i(E, Ev[2], Od[2])  # w8^2 = -i
    out[3] = cadd(E, Ev[3], T3);    out[7] = csub(E, Ev[3], T3)
    return out

def dft9(E, x):
    A = [dft3(E, [x[j2], x[3+j2], x[6+j2]]) for j2 in range(3)]
    B = [[None]*3 for _ in range(3)]
    for j2 in range(3):
        for k1 in range(3):
            t = (j2*k1) % 9
            B[j2][k1] = A[j2][k1] if t == 0 else cmulw(E, A[j2][k1], tw(t,9))
    out = [None]*9
    for k1 in range(3):
        Y = dft3(E, [B[0][k1], B[1][k1], B[2][k1]])
        for k2 in range(3):
            out[k1+3*k2] = Y[k2]
    return out

def emit_prime_tiled(e, P, ld, st, T=4):
    """prime DFT with s/d staged through scratch and k-tiling (low reg pressure)."""
    h = (P - 1) // 2
    sd = 'sd%d' % P
    e.scratch(sd, 4 * h)
    x0 = ld(0)
    X0r = None
    for j in range(1, h + 1):
        xj = ld(j); xPj = ld(P - j)
        s = cadd(e, xj, xPj); d = csub(e, xj, xPj)
        base = (j - 1) * 32
        e.store(sd + '+%d' % (base + 0), s[0])
        e.store(sd + '+%d' % (base + 8), s[1])
        e.store(sd + '+%d' % (base + 16), d[0])
        e.store(sd + '+%d' % (base + 24), d[1])
        if X0r is None:
            X0r = e.add(x0[0], s[0]); X0i = e.add(x0[1], s[1])
        else:
            X0r = e.add(X0r, s[0]); X0i = e.add(X0i, s[1])
    st(0, (X0r, X0i))
    ks = list(range(1, h + 1))
    for t0 in range(0, h, T):
        tile = ks[t0:t0 + T]
        Ar = {}; Ai = {}; Or = {}; Oi = {}
        for j in range(1, h + 1):
            base = (j - 1) * 32
            sr = e.load(sd + '+%d' % (base + 0))
            si = e.load(sd + '+%d' % (base + 8))
            dr = e.load(sd + '+%d' % (base + 16))
            di = e.load(sd + '+%d' % (base + 24))
            for k in tile:
                re, im = tw(j * k, P)
                cre = e.kn(re); cim = e.kn(im)
                if j == 1:
                    Ar[k] = e.fmadd(sr, cre, x0[0]); Ai[k] = e.fmadd(si, cre, x0[1])
                    Or[k] = e.mul(dr, cim); Oi[k] = e.mul(di, cim)
                else:
                    Ar[k] = e.fmadd(sr, cre, Ar[k]); Ai[k] = e.fmadd(si, cre, Ai[k])
                    Or[k] = e.fmadd(dr, cim, Or[k]); Oi[k] = e.fmadd(di, cim, Oi[k])
        for k in tile:
            A = (Ar[k], Ai[k]); O = (Or[k], Oi[k])
            st(k, cadd_i(e, A, O))
            st(P - k, csub_i(e, A, O))

def emit_pfa_scratch(e, L, N1, N2, ld, st):
    """PFA N1 x N2 with stage-1 results staged through scratch."""
    # stage1: for a in N1: DFT_N2 over b of x[(N2*a+N1*b)%L] -> S[a][k2]
    # scratch layout: sc[(k2*N1 + a)*16 + {0:re,8:im}]
    sc = 'pf%d' % L
    e.scratch(sc, 2 * N1 * N2)
    dftN2 = {5: dft5, 9: dft9}[N2]
    dftN1 = {4: dft4, 9: dft9}[N1]
    for a in range(N1):
        xs = [ld((N2 * a + N1 * b) % L) for b in range(N2)]
        S = dftN2(e, xs)
        for k2 in range(N2):
            slot = (k2 * N1 + a) * 16
            e.store(sc + '+%d' % slot, S[k2][0])
            e.store(sc + '+%d' % (slot + 8), S[k2][1])
    if N1 == 4:
        e1 = [i for i in range(L) if i % N1 == 1 and i % N2 == 0][0]
        e2 = [i for i in range(L) if i % N2 == 1 and i % N1 == 0][0]
    else:
        e1 = [i for i in range(L) if i % N1 == 1 and i % N2 == 0][0]
        e2 = [i for i in range(L) if i % N2 == 1 and i % N1 == 0][0]
    for k2 in range(N2):
        xs = []
        for a in range(N1):
            slot = (k2 * N1 + a) * 16
            xs.append((e.load(sc + '+%d' % slot), e.load(sc + '+%d' % (slot + 8))))
        Y = dftN1(e, xs)
        for k1 in range(N1):
            st((e1 * k1 + e2 * k2) % L, Y[k1])

def emit_ct64_scratch(e, ld, st):
    """64 = 8x8 CT with twiddled stage-1 results staged through scratch."""
    sc = 'ct64'
    e.scratch(sc, 2 * 64)
    for j2 in range(8):
        xs = [ld(8 * j1 + j2) for j1 in range(8)]
        F = dft8(e, xs)
        for k1 in range(8):
            t = (j2 * k1) % 64
            v = F[k1] if t == 0 else cmulw(e, F[k1], tw(t, 64))
            slot = (k1 * 8 + j2) * 16
            e.store(sc + '+%d' % slot, v[0])
            e.store(sc + '+%d' % (slot + 8), v[1])
    for k1 in range(8):
        xs = []
        for j2 in range(8):
            slot = (k1 * 8 + j2) * 16
            xs.append((e.load(sc + '+%d' % slot), e.load(sc + '+%d' % (slot + 8))))
        Y = dft8(e, xs)
        for k2 in range(8):
            st(k1 + 8 * k2, Y[k2])

def fft_size(E, L, ld, st):
    if L == 6:
        out = dft6(E, [ld(j) for j in range(6)])
        for k in range(6): st(k, out[k])
    elif L == 8:
        out = dft8(E, [ld(j) for j in range(8)])
        for k in range(8): st(k, out[k])
    elif L in (13, 17, 23):
        emit_prime_tiled(E, L, ld, st, T=4)
    elif L == 36:
        emit_pfa_scratch(E, 36, 4, 9, ld, st)
    elif L == 45:
        emit_pfa_scratch(E, 45, 9, 5, ld, st)
    elif L == 64:
        emit_ct64_scratch(E, ld, st)
    else:
        raise ValueError(L)

# map -------------------------------------------------------------------------
MAP_RCP_DIV = False   # per-call override via rcpdiv arg

def emit_map_batch(E, zs, newton=True, clamp=True, rcpdiv=(L >= 36)):
    """zs: list of (re, im) values (each 8 complex). Returns list of (re', im')."""
    n = len(zs)
    ss = []
    for (zr, zi) in zs:
        s = E.fmadd(zr, zr, E.mul(zi, zi))
        ss.append(E.maxv(s, E.k(1e-280)) if clamp else s)
    if not newton:
        ts = [E.emit('_mm512_sqrt_pd(%s)' % s) for s in ss]
        ws = [E.add(t, E.k(1.0)) for t in ts]
        one = E.k(1.0)
        ds = [E.emit('_mm512_div_pd(%s, %s)' % (one, w)) for w in ws]
    else:
        half = E.k(0.5); c15 = E.k(1.5); one = E.k(1.0); two = E.k(2.0)
        rs = [E.emit('_mm512_rsqrt14_pd(%s)' % s) for s in ss]
        hs = [E.mul(s, half) for s in ss]
        rrs = [E.mul(r, r) for r in rs]
        es = [E.fnmadd(hs[i], rrs[i], c15) for i in range(n)]
        rs = [E.mul(rs[i], es[i]) for i in range(n)]
        t0s = [E.mul(ss[i], rs[i]) for i in range(n)]
        rhs = [E.mul(rs[i], half) for i in range(n)]
        dds = [E.fnmadd(t0s[i], t0s[i], ss[i]) for i in range(n)]
        ts = [E.fmadd(rhs[i], dds[i], t0s[i]) for i in range(n)]
        ws = [E.add(ts[i], one) for i in range(n)]
        if rcpdiv == 'mix':
            ds = [None]*n
            nrk = [i for i in range(n) if i % 2 == 1]
            for i in range(n):
                if i % 2 == 0:
                    ds[i] = E.emit('_mm512_div_pd(%s, %s)' % (one, ws[i]))
            qs = {i: E.emit('_mm512_rcp14_pd(%s)' % ws[i]) for i in nrk}
            e1s = {i: E.fnmadd(ws[i], qs[i], two) for i in nrk}
            qs = {i: E.mul(qs[i], e1s[i]) for i in nrk}
            e2s = {i: E.fnmadd(ws[i], qs[i], two) for i in nrk}
            for i in nrk:
                ds[i] = E.mul(qs[i], e2s[i])
        elif rcpdiv:
            ds = [E.emit('_mm512_div_pd(%s, %s)' % (one, w)) for w in ws]
        else:
            qs = [E.emit('_mm512_rcp14_pd(%s)' % w) for w in ws]
            e1s = [E.fnmadd(ws[i], qs[i], two) for i in range(n)]
            qs = [E.mul(qs[i], e1s[i]) for i in range(n)]
            e2s = [E.fnmadd(ws[i], qs[i], two) for i in range(n)]
            ds = [E.mul(qs[i], e2s[i]) for i in range(n)]
    return [(E.mul(zs[i][0], ds[i]), E.mul(zs[i][1], ds[i])) for i in range(n)]

# transpose8 ------------------------------------------------------------------
def emit_transpose8(E, r):
    """8x8 double transpose of zmm list r[0..7]; returns new list."""
    t = [None]*8
    for i in range(4):
        t[2*i]   = E.emit('_mm512_unpacklo_pd(%s, %s)' % (r[2*i], r[2*i+1]))
        t[2*i+1] = E.emit('_mm512_unpackhi_pd(%s, %s)' % (r[2*i], r[2*i+1]))
    u = [None]*8
    # 128-bit granule shuffle within 256 lanes: use shuffle_f64x2
    # stage 2: combine pairs (t0,t2),(t1,t3),(t4,t6),(t5,t7) at 128-bit granule
    u[0] = E.emit('_mm512_shuffle_f64x2(%s, %s, 0x88)' % (t[0], t[2]))
    u[1] = E.emit('_mm512_shuffle_f64x2(%s, %s, 0x88)' % (t[1], t[3]))
    u[2] = E.emit('_mm512_shuffle_f64x2(%s, %s, 0xDD)' % (t[0], t[2]))
    u[3] = E.emit('_mm512_shuffle_f64x2(%s, %s, 0xDD)' % (t[1], t[3]))
    u[4] = E.emit('_mm512_shuffle_f64x2(%s, %s, 0x88)' % (t[4], t[6]))
    u[5] = E.emit('_mm512_shuffle_f64x2(%s, %s, 0x88)' % (t[5], t[7]))
    u[6] = E.emit('_mm512_shuffle_f64x2(%s, %s, 0xDD)' % (t[4], t[6]))
    u[7] = E.emit('_mm512_shuffle_f64x2(%s, %s, 0xDD)' % (t[5], t[7]))
    w = [None]*8
    w[0] = E.emit('_mm512_shuffle_f64x2(%s, %s, 0x88)' % (u[0], u[4]))
    w[1] = E.emit('_mm512_shuffle_f64x2(%s, %s, 0x88)' % (u[1], u[5]))
    w[2] = E.emit('_mm512_shuffle_f64x2(%s, %s, 0x88)' % (u[2], u[6]))
    w[3] = E.emit('_mm512_shuffle_f64x2(%s, %s, 0x88)' % (u[3], u[7]))
    w[4] = E.emit('_mm512_shuffle_f64x2(%s, %s, 0xDD)' % (u[0], u[4]))
    w[5] = E.emit('_mm512_shuffle_f64x2(%s, %s, 0xDD)' % (u[1], u[5]))
    w[6] = E.emit('_mm512_shuffle_f64x2(%s, %s, 0xDD)' % (u[2], u[6]))
    w[7] = E.emit('_mm512_shuffle_f64x2(%s, %s, 0xDD)' % (u[3], u[7]))
    return w

# codelets --------------------------------------------------------------------
def gen_y_codelet(L, name):
    """FFT along rows (element stride RP), vectors across 8 contiguous z."""
    g = GEOM[L]; RP, IM = g['RP'], g['IMOFF']
    e = E()
    e.nocache = (L in (36, 45, 64))
    def ld(j):
        return (e.load('p+%d' % (j*RP)), e.load('p+%d' % (j*RP+IM)))
    def st(k, v):
        e.store('p+%d' % (k*RP), v[0]); e.store('p+%d' % (k*RP+IM), v[1])
    fft_size(e, L, ld, st)
    return ('static inline __attribute__((always_inline)) void %s(double* restrict p){\n    %s\n}\n'
            % (name, e.code()))

def gen_x_codelet(L, name, fused, newton=True, prefetch=0):
    g = GEOM[L]; SP, IM = g['SP'], g['IMOFF']
    e = E()
    e.nocache = (L in (36, 45, 64))
    if prefetch:
        hint = 'T1'
        for j in range(L):
            e.raw('_mm_prefetch((const char*)(p+%d), _MM_HINT_%s);' % (j*SP + prefetch, hint))
            e.raw('_mm_prefetch((const char*)(p+%d), _MM_HINT_%s);' % (j*SP + IM + prefetch, hint))
        if fused:
            for j in range(L):
                e.raw('_mm_prefetch((const char*)(c+%d), _MM_HINT_T0);' % (j*SP))
                e.raw('_mm_prefetch((const char*)(c+%d), _MM_HINT_T0);' % (j*SP + IM))
    def ld(j):
        return (e.load('p+%d' % (j*SP)), e.load('p+%d' % (j*SP+IM)))
    if not fused:
        def st(k, v):
            e.store('p+%d' % (k*SP), v[0]); e.store('p+%d' % (k*SP+IM), v[1])
        fft_size(e, L, ld, st)
        return ('static inline __attribute__((always_inline)) void %s(double* restrict p){\n    %s\n}\n'
                % (name, e.code()))
    clamp = (GEOM[L]['RP'] != L)   # pads exist -> need clamp to avoid denormal assists
    if L >= 36:
        # two-phase: store z=FFT+c raw, then dense map loop on L1-hot data
        def st(k, v):
            cr = e.load('c+%d' % (k*SP)); ci = e.load('c+%d' % (k*SP+IM))
            e.store('p+%d' % (k*SP), e.add(v[0], cr))
            e.store('p+%d' % (k*SP+IM), e.add(v[1], ci))
        fft_size(e, L, ld, st)
        for k0 in range(0, L, 8):
            ks = list(range(k0, min(k0+8, L)))
            zs = [(e.load('p+%d' % (k*SP)), e.load('p+%d' % (k*SP+IM))) for k in ks]
            outs = emit_map_batch(e, zs, newton, clamp=clamp, rcpdiv='mix')
            for k, ov in zip(ks, outs):
                e.store('p+%d' % (k*SP), ov[0]); e.store('p+%d' % (k*SP+IM), ov[1])
    else:
        pend = []
        GROUP = 8
        def flush(n):
            if not pend: return
            part = pend[:n]
            del pend[:n]
            zs = [v for _, v in part]
            outs = emit_map_batch(e, zs, newton, clamp=clamp, rcpdiv=False)
            for (k, _), ov in zip(part, outs):
                e.store('p+%d' % (k*SP), ov[0]); e.store('p+%d' % (k*SP+IM), ov[1])
        def st(k, v):
            cr = e.load('c+%d' % (k*SP)); ci = e.load('c+%d' % (k*SP+IM))
            zv = (e.add(v[0], cr), e.add(v[1], ci))
            pend.append((k, zv))
            if len(pend) > GROUP:     # software pipeline: flush previous group lazily
                flush(GROUP)
        fft_size(e, L, ld, st)
        while pend:
            flush(GROUP)
    return ('static inline __attribute__((always_inline)) void %s(double* restrict p, const double* restrict c){\n    %s\n}\n'
            % (name, e.code()))

def gen_z_codelet(L, name):
    """FFT along z (contiguous within row) for 8 rows at base p.
    Stages transposed input through scratch to keep register pressure low."""
    if L == 64:
        return gen_z64_codelet(name)
    g = GEOM[L]; RP, IM = g['RP'], g['IMOFF']
    G = RP // 8
    e = E()
    e.nocache = (L in (36, 45, 64))
    e.scratch('zin', 2 * G * 8)    # transposed inputs, element-major
    e.scratch('zout', 2 * G * 8)   # outputs, element-major
    for plane, off in (('r', 0), ('i', IM)):
        for t in range(G):
            rows = [e.load('p+%d' % (y * RP + t * 8 + off)) for y in range(8)]
            w = emit_transpose8(e, rows)
            for i in range(8):
                if t * 8 + i < L:
                    e.store('zin+%d' % (((t * 8 + i) * 2 + (0 if plane == 'r' else 1)) * 8), w[i])
    def ld(j):
        return (e.load('zin+%d' % ((j * 2) * 8)), e.load('zin+%d' % ((j * 2 + 1) * 8)))
    def st(k, v):
        e.store('zout+%d' % ((k * 2) * 8), v[0])
        e.store('zout+%d' % ((k * 2 + 1) * 8), v[1])
    fft_size(e, L, ld, st)
    z = None
    for k in range(L, G * 8):
        if z is None: z = e.zero()
        e.store('zout+%d' % ((k * 2) * 8), z)
        e.store('zout+%d' % ((k * 2 + 1) * 8), z)
    for plane, off in (('r', 0), ('i', IM)):
        for t in range(G):
            regs = [e.load('zout+%d' % (((t * 8 + i) * 2 + (0 if plane == 'r' else 1)) * 8)) for i in range(8)]
            w = emit_transpose8(e, regs)
            for y in range(8):
                e.store('p+%d' % (y * RP + t * 8 + off), w[y])
    return ('static inline __attribute__((always_inline)) void %s(double* restrict p){\n    %s\n}\n'
            % (name, e.code()))

Z64TBL = None
def gen_z64_codelet(name):
    """One-transpose scheme for L=64: chunk regs = j1, lanes = j2.
    Processes ONE row (64 z) per call. Uses lane-twiddle tables TW64R/TW64I."""
    g = GEOM[64]; RP, IM = g['RP'], g['IMOFF']
    e = E()
    e.nocache = True
    xr = [e.load('p+%d' % (t * 8)) for t in range(8)]
    xi = [e.load('p+%d' % (t * 8 + IM)) for t in range(8)]
    x = [(xr[t], xi[t]) for t in range(8)]
    F = dft8(e, x)      # over registers j1; lanes j2
    # twiddle: F[k1] *= [w64^(j2*k1)]_j2  (vector constants from table)
    T = [None] * 8
    T[0] = F[0]
    for k1 in range(1, 8):
        tr = e.load('TW64R+%d' % (k1 * 8))
        ti = e.load('TW64I+%d' % (k1 * 8))
        re = e.fmsub(F[k1][0], tr, e.mul(F[k1][1], ti))
        im = e.fmadd(F[k1][0], ti, e.mul(F[k1][1], tr))
        T[k1] = (re, im)
    # transpose re and im: regs k1 x lanes j2 -> regs j2 x lanes k1
    wr = emit_transpose8(e, [T[k1][0] for k1 in range(8)])
    wi = emit_transpose8(e, [T[k1][1] for k1 in range(8)])
    Y = dft8(e, [(wr[j2], wi[j2]) for j2 in range(8)])   # over regs j2; lanes k1
    for k2 in range(8):
        e.store('p+%d' % (k2 * 8), Y[k2][0])
        e.store('p+%d' % (k2 * 8 + IM), Y[k2][1])
    return ('static inline __attribute__((always_inline)) void %s(double* restrict p){\n    %s\n}\n'
            % (name, e.code()))

def z64_tables():
    out = ['static const double TW64R[64] __attribute__((aligned(64))) = {']
    vals = []
    for k1 in range(8):
        for j2 in range(8):
            vals.append(flit(tw(j2 * k1, 64)[0]))
    out.append('  ' + ', '.join(vals))
    out.append('};')
    out.append('static const double TW64I[64] __attribute__((aligned(64))) = {')
    vals = []
    for k1 in range(8):
        for j2 in range(8):
            vals.append(flit(tw(j2 * k1, 64)[1]))
    out.append('  ' + ', '.join(vals))
    out.append('};')
    return '\n'.join(out) + '\n'

# ---------------- batched-8-volumes path (lanes = volumes) ----------------
def gen_b_codelet(L, axis, name, newton=True, NX=1):
    """Batched: lanes = 8 volumes. axis: 'z' (ES=8), 'y' (ES=L*8), 'x' (ES=SP2, fused map).
    NX: number of consecutive columns per call (x flavor)."""
    g2 = GEOM2[L]; SP2, IM2 = g2['SP2'], g2['IM2']
    ES = {'z': 8, 'y': L * 8, 'x': SP2}[axis]
    fused = (axis == 'x')
    e = E()
    if not fused:
        CS = {'z': L * 8, 'y': 8, 'x': 8}[axis]   # column stride
        for t in range(NX):
            def ld(j, t=t):
                return (e.load('p+%d' % (j * ES + t * CS)), e.load('p+%d' % (j * ES + t * CS + IM2)))
            def st(k, v, t=t):
                e.store('p+%d' % (k * ES + t * CS), v[0]); e.store('p+%d' % (k * ES + t * CS + IM2), v[1])
            fft_size(e, L, ld, st)
        return ('static inline __attribute__((always_inline)) void %s(double* restrict p){\n    %s\n}\n'
                % (name, e.code()))
    pend = []
    def flush(n):
        if not pend: return
        part = pend[:n]
        del pend[:n]
        zs = [v for _, v in part]
        outs = emit_map_batch(e, zs, newton, clamp=False, rcpdiv=False)
        for (off, _), ov in zip(part, outs):
            e.store('p+%d' % off, ov[0]); e.store('p+%d' % (off + IM2), ov[1])
    for t in range(NX):
        def ld(j, t=t):
            return (e.load('p+%d' % (j * ES + t * 8)), e.load('p+%d' % (j * ES + t * 8 + IM2)))
        def st(k, v, t=t):
            off = k * ES + t * 8
            cr = e.load('c+%d' % off); ci = e.load('c+%d' % (off + IM2))
            zv = (e.add(v[0], cr), e.add(v[1], ci))
            pend.append((off, zv))
            if len(pend) > 8:
                flush(8)
        fft_size(e, L, ld, st)
    while pend:
        flush(8)
    return ('static inline __attribute__((always_inline)) void %s(double* restrict p, const double* restrict c){\n    %s\n}\n'
            % (name, e.code()))

def gen_b_convert(L):
    """tobuf8/frombuf8: 8 volumes (interleaved complex, stride vol2 doubles) <-> lane-major."""
    g2 = GEOM2[L]; SP2, IM2 = g2['SP2'], g2['IM2']
    L2 = L * L; L3 = L2 * L
    vol2 = 2 * L3
    out = []
    # tobuf8v: nv = number of real volumes (1..8); missing lanes zeroed
    out.append('static void tobuf8v_%d(double* restrict dst, const double* restrict src, int nv){' % L)
    out.append('  const __m512i ev = _mm512_loadu_si512(idx_even_);')
    out.append('  const __m512i od = _mm512_loadu_si512(idx_odd_);')
    out.append('  __m512d zero = _mm512_setzero_pd();')
    out.append('  for (int s = 0; s < %d; s++){' % L)
    out.append('    for (int blk = 0; blk < %d; blk++){' % ((L2 + 7) // 8))
    out.append('      long pbase = (long)s * %d + (long)blk * 8;     // position of first complex' % L2)
    out.append('      int npos = %d - blk * 8; if (npos > 8) npos = 8;' % L2)
    out.append('      __m512d re[8], im[8];')
    out.append('      for (int v = 0; v < 8; v++){')
    out.append('        if (v < nv){')
    out.append('          const double* r = src + (long)v * %d + pbase * 2;' % vol2)
    out.append('          __mmask8 m0 = (npos >= 4) ? 0xFF : (__mmask8)((1u << (npos * 2)) - 1);')
    out.append('          __mmask8 m1 = (npos <= 4) ? 0 : (__mmask8)((1u << ((npos - 4) * 2)) - 1);')
    out.append('          __m512d a = _mm512_maskz_loadu_pd(m0, r);')
    out.append('          __m512d b = _mm512_maskz_loadu_pd(m1, r + 8);')
    out.append('          re[v] = _mm512_permutex2var_pd(a, ev, b);')
    out.append('          im[v] = _mm512_permutex2var_pd(a, od, b);')
    out.append('        } else { re[v] = zero; im[v] = zero; }')
    out.append('      }')
    out.append('      double* dre = dst + (long)s * %d + (long)blk * 64;' % SP2)
    out.append('      TR8(re); TR8(im);')
    out.append('      for (int i = 0; i < npos; i++){')
    out.append('        _mm512_storeu_pd(dre + i * 8, re[i]);')
    out.append('        _mm512_storeu_pd(dre + i * 8 + %d, im[i]);' % IM2)
    out.append('      }')
    out.append('    }')
    out.append('  }')
    out.append('}')
    out.append('static void frombuf8v_%d(double* restrict dst, const double* restrict src, int nv){' % L)
    out.append('  const __m512i lo = _mm512_loadu_si512(idx_ilo_);')
    out.append('  const __m512i hi = _mm512_loadu_si512(idx_ihi_);')
    out.append('  for (int s = 0; s < %d; s++){' % L)
    out.append('    for (int blk = 0; blk < %d; blk++){' % ((L2 + 7) // 8))
    out.append('      long pbase = (long)s * %d + (long)blk * 8;' % L2)
    out.append('      int npos = %d - blk * 8; if (npos > 8) npos = 8;' % L2)
    out.append('      const double* dre = src + (long)s * %d + (long)blk * 64;' % SP2)
    out.append('      __m512d re[8], im[8];')
    out.append('      for (int i = 0; i < 8; i++){')
    out.append('        re[i] = _mm512_loadu_pd(dre + i * 8);')
    out.append('        im[i] = _mm512_loadu_pd(dre + i * 8 + %d);' % IM2)
    out.append('      }')
    out.append('      TR8(re); TR8(im);   // now indexed by volume, lanes = positions')
    out.append('      for (int v = 0; v < nv; v++){')
    out.append('        double* r = dst + (long)v * %d + pbase * 2;' % vol2)
    out.append('        __mmask8 m0 = (npos >= 4) ? 0xFF : (__mmask8)((1u << (npos * 2)) - 1);')
    out.append('        __mmask8 m1 = (npos <= 4) ? 0 : (__mmask8)((1u << ((npos - 4) * 2)) - 1);')
    out.append('        _mm512_mask_storeu_pd(r, m0, _mm512_permutex2var_pd(re[v], lo, im[v]));')
    out.append('        _mm512_mask_storeu_pd(r + 8, m1, _mm512_permutex2var_pd(re[v], hi, im[v]));')
    out.append('      }')
    out.append('    }')
    out.append('  }')
    out.append('}')
    return '\n'.join(out) + '\n'

B_NX = {6: 4, 8: 2, 13: 1}

def gen_b_size(L, newton=True):
    g2 = GEOM2[L]; SP2, IM2, BUF2 = g2['SP2'], g2['IM2'], g2['BUF2']
    L2 = L * L
    NX = B_NX.get(L, 1)
    NZY = 1
    assert L2 % NX == 0
    out = []
    out.append(gen_b_codelet(L, 'z', 'b%dz' % L, NX=NZY))
    out.append(gen_b_codelet(L, 'y', 'b%dy' % L, NX=NZY))
    if NZY > 1:
        out.append(gen_b_codelet(L, 'z', 'b%dz1' % L, NX=1))
        out.append(gen_b_codelet(L, 'y', 'b%dy1' % L, NX=1))
    out.append(gen_b_codelet(L, 'x', 'b%dx' % L, newton=newton, NX=NX))
    out.append('static double* bx%d; static double* bc%d;\n'
               'static void binit%d(void){ if (!bx%d){ bx%d = hp_alloc(%d); bc%d = hp_alloc(%d); } }\n'
               % (L, L, L, L, L, 8 * BUF2, L, 8 * BUF2))
    out.append(gen_b_convert(L))
    UN = '_Pragma("GCC unroll 1") '
    st = []
    st.append('static void bstep%d(double* restrict x, const double* restrict c){' % L)
    st.append('  _Pragma("GCC unroll 1") for (int s = 0; s < %d; s++){' % L)
    st.append('    double* p = x + (long)s * %d;' % SP2)
    nfz = (L // NZY) * NZY
    st.append('    %sfor (int xy = 0; xy < %d; xy += %d) b%dz(p + (long)xy * %d);' % (UN, nfz, NZY, L, 8 * L))
    if nfz < L:
        st.append('    b%dz1(p + %d);' % (L, nfz * 8 * L))
    st.append('    %sfor (int z = 0; z < %d; z += %d) b%dy(p + (long)z * 8);' % (UN, nfz, NZY, L))
    if nfz < L:
        st.append('    b%dy1(p + %d);' % (L, nfz * 8))
    st.append('  }')
    st.append('  %sfor (int q = 0; q < %d; q += %d) b%dx(x + (long)q * 8, c + (long)q * 8);' % (UN, L2, NX, L))
    st.append('}')
    out.append('\n'.join(st) + '\n')
    return '\n'.join(out)

# header / drivers ------------------------------------------------------------
HEADER = r'''// ============================================================================
// Specialized batched 3D complex FFT + nonlinear map iteration
//   z = FFT3(x) + c ;  x <- z / (1 + |z|)
// for cube sizes L in {6, 8, 13, 17, 23, 36, 45, 64}.
//
// All transform arithmetic in this file is original, generated for this task
// by the accompanying code generators (codegen.py / codegen_split.py /
// genmerge.py in the same directory). No FFT library code is used or linked.
// Algorithms: prime-factor (Good-Thomas) decompositions (36 = 4x9, 45 = 9x5,
// 6 = 2x3), Cooley-Tukey 8x8 for 64, symmetric half-length real-coefficient
// prime DFTs for 13/17/23, small codelets for 2/3/4/5/8/9. Twiddle constants
// are compile-time literals computed in extended precision.
// AVX-512 required; single-threaded by construction (no threads, no OpenMP).
// ============================================================================
#include <immintrin.h>
#include <string.h>
#include <stdint.h>
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

// deinterleave 8 complex (2 zmm) -> (re zmm, im zmm)
static const long long idx_even_[8] = {0,2,4,6,8,10,12,14};
static const long long idx_odd_[8]  = {1,3,5,7,9,11,13,15};
static const long long idx_ilo_[8]  = {0,8,1,9,2,10,3,11};
static const long long idx_ihi_[8]  = {4,12,5,13,6,14,7,15};

static inline __attribute__((always_inline)) void tr8(__m512d* r){
    __m512d t0 = _mm512_unpacklo_pd(r[0], r[1]);
    __m512d t1 = _mm512_unpackhi_pd(r[0], r[1]);
    __m512d t2 = _mm512_unpacklo_pd(r[2], r[3]);
    __m512d t3 = _mm512_unpackhi_pd(r[2], r[3]);
    __m512d t4 = _mm512_unpacklo_pd(r[4], r[5]);
    __m512d t5 = _mm512_unpackhi_pd(r[4], r[5]);
    __m512d t6 = _mm512_unpacklo_pd(r[6], r[7]);
    __m512d t7 = _mm512_unpackhi_pd(r[6], r[7]);
    __m512d u0 = _mm512_shuffle_f64x2(t0, t2, 0x88);
    __m512d u1 = _mm512_shuffle_f64x2(t1, t3, 0x88);
    __m512d u2 = _mm512_shuffle_f64x2(t0, t2, 0xDD);
    __m512d u3 = _mm512_shuffle_f64x2(t1, t3, 0xDD);
    __m512d u4 = _mm512_shuffle_f64x2(t4, t6, 0x88);
    __m512d u5 = _mm512_shuffle_f64x2(t5, t7, 0x88);
    __m512d u6 = _mm512_shuffle_f64x2(t4, t6, 0xDD);
    __m512d u7 = _mm512_shuffle_f64x2(t5, t7, 0xDD);
    r[0] = _mm512_shuffle_f64x2(u0, u4, 0x88);
    r[1] = _mm512_shuffle_f64x2(u1, u5, 0x88);
    r[2] = _mm512_shuffle_f64x2(u2, u6, 0x88);
    r[3] = _mm512_shuffle_f64x2(u3, u7, 0x88);
    r[4] = _mm512_shuffle_f64x2(u0, u4, 0xDD);
    r[5] = _mm512_shuffle_f64x2(u1, u5, 0xDD);
    r[6] = _mm512_shuffle_f64x2(u2, u6, 0xDD);
    r[7] = _mm512_shuffle_f64x2(u3, u7, 0xDD);
}
#define TR8(r) tr8(r)
'''

def gen_size(L, newton=True):
    g = GEOM[L]; RP, RG, SP, IM, BUFD = g['RP'], g['RG'], g['SP'], g['IMOFF'], g['BUFD']
    L2 = L*L
    out = []
    if L == 64:
        out.append(z64_tables())
    out.append(gen_z_codelet(L, 'f%dz' % L))
    out.append(gen_y_codelet(L, 'f%dy' % L))
    pfd = 128 if L >= 36 else 0
    out.append(gen_x_codelet(L, 'f%dx' % L, True, newton=newton, prefetch=pfd))
    out.append(gen_x_codelet(L, 'f%dxp' % L, False))

    # buffers + init
    out.append('static double* xb%d; static double* cb%d;\n'
               'void init%d(void){ if (!xb%d){ xb%d = hp_alloc(%d); cb%d = hp_alloc(%d); } }\n'
               % (L, L, L, L, L, 8*BUFD, L, 8*BUFD))

    # converters: numpy volume (interleaved complex, packed) <-> split padded planes
    nfull = L // 8          # full 8-complex chunks per row
    rem = L - nfull*8
    conv = []
    conv.append('static void tobuf%d(double* restrict dst, const double* restrict src){' % L)
    conv.append('  const __m512i ev = _mm512_loadu_si512(idx_even_);')
    conv.append('  const __m512i od = _mm512_loadu_si512(idx_odd_);')
    conv.append('  for (int s = 0; s < %d; s++){' % L)
    conv.append('    for (int y = 0; y < %d; y++){' % L)
    conv.append('      const double* r = src + ((long)s*%d + (long)y*%d)*2;' % (L2, L))
    conv.append('      double* d = dst + (long)s*%d + (long)y*%d;' % (SP, RP))
    for ch in range(nfull):
        conv.append('      {')
        conv.append('        __m512d a = _mm512_loadu_pd(r+%d), b = _mm512_loadu_pd(r+%d);' % (16*ch, 16*ch+8))
        conv.append('        _mm512_storeu_pd(d+%d, _mm512_permutex2var_pd(a, ev, b));' % (8*ch))
        conv.append('        _mm512_storeu_pd(d+%d, _mm512_permutex2var_pd(a, od, b));' % (8*ch + IM))
        conv.append('      }')
    if rem:
        mask = (1 << rem) - 1
        conv.append('      {')
        conv.append('        __m512d a = _mm512_maskz_loadu_pd(0x%x, r+%d);' % ((1<<min(rem*2,8))-1 if rem*2<=8 else 0xFF, 16*nfull))
        if rem*2 > 8:
            conv.append('        __m512d b = _mm512_maskz_loadu_pd(0x%x, r+%d);' % ((1<<(rem*2-8))-1, 16*nfull+8))
        else:
            conv.append('        __m512d b = _mm512_setzero_pd();')
        conv.append('        _mm512_mask_storeu_pd(d+%d, 0x%x, _mm512_permutex2var_pd(a, ev, b));' % (8*nfull, mask))
        conv.append('        _mm512_mask_storeu_pd(d+%d, 0x%x, _mm512_permutex2var_pd(a, od, b));' % (8*nfull + IM, mask))
        conv.append('      }')
    conv.append('    }')
    conv.append('  }')
    conv.append('}')
    out.append('\n'.join(conv) + '\n')

    conv = []
    conv.append('static void frombuf%d(double* restrict dst, const double* restrict src){' % L)
    conv.append('  const __m512i lo = _mm512_loadu_si512(idx_ilo_);')
    conv.append('  const __m512i hi = _mm512_loadu_si512(idx_ihi_);')
    conv.append('  for (int s = 0; s < %d; s++){' % L)
    conv.append('    for (int y = 0; y < %d; y++){' % L)
    conv.append('      double* r = dst + ((long)s*%d + (long)y*%d)*2;' % (L2, L))
    conv.append('      const double* d = src + (long)s*%d + (long)y*%d;' % (SP, RP))
    for ch in range(nfull):
        conv.append('      {')
        conv.append('        __m512d re = _mm512_loadu_pd(d+%d), im = _mm512_loadu_pd(d+%d);' % (8*ch, 8*ch + IM))
        conv.append('        _mm512_storeu_pd(r+%d, _mm512_permutex2var_pd(re, lo, im));' % (16*ch))
        conv.append('        _mm512_storeu_pd(r+%d, _mm512_permutex2var_pd(re, hi, im));' % (16*ch+8))
        conv.append('      }')
    if rem:
        conv.append('      {')
        conv.append('        __m512d re = _mm512_loadu_pd(d+%d), im = _mm512_loadu_pd(d+%d);' % (8*nfull, 8*nfull + IM))
        m1 = (1 << min(rem*2, 8)) - 1
        conv.append('        _mm512_mask_storeu_pd(r+%d, 0x%x, _mm512_permutex2var_pd(re, lo, im));' % (16*nfull, m1))
        if rem*2 > 8:
            m2 = (1 << (rem*2-8)) - 1
            conv.append('        _mm512_mask_storeu_pd(r+%d, 0x%x, _mm512_permutex2var_pd(re, hi, im));' % (16*nfull+8, m2))
        conv.append('      }')
    conv.append('    }')
    conv.append('  }')
    conv.append('}')
    out.append('\n'.join(conv) + '\n')

    # step
    step = []
    step.append('static void step%d(double* restrict x, const double* restrict c){' % L)
    step.append('  for (int s = 0; s < %d; s++){' % L)
    step.append('    double* p = x + (long)s * %d;' % SP)
    if L == 64:
        step.append('    _Pragma("GCC unroll 1") for (int y = 0; y < 64; y++) f64z(p + y * %d);' % RP)
    else:
        step.append('    _Pragma("GCC unroll 1") for (int g = 0; g < %d; g++) f%dz(p + g * %d);' % (RG//8, L, 8*RP))
    step.append('    _Pragma("GCC unroll 1") for (int zb = 0; zb < %d; zb++) f%dy(p + zb * 8);' % (RP//8, L))
    step.append('  }')
    step.append('  _Pragma("GCC unroll 1") for (long q = 0; q < %d; q += 8) f%dx(x + q, c + q);' % (L*RP, L))
    step.append('}')
    out.append('\n'.join(step) + '\n')

    # debug fft3d
    dbg = []
    dbg.append('void fft3d_%d(double* restrict x){' % L)
    dbg.append('  init%d();' % L)
    dbg.append('  double* restrict xb = xb%d;' % L)
    dbg.append('  tobuf%d(xb, x);' % L)
    dbg.append('  for (int s = 0; s < %d; s++){' % L)
    dbg.append('    double* p = xb + (long)s * %d;' % SP)
    if L == 64:
        dbg.append('    _Pragma("GCC unroll 1") for (int y = 0; y < 64; y++) f64z(p + y * %d);' % RP)
    else:
        dbg.append('    _Pragma("GCC unroll 1") for (int g = 0; g < %d; g++) f%dz(p + g * %d);' % (RG//8, L, 8*RP))
    dbg.append('    _Pragma("GCC unroll 1") for (int zb = 0; zb < %d; zb++) f%dy(p + zb * 8);' % (RP//8, L))
    dbg.append('  }')
    dbg.append('  _Pragma("GCC unroll 1") for (long q = 0; q < %d; q += 8) f%dxp(xb + q);' % (L*RP, L))
    dbg.append('  frombuf%d(x, xb);' % L)
    dbg.append('}')
    out.append('\n'.join(dbg) + '\n')

    # batched path for small sizes
    batched = L in GEOM2
    if batched:
        out.append(gen_b_size(L, newton=newton))
    # run
    run = []
    run.append('void run%d(double* restrict x, const double* restrict c, double* restrict o1, long B, long m){' % L)
    run.append('  const long vol2 = %d;' % (2*L*L2))
    run.append('  init%d();' % L)
    run.append('  double* restrict xb = xb%d; double* restrict cb = cb%d;' % (L, L))
    run.append('  long v0 = 0;')
    if batched:
        run.append('  binit%d();' % L)
        run.append('  for (long g = 0; g + 8 <= B || (B - g >= 5); g += 8){')
        run.append('    int nv = (B - g >= 8) ? 8 : (int)(B - g);')
        run.append('    tobuf8v_%d(bx%d, x + g*vol2, nv);' % (L, L))
        run.append('    tobuf8v_%d(bc%d, c + g*vol2, nv);' % (L, L))
        run.append('    for (long t = 0; t < m; t++){')
        run.append('      bstep%d(bx%d, bc%d);' % (L, L, L))
        run.append('      if (t == 0) frombuf8v_%d(o1 + g*vol2, bx%d, nv);' % (L, L))
        run.append('    }')
        run.append('    frombuf8v_%d(x + g*vol2, bx%d, nv);' % (L, L))
        run.append('    v0 = g + nv;')
        run.append('  }')
    run.append('  for (long v = v0; v < B; v++){')
    run.append('    tobuf%d(xb, x + v*vol2);' % L)
    run.append('    tobuf%d(cb, c + v*vol2);' % L)
    run.append('    for (long t = 0; t < m; t++){')
    run.append('      step%d(xb, cb);' % L)
    run.append('      if (t == 0) frombuf%d(o1 + v*vol2, xb);' % L)
    run.append('    }')
    run.append('    frombuf%d(x + v*vol2, xb);' % L)
    run.append('  }')
    run.append('}')
    out.append('\n'.join(run) + '\n')
    return '\n'.join(out)

def main(newton=True, path='implementation.c', sizes=SIZES):
    parts = [HEADER]
    for L in sizes:
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
