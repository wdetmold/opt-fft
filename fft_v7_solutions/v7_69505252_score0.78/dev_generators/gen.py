#!/usr/bin/env python3
# Generates implementation.c: specialized AVX-512 kernels for iterated 3D FFT sizes.
import numpy as np

LD = np.longdouble
TWO_PI = LD('6.283185307179586476925286766559005768394')

def hexd(v):
    d = float(v)
    if d == int(d) and abs(d) < 1e15:
        return f"{d:.1f}"
    return d.hex()

def tw(k, N):
    """exp(-2*pi*i*k/N) as (cos, -sin) doubles, careful reduction."""
    k = k % N
    ang = -TWO_PI * LD(k) / LD(N)
    return (float(np.cos(ang)), float(np.sin(ang)))

class E:
    def __init__(self):
        self.lines = []
        self.n = 0
    def t(self, pfx="t"):
        self.n += 1
        return f"{pfx}{self.n}"
    def l(self, s):
        self.lines.append("    " + s)
    def add(self, a, b):
        c = self.t()
        self.l(f"__m512d {c} = _mm512_add_pd({a}, {b});")
        return c
    def sub(self, a, b):
        c = self.t()
        self.l(f"__m512d {c} = _mm512_sub_pd({a}, {b});")
        return c
    def mul(self, a, b):
        c = self.t()
        self.l(f"__m512d {c} = _mm512_mul_pd({a}, {b});")
        return c
    def fma(self, a, b, c):   # a*b + c
        d = self.t()
        self.l(f"__m512d {d} = _mm512_fmadd_pd({a}, {b}, {c});")
        return d
    def fnma(self, a, b, c):  # c - a*b
        d = self.t()
        self.l(f"__m512d {d} = _mm512_fnmadd_pd({a}, {b}, {c});")
        return d
    def k(self, v):           # constant broadcast (inline so gcc can embed)
        return f"_mm512_set1_pd({hexd(v)})"
    def mulk(self, a, v):
        return self.mul(a, self.k(v))
    def fmak(self, a, v, c):
        return self.fma(a, self.k(v), c)
    def fnmak(self, a, v, c):
        return self.fnma(a, self.k(v), c)

# ---------------- complex helpers (cv = (re,im) names) ----------------
def cadd(e, a, b): return (e.add(a[0], b[0]), e.add(a[1], b[1]))
def csub(e, a, b): return (e.sub(a[0], b[0]), e.sub(a[1], b[1]))
def cmul_tw(e, x, t):
    """x * (cr + i*ci) with constants t=(cr,ci)."""
    cr, ci = t
    if cr == 1.0 and ci == 0.0: return x
    yr = e.fnmak(x[1], ci, e.mulk(x[0], cr))   # xr*cr - xi*ci
    yi = e.fmak(x[1], cr, e.mulk(x[0], ci))    # xr*ci + xi*cr
    return (yr, yi)
def c_a_m_ib(e, a, b):   # a - i*b  => (ar + bi, ai - br)
    return (e.add(a[0], b[1]), e.sub(a[1], b[0]))
def c_a_p_ib(e, a, b):   # a + i*b  => (ar - bi, ai + br)
    return (e.sub(a[0], b[1]), e.add(a[1], b[0]))

# ---------------- DFT codelets ----------------
def dft2(e, x0, x1):
    return [cadd(e, x0, x1), csub(e, x0, x1)]

def dft3(e, x0, x1, x2):
    S3 = float(np.sin(TWO_PI/LD(3)))   # sin(2pi/3) = sqrt(3)/2
    tr, ti = e.add(x1[0], x2[0]), e.add(x1[1], x2[1])
    ur, ui = e.sub(x1[0], x2[0]), e.sub(x1[1], x2[1])
    y0 = (e.add(x0[0], tr), e.add(x0[1], ti))
    mr = e.fnmak(tr, 0.5, x0[0])
    mi = e.fnmak(ti, 0.5, x0[1])
    vr = e.mulk(ur, S3)
    vi = e.mulk(ui, S3)
    y1 = (e.add(mr, vi), e.sub(mi, vr))   # m - i v
    y2 = (e.sub(mr, vi), e.add(mi, vr))   # m + i v
    return [y0, y1, y2]

def dft4(e, x0, x1, x2, x3):
    t0 = cadd(e, x0, x2); t1 = csub(e, x0, x2)
    t2 = cadd(e, x1, x3); t3 = csub(e, x1, x3)
    y0 = cadd(e, t0, t2); y2 = csub(e, t0, t2)
    y1 = c_a_m_ib(e, t1, t3)
    y3 = c_a_p_ib(e, t1, t3)
    return [y0, y1, y2, y3]

def dft5(e, x0, x1, x2, x3, x4):
    c1, s1n = tw(1, 5); c2, s2n = tw(2, 5)
    s1, s2 = -s1n, -s2n   # sin(2pi/5), sin(4pi/5) positive
    sa = cadd(e, x1, x4); da = csub(e, x1, x4)
    sb = cadd(e, x2, x3); db = csub(e, x2, x3)
    y0 = (e.add(e.add(x0[0], sa[0]), sb[0]), e.add(e.add(x0[1], sa[1]), sb[1]))
    A1 = (e.fmak(sb[0], c2, e.fmak(sa[0], c1, x0[0])),
          e.fmak(sb[1], c2, e.fmak(sa[1], c1, x0[1])))
    B1 = (e.fmak(db[0], s2, e.mulk(da[0], s1)),
          e.fmak(db[1], s2, e.mulk(da[1], s1)))
    A2 = (e.fmak(sb[0], c1, e.fmak(sa[0], c2, x0[0])),
          e.fmak(sb[1], c1, e.fmak(sa[1], c2, x0[1])))
    B2 = (e.fnmak(db[0], s1, e.mulk(da[0], s2)),
          e.fnmak(db[1], s1, e.mulk(da[1], s2)))
    y1 = c_a_m_ib(e, A1, B1); y4 = c_a_p_ib(e, A1, B1)
    y2 = c_a_m_ib(e, A2, B2); y3 = c_a_p_ib(e, A2, B2)
    return [y0, y1, y2, y3, y4]

def dft8(e, x):
    C = float(np.cos(TWO_PI/LD(8)))   # sqrt(2)/2
    e0 = cadd(e, x[0], x[4]); e1 = csub(e, x[0], x[4])
    e2 = cadd(e, x[2], x[6]); e3 = csub(e, x[2], x[6])
    e4 = cadd(e, x[1], x[5]); e5 = csub(e, x[1], x[5])
    e6 = cadd(e, x[3], x[7]); e7 = csub(e, x[3], x[7])
    E0 = cadd(e, e0, e2); E2 = csub(e, e0, e2)
    E1 = c_a_m_ib(e, e1, e3); E3 = c_a_p_ib(e, e1, e3)
    O0 = cadd(e, e4, e6); O2 = csub(e, e4, e6)
    O1 = c_a_m_ib(e, e5, e7); O3 = c_a_p_ib(e, e5, e7)
    y0 = cadd(e, E0, O0); y4 = csub(e, E0, O0)
    y2 = c_a_m_ib(e, E2, O2); y6 = c_a_p_ib(e, E2, O2)
    # k=1: W8^1*O1 = C*( (o_r+o_i) + i(o_i-o_r) )
    t1 = e.add(O1[0], O1[1]); u1 = e.sub(O1[1], O1[0])
    y1 = (e.fmak(t1, C, E1[0]), e.fmak(u1, C, E1[1]))
    y5 = (e.fnmak(t1, C, E1[0]), e.fnmak(u1, C, E1[1]))
    # k=3: W8^3*O3 = -C*( (o_r-o_i) + i(o_r+o_i) )
    t3 = e.sub(O3[0], O3[1]); u3 = e.add(O3[0], O3[1])
    y3 = (e.fnmak(t3, C, E3[0]), e.fnmak(u3, C, E3[1]))
    y7 = (e.fmak(t3, C, E3[0]), e.fmak(u3, C, E3[1]))
    return [y0, y1, y2, y3, y4, y5, y6, y7]

def dft9(e, x):
    """9-point DFT via 3x3 CT, natural order in and out."""
    U = []
    for b in range(3):
        U.append(dft3(e, x[b], x[b+3], x[b+6]))
    V = [[None]*3 for _ in range(3)]
    for b in range(3):
        for k1 in range(3):
            V[b][k1] = cmul_tw(e, U[b][k1], tw(b*k1, 9))
    y = [None]*9
    for k1 in range(3):
        r = dft3(e, V[0][k1], V[1][k1], V[2][k1])
        for k2 in range(3):
            y[k1 + 3*k2] = r[k2]
    return y

# ---------------- kernel body emitters ----------------
# LDf(j) -> cv ; STf(j, cv) -> emits store of output element j

def body_small(e, L, LDf, STf):
    if L == 8:
        x = [LDf(j) for j in range(8)]
        y = dft8(e, x)
        for j in range(8): STf(j, y[j])
    elif L == 6:
        # PFA(2,3): rows n1=0: x[0,2,4]; n1=1: x[3,5,1]; out k=(3k1+4k2)%6
        x = [LDf(j) for j in range(6)]
        U0 = dft3(e, x[0], x[2], x[4])
        U1 = dft3(e, x[3], x[5], x[1])
        outmap0 = [0, 4, 2]; outmap1 = [3, 1, 5]
        for k2 in range(3):
            a = cadd(e, U0[k2], U1[k2]); b = csub(e, U0[k2], U1[k2])
            STf(outmap0[k2], a); STf(outmap1[k2], b)
    else:
        raise ValueError

def body_pfa(e, L, LDf, STf):
    """PFA kernels for 36=4x9 and 45=9x5 with stack buffer between stages."""
    if L == 36:
        N1, N2, d2, = 4, 9, dft9
        inoff = lambda n1, n2: (9*n1 + 4*n2) % 36
        outoff = lambda k1, k2: (9*k1 + 28*k2) % 36
        d1 = dft4
    else:
        N1, N2 = 9, 5
        inoff = lambda n1, n2: (5*n1 + 9*n2) % 45
        outoff = lambda k1, k2: (10*k1 + 36*k2) % 45
        d2 = dft5; d1 = dft9
    bufr = e.t("bufr"); bufi = e.t("bufi")
    e.l(f"double {bufr}[{L}*8] __attribute__((aligned(64)));")
    e.l(f"double {bufi}[{L}*8] __attribute__((aligned(64)));")
    # stage 1: DFT-N2 over n2 for each n1; store at buf[k2*N1 + n1]
    for n1 in range(N1):
        xs = [LDf(inoff(n1, n2)) for n2 in range(N2)]
        if N2 == 9: ys = d2(e, xs)
        else: ys = d2(e, *xs)
        for k2 in range(N2):
            idx = k2*N1 + n1
            e.l(f"_mm512_store_pd({bufr} + {idx}*8, {ys[k2][0]});")
            e.l(f"_mm512_store_pd({bufi} + {idx}*8, {ys[k2][1]});")
    # stage 2: DFT-N1 over n1 for each k2
    for k2 in range(N2):
        xs = []
        for n1 in range(N1):
            idx = k2*N1 + n1
            r = e.t(); e.l(f"__m512d {r} = _mm512_load_pd({bufr} + {idx}*8);")
            i = e.t(); e.l(f"__m512d {i} = _mm512_load_pd({bufi} + {idx}*8);")
            xs.append((r, i))
        if N1 == 9: ys = d1(e, xs)
        else: ys = d1(e, *xs)
        for k1 in range(N1):
            STf(outoff(k1, k2), ys[k1])

def body_64(e, LDf, STf):
    """64 = 8x8 Cooley-Tukey with twiddles; buffer between stages."""
    bufr = e.t("bufr"); bufi = e.t("bufi")
    e.l(f"double {bufr}[64*8] __attribute__((aligned(64)));")
    e.l(f"double {bufi}[64*8] __attribute__((aligned(64)));")
    for b in range(8):
        xs = [LDf(8*a + b) for a in range(8)]
        ys = dft8(e, xs)
        for k1 in range(8):
            v = cmul_tw(e, ys[k1], tw(b*k1, 64))
            idx = k1*8 + b
            e.l(f"_mm512_store_pd({bufr} + {idx}*8, {v[0]});")
            e.l(f"_mm512_store_pd({bufi} + {idx}*8, {v[1]});")
    for k1 in range(8):
        xs = []
        for b in range(8):
            idx = k1*8 + b
            r = e.t(); e.l(f"__m512d {r} = _mm512_load_pd({bufr} + {idx}*8);")
            i = e.t(); e.l(f"__m512d {i} = _mm512_load_pd({bufi} + {idx}*8);")
            xs.append((r, i))
        ys = dft8(e, xs)
        for k2 in range(8):
            STf(k1 + 8*k2, ys[k2])

def body_prime_classic(e, p, LDf, STf):
    h = (p - 1)//2
    pfx = e.t("pk")
    names = {nm: f"{pfx}{nm}" for nm in ("sr","si","dr","di","aar","aai")}
    for nm in names.values():
        e.l(f"double {nm}[{h+1}*8] __attribute__((aligned(64)));")
    x0 = LDf(0)
    for m in range(1, h+1):
        a = LDf(m); b = LDf(p - m)
        s = cadd(e, a, b); d = csub(e, a, b)
        e.l(f"_mm512_store_pd({names['sr']} + {m}*8, {s[0]});")
        e.l(f"_mm512_store_pd({names["si"]} + {m}*8, {s[1]});")
        e.l(f"_mm512_store_pd({names["dr"]} + {m}*8, {d[0]});")
        e.l(f"_mm512_store_pd({names["di"]} + {m}*8, {d[1]});")
    accr = [None]*(h+1); acci = [None]*(h+1)
    for k in range(h+1):
        accr[k] = e.t("ar"); e.l(f"__m512d {accr[k]} = {x0[0]};")
        acci[k] = e.t("ai"); e.l(f"__m512d {acci[k]} = {x0[1]};")
    for m in range(1, h+1):
        smr = e.t(); e.l(f"__m512d {smr} = _mm512_load_pd({names["sr"]} + {m}*8);")
        smi = e.t(); e.l(f"__m512d {smi} = _mm512_load_pd({names["si"]} + {m}*8);")
        e.l(f"{accr[0]} = _mm512_add_pd({accr[0]}, {smr});")
        e.l(f"{acci[0]} = _mm512_add_pd({acci[0]}, {smi});")
        for k in range(1, h+1):
            cc = float(np.cos(TWO_PI * LD((k*m) % p) / LD(p)))
            e.l(f"{accr[k]} = _mm512_fmadd_pd({smr}, _mm512_set1_pd({hexd(cc)}), {accr[k]});")
            e.l(f"{acci[k]} = _mm512_fmadd_pd({smi}, _mm512_set1_pd({hexd(cc)}), {acci[k]});")
    STf(0, (accr[0], acci[0]))
    for k in range(1, h+1):
        e.l(f"_mm512_store_pd({names["aar"]} + {k}*8, {accr[k]});")
        e.l(f"_mm512_store_pd({names["aai"]} + {k}*8, {acci[k]});")
    BR = [None]*(h+1); BI = [None]*(h+1)
    for m in range(1, h+1):
        dmr = e.t(); e.l(f"__m512d {dmr} = _mm512_load_pd({names["dr"]} + {m}*8);")
        dmi = e.t(); e.l(f"__m512d {dmi} = _mm512_load_pd({names["di"]} + {m}*8);")
        for k in range(1, h+1):
            ss = float(np.sin(TWO_PI * LD((k*m) % p) / LD(p)))
            if m == 1:
                BR[k] = e.t("br"); e.l(f"__m512d {BR[k]} = _mm512_mul_pd({dmr}, _mm512_set1_pd({hexd(ss)}));")
                BI[k] = e.t("bi"); e.l(f"__m512d {BI[k]} = _mm512_mul_pd({dmi}, _mm512_set1_pd({hexd(ss)}));")
            else:
                e.l(f"{BR[k]} = _mm512_fmadd_pd({dmr}, _mm512_set1_pd({hexd(ss)}), {BR[k]});")
                e.l(f"{BI[k]} = _mm512_fmadd_pd({dmi}, _mm512_set1_pd({hexd(ss)}), {BI[k]});")
    for k in range(1, h+1):
        ar = e.t(); e.l(f"__m512d {ar} = _mm512_load_pd({names["aar"]} + {k}*8);")
        ai = e.t(); e.l(f"__m512d {ai} = _mm512_load_pd({names["aai"]} + {k}*8);")
        A = (ar, ai); Bv = (BR[k], BI[k])
        STf(k, c_a_m_ib(e, A, Bv))
        STf(p - k, c_a_p_ib(e, A, Bv))

def body_prime(e, p, LDf, STf):
    if p == 17:
        return body_prime_classic(e, p, LDf, STf)
    KH = 12 if p == 13 else 8
    h = (p - 1)//2
    pfx = e.t("pk")
    names = {nm: f"{pfx}{nm}" for nm in ("sr","si","dr","di","aar","aai")}
    for nm in names.values():
        e.l(f"double {nm}[{h+1}*8] __attribute__((aligned(64)));")
    x0 = LDf(0)
    x0r = e.t(); e.l(f"__m512d {x0r} = {x0[0]};")
    x0i = e.t(); e.l(f"__m512d {x0i} = {x0[1]};")
    for m in range(1, h+1):
        a = LDf(m); b = LDf(p - m)
        s = cadd(e, a, b); d = csub(e, a, b)
        for nm, v in (("sr", s[0]), ("si", s[1]), ("dr", d[0]), ("di", d[1])):
            e.l(f"_mm512_store_pd({names[nm]} + {m}*8, {v});")
    ks = list(range(0, h+1))
    blocks = [ks[i:i+KH] for i in range(0, len(ks), KH)]
    for blk in blocks:
        AR = {}; AI = {}
        for k in blk:
            AR[k] = e.t("ar"); e.l(f"__m512d {AR[k]} = {x0r};")
            AI[k] = e.t("ai"); e.l(f"__m512d {AI[k]} = {x0i};")
        for m in range(1, h+1):
            smr = e.t(); e.l(f"__m512d {smr} = _mm512_load_pd({names["sr"]} + {m}*8);")
            smi = e.t(); e.l(f"__m512d {smi} = _mm512_load_pd({names["si"]} + {m}*8);")
            for k in blk:
                if k == 0:
                    e.l(f"{AR[k]} = _mm512_add_pd({AR[k]}, {smr});")
                    e.l(f"{AI[k]} = _mm512_add_pd({AI[k]}, {smi});")
                else:
                    cc = float(np.cos(TWO_PI * LD((k*m) % p) / LD(p)))
                    e.l(f"{AR[k]} = _mm512_fmadd_pd({smr}, _mm512_set1_pd({hexd(cc)}), {AR[k]});")
                    e.l(f"{AI[k]} = _mm512_fmadd_pd({smi}, _mm512_set1_pd({hexd(cc)}), {AI[k]});")
        for k in blk:
            if k == 0:
                STf(0, (AR[0], AI[0]))
            else:
                e.l(f"_mm512_store_pd({names["aar"]} + {k}*8, {AR[k]});")
                e.l(f"_mm512_store_pd({names["aai"]} + {k}*8, {AI[k]});")
    ks = list(range(1, h+1))
    blocks = [ks[i:i+KH] for i in range(0, len(ks), KH)]
    for blk in blocks:
        BR = {}; BI = {}
        for m in range(1, h+1):
            dmr = e.t(); e.l(f"__m512d {dmr} = _mm512_load_pd({names["dr"]} + {m}*8);")
            dmi = e.t(); e.l(f"__m512d {dmi} = _mm512_load_pd({names["di"]} + {m}*8);")
            for k in blk:
                ss = float(np.sin(TWO_PI * LD((k*m) % p) / LD(p)))
                if m == 1:
                    BR[k] = e.t("br"); e.l(f"__m512d {BR[k]} = _mm512_mul_pd({dmr}, _mm512_set1_pd({hexd(ss)}));")
                    BI[k] = e.t("bi"); e.l(f"__m512d {BI[k]} = _mm512_mul_pd({dmi}, _mm512_set1_pd({hexd(ss)}));")
                else:
                    e.l(f"{BR[k]} = _mm512_fmadd_pd({dmr}, _mm512_set1_pd({hexd(ss)}), {BR[k]});")
                    e.l(f"{BI[k]} = _mm512_fmadd_pd({dmi}, _mm512_set1_pd({hexd(ss)}), {BI[k]});")
        for k in blk:
            ar = e.t(); e.l(f"__m512d {ar} = _mm512_load_pd({names["aar"]} + {k}*8);")
            ai = e.t(); e.l(f"__m512d {ai} = _mm512_load_pd({names["aai"]} + {k}*8);")
            A = (ar, ai); Bv = (BR[k], BI[k])
            STf(k, c_a_m_ib(e, A, Bv))
            STf(p - k, c_a_p_ib(e, A, Bv))

def emit_fft_body(e, L, LDf, STf):
    if L in (6, 8): body_small(e, L, LDf, STf)
    elif L in (13, 17, 23): body_prime(e, L, LDf, STf)
    elif L in (36, 45): body_pfa(e, L, LDf, STf)
    elif L == 64: body_64(e, LDf, STf)
    else: raise ValueError

# ---------------- nonlinear map N: x = z/(1+|z|) ----------------
def emit_N(e, zr, zi, jpar=None):
    # t = |z|^2 ; r = sqrt(t) via rsqrt14 + 1 NR (->2^-28) + 1 self-correcting sqrt step (->~1ulp)
    t = e.fma(zr, zr, e.mul(zi, zi))
    t2 = e.t(); e.l(f"__m512d {t2} = _mm512_max_pd({t}, _mm512_set1_pd(1e-300));")
    u = e.t(); e.l(f"__m512d {u} = _mm512_rsqrt14_pd({t2});")
    ht = e.mulk(t2, 0.5)
    hu = e.mul(ht, u)
    f = e.fnma(hu, u, "_mm512_set1_pd(1.5)")
    u = e.mul(u, f)            # u accurate to ~2^-28
    hu2 = e.mulk(u, 0.5)
    r0 = e.mul(t2, u)          # ~2^-28 sqrt
    err = e.fnma(r0, r0, t2)
    r = e.fma(err, hu2, r0)    # quadratic: ~2^-53 sqrt
    den = e.t(); e.l(f"__m512d {den} = _mm512_add_pd({r}, _mm512_set1_pd(1.0));")
    import os as _os
    mode = _os.environ.get("NMODE", "mix")
    use_div = (mode == "div") or (mode == "mix" and (jpar is None or (jpar & 1)))
    if use_div:
        v = e.t(); e.l(f"__m512d {v} = _mm512_div_pd(_mm512_set1_pd(1.0), {den});")
    else:
        v = e.t(); e.l(f"__m512d {v} = _mm512_rcp14_pd({den});")
        for _ in range(2):
            q = e.fnma(den, v, "_mm512_set1_pd(1.0)")
            v = e.fma(v, q, v)
    return e.mul(zr, v), e.mul(zi, v)

# ---------------- kernel function generators ----------------
FLAVA = (6, 8, 13, 17, 23)

def st_mem_g(e, S, pf=0):
    def STf(j, y):
        if pf:
            e.l(f"_mm_prefetch((const char*)(re + {j}*{S} + {pf}), _MM_HINT_T1);")
            e.l(f"_mm_prefetch((const char*)(im + {j}*{S} + {pf}), _MM_HINT_T1);")
        e.l(f"_mm512_store_pd(re + {j}*{S}, {y[0]});")
        e.l(f"_mm512_store_pd(im + {j}*{S}, {y[1]});")
    return STf

def emit_n_state_g(e, j, y, S):
    cr = e.t("cr"); e.l(f"__m512d {cr} = _mm512_load_pd(cre + {j}*{S});")
    ci = e.t("ci"); e.l(f"__m512d {ci} = _mm512_load_pd(cim + {j}*{S});")
    zr = e.add(y[0], cr); zi = e.add(y[1], ci)
    xr, xi = emit_N(e, zr, zi)
    e.l(f"_mm512_store_pd(re + {j}*{S}, {xr});")
    e.l(f"_mm512_store_pd(im + {j}*{S}, {xi});")

GSIZES = (6, 8, 13, 17)

def gen_kernels_g(L):
    out = []
    # kVY: stride 8L
    e = E()
    emit_fft_body(e, L, ld_mem(e, 8*L), st_mem_g(e, 8*L))
    out.append(fn(f"kVY_{L}", "double* restrict re, double* restrict im", e))
    # kVXN: stride = slab stride (padded for L=8)
    SS = slabstride(L)
    e = E()
    emit_fft_body(e, L, ld_mem(e, SS), lambda j, y: emit_n_state_g(e, j, y, SS))
    out.append(fn(f"kVXN_{L}", "double* restrict re, double* restrict im, const double* restrict cre, const double* restrict cim", e))
    return "\n".join(out)

def zpad(L):
    return {36: 40, 45: 48, 64: 72}.get(L, L)

def xstride(L):
    if L == 64: return 64*72 + 8
    if L == 45: return 45*48
    if L == 36: return 36*40
    return L*L

def slabstride(L):
    s = L*L*8
    return s + 8 if L == 8 else s

def gen_run_g(L):
    L2, L3 = L*L, L*L*L
    SS = slabstride(L)
    if SS == L2*8:
        convs_in = f"""  conv_in(x0, RE, IM, {L3}, {L3}, nv);
  conv_in(c, CR, CI, {L3}, {L3}, nv);"""
        conv1 = f"conv_out(RE, IM, o1, {L3}, {L3}, nv);"
        convm = f"conv_out(RE, IM, om, {L3}, {L3}, nv);"
    else:
        convs_in = f"""  for (long x = 0; x < {L}; x++) {{
    conv_in(x0 + 2*x*{L2}, RE + x*{SS}, IM + x*{SS}, {L2}, {L3}, nv);
    conv_in(c  + 2*x*{L2}, CR + x*{SS}, CI + x*{SS}, {L2}, {L3}, nv);
  }}"""
        conv1 = f"for (long x = 0; x < {L}; x++) conv_out(RE + x*{SS}, IM + x*{SS}, o1 + 2*x*{L2}, {L2}, {L3}, nv);"
        convm = f"for (long x = 0; x < {L}; x++) conv_out(RE + x*{SS}, IM + x*{SS}, om + 2*x*{L2}, {L2}, {L3}, nv);"
    return f"""static void rung_{L}(long nv, long m, const double* x0, const double* c, double* o1, double* om){{
{convs_in}
  for (long it = 1; it <= m; it++) {{
    for (long x = 0; x < {L}; x++) {{
      double* pr = RE + x*{SS}; double* pi = IM + x*{SS};
      for (long r = 0; r < {L}; r++) kZ_p_{L}(pr + r*{L}*8, pi + r*{L}*8);
      for (long z = 0; z < {L}; z++) kVY_{L}(pr + z*8, pi + z*8);
    }}
    for (long f0 = 0; f0 < {L2}; f0++) kVXN_{L}(RE + f0*8, IM + f0*8, CR + f0*8, CI + f0*8);
    if (it == 1) {conv1}
  }}
  {convm}
}}
"""

FLAVB = (36, 45, 64)
SIZES = (6, 8, 13, 17, 23, 36, 45, 64)

def fn(name, args, e):
    body = "\n".join(e.lines)
    return f"static void {name}({args}){{\n{body}\n}}\n"

def ld_mem(e, S, pf=0):
    def LDf(j):
        if pf:
            e.l(f"_mm_prefetch((const char*)(re + {j}*{S} + {pf}), _MM_HINT_T1);")
            e.l(f"_mm_prefetch((const char*)(im + {j}*{S} + {pf}), _MM_HINT_T1);")
        r = e.t("xr"); e.l(f"__m512d {r} = _mm512_loadu_pd(re + {j}*{S});")
        i = e.t("xi"); e.l(f"__m512d {i} = _mm512_loadu_pd(im + {j}*{S});")
        return (r, i)
    return LDf

def ld_xbuf(e):
    def LDf(j):
        r = e.t("xr"); e.l(f"__m512d {r} = _mm512_load_pd(xbr + {j}*8);")
        i = e.t("xi"); e.l(f"__m512d {i} = _mm512_load_pd(xbi + {j}*8);")
        return (r, i)
    return LDf

def st_mem(e, S, masked=True, pf=0):
    def STf(j, y):
        if pf:
            e.l(f"_mm_prefetch((const char*)(re + {j}*{S} + {pf}), _MM_HINT_T1);")
            e.l(f"_mm_prefetch((const char*)(im + {j}*{S} + {pf}), _MM_HINT_T1);")
        if masked:
            e.l(f"_mm512_mask_storeu_pd(re + {j}*{S}, msk, {y[0]});")
            e.l(f"_mm512_mask_storeu_pd(im + {j}*{S}, msk, {y[1]});")
        else:
            e.l(f"_mm512_store_pd(re + {j}*8, {y[0]});")
            e.l(f"_mm512_store_pd(im + {j}*8, {y[1]});")
    return STf

def emit_n_and_out(e, j, y, S, dest, pf=0, SOUT=None):
    if SOUT is None: SOUT = S
    if pf:
        e.l(f"_mm_prefetch((const char*)(cre + {j}*{S} + {pf}), _MM_HINT_T1);")
        e.l(f"_mm_prefetch((const char*)(cim + {j}*{S} + {pf}), _MM_HINT_T1);")
    cr = e.t("cr"); e.l(f"__m512d {cr} = _mm512_loadu_pd(cre + {j}*{S});")
    ci = e.t("ci"); e.l(f"__m512d {ci} = _mm512_loadu_pd(cim + {j}*{S});")
    zr = e.add(y[0], cr); zi = e.add(y[1], ci)
    xr, xi = emit_N(e, zr, zi, jpar=j)
    if dest == "mem":
        e.l(f"_mm512_mask_storeu_pd(re + {j}*{S}, msk, {xr});")
        e.l(f"_mm512_mask_storeu_pd(im + {j}*{S}, msk, {xi});")
        e.l(f"if (snap) ist(snap + 2*{j}*{SOUT}, {xr}, {xi}, msk);")
        e.l(f"if (fin) ist(fin + 2*{j}*{SOUT}, {xr}, {xi}, msk);")
    else:  # xbuf
        e.l(f"_mm512_store_pd(xbr + {j}*8, {xr});")
        e.l(f"_mm512_store_pd(xbi + {j}*8, {xi});")
        e.l(f"if (snap) ist(snap + 2*{j}*{SOUT}, {xr}, {xi}, msk);")

def gen_kernels_for(L):
    out = []
    L2 = L*L
    # kZ_p: z-buffer kernel (S=8), unmasked aligned
    e = E()
    emit_fft_body(e, L, ld_mem(e, 8), st_mem(e, 8, masked=False))
    out.append(fn(f"kZ_p_{L}", "double* restrict re, double* restrict im", e))
    # kY_p
    YS = zpad(L) if (L in FLAVB) else L
    e = E()
    emit_fft_body(e, L, ld_mem(e, YS), st_mem(e, YS))
    out.append(fn(f"kY_p_{L}", "double* restrict re, double* restrict im, __mmask8 msk", e))
    # kX_n
    pfx = zpad(L) if (L in FLAVB) else 0
    XS = xstride(L) if (L in FLAVB) else L2
    e = E()
    emit_fft_body(e, L, ld_mem(e, XS, pf=pfx), lambda j, y: emit_n_and_out(e, j, y, XS, "mem", pf=pfx, SOUT=L2))
    out.append(fn(f"kX_n_{L}",
        "double* restrict re, double* restrict im, const double* restrict cre, const double* restrict cim, double* restrict snap, double* restrict fin, __mmask8 msk", e))
    if L in FLAVB:
        # kY_n
        e = E()
        emit_fft_body(e, L, ld_mem(e, zpad(L), pf=xstride(L)), lambda j, y: emit_n_and_out(e, j, y, zpad(L), "mem", pf=xstride(L), SOUT=L))
        out.append(fn(f"kY_n_{L}",
            "double* restrict re, double* restrict im, const double* restrict cre, const double* restrict cim, double* restrict snap, double* restrict fin, __mmask8 msk", e))
        # fused KNK
        out.append(gen_knk(L, "X", xstride(L), zpad(L), SOUT=L2))
        out.append(gen_knk(L, "Y", zpad(L), xstride(L), SOUT=L))
    return "\n".join(out)

def gen_pz(L):
    # z-pass: rows of length L (contiguous), transpose-tiles through ZB; re/im interleaved per tile
    full = (L//8)*8
    s = []
    s.append(f"static void pz_{L}(double* restrict re, double* restrict im, long base, long rs, long nrows, long pfd){{")
    s.append(f"  (void)pfd;")
    s.append(f"  for (long rq = 0; rq < nrows; rq += 8) {{")
    s.append(f"    long cnt = nrows - rq; if (cnt > 8) cnt = 8;")
    s.append(f"    const long b0 = base + rq*rs;")
    s.append(f"    {{ const double* p = re + b0; const double* q = im + b0;")
    s.append(f"      for (long j0 = 0; j0 < {full}; j0 += 8) {{")
    for arr, zb, ptr in (("re","ZBR","p"), ("im","ZBI","q")):
        for l in range(8):
            s.append(f"        __m512d {arr}{l} = _mm512_loadu_pd({ptr} + j0 + {l}*rs);")
    for arr, zb in (("re","ZBR"), ("im","ZBI")):
        s.append(f"        TR8({arr}0,{arr}1,{arr}2,{arr}3,{arr}4,{arr}5,{arr}6,{arr}7);")
        for l in range(8):
            s.append(f"        _mm512_store_pd({zb} + (j0+{l})*8, {arr}{l});")
    s.append(f"      }}")
    if L % 8:
        rem = L % 8
        s.append(f"      {{")
        for arr, zb, ptr in (("re","ZBR","p"), ("im","ZBI","q")):
            for l in range(8):
                s.append(f"        __m512d {arr}{l} = _mm512_loadu_pd({ptr} + {full} + {l}*rs);")
        for arr, zb in (("re","ZBR"), ("im","ZBI")):
            s.append(f"        TR8({arr}0,{arr}1,{arr}2,{arr}3,{arr}4,{arr}5,{arr}6,{arr}7);")
            for l in range(rem):
                s.append(f"        _mm512_store_pd({zb} + ({full}+{l})*8, {arr}{l});")
        s.append(f"      }}")
    s.append(f"    }}")
    s.append(f"    kZ_p_{L}(ZBR, ZBI);")
    s.append(f"    {{ double* p = re + b0; double* q = im + b0;")
    s.append(f"      for (long j0 = 0; j0 < {full}; j0 += 8) {{")
    for arr, zb in (("re","ZBR"), ("im","ZBI")):
        for l in range(8):
            s.append(f"        __m512d {arr}{l} = _mm512_load_pd({zb} + (j0+{l})*8);")
        s.append(f"        TR8({arr}0,{arr}1,{arr}2,{arr}3,{arr}4,{arr}5,{arr}6,{arr}7);")
    s.append(f"        if (cnt == 8) {{")
    for arr, ptr in (("re","p"), ("im","q")):
        for l in range(8):
            s.append(f"          _mm512_storeu_pd({ptr} + j0 + {l}*rs, {arr}{l});")
    s.append(f"        }} else {{")
    for arr, ptr in (("re","p"), ("im","q")):
        for l in range(8):
            s.append(f"          if ({l} < cnt) _mm512_storeu_pd({ptr} + j0 + {l}*rs, {arr}{l});")
    s.append(f"        }}")
    s.append(f"      }}")
    if L % 8:
        rem = L % 8
        cm = (1 << rem) - 1
        s.append(f"      {{")
        for arr, zb in (("re","ZBR"), ("im","ZBI")):
            for l in range(8):
                idx = full + l
                if l < rem:
                    s.append(f"        __m512d {arr}{l} = _mm512_load_pd({zb} + {idx}*8);")
                else:
                    s.append(f"        __m512d {arr}{l} = _mm512_setzero_pd();")
            s.append(f"        TR8({arr}0,{arr}1,{arr}2,{arr}3,{arr}4,{arr}5,{arr}6,{arr}7);")
        for arr, ptr in (("re","p"), ("im","q")):
            for l in range(8):
                s.append(f"        if ({l} < cnt) _mm512_mask_storeu_pd({ptr} + {full} + {l}*rs, {cm}, {arr}{l});")
        s.append(f"      }}")
    s.append(f"    }}")
    s.append(f"  }}")
    s.append(f"}}")
    s.append("")
    return chr(10).join(s)

def gen_passes_A(L):
    L2 = L*L
    s = []
    s.append(f"""static void pass_y_{L}(double* restrict re, double* restrict im){{
  for (long x = 0; x < {L}; x++) {{
    double* pr = re + x*{L2}; double* pi = im + x*{L2};
    for (long z0 = 0; z0 < {L}; z0 += 8) {{
      __mmask8 mk = ({L} - z0 >= 8) ? (__mmask8)0xFF : (__mmask8)((1u << ({L} - z0)) - 1);
      kY_p_{L}(pr + z0, pi + z0, mk);
    }}
  }}
}}""")
    s.append(f"""static void pass_x_n_{L}(double* restrict re, double* restrict im, const double* restrict cre, const double* restrict cim, double* snap, double* fin){{
  for (long b0 = 0; b0 < {L2}; b0 += 8) {{
    __mmask8 mk = ({L2} - b0 >= 8) ? (__mmask8)0xFF : (__mmask8)((1u << ({L2} - b0)) - 1);
    kX_n_{L}(re + b0, im + b0, cre + b0, cim + b0, snap ? snap + 2*b0 : 0, fin ? fin + 2*b0 : 0, mk);
  }}
}}""")
    return "\n".join(s) + "\n"

def gen_sweeps_B(L):
    L2 = L*L
    s = []
    XS = xstride(L); ZP = zpad(L)
    for pt, S, kt in (("xz", L2, "X"), ("yz", L, "Y")):
        pb = f"g*{ZP}" if pt == "xz" else f"g*{XS}"
        rs = XS if pt == "xz" else ZP
        pfd = 0
        pbo_out = f"g*{L}" if pt == "xz" else f"g*{L2}"
        plain_call = f"kY_p_{L}(re + off, im + off, mk);" if pt == "yz" else "__builtin_unreachable();"
        s.append(f"""static void sweep_{pt}_{L}(double* restrict re, double* restrict im, const double* restrict cre, const double* restrict cim, double* snap, double* fin, int lag, int la){{
  for (long g = 0; g < {L}; g++) {{
    long pbo = {pb};
    for (long z0 = 0; z0 < {L}; z0 += 8) {{
      __mmask8 mk = ({L} - z0 >= 8) ? (__mmask8)0xFF : (__mmask8)((1u << ({L} - z0)) - 1);
      long off = pbo + z0;
      long offo = {pbo_out} + z0;
      if (lag) {{
        if (la) {{
          k{kt}_knk_{L}(re + off, im + off, cre + off, cim + off, snap ? snap + 2*offo : 0, mk);
        }} else {{
          k{kt}_n_{L}(re + off, im + off, cre + off, cim + off, snap ? snap + 2*offo : 0, fin ? fin + 2*offo : 0, mk);
        }}
      }} else {{
        {plain_call}
      }}
    }}
    if (la) pz_{L}(re, im, pbo, {rs}, {L}, {pfd});
  }}
}}""")
    return "\n".join(s) + "\n"


def emit_n_xbuf_g(e, j, y, S, pf=0):
    if pf:
        e.l(f"_mm_prefetch((const char*)(cre + {j}*{S} + {pf}), _MM_HINT_T1);")
        e.l(f"_mm_prefetch((const char*)(cim + {j}*{S} + {pf}), _MM_HINT_T1);")
    cr = e.t("cr"); e.l(f"__m512d {cr} = _mm512_load_pd(cre + {j}*{S});")
    ci = e.t("ci"); e.l(f"__m512d {ci} = _mm512_load_pd(cim + {j}*{S});")
    zr = e.add(y[0], cr); zi = e.add(y[1], ci)
    xr, xi = emit_N(e, zr, zi, jpar=j)
    e.l(f"_mm512_store_pd(xbr + {j}*8, {xr});")
    e.l(f"_mm512_store_pd(xbi + {j}*8, {xi});")
    e.l(f"if (snr) {{ _mm512_store_pd(snr + {j}*{S}, {xr}); _mm512_store_pd(sni + {j}*{S}, {xi}); }}")

def emit_n_state_g2(e, j, y, S, pf=0):
    if pf:
        e.l(f"_mm_prefetch((const char*)(cre + {j}*{S} + {pf}), _MM_HINT_T1);")
        e.l(f"_mm_prefetch((const char*)(cim + {j}*{S} + {pf}), _MM_HINT_T1);")
    cr = e.t("cr"); e.l(f"__m512d {cr} = _mm512_load_pd(cre + {j}*{S});")
    ci = e.t("ci"); e.l(f"__m512d {ci} = _mm512_load_pd(cim + {j}*{S});")
    zr = e.add(y[0], cr); zi = e.add(y[1], ci)
    xr, xi = emit_N(e, zr, zi)
    e.l(f"_mm512_store_pd(re + {j}*{S}, {xr});")
    e.l(f"_mm512_store_pd(im + {j}*{S}, {xi});")

BGSIZES = (13, 17, 23)


def gen_knk(L, tag, S, pf, group=False, SOUT=None):
    if SOUT is None: SOUT = S
    e = E()
    e.l(f"double txr[{L}*8] __attribute__((aligned(64)));")
    e.l(f"double txi[{L}*8] __attribute__((aligned(64)));")
    def st1(j, y):
        if pf:
            e.l(f"_mm_prefetch((const char*)(cre + {j}*{S} + {pf}), _MM_HINT_T1);")
            e.l(f"_mm_prefetch((const char*)(cim + {j}*{S} + {pf}), _MM_HINT_T1);")
        cr = e.t("cr"); e.l(f"__m512d {cr} = _mm512_loadu_pd(cre + {j}*{S});")
        ci = e.t("ci"); e.l(f"__m512d {ci} = _mm512_loadu_pd(cim + {j}*{S});")
        zr = e.add(y[0], cr); zi = e.add(y[1], ci)
        xr, xi = emit_N(e, zr, zi, jpar=j)
        e.l(f"_mm512_store_pd(txr + {j}*8, {xr});")
        e.l(f"_mm512_store_pd(txi + {j}*8, {xi});")
        if group:
            e.l(f"if (snr) {{ _mm512_store_pd(snr + {j}*{S}, {xr}); _mm512_store_pd(sni + {j}*{S}, {xi}); }}")
        else:
            e.l(f"if (snap) ist(snap + 2*{j}*{SOUT}, {xr}, {xi}, msk);")
    emit_fft_body(e, L, ld_mem(e, S, pf=pf), st1)
    def ld2(j):
        r = e.t("yr"); e.l(f"__m512d {r} = _mm512_load_pd(txr + {j}*8);")
        i = e.t("yi"); e.l(f"__m512d {i} = _mm512_load_pd(txi + {j}*8);")
        return (r, i)
    if group:
        emit_fft_body(e, L, ld2, st_mem_g(e, S))
        return fn(f"kg{tag}_knk_{L}",
            "double* restrict re, double* restrict im, const double* restrict cre, const double* restrict cim, double* restrict snr, double* restrict sni", e)
    else:
        emit_fft_body(e, L, ld2, st_mem(e, S))
        return fn(f"k{tag}_knk_{L}",
            "double* restrict re, double* restrict im, const double* restrict cre, const double* restrict cim, double* restrict snap, __mmask8 msk", e)


def gen_knk64(tag, S, pf):
    """Fused KNK for L=64: FFT1.ph1 -> buf1; [FFT1.ph2 | N | FFT2.ph1] -> buf2; FFT2.ph2 -> out."""
    e = E()
    b1r = e.t("b1r"); b1i = e.t("b1i"); b2r = e.t("b2r"); b2i = e.t("b2i")
    for nm in (b1r, b1i, b2r, b2i):
        e.l(f"double {nm}[64*8] __attribute__((aligned(64)));")
    LDf = ld_mem(e, S, pf=pf)
    # phase 1
    for b in range(8):
        xs = [LDf(8*a + b) for a in range(8)]
        ys = dft8(e, xs)
        for k1 in range(8):
            v = cmul_tw(e, ys[k1], tw(b*k1, 64))
            e.l(f"_mm512_store_pd({b1r} + {(k1*8+b)}*8, {v[0]});")
            e.l(f"_mm512_store_pd({b1i} + {(k1*8+b)}*8, {v[1]});")
    # phase 2: per k1: FFT1 stage2 + N + FFT2 stage1 (column b'=k1)
    for k1 in range(8):
        xs = []
        for b in range(8):
            r = e.t(); e.l(f"__m512d {r} = _mm512_load_pd({b1r} + {(k1*8+b)}*8);")
            i = e.t(); e.l(f"__m512d {i} = _mm512_load_pd({b1i} + {(k1*8+b)}*8);")
            xs.append((r, i))
        ys = dft8(e, xs)   # ys[k2] = FFT1 out at j = k1 + 8*k2
        ns = []
        for k2 in range(8):
            j = k1 + 8*k2
            if pf:
                e.l(f"_mm_prefetch((const char*)(cre + {j}*{S} + {pf}), _MM_HINT_T1);")
                e.l(f"_mm_prefetch((const char*)(cim + {j}*{S} + {pf}), _MM_HINT_T1);")
            cr = e.t("cr"); e.l(f"__m512d {cr} = _mm512_loadu_pd(cre + {j}*{S});")
            ci = e.t("ci"); e.l(f"__m512d {ci} = _mm512_loadu_pd(cim + {j}*{S});")
            zr = e.add(ys[k2][0], cr); zi = e.add(ys[k2][1], ci)
            xr, xi = emit_N(e, zr, zi, jpar=j)
            e.l(f"if (snap) ist(snap + 2*{j}*{S}, {xr}, {xi}, msk);")
            ns.append((xr, xi))
        # FFT2 stage-1 on column b' = k1 (inputs x'[8a+k1] = ns[a])
        us = dft8(e, ns)
        for k1p in range(8):
            v = cmul_tw(e, us[k1p], tw(k1*k1p, 64))
            e.l(f"_mm512_store_pd({b2r} + {(k1p*8+k1)}*8, {v[0]});")
            e.l(f"_mm512_store_pd({b2i} + {(k1p*8+k1)}*8, {v[1]});")
    # phase 3
    STf = st_mem(e, S)
    for k1p in range(8):
        xs = []
        for b in range(8):
            r = e.t(); e.l(f"__m512d {r} = _mm512_load_pd({b2r} + {(k1p*8+b)}*8);")
            i = e.t(); e.l(f"__m512d {i} = _mm512_load_pd({b2i} + {(k1p*8+b)}*8);")
            xs.append((r, i))
        ys = dft8(e, xs)
        for k2 in range(8):
            STf(k1p + 8*k2, ys[k2])
    return fn(f"k{tag}_knk_64",
        "double* restrict re, double* restrict im, const double* restrict cre, const double* restrict cim, double* restrict snap, __mmask8 msk", e)

def gen_kernels_bg(L):
    out = []
    L2 = L*L
    pfx = 8*L if L == 23 else 0     # next xz-plane
    pfy = 8*L2 if L == 23 else 0    # next yz-plane
    for tag, S, pf in (("X", 8*L2, pfx), ("Y", 8*L, pfy)):
        # nx: FFT -> c+N -> xbuf (+optional snap staging)
        e = E()
        emit_fft_body(e, L, ld_mem(e, S, pf=pf), lambda j, y: emit_n_xbuf_g(e, j, y, S, pf=pf))
        out.append(fn(f"kg{tag}_nx_{L}",
            "const double* restrict re, const double* restrict im, const double* restrict cre, const double* restrict cim, double* restrict snr, double* restrict sni, double* restrict xbr, double* restrict xbi", e))
        # lx: xbuf -> FFT -> state
        e = E()
        emit_fft_body(e, L, ld_xbuf(e), st_mem_g(e, S))
        out.append(fn(f"kg{tag}_lx_{L}",
            "const double* restrict xbr, const double* restrict xbi, double* restrict re, double* restrict im", e))
        # fused KNK
        out.append(gen_knk(L, tag, S, pf, group=True))
        # n: FFT -> c+N -> state (final sweep)
        e = E()
        emit_fft_body(e, L, ld_mem(e, S, pf=pf), lambda j, y: emit_n_state_g2(e, j, y, S, pf=pf))
        out.append(fn(f"kg{tag}_n_{L}",
            "double* restrict re, double* restrict im, const double* restrict cre, const double* restrict cim", e))
    # plain Y kernel for s=1 (reuse kVY if present; emit for 23)
    if L not in GSIZES:
        e = E()
        emit_fft_body(e, L, ld_mem(e, 8*L), st_mem_g(e, 8*L))
        out.append(fn(f"kVY_{L}", "double* restrict re, double* restrict im", e))
    return "\n".join(out)

def gen_run_bg(L):
    L2, L3 = L*L, L*L*L
    return f"""static void gsw_xz_{L}(const double* cre, const double* cim, int lag, int la, int snap){{
  for (long g = 0; g < {L}; g++) {{
    for (long z = 0; z < {L}; z++) {{
      long off = (g*{L} + z)*8;
      if (la) {{
        kgX_knk_{L}(RE+off, IM+off, cre+off, cim+off, snap?SN1R+off:0, snap?SN1I+off:0);
      }} else {{
        kgX_n_{L}(RE+off, IM+off, cre+off, cim+off);
      }}
    }}
    if (la) for (long x = 0; x < {L}; x++) {{ long o2 = (x*{L2} + g*{L})*8; kZ_p_{L}(RE+o2, IM+o2); }}
  }}
}}
static void gsw_yz_{L}(const double* cre, const double* cim, int lag, int la, int snap){{
  for (long f = 0; f < {L}; f++) {{
    for (long z = 0; z < {L}; z++) {{
      long off = (f*{L2} + z)*8;
      if (!lag) {{ kVY_{L}(RE+off, IM+off); }}
      else if (la) {{
        kgY_knk_{L}(RE+off, IM+off, cre+off, cim+off, snap?SN1R+off:0, snap?SN1I+off:0);
      }} else {{
        kgY_n_{L}(RE+off, IM+off, cre+off, cim+off);
      }}
    }}
    if (la) for (long y = 0; y < {L}; y++) {{ long o2 = (f*{L2} + y*{L})*8; kZ_p_{L}(RE+o2, IM+o2); }}
  }}
}}
static void rung_{L}(long nv, long m, const double* x0, const double* c, double* o1, double* om){{
  conv_in(x0, RE, IM, {L3}, {L3}, nv);
  conv_in(c, CR, CI, {L3}, {L3}, nv);
  for (long s = 1; s <= m + 1; s++) {{
    int la = (s <= m), lag = (s >= 2);
    int snap = (s == 2) && (m >= 2);
    if (s & 1) gsw_yz_{L}(CR, CI, lag, la, snap);
    else       gsw_xz_{L}(CR, CI, lag, la, snap);
  }}
  if (m >= 2) conv_out(SN1R, SN1I, o1, {L3}, {L3}, nv);
  else        conv_out(RE, IM, o1, {L3}, {L3}, nv);
  conv_out(RE, IM, om, {L3}, {L3}, nv);
}}
"""

def gen_run(L):
    L3 = L*L*L
    if L in FLAVA:
        body = f"""    for (long it = 1; it <= m; it++) {{
      pz_{L}(RE, IM, 0, {L}, {L*L}, 0);
      pass_y_{L}(RE, IM);
      pass_x_n_{L}(RE, IM, CR, CI, it == 1 ? snap : 0, it == m ? fin : 0);
    }}"""
    else:
        body = f"""    for (long s = 1; s <= m + 1; s++) {{
      int la = (s <= m), lag = (s >= 2);
      double* sn = (s == 2) ? snap : 0;
      double* fi = (s == m + 1) ? fin : 0;
      if (s & 1) sweep_yz_{L}(RE, IM, CR, CI, sn, fi, lag, la);
      else       sweep_xz_{L}(RE, IM, CR, CI, sn, fi, lag, la);
    }}"""
    gdisp = ""
    if L in GSIZES or L in BGSIZES:
        gdisp = f"""  while (B - b >= GTH_{L}) {{
    long nv = B - b; if (nv > 8) nv = 8;
    rung_{L}(nv, m, x0 + 2*b*{L3}, c + 2*b*{L3}, o1 + 2*b*{L3}, om + 2*b*{L3});
    b += nv;
  }}
"""
    XS = xstride(L); ZP = zpad(L)
    if L in FLAVB and ZP != L:
        dein = f"""    for (long x = 0; x < {L}; x++) {{
      deinter_rows(x0 + 2*b*{L3} + 2*x*{L*L}, RE + x*{XS}, IM + x*{XS}, {L}, {L}, {ZP});
      deinter_rows(c  + 2*b*{L3} + 2*x*{L*L}, CR + x*{XS}, CI + x*{XS}, {L}, {L}, {ZP});
    }}"""
    elif L in FLAVB and XS != L*L:
        dein = f"""    for (long x = 0; x < {L}; x++) {{
      deinter(x0 + 2*b*{L3} + 2*x*{L*L}, RE + x*{XS}, IM + x*{XS}, {L*L});
      deinter(c  + 2*b*{L3} + 2*x*{L*L}, CR + x*{XS}, CI + x*{XS}, {L*L});
    }}"""
    else:
        dein = f"""    deinter(x0 + 2*b*{L3}, RE, IM, {L3});
    deinter(c + 2*b*{L3}, CR, CI, {L3});"""
    return f"""static void run_{L}(long B, long m, const double* x0, const double* c, double* o1, double* om){{
  long b = 0;
{gdisp}  for (; b < B; b++) {{
{dein}
    double* snap = o1 + 2*b*{L3};
    double* fin  = om + 2*b*{L3};
{body}
  }}
}}
"""

# ---------------- C scaffold ----------------
def pair_masks():
    lo = []; hi = []
    for m in range(256):
        a = 0; b = 0
        for j in range(4):
            if (m >> j) & 1: a |= 3 << (2*j)
            if (m >> (4+j)) & 1: b |= 3 << (2*j)
        lo.append(a); hi.append(b)
    return lo, hi

HEADER = r"""
// Auto-generated AVX-512 implementation of iterated batched 3D complex FFTs.
// All transform arithmetic is original (PFA / Cooley-Tukey / symmetric direct DFT).
#include <immintrin.h>
#include <stdint.h>
#include <string.h>

#include <sys/mman.h>
#include <stdlib.h>
#define PAD 40960
#define GTH_6 4
#define GTH_8 6
#define GTH_13 5
#define GTH_17 6
#define GTH_23 8
static double *RE, *IM, *CR, *CI, *SN1R, *SN1I;
static double ZBR[64*8] __attribute__((aligned(64)));
static double ZBI[64*8] __attribute__((aligned(64)));
static double XBR[64*8] __attribute__((aligned(64)));
static double XBI[64*8] __attribute__((aligned(64)));

#define TR8(a0,a1,a2,a3,a4,a5,a6,a7) do{ \
  __m512d _t0=_mm512_unpacklo_pd(a0,a1), _t1=_mm512_unpackhi_pd(a0,a1); \
  __m512d _t2=_mm512_unpacklo_pd(a2,a3), _t3=_mm512_unpackhi_pd(a2,a3); \
  __m512d _t4=_mm512_unpacklo_pd(a4,a5), _t5=_mm512_unpackhi_pd(a4,a5); \
  __m512d _t6=_mm512_unpacklo_pd(a6,a7), _t7=_mm512_unpackhi_pd(a6,a7); \
  __m512d _u0=_mm512_shuffle_f64x2(_t0,_t2,0x88), _u2=_mm512_shuffle_f64x2(_t0,_t2,0xDD); \
  __m512d _u1=_mm512_shuffle_f64x2(_t1,_t3,0x88), _u3=_mm512_shuffle_f64x2(_t1,_t3,0xDD); \
  __m512d _u4=_mm512_shuffle_f64x2(_t4,_t6,0x88), _u6=_mm512_shuffle_f64x2(_t4,_t6,0xDD); \
  __m512d _u5=_mm512_shuffle_f64x2(_t5,_t7,0x88), _u7=_mm512_shuffle_f64x2(_t5,_t7,0xDD); \
  a0=_mm512_shuffle_f64x2(_u0,_u4,0x88); a4=_mm512_shuffle_f64x2(_u0,_u4,0xDD); \
  a1=_mm512_shuffle_f64x2(_u1,_u5,0x88); a5=_mm512_shuffle_f64x2(_u1,_u5,0xDD); \
  a2=_mm512_shuffle_f64x2(_u2,_u6,0x88); a6=_mm512_shuffle_f64x2(_u2,_u6,0xDD); \
  a3=_mm512_shuffle_f64x2(_u3,_u7,0x88); a7=_mm512_shuffle_f64x2(_u3,_u7,0xDD); \
}while(0)

static const uint8_t PMLO[256] = {PMLO_INIT};
static const uint8_t PMHI[256] = {PMHI_INIT};

static inline void ist(double* p, __m512d r, __m512d i, __mmask8 mk){
  const __m512i ILO = _mm512_setr_epi64(0,8,1,9,2,10,3,11);
  const __m512i IHI = _mm512_setr_epi64(4,12,5,13,6,14,7,15);
  _mm512_mask_storeu_pd(p,     (__mmask8)PMLO[mk], _mm512_permutex2var_pd(r, ILO, i));
  _mm512_mask_storeu_pd(p + 8, (__mmask8)PMHI[mk], _mm512_permutex2var_pd(r, IHI, i));
}


static void conv_in(const double* src, double* dr, double* di, long n, long vstep, long nv){
  const __m512i IDXE = _mm512_setr_epi64(0,2,4,6,8,10,12,14);
  const __m512i IDXO = _mm512_setr_epi64(1,3,5,7,9,11,13,15);
  for (long f0 = 0; f0 < n; f0 += 8){
    long r = n - f0; if (r > 8) r = 8;
    __mmask8 m1 = (2*r >= 8) ? (__mmask8)0xFF : (__mmask8)((1u<<(2*r))-1);
    __mmask8 m2 = (2*r >= 16) ? (__mmask8)0xFF : ((2*r > 8) ? (__mmask8)((1u<<(2*r-8))-1) : (__mmask8)0);
    __m512d R0,R1,R2,R3,R4,R5,R6,R7,I0,I1,I2,I3,I4,I5,I6,I7;
    #define LDV(v, RV, IV) do{ \
      if (v < nv){ \
        __m512d lo=_mm512_maskz_loadu_pd(m1, src + v*2*vstep + 2*f0); \
        __m512d hi=_mm512_maskz_loadu_pd(m2, src + v*2*vstep + 2*f0 + 8); \
        RV=_mm512_permutex2var_pd(lo,IDXE,hi); IV=_mm512_permutex2var_pd(lo,IDXO,hi); \
      } else { RV=_mm512_set1_pd(1.0); IV=_mm512_set1_pd(1.0); } \
    }while(0)
    LDV(0,R0,I0); LDV(1,R1,I1); LDV(2,R2,I2); LDV(3,R3,I3);
    LDV(4,R4,I4); LDV(5,R5,I5); LDV(6,R6,I6); LDV(7,R7,I7);
    #undef LDV
    TR8(R0,R1,R2,R3,R4,R5,R6,R7);
    TR8(I0,I1,I2,I3,I4,I5,I6,I7);
    #define STF(k, RV, IV) do{ if (k < r){ _mm512_store_pd(dr+(f0+k)*8, RV); _mm512_store_pd(di+(f0+k)*8, IV);} }while(0)
    STF(0,R0,I0); STF(1,R1,I1); STF(2,R2,I2); STF(3,R3,I3);
    STF(4,R4,I4); STF(5,R5,I5); STF(6,R6,I6); STF(7,R7,I7);
    #undef STF
  }
}

static void conv_out(const double* dr, const double* di, double* dst, long n, long vstep, long nv){
  for (long f0 = 0; f0 < n; f0 += 8){
    long r = n - f0; if (r > 8) r = 8;
    __mmask8 fm = (r >= 8) ? (__mmask8)0xFF : (__mmask8)((1u<<r)-1);
    __m512d R0,R1,R2,R3,R4,R5,R6,R7,I0,I1,I2,I3,I4,I5,I6,I7;
    #define LDF(k, RV, IV) do{ long idx = (k < r) ? (f0+k) : (n-1); RV=_mm512_load_pd(dr+idx*8); IV=_mm512_load_pd(di+idx*8); }while(0)
    LDF(0,R0,I0); LDF(1,R1,I1); LDF(2,R2,I2); LDF(3,R3,I3);
    LDF(4,R4,I4); LDF(5,R5,I5); LDF(6,R6,I6); LDF(7,R7,I7);
    #undef LDF
    TR8(R0,R1,R2,R3,R4,R5,R6,R7);
    TR8(I0,I1,I2,I3,I4,I5,I6,I7);
    #define STV(v, RV, IV) do{ if (v < nv) ist(dst + v*2*vstep + 2*f0, RV, IV, fm); }while(0)
    STV(0,R0,I0); STV(1,R1,I1); STV(2,R2,I2); STV(3,R3,I3);
    STV(4,R4,I4); STV(5,R5,I5); STV(6,R6,I6); STV(7,R7,I7);
    #undef STV
  }
}

static void deinter_rows(const double* src, double* dr, double* di, long L, long nrows, long zp){
  const __m512i IDXE = _mm512_setr_epi64(0,2,4,6,8,10,12,14);
  const __m512i IDXO = _mm512_setr_epi64(1,3,5,7,9,11,13,15);
  long full = (L/8)*8;
  for (long r = 0; r < nrows; r++){
    const double* s = src + 2*r*L;
    double* pr = dr + r*zp; double* pi = di + r*zp;
    long i = 0;
    for (; i < full; i += 8){
      __m512d lo = _mm512_loadu_pd(s + 2*i), hi = _mm512_loadu_pd(s + 2*i + 8);
      _mm512_storeu_pd(pr + i, _mm512_permutex2var_pd(lo, IDXE, hi));
      _mm512_storeu_pd(pi + i, _mm512_permutex2var_pd(lo, IDXO, hi));
    }
    if (i < L){
      long rr = L - i;
      __mmask8 m1 = (2*rr >= 8) ? (__mmask8)0xFF : (__mmask8)((1u<<(2*rr))-1);
      __mmask8 m2 = (2*rr >= 16) ? (__mmask8)0xFF : ((2*rr > 8) ? (__mmask8)((1u<<(2*rr-8))-1) : (__mmask8)0);
      __m512d lo = _mm512_maskz_loadu_pd(m1, s + 2*i);
      __m512d hi = _mm512_maskz_loadu_pd(m2, s + 2*i + 8);
      __mmask8 mo = (__mmask8)((1u << rr) - 1);
      _mm512_mask_storeu_pd(pr + i, mo, _mm512_permutex2var_pd(lo, IDXE, hi));
      _mm512_mask_storeu_pd(pi + i, mo, _mm512_permutex2var_pd(lo, IDXO, hi));
    }
  }
}

static void deinter(const double* src, double* dr, double* di, long n){
  const __m512i IDXE = _mm512_setr_epi64(0,2,4,6,8,10,12,14);
  const __m512i IDXO = _mm512_setr_epi64(1,3,5,7,9,11,13,15);
  long i = 0;
  for (; i + 8 <= n; i += 8){
    __m512d lo = _mm512_loadu_pd(src + 2*i), hi = _mm512_loadu_pd(src + 2*i + 8);
    _mm512_store_pd(dr + i, _mm512_permutex2var_pd(lo, IDXE, hi));
    _mm512_store_pd(di + i, _mm512_permutex2var_pd(lo, IDXO, hi));
  }
  if (i < n){
    long r = n - i;
    __mmask8 m1 = (2*r >= 8) ? (__mmask8)0xFF : (__mmask8)((1u << (2*r)) - 1);
    __mmask8 m2 = (2*r > 8) ? (__mmask8)((1u << (2*r - 8)) - 1) : (__mmask8)0;
    __m512d lo = _mm512_maskz_loadu_pd(m1, src + 2*i);
    __m512d hi = _mm512_maskz_loadu_pd(m2, src + 2*i + 8);
    __mmask8 mo = (__mmask8)((1u << r) - 1);
    _mm512_mask_store_pd(dr + i, mo, _mm512_permutex2var_pd(lo, IDXE, hi));
    _mm512_mask_store_pd(di + i, mo, _mm512_permutex2var_pd(lo, IDXO, hi));
  }
}
"""

FOOTER_TMPL = r"""
void ft_warm(void);
void ft_run(long L, long B, long m, const double* x0, const double* c, double* o1, double* om){
  if (!RE) ft_warm();
  unsigned int mxcsr_old = _mm_getcsr();
  _mm_setcsr(mxcsr_old | 0x8040u);   /* FTZ | DAZ: avoid denormal stalls; values ~1e-308 are irrelevant here */
  switch (L) {
SWITCH_CASES
  }
  _mm_setcsr(mxcsr_old);
}

static double* big_alloc(size_t n){
  size_t bytes = (n*sizeof(double) + (2UL<<20) - 1) & ~((2UL<<20)-1);
  void* p = mmap(0, bytes, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
  if (p == MAP_FAILED) {
    p = aligned_alloc(64, bytes);            /* fallback: plain pages */
  } else {
    madvise(p, bytes, MADV_HUGEPAGE);
  }
  if (p) memset(p, 0, bytes);
  return (double*)p;
}
static void fill1(double* p, long n){ for (long i = 0; i < n; i++) p[i] = 1.0; }
void ft_warm(void){
  if (!RE){
    RE = big_alloc(262144+PAD); IM = big_alloc(262144+PAD);
    CR = big_alloc(262144+PAD); CI = big_alloc(262144+PAD);
    SN1R = big_alloc(12167*8+64); SN1I = big_alloc(12167*8+64);
  }
  fill1(RE, 262144+PAD); fill1(IM, 262144+PAD);
  fill1(CR, 262144+PAD); fill1(CI, 262144+PAD);
  fill1(SN1R, 12167*8+64); fill1(SN1I, 12167*8+64);
  fill1(ZBR, 64*8); fill1(ZBI, 64*8); fill1(XBR, 64*8); fill1(XBI, 64*8);
}
"""

def main():
    lo, hi = pair_masks()
    parts = [HEADER.replace("{PMLO_INIT}", "{" + ",".join(map(str, lo)) + "}").replace("{PMHI_INIT}", "{" + ",".join(map(str, hi)) + "}")]
    for L in SIZES:
        parts.append(gen_kernels_for(L))
        parts.append(gen_pz(L))
        if L in FLAVA:
            parts.append(gen_passes_A(L))
        else:
            parts.append(gen_sweeps_B(L))
        if L in BGSIZES:
            if L in GSIZES:
                parts.append(gen_kernels_g(L))
            parts.append(gen_kernels_bg(L))
            parts.append(gen_run_bg(L))
        elif L in GSIZES:
            parts.append(gen_kernels_g(L))
            parts.append(gen_run_g(L))
        parts.append(gen_run(L))
    cases = "\n".join(f"    case {L}: run_{L}(B, m, x0, c, o1, om); break;" for L in SIZES)
    parts.append(FOOTER_TMPL.replace("SWITCH_CASES", cases))
    open("implementation.c", "w").write("\n".join(parts))
    print("wrote implementation.c")

if __name__ == "__main__":
    main()
