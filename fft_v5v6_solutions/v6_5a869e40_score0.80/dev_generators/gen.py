#!/usr/bin/env python3
"""Generator for implementation.c: specialized batched 3D FFT + nonlinear map."""
import numpy as np
NL = chr(10)

LDPI = np.longdouble('3.14159265358979323846264338327950288')

def clit(x):
    x = float(x)
    if x == int(x) and abs(x) < 1e15:
        return repr(x)
    return x.hex()

def cs(num, den):
    """cos,sin of -2*pi*num/den computed in long double, reduced mod den."""
    m = num % den
    ang = (-2*LDPI) * np.longdouble(m) / np.longdouble(den)
    return float(np.cos(ang)), float(np.sin(ang))

class W:
    def __init__(self, bits):
        self.bits = bits
        self.nc = bits // 128      # complex lanes
        self.nd = bits // 64       # doubles
        self.p = {512: '_mm512', 256: '_mm256', 128: '_mm'}[bits]
        self.V = {512: '__m512d', 256: '__m256d', 128: '__m128d'}[bits]
    def LD(self, ptr): return f'{self.p}_loadu_pd({ptr})'
    def ST(self, ptr, v): return f'{self.p}_storeu_pd({ptr}, {v});'
    def MASKZ_LD(self, mask, ptr): return f'{self.p}_maskz_loadu_pd({mask}, {ptr})'
    def MASK_ST(self, ptr, mask, v): return f'{self.p}_mask_storeu_pd({ptr}, {mask}, {v});'
    def ADD(self, a, b): return f'{self.p}_add_pd({a}, {b})'
    def SUB(self, a, b): return f'{self.p}_sub_pd({a}, {b})'
    def MUL(self, a, b): return f'{self.p}_mul_pd({a}, {b})'
    def MAX(self, a, b): return f'{self.p}_max_pd({a}, {b})'
    def FMA(self, a, b, c): return f'{self.p}_fmadd_pd({a}, {b}, {c})'
    def FMS(self, a, b, c): return f'{self.p}_fmsub_pd({a}, {b}, {c})'
    def FNMA(self, a, b, c): return f'{self.p}_fnmadd_pd({a}, {b}, {c})'
    def FMADDSUB(self, a, b, c): return f'{self.p}_fmaddsub_pd({a}, {b}, {c})'
    def SWAP(self, a):
        imm = {512: '0x55', 256: '0x5', 128: '0x1'}[self.bits]
        return f'{self.p}_permute_pd({a}, {imm})'
    def SET1(self, x): return f'{self.p}_set1_pd({x})'
    def RSQRT14(self, a): return f'{self.p}_rsqrt14_pd({a})'
    def RCP14(self, a): return f'{self.p}_rcp14_pd({a})'
    def ZERO(self): return f'{self.p}_setzero_pd()'

W512, W256, W128 = W(512), W(256), W(128)

class Em:
    _uid = 0
    def __init__(self, w, pref=None):
        if pref is None:
            Em._uid += 1
            pref = f't{Em._uid}_'
        self.w = w; self.lines = []; self.n = 0; self.pref = pref
        self._const = {}
        self._fconst = {}
    def v(self, expr):
        name = f'{self.pref}{self.n}'; self.n += 1
        self.lines.append(f'{self.w.V} {name} = {expr};')
        return name
    def vf(self, typ, expr):
        name = f'{self.pref}{self.n}'; self.n += 1
        self.lines.append(f'{typ} {name} = {expr};')
        return name
    def cf(self, typ, expr):
        key = expr
        if key not in self._fconst:
            self._fconst[key] = self.vf(typ, expr)
        return self._fconst[key]
    def raw(self, line):
        self.lines.append(line)
    def const(self, x):
        key = float(x)
        if key not in self._const:
            self._const[key] = self.v(self.w.SET1(clit(x)))
        return self._const[key]

# ---------------- DFT codelets (operate on lists of var names) ----------------

def cmul(em, x, num, den):
    """x * exp(-2πi num/den)"""
    w = em.w
    wr, wi = cs(num, den)
    s = em.v(w.SWAP(x))
    return em.v(w.FMADDSUB(x, em.const(wr), w.MUL(s, em.const(wi))))

def dft2(em, xs):
    w = em.w; a, b = xs
    return [em.v(w.ADD(a, b)), em.v(w.SUB(a, b))]

def dft3(em, xs):
    w = em.w; x0, x1, x2 = xs
    s3 = 0.8660254037844386467637231707529362  # sin(60deg); refine below
    s3 = float(np.sin(LDPI/np.longdouble(3)))
    t1 = em.v(w.ADD(x1, x2))
    t2 = em.v(w.SUB(x1, x2))
    X0 = em.v(w.ADD(x0, t1))
    m1 = em.v(w.FNMA(em.const(0.5), t1, x0))
    u = em.v(w.MUL(w.SWAP(t2), em.const(s3)))
    X2 = em.v(w.FMADDSUB(m1, em.const(1.0), u))   # m1 + i*s3*t2
    X1 = em.v(w.FMS(em.const(2.0), m1, X2))       # m1 - i*s3*t2
    return [X0, X1, X2]

def dft4(em, xs):
    w = em.w; x0, x1, x2, x3 = xs
    t0 = em.v(w.ADD(x0, x2)); t1 = em.v(w.SUB(x0, x2))
    t2 = em.v(w.ADD(x1, x3)); t3 = em.v(w.SUB(x1, x3))
    X0 = em.v(w.ADD(t0, t2)); X2 = em.v(w.SUB(t0, t2))
    s = em.v(w.SWAP(t3))
    X3 = em.v(w.FMADDSUB(t1, em.const(1.0), s))   # t1 + i*t3
    X1 = em.v(w.FMS(em.const(2.0), t1, X3))       # t1 - i*t3
    return [X0, X1, X2, X3]

def dft5(em, xs):
    w = em.w; x0, x1, x2, x3, x4 = xs
    c1, s1 = cs(1, 5); c2, s2 = cs(2, 5)   # cos/sin of -2pi/5, -4pi/5 (s negative)
    # use positive sines: sin(2pi/5) = -s1
    sp1, sp2 = -s1, -s2
    t1 = em.v(w.ADD(x1, x4)); t2 = em.v(w.SUB(x1, x4))
    t3 = em.v(w.ADD(x2, x3)); t4 = em.v(w.SUB(x2, x3))
    X0 = em.v(w.ADD(x0, w.ADD(t1, t3)))
    m1 = em.v(w.FMA(em.const(c1), t1, w.FMA(em.const(c2), t3, x0)))
    m2 = em.v(w.FMA(em.const(c2), t1, w.FMA(em.const(c1), t3, x0)))
    sw2 = em.v(w.SWAP(t2)); sw4 = em.v(w.SWAP(t4))
    S1 = em.v(w.FMA(em.const(sp2), sw4, w.MUL(em.const(sp1), sw2)))
    S2 = em.v(w.FNMA(em.const(sp1), sw4, w.MUL(em.const(sp2), sw2)))
    X4 = em.v(w.FMADDSUB(m1, em.const(1.0), S1))  # m1 + i*s1true
    X1 = em.v(w.FMS(em.const(2.0), m1, X4))
    X3 = em.v(w.FMADDSUB(m2, em.const(1.0), S2))
    X2 = em.v(w.FMS(em.const(2.0), m2, X3))
    return [X0, X1, X2, X3, X4]

def dft8(em, xs):
    w = em.w
    E = dft4(em, [xs[0], xs[2], xs[4], xs[6]])
    O = dft4(em, [xs[1], xs[3], xs[5], xs[7]])
    r2 = float(np.sqrt(np.longdouble(2))/2)
    X = [None]*8
    X[0] = em.v(w.ADD(E[0], O[0])); X[4] = em.v(w.SUB(E[0], O[0]))
    # w8^2 = -i : X2 = E2 - i O2, X6 = E2 + i O2
    q2 = em.v(w.SWAP(O[2]))
    X[6] = em.v(w.FMADDSUB(E[2], em.const(1.0), q2))
    X[2] = em.v(w.FMS(em.const(2.0), E[2], X[6]))
    # w8^1 O1 = r2*((O1r+O1i) + i(O1i-O1r)) ; d = (O1r+O1i, O1i-O1r)
    sw1 = em.v(w.SWAP(O[1]))
    # d = ADDSUB(O1, -sw1): even O1r+O1i, odd O1i-O1r -> FMADDSUB(O1, 1, -sw1)? need neg.
    nsw1 = em.v(w.SUB(em.const(0.0), sw1))
    d = em.v(w.FMADDSUB(O[1], em.const(1.0), nsw1))
    X[1] = em.v(w.FMA(em.const(r2), d, E[1]))
    X[5] = em.v(w.FNMA(em.const(r2), d, E[1]))
    # w8^3 O3 = -r2*((O3r-O3i) + i(O3i+O3r)); g = ADDSUB(O3, SWAP(O3))
    sw3 = em.v(w.SWAP(O[3]))
    g = em.v(w.FMADDSUB(O[3], em.const(1.0), sw3))
    X[3] = em.v(w.FNMA(em.const(r2), g, E[3]))
    X[7] = em.v(w.FMA(em.const(r2), g, E[3]))
    return X

def dft9(em, xs):
    w = em.w
    A = [dft3(em, [xs[b], xs[3+b], xs[6+b]]) for b in range(3)]   # A[b][d]
    T = [[None]*3 for _ in range(3)]
    for b in range(3):
        for d in range(3):
            if b == 0 or d == 0:
                T[b][d] = A[b][d]
            else:
                T[b][d] = cmul(em, A[b][d], b*d, 9)
    X = [None]*9
    for d in range(3):
        Z = dft3(em, [T[0][d], T[1][d], T[2][d]])
        for c in range(3):
            X[3*c+d] = Z[c]
    return X

def dft64(em, xs):
    w = em.w
    A = [dft8(em, [xs[8*a+b] for a in range(8)]) for b in range(8)]   # A[b][d]
    T = [[None]*8 for _ in range(8)]
    for b in range(8):
        for d in range(8):
            if b == 0 or d == 0:
                T[b][d] = A[b][d]
            else:
                T[b][d] = cmul(em, A[b][d], b*d, 64)
    X = [None]*64
    for d in range(8):
        Z = dft8(em, [T[b][d] for b in range(8)])
        for c in range(8):
            X[8*c+d] = Z[c]
    return X

DFT_BASE = {2: dft2, 3: dft3, 4: dft4, 5: dft5, 8: dft8, 9: dft9}

def dft_pfa(L, N1, N2):
    def f(em, xs):
        # y[u][v] = x[(N2*u + N1*v) % L]
        R = []
        for u in range(N1):
            row = [xs[(N2*u + N1*v) % L] for v in range(N2)]
            R.append(DFT_BASE[N2](em, row))      # R[u][k2]
        X = [None]*L
        # CRT output index
        for k2 in range(N2):
            col = DFT_BASE[N1](em, [R[u][k2] for u in range(N1)])
            for k1 in range(N1):
                # k with k%N1==k1, k%N2==k2
                for k in range(L):
                    if k % N1 == k1 and k % N2 == k2:
                        X[k] = col[k1]; break
        return X
    return f

def prime_sym(L):
    h = (L-1)//2
    def f(em, xs):
        w = em.w
        x0 = xs[0]
        E = [None]*(h+1); Os = [None]*(h+1)
        for j in range(1, h+1):
            E[j] = em.v(w.ADD(xs[j], xs[L-j]))
            od = em.v(w.SUB(xs[j], xs[L-j]))
            Os[j] = em.v(w.SWAP(od))
        acc = x0
        for j in range(1, h+1):
            acc = em.v(w.ADD(acc, E[j]))
        X = [None]*L
        X[0] = acc
        one = em.const(1.0); two = em.const(2.0)
        for k in range(1, h+1):
            C = x0; S = None
            for j in range(1, h+1):
                m = (j*k) % L
                cc = float(np.cos(2*LDPI*np.longdouble(m)/np.longdouble(L)))
                ss = float(np.sin(2*LDPI*np.longdouble(m)/np.longdouble(L)))
                C = em.v(w.FMA(E[j], em.const(cc), C))
                if S is None:
                    S = em.v(w.MUL(Os[j], em.const(ss)))
                else:
                    S = em.v(w.FMA(Os[j], em.const(ss), S))
            XLk = em.v(w.FMADDSUB(C, one, S))       # C + i*S_true
            Xk = em.v(w.FMS(two, C, XLk))           # C - i*S_true
            X[k] = Xk; X[L-k] = XLk
        return X
    return f

DFT = {
    6:  dft_pfa(6, 2, 3),
    8:  dft8,
    13: prime_sym(13),
    17: prime_sym(17),
    23: prime_sym(23),
    36: dft_pfa(36, 4, 9),
    45: dft_pfa(45, 9, 5),
    64: dft64,
}


# ---------------- map emission ----------------

def fl(w):
    """float helpers per width: (ftype, cvt_down, cvt_up, sqrtps, rsqrtps, divps, set1ps)"""
    if w.bits == 512:
        return ('__m256', '_mm512_cvtpd_ps', '_mm512_cvtps_pd', '_mm256_sqrt_ps', '_mm256_rsqrt_ps', '_mm256_div_ps', '_mm256_set1_ps')
    if w.bits == 256:
        return ('__m128', '_mm256_cvtpd_ps', '_mm256_cvtps_pd', '_mm_sqrt_ps', '_mm_rsqrt_ps', '_mm_div_ps', '_mm_set1_ps')
    return ('__m128', '_mm_cvtpd_ps', '_mm_cvtps_pd', '_mm_sqrt_ps', '_mm_rsqrt_ps', '_mm_div_ps', '_mm_set1_ps')

import os as _os
MAPV = _os.environ.get('MAPV', 'rsq')
MAPH = _os.environ.get('MAPH', '0') == '1'

def emit_inv1p_sqrt(em, mmvar):
    if MAPV == 'rsq':
        return emit_inv1p_sqrt_rsq(em, mmvar)
    return emit_inv1p_sqrt_float(em, mmvar)

def emit_inv1p_sqrt_rsq(em, mmvar):
    w = em.w
    half = em.const(0.5); c15 = em.const(1.5); one = em.const(1.0)
    r = em.v(w.RSQRT14(mmvar))
    hm = em.v(w.MUL(mmvar, half))
    for _ in range(2):
        p = em.v(w.MUL(r, r))
        e = em.v(w.FNMA(hm, p, c15))
        r = em.v(w.MUL(r, e))
    s = em.v(w.MUL(mmvar, r))
    if MAPH:
        q = em.v(w.FNMA(s, s, mmvar))
        hr = em.v(w.MUL(r, half))
        s = em.v(w.FMA(q, hr, s))
    d = em.v(w.ADD(one, s))
    rc = em.v(w.RCP14(d))
    for _ in range(2):
        e = em.v(w.FNMA(d, rc, one))
        rc = em.v(w.FMA(rc, e, rc))
    return rc

def emit_inv1p_sqrt_float(em, mmvar):
    """returns var holding 1/(1+sqrt(mm)) to ~1-2 ulp; mm must be >= ~1e-36."""
    w = em.w
    half = em.const(0.5); c15 = em.const(1.5); one = em.const(1.0)
    FT, CVTD, CVTU, SQRTPS, RSQRTPS, DIVPS, SET1PS = fl(w)
    mf = em.vf(FT, f'{CVTD}({mmvar})')
    sf = em.vf(FT, f'{SQRTPS}({mf})')
    rf = em.vf(FT, f'{RSQRTPS}({mf})')
    s0 = em.v(f'{CVTU}({sf})')
    r0 = em.v(f'{CVTU}({rf})')
    p = em.v(w.MUL(r0, r0))
    hm = em.v(w.MUL(mmvar, half))
    e = em.v(w.FNMA(hm, p, c15))
    r1 = em.v(w.MUL(r0, e))
    hr = em.v(w.MUL(r1, half))
    q = em.v(w.FNMA(s0, s0, mmvar))
    s1 = em.v(w.FMA(q, hr, s0))
    q2 = em.v(w.FNMA(s1, s1, mmvar))
    s2 = em.v(w.FMA(q2, hr, s1))
    d = em.v(w.ADD(one, s2))
    df = em.vf(FT, f'{CVTD}({d})')
    onef = em.cf(FT, f'{SET1PS}(1.0f)')
    wf = em.vf(FT, f'{DIVPS}({onef}, {df})')
    w0 = em.v(f'{CVTU}({wf})')
    e1 = em.v(w.FNMA(d, w0, one))
    w1 = em.v(w.FMA(w0, e1, w0))
    e2 = em.v(w.FNMA(d, w1, one))
    w2 = em.v(w.FMA(w1, e2, w1))
    return w2

def emit_map_store(em, zvar, out_ptr, mask=None):
    w = em.w
    tiny = em.const(1e-36)
    t = em.v(w.MUL(zvar, zvar))
    m = em.v(w.ADD(t, w.SWAP(t)))
    mm = em.v(w.MAX(m, tiny))
    w2 = emit_inv1p_sqrt(em, mm)
    res = em.v(w.MUL(zvar, w2))
    if mask:
        em.raw(w.MASK_ST(out_ptr, mask, res))
    else:
        em.raw(w.ST(out_ptr, res))

def dup_imm(w, odd):
    return {512: ('0x00','0xFF'), 256: ('0x0','0xF'), 128: ('0x0','0x3')}[w.bits][odd]

def emit_map_pair(em, z0, z1, p0, p1, mask=None):
    w = em.w
    tiny = em.const(1e-36)
    t0 = em.v(w.MUL(z0, z0)); t1 = em.v(w.MUL(z1, z1))
    lo = em.v(f'{w.p}_unpacklo_pd({t0}, {t1})')
    hi = em.v(f'{w.p}_unpackhi_pd({t0}, {t1})')
    m = em.v(w.ADD(lo, hi))
    mm = em.v(w.MAX(m, tiny))
    w2 = emit_inv1p_sqrt(em, mm)
    w0 = em.v(f'{w.p}_permute_pd({w2}, {dup_imm(w,0)})')
    w1 = em.v(f'{w.p}_permute_pd({w2}, {dup_imm(w,1)})')
    r0 = em.v(w.MUL(z0, w0)); r1 = em.v(w.MUL(z1, w1))
    if mask:
        em.raw(w.MASK_ST(p0, mask, r0)); em.raw(w.MASK_ST(p1, mask, r1))
    else:
        em.raw(w.ST(p0, r0)); em.raw(w.ST(p1, r1))

def emit_map_multi(em, pairs, mask=None):
    """pairs: list of (z0, z1, ptr0, ptr1). Emits all chains stage-interleaved."""
    w = em.w
    n = len(pairs)
    tiny = em.const(1e-36); half = em.const(0.5); c15 = em.const(1.5); one = em.const(1.0)
    t0 = [em.v(w.MUL(p[0], p[0])) for p in pairs]
    t1 = [em.v(w.MUL(p[1], p[1])) for p in pairs]
    lo = [em.v(f'{w.p}_unpacklo_pd({t0[i]}, {t1[i]})') for i in range(n)]
    hi = [em.v(f'{w.p}_unpackhi_pd({t0[i]}, {t1[i]})') for i in range(n)]
    m = [em.v(w.ADD(lo[i], hi[i])) for i in range(n)]
    mm = [em.v(w.MAX(m[i], tiny)) for i in range(n)]
    r = [em.v(w.RSQRT14(mm[i])) for i in range(n)]
    hm = [em.v(w.MUL(mm[i], half)) for i in range(n)]
    for _ in range(2):
        p_ = [em.v(w.MUL(r[i], r[i])) for i in range(n)]
        e_ = [em.v(w.FNMA(hm[i], p_[i], c15)) for i in range(n)]
        r = [em.v(w.MUL(r[i], e_[i])) for i in range(n)]
    s = [em.v(w.MUL(mm[i], r[i])) for i in range(n)]
    if MAPH:
        q = [em.v(w.FNMA(s[i], s[i], mm[i])) for i in range(n)]
        hr = [em.v(w.MUL(r[i], half)) for i in range(n)]
        s = [em.v(w.FMA(q[i], hr[i], s[i])) for i in range(n)]
    d = [em.v(w.ADD(one, s[i])) for i in range(n)]
    rc = [em.v(w.RCP14(d[i])) for i in range(n)]
    for _ in range(2):
        e_ = [em.v(w.FNMA(d[i], rc[i], one)) for i in range(n)]
        rc = [em.v(w.FMA(rc[i], e_[i], rc[i])) for i in range(n)]
    for i in range(n):
        z0, z1, p0, p1 = pairs[i]
        w0 = em.v(f'{w.p}_permute_pd({rc[i]}, {dup_imm(w,0)})')
        w1 = em.v(f'{w.p}_permute_pd({rc[i]}, {dup_imm(w,1)})')
        r0 = em.v(w.MUL(z0, w0)); r1 = em.v(w.MUL(z1, w1))
        if mask:
            em.raw(w.MASK_ST(p0, mask, r0)); em.raw(w.MASK_ST(p1, mask, r1))
        else:
            em.raw(w.ST(p0, r0)); em.raw(w.ST(p1, r1))

# ---------------- pass emitters ----------------

def col_groups(L, maxlanes=4):
    """yield (start, lanes, width, mask_or_None) covering 0..L-1 lanes along contiguous dim"""
    out = []
    k = 0
    while L - k >= 4:
        out.append((k, 4, W512, None)); k += 4
    r = L - k
    if r == 3:
        out.append((k, 3, W512, '0x3F'))
    elif r == 2:
        out.append((k, 2, W256, None))
    elif r == 1:
        out.append((k, 1, W128, None))
    return out

def emit_line_pass(L, estride, fn_name, with_map, lines_outer, base_expr, cbase_expr=None, src_expr=None):
    """Transform along axis with element stride estride (complex units), lanes along a2."""
    body = []
    body.append(f'void {fn_name}(double* restrict x, const double* restrict cc, const double* restrict src){{')
    body.append(f'  (void)cc; (void)src;')
    body.append(f'  for(long u=0; u<{lines_outer}; u++){{')
    body.append(f'    double* restrict p = {base_expr};')
    if src_expr:
        body.append(f'    const double* restrict sp = {src_expr};')
    if cbase_expr:
        body.append(f'    const double* restrict cp = {cbase_expr};')
    nfull = L // 4
    rem = L % 4
    rd = 'sp' if src_expr else 'p'

    def group_block(w, mask, qdecl):
        em = Em(w)
        xs = []
        for j in range(L):
            off = 2*(j*estride)
            if mask:
                xs.append(em.v(w.MASKZ_LD(mask, f'q + {off}')))
            else:
                xs.append(em.v(w.LD(f'q + {off}')))
        X = DFT[L](em, xs)
        if with_map:
            def ldc(k):
                off = 2*(k*estride)
                if mask:
                    return em.v(w.MASKZ_LD(mask, f'cq + {off}'))
                return em.v(w.LD(f'cq + {off}'))
            k = 0
            while k + 1 < L:
                z0 = em.v(w.ADD(X[k], ldc(k)))
                z1 = em.v(w.ADD(X[k+1], ldc(k+1)))
                emit_map_pair(em, z0, z1, f'qo + {2*(k*estride)}', f'qo + {2*((k+1)*estride)}', mask)
                k += 2
            if k < L:
                z = em.v(w.ADD(X[k], ldc(k)))
                emit_map_store(em, z, f'qo + {2*(k*estride)}', mask)
        else:
            for k in range(L):
                off = 2*(k*estride)
                if mask:
                    em.raw(w.MASK_ST(f'qo + {off}', mask, X[k]))
                else:
                    em.raw(w.ST(f'qo + {off}', X[k]))
        lines = [qdecl]
        if with_map:
            pass
        lines += em.lines
        return lines

    # full-width groups: C loop
    if nfull > 0:
        body.append(f'    for(long k=0;k<{2*4*nfull};k+=8){{')
        body.append(f'      const double* restrict q = {rd} + k;')
        body.append(f'      double* restrict qo = p + k;')
        if with_map:
            body.append(f'      const double* restrict cq = cp + k;')
        for ln in group_block(W512, None, '')[1:]:
            body.append('      ' + ln)
        body.append('    }')
    # tail
    if rem:
        k0 = 2*4*nfull
        w, mask = {3: (W512, '0x3F'), 2: (W256, None), 1: (W128, None)}[rem]
        body.append('    {')
        body.append(f'      const double* restrict q = {rd} + {k0};')
        body.append(f'      double* restrict qo = p + {k0};')
        if with_map:
            body.append(f'      const double* restrict cq = cp + {k0};')
        for ln in group_block(w, mask, '')[1:]:
            body.append('      ' + ln)
        body.append('    }')
    body.append('  }')
    body.append('}')
    return NL.join(body)

def emit_map_store(em, zvar, out_ptr, mask=None):
    w = em.w
    half = em.const(0.5); c15 = em.const(1.5); one = em.const(1.0)
    tiny = em.const(1e-300)
    t = em.v(w.MUL(zvar, zvar))
    m = em.v(w.ADD(t, w.SWAP(t)))
    mm = em.v(w.MAX(m, tiny))
    r = em.v(w.RSQRT14(mm))
    hm = em.v(w.MUL(mm, half))
    for _ in range(2):
        p = em.v(w.MUL(r, r))
        e = em.v(w.FNMA(hm, p, c15))
        r = em.v(w.MUL(r, e))
    s = em.v(w.MUL(mm, r))
    q = em.v(w.FNMA(s, s, mm))
    hr = em.v(w.MUL(r, half))
    s = em.v(w.FMA(q, hr, s))
    d = em.v(w.ADD(one, s))
    rc = em.v(w.RCP14(d))
    for _ in range(2):
        e = em.v(w.FNMA(d, rc, one))
        rc = em.v(w.FMA(rc, e, rc))
    res = em.v(w.MUL(zvar, rc))
    if mask:
        em.raw(w.MASK_ST(out_ptr, mask, res))
    else:
        em.raw(w.ST(out_ptr, res))

def emit_pass2(L, fn_name):
    """Transform along a2 (contiguous elements), lanes across a1 rows, per a0 plane."""
    body = []
    body.append(f'void {fn_name}(double* restrict x){{')
    body.append(f'  for(long u=0; u<{L}; u++){{')
    body.append(f'    double* restrict p = x + u*{2*L*L};')
    ct = L % 4                      # column tail size
    ncb = L // 4                    # full col blocks
    ngfull = L // 4                 # full row groups
    rtail = L % 4

    def zmm_group(nrows):
        em = Em(W512)
        w = W512
        v = [None]*L
        rows = [f'q + {2*(t*L)}' for t in range(nrows)]
        def fwd_transpose(t):
            x0 = em.v(f'_mm512_shuffle_f64x2({t[0]}, {t[1]}, 0x88)')
            x1 = em.v(f'_mm512_shuffle_f64x2({t[0]}, {t[1]}, 0xDD)')
            x2 = em.v(f'_mm512_shuffle_f64x2({t[2]}, {t[3]}, 0x88)')
            x3 = em.v(f'_mm512_shuffle_f64x2({t[2]}, {t[3]}, 0xDD)')
            return [em.v(f'_mm512_shuffle_f64x2({x0}, {x2}, 0x88)'),
                    em.v(f'_mm512_shuffle_f64x2({x1}, {x3}, 0x88)'),
                    em.v(f'_mm512_shuffle_f64x2({x0}, {x2}, 0xDD)'),
                    em.v(f'_mm512_shuffle_f64x2({x1}, {x3}, 0xDD)')]
        for cb in range(ncb + (1 if ct else 0)):
            k0 = cb*4
            tail = (cb == ncb)
            mask = {1: '0x03', 2: '0x0F', 3: '0x3F'}.get(ct) if tail else None
            t = []
            for tr in range(4):
                if tr < nrows:
                    if tail:
                        t.append(em.v(w.MASKZ_LD(mask, f'{rows[tr]} + {2*k0}')))
                    else:
                        t.append(em.v(w.LD(f'{rows[tr]} + {2*k0}')))
                else:
                    t.append(em.v(w.ZERO()))
            o = fwd_transpose(t)
            lim = ct if tail else 4
            for kk in range(lim):
                v[k0+kk] = o[kk]
        X = DFT[L](em, v)
        for cb in range(ncb + (1 if ct else 0)):
            k0 = cb*4
            tail = (cb == ncb)
            mask = {1: '0x03', 2: '0x0F', 3: '0x3F'}.get(ct) if tail else None
            lim = ct if tail else 4
            src4 = [X[k0+kk] if kk < lim else X[k0] for kk in range(4)]
            o = fwd_transpose(src4)
            for tr in range(nrows):
                if tail:
                    em.raw(w.MASK_ST(f'{rows[tr]} + {2*k0}', mask, o[tr]))
                else:
                    em.raw(w.ST(f'{rows[tr]} + {2*k0}', o[tr]))
        return em.lines

    def ymm_group():
        em = Em(W256); w = W256
        v = [None]*L
        rows = [f'q + {2*(t*L)}' for t in range(2)]
        nc2 = L // 2
        ct2 = L % 2
        for cb in range(nc2):
            k0 = cb*2
            t0 = em.v(w.LD(f'{rows[0]} + {2*k0}'))
            t1 = em.v(w.LD(f'{rows[1]} + {2*k0}'))
            v[k0]   = em.v(f'_mm256_permute2f128_pd({t0}, {t1}, 0x20)')
            v[k0+1] = em.v(f'_mm256_permute2f128_pd({t0}, {t1}, 0x31)')
        if ct2:
            k0 = nc2*2
            a = em.v(f'_mm_loadu_pd({rows[0]} + {2*k0})')
            b = em.v(f'_mm_loadu_pd({rows[1]} + {2*k0})')
            v[k0] = em.v(f'_mm256_insertf128_pd(_mm256_castpd128_pd256({a}), {b}, 1)')
        X = DFT[L](em, v)
        for cb in range(nc2):
            k0 = cb*2
            o0 = em.v(f'_mm256_permute2f128_pd({X[k0]}, {X[k0+1]}, 0x20)')
            o1 = em.v(f'_mm256_permute2f128_pd({X[k0]}, {X[k0+1]}, 0x31)')
            em.raw(w.ST(f'{rows[0]} + {2*k0}', o0))
            em.raw(w.ST(f'{rows[1]} + {2*k0}', o1))
        if ct2:
            k0 = nc2*2
            em.raw(f'_mm_storeu_pd({rows[0]} + {2*k0}, _mm256_castpd256_pd128({X[k0]}));')
            em.raw(f'_mm_storeu_pd({rows[1]} + {2*k0}, _mm256_extractf128_pd({X[k0]}, 1));')
        return em.lines

    def xmm_group():
        em = Em(W128); w = W128
        v = []
        for k in range(L):
            v.append(em.v(w.LD(f'q + {2*k}')))
        X = DFT[L](em, v)
        for k in range(L):
            em.raw(w.ST(f'q + {2*k}', X[k]))
        return em.lines

    if ngfull > 0:
        body.append(f'    for(long g=0;g<{ngfull};g++){{')
        body.append(f'      double* restrict q = p + g*{2*4*L};')
        for ln in zmm_group(4):
            body.append('      ' + ln)
        body.append('    }')
    if rtail:
        body.append('    {')
        body.append(f'      double* restrict q = p + {2*ngfull*4*L};')
        if rtail == 3:
            for ln in zmm_group(3):
                body.append('      ' + ln)
        elif rtail == 2:
            for ln in ymm_group():
                body.append('      ' + ln)
        else:
            for ln in xmm_group():
                body.append('      ' + ln)
        body.append('    }')
    body.append('  }')
    body.append('}')
    return NL.join(body)

# ---------------- top-level assembly ----------------

def gen_size(L):
    parts = []
    parts.append(emit_line_pass(L, L, f'pass1_{L}', False, L, f'x + u*{2*L*L}'))
    parts.append(emit_pass2(L, f'pass2_{L}'))
    parts.append(emit_line_pass(L, L*L, f'pass3m_{L}', True, L, f'x + u*{2*L}', f'cc + u*{2*L}'))
    parts.append(emit_line_pass(L, L*L, f'pass3p_{L}', False, L, f'x + u*{2*L}'))
    V = 2*L*L*L
    parts.append(f'''
void fft3_{L}(double* restrict x){{
  pass1_{L}(x, 0, 0); pass2_{L}(x); pass3p_{L}(x, 0, 0);
}}
void step_{L}(double* restrict x, const double* restrict c){{
  pass1_{L}(x, 0, 0); pass2_{L}(x); pass3m_{L}(x, c, 0);
}}
void run_{L}(long B, long m, const double* restrict x0, const double* restrict c,
             double* restrict out1, double* restrict outm){{
  for(long b=0;b<B;b++){{
    memcpy(STATE, x0 + b*{V}, {V}*sizeof(double));
    memcpy(CBUF, c + b*{V}, {V}*sizeof(double));
    for(long s=0;s<m;s++){{
      pass1_{L}(STATE, 0, 0); pass2_{L}(STATE); pass3m_{L}(STATE, CBUF, 0);
      if(s==0) memcpy(out1 + b*{V}, STATE, {V}*sizeof(double));
    }}
    memcpy(outm + b*{V}, STATE, {V}*sizeof(double));
  }}
}}''')
    return '\n\n'.join(parts)

def generate(sizes=(6, 8, 13, 17, 23, 36, 45, 64), fname='implementation.c'):
    out = []
    out.append('// auto-generated by gen.py -- specialized 3D FFT + map')
    out.append('#include <immintrin.h>')
    out.append('#include <string.h>')
    out.append('#include <sys/mman.h>')
    out.append('''
static double* STATE = 0;
static double* CBUF = 0;
#define HSZ (8u<<20)
static void* halloc(unsigned long sz){
  void* p = mmap(0, sz + (2u<<20), PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
  if(p == MAP_FAILED) return 0;
  unsigned long a = ((unsigned long)p + (2u<<20) - 1) & ~((2ul<<20)-1ul);
  madvise((void*)a, sz, MADV_HUGEPAGE);
  return (void*)a;
}
void init_mem(void){
  if(STATE) return;
  STATE = (double*)halloc(2*64*64*64*8ul);
  CBUF  = (double*)halloc(2*64*64*64*8ul);
  for(long i=0;i<2*64*64*64;i+=512){ STATE[i]=0.0; CBUF[i]=0.0; }
}
''')
    for L in sizes:
        out.append(gen_size(L))
    with open(fname, 'w') as f:
        f.write('\n\n'.join(out) + '\n')

if __name__ == '__main__':
    import sys
    sizes = tuple(int(a) for a in sys.argv[1:]) or (6, 8, 13, 17, 23, 36, 45, 64)
    generate(sizes)
    print('generated', sizes)
