import numpy as np
from math import cos, sin, pi, gcd

SIZES = (6, 8, 13, 17, 23, 36, 45, 64)
def ceil8(L): return ((L + 7) // 8) * 8
GROUP_SIZES = (6, 17)
def S_of(L): return ceil8(L)
def PX_of(L): return S_of(L)*S_of(L) + 8

def hexf(x):
    return float(x).hex()

class G:
    """Emitter with parallel numeric evaluation on 8 test lanes."""
    def __init__(self, rng):
        self.body = []
        self.cnt = 0
        self.vals = {}
        self.rng = rng
    def capture(self):
        mark = len(self.body)
        return mark
    def take(self, mark):
        blk = self.body[mark:]
        del self.body[mark:]
        return blk
    def zip_blocks(self, blocks):
        # interleave lines of independent SSA blocks round-robin
        idx = [0]*len(blocks)
        out = []
        while True:
            done = True
            for bi, b in enumerate(blocks):
                if idx[bi] < len(b):
                    out.append(b[idx[bi]]); idx[bi] += 1; done = False
            if done: break
        self.body.extend(out)
    def nt(self):
        self.cnt += 1
        return "v%d" % self.cnt
    def raw(self, s):
        self.body.append(s)
    def ld(self, addr, val):
        v = self.nt()
        self.body.append("V %s = LD(%s);" % (v, addr))
        self.vals[v] = np.array(val, dtype=np.float64)
        return v
    def st(self, addr, x):
        self.body.append("ST(%s, %s);" % (addr, x))
    # arithmetic
    def add(self, a, b):
        v = self.nt(); self.body.append("V %s = %s + %s;" % (v, a, b))
        self.vals[v] = self.vals[a] + self.vals[b]; return v
    def sub(self, a, b):
        v = self.nt(); self.body.append("V %s = %s - %s;" % (v, a, b))
        self.vals[v] = self.vals[a] - self.vals[b]; return v
    def neg(self, a):
        v = self.nt(); self.body.append("V %s = -%s;" % (v, a))
        self.vals[v] = -self.vals[a]; return v
    def mulc(self, c, a):   # c * a, c scalar const
        v = self.nt(); self.body.append("V %s = C(%s) * %s;" % (v, hexf(c), a))
        self.vals[v] = float(c) * self.vals[a]; return v
    def fmac(self, c, a, b):   # c*a + b
        v = self.nt(); self.body.append("V %s = C(%s) * %s + %s;" % (v, hexf(c), a, b))
        self.vals[v] = float(c) * self.vals[a] + self.vals[b]; return v
    def fmsc(self, c, a, b):   # b - c*a
        v = self.nt(); self.body.append("V %s = %s - C(%s) * %s;" % (v, b, hexf(c), a))
        self.vals[v] = self.vals[b] - float(c) * self.vals[a]; return v

# ---------------- complex helpers: values are (re_name, im_name) --------------
def cadd(g, x, y): return (g.add(x[0], y[0]), g.add(x[1], y[1]))
def csub(g, x, y): return (g.sub(x[0], y[0]), g.sub(x[1], y[1]))
def cneg(g, x): return (g.neg(x[0]), g.neg(x[1]))

def cmulw(g, w, x):
    """multiply by complex constant w"""
    wr, wi = float(np.real(w)), float(np.imag(w))
    TOL = 1e-15
    if abs(wi) < TOL:  # real
        if abs(wr-1) < TOL: return x
        if abs(wr+1) < TOL: return cneg(g, x)
        return (g.mulc(wr, x[0]), g.mulc(wr, x[1]))
    if abs(wr) < TOL:  # pure imaginary: w = i*wi: (r,i) -> (-wi*i, wi*r)
        if abs(wi+1) < TOL:  # w = -i: (r,i)->(i,-r)
            return (x[1], g.neg(x[0]))
        if abs(wi-1) < TOL:  # w = i: (r,i)->(-i,r)
            return (g.neg(x[1]), x[0])
        return (g.mulc(-wi, x[1]), g.mulc(wi, x[0]))
    # general: rr = wr*r - wi*i ; ii = wr*i + wi*r
    t1 = g.mulc(wr, x[0]); rr = g.fmsc(wi, x[1], t1)
    t2 = g.mulc(wr, x[1]); ii = g.fmac(wi, x[0], t2)
    return (rr, ii)

def w_of(N, k):
    ang = -2.0 * pi * (k % N) / N
    return complex(cos(ang), sin(ang))

# ---------------- DFT builders ----------------
def dft(g, xs):
    N = len(xs)
    if N == 1: return xs
    if N == 2:
        return [cadd(g, xs[0], xs[1]), csub(g, xs[0], xs[1])]
    if N == 4:
        t0 = cadd(g, xs[0], xs[2]); t1 = csub(g, xs[0], xs[2])
        t2 = cadd(g, xs[1], xs[3]); t3 = csub(g, xs[1], xs[3])
        X0 = cadd(g, t0, t2); X2 = csub(g, t0, t2)
        # X1 = t1 - i t3 ; X3 = t1 + i t3;  -i*t3 = (t3i, -t3r)
        X1 = (g.add(t1[0], t3[1]), g.sub(t1[1], t3[0]))
        X3 = (g.sub(t1[0], t3[1]), g.add(t1[1], t3[0]))
        return [X0, X1, X2, X3]
    if N in (3, 5, 11, 13, 17, 23):
        return dft_symdirect(g, xs)
    if N == 8: return dft_ct(g, xs, 2, 4)
    if N == 9: return dft_ct(g, xs, 3, 3)
    if N == 6: return dft_pfa(g, xs, 2, 3)
    if N == 36: return dft_pfa(g, xs, 4, 9)
    if N == 45: return dft_pfa(g, xs, 9, 5)
    if N == 64: return dft_ct(g, xs, 8, 8)
    raise ValueError(N)

def dft_symdirect(g, xs):
    N = len(xs); h = (N - 1) // 2
    u = {}; v = {}
    for j in range(1, h + 1):
        u[j] = cadd(g, xs[j], xs[N - j])
        v[j] = csub(g, xs[j], xs[N - j])
    # X0 = x0 + sum u  (balanced tree)
    terms = [xs[0]] + [u[j] for j in range(1, h + 1)]
    while len(terms) > 1:
        nx = []
        for i in range(0, len(terms) - 1, 2):
            nx.append(cadd(g, terms[i], terms[i + 1]))
        if len(terms) % 2: nx.append(terms[-1])
        terms = nx
    out = [None] * N
    out[0] = terms[0]
    import os as _os
    _zdef = '3' if N == 23 else '4'
    ZIPP = int(_os.environ.get('ZIPP', _zdef))
    blocks = []
    for k in range(1, h + 1):
        mk = g.capture()
        Ar, Ai = xs[0]
        Br, Bi = None, None
        for j in range(1, h + 1):
            c = cos(2 * pi * ((j * k) % N) / N)
            s = sin(2 * pi * ((j * k) % N) / N)
            Ar = g.fmac(c, u[j][0], Ar)
            Ai = g.fmac(c, u[j][1], Ai)
            if Br is None:
                Br = g.mulc(s, v[j][0]); Bi = g.mulc(s, v[j][1])
            else:
                Br = g.fmac(s, v[j][0], Br); Bi = g.fmac(s, v[j][1], Bi)
        # X[k] = A - i B -> (Ar + Bi, Ai - Br) ; X[N-k] = (Ar - Bi, Ai + Br)
        out[k] = (g.add(Ar, Bi), g.sub(Ai, Br))
        out[N - k] = (g.sub(Ar, Bi), g.add(Ai, Br))
        if ZIPP:
            blocks.append(g.take(mk))
            if len(blocks) == ZIPP:
                g.zip_blocks(blocks); blocks = []
    if ZIPP and blocks: g.zip_blocks(blocks)
    return out

def dft_ct(g, xs, N1, N2):
    """N = N1*N2, DIT: n = n1 + N1*n2 ; X[k2 + N2*k1]"""
    import os as _os
    ZT = int(_os.environ.get('ZT', '0'))
    N = N1 * N2
    Y = []
    z1 = (ZT > 0 and N2 <= ZT)
    z2 = (ZT > 0 and N1 <= ZT)
    blocks = []
    for n1 in range(N1):
        mk = g.capture()
        sub = dft(g, [xs[n1 + N1 * n2] for n2 in range(N2)])
        Y.append([cmulw(g, w_of(N, n1 * k2), sub[k2]) for k2 in range(N2)])
        if z1:
            blocks.append(g.take(mk))
            if len(blocks) == 2: g.zip_blocks(blocks); blocks = []
    if blocks: g.zip_blocks(blocks); blocks = []
    out = [None] * N
    for k2 in range(N2):
        mk = g.capture()
        sub = dft(g, [Y[n1][k2] for n1 in range(N1)])
        for k1 in range(N1):
            out[k2 + N2 * k1] = sub[k1]
        if z2:
            blocks.append(g.take(mk))
            if len(blocks) == 2: g.zip_blocks(blocks); blocks = []
    if blocks: g.zip_blocks(blocks)
    return out

def dft_pfa(g, xs, N1, N2):
    assert gcd(N1, N2) == 1
    N = N1 * N2
    # input map: n(n1,n2) = (N2*n1 + N1*n2) mod N
    # output:  X2[k1][k2] = X[k], k = CRT(inv(N2)%N1 * k1, inv(N1)%N2 * k2)
    i21 = pow(N2, -1, N1); i12 = pow(N1, -1, N2)
    import os as _os
    ZT = int(_os.environ.get('ZT', '0'))
    z1 = (ZT > 0 and N2 <= ZT)
    z2 = (ZT > 0 and N1 <= ZT)
    Y = []
    blocks = []
    for n1 in range(N1):
        mk = g.capture()
        sub = dft(g, [xs[(N2 * n1 + N1 * n2) % N] for n2 in range(N2)])
        Y.append(sub)
        if z1:
            blocks.append(g.take(mk))
            if len(blocks) == 2: g.zip_blocks(blocks); blocks = []
    if blocks: g.zip_blocks(blocks); blocks = []
    out = [None] * N
    for k2 in range(N2):
        mk = g.capture()
        if z2 and k2 == 0: pass
        sub = dft(g, [Y[n1][k2] for n1 in range(N1)])
        for k1 in range(N1):
            # X2[k1][k2] = X[k] with k = k1 (mod N1), k = k2 (mod N2)
            k = (k1 * N2 * i21 + k2 * N1 * i12) % N
            assert k % N1 == k1 and k % N2 == k2
            out[k] = sub[k1]
        if z2:
            blocks.append(g.take(mk))
            if len(blocks) == 2: g.zip_blocks(blocks); blocks = []
    if blocks: g.zip_blocks(blocks)
    assert all(o is not None for o in out)
    return out

# ---------------- verification of builders ----------------
def verify_builders():
    for N in SIZES + (3, 5, 9, 4, 2):
        rng = np.random.default_rng(1234 + N)
        g = G(rng)
        xr = rng.standard_normal((N, 8)); xi = rng.standard_normal((N, 8))
        xs = []
        for j in range(N):
            xs.append((g.ld("in+%d" % j, xr[j]), g.ld("im+%d" % j, xi[j])))
        out = dft(g, xs)
        res = np.array([g.vals[o[0]] + 1j * g.vals[o[1]] for o in out])
        ref = np.fft.fft(xr + 1j * xi, axis=0)
        err = np.abs(res - ref).max() / np.abs(ref).max()
        nops = sum(1 for l in g.body if "=" in l and not l.startswith("ST"))
        print("N=%2d ops=%4d err=%.2e %s" % (N, nops, err, "OK" if err < 1e-12 else "FAIL"))
        assert err < 1e-12, N


# =====================================================================
#  C file emission
# =====================================================================

def emit_header():
    return r'''
// AUTO-GENERATED - specialized iterated batched 3D FFT kernels (AVX-512)
#include <immintrin.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/mman.h>

typedef __m512d V;
#define LD(p)    _mm512_load_pd(p)
#define LDU(p)   _mm512_loadu_pd(p)
#define ST(p,x)  _mm512_store_pd((p),(x))
#define STU(p,x) _mm512_storeu_pd((p),(x))
#define C(x)     _mm512_set1_pd(x)
#define ZV       _mm512_setzero_pd()
#define NOIN     __attribute__((noinline))
#ifndef PIPE
#define PIPE 0
#endif
#ifndef P1ORD
#define P1ORD 0
#endif

#define TR8(r0,r1,r2,r3,r4,r5,r6,r7) do{ \
  V _t0=_mm512_unpacklo_pd(r0,r1), _t1=_mm512_unpackhi_pd(r0,r1); \
  V _t2=_mm512_unpacklo_pd(r2,r3), _t3=_mm512_unpackhi_pd(r2,r3); \
  V _t4=_mm512_unpacklo_pd(r4,r5), _t5=_mm512_unpackhi_pd(r4,r5); \
  V _t6=_mm512_unpacklo_pd(r6,r7), _t7=_mm512_unpackhi_pd(r6,r7); \
  V _u0=_mm512_shuffle_f64x2(_t0,_t2,0x88), _u1=_mm512_shuffle_f64x2(_t1,_t3,0x88); \
  V _u2=_mm512_shuffle_f64x2(_t0,_t2,0xDD), _u3=_mm512_shuffle_f64x2(_t1,_t3,0xDD); \
  V _u4=_mm512_shuffle_f64x2(_t4,_t6,0x88), _u5=_mm512_shuffle_f64x2(_t5,_t7,0x88); \
  V _u6=_mm512_shuffle_f64x2(_t4,_t6,0xDD), _u7=_mm512_shuffle_f64x2(_t5,_t7,0xDD); \
  r0=_mm512_shuffle_f64x2(_u0,_u4,0x88); r4=_mm512_shuffle_f64x2(_u0,_u4,0xDD); \
  r1=_mm512_shuffle_f64x2(_u1,_u5,0x88); r5=_mm512_shuffle_f64x2(_u1,_u5,0xDD); \
  r2=_mm512_shuffle_f64x2(_u2,_u6,0x88); r6=_mm512_shuffle_f64x2(_u2,_u6,0xDD); \
  r3=_mm512_shuffle_f64x2(_u3,_u7,0x88); r7=_mm512_shuffle_f64x2(_u3,_u7,0xDD); \
}while(0)

static const __m512i IDE = {1,3,5,7,9,11,13,15}; /* placeholder, set in setup */
'''

def gen_loads(g, L, addr_fmt, rng):
    xs = []
    xr = rng.standard_normal((L, 8)); xi = rng.standard_normal((L, 8))
    for j in range(L):
        xs.append((g.ld(addr_fmt("re", j), xr[j]), g.ld(addr_fmt("im", j), xi[j])))
    return xs, xr + 1j * xi

def check(g, out, ref):
    res = np.array([g.vals[o[0]] + 1j * g.vals[o[1]] for o in out])
    err = np.abs(res - np.fft.fft(ref, axis=0)).max() / max(1e-30, np.abs(res).max())
    assert err < 1e-12, err


def _creation_order(out):
    def key(k):
        n = out[k][0]
        try: return int(n[1:])
        except Exception: return 10**9
    return sorted(range(len(out)), key=key)

def emit_codelet(L, role):
    S = S_of(L); PX = PX_of(L)
    rng = np.random.default_rng(999 + L * 10 + (1 if role=='p0' else 2))
    g = G(rng)
    pre = []
    if role == 'p0':
        fmt = lambda arr, j: "%s + %d" % (arr, 8 * j)
        sig = "static void NOIN p0_%d(const double*restrict re, const double*restrict im, double*restrict ore, double*restrict oim)" % L
    elif role == 'p1':
        fmt = lambda arr, j: "%s + %d" % (arr, j * PX)
        sig = "static void NOIN p1_%d(double*restrict re, double*restrict im)" % L
    elif role == 'pb1':
        fmt = lambda arr, j: "%s + %d" % (arr, j * 8 * L * L)
        sig = "static void NOIN pb1_%d(double*restrict re, double*restrict im)" % L
    elif role == 'pb2':
        fmt = lambda arr, j: "%s + %d" % (arr, j * 8 * L)
        sig = "static void NOIN pb2_%d(double*restrict re, double*restrict im)" % L
    elif role == 'pb3':
        fmt = lambda arr, j: "%s + %d" % (arr, j * 8)
        sig = "static void NOIN pb3_%d(double*restrict re, double*restrict im, const double*restrict cpk)" % L
    elif role == 'p1x':
        sig = "static void NOIN p1x_%d(const double*restrict src, double*restrict re, double*restrict im, __mmask8 m0, __mmask8 m1)" % L
    elif role == 'p2':
        fmt = lambda arr, j: "%s + %d" % (arr, j * S)
        sig = "static void NOIN p2_%d(const double*restrict re, const double*restrict im, double*restrict ore, double*restrict oim)" % L
    elif role in ('p3', 'p3d', 'p3c', 'p3eT', 'p3eN', 'p3f'):
        fmt = lambda arr, j: "%s + %d" % (arr, j * S)
        if role == 'p3':
            sig = "static void NOIN p3_%d(const double*restrict re, const double*restrict im, double*restrict ore, double*restrict oim, const double*restrict cpk)" % L
        elif role == 'p3f':
            sig = "static void NOIN p3f_%d(const double*restrict re, const double*restrict im, double*restrict ore, double*restrict oim, const double*restrict cpk, double*restrict eo1, int x, int ab)" % L
        elif role == 'p3eT':
            sig = "static void NOIN p3eT_%d(const double*restrict re, const double*restrict im, const double*restrict csrc, int nrow, double*restrict eo1, double*restrict eo2, int x, int ab)" % L
        elif role == 'p3eN':
            sig = "static void NOIN p3eN_%d(const double*restrict re, const double*restrict im, const double*restrict csrc, __mmask8 m0, __mmask8 m1, double*restrict eo1, int x, int ab)" % L
        elif role == 'p3d':
            sig = "static void NOIN p3d_%d(const double*restrict re, const double*restrict im, double*restrict ore, double*restrict oim, const double*restrict csrc, __mmask8 m0, __mmask8 m1)" % L
        else:
            sig = "static void NOIN p3c_%d(const double*restrict re, const double*restrict im, double*restrict ore, double*restrict oim, const double*restrict csrc, int nrow)" % L
    # prefetches
    PF1 = lambda expr: '_mm_prefetch((const char*)(%s), _MM_HINT_T1);' % expr
    PF0 = lambda expr: '_mm_prefetch((const char*)(%s), _MM_HINT_T0);' % expr
    import os as _os
    P1D = int(_os.environ.get('P1D', '8'))
    P1H = _os.environ.get('P1H', 'T0')
    PFX = lambda expr, h: '_mm_prefetch((const char*)(%s), _MM_HINT_%s);' % (expr, h)
    if role == 'p1':
        pd = S if _os.environ.get('P1ORD','0') == '1' else P1D
        for j in range(L):
            g.raw(PFX('re + %d' % (j * PX + pd), P1H))
            g.raw(PFX('im + %d' % (j * PX + pd), P1H))
    elif role == 'pb1':
        for j in range(L):
            g.raw(PF1('re + %d' % (j * 8 * L * L + 8)))
            g.raw(PF1('im + %d' % (j * 8 * L * L + 8)))
    elif role == 'pb2':
        for j in range(L):
            g.raw(PF1('re + %d' % (j * 8 * L + 8)))
            g.raw(PF1('im + %d' % (j * 8 * L + 8)))
    elif role == 'pb3':
        for j in range(L):
            g.raw(PF1('cpk + %d' % (j * 16 + 16 * L)))
    elif role == 'p1x':
        for j in range(L):
            g.raw(PF1('src + %d' % (j * 2*L*L + 16)))
    elif role == 'p2':
        P2H = _os.environ.get('P2H', 'T0')
        for j in range(L):
            g.raw(PFX('re + %d' % (j * S + 8), P2H))
            g.raw(PFX('im + %d' % (j * S + 8), P2H))
    elif role in ('p3', 'p3f'):
        P3H = _os.environ.get('P3H', 'T0')
        for j in range(L):
            g.raw(PFX('cpk + %d' % (j * 16 + 16 * L), P3H))
    elif role == 'p3eN':
        for j in range(L):
            g.raw(PF1('csrc + %d' % (j * 2*L + 16)))
    elif role == 'p3d':
        for j in range(L):
            g.raw(PF1('csrc + %d' % (j * 2*L + 16)))
    # c-tile materialization for p3c: build cbuf on stack
    if role in ('p3c', 'p3eT'):
        g.raw("V cbr[%d], cbi[%d];" % (ceil8(L), ceil8(L)))
        g.raw("const __m512i ide = _mm512_set_epi64(14,12,10,8,6,4,2,0);")
        g.raw("const __m512i ido = _mm512_set_epi64(15,13,11,9,7,5,3,1);")
        tail = L & 7
        bits = (1 << (2*(tail if tail else 8))) - 1
        for jg in range(0, ceil8(L), 8):
            last = (jg + 8 > L)
            mm0 = (bits & 0xFF) if last and tail else 0xFF
            mm1 = (bits >> 8) if last and tail else 0xFF
            g.raw("{")
            g.raw("V tr[8], ti[8];")
            g.raw("for (int l = 0; l < 8; l++) {")
            g.raw("  if (l < nrow) {")
            g.raw("    V a = _mm512_maskz_loadu_pd(0x%02x, csrc + (long)l*%d + %d);" % (mm0, 2*L*1, 2*jg))
            g.raw("    V b = _mm512_maskz_loadu_pd(0x%02x, csrc + (long)l*%d + %d);" % (mm1, 2*L*1, 2*jg+8))
            g.raw("    tr[l] = _mm512_permutex2var_pd(a, ide, b);")
            g.raw("    ti[l] = _mm512_permutex2var_pd(a, ido, b);")
            g.raw("  } else { tr[l] = ZV; ti[l] = ZV; }")
            g.raw("}")
            g.raw("TR8(tr[0],tr[1],tr[2],tr[3],tr[4],tr[5],tr[6],tr[7]);")
            g.raw("TR8(ti[0],ti[1],ti[2],ti[3],ti[4],ti[5],ti[6],ti[7]);")
            for l in range(8):
                if jg + l < L:
                    g.raw("cbr[%d] = tr[%d]; cbi[%d] = ti[%d];" % (jg+l, l, jg+l, l))
            g.raw("}")
    # loads
    if role == 'p1x':
        g.raw("const __m512i ide = _mm512_set_epi64(14,12,10,8,6,4,2,0);")
        g.raw("const __m512i ido = _mm512_set_epi64(15,13,11,9,7,5,3,1);")
        xs = []
        xr = rng.standard_normal((L, 8)); xi = rng.standard_normal((L, 8))
        for j in range(L):
            a = g.nt(); b = g.nt()
            g.raw("V %s = _mm512_maskz_loadu_pd(m0, src + %d);" % (a, j*2*L*L))
            g.raw("V %s = _mm512_maskz_loadu_pd(m1, src + %d);" % (b, j*2*L*L+8))
            rr = g.nt(); iiv = g.nt()
            g.raw("V %s = _mm512_permutex2var_pd(%s, ide, %s);" % (rr, a, b))
            g.raw("V %s = _mm512_permutex2var_pd(%s, ido, %s);" % (iiv, a, b))
            g.vals[rr] = xr[j]; g.vals[iiv] = xi[j]
            xs.append((rr, iiv))
        ref = xr + 1j * xi
    else:
        xs, ref = gen_loads(g, L, fmt, rng)
    out = dft(g, xs)
    check(g, out, ref)
    # stores
    if role == 'p0':
        for k in range(L):
            g.st("ore + %d" % (8 * k), out[k][0]); g.st("oim + %d" % (8 * k), out[k][1])
    elif role == 'p1':
        _ord = _creation_order(out) if _os.environ.get('CORD','0') == '1' else range(L)
        for k in _ord:
            g.st("re + %d" % (k * PX), out[k][0]); g.st("im + %d" % (k * PX), out[k][1])
    elif role == 'p1x':
        for k in range(L):
            g.st("re + %d" % (k * PX), out[k][0]); g.st("im + %d" % (k * PX), out[k][1])
    elif role == 'p2':
        if True:
            g.raw("V obr[%d], obi[%d];" % (L, L))
            _ord = _creation_order(out) if _os.environ.get('CORD','0') == '1' else range(L)
            for k in _ord:
                g.raw("obr[%d] = %s; obi[%d] = %s;" % (k, out[k][0], k, out[k][1]))
            for gg in range(S // 8):
                for half in range(2):
                    arr = "ore" if half == 0 else "oim"
                    buf = "obr" if half == 0 else "obi"
                    qs = []
                    for l in range(8):
                        k = 8 * gg + l
                        v = g.nt()
                        if k < L:
                            g.raw("V %s = %s[%d];" % (v, buf, k))
                        else:
                            g.raw("V %s = ZV;" % v)
                        qs.append(v)
                    g.raw("TR8(%s);" % ",".join(qs))
                    for l in range(8):
                        g.st("%s + %d" % (arr, l * S + 8 * gg), qs[l])
        else:
            for gg in range(S // 8):
                for half in range(2):
                    arr = "ore" if half == 0 else "oim"
                    qs = []
                    for l in range(8):
                        k = 8 * gg + l
                        v = g.nt()
                        if k < L:
                            g.raw("V %s = %s;" % (v, out[k][half]))
                        else:
                            g.raw("V %s = ZV;" % v)
                        qs.append(v)
                    g.raw("TR8(%s);" % ",".join(qs))
                    for l in range(8):
                        g.st("%s + %d" % (arr, l * S + 8 * gg), qs[l])
    elif role in ('pb1', 'pb2'):
        st = 8 * (L * L if role == 'pb1' else L)
        _ord = _creation_order(out) if _os.environ.get('CORD','0') == '1' else range(L)
        for k in _ord:
            g.st("re + %d" % (k * st), out[k][0])
            g.st("im + %d" % (k * st), out[k][1])
    else:  # p3 family (incl pb3): add c, map, store
        _ord = _creation_order(out) if _os.environ.get('CORD','0') == '1' else range(L)
        if _os.environ.get('OBUF3') and role in ('p3', 'pb3'):
            g.raw("V odr[%d], odi[%d];" % (L, L))
            for k in _creation_order(out):
                g.raw("odr[%d] = %s; odi[%d] = %s;" % (k, out[k][0], k, out[k][1]))
            out = [("odr[%d]" % k, "odi[%d]" % k) for k in range(L)]
            for k in range(L):
                g.vals[out[k][0]] = np.zeros(8); g.vals[out[k][1]] = np.zeros(8)
        for k in _ord:
            if role in ('p3', 'p3f', 'pb3'):
                cr = g.ld("cpk + %d" % (k * 16), np.zeros(8))
                ci = g.ld("cpk + %d" % (k * 16 + 8), np.zeros(8))
            elif role == 'p3eT':
                cr = g.nt(); ci = g.nt()
                g.raw("V %s = cbr[%d]; V %s = cbi[%d];" % (cr, k, ci, k))
                g.vals[cr] = np.zeros(8); g.vals[ci] = np.zeros(8)
            elif role in ('p3d', 'p3eN'):
                a = g.nt(); b = g.nt()
                g.raw("V %s = _mm512_maskz_loadu_pd(m0, csrc + %d);" % (a, k*2*L))
                g.raw("V %s = _mm512_maskz_loadu_pd(m1, csrc + %d);" % (b, k*2*L+8))
                cr = g.nt(); ci = g.nt()
                g.raw("V %s = _mm512_permutex2var_pd(%s, ide2, %s);" % (cr, a, b))
                g.raw("V %s = _mm512_permutex2var_pd(%s, ido2, %s);" % (ci, a, b))
                g.vals[cr] = np.zeros(8); g.vals[ci] = np.zeros(8)
            else:  # p3c
                cr = g.nt(); ci = g.nt()
                g.raw("V %s = cbr[%d]; V %s = cbi[%d];" % (cr, k, ci, k))
                g.vals[cr] = np.zeros(8); g.vals[ci] = np.zeros(8)
            zr = g.add(out[k][0], cr)
            zi = g.add(out[k][1], ci)
            import os as _os2
            MAP = 'mix' if L >= 36 else _os2.environ.get('SMALLMAP', 'newton')

            t = g.nt(); g.raw("V %s = %s*%s + %s*%s;" % (t, zr, zr, zi, zi))
            a2 = g.nt(); g.raw("V %s = _mm512_max_pd(%s, C(1e-300));" % (a2, t))
            if MAP == 'hyb':
                mg = g.nt(); g.raw("V %s = _mm512_sqrt_pd(%s);" % (mg, a2))
                d = g.nt(); g.raw("V %s = C(1.0) + %s;" % (d, mg))
                r = g.nt(); g.raw("V %s = _mm512_rcp14_pd(%s);" % (r, d))
                r2 = g.nt(); g.raw("V %s = %s*(C(2.0) - %s*%s);" % (r2, r, d, r))
                r3 = g.nt(); g.raw("V %s = %s*(C(2.0) - %s*%s);" % (r3, r2, d, r2))
            elif MAP == 'hw' or (MAP == 'mix' and (k % 2 == 0)):
                mg = g.nt(); g.raw("V %s = _mm512_sqrt_pd(%s);" % (mg, a2))
                d = g.nt(); g.raw("V %s = C(1.0) + %s;" % (d, mg))
                r3 = g.nt(); g.raw("V %s = _mm512_div_pd(C(1.0), %s);" % (r3, d))
            else:  # newton: rsqrt14 + 2NR, then rcp14 + 2NR
                u = g.nt(); g.raw("V %s = _mm512_rsqrt14_pd(%s);" % (u, a2))
                th = g.nt(); g.raw("V %s = C(0.5) * %s;" % (th, a2))
                v1 = g.nt(); g.raw("V %s = %s*%s;" % (v1, u, u))
                w1 = g.nt(); g.raw("V %s = C(1.5) - %s*%s;" % (w1, th, v1))
                u2 = g.nt(); g.raw("V %s = %s*%s;" % (u2, u, w1))
                v2 = g.nt(); g.raw("V %s = %s*%s;" % (v2, u2, u2))
                w2 = g.nt(); g.raw("V %s = C(1.5) - %s*%s;" % (w2, th, v2))
                u3 = g.nt(); g.raw("V %s = %s*%s;" % (u3, u2, w2))
                mg = g.nt(); g.raw("V %s = %s*%s;" % (mg, a2, u3))
                d = g.nt(); g.raw("V %s = C(1.0) + %s;" % (d, mg))
                r = g.nt(); g.raw("V %s = _mm512_rcp14_pd(%s);" % (r, d))
                r2 = g.nt(); g.raw("V %s = %s*(C(2.0) - %s*%s);" % (r2, r, d, r))
                r3 = g.nt(); g.raw("V %s = %s*(C(2.0) - %s*%s);" % (r3, r2, d, r2))
            if role in ('p3eT', 'p3eN'):
                g.raw("mr[%d] = %s*%s; mi[%d] = %s*%s;" % (k, zr, r3, k, zi, r3))
            elif role == 'p3f':
                g.raw("mr[%d] = %s*%s; mi[%d] = %s*%s;" % (k, zr, r3, k, zi, r3))
                g.raw("ST(ore + %d, mr[%d]); ST(oim + %d, mi[%d]);" % (k * S, k, k * S, k))
            elif role == 'pb3':
                g.raw("ST(re + %d, %s*%s);" % (k * 8, zr, r3))
                g.raw("ST(im + %d, %s*%s);" % (k * 8, zi, r3))
            else:
                g.raw("ST(ore + %d, %s*%s);" % (k * S, zr, r3))
                g.raw("ST(oim + %d, %s*%s);" % (k * S, zi, r3))
    if role in ('p3eT', 'p3f'):
        EO2 = (role == 'p3eT')
        tail = L & 7
        bits = (1 << (2*(tail if tail else 8))) - 1
        g.raw("(void)0;")
        ep = []
        ep.append("const __m512i ilo = _mm512_set_epi64(11,3,10,2,9,1,8,0);")
        ep.append("const __m512i ihi = _mm512_set_epi64(15,7,14,6,13,5,12,4);")
        for gg in range(ceil8(L) // 8):
            last = (8*gg + 8 > L)
            qs_r = []; qs_i = []
            for l in range(8):
                k = 8*gg + l
                vr = "eqr%d_%d" % (gg, l); vi = "eqi%d_%d" % (gg, l)
                if k < L:
                    ep.append("V %s = mr[%d]; V %s = mi[%d];" % (vr, k, vi, k))
                else:
                    ep.append("V %s = ZV; V %s = ZV;" % (vr, vi))
                qs_r.append(vr); qs_i.append(vi)
            ep.append("TR8(%s);" % ",".join(qs_r))
            ep.append("TR8(%s);" % ",".join(qs_i))
            mm0 = (bits & 0xFF) if last and tail else 0xFF
            mm1 = (bits >> 8) if last and tail else 0xFF
            ep.append("for (int l = 0; l < 8; l++) {")
            ep.append("  int y = ab + l; if (y >= %d) break;" % L)
            ep.append("  long off = 2*(((long)x*%d + y)*%d + %d);" % (L, L, 8*gg))
            ep.append("  V lo, hi;")
            ep.append("  switch(l){")
            for l in range(8):
                ep.append("  case %d: lo = _mm512_permutex2var_pd(%s, ilo, %s); hi = _mm512_permutex2var_pd(%s, ihi, %s); break;" % (l, qs_r[l], qs_i[l], qs_r[l], qs_i[l]))
            ep.append("  }")
            if not last or not tail:
                if L >= 36:
                    ep.append("  if ((((uintptr_t)(eo1+off)) & 63) == 0) { _mm512_stream_pd(eo1+off, lo); _mm512_stream_pd(eo1+off+8, hi); }")
                    ep.append("  else { STU(eo1+off, lo); STU(eo1+off+8, hi); }")
                else:
                    ep.append("  STU(eo1+off, lo); STU(eo1+off+8, hi);")
                if EO2: ep.append("  if (eo2) { STU(eo2+off, lo); STU(eo2+off+8, hi); }")
            else:
                ep.append("  _mm512_mask_storeu_pd(eo1+off, 0x%02x, lo); _mm512_mask_storeu_pd(eo1+off+8, 0x%02x, hi);" % (mm0, mm1))
                if EO2: ep.append("  if (eo2) { _mm512_mask_storeu_pd(eo2+off, 0x%02x, lo); _mm512_mask_storeu_pd(eo2+off+8, 0x%02x, hi); }" % (mm0, mm1))
            ep.append("}")
        g.body.extend(ep)
    elif role == 'p3eN':
        tail = L & 7
        bits = (1 << (2*(tail if tail else 8))) - 1
        ep = []
        ep.append("const __m512i ilo = _mm512_set_epi64(11,3,10,2,9,1,8,0);")
        ep.append("const __m512i ihi = _mm512_set_epi64(15,7,14,6,13,5,12,4);")
        ep.append("int zt = (ab + 8 > %d);" % L)
        ep.append("for (int j = 0; j < %d; j++) {" % L)
        ep.append("  long off = 2*(((long)x*%d + j)*%d + ab);" % (L, L))
        ep.append("  V lo = _mm512_permutex2var_pd(mr[j], ilo, mi[j]);")
        ep.append("  V hi = _mm512_permutex2var_pd(mr[j], ihi, mi[j]);")
        ep.append("  if (!zt) {")
        if L >= 36:
            ep.append("    if ((((uintptr_t)(eo1+off)) & 63) == 0) { _mm512_stream_pd(eo1+off, lo); _mm512_stream_pd(eo1+off+8, hi); }")
            ep.append("    else { STU(eo1+off, lo); STU(eo1+off+8, hi); }")
        else:
            ep.append("    STU(eo1+off, lo); STU(eo1+off+8, hi);")
        ep.append("  } else {")
        ep.append("    _mm512_mask_storeu_pd(eo1+off, 0x%02x, lo); _mm512_mask_storeu_pd(eo1+off+8, 0x%02x, hi);" % (bits & 0xFF, (bits >> 8) if tail else 0xFF))
        ep.append("  }")
        ep.append("}")
        g.body.extend(ep)
    body = g.body
    if role in ('p3eT', 'p3eN', 'p3f'):
        body = ["V mr[%d], mi[%d];" % (L, L)] + body
    if role in ('p3d', 'p3eN'):
        body = ["const __m512i ide2 = _mm512_set_epi64(14,12,10,8,6,4,2,0);",
                "const __m512i ido2 = _mm512_set_epi64(15,13,11,9,7,5,3,1);"] + body
    lines = [sig + " {"] + ["  " + l for l in body] + ["}"]
    return "\n".join(lines)

def emit_size(L):
    S = S_of(L); PX = PX_of(L); VOL = L * L * L
    parts = []
    roles = ['p0', 'p1', 'p2', 'p3', 'p3d', 'p3c', 'p3eT', 'p3eN', 'p3f']
    if L in GROUP_SIZES: roles += ['pb1', 'pb2', 'pb3']
    for role in roles:
        parts.append(emit_codelet(L, role))
    NB = S // 8
    CPW = 16 * ceil8(L)
    tail = L & 7
    bits = (1 << (2*(tail if tail else 8))) - 1
    mm0 = (bits & 0xFF) if tail else 0xFF
    mm1 = (bits >> 8) if tail else 0xFF
    parts.append(f"""
static void pass1_{L}(double*restrict re, double*restrict im) {{
#if P1ORD
  for (int bb = 0; bb < {S}; bb += 8)
    for (int a = 0; a < {L}; a++)
      p1_{L}(re + a * {S} + bb, im + a * {S} + bb);
#else
  for (int a = 0; a < {L}; a++) {{
    double* r = re + a * {S};
    double* i2 = im + a * {S};
    for (int bb = 0; bb < {S}; bb += 8) p1_{L}(r + bb, i2 + bb);
  }}
#endif
}}
// kind: 0 -> p3 with split c planes; 1 -> p3d natural interleaved c; 2 -> p3c transposed interleaved c
static void pass23_{L}(double*restrict are, double*restrict aim,
                       const double*restrict cpk,
                       const double*restrict csrc, int kind) {{
  double* s0r = Sr_{L}; double* s0i = Si_{L};
  double* s1r = Sr2_{L}; double* s1i = Si2_{L};
  for (int x = 0; x < {L}; x++) {{
    double* ar = are + (long)x * {PX}; double* ai = aim + (long)x * {PX};
    double* pr = are + (long)(x-1) * {PX}; double* pi = aim + (long)(x-1) * {PX};
    for (int k = 0; k < {NB}; k++) {{
      int bb = 8 * k;
      p2_{L}(ar + bb, ai + bb, s0r + (long)bb * {S}, s0i + (long)bb * {S});
    }}
#if PIPE==0
    if (1) {{
      double* t = s0r; s0r = s1r; s1r = t;
      t = s0i; s0i = s1i; s1i = t;
      double* pr = ar; double* pi = ai;
      int x_ = x + 1;
      (void)x_;
      for (int k = 0; k < {NB}; k++) {{
        int bb = 8 * k;
        if (kind == 0)
          p3_{L}(s1r + bb, s1i + bb, pr + bb, pi + bb, cpk + ((long)x*{NB} + k)*{CPW});
        else if (kind == 1) {{
          __mmask8 m0 = 0xFF, m1 = 0xFF;
          if (bb + 8 > {L}) {{ m0 = 0x{mm0:02x}; m1 = 0x{mm1:02x}; }}
          p3d_{L}(s1r + bb, s1i + bb, pr + bb, pi + bb, csrc + 2*((long)x*{L}*{L} + bb), m0, m1);
        }} else {{
          int nrow = {L} - bb; if (nrow > 8) nrow = 8;
          p3c_{L}(s1r + bb, s1i + bb, pr + bb, pi + bb, csrc + 2*((long)x*{L}*{L} + (long)bb*{L}), nrow);
        }}
      }}
      // swap back so outer swap restores
      t = s0r; s0r = s1r; s1r = t;
      t = s0i; s0i = s1i; s1i = t;
    }}
#else
    for (int k = 0; k < {NB}; k++) {{
      int bb = 8 * k;
      if (x > 0) {{
        if (kind == 0)
          p3_{L}(s1r + bb, s1i + bb, pr + bb, pi + bb, cre + (long)(x-1)*{PX} + bb, cim + (long)(x-1)*{PX} + bb);
        else if (kind == 1) {{
          __mmask8 m0 = 0xFF, m1 = 0xFF;
          if (bb + 8 > {L}) {{ m0 = 0x{mm0:02x}; m1 = 0x{mm1:02x}; }}
          p3d_{L}(s1r + bb, s1i + bb, pr + bb, pi + bb, csrc + 2*((long)(x-1)*{L}*{L} + bb), m0, m1);
        }} else {{
          int nrow = {L} - bb; if (nrow > 8) nrow = 8;
          p3c_{L}(s1r + bb, s1i + bb, pr + bb, pi + bb, csrc + 2*((long)(x-1)*{L}*{L} + (long)bb*{L}), nrow);
        }}
      }}
    }}
#endif
    double* t;
    t = s0r; s0r = s1r; s1r = t;
    t = s0i; s0i = s1i; s1i = t;
  }}
#if PIPE==1
  {{
    double* pr = are + (long)({L}-1) * {PX}; double* pi = aim + (long)({L}-1) * {PX};
    for (int k = 0; k < {NB}; k++) {{
      int bb = 8 * k;
      if (kind == 0)
        p3_{L}(s1r + bb, s1i + bb, pr + bb, pi + bb, cpk + ((long)({L}-1)*{NB} + k)*{CPW});
      else if (kind == 1) {{
        __mmask8 m0 = 0xFF, m1 = 0xFF;
        if (bb + 8 > {L}) {{ m0 = 0x{mm0:02x}; m1 = 0x{mm1:02x}; }}
        p3d_{L}(s1r + bb, s1i + bb, pr + bb, pi + bb, csrc + 2*((long)({L}-1)*{L}*{L} + bb), m0, m1);
      }} else {{
        int nrow = {L} - bb; if (nrow > 8) nrow = 8;
        p3c_{L}(s1r + bb, s1i + bb, pr + bb, pi + bb, csrc + 2*((long)({L}-1)*{L}*{L} + (long)bb*{L}), nrow);
      }}
    }}
  }}
#endif
}}
// t==1 fused: p2 + p3f (state write + emitted out1), c packed
static void pass23f_{L}(double*restrict are, double*restrict aim,
                        const double*restrict cpk, double*restrict eo1) {{
  for (int x = 0; x < {L}; x++) {{
    double* ar = are + (long)x * {PX}; double* ai = aim + (long)x * {PX};
    for (int bb = 0; bb < {S}; bb += 8)
      p2_{L}(ar + bb, ai + bb, Sr_{L} + (long)bb * {S}, Si_{L} + (long)bb * {S});
    for (int k = 0; k < {NB}; k++) {{
      int bb = 8 * k;
      p3f_{L}(Sr_{L} + bb, Si_{L} + bb, ar + bb, ai + bb, cpk + ((long)x*{NB} + k)*{CPW}, eo1, x, bb);
    }}
  }}
}}
// final step: p2 into scratch, then emit directly (no state write). parity: 1 = transposed layout (odd t), 0 = natural
static void pass23e_{L}(double*restrict are, double*restrict aim,
                        const double*restrict csrc, double*restrict eo1, double*restrict eo2, int parity) {{
  for (int x = 0; x < {L}; x++) {{
    double* ar = are + (long)x * {PX}; double* ai = aim + (long)x * {PX};
    for (int bb = 0; bb < {S}; bb += 8)
      p2_{L}(ar + bb, ai + bb, Sr_{L} + (long)bb * {S}, Si_{L} + (long)bb * {S});
    for (int k = 0; k < {NB}; k++) {{
      int bb = 8 * k;
      if (parity) {{
        int nrow = {L} - bb; if (nrow > 8) nrow = 8;
        p3eT_{L}(Sr_{L} + bb, Si_{L} + bb, csrc + 2*((long)x*{L}*{L} + (long)bb*{L}), nrow, eo1, eo2, x, bb);
      }} else {{
        __mmask8 m0 = 0xFF, m1 = 0xFF;
        if (bb + 8 > {L}) {{ m0 = 0x{mm0:02x}; m1 = 0x{mm1:02x}; }}
        p3eN_{L}(Sr_{L} + bb, Si_{L} + bb, csrc + 2*((long)x*{L}*{L} + bb), m0, m1, eo1, x, bb);
      }}
    }}
  }}
}}
""")
    if L in GROUP_SIZES:
        parts.append(f"""
static void rung_{L}(const double* const* srcx, const double* const* srcc,
                     double* const* dst1, double* const* dstm, int nreal, long m) {{
  packB(srcx, Gr_{L}, Gi_{L}, {VOL});
  packBC(srcc, cG_{L}, {VOL});
  for (long t = 1; t <= m; t++) {{
    for (long i2 = 0; i2 < {L}*{L}; i2++) pb1_{L}(Gr_{L} + 8*i2, Gi_{L} + 8*i2);
    for (long x = 0; x < {L}; x++)
      for (long z = 0; z < {L}; z++) {{
        long e = x*{L}*{L} + z;
        pb2_{L}(Gr_{L} + 8*e, Gi_{L} + 8*e);
      }}
    for (long x = 0; x < {L}; x++)
      for (long y = 0; y < {L}; y++) {{
        long e = x*{L}*{L} + y*{L};
        pb3_{L}(Gr_{L} + 8*e, Gi_{L} + 8*e, cG_{L} + 16*e);
      }}
    if (t == 1) unpackB(Gr_{L}, Gi_{L}, dst1, nreal, {VOL});
  }}
  unpackB(Gr_{L}, Gi_{L}, dstm, nreal, {VOL});
}}
""")
    return "\n".join(parts)

GENERIC = r'''
// ---------------- generic (runtime-L) helpers: cold paths ----------------
static void deintl(const double*restrict src, double*restrict dre, double*restrict dim,
                   int L, int S, long PX) {
  const __m512i ide = _mm512_set_epi64(14,12,10,8,6,4,2,0);
  const __m512i ido = _mm512_set_epi64(15,13,11,9,7,5,3,1);
  int tail = L & 7;
  __mmask8 m0 = 0, m1 = 0;
  if (tail) { unsigned bits = (1u << (2*tail)) - 1u; m0 = (__mmask8)(bits & 0xFF); m1 = (__mmask8)(bits >> 8); }
  for (int x = 0; x < L; x++) for (int y = 0; y < L; y++) {
    const double* s = src + 2 * ((long)(x * L + y) * L);
    double* dr = dre + (long)x * PX + (long)y * S;
    double* di = dim + (long)x * PX + (long)y * S;
    int z = 0;
    for (; z + 8 <= L; z += 8) {
      V a = LDU(s + 2*z), b = LDU(s + 2*z + 8);
      ST(dr + z, _mm512_permutex2var_pd(a, ide, b));
      ST(di + z, _mm512_permutex2var_pd(a, ido, b));
    }
    if (tail) {
      V a = _mm512_maskz_loadu_pd(m0, s + 2*z);
      V b = _mm512_maskz_loadu_pd(m1, s + 2*z + 8);
      ST(dr + z, _mm512_permutex2var_pd(a, ide, b));
      ST(di + z, _mm512_permutex2var_pd(a, ido, b));
    }
  }
}

static void emitN(const double*restrict sre, const double*restrict sim,
                  double*restrict dst, double*restrict dst2,
                  int L, int S, long PX, int ntok) {
  const __m512i ilo = _mm512_set_epi64(11,3,10,2,9,1,8,0);
  const __m512i ihi = _mm512_set_epi64(15,7,14,6,13,5,12,4);
  int tail = L & 7;
  __mmask8 m0 = 0, m1 = 0;
  if (tail) { unsigned bits = (1u << (2*tail)) - 1u; m0 = (__mmask8)(bits & 0xFF); m1 = (__mmask8)(bits >> 8); }
  for (int x = 0; x < L; x++) for (int y = 0; y < L; y++) {
    const double* sr = sre + (long)x * PX + (long)y * S;
    const double* si = sim + (long)x * PX + (long)y * S;
    long off = 2 * ((long)(x * L + y) * L);
    double* d = dst + off;
    double* d2 = dst2 ? dst2 + off : 0;
    int nt = ntok && ((((uintptr_t)d) & 63) == 0) && (!d2 || ((((uintptr_t)d2) & 63) == 0));
    int z = 0;
    if (nt) for (; z + 8 <= L; z += 8) {
      V a = LD(sr + z), b = LD(si + z);
      V lo = _mm512_permutex2var_pd(a, ilo, b), hi = _mm512_permutex2var_pd(a, ihi, b);
      _mm512_stream_pd(d + 2*z, lo); _mm512_stream_pd(d + 2*z + 8, hi);
      if (d2) { _mm512_stream_pd(d2 + 2*z, lo); _mm512_stream_pd(d2 + 2*z + 8, hi); }
    }
    else for (; z + 8 <= L; z += 8) {
      V a = LD(sr + z), b = LD(si + z);
      V lo = _mm512_permutex2var_pd(a, ilo, b), hi = _mm512_permutex2var_pd(a, ihi, b);
      STU(d + 2*z, lo); STU(d + 2*z + 8, hi);
      if (d2) { STU(d2 + 2*z, lo); STU(d2 + 2*z + 8, hi); }
    }
    if (tail) {
      V a = LD(sr + z), b = LD(si + z);
      V lo = _mm512_permutex2var_pd(a, ilo, b), hi = _mm512_permutex2var_pd(a, ihi, b);
      _mm512_mask_storeu_pd(d + 2*z, m0, lo); _mm512_mask_storeu_pd(d + 2*z + 8, m1, hi);
      if (d2) { _mm512_mask_storeu_pd(d2 + 2*z, m0, lo); _mm512_mask_storeu_pd(d2 + 2*z + 8, m1, hi); }
    }
  }
}

// packed c preps: block for (x, k) holds, for j in [0,L): [re vec][im vec] at offset ((x*NB+k)*C8 + j)*16
// cprepTpk: vector lanes = y in [8k, 8k+8), index j = z   (for odd steps, layout [x][z][y])
static void cprepTpk(const double*restrict src, double*restrict pk, int L, int NB, int C8) {
  const __m512i ide = _mm512_set_epi64(14,12,10,8,6,4,2,0);
  const __m512i ido = _mm512_set_epi64(15,13,11,9,7,5,3,1);
  int tail = L & 7;
  __mmask8 m0 = 0, m1 = 0;
  if (tail) { unsigned bits = (1u << (2*tail)) - 1u; m0 = (__mmask8)(bits & 0xFF); m1 = (__mmask8)(bits >> 8); }
  for (int x = 0; x < L; x++)
    for (int k = 0; k < NB; k++) {
      int yb = 8 * k;
      double* dst = pk + ((long)x * NB + k) * (long)C8 * 16;
      for (int zb = 0; zb < L; zb += 8) {
        V rr[8], ii[8];
        int zt = (L - zb >= 8) ? 0 : 1;
        for (int l = 0; l < 8; l++) {
          int y = yb + l;
          if (y < L) {
            const double* s = src + 2 * ((long)(x * L + y) * L + zb);
            V a, b;
            if (!zt) { a = LDU(s); b = LDU(s + 8); }
            else { a = _mm512_maskz_loadu_pd(m0, s); b = _mm512_maskz_loadu_pd(m1, s + 8); }
            rr[l] = _mm512_permutex2var_pd(a, ide, b);
            ii[l] = _mm512_permutex2var_pd(a, ido, b);
          } else { rr[l] = ZV; ii[l] = ZV; }
        }
        TR8(rr[0],rr[1],rr[2],rr[3],rr[4],rr[5],rr[6],rr[7]);
        TR8(ii[0],ii[1],ii[2],ii[3],ii[4],ii[5],ii[6],ii[7]);
        for (int l = 0; l < 8; l++) {
          int z = zb + l;
          if (z < L) { ST(dst + (long)z*16, rr[l]); ST(dst + (long)z*16 + 8, ii[l]); }
        }
      }
    }
}

// cprepNpk: vector lanes = z in [8k, 8k+8), index j = y   (for even steps, layout [x][y][z])
static void cprepNpk(const double*restrict src, double*restrict pk, int L, int NB, int C8) {
  const __m512i ide = _mm512_set_epi64(14,12,10,8,6,4,2,0);
  const __m512i ido = _mm512_set_epi64(15,13,11,9,7,5,3,1);
  int tail = L & 7;
  __mmask8 mt0 = 0xFF, mt1 = 0xFF;
  if (tail) { unsigned bits = (1u << (2*tail)) - 1u; mt0 = (__mmask8)(bits & 0xFF); mt1 = (__mmask8)(bits >> 8); }
  for (int x = 0; x < L; x++)
    for (int k = 0; k < NB; k++) {
      int zb = 8 * k;
      int zt = (L - zb >= 8) ? 0 : 1;
      double* dst = pk + ((long)x * NB + k) * (long)C8 * 16;
      for (int j = 0; j < L; j++) {
        const double* s = src + 2 * ((long)(x * L + j) * L + zb);
        V a, b;
        if (!zt) { a = LDU(s); b = LDU(s + 8); }
        else { a = _mm512_maskz_loadu_pd(mt0, s); b = _mm512_maskz_loadu_pd(mt1, s + 8); }
        ST(dst + (long)j*16, _mm512_permutex2var_pd(a, ide, b));
        ST(dst + (long)j*16 + 8, _mm512_permutex2var_pd(a, ido, b));
      }
    }
}

// one sweep of interleaved c producing both packed layouts
static void cprepBoth(const double*restrict src, double*restrict pkT, double*restrict pkN, int L, int NB, int C8) {
  const __m512i ide = _mm512_set_epi64(14,12,10,8,6,4,2,0);
  const __m512i ido = _mm512_set_epi64(15,13,11,9,7,5,3,1);
  int tail = L & 7;
  __mmask8 m0 = 0, m1 = 0;
  if (tail) { unsigned bits = (1u << (2*tail)) - 1u; m0 = (__mmask8)(bits & 0xFF); m1 = (__mmask8)(bits >> 8); }
  for (int x = 0; x < L; x++)
    for (int k = 0; k < NB; k++) {
      int yb = 8 * k;
      double* dstT = pkT + ((long)x * NB + k) * (long)C8 * 16;
      for (int zb = 0; zb < L; zb += 8) {
        V rr[8], ii[8];
        int zt = (L - zb >= 8) ? 0 : 1;
        for (int l = 0; l < 8; l++) {
          int y = yb + l;
          if (y < L) {
            const double* s = src + 2 * ((long)(x * L + y) * L + zb);
            V a, b;
            if (!zt) { a = LDU(s); b = LDU(s + 8); }
            else { a = _mm512_maskz_loadu_pd(m0, s); b = _mm512_maskz_loadu_pd(m1, s + 8); }
            rr[l] = _mm512_permutex2var_pd(a, ide, b);
            ii[l] = _mm512_permutex2var_pd(a, ido, b);
            // natural packed: block (x, zb/8), slot j=y
            double* dN = pkN + (((long)x * NB + (zb >> 3)) * (long)C8 + y) * 16;
            ST(dN, rr[l]); ST(dN + 8, ii[l]);
          } else { rr[l] = ZV; ii[l] = ZV; }
        }
        TR8(rr[0],rr[1],rr[2],rr[3],rr[4],rr[5],rr[6],rr[7]);
        TR8(ii[0],ii[1],ii[2],ii[3],ii[4],ii[5],ii[6],ii[7]);
        for (int l = 0; l < 8; l++) {
          int z = zb + l;
          if (z < L) { ST(dstT + (long)z*16, rr[l]); ST(dstT + (long)z*16 + 8, ii[l]); }
        }
      }
    }
}

// ---- lane-batched (8 volumes in lanes) pack/unpack ----
static void packB(const double* const* srcs, double*restrict br, double*restrict bi, long VOL) {
  const __m512i ide = _mm512_set_epi64(14,12,10,8,6,4,2,0);
  const __m512i ido = _mm512_set_epi64(15,13,11,9,7,5,3,1);
  long i = 0;
  for (; i + 8 <= VOL; i += 8) {
    V rr[8], ii[8];
    for (int l = 0; l < 8; l++) {
      V a = LDU(srcs[l] + 2*i), b = LDU(srcs[l] + 2*i + 8);
      rr[l] = _mm512_permutex2var_pd(a, ide, b);
      ii[l] = _mm512_permutex2var_pd(a, ido, b);
    }
    TR8(rr[0],rr[1],rr[2],rr[3],rr[4],rr[5],rr[6],rr[7]);
    TR8(ii[0],ii[1],ii[2],ii[3],ii[4],ii[5],ii[6],ii[7]);
    for (int l2 = 0; l2 < 8; l2++) { ST(br + 8*(i+l2), rr[l2]); ST(bi + 8*(i+l2), ii[l2]); }
  }
  for (; i < VOL; i++)
    for (int l = 0; l < 8; l++) { br[8*i+l] = srcs[l][2*i]; bi[8*i+l] = srcs[l][2*i+1]; }
}
static void packBC(const double* const* srcs, double*restrict cg, long VOL) {
  const __m512i ide = _mm512_set_epi64(14,12,10,8,6,4,2,0);
  const __m512i ido = _mm512_set_epi64(15,13,11,9,7,5,3,1);
  long i = 0;
  for (; i + 8 <= VOL; i += 8) {
    V rr[8], ii[8];
    for (int l = 0; l < 8; l++) {
      V a = LDU(srcs[l] + 2*i), b = LDU(srcs[l] + 2*i + 8);
      rr[l] = _mm512_permutex2var_pd(a, ide, b);
      ii[l] = _mm512_permutex2var_pd(a, ido, b);
    }
    TR8(rr[0],rr[1],rr[2],rr[3],rr[4],rr[5],rr[6],rr[7]);
    TR8(ii[0],ii[1],ii[2],ii[3],ii[4],ii[5],ii[6],ii[7]);
    for (int l2 = 0; l2 < 8; l2++) { ST(cg + 16*(i+l2), rr[l2]); ST(cg + 16*(i+l2) + 8, ii[l2]); }
  }
  for (; i < VOL; i++)
    for (int l = 0; l < 8; l++) { cg[16*i+l] = srcs[l][2*i]; cg[16*i+8+l] = srcs[l][2*i+1]; }
}
static void unpackB(const double*restrict br, const double*restrict bi, double* const* dsts, int nd, long VOL) {
  const __m512i ilo = _mm512_set_epi64(11,3,10,2,9,1,8,0);
  const __m512i ihi = _mm512_set_epi64(15,7,14,6,13,5,12,4);
  long i = 0;
  for (; i + 8 <= VOL; i += 8) {
    V rr[8], ii[8];
    for (int l2 = 0; l2 < 8; l2++) { rr[l2] = LD(br + 8*(i+l2)); ii[l2] = LD(bi + 8*(i+l2)); }
    TR8(rr[0],rr[1],rr[2],rr[3],rr[4],rr[5],rr[6],rr[7]);
    TR8(ii[0],ii[1],ii[2],ii[3],ii[4],ii[5],ii[6],ii[7]);
    for (int l = 0; l < nd; l++) {
      STU(dsts[l] + 2*i, _mm512_permutex2var_pd(rr[l], ilo, ii[l]));
      STU(dsts[l] + 2*i + 8, _mm512_permutex2var_pd(rr[l], ihi, ii[l]));
    }
  }
  for (; i < VOL; i++)
    for (int l = 0; l < nd; l++) { dsts[l][2*i] = br[8*i+l]; dsts[l][2*i+1] = bi[8*i+l]; }
}

// src split [x][z][y] -> dst interleaved [x][y][z]
static void emitT(const double*restrict sre, const double*restrict sim,
                  double*restrict dst, double*restrict dst2,
                  int L, int S, long PX, int ntok) {
  const __m512i ilo = _mm512_set_epi64(11,3,10,2,9,1,8,0);
  const __m512i ihi = _mm512_set_epi64(15,7,14,6,13,5,12,4);
  int tail = L & 7;
  __mmask8 m0 = 0, m1 = 0;
  if (tail) { unsigned bits = (1u << (2*tail)) - 1u; m0 = (__mmask8)(bits & 0xFF); m1 = (__mmask8)(bits >> 8); }
  for (int x = 0; x < L; x++)
    for (int yb = 0; yb < L; yb += 8)
      for (int zb = 0; zb < L; zb += 8) {
        V rr[8], ii[8];
        for (int l = 0; l < 8; l++) {
          int z = zb + l;
          if (z < L) {
            rr[l] = LD(sre + (long)x * PX + (long)z * S + yb);
            ii[l] = LD(sim + (long)x * PX + (long)z * S + yb);
          } else { rr[l] = ZV; ii[l] = ZV; }
        }
        TR8(rr[0],rr[1],rr[2],rr[3],rr[4],rr[5],rr[6],rr[7]);
        TR8(ii[0],ii[1],ii[2],ii[3],ii[4],ii[5],ii[6],ii[7]);
        int zt = (L - zb >= 8) ? 0 : 1;
        for (int l = 0; l < 8; l++) {
          int y = yb + l;
          if (y >= L) break;
          long off = 2 * ((long)(x * L + y) * L + zb);
          double* d = dst + off;
          int nt = ntok && ((((uintptr_t)d) & 63) == 0) && (!dst2 || ((((uintptr_t)(dst2+off)) & 63) == 0));
          V lo = _mm512_permutex2var_pd(rr[l], ilo, ii[l]);
          V hi = _mm512_permutex2var_pd(rr[l], ihi, ii[l]);
          if (!zt && nt) {
            _mm512_stream_pd(d, lo); _mm512_stream_pd(d + 8, hi);
            if (dst2) { _mm512_stream_pd(dst2 + off, lo); _mm512_stream_pd(dst2 + off + 8, hi); }
          } else if (!zt) {
            STU(d, lo); STU(d + 8, hi);
            if (dst2) { STU(dst2 + off, lo); STU(dst2 + off + 8, hi); }
          } else {
            _mm512_mask_storeu_pd(d, m0, lo); _mm512_mask_storeu_pd(d + 8, m1, hi);
            if (dst2) { _mm512_mask_storeu_pd(dst2 + off, m0, lo); _mm512_mask_storeu_pd(dst2 + off + 8, m1, hi); }
          }
        }
      }
}
'''

def emit_run(L):
    S = S_of(L); PX = PX_of(L); VOL = L * L * L
    NB = S // 8; C8 = ceil8(L); NTOK = 1 if L >= 36 else 0
    if L in GROUP_SIZES:
        GROUP_DISPATCH = f'''  while (B - b0 >= 5) {{
    int nreal = (B - b0 >= 8) ? 8 : (int)(B - b0);
    const double* sx[8]; const double* sc[8]; double* d1[8]; double* dm[8];
    for (int l = 0; l < 8; l++) {{
      long bb = b0 + (l < nreal ? l : nreal - 1);
      sx[l] = x0 + 2*(long)bb*{VOL};
      sc[l] = c + 2*(long)bb*{VOL};
      d1[l] = out1 + 2*(long)bb*{VOL};
      dm[l] = outm + 2*(long)bb*{VOL};
    }}
    rung_{L}(sx, sc, d1, dm, nreal, m);
    b0 += nreal;
  }}'''
    else:
        GROUP_DISPATCH = ''
    return f"""
static void run_{L}(long B, long m, const double* x0, const double* c, double* out1, double* outm) {{
  long b0 = 0;
{GROUP_DISPATCH}
  for (long b = b0; b < B; b++) {{
    const double* xb = x0 + 2 * (long)b * {VOL};
    const double* cb = c + 2 * (long)b * {VOL};
    double* o1 = out1 + 2 * (long)b * {VOL};
    double* om = outm + 2 * (long)b * {VOL};
    int useT = (m >= 3);
    int useN = (m >= 4);
    if (useT && useN) cprepBoth(cb, cPo_{L}, cPe_{L}, {L}, {NB}, {C8});
    else if (useT) cprepTpk(cb, cPo_{L}, {L}, {NB}, {C8});
    deintl(xb, Ar_{L}, Ai_{L}, {L}, {S}, {PX});
    for (long t = 1; t <= m; t++) {{
      pass1_{L}(Ar_{L}, Ai_{L});
      if (t == m) {{
        if (t & 1) pass23e_{L}(Ar_{L}, Ai_{L}, cb, (m == 1 ? o1 : om), (m == 1 ? om : 0), 1);
        else       pass23e_{L}(Ar_{L}, Ai_{L}, cb, om, 0, 0);
        break;
      }}
      if (t == 1) {{
        if (useT) {{ pass23f_{L}(Ar_{L}, Ai_{L}, cPo_{L}, o1); continue; }}
        pass23_{L}(Ar_{L}, Ai_{L}, 0, cb, 2);
        emitT(Ar_{L}, Ai_{L}, o1, 0, {L}, {S}, {PX}, {NTOK});
      }} else if (t & 1) {{
        pass23_{L}(Ar_{L}, Ai_{L}, cPo_{L}, 0, 0);
      }} else {{
        if (useN) pass23_{L}(Ar_{L}, Ai_{L}, cPe_{L}, 0, 0);
        else      pass23_{L}(Ar_{L}, Ai_{L}, 0, cb, 1);
      }}
    }}
  }}
  _mm_sfence();
}}
"""

def emit_prof():
    cases = []
    for L in SIZES:
        S = S_of(L); PX = PX_of(L); NB = S//8; C8 = ceil8(L); NTOK = 1 if L >= 36 else 0
        cases.append(f"""  if (L == {L}) {{
    t=nowt(); for(long i=0;i<iters;i++) pass1_{L}(Ar_{L}, Ai_{L}); res[0]=(nowt()-t)/iters;
    t=nowt(); for(long i=0;i<iters;i++) pass23_{L}(Ar_{L}, Ai_{L}, cPo_{L}, 0, 0); res[1]=(nowt()-t)/iters;
    if(!ptmp) ptmp = (double*)aligned_alloc(64, sizeof(double)*2*300000);
    t=nowt(); for(long i=0;i<iters;i++) deintl(ptmp, Ar_{L}, Ai_{L}, {L}, {S}, {PX}); res[2]=(nowt()-t)/iters;
    t=nowt(); for(long i=0;i<iters;i++) cprepTpk(ptmp, cPo_{L}, {L}, {NB}, {C8}); res[3]=(nowt()-t)/iters;
    t=nowt(); for(long i=0;i<iters;i++) emitT(Ar_{L}, Ai_{L}, ptmp, 0, {L}, {S}, {PX}, {NTOK}); res[4]=(nowt()-t)/iters;
    t=nowt(); for(long i=0;i<iters;i++) emitN(Ar_{L}, Ai_{L}, ptmp, 0, {L}, {S}, {PX}, {NTOK}); res[5]=(nowt()-t)/iters;
    res[6]=0;
    t=nowt(); for(long i=0;i<iters;i++) for(int x=0;x<{L};x++) for(int ab=0;ab<{S};ab+=8) p3_{L}(Sr_{L}+ab, Si_{L}+ab, Ar_{L}+(long)x*{PX}+ab, Ai_{L}+(long)x*{PX}+ab, cPo_{L}+((long)x*{NB}+ab/8)*{C8}*16); res[7]=(nowt()-t)/iters;
    t=nowt(); for(long i=0;i<iters*{L};i++) p1_{L}(Ar_{L}, Ai_{L}); res[8]=(nowt()-t)/(iters*{L});
    t=nowt(); for(long i=0;i<iters*{L};i++) p2_{L}(Ar_{L}, Ai_{L}, Sr_{L}, Si_{L}); res[9]=(nowt()-t)/(iters*{L});
    t=nowt(); for(long i=0;i<iters*{L};i++) p3_{L}(Sr_{L}, Si_{L}, Ar_{L}, Ai_{L}, cPo_{L}); res[10]=(nowt()-t)/(iters*{L});
    t=nowt();
    (void)t;
    return;
  }}""")
    return ("""
#include <time.h>
static double nowt(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec + 1e-9*ts.tv_nsec; }
static double* ptmp = 0;
void dbg_prof(long L, long iters, double* res){
  double t;
""" + "\n".join(cases) + "\n}\n")

def emit_tail():
    alloc = []
    for L in SIZES:
        PX = PX_of(L)
        n = L * PX + 16
        for arr in ("Ar", "Ai"):
            alloc.append(f'  {arr}_{L} = alloc0({n});')
        npk = L * (S_of(L)//8) * ceil8(L) * 16 + 64
        alloc.append(f'  cPo_{L} = alloc0({npk});')
        alloc.append(f'  cPe_{L} = alloc0({npk});')
        if L in GROUP_SIZES:
            alloc.append(f'  Gr_{L} = alloc0({L**3*8 + 64});')
            alloc.append(f'  Gi_{L} = alloc0({L**3*8 + 64});')
            alloc.append(f'  cG_{L} = alloc0({L**3*16 + 64});')
        for arr in ("Sr", "Si", "Sr2", "Si2"):
            alloc.append(f'  {arr}_{L} = alloc0({S_of(L)*S_of(L) + 64});')
    dispatch = "\n".join(
        f'  if (L == {L}) {{ run_{L}(B, m, x0, c, out1, outm); return; }}' for L in SIZES)
    dbg = "\n".join(
        f'  if (L == {L}) {{ p0_{L}(ri, ii, ro, io); return; }}' for L in SIZES)
    return f'''
static double* alloc0(long n) {{
  long bytes = ((n * 8 + 2097151) / 2097152) * 2097152;
  double* p = (double*)aligned_alloc(2097152, bytes);
  if (p) madvise(p, bytes, MADV_HUGEPAGE);
  if (!p) p = (double*)aligned_alloc(64, bytes);
  if (!p) {{ p = (double*)malloc(bytes + 64); p = (double*)(((uintptr_t)p + 63) & ~(uintptr_t)63); }}
  memset(p, 0, bytes);
  return p;
}}
void setup(void) {{
{chr(10).join(alloc)}
}}
void run_size(long L, long B, long m, const double* x0, const double* c, double* out1, double* outm) {{
{dispatch}
}}
void dbg_line(long L, const double* ri, const double* ii, double* ro, double* io) {{
{dbg}
}}
'''

def main():
    verify_builders()
    parts = [emit_header(), GENERIC]
    decls = []
    for L in SIZES:
        decls.append(f"static double *Ar_{L}, *Ai_{L}, *Sr_{L}, *Si_{L}, *Sr2_{L}, *Si2_{L}, *cPo_{L}, *cPe_{L};")
        if L in GROUP_SIZES:
            decls.append(f"static double *Gr_{L}, *Gi_{L}, *cG_{L};")
    parts.append("\n".join(decls))
    for L in SIZES:
        parts.append(emit_size(L))
        parts.append(emit_run(L))
    parts.append(emit_tail())
    parts.append(emit_prof())
    src = "\n".join(parts)
    with open("/workdir/implementation.c", "w") as f:
        f.write(src)
    print("wrote implementation.c:", len(src.splitlines()), "lines")


if __name__ == "__main__":
    main()
