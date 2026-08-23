# Generator v2: bounded-liveness codelets
import math, sys
from fractions import Fraction
import mpmath as mp
mp.mp.prec = 120

SIZES = [6, 8, 13, 17, 23, 36, 45, 64]
W = 8

def lay(L):
    P   = -(-L // W) * W
    NCH = P // W
    RST = 2 * NCH * W + 8
    SP  = P * RST + 24
    return P, NCH, RST, SP

def masks(L):
    cnt = L - 8 * (L // 8) if L % 8 else 8
    d = 2 * cnt
    return cnt, (1 << min(d, 8)) - 1, (1 << max(d - 8, 0)) - 1

def is_prime(n):
    return n >= 2 and all(n % d for d in range(2, int(n**.5)+1))

def fmt(x): return float(x).hex()

def trig(num, den):
    num = num % den
    ang = 2*mp.pi*mp.mpf(num)/den
    return mp.cos(ang), mp.sin(ang)

class E:
    def __init__(self):
        self.lines = []; self.n = 0; self.cse = {}
    def raw(self, s): self.lines.append(s)
    def v(self, expr):
        if expr in self.cse: return self.cse[expr]
        self.n += 1; name = f"t{self.n}"
        self.lines.append(f"vd {name} = {expr};")
        self.cse[expr] = name
        return name
    def code(self, ind="  "): return ("\n"+ind).join(self.lines)

def cadd(e,x,y): return (e.v(f"{x[0]} + {y[0]}"), e.v(f"{x[1]} + {y[1]}"))
def csub(e,x,y): return (e.v(f"{x[0]} - {y[0]}"), e.v(f"{x[1]} - {y[1]}"))

SIGN = -1
def cmul_w(e, x, num, den):
    num = (SIGN*num) % den
    f = Fraction(num, den)
    xr, xi = x
    if f == 0: return x
    if f == Fraction(1,2): return (e.v(f"-{xr}"), e.v(f"-{xi}"))
    if f == Fraction(1,4): return (e.v(f"-{xi}"), xr)
    if f == Fraction(3,4): return (xi, e.v(f"-{xr}"))
    c, s = trig(num, den)
    cf, sf = fmt(c), fmt(s)
    if abs(abs(float(c))-abs(float(s))) < 1e-17:
        if float(s)*float(c) > 0:
            return (e.v(f"K({cf}) * ({xr} - {xi})"), e.v(f"K({sf}) * ({xr} + {xi})"))
        else:
            return (e.v(f"K({cf}) * ({xr} + {xi})"), e.v(f"K({sf}) * ({xr} - {xi})"))
    re = e.v(f"K({cf})*{xr} - K({sf})*{xi}")
    im = e.v(f"K({cf})*{xi} + K({sf})*{xr}")
    return (re, im)

# ---------- straight-line recursive DFT on var pairs ----------
def dft_sl(e, xs):
    N = len(xs)
    if N == 1: return xs
    if N == 2: return [cadd(e,xs[0],xs[1]), csub(e,xs[0],xs[1])]
    if is_prime(N):
        if N in RADER_SET: return dft_rader(e, xs, N)
        if N == 23 and RADER23BUF[0]:
            return dft_rader23_sl(e, xs)
        return prime_sl(e, xs)
    A = next((a for a in range(2,N) if N%a==0 and math.gcd(a,N//a)==1 and N//a>1), None)
    if A: return pfa_sl(e, xs, A, N//A)
    if N == 64: N1,N2 = 8,8
    elif N == 16: N1,N2 = 4,4
    elif N == 8: N1,N2 = 4,2
    elif N == 9: N1,N2 = 3,3
    elif N == 4: N1,N2 = 2,2
    else:
        p = min(d for d in range(2,N+1) if N%d==0); N1,N2 = p, N//p
    return ct_sl(e, xs, N1, N2)

def pfa_sl(e, xs, A, B):
    N = A*B
    cols = [dft_sl(e, [xs[(B*n1 + A*n2) % N] for n1 in range(A)]) for n2 in range(B)]
    rows = [dft_sl(e, [cols[n2][k1] for n2 in range(B)]) for k1 in range(A)]
    return [rows[k % A][k % B] for k in range(N)]

def ct_sl(e, xs, N1, N2):
    N = N1*N2
    inner = [dft_sl(e, [xs[N2*n1 + n2] for n1 in range(N1)]) for n2 in range(N2)]
    out = [None]*N
    for k1 in range(N1):
        row = [cmul_w(e, inner[n2][k1], n2*k1, N) for n2 in range(N2)]
        res = dft_sl(e, row)
        for k2 in range(N2): out[k1 + N1*k2] = res[k2]
    return out

def prime_sl(e, xs):
    p = len(xs); h = (p-1)//2
    x0r, x0i = xs[0]
    Ev, Ov = [], []
    for j in range(1, h+1):
        Ev.append(cadd(e, xs[j], xs[p-j])); Ov.append(csub(e, xs[j], xs[p-j]))
    sr, si = x0r, x0i
    for j in range(h):
        sr = e.v(f"{sr} + {Ev[j][0]}"); si = e.v(f"{si} + {Ev[j][1]}")
    out = [None]*p; out[0] = (sr, si)
    for k in range(1, h+1):
        Ar, Ai = x0r, x0i; Cr = None; Di = None
        for j in range(1, h+1):
            c, s = trig(k*j, p); cf, sf = fmt(c), fmt(s)
            Ar = e.v(f"K({cf}) * {Ev[j-1][0]} + {Ar}")
            Ai = e.v(f"K({cf}) * {Ev[j-1][1]} + {Ai}")
            Cr = e.v(f"K({sf}) * {Ov[j-1][0]} + {Cr}") if Cr else e.v(f"K({sf}) * {Ov[j-1][0]}")
            Di = e.v(f"K({sf}) * {Ov[j-1][1]} + {Di}") if Di else e.v(f"K({sf}) * {Ov[j-1][1]}")
        out[k]   = (e.v(f"{Ar} + {Di}"), e.v(f"{Ai} - {Cr}"))
        out[p-k] = (e.v(f"{Ar} - {Di}"), e.v(f"{Ai} + {Cr}"))
    return out

# ================= codelet emitters =================

def crt_index(N, A, B, k1, k2):
    for k in range(N):
        if k % A == k1 and k % B == k2: return k
    raise RuntimeError

class StridedLS:
    """loads/stores at q + j*ST (+8 for imag); optionally separate dst base"""
    def __init__(self, qin="q", qout=None, ST=0, STO=None):
        self.q = qin; self.o = qout or qin; self.ST = ST; self.STO = STO if STO is not None else ST
    def direct(self, e, j):
        return (e.v(f"LD({self.q} + {j*self.ST})"), e.v(f"LD({self.q} + {j*self.ST + 8})"))
    def load_rt(self, jexpr):  # runtime load exprs
        return (f"LD({self.q} + (long)({jexpr})*{self.ST})", f"LD({self.q} + (long)({jexpr})*{self.ST} + 8)")
    def store(self, e, k, r, i):
        e.raw(f"ST({self.o} + {k*self.STO}, {r}); ST({self.o} + {k*self.STO + 8}, {i});")
    def store_rt(self, e, kexpr, r, i):
        e.raw(f"ST({self.o} + (long)({kexpr})*{self.STO}, {r}); ST({self.o} + (long)({kexpr})*{self.STO} + 8, {i});")

class MapStorer:
    """fC store: z = y + c; w = map; plane store. offsets relative to (pl,cq) + cb16 + k*RST"""
    def __init__(self, RST, mapsel='ALT'): self.RST = RST; self.mapsel = mapsel; self.cnt = 0
    def _m(self):
        if self.mapsel == 'ALT':
            self.cnt += 1
            return "MAPW_H" if (self.cnt & 1) else "MAPW_R2"
        return "MAPW"
    def store(self, e, k, r, i):
        o = k*self.RST
        e.raw(f"{{ vd zr = {r} + LD(cq + cb16 + {o}); vd zi = {i} + LD(cq + cb16 + {o+8}); "
              f"{self._m()}(zr, zi, w); ST(pl + cb16 + {o}, zr*w); ST(pl + cb16 + {o+8}, zi*w); }}")
    def store_rt(self, e, kexpr, r, i):
        e.raw(f"{{ long o_ = (long)({kexpr})*{self.RST}; vd zr = {r} + LD(cq + cb16 + o_); vd zi = {i} + LD(cq + cb16 + o_ + 8); "
              f"{self._m()}(zr, zi, w); ST(pl + cb16 + o_, zr*w); ST(pl + cb16 + o_ + 8, zi*w); }}")

def emit_trans_phase0(e, L, NCH, RST, arr=True):
    """transposed loads from S rows cb8.. -> UR/UI arrays (arr=True) or named vars (returns names)"""
    names = {}
    if arr: e.raw(f"vd UR[{L}], UI[{L}];")
    for jb in range(NCH):
        for pname, off in (("re", 0), ("im", 8)):
            vs = [f"g{jb}{pname}{t}" for t in range(8)]
            for t in range(8):
                e.raw(f"vd {vs[t]} = LD(S + cb8*{RST} + {t*RST + jb*16 + off});")
            e.raw(f"TR8({','.join(vs)});")
            for s in range(8):
                j = jb*8 + s
                if j < L:
                    if arr:
                        e.raw(f"{'UR' if pname=='re' else 'UI'}[{j}] = {vs[s]};")
                    else:
                        names[(j, pname)] = vs[s]
    if arr: e.raw('__asm__("" ::: "memory");')
    return names

def emit_prime_loopy(e, p, load_mode, ls, storer, tables):
    """load_mode: 'strided' (use ls.load_rt) or 'array' (UR/UI present)"""
    h = (p-1)//2
    tables.add(p)
    e.raw(f"vd ER[{h}], EI[{h}], OR_[{h}], OI[{h}];")
    if load_mode == 'strided':
        r0, i0 = ls.direct(e, 0)
        pairs = []
        for j in range(1, h+1):
            ar, ai = ls.direct(e, j)
            br, bi = ls.direct(e, p-j)
            er = e.v(f"{ar} + {br}"); ei = e.v(f"{ai} + {bi}")
            e.raw(f"ER[{j-1}] = {er}; EI[{j-1}] = {ei}; OR_[{j-1}] = {ar} - {br}; OI[{j-1}] = {ai} - {bi};")
            pairs.append((er, ei))
    else:
        r0 = e.v("UR[0]"); i0 = e.v("UI[0]")
        pairs = []
        for j in range(1, h+1):
            ar = e.v(f"UR[{j}]"); ai = e.v(f"UI[{j}]")
            br = e.v(f"UR[{p-j}]"); bi = e.v(f"UI[{p-j}]")
            er = e.v(f"{ar} + {br}"); ei = e.v(f"{ai} + {bi}")
            e.raw(f"ER[{j-1}] = {er}; EI[{j-1}] = {ei}; OR_[{j-1}] = {ar} - {br}; OI[{j-1}] = {ai} - {bi};")
            pairs.append((er, ei))
    # X0 via balanced tree
    def tree(vs):
        while len(vs) > 1:
            nxt = []
            for i in range(0, len(vs)-1, 2):
                nxt.append(e.v(f"{vs[i]} + {vs[i+1]}"))
            if len(vs) % 2: nxt.append(vs[-1])
            vs = nxt
        return vs[0]
    sr = tree([r0] + [pr[0] for pr in pairs])
    si = tree([i0] + [pr[1] for pr in pairs])
    tmp = E()
    storer.store(tmp, 0, sr, si)
    e.raw("{ " + tmp.code() + " }")
    # k loop unrolled by 2
    def chains(e2, ks):  # ks: list of C row-index exprs (strings "kk-1","kk")
        rows = [f"SYMC_{p}[{k}]" for k in ks] + [f"SYMS_{p}[{k}]" for k in ks]
        ptrs = []
        for idx, rexpr in enumerate(rows):
            e2.raw(f"const double* restrict tp{idx} = {rexpr};")
            ptrs.append(f"tp{idx}")
        nk = len(ks)
        acc = []
        for t in range(nk):
            e2.raw(f"vd A{t}r = x0r_, A{t}i = x0i_, C{t}r = K(0.0), D{t}i = K(0.0);")
            acc.append((f"A{t}r", f"A{t}i", f"C{t}r", f"D{t}i"))
        for j in range(h):
            e2.raw(f"{{ vd er=ER[{j}], ei=EI[{j}], orr=OR_[{j}], oi=OI[{j}];")
            for t in range(nk):
                cp, sp = ptrs[t], ptrs[nk+t]
                e2.raw(f"  A{t}r += K2({cp}[{j}])*er; A{t}i += K2({cp}[{j}])*ei; "
                       f"C{t}r += K2({sp}[{j}])*orr; D{t}i += K2({sp}[{j}])*oi;" + ("}" if t==nk-1 else ""))
        return acc
    e.raw(f"vd x0r_ = {r0}, x0i_ = {i0};")
    UNR = UNRH if h >= 9 else 2
    nfull = (h // UNR) * UNR
    body = E()
    acc = chains(body, [f"kk-1+{t}" if t else "kk-1" for t in range(UNR)])
    for t in range(UNR):
        kexpr = f"kk+{t}" if t else "kk"
        Ar, Ai, Cr, Di = acc[t]
        body.raw(f"{{ vd xr1 = {Ar} + {Di}, xi1 = {Ai} - {Cr}, xr2 = {Ar} - {Di}, xi2 = {Ai} + {Cr};")
        tmp = E(); storer.store_rt(tmp, kexpr, "xr1", "xi1"); storer.store_rt(tmp, f"{p}-({kexpr})", "xr2", "xi2")
        body.raw(tmp.code() + " }")
    e.raw(f"int kk=1;")
    e.raw(f"for(; kk+{UNR-1}<={h}; kk+={UNR}){{")
    e.raw(body.code())
    e.raw("}")
    rem = h - nfull
    if rem:
        tail = E()
        acc = chains(tail, [f"kk-1+{t}" if t else "kk-1" for t in range(rem)])
        for t in range(rem):
            kexpr = f"kk+{t}" if t else "kk"
            Ar, Ai, Cr, Di = acc[t]
            tail.raw(f"{{ vd xr1 = {Ar} + {Di}, xi1 = {Ai} - {Cr}, xr2 = {Ar} - {Di}, xi2 = {Ai} + {Cr};")
            tmp = E(); storer.store_rt(tmp, kexpr, "xr1", "xi1"); storer.store_rt(tmp, f"{p}-({kexpr})", "xr2", "xi2")
            tail.raw(tmp.code() + " }")
        e.raw("{")
        e.raw(tail.code())
        e.raw("}")

def emit_buffered(e, N, loader_direct, storer):
    """N in {36,45,64}: PFA or CT with stack buffer between stages"""
    if N == 64:
        N1 = N2 = 8; kind = 'ct'
    elif N == 36:
        A, B = 4, 9; kind = 'pfa'
    elif N == 45:
        A, B = 5, 9; kind = 'pfa'
    else: raise RuntimeError
    e.raw(f"vd BR[{N}], BI[{N}];")
    if kind == 'ct':
        for n2 in range(N2):
            xs = [loader_direct(e, N2*n1 + n2) for n1 in range(N1)]
            ys = dft_sl(e, xs)
            for k1 in range(N1):
                r, i = cmul_w(e, ys[k1], n2*k1, N)
                e.raw(f"BR[{k1*N2+n2}] = {r}; BI[{k1*N2+n2}] = {i};")
        e.raw('__asm__("" ::: "memory");')
        for k1 in range(N1):
            xs = [(e.v(f"BR[{k1*N2+n2}]"), e.v(f"BI[{k1*N2+n2}]")) for n2 in range(N2)]
            ys = dft_sl(e, xs)
            for k2 in range(N2):
                storer.store(e, k1 + N1*k2, *ys[k2])
    else:
        for n2 in range(B):
            xs = [loader_direct(e, (B*n1 + A*n2) % N) for n1 in range(A)]
            ys = dft_sl(e, xs)
            for k1 in range(A):
                e.raw(f"BR[{k1*B+n2}] = {ys[k1][0]}; BI[{k1*B+n2}] = {ys[k1][1]};")
        e.raw('__asm__("" ::: "memory");')
        for k1 in range(A):
            xs = [(e.v(f"BR[{k1*B+n2}]"), e.v(f"BI[{k1*B+n2}]")) for n2 in range(B)]
            ys = dft_sl(e, xs)
            for k2 in range(B):
                storer.store(e, crt_index(N, A, B, k1, k2), *ys[k2])

def emit_straight(e, N, loader_direct, storer):
    xs = [loader_direct(e, j) for j in range(N)]
    ys = dft_sl(e, xs)
    for k in range(N):
        storer.store(e, k, *ys[k])

def strategy(L):
    if L in RADER_SET: return 'sl'
    if L in SL_SET: return 'sl'
    if L in (13, 17, 23): return 'prime'
    if L in (36, 45, 64): return 'buf'
    return 'sl'

# ================= top-level per-L functions =================

PRELUDE = r'''
#include <immintrin.h>
#include <string.h>
#include <stdint.h>
#include <sys/mman.h>

typedef double vd __attribute__((vector_size(64), aligned(64)));
#define K(x) ((vd){x,x,x,x,x,x,x,x})
#define K2(x) ((vd)_mm512_set1_pd(x))
#define LD(p) (*(const vd*)(p))
#define ST(p,v) (*(vd*)(p) = (v))
#define MD(x) ((__m512d)(x))
#define VD(x) ((vd)(x))

static const long long IDX_TAB[10][8] __attribute__((aligned(64))) = {
  {0,1,2,3,8,9,10,11},{4,5,6,7,12,13,14,15},
  {0,1,8,9,4,5,12,13},{2,3,10,11,6,7,14,15},
  {0,8,2,10,4,12,6,14},{1,9,3,11,5,13,7,15},
  {0,2,4,6,8,10,12,14},{1,3,5,7,9,11,13,15},
  {0,8,1,9,2,10,3,11},{4,12,5,13,6,14,7,15},
};
#define IX(i) (_mm512_load_si512((const void*)IDX_TAB[i]))

#define TR8(a0,a1,a2,a3,a4,a5,a6,a7) do{ \
  __m512d tq0_,tq1_,tq2_,tq3_,tq4_,tq5_,tq6_,tq7_; \
  tq0_=_mm512_permutex2var_pd(MD(a0),IX(0),MD(a4)); tq4_=_mm512_permutex2var_pd(MD(a0),IX(1),MD(a4)); \
  tq1_=_mm512_permutex2var_pd(MD(a1),IX(0),MD(a5)); tq5_=_mm512_permutex2var_pd(MD(a1),IX(1),MD(a5)); \
  tq2_=_mm512_permutex2var_pd(MD(a2),IX(0),MD(a6)); tq6_=_mm512_permutex2var_pd(MD(a2),IX(1),MD(a6)); \
  tq3_=_mm512_permutex2var_pd(MD(a3),IX(0),MD(a7)); tq7_=_mm512_permutex2var_pd(MD(a3),IX(1),MD(a7)); \
  __m512d tr0_,tr1_,tr2_,tr3_,tr4_,tr5_,tr6_,tr7_; \
  tr0_=_mm512_permutex2var_pd(tq0_,IX(2),tq2_); tr2_=_mm512_permutex2var_pd(tq0_,IX(3),tq2_); \
  tr1_=_mm512_permutex2var_pd(tq1_,IX(2),tq3_); tr3_=_mm512_permutex2var_pd(tq1_,IX(3),tq3_); \
  tr4_=_mm512_permutex2var_pd(tq4_,IX(2),tq6_); tr6_=_mm512_permutex2var_pd(tq4_,IX(3),tq6_); \
  tr5_=_mm512_permutex2var_pd(tq5_,IX(2),tq7_); tr7_=_mm512_permutex2var_pd(tq5_,IX(3),tq7_); \
  a0=VD(_mm512_permutex2var_pd(tr0_,IX(4),tr1_)); a1=VD(_mm512_permutex2var_pd(tr0_,IX(5),tr1_)); \
  a2=VD(_mm512_permutex2var_pd(tr2_,IX(4),tr3_)); a3=VD(_mm512_permutex2var_pd(tr2_,IX(5),tr3_)); \
  a4=VD(_mm512_permutex2var_pd(tr4_,IX(4),tr5_)); a5=VD(_mm512_permutex2var_pd(tr4_,IX(5),tr5_)); \
  a6=VD(_mm512_permutex2var_pd(tr6_,IX(4),tr7_)); a7=VD(_mm512_permutex2var_pd(tr6_,IX(5),tr7_)); \
}while(0)

static inline void tr8a(vd* a){ TR8(a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7]); }

static inline vd rcp_nr(vd d){
  __m512d w = _mm512_rcp14_pd(MD(d));
  w = _mm512_mul_pd(w, _mm512_fnmadd_pd(MD(d), w, _mm512_set1_pd(2.0)));
  w = _mm512_mul_pd(w, _mm512_fnmadd_pd(MD(d), w, _mm512_set1_pd(2.0)));
  return VD(w);
}
static inline vd rsqrt_nr2(vd t){
  __m512d r = _mm512_rsqrt14_pd(MD(t));
  __m512d h = _mm512_mul_pd(MD(t), _mm512_set1_pd(0.5));
  __m512d e;
  e = _mm512_fnmadd_pd(_mm512_mul_pd(h, r), r, _mm512_set1_pd(1.5)); r = _mm512_mul_pd(r, e);
  e = _mm512_fnmadd_pd(_mm512_mul_pd(h, r), r, _mm512_set1_pd(1.5)); r = _mm512_mul_pd(r, e);
  return VD(r);
}
static inline vd rsqrt_nr(vd t){
  /* returns approx 1/sqrt(t), t>0 (biased) */
  __m512d r = _mm512_rsqrt14_pd(MD(t));
  __m512d h = _mm512_mul_pd(MD(t), _mm512_set1_pd(0.5));
  __m512d e;
  e = _mm512_fnmadd_pd(_mm512_mul_pd(h, r), r, _mm512_set1_pd(1.5)); r = _mm512_mul_pd(r, e);
  e = _mm512_fnmadd_pd(_mm512_mul_pd(h, r), r, _mm512_set1_pd(1.5)); r = _mm512_mul_pd(r, e);
  e = _mm512_fnmadd_pd(_mm512_mul_pd(h, r), r, _mm512_set1_pd(1.5)); r = _mm512_mul_pd(r, e);
  return VD(r);
}
/* MAPW selected per size: MAPW_H = sqrt + rcp_nr hybrid; MAPW_R = rsqrt_nr based */
#define MAPW_H(zr,zi,wv) vd wv; { vd t_ = zr*zr + zi*zi; vd s_ = VD(_mm512_sqrt_pd(MD(t_))); wv = rcp_nr(K(1.0)+s_); }
#define MAPW_R(zr,zi,wv) vd wv; { vd t_ = zr*zr + (zi*zi + K(1e-300)); vd r_ = rsqrt_nr(t_); wv = rcp_nr(K(1.0)+t_*r_); }
#define MAPW_NONE(zr,zi,wv) vd wv = K(1.0);
#define MAPW_X(zr,zi,wv) vd wv; { vd t_ = zr*zr + zi*zi; vd s_ = VD(_mm512_sqrt_pd(MD(t_))); wv = K(1.0)/(K(1.0)+s_); }
static inline __m512d sqrt_ymm2(__m512d t){
  __m256d lo = _mm512_castpd512_pd256(t);
  __m256d hi = _mm512_extractf64x4_pd(t, 1);
  lo = _mm256_sqrt_pd(lo); hi = _mm256_sqrt_pd(hi);
  return _mm512_insertf64x4(_mm512_castpd256_pd512(lo), hi, 1);
}
#define MAPW_Y(zr,zi,wv) vd wv; { vd t_ = zr*zr + zi*zi; vd s_ = VD(sqrt_ymm2(MD(t_))); wv = rcp_nr(K(1.0)+s_); }
#define MAPW_R2(zr,zi,wv) vd wv; { vd t_ = zr*zr + (zi*zi + K(1e-300)); vd r_ = rsqrt_nr2(t_); wv = rcp_nr(K(1.0)+t_*r_); }

static void* xalloc(size_t bytes){
  bytes = (bytes + (2UL<<20) - 1) & ~((2UL<<20)-1);
  void* p = mmap(0, bytes, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
  madvise(p, bytes, MADV_HUGEPAGE);
  memset(p, 0, bytes);
  return p;
}
'''

def gen_tables(tables):
    out = []
    for p in sorted(tables):
        h = (p-1)//2
        for nm, fn in (("SYMC", mp.cos), ("SYMS", mp.sin)):
            rows = []
            for k in range(1, h+1):
                vals = []
                for j in range(1, h+1):
                    ang = 2*mp.pi*mp.mpf((k*j) % p)/p
                    vals.append(fmt(fn(ang)))
                rows.append("{" + ",".join(vals) + "}")
            out.append(f"static const double {nm}_{p}[{h}][{h}] __attribute__((aligned(64))) = {{\n  " + ",\n  ".join(rows) + "};")
    return "\n".join(out)

def gen_fA(L, P, NCH, RST, SP, prefetch, tables):
    e = E()
    if prefetch:
        e.raw(f"for(int j_=0;j_<{L};j_++) __builtin_prefetch((const char*)q + (long)j_*{SP*8} + 128);")
    ls = StridedLS("q", None, SP)
    st = strategy(L)
    if st == 'prime': emit_prime_loopy(e, L, 'strided', ls, ls, tables)
    elif st == 'buf': emit_buffered(e, L, ls.direct, ls)
    else: emit_straight(e, L, ls.direct, ls)
    return f"static void fA_{L}(double* restrict q){{\n  {e.code()}\n}}\n"

def gen_fB(L, P, NCH, RST, SP, tables, pfnext=False):
    e = E()
    if pfnext:
        e.raw(f"for(int j_=0;j_<{L};j_++){{ __builtin_prefetch((const char*)q + (long)j_*{RST*8} + {SP*8}); }}")
    ls = StridedLS("q", "s", RST)
    st = strategy(L)
    if st == 'prime': emit_prime_loopy(e, L, 'strided', ls, ls, tables)
    elif st == 'buf': emit_buffered(e, L, ls.direct, ls)
    else: emit_straight(e, L, ls.direct, ls)
    return f"static void fB_{L}(const double* restrict q, double* restrict s){{\n  {e.code()}\n}}\n"

def gen_fC(L, P, NCH, RST, SP, tables, mapkind):
    e = E()
    st = strategy(L)
    storer = MapStorer(RST)
    if st == 'sl':
        names = emit_trans_phase0(e, L, NCH, RST, arr=False)
        loader = lambda e2, j: (names[(j,"re")], names[(j,"im")])
        emit_straight(e, L, loader, storer)
    else:
        emit_trans_phase0(e, L, NCH, RST, arr=True)
        if st == 'prime':
            emit_prime_loopy(e, L, 'array', None, storer, tables)
        else:
            loader = lambda e2, j: (e2.v(f"UR[{j}]"), e2.v(f"UI[{j}]"))
            emit_buffered(e, L, loader, storer)
    mapw = "MAPW_H" if mapkind == 'H' else "MAPW_R"
    return f"""static void fC_{L}(double* restrict pl, const double* restrict S, const double* restrict cq){{
#define MAPW {mapw}
  for(int cb=0; cb<{NCH}; cb++){{
    const long cb8 = (long)cb*8, cb16 = (long)cb*16;
    {e.code(ind="    ")}
  }}
#undef MAPW
}}
"""

def gen_convs(L, P, NCH, RST, SP):
    cnt, ma, mb = masks(L)
    full = (L % 8 == 0)
    blocks = []
    for zb in range(NCH):
        last = (zb == NCH-1) and not full
        if not last:
            blocks.append(f"""{{ __m512d a=_mm512_loadu_pd(s+{zb*16}), b=_mm512_loadu_pd(s+{zb*16+8});
      ST(d+{zb*16}, VD(_mm512_permutex2var_pd(a,IX(6),b))); ST(d+{zb*16+8}, VD(_mm512_permutex2var_pd(a,IX(7),b))); }}""")
        else:
            blocks.append(f"""{{ __m512d a=_mm512_maskz_loadu_pd({ma}, s+{zb*16}), b=_mm512_maskz_loadu_pd({mb}, s+{zb*16+8});
      ST(d+{zb*16}, VD(_mm512_permutex2var_pd(a,IX(6),b))); ST(d+{zb*16+8}, VD(_mm512_permutex2var_pd(a,IX(7),b))); }}""")
    blocks_s = "\n      ".join(blocks)
    conv_in = f"""static void conv_in_{L}(const double* restrict src, double* restrict dst){{
  for(int x=0;x<{L};x++) for(int y=0;y<{L};y++){{
      const double* s = src + 2*((size_t)(x*{L}+y)*{L});
      double* d = dst + (size_t)x*{SP} + (size_t)y*{RST};
      {blocks_s}
  }}
}}
"""
    conv_c = f"""static void conv_c_{L}(const double* restrict src, double* restrict c0, double* restrict c1){{
  vd trr[{NCH}][8], tri[{NCH}][8];
  for(int x=0;x<{L};x++) for(int yb=0; yb<{NCH}; yb++){{
    int ycnt = {L} - yb*8; if(ycnt>8) ycnt=8;
    for(int t=0;t<8;t++){{
      int y = yb*8 + t;
      if(t < ycnt){{
        const double* s = src + 2*((size_t)(x*{L}+y)*{L});
        double* d = c0 + (size_t)x*{SP} + (size_t)y*{RST};
        {blocks_s}
        for(int zb=0; zb<{NCH}; zb++){{ trr[zb][t] = LD(d+zb*16); tri[zb][t] = LD(d+zb*16+8); }}
      }} else {{
        for(int zb=0; zb<{NCH}; zb++){{ trr[zb][t] = K(0.0); tri[zb][t] = K(0.0); }}
      }}
    }}
    for(int zb=0; zb<{NCH}; zb++){{
      int zcnt = {L} - zb*8; if(zcnt>8) zcnt=8;
      tr8a(trr[zb]); tr8a(tri[zb]);
      for(int u=0; u<zcnt; u++){{
        int r = zb*8+u;
        ST(c1 + (size_t)x*{SP} + (size_t)r*{RST} + yb*16,     trr[zb][u]);
        ST(c1 + (size_t)x*{SP} + (size_t)r*{RST} + yb*16 + 8, tri[zb][u]);
      }}
    }}
  }}
}}
"""
    oblocks = []
    for zb in range(NCH):
        last = (zb == NCH-1) and not full
        st = (f"_mm512_storeu_pd(o+{zb*16}, a); _mm512_storeu_pd(o+{zb*16+8}, b);" if not last else
              f"_mm512_mask_storeu_pd(o+{zb*16}, {ma}, a); _mm512_mask_storeu_pd(o+{zb*16+8}, {mb}, b);")
        oblocks.append(f"""{{ __m512d re=MD(LD(d+{zb*16})), im=MD(LD(d+{zb*16+8}));
      __m512d a=_mm512_permutex2var_pd(re,IX(8),im), b=_mm512_permutex2var_pd(re,IX(9),im); {st} }}""")
    oblocks_s = "\n      ".join(oblocks)
    conv_out0 = f"""static void conv_out0_{L}(const double* restrict buf, double* restrict dst){{
  for(int x=0;x<{L};x++) for(int y=0;y<{L};y++){{
      const double* d = buf + (size_t)x*{SP} + (size_t)y*{RST};
      double* o = dst + 2*((size_t)(x*{L}+y)*{L});
      {oblocks_s}
  }}
}}
"""
    conv_out1 = f"""static void conv_out1_{L}(const double* restrict buf, double* restrict dst){{
  vd trr[8], tri[8];
  for(int x=0;x<{L};x++) for(int yb=0; yb<{NCH}; yb++){{
    int ycnt = {L} - yb*8; if(ycnt>8) ycnt=8;
    for(int zb=0; zb<{NCH}; zb++){{
      int zcnt = {L} - zb*8; if(zcnt>8) zcnt=8;
      for(int u=0;u<8;u++){{
        int r = zb*8+u;
        if(u < zcnt){{ trr[u] = LD(buf + (size_t)x*{SP} + (size_t)r*{RST} + yb*16); tri[u] = LD(buf + (size_t)x*{SP} + (size_t)r*{RST} + yb*16 + 8); }}
        else {{ trr[u]=K(0.0); tri[u]=K(0.0); }}
      }}
      tr8a(trr); tr8a(tri);
      for(int t=0;t<ycnt;t++){{
        int y = yb*8+t;
        double* o = dst + 2*((size_t)(x*{L}+y)*{L} + (size_t)zb*8);
        __m512d a=_mm512_permutex2var_pd(MD(trr[t]),IX(8),MD(tri[t])), b=_mm512_permutex2var_pd(MD(trr[t]),IX(9),MD(tri[t]));
        if(zcnt==8){{ _mm512_storeu_pd(o, a); _mm512_storeu_pd(o+8, b); }}
        else {{ _mm512_mask_storeu_pd(o, {ma}, a); _mm512_mask_storeu_pd(o+8, {mb}, b); }}
      }}
    }}
  }}
}}
"""
    return conv_in + conv_c + conv_out0 + conv_out1

def gen_driver(L, P, NCH, RST, SP):
    n3 = L*L*L
    return f"""
static double *BUF_{L}, *C0_{L}, *C1_{L}, *SS_{L};
static void init_{L}(void){{
  BUF_{L} = (double*)xalloc((size_t){L}*{SP}*8 + 256);
  C0_{L}  = (double*)xalloc((size_t){L}*{SP}*8 + 256);
  C1_{L}  = (double*)xalloc((size_t){L}*{SP}*8 + 256);
  SS_{L}  = (double*)xalloc((size_t){P}*{RST}*8 + 256);
}}
static void iter_{L}(double* restrict buf, const double* restrict cb){{
  for(int y=0;y<{L};y++){{
    double* q = buf + (size_t)y*{RST};
    for(int zb=0; zb<{NCH}; zb++) fA_{L}(q + zb*16);
  }}
  for(int x=0;x<{L};x++){{
    double* pl = buf + (size_t)x*{SP};
    for(int zb=0; zb<{NCH}; zb++) fB_{L}(pl + zb*16, SS_{L} + zb*16);
    fC_{L}(pl, SS_{L}, cb + (size_t)x*{SP});
  }}
}}
static void runvol_{L}(const double* x0, const double* c, double* o1, double* om, long m){{
  conv_in_{L}(x0, BUF_{L});
  conv_c_{L}(c, C0_{L}, C1_{L});
  for(long t=1;t<=m;t++){{
    iter_{L}(BUF_{L}, (t&1)? C1_{L} : C0_{L});
    if(t==1){{
      conv_out1_{L}(BUF_{L}, o1);
      if(m==1){{ memcpy(om, o1, (size_t){n3}*16); return; }}
    }}
  }}
  if(m&1) conv_out1_{L}(BUF_{L}, om); else conv_out0_{L}(BUF_{L}, om);
}}
static void run_{L}(long B, long m, const double* x0, const double* c, double* o1, double* om){{
  for(long b=0;b<B;b++)
    runvol_{L}(x0 + (size_t)b*{2*n3}, c + (size_t)b*{2*n3}, o1 + (size_t)b*{2*n3}, om + (size_t)b*{2*n3}, m);
}}
"""

def generate(prefetch_sizes=(36,45,64), mapkinds=None):
    mapkinds = mapkinds or {}
    tables = set()
    body_parts = []
    for L in SIZES:
        P, NCH, RST, SP = lay(L)
        body_parts.append(gen_fA(L, P, NCH, RST, SP, L in prefetch_sizes, tables))
        body_parts.append(gen_fB(L, P, NCH, RST, SP, tables))
        body_parts.append(gen_fC(L, P, NCH, RST, SP, tables, mapkinds.get(L, 'H')))
        body_parts.append(gen_convs(L, P, NCH, RST, SP))
        body_parts.append(gen_driver(L, P, NCH, RST, SP))
    disp = ["void init_all(void){ " + " ".join(f"init_{L}();" for L in SIZES) + " }",
            "void run(int lid, long B, long m, const double* x0, const double* c, double* o1, double* om){",
            "  switch(lid){"]
    for idx, L in enumerate(SIZES):
        disp.append(f"    case {idx}: run_{L}(B,m,x0,c,o1,om); break;")
    disp.append("  }\n}")
    return PRELUDE + "\n" + gen_tables(tables) + "\n" + "\n".join(body_parts) + "\n".join(disp)

if __name__ == "__main__":
    out = generate()
    open(sys.argv[1] if len(sys.argv)>1 else "implementation.c","w").write(out)
    print("generated", len(out.splitlines()), "lines")

# ============ v3 additions: fused BC (NCH==1), 64-swap scheme ============

def gen_fBC_small(L, P, NCH, RST, SP, tables):
    """fused per-plane y-FFT + z-FFT + map for NCH==1 (L<=8)"""
    assert NCH == 1
    e = E()
    # load plane rows (y<L), FFT along y
    xs = []
    for j in range(L):
        xs.append((e.v(f"LD(pl + {j*RST})"), e.v(f"LD(pl + {j*RST+8})")))
    ys = dft_sl(e, xs)
    # assemble 8 row vars (pad rows zero), transpose
    rre, rim = [], []
    for t in range(8):
        if t < L:
            e.n += 1; r = f"t{e.n}"; e.raw(f"vd {r} = {ys[t][0]};")
            e.n += 1; i = f"t{e.n}"; e.raw(f"vd {i} = {ys[t][1]};")
        else:
            e.n += 1; r = f"t{e.n}"; e.raw(f"vd {r} = K(0.0);")
            e.n += 1; i = f"t{e.n}"; e.raw(f"vd {i} = K(0.0);")
        rre.append(r); rim.append(i)
    e.raw(f"TR8({','.join(rre)});")
    e.raw(f"TR8({','.join(rim)});")
    us = [(rre[j], rim[j]) for j in range(L)]
    zs = dft_sl(e, us)
    st = MapStorer(RST, mapsel='ALT')
    e.raw("const long cb16 = 0;")
    for k in range(L):
        st.store(e, k, *zs[k])
    return f"""static void fBC_{L}(double* restrict pl, const double* restrict cq){{
  {e.code()}
}}
"""

def SW(p):  # 8x8 digit swap
    return (p % 8)*8 + p//8

def gen_fB_64sw(RST, SP, pfnext=False):
    e = E()
    if pfnext:
        e.raw(f"for(int j_=0;j_<64;j_++){{ __builtin_prefetch((const char*)q + (long)j_*{RST*8} + {SP*8}); }}")
    N = 64; N1 = N2 = 8
    ls = StridedLS("q", None, RST)
    e.raw(f"vd BR[{N}], BI[{N}];")
    for n2 in range(N2):
        xs = [ls.direct(e, N2*n1 + n2) for n1 in range(N1)]
        ys = dft_sl(e, xs)
        for k1 in range(N1):
            r, i = cmul_w(e, ys[k1], n2*k1, N)
            e.raw(f"BR[{k1*N2+n2}] = {r}; BI[{k1*N2+n2}] = {i};")
    e.raw('__asm__("" ::: "memory");')
    for k1 in range(N1):
        xs = [(e.v(f"BR[{k1*N2+n2}]"), e.v(f"BI[{k1*N2+n2}]")) for n2 in range(N2)]
        ys = dft_sl(e, xs)   # outputs k = k1 + 8*k2 for k2=0..7
        rre = []; rim = []
        for k2 in range(8):
            e.n += 1; r = f"t{e.n}"; e.raw(f"vd {r} = {ys[k2][0]};"); rre.append(r)
            e.n += 1; i = f"t{e.n}"; e.raw(f"vd {i} = {ys[k2][1]};"); rim.append(i)
        e.raw(f"TR8({','.join(rre)});")
        e.raw(f"TR8({','.join(rim)});")
        # vector_s -> S row s*8+zb ; zb given by caller offset (q and s bases already at chunk zb*16? no:)
        for s in range(8):
            e.raw(f"ST(s8 + {(s*8)*RST} + {k1*16}, {rre[s]}); ST(s8 + {(s*8)*RST} + {k1*16 + 8}, {rim[s]});")
    # note: s8 = S + zb*RST  (row offset zb), q = plane + zb*16
    return f"static void fB_64(const double* restrict q, double* restrict s8){{\n  {e.code()}\n}}\n"

def gen_fC_64sw(RST, SP):
    e = E()
    storer = MapStorer(RST)
    ls = StridedLS("S + cb16", None, RST)   # loads S + j*RST + cb16
    def loader(e2, j):
        return (e2.v(f"LD(S + cb16 + {j*RST})"), e2.v(f"LD(S + cb16 + {j*RST+8})"))
    emit_buffered(e, 64, loader, storer)
    return f"""static void fC_64(double* restrict pl, const double* restrict S, const double* restrict cq){{
#define MAPW MAPW_H
  for(int cb=0; cb<8; cb++){{
    const long cb16 = (long)cb*16;
    {e.code(ind="    ")}
  }}
#undef MAPW
}}
"""

def gen_convs_64sw(RST, SP):
    L = 64
    # conv_in: per row: load 16 vds, deinterleave into re[zb], im[zb], TR8 both, store
    e = E()
    re = [f"a{z}" for z in range(8)]; im = [f"b{z}" for z in range(8)]
    for zb in range(8):
        e.raw(f"__m512d p{zb} = _mm512_loadu_pd(s+{zb*16}), q{zb} = _mm512_loadu_pd(s+{zb*16+8});")
        e.raw(f"vd {re[zb]} = VD(_mm512_permutex2var_pd(p{zb},IX(6),q{zb}));")
        e.raw(f"vd {im[zb]} = VD(_mm512_permutex2var_pd(p{zb},IX(7),q{zb}));")
    e.raw(f"TR8({','.join(re)});"); e.raw(f"TR8({','.join(im)});")
    for zb in range(8):
        e.raw(f"ST(d+{zb*16}, {re[zb]}); ST(d+{zb*16+8}, {im[zb]});")
    conv_in = f"""static void conv_in_64(const double* restrict src, double* restrict dst){{
  for(int x=0;x<64;x++) for(int y=0;y<64;y++){{
      const double* s = src + 2*((size_t)(x*64+y)*64);
      double* d = dst + (size_t)x*{SP} + (size_t)y*{RST};
      {e.code(ind="      ")}
  }}
}}
"""
    # conv_out0: inverse
    e = E()
    for zb in range(8):
        e.raw(f"vd {re[zb]} = LD(d+{zb*16}); vd {im[zb]} = LD(d+{zb*16+8});")
    e.raw(f"TR8({','.join(re)});"); e.raw(f"TR8({','.join(im)});")
    for zb in range(8):
        e.raw(f"_mm512_storeu_pd(o+{zb*16}, _mm512_permutex2var_pd(MD({re[zb]}),IX(8),MD({im[zb]})));")
        e.raw(f"_mm512_storeu_pd(o+{zb*16+8}, _mm512_permutex2var_pd(MD({re[zb]}),IX(9),MD({im[zb]})));")
    conv_out0 = f"""static void conv_out0_64(const double* restrict buf, double* restrict dst){{
  for(int x=0;x<64;x++) for(int y=0;y<64;y++){{
      const double* d = buf + (size_t)x*{SP} + (size_t)y*{RST};
      double* o = dst + 2*((size_t)(x*64+y)*64);
      {e.code(ind="      ")}
  }}
}}
"""
    # conv_c: per (x, kb): rows y = u*8+kb: produce c0 rows (swapped lanes) and stash pre-swap; then c1 tiles
    e = E()
    e.raw("vd t0r[8][8], t0i[8][8]; /* [zb][u] pre-swap */")
    e.raw("for(int kb=0; kb<8; kb++){")
    e.raw("  for(int u=0; u<8; u++){")
    e.raw("    int y = u*8+kb;")
    e.raw(f"    const double* s = src + 2*((size_t)(x*64+y)*64);")
    e.raw(f"    double* d = c0 + (size_t)x*{SP} + (size_t)y*16;")
    sub = E()
    for zb in range(8):
        sub.raw(f"__m512d p{zb} = _mm512_loadu_pd(s+{zb*16}), q{zb} = _mm512_loadu_pd(s+{zb*16+8});")
        sub.raw(f"vd {re[zb]} = VD(_mm512_permutex2var_pd(p{zb},IX(6),q{zb}));")
        sub.raw(f"vd {im[zb]} = VD(_mm512_permutex2var_pd(p{zb},IX(7),q{zb}));")
        sub.raw(f"t0r[{zb}][u] = {re[zb]}; t0i[{zb}][u] = {im[zb]};")
    sub.raw(f"TR8({','.join(re)});"); sub.raw(f"TR8({','.join(im)});")
    for zb in range(8):
        sub.raw(f"ST(d+{zb*1024}, {re[zb]}); ST(d+{zb*1024+8}, {im[zb]});")
    e.raw("    " + sub.code(ind="    "))
    e.raw("  }")
    e.raw("  for(int zb=0; zb<8; zb++){")
    e.raw("    tr8a(t0r[zb]); tr8a(t0i[zb]);")
    e.raw("    for(int u=0; u<8; u++){ int r = zb*8+u;")
    e.raw(f"      ST(c1 + (size_t)x*{SP} + (size_t)(kb*64 + r)*16, t0r[zb][u]);")
    e.raw(f"      ST(c1 + (size_t)x*{SP} + (size_t)(kb*64 + r)*16 + 8, t0i[zb][u]); }}")
    e.raw("  }")
    e.raw("}")
    conv_c = f"""static void conv_c_64(const double* restrict src, double* restrict c0, double* restrict c1){{
  for(int x=0;x<64;x++){{
    {e.code(ind="    ")}
  }}
}}
"""
    # conv_out1: rows r=z true, lane-block kb lane u <-> y=u*8+kb
    e = E()
    e.raw("vd trr[8], tri[8];")
    e.raw("for(int kb=0; kb<8; kb++) for(int zb=0; zb<8; zb++){")
    e.raw(f"  for(int u=0;u<8;u++){{ int r = zb*8+u; trr[u] = LD(buf + (size_t)x*{SP} + (size_t)r*{RST} + kb*16); tri[u] = LD(buf + (size_t)x*{SP} + (size_t)r*{RST} + kb*16 + 8); }}")
    e.raw("  tr8a(trr); tr8a(tri);")
    e.raw("  /* vector_u over z-lanes s for y = u*8+kb */")
    e.raw("  for(int u=0;u<8;u++){ int y = u*8+kb;")
    e.raw(f"    double* o = dst + 2*((size_t)(x*64+y)*64 + (size_t)zb*8);")
    e.raw("    _mm512_storeu_pd(o,   _mm512_permutex2var_pd(MD(trr[u]),IX(8),MD(tri[u])));")
    e.raw("    _mm512_storeu_pd(o+8, _mm512_permutex2var_pd(MD(trr[u]),IX(9),MD(tri[u])));")
    e.raw("  }")
    e.raw("}")
    conv_out1 = f"""static void conv_out1_64(const double* restrict buf, double* restrict dst){{
  for(int x=0;x<64;x++){{
    {e.code(ind="    ")}
  }}
}}
"""
    return conv_in + conv_c + conv_out0 + conv_out1

def gen_driver_v3(L, P, NCH, RST, SP):
    n3 = L*L*L
    if L == 64:
        inner = f"""  for(int x=0;x<{L};x++){{
    double* pl = buf + (size_t)x*{SP};
    for(int zb=0; zb<{NCH}; zb++) fB_64(pl + zb*16, SS_64 + (size_t)zb*{RST});
    fC_64(pl, SS_64, cb + (size_t)x*{SP});
  }}"""
    elif NCH == 1:
        inner = f"""  for(int x=0;x<{L};x++){{
    double* pl = buf + (size_t)x*{SP};
    fBC_{L}(pl, cb + (size_t)x*{SP});
  }}"""
    else:
        inner = f"""  for(int x=0;x<{L};x++){{
    double* pl = buf + (size_t)x*{SP};
    for(int zb=0; zb<{NCH}; zb++) fB_{L}(pl + zb*16, SS_{L} + zb*16);
    fC_{L}(pl, SS_{L}, cb + (size_t)x*{SP});
  }}"""
    return f"""
static double *BUF_{L}, *C0_{L}, *C1_{L}, *SS_{L};
static void init_{L}(void){{
  BUF_{L} = (double*)xalloc((size_t){L}*{SP}*8 + 256);
  C0_{L}  = (double*)xalloc((size_t){L}*{SP}*8 + 256);
  C1_{L}  = (double*)xalloc((size_t){L}*{SP}*8 + 256);
  SS_{L}  = (double*)xalloc((size_t){P}*{RST}*8 + 256);
}}
static void iter_{L}(double* restrict buf, const double* restrict cb){{
  for(int y=0;y<{L};y++){{
    double* q = buf + (size_t)y*{RST};
    for(int zb=0; zb<{NCH}; zb++) fA_{L}(q + zb*16);
  }}
{inner}
}}
static void runvol_{L}(const double* x0, const double* c, double* o1, double* om, long m){{
  conv_in_{L}(x0, BUF_{L});
  conv_c_{L}(c, C0_{L}, C1_{L});
  for(long t=1;t<=m;t++){{
    iter_{L}(BUF_{L}, (t&1)? C1_{L} : C0_{L});
    if(t==1){{
      conv_out1_{L}(BUF_{L}, o1);
      if(m==1){{ memcpy(om, o1, (size_t){n3}*16); return; }}
    }}
  }}
  if(m&1) conv_out1_{L}(BUF_{L}, om); else conv_out0_{L}(BUF_{L}, om);
}}
static void run_{L}(long B, long m, const double* x0, const double* c, double* o1, double* om){{
  for(long b=0;b<B;b++)
    runvol_{L}(x0 + (size_t)b*{2*n3}, c + (size_t)b*{2*n3}, o1 + (size_t)b*{2*n3}, om + (size_t)b*{2*n3}, m);
}}
"""

def generate_v3(prefetch_sizes=(64,), mapkinds=None):
    mapkinds = mapkinds or {}
    tables = set()
    parts = [PRELUDE]
    tablebuf = []
    body = []
    for L in SIZES:
        P, NCH, RST, SP = lay(L)
        body.append(gen_fA(L, P, NCH, RST, SP, L in prefetch_sizes, tables))
        if L == 64:
            body.append(gen_fB_64sw(RST, SP, pfnext=(L in pfB)))
            body.append(gen_fC_64sw(RST, SP))
            body.append(gen_convs_64sw(RST, SP))
        elif NCH == 1:
            body.append(gen_fBC_small(L, P, NCH, RST, SP, tables))
            body.append(gen_convs(L, P, NCH, RST, SP))
        else:
            body.append(gen_fB(L, P, NCH, RST, SP, tables, pfnext=(L in pfB)))
            body.append(gen_fC(L, P, NCH, RST, SP, tables, mapkinds.get(L,'H')))
            body.append(gen_convs(L, P, NCH, RST, SP))
        body.append(gen_driver_v3(L, P, NCH, RST, SP))
    disp = ["void init_all(void){ " + " ".join(f"init_{L}();" for L in SIZES) + " }",
            "void run(int lid, long B, long m, const double* x0, const double* c, double* o1, double* om){",
            "  switch(lid){"]
    for idx, L in enumerate(SIZES):
        disp.append(f"    case {idx}: run_{L}(B,m,x0,c,o1,om); break;")
    disp.append("  }\n}")
    return PRELUDE + "\n" + gen_tables(tables) + "\n" + "\n".join(body) + "\n".join(disp)

# ============ v4: map fused into pass-A (iteration order: y,z,x+map) ============

class CMapStorerS:
    """store with c+map at stride ST relative to q (state) and cq (c), same offsets.
       mapsel: None -> use MAPW macro; 'ALT' -> alternate MAPW_H / MAPW_R2 per store site"""
    def __init__(self, ST, mapsel=None): self.ST = ST; self.mapsel = mapsel; self.cnt = 0
    def _m(self):
        if self.mapsel == 'ALT':
            self.cnt += 1
            return "MAPW_H" if (self.cnt & 1) else "MAPW_R2"
        return "MAPW"
    def store(self, e, k, r, i):
        o = k*self.ST
        e.raw(f"{{ vd zr = {r} + LD(cq + {o}); vd zi = {i} + LD(cq + {o+8}); "
              f"{self._m()}(zr, zi, w); ST(q + {o}, zr*w); ST(q + {o+8}, zi*w); }}")
    def store_rt(self, e, kexpr, r, i):
        e.raw(f"{{ long o_ = (long)({kexpr})*{self.ST}; vd zr = {r} + LD(cq + o_); vd zi = {i} + LD(cq + o_ + 8); "
              f"{self._m()}(zr, zi, w); ST(q + o_, zr*w); ST(q + o_ + 8, zi*w); }}")

def gen_fA_v4(L, P, NCH, RST, SP, prefetch, tables, mapkind):
    e = E()
    if prefetch:
        e.raw(f"for(int j_=0;j_<{L};j_++) __builtin_prefetch((const char*)q + (long)j_*{SP*8} + 128);")
    ls = StridedLS("q", None, SP)
    st = CMapStorerS(SP)
    strat = strategy(L)
    if strat == 'prime': emit_prime_loopy(e, L, 'strided', ls, st, tables)
    elif strat == 'buf': emit_buffered(e, L, ls.direct, st)
    else: emit_straight(e, L, ls.direct, st)
    mapw = "MAPW_H" if mapkind == 'H' else "MAPW_R"
    return f"""static void fA_{L}(double* restrict q, const double* restrict cq){{
#define MAPW {mapw}
  {e.code()}
#undef MAPW
}}
"""

class PlaneStorer:
    """fC store without map: plane rows at cb16 + k*RST"""
    def __init__(self, RST): self.RST = RST
    def store(self, e, k, r, i):
        o = k*self.RST
        e.raw(f"ST(pl + cb16 + {o}, {r}); ST(pl + cb16 + {o+8}, {i});")
    def store_rt(self, e, kexpr, r, i):
        e.raw(f"{{ long o_ = (long)({kexpr})*{self.RST}; ST(pl + cb16 + o_, {r}); ST(pl + cb16 + o_ + 8, {i}); }}")

def gen_fC_v4(L, P, NCH, RST, SP, tables):
    e = E()
    strat = strategy(L)
    storer = PlaneStorer(RST)
    if strat == 'sl':
        names = emit_trans_phase0(e, L, NCH, RST, arr=False)
        loader = lambda e2, j: (names[(j,"re")], names[(j,"im")])
        emit_straight(e, L, loader, storer)
    else:
        emit_trans_phase0(e, L, NCH, RST, arr=True)
        if strat == 'prime':
            emit_prime_loopy(e, L, 'array', None, storer, tables)
        else:
            loader = lambda e2, j: (e2.v(f"UR[{j}]"), e2.v(f"UI[{j}]"))
            emit_buffered(e, L, loader, storer)
    return f"""static void fC_{L}(double* restrict pl, const double* restrict S){{
  for(int cb=0; cb<{NCH}; cb++){{
    const long cb8 = (long)cb*8, cb16 = (long)cb*16;
    {e.code(ind="    ")}
  }}
}}
"""

def gen_fC_64sw_v4(RST, SP):
    e = E()
    storer = PlaneStorer(RST)
    def loader(e2, j):
        return (e2.v(f"LD(S + cb16 + {j*RST})"), e2.v(f"LD(S + cb16 + {j*RST+8})"))
    emit_buffered(e, 64, loader, storer)
    return f"""static void fC_64(double* restrict pl, const double* restrict S){{
  for(int cb=0; cb<8; cb++){{
    const long cb16 = (long)cb*16;
    {e.code(ind="    ")}
  }}
}}
"""

def gen_fBC_small_v4(L, P, NCH, RST, SP, tables):
    assert NCH == 1
    e = E()
    xs = []
    for j in range(L):
        xs.append((e.v(f"LD(pl + {j*RST})"), e.v(f"LD(pl + {j*RST+8})")))
    ys = dft_sl(e, xs)
    rre, rim = [], []
    for t in range(8):
        if t < L:
            e.n += 1; r = f"t{e.n}"; e.raw(f"vd {r} = {ys[t][0]};")
            e.n += 1; i = f"t{e.n}"; e.raw(f"vd {i} = {ys[t][1]};")
        else:
            e.n += 1; r = f"t{e.n}"; e.raw(f"vd {r} = K(0.0);")
            e.n += 1; i = f"t{e.n}"; e.raw(f"vd {i} = K(0.0);")
        rre.append(r); rim.append(i)
    e.raw(f"TR8({','.join(rre)});")
    e.raw(f"TR8({','.join(rim)});")
    us = [(rre[j], rim[j]) for j in range(L)]
    zs = dft_sl(e, us)
    for k in range(L):
        e.raw(f"ST(pl + {k*RST}, {zs[k][0]}); ST(pl + {k*RST+8}, {zs[k][1]});")
    return f"static void fBC_{L}(double* restrict pl){{\n  {e.code()}\n}}\n"

def gen_driver_v4(L, P, NCH, RST, SP):
    n3 = L*L*L
    if L == 64:
        inner = f"""  for(int x=0;x<{L};x++){{
    double* pl = buf + (size_t)x*{SP};
    for(int zb=0; zb<{NCH}; zb++) fB_64(pl + zb*16, SS_64 + (size_t)zb*{RST});
    fC_64(pl, SS_64);
  }}"""
    elif NCH == 1:
        inner = f"""  for(int x=0;x<{L};x++) fBC_{L}(buf + (size_t)x*{SP});"""
    else:
        inner = f"""  for(int x=0;x<{L};x++){{
    double* pl = buf + (size_t)x*{SP};
    for(int zb=0; zb<{NCH}; zb++) fB_{L}(pl + zb*16, SS_{L} + zb*16);
    fC_{L}(pl, SS_{L});
  }}"""
    return f"""
static double *BUF_{L}, *C0_{L}, *C1_{L}, *SS_{L};
static void init_{L}(void){{
  BUF_{L} = (double*)xalloc((size_t){L}*{SP}*8 + 256);
  C0_{L}  = (double*)xalloc((size_t){L}*{SP}*8 + 256);
  C1_{L}  = (double*)xalloc((size_t){L}*{SP}*8 + 256);
  SS_{L}  = (double*)xalloc((size_t){P}*{RST}*8 + 256);
}}
static void iter_{L}(double* restrict buf, const double* restrict cb){{
{inner}
  for(int y=0;y<{L};y++){{
    double* q = buf + (size_t)y*{RST};
    const double* cq = cb + (size_t)y*{RST};
    for(int zb=0; zb<{NCH}; zb++) fA_{L}(q + zb*16, cq + zb*16);
  }}
}}
static void runvol_{L}(const double* x0, const double* c, double* o1, double* om, long m){{
  conv_in_{L}(x0, BUF_{L});
  conv_c_{L}(c, C0_{L}, C1_{L});
  for(long t=1;t<=m;t++){{
    iter_{L}(BUF_{L}, (t&1)? C1_{L} : C0_{L});
    if(t==1){{
      conv_out1_{L}(BUF_{L}, o1);
      if(m==1){{ memcpy(om, o1, (size_t){n3}*16); return; }}
    }}
  }}
  if(m&1) conv_out1_{L}(BUF_{L}, om); else conv_out0_{L}(BUF_{L}, om);
}}
static void run_{L}(long B, long m, const double* x0, const double* c, double* o1, double* om){{
  for(long b=0;b<B;b++)
    runvol_{L}(x0 + (size_t)b*{2*n3}, c + (size_t)b*{2*n3}, o1 + (size_t)b*{2*n3}, om + (size_t)b*{2*n3}, m);
}}
"""

def generate_v4(prefetch_sizes=(), mapkinds=None):
    mapkinds = mapkinds or {}
    tables = set()
    body = []
    for L in SIZES:
        P, NCH, RST, SP = lay(L)
        body.append(gen_fA_v4(L, P, NCH, RST, SP, L in prefetch_sizes, tables, mapkinds.get(L,'H')))
        if L == 64:
            body.append(gen_fB_64sw(RST, SP))
            body.append(gen_fC_64sw_v4(RST, SP))
            body.append(gen_convs_64sw(RST, SP))
        elif NCH == 1:
            body.append(gen_fBC_small_v4(L, P, NCH, RST, SP, tables))
            body.append(gen_convs(L, P, NCH, RST, SP))
        else:
            body.append(gen_fB(L, P, NCH, RST, SP, tables))
            body.append(gen_fC_v4(L, P, NCH, RST, SP, tables))
            body.append(gen_convs(L, P, NCH, RST, SP))
        body.append(gen_driver_v4(L, P, NCH, RST, SP))
    disp = ["void init_all(void){ " + " ".join(f"init_{L}();" for L in SIZES) + " }",
            "void run(int lid, long B, long m, const double* x0, const double* c, double* o1, double* om){",
            "  switch(lid){"]
    for idx, L in enumerate(SIZES):
        disp.append(f"    case {idx}: run_{L}(B,m,x0,c,o1,om); break;")
    disp.append("  }\n}")
    return PRELUDE + "\n" + gen_tables(tables) + "\n" + "\n".join(body) + "\n".join(disp)

# ============ v5: per-L choice of map placement + prefetch options ============

def gen_fA64_２ch(RST, SP):
    e = E()
    ls0 = StridedLS("q", None, SP)
    class LS2:
        def __init__(self, off): self.off = off
        def direct(self, e2, j):
            return (e2.v(f"LD(q + {j*SP + self.off})"), e2.v(f"LD(q + {j*SP + self.off + 8})"))
        def store(self, e2, k, r, i):
            e2.raw(f"ST(q + {k*SP + self.off}, {r}); ST(q + {k*SP + self.off + 8}, {i});")
    a = LS2(0); b = LS2(16)
    emit_buffered(e, 64, a.direct, a)
    e2 = E()
    emit_buffered(e2, 64, b.direct, b)
    import re as _re
    body2 = e2.code()
    body2 = _re.sub(r"\bt(\d+)\b", r"u\1", body2)
    body2 = body2.replace("BR", "BR2").replace("BI", "BI2")
    return f"static void fA_64(double* restrict q){{\n  {e.code()}\n  {body2}\n}}\n"

def gen_fA_v3(L, P, NCH, RST, SP, prefetch, tables):
    e = E()
    if prefetch:
        e.raw(f"for(int j_=0;j_<{L};j_++) __builtin_prefetch((const char*)q + (long)j_*{SP*8} + 128);")
    ls = StridedLS("q", None, SP)
    strat = strategy(L)
    if strat == 'prime': emit_prime_loopy(e, L, 'strided', ls, ls, tables)
    elif strat == 'buf': emit_buffered(e, L, ls.direct, ls)
    else: emit_straight(e, L, ls.direct, ls)
    return f"static void fA_{L}(double* restrict q){{\n  {e.code()}\n}}\n"

class MapStorer64:
    def __init__(self, RST): self.RST = RST; self.cnt = 0
    def _m(self):
        self.cnt += 1
        return "MAPW_H" if (self.cnt & 1) else "MAPW_R2"
    def store(self, e, k, r, i):
        o = k*self.RST; oc = k*16
        e.raw(f"{{ vd zr = {r} + LD(cq + cb64 + {oc}); vd zi = {i} + LD(cq + cb64 + {oc+8}); "
              f"{self._m()}(zr, zi, w); ST(pl + cb16 + {o}, zr*w); ST(pl + cb16 + {o+8}, zi*w); }}")

def gen_fC_64sw_map(RST, SP, cpf):
    e = E()
    storer = MapStorer64(RST)
    def loader(e2, j):
        return (e2.v(f"LD(S + cb16 + {j*RST})"), e2.v(f"LD(S + cb16 + {j*RST+8})"))
    emit_buffered(e, 64, loader, storer)
    pf = f"for(int k_=0;k_<64;k_++){{ __builtin_prefetch((const char*)(cq + cb16 + (long)k_*{RST} + {SP*8})); __builtin_prefetch((const char*)(cq + cb16 + (long)k_*{RST} + {SP*8} + 64)); }}\n    " if cpf else ""
    return f"""static void fC_64(double* restrict pl, const double* restrict S, const double* restrict cq){{
#define MAPW MAPW_H
  for(int cb=0; cb<8; cb++){{
    const long cb16 = (long)cb*16, cb64 = (long)cb*1024;
    {pf}{e.code(ind="    ")}
  }}
#undef MAPW
}}
"""

def gen_driver_v5(L, P, NCH, RST, SP, mapA):
    n3 = L*L*L
    if L == 64:
        plane = f"""    for(int zb=0; zb<{NCH}; zb++) fB_64(pl + zb*16, SS_64 + (size_t)zb*{RST});
    fC_64(pl, SS_64{'' if mapA else ', cb + (size_t)x*'+str(SP)});"""
    elif NCH == 1:
        plane = f"    fBC_{L}(pl{'' if mapA else ', cb + (size_t)x*'+str(SP)});"
    else:
        plane = f"""    for(int zb=0; zb<{NCH}; zb++) fB_{L}(pl + zb*16, SS_{L} + zb*16);
    fC_{L}(pl, SS_{L}{'' if mapA else ', cb + (size_t)x*'+str(SP)});"""
    if mapA:
        order = f"""  for(int x=0;x<{L};x++){{
    double* pl = buf + (size_t)x*{SP};
{plane}
  }}
  for(int y=0;y<{L};y++){{
    double* q = buf + (size_t)y*{RST};
    const double* cq = cb + (size_t)y*{RST};
    for(int zb=0; zb<{NCH}; zb++) fA_{L}(q + zb*16, cq + zb*16);
  }}"""
    else:
        order = f"""  for(int y=0;y<{L};y++){{
    double* q = buf + (size_t)y*{RST};
    for(int zb=0; zb<{NCH}; zb++) fA_{L}(q + zb*16);
  }}
  for(int x=0;x<{L};x++){{
    double* pl = buf + (size_t)x*{SP};
{plane}
  }}"""
    return f"""
static double *BUF_{L}, *C0_{L}, *C1_{L}, *SS_{L};
static void init_{L}(void){{
  BUF_{L} = (double*)xalloc((size_t){L}*{SP}*8 + 256);
  C0_{L}  = (double*)xalloc((size_t){L}*{SP}*8 + 256);
  C1_{L}  = (double*)xalloc((size_t){L}*{SP}*8 + 256);
  SS_{L}  = (double*)xalloc((size_t){P}*{RST}*8 + 256);
}}
static void iter_{L}(double* restrict buf, const double* restrict cb){{
{order}
}}
static void runvol_{L}(const double* x0, const double* c, double* o1, double* om, long m){{
  conv_in_{L}(x0, BUF_{L});
  conv_c_{L}(c, C0_{L}, C1_{L});
  for(long t=1;t<=m;t++){{
    iter_{L}(BUF_{L}, (t&1)? C1_{L} : C0_{L});
    if(t==1){{
      conv_out1_{L}(BUF_{L}, o1);
      if(m==1){{ memcpy(om, o1, (size_t){n3}*16); return; }}
    }}
  }}
  if(m&1) conv_out1_{L}(BUF_{L}, om); else conv_out0_{L}(BUF_{L}, om);
}}
static void run_{L}(long B, long m, const double* x0, const double* c, double* o1, double* om){{
  for(long b=0;b<B;b++)
    runvol_{L}(x0 + (size_t)b*{2*n3}, c + (size_t)b*{2*n3}, o1 + (size_t)b*{2*n3}, om + (size_t)b*{2*n3}, m);
}}
"""

def generate_v5(mapA=frozenset(), prefetchA=frozenset(), cpf64=False):
    tables = set()
    body = []
    for L in SIZES:
        P, NCH, RST, SP = lay(L)
        if L in mapA:
            body.append(gen_fA_v4(L, P, NCH, RST, SP, L in prefetchA, tables, 'H'))
        else:
            body.append(gen_fA_v3(L, P, NCH, RST, SP, L in prefetchA, tables))
        if L == 64:
            body.append(gen_fB_64sw(RST, SP))
            body.append(gen_fC_64sw_v4(RST, SP) if L in mapA else gen_fC_64sw_map(RST, SP, cpf64))
            body.append(gen_convs_64sw(RST, SP))
        elif NCH == 1:
            body.append(gen_fBC_small_v4(L, P, NCH, RST, SP, tables) if L in mapA else gen_fBC_small(L, P, NCH, RST, SP, tables))
            body.append(gen_convs(L, P, NCH, RST, SP))
        else:
            body.append(gen_fB(L, P, NCH, RST, SP, tables))
            body.append(gen_fC_v4(L, P, NCH, RST, SP, tables) if L in mapA else gen_fC(L, P, NCH, RST, SP, tables, 'H'))
            body.append(gen_convs(L, P, NCH, RST, SP))
        body.append(gen_driver_v5(L, P, NCH, RST, SP, L in mapA))
    disp = ["void init_all(void){ " + " ".join(f"init_{L}();" for L in SIZES) + " }",
            "void run(int lid, long B, long m, const double* x0, const double* c, double* o1, double* om){",
            "  switch(lid){"]
    for idx, L in enumerate(SIZES):
        disp.append(f"    case {idx}: run_{L}(B,m,x0,c,o1,om); break;")
    disp.append("  }\n}")
    return PRELUDE + "\n" + gen_tables(tables) + "\n" + "\n".join(body) + "\n".join(disp)

# ============ v6: batch-lane groups of 8 volumes for small L ============
GSIZES = (6, 8, 13, 17, 23)

def gSPL(L):
    import math as _m
    base = L*L*16
    pad = 0
    while True:
        adv = (((base+pad)*8)//64) % 64          # L1 set advance per plane (64 sets)
        npos = 64 // _m.gcd(adv if adv else 64, 64)
        if npos >= 1 and -(-L // max(npos,1)) <= 6 and adv % 2 == 1:
            break
        pad += 8
    return base + pad

def gen_group_codelets(L, tables, mapw='MAPW_H', pfA=False, pfcNTA=False, pfBnext=False, pfCnext=False, defer_map=False):
    n3 = L*L*L
    SA, SB, SC = gSPL(L), L*16, 16
    out = []
    # fA: x-pass
    e = E()
    if pfA:
        e.raw(f"for(int j_=0;j_<{L};j_++) __builtin_prefetch((const char*)q + (long)j_*{SA*8} + 128);")
    ls = StridedLS("q", None, SA)
    strat = strategy(L)
    if strat == 'prime': emit_prime_loopy(e, L, 'strided', ls, ls, tables)
    elif strat == 'buf': emit_buffered(e, L, ls.direct, ls)
    else: emit_straight(e, L, ls.direct, ls)
    out.append(f"static void gA_{L}(double* restrict q){{\n  {e.code()}\n}}\n")
    # fB: y-pass
    e = E()
    if pfBnext:
        e.raw(f"for(int j_=0;j_<{L};j_++) __builtin_prefetch((const char*)q + (long)j_*{SB*8} + 128);")
    ls = StridedLS("q", None, SB)
    if strat == 'prime': emit_prime_loopy(e, L, 'strided', ls, ls, tables)
    elif strat == 'buf': emit_buffered(e, L, ls.direct, ls)
    else: emit_straight(e, L, ls.direct, ls)
    out.append(f"static void gB_{L}(double* restrict q){{\n  {e.code()}\n}}\n")
    # fC: z-pass + c + map
    e = E()
    if pfCnext:
        e.raw(f"for(int j_=0;j_<{L};j_++){{ __builtin_prefetch((const char*)q + (long)j_*{SC*8} + {SB*8}); __builtin_prefetch((const char*)cq + (long)j_*{SC*8} + {SB*8}); }}")
    ls = StridedLS("q", None, SC)
    if mapw == 'ALT':
        st = CMapStorerS(SC, mapsel='ALT')
    elif defer_map:
        class AddCStorer:
            def store(self, e2, k, r, i):
                o = k*SC
                e2.raw(f"ST(q + {o}, {r} + LD(cq + {o})); ST(q + {o+8}, {i} + LD(cq + {o+8}));")
            def store_rt(self, e2, kexpr, r, i):
                e2.raw(f"{{ long o_ = (long)({kexpr})*{SC}; ST(q + o_, {r} + LD(cq + o_)); ST(q + o_ + 8, {i} + LD(cq + o_ + 8)); }}")
        st = AddCStorer()
    else:
        st = CMapStorerS(SC)
    if strat == 'prime': emit_prime_loopy(e, L, 'strided', ls, st, tables)
    elif strat == 'buf': emit_buffered(e, L, ls.direct, st)
    else: emit_straight(e, L, ls.direct, st)
    tail = ""
    if defer_map:
        tail = f"""
  for(int j_=0;j_<{L};j_++){{
    vd zr = LD(q + j_*{SC}), zi = LD(q + j_*{SC} + 8);
    MAPW(zr, zi, w);
    ST(q + j_*{SC}, zr*w); ST(q + j_*{SC} + 8, zi*w);
  }}"""
    pfc = f"for(int j_=0;j_<{L};j_++) __builtin_prefetch((const char*)(cq + j_*16 + {L*16*8}), 0, 0);\n  " if pfcNTA else ""
    out.append(f"""static void gC_{L}(double* restrict q, const double* restrict cq){{
#define MAPW {mapw}
  {pfc}{e.code()}{tail}
#undef MAPW
}}
""")
    return "\n".join(out)

def gen_group_convs(L):
    n3 = L*L*L
    SPL = gSPL(L)
    nfull = L // 8
    cnt = L % 8
    dma = (1 << min(2*cnt,8)) - 1
    dmb = (1 << max(2*cnt-8,0)) - 1
    # conv_in_g: for each (x,y) row: for each z-tile: load 8 vols' segments, deinterleave, TR8, store blocks
    body = []
    body.append(f"for(int x_=0; x_<{L}; x_++) for(int y_=0; y_<{L}; y_++){{")
    body.append(f"  double* d = dst + (size_t)x_*{SPL} + (size_t)y_*{L*16};")
    body.append(f"  const double* s0 = src + 2*((size_t)(x_*{L}+y_))*{L};")
    for zt in range(nfull + (1 if cnt else 0)):
        tail = (zt == nfull)
        body.append("  {")
        for t in range(8):
            if not tail:
                body.append(f"    __m512d p{t}a = _mm512_loadu_pd(s0 + (size_t){t}*{2*n3} + {zt*16});")
                body.append(f"    __m512d p{t}b = _mm512_loadu_pd(s0 + (size_t){t}*{2*n3} + {zt*16+8});")
            else:
                body.append(f"    __m512d p{t}a = _mm512_maskz_loadu_pd({dma}, s0 + (size_t){t}*{2*n3} + {zt*16});")
                body.append(f"    __m512d p{t}b = _mm512_maskz_loadu_pd({dmb}, s0 + (size_t){t}*{2*n3} + {zt*16+8});")
            body.append(f"    vd r{t} = VD(_mm512_permutex2var_pd(p{t}a,IX(6),p{t}b));")
            body.append(f"    vd i{t} = VD(_mm512_permutex2var_pd(p{t}a,IX(7),p{t}b));")
        body.append(f"    TR8(r0,r1,r2,r3,r4,r5,r6,r7); TR8(i0,i1,i2,i3,i4,i5,i6,i7);")
        rr = 8 if not tail else cnt
        for s in range(rr):
            body.append(f"    ST(d + {(zt*8+s)*16}, r{s}); ST(d + {(zt*8+s)*16+8}, i{s});")
        body.append("  }")
    body.append("}")
    conv_in = f"""static void gconv_in_{L}(const double* restrict src, double* restrict dst){{
  {chr(10).join(body)}
}}
"""
    # conv_out_g: inverse
    body = []
    body.append(f"for(int x_=0; x_<{L}; x_++) for(int y_=0; y_<{L}; y_++){{")
    body.append(f"  const double* d = src + (size_t)x_*{SPL} + (size_t)y_*{L*16};")
    body.append(f"  double* o0 = dst + 2*((size_t)(x_*{L}+y_))*{L};")
    for zt in range(nfull + (1 if cnt else 0)):
        tail = (zt == nfull)
        rr = 8 if not tail else cnt
        body.append("  {")
        for s in range(8):
            if s < rr:
                body.append(f"    vd r{s} = LD(d + {(zt*8+s)*16}); vd i{s} = LD(d + {(zt*8+s)*16+8});")
            else:
                body.append(f"    vd r{s} = K(0.0); vd i{s} = K(0.0);")
        body.append(f"    TR8(r0,r1,r2,r3,r4,r5,r6,r7); TR8(i0,i1,i2,i3,i4,i5,i6,i7);")
        for t in range(8):
            if not tail:
                body.append(f"    _mm512_storeu_pd(o0 + (size_t){t}*{2*n3} + {zt*16},   _mm512_permutex2var_pd(MD(r{t}),IX(8),MD(i{t})));")
                body.append(f"    _mm512_storeu_pd(o0 + (size_t){t}*{2*n3} + {zt*16+8}, _mm512_permutex2var_pd(MD(r{t}),IX(9),MD(i{t})));")
            else:
                body.append(f"    _mm512_mask_storeu_pd(o0 + (size_t){t}*{2*n3} + {zt*16}, {dma},  _mm512_permutex2var_pd(MD(r{t}),IX(8),MD(i{t})));")
                body.append(f"    _mm512_mask_storeu_pd(o0 + (size_t){t}*{2*n3} + {zt*16+8}, {dmb}, _mm512_permutex2var_pd(MD(r{t}),IX(9),MD(i{t})));")
        body.append("  }")
    body.append("}")
    conv_out = f"""static void gconv_out_{L}(const double* restrict src, double* restrict dst){{
  {chr(10).join(body)}
}}
"""
    return conv_in + conv_out

def gen_group_driver(L):
    n3 = L*L*L
    SPL = gSPL(L)
    return f"""
static double *GBUF_{L}, *GC_{L};
static void ginit_{L}(void){{
  GBUF_{L} = (double*)xalloc((size_t){L}*{SPL}*8 + 256);
  GC_{L}   = (double*)xalloc((size_t){L}*{SPL}*8 + 256);
}}
static void giter_{L}(double* restrict buf, const double* restrict cb){{
  for(int y=0;y<{L};y++) for(int z=0;z<{L};z++) gA_{L}(buf + (size_t)y*{L*16} + z*16);
  for(int x=0;x<{L};x++){{
    double* pb = buf + (size_t)x*{SPL};
    for(int z=0;z<{L};z++) gB_{L}(pb + z*16);
    for(int y=0;y<{L};y++) gC_{L}(pb + (size_t)y*{L*16}, cb + (size_t)x*{SPL} + (size_t)y*{L*16});
  }}
}}
static void rungrp_{L}(const double* x0, const double* c, double* o1, double* om, long m){{
  gconv_in_{L}(x0, GBUF_{L});
  gconv_in_{L}(c, GC_{L});
  for(long t=1;t<=m;t++){{
    giter_{L}(GBUF_{L}, GC_{L});
    if(t==1){{
      gconv_out_{L}(GBUF_{L}, o1);
      if(m==1){{ memcpy(om, o1, (size_t){n3}*8*16); return; }}
    }}
  }}
  gconv_out_{L}(GBUF_{L}, om);
}}
"""

def gen_run_L_v6(L, grouped):
    n3 = L*L*L
    if grouped:
        return f"""static void run_{L}(long B, long m, const double* x0, const double* c, double* o1, double* om){{
  long g8 = B/8;
  for(long g=0; g<g8; g++)
    rungrp_{L}(x0 + (size_t)g*{16*n3}, c + (size_t)g*{16*n3}, o1 + (size_t)g*{16*n3}, om + (size_t)g*{16*n3}, m);
  for(long b=g8*8; b<B; b++)
    runvol_{L}(x0 + (size_t)b*{2*n3}, c + (size_t)b*{2*n3}, o1 + (size_t)b*{2*n3}, om + (size_t)b*{2*n3}, m);
}}
"""
    return f"""static void run_{L}(long B, long m, const double* x0, const double* c, double* o1, double* om){{
  for(long b=0;b<B;b++)
    runvol_{L}(x0 + (size_t)b*{2*n3}, c + (size_t)b*{2*n3}, o1 + (size_t)b*{2*n3}, om + (size_t)b*{2*n3}, m);
}}
"""

def gen_driver_v6(L, P, NCH, RST, SP, mapA):
    # like v5 driver but without run_ (emitted separately)
    full = gen_driver_v5(L, P, NCH, RST, SP, mapA)
    cut = full.find("static void run_")
    return full[:cut]

def generate_v6(mapA=frozenset(), prefetchA=frozenset(), cpf64=False, gsizes=GSIZES, gmap=None, gpfA=frozenset(), pfB=frozenset(), gpfc=frozenset(), gpfB=frozenset(), gpfC=frozenset(), gdefer=frozenset()):
    gmap = gmap or {}
    tables = set()
    body = []
    for L in SIZES:
        P, NCH, RST, SP = lay(L)
        if L in mapA:
            body.append(gen_fA_v4(L, P, NCH, RST, SP, L in prefetchA, tables, 'H'))
        else:
            body.append(gen_fA_v3(L, P, NCH, RST, SP, L in prefetchA, tables))
        if L == 64:
            body.append(gen_fB_64sw(RST, SP))
            body.append(gen_fC_64sw_v4(RST, SP) if L in mapA else gen_fC_64sw_map(RST, SP, cpf64))
            body.append(gen_convs_64sw(RST, SP))
        elif NCH == 1:
            body.append(gen_fBC_small_v4(L, P, NCH, RST, SP, tables) if L in mapA else gen_fBC_small(L, P, NCH, RST, SP, tables))
            body.append(gen_convs(L, P, NCH, RST, SP))
        else:
            body.append(gen_fB(L, P, NCH, RST, SP, tables))
            body.append(gen_fC_v4(L, P, NCH, RST, SP, tables) if L in mapA else gen_fC(L, P, NCH, RST, SP, tables, 'H'))
            body.append(gen_convs(L, P, NCH, RST, SP))
        body.append(gen_driver_v6(L, P, NCH, RST, SP, L in mapA))
        if L in gsizes:
            body.append(gen_group_codelets(L, tables, mapw=gmap.get(L,'MAPW_H'), pfA=(L in gpfA), pfcNTA=(L in gpfc), pfBnext=(L in gpfB), pfCnext=(L in gpfC), defer_map=(L in gdefer)))
            body.append(gen_group_convs(L))
            body.append(gen_group_driver(L))
        body.append(gen_run_L_v6(L, L in gsizes))
    inits = " ".join(f"init_{L}();" for L in SIZES) + " " + " ".join(f"ginit_{L}();" for L in gsizes)
    disp = ["void init_all(void){ " + inits + " }",
            "void run(int lid, long B, long m, const double* x0, const double* c, double* o1, double* om){",
            "  switch(lid){"]
    for idx, L in enumerate(SIZES):
        disp.append(f"    case {idx}: run_{L}(B,m,x0,c,o1,om); break;")
    disp.append("  }\n}")
    return PRELUDE + "\n" + gen_tables(tables) + "\n" + "\n".join(body) + "\n".join(disp)

# ============ Rader option for primes ============
RADER_SET = set()

def primitive_root(p):
    fact = []
    n = p-1; d = 2
    while d*d <= n:
        if n % d == 0:
            fact.append(d)
            while n % d == 0: n //= d
        d += 1
    if n > 1: fact.append(n)
    for g in range(2, p):
        if all(pow(g, (p-1)//f, p) != 1 for f in fact):
            return g
    raise RuntimeError

def cmul_cc(e, x, cr, ci):
    xr, xi = x
    crf, cif = float(cr), float(ci)
    if abs(cif) < 1e-30:
        f = fmt(cr)
        return (e.v(f"K({f}) * {xr}"), e.v(f"K({f}) * {xi}"))
    if abs(crf) < 1e-30:
        f = fmt(ci)
        return (e.v(f"-(K({f}) * {xi})"), e.v(f"K({f}) * {xr}"))
    a, b = fmt(cr), fmt(ci)
    return (e.v(f"K({a})*{xr} - K({b})*{xi}"), e.v(f"K({a})*{xi} + K({b})*{xr}"))

def dft_rader(e, xs, p):
    N = p - 1
    g = primitive_root(p)
    ginv = pow(g, p-2, p)
    a = [xs[pow(g, q, p)] for q in range(N)]
    A = dft_sl(e, a)
    # constants C_q = DFT_N(w)[q]/N, w_s = W_p^{g^{-s}}
    C = []
    for q in range(N):
        s_ = mp.mpc(0)
        for s in range(N):
            idx = pow(ginv, s, p)
            ang = -2*mp.pi*mp.mpf(idx)/p
            w = mp.mpc(mp.cos(ang), mp.sin(ang))
            ang2 = -2*mp.pi*mp.mpf((q*s) % N)/N
            s_ += w * mp.mpc(mp.cos(ang2), mp.sin(ang2))
        C.append(s_ / N)
    D = [cmul_cc(e, A[q], C[q].real, C[q].imag) for q in range(N)]
    Eo = dft_sl(e, D)
    out = [None]*p
    x0r, x0i = xs[0]
    out[0] = (e.v(f"{x0r} + {A[0][0]}"), e.v(f"{x0i} + {A[0][1]}"))
    for q in range(N):
        er, ei = Eo[(N - q) % N]
        out[pow(ginv, q, p)] = (e.v(f"{x0r} + {er}"), e.v(f"{x0i} + {ei}"))
    return out

# ============ buffered Rader for 23 ============
def emit_rader23_buf(e, loader, storer):
    p = 23; N = 22
    g = primitive_root(p); ginv = pow(g, p-2, p)
    # constants C_q
    C = []
    for q in range(N):
        s_ = mp.mpc(0)
        for s in range(N):
            idx = pow(ginv, s, p)
            ang = -2*mp.pi*mp.mpf(idx)/p
            w = mp.mpc(mp.cos(ang), mp.sin(ang))
            ang2 = -2*mp.pi*mp.mpf((q*s) % N)/N
            s_ += w * mp.mpc(mp.cos(ang2), mp.sin(ang2))
        C.append(s_ / N)
    x0r, x0i = loader(e, 0)
    e.raw(f"vd X0R_ = {x0r}, X0I_ = {x0i};")
    # a_q = x[g^q mod p]
    aidx = [pow(g, q, p) for q in range(N)]
    tagc = [0]
    def dft22_buf(load_elem, store_out):
        """load_elem(e, q) -> pair; store_out(e, k, pair) with k in 0..21 natural order"""
        A, B = 2, 11
        tg = tagc[0]; tagc[0] += 1
        e.raw(f"vd T0R{tg}[11], T0I{tg}[11], T1R{tg}[11], T1I{tg}[11];")
        for n2 in range(B):
            i0 = (B*0 + A*n2) % N
            i1 = (B*1 + A*n2) % N
            u = load_elem(e, i0); v = load_elem(e, i1)
            s0 = cadd(e, u, v); s1 = csub(e, u, v)
            e.raw(f"T0R{tg}[{n2}] = {s0[0]}; T0I{tg}[{n2}] = {s0[1]}; T1R{tg}[{n2}] = {s1[0]}; T1I{tg}[{n2}] = {s1[1]};")
        e.raw('__asm__("" ::: "memory");')
        for k1 in range(A):
            xs = [(e.v(f"T{k1}R{tg}[{n2}]"), e.v(f"T{k1}I{tg}[{n2}]")) for n2 in range(B)]
            ys = dft_sl(e, xs)   # sym-11
            for k2 in range(B):
                k = crt_index(N, A, B, k1, k2)
                store_out(e, k, ys[k2])
    # stage 1: D[q] = DFT22(a)[q] * C[q]
    e.raw("vd DR[22], DI[22];")
    def load_a(e2, q):  # q is dft22 input position
        return loader(e2, aidx[q])
    def store_D(e2, k, pr):
        r, i = cmul_cc(e2, pr, C[k].real, C[k].imag)
        e2.raw(f"DR[{k}] = {r}; DI[{k}] = {i};")
    dft22_buf(load_a, store_D)
    e.raw('__asm__("" ::: "memory");')
    # X0 = x0 + A[0] ; A[0] = D[0]/C[0], C0 real
    c0 = float(C[0].real)
    e.raw(f"vd x0outr_ = X0R_ + K({(1.0/c0).hex()})*DR[0]; vd x0outi_ = X0I_ + K({(1.0/c0).hex()})*DI[0];")
    storer.store(e, 0, "x0outr_", "x0outi_")
    # stage 2: E = DFT22(D); X[ginv^q] = x0 + E[(22-q)%22]
    outmap = {}
    for q in range(N):
        outmap[(N - q) % N] = pow(ginv, q, p)
    def load_D(e2, q):
        return (e2.v(f"DR[{q}]"), e2.v(f"DI[{q}]"))
    def store_X(e2, k, pr):
        kk = outmap[k]
        r = e2.v(f"X0R_ + {pr[0]}"); i = e2.v(f"X0I_ + {pr[1]}")
        storer.store(e2, kk, r, i)
    dft22_buf(load_D, store_X)

RADER23BUF = [False]
UNRH = 3
SL_SET = set()

def dft_rader23_sl(e, xs):
    outs = [None]*23
    class _St:
        def store(self, e2, k, r, i):
            outs[k] = (r, i)
    def loader(e2, j):
        return xs[j]
    emit_rader23_buf(e, loader, _St())
    return outs
