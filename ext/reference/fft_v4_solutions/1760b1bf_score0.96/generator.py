#!/usr/bin/env python3
# Generator for implementation.c : specialized batched 3D FFT + nonlinear map
import math
from fractions import Fraction
from math import gcd

SIZES = [6, 8, 13, 17, 23, 36, 45, 64]

def ceil4(x): return (x + 3) // 4 * 4

LAYOUT = {}
for L in SIZES:
    S2 = ceil4(L)
    if L == 64: S2 = 68          # break L1 set conflicts for column loads
    R1 = ceil4(L)                # padded rows per slab
    S1 = R1 * S2
    S1 += 4                      # make S1*16/64 odd -> perfect L1 set spread for axis-0
    LAYOUT[L] = (S2, R1, S1)


LAYOUTG = {}
for L in SIZES:
    S2g = L + (4 if L == 64 else 0)
    S1g = L * S2g
    if S1g % 64 in (0, 16, 32, 48, 8, 24, 40, 56):
        # make set-step odd-ish: prefer odd step
        S1g += 1
    LAYOUTG[L] = (S2g, S1g)

# ---------------- IR ----------------
class Node:
    __slots__ = ('op','args','k','nid','uses')
    def __init__(self, op, args, k, nid):
        self.op, self.args, self.k, self.nid, self.uses = op, args, k, nid, 0

class Graph:
    def __init__(self):
        self.memo = {}
        self.nodes = []
    def mk(self, op, args, k=None):
        key = (op, tuple(a.nid for a in args), k)
        n = self.memo.get(key)
        if n is None:
            n = Node(op, args, k, len(self.nodes))
            self.nodes.append(n)
            self.memo[key] = n
        return n

G = None
def LD(i):     return G.mk('ld', (), i)
def ADD(a,b):  return G.mk('add', (a,b))
def SUB(a,b):  return G.mk('sub', (a,b))
def SWAP(a):
    if a.op == 'swap': return a.args[0]
    return G.mk('swap', (a,))
def MUL(a,k0,k1):
    if (k0,k1) == (1.0,1.0): return a
    if a.op == 'mul':
        return G.mk('mul', (a.args[0],), (a.k[0]*k0, a.k[1]*k1))
    return G.mk('mul', (a,), (float(k0),float(k1)))
def FMA(v,k0,k1,acc):   # v*(k0,k1) + acc
    if (k0,k1) == (1.0,1.0):   return ADD(acc, v)
    if (k0,k1) == (-1.0,-1.0): return SUB(acc, v)
    if v.op == 'mul':
        return FMA(v.args[0], v.k[0]*k0, v.k[1]*k1, acc)
    return G.mk('fma', (v,acc), (float(k0),float(k1)))

SQ2 = math.sqrt(2.0)/2.0
SQ3 = math.sqrt(3.0)/2.0
def snap(x):
    for v in (0.0, 0.5, -0.5, 1.0, -1.0, SQ2, -SQ2, SQ3, -SQ3):
        if abs(x - v) < 1e-12: return v
    return x

def cmulW(a, num, den):
    """multiply by W = exp(-2*pi*i*num/den)"""
    f = Fraction(num % den, den)
    if f == 0: return a
    if f == Fraction(1,2): return MUL(a, -1.0, -1.0)
    if f == Fraction(1,4): return MUL(SWAP(a), 1.0, -1.0)    # * -i
    if f == Fraction(3,4): return MUL(SWAP(a), -1.0, 1.0)    # * +i
    th = 2.0*math.pi*f.numerator/f.denominator
    c = snap(math.cos(th)); d = snap(-math.sin(th))          # W = c + i*d
    return FMA(SWAP(a), -d, d, MUL(a, c, c))

# ---------------- FFT builders ----------------
FACT = {6:(2,3), 8:(2,4), 9:(3,3), 36:(4,9), 45:(9,5), 64:(8,8), 16:(4,4), 32:(4,8)}

def fft(N, x):
    assert len(x) == N
    if N == 1: return list(x)
    if N == 2: return [ADD(x[0],x[1]), SUB(x[0],x[1])]
    if N == 4:
        t0=ADD(x[0],x[2]); t1=SUB(x[0],x[2]); t2=ADD(x[1],x[3]); t3=SUB(x[1],x[3])
        sw=SWAP(t3)
        return [ADD(t0,t2), FMA(sw,1.0,-1.0,t1), SUB(t0,t2), FMA(sw,-1.0,1.0,t1)]
    if N in (3,5,7,11,13,17,19,23):
        return prime_sym(N, x)
    P,Q = FACT[N]
    if gcd(P,Q) == 1: return pfa(N,P,Q,x)
    return ct(N,P,Q,x)

def prime_sym(N, x):
    h = (N-1)//2
    s = {}; d = {}
    for j in range(1,h+1):
        s[j] = ADD(x[j], x[N-j]); d[j] = SUB(x[j], x[N-j])
    X = [None]*N
    t = x[0]
    for j in range(1,h+1): t = ADD(t, s[j])
    X[0] = t
    for k in range(1,h+1):
        A = x[0]; B = None
        for j in range(1,h+1):
            ang = 2.0*math.pi*((j*k) % N)/N
            c = snap(math.cos(ang)); sn = snap(math.sin(ang))
            A = FMA(s[j], c, c, A)
            B = MUL(d[j], sn, sn) if B is None else FMA(d[j], sn, sn, B)
        sw = SWAP(B)
        X[k]   = FMA(sw,  1.0, -1.0, A)   # A - i*B
        X[N-k] = FMA(sw, -1.0,  1.0, A)   # A + i*B
    return X

def ct(N,P,Q,x):
    U = [fft(P, [x[Q*a+b] for a in range(P)]) for b in range(Q)]
    X = [None]*N
    for c in range(P):
        t = [cmulW(U[b][c], b*c, N) for b in range(Q)]
        Y = fft(Q, t)
        for dd in range(Q):
            X[c + P*dd] = Y[dd]
    return X

def crt(c,P,e,Q,N):
    for k in range(N):
        if k % P == c and k % Q == e: return k
    raise RuntimeError

def pfa(N,P,Q,x):
    U = [fft(P, [x[(Q*a+P*b) % N] for a in range(P)]) for b in range(Q)]
    X = [None]*N
    for c in range(P):
        Y = fft(Q, [U[b][c] for b in range(Q)])
        for e in range(Q):
            X[crt(c,P,e,Q,N)] = Y[e]
    return X

# ---------------- emission ----------------
def fhex(x):
    if x == int(x) and abs(x) < 16: return '%.1f' % x
    return float(x).hex()

def render_graph(Xs, width, ldexpr, stname):
    """emit SSA lines for graph with outputs Xs; returns (lines, outnames)"""
    V = {8:'V8',4:'V4',2:'V2'}[width]
    SW = {8:'vswap8',4:'vswap4',2:'vswap2'}[width]
    def pair(k0,k1):
        if k0 == k1: return fhex(k0)
        elems = []
        for _ in range(width//2): elems += [fhex(k0), fhex(k1)]
        return '(%s){%s}' % (V, ','.join(elems))
    order = []; seen = set()
    def visit(n):
        if n.nid in seen: return
        seen.add(n.nid)
        for a in n.args: visit(a)
        order.append(n)
    for X in Xs: visit(X)
    lines = []; nameof = {}
    for n in order:
        t = '%s%d' % (stname, n.nid)
        nameof[n.nid] = t
        if n.op == 'ld':
            lines.append('%s %s = %s;' % (V, t, ldexpr(n.k)))
        elif n.op == 'add':
            lines.append('%s %s = %s + %s;' % (V, t, nameof[n.args[0].nid], nameof[n.args[1].nid]))
        elif n.op == 'sub':
            lines.append('%s %s = %s - %s;' % (V, t, nameof[n.args[0].nid], nameof[n.args[1].nid]))
        elif n.op == 'swap':
            lines.append('%s %s = %s(%s);' % (V, t, SW, nameof[n.args[0].nid]))
        elif n.op == 'mul':
            lines.append('%s %s = %s * %s;' % (V, t, nameof[n.args[0].nid], pair(*n.k)))
        elif n.op == 'fma':
            lines.append('%s %s = %s * %s + %s;' % (V, t, nameof[n.args[0].nid], pair(*n.k), nameof[n.args[1].nid]))
    return lines, [nameof[X.nid] for X in Xs]

PRIME_PHASE = (13, 17, 23)

def emit_prime_codelet(N, ldexpr, stexpr, out, pwload=None, post=''):
    """Phase-structured symmetric prime DFT: register-resident s/consts per sweep.
    pwload: None (plain loads) or 'fast' -> z=x+c then pw at load time."""
    h = (N-1)//2
    kb = [(1,min(8,h))] + ([(9,h)] if h > 8 else [])
    B = []
    B.append('V8 db[%d]; V8 ab[%d];' % (h+1, h+1))
    # pre-phase
    if pwload is None:
        B.append('V8 x0 = %s;' % ldexpr(0))
        for j in range(1, h+1):
            B.append('V8 s%d; { V8 a_ = %s, b_ = %s; s%d = a_+b_; db[%d] = a_-b_; }'
                     % (j, ldexpr(j), ldexpr(N-j), j, j))
    else:
        B.append('V8 x0 = pw_full_fast(%s + %s);' % (ldexpr(0), ldexpr(0).replace('(p','(c')))
        for j in range(1, h+1):
            B.append('V8 s%d; { V8 za_ = %s + %s; V8 zb_ = %s + %s; V8 a_, b_; pw_pair_fast(za_, zb_, &a_, &b_); s%d = a_+b_; db[%d] = a_-b_; }'
                     % (j, ldexpr(j), ldexpr(j).replace('(p','(c'), ldexpr(N-j), ldexpr(N-j).replace('(p','(c'), j, j))
    B.append('V8 tsum_ = %s;' % '+'.join(['x0']+['s%d'%j for j in range(1,h+1)]))
    B.append(stexpr(0, 'tsum_'))
    # A phase
    for r in range(1, h+1):
        B.append('V8 C%d = (V8)_mm512_set1_pd(%s);' % (r, fhex(math.cos(2*math.pi*r/N))))
    def cidx(j,k):
        r = (j*k) % N
        return (N-r, -1) if r > h else (r, 1)
    for (k0,k1) in kb:
        for k in range(k0,k1+1):
            B.append('V8 A%d = x0;' % k)
        for j in range(1, h+1):
            for k in range(k0,k1+1):
                r,_ = cidx(j,k)
                B.append('A%d = A%d + s%d*C%d;' % (k,k,j,r))
        for k in range(k0,k1+1):
            B.append('ab[%d] = A%d;' % (k,k))
    # B phase
    for r in range(1, h+1):
        B.append('V8 S%d = (V8)_mm512_set1_pd(%s);' % (r, fhex(math.sin(2*math.pi*r/N))))
    for j in range(1, h+1):
        B.append('V8 d%d = db[%d];' % (j,j))
    B.append('static const V8 SGpm_ = {1,-1,1,-1,1,-1,1,-1};')
    for (k0,k1) in kb:
        for k in range(k0,k1+1):
            B.append('V8 B%d = (V8)_mm512_setzero_pd();' % k)
        for j in range(1, h+1):
            for k in range(k0,k1+1):
                r,sg = cidx(j,k)
                B.append('B%d = B%d %s d%d*S%d;' % (k,k,'+' if sg>0 else '-',j,r))
        for k in range(k0,k1+1):
            B.append('V8 sw%d_ = vswap8(B%d); V8 Ak%d_ = ab[%d];' % (k,k,k,k))
            B.append('V8 Xa%d_ = Ak%d_ + sw%d_*SGpm_; V8 Xb%d_ = Ak%d_ - sw%d_*SGpm_;' % (k,k,k,k,k,k))
            B.append(stexpr(k, 'Xa%d_' % k))
            B.append(stexpr(N-k, 'Xb%d_' % k))
    txt = '\n  '.join(B)
    if callable(post): post = post()
    return ('static inline __attribute__((always_inline)) void %s {\n  %s\n  %s\n}\n'
            % (out, txt, post))

STAGED = {36:(4,9,'pfa'), 45:(9,5,'pfa'), 64:(4,16,'ct')}

def emit_codelet(N, width, _unused, ldexpr, stexpr, out, pre='', post='', blockpre=None):
    """stexpr(k, exprname) -> C statement string.  Staged for big N."""
    global G
    body = []
    if N in STAGED and width == 8:
        P,Q,kind = STAGED[N]
        body.append('V8 tmp[%d];' % N)
        for b in range(Q):
            G = Graph()
            if kind == 'ct':
                xs = [LD(a) for a in range(P)]
                Xs = fft(P, xs)
                Xs = [cmulW(Xs[c], b*c, N) for c in range(P)]
                idx = [Q*a+b for a in range(P)]
            else:
                xs = [LD(a) for a in range(P)]
                Xs = fft(P, xs)
                idx = [(Q*a+P*b) % N for a in range(P)]
            if blockpre is not None:
                blines, bld = blockpre(b, idx)
                body += ['  '+l for l in blines]
                lines, outn = render_graph(Xs, width, bld, 's%d_'%b)
            else:
                lines, outn = render_graph(Xs, width, lambda i,idx=idx: ldexpr(idx[i]), 's%d_'%b)
            body.append('// stage1 b=%d' % b)
            body += ['  '+l for l in lines]
            for c in range(P):
                body.append('  tmp[%d] = %s;' % (c*Q+b, outn[c]))
        for c in range(P):
            G = Graph()
            xs = [LD(b) for b in range(Q)]
            Ys = fft(Q, xs)
            if kind == 'ct':
                kidx = [c + P*d for d in range(Q)]
            else:
                kidx = [crt(c,P,e,Q,N) for e in range(Q)]
            lines, outn = render_graph(Ys, width, lambda i,c=c: 'tmp[%d]' % (c*Q+i), 'f%d_'%c)
            body.append('// stage2 c=%d' % c)
            body += ['  '+l for l in lines]
            for e in range(Q):
                body.append('  ' + stexpr(kidx[e], outn[e]))
    else:
        G = Graph()
        xs = [LD(i) for i in range(N)]
        Xs = fft(N, xs)
        lines, outn = render_graph(Xs, width, ldexpr, 't')
        # place stores as soon as their operand is defined (reduces live range)
        defpos = {}
        for i,l in enumerate(lines):
            nm = l.split(' ',2)[1]
            defpos[nm] = i
        ready = sorted(range(N), key=lambda k: defpos[outn[k]])
        ins = {}
        for k in ready:
            ins.setdefault(defpos[outn[k]], []).append(k)
        out_lines = []
        for i,l in enumerate(lines):
            out_lines.append(l)
            if i in ins:
                for k in ins[i]:
                    out_lines.append(stexpr(k, outn[k]))
        body += out_lines
    txt = '\n  '.join(body)
    if callable(post): post = post()
    return ('static inline __attribute__((always_inline)) void %s {\n  %s\n  %s\n  %s\n}\n'
            % (out, pre, txt, post))

def opcount(N):
    global G
    G = Graph()
    xs=[LD(i) for i in range(N)]
    fft(N,xs)
    from collections import Counter
    c = Counter(n.op for n in G.nodes)
    return dict(c)

if __name__ == '__main__':
    for N in SIZES:
        print(N, opcount(N), LAYOUT[N])

# ======================= full file emission =======================
FACT[64] = (4,16)   # best op count

HEADER = r'''
// Auto-generated: specialized batched 3D complex FFT + z/(1+|z|) iteration.
// Single-threaded, AVX-512.
#include <immintrin.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#ifdef __linux__
#include <sys/mman.h>
#endif

#define OPT_SCHED __attribute__((optimize("schedule-insns","sched-pressure","rename-registers")))
#define OPT_PLAIN
typedef double V8 __attribute__((vector_size(64)));
typedef double V4 __attribute__((vector_size(32)));
typedef double V2 __attribute__((vector_size(16)));

static inline V8 vswap8(V8 x){ return __builtin_shufflevector(x,x,1,0,3,2,5,4,7,6); }
static inline V4 vswap4(V4 x){ return __builtin_shufflevector(x,x,1,0,3,2); }
static inline V2 vswap2(V2 x){ return __builtin_shufflevector(x,x,1,0); }

#define LDA(p) (*(const V8*)(p))
typedef V8 V8u __attribute__((aligned(8)));
#define LDU(p) (*(const V8u*)(p))
#define STU(p,v) (*(V8u*)(p) = (v))
#define STA(p,v) (*(V8*)(p) = (v))

// ---- 4x4 complex transpose (8 permutes) ----
static const long long IDXA_[8] = {0,1,8,9,2,3,10,11};
static const long long IDXB_[8] = {4,5,12,13,6,7,14,15};
static const long long IDXP_[8] = {0,2,4,6,8,10,12,14};
static const long long IDXL_[8] = {0,0,1,1,2,2,3,3};
static const long long IDXH_[8] = {4,4,5,5,6,6,7,7};
#define TRANSP4(a0,a1,a2,a3,o0,o1,o2,o3) do{ \
  __m512i tp_ia_ = _mm512_load_si512((const void*)IDXA_); \
  __m512i tp_ib_ = _mm512_load_si512((const void*)IDXB_); \
  __m512d tp_b0_ = _mm512_permutex2var_pd((__m512d)(a0), tp_ia_, (__m512d)(a2)); \
  __m512d tp_b1_ = _mm512_permutex2var_pd((__m512d)(a1), tp_ia_, (__m512d)(a3)); \
  __m512d tp_b2_ = _mm512_permutex2var_pd((__m512d)(a0), tp_ib_, (__m512d)(a2)); \
  __m512d tp_b3_ = _mm512_permutex2var_pd((__m512d)(a1), tp_ib_, (__m512d)(a3)); \
  o0 = (V8)_mm512_permutex2var_pd(tp_b0_, tp_ia_, tp_b1_); \
  o1 = (V8)_mm512_permutex2var_pd(tp_b0_, tp_ib_, tp_b1_); \
  o2 = (V8)_mm512_permutex2var_pd(tp_b2_, tp_ia_, tp_b3_); \
  o3 = (V8)_mm512_permutex2var_pd(tp_b2_, tp_ib_, tp_b3_); \
}while(0)

// ---- pointwise z/(1+|z|) ----
// PW_STYLE 1: hw-sqrt + Newton recip ; 2: Newton rsqrt + hw div
#ifndef PW_STYLE
#define PW_STYLE 1
#endif
#if PW_STYLE == 1
#define PW_CORE(sp, w, FULL) \
  V8 u_ = (V8)_mm512_sqrt_pd((__m512d)sp); \
  V8 a_ = u_ + 1.0; \
  w = (V8)_mm512_cvtps_pd(_mm256_rcp_ps(_mm512_cvtpd_ps((__m512d)a_))); \
  w = w + w*(1.0 - a_*w); \
  w = w + w*(1.0 - a_*w); \
  if (FULL) { w = w + w*(1.0 - a_*w); }
#else
#define PW_CORE(sp, w, FULL) \
  sp = (V8)_mm512_max_pd((__m512d)sp, _mm512_set1_pd(1e-36)); \
  V8 r_ = (V8)_mm512_cvtps_pd(_mm256_rsqrt_ps(_mm512_cvtpd_ps((__m512d)sp))); \
  V8 h_ = 0.5*sp; \
  V8 q1_ = r_*r_;  r_ = r_*(1.5 - h_*q1_); \
  V8 q2_ = r_*r_;  r_ = r_*(1.5 - h_*q2_); \
  V8 y_ = sp*r_; \
  if (FULL) { y_ = y_ + (0.5*r_)*(sp - y_*y_); } \
  V8 a_ = y_ + 1.0; \
  w = (V8)_mm512_div_pd(_mm512_set1_pd(1.0), (__m512d)a_);
#endif

static inline V8 pw_core_apply(V8 z, int full){
  V8 t = z*z;
  V8 sp = t + vswap8(t);
  V8 w;
  PW_CORE(sp, w, full)
  return z*w;
}
static inline V8 pw_full(V8 z){ return pw_core_apply(z, 1); }
static inline V8 pw_full_fast(V8 z){ return pw_core_apply(z, 0); }
static inline void pw_pair_gen(V8 z0, V8 z1, V8 *o0, V8 *o1, int full){
  __m512i ip = _mm512_load_si512((const void*)IDXP_);
  __m512i il = _mm512_load_si512((const void*)IDXL_);
  __m512i ih = _mm512_load_si512((const void*)IDXH_);
  V8 t0 = z0*z0, t1 = z1*z1;
  V8 s0 = t0 + vswap8(t0), s1 = t1 + vswap8(t1);
  V8 sp = (V8)_mm512_permutex2var_pd((__m512d)s0, ip, (__m512d)s1);
  V8 w;
  PW_CORE(sp, w, full)
  V8 w0 = (V8)_mm512_permutexvar_pd(il, (__m512d)w);
  V8 w1 = (V8)_mm512_permutexvar_pd(ih, (__m512d)w);
  *o0 = z0*w0; *o1 = z1*w1;
}
static inline void pw_pair(V8 z0, V8 z1, V8 *o0, V8 *o1){ pw_pair_gen(z0,z1,o0,o1,1); }
static inline void pw_pair_fast(V8 z0, V8 z1, V8 *o0, V8 *o1){ pw_pair_gen(z0,z1,o0,o1,0); }
'''

SCHED = {6:'OPT_PLAIN', 8:'OPT_SCHED', 13:'OPT_SCHED', 17:'OPT_SCHED', 23:'OPT_SCHED', 36:'OPT_SCHED', 45:'OPT_PLAIN', 64:'OPT_PLAIN'}

def gen_for_L(L):
    S2,R1,S1 = LAYOUT[L]
    OA = SCHED[L]
    ng = L//4 if (L % 4 == 1) else R1//4
    EMIT = (lambda n,w,u,ld,st,out,pre='',post='',blockpre=None:
            emit_prime_codelet(n, ld, st, out, post=post) if (n in PRIME_PHASE and w == 8 and blockpre is None and pre == '')
            else emit_codelet(n,w,u,ld,st,out,pre,post,blockpre))
    C4 = ceil4(L)//4         # number of 4-complex chunks covering the L columns
    parts = []
    # --- A1 codelet: columns along j1, contiguous 4-complex lanes over j2
    parts.append(EMIT(L, 8, '',
        lambda i: 'LDA(p+%d)' % (i*2*S2),
        lambda k,e: 'STA(p+%d, %s);' % (k*2*S2, e),
        out='fft%d_a1(double *restrict p)' % L))
    # variant applying z=x+c then pw at load time (iterations >= 2)
    if L in STAGED:
        def blockpre(b, idx):
            bl = []
            for t,i in enumerate(idx):
                bl.append('V8 zb%d_%d = LDA(p+%d) + LDA(c+%d);' % (b, t, i*2*S2, i*2*S2))
            t = 0
            P = len(idx)
            while t+1 < P:
                bl.append('V8 wb%d_%d, wb%d_%d; pw_pair_fast(zb%d_%d,zb%d_%d,&wb%d_%d,&wb%d_%d);'
                          % (b,t,b,t+1,b,t,b,t+1,b,t,b,t+1))
                t += 2
            if P % 2:
                bl.append('V8 wb%d_%d = pw_full_fast(zb%d_%d);' % (b,P-1,b,P-1))
            return bl, (lambda i, b=b: 'wb%d_%d' % (b, i))
        parts.append(EMIT(L, 8, '',
            None,
            lambda k,e: 'STA(p+%d, %s);' % (k*2*S2, e),
            out='fft%d_a1pw(double *restrict p, const double *restrict c)' % L,
            blockpre=blockpre))
    elif L in PRIME_PHASE:
        parts.append(emit_prime_codelet(L,
            lambda i: 'LDA(p+%d)' % (i*2*S2),
            lambda k,e: 'STA(p+%d, %s);' % (k*2*S2, e),
            out='fft%d_a1pw(double *restrict p, const double *restrict c)' % L,
            pwload='fast'))
    else:
        # pair loads as (j, N-j) so values are consumed immediately by the s/d stage
        pre_lines = ['V8 z0 = LDA(p) + LDA(c); V8 w0 = pw_full_fast(z0);']
        j = 1
        while j < L - j:
            pre_lines.append('V8 z%d = LDA(p+%d) + LDA(c+%d);' % (j, j*2*S2, j*2*S2))
            pre_lines.append('V8 z%d = LDA(p+%d) + LDA(c+%d);' % (L-j, (L-j)*2*S2, (L-j)*2*S2))
            pre_lines.append('V8 w%d, w%d; pw_pair_fast(z%d,z%d,&w%d,&w%d);' % (j,L-j,j,L-j,j,L-j))
            j += 1
        if L % 2 == 0:
            h = L//2
            pre_lines.append('V8 z%d = LDA(p+%d) + LDA(c+%d); V8 w%d = pw_full_fast(z%d);' % (h,h*2*S2,h*2*S2,h,h))
        parts.append(EMIT(L, 8, '',
            lambda i: 'w%d' % i,
            lambda k,e: 'STA(p+%d, %s);' % (k*2*S2, e),
            out='fft%d_a1pw(double *restrict p, const double *restrict c)' % L,
            pre='\n  '.join(pre_lines)))
    # --- io codelet for A2 (in/out on V8 array)
    parts.append(EMIT(L, 8, '',
        lambda i: 'io[%d]' % i,
        lambda k,e: 'io[%d] = %s;' % (k,e),
        out='fft%d_io(V8 *restrict io)' % L))
    # --- B codelet: plain axis-0 FFT, in place ---
    parts.append(EMIT(L, 8, '',
        lambda i: 'LDA(p+%d)' % (i*2*S1),
        lambda k,e: 'STA(p+%d, %s);' % (k*2*S1, e),
        out='fft%d_b(double *restrict p)' % L))
    # --- B codelet with fused c add + fast pw at store ---
    pend = [None]
    def st_bpw(k, e):
        s = 'V8 y%d = %s + LDA(c+%d);' % (k, e, k*2*S1)
        if pend[0] is None:
            pend[0] = k
            return s
        k0 = pend[0]; pend[0] = None
        return (s + ' { V8 r0_,r1_; pw_pair_fast(y%d,y%d,&r0_,&r1_); STA(p+%d,r0_); STA(p+%d,r1_); }'
                % (k0, k, k0*2*S1, k*2*S1))
    parts.append(EMIT(L, 8, '',
        lambda i: 'LDA(p+%d)' % (i*2*S1),
        st_bpw,
        out='fft%d_bpw(double *restrict p, const double *restrict c)' % L,
        post=lambda: ('STA(p+%d, pw_full_fast(y%d));' % (pend[0]*2*S1, pend[0])) if pend[0] is not None else ''))
    # --- fused A2 for L=64: Tin groups -> stage1 -> tmp; stage2 -> io2; Tout ---
    if L == 64:
        P,Q,kind = STAGED[64]   # (4,16,'ct')
        a2f = []
        a2f.append('static %s void fft64_a2f(double *restrict p){' % OA)
        a2f.append('  V8 tmp[64]; V8 io2[64];')
        for g in range(4):
            # load Tin blocks {g, g+4, g+8, g+12}: rows 0..3, cols 4t..4t+3
            for bi, t in enumerate((g, g+4, g+8, g+12)):
                a2f.append('  { V8 a0=LDA(p+%d), a1=LDA(p+%d), a2_=LDA(p+%d), a3=LDA(p+%d);'
                           % (t*8 + 0*2*S2, t*8 + 1*2*S2, t*8 + 2*2*S2, t*8 + 3*2*S2))
                a2f.append('    TRANSP4(a0,a1,a2_,a3, cg%d_%d, cg%d_%d, cg%d_%d, cg%d_%d); }'
                           % (g,4*bi,g,4*bi+1,g,4*bi+2,g,4*bi+3))
            # declare col regs before use
        # declarations must precede; restructure: declare all cg vars first
        a2f = []
        a2f.append('static %s void fft64_a2f(double *restrict p){' % OA)
        a2f.append('  V8 tmp[64]; V8 io2[64];')
        for g in range(4):
            decl = ', '.join('cg%d' % i for i in range(16))
            a2f.append('  { V8 %s;' % decl)
            for bi, t in enumerate((g, g+4, g+8, g+12)):
                a2f.append('    { V8 a0=LDA(p+%d), a1=LDA(p+%d), a2_=LDA(p+%d), a3=LDA(p+%d);'
                           % (t*8 + 0*2*S2, t*8 + 1*2*S2, t*8 + 2*2*S2, t*8 + 3*2*S2))
                a2f.append('      TRANSP4(a0,a1,a2_,a3, cg%d, cg%d, cg%d, cg%d); }'
                           % (4*bi, 4*bi+1, 4*bi+2, 4*bi+3))
            # cg[k] holds column index: block (g + 4*bi) covers cols 4*(g+4*bi) .. +3
            # stage1 blocks b = 4g + j (j=0..3): inputs cols {b, 16+b, 32+b, 48+b}
            #   col b      = 4g+j      -> block bi=0 (cols 4g..4g+3) lane j  -> cg[j]
            #   col 16+b   = 16+4g+j   -> block bi=1 lane j                  -> cg[4+j]
            #   col 32+b                -> cg[8+j] ;  col 48+b -> cg[12+j]
            for j in range(4):
                b = 4*g + j
                G2 = Graph()
                import_gen = None
                global G
                G = G2
                xs = [LD(a) for a in range(4)]
                Xs = fft(4, xs)
                Xs = [cmulW(Xs[c_], b*c_, 64) for c_ in range(4)]
                lines, outn = render_graph(Xs, 8, lambda i, j=j: 'cg%d' % (4*i + j), 'sg%d_%d_' % (g, j))
                a2f += ['    '+l for l in lines]
                for c_ in range(4):
                    a2f.append('    tmp[%d] = %s;' % (c_*16 + b, outn[c_]))
            a2f.append('  }')
        # stage2: 4 blocks of FFT16 over tmp[c*16 + 0..15] -> io2[c + 4d]
        for c_ in range(4):
            G = Graph()
            xs = [LD(bb) for bb in range(16)]
            Ys = fft(16, xs)
            lines, outn = render_graph(Ys, 8, lambda i, c_=c_: 'tmp[%d]' % (c_*16 + i), 'f2%d_' % c_)
            a2f.append('  // stage2 c=%d' % c_)
            a2f += ['  '+l for l in lines]
            for dd in range(16):
                a2f.append('  io2[%d] = %s;' % (c_ + 4*dd, outn[dd]))
        # Tout
        a2f.append('  for(int t=0;t<16;++t){')
        a2f.append('    V8 b0,b1,b2,b3;')
        a2f.append('    TRANSP4(io2[4*t],io2[4*t+1],io2[4*t+2],io2[4*t+3], b0,b1,b2,b3);')
        a2f.append('    STA(p+t*8+%d,b0); STA(p+t*8+%d,b1); STA(p+t*8+%d,b2); STA(p+t*8+%d,b3);'
                   % (0*2*S2, 1*2*S2, 2*2*S2, 3*2*S2))
        a2f.append('  }')
        a2f.append('}')
        parts.append('\n'.join(a2f)+'\n')
    # --- A2 driver: 4-row groups, transpose in/out
    TAIL1 = L % 4 == 1
    if TAIL1:
        parts.append(EMIT(L, 2, '',
            lambda i: '(*(const V2*)(p+%d))' % (2*i),
            lambda k,e: '(*(V2*)(p+%d)) = %s;' % (2*k, e),
            out='fft%d_row2(double *restrict p)' % L))
    a2 = []
    a2.append('static ' + OA + ' void fft%d_a2(double *restrict p){' % L)
    a2.append('  V8 io[%d];' % (4*C4))
    a2.append('  for(int t=0;t<%d;++t){' % C4)
    a2.append('    V8 a0=LDA(p+t*8+%d), a1=LDA(p+t*8+%d), a2_=LDA(p+t*8+%d), a3=LDA(p+t*8+%d);'
              % (0*2*S2, 1*2*S2, 2*2*S2, 3*2*S2))
    a2.append('    TRANSP4(a0,a1,a2_,a3, io[4*t], io[4*t+1], io[4*t+2], io[4*t+3]);')
    a2.append('  }')
    a2.append('  fft%d_io(io);' % L)
    a2.append('  for(int t=0;t<%d;++t){' % C4)
    a2.append('    V8 b0,b1,b2,b3;')
    a2.append('    TRANSP4(io[4*t],io[4*t+1],io[4*t+2],io[4*t+3], b0,b1,b2,b3);')
    a2.append('    STA(p+t*8+%d,b0); STA(p+t*8+%d,b1); STA(p+t*8+%d,b2); STA(p+t*8+%d,b3);'
              % (0*2*S2, 1*2*S2, 2*2*S2, 3*2*S2))
    a2.append('  }')
    a2.append('}')
    parts.append('\n'.join(a2)+'\n')
    # --- passes
    p = []
    p.append('static ' + OA + ' void passA_%d(double *restrict x, const double *restrict c, int withpw){' % L)
    p.append('  for(int s=0;s<%d;++s){' % L)
    p.append('    double *restrict sl = x + (long)s*%d;' % (2*S1))
    p.append('    const double *restrict cl = c + (long)s*%d;' % (2*S1))
    p.append('    if (withpw) { for(int ch=0;ch<%d;++ch) fft%d_a1pw(sl + ch*8, cl + ch*8); }' % (C4, L))
    p.append('    else        { for(int ch=0;ch<%d;++ch) fft%d_a1(sl + ch*8); }' % (C4, L))
    if L == 64:
        p.append('    for(int g=0;g<16;++g) fft64_a2f(sl + (long)g*%d);' % (4*2*S2))
    else:
        p.append('    for(int g=0;g<%d;++g) fft%d_a2(sl + (long)g*%d);' % (ng, L, 4*2*S2))
    if L % 4 == 1:
        p.append('    fft%d_row2(sl + %d);' % (L, (L-1)*2*S2))
    p.append('  }')
    p.append('}')
    p.append('static ' + OA + ' void passAE_%d(double *restrict x, const double *restrict c){' % L)
    p.append('  for(int s=0;s<%d;++s){' % L)
    p.append('    double *restrict sl = x + (long)s*%d;' % (2*S1))
    p.append('    const double *restrict cl = c + (long)s*%d;' % (2*S1))
    tot = 2*S2*L
    p.append('    for(long i=0;i<%d;i+=16){' % (tot - tot % 16))
    p.append('      V8 z0 = LDA(sl+i) + LDA(cl+i), z1 = LDA(sl+i+8) + LDA(cl+i+8), o0_, o1_;')
    p.append('      pw_pair_fast(z0,z1,&o0_,&o1_);')
    p.append('      STA(sl+i,o0_); STA(sl+i+8,o1_);')
    p.append('    }')
    if tot % 16:
        p.append('    STA(sl+%d, pw_full_fast(LDA(sl+%d) + LDA(cl+%d)));' % (tot-8, tot-8, tot-8))
    p.append('    for(int ch=0;ch<%d;++ch) fft%d_a1(sl + ch*8);' % (C4, L))
    if L == 64:
        p.append('    for(int g=0;g<16;++g) fft64_a2f(sl + (long)g*%d);' % (4*2*S2))
    else:
        p.append('    for(int g=0;g<%d;++g) fft%d_a2(sl + (long)g*%d);' % (ng, L, 4*2*S2))
    if L % 4 == 1:
        p.append('    fft%d_row2(sl + %d);' % (L, (L-1)*2*S2))
    p.append('  }')
    p.append('}')
    p.append('static ' + OA + ' void passB_%d(double *restrict x){' % L)
    p.append('  for(int j1=0;j1<%d;++j1){' % L)
    p.append('    for(int ch=0;ch<%d;++ch){' % C4)
    p.append('      long off = (long)j1*%d + ch*8;' % (2*S2))
    p.append('      fft%d_b(x + off);' % L)
    p.append('    }')
    p.append('  }')
    p.append('}')
    p.append('static ' + OA + ' void passBpw_%d(double *restrict x, const double *restrict c){' % L)
    p.append('  for(int j1=0;j1<%d;++j1){' % L)
    p.append('    for(int ch=0;ch<%d;++ch){' % C4)
    p.append('      long off = (long)j1*%d + ch*8;' % (2*S2))
    p.append('      fft%d_bpw(x + off, c + off);' % L)
    p.append('    }')
    p.append('  }')
    p.append('}')
    p.append('static ' + OA + ' void sweep_cpw_%d(double *restrict x, const double *restrict c){' % L)
    p.append('  for(int j0=0;j0<%d;++j0){' % L)
    p.append('    double *restrict r = x + (long)j0*%d;' % (2*S1))
    p.append('    const double *restrict rc = c + (long)j0*%d;' % (2*S1))
    tot = 2*S2*L
    p.append('    for(long i=0;i<%d;i+=16){' % (tot - tot % 16))
    p.append('      V8 z0 = LDA(r+i) + LDA(rc+i), z1 = LDA(r+i+8) + LDA(rc+i+8), o0_, o1_;')
    p.append('      pw_pair_fast(z0,z1,&o0_,&o1_);')
    p.append('      STA(r+i,o0_); STA(r+i+8,o1_);')
    p.append('    }')
    if tot % 16:
        p.append('    STA(r+%d, pw_full_fast(LDA(r+%d) + LDA(rc+%d)));' % (tot-8, tot-8, tot-8))
    p.append('  }')
    p.append('}')
    # --- pack/unpack
    p.append('static void unpack_%d(double *restrict dst, const double *restrict src){' % L)
    p.append('  for(int j0=0;j0<%d;++j0) for(int j1=0;j1<%d;++j1)' % (L,L))
    p.append('    memcpy(dst + (long)j0*%d + j1*%d, src + ((long)j0*%d + j1)*%d, %d);'
             % (2*S1, 2*S2, L, 2*L, 16*L))
    p.append('}')
    p.append('static void pack_%d(double *restrict dst, const double *restrict src){' % L)
    p.append('  for(int j0=0;j0<%d;++j0) for(int j1=0;j1<%d;++j1)' % (L,L))
    p.append('    memcpy(dst + ((long)j0*%d + j1)*%d, src + (long)j0*%d + j1*%d, %d);'
             % (L, 2*L, 2*S1, 2*S2, 16*L))
    p.append('}')
    p.append('static ' + OA + ' void pack_pw_%d(double *restrict dst, const double *restrict src, const double *restrict c){' % L)
    p.append('  V8 rowbuf[%d];' % (S2//4))
    p.append('  for(int j0=0;j0<%d;++j0) for(int j1=0;j1<%d;++j1){' % (L,L))
    p.append('    const double *restrict r = src + (long)j0*%d + j1*%d;' % (2*S1, 2*S2))
    p.append('    const double *restrict rc = c + (long)j0*%d + j1*%d;' % (2*S1, 2*S2))
    p.append('    for(int t=0;t+1<%d;t+=2){' % ((L+3)//4))
    p.append('      V8 z0 = LDA(r+t*8) + LDA(rc+t*8), z1 = LDA(r+t*8+8) + LDA(rc+t*8+8);')
    p.append('      pw_pair(z0, z1, &rowbuf[t], &rowbuf[t+1]);')
    p.append('    }')
    if ((L+3)//4) % 2:
        p.append('    rowbuf[%d] = pw_full(LDA(r+%d) + LDA(rc+%d));' % ((L+3)//4-1, ((L+3)//4-1)*8, ((L+3)//4-1)*8))
    p.append('    memcpy(dst + ((long)j0*%d + j1)*%d, rowbuf, %d);' % (L, 2*L, 16*L))
    p.append('  }')
    p.append('}')
    # --- buffers + run
    p.append('static double *bufx_%d, *bufc_%d;' % (L,L))
    p.append('void run%d_A(const double *x0, const double *cc, double *o1, double *om, long B, long m){' % L)
    p.append('  const long NV = %d;' % (2*L*L*L))
    p.append('  for(long b=0;b<B;++b){')
    p.append('    unpack_%d(bufx_%d, x0 + b*NV);' % (L,L))
    p.append('    unpack_%d(bufc_%d, cc + b*NV);' % (L,L))
    p.append('    if(m <= 0){')
    p.append('      memcpy(om + b*NV, x0 + b*NV, NV*8);')
    p.append('      passA_%d(bufx_%d, bufc_%d, 0); passB_%d(bufx_%d);' % (L,L,L,L,L))
    p.append('      pack_pw_%d(o1 + b*NV, bufx_%d, bufc_%d);' % (L,L,L))
    p.append('      continue;')
    p.append('    }')
    p.append('    for(long t=0;t<m;++t){')
    p.append('      passA_%d(bufx_%d, bufc_%d, t!=0); passB_%d(bufx_%d);' % (L,L,L,L,L))
    p.append('      if(t==0) pack_pw_%d(o1 + b*NV, bufx_%d, bufc_%d);' % (L,L,L))
    p.append('    }')
    p.append('    pack_pw_%d(om + b*NV, bufx_%d, bufc_%d);' % (L,L,L))
    p.append('  }')
    p.append('}')
    p.append('void run%d_D(const double *x0, const double *cc, double *o1, double *om, long B, long m){' % L)
    p.append('  const long NV = %d;' % (2*L*L*L))
    p.append('  for(long b=0;b<B;++b){')
    p.append('    unpack_%d(bufx_%d, x0 + b*NV);' % (L,L))
    p.append('    unpack_%d(bufc_%d, cc + b*NV);' % (L,L))
    p.append('    passA_%d(bufx_%d, bufc_%d, 0); passB_%d(bufx_%d);' % (L,L,L,L,L))
    p.append('    if(m <= 0){')
    p.append('      memcpy(om + b*NV, x0 + b*NV, NV*8);')
    p.append('      pack_pw_%d(o1 + b*NV, bufx_%d, bufc_%d);' % (L,L,L))
    p.append('      continue;')
    p.append('    }')
    p.append('    pack_pw_%d(o1 + b*NV, bufx_%d, bufc_%d);' % (L,L,L))
    p.append('    if(m == 1){ memcpy(om + b*NV, o1 + b*NV, NV*8); continue; }')
    p.append('    for(long t=1;t<m;++t){')
    p.append('      sweep_cpw_%d(bufx_%d, bufc_%d);' % (L,L,L))
    p.append('      passA_%d(bufx_%d, bufc_%d, 0); passB_%d(bufx_%d);' % (L,L,L,L,L))
    p.append('    }')
    p.append('    pack_pw_%d(om + b*NV, bufx_%d, bufc_%d);' % (L,L,L))
    p.append('  }')
    p.append('}')
    p.append('void run%d_E(const double *x0, const double *cc, double *o1, double *om, long B, long m){' % L)
    p.append('  const long NV = %d;' % (2*L*L*L))
    p.append('  for(long b=0;b<B;++b){')
    p.append('    unpack_%d(bufx_%d, x0 + b*NV);' % (L,L))
    p.append('    unpack_%d(bufc_%d, cc + b*NV);' % (L,L))
    p.append('    passA_%d(bufx_%d, bufc_%d, 0); passB_%d(bufx_%d);' % (L,L,L,L,L))
    p.append('    if(m <= 0){')
    p.append('      memcpy(om + b*NV, x0 + b*NV, NV*8);')
    p.append('      pack_pw_%d(o1 + b*NV, bufx_%d, bufc_%d);' % (L,L,L))
    p.append('      continue;')
    p.append('    }')
    p.append('    pack_pw_%d(o1 + b*NV, bufx_%d, bufc_%d);' % (L,L,L))
    p.append('    if(m == 1){ memcpy(om + b*NV, o1 + b*NV, NV*8); continue; }')
    p.append('    for(long t=1;t<m;++t){')
    p.append('      passAE_%d(bufx_%d, bufc_%d); passB_%d(bufx_%d);' % (L,L,L,L,L))
    p.append('    }')
    p.append('    pack_pw_%d(om + b*NV, bufx_%d, bufc_%d);' % (L,L,L))
    p.append('  }')
    p.append('}')
    p.append('void run%d_B(const double *x0, const double *cc, double *o1, double *om, long B, long m){' % L)
    p.append('  const long NV = %d;' % (2*L*L*L))
    p.append('  for(long b=0;b<B;++b){')
    p.append('    unpack_%d(bufx_%d, x0 + b*NV);' % (L,L))
    p.append('    unpack_%d(bufc_%d, cc + b*NV);' % (L,L))
    p.append('    passA_%d(bufx_%d, bufc_%d, 0); passB_%d(bufx_%d);' % (L,L,L,L,L))
    p.append('    if(m <= 0){')
    p.append('      memcpy(om + b*NV, x0 + b*NV, NV*8);')
    p.append('      pack_pw_%d(o1 + b*NV, bufx_%d, bufc_%d);' % (L,L,L))
    p.append('      continue;')
    p.append('    }')
    p.append('    pack_pw_%d(o1 + b*NV, bufx_%d, bufc_%d);' % (L,L,L))
    p.append('    if(m == 1){ memcpy(om + b*NV, o1 + b*NV, NV*8); continue; }')
    p.append('    sweep_cpw_%d(bufx_%d, bufc_%d);' % (L,L,L))
    p.append('    for(long t=1;t<m;++t){')
    p.append('      passA_%d(bufx_%d, bufc_%d, 0); passBpw_%d(bufx_%d, bufc_%d);' % (L,L,L,L,L,L))
    p.append('    }')
    p.append('    pack_%d(om + b*NV, bufx_%d);' % (L,L))
    p.append('  }')
    p.append('}')
    # --- debug: single codelet test via io flavor
    p.append('void test1d_%d(double *inout){ fft%d_io((V8*)inout); }' % (L,L))
    parts.append('\n'.join(p)+'\n')
    return '\n'.join(parts)


def gen_batched_L(L):
    """4-volume batched layout: each vector = one element of 4 volumes."""
    S2g, S1g = LAYOUTG[L]
    OA = SCHED[L]
    EMITX = (lambda n,w,u,ld,st,out,pre='',post='',blockpre=None:
            emit_prime_codelet(n, ld, st, out, post=post) if (n in PRIME_PHASE and w == 8 and blockpre is None and pre == '')
            else emit_codelet(n,w,u,ld,st,out,pre,post,blockpre))
    parts = []
    V = 8  # doubles per vector
    # --- codelets: g1 (stride 1 vec), g2 (stride S2g), g3 (stride S1g) ---
    parts.append(EMITX(L, 8, '', lambda i: 'LDA(p+%d)' % (i*8),
        lambda k,e: 'STA(p+%d, %s);' % (k*8, e), out='fftg%d_1(double *restrict p)' % L))
    parts.append(EMITX(L, 8, '', lambda i: 'LDA(p+%d)' % (i*8*S2g),
        lambda k,e: 'STA(p+%d, %s);' % (k*8*S2g, e), out='fftg%d_2(double *restrict p)' % L))
    parts.append(EMITX(L, 8, '', lambda i: 'LDA(p+%d)' % (i*8*S1g),
        lambda k,e: 'STA(p+%d, %s);' % (k*8*S1g, e), out='fftg%d_3(double *restrict p)' % L))
    # --- g1pw: c+pw fused at loads of the j2 (contiguous) codelet ---
    if L in STAGED:
        P,Q,kind = STAGED[L]
        def blockpre(b, idx):
            bl = []
            for t,i in enumerate(idx):
                bl.append('V8 zb%d_%d = LDA(p+%d) + LDA(c+%d);' % (b, t, i*8, i*8))
            t = 0
            while t+1 < len(idx):
                bl.append('V8 wb%d_%d, wb%d_%d; pw_pair_fast(zb%d_%d,zb%d_%d,&wb%d_%d,&wb%d_%d);'
                          % (b,t,b,t+1,b,t,b,t+1,b,t,b,t+1))
                t += 2
            if len(idx) % 2:
                bl.append('V8 wb%d_%d = pw_full_fast(zb%d_%d);' % (b,len(idx)-1,b,len(idx)-1))
            return bl, (lambda i, b=b: 'wb%d_%d' % (b, i))
        parts.append(emit_codelet(L, 8, '', None,
            lambda k,e: 'STA(p+%d, %s);' % (k*8, e),
            out='fftg%d_1pw(double *restrict p, const double *restrict c)' % L,
            blockpre=blockpre))
    elif L in PRIME_PHASE:
        parts.append(emit_prime_codelet(L,
            lambda i: 'LDA(p+%d)' % (i*8),
            lambda k,e: 'STA(p+%d, %s);' % (k*8, e),
            out='fftg%d_1pw(double *restrict p, const double *restrict c)' % L,
            pwload='fast'))
    else:
        pre_lines = ['V8 z0 = LDA(p) + LDA(c); V8 w0 = pw_full_fast(z0);']
        j = 1
        while j < L - j:
            pre_lines.append('V8 z%d = LDA(p+%d) + LDA(c+%d);' % (j, j*8, j*8))
            pre_lines.append('V8 z%d = LDA(p+%d) + LDA(c+%d);' % (L-j, (L-j)*8, (L-j)*8))
            pre_lines.append('V8 w%d, w%d; pw_pair_fast(z%d,z%d,&w%d,&w%d);' % (j,L-j,j,L-j,j,L-j))
            j += 1
        if L % 2 == 0:
            h2 = L//2
            pre_lines.append('V8 z%d = LDA(p+%d) + LDA(c+%d); V8 w%d = pw_full_fast(z%d);' % (h2,h2*8,h2*8,h2,h2))
        parts.append(emit_codelet(L, 8, '',
            lambda i: 'w%d' % i,
            lambda k,e: 'STA(p+%d, %s);' % (k*8, e),
            out='fftg%d_1pw(double *restrict p, const double *restrict c)' % L,
            pre='\n  '.join(pre_lines)))
    # --- g3pw: c+pw fused at stores of the j0 codelet ---
    pend = [None]
    def st_g3pw(k, e):
        s = 'V8 y%d = %s + LDA(c+%d);' % (k, e, k*8*S1g)
        if pend[0] is None:
            pend[0] = k
            return s
        k0 = pend[0]; pend[0] = None
        return (s + ' { V8 r0_,r1_; pw_pair_fast(y%d,y%d,&r0_,&r1_); STA(p+%d,r0_); STA(p+%d,r1_); }'
                % (k0, k, k0*8*S1g, k*8*S1g))
    parts.append(EMITX(L, 8, '',
        lambda i: 'LDA(p+%d)' % (i*8*S1g),
        st_g3pw,
        out='fftg%d_3pw(double *restrict p, const double *restrict c)' % L,
        post=lambda: ('STA(p+%d, pw_full_fast(y%d));' % (pend[0]*8*S1g, pend[0])) if pend[0] is not None else ''))
    # --- passes ---
    p = []
    p.append('static %s void passGA_%d(double *restrict x, const double *restrict c, int mode){' % (OA, L))
    p.append('  for(int s=0;s<%d;++s){' % L)
    p.append('    double *restrict sl = x + (long)s*%d;' % (8*S1g))
    p.append('    const double *restrict cl = c + (long)s*%d;' % (8*S1g))
    p.append('    if (mode == 2){  // E: sweep rows then plain')
    p.append('      for(int j1=0;j1<%d;++j1){' % L)
    p.append('        double *restrict r = sl + (long)j1*%d; const double *restrict rc = cl + (long)j1*%d;' % (8*S2g, 8*S2g))
    p.append('        for(int i=0;i<%d;i+=16){' % (8*(L - L%2)))
    p.append('          V8 z0 = LDA(r+i)+LDA(rc+i), z1 = LDA(r+i+8)+LDA(rc+i+8), o0_,o1_;')
    p.append('          pw_pair_fast(z0,z1,&o0_,&o1_); STA(r+i,o0_); STA(r+i+8,o1_);')
    p.append('        }')
    if L % 2:
        p.append('        STA(r+%d, pw_full_fast(LDA(r+%d)+LDA(rc+%d)));' % (8*(L-1),8*(L-1),8*(L-1)))
    p.append('      }')
    p.append('    }')
    p.append('    if (mode == 1){ for(int j1=0;j1<%d;++j1) fftg%d_1pw(sl + (long)j1*%d, cl + (long)j1*%d); }' % (L, L, 8*S2g, 8*S2g))
    p.append('    else { for(int j1=0;j1<%d;++j1) fftg%d_1(sl + (long)j1*%d); }' % (L, L, 8*S2g))
    p.append('    for(int j2=0;j2<%d;++j2) fftg%d_2(sl + j2*8);' % (L, L))
    p.append('  }')
    p.append('}')
    p.append('static %s void passGB_%d(double *restrict x){' % (OA, L))
    p.append('  for(int j1=0;j1<%d;++j1){' % L)
    p.append('    for(int j2=0;j2<%d;++j2){' % L)
    p.append('      long off = (long)j1*%d + j2*8;' % (8*S2g))
    p.append('      fftg%d_3(x + off);' % L)
    p.append('    }')
    p.append('  }')
    p.append('}')
    p.append('static %s void passGBpw_%d(double *restrict x, const double *restrict c){' % (OA, L))
    p.append('  for(int j1=0;j1<%d;++j1){' % L)
    p.append('    for(int j2=0;j2<%d;++j2){' % L)
    p.append('      long off = (long)j1*%d + j2*8;' % (8*S2g))
    p.append('      fftg%d_3pw(x + off, c + off);' % L)
    p.append('    }')
    p.append('  }')
    p.append('}')
    # --- unpack/pack with 4x4 complex transposes; masked tails for L%4 != 0 ---
    T = L % 4
    mask = (1 << (2*T)) - 1
    p.append('static %s void unpackg_%d(double *restrict dst, const double *restrict src, long NV){' % (OA, L))
    p.append('  // src: 4 consecutive volumes (NV doubles each); dst: batched layout')
    p.append('  for(int j0=0;j0<%d;++j0) for(int j1=0;j1<%d;++j1){' % (L,L))
    p.append('    long rb = ((long)j0*%d + j1)*%d;' % (L, 2*L))
    p.append('    double *restrict d = dst + (long)j0*%d + (long)j1*%d;' % (8*S1g, 8*S2g))
    p.append('    int t=0;')
    p.append('    for(; t+4<=%d; t+=4){' % L)
    p.append('      V8 a0 = LDU(src + rb + t*2);')
    p.append('      V8 a1 = LDU(src + NV + rb + t*2);')
    p.append('      V8 a2_ = LDU(src + 2*NV + rb + t*2);')
    p.append('      V8 a3 = LDU(src + 3*NV + rb + t*2);')
    p.append('      V8 o0,o1,o2,o3; TRANSP4(a0,a1,a2_,a3,o0,o1,o2,o3);')
    p.append('      STA(d + t*8, o0); STA(d + t*8+8, o1); STA(d + t*8+16, o2); STA(d + t*8+24, o3);')
    p.append('    }')
    if T:
        p.append('    { __mmask8 mk = %d;' % mask)
        p.append('      V8 a0 = (V8)_mm512_maskz_loadu_pd(mk, src + rb + t*2);')
        p.append('      V8 a1 = (V8)_mm512_maskz_loadu_pd(mk, src + NV + rb + t*2);')
        p.append('      V8 a2_ = (V8)_mm512_maskz_loadu_pd(mk, src + 2*NV + rb + t*2);')
        p.append('      V8 a3 = (V8)_mm512_maskz_loadu_pd(mk, src + 3*NV + rb + t*2);')
        p.append('      V8 o0,o1,o2,o3; TRANSP4(a0,a1,a2_,a3,o0,o1,o2,o3);')
        st = ['STA(d + t*8, o0);','STA(d + t*8+8, o1);','STA(d + t*8+16, o2);']
        p.append('      ' + ' '.join(st[:T]) + ' }')
    p.append('  }')
    p.append('}')
    p.append('static %s void packg_%d(double *restrict dst, const double *restrict src, long NV){' % (OA, L))
    p.append('  for(int j0=0;j0<%d;++j0) for(int j1=0;j1<%d;++j1){' % (L,L))
    p.append('    long rb = ((long)j0*%d + j1)*%d;' % (L, 2*L))
    p.append('    const double *restrict s = src + (long)j0*%d + (long)j1*%d;' % (8*S1g, 8*S2g))
    p.append('    int t=0;')
    p.append('    for(; t+4<=%d; t+=4){' % L)
    p.append('      V8 a0 = LDA(s + t*8), a1 = LDA(s + t*8+8), a2_ = LDA(s + t*8+16), a3 = LDA(s + t*8+24);')
    p.append('      V8 o0,o1,o2,o3; TRANSP4(a0,a1,a2_,a3,o0,o1,o2,o3);')
    p.append('      STU(dst + rb + t*2, o0); STU(dst + NV + rb + t*2, o1);')
    p.append('      STU(dst + 2*NV + rb + t*2, o2); STU(dst + 3*NV + rb + t*2, o3);')
    p.append('    }')
    if T:
        ld = ['V8 a0 = LDA(s + t*8);','V8 a1 = LDA(s + t*8+8);','V8 a2_ = LDA(s + t*8+16);']
        zz = ['V8 a0;','V8 a1;','V8 a2_;']
        decl = ' '.join(ld[:T] + (['V8 a%s = a0;' % ('1' if T==1 else '2_' if T==2 else '3')]))
        p.append('    { __mmask8 mk = %d;' % mask)
        p.append('      ' + ' '.join(ld[:T]))
        for miss in range(T,4):
            nm = ['a0','a1','a2_','a3'][miss]
            p.append('      V8 %s = a0;' % nm)
        p.append('      V8 o0,o1,o2,o3; TRANSP4(a0,a1,a2_,a3,o0,o1,o2,o3);')
        p.append('      _mm512_mask_storeu_pd(dst + rb + t*2, mk, (__m512d)o0);')
        p.append('      _mm512_mask_storeu_pd(dst + NV + rb + t*2, mk, (__m512d)o1);')
        p.append('      _mm512_mask_storeu_pd(dst + 2*NV + rb + t*2, mk, (__m512d)o2);')
        p.append('      _mm512_mask_storeu_pd(dst + 3*NV + rb + t*2, mk, (__m512d)o3);')
        p.append('    }')
    p.append('  }')
    p.append('}')
    # pack with c-add + FULL pw applied (outputs)
    p.append('static %s void packpwg_%d(double *restrict dst, const double *restrict src, const double *restrict c, long NV){' % (OA, L))
    p.append('  for(int j0=0;j0<%d;++j0) for(int j1=0;j1<%d;++j1){' % (L,L))
    p.append('    long rb = ((long)j0*%d + j1)*%d;' % (L, 2*L))
    p.append('    const double *restrict s = src + (long)j0*%d + (long)j1*%d;' % (8*S1g, 8*S2g))
    p.append('    const double *restrict sc = c + (long)j0*%d + (long)j1*%d;' % (8*S1g, 8*S2g))
    p.append('    V8 rowbuf[%d];' % ((L+3)//4*4))
    p.append('    for(int t=0;t+1<%d;t+=2){' % L)
    p.append('      V8 z0 = LDA(s+t*8)+LDA(sc+t*8), z1 = LDA(s+t*8+8)+LDA(sc+t*8+8);')
    p.append('      pw_pair(z0,z1,&rowbuf[t],&rowbuf[t+1]);')
    p.append('    }')
    if L % 2:
        p.append('    rowbuf[%d] = pw_full(LDA(s+%d)+LDA(sc+%d));' % (L-1, (L-1)*8, (L-1)*8))
    p.append('    int t=0;')
    p.append('    for(; t+4<=%d; t+=4){' % L)
    p.append('      V8 o0,o1,o2,o3; TRANSP4(rowbuf[t],rowbuf[t+1],rowbuf[t+2],rowbuf[t+3],o0,o1,o2,o3);')
    p.append('      STU(dst + rb + t*2, o0); STU(dst + NV + rb + t*2, o1);')
    p.append('      STU(dst + 2*NV + rb + t*2, o2); STU(dst + 3*NV + rb + t*2, o3);')
    p.append('    }')
    if T:
        p.append('    { __mmask8 mk = %d;' % mask)
        for miss in range(T,4):
            p.append('      rowbuf[t+%d] = rowbuf[t];' % miss)
        p.append('      V8 o0,o1,o2,o3; TRANSP4(rowbuf[t],rowbuf[t+1],rowbuf[t+2],rowbuf[t+3],o0,o1,o2,o3);')
        p.append('      _mm512_mask_storeu_pd(dst + rb + t*2, mk, (__m512d)o0);')
        p.append('      _mm512_mask_storeu_pd(dst + NV + rb + t*2, mk, (__m512d)o1);')
        p.append('      _mm512_mask_storeu_pd(dst + 2*NV + rb + t*2, mk, (__m512d)o2);')
        p.append('      _mm512_mask_storeu_pd(dst + 3*NV + rb + t*2, mk, (__m512d)o3);')
        p.append('    }')
    p.append('  }')
    p.append('}')
    # --- buffers + run (modes: 1 = pw at g1 loads (A-like), 3 = pw at g3 stores (B-like), 2 = slab sweep (E-like)) ---
    p.append('static double *bufgx_%d, *bufgc_%d;' % (L,L))
    p.append('void rung%d(const double *x0, const double *cc, double *o1, double *om, long NG, long m, int mode){' % L)
    p.append('  const long NV = %d;' % (2*L*L*L))
    p.append('  for(long g=0;g<NG;++g){')
    p.append('    const double *xs = x0 + 4*g*NV; const double *cs = cc + 4*g*NV;')
    p.append('    unpackg_%d(bufgx_%d, xs, NV);' % (L,L))
    p.append('    unpackg_%d(bufgc_%d, cs, NV);' % (L,L))
    p.append('    passGA_%d(bufgx_%d, bufgc_%d, 0); passGB_%d(bufgx_%d);' % (L,L,L,L,L))
    p.append('    if(m <= 0){')
    p.append('      for(int v=0;v<4;v++) memcpy(om + 4*g*NV + v*NV, xs + v*NV, NV*8);')
    p.append('      packpwg_%d(o1 + 4*g*NV, bufgx_%d, bufgc_%d, NV);' % (L,L,L))
    p.append('      continue;')
    p.append('    }')
    p.append('    packpwg_%d(o1 + 4*g*NV, bufgx_%d, bufgc_%d, NV);' % (L,L,L))
    p.append('    if(m == 1){ memcpy(om + 4*g*NV, o1 + 4*g*NV, 4*NV*8); continue; }')
    p.append('    if(mode == 3){')
    p.append('      // B-like: state post-pw; need one sweep to convert raw z_1 -> x_1')
    p.append('      for(int s=0;s<%d;++s){' % L)
    p.append('        double *restrict sl = bufgx_%d + (long)s*%d; const double *restrict cl = bufgc_%d + (long)s*%d;' % (L, 8*S1g, L, 8*S1g))
    p.append('        for(int j1=0;j1<%d;++j1){' % L)
    p.append('          double *restrict r = sl + (long)j1*%d; const double *restrict rc = cl + (long)j1*%d;' % (8*S2g, 8*S2g))
    p.append('          for(int i=0;i<%d;i+=16){' % (8*(L - L%2)))
    p.append('            V8 z0 = LDA(r+i)+LDA(rc+i), z1 = LDA(r+i+8)+LDA(rc+i+8), o0_,o1_;')
    p.append('            pw_pair_fast(z0,z1,&o0_,&o1_); STA(r+i,o0_); STA(r+i+8,o1_);')
    p.append('          }')
    if L % 2:
        p.append('          STA(r+%d, pw_full_fast(LDA(r+%d)+LDA(rc+%d)));' % (8*(L-1),8*(L-1),8*(L-1)))
    p.append('        }')
    p.append('      }')
    p.append('      for(long t=1;t<m;++t){ passGA_%d(bufgx_%d, bufgc_%d, 0); passGBpw_%d(bufgx_%d, bufgc_%d); }' % (L,L,L,L,L,L))
    p.append('      packg_%d(om + 4*g*NV, bufgx_%d, NV);' % (L,L))
    p.append('    } else {')
    p.append('      for(long t=1;t<m;++t){ passGA_%d(bufgx_%d, bufgc_%d, mode); passGB_%d(bufgx_%d); }' % (L,L,L,L,L))
    p.append('      packpwg_%d(om + 4*g*NV, bufgx_%d, bufgc_%d, NV);' % (L,L,L))
    p.append('    }')
    p.append('  }')
    p.append('}')
    parts.append('\n'.join(p)+'\n')
    return '\n'.join(parts)

def emit_file(path):
    chunks = [HEADER]
    for L in SIZES:
        chunks.append(gen_for_L(L))
        chunks.append(gen_batched_L(L))
    # setup
    s = []
    s.append('static void* big_alloc(size_t n){')
    s.append('  n = (n + (2UL<<20) - 1) & ~((2UL<<20)-1);')
    s.append('  void *p = aligned_alloc(2UL<<20, n);')
    s.append('#ifdef __linux__')
    s.append('  if(p) madvise(p, n, MADV_HUGEPAGE);')
    s.append('#endif')
    s.append('  if(!p) p = aligned_alloc(4096, n);')
    s.append('  if(!p){ if(posix_memalign(&p, 64, n) != 0) p = 0; }')
    s.append('  if(!p) abort();')
    s.append('  memset(p, 0, n);')
    s.append('  return p;')
    s.append('}')
    s.append('void setup(void){')
    for L in SIZES:
        S2,R1,S1 = LAYOUT[L]
        nb = L*S1*16
        s.append('  bufx_%d = (double*)big_alloc(%d); bufc_%d = (double*)big_alloc(%d);' % (L,nb,L,nb))
    for L in SIZES:
        S2g,S1g = LAYOUTG[L]
        nbg = L*S1g*64
        s.append('  bufgx_%d = (double*)big_alloc(%d); bufgc_%d = (double*)big_alloc(%d);' % (L,nbg,L,nbg))
    s.append('}')
    chunks.append('\n'.join(s)+'\n')
    with open(path,'w') as f:
        f.write('\n'.join(chunks))

if __name__ == '__main__':
    import sys
    emit_file(sys.argv[1] if len(sys.argv)>1 else 'implementation.c')
    print('written')
