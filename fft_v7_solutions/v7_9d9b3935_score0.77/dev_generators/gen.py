#!/usr/bin/env python3
"""Codelet generator: builds straight-line vectorized FFT codelets as C text.
All my own arithmetic: folded direct DFT, PFA, Cooley-Tukey, Rader.
Runs at DEV time only; output C is static."""
import math, sys
from functools import lru_cache
import mpmath as mp
mp.mp.prec = 160

# ---------------- high precision constants ----------------
def omega_re_im(num, den, sign):
    # e^{sign*2*pi*i*num/den} -> (mpf cos, mpf sin) with exact rational reduction
    num = num % den
    ang = mp.mpf(2) * mp.pi * mp.mpf(num) / mp.mpf(den)
    return (mp.cos(ang), mp.mpf(sign) * mp.sin(ang))

# ---------------- expression DAG ----------------
class Dag:
    def __init__(self):
        self.nodes = []     # tuples
        self.memo = {}
        self.parents = {}   # id -> count of uses
        self.ZERO = self._raw(('zero',))
    def _raw(self, t):
        if t in self.memo: return self.memo[t]
        i = len(self.nodes); self.nodes.append(t); self.memo[t] = i
        return i
    def inp(self, tag):
        return self._raw(('in', tag))
    def kind(self, i): return self.nodes[i][0]
    def neg(self, a):
        if a == self.ZERO: return a
        t = self.nodes[a]
        if t[0] == 'neg': return t[1]
        return self._raw(('neg', a))
    def add(self, a, b):
        if a == self.ZERO: return b
        if b == self.ZERO: return a
        ta, tb = self.nodes[a], self.nodes[b]
        if ta[0] == 'neg' and tb[0] == 'neg': return self.neg(self.add(ta[1], tb[1]))
        if tb[0] == 'neg': return self.sub(a, tb[1])
        if ta[0] == 'neg': return self.sub(b, ta[1])
        if a > b: a, b = b, a
        return self._raw(('add', a, b))
    def sub(self, a, b):
        if b == self.ZERO: return a
        if a == self.ZERO: return self.neg(b)
        if a == b: return self.ZERO
        ta, tb = self.nodes[a], self.nodes[b]
        if tb[0] == 'neg': return self.add(a, tb[1])
        if ta[0] == 'neg': return self.neg(self.add(ta[1], b))
        return self._raw(('sub', a, b))
    def mul(self, c, a):
        # c: mpf constant
        if a == self.ZERO: return a
        cf = float(c)
        if cf == 0.0: return self.ZERO
        neg = False
        if cf < 0: c, cf, neg = -c, -cf, True
        t = self.nodes[a]
        if t[0] == 'neg': a = t[1]; neg = not neg
        if cf == 1.0 and c == 1:
            r = a
        else:
            t = self.nodes[a]
            if t[0] == 'mul':
                # fold constants in high precision
                c = c * t[1][1]
                a = t[2]
                cf = float(c)
                if cf == 1.0: r = a
                else: r = self._raw(('mul', ('c', c), a))
            else:
                r = self._raw(('mul', ('c', c), a))
        return self.neg(r) if neg else r

# complex helpers: values are (re_id, im_id)
def cadd(d, x, y): return (d.add(x[0], y[0]), d.add(x[1], y[1]))
def csub(d, x, y): return (d.sub(x[0], y[0]), d.sub(x[1], y[1]))
def cneg(d, x): return (d.neg(x[0]), d.neg(x[1]))
def cmulw(d, x, wr, wi):
    # x * (wr + i wi), wr/wi mpf
    fr, fi = float(wr), float(wi)
    if fi == 0.0:
        return (d.mul(wr, x[0]), d.mul(wr, x[1]))
    if fr == 0.0:
        return (d.neg(d.mul(wi, x[1])), d.mul(wi, x[0]))
    if abs(abs(fr) - abs(fi)) < 1e-300 and mp.fabs(mp.fabs(wr) - mp.fabs(wi)) < mp.mpf(2)**-140:
        # w = c*(s1 + i s2), |s1|=|s2|=1 -> x*w = c*((s1 xr - s2 xi) + i(s1 xi + s2 xr))
        s1 = 1 if fr > 0 else -1
        s2 = 1 if fi > 0 else -1
        c = mp.fabs(wr)
        xr, xi = x
        if s1 > 0:
            rr = d.sub(xr, xi) if s2 > 0 else d.add(xr, xi)
            ri = d.add(xi, xr) if s2 > 0 else d.sub(xi, xr)
        else:
            rr = d.neg(d.add(xr, xi)) if s2 > 0 else d.sub(xi, xr)
            ri = d.sub(xr, xi) if s2 > 0 else d.neg(d.add(xr, xi))
        return (d.mul(c, rr), d.mul(c, ri))
    xr, xi = x
    rr = d.sub(d.mul(wr, xr), d.mul(wi, xi))
    ri = d.add(d.mul(wr, xi), d.mul(wi, xr))
    return (rr, ri)

# ---------------- DFT builders ----------------
def primitive_root(p):
    fac = []
    q = p - 1; d = 2
    while d*d <= q:
        if q % d == 0:
            fac.append(d)
            while q % d == 0: q //= d
        d += 1
    if q > 1: fac.append(q)
    for g in range(2, p):
        if all(pow(g, (p-1)//f, p) != 1 for f in fac):
            return g
    raise ValueError

PLAN_CHOICE = {}  # n -> plan descriptor (for reporting)

def build_dft(d, xs, sign, plan=None):
    n = len(xs)
    if n == 1: return list(xs)
    if plan is None:
        plan = best_plan(n)
    k = plan[0]
    if k == 'direct':
        return direct_dft(d, xs, sign)
    if k == 'pfa':
        return pfa_dft(d, xs, sign, plan[1], plan[2])
    if k == 'ct':
        return ct_dft(d, xs, sign, plan[1], plan[2])
    if k == 'rader':
        return rader_dft(d, xs, sign)
    raise ValueError(plan)

def direct_dft(d, xs, sign):
    n = len(xs)
    if n == 2:
        return [cadd(d, xs[0], xs[1]), csub(d, xs[0], xs[1])]
    if n == 4:
        a = cadd(d, xs[0], xs[2]); b = csub(d, xs[0], xs[2])
        c = cadd(d, xs[1], xs[3]); e = csub(d, xs[1], xs[3])
        X0 = cadd(d, a, c); X2 = csub(d, a, c)
        # e * (-i*sign?): forward sign=-1: X1 = b + (-i)e = (br+ei, bi-er)
        if sign < 0:
            X1 = (d.add(b[0], e[1]), d.sub(b[1], e[0]))
            X3 = (d.sub(b[0], e[1]), d.add(b[1], e[0]))
        else:
            X1 = (d.sub(b[0], e[1]), d.add(b[1], e[0]))
            X3 = (d.add(b[0], e[1]), d.sub(b[1], e[0]))
        return [X0, X1, X2, X3]
    if n % 2 == 1:
        # folded direct for odd n
        h = (n-1)//2
        s = [None]*(h+1); t = [None]*(h+1)
        for j in range(1, h+1):
            s[j] = cadd(d, xs[j], xs[n-j])
            t[j] = csub(d, xs[j], xs[n-j])
        X = [None]*n
        acc = xs[0]
        for j in range(1, h+1): acc = cadd(d, acc, s[j])
        X[0] = acc
        for k in range(1, h+1):
            Ar, Ai = xs[0]
            Br, Bi = d.ZERO, d.ZERO
            for j in range(1, h+1):
                cjk, sjk = omega_re_im(j*k, n, -1)  # cos(2pi jk/n), -sin(...)
                # use cos and sin magnitudes: cos=cjk ; sin = -sjk is sin(2pi jk/n)
                co = cjk; si = -sjk
                Ar = d.add(Ar, d.mul(co, s[j][0]))
                Ai = d.add(Ai, d.mul(co, s[j][1]))
                Br = d.add(Br, d.mul(si, t[j][0]))
                Bi = d.add(Bi, d.mul(si, t[j][1]))
            if sign < 0:
                X[k]   = (d.add(Ar, Bi), d.sub(Ai, Br))
                X[n-k] = (d.sub(Ar, Bi), d.add(Ai, Br))
            else:
                X[k]   = (d.sub(Ar, Bi), d.add(Ai, Br))
                X[n-k] = (d.add(Ar, Bi), d.sub(Ai, Br))
        return X
    # even n fallback: naive (shouldn't be chosen)
    n_ = n
    X = []
    for k in range(n_):
        accr, acci = d.ZERO, d.ZERO
        cur = (accr, acci)
        for j in range(n_):
            wr, wi = omega_re_im(j*k, n_, sign)
            cur = cadd(d, cur, cmulw(d, xs[j], wr, wi))
        X.append(cur)
    return X

def pfa_dft(d, xs, sign, n1, n2):
    n = n1*n2
    # input map i = (n2*a + n1*b) mod n
    cols = []
    for a in range(n1):
        col = [xs[(n2*a + n1*b) % n] for b in range(n2)]
        cols.append(build_dft(d, col, sign))
    # now cols[a][k2]; for each k2: DFT over a
    Z = [[None]*n2 for _ in range(n1)]
    for k2 in range(n2):
        vec = [cols[a][k2] for a in range(n1)]
        out = build_dft(d, vec, sign)
        for k1 in range(n1): Z[k1][k2] = out[k1]
    X = [None]*n
    for k in range(n):
        X[k] = Z[k % n1][k % n2]
    return X

def ct_dft(d, xs, sign, n1, n2):
    # n = n1*n2 ; x[n2*a + b], a in [0,n1), b in [0,n2)
    n = n1*n2
    A = []
    for b in range(n2):
        vec = [xs[n2*a + b] for a in range(n1)]
        A.append(build_dft(d, vec, sign))  # A[b][k1]
    # twiddle: A[b][k1] *= w_n^{b*k1}
    for b in range(n2):
        for k1 in range(n1):
            if b and k1:
                wr, wi = omega_re_im(b*k1, n, sign)
                A[b][k1] = cmulw(d, A[b][k1], wr, wi)
    X = [None]*n
    for k1 in range(n1):
        vec = [A[b][k1] for b in range(n2)]
        out = build_dft(d, vec, sign)  # over b -> k2
        for k2 in range(n2):
            X[k1 + n1*k2] = out[k2]
    return X

def rader_dft(d, xs, sign):
    p = len(xs)
    g = primitive_root(p)
    q = p - 1
    ginv = pow(g, p-2, p)
    # a_u = x[g^u]
    a = [xs[pow(g, u, p)] for u in range(q)]
    ahat = build_dft(d, a, -1)
    # bhat_t = (1/q) * sum_u w_p^{sign*g^{-u}} e^{-2pi i u t / q}
    bhat = []
    for t in range(q):
        sr = mp.mpf(0); si = mp.mpf(0)
        for u in range(q):
            e = pow(ginv, u, p)
            wr, wi = omega_re_im(e, p, sign)
            er, ei = omega_re_im(u*t, q, -1)
            sr += wr*er - wi*ei
            si += wr*ei + wi*er
        bhat.append((sr/q, si/q))
    ch = [cmulw(d, ahat[t], bhat[t][0], bhat[t][1]) for t in range(q)]
    ch[0] = cadd(d, ch[0], xs[0])
    c = build_dft(d, ch, +1)
    X = [None]*p
    X[0] = cadd(d, xs[0], ahat[0])
    for v in range(q):
        X[pow(ginv, v, p)] = c[v]
    return X

# ---------------- plan search ----------------
def divisors(n):
    return [i for i in range(2, n) if n % i == 0]

def is_prime(n):
    if n < 2: return False
    i = 2
    while i*i <= n:
        if n % i == 0: return False
        i += 1
    return True

def candidate_plans(n):
    out = []
    if n <= 5 or n % 2 == 1:
        out.append(('direct',))
    if is_prime(n) and n > 5:
        out.append(('rader',))
    seen = set()
    for a in divisors(n):
        b = n // a
        if math.gcd(a, b) == 1:
            if (min(a,b), max(a,b)) not in seen:
                seen.add((min(a,b), max(a,b)))
                out.append(('pfa', a, b))
        out.append(('ct', a, b))
    return out

def measure_cost(n, plan, sign=-1):
    d = Dag()
    xs = [(d.inp(('r', j)), d.inp(('i', j))) for j in range(n)]
    X = build_dft(d, xs, sign, plan)
    # count reachable ops with fusion estimate
    roots = [i for xy in X for i in xy]
    seen = set()
    stack = list(roots)
    uses = {}
    order = []
    while stack:
        i = stack.pop()
        if i in seen: continue
        seen.add(i); order.append(i)
        t = d.nodes[i]
        for ch in t[1:]:
            if isinstance(ch, int):
                uses[ch] = uses.get(ch, 0) + 1
                stack.append(ch)
    adds = muls = negs = fused = 0
    addsubs = set()
    # find consumers kind
    consumers = {}
    for i in seen:
        t = d.nodes[i]
        for ch in t[1:]:
            if isinstance(ch, int):
                consumers.setdefault(ch, []).append(t[0])
    for i in seen:
        t = d.nodes[i]
        if t[0] in ('add','sub'): adds += 1
        elif t[0] == 'mul':
            muls += 1
            cons = consumers.get(i, [])
            if len(cons) == 1 and cons[0] in ('add','sub'):
                fused += 1
        elif t[0] == 'neg': negs += 1
    cost = adds + muls - fused + 0.4*negs
    return cost, adds, muls, negs, fused

BEST = {}
def best_plan(n):
    if n in BEST: return BEST[n]
    best = None; bestc = None
    for plan in candidate_plans(n):
        # to allow recursion, temporarily need sub-plans decided first
        try:
            c = measure_cost(n, plan)[0]
        except RecursionError:
            continue
        if bestc is None or c < bestc:
            bestc, best = c, plan
    BEST[n] = best
    PLAN_CHOICE[n] = (best, bestc)
    return best

if __name__ == '__main__':
    for n in (2,3,4,5,6,8,9,10,11,12,13,16,17,22,23,36,45,64):
        p = best_plan(n)
        c, adds, muls, negs, fused = measure_cost(n, p)
        print(f"n={n:3d} plan={p} cost={c:8.1f} adds={adds} muls={muls} negs={negs} fused={fused}")
        for plan in candidate_plans(n):
            cc = measure_cost(n, plan)
            print(f"      {plan}: cost={cc[0]:.1f} a={cc[1]} m={cc[2]} n={cc[3]} f={cc[4]}")

# ---------------- C emission ----------------

SCHED = 'dfs'
def topo_order(d, roots):
    order = []
    state = {}
    def visit(i):
        st = state.get(i, 0)
        if st == 2: return
        if st == 1: raise RuntimeError("cycle")
        state[i] = 1
        t = d.nodes[i]
        for ch in t[1:]:
            if isinstance(ch, int): visit(ch)
        state[i] = 2
        order.append(i)
    for r in roots: visit(r)
    if SCHED == 'dfs':
        return order
    # level order: depth = longest path from input
    depth = {}
    for i in order:
        t = d.nodes[i]
        ch = [c for c in t[1:] if isinstance(c, int)]
        depth[i] = 0 if not ch else 1 + max(depth[c] for c in ch)
    order2 = sorted(order, key=lambda i: (depth[i], order.index(i)))
    return order2

WIDTHS = {16: ('V16', 'K16'), 8: ('V8', 'K8'), 4: ('V4', 'K4'), 2: ('V2', 'K2'), 1: ('double', 'K1')}

def emit_codelet(n, width, sign=-1, name=None):
    d = Dag()
    xs = [(d.inp(('r', j)), d.inp(('i', j))) for j in range(n)]
    X = build_dft(d, xs, sign)
    ty, K = WIDTHS[width]
    name = name or f"fft{n}_w{width}"
    lines = []
    lines.append(f"static void {name}(const double* ri, const double* ii, ptrdiff_t is, double* ro, double* io, ptrdiff_t os) {{")
    # input loads first
    var = {}
    for j in range(n):
        ir, im_ = xs[j]
        var[ir] = f"xr{j}"; var[im_] = f"xi{j}"
        if width == 1:
            lines.append(f"  double xr{j} = ri[{j}*is]; double xi{j} = ii[{j}*is];")
        else:
            lines.append(f"  {ty} xr{j} = LD{width}(ri + {j}*is); {ty} xi{j} = LD{width}(ii + {j}*is);")
    # topo order over reachable nodes
    roots = [i for xy in X for i in xy]
    order = topo_order(d, roots)
    for i in order:
        if i in var: continue
        t = d.nodes[i]
        k = t[0]
        if k == 'in': continue
        if k == 'zero':
            var[i] = f"{K}(0.0)"; continue
        v = f"t{i}"
        if k == 'add':
            lines.append(f"  {ty} {v} = {var[t[1]]} + {var[t[2]]};")
        elif k == 'sub':
            lines.append(f"  {ty} {v} = {var[t[1]]} - {var[t[2]]};")
        elif k == 'neg':
            lines.append(f"  {ty} {v} = -{var[t[1]]};")
        elif k == 'mul':
            c = float(t[1][1])
            lines.append(f"  {ty} {v} = {K}({c!r}) * {var[t[2]]};")
        else:
            raise ValueError(k)
        var[i] = v
    for kk in range(n):
        r, im_ = X[kk]
        if width == 1:
            lines.append(f"  ro[{kk}*os] = {var[r]}; io[{kk}*os] = {var[im_]};")
        else:
            lines.append(f"  ST{width}(ro + {kk}*os, {var[r]}); ST{width}(io + {kk}*os, {var[im_]});")
    lines.append("}")
    return "\n".join(lines)

PRELUDE = r"""
#include <stddef.h>
#include <stdint.h>
typedef double V16 __attribute__((vector_size(128), aligned(8)));
typedef double V8 __attribute__((vector_size(64), aligned(8)));
typedef double V4 __attribute__((vector_size(32), aligned(8)));
typedef double V2 __attribute__((vector_size(16), aligned(8)));
#define K16(x) ((V16){(x),(x),(x),(x),(x),(x),(x),(x),(x),(x),(x),(x),(x),(x),(x),(x)})
#define K8(x) ((V8){(x),(x),(x),(x),(x),(x),(x),(x)})
#define K4(x) ((V4){(x),(x),(x),(x)})
#define K2(x) ((V2){(x),(x)})
#define K1(x) (x)
#define LD1(p) (*(p))
#define ST1(p,v) (*(p) = (v))
#define LD16(p) (*(const V16*)(p))
#define ST16(p,v) (*(V16*)(p) = (v))
#define LD8(p) (*(const V8*)(p))
#define LD4(p) (*(const V4*)(p))
#define LD2(p) (*(const V2*)(p))
#define ST8(p,v) (*(V8*)(p) = (v))
#define ST4(p,v) (*(V4*)(p) = (v))
#define ST2(p,v) (*(V2*)(p) = (v))
"""

if len(sys.argv) > 1 and sys.argv[1] == 'test':
    sizes = (2,3,4,5,6,8,13,17,23,36,45,64)
    parts = [PRELUDE]
    for n in sizes:
        parts.append(emit_codelet(n, 8))
        parts.append(emit_codelet(n, 1))
    # test driver
    parts.append(r"""
#include <string.h>
void run_test(int n, const double* ri, const double* ii, double* ro, double* io) {
  switch(n) {
""")
    for n in sizes:
        parts.append(f"    case {n}: fft{n}_w8(ri, ii, 8, ro, io, 8); break;")
    parts.append("""  }
}
void run_test1(int n, const double* ri, const double* ii, double* ro, double* io) {
  switch(n) {
""")
    for n in sizes:
        parts.append(f"    case {n}: fft{n}_w1(ri, ii, 1, ro, io, 1); break;")
    parts.append("""  }
}
""")
    open('codelets_test.c','w').write("\n".join(parts))
    print("wrote codelets_test.c")

# ---------------- staged emission for composite sizes ----------------
def emit_codelet_staged(n, width, fac=None, name=None, mapstore=False, pf=0, pfc=0):
    """Two-stage CT or PFA with explicit scratch arrays; bounded liveness.
    fac: (kind, n1, n2); pf: prefetch distance in doubles for inputs; pfc: same for c/out"""
    if fac is None:
        fac = {36: ('pfa', 4, 9), 45: ('pfa', 5, 9), 64: ('ct', 8, 8)}[n]
    kind, n1, n2 = fac
    assert n1 * n2 == n
    ty, K = WIDTHS[width]
    name = name or (f"fftmap{n}_w{width}" if mapstore else f"fft{n}_w{width}")
    sig = f"static void {name}(const double* ri, const double* ii, ptrdiff_t is, double* ro, double* io, ptrdiff_t os"
    if mapstore: sig += ", const double* cr, const double* ci"
    sig += ") {"
    lines = [sig]
    nb = n  # scratch complex count
    lines.append(f"  {ty} SR[{nb}] __attribute__((aligned(64)));")
    lines.append(f"  {ty} SI[{nb}] __attribute__((aligned(64)));")
    # ---- stage 1 ----
    if kind == 'ct':
        # groups over b in [0,n2): inputs x[n2*a+b], dft_n1 over a, twiddle w_n^{b*k1}, store [b*n1+k1]
        for b in range(n2):
            d = Dag()
            xs = [(d.inp(('r', j)), d.inp(('i', j))) for j in range(n1)]
            X = build_dft(d, xs, -1)
            X2 = []
            for k1 in range(n1):
                if b and k1:
                    wr, wi = omega_re_im(b*k1, n, -1)
                    X2.append(cmulw(d, X[k1], wr, wi))
                else:
                    X2.append(X[k1])
            ins = [(n2*a + b) for a in range(n1)]
            outs = [(b*n1 + k1) for k1 in range(n1)]
            lines.append(f"  {{ // stage1 group {b}")
            lines += emit_block(d, xs, X2, ins, outs, width, 'in', 'scr', pf=pf)
            lines.append("  }")
    else:
        # PFA stage1: for a in [0,n1): dft_n2 over b of x[(n2*a+n1*b)%n] -> Y[a][k2] store [a*n2+k2]
        for a in range(n1):
            d = Dag()
            xs = [(d.inp(('r', j)), d.inp(('i', j))) for j in range(n2)]
            X = build_dft(d, xs, -1)
            ins = [((n2*a + n1*b) % n) for b in range(n2)]
            outs = [(a*n2 + k2) for k2 in range(n2)]
            lines.append(f"  {{ // stage1 pfa group a={a}")
            lines += emit_block(d, xs, X, ins, outs, width, 'in', 'scr', pf=pf)
            lines.append("  }")
    # ---- stage 2 ----
    if kind == 'ct':
        for k1 in range(n1):
            d = Dag()
            xs = [(d.inp(('r', j)), d.inp(('i', j))) for j in range(n2)]
            X = build_dft(d, xs, -1)
            ins = [(b*n1 + k1) for b in range(n2)]
            outs = [(k1 + n1*k2) for k2 in range(n2)]
            lines.append(f"  {{ // stage2 group {k1}")
            lines += emit_block(d, xs, X, ins, outs, width, 'scr', 'mapout' if mapstore else 'out', pfc=pfc)
            lines.append("  }")
    else:
        for k2 in range(n2):
            d = Dag()
            xs = [(d.inp(('r', j)), d.inp(('i', j))) for j in range(n1)]
            X = build_dft(d, xs, -1)
            ins = [(a*n2 + k2) for a in range(n1)]
            # X[k] = Z[k%n1][k%n2]: outputs k with k%n2==k2, ordered by k1=k%n1
            outs = []
            for k1 in range(n1):
                # solve k ≡ k1 (n1), k ≡ k2 (n2) by CRT
                k = (k1 * n2 * pow(n2, -1, n1) + k2 * n1 * pow(n1, -1, n2)) % n
                outs.append(k)
            lines.append(f"  {{ // stage2 pfa group k2={k2}")
            lines += emit_block(d, xs, X, ins, outs, width, 'scr', 'mapout' if mapstore else 'out', pfc=pfc)
            lines.append("  }")
    lines.append("}")
    return "\n".join(lines)

def emit_block(d, xs, X, ins, outs, width, src, dst, pf=0, pfc=0):
    """emit one group: loads from src ('in' = ri/ii strided, 'scr' = SR/SI), compute, store to dst."""
    ty, K = WIDTHS[width]
    lines = []
    var = {}
    for idx, (j, xp) in enumerate(zip(ins, xs)):
        var[xp[0]] = f"a{idx}r"; var[xp[1]] = f"a{idx}i"
        if src == 'in':
            lines.append(f"    {ty} a{idx}r = LD{width}(ri + {j}*is); {ty} a{idx}i = LD{width}(ii + {j}*is);")
            if pf:
                lines.append(f"    __builtin_prefetch(ri + {j}*is + {pf}); __builtin_prefetch(ii + {j}*is + {pf});")
        else:
            lines.append(f"    {ty} a{idx}r = SR[{j}]; {ty} a{idx}i = SI[{j}];")
    roots = [i for xy in X for i in xy]
    order = topo_order(d, roots)
    for i in order:
        if i in var: continue
        t = d.nodes[i]
        k = t[0]
        if k == 'in': continue
        if k == 'zero':
            var[i] = f"{K}(0.0)"; continue
        v = f"t{i}"
        if k == 'add': lines.append(f"    {ty} {v} = {var[t[1]]} + {var[t[2]]};")
        elif k == 'sub': lines.append(f"    {ty} {v} = {var[t[1]]} - {var[t[2]]};")
        elif k == 'neg': lines.append(f"    {ty} {v} = -{var[t[1]]};")
        elif k == 'mul': lines.append(f"    {ty} {v} = {K}({float(t[1][1])!r}) * {var[t[2]]};")
        var[i] = v
    for pos, (j, xy) in enumerate(zip(outs, X)):
        r, im_ = xy
        if dst == 'scr':
            lines.append(f"    SR[{j}] = {var[r]}; SI[{j}] = {var[im_]};")
        elif dst == 'out':
            lines.append(f"    ST{width}(ro + {j}*os, {var[r]}); ST{width}(io + {j}*os, {var[im_]});")
        else:  # mapout
            if pfc:
                lines.append(f"    __builtin_prefetch(cr + {j}*os + {pfc}); __builtin_prefetch(ci + {j}*os + {pfc});")
                lines.append(f"    __builtin_prefetch(ro + {j}*os + {pfc}, 1); __builtin_prefetch(io + {j}*os + {pfc}, 1);")
            lines.append(f"    {{ V8 zr = {var[r]} + LD8(cr + {j}*os); V8 zi = {var[im_]} + LD8(ci + {j}*os); V8 mr, mi; MAPCALL(zr, zi, &mr, &mi); ST8(ro + {j}*os, mr); ST8(io + {j}*os, mi); }}")
    return lines

def emit_direct_staged(n, width, ktile=3, name=None, mapstore=False):
    """Folded direct DFT for odd n, staged: block A computes s/t (+X0), blocks B_g compute k-tiles."""
    assert n % 2 == 1
    h = (n - 1) // 2
    ty, K = WIDTHS[width]
    name = name or (f"fftmap{n}_w{width}" if mapstore else f"fft{n}_w{width}")
    sig = f"static void {name}(const double* ri, const double* ii, ptrdiff_t is, double* ro, double* io, ptrdiff_t os"
    if mapstore: sig += ", const double* cr, const double* ci"
    sig += ") {"
    lines = [sig]
    lines.append(f"  {ty} SR[{2*h+1}] __attribute__((aligned(64)));")
    lines.append(f"  {ty} SI[{2*h+1}] __attribute__((aligned(64)));")
    # block A
    d = Dag()
    xs = [(d.inp(('r', j)), d.inp(('i', j))) for j in range(n)]
    outsA = []   # (slot, (re,im)) for scratch
    outsX = []   # X0
    s_list = [None]*(h+1); t_list = [None]*(h+1)
    for j in range(1, h+1):
        s_list[j] = cadd(d, xs[j], xs[n-j])
        t_list[j] = csub(d, xs[j], xs[n-j])
    acc = xs[0]
    for j in range(1, h+1): acc = cadd(d, acc, s_list[j])
    # emit block A
    lines.append("  { // s/t + X0")
    blk = emit_block(d, xs, [acc], list(range(n)), [0], width, 'in', 'mapout' if mapstore else 'out')
    # need scratch stores too: slot 0 = x0, slots 1..h = s_j, h+j = t_j
    # remove the store line for X0? keep; add scratch stores before closing
    var_needed = []
    lines += blk[:-1]  # all except the out-store line? emit_block appends stores at end; outputs len 1 -> last line is store
    store_line = blk[-1]
    # scratch stores: we must re-emit with var map... simpler: rebuild with a custom approach below.
    lines = lines[:2]  # restart; do manual emission
    lines = [sig, f"  {ty} SR[{2*h+1}] __attribute__((aligned(64)));", f"  {ty} SI[{2*h+1}] __attribute__((aligned(64)));"]
    # manual block A emission
    d = Dag()
    xs = [(d.inp(('r', j)), d.inp(('i', j))) for j in range(n)]
    s_list = [None]*(h+1); t_list = [None]*(h+1)
    for j in range(1, h+1):
        s_list[j] = cadd(d, xs[j], xs[n-j])
        t_list[j] = csub(d, xs[j], xs[n-j])
    acc = xs[0]
    for j in range(1, h+1): acc = cadd(d, acc, s_list[j])
    roots = [acc[0], acc[1]]
    for j in range(1, h+1): roots += [s_list[j][0], s_list[j][1], t_list[j][0], t_list[j][1]]
    order = topo_order(d, roots)
    var = {}
    lines.append("  {")
    for j in range(n):
        var[xs[j][0]] = f"x{j}r"; var[xs[j][1]] = f"x{j}i"
        lines.append(f"    {ty} x{j}r = LD{width}(ri + {j}*is); {ty} x{j}i = LD{width}(ii + {j}*is);")
    def emit_nodes(order, var, ind='    '):
        out = []
        for i in order:
            if i in var: continue
            t = d.nodes[i]
            k = t[0]
            if k == 'in': continue
            if k == 'zero':
                var[i] = f"{K}(0.0)"; continue
            v = f"t{i}"
            if k == 'add': out.append(f"{ind}{ty} {v} = {var[t[1]]} + {var[t[2]]};")
            elif k == 'sub': out.append(f"{ind}{ty} {v} = {var[t[1]]} - {var[t[2]]};")
            elif k == 'neg': out.append(f"{ind}{ty} {v} = -{var[t[1]]};")
            elif k == 'mul': out.append(f"{ind}{ty} {v} = {K}({float(t[1][1])!r}) * {var[t[2]]};")
            var[i] = v
        return out
    lines += emit_nodes(order, var)
    lines.append(f"    SR[0] = {var[xs[0][0]]}; SI[0] = {var[xs[0][1]]};")
    for j in range(1, h+1):
        lines.append(f"    SR[{j}] = {var[s_list[j][0]]}; SI[{j}] = {var[s_list[j][1]]};")
        lines.append(f"    SR[{h+j}] = {var[t_list[j][0]]}; SI[{h+j}] = {var[t_list[j][1]]};")
    if mapstore:
        lines.append(f"    {{ V8 zr = {var[acc[0]]} + LD8(cr + 0); V8 zi = {var[acc[1]]} + LD8(ci + 0); V8 mr, mi; MAPCALL(zr, zi, &mr, &mi); ST8(ro + 0, mr); ST8(io + 0, mi); }}")
    else:
        lines.append(f"    ST{width}(ro + 0, {var[acc[0]]}); ST{width}(io + 0, {var[acc[1]]});")
    lines.append("  }")
    # blocks B: k tiles
    ks = list(range(1, h+1))
    for g0 in range(0, len(ks), ktile):
        tile = ks[g0:g0+ktile]
        d = Dag()
        x0 = (d.inp(('sr', 0)), d.inp(('si', 0)))
        s_in = {j: (d.inp(('sr', j)), d.inp(('si', j))) for j in range(1, h+1)}
        t_in = {j: (d.inp(('sr', h+j)), d.inp(('si', h+j))) for j in range(1, h+1)}
        var = {}
        lines.append(f"  {{ // k tile {tile}")
        var[x0[0]] = "s0r"; var[x0[1]] = "s0i"
        lines.append(f"    {ty} s0r = SR[0]; {ty} s0i = SI[0];")
        for j in range(1, h+1):
            var[s_in[j][0]] = f"s{j}r"; var[s_in[j][1]] = f"s{j}i"
            var[t_in[j][0]] = f"u{j}r"; var[t_in[j][1]] = f"u{j}i"
            lines.append(f"    {ty} s{j}r = SR[{j}]; {ty} s{j}i = SI[{j}];")
            lines.append(f"    {ty} u{j}r = SR[{h+j}]; {ty} u{j}i = SI[{h+j}];")
        outs = []
        for k in tile:
            Ar, Ai = x0
            Br, Bi = d.ZERO, d.ZERO
            for j in range(1, h+1):
                cjk, sjk = omega_re_im(j*k, n, -1)
                co = cjk; si = -sjk
                Ar = d.add(Ar, d.mul(co, s_in[j][0]))
                Ai = d.add(Ai, d.mul(co, s_in[j][1]))
                Br = d.add(Br, d.mul(si, t_in[j][0]))
                Bi = d.add(Bi, d.mul(si, t_in[j][1]))
            Xk = (d.add(Ar, Bi), d.sub(Ai, Br))
            Xnk = (d.sub(Ar, Bi), d.add(Ai, Br))
            outs.append((k, Xk)); outs.append((n-k, Xnk))
        roots = [c for _, xy in outs for c in xy]
        order = topo_order(d, roots)
        lines += emit_nodes(order, var)
        for k, xy in outs:
            r, im_ = xy
            if mapstore:
                lines.append(f"    {{ V8 zr = {var[r]} + LD8(cr + {k}*os); V8 zi = {var[im_]} + LD8(ci + {k}*os); V8 mr, mi; MAPCALL(zr, zi, &mr, &mi); ST8(ro + {k}*os, mr); ST8(io + {k}*os, mi); }}")
            else:
                lines.append(f"    ST{width}(ro + {k}*os, {var[r]}); ST{width}(io + {k}*os, {var[im_]});")
        lines.append("  }")
    lines.append("}")
    return "\n".join(lines)

# ---------------- generalized block-program emission ----------------
class Prog:
    """Sequence of straight-line blocks communicating via scratch slots (complex)."""
    def __init__(self, width, mapstore=False):
        self.width = width
        self.mapstore = mapstore
        self.nscr = 0
        self.blocks = []   # (comment, [(ref_in)], fn(dag, xs)->[(refs_out)])
        self.lines = []
    def scr(self, count):
        base = self.nscr
        self.nscr += count
        return list(range(base, base + count))
    def block(self, comment, ins, compute, outs):
        """ins: list of ('in', j) or ('scr', slot); compute(d, xs)->list of complex pairs;
        outs: list of ('scr', slot) or ('out', k) matching compute results."""
        self.blocks.append((comment, ins, compute, outs))
    def emit(self, name):
        ty, K = WIDTHS[self.width]
        W = self.width
        sig = f"static void {name}(const double* ri, const double* ii, ptrdiff_t is, double* ro, double* io, ptrdiff_t os"
        if self.mapstore: sig += ", const double* cr, const double* ci"
        sig += ") {"
        lines = [sig]
        if self.nscr:
            lines.append(f"  {ty} SR[{self.nscr}] __attribute__((aligned(64)));")
            lines.append(f"  {ty} SI[{self.nscr}] __attribute__((aligned(64)));")
        for comment, ins, compute, outs in self.blocks:
            d = Dag()
            xs = [(d.inp(('r', t)), d.inp(('i', t))) for t in range(len(ins))]
            res = compute(d, xs)
            assert len(res) == len(outs), (comment, len(res), len(outs))
            lines.append(f"  {{ // {comment}")
            var = {}
            for t, (src, xp) in enumerate(zip(ins, xs)):
                var[xp[0]] = f"a{t}r"; var[xp[1]] = f"a{t}i"
                kind, j = src
                if kind == 'in':
                    if W == 1:
                        lines.append(f"    double a{t}r = ri[{j}*is]; double a{t}i = ii[{j}*is];")
                    else:
                        lines.append(f"    {ty} a{t}r = LD{W}(ri + {j}*is); {ty} a{t}i = LD{W}(ii + {j}*is);")
                else:
                    lines.append(f"    {ty} a{t}r = SR[{j}]; {ty} a{t}i = SI[{j}];")
            roots = [c for xy in res for c in xy]
            order = topo_order(d, roots)
            for i in order:
                if i in var: continue
                tt = d.nodes[i]
                k = tt[0]
                if k == 'in': continue
                if k == 'zero':
                    var[i] = f"{K}(0.0)"; continue
                v = f"t{i}"
                if k == 'add': lines.append(f"    {ty} {v} = {var[tt[1]]} + {var[tt[2]]};")
                elif k == 'sub': lines.append(f"    {ty} {v} = {var[tt[1]]} - {var[tt[2]]};")
                elif k == 'neg': lines.append(f"    {ty} {v} = -{var[tt[1]]};")
                elif k == 'mul': lines.append(f"    {ty} {v} = {K}({float(tt[1][1])!r}) * {var[tt[2]]};")
                var[i] = v
            for (dk, dj), xy in zip(outs, res):
                r, im_ = xy
                if dk == 'scr':
                    lines.append(f"    SR[{dj}] = {var[r]}; SI[{dj}] = {var[im_]};")
                elif dk == 'out':
                    if W == 1:
                        lines.append(f"    ro[{dj}*os] = {var[r]}; io[{dj}*os] = {var[im_]};")
                    else:
                        lines.append(f"    ST{W}(ro + {dj}*os, {var[r]}); ST{W}(io + {dj}*os, {var[im_]});")
                else:  # mapout
                    lines.append(f"    {{ V8 zr = {var[r]} + LD8(cr + {dj}*os); V8 zi = {var[im_]} + LD8(ci + {dj}*os); V8 mr, mi; MAPCALL(zr, zi, &mr, &mi); ST8(ro + {dj}*os, mr); ST8(io + {dj}*os, mi); }}")
            lines.append("  }")
        lines.append("}")
        return "\n".join(lines)

DIRECT_SPLIT = set()
def plan_blocks(prog, refs, sign, dsts, maxblock=13, pre=None):
    """Recursively add blocks computing DFT(refs) with given sign, storing to dsts.
    refs: list of ('in',j)/('scr',slot); dsts: list of ('scr',slot)/('out',k)/('mapout',k)."""
    n = len(refs)
    plan = best_plan(n)
    if n <= maxblock and (n <= 5 or plan[0] in ('direct',)):
        if (not pre) and n in DIRECT_SPLIT and n % 2 == 1 and n > 7:
            h = (n-1)//2
            st = prog.scr(2*h+1)
            def compA(d, xs, h=h, n=n, sign=sign):
                res = [xs[0]]
                acc = xs[0]
                for j in range(1, h+1):
                    res.append(cadd(d, xs[j], xs[n-j]))
                    res.append(csub(d, xs[j], xs[n-j]))
                for j in range(1, h+1): acc = cadd(d, acc, res[1 + 2*(j-1)])
                return res + [acc]
            outsA = [('scr', st[0])]
            for j in range(1, h+1):
                outsA += [('scr', st[j]), ('scr', st[h+j])]
            outsA.append(dsts[0])
            prog.block(f"direct{n} fold", refs, compA, outsA)
            ktile = 3
            ks = list(range(1, h+1))
            for g0 in range(0, len(ks), ktile):
                tile = ks[g0:g0+ktile]
                ins = [('scr', st[0])] + [('scr', st[j]) for j in range(1, h+1)] + [('scr', st[h+j]) for j in range(1, h+1)]
                outs = []
                def compB(d, xs, tile=tile, h=h, n=n, sign=sign):
                    x0 = xs[0]
                    s_in = {j: xs[j] for j in range(1, h+1)}
                    t_in = {j: xs[h+j] for j in range(1, h+1)}
                    res = []
                    for k in tile:
                        Ar, Ai = x0
                        Br, Bi = d.ZERO, d.ZERO
                        for j in range(1, h+1):
                            cjk, sjk = omega_re_im(j*k, n, -1)
                            co = cjk; si = -sjk
                            Ar = d.add(Ar, d.mul(co, s_in[j][0]))
                            Ai = d.add(Ai, d.mul(co, s_in[j][1]))
                            Br = d.add(Br, d.mul(si, t_in[j][0]))
                            Bi = d.add(Bi, d.mul(si, t_in[j][1]))
                        if sign < 0:
                            res.append((d.add(Ar, Bi), d.sub(Ai, Br)))
                            res.append((d.sub(Ar, Bi), d.add(Ai, Br)))
                        else:
                            res.append((d.sub(Ar, Bi), d.add(Ai, Br)))
                            res.append((d.add(Ar, Bi), d.sub(Ai, Br)))
                    return res
                for k in tile:
                    outs += [dsts[k], dsts[n-k]]
                prog.block(f"direct{n} ktile {tile}", ins, compB, outs)
            return
        extra = []
        if pre:
            for t in pre:
                if pre[t][2] is not None and pre[t][2] not in extra:
                    extra.append(pre[t][2])
        def comp(d, xs, sign=sign, n=n, pre=pre, nex=len(extra), extra=tuple(extra)):
            base = list(xs[:len(xs)-nex]) if nex else list(xs)
            if pre:
                for t, (wr, wi, ax) in pre.items():
                    v = cmulw(d, base[t], wr, wi)
                    if ax is not None:
                        v = cadd(d, v, xs[len(xs)-nex + list(extra).index(ax)])
                    base[t] = v
            return build_dft(d, base, sign)
        prog.block(f"dft{n} sign={sign}", list(refs) + extra, comp, dsts)
        return
    if plan[0] == 'pfa':
        n1, n2 = plan[1], plan[2]
        mid = prog.scr(n)
        # stage1: for a: DFT_n2 over b of x[(n2*a+n1*b)%n] -> mid[a*n2+k2]
        for a in range(n1):
            idxs = [(n2*a + n1*b) % n for b in range(n2)]
            ins = [refs[i] for i in idxs]
            outs = [('scr', mid[a*n2 + k2]) for k2 in range(n2)]
            sub_pre = None
            if pre:
                sub_pre = {b: pre[i] for b, i in enumerate(idxs) if i in pre} or None
            plan_blocks(prog, ins, sign, outs, maxblock, pre=sub_pre)
        # stage2: for k2: DFT_n1 over a of mid[a*n2+k2] -> X[crt(k1,k2)]
        if n1 <= 4:
            grp = max(1, 8 // n1)
            for k20 in range(0, n2, grp):
                k2s = list(range(k20, min(k20+grp, n2)))
                ins = []
                for k2 in k2s:
                    ins += [('scr', mid[a*n2 + k2]) for a in range(n1)]
                outs = []
                for k2 in k2s:
                    for k1 in range(n1):
                        k = (k1 * n2 * pow(n2, -1, n1) + k2 * n1 * pow(n1, -1, n2)) % n
                        outs.append(dsts[k])
                def comp(d, xs, n1=n1, cnt=len(k2s)):
                    res = []
                    for t in range(cnt):
                        res += build_dft(d, xs[t*n1:(t+1)*n1], sign)
                    return res
                prog.block(f"pfa{n} stage2 merged k2={k2s[0]}..", ins, comp, outs)
        else:
            for k2 in range(n2):
                ins = [('scr', mid[a*n2 + k2]) for a in range(n1)]
                outs = []
                for k1 in range(n1):
                    k = (k1 * n2 * pow(n2, -1, n1) + k2 * n1 * pow(n1, -1, n2)) % n
                    outs.append(dsts[k])
                plan_blocks(prog, ins, sign, outs, maxblock)
        return
    if plan[0] == 'ct':
        n1, n2 = plan[1], plan[2]
        mid = prog.scr(n)
        for b in range(n2):
            idxs = [n2*a + b for a in range(n1)]
            ins = [refs[i] for i in idxs]
            outs = [('scr', mid[b*n1 + k1]) for k1 in range(n1)]
            sub_pre = None
            if pre:
                sub_pre = {a: pre[i] for a, i in enumerate(idxs) if i in pre} or None
            extra = []
            if sub_pre:
                for t in sub_pre:
                    if sub_pre[t][2] is not None and sub_pre[t][2] not in extra:
                        extra.append(sub_pre[t][2])
            ins = ins + extra
            def comp(d, xs, sign=sign, b=b, n1=n1, n=n, sub_pre=sub_pre, extra=tuple(extra)):
                nex = len(extra)
                base = list(xs[:len(xs)-nex]) if nex else list(xs)
                if sub_pre:
                    for t, (wr, wi, ax) in sub_pre.items():
                        v = cmulw(d, base[t], wr, wi)
                        if ax is not None:
                            v = cadd(d, v, xs[len(xs)-nex + list(extra).index(ax)])
                        base[t] = v
                X = build_dft(d, base, sign)
                X2 = []
                for k1 in range(n1):
                    if b and k1:
                        wr, wi = omega_re_im(b*k1, n, sign)
                        X2.append(cmulw(d, X[k1], wr, wi))
                    else:
                        X2.append(X[k1])
                return X2
            prog.block(f"ct{n} stage1 b={b}", ins, comp, outs)
        if n2 <= 4:
            grp = max(1, 8 // n2)
            for k10 in range(0, n1, grp):
                k1s = list(range(k10, min(k10+grp, n1)))
                ins = []
                for k1 in k1s:
                    ins += [('scr', mid[b*n1 + k1]) for b in range(n2)]
                outs = []
                for k1 in k1s:
                    outs += [dsts[k1 + n1*k2] for k2 in range(n2)]
                def comp(d, xs, n2=n2, cnt=len(k1s)):
                    res = []
                    for t in range(cnt):
                        res += build_dft(d, xs[t*n2:(t+1)*n2], sign)
                    return res
                prog.block(f"ct{n} stage2 merged", ins, comp, outs)
        else:
            for k1 in range(n1):
                ins = [('scr', mid[b*n1 + k1]) for b in range(n2)]
                outs = [dsts[k1 + n1*k2] for k2 in range(n2)]
                plan_blocks(prog, ins, sign, outs, maxblock)
        return
    if plan[0] == 'rader' or (is_prime(n) and n > maxblock):
        rader_blocks(prog, refs, sign, dsts, maxblock, pre=pre)
        return
    raise ValueError((n, plan))

def rader_blocks(prog, refs, sign, dsts, maxblock, pre=None):
    p = len(refs)
    if pre:
        # materialize pre-multipliers into scratch before Rader
        mat = prog.scr(p)
        exx = []
        for t in pre:
            if pre[t][2] is not None and pre[t][2] not in exx:
                exx.append(pre[t][2])
        ins = list(refs) + exx
        def compM(d, xs, pre=pre, np_=p, nex=len(exx), exx=tuple(exx)):
            base = list(xs[:np_])
            for t, (wr, wi, ax) in pre.items():
                v = cmulw(d, base[t], wr, wi)
                if ax is not None:
                    v = cadd(d, v, xs[np_ + list(exx).index(ax)])
                base[t] = v
            return base
        prog.block(f"pre-mat rader{p}", ins, compM, [('scr', s) for s in mat])
        refs = [('scr', s) for s in mat]

    g = primitive_root(p)
    q = p - 1
    ginv = pow(g, p-2, p)
    a_refs = [refs[pow(g, u, p)] for u in range(q)]
    ahat = prog.scr(q)
    plan_blocks(prog, a_refs, -1, [('scr', s) for s in ahat], maxblock)
    # pointwise multiply by bhat/q, add x0 into slot0; also X[0] = x0 + ahat[0]
    bh = []
    for t in range(q):
        sr = mp.mpf(0); si = mp.mpf(0)
        for u in range(q):
            e = pow(ginv, u, p)
            wr, wi = omega_re_im(e, p, sign)
            er, ei = omega_re_im(u*t, q, -1)
            sr += wr*er - wi*ei
            si += wr*ei + wi*er
        bh.append((sr/q, si/q))
    # stage x0 into scratch first (in-place safety: output stores may clobber ri)
    x0s = prog.scr(1)[0]
    def comp_savex0(d, xs):
        return [xs[0]]
    prog.block(f"rader{p} save x0", [refs[0]], comp_savex0, [('scr', x0s)])
    # X0 = x0 + ahat[0]
    def compX0(d, xs):
        return [cadd(d, xs[0], xs[1])]
    prog.block(f"rader{p} X0", [('scr', ahat[0]), ('scr', x0s)], compX0, [dsts[0]])
    # inverse DFT_q with pointwise multiplies fused into first consumption
    final = [None]*q
    for v in range(q):
        final[v] = dsts[pow(ginv, v, p)]
    pre = {t: (bh[t][0], bh[t][1], ('scr', x0s) if t == 0 else None) for t in range(q)}
    plan_blocks(prog, [('scr', s) for s in ahat], +1, final, maxblock, pre=pre)

def emit_rader(n, width, mapstore=False, name=None, maxblock=13):
    prog = Prog(width, mapstore)
    refs = [('in', j) for j in range(n)]
    dk = 'mapout' if mapstore else 'out'
    dsts = [(dk, k) for k in range(n)]
    plan_blocks(prog, refs, -1, dsts, maxblock)
    name = name or (f"fftmap{n}_w{width}" if mapstore else f"fft{n}_w{width}")
    return prog.emit(name)

def emit_ax_stage1(r, XS, shape, g, name):
    """Stage-1 of blocked x-FFT: FFT_r over planes with twiddle row g baked in.
    shape 'strided': plane stride r*XS; 'consec': plane stride XS. In-place, lanes width 8."""
    ps = r*XS if shape == 'strided' else XS
    d = Dag()
    xs = [(d.inp(('r', j)), d.inp(('i', j))) for j in range(r)]
    X = build_dft(d, xs, -1)
    X2 = []
    for k1 in range(r):
        if g and k1:
            wr, wi = omega_re_im(g*k1, r*r, -1)
            X2.append(cmulw(d, X[k1], wr, wi))
        else:
            X2.append(X[k1])
    ty, K = WIDTHS[8]
    lines = [f"static inline __attribute__((always_inline)) void {name}(double* ri, double* ii) {{"]
    var = {}
    for j in range(r):
        var[xs[j][0]] = f"x{j}r"; var[xs[j][1]] = f"x{j}i"
        lines.append(f"  {ty} x{j}r = LD8(ri + {j*ps}); {ty} x{j}i = LD8(ii + {j*ps});")
    roots = [c for xy in X2 for c in xy]
    order = topo_order(d, roots)
    for i in order:
        if i in var: continue
        t = d.nodes[i]
        k = t[0]
        if k == 'in': continue
        if k == 'zero':
            var[i] = f"{K}(0.0)"; continue
        v = f"t{i}"
        if k == 'add': lines.append(f"  {ty} {v} = {var[t[1]]} + {var[t[2]]};")
        elif k == 'sub': lines.append(f"  {ty} {v} = {var[t[1]]} - {var[t[2]]};")
        elif k == 'neg': lines.append(f"  {ty} {v} = -{var[t[1]]};")
        elif k == 'mul': lines.append(f"  {ty} {v} = {K}({float(t[1][1])!r}) * {var[t[2]]};")
        var[i] = v
    for k1 in range(r):
        rr, ii_ = X2[k1]
        lines.append(f"  ST8(ri + {k1*ps}, {var[rr]}); ST8(ii + {k1*ps}, {var[ii_]});")
    lines.append("}")
    return "\n".join(lines)

def emit_ax_stage2map(r, XS, shape, name):
    """Stage-2: FFT_r over planes + c + map, in-place. shape as above."""
    ps = r*XS if shape == 'strided' else XS
    d = Dag()
    xs = [(d.inp(('r', j)), d.inp(('i', j))) for j in range(r)]
    X = build_dft(d, xs, -1)
    ty, K = WIDTHS[8]
    lines = [f"static inline __attribute__((always_inline)) void {name}(double* ri, double* ii, const double* cr, const double* ci) {{"]
    var = {}
    for j in range(r):
        var[xs[j][0]] = f"x{j}r"; var[xs[j][1]] = f"x{j}i"
        lines.append(f"  {ty} x{j}r = LD8(ri + {j*ps}); {ty} x{j}i = LD8(ii + {j*ps});")
    roots = [c for xy in X for c in xy]
    order = topo_order(d, roots)
    for i in order:
        if i in var: continue
        t = d.nodes[i]
        k = t[0]
        if k == 'in': continue
        if k == 'zero':
            var[i] = f"{K}(0.0)"; continue
        v = f"t{i}"
        if k == 'add': lines.append(f"  {ty} {v} = {var[t[1]]} + {var[t[2]]};")
        elif k == 'sub': lines.append(f"  {ty} {v} = {var[t[1]]} - {var[t[2]]};")
        elif k == 'neg': lines.append(f"  {ty} {v} = -{var[t[1]]};")
        elif k == 'mul': lines.append(f"  {ty} {v} = {K}({float(t[1][1])!r}) * {var[t[2]]};")
        var[i] = v
    for k2 in range(r):
        rr, ii_ = X[k2]
        off = k2*ps
        lines.append(f"  {{ V8 zr = {var[rr]} + LD8(cr + {off}); V8 zi = {var[ii_]} + LD8(ci + {off}); V8 mr, mi; MAPCALL(zr, zi, &mr, &mi); ST8(ri + {off}, mr); ST8(ii + {off}, mi); }}")
    lines.append("}")
    return "\n".join(lines)

def _emit_simple(d, X2, xs, loads, stores, name, args, mapstore=False):
    ty, K = WIDTHS[8]
    lines = [f"static inline __attribute__((always_inline)) void {name}({args}) {{"]
    var = {}
    for j, (ld_r, ld_i) in enumerate(loads):
        var[xs[j][0]] = f"x{j}r"; var[xs[j][1]] = f"x{j}i"
        lines.append(f"  {ty} x{j}r = LD8({ld_r}); {ty} x{j}i = LD8({ld_i});")
    roots = [c for xy in X2 for c in xy]
    order = topo_order(d, roots)
    for i in order:
        if i in var: continue
        t = d.nodes[i]
        k = t[0]
        if k == 'in': continue
        if k == 'zero':
            var[i] = f"{K}(0.0)"; continue
        v = f"t{i}"
        if k == 'add': lines.append(f"  {ty} {v} = {var[t[1]]} + {var[t[2]]};")
        elif k == 'sub': lines.append(f"  {ty} {v} = {var[t[1]]} - {var[t[2]]};")
        elif k == 'neg': lines.append(f"  {ty} {v} = -{var[t[1]]};")
        elif k == 'mul': lines.append(f"  {ty} {v} = {K}({float(t[1][1])!r}) * {var[t[2]]};")
        var[i] = v
    for kk, (st_r, st_i, cr_e, ci_e) in enumerate(stores):
        r, im_ = X2[kk]
        if mapstore:
            lines.append(f"  {{ V8 zr = {var[r]} + LD8({cr_e}); V8 zi = {var[im_]} + LD8({ci_e}); V8 mr, mi; MAPCALL(zr, zi, &mr, &mi); ST8({st_r}, mr); ST8({st_i}, mi); }}")
        else:
            lines.append(f"  ST8({st_r}, {var[r]}); ST8({st_i}, {var[im_]});")
    lines.append("}")
    return "\n".join(lines)

def emit_oop_stage1(n1, n2, XS, b, name):
    """x = n2*a + b; FFT_{n1} over a from ri/ii (stride n2*XS), twiddle w_n^{b*k1},
    write to ro/io at k1*XS (consecutive)."""
    n = n1*n2
    d = Dag()
    xs = [(d.inp(('r', j)), d.inp(('i', j))) for j in range(n1)]
    X = build_dft(d, xs, -1)
    X2 = []
    for k1 in range(n1):
        if b and k1:
            wr, wi = omega_re_im(b*k1, n, -1)
            X2.append(cmulw(d, X[k1], wr, wi))
        else:
            X2.append(X[k1])
    loads = [(f"ri + {a*n2*XS}", f"ii + {a*n2*XS}") for a in range(n1)]
    stores = [(f"ro + {k1*XS}", f"io + {k1*XS}", None, None) for k1 in range(n1)]
    return _emit_simple(d, X2, xs, loads, stores, name,
        "const double* ri, const double* ii, double* ro, double* io")

def emit_oop_stage2map(n1, n2, XS, name):
    """FFT_{n2} over b reading ri/ii at b*(n1*XS); +c+map; write ro/io at k2*(n1*XS)."""
    d = Dag()
    xs = [(d.inp(('r', j)), d.inp(('i', j))) for j in range(n2)]
    X = build_dft(d, xs, -1)
    loads = [(f"ri + {b*n1*XS}", f"ii + {b*n1*XS}") for b in range(n2)]
    stores = [(f"ro + {k2*n1*XS}", f"io + {k2*n1*XS}", f"cr + {k2*n1*XS}", f"ci + {k2*n1*XS}") for k2 in range(n2)]
    return _emit_simple(d, X, xs, loads, stores, name,
        "const double* ri, const double* ii, double* ro, double* io, const double* cr, const double* ci", mapstore=True)


def emit_ax45_stage1(n1, n2, XS, g, name, consec):
    """45: FFT_{n1} over a, group g (=b), twiddle w45^{g*k1}.
    consec: positions = base + a*XS (consecutive planes); else stride n2*XS."""
    ps = XS if consec else n2*XS
    n = n1*n2
    d = Dag()
    xs = [(d.inp(('r', j)), d.inp(('i', j))) for j in range(n1)]
    X = build_dft(d, xs, -1)
    X2 = []
    for k1 in range(n1):
        if g and k1:
            wr, wi = omega_re_im(g*k1, n, -1)
            X2.append(cmulw(d, X[k1], wr, wi))
        else:
            X2.append(X[k1])
    loads = [(f"ri + {a*ps}", f"ii + {a*ps}") for a in range(n1)]
    stores = [(f"ri + {k1*ps}", f"ii + {k1*ps}", None, None) for k1 in range(n1)]
    return _emit_simple(d, X2, xs, loads, stores, name, "double* ri, double* ii")

def emit_ax45_stage2map(n2, XS, name, consec):
    """45 stage2: FFT_{n2} over b + c + map, in-place at positions base + b*step."""
    ps = XS if consec else 9*XS if n2 == 5 else None
    if not consec and n2 == 5:
        ps = 9*XS
    d = Dag()
    xs = [(d.inp(('r', j)), d.inp(('i', j))) for j in range(n2)]
    X = build_dft(d, xs, -1)
    loads = [(f"ri + {b*ps}", f"ii + {b*ps}") for b in range(n2)]
    stores = [(f"ri + {k2*ps}", f"ii + {k2*ps}", f"cr + {k2*ps}", f"ci + {k2*ps}") for k2 in range(n2)]
    return _emit_simple(d, X, xs, loads, stores, name,
        "double* ri, double* ii, const double* cr, const double* ci", mapstore=True)
