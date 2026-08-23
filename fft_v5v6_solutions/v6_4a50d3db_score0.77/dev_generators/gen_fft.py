#!/usr/bin/env python3
"""
Codelet generator: emits implementation.c with straight-line AVX-512 DFT
codelets specialized for sizes 6,8,13,17,23,36,45,64 plus drivers for the
iterated map  x <- (F3(x)+c)/(1+|F3(x)+c|).
Run at development time; output C is committed.
"""
import numpy as np, math, struct
from functools import lru_cache

LD = np.longdouble
PI = LD('3.14159265358979323846264338327950288419716939937510')

# ---------------- expression IR with hash consing ----------------
class Prog:
    def __init__(self):
        self.nodes = []     # (op, a, b, const)
        self.memo = {}
    def _mk(self, op, a=-1, b=-1, c=None):
        key = (op, a, b, None if c is None else struct.pack('<d', c))
        idx = self.memo.get(key)
        if idx is None:
            idx = len(self.nodes)
            self.nodes.append((op, a, b, c))
            self.memo[key] = idx
        return idx
    def inp(self, name):
        # name unique per input slot
        key = ('in', name)
        idx = self.memo.get(key)
        if idx is None:
            idx = len(self.nodes)
            self.nodes.append(('in', name, -1, None))
            self.memo[key] = idx
        return idx
    def op(self, i): return self.nodes[i][0]
    def add(self, a, b):
        if self.op(a) == 'neg' and self.op(b) == 'neg':
            return self.neg(self.add(self.nodes[a][1], self.nodes[b][1]))
        if self.op(b) == 'neg': return self.sub(a, self.nodes[b][1])
        if self.op(a) == 'neg': return self.sub(b, self.nodes[a][1])
        if b < a: a, b = b, a
        return self._mk('add', a, b)
    def sub(self, a, b):
        if self.op(b) == 'neg': return self.add(a, self.nodes[b][1])
        if self.op(a) == 'neg': return self.neg(self.add(self.nodes[a][1], b))
        if a == b:
            raise ValueError("sub(a,a) -> zero not supported")
        return self._mk('sub', a, b)
    def neg(self, a):
        if self.op(a) == 'neg': return self.nodes[a][1]
        if self.op(a) == 'mul': return self.mul(-self.nodes[a][3], self.nodes[a][1])
        return self._mk('neg', a)
    def mul(self, c, a):
        c = float(c)
        assert c != 0.0
        if c == 1.0: return a
        if c == -1.0: return self.neg(a)
        if self.op(a) == 'neg': return self._mk('mul', self.nodes[a][1], c=-c)
        return self._mk('mul', a, c=c)

# complex helpers: z = (re_idx, im_idx)
def cadd(P,x,y): return (P.add(x[0],y[0]), P.add(x[1],y[1]))
def csub(P,x,y): return (P.sub(x[0],y[0]), P.sub(x[1],y[1]))
def cneg(P,x): return (P.neg(x[0]), P.neg(x[1]))
def cmuli(P,x):   # multiply by +i: (a+bi)i = -b + ai
    return (P.neg(x[1]), x[0])
def cmulni(P,x):  # multiply by -i: b - ai
    return (x[1], P.neg(x[0]))
def cscale(P,c,x): return (P.mul(c,x[0]), P.mul(c,x[1]))
def cmulc(P, cr, ci, x):
    # multiply by complex constant cr + i*ci
    if ci == 0.0: return cscale(P, cr, x)
    if cr == 0.0:
        if ci == 1.0: return cmuli(P, x)
        if ci == -1.0: return cmulni(P, x)
        return (P.mul(-ci, x[1]), P.mul(ci, x[0]))
    if cr == 1.0 and ci == 0.0: return x
    re = P.sub(P.mul(cr,x[0]), P.mul(ci,x[1])) if ci>0 or True else None
    re = P.sub(P.mul(cr,x[0]), P.mul(ci,x[1]))
    im = P.add(P.mul(ci,x[0]), P.mul(cr,x[1]))
    return (re, im)
def csum(P, lst):
    # balanced tree sum
    lst = list(lst)
    while len(lst) > 1:
        nxt = []
        for i in range(0, len(lst)-1, 2):
            nxt.append(cadd(P, lst[i], lst[i+1]))
        if len(lst) % 2: nxt.append(lst[-1])
        lst = nxt
    return lst[0]

# ---------------- exact-ish twiddle constants ----------------
def twid(n, k):
    """(cos, sin) of -2*pi*k/n rounded from long double; exact for multiples of n/4."""
    k = k % n
    if 4*k % n == 0:
        q = (4*k)//n
        return [(1.0,0.0),(0.0,-1.0),(-1.0,0.0),(0.0,1.0)][q]
    ang = LD(-2)*PI*LD(k)/LD(n)
    return (float(np.cos(ang)), float(np.sin(ang)))

def cospi2(n, k):  # cos(2 pi k / n), sin(2 pi k /n) exact-ish
    c,s = twid(n, -k % n)
    return c, s

# ---------------- DFT builders ----------------
PLAN = {}   # n -> tuple like ('pfa',a,b) ('ct',p) ('rader',) ('dsym',) ('base',)

def dft(P, n, xs):
    assert len(xs) == n
    if n == 1: return list(xs)
    plan = PLAN.get(n)
    if plan is None: raise ValueError(f"no plan for {n}")
    kind = plan[0]
    if kind == 'base':
        return base_dft(P, n, xs)
    if kind == 'dsym':
        return direct_sym(P, n, xs)
    if kind == 'pfa':
        return pfa(P, plan[1], plan[2], xs)
    if kind == 'ct':
        return ct(P, plan[1], n, xs)
    if kind == 'rader':
        return rader(P, n, xs)
    if kind == 'sr':
        return splitradix(P, n, xs)
    raise ValueError(kind)

def base_dft(P, n, xs):
    if n == 2:
        return [cadd(P,xs[0],xs[1]), csub(P,xs[0],xs[1])]
    if n == 3:
        t = cadd(P, xs[1], xs[2]); u = csub(P, xs[1], xs[2])
        X0 = cadd(P, xs[0], t)
        # v = x0 - t/2
        v = (P.add(xs[0][0], P.mul(-0.5, t[0])), P.add(xs[0][1], P.mul(-0.5, t[1])))
        s = float(np.sqrt(LD(3))/LD(2))
        w = cscale(P, s, u)   # w = s*u
        # X1 = v - i*w ; X2 = v + i*w   (check: ang = -2pi/3: c=-1/2, s=-sqrt3/2; X1 = x0 + c*t + i*s*u = v - i*(sqrt3/2)u )
        X1 = cadd(P, v, cmulni(P, w))
        X2 = cadd(P, v, cmuli(P, w))
        return [X0, X1, X2]
    if n == 4:
        t0 = cadd(P,xs[0],xs[2]); t1 = csub(P,xs[0],xs[2])
        t2 = cadd(P,xs[1],xs[3]); t3 = csub(P,xs[1],xs[3])
        return [cadd(P,t0,t2), cadd(P,t1,cmulni(P,t3)), csub(P,t0,t2), cadd(P,t1,cmuli(P,t3))]
    raise ValueError(n)

def direct_sym(P, p, xs):
    # odd prime direct with +-k symmetry
    h = (p-1)//2
    t = [None]*(h+1); u = [None]*(h+1)
    for j in range(1, h+1):
        t[j] = cadd(P, xs[j], xs[p-j])
        u[j] = csub(P, xs[j], xs[p-j])
    X = [None]*p
    X[0] = csum(P, [xs[0]] + [t[j] for j in range(1,h+1)])
    def afma(acc, coeff, v):
        if coeff == 0.0: return acc
        if acc is None:
            return v if coeff == 1.0 else P.neg(v) if coeff == -1.0 else P.mul(coeff, v)
        if coeff == 1.0: return P.add(acc, v)
        if coeff == -1.0: return P.sub(acc, v)
        return P.add(acc, P.mul(coeff, v))
    for k in range(1, h+1):
        Cr, Ci = xs[0]
        for j in range(1, h+1):
            c,_ = cospi2(p, j*k)
            Cr = afma(Cr, c, t[j][0]); Ci = afma(Ci, c, t[j][1])
        Sr, Si = None, None
        for j in range(1, h+1):
            _,s = cospi2(p, j*k)
            Sr = afma(Sr, s, u[j][0]); Si = afma(Si, s, u[j][1])
        # X_k = C - i*S ; X_{p-k} = C + i*S
        X[k]   = (P.add(Cr, Si), P.sub(Ci, Sr))
        X[p-k] = (P.sub(Cr, Si), P.add(Ci, Sr))
    return X

def pfa(P, n1, n2, xs):
    n = n1*n2
    assert math.gcd(n1,n2) == 1
    # input map j = (j1*n2 + j2*n1) % n ; columns DFT_n2 then rows DFT_n1; X[k] = z[k%n1][k%n2]
    grid = [[xs[(j1*n2 + j2*n1) % n] for j2 in range(n2)] for j1 in range(n1)]
    inner = [dft(P, n2, row) for row in grid]       # each row transformed along j2
    cols = []
    for c in range(n2):
        col = [inner[j1][c] for j1 in range(n1)]
        cols.append(dft(P, n1, col))                # z[.][c]
    return [cols[k % n2][k % n1] for k in range(n)]

def ct(P, p, n, xs):
    m = n // p
    assert m*p == n
    Y = [dft(P, m, xs[r::p]) for r in range(p)]
    X = [None]*n
    for a in range(m):
        ts = []
        for r in range(p):
            cr, ci = twid(n, r*a)
            ts.append(cmulc(P, cr, ci, Y[r][a]))
        Z = dft(P, p, ts)
        for b in range(p):
            X[a + m*b] = Z[b]
    return X

def splitradix(P, n, xs):
    if n == 1: return list(xs)
    if n == 2: return base_dft(P, 2, xs)
    if n == 4: return base_dft(P, 4, xs)
    U  = dft(P, n//2, xs[0::2])
    Z  = dft(P, n//4, xs[1::4])
    Z3 = dft(P, n//4, xs[3::4])
    X = [None]*n
    for k in range(n//4):
        c1 = twid(n, k); c3 = twid(n, 3*k)
        wz  = cmulc(P, c1[0], c1[1], Z[k])
        wz3 = cmulc(P, c3[0], c3[1], Z3[k])
        sm = cadd(P, wz, wz3)
        df = csub(P, wz, wz3)
        X[k]        = cadd(P, U[k], sm)
        X[k+n//2]   = csub(P, U[k], sm)
        X[k+n//4]   = cadd(P, U[k+n//4], cmulni(P, df))
        X[k+3*n//4] = cadd(P, U[k+n//4], cmuli(P, df))
    return X

@lru_cache(maxsize=None)
def primitive_root(p):
    fac = []
    x = p-1; d = 2
    while d*d <= x:
        if x % d == 0:
            fac.append(d)
            while x % d == 0: x //= d
        d += 1
    if x > 1: fac.append(x)
    for g in range(2, p):
        if all(pow(g, (p-1)//q, p) != 1 for q in fac):
            return g
    raise ValueError

def rader_tables(p):
    g = primitive_root(p)
    gi = pow(g, p-2, p)
    n = p-1
    # b_t = omega_p^{g^{-t}} in long double
    idx = [pow(gi, t, p) for t in range(n)]
    ang = [LD(-2)*PI*LD(i)/LD(p) for i in idx]
    br = np.array([np.cos(a) for a in ang], dtype=LD)
    bi = np.array([np.sin(a) for a in ang], dtype=LD)
    # B'_q = DFT_n(b)[q] / n  computed in long double
    Br = np.zeros(n, dtype=LD); Bi = np.zeros(n, dtype=LD)
    for q in range(n):
        accr = LD(0); acci = LD(0)
        for t in range(n):
            a = LD(-2)*PI*LD((t*q) % n)/LD(n)
            c = np.cos(a); s = np.sin(a)
            accr += br[t]*c - bi[t]*s
            acci += br[t]*s + bi[t]*c
        Br[q] = accr/LD(n); Bi[q] = acci/LD(n)
    return g, gi, [(float(Br[q]), float(Bi[q])) for q in range(n)]

def rader(P, p, xs):
    n = p-1
    g, gi, Bp = rader_tables(p)
    a = [xs[pow(g, q, p)] for q in range(n)]
    A = dft(P, n, a)
    X = [None]*p
    X[0] = cadd(P, xs[0], A[0])
    V = [None]*n
    for q in range(n):
        cr, ci = Bp[q]
        # clean tiny components (exact zeros polluted by rounding)
        m = max(abs(cr), abs(ci))
        if abs(cr) < 1e-17*m: cr = 0.0
        if abs(ci) < 1e-17*m: ci = 0.0
        V[q] = cmulc(P, cr, ci, A[q])
    V[0] = cadd(P, V[0], xs[0])
    C = dft(P, n, V)
    for mm in range(n):
        X[pow(gi, mm, p)] = C[(n - mm) % n]
    return X

# ---------------- evaluation for testing ----------------
def evaluate(P, outputs, invals):
    """outputs: list of (re,im) idx pairs; invals: dict name->complex numpy array"""
    vals = [None]*len(P.nodes)
    for i,(op,a,b,c) in enumerate(P.nodes):
        if op == 'in':
            nm = a
            vals[i] = invals[nm]
        elif op == 'add': vals[i] = vals[a] + vals[b]
        elif op == 'sub': vals[i] = vals[a] - vals[b]
        elif op == 'neg': vals[i] = -vals[a]
        elif op == 'mul': vals[i] = vals[a] * c
    return [(vals[r], vals[m]) for (r,m) in outputs]

def live_ops(P, outputs):
    seen = set()
    stack = [i for pair in outputs for i in pair]
    while stack:
        i = stack.pop()
        if i in seen: continue
        seen.add(i)
        op,a,b,c = P.nodes[i]
        if op in ('add','sub'): stack += [a,b]
        elif op in ('neg','mul'): stack.append(a)
    counts = {}
    for i in seen:
        op = P.nodes[i][0]
        counts[op] = counts.get(op,0)+1
    return counts

def test_dft(n, verbose=True):
    P = Prog()
    xs = [(P.inp(f"r{j}"), P.inp(f"i{j}")) for j in range(n)]
    X = dft(P, n, xs)
    rng = np.random.default_rng(42)
    xr = rng.standard_normal((n, 13)); xi = rng.standard_normal((n, 13))
    inv = {}
    for j in range(n):
        inv[f"r{j}"] = xr[j]; inv[f"i{j}"] = xi[j]
    got = evaluate(P, X, inv)
    gotc = np.array([g[0] + 1j*g[1] for g in got])
    ref = np.fft.fft((xr + 1j*xi), axis=0)
    err = np.linalg.norm(gotc - ref)/np.linalg.norm(ref)
    cnt = live_ops(P, X)
    nops = sum(v for k,v in cnt.items() if k != 'in')
    if verbose:
        print(f"n={n:3d} relerr={err:.2e} ops={nops} {cnt}")
    assert err < 5e-15, (n, err)
    return nops

if __name__ == '__main__':
    # default plans
    PLAN.update({2:('base',),3:('base',),4:('base',),5:('dsym',),
                 6:('pfa',2,3),8:('ct',2),9:('ct',3),10:('pfa',2,5),
                 11:('rader',),12:('pfa',4,3),13:('rader',),16:('ct',4),
                 17:('rader',),22:('pfa',2,11),23:('rader',),
                 36:('pfa',4,9),45:('pfa',9,5),64:('ct',8)})
    for n in (2,3,4,5,6,8,9,10,11,12,13,16,17,22,23,36,45,64):
        test_dft(n)
