import numpy as np
from math import gcd

LD = np.longdouble
PI_LD = LD('3.14159265358979323846264338327950288')

class Builder:
    def __init__(self):
        self.nodes = []      # list of tuples
        self.memo = {}
        self.ZERO = self._new(('const', 0.0))
    def _new(self, t):
        if t in self.memo: return self.memo[t]
        i = len(self.nodes); self.nodes.append(t); self.memo[t] = i
        return i
    def var(self, name): return self._new(('var', name))
    def const(self, c):
        c = float(c)
        if c == 0.0: c = 0.0   # normalize -0.0
        return self._new(('const', c))
    def kind(self, i): return self.nodes[i][0]
    def cval(self, i): return self.nodes[i][1]
    def is_const(self, i): return self.nodes[i][0]=='const'
    def neg(self, a):
        n = self.nodes[a]
        if n[0]=='const': return self.const(-n[1])
        if n[0]=='neg': return n[1]
        return self._new(('neg', a))
    def add(self, a, b):
        na, nb = self.nodes[a], self.nodes[b]
        if na[0]=='const' and nb[0]=='const': return self.const(na[1]+nb[1])
        if na[0]=='const' and na[1]==0.0: return b
        if nb[0]=='const' and nb[1]==0.0: return a
        if nb[0]=='neg': return self.sub(a, nb[1])
        if na[0]=='neg': return self.sub(b, na[1])
        if a > b: a, b = b, a
        return self._new(('add', a, b))
    def sub(self, a, b):
        na, nb = self.nodes[a], self.nodes[b]
        if na[0]=='const' and nb[0]=='const': return self.const(na[1]-nb[1])
        if nb[0]=='const' and nb[1]==0.0: return a
        if na[0]=='const' and na[1]==0.0: return self.neg(b)
        if nb[0]=='neg': return self.add(a, nb[1])
        if na[0]=='neg': return self.neg(self.add(na[1], b))
        if a == b: return self.const(0.0)
        return self._new(('sub', a, b))
    def mul(self, a, b):
        na, nb = self.nodes[a], self.nodes[b]
        if na[0]=='const' and nb[0]=='const': return self.const(na[1]*nb[1])
        if na[0]=='const': a_, b_ = a, b
        elif nb[0]=='const': a_, b_ = b, a; na = nb
        else:
            if nb[0]=='neg' and na[0]=='neg': return self.mul(self.nodes[a][1], self.nodes[b][1])
            if nb[0]=='neg': return self.neg(self.mul(a, nb[1]))
            if na[0]=='neg': return self.neg(self.mul(na[1], b))
            if a > b: a, b = b, a
            return self._new(('mul', a, b))
        c = na[1]
        if c == 0.0: return self.const(0.0)
        if c == 1.0: return b_
        if c == -1.0: return self.neg(b_)
        nb2 = self.nodes[b_]
        if nb2[0]=='neg': return self.neg(self.mul(self.const(-0.0+ -1*0 + c*-1), nb2[1])) if False else self.neg(self.mul(self.const(c), nb2[1])) if False else self._mulcn(c, nb2[1])
        return self._new(('mul', a_, b_))
    def _mulcn(self, c, x):  # c * neg-stripped x -> neg(c*x) handled via const sign flip
        return self.mul(self.const(-c), x)

# complex helpers
def cadd(B, z, w): return (B.add(z[0], w[0]), B.add(z[1], w[1]))
def csub(B, z, w): return (B.sub(z[0], w[0]), B.sub(z[1], w[1]))
def cneg(B, z): return (B.neg(z[0]), B.neg(z[1]))

def cmulw(B, z, wr, wi):
    """multiply complex z by constant (wr + i wi)"""
    a, b = z
    if wi == 0.0:
        return (B.mul(B.const(wr), a), B.mul(B.const(wr), b))
    if wr == 0.0:
        # (a+ib)(i wi) = -wi b + i wi a
        return (B.neg(B.mul(B.const(wi), b)) if wi > 0 else B.mul(B.const(-wi), b),
                B.mul(B.const(wi), a))
    if wr == wi:
        # wr (a - b) ... (a+ib)(w + iw) = w(a-b) + i w(a+b)
        return (B.mul(B.const(wr), B.sub(a, b)), B.mul(B.const(wr), B.add(a, b)))
    if wr == -wi:
        # (a+ib)(w - iw) = w(a+b) + i w(b-a)
        return (B.mul(B.const(wr), B.add(a, b)), B.mul(B.const(wr), B.sub(b, a)))
    return (B.sub(B.mul(B.const(wr), a), B.mul(B.const(wi), b)),
            B.add(B.mul(B.const(wr), b), B.mul(B.const(wi), a)))

_SQ2H = float(np.sqrt(LD(2))/LD(2))
def tw(N, t):
    """W_N^t = exp(-2 pi i t / N) as pair of doubles, computed in long double with exact mod"""
    t = t % N
    if (8*t) % N == 0:
        k = (8*t)//N   # angle = -k*pi/4
        tab = {0:(1.0,0.0), 1:(_SQ2H,-_SQ2H), 2:(0.0,-1.0), 3:(-_SQ2H,-_SQ2H),
               4:(-1.0,0.0), 5:(-_SQ2H,_SQ2H), 6:(0.0,1.0), 7:(_SQ2H,_SQ2H)}
        return tab[k % 8]
    ang = (LD(-2) * PI_LD) * LD(t) / LD(N)
    return float(np.cos(ang)), float(np.sin(ang))

def dft(B, N, xs):
    assert len(xs) == N
    if N == 1: return list(xs)
    if N == 2:
        return [cadd(B, xs[0], xs[1]), csub(B, xs[0], xs[1])]
    if N == 4:
        t0 = cadd(B, xs[0], xs[2]); t1 = csub(B, xs[0], xs[2])
        t2 = cadd(B, xs[1], xs[3]); t3 = csub(B, xs[1], xs[3])
        # X1 = t1 - i t3 ; X3 = t1 + i t3 ; (-i)(a+ib) = b - ia
        mi = (t3[1], B.neg(t3[0]))   # -i * t3
        return [cadd(B, t0, t2), cadd(B, t1, mi), csub(B, t0, t2), csub(B, t1, mi)]
    # coprime factorization (PFA)
    for N1 in (9, 8, 5, 4, 3, 2):
        if N % N1 == 0:
            N2 = N // N1
            if N2 > 1 and gcd(N1, N2) == 1:
                return pfa(B, N1, N2, xs)
    if is_prime(N):
        return prime_sym(B, N, xs)
    # prime power: Cooley-Tukey
    for r in (8, 4, 9, 2, 3, 5):
        if N % r == 0 and N // r > 1:
            return ct(B, r, N // r, xs)
    raise ValueError(N)

def is_prime(n):
    return n > 1 and all(n % p for p in range(2, int(n**0.5)+1))

def ct(B, r, m, xs):
    """N = r*m, decimation in time: j = r*a + b"""
    N = r * m
    subs = [dft(B, m, [xs[r*a + b] for a in range(m)]) for b in range(r)]
    out = [None]*N
    for d in range(m):
        t = []
        for b in range(r):
            wr, wi = tw(N, b*d)
            t.append(cmulw(B, subs[b][d], wr, wi))
        y = dft(B, r, t)
        for c in range(r):
            out[d + m*c] = y[c]
    return out

def pfa(B, N1, N2, xs):
    N = N1*N2
    # input map j = (N2*j1 + N1*j2) % N
    cols = []
    for j2 in range(N2):
        col = [xs[(N2*j1 + N1*j2) % N] for j1 in range(N1)]
        cols.append(dft(B, N1, col))   # cols[j2][k1]
    out = [None]*N
    rows = [dft(B, N2, [cols[j2][k1] for j2 in range(N2)]) for k1 in range(N1)]
    # output k: k % N1 == k1, k % N2 == k2
    for k in range(N):
        out[k] = rows[k % N1][k % N2]
    return out

def prime_sym(B, N, xs):
    h = (N-1)//2
    x0 = xs[0]
    ps = {}; qs = {}
    for j in range(1, h+1):
        ps[j] = cadd(B, xs[j], xs[N-j])
        qs[j] = csub(B, xs[j], xs[N-j])
    out = [None]*N
    # X0
    sr, si = x0
    for j in range(1, h+1):
        sr = B.add(sr, ps[j][0]); si = B.add(si, ps[j][1])
    out[0] = (sr, si)
    for k in range(1, h+1):
        ar, ai = x0
        br, bi = B.ZERO, B.ZERO
        first = True
        for j in range(1, h+1):
            t = (k*j) % N
            ang = (LD(2)*PI_LD)*LD(t)/LD(N)
            c = float(np.cos(ang)); s = float(np.sin(ang))
            ar = B.add(ar, B.mul(B.const(c), ps[j][0]))
            ai = B.add(ai, B.mul(B.const(c), ps[j][1]))
            if first:
                br = B.mul(B.const(s), qs[j][0]); bi = B.mul(B.const(s), qs[j][1]); first = False
            else:
                br = B.add(br, B.mul(B.const(s), qs[j][0]))
                bi = B.add(bi, B.mul(B.const(s), qs[j][1]))
        # X[k] = (ar + bi) + i (ai - br) ;  X[N-k] = (ar - bi) + i (ai + br)
        out[k]   = (B.add(ar, bi), B.sub(ai, br))
        out[N-k] = (B.sub(ar, bi), B.add(ai, br))
    return out

def eval_dag(B, outs, env):
    """evaluate with numpy complex env: env maps var name -> float"""
    vals = {}
    def ev(i):
        if i in vals: return vals[i]
        n = B.nodes[i]
        if n[0]=='var': v = env[n[1]]
        elif n[0]=='const': v = n[1]
        elif n[0]=='neg': v = -ev(n[1])
        elif n[0]=='add': v = ev(n[1]) + ev(n[2])
        elif n[0]=='sub': v = ev(n[1]) - ev(n[2])
        elif n[0]=='mul': v = ev(n[1]) * ev(n[2])
        vals[i] = v
        return v
    import sys
    sys.setrecursionlimit(100000)
    return [ (ev(r), ev(im)) for (r, im) in outs ]

def count_ops(B, outs):
    seen = set(); cnt = {'add':0,'sub':0,'mul':0,'neg':0}
    def walk(i):
        if i in seen: return
        seen.add(i)
        n = B.nodes[i]
        if n[0] in cnt: cnt[n[0]] += 1
        for a in n[1:]:
            if isinstance(a, int) and n[0] != 'var' and n[0] != 'const': walk(a)
    import sys
    sys.setrecursionlimit(100000)
    for r, im in outs: walk(r); walk(im)
    return cnt
