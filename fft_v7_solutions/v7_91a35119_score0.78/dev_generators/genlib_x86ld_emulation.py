import numpy as np
from math import gcd
from mpmath import mp as _mp, mpf as _mpf, cos as _c64, sin as _s64
_mp.prec = 64  # emulate x86 80-bit extended (64-bit mantissa) longdouble

_PI64 = _mpf('3.14159265358979323846264338327950288419716939937510582097494459')

def _tw64(n, kj):
    ang = (-2*_PI64)*_mpf(kj)/_mpf(n)
    return _c64(ang), _s64(ang)

def tw(n, kj):
    """exact e^{-2pi i kj/n} rounded to double, with exact snapping of quarter turns"""
    kj = kj % n
    if kj == 0: return (1.0, 0.0)
    if 4*kj == n: return (0.0, -1.0)
    if 2*kj == n: return (-1.0, 0.0)
    if 4*kj == 3*n: return (0.0, 1.0)
    c, s = _tw64(n, kj)
    return (float(c), float(s))

class G:
    def __init__(self):
        self.nodes = []
        self.memo = {}
    def _mk(self, t):
        if t in self.memo: return self.memo[t]
        i = len(self.nodes); self.nodes.append(t); self.memo[t] = i
        return i
    def inp(self, name): return self._mk(('in', name))
    def add(self, a, b):
        na, nb = self.nodes[a], self.nodes[b]
        if nb[0] == 'neg': return self.sub(a, nb[1])
        if na[0] == 'neg': return self.sub(b, na[1])
        if b < a: a, b = b, a
        return self._mk(('add', a, b))
    def sub(self, a, b):
        if a == b:
            # zero: shouldn't normally happen in DFT; represent as mul(0)? keep explicit
            return self._mk(('zero',))
        nb = self.nodes[b]
        if nb[0] == 'neg': return self.add(a, nb[1])
        na = self.nodes[a]
        if na[0] == 'neg': return self.neg(self.add(na[1], b))
        return self._mk(('sub', a, b))
    def neg(self, a):
        na = self.nodes[a]
        if na[0] == 'neg': return na[1]
        if na[0] == 'sub': return self._mk(('sub', na[2], na[1]))
        if na[0] == 'mul': return self.mul(-na[1], na[2])
        return self._mk(('neg', a))
    def mul(self, c, a):
        c = float(c)
        assert c == c and abs(c) != float('inf')
        if c == 1.0: return a
        if c == -1.0: return self.neg(a)
        na = self.nodes[a]
        if na[0] == 'neg': return self.mul(-c, na[1])
        if na[0] == 'mul': return self.mul(c * na[1], na[2])
        return self._mk(('mul', c, a))

def cadd(g,a,b): return (g.add(a[0],b[0]), g.add(a[1],b[1]))
def csub(g,a,b): return (g.sub(a[0],b[0]), g.sub(a[1],b[1]))
def cmulc(g,a,c):
    cr, ci = c
    ar, ai = a
    if ci == 0.0: return (g.mul(cr,ar), g.mul(cr,ai))
    if cr == 0.0: return (g.mul(-ci,ai), g.mul(ci,ar))
    return (g.sub(g.mul(cr,ar), g.mul(ci,ai)), g.add(g.mul(cr,ai), g.mul(ci,ar)))

def factor(n):
    f = {}
    d = 2
    while d*d <= n:
        while n % d == 0: f[d] = f.get(d,0)+1; n //= d
        d += 1
    if n > 1: f[n] = f.get(n,0)+1
    return f

def dft(g, n, x):
    assert len(x) == n
    if n == 1: return list(x)
    if n == 2:
        return [cadd(g,x[0],x[1]), csub(g,x[0],x[1])]
    if n == 3:
        t = cadd(g,x[1],x[2]); d = csub(g,x[1],x[2])
        X0 = cadd(g,x[0],t)
        mr = g.add(x[0][0], g.mul(-0.5, t[0])); mi = g.add(x[0][1], g.mul(-0.5, t[1]))
        s = tw(3,1)[1]  # sin(-2pi/3) = -sin60
        # X1 = m + i*s*d ; (i*(dr+idi)) = -di + i dr -> X1 = (mr - s*di, mi + s*dr) with s negative
        p = g.mul(s, d[1]); q = g.mul(s, d[0])
        X1 = (g.sub(mr, p), g.add(mi, q))
        X2 = (g.add(mr, p), g.sub(mi, q))
        return [X0, X1, X2]
    if n == 4:
        t0 = cadd(g,x[0],x[2]); t1 = csub(g,x[0],x[2])
        t2 = cadd(g,x[1],x[3]); t3 = csub(g,x[1],x[3])
        # X1 = t1 - i t3 ; X3 = t1 + i t3
        return [cadd(g,t0,t2),
                (g.add(t1[0], t3[1]), g.sub(t1[1], t3[0])),
                csub(g,t0,t2),
                (g.sub(t1[0], t3[1]), g.add(t1[1], t3[0]))]
    if n == 5:
        t1 = cadd(g,x[1],x[4]); t3 = csub(g,x[1],x[4])
        t2 = cadd(g,x[2],x[3]); t4 = csub(g,x[2],x[3])
        t5 = cadd(g,t1,t2)
        X0 = cadd(g,x[0],t5)
        br = g.add(x[0][0], g.mul(-0.25, t5[0])); bi = g.add(x[0][1], g.mul(-0.25, t5[1]))
        c72, s72n = tw(5,1); c144, s144n = tw(5,2)
        K2 = (c72 - c144)/2.0
        dr = g.sub(t1[0], t2[0]); di = g.sub(t1[1], t2[1])
        u1 = (g.add(br, g.mul(K2,dr)), g.add(bi, g.mul(K2,di)))
        u2 = (g.sub(br, g.mul(K2,dr)), g.sub(bi, g.mul(K2,di)))
        # W1 = s72n*t3 + s144n*t4 (pure-imag coefficient combos): X1 = u1 + i*W1? derive:
        # X1 = sum x_j w^j = u1 + i*(s72n*t3 + s144n*t4)  with w^j = c + i*s_n (s_n negative sins)
        w1r = g.add(g.mul(s72n, t3[0]), g.mul(s144n, t4[0]))
        w1i = g.add(g.mul(s72n, t3[1]), g.mul(s144n, t4[1]))
        w2r = g.sub(g.mul(s144n, t3[0]), g.mul(s72n, t4[0]))
        w2i = g.sub(g.mul(s144n, t3[1]), g.mul(s72n, t4[1]))
        X1 = (g.sub(u1[0], w1i), g.add(u1[1], w1r))
        X4 = (g.add(u1[0], w1i), g.sub(u1[1], w1r))
        X2 = (g.sub(u2[0], w2i), g.add(u2[1], w2r))
        X3 = (g.add(u2[0], w2i), g.sub(u2[1], w2r))
        return [X0, X1, X2, X3, X4]
    f = factor(n)
    if len(f) > 1:
        p = sorted(f)[0]
        n1 = p**f[p]; n2 = n//n1
        return pfa(g, n, n1, n2, x)
    p = sorted(f)[0]
    if p == 2:
        r = 8 if n % 8 == 0 and n > 8 else 2
        return ct(g, n, r, x)
    if n == p:
        return symodd(g, n, x)
    return ct(g, n, p, x)

def ct(g, n, r, x):
    m = n // r
    ys = [dft(g, m, [x[r*t+s] for t in range(m)]) for s in range(r)]
    X = [None]*n
    for mi in range(m):
        zs = [cmulc(g, ys[s][mi], tw(n, s*mi)) for s in range(r)]
        out = dft(g, r, zs)
        for a in range(r):
            X[mi + m*a] = out[a]
    return X

def pfa(g, n, n1, n2, x):
    grid = [[x[(a*n2 + b*n1) % n] for b in range(n2)] for a in range(n1)]
    cols = [dft(g, n1, [grid[a][b] for a in range(n1)]) for b in range(n2)]  # cols[b][k1]
    rows = [dft(g, n2, [cols[b][k1] for b in range(n2)]) for k1 in range(n1)]  # rows[k1][k2]
    return [rows[k % n1][k % n2] for k in range(n)]

def rsum(g, terms):
    # balanced tree sum of term node-ids (keeps FMA-fusable mul+add pairs adjacent)
    t = list(terms)
    while len(t) > 1:
        nt = []
        for i in range(0, len(t)-1, 2):
            nt.append(g.add(t[i], t[i+1]))
        if len(t) % 2: nt.append(t[-1])
        t = nt
    return t[0]

def symodd(g, n, x):
    h = (n-1)//2
    e = [None]+[cadd(g, x[j], x[n-j]) for j in range(1,h+1)]
    o = [None]+[csub(g, x[j], x[n-j]) for j in range(1,h+1)]
    X = [None]*n
    X[0] = (rsum(g, [x[0][0]] + [e[j][0] for j in range(1,h+1)]),
            rsum(g, [x[0][1]] + [e[j][1] for j in range(1,h+1)]))
    for k in range(1,h+1):
        cs = [tw(n, k*j) for j in range(1,h+1)]
        Pr = rsum(g, [x[0][0]] + [g.mul(cs[j-1][0], e[j][0]) for j in range(1,h+1)])
        Pi = rsum(g, [x[0][1]] + [g.mul(cs[j-1][0], e[j][1]) for j in range(1,h+1)])
        Qr = rsum(g, [g.mul(-cs[j-1][1], o[j][0]) for j in range(1,h+1)])
        Qi = rsum(g, [g.mul(-cs[j-1][1], o[j][1]) for j in range(1,h+1)])
        X[k]   = (g.add(Pr, Qi), g.sub(Pi, Qr))
        X[n-k] = (g.sub(Pr, Qi), g.add(Pi, Qr))
    return X

# ---------------- evaluator for verification ----------------
def eval_outputs(g, outs, invals):
    """outs: list of (re_id, im_id); invals: dict name->float; returns complex array"""
    vals = {}
    import sys
    sys.setrecursionlimit(100000)
    def ev(i):
        if i in vals: return vals[i]
        t = g.nodes[i]
        k = t[0]
        if k == 'in': v = invals[t[1]]
        elif k == 'add': v = ev(t[1]) + ev(t[2])
        elif k == 'sub': v = ev(t[1]) - ev(t[2])
        elif k == 'neg': v = -ev(t[1])
        elif k == 'mul': v = t[1] * ev(t[2])
        elif k == 'zero': v = 0.0
        else: raise Exception(k)
        vals[i] = v
        return v
    return np.array([complex(ev(r), ev(im)) for (r, im) in outs])

def verify(n, trials=3):
    rng = np.random.default_rng(42+n)
    for _ in range(trials):
        g = G()
        xin = [(g.inp(('x',j,0)), g.inp(('x',j,1))) for j in range(n)]
        outs = dft(g, n, xin)
        xv = rng.standard_normal(n) + 1j*rng.standard_normal(n)
        invals = {}
        for j in range(n):
            invals[('x',j,0)] = xv[j].real; invals[('x',j,1)] = xv[j].imag
        got = eval_outputs(g, outs, invals)
        want = np.fft.fft(xv)
        err = np.linalg.norm(got-want)/np.linalg.norm(want)
        assert err < 1e-13, (n, err)
    return count_ops(g, outs)

def count_ops(g, outs):
    seen = set()
    cnt = {'add':0,'sub':0,'mul':0,'neg':0}
    def walk(i):
        if i in seen: return
        seen.add(i)
        t = g.nodes[i]
        if t[0] in ('add','sub'):
            walk(t[1]); walk(t[2]); cnt[t[0]] += 1
        elif t[0] == 'mul':
            walk(t[2]); cnt['mul'] += 1
        elif t[0] == 'neg':
            walk(t[1]); cnt['neg'] += 1
    for (r,i) in outs:
        walk(r); walk(i)
    return cnt

if __name__ == '__main__':
    for n in (2,3,4,5,6,8,9,13,16,17,23,36,45,64):
        c = verify(n)
        tot = c['add']+c['sub']+c['mul']+c['neg']
        print(f"n={n:3d}: ops={tot:5d}  (add {c['add']}, sub {c['sub']}, mul {c['mul']}, neg {c['neg']})")


def find_gen(p):
    for gg in range(2, p):
        s = set(); x = 1
        for _ in range(p-1):
            x = x*gg % p; s.add(x)
        if len(s) == p-1: return gg

def rader_kernel(p):
    """returns complex kernel K_t = DFT(w)_t / n as python complex (double), computed in longdouble"""
    gg = find_gen(p)
    n = p-1
    # w_s = W^{g^{(n-s)%n}}
    wr = [_mpf(0)]*n; wi = [_mpf(0)]*n
    for s in range(n):
        c, sn = tw_ld(p, pow(gg, (n-s) % n, p))
        wr[s] = c; wi[s] = sn
    Kr = []; Ki = []
    for t in range(n):
        ar = _mpf(0); ai = _mpf(0)
        for s in range(n):
            c, sn = tw_ld(n, s*t)
            ar += wr[s]*c - wi[s]*sn
            ai += wr[s]*sn + wi[s]*c
        Kr.append(float(ar/n)); Ki.append(float(ai/n))
    return gg, list(zip(Kr, Ki))

def tw_ld(n, kj):
    kj = kj % n
    if kj == 0: return (_mpf(1), _mpf(0))
    if 4*kj == n: return (_mpf(0), _mpf(-1))
    if 2*kj == n: return (_mpf(-1), _mpf(0))
    if 4*kj == 3*n: return (_mpf(0), _mpf(1))
    return _tw64(n, kj)

def rader(g, p, x):
    gg, K = rader_kernel(p)
    n = p-1
    perm = [pow(gg, b, p) for b in range(n)]
    u = [x[j] for j in perm]
    U = dft(g, n, u)
    # X0 = x0 + U[0]
    X = [None]*p
    X[0] = cadd(g, x[0], U[0])
    # y_t = U_t*K_t ; y_0 += x0
    y = [cmulc(g, U[t], K[t]) for t in range(n)]
    y[0] = cadd(g, y[0], x[0])
    # conv = conj(dft(conj(y)))
    yc = [(re, g.neg(im)) for (re, im) in y]
    C = dft(g, n, yc)
    conv = [(re, g.neg(im)) for (re, im) in C]
    ginv = pow(gg, p-2, p)
    for a in range(n):
        k = pow(ginv, a, p)
        X[k] = conv[a]
    return X
