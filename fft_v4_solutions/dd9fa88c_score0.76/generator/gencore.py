"""Expression-DAG mini-genfft: builds straight-line DFT codelets with CSE+simplification."""
import math
from fractions import Fraction
import mpmath as mp
mp.mp.dps = 60

# ---------------- high precision trig constants ----------------
_trig_cache = {}
def wcs(num, den):
    """(cos, sin) of -2*pi*num/den, correctly rounded doubles, exact zeros/ones snapped."""
    num = num % den
    key = Fraction(num, den)
    if key in _trig_cache: return _trig_cache[key]
    ang = -2 * mp.pi * mp.mpf(key.numerator) / mp.mpf(key.denominator)
    c, s = mp.cos(ang), mp.sin(ang)
    def snap(v):
        fv = float(v)
        if abs(v) < mp.mpf('1e-40'): return 0.0
        if abs(v - 1) < mp.mpf('1e-40'): return 1.0
        if abs(v + 1) < mp.mpf('1e-40'): return -1.0
        return fv
    r = (snap(c), snap(s))
    _trig_cache[key] = r
    return r

# ---------------- DAG ----------------
class G:
    def __init__(self):
        self.ops = []      # list of tuples
        self.memo = {}
        self.ZERO = self._raw(('k', 0.0))
    def _raw(self, t):
        key = self._key(t)
        if key in self.memo: return self.memo[key]
        self.ops.append(t)
        i = len(self.ops) - 1
        self.memo[key] = i
        return i
    def _key(self, t):
        if t[0] == '+':
            a, b = sorted(t[1:])
            return ('+', a, b)
        if t[0] == 'm2':
            a, b = sorted(t[1:])
            return ('m2', a, b)
        if t[0] == 'k':
            return ('k', t[1].hex())
        if t[0] == '*':
            return ('*', t[1].hex(), t[2])
        return t
    def op(self, i): return self.ops[i]
    def isk(self, i): return self.ops[i][0] == 'k'
    def kval(self, i): return self.ops[i][1]

    def inp(self, tag): return self._raw(('in', tag))
    def k(self, v):
        return self._raw(('k', float(v)))
    def add(self, a, b):
        if a == self.ZERO: return b
        if b == self.ZERO: return a
        oa, ob = self.ops[a], self.ops[b]
        if oa[0] == 'k' and ob[0] == 'k': return self.k(oa[1] + ob[1])
        if oa[0] == 'n' and ob[0] == 'n': return self.neg(self.add(oa[1], ob[1]))
        if ob[0] == 'n': return self.sub(a, ob[1])
        if oa[0] == 'n': return self.sub(b, oa[1])
        return self._raw(('+', a, b))
    def sub(self, a, b):
        if a == b: return self.ZERO
        if b == self.ZERO: return a
        if a == self.ZERO: return self.neg(b)
        oa, ob = self.ops[a], self.ops[b]
        if oa[0] == 'k' and ob[0] == 'k': return self.k(oa[1] - ob[1])
        if ob[0] == 'n': return self.add(a, ob[1])
        if oa[0] == 'n': return self.neg(self.add(oa[1], b))
        return self._raw(('-', a, b))
    def neg(self, a):
        oa = self.ops[a]
        if oa[0] == 'k': return self.k(-oa[1])
        if oa[0] == 'n': return oa[1]
        return self._raw(('n', a))
    def mul(self, kv, a):
        kv = float(kv)
        if kv == 0.0: return self.ZERO
        oa = self.ops[a]
        if oa[0] == 'k': return self.k(kv * oa[1])
        if kv == 1.0: return a
        if kv == -1.0: return self.neg(a)
        if oa[0] == 'n': return self.neg(self.mul(-(-kv), oa[1])) if False else self.neg(self.mul(kv, oa[1]))
        if oa[0] == '*': return self.mul(kv * oa[1], oa[2])
        if kv < 0: return self.neg(self.mul(-kv, a))
        return self._raw(('*', kv, a))
    def mul2(self, a, b):
        oa, ob = self.ops[a], self.ops[b]
        if oa[0] == 'k': return self.mul(oa[1], b)
        if ob[0] == 'k': return self.mul(ob[1], a)
        return self._raw(('m2', a, b))
    def rsqrt(self, a): return self._raw(('rsqrt', a))
    def rcp(self, a): return self._raw(('rcp', a))

    # ---- complex helpers: z = (re_id, im_id) ----
    def cadd(self, z, w): return (self.add(z[0], w[0]), self.add(z[1], w[1]))
    def csub(self, z, w): return (self.sub(z[0], w[0]), self.sub(z[1], w[1]))
    def cneg(self, z): return (self.neg(z[0]), self.neg(z[1]))
    def cmulk(self, c, s, z):
        """(c + i*s) * z with c,s float constants."""
        zr, zi = z
        if s == 0.0:
            return (self.mul(c, zr), self.mul(c, zi))
        if c == 0.0:
            return (self.neg(self.mul(s, zi)) if s > 0 else self.mul(-s, zi),
                    self.mul(s, zr))
        if c == s:
            # c*(zr - zi) + i*c*(zr + zi)
            return (self.mul(c, self.sub(zr, zi)), self.mul(c, self.add(zr, zi)))
        if c == -s:
            # c*(zr + zi) + i*c*(zi - zr)
            return (self.mul(c, self.add(zr, zi)), self.mul(c, self.sub(zi, zr)))
        return (self.sub(self.mul(c, zr), self.mul(s, zi)),
                self.add(self.mul(c, zi), self.mul(s, zr)))
    def csum_tree(self, terms):
        """balanced-tree sum of complex terms"""
        assert terms
        cur = list(terms)
        while len(cur) > 1:
            nxt = []
            for i in range(0, len(cur) - 1, 2):
                nxt.append(self.cadd(cur[i], cur[i + 1]))
            if len(cur) % 2: nxt.append(cur[-1])
            cur = nxt
        return cur[0]
    def sum_chain(self, terms):
        cur = terms[0]
        for t in terms[1:]:
            cur = self.add(cur, t)
        return cur
    def sum_chain2(self, terms):
        h = (len(terms) + 1) // 2
        if len(terms) < 4: return self.sum_chain(terms)
        return self.add(self.sum_chain(terms[:h]), self.sum_chain(terms[h:]))
    def sum_tree(self, terms):
        cur = list(terms)
        while len(cur) > 1:
            nxt = []
            for i in range(0, len(cur) - 1, 2):
                nxt.append(self.add(cur[i], cur[i + 1]))
            if len(cur) % 2: nxt.append(cur[-1])
            cur = nxt
        return cur[0]

# ---------------- numeric evaluation for verification ----------------
import numpy as np
def eval_nodes(g, inputs, want):
    """inputs: dict tag->np array; want: list of node ids; returns list of arrays"""
    vals = [None] * len(g.ops)
    # evaluate everything reachable (simple: evaluate all in order)
    for i, t in enumerate(g.ops):
        o = t[0]
        if o == 'k': vals[i] = t[1]
        elif o == 'in': vals[i] = inputs[t[1]]
        elif o == '+': vals[i] = vals[t[1]] + vals[t[2]]
        elif o == '-': vals[i] = vals[t[1]] - vals[t[2]]
        elif o == 'n': vals[i] = -vals[t[1]]
        elif o == '*': vals[i] = t[1] * vals[t[2]]
        elif o == 'm2': vals[i] = vals[t[1]] * vals[t[2]]
        elif o == 'rsqrt': vals[i] = 1.0 / np.sqrt(vals[t[1]])
        elif o == 'rcp': vals[i] = 1.0 / vals[t[1]]
        else: raise ValueError(o)
    return [vals[w] for w in want]
