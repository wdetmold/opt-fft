"""DFT strategy builders over the expression DAG + automatic plan selection."""
import math
from math import gcd
from fractions import Fraction
import mpmath as mp
from gencore import G, wcs

def is_prime(n):
    if n < 2: return False
    for p in range(2, int(n**0.5) + 1):
        if n % p == 0: return False
    return True

def primroot(p):
    for g_ in range(2, p):
        seen, x = set(), 1
        ok = True
        for _ in range(p - 1):
            x = x * g_ % p
            if x in seen: ok = False; break
            seen.add(x)
        if ok and len(seen) == p - 1: return g_
    raise ValueError

# ---------- strategies; each: build(g, xs) -> ys (list of complex node pairs) ----------

def b_dft2(g, xs):
    a, b = xs
    return [g.cadd(a, b), g.csub(a, b)]

def b_direct_sym(n):
    """odd n: symmetric direct DFT"""
    assert n % 2 == 1
    h = (n - 1) // 2
    def build(g, xs):
        sp = [None] * (h + 1); sm = [None] * (h + 1)
        for j in range(1, h + 1):
            sp[j] = g.cadd(xs[j], xs[n - j])
            sm[j] = g.csub(xs[j], xs[n - j])
        ys = [None] * n
        ys[0] = g.csum_tree([xs[0]] + [sp[j] for j in range(1, h + 1)])
        for k in range(1, h + 1):
            # A = sum_j cos(2pi jk/n) sp_j ; B = sum_j sin(2pi jk/n) sm_j
            at_r, at_i, bt_r, bt_i = [], [], [], []
            for j in range(1, h + 1):
                c, s = wcs(j * k, n)   # cos(-2pi jk/n), sin(-2pi jk/n)
                # careful: wcs returns cos & sin of NEGATIVE angle: cos same, sin negated
                cc, ss = c, -s
                at_r.append(g.mul(cc, sp[j][0])); at_i.append(g.mul(cc, sp[j][1]))
                bt_r.append(g.mul(ss, sm[j][0])); bt_i.append(g.mul(ss, sm[j][1]))
            Ar = g.sum_tree(at_r); Ai = g.sum_tree(at_i)
            Br = g.sum_tree(bt_r); Bi = g.sum_tree(bt_i)
            x0r, x0i = xs[0]
            ys[k]     = (g.add(g.add(x0r, Ar), Bi), g.sub(g.add(x0i, Ai), Br))
            ys[n - k] = (g.sub(g.add(x0r, Ar), Bi), g.add(g.add(x0i, Ai), Br))
        return ys
    return build

def b_pfa(n1, n2, plan1, plan2):
    """Good-Thomas for coprime n1,n2"""
    n = n1 * n2
    inv1 = pow(n1, -1, n2)
    inv2 = pow(n2, -1, n1)
    def build(g, xs):
        # rows: for each j2, dft_n1 over j1 of x[(n2*j1 + n1*j2) % n]
        t = [[None] * n2 for _ in range(n1)]
        for j2 in range(n2):
            sub = [xs[(n2 * j1 + n1 * j2) % n] for j1 in range(n1)]
            row = plan1(g, sub)
            for k1 in range(n1):
                t[k1][j2] = row[k1]
        ys = [None] * n
        for k1 in range(n1):
            col = plan2(g, t[k1])
            for k2 in range(n2):
                k = (k1 * n2 * inv2 + k2 * n1 * inv1) % n
                ys[k] = col[k2]
        return ys
    return build

def b_ct(n1, n2, plan1, plan2):
    """Cooley-Tukey: n=n1*n2; inner dfts length n2 on stride-n1 subseqs, twiddle, outer n1."""
    n = n1 * n2
    def build(g, xs):
        inner = []
        for j1 in range(n1):
            sub = [xs[n1 * j2 + j1] for j2 in range(n2)]
            inner.append(plan2(g, sub))
        ys = [None] * n
        for k2 in range(n2):
            col = []
            for j1 in range(n1):
                c, s = wcs(j1 * k2, n)
                col.append(g.cmulk(c, s, inner[j1][k2]))
            out = plan1(g, col)
            for k1 in range(n1):
                ys[n2 * k1 + k2] = out[k1]
        return ys
    return build

def b_rader(p, planq):
    """Rader for odd prime p: q=p-1 convolution via two dft_q."""
    q = p - 1
    gr = primroot(p)
    # permutations
    gpow = [pow(gr, l, p) for l in range(q)]            # g^l
    ginv = [pow(gr, (q - l) % q, p) for l in range(q)]  # g^{-l}
    # B'_t = (1/q) * sum_l w_p^{g^{-l}} * w_q^{-t l}   (high precision)
    Bc = []
    for t in range(q):
        acc = mp.mpc(0)
        for l in range(q):
            a1 = -2 * mp.pi * mp.mpf(ginv[l]) / p
            a2 = -2 * mp.pi * mp.mpf((t * l) % q) / q
            acc += (mp.cos(a1) + 1j * mp.sin(a1)) * (mp.cos(a2) + 1j * mp.sin(a2))
        acc /= q
        Bc.append((float(mp.re(acc)), float(mp.im(acc))))
    def build(g, xs):
        a = [xs[gpow[l]] for l in range(q)]
        A = planq(g, a)
        x0 = xs[0]
        # X[0] = x0 + A[0]
        P = [g.cmulk(Bc[t][0], Bc[t][1], A[t]) for t in range(q)]
        D = planq(g, P)
        ys = [None] * p
        ys[0] = g.cadd(x0, A[0])
        for m in range(q):
            cm = D[(q - m) % q]
            ys[ginv[m]] = g.cadd(x0, cm)
        return ys
    return build

# ---------------- plan selection ----------------
_best = {}

def factor_pairs(n):
    out = []
    for a in range(2, int(n**0.5) + 1):
        if n % a == 0:
            out.append((a, n // a))
    return out

def count_ops(build, n):
    g = G()
    xs = [(g.inp(f"r{j}"), g.inp(f"i{j}")) for j in range(n)]
    ys = build(g, xs)
    # count real ops (reachable from outputs)
    reach = set()
    stack = [c for y in ys for c in y]
    while stack:
        i = stack.pop()
        if i in reach: continue
        reach.add(i)
        t = g.ops[i]
        if t[0] in ('+', '-', 'm2'): stack += [t[1], t[2]]
        elif t[0] in ('n',): stack.append(t[1])
        elif t[0] == '*': stack.append(t[2])
    cnt = {'+': 0, '-': 0, 'n': 0, '*': 0, 'm2': 0}
    for i in reach:
        o = g.ops[i][0]
        if o in cnt: cnt[o] += 1
    # weight: mul that can fuse into add -> treat all as 1; negs are cheap-ish (xor) ~1
    total = cnt['+'] + cnt['-'] + cnt['*'] + cnt['m2'] + 0.6 * cnt['n']
    # estimate FMA fusion: each '*' feeding exactly one +/- can fuse: approximate by
    # counting pairs: fused = min(#mul, #add+#sub)*... keep simple: subtract 0.45*min(muls, adds)
    fuse = min(cnt['*'] + cnt['m2'], cnt['+'] + cnt['-'])
    total -= 0.45 * fuse
    return total, cnt

def best_plan(n, depth=0):
    if n in _best: return _best[n]
    cands = []
    if n == 1:
        b = lambda g, xs: xs
        _best[n] = (b, 0, 'id'); return _best[n]
    if n == 2:
        _best[n] = (b_dft2, 4, 'dft2'); return _best[n]
    if n % 2 == 1 and n <= 25:
        cands.append((b_direct_sym(n), f'dsym{n}'))
    if is_prime(n) and n >= 5:
        planq = best_plan(n - 1)[0]
        cands.append((b_rader(n, planq), f'rader{n}({_best[n-1][2]})'))
    for (a, b) in factor_pairs(n):
        pa = best_plan(a)[0]; pb = best_plan(b)[0]
        if gcd(a, b) == 1:
            cands.append((b_pfa(a, b, pa, pb), f'pfa({a},{b})'))
            cands.append((b_pfa(b, a, pb, pa), f'pfa({b},{a})'))
        else:
            cands.append((b_ct(a, b, pa, pb), f'ct({a},{b})'))
            cands.append((b_ct(b, a, pb, pa), f'ct({b},{a})'))
    scored = []
    for bld, name in cands:
        c, cnt = count_ops(bld, n)
        scored.append((c, bld, name, cnt))
    scored.sort(key=lambda t: t[0])
    c, bld, name, cnt = scored[0]
    _best[n] = (bld, c, name)
    return _best[n]

if __name__ == '__main__':
    import numpy as np
    from gencore import eval_nodes
    rng = np.random.default_rng(0)
    for n in (2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 16, 17, 22, 23, 36, 45, 64):
        bld, cost, name = best_plan(n)
        g = G()
        xs = [(g.inp(f"r{j}"), g.inp(f"i{j}")) for j in range(n)]
        ys = bld(g, xs)
        x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        inputs = {}
        for j in range(n):
            inputs[f"r{j}"] = x[j].real; inputs[f"i{j}"] = x[j].imag
        flat = eval_nodes(g, inputs, [c for y in ys for c in y])
        got = np.array([flat[2*i] + 1j*flat[2*i+1] for i in range(n)])
        ref = np.fft.fft(x)
        err = np.linalg.norm(got - ref) / np.linalg.norm(ref)
        _, cnt = count_ops(bld, n)
        print(f"n={n:3d} plan={name:28s} cost={cost:8.1f} ops={cnt} relerr={err:.2e}")
        assert err < 1e-13, (n, err)
