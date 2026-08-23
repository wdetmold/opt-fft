# DFT network emitters: split-complex, operating on abstract value names.
# Each emitter takes an emitter context E and a list of (re,im) expression names,
# returns list of (re,im) names for outputs. All constants baked as hex doubles
# computed in long double.
import numpy as np

LD = np.longdouble
PI = LD('3.14159265358979323846264338327950288')

def fmt(x):
    return float(x).hex()

def trigc(num, den):
    # cos(2*pi*num/den) computed in long double with exact mod reduction
    num = num % den
    ang = LD(-2) * PI * LD(num) / LD(den)
    return np.cos(ang)

def trigs(num, den):
    num = num % den
    ang = LD(-2) * PI * LD(num) / LD(den)
    return np.sin(ang)

class E:
    def __init__(self):
        self.lines = []
        self.n = 0
        self.stats = {}
    def raw(self, s):
        self.lines.append(s)
    def v(self, expr):
        self.n += 1
        name = f"t{self.n}"
        self.lines.append(f"vd {name} = {expr};")
        return name
    def code(self, ind="  "):
        return ("\n"+ind).join(self.lines)

def cadd(e, x, y): return (e.v(f"{x[0]} + {y[0]}"), e.v(f"{x[1]} + {y[1]}"))
def csub(e, x, y): return (e.v(f"{x[0]} - {y[0]}"), e.v(f"{x[1]} - {y[1]}"))
def cneg(e, x):    return (e.v(f"-{x[0]}"), e.v(f"-{x[1]}"))

def KC(val):
    return f"K({fmt(val)})"

def cmul_const(e, x, num, den):
    """multiply complex x by W_den^num = exp(-2*pi*i*num/den); exploit special cases"""
    num = num % den
    if num == 0: return x
    c = trigc(num, den); s = trigs(num, den)
    # special: pure real/imag multiples
    if 4*num % den == 0:
        q = (4*num)//den
        if q == 1:  # -i
            return (x[1], e.v(f"-{x[0]}"))
        if q == 2:  # -1
            return (e.v(f"-{x[0]}"), e.v(f"-{x[1]}"))
        if q == 3:  # i
            return (e.v(f"-{x[1]}"), x[0])
    if 8*num % den == 0:
        # c = +-s: (c + i s)(a+ib) = c(a - b sgn) + i c(b + a sgn) with s = sgn*c
        sgn = 1.0 if s > 0 else -1.0
        op1 = '-' if sgn > 0 else '+'
        op2 = '+' if sgn > 0 else '-'
        r = e.v(f"{KC(c)} * ({x[0]} {op1} {x[1]})")
        i = e.v(f"{KC(c)} * ({x[1]} {op2} {x[0]})")
        return (r, i)
    # general: 2 mul + 2 fma  (r = c*a - s*b ; i = c*b + s*a)
    r = e.v(f"FMS({KC(c)}, {x[0]}, {KC(s)} * {x[1]})")
    i = e.v(f"FMA({KC(c)}, {x[1]}, {KC(s)} * {x[0]})")
    return (r, i)

# ---------------- small DFT codelets ----------------

def dft2(e, xs):
    a, b = xs
    return [cadd(e, a, b), csub(e, a, b)]

def dft3(e, xs, scale=1):
    # y0 = x0+x1+x2 ; y1 = x0 + w x1 + w^2 x2; y2 = x0 + w^2 x1 + w x2
    x0, x1, x2 = xs
    tr = e.v(f"{x1[0]} + {x2[0]}"); ti = e.v(f"{x1[1]} + {x2[1]}")
    ur = e.v(f"{x1[0]} - {x2[0]}"); ui = e.v(f"{x1[1]} - {x2[1]}")
    y0 = (e.v(f"{x0[0]} + {tr}"), e.v(f"{x0[1]} + {ti}"))
    mr = e.v(f"FNMA(K(0.5), {tr}, {x0[0]})")
    mi = e.v(f"FNMA(K(0.5), {ti}, {x0[1]})")
    s = trigs(1,3)  # = -sqrt(3)/2 (forward)
    vr = e.v(f"{KC(s)} * {ur}"); vi = e.v(f"{KC(s)} * {ui}")
    # y1 = m + i*v  (since x1 coeff: cos + i sin applied via symmetric fold)
    y1 = (e.v(f"{mr} - {vi}"), e.v(f"{mi} + {vr}"))
    y2 = (e.v(f"{mr} + {vi}"), e.v(f"{mi} - {vr}"))
    return [y0, y1, y2]

def dft4(e, xs):
    x0, x1, x2, x3 = xs
    t0 = cadd(e, x0, x2); t1 = csub(e, x0, x2)
    t2 = cadd(e, x1, x3); t3 = csub(e, x1, x3)
    y0 = cadd(e, t0, t2); y2 = csub(e, t0, t2)
    # y1 = t1 - i t3 ; y3 = t1 + i t3   (forward)
    y1 = (e.v(f"{t1[0]} + {t3[1]}"), e.v(f"{t1[1]} - {t3[0]}"))
    y3 = (e.v(f"{t1[0]} - {t3[1]}"), e.v(f"{t1[1]} + {t3[0]}"))
    return [y0, y1, y2, y3]

def dft5(e, xs):
    x0, x1, x2, x3, x4 = xs
    t1 = cadd(e, x1, x4); t3 = csub(e, x1, x4)
    t2 = cadd(e, x2, x3); t4 = csub(e, x2, x3)
    t5 = cadd(e, t1, t2)
    y0 = cadd(e, x0, t5)
    c1, c2 = trigc(1,5), trigc(2,5)
    s1, s2 = trigs(1,5), trigs(2,5)
    # A1 = x0 + c1 t1 + c2 t2 ; B1 = s1 t3 + s2 t4 ; y1 = A1 + iB1 ... sign per fold
    a1r = e.v(f"FMA({KC(c1)}, {t1[0]}, FMA({KC(c2)}, {t2[0]}, {x0[0]}))")
    a1i = e.v(f"FMA({KC(c1)}, {t1[1]}, FMA({KC(c2)}, {t2[1]}, {x0[1]}))")
    b1r = e.v(f"FMA({KC(s1)}, {t3[0]}, {KC(s2)} * {t4[0]})")
    b1i = e.v(f"FMA({KC(s1)}, {t3[1]}, {KC(s2)} * {t4[1]})")
    a2r = e.v(f"FMA({KC(c2)}, {t1[0]}, FMA({KC(c1)}, {t2[0]}, {x0[0]}))")
    a2i = e.v(f"FMA({KC(c2)}, {t1[1]}, FMA({KC(c1)}, {t2[1]}, {x0[1]}))")
    b2r = e.v(f"FMA({KC(s2)}, {t3[0]}, -({KC(s1)} * {t4[0]}))")
    b2i = e.v(f"FMA({KC(s2)}, {t3[1]}, -({KC(s1)} * {t4[1]}))")
    # X_k = A - iB with B = sum sin(2pi jk/5) o_j ; s already negative (forward)
    # X1 = A1 + i B1? derive: X1 = x0 + sum_j [e_j cos(2pi j/5) + ... ]; using w=e^{-2pi i/5}:
    # X1 = x0 + c1 e1 + c2 e2 + i(s1 o1 + s2 o2) where s=sin(-2pi j/5)<0
    y1 = (e.v(f"{a1r} - {b1i}"), e.v(f"{a1i} + {b1r}"))
    y4 = (e.v(f"{a1r} + {b1i}"), e.v(f"{a1i} - {b1r}"))
    y2 = (e.v(f"{a2r} - {b2i}"), e.v(f"{a2i} + {b2r}"))
    y3 = (e.v(f"{a2r} + {b2i}"), e.v(f"{a2i} - {b2r}"))
    return [y0, y1, y2, y3, y4]

def dft8(e, xs):
    x = xs
    t0 = cadd(e, x[0], x[4]); t1 = csub(e, x[0], x[4])
    t2 = cadd(e, x[2], x[6]); t3 = csub(e, x[2], x[6])
    t4 = cadd(e, x[1], x[5]); t5 = csub(e, x[1], x[5])
    t6 = cadd(e, x[3], x[7]); t7 = csub(e, x[3], x[7])
    u0 = cadd(e, t0, t2); u1 = csub(e, t0, t2)
    u2 = cadd(e, t4, t6); u3 = csub(e, t4, t6)
    y0 = cadd(e, u0, u2); y4 = csub(e, u0, u2)
    # y2 = u1 - i u3, y6 = u1 + i u3
    y2 = (e.v(f"{u1[0]} + {u3[1]}"), e.v(f"{u1[1]} - {u3[0]}"))
    y6 = (e.v(f"{u1[0]} - {u3[1]}"), e.v(f"{u1[1]} + {u3[0]}"))
    # odd part: v0 = t1, v1 = t5*w8, v2 = -i t3, v3 = t7*w8^3
    c = trigc(1,8)  # sqrt(2)/2
    # w8 = c - ic : t5*(c-ic) = c(t5r + t5i) + i c(t5i - t5r)
    a5r = e.v(f"{KC(c)} * ({t5[0]} + {t5[1]})"); a5i = e.v(f"{KC(c)} * ({t5[1]} - {t5[0]})")
    # w8^3 = -c - ic: t7*(-c-ic) = c(-t7r + t7i) + i c(-t7i - t7r)
    a7r = e.v(f"{KC(c)} * ({t7[1]} - {t7[0]})"); a7i = e.v(f"-{KC(c)} * ({t7[1]} + {t7[0]})")
    # -i t3 = (t3i, -t3r)
    m3 = (t3[1], e.v(f"-{t3[0]}"))
    w0 = cadd(e, t1, m3); w1 = csub(e, t1, m3)   # t1 +- (-i t3)
    w2 = cadd(e, (a5r,a5i), (a7r,a7i)); w3 = csub(e, (a5r,a5i), (a7r,a7i))
    y1 = cadd(e, w0, w2); y5 = csub(e, w0, w2)
    # y3 = w1 - i w3? verify numerically; tentative:
    y3 = (e.v(f"{w1[0]} + {w3[1]}"), e.v(f"{w1[1]} - {w3[0]}"))
    y7 = (e.v(f"{w1[0]} - {w3[1]}"), e.v(f"{w1[1]} + {w3[0]}"))
    return [y0, y1, y2, y3, y4, y5, y6, y7]

def dft_ct(e, xs, N1, N2):
    """Cooley-Tukey N = N1*N2 (N1 = radix over strided inputs), generic with twiddles.
       x[n] with n = N2*n1 + n2 -> X[k] with k = N1*k2 + k1"""
    N = N1 * N2
    # inner DFTs over n1 for each n2: input stride N2
    inner = {}
    for n2 in range(N2):
        sub = [xs[N2*n1 + n2] for n1 in range(N1)]
        ys = dft_small(e, sub, N1)
        for k1 in range(N1):
            inner[(k1, n2)] = cmul_const(e, ys[k1], k1*n2, N)
    out = [None]*N
    for k1 in range(N1):
        sub = [inner[(k1, n2)] for n2 in range(N2)]
        ys = dft_small(e, sub, N2)
        for k2 in range(N2):
            out[N1*k2 + k1] = ys[k2]
    return out

def dft_pfa(e, xs, A, B):
    """Good-Thomas prime factor N=A*B, gcd(A,B)=1. Twiddle-free.
       input index n: n == B*a + A*b mod N ruritanian; output k via CRT"""
    N = A * B
    # input map: n1 = n mod A, n2 = n mod B; x[(B*n1 + A*n2) mod N]
    inner = {}
    for n2 in range(B):
        sub = [xs[(B*n1 + A*n2) % N] for n1 in range(A)]
        ys = dft_small(e, sub, A)
        for k1 in range(A):
            inner[(k1, n2)] = ys[k1]
    out = [None]*N
    # output: X[k] where k mod A = k1*B mod A ... use CRT: k such that k=k1 (mod A), k=k2 (mod B)
    import math
    Binv = pow(B, -1, A)
    Ainv = pow(A, -1, B)
    for k1 in range(A):
        sub = [inner[(k1, n2)] for n2 in range(B)]
        ys = dft_small(e, sub, B)
        for k2 in range(B):
            k = (k1 * B * Binv + k2 * A * Ainv) % N
            out[k] = ys[k2]
    return out

def dft_prime_folded(e, xs, p):
    """direct symmetric folded DFT for prime p"""
    h = (p-1)//2
    x0 = xs[0]
    er, ei, orr, oi = [], [], [], []
    for j in range(1, h+1):
        er.append(e.v(f"{xs[j][0]} + {xs[p-j][0]}"))
        ei.append(e.v(f"{xs[j][1]} + {xs[p-j][1]}"))
        orr.append(e.v(f"{xs[j][0]} - {xs[p-j][0]}"))
        oi.append(e.v(f"{xs[j][1]} - {xs[p-j][1]}"))
    # y0
    sr, si = x0[0], x0[1]
    for j in range(h):
        sr = e.v(f"{sr} + {er[j]}")
        si = e.v(f"{si} + {ei[j]}")
    out = [None]*p
    out[0] = (sr, si)
    for k in range(1, h+1):
        cr = x0[0]; ci = x0[1]
        for j in range(1, h+1):
            c = trigc(k*j, p)
            cr = e.v(f"FMA({KC(c)}, {er[j-1]}, {cr})")
            ci = e.v(f"FMA({KC(c)}, {ei[j-1]}, {ci})")
        s0 = trigs(k, p)
        srr = e.v(f"{KC(s0)} * {orr[0]}")
        sii = e.v(f"{KC(s0)} * {oi[0]}")
        for j in range(2, h+1):
            s = trigs(k*j, p)
            srr = e.v(f"FMA({KC(s)}, {orr[j-1]}, {srr})")
            sii = e.v(f"FMA({KC(s)}, {oi[j-1]}, {sii})")
        # X_k = C + iS where S = sum s_kj o_j (s negative builtin)
        out[k]   = (e.v(f"{cr} - {sii}"), e.v(f"{ci} + {srr}"))
        out[p-k] = (e.v(f"{cr} + {sii}"), e.v(f"{ci} - {srr}"))
    return out

def dft_small(e, xs, N):
    if N == 2: return dft2(e, xs)
    if N == 3: return dft3(e, xs)
    if N == 4: return dft4(e, xs)
    if N == 5: return dft5(e, xs)
    if N == 8: return dft8(e, xs)
    if N == 9: return dft_ct(e, xs, 3, 3)
    if N == 6: return dft_pfa(e, xs, 2, 3)
    if N == 36: return dft_pfa(e, xs, 4, 9)
    if N == 45: return dft_pfa(e, xs, 9, 5)
    if N == 64: return dft_ct(e, xs, 8, 8)
    if N in (13, 17, 23): return dft_prime_folded(e, xs, N)
    raise ValueError(N)
