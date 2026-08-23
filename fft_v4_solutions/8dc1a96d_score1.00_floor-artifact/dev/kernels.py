# DFT kernel algorithm definitions over an abstract backend.
# Backend ops work on "values" = opaque handles for (re, im) component pairs.
import mpmath as mp
mp.mp.dps = 50

def tw(num, den):
    """exp(-2*pi*i*num/den) as (float cos, float sin-part) i.e. (re, im)."""
    num = num % den
    a = -2 * mp.pi * num / den
    return (float(mp.cos(a)), float(mp.sin(a)))

def cospi2(num, den):  # cos(2*pi*num/den)
    return float(mp.cos(2 * mp.pi * (num % den) / den))
def sinpi2(num, den):
    return float(mp.sin(2 * mp.pi * (num % den) / den))

class Backend:
    # subclass must provide: cadd, csub, neg, mul(value_component..., ) etc.
    pass

def dft2(be, xs):
    a, b = xs
    return [be.cadd(a, b), be.csub(a, b)]

def dft4(be, xs):
    x0, x1, x2, x3 = xs
    t0 = be.cadd(x0, x2); t1 = be.csub(x0, x2)
    t2 = be.cadd(x1, x3); t3 = be.csub(x1, x3)
    X0 = be.cadd(t0, t2); X2 = be.csub(t0, t2)
    # X1 = t1 - i t3 ; X3 = t1 + i t3
    X1 = be.cmix(t1, t3, -1)   # t1 + (-i)*t3
    X3 = be.cmix(t1, t3, +1)   # t1 + (+i)*t3
    return [X0, X1, X2, X3]

def dft_odd_prime(be, xs, p):
    """Even/odd direct DFT for odd prime p."""
    h = (p - 1) // 2
    x0 = xs[0]
    s = [None] * (h + 1); d = [None] * (h + 1)
    for j in range(1, h + 1):
        s[j] = be.cadd(xs[j], xs[p - j])
        d[j] = be.csub(xs[j], xs[p - j])
    X = [None] * p
    X[0] = be.csum([x0] + [s[j] for j in range(1, h + 1)])
    for k in range(1, h + 1):
        A = be.racc([(cospi2(j * k, p), s[j]) for j in range(1, h + 1)])
        Bv = be.racc([(sinpi2(j * k, p), d[j]) for j in range(1, h + 1)])
        T = be.cadd(x0, A)
        # X[k] = T - i*Bv ; X[p-k] = T + i*Bv
        X[k] = be.cmix(T, Bv, -1)
        X[p - k] = be.cmix(T, Bv, +1)
    return X

def dft_ct(be, xs, N1, N2):
    """Cooley-Tukey: N = N1*N2, input x[N2*j1 + j2], output X[k1 + N1*k2]."""
    N = N1 * N2
    A = []
    for j2 in range(N2):
        A.append(dft(be, [xs[N2 * j1 + j2] for j1 in range(N1)], N1))
    for j2 in range(N2):
        for k1 in range(N1):
            A[j2][k1] = be.cmulw(A[j2][k1], tw(j2 * k1, N))
    out = [None] * N
    for k1 in range(N1):
        col = dft(be, [A[j2][k1] for j2 in range(N2)], N2)
        for k2 in range(N2):
            out[k1 + N1 * k2] = col[k2]
    return out

def dft_pfa(be, xs, N1, N2):
    """Good-Thomas PFA: N = N1*N2 with gcd(N1,N2)=1."""
    N = N1 * N2
    y = [[xs[(N2 * j1 + N1 * j2) % N] for j2 in range(N2)] for j1 in range(N1)]
    A = []
    for j2 in range(N2):
        A.append(dft(be, [y[j1][j2] for j1 in range(N1)], N1))  # A[j2][k1]
    Bv = []
    for k1 in range(N1):
        Bv.append(dft(be, [A[j2][k1] for j2 in range(N2)], N2))  # Bv[k1][k2]
    out = [None] * N
    for k in range(N):
        out[k] = Bv[k % N1][k % N2]
    return out

def dft(be, xs, L):
    if L == 2: return dft2(be, xs)
    if L == 4: return dft4(be, xs)
    if L in (3, 5, 13, 17, 23): return dft_odd_prime(be, xs, L)
    if L == 6: return dft_pfa(be, xs, 2, 3)
    if L == 8: return dft_ct(be, xs, 2, 4)
    if L == 9: return dft_ct(be, xs, 3, 3)
    if L == 36: return dft_pfa(be, xs, 4, 9)
    if L == 45: return dft_pfa(be, xs, 9, 5)
    if L == 64: return dft_ct(be, xs, 8, 8)
    raise ValueError(L)
