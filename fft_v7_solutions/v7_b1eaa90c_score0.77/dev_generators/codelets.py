# Codelet generator: straight-line complex DFT programs over 8-lane vectors.
import numpy as np
from math import gcd

LD = np.longdouble
TWO_PI = LD('6.283185307179586476925286766559005768394')

def tw(j, N):
    """exact-ish twiddle W_N^j = exp(-2*pi*i*j/N), double rounded from longdouble"""
    j = j % N
    ang = -TWO_PI * LD(j) / LD(N)
    return float(np.cos(ang)), float(np.sin(ang))

def hexf(x):
    if x == int(x) and abs(x) < 1e15:
        return f"{x:.1f}"
    return float(x).hex()

class E:
    """emitter with value numbering"""
    def __init__(self):
        self.stmts = []
        self.n = 0
    def t(self, expr):
        v = f"t{self.n}"; self.n += 1
        self.stmts.append(f"V {v} = {expr};")
        return v
    def code(self, indent="  "):
        return "\n".join(indent + s for s in self.stmts)

def cadd(e, a, b):  return (e.t(f"{a[0]} + {b[0]}"), e.t(f"{a[1]} + {b[1]}"))
def csub(e, a, b):  return (e.t(f"{a[0]} - {b[0]}"), e.t(f"{a[1]} - {b[1]}"))
def cneg(e, a):     return (e.t(f"-{a[0]}"), e.t(f"-{a[1]}"))
def cmuli(e, a):    # a * (+i)
    return (e.t(f"-{a[1]}"), e.t(f"{a[0]}"))
def cmulmi(e, a):   # a * (-i)
    return (e.t(f"{a[1]}"), e.t(f"-{a[0]}"))
def cmul_w(e, a, c, s):
    """a * (c + i s), general"""
    if s == 0.0:
        if c == 1.0: return a
        if c == -1.0: return cneg(e, a)
        return (e.t(f"{hexf(c)} * {a[0]}"), e.t(f"{hexf(c)} * {a[1]}"))
    if c == 0.0:
        if s == 1.0: return cmuli(e, a)
        if s == -1.0: return cmulmi(e, a)
        return (e.t(f"-({hexf(s)} * {a[1]})"), e.t(f"{hexf(s)} * {a[0]}"))
    re = e.t(f"{hexf(c)} * {a[0]} - {hexf(s)} * {a[1]}")
    im = e.t(f"{hexf(c)} * {a[1]} + {hexf(s)} * {a[0]}")
    return (re, im)

# ---------------- base codelets: return list of outputs X[k] ----------------
def fft2(e, x):
    return [cadd(e, x[0], x[1]), csub(e, x[0], x[1])]

def fft3(e, x):
    # X1 = x0 + c*t1 + i*s*t2 ; with w = exp(-2pi i/3) = c + i s
    c, s = tw(1, 3)
    t1 = cadd(e, x[1], x[2])
    t2 = csub(e, x[1], x[2])
    X0 = cadd(e, x[0], t1)
    ur = e.t(f"{hexf(c)} * {t1[0]} + {x[0][0]}")
    ui = e.t(f"{hexf(c)} * {t1[1]} + {x[0][1]}")
    # v = i*s*t2 = (-s*t2_im, s*t2_re)
    vr = e.t(f"-({hexf(s)} * {t2[1]})")
    vi = e.t(f"{hexf(s)} * {t2[0]}")
    X1 = (e.t(f"{ur} + {vr}"), e.t(f"{ui} + {vi}"))
    X2 = (e.t(f"{ur} - {vr}"), e.t(f"{ui} - {vi}"))
    return [X0, X1, X2]

def fft4(e, x):
    t0 = cadd(e, x[0], x[2]); t1 = csub(e, x[0], x[2])
    t2 = cadd(e, x[1], x[3]); t3 = csub(e, x[1], x[3])
    X0 = cadd(e, t0, t2); X2 = csub(e, t0, t2)
    mt3 = cmulmi(e, t3)          # -i * t3
    X1 = cadd(e, t1, mt3); X3 = csub(e, t1, mt3)
    return [X0, X1, X2, X3]

def fft5(e, x):
    c1, s1 = tw(1,5); c2, s2 = tw(2,5)
    u1 = cadd(e, x[1], x[4]); u2 = cadd(e, x[2], x[3])
    v1 = csub(e, x[1], x[4]); v2 = csub(e, x[2], x[3])
    X0 = cadd(e, x[0], cadd(e, u1, u2))
    # E_k = x0 + c_k1*u1 + c_k2*u2 ; S_k = s_k1*v1 + s_k2*v2 (complex, real coefs)
    def dotE(cA, cB):
        rr = e.t(f"{hexf(cA)} * {u1[0]} + {hexf(cB)} * {u2[0]} + {x[0][0]}")
        ii = e.t(f"{hexf(cA)} * {u1[1]} + {hexf(cB)} * {u2[1]} + {x[0][1]}")
        return (rr, ii)
    def dotS(sA, sB):
        rr = e.t(f"{hexf(sA)} * {v1[0]} + {hexf(sB)} * {v2[0]}")
        ii = e.t(f"{hexf(sA)} * {v1[1]} + {hexf(sB)} * {v2[1]}")
        return (rr, ii)
    E1 = dotE(c1, c2); S1 = dotS(s1, s2)
    E2 = dotE(c2, c1); S2 = dotS(s2, -s1)
    # X[k] = E_k - i*S_k ; X[N-k] = E_k + i*S_k
    def comb(Ek, Sk):
        a = (e.t(f"{Ek[0]} - {Sk[1]}"), e.t(f"{Ek[1]} + {Sk[0]}"))
        b = (e.t(f"{Ek[0]} + {Sk[1]}"), e.t(f"{Ek[1]} - {Sk[0]}"))
        return a, b
    X1, X4 = comb(E1, S1)
    X2, X3 = comb(E2, S2)
    return [X0, X1, X2, X3, X4]

def fft8(e, x):
    Ev = fft4(e, [x[0], x[2], x[4], x[6]])
    Od = fft4(e, [x[1], x[3], x[5], x[7]])
    C = "0x1.6a09e667f3bcdp-1"  # sqrt(2)/2
    out = [None]*8
    for k in range(4):
        o = Od[k]
        if k == 0:
            w = o
        elif k == 1:   # W8^1 = c*(1 - i)
            w = (e.t(f"{C} * ({o[0]} + {o[1]})"), e.t(f"{C} * ({o[1]} - {o[0]})"))
        elif k == 2:   # -i
            w = cmulmi(e, o)
        else:          # W8^3 = -c*(1 + i)
            w = (e.t(f"{C} * ({o[1]} - {o[0]})"), e.t(f"-({C} * ({o[0]} + {o[1]}))"))
        out[k]   = cadd(e, Ev[k], w)
        out[k+4] = csub(e, Ev[k], w)
    return out

def fft_ct(e, x, P, Q):
    """N = P*Q Cooley-Tukey DIT: n = Q*t + s (s in [0,Q)), X[P*k2+k1]"""
    N = P*Q
    A = []   # A[s] = P-point DFT of x[Q*t+s]
    for s in range(Q):
        A.append(fft_any(e, [x[Q*t + s] for t in range(P)], P))
    out = [None]*N
    for k1 in range(P):
        B = []
        for s in range(Q):
            c, sn = tw(s*k1, N)
            B.append(cmul_w(e, A[s][k1], c, sn))
        C = fft_any(e, B, Q)
        for k2 in range(Q):
            out[P*k2 + k1] = C[k2]
    return out

def fft_pfa(e, x, N1, N2):
    """N = N1*N2, gcd=1. input map n=(N2*a+N1*b)%N ; X[k]=C[k%N1][k%N2]"""
    N = N1*N2
    assert gcd(N1, N2) == 1
    A = []  # A[b] = N1-point DFT over a of x[(N2*a+N1*b)%N]
    for b in range(N2):
        A.append(fft_any(e, [x[(N2*a + N1*b) % N] for a in range(N1)], N1))
    # stage2: for each k1: N2-point DFT over b
    C = [[None]*N2 for _ in range(N1)]
    for k1 in range(N1):
        res = fft_any(e, [A[b][k1] for b in range(N2)], N2)
        for k2 in range(N2):
            C[k1][k2] = res[k2]
    out = [None]*N
    for k in range(N):
        out[k] = C[k % N1][k % N2]
    return out

def fft_halfmatrix(e, x, N):
    """odd prime N via symmetric half-matrix"""
    h = (N-1)//2
    u = [None]*(h+1); v = [None]*(h+1)
    for j in range(1, h+1):
        u[j] = cadd(e, x[j], x[N-j])
        v[j] = csub(e, x[j], x[N-j])
    # X0 = x0 + sum u
    acc = u[1]
    for j in range(2, h+1):
        acc = cadd(e, acc, u[j])
    X0 = cadd(e, x[0], acc)
    out = [None]*N
    out[0] = X0
    for k in range(1, h+1):
        cs = [tw(j*k, N) for j in range(1, h+1)]
        # E = x0 + sum_j cos*u[j] (complex); S = sum_j sin*v[j]
        er = f"{hexf(cs[0][0])} * {u[1][0]} + {x[0][0]}"
        ei = f"{hexf(cs[0][0])} * {u[1][1]} + {x[0][1]}"
        sr = f"{hexf(cs[0][1])} * {v[1][0]}"
        si = f"{hexf(cs[0][1])} * {v[1][1]}"
        er = e.t(er); ei = e.t(ei); sr = e.t(sr); si = e.t(si)
        for j in range(2, h+1):
            c, s = cs[j-1]
            er = e.t(f"{hexf(c)} * {u[j][0]} + {er}")
            ei = e.t(f"{hexf(c)} * {u[j][1]} + {ei}")
            sr = e.t(f"{hexf(s)} * {v[j][0]} + {sr}")
            si = e.t(f"{hexf(s)} * {v[j][1]} + {si}")
        out[k]   = (e.t(f"{er} - {si}"), e.t(f"{ei} + {sr}"))
        out[N-k] = (e.t(f"{er} + {si}"), e.t(f"{ei} - {sr}"))
    return out

def fft_any(e, x, N):
    if N == 6: return fft_pfa(e, x, 2, 3)
    if N == 2: return fft2(e, x)
    if N == 3: return fft3(e, x)
    if N == 4: return fft4(e, x)
    if N == 5: return fft5(e, x)
    if N == 8: return fft8(e, x)
    if N == 9: return fft_ct(e, x, 3, 3)
    raise ValueError(N)

def gen_staged(N, fac, pfa):
    """two-stage core with explicit intermediate arrays; fac=(P,Q).
    pfa=True: PFA maps, no twiddles. Loads xr[i*is_], stores yr[k*os_]."""
    P, Q = fac
    assert P*Q == N
    segs = []
    # stage1: for each second-index b in [0,Q): P-point DFT
    for b in range(Q):
        e = E()
        if pfa:
            idxs = [(Q*a + P*b) % N for a in range(P)]
        else:
            idxs = [Q*t + b for t in range(P)]
        x = [(e.t(f"xr[{i}*is_]"), e.t(f"xi[{i}*is_]")) for i in idxs]
        out = fft_any(e, x, P)
        if not pfa:
            out2 = []
            for k1 in range(P):
                c, s = tw(b*k1, N)
                out2.append(cmul_w(e, out[k1], c, s))
            out = out2
        stores = "\n".join(f"    Ar[{b*P+k1}] = {out[k1][0]}; Ai[{b*P+k1}] = {out[k1][1]};" for k1 in range(P))
        segs.append("  {\n" + e.code(indent="    ") + "\n" + stores + "\n  }")
    # stage2: for each k1 in [0,P): Q-point DFT over b
    for k1 in range(P):
        e = E()
        x = [(e.t(f"Ar[{b*P+k1}]"), e.t(f"Ai[{b*P+k1}]")) for b in range(Q)]
        out = fft_any(e, x, Q)
        stores = []
        for k2 in range(Q):
            if pfa:
                # X[k] for the unique k with k%P==?? : stage1 over 'a' computed DFT_P -> index k%P ; stage2 -> k%Q
                k = [kk for kk in range(N) if kk % P == k1 and kk % Q == k2][0]
            else:
                k = P*k2 + k1
            stores.append(f"    ST({k}, {out[k2][0]}, {out[k2][1]});")
        segs.append("  {\n" + e.code(indent="    ") + "\n" + "\n".join(stores) + "\n  }")
    body = "\n".join(segs)
    return body

def gen_core(N):
    """generate full N-point program; returns (code, nops). Loads from xr[i*IS],xi[i*IS], stores yr[i*OS],yi[i*OS]."""
    if N in (36, 45, 64):
        fac = {36:(4,9), 45:(9,5), 64:(8,8)}[N]
        pfa = N != 64
        body = gen_staged(N, fac, pfa)
        code = f"""static inline __attribute__((always_inline)) void fft{N}_core(const V* xr, const V* xi, V* yr, V* yi, long is_, long os_) {{
  V Ar[{N}], Ai[{N}];
#define ST(k, a, b) do {{ yr[(k)*os_] = (a); yi[(k)*os_] = (b); }} while (0)
{body}
#undef ST
}}
"""
        return code, 0
    e = E()
    x = []
    for i in range(N):
        x.append((e.t(f"xr[{i}*is_]"), e.t(f"xi[{i}*is_]")))
    if N == 6:    out = fft_pfa(e, x, 2, 3)
    elif N == 8:  out = fft8(e, x)
    elif N in (13, 17, 23): out = fft_halfmatrix(e, x, N)
    elif N in (2,3,4,5,9): out = fft_any(e, x, N)
    else: raise ValueError(N)
    body = e.code()
    stores = "\n".join(f"  yr[{k}*os_] = {out[k][0]}; yi[{k}*os_] = {out[k][1]};" for k in range(N))
    code = f"""static inline __attribute__((always_inline)) void fft{N}_core(const V* xr, const V* xi, V* yr, V* yi, long is_, long os_) {{
{body}
{stores}
}}
"""
    return code, e.n

if __name__ == "__main__":
    for N in (6,8,13,17,23,36,45,64):
        code, n = gen_core(N)
        print(f"N={N}: {n} temps, ~{len(code.splitlines())} lines")
