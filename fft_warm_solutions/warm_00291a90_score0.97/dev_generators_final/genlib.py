import numpy as np
LD = np.longdouble
PI = LD('3.14159265358979323846264338327950288')

def hexd(v):
    return float(v).hex()

def tw(N, k):
    a = (-2*PI) * LD(k % N) / LD(N)
    c = float(np.cos(a)); s = float(np.sin(a))
    # snap near-exact values
    for v in (0.0, 1.0, -1.0, 0.5, -0.5):
        if abs(c - v) < 1e-17: c = v
        if abs(s - v) < 1e-17: s = v
    return c, s

class E:
    """emitter of straight-line __m512d code"""
    def __init__(self, pfx="t"):
        self.lines = []
        self.n = 0
        self.pfx = pfx
    def v(self, init=None):
        self.n += 1
        name = f"{self.pfx}{self.n}"
        if init is not None:
            self.lines.append(f"__m512d {name} = {init};")
        return name
    def raw(self, s):
        self.lines.append(s)
    def code(self, indent="    "):
        return "\n".join(indent + l for l in self.lines)

def setc(e, val):
    return e.v(f"_mm512_set1_pd({hexd(val)})")

def add(e,a,b): return e.v(f"_mm512_add_pd({a}, {b})")
def sub(e,a,b): return e.v(f"_mm512_sub_pd({a}, {b})")
def mul(e,a,b): return e.v(f"_mm512_mul_pd({a}, {b})")
def fmadd(e,a,b,c): return e.v(f"_mm512_fmadd_pd({a}, {b}, {c})")
def fnmadd(e,a,b,c): return e.v(f"_mm512_fnmadd_pd({a}, {b}, {c})")
def fmsub(e,a,b,c): return e.v(f"_mm512_fmsub_pd({a}, {b}, {c})")
def load(e, base, off):
    return e.v(f"_mm512_load_pd({base} + {off})")
def store(e, base, off, v):
    e.raw(f"_mm512_store_pd({base} + {off}, {v});")

def cadd(e,x,y): return (add(e,x[0],y[0]), add(e,x[1],y[1]))
def csub(e,x,y): return (sub(e,x[0],y[0]), sub(e,x[1],y[1]))
def cloadv(e, base, q):   # slot q: re at q*16, im at q*16+8 relative double ptr
    return (load(e, base, f"{q}*16"), load(e, base, f"{q}*16+8"))
def cstorev(e, base, q, v):
    store(e, base, f"{q}*16", v[0]); store(e, base, f"{q}*16+8", v[1])

def cmulw(e, x, N, k):
    k = k % N
    if k == 0: return x
    wr, wi = tw(N,k)
    xr, xi = x
    if wi == 0.0:
        if wr == 1.0: return x
        if wr == -1.0: return (e.v(f"_mm512_sub_pd(_mm512_setzero_pd(), {xr})"), e.v(f"_mm512_sub_pd(_mm512_setzero_pd(), {xi})"))
        W = setc(e, wr); return (mul(e,xr,W), mul(e,xi,W))
    if wr == 0.0:
        W = setc(e, abs(wi))
        if wi < 0:  # *(-i*|wi|): yr = wi... y = x*(0+wi i) = (-xi*wi... yr=-wi*xi? y=(xr+ixi)(iwi)= -wi xi + i wi xr; wi<0
            return (mul(e, xi, setc(e,-wi)), mul(e, xr, setc(e,wi)))
        else:
            return (mul(e, xi, setc(e,-wi)), mul(e, xr, setc(e,wi)))
    WR = setc(e, wr); WI = setc(e, wi)
    t = mul(e, xr, WR)
    yr = fnmadd(e, xi, WI, t)
    u = mul(e, xi, WR)
    yi = fmadd(e, xr, WI, u)
    return (yr, yi)

# ---- small DFT graphs on complex reg pairs ----
def dft2(e, x):
    return [cadd(e,x[0],x[1]), csub(e,x[0],x[1])]

def dft3(e, x):
    # X0 = x0+x1+x2; X1 = x0 + w x1 + w2 x2; X2 = conj-ish
    # t1 = x1+x2; t2 = x0 - 0.5 t1; t3 = sin(2pi/3)*(x1-x2) ; X1 = t2 - i t3; X2 = t2 + i t3
    s = 0.8660254037844386467637231707529362  # sin(pi/3)
    t1 = cadd(e, x[1], x[2])
    X0 = cadd(e, x[0], t1)
    H = setc(e, 0.5)
    t2 = (fnmadd(e, t1[0], H, x[0][0]), fnmadd(e, t1[1], H, x[0][1]))
    d = csub(e, x[1], x[2])
    S = setc(e, s)
    t3 = (mul(e, d[0], S), mul(e, d[1], S))
    # X1 = t2 - i t3 = (t2r + t3i, t2i - t3r) ; X2 = (t2r - t3i, t2i + t3r)
    X1 = (add(e, t2[0], t3[1]), sub(e, t2[1], t3[0]))
    X2 = (sub(e, t2[0], t3[1]), add(e, t2[1], t3[0]))
    return [X0, X1, X2]

def dft4(e, x):
    t0 = cadd(e, x[0], x[2]); t1 = csub(e, x[0], x[2])
    t2 = cadd(e, x[1], x[3]); t3 = csub(e, x[1], x[3])
    X0 = cadd(e, t0, t2); X2 = csub(e, t0, t2)
    # X1 = t1 - i t3 ; X3 = t1 + i t3
    X1 = (add(e, t1[0], t3[1]), sub(e, t1[1], t3[0]))
    X3 = (sub(e, t1[0], t3[1]), add(e, t1[1], t3[0]))
    return [X0, X1, X2, X3]

def dft5(e, x):
    # Winograd-style DFT5
    # constants
    c1 = float(np.cos(-2*PI/5)); s1 = float(np.sin(-2*PI/5))
    c2 = float(np.cos(-4*PI/5)); s2 = float(np.sin(-4*PI/5))
    t1 = cadd(e, x[1], x[4]); t2 = csub(e, x[1], x[4])
    t3 = cadd(e, x[2], x[3]); t4 = csub(e, x[2], x[3])
    X0r = add(e, x[0][0], add(e, t1[0], t3[0]))
    X0i = add(e, x[0][1], add(e, t1[1], t3[1]))
    C1 = setc(e, c1); C2 = setc(e, c2); S1 = setc(e, s1); S2 = setc(e, s2)
    # A1 = x0 + c1 t1 + c2 t3 ; A2 = x0 + c2 t1 + c1 t3
    A1r = fmadd(e, t3[0], C2, fmadd(e, t1[0], C1, x[0][0]))
    A1i = fmadd(e, t3[1], C2, fmadd(e, t1[1], C1, x[0][1]))
    A2r = fmadd(e, t3[0], C1, fmadd(e, t1[0], C2, x[0][0]))
    A2i = fmadd(e, t3[1], C1, fmadd(e, t1[1], C2, x[0][1]))
    # B1 = s1 t2 + s2 t4 ; B2 = s2 t2 - s1 t4   (as multipliers of -i)
    B1r = fmadd(e, t4[0], S2, mul(e, t2[0], S1))
    B1i = fmadd(e, t4[1], S2, mul(e, t2[1], S1))
    B2r = fmsub(e, t2[0], S2, mul(e, t4[0], S1))
    B2i = fmsub(e, t2[1], S2, mul(e, t4[1], S1))
    # X1 = A1 + i B1? check: X1 = x0 + w x1 + w2 x2 + w3 x3 + w4 x4
    #  = x0 + c1(x1+x4) + c2(x2+x3) + i[s1(x1-x4) + s2(x2-x3)]
    X1 = (sub(e, A1r, B1i), add(e, A1i, B1r))
    X4 = (add(e, A1r, B1i), sub(e, A1i, B1r))
    X2 = (sub(e, A2r, B2i), add(e, A2i, B2r))
    X3 = (add(e, A2r, B2i), sub(e, A2i, B2r))
    return [(X0r,X0i), X1, X2, X3, X4]

def dft8(e, x):
    # radix-2 DIT: split even/odd
    c = 0.7071067811865475244008443621048490
    t0 = cadd(e,x[0],x[4]); t4 = csub(e,x[0],x[4])
    t2 = cadd(e,x[2],x[6]); t6 = csub(e,x[2],x[6])
    t1 = cadd(e,x[1],x[5]); t5 = csub(e,x[1],x[5])
    t3 = cadd(e,x[3],x[7]); t7 = csub(e,x[3],x[7])
    # even part dft4 on (t0,t2), odd dft4 on (t1,t3)
    a0 = cadd(e,t0,t2); a2 = csub(e,t0,t2)
    a1 = cadd(e,t1,t3); a3 = csub(e,t1,t3)
    X0 = cadd(e,a0,a1); X4 = csub(e,a0,a1)
    # X2 = a2 - i a3 ; X6 = a2 + i a3
    X2 = (add(e,a2[0],a3[1]), sub(e,a2[1],a3[0]))
    X6 = (sub(e,a2[0],a3[1]), add(e,a2[1],a3[0]))
    # odd outputs: b0 = t4 - i t6 ; b2 = t4 + i t6
    b0 = (add(e,t4[0],t6[1]), sub(e,t4[1],t6[0]))
    b2 = (sub(e,t4[0],t6[1]), add(e,t4[1],t6[0]))
    # c0 = (t5 - i t7) * w8 = ((t5r+t7i) + i(t5i-t7r)) * (c - ic)
    C = setc(e, c)
    u0r = add(e,t5[0],t7[1]); u0i = sub(e,t5[1],t7[0])
    c0 = (mul(e, add(e,u0r,u0i), C), mul(e, sub(e,u0i,u0r), C))
    # c2 = (t5 + i t7) * w8^3 = u2 * (-c - i c): r = -c*(u2r - u2i); i = -c*(u2i + u2r)
    u2r = sub(e,t5[0],t7[1]); u2i = add(e,t5[1],t7[0])
    c2 = (mul(e, sub(e,u2i,u2r), C), e.v(f"_mm512_xor_pd(_mm512_set1_pd(-0.0), {mul(e, add(e,u2i,u2r), C)})"))
    X1 = cadd(e,b0,c0); X5 = csub(e,b0,c0)
    X3 = cadd(e,b2,c2); X7 = csub(e,b2,c2)
    return [X0,X1,X2,X3,X4,X5,X6,X7]

def pfa_maps(N1, N2):
    """Good-Thomas input/output index maps for N=N1*N2, gcd=1.
    in: n = (N2*n1 + N1*n2) mod N ; out: k = CRT(k1, k2)"""
    N = N1*N2
    inmap = [[ (N2*n1 + N1*n2) % N for n2 in range(N2)] for n1 in range(N1)]
    # output: k ≡ k1 mod N1, k ≡ k2 mod N2
    # k = (k1 * N2 * (N2^-1 mod N1) + k2 * N1 * (N1^-1 mod N2)) mod N
    i2 = pow(N2, -1, N1); i1 = pow(N1, -1, N2)
    outmap = [[ (k1*N2*i2 + k2*N1*i1) % N for k2 in range(N2)] for k1 in range(N1)]
    return inmap, outmap
