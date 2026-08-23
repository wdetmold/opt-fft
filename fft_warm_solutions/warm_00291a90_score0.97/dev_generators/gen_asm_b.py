import numpy as np
from genlib import LD, PI, hexd, pfa_maps
from gen_asm import A

class Tab:
    """per-size vector-broadcast constant table"""
    def __init__(self, tag):
        self.tag = tag
        self.vals = []
        self.idx = {}
    def get(self, v):
        key = float(v).hex()
        if key not in self.idx:
            self.idx[key] = len(self.vals)
            self.vals.append(float(v))
        return self.idx[key]
    def decl(self):
        rows = []
        for v in self.vals:
            h = hexd(v)
            rows.append("{" + ", ".join([h]*8) + "}")
        return f"static const double CTV_{self.tag}[{max(1,len(self.vals))}][8] ALIGN64 = {{ {', '.join(rows) if rows else '{0}'} }};\n"

class Ctx:
    """asm emission context with constant caching"""
    def __init__(self, a, tab):
        self.a = a; self.tab = tab
        self.cache = {}
    def const(self, v):
        key = float(v).hex()
        if key in self.cache: return self.cache[key]
        idx = self.tab.get(v)
        r = self.a.alloc()
        self.a.ins(f"vmovapd {idx*64}(%[ctv]), %%zmm{r}")
        self.cache[key] = r
        return r
    def flush(self):
        for r in self.cache.values():
            self.a.rel(r)
        self.cache = {}

def tw_ld(N, k):
    a = (-2*PI) * LD(k % N) / LD(N)
    c = float(np.cos(a)); s = float(np.sin(a))
    for v in (0.0, 1.0, -1.0, 0.5, -0.5):
        if abs(c - v) < 1e-17: c = v
        if abs(s - v) < 1e-17: s = v
    return c, s

def acadd(c, x, y, kill=False):
    a = c.a
    if kill:
        return (a.add(x[0], y[0], kill=(x[0], y[0])) if False else a.add(x[0], y[0]), a.add(x[1], y[1]))
    return (a.add(x[0], y[0]), a.add(x[1], y[1]))

def acsub(c, x, y):
    a = c.a
    return (a.sub(x[0], y[0]), a.sub(x[1], y[1]))

def arel(c, x):
    c.a.rel(x[0]); c.a.rel(x[1])

def acmulw(c, x, N, k, kill=True):
    """x * exp(-2pi i k/N); kills x if kill"""
    a = c.a
    k = k % N
    if k == 0: return x
    wr, wi = tw_ld(N, k)
    xr, xi = x
    if wi == 0.0 and wr == -1.0:
        z = c.const(0.0)
        yr = a.sub(z, xr); yi = a.sub(z, xi)
        if kill: a.rel(xr); a.rel(xi)
        return (yr, yi)
    if wr == 0.0:
        # x*(i wi) = (-wi*xi, wi*xr)
        W = c.const(wi); NW = c.const(-wi)
        yr = a.mul(xi, NW); yi = a.mul(xr, W)
        if kill: a.rel(xr); a.rel(xi)
        return (yr, yi)
    WR = c.const(wr); WI = c.const(wi)
    t = a.mul(xr, WR)
    a.fnma(t, xi, WI)           # t = wr*xr - wi*xi
    u = a.mul(xi, WR)
    a.fma(u, xr, WI)            # u = wr*xi + wi*xr
    if kill: a.rel(xr); a.rel(xi)
    return (t, u)

def adft2(c, x):
    X0 = acadd(c, x[0], x[1]); X1 = acsub(c, x[0], x[1])
    arel(c, x[0]); arel(c, x[1])
    return [X0, X1]

def adft3(c, x):
    a = c.a
    s3 = 0.8660254037844386467637231707529362
    t1 = acadd(c, x[1], x[2])
    X0 = acadd(c, x[0], t1)
    H = c.const(0.5)
    # t2 = x0 - 0.5*t1 : t2 = fnma into copy of x0... use: t2r = mov(x0r); fnma(t2r, t1r, H)
    t2r = a.mov(x[0][0]); a.fnma(t2r, t1[0], H)
    t2i = a.mov(x[0][1]); a.fnma(t2i, t1[1], H)
    arel(c, t1); arel(c, x[0])
    d = acsub(c, x[1], x[2])
    arel(c, x[1]); arel(c, x[2])
    S = c.const(s3)
    t3r = a.mul(d[0], S); t3i = a.mul(d[1], S)
    arel(c, d)
    X1 = (a.add(t2r, t3i), a.sub(t2i, t3r))
    X2 = (a.sub(t2r, t3i), a.add(t2i, t3r))
    a.rel(t2r); a.rel(t2i); a.rel(t3r); a.rel(t3i)
    return [X0, X1, X2]

def adft4(c, x):
    a = c.a
    t0 = acadd(c, x[0], x[2]); t1 = acsub(c, x[0], x[2])
    arel(c, x[0]); arel(c, x[2])
    t2 = acadd(c, x[1], x[3]); t3 = acsub(c, x[1], x[3])
    arel(c, x[1]); arel(c, x[3])
    X0 = acadd(c, t0, t2); X2 = acsub(c, t0, t2)
    arel(c, t0); arel(c, t2)
    X1 = (a.add(t1[0], t3[1]), a.sub(t1[1], t3[0]))
    X3 = (a.sub(t1[0], t3[1]), a.add(t1[1], t3[0]))
    arel(c, t1); arel(c, t3)
    return [X0, X1, X2, X3]

def adft5(c, x):
    a = c.a
    c1, s1 = tw_ld(5, 1); c2, s2 = tw_ld(5, 2)
    t1 = acadd(c, x[1], x[4]); t2 = acsub(c, x[1], x[4])
    arel(c, x[1]); arel(c, x[4])
    t3 = acadd(c, x[2], x[3]); t4 = acsub(c, x[2], x[3])
    arel(c, x[2]); arel(c, x[3])
    C1 = c.const(c1); C2 = c.const(c2); S1 = c.const(s1); S2 = c.const(s2)
    X0r = a.add(x[0][0], t1[0]); a.ins(f"vaddpd %%zmm{t3[0]}, %%zmm{X0r}, %%zmm{X0r}")
    X0i = a.add(x[0][1], t1[1]); a.ins(f"vaddpd %%zmm{t3[1]}, %%zmm{X0i}, %%zmm{X0i}")
    A1r = a.mov(x[0][0]); a.fma(A1r, t1[0], C1); a.fma(A1r, t3[0], C2)
    A1i = a.mov(x[0][1]); a.fma(A1i, t1[1], C1); a.fma(A1i, t3[1], C2)
    A2r = a.mov(x[0][0]); a.fma(A2r, t1[0], C2); a.fma(A2r, t3[0], C1)
    A2i = a.mov(x[0][1]); a.fma(A2i, t1[1], C2); a.fma(A2i, t3[1], C1)
    arel(c, x[0]); arel(c, t1); arel(c, t3)
    B1r = a.mul(t2[0], S1); a.fma(B1r, t4[0], S2)
    B1i = a.mul(t2[1], S1); a.fma(B1i, t4[1], S2)
    B2r = a.mul(t2[0], S2); a.fnma(B2r, t4[0], S1)
    B2i = a.mul(t2[1], S2); a.fnma(B2i, t4[1], S1)
    arel(c, t2); arel(c, t4)
    X1 = (a.sub(A1r, B1i), a.add(A1i, B1r))
    X4 = (a.add(A1r, B1i), a.sub(A1i, B1r))
    a.rel(A1r); a.rel(A1i); a.rel(B1r); a.rel(B1i)
    X2 = (a.sub(A2r, B2i), a.add(A2i, B2r))
    X3 = (a.add(A2r, B2i), a.sub(A2i, B2r))
    a.rel(A2r); a.rel(A2i); a.rel(B2r); a.rel(B2i)
    return [(X0r, X0i), X1, X2, X3, X4]

def adft8(c, x):
    a = c.a
    cc = 0.7071067811865475244008443621048490
    t0 = acadd(c, x[0], x[4]); t4 = acsub(c, x[0], x[4]); arel(c, x[0]); arel(c, x[4])
    t2 = acadd(c, x[2], x[6]); t6 = acsub(c, x[2], x[6]); arel(c, x[2]); arel(c, x[6])
    t1 = acadd(c, x[1], x[5]); t5 = acsub(c, x[1], x[5]); arel(c, x[1]); arel(c, x[5])
    t3 = acadd(c, x[3], x[7]); t7 = acsub(c, x[3], x[7]); arel(c, x[3]); arel(c, x[7])
    a0 = acadd(c, t0, t2); a2 = acsub(c, t0, t2); arel(c, t0); arel(c, t2)
    a1 = acadd(c, t1, t3); a3 = acsub(c, t1, t3); arel(c, t1); arel(c, t3)
    X0 = acadd(c, a0, a1); X4 = acsub(c, a0, a1); arel(c, a0); arel(c, a1)
    X2 = (a.add(a2[0], a3[1]), a.sub(a2[1], a3[0]))
    X6 = (a.sub(a2[0], a3[1]), a.add(a2[1], a3[0]))
    arel(c, a2); arel(c, a3)
    b0 = (a.add(t4[0], t6[1]), a.sub(t4[1], t6[0]))
    b2 = (a.sub(t4[0], t6[1]), a.add(t4[1], t6[0]))
    arel(c, t4); arel(c, t6)
    C = c.const(cc)
    u0r = a.add(t5[0], t7[1]); u0i = a.sub(t5[1], t7[0])
    u2r = a.sub(t5[0], t7[1]); u2i = a.add(t5[1], t7[0])
    arel(c, t5); arel(c, t7)
    c0r = a.add(u0r, u0i); c0r2 = a.mul(c0r, C, kill=(c0r,))
    c0i = a.sub(u0i, u0r); c0i2 = a.mul(c0i, C, kill=(c0i,))
    a.rel(u0r); a.rel(u0i)
    c2r = a.sub(u2i, u2r); c2r2 = a.mul(c2r, C, kill=(c2r,))
    c2i = a.add(u2i, u2r)
    NC = c.const(-cc)
    c2i2 = a.mul(c2i, NC, kill=(c2i,))
    a.rel(u2r); a.rel(u2i)
    X1 = (a.add(b0[0], c0r2), a.add(b0[1], c0i2))
    X5 = (a.sub(b0[0], c0r2), a.sub(b0[1], c0i2))
    arel(c, b0); a.rel(c0r2); a.rel(c0i2)
    X3 = (a.add(b2[0], c2r2), a.add(b2[1], c2i2))
    X7 = (a.sub(b2[0], c2r2), a.sub(b2[1], c2i2))
    arel(c, b2); a.rel(c2r2); a.rel(c2i2)
    return [X0, X1, X2, X3, X4, X5, X6, X7]

def adft9(c, x):
    """3x3 CT; consumes x (9 pairs)"""
    sub = []
    for b in range(3):
        sub.append(adft3(c, [x[3*a+b] for a in range(3)]))
    out = [None]*9
    for q in range(3):
        col = [acmulw(c, sub[b][q], 9, q*b) for b in range(3)]
        res = adft3(c, col)
        for t in range(3):
            out[3*t+q] = res[t]
    return out

def adft_graph(c, x, N):
    return {2:adft2, 3:adft3, 4:adft4, 5:adft5, 8:adft8, 9:adft9}[N](c, x)

def emit_twostage_asm(a, tab, L, src, ses, dst, des):
    """two-stage DFT via scratch SC (operand 'SC', stride 16)."""
    if L == 36: N1, N2, mode = 4, 9, 'pfa'
    elif L == 45: N1, N2, mode = 5, 9, 'pfa'
    elif L == 64: N1, N2, mode = 8, 8, 'ct'
    ctx = Ctx(a, tab)
    if mode == 'pfa':
        inm, outm = pfa_maps(N1, N2)
        for n2 in range(N2):
            x = []
            for n1 in range(N1):
                q = inm[n1][n2]
                x.append((a.ld(src, q*ses*8), a.ld(src, q*ses*8+64)))
            y = adft_graph(ctx, x, N1)
            for k1 in range(N1):
                a.st('SC', (k1*N2+n2)*128, y[k1][0]); a.st('SC', (k1*N2+n2)*128+64, y[k1][1])
                arel(ctx, y[k1])
            ctx.flush()
        for k1 in range(N1):
            x = [(a.ld('SC', (k1*N2+n2)*128), a.ld('SC', (k1*N2+n2)*128+64)) for n2 in range(N2)]
            y = adft_graph(ctx, x, N2)
            for k2 in range(N2):
                ko = outm[k1][k2]
                a.st(dst, ko*des*8, y[k2][0]); a.st(dst, ko*des*8+64, y[k2][1])
                arel(ctx, y[k2])
            ctx.flush()
    else:
        for n2 in range(N2):
            x = [(a.ld(src, (8*n1+n2)*ses*8), a.ld(src, (8*n1+n2)*ses*8+64)) for n1 in range(N1)]
            y = adft_graph(ctx, x, N1)
            for k1 in range(N1):
                t = acmulw(ctx, y[k1], 64, k1*n2)
                a.st('SC', (k1*N2+n2)*128, t[0]); a.st('SC', (k1*N2+n2)*128+64, t[1])
                arel(ctx, t)
            ctx.flush()
        for k1 in range(N1):
            x = [(a.ld('SC', (k1*N2+n2)*128), a.ld('SC', (k1*N2+n2)*128+64)) for n2 in range(N2)]
            y = adft_graph(ctx, x, N2)
            for k2 in range(N2):
                ko = k2*8 + k1
                a.st(dst, ko*des*8, y[k2][0]); a.st(dst, ko*des*8+64, y[k2][1])
                arel(ctx, y[k2])
            ctx.flush()

def gen_twostage_asm_fns(L, RS, PS):
    tab = Tab(str(L))
    fns = []
    for nm, ses, des in ((f"dft{L}_rr", RS, RS), (f"dft{L}_cc", 16, 16), (f"dft{L}_pc", PS, 16), (f"dft{L}_cp", 16, PS)):
        a = A()
        emit_twostage_asm(a, tab, L, 'src', ses, 'dst', des)
        assert not a.live, a.live
        body = "\\n\\t".join(a.lines)
        clob = ", ".join(f'"zmm{i}"' for i in range(32)) + ', "memory"'
        fns.append(f"""static void __attribute__((noinline)) {nm}(const double* src, double* dst){{
    __asm__ volatile("{body}"
    : : [src]"r"(src), [dst]"r"(dst), [SC]"r"(SCB_{L}), [ctv]"r"(CTV_{L})
    : {clob});
}}
""")
    decl = f"static double SCB_{L}[{L}][16] ALIGN64;\n" + tab.decl()
    return decl + "\n".join(fns)
