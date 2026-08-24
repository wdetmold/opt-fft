import numpy as np
from genlib import hexd

PI = np.longdouble('3.14159265358979323846264338327950288')

def W(L, t):
    """exact e^{-2pi i t/L} in longdouble -> (cos_hex, sin_hex) doubles"""
    ang = (-2*PI)*np.longdouble(t % L)/np.longdouble(L)
    return float(np.cos(ang)), float(np.sin(ang))

class Emit:
    def __init__(self, pfx="v"):
        self.lines=[]; self.n=0
        self.pfx=pfx
        self.consts = {}   # value-> name
    def v(self, expr):
        nm = f"{self.pfx}{self.n}"; self.n+=1
        self.lines.append(f"__m512d {nm} = {expr};")
        return nm
    def raw(self,s): self.lines.append(s)
    def K(self, val):
        """named broadcast constant, deduped; declared at function top later"""
        h = hexd(val)
        if h in self.consts: return self.consts[h]
        nm = f"K{self.pfx}{len(self.consts)}"
        self.consts[h] = nm
        return nm
    def const_decls(self):
        return [f"__m512d {nm} = _mm512_set1_pd({h});" for h,nm in self.consts.items()]
    def code(self, indent="    "):
        return "\n".join(indent+l for l in self.lines)

def cadd(e,a,b): return (e.v(f"_mm512_add_pd({a[0]},{b[0]})"), e.v(f"_mm512_add_pd({a[1]},{b[1]})"))
def csub(e,a,b): return (e.v(f"_mm512_sub_pd({a[0]},{b[0]})"), e.v(f"_mm512_sub_pd({a[1]},{b[1]})"))

def cmul_const(e, a, c, s):
    """a * (c + i s), constants"""
    if s == 0.0:
        if c == 1.0: return a
        if c == -1.0: return (e.v(f"_mm512_sub_pd(_mm512_setzero_pd(),{a[0]})"), e.v(f"_mm512_sub_pd(_mm512_setzero_pd(),{a[1]})"))
        k = e.K(c)
        return (e.v(f"_mm512_mul_pd({k},{a[0]})"), e.v(f"_mm512_mul_pd({k},{a[1]})"))
    if c == 0.0:
        k = e.K(s)
        if s == -1.0: return (a[1], e.v(f"_mm512_sub_pd(_mm512_setzero_pd(),{a[0]})"))
        if s == 1.0:  return (e.v(f"_mm512_sub_pd(_mm512_setzero_pd(),{a[1]})"), a[0])
        # (ar+i ai)(i s) = -s ai + i s ar
        return (e.v(f"_mm512_mul_pd(_mm512_sub_pd(_mm512_setzero_pd(),{k}),{a[1]})"),
                e.v(f"_mm512_mul_pd({k},{a[0]})"))
    kc, ks = e.K(c), e.K(s)
    re = e.v(f"_mm512_mul_pd({kc},{a[0]})")
    re = e.v(f"_mm512_fnmadd_pd({ks},{a[1]},{re})")
    im = e.v(f"_mm512_mul_pd({kc},{a[1]})")
    im = e.v(f"_mm512_fmadd_pd({ks},{a[0]},{im})")
    return (re, im)

def dft2(e, x):
    return [cadd(e,x[0],x[1]), csub(e,x[0],x[1])]

def dft4(e, x):
    t0 = cadd(e,x[0],x[2]); t1 = csub(e,x[0],x[2])
    t2 = cadd(e,x[1],x[3]); t3 = csub(e,x[1],x[3])
    X0 = cadd(e,t0,t2); X2 = csub(e,t0,t2)
    # X1 = t1 - i t3 ; X3 = t1 + i t3
    X1 = (e.v(f"_mm512_add_pd({t1[0]},{t3[1]})"), e.v(f"_mm512_sub_pd({t1[1]},{t3[0]})"))
    X3 = (e.v(f"_mm512_sub_pd({t1[0]},{t3[1]})"), e.v(f"_mm512_add_pd({t1[1]},{t3[0]})"))
    return [X0,X1,X2,X3]

def dft3(e, x):
    c, s = W(3,1)   # cos(-2pi/3) = -0.5, sin = -sqrt(3)/2
    t  = cadd(e,x[1],x[2])
    X0 = cadd(e,x[0],t)
    # u = x0 - t/2
    kh = e.K(0.5)
    ur = e.v(f"_mm512_fnmadd_pd({kh},{t[0]},{x[0][0]})")
    ui = e.v(f"_mm512_fnmadd_pd({kh},{t[1]},{x[0][1]})")
    # v = s*(x1-x2) where s = sin(-2pi/3) (negative); X1 = u + i v ; X2 = u - i v
    d  = csub(e,x[1],x[2])
    ks = e.K(s)
    vr = e.v(f"_mm512_mul_pd({ks},{d[0]})")
    vi = e.v(f"_mm512_mul_pd({ks},{d[1]})")
    # X1 = u + i*v = (ur - vi, ui + vr)
    X1 = (e.v(f"_mm512_sub_pd({ur},{vi})"), e.v(f"_mm512_add_pd({ui},{vr})"))
    X2 = (e.v(f"_mm512_add_pd({ur},{vi})"), e.v(f"_mm512_sub_pd({ui},{vr})"))
    return [X0,X1,X2]

def dft5(e, x):
    # folded prime style
    c1,s1 = W(5,1); c2,s2 = W(5,2)   # e^{-2pi i/5}
    S1 = cadd(e,x[1],x[4]); D1 = csub(e,x[1],x[4])
    S2 = cadd(e,x[2],x[3]); D2 = csub(e,x[2],x[3])
    X0 = cadd(e, cadd(e,S1,S2), x[0])
    k1, k2 = e.K(c1), e.K(c2)
    p1r = e.v(f"_mm512_fmadd_pd({k1},{S1[0]},{x[0][0]})"); p1r = e.v(f"_mm512_fmadd_pd({k2},{S2[0]},{p1r})")
    p1i = e.v(f"_mm512_fmadd_pd({k1},{S1[1]},{x[0][1]})"); p1i = e.v(f"_mm512_fmadd_pd({k2},{S2[1]},{p1i})")
    p2r = e.v(f"_mm512_fmadd_pd({k2},{S1[0]},{x[0][0]})"); p2r = e.v(f"_mm512_fmadd_pd({k1},{S2[0]},{p2r})")
    p2i = e.v(f"_mm512_fmadd_pd({k2},{S1[1]},{x[0][1]})"); p2i = e.v(f"_mm512_fmadd_pd({k1},{S2[1]},{p2i})")
    # Q_k = sum_j sin(-2pi jk/5) d_j ; X_k = P_k + i Q_k?? derive: X_k = x0 + sum s_j cos + i? 
    # X_k = x0 + sum_j [S_j cos(2pi jk/5) - i D_j sin(2pi jk/5)] (with sin positive convention)
    # our s1 = sin(-2pi/5) = -sin(2pi/5). Let sg1 = sin(2pi/5) = -s1, sg2 = sin(4pi/5) = -s2.
    m1, m2 = e.K(-s1), e.K(-s2)
    q1r = e.v(f"_mm512_mul_pd({m1},{D1[1]})"); q1r = e.v(f"_mm512_fmadd_pd({m2},{D2[1]},{q1r})")
    q1i = e.v(f"_mm512_mul_pd({m1},{D1[0]})"); q1i = e.v(f"_mm512_fmadd_pd({m2},{D2[0]},{q1i})")
    q2r = e.v(f"_mm512_mul_pd({m2},{D1[1]})"); q2r = e.v(f"_mm512_fnmadd_pd({m1},{D2[1]},{q2r})")
    q2i = e.v(f"_mm512_mul_pd({m2},{D1[0]})"); q2i = e.v(f"_mm512_fnmadd_pd({m1},{D2[0]},{q2i})")
    # X1 = (p1r + q1r_part...) : -i D sin term: re += sin*Di ; im -= sin*Dr
    X1 = (e.v(f"_mm512_add_pd({p1r},{q1r})"), e.v(f"_mm512_sub_pd({p1i},{q1i})"))
    X4 = (e.v(f"_mm512_sub_pd({p1r},{q1r})"), e.v(f"_mm512_add_pd({p1i},{q1i})"))
    X2 = (e.v(f"_mm512_add_pd({p2r},{q2r})"), e.v(f"_mm512_sub_pd({p2i},{q2i})"))
    X3 = (e.v(f"_mm512_sub_pd({p2r},{q2r})"), e.v(f"_mm512_add_pd({p2i},{q2i})"))
    return [X0,X1,X2,X3,X4]

def dft8(e, x):
    # radix-2 DIT style split: evens/odds
    ev = dft4(e, [x[0],x[2],x[4],x[6]])
    # odd part: DFT4 of odds then twiddle W8^k
    od = dft4(e, [x[1],x[3],x[5],x[7]])
    out = [None]*8
    for k in range(4):
        c,s = W(8,k)
        t = cmul_const(e, od[k], c, s)
        out[k]   = cadd(e, ev[k], t)
        out[k+4] = csub(e, ev[k], t)
    return out

def dft9(e, x):
    # CT 3x3: j = 3a+b ; X[3t+q] = DFT3_t( W9^{qb} DFT3_a(x[3a+b])[q] )
    sub = [dft3(e, [x[3*a+b] for a in range(3)]) for b in range(3)]   # sub[b][q]
    out = [None]*9
    for q in range(3):
        col = []
        for b in range(3):
            c,s = W(9, q*b)
            col.append(cmul_const(e, sub[b][q], c, s))
        res = dft3(e, col)
        for t in range(3):
            out[3*t+q] = res[t]
    return out

DFTS = {2:dft2, 3:dft3, 4:dft4, 5:dft5, 8:dft8, 9:dft9}

def pfa_maps(N1, N2):
    """PFA index maps: input j for (j1,j2); output k position for (k1,k2)"""
    N = N1*N2
    inm = [[ (N2*j1 + N1*j2) % N for j2 in range(N2)] for j1 in range(N1)]
    # output: k such that k mod N1 = k1, k mod N2 = k2  (CRT)
    outm = [[0]*N2 for _ in range(N1)]
    for k1 in range(N1):
        for k2 in range(N2):
            for k in range(N):
                if k % N1 == k1 and k % N2 == k2:
                    outm[k1][k2] = k; break
    return inm, outm
