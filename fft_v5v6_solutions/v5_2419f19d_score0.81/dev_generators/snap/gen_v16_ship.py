#!/usr/bin/env python3
"""v2: specialized AVX-512 batched 3D FFT + nonlinear map, DSB-friendly structure."""
import numpy as np

LD = np.longdouble
PI = np.longdouble('3.14159265358979323846264338327950288')

def wlit(x):
    d = float(x)
    return d.hex()

def tw(n, e):
    e = int(e) % n
    ang = (-2*PI) * LD(e) / LD(n)
    return float(np.cos(ang)), float(np.sin(ang))

class Ctx:
    def __init__(s):
        s.lines = []
        s.n = 0
        s.mapalt = 0
        s.pretables = []
    def t(s):
        s.n += 1
        return f"t{s.n}"
    def emit(s, l):
        s.lines.append(l)

class WZ:
    name='z4'; vt='__m512d'; K=4; masked=False
    def ld(s,p): return f"_mm512_loadu_pd({p})"
    def st(s,p,v): return f"_mm512_storeu_pd({p}, {v});"
    def add(s,a,b): return f"_mm512_add_pd({a}, {b})"
    def sub(s,a,b): return f"_mm512_sub_pd({a}, {b})"
    def mul(s,a,b): return f"_mm512_mul_pd({a}, {b})"
    def fma(s,a,b,c): return f"_mm512_fmadd_pd({a}, {b}, {c})"
    def fnma(s,a,b,c): return f"_mm512_fnmadd_pd({a}, {b}, {c})"
    def fmaddsub(s,a,b,c): return f"_mm512_fmaddsub_pd({a}, {b}, {c})"
    def swap(s,a): return f"_mm512_permute_pd({a}, 0x55)"
    def set1(s,x): return f"_mm512_set1_pd({x})"
    def xor(s,a,b): return f"_mm512_xor_pd({a}, {b})"
    def negs(s, even):
        if even: return "_mm512_set_pd(0.0,-0.0,0.0,-0.0,0.0,-0.0,0.0,-0.0)"
        else:    return "_mm512_set_pd(-0.0,0.0,-0.0,0.0,-0.0,0.0,-0.0,0.0)"
    def pmvec(s, v):
        return f"_mm512_set_pd({wlit(v)},-{wlit(v)},{wlit(v)},-{wlit(v)},{wlit(v)},-{wlit(v)},{wlit(v)},-{wlit(v)})"
    def mapsq(s): return "map_sd_z"
    def mapnr(s): return "map_nr_z"

class WZM(WZ):
    name='z4m'; masked=True
    def ld(s,p): return f"_mm512_maskz_loadu_pd(mk, {p})"
    def st(s,p,v): return f"_mm512_mask_storeu_pd({p}, mk, {v});"

class WS:
    name='s1'; vt='__m128d'; K=1; masked=False
    def ld(s,p): return f"_mm_loadu_pd({p})"
    def st(s,p,v): return f"_mm_storeu_pd({p}, {v});"
    def add(s,a,b): return f"_mm_add_pd({a}, {b})"
    def sub(s,a,b): return f"_mm_sub_pd({a}, {b})"
    def mul(s,a,b): return f"_mm_mul_pd({a}, {b})"
    def fma(s,a,b,c): return f"_mm_fmadd_pd({a}, {b}, {c})"
    def fnma(s,a,b,c): return f"_mm_fnmadd_pd({a}, {b}, {c})"
    def fmaddsub(s,a,b,c): return f"_mm_fmaddsub_pd({a}, {b}, {c})"
    def swap(s,a): return f"_mm_shuffle_pd({a}, {a}, 1)"
    def set1(s,x): return f"_mm_set1_pd({x})"
    def xor(s,a,b): return f"_mm_xor_pd({a}, {b})"
    def negs(s, even):
        if even: return "_mm_set_pd(0.0,-0.0)"
        else:    return "_mm_set_pd(-0.0,0.0)"
    def pmvec(s, v): return f"_mm_set_pd({wlit(v)},-{wlit(v)})"
    def mapsq(s): return "map_s1"
    def mapnr(s): return "map_s1"

def V(W,c,expr):
    t=c.t(); c.emit(f"{W.vt} {t} = {expr};"); return t

def MULI(W,c,a):
    return V(W,c, W.xor(W.swap(a), W.negs(True)))
def MULNI(W,c,a):
    return V(W,c, W.xor(W.swap(a), W.negs(False)))
def CMUL(W,c,a,wr,wi):
    if wi == 0.0:
        if wr == 1.0: return a
        if wr == -1.0: return V(W,c, W.xor(a, "NEGBOTH"))
        return V(W,c, W.mul(a, W.set1(wlit(wr))))
    if wr == 0.0:
        if wi == -1.0: return MULNI(W,c,a)
        if wi == 1.0:  return MULI(W,c,a)
    sw = V(W,c, W.swap(a))
    t  = V(W,c, W.mul(sw, W.set1(wlit(wi))))
    return V(W,c, W.fmaddsub(a, W.set1(wlit(wr)), t))

def fft2(W,c,x):
    return [V(W,c,W.add(x[0],x[1])), V(W,c,W.sub(x[0],x[1]))]

def fft3(W,c,x):
    s3 = float(np.sin(2*PI/LD(3)))
    t = V(W,c,W.add(x[1],x[2])); d = V(W,c,W.sub(x[1],x[2]))
    y0 = V(W,c,W.add(x[0],t))
    m = V(W,c,W.fma(W.set1(wlit(-0.5)), t, x[0]))
    sw = V(W,c,W.swap(d))
    e = V(W,c,W.mul(sw, W.pmvec(s3)))
    return [y0, V(W,c,W.sub(m,e)), V(W,c,W.add(m,e))]

def fft4(W,c,x):
    t0=V(W,c,W.add(x[0],x[2])); t2=V(W,c,W.sub(x[0],x[2]))
    t1=V(W,c,W.add(x[1],x[3])); t3=V(W,c,W.sub(x[1],x[3]))
    y0=V(W,c,W.add(t0,t1)); y2=V(W,c,W.sub(t0,t1))
    u=MULNI(W,c,t3)
    return [y0, V(W,c,W.add(t2,u)), y2, V(W,c,W.sub(t2,u))]

def fft5(W,c,x):
    c72,s72n = tw(5,1); c144,s144n = tw(5,2)
    s72 = -s72n; s144 = -s144n
    ta=V(W,c,W.add(x[1],x[4])); sa=V(W,c,W.sub(x[1],x[4]))
    tb=V(W,c,W.add(x[2],x[3])); sb=V(W,c,W.sub(x[2],x[3]))
    u=V(W,c,W.add(ta,tb)); y0=V(W,c,W.add(x[0],u))
    A1=V(W,c,W.fma(W.set1(wlit(c72)), ta, W.fma(W.set1(wlit(c144)), tb, x[0])))
    A2=V(W,c,W.fma(W.set1(wlit(c144)), ta, W.fma(W.set1(wlit(c72)), tb, x[0])))
    B1=V(W,c,W.fma(W.set1(wlit(s144)), sb, W.mul(W.set1(wlit(s72)), sa)))
    B2=V(W,c,W.fnma(W.set1(wlit(s72)), sb, W.mul(W.set1(wlit(s144)), sa)))
    iB1=MULI(W,c,B1); iB2=MULI(W,c,B2)
    return [y0, V(W,c,W.sub(A1,iB1)), V(W,c,W.sub(A2,iB2)),
            V(W,c,W.add(A2,iB2)), V(W,c,W.add(A1,iB1))]

def fft8(W,c,x):
    s = float(np.sqrt(LD(0.5)))
    t0=V(W,c,W.add(x[0],x[4])); t4=V(W,c,W.sub(x[0],x[4]))
    t1=V(W,c,W.add(x[1],x[5])); t5=V(W,c,W.sub(x[1],x[5]))
    t2=V(W,c,W.add(x[2],x[6])); t6=V(W,c,W.sub(x[2],x[6]))
    t3=V(W,c,W.add(x[3],x[7])); t7=V(W,c,W.sub(x[3],x[7]))
    u0=V(W,c,W.add(t0,t2)); u2=V(W,c,W.sub(t0,t2))
    u1=V(W,c,W.add(t1,t3)); u3=V(W,c,W.sub(t1,t3))
    y0=V(W,c,W.add(u0,u1)); y4=V(W,c,W.sub(u0,u1))
    v3=MULNI(W,c,u3)
    y2=V(W,c,W.add(u2,v3)); y6=V(W,c,W.sub(u2,v3))
    ni5=MULNI(W,c,t5)
    z1=V(W,c,W.mul(W.add(t5,ni5), W.set1(wlit(s))))
    z2=MULNI(W,c,t6)
    ni7=MULNI(W,c,t7)
    z3=V(W,c,W.mul(W.sub(ni7,t7), W.set1(wlit(s))))
    o=fft4(W,c,[t4,z1,z2,z3])
    return [y0,o[0],y2,o[1],y4,o[2],y6,o[3]]

def fft9(W,c,x):
    a = {}
    for j1 in range(3):
        y = fft3(W,c,[x[j1], x[j1+3], x[j1+6]])
        for k2 in range(3):
            e = (j1*k2) % 9
            wr,wi = tw(9,e)
            a[(j1,k2)] = CMUL(W,c,y[k2],wr,wi) if e else y[k2]
    out = [None]*9
    for k2 in range(3):
        y = fft3(W,c,[a[(0,k2)],a[(1,k2)],a[(2,k2)]])
        for k1 in range(3):
            out[3*k1+k2] = y[k1]
    return out

def fftN(W,c,x,n):
    return {2:fft2,3:fft3,4:fft4,5:fft5,8:fft8,9:fft9}[n](W,c,x)

def store_out(W,c,addr,val,fused,caddr):
    """fused: False | 'c' (add c then map)"""
    if not fused:
        c.emit(W.st(addr, val))
    else:
        z = V(W,c, W.add(val, W.ld(caddr)))
        f = W.mapsq() if (c.mapalt % 4 == 0) else W.mapnr()
        c.mapalt += 1
        c.emit(W.st(addr, f"{f}({z})"))

def mload(W,c,expr,do_map):
    v = V(W,c,expr)
    if not do_map:
        return v
    f = W.mapsq() if (c.mapalt % 4 == 0) else W.mapnr()
    c.mapalt += 1
    return V(W,c,f"{f}({v})")

# ---------------- kernel emitters v2 ----------------
def onestage_small(W,c,n,stride,fused,cb="cb",ml=False):
    xs = [mload(W,c, W.ld(f"p + {j*2*stride}"), ml) for j in range(n)]
    if n == 8:
        ys = fft8(W,c,xs); order = list(range(8))
        for k in order:
            store_out(W,c,f"p + {k*2*stride}", ys[k], fused, f"{cb} + {k*2*stride}")
    elif n == 6:
        pr=[];qr=[]
        for j1 in range(3):
            a=xs[(2*j1)%6]; b=xs[(2*j1+3)%6]
            pr.append(V(W,c,W.add(a,b))); qr.append(V(W,c,W.sub(a,b)))
        yp=fft3(W,c,pr); yq=fft3(W,c,qr)
        for k1 in range(3):
            k=(4*k1)%6
            store_out(W,c,f"p + {k*2*stride}", yp[k1], fused, f"{cb} + {k*2*stride}")
            k=(4*k1+3)%6
            store_out(W,c,f"p + {k*2*stride}", yq[k1], fused, f"{cb} + {k*2*stride}")
    else:
        raise ValueError(n)

def onestage_prime(W,c,p,stride,fused,cb="cb",ml=False):
    h=(p-1)//2
    x0=mload(W,c,W.ld(f"p + 0"), ml)
    A=[None]*(h+1); B=[None]*(h+1); su=None
    for j in range(1,h+1):
        a=mload(W,c,W.ld(f"p + {j*2*stride}"), ml)
        b=mload(W,c,W.ld(f"p + {(p-j)*2*stride}"), ml)
        u=V(W,c,W.add(a,b)); v=V(W,c,W.sub(a,b))
        su = u if su is None else V(W,c,W.add(su,u))
        for k in range(1,h+1):
            e=(k*j)%p
            ang=(-2*PI)*LD(e)/LD(p)
            ck=float(np.cos(ang)); sk=float(-np.sin(ang))
            if j==1:
                A[k]=V(W,c,W.fma(W.set1(wlit(ck)), u, x0))
                B[k]=V(W,c,W.mul(W.set1(wlit(sk)), v))
            else:
                A[k]=V(W,c,W.fma(W.set1(wlit(ck)), u, A[k]))
                B[k]=V(W,c,W.fma(W.set1(wlit(sk)), v, B[k]))
    y0=V(W,c,W.add(x0,su))
    store_out(W,c,"p + 0", y0, fused, f"{cb} + 0")
    for k in range(1,h+1):
        nib=MULNI(W,c,B[k])
        store_out(W,c,f"p + {k*2*stride}", V(W,c,W.add(A[k],nib)), fused, f"{cb} + {k*2*stride}")
        store_out(W,c,f"p + {(p-k)*2*stride}", V(W,c,W.sub(A[k],nib)), fused, f"{cb} + {(p-k)*2*stride}")


def prime_multi(W, c, p, stride, nch, choff):
    """loop-structured multi-chunk prime kernel: A/B split passes, coeff tables."""
    h=(p-1)//2
    # coefficient tables (emitted once per p; guard with tag)
    tag=f"P{p}"
    if tag not in EMITTED_TABLES:
        EMITTED_TABLES.add(tag)
        ck=[]; sk=[]
        for k in range(1,h+1):
            for j in range(1,h+1):
                e=(k*j)%p
                ang=(-2*PI)*LD(e)/LD(p)
                ck.append(wlit(float(np.cos(ang))))
                sk.append(wlit(float(-np.sin(ang))))
        c.pretables.append(f"static const double CK{p}[{h*h}] = {{{','.join(ck)}}};")
        c.pretables.append(f"static const double SK{p}[{h*h}] = {{{','.join(sk)}}};")
    c.emit(f"__attribute__((aligned(64))) double US[{nch*h*8}];")
    c.emit(f"__attribute__((aligned(64))) double VS[{nch*h*8}];")
    c.emit(f"__attribute__((aligned(64))) double AS[{nch*h*8}];")
    c.emit(f"__attribute__((aligned(64))) double BS[{nch*h*8}];")
    # pre-pass (loop over j)
    for ch in range(nch):
        c.emit(f"__m512d x0_{ch} = _mm512_loadu_pd(p + {ch*choff});")
        c.emit(f"__m512d su_{ch} = _mm512_setzero_pd();")
    c.emit(f"for (long j=1;j<={h};j++) {{")
    c.emit(f"  const double* pj = p + j*{2*stride};")
    c.emit(f"  const double* pq = p + ({p}-j)*{2*stride};")
    for ch in range(nch):
        c.emit(f"  {{ __m512d a=_mm512_loadu_pd(pj + {ch*choff}), b=_mm512_loadu_pd(pq + {ch*choff});")
        c.emit(f"    __m512d u=_mm512_add_pd(a,b), v=_mm512_sub_pd(a,b);")
        c.emit(f"    su_{ch}=_mm512_add_pd(su_{ch},u);")
        c.emit(f"    _mm512_store_pd(US + (j-1)*8 + {ch*h*8}, u);")
        c.emit(f"    _mm512_store_pd(VS + (j-1)*8 + {ch*h*8}, v); }}")
    c.emit(f"}}")
    kh1 = list(range(1,(h+1)//2+1)); kh2 = list(range((h+1)//2+1, h+1))
    khalves = [kh for kh in (kh1,kh2) if kh]
    for mat in ('A','B'):
        srcarr = 'US' if mat=='A' else 'VS'
        dstarr = 'AS' if mat=='A' else 'BS'
        tab = f"CK{p}" if mat=='A' else f"SK{p}"
        for kh in khalves:
            accd = []
            for k in kh:
                for ch in range(nch):
                    c.emit(f"__m512d acc_{mat}{k}_{ch} = _mm512_setzero_pd();")
            c.emit(f"for (long j=0;j<{h};j++) {{")
            for ch in range(nch):
                c.emit(f"  __m512d u{ch} = _mm512_load_pd({srcarr} + j*8 + {ch*h*8});")
            for k in kh:
                c.emit(f"  {{ __m512d cf = _mm512_set1_pd({tab}[{(k-1)}*{h}+j]);")
                for ch in range(nch):
                    c.emit(f"    acc_{mat}{k}_{ch} = _mm512_fmadd_pd(cf, u{ch}, acc_{mat}{k}_{ch});")
                c.emit(f"  }}")
            c.emit(f"}}")
            for k in kh:
                for ch in range(nch):
                    c.emit(f"_mm512_store_pd({dstarr} + {(k-1)*8} + {ch*h*8}, acc_{mat}{k}_{ch});")
    # epilogue loop over k
    for ch in range(nch):
        c.emit(W.st(f"p + {ch*choff}", f"_mm512_add_pd(x0_{ch}, su_{ch})"))
    c.emit(f"{{ double* pk = p + {2*stride}; double* pq = p + {(p-1)*2*stride};")
    c.emit(f"  for (long k=0;k<{h};k++) {{")
    for ch in range(nch):
        c.emit(f"    {{ __m512d Av=_mm512_load_pd(AS + k*8 + {ch*h*8});")
        c.emit(f"      __m512d Bv=_mm512_load_pd(BS + k*8 + {ch*h*8});")
        c.emit(f"      __m512d nib=_mm512_xor_pd(_mm512_permute_pd(Bv,0x55), _mm512_set_pd(-0.0,0.0,-0.0,0.0,-0.0,0.0,-0.0,0.0));")
        c.emit(f"      __m512d t=_mm512_add_pd(x0_{ch}, Av);")
        c.emit(f"      _mm512_storeu_pd(pk + {ch*choff}, _mm512_add_pd(t,nib));")
        c.emit(f"      _mm512_storeu_pd(pq + {ch*choff}, _mm512_sub_pd(t,nib)); }}")
    c.emit(f"    pk += {2*stride}; pq -= {2*stride};")
    c.emit(f"  }}")
    c.emit(f"}}")

EMITTED_TABLES=set()


def map_stream(W, c, L, stride, pm="pm", cm="cm"):
    """emit map for one chunk (L outputs at pm + k*2*stride); batched reciprocal per 8"""
    ONE='_mm512_set1_pd(1.0)'; HALF='_mm512_set1_pd(0.5)'; THR='_mm512_set1_pd(1.5)'; TINY='_mm512_set1_pd(2.5e-301)'
    def emit_d(z):
        if (c.mapalt % 4) == 0:
            zz=V(W,c,f"_mm512_mul_pd({z},{z})")
            s2=V(W,c,f"_mm512_add_pd({zz},_mm512_permute_pd({zz},0x55))")
            d=V(W,c,f"_mm512_add_pd(_mm512_sqrt_pd({s2}),{ONE})")
        else:
            zz=V(W,c,f"_mm512_fmadd_pd({z},{z},{TINY})")
            s2=V(W,c,f"_mm512_add_pd({zz},_mm512_permute_pd({zz},0x55))")
            y=V(W,c,f"_mm512_rsqrt14_pd({s2})")
            h=V(W,c,f"_mm512_mul_pd({s2},{HALF})")
            yy=V(W,c,f"_mm512_mul_pd({y},{y})")
            y=V(W,c,f"_mm512_mul_pd({y},_mm512_fnmadd_pd({h},{yy},{THR}))")
            yy=V(W,c,f"_mm512_mul_pd({y},{y})")
            y=V(W,c,f"_mm512_mul_pd({y},_mm512_fnmadd_pd({h},{yy},{THR}))")
            d=V(W,c,f"_mm512_fmadd_pd({s2},{y},{ONE})")
        c.mapalt += 1
        return d
    ks = list(range(L))
    gi = 0
    while gi < L:
        G = min(8, L - gi)
        if G < 8 and L >= 8:
            for k in range(gi, L):
                z = V(W,c, W.add(W.ld(f"{pm} + {k*2*stride}"), W.ld(f"{cm} + {k*2*stride}")))
                f = W.mapsq() if (c.mapalt % 4 == 0) else W.mapnr()
                c.mapalt += 1
                c.emit(W.st(f"{pm} + {k*2*stride}", f"{f}({z})"))
            break
        zb=[]; db=[]
        for t in range(G):
            k=gi+t
            zb.append(V(W,c, W.add(W.ld(f"{pm} + {k*2*stride}"), W.ld(f"{cm} + {k*2*stride}"))))
        for t in range(G):
            db.append(emit_d(zb[t]))
        p=[db[0]]
        for t in range(1,G):
            p.append(V(W,c,f"_mm512_mul_pd({p[t-1]},{db[t]})"))
        R=V(W,c,f"_mm512_div_pd({ONE},{p[G-1]})")
        wb=[None]*G
        for t in range(G-1,0,-1):
            wb[t]=V(W,c,f"_mm512_mul_pd({R},{p[t-1]})")
            if t>1:
                R=V(W,c,f"_mm512_mul_pd({R},{db[t]})")
        if G>1:
            R=V(W,c,f"_mm512_mul_pd({R},{db[1]})")
        wb[0]=R
        for t in range(G):
            k=gi+t
            c.emit(W.st(f"{pm} + {k*2*stride}", f"_mm512_mul_pd({zb[t]},{wb[t]})"))
        gi += G

def zip_lines(a, b):
    """interleave two statement lists proportionally, preserving each order"""
    out=[]; na,nb=len(a),len(b)
    ia=ib=0
    tot=na+nb
    for t in range(tot):
        # choose stream to keep ratio
        if ia*nb <= ib*na and ia<na:
            out.append(a[ia]); ia+=1
        elif ib<nb:
            out.append(b[ib]); ib+=1
        else:
            out.append(a[ia]); ia+=1
    return out

# PFA config: (n1 outer, n2 inner)
PFA = {36:(4,9), 45:(9,5)}
def pfa_tables(n):
    n1,n2 = PFA[n]
    jin = [[(n2*j1 + n1*j2) % n for j2 in range(n2)] for j1 in range(n1)]
    inv1 = pow(n2,-1,n1); inv2 = pow(n1,-1,n2)
    e1 = n2*inv1; e2 = n1*inv2
    kout = [[(e1*k1 + e2*k2) % n for k1 in range(n1)] for k2 in range(n2)]
    return jin, kout

def twostage_pfa(W,c,n,stride,fused,cb="cb",ml=False):
    """fully unrolled PFA kernel (constant offsets, SSA)"""
    n1,n2 = PFA[n]
    jin, kout = pfa_tables(n)
    S2 = {}
    for j1 in range(n1):
        xs=[mload(W,c,W.ld(f"p + {jin[j1][j2]*2*stride}"), ml) for j2 in range(n2)]
        ys=fftN(W,c,xs,n2)
        for k2 in range(n2):
            S2[(j1,k2)] = ys[k2]
    for k2 in range(n2):
        xs=[S2[(j1,k2)] for j1 in range(n1)]
        ys=fftN(W,c,xs,n1)
        for k1 in range(n1):
            k = kout[k2][k1]
            store_out(W,c,f"p + {k*2*stride}", ys[k1], fused, f"{cb} + {k*2*stride}")

def twostage_ct64(W,c,stride,fused,cb="cb",ml=False):
    """loop-structured 8x8 CT for 64 with twiddle table TW64 (runtime-indexed)."""
    c.emit(f"{W.vt} S2[64];")
    c.emit(f"for (int j1=0;j1<8;j1++) {{")
    sub=Ctx(); sub.n=c.n; sub.mapalt=c.mapalt
    xs=[mload(W,sub,W.ld(f"p + (j1 + {8*j2})*{2*stride}"), ml) for j2 in range(8)]
    ys=fft8(W,sub,xs)
    sub.emit(f"S2[j1] = {ys[0]};")
    for k2 in range(1,8):
        v=ys[k2]
        sw=V(W,sub,W.swap(v))
        t=V(W,sub,W.mul(sw, W.set1(f"TW64[j1][{k2}][1]")))
        r=V(W,sub,W.fmaddsub(v, W.set1(f"TW64[j1][{k2}][0]"), t))
        sub.emit(f"S2[{k2*8}+j1] = {r};")
    c.n=sub.n; c.mapalt=sub.mapalt
    for l in sub.lines: c.emit("  "+l)
    c.emit("}")
    c.emit(f"for (int k2=0;k2<8;k2++) {{")
    sub=Ctx(); sub.n=c.n; sub.mapalt=c.mapalt
    xs=[f"S2[k2*8+{j1}]" for j1 in range(8)]
    ys=fft8(W,sub,xs)
    for k1 in range(8):
        store_out(W,sub,f"p + (k2 + {8*k1})*{2*stride}", ys[k1], fused, f"{cb} + (k2 + {8*k1})*{2*stride}")
    c.n=sub.n; c.mapalt=sub.mapalt
    for l in sub.lines: c.emit("  "+l)
    c.emit("}")

KCFG = { 6:'small', 8:'small', 13:'prime', 17:'prime', 23:'prime',
         36:'pfa', 45:'pfa', 64:'ct64' }

def gen_kernel(L, variant, nch=1):
    kind = KCFG[L]
    NI = "static inline" if L in (6,8) else "NOINL static"
    fused = False
    if variant == 'pxm':
        W=WZ(); stride=L*L
        c1=Ctx()
        if kind=='prime': onestage_prime(W,c1,L,stride,False)
        elif kind=='small': onestage_small(W,c1,L,stride,False)
        elif kind=='pfa': twostage_pfa(W,c1,L,stride,False)
        else: raise ValueError
        c2=Ctx(); c2.n = 100000
        map_stream(W,c2,L,stride)
        body="\n    ".join(zip_lines(c1.lines, c2.lines))
        pre = "\n".join(c1.pretables)
        if pre: pre += "\n"
        return f"{pre}NOINL static void f{L}_pxm(double* restrict p, double* restrict pm, const double* restrict cm) {{\n    {body}\n}}\n"
    if variant in ('mq','mq_t'):
        W = WZM() if variant=='mq_t' else WZ()
        stride=L*L
        c1=Ctx()
        map_stream(W,c1,L,stride,pm="pm",cm="cm")
        body="\n    ".join(c1.lines)
        m = ", __mmask8 mk" if W.masked else ""
        return f"NOINL static void f{L}_{variant}(double* restrict pm, const double* restrict cm{m}) {{\n    {body}\n}}\n"
    if variant=='u1':
        W=WS(); stride=1
        sig=f"{NI} void f{L}_u1(double* restrict p)"
    elif variant.startswith('u'):
        W=WZ(); stride=4
        suf = str(nch) if nch>1 else ''
        sig=f"{NI} void f{L}_u{suf}(double* restrict p)"
    else:
        W = WZM() if variant.endswith('_t') else WZ()
        stride = L*L if variant.startswith('px') else L
        m = ", __mmask8 mk" if W.masked else ""
        suf = (str(nch) if nch>1 else '') + ('_t' if W.masked else '')
        base = 'px' if variant.startswith('px') else 'py'
        sig=f"{NI} void f{L}_{base}{suf}(double* restrict p{m})"
    c=Ctx()
    if kind=='prime' and nch>1:
        choff = L*8 if variant.startswith('u') else 8
        prime_multi(W,c,L,stride,nch,choff)
    elif kind=='small': onestage_small(W,c,L,stride,fused)
    elif kind=='prime': onestage_prime(W,c,L,stride,fused)
    elif kind=='pfa': twostage_pfa(W,c,L,stride,fused)
    else: twostage_ct64(W,c,stride,fused)
    body="\n    ".join(c.lines)
    pre = "\n".join(c.pretables)
    if pre: pre += "\n"
    return f"{pre}{sig} {{\n    {body}\n}}\n"

# ---------------- 64-specific kernels ----------------
def gen_f64_z(fused=False):
    W=WZ(); c=Ctx()
    # stage A: two lane-groups g: lanes j1=4g+l; vertical fft8 over j2
    AR = {}
    for g in range(2):
        xs=[V(W,c,W.ld(f"p + {(4*g + 8*j2)*2}")) for j2 in range(8)]
        ys=fft8(W,c,xs)
        for k2 in range(8):
            v=ys[k2]
            if not (g==0 and k2==0):
                sw=V(W,c,W.swap(v))
                t=V(W,c,W.mul(sw, f"_mm512_load_pd(T64ZI[{g}][{k2}])"))
                v=V(W,c,W.fmaddsub(ys[k2], f"_mm512_load_pd(T64ZR[{g}][{k2}])", t))
            AR[(g,k2)]=v
    # transpose blocks (g,h): A[g][4h+t] -> B[h][4g+t]
    BR={}
    for h in range(2):
        for g in range(2):
            a,b,d,e = (AR[(g,4*h+t)] for t in range(4))
            r0=V(W,c,f"_mm512_shuffle_f64x2({a}, {b}, 0x88)")
            r1=V(W,c,f"_mm512_shuffle_f64x2({a}, {b}, 0xDD)")
            r2=V(W,c,f"_mm512_shuffle_f64x2({d}, {e}, 0x88)")
            r3=V(W,c,f"_mm512_shuffle_f64x2({d}, {e}, 0xDD)")
            BR[(h,4*g+0)]=V(W,c,f"_mm512_shuffle_f64x2({r0}, {r2}, 0x88)")
            BR[(h,4*g+1)]=V(W,c,f"_mm512_shuffle_f64x2({r1}, {r3}, 0x88)")
            BR[(h,4*g+2)]=V(W,c,f"_mm512_shuffle_f64x2({r0}, {r2}, 0xDD)")
            BR[(h,4*g+3)]=V(W,c,f"_mm512_shuffle_f64x2({r1}, {r3}, 0xDD)")
    for h in range(2):
        xs=[BR[(h,j1)] for j1 in range(8)]
        ys=fft8(W,c,xs)
        for k1 in range(8):
            if fused:
                z=V(W,c,W.add(ys[k1], W.ld(f"cl + {(8*k1+4*h)*2}")))
                f = W.mapsq() if (c.mapalt % 4 == 0) else W.mapnr()
                c.mapalt += 1
                c.emit(W.st(f"p + {(8*k1+4*h)*2}", f"{f}({z})"))
            else:
                c.emit(W.st(f"p + {(8*k1+4*h)*2}", ys[k1]))
    body="\n    ".join(c.lines)
    if fused:
        return f"NOINL static void f64_zf(double* restrict p, const double* restrict cl) {{\n    {body}\n}}\n"
    return f"NOINL static void f64_z(double* restrict p) {{\n    {body}\n}}\n"

# ---------------- drivers ----------------
def gen_driver(L):
    L2=L*L; L3=L*L*L
    G=L//4; REM=L%4
    FCX=L2//4; REMX=L2%4
    FCY=L//4; REMY=L%4
    MKX=(1<<(2*REMX))-1
    MKY=(1<<(2*REMY))-1
    PRIME = (KCFG[L]=='prime')
    PR = PRIME and L in MULTI_PRIMES
    lines=[]
    a=lines.append
    def decomp(n):
        n4,r = divmod(n,16); n2,r = divmod(r,8); n1,r = divmod(r,4)
        return n4,n2,n1,r
    a(f"void run{L}(const double* restrict x0, const double* restrict c, double* restrict one, double* restrict fin, long B, long m) {{")
    a(f"  _mm_setcsr(_mm_getcsr() | 0x8040);")
    a(f"  const size_t vol={L3}, bytes=(size_t){L3}*16;")
    a(f"  if (B<=0) return;")
    a(f"  if (m<=0) {{ memcpy(one,x0,B*bytes); memcpy(fin,x0,B*bytes); return; }}")
    a(f"  const size_t abytes = (bytes+63)&~(size_t)63;")
    if L>=64:
        a(f"  size_t wgot=0; double* wv = (double*)big_alloc(abytes, &wgot);")
    else:
        a(f"  double* wv = (double*)aligned_alloc(64, abytes);")
    if L==64:
        a(f"  size_t cgot=0; double* cx = (double*)big_alloc(abytes, &cgot);")

    a(f"  for (long b=0;b<B;b++) {{")
    a(f"    const double* cv = c + b*vol*2;")

    if L==64:
        a(f"    for (long q=0;q<64;q++) memcpy(cx + q*8192, cv + SW8[q]*8192, 65536);")
    a(f"    memcpy(wv, x0 + b*vol*2, bytes);")
    a(f"    for (long s=0;s<m;s++) {{")
    if L==64:
        a(f"      for (long q=0;q<64;q++) {{ double* pl = wv + q*{2*L2}; for (long y=0;y<64;y++) f64_z(pl + y*128); for (long z0=0;z0<64;z0+=4) f64_py(pl + z0*2); }}")
        a(f"      if ((s&1)==0) {{ f64_xA(wv, 1, 8); f64_xBu(wv, cx, 8, 1); }}")
        a(f"      else          {{ f64_xA(wv, 8, 1); f64_xBu(wv, cx, 1, 8); }}")
        a(f"      if (s==0) {{ for (long x=0;x<64;x++) memcpy(one + b*vol*2 + x*{2*L2}, wv + SW8[x]*{2*L2}, {L2*16}); }}")
        a(f"    }}")
        a(f"    if (m&1) {{ for (long x=0;x<64;x++) memcpy(fin + b*vol*2 + x*{2*L2}, wv + SW8[x]*{2*L2}, {L2*16}); }}")
        a(f"    else memcpy(fin + b*vol*2, wv, bytes);")
    else:
        # ---- pass_y per plane ----
        a(f"      for (long x=0;x<{L};x++) {{")
        a(f"        double* pl = wv + x*{2*L2};")
        if PR:
            n4,n2,n1,r = decomp(L)
            off=0
            if n4==1: a(f"        f{L}_py4(pl);"); off=16
            elif n4>1: a(f"        for (long z0=0;z0<{16*n4};z0+=16) f{L}_py4(pl + z0*2);"); off=16*n4
            for _ in range(n2): a(f"        f{L}_py2(pl + {off*2});"); off+=8
            for _ in range(n1): a(f"        f{L}_py(pl + {off*2});"); off+=4
            if r: a(f"        f{L}_py_t(pl + {off*2}, (__mmask8){(1<<(2*r))-1});")
        else:
            if L==45:
                a(f"        if (s) mapc(pl, cv + x*{2*L2}, {2*L2});")
            a(f"        for (long z0=0;z0<{4*FCY};z0+=4) f{L}_py(pl + z0*2);")
            if REMY:
                a(f"        f{L}_py_t(pl + {4*FCY*2}, (__mmask8){MKY});")
        a(f"      }}")
        # ---- pass_z: groups of lines over whole volume ----
        NL_ = L2
        if PR:
            zn4,zn2,zn1,zr = decomp(NL_)
        else:
            zn4,zn2,zn1,zr = 0,0,NL_//4,NL_%4
        a(f"      {{")
        a(f"        __attribute__((aligned(64))) double S[{(4 if zn4 else (2 if zn2 else 1))*L*8}];")
        def emit_group(base_expr, nch):
            sfx = {1:'',2:'2',4:'4'}[nch]
            for ch in range(nch):
                a(f"          {{ double* lb = {base_expr} + {ch*8*L};")
                a(f"            tin4(lb, lb+{2*L}, lb+{4*L}, lb+{6*L}, S+{ch*L*8}, {L//4});")
                for j in range(4*(L//4), L):
                    a(f"            _mm512_store_pd(S+{ch*L*8+j*8}, ld4l(lb, lb+{2*L}, lb+{4*L}, lb+{6*L}, {j}));")
                a(f"          }}")
            a(f"          f{L}_u{sfx}(S);")
            for ch in range(nch):
                a(f"          {{ double* lb = {base_expr} + {ch*8*L};")
                a(f"            tout4(lb, lb+{2*L}, lb+{4*L}, lb+{6*L}, S+{ch*L*8}, {L//4});")
                for j in range(4*(L//4), L):
                    a(f"            st4l(lb, lb+{2*L}, lb+{4*L}, lb+{6*L}, {j}, _mm512_load_pd(S+{ch*L*8+j*8}));")
                a(f"          }}")
        off=0
        if zn4:
            a(f"        for (long g=0;g<{zn4};g++) {{")
            a(f"          double* gl = wv + g*{32*L};")
            emit_group("gl", 4)
            a(f"        }}")
            off = zn4*16
        for _ in range(zn2):
            a(f"        {{ double* gl = wv + {off*2*L};")
            emit_group("gl", 2)
            a(f"        }}")
            off += 8
        if zn1:
            if zn1>1:
                a(f"        for (long g=0;g<{zn1};g++) {{")
                a(f"          double* gl = wv + {off*2*L} + g*{8*L};")
                emit_group("gl", 1)
                a(f"        }}")
            else:
                a(f"        {{ double* gl = wv + {off*2*L};")
                emit_group("gl", 1)
                a(f"        }}")
            off += zn1*4
        for t in range(zr):
            a(f"        f{L}_u1(wv + {(off+t)*2*L});")
        a(f"      }}")
        # ---- pass_x (+ zipped map for primes/small; pfa defers map) ----
        if PRIME or KCFG[L]=='small':
            QU = L2//4; xr = L2%4
            a(f"      f{L}_px(wv);")
            a(f"      for (long col=4;col<{4*QU};col+=4) f{L}_pxm(wv + col*2, wv + col*2 - 8, cv + col*2 - 8);")
            a(f"      f{L}_mq(wv + {(4*QU-4)*2}, cv + {(4*QU-4)*2});")
            if xr:
                a(f"      f{L}_px_t(wv + {4*QU*2}, (__mmask8){(1<<(2*xr))-1});")
                a(f"      f{L}_mq_t(wv + {4*QU*2}, cv + {4*QU*2}, (__mmask8){(1<<(2*xr))-1});")
        else:
            a(f"      for (long col=0;col<{4*FCX};col+=4) f{L}_px(wv + col*2);")
            if REMX:
                a(f"      f{L}_px_t(wv + {4*FCX*2}, (__mmask8){MKX});")
            if L!=45:
                a(f"      mapc(wv, cv, vol*2);")
        if L==45:
            a(f"      if (s==0) mapcopy_c(one + b*vol*2, wv, cv, vol*2);")
            a(f"    }}")
            a(f"    mapcopy_c(fin + b*vol*2, wv, cv, vol*2);")
        else:
            a(f"      if (s==0) memcpy(one + b*vol*2, wv, bytes);")
            a(f"    }}")
            a(f"    memcpy(fin + b*vol*2, wv, bytes);")
    a(f"  }}")
    if L>=64:
        a(f"  big_free(wv, wgot);")
    else:
        a(f"  free(wv);")
    if L==64:
        a(f"  big_free(cx, cgot);")
    a(f"}}")
    return "\n".join(lines)+"\n"

PRELUDE_TOP = r'''
// ============================================================================
// Batched iterated 3D complex FFT + nonlinear map  z/(1+|z|)  for fixed cube
// sizes L in {6,8,13,17,23,36,45,64}.  Auto-generated by gen.py.
//
// All transform arithmetic is our own (no FFT libraries):
//   * L=6,8:      fully inlined PFA(3x2) / radix-2 8-point kernels
//   * L=13,17,23: symmetric real-coefficient direct DFT (cos/sin half-matrix
//                 dot products); size 23 also has 4-column GEMM-style kernels
//   * L=36:       prime-factor algorithm 4x9 (fft4 (x) fft9), twiddle-free
//   * L=45:       prime-factor algorithm 9x5 (fft9 (x) fft5), twiddle-free
//   * L=64:       8x8 Cooley-Tukey; x-axis as two in-place half-sweeps over
//                 8-plane groups (digit-swapped layout alternates per step);
//                 z-axis via an in-register lane-transposed kernel
// Twiddles/coefficients are compile-time literals rounded from long double.
// The nonlinear map is evaluated in fp64 via sqrt+div / rsqrt14+Newton hybrid
// (~2-3 ulp), software-pipelined into the final FFT pass of each step.
// Single-threaded AVX-512. Each batch volume is iterated to completion while
// cache-resident; one/final outputs are copied out at step 1 and step m.
// ============================================================================
#include <immintrin.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#define NOINL __attribute__((noinline))
#include <sys/mman.h>
static void* big_alloc(size_t bytes, size_t* got){
  size_t sz = (bytes + (2UL<<20) - 1) & ~((2UL<<20)-1);
  void* p = mmap(NULL, sz, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
  if (p==MAP_FAILED){ *got=0; return aligned_alloc(64, (bytes+63)&~(size_t)63); }
  madvise(p, sz, MADV_HUGEPAGE);
  *got=sz; return p;
}
static void big_free(void* p, size_t got){
  if (got) munmap(p, got); else free(p);
}

static inline __m512d map_sq_z(__m512d z){
  __m512d zz=_mm512_mul_pd(z,z);
  __m512d s2=_mm512_add_pd(zz,_mm512_permute_pd(zz,0x55));
  __m512d r=_mm512_sqrt_pd(s2);
  __m512d d=_mm512_add_pd(r,_mm512_set1_pd(1.0));
  __m512d w=_mm512_rcp14_pd(d);
  __m512d two=_mm512_set1_pd(2.0);
  w=_mm512_mul_pd(w,_mm512_fnmadd_pd(d,w,two));
  w=_mm512_mul_pd(w,_mm512_fnmadd_pd(d,w,two));
  return _mm512_mul_pd(z,w);
}
static inline __m512d map_nr_z(__m512d z){
  __m512d zz=_mm512_fmadd_pd(z,z,_mm512_set1_pd(2.5e-301));
  __m512d s2=_mm512_add_pd(zz,_mm512_permute_pd(zz,0x55));
  __m512d y=_mm512_rsqrt14_pd(s2);
  __m512d h=_mm512_mul_pd(s2,_mm512_set1_pd(0.5));
  __m512d thr=_mm512_set1_pd(1.5);
  __m512d yy=_mm512_mul_pd(y,y);
  y=_mm512_mul_pd(y,_mm512_fnmadd_pd(h,yy,thr));
  yy=_mm512_mul_pd(y,y);
  y=_mm512_mul_pd(y,_mm512_fnmadd_pd(h,yy,thr));
  __m512d r=_mm512_mul_pd(s2,y);
  __m512d d=_mm512_add_pd(r,_mm512_set1_pd(1.0));
  __m512d w=_mm512_rcp14_pd(d);
  __m512d two=_mm512_set1_pd(2.0);
  w=_mm512_mul_pd(w,_mm512_fnmadd_pd(d,w,two));
  w=_mm512_mul_pd(w,_mm512_fnmadd_pd(d,w,two));
  return _mm512_mul_pd(z,w);
}
static inline __m512d map_sd_z(__m512d z){
  __m512d zz=_mm512_mul_pd(z,z);
  __m512d s2=_mm512_add_pd(zz,_mm512_permute_pd(zz,0x55));
  __m512d r=_mm512_sqrt_pd(s2);
  __m512d d=_mm512_add_pd(r,_mm512_set1_pd(1.0));
  return _mm512_div_pd(z,d);
}
#define NEGBOTH _mm512_set1_pd(-0.0)
#define MAP_D8(K) { __m512d zz,s2; \
    if (((K)&3)==0){ zz=_mm512_mul_pd(zb[K],zb[K]); s2=_mm512_add_pd(zz,_mm512_permute_pd(zz,0x55)); db[K]=_mm512_add_pd(_mm512_sqrt_pd(s2),vone); } \
    else { zz=_mm512_fmadd_pd(zb[K],zb[K],vtiny); s2=_mm512_add_pd(zz,_mm512_permute_pd(zz,0x55)); \
      __m512d y=_mm512_rsqrt14_pd(s2); __m512d h=_mm512_mul_pd(s2,vhalf); \
      __m512d yy=_mm512_mul_pd(y,y); y=_mm512_mul_pd(y,_mm512_fnmadd_pd(h,yy,vthr)); \
      yy=_mm512_mul_pd(y,y); y=_mm512_mul_pd(y,_mm512_fnmadd_pd(h,yy,vthr)); \
      db[K]=_mm512_fmadd_pd(s2,y,vone); } }
#define MAP_BATCH8 { \
    __m512d p1=_mm512_mul_pd(db[0],db[1]); \
    __m512d p2=_mm512_mul_pd(p1,db[2]); \
    __m512d p3=_mm512_mul_pd(p2,db[3]); \
    __m512d p4=_mm512_mul_pd(p3,db[4]); \
    __m512d p5=_mm512_mul_pd(p4,db[5]); \
    __m512d p6=_mm512_mul_pd(p5,db[6]); \
    __m512d p7=_mm512_mul_pd(p6,db[7]); \
    __m512d R=_mm512_div_pd(vone, p7); \
    wb[7]=_mm512_mul_pd(R,p6);  R=_mm512_mul_pd(R,db[7]); \
    wb[6]=_mm512_mul_pd(R,p5);  R=_mm512_mul_pd(R,db[6]); \
    wb[5]=_mm512_mul_pd(R,p4);  R=_mm512_mul_pd(R,db[5]); \
    wb[4]=_mm512_mul_pd(R,p3);  R=_mm512_mul_pd(R,db[4]); \
    wb[3]=_mm512_mul_pd(R,p2);  R=_mm512_mul_pd(R,db[3]); \
    wb[2]=_mm512_mul_pd(R,p1);  R=_mm512_mul_pd(R,db[2]); \
    wb[1]=_mm512_mul_pd(R,db[0]); \
    wb[0]=_mm512_mul_pd(R,db[1]); }
#define MAP_CONSTS __m512d vone=_mm512_set1_pd(1.0), vhalf=_mm512_set1_pd(0.5), vthr=_mm512_set1_pd(1.5), vtiny=_mm512_set1_pd(2.5e-301);
NOINL static void mapc(double* restrict v, const double* restrict cv, long nd){
  MAP_CONSTS
  long i=0;
  for (; i+64<=nd; i+=64){
    __m512d zb[8], db[8], wb[8];
    for (int k=0;k<8;k++) zb[k]=_mm512_add_pd(_mm512_loadu_pd(v+i+8*k), _mm512_loadu_pd(cv+i+8*k));
    MAP_D8(0) MAP_D8(1) MAP_D8(2) MAP_D8(3) MAP_D8(4) MAP_D8(5) MAP_D8(6) MAP_D8(7)
    MAP_BATCH8
    for (int k=0;k<8;k++) _mm512_storeu_pd(v+i+8*k, _mm512_mul_pd(zb[k],wb[k]));
  }
  for (; i+8<=nd; i+=8)
    { __m512d zb1 = _mm512_add_pd(_mm512_loadu_pd(v+i), _mm512_loadu_pd(cv+i));
      _mm512_storeu_pd(v+i, map_nr_z(zb1)); }
  if (i<nd){
    __mmask8 mk = (__mmask8)((1u<<(nd-i))-1);
    __m512d zb1 = _mm512_add_pd(_mm512_maskz_loadu_pd(mk, v+i), _mm512_maskz_loadu_pd(mk, cv+i));
    _mm512_mask_storeu_pd(v+i, mk, map_nr_z(zb1));
  }
}

NOINL static void mapcopy_c(double* restrict dst, const double* restrict srcp, const double* restrict cv, long nd){
  MAP_CONSTS
  long i=0;
  for (; i+64<=nd; i+=64){
    __m512d zb[8], db[8], wb[8];
    for (int k=0;k<8;k++) zb[k]=_mm512_add_pd(_mm512_loadu_pd(srcp+i+8*k), _mm512_loadu_pd(cv+i+8*k));
    MAP_D8(0) MAP_D8(1) MAP_D8(2) MAP_D8(3) MAP_D8(4) MAP_D8(5) MAP_D8(6) MAP_D8(7)
    MAP_BATCH8
    for (int k=0;k<8;k++) _mm512_storeu_pd(dst+i+8*k, _mm512_mul_pd(zb[k],wb[k]));
  }
  for (; i+8<=nd; i+=8)
    _mm512_storeu_pd(dst+i, map_nr_z(_mm512_add_pd(_mm512_loadu_pd(srcp+i), _mm512_loadu_pd(cv+i))));
  if (i<nd){
    __mmask8 mk = (__mmask8)((1u<<(nd-i))-1);
    __m512d z = _mm512_add_pd(_mm512_maskz_loadu_pd(mk, srcp+i), _mm512_maskz_loadu_pd(mk, cv+i));
    _mm512_mask_storeu_pd(dst+i, mk, map_nr_z(z));
  }
}
NOINL static void mapcopy(double* restrict dst, const double* restrict srcp, long nd){
  // nd doubles (2*ncomplex); applies x = z/(1+|z|)
  long i=0;
  for (; i+32<=nd; i+=32){
    _mm512_storeu_pd(dst+i,    map_sd_z(_mm512_loadu_pd(srcp+i)));
    _mm512_storeu_pd(dst+i+8,  map_nr_z(_mm512_loadu_pd(srcp+i+8)));
    _mm512_storeu_pd(dst+i+16, map_nr_z(_mm512_loadu_pd(srcp+i+16)));
    _mm512_storeu_pd(dst+i+24, map_nr_z(_mm512_loadu_pd(srcp+i+24)));
  }
  for (; i+8<=nd; i+=8)
    _mm512_storeu_pd(dst+i, map_nr_z(_mm512_loadu_pd(srcp+i)));
  if (i<nd){
    __mmask8 mk = (__mmask8)((1u<<(nd-i))-1);
    __m512d z = _mm512_maskz_loadu_pd(mk, srcp+i);
    _mm512_mask_storeu_pd(dst+i, mk, map_nr_z(z));
  }
}

static inline void tin4(const double* l0,const double* l1,const double* l2,const double* l3,double* S,long nb){
  for (long jb=0;jb<nb;jb++){
    __m512d a=_mm512_loadu_pd(l0+8*jb), b=_mm512_loadu_pd(l1+8*jb);
    __m512d c=_mm512_loadu_pd(l2+8*jb), d=_mm512_loadu_pd(l3+8*jb);
    __m512d r0=_mm512_shuffle_f64x2(a,b,0x88), r1=_mm512_shuffle_f64x2(a,b,0xDD);
    __m512d r2=_mm512_shuffle_f64x2(c,d,0x88), r3=_mm512_shuffle_f64x2(c,d,0xDD);
    _mm512_store_pd(S+32*jb+0,  _mm512_shuffle_f64x2(r0,r2,0x88));
    _mm512_store_pd(S+32*jb+8,  _mm512_shuffle_f64x2(r1,r3,0x88));
    _mm512_store_pd(S+32*jb+16, _mm512_shuffle_f64x2(r0,r2,0xDD));
    _mm512_store_pd(S+32*jb+24, _mm512_shuffle_f64x2(r1,r3,0xDD));
  }
}
static inline void tout4(double* l0,double* l1,double* l2,double* l3,const double* S,long nb){
  for (long jb=0;jb<nb;jb++){
    __m512d a=_mm512_load_pd(S+32*jb+0), b=_mm512_load_pd(S+32*jb+8);
    __m512d c=_mm512_load_pd(S+32*jb+16), d=_mm512_load_pd(S+32*jb+24);
    __m512d r0=_mm512_shuffle_f64x2(a,b,0x88), r1=_mm512_shuffle_f64x2(a,b,0xDD);
    __m512d r2=_mm512_shuffle_f64x2(c,d,0x88), r3=_mm512_shuffle_f64x2(c,d,0xDD);
    _mm512_storeu_pd(l0+8*jb, _mm512_shuffle_f64x2(r0,r2,0x88));
    _mm512_storeu_pd(l1+8*jb, _mm512_shuffle_f64x2(r1,r3,0x88));
    _mm512_storeu_pd(l2+8*jb, _mm512_shuffle_f64x2(r0,r2,0xDD));
    _mm512_storeu_pd(l3+8*jb, _mm512_shuffle_f64x2(r1,r3,0xDD));
  }
}
static inline __m512d ld4l(const double* l0,const double* l1,const double* l2,const double* l3,long j){
  __m256d lo=_mm256_insertf128_pd(_mm256_castpd128_pd256(_mm_loadu_pd(l0+2*j)),_mm_loadu_pd(l1+2*j),1);
  __m256d hi=_mm256_insertf128_pd(_mm256_castpd128_pd256(_mm_loadu_pd(l2+2*j)),_mm_loadu_pd(l3+2*j),1);
  return _mm512_insertf64x4(_mm512_castpd256_pd512(lo),hi,1);
}
static inline void st4l(double* l0,double* l1,double* l2,double* l3,long j,__m512d v){
  __m256d lo=_mm512_castpd512_pd256(v), hi=_mm512_extractf64x4_pd(v,1);
  _mm_storeu_pd(l0+2*j,_mm256_castpd256_pd128(lo));
  _mm_storeu_pd(l1+2*j,_mm256_extractf128_pd(lo,1));
  _mm_storeu_pd(l2+2*j,_mm256_castpd256_pd128(hi));
  _mm_storeu_pd(l3+2*j,_mm256_extractf128_pd(hi,1));
}
static inline void fft8q(__m512d* x){
  __m512d NI=_mm512_set_pd(-0.0,0.0,-0.0,0.0,-0.0,0.0,-0.0,0.0);
  __m512d s=_mm512_set1_pd(0x1.6a09e667f3bcdp-1);
  __m512d t0=_mm512_add_pd(x[0],x[4]), t4=_mm512_sub_pd(x[0],x[4]);
  __m512d t1=_mm512_add_pd(x[1],x[5]), t5=_mm512_sub_pd(x[1],x[5]);
  __m512d t2=_mm512_add_pd(x[2],x[6]), t6=_mm512_sub_pd(x[2],x[6]);
  __m512d t3=_mm512_add_pd(x[3],x[7]), t7=_mm512_sub_pd(x[3],x[7]);
  __m512d u0=_mm512_add_pd(t0,t2), u2=_mm512_sub_pd(t0,t2);
  __m512d u1=_mm512_add_pd(t1,t3), u3=_mm512_sub_pd(t1,t3);
  __m512d y0=_mm512_add_pd(u0,u1), y4=_mm512_sub_pd(u0,u1);
  __m512d v3=_mm512_xor_pd(_mm512_permute_pd(u3,0x55),NI);
  __m512d y2=_mm512_add_pd(u2,v3), y6=_mm512_sub_pd(u2,v3);
  __m512d ni5=_mm512_xor_pd(_mm512_permute_pd(t5,0x55),NI);
  __m512d z1=_mm512_mul_pd(_mm512_add_pd(t5,ni5), s);
  __m512d z2=_mm512_xor_pd(_mm512_permute_pd(t6,0x55),NI);
  __m512d ni7=_mm512_xor_pd(_mm512_permute_pd(t7,0x55),NI);
  __m512d z3=_mm512_mul_pd(_mm512_sub_pd(ni7,t7), s);
  __m512d w0=_mm512_add_pd(t4,z2), w2=_mm512_sub_pd(t4,z2);
  __m512d w1=_mm512_add_pd(z1,z3), w3=_mm512_sub_pd(z1,z3);
  __m512d y1=_mm512_add_pd(w0,w1), y5=_mm512_sub_pd(w0,w1);
  __m512d q=_mm512_xor_pd(_mm512_permute_pd(w3,0x55),NI);
  __m512d y3=_mm512_add_pd(w2,q), y7=_mm512_sub_pd(w2,q);
  x[0]=y0;x[1]=y1;x[2]=y2;x[3]=y3;x[4]=y4;x[5]=y5;x[6]=y6;x[7]=y7;
}
'''

PRELUDE_64 = r'''
NOINL static void f64_xA(double* restrict v, long bm, long st){
  const long S1 = st*8192;
  for (long g=0; g<8; g++){
    double* b = v + g*bm*8192;
    for (long col=0; col<8192; col+=8){
      __m512d x[8];
      for (int j=0;j<8;j++) x[j]=_mm512_load_pd(b + j*S1 + col);
      fft8q(x);
      _mm512_store_pd(b + col, x[0]);
      for (int k2=1;k2<8;k2++){
        __m512d wr=_mm512_set1_pd(TW64[g][k2][0]), wi=_mm512_set1_pd(TW64[g][k2][1]);
        __m512d t=x[k2];
        _mm512_store_pd(b + k2*S1 + col, _mm512_fmaddsub_pd(t, wr, _mm512_mul_pd(_mm512_permute_pd(t,0x55), wi)));
      }
    }
  }
}
NOINL static void f64_xB_sw(double* restrict v, const double* restrict cc, long bm, long st){
  const long S1 = st*8192;
  for (long g=0; g<8; g++){
    double* b = v + g*bm*8192;
    const double* cg = cc + g*bm*8192;
    for (long col=0; col<8192; col+=8){
      __m512d x[8];
      for (int j=0;j<8;j++) x[j]=_mm512_load_pd(b + j*S1 + col);
      fft8q(x);
      for (int k1=0;k1<8;k1++){
        __m512d z=_mm512_add_pd(x[k1], _mm512_load_pd(cg + k1*S1 + col));
        z = (k1&3) ? map_nr_z(z) : map_sd_z(z);
        _mm512_store_pd(b + k1*S1 + col, z);
      }
    }
  }
}
NOINL static void f64_xBp(double* restrict v, long bm, long st){
  const long S1 = st*8192;
  for (long g=0; g<8; g++){
    double* b = v + g*bm*8192;
    for (long col=0; col<8192; col+=8){
      __m512d x[8];
      for (int j=0;j<8;j++) x[j]=_mm512_load_pd(b + j*S1 + col);
      fft8q(x);
      for (int k1=0;k1<8;k1++) _mm512_store_pd(b + k1*S1 + col, x[k1]);
    }
  }
}
NOINL static void f64_xBu(double* restrict v, const double* restrict cc, long bm, long st){
  MAP_CONSTS
  const long S1 = st*8192;
  for (long g=0; g<8; g++){
    double* b = v + g*bm*8192;
    const double* cg = cc + g*8*8192;
    for (long col=0; col<8192; col+=8){
      __m512d x[8], zb[8], db[8], wb[8];
      for (int j=0;j<8;j++) x[j]=_mm512_load_pd(b + j*S1 + col);
      fft8q(x);
      for (int k=0;k<8;k++) zb[k]=_mm512_add_pd(x[k], _mm512_load_pd(cg + k*8192 + col));
      MAP_D8(0) MAP_D8(1) MAP_D8(2) MAP_D8(3) MAP_D8(4) MAP_D8(5) MAP_D8(6) MAP_D8(7)
      MAP_BATCH8
      for (int k=0;k<8;k++) _mm512_store_pd(b + k*S1 + col, _mm512_mul_pd(zb[k],wb[k]));
    }
  }
}
'''

def gen_tables():
    out=[]
    # TW64[j1][k2]
    rows=[]
    for a_ in range(8):
        cols=[]
        for b_ in range(8):
            wr,wi = tw(64, a_*b_)
            cols.append(f"{{{wlit(wr)},{wlit(wi)}}}")
        rows.append("{"+",".join(cols)+"}")
    out.append("static const double TW64[8][8][2] = {"+",".join(rows)+"};")
    # T64Z lane-varying twiddles: lanes l=0..3 (complex), duplicated per double lane
    tr=[];ti=[]
    for g in range(2):
        rr=[];ii=[]
        for k2 in range(8):
            re=[];im=[]
            for l in range(4):
                wr,wi = tw(64,(4*g+l)*k2)
                re += [wlit(wr)]*2
                im += [wlit(wi)]*2
            rr.append("{"+",".join(re)+"}")
            ii.append("{"+",".join(im)+"}")
        tr.append("{"+",".join(rr)+"}")
        ti.append("{"+",".join(ii)+"}")
    out.append("static const double __attribute__((aligned(64))) T64ZR[2][8][8] = {"+",".join(tr)+"};")
    out.append("static const double __attribute__((aligned(64))) T64ZI[2][8][8] = {"+",".join(ti)+"};")
    sw=[str(8*(x%8)+x//8) for x in range(64)]
    out.append("static const int SW8[64] = {"+",".join(sw)+"};")
    return "\n".join(out)+"\n"

MULTI_PRIMES = {23}   # tuned: which primes use multi-chunk kernels

def main():
    SIZES=(6,8,13,17,23,36,45,64)
    out=[PRELUDE_TOP, gen_tables(), PRELUDE_64]
    for L in SIZES:
        if L==64:
            out.append(gen_kernel(64,'py'))
            out.append(gen_f64_z())
            out.append(gen_f64_z(fused=True))
        elif KCFG[L]=='prime':
            for v in ('py','u'):
                if L in MULTI_PRIMES:
                    out.append(gen_kernel(L,v,4))
                    out.append(gen_kernel(L,v,2))
                out.append(gen_kernel(L,v,1))
            out.append(gen_kernel(L,'px'))
            out.append(gen_kernel(L,'pxm'))
            out.append(gen_kernel(L,'mq'))
            out.append(gen_kernel(L,'mq_t'))
            out.append(gen_kernel(L,'px_t'))
            out.append(gen_kernel(L,'py_t'))
            out.append(gen_kernel(L,'u1'))
        else:
            out.append(gen_kernel(L,'px'))
            if KCFG[L]=='small':
                out.append(gen_kernel(L,'pxm'))
                out.append(gen_kernel(L,'mq'))
            if (L*L)%4:
                out.append(gen_kernel(L,'px_t'))
            out.append(gen_kernel(L,'py'))
            if L%4:
                out.append(gen_kernel(L,'py_t'))
            out.append(gen_kernel(L,'u'))
            if L%4: out.append(gen_kernel(L,'u1'))
        out.append(gen_driver(L))
    src="\n".join(out)
    with open("implementation.c","w") as f:
        f.write(src)
    print("wrote implementation.c:", len(src), "bytes")

if __name__=="__main__":
    main()
