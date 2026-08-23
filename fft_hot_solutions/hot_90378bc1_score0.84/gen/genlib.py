import numpy as np

def hexd(x):
    """double -> C hex literal, exact"""
    import struct
    b = struct.pack('<d', float(x))
    u = struct.unpack('<Q', b)[0]
    return f'__builtin_bit_cast_not'  # placeholder

def dbl(x):
    # print with enough digits to round-trip exactly
    return repr(float(x))

def chex(x):
    import struct
    u = struct.unpack('<Q', struct.pack('<d', float(x)))[0]
    # use hex float literal for exactness
    return float(x).hex()

class Emitter:
    def __init__(self):
        self.lines = []
        self.ind = 0
    def __call__(self, s=''):
        for ln in s.split('\n'):
            self.lines.append('    '*self.ind + ln)
    def out(self):
        return '\n'.join(self.lines) + '\n'

# ---------------- DFT codelets on named complex zmm variables ----------------
# Each codelet takes a prefix-name list of (re, im) C expr names for inputs and
# writes outputs to given names. Emits intrinsics; constants referenced as C
# identifiers that the caller must have defined (we collect them).

class Consts:
    def __init__(self):
        self.vals = {}   # name -> python float (value to splat)
    def get(self, name, val):
        if name in self.vals:
            assert self.vals[name] == val, (name, val, self.vals[name])
        else:
            self.vals[name] = val
        return name
    def decl(self, em):
        pass
    def loads(self):
        return '\n'.join(f'const __m512d {n} = _mm512_set1_pd({chex(v)});' for n, v in self.vals.items())

LD = np.longdouble

def tw(N, k):
    """exact-ish twiddle via long double"""
    ang = -2*np.pi*np.longdouble(k % N)/np.longdouble(N)
    # use higher precision pi
    PI = np.longdouble('3.14159265358979323846264338327950288')
    ang = -2*PI*np.longdouble(k % N)/np.longdouble(N)
    return float(np.cos(ang)), float(np.sin(ang))

_tmpctr = [0]
def T(em, expr, ty='__m512d'):
    n = f't{_tmpctr[0]}'; _tmpctr[0] += 1
    em(f'{ty} {n} = {expr};')
    return n

def ADD(em,a,b): return T(em, f'_mm512_add_pd({a},{b})')
def SUB(em,a,b): return T(em, f'_mm512_sub_pd({a},{b})')
def MUL(em,a,b): return T(em, f'_mm512_mul_pd({a},{b})')
def FMA(em,a,b,c): return T(em, f'_mm512_fmadd_pd({a},{b},{c})')
def FNMA(em,a,b,c): return T(em, f'_mm512_fnmadd_pd({a},{b},{c})')
def FMS(em,a,b,c): return T(em, f'_mm512_fmsub_pd({a},{b},{c})')

S3 = 0.86602540378443864676372317075293618347140262690519031402790348972596650845440002

def dft3(em, C, x):
    """x: list of 3 (re,im); returns 3 (re,im). 12 port05 ops."""
    (x0r,x0i),(x1r,x1i),(x2r,x2i) = x
    half = C.get('C_half', 0.5); s3 = C.get('C_s3', S3)
    tr = ADD(em,x1r,x2r); ti = ADD(em,x1i,x2i)
    X0r = ADD(em,x0r,tr); X0i = ADD(em,x0i,ti)
    mr = FNMA(em,half,tr,x0r); mi = FNMA(em,half,ti,x0i)
    dr = SUB(em,x1r,x2r); di = SUB(em,x1i,x2i)
    X1r = FMA(em,s3,di,mr); X1i = FNMA(em,s3,dr,mi)
    X2r = FNMA(em,s3,di,mr); X2i = FMA(em,s3,dr,mi)
    return [(X0r,X0i),(X1r,X1i),(X2r,X2i)]

def ctwid(em, C, xr, xi, wr_name, wi_name):
    """complex multiply by constant (wr,wi): 4 ops (2 mul + 2 fma)"""
    rr = MUL(em, wr_name, xr)
    ri = MUL(em, wr_name, xi)
    yr = FNMA(em, wi_name, xi, rr)
    yi = FMA(em, wi_name, xr, ri)
    return yr, yi

def dft9(em, C, x):
    """9-point complex DFT, CT 3x3. x list of 9 (re,im) in natural j order.
    Returns 9 (re,im) in natural k order. 88 port05 ops."""
    # stage 1: DFT3 over j1 for each j0: groups (j0, j0+3, j0+6)
    Y = {}
    for j0 in range(3):
        Y[j0] = dft3(em, C, [x[j0], x[j0+3], x[j0+6]])  # indexed by k0
    # twiddle: Y[j0][k0] *= W9^{j0*k0} for j0,k0 in {1,2}
    for j0 in (1,2):
        for k0 in (1,2):
            wr, wi = tw(9, j0*k0)
            nr = C.get(f'C_w9_{j0*k0}_r', wr); ni = C.get(f'C_w9_{j0*k0}_i', wi)
            yr, yi = Y[j0][k0]
            Y[j0][k0] = ctwid(em, C, yr, yi, nr, ni)
    # stage 2: for each k0: DFT3 over j0 -> k1
    X = [None]*9
    for k0 in range(3):
        Z = dft3(em, C, [Y[0][k0], Y[1][k0], Y[2][k0]])
        for k1 in range(3):
            X[k0 + 3*k1] = Z[k1]
    return X

C5_1 = tw(5,1); C5_2 = tw(5,2)

def dft5(em, C, x):
    """5-point complex DFT. 36 port05 ops (x0 counted). Natural k order out."""
    (x0r,x0i) = x[0]
    c1 = C.get('C_c51', C5_1[0]); c2 = C.get('C_c52', C5_2[0])
    s1 = C.get('C_s51', -C5_1[1]); s2 = C.get('C_s52', -C5_2[1])  # s_k = sin(2pi k/5) > 0
    u1r = ADD(em,x[1][0],x[4][0]); u1i = ADD(em,x[1][1],x[4][1])
    u2r = ADD(em,x[2][0],x[3][0]); u2i = ADD(em,x[2][1],x[3][1])
    v1r = SUB(em,x[1][0],x[4][0]); v1i = SUB(em,x[1][1],x[4][1])
    v2r = SUB(em,x[2][0],x[3][0]); v2i = SUB(em,x[2][1],x[3][1])
    sr = ADD(em,u1r,u2r); si = ADD(em,u1i,u2i)
    X0r = ADD(em,x0r,sr); X0i = ADD(em,x0i,si)
    # A = x0 + c1 u1 + c2 u2 ; B = x0 + c2 u1 + c1 u2
    Ar = FMA(em,c2,u2r,FMA(em,c1,u1r,x0r)); Ai = FMA(em,c2,u2i,FMA(em,c1,u1i,x0i))
    Br = FMA(em,c1,u2r,FMA(em,c2,u1r,x0r)); Bi = FMA(em,c1,u2i,FMA(em,c2,u1i,x0i))
    # X1 = A - i(s1 v1 + s2 v2) -> X1r = Ar + (s1 v1i + s2 v2i); X1i = Ai - (s1 v1r + s2 v2r)
    X1r = FMA(em,s2,v2i,FMA(em,s1,v1i,Ar)); X1i = FNMA(em,s2,v2r,FNMA(em,s1,v1r,Ai))
    X4r = FNMA(em,s2,v2i,FNMA(em,s1,v1i,Ar)); X4i = FMA(em,s2,v2r,FMA(em,s1,v1r,Ai))
    # X2 = B - i(s2 v1 - s1 v2) -> X2r = Br + s2 v1i - s1 v2i; X2i = Bi - s2 v1r + s1 v2r
    X2r = FNMA(em,s1,v2i,FMA(em,s2,v1i,Br)); X2i = FMA(em,s1,v2r,FNMA(em,s2,v1r,Bi))
    X3r = FMA(em,s1,v2i,FNMA(em,s2,v1i,Br)); X3i = FNMA(em,s1,v2r,FMA(em,s2,v1r,Bi))
    return [(X0r,X0i),(X1r,X1i),(X2r,X2i),(X3r,X3i),(X4r,X4i)]

def dft4(em, C, x):
    """4-point complex DFT, 16 ops."""
    t0r = ADD(em,x[0][0],x[2][0]); t0i = ADD(em,x[0][1],x[2][1])
    t1r = ADD(em,x[1][0],x[3][0]); t1i = ADD(em,x[1][1],x[3][1])
    d0r = SUB(em,x[0][0],x[2][0]); d0i = SUB(em,x[0][1],x[2][1])
    d1r = SUB(em,x[1][0],x[3][0]); d1i = SUB(em,x[1][1],x[3][1])
    X0r = ADD(em,t0r,t1r); X0i = ADD(em,t0i,t1i)
    X2r = SUB(em,t0r,t1r); X2i = SUB(em,t0i,t1i)
    # X1 = d0 - i d1 ; X3 = d0 + i d1
    X1r = ADD(em,d0r,d1i); X1i = SUB(em,d0i,d1r)
    X3r = SUB(em,d0r,d1i); X3i = ADD(em,d0i,d1r)
    return [(X0r,X0i),(X1r,X1i),(X2r,X2i),(X3r,X3i)]


def emit_map_fs(em, C, zr, zi, outr, outi):
    """z/(1+|z|), float-seeded (no microcoded ops). ~29 uops per 8 el."""
    half = C.get('C_half',0.5); c15 = C.get('C_15',1.5)
    one = C.get('C_one',1.0); two = C.get('C_two',2.0); tiny = C.get('C_tiny',1e-30)
    t  = T(em, f'_mm512_fmadd_pd({zi},{zi},_mm512_mul_pd({zr},{zr}))')
    t2 = T(em, f'_mm512_max_pd({t},{tiny})')
    tf = T(em, f'_mm512_cvtpd_ps({t2})', ty='__m256')
    pf = T(em, f'_mm256_rsqrt_ps({tf})', ty='__m256')
    p0 = T(em, f'_mm512_cvtps_pd({pf})')
    h  = T(em, f'_mm512_mul_pd({half},{t2})')
    q1 = T(em, f'_mm512_mul_pd({p0},{p0})')
    e1 = T(em, f'_mm512_fnmadd_pd({h},{q1},{c15})')
    p1 = T(em, f'_mm512_mul_pd({p0},{e1})')
    q2 = T(em, f'_mm512_mul_pd({p1},{p1})')
    e2 = T(em, f'_mm512_fnmadd_pd({h},{q2},{c15})')
    p2 = T(em, f'_mm512_mul_pd({p1},{e2})')
    r  = T(em, f'_mm512_mul_pd({t2},{p2})')
    hp = T(em, f'_mm512_mul_pd({half},{p2})')
    dd = T(em, f'_mm512_fnmadd_pd({r},{r},{t2})')
    r2 = T(em, f'_mm512_fmadd_pd({hp},{dd},{r})')
    d  = T(em, f'_mm512_add_pd({one},{r2})')
    df = T(em, f'_mm512_cvtpd_ps({d})', ty='__m256')
    wf = T(em, f'_mm256_rcp_ps({df})', ty='__m256')
    w0 = T(em, f'_mm512_cvtps_pd({wf})')
    e3 = T(em, f'_mm512_fnmadd_pd({d},{w0},{two})')
    w1 = T(em, f'_mm512_mul_pd({w0},{e3})')
    e4 = T(em, f'_mm512_fnmadd_pd({d},{w1},{two})')
    w2 = T(em, f'_mm512_mul_pd({w1},{e4})')
    e5 = T(em, f'_mm512_fnmadd_pd({d},{w2},{two})')
    w3 = T(em, f'_mm512_mul_pd({w2},{e5})')
    em(f'{outr} = _mm512_mul_pd({zr},{w3});')
    em(f'{outi} = _mm512_mul_pd({zi},{w3});')


BCASTV_DEF = """
static inline __m512d bcastv(const double* p){
  __m512d v;
  __asm__ volatile("vbroadcastsd %1, %0" : "=v"(v) : "m"(*p));
  return v;
}
"""
