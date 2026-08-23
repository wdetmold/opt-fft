# Small-DFT straight-line emitters producing C on __m512d vec-pairs.
import numpy as np
LD = np.longdouble
PI = np.longdouble('3.14159265358979323846264338327950288')

def hexd(x):
    return float(np.double(x)).hex()

class Emitter:
    def __init__(self):
        self.lines = []
        self.tmp = 0
        self.consts = {}
    def newv(self):
        self.tmp += 1
        return f"t{self.tmp}"
    def emit(self, s):
        self.lines.append("    " + s)
    def const(self, val):
        # returns C var name for set1 constant
        key = hexd(val)
        if key not in self.consts:
            name = f"k{len(self.consts)}"
            self.consts[key] = name
        return self.consts[key]
    def add(self, a, b):
        v = self.newv(); self.emit(f"__m512d {v} = _mm512_add_pd({a}, {b});"); return v
    def sub(self, a, b):
        v = self.newv(); self.emit(f"__m512d {v} = _mm512_sub_pd({a}, {b});"); return v
    def mul(self, a, b):
        v = self.newv(); self.emit(f"__m512d {v} = _mm512_mul_pd({a}, {b});"); return v
    def fmadd(self, a, b, c):
        v = self.newv(); self.emit(f"__m512d {v} = _mm512_fmadd_pd({a}, {b}, {c});"); return v
    def fnmadd(self, a, b, c):
        v = self.newv(); self.emit(f"__m512d {v} = _mm512_fnmadd_pd({a}, {b}, {c});"); return v
    def cdecls(self):
        return ["    const __m512d %s = _mm512_set1_pd(%s);" % (n, k) for k, n in self.consts.items()]

# complex value = (re_name, im_name)
def cadd(E, x, y): return (E.add(x[0], y[0]), E.add(x[1], y[1]))
def csub(E, x, y): return (E.sub(x[0], y[0]), E.sub(x[1], y[1]))
def rmul(E, x, c): # real constant * complex
    cn = E.const(c)
    return (E.mul(x[0], cn), E.mul(x[1], cn))
def cmul_i(E, x):  # multiply by +i: (re,im) -> (-im, re); represent by swap with sign later
    return ('NEG:'+x[1], x[0])  # caller must handle; avoid: use explicit ops below

def dft_ref(N):
    j = np.arange(N)
    ang = (-2*PI) * ((np.outer(j, j)) % N).astype(LD) / LD(N)
    return np.cos(ang) + 1j*np.sin(ang)

# Generic Winograd-free emitters for tiny N using symmetric folding (works for any N).
def emit_dft(E, N, xin, fwd=True):
    """xin: list of N complex pairs. returns list of N complex pairs (DFT)."""
    if N == 2:
        a = cadd(E, xin[0], xin[1]); b = csub(E, xin[0], xin[1])
        return [a, b]
    if N == 4:
        t0 = cadd(E, xin[0], xin[2]); t1 = csub(E, xin[0], xin[2])
        t2 = cadd(E, xin[1], xin[3]); t3 = csub(E, xin[1], xin[3])
        X0 = cadd(E, t0, t2); X2 = csub(E, t0, t2)
        # X1 = t1 - i t3 ; X3 = t1 + i t3   (fwd: w^1 = -i)
        X1 = (E.add(t1[0], t3[1]), E.sub(t1[1], t3[0]))
        X3 = (E.sub(t1[0], t3[1]), E.add(t1[1], t3[0]))
        return [X0, X1, X2, X3]
    # odd prime or odd N: symmetric fold (N odd)
    assert N % 2 == 1
    h = (N-1)//2
    cos = [np.cos(2*PI*n/LD(N)) for n in range(h+1)]
    sin = [np.sin(2*PI*n/LD(N)) for n in range(h+1)]
    a = []; b = []
    for j in range(1, h+1):
        a.append(cadd(E, xin[j], xin[N-j]))
        b.append(csub(E, xin[j], xin[N-j]))
    # X0
    s = xin[0]
    for j in range(h): s = cadd(E, s, a[j])
    out = [None]*N
    out[0] = s
    for k in range(1, h+1):
        # E_k = x0 + sum a_j cos(kj), O_k = sum b_j sin(kj)
        er, ei = xin[0]
        for j in range(1, h+1):
            n = (k*j) % N; n2 = min(n, N-n)
            cn = E.const(cos[n2])
            er = E.fmadd(cn, a[j-1][0], er)
            ei = E.fmadd(cn, a[j-1][1], ei)
        orr = None; oi = None
        for j in range(1, h+1):
            n = (k*j) % N
            sgnpos = (n <= h)
            sn = E.const(sin[min(n, N-n)])
            if orr is None:
                if sgnpos:
                    orr = E.mul(sn, b[j-1][0]); oi = E.mul(sn, b[j-1][1])
                else:
                    orr = E.mul(E.const(-sin[N-n]), b[j-1][0]); oi = E.mul(E.const(-sin[N-n]), b[j-1][1])
            else:
                if sgnpos:
                    orr = E.fmadd(sn, b[j-1][0], orr); oi = E.fmadd(sn, b[j-1][1], oi)
                else:
                    orr = E.fnmadd(sn, b[j-1][0], orr); oi = E.fnmadd(sn, b[j-1][1], oi)
        # X[k] = E - i O ; X[N-k] = E + i O
        out[k]   = (E.add(er, oi), E.sub(ei, orr))
        out[N-k] = (E.sub(er, oi), E.add(ei, orr))
    return out

def emit_dft8(E, xin):
    # split-radix-ish radix-2 DIT 8, standard
    # stage: 4 x DFT2 on (j, j+4)
    a = [cadd(E, xin[j], xin[j+4]) for j in range(4)]
    bb = [csub(E, xin[j], xin[j+4]) for j in range(4)]
    # twiddle b: w8^j for j=0..3: 1, (1-i)/r2? fwd w = e^{-2pi i/8}: w^1 = (1-i)/sqrt2... apply on b[j]
    r2 = np.cos(2*PI/LD(8))  # sqrt(2)/2
    # b1 * w8^1 = (b1r + b1i)/r2 ... compute: (br*c + bi*c, bi*c - br*c) with c = r2
    c = E.const(r2)
    b1 = (E.mul(E.add(bb[1][0], bb[1][1]), c), E.mul(E.sub(bb[1][1], bb[1][0]), c))
    b2 = (bb[2][1], 'NEG')  # b2 * w8^2 = b2 * (-i) = (b2i, -b2r)
    b2 = (bb[2][1], E.sub('_mm512_setzero_pd()', bb[2][0]) if False else None)
    # cleaner: negate via sub(0,x)
    zb = E.newv(); E.emit(f"__m512d {zb} = _mm512_setzero_pd();")
    b2 = (bb[2][1], E.sub(zb, bb[2][0]))
    b3 = (E.mul(E.sub(bb[3][1], bb[3][0]), c), E.mul(E.sub(zb, E.add(bb[3][0], bb[3][1])), c))
    # now DFT4 on a -> X[0,2,4,6]; DFT4 on (b0,b1,b2,b3) -> X[1,3,5,7]
    A = emit_dft(E, 4, a)
    Bv = emit_dft(E, 4, [bb[0], b1, b2, b3])
    out = [None]*8
    for k in range(4): out[2*k] = A[k]
    for k in range(4): out[2*k+1] = Bv[k]
    return out

def test_codelet(N, fn_emit):
    E = Emitter()
    xin = [(f"xr{j}", f"xi{j}") for j in range(N)]
    out = fn_emit(E, xin)
    body = []
    body.append(f"void testdft{N}(const double* in, double* outp){{")
    for j in range(N):
        body.append(f"    __m512d xr{j} = _mm512_load_pd(in+{j*16});")
        body.append(f"    __m512d xi{j} = _mm512_load_pd(in+{j*16+8});")
    body += E.cdecls()
    body += E.lines
    for k in range(N):
        body.append(f"    _mm512_store_pd(outp+{k*16}, {out[k][0]});")
        body.append(f"    _mm512_store_pd(outp+{k*16+8}, {out[k][1]});")
    body.append("}")
    return "\n".join(body)

if __name__ == "__main__":
    src = ["#include <immintrin.h>"]
    src.append(test_codelet(5, lambda E, x: emit_dft(E, 5, x)))
    src.append(test_codelet(9, lambda E, x: emit_dft(E, 9, x)))
    src.append(test_codelet(8, emit_dft8))
    src.append(test_codelet(3, lambda E, x: emit_dft(E, 3, x)))
    src.append(test_codelet(4, lambda E, x: emit_dft(E, 4, x)))
    open("tdft.c","w").write("\n".join(src))
    print("ok")
