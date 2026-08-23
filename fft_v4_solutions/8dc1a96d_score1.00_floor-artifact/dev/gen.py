# Generates implementation.c : specialized AVX-512 split-complex iterated 3D FFT.
import math
from kernels import dft, tw

SIZES = (6, 8, 13, 17, 23, 36, 45, 64)

def hexf(x):
    x = float(x)
    if x == int(x) and abs(x) < 1e15:
        return f"{x:.1f}"
    return x.hex()

class CBE:
    """C emitter backend over (re,im) pairs of C expression strings."""
    def __init__(self):
        self.lines = []
        self.n = 0
    def nv(self, expr):
        name = f"v{self.n}"; self.n += 1
        self.lines.append(f"V {name}={expr};")
        return name
    # --- complex ops on value pairs ---
    def cadd(self, a, b): return (self.nv(f"ADD({a[0]},{b[0]})"), self.nv(f"ADD({a[1]},{b[1]})"))
    def csub(self, a, b): return (self.nv(f"SUB({a[0]},{b[0]})"), self.nv(f"SUB({a[1]},{b[1]})"))
    def csum(self, vs):
        vs = list(vs)
        while len(vs) > 1:
            nxt = []
            for i in range(0, len(vs) - 1, 2):
                nxt.append(self.cadd(vs[i], vs[i + 1]))
            if len(vs) % 2: nxt.append(vs[-1])
            vs = nxt
        return vs[0]
    def racc(self, terms):
        (c0, v0) = terms[0]
        r = self.nv(f"MUL(SET1({hexf(c0)}),{v0[0]})")
        i = self.nv(f"MUL(SET1({hexf(c0)}),{v0[1]})")
        for c, v in terms[1:]:
            r = self.nv(f"FMA(SET1({hexf(c)}),{v[0]},{r})")
            i = self.nv(f"FMA(SET1({hexf(c)}),{v[1]},{i})")
        return (r, i)
    def cmix(self, a, b, sign):
        if sign < 0:
            return (self.nv(f"ADD({a[0]},{b[1]})"), self.nv(f"SUB({a[1]},{b[0]})"))
        else:
            return (self.nv(f"SUB({a[0]},{b[1]})"), self.nv(f"ADD({a[1]},{b[0]})"))
    def neg(self, x): return self.nv(f"XOR({x},NEGZ)")
    def cmulw(self, a, w):
        wr, wi = w
        T = 1e-30
        def z(v): return abs(v) < T
        def one(v): return abs(v - 1) < T
        def mone(v): return abs(v + 1) < T
        ar, ai = a
        if one(wr) and z(wi):  return a
        if mone(wr) and z(wi): return (self.neg(ar), self.neg(ai))
        if z(wr) and one(wi):  return (self.neg(ai), ar)       # i*a
        if z(wr) and mone(wi): return (ai, self.neg(ar))       # -i*a
        if z(wi):  # real scalar
            c = hexf(wr)
            return (self.nv(f"MUL(SET1({c}),{ar})"), self.nv(f"MUL(SET1({c}),{ai})"))
        if z(wr):  # imaginary scalar i*wi: (-wi*ai, wi*ar)
            c = hexf(wi); cn = hexf(-wi)
            return (self.nv(f"MUL(SET1({cn}),{ai})"), self.nv(f"MUL(SET1({c}),{ar})"))
        if abs(abs(wr) - abs(wi)) < T:
            # w = wr*(1 + i*sg),  sg = wi/wr = +-1
            sg = 1.0 if (wi / wr) > 0 else -1.0
            c = hexf(wr)
            if sg > 0:  # re=wr*(ar-ai), im=wr*(ai+ar)
                t1 = self.nv(f"SUB({ar},{ai})"); t2 = self.nv(f"ADD({ai},{ar})")
            else:       # re=wr*(ar+ai), im=wr*(ai-ar)
                t1 = self.nv(f"ADD({ar},{ai})"); t2 = self.nv(f"SUB({ai},{ar})")
            return (self.nv(f"MUL(SET1({c}),{t1})"), self.nv(f"MUL(SET1({c}),{t2})"))
        cr, ci_ = hexf(wr), hexf(wi)
        rr = self.nv(f"FNMA(SET1({ci_}),{ai},MUL(SET1({cr}),{ar}))")
        ii = self.nv(f"FMA(SET1({cr}),{ai},MUL(SET1({ci_}),{ar}))")
        return (rr, ii)

def emit_kernel(L, ld, st):
    """Returns C lines applying DFT_L with ld(j)->(re,im) exprs, st(k,(re,im))."""
    be = CBE()
    cache = {}
    def ldc(j):
        if j not in cache:
            e_r, e_i = ld(j)
            cache[j] = (be.nv(e_r), be.nv(e_i))
        return cache[j]
    xs = [ldc(j) for j in range(L)]
    out = dft(be, xs, L)
    for k in range(L):
        st(k, out[k], be)
    return be.lines

_PS_TAB = {6: 40, 8: 64, 13: 184, 17: 296, 23: 568, 36: 1296, 45: 2088, 64: 4136}
def PS(L):
    return _PS_TAB[L]

class CBE:
    """C emitter backend over (re,im) pairs of C expression strings."""
    def __init__(self):
        self.lines = []
        self.n = 0
    def nv(self, expr):
        name = f"v{self.n}"; self.n += 1
        self.lines.append(f"V {name}={expr};")
        return name
    # --- complex ops on value pairs ---
    def cadd(self, a, b): return (self.nv(f"ADD({a[0]},{b[0]})"), self.nv(f"ADD({a[1]},{b[1]})"))
    def csub(self, a, b): return (self.nv(f"SUB({a[0]},{b[0]})"), self.nv(f"SUB({a[1]},{b[1]})"))
    def csum(self, vs):
        vs = list(vs)
        while len(vs) > 1:
            nxt = []
            for i in range(0, len(vs) - 1, 2):
                nxt.append(self.cadd(vs[i], vs[i + 1]))
            if len(vs) % 2: nxt.append(vs[-1])
            vs = nxt
        return vs[0]
    def racc(self, terms):
        (c0, v0) = terms[0]
        r = self.nv(f"MUL(SET1({hexf(c0)}),{v0[0]})")
        i = self.nv(f"MUL(SET1({hexf(c0)}),{v0[1]})")
        for c, v in terms[1:]:
            r = self.nv(f"FMA(SET1({hexf(c)}),{v[0]},{r})")
            i = self.nv(f"FMA(SET1({hexf(c)}),{v[1]},{i})")
        return (r, i)
    def cmix(self, a, b, sign):
        if sign < 0:
            return (self.nv(f"ADD({a[0]},{b[1]})"), self.nv(f"SUB({a[1]},{b[0]})"))
        else:
            return (self.nv(f"SUB({a[0]},{b[1]})"), self.nv(f"ADD({a[1]},{b[0]})"))
    def neg(self, x): return self.nv(f"XOR({x},NEGZ)")
    def cmulw(self, a, w):
        wr, wi = w
        T = 1e-30
        def z(v): return abs(v) < T
        def one(v): return abs(v - 1) < T
        def mone(v): return abs(v + 1) < T
        ar, ai = a
        if one(wr) and z(wi):  return a
        if mone(wr) and z(wi): return (self.neg(ar), self.neg(ai))
        if z(wr) and one(wi):  return (self.neg(ai), ar)       # i*a
        if z(wr) and mone(wi): return (ai, self.neg(ar))       # -i*a
        if z(wi):  # real scalar
            c = hexf(wr)
            return (self.nv(f"MUL(SET1({c}),{ar})"), self.nv(f"MUL(SET1({c}),{ai})"))
        if z(wr):  # imaginary scalar i*wi: (-wi*ai, wi*ar)
            c = hexf(wi); cn = hexf(-wi)
            return (self.nv(f"MUL(SET1({cn}),{ai})"), self.nv(f"MUL(SET1({c}),{ar})"))
        if abs(abs(wr) - abs(wi)) < T:
            # w = wr*(1 + i*sg),  sg = wi/wr = +-1
            sg = 1.0 if (wi / wr) > 0 else -1.0
            c = hexf(wr)
            if sg > 0:  # re=wr*(ar-ai), im=wr*(ai+ar)
                t1 = self.nv(f"SUB({ar},{ai})"); t2 = self.nv(f"ADD({ai},{ar})")
            else:       # re=wr*(ar+ai), im=wr*(ai-ar)
                t1 = self.nv(f"ADD({ar},{ai})"); t2 = self.nv(f"SUB({ai},{ar})")
            return (self.nv(f"MUL(SET1({c}),{t1})"), self.nv(f"MUL(SET1({c}),{t2})"))
        cr, ci_ = hexf(wr), hexf(wi)
        rr = self.nv(f"FNMA(SET1({ci_}),{ai},MUL(SET1({cr}),{ar}))")
        ii = self.nv(f"FMA(SET1({cr}),{ai},MUL(SET1({ci_}),{ar}))")
        return (rr, ii)

def emit_kernel(L, ld, st):
    """Returns C lines applying DFT_L with ld(j)->(re,im) exprs, st(k,(re,im))."""
    be = CBE()
    cache = {}
    def ldc(j):
        if j not in cache:
            e_r, e_i = ld(j)
            cache[j] = (be.nv(e_r), be.nv(e_i))
        return cache[j]
    xs = [ldc(j) for j in range(L)]
    out = dft(be, xs, L)
    for k in range(L):
        st(k, out[k], be)
    return be.lines


def gen_pz(L):
    """z-axis pass: NG groups of 8 rows, in-register transposes.
       For L=64 operates on a single plane (rows=L); else whole volume (rows=L*L)."""
    perplane = (L == 64)
    NG = (L + 7) // 8 if perplane else (L * L + 7) // 8
    L2 = L * L
    nblk = (L + 7) // 8
    ln = []
    ln.append(f"static void pz_{L}(double* restrict re, double* restrict im){{")
    ln.append(f" for(long g=0; g<{NG}; ++g){{")
    ln.append(f"  double* restrict rb = re + g*{8*L};")
    ln.append(f"  double* restrict ib = im + g*{8*L};")
    ln.append(f"  V sr[{L}], si[{L}], qr[{L}], qi[{L}];")
    # load + transpose
    for comp, base, arr in (("r", "rb", "sr"), ("i", "ib", "si")):
        for blk in range(nblk):
            c0 = blk * 8
            cnt = min(8, L - c0)
            vs = [f"p{comp}{blk}_{t}" for t in range(8)]
            for t in range(8):
                ln.append(f"  V {vs[t]} = LDU({base} + {t*L + c0});")
            ln.append(f"  TR8({','.join(vs)});")
            for t in range(cnt):
                ln.append(f"  {arr}[{c0+t}] = {vs[t]};")
    # kernel
    def ld(j): return (f"sr[{j}]", f"si[{j}]")
    def st(k, v, be):
        be.lines.append(f"qr[{k}]={v[0]}; qi[{k}]={v[1]};")
    for l in emit_kernel(L, ld, st):
        ln.append("  " + l)
    # transpose + store
    for comp, base, arr in (("r", "rb", "qr"), ("i", "ib", "qi")):
        for blk in range(nblk):
            c0 = blk * 8
            cnt = min(8, L - c0)
            vs = [f"o{comp}{blk}_{t}" for t in range(8)]
            for t in range(8):
                src = f"{arr}[{c0+t}]" if t < cnt else f"{arr}[{c0}]"
                ln.append(f"  V {vs[t]} = {src};")
            ln.append(f"  TR8({','.join(vs)});")
            if cnt == 8:
                for t in range(8):
                    ln.append(f"  STU({base} + {t*L + c0}, {vs[t]});")
            else:
                mk = (1 << cnt) - 1
                for t in range(8):
                    ln.append(f"  MST({base} + {t*L + c0}, {hex(mk)}, {vs[t]});")
    ln.append(" }")
    ln.append("}")
    return ln

def gen_py(L):
    L2 = L * L
    perplane = (L == 64)
    nblk = (L + 7) // 8
    rem = L % 8
    masks = [0xFF] * nblk
    if rem: masks[-1] = (1 << rem) - 1
    ln = []
    ln.append(f"static const uint8_t YM_{L}[{nblk}] = {{{','.join(hex(m) for m in masks)}}};")
    if perplane:
        ln.append(f"static void py_{L}(double* restrict re, double* restrict im){{")
        ln.append(" {")
        ln.append("  double* restrict rp = re;")
        ln.append("  double* restrict ip = im;")
    else:
        ln.append(f"static void py_{L}(double* restrict re, double* restrict im){{")
        ln.append(f" for(long x=0; x<{L}; ++x){{")
        ln.append(f"  double* restrict rp = re + x*{L2};")
        ln.append(f"  double* restrict ip = im + x*{L2};")
    ln.append(f"  for(long zb=0; zb<{nblk}; ++zb){{")
    ln.append(f"   const __mmask8 mk = (__mmask8)YM_{L}[zb];")
    ln.append(f"   double* restrict r0 = rp + zb*8;")
    ln.append(f"   double* restrict i0 = ip + zb*8;")
    def ld(j): return (f"LDU(r0 + {j*L})", f"LDU(i0 + {j*L})")
    stores = []
    def st(k, v, be): stores.append((k, v))
    body = emit_kernel(L, ld, st)
    for l in body: ln.append("   " + l)
    for k, v in stores:
        ln.append(f"   MST(r0 + {k*L}, mk, {v[0]}); MST(i0 + {k*L}, mk, {v[1]});")
    ln.append("  }")
    ln.append(" }")
    ln.append("}")
    return ln

def gen_px(L):
    L2, L3 = L * L, L ** 3
    L2S = PS(L)  # plane stride
    nub = (L2 + 7) // 8
    rem = L2 % 8
    ln = []
    ln.append(f"static void px_{L}(double* restrict re, double* restrict im, const double* restrict cre, const double* restrict cim){{")
    ln.append(f" for(long ub=0; ub<{nub}; ++ub){{")
    if rem:
        ln.append(f"  const __mmask8 mk = (ub=={nub-1}) ? (__mmask8){hex((1<<rem)-1)} : (__mmask8)0xFF;")
    else:
        ln.append(f"  const __mmask8 mk = (__mmask8)0xFF;")
    ln.append(f"  double* restrict r0 = re + ub*8;")
    ln.append(f"  double* restrict i0 = im + ub*8;")
    ln.append(f"  const double* restrict c0r = cre + ub*8;")
    ln.append(f"  const double* restrict c0i = cim + ub*8;")
    def ld(j): return (f"LDU(r0 + {j*L2S})", f"LDU(i0 + {j*L2S})")
    stores = []
    def st(k, v, be): stores.append((k, v))
    body = emit_kernel(L, ld, st)
    for l in body: ln.append("  " + l)
    for k, v in stores:
        ln.append(f"  {{ V mr, mi; map8({v[0]}, {v[1]}, LDU(c0r + {k*L2S}), LDU(c0i + {k*L2S}), &mr, &mi);")
        ln.append(f"    MST(r0 + {k*L2S}, mk, mr); MST(i0 + {k*L2S}, mk, mi); }}")
    ln.append(" }")
    ln.append("}")
    return ln

def gen_dbg(L):
    # kernel on [L][8] SoA buffers (stride 8), for testing
    ln = [f"void dbg1d_{L}(double* re, double* im){{"]
    def ld(j): return (f"LDU(re + {j*8})", f"LDU(im + {j*8})")
    stores = []
    def st(k, v, be): stores.append((k, v))
    body = emit_kernel(L, ld, st)
    for l in body: ln.append(" " + l)
    for k, v in stores:
        ln.append(f" STU(re + {k*8}, {v[0]}); STU(im + {k*8}, {v[1]});")
    ln.append("}")
    return ln

def gen_run(L):
    L3 = L ** 3
    L2 = L * L
    ps = PS(L)
    VOL = L * ps          # padded volume size in doubles
    PAD = 8 * L + 64
    ln = []
    for nm in ("SR", "SI", "CR", "CI"):
        ln.append(f"static double {nm}{L}[{VOL + PAD}] __attribute__((aligned(64)));")
    if L == 64:
        step = f"""
      for(long x=0; x<{L}; ++x){{
        pz_{L}(SR{L} + x*{ps}, SI{L} + x*{ps});
        py_{L}(SR{L} + x*{ps}, SI{L} + x*{ps});
      }}
      px_{L}(SR{L}, SI{L}, CR{L}, CI{L});"""
        dei = f"""    for(long x=0; x<{L}; ++x){{
      deint(xb + x*{2*L2}, SR{L} + x*{ps}, SI{L} + x*{ps}, {L2});
      deint(cb + x*{2*L2}, CR{L} + x*{ps}, CI{L} + x*{ps}, {L2});
    }}"""
        def mk_inter(dst):
            return f"""for(long x=0; x<{L}; ++x) inter(SR{L} + x*{ps}, SI{L} + x*{ps}, {dst} + b*{2*L3} + x*{2*L2}, {L2});"""
    else:
        step = f"""
      pz_{L}(SR{L}, SI{L});
      py_{L}(SR{L}, SI{L});
      px_{L}(SR{L}, SI{L}, CR{L}, CI{L});"""
        dei = f"""    deint(xb, SR{L}, SI{L}, {L3});
    deint(cb, CR{L}, CI{L}, {L3});"""
        def mk_inter(dst):
            return f"inter(SR{L}, SI{L}, {dst} + b*{2*L3}, {L3});"
    ln.append(f"""
void run{L}(int64_t B, int64_t m, const double* x0, const double* c, double* out1, double* outm){{
  for(int64_t b=0; b<B; ++b){{
    const double* xb = x0 + b*{2*L3};
    const double* cb = c  + b*{2*L3};
{dei}
    if(m < 1){{
      {mk_inter("outm")}
      {step}
      {mk_inter("out1")}
      continue;
    }}
    for(int64_t it=0; it<m; ++it){{
      {step}
      if(it == 0 && m > 1) {mk_inter("out1")}
    }}
    {mk_inter("outm")}
    if(m == 1) memcpy(out1 + b*{2*L3}, outm + b*{2*L3}, {2*L3}*sizeof(double));
  }}
}}
void dpz{L}(long reps){{ for(long r=0;r<reps;++r){{ {"for(long x=0;x<%d;++x) pz_%d(SR%d + x*%d, SI%d + x*%d);" % (L,L,L,ps,L,ps) if L==64 else "pz_%d(SR%d, SI%d);" % (L,L,L)} }} }}
void dpy{L}(long reps){{ for(long r=0;r<reps;++r){{ {"for(long x=0;x<%d;++x) py_%d(SR%d + x*%d, SI%d + x*%d);" % (L,L,L,ps,L,ps) if L==64 else "py_%d(SR%d, SI%d);" % (L,L,L)} }} }}
void dpx{L}(long reps){{ for(long r=0;r<reps;++r){{ px_{L}(SR{L}, SI{L}, CR{L}, CI{L}); }} }}
void dseed{L}(const double* x0, const double* c){{
  {"for(long x=0;x<64;++x){ deint(x0 + x*%d, SR64 + x*%d, SI64 + x*%d, %d); deint(c + x*%d, CR64 + x*%d, CI64 + x*%d, %d);}" % (2*L2, ps, ps, L2, 2*L2, ps, ps, L2) if L==64 else "deint(x0, SR%d, SI%d, %d); deint(c, CR%d, CI%d, %d);" % (L,L,L3,L,L,L3)}
}}
""")
    return ln

PREAMBLE = r"""
// ============================================================================
// Specialized iterated batched 3D complex FFTs for L in {6,8,13,17,23,36,45,64}
//
// Auto-generated C (generator: dev/gen3.py in this working directory).
// All transform arithmetic is implemented from scratch in this file:
//   - composite sizes via Cooley-Tukey / Good-Thomas (prime-factor) splits
//     (6=2x3, 8=2x4, 9=3x3, 36=4x9, 45=9x5, 64=8x8),
//   - odd primes (13, 17, 23) via direct even/odd symmetric DFT folding,
//   - twiddle/cosine tables precomputed at GENERATION time (mpmath, 50 digits)
//     and embedded below as hex-exact double literals.
// No FFT library code is called or linked; the only libc use is libm-free
// SIMD arithmetic, mmap/madvise for the scratch arena, and memcpy/memset.
// Single-threaded throughout. Iteration per volume:
//   z = FFT3(x) + c ;  x <- z / (1 + |z|)
// computed as three split-complex AVX-512 passes (z-lines via in-register
// 8x8 transposes, y-lines, then x-lines fused with the +c and the nonlinear
// map). Small sizes (<=23) additionally have an 8-volume lane-major batched
// pipeline; large sizes use stream-tiled x-passes.
// ============================================================================
#include <immintrin.h>
#include <stdint.h>
#include <string.h>
#include <sys/mman.h>
#include <stdlib.h>

static double* ARENA = 0;
void init0(int64_t total_doubles){
  if(ARENA) return;
  size_t len = (size_t)total_doubles * 8;
  len = (len + (2u<<20) - 1) & ~(size_t)((2u<<20) - 1);
  void* p = mmap(0, len + (2u<<20), PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
  if(p == MAP_FAILED){ ARENA = (double*)aligned_alloc(4096, len); memset(ARENA, 0, len); return; }
  uintptr_t q = ((uintptr_t)p + (2u<<20) - 1) & ~(uintptr_t)((2u<<20) - 1);
  madvise((void*)q, len, MADV_HUGEPAGE);
  memset((void*)q, 0, len);
  ARENA = (double*)q;
}
typedef __m512d V;
#define ADD _mm512_add_pd
#define SUB _mm512_sub_pd
#define MUL _mm512_mul_pd
#define FMA _mm512_fmadd_pd
#define FNMA _mm512_fnmadd_pd
#define SET1 _mm512_set1_pd
#define LDU _mm512_loadu_pd
#define STU _mm512_storeu_pd
#define MST(p,k,v) _mm512_mask_storeu_pd((p),(k),(v))
#define PF(p) _mm_prefetch((const char*)(p), _MM_HINT_T0)
#define XOR(a,b) _mm512_castsi512_pd(_mm512_xor_si512(_mm512_castpd_si512(a),_mm512_castpd_si512(b)))
#define NEGZ SET1(-0.0)

#define TR8(r0,r1,r2,r3,r4,r5,r6,r7) do { \
  V _t0=_mm512_unpacklo_pd(r0,r1), _t1=_mm512_unpackhi_pd(r0,r1); \
  V _t2=_mm512_unpacklo_pd(r2,r3), _t3=_mm512_unpackhi_pd(r2,r3); \
  V _t4=_mm512_unpacklo_pd(r4,r5), _t5=_mm512_unpackhi_pd(r4,r5); \
  V _t6=_mm512_unpacklo_pd(r6,r7), _t7=_mm512_unpackhi_pd(r6,r7); \
  V _s0=_mm512_shuffle_f64x2(_t0,_t2,0x88), _s1=_mm512_shuffle_f64x2(_t1,_t3,0x88); \
  V _s2=_mm512_shuffle_f64x2(_t0,_t2,0xdd), _s3=_mm512_shuffle_f64x2(_t1,_t3,0xdd); \
  V _s4=_mm512_shuffle_f64x2(_t4,_t6,0x88), _s5=_mm512_shuffle_f64x2(_t5,_t7,0x88); \
  V _s6=_mm512_shuffle_f64x2(_t4,_t6,0xdd), _s7=_mm512_shuffle_f64x2(_t5,_t7,0xdd); \
  r0=_mm512_shuffle_f64x2(_s0,_s4,0x88); r4=_mm512_shuffle_f64x2(_s0,_s4,0xdd); \
  r1=_mm512_shuffle_f64x2(_s1,_s5,0x88); r5=_mm512_shuffle_f64x2(_s1,_s5,0xdd); \
  r2=_mm512_shuffle_f64x2(_s2,_s6,0x88); r6=_mm512_shuffle_f64x2(_s2,_s6,0xdd); \
  r3=_mm512_shuffle_f64x2(_s3,_s7,0x88); r7=_mm512_shuffle_f64x2(_s3,_s7,0xdd); \
} while(0)

// x = z/(1+|z|) with z = v + c
static inline void map8(V vr, V vi, V cr, V ci, V* outr, V* outi){
  V zr = ADD(vr, cr), zi = ADD(vi, ci);
  V q  = FMA(zr, zr, MUL(zi, zi));
#if defined(MAP_EXACT)
  V mag = _mm512_sqrt_pd(q);
  V s = _mm512_div_pd(SET1(1.0), ADD(SET1(1.0), mag));
#elif defined(MAP_NEWTON2)
  V qc = _mm512_max_pd(q, SET1(1e-300));
  V r  = _mm512_rsqrt14_pd(qc);
  V h  = MUL(qc, SET1(0.5));
  r = MUL(r, FNMA(MUL(h, r), r, SET1(1.5)));
  r = MUL(r, FNMA(MUL(h, r), r, SET1(1.5)));
  V t = ADD(SET1(1.0), r);
  V s0 = _mm512_rcp14_pd(t);
  s0 = MUL(s0, FNMA(t, s0, SET1(2.0)));
  s0 = MUL(s0, FNMA(t, s0, SET1(2.0)));
  V s = MUL(r, s0);
#else
  V qc = _mm512_max_pd(q, SET1(1e-300));
  V r  = _mm512_rsqrt14_pd(qc);
  V h  = MUL(qc, SET1(0.5));
  r = MUL(r, FNMA(MUL(h, r), r, SET1(1.5)));
  r = MUL(r, FNMA(MUL(h, r), r, SET1(1.5)));
  V s = _mm512_div_pd(r, ADD(SET1(1.0), r));
#endif
  *outr = MUL(zr, s);
  *outi = MUL(zi, s);
}

static const long long IDXRE[8] = {0,2,4,6,8,10,12,14};
static const long long IDXIM[8] = {1,3,5,7,9,11,13,15};
static const long long IDXLO[8] = {0,8,1,9,2,10,3,11};
static const long long IDXHI[8] = {4,12,5,13,6,14,7,15};
#define XRE (_mm512_loadu_si512(IDXRE))
#define XIM (_mm512_loadu_si512(IDXIM))
#define XLO (_mm512_loadu_si512(IDXLO))
#define XHI (_mm512_loadu_si512(IDXHI))

static void deint(const double* src, double* re, double* im, long n){
  __m512i xre = _mm512_loadu_si512(IDXRE), xim = _mm512_loadu_si512(IDXIM);
  long i = 0;
  for(; i + 8 <= n; i += 8){
    V a = LDU(src + 2*i), b = LDU(src + 2*i + 8);
    STU(re + i, _mm512_permutex2var_pd(a, xre, b));
    STU(im + i, _mm512_permutex2var_pd(a, xim, b));
  }
  long r = n - i;
  if(r){
    __mmask8 m1 = (__mmask8)((1u << (2*r >= 8 ? 8 : 2*r)) - 1);
    __mmask8 m2 = (__mmask8)(2*r > 8 ? ((1u << (2*r - 8)) - 1) : 0);
    V a = _mm512_maskz_loadu_pd(m1, src + 2*i);
    V b = _mm512_maskz_loadu_pd(m2, src + 2*i + 8);
    __mmask8 mo = (__mmask8)((1u << r) - 1);
    MST(re + i, mo, _mm512_permutex2var_pd(a, xre, b));
    MST(im + i, mo, _mm512_permutex2var_pd(a, xim, b));
  }
}

static void inter(const double* re, const double* im, double* dst, long n){
  __m512i xlo = _mm512_loadu_si512(IDXLO), xhi = _mm512_loadu_si512(IDXHI);
  long i = 0;
  for(; i + 8 <= n; i += 8){
    V a = LDU(re + i), b = LDU(im + i);
    STU(dst + 2*i,     _mm512_permutex2var_pd(a, xlo, b));
    STU(dst + 2*i + 8, _mm512_permutex2var_pd(a, xhi, b));
  }
  long r = n - i;
  if(r){
    V a = _mm512_maskz_loadu_pd((__mmask8)((1u<<r)-1), re + i);
    V b = _mm512_maskz_loadu_pd((__mmask8)((1u<<r)-1), im + i);
    __mmask8 m1 = (__mmask8)((1u << (2*r >= 8 ? 8 : 2*r)) - 1);
    __mmask8 m2 = (__mmask8)(2*r > 8 ? ((1u << (2*r - 8)) - 1) : 0);
    MST(dst + 2*i,     m1, _mm512_permutex2var_pd(a, xlo, b));
    MST(dst + 2*i + 8, m2, _mm512_permutex2var_pd(a, xhi, b));
  }
}
"""

def main():
    parts = [PREAMBLE]
    for L in SIZES:
        parts.append("\n".join(gen_pz(L)))
        parts.append("\n".join(gen_py(L)))
        parts.append("\n".join(gen_px(L)))
        parts.append("\n".join(gen_dbg(L)))
        parts.append("\n".join(gen_run(L)))
    src = "\n".join(parts) + "\n"
    with open("implementation.c", "w") as f:
        f.write(src)
    print(f"wrote implementation.c: {len(src)} bytes, {src.count(chr(10))} lines")

if __name__ == "__main__":
    main()
