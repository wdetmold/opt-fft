import sys
import numpy as np
from netlib import E, dft_small, fmt, KC, cmul_const
from primelib2 import emit_prime_reim, KSPLITS

SOA_SIZES = (6, 8, 13, 17, 23, 36, 45)
PRIMES = (13, 17, 23)

PRELUDE = r'''
#include <immintrin.h>
#include <string.h>
#include <stdint.h>
#include <sys/mman.h>

typedef double vd __attribute__((vector_size(64), aligned(64)));
#define K(x) ((vd){x,x,x,x,x,x,x,x})
#define K2(x) ((vd)_mm512_set1_pd(x))
#define FMA(a,b,c) ((vd)_mm512_fmadd_pd((__m512d)(a),(__m512d)(b),(__m512d)(c)))
#define FMS(a,b,c) ((vd)_mm512_fmsub_pd((__m512d)(a),(__m512d)(b),(__m512d)(c)))
#define FNMA(a,b,c) ((vd)_mm512_fnmadd_pd((__m512d)(a),(__m512d)(b),(__m512d)(c)))
#define LD(p) (*(const vd*)(p))
#define ST(p,v) (*(vd*)(p) = (v))
#define MD(x) ((__m512d)(x))
#define VD(x) ((vd)(x))
#define BAR() __asm__ volatile("" ::: "memory")

static void* xalloc(size_t bytes){
  bytes = (bytes + (2UL<<20) - 1) & ~((2UL<<20)-1);
  void* p = mmap(0, bytes, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
  madvise(p, bytes, MADV_HUGEPAGE);
  memset(p, 0, bytes);
  return p;
}

static inline vd rcp_nr(vd d){
  __m512d w = _mm512_rcp14_pd(MD(d));
  w = _mm512_mul_pd(w, _mm512_fnmadd_pd(MD(d), w, _mm512_set1_pd(2.0)));
  w = _mm512_mul_pd(w, _mm512_fnmadd_pd(MD(d), w, _mm512_set1_pd(2.0)));
  return VD(w);
}
static inline vd rsqrt_nr2(vd t){
  __m512d r = _mm512_rsqrt14_pd(MD(t));
  __m512d h = _mm512_mul_pd(MD(t), _mm512_set1_pd(0.5));
  __m512d e;
  e = _mm512_fnmadd_pd(_mm512_mul_pd(h, r), r, _mm512_set1_pd(1.5)); r = _mm512_mul_pd(r, e);
  e = _mm512_fnmadd_pd(_mm512_mul_pd(h, r), r, _mm512_set1_pd(1.5)); r = _mm512_mul_pd(r, e);
  return VD(r);
}
#define MAPH(zr,zi,wv) vd wv; { vd t_ = FMA(zr,zr, zi*zi); vd s_ = VD(_mm512_sqrt_pd(MD(t_))); wv = rcp_nr(K(1.0)+s_); }
#define MAPR(zr,zi,wv) vd wv; { vd t_ = FMA(zr,zr, FMA(zi,zi,K(1e-300))); vd r_ = rsqrt_nr2(t_); wv = rcp_nr(FMA(t_,r_,K(1.0))); }
'''

def pad_plane(L):
    base = L*L*16
    for p in (0, 16, 32, 48, 64, 80, 96, 112, 128):
        s = base + p
        r = (s*8) % 4096
        if 1024 <= r <= 3072:
            return s
    return base + 136

class Loader:
    def __init__(self, base, stride):
        self.base = base; self.stride = stride
    def full(self, e, j):
        o = j*self.stride
        return (e.v(f"LD({self.base} + {o})"), e.v(f"LD({self.base} + {o+8})"))
    def one(self, e, j, comp):
        o = j*self.stride + 8*comp
        return e.v(f"LD({self.base} + {o})")

class Storer:
    def __init__(self, base, stride):
        self.base = base; self.stride = stride
    def __call__(self, e, k, r, i):
        o = k*self.stride
        e.raw(f"ST({self.base} + {o}, {r}); ST({self.base} + {o+8}, {i});")

class MapStorer:
    def __init__(self, base, cbase, stride, pattern="HR"):
        self.base = base; self.cbase = cbase; self.stride = stride
        self.cnt = 0; self.pattern = pattern
    def __call__(self, e, k, r, i):
        o = k*self.stride
        m = "MAP" + self.pattern[self.cnt % len(self.pattern)]
        self.cnt += 1
        e.raw(f"{{ vd zr = {r} + LD({self.cbase} + {o}); vd zi = {i} + LD({self.cbase} + {o+8}); "
              f"{m}(zr, zi, w); ST({self.base} + {o}, zr*w); ST({self.base} + {o+8}, zi*w); }}")

def crt_maps(N, A, B):
    Binv = pow(B, -1, A); Ainv = pow(A, -1, B)
    inmap = {(n1,n2): (B*n1 + A*n2) % N for n1 in range(A) for n2 in range(B)}
    outmap = {(k1,k2): (k1*B*Binv + k2*A*Ainv) % N for k1 in range(A) for k2 in range(B)}
    return inmap, outmap

def plan(L):
    if L == 6: return ('pfa', 2, 3)
    if L == 8: return ('direct',)
    if L in PRIMES: return ('prime',)
    if L == 36: return ('pfa', 4, 9)
    if L == 45: return ('pfa', 9, 5)
    if L == 64: return ('ct', 8, 8)
    raise ValueError

def emit_net(e, L, loader, storer, sc):
    p = plan(L)
    if p[0] == 'prime':
        emit_prime_reim(e, L, lambda e2, j: (loader.one(e2, j, 0), loader.one(e2, j, 1)) if False else None, storer, sc)
        return
    if p[0] == 'direct' or L in (6, 8):
        xs = [loader.full(e, j) for j in range(L)]
        ys = dft_small(e, xs, L)
        for k in range(L):
            storer(e, k, ys[k][0], ys[k][1])
        return
    kind, A, B = p
    if kind == 'pfa':
        inmap, outmap = crt_maps(L, A, B)
        for n2 in range(B):
            xs = [loader.full(e, inmap[(n1,n2)]) for n1 in range(A)]
            ys = dft_small(e, xs, A)
            for k1 in range(A):
                e.raw(f"ST({sc} + {(n2*A+k1)*16}, {ys[k1][0]}); ST({sc} + {(n2*A+k1)*16+8}, {ys[k1][1]});")
        e.raw("BAR();")
        for k1 in range(A):
            xs = [(e.v(f"LD({sc} + {(n2*A+k1)*16})"), e.v(f"LD({sc} + {(n2*A+k1)*16+8})")) for n2 in range(B)]
            ys = dft_small(e, xs, B)
            for k2 in range(B):
                storer(e, outmap[(k1,k2)], ys[k2][0], ys[k2][1])
    else:
        for n2 in range(B):
            xs = [loader.full(e, B*n1 + n2) for n1 in range(A)]
            ys = dft_small(e, xs, A)
            for k1 in range(A):
                w = cmul_const(e, ys[k1], k1*n2, L)
                e.raw(f"ST({sc} + {(n2*A+k1)*16}, {w[0]}); ST({sc} + {(n2*A+k1)*16+8}, {w[1]});")
        e.raw("BAR();")
        for k1 in range(A):
            xs = [(e.v(f"LD({sc} + {(n2*A+k1)*16})"), e.v(f"LD({sc} + {(n2*A+k1)*16+8})")) for n2 in range(B)]
            ys = dft_small(e, xs, B)
            for k2 in range(B):
                storer(e, A*k2 + k1, ys[k2][0], ys[k2][1])

class PLoader:
    """prime loader adapter: one-component loads"""
    def __init__(self, base, stride):
        self.base = base; self.stride = stride
    def __call__(self, e, j):
        # returns lazy pair accessed by [comp]
        base, stride = self.base, self.stride
        class P:
            def __getitem__(self, comp):
                return e.v(f"LD({base} + {j*stride + 8*comp})")
        return P()

def emit_pass(L, base_expr, stride, npencils, pencil_step, map_c=None, fname="f"):
    """emit a full pass function: loop over pencils; pencil base = base_expr + r*pencil_step"""
    e = E()
    e.raw(f"for(long r=0;r<{npencils};r++){{")
    e.raw(f"double* restrict q = {base_expr} + r*{pencil_step};")
    if map_c is not None:
        e.raw(f"const double* restrict cq = {map_c} + r*{pencil_step};")
    loader = Loader("q", stride)
    storer = MapStorer("q", "cq", stride) if map_c is not None else Storer("q", stride)
    if plan(L)[0] == 'prime':
        emit_prime_reim(e, L, PLoader("q", stride), storer, "SC")
    else:
        emit_net(e, L, loader, storer, "SC")
    e.raw("}")
    return e.code()

def gen_soa(L):
    RS = L*16
    PS = pad_plane(L)
    n3 = L*L*L
    out = [f"#define SC SC_{L}\n", f"static double SC_{L}[{(L+4)*32}] __attribute__((aligned(64)));\n"]
    # x-pass: all pencils of a full volume: base = buf + r*16 (r in 0..L*L-1), stride PS
    body_x = emit_pass(L, "buf", PS, L*L, 16)
    out.append(f"static void px_{L}(double* restrict buf){{\n  {body_x}\n}}\n")
    # y-pass for one plane: base = pl + z*16, stride RS
    body_y = emit_pass(L, "pl", RS, L, 16)
    out.append(f"static void py_{L}(double* restrict pl){{\n  {body_y}\n}}\n")
    # z-pass+map for one plane: base = pl + y*RS, stride 16
    body_z = emit_pass(L, "pl", 16, L, RS, map_c="cp")
    out.append(f"static void pz_{L}(double* restrict pl, const double* restrict cp){{\n  {body_z}\n}}\n")
    out.append(f"""
static double *BUF_{L}, *CB_{L};
static void init_{L}(void){{
  BUF_{L} = (double*)xalloc((size_t){L}*{PS}*8 + 4096);
  CB_{L}  = (double*)xalloc((size_t){L}*{PS}*8 + 4096);
}}
static void step_{L}(double* restrict buf, const double* restrict cb){{
  px_{L}(buf);
  for(int x=0;x<{L};x++){{
    double* pl = buf + (size_t)x*{PS};
    const double* cp = cb + (size_t)x*{PS};
    py_{L}(pl);
    pz_{L}(pl, cp);
  }}
}}
void bench_pass_{L}(int which, long iters){{
  for(long it=0; it<iters; it++){{
    if(which==0) px_{L}(BUF_{L});
    else if(which==1){{ for(int x=0;x<{L};x++) py_{L}(BUF_{L} + (size_t)x*{PS}); }}
    else {{ for(int x=0;x<{L};x++) pz_{L}(BUF_{L} + (size_t)x*{PS}, CB_{L} + (size_t)x*{PS}); }}
  }}
}}
static void conv_in_{L}(const double* restrict src, double* restrict dst, int nv){{
  for(int v=0;v<nv;v++){{
    const double* s = src + (size_t)v*{2*n3};
    for(int x=0;x<{L};x++){{
      double* dp = dst + (size_t)x*{PS};
      const double* sp = s + (size_t)x*{2*L*L};
      for(int r=0;r<{L*L};r++){{
        dp[r*16 + v] = sp[2*r];
        dp[r*16 + 8 + v] = sp[2*r+1];
      }}
    }}
  }}
}}
static void conv_out_{L}(const double* restrict src, double* restrict dst, int nv){{
  for(int v=0;v<nv;v++){{
    double* d = dst + (size_t)v*{2*n3};
    for(int x=0;x<{L};x++){{
      const double* sp = src + (size_t)x*{PS};
      double* dp = d + (size_t)x*{2*L*L};
      for(int r=0;r<{L*L};r++){{
        dp[2*r] = sp[r*16 + v];
        dp[2*r+1] = sp[r*16 + 8 + v];
      }}
    }}
  }}
}}
static void run_{L}(long B, long m, const double* restrict x0, const double* restrict c, double* restrict o1, double* restrict om){{
  for(long g=0; g<B; g+=8){{
    int nv = (int)((B-g) < 8 ? (B-g) : 8);
    long off = g*{2*n3};
    if(nv < 8){{ memset(BUF_{L}, 0, (size_t){L}*{PS}*8); memset(CB_{L}, 0, (size_t){L}*{PS}*8); }}
    conv_in_{L}(x0 + off, BUF_{L}, nv);
    conv_in_{L}(c + off, CB_{L}, nv);
    for(long t=1;t<=m;t++){{
      step_{L}(BUF_{L}, CB_{L});
      if(t==1) conv_out_{L}(BUF_{L}, o1 + off, nv);
    }}
    conv_out_{L}(BUF_{L}, om + off, nv);
  }}
}}
#undef SC
""")
    return "".join(out)

def generate():
    parts = [PRELUDE]
    for L in SOA_SIZES:
        parts.append(gen_soa(L))
    parts.append("void init_all(void){ " + " ".join(f"init_{L}();" for L in SOA_SIZES) + " }\n")
    parts.append("""
void run_size(int L, long B, long m, const double* x0, const double* c, double* o1, double* om){
  switch(L){
""" + "".join(f"    case {L}: run_{L}(B,m,x0,c,o1,om); break;\n" for L in SOA_SIZES) + """
  }
}
""")
    return "".join(parts)

if __name__ == "__main__":
    src = generate()
    open(sys.argv[1] if len(sys.argv)>1 else "implementation.c", "w").write(src)
    print("lines:", len(src.splitlines()))
