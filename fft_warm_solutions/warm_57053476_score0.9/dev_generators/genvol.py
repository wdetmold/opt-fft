# per-volume engine: split re/im rows, padded; y/x passes vertical, z-pass via 8x8 transposes
import numpy as np
from netlib import E, dft_small, fmt, KC, cmul_const

def pad8(n): return (n+7)//8*8

def vol_layout(L):
    RW = pad8(L)                  # row width (doubles per component)
    Lr = pad8(L)                  # rows per plane (padded for z-tiles)
    base = Lr*2*RW
    PLS = base
    for p in (0, 32, 64, 96, 128, 160, 192, 224):
        s = base + p
        if 1024 <= (s*8) % 4096 <= 3072:
            PLS = s; break
    else:
        PLS = base + 136
    return RW, Lr, PLS

TRANS_HELPERS = r'''
static const long long IDX_TAB[6][8] __attribute__((aligned(64))) = {
  {0,1,2,3,8,9,10,11},{4,5,6,7,12,13,14,15},
  {0,1,8,9,4,5,12,13},{2,3,10,11,6,7,14,15},
  {0,8,2,10,4,12,6,14},{1,9,3,11,5,13,7,15},
};
#define IX(i) (_mm512_load_si512((const void*)IDX_TAB[i]))
#define TR8(a0,a1,a2,a3,a4,a5,a6,a7) do{ \
  __m512d tq0_,tq1_,tq2_,tq3_,tq4_,tq5_,tq6_,tq7_; \
  tq0_=_mm512_permutex2var_pd(MD(a0),IX(0),MD(a4)); tq4_=_mm512_permutex2var_pd(MD(a0),IX(1),MD(a4)); \
  tq1_=_mm512_permutex2var_pd(MD(a1),IX(0),MD(a5)); tq5_=_mm512_permutex2var_pd(MD(a1),IX(1),MD(a5)); \
  tq2_=_mm512_permutex2var_pd(MD(a2),IX(0),MD(a6)); tq6_=_mm512_permutex2var_pd(MD(a2),IX(1),MD(a6)); \
  tq3_=_mm512_permutex2var_pd(MD(a3),IX(0),MD(a7)); tq7_=_mm512_permutex2var_pd(MD(a3),IX(1),MD(a7)); \
  __m512d tr0_,tr1_,tr2_,tr3_,tr4_,tr5_,tr6_,tr7_; \
  tr0_=_mm512_permutex2var_pd(tq0_,IX(2),tq2_); tr2_=_mm512_permutex2var_pd(tq0_,IX(3),tq2_); \
  tr1_=_mm512_permutex2var_pd(tq1_,IX(2),tq3_); tr3_=_mm512_permutex2var_pd(tq1_,IX(3),tq3_); \
  tr4_=_mm512_permutex2var_pd(tq4_,IX(2),tq6_); tr6_=_mm512_permutex2var_pd(tq4_,IX(3),tq6_); \
  tr5_=_mm512_permutex2var_pd(tq5_,IX(2),tq7_); tr7_=_mm512_permutex2var_pd(tq5_,IX(3),tq7_); \
  a0=VD(_mm512_permutex2var_pd(tr0_,IX(4),tr1_)); a1=VD(_mm512_permutex2var_pd(tr0_,IX(5),tr1_)); \
  a2=VD(_mm512_permutex2var_pd(tr2_,IX(4),tr3_)); a3=VD(_mm512_permutex2var_pd(tr2_,IX(5),tr3_)); \
  a4=VD(_mm512_permutex2var_pd(tr4_,IX(4),tr5_)); a5=VD(_mm512_permutex2var_pd(tr4_,IX(5),tr5_)); \
  a6=VD(_mm512_permutex2var_pd(tr6_,IX(4),tr7_)); a7=VD(_mm512_permutex2var_pd(tr6_,IX(5),tr7_)); \
}while(0)
'''

class VLoader:
    """loads point j: re at base + j*stride, im at +RW"""
    def __init__(self, base, stride, RW):
        self.base=base; self.stride=stride; self.RW=RW
    def full(self, e, j):
        o=j*self.stride
        return (e.v(f"LD({self.base} + {o})"), e.v(f"LD({self.base} + {o} + {self.RW})"))
    def one(self, e, j, comp):
        o=j*self.stride + (self.RW if comp else 0)
        return e.v(f"LD({self.base} + {o})")

class VStorer:
    def __init__(self, base, stride, RW):
        self.base=base; self.stride=stride; self.RW=RW
    def __call__(self, e, k, r, i):
        o=k*self.stride
        e.raw(f"ST({self.base} + {o}, {r}); ST({self.base} + {o} + {self.RW}, {i});")

class VMapStorer:
    def __init__(self, base, cbase, stride, RW, pattern="HR"):
        self.base=base; self.cbase=cbase; self.stride=stride; self.RW=RW
        self.cnt=0; self.pattern=pattern
    def __call__(self, e, k, r, i):
        o=k*self.stride
        m = "MAP" + self.pattern[self.cnt % len(self.pattern)]
        self.cnt+=1
        e.raw(f"{{ vd zr = {r} + LD({self.cbase} + {o}); vd zi = {i} + LD({self.cbase} + {o} + {self.RW}); "
              f"{m}(zr, zi, w); ST({self.base} + {o}, zr*w); ST({self.base} + {o} + {self.RW}, zi*w); }}")

# scratch-interleaved loader/storer for z-net (SC2: point j re at j*16, im at j*16+8)
class SLoader:
    def __init__(self, name): self.name=name
    def full(self, e, j):
        return (e.v(f"LD({self.name} + {j*16})"), e.v(f"LD({self.name} + {j*16+8})"))
    def one(self, e, j, comp):
        return e.v(f"LD({self.name} + {j*16+8*comp})")
class SStorer:
    def __init__(self, name): self.name=name
    def __call__(self, e, k, r, i):
        e.raw(f"ST({self.name} + {k*16}, {r}); ST({self.name} + {k*16+8}, {i});")

def emit_vol_net(e, L, loader, storer, sc, plan, prime_emit):
    kind = plan[0]
    if kind == 'prime':
        prime_emit(e, L, loader, storer, sc)
        return
    if kind == 'direct':
        xs=[loader.full(e,j) for j in range(L)]
        ys=dft_small(e,xs,L)
        for k in range(L): storer(e,k,ys[k][0],ys[k][1])
        return
    _, A, B = plan
    if kind=='pfa':
        Binv=pow(B,-1,A); Ainv=pow(A,-1,B)
        inm={(n1,n2):(B*n1+A*n2)%L for n1 in range(A) for n2 in range(B)}
        outm={(k1,k2):(k1*B*Binv+k2*A*Ainv)%L for k1 in range(A) for k2 in range(B)}
        tw=lambda e,x,k1,n2: x
    else:
        inm={(n1,n2): B*n1+n2 for n1 in range(A) for n2 in range(B)}
        outm={(k1,k2): A*k2+k1 for k1 in range(A) for k2 in range(B)}
        tw=lambda e,x,k1,n2: cmul_const(e,x,k1*n2,L)
    for n2 in range(B):
        xs=[loader.full(e, inm[(n1,n2)]) for n1 in range(A)]
        ys=dft_small(e,xs,A)
        for k1 in range(A):
            w=tw(e,ys[k1],k1,n2)
            e.raw(f"ST({sc} + {(n2*A+k1)*16}, {w[0]}); ST({sc} + {(n2*A+k1)*16+8}, {w[1]});")
    e.raw("BAR();")
    for k1 in range(A):
        xs=[(e.v(f"LD({sc} + {(n2*A+k1)*16})"), e.v(f"LD({sc} + {(n2*A+k1)*16+8})")) for n2 in range(B)]
        ys=dft_small(e,xs,B)
        for k2 in range(B):
            storer(e, outm[(k1,k2)], ys[k2][0], ys[k2][1])

def gen_vol(L, plan, prime_emit=None):
    RW, Lr, PLS = vol_layout(L)
    NT = RW//8            # col tiles
    NYT = Lr//8           # row tiles
    n3=L*L*L
    parts=[]
    parts.append(f"static double VSC_{L}[{(L+6)*32}] __attribute__((aligned(64)));\n")
    parts.append(f"static double VS2_{L}[{Lr*16+64}] __attribute__((aligned(64)));\n")
    # --- x-pass (vertical over v-columns), per (y row, v): stride PLS
    e=E()
    e.raw(f"for(long r=0;r<{L*NT};r++){{")
    e.raw(f"long y = r/{NT}, v = r%{NT};")
    e.raw(f"double* restrict q = buf + y*{2*RW} + v*8;")
    emit_vol_net(e, L, VLoader("q", PLS, RW), VStorer("q", PLS, RW), f"VSC_{L}", plan, prime_emit)
    e.raw("}")
    parts.append(f"static void vx_{L}(double* restrict buf){{\n  {e.code()}\n}}\n")
    # --- y-pass per plane
    e=E()
    e.raw(f"for(long v=0;v<{NT};v++){{")
    e.raw(f"double* restrict q = pl + v*8;")
    emit_vol_net(e, L, VLoader("q", 2*RW, RW), VStorer("q", 2*RW, RW), f"VSC_{L}", plan, prime_emit)
    e.raw("}")
    parts.append(f"static void vy_{L}(double* restrict pl){{\n  {e.code()}\n}}\n")
    # --- z-pass per plane: tiles + net + map
    e=E()
    e.raw(f"for(long yt=0;yt<{NYT};yt++){{")
    e.raw(f"double* restrict q = pl + yt*{8*2*RW};")
    e.raw(f"const double* restrict cq = cp + yt*{8*2*RW};")
    # forward transpose into VS2 (interleaved re/im)
    for comp in range(2):
        co = RW*comp
        for t in range(NT):
            names=[]
            for row in range(8):
                nm=e.v(f"LD(q + {row*2*RW + t*8 + co})")
                names.append(nm)
            e.raw f"TR8({','.join(names)});" if False else e.raw(f"TR8({','.join(names)});")
            for i in range(8):
                z = t*8+i
                e.raw(f"ST(VS2_{L} + {z*16 + 8*comp}, {names[i]});")
    e.raw("BAR();")
    # z-net on VS2 in place (only z<L inputs; outputs z<L; pad cols zeroed below)
    emit_vol_net(e, L, SLoader(f"VS2_{L}"), SStorer(f"VS2_{L}"), f"VSC_{L}", plan, prime_emit)
    if RW > L:
        for z in range(L, RW):
            e.raw(f"ST(VS2_{L} + {z*16}, K(0.0)); ST(VS2_{L} + {z*16+8}, K(0.0));")
    e.raw("BAR();")
    # inverse transpose + c + map + store
    cnt=[0]
    for t in range(NT):
        rn=[]; im=[]
        for i in range(8):
            z=t*8+i
            rn.append(e.v(f"LD(VS2_{L} + {z*16})"))
            im.append(e.v(f"LD(VS2_{L} + {z*16+8})"))
        e.raw(f"TR8({','.join(rn)});")
        e.raw(f"TR8({','.join(im)});")
        for row in range(8):
            o=row*2*RW + t*8
            m = "MAPH" if (cnt[0]&1)==0 else "MAPR"
            cnt[0]+=1
            e.raw(f"{{ vd zr = {rn[row]} + LD(cq + {o}); vd zi = {im[row]} + LD(cq + {o} + {RW}); "
                  f"{m}(zr, zi, w); ST(q + {o}, zr*w); ST(q + {o} + {RW}, zi*w); }}")
    e.raw("}")
    parts.append(f"static void vz_{L}(double* restrict pl, const double* restrict cp){{\n  {e.code()}\n}}\n")
    # --- driver
    parts.append(f"""
static double *VBUF_{L}, *VCB_{L};
static void vinit_{L}(void){{
  VBUF_{L} = (double*)xalloc((size_t){L}*{PLS}*8 + 4096);
  VCB_{L}  = (double*)xalloc((size_t){L}*{PLS}*8 + 4096);
}}
static void vstep_{L}(double* restrict buf, const double* restrict cb){{
  vx_{L}(buf);
  for(int x=0;x<{L};x++){{
    double* pl = buf + (size_t)x*{PLS};
    const double* cp = cb + (size_t)x*{PLS};
    vy_{L}(pl);
    vz_{L}(pl, cp);
  }}
}}
void vbench_pass_{L}(int which, long iters){{
  for(long it=0; it<iters; it++){{
    if(which==0) vx_{L}(VBUF_{L});
    else if(which==1){{ for(int x=0;x<{L};x++) vy_{L}(VBUF_{L} + (size_t)x*{PLS}); }}
    else {{ for(int x=0;x<{L};x++) vz_{L}(VBUF_{L} + (size_t)x*{PLS}, VCB_{L} + (size_t)x*{PLS}); }}
  }}
}}
static void vconv_in_{L}(const double* restrict src, double* restrict dst){{
  for(int x=0;x<{L};x++){{
    double* dp = dst + (size_t)x*{PLS};
    const double* sp = src + (size_t)x*{2*L*L};
    for(int y=0;y<{L};y++)
      for(int z=0;z<{L};z++){{
        dp[y*{2*RW} + z] = sp[(y*{L}+z)*2];
        dp[y*{2*RW} + {RW} + z] = sp[(y*{L}+z)*2+1];
      }}
  }}
}}
static void vconv_out_{L}(const double* restrict src, double* restrict dst){{
  for(int x=0;x<{L};x++){{
    const double* sp = src + (size_t)x*{PLS};
    double* dp = dst + (size_t)x*{2*L*L};
    for(int y=0;y<{L};y++)
      for(int z=0;z<{L};z++){{
        dp[(y*{L}+z)*2] = sp[y*{2*RW} + z];
        dp[(y*{L}+z)*2+1] = sp[y*{2*RW} + {RW} + z];
      }}
  }}
}}
static void vrun_{L}(long B, long m, const double* restrict x0, const double* restrict c, double* restrict o1, double* restrict om){{
  for(long b=0;b<B;b++){{
    long off = b*{2*n3};
    vconv_in_{L}(x0 + off, VBUF_{L});
    vconv_in_{L}(c + off, VCB_{L});
    for(long t=1;t<=m;t++){{
      vstep_{L}(VBUF_{L}, VCB_{L});
      if(t==1) vconv_out_{L}(VBUF_{L}, o1 + off);
    }}
    vconv_out_{L}(VBUF_{L}, om + off);
  }}
}}
""")
    return "".join(parts)
