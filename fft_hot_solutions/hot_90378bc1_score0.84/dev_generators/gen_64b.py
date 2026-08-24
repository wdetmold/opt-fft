# L=64 engine: digits n = 8*r + l per axis. State P: [rlin 512 rows][l-lanes 512(+8 pad stride)]
# Step: phaseA: DFT8 over r1,r2,r3 (vertical, chunk-column-tiled) + twiddle W64^{l_i * o_i}
#       phaseB: per 8-row chunk: transpose-in (fused with first lane-digit DFT8) ->
#               scratch [l 512 rows][8 r-lane cols]; DFT8 over l2, l1(+c+map);
#               strided store into OTHER buffer as rows=old-lane-digit-outputs.
# Buffers ping-pong each step; same code both parities.
import sys; sys.path.insert(0,'/tmp/g')
from genlib import *
import numpy as np

N = 64
ROWS = 512
LANES = 512
LP = LANES + 8          # row stride doubles
def dft8(em, C, x):
    """8-point complex DFT, natural in, natural out. 52 port05 ops."""
    s = C.get('C_r2', 0.7071067811865475244008443621048490392848359376884740365883398689)
    # radix-2 DIT style explicit
    a0r=ADD(em,x[0][0],x[4][0]); a0i=ADD(em,x[0][1],x[4][1])
    a1r=SUB(em,x[0][0],x[4][0]); a1i=SUB(em,x[0][1],x[4][1])
    a2r=ADD(em,x[2][0],x[6][0]); a2i=ADD(em,x[2][1],x[6][1])
    a3r=SUB(em,x[2][0],x[6][0]); a3i=SUB(em,x[2][1],x[6][1])
    a4r=ADD(em,x[1][0],x[5][0]); a4i=ADD(em,x[1][1],x[5][1])
    a5r=SUB(em,x[1][0],x[5][0]); a5i=SUB(em,x[1][1],x[5][1])
    a6r=ADD(em,x[3][0],x[7][0]); a6i=ADD(em,x[3][1],x[7][1])
    a7r=SUB(em,x[3][0],x[7][0]); a7i=SUB(em,x[3][1],x[7][1])
    # DFT4 on evens (a0,a2 | a1,a3 with -i twist)
    b0r=ADD(em,a0r,a2r); b0i=ADD(em,a0i,a2i)
    b2r=SUB(em,a0r,a2r); b2i=SUB(em,a0i,a2i)
    b1r=ADD(em,a1r,a3i); b1i=SUB(em,a1i,a3r)   # a1 - i a3
    b3r=SUB(em,a1r,a3i); b3i=ADD(em,a1i,a3r)   # a1 + i a3
    # DFT4 on odds
    c0r=ADD(em,a4r,a6r); c0i=ADD(em,a4i,a6i)
    c2r=SUB(em,a4r,a6r); c2i=SUB(em,a4i,a6i)
    c1r=ADD(em,a5r,a7i); c1i=SUB(em,a5i,a7r)
    c3r=SUB(em,a5r,a7i); c3i=ADD(em,a5i,a7r)
    # twiddles: W8^0=1, W8^1=s(1-i), W8^2=-i, W8^3=-s(1+i)
    # k=0: X0 = b0 + c0 ; X4 = b0 - c0
    X0r=ADD(em,b0r,c0r); X0i=ADD(em,b0i,c0i)
    X4r=SUB(em,b0r,c0r); X4i=SUB(em,b0i,c0i)
    # k=1: t = W8 c1 = s*(c1r+c1i) + i s*(c1i-c1r)
    t1r=T(em,f'_mm512_mul_pd({s},_mm512_add_pd({c1r},{c1i}))')
    t1i=T(em,f'_mm512_mul_pd({s},_mm512_sub_pd({c1i},{c1r}))')
    X1r=ADD(em,b1r,t1r); X1i=ADD(em,b1i,t1i)
    X5r=SUB(em,b1r,t1r); X5i=SUB(em,b1i,t1i)
    # k=2: t = -i c2 = (c2i, -c2r)
    X2r=ADD(em,b2r,c2i); X2i=SUB(em,b2i,c2r)
    X6r=SUB(em,b2r,c2i); X6i=ADD(em,b2i,c2r)
    # k=3: t = W8^3 c3 = -s(c3r - c3i) + i(-s)(c3i + c3r) = s*(c3i-c3r) - i s*(c3r+c3i)
    t3r=T(em,f'_mm512_mul_pd({s},_mm512_sub_pd({c3i},{c3r}))')
    t3i=T(em,f'_mm512_mul_pd({s},_mm512_add_pd({c3r},{c3i}))')
    X3r=ADD(em,b3r,t3r); X3i=SUB(em,b3i,t3i)
    X7r=SUB(em,b3r,t3r); X7i=ADD(em,b3i,t3i)
    return [(X0r,X0i),(X1r,X1i),(X2r,X2i),(X3r,X3i),(X4r,X4i),(X5r,X5i),(X6r,X6i),(X7r,X7i)]

from gen_fg import emit_map, emit_transpose8

em = Emitter()
em(r'''
// ==================== L = 64 engine ====================
static double *S64A_re, *S64A_im, *S64B_re, *S64B_im;   // ping-pong states
static double *C64P_re, *C64P_im;  // c in P layout (consumed when writing P... see driver)
static int32_t *SPOS64;            // flat -> state pos (same formula both parities)
static double TW3_re[8*8] __attribute__((aligned(64)));  // axis3 twiddle vectors [o3][l3]
static double TW3_im[8*8] __attribute__((aligned(64)));
static double TW12_re[8*8] __attribute__((aligned(64))); // scalar twiddle W64^{d*o} [d][o]
static double TW12_im[8*8] __attribute__((aligned(64)));
static double SCR64_re[512*8] __attribute__((aligned(64)));
static double SCR64_im[512*8] __attribute__((aligned(64)));
void init_64(void){
  S64A_re = huge_alloc(512*%LP%*8); S64A_im = huge_alloc(512*%LP%*8);
  S64B_re = huge_alloc(512*%LP%*8); S64B_im = huge_alloc(512*%LP%*8);
  C64P_re = huge_alloc(512*%LP%*8); C64P_im = huge_alloc(512*%LP%*8);
  SPOS64 = (int32_t*)huge_alloc(262144*4);
  for(int o=0;o<8;o++)for(int l=0;l<8;l++){
    long double ang = -2.0L*3.14159265358979323846264338327950288L*((o*l)&63)/64.0L;
    TW3_re[o*8+l] = (double)cosl(ang); TW3_im[o*8+l] = (double)sinl(ang);
    TW12_re[o*8+l] = (double)cosl(ang); TW12_im[o*8+l] = (double)sinl(ang);
  }
  for(int x=0;x<64;x++)for(int y=0;y<64;y++)for(int z=0;z<64;z++){
    int r1=x>>3, l1=x&7, r2=y>>3, l2=y&7, r3=z>>3, l3=z&7;
    int rlin=(r1*8+r2)*8+r3, llin=(l1*8+l2)*8+l3;
    SPOS64[(x*64+y)*64+z] = rlin*%LP% + llin;
  }
}
'''.replace('%LP%', str(LP)))

# ---------------- phase A ----------------
# chunk-columns: lc in 0..63 (8 lanes each); within column do axis r3 (stride 1 row),
# r2 (stride 8), r1 (stride 64), each DFT8 + twiddle.
CA = Consts()
pa = Emitter()
pa(f'''
static void phaseA_64(double* re, double* im){{
%CONSTS%
  for(int lc=0;lc<64;lc++){{
    int l3base = 0;        // l3 = lane index within chunk
    int l2 = lc & 7, l1 = lc >> 3;
    // ---- axis r3: pencils rows 8*o..8*o+7 (o = r1*8+r2) ----
    for(int o=0;o<64;o++){{
      double* br = re + (size_t)(o*8)*{LP} + lc*8;
      double* bi = im + (size_t)(o*8)*{LP} + lc*8;
      for(int pf=0;pf<8;pf++){{ _mm_prefetch((const char*)(br+(pf+8)*{LP}), _MM_HINT_T0); _mm_prefetch((const char*)(bi+(pf+8)*{LP}), _MM_HINT_T0); }}''')
def pencilA(pa, stride, twsel):
    x=[]
    for j in range(8):
        r = T(pa, f'_mm512_load_pd(br+{j*stride}*{LP})')
        i = T(pa, f'_mm512_load_pd(bi+{j*stride}*{LP})')
        x.append((r,i))
    X = dft8(pa, CA, x)
    # twiddle + store
    for k in range(8):
        xr, xi = X[k]
        if k > 0:
            if twsel == 3:
                wr = T(pa, f'_mm512_load_pd(TW3_re+{k}*8)')
                wi = T(pa, f'_mm512_load_pd(TW3_im+{k}*8)')
            else:
                d = 'l2' if twsel == 2 else 'l1'
                wr = T(pa, f'_mm512_set1_pd(TW12_re[{d}*8+{k}])')
                wi = T(pa, f'_mm512_set1_pd(TW12_im[{d}*8+{k}])')
            rr = MUL(pa, wr, xr); ri = MUL(pa, wr, xi)
            nr = FNMA(pa, wi, xi, rr); ni = FMA(pa, wi, xr, ri)
            xr, xi = nr, ni
        pa(f'_mm512_store_pd(br+{k*stride}*{LP}, {xr});')
        pa(f'_mm512_store_pd(bi+{k*stride}*{LP}, {xi});')
pencilA(pa, 1, 3)
pa('}')
pa(f'''
    // ---- axis r2: stride 8 rows; outer r1 (8), r3 (8) ----
    for(int r1=0;r1<8;r1++)for(int r3=0;r3<8;r3++){{
      double* br = re + (size_t)(r1*64+r3)*{LP} + lc*8;
      double* bi = im + (size_t)(r1*64+r3)*{LP} + lc*8;''')
pencilA(pa, 8, 2)
pa('}')
pa(f'''
    // ---- axis r1: stride 64 rows; outer o=(r2,r3) ----
    for(int o=0;o<64;o++){{
      double* br = re + (size_t)o*{LP} + lc*8;
      double* bi = im + (size_t)o*{LP} + lc*8;''')
pencilA(pa, 64, 1)
pa('}')
pa('}')
pa('}')
patext = pa.out().replace('%CONSTS%', CA.loads())
em(patext)

# ---------------- phase B ----------------
CB = Consts()
pb = Emitter()
pb(f'''
static void phaseB_64(const double* re, const double* im, double* ore, double* oim,
                      const double* cre, const double* cim){{
%CONSTS%
  for(int rc=0;rc<64;rc++){{
    const double* pr = re + (size_t)rc*8*{LP};
    const double* pi = im + (size_t)rc*8*{LP};
    // transpose-in fused with l3-axis DFT8 (8 consecutive scratch rows = one l3 pencil)
    for(int ub=0;ub<64;ub++){{
      _mm_prefetch((const char*)(pr+(ub&7)*{LP}+ub*8+512), _MM_HINT_T0);
      _mm_prefetch((const char*)(pi+(ub&7)*{LP}+ub*8+512), _MM_HINT_T0);''')
xv = []
for arr in ('re','im'):
    src = 'pr' if arr=='re' else 'pi'
    tin = [T(pb, f'_mm512_load_pd({src}+{r}*{LP}+ub*8)') for r in range(8)]
    outv = [f'q{arr}{j}' for j in range(8)]
    for o in outv: pb(f'__m512d {o};')
    emit_transpose8(pb, tin, outv)
for j in range(8):
    xv.append((f'qre{j}', f'qim{j}'))
X = dft8(pb, CB, xv)
for k in range(8):
    pb(f'_mm512_store_pd(SCR64_re+(ub*8+{k})*8, {X[k][0]});')
    pb(f'_mm512_store_pd(SCR64_im+(ub*8+{k})*8, {X[k][1]});')
pb('}')
# l2 pass: stride 8 scratch rows; outer l1(8) x l3(8)
pb(f'for(int l1=0;l1<8;l1++)for(int l3=0;l3<8;l3++){{ int base=l1*64+l3;')
x2=[]
for j in range(8):
    r = T(pb, f'_mm512_load_pd(SCR64_re+({j}*8+base)*8)')
    i = T(pb, f'_mm512_load_pd(SCR64_im+({j}*8+base)*8)')
    x2.append((r,i))
X2 = dft8(pb, CB, x2)
for k in range(8):
    pb(f'_mm512_store_pd(SCR64_re+({k}*8+base)*8, {X2[k][0]});')
    pb(f'_mm512_store_pd(SCR64_im+({k}*8+base)*8, {X2[k][1]});')
pb('}')
# l1 pass (stride 64) + c + map + strided store: out row = k*... new row digit = output index k1 of l1-DFT? 
# Careful: scratch rows llin=(l1*8+l2)*8+l3; output of l-DFTs: digits (o1,o2,o3) -> new state row should be
# olin=(o1*8+o2)*8+o3 and lanes = the transposed r-output digit chunk (rc gives o-chunk lanes: lanes hold the
# 8 phaseA outputs within row-chunk rc: new lane digit = old row digit OUTPUT... lanes of scratch = old rows
# rc*8..rc*8+7 AFTER phaseA = output digits already in row position: lane j of scratch = old row rc*8+j, which
# holds phaseA output digit-combo (o of r-DFTs) = row index in P of "k8" outputs. So new LANE block = rc.
pb(f'for(int o=0;o<64;o++){{ int base=o; ')
x3=[]
for j in range(8):
    r = T(pb, f'_mm512_load_pd(SCR64_re+({j}*64+base)*8)')
    i = T(pb, f'_mm512_load_pd(SCR64_im+({j}*64+base)*8)')
    x3.append((r,i))
X3 = dft8(pb, CB, x3)
subs=[]
for k in range(8):
    bu = Emitter()
    # new row = (k*64+base) interpreting (o1=k, rest=base)? base = l2*8+l3 outputs (o2,o3) stored in place:
    # after l3 pass outputs stored at same l3 slot => actually we must track: l3-pass wrote output digit o3 to
    # slot index o3 (natural store) - yes stores X[k] at slot k. Same l2. So after 3 passes slot (a*8+b)*8+c holds
    # output digit combo (a,b,c) = new row index. Store to new state row (k*64+base).
    zr = T(bu, f'_mm512_add_pd({X3[k][0]}, _mm512_load_pd(cre + (size_t)({k}*64+base)*{LP} + rc*8))')
    zi = T(bu, f'_mm512_add_pd({X3[k][1]}, _mm512_load_pd(cim + (size_t)({k}*64+base)*{LP} + rc*8))')
    bu(f'__m512d fr{k}, fi{k};')
    emit_map(bu, CB, zr, zi, f'fr{k}', f'fi{k}')
    bu(f'_mm512_stream_pd(ore + (size_t)({k}*64+base)*{LP} + rc*8, fr{k});')
    bu(f'_mm512_stream_pd(oim + (size_t)({k}*64+base)*{LP} + rc*8, fi{k});')
    subs.append(bu.out().split(chr(10)))
mx = max(len(s) for s in subs)
for row in range(mx):
    for s in subs:
        if row < len(s): pb(s[row])
pb('}')
pb('}')
pb('}')
em(pb.out().replace('%CONSTS%', CB.loads()))

# ---------------- driver ----------------
em(f'''
static void convin_64(const double* x0, double* sre, double* sim){{
  for(int f=0;f<262144;f++){{ int sp=SPOS64[f]; sre[sp]=x0[2*f]; sim[sp]=x0[2*f+1]; }}
}}
static void convout_64(const double* sre, const double* sim, double* out){{
  for(int f=0;f<262144;f++){{ int sp=SPOS64[f]; out[2*f]=sre[sp]; out[2*f+1]=sim[sp]; }}
}}
uint64_t bench_64(int which, long reps){{
  uint64_t t0=__rdtsc();
  for(long r=0;r<reps;r++){{
    if(which==0) phaseA_64(S64A_re,S64A_im);
    else phaseB_64(S64A_re,S64A_im,S64B_re,S64B_im,C64P_re,C64P_im);
  }}
  return __rdtsc()-t0;
}}
void run_64(const double* x0, const double* c, double* one, double* fin, long B, long m){{
  for(long b=0;b<B;b++){{
    convin_64(x0 + b*2*262144, S64A_re, S64A_im);
    convin_64(c + b*2*262144, C64P_re, C64P_im);
    double *cur_r=S64A_re, *cur_i=S64A_im, *oth_r=S64B_re, *oth_i=S64B_im;
    for(long s=0;s<m;s++){{
      phaseA_64(cur_r, cur_i);
      phaseB_64(cur_r, cur_i, oth_r, oth_i, C64P_re, C64P_im);
      double* t;
      t=cur_r; cur_r=oth_r; oth_r=t;
      t=cur_i; cur_i=oth_i; oth_i=t;
      if(s==0) convout_64(cur_r, cur_i, one + b*2*262144);
    }}
    convout_64(cur_r, cur_i, fin + b*2*262144);
  }}
}}''')

PRELUDE = open('/tmp/g/impl_fg.c').read().split('// =================')[0]  # reuse prelude
src = PRELUDE + '#include <math.h>\n#include <x86intrin.h>\n' + em.out()
open('/tmp/g/impl_64.c','w').write(src)
print('wrote impl_64.c', len(src))
