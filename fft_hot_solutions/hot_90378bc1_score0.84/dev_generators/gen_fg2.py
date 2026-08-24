# Generator for L = F*G engines (36 = 4*9, 45 = 5*9): 6D-PFA digit-state engine.
import sys; sys.path.insert(0,'/tmp/g')
from genlib import *
import numpy as np

def emit_transpose8(em, inv, outv):
    """8x8 double transpose: inv/outv lists of 8 __m512d names."""
    t=[None]*8; s=[None]*8
    for i in range(4):
        t[2*i]   = T(em, f'_mm512_unpacklo_pd({inv[2*i]},{inv[2*i+1]})')
        t[2*i+1] = T(em, f'_mm512_unpackhi_pd({inv[2*i]},{inv[2*i+1]})')
    # stage2: 4x4 lane blocks: shuffle_f64x2 imm combos
    for i in range(2):
        for j in range(2):
            s[4*i+j]   = T(em, f'_mm512_shuffle_f64x2({t[4*i+j]},{t[4*i+j+2]},0x88)')
            s[4*i+j+2] = T(em, f'_mm512_shuffle_f64x2({t[4*i+j]},{t[4*i+j+2]},0xdd)')
    for j in range(4):
        em(f'{outv[j]} = _mm512_shuffle_f64x2({s[j]},{s[j+4]},0x88);')
        em(f'{outv[j+4]} = _mm512_shuffle_f64x2({s[j]},{s[j+4]},0xdd);')

def emit_map(em, C, zr, zi, outr, outi):
    """z/(1+|z|) on zmm pair (c already added). ~22 ops. Writes to outr/outi names."""
    half = C.get('C_half',0.5); c15 = C.get('C_15',1.5)
    one = C.get('C_one',1.0); two = C.get('C_two',2.0); tiny = C.get('C_tiny',1e-30)
    t  = T(em, f'_mm512_fmadd_pd({zi},{zi},_mm512_mul_pd({zr},{zr}))')
    t2 = T(em, f'_mm512_max_pd({t},{tiny})')
    p  = T(em, f'_mm512_rsqrt14_pd({t2})')
    h  = T(em, f'_mm512_mul_pd({half},{t2})')
    # NR1: p *= (1.5 - h*p*p)
    q1 = T(em, f'_mm512_mul_pd({p},{p})')
    e1 = T(em, f'_mm512_fnmadd_pd({h},{q1},{c15})')
    p1 = T(em, f'_mm512_mul_pd({p},{e1})')
    q2 = T(em, f'_mm512_mul_pd({p1},{p1})')
    e2 = T(em, f'_mm512_fnmadd_pd({h},{q2},{c15})')
    p2 = T(em, f'_mm512_mul_pd({p1},{e2})')
    r  = T(em, f'_mm512_mul_pd({t2},{p2})')
    # Heron touch-up: r = r + 0.5*p2*(t2 - r*r)
    hp = T(em, f'_mm512_mul_pd({half},{p2})')
    dd = T(em, f'_mm512_fnmadd_pd({r},{r},{t2})')
    r2 = T(em, f'_mm512_fmadd_pd({hp},{dd},{r})')
    d  = T(em, f'_mm512_add_pd({one},{r2})')
    w  = T(em, f'_mm512_rcp14_pd({d})')
    e3 = T(em, f'_mm512_fnmadd_pd({d},{w},{two})')
    w1 = T(em, f'_mm512_mul_pd({w},{e3})')
    e4 = T(em, f'_mm512_fnmadd_pd({d},{w1},{two})')
    w2 = T(em, f'_mm512_mul_pd({w1},{e4})')
    em(f'{outr} = _mm512_mul_pd({zr},{w2});')
    em(f'{outi} = _mm512_mul_pd({zi},{w2});')

def modinv(a, m):
    for x in range(1, m):
        if (a*x) % m == 1: return x
    raise ValueError

import os
SKIP = os.environ.get('SKIP','')
def gen_engine(F, G):
    N = F*G
    FG3 = F**3
    LANES = ((FG3 + 7)//8)*8          # u-lane count padded (64 for 36, 128 for 45)
    LANESP = LANES + 8                 # row stride in doubles (4K-alias skew)
    ROWS = G**3                        # 729
    ROWSP = ((ROWS + 7)//8)*8          # 736
    Ginv = modinv(G % F, F) if F > 1 else 0
    Finv = modinv(F % G, G)
    sF = [ (k*Ginv) % F for k in range(F) ]   # DFT_F output k -> digit slot
    sG = [ (k*Finv) % G for k in range(G) ]
    em = Emitter(); C = Consts()
    name = f'{N}'
    RB = LANES*8                       # row bytes per (re or im) = LANES doubles
    em(f'''
// ================= L = {N} = {F} x {G} PFA digit engine =================
#define N{N} {N}
static double *S{N}_re, *S{N}_im;        // state: ROWSP x LANES doubles
static double *CT{N}_re, *CT{N}_im;      // c tiles: (ROWSP/8) x LANES x 8
static int32_t *SPOS{N};                 // flat spatial -> state pos
static int32_t *TPOS{N};                 // flat spatial -> ctile pos
static double SCR{N}_re[{LANES}*8] __attribute__((aligned(64)));
static double SCR{N}_im[{LANES}*8] __attribute__((aligned(64)));
void init_{N}(void){{
  S{N}_re = huge_alloc({ROWSP}*{LANESP}*8);
  S{N}_im = huge_alloc({ROWSP}*{LANESP}*8);
  CT{N}_re = huge_alloc({ROWSP//8 if ROWS%8==0 else ROWSP//8}*{LANES}*8*8);
  CT{N}_im = huge_alloc({ROWSP//8}*{LANES}*8*8);
  SPOS{N} = (int32_t*)huge_alloc({N**3}*4);
  TPOS{N} = (int32_t*)huge_alloc({N**3}*4);
  memset(S{N}_re, 0, {ROWSP}*{LANESP}*8);
  memset(S{N}_im, 0, {ROWSP}*{LANESP}*8);
  memset(CT{N}_re, 0, {ROWSP//8}*{LANES}*8*8);
  memset(CT{N}_im, 0, {ROWSP//8}*{LANES}*8*8);
  for(int x=0;x<{N};x++)for(int y=0;y<{N};y++)for(int z=0;z<{N};z++){{
    int u1=((x%{F})*{Ginv})%{F}, v1=((x%{G})*{Finv})%{G};
    int u2=((y%{F})*{Ginv})%{F}, v2=((y%{G})*{Finv})%{G};
    int u3=((z%{F})*{Ginv})%{F}, v3=((z%{G})*{Finv})%{G};
    int vlin=(v1*{G}+v2)*{G}+v3, ul=(u1*{F}+u2)*{F}+u3;
    int f=(x*{N}+y)*{N}+z;
    SPOS{N}[f] = vlin*{LANESP} + ul;
    TPOS{N}[f] = (vlin>>3)*{LANES*8} + ul*8 + (vlin&7);
  }}
}}''')
    # ---------------- phase A: three DFT_G passes, lane-chunked ----------------
    # pass over axis a (stride st rows), pencils for all other-row combos
    em(f'static void phaseA_{N}(void){{')
    em(C.loads() if False else '')
    body = Emitter()
    CA = Consts()
    # one codelet: given base row pointer expr and row stride (in doubles), lane offset
    body(f'''
  for(int lc=0;lc<{LANES};lc+=8){{
    // axis v3: stride 1 row; (v1,v2) outer
    for(int o=0;o<{G*G};o++){{
      double* br = S{N}_re + (size_t)o*{G}*{LANESP} + lc;
      double* bi = S{N}_im + (size_t)o*{G}*{LANESP} + lc;''')
    def pencil(body, stride_rows):
        x=[]
        for j in range(G):
            r = T(body, f'_mm512_load_pd(br+{j*stride_rows}*{LANESP})')
            i = T(body, f'_mm512_load_pd(bi+{j*stride_rows}*{LANESP})')
            x.append((r,i))
        X = dft9(body, CA, x)
        for k in range(G):
            body(f'_mm512_store_pd(br+{sG[k]*stride_rows}*{LANESP}, {X[k][0]});')
            body(f'_mm512_store_pd(bi+{sG[k]*stride_rows}*{LANESP}, {X[k][1]});')
    p1 = Emitter(); pencil(p1, 1); body(p1.out()); body('}')
    body(f'''
    // axis v2: stride {G} rows; outer (v1, v3)
    for(int v1=0;v1<{G};v1++)for(int v3=0;v3<{G};v3++){{
      double* br = S{N}_re + ((size_t)v1*{G*G}+v3)*{LANESP} + lc;
      double* bi = S{N}_im + ((size_t)v1*{G*G}+v3)*{LANESP} + lc;''')
    p2 = Emitter(); pencil(p2, G); body(p2.out()); body('}')
    body(f'''
    // axis v1: stride {G*G} rows; outer (v2, v3)
    for(int o=0;o<{G*G};o++){{
      double* br = S{N}_re + (size_t)o*{LANESP} + lc;
      double* bi = S{N}_im + (size_t)o*{LANESP} + lc;''')
    p3 = Emitter(); pencil(p3, G*G); body(p3.out()); body('}')
    body('}')
    em(CA.loads())
    em(body.out())
    em('}')
    # ---------------- phase B ----------------
    CB = Consts()
    bb = Emitter()
    nch = ROWSP//8
    bb(f'''
  for(int vc=0;vc<{nch};vc++){{
    double* pr = S{N}_re + (size_t)vc*8*{LANESP};
    double* pi = S{N}_im + (size_t)vc*8*{LANESP};
    const double* cr = CT{N}_re + (size_t)vc*{LANES*8};
    const double* ci = CT{N}_im + (size_t)vc*{LANES*8};
    // transpose in: scratch[u][lane] = state[row r=vc*8+lane][u]
    for(int ub=0;ub<{LANES//8};ub++){{''')
    for arr in ('re','im'):
        src = 'pr' if arr=='re' else 'pi'
        tin = [T(bb, f'_mm512_load_pd({src}+{r}*{LANESP}+ub*8)') for r in range(8)]
        outv = [f'o{arr}{j}' for j in range(8)]
        for o in outv: bb(f'__m512d {o};')
        emit_transpose8(bb, tin, outv)
        for j in range(8):
            bb(f'_mm512_store_pd(SCR{N}_{arr}+(ub*8+{j})*8, {outv[j]});')
    bb('}')
    # DFT_F passes over scratch rows (each row = 8 doubles = 1 zmm)
    dftF = {4:dft4, 5:dft5}[F]
    def fpass(bb, stride, fuse_map):
        x=[]
        for j in range(F):
            r = T(bb, f'_mm512_load_pd(SCR{N}_re+({j}*{stride}+base)*8)')
            i = T(bb, f'_mm512_load_pd(SCR{N}_im+({j}*{stride}+base)*8)')
            x.append((r,i))
        X = dftF(bb, CB, x)
        for k in range(F):
            tgt = sF[k]*stride
            if not fuse_map:
                bb(f'_mm512_store_pd(SCR{N}_re+({tgt}+base)*8, {X[k][0]});')
                bb(f'_mm512_store_pd(SCR{N}_im+({tgt}+base)*8, {X[k][1]});')
            else:
                zr = T(bb, f'_mm512_add_pd({X[k][0]}, _mm512_load_pd(cr+({tgt}+base)*8))')
                zi = T(bb, f'_mm512_add_pd({X[k][1]}, _mm512_load_pd(ci+({tgt}+base)*8))')
                bb(f'__m512d mr{k}, mi{k};')
                emit_map(bb, CB, zr, zi, f'mr{k}', f'mi{k}')
                bb(f'_mm512_store_pd(SCR{N}_re+({tgt}+base)*8, mr{k});')
                bb(f'_mm512_store_pd(SCR{N}_im+({tgt}+base)*8, mi{k});')
    # pass u3: stride 1, base = (u1*F+u2)*F
    if 'u3' not in SKIP:
        bb(f'for(int o=0;o<{F*F};o++){{ int base=o*{F};')
        fpass(bb, 1, False); bb('}')
    # pass u2: stride F, base = u1*F*F + u3
    if 'u2' not in SKIP:
        bb(f'for(int u1=0;u1<{F};u1++)for(int u3=0;u3<{F};u3++){{ int base=u1*{F*F}+u3;')
        fpass(bb, F, False); bb('}')
    # pass u1: stride F*F, base = u2*F+u3, fused c+map
    if 'u1' not in SKIP:
        bb(f'for(int o=0;o<{F*F};o++){{ int base=o;')
        fpass(bb, F*F, 'map' not in SKIP); bb('}')
    # transpose back
    bb(f'for(int ub=0;ub<{LANES//8};ub++){{')
    for arr in ('re','im'):
        dst = 'pr' if arr=='re' else 'pi'
        tin = [T(bb, f'_mm512_load_pd(SCR{N}_{arr}+(ub*8+{j})*8)') for j in range(8)]
        outv = [f'b{arr}{j}' for j in range(8)]
        for o in outv: bb(f'__m512d {o};')
        emit_transpose8(bb, tin, outv)
        for r in range(8):
            bb(f'_mm512_store_pd({dst}+{r}*{LANESP}+ub*8, {outv[r]});')
    bb('}')
    bb('}')
    em(f'static void phaseB_{N}(void){{')
    em(CB.loads())
    em(bb.out())
    em('}')
    # ---------------- conversions + driver ----------------
    em(f'''
static void convin_{N}(const double* x0){{
  for(int f=0;f<{N**3};f++){{
    int sp = SPOS{N}[f];
    S{N}_re[sp] = x0[2*f];
    S{N}_im[sp] = x0[2*f+1];
  }}
}}
static void convc_{N}(const double* c){{
  for(int f=0;f<{N**3};f++){{
    int tp = TPOS{N}[f];
    CT{N}_re[tp] = c[2*f];
    CT{N}_im[tp] = c[2*f+1];
  }}
}}
static void convout_{N}(double* out){{
  for(int f=0;f<{N**3};f++){{
    int sp = SPOS{N}[f];
    out[2*f]   = S{N}_re[sp];
    out[2*f+1] = S{N}_im[sp];
  }}
}}
uint64_t bench_{N}(int which, long reps){{
  uint64_t t0 = __rdtsc();
  for(long r=0;r<reps;r++){{ if(which==0) phaseA_{N}(); else phaseB_{N}(); }}
  return __rdtsc() - t0;
}}
void run_{N}(const double* x0, const double* c, double* one, double* fin, long B, long m){{
  for(long b=0;b<B;b++){{
    convin_{N}(x0 + b*2*{N**3});
    convc_{N}(c + b*2*{N**3});
    for(long s=0;s<m;s++){{
      phaseA_{N}();
      phaseB_{N}();
      if(s==0) convout_{N}(one + b*2*{N**3});
    }}
    convout_{N}(fin + b*2*{N**3});
  }}
}}''')
    pre = Emitter()
    C.decl(pre); CA.decl(pre); CB.decl(pre)
    return pre.out() + em.out()

PRELUDE = r'''
#include <immintrin.h>
#include <x86intrin.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <sys/mman.h>
static double* huge_alloc(size_t bytes){
  size_t sz = (bytes + (2u<<20) - 1) & ~(size_t)((2u<<20)-1);
  void* p = mmap(0, sz, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
  if(p == MAP_FAILED){ abort(); }
  madvise(p, sz, MADV_HUGEPAGE);
  memset(p, 0, sz);
  return (double*)p;
}
'''

if __name__ == '__main__':
    src = PRELUDE
    src += gen_engine(4, 9)
    src += gen_engine(5, 9)
    open('/tmp/g/impl_fg.c','w').write(src)
    print('wrote impl_fg.c', len(src))
