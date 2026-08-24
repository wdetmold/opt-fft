# experiment: cdft36 variants: baked strides, looped vs straight-line, 1/2 cols
import numpy as np
from genlib import hexd
from genb import Emit, DFTS, pfa_maps, W
from genprime import MAP_MACRO

N1,N2 = 4,9
inm, outm = pfa_maps(N1,N2)
RW = 80

def var_straight(name, ncol, RWc):
    e = Emit('a')
    for j2 in range(N2):
        for cc in range(ncol):
            xs=[]
            for j1 in range(N1):
                off = inm[j1][j2]*RWc + 16*cc
                xs.append((e.v(f"_mm512_load_pd(col + {off})"), e.v(f"_mm512_load_pd(col + {off}+8)")))
            ys = DFTS[N1](e,xs)
            for k1 in range(N1):
                e.raw(f"_mm512_store_pd(SC_{name} + {((j2*N1+k1)*ncol+cc)*16}, {ys[k1][0]});")
                e.raw(f"_mm512_store_pd(SC_{name} + {((j2*N1+k1)*ncol+cc)*16+8}, {ys[k1][1]});")
    e2 = Emit('b')
    for k1 in range(N1):
        for cc in range(ncol):
            xs=[]
            for j2 in range(N2):
                off = ((j2*N1+k1)*ncol+cc)*16
                xs.append((e2.v(f"_mm512_load_pd(SC_{name} + {off})"), e2.v(f"_mm512_load_pd(SC_{name} + {off}+8)")))
            ys = DFTS[N2](e2,xs)
            for k2 in range(N2):
                off = outm[k1][k2]*RWc + 16*cc
                e2.raw(f"_mm512_store_pd(col + {off}, {ys[k2][0]});")
                e2.raw(f"_mm512_store_pd(col + {off}+8, {ys[k2][1]});")
    return f"""
static double SC_{name}[{N1*N2*ncol*16}] ALIGN64;
static void __attribute__((noinline)) {name}(double* restrict col){{
    {{
    {chr(10).join(e.const_decls())}
{e.code()}
    }}
    __asm__ volatile("" ::: "memory");
    {{
    {chr(10).join(e2.const_decls())}
{e2.code()}
    }}
}}"""

def var_looped(name, ncol, RWc):
    # j2-loop with static byte-offset tables
    e = Emit('a')
    for cc in range(ncol):
        xs=[]
        for j1 in range(N1):
            xs.append((e.v(f"*(const __m512d*)(pc + po[{j1}] + {16*cc*8})"), e.v(f"*(const __m512d*)(pc + po[{j1}] + {16*cc*8+64})")))
        ys = DFTS[N1](e,xs)
        for k1 in range(N1):
            e.raw(f"_mm512_store_pd(sc + {(k1*ncol+cc)*16}, {ys[k1][0]});")
            e.raw(f"_mm512_store_pd(sc + {(k1*ncol+cc)*16+8}, {ys[k1][1]});")
    e2 = Emit('b')
    for cc in range(ncol):
        xs=[]
        for j2 in range(N2):
            xs.append((e2.v(f"_mm512_load_pd(sc2 + {(j2*N1*ncol+cc)*16})"), e2.v(f"_mm512_load_pd(sc2 + {(j2*N1*ncol+cc)*16+8})")))
        ys = DFTS[N2](e2,xs)
        for k2 in range(N2):
            e2.raw(f"*(__m512d*)(pc2 + qo[{k2}] + {16*cc*8}) = {ys[k2][0]};")
            e2.raw(f"*(__m512d*)(pc2 + qo[{k2}] + {16*cc*8+64}) = {ys[k2][1]};")
    t1 = ", ".join("{"+", ".join(str(inm[j1][j2]*RWc*8) for j1 in range(N1))+"}" for j2 in range(N2))
    t2 = ", ".join("{"+", ".join(str(outm[k1][k2]*RWc*8) for k2 in range(N2))+"}" for k1 in range(N1))
    return f"""
static double SC_{name}[{N1*N2*ncol*16}] ALIGN64;
static const long PO1_{name}[{N2}][{N1}] = {{ {t1} }};
static const long PO2_{name}[{N1}][{N2}] = {{ {t2} }};
static void __attribute__((noinline)) {name}(double* restrict col){{
    {{
    {chr(10).join(e.const_decls())}
    double* sc = SC_{name};
    const char* pc = (const char*)col;
    for(int j2=0;j2<{N2};j2++){{
        const long* po = PO1_{name}[j2];
{e.code(indent="        ")}
        sc += {N1*ncol*16};
    }}
    }}
    __asm__ volatile("" ::: "memory");
    {{
    {chr(10).join(e2.const_decls())}
    for(int k1=0;k1<{N1};k1++){{
        const double* sc2 = SC_{name} + k1*{ncol*16};
        char* pc2 = (char*)col;
        const long* qo = PO2_{name}[k1];
{e2.code(indent="        ")}
    }}
    }}
}}"""

hdr = '''#include <immintrin.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#define ALIGN64 __attribute__((aligned(64)))
static double now(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+1e-9*ts.tv_nsec; }
static double plane[40*80] ALIGN64;
'''
src = hdr
src += var_straight("s1", 1, RW)
src += var_straight("s2", 2, RW)
src += var_looped("l1", 1, RW)
src += var_looped("l2", 2, RW)
src += '''
int main(){
    for(int i=0;i<40*80;i++) plane[i]=0.01*((i*37)%113)-0.5;
    long N=400000;
    double t;
    for(int rep=0; rep<2; rep++){
#define B(nm, ncol) \\
    for(int r=0;r<1000;r++) nm(plane); \\
    t=now(); for(long r=0;r<N;r++){ nm(plane); } \\
    printf("%-3s: %6.1f cyc per col (pair covers 2)\\n", #nm, (now()-t)*2.6e9/N/ncol);
    B(s1,1) B(s2,2) B(l1,1) B(l2,2)
    printf("---\\n");
    }
    return (int)plane[3];
}
'''
open('expcol.c','w').write(src)
print("written")
