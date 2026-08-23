import numpy as np
from genlib import E, trig, hexd
from genprime import emit_prime_dft, MAP_MACRO

PRELUDE = r'''
#include <immintrin.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#define ALIGN64 __attribute__((aligned(64)))
static double* alloc_arena(long bytes){
    long HP = (long)2<<20;
    bytes = (bytes + HP - 1) & ~(HP-1);
    void* p = mmap(0, bytes + HP, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    if(p == MAP_FAILED) { void* q = aligned_alloc(64, bytes); memset(q,0,bytes); return (double*)q; }
    char* a = (char*)(((unsigned long)p + HP - 1) & ~((unsigned long)HP-1));
    if(a > (char*)p) munmap(p, a - (char*)p);
    long tail = ((char*)p + bytes + HP) - (a + bytes);
    if(tail > 0) munmap(a + bytes, tail);
    madvise(a, bytes, MADV_HUGEPAGE);
    memset(a, 0, bytes);
    return (double*)a;
}
static __m512d V_ONE, V_HALF, V_TINY;
__attribute__((constructor)) static void init_consts(void){
    V_ONE = _mm512_set1_pd(1.0); V_HALF = _mm512_set1_pd(0.5); V_TINY = _mm512_set1_pd(1e-30);
}
#define TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7) do{ \
    __m512d _t0=_mm512_unpacklo_pd(r0,r1), _t1=_mm512_unpackhi_pd(r0,r1); \
    __m512d _t2=_mm512_unpacklo_pd(r2,r3), _t3=_mm512_unpackhi_pd(r2,r3); \
    __m512d _t4=_mm512_unpacklo_pd(r4,r5), _t5=_mm512_unpackhi_pd(r4,r5); \
    __m512d _t6=_mm512_unpacklo_pd(r6,r7), _t7=_mm512_unpackhi_pd(r6,r7); \
    __m512d _u0=_mm512_shuffle_f64x2(_t0,_t2,0x44), _u1=_mm512_shuffle_f64x2(_t4,_t6,0x44); \
    __m512d _u2=_mm512_shuffle_f64x2(_t0,_t2,0xee), _u3=_mm512_shuffle_f64x2(_t4,_t6,0xee); \
    __m512d _u4=_mm512_shuffle_f64x2(_t1,_t3,0x44), _u5=_mm512_shuffle_f64x2(_t5,_t7,0x44); \
    __m512d _u6=_mm512_shuffle_f64x2(_t1,_t3,0xee), _u7=_mm512_shuffle_f64x2(_t5,_t7,0xee); \
    o0=_mm512_shuffle_f64x2(_u0,_u1,0x88); o2=_mm512_shuffle_f64x2(_u0,_u1,0xdd); \
    o4=_mm512_shuffle_f64x2(_u2,_u3,0x88); o6=_mm512_shuffle_f64x2(_u2,_u3,0xdd); \
    o1=_mm512_shuffle_f64x2(_u4,_u5,0x88); o3=_mm512_shuffle_f64x2(_u4,_u5,0xdd); \
    o5=_mm512_shuffle_f64x2(_u6,_u7,0x88); o7=_mm512_shuffle_f64x2(_u6,_u7,0xdd); \
}while(0)
'''

def gen_conv(L):
    """conv_in_L(srcs[8] base ptrs, double* G) and conv_out_L(G, dst[8])"""
    L3 = L*L*L
    nb = L3 // 4
    tail = L3 % 4
    s = f"""
static void conv_in_{L}(const double* const* src, double* restrict G){{
    for(long e=0; e+4<={L3}; e+=4){{
        __m512d r0=_mm512_loadu_pd(src[0]+2*e), r1=_mm512_loadu_pd(src[1]+2*e);
        __m512d r2=_mm512_loadu_pd(src[2]+2*e), r3=_mm512_loadu_pd(src[3]+2*e);
        __m512d r4=_mm512_loadu_pd(src[4]+2*e), r5=_mm512_loadu_pd(src[5]+2*e);
        __m512d r6=_mm512_loadu_pd(src[6]+2*e), r7=_mm512_loadu_pd(src[7]+2*e);
        __m512d o0,o1,o2,o3,o4,o5,o6,o7;
        TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
        double* g = G + e*16;
        _mm512_store_pd(g,o0); _mm512_store_pd(g+8,o1);
        _mm512_store_pd(g+16,o2); _mm512_store_pd(g+24,o3);
        _mm512_store_pd(g+32,o4); _mm512_store_pd(g+40,o5);
        _mm512_store_pd(g+48,o6); _mm512_store_pd(g+56,o7);
    }}
    for(long e={L3 - tail}; e<{L3}; e++)
        for(int v=0; v<8; v++){{ G[e*16+v] = src[v][2*e]; G[e*16+8+v] = src[v][2*e+1]; }}
}}
static void conv_out_{L}(const double* restrict G, double* const* dst){{
    for(long e=0; e+4<={L3}; e+=4){{
        const double* g = G + e*16;
        __m512d r0=_mm512_load_pd(g),    r1=_mm512_load_pd(g+8);
        __m512d r2=_mm512_load_pd(g+16), r3=_mm512_load_pd(g+24);
        __m512d r4=_mm512_load_pd(g+32), r5=_mm512_load_pd(g+40);
        __m512d r6=_mm512_load_pd(g+48), r7=_mm512_load_pd(g+56);
        __m512d o0,o1,o2,o3,o4,o5,o6,o7;
        TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
        _mm512_storeu_pd(dst[0]+2*e,o0); _mm512_storeu_pd(dst[1]+2*e,o1);
        _mm512_storeu_pd(dst[2]+2*e,o2); _mm512_storeu_pd(dst[3]+2*e,o3);
        _mm512_storeu_pd(dst[4]+2*e,o4); _mm512_storeu_pd(dst[5]+2*e,o5);
        _mm512_storeu_pd(dst[6]+2*e,o6); _mm512_storeu_pd(dst[7]+2*e,o7);
    }}
    for(long e={L3 - tail}; e<{L3}; e++)
        for(int v=0; v<8; v++){{ dst[v][2*e] = G[e*16+v]; dst[v][2*e+1] = G[e*16+8+v]; }}
}}
"""
    return s

def gen_prime_run(p):
    L2, L3 = p*p, p*p*p
    s = emit_prime_dft(p, fuse_map=False)
    s += "\n" + emit_prime_dft(p, fuse_map=True)
    s += f"""
static double* G_{p} = 0;
static double* CP_{p} = 0;
static void step_{p}(double* restrict G, const double* restrict CP){{
    // pass1: per x-plane: z-DFTs then y-DFTs (L1-resident plane)
    for(int x=0; x<{p}; x++){{
        double* pl = G + (long)x*{L2}*16;
        for(int y=0; y<{p}; y++) dft{p}(pl + (long)y*{p}*16, 16, pl + (long)y*{p}*16, 16);
        for(int z=0; z<{p}; z++) dft{p}(pl + (long)z*16, {p}*16, pl + (long)z*16, {p}*16);
    }}
    // pass2: x-DFT + c + map
    for(int y=0; y<{p}; y++)
        for(int z=0; z<{p}; z++){{
            long off = ((long)y*{p}+z)*16;
            dft{p}m(G + off, {L2}*16, G + off, {L2}*16, CP + off, {L2}*16);
        }}
}}
void run_{p}_g(const double* x0, const double* c, double* out1, double* outm, long B, long m){{
    if(!G_{p}){{ G_{p} = alloc_arena({L3}*16*8); CP_{p} = alloc_arena({L3}*16*8); }}
    long G8 = B/8;
    for(long g=0; g<G8; g++){{
        const double* sx[8]; const double* sc[8]; double* d1[8]; double* dm[8];
        for(int v=0; v<8; v++){{
            long off = (g*8+v)*(long){L3}*2;
            sx[v] = x0+off; sc[v] = c+off; d1[v] = out1+off; dm[v] = outm+off;
        }}
        conv_in_{p}(sx, G_{p});
        conv_in_{p}(sc, CP_{p});
        for(long t=0; t<m; t++){{
            step_{p}(G_{p}, CP_{p});
            if(t==0 && m>1) conv_out_{p}(G_{p}, d1);
        }}
        conv_out_{p}(G_{p}, dm);
        if(m==1) for(int v=0; v<8; v++) memcpy(d1[v], dm[v], {L3}*16);
    }}
}}
"""
    return s

def build(ps=(13,17,23)):
    parts = [PRELUDE, MAP_MACRO]
    for p in ps:
        parts.append(gen_conv(p))
        parts.append(gen_prime_run(p))
    return "\n".join(parts)

if __name__ == "__main__":
    src = build()
    open("v8prime.c","w").write(src)
    print("wrote v8prime.c", len(src))
