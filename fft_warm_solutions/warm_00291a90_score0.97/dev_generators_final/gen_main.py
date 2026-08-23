PAIR_MN = {6}
import numpy as np
from genlib import *
from gen_a import gen_composite, gen_prime

PRELUDE = r'''
#include <immintrin.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#define ALIGN64 __attribute__((aligned(64)))
static double* alloc_huge(long bytes){
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
static long stagger_ctr = 0;
static double* alloc_huge_st(long bytes){
    long st = ((stagger_ctr++ % 29) + 1) * 4672;
    double* p = alloc_huge(bytes + st + 8192);
    return (double*)((char*)p + st);
}
// 8x8 double transpose
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

def gen_conv(L, PS):
    L2, L3 = L*L, L*L*L
    PSZ = PS*16
    return f"""
static void convin_{L}(const double* const* src, double* G){{
    for(int i=0;i<{L};i++){{
        double* gp = G + (long)i*{PSZ};
        long base = (long)i*{L2};
        long e=0;
        for(; e+4<={L2}; e+=4){{
            __m512d r0=_mm512_loadu_pd(src[0]+2*(base+e)), r1=_mm512_loadu_pd(src[1]+2*(base+e));
            __m512d r2=_mm512_loadu_pd(src[2]+2*(base+e)), r3=_mm512_loadu_pd(src[3]+2*(base+e));
            __m512d r4=_mm512_loadu_pd(src[4]+2*(base+e)), r5=_mm512_loadu_pd(src[5]+2*(base+e));
            __m512d r6=_mm512_loadu_pd(src[6]+2*(base+e)), r7=_mm512_loadu_pd(src[7]+2*(base+e));
            __m512d o0,o1,o2,o3,o4,o5,o6,o7;
            TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
            _mm512_store_pd(gp+e*16+0,  o0); _mm512_store_pd(gp+e*16+8,  o1);
            _mm512_store_pd(gp+e*16+16, o2); _mm512_store_pd(gp+e*16+24, o3);
            _mm512_store_pd(gp+e*16+32, o4); _mm512_store_pd(gp+e*16+40, o5);
            _mm512_store_pd(gp+e*16+48, o6); _mm512_store_pd(gp+e*16+56, o7);
        }}
        for(; e<{L2}; e++){{
            for(int v=0;v<8;v++){{ gp[e*16+v] = src[v][2*(base+e)]; gp[e*16+8+v] = src[v][2*(base+e)+1]; }}
        }}
    }}
}}
static void convout_{L}(const double* G, double* const* dst, int nv){{
    for(int i=0;i<{L};i++){{
        const double* gp = G + (long)i*{PSZ};
        long base = (long)i*{L2};
        long e=0;
        for(; e+4<={L2}; e+=4){{
            __m512d r0=_mm512_load_pd(gp+e*16+0),  r1=_mm512_load_pd(gp+e*16+8);
            __m512d r2=_mm512_load_pd(gp+e*16+16), r3=_mm512_load_pd(gp+e*16+24);
            __m512d r4=_mm512_load_pd(gp+e*16+32), r5=_mm512_load_pd(gp+e*16+40);
            __m512d r6=_mm512_load_pd(gp+e*16+48), r7=_mm512_load_pd(gp+e*16+56);
            __m512d o0,o1,o2,o3,o4,o5,o6,o7;
            TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
            __m512d oo[8] = {{o0,o1,o2,o3,o4,o5,o6,o7}};
            for(int v=0;v<nv;v++) _mm512_storeu_pd(dst[v]+2*(base+e), oo[v]);
        }}
        for(; e<{L2}; e++){{
            for(int v=0;v<nv;v++){{ dst[v][2*(base+e)] = gp[e*16+v]; dst[v][2*(base+e)+1] = gp[e*16+8+v]; }}
        }}
    }}
}}
"""

DRIVER_RUN = """void run_{L}(const double* x0, const double* c, double* out1, double* outm, long B, long m){{
    if(m < 1) m = 1;
    if(!XG_{L}){{
        XG_{L} = alloc_huge_st((long){L}*{PSZ}*8);
        CG_{L} = alloc_huge_st((long){L}*{PSZ}*8);
        CP_{L} = alloc_huge_st((long){L3}*16*8);
        SN_{L} = alloc_huge_st((long){L}*{PSZ}*8);
    }}
    {{
        long rem = B % 8;
        if(rem && rem <= {THR}){{
            long Bm = B - rem;
            run_{L}pv(x0 + Bm*2*{L3}, c + Bm*2*{L3}, out1 + Bm*2*{L3}, outm + Bm*2*{L3}, rem, m);
            B = Bm;
            if(!B) return;
        }}
    }}
    long G = (B + 7) / 8;
    const double* srcs[8]; double* dsts[8];
    for(long g=0; g<G; g++){{
        long v0 = g*8;
        int nv = (int)((B - v0) < 8 ? (B - v0) : 8);
        for(int v=0; v<8; v++) srcs[v] = x0 + (v0 + (v < nv ? v : nv-1))*2*{L3};
        convin_{L}(srcs, XG_{L});
        for(int v=0; v<8; v++) srcs[v] = c + (v0 + (v < nv ? v : nv-1))*2*{L3};
        convin_{L}(srcs, CG_{L});
        // pencil-major copy of c: CP[e][q] = CG[q-plane][e]
        for(long e=0;e<{L2};e++)
            for(long q=0;q<{L};q++){{
                _mm512_store_pd(CP_{L} + e*{L*16} + q*16,     _mm512_load_pd(CG_{L} + q*{PSZ} + e*16));
                _mm512_store_pd(CP_{L} + e*{L*16} + q*16 + 8, _mm512_load_pd(CG_{L} + q*{PSZ} + e*16 + 8));
            }}
        S_{L}(XG_{L}, CG_{L}, 0);
        if(m == 1){{
            P_{L}(XG_{L}, CP_{L}, 1, 0);
            for(int v=0;v<nv;v++) dsts[v] = out1 + (v0+v)*2*{L3};
            convout_{L}(XG_{L}, dsts, nv);
            for(int v=0;v<nv;v++) dsts[v] = outm + (v0+v)*2*{L3};
            convout_{L}(XG_{L}, dsts, nv);
            continue;
        }}
        P_{L}(XG_{L}, CP_{L}, 3, SN_{L});
        for(int v=0;v<nv;v++) dsts[v] = out1 + (v0+v)*2*{L3};
        convout_{L}(SN_{L}, dsts, nv);
        long t = 2;
        while(1){{
            S_{L}(XG_{L}, CG_{L}, t==m ? 1 : 2);
            if(t == m) break;
            t++;
            P_{L}(XG_{L}, CP_{L}, t==m ? 1 : 2, 0);
            if(t == m) break;
            t++;
        }}
        for(int v=0;v<nv;v++) dsts[v] = outm + (v0+v)*2*{L3};
        convout_{L}(XG_{L}, dsts, nv);
    }}
}}
"""

def gen_driver(L, PS, pair_y=False, prime_asm=False):
    L2, L3 = L*L, L*L*L
    THR = {6:4,8:6,13:5,17:5,23:7}.get(L,0)
    if L in PAIR_MN:
        ZMN_BODY = f'_Pragma(\"GCC unroll 1\") for(int j=0;j<{L};j+=2) cd{L}_mn2(sl + (long)j*{L*16}, sl + (long)(j+1)*{L*16}, 16, cl + (long)j*{L*16}, cl + (long)(j+1)*{L*16});'
        XMN_BODY = f'cd{L}_mn2(p, p + 16, {PS*16}, pc, pc + {L*16}); e++;'
    else:
        ZMN_BODY = f'_Pragma(\"GCC unroll 1\") for(int j=0;j<{L};j++) cd{L}_mn(sl + (long)j*{L*16}, 16, cl + (long)j*{L*16});'
        XMN_BODY = f'cd{L}_mn(p, {PS*16}, pc);' 
    PSZ = PS*16
    cd = f"cd{L}"
    if prime_asm:
        AXY = f'_Pragma(\"GCC unroll 1\") for(int k=0;k<{L};k++) cd{L}_y_p(sl + k*16);'
        return f"""
// ---------------- family A driver (asm prime), L={L} (PS={PS}) ----------------
static double* XG_{L} = 0;
static double* CG_{L} = 0;
static double* CP_{L} = 0;
static double* SN_{L} = 0;
static void S_{L}(double* X, const double* C, int mode){{
    for(int i=0;i<{L};i++){{
        double* sl = X + (long)i*{PSZ};
        const double* cl = C + (long)i*{PSZ};
        {AXY}
        if(mode == 0){{
            _Pragma(\"GCC unroll 1\") for(int j=0;j<{L};j++) cd{L}_z_p(sl + (long)j*{L*16});
        }} else if(mode == 1){{
            _Pragma(\"GCC unroll 1\") for(int j=0;j<{L};j++) cd{L}_z_m(sl + (long)j*{L*16}, cl + (long)j*{L*16});
        }} else {{
            _Pragma(\"GCC unroll 1\") for(int j=0;j<{L};j++) cd{L}_z_mn(sl + (long)j*{L*16}, cl + (long)j*{L*16});
            {AXY}
        }}
    }}
}}
static void P_{L}(double* X, const double* CP, int mode, double* SNAP){{
    _Pragma(\"GCC unroll 1\") for(long e=0;e<{L2};e++){{
        double* p = X + e*16;
        const double* pc = CP + e*{L*16};
        if(mode == 1) cd{L}_x_m(p, pc);
        else if(mode == 2) cd{L}_x_mn(p, pc, CP + ((e+1) % {L2})*{L*16});
        else cd{L}_x_mns(p, pc, SNAP + e*16);
    }}
}}
""" + DRIVER_RUN.format(L=L, L2=L2, L3=L3, PSZ=PSZ, THR={6:4,8:6,13:5,17:5,23:7}.get(L,0), **{"L*16": L*16})
    if pair_y:
        AXY = f"""{{ int k=0; for(;k+2<={L};k+=2) {cd}_pp(sl + k*16, sl + k*16 + 16, {L*16}); for(;k<{L};k++) {cd}_p(sl + k*16, {L*16}); }}"""
    else:
        AXY = f"_Pragma(\"GCC unroll 1\") for(int k=0;k<{L};k++) {cd}_p(sl + k*16, {L*16});"
    return f"""
// ---------------- family A driver, L={L} (PS={PS}) ----------------
static double* XG_{L} = 0;
static double* CG_{L} = 0;
static double* CP_{L} = 0;
static double* SN_{L} = 0;
// S-visit: mode 0: y,z (plain); 1: y, z+map (final); 2: y, z+map+z', y'
static void S_{L}(double* X, const double* C, int mode){{
    for(int i=0;i<{L};i++){{
        double* sl = X + (long)i*{PSZ};
        const double* cl = C + (long)i*{PSZ};
        {AXY}
        if(mode == 0){{
            _Pragma(\"GCC unroll 1\") for(int j=0;j<{L};j++) {cd}_p(sl + (long)j*{L*16}, 16);
        }} else if(mode == 1){{
            _Pragma(\"GCC unroll 1\") for(int j=0;j<{L};j++) {cd}_m(sl + (long)j*{L*16}, 16, cl + (long)j*{L*16});
        }} else {{
            {ZMN_BODY}
            {AXY}
        }}
    }}
}}
// P-visit: mode 1: x+map; 2: x+map+x'; 3: x+map+snap+x'
static void P_{L}(double* X, const double* CP, int mode, double* SNAP){{
    _Pragma(\"GCC unroll 1\") for(long e=0;e<{L2};e++){{
        double* p = X + e*16;
        const double* pc = CP + e*{L*16};
        if(mode == 1) {cd}_m(p, {PSZ}, pc);
        else if(mode == 2){{
            {XMN_BODY}
        }}
        else {cd}_mns(p, {PSZ}, pc, SNAP + e*16, {PSZ});
    }}
}}
void run_{L}(const double* x0, const double* c, double* out1, double* outm, long B, long m){{
    if(m < 1) m = 1;
    if(!XG_{L}){{
        XG_{L} = alloc_huge_st((long){L}*{PSZ}*8);
        CG_{L} = alloc_huge_st((long){L}*{PSZ}*8);
        CP_{L} = alloc_huge_st((long){L3}*16*8);
        SN_{L} = alloc_huge_st((long){L}*{PSZ}*8);
    }}
    {{
        long rem = B % 8;
        if(rem && rem <= {THR}){{
            long Bm = B - rem;
            run_{L}pv(x0 + Bm*2*{L3}, c + Bm*2*{L3}, out1 + Bm*2*{L3}, outm + Bm*2*{L3}, rem, m);
            B = Bm;
            if(!B) return;
        }}
    }}
    long G = (B + 7) / 8;
    const double* srcs[8]; double* dsts[8];
    for(long g=0; g<G; g++){{
        long v0 = g*8;
        int nv = (int)((B - v0) < 8 ? (B - v0) : 8);
        for(int v=0; v<8; v++) srcs[v] = x0 + (v0 + (v < nv ? v : nv-1))*2*{L3};
        convin_{L}(srcs, XG_{L});
        for(int v=0; v<8; v++) srcs[v] = c + (v0 + (v < nv ? v : nv-1))*2*{L3};
        convin_{L}(srcs, CG_{L});
        // pencil-major copy of c: CP[e][q] = CG[q-plane][e]
        for(long e=0;e<{L2};e++)
            for(long q=0;q<{L};q++){{
                _mm512_store_pd(CP_{L} + e*{L*16} + q*16,     _mm512_load_pd(CG_{L} + q*{PSZ} + e*16));
                _mm512_store_pd(CP_{L} + e*{L*16} + q*16 + 8, _mm512_load_pd(CG_{L} + q*{PSZ} + e*16 + 8));
            }}
        S_{L}(XG_{L}, CG_{L}, 0);
        if(m == 1){{
            P_{L}(XG_{L}, CP_{L}, 1, 0);
            for(int v=0;v<nv;v++) dsts[v] = out1 + (v0+v)*2*{L3};
            convout_{L}(XG_{L}, dsts, nv);
            for(int v=0;v<nv;v++) dsts[v] = outm + (v0+v)*2*{L3};
            convout_{L}(XG_{L}, dsts, nv);
            continue;
        }}
        P_{L}(XG_{L}, CP_{L}, 3, SN_{L});
        for(int v=0;v<nv;v++) dsts[v] = out1 + (v0+v)*2*{L3};
        convout_{L}(SN_{L}, dsts, nv);
        long t = 2;
        while(1){{
            S_{L}(XG_{L}, CG_{L}, t==m ? 1 : 2);
            if(t == m) break;
            t++;
            P_{L}(XG_{L}, CP_{L}, t==m ? 1 : 2, 0);
            if(t == m) break;
            t++;
        }}
        for(int v=0;v<nv;v++) dsts[v] = outm + (v0+v)*2*{L3};
        convout_{L}(XG_{L}, dsts, nv);
    }}
}}
"""

def build(sizes=(6,8,13,17,23), kb={13:(6,6),17:(8,8),23:(6,6)}):
    from gen_asm_prime import gen_prime_asm
    parts = [PRELUDE]
    for L in sizes:
        PS = L*L if (L*L) % 2 == 1 else L*L+1
        if L in (6, 8):
            parts.append(gen_composite(L, PS*16))
            parts.append(gen_conv(L, PS))
            parts.append(gen_driver(L, PS, pair_y=True))
        else:
            parts.append(gen_prime_asm(L, kb[L][0], kb[L][1], PS))
            parts.append(gen_conv(L, PS))
            parts.append(gen_driver(L, PS, prime_asm=True))
    return "\n".join(parts)

if __name__ == "__main__":
    code = build()
    open("impl_a.c","w").write(code)
    print("wrote impl_a.c", len(code), "bytes")

def build_debug(sizes=(13,17,23)):
    from gen_a import gen_prime
    parts = [PRELUDE]
    for L in sizes:
        kb = {13:(6,6),17:(8,8),23:(6,6)}[L]
        parts.append(gen_prime(L, *kb))
        parts.append(f"void dbg_{L}(double* px){{ cd{L}_p(px, 16); }}")
    open("impl_dbg.c","w").write("\n".join(parts))
