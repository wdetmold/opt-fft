# Generator for my FFT3 engine: batched 8-volume SoA layout for primes 13/17/23.
import numpy as np
import sys, os as _osp
sys.path.insert(0, _osp.path.dirname(_osp.path.abspath(__file__)))
from dftgen import Emitter, emit_dft, emit_dft8
import genasm

LD = np.longdouble
PI = np.longdouble('3.14159265358979323846264338327950288')

def hexd(x):
    return float(np.double(x)).hex()

HDR = r'''
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

static const __m512d VONE_ = {1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0};
#define VONE VONE_
static const __m512d VHALF_ = {0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5};
#define VHALF VHALF_

/* map z/(1+|z|): zr,zi -> xr,xi. ~1ulp via rsqrt14+NR+Heron and rcp14+NR2 */
static inline void maphw(__m512d zr, __m512d zi, __m512d* oxr, __m512d* oxi){
    const __m512d TINY = _mm512_set1_pd(1e-30);
    __m512d m  = _mm512_fmadd_pd(zr, zr, _mm512_fmadd_pd(zi, zi, TINY));
    __m512d s  = _mm512_sqrt_pd(m);
    __m512d u  = _mm512_add_pd(VONE, s);
    __m512d w  = _mm512_div_pd(VONE, u);
    *oxr = _mm512_mul_pd(zr, w);
    *oxi = _mm512_mul_pd(zi, w);
}
static inline void map2(__m512d zr, __m512d zi, __m512d* oxr, __m512d* oxi){
    const __m512d TINY = _mm512_set1_pd(1e-30);
    __m512d m  = _mm512_fmadd_pd(zr, zr, _mm512_fmadd_pd(zi, zi, TINY));
    __m512d r0 = _mm512_rsqrt14_pd(m);
    __m512d t  = _mm512_mul_pd(m, r0);
    __m512d hr = _mm512_mul_pd(r0, VHALF);
    __m512d eh = _mm512_fnmadd_pd(t, hr, VHALF);
    __m512d r1 = _mm512_fmadd_pd(r0, eh, r0);
    __m512d mg0= _mm512_mul_pd(m, r1);
    __m512d hr1= _mm512_mul_pd(r1, VHALF);
    __m512d e2 = _mm512_fnmadd_pd(mg0, mg0, m);
    __m512d mag= _mm512_fmadd_pd(e2, hr1, mg0);
    __m512d u  = _mm512_add_pd(VONE, mag);
    __m512d w0 = _mm512_rcp14_pd(u);
    __m512d e3 = _mm512_fnmadd_pd(u, w0, VONE);
    __m512d a  = _mm512_fmadd_pd(w0, e3, w0);
    __m512d ee = _mm512_mul_pd(e3, e3);
    __m512d w2 = _mm512_fmadd_pd(a, ee, a);
    *oxr = _mm512_mul_pd(zr, w2);
    *oxi = _mm512_mul_pd(zi, w2);
}

/* 8x8 transpose */
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
} while(0)
'''

HDR += """
/* in-place +c+map over n contiguous element-vecs, hw/NR alternating */
static void __attribute__((noinline)) mapslab(double* restrict X, const double* restrict C, long n){
    long e = 0;
    for(; e+2 <= n; e += 2){
        __m512d zr0 = _mm512_add_pd(_mm512_load_pd(X+e*16),    _mm512_load_pd(C+e*16));
        __m512d zi0 = _mm512_add_pd(_mm512_load_pd(X+e*16+8),  _mm512_load_pd(C+e*16+8));
        __m512d zr1 = _mm512_add_pd(_mm512_load_pd(X+e*16+16), _mm512_load_pd(C+e*16+16));
        __m512d zi1 = _mm512_add_pd(_mm512_load_pd(X+e*16+24), _mm512_load_pd(C+e*16+24));
        maphw(zr0, zi0, &zr0, &zi0);
        map2(zr1, zi1, &zr1, &zi1);
        _mm512_store_pd(X+e*16,    zr0); _mm512_store_pd(X+e*16+8,  zi0);
        _mm512_store_pd(X+e*16+16, zr1); _mm512_store_pd(X+e*16+24, zi1);
    }
    for(; e < n; e++){
        __m512d zr = _mm512_add_pd(_mm512_load_pd(X+e*16),   _mm512_load_pd(C+e*16));
        __m512d zi = _mm512_add_pd(_mm512_load_pd(X+e*16+8), _mm512_load_pd(C+e*16+8));
        map2(zr, zi, &zr, &zi);
        _mm512_store_pd(X+e*16, zr); _mm512_store_pd(X+e*16+8, zi);
    }
}
"""

# NOTE: TR8 above has a bug on purpose? No - fix: o1/o5 use _u4/_u5, o3/o7 must use different.
def gen_prime_codelet(N, KBs, name, fuse_map=False, KBo=None, map_in=False):
    """Strided batched prime DFT codelet, phase-split with a/b scratch.
    X layout: element e at X + e*es*16 (re vec at +0, im vec at +8).
    fuse_map: after DFT, z+=C then map, store mapped value (C same layout/stride)."""
    h = (N-1)//2
    cos = [np.cos(2*PI*n/LD(N)) for n in range(h+1)]
    sin = [np.sin(2*PI*n/LD(N)) for n in range(h+1)]
    o = []
    w = o.append
    args = "double* restrict X, long es, int dopf" + (", const double* restrict C, long ces" if fuse_map else "")
    w(f"static __attribute__((always_inline)) inline void {name}({args}){{")
    w("    const long s = es*16;")
    if fuse_map:
        w("    const long cs = ces*16;")
    w(f"    double AB[{8*h}*8] ALIGN64;")   # ar,ai,br,bi per j: 4 vecs * h
    w(f"    double Escr[{2*h}*8] ALIGN64;")
    w("    __m512d x0r = _mm512_load_pd(X);")
    w("    if(dopf){ _mm_prefetch((const char*)(X+%d), _MM_HINT_T0); _mm_prefetch((const char*)(X+%d), _MM_HINT_T0); }" % (N*16, N*16+8))
    w("    __m512d x0i = _mm512_load_pd(X+8);")
    if map_in:
        w("    maphw(x0r, x0i, &x0r, &x0i);" if MAPMIX else "    map2(x0r, x0i, &x0r, &x0i);")
    kb_list = []
    k0 = 1
    for kb in KBs:
        kb_list.append(list(range(k0, k0+kb))); k0 += kb
    assert k0 == h+1
    kbo_list = []
    k0 = 1
    for kb in (KBo or KBs):
        kbo_list.append(list(range(k0, k0+kb))); k0 += kb
    assert k0 == h+1
    first = True
    for blk in kb_list:
        needed = sorted(set(min((k*j)%N, N-((k*j)%N)) for k in blk for j in range(1,h+1)))
        decl = ", ".join(f"c{n} = _mm512_set1_pd({hexd(cos[n])})" for n in needed)
        w(f"    {{ __m512d {decl};")
        for k in blk:
            w(f"    __m512d e{k}r = x0r, e{k}i = x0i;")
        if first:
            w("    __m512d sr = x0r, si = x0i;")
        for j in range(1, h+1):
            if first:
                w(f"    {{ __m512d pr = _mm512_load_pd(X+{j}*s), qr = _mm512_load_pd(X+{N-j}*s);")
                w(f"      if(dopf){{ _mm_prefetch((const char*)(X+{j}*s+{N*16}), _MM_HINT_T0); _mm_prefetch((const char*)(X+{j}*s+{N*16}+8), _MM_HINT_T0);")
                w(f"                _mm_prefetch((const char*)(X+{N-j}*s+{N*16}), _MM_HINT_T0); _mm_prefetch((const char*)(X+{N-j}*s+{N*16}+8), _MM_HINT_T0); }}")
                w(f"      __m512d pi = _mm512_load_pd(X+{j}*s+8), qi = _mm512_load_pd(X+{N-j}*s+8);")
                if map_in:
                    if MAPMIX and j % 2 == 1:
                        w("      map2(pr, pi, &pr, &pi); maphw(qr, qi, &qr, &qi);")
                    elif MAPMIX:
                        w("      maphw(pr, pi, &pr, &pi); map2(qr, qi, &qr, &qi);")
                    else:
                        w("      map2(pr, pi, &pr, &pi); map2(qr, qi, &qr, &qi);")
                w(f"      __m512d ur = _mm512_add_pd(pr,qr), ui = _mm512_add_pd(pi,qi);")
                w(f"      __m512d vr = _mm512_sub_pd(pr,qr), vi = _mm512_sub_pd(pi,qi);")
                w(f"      _mm512_store_pd(AB+{(j-1)*32}, ur);    _mm512_store_pd(AB+{(j-1)*32+8}, ui);")
                w(f"      _mm512_store_pd(AB+{(j-1)*32+16}, vr); _mm512_store_pd(AB+{(j-1)*32+24}, vi);")
                w("      sr = _mm512_add_pd(sr, ur); si = _mm512_add_pd(si, ui);")
            else:
                w(f"    {{ __m512d ur = _mm512_load_pd(AB+{(j-1)*32});")
                w(f"      __m512d ui = _mm512_load_pd(AB+{(j-1)*32+8});")
            for k in blk:
                n = (k*j)%N; n = min(n, N-n)
                w(f"      e{k}r = _mm512_fmadd_pd(c{n}, ur, e{k}r); e{k}i = _mm512_fmadd_pd(c{n}, ui, e{k}i);")
            w("    }")
        if first:
            if fuse_map:
                w("    { __m512d zr = _mm512_add_pd(sr, _mm512_load_pd(C)), zi = _mm512_add_pd(si, _mm512_load_pd(C+8));")
                if MAPX:
                    w("      maphw(zr, zi, &zr, &zi);")
                w("      _mm512_store_pd(X, zr); _mm512_store_pd(X+8, zi); }")
            else:
                w("    _mm512_store_pd(X, sr); _mm512_store_pd(X+8, si);")
            first = False
        for k in blk:
            w(f"    _mm512_store_pd(Escr+{(k-1)*16}, e{k}r); _mm512_store_pd(Escr+{(k-1)*16+8}, e{k}i);")
        w("    }")
    for blk in kbo_list:
        needed = sorted(set(min((k*j)%N, N-((k*j)%N)) for k in blk for j in range(1,h+1)))
        decl = ", ".join(f"s{n} = _mm512_set1_pd({hexd(sin[n])})" for n in needed)
        w(f"    {{ __m512d {decl};")
        for k in blk:
            w(f"    __m512d o{k}r = _mm512_setzero_pd(), o{k}i = _mm512_setzero_pd();")
        for j in range(1, h+1):
            w(f"    {{ __m512d vr = _mm512_load_pd(AB+{(j-1)*32+16});")
            w(f"      __m512d vi = _mm512_load_pd(AB+{(j-1)*32+24});")
            for k in blk:
                n = (k*j)%N
                if n <= h:
                    w(f"      o{k}r = _mm512_fmadd_pd(s{n}, vr, o{k}r); o{k}i = _mm512_fmadd_pd(s{n}, vi, o{k}i);")
                else:
                    w(f"      o{k}r = _mm512_fnmadd_pd(s{N-n}, vr, o{k}r); o{k}i = _mm512_fnmadd_pd(s{N-n}, vi, o{k}i);")
            w("    }")
        for k in blk:
            w(f"    {{ __m512d er = _mm512_load_pd(Escr+{(k-1)*16}), ei = _mm512_load_pd(Escr+{(k-1)*16+8});")
            if fuse_map:
                w(f"      __m512d zr1 = _mm512_add_pd(_mm512_add_pd(er, o{k}i), _mm512_load_pd(C+{k}*cs));")
                w(f"      __m512d zi1 = _mm512_add_pd(_mm512_sub_pd(ei, o{k}r), _mm512_load_pd(C+{k}*cs+8));")
                w(f"      __m512d zr2 = _mm512_add_pd(_mm512_sub_pd(er, o{k}i), _mm512_load_pd(C+{N-k}*cs));")
                w(f"      __m512d zi2 = _mm512_add_pd(_mm512_add_pd(ei, o{k}r), _mm512_load_pd(C+{N-k}*cs+8));")
                if MAPX:
                    w(f"      {'maphw' if k%2==0 else 'map2'}(zr1, zi1, &zr1, &zi1); {'map2' if k%2==0 else 'maphw'}(zr2, zi2, &zr2, &zi2);")
                w(f"      _mm512_store_pd(X+{k}*s, zr1);   _mm512_store_pd(X+{k}*s+8, zi1);")
                w(f"      _mm512_store_pd(X+{N-k}*s, zr2); _mm512_store_pd(X+{N-k}*s+8, zi2); }}")
            else:
                w(f"      _mm512_store_pd(X+{k}*s,   _mm512_add_pd(er, o{k}i));")
                w(f"      _mm512_store_pd(X+{k}*s+8, _mm512_sub_pd(ei, o{k}r));")
                w(f"      _mm512_store_pd(X+{N-k}*s,   _mm512_sub_pd(er, o{k}i));")
                w(f"      _mm512_store_pd(X+{N-k}*s+8, _mm512_add_pd(ei, o{k}r)); }}")
        w("    }")
    w("}")
    return "\n".join(o)

INOUT = r'''
/* ingest: 8 volumes AoS complex -> group SoA [e][2][8] */
static void ingest_@N@(const double* const* src, double* G){
    for(long e=0; e<@NE4@; e+=4){
        __m512d r0=_mm512_loadu_pd(src[0]+2*e), r1=_mm512_loadu_pd(src[1]+2*e);
        __m512d r2=_mm512_loadu_pd(src[2]+2*e), r3=_mm512_loadu_pd(src[3]+2*e);
        __m512d r4=_mm512_loadu_pd(src[4]+2*e), r5=_mm512_loadu_pd(src[5]+2*e);
        __m512d r6=_mm512_loadu_pd(src[6]+2*e), r7=_mm512_loadu_pd(src[7]+2*e);
        __m512d o0,o1,o2,o3,o4,o5,o6,o7;
        TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
        _mm512_store_pd(G+e*16,    o0); _mm512_store_pd(G+e*16+8,  o1);
        _mm512_store_pd(G+e*16+16, o2); _mm512_store_pd(G+e*16+24, o3);
        _mm512_store_pd(G+e*16+32, o4); _mm512_store_pd(G+e*16+40, o5);
        _mm512_store_pd(G+e*16+48, o6); _mm512_store_pd(G+e*16+56, o7);
    }
    { /* tail of @NT@ elements */
        const long e = @NE4@;
        const __mmask8 mk = (__mmask8)((1u<<(2*@NT@))-1u);
        __m512d r0=_mm512_maskz_loadu_pd(mk, src[0]+2*e), r1=_mm512_maskz_loadu_pd(mk, src[1]+2*e);
        __m512d r2=_mm512_maskz_loadu_pd(mk, src[2]+2*e), r3=_mm512_maskz_loadu_pd(mk, src[3]+2*e);
        __m512d r4=_mm512_maskz_loadu_pd(mk, src[4]+2*e), r5=_mm512_maskz_loadu_pd(mk, src[5]+2*e);
        __m512d r6=_mm512_maskz_loadu_pd(mk, src[6]+2*e), r7=_mm512_maskz_loadu_pd(mk, src[7]+2*e);
        __m512d o0,o1,o2,o3,o4,o5,o6,o7;
        TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
        __m512d A[8]; A[0]=o0;A[1]=o1;A[2]=o2;A[3]=o3;A[4]=o4;A[5]=o5;A[6]=o6;A[7]=o7;
        for(int q=0;q<2*@NT@;q++) _mm512_store_pd(G+e*16+q*8, A[q]);
    }
}
/* output with map: group SoA (unmapped z) -> nv AoS volumes */
static void output_@N@(const double* G, double* const* dst, int nv){
    for(long e=0; e<@NE4@; e+=4){
        __m512d i0=_mm512_load_pd(G+e*16),    i1=_mm512_load_pd(G+e*16+8);
        __m512d i2=_mm512_load_pd(G+e*16+16), i3=_mm512_load_pd(G+e*16+24);
        __m512d i4=_mm512_load_pd(G+e*16+32), i5=_mm512_load_pd(G+e*16+40);
        __m512d i6=_mm512_load_pd(G+e*16+48), i7=_mm512_load_pd(G+e*16+56);
@MAPOUT4@
        __m512d o0,o1,o2,o3,o4,o5,o6,o7;
        TR8(i0,i1,i2,i3,i4,i5,i6,i7,o0,o1,o2,o3,o4,o5,o6,o7);
        __m512d* O[8] = {&o0,&o1,&o2,&o3,&o4,&o5,&o6,&o7};
        for(int v=0; v<nv; v++) _mm512_storeu_pd(dst[v]+2*e, *O[v]);
    }
    { /* tail */
        const long e = @NE4@;
        const __mmask8 mk = (__mmask8)((1u<<(2*@NT@))-1u);
        __m512d A[8];
        for(int q=0;q<2*@NT@;q++) A[q] = _mm512_load_pd(G+e*16+q*8);
@MAPOUTT@
        for(int q=2*@NT@;q<8;q++) A[q] = _mm512_setzero_pd();
        __m512d o0,o1,o2,o3,o4,o5,o6,o7;
        TR8(A[0],A[1],A[2],A[3],A[4],A[5],A[6],A[7],o0,o1,o2,o3,o4,o5,o6,o7);
        __m512d* O[8] = {&o0,&o1,&o2,&o3,&o4,&o5,&o6,&o7};
        for(int v=0; v<nv; v++) _mm512_mask_storeu_pd(dst[v]+2*e, mk, *O[v]);
    }
}
'''

SWEEPS = r'''
static void __attribute__((INLINEATTR)) OPTATTR@N@ dft@N@_one(double* X, long es){ dft@N@(X, es, 0); }
static void __attribute__((INLINEATTR)) OPTATTR@N@ dft@N@_onez(double* X){ dft@N@(X, 1, PFPRIME_@N@); }
static void __attribute__((INLINEATTR)) OPTATTR@N@ dft@N@_onezm(double* X){ dft@N@zm(X, 1, PFPRIME_@N@); }
static void __attribute__((INLINEATTR)) OPTATTR@N@ dft@N@_onem(double* X, long es, const double* Ct){ dft@N@m(X, es, 0, Ct, 1); }
static void dft@N@_sweep_zy(double* restrict X){
    for(long x=0; x<@N@; x++){
        double* P = X + x*@N2@*16;
        for(long y=0; y<@N@; y++) dft@N@_onez(P + y*@N@*16);
        for(long z=0; z<@N@; z++) dft@N@_one(P + z*16, @N@);
    }
}
#if MAPZB_FLAG
static void __attribute__((noinline)) mapblk_@N@(double* restrict P){
    for(long e=0; e<@N@; e+=1){
        __m512d zr = _mm512_load_pd(P + e*16);
        __m512d zi = _mm512_load_pd(P + e*16 + 8);
        if(e & 1){ map2(zr, zi, &zr, &zi); } else { maphw(zr, zi, &zr, &zi); }
        _mm512_store_pd(P + e*16, zr);
        _mm512_store_pd(P + e*16 + 8, zi);
    }
}
static void dft@N@_sweep_zym(double* restrict X){
    for(long x=0; x<@N@; x++){
        double* P = X + x*@N2@*16;
        for(long y=0; y<@N@; y++){ mapblk_@N@(P + y*@N@*16); dft@N@_one(P + y*@N@*16, 1); }
        for(long z=0; z<@N@; z++) dft@N@_one(P + z*16, @N@);
    }
}
#else
static void dft@N@_sweep_zym(double* restrict X){
    for(long x=0; x<@N@; x++){
        double* P = X + x*@N2@*16;
        for(long y=0; y<@N@; y++) dft@N@_onezm(P + y*@N@*16);
        for(long z=0; z<@N@; z++) dft@N@_one(P + z*16, @N@);
    }
}
#endif
static void dft@N@_sweep_x_map(double* restrict X, const double* restrict Ct){
    for(long p=0; p<@N2@; p++) dft@N@_onem(X + p*16, @N2@, Ct + p*@N@*16);
}
static void dft@N@_sweep_x_plain(double* restrict X){
    for(long p=0; p<@N2@; p++) dft@N@_one(X + p*16, @N2@);
}
static void dft@N@_sweep_zy_ms(double* restrict X, const double* restrict C){
    for(long x=0; x<@N@; x++){
        double* P = X + x*@N2@*16;
        for(long y=0; y<@N@; y++) dft@N@_one(P + y*@N@*16, 1);
        for(long z=0; z<@N@; z++) dft@N@_one(P + z*16, @N@);
        if(x) mapslab(X + (x-1)*@N2@*16, C + (x-1)*@N2@*16, @N2@);
    }
    mapslab(X + (@N@-1)*@N2@*16, C + (@N@-1)*@N2@*16, @N2@);
}
'''

DRIVER = r'''
static double* Xg_@N@ = 0;
static double* Cg_@N@ = 0;
void hot2_@N@(long which){
    if(!Xg_@N@){ Xg_@N@ = alloc_huge_st((@NE@+64*@N@)*16*8); Cg_@N@ = alloc_huge_st(@NE@*16*8); }
    double* P = Xg_@N@;
    if(which==99){ for(long i=0;i<@N2@*16;i++) P[i] = 0.5 + 1e-6*(i%97); return; }
    if(which==0 || which==2) for(long y=0; y<@N@; y++) dft@N@_one(P + y*@N@*16, 1);
    if(which==1 || which==2) for(long z=0; z<@N@; z++) dft@N@_one(P + z*16, @N@);
}
void bsweep_@N@(long which, long n){
    if(!Xg_@N@){ Xg_@N@ = alloc_huge_st(@NE@*16*8); Cg_@N@ = alloc_huge_st(@NE@*16*8); }
    for(long i=0;i<@NE@*16;i++){ Xg_@N@[i] = 0.5 + 1e-6*(i%97); Cg_@N@[i] = 0.01; }
    for(long r=0;r<n;r++){
        if(which==0) dft@N@_sweep_zy(Xg_@N@);
        else if(which==1) dft@N@_sweep_zym(Xg_@N@);
        else if(which==2) dft@N@_sweep_x_map(Xg_@N@, Cg_@N@);
#if USEASM_FLAG
#endif
        if((r&7)==7) for(long i=0;i<@NE@*16;i+=997) Xg_@N@[i] = 0.5;
    }
}
void run_@N@(const double* x0, const double* c, double* out1, double* outm, long B, long m){
    const long NE = @NE@;
    if(!Xg_@N@){ Xg_@N@ = alloc_huge_st(NE*16*8); Cg_@N@ = alloc_huge_st(NE*16*8); }
    double* X = Xg_@N@; double* Ct = Cg_@N@;
    for(long g0=0; g0<B; g0+=8){
        int nv = (int)((B - g0) < 8 ? (B - g0) : 8);
        const double* src[8]; const double* csrc[8];
        double* d1[8]; double* dm[8];
        for(int v=0; v<8; v++){
            int vv = v < nv ? v : 0;
            src[v] = x0 + (g0+vv)*2*NE; csrc[v] = c + (g0+vv)*2*NE;
            if(v<nv){ d1[v] = out1 + (g0+v)*2*NE; dm[v] = outm + (g0+v)*2*NE; }
        }
        /* ingest c (consumption order for MAPX/deferred paths; plain for PXF) */
#if PXF_@N@
        ingest_@N@(csrc, Ct);
#else
        ingest_@N@(csrc, X);
        for(long p=0; p<@N2@; p++)
            for(long k=0; k<@N@; k++){
                _mm512_store_pd(Ct + (p*@N@+k)*16,     _mm512_load_pd(X + (k*@N2@+p)*16));
                _mm512_store_pd(Ct + (p*@N@+k)*16 + 8, _mm512_load_pd(X + (k*@N2@+p)*16 + 8));
            }
#endif
        ingest_@N@(src, X);
        for(long t=0; t<m; t++){
#if PXF_@N@
            dft@N@_sweep_x_plain(X);
            dft@N@_sweep_zy_ms(X, Ct);
#elif USEASM_FLAG
            dft@N@_sweep_zy_asm(X, t>0);
            dft@N@_sweep_x_asm(X, Ct);
#else
#if MAPX_@N@
            dft@N@_sweep_zy(X);
            dft@N@_sweep_x_map(X, Ct);
#else
            if(t==0) dft@N@_sweep_zy(X); else dft@N@_sweep_zym(X);
            dft@N@_sweep_x_map(X, Ct);
#endif
#endif
            if(t==0 && m>1) output_@N@(X, d1, nv);
        }
        output_@N@(X, dm, nv);
        if(m==1) output_@N@(X, d1, nv);
    }
}
'''

def pfa_maps(N1, N2):
    N = N1*N2
    inv21 = pow(N2, -1, N1); inv12 = pow(N1, -1, N2)
    jm = [[(N2*a + N1*b) % N for b in range(N2)] for a in range(N1)]
    km = [[(a*N2*inv21 + b*N1*inv12) % N for b in range(N2)] for a in range(N1)]
    return jm, km

def riffle_pf(body_lines, N, pfexpr):
    """insert 2N prefetcht0 lines evenly through body (C source lines)"""
    pfs = [f'    _mm_prefetch((const char*)({pfexpr}) + {64*i}, _MM_HINT_T0);' for i in range(2*N)]
    out = []
    n = len(body_lines)
    k = len(pfs)
    step = max(1, n // (k+1))
    pi = 0
    for i, l in enumererate(body_lines) if False else enumerate(body_lines):
        out.append(l)
        if pi < k and (i+1) % step == 0 and i > 2:
            out.append(pfs[pi]); pi += 1
    while pi < k:
        out.append(pfs[pi]); pi += 1
    return out

def gen_composite_codelet(N, name, fuse_map=False, seqload=False, pfoff=None):
    if N == 45: N1, N2 = 9, 5
    elif N == 36: N1, N2 = 9, 4
    elif N == 6: N1, N2 = 3, 2
    elif N == 8: return gen_single_codelet(N, name, fuse_map)
    else: raise ValueError
    jm, km = pfa_maps(N1, N2)
    o = []
    w = o.append
    args = "double* restrict X, long es" + (", const double* restrict C" if fuse_map else "")
    REGRES = (N <= 8)   # register-resident: no SCR round trip
    w(f"static void {name}({args}){{")
    w("    const long s = es*16;")
    if not REGRES:
        w(f"    double SCR[{N*16}] ALIGN64;")
    if seqload:
        w(f"    double SQ[{N*16}] ALIGN64;")
        w(f"    for(long q=0; q<{N*16}; q+=16){{")
        w(f"        _mm512_store_pd(SQ+q,   _mm512_load_pd(X+q));")
        w(f"        _mm512_store_pd(SQ+q+8, _mm512_load_pd(X+q+8));")
        w(f"    }}")
    mid = {}
    REGSTASH = []
    if REGRES:
        E0 = Emitter()
        for b in range(N2):
            xin = []
            for a in range(N1):
                j = jm[a][b]
                base = "SQ+%d*16" % j if seqload else "X+%d*s" % j
                vr, vi = E0.newv(), E0.newv()
                E0.emit(f"__m512d {vr} = _mm512_load_pd({base});")
                E0.emit(f"__m512d {vi} = _mm512_load_pd({base}+8);")
                xin.append((vr, vi))
            out = emit_dft(E0, N1, xin)
            for a in range(N1):
                mid[(a,b)] = out[a]
    else:
        for b in range(N2):
            E = Emitter()
            xin = []
            for a in range(N1):
                j = jm[a][b]
                base = "SQ+%d*16" % j if seqload else "X+%d*s" % j
                E.emit(f"__m512d r{a} = _mm512_load_pd({base});")
                if pfoff is not None:
                    E.emit(f"_mm_prefetch((const char*)(X+{j}*s+{pfoff}), _MM_HINT_T0);")
                E.emit(f"__m512d i{a} = _mm512_load_pd({base}+8);")
                if pfoff is not None:
                    E.emit(f"_mm_prefetch((const char*)(X+{j}*s+{pfoff}+8), _MM_HINT_T0);")
                xin.append((f"r{a}", f"i{a}"))
            out = emit_dft(E, N1, xin)
            w("    {")
            for cl in E.cdecls(): w(cl)
            for l in E.lines: w(l)
            for a in range(N1):
                w(f"    _mm512_store_pd(SCR+{(a*N2+b)*16}, {out[a][0]});")
                w(f"    _mm512_store_pd(SCR+{(a*N2+b)*16+8}, {out[a][1]});")
            w("    }")
    for a in range(N1):
        if REGRES:
            E = E0 if a == 0 else E
            xin = [mid[(a,b)] for b in range(N2)]
            if a == 0:
                pass
            out = emit_dft(E, N2, xin)
        else:
            E = Emitter()
            xin = []
            for b in range(N2):
                E.emit(f"__m512d r{b} = _mm512_load_pd(SCR+{(a*N2+b)*16});")
                E.emit(f"__m512d i{b} = _mm512_load_pd(SCR+{(a*N2+b)*16+8});")
                xin.append((f"r{b}", f"i{b}"))
            out = emit_dft(E, N2, xin)
        if not REGRES:
            w("    {")
            for cl in E.cdecls(): w(cl)
            for l in E.lines: w(l)
        for b in range(N2):
            k = km[a][b]
            rr, ii = out[b]
            if REGRES:
                REGSTASH.append((k, rr, ii))
                continue
            if fuse_map:
                mfn = "map2" if (k % 2 or not MAPMIX) else "maphw"
                w(f"    {{ __m512d zr = _mm512_add_pd({rr}, _mm512_load_pd(C+{k}*16));")
                w(f"      __m512d zi = _mm512_add_pd({ii}, _mm512_load_pd(C+{k}*16+8));")
                w(f"      {mfn}(zr, zi, &zr, &zi);")
                w(f"      _mm512_store_pd(X+{k}*s, zr); _mm512_store_pd(X+{k}*s+8, zi); }}")
            else:
                w(f"    _mm512_store_pd(X+{k}*s, {rr});")
                w(f"    _mm512_store_pd(X+{k}*s+8, {ii});")
        if REGRES:
            if a == N1-1:
                # print everything now
                w("    {")
                for cl in E.cdecls(): w(cl)
                for l in E.lines: w(l)
                for (k, rr, ii) in REGSTASH:
                    if fuse_map:
                        mfn = "map2" if (k % 2 or not MAPMIX) else "maphw"
                        w(f"    {{ __m512d zr = _mm512_add_pd({rr}, _mm512_load_pd(C+{k}*16));")
                        w(f"      __m512d zi = _mm512_add_pd({ii}, _mm512_load_pd(C+{k}*16+8));")
                        w(f"      {mfn}(zr, zi, &zr, &zi);")
                        w(f"      _mm512_store_pd(X+{k}*s, zr); _mm512_store_pd(X+{k}*s+8, zi); }}")
                    else:
                        w(f"    _mm512_store_pd(X+{k}*s, {rr});")
                        w(f"    _mm512_store_pd(X+{k}*s+8, {ii});")
                w("    }")
        else:
            w("    }")
    w("}")
    return "\n".join(o)

def gen_single_codelet(N, name, fuse_map=False):
    assert N == 8
    o = []
    w = o.append
    args = "double* restrict X, long es" + (", const double* restrict C" if fuse_map else "")
    w(f"static void {name}({args}){{")
    w("    const long s = es*16;")
    E = Emitter()
    xin = []
    for j in range(8):
        E.emit(f"__m512d r{j} = _mm512_load_pd(X+{j}*s);")
        E.emit(f"__m512d i{j} = _mm512_load_pd(X+{j}*s+8);")
        xin.append((f"r{j}", f"i{j}"))
    out = emit_dft8(E, xin)
    w("    {")
    for cl in E.cdecls(): w(cl)
    for l in E.lines: w(l)
    for k in range(8):
        rr, ii = out[k]
        if fuse_map:
            mfn = "map2" if (k % 2) else "maphw"
            w(f"    {{ __m512d zr = _mm512_add_pd({rr}, _mm512_load_pd(C+{k}*16));")
            w(f"      __m512d zi = _mm512_add_pd({ii}, _mm512_load_pd(C+{k}*16+8));")
            w(f"      {mfn}(zr, zi, &zr, &zi);")
            w(f"      _mm512_store_pd(X+{k}*s, zr); _mm512_store_pd(X+{k}*s+8, zi); }}")
        else:
            w(f"    _mm512_store_pd(X+{k}*s, {rr});")
            w(f"    _mm512_store_pd(X+{k}*s+8, {ii});")
    w("    }")
    w("}")
    return "\n".join(o)

def gen_size_comp(N):
    global MAPMIX
    MAPMIX = 0 if N in _MAPMIX_OFF else 1
    # padded plane stride (elements): odd so that line-step mod 64 has gcd 2 (cycle 32)
    N2 = N*N
    SP = N2
    best = None
    for pad in range(0, 64):
        sp = N2 + pad
        step = (sp*2) % 64  # lines per element-stride mod 64 sets
        import math
        cyc = 64 // math.gcd(step if step else 64, 64)
        # also check L2 (1024 sets): step2 = sp*2 mod 1024
        step2 = (sp*2) % 1024
        cyc2 = 1024 // math.gcd(step2 if step2 else 1024, 1024)
        score = (min(cyc, 32), min(cyc2, 64), -pad)
        if best is None or score > best[0]:
            best = (score, sp)
    SP = best[1]
    code = []
    PFIN = set(int(x) for x in os.environ.get("PFIN","").split(",") if x)
    code.append(gen_composite_codelet(N, f"dft{N}", fuse_map=False))
    if N in PFIN:
        # z codelet prefetching next z pencil; x codelet prefetching next x pencil
        code.append(gen_composite_codelet(N, f"dft{N}pfz", fuse_map=False, pfoff=f"{N}*16"))
        code.append(gen_composite_codelet(N, f"dft{N}mpf", fuse_map=True, pfoff="16"))
        # note: pfoff is in doubles added to element address X + j*s
    if N >= 36:
        code.append(gen_composite_codelet(N, f"dft{N}sq", fuse_map=False, seqload=True))
    code.append(gen_composite_codelet(N, f"dft{N}m", fuse_map=True))
    sweeps = '''
static void __attribute__((INLINEATTR)) dft@N@_one(double* X, long es){ dft@N@(X, es); }
#if PFIN_@N@
static void __attribute__((INLINEATTR)) dft@N@_onesq(double* X){ dft@N@pfz(X, 1); }
static void __attribute__((INLINEATTR)) dft@N@_onem(double* X, long es, const double* Ct){ dft@N@m(X, es, Ct); }
#elif 0
static void __attribute__((INLINEATTR)) dft@N@_onem_unused(double* X, long es, const double* Ct){ dft@N@mpf(X, es, Ct); }
#else
#if @SEQSW@
static void __attribute__((INLINEATTR)) dft@N@_onesq(double* X){ dft@N@sq(X, 1); }
#else
static void __attribute__((INLINEATTR)) dft@N@_onesq(double* X){ dft@N@(X, 1); }
#endif
static void __attribute__((INLINEATTR)) dft@N@_onem(double* X, long es, const double* Ct){ dft@N@m(X, es, Ct); }
#endif
static void dft@N@_sweep_zy(double* restrict X){
#if PFCOMP
    const long PB = (@N2@*2 + @N@ - 1)/@N@;   /* 128B-blocks of next plane per y-codelet */
    for(long x=0; x<@N@; x++){
        double* P = X + x*@SP@*16;
        const char* nxt = (const char*)(P + @SP@*16);
        for(long y=0; y<@N@; y++) dft@N@_one(P + y*@N@*16, 1);
        for(long z=0; z<@N@; z++){
            if(x+1 < @N@){
                const char* q = nxt + z*PB*128;
                for(long l=0; l<PB; l++) _mm_prefetch(q + l*128, _MM_HINT_T0);
            }
            dft@N@_one(P + z*16, @N@);
        }
    }
#else
    for(long x=0; x<@N@; x++){
        double* P = X + x*@SP@*16;
        for(long y=0; y<@N@; y++) dft@N@_onesq(P + y*@N@*16);
        for(long z=0; z<@N@; z++) dft@N@_one(P + z*16, @N@);
    }
#endif
}
static void dft@N@_sweep_x_map(double* restrict X, const double* restrict Ct){
    for(long p=0; p<@N2@; p++) dft@N@_onem(X + p*16, @SP@, Ct + p*@N@*16);
}
static void dft@N@_sweep_x_plain(double* restrict X){
    for(long p=0; p<@N2@; p++) dft@N@_one(X + p*16, @SP@);
}
static void dft@N@_sweep_zy_ms(double* restrict X, const double* restrict C){
    for(long x=0; x<@N@; x++){
        double* P = X + x*@SP@*16;
        for(long y=0; y<@N@; y++) dft@N@_onesq(P + y*@N@*16);
        for(long z=0; z<@N@; z++) dft@N@_one(P + z*16, @N@);
        if(x) mapslab(X + (x-1)*@SP@*16, C + (x-1)*@SP@*16, @N2@);
    }
    mapslab(X + (@N@-1)*@SP@*16, C + (@N@-1)*@SP@*16, @N2@);
}
/* plane-wise ingest/output (padded plane stride @SP@) */
static void ingest_@N@(const double* const* src, double* G){
    for(long x=0; x<@N@; x++){
        const long base = x*@N2@;
        double* Gp = G + x*@SP@*16;
        for(long e=0; e<@N2T4@; e+=4){
            __m512d r0=_mm512_loadu_pd(src[0]+2*(base+e)), r1=_mm512_loadu_pd(src[1]+2*(base+e));
            __m512d r2=_mm512_loadu_pd(src[2]+2*(base+e)), r3=_mm512_loadu_pd(src[3]+2*(base+e));
            __m512d r4=_mm512_loadu_pd(src[4]+2*(base+e)), r5=_mm512_loadu_pd(src[5]+2*(base+e));
            __m512d r6=_mm512_loadu_pd(src[6]+2*(base+e)), r7=_mm512_loadu_pd(src[7]+2*(base+e));
            __m512d o0,o1,o2,o3,o4,o5,o6,o7;
            TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
            _mm512_store_pd(Gp+e*16,    o0); _mm512_store_pd(Gp+e*16+8,  o1);
            _mm512_store_pd(Gp+e*16+16, o2); _mm512_store_pd(Gp+e*16+24, o3);
            _mm512_store_pd(Gp+e*16+32, o4); _mm512_store_pd(Gp+e*16+40, o5);
            _mm512_store_pd(Gp+e*16+48, o6); _mm512_store_pd(Gp+e*16+56, o7);
        }
#if @N2T@ > 0
        {
            const long e = @N2T4@;
            const __mmask8 mk = (__mmask8)((1u<<(2*@N2T@))-1u);
            __m512d r0=_mm512_maskz_loadu_pd(mk, src[0]+2*(base+e)), r1=_mm512_maskz_loadu_pd(mk, src[1]+2*(base+e));
            __m512d r2=_mm512_maskz_loadu_pd(mk, src[2]+2*(base+e)), r3=_mm512_maskz_loadu_pd(mk, src[3]+2*(base+e));
            __m512d r4=_mm512_maskz_loadu_pd(mk, src[4]+2*(base+e)), r5=_mm512_maskz_loadu_pd(mk, src[5]+2*(base+e));
            __m512d r6=_mm512_maskz_loadu_pd(mk, src[6]+2*(base+e)), r7=_mm512_maskz_loadu_pd(mk, src[7]+2*(base+e));
            __m512d o0,o1,o2,o3,o4,o5,o6,o7;
            TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
            __m512d A[8]; A[0]=o0;A[1]=o1;A[2]=o2;A[3]=o3;A[4]=o4;A[5]=o5;A[6]=o6;A[7]=o7;
            for(int q=0;q<2*@N2T@;q++) _mm512_store_pd(Gp+e*16+q*8, A[q]);
        }
#endif
    }
}
static void output_@N@(const double* G, double* const* dst, int nv){
    for(long x=0; x<@N@; x++){
        const long base = x*@N2@;
        const double* Gp = G + x*@SP@*16;
        for(long e=0; e<@N2T4@; e+=4){
            __m512d i0=_mm512_load_pd(Gp+e*16),    i1=_mm512_load_pd(Gp+e*16+8);
            __m512d i2=_mm512_load_pd(Gp+e*16+16), i3=_mm512_load_pd(Gp+e*16+24);
            __m512d i4=_mm512_load_pd(Gp+e*16+32), i5=_mm512_load_pd(Gp+e*16+40);
            __m512d i6=_mm512_load_pd(Gp+e*16+48), i7=_mm512_load_pd(Gp+e*16+56);
            __m512d o0,o1,o2,o3,o4,o5,o6,o7;
            TR8(i0,i1,i2,i3,i4,i5,i6,i7,o0,o1,o2,o3,o4,o5,o6,o7);
            __m512d* O[8] = {&o0,&o1,&o2,&o3,&o4,&o5,&o6,&o7};
            for(int v=0; v<nv; v++) _mm512_storeu_pd(dst[v]+2*(base+e), *O[v]);
        }
#if @N2T@ > 0
        {
            const long e = @N2T4@;
            const __mmask8 mk = (__mmask8)((1u<<(2*@N2T@))-1u);
            __m512d A[8];
            for(int q=0;q<2*@N2T@;q++) A[q] = _mm512_load_pd(Gp+e*16+q*8);
            for(int q=2*@N2T@;q<8;q++) A[q] = _mm512_setzero_pd();
            __m512d o0,o1,o2,o3,o4,o5,o6,o7;
            TR8(A[0],A[1],A[2],A[3],A[4],A[5],A[6],A[7],o0,o1,o2,o3,o4,o5,o6,o7);
            __m512d* O[8] = {&o0,&o1,&o2,&o3,&o4,&o5,&o6,&o7};
            for(int v=0; v<nv; v++) _mm512_mask_storeu_pd(dst[v]+2*(base+e), mk, *O[v]);
        }
#endif
    }
}
'''
    driver = '''
static double* Xg_@N@ = 0;
static double* Cg_@N@ = 0;
void hot_@N@(long n){
    if(!Xg_@N@){ Xg_@N@ = alloc_huge_st(@N@*@SP@*16*8); Cg_@N@ = alloc_huge_st(@NE@*16*8); }
    for(long i=0;i<@N@*@SP@*16;i++) Xg_@N@[i] = 0.5 + 1e-6*(i%97);
    for(long r=0;r<n;r++){
        double* P = Xg_@N@;
        for(long y=0; y<@N@; y++) dft@N@_one(P + y*@N@*16, 1);
        for(long z=0; z<@N@; z++) dft@N@_one(P + z*16, @N@);
        if((r&1)==1) for(long i=0;i<@N2@*16;i++) Xg_@N@[i] = 0.5 + 1e-6*(i%97);
    }
}
void hot2_@N@(long which){
    if(!Xg_@N@){ Xg_@N@ = alloc_huge_st((@NE@+64*@N@)*16*8); Cg_@N@ = alloc_huge_st(@NE@*16*8); }
    double* P = Xg_@N@;
    if(which==99){ for(long i=0;i<@N2@*16;i++) P[i] = 0.5 + 1e-6*(i%97); return; }
    if(which==0 || which==2) for(long y=0; y<@N@; y++) dft@N@_one(P + y*@N@*16, 1);
    if(which==1 || which==2) for(long z=0; z<@N@; z++) dft@N@_one(P + z*16, @N@);
}
void bsweep_@N@(long which, long n){
    if(!Xg_@N@){ Xg_@N@ = alloc_huge_st(@N@*@SP@*16*8); Cg_@N@ = alloc_huge_st(@NE@*16*8); }
    for(long i=0;i<@N@*@SP@*16;i++) Xg_@N@[i] = 0.5 + 1e-6*(i%97);
    for(long i=0;i<@NE@*16;i++) Cg_@N@[i] = 0.01;
    for(long r=0;r<n;r++){
        if(which==0) dft@N@_sweep_zy(Xg_@N@);
        else if(which==2) dft@N@_sweep_x_map(Xg_@N@, Cg_@N@);
        if((r&3)==3) for(long i=0;i<@N@*@SP@*16;i+=997) Xg_@N@[i] = 0.5;
    }
}
void diag_@N@(long which, long n){
    if(!Xg_@N@){ Xg_@N@ = alloc_huge_st(@N@*@SP@*16*8); Cg_@N@ = alloc_huge_st(@N@*@SP@*16*8); }
    for(long i=0;i<@N@*@SP@*16;i++){ Xg_@N@[i] = 0.5 + 1e-6*(i%97); Cg_@N@[i] = 0.01; }
    for(long r=0;r<n;r++){
        if(which==0){ for(long x=0;x<@N@;x++) mapslab(Xg_@N@ + x*@SP@*16, Cg_@N@ + x*@SP@*16, @N2@); }
        else if(which==1) dft@N@_sweep_zy(Xg_@N@);
        else dft@N@_sweep_x_plain(Xg_@N@);
        if((r&1)==1) for(long i=0;i<@N@*@SP@*16;i+=997) Xg_@N@[i] = 0.5;
    }
}
void run_@N@(const double* x0, const double* c, double* out1, double* outm, long B, long m){
    const long NE = @NE@;
    if(!Xg_@N@){ Xg_@N@ = alloc_huge_st(@N@*@SP@*16*8); Cg_@N@ = alloc_huge_st(@N@*@SP@*16*8); }
    double* X = Xg_@N@; double* Ct = Cg_@N@;
    for(long g0=0; g0<B; g0+=8){
        int nv = (int)((B - g0) < 8 ? (B - g0) : 8);
        const double* src[8]; const double* csrc[8];
        double* d1[8]; double* dm[8];
        for(int v=0; v<8; v++){
            int vv = v < nv ? v : 0;
            src[v] = x0 + (g0+vv)*2*NE; csrc[v] = c + (g0+vv)*2*NE;
            if(v<nv){ d1[v] = out1 + (g0+v)*2*NE; dm[v] = outm + (g0+v)*2*NE; }
        }
#if XFIRST_FLAG
        ingest_@N@(csrc, Ct);   /* c in padded group layout */
        ingest_@N@(src, X);
        for(long t=0; t<m; t++){
            dft@N@_sweep_x_plain(X);
            dft@N@_sweep_zy_ms(X, Ct);
            if(t==0 && m>1) output_@N@(X, d1, nv);
        }
#else
        ingest_@N@(csrc, X);
        for(long p=0; p<@N2@; p++)
            for(long k=0; k<@N@; k++){
                _mm512_store_pd(Ct + (p*@N@+k)*16,     _mm512_load_pd(X + (k*@SP@+p)*16));
                _mm512_store_pd(Ct + (p*@N@+k)*16 + 8, _mm512_load_pd(X + (k*@SP@+p)*16 + 8));
            }
        ingest_@N@(src, X);
        for(long t=0; t<m; t++){
            dft@N@_sweep_zy(X);
            dft@N@_sweep_x_map(X, Ct);
            if(t==0 && m>1) output_@N@(X, d1, nv);
        }
#endif
        output_@N@(X, dm, nv);
        if(m==1) output_@N@(X, d1, nv);
    }
}
'''
    out = ["#define PFIN_%d %d" % (N, 1 if N in PFIN else 0)]
    for tpl in (sweeps, driver):
        d = (tpl.replace("@N2T4@", str(N2//4*4)).replace("@N2T@", str(N2%4))
               .replace("@SP@", str(SP)).replace("@N2@", str(N2))
               .replace("@SEQSW@", "1" if N >= 36 else "0")
               .replace("@NE@", str(N**3)).replace("@N@", str(N)))
        out.append(d)
    return "\n".join(code) + "\n" + "\n".join(out)

def gen_size(N, KBs, KBo=None):
    global MAPMIX, MAPX
    MAPMIX = 0 if N in _MAPMIX_OFF else 1
    MAPX = 1 if N in _MAPX_SET else 0
    code = []
    USEASM = int(os.environ.get("ASM", "0"))
    if USEASM:
        t, hh = genasm.gen_tbl(N)
        code.append(t)
        for v in ('z','y','x','xm','mb'):
            code.append(genasm.emit_prime_asm(N, v))
        code.append(f'''
static double AB_{N}[{8*(N//2+1)}*8] ALIGN64;
static double ES_{N}[{2*(N//2+2)}*8] ALIGN64;
static void dft{N}_sweep_zy_asm(double* restrict X, int domap){{
    (void)domap;
    for(long x=0; x<{N}; x++){{
        double* P = X + x*{N*N}*16;
        for(long y=0; y<{N}; y++) adft{N}_z(P + y*{N}*16, AB_{N}, ES_{N});
        for(long zz=0; zz<{N}; zz++) adft{N}_y(P + zz*16, AB_{N}, ES_{N});
    }}
}}
static void dft{N}_sweep_x_asm(double* restrict X, const double* restrict Ct){{
    adft{N}_x(X, AB_{N}, ES_{N}, Ct);
    for(long p=1; p<{N*N}; p++) adft{N}_xm(X + p*16, AB_{N}, ES_{N}, Ct + p*{N}*16, X + (p-1)*16);
    adft{N}_mb(X + ({N*N}-1)*16);
}}
''')
    code.append(gen_prime_codelet(N, KBs, f"dft{N}", fuse_map=False, KBo=KBo))
    code.append(gen_prime_codelet(N, KBs, f"dft{N}zm", fuse_map=False, KBo=KBo, map_in=True))
    code.append(gen_prime_codelet(N, KBs, f"dft{N}m", fuse_map=True, KBo=KBo))
    PXF = 1 if N in _PXF_SET else 0
    PFP = set(int(x) for x in os.environ.get("PFPRIME","").split(",") if x)
    code.append("#define MAPX_%d %d" % (N, MAPX))
    code.append("#define PXF_%d %d" % (N, PXF))
    code.append("#define PFPRIME_%d %d" % (N, 1 if N in PFP else 0))
    for tpl in (SWEEPS, INOUT, DRIVER):
        d = tpl.replace("@N@", str(N)).replace("@NE4@", str(N**3//4*4)).replace("@NT@", str(N**3%4)).replace("@NE@", str(N**3)).replace("@N2@", str(N*N))
        if USEASM or MAPX or PXF:
            d = d.replace("@MAPOUT4@", "").replace("@MAPOUTT@", "")
        else:
            d = d.replace("@MAPOUT4@", "        map2(i0,i1,&i0,&i1); map2(i2,i3,&i2,&i3); map2(i4,i5,&i4,&i5); map2(i6,i7,&i6,&i7);")
            d = d.replace("@MAPOUTT@", "        for(int q=0;q<2*@NT@;q+=2) map2(A[q],A[q+1],&A[q],&A[q+1]);".replace("@NT@", str(N**3%4)))
        code.append(d)
    return "\n".join(code)

import os
_MAPX_SET = {17}
import os as _os0
_PXF_SET = set(int(x) for x in _os0.environ.get("PXF","").split(",") if x)
MAPX = 0
MAPMIX = 1  # overridden per size in gen_size calls via global
_MAPMIX_OFF = {13}
INL = os.environ.get("INL", "noinline")
def pcfg(s, default):
    v = os.environ.get(s)
    if not v: return default
    a, b = v.split(":") if ":" in v else (v, v)
    return tuple(int(x) for x in a.split(",")), tuple(int(x) for x in b.split(","))
KB13 = pcfg("KB13", ((6,),(6,)))
KB17 = pcfg("KB17", ((4,4),(4,4)))
KB23 = pcfg("KB23", ((6,5),(6,5)))
PFH = os.environ.get("PFH", "_MM_HINT_T0")
src = ["#define PFHINT %s" % PFH, "#define USEASM_FLAG %s" % os.environ.get("ASM","0"), "#define PFCOMP %s" % os.environ.get("PFCOMP","0"), "#define MAPZB_FLAG %s" % os.environ.get("MAPZB","0"), "#define XFIRST_FLAG %s" % os.environ.get("XFIRST","0"), HDR]
src.append(gen_size_comp(6))
src.append(gen_size_comp(8))
src.append(gen_size_comp(36))
src.append(gen_size_comp(45))
src.append(gen_size(13, KB13[0], KB13[1]))
src.append(gen_size(17, KB17[0], KB17[1]))
src.append(gen_size(23, KB23[0], KB23[1]))
out = "\n".join(src).replace("INLINEATTR", INL)
out = out.replace("OPTATTR13", '__attribute__((optimize("schedule-insns,sched-pressure")))')
out = out.replace("OPTATTR17", '').replace("OPTATTR23", '')
open(os.environ.get("OUT","mine.c"),"w").write(out)
print("generated")
