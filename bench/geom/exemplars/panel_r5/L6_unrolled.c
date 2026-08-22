/* L6_unrolled -- forward complex-double 3D DFT of a 6x6x6 cube, batched.
 *
 * TECHNIQUE
 * ---------
 * A single fully-unrolled straight-line 6-point codelet, applied along all three
 * axes of the cube with no loops over the codelet body, no runtime twiddle table,
 * and no data-dependent branches.  The codelet is Good-Thomas / PFA 2x3, so it
 * carries *no* twiddle factors at all:
 *
 *     6 = 2*3, gcd(2,3)=1.  With  j = (3*j1 + 2*j2) mod 6  and
 *     k = (3*k1 + 4*k2) mod 6  the kernel factors exactly as
 *     w6^(jk) = (-1)^(j1 k1) * w3^(j2 k2),   w3 = exp(-2 pi i/3),
 *     i.e. DFT6 = DFT2 (x) DFT3 with a pure index permutation and nothing else.
 *
 * The only irrational constants in the whole transform are 1/2 and sqrt(3)/2,
 * both written literally.  Multiplication by -i*sqrt(3)/2 in the 3-point module
 * is done as one re/im swap (vpermilpd) followed by an FMA against the fixed
 * vector (+s,-s,+s,-s), so no complex multiply routine ever appears.
 *
 * OPERATION COUNT (per 6^3 volume)
 * --------------------------------
 *   DFT6 (PFA 2x3):  3*DFT2 + 2*DFT3
 *       3*DFT2 = 12 real adds
 *       1*DFT3 = 6 real adds + 1 FMA-pair(m) + 4 FMA  ->  6 arithmetic
 *                instructions + 1 shuffle per SIMD vector, 18 real flops
 *       DFT6   = 18 arithmetic instructions + 2 shuffles per vector,
 *                48 real flops per 6-point line  (= the provably optimal
 *                Good-Thomas count; FFTW's n1_6 reaches the same 48/36)
 *   volume = 3 axes * 36 lines = 108 line-DFTs
 *          = 5184 real flops and 3888 real arithmetic instructions
 *            (108 * 36) in scalar-equivalent terms
 *   SIMD: each __m256d holds 2 complex, so 54 vector codelet instances per
 *   volume = 972 vector arithmetic uops + 108 in-codelet shuffles.  On the
 *   Skylake/Cascade-Lake port scheme (FP add and FMA both on ports 0 and 1)
 *   that is a hard floor of 486 cycles per 6^3 volume for the whole 3D
 *   transform; the rest of the kernel (216 loads, 216 stores, 324 port-5
 *   shuffles in the y+z-fused variant) fits underneath it.
 *
 * LAYOUT / SIMD
 * -------------
 * Interleaved complex throughout (the driver's own layout, so no repack pass).
 * A __m256d = 2 adjacent complex = 2 adjacent z (or 2 adjacent (y,z)) values.
 *   x-pass: lanes are 2 adjacent (y,z) positions, axis stride 36 complex.
 *   y-pass: lanes are 2 adjacent z, axis stride 6 complex.
 *   z-pass: the axis is contiguous, so 2 whole z-pencils are brought into
 *           "z-major" form with 6 vperm2f128 and pushed back with 6 more; the
 *           12 outputs of the pair land as 6 consecutive 32-byte stores that
 *           cover exactly 3 whole 64-byte cache lines (=> clean NT stores).
 * Every load and store in the transform is 32-byte aligned by construction
 * (volume = 3456 B = 54*64 B, plane = 576 B, pencil = 96 B).
 *
 * ASSUMPTIONS
 * -----------
 *  * L == 6 only.
 *  * in/out are 64-byte aligned, as fft3d_api.h guarantees, and distinct.
 *  * double _Complex is a pair of adjacent doubles (C99 guarantees this).
 *  * 32 kernel variants ({3-pass, y+z fused in registers} x {normal, streaming
 *    stores} x {no prefetch, T0/T1 input prefetch, prefetchw output prefetch at
 *    1-2 volumes} + split-z-store "_s" shapes that replace the z-pass output
 *    vperm2f128s with direct half-vector stores + mixed-width AVX-512 shapes,
 *    panel_r5) are RACED AGAINST EACH OTHER AND VALIDATED against a
 *    scalar reference inside fft3d_create(); a variant that disagrees with the
 *    reference by more than 1e-11 relative is disqualified and can never be
 *    selected.  This is what makes an untestable-locally code path safe.
 *    L6_FORCE=<name> (env or -DL6_FORCE_DEFAULT) bypasses the race for node
 *    A/B tests; a forced pick is reported as variant=<name>! in the
 *    description, next to clk=<256-bit>/<512-bit> sustained-GHz probes.
 */

#include "fft3d_api.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define L6_HAVE_AVX2 1
#endif

#define L6_L    6
#define L6_VOL  216            /* complex per volume */
#define L6_VD   432            /* doubles  per volume */
#define L6_MARGIN 0.025        /* a later race candidate must win by >2.5%.
                                * Raised from 1.5% in panel_r4: the r3 node race
                                * promoted fused_pfw at B=4096 on a within-2%
                                * race win that the driver then measured as a 2%
                                * loss (L3-marginal regime); a wider hysteresis
                                * keeps marginal challengers out. */

static const double L6_S3 = 0.86602540378443864676372317075294; /* sqrt(3)/2 */

/* ------------------------------------------------------------------ *
 * Scalar reference: the same PFA 2x3 codelet, three strided passes.
 * Used as the plan-time correctness witness and as the portable path.
 * ------------------------------------------------------------------ */

#define SD6(A0,A1,A2,A3,A4,A5, R0,R1,R2,R3,R4,R5)                       \
    do {                                                                \
        double p0r=(A0##r)+(A3##r), p0i=(A0##i)+(A3##i);                 \
        double q0r=(A0##r)-(A3##r), q0i=(A0##i)-(A3##i);                 \
        double p1r=(A2##r)+(A5##r), p1i=(A2##i)+(A5##i);                 \
        double q1r=(A2##r)-(A5##r), q1i=(A2##i)-(A5##i);                 \
        double p2r=(A4##r)+(A1##r), p2i=(A4##i)+(A1##i);                 \
        double q2r=(A4##r)-(A1##r), q2i=(A4##i)-(A1##i);                 \
        double ar=p1r+p2r, ai=p1i+p2i, br=p1r-p2r, bi=p1i-p2i;          \
        double cr=q1r+q2r, ci=q1i+q2i, dr=q1r-q2r, di=q1i-q2i;          \
        double mr=p0r-0.5*ar, mi=p0i-0.5*ai;                            \
        double nr=q0r-0.5*cr, ni=q0i-0.5*ci;                            \
        double jbr= L6_S3*bi, jbi=-L6_S3*br;                            \
        double jdr= L6_S3*di, jdi=-L6_S3*dr;                            \
        (R0##r)=p0r+ar; (R0##i)=p0i+ai;                                 \
        (R3##r)=q0r+cr; (R3##i)=q0i+ci;                                 \
        (R4##r)=mr+jbr; (R4##i)=mi+jbi;                                 \
        (R2##r)=mr-jbr; (R2##i)=mi-jbi;                                 \
        (R1##r)=nr+jdr; (R1##i)=ni+jdi;                                 \
        (R5##r)=nr-jdr; (R5##i)=ni-jdi;                                 \
    } while (0)

static void l6_line_scalar(const double *s, long ss, double *d, long ds)
{
    /* ss, ds are strides in COMPLEX units */
    double x0r=s[0],        x0i=s[1];
    double x1r=s[2*ss],     x1i=s[2*ss+1];
    double x2r=s[4*ss],     x2i=s[4*ss+1];
    double x3r=s[6*ss],     x3i=s[6*ss+1];
    double x4r=s[8*ss],     x4i=s[8*ss+1];
    double x5r=s[10*ss],    x5i=s[10*ss+1];
    double y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i,y4r,y4i,y5r,y5i;
    SD6(x0,x1,x2,x3,x4,x5, y0,y1,y2,y3,y4,y5);
    d[0]=y0r;       d[1]=y0i;
    d[2*ds]=y1r;    d[2*ds+1]=y1i;
    d[4*ds]=y2r;    d[4*ds+1]=y2i;
    d[6*ds]=y3r;    d[6*ds+1]=y3i;
    d[8*ds]=y4r;    d[8*ds+1]=y4i;
    d[10*ds]=y5r;   d[10*ds+1]=y5i;
}

static void l6_run_scalar(double *t1, double *t2,
                          const double *in, double *out, long nvol)
{
    for (long b = 0; b < nvol; ++b) {
        const double *ip = in  + b * (long)L6_VD;
        double       *op = out + b * (long)L6_VD;
        for (int p = 0; p < 36; ++p)                  /* x: stride 36 complex */
            l6_line_scalar(ip + 2*p, 36, t1 + 2*p, 36);
        for (int x = 0; x < 6; ++x)                   /* y: stride 6 complex  */
            for (int z = 0; z < 6; ++z)
                l6_line_scalar(t1 + 2*(36*x + z), 6, t2 + 2*(36*x + z), 6);
        for (int x = 0; x < 6; ++x)                   /* z: stride 1 complex  */
            for (int y = 0; y < 6; ++y)
                l6_line_scalar(t2 + 2*(36*x + 6*y), 1, op + 2*(36*x + 6*y), 1);
    }
}

/* ------------------------------------------------------------------ *
 * AVX2 / FMA kernels.  __m256d = 2 complex.
 * ------------------------------------------------------------------ */
#ifdef L6_HAVE_AVX2

/* One straight-line PFA 2x3 six-point codelet on 2 complex lanes.
 * 18 arithmetic instructions + 2 shuffles.  Safe to use in place. */
#define VD6(i0,i1,i2,i3,i4,i5, o0,o1,o2,o3,o4,o5)                       \
    do {                                                                \
        __m256d _p0 = _mm256_add_pd(i0,i3), _q0 = _mm256_sub_pd(i0,i3); \
        __m256d _p1 = _mm256_add_pd(i2,i5), _q1 = _mm256_sub_pd(i2,i5); \
        __m256d _p2 = _mm256_add_pd(i4,i1), _q2 = _mm256_sub_pd(i4,i1); \
        __m256d _a  = _mm256_add_pd(_p1,_p2), _b = _mm256_sub_pd(_p1,_p2); \
        __m256d _c  = _mm256_add_pd(_q1,_q2), _d = _mm256_sub_pd(_q1,_q2); \
        __m256d _m  = _mm256_fnmadd_pd(vhalf,_a,_p0);                   \
        __m256d _n  = _mm256_fnmadd_pd(vhalf,_c,_q0);                   \
        __m256d _bs = _mm256_permute_pd(_b,0x5);                        \
        __m256d _ds = _mm256_permute_pd(_d,0x5);                        \
        __m256d _s0 = _mm256_add_pd(_p0,_a);                            \
        __m256d _s3 = _mm256_add_pd(_q0,_c);                            \
        __m256d _s4 = _mm256_fmadd_pd (vk,_bs,_m);                      \
        __m256d _s2 = _mm256_fnmadd_pd(vk,_bs,_m);                      \
        __m256d _s1 = _mm256_fmadd_pd (vk,_ds,_n);                      \
        __m256d _s5 = _mm256_fnmadd_pd(vk,_ds,_n);                      \
        o0=_s0; o1=_s1; o2=_s2; o3=_s3; o4=_s4; o5=_s5;                 \
    } while (0)

#define VSET __m256d vhalf = _mm256_set1_pd(0.5);                        \
             __m256d vk    = _mm256_setr_pd(L6_S3,-L6_S3,L6_S3,-L6_S3)

/* Software prefetch hooks for the x-pass loop (adopted from L6_pfa's v8,
 * which won the large-batch cases in panel_r1 with exactly this: touch the
 * NEXT volume's input, 3 cache lines per x-pass group = all 54 lines of a
 * 3456-B volume, giving one volume (~800 cycles) of lead time).  Prefetch
 * never faults, so running past the end of the batch is safe.
 *
 * New in panel_r3:
 *  - T1 hint on the input (adopted from L6_pfa's panel_r3 pf=3 column, which
 *    beat T0 by 7% in wallaby's DRAM regime): fills L2 without displacing the
 *    L1-resident scratch.
 *  - write-intent prefetch (prefetchw) of the NEXT volume's OUTPUT lines.
 *    The node rejected NT stores at every batch size (panel_r2, both L=6
 *    entries), so the scored kernels pay a write-allocate RFO per output
 *    line; prefetchw issues that RFO one volume early instead of letting it
 *    stall the store buffer.  __builtin_prefetch(p,1,3) compiles everywhere
 *    and emits prefetchw when the target has PRFCHW (Cascade Lake and
 *    Sapphire Rapids both do). */
#define L6_PF_NONE(SRC,OUT,g)  do { } while (0)
#define L6_PF_AT(SRC,g,DIST,HINT)                                       \
    do {                                                                \
        const char *_pf = (const char *)((SRC) + (DIST)*(long)L6_VD)    \
                          + 192*(g);                                    \
        _mm_prefetch(_pf,      HINT);                                   \
        _mm_prefetch(_pf + 64, HINT);                                   \
        _mm_prefetch(_pf + 128,HINT);                                   \
    } while (0)
#define L6_PF_W_AT(OUT,g,DIST)                                          \
    do {                                                                \
        const char *_pw = (const char *)((OUT) + (DIST)*(long)L6_VD)    \
                          + 192*(g);                                    \
        __builtin_prefetch(_pw,       1, 3);                            \
        __builtin_prefetch(_pw + 64,  1, 3);                            \
        __builtin_prefetch(_pw + 128, 1, 3);                            \
    } while (0)
#define L6_PF_T0_1(SRC,OUT,g)  L6_PF_AT(SRC,g,1,_MM_HINT_T0)
#define L6_PF_T0_2(SRC,OUT,g)  L6_PF_AT(SRC,g,2,_MM_HINT_T0)
#define L6_PF_T1_1(SRC,OUT,g)  L6_PF_AT(SRC,g,1,_MM_HINT_T1)
#define L6_PF_T0W_1(SRC,OUT,g)                                          \
    do { L6_PF_AT(SRC,g,1,_MM_HINT_T0); L6_PF_W_AT(OUT,g,1); } while (0)
#define L6_PF_T0W_2(SRC,OUT,g)                                          \
    do { L6_PF_AT(SRC,g,2,_MM_HINT_T0); L6_PF_W_AT(OUT,g,2); } while (0)
#define L6_PF_T1W_1(SRC,OUT,g)                                          \
    do { L6_PF_AT(SRC,g,1,_MM_HINT_T1); L6_PF_W_AT(OUT,g,1); } while (0)
#define L6_PF_W_1(SRC,OUT,g)   L6_PF_W_AT(OUT,g,1)

/* x-pass: in -> t1.  lanes = 2 adjacent (y,z); axis stride 72 doubles.
 * PF(SRC,OUT,g) is a prefetch hook run once per group; OUT is the volume's
 * final output pointer, only used by the prefetchw hooks. */
#define L6_PASS_X(SRC,DST,OUT,PF)                                       \
    do {                                                                \
        for (int g = 0; g < 18; ++g) {                                  \
            const double *s = (SRC) + 4*g;                              \
            double *d = (DST) + 4*g;                                    \
            PF(SRC,OUT,g);                                              \
            __m256d v0=_mm256_load_pd(s+  0), v1=_mm256_load_pd(s+ 72); \
            __m256d v2=_mm256_load_pd(s+144), v3=_mm256_load_pd(s+216); \
            __m256d v4=_mm256_load_pd(s+288), v5=_mm256_load_pd(s+360); \
            VD6(v0,v1,v2,v3,v4,v5, v0,v1,v2,v3,v4,v5);                  \
            _mm256_store_pd(d+  0,v0); _mm256_store_pd(d+ 72,v1);       \
            _mm256_store_pd(d+144,v2); _mm256_store_pd(d+216,v3);       \
            _mm256_store_pd(d+288,v4); _mm256_store_pd(d+360,v5);       \
        }                                                               \
    } while (0)

/* y-pass: t1 -> t2.  lanes = 2 adjacent z; axis stride 12 doubles. */
#define L6_PASS_Y(SRC,DST)                                              \
    do {                                                                \
        for (int x = 0; x < 6; ++x) {                                   \
            for (int c = 0; c < 3; ++c) {                               \
                const double *s = (SRC) + 72*x + 4*c;                   \
                double *d = (DST) + 72*x + 4*c;                         \
                __m256d v0=_mm256_load_pd(s+ 0), v1=_mm256_load_pd(s+12);\
                __m256d v2=_mm256_load_pd(s+24), v3=_mm256_load_pd(s+36);\
                __m256d v4=_mm256_load_pd(s+48), v5=_mm256_load_pd(s+60);\
                VD6(v0,v1,v2,v3,v4,v5, v0,v1,v2,v3,v4,v5);              \
                _mm256_store_pd(d+ 0,v0); _mm256_store_pd(d+12,v1);     \
                _mm256_store_pd(d+24,v2); _mm256_store_pd(d+36,v3);     \
                _mm256_store_pd(d+48,v4); _mm256_store_pd(d+60,v5);     \
            }                                                           \
        }                                                               \
    } while (0)

/* z-pass: SRC -> DST, two z-pencils at a time, in-register transposed.
 * ST is the store intrinsic (_mm256_store_pd or _mm256_stream_pd). */
#define L6_PASS_Z(SRC,DST,ST)                                           \
    do {                                                                \
        for (int x = 0; x < 6; ++x) {                                   \
            for (int yp = 0; yp < 3; ++yp) {                            \
                const double *s = (SRC) + 72*x + 24*yp;                 \
                double *d = (DST) + 72*x + 24*yp;                       \
                __m256d A0=_mm256_load_pd(s+ 0), A1=_mm256_load_pd(s+ 4);\
                __m256d A2=_mm256_load_pd(s+ 8), B0=_mm256_load_pd(s+12);\
                __m256d B1=_mm256_load_pd(s+16), B2=_mm256_load_pd(s+20);\
                __m256d v0=_mm256_permute2f128_pd(A0,B0,0x20);          \
                __m256d v1=_mm256_permute2f128_pd(A0,B0,0x31);          \
                __m256d v2=_mm256_permute2f128_pd(A1,B1,0x20);          \
                __m256d v3=_mm256_permute2f128_pd(A1,B1,0x31);          \
                __m256d v4=_mm256_permute2f128_pd(A2,B2,0x20);          \
                __m256d v5=_mm256_permute2f128_pd(A2,B2,0x31);          \
                VD6(v0,v1,v2,v3,v4,v5, v0,v1,v2,v3,v4,v5);              \
                ST(d+ 0,_mm256_permute2f128_pd(v0,v1,0x20));            \
                ST(d+ 4,_mm256_permute2f128_pd(v2,v3,0x20));            \
                ST(d+ 8,_mm256_permute2f128_pd(v4,v5,0x20));            \
                ST(d+12,_mm256_permute2f128_pd(v0,v1,0x31));            \
                ST(d+16,_mm256_permute2f128_pd(v2,v3,0x31));            \
                ST(d+20,_mm256_permute2f128_pd(v4,v5,0x31));            \
            }                                                           \
        }                                                               \
    } while (0)

/* --- variant A: three separate passes ---------------------------------- */
#define L6_DEF_3PASS(NAME,ST,PF)                                        \
static void NAME(double *t1, double *t2,                                \
                 const double *in, double *out, long nvol)              \
{                                                                       \
    VSET;                                                               \
    for (long b = 0; b < nvol; ++b) {                                   \
        const double *ip = in  + b * (long)L6_VD;                       \
        double       *op = out + b * (long)L6_VD;                       \
        L6_PASS_X(ip, t1, op, PF);                                      \
        L6_PASS_Y(t1, t2);                                              \
        L6_PASS_Z(t2, op, ST);                                          \
    }                                                                   \
}

L6_DEF_3PASS(l6_run_3pass,        _mm256_store_pd,  L6_PF_NONE)
L6_DEF_3PASS(l6_run_3pass_pf,     _mm256_store_pd,  L6_PF_T0_1)
L6_DEF_3PASS(l6_run_3pass_pf2,    _mm256_store_pd,  L6_PF_T0_2)
L6_DEF_3PASS(l6_run_3pass_pft1,   _mm256_store_pd,  L6_PF_T1_1)
L6_DEF_3PASS(l6_run_3pass_pfw,    _mm256_store_pd,  L6_PF_T0W_1)
L6_DEF_3PASS(l6_run_3pass_pfw2,   _mm256_store_pd,  L6_PF_T0W_2)
L6_DEF_3PASS(l6_run_3pass_pft1w,  _mm256_store_pd,  L6_PF_T1W_1)
L6_DEF_3PASS(l6_run_3pass_nt,     _mm256_stream_pd, L6_PF_NONE)
L6_DEF_3PASS(l6_run_3pass_nt_pf,  _mm256_stream_pd, L6_PF_T0_1)
L6_DEF_3PASS(l6_run_3pass_nt_pf2, _mm256_stream_pd, L6_PF_T0_2)
L6_DEF_3PASS(l6_run_3pass_nt_pft1,_mm256_stream_pd, L6_PF_T1_1)

/* --- variant B: x-pass, then y and z fused per x-plane in registers ----
 * The 6x6 (y,z) plane is 18 __m256d; with the codelet temporaries that is
 * ~26 live vectors, which fits the 32 ymm of an AVX-512VL machine but not
 * the 16 of plain AVX2.  Raced at plan time against variant A.          */
#define L6_DEF_FUSED(NAME,ST,PF)                                        \
static void NAME(double *t1, double *t2,                                \
                 const double *in, double *out, long nvol)              \
{                                                                       \
    VSET;                                                               \
    (void)t2;                                                           \
    for (long b = 0; b < nvol; ++b) {                                   \
        const double *ip = in  + b * (long)L6_VD;                       \
        double       *op = out + b * (long)L6_VD;                       \
        L6_PASS_X(ip, t1, op, PF);                                      \
        for (int x = 0; x < 6; ++x) {                                   \
            const double *s = t1 + 72*x;                                \
            double *d = op + 72*x;                                      \
            __m256d P00=_mm256_load_pd(s+ 0),P01=_mm256_load_pd(s+ 4),  \
                    P02=_mm256_load_pd(s+ 8);                           \
            __m256d P10=_mm256_load_pd(s+12),P11=_mm256_load_pd(s+16),  \
                    P12=_mm256_load_pd(s+20);                           \
            __m256d P20=_mm256_load_pd(s+24),P21=_mm256_load_pd(s+28),  \
                    P22=_mm256_load_pd(s+32);                           \
            __m256d P30=_mm256_load_pd(s+36),P31=_mm256_load_pd(s+40),  \
                    P32=_mm256_load_pd(s+44);                           \
            __m256d P40=_mm256_load_pd(s+48),P41=_mm256_load_pd(s+52),  \
                    P42=_mm256_load_pd(s+56);                           \
            __m256d P50=_mm256_load_pd(s+60),P51=_mm256_load_pd(s+64),  \
                    P52=_mm256_load_pd(s+68);                           \
            VD6(P00,P10,P20,P30,P40,P50, P00,P10,P20,P30,P40,P50);      \
            VD6(P01,P11,P21,P31,P41,P51, P01,P11,P21,P31,P41,P51);      \
            VD6(P02,P12,P22,P32,P42,P52, P02,P12,P22,P32,P42,P52);      \
            L6_ZPAIR(P00,P01,P02,P10,P11,P12, d+ 0, ST);                \
            L6_ZPAIR(P20,P21,P22,P30,P31,P32, d+24, ST);                \
            L6_ZPAIR(P40,P41,P42,P50,P51,P52, d+48, ST);                \
        }                                                               \
    }                                                                   \
}

#define L6_ZPAIR(A0,A1,A2,B0,B1,B2, D, ST)                              \
    do {                                                                \
        __m256d w0=_mm256_permute2f128_pd(A0,B0,0x20);                  \
        __m256d w1=_mm256_permute2f128_pd(A0,B0,0x31);                  \
        __m256d w2=_mm256_permute2f128_pd(A1,B1,0x20);                  \
        __m256d w3=_mm256_permute2f128_pd(A1,B1,0x31);                  \
        __m256d w4=_mm256_permute2f128_pd(A2,B2,0x20);                  \
        __m256d w5=_mm256_permute2f128_pd(A2,B2,0x31);                  \
        VD6(w0,w1,w2,w3,w4,w5, w0,w1,w2,w3,w4,w5);                      \
        ST((D)+ 0,_mm256_permute2f128_pd(w0,w1,0x20));                  \
        ST((D)+ 4,_mm256_permute2f128_pd(w2,w3,0x20));                  \
        ST((D)+ 8,_mm256_permute2f128_pd(w4,w5,0x20));                  \
        ST((D)+12,_mm256_permute2f128_pd(w0,w1,0x31));                  \
        ST((D)+16,_mm256_permute2f128_pd(w2,w3,0x31));                  \
        ST((D)+20,_mm256_permute2f128_pd(w4,w5,0x31));                  \
    } while (0)

/* --- split-store z output (new in panel_r4) ---------------------------
 * The z-pass output permutes are pure data movement: w_k = (A_k | B_k)
 * with pencil A wanted contiguous at D+0..11 and pencil B at D+12..23.
 * Instead of 6 vperm2f128 (port 5) + 6 ymm stores, store each half
 * directly: 6 xmm stores of the low halves + 6 vextractf128-to-memory of
 * the high halves.  On SKX/CLX extract-to-memory is handled by the store
 * pipes with no shuffle uop, so this trades 6 port-5 uops per pencil pair
 * for 6 extra store uops (port 4 has headroom: 216 -> 324 stores/volume
 * against the 486-cycle FP floor).  Port-5 pressure per volume drops
 * 324 -> 216 (fused shape) or -> 108 (3pass with split loads too).
 * All 16-byte stores are 16-byte aligned by construction. */
#define L6_ZSPLITST(w0,w1,w2,w3,w4,w5, D)                               \
    do {                                                                \
        _mm_store_pd((D)+ 0,_mm256_castpd256_pd128(w0));                \
        _mm_store_pd((D)+ 2,_mm256_castpd256_pd128(w1));                \
        _mm_store_pd((D)+ 4,_mm256_castpd256_pd128(w2));                \
        _mm_store_pd((D)+ 6,_mm256_castpd256_pd128(w3));                \
        _mm_store_pd((D)+ 8,_mm256_castpd256_pd128(w4));                \
        _mm_store_pd((D)+10,_mm256_castpd256_pd128(w5));                \
        _mm_store_pd((D)+12,_mm256_extractf128_pd(w0,1));               \
        _mm_store_pd((D)+14,_mm256_extractf128_pd(w1,1));               \
        _mm_store_pd((D)+16,_mm256_extractf128_pd(w2,1));               \
        _mm_store_pd((D)+18,_mm256_extractf128_pd(w3,1));               \
        _mm_store_pd((D)+20,_mm256_extractf128_pd(w4,1));               \
        _mm_store_pd((D)+22,_mm256_extractf128_pd(w5,1));               \
    } while (0)

/* build (A_k | B_k) with an insertf128 *from memory* (load pipe + blend,
 * no port-5 shuffle) instead of ymm load + vperm2f128 */
#define L6_INS2(LO,HI) \
    _mm256_insertf128_pd(_mm256_castpd128_pd256(LO),(HI),1)

/* z-pass, fully split: 12 half-loads in, VD6, 12 half-stores out;
 * zero z-transpose shuffles.  Bit-identical arithmetic to L6_PASS_Z. */
#define L6_PASS_Z_S(SRC,DST)                                            \
    do {                                                                \
        for (int x = 0; x < 6; ++x) {                                   \
            for (int yp = 0; yp < 3; ++yp) {                            \
                const double *s = (SRC) + 72*x + 24*yp;                 \
                double *d = (DST) + 72*x + 24*yp;                       \
                __m256d v0=L6_INS2(_mm_load_pd(s+ 0),_mm_load_pd(s+12));\
                __m256d v1=L6_INS2(_mm_load_pd(s+ 2),_mm_load_pd(s+14));\
                __m256d v2=L6_INS2(_mm_load_pd(s+ 4),_mm_load_pd(s+16));\
                __m256d v3=L6_INS2(_mm_load_pd(s+ 6),_mm_load_pd(s+18));\
                __m256d v4=L6_INS2(_mm_load_pd(s+ 8),_mm_load_pd(s+20));\
                __m256d v5=L6_INS2(_mm_load_pd(s+10),_mm_load_pd(s+22));\
                VD6(v0,v1,v2,v3,v4,v5, v0,v1,v2,v3,v4,v5);              \
                L6_ZSPLITST(v0,v1,v2,v3,v4,v5, d);                      \
            }                                                           \
        }                                                               \
    } while (0)

/* in-register inputs (fused shape): keep the 6 input permutes, split the
 * 6 output permutes into direct half-stores */
#define L6_ZPAIR_S(A0,A1,A2,B0,B1,B2, D)                                \
    do {                                                                \
        __m256d w0=_mm256_permute2f128_pd(A0,B0,0x20);                  \
        __m256d w1=_mm256_permute2f128_pd(A0,B0,0x31);                  \
        __m256d w2=_mm256_permute2f128_pd(A1,B1,0x20);                  \
        __m256d w3=_mm256_permute2f128_pd(A1,B1,0x31);                  \
        __m256d w4=_mm256_permute2f128_pd(A2,B2,0x20);                  \
        __m256d w5=_mm256_permute2f128_pd(A2,B2,0x31);                  \
        VD6(w0,w1,w2,w3,w4,w5, w0,w1,w2,w3,w4,w5);                      \
        L6_ZSPLITST(w0,w1,w2,w3,w4,w5, D);                              \
    } while (0)

/* variant A-s: three passes, split z */
#define L6_DEF_3PASS_S(NAME,PF)                                         \
static void NAME(double *t1, double *t2,                                \
                 const double *in, double *out, long nvol)              \
{                                                                       \
    VSET;                                                               \
    for (long b = 0; b < nvol; ++b) {                                   \
        const double *ip = in  + b * (long)L6_VD;                       \
        double       *op = out + b * (long)L6_VD;                       \
        L6_PASS_X(ip, t1, op, PF);                                      \
        L6_PASS_Y(t1, t2);                                              \
        L6_PASS_Z_S(t2, op);                                            \
    }                                                                   \
}

/* variant B-s: x pass, then y+z fused with split z stores */
#define L6_DEF_FUSED_S(NAME,PF)                                         \
static void NAME(double *t1, double *t2,                                \
                 const double *in, double *out, long nvol)              \
{                                                                       \
    VSET;                                                               \
    (void)t2;                                                           \
    for (long b = 0; b < nvol; ++b) {                                   \
        const double *ip = in  + b * (long)L6_VD;                       \
        double       *op = out + b * (long)L6_VD;                       \
        L6_PASS_X(ip, t1, op, PF);                                      \
        for (int x = 0; x < 6; ++x) {                                   \
            const double *s = t1 + 72*x;                                \
            double *d = op + 72*x;                                      \
            __m256d P00=_mm256_load_pd(s+ 0),P01=_mm256_load_pd(s+ 4),  \
                    P02=_mm256_load_pd(s+ 8);                           \
            __m256d P10=_mm256_load_pd(s+12),P11=_mm256_load_pd(s+16),  \
                    P12=_mm256_load_pd(s+20);                           \
            __m256d P20=_mm256_load_pd(s+24),P21=_mm256_load_pd(s+28),  \
                    P22=_mm256_load_pd(s+32);                           \
            __m256d P30=_mm256_load_pd(s+36),P31=_mm256_load_pd(s+40),  \
                    P32=_mm256_load_pd(s+44);                           \
            __m256d P40=_mm256_load_pd(s+48),P41=_mm256_load_pd(s+52),  \
                    P42=_mm256_load_pd(s+56);                           \
            __m256d P50=_mm256_load_pd(s+60),P51=_mm256_load_pd(s+64),  \
                    P52=_mm256_load_pd(s+68);                           \
            VD6(P00,P10,P20,P30,P40,P50, P00,P10,P20,P30,P40,P50);      \
            VD6(P01,P11,P21,P31,P41,P51, P01,P11,P21,P31,P41,P51);      \
            VD6(P02,P12,P22,P32,P42,P52, P02,P12,P22,P32,P42,P52);      \
            L6_ZPAIR_S(P00,P01,P02,P10,P11,P12, d+ 0);                  \
            L6_ZPAIR_S(P20,P21,P22,P30,P31,P32, d+24);                  \
            L6_ZPAIR_S(P40,P41,P42,P50,P51,P52, d+48);                  \
        }                                                               \
    }                                                                   \
}

L6_DEF_FUSED(l6_run_fused,        _mm256_store_pd,  L6_PF_NONE)
L6_DEF_FUSED(l6_run_fused_pf,     _mm256_store_pd,  L6_PF_T0_1)
L6_DEF_FUSED(l6_run_fused_pf2,    _mm256_store_pd,  L6_PF_T0_2)
L6_DEF_FUSED(l6_run_fused_pft1,   _mm256_store_pd,  L6_PF_T1_1)
L6_DEF_FUSED(l6_run_fused_pfw,    _mm256_store_pd,  L6_PF_T0W_1)
L6_DEF_FUSED(l6_run_fused_pfw2,   _mm256_store_pd,  L6_PF_T0W_2)
L6_DEF_FUSED(l6_run_fused_pft1w,  _mm256_store_pd,  L6_PF_T1W_1)
L6_DEF_FUSED(l6_run_fused_nt,     _mm256_stream_pd, L6_PF_NONE)

/* split-z shapes (panel_r4); normal stores only -- NT needs full-line
 * 32-byte bursts, which the half-store form deliberately gives up */
L6_DEF_3PASS_S(l6_run_3pass_s,      L6_PF_NONE)
L6_DEF_3PASS_S(l6_run_3pass_s_pfw,  L6_PF_T0W_1)
L6_DEF_FUSED_S(l6_run_fused_s,      L6_PF_NONE)
L6_DEF_FUSED_S(l6_run_fused_s_pf,   L6_PF_T0_1)
L6_DEF_FUSED_S(l6_run_fused_s_pfw,  L6_PF_T0W_1)

/* ------------------------------------------------------------------ *
 * AVX-512 mixed-width kernels (new in panel_r5).  __m512d = 4 complex.
 *
 * Motivation (panel_r4 VERDICT §5a/§6): the node's measured AVX2 clock is
 * 3.89 GHz, so B=1 at 0.219 us is 852 cycles against the 486-cycle FP-port
 * floor -- the bottleneck is NOT the FP ports, and on a 1-FMA-unit Gold
 * 5218 zmm buys zero FP throughput anyway.  What zmm does buy is ~25-30%
 * fewer uops per volume (1728 -> 1266..1464), which attacks every
 * non-port limit (front-end width, OOO window reach across the pass
 * boundaries) at once.  Direct node evidence that this trade wins on this
 * part: L17_rader's mixed zmm+ymm plane passes were selected by the node
 * tuner in all four cells (panel_r4), and L8_batchsimd holds L=8 B=1 with
 * a full-zmm kernel.  All zmm kernels race in the same tournament and
 * must beat the ymm incumbents by the margin, so licence downclocking on
 * the node disqualifies them automatically if it dominates.
 *
 * Shapes:
 *   x-pass zmm: lanes = 4 adjacent (y,z) sites, 9 groups, all 64B-aligned.
 *   y-pass zmm+ymm: per x-plane, one zmm codelet on z=0..3 (loadu, the
 *       12-double row stride makes half the accesses line-split -- accepted,
 *       measured) plus one ymm codelet on the z=4,5 tail (32B-aligned).
 *   z-pass 'zt': 4 whole z-pencils per iteration, 6 aligned zmm loads,
 *       in-register 4x6-complex transpose (12 vpermt2pd + 6 blends each
 *       way, 4+6 index constants), one zmm codelet, 6 aligned stores.
 *       594 uops/volume vs the ymm pairing's 792.
 * ------------------------------------------------------------------ */
#if defined(__AVX512F__) && defined(__AVX512VL__) && defined(__AVX512DQ__)
#define L6_HAVE_AVX512 1

#define VD6Z(i0,i1,i2,i3,i4,i5, o0,o1,o2,o3,o4,o5)                       \
    do {                                                                \
        __m512d _p0 = _mm512_add_pd(i0,i3), _q0 = _mm512_sub_pd(i0,i3); \
        __m512d _p1 = _mm512_add_pd(i2,i5), _q1 = _mm512_sub_pd(i2,i5); \
        __m512d _p2 = _mm512_add_pd(i4,i1), _q2 = _mm512_sub_pd(i4,i1); \
        __m512d _a  = _mm512_add_pd(_p1,_p2), _b = _mm512_sub_pd(_p1,_p2); \
        __m512d _c  = _mm512_add_pd(_q1,_q2), _d = _mm512_sub_pd(_q1,_q2); \
        __m512d _m  = _mm512_fnmadd_pd(vhalfz,_a,_p0);                  \
        __m512d _n  = _mm512_fnmadd_pd(vhalfz,_c,_q0);                  \
        __m512d _bs = _mm512_permute_pd(_b,0x55);                       \
        __m512d _ds = _mm512_permute_pd(_d,0x55);                       \
        __m512d _s0 = _mm512_add_pd(_p0,_a);                            \
        __m512d _s3 = _mm512_add_pd(_q0,_c);                            \
        __m512d _s4 = _mm512_fmadd_pd (vkz,_bs,_m);                     \
        __m512d _s2 = _mm512_fnmadd_pd(vkz,_bs,_m);                     \
        __m512d _s1 = _mm512_fmadd_pd (vkz,_ds,_n);                     \
        __m512d _s5 = _mm512_fnmadd_pd(vkz,_ds,_n);                     \
        o0=_s0; o1=_s1; o2=_s2; o3=_s3; o4=_s4; o5=_s5;                 \
    } while (0)

#define VSETZ __m512d vhalfz = _mm512_set1_pd(0.5);                      \
              __m512d vkz    = _mm512_setr_pd(L6_S3,-L6_S3,L6_S3,-L6_S3, \
                                              L6_S3,-L6_S3,L6_S3,-L6_S3)

/* prefetch hooks for the 9-group zmm x-pass: 6 lines per group = all 54
 * lines of the next volume, same coverage as the ymm hooks */
#define L6Z_PF_NONE(SRC,OUT,g)  do { } while (0)
#define L6Z_PF_T0_1(SRC,OUT,g)                                          \
    do {                                                                \
        const char *_pf = (const char *)((SRC) + (long)L6_VD) + 384*(g);\
        _mm_prefetch(_pf,      _MM_HINT_T0);                            \
        _mm_prefetch(_pf +  64,_MM_HINT_T0);                            \
        _mm_prefetch(_pf + 128,_MM_HINT_T0);                            \
        _mm_prefetch(_pf + 192,_MM_HINT_T0);                            \
        _mm_prefetch(_pf + 256,_MM_HINT_T0);                            \
        _mm_prefetch(_pf + 320,_MM_HINT_T0);                            \
    } while (0)
#define L6Z_PF_T0W_1(SRC,OUT,g)                                         \
    do {                                                                \
        L6Z_PF_T0_1(SRC,OUT,g);                                         \
        const char *_pw = (const char *)((OUT) + (long)L6_VD) + 384*(g);\
        __builtin_prefetch(_pw,       1, 3);                            \
        __builtin_prefetch(_pw +  64, 1, 3);                            \
        __builtin_prefetch(_pw + 128, 1, 3);                            \
        __builtin_prefetch(_pw + 192, 1, 3);                            \
        __builtin_prefetch(_pw + 256, 1, 3);                            \
        __builtin_prefetch(_pw + 320, 1, 3);                            \
    } while (0)

/* x-pass: lanes = 4 adjacent (y,z); axis stride 72 doubles; 9 groups,
 * every access 64B-aligned (volume 3456 B, group offset 8g doubles) */
#define L6Z_PASS_X(SRC,DST,OUT,PF)                                      \
    do {                                                                \
        for (int g = 0; g < 9; ++g) {                                   \
            const double *s = (SRC) + 8*g;                              \
            double *d = (DST) + 8*g;                                    \
            PF(SRC,OUT,g);                                              \
            __m512d v0=_mm512_load_pd(s+  0), v1=_mm512_load_pd(s+ 72); \
            __m512d v2=_mm512_load_pd(s+144), v3=_mm512_load_pd(s+216); \
            __m512d v4=_mm512_load_pd(s+288), v5=_mm512_load_pd(s+360); \
            VD6Z(v0,v1,v2,v3,v4,v5, v0,v1,v2,v3,v4,v5);                 \
            _mm512_store_pd(d+  0,v0); _mm512_store_pd(d+ 72,v1);       \
            _mm512_store_pd(d+144,v2); _mm512_store_pd(d+216,v3);       \
            _mm512_store_pd(d+288,v4); _mm512_store_pd(d+360,v5);       \
        }                                                               \
    } while (0)

/* y-pass: per x-plane, zmm codelet on z=0..3 (row stride 12 doubles ->
 * loadu/storeu) + ymm codelet on the z=4,5 tail (32B-aligned) */
#define L6Z_PASS_Y(SRC,DST)                                             \
    do {                                                                \
        for (int x = 0; x < 6; ++x) {                                   \
            const double *s = (SRC) + 72*x;                             \
            double *d = (DST) + 72*x;                                   \
            __m512d z0=_mm512_loadu_pd(s+ 0), z1=_mm512_loadu_pd(s+12); \
            __m512d z2=_mm512_loadu_pd(s+24), z3=_mm512_loadu_pd(s+36); \
            __m512d z4=_mm512_loadu_pd(s+48), z5=_mm512_loadu_pd(s+60); \
            VD6Z(z0,z1,z2,z3,z4,z5, z0,z1,z2,z3,z4,z5);                 \
            _mm512_storeu_pd(d+ 0,z0); _mm512_storeu_pd(d+12,z1);       \
            _mm512_storeu_pd(d+24,z2); _mm512_storeu_pd(d+36,z3);       \
            _mm512_storeu_pd(d+48,z4); _mm512_storeu_pd(d+60,z5);       \
            __m256d y0=_mm256_load_pd(s+ 8), y1=_mm256_load_pd(s+20);   \
            __m256d y2=_mm256_load_pd(s+32), y3=_mm256_load_pd(s+44);   \
            __m256d y4=_mm256_load_pd(s+56), y5=_mm256_load_pd(s+68);   \
            VD6(y0,y1,y2,y3,y4,y5, y0,y1,y2,y3,y4,y5);                  \
            _mm256_store_pd(d+ 8,y0); _mm256_store_pd(d+20,y1);         \
            _mm256_store_pd(d+32,y2); _mm256_store_pd(d+44,y3);         \
            _mm256_store_pd(d+56,y4); _mm256_store_pd(d+68,y5);         \
        }                                                               \
    } while (0)

/* z-pass 'zt': 4 whole z-pencils per iteration (48 contiguous doubles,
 * all accesses 64B-aligned).  Forward transpose to z-major: V_z holds
 * complex z of pencils p0..p3; each half of V_z draws from exactly two
 * of the six source vectors, so it is 2 vpermt2pd + 1 blend, and the
 * lo/hi permute patterns coincide, so only 4 forward index vectors are
 * needed.  Inverse needs 6 (lo/hi patterns differ where a pencil
 * boundary crosses a vector).  Index derivation in strategies/ r5. */
#define L6Z_ZT_CONSTS                                                   \
    const __m512i ztP0 = _mm512_setr_epi64(0,1,12,13, 0,1,12,13);       \
    const __m512i ztP1 = _mm512_setr_epi64(2,3,14,15, 2,3,14,15);       \
    const __m512i ztP2 = _mm512_setr_epi64(4,5, 8, 9, 4,5, 8, 9);       \
    const __m512i ztP3 = _mm512_setr_epi64(6,7,10,11, 6,7,10,11);       \
    const __m512i ztQ0 = _mm512_setr_epi64(0,1, 8, 9, 0,1, 8, 9);       \
    const __m512i ztQ1 = _mm512_setr_epi64(0,1, 8, 9, 2,3,10,11);       \
    const __m512i ztQ2 = _mm512_setr_epi64(2,3,10,11, 2,3,10,11);       \
    const __m512i ztQ3 = _mm512_setr_epi64(4,5,12,13, 4,5,12,13);       \
    const __m512i ztQ4 = _mm512_setr_epi64(4,5,12,13, 6,7,14,15);       \
    const __m512i ztQ5 = _mm512_setr_epi64(6,7,14,15, 6,7,14,15)

#define L6Z_MERGE(LO,HI) _mm512_mask_blend_pd((__mmask8)0xF0,(LO),(HI))

#define L6Z_PASS_ZT(SRC,DST)                                            \
    do {                                                                \
        L6Z_ZT_CONSTS;                                                  \
        for (int q = 0; q < 9; ++q) {                                   \
            const double *s = (SRC) + 48*q;                             \
            double *d = (DST) + 48*q;                                   \
            __m512d A0=_mm512_load_pd(s+ 0), A1=_mm512_load_pd(s+ 8);   \
            __m512d A2=_mm512_load_pd(s+16), A3=_mm512_load_pd(s+24);   \
            __m512d A4=_mm512_load_pd(s+32), A5=_mm512_load_pd(s+40);   \
            __m512d V0=L6Z_MERGE(_mm512_permutex2var_pd(A0,ztP0,A1),    \
                                 _mm512_permutex2var_pd(A3,ztP0,A4));   \
            __m512d V1=L6Z_MERGE(_mm512_permutex2var_pd(A0,ztP1,A1),    \
                                 _mm512_permutex2var_pd(A3,ztP1,A4));   \
            __m512d V2=L6Z_MERGE(_mm512_permutex2var_pd(A0,ztP2,A2),    \
                                 _mm512_permutex2var_pd(A3,ztP2,A5));   \
            __m512d V3=L6Z_MERGE(_mm512_permutex2var_pd(A0,ztP3,A2),    \
                                 _mm512_permutex2var_pd(A3,ztP3,A5));   \
            __m512d V4=L6Z_MERGE(_mm512_permutex2var_pd(A1,ztP0,A2),    \
                                 _mm512_permutex2var_pd(A4,ztP0,A5));   \
            __m512d V5=L6Z_MERGE(_mm512_permutex2var_pd(A1,ztP1,A2),    \
                                 _mm512_permutex2var_pd(A4,ztP1,A5));   \
            VD6Z(V0,V1,V2,V3,V4,V5, V0,V1,V2,V3,V4,V5);                 \
            _mm512_store_pd(d+ 0,L6Z_MERGE(                             \
                _mm512_permutex2var_pd(V0,ztQ0,V1),                     \
                _mm512_permutex2var_pd(V2,ztQ0,V3)));                   \
            _mm512_store_pd(d+ 8,L6Z_MERGE(                             \
                _mm512_permutex2var_pd(V4,ztQ1,V5),                     \
                _mm512_permutex2var_pd(V0,ztQ1,V1)));                   \
            _mm512_store_pd(d+16,L6Z_MERGE(                             \
                _mm512_permutex2var_pd(V2,ztQ2,V3),                     \
                _mm512_permutex2var_pd(V4,ztQ2,V5)));                   \
            _mm512_store_pd(d+24,L6Z_MERGE(                             \
                _mm512_permutex2var_pd(V0,ztQ3,V1),                     \
                _mm512_permutex2var_pd(V2,ztQ3,V3)));                   \
            _mm512_store_pd(d+32,L6Z_MERGE(                             \
                _mm512_permutex2var_pd(V4,ztQ4,V5),                     \
                _mm512_permutex2var_pd(V0,ztQ4,V1)));                   \
            _mm512_store_pd(d+40,L6Z_MERGE(                             \
                _mm512_permutex2var_pd(V2,ztQ5,V3),                     \
                _mm512_permutex2var_pd(V4,ztQ5,V5)));                   \
        }                                                               \
    } while (0)

/* zmm 3-pass kernel: zmm x, zmm+ymm y, z-pass supplied as a statement
 * (ymm perm / ymm split / zmm transpose) so all combos share one body */
#define L6Z_DEF(NAME,PF,ZSTMT)                                          \
static void NAME(double *t1, double *t2,                                \
                 const double *in, double *out, long nvol)              \
{                                                                       \
    VSET; VSETZ;                                                        \
    for (long b = 0; b < nvol; ++b) {                                   \
        const double *ip = in  + b * (long)L6_VD;                       \
        double       *op = out + b * (long)L6_VD;                       \
        L6Z_PASS_X(ip, t1, op, PF);                                     \
        L6Z_PASS_Y(t1, t2);                                             \
        ZSTMT;                                                          \
    }                                                                   \
}

L6Z_DEF(l6_run_z2p,     L6Z_PF_NONE,  L6_PASS_Z(t2, op, _mm256_store_pd))
L6Z_DEF(l6_run_z2p_pf,  L6Z_PF_T0_1,  L6_PASS_Z(t2, op, _mm256_store_pd))
L6Z_DEF(l6_run_z2s,     L6Z_PF_NONE,  L6_PASS_Z_S(t2, op))
L6Z_DEF(l6_run_z2s_pf,  L6Z_PF_T0_1,  L6_PASS_Z_S(t2, op))
L6Z_DEF(l6_run_z2s_pfw, L6Z_PF_T0W_1, L6_PASS_Z_S(t2, op))
L6Z_DEF(l6_run_z3t,     L6Z_PF_NONE,  L6Z_PASS_ZT(t2, op))
L6Z_DEF(l6_run_z3t_pf,  L6Z_PF_T0_1,  L6Z_PASS_ZT(t2, op))
L6Z_DEF(l6_run_z3t_pfw, L6Z_PF_T0W_1, L6Z_PASS_ZT(t2, op))

#endif /* L6_HAVE_AVX512 */

#endif /* L6_HAVE_AVX2 */

/* ------------------------------------------------------------------ *
 * Plan
 * ------------------------------------------------------------------ */

typedef void (*l6_kernel)(double *t1, double *t2,
                          const double *in, double *out, long nvol);

struct fft3d_plan {
    int       L, batch;
    double   *arena;          /* owns the scratch; t1/t2 are placed inside it */
    double   *t1, *t2;
    l6_kernel run;
    int       fence;          /* nonzero if the chosen kernel uses NT stores */
    int       placed;         /* scratch already positioned for these buffers */
    int       forced;         /* kernel forced via L6_FORCE, not raced */
    const char *chosen;
};

/* 4K-aliasing defence.  A store to S followed by a load from L with
 * (S-L) == 0 mod 4096 is falsely flagged as dependent and the load replays;
 * measured cost when a whole pass is in that state: +22% at B=1 (Haswell,
 * see strategies/L6_unrolled.md).  The scratch is therefore carved out of a
 * 4 KiB-oversized arena and positioned, once, so that every store->load
 * delta the kernels can produce sits as far as possible from 0 mod 4096.
 * This changes no arithmetic and no output, only addresses. */
static long l6_cyc(long d)
{
    d &= 4095;
    return d < 4096 - d ? d : 4096 - d;
}

static void l6_place(fft3d_plan *p, const void *in, const void *out)
{
    long D = (long)(((uintptr_t)out - (uintptr_t)in) & 4095u);
    long bestr = 0, bestscore = -1;
    for (long r = 0; r < 4096; r += 64) {
        long s1 = l6_cyc(r);                                  /* t1 - in      */
        long s2 = l6_cyc(D - r);                              /* out - t1     */
        long s3 = l6_cyc(D - r - L6_VD * (long)sizeof(double));/* out - t2    */
        long sc = s1 < s2 ? s1 : s2;
        if (s3 < sc) sc = s3;
        if (sc > bestscore) { bestscore = sc; bestr = r; }
    }
    long off = (long)((((uintptr_t)in + (uintptr_t)bestr
                        - (uintptr_t)p->arena) & 4095u) / sizeof(double));
    p->t1 = p->arena + off;
    p->t2 = p->t1 + L6_VD;
    p->placed = 1;
}

const char *fft3d_name(void) { return "L6_unrolled"; }

static char l6_desc[256] =
    "L=6: unrolled straight-line PFA 2x3 six-point codelet (48 flops/36 instr, "
    "no twiddles) on all three axes, AVX2/FMA 2-complex lanes, in-register "
    "z-pencil transposes; variant=auto";

const char *fft3d_description(void) { return l6_desc; }

int fft3d_supports(int L) { return L == L6_L; }

static void *l6_alloc(size_t bytes)
{
    void *p = NULL;
    if (posix_memalign(&p, 64, bytes) != 0) return NULL;
    return p;
}

#ifdef L6_HAVE_AVX2
static double l6_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* Effective AVX2 core clock, measured at plan time (unscored) and reported
 * in fft3d_description() as clk=X.XXGHz.  This answers the standing L=6
 * question (VERDICT panel_r3 §6: does the node turbo at B=1, i.e. is the
 * remaining headroom ~4% or ~40%?) through the same leaderboard channel as
 * variant=.  Method: a serially dependent 256-bit FMA chain; FMA latency is
 * 4 cycles on SKX/CLX/ICL/SPR, so freq = iters*4/time.  (Haswell's latency
 * is 5, so on the AVX2 dev machine this over-reads by 25% -- dev caveat
 * only; both wallaby and the scored node are 4-cycle parts.)  Runs after
 * the tournament, when the core is already warm, ~10 ms total. */
static double l6_probe_ghz(void)
{
    __m256d x = _mm256_set1_pd(1.0);
    const __m256d a = _mm256_set1_pd(1.0 + 1e-15);
    const __m256d b = _mm256_set1_pd(1e-300);
    double best = 0.0;
    double warm_until = l6_now() + 5e-3;
    do {
        for (int i = 0; i < 8192; ++i) x = _mm256_fmadd_pd(x, a, b);
    } while (l6_now() < warm_until);
    for (int trial = 0; trial < 5; ++trial) {
        double t0 = l6_now();
        for (int i = 0; i < 262144; ++i) x = _mm256_fmadd_pd(x, a, b);
        double dt = l6_now() - t0;
        double ghz = 262144.0 * 4.0 / dt * 1e-9;
        if (ghz > best) best = ghz;
    }
    double lane[4];
    _mm256_storeu_pd(lane, x);           /* keep the chain observable */
    if (!(lane[0] > 0.0) || best > 9.9) best = 0.0;
    return best;
}

#ifdef L6_HAVE_AVX512
/* Same probe with a 512-bit FMA chain: the sustained clock under the
 * AVX-512 heavy licence, which nobody has measured on the node (panel_r4
 * VERDICT §5b asks for exactly this).  zmm FMA latency is 4 on
 * SKX/CLX/ICL/SPR like ymm, so the two numbers are directly comparable;
 * the licence transition settles inside the 5 ms warm loop. */
static double l6_probe_ghz512(void)
{
    __m512d x = _mm512_set1_pd(1.0);
    const __m512d a = _mm512_set1_pd(1.0 + 1e-15);
    const __m512d b = _mm512_set1_pd(1e-300);
    double best = 0.0;
    double warm_until = l6_now() + 5e-3;
    do {
        for (int i = 0; i < 8192; ++i) x = _mm512_fmadd_pd(x, a, b);
    } while (l6_now() < warm_until);
    for (int trial = 0; trial < 5; ++trial) {
        double t0 = l6_now();
        for (int i = 0; i < 262144; ++i) x = _mm512_fmadd_pd(x, a, b);
        double dt = l6_now() - t0;
        double ghz = 262144.0 * 4.0 / dt * 1e-9;
        if (ghz > best) best = ghz;
    }
    double lane[8];
    _mm512_storeu_pd(lane, x);           /* keep the chain observable */
    if (!(lane[0] > 0.0) || best > 9.9) best = 0.0;
    return best;
}
#endif
#endif

fft3d_plan *fft3d_create(int L, int batch)
{
    if (L != L6_L || batch <= 0) return NULL;

    fft3d_plan *p = (fft3d_plan *)calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->L = L;
    p->batch = batch;
    /* 4096 bytes of slack for the placement search + the two scratch volumes */
    p->arena = (double *)l6_alloc(4096 + 2 * L6_VD * sizeof(double) + 64);
    if (!p->arena) { fft3d_destroy(p); return NULL; }
    p->t1 = p->arena;
    p->t2 = p->arena + L6_VD;
    p->run = l6_run_scalar;
    p->fence = 0;
    p->chosen = "scalar";

#ifdef L6_HAVE_AVX2
    {
        /* Ordered "safest first": a later candidate must beat the incumbent by
         * more than L6_MARGIN to take over, so measurement noise on a loaded
         * machine cannot promote a streaming-store kernel at a batch size
         * where the working set is cache resident.  The prefetch variants are
         * adopted from L6_pfa (panel_r1), whose fused+nt+prefetch kernel won
         * every large-batch case. */
        /* panel_r4 grid: +5 split-z shapes (_s), -3 fused_nt_pf* (dominated
         * by 3pass_nt_pf* in every r2/r3 measurement on both machines). */
        static const struct { l6_kernel k; int fence; const char *nm; } cand[] = {
            { l6_run_3pass,         0, "3pass"         },
            { l6_run_fused,         0, "fused"         },
            { l6_run_3pass_s,       0, "3pass_s"       },
            { l6_run_fused_s,       0, "fused_s"       },
            { l6_run_3pass_pf,      0, "3pass_pf"      },
            { l6_run_fused_pf,      0, "fused_pf"      },
            { l6_run_fused_s_pf,    0, "fused_s_pf"    },
            { l6_run_3pass_pf2,     0, "3pass_pf2"     },
            { l6_run_fused_pf2,     0, "fused_pf2"     },
            { l6_run_3pass_pft1,    0, "3pass_pft1"    },
            { l6_run_fused_pft1,    0, "fused_pft1"    },
            { l6_run_3pass_pfw,     0, "3pass_pfw"     },
            { l6_run_fused_pfw,     0, "fused_pfw"     },
            { l6_run_3pass_s_pfw,   0, "3pass_s_pfw"   },
            { l6_run_fused_s_pfw,   0, "fused_s_pfw"   },
            { l6_run_3pass_pfw2,    0, "3pass_pfw2"    },
            { l6_run_fused_pfw2,    0, "fused_pfw2"    },
            { l6_run_3pass_pft1w,   0, "3pass_pft1w"   },
            { l6_run_fused_pft1w,   0, "fused_pft1w"   },
            { l6_run_3pass_nt,      1, "3pass_nt"      },
            { l6_run_fused_nt,      1, "fused_nt"      },
            { l6_run_3pass_nt_pf,   1, "3pass_nt_pf"   },
            { l6_run_3pass_nt_pf2,  1, "3pass_nt_pf2"  },
            { l6_run_3pass_nt_pft1, 1, "3pass_nt_pft1" },
#ifdef L6_HAVE_AVX512
            /* panel_r5: mixed-width zmm kernels, last = must beat the ymm
             * incumbents by the margin (licence-downclock defence) */
            { l6_run_z2p,           0, "z2p"           },
            { l6_run_z2s,           0, "z2s"           },
            { l6_run_z3t,           0, "z3t"           },
            { l6_run_z2p_pf,        0, "z2p_pf"        },
            { l6_run_z2s_pf,        0, "z2s_pf"        },
            { l6_run_z3t_pf,        0, "z3t_pf"        },
            { l6_run_z2s_pfw,       0, "z2s_pfw"       },
            { l6_run_z3t_pfw,       0, "z3t_pfw"       },
#endif
        };
        const int ncand = (int)(sizeof(cand)/sizeof(cand[0]));

        /* ---- correctness gate: every candidate must reproduce the scalar
         * reference bit-closely on random data before it may be timed. ---- */
        long nval = batch < 4 ? batch : 4;
        double *vin  = (double *)l6_alloc((size_t)nval * L6_VD * sizeof(double));
        double *vref = (double *)l6_alloc((size_t)nval * L6_VD * sizeof(double));
        double *vgot = (double *)l6_alloc((size_t)nval * L6_VD * sizeof(double));
        int ok[48]; for (int i = 0; i < ncand; ++i) ok[i] = 0;

        if (vin && vref && vgot) {
            uint64_t st = 0x9E3779B97F4A7C15ull;
            for (long i = 0; i < nval * L6_VD; ++i) {
                st ^= st << 13; st ^= st >> 7; st ^= st << 17;
                vin[i] = (double)(int64_t)(st >> 11) * (1.0 / 9007199254740992.0);
            }
            l6_run_scalar(p->t1, p->t2, vin, vref, nval);
            double nrm = 0.0;
            for (long i = 0; i < nval * L6_VD; ++i) nrm += vref[i] * vref[i];
            nrm = sqrt(nrm) + 1e-300;
            for (int c = 0; c < ncand; ++c) {
                memset(vgot, 0, (size_t)nval * L6_VD * sizeof(double));
                cand[c].k(p->t1, p->t2, vin, vgot, nval);
                if (cand[c].fence) _mm_sfence();
                double e = 0.0;
                for (long i = 0; i < nval * L6_VD; ++i) {
                    double d = vgot[i] - vref[i];
                    e += d * d;
                }
                ok[c] = (sqrt(e) / nrm) < 1e-11;
            }
        }
        free(vin); free(vref); free(vgot);

        /* ---- forced variant (panel_r5): the L6_FORCE env var (or a
         * -DL6_FORCE_DEFAULT='"name"' compile flag) names a candidate that
         * is selected unconditionally, skipping the race -- it must still
         * pass the correctness gate above.  This is the node A/B switch
         * the panel_r4 VERDICT asked for ("force the _s shapes on the
         * node"); the pick is reported as variant=<name>! so a forced
         * leaderboard line cannot be mistaken for a tournament pick. */
        int forced = -1;
        {
            const char *fv = getenv("L6_FORCE");
#ifdef L6_FORCE_DEFAULT
            if (!fv || !fv[0]) fv = L6_FORCE_DEFAULT;
#endif
            if (fv && fv[0])
                for (int c = 0; c < ncand; ++c)
                    if (ok[c] && strcmp(cand[c].nm, fv) == 0) { forced = c; break; }
        }

        /* ---- race the survivors at (a truncation of) the real batch size ----
         * The cap must keep the raced working set out of L3 when the real one
         * is: 4096 volumes = 27 MiB barely exceeds the node's 22 MiB L3, and on
         * wallaby (60 MiB L3) it is cache-resident, which mis-picks a normal-
         * store kernel for a DRAM-bound batch.  16384 volumes = 113 MiB is
         * unambiguous on both machines. */
        long nt = batch;
        if (nt > 16384) nt = 16384;
        double *ain = NULL, *aout = NULL;
        int best = forced;
        if (best < 0) {
            ain  = (double *)l6_alloc((size_t)nt * L6_VD * sizeof(double));
            aout = (double *)l6_alloc((size_t)nt * L6_VD * sizeof(double));
        }
        if (best < 0 && ain && aout) {
            uint64_t st = 0xD1B54A32D192ED03ull;
            for (long i = 0; i < nt * L6_VD; ++i) {
                st ^= st << 13; st ^= st >> 7; st ^= st << 17;
                ain[i] = (double)(int64_t)(st >> 11) * (1.0 / 9007199254740992.0);
            }
            memset(aout, 0, (size_t)nt * L6_VD * sizeof(double));
            l6_place(p, ain, aout);
            /* how many repeats to make one trial last ~2 ms */
            long reps = (long)(2e-3 / (nt * 2.5e-7));
            if (reps < 1) reps = 1;
            if (reps > 20000) reps = 20000;
            /* Round-robin tournament (adopted from L6_pfa's record, which
             * documents a sequential per-candidate race mis-picking by 21%
             * when background load drifts between candidates): every round
             * times each surviving candidate once, and each candidate keeps
             * its own minimum, so drift hits all candidates alike. */
            double tmin[48];
            for (int c = 0; c < ncand; ++c) {
                tmin[c] = 1e300;
                if (!ok[c]) continue;
                cand[c].k(p->t1, p->t2, ain, aout, nt);      /* warm */
                if (cand[c].fence) _mm_sfence();
            }
            for (int round = 0; round < 7; ++round) {
                for (int c = 0; c < ncand; ++c) {
                    if (!ok[c]) continue;
                    double t0 = l6_now();
                    for (long r = 0; r < reps; ++r)
                        cand[c].k(p->t1, p->t2, ain, aout, nt);
                    if (cand[c].fence) _mm_sfence();
                    double dt = (l6_now() - t0) / (double)reps;
                    if (dt < tmin[c]) tmin[c] = dt;
                }
            }
            double bestt = 1e300;
            for (int c = 0; c < ncand; ++c) {
                if (!ok[c]) continue;
                if (best < 0 || tmin[c] < bestt * (1.0 - L6_MARGIN)) {
                    bestt = tmin[c]; best = c;
                } else if (tmin[c] < bestt) bestt = tmin[c];
            }
            if (getenv("L6_VERBOSE"))
                for (int c = 0; c < ncand; ++c)
                    fprintf(stderr, "L6_unrolled race: %-14s %s %10.4f us/vol%s\n",
                            cand[c].nm, ok[c] ? "ok " : "BAD",
                            ok[c] ? tmin[c] / (double)nt * 1e6 : 0.0,
                            c == best ? "   <-- chosen" : "");
        }
        if (best < 0)
            for (int c = 0; c < ncand; ++c) if (ok[c]) { best = c; break; }
        free(ain); free(aout);
        p->t1 = p->arena; p->t2 = p->arena + L6_VD; p->placed = 0;

        if (best >= 0) {
            p->run    = cand[best].k;
            p->fence  = cand[best].fence;
            p->chosen = cand[best].nm;
            p->forced = (best == forced);
        }
    }
#endif
    {   /* report the raced winner and the measured core clocks on the
         * leaderboard line.  clk=<256-bit>/<512-bit> sustained GHz: the
         * second number is the AVX-512 licence clock the r4 VERDICT asked
         * for; a trailing ! marks an L6_FORCE pick (not a tournament one) */
        double ghz = 0.0, ghz5 = 0.0;
#ifdef L6_HAVE_AVX2
        ghz = l6_probe_ghz();
#ifdef L6_HAVE_AVX512
        ghz5 = l6_probe_ghz512();
#endif
#endif
        if (ghz > 0.0)
            snprintf(l6_desc, sizeof(l6_desc),
                     "L=6: unrolled PFA 2x3 codelet (48 flop/36 instr, no "
                     "twiddles), ymm+zmm raced; variant=%s%s clk=%.2f/%.2fGHz",
                     p->chosen, p->forced ? "!" : "", ghz, ghz5);
        else
            snprintf(l6_desc, sizeof(l6_desc),
                     "L=6: unrolled PFA 2x3 codelet, portable scalar path; "
                     "variant=%s", p->chosen);
    }
    return p;
}

void fft3d_execute(fft3d_plan *plan, const double _Complex *in, double _Complex *out)
{
    if (!plan->placed) l6_place(plan, in, out);
    plan->run(plan->t1, plan->t2,
              (const double *)in, (double *)out, (long)plan->batch);
#ifdef L6_HAVE_AVX2
    if (plan->fence) _mm_sfence();
#endif
}

void fft3d_destroy(fft3d_plan *plan)
{
    if (!plan) return;
    free(plan->arena);
    free(plan);
}
