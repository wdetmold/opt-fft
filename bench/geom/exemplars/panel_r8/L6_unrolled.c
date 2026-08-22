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
 *  * 10 kernel variants are RACED AGAINST EACH OTHER AND VALIDATED against a
 *    scalar reference inside fft3d_create(); a variant that disagrees with the
 *    reference by more than 1e-11 relative is disqualified and can never be
 *    selected.  This is what makes an untestable-locally code path safe.
 *    The grid was pruned hard in panel_r6 on four rounds of node pick data:
 *    kept {3pass,fused} x {plain, pfT0, pfT0+W} (the only shapes the node has
 *    ever selected), the two split-z "_s" shapes (wallaby's B=1 winners, kept
 *    for dev A/Bs), one NT representative (3pass_nt_pf, wallaby DRAM winner;
 *    NT is 0-for-4 rounds on the node), and one mixed-width AVX-512 kernel
 *    (z2s, kept ONLY as the L6_FORCE perf-counter A/B target the r5 VERDICT
 *    asks for -- clk512=2.89 vs clk256>=2.89 killed zmm at L=6, 0 picks in
 *    12 invocations).
 *    L6_FORCE=<name> (env or -DL6_FORCE_DEFAULT) bypasses the race for node
 *    A/B tests; a forced pick is reported as variant=<name>! in the
 *    description.  The description also carries a five-point clock ladder,
 *    clk256=<sparse,mid,sat> clk512=<sparse,sat> sustained GHz (panel_r6:
 *    the saturating probes adopt L17_winograd's design; 256-bit is issued
 *    before 512-bit so licence dwell cannot leak backwards).  sat256 is the
 *    number the r5 VERDICT asks for: it decides whether the node's dense ymm
 *    kernels run at 3.89 or 2.89 GHz, i.e. whether B=1 has ~366 or ~147
 *    unexplained cycles.
 *  * panel_r6 fix: create() used to END with a 512-bit probe, leaving the
 *    core in the AVX-512 licence; at B=1 the driver's whole sample set is
 *    ~0.5 ms and can complete inside the licence-recovery window, which is
 *    the leading suspect for r5's B=1 regression (0.219 -> 0.227 typical,
 *    identical pick).  create() now ends with a ~20 ms 256-bit spin so the
 *    driver never times inside the 512 licence.
 *  * panel_r7: the brief's corrected turbo table (Intel 338848-028US: AVX2
 *    and AVX-512 licences are BOTH 2.9 GHz at 1-8 active cores on the Gold
 *    5218) reopens the zmm question r5 closed.  Two new mixed-width shapes
 *    join the race, ordered after every ymm incumbent so they must win by
 *    the 2.5% margin: zxf (zmm x-pass + the node-proven ymm fused y+z) and
 *    zff (zmm x-pass + fully fused zmm/ymm y+z per plane: 9 aligned zmm
 *    plane loads + 3 valignq -- no split loads, unlike r5's z2s -- and no
 *    t2 round trip; ~1296 uops/volume vs fused's 1728).  The tournament
 *    also gains a per-candidate licence warm-up (LITERATURE 08 s4.3): each
 *    candidate runs itself untimed ~0.7 ms before every timed trial, so a
 *    ymm candidate is never timed inside a zmm predecessor's licence tail.
 *    And the description now carries kclk=<GHz> -- L6_pfa's r6 probe design
 *    (adopted, attributed): dwell ~2 ms in the CHOSEN kernel, then time a
 *    ~150 us sparse ymm chain that reads the licence the kernel itself
 *    established; median of 9.  That is the number that converts this
 *    entry's node times into cycles.
 *  * panel_r8: the r7 node data closed the width question (zxf/zff/z2s and
 *    L6_pfa's fused_zx: 0 picks in 8 cells x 3 processes, with the race
 *    licence-fair and kclk measured at 2.89 both widths), so the zmm shapes
 *    no longer RACE: zff and z2s stay compiled, validated and L6_FORCE-able
 *    only, because the r7 VERDICT's outstanding perf-counter A/B names them.
 *    zxf/zxf_pf/zff_pf are deleted.  Two changes adopted from L6_pfa
 *    (attributed): (a) `restrict` on every kernel signature -- the one
 *    systematic codegen difference between our same-shape kernels; without
 *    it gcc must order every t1-plane load after the preceding out-stores
 *    and every next-group input load after the scratch stores, i.e. the
 *    exact pass-boundary joints the r7 VERDICT names as the remaining B=1
 *    suspect; their fused_pf_xa (identical shape, restrict-qualified) beat
 *    my fused_pf by ~3.7% at B=4096 in 2 of 3 r7 processes; (b) create()
 *    now ENDS by dwelling ~3 ms in the CHOSEN kernel, so the driver is
 *    handed a core in the scored kernel's own licence/clock steady state,
 *    never a probe's or a generic spin's.
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

static void l6_run_scalar(double *restrict t1, double *restrict t2,
                          const double *restrict in, double *restrict out,
                          long nvol)
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
#define L6_PF_T0W_1(SRC,OUT,g)                                          \
    do { L6_PF_AT(SRC,g,1,_MM_HINT_T0); L6_PF_W_AT(OUT,g,1); } while (0)
/* pruned in panel_r6 (never picked on the node in 4 rounds of stable pick
 * reporting): distance-2 hooks, T1 hooks, W-only.  See strategies/ r6. */

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

/* All kernels are entered through a function pointer, so their placement in
 * the binary is at the linker's mercy; panel_r6 pins every kernel entry to a
 * 64-byte boundary after r5's B=1 regressed 0.219->0.227 typical with an
 * IDENTICAL pick string when ~500 lines of zmm code were added to the file
 * (same disease the r5 VERDICT names at L36_mixedradix B=1: code layout). */
#define L6_KALIGN __attribute__((aligned(64)))

/* --- variant A: three separate passes ---------------------------------- */
#define L6_DEF_3PASS(NAME,ST,PF)                                        \
L6_KALIGN static void NAME(double *restrict t1, double *restrict t2,    \
                 const double *restrict in, double *restrict out,       \
                 long nvol)                                             \
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
L6_DEF_3PASS(l6_run_3pass_pfw,    _mm256_store_pd,  L6_PF_T0W_1)
L6_DEF_3PASS(l6_run_3pass_nt_pf,  _mm256_stream_pd, L6_PF_T0_1)

/* --- variant B: x-pass, then y and z fused per x-plane in registers ----
 * The 6x6 (y,z) plane is 18 __m256d; with the codelet temporaries that is
 * ~26 live vectors, which fits the 32 ymm of an AVX-512VL machine but not
 * the 16 of plain AVX2.  Raced at plan time against variant A.
 * The plane loop is a macro of its own (panel_r7) so the zmm-x `zxf`
 * kernels can reuse it token-for-token behind a different x-pass.       */
#define L6_FUSED_YZ(SRC,DST,ST)                                         \
    do {                                                                \
        for (int x = 0; x < 6; ++x) {                                   \
            const double *s = (SRC) + 72*x;                             \
            double *d = (DST) + 72*x;                                   \
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
    } while (0)

#define L6_DEF_FUSED(NAME,ST,PF)                                        \
L6_KALIGN static void NAME(double *restrict t1, double *restrict t2,    \
                 const double *restrict in, double *restrict out,       \
                 long nvol)                                             \
{                                                                       \
    VSET;                                                               \
    (void)t2;                                                           \
    for (long b = 0; b < nvol; ++b) {                                   \
        const double *ip = in  + b * (long)L6_VD;                       \
        double       *op = out + b * (long)L6_VD;                       \
        L6_PASS_X(ip, t1, op, PF);                                      \
        L6_FUSED_YZ(t1, op, ST);                                        \
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

L6_DEF_FUSED(l6_run_fused,        _mm256_store_pd,  L6_PF_NONE)
L6_DEF_FUSED(l6_run_fused_pf,     _mm256_store_pd,  L6_PF_T0_1)
L6_DEF_FUSED(l6_run_fused_pfw,    _mm256_store_pd,  L6_PF_T0W_1)

/* panel_r7 prune: the ymm split-store shapes 3pass_s / fused_s (and the
 * L6_ZPAIR_S macro) are deleted -- 0 picks in 12 node invocations across
 * r4/r5, SPR-only mechanism.  L6_PASS_Z_S survives because z2s uses it. */

/* ------------------------------------------------------------------ *
 * AVX-512 mixed-width kernels -- RETIRED FROM THE RACE in panel_r8.
 *
 * History: eight r5 shapes rejected 0-for-12; r7 re-armed the question
 * with zxf (-17% uops) and zff (-25%, aligned loads, no t2 round trip)
 * after the brief's corrected turbo table, with a licence-fair race
 * (per-candidate dwell) and the kernel-context clock measured
 * (kclk = 2.89 GHz both widths).  The r7 node rejected every zmm shape
 * again: 0 picks in 8 cells x 3 processes, this time with no confound
 * left.  The r7 VERDICT's synthesis: width buys nothing where the
 * kernel is not front-end-bound, and at L=6 it is not.  Closed.
 *
 * What survives, compiled + validated + L6_FORCE-able but NEVER raced:
 *   zff -- the r7 VERDICT §6 perf-counter A/B target (fused vs zff vs
 *          L6_pfa's fused_zx at B=1), the ask that decides whether the
 *          remaining ~147 B=1 cycles are the t1 store->load joint.
 *   z2s -- the r5 baseline, kept for probe/perf continuity.
 * zxf/zxf_pf/zff_pf and the zmm prefetch hook are deleted.
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

#define L6Z_PF_NONE(SRC,OUT,g)  do { } while (0)

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

/* zmm 3-pass kernel: zmm x, zmm+ymm y, z-pass supplied as a statement
 * (ymm perm / ymm split / zmm transpose) so all combos share one body */
#define L6Z_DEF(NAME,PF,ZSTMT)                                          \
L6_KALIGN static void NAME(double *restrict t1, double *restrict t2,    \
                 const double *restrict in, double *restrict out,       \
                 long nvol)                                             \
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

L6Z_DEF(l6_run_z2s,     L6Z_PF_NONE,  L6_PASS_Z_S(t2, op))

/* --- zff (panel_r7): zmm x-pass + fully fused zmm/ymm y+z per plane.
 *
 * Every 6x6 (y,z) plane of t1 is 72 doubles = 576 B = 9 whole cache
 * lines, and plane bases are 64B-aligned by construction, so the plane
 * is brought in as 9 ALIGNED zmm loads v0..v8.  Row y's (z0..z3) half
 * lives at qwords 12y..12y+7:
 *   rows 0/2/4 are v0/v3/v6 directly;
 *   rows 1/3/5 are valignq(v2,v1,4) / (v5,v4,4) / (v8,v7,4)  (1 p5 uop
 *   each -- this is what replaces z2s's line-splitting stride-12 loadu).
 * Row y's (z4,z5) half is a ymm: rows 0/2/4 are the low ymm of v1/v4/v7
 * (free casts); rows 1/3/5 are vextractf64x4(v2/v5/v8, 1).
 * One VD6Z across the six zmm rows + one VD6 across the six ymm tails =
 * the whole y-DFT of the plane in 36 arithmetic instructions.
 * z-DFT per row pair (y,y+1): 4 vpermt2pd gather lane z of both rows
 * into z-major ymm v_z (z=0..3), 2 vperm2f128 do z=4,5 from the tails,
 * then the standard ymm codelet and the L6_ZPAIR store tail (full-width
 * 32B stores; the node has never picked a split-store shape). */
#define L6_ALIGN4(HI,LO)                                                \
    _mm512_castsi512_pd(_mm512_alignr_epi64(                            \
        _mm512_castpd_si512(HI), _mm512_castpd_si512(LO), 4))

#define VSETZI                                                          \
    const __m512i zidx0 = _mm512_setr_epi64(0,1, 8, 9,0,1, 8, 9);       \
    const __m512i zidx1 = _mm512_setr_epi64(2,3,10,11,2,3,10,11);       \
    const __m512i zidx2 = _mm512_setr_epi64(4,5,12,13,4,5,12,13);       \
    const __m512i zidx3 = _mm512_setr_epi64(6,7,14,15,6,7,14,15)

#define L6_ZFF_PAIR(RA,RB,HA,HB,D)                                      \
    do {                                                                \
        __m256d w0=_mm512_castpd512_pd256(                              \
                       _mm512_permutex2var_pd(RA,zidx0,RB));            \
        __m256d w1=_mm512_castpd512_pd256(                              \
                       _mm512_permutex2var_pd(RA,zidx1,RB));            \
        __m256d w2=_mm512_castpd512_pd256(                              \
                       _mm512_permutex2var_pd(RA,zidx2,RB));            \
        __m256d w3=_mm512_castpd512_pd256(                              \
                       _mm512_permutex2var_pd(RA,zidx3,RB));            \
        __m256d w4=_mm256_permute2f128_pd(HA,HB,0x20);                  \
        __m256d w5=_mm256_permute2f128_pd(HA,HB,0x31);                  \
        VD6(w0,w1,w2,w3,w4,w5, w0,w1,w2,w3,w4,w5);                      \
        _mm256_store_pd((D)+ 0,_mm256_permute2f128_pd(w0,w1,0x20));     \
        _mm256_store_pd((D)+ 4,_mm256_permute2f128_pd(w2,w3,0x20));     \
        _mm256_store_pd((D)+ 8,_mm256_permute2f128_pd(w4,w5,0x20));     \
        _mm256_store_pd((D)+12,_mm256_permute2f128_pd(w0,w1,0x31));     \
        _mm256_store_pd((D)+16,_mm256_permute2f128_pd(w2,w3,0x31));     \
        _mm256_store_pd((D)+20,_mm256_permute2f128_pd(w4,w5,0x31));     \
    } while (0)

#define L6_ZFF_PLANE(S,D)                                               \
    do {                                                                \
        __m512d v0=_mm512_load_pd((S)+ 0), v1=_mm512_load_pd((S)+ 8);   \
        __m512d v2=_mm512_load_pd((S)+16), v3=_mm512_load_pd((S)+24);   \
        __m512d v4=_mm512_load_pd((S)+32), v5=_mm512_load_pd((S)+40);   \
        __m512d v6=_mm512_load_pd((S)+48), v7=_mm512_load_pd((S)+56);   \
        __m512d v8=_mm512_load_pd((S)+64);                              \
        __m512d R1=L6_ALIGN4(v2,v1);                                    \
        __m512d R3=L6_ALIGN4(v5,v4);                                    \
        __m512d R5=L6_ALIGN4(v8,v7);                                    \
        __m256d h0=_mm512_castpd512_pd256(v1);                          \
        __m256d h1=_mm512_extractf64x4_pd(v2,1);                        \
        __m256d h2=_mm512_castpd512_pd256(v4);                          \
        __m256d h3=_mm512_extractf64x4_pd(v5,1);                        \
        __m256d h4=_mm512_castpd512_pd256(v7);                          \
        __m256d h5=_mm512_extractf64x4_pd(v8,1);                        \
        __m512d R0=v0, R2=v3, R4=v6;                                    \
        VD6Z(R0,R1,R2,R3,R4,R5, R0,R1,R2,R3,R4,R5);                     \
        VD6 (h0,h1,h2,h3,h4,h5, h0,h1,h2,h3,h4,h5);                     \
        L6_ZFF_PAIR(R0,R1,h0,h1,(D)+ 0);                                \
        L6_ZFF_PAIR(R2,R3,h2,h3,(D)+24);                                \
        L6_ZFF_PAIR(R4,R5,h4,h5,(D)+48);                                \
    } while (0)

#define L6_DEF_ZFF(NAME,PF)                                             \
L6_KALIGN static void NAME(double *restrict t1, double *restrict t2,    \
                 const double *restrict in, double *restrict out,       \
                 long nvol)                                             \
{                                                                       \
    VSET; VSETZ; VSETZI;                                                \
    (void)t2;                                                           \
    for (long b = 0; b < nvol; ++b) {                                   \
        const double *ip = in  + b * (long)L6_VD;                       \
        double       *op = out + b * (long)L6_VD;                       \
        L6Z_PASS_X(ip, t1, op, PF);                                     \
        for (int x = 0; x < 6; ++x)                                     \
            L6_ZFF_PLANE(t1 + 72*x, op + 72*x);                         \
    }                                                                   \
}

L6_DEF_ZFF(l6_run_zff,     L6Z_PF_NONE)

#endif /* L6_HAVE_AVX512 */

#endif /* L6_HAVE_AVX2 */

/* ------------------------------------------------------------------ *
 * Plan
 * ------------------------------------------------------------------ */

typedef void (*l6_kernel)(double *restrict t1, double *restrict t2,
                          const double *restrict in, double *restrict out,
                          long nvol);

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

/* ------------------------------------------------------------------ *
 * Clock ladder (panel_r6).  The r5 VERDICT's #1 L=6 ask: five probes on
 * the node disagreed on the 256-bit clock (3.89 / 3.27 / 2.89 GHz) and
 * the plausible cause is chain DENSITY -- a sparse chain never engages
 * the AVX2 licence and reads the non-AVX clock.  So this round ships the
 * whole ladder in one process, in one description string:
 *
 *   clk256 = sparse (1 chain, 0.25 FMA/cy), mid (5 chains, 1.25 FMA/cy,
 *            ~ the real kernel's FP density), sat (8 chains, 2.0 FMA/cy,
 *            both FMA ports saturated -- L17_winograd's design, adopted)
 *   clk512 = sparse (1 chain), sat (4 chains = 1.0 FMA/cy, saturating on
 *            the node's single 512-bit unit and latency-bound at the
 *            same 1/cy on wallaby's two, so comparable across machines)
 *
 * 256-bit probes are issued BEFORE the 512-bit ones (L17_winograd's
 * ordering) so licence dwell cannot leak backwards.  FMA latency is 4 on
 * SKX/CLX/ICL/SPR at both widths; every chain count C here satisfies
 * C <= 4*throughput, so the steady state is latency-bound at exactly 4
 * cycles per outer iteration and freq = iters*4/dt for every probe.
 * (Haswell latency is 5: on wombat all numbers over-read 25%; both
 * wallaby and the node are 4-cycle parts.)  ~0.3 ms per trial, 5 trials
 * per probe, warm loop first; all unscored.
 * ------------------------------------------------------------------ */
/* Trial harness shared by all five probes.  The chain variables MUST be
 * plain locals, never an array: a first cut used `TY x[C]` and gcc kept
 * the chains in memory, adding a store-forward round trip to every FMA --
 * the probe then under-read the clock by an exact factor ~2 (measured
 * 2.10 GHz in the same process whose kernel timing implied >=3.9).  */
#define L6_PROBE_TRIALS(BEST, ...)                                      \
    do {                                                                \
        double warm_until = l6_now() + 2e-2;                            \
        do {                                                            \
            for (int i = 0; i < 4096; ++i) { __VA_ARGS__; }             \
        } while (l6_now() < warm_until);                                \
        for (int trial = 0; trial < 5; ++trial) {                      \
            double t0 = l6_now();                                       \
            for (int i = 0; i < 262144; ++i) { __VA_ARGS__; }           \
            double dt = l6_now() - t0;                                  \
            double ghz = 262144.0 * 4.0 / dt * 1e-9;                    \
            if (ghz > (BEST)) (BEST) = ghz;                             \
        }                                                               \
    } while (0)

static double l6_probe256_sparse(void)
{
    const __m256d a = _mm256_set1_pd(1.0 + 1e-15);
    const __m256d b = _mm256_set1_pd(1e-300);
    __m256d x0 = _mm256_set1_pd(1.0);
    double best = 0.0;
    L6_PROBE_TRIALS(best, x0 = _mm256_fmadd_pd(x0, a, b));
    double lane[4];
    _mm256_storeu_pd(lane, x0);          /* keep the chain observable */
    if (!(lane[0] > 0.0) || best > 9.9) best = 0.0;
    return best;
}

static double l6_probe256_mid(void)
{
    const __m256d a = _mm256_set1_pd(1.0 + 1e-15);
    const __m256d b = _mm256_set1_pd(1e-300);
    __m256d x0 = _mm256_set1_pd(1.0), x1 = x0, x2 = x0, x3 = x0, x4 = x0;
    double best = 0.0;
    L6_PROBE_TRIALS(best,
        x0 = _mm256_fmadd_pd(x0, a, b); x1 = _mm256_fmadd_pd(x1, a, b);
        x2 = _mm256_fmadd_pd(x2, a, b); x3 = _mm256_fmadd_pd(x3, a, b);
        x4 = _mm256_fmadd_pd(x4, a, b));
    __m256d s = _mm256_add_pd(_mm256_add_pd(x0, x1),
                              _mm256_add_pd(_mm256_add_pd(x2, x3), x4));
    double lane[4];
    _mm256_storeu_pd(lane, s);
    if (!(lane[0] > 0.0) || best > 9.9) best = 0.0;
    return best;
}

static double l6_probe256_sat(void)
{
    const __m256d a = _mm256_set1_pd(1.0 + 1e-15);
    const __m256d b = _mm256_set1_pd(1e-300);
    __m256d x0 = _mm256_set1_pd(1.0), x1 = x0, x2 = x0, x3 = x0,
            x4 = x0, x5 = x0, x6 = x0, x7 = x0;
    double best = 0.0;
    L6_PROBE_TRIALS(best,
        x0 = _mm256_fmadd_pd(x0, a, b); x1 = _mm256_fmadd_pd(x1, a, b);
        x2 = _mm256_fmadd_pd(x2, a, b); x3 = _mm256_fmadd_pd(x3, a, b);
        x4 = _mm256_fmadd_pd(x4, a, b); x5 = _mm256_fmadd_pd(x5, a, b);
        x6 = _mm256_fmadd_pd(x6, a, b); x7 = _mm256_fmadd_pd(x7, a, b));
    __m256d s = _mm256_add_pd(
        _mm256_add_pd(_mm256_add_pd(x0, x1), _mm256_add_pd(x2, x3)),
        _mm256_add_pd(_mm256_add_pd(x4, x5), _mm256_add_pd(x6, x7)));
    double lane[4];
    _mm256_storeu_pd(lane, s);
    if (!(lane[0] > 0.0) || best > 9.9) best = 0.0;
    return best;
}

#ifdef L6_HAVE_AVX512
static double l6_probe512_sparse(void)
{
    const __m512d a = _mm512_set1_pd(1.0 + 1e-15);
    const __m512d b = _mm512_set1_pd(1e-300);
    __m512d x0 = _mm512_set1_pd(1.0);
    double best = 0.0;
    L6_PROBE_TRIALS(best, x0 = _mm512_fmadd_pd(x0, a, b));
    double lane[8];
    _mm512_storeu_pd(lane, x0);
    if (!(lane[0] > 0.0) || best > 9.9) best = 0.0;
    return best;
}

static double l6_probe512_sat(void)
{
    const __m512d a = _mm512_set1_pd(1.0 + 1e-15);
    const __m512d b = _mm512_set1_pd(1e-300);
    __m512d x0 = _mm512_set1_pd(1.0), x1 = x0, x2 = x0, x3 = x0;
    double best = 0.0;
    L6_PROBE_TRIALS(best,
        x0 = _mm512_fmadd_pd(x0, a, b); x1 = _mm512_fmadd_pd(x1, a, b);
        x2 = _mm512_fmadd_pd(x2, a, b); x3 = _mm512_fmadd_pd(x3, a, b));
    __m512d s = _mm512_add_pd(_mm512_add_pd(x0, x1), _mm512_add_pd(x2, x3));
    double lane[8];
    _mm512_storeu_pd(lane, s);
    if (!(lane[0] > 0.0) || best > 9.9) best = 0.0;
    return best;
}
#endif

/* Busy 256-bit spin for SECS wall seconds.  Used (a) before the race, so
 * round 0 is not ranked on a ramping clock (L17_rader's r5 settle-spin
 * finding: 76% mis-ranking from table order alone), and (b) at the very
 * END of create(), AFTER the 512-bit probes: r5's create() returned with
 * the core still in the AVX-512 licence, and at B=1 the driver's whole
 * sample set (~0.5 ms) can complete inside the licence-recovery window --
 * the leading suspect for r5's B=1 0.219 -> 0.227-typical regression.
 * Active 256-bit work lets the 512 licence expire without the core ever
 * idling down. */
static void l6_spin256(double secs)
{
    __m256d x = _mm256_set1_pd(1.0);
    const __m256d a = _mm256_set1_pd(1.0 + 1e-15);
    const __m256d b = _mm256_set1_pd(1e-300);
    double until = l6_now() + secs;
    do {
        for (int i = 0; i < 8192; ++i) x = _mm256_fmadd_pd(x, a, b);
    } while (l6_now() < until);
    double lane[4];
    _mm256_storeu_pd(lane, x);
    if (!(lane[0] > 0.0)) fprintf(stderr, "l6_spin256: impossible\n");
}

/* ------------------------------------------------------------------ *
 * kclk (panel_r7) -- ADOPTED FROM L6_pfa's panel_r6 record: the clock
 * the CHOSEN kernel actually runs at, measured directly instead of
 * inferred from a synthetic chain's density.  Dwell ~2 ms in the real
 * kernel, then immediately time a ~150 us sparse ymm FMA chain: CLX
 * licence state persists ~670 us after the last heavy instruction, so
 * the sparse chain (which by itself never raises the licence) reads
 * the licence the kernel established.  Median of 9 dwell/read pairs.
 * freq = iters*4/dt (latency-4 FMA; the Haswell latency-5 over-read
 * caveat from the r4 probe applies to wombat only).  ~20 ms, unscored.
 * ------------------------------------------------------------------ */
static double l6_kclk(l6_kernel run, int fence, double *t1, double *t2)
{
    const long nd = 4;
    double *din  = (double *)l6_alloc((size_t)nd * L6_VD * sizeof(double));
    double *dout = (double *)l6_alloc((size_t)nd * L6_VD * sizeof(double));
    if (!din || !dout) { free(din); free(dout); return 0.0; }
    uint64_t st = 0xA0761D6478BD642Full;
    for (long i = 0; i < nd * L6_VD; ++i) {
        st ^= st << 13; st ^= st >> 7; st ^= st << 17;
        din[i] = (double)(int64_t)(st >> 11) * (1.0 / 9007199254740992.0);
    }
    const __m256d a = _mm256_set1_pd(1.0 + 1e-15);
    const __m256d b = _mm256_set1_pd(1e-300);
    double vals[9];
    for (int r = 0; r < 9; ++r) {
        double until = l6_now() + 2e-3;
        do {
            run(t1, t2, din, dout, nd);
        } while (l6_now() < until);
        if (fence) _mm_sfence();
        __m256d x = _mm256_set1_pd(1.0);       /* named chain, never array */
        double t0 = l6_now();
        for (int i = 0; i < 131072; ++i) x = _mm256_fmadd_pd(x, a, b);
        double dt = l6_now() - t0;
        double lane[4];
        _mm256_storeu_pd(lane, x);
        vals[r] = (lane[0] > 0.0) ? 131072.0 * 4.0 / dt * 1e-9 : 0.0;
    }
    free(din); free(dout);
    for (int i = 1; i < 9; ++i) {              /* insertion sort, median */
        double v = vals[i]; int j = i;
        while (j > 0 && vals[j-1] > v) { vals[j] = vals[j-1]; --j; }
        vals[j] = v;
    }
    return vals[4] > 9.9 ? 0.0 : vals[4];
}

/* End-of-create dwell in the CHOSEN kernel (panel_r8, ADOPTED FROM
 * L6_pfa's panel_r7 refinement of my r6 licence-tail fix): ~3 ms of the
 * scored kernel itself, so the driver is handed a core in that kernel's
 * own licence/clock steady state -- never a probe's licence and never a
 * sparse spin's light-licence clock.  Falls back silently (the r6
 * spin256 tail already ran) if the scratch cannot be allocated. */
static void l6_dwell_chosen(l6_kernel run, int fence, double *t1, double *t2)
{
    const long nd = 4;
    double *din  = (double *)l6_alloc((size_t)nd * L6_VD * sizeof(double));
    double *dout = (double *)l6_alloc((size_t)nd * L6_VD * sizeof(double));
    if (!din || !dout) { free(din); free(dout); return; }
    uint64_t st = 0xE7037ED1A0B428DBull;
    for (long i = 0; i < nd * L6_VD; ++i) {
        st ^= st << 13; st ^= st >> 7; st ^= st << 17;
        din[i] = (double)(int64_t)(st >> 11) * (1.0 / 9007199254740992.0);
    }
    double until = l6_now() + 3e-3;
    do {
        run(t1, t2, din, dout, nd);
    } while (l6_now() < until);
    if (fence) _mm_sfence();
    free(din); free(dout);
}
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
         * where the working set is cache resident.
         *
         * panel_r8 grid: 7 raced ymm + 2 UNRACED zmm.  The ymm side is
         * the panel_r6 pruning (the node has only ever selected
         * {3pass,fused} x {plain,pf,pfw}; NT 0-for-5 rounds but one
         * representative kept).  The zmm side is retired from the race
         * on the r7 node data (0 picks in 8 cells x 3 processes with a
         * licence-fair race and kclk measured at 2.89 both widths --
         * the width question is CLOSED at L=6); zff and z2s stay
         * compiled + validated + L6_FORCE-able because the r7 VERDICT
         * §6 perf-counter ask names them.  raced=0 candidates are never
         * timed and can never be picked.  Layout defence: every kernel
         * entry is 64B-pinned (L6_KALIGN, since r6). */
        static const struct {
            l6_kernel k; int fence; int raced; const char *nm;
        } cand[] = {
            { l6_run_3pass,         0, 1, "3pass"         },
            { l6_run_fused,         0, 1, "fused"         },
            { l6_run_3pass_pf,      0, 1, "3pass_pf"      },
            { l6_run_fused_pf,      0, 1, "fused_pf"      },
            { l6_run_3pass_pfw,     0, 1, "3pass_pfw"     },
            { l6_run_fused_pfw,     0, 1, "fused_pfw"     },
            { l6_run_3pass_nt_pf,   1, 1, "3pass_nt_pf"   },
#ifdef L6_HAVE_AVX512
            /* forced-only perf A/B targets (r7 VERDICT §6); never raced */
            { l6_run_zff,           0, 0, "zff"           },
            { l6_run_z2s,           0, 0, "z2s"           },
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
            /* settle spin (L17_rader's r5 finding: on a ramping clock a
             * fixed-order table mis-ranked bit-identical work by 76%);
             * 100 ms of 256-bit work brings the core to its steady licence
             * clock before round 0 is timed.  Unscored. */
            l6_spin256(0.1);
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
                if (!ok[c] || !cand[c].raced) continue;
                cand[c].k(p->t1, p->t2, ain, aout, nt);      /* warm */
                if (cand[c].fence) _mm_sfence();
            }
            for (int round = 0; round < 7; ++round) {
                for (int c = 0; c < ncand; ++c) {
                    if (!ok[c] || !cand[c].raced) continue;
                    /* per-candidate licence warm-up (panel_r7, per
                     * LITERATURE 08 s4.3): run the candidate itself,
                     * untimed, for ~0.7 ms, so a ymm candidate is never
                     * timed inside a zmm predecessor's licence tail
                     * (CLX licence dwell is ~670 us -- comparable to a
                     * whole 2 ms trial) and a zmm candidate never pays
                     * its upward transition inside its own timing. */
                    double wu = l6_now() + 7e-4;
                    do {
                        cand[c].k(p->t1, p->t2, ain, aout, nt);
                    } while (l6_now() < wu);
                    if (cand[c].fence) _mm_sfence();
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
                if (!ok[c] || !cand[c].raced) continue;
                if (best < 0 || tmin[c] < bestt * (1.0 - L6_MARGIN)) {
                    bestt = tmin[c]; best = c;
                } else if (tmin[c] < bestt) bestt = tmin[c];
            }
            if (getenv("L6_VERBOSE"))
                for (int c = 0; c < ncand; ++c)
                    fprintf(stderr, "L6_unrolled race: %-14s %s %10.4f us/vol%s\n",
                            cand[c].nm,
                            !ok[c] ? "BAD" : cand[c].raced ? "ok " : "unr",
                            (ok[c] && cand[c].raced)
                                ? tmin[c] / (double)nt * 1e6 : 0.0,
                            c == best ? "   <-- chosen" : "");
        }
        if (best < 0)
            for (int c = 0; c < ncand; ++c)
                if (ok[c] && cand[c].raced) { best = c; break; }
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
    {   /* report the raced winner and the full clock ladder on the
         * leaderboard line: clk256=<sparse,mid,sat> clk512=<sparse,sat>
         * sustained GHz (see the ladder comment above; sat256 is the
         * number the r5 VERDICT asks for, and mid256 is the licence
         * clock at approximately the real kernel's FP density).  A
         * trailing ! marks an L6_FORCE pick (not a tournament one).
         * Order: all 256-bit probes strictly before any 512-bit one. */
        double g2s = 0.0, g2m = 0.0, g2x = 0.0, g5s = 0.0, g5x = 0.0;
        double gk = 0.0;
#ifdef L6_HAVE_AVX2
        /* kclk first (it needs the chosen kernel and nothing else);
         * the ladder's own 20 ms warm loops then clear whatever
         * licence the dwell established before anything else is read */
        gk = l6_kclk(p->run, p->fence, p->t1, p->t2);
        /* two full ladder passes, per-probe max: wallaby can spend the
         * first ~1 s of a session ramping (validated: a ladder read 2.10
         * while the driver, seconds later, timed the kernel at full
         * clock), and one pass can land entirely inside the ramp.  Each
         * probe's 20 ms 256-bit warm loop also clears any 512 licence
         * left by the preceding probe, so pass 2's 256-bit numbers stay
         * honest. */
        for (int pass = 0; pass < 2; ++pass) {
            double t;
            t = l6_probe256_sparse(); if (t > g2s) g2s = t;
            t = l6_probe256_mid();    if (t > g2m) g2m = t;
            t = l6_probe256_sat();    if (t > g2x) g2x = t;
#ifdef L6_HAVE_AVX512
            t = l6_probe512_sparse(); if (t > g5s) g5s = t;
            t = l6_probe512_sat();    if (t > g5x) g5x = t;
#endif
        }
        /* r6 fix: never return to the driver still inside the AVX-512
         * licence -- at B=1 the whole measurement fits in the recovery
         * window (the leading suspect for r5's B=1 regression). */
        l6_spin256(0.02);
        /* r8: then hand the driver the CHOSEN kernel's own steady state
         * (L6_pfa r7).  For a ymm pick this establishes the AVX2-heavy
         * licence the scored kernel runs in; for a forced zmm pick, its
         * own licence -- either way the driver never times a transition
         * that create() caused. */
        l6_dwell_chosen(p->run, p->fence, p->t1, p->t2);
#endif
        if (g2s > 0.0)
            snprintf(l6_desc, sizeof(l6_desc),
                     "L=6: unrolled PFA 2x3 codelet (48 flop/36 instr, no "
                     "twiddles), ymm raced, zmm forced-only; variant=%s%s "
                     "clk256=%.2f,%.2f,%.2f clk512=%.2f,%.2f kclk=%.2fGHz",
                     p->chosen, p->forced ? "!" : "",
                     g2s, g2m, g2x, g5s, g5x, gk);
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
