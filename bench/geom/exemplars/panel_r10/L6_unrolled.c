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
 *  * panel_r9: (a) the codelet macros are parameterized (VD6 vs VD63) and
 *    three raced fused3 twins ship: VD63 is the SAME PFA 2x3 DFT factored
 *    DFT3-first/DFT2-last (L6_pfa's ordering; the r8 VERDICT's standing
 *    hypothesis for the B=64 gap), token-identical kernels otherwise --
 *    the controlled test of the last named mechanism at this geometry.
 *    Not bit-identical to VD6 (different association, ~1e-16); the plan
 *    gate and numpy bound it.  (b) an in-plan B=1 discriminator (adopted
 *    from L36_pfa's r8 in-plan probe pattern, per the r8 VERDICT's L=6
 *    instruction): create() times fused / fused3 / zff at nvol=1,
 *    licence-fair, and reports min ns/vol as ab1=... in the description,
 *    plus the fused3-vs-fused race delta as f3d=...  perf_event_open is
 *    closed cluster-wide (perf_event_paranoid=4), so timing is the only
 *    in-plan discriminator available.
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
 *  * panel_r10: the r9 node data answered every question the r9 machinery
 *    carried, so the file is cut to what is still open.  (a) DELETED: the
 *    fused3/VD63 twins (f3d read +3.3..+6.2% on the node -- the radix-2-
 *    first VD6 is confirmed the winning association, both L=6 entries now
 *    run it), the whole AVX-512 section incl. zff/z2s (ab1 zf/f = 1.14-1.18
 *    on the node closed the uop theory with a published number, and the r9
 *    VERDICT formally withdrew the perf-counter ask that kept them
 *    compiled), and the 5-probe clock ladder (the clock consensus is
 *    settled panel-wide: 3.89 non-AVX / 2.89 licence; kclk stays as the
 *    one-number regression check).  My own r5 lesson motivates the prune:
 *    ~500 lines of never-executed zmm code once moved B=1 by +3.5% at an
 *    identical pick -- dead weight is not free in this file.  (b) ADOPTED
 *    FROM L6_pfa: the zp-outer/y-inner x-pass group order (their PASS_X,
 *    in their file since round 1) as raced twins fused_zp{,_pf,_pfw}.
 *    Their r9 B=1 winner fused_d2 = my VD6 graph + THEIR zp-outer x order;
 *    the ascending-x + VD6 + no-pf combination (= my fused) raced only in
 *    MY file, and mine reads 0.2174-0.2305 where theirs reads 0.2068x3 --
 *    the x group order is the one structural difference left between the
 *    two kernels, so it races here and ab1 now publishes the f-vs-fx A/B
 *    (fused vs fused_zp at nvol=1) on every leaderboard line.  The twins
 *    are output-bit-identical to their parents (same per-line arithmetic,
 *    different order over independent lines), so they carry a reduced 1.0%
 *    takeover margin where genuinely-new shapes keep 2.5%.
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
/* Takeover margins are per-candidate since panel_r10 (see the cand[] table):
 * 2.5% for genuinely different shapes (raised from 1.5% in panel_r4 after an
 * L3-marginal mis-pick at B=4096), 1.0% for the fused_zp twins, whose output
 * is bit-identical to their parents so a mis-pick is bounded and harmless. */

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

/* panel_r10 prune: the VD63 (DFT3-first) codelet twin is DELETED.  The r9
 * node data answered the question it existed for: f3d = +3.3..+6.2% in all
 * six readings, ab1 f < f3 in every process -- the radix-2-first VD6 above
 * is the winning association on Cascade Lake and both L=6 entries now run
 * it.  Derivation and numbers live in strategies/L6_unrolled.md (r9). */

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
 * final output pointer, only used by the prefetchw hooks.  CD is the
 * codelet macro.  Groups walk offsets 4g, g = 0..17: loads and stores in
 * strictly ascending 32B steps. */
#define L6_PASS_X(SRC,DST,OUT,PF,CD)                                    \
    do {                                                                \
        for (int g = 0; g < 18; ++g) {                                  \
            const double *s = (SRC) + 4*g;                              \
            double *d = (DST) + 4*g;                                    \
            PF(SRC,OUT,g);                                              \
            __m256d v0=_mm256_load_pd(s+  0), v1=_mm256_load_pd(s+ 72); \
            __m256d v2=_mm256_load_pd(s+144), v3=_mm256_load_pd(s+216); \
            __m256d v4=_mm256_load_pd(s+288), v5=_mm256_load_pd(s+360); \
            CD(v0,v1,v2,v3,v4,v5, v0,v1,v2,v3,v4,v5);                   \
            _mm256_store_pd(d+  0,v0); _mm256_store_pd(d+ 72,v1);       \
            _mm256_store_pd(d+144,v2); _mm256_store_pd(d+216,v3);       \
            _mm256_store_pd(d+288,v4); _mm256_store_pd(d+360,v5);       \
        }                                                               \
    } while (0)

/* x-pass, zp-outer/y-inner group order -- ADOPTED FROM L6_pfa's PASS_X
 * (in their file since round 1; their r9 node B=1 winner fused_d2 = the
 * VD6 graph behind exactly this group order).  Same 18 groups, same
 * per-group body, walked as g = (y,zp) with zp outer: offsets 12y + 4zp
 * doubles, i.e. 96-byte steps within a zp block.  Output bit-identical to
 * L6_PASS_X (the groups are independent; only their order changes).  The
 * prefetch index 6*zp+y still walks the next volume's 54 lines in strictly
 * ascending address order, so the streamer sees the same stream. */
#define L6_PASS_X_ZP(SRC,DST,OUT,PF,CD)                                 \
    do {                                                                \
        for (int zp = 0; zp < 3; ++zp)                                  \
            for (int y = 0; y < 6; ++y) {                               \
                const double *s = (SRC) + 12*y + 4*zp;                  \
                double *d = (DST) + 12*y + 4*zp;                        \
                PF(SRC,OUT,6*zp + y);                                   \
                __m256d v0=_mm256_load_pd(s+  0), v1=_mm256_load_pd(s+ 72); \
                __m256d v2=_mm256_load_pd(s+144), v3=_mm256_load_pd(s+216); \
                __m256d v4=_mm256_load_pd(s+288), v5=_mm256_load_pd(s+360); \
                CD(v0,v1,v2,v3,v4,v5, v0,v1,v2,v3,v4,v5);               \
                _mm256_store_pd(d+  0,v0); _mm256_store_pd(d+ 72,v1);   \
                _mm256_store_pd(d+144,v2); _mm256_store_pd(d+216,v3);   \
                _mm256_store_pd(d+288,v4); _mm256_store_pd(d+360,v5);   \
            }                                                           \
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
        L6_PASS_X(ip, t1, op, PF, VD6);                                 \
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
#define L6_FUSED_YZ(SRC,DST,ST,CD)                                      \
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
            CD(P00,P10,P20,P30,P40,P50, P00,P10,P20,P30,P40,P50);       \
            CD(P01,P11,P21,P31,P41,P51, P01,P11,P21,P31,P41,P51);       \
            CD(P02,P12,P22,P32,P42,P52, P02,P12,P22,P32,P42,P52);       \
            L6_ZPAIR(P00,P01,P02,P10,P11,P12, d+ 0, ST, CD);            \
            L6_ZPAIR(P20,P21,P22,P30,P31,P32, d+24, ST, CD);            \
            L6_ZPAIR(P40,P41,P42,P50,P51,P52, d+48, ST, CD);            \
        }                                                               \
    } while (0)

/* PX = L6_PASS_X (ascending) or L6_PASS_X_ZP (zp-outer, panel_r10). */
#define L6_DEF_FUSED_PX(NAME,PX,ST,PF,CD)                               \
L6_KALIGN static void NAME(double *restrict t1, double *restrict t2,    \
                 const double *restrict in, double *restrict out,       \
                 long nvol)                                             \
{                                                                       \
    VSET;                                                               \
    (void)t2;                                                           \
    for (long b = 0; b < nvol; ++b) {                                   \
        const double *ip = in  + b * (long)L6_VD;                       \
        double       *op = out + b * (long)L6_VD;                       \
        PX(ip, t1, op, PF, CD);                                         \
        L6_FUSED_YZ(t1, op, ST, CD);                                    \
    }                                                                   \
}
#define L6_DEF_FUSED(NAME,ST,PF,CD) \
        L6_DEF_FUSED_PX(NAME,L6_PASS_X,ST,PF,CD)

#define L6_ZPAIR(A0,A1,A2,B0,B1,B2, D, ST, CD)                          \
    do {                                                                \
        __m256d w0=_mm256_permute2f128_pd(A0,B0,0x20);                  \
        __m256d w1=_mm256_permute2f128_pd(A0,B0,0x31);                  \
        __m256d w2=_mm256_permute2f128_pd(A1,B1,0x20);                  \
        __m256d w3=_mm256_permute2f128_pd(A1,B1,0x31);                  \
        __m256d w4=_mm256_permute2f128_pd(A2,B2,0x20);                  \
        __m256d w5=_mm256_permute2f128_pd(A2,B2,0x31);                  \
        CD(w0,w1,w2,w3,w4,w5, w0,w1,w2,w3,w4,w5);                       \
        ST((D)+ 0,_mm256_permute2f128_pd(w0,w1,0x20));                  \
        ST((D)+ 4,_mm256_permute2f128_pd(w2,w3,0x20));                  \
        ST((D)+ 8,_mm256_permute2f128_pd(w4,w5,0x20));                  \
        ST((D)+12,_mm256_permute2f128_pd(w0,w1,0x31));                  \
        ST((D)+16,_mm256_permute2f128_pd(w2,w3,0x31));                  \
        ST((D)+20,_mm256_permute2f128_pd(w4,w5,0x31));                  \
    } while (0)

L6_DEF_FUSED(l6_run_fused,        _mm256_store_pd,  L6_PF_NONE,  VD6)
L6_DEF_FUSED(l6_run_fused_pf,     _mm256_store_pd,  L6_PF_T0_1,  VD6)
L6_DEF_FUSED(l6_run_fused_pfw,    _mm256_store_pd,  L6_PF_T0W_1, VD6)

/* panel_r10: the zp-outer x-pass twins (group order adopted from L6_pfa;
 * see L6_PASS_X_ZP).  Token-identical to the three kernels above except
 * for the x-pass group order; output bit-identical to them. */
L6_DEF_FUSED_PX(l6_run_fused_zp,     L6_PASS_X_ZP, _mm256_store_pd, L6_PF_NONE,  VD6)
L6_DEF_FUSED_PX(l6_run_fused_zp_pf,  L6_PASS_X_ZP, _mm256_store_pd, L6_PF_T0_1,  VD6)
L6_DEF_FUSED_PX(l6_run_fused_zp_pfw, L6_PASS_X_ZP, _mm256_store_pd, L6_PF_T0W_1, VD6)

/* panel_r7 prune: the ymm split-store shapes 3pass_s / fused_s (and the
 * L6_ZPAIR_S macro) are deleted -- 0 picks in 12 node invocations across
 * r4/r5, SPR-only mechanism.  panel_r9 prune: the fused3/VD63 twins are
 * deleted (question answered on the node, f3d = +3.3..+6.2%). */


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

static char l6_desc[512] =
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

/* Busy 256-bit spin for SECS wall seconds.  Used (a) before the race, so
 * round 0 is not ranked on a ramping clock (L17_rader's r5 settle-spin
 * finding: 76% mis-ranking from table order alone), and (b) near the END
 * of create() as belt-and-braces before the chosen-kernel dwell.  (The
 * original motivation -- letting the AVX-512 licence expire after the r5
 * zmm probes -- is moot since panel_r10: no 512-bit instruction exists in
 * this file any more.) */
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

/* ------------------------------------------------------------------ *
 * In-plan B=1 discriminator (panel_r9) -- ADOPTED FROM L36_pfa's r8
 * in-plan node probe pattern, which the r8 VERDICT calls the round's
 * most reusable idea and explicitly asks L=6 to apply: when the monitor
 * cannot run your counter, build the discriminator into create() and
 * route the result through fft3d_description().  perf_event_open is
 * closed on this cluster (perf_event_paranoid=4 on every machine we can
 * see), so this is a TIMED discriminator: min ns/volume at nvol=1,
 * driver-like conditions (same in/out every call, placed scratch), 9
 * trials of 256 reps, each trial preceded by ~0.7 ms of the kernel
 * itself so every candidate is measured in its own licence/clock steady
 * state (the r7 licence-fair race rule).  panel_r10 retarget: the r9
 * questions (fused vs fused3, fused vs zff) are answered and those
 * kernels deleted; ab1 now runs for fused / fused_zp -- the ascending
 * vs zp-outer x-pass group order, the one structural difference left
 * between my B=1 kernel and L6_pfa's node-winning fused_d2 -- so the
 * node publishes the A/B regardless of which one the tournament picks.
 * ~17 ms total, unscored.
 * ------------------------------------------------------------------ */
static double l6_ab1(l6_kernel run, int fence, double *t1, double *t2,
                     const double *din, double *dout)
{
    double best = 1e300;
    for (int trial = 0; trial < 9; ++trial) {
        double wu = l6_now() + 7e-4;
        do { run(t1, t2, din, dout, 1); } while (l6_now() < wu);
        if (fence) _mm_sfence();
        double t0 = l6_now();
        for (int r = 0; r < 256; ++r) run(t1, t2, din, dout, 1);
        if (fence) _mm_sfence();
        double dt = (l6_now() - t0) * (1.0 / 256.0);
        if (dt < best) best = dt;
    }
    return best * 1e9;   /* ns per volume */
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

    /* panel_r10 in-plan discriminator results (see l6_ab1) and the
     * fused_zp-vs-fused race delta at the plan's own batch size. */
    double ab_f = 0.0, ab_fx = 0.0, xod = 0.0;
    int have_xod = 0;

#ifdef L6_HAVE_AVX2
    {
        /* Ordered "safest first": a later candidate must beat the incumbent by
         * more than its own takeover margin, so measurement noise on a loaded
         * machine cannot promote a streaming-store kernel at a batch size
         * where the working set is cache resident.
         *
         * panel_r10 grid: 10 raced ymm.  The incumbent 7 are the panel_r6
         * pruning (the node has only ever selected {3pass,fused} x
         * {plain,pf,pfw}; NT 0-for-6 rounds but one representative kept).
         * New: the three fused_zp twins (zp-outer x-pass, adopted from
         * L6_pfa), each placed directly after its ascending parent.  They
         * are output-BIT-IDENTICAL to their parents (same per-line
         * arithmetic, different order over independent groups), so a
         * mis-pick between twin and parent is bounded by the margin and
         * cannot change the answer -- which is why they carry a reduced
         * 1.0% margin where genuinely different shapes keep the 2.5%
         * hysteresis (raised in r4 after an L3-marginal mis-pick).
         * The fused3 twins and the zmm forced-only kernels are deleted
         * (questions answered on the r9 node, see the file header).
         * Layout defence: every kernel entry 64B-pinned (since r6). */
        static const struct {
            l6_kernel k; int fence; double mg; const char *nm;
        } cand[] = {
            { l6_run_3pass,         0, 0.025, "3pass"         },
            { l6_run_fused,         0, 0.025, "fused"         },
            { l6_run_fused_zp,      0, 0.010, "fused_zp"      },
            { l6_run_3pass_pf,      0, 0.025, "3pass_pf"      },
            { l6_run_fused_pf,      0, 0.025, "fused_pf"      },
            { l6_run_fused_zp_pf,   0, 0.010, "fused_zp_pf"   },
            { l6_run_3pass_pfw,     0, 0.025, "3pass_pfw"     },
            { l6_run_fused_pfw,     0, 0.025, "fused_pfw"     },
            { l6_run_fused_zp_pfw,  0, 0.010, "fused_zp_pfw"  },
            { l6_run_3pass_nt_pf,   1, 0.025, "3pass_nt_pf"   },
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
                if (!ok[c]) continue;
                cand[c].k(p->t1, p->t2, ain, aout, nt);      /* warm */
                if (cand[c].fence) _mm_sfence();
            }
            for (int round = 0; round < 7; ++round) {
                for (int c = 0; c < ncand; ++c) {
                    if (!ok[c]) continue;
                    /* per-candidate licence warm-up (panel_r7, per
                     * LITERATURE 08 s4.3): run the candidate itself,
                     * untimed, for ~0.7 ms, so every candidate is timed
                     * in its own licence/clock steady state, never a
                     * predecessor's tail (CLX licence dwell is ~670 us
                     * -- comparable to a whole 2 ms trial). */
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
            /* Safest-first with a per-candidate takeover margin (panel_r10:
             * the bit-identical zp twins carry 1.0%, everything else keeps
             * the 2.5% hysteresis).  The reference time tracks the true
             * minimum even when the incumbent survives, so a chain of
             * sub-margin steps cannot drift the pick. */
            double bestt = 1e300;
            for (int c = 0; c < ncand; ++c) {
                if (!ok[c]) continue;
                if (best < 0 || tmin[c] < bestt * (1.0 - cand[c].mg)) {
                    bestt = tmin[c]; best = c;
                } else if (tmin[c] < bestt) bestt = tmin[c];
            }
            if (getenv("L6_VERBOSE"))
                for (int c = 0; c < ncand; ++c)
                    fprintf(stderr, "L6_unrolled race: %-14s %s %10.4f us/vol%s\n",
                            cand[c].nm,
                            !ok[c] ? "BAD" : "ok ",
                            ok[c] ? tmin[c] / (double)nt * 1e6 : 0.0,
                            c == best ? "   <-- chosen" : "");
            /* fused_zp-vs-fused delta at the plan's own raced batch size,
             * best of family vs best of family, for the description
             * (positive = the zp-outer x order is slower). */
            {
                double bf = 1e300, bfx = 1e300;
                for (int c = 0; c < ncand; ++c) {
                    if (!ok[c]) continue;
                    if (strncmp(cand[c].nm, "fused_zp", 8) == 0) {
                        if (tmin[c] < bfx) bfx = tmin[c];
                    } else if (strncmp(cand[c].nm, "fused", 5) == 0) {
                        if (tmin[c] < bf) bf = tmin[c];
                    }
                }
                if (bf < 1e299 && bfx < 1e299) {
                    xod = (bfx / bf - 1.0) * 100.0;
                    have_xod = 1;
                }
            }
        }
        if (best < 0)
            for (int c = 0; c < ncand; ++c)
                if (ok[c]) { best = c; break; }
        free(ain); free(aout);
        p->t1 = p->arena; p->t2 = p->arena + L6_VD; p->placed = 0;

        if (best >= 0) {
            p->run    = cand[best].k;
            p->fence  = cand[best].fence;
            p->chosen = cand[best].nm;
            p->forced = (best == forced);
        }

        /* ---- in-plan B=1 discriminator (panel_r10, see l6_ab1's comment):
         * fused (ascending x-pass) vs fused_zp (zp-outer).  Runs at every
         * batch size (it is ~17 ms and the question is B=1-specific either
         * way). ---- */
        {
            double *bin  = (double *)l6_alloc(L6_VD * sizeof(double));
            double *bout = (double *)l6_alloc(L6_VD * sizeof(double));
            if (bin && bout) {
                uint64_t st = 0xBF58476D1CE4E5B9ull;
                for (long i = 0; i < L6_VD; ++i) {
                    st ^= st << 13; st ^= st >> 7; st ^= st << 17;
                    bin[i] = (double)(int64_t)(st >> 11)
                             * (1.0 / 9007199254740992.0);
                }
                memset(bout, 0, L6_VD * sizeof(double));
                l6_place(p, bin, bout);
                for (int c = 0; c < ncand; ++c) {
                    if (!ok[c]) continue;
                    if (strcmp(cand[c].nm, "fused") == 0)
                        ab_f  = l6_ab1(cand[c].k, cand[c].fence,
                                       p->t1, p->t2, bin, bout);
                    else if (strcmp(cand[c].nm, "fused_zp") == 0)
                        ab_fx = l6_ab1(cand[c].k, cand[c].fence,
                                       p->t1, p->t2, bin, bout);
                }
                p->t1 = p->arena; p->t2 = p->arena + L6_VD; p->placed = 0;
            }
            free(bin); free(bout);
        }
    }
#endif
    {   /* report the raced winner, the kernel-context clock (kclk, the one
         * clock number still worth a line: the panel consensus is settled
         * at 3.89 non-AVX / 2.89 licence and kclk is the regression check),
         * and the ab1/xod x-order A/B.  A trailing ! marks an L6_FORCE
         * pick (not a tournament one). */
        double gk = 0.0;
#ifdef L6_HAVE_AVX2
        gk = l6_kclk(p->run, p->fence, p->t1, p->t2);
        /* belt-and-braces spin (nothing 512-bit exists in this file any
         * more, so there is no licence to clear -- kept because it is
         * free and it re-establishes the heavy-256 clock after kclk's
         * sparse read chain)... */
        l6_spin256(0.02);
        /* ...then hand the driver the CHOSEN kernel's own steady state
         * (r8, from L6_pfa's r7 refinement of my r6 licence-tail fix),
         * so the driver never times a transition that create() caused. */
        l6_dwell_chosen(p->run, p->fence, p->t1, p->t2);
#endif
        if (gk > 0.0) {
            /* ab1 = in-plan B=1 discriminator, min ns/volume at nvol=1,
             * licence-fair (each kernel self-warmed): f = fused
             * (ascending x-pass), fx = fused_zp (zp-outer x-pass,
             * L6_pfa's group order; 0.0 = not available/not gated).
             * xod = fused_zp vs fused family-best race delta at the
             * plan's batch size, positive = zp-outer slower. */
            int n = snprintf(l6_desc, sizeof(l6_desc),
                     "L=6: unrolled PFA 2x3 codelet (48 flop/36 instr, no "
                     "twiddles), radix-2-first VD6, ymm; variant=%s%s "
                     "kclk=%.2fGHz ab1=f%.1f,fx%.1fns",
                     p->chosen, p->forced ? "!" : "",
                     gk, ab_f, ab_fx);
            if (have_xod && n > 0 && (size_t)n < sizeof(l6_desc))
                snprintf(l6_desc + n, sizeof(l6_desc) - (size_t)n,
                         " xod=%+.1f%%", xod);
        } else
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
