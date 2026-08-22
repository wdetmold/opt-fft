/* L45_mixedradix -- forward complex-double 3D DFT of a fixed 45^3 cube.
 *
 * TECHNIQUE
 *   Row-column (three 1-D passes) with a Good-Thomas / prime-factor 9x5 line
 *   transform, batch-vectorised across the *lines* of each pass, and blocked
 *   per x-plane.  This is L36_mixedradix's architecture (the L=36 leader since
 *   panel_r1) transplanted to L=45; the differences forced by 45 being odd are
 *   documented in strategies/L45_mixedradix.md.
 *
 *   45 = 9*5 with gcd(9,5)=1, so the Good-Thomas (prime-factor) map applies
 *   and the entire inter-stage twiddle stage vanishes:
 *
 *       n = (5*n1 + 9*n2) mod 45        (Ruritanian input map, n1<9, n2<5)
 *       k = (10*k1 + 36*k2) mod 45      (CRT output map: 10 = 5*[5^-1]_9,
 *                                        36 = 9*[9^-1]_5)
 *       n*k = 5*n1*k1 + 9*n2*k2 (mod 45)  =>  W45^{nk} = W9^{n1 k1} W5^{n2 k2}
 *
 *   so a 45-point DFT is exactly 9 independent 5-point DFTs followed by 5
 *   independent 9-point DFTs with *no* twiddles in between -- only a
 *   compile-time index permutation, free because every pass is already a
 *   strided gather/scatter.  The 9-point module is Cooley-Tukey 3x3 (9 = 3^2
 *   is a prime power, PFA cannot help inside it): 3 DFT3, 4 nontrivial complex
 *   multiplies by W9 constants, 3 DFT3.  The 5-point module is FFTW n1_5's
 *   FMA DAG (sqrt(5)/4 cosine split + scaled-sine trick; borrowed from
 *   L45_pfa r6), 2 FMA-port ops cheaper than the classic two-rotation form.
 *   A plain Cooley-Tukey 9x5 line transform (twiddles kept, the literal
 *   reading of this entry's name) was counted and rejected: it needs 32
 *   nontrivial inter-stage CMULs = +64 FMA-port ops per line (428 vs 364).
 *
 *   Interleaved complex is kept end to end.  A vector holds PW complex
 *   numbers = PW *lines* of the current pass, so every constant is
 *   lane-invariant and there is no cross-lane operation inside the transform.
 *   Only the z pass -- whose transform axis is the contiguous one -- needs a
 *   transpose, done in registers as PW x PW blocks of 128-bit complex lanes.
 *
 *   45 is not a multiple of PW (4 or 2), so each pass has a tail:
 *     * z pass  (in -> plane) and y pass (plane -> out) are out-of-place, so
 *       the last block simply OVERLAPS the previous one (starts at 45-PW):
 *       the overlapped lines are recomputed with identical results.  Costs
 *       one extra block in 12 (PW=4), same instruction count as a masked
 *       tail, and needs no extra codelet.
 *     * x pass (in place in `out`) cannot overlap, but it CAN tile across y
 *       boundaries: all 2025 (y,z) lines share the same x-stride and are
 *       contiguous in flat index, so 2025 = 506*PW + 1 at PW=4 leaves ONE
 *       masked-load/masked-store tail call per volume (was one per y).
 *     * the z-pass gather/scatter handles its odd 45th *column* with masked
 *       loads/stores through the same PWxPW transpose.
 *   The plane scratch rows are padded to PPITCH = 52 complex (832 B = 13
 *   cache lines, 64B-aligned rows, coprime with the 64 L1 sets -- see the
 *   PPITCH comment); accesses to `in`/`out` rotate alignment mod 64 (odd-L
 *   fact of life).
 *
 *   Pass structure -- two sweeps over the volume, not three:
 *     phase 1, per x-plane (45x45 complex = 31.6 KiB + padded plane scratch):
 *              45 z-lines (transpose in, transform, transpose out into the
 *              plane scratch), then 45 y-lines straight into `out`.
 *     phase 2: x-lines, in place in `out` (stride 32400 B), tiled PW
 *              consecutive z at a time.
 *   The 1.39 MiB volume does NOT fit the target's 1 MiB L2 (unlike L=36), so
 *   phase 2 runs out of L3 at B=1; the volume+volume working set of 2.78 MiB
 *   sits comfortably in the 22 MiB L3.
 *
 *   MEMORY MECHANISMS (tournament-gated candidates, never defaults; since r8
 *   they are ORTHOGONAL candidates, not a cumulative ladder -- the node's r7
 *   verdict showed the phase-2 poke is pure overhead there, and it was
 *   poisoning every bundled pfin/pfw candidate):
 *   * pf:   phase 2 prefetches its 45 x-streams by hand (1 line ahead) --
 *           more streams than the L2 streamer tracks.  (L36_mixedradix r1.)
 *           pkw variant pokes with PREFETCHW instead of T0: phase 2 is an
 *           in-place read-modify-write, so fetch-for-ownership does both
 *           jobs (shape of L64_radix8's node-picked pfw).
 *   * pfin: paced T1 prefetch of the phase-1 `in` read stream, one x-plane
 *           (31.6 KB) ahead, spread over the z-subloop's 12 codelet calls, so
 *           the read of plane x+1 overlaps the compute of plane x.  Positional
 *           (derived from the plane pointer), so it cannot drift across a
 *           batch.  Plus a 1-line-per-call re-cover of in[b+1]'s first plane
 *           from phase 2 (L36_pfa's PFIN/PFNX, via L36_mixedradix r4, both
 *           attributed).
 *   * pfw:  paced WRITE-INTENT prefetch (__builtin_prefetch(p,1,3) ->
 *           prefetchw) over phase 1's cold-`out` store stream, one plane
 *           ahead, spread over the y-subloop; acquires lines exclusive so the
 *           RFO overlaps compute (L36_pfa r5's PFWMID; L6_unrolled r3).
 *   * cpy:  the y pass stores into an L1-hot plane image and one ERMS
 *           rep-movsb per x-plane copies it to `out`.  ERMS full-line writes
 *           skip the RFO read of the destination, deleting 1.46 MB/volume of
 *           cold-`out` fill traffic (L3 at B=1, DRAM when streaming) instead
 *           of merely overlapping it, and unlike NT stores the lines stay in
 *           the cache hierarchy for phase 2 to re-read.  NEW in r8, mine.
 *   No NT-store path: the node rejected NT in every L=36 tournament for four
 *   consecutive rounds (L36_mixedradix r6 retired it; see that record).
 *
 * OPERATION COUNT (per 45-point line, as vector instructions over PW lanes)
 *       DFT5  : 16 FMA-port ops + 2 shuffles        x9  = 144 + 18
 *       DFT3  :  6 FMA-port ops + 1 shuffle         x30 = 180 + 30
 *       CMUL  :  2 FMA-port ops + 1 shuffle         x20 =  40 + 20
 *       total :                        364 FMA-port ops + 68 shuffles / PW lines
 *   Real flops: 9*48 + 5*(6*18 + 4*6) = 432 + 660 = 1092 per 45-point line;
 *   per volume 3 * 45^2 * 1092 = 6,633,900 flops in 6075 line transforms
 *   (plus 6.7% overlap recompute in the z and y passes; phase 2 is exactly
 *   506 full + 1 masked call per volume at PW=4).
 *
 * ASSUMPTIONS
 *   * L == 45 only; fft3d_supports() refuses everything else.
 *   * `in` and `out` are 64-byte aligned and distinct (driver guarantees).
 *   * `out` doubles as the working buffer between phase 1 and phase 2; `in`
 *     is never written.
 *   * Three kernels: V0 = AVX2+FMA (2 complex lanes, 16 ymm), V1 = AVX-512F
 *     (4 complex lanes, 32 zmm), V2 = AVX-512VL (2 complex lanes, 32 ymm).
 *     fft3d_create() verifies each against V0 numerically, then times all
 *     surviving (kernel x mechanism) combinations and installs the fastest,
 *     with a 3% simplest-first hysteresis (L36_mixedradix r4/r5 tuner).
 */

#ifndef VAR
/* ======================= common (width-independent) part ================== */

#define _POSIX_C_SOURCE 200809L

#include <complex.h>
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "fft3d_api.h"

#define LSIDE 45
#define NPLANE (LSIDE * LSIDE)          /* complex per x-plane = 2025    */
#define NVOL   (LSIDE * LSIDE * LSIDE)  /* complex per volume  = 91125   */
#ifndef PPITCH
#define PPITCH 52                       /* plane-scratch row pitch, complex:
                                           52*16 = 832 B = 13 cache lines,
                                           64-byte aligned rows.  13 is coprime
                                           with the 64 L1 sets, so the y-pass
                                           column walk (45 rows, stride PPITCH)
                                           spreads over all sets; the "obvious"
                                           pad 48 = 12 lines has gcd(12,64)=4
                                           and hits only 16 sets.  Borrowed
                                           from L45_pfa r6 (their A/B: pitch 52
                                           15-20% faster at B=1); overridable
                                           with -DPPITCH=48 for node A/B. */
#endif
#define PFLINES 508                     /* 64B lines covering one x-plane     */

/* ---- pre-splatted constants, 8-double 64-byte rows usable as memory
   operands by both the 256-bit and 512-bit kernels. ---------------------- */
#define SPLAT8(v) { (v), (v), (v), (v), (v), (v), (v), (v) }
#define ALT8(v)   { (v), -(v), (v), -(v), (v), -(v), (v), -(v) }

static const double KC_HALF[8] __attribute__((aligned(64))) = SPLAT8(0.5);
/* sqrt(3)/2 alternating: swap(m) * this = -i*s*m in interleaved layout */
static const double KC_KS[8] __attribute__((aligned(64)))
    = ALT8(8.66025403784438646764e-01);
/* W9^k = exp(-2*pi*i*k/9): C = Re, D = Im, k = 1, 2, 4 */
static const double KC_C1[8] __attribute__((aligned(64))) = SPLAT8( 7.66044443118978035202e-01);
static const double KC_D1[8] __attribute__((aligned(64))) = SPLAT8(-6.42787609686539326323e-01);
static const double KC_C2[8] __attribute__((aligned(64))) = SPLAT8( 1.73648177666930348852e-01);
static const double KC_D2[8] __attribute__((aligned(64))) = SPLAT8(-9.84807753012208059367e-01);
static const double KC_C4[8] __attribute__((aligned(64))) = SPLAT8(-9.39692620785908384054e-01);
static const double KC_D4[8] __attribute__((aligned(64))) = SPLAT8(-3.42020143325668733044e-01);
/* 5-point, FFTW n1_5 FMA constants (borrowed from L45_pfa r6):
 * c1,c2 = -1/4 +- sqrt(5)/4 (cosine split), s2 = phi^-1 * s1 (scaled sine) */
static const double KC_Q4[8] __attribute__((aligned(64))) = SPLAT8(2.5e-01);
static const double KC_S5[8] __attribute__((aligned(64))) = SPLAT8(5.59016994374947424102e-01);
static const double KC_PHI[8] __attribute__((aligned(64))) = SPLAT8(6.18033988749894848205e-01);
static const double KC_S1[8] __attribute__((aligned(64))) = ALT8(9.51056516295153572116e-01);
/* one-complex tail mask for the AVX2 maskload/maskstore path */
static const long long KC_TMASK[4] __attribute__((aligned(32))) = { -1, -1, 0, 0 };

/* stage 1: nine 5-point DFTs; stage 2: five 9-point DFTs */
#define REP9(M) M(0) M(1) M(2) M(3) M(4) M(5) M(6) M(7) M(8)
#define REP5(M) M(0) M(1) M(2) M(3) M(4)

typedef void (*exec_fn)(const double *, double *, long, double *);

struct fft3d_plan {
    long batch;
    exec_fn fn;
    double *plane;     /* 45 rows x 48-complex pitch scratch for phase 1 */
    void *raw;
};

/* one ERMS rep-movsb per x-plane: with cpy, the y pass stores into an L1-hot
 * plane image and this copies it to `out`.  ERMS full-cache-line writes do
 * not read-for-ownership the destination, so the 1.46 MB/volume RFO read of
 * cold `out` (L3-resident at B=1, DRAM in the streaming regime) disappears
 * from the traffic, while the lines still land in the cache hierarchy
 * (unlike NT stores, which the node rejected four rounds running at L=36
 * and which sweep 2 would then have to re-read from DRAM). */
static inline void plane_copy(void *dst, const void *src, size_t n)
{
#if defined(__x86_64__)
    void *d = dst;
    const void *s = src;
    size_t c = n;
    __asm__ volatile("rep movsb" : "+D"(d), "+S"(s), "+c"(c) : : "memory");
#else
    memcpy(dst, src, n);
#endif
}

/* exec_<variant>_<mech>; mechanism codes (see MJ[] in fft3d_create):
 *   0 pf0            no memory mechanism
 *   1 pf1            phase-2 45-stream T0 poke, 1 line ahead
 *   2 pfin           paced phase-1 input T1 prefetch (+ next-volume re-cover)
 *   3 pfw            paced phase-1 prefetchw on cold out, 1 plane ahead
 *   4 pfin-pfw       2 + 3
 *   5 pkw            phase-2 poke with PREFETCHW (in-place RMW pass: one
 *                    fetch-for-ownership instead of T0 read + store upgrade)
 *   6 pkw-pfin-pfw   5 + 2 + 3
 *   7 pf1-pfin-pfw   1 + 2 + 3   (round r7's streaming incumbent)
 *   8 cpy            y pass -> plane image, rep movsb -> out (no-RFO stores)
 *   9 cpy-pfin       8 + 2
 */
#define DECL_EXEC(V)                                                     \
static void exec_##V##_0(const double *, double *, long, double *);      \
static void exec_##V##_1(const double *, double *, long, double *);      \
static void exec_##V##_2(const double *, double *, long, double *);      \
static void exec_##V##_3(const double *, double *, long, double *);      \
static void exec_##V##_4(const double *, double *, long, double *);      \
static void exec_##V##_5(const double *, double *, long, double *);      \
static void exec_##V##_6(const double *, double *, long, double *);      \
static void exec_##V##_7(const double *, double *, long, double *);      \
static void exec_##V##_8(const double *, double *, long, double *);      \
static void exec_##V##_9(const double *, double *, long, double *);
DECL_EXEC(0)
DECL_EXEC(1)
DECL_EXEC(2)
#undef DECL_EXEC

/* instantiate the kernel once per (ISA, vector width) variant */
#define VAR 0
#include __FILE__
#undef VAR
#define VAR 1
#include __FILE__
#undef VAR
#define VAR 2
#include __FILE__
#undef VAR

/* ------------------------------ the API ---------------------------------- */

const char *fft3d_name(void) { return "L45_mixedradix"; }

static char g_desc[224];

const char *fft3d_description(void)
{
    return g_desc[0] ? g_desc
                     : "row-column PFA 9x5 line codelet, batch-vectorised over "
                       "lines, 2 sweeps, AVX2/AVX-512 + prefetch autotuned";
}

int fft3d_supports(int L) { return L == LSIDE; }

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

fft3d_plan *fft3d_create(int L, int batch)
{
    if (L != LSIDE || batch < 1) return NULL;

    fft3d_plan *p = (fft3d_plan *)calloc(1, sizeof *p);
    if (!p) return NULL;
    p->batch = batch;
    void *pl = NULL;
    /* [0 .. LSIDE*PPITCH) padded plane scratch for the z->y handoff;
     * [LSIDE*PPITCH .. +NPLANE) plane image for the cpy mechanism (laid out
     * exactly as one x-plane of `out`; offset 37440 B keeps it 64B-aligned) */
    if (posix_memalign(&pl, 64,
                       (size_t)(LSIDE * PPITCH + NPLANE) * 2 * sizeof(double)) != 0 || !pl) {
        free(p);
        return NULL;
    }
    p->plane = (double *)pl;
    p->raw = pl;
    p->fn = exec_0_0;

    /* regime: does the batch stream through this machine's LLC?  Decides
     * which candidates are IN PLAY (pfw only where `out` is genuinely cold;
     * two entries measured prefetchw at +11..17% on cache-resident volumes)
     * and how the tuning arena is sized. */
    long l3 = -1;
    {
#ifdef _SC_LEVEL3_CACHE_SIZE
        l3 = sysconf(_SC_LEVEL3_CACHE_SIZE);
#endif
        if (l3 <= 0) l3 = 22l << 20;   /* the scoring node's 22 MiB */
    }
    double foot = (double)batch * (double)NVOL * 16.0 * 2.0;
    int streaming = foot > 1.25 * (double)l3;

    /* self-tuning; all setup, excluded from the score.  Every candidate must
     * match exec_0_0's output to 1e-13 relative before it is eligible.
     * In the streaming regime the arena must actually STREAM on the machine
     * doing the tuning (L36_pfa r2 lesson, reproduced by L36_mixedradix r3):
     * in+out = 2.5x this machine's L3, clamped to [12, 64] volumes. */
    exec_fn cand[24];
    const char *cnm[24];
    int ncand = 0;

    long nt;
    if (streaming) {
        long arena = (long)(2.5 * (double)l3 / ((double)NVOL * 32.0)) + 1;
        if (arena < 12) arena = 12;
        if (arena > 64) arena = 64;
        nt = batch < arena ? batch : arena;
    } else {
        nt = batch < 4 ? batch : 4;
    }
    size_t nd = (size_t)NVOL * 2 * (size_t)nt;
    double *ti = NULL, *o0 = NULL, *ox = NULL;
    if (posix_memalign((void **)&ti, 4096, nd * sizeof(double)) == 0 &&
        posix_memalign((void **)&o0, 4096, nd * sizeof(double)) == 0 &&
        posix_memalign((void **)&ox, 4096, nd * sizeof(double)) == 0) {

        uint64_t s = 0x9E3779B97F4A7C15ull;
        for (size_t i = 0; i < nd; ++i) {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            ti[i] = (double)(int64_t)(s >> 11) * (1.0 / 9007199254740992.0);
        }
        memset(o0, 0, nd * sizeof(double));
        exec_0_0(ti, o0, nt, p->plane);

        int have_512 = 0, have_vl = 0;
#if defined(__x86_64__)
        __builtin_cpu_init();
        have_512 = __builtin_cpu_supports("avx512f");
        have_vl  = have_512 && __builtin_cpu_supports("avx512vl");
#endif
        /* Mechanism table; indices are the exec_<v>_<mech> suffix.  r8 change:
         * the mechanisms are now ORTHOGONAL candidates instead of a cumulative
         * ladder.  Node r7 evidence (9/9 runs each): v1-pf0 won B=1/B=2 and
         * the verdict calls the phase-2 T0 poke "pure overhead on the scoring
         * machine" -- but pfin and pfw were only ever offered bundled WITH
         * that poke, so they were never cleanly tested where they matter most
         * (out does not fit the node's 1 MB L2 even at B=1, unlike L=36, so
         * the L36 "no pfw when cache-resident" rule does not map here). */
        static const struct {
            unsigned char pf, pfk, pfin, pfw, cpy;
            const char *nm;
        } MJ[10] = {
            {0,0,0,0,0,"pf0"},          {1,0,0,0,0,"pf1"},
            {0,0,1,0,0,"pfin"},         {0,0,0,1,0,"pfw"},
            {0,0,1,1,0,"pfin-pfw"},     {1,1,0,0,0,"pkw"},
            {1,1,1,1,0,"pkw-pfin-pfw"}, {1,0,1,1,0,"pf1-pfin-pfw"},
            {0,0,0,0,1,"cpy"},          {0,0,1,0,1,"cpy-pfin"},
        };
        exec_fn XT[3][10] = {
            {exec_0_0, exec_0_1, exec_0_2, exec_0_3, exec_0_4,
             exec_0_5, exec_0_6, exec_0_7, exec_0_8, exec_0_9},
            {exec_1_0, exec_1_1, exec_1_2, exec_1_3, exec_1_4,
             exec_1_5, exec_1_6, exec_1_7, exec_1_8, exec_1_9},
            {exec_2_0, exec_2_1, exec_2_2, exec_2_3, exec_2_4,
             exec_2_5, exec_2_6, exec_2_7, exec_2_8, exec_2_9},
        };
        /* diagnostic overrides for paired A/B runs, read once at plan time so
         * execution stays repeatable: FFT45_PFIN=0|1, FFT45_PFW=0|1,
         * FFT45_CPY=0|1, FFT45_PF=0|1|w (phase-2 poke: none / T0 / prefetchw),
         * FFT45_V=0|1|2 restricts the pool to one kernel width. */
        int pfinmode = -1, pfwmode = -1, vmode = -1, cpymode = -1, pfmode = -1;
        {
            const char *po = getenv("FFT45_PFIN");
            if (po && (*po == '0' || *po == '1')) pfinmode = *po - '0';
            const char *wo = getenv("FFT45_PFW");
            if (wo && (*wo == '0' || *wo == '1')) pfwmode = *wo - '0';
            const char *co = getenv("FFT45_CPY");
            if (co && (*co == '0' || *co == '1')) cpymode = *co - '0';
            const char *fo = getenv("FFT45_PF");
            if (fo && (*fo == '0' || *fo == '1')) pfmode = *fo - '0';
            else if (fo && (*fo == 'w' || *fo == 'W')) pfmode = 2;
            const char *vo = getenv("FFT45_V");
            if (vo && *vo >= '0' && *vo <= '2') vmode = *vo - '0';
        }

        /* Pools as (variant, mech) pairs, INCUMBENT FIRST: the 3% hysteresis
         * pick favours earlier candidates, so the r7 node winners head their
         * pools and every new mechanism must beat them by a real margin.
         * Kernel-width order follows the node's own 9/9-stable picks
         * (V1 in the cached regime, V2 in the streaming one); the other
         * widths keep a thin insurance slate. */
        /* singles before combos within a width: the hysteresis resolves
         * within-3% ties to the EARLIER candidate, and a single mechanism is
         * the simpler explanation (measured at B=8 on wallaby: pfw 187.6
         * beat pfin-pfw 191.8, but listed after it, lost the tie). */
        static const unsigned char POOL_CACHED[][2] = {
            {1,0}, {1,8}, {1,2}, {1,3}, {1,5}, {1,9}, {1,4}, {1,6},
            {2,0}, {2,9}, {2,4},
            {0,0}, {0,9},
        };
        static const unsigned char POOL_STREAM[][2] = {
            {2,7}, {2,9}, {2,6}, {2,4},
            {1,7}, {1,9}, {1,6},
            {0,7}, {0,9},
        };
        const unsigned char (*pool)[2] = streaming ? POOL_STREAM : POOL_CACHED;
        int npool = streaming ? (int)(sizeof POOL_STREAM / 2)
                              : (int)(sizeof POOL_CACHED / 2);

        exec_fn probe[24];
        const char *pnm[24];
        static char pbuf[24][24];
        int nprobe = 0;
        for (int i = 0; i < npool; ++i) {
            int v = pool[i][0], m = pool[i][1];
            if (v == 1 && !have_512) continue;
            if (v == 2 && !have_vl) continue;
            if (vmode >= 0 && v != vmode) continue;
            if (pfinmode >= 0 && MJ[m].pfin != pfinmode) continue;
            if (pfwmode >= 0 && MJ[m].pfw != pfwmode) continue;
            if (cpymode >= 0 && MJ[m].cpy != cpymode) continue;
            if (pfmode == 0 && MJ[m].pf) continue;
            if (pfmode == 1 && !(MJ[m].pf && !MJ[m].pfk)) continue;
            if (pfmode == 2 && !MJ[m].pfk) continue;
            snprintf(pbuf[nprobe], sizeof pbuf[0], "v%d-%s", v, MJ[m].nm);
            probe[nprobe] = XT[v][m];
            pnm[nprobe] = pbuf[nprobe];
            ++nprobe;
        }
        for (int k = 0; k < nprobe; ++k) {
            memset(ox, 0, nd * sizeof(double));
            probe[k](ti, ox, nt, p->plane);
            double num = 0.0, den = 0.0;
            for (size_t i = 0; i < nd; ++i) {
                double d = ox[i] - o0[i];
                num += d * d;
                den += o0[i] * o0[i];
            }
            if (den > 0.0 && sqrt(num / den) < 1e-13) {
                cand[ncand] = probe[k];
                cnm[ncand] = pnm[k];
                ++ncand;
            }
        }
        if (ncand == 0) {
            cand[ncand] = exec_0_0; cnm[ncand] = "v0-pf0-fallback"; ++ncand;
            cand[ncand] = exec_0_1; cnm[ncand] = "v0-pf1-fallback"; ++ncand;
        }

        /* time every survivor, several interleaved rounds, keep per-candidate
         * minimum; small arenas get more reps (under-sampling flipped picks
         * at L=36 until r4 fixed it this way). */
        double best[24];
        for (int k = 0; k < ncand; ++k) best[k] = 1e300;
        /* more rounds at tiny arenas: wallaby showed a fast/slow machine
         * state toggling on a seconds scale, and a toggle edge crossing the
         * tournament mis-ranks kernels; min-over-more-interleaved-rounds
         * gives every candidate a shot at the fast state. */
        int reps = nt >= 16 ? 1 : (nt >= 4 ? 4 : 16);
        int rounds = nt >= 16 ? 4 : (nt >= 4 ? 6 : 10);
        for (int round = 0; round < rounds; ++round) {
            for (int k = 0; k < ncand; ++k) {
                cand[k](ti, ox, nt, p->plane);        /* warm */
                double t0 = now_s();
                for (int r = 0; r < reps; ++r)
                    cand[k](ti, ox, nt, p->plane);
                double dt = now_s() - t0;
                if (dt < best[k]) best[k] = dt;
            }
        }
        if (getenv("FFT45_VERBOSE"))
            for (int k = 0; k < ncand; ++k)
                fprintf(stderr, "cand %-20s best %.3f us/vol\n",
                        cnm[k], best[k] * 1e6 / ((double)reps * (double)nt));
        /* hysteresis pick: a later (more speculative) candidate must beat the
         * incumbent by > 3% (coin-flip zone measured at 2.4% at L=36). */
        int bk = 0;
        for (int k = 1; k < ncand; ++k)
            if (best[k] < 0.97 * best[bk]) bk = k;
        p->fn = cand[bk];
        snprintf(g_desc, sizeof g_desc,
                 "PFA 9x5 2-sweep, lanes=lines; pick=%s (B=%d, arena=%ld vol, "
                 "stream=%d, %d cand)", cnm[bk], batch, nt, streaming, ncand);
    }
    free(ti); free(o0); free(ox);
    return p;
}

void fft3d_execute(fft3d_plan *plan, const double _Complex *in, double _Complex *out)
{
    plan->fn((const double *)in, (double *)out, plan->batch, plan->plane);
}

void fft3d_destroy(fft3d_plan *plan)
{
    if (!plan) return;
    free(plan->raw);
    free(plan);
}

#else /* ================= per-variant instantiation ======================== */

#define XCAT2(a, b) a##b
#define XCAT(a, b) XCAT2(a, b)
#define FN(n)     XCAT(XCAT(n##_, VAR), _0)
#define FNC(n, c) XCAT(XCAT(XCAT(n##_, VAR), _), c)

#if VAR == 1
/* ---- 512-bit: 4 complex lanes per zmm, 32 registers ---- */
#pragma GCC push_options
#pragma GCC target("avx512f")
#define PW 4
#define VD __m512d
#define VLOAD(p)          _mm512_loadu_pd(p)
#define VSTORE(p, v)      _mm512_storeu_pd((p), (v))
#define VLOADT(p)         _mm512_maskz_loadu_pd((__mmask8)0x03, (p))
#define VSTORET(p, v)     _mm512_mask_storeu_pd((p), (__mmask8)0x03, (v))
#define VADD(a, b)        _mm512_add_pd((a), (b))
#define VSUB(a, b)        _mm512_sub_pd((a), (b))
#define VMUL(a, b)        _mm512_mul_pd((a), (b))
#define VFMADD(a, b, c)   _mm512_fmadd_pd((a), (b), (c))
#define VFMSUB(a, b, c)   _mm512_fmsub_pd((a), (b), (c))
#define VFNMADD(a, b, c)  _mm512_fnmadd_pd((a), (b), (c))
#define VFMADDSUB(a,b,c)  _mm512_fmaddsub_pd((a), (b), (c))
#define VSWAP(a)          _mm512_permute_pd((a), 0x55)
#define PWLIST(M)         M(0) M(1) M(2) M(3)
/* 4x4 transpose of 128-bit complex lanes (involution) */
#define TRANSP(A, B) do {                                                 \
    VD z0 = _mm512_shuffle_f64x2((A)[0], (A)[1], 0x88);                   \
    VD z1 = _mm512_shuffle_f64x2((A)[0], (A)[1], 0xDD);                   \
    VD z2 = _mm512_shuffle_f64x2((A)[2], (A)[3], 0x88);                   \
    VD z3 = _mm512_shuffle_f64x2((A)[2], (A)[3], 0xDD);                   \
    (B)[0] = _mm512_shuffle_f64x2(z0, z2, 0x88);                          \
    (B)[2] = _mm512_shuffle_f64x2(z0, z2, 0xDD);                          \
    (B)[1] = _mm512_shuffle_f64x2(z1, z3, 0x88);                          \
    (B)[3] = _mm512_shuffle_f64x2(z1, z3, 0xDD);                          \
} while (0)
#else
/* ---- 256-bit: 2 complex lanes per ymm.  VAR 0 = VEX/AVX2 (16 registers),
   VAR 2 = EVEX/AVX-512VL (32 registers, no 512-bit path) ---- */
#pragma GCC push_options
#if VAR == 0
#pragma GCC target("avx2,fma")
#else
#pragma GCC target("avx512vl,avx512f,fma")
#endif
#define PW 2
#define VD __m256d
#define VLOAD(p)          _mm256_loadu_pd(p)
#define VSTORE(p, v)      _mm256_storeu_pd((p), (v))
#define VLOADT(p)         _mm256_maskload_pd((p), _mm256_load_si256((const __m256i *)KC_TMASK))
#define VSTORET(p, v)     _mm256_maskstore_pd((p), _mm256_load_si256((const __m256i *)KC_TMASK), (v))
#define VADD(a, b)        _mm256_add_pd((a), (b))
#define VSUB(a, b)        _mm256_sub_pd((a), (b))
#define VMUL(a, b)        _mm256_mul_pd((a), (b))
#define VFMADD(a, b, c)   _mm256_fmadd_pd((a), (b), (c))
#define VFMSUB(a, b, c)   _mm256_fmsub_pd((a), (b), (c))
#define VFNMADD(a, b, c)  _mm256_fnmadd_pd((a), (b), (c))
#define VFMADDSUB(a,b,c)  _mm256_fmaddsub_pd((a), (b), (c))
#define VSWAP(a)          _mm256_permute_pd((a), 0x5)
#define PWLIST(M)         M(0) M(1)
/* 2x2 transpose of 128-bit complex lanes; involution */
#define TRANSP(A, B) do {                                        \
    VD z0 = _mm256_permute2f128_pd((A)[0], (A)[1], 0x20);        \
    VD z1 = _mm256_permute2f128_pd((A)[0], (A)[1], 0x31);        \
    (B)[0] = z0; (B)[1] = z1;                                    \
} while (0)
#endif

#define NB   ((LSIDE + PW - 1) / PW)   /* blocks per pass: 12 (PW=4), 23 (PW=2) */
#define NGF  (LSIDE / PW)              /* full blocks: 11 (PW=4), 22 (PW=2)     */
#define PFCH ((PFLINES + NB - 1) / NB) /* prefetch lines per paced call         */

#define C_HALF VLOAD(KC_HALF)
#define C_KS   VLOAD(KC_KS)
#define C_Q4   VLOAD(KC_Q4)
#define C_S5   VLOAD(KC_S5)
#define C_PHI  VLOAD(KC_PHI)
#define C_S1   VLOAD(KC_S1)

/* 3-point DFT, interleaved complex: 6 FMA-port ops + 1 shuffle.
 *   t = x1+x2 ; m = x1-x2 ; a = x0 - t/2
 *   y0 = x0 + t ; y1 = a - i*s*m ; y2 = a + i*s*m ,  s = sqrt(3)/2
 * and -i*s*m is swap(m) * [s,-s,...] in interleaved layout. */
#define DFT3M(a0, a1, a2, r0, r1, r2) {                 \
    VD _t = VADD((a1), (a2));                           \
    VD _m = VSUB((a1), (a2));                           \
    VD _a = VFNMADD(C_HALF, _t, (a0));                  \
    VD _s = VSWAP(_m);                                  \
    VD _q0 = VADD((a0), _t);                            \
    VD _q1 = VFMADD(_s, C_KS, _a);                      \
    VD _q2 = VFNMADD(_s, C_KS, _a);                     \
    (r0) = _q0; (r1) = _q1; (r2) = _q2;                 \
}

/* multiply by the constant (c + i d): 2 FMA-port ops + 1 shuffle */
#define CMULM(v, KC, KD) \
    VFMADDSUB((v), VLOAD(KC), VMUL(VSWAP(v), VLOAD(KD)))

/* stage 1: nine 5-point DFTs over n2, PFA input map n = (5*n1 + 9*n2) mod 45.
 * FFTW n1_5's FMA DAG (borrowed from L45_pfa r6, from the corpus):
 *   t1 = x1+x4, t4 = x1-x4, t2 = x2+x3, t3 = x2-x3
 *   te = t1+t2, ta = t1-t2;  y0 = x0 + te
 *   cosine split c1,c2 = -1/4 +- sqrt(5)/4:
 *     tm = x0 - te/4;  tp = tm + (sqrt(5)/4)*ta;  tq = tm - (sqrt(5)/4)*ta
 *   scaled sine s2 = phi^-1 * s1 (phi^-1 = 0.618...):
 *     w1 = swap(t4) + phi^-1*swap(t3);  w2 = phi^-1*swap(t4) - swap(t3)
 *   y1/y4 = tp +/- ALT(s1)*w1 ;  y2/y3 = tq +/- ALT(s1)*w2
 * with -i*s*t = swap(t) * [s,-s,...].  16 FMA-port ops + 2 shuffles
 * (2 ops fewer than the two-rotation form used in rounds up to r6).
 * Output y[k2] goes to TT[k2*9 + N1] so stage 2 reads contiguous runs. */
#define ST1G(TT, LS, N1) {                                     \
    VD x0 = LS((5 * (N1) + 9 * 0) % 45);                       \
    VD x1 = LS((5 * (N1) + 9 * 1) % 45);                       \
    VD x2 = LS((5 * (N1) + 9 * 2) % 45);                       \
    VD x3 = LS((5 * (N1) + 9 * 3) % 45);                       \
    VD x4 = LS((5 * (N1) + 9 * 4) % 45);                       \
    VD t1 = VADD(x1, x4), t4 = VSUB(x1, x4);                   \
    VD t2 = VADD(x2, x3), t3 = VSUB(x2, x3);                   \
    VD te = VADD(t1, t2), ta = VSUB(t1, t2);                   \
    VD tm = VFNMADD(C_Q4, te, x0);                             \
    VD tp = VFMADD(ta, C_S5, tm);                              \
    VD tq = VFNMADD(ta, C_S5, tm);                             \
    VD u4 = VSWAP(t4), u3 = VSWAP(t3);                         \
    VD w1 = VFMADD(u3, C_PHI, u4);                             \
    VD w2 = VFMSUB(u4, C_PHI, u3);                             \
    TT[0 * 9 + (N1)] = VADD(x0, te);                           \
    TT[1 * 9 + (N1)] = VFMADD(w1, C_S1, tp);                   \
    TT[2 * 9 + (N1)] = VFMADD(w2, C_S1, tq);                   \
    TT[3 * 9 + (N1)] = VFNMADD(w2, C_S1, tq);                  \
    TT[4 * 9 + (N1)] = VFNMADD(w1, C_S1, tp);                  \
}
#define ST1(N1) ST1G(T, LSRC, N1)

/* stage 2: five 9-point DFTs over n1 (Cooley-Tukey 3x3), PFA output map
 * k = (10*k1 + 36*k2) mod 45. */
#define ST2G(TT, SD, K2) {                                     \
    VD u0 = TT[(K2) * 9 + 0], u1 = TT[(K2) * 9 + 1];           \
    VD u2 = TT[(K2) * 9 + 2], u3 = TT[(K2) * 9 + 3];           \
    VD u4 = TT[(K2) * 9 + 4], u5 = TT[(K2) * 9 + 5];           \
    VD u6 = TT[(K2) * 9 + 6], u7 = TT[(K2) * 9 + 7];           \
    VD u8 = TT[(K2) * 9 + 8];                                  \
    VD a0, a1, a2, b0, b1, b2, c0, c1, c2;                     \
    DFT3M(u0, u3, u6, a0, a1, a2)                              \
    DFT3M(u1, u4, u7, b0, b1, b2)                              \
    DFT3M(u2, u5, u8, c0, c1, c2)                              \
    b1 = CMULM(b1, KC_C1, KC_D1);                              \
    b2 = CMULM(b2, KC_C2, KC_D2);                              \
    c1 = CMULM(c1, KC_C2, KC_D2);                              \
    c2 = CMULM(c2, KC_C4, KC_D4);                              \
    VD v0, v1, v2, v3, v4, v5, v6, v7, v8;                     \
    DFT3M(a0, b0, c0, v0, v3, v6)                              \
    DFT3M(a1, b1, c1, v1, v4, v7)                              \
    DFT3M(a2, b2, c2, v2, v5, v8)                              \
    SD((10 * 0 + 36 * (K2)) % 45, v0);                         \
    SD((10 * 1 + 36 * (K2)) % 45, v1);                         \
    SD((10 * 2 + 36 * (K2)) % 45, v2);                         \
    SD((10 * 3 + 36 * (K2)) % 45, v3);                         \
    SD((10 * 4 + 36 * (K2)) % 45, v4);                         \
    SD((10 * 5 + 36 * (K2)) % 45, v5);                         \
    SD((10 * 6 + 36 * (K2)) % 45, v6);                         \
    SD((10 * 7 + 36 * (K2)) % 45, v7);                         \
    SD((10 * 8 + 36 * (K2)) % 45, v8);                         \
}
#define ST2(K2) ST2G(T, SDST, K2)

/* 45-point PFA line transform, y axis: reads the padded plane scratch
 * (row pitch 48 complex, 64B-aligned rows), writes `out` at its natural
 * 45-complex stride.  Strides are compile-time literals. */
static inline __attribute__((always_inline))
void FN(dft45_y)(const double *src, double *dst)
{
    VD T[45];
#define LSRC(i)     VLOAD(src + (long)(i) * (PPITCH * 2))
#define SDST(i, v)  VSTORE(dst + (long)(i) * (LSIDE * 2), (v))
    REP9(ST1)
    REP5(ST2)
#undef LSRC
#undef SDST
}

/* 45-point PFA line transform, x axis, in place: stride 2025 complex. */
static inline __attribute__((always_inline))
void FN(dft45_x)(double *base)
{
    VD T[45];
#define LSRC(i)     VLOAD(base + (long)(i) * (NPLANE * 2))
#define SDST(i, v)  VSTORE(base + (long)(i) * (NPLANE * 2), (v))
    REP9(ST1)
    REP5(ST2)
#undef LSRC
#undef SDST
}

/* masked tail of the x pass: the single line at flat index 2024 = (y,z) =
 * (44,44) (2025 is odd and the in-place pass cannot overlap-recompute).
 * Only lane 0 is live; the dead lanes compute on zeros and are masked off
 * at the store.  One call per volume. */
static inline __attribute__((always_inline))
void FN(dft45_xt)(double *base)
{
    VD T[45];
#define LSRC(i)     VLOADT(base + (long)(i) * (NPLANE * 2))
#define SDST(i, v)  VSTORET(base + (long)(i) * (NPLANE * 2), (v))
    REP9(ST1)
    REP5(ST2)
#undef LSRC
#undef SDST
}

/* 45-point PFA line transform, vector array in/out (z axis) */
static inline __attribute__((always_inline))
void FN(dft45_v)(const VD *X, VD *Y)
{
    VD T[45];
#define LSRC(i)     X[(i)]
#define SDST(i, v)  Y[(i)] = (v)
    REP9(ST1)
    REP5(ST2)
#undef LSRC
#undef SDST
}

static inline __attribute__((always_inline))
void FN(body)(const double *in, double *out, long batch, double *plane,
              const int pf, const int pfk, const int pfin, const int pfw,
              const int cpy)
{
    const double *inend = in + batch * (long)NVOL * 2;
    double *outend = out + batch * (long)NVOL * 2;
    double *plane2 = plane + (long)LSIDE * PPITCH * 2;
    for (long b = 0; b < batch; ++b) {
        const double *vin  = in  + b * (long)NVOL * 2;
        double       *vout = out + b * (long)NVOL * 2;

        /* -------- phase 1: for each x-plane, z-lines then y-lines -------- */
        for (long x = 0; x < LSIDE; ++x) {
            const double *pin  = vin  + x * (long)NPLANE * 2;
            double       *pout = vout + x * (long)NPLANE * 2;
            /* the input/output planes one ahead of the ones in flight; `in`
             * and `out` are contiguous across planes AND volumes, so this
             * crosses volume boundaries by itself (the PFNX cold window is
             * only what phase 2 later evicts). */
            const double *npf = pin  + (long)NPLANE * 2;
            double       *npw = pout + (long)NPLANE * 2;

            /* z pass: lanes = PW consecutive y-rows; last block overlaps. */
            for (long yb = 0; yb < NB; ++yb) {
                long y0 = yb * PW;
                if (y0 > LSIDE - PW) y0 = LSIDE - PW;
                if (pfin) {
                    /* paced positional cursor: cover one plane of in, one
                     * plane ahead, spread over this subloop's NB calls */
                    long s = yb * PFCH, e = s + PFCH;
                    if (e > PFLINES) e = PFLINES;
#pragma GCC unroll 8
                    for (long i = s; i < e; ++i) {
                        const double *q = npf + i * 8;
                        if (q < inend)
                            _mm_prefetch((const char *)q, _MM_HINT_T1);
                    }
                }
                VD Xv[LSIDE], Yv[LSIDE];
                const double *rows = pin + y0 * (LSIDE * 2);
                /* explicit unrolls: the monitor's build has no -funroll-loops,
                 * and leaving these rolled cost 30% end to end (measured) */
#pragma GCC unroll 22
                for (long g = 0; g < NGF; ++g) {
                    VD A[PW], B[PW];
#define LDR(j) A[j] = VLOAD(rows + (long)(j) * (LSIDE * 2) + g * (PW * 2));
                    PWLIST(LDR)
#undef LDR
                    TRANSP(A, B);
#define PUT(j) Xv[g * PW + (j)] = B[j];
                    PWLIST(PUT)
#undef PUT
                }
                {   /* odd 45th column, z = 44: masked gather via transpose */
                    VD A[PW], B[PW];
#define LDT(j) A[j] = VLOADT(rows + (long)(j) * (LSIDE * 2) + 44 * 2);
                    PWLIST(LDT)
#undef LDT
                    TRANSP(A, B);
                    Xv[44] = B[0];
                }
                FN(dft45_v)(Xv, Yv);
                double *prow = plane + y0 * (PPITCH * 2);
#pragma GCC unroll 22
                for (long g = 0; g < NGF; ++g) {
                    VD A[PW], B[PW];
#define GET(j) A[j] = Yv[g * PW + (j)];
                    PWLIST(GET)
#undef GET
                    TRANSP(A, B);
#define PST(j) VSTORE(prow + (long)(j) * (PPITCH * 2) + g * (PW * 2), B[j]);
                    PWLIST(PST)
#undef PST
                }
                {   /* odd column scatter: B[j].lane0 = Yv[44].lane j */
                    VD A[PW], B[PW];
#define GT(j) A[j] = Yv[44];
                    PWLIST(GT)
#undef GT
                    TRANSP(A, B);
#define PTT(j) VSTORET(prow + (long)(j) * (PPITCH * 2) + 44 * 2, B[j]);
                    PWLIST(PTT)
#undef PTT
                }
            }

            /* y pass: lanes = PW consecutive z; last block overlaps
             * (plane -> out is out-of-place, recompute is idempotent).
             * With cpy the stores go to the L1-hot plane image instead of
             * cold `out`, and one ERMS rep-movsb per plane moves the image
             * out without the RFO read (see plane_copy). */
            double *ydst = cpy ? plane2 : pout;
            for (long zb = 0; zb < NB; ++zb) {
                long z0 = zb * PW;
                if (z0 > LSIDE - PW) z0 = LSIDE - PW;
                if (pfw) {
                    long s = zb * PFCH, e = s + PFCH;
                    if (e > PFLINES) e = PFLINES;
#pragma GCC unroll 8
                    for (long i = s; i < e; ++i) {
                        double *q = npw + i * 8;
                        if (q < outend)
                            __builtin_prefetch(q, 1, 3);
                    }
                }
                FN(dft45_y)(plane + z0 * 2, ydst + z0 * 2);
            }
            if (cpy)
                plane_copy(pout, plane2, (size_t)NPLANE * 2 * sizeof(double));
        }

        /* pfin cold-window re-cover of in[b+1]'s first plane (phase 2's
         * traffic would evict what phase 1's cursor already fetched) */
        const double *ncw = (pfin && b + 1 < batch)
                                ? in + (b + 1) * (long)NVOL * 2
                                : (const double *)0;
        long cwl = 0;

        /* -------- phase 2: x-lines, in place in `out`, tiled over the FLAT
         * (y,z) index.  All 2025 lines of a y-z plane share the same x-stride
         * and are contiguous in flat index, so nothing forces tiles to respect
         * y boundaries: 2025 = 506*4 + 1 (PW=4) or 1012*2 + 1 (PW=2), i.e.
         * ONE masked tail call per volume instead of one per y (was 45).
         * Kills 33 full-width codelet calls per volume (~2% of the ops). */
        for (long t = 0; t < NPLANE / PW; ++t) {
            double *base = vout + t * (PW * 2);
            if (pf) {
                /* 45 x-streams, each advancing 64 B per tile: more than
                 * the L2 streamer tracks, poke them `pf` lines ahead.
                 * pfk=1 pokes with PREFETCHW instead of T0: this pass reads
                 * then rewrites every line in place, so one
                 * fetch-for-ownership replaces a read fetch plus a store
                 * upgrade (the shape of L64_radix8's node-picked pfw). */
#pragma GCC unroll 45
                for (int i = 0; i < LSIDE; ++i) {
                    const char *q =
                        (const char *)(base + (long)i * (NPLANE * 2) + pf * 8);
                    if (pfk)
                        __builtin_prefetch(q, 1, 3);
                    else
                        _mm_prefetch(q, _MM_HINT_T0);
                }
            }
            if (ncw && cwl < PFLINES) {
                _mm_prefetch((const char *)(ncw + cwl * 8), _MM_HINT_T1);
                ++cwl;
            }
            FN(dft45_x)(base);
        }
        FN(dft45_xt)(vout + (long)(NPLANE - 1) * 2);
    }
}

/* one wrapper per mechanism code; each instantiates the always_inline body
 * with compile-time-constant flags (pf, pfk, pfin, pfw, cpy) */
#define MKEXEC(code, pf, pfk, pfi, pfw, cpy)                                  \
static void FNC(exec, code)(const double *in, double *out, long batch,        \
                            double *plane)                                    \
{                                                                             \
    FN(body)(in, out, batch, plane, pf, pfk, pfi, pfw, cpy);                  \
}
MKEXEC(0, 0, 0, 0, 0, 0)   /* pf0            */
MKEXEC(1, 1, 0, 0, 0, 0)   /* pf1            */
MKEXEC(2, 0, 0, 1, 0, 0)   /* pfin           */
MKEXEC(3, 0, 0, 0, 1, 0)   /* pfw            */
MKEXEC(4, 0, 0, 1, 1, 0)   /* pfin-pfw       */
MKEXEC(5, 1, 1, 0, 0, 0)   /* pkw            */
MKEXEC(6, 1, 1, 1, 1, 0)   /* pkw-pfin-pfw   */
MKEXEC(7, 1, 0, 1, 1, 0)   /* pf1-pfin-pfw   */
MKEXEC(8, 0, 0, 0, 0, 1)   /* cpy            */
MKEXEC(9, 0, 0, 1, 0, 1)   /* cpy-pfin       */
#undef MKEXEC

#pragma GCC pop_options

#undef XCAT2
#undef XCAT
#undef FN
#undef FNC
#undef PW
#undef VD
#undef VLOAD
#undef VSTORE
#undef VLOADT
#undef VSTORET
#undef VADD
#undef VSUB
#undef VMUL
#undef VFMADD
#undef VFMSUB
#undef VFNMADD
#undef VFMADDSUB
#undef VSWAP
#undef PWLIST
#undef TRANSP
#undef NB
#undef NGF
#undef PFCH
#undef C_HALF
#undef C_KS
#undef C_Q4
#undef C_S5
#undef C_PHI
#undef C_S1
#undef DFT3M
#undef CMULM
#undef ST1G
#undef ST1
#undef ST2G
#undef ST2

#endif /* VAR */
