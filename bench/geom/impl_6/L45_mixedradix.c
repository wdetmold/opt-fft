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
 *   multiplies by W9 constants, 3 DFT3.  The 5-point module is the classic
 *   two-real-rotation form (t1..t4, two cos-FMA chains, two -i*sin chains).
 *   A plain Cooley-Tukey 9x5 line transform (twiddles kept, the literal
 *   reading of this entry's name) was counted and rejected: it needs 32
 *   nontrivial inter-stage CMULs = +64 FMA-port ops per line (446 vs 382).
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
 *     * x pass (in place in `out`) cannot overlap; its tail (z = 44) is a
 *       masked-load/masked-store instantiation of the same codelet, one
 *       column per y.
 *     * the z-pass gather/scatter handles its odd 45th *column* with masked
 *       loads/stores through the same PWxPW transpose.
 *   The plane scratch is padded to a 48-complex row pitch (64B-aligned rows)
 *   so all plane accesses in the z-scatter and y-load are 64-byte aligned;
 *   accesses to `in`/`out` rotate alignment mod 64 (odd-L fact of life).
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
 *   MEMORY MECHANISMS (tournament-gated candidates, never defaults):
 *   * pf:   phase 2 prefetches its 45 x-streams by hand (`pf` lines ahead) --
 *           more streams than the L2 streamer tracks.  (L36_mixedradix r1.)
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
 *   No NT-store path: the node rejected NT in every L=36 tournament for four
 *   consecutive rounds (L36_mixedradix r6 retired it; see that record).
 *
 * OPERATION COUNT (per 45-point line, as vector instructions over PW lanes)
 *       DFT5  : 18 FMA-port ops + 2 shuffles        x9  = 162 + 18
 *       DFT3  :  6 FMA-port ops + 1 shuffle         x30 = 180 + 30
 *       CMUL  :  2 FMA-port ops + 1 shuffle         x20 =  40 + 20
 *       total :                        382 FMA-port ops + 68 shuffles / PW lines
 *   Real flops: 9*48 + 5*(6*18 + 4*6) = 432 + 660 = 1092 per 45-point line;
 *   per volume 3 * 45^2 * 1092 = 6,633,900 flops in 6075 line transforms
 *   (plus 6.7% overlap recompute in the z and y passes).
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
#define PPITCH 48                       /* plane-scratch row pitch, complex:
                                           48*16 = 768 B, 64-byte aligned rows */
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
/* 5-point: cos/sin(2*pi/5), cos/sin(4*pi/5); sines in alternating form */
static const double KC_Q1[8] __attribute__((aligned(64))) = SPLAT8( 3.09016994374947424102e-01);
static const double KC_Q2[8] __attribute__((aligned(64))) = SPLAT8(-8.09016994374947424102e-01);
static const double KC_R1[8] __attribute__((aligned(64))) = ALT8(9.51056516295153572116e-01);
static const double KC_R2[8] __attribute__((aligned(64))) = ALT8(5.87785252292473129169e-01);
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

/* exec_<variant>_<code>:
 *   0 = cached, no prefetch                        (pf0)
 *   1 = cached, phase-2 streams 1 line ahead       (pf1)
 *   2 = cached, phase-2 streams 4 lines ahead      (pf4)
 *   3 = code 1 + paced phase-1 input prefetch      (pf1-pfin)
 *   4 = code 3 + paced phase-1 prefetchw on out    (pf1-pfin-pfw)
 */
static void exec_0_0(const double *, double *, long, double *);
static void exec_0_1(const double *, double *, long, double *);
static void exec_0_2(const double *, double *, long, double *);
static void exec_0_3(const double *, double *, long, double *);
static void exec_0_4(const double *, double *, long, double *);
static void exec_1_0(const double *, double *, long, double *);
static void exec_1_1(const double *, double *, long, double *);
static void exec_1_2(const double *, double *, long, double *);
static void exec_1_3(const double *, double *, long, double *);
static void exec_1_4(const double *, double *, long, double *);
static void exec_2_0(const double *, double *, long, double *);
static void exec_2_1(const double *, double *, long, double *);
static void exec_2_2(const double *, double *, long, double *);
static void exec_2_3(const double *, double *, long, double *);
static void exec_2_4(const double *, double *, long, double *);

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
    if (posix_memalign(&pl, 64, (size_t)LSIDE * PPITCH * 2 * sizeof(double)) != 0 || !pl) {
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
    exec_fn cand[20];
    const char *cnm[20];
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
        /* diagnostic overrides for paired A/B runs, read once at plan time so
         * execution stays repeatable: FFT45_PFIN=0|1, FFT45_PFW=0|1, and
         * FFT45_V=0|1|2 restricts the pool to one kernel width. */
        int pfinmode = -1, pfwmode = -1, vmode = -1;
        {
            const char *po = getenv("FFT45_PFIN");
            if (po && (*po == '0' || *po == '1')) pfinmode = *po - '0';
            const char *wo = getenv("FFT45_PFW");
            if (wo && (*wo == '0' || *wo == '1')) pfwmode = *wo - '0';
            const char *vo = getenv("FFT45_V");
            if (vo && *vo >= '0' && *vo <= '2') vmode = *vo - '0';
        }
        int in_plain = (pfinmode != 1) && (pfwmode != 1);
        int in_pfin  = (pfinmode != 0) && (pfwmode != 1);
        int in_pfw   = (pfinmode != 0) && (pfwmode != 0);
        int in_v0 = (vmode == -1 || vmode == 0);
        int in_v1 = have_512 && (vmode == -1 || vmode == 1);
        int in_v2 = have_vl  && (vmode == -1 || vmode == 2);

        exec_fn probe[20];
        const char *pnm[20];
        int nprobe = 0;
#define PROBE(fn, nm) do { probe[nprobe] = (fn); pnm[nprobe] = (nm); ++nprobe; } while (0)
        if (streaming) {
            /* streaming pool, simplest first (3% hysteresis below) */
            if (in_v0 && in_plain) PROBE(exec_0_1, "v0-pf1");
            if (in_v0 && in_pfin)  PROBE(exec_0_3, "v0-pf1-pfin");
            if (in_v0 && in_pfw)   PROBE(exec_0_4, "v0-pf1-pfin-pfw");
            if (in_v1) {
                if (in_plain) PROBE(exec_1_1, "v1-pf1");
                if (in_pfin)  PROBE(exec_1_3, "v1-pf1-pfin");
                if (in_pfw)   PROBE(exec_1_4, "v1-pf1-pfin-pfw");
            }
            if (in_v2) {
                if (in_plain) PROBE(exec_2_1, "v2-pf1");
                if (in_pfin)  PROBE(exec_2_3, "v2-pf1-pfin");
                if (in_pfw)   PROBE(exec_2_4, "v2-pf1-pfin-pfw");
            }
        } else {
            /* cache-resident pool: no pfw (measured +11..17% loss in this
             * regime at L=36 by two entries; not re-litigated). */
            if (in_v0 && in_plain) {
                cand[ncand] = exec_0_0; cnm[ncand] = "v0-pf0"; ++ncand;
                cand[ncand] = exec_0_1; cnm[ncand] = "v0-pf1"; ++ncand;
                cand[ncand] = exec_0_2; cnm[ncand] = "v0-pf4"; ++ncand;
            }
            if (in_v0 && in_pfin) {
                cand[ncand] = exec_0_3; cnm[ncand] = "v0-pf1-pfin"; ++ncand;
            }
            if (in_v1) {
                if (in_plain) {
                    PROBE(exec_1_0, "v1-pf0");
                    PROBE(exec_1_1, "v1-pf1");
                    PROBE(exec_1_2, "v1-pf4");
                }
                if (in_pfin) PROBE(exec_1_3, "v1-pf1-pfin");
            }
            if (in_v2) {
                if (in_plain) {
                    PROBE(exec_2_0, "v2-pf0");
                    PROBE(exec_2_1, "v2-pf1");
                    PROBE(exec_2_2, "v2-pf4");
                }
                if (in_pfin) PROBE(exec_2_3, "v2-pf1-pfin");
            }
        }
#undef PROBE
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
        double best[20];
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
#define FN(n)   XCAT(XCAT(n##_, VAR), _0)
#define FNP1(n) XCAT(XCAT(n##_, VAR), _1)
#define FNP2(n) XCAT(XCAT(n##_, VAR), _2)
#define FNP3(n) XCAT(XCAT(n##_, VAR), _3)
#define FNP4(n) XCAT(XCAT(n##_, VAR), _4)

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
#define C_Q1   VLOAD(KC_Q1)
#define C_Q2   VLOAD(KC_Q2)
#define C_R1   VLOAD(KC_R1)
#define C_R2   VLOAD(KC_R2)

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
 *   t1 = x1+x4, t4 = x1-x4, t2 = x2+x3, t3 = x2-x3
 *   y0 = x0+t1+t2
 *   y1/y4 = x0 + c1*t1 + c2*t2 -/+ i*(s1*t4 + s2*t3)
 *   y2/y3 = x0 + c2*t1 + c1*t2 -/+ i*(s2*t4 - s1*t3)
 * with -i*s*t = swap(t) * [s,-s,...].  18 FMA-port ops + 2 shuffles.
 * Output y[k2] goes to TT[k2*9 + N1] so stage 2 reads contiguous runs. */
#define ST1G(TT, LS, N1) {                                     \
    VD x0 = LS((5 * (N1) + 9 * 0) % 45);                       \
    VD x1 = LS((5 * (N1) + 9 * 1) % 45);                       \
    VD x2 = LS((5 * (N1) + 9 * 2) % 45);                       \
    VD x3 = LS((5 * (N1) + 9 * 3) % 45);                       \
    VD x4 = LS((5 * (N1) + 9 * 4) % 45);                       \
    VD t1 = VADD(x1, x4), t4 = VSUB(x1, x4);                   \
    VD t2 = VADD(x2, x3), t3 = VSUB(x2, x3);                   \
    VD a1 = VFMADD(t2, C_Q2, VFMADD(t1, C_Q1, x0));            \
    VD a2 = VFMADD(t2, C_Q1, VFMADD(t1, C_Q2, x0));            \
    VD s4 = VSWAP(t4), s3 = VSWAP(t3);                         \
    VD b1 = VFMADD(s3, C_R2, VMUL(s4, C_R1));                  \
    VD b2 = VFNMADD(s3, C_R1, VMUL(s4, C_R2));                 \
    TT[0 * 9 + (N1)] = VADD(x0, VADD(t1, t2));                 \
    TT[1 * 9 + (N1)] = VADD(a1, b1);                           \
    TT[2 * 9 + (N1)] = VADD(a2, b2);                           \
    TT[3 * 9 + (N1)] = VSUB(a2, b2);                           \
    TT[4 * 9 + (N1)] = VSUB(a1, b1);                           \
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

/* masked tail of the x pass: the single line at z = 44 (45 is odd and the
 * in-place pass cannot overlap-recompute).  Only lane 0 is live; the dead
 * lanes compute on zeros and are masked off at the store. */
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
              const int pf, const int pfin, const int pfw)
{
    const double *inend = in + batch * (long)NVOL * 2;
    double *outend = out + batch * (long)NVOL * 2;
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
             * (plane -> out is out-of-place, recompute is idempotent). */
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
                FN(dft45_y)(plane + z0 * 2, pout + z0 * 2);
            }
        }

        /* pfin cold-window re-cover of in[b+1]'s first plane (phase 2's
         * traffic would evict what phase 1's cursor already fetched) */
        const double *ncw = (pfin && b + 1 < batch)
                                ? in + (b + 1) * (long)NVOL * 2
                                : (const double *)0;
        long cwl = 0;

        /* -------- phase 2: x-lines, in place in `out` -------------------- */
        for (long y = 0; y < LSIDE; ++y) {
            for (long zb = 0; zb < NGF; ++zb) {
                double *base = vout + (y * LSIDE + zb * PW) * 2;
                if (pf) {
                    /* 45 x-streams, each advancing 64 B per tile: more than
                     * the L2 streamer tracks, poke them `pf` lines ahead */
#pragma GCC unroll 45
                    for (int i = 0; i < LSIDE; ++i)
                        _mm_prefetch((const char *)(base + (long)i * (NPLANE * 2) + pf * 8),
                                     _MM_HINT_T0);
                }
                if (ncw && cwl < PFLINES) {
                    _mm_prefetch((const char *)(ncw + cwl * 8), _MM_HINT_T1);
                    ++cwl;
                }
                FN(dft45_x)(base);
            }
            FN(dft45_xt)(vout + (y * LSIDE + 44) * 2);
        }
    }
}

static void FN(exec)(const double *in, double *out, long batch, double *plane)
{
    FN(body)(in, out, batch, plane, 0, 0, 0);
}

static void FNP1(exec)(const double *in, double *out, long batch, double *plane)
{
    FN(body)(in, out, batch, plane, 1, 0, 0);
}

static void FNP2(exec)(const double *in, double *out, long batch, double *plane)
{
    FN(body)(in, out, batch, plane, 4, 0, 0);
}

static void FNP3(exec)(const double *in, double *out, long batch, double *plane)
{
    FN(body)(in, out, batch, plane, 1, 1, 0);
}

static void FNP4(exec)(const double *in, double *out, long batch, double *plane)
{
    FN(body)(in, out, batch, plane, 1, 1, 1);
}

#pragma GCC pop_options

#undef XCAT2
#undef XCAT
#undef FN
#undef FNP1
#undef FNP2
#undef FNP3
#undef FNP4
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
#undef C_Q1
#undef C_Q2
#undef C_R1
#undef C_R2
#undef DFT3M
#undef CMULM
#undef ST1G
#undef ST1
#undef ST2G
#undef ST2

#endif /* VAR */
