/* L36_mixedradix -- forward complex-double 3D DFT of a fixed 36^3 cube.
 *
 * TECHNIQUE
 *   Row-column (three 1-D passes) with a Good-Thomas / prime-factor 4x9 line
 *   transform, batch-vectorised across the *lines* of each pass, and blocked so
 *   that each pass's working set is L1-resident.
 *
 *   36 = 4*9 and gcd(4,9)=1, so the Good-Thomas (prime-factor) map applies and
 *   the entire inter-stage twiddle stage vanishes:
 *
 *       n = (9*n1 + 4*n2) mod 36        (Ruritanian input map, n1<4, n2<9)
 *       k = (9*k1 + 28*k2) mod 36       (CRT output map, [9^-1]_4 = 1, [4^-1]_9 = 7)
 *       n*k = 9*n1*k1 + 4*n2*k2  (mod 36)  =>  W36^{nk} = W4^{n1 k1} * W9^{n2 k2}
 *
 *   so a 36-point DFT is exactly 9 independent 4-point DFTs followed by 4
 *   independent 9-point DFTs with *no* twiddles in between -- only a
 *   compile-time index permutation, which is free because every pass is already
 *   a strided gather/scatter.  The 9-point module is Cooley-Tukey 3x3:
 *   3 DFT3, 4 nontrivial complex multiplies, 3 DFT3.
 *
 *   Interleaved complex is kept end to end (it is the driver's layout, so no
 *   de/re-interleaving pass exists at all).  A vector holds PW complex numbers
 *   = PW *lines* of the current pass, so every twiddle constant is
 *   lane-invariant and there is not one cross-lane operation inside the
 *   transform.  The y and x passes vectorise over the contiguous z index and
 *   need no data reorganisation whatsoever; only the z pass -- whose transform
 *   axis *is* the contiguous one -- needs a transpose, done in registers as
 *   PW x PW blocks of 128-bit complex lanes.
 *
 *   Pass structure -- two passes over the volume, not three:
 *     phase 1, per x-plane (36x36 complex = 20.25 KiB, L1-resident):
 *              36 z-lines (transpose in, transform, transpose out, into an L1
 *              plane scratch), then 36 y-lines straight into `out`.
 *     phase 2: 36*(36/PW) x-lines, in place in `out` (stride 20736 B), tiled PW
 *              consecutive z at a time so every touched cache line is consumed
 *              in full and each of the 36 x-streams is sequential.
 *   Traffic is read(in) + write(out) + read(out) + write(out); the 746 KiB
 *   volume fits the 1 MiB L2 of the target part, so phase 2 runs out of L2.
 *
 * OPERATION COUNT (per 36-point line, as vector instructions over PW lanes)
 *       DFT4  :  8 FMA-port ops + 1 shuffle       x9   =  72 + 9
 *       DFT3  :  6 FMA-port ops + 1 shuffle       x24  = 144 + 24
 *       CMUL  :  2 FMA-port ops + 1 shuffle       x16  =  32 + 16
 *       total :                        248 FMA-port ops + 49 shuffles / PW lines
 *   In real flops: 9*20 + 4*(6*18 + 4*6) = 180 + 528 = 708 flops per 36-point
 *   line.  (The literature's PFA-4x9 reference is 688 flops / 464 instructions;
 *   the 20 extra flops come from using FMAs with a unit multiplier instead of
 *   add/sub pairs -- more flops, fewer instructions, which is the currency.)
 *   Per volume: 3 * 36^2 * 708 = 2,752,704 flops in 3 * 36^2 * 464/PW... i.e.
 *   3888 line transforms.  The driver's nominal yardstick is
 *   5*46656*log2(46656) = 3.618 Mflop.
 *   The z pass additionally spends 2*(36/PW)*(PW*PW-block) lane shuffles per PW
 *   lines; those land on port 5 and hide under the FMA port.
 *
 * ASSUMPTIONS
 *   * L == 36 only; fft3d_supports() refuses everything else.
 *   * `in` and `out` are 64-byte aligned and distinct (the driver guarantees
 *     both).  Every vector access made here is at a multiple of 64 bytes from
 *     the volume base -- (x*1296 + y*36 + zb*PW)*16 with 4 | 1296, 4 | 36 --
 *     so all accesses are naturally aligned even for 512-bit vectors.
 *     Unaligned intrinsics are used anyway; that costs nothing here.
 *   * `out` doubles as the working buffer between phase 1 and phase 2.  `in` is
 *     never written.
 *   * Three kernels are compiled: V0 = AVX2+FMA (2 complex lanes, 16 ymm),
 *     V1 = AVX-512F (4 complex lanes, 32 zmm), V2 = AVX-512VL (2 complex lanes
 *     but 32 registers and no 512-bit licence).  fft3d_create() checks CPU
 *     support, verifies each AVX-512 kernel numerically against the AVX2 one,
 *     then times all surviving (kernel x prefetch) combinations and keeps the
 *     fastest.  Nothing about AVX-512 being faster is assumed -- see
 *     strategies/L36_mixedradix.md.
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

#define LSIDE 36
#define NPLANE (LSIDE * LSIDE)          /* complex per x-plane          */
#define NVOL   (LSIDE * LSIDE * LSIDE)  /* complex per volume  = 46656  */

/* ---- pre-splatted constants.  Kept as 8-double, 64-byte-aligned rows so both
   the 256-bit and the 512-bit kernels can use them as a plain memory operand
   and spend no register on them. ---------------------------------------- */
#define SPLAT8(v) { (v), (v), (v), (v), (v), (v), (v), (v) }
#define ALT8(v)   { (v), -(v), (v), -(v), (v), -(v), (v), -(v) }

static const double KC_ONE[8]  __attribute__((aligned(64))) = SPLAT8(1.0);
static const double KC_HALF[8] __attribute__((aligned(64))) = SPLAT8(0.5);
/* sqrt(3)/2 in alternating form [s,-s,...]: multiplying the re/im-swapped
   difference by this yields -i*s*m in interleaved layout. */
static const double KC_KS[8] __attribute__((aligned(64)))
    = ALT8(8.66025403784438646764e-01);
/* W9^k = exp(-2*pi*i*k/9): C = Re, D = Im, for k = 1, 2, 4 */
static const double KC_C1[8] __attribute__((aligned(64))) = SPLAT8( 7.66044443118978013452e-01);
static const double KC_D1[8] __attribute__((aligned(64))) = SPLAT8(-6.42787609686539251896e-01);
static const double KC_C2[8] __attribute__((aligned(64))) = SPLAT8( 1.73648177666930414453e-01);
static const double KC_D2[8] __attribute__((aligned(64))) = SPLAT8(-9.84807753012208020316e-01);
static const double KC_C4[8] __attribute__((aligned(64))) = SPLAT8(-9.39692620785908316883e-01);
static const double KC_D4[8] __attribute__((aligned(64))) = SPLAT8(-3.42020143325668879442e-01);

/* nine 4-point stages, then four 9-point stages */
#define REP9(M) M(0) M(1) M(2) M(3) M(4) M(5) M(6) M(7) M(8)
#define REP4(M) M(0) M(1) M(2) M(3)

typedef void (*exec_fn)(const double *, double *, long, double *, double *);

struct fft3d_plan {
    long batch;
    exec_fn fn;
    double *plane;     /* 36*36 complex L1 scratch for the fused z+y phase */
    double *vol;       /* one-volume scratch, only for the NT (streaming) path */
    void *raw;
    void *rawvol;
};

/* exec_<variant>_<code>: code 0 = no prefetch, 1 = one 64-byte line ahead,
 * 2 = four lines ahead, 3 = the streaming path (phase 1 -> volume scratch,
 * phase 2 scratch -> out with non-temporal stores; no prefetch), 4 = the
 * streaming path with a one-line phase-2 prefetch on the scratch reads,
 * 5 = code 4 plus the cross-volume pipeline: while volume b's phase 2 is
 * store-drain-bound, volume b+1's `in` is prefetched into L2 (T1), paced
 * one 36-line granule per phase-2 tile so it exactly covers the volume,
 * 6 = code 4 plus the paced phase-1 input-stream prefetch ("pfin", from
 * L36_pfa's round-3 PFIN): a T1 cursor runs 32 KB ahead of the plane being
 * consumed, advancing 36*PW doubles per codelet call (2*36/PW calls/plane, so
 * exactly one plane of prefetches issues per plane processed), plus a small
 * cold-window pre-coverage of in[b+1] from phase 2 (3 lines/tile = 62 KB,
 * their PFNX) so the next volume never starts against raw DRAM latency.
 * Which set is eligible is decided by a working-set threshold in
 * fft3d_create(); the survivors are timed there. */
static void exec_0_0(const double *, double *, long, double *, double *);
static void exec_0_1(const double *, double *, long, double *, double *);
static void exec_0_2(const double *, double *, long, double *, double *);
static void exec_0_3(const double *, double *, long, double *, double *);
static void exec_0_4(const double *, double *, long, double *, double *);
static void exec_0_5(const double *, double *, long, double *, double *);
static void exec_0_6(const double *, double *, long, double *, double *);
static void exec_1_0(const double *, double *, long, double *, double *);
static void exec_1_1(const double *, double *, long, double *, double *);
static void exec_1_2(const double *, double *, long, double *, double *);
static void exec_1_3(const double *, double *, long, double *, double *);
static void exec_1_4(const double *, double *, long, double *, double *);
static void exec_1_5(const double *, double *, long, double *, double *);
static void exec_1_6(const double *, double *, long, double *, double *);
static void exec_2_0(const double *, double *, long, double *, double *);
static void exec_2_1(const double *, double *, long, double *, double *);
static void exec_2_2(const double *, double *, long, double *, double *);
static void exec_2_3(const double *, double *, long, double *, double *);
static void exec_2_4(const double *, double *, long, double *, double *);
static void exec_2_5(const double *, double *, long, double *, double *);
static void exec_2_6(const double *, double *, long, double *, double *);

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

const char *fft3d_name(void) { return "L36_mixedradix"; }

/* The chosen candidate is spliced in by fft3d_create() so the monitor can read
 * the tuner's verdict off the leaderboard / raw JSON (the panel_r2 VERDICT's
 * cross-cutting request; the mechanism is borrowed from L6_pfa). */
static char g_desc[224];

const char *fft3d_description(void)
{
    return g_desc[0] ? g_desc
                     : "row-column PFA 4x9 line codelet, batch-vectorised over "
                       "lines, 2 sweeps, AVX2/AVX-512 + store policy autotuned";
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
    if (posix_memalign(&pl, 64, (size_t)NPLANE * 2 * sizeof(double)) != 0 || !pl) {
        free(p);
        return NULL;
    }
    p->plane = (double *)pl;
    p->raw = pl;
    p->fn = exec_0_0;

    /* ---- store policy: a working-set threshold, not the timer, decides.
     * The tuning buffers below are at most 4 volumes, which always fit L3, so
     * a timing comparison there would systematically reject NT stores that win
     * at the real batch size (L6_pfa's record documents exactly that mis-pick;
     * L8_fusedaxes' threshold form is what is borrowed here).  NT is chosen
     * when in+out exceed 1.25x L3, i.e. when the RFO on `out` is a real DRAM
     * round trip rather than a cache hit. */
    int use_nt = 0;
    long l3 = -1;
    {
#ifdef _SC_LEVEL3_CACHE_SIZE
        l3 = sysconf(_SC_LEVEL3_CACHE_SIZE);
#endif
        if (l3 <= 0) l3 = 22l << 20;   /* the scoring node's 22 MiB */
        double foot = (double)batch * (double)NVOL * 16.0 * 2.0;
        use_nt = foot > 1.25 * (double)l3;
        /* diagnostic override, read once at plan time (execution stays
         * repeatable): FFT36_NT=0 forces the cached-store path, =1 forces the
         * streaming path */
        const char *ov = getenv("FFT36_NT");
        if (ov && (*ov == '0' || *ov == '1')) use_nt = *ov - '0';
    }
    if (use_nt) {
        void *pv = NULL;
        if (posix_memalign(&pv, 64, (size_t)NVOL * 2 * sizeof(double)) == 0 && pv) {
            p->vol = (double *)pv;
            p->rawvol = pv;
        } else {
            use_nt = 0;
        }
    }

    /* ---- self-tuning.  All of this is setup, hence excluded from the score.
     * Non-NT candidates are (kernel, prefetch) pairs; NT candidates are the
     * three kernels' streaming bodies.  Every candidate except the reference
     * must match exec_0_0's output to 1e-13 relative before it is eligible.
     *
     * Arena size: on the NT path the arena must actually STREAM on the machine
     * doing the tuning, or the ranking is systematically wrong for the real
     * run -- L36_pfa's round-2 record measured exactly this (a 16-volume arena
     * inside wallaby's L3 made NT look 12% slower while it was 40% faster at
     * the real size), and a fixed 32-volume arena reproduced it here on
     * wallaby's 60 MB L3 (the tuner dropped the xv candidate that a forced A/B
     * showed 13% faster).  So the arena is sized off THIS machine's L3: in+out
     * = 2.5x L3 per call, clamped to [32, 128] volumes and to the batch. */
    exec_fn cand[16];
    const char *cnm[16];
    int ncand = 0;

    long nt;
    if (use_nt) {
        long arena = (long)(2.5 * (double)l3 / ((double)NVOL * 32.0)) + 1;
        if (arena < 32)  arena = 32;
        if (arena > 128) arena = 128;
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
        exec_0_0(ti, o0, nt, p->plane, p->vol);

        int have_512 = 0, have_vl = 0;
#if defined(__x86_64__)
        __builtin_cpu_init();
        have_512 = __builtin_cpu_supports("avx512f");
        have_vl  = have_512 && __builtin_cpu_supports("avx512vl");
#endif
        /* diagnostic overrides for A/B runs: FFT36_XV=0 excludes the xv
         * candidates, =1 admits only them; FFT36_PFIN=0|1 likewise for the
         * paced input-prefetch candidates.  (Do not set both to 1.)  Read
         * once at plan time; execution stays repeatable. */
        int xvmode = -1, pfinmode = -1;
        {
            const char *xo = getenv("FFT36_XV");
            if (xo && (*xo == '0' || *xo == '1')) xvmode = *xo - '0';
            const char *po = getenv("FFT36_PFIN");
            if (po && (*po == '0' || *po == '1')) pfinmode = *po - '0';
        }

        exec_fn probe[12];
        const char *pnm[12];
        int nprobe = 0;
#define PROBE(fn, nm) do { probe[nprobe] = (fn); pnm[nprobe] = (nm); ++nprobe; } while (0)
        if (use_nt) {
            /* per kernel, simplest first: the hysteresis pick below prefers
             * an earlier candidate unless a later one wins by > 1% */
            if (xvmode != 1 && pfinmode != 1) {
                PROBE(exec_0_3, "v0-nt-pf0");
                PROBE(exec_0_4, "v0-nt-pf1");
            }
            if (pfinmode != 0 && xvmode != 1) PROBE(exec_0_6, "v0-nt-pf1-pfin");
            if (xvmode != 0 && pfinmode != 1) PROBE(exec_0_5, "v0-nt-pf1-xv");
            if (have_512) {
                if (xvmode != 1 && pfinmode != 1) {
                    PROBE(exec_1_3, "v1-nt-pf0");
                    PROBE(exec_1_4, "v1-nt-pf1");
                }
                if (pfinmode != 0 && xvmode != 1) PROBE(exec_1_6, "v1-nt-pf1-pfin");
                if (xvmode != 0 && pfinmode != 1) PROBE(exec_1_5, "v1-nt-pf1-xv");
            }
            if (have_vl) {
                if (xvmode != 1 && pfinmode != 1) {
                    PROBE(exec_2_3, "v2-nt-pf0");
                    PROBE(exec_2_4, "v2-nt-pf1");
                }
                if (pfinmode != 0 && xvmode != 1) PROBE(exec_2_6, "v2-nt-pf1-pfin");
                if (xvmode != 0 && pfinmode != 1) PROBE(exec_2_5, "v2-nt-pf1-xv");
            }
        } else {
            cand[ncand] = exec_0_0; cnm[ncand] = "v0-cached-pf0"; ++ncand;
            cand[ncand] = exec_0_1; cnm[ncand] = "v0-cached-pf1"; ++ncand;
            cand[ncand] = exec_0_2; cnm[ncand] = "v0-cached-pf4"; ++ncand;
            if (have_512) {
                PROBE(exec_1_0, "v1-cached-pf0");
                PROBE(exec_1_1, "v1-cached-pf1");
                PROBE(exec_1_2, "v1-cached-pf4");
            }
            if (have_vl) {
                PROBE(exec_2_0, "v2-cached-pf0");
                PROBE(exec_2_1, "v2-cached-pf1");
                PROBE(exec_2_2, "v2-cached-pf4");
            }
        }
#undef PROBE
        for (int k = 0; k < nprobe; ++k) {
            memset(ox, 0, nd * sizeof(double));
            probe[k](ti, ox, nt, p->plane, p->vol);
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
        if (ncand == 0) {           /* no NT candidate survived: fall back */
            cand[ncand] = exec_0_0; cnm[ncand] = "v0-cached-pf0-fallback"; ++ncand;
            cand[ncand] = exec_0_1; cnm[ncand] = "v0-cached-pf1-fallback"; ++ncand;
            cand[ncand] = exec_0_2; cnm[ncand] = "v0-cached-pf4-fallback"; ++ncand;
        }

        /* time every surviving candidate, several interleaved rounds, keep the
         * per-candidate minimum.  Small arenas get more reps and rounds: the
         * panel_r3 verdict measured this tuner flipping pf4/pf0/pf1 across
         * the node's three B=1 runs, a 3.9% spread caused by under-sampling,
         * not the machine. */
        double best[16];
        for (int k = 0; k < ncand; ++k) best[k] = 1e300;
        int reps = nt >= 16 ? 1 : (nt >= 4 ? 4 : 16);
        int rounds = nt >= 16 ? 4 : 6;
        for (int round = 0; round < rounds; ++round) {
            for (int k = 0; k < ncand; ++k) {
                cand[k](ti, ox, nt, p->plane, p->vol);        /* warm */
                double t0 = now_s();
                for (int r = 0; r < reps; ++r)
                    cand[k](ti, ox, nt, p->plane, p->vol);
                double dt = now_s() - t0;
                if (dt < best[k]) best[k] = dt;
            }
        }
        /* hysteresis pick: candidates are listed simplest-first per kernel,
         * so a later (more speculative) candidate must beat the incumbent by
         * more than 1% to be installed.  Near-ties go to the simpler code. */
        int bk = 0;
        for (int k = 1; k < ncand; ++k)
            if (best[k] < 0.99 * best[bk]) bk = k;
        p->fn = cand[bk];
        snprintf(g_desc, sizeof g_desc,
                 "PFA 4x9 2-sweep, lanes=lines; pick=%s (B=%d, arena=%ld vol, "
                 "ntpolicy=%d, %d cand)", cnm[bk], batch, nt, use_nt, ncand);
    }
    free(ti); free(o0); free(ox);
    return p;
}

void fft3d_execute(fft3d_plan *plan, const double _Complex *in, double _Complex *out)
{
    plan->fn((const double *)in, (double *)out, plan->batch, plan->plane, plan->vol);
}

void fft3d_destroy(fft3d_plan *plan)
{
    if (!plan) return;
    free(plan->raw);
    free(plan->rawvol);
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
#define FNP5(n) XCAT(XCAT(n##_, VAR), _5)
#define FNP6(n) XCAT(XCAT(n##_, VAR), _6)

#if VAR == 1
/* ---- 512-bit: 4 complex lanes per zmm, 32 registers ---- */
#pragma GCC push_options
#pragma GCC target("avx512f")
#define PW 4
#define VD __m512d
#define VLOAD(p)          _mm512_loadu_pd(p)
#define VSTORE(p, v)      _mm512_storeu_pd((p), (v))
#define VADD(a, b)        _mm512_add_pd((a), (b))
#define VSUB(a, b)        _mm512_sub_pd((a), (b))
#define VMUL(a, b)        _mm512_mul_pd((a), (b))
#define VFMADD(a, b, c)   _mm512_fmadd_pd((a), (b), (c))
#define VFNMADD(a, b, c)  _mm512_fnmadd_pd((a), (b), (c))
#define VFMADDSUB(a,b,c)  _mm512_fmaddsub_pd((a), (b), (c))
#define VFMSUBADD(a,b,c)  _mm512_fmsubadd_pd((a), (b), (c))
#define VSWAP(a)          _mm512_permute_pd((a), 0x55)
#define VSTREAM(p, v)     _mm512_stream_pd((p), (v))
#define PWLIST(M)         M(0) M(1) M(2) M(3)
/* 4x4 transpose of 128-bit complex lanes (an involution, so the same macro
   serves for gathering into lanes and scattering back out) */
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
   VAR 2 = EVEX/AVX-512VL (32 registers, no 512-bit frequency licence) ---- */
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
#define VADD(a, b)        _mm256_add_pd((a), (b))
#define VSUB(a, b)        _mm256_sub_pd((a), (b))
#define VMUL(a, b)        _mm256_mul_pd((a), (b))
#define VFMADD(a, b, c)   _mm256_fmadd_pd((a), (b), (c))
#define VFNMADD(a, b, c)  _mm256_fnmadd_pd((a), (b), (c))
#define VFMADDSUB(a,b,c)  _mm256_fmaddsub_pd((a), (b), (c))
#define VFMSUBADD(a,b,c)  _mm256_fmsubadd_pd((a), (b), (c))
#define VSWAP(a)          _mm256_permute_pd((a), 0x5)
#define VSTREAM(p, v)     _mm256_stream_pd((p), (v))
#define PWLIST(M)         M(0) M(1)
/* 2x2 transpose of 128-bit complex lanes; involution */
#define TRANSP(A, B) do {                                        \
    VD z0 = _mm256_permute2f128_pd((A)[0], (A)[1], 0x20);        \
    VD z1 = _mm256_permute2f128_pd((A)[0], (A)[1], 0x31);        \
    (B)[0] = z0; (B)[1] = z1;                                    \
} while (0)
#endif

#define C_ONE  VLOAD(KC_ONE)
#define C_HALF VLOAD(KC_HALF)
#define C_KS   VLOAD(KC_KS)

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

/* stage 1: nine 4-point DFTs over n1, PFA input map n = (9*n1 + 4*n2) mod 36.
 *   t1 = x0-x2, t3 = x1-x3, y1 = t1 - i*t3, y3 = t1 + i*t3, and -i*t3 =
 *   (t3.im, -t3.re) folds into the add as fmsubadd / fmaddsub. */
#define ST1(N2) {                                              \
    VD x0 = LSRC((9 * 0 + 4 * (N2)) % 36);                     \
    VD x1 = LSRC((9 * 1 + 4 * (N2)) % 36);                     \
    VD x2 = LSRC((9 * 2 + 4 * (N2)) % 36);                     \
    VD x3 = LSRC((9 * 3 + 4 * (N2)) % 36);                     \
    VD t0 = VADD(x0, x2), t1 = VSUB(x0, x2);                   \
    VD t2 = VADD(x1, x3), t3 = VSUB(x1, x3);                   \
    VD sw = VSWAP(t3);                                         \
    T[0 * 9 + (N2)] = VADD(t0, t2);                            \
    T[2 * 9 + (N2)] = VSUB(t0, t2);                            \
    T[1 * 9 + (N2)] = VFMSUBADD(t1, C_ONE, sw);                \
    T[3 * 9 + (N2)] = VFMADDSUB(t1, C_ONE, sw);                \
}

/* stage 2: four 9-point DFTs over n2 (Cooley-Tukey 3x3), PFA output map
 * k = (9*k1 + 28*k2) mod 36.  Note 9*k1 + 28*k2 == k1 (mod 4), so DFT9 number
 * k1 lands entirely on output slots congruent to k1 mod 4. */
#define ST2(K1) {                                              \
    VD u0 = T[(K1) * 9 + 0], u1 = T[(K1) * 9 + 1];             \
    VD u2 = T[(K1) * 9 + 2], u3 = T[(K1) * 9 + 3];             \
    VD u4 = T[(K1) * 9 + 4], u5 = T[(K1) * 9 + 5];             \
    VD u6 = T[(K1) * 9 + 6], u7 = T[(K1) * 9 + 7];             \
    VD u8 = T[(K1) * 9 + 8];                                   \
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
    SDST((9 * (K1) + 28 * 0) % 36, v0);                        \
    SDST((9 * (K1) + 28 * 1) % 36, v1);                        \
    SDST((9 * (K1) + 28 * 2) % 36, v2);                        \
    SDST((9 * (K1) + 28 * 3) % 36, v3);                        \
    SDST((9 * (K1) + 28 * 4) % 36, v4);                        \
    SDST((9 * (K1) + 28 * 5) % 36, v5);                        \
    SDST((9 * (K1) + 28 * 6) % 36, v6);                        \
    SDST((9 * (K1) + 28 * 7) % 36, v7);                        \
    SDST((9 * (K1) + 28 * 8) % 36, v8);                        \
}

/* 36-point PFA line transform, y axis: line stride 36 complex = 72 doubles.
 * The stride is a compile-time literal, so every access is a displacement off
 * a single base register and there is no address arithmetic at all. */
static inline __attribute__((always_inline))
void FN(dft36_y)(const double *src, double *dst)
{
    VD T[36];
#define LSRC(i)     VLOAD(src + (long)(i) * (36 * 2))
#define SDST(i, v)  VSTORE(dst + (long)(i) * (36 * 2), (v))
    REP9(ST1)
    REP4(ST2)
#undef LSRC
#undef SDST
}

/* 36-point PFA line transform, x axis, in place: stride 1296 complex. */
static inline __attribute__((always_inline))
void FN(dft36_x)(double *base)
{
    VD T[36];
#define LSRC(i)     VLOAD(base + (long)(i) * (NPLANE * 2))
#define SDST(i, v)  VSTORE(base + (long)(i) * (NPLANE * 2), (v))
    REP9(ST1)
    REP4(ST2)
#undef LSRC
#undef SDST
}

#if PW == 4
/* 36-point x-line, out of place, non-temporal stores.  Every SDST target is a
 * full 64-byte line at a 64-byte-aligned address ((y*36 + zb*4)*16 with
 * 64 | y*576 and 64 | zb*64, plus k*20736 with 64 | 20736), so each store is
 * one complete write-combining line and no RFO is ever issued on `out`. */
static inline __attribute__((always_inline))
void FN(dft36_xnt)(const double *src, double *dst)
{
    VD T[36];
#define LSRC(i)     VLOAD(src + (long)(i) * (NPLANE * 2))
#define SDST(i, v)  VSTREAM(dst + (long)(i) * (NPLANE * 2), (v))
    REP9(ST1)
    REP4(ST2)
#undef LSRC
#undef SDST
}
#else
/* 256-bit NT stores are only half a cache line, and phase 2 touches 36
 * different lines per tile -- far more than the ~10 write-combining buffers --
 * so partial-line evictions would kill it.  Instead two adjacent z-tiles are
 * transformed into a 36x2-vector stack stage, then flushed line by line with
 * two back-to-back 32-byte NT stores that the WC buffer merges into one full
 * 64-byte line write. */
static inline __attribute__((always_inline))
void FN(dft36_xst)(const double *src, VD *stg)
{
    VD T[36];
#define LSRC(i)     VLOAD(src + (long)(i) * (NPLANE * 2))
#define SDST(i, v)  stg[(i) * 2] = (v)
    REP9(ST1)
    REP4(ST2)
#undef LSRC
#undef SDST
}
#endif

/* 36-point PFA line transform, vector array in, vector array out (z axis) */
static inline __attribute__((always_inline))
void FN(dft36_v)(const VD *X, VD *Y)
{
    VD T[36];
#define LSRC(i)     X[(i)]
#define SDST(i, v)  Y[(i)] = (v)
    REP9(ST1)
    REP4(ST2)
#undef LSRC
#undef SDST
}

static inline __attribute__((always_inline))
void FN(body)(const double *in, double *out, long batch, double *plane,
              double *vol, const int pf, const int nt, const int xv,
              const int pfin)
{
    /* pfin: paced T1 prefetch of the phase-1 input stream (L36_pfa r3's
     * PFIN, attributed).  A cursor runs PFIN_D doubles = 32 KB ahead of the
     * plane being consumed; each of the 2*(36/PW) codelet calls per plane
     * issues PFIN_L line-prefetches and advances, so exactly one plane of
     * prefetches issues per plane processed and the read stream stays busy
     * through the y-subloop's compute-only stretch.  The yb subloop consumes
     * `in` at 2x the cursor rate, so mid-plane the cursor falls up to
     * 10.4 KB behind -- 32 KB of distance absorbs that with margin.  At each
     * volume boundary the cursor is naturally 32 KB into in[b+1]. */
#define PFIN_D 4096                    /* cursor distance, doubles = 32 KB  */
#define PFIN_L (36 * PW / 8)           /* lines per codelet call: 18 / 9    */
    const double *pfp = 0, *pfend = 0;
    if (pfin) {
        pfend = in + batch * (long)NVOL * 2;
        pfp   = in + PFIN_D;
        if (pfp > pfend) pfp = pfend;
    }
    for (long b = 0; b < batch; ++b) {
        const double *vin  = in  + b * (long)NVOL * 2;
        double       *vout = out + b * (long)NVOL * 2;
        /* nt: phase 1 lands in the one-volume scratch (cache-resident and
         * reused every volume, so it costs no DRAM traffic after the first),
         * and phase 2 streams scratch -> out.  DRAM traffic per volume drops
         * from read(in) + RFO(out) + writeback(out) to read(in) + write(out),
         * the compulsory minimum. */
        double *vmid = nt ? vol : vout;

        /* -------- phase 1: for each x-plane, z-lines then y-lines -------- */
        for (long x = 0; x < LSIDE; ++x) {
            const double *pin  = vin  + x * (long)NPLANE * 2;
            double       *pout = vmid + x * (long)NPLANE * 2;

            for (long yb = 0; yb < LSIDE / PW; ++yb) {
                if (pfin) {
                    long npl = (pfend - pfp) / 8;
                    if (npl > PFIN_L) npl = PFIN_L;
                    for (long i = 0; i < npl; ++i)
                        _mm_prefetch((const char *)(pfp + i * 8), _MM_HINT_T1);
                    pfp += npl * 8;
                }
                VD Xv[36], Yv[36];
                for (long g = 0; g < LSIDE / PW; ++g) {
                    VD A[PW], B[PW];
                    const double *q = pin + (yb * PW * 36 + g * PW) * 2;
#define LDR(j) A[j] = VLOAD(q + (j) * 36 * 2);
                    PWLIST(LDR)
#undef LDR
                    TRANSP(A, B);
#define PUT(j) Xv[g * PW + (j)] = B[j];
                    PWLIST(PUT)
#undef PUT
                }
                FN(dft36_v)(Xv, Yv);
                for (long g = 0; g < LSIDE / PW; ++g) {
                    VD A[PW], B[PW];
                    double *q = plane + (yb * PW * 36 + g * PW) * 2;
#define GET(j) A[j] = Yv[g * PW + (j)];
                    PWLIST(GET)
#undef GET
                    TRANSP(A, B);
#define PST(j) VSTORE(q + (j) * 36 * 2, B[j]);
                    PWLIST(PST)
#undef PST
                }
            }
            for (long zb = 0; zb < LSIDE / PW; ++zb) {
                if (pfin) {
                    long npl = (pfend - pfp) / 8;
                    if (npl > PFIN_L) npl = PFIN_L;
                    for (long i = 0; i < npl; ++i)
                        _mm_prefetch((const char *)(pfp + i * 8), _MM_HINT_T1);
                    pfp += npl * 8;
                }
                FN(dft36_y)(plane + zb * PW * 2, pout + zb * PW * 2);
            }
        }

        /* -------- phase 2: x-lines --------------------------------------- */
        if (!nt) {
            /* in place in `out` */
            for (long y = 0; y < LSIDE; ++y) {
                for (long zb = 0; zb < LSIDE / PW; ++zb) {
                    double *base = vout + (y * 36 + zb * PW) * 2;
                    if (pf) {
                        /* the 36 x-streams each advance by 64 bytes per zb:
                         * more streams than the L2 prefetcher tracks, so poke
                         * them by hand, `pf` cache lines ahead */
                        for (int i = 0; i < 36; ++i)
                            _mm_prefetch((const char *)(base + i * (NPLANE * 2) + pf * 8),
                                         _MM_HINT_T0);
                    }
                    FN(dft36_x)(base);
                }
            }
        } else {
            /* scratch -> out, streaming stores.  `vol` may have been partly
             * evicted to L3 by this volume's phase-1 `in` stream (729 KiB of
             * scratch + 729 KiB of stream against a 1 MiB L2 on the scoring
             * node), so the same 36-stream hand prefetch is offered here too,
             * as a tuner candidate.
             *
             * xv = the cross-volume software pipeline (the panel_r2 VERDICT's
             * quantified L36 target): phase 2 is NT-store-drain-bound and its
             * read side is idle, so volume b+1's `in` is prefetched here with
             * T1 (into L2, sparing L1), paced at one 36-line granule per tile
             * -- 324 tiles x 36 lines = 11664 lines = exactly one volume, and
             * the pace matches the rate at which phase 2 retires scratch
             * lines, so the prefetched stream replaces dead data.  Phase 1 of
             * volume b+1 then finds its input L2-resident instead of paying a
             * serial DRAM read. */
            const double *nin = (xv && b + 1 < batch)
                                    ? in + (b + 1) * (long)NVOL * 2
                                    : (const double *)0;
            /* pfin cold-window pre-coverage (L36_pfa r3's PFNX): the paced
             * cursor leaves only the first 32 KB of in[b+1] exposed to
             * phase-2 scratch-read eviction; 3 lines per tile x 324 tiles
             * = 62 KB re-covers it from the store-drain-bound phase 2. */
            const double *ncw = (pfin && b + 1 < batch)
                                    ? in + (b + 1) * (long)NVOL * 2
                                    : (const double *)0;
#if PW == 4
            for (long y = 0; y < LSIDE; ++y)
                for (long zb = 0; zb < LSIDE / PW; ++zb) {
                    long off = (y * 36 + zb * PW) * 2;
                    if (pf) {
                        const double *s = vol + off;
                        for (int i = 0; i < 36; ++i)
                            _mm_prefetch((const char *)(s + i * (NPLANE * 2) + pf * 8),
                                         _MM_HINT_T0);
                    }
                    if (nin) {
                        const double *q = nin + (y * 9 + zb) * 288;
                        for (int i = 0; i < 36; ++i)
                            _mm_prefetch((const char *)(q + i * 8), _MM_HINT_T1);
                    }
                    if (ncw) {
                        const double *q = ncw + (y * 9 + zb) * 24;
                        _mm_prefetch((const char *)(q),      _MM_HINT_T1);
                        _mm_prefetch((const char *)(q + 8),  _MM_HINT_T1);
                        _mm_prefetch((const char *)(q + 16), _MM_HINT_T1);
                    }
                    FN(dft36_xnt)(vol + off, vout + off);
                }
#else
            for (long y = 0; y < LSIDE; ++y)
                for (long zp = 0; zp < 9; ++zp) {
                    VD stage[72] __attribute__((aligned(64)));
                    long off = (y * 36 + zp * 4) * 2;
                    if (pf) {
                        const double *s = vol + off;
                        for (int i = 0; i < 36; ++i)
                            _mm_prefetch((const char *)(s + i * (NPLANE * 2) + pf * 8),
                                         _MM_HINT_T0);
                    }
                    if (nin) {
                        const double *q = nin + (y * 9 + zp) * 288;
                        for (int i = 0; i < 36; ++i)
                            _mm_prefetch((const char *)(q + i * 8), _MM_HINT_T1);
                    }
                    if (ncw) {
                        const double *q = ncw + (y * 9 + zp) * 24;
                        _mm_prefetch((const char *)(q),      _MM_HINT_T1);
                        _mm_prefetch((const char *)(q + 8),  _MM_HINT_T1);
                        _mm_prefetch((const char *)(q + 16), _MM_HINT_T1);
                    }
                    FN(dft36_xst)(vol + off,     stage);
                    FN(dft36_xst)(vol + off + 4, stage + 1);
                    for (int i = 0; i < 36; ++i) {
                        double *d = vout + off + (long)i * (NPLANE * 2);
                        VSTREAM(d,     stage[2 * i]);
                        VSTREAM(d + 4, stage[2 * i + 1]);
                    }
                }
#endif
        }
    }
    if (nt) _mm_sfence();
#undef PFIN_D
#undef PFIN_L
}

static void FN(exec)(const double *in, double *out, long batch, double *plane,
                     double *vol)
{
    FN(body)(in, out, batch, plane, vol, 0, 0, 0, 0);
}

static void FNP1(exec)(const double *in, double *out, long batch, double *plane,
                       double *vol)
{
    FN(body)(in, out, batch, plane, vol, 1, 0, 0, 0);
}

static void FNP2(exec)(const double *in, double *out, long batch, double *plane,
                       double *vol)
{
    FN(body)(in, out, batch, plane, vol, 4, 0, 0, 0);
}

static void FNP3(exec)(const double *in, double *out, long batch, double *plane,
                       double *vol)
{
    FN(body)(in, out, batch, plane, vol, 0, 1, 0, 0);
}

static void FNP4(exec)(const double *in, double *out, long batch, double *plane,
                       double *vol)
{
    FN(body)(in, out, batch, plane, vol, 1, 1, 0, 0);
}

static void FNP5(exec)(const double *in, double *out, long batch, double *plane,
                       double *vol)
{
    FN(body)(in, out, batch, plane, vol, 1, 1, 1, 0);
}

static void FNP6(exec)(const double *in, double *out, long batch, double *plane,
                       double *vol)
{
    FN(body)(in, out, batch, plane, vol, 1, 1, 0, 1);
}

#pragma GCC pop_options

#undef XCAT2
#undef XCAT
#undef FN
#undef FNP1
#undef FNP2
#undef FNP3
#undef FNP4
#undef FNP5
#undef FNP6
#undef PW
#undef VD
#undef VLOAD
#undef VSTORE
#undef VADD
#undef VSUB
#undef VMUL
#undef VFMADD
#undef VFNMADD
#undef VFMADDSUB
#undef VFMSUBADD
#undef VSWAP
#undef VSTREAM
#undef PWLIST
#undef TRANSP
#undef C_ONE
#undef C_HALF
#undef C_KS
#undef DFT3M
#undef CMULM
#undef ST1
#undef ST2

#endif /* VAR */
