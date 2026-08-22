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
 *   MEMORY MECHANISMS (all tournament-gated candidates, never defaults):
 *   * pf:   phase 2 prefetches its 36 x-streams by hand (`pf` lines ahead) --
 *           more streams than the L2 streamer tracks.
 *   * pfin: paced T1 prefetch of the phase-1 `in` read stream, 32 KB ahead
 *           (L36_pfa r3's PFIN, attributed), plus a small cold-window
 *           pre-coverage of in[b+1] from phase 2 (their PFNX).
 *   * pfw:  paced WRITE-INTENT prefetch (`prefetchw` via
 *           __builtin_prefetch(p,1,3)) over phase 1's cold-`out` store stream,
 *           one plane (2592 doubles) ahead, same pacing as pfin.  At streaming
 *           batch sizes every one of the 11664 output lines per volume costs a
 *           demand RFO from DRAM that nothing overlaps; prefetchw acquires the
 *           line exclusive ahead of the store while keeping the normal-store
 *           shape this node prefers over NT.  Borrowed from L36_pfa round
 *           panel_r5 (their pf=2 / PFWMID, node-selected at B=32 and B=256;
 *           in-arena inplace-pf2 90.5 vs inplace-pf1 156.6 at B=256), which in
 *           turn adopted it from L6_unrolled r3 (fused_pfw).
 *   * sp2:  two independent 36-point line transforms interleaved at source
 *           level in the y and x passes, so the DFT9 latency chains of one
 *           transform overlap the independent DFT4s of the other (the
 *           five-rounds-untried B=1 lever; panel_r5 VERDICT section 6).
 *           Measured a LOSS on wallaby (+7.7%) and killed independently by
 *           L36_pencilfused r6's PFA36X2 (+1..3% pw4, -17% pw2); kept only as
 *           a gated candidate so the node can vote once.
 *   * nta:  constant-lead (4 KB) prefetchNTA over phase 1's `in` read stream,
 *           paced at exactly the consumption rate inside the z-subloop (the
 *           only place `in` is read), so the single-use input stream fills L1
 *           and BYPASSES L2.  On the scoring node (1 MB L2 vs in+out = 1.5 MB
 *           at B=1) the in-read otherwise evicts `out` mid-execute and every
 *           phase-1 store RFOs from L3, every phase-2 read comes back from L3;
 *           with the read stream kept out of L2, `out` stays L2-modified
 *           across executes and the L3 traffic collapses to the compulsory
 *           in-read.  Borrowed from L36_pfa round panel_r6 (their pf=4 design
 *           and their 512-double lead, measured best of {128,256,512} there),
 *           which round was never node-timed; it loses on wallaby (2 MB L2
 *           holds in+out at B=1, so there is nothing to fix and the L1
 *           quick-evict tax shows raw), so it ships as a gated candidate and
 *           the node's own nv=1 steady-state tournament makes the call.
 *
 *   The non-temporal-store path and the cross-volume (xv) full-volume prefetch
 *   that earlier rounds carried were RETIRED this round: the node rejected NT
 *   in every entry's tournament for four consecutive rounds (panel_r5 VERDICT
 *   section on store policy), my own r5 node picks were cached at both
 *   streaming cells with NT in the pool, and xv was dominated by pfin on both
 *   machines since r4.  See strategies/L36_mixedradix.md round panel_r6.
 *
 * OPERATION COUNT (per 36-point line, as vector instructions over PW lanes)
 *       DFT4  :  8 FMA-port ops + 1 shuffle       x9   =  72 + 9
 *       DFT3  :  6 FMA-port ops + 1 shuffle       x24  = 144 + 24
 *       CMUL  :  2 FMA-port ops + 1 shuffle       x16  =  32 + 16
 *       total :                        248 FMA-port ops + 49 shuffles / PW lines
 *   In real flops: 9*20 + 4*(6*18 + 4*6) = 180 + 528 = 708 flops per 36-point
 *   line; per volume 3 * 36^2 * 708 = 2,752,704 flops in 3888 line transforms.
 *
 * ASSUMPTIONS
 *   * L == 36 only; fft3d_supports() refuses everything else.
 *   * `in` and `out` are 64-byte aligned and distinct (the driver guarantees
 *     both).  Every vector access made here is at a multiple of 64 bytes from
 *     the volume base -- (x*1296 + y*36 + zb*PW)*16 with 4 | 1296, 4 | 36.
 *   * `out` doubles as the working buffer between phase 1 and phase 2.  `in` is
 *     never written.
 *   * Three kernels are compiled: V0 = AVX2+FMA (2 complex lanes, 16 ymm),
 *     V1 = AVX-512F (4 complex lanes, 32 zmm), V2 = AVX-512VL (2 complex lanes
 *     but 32 registers).  fft3d_create() checks CPU support, verifies each
 *     AVX-512 kernel numerically against the AVX2 one, then times all
 *     surviving (kernel x mechanism) combinations and keeps the fastest.
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

/* pfw write-intent cursor distance (doubles): default one x-plane = 20.25 KB,
 * L36_pfa's pacing arithmetic.  Overridable with -DFFT36_PFW_DIST=... so the
 * monitor's 1296/2592/5184 sweep needs no source edit. */
#ifndef FFT36_PFW_DIST
#define FFT36_PFW_DIST (NPLANE * 2)
#endif
/* nta read-cursor lead (doubles): 512 = 4 KB, the best of {128,256,512} in
 * L36_pfa r6's wallaby sweep.  Constant lead: issued at consumption rate in
 * the z-subloop only, so it never swings the way the pfin T1 cursor does
 * (fine for L2-bound prefetch, fatal for L1-resident NTA lines). */
#ifndef FFT36_NTA_DIST
#define FFT36_NTA_DIST 512
#endif

typedef void (*exec_fn)(const double *, double *, long, double *);

struct fft3d_plan {
    long batch;
    exec_fn fn;
    double *plane;     /* 36*36 complex L1 scratch for the fused z+y phase */
    void *raw;
};

/* exec_<variant>_<code>:
 *   0 = cached, no prefetch                        (pf0)
 *   1 = cached, phase-2 streams 1 line ahead       (pf1)
 *   2 = cached, phase-2 streams 4 lines ahead      (pf4)
 *   3 = code 1 + paced phase-1 input prefetch      (pf1-pfin)
 *   4 = code 3 + paced phase-1 prefetchw on out    (pf1-pfin-pfw)
 *   5 = cached, no prefetch, software-pipelined
 *       pairs of line transforms in y and x passes (pf0-sp2)
 *   6 = code 1 + constant-lead NTA prefetch of the
 *       phase-1 input stream, no pfin/PFNX          (pf1-nta)
 */
static void exec_0_0(const double *, double *, long, double *);
static void exec_0_1(const double *, double *, long, double *);
static void exec_0_2(const double *, double *, long, double *);
static void exec_0_3(const double *, double *, long, double *);
static void exec_0_4(const double *, double *, long, double *);
static void exec_0_5(const double *, double *, long, double *);
static void exec_0_6(const double *, double *, long, double *);
static void exec_1_0(const double *, double *, long, double *);
static void exec_1_1(const double *, double *, long, double *);
static void exec_1_2(const double *, double *, long, double *);
static void exec_1_3(const double *, double *, long, double *);
static void exec_1_4(const double *, double *, long, double *);
static void exec_1_5(const double *, double *, long, double *);
static void exec_1_6(const double *, double *, long, double *);
static void exec_2_0(const double *, double *, long, double *);
static void exec_2_1(const double *, double *, long, double *);
static void exec_2_2(const double *, double *, long, double *);
static void exec_2_3(const double *, double *, long, double *);
static void exec_2_4(const double *, double *, long, double *);
static void exec_2_5(const double *, double *, long, double *);
static void exec_2_6(const double *, double *, long, double *);

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
 * the tuner's verdict off the leaderboard / raw JSON. */
static char g_desc[224];

const char *fft3d_description(void)
{
    return g_desc[0] ? g_desc
                     : "row-column PFA 4x9 line codelet, batch-vectorised over "
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
    if (posix_memalign(&pl, 64, (size_t)NPLANE * 2 * sizeof(double)) != 0 || !pl) {
        free(p);
        return NULL;
    }
    p->plane = (double *)pl;
    p->raw = pl;
    p->fn = exec_0_0;

    /* ---- regime: does the batch stream through this machine's LLC?  This
     * decides only which candidates are IN PLAY (pfw is offered only where
     * `out` is genuinely cold -- L36_pfa and L6_unrolled both measured
     * prefetchw at +11..17% on cache-resident volumes), and how the tuning
     * arena is sized.  Store policy itself is no longer a question: the node
     * rejected NT stores in every tournament for four consecutive rounds, and
     * my own r5 node picks were cached at both streaming cells with the full
     * NT candidate set in the pool.  The NT/xv machinery is retired. */
    long l3 = -1;
    {
#ifdef _SC_LEVEL3_CACHE_SIZE
        l3 = sysconf(_SC_LEVEL3_CACHE_SIZE);
#endif
        if (l3 <= 0) l3 = 22l << 20;   /* the scoring node's 22 MiB */
    }
    double foot = (double)batch * (double)NVOL * 16.0 * 2.0;
    int streaming = foot > 1.25 * (double)l3;

    /* ---- self-tuning.  All of this is setup, hence excluded from the score.
     * Every candidate except the reference must match exec_0_0's output to
     * 1e-13 relative before it is eligible.
     *
     * Arena size: in the streaming regime the arena must actually STREAM on
     * the machine doing the tuning, or the ranking is systematically wrong
     * for the real run (L36_pfa's round-2 lesson, reproduced here in r3).
     * So the arena is sized off THIS machine's L3: in+out = 2.5x L3 per
     * call, clamped to [32, 128] volumes and to the batch. */
    exec_fn cand[32];
    const char *cnm[32];
    int ncand = 0;

    long nt;
    if (streaming) {
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
        exec_0_0(ti, o0, nt, p->plane);

        int have_512 = 0, have_vl = 0;
#if defined(__x86_64__)
        __builtin_cpu_init();
        have_512 = __builtin_cpu_supports("avx512f");
        have_vl  = have_512 && __builtin_cpu_supports("avx512vl");
#endif
        /* diagnostic overrides for A/B runs, read once at plan time so
         * execution stays repeatable: FFT36_PFIN=0 drops the paced-input-
         * prefetch candidates (including pfw, which builds on it), =1 admits
         * only them; FFT36_PFW=0|1 likewise for the write-intent-prefetch
         * candidates; FFT36_SP2=0|1 for the software-pipelined-pair ones;
         * FFT36_NTA=0|1 for the constant-lead NTA input-read ones.
         * (FFT36_NT / FFT36_XV from rounds r2-r5 are gone with the NT path.) */
        int pfinmode = -1, pfwmode = -1, sp2mode = -1, ntamode = -1;
        {
            const char *po = getenv("FFT36_PFIN");
            if (po && (*po == '0' || *po == '1')) pfinmode = *po - '0';
            const char *wo = getenv("FFT36_PFW");
            if (wo && (*wo == '0' || *wo == '1')) pfwmode = *wo - '0';
            const char *so = getenv("FFT36_SP2");
            if (so && (*so == '0' || *so == '1')) sp2mode = *so - '0';
            const char *no = getenv("FFT36_NTA");
            if (no && (*no == '0' || *no == '1')) ntamode = *no - '0';
        }
        int in_plain = (pfinmode != 1) && (pfwmode != 1) && (sp2mode != 1) && (ntamode != 1);
        int in_pfin  = (pfinmode != 0) && (pfwmode != 1) && (sp2mode != 1) && (ntamode != 1);
        int in_pfw   = (pfinmode != 0) && (pfwmode != 0) && (sp2mode != 1) && (ntamode != 1);
        int in_sp2   = (sp2mode != 0)  && (pfinmode != 1) && (pfwmode != 1) && (ntamode != 1);
        int in_nta   = (ntamode != 0)  && (pfinmode != 1) && (pfwmode != 1) && (sp2mode != 1);

        exec_fn probe[32];
        const char *pnm[32];
        int nprobe = 0;
#define PROBE(fn, nm) do { probe[nprobe] = (fn); pnm[nprobe] = (nm); ++nprobe; } while (0)
        /* Candidate order = hysteresis order: within a kernel, simplest
         * mechanism first (a speculative mechanism must beat the incumbent by
         * > 3% to be installed); across kernels, V1 FIRST -- it has won every
         * node cell in every round since r1, so V0/V2 must now clear the 3%
         * bar to displace it.  (A noisy-window wallaby plan mis-picked
         * v2-cached-pf4 at +40% this round with V0-first ordering; V1-first
         * makes that class of mis-pick need a 3% fake win instead of a tie.) */
        if (streaming) {
            /* streaming pool: pfw is offered only here -- L36_pfa and
             * L6_unrolled both measured prefetchw at +11..17% on
             * cache-resident volumes.  nta is NOT offered here (L36_pfa
             * measured NTA on DRAM-rate streams at +14% over no prefetch at
             * all, wallaby B=32) except under an explicit FFT36_NTA=1 force,
             * so the diagnostic never silently falls back. */
            if (have_512) {
                if (in_plain) PROBE(exec_1_1, "v1-cached-pf1");
                if (in_pfin)  PROBE(exec_1_3, "v1-cached-pf1-pfin");
                if (in_pfw)   PROBE(exec_1_4, "v1-cached-pf1-pfin-pfw");
                if (ntamode == 1) PROBE(exec_1_6, "v1-cached-pf1-nta");
            }
            if (have_vl) {
                if (in_plain) PROBE(exec_2_1, "v2-cached-pf1");
                if (in_pfin)  PROBE(exec_2_3, "v2-cached-pf1-pfin");
                if (in_pfw)   PROBE(exec_2_4, "v2-cached-pf1-pfin-pfw");
                if (ntamode == 1) PROBE(exec_2_6, "v2-cached-pf1-nta");
            }
            if (in_plain) PROBE(exec_0_1, "v0-cached-pf1");
            if (in_pfin)  PROBE(exec_0_3, "v0-cached-pf1-pfin");
            if (in_pfw)   PROBE(exec_0_4, "v0-cached-pf1-pfin-pfw");
            if (ntamode == 1) PROBE(exec_0_6, "v0-cached-pf1-nta");
        } else {
            /* cached pool.  pfw joins it at batch >= 2 (L36_pfa r6: pf=2 beat
             * pf=0 by 8% at B=4 in a quiet window -- at B>=2 `out` volumes
             * cycle through L2/L3 so the store stream's RFO is exposed even
             * though the batch does not stream; at B=1 out is steady-state
             * resident and prefetchw is pure tax, +11..17% measured). */
            if (have_512) {
                if (in_plain) {
                    PROBE(exec_1_0, "v1-cached-pf0");
                    PROBE(exec_1_1, "v1-cached-pf1");
                    PROBE(exec_1_2, "v1-cached-pf4");
                }
                if (in_pfin) PROBE(exec_1_3, "v1-cached-pf1-pfin");
                if (in_nta)  PROBE(exec_1_6, "v1-cached-pf1-nta");
                if (in_pfw && batch >= 2) PROBE(exec_1_4, "v1-cached-pf1-pfin-pfw");
                if (in_sp2)  PROBE(exec_1_5, "v1-cached-pf0-sp2");
            }
            if (have_vl) {
                if (in_plain) {
                    PROBE(exec_2_0, "v2-cached-pf0");
                    PROBE(exec_2_1, "v2-cached-pf1");
                    PROBE(exec_2_2, "v2-cached-pf4");
                }
                if (in_pfin) PROBE(exec_2_3, "v2-cached-pf1-pfin");
                if (in_nta)  PROBE(exec_2_6, "v2-cached-pf1-nta");
                if (in_pfw && batch >= 2) PROBE(exec_2_4, "v2-cached-pf1-pfin-pfw");
                if (in_sp2)  PROBE(exec_2_5, "v2-cached-pf0-sp2");
            }
            if (in_plain) {
                PROBE(exec_0_0, "v0-cached-pf0");
                PROBE(exec_0_1, "v0-cached-pf1");
                PROBE(exec_0_2, "v0-cached-pf4");
            }
            if (in_pfin) PROBE(exec_0_3, "v0-cached-pf1-pfin");
            if (in_nta)  PROBE(exec_0_6, "v0-cached-pf1-nta");
            if (in_pfw && batch >= 2) PROBE(exec_0_4, "v0-cached-pf1-pfin-pfw");
            if (in_sp2)  PROBE(exec_0_5, "v0-cached-pf0-sp2");
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
        if (ncand == 0) {           /* nothing survived admission: fall back */
            cand[ncand] = exec_0_0; cnm[ncand] = "v0-cached-pf0-fallback"; ++ncand;
            cand[ncand] = exec_0_1; cnm[ncand] = "v0-cached-pf1-fallback"; ++ncand;
        }

        /* time every surviving candidate, several interleaved rounds, keep the
         * per-candidate minimum.  Small arenas get more reps and rounds (the
         * panel_r3 verdict measured this tuner flipping picks across runs at
         * B=1 -- an under-sampling artifact, fixed in r4). */
        double best[32];
        for (int k = 0; k < ncand; ++k) best[k] = 1e300;
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
        /* hysteresis pick: candidates are listed simplest-first per kernel,
         * so a later (more speculative) candidate must beat the incumbent by
         * more than 3% to be installed.  Near-ties go to the simpler code
         * (L36_pfa measured the coin-flip zone at 2.4%; every genuine win on
         * this board is >= 6%). */
        int bk = 0;
        for (int k = 1; k < ncand; ++k)
            if (best[k] < 0.97 * best[bk]) bk = k;
        p->fn = cand[bk];
        snprintf(g_desc, sizeof g_desc,
                 "PFA 4x9 2-sweep, lanes=lines; pick=%s (B=%d, arena=%ld vol, "
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
#define VADD(a, b)        _mm256_add_pd((a), (b))
#define VSUB(a, b)        _mm256_sub_pd((a), (b))
#define VMUL(a, b)        _mm256_mul_pd((a), (b))
#define VFMADD(a, b, c)   _mm256_fmadd_pd((a), (b), (c))
#define VFNMADD(a, b, c)  _mm256_fnmadd_pd((a), (b), (c))
#define VFMADDSUB(a,b,c)  _mm256_fmaddsub_pd((a), (b), (c))
#define VFMSUBADD(a,b,c)  _mm256_fmsubadd_pd((a), (b), (c))
#define VSWAP(a)          _mm256_permute_pd((a), 0x5)
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
 *   (t3.im, -t3.re) folds into the add as fmsubadd / fmaddsub.
 * Generic over the temp array TT and load macro LS, so a software-pipelined
 * pair of transforms can instantiate two interleaved copies. */
#define ST1G(TT, LS, N2) {                                     \
    VD x0 = LS((9 * 0 + 4 * (N2)) % 36);                       \
    VD x1 = LS((9 * 1 + 4 * (N2)) % 36);                       \
    VD x2 = LS((9 * 2 + 4 * (N2)) % 36);                       \
    VD x3 = LS((9 * 3 + 4 * (N2)) % 36);                       \
    VD t0 = VADD(x0, x2), t1 = VSUB(x0, x2);                   \
    VD t2 = VADD(x1, x3), t3 = VSUB(x1, x3);                   \
    VD sw = VSWAP(t3);                                         \
    TT[0 * 9 + (N2)] = VADD(t0, t2);                           \
    TT[2 * 9 + (N2)] = VSUB(t0, t2);                           \
    TT[1 * 9 + (N2)] = VFMSUBADD(t1, C_ONE, sw);               \
    TT[3 * 9 + (N2)] = VFMADDSUB(t1, C_ONE, sw);               \
}
#define ST1(N2) ST1G(T, LSRC, N2)

/* stage 2: four 9-point DFTs over n2 (Cooley-Tukey 3x3), PFA output map
 * k = (9*k1 + 28*k2) mod 36.  Note 9*k1 + 28*k2 == k1 (mod 4), so DFT9 number
 * k1 lands entirely on output slots congruent to k1 mod 4. */
#define ST2G(TT, SD, K1) {                                     \
    VD u0 = TT[(K1) * 9 + 0], u1 = TT[(K1) * 9 + 1];           \
    VD u2 = TT[(K1) * 9 + 2], u3 = TT[(K1) * 9 + 3];           \
    VD u4 = TT[(K1) * 9 + 4], u5 = TT[(K1) * 9 + 5];           \
    VD u6 = TT[(K1) * 9 + 6], u7 = TT[(K1) * 9 + 7];           \
    VD u8 = TT[(K1) * 9 + 8];                                  \
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
    SD((9 * (K1) + 28 * 0) % 36, v0);                          \
    SD((9 * (K1) + 28 * 1) % 36, v1);                          \
    SD((9 * (K1) + 28 * 2) % 36, v2);                          \
    SD((9 * (K1) + 28 * 3) % 36, v3);                          \
    SD((9 * (K1) + 28 * 4) % 36, v4);                          \
    SD((9 * (K1) + 28 * 5) % 36, v5);                          \
    SD((9 * (K1) + 28 * 6) % 36, v6);                          \
    SD((9 * (K1) + 28 * 7) % 36, v7);                          \
    SD((9 * (K1) + 28 * 8) % 36, v8);                          \
}
#define ST2(K1) ST2G(T, SDST, K1)

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

/* ---- software-pipelined pairs (sp2): two INDEPENDENT 36-point transforms
 * interleaved at source level, DFT4-by-DFT4 in stage 1 and DFT9-by-DFT9 in
 * stage 2, so each long DFT9 dependency chain has the other transform's
 * independent work to overlap with.  Same operations, same per-transform
 * order of additions -- output is bit-identical to two sequential calls;
 * only the instruction interleaving differs.  This is the B=1 latency lever
 * the panel_r5 VERDICT section 6 called for. */
static inline __attribute__((always_inline))
void FN(dft36_y2)(const double *s0, double *d0, const double *s1, double *d1)
{
    VD Ta[36], Tb[36];
#define LSA(i)     VLOAD(s0 + (long)(i) * (36 * 2))
#define LSB(i)     VLOAD(s1 + (long)(i) * (36 * 2))
#define SDA(i, v)  VSTORE(d0 + (long)(i) * (36 * 2), (v))
#define SDB(i, v)  VSTORE(d1 + (long)(i) * (36 * 2), (v))
#define P1(N2) ST1G(Ta, LSA, N2) ST1G(Tb, LSB, N2)
    REP9(P1)
#undef P1
#define P2(K1) ST2G(Ta, SDA, K1) ST2G(Tb, SDB, K1)
    REP4(P2)
#undef P2
#undef LSA
#undef LSB
#undef SDA
#undef SDB
}

static inline __attribute__((always_inline))
void FN(dft36_x2)(double *p0, double *p1)
{
    VD Ta[36], Tb[36];
#define LSA(i)     VLOAD(p0 + (long)(i) * (NPLANE * 2))
#define LSB(i)     VLOAD(p1 + (long)(i) * (NPLANE * 2))
#define SDA(i, v)  VSTORE(p0 + (long)(i) * (NPLANE * 2), (v))
#define SDB(i, v)  VSTORE(p1 + (long)(i) * (NPLANE * 2), (v))
#define P1(N2) ST1G(Ta, LSA, N2) ST1G(Tb, LSB, N2)
    REP9(P1)
#undef P1
#define P2(K1) ST2G(Ta, SDA, K1) ST2G(Tb, SDB, K1)
    REP4(P2)
#undef P2
#undef LSA
#undef LSB
#undef SDA
#undef SDB
}

static inline __attribute__((always_inline))
void FN(body)(const double *in, double *out, long batch, double *plane,
              const int pf, const int pfin, const int pfw, const int sp2,
              const int nta)
{
    /* pfin: paced T1 prefetch of the phase-1 input stream (L36_pfa r3's
     * PFIN, attributed).  A cursor runs PFIN_D doubles = 32 KB ahead of the
     * plane being consumed; each of the 2*(36/PW) codelet calls per plane
     * issues PFIN_L line-prefetches and advances, so exactly one plane of
     * prefetches issues per plane processed and the read stream stays busy
     * through the y-subloop's compute-only stretch.
     *
     * pfw: paced WRITE-INTENT cursor over phase 1's store stream into cold
     * `out`, one plane (PFW_D = 2592 doubles = 20.25 KB) ahead, advancing at
     * the same rate, so one plane's worth of prefetchw issues per plane
     * stored.  __builtin_prefetch(p,1,3) emits `prefetchw` on PRFCHW parts
     * (Cascade Lake, Sapphire Rapids), acquiring the line exclusive before
     * the store so the RFO overlaps compute instead of stalling the store
     * buffer.  (L36_pfa r5's PFWMID; ultimately L6_unrolled r3.) */
#define PFIN_D 4096                    /* read cursor distance = 32 KB      */
#define PFW_D  FFT36_PFW_DIST          /* write cursor distance, default 1 plane */
#define PFIN_L (36 * PW / 8)           /* lines per codelet call: 18 / 9    */
    const double *pfp = 0, *pfend = 0;
    if (pfin || nta) {
        pfend = in + batch * (long)NVOL * 2;
        pfp   = in + PFIN_D;
        if (pfp > pfend) pfp = pfend;
    }
    double *pwend = out + batch * (long)NVOL * 2;
    for (long b = 0; b < batch; ++b) {
        const double *vin  = in  + b * (long)NVOL * 2;
        double       *vout = out + b * (long)NVOL * 2;
        double *pwp = pfw ? vout + PFW_D : 0;

        /* -------- phase 1: for each x-plane, z-lines then y-lines -------- */
        for (long x = 0; x < LSIDE; ++x) {
            const double *pin  = vin  + x * (long)NPLANE * 2;
            double       *pout = vout + x * (long)NPLANE * 2;

            for (long yb = 0; yb < LSIDE / PW; ++yb) {
                if (pfin) {
                    long npl = (pfend - pfp) / 8;
                    if (npl > PFIN_L) npl = PFIN_L;
                    for (long i = 0; i < npl; ++i)
                        _mm_prefetch((const char *)(pfp + i * 8), _MM_HINT_T1);
                    pfp += npl * 8;
                }
                if (nta) {
                    /* stateless constant-lead cursor: this yb-iteration reads
                     * the 9*PW `in` lines starting at pin + yb*PW*72; prefetch
                     * the same window FFT36_NTA_DIST doubles (4 KB) ahead, NTA
                     * hint, so the single-use read stream fills L1 and never
                     * allocates L2.  Issued only here -- the z-subloop is the
                     * only consumer of `in` -- so the lead never swings. */
                    const double *q = pin + yb * (PW * 72) + FFT36_NTA_DIST;
                    for (long i = 0; i < 9 * PW; ++i)
                        if (q + i * 8 < pfend)
                            _mm_prefetch((const char *)(q + i * 8), _MM_HINT_NTA);
                }
                if (pfw) {
                    long npl = (pwend - pwp) / 8;
                    if (npl > PFIN_L) npl = PFIN_L;
                    for (long i = 0; i < npl; ++i)
                        __builtin_prefetch(pwp + i * 8, 1, 3);
                    pwp += npl * 8;
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
            if (sp2) {
                long zb = 0;
                for (; zb + 1 < LSIDE / PW; zb += 2)
                    FN(dft36_y2)(plane + zb * PW * 2, pout + zb * PW * 2,
                                 plane + (zb + 1) * PW * 2,
                                 pout + (zb + 1) * PW * 2);
                for (; zb < LSIDE / PW; ++zb)
                    FN(dft36_y)(plane + zb * PW * 2, pout + zb * PW * 2);
            } else {
                for (long zb = 0; zb < LSIDE / PW; ++zb) {
                    if (pfin) {
                        long npl = (pfend - pfp) / 8;
                        if (npl > PFIN_L) npl = PFIN_L;
                        for (long i = 0; i < npl; ++i)
                            _mm_prefetch((const char *)(pfp + i * 8), _MM_HINT_T1);
                        pfp += npl * 8;
                    }
                    if (pfw) {
                        long npl = (pwend - pwp) / 8;
                        if (npl > PFIN_L) npl = PFIN_L;
                        for (long i = 0; i < npl; ++i)
                            __builtin_prefetch(pwp + i * 8, 1, 3);
                        pwp += npl * 8;
                    }
                    FN(dft36_y)(plane + zb * PW * 2, pout + zb * PW * 2);
                }
            }
        }

        /* pfin cold-window pre-coverage (L36_pfa r3's PFNX): the paced
         * cursor leaves only the first 32 KB of in[b+1] exposed to phase-2
         * eviction; 3 lines per 36-line tile group x 324 groups = 62 KB
         * re-covers it from phase 2, whose own read stream is cache-resident. */
        const double *ncw = (pfin && b + 1 < batch)
                                ? in + (b + 1) * (long)NVOL * 2
                                : (const double *)0;

        /* -------- phase 2: x-lines, in place in `out` -------------------- */
        if (sp2) {
            for (long y = 0; y < LSIDE; ++y) {
                long zb = 0;
                for (; zb + 1 < LSIDE / PW; zb += 2)
                    FN(dft36_x2)(vout + (y * 36 + zb * PW) * 2,
                                 vout + (y * 36 + (zb + 1) * PW) * 2);
                for (; zb < LSIDE / PW; ++zb)
                    FN(dft36_x)(vout + (y * 36 + zb * PW) * 2);
            }
        } else {
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
#if PW == 4
                    if (ncw) {
                        const double *q = ncw + (y * 9 + zb) * 24;
                        _mm_prefetch((const char *)(q),      _MM_HINT_T1);
                        _mm_prefetch((const char *)(q + 8),  _MM_HINT_T1);
                        _mm_prefetch((const char *)(q + 16), _MM_HINT_T1);
                    }
#else
                    if (ncw && !(zb & 1)) {
                        const double *q = ncw + (y * 9 + (zb >> 1)) * 24;
                        _mm_prefetch((const char *)(q),      _MM_HINT_T1);
                        _mm_prefetch((const char *)(q + 8),  _MM_HINT_T1);
                        _mm_prefetch((const char *)(q + 16), _MM_HINT_T1);
                    }
#endif
                    FN(dft36_x)(base);
                }
            }
        }
    }
#undef PFIN_D
#undef PFW_D
#undef PFIN_L
}

static void FN(exec)(const double *in, double *out, long batch, double *plane)
{
    FN(body)(in, out, batch, plane, 0, 0, 0, 0, 0);
}

static void FNP1(exec)(const double *in, double *out, long batch, double *plane)
{
    FN(body)(in, out, batch, plane, 1, 0, 0, 0, 0);
}

static void FNP2(exec)(const double *in, double *out, long batch, double *plane)
{
    FN(body)(in, out, batch, plane, 4, 0, 0, 0, 0);
}

static void FNP3(exec)(const double *in, double *out, long batch, double *plane)
{
    FN(body)(in, out, batch, plane, 1, 1, 0, 0, 0);
}

static void FNP4(exec)(const double *in, double *out, long batch, double *plane)
{
    FN(body)(in, out, batch, plane, 1, 1, 1, 0, 0);
}

static void FNP5(exec)(const double *in, double *out, long batch, double *plane)
{
    FN(body)(in, out, batch, plane, 0, 0, 0, 1, 0);
}

static void FNP6(exec)(const double *in, double *out, long batch, double *plane)
{
    FN(body)(in, out, batch, plane, 1, 0, 0, 0, 1);
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
#undef PWLIST
#undef TRANSP
#undef C_ONE
#undef C_HALF
#undef C_KS
#undef DFT3M
#undef CMULM
#undef ST1G
#undef ST1
#undef ST2G
#undef ST2

#endif /* VAR */
