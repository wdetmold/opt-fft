/* L36_mixedradix -- forward complex-double 3D DFT of a fixed 36^3 cube.
 * MULTICORE (round mt_r1).  The phase-1 single-thread kernel below is kept
 * intact as the per-thread body; this round adds the 32-core layer:
 *
 *   * BATCHED (B >= 2): VOLUME-PARALLEL.  Volumes are independent, so each
 *     thread runs the tuned serial body on a contiguous block of volumes with
 *     its own scratch -- zero synchronisation inside a call, and the serial
 *     body's L2 blocking (one 746 KiB volume live at a time per thread) is
 *     exactly the right unit: DRAM traffic per volume stays at the serial
 *     floor of read(in) + RFO(out) + writeback(out) ~ 2.2 MB.  Contiguous
 *     blocks match OMP_PROC_BIND=close.  NOTE: the driver first-touches both
 *     caller buffers on its main thread, so on the two-socket node ALL caller
 *     pages live on socket 0 and the streaming cells are capped by one
 *     socket's DRAM bandwidth whatever the decomposition; the T=16
 *     (one-socket) team-size candidates in the pool exist to measure whether
 *     the remote 16 cores still pay for themselves through UPI.
 *
 *   * B = 1: WITHIN-VOLUME SPLIT.  Phase 1 parallelises over the 36
 *     independent x-planes (each thread does the z+y subloops of its planes
 *     on its own plane scratch; schedule(static,1), one implicit barrier);
 *     phase 2 parallelises over output rows/tiles with nowait.  One barrier
 *     per volume plus the region fork/join is the entire sync cost.  Units
 *     are whole planes (20736 B) and whole rows (576 B x 36) -- every unit
 *     boundary is 64-byte aligned, so no false sharing.  T=18 divides both
 *     36-unit phases exactly (2+2 units per thread) and T=16 keeps the team
 *     on one socket; both are pool candidates alongside T=32.
 *
 *   * Scratch is NT_MAX per-thread chunks, page-multiple apart, each
 *     first-touched by its own pinned thread in fft3d_create() (NUMA-local;
 *     pool spin-up happens there too, outside the timed region).
 *
 *   * The plan-time tournament (all setup) now ranges over
 *     {serial, split(T), vol(T)} x {V0,V1} x {pf mechanisms}, with the
 *     streaming arena sized against the COMBINED 32-thread cache (two L3s +
 *     32 L2s), not one socket's L3, so streaming candidates actually stream
 *     while being tuned.  Env for monitor A/Bs: FFT36_MT_T=<n> restricts
 *     team size, FFT36_MT_MODE=ser|split|vol restricts strategy;
 *     FFT36_PFIN/PFW/PIND/ZY keep their phase-1 meanings.
 *
 * Everything below this block is the phase-1 kernel unchanged -- see
 * ../../geom/strategies/L36_mixedradix.md for its full history.
 *
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
 *   a strided gather/scatter.  The 9-point module is genfft's n1_9 FMA DAG
 *   (fftw-3.3.10/dft/scalar/codelets/n1_9.c) transcribed to interleaved
 *   vectors -- see the ST2G comment (the CT 3x3 form it replaced was retired
 *   in r11 after the r10 node run confirmed the n1_9 form at -5.9%).
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
 *
 *   N1_9 DFT9 (r10; node-confirmed).  genfft's n1_9 FMA DAG (fftw-3.3.10
 *   n1_9.c, via L45_pfa r9's transcription rule) replaced the CT 3x3 module:
 *   248 -> 232 FMA-port ops per line, and the r10 node run priced it at
 *   -5.9% at B=1 (120.5 -> 113.4) -- port 0 binds at L=36.  The ct9 probe
 *   twins that carried that A/B are retired this round (question answered).
 *
 *   ZY CROSS-PLANE INTERLEAVE (this round's change -- the r10 VERDICT's
 *   L=36 directive: "attack phase 1's structure").  The node phase split
 *   says phase 1 (z+y subloops, 648 of the 972 codelet calls) runs at
 *   ~1.9x its port share while phase 2 runs at ~1.35x.  At L=36 the L45
 *   three-term costing collapses: the split-access toll is ZERO (every
 *   stride here is 0 mod 64), the plane round trip is L1-cheap, and the
 *   compulsory-L3 term was nulled by r8's fu=p1+p2w probe -- so the excess
 *   is in-core, and the shape of the z-call says where: each z-call is a
 *   serial transpose-in (72 port-5 shuffles) -> transform -> transpose-out
 *   (72 more), so consecutive z-calls put ~144 back-to-back port-5 shuffles
 *   at every call boundary while port 0 starves (r1 measured the z-pass at
 *   182 cycles/line vs y's 113 at identical arithmetic).  The zy bodies
 *   (exec codes 5/6) interleave, at CALL granularity, plane x+1's z-calls
 *   with plane x's y-calls (independent data, equal trip counts, two
 *   ping-pong plane scratches offset 2048 mod 4096 against 4K store->load
 *   aliasing): every transpose burst then sits next to 232 independent
 *   port-5-free FMAs.  Unlike the dead instruction-level interleaves (my
 *   sp2 r6 +7.7%, L36_pencilfused's PFA36X2 r6), live vector state is NOT
 *   doubled -- the calls stay sequential blocks and only the loop order
 *   changes, so the output is bit-identical to the plain body (same bit
 *   class, hence installable).  Direction named by L36_pfa r10 Next #1
 *   ("fusing the two phase-1 subloop passes over pl") and r10 VERDICT SS6,
 *   attributed.  Wallaby cannot price it (its B=1 is port-5 bound with two
 *   FMA pipes; interleaving moves no port-5 work) -- the node tournament
 *   decides, and the plan description reports the pf0/zy-pf0 twins' arena
 *   times so the verdict rides the leaderboard whatever the pick.
 *
 *   ANTI-ALIAS SCRATCH PINNING (r8; now DEFAULT-OFF everywhere).  The
 *   y-subloop loads the plane scratch and stores `out` at the same 576-byte
 *   stride; pinning slides the scratch inside 4 KB of slack so
 *   (pout - pl) mod 4096 is a chosen constant (best plateau PIND = 2112).
 *   r8 shipped it always-on in the cached regime and the node priced it at
 *   0 to -1.2% (B=1 118.5 -> 120.0, the only change on that path), and the
 *   lottery rationale died separately: L17_matrixsimd r8 showed glibc's
 *   mmap'd allocations give FIXED relative offsets across processes.  The
 *   machinery stays (zero cost when off) and FFT36_PIND=<bytes> (env, read
 *   once at plan time; -1 = off) remains absolute for monitor A/Bs.
 *
 *   RETIRED (see the strategy record for the numbers): the NT-store path and
 *   cross-volume xv prefetch (r6; node rejected NT four rounds running), sp2
 *   source-interleaved transform pairs (r8; +7.7% wallaby, +5% Haswell, and
 *   the node's own r7 tournament declined it), nta constant-lead input
 *   prefetch (r8; zero node picks across all three L=36 entries' three
 *   independent forms in r7 -- closed by null), the V2 kernel (256-bit
 *   EVEX; never picked on the node in seven rounds, and the only source of
 *   noisy-window mis-picks on wallaby), the rolled DSB-resident codelet
 *   (r10; node probe read rolled +22..24% SLOWER than unrolled at B=1 --
 *   the front-end theory measured absent, r9 VERDICT SS2), and the ct9
 *   CT-3x3 probe twins (r11; their question was answered by the r10 node
 *   run: n19/ct9 probe ratio 0.958-0.971, cell -5.9% -- port 0 binds).
 *
 * OPERATION COUNT (per 36-point line, as vector instructions over PW lanes)
 *       DFT4       :  8 FMA-port ops + 1 shuffle   x9  =  72 +  9
 *       DFT9 n1_9  : 40 FMA-port ops + 12 shuffles x4  = 160 + 48
 *       total      :                    232 FMA-port ops + 57 shuffles / PW lines
 *   In real flops: 9*20 + 4*(24 + 2*56) = 180 + 544 = 724 flops per 36-point
 *   line; per volume 3 * 36^2 * 724 = 2,814,912 flops in 3888 line transforms
 *   (the DAG trades 16 flops/line for 16 fewer FMA-port instructions).
 *
 * ASSUMPTIONS
 *   * L == 36 only; fft3d_supports() refuses everything else.
 *   * `in` and `out` are 64-byte aligned and distinct (the driver guarantees
 *     both).  Every vector access made here is at a multiple of 64 bytes from
 *     the volume base -- (x*1296 + y*36 + zb*PW)*16 with 4 | 1296, 4 | 36.
 *   * `out` doubles as the working buffer between phase 1 and phase 2.  `in` is
 *     never written.
 *   * Two kernels are compiled: V0 = AVX2+FMA (2 complex lanes, 16 ymm) and
 *     V1 = AVX-512F (4 complex lanes, 32 zmm).  fft3d_create() checks CPU
 *     support, verifies the AVX-512 kernel numerically against the AVX2 one,
 *     then times all surviving (kernel x mechanism) combinations and keeps
 *     the fastest.
 */

#ifndef VAR
/* ======================= common (width-independent) part ================== */

#define _POSIX_C_SOURCE 200809L

#include <complex.h>
#include <immintrin.h>
#include <math.h>
#include <omp.h>
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
/* genfft n1_9 (FMA form) DAG constants, fftw-3.3.10 n1_9.c, via L45_pfa r9.
 * Alternating rows [c,-c,...] fold the sign of a re/im-swapped operand;
 * VPAIR(-c,c) call sites become the opposite FMA flavour of ALT8(c). */
static const double KC_A176[8] __attribute__((aligned(64)))
    = ALT8(0.17632698070846497347109038686862);   /* tan(pi/18) */
static const double KC_A839[8] __attribute__((aligned(64)))
    = ALT8(0.83909963117728001176312729812318);
static const double KC_A777[8] __attribute__((aligned(64)))
    = ALT8(0.77786191343020616002817797731863);
static const double KC_A984[8] __attribute__((aligned(64)))
    = ALT8(0.98480775301220805936674302458952);
static const double KC_A492[8] __attribute__((aligned(64)))
    = ALT8(0.49240387650610402968337151229476);
static const double KC_A363[8] __attribute__((aligned(64)))
    = ALT8(0.36397023426620236135104788277683);   /* tan(pi/9) */
static const double KC_A954[8] __attribute__((aligned(64)))
    = ALT8(0.95418889413867113349926836418725);
static const double KC_852[8] __attribute__((aligned(64)))
    = SPLAT8(0.85286853195244320962825096394007);

/* nine 4-point stages, then four 9-point stages */
#define REP9(M) M(0) M(1) M(2) M(3) M(4) M(5) M(6) M(7) M(8)
#define REP4(M) M(0) M(1) M(2) M(3)

/* pfw write-intent cursor distance (doubles): default one x-plane = 20.25 KB,
 * L36_pfa's pacing arithmetic.  Overridable with -DFFT36_PFW_DIST=... so the
 * monitor's 1296/2592/5184 sweep needs no source edit. */
#ifndef FFT36_PFW_DIST
#define FFT36_PFW_DIST (NPLANE * 2)
#endif
/* anti-alias scratch pin target: (pout - pl) mod 4096, in bytes, multiple of
 * 64.  DEFAULT OFF since r9 (the r8 node run priced always-on pinning at 0 to
 * -1.2% at B=1).  Runtime override: FFT36_PIND env (read once at plan time;
 * e.g. 2112 = the minimum-alias plateau center; -1 = off). */
static long g_pind = -1;

typedef void (*exec_fn)(const double *, double *, long, double *);

/* zy ping-pong scratch B sits ZY_OFF_D doubles past the plane base:
 * 26624 B = 6.5 pages, so (plB - plA) mod 4096 = 2048 -- maximally separated
 * from plA's 576-B-stride loads in the 4K store->load alias window -- and
 * clear of the pind slide region [plane, plane + 4096 + 20736). */
#define ZY_OFF_D 3328

/* ---- multicore layer ----
 * Per-thread scratch: NT_MAX chunks of SCR_STRIDE doubles.  SCR_STRIDE is a
 * page multiple (6144 * 8 = 48 KiB = 12 pages; the body needs ZY_OFF_D +
 * NPLANE*2 = 5920), so no two threads ever share a page, and each chunk is
 * first-touched by its own pinned thread in fft3d_create() (NUMA-local). */
#define NT_MAX 32
#define SCR_STRIDE 6144

typedef void (*mt_fn)(const double *, double *, long, const fft3d_plan *);

struct fft3d_plan {
    long batch;
    mt_fn run;         /* installed top-level strategy                     */
    exec_fn bfn;       /* per-slab serial body, used by mt_vol_run         */
    int T;             /* team size for the installed strategy             */
    double *scratch;   /* NT_MAX * SCR_STRIDE doubles, per-thread chunks   */
    void *raw;
};

/* exec_<variant>_<code>:
 *   0 = cached, no prefetch                        (pf0)
 *   1 = cached, phase-2 streams 1 line ahead       (pf1)
 *   2 = cached, phase-2 streams 4 lines ahead      (pf4)
 *   3 = code 1 + paced phase-1 input prefetch      (pf1-pfin)
 *   4 = code 3 + paced phase-1 prefetchw on out    (pf1-pfin-pfw)
 *   5 = zy cross-plane z/y interleave, no prefetch (zy-pf0)
 *   6 = zy cross-plane z/y interleave, pf 1 line   (zy-pf1)
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

/* within-volume split bodies (B=1 / small batch); _p0/_p1 = phase-2 x-stream
 * prefetch off / 1 line ahead */
static void split_0_p0(const double *, double *, long, const fft3d_plan *);
static void split_0_p1(const double *, double *, long, const fft3d_plan *);
static void split_1_p0(const double *, double *, long, const fft3d_plan *);
static void split_1_p1(const double *, double *, long, const fft3d_plan *);

/* volume-parallel: a contiguous block of volumes per thread (contiguity
 * matches OMP_PROC_BIND=close), each thread running the tuned serial body on
 * its own slab with its own scratch.  No synchronisation inside the call. */
static void mt_vol_run(const double *in, double *out, long batch, const fft3d_plan *p)
{
    const int T = p->T;
    if (T <= 1 || batch < 2) {
        p->bfn(in, out, batch, p->scratch);
        return;
    }
#pragma omp parallel num_threads(T)
    {
        int t = omp_get_thread_num();
        long v0 = batch * (long)t / T, v1 = batch * (long)(t + 1) / T;
        if (v1 > v0)
            p->bfn(in + v0 * (long)NVOL * 2, out + v0 * (long)NVOL * 2,
                   v1 - v0, p->scratch + (long)t * SCR_STRIDE);
    }
}

/* volume-parallel, work-stealing: on the scoring node every caller page is
 * on socket 0 (the driver first-touches serially), so socket-1 threads run
 * each volume slower through UPI and a static split leaves socket-0 threads
 * idle at the join.  dynamic,1 rebalances that speed asymmetry; only offered
 * at streaming batch, where there is no cross-call cache reuse for static
 * ownership to protect.  Costs pfin/pfw their cross-volume cursor continuity
 * (each volume is a fresh body call), which the tournament prices. */
static void mt_vol_dyn_run(const double *in, double *out, long batch,
                           const fft3d_plan *p)
{
    const int T = p->T;
    if (T <= 1 || batch < 2) {
        p->bfn(in, out, batch, p->scratch);
        return;
    }
#pragma omp parallel num_threads(T)
    {
        double *pl = p->scratch + (long)omp_get_thread_num() * SCR_STRIDE;
#pragma omp for schedule(dynamic, 1) nowait
        for (long b = 0; b < batch; ++b)
            p->bfn(in + b * (long)NVOL * 2, out + b * (long)NVOL * 2, 1, pl);
    }
}

/* instantiate the kernel once per (ISA, vector width) variant */
#define VAR 0
#include __FILE__
#undef VAR
#define VAR 1
#include __FILE__
#undef VAR

/* ------------------------------ the API ---------------------------------- */

const char *fft3d_name(void) { return "L36_mixedradix"; }

/* The chosen candidate is spliced in by fft3d_create() so the monitor can read
 * the tuner's verdict off the leaderboard / raw JSON. */
static char g_desc[320];

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
    void *sc = NULL;
    if (posix_memalign(&sc, 4096,
                       (size_t)NT_MAX * SCR_STRIDE * sizeof(double)) != 0 || !sc) {
        free(p);
        return NULL;
    }
    p->scratch = (double *)sc;
    p->raw = sc;
    p->run = mt_vol_run;
    p->bfn = exec_0_0;
    p->T = 1;

    int m = omp_get_max_threads();
    if (m > NT_MAX) m = NT_MAX;
    if (m < 1) m = 1;

    /* Team spin-up and NUMA placement are setup: instantiate the pool ONCE
     * here, and let each pinned thread first-touch its own scratch chunk so
     * those pages are local to the socket that will use them
     * (OMP_PROC_BIND=close pins thread t to the same core in every region). */
#pragma omp parallel num_threads(m)
    {
        memset(p->scratch + (long)omp_get_thread_num() * SCR_STRIDE, 0,
               (size_t)SCR_STRIDE * sizeof(double));
    }

    /* anti-alias pin target, read ONCE at plan time so execution stays
     * repeatable: FFT36_PIND=<bytes>, -1/unset = off (see phase-1 history:
     * the r8 node run priced always-on pinning at 0 to -1.2% at B=1). */
    {
        const char *pe = getenv("FFT36_PIND");
        if (pe && *pe) {
            long v = strtol(pe, NULL, 0);
            g_pind = v < 0 ? -1 : (v & 4095l & ~63l);
        }
    }

    /* ---- regime: does the batch stream through ONE socket's LLC?  Decides
     * which serial-body mechanisms are in play (pfw only where `out` is
     * genuinely cold; NT stores stay retired -- node-rejected four rounds
     * running in phase 1). */
    long l3 = -1;
    {
#ifdef _SC_LEVEL3_CACHE_SIZE
        l3 = sysconf(_SC_LEVEL3_CACHE_SIZE);
#endif
        if (l3 <= 0) l3 = 22l << 20;   /* the scoring node's 22 MiB */
    }
    double foot = (double)batch * (double)NVOL * 16.0 * 2.0;
    int streaming = foot > 1.25 * (double)l3;

    /* diagnostic overrides for monitor A/Bs, read once at plan time:
     * FFT36_PFIN/PFW=0|1 gate the paced-prefetch mechanisms as in phase 1;
     * FFT36_ZY=1 re-admits the zy interleave bodies (default out this round:
     * they were a single-core port-5 bet, unpriced by the node before the
     * phase boundary); FFT36_MT_T=<n> restricts parallel candidates to one
     * team size; FFT36_MT_MODE=ser|split|vol restricts the strategy pool. */
    int pfinmode = -1, pfwmode = -1, zymode = -1;
    long mtT = 0;
    int mtmode = 0;
    {
        const char *po = getenv("FFT36_PFIN");
        if (po && (*po == '0' || *po == '1')) pfinmode = *po - '0';
        const char *wo = getenv("FFT36_PFW");
        if (wo && (*wo == '0' || *wo == '1')) pfwmode = *wo - '0';
        const char *zo = getenv("FFT36_ZY");
        if (zo && (*zo == '0' || *zo == '1')) zymode = *zo - '0';
        const char *te = getenv("FFT36_MT_T");
        if (te && *te) {
            mtT = strtol(te, NULL, 0);
            if (mtT < 0) mtT = 0;
            if (mtT > m) mtT = m;
        }
        const char *mo = getenv("FFT36_MT_MODE");
        if (mo && *mo) {
            if (!strcmp(mo, "ser")) mtmode = 1;
            else if (!strcmp(mo, "split")) mtmode = 2;
            else if (!strcmp(mo, "vol")) mtmode = 3;
        }
    }
    int in_plain = (pfinmode != 1) && (pfwmode != 1);
    int in_pfin  = (pfinmode != 0) && (pfwmode != 1);
    int in_pfw   = (pfinmode != 0) && (pfwmode != 0);

    /* ---- self-tuning.  All of this is setup, hence excluded from the score.
     * Every candidate must match exec_0_0's output to 1e-13 relative (the
     * parallel bodies run the same codelets on the same data in a different
     * order, so they are in fact bit-identical; the gate is a safety net).
     *
     * Arena: with a 32-thread team the arena must stream against the
     * COMBINED cache -- two sockets' L3 plus every core's L2 -- or streaming
     * candidates are ranked on a cached arena (L36_pfa's round-2 lesson,
     * upgraded for phase 2). */
    long nt;
    if (streaming) {
        long arena = (long)(2.5 * (2.0 * (double)l3 + (double)m * (1l << 20)) /
                            ((double)NVOL * 32.0)) + 1;
        if (arena < 48)  arena = 48;
        if (arena > 128) arena = 128;
        nt = batch < arena ? batch : arena;
    } else {
        nt = batch < NT_MAX ? batch : NT_MAX;
    }
    size_t nd = (size_t)NVOL * 2 * (size_t)nt;
    double *ti = NULL, *o0 = NULL, *ox = NULL;
    if (posix_memalign((void **)&ti, 4096, nd * sizeof(double)) == 0 &&
        posix_memalign((void **)&o0, 4096, nd * sizeof(double)) == 0 &&
        posix_memalign((void **)&ox, 4096, nd * sizeof(double)) == 0) {

        /* serial fill, like the driver's fread: every arena page lands on
         * the create-caller's socket, which is exactly what the parallel
         * candidates will face on the caller's buffers */
        uint64_t s = 0x9E3779B97F4A7C15ull;
        for (size_t i = 0; i < nd; ++i) {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            ti[i] = (double)(int64_t)(s >> 11) * (1.0 / 9007199254740992.0);
        }
        memset(o0, 0, nd * sizeof(double));
        exec_0_0(ti, o0, nt, p->scratch);

        int have_512 = 0;
#if defined(__x86_64__)
        __builtin_cpu_init();
        have_512 = __builtin_cpu_supports("avx512f");
#endif

        /* ---- candidate pool.  Order = hysteresis order: serial first (the
         * shape phase 1 certified, and the denominator of the parallel-
         * efficiency number), then V1 before V0 (V1 won every node cell in
         * every phase-1 round), then simplest strategy first within a
         * kernel.  A later candidate must beat the incumbent by > 3%. */
        typedef struct { mt_fn run; exec_fn bfn; int T; const char *nm; } mtcand;
        static char nmb[64][40];
        mtcand cd[64];
        int nc = 0;
#define CAND(RUNF, BFN, TT, ...) do { if (nc < 64) {                         \
            snprintf(nmb[nc], sizeof nmb[nc], __VA_ARGS__);                  \
            cd[nc].run = (RUNF); cd[nc].bfn = (BFN); cd[nc].T = (TT);        \
            cd[nc].nm = nmb[nc]; ++nc; } } while (0)
        int vs[2], nv = 0;
        if (have_512) vs[nv++] = 1;
        vs[nv++] = 0;
        for (int vi = 0; vi < nv; ++vi) {
            const int v = vs[vi];
            exec_fn e0 = v ? exec_1_0 : exec_0_0;   /* pf0          */
            exec_fn e1 = v ? exec_1_1 : exec_0_1;   /* pf1          */
            exec_fn e2 = v ? exec_1_2 : exec_0_2;   /* pf4          */
            exec_fn e3 = v ? exec_1_3 : exec_0_3;   /* pf1-pfin     */
            exec_fn e4 = v ? exec_1_4 : exec_0_4;   /* pf1-pfin-pfw */
            exec_fn e5 = v ? exec_1_5 : exec_0_5;   /* zy-pf0       */
            exec_fn e6 = v ? exec_1_6 : exec_0_6;   /* zy-pf1       */
            mt_fn sp0 = v ? split_1_p0 : split_0_p0;
            mt_fn sp1 = v ? split_1_p1 : split_0_p1;

            if (mtmode <= 1 && (mtT == 0 || mtT == 1)) {
                CAND(mt_vol_run, e0, 1, "v%d-ser-pf0", v);
                if (batch >= 2)
                    CAND(mt_vol_run, streaming ? e1 : e2, 1,
                         "v%d-ser-%s", v, streaming ? "pf1" : "pf4");
                if (zymode == 1) CAND(mt_vol_run, e5, 1, "v%d-ser-zy0", v);
            }
            if (mtmode == 1) continue;

            if (batch >= 2 && (mtmode == 0 || mtmode == 3)) {
                /* volume-parallel team sizes: full width; one socket of the
                 * scoring node; and, below full width, one volume/thread */
                int Tv[3] = { m, 16, (int)(batch < (long)m ? batch : 0) };
                for (int i = 0; i < 3; ++i) {
                    int T = Tv[i];
                    if (T < 2 || T > m) continue;
                    if (i > 0 && T >= m) continue;      /* dedupe vs Tv[0] */
                    if (i == 2 && T == 16) continue;    /* dedupe vs Tv[1] */
                    if (mtT && T != mtT) continue;
                    if (in_plain && !streaming)
                        CAND(mt_vol_run, e0, T, "v%d-vol%d-pf0", v, T);
                    if (in_plain)
                        CAND(mt_vol_run, e1, T, "v%d-vol%d-pf1", v, T);
                    if (in_plain && !streaming)
                        CAND(mt_vol_run, e2, T, "v%d-vol%d-pf4", v, T);
                    if (in_pfin)
                        CAND(mt_vol_run, e3, T, "v%d-vol%d-pfin", v, T);
                    if (in_pfw)
                        CAND(mt_vol_run, e4, T, "v%d-vol%d-pfw", v, T);
                    if (streaming && batch > (long)T) {
                        if (in_plain)
                            CAND(mt_vol_dyn_run, e1, T, "v%d-dyn%d-pf1", v, T);
                        if (in_pfw)
                            CAND(mt_vol_dyn_run, e4, T, "v%d-dyn%d-pfw", v, T);
                    }
                    if (zymode == 1)
                        CAND(mt_vol_run, e6, T, "v%d-vol%d-zy1", v, T);
                }
            }
            if (batch < (long)m && (mtmode == 0 || mtmode == 2)) {
                /* within-volume split; T=18 divides both 36-unit phases
                 * exactly, T=16 is one full socket on the scoring node */
                int Ts[4] = { m, 18, 16, 8 };
                for (int i = 0; i < 4; ++i) {
                    int T = Ts[i];
                    if (T < 2 || T > m) continue;
                    if (i > 0 && T >= m) continue;
                    if (mtT && T != mtT) continue;
                    CAND(sp0, e0, T, "v%d-split%d-pf0", v, T);
                    CAND(sp1, e0, T, "v%d-split%d-pf1", v, T);
                }
            }
        }
#undef CAND

        int keep[64], ns = 0;
        for (int k = 0; k < nc; ++k) {
            memset(ox, 0, nd * sizeof(double));
            p->bfn = cd[k].bfn; p->T = cd[k].T;
            cd[k].run(ti, ox, nt, p);
            double num = 0.0, den = 0.0;
            for (size_t i = 0; i < nd; ++i) {
                double d = ox[i] - o0[i];
                num += d * d;
                den += o0[i] * o0[i];
            }
            if (den > 0.0 && sqrt(num / den) < 1e-13) keep[ns++] = k;
        }
        if (ns == 0) {              /* nothing survived admission: fall back */
            cd[0].run = mt_vol_run; cd[0].bfn = exec_0_0; cd[0].T = 1;
            cd[0].nm = "v0-ser-pf0-fallback";
            keep[ns++] = 0;
        }

        /* time every survivor, several interleaved rounds, keep the
         * per-candidate minimum (phase-1 r4's under-sampling fix kept) */
        double best[64];
        for (int k = 0; k < ns; ++k) best[k] = 1e300;
        int reps = nt >= 16 ? 1 : (nt >= 4 ? 4 : 16);
        int rounds = nt >= 16 ? 4 : (nt >= 4 ? 6 : 10);
        for (int round = 0; round < rounds; ++round) {
            for (int k = 0; k < ns; ++k) {
                const mtcand *c = &cd[keep[k]];
                p->bfn = c->bfn; p->T = c->T;
                c->run(ti, ox, nt, p);              /* warm */
                double t0 = now_s();
                for (int r = 0; r < reps; ++r)
                    c->run(ti, ox, nt, p);
                double dt = now_s() - t0;
                if (dt < best[k]) best[k] = dt;
            }
        }
        int bk = 0;
        for (int k = 1; k < ns; ++k)
            if (best[k] < 0.97 * best[bk]) bk = k;
        const mtcand *w = &cd[keep[bk]];
        p->run = w->run; p->bfn = w->bfn; p->T = w->T;

        /* publish the serial-vs-pick pair so parallel efficiency rides the
         * leaderboard (the brief asks for it; L36_pfa r8's probe-through-
         * description pattern) */
        double t_ser = -1.0;
        for (int k = 0; k < ns; ++k)
            if (cd[keep[k]].T == 1 && (t_ser < 0.0 || best[k] < t_ser))
                t_ser = best[k];
        double us_win = best[bk] / reps / nt * 1e6;
        double us_ser = t_ser > 0.0 ? t_ser / reps / nt * 1e6 : -1.0;
        double eff = (us_ser > 0.0 && w->T > 1) ? us_ser / ((double)w->T * us_win)
                                                : 1.0;
        snprintf(g_desc, sizeof g_desc,
                 "MT PFA 4x9 n1_9; pick=%s (B=%d m=%d arena=%ld stream=%d %dc) "
                 "us/vol ser=%.1f pick=%.1f eff=%.2f",
                 w->nm, batch, m, nt, streaming, ns, us_ser, us_win, eff);
    }
    free(ti); free(o0); free(ox);
    return p;
}

void fft3d_execute(fft3d_plan *plan, const double _Complex *in, double _Complex *out)
{
    plan->run((const double *)in, (double *)out, plan->batch, plan);
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
/* ---- 256-bit: 2 complex lanes per ymm, VEX/AVX2, 16 registers ---- */
#pragma GCC push_options
#pragma GCC target("avx2,fma")
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

/* stage 2: four 9-point DFTs over n2, PFA output map k = (9*k1 + 28*k2)
 * mod 36.  Note 9*k1 + 28*k2 == k1 (mod 4), so DFT9 number k1 lands entirely
 * on output slots congruent to k1 mod 4.
 *
 * The DFT9 is genfft's n1_9 FMA DAG (fftw-3.3.10 dft/scalar/codelets/n1_9.c,
 * 24 add + 56 fma = 80 scalar FMA-port ops) transcribed pairwise to
 * interleaved vectors: 40 FMA-port ops + 12 shuffles (the CT 3x3 twin below:
 * 44 + 10).  Transcription rule (L45_pfa round panel_r9, attributed): every
 * scalar re/im line pair is one vector op; every re/im crossing is one VSWAP
 * with the signs folded into an alternating [c,-c,...] constant row.
 * Stage A per radix-3 column j = {xj, xj+3, xj+6}: sJ = column sum, SJ =
 * full sum, aJ = xJ - sJ/2, iJ = VSWAP(xJ+3 - xJ+6); pJ/qJ = aJ -+ i*s3*eJ
 * (the two rotated DFT3 outputs).  Block k={0,3,6} is a DFT3 on the sums;
 * blocks k={1,4,7} / k={2,5,8} build w = (1 -+ c*i)*p (one VSWAP+FMA each),
 * cross them (u, z), and fan out through the K984/K492/K852 spine. */
#define ST2G(TT, SD, K1) {                                     \
    VD x0 = TT[(K1) * 9 + 0], x1 = TT[(K1) * 9 + 1];           \
    VD x2 = TT[(K1) * 9 + 2], x3 = TT[(K1) * 9 + 3];           \
    VD x4 = TT[(K1) * 9 + 4], x5 = TT[(K1) * 9 + 5];           \
    VD x6 = TT[(K1) * 9 + 6], x7 = TT[(K1) * 9 + 7];           \
    VD x8 = TT[(K1) * 9 + 8];                                  \
    VD s0 = VADD(x3, x6), e0 = VSUB(x3, x6);                   \
    VD S0 = VADD(x0, s0), a0 = VFNMADD(C_HALF, s0, x0);        \
    VD i0 = VSWAP(e0);                                         \
    VD s1 = VADD(x4, x7), e1 = VSUB(x4, x7);                   \
    VD S1 = VADD(x1, s1), a1 = VFNMADD(C_HALF, s1, x1);        \
    VD i1 = VSWAP(e1);                                         \
    VD p1 = VFMADD (i1, C_KS, a1);                             \
    VD q1 = VFNMADD(i1, C_KS, a1);                             \
    VD s2 = VADD(x5, x8), e2 = VSUB(x5, x8);                   \
    VD S2 = VADD(x2, s2), a2 = VFNMADD(C_HALF, s2, x2);        \
    VD i2 = VSWAP(e2);                                         \
    VD p2 = VFMADD (i2, C_KS, a2);                             \
    VD q2 = VFNMADD(i2, C_KS, a2);                             \
    /* k = 0, 3, 6: DFT3 on the column sums */                 \
    VD sg = VADD(S1, S2), d3 = VSUB(S2, S1), id = VSWAP(d3);   \
    VD b0 = VFNMADD(C_HALF, sg, S0);                           \
    SD((9 * (K1) + 28 * 0) % 36, VADD(S0, sg));                \
    SD((9 * (K1) + 28 * 3) % 36, VFNMADD(id, C_KS, b0));       \
    SD((9 * (K1) + 28 * 6) % 36, VFMADD (id, C_KS, b0));       \
    /* k = 1, 4, 7 */                                          \
    {                                                          \
    VD v1 = VFMADD (i0, C_KS, a0);                             \
    VD w2 = VFNMADD(VSWAP(p2), VLOAD(KC_A176), p2);            \
    VD w1 = VFMADD (VSWAP(p1), VLOAD(KC_A839), p1);            \
    VD u1 = VFMADD (w1, VLOAD(KC_A777), VSWAP(w2));            \
    VD z1 = VFMADD (VSWAP(w1), VLOAD(KC_A777), w2);            \
    SD((9 * (K1) + 28 * 1) % 36, VFMADD(u1, VLOAD(KC_A984), v1)); \
    VD r1 = VFNMADD(u1, VLOAD(KC_A492), v1);                   \
    SD((9 * (K1) + 28 * 4) % 36, VFMADD (z1, VLOAD(KC_852), r1)); \
    SD((9 * (K1) + 28 * 7) % 36, VFNMADD(z1, VLOAD(KC_852), r1)); \
    }                                                          \
    /* k = 2, 5, 8 */                                          \
    {                                                          \
    VD v2 = VFNMADD(i0, C_KS, a0);                             \
    VD wA = VFMADD (q1, VLOAD(KC_A176), VSWAP(q1));            \
    VD wB = VFNMADD(VSWAP(q2), VLOAD(KC_A363), q2);            \
    VD uB = VFNMADD(wB, VLOAD(KC_A954), wA);                   \
    VD zB = VFNMADD(VSWAP(wB), VLOAD(KC_A954), VSWAP(wA));     \
    SD((9 * (K1) + 28 * 2) % 36, VFMADD(uB, VLOAD(KC_A984), v2)); \
    VD rB = VFNMADD(uB, VLOAD(KC_A492), v2);                   \
    SD((9 * (K1) + 28 * 5) % 36, VFNMADD(zB, VLOAD(KC_852), rB)); \
    SD((9 * (K1) + 28 * 8) % 36, VFMADD (zB, VLOAD(KC_852), rB)); \
    }                                                          \
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

/* one z-axis block: transpose PW lines in from row layout at `pin`,
 * 36-point transform, transpose back out into the plane scratch `pl`.
 * (Factored out of the plain body so the zy bodies share it verbatim --
 * identical code, identical bits.) */
static inline __attribute__((always_inline))
void FN(zblock)(const double *pin, double *pl, long yb)
{
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
        double *q = pl + (yb * PW * 36 + g * PW) * 2;
#define GET(j) A[j] = Yv[g * PW + (j)];
        PWLIST(GET)
#undef GET
        TRANSP(A, B);
#define PST(j) VSTORE(q + (j) * 36 * 2, B[j]);
        PWLIST(PST)
#undef PST
    }
}

static inline __attribute__((always_inline))
void FN(body)(const double *in, double *out, long batch, double *plane,
              const int pf, const int pfin, const int pfw)
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
    if (pfin) {
        pfend = in + batch * (long)NVOL * 2;
        pfp   = in + PFIN_D;
        if (pfp > pfend) pfp = pfend;
    }
    double *pwend = out + batch * (long)NVOL * 2;
    const long pind = g_pind;
    for (long b = 0; b < batch; ++b) {
        const double *vin  = in  + b * (long)NVOL * 2;
        double       *vout = out + b * (long)NVOL * 2;
        double *pwp = pfw ? vout + PFW_D : 0;

        /* -------- phase 1: for each x-plane, z-lines then y-lines -------- */
        for (long x = 0; x < LSIDE; ++x) {
            const double *pin  = vin  + x * (long)NPLANE * 2;
            double       *pout = vout + x * (long)NPLANE * 2;
            /* anti-alias pin: slide the plane scratch inside its 4 KB slack
             * so (pout - pl) mod 4096 == pind for EVERY plane (pout advances
             * 256 mod 4096 per plane, so a fixed scratch cannot).  Chosen so
             * the y-subloop's plane-loads dodge the 4K-alias shadow of its
             * in-flight pout-stores; pind is a multiple of 64 and both bases
             * are 64/4096-aligned, so pl stays 64-byte aligned. */
            double *pl = plane;
            if (pind >= 0)
                pl = (double *)((char *)plane +
                     ((((uintptr_t)pout - (uintptr_t)plane) - (uintptr_t)pind)
                      & 4095u));

            for (long yb = 0; yb < LSIDE / PW; ++yb) {
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
                FN(zblock)(pin, pl, yb);
            }
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
                FN(dft36_y)(pl + zb * PW * 2, pout + zb * PW * 2);
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
#undef PFIN_D
#undef PFW_D
#undef PFIN_L
}

/* zy body: phase 1 interleaves plane x+1's z-blocks with plane x's y-lines
 * at call granularity, over two ping-pong plane scratches, so every port-5
 * transpose burst has 232 independent port-5-free FMAs beside it in the OOO
 * window.  Each call is the same inlined codelet on the same data as the
 * plain body -- only the order of INDEPENDENT calls changes, so the output
 * is bit-identical.  The pind slide is not applied here (two scratches with
 * an engineered mutual offset replace it; pind stays a plain-body A/B). */
static inline __attribute__((always_inline))
void FN(body_zy)(const double *in, double *out, long batch, double *plane,
                 const int pf)
{
    double *const plane_b = plane + ZY_OFF_D;
    for (long b = 0; b < batch; ++b) {
        const double *vin  = in  + b * (long)NVOL * 2;
        double       *vout = out + b * (long)NVOL * 2;
        double *pa = plane, *pb = plane_b;

        /* prologue: plane 0's z-subloop fills scratch A */
        for (long yb = 0; yb < LSIDE / PW; ++yb)
            FN(zblock)(vin, pa, yb);
        /* steady state: z(x+1) into the idle scratch, y(x) out of the hot
         * one, one iteration each, alternating */
        for (long x = 0; x < LSIDE - 1; ++x) {
            const double *pin  = vin  + (x + 1) * (long)NPLANE * 2;
            double       *pout = vout + x * (long)NPLANE * 2;
            for (long i = 0; i < LSIDE / PW; ++i) {
                FN(zblock)(pin, pb, i);
                FN(dft36_y)(pa + i * PW * 2, pout + i * PW * 2);
            }
            double *t = pa; pa = pb; pb = t;
        }
        /* epilogue: plane 35's y-subloop drains the last scratch */
        {
            double *pout = vout + (LSIDE - 1) * (long)NPLANE * 2;
            for (long zb = 0; zb < LSIDE / PW; ++zb)
                FN(dft36_y)(pa + zb * PW * 2, pout + zb * PW * 2);
        }

        /* phase 2: unchanged from the plain body */
        for (long y = 0; y < LSIDE; ++y) {
            for (long zb = 0; zb < LSIDE / PW; ++zb) {
                double *base = vout + (y * 36 + zb * PW) * 2;
                if (pf) {
                    for (int i = 0; i < 36; ++i)
                        _mm_prefetch((const char *)(base + i * (NPLANE * 2) + pf * 8),
                                     _MM_HINT_T0);
                }
                FN(dft36_x)(base);
            }
        }
    }
}

/* -------- multicore within-volume split (B=1 / small batch) --------------
 * Called INSIDE a parallel region.  Phase 1: the 36 x-planes are independent
 * given per-thread plane scratch; schedule(static,1) round-robins them (a
 * plane is 20736 B, so no two threads share a cache line) and its implicit
 * barrier is the one phase1->phase2 sync the data flow requires.  Phase 2:
 * units are whole (y,zb) tiles at PW=4 (64-B stores) or whole y-rows at PW=2
 * (32-B stores would false-share across a tile boundary; a row is 576 B per
 * x-line), nowait so a finished thread runs ahead into the next volume's
 * phase 1 (which touches only volume b+1, disjoint from phase 2 of b). */
static inline __attribute__((always_inline))
void FN(split_body)(const double *in, double *out, long batch,
                    const fft3d_plan *p, const int pf)
{
    double *pl = p->scratch + (long)omp_get_thread_num() * SCR_STRIDE;
    for (long b = 0; b < batch; ++b) {
        const double *vin  = in  + b * (long)NVOL * 2;
        double       *vout = out + b * (long)NVOL * 2;
#pragma omp for schedule(static, 1)
        for (long x = 0; x < LSIDE; ++x) {
            const double *pin  = vin  + x * (long)NPLANE * 2;
            double       *pout = vout + x * (long)NPLANE * 2;
            for (long yb = 0; yb < LSIDE / PW; ++yb)
                FN(zblock)(pin, pl, yb);
            for (long zb = 0; zb < LSIDE / PW; ++zb)
                FN(dft36_y)(pl + zb * PW * 2, pout + zb * PW * 2);
        }
#if PW == 4
#pragma omp for schedule(static) nowait
        for (long u = 0; u < LSIDE * (LSIDE / PW); ++u) {
            long y = u / (LSIDE / PW), zb = u % (LSIDE / PW);
            double *base = vout + (y * 36 + zb * PW) * 2;
            if (pf)
                for (int i = 0; i < 36; ++i)
                    _mm_prefetch((const char *)(base + i * (NPLANE * 2) + pf * 8),
                                 _MM_HINT_T0);
            FN(dft36_x)(base);
        }
#else
#pragma omp for schedule(static, 1) nowait
        for (long y = 0; y < LSIDE; ++y) {
            for (long zb = 0; zb < LSIDE / PW; ++zb) {
                double *base = vout + (y * 36 + zb * PW) * 2;
                if (pf)
                    for (int i = 0; i < 36; ++i)
                        _mm_prefetch((const char *)(base + i * (NPLANE * 2) + pf * 8),
                                     _MM_HINT_T0);
                FN(dft36_x)(base);
            }
        }
#endif
    }
}

static void XCAT(XCAT(split_, VAR), _p0)(const double *in, double *out,
                                         long batch, const fft3d_plan *p)
{
#pragma omp parallel num_threads(p->T)
    FN(split_body)(in, out, batch, p, 0);
}

static void XCAT(XCAT(split_, VAR), _p1)(const double *in, double *out,
                                         long batch, const fft3d_plan *p)
{
#pragma omp parallel num_threads(p->T)
    FN(split_body)(in, out, batch, p, 1);
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

static void FNP5(exec)(const double *in, double *out, long batch, double *plane)
{
    FN(body_zy)(in, out, batch, plane, 0);
}

static void FNP6(exec)(const double *in, double *out, long batch, double *plane)
{
    FN(body_zy)(in, out, batch, plane, 1);
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
#undef ST1G
#undef ST1
#undef ST2G
#undef ST2

#endif /* VAR */
