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
 *   ICE_R2 -- RETUNED FOR THE ICE LAKE PANEL (bare-metal Xeon Gold 6326,
 *   2x512-bit FMA pipes, graded workload = cases.txt 36:8:64, a unitary
 *   chain whose three 5.97 MB buffers are L3-resident and whose driver-side
 *   scale pass rereads/rewrites the whole output after EVERY step).  What
 *   changed, in order of measured effect:
 *   * pind DEFAULT 2112 + plan-time residue race (stage B).  The round's
 *     discovery: on ICX the y-subloop's plane-scratch loads false-alias
 *     (4K) against its own in-flight dst stores badly enough that pinning
 *     (pout - pl) mod 4096 = 2112 read -22% END-TO-END on the node's B=1
 *     graded chain (122.5 -> 95.4 us/xform, MKL steady both runs); the tsc
 *     phase split moved y 38.4% -> 32.7%.  On CLX the same knob was 0 to
 *     -1.2% -- machine-specific, now default-on here.  At B=8 it prices
 *     flat (the y-subloop is RFO-drain-bound there and alias stalls hide
 *     under it); kept because the race costs ~0.3 s and B=1-like cells win.
 *     2112 = 64*(1+9*32): colliding row pairs sit 32 rows apart, stores
 *     long retired.
 *   * settle: ~150 ms of dependent FP at the top of fft3d_create() so the
 *     tuner ranks candidates on a ramped core (schedutil governor;
 *     adopted from L17_winograd via L17_matrixsimd ice_r1, attributed).
 *   * chain-shaped tuner: cached-regime candidates are timed under the
 *     driver's own loop -- nt = min(batch, 8) volumes, output unitarily
 *     scaled after every step and fed back as the next input, ping-ponging
 *     two full-size destination arenas -- with only the execute() spans
 *     accumulated.  The old fresh-src arena measured a milder regime;
 *     ice_r1's 12%-spread pf0 pick came out of it.  (Adopted from
 *     L17_matrixsimd ice_r1 stage 1g, attributed.)
 *   * mechanism pool retuned for ICX: pf1 is the incumbent (beat pf0 in
 *     every window measured); the write-intent family (pfw, pfin-pfw,
 *     l1-pfw) is fielded at every batch and wins 25-30% under memory
 *     contention while pricing ~flat in quiet windows -- the tuner sees the
 *     same conditions as the imminent run, so the pick adapts.  l1 (codes
 *     8/9) is a two-level read-side composite: far T1 cursor (32 KB, L3->
 *     L2) + near T0 cursor (2 z-blocks, L2->L1) on src, plus the next
 *     y-call's 36 scratch lines T0 (half of pl is L2 by then).  nta (codes
 *     5/6, PREFETCHNTA L2-bypass) stays compiled for FORCEPICK A/Bs but is
 *     not fielded: it priced null everywhere (the L1-eviction loss cancels
 *     the L2-capacity win).
 *
 *   RETIRED ON ICE: the zy cross-plane z/y call interleave (geom panel_r11's
 *   bet, built for a 1-FMA-pipe CLX where port 5 rode free).  The ice_r1
 *   node probe priced it at +18% (pf0=116.9 vs zy=137.9 us/vol in-arena) --
 *   on a 2x512-pipe part the y-call FMAs it donates to the z-transpose
 *   window are no longer free, they were the second pipe's food.  Also
 *   null here: the pre-RA scheduling pragma (corpus SS10 predicted it hurts
 *   the 36/45/64 class; MKL-normalized A/B agreed), and bare nta/l1 (see
 *   above).
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

/* nta cursor lead, in doubles: 576 = 72 lines = 2 z-blocks ahead of the
 * z-subloop's consumption point.  Short on purpose -- NTA lines live in L1
 * (not L2), and a longer lead has them evicted by the y-subloop's 40 KB of
 * scratch+store churn before use.  -DFFT36_NTA_LEAD=... to sweep. */
#ifndef FFT36_NTA_LEAD
#define FFT36_NTA_LEAD 576
#endif

typedef void (*exec_fn)(const double *, double *, long, double *);

/* -DFFT36_TSC: dev-only phase-split accounting (perf_event_open is blocked
 * on the node, paranoid=4).  Accumulates rdtsc cycles over every execution
 * (tuner included); fft3d_destroy prints the split.  ~1.6% overhead. */
#ifdef FFT36_TSC
static unsigned long long g_tsc[3];   /* z-subloop, y-subloop, phase 2 */
#endif

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
 *   3 = code 1 + paced phase-1 T1 input prefetch   (pfin)
 *   4 = code 3 + paced phase-1 prefetchw on out    (pfin-pfw)
 *   5 = code 1 + paced phase-1 NTA input prefetch  (nta)
 *   6 = code 5 + paced phase-1 prefetchw on out    (nta-pfw)
 *   7 = code 1 + paced phase-1 prefetchw on out    (pfw, write-intent only)
 *   8 = code 1 + short-lead T0 on src (z-subloop) and on the plane scratch
 *       one y-call ahead (y-subloop) -- converts exposed L2-hit latency on
 *       the two phase-1 read streams into L1 hits               (l1)
 *   9 = code 8 + paced phase-1 prefetchw on out    (l1-pfw)
 */
static void exec_0_0(const double *, double *, long, double *);
static void exec_0_1(const double *, double *, long, double *);
static void exec_0_2(const double *, double *, long, double *);
static void exec_0_3(const double *, double *, long, double *);
static void exec_0_4(const double *, double *, long, double *);
static void exec_0_5(const double *, double *, long, double *);
static void exec_0_6(const double *, double *, long, double *);
static void exec_0_7(const double *, double *, long, double *);
static void exec_0_8(const double *, double *, long, double *);
static void exec_0_9(const double *, double *, long, double *);
static void exec_1_0(const double *, double *, long, double *);
static void exec_1_1(const double *, double *, long, double *);
static void exec_1_2(const double *, double *, long, double *);
static void exec_1_3(const double *, double *, long, double *);
static void exec_1_4(const double *, double *, long, double *);
static void exec_1_5(const double *, double *, long, double *);
static void exec_1_6(const double *, double *, long, double *);
static void exec_1_7(const double *, double *, long, double *);
static void exec_1_8(const double *, double *, long, double *);
static void exec_1_9(const double *, double *, long, double *);

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

/* ~150 ms of dependent FP work so the tuner's rankings run on a ramped core:
 * the node's schedutil governor otherwise leaves part of the plan at (or
 * below) the 2.9 GHz base and the rankings inherit the ramp.  Scalar on
 * purpose -- the common section carries no ISA pragma, and ICX has no
 * AVX-512 licence cliff to warm through (corpus §10, three sessions).
 * Adopted from L17_winograd's tuner protocol via L17_matrixsimd ice_r1. */
static void settle_spin(void)
{
    volatile double v0 = 1.0000001, v1 = 1.0000002,
                    v2 = 1.0000003, v3 = 1.0000004;
    double t0 = now_s();
    do {
        double a = v0, b = v1, c = v2, d = v3;
        for (int i = 0; i < 40000; ++i) {
            a = a * 1.0000001 + 1e-9; b = b * 1.0000001 + 1e-9;
            c = c * 1.0000001 + 1e-9; d = d * 1.0000001 + 1e-9;
        }
        v0 = a; v1 = b; v2 = c; v3 = d;
    } while (now_s() - t0 < 0.15);
}

/* -DFFT36_PMU: dev-only port/stall accounting for the picked body, printed
 * to stderr from create() (the node exposes perf_event_open; the perf tool
 * is absent).  Events are Ice Lake-SP encodings. */
#ifdef FFT36_PMU
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
static int pmu_open(uint64_t config, int group)
{
    struct perf_event_attr a;
    memset(&a, 0, sizeof a);
    a.size = sizeof a;
    a.type = config == (uint64_t)-1 ? PERF_TYPE_HARDWARE : PERF_TYPE_RAW;
    a.config = config == (uint64_t)-1 ? PERF_COUNT_HW_CPU_CYCLES : config;
    a.disabled = group < 0;
    a.exclude_kernel = 1;
    a.exclude_hv = 1;
    return (int)syscall(SYS_perf_event_open, &a, 0, -1, group, 0);
}
static void pmu_probe(exec_fn fn, const double *src, double *dst, long nt,
                      double *plane, const char *tag)
{
    /* cycles + UOPS_DISPATCHED.PORT_0 / PORT_5 / PORT_2_3(loads) /
     * PORT_4_9(stores) + CYCLE_ACTIVITY.STALLS_MEM_ANY (0x14a3 cmask hack:
     * event 0xA3 umask 0x14 cmask 20 -> use raw 0x145314a3? too fragile --
     * keep the five dispatch/cycle counters, they answer port vs stall). */
    int lead = pmu_open((uint64_t)-1, -1);
    if (lead < 0) { fprintf(stderr, "[fft36 pmu] unavailable\n"); return; }
    int p0 = pmu_open(0x01a1, lead), p5 = pmu_open(0x20a1, lead);
    int ld = pmu_open(0x04a1, lead), st = pmu_open(0x10a1, lead);
    fn(src, dst, nt, plane);                       /* warm */
    ioctl(lead, PERF_EVENT_IOC_RESET,  PERF_IOC_FLAG_GROUP);
    ioctl(lead, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);
    for (int r = 0; r < 4; ++r) fn(src, dst, nt, plane);
    ioctl(lead, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP);
    long long c = 0, v0 = 0, v5 = 0, vl = 0, vs = 0;
    (void)!read(lead, &c, 8);
    if (p0 >= 0) (void)!read(p0, &v0, 8);
    if (p5 >= 0) (void)!read(p5, &v5, 8);
    if (ld >= 0) (void)!read(ld, &vl, 8);
    if (st >= 0) (void)!read(st, &vs, 8);
    double pv = 4.0 * (double)nt;
    fprintf(stderr,
            "[fft36 pmu] %s: cyc/vol=%.0f p0/vol=%.0f p5/vol=%.0f "
            "ld/vol=%.0f st/vol=%.0f  p0util=%.2f p5util=%.2f\n",
            tag, c / pv, v0 / pv, v5 / pv, vl / pv, vs / pv,
            v0 / (double)c, v5 / (double)c);
    if (p0 >= 0) close(p0);
    if (p5 >= 0) close(p5);
    if (ld >= 0) close(ld);
    if (st >= 0) close(st);
    close(lead);
}
#endif

fft3d_plan *fft3d_create(int L, int batch)
{
    if (L != LSIDE || batch < 1) return NULL;

    fft3d_plan *p = (fft3d_plan *)calloc(1, sizeof *p);
    if (!p) return NULL;
    p->batch = batch;
    /* 4 KB of slack + page alignment so execute can pin (pout - pl) mod 4096
     * to g_pind while keeping every access 64-byte aligned. */
    void *pl = NULL;
    if (posix_memalign(&pl, 4096,
                       (size_t)(512 + NPLANE * 2) * sizeof(double)) != 0 || !pl) {
        free(p);
        return NULL;
    }
    p->plane = (double *)pl;
    p->raw = pl;
    p->fn = exec_0_0;

    /* anti-alias pin target, read ONCE at plan time so execution stays
     * repeatable: FFT36_PIND=<bytes> (rounded down to a line), -1 = pinning
     * off (pl = plane always).  Unset = OFF everywhere since r9: the r8 node
     * run priced always-on cached-regime pinning at 0 to -1.2% at B=1, and
     * the allocator-lottery rationale is dead (L17_matrixsimd r8: glibc's
     * mmap'd buffers give fixed relative offsets across processes).  An
     * explicit env value is absolute so the monitor can A/B any cell. */
    int pind_forced = 0;
    {
#ifdef FFT36_PIND_DEF
        g_pind = (long)(FFT36_PIND_DEF) < 0
                     ? -1 : ((long)(FFT36_PIND_DEF) & 4095l & ~63l);
        pind_forced = 1;
#endif
        const char *pe = getenv("FFT36_PIND");
        if (pe && *pe) {
            long v = strtol(pe, NULL, 0);
            g_pind = v < 0 ? -1 : (v & 4095l & ~63l);
            pind_forced = 1;
        }
        /* Unforced default: 2112 (the ice_r2 discovery -- see the strategy
         * record: pinning the y-subloop's (pout - pl) mod 4096 residue read
         * -22% END-TO-END on the node's graded chain at B=1, 122.5 -> 95.4
         * us/xform, MKL steady in both runs).  2112 = 64*(1 + 9*32): the
         * colliding load/store row pairs sit 32 rows apart, so the aliased
         * stores are ~36 stores old and long retired.  Stage B below races
         * the residue; this is the incumbent. */
        if (!pind_forced) g_pind = 2112;
    }

    /* ramp the core BEFORE anything is timed (see settle_spin) */
    settle_spin();

    /* ---- regime: does the batch stream through this machine's LLC?  This
     * decides only which candidates are IN PLAY and how the tuning arena is
     * sized.  Store policy itself is no longer a question: NT stores lost
     * every node tournament for four consecutive geom rounds, and on the ice
     * panel's L3-resident chain L17_matrixsimd measured them at 2x (28.9 vs
     * 14.9 us/step) -- retired. */
    long l3 = -1;
    {
#ifdef _SC_LEVEL3_CACHE_SIZE
        l3 = sysconf(_SC_LEVEL3_CACHE_SIZE);
#endif
        if (l3 <= 0) l3 = 22l << 20;
    }
    double foot = (double)batch * (double)NVOL * 16.0 * 2.0;
    int streaming = foot > 1.25 * (double)l3;

    /* ---- self-tuning.  All of this is setup, hence excluded from the score.
     * Every candidate except the reference must match exec_0_0's output to
     * 1e-13 relative before it is eligible (all bodies here are one bit
     * class -- same calls, same order per volume -- so this is a hard
     * equality up to admission noise).
     *
     * Arena: in the CACHED regime the arena is the GRADED SHAPE -- nt =
     * min(batch, 8) volumes timed as a ping-pong chain, output unitarily
     * scaled after every step exactly as the driver does, with only the
     * execute() spans accumulated.  The old fresh-src single-dst arena
     * measured a milder regime (clean input, no dirty ping-pong, no scale
     * pass thrash) and produced ice_r1's 12%-spread pf0 pick.  In the
     * streaming regime the arena must actually STREAM on the machine doing
     * the tuning (L36_pfa's round-2 lesson): sized off THIS machine's L3,
     * clamped to [32, 128] volumes and the batch. */
    exec_fn cand[32];
    const char *cnm[32];
    int cinst[32];
    int ncand = 0;

    long nt;
    if (streaming) {
        long arena = (long)(2.5 * (double)l3 / ((double)NVOL * 32.0)) + 1;
        if (arena < 32)  arena = 32;
        if (arena > 128) arena = 128;
        nt = batch < arena ? batch : arena;
    } else {
        nt = batch < 8 ? batch : 8;
    }
    size_t nd = (size_t)NVOL * 2 * (size_t)nt;
    double *ti = NULL, *o0 = NULL, *ox = NULL, *tc = NULL;
    if (posix_memalign((void **)&ti, 4096, nd * sizeof(double)) == 0 &&
        posix_memalign((void **)&o0, 4096, nd * sizeof(double)) == 0 &&
        posix_memalign((void **)&ox, 4096, nd * sizeof(double)) == 0 &&
        posix_memalign((void **)&tc, 4096, nd * sizeof(double)) == 0) {

        uint64_t s = 0x9E3779B97F4A7C15ull;
        for (size_t i = 0; i < nd; ++i) {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            ti[i] = (double)(int64_t)(s >> 11) * (1.0 / 9007199254740992.0);
        }
        memset(o0, 0, nd * sizeof(double));
        exec_0_0(ti, o0, nt, p->plane);

        int have_512 = 0;
#if defined(__x86_64__)
        __builtin_cpu_init();
        have_512 = __builtin_cpu_supports("avx512f");
#endif
        /* diagnostic overrides, read once at plan time so execution stays
         * repeatable.  Env for interactive node A/Bs: FFT36_PFIN=0|1,
         * FFT36_PFW=0|1, FFT36_NTA=0|1 drop/force those mechanism families.
         * tryout.sh cannot pass env through ssh (L17_matrixsimd ice_r1), so
         * dev A/Bs use compile hooks instead: -DFFT36_FORCEPICK=<code 0..7>
         * installs that exec code unconditionally (after the tournament, so
         * the probe string still reports), -DFFT36_VERBOSE prints the whole
         * tuner table to stderr.  (FFT36_ZY died with zy this round --
         * ice_r1 node-closed at +18%.) */
        int pfinmode = -1, pfwmode = -1, ntamode = -1;
        {
            const char *po = getenv("FFT36_PFIN");
            if (po && (*po == '0' || *po == '1')) pfinmode = *po - '0';
            const char *wo = getenv("FFT36_PFW");
            if (wo && (*wo == '0' || *wo == '1')) pfwmode = *wo - '0';
            const char *no = getenv("FFT36_NTA");
            if (no && (*no == '0' || *no == '1')) ntamode = *no - '0';
        }
        int in_plain = (pfinmode != 1) && (pfwmode != 1) && (ntamode != 1);
        int in_pfin  = (pfinmode != 0) && (ntamode != 1);
        int in_pfw   = (pfwmode != 0)  && (ntamode != 1) && (pfinmode != 1);
        int in_nta   = (ntamode != 0)  && (pfinmode != 1) && (pfwmode != 1);

        exec_fn probe[32];
        const char *pnm[32];
        int pin_[32];
        int nprobe = 0;
#define PROBE(fn, nm, inst) do { probe[nprobe] = (fn); pnm[nprobe] = (nm); \
                                 pin_[nprobe] = (inst); ++nprobe; } while (0)
        /* Candidate order = hysteresis order: simplest mechanism first (a
         * speculative mechanism must beat the incumbent by > 3% to install).
         * V1 only when AVX-512 exists: it has won every node cell in every
         * round since geom r1, and on 2x512-pipe ICX L17_matrixsimd measured
         * every 256-bit variant at +34% -- V0 is the no-AVX-512 fallback,
         * not a candidate. */
        if (have_512) {
            if (streaming) {
                if (in_plain) PROBE(exec_1_1, "v1-pf1", 1);
                if (in_pfw)   PROBE(exec_1_7, "v1-pfw", 1);
                if (in_pfin)  PROBE(exec_1_3, "v1-pfin", 1);
                if (in_pfin && pfwmode != 0) PROBE(exec_1_4, "v1-pfin-pfw", 1);
            } else {
                /* Cached pool, hysteresis order from the ice_r2 window
                 * sweeps: pf1 first (it beat pf0 in every window measured,
                 * quiet and loaded, by 1.5-3%), then pf0, then the
                 * mechanism composites.  pfw is fielded at EVERY batch on
                 * this panel: the ice B=1 tsc split put the y-subloop (the
                 * store-into-cold-dst one) at 336 cyc/call vs its
                 * identical-arithmetic x twin's 197 -- the RFO drain is
                 * exposed even with all three chain buffers cache-resident,
                 * because dst was last touched a full step ago.  Under
                 * core-lease contention the pfw family won by 25-30%
                 * (pf0 162.0 vs pfin-pfw 122.2, l1-pfw 119.3): the tuner
                 * runs under the same conditions as the imminent run, so
                 * the pick adapts per window.  pf4, bare nta, and bare l1
                 * never led a window and were trimmed (still compiled;
                 * FFT36_FORCEPICK can field them). */
                if (in_plain) {
                    PROBE(exec_1_1, "v1-pf1", 1);
                    PROBE(exec_1_0, "v1-pf0", 1);
                }
                if (in_pfin)               PROBE(exec_1_3, "v1-pfin", 1);
                if (in_pfw)                PROBE(exec_1_7, "v1-pfw", 1);
                if (in_pfin && pfwmode != 0)
                                           PROBE(exec_1_4, "v1-pfin-pfw", 1);
                if (in_nta && pfwmode != 0)
                                           PROBE(exec_1_9, "v1-l1-pfw", 1);
            }
        } else {
            if (streaming) {
                if (in_plain) PROBE(exec_0_1, "v0-pf1", 1);
                if (in_pfw)   PROBE(exec_0_7, "v0-pfw", 1);
                if (in_pfin)  PROBE(exec_0_3, "v0-pfin", 1);
                if (in_pfin && pfwmode != 0) PROBE(exec_0_4, "v0-pfin-pfw", 1);
            } else {
                if (in_plain) {
                    PROBE(exec_0_1, "v0-pf1", 1);
                    PROBE(exec_0_0, "v0-pf0", 1);
                }
                if (in_pfin)               PROBE(exec_0_3, "v0-pfin", 1);
                if (in_pfw)                PROBE(exec_0_7, "v0-pfw", 1);
                if (in_pfin && pfwmode != 0)
                                           PROBE(exec_0_4, "v0-pfin-pfw", 1);
                if (in_nta && pfwmode != 0)
                                           PROBE(exec_0_9, "v0-l1-pfw", 1);
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
                cinst[ncand] = pin_[k];
                ++ncand;
            }
        }
        if (ncand == 0) {           /* nothing survived admission: fall back */
            cand[ncand] = exec_0_0; cnm[ncand] = "v0-pf0-fallback";
            cinst[ncand] = 1; ++ncand;
            cand[ncand] = exec_0_1; cnm[ncand] = "v0-pf1-fallback";
            cinst[ncand] = 1; ++ncand;
        }

        /* CHAIN-SHAPED TIMING: every candidate runs the driver's own loop --
         * 1 warm step + S timed steps, dst unitarily scaled after every step
         * and ping-ponged back as the next src.  Only the execute() spans
         * are accumulated: the scale pass shapes the cache/dirty state (that
         * is its job here) but is the same additive constant for every
         * candidate and would only dilute the contrast.  Rounds interleave
         * candidates to decorrelate slow drift; keep the per-candidate
         * minimum round. */
        double best[32];
        for (int k = 0; k < ncand; ++k) best[k] = 1e300;
        const double usc = 1.0 / 216.0;   /* 1/sqrt(36^3), the driver's */
        int S      = streaming ? 2 : (nt >= 8 ? 4 : 8);
        int rounds = streaming ? 3 : (nt >= 8 ? 6 : 8);
        for (int round = 0; round < rounds; ++round) {
            for (int k = 0; k < ncand; ++k) {
                const double *src = ti;
                double *dst = ox, *alt = tc;
                double acc = 0.0;
                for (int st = 0; st <= S; ++st) {
                    double t0 = now_s();
                    cand[k](src, dst, nt, p->plane);
                    double t1 = now_s();
                    if (st > 0) acc += t1 - t0;
                    for (size_t j = 0; j < nd; ++j) dst[j] *= usc;
                    src = dst;
                    double *t = dst == ox ? alt : ox;
                    alt = dst; dst = t;
                }
                if (acc < best[k]) best[k] = acc;
            }
        }
        /* hysteresis pick: candidates are listed simplest-first, so a later
         * (more speculative) candidate must beat the incumbent by more than
         * 3% to be installed.  Near-ties go to the simpler code. */
        int bk = -1;
        for (int k = 0; k < ncand; ++k) {
            if (!cinst[k]) continue;
            if (bk < 0 || best[k] < 0.97 * best[bk]) bk = k;
        }
        if (bk < 0) bk = 0;
        p->fn = cand[bk];

        /* ---- stage B: race the anti-alias pin residue on the picked body.
         * pind is bits-neutral (it slides a scratch whose contents are
         * identical), so this is an ordinary in-class knob.  2112 is the
         * incumbent (listed first; alternates must clear the 3% bar): it
         * puts the colliding 576-B-stride load/store row pairs 32 rows
         * apart (stores long retired).  1536 / 2688 are the m=+31/-31
         * neighbours, -1 is the unpinned control.  Skipped when the env /
         * -D override forces a residue (monitor A/Bs stay absolute). */
        static const long pgrid[4] = { 2112, -1, 1536, 2688 };
        double pbest[4] = { 1e300, 1e300, 1e300, 1e300 };
        if (!pind_forced) {
            const long saved = g_pind;
            for (int round = 0; round < rounds; ++round) {
                for (int i = 0; i < 4; ++i) {
                    g_pind = pgrid[i];
                    const double *src = ti;
                    double *dst = ox, *alt = tc;
                    double acc = 0.0;
                    for (int st = 0; st <= S; ++st) {
                        double t0 = now_s();
                        cand[bk](src, dst, nt, p->plane);
                        double t1 = now_s();
                        if (st > 0) acc += t1 - t0;
                        for (size_t j = 0; j < nd; ++j) dst[j] *= usc;
                        src = dst;
                        double *t = dst == ox ? alt : ox;
                        alt = dst; dst = t;
                    }
                    if (acc < pbest[i]) pbest[i] = acc;
                }
            }
            g_pind = saved;
            int pb = 0;
            for (int i = 1; i < 4; ++i)
                if (pbest[i] < 0.97 * pbest[pb]) pb = i;
            g_pind = pgrid[pb];
        }
#ifdef FFT36_PMU
        pmu_probe(cand[bk], ti, ox, nt, p->plane, cnm[bk]);
#endif
#ifdef FFT36_VERBOSE
        for (int k = 0; k < ncand; ++k)
            fprintf(stderr, "[fft36 tuner] %-14s %8.2f us/vol%s\n", cnm[k],
                    best[k] / S / nt * 1e6, k == bk ? "  <== pick" : "");
        if (!pind_forced)
            fprintf(stderr, "[fft36 tuner] pind 2112/-1/1536/2688 = "
                    "%.2f/%.2f/%.2f/%.2f -> %ld\n",
                    pbest[0] / S / nt * 1e6, pbest[1] / S / nt * 1e6,
                    pbest[2] / S / nt * 1e6, pbest[3] / S / nt * 1e6, g_pind);
#endif
#ifdef FFT36_FORCEPICK
        {
            exec_fn f0tab[10] = { exec_0_0, exec_0_1, exec_0_2, exec_0_3,
                                  exec_0_4, exec_0_5, exec_0_6, exec_0_7,
                                  exec_0_8, exec_0_9 };
            exec_fn f1tab[10] = { exec_1_0, exec_1_1, exec_1_2, exec_1_3,
                                  exec_1_4, exec_1_5, exec_1_6, exec_1_7,
                                  exec_1_8, exec_1_9 };
            p->fn = have_512 ? f1tab[FFT36_FORCEPICK % 10]
                             : f0tab[FFT36_FORCEPICK % 10];
        }
#endif
        /* probe-through-description: publish the chain-arena execute-only
         * times of the mechanism corners (pf0 incumbent, write-intent, NTA
         * L2-bypass, and their composite) so the node's memory-mechanism
         * verdict rides the leaderboard whatever the pick. */
        double t_p = -1.0, t_w = -1.0, t_n = -1.0, t_nw = -1.0;
        for (int k = 0; k < ncand; ++k) {
            const char *s = strchr(cnm[k], '-');
            if (!s) continue;
            double us = best[k] / S / nt * 1e6;
            if (strcmp(s, "-pf0") == 0)      t_p  = us;
            if (strcmp(s, "-pfw") == 0)      t_w  = us;
            if (strcmp(s, "-pfin-pfw") == 0) t_n  = us;
            if (strcmp(s, "-l1-pfw") == 0)   t_nw = us;
        }
        int n = snprintf(g_desc, sizeof g_desc,
                 "PFA 4x9 2-sweep, lanes=lines, n1_9 DFT9, chain-tuned; "
                 "pick=%s (B=%d, arena=%ld vol, stream=%d, %d cand, pinD=%ld)"
#ifdef FFT36_FORCEPICK
                 " FORCED=%d"
#endif
                 , cnm[bk], batch, nt, streaming, ncand, g_pind
#ifdef FFT36_FORCEPICK
                 , (int)(FFT36_FORCEPICK & 7)
#endif
                 );
        if (t_p > 0.0 && n > 0 && (size_t)n < sizeof g_desc)
            snprintf(g_desc + n, sizeof g_desc - (size_t)n,
                     " ex us/vol pf0=%.1f pfw=%.1f ppw=%.1f l1w=%.1f",
                     t_p, t_w, t_n, t_nw);
    }
    free(ti); free(o0); free(ox); free(tc);
    return p;
}

void fft3d_execute(fft3d_plan *plan, const double _Complex *in, double _Complex *out)
{
    plan->fn((const double *)in, (double *)out, plan->batch, plan->plane);
}

void fft3d_destroy(fft3d_plan *plan)
{
    if (!plan) return;
#ifdef FFT36_TSC
    {
        unsigned long long tot = g_tsc[0] + g_tsc[1] + g_tsc[2];
        if (tot)
            fprintf(stderr,
                    "[fft36 tsc] z=%.1f%% y=%.1f%% p2=%.1f%% of %llu cycles\n",
                    100.0 * g_tsc[0] / tot, 100.0 * g_tsc[1] / tot,
                    100.0 * g_tsc[2] / tot, tot);
    }
#endif
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
#define FNP7(n) XCAT(XCAT(n##_, VAR), _7)
#define FNP8(n) XCAT(XCAT(n##_, VAR), _8)
#define FNP9(n) XCAT(XCAT(n##_, VAR), _9)

/* -DFFT36_SCHED: pre-RA instruction scheduling on the kernel bodies (GCC
 * does none on x86 by default, so straight-line codelets reach the issue
 * queue in text order).  Corpus §10: +20% on prime passes but HURTS 45/64;
 * L17_matrixsimd ice_r1 measured -7.7% on their chunk kernels.  A/B hook. */
#ifdef FFT36_SCHED
#pragma GCC optimize("schedule-insns", "sched-pressure")
#endif

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
    /* pfin == 1: paced T1 prefetch of the phase-1 input stream (L36_pfa
     * r3's PFIN, attributed).  A cursor runs PFIN_D doubles = 32 KB ahead
     * of the plane being consumed; each of the 2*(36/PW) codelet calls per
     * plane issues PFIN_L line-prefetches and advances, so exactly one
     * plane of prefetches issues per plane processed.
     *
     * pfin == 2: paced NTA prefetch of the same stream, issued from the
     * z-subloop ONLY (the y-subloop reads just the L1 plane scratch), 36
     * lines per z-call = exactly the 4 rows the z-call two blocks later
     * consumes, FFT36_NTA_LEAD doubles ahead.  NTA fills L1+LLC and skips
     * L2 on SKX/ICX, so the once-read src stream stops evicting the dst
     * volume from the 1.25 MB L2 (in+out = 1.46 MB/volume) and phase 2's
     * in-place reread stays L2-hit.  The short lead is deliberate: these
     * lines are L1-or-L3, and the y-subloop churns ~40 KB between planes.
     *
     * pfw: paced WRITE-INTENT cursor over phase 1's store stream into cold
     * `out`, PFW_D doubles ahead, advancing at the consumption rate.
     * __builtin_prefetch(p,1,3) emits `prefetchw`, acquiring the line
     * exclusive before the store so the RFO overlaps compute.  (L36_pfa
     * r5's PFWMID; ultimately L6_unrolled r3; unbundled-from-pfin shape is
     * L36_pfa r11's.) */
#define PFIN_D 4096                    /* T1 read cursor distance = 32 KB   */
#define PFW_D  FFT36_PFW_DIST          /* write cursor distance, default 1 plane */
#define PFIN_L (36 * PW / 8)           /* lines per codelet call: 18 / 9    */
#define NTA_L  36                      /* lines per z-call: one 4-row block */
    const double *pfp = 0, *pfend = 0, *pfq = 0;
    if (pfin) {
        pfend = in + batch * (long)NVOL * 2;
        pfp   = in + (pfin == 2 ? FFT36_NTA_LEAD : PFIN_D);
        if (pfp > pfend) pfp = pfend;
        pfq   = in + FFT36_NTA_LEAD;      /* near T0 cursor (pfin == 3) */
        if (pfq > pfend) pfq = pfend;
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

#ifdef FFT36_TSC
            unsigned long long tsc0 = __rdtsc();
#endif
            for (long yb = 0; yb < LSIDE / PW; ++yb) {
                if (pfin == 1 || pfin == 3) {
                    long npl = (pfend - pfp) / 8;
                    if (npl > PFIN_L) npl = PFIN_L;
                    for (long i = 0; i < npl; ++i)
                        _mm_prefetch((const char *)(pfp + i * 8), _MM_HINT_T1);
                    pfp += npl * 8;
                } else if (pfin == 2) {
                    long npl = (pfend - pfp) / 8;
                    if (npl > NTA_L) npl = NTA_L;
                    for (long i = 0; i < npl; ++i)
                        _mm_prefetch((const char *)(pfp + i * 8),
                                     _MM_HINT_NTA);
                    pfp += npl * 8;
                }
                if (pfin == 3) {
                    /* near T0 cursor: the 4 rows the z-call two blocks
                     * later consumes, already L2 via the far cursor */
                    long npl = (pfend - pfq) / 8;
                    if (npl > NTA_L) npl = NTA_L;
                    for (long i = 0; i < npl; ++i)
                        _mm_prefetch((const char *)(pfq + i * 8), _MM_HINT_T0);
                    pfq += npl * 8;
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
#ifdef FFT36_TSC
            unsigned long long tsc1 = __rdtsc();
            g_tsc[0] += tsc1 - tsc0;
#endif
            for (long zb = 0; zb < LSIDE / PW; ++zb) {
                if (pfin == 1 || pfin == 3) {
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
                if (pfin == 3 && zb + 1 < LSIDE / PW) {
                    /* next y-call's 36 scratch lines (one 64-B line per
                     * 576-B row): half of pl has usually been evicted to L2
                     * by this subloop's own store stream by now */
                    const double *q = pl + (zb + 1) * PW * 2;
                    for (int k = 0; k < 36; ++k)
                        _mm_prefetch((const char *)(q + k * 72), _MM_HINT_T0);
                }
                FN(dft36_y)(pl + zb * PW * 2, pout + zb * PW * 2);
            }
#ifdef FFT36_TSC
            g_tsc[1] += __rdtsc() - tsc1;
#endif
        }

        /* pfin cold-window pre-coverage (L36_pfa r3's PFNX): the paced T1
         * cursor leaves only the first 32 KB of in[b+1] exposed to phase-2
         * eviction; 3 lines per 36-line tile group x 324 groups = 62 KB
         * re-covers it from phase 2, whose own read stream is cache-resident.
         * (T1 path only: the NTA cursor's lead is 4.6 KB -- nothing worth
         * covering, and T1 fills here would re-pollute the L2 it protects.) */
        const double *ncw = ((pfin == 1 || pfin == 3) && b + 1 < batch)
                                ? in + (b + 1) * (long)NVOL * 2
                                : (const double *)0;

        /* -------- phase 2: x-lines, in place in `out` -------------------- */
#ifdef FFT36_TSC
        unsigned long long tsc2 = __rdtsc();
#endif
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
#ifdef FFT36_TSC
        g_tsc[2] += __rdtsc() - tsc2;
#endif
    }
#undef PFIN_D
#undef PFW_D
#undef PFIN_L
#undef NTA_L
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
    FN(body)(in, out, batch, plane, 1, 2, 0);
}

static void FNP6(exec)(const double *in, double *out, long batch, double *plane)
{
    FN(body)(in, out, batch, plane, 1, 2, 1);
}

static void FNP7(exec)(const double *in, double *out, long batch, double *plane)
{
    FN(body)(in, out, batch, plane, 1, 0, 1);
}

static void FNP8(exec)(const double *in, double *out, long batch, double *plane)
{
    FN(body)(in, out, batch, plane, 1, 3, 0);
}

static void FNP9(exec)(const double *in, double *out, long batch, double *plane)
{
    FN(body)(in, out, batch, plane, 1, 3, 1);
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
#undef FNP7
#undef FNP8
#undef FNP9
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
