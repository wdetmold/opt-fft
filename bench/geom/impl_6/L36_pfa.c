/* L36_pfa.c -- forward complex 3D DFT of a 36^3 cube, batched, out-of-place.
 *
 * ROUND panel_r2.  The round-1 version of this file (three passes over the
 * volume through a slab-blocked 830 KB intermediate, split-complex SoA) lost
 * to L36_mixedradix on the node, 225 vs 118 us at B=1, because it crosses
 * memory three times where two suffice.  This rewrite adopts the structure
 * that won and adds what it lacked in the batched regime.
 *
 * TECHNIQUE
 *   Row-column 3D DFT; every 36-point line is a Good-Thomas / prime-factor
 *   4 x 9 codelet (zero twiddles between the stages), on INTERLEAVED complex
 *   vectors whose lanes are a spectator axis.  Two sweeps over the volume:
 *
 *   phase 1, per x-plane (36x36 complex = 20.25 KB, L1-resident):
 *       z transform: lanes = PW y-rows, via PWxPW complex-granule register
 *                    transposes on load and store (the one unavoidable
 *                    transpose pair), into a plane scratch pl[y][kz]
 *       y transform: lanes = PW kz (contiguous in pl), store to mid[x][ky][kz]
 *   phase 2:
 *       x transform: lanes = PW kz, stride 2592 doubles between x, from mid
 *                    into out (or in place when mid == out)
 *
 *   `mid` is either `out` itself (in-place phase 2: smallest resident set,
 *   746 KB, fits the node's 1 MB L2 -- the L36_mixedradix arrangement) or a
 *   plan-owned scratch volume S that is REUSED for every volume of the batch,
 *   so S stays cache-resident across the whole batch and never costs DRAM
 *   traffic; phase 2 then writes `out` with non-temporal stores.  At large B
 *   that cuts DRAM traffic per volume to the compulsory read-in + write-out
 *   (1.5 MB), where in-place pays an extra RFO + writeback round (2.2 MB).
 *   fft3d_create() times {PW=2, PW=4} x {inplace, scratch, scratch+NT} on a
 *   dummy batch, verifies every candidate against a reference, and installs
 *   the fastest -- so the AVX-512-licence and NT questions are settled by
 *   measurement on the machine that matters, per candidate, per batch size.
 *
 * ATTRIBUTION (round-2 rules: say what you borrowed)
 *   - Two-sweep plane-fused pass structure, interleaved-complex lanes, and the
 *     6-op DFT3 / 8-op DFT4 / 2-op CMUL forms: from L36_mixedradix round 1.
 *   - NT stores on the final write at large batch (+53% at B=32) and the
 *     create-time correctness interlock for variants: from L36_pencilfused.
 *   - The reused-scratch + NT combination and the PFA index maps folded into
 *     compile-time addressing: this file.
 *
 * OPERATION COUNT (per 36-point line over PW lanes, FMA-port vector ops)
 *   9 x DFT4 (8 ops, x(-i) folded into two +-1-pair FMAs)          =  72
 *   4 x DFT9 = CT 3x3: 6 DFT3 (6 ops) + 4 twiddle CMUL (2 ops)     = 176
 *   total 248 FMA-port ops + 49 swaps (port 5) per PW lines
 *   Per volume: 3888 lines -> 241 056 FMA-port ops at PW=4; on a 1-FMA-unit
 *   Gold 5218 that is ~241k cycles = 105 us of pure port-0 work -- the floor.
 *
 * ACCURACY: same modules as round 1, expect ~4e-16 relative L2 vs numpy.
 *
 * ROUND panel_r3: memory-level parallelism for the batched regime.
 *   The r2 node result (B=1 119.3 us vs B=256 238.8 us) means ~119 us/volume of
 *   the batched time is UN-overlapped memory: compulsory traffic is only
 *   1.49 MB/volume and other entries sustain 10.5-12.1 GB/s single-core on this
 *   node (monitor's r2 verdict).  Fix: (a) phase 1 software-prefetches its own
 *   input stream at a fixed byte distance (FFT36_PFD), paced evenly across both
 *   subloops -- the stream is perfectly linear over the volume, but the L2
 *   streamer alone does not hide DRAM across 4 KB page boundaries; (b) phase 2
 *   prefetches the first ~62 KB of the NEXT volume's input, so the phase-1
 *   cursor never starts cold (cross-volume overlap, the shape L6_unrolled/
 *   L6_pfa proved at L=6: next-volume coverage with prefetchtX; NTA is their
 *   documented catastrophe, not retried).  Prefetch is a tuner dimension (pf),
 *   so the B=1 in-place path can keep its r2 behaviour if prefetch costs there.
 *
 * ROUND panel_r4: cross-volume ping-pong pipeline as an ADDITIONAL candidate,
 *   tuner pick reported, tuner hysteresis.
 *   The r3 node result (B=256 227.5 us vs a ~124 us bandwidth ceiling) says the
 *   read burst of phase 1 and the write burst of phase 2 still run serially:
 *   prefetch alone bought 4.7% on the node against 24% on wallaby.  New mode
 *   M_PIPE: two plan-owned scratch volumes S0/S1; while phase 2 of volume b
 *   drains S_a to `out` with NT stores, phase 1 of volume b+1 fills S_b --
 *   interleaved at plane granularity (both phases have exactly 36 outer
 *   iterations per volume), so the DRAM read stream (in[b+1]) and the NT write
 *   stream (out[b]) are busy SIMULTANEOUSLY at ~6.3 GB/s each instead of
 *   alternating 12+ GB/s bursts.  Cost: 1.5 MB of scratch, which no longer fits
 *   the node's 1 MB L2 -- the r3 record deferred this design on exactly that
 *   ground, so it ships as one more tuner candidate (the r3 verdict's process
 *   lesson: ADD candidates, never replace structures), correctness-gated and
 *   selected only if it wins on the machine that matters.
 *   Also per the r3 verdict: fft3d_description() now reports the tuner's pick
 *   (mechanism borrowed from L36_mixedradix/L6_pfa); the tuner prefers the
 *   simplest candidate within 2% of the best time (hysteresis, against the
 *   pf0/pf1 coin-flips the verdict measured costing 3.9-6.7% elsewhere); and
 *   FFT36_PW / FFT36_MODE / FFT36_PF environment variables force the choice at
 *   plan time so the monitor can run forced-variant controls without recompiling.
 *
 * ROUND panel_r5: write-intent prefetch (pf=2), and the per-plane refactor
 *   inlined away.
 *   The r4 node picks settled the mode question: INPLACE + paced read prefetch
 *   won BOTH streaming cells (B=32 174.2 us, B=256 218.9 us, first on the
 *   board) -- every scratch/NT/pipe candidate lost on the machine that counts.
 *   What in-place leaves exposed is the RFO: phase 1 stores to a cold `out`
 *   volume, so every 64-byte line of 746 KB costs a demand read-for-ownership
 *   from DRAM that nothing overlaps.  L6_unrolled's r3 headline (adopted by
 *   L6_pfa, confirmed on the node: `prefetchw` variants selected at all DRAM
 *   batch sizes) is exactly this fix: __builtin_prefetch(p,1,3) emits
 *   `prefetchw` on PRFCHW machines (Cascade Lake and Sapphire Rapids both),
 *   acquiring the line exclusive AHEAD of the store while keeping the
 *   normal-store shape the node demonstrably prefers over NT.  Here it ships
 *   as tuner level pf=2: phase 1 paces a write-intent cursor one plane
 *   (FFT36_PFWD = 2592 doubles) ahead of its mid-plane stores when mid==out
 *   (M_INPLACE), and phase 2 write-prefetches its 36 out-streams one line
 *   ahead when out is store-cold (M_SCRATCH); NT/PIPE modes gain nothing from
 *   it (NT stores do not RFO), so pf=2 is only instantiated for those two
 *   modes.  L6's warning is respected by construction: prefetchw LOST 17% at
 *   cache-resident sizes there, so it enters as a candidate the tournament
 *   must select, never a default.
 *   Second change: phase1_plane/phase2_yplane (the r4 per-plane refactor that
 *   M_PIPE needed) are now always_inline, restoring r3's monolithic code
 *   layout for the non-pipe paths -- the r4 verdict names that refactor the
 *   only candidate cause for this entry's +2.4% B=4 node regression
 *   (129.2 -> 132.3 us against a 0.5% spread, same pick string).
 *
 * ROUND panel_r6: NTA read stream to protect `out`'s L2 residency (pf=3/4).
 *   The r4/r5 verdicts settled the node clock at ~3.89 GHz sustained, so the
 *   B=1 port floor is ~62 us against 120.4 measured -- and the extra ~230k
 *   cycles are node-specific (wallaby's B=1 sits exactly on ITS port-5 model:
 *   120.5k half-FMA + ~95k shuffles = 215k cycles = the 51.4 us measured).
 *   The node's difference from wallaby at B=1 is one number: L2 = 1 MB vs
 *   2 MB, against in+out = 1.5 MB.  The sequential in-read (demand or T1
 *   prefetch) allocates every line into L2, evicting `out` mid-execute, so
 *   the in-place phase-1 stores RFO from L3, phase 2 re-reads from L3, and
 *   modified out lines write back -- 2-3 MB of L2<->L3 round trips per
 *   execute where the compulsory traffic is 746 KB.  Fix: prefetch the in
 *   stream with hint NTA (fills L1, bypasses L2 on SKX-class cores) at a
 *   SHORT constant lead (FFT36_PFDN = 512 doubles = 4 KB), paced at exactly
 *   the consumption rate inside the yb-subloop only (2*PFSTEP per iteration,
 *   nothing in the zb-subloop) so the lead never swings.  A demand load that
 *   finds its line already in L1 allocates nothing into L2, so `out` (746 KB
 *   + the 20 KB plane) stays L2-resident across the whole execute -- and at
 *   B=1, across EXECUTES: steady state leaves out's lines L2-modified, so
 *   the RFO and the writeback disappear entirely and L3 traffic collapses to
 *   the compulsory 746 KB in-read.  The r3 NTA catastrophe (135.1 vs 104.4
 *   at B=256) was a 32 KB lead on DRAM-latency streams -- lines evicted from
 *   L1 before use, and nothing behind them in L2 by design; a 4 KB lead on
 *   L3-resident data is a different regime, and it enters as a gated
 *   candidate the tournament must select, never a default.  Two levels:
 *     pf=3  NTA in-read + the pf=2 write-intent mechanism (streaming cells)
 *     pf=4  NTA in-read alone (B=1/B=4: out is already L2-M in steady state,
 *           so prefetchw is pure uop tax there -- ranked SIMPLER than pf=3)
 *   Both only for INPLACE/SCRATCH; PFNX is off (cache-resident target).
 */
#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <immintrin.h>

#include "fft3d_api.h"

#ifndef L36_PFA_ONCE            /* ============ COMMON, first pass ============ */
#define L36_PFA_ONCE

#define L    36
#define LSQ  1296                /* 36*36                                       */
#define VDBL ((size_t)2 * L * LSQ)   /* doubles per volume = 93312             */

#define CAT_(a,b) a##b
#define CAT(a,b)  CAT_(a,b)

/* phase-1 input prefetch distance, in doubles (32 KB): must exceed the ~10.4 KB
 * pacing deficit the yb-subloop accumulates (it consumes 2x faster than the
 * prefetch cursor advances; the zb-subloop catches up), and stay small enough
 * that the in-flight window never evicts the scratch volume from the node's
 * 1 MB L2. */
#ifndef FFT36_PFD
# define FFT36_PFD 4096
#endif
/* prefetch hint: 3=T0 (all levels), 2=T1 (L2+), 1=T2, 0=NTA */
#ifndef FFT36_PFH
# define FFT36_PFH 2
#endif
/* cache lines of the NEXT volume's input prefetched per phase-2 tile */
#ifndef FFT36_PFN
# define FFT36_PFN 3
#endif
/* pf=2: distance (doubles) the phase-1 write-intent cursor runs ahead of the
 * mid-plane stores.  One plane = 2592 doubles = 20.25 KB gives every line an
 * 18-27-iteration lead (0.5-1.5 planes, ~1.5-4 us) -- long enough to cover a
 * DRAM RFO, short enough that nothing prefetched is evicted before its store. */
#ifndef FFT36_PFWD
# define FFT36_PFWD 2592
#endif
/* pf=3/4: constant lead (doubles) of the NTA in-read cursor.  512 doubles =
 * 4 KB = 64 lines: ~18x the ~220 B a ~70-cycle L3 latency needs at the
 * observed consumption rate, small enough that the L1 footprint (4 KB of
 * NTA lines against the 20 KB plane buffer) does not evict lines before use.
 * NTA lines evicted from L1 were never in L2, so a too-large lead silently
 * doubles the L3 read -- keep this SHORT. */
#ifndef FFT36_PFDN
# define FFT36_PFDN 512
#endif

/* W3 = exp(-2*pi*i/3): sqrt(3)/2 */
#define KS3  0.86602540378443864676372317075294
/* W9^m = cos(2*pi*m/9) - i*sin(2*pi*m/9) */
#define W1R  0.76604444311897803520239265055542
#define W1I (-0.64278760968653932632264340990726)
#define W2R  0.17364817766693034885171662676931
#define W2I (-0.98480775301220805936674302458952)
#define W4R (-0.93969262078590838405410927732473)
#define W4I (-0.34202014332566873304409961468226)

/* instantiate the kernel template at 256-bit, and at 512-bit where possible */
#define PW 2
#include __FILE__
#undef PW
#ifdef __AVX512F__
# define PW 4
# include __FILE__
# undef PW
# define HAVE_PW4 1
#endif

/* ---- plan, tuner, API ---------------------------------------------------- */

enum { M_INPLACE = 0, M_SCRATCH = 1, M_SCRATCH_NT = 2, M_PIPE = 3 };
static const char *const mode_name[] = {"inplace", "scratch", "scratch+nt", "pipe"};

struct fft3d_plan {
    int     batch;
    int     pw;                  /* 2 or 4                         */
    int     mode;                /* one of M_*                     */
    int     pf;                  /* software-prefetch the in stream */
    double *S;                   /* reused scratch volume          */
    double *S1;                  /* second scratch (M_PIPE ping-pong) */
    void   *rawS, *rawS1;
};

const char *fft3d_name(void) { return "L36_pfa"; }

/* Filled by fft3d_create() with the tuner's pick, so the leaderboard shows
 * which variant actually ran (r3 verdict requirement; mechanism borrowed from
 * L36_mixedradix / L6_pfa). */
static char g_desc[224];

const char *fft3d_description(void)
{
    return g_desc[0] ? g_desc
                     : "Good-Thomas PFA 4x9, interleaved-complex lanes, two "
                       "sweeps; {inplace, reused scratch(+NT), 2-scratch "
                       "ping-pong pipeline} x {pw,pf} autotuned in create()";
}
int fft3d_supports(int Lq) { return Lq == L; }

static void run_vols(int pw, int mode, int pf, double *S, double *S1,
                     const double *in, double *out, int nvol)
{
    if (mode == M_PIPE) {
#ifdef HAVE_PW4
        if (pw == 4) pipe_vols_pw4(in, out, S, S1, nvol, pf);
        else
#endif
        { (void)pw; pipe_vols_pw2(in, out, S, S1, nvol, pf); }
        _mm_sfence();
        return;
    }
    /* pf=2/3 add write-intent prefetch to whichever pass stores cold lines:
     * phase 1 when mid==out (INPLACE), phase 2 when out is store-cold and the
     * stores are normal (SCRATCH).  NT stores do not RFO, nothing to hide.
     * pfr is the read-prefetch kind: 0 none, 1 paced T1 (pf=1/2), 2 NTA at
     * consumption rate (pf=3/4). */
    const int p1w = ((pf == 2 || pf == 3) && mode == M_INPLACE);
    const int p2w = ((pf == 2 || pf == 3) && mode == M_SCRATCH);
    const int pfr = (pf >= 3) ? 2 : (pf ? 1 : 0);
    for (int b = 0; b < nvol; ++b) {
        const double *i = in  + (size_t)b * VDBL;
        double       *o = out + (size_t)b * VDBL;
        double     *mid = (mode == M_INPLACE) ? o : S;
        const int    nt = (mode == M_SCRATCH_NT);
        /* next volume's input, pre-covered from inside phase 2 (pf=1/2 only:
         * pf=3/4 target cache-resident regimes where the cold window is an
         * L3 hit, and 62 KB of pre-coverage would churn L2 for nothing) */
        const double *nx = (pf == 1 || pf == 2) && b + 1 < nvol ? i + VDBL : NULL;
        /* FFT36_SKIP1/2: temporary diagnostics, wrong answers, timing only */
#ifndef FFT36_SKIP1
# define P1(f, a, b, c, d) f(a, b, c, d)
#else
# define P1(f, a, b, c, d) ((void)0)
#endif
#ifndef FFT36_SKIP2
# define P2(f, a, b, c, d, e) f(a, b, c, d, e)
#else
# define P2(f, a, b, c, d, e) ((void)0)
#endif
#ifdef HAVE_PW4
        if (pw == 4) { P1(phase1_pw4, i, mid, pfr, p1w); P2(phase2_pw4, mid, o, nt, nx, p2w); }
        else
#endif
        { (void)pw; P1(phase1_pw2, i, mid, pfr, p1w); P2(phase2_pw2, mid, o, nt, nx, p2w); }
    }
    if (mode == M_SCRATCH_NT) _mm_sfence();
}

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* hysteresis rank: lower = simpler; a candidate must beat a simpler one by
 * >3% to be installed (r3 verdict: tuner coin-flips cost 3.9-6.7% elsewhere,
 * and pf=1 winning in-arena at B=32 while losing end-to-end is the leading
 * hypothesis for this entry's own r3 regression there; genuine wins measured
 * on wallaby -- pf at B=256, pipe vs scratch+nt -- are all >=10%, so the band
 * costs nothing where the choice is real) */
static int cand_rank(int mode, int pf)
{
    /* pf complexity: 0 < 1 < 2 < 4 < 3 -- pf=4 (NTA read alone) is one
     * mechanism, pf=3 (NTA read + write-intent) is two */
    static const int pfc[5] = {0, 1, 2, 4, 3};
    return pfc[pf] * 4 + mode;   /* inplace < scratch < scratch+nt < pipe */
}

fft3d_plan *fft3d_create(int Lq, int batch)
{
    if (Lq != L || batch < 1) return NULL;
    fft3d_plan *p = calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->batch = batch;
    if (posix_memalign(&p->rawS, 64, VDBL * sizeof(double)) != 0) {
        free(p); return NULL;
    }
    p->S = (double *)p->rawS;
    memset(p->S, 0, VDBL * sizeof(double));
    if (posix_memalign(&p->rawS1, 64, VDBL * sizeof(double)) == 0) {
        p->S1 = (double *)p->rawS1;
        memset(p->S1, 0, VDBL * sizeof(double));
    }                            /* on failure: no pipe candidates, rest works */
    p->pw = 2; p->mode = M_SCRATCH; p->pf = 0;   /* safe default */

    /* candidate list, first entry doubles as the numerical reference.
     * pf=2/3 (write-intent prefetch) and pf=4 (NTA read alone) exist only
     * where normal stores hit cold lines / where L2 residency of the store
     * target is the prize: INPLACE (phase-1 mid==out) and SCRATCH. */
    struct cand { int pw, mode, pf; } cands[32];
    int nc = 0;
    for (int pf = 0; pf <= 4; ++pf) {
        cands[nc++] = (struct cand){2, M_SCRATCH,    pf};
        cands[nc++] = (struct cand){2, M_INPLACE,    pf};
        if (pf <= 1) {
            cands[nc++] = (struct cand){2, M_SCRATCH_NT, pf};
            if (p->S1) cands[nc++] = (struct cand){2, M_PIPE, pf};
        }
#ifdef HAVE_PW4
        cands[nc++] = (struct cand){4, M_SCRATCH,    pf};
        cands[nc++] = (struct cand){4, M_INPLACE,    pf};
        if (pf <= 1) {
            cands[nc++] = (struct cand){4, M_SCRATCH_NT, pf};
            if (p->S1) cands[nc++] = (struct cand){4, M_PIPE, pf};
        }
#endif
    }
    /* run-time forcing for the monitor's control jobs (no recompile needed) */
    { const char *e;
      if ((e = getenv("FFT36_PW"))) {
          int v = atoi(e), w = 0;
          for (int c = 0; c < nc; ++c) if (cands[c].pw == v) cands[w++] = cands[c];
          if (w) nc = w;
      }
      if ((e = getenv("FFT36_MODE"))) {
          int num = (e[0] >= '0' && e[0] <= '9'), w = 0;
          for (int c = 0; c < nc; ++c)
              if (num ? cands[c].mode == atoi(e)
                      : !strcmp(mode_name[cands[c].mode], e)) cands[w++] = cands[c];
          if (w) nc = w;
      }
      if ((e = getenv("FFT36_PF"))) {
          int v = atoi(e), w = 0;
          for (int c = 0; c < nc; ++c) if (cands[c].pf == v) cands[w++] = cands[c];
          if (w) nc = w;
      } }
#ifdef FFT_FORCE_PW
    { int w = 0;
      for (int c = 0; c < nc; ++c) if (cands[c].pw == FFT_FORCE_PW) cands[w++] = cands[c];
      if (w) nc = w; }
#endif
#ifdef FFT_FORCE_MODE
    { int w = 0;
      for (int c = 0; c < nc; ++c) if (cands[c].mode == FFT_FORCE_MODE) cands[w++] = cands[c];
      if (w) nc = w; }
#endif
#ifdef FFT36_FORCE_PF
    { int w = 0;
      for (int c = 0; c < nc; ++c) if (cands[c].pf == FFT36_FORCE_PF) cands[w++] = cands[c];
      if (w) nc = w; }
#endif

    /* tuning arena: must actually leave L3 at large batch, or the NT-vs-
     * cached-store ranking inverts (a 16-volume arena fit wallaby's 60 MB L3
     * and mis-picked cached stores at B=256; 64 volumes = 96 MB does not) */
    const int nv = batch < 64 ? batch : 64;
    void *ri = NULL, *ro = NULL, *rr = NULL;
    if (posix_memalign(&ri, 64, (size_t)nv * VDBL * sizeof(double)) ||
        posix_memalign(&ro, 64, (size_t)nv * VDBL * sizeof(double)) ||
        posix_memalign(&rr, 64, (size_t)nv * VDBL * sizeof(double))) {
        free(ri); free(ro);                 /* keep the safe default */
        p->pw = cands[0].pw; p->mode = cands[0].mode; p->pf = cands[0].pf;
        snprintf(g_desc, sizeof g_desc,
                 "GT-PFA 4x9 two-sweep; tuner SKIPPED (arena alloc failed): "
                 "pw=%d mode=%s pf=%d", p->pw, mode_name[p->mode], p->pf);
        return p;
    }
    double *tin = ri, *tout = ro, *ref = rr;
    unsigned long long s = 0x9E3779B97F4A7C15ull;
    for (size_t i = 0; i < (size_t)nv * VDBL; ++i) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        tin[i] = (double)(long long)(s >> 11) * 0x1p-53;
    }

    run_vols(cands[0].pw, cands[0].mode, cands[0].pf, p->S, p->S1, tin, ref, nv);

    int    ok[32];
    double tc[32];
    for (int c = 0; c < nc; ++c) {
        run_vols(cands[c].pw, cands[c].mode, cands[c].pf, p->S, p->S1, tin, tout, nv);
        double num = 0.0, den = 0.0;
        for (size_t i = 0; i < (size_t)nv * VDBL; ++i) {
            double d = tout[i] - ref[i];
            num += d * d; den += ref[i] * ref[i];
        }
        ok[c] = (num <= den * 1e-26);       /* rel L2 < 1e-13 vs reference */
        tc[c] = 1e300;
    }
    const int R = (nv >= 8) ? 1 : (nv >= 2 ? 3 : 8);
    for (int round = 0; round < 5; ++round)
        for (int c = 0; c < nc; ++c) {
            if (!ok[c]) continue;
            /* self-warming (adopted from L36_pencilfused r5): one untimed
             * exec so each candidate is timed from its OWN steady-state cache,
             * not its predecessor's -- an NT/pipe candidate flushes tout and
             * was charging its successor a deterministic cold-RFO penalty
             * (their measured artifact: 167.4 vs a true 89.8 us/vol). */
            run_vols(cands[c].pw, cands[c].mode, cands[c].pf, p->S, p->S1, tin, tout, nv);
            double t0 = now_s();
            for (int r = 0; r < R; ++r)
                run_vols(cands[c].pw, cands[c].mode, cands[c].pf, p->S, p->S1, tin, tout, nv);
            double t = (now_s() - t0) / R;
            if (t < tc[c]) tc[c] = t;
        }
    int best = 0;
    for (int c = 1; c < nc; ++c) if (ok[c] && tc[c] < tc[best]) best = c;
    if (ok[best]) {
        /* hysteresis: any simpler candidate within 3% of the best wins */
        int pick = best;
        for (int c = 0; c < nc; ++c)
            if (ok[c] && tc[c] <= tc[best] * 1.03 &&
                cand_rank(cands[c].mode, cands[c].pf) <
                cand_rank(cands[pick].mode, cands[pick].pf)) pick = c;
        p->pw = cands[pick].pw; p->mode = cands[pick].mode; p->pf = cands[pick].pf;
    }
    snprintf(g_desc, sizeof g_desc,
             "GT-PFA 4x9 two-sweep; tuner pick: pw=%d mode=%s pf=%d (B=%d, nv=%d)",
             p->pw, mode_name[p->mode], p->pf, batch, nv);

#ifdef FFT36_LOUD
    if (1) {
#else
    if (getenv("FFT36_VERBOSE")) {
#endif
        for (int c = 0; c < nc; ++c)
            fprintf(stderr, "L36_pfa tuner: pw=%d mode=%-10s pf=%d  %s  %.1f us/vol\n",
                    cands[c].pw, mode_name[cands[c].mode], cands[c].pf,
                    ok[c] ? "ok " : "BAD", ok[c] ? tc[c] * 1e6 / nv : 0.0);
        fprintf(stderr, "L36_pfa tuner: chose pw=%d mode=%s pf=%d (nv=%d)\n",
                p->pw, mode_name[p->mode], p->pf, nv);
    }
    free(ri); free(ro); free(rr);
    return p;
}

void fft3d_destroy(fft3d_plan *p)
{
    if (!p) return;
    free(p->rawS); free(p->rawS1); free(p);
}

void fft3d_execute(fft3d_plan *plan, const double _Complex *in, double _Complex *out)
{
    run_vols(plan->pw, plan->mode, plan->pf, plan->S, plan->S1,
             (const double *)in, (double *)out, plan->batch);
}

#else /* ================= KERNEL TEMPLATE, PW = 2 or 4 ==================== */

#define vec   CAT(vec_pw,  PW)
#define veci  CAT(veci_pw, PW)
#define FN(n) CAT(n, CAT(_pw, PW))

typedef double    vec  __attribute__((vector_size(PW * 16)));
typedef long long veci __attribute__((vector_size(PW * 16)));

#ifdef __clang__
# define VSH(a,b,...) __builtin_shufflevector(a, b, __VA_ARGS__)
#else
# define VSH(a,b,...) __builtin_shuffle(a, b, (veci){__VA_ARGS__})
#endif

#define NVR (L / PW)             /* vectors per 36-complex row: 18 or 9 */

#if PW == 4
# define VSPLAT(a)  ((vec){(a),(a),(a),(a),(a),(a),(a),(a)})
# define VPAIR(a,b) ((vec){(a),(b),(a),(b),(a),(b),(a),(b)})
# define SWAP(v)    VSH((v),(v), 1,0,3,2,5,4,7,6)
# define VFMA(a,b,c)  ((vec)_mm512_fmadd_pd((__m512d)(a),(__m512d)(b),(__m512d)(c)))
# define VFNMA(a,b,c) ((vec)_mm512_fnmadd_pd((__m512d)(a),(__m512d)(b),(__m512d)(c)))
# define STREAM_ST(p,v) _mm512_stream_pd((p), (__m512d)(v))
#else
# define VSPLAT(a)  ((vec){(a),(a),(a),(a)})
# define VPAIR(a,b) ((vec){(a),(b),(a),(b)})
# define SWAP(v)    VSH((v),(v), 1,0,3,2)
# ifdef __FMA__
#  define VFMA(a,b,c)  ((vec)_mm256_fmadd_pd((__m256d)(a),(__m256d)(b),(__m256d)(c)))
#  define VFNMA(a,b,c) ((vec)_mm256_fnmadd_pd((__m256d)(a),(__m256d)(b),(__m256d)(c)))
# else
#  define VFMA(a,b,c)  ((a)*(b) + (c))
#  define VFNMA(a,b,c) ((c) - (a)*(b))
# endif
# ifdef __AVX__
#  define STREAM_ST(p,v) _mm256_stream_pd((p), (__m256d)(v))
# else
#  define STREAM_ST(p,v) (*(vec *)(p) = (v))
# endif
#endif

/* PW x PW transpose of 128-bit complex granules; involution, 8 shuffles per
 * 4 vectors at PW=4, 2 per 2 vectors at PW=2. */
#if PW == 4
# define TRNC(r, c) do {                                                     \
    vec u0_ = VSH((r)[0], (r)[1], 0,1,8,9,4,5,12,13);                        \
    vec u1_ = VSH((r)[0], (r)[1], 2,3,10,11,6,7,14,15);                      \
    vec u2_ = VSH((r)[2], (r)[3], 0,1,8,9,4,5,12,13);                        \
    vec u3_ = VSH((r)[2], (r)[3], 2,3,10,11,6,7,14,15);                      \
    (c)[0] = VSH(u0_, u2_, 0,1,2,3,8,9,10,11);                               \
    (c)[2] = VSH(u0_, u2_, 4,5,6,7,12,13,14,15);                             \
    (c)[1] = VSH(u1_, u3_, 0,1,2,3,8,9,10,11);                               \
    (c)[3] = VSH(u1_, u3_, 4,5,6,7,12,13,14,15);                             \
} while (0)
#else
# define TRNC(r, c) do {                                                     \
    vec u0_ = VSH((r)[0], (r)[1], 0,1,4,5);                                  \
    vec u1_ = VSH((r)[0], (r)[1], 2,3,6,7);                                  \
    (c)[0] = u0_; (c)[1] = u1_;                                              \
} while (0)
#endif

/* y1 = a - i*s*m, y2 = a + i*s*m via one swap and two FMAs (6 arith + 1 shuf) */
#define DFT3M(x0, x1, x2, y0, y1, y2) do {                                   \
    vec t3_ = (x1) + (x2), m3_ = (x1) - (x2);                                \
    vec a3_ = VFNMA(t3_, VSPLAT(0.5), (x0));                                 \
    vec s3_ = SWAP(m3_);                                                     \
    (y0) = (x0) + t3_;                                                       \
    (y1) = VFMA(s3_, VPAIR(KS3, -KS3), a3_);                                 \
    (y2) = VFNMA(s3_, VPAIR(KS3, -KS3), a3_);                                \
} while (0)

/* v * (cr + i*ci): 1 shuffle + mul + FMA */
#define CMULW(v, cr, ci) VFMA((v), VSPLAT(cr), SWAP(v) * VPAIR(-(ci), (ci)))

/* The 36-point Good-Thomas 4x9 codelet over PW interleaved-complex lanes.
 * LD(n) must yield input element n as a vec rvalue; ST(k, v) must consume
 * output element k.  Both index maps fold to compile-time constants once the
 * loops unroll.  All LD reads happen before the first ST, so LD/ST may alias. */
#define PFA36(LD, ST) do {                                                   \
    vec A_[36];                            /* A_[9*k1 + n2] */               \
    _Pragma("GCC unroll 9")                                                  \
    for (int n2_ = 0; n2_ < 9; ++n2_) {                                      \
        vec x0_ = LD(( 0 + 4*n2_) % 36), x1_ = LD(( 9 + 4*n2_) % 36);        \
        vec x2_ = LD((18 + 4*n2_) % 36), x3_ = LD((27 + 4*n2_) % 36);        \
        vec t0_ = x0_ + x2_, t1_ = x0_ - x2_;                                \
        vec t2_ = x1_ + x3_, t3_ = x1_ - x3_;                                \
        vec sw_ = SWAP(t3_);                                                 \
        A_[     n2_] = t0_ + t2_;                                            \
        A_[18 + n2_] = t0_ - t2_;                                            \
        A_[ 9 + n2_] = VFMA (sw_, VPAIR(1.0, -1.0), t1_);                    \
        A_[27 + n2_] = VFNMA(sw_, VPAIR(1.0, -1.0), t1_);                    \
    }                                                                        \
    _Pragma("GCC unroll 4")                                                  \
    for (int k1_ = 0; k1_ < 4; ++k1_) {                                      \
        const vec *g_ = A_ + 9 * k1_;                                        \
        vec B_[9];                         /* B_[3*r + b] */                 \
        _Pragma("GCC unroll 3")                                              \
        for (int b_ = 0; b_ < 3; ++b_)                                       \
            DFT3M(g_[b_], g_[3 + b_], g_[6 + b_],                            \
                  B_[b_], B_[3 + b_], B_[6 + b_]);                           \
        B_[4] = CMULW(B_[4], W1R, W1I);                                      \
        B_[5] = CMULW(B_[5], W2R, W2I);                                      \
        B_[7] = CMULW(B_[7], W2R, W2I);                                      \
        B_[8] = CMULW(B_[8], W4R, W4I);                                      \
        _Pragma("GCC unroll 3")                                              \
        for (int r_ = 0; r_ < 3; ++r_) {                                     \
            vec y0_, y1_, y2_;                                               \
            DFT3M(B_[3*r_], B_[3*r_ + 1], B_[3*r_ + 2], y0_, y1_, y2_);      \
            ST((9*k1_ + 28*(r_    )) % 36, y0_);                             \
            ST((9*k1_ + 28*(r_ + 3)) % 36, y1_);                             \
            ST((9*k1_ + 28*(r_ + 6)) % 36, y2_);                             \
        }                                                                    \
    }                                                                        \
} while (0)

/* Paced prefetch of the (perfectly linear) `in` stream.  Each of the 2*NVR
 * loop iterations per x-plane advances the cursor by PFSTEP doubles, so one
 * plane's worth of prefetches issues per plane processed, spread evenly over
 * both subloops (the zb subloop touches no `in` bytes, so pacing through it
 * keeps the DRAM read stream busy during the y transform too). */
#define PFSTEP (36 * PW)                   /* doubles per iteration            */
#define PFIN(p) do {                                                          \
    _Pragma("GCC unroll 18")                                                  \
    for (int q_ = 0; q_ < PFSTEP / 8; ++q_)                                   \
        __builtin_prefetch((p) + 8 * q_, 0, FFT36_PFH);                       \
} while (0)

/* pf=3/4: NTA prefetch of the in stream at EXACTLY the consumption rate of
 * the yb-subloop (which reads 2*PFSTEP doubles per iteration); the zb-subloop
 * touches no in bytes and issues nothing, so the cursor lead is a constant
 * FFT36_PFDN.  NTA fills L1 and bypasses L2 on SKX-class cores, so the read
 * stream stops evicting `out` / S from L2 -- the whole point (r6 header). */
#define PFIN_NTA(p) do {                                                      \
    _Pragma("GCC unroll 36")                                                  \
    for (int q_ = 0; q_ < PFSTEP / 4; ++q_)                                   \
        __builtin_prefetch((p) + 8 * q_, 0, 0);                               \
} while (0)

/* pf=2 (M_INPLACE): paced WRITE-INTENT cursor over the mid-plane store
 * stream, same pacing arithmetic as PFIN (one plane's worth per plane
 * processed), one plane ahead.  __builtin_prefetch(p,1,3) emits `prefetchw`
 * on PRFCHW machines (Cascade Lake, Sapphire Rapids), acquiring the line
 * exclusive before the store so the cold-out RFO overlaps compute instead of
 * stalling the store buffer.  Adopted from L6_unrolled r3 (their fused_pfw,
 * -29% at DRAM sizes; node-confirmed via L6_pfa r4). */
#define PFWMID(p) do {                                                        \
    _Pragma("GCC unroll 18")                                                  \
    for (int q_ = 0; q_ < PFSTEP / 8; ++q_)                                   \
        __builtin_prefetch((p) + 8 * q_, 1, 3);                               \
} while (0)

/* phase 1, ONE x-plane: z transform (transposed lanes) then y transform.
 * Reads plane x of `in` sequentially; writes mid[x][ky][kz].  The paced
 * prefetch cursor is stateless across planes: each plane issues exactly one
 * plane's worth (2*LSQ doubles) of prefetches, 2*NVR iterations x PFSTEP,
 * so cursor(x) = in + PFD + x*2*LSQ reproduces the r3 per-volume pacing
 * exactly while letting M_PIPE call planes individually. */
static inline __attribute__((always_inline))
void FN(phase1_plane)(const double *restrict in, double *restrict mid,
                      int x, int pfr, int pfw)
{
    vec pl[L * NVR];                       /* plane [y][kz], 20.25 KB */
    const double *pfc = in  + FFT36_PFD  + (size_t)x * (2 * LSQ);
    const double *pfn = in  + FFT36_PFDN + (size_t)x * (2 * LSQ);
    double       *pwc = mid + FFT36_PFWD + (size_t)x * (2 * LSQ);
    const double *px  = in  + (size_t)x * (2 * LSQ);
    double       *mx  = mid + (size_t)x * (2 * LSQ);

    for (int yb = 0; yb < L; yb += PW) {
        if (pfr == 1)      { PFIN(pfc);     pfc += PFSTEP; }
        else if (pfr == 2) { PFIN_NTA(pfn); pfn += 2 * PFSTEP; }
        if (pfw) { PFWMID(pwc); pwc += PFSTEP; }
        vec Zv[36], Wv[36];
        _Pragma("GCC unroll 18")
        for (int zb = 0; zb < NVR; ++zb) {
            vec r_[PW];
            _Pragma("GCC unroll 4")
            for (int j = 0; j < PW; ++j)
                r_[j] = *(const vec *)(px + ((size_t)(yb + j) * L + (size_t)zb * PW) * 2);
            TRNC(r_, &Zv[zb * PW]);
        }
#define LD1(n)    Zv[n]
#define ST1(k, v) (Wv[k] = (v))
        PFA36(LD1, ST1);
#undef LD1
#undef ST1
        _Pragma("GCC unroll 18")
        for (int zb = 0; zb < NVR; ++zb) {
            vec r_[PW];
            TRNC(&Wv[zb * PW], r_);
            _Pragma("GCC unroll 4")
            for (int j = 0; j < PW; ++j)
                pl[(size_t)(yb + j) * NVR + zb] = r_[j];
        }
    }

    for (int zb = 0; zb < NVR; ++zb) {
        if (pfr == 1) { PFIN(pfc); pfc += PFSTEP; }
        if (pfw) { PFWMID(pwc); pwc += PFSTEP; }
#define LD2(n)    pl[(size_t)(n) * NVR + zb]
#define ST2(k, v) (*(vec *)(mx + ((size_t)(k) * L + (size_t)zb * PW) * 2) = (v))
        PFA36(LD2, ST2);
#undef LD2
#undef ST2
    }
}

static void FN(phase1)(const double *restrict in, double *restrict mid,
                       int pfr, int pfw)
{
    for (int x = 0; x < L; ++x)
        FN(phase1_plane)(in, mid, x, pfr, pfw);
}

/* one x-line-group tile: 36 strided loads -> codelet -> W_arr */
#define TILE(srcp, W_arr) do {                                               \
    const double *s_ = (srcp);                                               \
    vec *W_ = (W_arr);                                                       \
    PF36(s_);                                                                \
    PFA36(LD3, ST3);                                                         \
} while (0)
#define LD3(n)    (*(const vec *)(s_ + (size_t)(n) * (2 * LSQ)))
#define ST3(k, v) (W_[k] = (v))
#ifndef FFT36_NOPF
# define PF36(s_) do {                                                       \
    _Pragma("GCC unroll 36")                                                 \
    for (int n_ = 0; n_ < 36; ++n_)                                          \
        __builtin_prefetch(s_ + (size_t)n_ * (2 * LSQ) + 8, 0, 3);           \
} while (0)
#else
# define PF36(s_) do { } while (0)
#endif

/* Per-tile pre-coverage of the NEXT volume's input: 3 lines per tile fills
 * the first 62 KB (>= FFT36_PFD + the phase-1 pacing deficit) of in[b+1]
 * before its phase 1 starts, so the prefetch cursor never starts cold. */
#define PFNX() do { if (pn_) {                                                \
    _Pragma("GCC unroll 9")                                                   \
    for (int q_ = 0; q_ < FFT36_PFN; ++q_)                                    \
        __builtin_prefetch(pn_ + 8 * q_, 0, FFT36_PFH);                       \
    pn_ += 8 * FFT36_PFN; } } while (0)

/* phase 2, ONE y-plane: x transform, 36 sequential source streams of stride
 * 20736 B.  Safe when mid == out: the codelet reads all 36 inputs before its
 * first store, so results go straight from registers to `out` with no
 * staging.  The next-volume prefetch cursor is recomputed per plane from the
 * per-plane tile count, reproducing the r3 whole-volume walk exactly. */
/* pf=2 (M_SCRATCH): write-intent prefetch of the 36 out-streams one 64-B
 * line ahead of the tile being stored -- mirrors PF36 on the read side. */
#define PFW36(d_) do {                                                       \
    _Pragma("GCC unroll 36")                                                 \
    for (int n_ = 0; n_ < 36; ++n_)                                          \
        __builtin_prefetch((d_) + (size_t)n_ * (2 * LSQ) + 8, 1, 3);         \
} while (0)

static inline __attribute__((always_inline))
void FN(phase2_yplane)(const double *mid, double *out, int nt,
                       const double *pnext, int y, int pfw)
{
#if PW == 2
    const size_t tiles_ = nt ? (size_t)(NVR / 2) : (size_t)NVR;
#else
    const size_t tiles_ = (size_t)NVR;
#endif
    const double *pn_ = pnext ? pnext + (size_t)y * tiles_ * 8 * FFT36_PFN : NULL;
    if (!nt) {
        for (int zb = 0; zb < NVR; ++zb) {
            const size_t o = ((size_t)y * L + (size_t)zb * PW) * 2;
            const double *s_ = mid + o;
            double       *d_ = out + o;
            PF36(s_);
            if (pfw) PFW36(d_);
            PFNX();
#define ST3D(k, v) (*(vec *)(d_ + (size_t)(k) * (2 * LSQ)) = (v))
            PFA36(LD3, ST3D);
#undef ST3D
        }
    } else {
#if PW == 2
        /* pair two z-blocks so every NT write completes a 64-byte line */
        for (int zb = 0; zb < NVR; zb += 2) {
            const size_t o = ((size_t)y * L + (size_t)zb * PW) * 2;
            vec Wa[36], Wb[36];
            TILE(mid + o,     Wa);
            TILE(mid + o + 4, Wb);
            PFNX();
            double *dst = out + o;
            _Pragma("GCC unroll 36")
            for (int k = 0; k < 36; ++k) {
                STREAM_ST(dst + (size_t)k * (2 * LSQ),     Wa[k]);
                STREAM_ST(dst + (size_t)k * (2 * LSQ) + 4, Wb[k]);
            }
        }
#else
        for (int zb = 0; zb < NVR; ++zb) {
            const size_t o = ((size_t)y * L + (size_t)zb * PW) * 2;
            const double *s_ = mid + o;
            double       *d_ = out + o;
            PF36(s_);
            PFNX();
#define ST3N(k, v) STREAM_ST(d_ + (size_t)(k) * (2 * LSQ), (v))
            PFA36(LD3, ST3N);
#undef ST3N
        }
#endif
    }
}

static void FN(phase2)(const double *mid, double *out, int nt,
                       const double *pnext, int pfw)
{
    for (int y = 0; y < L; ++y)
        FN(phase2_yplane)(mid, out, nt, pnext, y, pfw);
}

/* M_PIPE: cross-volume ping-pong pipeline (round 4).  While phase 2 drains
 * volume b from scratch S_a to `out` with NT stores, phase 1 of volume b+1
 * fills S_b -- interleaved at plane granularity (both phases have exactly L
 * outer iterations per volume), so the compulsory DRAM read (in[b+1]) and NT
 * write (out[b]) streams run CONCURRENTLY at ~half rate each instead of as
 * alternating full-rate bursts with the other stream idle.  Phase 2 goes
 * first in each unit so its NT stores drain asynchronously under phase 1's
 * compute.  PFNX is off (pnext = NULL): phase 1 of b+1 is itself touching
 * in[b+1] concurrently, which supersedes the pre-coverage trick. */
static void FN(pipe_vols)(const double *restrict in, double *restrict out,
                          double *S0, double *S1, int nvol, int pf)
{
    double *Sa = S0, *Sb = S1;
    for (int x = 0; x < L; ++x)                     /* prologue: volume 0 */
        FN(phase1_plane)(in, Sa, x, pf, 0);
    for (int b = 0; b < nvol; ++b) {
        const double *nin = (b + 1 < nvol) ? in + (size_t)(b + 1) * VDBL : NULL;
        double *o = out + (size_t)b * VDBL;
        for (int u = 0; u < L; ++u) {
            FN(phase2_yplane)(Sa, o, 1, NULL, u, 0);
            if (nin) FN(phase1_plane)(nin, Sb, u, pf, 0);
        }
        double *sw = Sa; Sa = Sb; Sb = sw;
    }
}

#undef TILE
#undef LD3
#undef ST3
#undef PF36
#undef PFW36
#undef PFNX
#undef PFIN
#undef PFIN_NTA
#undef PFWMID
#undef PFSTEP
#undef PFA36
#undef CMULW
#undef DFT3M
#undef TRNC
#undef STREAM_ST
#undef VFMA
#undef VFNMA
#undef SWAP
#undef VPAIR
#undef VSPLAT
#undef NVR
#undef VSH
#undef FN
#undef veci
#undef vec

#endif /* template */
