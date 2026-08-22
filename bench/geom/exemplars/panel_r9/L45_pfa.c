/* L45_pfa.c -- forward complex 3D DFT of a 45^3 cube, batched, out-of-place.
 *
 * ROUND panel_r9 (fourth round).  r9 changes:
 *   1. DFT9 module replaced: the hand CT 3x3 form (44 FMA-port vector ops +
 *      10 swaps) is replaced by a pairwise transcription of genfft's FMA
 *      n1_9 DAG (fftw-3.3.10/dft/scalar/codelets/n1_9.c: 24 add + 56 fma =
 *      80 scalar FMA-port ops) onto interleaved-complex vectors: 40 FMA-port
 *      ops + 12 swaps.  Every scalar re/im line pair maps to ONE vector op;
 *      the points where the scalar DAG crosses re and im (multiplies by i and
 *      the (1 + c*i) spiral factors) each cost one SWAP, with all signs
 *      folded into VPAIR constants.  Per line: 364 -> 344 port-0 ops (-5.5%,
 *      the node's bottleneck port), 68 -> 78 port-5 shuffles (still <= half
 *      of port 0).  L36_pfa r1 burned three attempts deriving this by hand;
 *      the transcription of the actual generated DAG is mechanical and the
 *      create()-time reference gate proves it.
 *   2. The file-level '#pragma GCC optimize("unroll-loops")' is REMOVED
 *      (r8 VERDICT 3c: the scored build has carried -funroll-loops in its
 *      Makefile all along, so the pragma's premise was false, and L17_rader
 *      measured the pragma FORM as a ~2% tax because optimize() rebuilds the
 *      whole per-function option set).  -DFFT45_UNROLL_PRAGMA restores it
 *      for A/B.
 *
 * ROUND panel_r8 (r6, r7 before).  r8 changes:
 *   1. Compile-time specialization of the prefetch ladder and the mid-buffer
 *      layout (borrowed from L45_mixedradix's exec-variant structure): phase 1
 *      and phase 2 are always_inline bodies taking const flags and const
 *      strides, instantiated into per-(pw, mode, pf) exec functions.  The
 *      node's r7 pick at B=1/B=2 was pw4-inplace-pf0, and in r7 that path
 *      still carried runtime pf branches, two cursor pointers per plane, and
 *      ~180 never-executed prefetch instructions inside the hot loops (my
 *      3535 hot instructions vs the rival's 3134 -- and they beat me 328 vs
 *      343 us at equal op count).  Now pf0 compiles to zero prefetch code.
 *   2. NEW padded-scratch mode (scratchp): phase 1 writes a plan-owned S with
 *      row pitch 52 complex (832 B = 13 lines, odd, 64B-aligned rows) and
 *      plane pitch 45*52 = 2340 complex (37440 B = 585 lines, odd).  The
 *      y-pass stores and the x-pass loads become 64-byte ALIGNED, deleting
 *      two of the three split-access classes (~75% of the 64 B accesses at
 *      the natural 720 B row stride split a cache line; only the x-pass
 *      stores to `out` keep paying the odd-L toll).  The odd-line pitches
 *      also break the same-stride read/write mod-4096 correlation that
 *      L23_rader measured at -25..30% (their self-inflicted-aliasing rule):
 *      r6's unpadded scratch read S and wrote out at the SAME 32400 B plane
 *      stride and lost ~66% -- padding recovers most of that (wallaby r8:
 *      sp is within 11-13% of inplace at B=1, was ~66% behind) but does not
 *      win there; it is a candidate for the node's different cache/split
 *      physics, not a wallaby winner.  scratchp's x pass is out-of-place
 *      (S -> out), so its odd
 *      tails overlap-recompute per y-row (12 tiles/row at PW=4, +2.2% volume
 *      ops vs flat tiling) instead of needing masked calls.
 *
 * TECHNIQUE (unchanged from r7 otherwise)
 *   Row-column 3D DFT; every 45-point line is a Good-Thomas / prime-factor
 *   9 x 5 codelet, on INTERLEAVED complex vectors whose lanes are a spectator
 *   axis.  gcd(9,5) = 1, so with
 *
 *       input  (Ruritanian): n = (5*n1 + 9*n2) mod 45     n1 in [0,9), n2 in [0,5)
 *       output (CRT):        k = (10*k1 + 36*k2) mod 45   (10 = 5*[5^-1]_9,
 *                                                          36 = 9*[9^-1]_5)
 *
 *   W45^{nk} = W9^{n1 k1} * W5^{n2 k2} exactly: 5 DFT9s then 9 DFT5s with NO
 *   twiddles in between, both maps folded into compile-time addressing.
 *   DFT9 = genfft n1_9 FMA DAG transcribed to interleaved vectors (40
 *   FMA-port ops + 12 swaps, r9), DFT5 = FFTW n1_5's FMA form (16 ops).
 *   Per 45-point line: 344 FMA-port vector ops + 78 shuffles.
 *
 *   Two sweeps (the structure that won L=36 and L=45 on the node):
 *   phase 1, per x-plane:
 *       z transform: lanes = PW y-rows, PWxPW complex-granule register
 *                    transposes on load and store, into plane scratch
 *                    pl[y][kz] (row pitch PPITCH = 52 complex, 13 lines,
 *                    coprime with the 64 L1 sets)
 *       y transform: lanes = PW kz (contiguous in pl), store to mid[x][ky][kz]
 *                    where mid = out (INPLACE) or the padded S (SCRATCHP)
 *   phase 2:
 *       x transform: lanes = PW kz, 45 streams at the plane stride.
 *       INPLACE:  mid = out, in place; tiles the FLAT (y,z) index
 *                 (2025 = 506*4 + 1: ONE masked tail call per volume).
 *       SCRATCHP: S -> out, out of place; tiles per y-row with an
 *                 overlapping last tile (rows are padded so flat tiling
 *                 does not map affinely to out).
 *
 *   Tail policy: out-of-place subpasses OVERLAP the last lane group
 *   (recompute is idempotent); the z-line's odd 45th column is a 16 B
 *   column gather/scatter (GCOL/SCOL); the in-place x pass keeps the single
 *   masked flat-tail call.
 *
 *   A transpose-count note for the record: a lanes=x phase-1 variant (x is a
 *   spectator of BOTH the z and y transforms) still needs one transpose-class
 *   gather on the z load and one more entering phase 2 -- every arrangement
 *   of this pass structure pays exactly 2 granule transposes per element,
 *   which is what both L=45 entries already pay.  Not a lever; documented so
 *   nobody chases it.
 *
 *   PREFETCH LADDER (tournament-gated, all compile-time now):
 *     ip-pf0: nothing.
 *     ip-pf1: phase 2 pokes its 45 x-read-streams one line ahead (PF45).
 *     ip-pf2: + paced T1 read prefetch of phase 1's in-stream (PFIN) and
 *             per-tile pre-coverage of the next volume's input (PFNX).
 *     ip-pf3: + write-intent prefetchw of phase 1's cold mid-plane stores.
 *     sp-pf0: nothing (S is cache-hot; out is only touched in phase 2).
 *     sp-pfs: PFIN on in + prefetchw poke of phase 2's 45 cold out-streams
 *             + PFNX (the streaming set for scratchp).
 *   fft3d_create() gates every candidate against a scalar O(n^2)-per-line
 *   reference at 1e-13 (volume 0 AND the last arena volume, so S-reuse bugs
 *   across a batch cannot hide), times them interleaved, and installs the
 *   fastest with a 3% simplest-first hysteresis.  FFT45_PW / FFT45_MODE /
 *   FFT45_PF force the choice at plan time for the monitor's control runs.
 *
 * ATTRIBUTION
 *   - Const-propagated exec-variant instantiation (pf compiled out of the
 *     pf0 path): L45_mixedradix (their body()/exec_v_c structure, all
 *     rounds).  Their leaner pf0 loop is why they won r7 on the node.
 *   - Padded-scratch odd-line pitches: L23_rader r6/r7 (the 1058->1064
 *     stride pad, -25..30% at B=1, and the "pad so every stride is an odd
 *     number of cache lines" corpus rule, LITERATURE 04/08).
 *   - Two-sweep plane-fused structure, spectator lanes, 6-op DFT3 / 2-op
 *     CMUL, TRNC transpose, PFIN/PFNX, prefetchw, tuner + hysteresis + env
 *     forcing: L36_pfa (r2-r5), transitively L36_mixedradix r1, L6_unrolled
 *     r3.  Flat phase-2 tiling: L45_mixedradix r7.  Masked z-column tail:
 *     L45_mixedradix r6 (as 128-bit inserts/extracts).  16-op DFT5: FFTW
 *     n1_5 via the corpus.
 *
 * OPERATION COUNT (PW=4)
 *   INPLACE: 45*(12+12) + 506 + 1 = 1587 codelet calls x 344 = 545,928
 *   zmm FMA-port ops/volume (r8: 577,668; the n1_9 DFT9 cuts 5.5%).
 *   SCRATCHP: 1620 calls = 557,280.  Node port-0 floor (1 FMA unit)
 *   ~ 546k cycles ~ 188 us at 2.9 GHz; the r8 node measurement was
 *   319.6 us = 1.61x the old 199 us floor.
 *
 * ACCURACY: ~4e-16 relative L2 vs numpy (unchanged module family).
 */
#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <immintrin.h>

#include "fft3d_api.h"

#ifndef L45_PFA_ONCE            /* ============ COMMON, first pass ============ */
#define L45_PFA_ONCE

/* r7 shipped '#pragma GCC optimize("unroll-loops")' here on the belief the
 * scored build lacked -funroll-loops.  r8's VERDICT 3c proved the premise
 * false (Makefile:15 has carried the flag for >= 3 rounds), and L17_rader
 * measured the pragma FORM as a ~2% tax in its own file (optimize() rebuilds
 * the entire per-function option set, not just the named flag).  Default is
 * now OFF; -DFFT45_UNROLL_PRAGMA restores it for a control build. */
#ifdef FFT45_UNROLL_PRAGMA
# pragma GCC optimize("unroll-loops")
#endif

#define L      45
#define NPLANE 2025              /* 45*45 complex per x-plane                  */
#define PLND   4050              /* doubles per x-plane = x-stream stride      */
#define VDBL   ((size_t)2 * L * NPLANE)  /* doubles per volume = 182250        */
/* plane-scratch row pitch in complex.  52 complex = 832 B = 13 cache lines:
 * coprime with 64 sets, so the y-pass column walk spreads over all L1 sets
 * (pitch 48 = 12 lines measured 15-20% slower in r6, both entries). */
#ifndef PPITCH
# define PPITCH 52
#endif
/* padded mid-volume S for SCRATCHP: row pitch 52 complex (13 lines, odd),
 * plane pitch 45*52 = 2340 complex (585 lines, odd).  Odd-line pitches so no
 * fixed mod-4096 relation between phase 2's S reads (37440 B stride) and its
 * out writes (32400 B stride) can lock in -- the L23_rader aliasing rule. */
#ifndef SPITCH
# define SPITCH 52
#endif
#define SROWD  (2 * SPITCH)              /* S row stride, doubles = 104        */
#define SPLND  (2 * L * SPITCH)          /* S plane stride, doubles = 4680     */
#define SVDBL  ((size_t)L * SPLND)       /* doubles per padded S volume        */

#define CAT_(a,b) a##b
#define CAT(a,b)  CAT_(a,b)

/* phase-1 input prefetch distance, doubles (32 KB) */
#ifndef FFT45_PFD
# define FFT45_PFD 4096
#endif
/* prefetch hint: 3=T0, 2=T1 (the measured winner at L=36/L=6), 0=NTA */
#ifndef FFT45_PFH
# define FFT45_PFH 2
#endif
/* cache lines of the NEXT volume's input prefetched per phase-2 tile */
#ifndef FFT45_PFN
# define FFT45_PFN 2
#endif
/* ip-pf3: write-intent cursor lead over the mid-plane stores, doubles */
#ifndef FFT45_PFWD
# define FFT45_PFWD 4050
#endif

/* sqrt(3)/2 (the DFT3 rotation, and the radix-3 spine of the 9-point DAG) */
#define KS3  0.86602540378443864676372317075294
/* genfft n1_9 (FMA form) DAG constants, fftw-3.3.10 n1_9.c */
#define K176 0.17632698070846497347109038686862  /* tan(pi/18)                */
#define K839 0.83909963117728001176312729812318
#define K777 0.77786191343020616002817797731863
#define K984 0.98480775301220805936674302458952
#define K492 0.49240387650610402968337151229476
#define K852 0.85286853195244320962825096394007
#define K363 0.36397023426620236135104788277683  /* tan(pi/9)                 */
#define K954 0.95418889413867113349926836418725
/* 5-point module constants (FFTW n1_5 form) */
#define K59  0.55901699437494742410229341718282   /* sqrt(5)/4                 */
#define KIG  0.61803398874989484820458683436564   /* sin(4pi/5)/sin(2pi/5)     */
#define KS5  0.95105651629515357211665325776975   /* sin(2pi/5)                */

/* exec variant signature: whole batch, so dispatch happens once per execute */
typedef void (*exec45_fn)(const double *in, double *out, long nvol,
                          double *S, double *P);

/* instantiate the kernel template: PW=2, and PW=4 on AVX-512 */
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

enum { M_INPLACE = 0, M_SCRATCHP = 1 };

/* candidate table, rank = simplest-first tie-break order (3% hysteresis).
 * pw4 ranks ahead of pw2 (the V1-first hardening, L36_mixedradix r6): a
 * narrower kernel must now beat pw4's minimum by >3%, not win a coin flip. */
struct cand45 {
    exec45_fn   fn;
    int         pw, mode, pf, rank;
    const char *nm;
};
static const struct cand45 g_cands[] = {
#ifdef HAVE_PW4
    { x_ip0_pw4, 4, M_INPLACE,  0, 0, "pw4-ip-pf0" },
    { x_ip1_pw4, 4, M_INPLACE,  1, 1, "pw4-ip-pf1" },
    { x_ip2_pw4, 4, M_INPLACE,  2, 2, "pw4-ip-pf2" },
    { x_ip3_pw4, 4, M_INPLACE,  3, 3, "pw4-ip-pf3" },
    { x_sp0_pw4, 4, M_SCRATCHP, 0, 4, "pw4-sp-pf0" },
    { x_sps_pw4, 4, M_SCRATCHP, 2, 5, "pw4-sp-pfs" },
#endif
    { x_ip0_pw2, 2, M_INPLACE,  0, 6, "pw2-ip-pf0" },
    { x_ip3_pw2, 2, M_INPLACE,  3, 7, "pw2-ip-pf3" },
    { x_sp0_pw2, 2, M_SCRATCHP, 0, 8, "pw2-sp-pf0" },
    { x_sps_pw2, 2, M_SCRATCHP, 2, 9, "pw2-sp-pfs" },
};
#define NCAND ((int)(sizeof g_cands / sizeof g_cands[0]))

struct fft3d_plan {
    int       batch;
    exec45_fn fn;
    double   *S;                 /* padded scratch volume (SCRATCHP mid)    */
    double   *P;                 /* plane scratch: page-aligned heap, NOT the
                                    stack -- a stack plane 4K-aliases the
                                    in/out streams in unlucky runs (measured
                                    bimodal 204 vs 377 us at B=1, r6) */
    void     *rawS, *rawP;
};

const char *fft3d_name(void) { return "L45_pfa"; }

static char g_desc[224];

const char *fft3d_description(void)
{
    return g_desc[0] ? g_desc
                     : "Good-Thomas PFA 9x5, interleaved-complex lanes, two "
                       "sweeps; {inplace, padded-scratch} x {pw, pf} exec "
                       "variants (pf compiled out of pf0) autotuned in create()";
}
int fft3d_supports(int Lq) { return Lq == L; }

/* scalar O(L^2)-per-line reference: independent ground truth for the gate */
static void ref3d(const double _Complex *in, double _Complex *out)
{
    double _Complex Wt[L], buf[L];
    for (int k = 0; k < L; ++k)
        Wt[k] = cexp(-2.0 * M_PI * I * (double)k / (double)L);
    for (int x = 0; x < L; ++x)                       /* z axis: in -> out */
        for (int y = 0; y < L; ++y) {
            const double _Complex *r = in  + ((size_t)x * L + y) * L;
            double _Complex       *w = out + ((size_t)x * L + y) * L;
            for (int k = 0; k < L; ++k) {
                double _Complex s = 0;
                for (int j = 0; j < L; ++j) s += r[j] * Wt[(j * k) % L];
                w[k] = s;
            }
        }
    for (int x = 0; x < L; ++x)                       /* y axis, in place  */
        for (int z = 0; z < L; ++z) {
            double _Complex *base = out + (size_t)x * NPLANE + z;
            for (int j = 0; j < L; ++j) buf[j] = base[(size_t)j * L];
            for (int k = 0; k < L; ++k) {
                double _Complex s = 0;
                for (int j = 0; j < L; ++j) s += buf[j] * Wt[(j * k) % L];
                base[(size_t)k * L] = s;
            }
        }
    for (int y = 0; y < L; ++y)                       /* x axis, in place  */
        for (int z = 0; z < L; ++z) {
            double _Complex *base = out + (size_t)y * L + z;
            for (int j = 0; j < L; ++j) buf[j] = base[(size_t)j * NPLANE];
            for (int k = 0; k < L; ++k) {
                double _Complex s = 0;
                for (int j = 0; j < L; ++j) s += buf[j] * Wt[(j * k) % L];
                base[(size_t)k * NPLANE] = s;
            }
        }
}

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static int rel_ok(const double *got, const double *ref, size_t n)
{
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double d = got[i] - ref[i];
        num += d * d; den += ref[i] * ref[i];
    }
    return num <= den * 1e-26;                        /* rel L2 < 1e-13 */
}

fft3d_plan *fft3d_create(int Lq, int batch)
{
    if (Lq != L || batch < 1) return NULL;
    fft3d_plan *p = calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->batch = batch;
    if (posix_memalign(&p->rawS, 4096, SVDBL * sizeof(double)) != 0) {
        free(p); return NULL;
    }
    p->S = (double *)p->rawS;
    memset(p->S, 0, SVDBL * sizeof(double));  /* pad columns stay 0 forever */
    if (posix_memalign(&p->rawP, 4096, (size_t)L * PPITCH * 2 * sizeof(double)) != 0) {
        free(p->rawS); free(p); return NULL;
    }
    p->P = (double *)p->rawP;
    memset(p->P, 0, (size_t)L * PPITCH * 2 * sizeof(double));
    p->fn = g_cands[0].fn;                            /* safe default        */

    int    live[NCAND];
    double tc[NCAND];
    for (int c = 0; c < NCAND; ++c) { live[c] = 1; tc[c] = 1e300; }

    /* run-time forcing for the monitor's control jobs */
    { const char *e;
      if ((e = getenv("FFT45_PW"))) {
          int v = atoi(e);
          for (int c = 0; c < NCAND; ++c) if (g_cands[c].pw != v) live[c] = 0;
      }
      if ((e = getenv("FFT45_MODE"))) {
          int v = (e[0] >= '0' && e[0] <= '9') ? atoi(e)
                : (e[0] == 's' ? M_SCRATCHP : M_INPLACE);
          for (int c = 0; c < NCAND; ++c) if (g_cands[c].mode != v) live[c] = 0;
      }
      if ((e = getenv("FFT45_PF"))) {
          int v = atoi(e);
          for (int c = 0; c < NCAND; ++c) if (g_cands[c].pf != v) live[c] = 0;
      }
      { int any = 0;
        for (int c = 0; c < NCAND; ++c) any |= live[c];
        if (!any) for (int c = 0; c < NCAND; ++c) live[c] = 1; } }

    /* tuning arena: must actually stream at large batch (L36_pfa r2 lesson);
     * 32 volumes = 2 x 44.5 MB in+out, past both machines' L3 on the walk */
    const int nv = batch < 32 ? batch : 32;
    void *ri = NULL, *ro = NULL, *r0 = NULL, *r1 = NULL;
    if (posix_memalign(&ri, 64, (size_t)nv * VDBL * sizeof(double)) ||
        posix_memalign(&ro, 64, (size_t)nv * VDBL * sizeof(double)) ||
        posix_memalign(&r0, 64, VDBL * sizeof(double)) ||
        (nv > 1 && posix_memalign(&r1, 64, VDBL * sizeof(double)))) {
        free(ri); free(ro); free(r0);
        snprintf(g_desc, sizeof g_desc,
                 "GT-PFA 9x5 two-sweep; tuner SKIPPED (arena alloc failed): %s",
                 g_cands[0].nm);
        return p;
    }
    double *tin = ri, *tout = ro, *ref0 = r0, *refN = r1;
    unsigned long long s = 0x9E3779B97F4A7C15ull;
    for (size_t i = 0; i < (size_t)nv * VDBL; ++i) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        tin[i] = (double)(long long)(s >> 11) * 0x1p-53;
    }
    ref3d((const double _Complex *)tin, (double _Complex *)ref0);
    if (nv > 1)          /* gate the LAST volume too: catches S-reuse bugs */
        ref3d((const double _Complex *)(tin + (size_t)(nv - 1) * VDBL),
              (double _Complex *)refN);

    for (int c = 0; c < NCAND; ++c) {
        if (!live[c]) continue;
        memset(tout, 0, (size_t)nv * VDBL * sizeof(double));
        g_cands[c].fn(tin, tout, nv, p->S, p->P);
        if (!rel_ok(tout, ref0, VDBL)) live[c] = 0;
        if (nv > 1 && live[c] &&
            !rel_ok(tout + (size_t)(nv - 1) * VDBL, refN, VDBL)) live[c] = 0;
    }
    /* small arenas get more interleaved rounds: wallaby toggles a fast/slow
     * machine state on a seconds scale, and a toggle edge crossing the
     * tournament mis-ranks kernels (observed here in r8: a slow-window race
     * installed sp-pfs at B=1, 205 vs 175 us).  Min over more rounds gives
     * every candidate a shot at the fast state.  (L36/L45_mixedradix's
     * tuner-hardening lesson.) */
    const int R = (nv >= 8) ? 1 : (nv >= 2 ? 3 : 8);
    const int NR = (nv >= 16) ? 4 : (nv >= 4 ? 6 : 10);
    for (int round = 0; round < NR; ++round)
        for (int c = 0; c < NCAND; ++c) {
            if (!live[c]) continue;
            /* self-warm so each candidate is timed from its own steady state */
            g_cands[c].fn(tin, tout, nv, p->S, p->P);
            double t0 = now_s();
            for (int r = 0; r < R; ++r)
                g_cands[c].fn(tin, tout, nv, p->S, p->P);
            double t = (now_s() - t0) / R;
            if (t < tc[c]) tc[c] = t;
        }
    int best = -1;
    for (int c = 0; c < NCAND; ++c)
        if (live[c] && (best < 0 || tc[c] < tc[best])) best = c;
    int pick = best < 0 ? 0 : best;
    if (best >= 0)
        for (int c = 0; c < NCAND; ++c)
            if (live[c] && tc[c] <= tc[best] * 1.03 &&
                g_cands[c].rank < g_cands[pick].rank) pick = c;
    p->fn = g_cands[pick].fn;
    snprintf(g_desc, sizeof g_desc,
             "GT-PFA 9x5 two-sweep; tuner pick: %s (B=%d, nv=%d)",
             g_cands[pick].nm, batch, nv);

#ifdef FFT45_LOUD
    if (1) {
#else
    if (getenv("FFT45_VERBOSE")) {
#endif
        for (int c = 0; c < NCAND; ++c)
            fprintf(stderr, "L45_pfa tuner: %-12s  %s  %.1f us/vol\n",
                    g_cands[c].nm, live[c] ? "ok " : "OUT",
                    live[c] ? tc[c] * 1e6 / nv : 0.0);
        fprintf(stderr, "L45_pfa tuner: chose %s (nv=%d)\n",
                g_cands[pick].nm, nv);
    }
    free(ri); free(ro); free(r0); free(r1);
    return p;
}

void fft3d_destroy(fft3d_plan *p)
{
    if (!p) return;
    free(p->rawS); free(p->rawP); free(p);
}

void fft3d_execute(fft3d_plan *plan, const double _Complex *in, double _Complex *out)
{
    plan->fn((const double *)in, (double *)out, plan->batch, plan->S, plan->P);
}

#else /* ================ KERNEL TEMPLATE, PW = 2 or 4 ====================== */

#define vec   CAT(vec_pw,  PW)
#define uvec  CAT(uvec_pw, PW)
#define veci  CAT(veci_pw, PW)
#define FN(n) CAT(n, CAT(_pw, PW))

typedef double    vec  __attribute__((vector_size(PW * 16)));
typedef double    uvec __attribute__((vector_size(PW * 16), aligned(8)));
typedef long long veci __attribute__((vector_size(PW * 16)));

#ifdef __clang__
# define VSH(a,b,...) __builtin_shufflevector(a, b, __VA_ARGS__)
#else
# define VSH(a,b,...) __builtin_shuffle(a, b, (veci){__VA_ARGS__})
#endif

/* in/out accesses are potentially 16-byte aligned only (720 B rows); the
 * plane scratch and padded S are always 64B-aligned but share the macro
 * (an aligned address through an unaligned op costs nothing on these cores) */
#define LDU(p)    ((vec)*(const uvec *)(p))
#define STU(p, v) (*(uvec *)(p) = (uvec)(v))

#define NGRP  ((L + PW - 1) / PW)     /* lane groups incl. overlap tail: 23/12 */
#define NFULL (L / PW)                /* full groups: 22/11                    */

#if PW == 4
# define VSPLAT(a)  ((vec){(a),(a),(a),(a),(a),(a),(a),(a)})
# define VPAIR(a,b) ((vec){(a),(b),(a),(b),(a),(b),(a),(b)})
# define SWAP(v)    VSH((v),(v), 1,0,3,2,5,4,7,6)
# define VFMA(a,b,c)  ((vec)_mm512_fmadd_pd((__m512d)(a),(__m512d)(b),(__m512d)(c)))
# define VFNMA(a,b,c) ((vec)_mm512_fnmadd_pd((__m512d)(a),(__m512d)(b),(__m512d)(c)))
/* one-complex masked load/store (lane 0), dead lanes ZERO (no denormal risk) */
# define LDT(p)    ((vec)_mm512_maskz_loadu_pd((__mmask8)0x03, (p)))
# define STT(p, v) _mm512_mask_storeu_pd((p), (__mmask8)0x03, (__m512d)(v))
/* gather/scatter one complex COLUMN (PW rows at `st` doubles apart) into/from
 * the lanes of one vector: the z-pass odd-column tail.  4 x 16 B accesses:
 * complex elements are 16B-aligned, never split a line. */
# define GCOL(dst, p, st) do {                                               \
    __m256d lo_ = _mm256_insertf128_pd(                                      \
        _mm256_castpd128_pd256(_mm_loadu_pd((p))),                           \
        _mm_loadu_pd((p) + (st)), 1);                                        \
    __m256d hi_ = _mm256_insertf128_pd(                                      \
        _mm256_castpd128_pd256(_mm_loadu_pd((p) + 2 * (st))),                \
        _mm_loadu_pd((p) + 3 * (st)), 1);                                    \
    (dst) = (vec)_mm512_insertf64x4(_mm512_castpd256_pd512(lo_), hi_, 1);    \
} while (0)
# define SCOL(src, p, st) do {                                               \
    __m512d v_ = (__m512d)(src);                                             \
    __m256d h_ = _mm512_extractf64x4_pd(v_, 1);                              \
    _mm_storeu_pd((p),            _mm512_castpd512_pd128(v_));               \
    _mm_storeu_pd((p) +     (st), _mm256_extractf128_pd(                     \
                                      _mm512_castpd512_pd256(v_), 1));       \
    _mm_storeu_pd((p) + 2 * (st), _mm256_castpd256_pd128(h_));               \
    _mm_storeu_pd((p) + 3 * (st), _mm256_extractf128_pd(h_, 1));             \
} while (0)
#elif PW == 2
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
#  define LDT(p)    ((vec)_mm256_maskload_pd((p),                            \
                         (__m256i){-1LL, -1LL, 0LL, 0LL}))
#  define STT(p, v) _mm256_maskstore_pd((p),                                 \
                         (__m256i){-1LL, -1LL, 0LL, 0LL}, (__m256d)(v))
#  define GCOL(dst, p, st)                                                   \
    ((dst) = (vec)_mm256_insertf128_pd(                                      \
        _mm256_castpd128_pd256(_mm_loadu_pd((p))),                           \
        _mm_loadu_pd((p) + (st)), 1))
#  define SCOL(src, p, st) do {                                              \
    _mm_storeu_pd((p),        _mm256_castpd256_pd128((__m256d)(src)));       \
    _mm_storeu_pd((p) + (st), _mm256_extractf128_pd((__m256d)(src), 1));     \
} while (0)
# else  /* portable fallbacks so a pre-AVX build still compiles */
#  define LDT(p)    ((vec){(p)[0], (p)[1], 0.0, 0.0})
#  define STT(p, v) do { (p)[0] = (v)[0]; (p)[1] = (v)[1]; } while (0)
#  define GCOL(dst, p, st)                                                   \
    ((dst) = (vec){(p)[0], (p)[1], (p)[(st)], (p)[(st) + 1]})
#  define SCOL(src, p, st) do {                                              \
    (p)[0] = (src)[0]; (p)[1] = (src)[1];                                    \
    (p)[(st)] = (src)[2]; (p)[(st) + 1] = (src)[3];                          \
} while (0)
# endif
#endif

/* PW x PW transpose of 128-bit complex granules (involution) */
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
#elif PW == 2
# define TRNC(r, c) do {                                                     \
    vec u0_ = VSH((r)[0], (r)[1], 0,1,4,5);                                  \
    vec u1_ = VSH((r)[0], (r)[1], 2,3,6,7);                                  \
    (c)[0] = u0_; (c)[1] = u1_;                                              \
} while (0)
#endif

/* 9-point DFT on interleaved-complex vectors: genfft's FMA n1_9 DAG
 * (fftw-3.3.10, 24 add + 56 fma = 80 scalar FMA-port ops) transcribed
 * pairwise -- each scalar re/im line pair is one vector op; each re/im
 * crossing (mult by i, and the (1 +- c*i) spiral factors on p/q) is one
 * SWAP with the signs folded into a VPAIR constant.
 * 40 FMA-port ops + 12 swaps (the CT 3x3 it replaces: 44 + 10).
 * Scalar-to-vector key, stage A (radix-3 columns {n, n+3, n+6}):
 *   sJ_ = column sum, SJ_ = full sum, aJ_ = xJ - sJ/2, eJ_ = x(J+3)-x(J+6),
 *   iJ_ = SWAP(eJ_);  p/q = aJ -+ 866*i*eJ  (the two rotated DFT3 outputs).
 * Blocks: k={0,3,6} is a DFT3 on the sums; k={1,4,7} and k={2,5,8} build
 * w = (1 +- c*i)*p (one SWAP+FMA each), cross them (u, z), and fan out. */
#define DFT9F(x0,x1,x2,x3,x4,x5,x6,x7,x8, o0,o1,o2,o3,o4,o5,o6,o7,o8) do {   \
    vec s0_ = (x3) + (x6), e0_ = (x3) - (x6);                                \
    vec S0_ = (x0) + s0_,  a0_ = VFNMA(s0_, VSPLAT(0.5), (x0));              \
    vec i0_ = SWAP(e0_);                                                     \
    vec s1_ = (x4) + (x7), e1_ = (x4) - (x7);                                \
    vec S1_ = (x1) + s1_,  a1_ = VFNMA(s1_, VSPLAT(0.5), (x1));              \
    vec i1_ = SWAP(e1_);                                                     \
    vec p1_ = VFMA (i1_, VPAIR(KS3, -KS3), a1_);                             \
    vec q1_ = VFNMA(i1_, VPAIR(KS3, -KS3), a1_);                             \
    vec s2_ = (x5) + (x8), e2_ = (x5) - (x8);                                \
    vec S2_ = (x2) + s2_,  a2_ = VFNMA(s2_, VSPLAT(0.5), (x2));              \
    vec i2_ = SWAP(e2_);                                                     \
    vec p2_ = VFMA (i2_, VPAIR(KS3, -KS3), a2_);                             \
    vec q2_ = VFNMA(i2_, VPAIR(KS3, -KS3), a2_);                             \
    /* k = 0, 3, 6: DFT3 on the column sums */                               \
    vec sg_ = S1_ + S2_, d3_ = S2_ - S1_, id_ = SWAP(d3_);                   \
    vec b0_ = VFNMA(sg_, VSPLAT(0.5), S0_);                                  \
    (o0) = S0_ + sg_;                                                        \
    (o3) = VFNMA(id_, VPAIR(KS3, -KS3), b0_);                                \
    (o6) = VFMA (id_, VPAIR(KS3, -KS3), b0_);                                \
    /* k = 1, 4, 7 */                                                        \
    {                                                                        \
    vec v1_ = VFMA (i0_, VPAIR(KS3, -KS3), a0_);                             \
    vec w2_ = VFMA (SWAP(p2_), VPAIR(-K176, K176), p2_);                     \
    vec w1_ = VFMA (SWAP(p1_), VPAIR(K839, -K839), p1_);                     \
    vec u1_ = VFMA (w1_, VPAIR(K777, -K777), SWAP(w2_));                     \
    vec z1_ = VFMA (SWAP(w1_), VPAIR(K777, -K777), w2_);                     \
    (o1) = VFMA (u1_, VPAIR(K984, -K984), v1_);                              \
    vec r1_ = VFNMA(u1_, VPAIR(K492, -K492), v1_);                           \
    (o4) = VFMA (z1_, VSPLAT(K852), r1_);                                    \
    (o7) = VFNMA(z1_, VSPLAT(K852), r1_);                                    \
    }                                                                        \
    /* k = 2, 5, 8 */                                                        \
    {                                                                        \
    vec v2_ = VFNMA(i0_, VPAIR(KS3, -KS3), a0_);                             \
    vec wA_ = VFMA (q1_, VPAIR(K176, -K176), SWAP(q1_));                     \
    vec wB_ = VFNMA(SWAP(q2_), VPAIR(K363, -K363), q2_);                     \
    vec uB_ = VFMA (wB_, VPAIR(-K954, K954), wA_);                           \
    vec zB_ = VFMA (SWAP(wB_), VPAIR(-K954, K954), SWAP(wA_));               \
    (o2) = VFMA (uB_, VPAIR(K984, -K984), v2_);                              \
    vec rB_ = VFNMA(uB_, VPAIR(K492, -K492), v2_);                           \
    (o5) = VFNMA(zB_, VSPLAT(K852), rB_);                                    \
    (o8) = VFMA (zB_, VSPLAT(K852), rB_);                                    \
    }                                                                        \
} while (0)

/* The 45-point Good-Thomas 9x5 codelet over PW interleaved-complex lanes.
 * LD(n) must yield input element n as a vec rvalue; ST(k, v) consumes output
 * element k.  Both index maps fold to compile-time constants once the loops
 * unroll.  ALL LDs (stage 1) complete before any ST (stage 2), so LD/ST may
 * alias freely.  Stage 1: 5 x DFT9 (genfft n1_9 DAG, 40 FMA-port ops);
 * stage 2: 9 x DFT5 (16 ops).  344 FMA-port ops + 78 shuffles per call. */
#define PFA45(LD, ST) do {                                                   \
    vec A_[45];                            /* A_[5*k1 + n2] */               \
    _Pragma("GCC unroll 5")                                                  \
    for (int n2_ = 0; n2_ < 5; ++n2_) {                                      \
        vec g_[9];                                                           \
        _Pragma("GCC unroll 9")                                              \
        for (int n1_ = 0; n1_ < 9; ++n1_)                                    \
            g_[n1_] = LD((5 * n1_ + 9 * n2_) % 45);                          \
        DFT9F(g_[0], g_[1], g_[2], g_[3], g_[4], g_[5], g_[6], g_[7], g_[8],\
              A_[      n2_], A_[ 5 + n2_], A_[10 + n2_],                     \
              A_[15 + n2_], A_[20 + n2_], A_[25 + n2_],                      \
              A_[30 + n2_], A_[35 + n2_], A_[40 + n2_]);                     \
    }                                                                        \
    _Pragma("GCC unroll 9")                                                  \
    for (int k1_ = 0; k1_ < 9; ++k1_) {                                      \
        const vec *f_ = A_ + 5 * k1_;                                        \
        vec t1_ = f_[1] + f_[4], t4_ = f_[1] - f_[4];                        \
        vec t2_ = f_[2] + f_[3], t7_ = f_[2] - f_[3];                        \
        vec te_ = t1_ + t2_,     ta_ = t1_ - t2_;                            \
        ST((10 * k1_      ) % 45, f_[0] + te_);                              \
        vec tm_ = VFNMA(te_, VSPLAT(0.25), f_[0]);                           \
        vec tp_ = VFMA (ta_, VSPLAT(K59), tm_);                              \
        vec tq_ = VFNMA(ta_, VSPLAT(K59), tm_);                              \
        vec tv_ = VFMA (t7_, VSPLAT(KIG), t4_);                              \
        vec tw_ = VFNMA(t4_, VSPLAT(KIG), t7_);                              \
        vec sv_ = SWAP(tv_), sw_ = SWAP(tw_);                                \
        ST((10 * k1_ +  36) % 45, VFMA (sv_, VPAIR(KS5, -KS5), tp_));        \
        ST((10 * k1_ +  72) % 45, VFNMA(sw_, VPAIR(KS5, -KS5), tq_));        \
        ST((10 * k1_ + 108) % 45, VFMA (sw_, VPAIR(KS5, -KS5), tq_));        \
        ST((10 * k1_ + 144) % 45, VFNMA(sv_, VPAIR(KS5, -KS5), tp_));        \
    }                                                                        \
} while (0)

/* paced-prefetch step: cover one x-plane (PLND doubles = 507 lines) in
 * 2*NGRP steps -> 22 lines/step at PW=4, 12 at PW=2 */
#define PFL    ((PLND + 16 * NGRP - 1) / (16 * NGRP))
#define PFSTEP (8 * PFL)
#define PFIN(p) do {                                                          \
    _Pragma("GCC unroll 24")                                                  \
    for (int q_ = 0; q_ < PFL; ++q_)                                          \
        __builtin_prefetch((p) + 8 * q_, 0, FFT45_PFH);                       \
} while (0)
/* ip-pf3: write-intent (prefetchw) cursor over the mid-plane stores */
#define PFWMID(p) do {                                                        \
    _Pragma("GCC unroll 24")                                                  \
    for (int q_ = 0; q_ < PFL; ++q_)                                          \
        __builtin_prefetch((p) + 8 * q_, 1, 3);                               \
} while (0)

/* phase 1, ONE x-plane: z transform (transposed lanes) into plane scratch,
 * then y transform into mid[x][ky][kz].  pfr/pfw are compile-time constants
 * at every instantiation site (always_inline + const), so the pf0 execs
 * carry NO prefetch code and no cursor arithmetic.  mrow/mpln are the mid
 * row/plane strides in doubles: (90, 4050) inplace, (104, 4680) scratchp --
 * also compile-time, so every mid access folds to base + literal. */
static inline __attribute__((always_inline))
void FN(phase1_plane)(const double *restrict in, double *restrict mid,
                      double *restrict pld, int x, const int pfr,
                      const int pfw, const long mrow, const long mpln)
{
    const double *pfc = pfr ? in  + FFT45_PFD  + (size_t)x * PLND : 0;
    double       *pwc = pfw ? mid + FFT45_PFWD + (size_t)x * mpln : 0;
    const double *px  = in  + (size_t)x * PLND;
    double       *mx  = mid + (size_t)x * mpln;

    for (int yg = 0; yg < NGRP; ++yg) {
        const int yb = (yg == NGRP - 1) ? (L - PW) : yg * PW;
        /* ONE runtime base per block; every access below is base + a
         * compile-time constant (the r7 single-base fix: folding yb into
         * each address made gcc spill a 48-entry offset table, -14%). */
        const double *rows = px  + (size_t)yb * (2 * L);
        double       *prow = pld + (size_t)yb * (2 * PPITCH);
        if (pfr) { PFIN(pfc);   pfc += PFSTEP; }
        if (pfw) { PFWMID(pwc); pwc += PFSTEP; }
        vec Zv[L], Wv[L];
        _Pragma("GCC unroll 22")
        for (int zg = 0; zg < NFULL; ++zg) {
            const int zb = zg * PW;
            vec r_[PW];
            _Pragma("GCC unroll 4")
            for (int j = 0; j < PW; ++j)
                r_[j] = LDU(rows + (size_t)j * (2 * L) + 2 * zb);
            TRNC(r_, &Zv[zb]);
        }
        /* odd 45th column via 16 B column gather (no overlapped split loads) */
        GCOL(Zv[L - 1], rows + 2 * (L - 1), 2 * L);
#define LD1(n)    Zv[n]
#define ST1(k, v) (Wv[k] = (v))
        PFA45(LD1, ST1);
#undef LD1
#undef ST1
        _Pragma("GCC unroll 22")
        for (int zg = 0; zg < NFULL; ++zg) {
            const int zb = zg * PW;
            vec r_[PW];
            TRNC(&Wv[zb], r_);
            _Pragma("GCC unroll 4")
            for (int j = 0; j < PW; ++j)
                STU(prow + (size_t)j * (2 * PPITCH) + 2 * zb, r_[j]);
        }
        /* odd column scatter: lane j of Wv[44] -> row yb+j, 16 B stores */
        SCOL(Wv[L - 1], prow + 2 * (L - 1), 2 * PPITCH);
    }

    for (int zg = 0; zg < NGRP; ++zg) {
        const int zb = (zg == NGRP - 1) ? (L - PW) : zg * PW;
        const double *pcol = pld + 2 * zb;      /* single runtime base, again */
        double       *mcol = mx  + 2 * zb;
        /* opaque-base barrier: without it gcc re-associates every load
         * address as pld + (zb*16) + n*832, hoists the 45 loop-invariant
         * (pld + n*832) leas, SPILLS them, and reloads one per vector load
         * -- the r6 offset-table pathology relocated into the y-subloop
         * (measured here: 48 leas + 37 GPR spills per exec, -163 instr and
         * all spill-reload serialization gone with the barrier). */
        __asm__("" : "+r"(pcol), "+r"(mcol));
        if (pfr) { PFIN(pfc);   pfc += PFSTEP; }
        if (pfw) { PFWMID(pwc); pwc += PFSTEP; }
#define LD2(n)    LDU(pcol + (size_t)(n) * (2 * PPITCH))
#define ST2(k, v) STU(mcol + (size_t)(k) * mrow, (v))
        PFA45(LD2, ST2);
#undef LD2
#undef ST2
    }
}

static inline __attribute__((always_inline))
void FN(phase1)(const double *restrict in, double *restrict mid,
                double *restrict pld, const int pfr, const int pfw,
                const long mrow, const long mpln)
{
    for (int x = 0; x < L; ++x)
        FN(phase1_plane)(in, mid, pld, x, pfr, pfw, mrow, mpln);
}

/* phase-2 read prefetch: 45 sequential streams at PLND-double stride is more
 * than the L2 streamer tracks (a runtime pf level in r7; compile-time now) */
#define PF45(s_) do {                                                        \
    _Pragma("GCC unroll 45")                                                 \
    for (int n_ = 0; n_ < 45; ++n_)                                          \
        __builtin_prefetch((s_) + (size_t)n_ * PLND + 8, 0, 3);              \
} while (0)
/* sp-pfs: write-intent prefetch of the 45 cold out-streams one tile ahead */
#define PFW45(d_) do {                                                       \
    _Pragma("GCC unroll 45")                                                 \
    for (int n_ = 0; n_ < 45; ++n_)                                          \
        __builtin_prefetch((d_) + (size_t)n_ * PLND + 8, 1, 3);              \
} while (0)
/* per-tile pre-coverage of the NEXT volume's input (first ~63 KB) */
#define PFNX() do { if (pfn && pn_) {                                         \
    _Pragma("GCC unroll 4")                                                   \
    for (int q_ = 0; q_ < FFT45_PFN; ++q_)                                    \
        __builtin_prefetch(pn_ + 8 * q_, 0, FFT45_PFH);                       \
    pn_ += 8 * FFT45_PFN; } } while (0)

/* phase 2, INPLACE: x transform in place in `out` (the codelet reads all 45
 * inputs before its first store).  Tiled over the FLAT (y,z) index:
 * 2025 = 506*4 + 1 (PW=4) leaves ONE masked tail call per volume.
 * (Flat tiling borrowed from L45_mixedradix r7.) */
static inline __attribute__((always_inline))
void FN(phase2)(const double *mid, double *out, const double *pnext,
                const int pfx, const int pfn)
{
    const double *pn_ = pnext;
    for (int t = 0; t < NPLANE / PW; ++t) {
        const size_t o = (size_t)t * (PW * 2);
        const double *s_ = mid + o;
        double       *d_ = out + o;
        if (pfx) PF45(s_);
        PFNX();
#define LD3(n)    LDU(s_ + (size_t)(n) * PLND)
#define ST3(k, v) STU(d_ + (size_t)(k) * PLND, (v))
        PFA45(LD3, ST3);
#undef LD3
#undef ST3
    }
    {   /* tail: flat line 2024 = (y,z) = (44,44); lane 0 live, dead lanes 0 */
        const double *s_ = mid + (size_t)(NPLANE - 1) * 2;
        double       *d_ = out + (size_t)(NPLANE - 1) * 2;
#define LD4(n)    LDT(s_ + (size_t)(n) * PLND)
#define ST4(k, v) STT(d_ + (size_t)(k) * PLND, (v))
        PFA45(LD4, ST4);
#undef LD4
#undef ST4
    }
}

/* phase 2, SCRATCHP: x transform S -> out, out of place.  S rows are padded
 * (SPITCH complex), so the flat (y,z) index does not map affinely to out and
 * tiling is per y-row: NFULL full tiles + ONE OVERLAPPING tile (starts at
 * z = 45-PW; recompute is idempotent because the pass is out of place).
 * All S loads are 64B-aligned; only the out stores pay the odd-L split toll,
 * and pfw covers their RFO in the streaming variant. */
static inline __attribute__((always_inline))
void FN(phase2s)(const double *S, double *out, const double *pnext,
                 const int pfw, const int pfn)
{
    const double *pn_ = pnext;
    for (int y = 0; y < L; ++y) {
        const double *sy = S   + (size_t)y * SROWD;
        double       *oy = out + (size_t)y * (2 * L);
        for (int tg = 0; tg < NGRP; ++tg) {
            const int zb = (tg == NGRP - 1) ? (L - PW) : tg * PW;
            const double *s_ = sy + 2 * zb;
            double       *d_ = oy + 2 * zb;
            if (pfw) PFW45(d_);
            PFNX();
#define LD5(n)    LDU(s_ + (size_t)(n) * SPLND)
#define ST5(k, v) STU(d_ + (size_t)(k) * PLND, (v))
            PFA45(LD5, ST5);
#undef LD5
#undef ST5
        }
    }
}

/* ---- exec variants: every pf/mode/stride argument below is a literal, so
 * const-propagation through the always_inline bodies specializes each one
 * completely (the L45_mixedradix structure). ------------------------------ */

static void FN(x_ip0)(const double *in, double *out, long nvol,
                      double *S, double *P)
{
    (void)S;
    for (long b = 0; b < nvol; ++b) {
        const double *i = in  + (size_t)b * VDBL;
        double       *o = out + (size_t)b * VDBL;
        FN(phase1)(i, o, P, 0, 0, 2 * L, PLND);
        FN(phase2)(o, o, 0, 0, 0);
    }
}

#if PW == 4      /* the pw2 tournament pool only carries ip0/ip3/sp0/sps */
static void FN(x_ip1)(const double *in, double *out, long nvol,
                      double *S, double *P)
{
    (void)S;
    for (long b = 0; b < nvol; ++b) {
        const double *i = in  + (size_t)b * VDBL;
        double       *o = out + (size_t)b * VDBL;
        FN(phase1)(i, o, P, 0, 0, 2 * L, PLND);
        FN(phase2)(o, o, 0, 1, 0);
    }
}

static void FN(x_ip2)(const double *in, double *out, long nvol,
                      double *S, double *P)
{
    (void)S;
    for (long b = 0; b < nvol; ++b) {
        const double *i = in  + (size_t)b * VDBL;
        double       *o = out + (size_t)b * VDBL;
        const double *nx = (b + 1 < nvol) ? i + VDBL : 0;
        FN(phase1)(i, o, P, 1, 0, 2 * L, PLND);
        FN(phase2)(o, o, nx, 1, 1);
    }
}
#endif

static void FN(x_ip3)(const double *in, double *out, long nvol,
                      double *S, double *P)
{
    (void)S;
    for (long b = 0; b < nvol; ++b) {
        const double *i = in  + (size_t)b * VDBL;
        double       *o = out + (size_t)b * VDBL;
        const double *nx = (b + 1 < nvol) ? i + VDBL : 0;
        FN(phase1)(i, o, P, 1, 1, 2 * L, PLND);
        FN(phase2)(o, o, nx, 1, 1);
    }
}

static void FN(x_sp0)(const double *in, double *out, long nvol,
                      double *S, double *P)
{
    for (long b = 0; b < nvol; ++b) {
        const double *i = in  + (size_t)b * VDBL;
        double       *o = out + (size_t)b * VDBL;
        FN(phase1)(i, S, P, 0, 0, SROWD, SPLND);
        FN(phase2s)(S, o, 0, 0, 0);
    }
}

static void FN(x_sps)(const double *in, double *out, long nvol,
                      double *S, double *P)
{
    for (long b = 0; b < nvol; ++b) {
        const double *i = in  + (size_t)b * VDBL;
        double       *o = out + (size_t)b * VDBL;
        const double *nx = (b + 1 < nvol) ? i + VDBL : 0;
        FN(phase1)(i, S, P, 1, 0, SROWD, SPLND);
        FN(phase2s)(S, o, nx, 1, 1);
    }
}

#undef PF45
#undef PFW45
#undef PFNX
#undef PFIN
#undef PFWMID
#undef PFSTEP
#undef PFL

#undef PFA45
#undef DFT9F
#undef TRNC
#undef LDT
#undef STT
#undef GCOL
#undef SCOL
#undef VFMA
#undef VFNMA
#undef SWAP
#undef VPAIR
#undef VSPLAT
#undef NFULL
#undef NGRP
#undef LDU
#undef STU
#undef VSH
#undef FN
#undef veci
#undef uvec
#undef vec

#endif /* template */
