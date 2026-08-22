/* L45_pfa.c -- forward complex 3D DFT of a 45^3 cube, batched, out-of-place.
 *
 * ROUND panel_r6, first implementation for this geometry.
 *
 * TECHNIQUE
 *   Row-column 3D DFT; every 45-point line is a Good-Thomas / prime-factor
 *   9 x 5 codelet, on INTERLEAVED complex vectors whose lanes are a spectator
 *   axis.  gcd(9,5) = 1, so with
 *
 *       input  (Ruritanian): n = (5*n1 + 9*n2) mod 45     n1 in [0,9), n2 in [0,5)
 *       output (CRT):        k = (10*k1 + 36*k2) mod 45   (10 = 5*[5^-1]_9,
 *                                                          36 = 9*[9^-1]_5)
 *
 *   W45^{nk} = W9^{n1 k1} * W5^{n2 k2} exactly: the 45-point DFT is 5 DFT9s
 *   then 9 DFT5s with NO twiddles in between, and both index maps fold into
 *   compile-time addressing.  The DFT9 is Cooley-Tukey 3x3 (44 FMA-port ops:
 *   6 DFT3 at 6 + 4 twiddle CMULs at 2); the DFT5 is FFTW n1_5's FMA form
 *   (16 FMA-port ops: the sqrt(5)/4 cosine split and the s2/s1-scaled-sine
 *   trick save 2 ops over the plain two-rotation form).  Per 45-point line:
 *   5*44 + 9*16 = 364 FMA-port vector ops + 68 shuffles over PW lanes.
 *
 *   Two sweeps over the volume (the structure that won L=36 on the node,
 *   adopted from L36_pfa r2 <- L36_mixedradix r1):
 *
 *   phase 1, per x-plane (45x45 complex = 31.6 KB + padded plane scratch):
 *       z transform: lanes = PW y-rows, PWxPW complex-granule register
 *                    transposes on load and store, into plane scratch
 *                    pl[y][kz] (row pitch padded to 48 complex = 64B rows)
 *       y transform: lanes = PW kz (contiguous in pl), store to mid[x][ky][kz]
 *   phase 2:
 *       x transform: lanes = PW kz, stride 4050 doubles between the 45
 *                    x-streams, mid -> out (in place when mid == out)
 *
 *   45 is odd, so no vector width divides it.  Tail policy:
 *     * out-of-place subpasses (z: in->pl, y: pl->mid) OVERLAP the last
 *       group (starts at 45-PW); overlapped lines are recomputed with
 *       identical results.  Costs one extra group in 12 at PW=4 (+6.7%),
 *       needs no second codelet.
 *     * the in-place x pass cannot overlap (it would re-read transformed
 *       columns); its tail column kz=44 runs a PW=1 (SSE, one complex per
 *       vector) instantiation of the same codelet template, 45 lines/volume
 *       (~3% of the arithmetic at PW=4).
 *   All in/out/mid accesses use unaligned vector ops: 45 complex = 720 B row
 *   stride rotates alignment mod 64 (odd-L fact of life; the plane scratch's
 *   768 B pitch keeps the hot in-plane accesses aligned).
 *
 *   MODES AND PREFETCH (tournament-gated candidates, never defaults):
 *     mode: mid = out (INPLACE) or mid = one reused plan-owned volume
 *           (SCRATCH; S stays cache-resident across the batch, out is
 *           written once, normally).  NO NT-store or pipe mode: the node
 *           rejected NT at L=36 in four consecutive rounds, and 720 B row
 *           alignment rotation makes 64B-aligned stream stores structurally
 *           impossible here anyway.
 *     pf=1: paced T1 prefetch of phase 1's linear in-stream, FFT45_PFD = 32 KB
 *           ahead, spread over both subloops (2*NGRP steps/plane); plus
 *           per-tile pre-coverage of the NEXT volume's first ~63 KB from
 *           phase 2 (PFNX).  (L36_pfa r3's PFIN/PFNX.)
 *     pf=2: pf=1 plus WRITE-INTENT prefetch (prefetchw) of whichever pass
 *           stores cold lines: phase 1's mid-plane stream when INPLACE
 *           (one plane ahead), phase 2's 45 out-streams when SCRATCH.
 *           (L36_pfa r5's PFWMID/PFW36 <- L6_unrolled r3.)
 *     Phase 2 always hand-prefetches its 45 x-read-streams one line ahead
 *     (more streams than the L2 streamer tracks; dropping this cost L36 14%).
 *   fft3d_create() times {pw2, pw4} x {inplace, scratch} x {pf0,1,2}, gates
 *   every candidate against a scalar O(n^2)-per-line reference DFT at 1e-13,
 *   installs the fastest with a 3% simplest-first hysteresis, and reports the
 *   pick in fft3d_description().  FFT45_PW / FFT45_MODE / FFT45_PF env vars
 *   force the choice at plan time for the monitor's control runs.
 *
 * ATTRIBUTION
 *   - Two-sweep plane-fused structure, interleaved-complex spectator lanes,
 *     6-op DFT3 / 2-op CMUL forms, TRNC granule transpose, PFIN/PFNX paced
 *     read prefetch, pf=2 prefetchw, tuner + 3% hysteresis + env forcing +
 *     pick reporting: L36_pfa (r2-r5), transitively L36_mixedradix r1 and
 *     L6_unrolled r3.
 *   - Overlap-tail for out-of-place subpasses + separate tail path for the
 *     in-place x pass: converged with L45_mixedradix (this round, read after
 *     my design; their header documents the same policy with a masked tail
 *     where this file uses a PW=1 codelet).
 *   - 16-op DFT5: FFTW n1_5's DAG (sqrt(5)/4 split, s2 = KIG*s1), from the
 *     corpus; 2 FMA-port ops/DFT5 fewer than the plain two-rotation form.
 *
 * OPERATION COUNT (PW=4)
 *   Per volume: 3*2025 = 6075 lines; vector codelets = 45*(12+12+11) = 1575
 *   at 364 FMA-port ops = 573,300 zmm FMA-port ops + 45 PW=1 tail lines
 *   (16.4k SSE ops) + ~211k port-5 shuffles (codelet swaps + z-pass granule
 *   transposes).  On the 1-FMA-unit Gold 5218 the port-0 floor is ~573k
 *   cycles; wallaby (2 FMA pipes) halves that but shares port 5.
 *
 * ACCURACY: expect ~3-4e-16 relative L2 vs numpy (same module family as the
 * L=36 entries).
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

#define L      45
#define NPLANE 2025              /* 45*45 complex per x-plane                  */
#define PLND   4050              /* doubles per x-plane = x-stream stride      */
#define VDBL   ((size_t)2 * L * NPLANE)  /* doubles per volume = 182250        */
/* plane-scratch row pitch in complex.  52 complex = 832 B = 13 cache lines:
 * 13 is coprime with 64 sets, so the y-pass column walk (45 rows, one line
 * per row) spreads over all L1 sets; 48 complex = 12 lines lands the whole
 * column on 16 sets (gcd(12,64) = 4).  Rows stay 64-byte aligned either way. */
#ifndef PPITCH
# define PPITCH 52
#endif

#define CAT_(a,b) a##b
#define CAT(a,b)  CAT_(a,b)

/* phase-1 input prefetch distance, doubles (32 KB): must exceed the ~16 KB
 * pacing deficit the z-subloop accumulates (it consumes ~2x the cursor rate;
 * the y-subloop catches up), small enough not to churn the node's 1 MB L2. */
#ifndef FFT45_PFD
# define FFT45_PFD 4096
#endif
/* prefetch hint: 3=T0, 2=T1 (L2+, the measured winner at L=36/L=6), 0=NTA */
#ifndef FFT45_PFH
# define FFT45_PFH 2
#endif
/* cache lines of the NEXT volume's input prefetched per phase-2 tile:
 * 45 y * 11 tiles * 2 lines * 64 B = 63 KB > PFD + deficit. */
#ifndef FFT45_PFN
# define FFT45_PFN 2
#endif
/* pf=2 INPLACE: write-intent cursor lead over the mid-plane stores, doubles
 * (one x-plane = 31.6 KB): every line 0.5-1.5 planes of lead. */
#ifndef FFT45_PFWD
# define FFT45_PFWD 4050
#endif

/* W3: sqrt(3)/2 */
#define KS3  0.86602540378443864676372317075294
/* W9^m = cos(2*pi*m/9) - i*sin(2*pi*m/9), m = 1, 2, 4 */
#define W1R  0.76604444311897803520239265055542
#define W1I (-0.64278760968653932632264340990726)
#define W2R  0.17364817766693034885171662676931
#define W2I (-0.98480775301220805936674302458952)
#define W4R (-0.93969262078590838405410927732473)
#define W4I (-0.34202014332566873304409961468226)
/* 5-point module constants (FFTW n1_5 form) */
#define K59  0.55901699437494742410229341718282   /* sqrt(5)/4                 */
#define KIG  0.61803398874989484820458683436564   /* sin(4pi/5)/sin(2pi/5)     */
#define KS5  0.95105651629515357211665325776975   /* sin(2pi/5)                */

/* instantiate the kernel template: PW=1 (tail), PW=2, and PW=4 on AVX-512 */
#define PW 1
#include __FILE__
#undef PW
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

enum { M_INPLACE = 0, M_SCRATCH = 1 };
static const char *const mode_name[] = {"inplace", "scratch"};

struct fft3d_plan {
    int     batch;
    int     pw;                  /* 2 or 4                          */
    int     mode;                /* one of M_*                      */
    int     pf;                  /* 0 none, 1 read, 2 read+write    */
    double *S;                   /* reused scratch volume           */
    double *P;                   /* plane scratch: page-aligned heap, NOT the
                                    stack -- a stack plane gets a random page
                                    offset per process (ASLR) and 4K-aliases
                                    the in/out streams in unlucky runs
                                    (measured: bimodal 204 vs 377 us at B=1) */
    void   *rawS, *rawP;
};

const char *fft3d_name(void) { return "L45_pfa"; }

static char g_desc[224];

const char *fft3d_description(void)
{
    return g_desc[0] ? g_desc
                     : "Good-Thomas PFA 9x5, interleaved-complex lanes, two "
                       "sweeps, overlap tails + PW1 x-tail; {inplace, reused "
                       "scratch} x {pw, pf} autotuned in create()";
}
int fft3d_supports(int Lq) { return Lq == L; }

static void run_vols(int pw, int mode, int pf, double *S, double *P,
                     const double *in, double *out, int nvol)
{
    const int p1w = (pf == 2 && mode == M_INPLACE);
    const int p2w = (pf == 2 && mode == M_SCRATCH);
    const int pfr = (pf >= 1);
    for (int b = 0; b < nvol; ++b) {
        const double *i = in  + (size_t)b * VDBL;
        double       *o = out + (size_t)b * VDBL;
        double     *mid = (mode == M_INPLACE) ? o : S;
        const double *nx = (pf >= 1 && b + 1 < nvol) ? i + VDBL : NULL;
        /* FFT45_SKIP1/2: temporary diagnostics, wrong answers, timing only */
#ifndef FFT45_SKIP1
# define P1_(f, a, b, c, d, e) f(a, b, c, d, e)
#else
# define P1_(f, a, b, c, d, e) ((void)0)
#endif
#ifndef FFT45_SKIP2
# define P2_(f, a, b, c, d) f(a, b, c, d)
#else
# define P2_(f, a, b, c, d) ((void)0)
#endif
#ifdef HAVE_PW4
        if (pw == 4) { P1_(phase1_pw4, i, mid, P, pfr, p1w); P2_(phase2_pw4, mid, o, nx, p2w); }
        else
#endif
        { (void)pw; P1_(phase1_pw2, i, mid, P, pfr, p1w); P2_(phase2_pw2, mid, o, nx, p2w); }
    }
}

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

/* hysteresis rank: lower = simpler (3% band; L36_pfa r4's tuner lesson) */
static int cand_rank(int mode, int pf) { return pf * 2 + mode; }

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
    if (posix_memalign(&p->rawP, 4096, (size_t)L * PPITCH * 2 * sizeof(double)) != 0) {
        free(p->rawS); free(p); return NULL;
    }
    p->P = (double *)p->rawP;
    memset(p->P, 0, (size_t)L * PPITCH * 2 * sizeof(double));
#ifdef HAVE_PW4
    p->pw = 4;
#else
    p->pw = 2;
#endif
    p->mode = M_INPLACE; p->pf = (batch >= 8) ? 1 : 0;   /* safe default */

    struct cand { int pw, mode, pf; } cands[16];
    int nc = 0;
    for (int pf = 0; pf <= 2; ++pf) {
        cands[nc++] = (struct cand){2, M_INPLACE, pf};
        cands[nc++] = (struct cand){2, M_SCRATCH, pf};
#ifdef HAVE_PW4
        cands[nc++] = (struct cand){4, M_INPLACE, pf};
        cands[nc++] = (struct cand){4, M_SCRATCH, pf};
#endif
    }
    /* run-time forcing for the monitor's control jobs */
    { const char *e;
      if ((e = getenv("FFT45_PW"))) {
          int v = atoi(e), w = 0;
          for (int c = 0; c < nc; ++c) if (cands[c].pw == v) cands[w++] = cands[c];
          if (w) nc = w;
      }
      if ((e = getenv("FFT45_MODE"))) {
          int num = (e[0] >= '0' && e[0] <= '9'), w = 0;
          for (int c = 0; c < nc; ++c)
              if (num ? cands[c].mode == atoi(e)
                      : !strcmp(mode_name[cands[c].mode], e)) cands[w++] = cands[c];
          if (w) nc = w;
      }
      if ((e = getenv("FFT45_PF"))) {
          int v = atoi(e), w = 0;
          for (int c = 0; c < nc; ++c) if (cands[c].pf == v) cands[w++] = cands[c];
          if (w) nc = w;
      } }

    /* tuning arena: must actually stream at large batch (L36_pfa r2 lesson);
     * 32 volumes = 2 x 44.5 MB in+out, past both machines' L3 on the walk */
    const int nv = batch < 32 ? batch : 32;
    void *ri = NULL, *ro = NULL, *rr = NULL;
    if (posix_memalign(&ri, 64, (size_t)nv * VDBL * sizeof(double)) ||
        posix_memalign(&ro, 64, (size_t)nv * VDBL * sizeof(double)) ||
        posix_memalign(&rr, 64, VDBL * sizeof(double))) {
        free(ri); free(ro);
        snprintf(g_desc, sizeof g_desc,
                 "GT-PFA 9x5 two-sweep; tuner SKIPPED (arena alloc failed): "
                 "pw=%d mode=%s pf=%d", p->pw, mode_name[p->mode], p->pf);
        return p;
    }
    double *tin = ri, *tout = ro, *refv = rr;
    unsigned long long s = 0x9E3779B97F4A7C15ull;
    for (size_t i = 0; i < (size_t)nv * VDBL; ++i) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        tin[i] = (double)(long long)(s >> 11) * 0x1p-53;
    }
    ref3d((const double _Complex *)tin, (double _Complex *)refv);

    int    ok[16];
    double tc[16];
    for (int c = 0; c < nc; ++c) {
        run_vols(cands[c].pw, cands[c].mode, cands[c].pf, p->S, p->P, tin, tout, nv);
        double num = 0.0, den = 0.0;
        for (size_t i = 0; i < VDBL; ++i) {           /* volume 0 vs scalar ref */
            double d = tout[i] - refv[i];
            num += d * d; den += refv[i] * refv[i];
        }
        ok[c] = (num <= den * 1e-26);                 /* rel L2 < 1e-13 */
        tc[c] = 1e300;
    }
    const int R = (nv >= 8) ? 1 : (nv >= 2 ? 3 : 8);
    for (int round = 0; round < 5; ++round)
        for (int c = 0; c < nc; ++c) {
            if (!ok[c]) continue;
            /* self-warm so each candidate is timed from its own steady state */
            run_vols(cands[c].pw, cands[c].mode, cands[c].pf, p->S, p->P, tin, tout, nv);
            double t0 = now_s();
            for (int r = 0; r < R; ++r)
                run_vols(cands[c].pw, cands[c].mode, cands[c].pf, p->S, p->P, tin, tout, nv);
            double t = (now_s() - t0) / R;
            if (t < tc[c]) tc[c] = t;
        }
    int best = -1;
    for (int c = 0; c < nc; ++c) if (ok[c] && (best < 0 || tc[c] < tc[best])) best = c;
    if (best >= 0) {
        int pick = best;
        for (int c = 0; c < nc; ++c)
            if (ok[c] && tc[c] <= tc[best] * 1.03 &&
                cand_rank(cands[c].mode, cands[c].pf) <
                cand_rank(cands[pick].mode, cands[pick].pf)) pick = c;
        p->pw = cands[pick].pw; p->mode = cands[pick].mode; p->pf = cands[pick].pf;
    }
    snprintf(g_desc, sizeof g_desc,
             "GT-PFA 9x5 two-sweep; tuner pick: pw=%d mode=%s pf=%d (B=%d, nv=%d)",
             p->pw, mode_name[p->mode], p->pf, batch, nv);

#ifdef FFT45_LOUD
    if (1) {
#else
    if (getenv("FFT45_VERBOSE")) {
#endif
        for (int c = 0; c < nc; ++c)
            fprintf(stderr, "L45_pfa tuner: pw=%d mode=%-8s pf=%d  %s  %.1f us/vol\n",
                    cands[c].pw, mode_name[cands[c].mode], cands[c].pf,
                    ok[c] ? "ok " : "BAD", ok[c] ? tc[c] * 1e6 / nv : 0.0);
        fprintf(stderr, "L45_pfa tuner: chose pw=%d mode=%s pf=%d (nv=%d)\n",
                p->pw, mode_name[p->mode], p->pf, nv);
    }
    free(ri); free(ro); free(rr);
    return p;
}

void fft3d_destroy(fft3d_plan *p)
{
    if (!p) return;
    free(p->rawS); free(p->rawP); free(p);
}

void fft3d_execute(fft3d_plan *plan, const double _Complex *in, double _Complex *out)
{
    run_vols(plan->pw, plan->mode, plan->pf, plan->S, plan->P,
             (const double *)in, (double *)out, plan->batch);
}

#else /* ================ KERNEL TEMPLATE, PW = 1, 2 or 4 =================== */

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

/* every in/out/mid access is potentially 16-byte aligned only (720 B rows) */
#define LDU(p)    ((vec)*(const uvec *)(p))
#define STU(p, v) (*(uvec *)(p) = (uvec)(v))

#define NGRP  ((L + PW - 1) / PW)     /* groups incl. overlap tail: 45/23/12 */
#define NFULL (L / PW)                /* full groups: 45/22/11               */

#if PW == 4
# define VSPLAT(a)  ((vec){(a),(a),(a),(a),(a),(a),(a),(a)})
# define VPAIR(a,b) ((vec){(a),(b),(a),(b),(a),(b),(a),(b)})
# define SWAP(v)    VSH((v),(v), 1,0,3,2,5,4,7,6)
# define VFMA(a,b,c)  ((vec)_mm512_fmadd_pd((__m512d)(a),(__m512d)(b),(__m512d)(c)))
# define VFNMA(a,b,c) ((vec)_mm512_fnmadd_pd((__m512d)(a),(__m512d)(b),(__m512d)(c)))
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
#else /* PW == 1: one complex per 128-bit vector, for the in-place x tail */
# define VSPLAT(a)  ((vec){(a),(a)})
# define VPAIR(a,b) ((vec){(a),(b)})
# define SWAP(v)    VSH((v),(v), 1,0)
# ifdef __FMA__
#  define VFMA(a,b,c)  ((vec)_mm_fmadd_pd((__m128d)(a),(__m128d)(b),(__m128d)(c)))
#  define VFNMA(a,b,c) ((vec)_mm_fnmadd_pd((__m128d)(a),(__m128d)(b),(__m128d)(c)))
# else
#  define VFMA(a,b,c)  ((a)*(b) + (c))
#  define VFNMA(a,b,c) ((c) - (a)*(b))
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

/* y1 = a - i*s*m, y2 = a + i*s*m via one swap and two FMAs */
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

/* The 45-point Good-Thomas 9x5 codelet over PW interleaved-complex lanes.
 * LD(n) must yield input element n as a vec rvalue; ST(k, v) consumes output
 * element k.  Both index maps fold to compile-time constants once the loops
 * unroll.  All LD reads happen before the first non-y0 ST... in fact ALL LDs
 * (stage 1) complete before any ST (stage 2), so LD/ST may alias freely.
 * Stage 1: 5 x DFT9 (CT 3x3, 44 FMA-port ops); stage 2: 9 x DFT5 (16 ops). */
#define PFA45(LD, ST) do {                                                   \
    vec A_[45];                            /* A_[5*k1 + n2] */               \
    _Pragma("GCC unroll 5")                                                  \
    for (int n2_ = 0; n2_ < 5; ++n2_) {                                      \
        vec g_[9];                                                           \
        _Pragma("GCC unroll 9")                                              \
        for (int n1_ = 0; n1_ < 9; ++n1_)                                    \
            g_[n1_] = LD((5 * n1_ + 9 * n2_) % 45);                          \
        vec B_[9];                         /* B_[3*r + b] = C_b[r] */        \
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
            A_[5*(r_    ) + n2_] = y0_;                                      \
            A_[5*(r_ + 3) + n2_] = y1_;                                      \
            A_[5*(r_ + 6) + n2_] = y2_;                                      \
        }                                                                    \
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

#if PW == 1
/* In-place x-pass tail: the single kz = 44 column for one y, 45-point line
 * along x at stride PLND doubles.  mid may equal out (all loads precede all
 * stores inside PFA45). */
static void xcol_pw1(const double *mid, double *out, int y)
{
    const size_t o = ((size_t)y * L + (L - 1)) * 2;
    const double *s_ = mid + o;
    double       *d_ = out + o;
    _Pragma("GCC unroll 45")
    for (int n_ = 0; n_ < 45; ++n_)
        __builtin_prefetch(s_ + (size_t)n_ * PLND, 0, 3);
#define LDX(n)    LDU(s_ + (size_t)(n) * PLND)
#define STX(k, v) STU(d_ + (size_t)(k) * PLND, (v))
    PFA45(LDX, STX);
#undef LDX
#undef STX
}
#else  /* PW >= 2: the actual pass machinery */

/* paced-prefetch step: cover one x-plane (PLND doubles = 507 lines) in
 * 2*NGRP steps -> 22 lines/step at PW=4, 12 at PW=2 */
#define PFL    ((PLND + 16 * NGRP - 1) / (16 * NGRP))
#define PFSTEP (8 * PFL)
#define PFIN(p) do {                                                          \
    _Pragma("GCC unroll 24")                                                  \
    for (int q_ = 0; q_ < PFL; ++q_)                                          \
        __builtin_prefetch((p) + 8 * q_, 0, FFT45_PFH);                       \
} while (0)
/* pf=2 INPLACE: write-intent (prefetchw) cursor over the mid-plane stores */
#define PFWMID(p) do {                                                        \
    _Pragma("GCC unroll 24")                                                  \
    for (int q_ = 0; q_ < PFL; ++q_)                                          \
        __builtin_prefetch((p) + 8 * q_, 1, 3);                               \
} while (0)

/* phase 1, ONE x-plane: z transform (transposed lanes) into plane scratch,
 * then y transform into mid[x][ky][kz].  The prefetch cursors are positional
 * (recomputed per plane), so they cannot drift across a batch. */
static inline __attribute__((always_inline))
void FN(phase1_plane)(const double *restrict in, double *restrict mid,
                      double *restrict pld, int x, int pfr, int pfw)
{
    const double *pfc = in  + FFT45_PFD  + (size_t)x * PLND;
    double       *pwc = mid + FFT45_PFWD + (size_t)x * PLND;
    const double *px  = in  + (size_t)x * PLND;
    double       *mx  = mid + (size_t)x * PLND;

#ifndef FFT45_SKIPZ
    for (int yg = 0; yg < NGRP; ++yg) {
        const int yb = (yg == NGRP - 1) ? (L - PW) : yg * PW;
        if (pfr) { PFIN(pfc);   pfc += PFSTEP; }
        if (pfw) { PFWMID(pwc); pwc += PFSTEP; }
        vec Zv[L], Wv[L];
        _Pragma("GCC unroll 12")
        for (int zg = 0; zg < NGRP; ++zg) {
            const int zb = (zg == NGRP - 1) ? (L - PW) : zg * PW;
            vec r_[PW];
            _Pragma("GCC unroll 4")
            for (int j = 0; j < PW; ++j)
                r_[j] = LDU(px + ((size_t)(yb + j) * L + zb) * 2);
            TRNC(r_, &Zv[zb]);
        }
#define LD1(n)    Zv[n]
#define ST1(k, v) (Wv[k] = (v))
#ifndef FFT45_ZCOPY            /* diagnostic: data movement only, wrong answers */
        PFA45(LD1, ST1);
#else
        for (int k = 0; k < L; ++k) Wv[k] = Zv[k];
#endif
#undef LD1
#undef ST1
        _Pragma("GCC unroll 12")
        for (int zg = 0; zg < NGRP; ++zg) {
            const int zb = (zg == NGRP - 1) ? (L - PW) : zg * PW;
            vec r_[PW];
            TRNC(&Wv[zb], r_);
            _Pragma("GCC unroll 4")
            for (int j = 0; j < PW; ++j)
                STU(pld + ((size_t)(yb + j) * PPITCH + zb) * 2, r_[j]);
        }
    }
#endif

#ifndef FFT45_SKIPY
    for (int zg = 0; zg < NGRP; ++zg) {
        const int zb = (zg == NGRP - 1) ? (L - PW) : zg * PW;
        if (pfr) { PFIN(pfc);   pfc += PFSTEP; }
        if (pfw) { PFWMID(pwc); pwc += PFSTEP; }
#define LD2(n)    LDU(pld + ((size_t)(n) * PPITCH + zb) * 2)
#define ST2(k, v) STU(mx + ((size_t)(k) * L + zb) * 2, (v))
#ifndef FFT45_YCOPY            /* diagnostic: data movement only, wrong answers */
        PFA45(LD2, ST2);
#else
        for (int k = 0; k < L; ++k) ST2(k, LD2(k));
#endif
#undef LD2
#undef ST2
    }
#endif
}

static void FN(phase1)(const double *restrict in, double *restrict mid,
                       double *restrict pld, int pfr, int pfw)
{
    for (int x = 0; x < L; ++x)
        FN(phase1_plane)(in, mid, pld, x, pfr, pfw);
}

/* phase-2 read prefetch: 45 sequential streams at PLND-double stride is more
 * than the L2 streamer tracks (dropping the analogous prefetch cost L36 14%) */
#define PF45(s_) do {                                                        \
    _Pragma("GCC unroll 45")                                                 \
    for (int n_ = 0; n_ < 45; ++n_)                                          \
        __builtin_prefetch((s_) + (size_t)n_ * PLND + 8, 0, 3);              \
} while (0)
/* pf=2 SCRATCH: write-intent prefetch of the 45 cold out-streams */
#define PFW45(d_) do {                                                       \
    _Pragma("GCC unroll 45")                                                 \
    for (int n_ = 0; n_ < 45; ++n_)                                          \
        __builtin_prefetch((d_) + (size_t)n_ * PLND + 8, 1, 3);              \
} while (0)
/* per-tile pre-coverage of the NEXT volume's input (first ~63 KB) */
#define PFNX() do { if (pn_) {                                                \
    _Pragma("GCC unroll 4")                                                   \
    for (int q_ = 0; q_ < FFT45_PFN; ++q_)                                    \
        __builtin_prefetch(pn_ + 8 * q_, 0, FFT45_PFH);                       \
    pn_ += 8 * FFT45_PFN; } } while (0)

/* phase 2, ONE y-plane: x transform, in place when mid == out (the codelet
 * reads all 45 inputs before its first store).  Full-vector tiles cover
 * kz 0..NFULL*PW-1; the kz = 44 tail column runs the PW=1 codelet (an
 * overlapped tile would re-read already-transformed columns). */
static inline __attribute__((always_inline))
void FN(phase2_yplane)(const double *mid, double *out,
                       const double *pnext, int y, int pfw)
{
    const double *pn_ = pnext ? pnext + (size_t)y * NFULL * 8 * FFT45_PFN : NULL;
    for (int zg = 0; zg < NFULL; ++zg) {
        const size_t o = ((size_t)y * L + (size_t)zg * PW) * 2;
        const double *s_ = mid + o;
        double       *d_ = out + o;
        PF45(s_);
        if (pfw) PFW45(d_);
        PFNX();
#define LD3(n)    LDU(s_ + (size_t)(n) * PLND)
#define ST3(k, v) STU(d_ + (size_t)(k) * PLND, (v))
        PFA45(LD3, ST3);
#undef LD3
#undef ST3
    }
    xcol_pw1(mid, out, y);
}

static void FN(phase2)(const double *mid, double *out,
                       const double *pnext, int pfw)
{
    for (int y = 0; y < L; ++y)
        FN(phase2_yplane)(mid, out, pnext, y, pfw);
}

#undef PF45
#undef PFW45
#undef PFNX
#undef PFIN
#undef PFWMID
#undef PFSTEP
#undef PFL
#endif /* PW >= 2 */

#undef PFA45
#undef CMULW
#undef DFT3M
#ifdef TRNC
# undef TRNC
#endif
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
