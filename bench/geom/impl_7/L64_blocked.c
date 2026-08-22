/* L64_blocked.c -- forward complex 3D DFT of a 64^3 cube, batched, out-of-place.
 *
 * ROUND panel_r6: first implementation for this geometry.
 * ROUND panel_r7: two changes, both adopted from L64_radix8's r6 record:
 *   1. mid lives in a 2 MB-hugepage mmap (MADV_HUGEPAGE + touch in create):
 *      the strided column walks otherwise touch ~1090 4-KB pages per sweep.
 *   2. A 2-SWEEP structure variant (st=1) joins the tuner: pass A does z+y
 *      only (planes in natural order, so the cold in-read is one sequential
 *      run), and a new pass B2 does the FULL two-stage 64-point x-FFT per
 *      (ky, z-column) directly out of mid -- 64 read streams at the padded
 *      (odd-line) plane stride, per-load prefetcht0 FFT64B_PFXC columns
 *      ahead (L64_radix8 measured that hint +12% at B=8), stores straight
 *      to out (full-line NT at PW=4, or cached).  This removes the x1 RMW
 *      pass entirely; the 3-sweep st=0 path is kept as candidates and the
 *      create-time tournament decides per {B, machine}.
 *
 * TECHNIQUE (see ../strategies/L64_blocked.md for the full derivation)
 *   Row-column 3D DFT on INTERLEAVED complex vectors whose lanes are PW
 *   consecutive z (the contiguous axis), so the y- and x-passes are
 *   shuffle-free and only the z-transform pays the one unavoidable
 *   transpose pair (proof in L36_pencilfused r1, adopted).
 *
 *   Every 64-point line is TWO RADIX-8 STAGES (64 = 8*8, DIT):
 *       X[8c+d] = sum_s W8^{sc} * ( W64^{sd} * sum_a W8^{ad} x[8a+s] )
 *   The 8-point module is the 4-mul/52-add radix-8 codelet (26 FMA-port ops
 *   + 5 swaps per PW lanes); the only irrational constant is 1/sqrt(2).
 *
 *   The volume is 4.19 MB -- it does NOT fit the scoring node's 1 MB L2, and
 *   at L=64 every natural stride is a power of two (the z-row is exactly one
 *   L1 way, the x-stride exactly one L2 way), so this file's charter is the
 *   cache-blocking + padding question.  Structure:
 *
 *   pass A, per x-plane (64x64 complex = 64 KB), planes visited in stride-8
 *   groups {r, r+8, ..., r+56}:
 *       z transform: lanes = PW y-rows via PWxPW complex-granule register
 *                    transposes on load AND store (both against the cheap
 *                    side), into a PADDED plane scratch P[y][kz]
 *       y transform: lanes = PW kz (contiguous in P), store to the PADDED
 *                    scratch volume mid[p][ky][kz]
 *       x stage 1:   after the group's 8 planes land in mid (~545 KB, still
 *                    L2/L3-warm), DFT-8 across the group IN PLACE on mid
 *                    with twiddles W64^{r*d}; 8 sequential read + 8
 *                    sequential write streams, never 64
 *   pass B:
 *       x stage 2:   DFT-8 over 8 CONSECUTIVE mid planes (one sequential
 *                    ~545 KB read run per octet), writing out[8c+d] through
 *                    8 sequential plane-streams (cached / +prefetchw / NT,
 *                    autotuned)
 *
 *   Splitting the x-transform's two radix-8 stages across the two passes is
 *   what kills the classic pathology: a monolithic x-pass needs 64 concurrent
 *   read streams of stride 64 KB (all one L2 set, unprefetchable); here no
 *   loop in the file ever runs more than 8 streams, all sequential.
 *
 *   PADDING (the stub's charter): mid's z-row stride is 68 complex = 17
 *   cache lines (odd) and its plane stride 4356 complex = 1089 lines (odd),
 *   so gcd(stride, sets) = 1 at both L1 and L2 and Bailey's single-set
 *   worst case cannot form on any scratch access.  in/out keep the driver's
 *   power-of-two layout but are only ever touched as <=8 sequential streams.
 *
 * ATTRIBUTION (what this file borrows, per the panel rules)
 *   - Interleaved-complex spectator-axis lanes, CMUL/DFT-with-swap idioms,
 *     PWxPW TRNC transpose, the #include-__FILE__ two-width template, and the
 *     "z-first transpose-on-load against the cold buffer" pass shape:
 *     L36_mixedradix r1 via L36_pfa / L36_pencilfused (r2-r5 records).
 *   - Paced T1 read-prefetch cursor, write-intent prefetchw on cold out
 *     streams, NTA read at consumption rate to protect L2 residency, and
 *     next-volume pre-coverage: L36_pfa r3-r6 (`pf` levels; ultimately
 *     L6_unrolled r3's prefetchw), constants rescaled to 64 KB planes.
 *   - Self-warming interleaved-rounds tuner with correctness interlock,
 *     physics gates and simplest-wins hysteresis: L36_pencilfused r5 +
 *     L36_pfa r4.
 *   - NT stores as a gated candidate, never a default (node rejected NT at
 *     L=36 three rounds running; wallaby loves it -- the tournament decides):
 *     L36_pencilfused r1/r4 evidence.
 *
 * OPERATION COUNT (per 64-point line over PW lanes, FMA-port vector ops)
 *   16 x DFT8 (26 ops + 5 swaps) + 49 twiddle CMUL (2 ops + 1 swap)
 *     = 514 FMA-port ops + 129 swaps
 *   Per volume: 3 * 1024 line-groups at PW=4 -> ~1.6M FMA-port ops; on the
 *   node's single 512-bit FMA pipe at 2.9 GHz that is ~550 us of port work.
 *   The binding constraint is L3: ~18 MB of scratch round trips per volume
 *   at 18.2 GB/s single-core -- this geometry is memory-shaped everywhere,
 *   which is why the schedule (prefetch pacing, stream counts) is the design.
 */
#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <sys/mman.h>
#ifdef __x86_64__
# include <immintrin.h>
#endif

#include "fft3d_api.h"

#ifndef L64B_ONCE               /* ============ COMMON, first pass ============ */
#define L64B_ONCE

#define L    64
#define LSQ  4096                    /* 64*64                                  */
#define VDBL ((size_t)2 * L * LSQ)   /* doubles per volume = 524288 (4.19 MB)  */

/* padded scratch strides, in complex units: both an ODD number of 64-B lines.
 * -DFFT64B_NOPAD builds the power-of-two-stride control (the Bailey worst
 * case this file's charter is to measure). */
#ifdef FFT64B_NOPAD
# define RS  64
# define PS  4096
#else
# define RS  68                      /* z-row stride: 68*16 = 1088 B = 17 lines */
# define PS  4356                    /* plane stride: 64*RS+4 = 69696 B = 1089  */
#endif
#define MIDDBL ((size_t)2 * L * PS)  /* doubles in the scratch volume          */

#define CAT_(a,b) a##b
#define CAT(a,b)  CAT_(a,b)

/* pf=1/2: paced T1 read cursor -- one plane's worth of prefetches per plane
 * processed, spread over both pass-A subloops, aimed at the NEXT plane in
 * VISIT order (the group walks planes at stride 8, so a linear +32KB lead
 * would waste half its coverage on a plane we do not read next). */
#ifndef FFT64B_PFH
# define FFT64B_PFH 2                /* 3=T0 2=T1 1=T2 0=NTA */
#endif
/* pf=3/4: NTA read lead in doubles (4 KB), consumption-rate paced in the A1
 * subloop only; fills L1, bypasses L2 on SKX-class cores, so the in-read
 * stops evicting the group's mid planes before x-stage-1 needs them. */
#ifndef FFT64B_PFDN
# define FFT64B_PFDN 512
#endif
/* cache lines of the NEXT volume's input pre-covered per pass-B ky step */
#ifndef FFT64B_PFN
# define FFT64B_PFN 2
#endif
/* pf=2/3: write-intent prefetch lead on pass B's 8 out streams, in lines */
#ifndef FFT64B_PFWL
# define FFT64B_PFWL 4
#endif
/* st=1 pass B2: read-prefetch lead over mid's strided columns, in COLUMNS
 * (one column = one vector = one line at PW=4).  One prefetch per load,
 * adopted from L64_radix8 r6's next-column prefetcht0 (+12% at B=8 there);
 * a 2-column lead is ~2 FFT64V bodies ~ 400+ cycles, well past L3 latency. */
#ifndef FFT64B_PFXC
# define FFT64B_PFXC 2
#endif

/* W64^m = twre8[m] + i*(-twia8[m][even]); rows are pre-splatted vector forms:
 * twre8[m][j] = Re(W64^m) in every lane; twia8[m][j] = (-Im, +Im) alternating,
 * so cmul is 1 swap + 1 mul + 1 fma with two 64-B table loads.  Filled once
 * in fft3d_create() (libm in setup is allowed; execute only loads). */
static double twre8[64][8] __attribute__((aligned(64)));
static double twia8[64][8] __attribute__((aligned(64)));

static void fill_twiddles(void)
{
    for (int m = 0; m < 64; ++m) {
        double a  = -2.0 * M_PI * (double)m / 64.0;   /* forward: W = e^{-2pi i m/64} */
        double cr = cos(a), ci = sin(a);
        for (int j = 0; j < 8; ++j) {
            twre8[m][j] = cr;
            twia8[m][j] = (j & 1) ? ci : -ci;
        }
    }
}

enum { M_CACHED = 0, M_NT = 1 };

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

static const char *const mode_name[] = {"cached", "nt"};

struct fft3d_plan {
    int     batch;
    int     pw;                  /* 2 or 4                                  */
    int     mode;                /* M_CACHED / M_NT (final out stores)      */
    int     pf;                  /* 0 none; 1 paced T1 read (+PFNX);        */
                                 /* 2 = 1 + prefetchw out; 3 NTA read +     */
                                 /* prefetchw; 4 NTA read alone             */
    int     st;                  /* 0 = 3-sweep (A+x1, B octets);           */
                                 /* 1 = 2-sweep (A z+y only, B2 full-x)     */
    double *S;                   /* padded scratch volume, reused per batch */
    void   *rawS;
    size_t  map_bytes;           /* nonzero: rawS is an mmap of this size   */
};

const char *fft3d_name(void) { return "L64_blocked"; }

static char g_desc[224];

const char *fft3d_description(void)
{
    return g_desc[0] ? g_desc
                     : "64=8x8 two-stage radix-8; plane-fused z/y; x either "
                       "split over 8-plane groups + octet pass (st=0) or one "
                       "strided full-x pass (st=1); hugepage odd-line-padded "
                       "scratch; {pw,mode,pf,st} autotuned";
}
int fft3d_supports(int Lq) { return Lq == L; }

static void run_vols(int pw, int mode, int pf, int st, double *S,
                     const double *in, double *out, int nvol)
{
#ifdef HAVE_PW4
    if (pw == 4) run_vols_pw4(in, out, S, nvol, mode, pf, st);
    else
#endif
    { (void)pw; run_vols_pw2(in, out, S, nvol, mode, pf, st); }
#if defined(__SSE2__)
    if (mode == M_NT) _mm_sfence();
#endif
}

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* hysteresis rank: lower = simpler (adopted from L36_pfa r4).  Read-side
 * prefetch levels before store-side machinery, incumbent 3-sweep before
 * 2-sweep, cached before NT. */
static int cand_rank(int mode, int pf, int st)
{
    static const int pfc[5] = {0, 1, 3, 4, 2};   /* 0 < 1 < 4 < 2 < 3 */
    return pfc[pf] * 4 + st * 2 + mode;
}

fft3d_plan *fft3d_create(int Lq, int batch)
{
    if (Lq != L || batch < 1) return NULL;
    fill_twiddles();
    fft3d_plan *p = calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->batch = batch;
    /* mid on 2 MB hugepages (adopted from L64_radix8 r6): the strided sweeps
     * otherwise walk ~1090 4-KB pages; madvise BEFORE the faulting memset so
     * THP-madvise kernels back it synchronously.  Over-map by one hugepage so
     * the working base can be 2 MB-aligned; fall back to posix_memalign. */
    {
        const size_t hp = (size_t)2 << 20;
        size_t bytes = MIDDBL * sizeof(double);
        size_t mb = ((bytes + hp - 1) & ~(hp - 1)) + hp;
        void *m = mmap(NULL, mb, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (m != MAP_FAILED) {
            uintptr_t a = ((uintptr_t)m + hp - 1) & ~(uintptr_t)(hp - 1);
            p->rawS = m; p->map_bytes = mb; p->S = (double *)a;
#if defined(MADV_HUGEPAGE) && !defined(FFT64B_NOHP)
            madvise(m, mb, MADV_HUGEPAGE);   /* -DFFT64B_NOHP = control */
#endif
        } else if (posix_memalign(&p->rawS, 64, bytes) == 0) {
            p->map_bytes = 0; p->S = (double *)p->rawS;
        } else { free(p); return NULL; }
    }
    memset(p->S, 0, MIDDBL * sizeof(double));
    p->pw = 2; p->mode = M_CACHED; p->pf = 1; p->st = 0;   /* safe default */

    /* streaming gate: NT / prefetchw only make sense once the batch's in+out
     * footprint decisively leaves the node's 22 MB L3 (8.4 MB per volume) */
    const int streaming = ((size_t)batch * 2 * VDBL * sizeof(double) > (size_t)24 << 20);

    struct cand { int pw, mode, pf, st; } cands[32];
    int nc = 0;
    /* cands[0] doubles as the numerical reference: pw2/cached/pf0/st0 */
    static const int pfs_c[] = {0, 1, 4, 2, 3};  /* cached-store pf levels */
    for (int i = 0; i < 5; ++i) {
        int pf = pfs_c[i];
        if ((pf == 2 || pf == 3) && !streaming) continue;
        cands[nc++] = (struct cand){2, M_CACHED, pf, 0};
#ifdef HAVE_PW4
        cands[nc++] = (struct cand){4, M_CACHED, pf, 0};
#endif
    }
    if (streaming)
        for (int pf = 0; pf <= 1; ++pf) {
            cands[nc++] = (struct cand){2, M_NT, pf, 0};
#ifdef HAVE_PW4
            cands[nc++] = (struct cand){4, M_NT, pf, 0};
#endif
        }
    /* st=1 (2-sweep, full-x pass B2): pw4 only -- its NT stores are exactly
     * one full line, and pw2 loses 13-15% everywhere anyway.  NT is NOT gated
     * on streaming here: B2's cached stores are 64 RFO streams at 64-KB
     * stride (one L1 set, and one L2 set on the node), so bypassing the
     * caches can pay even at B=1 -- the tournament decides. */
#ifdef HAVE_PW4
    for (int pf = 0; pf <= 1; ++pf) {
        cands[nc++] = (struct cand){4, M_CACHED, pf, 1};
        cands[nc++] = (struct cand){4, M_NT, pf, 1};
    }
#else
    cands[nc++] = (struct cand){2, M_CACHED, 0, 1};
    cands[nc++] = (struct cand){2, M_CACHED, 1, 1};
#endif

    /* run-time forcing for the monitor's control jobs */
    { const char *e;
      if ((e = getenv("FFT64B_PW"))) {
          int v = atoi(e), w = 0;
          for (int c = 0; c < nc; ++c) if (cands[c].pw == v) cands[w++] = cands[c];
          if (w) nc = w;
      }
      if ((e = getenv("FFT64B_MODE"))) {
          int num = (e[0] >= '0' && e[0] <= '9'), w = 0;
          for (int c = 0; c < nc; ++c)
              if (num ? cands[c].mode == atoi(e)
                      : !strcmp(mode_name[cands[c].mode], e)) cands[w++] = cands[c];
          if (w) nc = w;
      }
      if ((e = getenv("FFT64B_PF"))) {
          int v = atoi(e), w = 0;
          for (int c = 0; c < nc; ++c) if (cands[c].pf == v) cands[w++] = cands[c];
          if (w) nc = w;
      }
      if ((e = getenv("FFT64B_ST"))) {
          int v = atoi(e), w = 0;
          for (int c = 0; c < nc; ++c) if (cands[c].st == v) cands[w++] = cands[c];
          if (w) nc = w;
      } }
#ifdef FFT64B_FORCE_PW
    { int w = 0;
      for (int c = 0; c < nc; ++c) if (cands[c].pw == FFT64B_FORCE_PW) cands[w++] = cands[c];
      if (w) nc = w; }
#endif
#ifdef FFT64B_FORCE_MODE
    { int w = 0;
      for (int c = 0; c < nc; ++c) if (cands[c].mode == FFT64B_FORCE_MODE) cands[w++] = cands[c];
      if (w) nc = w; }
#endif
#ifdef FFT64B_FORCE_PF
    { int w = 0;
      for (int c = 0; c < nc; ++c) if (cands[c].pf == FFT64B_FORCE_PF) cands[w++] = cands[c];
      if (w) nc = w; }
#endif
#ifdef FFT64B_FORCE_ST
    { int w = 0;
      for (int c = 0; c < nc; ++c) if (cands[c].st == FFT64B_FORCE_ST) cands[w++] = cands[c];
      if (w) nc = w; }
#endif

    /* tuning arena: 8 volumes = 67 MB in+out streams past every L3 involved */
    const int nv = batch < 8 ? batch : 8;
    void *ri = NULL, *ro = NULL, *rr = NULL;
    if (posix_memalign(&ri, 64, (size_t)nv * VDBL * sizeof(double)) ||
        posix_memalign(&ro, 64, (size_t)nv * VDBL * sizeof(double)) ||
        posix_memalign(&rr, 64, (size_t)nv * VDBL * sizeof(double))) {
        free(ri); free(ro);
        p->pw = cands[0].pw; p->mode = cands[0].mode;
        p->pf = cands[0].pf; p->st = cands[0].st;
        snprintf(g_desc, sizeof g_desc,
                 "L64 8x8 blocked; tuner SKIPPED (arena alloc failed): "
                 "pw=%d mode=%s pf=%d st=%d",
                 p->pw, mode_name[p->mode], p->pf, p->st);
        return p;
    }
    double *tin = ri, *tout = ro, *ref = rr;
    unsigned long long s = 0x9E3779B97F4A7C15ull;
    for (size_t i = 0; i < (size_t)nv * VDBL; ++i) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        tin[i] = (double)(long long)(s >> 11) * 0x1p-53;
    }

    run_vols(cands[0].pw, cands[0].mode, cands[0].pf, cands[0].st, p->S, tin, ref, nv);

    int    ok[32];
    double tc[32];
    for (int c = 0; c < nc; ++c) {
        run_vols(cands[c].pw, cands[c].mode, cands[c].pf, cands[c].st, p->S, tin, tout, nv);
        double num = 0.0, den = 0.0;
        for (size_t i = 0; i < (size_t)nv * VDBL; ++i) {
            double d = tout[i] - ref[i];
            num += d * d; den += ref[i] * ref[i];
        }
        ok[c] = (num <= den * 1e-26);       /* rel L2 < 1e-13 vs reference */
        tc[c] = 1e300;
    }
    const int R = (nv >= 4) ? 1 : (nv >= 2 ? 2 : 4);
    for (int round = 0; round < 4; ++round)
        for (int c = 0; c < nc; ++c) {
            if (!ok[c]) continue;
            /* self-warming: one untimed exec so each candidate is timed from
             * its OWN steady-state cache (L36_pencilfused r5's 86% phantom-
             * penalty fix: an NT candidate flushes tout and poisons whoever
             * runs next in the rotation). */
            run_vols(cands[c].pw, cands[c].mode, cands[c].pf, cands[c].st, p->S, tin, tout, nv);
            double t0 = now_s();
            for (int r = 0; r < R; ++r)
                run_vols(cands[c].pw, cands[c].mode, cands[c].pf, cands[c].st, p->S, tin, tout, nv);
            double t = (now_s() - t0) / R;
            if (t < tc[c]) tc[c] = t;
        }
    int best = 0;
    for (int c = 1; c < nc; ++c) if (ok[c] && tc[c] < tc[best]) best = c;
    if (ok[best]) {
        int pick = best;
        for (int c = 0; c < nc; ++c)     /* 3% simplest-wins hysteresis */
            if (ok[c] && tc[c] <= tc[best] * 1.03 &&
                cand_rank(cands[c].mode, cands[c].pf, cands[c].st) <
                cand_rank(cands[pick].mode, cands[pick].pf, cands[pick].st)) pick = c;
        p->pw = cands[pick].pw; p->mode = cands[pick].mode;
        p->pf = cands[pick].pf; p->st = cands[pick].st;
    }
    snprintf(g_desc, sizeof g_desc,
             "L64 8x8 two-stage, hugepage odd-line-padded scratch; "
             "tuner pick: pw=%d mode=%s pf=%d st=%d(%s) (B=%d, nv=%d)",
             p->pw, mode_name[p->mode], p->pf, p->st,
             p->st ? "2-sweep" : "3-sweep", batch, nv);

    if (getenv("FFT64B_VERBOSE")) {
        for (int c = 0; c < nc; ++c)
            fprintf(stderr, "L64_blocked tuner: pw=%d mode=%-7s pf=%d st=%d  %s  %.1f us/vol\n",
                    cands[c].pw, mode_name[cands[c].mode], cands[c].pf, cands[c].st,
                    ok[c] ? "ok " : "BAD", ok[c] ? tc[c] * 1e6 / nv : 0.0);
        fprintf(stderr, "L64_blocked tuner: chose pw=%d mode=%s pf=%d st=%d (nv=%d)\n",
                p->pw, mode_name[p->mode], p->pf, p->st, nv);
    }
    free(ri); free(ro); free(rr);
    return p;
}

void fft3d_destroy(fft3d_plan *p)
{
    if (!p) return;
    if (p->map_bytes) munmap(p->rawS, p->map_bytes);
    else              free(p->rawS);
    free(p);
}

void fft3d_execute(fft3d_plan *plan, const double _Complex *in, double _Complex *out)
{
    run_vols(plan->pw, plan->mode, plan->pf, plan->st, plan->S,
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

#define NV (L / PW)                  /* z-vectors per 64-complex row: 32 or 16 */
#define RSV (RS / PW)                /* padded P-buffer row stride in vecs     */

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

/* PW x PW transpose of 128-bit complex granules (from L36_pfa, verbatim) */
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

#define PMV VPAIR(1.0, -1.0)
#define CSV VSPLAT(0.70710678118654752440084436210485)

/* pre-splatted twiddle table rows (first 2*PW doubles of the 8-double rows) */
#define TRE(m) (*(const vec *)&twre8[m][0])
#define TIA(m) (*(const vec *)&twia8[m][0])
/* v * W64^m: 1 swap + 1 mul + 1 fma, two table loads */
#define CMULT(v, m) VFMA((v), TRE(m), SWAP(v) * TIA(m))

/* Forward 8-point DFT, natural in/out order, inputs may be memory refs (all
 * reads happen in the first two statement rows).  26 FMA-port ops + 5 swaps. */
#define DFT8M(x0,x1,x2,x3,x4,x5,x6,x7, y0,y1,y2,y3,y4,y5,y6,y7) do {         \
    vec a0_=(x0)+(x4), a1_=(x0)-(x4), a2_=(x2)+(x6), a3_=(x2)-(x6);          \
    vec b0_=(x1)+(x5), b1_=(x1)-(x5), b2_=(x3)+(x7), b3_=(x3)-(x7);          \
    vec sE_ = SWAP(a3_), sO_ = SWAP(b3_);                                    \
    vec E0_=a0_+a2_, E2_=a0_-a2_;                                            \
    vec E1_=VFMA(sE_, PMV, a1_), E3_=VFNMA(sE_, PMV, a1_);                   \
    vec O0_=b0_+b2_, O2_=b0_-b2_;                                            \
    vec O1_=VFMA(sO_, PMV, b1_), O3_=VFNMA(sO_, PMV, b1_);                   \
    vec s2_ = SWAP(O2_);                                                     \
    vec q0_ = VFMA (SWAP(O1_), PMV, O1_);        /* (1-i)*O1 */              \
    vec q1_ = VFNMA(SWAP(O3_), PMV, O3_);        /* (1+i)*O3 */              \
    (y0) = E0_ + O0_;               (y4) = E0_ - O0_;                        \
    (y2) = VFMA (s2_, PMV, E2_);    (y6) = VFNMA(s2_, PMV, E2_);             \
    (y1) = VFMA (q0_, CSV, E1_);    (y5) = VFNMA(q0_, CSV, E1_);             \
    (y3) = VFNMA(q1_, CSV, E3_);    (y7) = VFMA (q1_, CSV, E3_);             \
} while (0)

/* The 64-point line as two radix-8 stages with W64^{s*d} between them.
 * LD(n) yields input element n; ST(k, v) consumes output element k, both in
 * NATURAL order.  All LD reads complete inside stage 1, so LD/ST may alias. */
#define FFT64V(LD, ST) do {                                                  \
    vec H_[64];                              /* H_[8d+s] = W64^{sd} G_s[d] */\
    _Pragma("GCC unroll 8")                                                  \
    for (int s_ = 0; s_ < 8; ++s_) {                                         \
        vec y0_,y1_,y2_,y3_,y4_,y5_,y6_,y7_;                                 \
        DFT8M(LD(s_),LD(8+s_),LD(16+s_),LD(24+s_),                           \
              LD(32+s_),LD(40+s_),LD(48+s_),LD(56+s_),                       \
              y0_,y1_,y2_,y3_,y4_,y5_,y6_,y7_);                              \
        if (s_) {                                                            \
            y1_ = CMULT(y1_, 1*s_); y2_ = CMULT(y2_, 2*s_);                  \
            y3_ = CMULT(y3_, 3*s_); y4_ = CMULT(y4_, 4*s_);                  \
            y5_ = CMULT(y5_, 5*s_); y6_ = CMULT(y6_, 6*s_);                  \
            y7_ = CMULT(y7_, 7*s_);                                          \
        }                                                                    \
        H_[     s_] = y0_; H_[ 8 + s_] = y1_; H_[16 + s_] = y2_;             \
        H_[24 + s_] = y3_; H_[32 + s_] = y4_; H_[40 + s_] = y5_;             \
        H_[48 + s_] = y6_; H_[56 + s_] = y7_;                                \
    }                                                                        \
    _Pragma("GCC unroll 8")                                                  \
    for (int d_ = 0; d_ < 8; ++d_) {                                         \
        vec z0_,z1_,z2_,z3_,z4_,z5_,z6_,z7_;                                 \
        DFT8M(H_[8*d_  ],H_[8*d_+1],H_[8*d_+2],H_[8*d_+3],                   \
              H_[8*d_+4],H_[8*d_+5],H_[8*d_+6],H_[8*d_+7],                   \
              z0_,z1_,z2_,z3_,z4_,z5_,z6_,z7_);                              \
        ST(     d_, z0_); ST( 8 + d_, z1_); ST(16 + d_, z2_);                \
        ST(24 + d_, z3_); ST(32 + d_, z4_); ST(40 + d_, z5_);                \
        ST(48 + d_, z6_); ST(56 + d_, z7_);                                  \
    }                                                                        \
} while (0)

/* pass-A pacing: one plane (2*LSQ doubles) of prefetches per plane processed,
 * spread over the 2*NV subloop iterations, aimed at pfnext (the next plane in
 * VISIT order, which is 8 planes away in memory). */
#define PFSTEP (LSQ * PW / L)        /* = LSQ*PW/64 doubles per iteration */
#define PFA1(p) do {                                                         \
    _Pragma("GCC unroll 32")                                                 \
    for (int q_ = 0; q_ < PFSTEP / 8; ++q_)                                  \
        __builtin_prefetch((p) + 8 * q_, 0, FFT64B_PFH);                     \
} while (0)
/* NTA at consumption rate: the A1 subloop reads 2*PFSTEP doubles per
 * iteration; constant lead FFT64B_PFDN, nothing issued in A2. */
#define PFA1_NTA(p) do {                                                     \
    _Pragma("GCC unroll 64")                                                 \
    for (int q_ = 0; q_ < PFSTEP / 4; ++q_)                                  \
        __builtin_prefetch((p) + 8 * q_, 0, 0);                              \
} while (0)

/* pass A for ONE x-plane p: z transform (transpose pair, both sides against
 * the L1 data) then y transform, in[p] -> mid[p].  Sequential cold reads;
 * pfr: 0 none, 1 paced T1 at pfnext, 2 NTA at consumption rate. */
static void FN(passA_plane)(const double *restrict in, double *restrict mid,
                            int p, const double *pfnext, int pfr)
{
    vec P_[L * RSV];                 /* padded plane scratch P[y][kz], ~70 KB */
    const double *px  = in  + (size_t)p * (2 * LSQ);
    double       *mx  = mid + (size_t)p * (2 * PS);
    const double *pfc = pfnext;
    const double *pfn = px + FFT64B_PFDN;

    for (int yb = 0; yb < L; yb += PW) {
        if (pfr == 1 && pfc) { PFA1(pfc); pfc += PFSTEP; }
        else if (pfr == 2)   { PFA1_NTA(pfn); pfn += 2 * PFSTEP; }
        vec Zv[64];
        _Pragma("GCC unroll 32")
        for (int zb = 0; zb < NV; ++zb) {
            vec r_[PW];
            _Pragma("GCC unroll 4")
            for (int j = 0; j < PW; ++j)
                r_[j] = *(const vec *)(px + ((size_t)(yb + j) * L + (size_t)zb * PW) * 2);
            TRNC(r_, &Zv[zb * PW]);
        }
#define LDZ(n)    Zv[n]
#define STZ(k, v) (Zv[k] = (v))
        FFT64V(LDZ, STZ);
#undef LDZ
#undef STZ
        _Pragma("GCC unroll 32")
        for (int kb = 0; kb < NV; ++kb) {
            vec r_[PW];
            TRNC(&Zv[kb * PW], r_);
            _Pragma("GCC unroll 4")
            for (int j = 0; j < PW; ++j)
                P_[(size_t)(yb + j) * RSV + kb] = r_[j];
        }
    }

    for (int zb = 0; zb < NV; ++zb) {
        if (pfr == 1 && pfc) { PFA1(pfc); pfc += PFSTEP; }
#define LDY(n)    P_[(size_t)(n) * RSV + zb]
#define STY(k, v) (*(vec *)(mx + ((size_t)(k) * RS + (size_t)zb * PW) * 2) = (v))
        FFT64V(LDY, STY);
#undef LDY
#undef STY
    }
}

/* x stage 1, group r: DFT-8 across planes {r, r+8, ..., r+56} of mid with
 * twiddles W64^{r*d}, IN PLACE (all 8 loads precede the first store).
 * 8 sequential read streams + 8 sequential write streams at padded stride. */
#define X1BODY(TWIDDLE) do {                                                 \
    for (int ky = 0; ky < L; ++ky)                                           \
        for (int zb = 0; zb < NV; ++zb) {                                    \
            double *b_ = mid + (size_t)r * (2 * PS)                          \
                             + ((size_t)ky * RS + (size_t)zb * PW) * 2;      \
            vec v0_ = *(const vec *)(b_             );                       \
            vec v1_ = *(const vec *)(b_ +  8 * 2 * PS);                      \
            vec v2_ = *(const vec *)(b_ + 16 * 2 * PS);                      \
            vec v3_ = *(const vec *)(b_ + 24 * 2 * PS);                      \
            vec v4_ = *(const vec *)(b_ + 32 * 2 * PS);                      \
            vec v5_ = *(const vec *)(b_ + 40 * 2 * PS);                      \
            vec v6_ = *(const vec *)(b_ + 48 * 2 * PS);                      \
            vec v7_ = *(const vec *)(b_ + 56 * 2 * PS);                      \
            vec g0_,g1_,g2_,g3_,g4_,g5_,g6_,g7_;                             \
            DFT8M(v0_,v1_,v2_,v3_,v4_,v5_,v6_,v7_,                           \
                  g0_,g1_,g2_,g3_,g4_,g5_,g6_,g7_);                          \
            TWIDDLE;                                                         \
            *(vec *)(b_             ) = g0_;                                 \
            *(vec *)(b_ +  8 * 2 * PS) = g1_;                                \
            *(vec *)(b_ + 16 * 2 * PS) = g2_;                                \
            *(vec *)(b_ + 24 * 2 * PS) = g3_;                                \
            *(vec *)(b_ + 32 * 2 * PS) = g4_;                                \
            *(vec *)(b_ + 40 * 2 * PS) = g5_;                                \
            *(vec *)(b_ + 48 * 2 * PS) = g6_;                                \
            *(vec *)(b_ + 56 * 2 * PS) = g7_;                                \
        }                                                                    \
} while (0)

static void FN(x1_group)(double *restrict mid, int r)
{
    if (r == 0) {
        X1BODY((void)0);
    } else {
        /* twiddle vectors are loop-invariant per group: hoisted here */
        const vec t1r = TRE(1*r), t1i = TIA(1*r), t2r = TRE(2*r), t2i = TIA(2*r);
        const vec t3r = TRE(3*r), t3i = TIA(3*r), t4r = TRE(4*r), t4i = TIA(4*r);
        const vec t5r = TRE(5*r), t5i = TIA(5*r), t6r = TRE(6*r), t6i = TIA(6*r);
        const vec t7r = TRE(7*r), t7i = TIA(7*r);
        X1BODY(do {
            g1_ = VFMA(g1_, t1r, SWAP(g1_) * t1i);
            g2_ = VFMA(g2_, t2r, SWAP(g2_) * t2i);
            g3_ = VFMA(g3_, t3r, SWAP(g3_) * t3i);
            g4_ = VFMA(g4_, t4r, SWAP(g4_) * t4i);
            g5_ = VFMA(g5_, t5r, SWAP(g5_) * t5i);
            g6_ = VFMA(g6_, t6r, SWAP(g6_) * t6i);
            g7_ = VFMA(g7_, t7r, SWAP(g7_) * t7i);
        } while (0));
    }
}
#undef X1BODY

/* x stage 2, octet d: DFT-8 over the 8 CONSECUTIVE mid planes {8d..8d+7}
 * (one ~545 KB sequential read run), outputs to out planes {d, 8+d, ..}
 * through 8 sequential plane-streams.  nt: stream stores; pfw: write-intent
 * prefetch one line ahead on the 8 cold out streams; nx: next volume's in,
 * pre-covered FFT64B_PFN lines per ky step. */
static void FN(passB_group)(const double *restrict mid, double *restrict out,
                            int d, int nt, int pfw, const double *nx)
{
    const double *gb = mid + (size_t)(8 * d) * (2 * PS);
    double       *ob = out + (size_t)d * (2 * LSQ);
    for (int ky = 0; ky < L; ++ky) {
        if (nx) {
            const double *pn_ = nx + ((size_t)d * L + (size_t)ky) * (8 * FFT64B_PFN);
            _Pragma("GCC unroll 4")
            for (int q_ = 0; q_ < FFT64B_PFN; ++q_)
                __builtin_prefetch(pn_ + 8 * q_, 0, FFT64B_PFH);
        }
#if PW == 2
        if (nt) {   /* pair z-blocks so every NT store completes a 64-B line */
            for (int zb = 0; zb < NV; zb += 2) {
                const double *sa = gb + ((size_t)ky * RS + (size_t)zb * PW) * 2;
                const double *sb = sa + 2 * PW;
                vec Za[8], Zb[8];
                DFT8M(*(const vec *)(sa           ), *(const vec *)(sa + 1*(2*PS)),
                      *(const vec *)(sa + 2*(2*PS)), *(const vec *)(sa + 3*(2*PS)),
                      *(const vec *)(sa + 4*(2*PS)), *(const vec *)(sa + 5*(2*PS)),
                      *(const vec *)(sa + 6*(2*PS)), *(const vec *)(sa + 7*(2*PS)),
                      Za[0],Za[1],Za[2],Za[3],Za[4],Za[5],Za[6],Za[7]);
                DFT8M(*(const vec *)(sb           ), *(const vec *)(sb + 1*(2*PS)),
                      *(const vec *)(sb + 2*(2*PS)), *(const vec *)(sb + 3*(2*PS)),
                      *(const vec *)(sb + 4*(2*PS)), *(const vec *)(sb + 5*(2*PS)),
                      *(const vec *)(sb + 6*(2*PS)), *(const vec *)(sb + 7*(2*PS)),
                      Zb[0],Zb[1],Zb[2],Zb[3],Zb[4],Zb[5],Zb[6],Zb[7]);
                double *db = ob + ((size_t)ky * L + (size_t)zb * PW) * 2;
                _Pragma("GCC unroll 8")
                for (int c_ = 0; c_ < 8; ++c_) {
                    STREAM_ST(db + (size_t)c_ * (8 * 2 * LSQ),          Za[c_]);
                    STREAM_ST(db + (size_t)c_ * (8 * 2 * LSQ) + 2 * PW, Zb[c_]);
                }
            }
            continue;
        }
#endif
        for (int zb = 0; zb < NV; ++zb) {
            const double *s_ = gb + ((size_t)ky * RS + (size_t)zb * PW) * 2;
            double       *d_ = ob + ((size_t)ky * L + (size_t)zb * PW) * 2;
            if (pfw) {
                _Pragma("GCC unroll 8")
                for (int c_ = 0; c_ < 8; ++c_)
                    __builtin_prefetch(d_ + (size_t)c_ * (8 * 2 * LSQ) + 8 * FFT64B_PFWL, 1, 3);
            }
            vec z0_,z1_,z2_,z3_,z4_,z5_,z6_,z7_;
            DFT8M(*(const vec *)(s_           ), *(const vec *)(s_ + 1*(2*PS)),
                  *(const vec *)(s_ + 2*(2*PS)), *(const vec *)(s_ + 3*(2*PS)),
                  *(const vec *)(s_ + 4*(2*PS)), *(const vec *)(s_ + 5*(2*PS)),
                  *(const vec *)(s_ + 6*(2*PS)), *(const vec *)(s_ + 7*(2*PS)),
                  z0_,z1_,z2_,z3_,z4_,z5_,z6_,z7_);
            if (nt) {
                STREAM_ST(d_                          , z0_);
                STREAM_ST(d_ + (size_t)1 * (8*2*LSQ), z1_);
                STREAM_ST(d_ + (size_t)2 * (8*2*LSQ), z2_);
                STREAM_ST(d_ + (size_t)3 * (8*2*LSQ), z3_);
                STREAM_ST(d_ + (size_t)4 * (8*2*LSQ), z4_);
                STREAM_ST(d_ + (size_t)5 * (8*2*LSQ), z5_);
                STREAM_ST(d_ + (size_t)6 * (8*2*LSQ), z6_);
                STREAM_ST(d_ + (size_t)7 * (8*2*LSQ), z7_);
            } else {
                *(vec *)(d_                          ) = z0_;
                *(vec *)(d_ + (size_t)1 * (8*2*LSQ)) = z1_;
                *(vec *)(d_ + (size_t)2 * (8*2*LSQ)) = z2_;
                *(vec *)(d_ + (size_t)3 * (8*2*LSQ)) = z3_;
                *(vec *)(d_ + (size_t)4 * (8*2*LSQ)) = z4_;
                *(vec *)(d_ + (size_t)5 * (8*2*LSQ)) = z5_;
                *(vec *)(d_ + (size_t)6 * (8*2*LSQ)) = z6_;
                *(vec *)(d_ + (size_t)7 * (8*2*LSQ)) = z7_;
            }
        }
    }
}

/* st=1 pass B2: the FULL 64-point x-FFT (both radix-8 stages in registers,
 * the same FFT64V used for the z- and y-lines) per (ky, z-column), straight
 * from mid to out.  Reads are 64 streams at the padded plane stride --
 * odd-line padding spreads them over sets, hugepages keep them on 3 TLB
 * entries, and one prefetcht0 per load FFT64B_PFXC columns ahead covers the
 * L3 latency (the whole group of 64 loads is also independent, so the OoO
 * window supplies MLP on top).  Stores are 64 plane-streams to out: at PW=4
 * every store is exactly one full line, so NT stores are fill-buffer-clean
 * at any stride.  (At PW=2 an NT store is half a line -- correct but slow;
 * pw2/nt/st1 candidates are never generated, only env-forcible.)
 * Adopted from L64_radix8 r6's fused pass 2+3 (their strided in-place x-FFT
 * + next-column prefetch, +12% at B=8 there); this removes st=0's x1 RMW
 * sweep entirely. */
static void FN(passB2)(const double *restrict mid, double *restrict out,
                       int nt, const double *nx)
{
    for (int ky = 0; ky < L; ++ky)
        for (int zb = 0; zb < NV; ++zb) {
            const double *xsrc_ = mid + ((size_t)ky * RS + (size_t)zb * PW) * 2;
            double       *xdst_ = out + ((size_t)ky * L  + (size_t)zb * PW) * 2;
            if (nx) {   /* token pre-coverage of the next volume's input */
                const double *pn_ = nx + ((size_t)ky * NV + (size_t)zb)
                                         * (8 * FFT64B_PFN);
                _Pragma("GCC unroll 4")
                for (int q_ = 0; q_ < FFT64B_PFN; ++q_)
                    __builtin_prefetch(pn_ + 8 * q_, 0, FFT64B_PFH);
            }
            _Pragma("GCC unroll 64")
            for (int n_ = 0; n_ < L; ++n_)
                __builtin_prefetch(xsrc_ + (size_t)n_ * (2 * PS)
                                         + FFT64B_PFXC * (2 * PW), 0, 3);
#define LDX(n)    (*(const vec *)(xsrc_ + (size_t)(n) * (2 * PS)))
#define STX(k, v) do {                                                        \
                double *da_ = xdst_ + (size_t)(k) * (2 * LSQ);                \
                if (nt) STREAM_ST(da_, (v)); else *(vec *)da_ = (v);          \
            } while (0)
            FFT64V(LDX, STX);
#undef LDX
#undef STX
        }
}

static void FN(run_vols)(const double *restrict in, double *restrict out,
                         double *restrict S, int nvol, int mode, int pf, int st)
{
    const int pfr = (pf >= 3) ? 2 : (pf ? 1 : 0);
    const int pfw = (pf == 2 || pf == 3) && mode == M_CACHED;
    const int nt  = (mode == M_NT);
    for (int b = 0; b < nvol; ++b) {
        const double *iv = in  + (size_t)b * VDBL;
        double       *ov = out + (size_t)b * VDBL;
        const double *nx = (pf == 1 || pf == 2) && b + 1 < nvol ? iv + VDBL : NULL;
        if (st) {
            /* 2-sweep: no x1, so pass A visits planes in NATURAL order and
             * the cold in-read is one sequential 4.19 MB run */
#ifndef FFT64B_SKIPA
            for (int p = 0; p < L; ++p) {
                const double *pfnext =
                    (p < L - 1) ? iv + (size_t)(p + 1) * (2 * LSQ) : nx;
                FN(passA_plane)(iv, S, p, pfnext, pfr);
            }
#endif
#ifndef FFT64B_SKIPB
            FN(passB2)(S, ov, nt, nx);
#endif
            (void)pfw;
            continue;
        }
        /* FFT64B_SKIP*: timing-only diagnostics; the answer is WRONG with any
         * of them set (same convention as L36_pfa's FFT36_SKIP1/2) */
        for (int r = 0; r < 8; ++r) {
#ifndef FFT64B_SKIPA
            for (int a = 0; a < 8; ++a) {
                int p = r + 8 * a;
                /* next plane in VISIT order: a+1 in this group, else the
                 * next group's first plane, else the next volume */
                const double *pfnext =
                    (a < 7) ? iv + (size_t)(p + 8) * (2 * LSQ)
                  : (r < 7) ? iv + (size_t)(r + 1) * (2 * LSQ)
                  : nx;
                FN(passA_plane)(iv, S, p, pfnext, pfr);
            }
#endif
#ifndef FFT64B_SKIPX1
            FN(x1_group)(S, r);
#endif
        }
#ifndef FFT64B_SKIPB
        for (int d = 0; d < 8; ++d)
            FN(passB_group)(S, ov, d, nt, pfw, nx);
#else
        (void)ov; (void)nt; (void)pfw;
#endif
    }
}

#undef PFA1
#undef PFA1_NTA
#undef PFSTEP
#undef FFT64V
#undef DFT8M
#undef CMULT
#undef TIA
#undef TRE
#undef CSV
#undef PMV
#undef TRNC
#undef STREAM_ST
#undef VFMA
#undef VFNMA
#undef SWAP
#undef VPAIR
#undef VSPLAT
#undef RSV
#undef NV
#undef VSH
#undef FN
#undef veci
#undef vec

#endif /* template */
