/* =====================================================================================
 * L36_pencilfused.c -- forward complex-double 3D DFT of a fixed 36^3 cube, batched.
 *
 * TECHNIQUE (round panel_r3 revision)
 *   Two-pass pencil/tile-fused row-column transform with Good-Thomas (PFA) 4x9
 *   line kernels, INTERLEAVED-complex SIMD with a *spectator* axis in the vector
 *   lanes (a lane = one 128-bit re/im pair = one line of the pass).
 *
 *   ROUND panel_r3 CHANGE: the two passes are SWAPPED relative to round r2.  The
 *   plane-fused pass now runs FIRST, reading `in` (the only DRAM-cold buffer)
 *   plane-sequentially, and the strided x-pass runs SECOND against the cache-
 *   resident intermediate.  Round r2 had it the other way around, so its 36
 *   concurrent 20736-B-stride read streams hit DRAM-cold `in` -- the structural
 *   reason it trailed L36_pfa/L36_mixedradix by 5% (B=1) to 19% (batched) on the
 *   node.  This ordering is adopted from L36_pfa round r2 (who adopted it from
 *   L36_mixedradix round 1); both put the strided access on the warm buffer.
 *
 *     pass A (per x-plane, 36x36 complex = 20.25 KB, L1-resident):
 *         A1  y-transform, lanes = z (contiguous, shuffle-free), output
 *             transposed in PWxPW complex-lane blocks into a plane buffer P[z][ky]
 *         A2  z-transform, lanes = ky (P rows contiguous in ky), transposed back,
 *             stored to mid[x][ky][kz] (plane-sequential write)
 *     pass B (streaming): x-transform, lanes = PW consecutive flat (ky,kz).
 *         36 read streams from mid + 36 write streams to out, both stride
 *         20736 B; the reads carry an unconditional one-line-ahead software
 *         prefetch (L36_pfa measured removing it costs 14% at B=1 -- 36 streams
 *         exceed the L2 streamer).  Stores are non-temporal in the NT modes.
 *
 *   Two crossings of the grid, no materialised transpose anywhere; the only
 *   shuffles are the per-module swaps and the PWxPW register transposes on
 *   L1-resident data.  Both unavoidable volume transposes (see the round-1
 *   record for the proof there must be exactly one pair) act inside pass A.
 *
 *   MODES (self-tuned in fft3d_create(), which is not scored):
 *     0 INPLACE     mid = out.  Smallest resident set (in + out = 1.46 MB), the
 *                   small-batch winner: pass B rewrites out in place out of L2.
 *     1 SCRATCH     mid = a private one-volume scratch reused for every volume
 *                   (cache-resident across the call), cached final stores.
 *     2 SCRATCH+NT  as 1 but pass B's stores are non-temporal: DRAM traffic per
 *                   volume drops to the compulsory read-in + NT-write-out
 *                   (~1.5 MB) vs INPLACE's ~2.2 MB with the RFO.  At PW=2 two
 *                   flat groups are paired so every NT write completes a full
 *                   64-byte line (half-line NT stores thrash the ~12 WC buffers;
 *                   pairing trick from L36_pfa/L36_mixedradix round r2).
 *     3 SCRATCH+NT+XV  as 2, plus a CROSS-VOLUME software pipeline: while pass B
 *                   of volume v streams NT stores (store-buffer-bound, load
 *                   ports idle), prefetch volume v+1's `in` into L3
 *                   (prefetcht2, 36 lines per line-group; 324 groups x 36 lines
 *                   = exactly one volume).  Pass A of v+1 then reads L3, not
 *                   DRAM.  This is new this round and targets the monitor's
 *                   panel_r2 finding that L=36 batched runs at half the node's
 *                   demonstrated single-core streaming rate because reads and
 *                   compute do not overlap (VERDICT panel_r2 section 6: ceiling
 *                   ~123 us/volume at B=256 vs the measured 238.8).
 *     4 PIPE (new in panel_r4)  TRUE cross-volume software pipeline with REAL
 *                   work, not prefetches.  Two ping-pong mid buffers; pass A of
 *                   volume v+1 is interleaved with pass B of volume v at plane
 *                   granularity (one pass-A plane of v+1, then 9 pass-B line
 *                   groups of v, 36 times per volume).  The node rejected the
 *                   prefetch-based XV in r3 -- the leading explanation is that
 *                   prefetches issued while pass B's NT drains hold the fill
 *                   buffers simply get DROPPED, exactly when they are needed.
 *                   Demand loads cannot be dropped: interleaving the two passes
 *                   forces volume v+1's DRAM reads to execute between volume
 *                   v's NT store bursts, so the memory system always has both
 *                   reads and writes in flight.  Cost: the live mid set is two
 *                   buffers (cur draining + next filling; combined live bytes
 *                   stay ~1 volume, but LRU demotes the older buffer to L3, so
 *                   pass B's mid reads become L3 hits on the node's 1 MB L2).
 *                   This is the ping-pong pipeline L36_pfa's r3 record designed
 *                   and deferred ("if B=256 lands >= 180 us on the node, build
 *                   it" -- it landed 227.5); built here as an ADDITIONAL tuner
 *                   candidate per the r3 verdict's add-don't-replace lesson.
 *     5 SEQNT (new in panel_r4)  Sequential-store discipline: pass B runs IN
 *                   PLACE on the cache-resident mid (cached stores, PFA36
 *                   loads a line group entirely before its first store, so
 *                   in-place is well-defined), then one perfectly SEQUENTIAL
 *                   NT copy mid -> out.  Rationale: modes 2-4 drain NT stores
 *                   through 36 concurrent streams of stride 20736 B -- a DRAM
 *                   row-buffer-thrash pattern -- while the node's demonstrated
 *                   12.3 GB/s single-core stream (L6_unrolled r3) is a
 *                   SEQUENTIAL store stream.  The r3 verdict (section 5) found
 *                   store ORDER worth 18.5% at L=8 (L36-transposed precedent:
 *                   L8_radix8 r3 added a pass to sequentialize stores); this is
 *                   that move at L=36, cost = one extra cache-resident volume
 *                   round trip (~746 KB through L2/L3, no extra DRAM traffic).
 *     6 PIPESEQ (new in panel_r4)  Mode 5 with the copy pipelined: the
 *                   sequential NT copy of volume v-1 is interleaved into pass
 *                   B of volume v (36 vectors = 9*PW lines per group), as pass B
 *                   in-place on mid is pure compute with an idle memory system
 *                   -- the natural slot to hide the entire NT drain under.
 *                   Pass A's cold reads hide under its own compute via the
 *                   plane-ahead prefetch.  Ideal node schedule: ~65 us (pass A,
 *                   reads under compute) + max(60 compute, 62 NT drain) ~= 130
 *                   us/volume against the monitor's ~124 ceiling.  Ping-pong
 *                   mids as in PIPE.
 *
 * OPERATION COUNT (per 36-point line over PW lanes; unchanged from round r2)
 *   Good-Thomas 4x9: n = (9 n1 + 4 n2) mod 36, k = (9 k1 + 28 k2) mod 36, so
 *   W36^{nk} = W4^{n1 k1} * W9^{n2 k2} and the inter-factor twiddle stage vanishes.
 *
 *     DFT4  interleaved: 8 FMA-port ops + 1 swap           x9   72 + 9
 *     DFT3  interleaved: 6 FMA-port ops + 1 swap           x24 144 + 24
 *     CMUL  by constant: 2 FMA-port ops + 1 swap           x16  32 + 16
 *     PFA-36 line:       248 FMA-port ops + 49 port-5 shuffles, 36 ld + 36 st
 *
 *   Per volume: 3 x 1296 lines -> 3888/PW kernel calls; 241k FMA-port vector ops
 *   at PW=4, floor ~105 us at 2.3 GHz on the node's single 512-bit FMA pipe (or
 *   two 256-bit ones -- identical by construction, so both widths are built and
 *   the plan-time tuner decides).  Register transposes add 8 shuffles per PW
 *   vectors of PW complex, on port 5, which the FMA work leaves idle.
 *
 * ASSUMPTIONS
 *   * L == 36 only; in/out distinct and 64-byte aligned (driver guarantees); `in` const.
 *   * gcc vector extensions + explicit FMA forms (via contraction) for the modules.
 *   * Two instantiations from one source text via #include __FILE__: PW=2 with no
 *     target attribute (inherits -march=native: AVX2 on the dev box, EVEX-encoded
 *     256-bit with 32 registers on the node) and PW=4 under target("avx512f,...").
 *     fft3d_create() times all (width x mode) configurations, interleaved-rounds
 *     protocol, and every candidate must reproduce the PW=2 INPLACE answer to
 *     1e-11 relative before it is eligible, so a path that cannot be executed at
 *     development time can cost speed but never correctness.
 *   * FFT36PF_SKIPA / FFT36PF_SKIPB compile out one pass -- WRONG ANSWERS, purely
 *     a phase-timing diagnostic for tryout runs with explicit -D flags.
 * ===================================================================================== */

#ifndef L36_PENCILFUSED_TEMPLATE
#define _POSIX_C_SOURCE 200809L
/* ------------------------------------------------------------------ common part ---- */
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <stdio.h>
#include <time.h>
#include <immintrin.h>

#include "fft3d_api.h"

#define LL      36
#define NPLANE  1296                /* 36*36 complex in one (y,z) plane               */
#define NVOL2   93312               /* doubles per volume, interleaved complex        */
#define PST     36                  /* P-buffer row stride, complex (576 B = 9 lines, */
                                    /* odd in lines -> touches all 64 L1 sets)        */
#define MIDSKIP (NVOL2 + 64)        /* doubles between the two ping-pong mid buffers  */
                                    /* (+64 keeps 64-B alignment and offsets L2 sets) */

/* twiddles: W9^m = exp(-2 pi i m / 9), and sqrt(3)/2 for the 3-point module */
#define K3     0.86602540378443864676
#define W91R   0.76604444311897803520
#define W91I  (-0.64278760968653932632)
#define W92R   0.17364817766693034885
#define W92I  (-0.98480775301220805937)
#define W94R  (-0.93969262078590838405)
#define W94I  (-0.34202014332566873305)

struct fft3d_plan {
    int    batch;
    int    pw;                      /* 2 or 4 complex lanes, chosen by measurement    */
    int    mode;                    /* 0 INPLACE, 1 SCRATCH, 2 +NT, 3 +NT+XV, 4 PIPE  */
    double *mid;                    /* two-volume ping-pong scratch (PIPE uses both)  */
    double *pp;                     /* 36 x PST complex plane buffer                  */
    void   *arena;
};

/* ---------------------------------------------------------------- instantiations ---- */
#define L36_PENCILFUSED_TEMPLATE 1

#define PW     2
#define TS(x)  x##_v2
#define TATTR
#include __FILE__
#undef PW
#undef TS
#undef TATTR

#define PW     4
#define TS(x)  x##_v4
#define TATTR  __attribute__((target("avx512f,avx512dq,avx512vl")))
#include __FILE__
#undef PW
#undef TS
#undef TATTR

/* --------------------------------------------------------------------------- API ---- */

static const char *mode_name(int m)
{
    return m == 0 ? "inplace" : m == 1 ? "scratch" : m == 2 ? "scratch+nt"
                              : m == 3 ? "scratch+nt+xvpf" : m == 4 ? "pipe"
                              : m == 5 ? "scratch+seqnt" : "pipeseq";
}

/* the tuner's pick is written here so the monitor can read it off the raw
 * per-case json (VERDICT panel_r2, cross-cutting item 2) */
static char g_desc[224] =
    "L=36: plane-fused y+z pass then strided x pass, PFA 4x9 interleaved-complex "
    "line kernel, in-place/NT/cross-volume-prefetch modes self-tuned (pre-create)";

const char *fft3d_name(void)        { return "L36_pencilfused"; }
const char *fft3d_description(void) { return g_desc; }
int fft3d_supports(int L)           { return L == LL; }

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

void fft3d_execute(fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    const double *ind = (const double *)in;
    double *outd = (double *)out;
    if (p->pw == 4)
        exec_v4(ind, outd, p->mid, p->pp, p->batch, p->mode);
    else
        exec_v2(ind, outd, p->mid, p->pp, p->batch, p->mode);
}

fft3d_plan *fft3d_create(int L, int batch)
{
    if (L != LL || batch < 1) return NULL;

    fft3d_plan *p = calloc(1, sizeof *p);
    if (!p) return NULL;
    p->batch = batch;

    const size_t nMid = (size_t)2 * MIDSKIP;      /* two volumes: PIPE ping-pongs */
    const size_t nP   = (size_t)36 * PST * 2 + 64;
    void *arena = NULL;
    if (posix_memalign(&arena, 64, (nMid + nP) * sizeof(double)) != 0 || !arena) {
        free(p);
        return NULL;
    }
    memset(arena, 0, (nMid + nP) * sizeof(double));
    p->arena = arena;
    p->mid = (double *)arena;
    p->pp  = p->mid + nMid;

    /* defaults if the tuning allocation fails: 256-bit; NT once the batch has
     * clearly left L3 (r1/r2 evidence: NT is neutral-to-winning only there) */
    p->pw   = 2;
    p->mode = ((double)batch * 1492992.0 > 16.0 * 1024.0 * 1024.0) ? 2 : 0;

    /* ---- self-tune: {PW=2, PW=4} x {INPLACE, SCRATCH, NT, NT+XV}, timed and
     * CROSS-CHECKED.  Every configuration must reproduce the PW=2 INPLACE answer
     * on this machine to 1e-11 relative before it may be selected.
     * The arena is capped at 64 volumes (~191 MB in+out) so the NT-vs-cached
     * ranking actually streams past every L3 this code will meet (L36_pfa r2:
     * a 16-volume arena fit wallaby's 60 MB L3 and mis-picked cached stores;
     * this file's first r3 attempt repeated that at 48 volumes -- tuner said
     * cached 80.8 vs NT+XV 95.4, the full 382 MB run said 146.4 vs 119.2). */
    const long tb = batch < 64 ? batch : 64;
    const long rv = tb < 2 ? tb : 2;    /* volumes checked for admission */
    double *din = NULL, *dout = NULL, *dref = NULL;
    if (posix_memalign((void **)&din,  64, (size_t)tb * NVOL2 * sizeof(double)) == 0 &&
        posix_memalign((void **)&dout, 64, (size_t)tb * NVOL2 * sizeof(double)) == 0 &&
        posix_memalign((void **)&dref, 64, (size_t)rv * NVOL2 * sizeof(double)) == 0) {

        unsigned st = 12345u;
        for (size_t i = 0; i < (size_t)tb * NVOL2; ++i) {
            st = st * 1103515245u + 12345u;
            din[i] = (double)(int)(st >> 16) * 3.0517578125e-5;
        }
        exec_v2(din, dout, p->mid, p->pp, rv, 0);
        memcpy(dref, dout, (size_t)rv * NVOL2 * sizeof(double));
        double refmax = 0.0;
        for (size_t i = 0; i < (size_t)rv * NVOL2; ++i) {
            double a = dref[i] < 0 ? -dref[i] : dref[i];
            if (a > refmax) refmax = a;
        }
        if (refmax <= 0.0) refmax = 1.0;

        /* Admission first (correctness gate), then time the survivors over
         * INTERLEAVED rounds keeping each candidate's minimum -- contiguous
         * per-candidate blocks let one load spike mis-rank the whole plan
         * (round-r2 lesson, protocol from L36_mixedradix round 1).  NT/XV
         * candidates are admitted only once the batch has clearly left L3;
         * below that a forced DRAM write per call can only lose (round-1
         * measurement: NT neutral at B<=4, 1.53x win at B=32). */
        const int have512  = __builtin_cpu_supports("avx512f");
        const int allow_nt = ((double)batch * 1492992.0 > 16.0 * 1024.0 * 1024.0);
        double bestc[14];
        int ok[14];
        for (int cfg = 0; cfg < 14; ++cfg) {         /* cfg = 7*(pw4?1:0) + mode */
            const int w = (cfg >= 7) ? 4 : 2;
            const int m = cfg % 7;
            bestc[cfg] = 1e300;
            ok[cfg] = 0;
            if (w == 4 && !have512) continue;
            if (m >= 2 && !allow_nt) continue;
#ifdef FFT36PF_FORCE_PW
            if (w != FFT36PF_FORCE_PW) continue;
#endif
#ifdef FFT36PF_FORCE_MODE
            if (m != FFT36PF_FORCE_MODE) continue;
#endif
            if (w == 4) exec_v4(din, dout, p->mid, p->pp, rv, m);
            else        exec_v2(din, dout, p->mid, p->pp, rv, m);
            double err = 0.0;
            for (size_t i = 0; i < (size_t)rv * NVOL2; ++i) {
                double d = dout[i] - dref[i];
                if (d < 0) d = -d;
                if (d > err) err = d;
            }
            ok[cfg] = (err <= 1e-11 * refmax);
        }
        /* many short rounds rather than few long ones: the finer the interleave,
         * the less a load burst on a shared machine can distort one minimum */
        const int inner  = (tb <= 2) ? 6 : (tb <= 8 ? 3 : 1);
        const int rounds = (tb <= 2) ? 12 : (tb <= 8 ? 8 : 4);
        for (int r = 0; r < rounds; ++r) {
            for (int cfg = 0; cfg < 14; ++cfg) {
                if (!ok[cfg]) continue;
                const int w = (cfg >= 7) ? 4 : 2;
                const int m = cfg % 7;
                /* round 0 doubles as warm-up; its (worst-case) timing enters
                 * the minimum, which is harmless */
                double t0 = now_s();
                for (int q = 0; q < inner; ++q) {
                    if (w == 4) exec_v4(din, dout, p->mid, p->pp, tb, m);
                    else        exec_v2(din, dout, p->mid, p->pp, tb, m);
                }
                double t = (now_s() - t0) / inner;
                if (t < bestc[cfg]) bestc[cfg] = t;
            }
        }
        double bestt = 1e300;
        for (int cfg = 0; cfg < 14; ++cfg) {
            if (!ok[cfg] || bestc[cfg] >= bestt) continue;
            bestt = bestc[cfg];
            p->pw   = (cfg >= 7) ? 4 : 2;
            p->mode = cfg % 7;
        }
        if (getenv("FFT36PF_VERBOSE")) {
            for (int cfg = 0; cfg < 14; ++cfg)
                if (ok[cfg])
                    fprintf(stderr, "L36_pencilfused tuner: pw=%d %-15s %8.1f us/vol\n",
                            (cfg >= 7) ? 4 : 2, mode_name(cfg % 7),
                            bestc[cfg] * 1e6 / tb);
            fprintf(stderr, "L36_pencilfused tuner: chose pw=%d %s (tb=%ld)\n",
                    p->pw, mode_name(p->mode), tb);
        }
    }
    free(din);
    free(dout);
    free(dref);

    snprintf(g_desc, sizeof g_desc,
             "L=36 plane-fused y+z then strided x, PFA4x9 interleaved lanes; "
             "tuner picked pw=%d mode=%s (B=%d)",
             p->pw, mode_name(p->mode), batch);
    return p;
}

void fft3d_destroy(fft3d_plan *p)
{
    if (!p) return;
    free(p->arena);
    free(p);
}

#else
/* ============================== templated kernel body ============================== */
/* A vector holds PW complex numbers = 2*PW doubles, interleaved re,im.               */

typedef double    TS(vd) __attribute__((vector_size(2 * PW * 8)));
typedef long long TS(vi) __attribute__((vector_size(2 * PW * 8)));
typedef double    TS(vu) __attribute__((vector_size(2 * PW * 8), aligned(8), may_alias));

#define vd  TS(vd)
#define vi  TS(vi)
#define vu  TS(vu)
#define LD(p)     (*(const vu *)(const void *)(p))
#define ST(p, v)  (*(vu *)(void *)(p) = (v))
#if PW == 2
#  ifdef __AVX__
#    define STNT(p, v) _mm256_stream_pd((double *)(p), (__m256d)(v))
#  else
#    define STNT(p, v) ST(p, v)
#  endif
#  define VC(a, b) ((vd){ (a), (b), (a), (b) })
#  define SWPM ((vi){ 1, 0, 3, 2 })
#else
#  define STNT(p, v) _mm512_stream_pd((double *)(p), (__m512d)(v))
#  define VC(a, b) ((vd){ (a), (b), (a), (b), (a), (b), (a), (b) })
#  define SWPM ((vi){ 1, 0, 3, 2, 5, 4, 7, 6 })
#endif
#define VS(a)    VC(a, a)
#define CSWAP(v) __builtin_shuffle((v), SWPM)   /* swap re/im in every complex lane */

/* --- PW x PW transpose of complex (128-bit) lanes, butterfly form ------------------ */
#define CTRSTEP(a, i, s) do {                                                          \
    vi ml_, mh_;                                                                      \
    for (int j_ = 0; j_ < 2 * PW; ++j_) {                                             \
        int g_ = j_ / 2, d_ = j_ & 1;                                                 \
        int b_ = g_ / (s), o_ = g_ % (s);                                             \
        long long sl_ = (b_ & 1) ? (PW + (b_ - 1) * (s) + o_) : (b_ * (s) + o_);      \
        ((long long *)&ml_)[j_] = 2 * sl_ + d_;                                       \
        ((long long *)&mh_)[j_] = 2 * (sl_ + (s)) + d_;                               \
    }                                                                                 \
    vd lo_ = __builtin_shuffle((a)[i], (a)[(i) + (s)], ml_);                           \
    vd hi_ = __builtin_shuffle((a)[i], (a)[(i) + (s)], mh_);                           \
    (a)[i] = lo_; (a)[(i) + (s)] = hi_;                                                \
} while (0)

#if PW == 2
#  define CTRANSPOSE(a) do { CTRSTEP(a,0,1); } while (0)
#else
#  define CTRANSPOSE(a) do { CTRSTEP(a,0,1); CTRSTEP(a,2,1);                           \
                             CTRSTEP(a,0,2); CTRSTEP(a,1,2); } while (0)
#endif

/* --- the modules (interleaved complex; constants are lane-splatted pairs) ---------- */
/* forward DFT-3: y1 = x0 - t/2 - i*K3*(x1-x2), y2 the conjugate partner.
 * 6 FMA-port ops + 1 swap: KSV = (K3,-K3,...) makes KSV*swap(m) = -i*K3*m exactly. */
#define DFT3I(X0, X1, X2, Y0, Y1, Y2) do {                                             \
    vd t_ = (X1) + (X2), m_ = (X1) - (X2);                                             \
    (Y0) = (X0) + t_;                                                                  \
    vd a_ = (X0) - VS(0.5) * t_;                                                       \
    vd ms_ = CSWAP(m_);                                                                \
    (Y1) = a_ + VC(K3, -K3) * ms_;                                                     \
    (Y2) = a_ - VC(K3, -K3) * ms_;                                                     \
} while (0)

/* complex multiply by the constant (WR + i WI): 2 FMA-port ops + 1 swap.
 * out = WR*v + (-WI,WI)*swap(v). */
#define CMULI(V, WR, WI, O) do {                                                       \
    vd q_ = CSWAP(V);                                                                  \
    (O) = VS(WR) * (V) + VC(-(WI), (WI)) * q_;                                         \
} while (0)

/* Good-Thomas index maps for 36 = 4 * 9:  n = (9 n1 + 4 n2) mod 36,
 * k = (9 k1 + 28 k2) mod 36, whence n k = 9 n1 k1 + 4 n2 k2 (mod 36). */
#define IX(n1,n2)  (((9 * (n1) + 4 * (n2)) % 36))
#define OX(k1,k2)  (((9 * (k1) + 28 * (k2)) % 36))
#define UU(k1,j)   ((k1) * 9 + (j))

/* stage A, one of nine DFT-4 over n1.  y1 = t1 - i*t3 and y3 = t1 + i*t3 via
 * (1,-1)*swap(t3).  8 FMA-port ops + 1 swap.  LOAD(j,X) supplied by the caller. */
#define SA_(n2, LOAD) do {                                                             \
    vd q0, q1, q2, q3;                                                                 \
    LOAD(IX(0,n2), q0)                                                                 \
    LOAD(IX(1,n2), q1)                                                                 \
    LOAD(IX(2,n2), q2)                                                                 \
    LOAD(IX(3,n2), q3)                                                                 \
    vd t0_ = q0 + q2, t1_ = q0 - q2;                                                   \
    vd t2_ = q1 + q3, t3_ = q1 - q3;                                                   \
    u[UU(0,n2)] = t0_ + t2_;                                                           \
    u[UU(2,n2)] = t0_ - t2_;                                                           \
    vd sw_ = CSWAP(t3_);                                                               \
    u[UU(1,n2)] = t1_ + VC(1.0, -1.0) * sw_;                                           \
    u[UU(3,n2)] = t1_ - VC(1.0, -1.0) * sw_;                                           \
} while (0)

/* stage B, one of four DFT-9 over n2, Cooley-Tukey 3x3: three DFT-3 on the decimated
 * sub-sequences, four nontrivial W9 twiddles, three DFT-3 across. */
#define SB_(k1, STORE) do {                                                            \
    vd a00, a01, a02, a10, a11, a12, a20, a21, a22;                                    \
    DFT3I(u[UU(k1,0)], u[UU(k1,3)], u[UU(k1,6)], a00, a01, a02);                       \
    DFT3I(u[UU(k1,1)], u[UU(k1,4)], u[UU(k1,7)], a10, a11, a12);                       \
    DFT3I(u[UU(k1,2)], u[UU(k1,5)], u[UU(k1,8)], a20, a21, a22);                       \
    { vd o0, o1, o2;                                                                   \
      DFT3I(a00, a10, a20, o0, o1, o2);                                                \
      STORE(OX(k1,0), o0) STORE(OX(k1,3), o1) STORE(OX(k1,6), o2) }                    \
    { vd b1, b2, o0, o1, o2;                                                           \
      CMULI(a11, W91R, W91I, b1);                                                      \
      CMULI(a21, W92R, W92I, b2);                                                      \
      DFT3I(a01, b1, b2, o0, o1, o2);                                                  \
      STORE(OX(k1,1), o0) STORE(OX(k1,4), o1) STORE(OX(k1,7), o2) }                    \
    { vd c1, c2, o0, o1, o2;                                                           \
      CMULI(a12, W92R, W92I, c1);                                                      \
      CMULI(a22, W94R, W94I, c2);                                                      \
      DFT3I(a02, c1, c2, o0, o1, o2);                                                  \
      STORE(OX(k1,2), o0) STORE(OX(k1,5), o1) STORE(OX(k1,8), o2) }                    \
} while (0)

/* the whole 36-point line: 248 FMA-port ops + 49 shuffles over PW lanes.
 * ALL 36 loads happen in the SA_ stage, before the first SB_ store, so LOAD and
 * STORE may alias -- pass B relies on this for its in-place mode. */
#define PFA36(LOAD, STORE) do {                                                        \
    vd u[36];                                                                          \
    SA_(0,LOAD); SA_(1,LOAD); SA_(2,LOAD); SA_(3,LOAD); SA_(4,LOAD);                   \
    SA_(5,LOAD); SA_(6,LOAD); SA_(7,LOAD); SA_(8,LOAD);                                \
    SB_(0,STORE); SB_(1,STORE); SB_(2,STORE); SB_(3,STORE);                            \
} while (0)

/* --- pass A over ONE x-plane, streaming (z-first, transpose-on-load) variant ----
 * A1: z-transform, lanes = PW y-rows via transpose-on-load, output transposed
 * back into P[y][kz].  The transpose-on-load order keeps the cold `in` reads
 * SEQUENTIAL in PW streams (adopted from L36_pfa r2 phase 1): the y-first order
 * read the plane through 36 stride-576B streams and measured 101 vs 58 us/vol
 * at B=256 cold.  A2: y-transform, lanes = kz (P rows contiguous, shuffle-free),
 * stored straight to the mid plane -- 36 scattered 64-B store streams, which
 * the store buffer absorbs (scattered LOADS were the r2 mistake).
 * npf, when non-null, is the NEXT plane to pull into L2 (T1) while this plane
 * computes -- worth 8.5 us/vol at B=256 (r3 measurement). ---------------------- */
TATTR static void TS(passA_plane)(const double *restrict rsrc, double *restrict rdst,
                                  double *restrict pp, const double *restrict npf)
{
#ifdef FFT36PF_NOPAPF
    npf = 0;
#endif
    for (int yb = 0; yb < 36; yb += PW) {
        vd Zv[36], Wv[36];
        for (int zb = 0; zb < 36 / PW; ++zb) {
            vd t[PW];
            for (int j = 0; j < PW; ++j)
                t[j] = LD(rsrc + 2 * ((size_t)(yb + j) * 36 + (size_t)zb * PW));
            CTRANSPOSE(t);
            for (int l = 0; l < PW; ++l)
                Zv[zb * PW + l] = t[l];
        }
        if (npf)
            for (int j_ = 0; j_ < (9 * PW); ++j_)
                __builtin_prefetch(npf + ((size_t)(yb / PW) * (9 * PW) + j_) * 8, 0, 2);
#define ZLOAD(j, X)  { (X) = Zv[j]; }
#define ZSTORE(k, X) { Wv[k] = (X); }
        PFA36(ZLOAD, ZSTORE);
#undef ZLOAD
#undef ZSTORE
        for (int kb = 0; kb < 36 / PW; ++kb) {
            vd t[PW];
            for (int i = 0; i < PW; ++i) t[i] = Wv[kb * PW + i];
            CTRANSPOSE(t);
            for (int l = 0; l < PW; ++l)
                ST(pp + 2 * ((size_t)(yb + l) * PST + (size_t)kb * PW), t[l]);
        }
    }
    for (int kb = 0; kb < 36 / PW; ++kb) {
        const double *s = pp + 2 * (size_t)kb * PW;
#define YLOAD(j, X)  { (X) = LD(s + 2 * ((size_t)(j) * PST)); }
#define YSTORE(k, X) { ST(rdst + 2 * ((size_t)(k) * 36 + (size_t)kb * PW), X); }
        PFA36(YLOAD, YSTORE);
#undef YLOAD
#undef YSTORE
    }
}

/* --- pass B, cached-store path, flat groups [g0,g1).  NO restrict: INPLACE mode
 * calls this with mid == outv (PFA36 loads everything before its first store,
 * so aliasing is well-defined).  wpf adds a write-intent prefetch on the dst
 * streams 4 lines ahead (SCRATCH mode only, where dst is cold and demand-RFO
 * would serialize; adopted from L6_unrolled's prefetchw, node-validated). ------ */
TATTR static void TS(passB_cached)(const double *mid, double *outv,
                                   int g0, int g1, int wpf)
{
    for (int g = g0; g < g1; ++g) {
        const double *src = mid  + 2 * (size_t)g * PW;
        double       *dst = outv + 2 * (size_t)g * PW;
        for (int j_ = 0; j_ < 36; ++j_)
            __builtin_prefetch(src + (size_t)j_ * (2 * NPLANE) + 8, 0, 3);
        if (wpf)
            for (int j_ = 0; j_ < 36; ++j_)
                __builtin_prefetch(dst + (size_t)j_ * (2 * NPLANE) + 32, 1, 3);
#define BLOAD(j, X)  { (X) = LD(src + (size_t)(j) * (2 * NPLANE)); }
#define BSTORE(k, X) { ST(dst + (size_t)(k) * (2 * NPLANE), X); }
        PFA36(BLOAD, BSTORE);
#undef BLOAD
#undef BSTORE
    }
}

/* --- pass B, NT-store path, over UNITS [u0,u1) where a unit is one flat group
 * at PW=4 and one PAIR of flat groups at PW=2 (32-B NT stores are half a cache
 * line; pairing completes every 64-B line back-to-back -- L36_pfa/mixedradix
 * r2 trick).  Either way there are exactly 324 units per volume, 9 per output
 * x-slot, which is what the PIPE interleave relies on.  nxt (mode 3 only)
 * prefetches the next volume's input, 36 lines per unit = one volume total.
 * pfd is the src prefetch distance in doubles: 8 (one line) when mid is
 * L2-resident (modes 2/3), 16 when mid lives in L3 (PIPE). ------------------- */
TATTR static void TS(passB_nt)(const double *restrict mid, double *restrict outv,
                               int u0, int u1, const double *restrict nxt, int pfd)
{
#if PW == 4
    for (int g = u0; g < u1; ++g) {
        const double *src = mid  + 2 * (size_t)g * PW;
        double       *dst = outv + 2 * (size_t)g * PW;
        for (int j_ = 0; j_ < 36; ++j_)
            __builtin_prefetch(src + (size_t)j_ * (2 * NPLANE) + pfd, 0, 3);
        /* XV: stage the NEXT volume's input into L3 while this pass is
         * store-drain-bound.  324 units x 36 lines = the whole volume. */
        if (nxt)
            for (int j_ = 0; j_ < 36; ++j_)
                __builtin_prefetch(nxt + ((size_t)g * 36 + j_) * 8, 0, 1);
#define BLOAD(j, X)    { (X) = LD(src + (size_t)(j) * (2 * NPLANE)); }
#define BSTORENT(k, X) { STNT(dst + (size_t)(k) * (2 * NPLANE), X); }
        PFA36(BLOAD, BSTORENT);
#undef BLOAD
#undef BSTORENT
    }
#else
    for (int gp = u0; gp < u1; ++gp) {
        const double *src = mid  + 4 * (size_t)gp * PW;
        double       *dst = outv + 4 * (size_t)gp * PW;
        for (int j_ = 0; j_ < 36; ++j_)
            __builtin_prefetch(src + (size_t)j_ * (2 * NPLANE) + pfd, 0, 3);
        if (nxt)
            for (int j_ = 0; j_ < 36; ++j_)
                __builtin_prefetch(nxt + ((size_t)gp * 36 + j_) * 8, 0, 1);
        vd Wa[36], Wb[36];
#define BLOADA(j, X) { (X) = LD(src + (size_t)(j) * (2 * NPLANE)); }
#define BSTA(k, X)   { Wa[k] = (X); }
        PFA36(BLOADA, BSTA);
#undef BLOADA
#undef BSTA
#define BLOADB(j, X) { (X) = LD(src + (size_t)(j) * (2 * NPLANE) + 2 * PW); }
#define BSTB(k, X)   { Wb[k] = (X); }
        PFA36(BLOADB, BSTB);
#undef BLOADB
#undef BSTB
        for (int k_ = 0; k_ < 36; ++k_) {
            STNT(dst + (size_t)k_ * (2 * NPLANE),          Wa[k_]);
            STNT(dst + (size_t)k_ * (2 * NPLANE) + 2 * PW, Wb[k_]);
        }
    }
#endif
}

/* --- pass B in place on mid, with an optional interleaved SEQUENTIAL NT copy of
 * ANOTHER (already fully transformed) volume csrc -> cdst: 36 vectors' worth of
 * copy per line group = 9*PW lines, so the copy finishes exactly with the pass.
 * The pass itself is pure cache-resident compute (loads and stores both on mid),
 * so the interleaved NT drain rides on an otherwise idle memory system. -------- */
TATTR static void TS(passB_copy)(double *mid, const double *restrict csrc,
                                 double *restrict cdst)
{
    for (int g = 0; g < NPLANE / PW; ++g) {
        const double *src = mid + 2 * (size_t)g * PW;
        double       *dst = mid + 2 * (size_t)g * PW;
        for (int j_ = 0; j_ < 36; ++j_)
            __builtin_prefetch(src + (size_t)j_ * (2 * NPLANE) + 8, 0, 3);
        if (csrc) {
            const double *cs = csrc + (size_t)g * (9 * PW) * 8;
            double       *cd = cdst + (size_t)g * (9 * PW) * 8;
            for (int c = 0; c < 36; ++c)
                STNT(cd + (size_t)c * (2 * PW), LD(cs + (size_t)c * (2 * PW)));
        }
#define BLOAD(j, X)  { (X) = LD(src + (size_t)(j) * (2 * NPLANE)); }
#define BSTORE(k, X) { ST(dst + (size_t)(k) * (2 * NPLANE), X); }
        PFA36(BLOAD, BSTORE);
#undef BLOAD
#undef BSTORE
    }
}

/* sequential NT copy of one whole volume (the pipeline tail / mode-5 phase 3) */
TATTR static void TS(seqcopy_nt)(const double *restrict csrc, double *restrict cdst)
{
    for (size_t c = 0; c < NVOL2 / (2 * PW); ++c)
        STNT(cdst + c * (2 * PW), LD(csrc + c * (2 * PW)));
}

/* --- the driver ------------------------------------------------------------------ */
TATTR static void TS(exec)(const double *restrict in, double *restrict out,
                           double *restrict mids, double *restrict pp,
                           long nvol, int mode)
{
    if (mode == 5) {
        /* SEQNT, phase-serial: A (in -> mid), B in place on mid, sequential NT
         * copy mid -> out.  One mid; the extra volume round trip stays in cache. */
        for (long v = 0; v < nvol; ++v) {
            const double *inv  = in  + (size_t)v * NVOL2;
            double       *outv = out + (size_t)v * NVOL2;
#ifndef FFT36PF_SKIPA
            for (int x = 0; x < 36; ++x)
                TS(passA_plane)(inv + (size_t)x * 2 * NPLANE,
                                mids + (size_t)x * 2 * NPLANE, pp,
                                x + 1 < 36 ? inv + (size_t)(x + 1) * 2 * NPLANE
                                           : (const double *)0);
#endif
#ifndef FFT36PF_SKIPB
            TS(passB_copy)(mids, (const double *)0, (double *)0);
#endif
            TS(seqcopy_nt)(mids, outv);
        }
        _mm_sfence();
        return;
    }

    if (mode == 6) {
        /* PIPESEQ: per volume, pass A (cold reads under its own compute via the
         * plane-ahead prefetch), then pass B in place with the PREVIOUS volume's
         * sequential NT copy interleaved into it.  Last volume's copy runs bare. */
        double *bufa = mids, *bufb = mids + MIDSKIP;
        for (long v = 0; v < nvol; ++v) {
            const double *inv = in + (size_t)v * NVOL2;
            double *nb = (v & 1) ? bufb : bufa;      /* mid for volume v        */
            double *pb = (v & 1) ? bufa : bufb;      /* holds transformed v-1   */
#ifndef FFT36PF_SKIPA
            for (int x = 0; x < 36; ++x)
                TS(passA_plane)(inv + (size_t)x * 2 * NPLANE,
                                nb + (size_t)x * 2 * NPLANE, pp,
                                x + 1 < 36 ? inv + (size_t)(x + 1) * 2 * NPLANE
                                           : (const double *)0);
#endif
#ifndef FFT36PF_SKIPB
            TS(passB_copy)(nb, v > 0 ? pb : (const double *)0,
                           v > 0 ? out + (size_t)(v - 1) * NVOL2 : (double *)0);
#endif
        }
        TS(seqcopy_nt)((nvol & 1) ? bufa : bufb, out + (size_t)(nvol - 1) * NVOL2);
        _mm_sfence();
        return;
    }

    if (mode == 4) {
        /* PIPE: pass A of volume v+1 interleaved with pass B (NT) of volume v
         * at plane granularity -- one pass-A plane (cold sequential DRAM reads
         * + plane-ahead T1 prefetch), then 9 pass-B units (NT store drains).
         * The demand reads CANNOT be dropped the way mode 3's prefetches can,
         * so the memory system always holds both reads and writes in flight.
         * Two ping-pong mid buffers; combined live set stays ~1 volume, but
         * the buffer being read is LRU-old, so its lines come from L3 (hence
         * pfd=16, two lines ahead, on the mid streams). */
        double *bufa = mids, *bufb = mids + MIDSKIP;
#ifndef FFT36PF_SKIPA
        for (int x = 0; x < 36; ++x)
            TS(passA_plane)(in + (size_t)x * 2 * NPLANE,
                            bufa + (size_t)x * 2 * NPLANE, pp,
                            x + 1 < 36 ? in + (size_t)(x + 1) * 2 * NPLANE
                                       : (const double *)0);
#endif
        for (long v = 0; v < nvol; ++v) {
            double       *cur  = (v & 1) ? bufb : bufa;
            double       *nmid = (v & 1) ? bufa : bufb;
            double       *outv = out + (size_t)v * NVOL2;
            const double *nin  = (v + 1 < nvol)
                                 ? in + (size_t)(v + 1) * NVOL2 : (const double *)0;
            for (int x = 0; x < 36; ++x) {
#ifndef FFT36PF_SKIPA
                if (nin)
                    TS(passA_plane)(nin + (size_t)x * 2 * NPLANE,
                                    nmid + (size_t)x * 2 * NPLANE, pp,
                                    x + 1 < 36 ? nin + (size_t)(x + 1) * 2 * NPLANE
                                               : (const double *)0);
#endif
#ifndef FFT36PF_SKIPB
                TS(passB_nt)(cur, outv, x * 9, x * 9 + 9, (const double *)0, 16);
#endif
            }
        }
        _mm_sfence();
        return;
    }

    const int nt = (mode >= 2);
    for (long v = 0; v < nvol; ++v) {
        const double *inv  = in  + (size_t)v * NVOL2;
        double       *outv = out + (size_t)v * NVOL2;
        double       *mid  = (mode == 0) ? outv : mids;
        const double *nxt  = (mode == 3 && v + 1 < nvol)
                             ? in + (size_t)(v + 1) * NVOL2 : (const double *)0;

#ifndef FFT36PF_SKIPA
        /* ------- pass A: two transforms fused over one L1-resident x-plane ------- */
        if (mode == 0) {
        /* INPLACE (small-batch) variant: y-transform first, lanes = z, so the
         * kernel's loads come straight off the plane with no staging array and
         * no load-side shuffles -- the register-pressure-friendly order.  The
         * plane is cache-warm at small batch, so the scattered (stride-576B)
         * load pattern costs nothing here; at streaming batch it costs ~2x
         * (101 vs 58 us/vol measured at B=256 cold), hence the else-branch. */
        for (int x = 0; x < 36; ++x) {
            const double *rsrc = inv  + (size_t)x * 2 * NPLANE;
            double       *rdst = outv + (size_t)x * 2 * NPLANE;

            /* y-transform (lanes = z), transposed on the way into P[z][ky] */
            for (int zg = 0; zg < 36 / PW; ++zg) {
                const double *s = rsrc + 2 * (size_t)zg * PW;
                vd yv[36];
#define YLOAD(j, X)  { (X) = LD(s + (size_t)(j) * 72); }
#define YSTORE(k, X) { yv[k] = (X); }
                PFA36(YLOAD, YSTORE);
#undef YLOAD
#undef YSTORE
                for (int kg = 0; kg < 36 / PW; ++kg) {
                    vd t[PW];
                    for (int i = 0; i < PW; ++i) t[i] = yv[kg * PW + i];
                    CTRANSPOSE(t);
                    for (int l = 0; l < PW; ++l)
                        ST(pp + 2 * ((size_t)(zg * PW + l) * PST + kg * PW), t[l]);
                }
            }

            /* z-transform (lanes = ky), transposed back, stored to the plane */
            for (int kg = 0; kg < 36 / PW; ++kg) {
                const double *s = pp + 2 * (size_t)kg * PW;
                vd zv[36];
#define ZLOAD(j, X)  { (X) = LD(s + 2 * ((size_t)(j) * PST)); }
#define ZSTORE(k, X) { zv[k] = (X); }
                PFA36(ZLOAD, ZSTORE);
#undef ZLOAD
#undef ZSTORE
                for (int cg = 0; cg < 36 / PW; ++cg) {
                    vd t[PW];
                    for (int i = 0; i < PW; ++i) t[i] = zv[cg * PW + i];
                    CTRANSPOSE(t);
                    for (int l = 0; l < PW; ++l)
                        ST(rdst + 2 * ((size_t)(kg * PW + l) * 36 + cg * PW), t[l]);
                }
            }
        }
        } else {
        /* SCRATCH/streaming variant: z-transform first via transpose-on-load */
        for (int x = 0; x < 36; ++x)
            TS(passA_plane)(inv + (size_t)x * 2 * NPLANE,
                            mid + (size_t)x * 2 * NPLANE, pp,
                            x + 1 < 36 ? inv + (size_t)(x + 1) * 2 * NPLANE
                                       : (const double *)0);
        }
#endif /* FFT36PF_SKIPA */

#ifndef FFT36PF_SKIPB
        /* -------- pass B: x-transform, lanes = PW consecutive flat (ky,kz).
         * 36 read + 36 write streams of stride 20736 B; the read streams exceed
         * the L2 streamer, so prefetch each one line ahead unconditionally. ---- */
        if (!nt)
            TS(passB_cached)(mid, outv, 0, NPLANE / PW, mode == 1);
        else
            TS(passB_nt)(mid, outv, 0, 324, nxt, 8);
#endif /* FFT36PF_SKIPB */
    }
    if (nt) _mm_sfence();
}

#undef vd
#undef vi
#undef vu
#undef LD
#undef ST
#undef STNT
#undef VC
#undef VS
#undef SWPM
#undef CSWAP
#undef CTRSTEP
#undef CTRANSPOSE
#undef DFT3I
#undef CMULI
#undef IX
#undef OX
#undef UU
#undef SA_
#undef SB_
#undef PFA36
#endif
