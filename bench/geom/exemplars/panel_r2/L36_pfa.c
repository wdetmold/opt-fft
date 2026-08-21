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

enum { M_INPLACE = 0, M_SCRATCH = 1, M_SCRATCH_NT = 2 };

struct fft3d_plan {
    int     batch;
    int     pw;                  /* 2 or 4                         */
    int     mode;                /* one of M_*                     */
    double *S;                   /* one reused scratch volume      */
    void   *rawS;
};

const char *fft3d_name(void) { return "L36_pfa"; }
const char *fft3d_description(void)
{
    return "Good-Thomas PFA 4x9, interleaved-complex lanes, two sweeps "
           "(z+y fused per x-plane, x in place or via reused scratch + NT "
           "stores), variant autotuned in create()";
}
int fft3d_supports(int Lq) { return Lq == L; }

static void run_vols(int pw, int mode, double *S,
                     const double *in, double *out, int nvol)
{
    for (int b = 0; b < nvol; ++b) {
        const double *i = in  + (size_t)b * VDBL;
        double       *o = out + (size_t)b * VDBL;
        double     *mid = (mode == M_INPLACE) ? o : S;
        const int    nt = (mode == M_SCRATCH_NT);
        /* FFT36_SKIP1/2: temporary diagnostics, wrong answers, timing only */
#ifndef FFT36_SKIP1
# define P1(f, a, b) f(a, b)
#else
# define P1(f, a, b) ((void)0)
#endif
#ifndef FFT36_SKIP2
# define P2(f, a, b, c) f(a, b, c)
#else
# define P2(f, a, b, c) ((void)0)
#endif
#ifdef HAVE_PW4
        if (pw == 4) { P1(phase1_pw4, i, mid); P2(phase2_pw4, mid, o, nt); }
        else
#endif
        { (void)pw; P1(phase1_pw2, i, mid); P2(phase2_pw2, mid, o, nt); }
    }
    if (mode == M_SCRATCH_NT) _mm_sfence();
}

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
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
    p->pw = 2; p->mode = M_SCRATCH;      /* safe default */

    /* candidate list, first entry doubles as the numerical reference */
    struct cand { int pw, mode; } cands[6];
    int nc = 0;
    cands[nc++] = (struct cand){2, M_SCRATCH};
    cands[nc++] = (struct cand){2, M_INPLACE};
    cands[nc++] = (struct cand){2, M_SCRATCH_NT};
#ifdef HAVE_PW4
    cands[nc++] = (struct cand){4, M_SCRATCH};
    cands[nc++] = (struct cand){4, M_INPLACE};
    cands[nc++] = (struct cand){4, M_SCRATCH_NT};
#endif
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

    /* tuning arena: must actually leave L3 at large batch, or the NT-vs-
     * cached-store ranking inverts (a 16-volume arena fit wallaby's 60 MB L3
     * and mis-picked cached stores at B=256; 64 volumes = 96 MB does not) */
    const int nv = batch < 64 ? batch : 64;
    void *ri = NULL, *ro = NULL, *rr = NULL;
    if (posix_memalign(&ri, 64, (size_t)nv * VDBL * sizeof(double)) ||
        posix_memalign(&ro, 64, (size_t)nv * VDBL * sizeof(double)) ||
        posix_memalign(&rr, 64, (size_t)nv * VDBL * sizeof(double))) {
        free(ri); free(ro);                 /* keep the safe default */
        p->pw = cands[0].pw; p->mode = cands[0].mode;
        return p;
    }
    double *tin = ri, *tout = ro, *ref = rr;
    unsigned long long s = 0x9E3779B97F4A7C15ull;
    for (size_t i = 0; i < (size_t)nv * VDBL; ++i) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        tin[i] = (double)(long long)(s >> 11) * 0x1p-53;
    }

    run_vols(cands[0].pw, cands[0].mode, p->S, tin, ref, nv);

    int    ok[6];
    double tc[6];
    for (int c = 0; c < nc; ++c) {
        run_vols(cands[c].pw, cands[c].mode, p->S, tin, tout, nv);
        double num = 0.0, den = 0.0;
        for (size_t i = 0; i < (size_t)nv * VDBL; ++i) {
            double d = tout[i] - ref[i];
            num += d * d; den += ref[i] * ref[i];
        }
        ok[c] = (num <= den * 1e-26);       /* rel L2 < 1e-13 vs reference */
        tc[c] = 1e300;
    }
    const int R = (nv >= 8) ? 1 : (nv >= 2 ? 3 : 8);
    for (int round = 0; round < 3; ++round)
        for (int c = 0; c < nc; ++c) {
            if (!ok[c]) continue;
            double t0 = now_s();
            for (int r = 0; r < R; ++r)
                run_vols(cands[c].pw, cands[c].mode, p->S, tin, tout, nv);
            double t = (now_s() - t0) / R;
            if (t < tc[c]) tc[c] = t;
        }
    int best = 0;
    for (int c = 1; c < nc; ++c) if (ok[c] && tc[c] < tc[best]) best = c;
    if (ok[best]) { p->pw = cands[best].pw; p->mode = cands[best].mode; }

#ifdef FFT36_LOUD
    if (1) {
#else
    if (getenv("FFT36_VERBOSE")) {
#endif
        for (int c = 0; c < nc; ++c)
            fprintf(stderr, "L36_pfa tuner: pw=%d mode=%d  %s  %.1f us/vol\n",
                    cands[c].pw, cands[c].mode, ok[c] ? "ok " : "BAD",
                    ok[c] ? tc[c] * 1e6 / nv : 0.0);
        fprintf(stderr, "L36_pfa tuner: chose pw=%d mode=%d (nv=%d)\n",
                p->pw, p->mode, nv);
    }
    free(ri); free(ro); free(rr);
    return p;
}

void fft3d_destroy(fft3d_plan *p)
{
    if (!p) return;
    free(p->rawS); free(p);
}

void fft3d_execute(fft3d_plan *plan, const double _Complex *in, double _Complex *out)
{
    run_vols(plan->pw, plan->mode, plan->S,
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

/* phase 1: per x-plane, z transform (transposed lanes) then y transform.
 * Reads `in` once, sequentially; writes mid[x][ky][kz]. */
static void FN(phase1)(const double *restrict in, double *restrict mid)
{
    vec pl[L * NVR];                       /* plane [y][kz], 20.25 KB */
    for (int x = 0; x < L; ++x) {
        const double *px = in  + (size_t)x * (2 * LSQ);
        double       *mx = mid + (size_t)x * (2 * LSQ);

        for (int yb = 0; yb < L; yb += PW) {
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
#define LD2(n)    pl[(size_t)(n) * NVR + zb]
#define ST2(k, v) (*(vec *)(mx + ((size_t)(k) * L + (size_t)zb * PW) * 2) = (v))
            PFA36(LD2, ST2);
#undef LD2
#undef ST2
        }
    }
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

/* phase 2: x transform, 36 sequential source streams of stride 20736 B.
 * Safe when mid == out: the codelet reads all 36 inputs before its first
 * store, so results go straight from registers to `out` with no staging. */
static void FN(phase2)(const double *mid, double *out, int nt)
{
    if (!nt) {
        for (int y = 0; y < L; ++y)
            for (int zb = 0; zb < NVR; ++zb) {
                const size_t o = ((size_t)y * L + (size_t)zb * PW) * 2;
                const double *s_ = mid + o;
                double       *d_ = out + o;
                PF36(s_);
#define ST3D(k, v) (*(vec *)(d_ + (size_t)(k) * (2 * LSQ)) = (v))
                PFA36(LD3, ST3D);
#undef ST3D
            }
    } else {
#if PW == 2
        /* pair two z-blocks so every NT write completes a 64-byte line */
        for (int y = 0; y < L; ++y)
            for (int zb = 0; zb < NVR; zb += 2) {
                const size_t o = ((size_t)y * L + (size_t)zb * PW) * 2;
                vec Wa[36], Wb[36];
                TILE(mid + o,     Wa);
                TILE(mid + o + 4, Wb);
                double *dst = out + o;
                _Pragma("GCC unroll 36")
                for (int k = 0; k < 36; ++k) {
                    STREAM_ST(dst + (size_t)k * (2 * LSQ),     Wa[k]);
                    STREAM_ST(dst + (size_t)k * (2 * LSQ) + 4, Wb[k]);
                }
            }
#else
        for (int y = 0; y < L; ++y)
            for (int zb = 0; zb < NVR; ++zb) {
                const size_t o = ((size_t)y * L + (size_t)zb * PW) * 2;
                const double *s_ = mid + o;
                double       *d_ = out + o;
                PF36(s_);
#define ST3N(k, v) STREAM_ST(d_ + (size_t)(k) * (2 * LSQ), (v))
                PFA36(LD3, ST3N);
#undef ST3N
            }
#endif
    }
}

#undef TILE
#undef LD3
#undef ST3
#undef PF36
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
