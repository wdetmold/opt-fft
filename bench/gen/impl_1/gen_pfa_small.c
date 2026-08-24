/* gen_pfa_small.c -- PFA of coprime pairs, small: L = 10, 12, 15, 20.
 *
 * ROUND gen_r1, WHAT SHIPS (replacing the dense O(L^4) stub):
 *
 * ALGORITHM: row-column 3D transform, every line transform done by the
 * Good-Thomas / prime-factor algorithm over the coprime split
 *     10 = 2*5,  12 = 4*3,  15 = 3*5,  20 = 4*5,
 * so there is NO twiddle stage anywhere -- only the exact-constant DFT2 /
 * DFT3 / DFT4 / DFT5 modules, with the Ruritanian input map and CRT output
 * map baked into the (fully unrolled) codelet's load/store indices.  This is
 * the seed lesson from the ice campaign's L6_pfa carried to four new sizes.
 *
 * LAYOUT (the main lever, per LITERATURE.md sect. 08: layout beats kernels):
 * split-complex SoA with 8 VOLUMES IN THE ZMM LANES ("q4" structure BORROWED
 * from ice L45_pfa ice_r7, transitively rival v6_5a869e40's run4_: "SIMD
 * lanes = volumes, no transposes, no tail lanes", widened to 8 lanes for the
 * two Ice Lake 512-bit FMA pipes).  Point p of volumes v..v+7 is one zmm of
 * reals + one of imags; every codelet access in ALL THREE axis passes is a
 * full-width contiguous 64B load/store, and split complex makes every
 * x(+-i) and every module constant a pure FMA -- zero shuffle-class
 * instructions inside the transform.  Pack/unpack (8x8 in-register
 * transposes) happen once per execute, or once per CHAIN.
 *
 * CHAIN: fft3d_chain is exported.  The state stays in the SoA arena for the
 * whole m-step graded chain (pack x0 and c once, unpack once at the end);
 * the map z/(1+|z|) is fused onto the z-pass stores of each step while the
 * pencil is L1-hot (ice L45_pfa's "map at the stores of the last axis"
 * lesson).  The map s = 1/(1+sqrt(re^2+im^2)) avoids the divider unit:
 * rsqrt14 + 2 Newton steps for sqrt, rcp14 + 2 Newton steps for the
 * reciprocal (~1e-15 rel/point vs the 1.5e-14/step contract; measured
 * two-step gate 1.3e-15).  This alone was -24..-32% of the whole chain
 * step: vsqrtpd+vdivpd zmm are ~30 serial divider cycles per vector.
 *
 * B % 8 REMAINDERS AND B = 1: a per-volume split-complex path reusing the
 * same pencil codelets -- lanes are 8 consecutive inner points (x pass:
 * flat (y,z) index; y pass: z chunks; z pass: y lanes via in-register 8x8
 * transposes), overlapped chunks for the non-multiple-of-8 tails (passes are
 * out-of-place ping-pong, so recomputing an overlap is idempotent).
 *
 * Correctness: single call rel L2 ~1e-16 vs numpy (exact 20-digit module
 * constants, no trig recurrences); the chain map is arithmetically the
 * driver fallback's own formula.
 */
#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#if defined(__AVX512F__)
#include <immintrin.h>
#endif

#include "../fft3d_api.h"

const char *fft3d_name(void) { return "gen_pfa_small"; }
const char *fft3d_description(void)
{
    return "PFA coprime (10=2x5,12=4x3,15=3x5,20=4x5), no twiddles; split-complex "
           "SoA 8 vols/zmm, 3 in-place passes, chain owns state w/ map fused in z-pass; "
           "B%8 via per-volume split path";
}
int fft3d_supports(int L) { return L == 10 || L == 12 || L == 15 || L == 20; }

/* ------------------------------------------------------------------ SIMD */

typedef double v8 __attribute__((vector_size(64)));
typedef long long v8i __attribute__((vector_size(64)));

static inline v8 vload(const double *p) { v8 v; __builtin_memcpy(&v, p, 64); return v; }
static inline void vstore(double *p, v8 v) { __builtin_memcpy(p, &v, 64); }

#define SHUF(a, b, ...) __builtin_shuffle((a), (b), (v8i){__VA_ARGS__})

/* 8x8 doubles transpose, in place on m[0..7]: m'[j] = old column j. */
static inline void tr8(v8 *m)
{
    v8 u0 = SHUF(m[0], m[1], 0, 8, 2, 10, 4, 12, 6, 14);
    v8 u1 = SHUF(m[0], m[1], 1, 9, 3, 11, 5, 13, 7, 15);
    v8 u2 = SHUF(m[2], m[3], 0, 8, 2, 10, 4, 12, 6, 14);
    v8 u3 = SHUF(m[2], m[3], 1, 9, 3, 11, 5, 13, 7, 15);
    v8 u4 = SHUF(m[4], m[5], 0, 8, 2, 10, 4, 12, 6, 14);
    v8 u5 = SHUF(m[4], m[5], 1, 9, 3, 11, 5, 13, 7, 15);
    v8 u6 = SHUF(m[6], m[7], 0, 8, 2, 10, 4, 12, 6, 14);
    v8 u7 = SHUF(m[6], m[7], 1, 9, 3, 11, 5, 13, 7, 15);
    v8 w0 = SHUF(u0, u2, 0, 1, 8, 9, 4, 5, 12, 13);
    v8 w2 = SHUF(u0, u2, 2, 3, 10, 11, 6, 7, 14, 15);
    v8 w1 = SHUF(u1, u3, 0, 1, 8, 9, 4, 5, 12, 13);
    v8 w3 = SHUF(u1, u3, 2, 3, 10, 11, 6, 7, 14, 15);
    v8 w4 = SHUF(u4, u6, 0, 1, 8, 9, 4, 5, 12, 13);
    v8 w6 = SHUF(u4, u6, 2, 3, 10, 11, 6, 7, 14, 15);
    v8 w5 = SHUF(u5, u7, 0, 1, 8, 9, 4, 5, 12, 13);
    v8 w7 = SHUF(u5, u7, 2, 3, 10, 11, 6, 7, 14, 15);
    m[0] = SHUF(w0, w4, 0, 1, 2, 3, 8, 9, 10, 11);
    m[4] = SHUF(w0, w4, 4, 5, 6, 7, 12, 13, 14, 15);
    m[1] = SHUF(w1, w5, 0, 1, 2, 3, 8, 9, 10, 11);
    m[5] = SHUF(w1, w5, 4, 5, 6, 7, 12, 13, 14, 15);
    m[2] = SHUF(w2, w6, 0, 1, 2, 3, 8, 9, 10, 11);
    m[6] = SHUF(w2, w6, 4, 5, 6, 7, 12, 13, 14, 15);
    m[3] = SHUF(w3, w7, 0, 1, 2, 3, 8, 9, 10, 11);
    m[7] = SHUF(w3, w7, 4, 5, 6, 7, 12, 13, 14, 15);
}

/* ------------------------------------------------- exact module constants */

#define K3  0.86602540378443864676   /* sin(pi/3)   */
#define C51 0.30901699437494742410   /* cos(2pi/5)  */
#define C52 (-0.80901699437494742410) /* cos(4pi/5) */
#define S51 0.95105651629515357212   /* sin(2pi/5)  */
#define S52 0.58778525229247312917   /* sin(4pi/5)  */

/* Forward DFT modules on split vectors, in place on the named registers.
 * All x(+-i) are free (component swap + sign folded into the add). */

#define M_DFT2(r0, i0, r1, i1) do {                                   \
        v8 tr_ = r0 - r1, ti_ = i0 - i1;                              \
        r0 += r1; i0 += i1; r1 = tr_; i1 = ti_;                       \
    } while (0)

#define M_DFT3(r0, i0, r1, i1, r2, i2) do {                           \
        v8 ar_ = r1 + r2, ai_ = i1 + i2;                              \
        v8 dr_ = r1 - r2, di_ = i1 - i2;                              \
        v8 er_ = r0 - 0.5 * ar_, ei_ = i0 - 0.5 * ai_;                \
        r0 += ar_; i0 += ai_;                                         \
        r1 = er_ + K3 * di_; i1 = ei_ - K3 * dr_;                     \
        r2 = er_ - K3 * di_; i2 = ei_ + K3 * dr_;                     \
    } while (0)

#define M_DFT4(r0, i0, r1, i1, r2, i2, r3, i3) do {                   \
        v8 t0r_ = r0 + r2, t0i_ = i0 + i2;                            \
        v8 t1r_ = r0 - r2, t1i_ = i0 - i2;                            \
        v8 t2r_ = r1 + r3, t2i_ = i1 + i3;                            \
        v8 t3r_ = r1 - r3, t3i_ = i1 - i3;                            \
        r0 = t0r_ + t2r_; i0 = t0i_ + t2i_;                           \
        r2 = t0r_ - t2r_; i2 = t0i_ - t2i_;                           \
        r1 = t1r_ + t3i_; i1 = t1i_ - t3r_;                           \
        r3 = t1r_ - t3i_; i3 = t1i_ + t3r_;                           \
    } while (0)

#define M_DFT5(r0, i0, r1, i1, r2, i2, r3, i3, r4, i4) do {           \
        v8 ar_ = r1 + r4, ai_ = i1 + i4;                              \
        v8 cr_ = r1 - r4, ci_ = i1 - i4;                              \
        v8 br_ = r2 + r3, bi_ = i2 + i3;                              \
        v8 dr_ = r2 - r3, di_ = i2 - i3;                              \
        v8 e1r_ = r0 + C51 * ar_ + C52 * br_;                         \
        v8 e1i_ = i0 + C51 * ai_ + C52 * bi_;                         \
        v8 e2r_ = r0 + C52 * ar_ + C51 * br_;                         \
        v8 e2i_ = i0 + C52 * ai_ + C51 * bi_;                         \
        r0 += ar_ + br_; i0 += ai_ + bi_;                             \
        v8 o1r_ = S51 * cr_ + S52 * dr_, o1i_ = S51 * ci_ + S52 * di_;\
        v8 o2r_ = S52 * cr_ - S51 * dr_, o2i_ = S52 * ci_ - S51 * di_;\
        r1 = e1r_ + o1i_; i1 = e1i_ - o1r_;                           \
        r4 = e1r_ - o1i_; i4 = e1i_ + o1r_;                           \
        r2 = e2r_ + o2i_; i2 = e2i_ - o2r_;                           \
        r3 = e2r_ - o2i_; i3 = e2i_ + o2r_;                           \
    } while (0)

/* ------------------------------------------------------------- PFA maps
 * L = n1*n2 coprime.  in(j1,j2) = (n2*j1 + n1*j2) mod L (Ruritanian);
 * out(k1,k2) = the CRT solution of k = k1 mod n1, k = k2 mod n2.  With
 * these two maps the length-L DFT is exactly DFT_n1 (x) DFT_n2, no
 * twiddles.  Tables verified at create() time against the definition. */

static const int IN10[2][5]  = {{0, 2, 4, 6, 8}, {5, 7, 9, 1, 3}};
static const int OUT10[2][5] = {{0, 6, 2, 8, 4}, {5, 1, 7, 3, 9}};

static const int IN12[4][3]  = {{0, 4, 8}, {3, 7, 11}, {6, 10, 2}, {9, 1, 5}};
static const int OUT12[4][3] = {{0, 4, 8}, {9, 1, 5}, {6, 10, 2}, {3, 7, 11}};

static const int IN15[3][5]  = {{0, 3, 6, 9, 12}, {5, 8, 11, 14, 2}, {10, 13, 1, 4, 7}};
static const int OUT15[3][5] = {{0, 6, 12, 3, 9}, {10, 1, 7, 13, 4}, {5, 11, 2, 8, 14}};

static const int IN20[4][5]  = {{0, 4, 8, 12, 16}, {5, 9, 13, 17, 1},
                                {10, 14, 18, 2, 6}, {15, 19, 3, 7, 11}};
static const int OUT20[4][5] = {{0, 16, 12, 8, 4}, {5, 1, 17, 13, 9},
                                {10, 6, 2, 18, 14}, {15, 11, 7, 3, 19}};

/* --------------------------------------------------------------- pencils
 * One length-L line transform on 8 lanes.  s = stride in doubles between
 * consecutive line points.  sr/dr may be the same buffer (stage 1 fully
 * consumes the input into t[] before stage 2 writes). */

static inline void pencil10(const double *sr, const double *si,
                            double *dr, double *di, ptrdiff_t s)
{
    v8 tr[10], ti[10];
    for (int j2 = 0; j2 < 5; ++j2) {
        v8 r0 = vload(sr + IN10[0][j2] * s), i0 = vload(si + IN10[0][j2] * s);
        v8 r1 = vload(sr + IN10[1][j2] * s), i1 = vload(si + IN10[1][j2] * s);
        M_DFT2(r0, i0, r1, i1);
        tr[j2] = r0; ti[j2] = i0; tr[5 + j2] = r1; ti[5 + j2] = i1;
    }
    for (int k1 = 0; k1 < 2; ++k1) {
        v8 r0 = tr[5 * k1 + 0], i0 = ti[5 * k1 + 0];
        v8 r1 = tr[5 * k1 + 1], i1 = ti[5 * k1 + 1];
        v8 r2 = tr[5 * k1 + 2], i2 = ti[5 * k1 + 2];
        v8 r3 = tr[5 * k1 + 3], i3 = ti[5 * k1 + 3];
        v8 r4 = tr[5 * k1 + 4], i4 = ti[5 * k1 + 4];
        M_DFT5(r0, i0, r1, i1, r2, i2, r3, i3, r4, i4);
        vstore(dr + OUT10[k1][0] * s, r0); vstore(di + OUT10[k1][0] * s, i0);
        vstore(dr + OUT10[k1][1] * s, r1); vstore(di + OUT10[k1][1] * s, i1);
        vstore(dr + OUT10[k1][2] * s, r2); vstore(di + OUT10[k1][2] * s, i2);
        vstore(dr + OUT10[k1][3] * s, r3); vstore(di + OUT10[k1][3] * s, i3);
        vstore(dr + OUT10[k1][4] * s, r4); vstore(di + OUT10[k1][4] * s, i4);
    }
}

static inline void pencil12(const double *sr, const double *si,
                            double *dr, double *di, ptrdiff_t s)
{
    v8 tr[12], ti[12];
    for (int j2 = 0; j2 < 3; ++j2) {
        v8 r0 = vload(sr + IN12[0][j2] * s), i0 = vload(si + IN12[0][j2] * s);
        v8 r1 = vload(sr + IN12[1][j2] * s), i1 = vload(si + IN12[1][j2] * s);
        v8 r2 = vload(sr + IN12[2][j2] * s), i2 = vload(si + IN12[2][j2] * s);
        v8 r3 = vload(sr + IN12[3][j2] * s), i3 = vload(si + IN12[3][j2] * s);
        M_DFT4(r0, i0, r1, i1, r2, i2, r3, i3);
        tr[j2] = r0; ti[j2] = i0; tr[3 + j2] = r1; ti[3 + j2] = i1;
        tr[6 + j2] = r2; ti[6 + j2] = i2; tr[9 + j2] = r3; ti[9 + j2] = i3;
    }
    for (int k1 = 0; k1 < 4; ++k1) {
        v8 r0 = tr[3 * k1 + 0], i0 = ti[3 * k1 + 0];
        v8 r1 = tr[3 * k1 + 1], i1 = ti[3 * k1 + 1];
        v8 r2 = tr[3 * k1 + 2], i2 = ti[3 * k1 + 2];
        M_DFT3(r0, i0, r1, i1, r2, i2);
        vstore(dr + OUT12[k1][0] * s, r0); vstore(di + OUT12[k1][0] * s, i0);
        vstore(dr + OUT12[k1][1] * s, r1); vstore(di + OUT12[k1][1] * s, i1);
        vstore(dr + OUT12[k1][2] * s, r2); vstore(di + OUT12[k1][2] * s, i2);
    }
}

static inline void pencil15(const double *sr, const double *si,
                            double *dr, double *di, ptrdiff_t s)
{
    v8 tr[15], ti[15];
    for (int j2 = 0; j2 < 5; ++j2) {
        v8 r0 = vload(sr + IN15[0][j2] * s), i0 = vload(si + IN15[0][j2] * s);
        v8 r1 = vload(sr + IN15[1][j2] * s), i1 = vload(si + IN15[1][j2] * s);
        v8 r2 = vload(sr + IN15[2][j2] * s), i2 = vload(si + IN15[2][j2] * s);
        M_DFT3(r0, i0, r1, i1, r2, i2);
        tr[j2] = r0; ti[j2] = i0; tr[5 + j2] = r1; ti[5 + j2] = i1;
        tr[10 + j2] = r2; ti[10 + j2] = i2;
    }
    for (int k1 = 0; k1 < 3; ++k1) {
        v8 r0 = tr[5 * k1 + 0], i0 = ti[5 * k1 + 0];
        v8 r1 = tr[5 * k1 + 1], i1 = ti[5 * k1 + 1];
        v8 r2 = tr[5 * k1 + 2], i2 = ti[5 * k1 + 2];
        v8 r3 = tr[5 * k1 + 3], i3 = ti[5 * k1 + 3];
        v8 r4 = tr[5 * k1 + 4], i4 = ti[5 * k1 + 4];
        M_DFT5(r0, i0, r1, i1, r2, i2, r3, i3, r4, i4);
        vstore(dr + OUT15[k1][0] * s, r0); vstore(di + OUT15[k1][0] * s, i0);
        vstore(dr + OUT15[k1][1] * s, r1); vstore(di + OUT15[k1][1] * s, i1);
        vstore(dr + OUT15[k1][2] * s, r2); vstore(di + OUT15[k1][2] * s, i2);
        vstore(dr + OUT15[k1][3] * s, r3); vstore(di + OUT15[k1][3] * s, i3);
        vstore(dr + OUT15[k1][4] * s, r4); vstore(di + OUT15[k1][4] * s, i4);
    }
}

static inline void pencil20(const double *sr, const double *si,
                            double *dr, double *di, ptrdiff_t s)
{
    v8 tr[20], ti[20];
    for (int j2 = 0; j2 < 5; ++j2) {
        v8 r0 = vload(sr + IN20[0][j2] * s), i0 = vload(si + IN20[0][j2] * s);
        v8 r1 = vload(sr + IN20[1][j2] * s), i1 = vload(si + IN20[1][j2] * s);
        v8 r2 = vload(sr + IN20[2][j2] * s), i2 = vload(si + IN20[2][j2] * s);
        v8 r3 = vload(sr + IN20[3][j2] * s), i3 = vload(si + IN20[3][j2] * s);
        M_DFT4(r0, i0, r1, i1, r2, i2, r3, i3);
        tr[j2] = r0; ti[j2] = i0; tr[5 + j2] = r1; ti[5 + j2] = i1;
        tr[10 + j2] = r2; ti[10 + j2] = i2; tr[15 + j2] = r3; ti[15 + j2] = i3;
    }
    for (int k1 = 0; k1 < 4; ++k1) {
        v8 r0 = tr[5 * k1 + 0], i0 = ti[5 * k1 + 0];
        v8 r1 = tr[5 * k1 + 1], i1 = ti[5 * k1 + 1];
        v8 r2 = tr[5 * k1 + 2], i2 = ti[5 * k1 + 2];
        v8 r3 = tr[5 * k1 + 3], i3 = ti[5 * k1 + 3];
        v8 r4 = tr[5 * k1 + 4], i4 = ti[5 * k1 + 4];
        M_DFT5(r0, i0, r1, i1, r2, i2, r3, i3, r4, i4);
        vstore(dr + OUT20[k1][0] * s, r0); vstore(di + OUT20[k1][0] * s, i0);
        vstore(dr + OUT20[k1][1] * s, r1); vstore(di + OUT20[k1][1] * s, i1);
        vstore(dr + OUT20[k1][2] * s, r2); vstore(di + OUT20[k1][2] * s, i2);
        vstore(dr + OUT20[k1][3] * s, r3); vstore(di + OUT20[k1][3] * s, i3);
        vstore(dr + OUT20[k1][4] * s, r4); vstore(di + OUT20[k1][4] * s, i4);
    }
}

/* ------------------------------------------------------------- the map
 * Exactly the driver fallback's arithmetic: s = 1/(1+sqrt(re^2+im^2)).
 * Contiguous split spans; n need not be a multiple of 8. */

static inline void map_span(double *zr, double *zi,
                            const double *cr, const double *ci, ptrdiff_t n)
{
    ptrdiff_t i = 0;
#if defined(__AVX512F__)
    /* 1/(1+sqrt(m)) without the divider unit: rsqrt14 + 2 Newton steps for
     * sqrt (rel err ~4 ulp, quadratic convergence from 2^-14), rcp14 + 2
     * Newton steps for the reciprocal.  ~1e-15 rel per point against the
     * 1.5e-14/step precision contract; the vsqrtpd+vdivpd form it replaces
     * was ~30 divider cycles per vector and dominated the whole chain step.
     * max(m, DBL_MIN) keeps rsqrt(0) from making 0*inf = NaN at z = 0
     * (output is 0 either way, but sc must stay finite). */
    const __m512d one = _mm512_set1_pd(1.0), two = _mm512_set1_pd(2.0);
    const __m512d half = _mm512_set1_pd(0.5), three = _mm512_set1_pd(3.0);
    const __m512d tiny = _mm512_set1_pd(2.2250738585072014e-308);
    for (; i + 8 <= n; i += 8) {
        __m512d a = _mm512_add_pd(_mm512_loadu_pd(zr + i), _mm512_loadu_pd(cr + i));
        __m512d b = _mm512_add_pd(_mm512_loadu_pd(zi + i), _mm512_loadu_pd(ci + i));
        __m512d m = _mm512_fmadd_pd(a, a, _mm512_mul_pd(b, b));
        m = _mm512_max_pd(m, tiny);
        __m512d r = _mm512_rsqrt14_pd(m);
        __m512d t = _mm512_mul_pd(m, r);
        r = _mm512_mul_pd(_mm512_mul_pd(half, r), _mm512_fnmadd_pd(t, r, three));
        t = _mm512_mul_pd(m, r);
        r = _mm512_mul_pd(_mm512_mul_pd(half, r), _mm512_fnmadd_pd(t, r, three));
        __m512d d = _mm512_add_pd(one, _mm512_mul_pd(m, r));
        __m512d y = _mm512_rcp14_pd(d);
        y = _mm512_mul_pd(y, _mm512_fnmadd_pd(d, y, two));
        y = _mm512_mul_pd(y, _mm512_fnmadd_pd(d, y, two));
        _mm512_storeu_pd(zr + i, _mm512_mul_pd(a, y));
        _mm512_storeu_pd(zi + i, _mm512_mul_pd(b, y));
    }
#endif
    for (; i < n; ++i) {
        double a = zr[i] + cr[i], b = zi[i] + ci[i];
        double sc = 1.0 / (1.0 + sqrt(a * a + b * b));
        zr[i] = a * sc;
        zi[i] = b * sc;
    }
}

/* ------------------------------------------- SoA-8 passes (8 volumes/zmm)
 * State: qr/qi of L^3 v8s, point index major, volume lanes minor.
 * All three passes in place; pencil strides are compile-time constants. */

#define DEF_SOA(L)                                                              \
static void soa_fft_##L(double *qr, double *qi)                                 \
{                                                                               \
    for (int b = 0; b < (L) * (L); ++b)                                         \
        pencil##L(qr + b * 8, qi + b * 8, qr + b * 8, qi + b * 8,               \
                  (ptrdiff_t)8 * (L) * (L));                                    \
    for (int x = 0; x < (L); ++x)                                               \
        for (int z = 0; z < (L); ++z) {                                         \
            ptrdiff_t o = ((ptrdiff_t)x * (L) * (L) + z) * 8;                   \
            pencil##L(qr + o, qi + o, qr + o, qi + o, (ptrdiff_t)8 * (L));      \
        }                                                                       \
    for (int r = 0; r < (L) * (L); ++r) {                                       \
        ptrdiff_t o = (ptrdiff_t)r * (L) * 8;                                   \
        pencil##L(qr + o, qi + o, qr + o, qi + o, 8);                           \
    }                                                                           \
}                                                                               \
static void soa_step_##L(double *qr, double *qi,                                \
                         const double *cqr, const double *cqi)                  \
{                                                                               \
    for (int b = 0; b < (L) * (L); ++b)                                         \
        pencil##L(qr + b * 8, qi + b * 8, qr + b * 8, qi + b * 8,               \
                  (ptrdiff_t)8 * (L) * (L));                                    \
    for (int x = 0; x < (L); ++x)                                               \
        for (int z = 0; z < (L); ++z) {                                         \
            ptrdiff_t o = ((ptrdiff_t)x * (L) * (L) + z) * 8;                   \
            pencil##L(qr + o, qi + o, qr + o, qi + o, (ptrdiff_t)8 * (L));      \
        }                                                                       \
    for (int r = 0; r < (L) * (L); ++r) {                                       \
        ptrdiff_t o = (ptrdiff_t)r * (L) * 8;                                   \
        pencil##L(qr + o, qi + o, qr + o, qi + o, 8);                           \
        map_span(qr + o, qi + o, cqr + o, cqi + o, (ptrdiff_t)8 * (L));         \
    }                                                                           \
}

DEF_SOA(10)
DEF_SOA(12)
DEF_SOA(15)
DEF_SOA(20)

/* --------------------------- per-volume split-complex path (B=1, B%8)
 * Ping-pong passes S->D, D->S, S->D; lanes are 8 consecutive inner points,
 * tails handled by overlapped (idempotent, out-of-place) chunks.  The z
 * pass turns lanes into y via in-register 8x8 transposes. */

#define DEF_SPLIT(L)                                                            \
static void split_fft_##L(double *Sr, double *Si, double *Dr, double *Di)       \
{                                                                               \
    /* x pass: lanes = flat (y,z) index, stride L^2 */                          \
    for (int b = 0; b < (L) * (L); b += 8) {                                    \
        int o = (b + 8 <= (L) * (L)) ? b : (L) * (L) - 8;                       \
        pencil##L(Sr + o, Si + o, Dr + o, Di + o, (ptrdiff_t)(L) * (L));        \
    }                                                                           \
    /* y pass: per x slab, lanes = z chunk, stride L */                         \
    for (int x = 0; x < (L); ++x) {                                             \
        ptrdiff_t xo = (ptrdiff_t)x * (L) * (L);                                \
        for (int z = 0; z < (L); z += 8) {                                      \
            int o = (z + 8 <= (L)) ? z : (L) - 8;                               \
            pencil##L(Dr + xo + o, Di + xo + o, Sr + xo + o, Si + xo + o,       \
                      (ptrdiff_t)(L));                                          \
        }                                                                       \
    }                                                                           \
    /* z pass: per x slab, lanes = y via 8x8 transposes */                      \
    for (int x = 0; x < (L); ++x) {                                             \
        ptrdiff_t xo = (ptrdiff_t)x * (L) * (L);                                \
        for (int y = 0; y < (L); y += 8) {                                      \
            int yo = (y + 8 <= (L)) ? y : (L) - 8;                              \
            v8 pzr[L], pzi[L], por[L], poi[L];                                  \
            for (int z = 0; z < (L); z += 8) {                                  \
                int zo = (z + 8 <= (L)) ? z : (L) - 8;                          \
                v8 blk[8];                                                      \
                for (int q = 0; q < 8; ++q)                                     \
                    blk[q] = vload(Sr + xo + (ptrdiff_t)(yo + q) * (L) + zo);   \
                tr8(blk);                                                       \
                for (int q = 0; q < 8; ++q) pzr[zo + q] = blk[q];               \
                for (int q = 0; q < 8; ++q)                                     \
                    blk[q] = vload(Si + xo + (ptrdiff_t)(yo + q) * (L) + zo);   \
                tr8(blk);                                                       \
                for (int q = 0; q < 8; ++q) pzi[zo + q] = blk[q];               \
            }                                                                   \
            pencil##L((const double *)pzr, (const double *)pzi,                 \
                      (double *)por, (double *)poi, 8);                         \
            for (int z = 0; z < (L); z += 8) {                                  \
                int zo = (z + 8 <= (L)) ? z : (L) - 8;                          \
                v8 blk[8];                                                      \
                for (int q = 0; q < 8; ++q) blk[q] = por[zo + q];               \
                tr8(blk);                                                       \
                for (int q = 0; q < 8; ++q)                                     \
                    vstore(Dr + xo + (ptrdiff_t)(yo + q) * (L) + zo, blk[q]);   \
                for (int q = 0; q < 8; ++q) blk[q] = poi[zo + q];               \
                tr8(blk);                                                       \
                for (int q = 0; q < 8; ++q)                                     \
                    vstore(Di + xo + (ptrdiff_t)(yo + q) * (L) + zo, blk[q]);   \
            }                                                                   \
        }                                                                       \
    }                                                                           \
}

DEF_SPLIT(10)
DEF_SPLIT(12)
DEF_SPLIT(15)
DEF_SPLIT(20)

/* ----------------------------------------------------- packing helpers */

/* 8 interleaved volumes (stride vol complex) -> SoA-8 split arena. */
static void pack8(const double _Complex *in, size_t vol, double *qr, double *qi)
{
    const double *base = (const double *)in;
    size_t p = 0;
    for (; p + 4 <= vol; p += 4) {
        v8 rows[8];
        for (int k = 0; k < 8; ++k)
            rows[k] = vload(base + 2 * ((size_t)k * vol + p));
        tr8(rows);
        vstore(qr + p * 8, rows[0]);       vstore(qi + p * 8, rows[1]);
        vstore(qr + (p + 1) * 8, rows[2]); vstore(qi + (p + 1) * 8, rows[3]);
        vstore(qr + (p + 2) * 8, rows[4]); vstore(qi + (p + 2) * 8, rows[5]);
        vstore(qr + (p + 3) * 8, rows[6]); vstore(qi + (p + 3) * 8, rows[7]);
    }
    for (; p < vol; ++p)
        for (int k = 0; k < 8; ++k) {
            qr[p * 8 + k] = base[2 * ((size_t)k * vol + p)];
            qi[p * 8 + k] = base[2 * ((size_t)k * vol + p) + 1];
        }
}

static void unpack8(const double *qr, const double *qi,
                    double _Complex *out, size_t vol)
{
    double *base = (double *)out;
    size_t p = 0;
    for (; p + 4 <= vol; p += 4) {
        v8 rows[8];
        rows[0] = vload(qr + p * 8);       rows[1] = vload(qi + p * 8);
        rows[2] = vload(qr + (p + 1) * 8); rows[3] = vload(qi + (p + 1) * 8);
        rows[4] = vload(qr + (p + 2) * 8); rows[5] = vload(qi + (p + 2) * 8);
        rows[6] = vload(qr + (p + 3) * 8); rows[7] = vload(qi + (p + 3) * 8);
        tr8(rows);
        for (int k = 0; k < 8; ++k)
            vstore(base + 2 * ((size_t)k * vol + p), rows[k]);
    }
    for (; p < vol; ++p)
        for (int k = 0; k < 8; ++k) {
            base[2 * ((size_t)k * vol + p)] = qr[p * 8 + k];
            base[2 * ((size_t)k * vol + p) + 1] = qi[p * 8 + k];
        }
}

static void deinterleave(const double _Complex *x, double *r, double *i, size_t n)
{
    const double *p = (const double *)x;
    for (size_t k = 0; k < n; ++k) { r[k] = p[2 * k]; i[k] = p[2 * k + 1]; }
}

static void interleave(const double *r, const double *i, double _Complex *x, size_t n)
{
    double *p = (double *)x;
    for (size_t k = 0; k < n; ++k) { p[2 * k] = r[k]; p[2 * k + 1] = i[k]; }
}

/* ------------------------------------------------------------------ plan */

struct fft3d_plan {
    int L, batch;
    size_t vol;
    double *arena;
    double *qr, *qi, *cqr, *cqi;         /* SoA-8: vol*8 doubles each      */
    double *Sr, *Si, *Dr, *Di, *Cr, *Ci; /* split per-volume: vol each     */
    void (*soa_fft)(double *, double *);
    void (*soa_step)(double *, double *, const double *, const double *);
    void (*split_fft)(double *, double *, double *, double *);
};

fft3d_plan *fft3d_create(int L, int batch)
{
    if (!fft3d_supports(L) || batch < 1) return NULL;
    fft3d_plan *p = calloc(1, sizeof *p);
    if (!p) return NULL;
    p->L = L;
    p->batch = batch;
    p->vol = (size_t)L * L * L;

    /* +8 doubles (one cache line) of stagger between components: at L=12
     * and L=20 the un-padded component stride is an exact multiple of 4096,
     * so every paired re/im access would collide in the low 12 address bits
     * (4K store-to-load aliasing).  ducc0's odd-line-stagger rule. */
    size_t soa = p->vol * 8 + 8;           /* one SoA component            */
    size_t svol = p->vol + 8;              /* one split component          */
    size_t total = 4 * soa + 6 * svol;     /* qr qi cqr cqi + 6 split bufs */
    if (posix_memalign((void **)&p->arena, 64, total * sizeof(double)) != 0) {
        free(p);
        return NULL;
    }
    memset(p->arena, 0, total * sizeof(double));
    p->qr = p->arena;
    p->qi = p->qr + soa;
    p->cqr = p->qi + soa;
    p->cqi = p->cqr + soa;
    p->Sr = p->cqi + soa;
    p->Si = p->Sr + svol;
    p->Dr = p->Si + svol;
    p->Di = p->Dr + svol;
    p->Cr = p->Di + svol;
    p->Ci = p->Cr + svol;

    switch (L) {
    case 10: p->soa_fft = soa_fft_10; p->soa_step = soa_step_10; p->split_fft = split_fft_10; break;
    case 12: p->soa_fft = soa_fft_12; p->soa_step = soa_step_12; p->split_fft = split_fft_12; break;
    case 15: p->soa_fft = soa_fft_15; p->soa_step = soa_step_15; p->split_fft = split_fft_15; break;
    case 20: p->soa_fft = soa_fft_20; p->soa_step = soa_step_20; p->split_fft = split_fft_20; break;
    }
    return p;
}

void fft3d_execute(fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    const size_t vol = p->vol;
    const int g8 = p->batch / 8;

    for (int g = 0; g < g8; ++g) {
        const double _Complex *src = in + (size_t)g * 8 * vol;
        double _Complex *dst = out + (size_t)g * 8 * vol;
        pack8(src, vol, p->qr, p->qi);
        p->soa_fft(p->qr, p->qi);
        unpack8(p->qr, p->qi, dst, vol);
    }
    for (int v = g8 * 8; v < p->batch; ++v) {
        deinterleave(in + (size_t)v * vol, p->Sr, p->Si, vol);
        p->split_fft(p->Sr, p->Si, p->Dr, p->Di);
        interleave(p->Dr, p->Di, out + (size_t)v * vol, vol);
    }
}

/* The whole graded chain: state <- (FFT(state)+c)/(1+|FFT(state)+c|), m
 * times, final MAPPED state to final_out.  State lives in the SoA arena
 * across all m steps: pack twice, unpack once, map fused into the z pass. */
void fft3d_chain(fft3d_plan *p, const double _Complex *x0,
                 const double _Complex *c, double _Complex *final_out, int m)
{
    const size_t vol = p->vol;
    const int g8 = p->batch / 8;

    for (int g = 0; g < g8; ++g) {
        const size_t off = (size_t)g * 8 * vol;
        pack8(x0 + off, vol, p->qr, p->qi);
        pack8(c + off, vol, p->cqr, p->cqi);
        for (int s = 0; s < m; ++s)
            p->soa_step(p->qr, p->qi, p->cqr, p->cqi);
        unpack8(p->qr, p->qi, final_out + off, vol);
    }
    for (int v = g8 * 8; v < p->batch; ++v) {
        const size_t off = (size_t)v * vol;
        deinterleave(x0 + off, p->Sr, p->Si, vol);
        deinterleave(c + off, p->Cr, p->Ci, vol);
        double *sr = p->Sr, *si = p->Si, *dr = p->Dr, *di = p->Di;
        for (int s = 0; s < m; ++s) {
            switch (p->L) {
            case 10: split_fft_10(sr, si, dr, di); break;
            case 12: split_fft_12(sr, si, dr, di); break;
            case 15: split_fft_15(sr, si, dr, di); break;
            case 20: split_fft_20(sr, si, dr, di); break;
            }
            map_span(dr, di, p->Cr, p->Ci, (ptrdiff_t)vol);
            double *t;
            t = sr; sr = dr; dr = t;
            t = si; si = di; di = t;
        }
        interleave(sr, si, final_out + off, vol);
    }
}

void fft3d_destroy(fft3d_plan *p)
{
    if (!p) return;
    free(p->arena);
    free(p);
}
