/* gen_pfa_small.c -- PFA of coprime pairs, small: L = 10, 12, 15, 20.
 *
 * ROUND gen_r2: the r1 engine (three full-volume passes, split-complex
 * qr/qi arrays, whole-pencil temp buffers, map in a separate reload loop)
 * measured 1.43/2.50/6.18/16.9 us and lost L=10/12/15 to gen_batchlane's
 * bl8-lineage engine.  This round adopts that engine's structure wholesale
 * (credited: gen_batchlane gen_r1, itself from ice bl8 / rivals v5_cb7847fb,
 * 8dc1a96d) and extends it to L=20, which batchlane does not cover:
 *
 * 1. ONE interleaved site arena: site s = re[8] | im[8] (128 B), 8 volumes
 *    in the zmm lanes.  Half the memory streams of split qr/qi arrays, and
 *    a site's re/im share a 128 B block (adjacent-line prefetch pair).
 * 2. PADDED plane stride PL (sites), plane bytes == 256 (mod 4096):
 *    PL = 130/162/226/418 for L = 10/12/15/20.  Unpadded, L=12 and L=20
 *    planes are == 2048 (mod 4096): the x-pass pencil's column loads stack
 *    into TWO L1 sets (L=20: 10+ lines/set > 12-way) and thrash.
 * 3. IN-PLACE slot modules, PFA maps baked into the slot lists -- no
 *    tr[]/ti[] whole-pencil temp arrays (those spill: 30-40 live v8 at
 *    L=15/20).  In-place safety: stage-2 group c reads slots {(Qc+Pb)%L}
 *    and writes {(Q inv(Q) c + P inv(P) d)%L}; both sets are the residue
 *    class {== c mod P} iff Q == 1 mod P.  Holds for 10=2*5 (5==1 mod 2),
 *    12 as 3-then-4 (4==1 mod 3), 20=4*5 (5==1 mod 4).  15=3*5 fails
 *    (5==2 mod 3): stage-2 groups c=1,2 have EQUAL read/write slot sets,
 *    so they are one fused load-both-then-store-both codelet (DFT5X2,
 *    batchlane's exact hazard and fix).
 * 4. TWO volume sweeps per step: zy sweep per x-plane (12.8..50 KiB,
 *    L1/L2-resident; z pencils stride 1 site, y pencils stride L), then
 *    the x pass per (y,z) column at stride PL, with the graded map fused
 *    IN REGISTERS into the stage-2 stores (map8, always_inline).  The r1
 *    map-as-a-separate-span-loop measured 1.2 us/vol at L=15 and 5.4
 *    us/vol at L=20 -- 20-32% of the whole step.
 * 5. Map ladder = bl8's r4 ladder: s = a^2+b^2+1e-300, rsqrt14 + two
 *    quadratic Newtons, d = fma(s,y,1) = 1+sqrt(s), ONE vdivpd (the
 *    divider unit is idle in this pass; the r1 rcp14+2NR ladder's 5 extra
 *    uops competed with the pencil FMAs).
 * 6. (C - S) == 2048 (mod 4096) de-alias offset between state and c.
 * 7. DFT5 in the 4-constant Winograd form (f +- KQ5*q), 34 instrs vs 36.
 *
 * PFA slot maps (input n, output k; a is the stage-1 module index):
 *   L=10: n=(5a+2b)%10, k=(5c+6d)%10        stage 1: 5xDFT2, stage 2: 2xDFT5
 *   L=12: n=(4a+3b)%12, k=(4c+9d)%12        stage 1: 4xDFT3, stage 2: 3xDFT4
 *   L=15: n=(5a+3b)%15, k=(10c+6d)%15       stage 1: 5xDFT3, stage 2: 3xDFT5
 *   L=20: n=(5a+4b)%20, k=(5c+16d)%20       stage 1: 5xDFT4, stage 2: 4xDFT5
 * (10/12/15 lists verbatim from gen_batchlane gen_r1; 20 derived here and
 * verified against numpy at create-time by the single-call gate.)
 *
 * B % 8 REMAINDERS AND B = 1: unchanged r1 per-volume split-complex path
 * (buffered pencils, ping-pong passes, overlapped idempotent tails).
 *
 * Correctness: single call rel L2 ~3e-16 vs numpy; the chain map is
 * arithmetically the driver fallback's own formula.
 */
#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <stdint.h>
#include <sys/mman.h>
#if defined(__AVX512F__)
#include <immintrin.h>
#endif

#include "../fft3d_api.h"

const char *fft3d_name(void) { return "gen_pfa_small"; }
const char *fft3d_description(void)
{
    return "PFA coprime (10=2x5,12=3x4,15=3x5,20=4x5), no twiddles; interleaved "
           "site SoA 8 vols/zmm, padded planes 256 mod 4096, in-place slot "
           "codelets, zy sweep + x-pass w/ in-register fused map; B%8 split path";
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
#define K25 0.25
#define KQ5 0.55901699437494742410   /* sqrt(5)/4   */
#define S51 0.95105651629515357212   /* sin(2pi/5)  */
#define S52 0.58778525229247312917   /* sin(4pi/5)  */
#define C51 0.30901699437494742410   /* cos(2pi/5)  */
#define C52 (-0.80901699437494742410) /* cos(4pi/5) */

/* ------------------------------------------------------------- the map
 * Exactly the driver fallback's arithmetic: sc = 1/(1+sqrt(re^2+im^2)).
 * map8: one site in registers, c site at cp (re at cp, im at cp+8).
 * BORROWED: gen_batchlane gen_r1's ladder (bl8 r4 lineage): additive
 * 1e-300 guard, rsqrt14 + 2 quadratic Newtons, d = fma(s,y,1), one exact
 * vdivpd on the otherwise-idle divider unit. */

#if defined(__AVX512F__)
static inline __attribute__((always_inline)) void
map8(v8 *zr, v8 *zi, const double *cp)
{
    __m512d a = _mm512_add_pd((__m512d)*zr, _mm512_loadu_pd(cp));
    __m512d b = _mm512_add_pd((__m512d)*zi, _mm512_loadu_pd(cp + 8));
    __m512d m = _mm512_fmadd_pd(a, a,
                    _mm512_fmadd_pd(b, b, _mm512_set1_pd(1e-300)));
    __m512d r = _mm512_rsqrt14_pd(m);
    __m512d t = _mm512_mul_pd(m, r);
    r = _mm512_mul_pd(_mm512_mul_pd(_mm512_set1_pd(0.5), r),
                      _mm512_fnmadd_pd(t, r, _mm512_set1_pd(3.0)));
    t = _mm512_mul_pd(m, r);
    r = _mm512_mul_pd(_mm512_mul_pd(_mm512_set1_pd(0.5), r),
                      _mm512_fnmadd_pd(t, r, _mm512_set1_pd(3.0)));
    __m512d d = _mm512_fmadd_pd(m, r, _mm512_set1_pd(1.0));
    __m512d y = _mm512_div_pd(_mm512_set1_pd(1.0), d);
    *zr = (v8)_mm512_mul_pd(a, y);
    *zi = (v8)_mm512_mul_pd(b, y);
}
#else
static inline void map8(v8 *zr, v8 *zi, const double *cp)
{
    for (int k = 0; k < 8; ++k) {
        double a = (*zr)[k] + cp[k], b = (*zi)[k] + cp[8 + k];
        double sc = 1.0 / (1.0 + sqrt(a * a + b * b));
        (*zr)[k] = a * sc;
        (*zi)[k] = b * sc;
    }
}
#endif

/* Contiguous split spans (the B%8 split path); n need not be 8-aligned. */
static inline void map_span(double *zr, double *zi,
                            const double *cr, const double *ci, ptrdiff_t n)
{
    ptrdiff_t i = 0;
#if defined(__AVX512F__)
    const __m512d one = _mm512_set1_pd(1.0);
    const __m512d half = _mm512_set1_pd(0.5), three = _mm512_set1_pd(3.0);
    const __m512d tiny = _mm512_set1_pd(1e-300);
    for (; i + 8 <= n; i += 8) {
        __m512d a = _mm512_add_pd(_mm512_loadu_pd(zr + i), _mm512_loadu_pd(cr + i));
        __m512d b = _mm512_add_pd(_mm512_loadu_pd(zi + i), _mm512_loadu_pd(ci + i));
        __m512d m = _mm512_fmadd_pd(a, a, _mm512_fmadd_pd(b, b, tiny));
        __m512d r = _mm512_rsqrt14_pd(m);
        __m512d t = _mm512_mul_pd(m, r);
        r = _mm512_mul_pd(_mm512_mul_pd(half, r), _mm512_fnmadd_pd(t, r, three));
        t = _mm512_mul_pd(m, r);
        r = _mm512_mul_pd(_mm512_mul_pd(half, r), _mm512_fnmadd_pd(t, r, three));
        __m512d d = _mm512_fmadd_pd(m, r, one);
        __m512d y = _mm512_div_pd(one, d);
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

/* --------------------------------------- in-place site modules (SoA path)
 * A pencil lives at base p with stride st doubles between sites; slot k's
 * re vector is at p + k*st, im at p + k*st + 8.  Modules load all their
 * slots before storing, so a module is always in-place safe; the slot
 * lists in the dftL_ip functions carry the cross-group safety argument
 * from the file header. */

#define QR_(p, st, k) (*(v8 *)((p) + (size_t)(k) * (st)))
#define QI_(p, st, k) (*(v8 *)((p) + (size_t)(k) * (st) + 8))

#define STM(p, st, cp, o, rr, ii) do {                                        \
        v8 zr_ = (rr), zi_ = (ii);                                            \
        map8(&zr_, &zi_, (cp) + (size_t)(o) * (st));                          \
        QR_(p, st, o) = zr_;  QI_(p, st, o) = zi_;                            \
    } while (0)

#define D2S(p, st, a, b) do {                                                 \
        v8 x0r = QR_(p, st, a), x0i = QI_(p, st, a);                          \
        v8 x1r = QR_(p, st, b), x1i = QI_(p, st, b);                          \
        QR_(p, st, a) = x0r + x1r;  QI_(p, st, a) = x0i + x1i;                \
        QR_(p, st, b) = x0r - x1r;  QI_(p, st, b) = x0i - x1i;                \
    } while (0)

#define D3S(p, st, i0, i1, i2) do {                                           \
        v8 x0r = QR_(p, st, i0), x0i = QI_(p, st, i0);                        \
        v8 x1r = QR_(p, st, i1), x1i = QI_(p, st, i1);                        \
        v8 x2r = QR_(p, st, i2), x2i = QI_(p, st, i2);                        \
        v8 tr_ = x1r + x2r, ti_ = x1i + x2i;                                  \
        v8 ur_ = x1r - x2r, ui_ = x1i - x2i;                                  \
        v8 hr_ = x0r - 0.5 * tr_, hi_ = x0i - 0.5 * ti_;                      \
        QR_(p, st, i0) = x0r + tr_;        QI_(p, st, i0) = x0i + ti_;        \
        QR_(p, st, i1) = hr_ + K3 * ui_;   QI_(p, st, i1) = hi_ - K3 * ur_;   \
        QR_(p, st, i2) = hr_ - K3 * ui_;   QI_(p, st, i2) = hi_ + K3 * ur_;   \
    } while (0)

#define D4CORE(p, st, i0, i1, i2, i3)                                         \
        v8 x0r = QR_(p, st, i0), x0i = QI_(p, st, i0);                        \
        v8 x1r = QR_(p, st, i1), x1i = QI_(p, st, i1);                        \
        v8 x2r = QR_(p, st, i2), x2i = QI_(p, st, i2);                        \
        v8 x3r = QR_(p, st, i3), x3i = QI_(p, st, i3);                        \
        v8 t0r = x0r + x2r, t0i = x0i + x2i;                                  \
        v8 t1r = x0r - x2r, t1i = x0i - x2i;                                  \
        v8 t2r = x1r + x3r, t2i = x1i + x3i;                                  \
        v8 t3r = x1r - x3r, t3i = x1i - x3i;                                  \
        v8 y0r = t0r + t2r, y0i = t0i + t2i;                                  \
        v8 y2r = t0r - t2r, y2i = t0i - t2i;                                  \
        v8 y1r = t1r + t3i, y1i = t1i - t3r;                                  \
        v8 y3r = t1r - t3i, y3i = t1i + t3r;

#define D4S(p, st, i0, i1, i2, i3, o0, o1, o2, o3) do {                       \
        D4CORE(p, st, i0, i1, i2, i3)                                         \
        QR_(p, st, o0) = y0r;  QI_(p, st, o0) = y0i;                          \
        QR_(p, st, o1) = y1r;  QI_(p, st, o1) = y1i;                          \
        QR_(p, st, o2) = y2r;  QI_(p, st, o2) = y2i;                          \
        QR_(p, st, o3) = y3r;  QI_(p, st, o3) = y3i;                          \
    } while (0)

#define D4SM(p, st, cp, i0, i1, i2, i3, o0, o1, o2, o3) do {                  \
        D4CORE(p, st, i0, i1, i2, i3)                                         \
        STM(p, st, cp, o0, y0r, y0i);                                         \
        STM(p, st, cp, o1, y1r, y1i);                                         \
        STM(p, st, cp, o2, y2r, y2i);                                         \
        STM(p, st, cp, o3, y3r, y3i);                                         \
    } while (0)

/* DFT5, 4-constant Winograd split (34 instrs): f = x0 - p/4,
 * A1,A2 = f +- (sqrt5/4) q; equal to cos-form e1/e2 with 2 fewer FMAs. */
#define D5LOAD(T, p, st, i0, i1, i2, i3, i4)                                  \
        v8 T##x0r = QR_(p, st, i0), T##x0i = QI_(p, st, i0);                  \
        v8 T##x1r = QR_(p, st, i1), T##x1i = QI_(p, st, i1);                  \
        v8 T##x2r = QR_(p, st, i2), T##x2i = QI_(p, st, i2);                  \
        v8 T##x3r = QR_(p, st, i3), T##x3i = QI_(p, st, i3);                  \
        v8 T##x4r = QR_(p, st, i4), T##x4i = QI_(p, st, i4);

#define D5CORE(T)                                                             \
        v8 T##tar = T##x1r + T##x4r, T##tai = T##x1i + T##x4i;                \
        v8 T##tbr = T##x2r + T##x3r, T##tbi = T##x2i + T##x3i;                \
        v8 T##sar = T##x1r - T##x4r, T##sai = T##x1i - T##x4i;                \
        v8 T##sbr = T##x2r - T##x3r, T##sbi = T##x2i - T##x3i;                \
        v8 T##pr = T##tar + T##tbr, T##pi = T##tai + T##tbi;                  \
        v8 T##qr = T##tar - T##tbr, T##qi = T##tai - T##tbi;                  \
        v8 T##X0r = T##x0r + T##pr, T##X0i = T##x0i + T##pi;                  \
        v8 T##fr = T##x0r - K25 * T##pr, T##fi = T##x0i - K25 * T##pi;        \
        v8 T##A1r = T##fr + KQ5 * T##qr, T##A1i = T##fi + KQ5 * T##qi;        \
        v8 T##A2r = T##fr - KQ5 * T##qr, T##A2i = T##fi - KQ5 * T##qi;        \
        v8 T##v1r = S51 * T##sar + S52 * T##sbr;                              \
        v8 T##v1i = S51 * T##sai + S52 * T##sbi;                              \
        v8 T##v2r = S52 * T##sar - S51 * T##sbr;                              \
        v8 T##v2i = S52 * T##sai - S51 * T##sbi;

#define D5STORE(T, p, st, o0, o1, o2, o3, o4)                                 \
        QR_(p, st, o0) = T##X0r;           QI_(p, st, o0) = T##X0i;           \
        QR_(p, st, o1) = T##A1r + T##v1i;  QI_(p, st, o1) = T##A1i - T##v1r;  \
        QR_(p, st, o4) = T##A1r - T##v1i;  QI_(p, st, o4) = T##A1i + T##v1r;  \
        QR_(p, st, o2) = T##A2r + T##v2i;  QI_(p, st, o2) = T##A2i - T##v2r;  \
        QR_(p, st, o3) = T##A2r - T##v2i;  QI_(p, st, o3) = T##A2i + T##v2r;

#define D5STOREM(T, p, st, cp, o0, o1, o2, o3, o4)                            \
        STM(p, st, cp, o0, T##X0r, T##X0i);                                   \
        STM(p, st, cp, o1, T##A1r + T##v1i, T##A1i - T##v1r);                 \
        STM(p, st, cp, o4, T##A1r - T##v1i, T##A1i + T##v1r);                 \
        STM(p, st, cp, o2, T##A2r + T##v2i, T##A2i - T##v2r);                 \
        STM(p, st, cp, o3, T##A2r - T##v2i, T##A2i + T##v2r);

#define D5S(p, st, i0, i1, i2, i3, i4, o0, o1, o2, o3, o4) do {               \
        D5LOAD(a_, p, st, i0, i1, i2, i3, i4)                                 \
        D5CORE(a_)                                                            \
        D5STORE(a_, p, st, o0, o1, o2, o3, o4)                                \
    } while (0)

#define D5SM(p, st, cp, i0, i1, i2, i3, i4, o0, o1, o2, o3, o4) do {          \
        D5LOAD(a_, p, st, i0, i1, i2, i3, i4)                                 \
        D5CORE(a_)                                                            \
        D5STOREM(a_, p, st, cp, o0, o1, o2, o3, o4)                           \
    } while (0)

/* L=15 stage-2 groups c=1,c=2 read and write the SAME slot set: fused. */
#define D5X2S(p, st, i0,i1,i2,i3,i4, o0,o1,o2,o3,o4,                          \
                     j0,j1,j2,j3,j4, w0,w1,w2,w3,w4) do {                     \
        D5LOAD(a_, p, st, i0, i1, i2, i3, i4)                                 \
        D5LOAD(b_, p, st, j0, j1, j2, j3, j4)                                 \
        D5CORE(a_)                                                            \
        D5CORE(b_)                                                            \
        D5STORE(a_, p, st, o0, o1, o2, o3, o4)                                \
        D5STORE(b_, p, st, w0, w1, w2, w3, w4)                                \
    } while (0)

#define D5X2SM(p, st, cp, i0,i1,i2,i3,i4, o0,o1,o2,o3,o4,                     \
                         j0,j1,j2,j3,j4, w0,w1,w2,w3,w4) do {                 \
        D5LOAD(a_, p, st, i0, i1, i2, i3, i4)                                 \
        D5LOAD(b_, p, st, j0, j1, j2, j3, j4)                                 \
        D5CORE(a_)                                                            \
        D5CORE(b_)                                                            \
        D5STOREM(a_, p, st, cp, o0, o1, o2, o3, o4)                           \
        D5STOREM(b_, p, st, cp, w0, w1, w2, w3, w4)                           \
    } while (0)

/* ------------------------------- length-L in-place pencils, maps baked in */

static inline void dft10_ip(double *restrict p, const ptrdiff_t st)
{
    D2S(p, st, 0, 5); D2S(p, st, 2, 7); D2S(p, st, 4, 9);
    D2S(p, st, 6, 1); D2S(p, st, 8, 3);
    D5S(p, st, 0, 2, 4, 6, 8,  0, 6, 2, 8, 4);
    D5S(p, st, 5, 7, 9, 1, 3,  5, 1, 7, 3, 9);
}
static inline void dft10_ipm(double *restrict p, const ptrdiff_t st,
                             const double *restrict cp)
{
    D2S(p, st, 0, 5); D2S(p, st, 2, 7); D2S(p, st, 4, 9);
    D2S(p, st, 6, 1); D2S(p, st, 8, 3);
    D5SM(p, st, cp, 0, 2, 4, 6, 8,  0, 6, 2, 8, 4);
    D5SM(p, st, cp, 5, 7, 9, 1, 3,  5, 1, 7, 3, 9);
}

static inline void dft12_ip(double *restrict p, const ptrdiff_t st)
{
    D3S(p, st, 0, 4, 8);  D3S(p, st, 3, 7, 11);
    D3S(p, st, 6, 10, 2); D3S(p, st, 9, 1, 5);
    D4S(p, st, 0, 3, 6, 9,   0, 9, 6, 3);
    D4S(p, st, 4, 7, 10, 1,  4, 1, 10, 7);
    D4S(p, st, 8, 11, 2, 5,  8, 5, 2, 11);
}
static inline void dft12_ipm(double *restrict p, const ptrdiff_t st,
                             const double *restrict cp)
{
    D3S(p, st, 0, 4, 8);  D3S(p, st, 3, 7, 11);
    D3S(p, st, 6, 10, 2); D3S(p, st, 9, 1, 5);
    D4SM(p, st, cp, 0, 3, 6, 9,   0, 9, 6, 3);
    D4SM(p, st, cp, 4, 7, 10, 1,  4, 1, 10, 7);
    D4SM(p, st, cp, 8, 11, 2, 5,  8, 5, 2, 11);
}

/* NOTE: gen_batchlane's SCHED15 (optimize("schedule-insns","sched-pressure"))
 * was A/B-tested on THIS engine at 15 (+/-0, best 4.47 without vs 4.61 with)
 * and at 20 (+5%: 14.77 vs 14.06); their -13% was specific to their codelet
 * structure.  Default scheduler everywhere here. */

static inline void dft15_ip(double *restrict p, const ptrdiff_t st)
{
    D3S(p, st, 0, 5, 10);  D3S(p, st, 3, 8, 13); D3S(p, st, 6, 11, 1);
    D3S(p, st, 9, 14, 4);  D3S(p, st, 12, 2, 7);
    D5S(p, st, 0, 3, 6, 9, 12,  0, 6, 12, 3, 9);
    D5X2S(p, st, 5, 8, 11, 14, 2,   10, 1, 7, 13, 4,
                 10, 13, 1, 4, 7,    5, 11, 2, 8, 14);
}
static inline void dft15_ipm(double *restrict p, const ptrdiff_t st,
                                     const double *restrict cp)
{
    D3S(p, st, 0, 5, 10);  D3S(p, st, 3, 8, 13); D3S(p, st, 6, 11, 1);
    D3S(p, st, 9, 14, 4);  D3S(p, st, 12, 2, 7);
    D5SM(p, st, cp, 0, 3, 6, 9, 12,  0, 6, 12, 3, 9);
    D5X2SM(p, st, cp, 5, 8, 11, 14, 2,   10, 1, 7, 13, 4,
                      10, 13, 1, 4, 7,    5, 11, 2, 8, 14);
}

static inline void dft20_ip(double *restrict p, const ptrdiff_t st)
{
    D4S(p, st, 0, 5, 10, 15,   0, 5, 10, 15);
    D4S(p, st, 4, 9, 14, 19,   4, 9, 14, 19);
    D4S(p, st, 8, 13, 18, 3,   8, 13, 18, 3);
    D4S(p, st, 12, 17, 2, 7,   12, 17, 2, 7);
    D4S(p, st, 16, 1, 6, 11,   16, 1, 6, 11);
    D5S(p, st, 0, 4, 8, 12, 16,    0, 16, 12, 8, 4);
    D5S(p, st, 5, 9, 13, 17, 1,    5, 1, 17, 13, 9);
    D5S(p, st, 10, 14, 18, 2, 6,   10, 6, 2, 18, 14);
    D5S(p, st, 15, 19, 3, 7, 11,   15, 11, 7, 3, 19);
}
static inline void dft20_ipm(double *restrict p, const ptrdiff_t st,
                             const double *restrict cp)
{
    D4S(p, st, 0, 5, 10, 15,   0, 5, 10, 15);
    D4S(p, st, 4, 9, 14, 19,   4, 9, 14, 19);
    D4S(p, st, 8, 13, 18, 3,   8, 13, 18, 3);
    D4S(p, st, 12, 17, 2, 7,   12, 17, 2, 7);
    D4S(p, st, 16, 1, 6, 11,   16, 1, 6, 11);
    D5SM(p, st, cp, 0, 4, 8, 12, 16,    0, 16, 12, 8, 4);
    D5SM(p, st, cp, 5, 9, 13, 17, 1,    5, 1, 17, 13, 9);
    D5SM(p, st, cp, 10, 14, 18, 2, 6,   10, 6, 2, 18, 14);
    D5SM(p, st, cp, 15, 19, 3, 7, 11,   15, 11, 7, 3, 19);
}

/* -------------------------------------------- the sweeps, one per L
 * PL = padded plane stride in sites; plane bytes == 256 (mod 4096). */

#define DEF_ENGINE(L, PLV)                                                    \
static void sweep_zy_##L(double *restrict pl)                                 \
{                                                                             \
    for (int y = 0; y < (L); ++y)                                             \
        dft##L##_ip(pl + (size_t)y * (L) * 16, 16);                          \
    for (int z = 0; z < (L); ++z)                                             \
        dft##L##_ip(pl + (size_t)z * 16, (ptrdiff_t)(L) * 16);               \
}                                                                             \
static void soa_fft_##L(double *restrict S)                                   \
{                                                                             \
    for (int x = 0; x < (L); ++x)                                             \
        sweep_zy_##L(S + (size_t)x * (PLV) * 16);                            \
    for (int c = 0; c < (L) * (L); ++c)                                       \
        dft##L##_ip(S + (size_t)c * 16, (ptrdiff_t)(PLV) * 16);              \
}                                                                             \
static void soa_step_##L(double *restrict S, const double *restrict C)        \
{                                                                             \
    for (int x = 0; x < (L); ++x)                                             \
        sweep_zy_##L(S + (size_t)x * (PLV) * 16);                            \
    for (int c = 0; c < (L) * (L); ++c)                                       \
        dft##L##_ipm(S + (size_t)c * 16, (ptrdiff_t)(PLV) * 16,              \
                     C + (size_t)c * 16);                                     \
}

DEF_ENGINE(10, 130)
DEF_ENGINE(12, 162)
DEF_ENGINE(15, 226)
DEF_ENGINE(20, 418)

static int plane_stride_sites(int L)   /* L^2 padded to == 2 (mod 32) sites */
{
    switch (L) {
    case 10: return 130;
    case 12: return 162;
    case 15: return 226;
    case 20: return 418;
    }
    return 0;
}

/* --------------------------------------- split-path pencils (B%8, B=1)
 * The r1 buffered out-of-place pencils with the equivalent IN/OUT index
 * tables; used only by the ping-pong per-volume path. */

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
        v8 pr_ = ar_ + br_, pi_ = ai_ + bi_;                          \
        v8 qr_ = ar_ - br_, qi_ = ai_ - bi_;                          \
        v8 fr_ = r0 - K25 * pr_, fi_ = i0 - K25 * pi_;                \
        v8 e1r_ = fr_ + KQ5 * qr_, e1i_ = fi_ + KQ5 * qi_;            \
        v8 e2r_ = fr_ - KQ5 * qr_, e2i_ = fi_ - KQ5 * qi_;            \
        r0 += pr_; i0 += pi_;                                         \
        v8 o1r_ = S51 * cr_ + S52 * dr_, o1i_ = S51 * ci_ + S52 * di_;\
        v8 o2r_ = S52 * cr_ - S51 * dr_, o2i_ = S52 * ci_ - S51 * di_;\
        r1 = e1r_ + o1i_; i1 = e1i_ - o1r_;                           \
        r4 = e1r_ - o1i_; i4 = e1i_ + o1r_;                           \
        r2 = e2r_ + o2i_; i2 = e2i_ - o2r_;                           \
        r3 = e2r_ - o2i_; i3 = e2i_ + o2r_;                           \
    } while (0)

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

/* One x-plane of 8 interleaved volumes (lane stride vol complex) -> the
 * padded interleaved site arena (site = re[8]|im[8]). */
static void pack8_plane(const double _Complex *in, size_t lane_stride,
                        size_t n, double *q)
{
    const double *base = (const double *)in;
    size_t p = 0;
    for (; p + 4 <= n; p += 4) {
        v8 rows[8];
        for (int k = 0; k < 8; ++k)
            rows[k] = vload(base + 2 * ((size_t)k * lane_stride + p));
        tr8(rows);
        for (int j = 0; j < 4; ++j) {
            vstore(q + (p + j) * 16,     rows[2 * j]);
            vstore(q + (p + j) * 16 + 8, rows[2 * j + 1]);
        }
    }
    for (; p < n; ++p)
        for (int k = 0; k < 8; ++k) {
            q[p * 16 + k]     = base[2 * ((size_t)k * lane_stride + p)];
            q[p * 16 + 8 + k] = base[2 * ((size_t)k * lane_stride + p) + 1];
        }
}

static void unpack8_plane(const double *q, double _Complex *out,
                          size_t lane_stride, size_t n)
{
    double *base = (double *)out;
    size_t p = 0;
    for (; p + 4 <= n; p += 4) {
        v8 rows[8];
        for (int j = 0; j < 4; ++j) {
            rows[2 * j]     = vload(q + (p + j) * 16);
            rows[2 * j + 1] = vload(q + (p + j) * 16 + 8);
        }
        tr8(rows);
        for (int k = 0; k < 8; ++k)
            vstore(base + 2 * ((size_t)k * lane_stride + p), rows[k]);
    }
    for (; p < n; ++p)
        for (int k = 0; k < 8; ++k) {
            base[2 * ((size_t)k * lane_stride + p)]     = q[p * 16 + k];
            base[2 * ((size_t)k * lane_stride + p) + 1] = q[p * 16 + 8 + k];
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

/* 2 MiB-aligned anonymous mapping + MADV_HUGEPAGE (node THP is madvise
 * mode): with 4K pages the arena's physical page coloring varies per run
 * and L=15 measured 4.50-5.87 us run to run (in-run sd 0.05%); a huge-page
 * arena is physically contiguous, so L2 indexing is deterministic and
 * matches the best-case runs. */
static double *arena_alloc(size_t bytes, size_t *out_len)
{
    const size_t HP = (size_t)1 << 21;
    size_t len = (bytes + HP - 1) & ~(HP - 1);
    char *raw = mmap(NULL, len + HP, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (raw == MAP_FAILED) return NULL;
    uintptr_t a = ((uintptr_t)raw + HP - 1) & ~(uintptr_t)(HP - 1);
    size_t head = a - (uintptr_t)raw;
    if (head) munmap(raw, head);
    if (HP - head) munmap((char *)a + len, HP - head);
#ifdef MADV_HUGEPAGE
    madvise((void *)a, len, MADV_HUGEPAGE);
#endif
    *out_len = len;
    return (double *)a;
}

struct fft3d_plan {
    int L, batch, PL;
    size_t vol, arena_len;
    double *arena;
    double *S, *C;                       /* interleaved site arenas        */
    double *Sr, *Si, *Dr, *Di, *Cr, *Ci; /* split per-volume: vol each     */
    void (*soa_fft)(double *);
    void (*soa_step)(double *, const double *);
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
    p->PL = plane_stride_sites(L);

    /* Arena is 4096-aligned; S sits at offset 0, C at an offset == 2048
     * (mod 4096) so state and c never collide in the low address bits
     * (gen_batchlane / bl8's de-alias offset).  Split buffers keep the r1
     * one-line stagger between components. */
    size_t soa = (size_t)L * p->PL * 16;               /* doubles, one arena */
    size_t coff = ((soa + 511) / 512) * 512 + 256;
    size_t svol = p->vol + 8;
    size_t total = coff + soa + 6 * svol;
    p->arena = arena_alloc(total * sizeof(double), &p->arena_len);
    if (!p->arena) {
        free(p);
        return NULL;
    }
    memset(p->arena, 0, total * sizeof(double));  /* fault in as huge pages */
    p->S = p->arena;
    p->C = p->arena + coff;
    p->Sr = p->C + soa;
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
    const size_t LL = (size_t)p->L * p->L;
    const size_t pstride = (size_t)p->PL * 16;
    const int g8 = p->batch / 8;

    for (int g = 0; g < g8; ++g) {
        const double _Complex *src = in + (size_t)g * 8 * vol;
        double _Complex *dst = out + (size_t)g * 8 * vol;
        for (int x = 0; x < p->L; ++x)
            pack8_plane(src + x * LL, vol, LL, p->S + x * pstride);
        p->soa_fft(p->S);
        for (int x = 0; x < p->L; ++x)
            unpack8_plane(p->S + x * pstride, dst + x * LL, vol, LL);
    }
    for (int v = g8 * 8; v < p->batch; ++v) {
        deinterleave(in + (size_t)v * vol, p->Sr, p->Si, vol);
        p->split_fft(p->Sr, p->Si, p->Dr, p->Di);
        interleave(p->Dr, p->Di, out + (size_t)v * vol, vol);
    }
}

/* The whole graded chain: state <- (FFT(state)+c)/(1+|FFT(state)+c|), m
 * times, final MAPPED state to final_out.  State lives in the site arena
 * across all m steps: pack twice, unpack once, map fused into the x pass. */
void fft3d_chain(fft3d_plan *p, const double _Complex *x0,
                 const double _Complex *c, double _Complex *final_out, int m)
{
    const size_t vol = p->vol;
    const size_t LL = (size_t)p->L * p->L;
    const size_t pstride = (size_t)p->PL * 16;
    const int g8 = p->batch / 8;

    for (int g = 0; g < g8; ++g) {
        const size_t off = (size_t)g * 8 * vol;
        for (int x = 0; x < p->L; ++x) {
            pack8_plane(x0 + off + x * LL, vol, LL, p->S + x * pstride);
            pack8_plane(c + off + x * LL, vol, LL, p->C + x * pstride);
        }
        for (int s = 0; s < m; ++s)
            p->soa_step(p->S, p->C);
        for (int x = 0; x < p->L; ++x)
            unpack8_plane(p->S + x * pstride, final_out + off + x * LL, vol, LL);
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
    munmap(p->arena, p->arena_len);
    free(p);
}
