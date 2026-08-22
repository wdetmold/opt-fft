/* L13_rader -- forward 3D DFT of a 13^3 cube, batched, single-threaded.
 *
 * ALGORITHM (per 1D 13-point line): Rader's prime-length construction with the
 * length-12 cyclic convolution evaluated by CRT/symmetry splitting rather than
 * by a pair of FFT-12s.  Fold u_m = x[g^m] + x[13-g^m], v_m = x[g^m] - x[13-g^m]
 * over the order-6 quotient of (Z/13)* by {+-1} (g = 2); then
 *     X[g^n]      = x0 + CC_n + i*SS_n
 *     X[13-g^n]   = x0 + CC_n - i*SS_n
 * where CC is a length-6 CYCLIC correlation of u with cos(2*pi*g^t/13) -- split
 * once more (the Z2 factor of Z6) into an x0-seeded cyclic-3 plus a
 * negacyclic-3, both with halved constants -- and SS is a length-6 NEGACYCLIC
 * correlation of v with -sin(2*pi*g^t/13), done densely.
 * 186 vector FP instructions (108 FMA + 78 add/sub) per 13-point transform,
 * against 204 for the dense conjugate-symmetric matvec and 238 for Rader with
 * two PFA FFT-12s.  All constants are real scalars broadcast from memory; the
 * vector lanes hold INDEPENDENT transforms, so there is no cross-lane op in
 * the kernel and Rader's permutations are compile-time load/store offsets that
 * cost nothing.
 *
 * The kernel structure (fold -> seeded cyclic + negacyclic -> dense nega on v)
 * is the L17_winograd module scheme, adopted via L17_rader panel_r2, re-derived
 * for p = 13.  3D architecture (X-first, plane-fused, VW=8 path):
 *   x pass FIRST over `in` (169 contiguous lanes m = y*13+z, deinterleaving
 *     loads), aligned split stores into A[kx][y*13+z];
 *   z pass over the volume's 169 GLOBAL rows g = kx*13+y in 22 blocks of 8
 *     (blocks straddle kx planes -- the per-plane form needed 26), each block
 *     a fused transposing-load / 13-point-kernel / transposing-store into the
 *     double-buffered per-plane U[y][kz];
 *   y pass per kx plane (lanes = kz), fired the moment that plane's U
 *     completes, interleaving stores DIRECTLY into out, with the out plane
 *     prefetchw'd one pipeline step ahead whenever in+out stream past this
 *     machine's L2 (the r6/r7 staged burst copy lost to direct+pfw in every
 *     streaming regime once pfw hid the RFO; panel_r8).
 * Everything is plain GNU C vector extensions -- no intrinsics -- so the same
 * source builds on AVX-512 (VW=8, node/wallaby), AVX2 (VW=4, wombat) and
 * anything gcc can emulate.
 *
 * Contract: ../fft3d_api.h.  Strategy record: ../strategies/L13_rader.md.
 */
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "../fft3d_api.h"

#define LN    13
#define NPL   169              /* 13*13 */
#define NVOL  2197             /* 13^3  */
#define TR    16               /* plane row stride (13 padded up) */

#if defined(__AVX512F__)
#  define VW 8
#else
#  define VW 4
#endif

/* A plane pitch: 176 (VW=8) / 172.  L23_rader's panel_r7 odd-cache-line pad
 * (176 -> 184 = 23 lines) was A/B'd here on wallaby and did NOT transfer:
 * B=1 a wash, B=512 direct+pw 1994 vs 1926 us -- the x pass's 2704 B input
 * row stride (gcd 16 with 4096) already decorrelates the residues.
 * -DL13R_PS re-opens the experiment. */
#ifdef L13R_PS
#  define PS L13R_PS
#else
#  define PS (((NPL + VW - 1) / VW) * VW)
#endif
#define NLB ((LN  + VW - 1) / VW)          /* lane blocks in a 13-wide space  */
#define NXB ((NPL + VW - 1) / VW)          /* lane blocks in the x pass       */

typedef double    vd  __attribute__((vector_size(VW * 8), aligned(8)));
typedef long long vl  __attribute__((vector_size(VW * 8)));
typedef double    v4d __attribute__((vector_size(32), aligned(8)));
typedef long long v4l __attribute__((vector_size(32)));

#if defined(__clang__)
#  define SH4(a,b,m0,m1,m2,m3) __builtin_shufflevector((a),(b),m0,m1,m2,m3)
#  if VW == 8
#    define ILO(r,i) __builtin_shufflevector((r),(i),0,8,1,9,2,10,3,11)
#    define IHI(r,i) __builtin_shufflevector((r),(i),4,12,5,13,6,14,7,15)
#    define DLE(a,b) __builtin_shufflevector((a),(b),0,2,4,6,8,10,12,14)
#    define DLO(a,b) __builtin_shufflevector((a),(b),1,3,5,7,9,11,13,15)
#  else
#    define ILO(r,i) __builtin_shufflevector((r),(i),0,4,1,5)
#    define IHI(r,i) __builtin_shufflevector((r),(i),2,6,3,7)
#    define DLE(a,b) __builtin_shufflevector((a),(b),0,2,4,6)
#    define DLO(a,b) __builtin_shufflevector((a),(b),1,3,5,7)
#  endif
#else
#  define SH4(a,b,m0,m1,m2,m3) __builtin_shuffle((a),(b),(v4l){m0,m1,m2,m3})
#  if VW == 8
#    define ILO(r,i) __builtin_shuffle((r),(i),(vl){0,8,1,9,2,10,3,11})
#    define IHI(r,i) __builtin_shuffle((r),(i),(vl){4,12,5,13,6,14,7,15})
#    define DLE(a,b) __builtin_shuffle((a),(b),(vl){0,2,4,6,8,10,12,14})
#    define DLO(a,b) __builtin_shuffle((a),(b),(vl){1,3,5,7,9,11,13,15})
#  else
#    define ILO(r,i) __builtin_shuffle((r),(i),(vl){0,4,1,5})
#    define IHI(r,i) __builtin_shuffle((r),(i),(vl){2,6,3,7})
#    define DLE(a,b) __builtin_shuffle((a),(b),(vl){0,2,4,6})
#    define DLO(a,b) __builtin_shuffle((a),(b),(vl){1,3,5,7})
#  endif
#endif

#define VL_(p)   (*(const vd *)(const void *)(p))
#define VS_(p,x) (*(vd *)(void *)(p) = (x))

/* Pin a broadcast constant in a register via an empty asm barrier.  At L=17
 * pinning lost (register pressure); L13_direct measured all-pinning winning
 * 10% at L=13 ("below ~15 distinct constants, pin everything").  Gated so
 * the tradeoff stays measurable per kernel stage. */
#if defined(__AVX512VL__)
#  define KPIN(c) ({ vd _t = ((vd){0} + (c)); __asm__("" : "+v"(_t)); _t; })
#else
#  define KPIN(c) ((vd){0} + (c))
#endif
#ifndef L13R_PIN
#define L13R_PIN 1
#endif
#if L13R_PIN
#  define KC(c) KPIN(c)
#else
#  define KC(c) ((vd){0} + (c))
#endif
#if L13R_PIN >= 2
#  define KC2(c) KPIN(c)
#else
#  define KC2(c) ((vd){0} + (c))
#endif

/* Kernel constants.  g^t mod 13 for t = 0..5 is [1,2,4,8,3,6];
 *   CPt = (cos(2pi g^t/13) + cos(2pi g^(t+3)/13)) / 2   (cyclic-3 kernel)
 *   CMt = (cos(2pi g^t/13) - cos(2pi g^(t+3)/13)) / 2   (negacyclic-3 kernel)
 *   SNt = -sin(2pi g^t/13)                              (negacyclic-6 kernel) */
#define CP0 ( 6.84726387410543036e-02)
#define CP1 ( 3.44300713493239485e-01)
#define CP2 (-6.62773352234293789e-01)
#define CM0 ( 8.16983386912155662e-01)
#define CM1 ( 2.23764033237916465e-01)
#define CM2 ( 3.08168465191758278e-01)
#define SN0 (-4.64723172043768507e-01)
#define SN1 (-8.22983865893656352e-01)
#define SN2 (-9.35016242685414833e-01)
#define SN3 ( 6.63122658240794971e-01)
#define SN4 (-9.92708874098053973e-01)
#define SN5 (-2.39315664287557683e-01)

/* ------------------------------------------------------------- transposes */

/* d[j*ds + i] = s[i*ss + j], i,j in [0,13): 4x4 ymm tiles + scalar edges.
 * Tile shuffles verbatim from L17_rader (exemplars/panel_r5), tails at 12. */
static inline __attribute__((always_inline))
void transpose13(const double *s, long ss, double *d, long ds)
{
    for (int i0 = 0; i0 < 12; i0 += 4)
        for (int j0 = 0; j0 < 12; j0 += 4) {
            const double *p = s + (long)i0*ss + j0;
            v4d r0 = *(const v4d *)(const void *)(p);
            v4d r1 = *(const v4d *)(const void *)(p + ss);
            v4d r2 = *(const v4d *)(const void *)(p + 2*ss);
            v4d r3 = *(const v4d *)(const void *)(p + 3*ss);
            v4d h0 = SH4(r0,r1, 0,4,2,6), h1 = SH4(r0,r1, 1,5,3,7);
            v4d h2 = SH4(r2,r3, 0,4,2,6), h3 = SH4(r2,r3, 1,5,3,7);
            double *q = d + (long)j0*ds + i0;
            *(v4d *)(void *)(q)        = SH4(h0,h2, 0,1,4,5);
            *(v4d *)(void *)(q + ds)   = SH4(h1,h3, 0,1,4,5);
            *(v4d *)(void *)(q + 2*ds) = SH4(h0,h2, 2,3,6,7);
            *(v4d *)(void *)(q + 3*ds) = SH4(h1,h3, 2,3,6,7);
        }
    for (int i = 0; i < 12; ++i) d[12*ds + i] = s[(long)i*ss + 12];
    for (int j = 0; j < LN; ++j) d[(long)j*ds + 12] = s[12*ss + j];
}

/* dr[j*ds + i] = Re s[i*13 + j], di likewise; s is interleaved complex.
 * Complex 4x4 tile transpose at 128-bit granularity, then re/im split:
 * 16 shuffles per 16 complex elements (L17_rader round 1's measured 5.31 ->
 * 3.34 us ordering). */
static inline __attribute__((always_inline))
void deint_transpose13(const double *s, double *dr, double *di, long ds)
{
    for (int i0 = 0; i0 < 12; i0 += 4)
        for (int j0 = 0; j0 < 12; j0 += 4) {
            const double *p = s + 2*((long)i0*LN + j0);
            v4d a0 = *(const v4d *)(const void *)(p);
            v4d b0 = *(const v4d *)(const void *)(p + 4);
            v4d a1 = *(const v4d *)(const void *)(p + 2*LN);
            v4d b1 = *(const v4d *)(const void *)(p + 2*LN + 4);
            v4d a2 = *(const v4d *)(const void *)(p + 4*LN);
            v4d b2 = *(const v4d *)(const void *)(p + 4*LN + 4);
            v4d a3 = *(const v4d *)(const void *)(p + 6*LN);
            v4d b3 = *(const v4d *)(const void *)(p + 6*LN + 4);
            v4d l0 = SH4(a0,a1, 0,1,4,5), h0 = SH4(a2,a3, 0,1,4,5);
            v4d l1 = SH4(a0,a1, 2,3,6,7), h1 = SH4(a2,a3, 2,3,6,7);
            v4d l2 = SH4(b0,b1, 0,1,4,5), h2 = SH4(b2,b3, 0,1,4,5);
            v4d l3 = SH4(b0,b1, 2,3,6,7), h3 = SH4(b2,b3, 2,3,6,7);
            double *qr = dr + (long)j0*ds + i0, *qi = di + (long)j0*ds + i0;
            *(v4d *)(void *)(qr)        = SH4(l0,h0, 0,2,4,6);
            *(v4d *)(void *)(qi)        = SH4(l0,h0, 1,3,5,7);
            *(v4d *)(void *)(qr + ds)   = SH4(l1,h1, 0,2,4,6);
            *(v4d *)(void *)(qi + ds)   = SH4(l1,h1, 1,3,5,7);
            *(v4d *)(void *)(qr + 2*ds) = SH4(l2,h2, 0,2,4,6);
            *(v4d *)(void *)(qi + 2*ds) = SH4(l2,h2, 1,3,5,7);
            *(v4d *)(void *)(qr + 3*ds) = SH4(l3,h3, 0,2,4,6);
            *(v4d *)(void *)(qi + 3*ds) = SH4(l3,h3, 1,3,5,7);
        }
    for (int i = 0; i < 12; ++i) {
        dr[12*ds + i] = s[2*((long)i*LN + 12)];
        di[12*ds + i] = s[2*((long)i*LN + 12) + 1];
    }
    for (int j = 0; j < LN; ++j) {
        dr[(long)j*ds + 12] = s[2*(12*LN + j)];
        di[(long)j*ds + 12] = s[2*(12*LN + j) + 1];
    }
}

/* -------------------------------------------------------------- the kernel */

/* VW independent 13-point DFTs, split re/im.
 *   lmode 0: split loads at xr/xi + k*xs
 *   lmode 1: interleaved loads at isrc + 2*(k*NPL + im0) with a
 *            deinterleaving shuffle pair per input (the x pass reads the
 *            caller's `in` directly; idea from L17_rader / L17_matrixsimd)
 *   smode 0: split store at orr/oii + k*os
 *   smode 1: interleaved complex store at dst + 2*(k*os + m0)
 * Every input is loaded before any output is stored, so lmode 0 / smode 0 may
 * run in place (the z pass does).  Negated wrap terms of the negacyclic parts
 * are written as -= with the SAME constant so gcc emits vfnmadd against one
 * .LC slot instead of materialising negated copies (L17_rader's note). */
static inline __attribute__((always_inline)) void kern13(
        const double *xr, const double *xi, long xs,
        const double *isrc, long im0,
        double *orr, double *oii,
        double *dst, long m0, long os, const int lmode, const int smode)
{
#define LD(k, vr, vi) do {                                                    \
        if (lmode == 0) {                                                     \
            (vr) = VL_(xr + (long)(k)*xs);  (vi) = VL_(xi + (long)(k)*xs);    \
        } else {                                                              \
            const double *_q = isrc + 2*((long)(k)*NPL + im0);                \
            vd _a = VL_(_q), _b = VL_(_q + VW);                               \
            (vr) = DLE(_a,_b);  (vi) = DLO(_a,_b);                            \
        }                                                                     \
    } while (0)
#define ST(k, vr, vi) do {                                                    \
        if (smode == 0) {                                                     \
            VS_(orr + (long)(k)*os, (vr));  VS_(oii + (long)(k)*os, (vi));    \
        } else {                                                              \
            double *_p = dst + 2*((long)(k)*os + m0);                         \
            *(vd *)(void *)(_p)      = ILO((vr),(vi));                        \
            *(vd *)(void *)(_p + VW) = IHI((vr),(vi));                        \
        }                                                                     \
    } while (0)

    const vd kCP0 = KC2(CP0), kCP1 = KC2(CP1), kCP2 = KC2(CP2);
    const vd kCM0 = KC2(CM0), kCM1 = KC2(CM1), kCM2 = KC2(CM2);

    vd x0r, x0i;
    LD(0, x0r, x0i);
    vd dcr = x0r, dci = x0i;

    vd vvr[6], vvi[6];                 /* v_m = x[g^m] - x[13-g^m]           */
    vd a0r,a1r,a2r, a0i,a1i,a2i;       /* x0-seeded cyclic-3 accumulators    */
    vd b0r,b1r,b2r, b0i,b1i,b2i;       /* negacyclic-3 accumulators          */

    /* Fold block t: rows (g^t, 13-g^t) give quotient slot t, rows
     * (g^(t+3), 13-g^(t+3)) slot t+3; P_t/Q_t feed the split cyclic-6. */
#define FOLD(t, j0,k0, j1,k1, BODY) do {                                      \
        vd e0r,e0i,f0r,f0i,e1r,e1i,f1r,f1i;                                   \
        LD(j0,e0r,e0i); LD(k0,f0r,f0i); LD(j1,e1r,e1i); LD(k1,f1r,f1i);       \
        vvr[t]     = e0r - f0r;  vvi[t]     = e0i - f0i;                      \
        vvr[(t)+3] = e1r - f1r;  vvi[(t)+3] = e1i - f1i;                      \
        vd u0r = e0r + f0r, u0i = e0i + f0i;                                  \
        vd u1r = e1r + f1r, u1i = e1i + f1i;                                  \
        vd pr = u0r + u1r, pi = u0i + u1i;                                    \
        vd qr = u0r - u1r, qi = u0i - u1i;                                    \
        dcr += pr;  dci += pi;                                                \
        BODY                                                                  \
    } while (0)

    FOLD(0,  1,12,  8, 5,
        a0r = x0r + pr*kCP0;  a1r = x0r + pr*kCP1;  a2r = x0r + pr*kCP2;
        a0i = x0i + pi*kCP0;  a1i = x0i + pi*kCP1;  a2i = x0i + pi*kCP2;
        b0r = qr*kCM0;  b1r = qr*kCM1;  b2r = qr*kCM2;
        b0i = qi*kCM0;  b1i = qi*kCM1;  b2i = qi*kCM2; );
    FOLD(1,  2,11,  3,10,
        a0r += pr*kCP1;  a1r += pr*kCP2;  a2r += pr*kCP0;
        a0i += pi*kCP1;  a1i += pi*kCP2;  a2i += pi*kCP0;
        b0r += qr*kCM1;  b1r += qr*kCM2;  b2r -= qr*kCM0;
        b0i += qi*kCM1;  b1i += qi*kCM2;  b2i -= qi*kCM0; );
    FOLD(2,  4, 9,  6, 7,
        a0r += pr*kCP2;  a1r += pr*kCP0;  a2r += pr*kCP1;
        a0i += pi*kCP2;  a1i += pi*kCP0;  a2i += pi*kCP1;
        b0r += qr*kCM2;  b1r -= qr*kCM0;  b2r -= qr*kCM1;
        b0i += qi*kCM2;  b1i -= qi*kCM0;  b2i -= qi*kCM1; );
#undef FOLD

    ST(0, dcr, dci);                            /* X[0] = x0 + sum u        */

    /* cc[n] = x0 + CC_n = a_n + b_n; cc[n+3] = a_n - b_n */
    vd cc0r = a0r + b0r, cc0i = a0i + b0i, cc3r = a0r - b0r, cc3i = a0i - b0i;
    vd cc1r = a1r + b1r, cc1i = a1i + b1i, cc4r = a1r - b1r, cc4i = a1i - b1i;
    vd cc2r = a2r + b2r, cc2i = a2i + b2i, cc5r = a2r - b2r, cc5i = a2i - b2i;

    const vd kSN0 = KC(SN0), kSN1 = KC(SN1), kSN2 = KC(SN2),
             kSN3 = KC(SN3), kSN4 = KC(SN4), kSN5 = KC(SN5);

    /* Dense negacyclic-6 on v, three outputs at a time:
     *   SS_n = sum_m SN[(m+n) mod 6] * (-1)^floor((m+n)/6) * v_m
     * then X[g^n] = cc_n + i*SS_n, X[13-g^n] = cc_n - i*SS_n; with split
     * re/im the *(+-i) is a rename plus sign in the add. */
#define SA1(m, w0,w1,w2) do {                                                 \
        vd tr_ = vvr[m], ti_ = vvi[m];                                        \
        a0r = tr_*(w0);  a1r = tr_*(w1);  a2r = tr_*(w2);                     \
        a0i = ti_*(w0);  a1i = ti_*(w1);  a2i = ti_*(w2);                     \
    } while (0)
#define SA(m, s0,w0, s1,w1, s2,w2) do {                                       \
        vd tr_ = vvr[m], ti_ = vvi[m];                                        \
        a0r s0 tr_*(w0);  a1r s1 tr_*(w1);  a2r s2 tr_*(w2);                  \
        a0i s0 ti_*(w0);  a1i s1 ti_*(w1);  a2i s2 ti_*(w2);                  \
    } while (0)

    /* first half: n = 0,1,2 -> output rows (1,12), (2,11), (4,9) */
    SA1(0,     kSN0,    kSN1,    kSN2);
    SA (1, +=, kSN1, +=,kSN2, +=,kSN3);
    SA (2, +=, kSN2, +=,kSN3, +=,kSN4);
    SA (3, +=, kSN3, +=,kSN4, +=,kSN5);
    SA (4, +=, kSN4, +=,kSN5, -=,kSN0);
    SA (5, +=, kSN5, -=,kSN0, -=,kSN1);

    ST( 1, cc0r - a0i, cc0i + a0r);  ST(12, cc0r + a0i, cc0i - a0r);
    ST( 2, cc1r - a1i, cc1i + a1r);  ST(11, cc1r + a1i, cc1i - a1r);
    ST( 4, cc2r - a2i, cc2i + a2r);  ST( 9, cc2r + a2i, cc2i - a2r);

    /* second half: n = 3,4,5 -> output rows (8,5), (3,10), (6,7) */
    SA1(0,     kSN3,    kSN4,    kSN5);
    SA (1, +=, kSN4, +=,kSN5, -=,kSN0);
    SA (2, +=, kSN5, -=,kSN0, -=,kSN1);
    SA (3, -=, kSN0, -=,kSN1, -=,kSN2);
    SA (4, -=, kSN1, -=,kSN2, -=,kSN3);
    SA (5, -=, kSN2, -=,kSN3, -=,kSN4);
#undef SA
#undef SA1

    ST( 8, cc3r - a0i, cc3i + a0r);  ST( 5, cc3r + a0i, cc3i - a0r);
    ST( 3, cc4r - a1i, cc4i + a1r);  ST(10, cc4r + a1i, cc4i - a1r);
    ST( 6, cc5r - a2i, cc5i + a2r);  ST( 7, cc5r + a2i, cc5i - a2r);
#undef ST
#undef LD
}

/* kern13_regs: the same 13-point DFT with inputs and outputs in register
 * arrays -- used by the fused z-plane kernel below, which needs all 26
 * outputs in registers before its transposing store.  The math is a verbatim
 * copy of kern13's; keep the two in sync. */
static inline __attribute__((always_inline)) void kern13_regs(
        const vd *xr, const vd *xi, vd *yr, vd *yi)
{
    const vd kCP0 = KC2(CP0), kCP1 = KC2(CP1), kCP2 = KC2(CP2);
    const vd kCM0 = KC2(CM0), kCM1 = KC2(CM1), kCM2 = KC2(CM2);

    vd x0r = xr[0], x0i = xi[0];
    vd dcr = x0r, dci = x0i;
    vd vvr[6], vvi[6];
    vd a0r,a1r,a2r, a0i,a1i,a2i;
    vd b0r,b1r,b2r, b0i,b1i,b2i;

#define FOLDR(t, j0,k0, j1,k1, BODY) do {                                     \
        vd e0r = xr[j0], e0i = xi[j0], f0r = xr[k0], f0i = xi[k0];            \
        vd e1r = xr[j1], e1i = xi[j1], f1r = xr[k1], f1i = xi[k1];            \
        vvr[t]     = e0r - f0r;  vvi[t]     = e0i - f0i;                      \
        vvr[(t)+3] = e1r - f1r;  vvi[(t)+3] = e1i - f1i;                      \
        vd u0r = e0r + f0r, u0i = e0i + f0i;                                  \
        vd u1r = e1r + f1r, u1i = e1i + f1i;                                  \
        vd pr = u0r + u1r, pi = u0i + u1i;                                    \
        vd qr = u0r - u1r, qi = u0i - u1i;                                    \
        dcr += pr;  dci += pi;                                                \
        BODY                                                                  \
    } while (0)

    FOLDR(0,  1,12,  8, 5,
        a0r = x0r + pr*kCP0;  a1r = x0r + pr*kCP1;  a2r = x0r + pr*kCP2;
        a0i = x0i + pi*kCP0;  a1i = x0i + pi*kCP1;  a2i = x0i + pi*kCP2;
        b0r = qr*kCM0;  b1r = qr*kCM1;  b2r = qr*kCM2;
        b0i = qi*kCM0;  b1i = qi*kCM1;  b2i = qi*kCM2; );
    FOLDR(1,  2,11,  3,10,
        a0r += pr*kCP1;  a1r += pr*kCP2;  a2r += pr*kCP0;
        a0i += pi*kCP1;  a1i += pi*kCP2;  a2i += pi*kCP0;
        b0r += qr*kCM1;  b1r += qr*kCM2;  b2r -= qr*kCM0;
        b0i += qi*kCM1;  b1i += qi*kCM2;  b2i -= qi*kCM0; );
    FOLDR(2,  4, 9,  6, 7,
        a0r += pr*kCP2;  a1r += pr*kCP0;  a2r += pr*kCP1;
        a0i += pi*kCP2;  a1i += pi*kCP0;  a2i += pi*kCP1;
        b0r += qr*kCM2;  b1r -= qr*kCM0;  b2r -= qr*kCM1;
        b0i += qi*kCM2;  b1i -= qi*kCM0;  b2i -= qi*kCM1; );
#undef FOLDR

    yr[0] = dcr;  yi[0] = dci;

    vd cc0r = a0r + b0r, cc0i = a0i + b0i, cc3r = a0r - b0r, cc3i = a0i - b0i;
    vd cc1r = a1r + b1r, cc1i = a1i + b1i, cc4r = a1r - b1r, cc4i = a1i - b1i;
    vd cc2r = a2r + b2r, cc2i = a2i + b2i, cc5r = a2r - b2r, cc5i = a2i - b2i;

    const vd kSN0 = KC(SN0), kSN1 = KC(SN1), kSN2 = KC(SN2),
             kSN3 = KC(SN3), kSN4 = KC(SN4), kSN5 = KC(SN5);

#define SR1(m, w0,w1,w2) do {                                                 \
        vd tr_ = vvr[m], ti_ = vvi[m];                                        \
        a0r = tr_*(w0);  a1r = tr_*(w1);  a2r = tr_*(w2);                     \
        a0i = ti_*(w0);  a1i = ti_*(w1);  a2i = ti_*(w2);                     \
    } while (0)
#define SR(m, s0,w0, s1,w1, s2,w2) do {                                       \
        vd tr_ = vvr[m], ti_ = vvi[m];                                        \
        a0r s0 tr_*(w0);  a1r s1 tr_*(w1);  a2r s2 tr_*(w2);                  \
        a0i s0 ti_*(w0);  a1i s1 ti_*(w1);  a2i s2 ti_*(w2);                  \
    } while (0)

    SR1(0,     kSN0,    kSN1,    kSN2);
    SR (1, +=, kSN1, +=,kSN2, +=,kSN3);
    SR (2, +=, kSN2, +=,kSN3, +=,kSN4);
    SR (3, +=, kSN3, +=,kSN4, +=,kSN5);
    SR (4, +=, kSN4, +=,kSN5, -=,kSN0);
    SR (5, +=, kSN5, -=,kSN0, -=,kSN1);

    yr[ 1] = cc0r - a0i;  yi[ 1] = cc0i + a0r;
    yr[12] = cc0r + a0i;  yi[12] = cc0i - a0r;
    yr[ 2] = cc1r - a1i;  yi[ 2] = cc1i + a1r;
    yr[11] = cc1r + a1i;  yi[11] = cc1i - a1r;
    yr[ 4] = cc2r - a2i;  yi[ 4] = cc2i + a2r;
    yr[ 9] = cc2r + a2i;  yi[ 9] = cc2i - a2r;

    SR1(0,     kSN3,    kSN4,    kSN5);
    SR (1, +=, kSN4, +=,kSN5, -=,kSN0);
    SR (2, +=, kSN5, -=,kSN0, -=,kSN1);
    SR (3, -=, kSN0, -=,kSN1, -=,kSN2);
    SR (4, -=, kSN1, -=,kSN2, -=,kSN3);
    SR (5, -=, kSN2, -=,kSN3, -=,kSN4);
#undef SR
#undef SR1

    yr[ 8] = cc3r - a0i;  yi[ 8] = cc3i + a0r;
    yr[ 5] = cc3r + a0i;  yi[ 5] = cc3i - a0r;
    yr[ 3] = cc4r - a1i;  yi[ 3] = cc4i + a1r;
    yr[10] = cc4r + a1i;  yi[10] = cc4i - a1r;
    yr[ 6] = cc5r - a2i;  yi[ 6] = cc5i + a2r;
    yr[ 7] = cc5r + a2i;  yi[ 7] = cc5i - a2r;
}

#if VW == 8
/* 8x8 zmm transpose, 24 two-source shuffles (3 stages x 8). */
#if defined(__clang__)
#  define SH8(a,b,i0,i1,i2,i3,i4,i5,i6,i7) \
        __builtin_shufflevector((a),(b),i0,i1,i2,i3,i4,i5,i6,i7)
#else
#  define SH8(a,b,i0,i1,i2,i3,i4,i5,i6,i7) \
        __builtin_shuffle((a),(b),(vl){i0,i1,i2,i3,i4,i5,i6,i7})
#endif
#define TR8(r0,r1,r2,r3,r4,r5,r6,r7, o0,o1,o2,o3,o4,o5,o6,o7) do {            \
        vd _a0 = SH8(r0,r1, 0, 8,2,10,4,12,6,14);                             \
        vd _b0 = SH8(r0,r1, 1, 9,3,11,5,13,7,15);                             \
        vd _a1 = SH8(r2,r3, 0, 8,2,10,4,12,6,14);                             \
        vd _b1 = SH8(r2,r3, 1, 9,3,11,5,13,7,15);                             \
        vd _a2 = SH8(r4,r5, 0, 8,2,10,4,12,6,14);                             \
        vd _b2 = SH8(r4,r5, 1, 9,3,11,5,13,7,15);                             \
        vd _a3 = SH8(r6,r7, 0, 8,2,10,4,12,6,14);                             \
        vd _b3 = SH8(r6,r7, 1, 9,3,11,5,13,7,15);                             \
        vd _c0 = SH8(_a0,_a1, 0,1, 8, 9,4,5,12,13);                           \
        vd _c2 = SH8(_a0,_a1, 2,3,10,11,6,7,14,15);                           \
        vd _d0 = SH8(_b0,_b1, 0,1, 8, 9,4,5,12,13);                           \
        vd _d2 = SH8(_b0,_b1, 2,3,10,11,6,7,14,15);                           \
        vd _c4 = SH8(_a2,_a3, 0,1, 8, 9,4,5,12,13);                           \
        vd _c6 = SH8(_a2,_a3, 2,3,10,11,6,7,14,15);                           \
        vd _d4 = SH8(_b2,_b3, 0,1, 8, 9,4,5,12,13);                           \
        vd _d6 = SH8(_b2,_b3, 2,3,10,11,6,7,14,15);                           \
        o0 = SH8(_c0,_c4, 0,1,2,3, 8, 9,10,11);                               \
        o4 = SH8(_c0,_c4, 4,5,6,7,12,13,14,15);                               \
        o1 = SH8(_d0,_d4, 0,1,2,3, 8, 9,10,11);                               \
        o5 = SH8(_d0,_d4, 4,5,6,7,12,13,14,15);                               \
        o2 = SH8(_c2,_c6, 0,1,2,3, 8, 9,10,11);                               \
        o6 = SH8(_c2,_c6, 4,5,6,7,12,13,14,15);                               \
        o3 = SH8(_d2,_d6, 0,1,2,3, 8, 9,10,11);                               \
        o7 = SH8(_d2,_d6, 4,5,6,7,12,13,14,15);                               \
    } while (0)

/* Fused z pass over 8 GLOBAL rows g = kx*13 + y of the split A volume:
 * transposing 8x16 loads (row r at apr + soff[r]; the 3-double tail of each
 * 16 reads into the next row / plane pad, harmless), the 13-point kernel on
 * registers, transposing stores into U rows (row r at ur0 + doff[r], which
 * encodes both the y*TR offset and WHICH double-buffered U plane; columns
 * 13..15 get deterministic garbage nothing reads).  The 8 transposes ride
 * inside the kernel's FMA stream (L17_rader panel_r3 "next" item 2, built in
 * panel_r6); this replaces both standalone transpose13 passes, the T buffer,
 * and every scalar edge op.  Blocks straddle kx-plane boundaries (this
 * round's change), so the volume's 169 z-rows take 22 blocks, not 13x2=26:
 * the per-plane form idled 3 of 16 lane-slots per plane. */
static inline __attribute__((always_inline)) void zkern13_rows(
        const double *apr, const double *api, double *ur0, double *ui0,
        const long *soff, const long *doff)
{
    vd xr[16], xi[16], q[16];
    for (int r = 0; r < 8; ++r) {
        const double *row = apr + soff[r];
        q[r]     = VL_(row);
        q[8 + r] = VL_(row + 8);
    }
    TR8(q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],
        xr[0],xr[1],xr[2],xr[3],xr[4],xr[5],xr[6],xr[7]);
    TR8(q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],
        xr[8],xr[9],xr[10],xr[11],xr[12],xr[13],xr[14],xr[15]);
    for (int r = 0; r < 8; ++r) {
        const double *row = api + soff[r];
        q[r]     = VL_(row);
        q[8 + r] = VL_(row + 8);
    }
    TR8(q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],
        xi[0],xi[1],xi[2],xi[3],xi[4],xi[5],xi[6],xi[7]);
    TR8(q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],
        xi[8],xi[9],xi[10],xi[11],xi[12],xi[13],xi[14],xi[15]);

    vd yr[13], yi[13];
    kern13_regs(xr, xi, yr, yi);

    TR8(yr[0],yr[1],yr[2],yr[3],yr[4],yr[5],yr[6],yr[7],
        q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7]);
    TR8(yr[8],yr[9],yr[10],yr[11],yr[12],yr[12],yr[12],yr[12],
        q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15]);
    for (int r = 0; r < 8; ++r) {
        VS_(ur0 + doff[r],     q[r]);
        VS_(ur0 + doff[r] + 8, q[8 + r]);
    }
    TR8(yi[0],yi[1],yi[2],yi[3],yi[4],yi[5],yi[6],yi[7],
        q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7]);
    TR8(yi[8],yi[9],yi[10],yi[11],yi[12],yi[12],yi[12],yi[12],
        q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15]);
    for (int r = 0; r < 8; ++r) {
        VS_(ui0 + doff[r],     q[r]);
        VS_(ui0 + doff[r] + 8, q[8 + r]);
    }
}
#endif /* VW == 8 */

/* ------------------------------------------------------------------- plan */

struct fft3d_plan {
    int batch;
    double *mem;
    double *ar, *ai;     /* split A[kx][y*13+z], plane pitch PS             */
    double *tr[2], *ti[2]; /* plane buffers T[z][y], row stride TR, zero
                            * pads; DOUBLE-BUFFERED so consecutive kx planes
                            * have no WAR hazard and the core overlaps them  */
    double *ur[2], *ui[2]; /* plane buffers U[y][kz], double-buffered        */
    double *sb[2];       /* interleaved staging planes for the batched path  */
    int ys;              /* y pass stages planes and burst-copies to out     */
    int pf;              /* cross-volume input prefetch (only past this
                          * machine's L3: it costs ~4% when the batch is
                          * L3-resident, wallaby B=512 measured)             */
    int pw;              /* prefetchw the next out plane ahead of its staged
                          * burst copy (hide the RFO, don't NT-avoid it --
                          * the panel-wide r5 verdict; gated like pf because
                          * pfw measured 17% WORSE when out is L3-resident,
                          * L6_unrolled r3)                                  */
    /* z-pass block table, built at plan time: block b transforms the 8
     * global z-rows g = kx*13+y at A offsets zs[b][r], storing to U at
     * offsets zd[b][r] (relative to ur[0]/ui[0]; the offset encodes which
     * double-buffered U plane).  znc[b] = kx planes fully in U after block
     * b, i.e. how far the y pass may advance.  L13R_ZG=1 (default): 22
     * plane-straddling blocks; L13R_ZG=0: the old 26 per-plane blocks. */
    int znb;
    int znc[26];
    long zs[26][8], zd[26][8];
};

const char *fft3d_name(void) { return "L13_rader"; }
const char *fft3d_description(void)
{
    return "Rader-13 as split cyclic/negacyclic correlations (186 FP/pt), "
           "X-first, global-row fused z-kernel, direct+prefetchw out, "
#if VW == 8
           "512-bit";
#else
           "256-bit";
#endif
}
int fft3d_supports(int L) { return L == LN; }

/* lane-block starts for a 13-wide lane space whose stores must not overrun
 * lane 12: the last block overlaps and recomputes, which is free because the
 * block count is unchanged (needs out-of-place, which the y pass is). */
static const int LBOFF[NLB] = {
#if VW == 8
    0, 5
#else
    0, 4, 8, 9
#endif
};

fft3d_plan *fft3d_create(int L, int batch)
{
    if (L != LN || batch < 1) return NULL;
    fft3d_plan *p = calloc(1, sizeof *p);
    if (!p) return NULL;
    p->batch = batch;
    size_t nd = 2*(size_t)LN*PS + 8*(size_t)LN*TR + 2*352;
    void *m = NULL;
    if (posix_memalign(&m, 64, nd * sizeof(double))) { free(p); return NULL; }
    memset(m, 0, nd * sizeof(double));        /* pad lanes stay zero forever */
    p->mem = m;
    double *q = p->mem;
    p->ar = q;  q += (size_t)LN*PS;
    p->ai = q;  q += (size_t)LN*PS;
    for (int h = 0; h < 2; ++h) {
        p->tr[h] = q;  q += (size_t)LN*TR;
        p->ti[h] = q;  q += (size_t)LN*TR;
        p->ur[h] = q;  q += (size_t)LN*TR;
        p->ui[h] = q;  q += (size_t)LN*TR;
    }
    for (int h = 0; h < 2; ++h) { p->sb[h] = q;  q += 352; }
    /* Store policy, settled by round panel_r8's A/Bs: DIRECT y-pass stores
     * plus prefetchw of the out plane one pipeline step ahead beat the r6/r7
     * staged burst copy in every streaming regime on wallaby (B=512: 1923 us
     * direct+pw vs 2032 staged, B=2048: 10774 vs 11483), and the node's r7
     * leaderboard showed staging alone costing B=16 +20% over B=1 (7.279 vs
     * 6.054 us, ys the only difference) and B=512 losing the cell to MKL.
     * So ys defaults OFF everywhere (kept only as a FORCE knob), and pw --
     * which staged-r7 gated on L3 -- moves to the L2 gate: direct stores
     * without pw are catastrophic as soon as out streams past L2 (wallaby
     * B=512: 2386 us direct-no-pw vs 1923 with).  pw when everything is
     * L2-resident costs ~2% (B=1: 3.25 vs 3.18), hence the gate. */
#ifdef L13R_FORCE_YS
    p->ys = L13R_FORCE_YS;
#else
    p->ys = 0;
#endif
    {
        long l2 = 0;
#ifdef _SC_LEVEL2_CACHE_SIZE
        l2 = sysconf(_SC_LEVEL2_CACHE_SIZE);
#endif
        if (l2 <= 0) l2 = 1 << 20;
#ifdef L13R_FORCE_PW
        p->pw = L13R_FORCE_PW;
#else
        p->pw = ((size_t)batch * NVOL * 32 > (size_t)l2);
#endif
    }
#ifdef L13R_FORCE_PF
    p->pf = L13R_FORCE_PF;
#else
    {
        long l3 = 0;
#ifdef _SC_LEVEL3_CACHE_SIZE
        l3 = sysconf(_SC_LEVEL3_CACHE_SIZE);
#endif
        if (l3 <= 0) l3 = 22l << 20;
        p->pf = ((size_t)batch * NVOL * 32 > (size_t)l3);
    }
#endif
#ifndef L13R_ZG
#define L13R_ZG 1
#endif
    {
        const long UD = 4*(long)LN*TR;      /* p->ur[1] - p->ur[0] */
        p->znb = L13R_ZG ? 22 : 26;
        for (int blk = 0; blk < p->znb; ++blk) {
            long g0;
            if (L13R_ZG) {
                g0 = 8*(long)blk;
                if (g0 > 161) g0 = 161;     /* overlap block, recomputes    */
                p->znc[blk] = (int)((g0 + 8) / LN);
            } else {
                g0 = LN*(long)(blk >> 1) + ((blk & 1) ? 5 : 0);
                p->znc[blk] = (blk + 1) >> 1;
            }
            for (int r = 0; r < 8; ++r) {
                long g = g0 + r, kx = g / LN, y = g - LN*kx;
                p->zs[blk][r] = kx*PS + y*LN;
                p->zd[blk][r] = ((kx & 1) ? UD : 0) + y*TR;
            }
        }
    }
    return p;
}

/* Dev-only in-situ phase cycle counters (-DL13R_TSC): prints per-volume
 * cycles per phase at destroy time.  Adds rdtsc overhead; never on for a
 * scored build. */
#ifdef L13R_TSC
#include <stdio.h>
#include <x86intrin.h>
static unsigned long long l13r_ph[6], l13r_nv;
#define TSC_T0 unsigned long long _t0 = __rdtsc(), _t1
#define TSC_P(i) do { _t1 = __rdtsc(); l13r_ph[i] += _t1 - _t0; _t0 = _t1; } while (0)
#else
#define TSC_T0 do {} while (0)
#define TSC_P(i) do {} while (0)
#endif

void fft3d_execute(fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    double *const ar = p->ar, *const ai = p->ai;
    const int ys = p->ys, pw = p->pw;

    for (int b = 0; b < p->batch; ++b) {
        const double *src = (const double *)in + (size_t)2*NVOL*b;
        double *dst       = (double *)out      + (size_t)2*NVOL*b;
        TSC_T0;

        /* x pass FIRST: interleaved loads straight from `in` (lanes m =
         * y*13+z, 169 contiguous), aligned split stores into A[kx][y][z].
         * X-first, so the final interleaved `out` stores happen per kx
         * plane inside a 2.7 KB window instead of strided across the whole
         * volume (L17_matrixsimd panel_r3's reorder; measured here: the
         * X-last direct store cost 4.8 us of a 7.5 us volume). */
#if defined(L13R_X2) && VW == 8
        /* Two independent blocks per loop body: gcc schedules the two
         * inlined kernels' load streams together, doubling the MLP against
         * the 2704 B-strided in rows at the price of spills (56 live vs 32
         * regs).  -DL13R_X2 to enable. */
        for (int blk = 0; blk + 2 < NXB; blk += 2) {
            long m0 = (long)blk * VW;
            kern13(0, 0, 0, src, m0,      ar + m0,      ai + m0,      0, 0, PS, 1, 0);
            kern13(0, 0, 0, src, m0 + VW, ar + m0 + VW, ai + m0 + VW, 0, 0, PS, 1, 0);
        }
        /* last pair: blocks 20 (m0=160) and 21 (clamped to 161); the 7
         * shared lanes are recomputed bit-identically, as before */
        kern13(0, 0, 0, src, (NXB-2)*VW,  ar + (NXB-2)*VW, ai + (NXB-2)*VW, 0, 0, PS, 1, 0);
        kern13(0, 0, 0, src, NPL - VW,    ar + NPL - VW,   ai + NPL - VW,   0, 0, PS, 1, 0);
#else
        for (int blk = 0; blk < NXB; ++blk) {
            long m0 = (long)blk * VW;
            if (m0 > NPL - VW) m0 = NPL - VW;
#ifdef L13R_XPF
            if (blk + 1 < NXB) {
                const double *pp = src + 2*(m0 + VW);
                for (int k = 0; k < LN; ++k) {
                    __builtin_prefetch(pp + 2*(long)k*NPL,     0, 3);
                    __builtin_prefetch(pp + 2*(long)k*NPL + 8, 0, 3);
                }
            }
#endif
            kern13(0, 0, 0, src, m0, ar + m0, ai + m0, 0, 0, PS, 1, 0);
        }
#endif
        TSC_P(0);

        /* Finish kx planes through U, SOFTWARE-PIPELINED: z blocks run over
         * GLOBAL rows g = kx*13+y (22 blocks of 8 for the volume's 169
         * z-rows -- straddling plane boundaries reclaims the 3 idle lanes
         * of the old 2-blocks-per-plane form), and each plane's y pass
         * fires the moment its U completes, so the shuffle-heavy z stage
         * and the FMA-heavy y stage still share ports. */
#if VW == 8
#define YSTEP(kx) do {                                                        \
        double *const _ur = p->ur[(kx) & 1], *const _ui = p->ui[(kx) & 1];    \
        double *_dp = dst + 2*(long)(kx)*NPL;                                 \
        double *_tp = ys ? p->sb[(kx) & 1] : _dp;                             \
        for (int lb = 0; lb < NLB; ++lb) {                                    \
            const long m0 = LBOFF[lb];                                        \
            kern13(_ur + m0, _ui + m0, TR, 0, 0, 0, 0, _tp, m0, LN, 0, 1);    \
        }                                                                     \
        if (ys) memcpy(_dp, _tp, (size_t)NPL * 16);                           \
    } while (0)

        /* prefetchw plane kx of `out` one full z+y pipeline step before
         * YSTEP(kx)'s burst copy stores it (~0.35 us of lead on the node),
         * so the copy hits M-state lines instead of paying a cold RFO. */
#define PWSTEP(kx) do {                                                       \
        if (pw) {                                                             \
            const double *_fp = dst + 2*(long)(kx)*NPL;                       \
            for (int q0 = 0; q0 < 2*NPL; q0 += 8)                             \
                __builtin_prefetch(_fp + q0, 1, 3);                           \
        }                                                                     \
    } while (0)

        /* cross-volume input prefetch (L17_winograd round 2): while the
         * plane pipeline computes, pull the next volume's `in` up through
         * L2 in 13 slices; free at B=1 via the NULL guard. */
        const double *nxt =
            (p->pf && b + 1 < p->batch) ? src + (size_t)2*NVOL : 0;

        PWSTEP(0);
        {
            double *const ur0 = p->ur[0], *const ui0 = p->ui[0];
            const int znb = p->znb;
            int ydone = 0;
            for (int blk = 0; blk < znb; ++blk) {
                zkern13_rows(ar, ai, ur0, ui0, p->zs[blk], p->zd[blk]);
                /* planes with all 13 z-rows now done through this block */
                const int ncomp = p->znc[blk];
                while (ydone < ncomp) {
                    if (nxt) {
                        const double *pp = nxt + 2*(long)ydone*NPL;
                        for (int q0 = 0; q0 < 2*NPL; q0 += 8)
                            __builtin_prefetch(pp + q0, 0, 2);
                    }
                    if (ydone + 1 < LN) PWSTEP(ydone + 1);
                    YSTEP(ydone);
                    ydone++;
                }
            }
        }
        TSC_P(2);
#undef PWSTEP
#undef YSTEP
#else  /* VW != 8: reference-quality path via T/U plane transposes */
        for (int kx = 0; kx < LN; ++kx) {
            double *const tr = p->tr[kx & 1], *const ti = p->ti[kx & 1];
            double *const ur = p->ur[kx & 1], *const ui = p->ui[kx & 1];

            if (pw) {
                const double *fp = dst + 2*(long)kx*NPL;
                for (int q0 = 0; q0 < 2*NPL; q0 += 8)
                    __builtin_prefetch(fp + q0, 1, 3);
            }

            /* A[kx][y][z] -> T[z][y], split re/im */
            transpose13(ar + (long)kx*PS, LN, tr, TR);
            transpose13(ai + (long)kx*PS, LN, ti, TR);

            /* z pass, in place on T: axis stride TR, lanes = y; aligned
             * blocks cover the zero pad columns, which map to zeros. */
            for (int o = 0; o < TR; o += VW)
                kern13(tr + o, ti + o, TR, 0, 0, tr + o, ti + o, 0, 0, TR, 0, 0);

            transpose13(tr, TR, ur, TR);       /* T[kz][y] -> U[y][kz] */
            transpose13(ti, TR, ui, TR);

            /* y pass: axis stride TR, lanes = kz, interleaving store into
             * out[kx][ky][kz] (row stride 13 complex; the overlap block
             * recomputes lanes bit-identically). */
            double *dp = dst + 2*(long)kx*NPL;
            double *tp = ys ? p->sb[kx & 1] : dp;
            for (int lb = 0; lb < NLB; ++lb) {
                const long m0 = LBOFF[lb];
                kern13(ur + m0, ui + m0, TR, 0, 0, 0, 0, tp, m0, LN, 0, 1);
            }
            if (ys) memcpy(dp, tp, (size_t)NPL * 16);
        }
#endif /* VW */
#ifdef L13R_TSC
        l13r_nv++;
#endif
    }
}

void fft3d_destroy(fft3d_plan *p)
{
    if (!p) return;
#ifdef L13R_TSC
    if (l13r_nv) {
        const char *nm[5] = {"xpass","transA","zpass","transU","ypass"};
        fprintf(stderr, "L13R_TSC per volume (%llu volumes):\n", l13r_nv);
        for (int i = 0; i < 5; ++i)
            fprintf(stderr, "  %-6s %8.0f cyc\n", nm[i],
                    (double)l13r_ph[i] / (double)l13r_nv);
    }
#endif

    free(p->mem);
    free(p);
}
