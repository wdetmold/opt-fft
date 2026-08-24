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
 *     double-buffered per-plane U[y][kz]; at B=1 ONLY, each z block also
 *     carries one y lane-block port-fused in the same instruction stream
 *     (panel_r9's zykern13_f -- the node priced it -0.7% at B=1 but +4.2%
 *     at B=16 / +2.8% at B=512, so panel_r10 rolled it back at batch);
 *   y pass per kx plane (lanes = kz), fired the moment that plane's U
 *     completes, interleaving stores DIRECTLY into out, with the out plane
 *     prefetchw'd one pipeline step ahead whenever in+out stream past this
 *     machine's L2 (the r6/r7 staged burst copy lost to direct+pfw in every
 *     streaming regime once pfw hid the RFO; panel_r8).
 * Everything is plain GNU C vector extensions -- no intrinsics -- so the same
 * source builds on AVX-512 (VW=8, node/wallaby), AVX2 (VW=4, wombat) and
 * anything gcc can emulate.
 *
 * Since panel_r11, create() runs an IN-PLAN TIMED RACE of the knob variants
 * the panel could never A/B on the node (fused-vs-unfused at B=1; pw, and pf
 * when engaged, at the full scored batch), on private buffers, and both
 * reports the readings through fft3d_description() and adopts a challenger
 * that beats the sysconf incumbent by >3% (L6_pfa's incumbency rule).  Every
 * raced pair is output-BIT-IDENTICAL (fused/unfused verified by cmp in r9;
 * pw/pf are pure prefetches), so a flip can change time but never results.
 * Instrument pattern from L13_direct r10 <- L6_unrolled r9 <- L36_pfa r8.
 *
 * Contract: ../fft3d_api.h.  Strategy record: ../strategies/L13_rader.md.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
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
/* Batched-loop prefetch pacing: 1 (default) = pfw before the plane's y
 * stores, input pf slice AFTER them (L13_direct's placement); 0 = both
 * together before the y stores (the r8 placement). */
#ifndef L13R_PACE
#define L13R_PACE 1
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

/* zykern13_f: one z block FUSED at source-stage level with one y lane-block
 * (round panel_r9).  Rationale: a z block is ~450 uops of dependency-SERIAL
 * port-homogeneous phases (96 shuffles -> 186 FMA -> 96 shuffles), larger
 * than the ROB, so port 5 and port 0 idle alternately and the block costs
 * ~378 cycles on the node instead of max(186,192) -- that serial model
 * predicts 17.2k cycles/volume, and the node measured 17.6k (r8: 6.074 us
 * at 2.89 GHz) against a 13.0k port floor.  Interleaving a y lane-block
 * (186 FMA, ~26 shuffles) in ~60-130-uop stages between the z stages puts
 * FMA work inside the OOO window of every z transpose burst.  The z kernel
 * math is restructured into independent re/im halves (Z_HALF) so its stages
 * can be spaced out: the components only meet in the final +-i*SS mix.
 * Operation trees are unchanged from kern13_regs -- outputs bit-identical.
 *
 * The y lane-block is the same math as kern13(lmode 0, smode 1); the y
 * consumption schedule (one lane-block per fused call, planned in
 * fft3d_create as p->zy[]) lags the z writes by up to ~4 planes, so U is
 * 8-deep double-buffered (p->ur[0..7]; 4-deep fails -- z block 17 touches
 * plane 11 in the call that consumes plane 7's y, same buffer mod 4;
 * create() verifies the invariant and falls back to the unfused loop if
 * the schedule ever violates it). */
#define ZK(c) KPIN(c)

/* One component (re or im) of the 13-point kernel: X[0..12] in, O[13] out as
 * {dc, cc0..cc5, s0..s5}.  Same operation trees as kern13_regs, so results
 * are bit-identical; only the re/im interleaving order differs. */
#define Z_HALF(X, O) do {                                                     \
    const vd cP0 = ZK(CP0), cP1 = ZK(CP1), cP2 = ZK(CP2);                     \
    const vd cM0 = ZK(CM0), cM1 = ZK(CM1), cM2 = ZK(CM2);                     \
    vd x0 = (X)[0], dc = x0;                                                  \
    vd v0,v1,v2,v3,v4,v5, a0,a1,a2, b0,b1,b2;                                 \
    { vd e0 = (X)[1], f0 = (X)[12], e1 = (X)[8], f1 = (X)[5];                 \
      v0 = e0 - f0;  v3 = e1 - f1;                                            \
      vd u0 = e0 + f0, u1 = e1 + f1;                                          \
      vd pp = u0 + u1, qq = u0 - u1;  dc += pp;                               \
      a0 = x0 + pp*cP0;  a1 = x0 + pp*cP1;  a2 = x0 + pp*cP2;                 \
      b0 = qq*cM0;  b1 = qq*cM1;  b2 = qq*cM2; }                              \
    { vd e0 = (X)[2], f0 = (X)[11], e1 = (X)[3], f1 = (X)[10];                \
      v1 = e0 - f0;  v4 = e1 - f1;                                            \
      vd u0 = e0 + f0, u1 = e1 + f1;                                          \
      vd pp = u0 + u1, qq = u0 - u1;  dc += pp;                               \
      a0 += pp*cP1;  a1 += pp*cP2;  a2 += pp*cP0;                             \
      b0 += qq*cM1;  b1 += qq*cM2;  b2 -= qq*cM0; }                           \
    { vd e0 = (X)[4], f0 = (X)[9], e1 = (X)[6], f1 = (X)[7];                  \
      v2 = e0 - f0;  v5 = e1 - f1;                                            \
      vd u0 = e0 + f0, u1 = e1 + f1;                                          \
      vd pp = u0 + u1, qq = u0 - u1;  dc += pp;                               \
      a0 += pp*cP2;  a1 += pp*cP0;  a2 += pp*cP1;                             \
      b0 += qq*cM2;  b1 -= qq*cM0;  b2 -= qq*cM1; }                           \
    (O)[0] = dc;                                                              \
    (O)[1] = a0 + b0;  (O)[4] = a0 - b0;                                      \
    (O)[2] = a1 + b1;  (O)[5] = a1 - b1;                                      \
    (O)[3] = a2 + b2;  (O)[6] = a2 - b2;                                      \
    const vd sn0 = ZK(SN0), sn1 = ZK(SN1), sn2 = ZK(SN2),                     \
             sn3 = ZK(SN3), sn4 = ZK(SN4), sn5 = ZK(SN5);                     \
    a0 = v0*sn0;   a1 = v0*sn1;   a2 = v0*sn2;                                \
    a0 += v1*sn1;  a1 += v1*sn2;  a2 += v1*sn3;                               \
    a0 += v2*sn2;  a1 += v2*sn3;  a2 += v2*sn4;                               \
    a0 += v3*sn3;  a1 += v3*sn4;  a2 += v3*sn5;                               \
    a0 += v4*sn4;  a1 += v4*sn5;  a2 -= v4*sn0;                               \
    a0 += v5*sn5;  a1 -= v5*sn0;  a2 -= v5*sn1;                               \
    (O)[7] = a0;  (O)[8] = a1;  (O)[9] = a2;                                  \
    a0 = v0*sn3;   a1 = v0*sn4;   a2 = v0*sn5;                                \
    a0 += v1*sn4;  a1 += v1*sn5;  a2 -= v1*sn0;                               \
    a0 += v2*sn5;  a1 -= v2*sn0;  a2 -= v2*sn1;                               \
    a0 -= v3*sn0;  a1 -= v3*sn1;  a2 -= v3*sn2;                               \
    a0 -= v4*sn1;  a1 -= v4*sn2;  a2 -= v4*sn3;                               \
    a0 -= v5*sn2;  a1 -= v5*sn3;  a2 -= v5*sn4;                               \
    (O)[10] = a0;  (O)[11] = a1;  (O)[12] = a2;                               \
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

/* The fused call (see the block comment above Z_HALF).  restrict is load-
 * bearing: without it gcc must keep the y stores to `out` ordered against
 * the z loads from A and the interleave dies at compile time.  All access
 * sets are disjoint by construction (A vs U vs out; the y-read U buffer is
 * never a plane the z part writes -- create() verifies the schedule). */
static inline __attribute__((always_inline)) void zykern13_f(
        const double *restrict apr, const double *restrict api,
        double *restrict ur0, double *restrict ui0,
        const long *restrict soff, const long *restrict doff,
        const double *restrict yxr, const double *restrict yxi,
        double *restrict ydst, long ym0)
{
    vd q[16], zx[16];

    /* -- stage 1: z re transposing loads (16 loads, 48 shuffles) -- */
    for (int r = 0; r < 8; ++r) {
        const double *row = apr + soff[r];
        q[r]     = VL_(row);
        q[8 + r] = VL_(row + 8);
    }
    TR8(q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],
        zx[0],zx[1],zx[2],zx[3],zx[4],zx[5],zx[6],zx[7]);
    TR8(q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],
        zx[8],zx[9],zx[10],zx[11],zx[12],zx[13],zx[14],zx[15]);

#define YLD(k, vr, vi) do {                                                   \
        (vr) = VL_(yxr + (long)(k)*TR);  (vi) = VL_(yxi + (long)(k)*TR);      \
    } while (0)
#define YST(k, vr, vi) do {                                                   \
        double *_p = ydst + 2*((long)(k)*LN + ym0);                           \
        *(vd *)(void *)(_p)      = ILO((vr),(vi));                            \
        *(vd *)(void *)(_p + VW) = IHI((vr),(vi));                            \
    } while (0)

    /* -- stage 2: y loads + fold + cyclic (26 loads, ~105 FP) -- */
    vd yvr[6], yvi[6];
    vd ycc0r,ycc1r,ycc2r,ycc3r,ycc4r,ycc5r;
    vd ycc0i,ycc1i,ycc2i,ycc3i,ycc4i,ycc5i;
    {
        const vd kCP0 = ZK(CP0), kCP1 = ZK(CP1), kCP2 = ZK(CP2);
        const vd kCM0 = ZK(CM0), kCM1 = ZK(CM1), kCM2 = ZK(CM2);
        vd x0r, x0i;
        YLD(0, x0r, x0i);
        vd dcr = x0r, dci = x0i;
        vd a0r,a1r,a2r, a0i,a1i,a2i, b0r,b1r,b2r, b0i,b1i,b2i;
#define YFOLD(t, j0,k0, j1,k1, BODY) do {                                     \
        vd e0r,e0i,f0r,f0i,e1r,e1i,f1r,f1i;                                   \
        YLD(j0,e0r,e0i); YLD(k0,f0r,f0i); YLD(j1,e1r,e1i); YLD(k1,f1r,f1i);   \
        yvr[t]     = e0r - f0r;  yvi[t]     = e0i - f0i;                      \
        yvr[(t)+3] = e1r - f1r;  yvi[(t)+3] = e1i - f1i;                      \
        vd u0r = e0r + f0r, u0i = e0i + f0i;                                  \
        vd u1r = e1r + f1r, u1i = e1i + f1i;                                  \
        vd pr = u0r + u1r, pi = u0i + u1i;                                    \
        vd qr = u0r - u1r, qi = u0i - u1i;                                    \
        dcr += pr;  dci += pi;                                                \
        BODY                                                                  \
    } while (0)
        YFOLD(0,  1,12,  8, 5,
            a0r = x0r + pr*kCP0;  a1r = x0r + pr*kCP1;  a2r = x0r + pr*kCP2;
            a0i = x0i + pi*kCP0;  a1i = x0i + pi*kCP1;  a2i = x0i + pi*kCP2;
            b0r = qr*kCM0;  b1r = qr*kCM1;  b2r = qr*kCM2;
            b0i = qi*kCM0;  b1i = qi*kCM1;  b2i = qi*kCM2; );
        YFOLD(1,  2,11,  3,10,
            a0r += pr*kCP1;  a1r += pr*kCP2;  a2r += pr*kCP0;
            a0i += pi*kCP1;  a1i += pi*kCP2;  a2i += pi*kCP0;
            b0r += qr*kCM1;  b1r += qr*kCM2;  b2r -= qr*kCM0;
            b0i += qi*kCM1;  b1i += qi*kCM2;  b2i -= qi*kCM0; );
        YFOLD(2,  4, 9,  6, 7,
            a0r += pr*kCP2;  a1r += pr*kCP0;  a2r += pr*kCP1;
            a0i += pi*kCP2;  a1i += pi*kCP0;  a2i += pi*kCP1;
            b0r += qr*kCM2;  b1r -= qr*kCM0;  b2r -= qr*kCM1;
            b0i += qi*kCM2;  b1i -= qi*kCM0;  b2i -= qi*kCM1; );
#undef YFOLD
        YST(0, dcr, dci);
        ycc0r = a0r + b0r;  ycc0i = a0i + b0i;  ycc3r = a0r - b0r;  ycc3i = a0i - b0i;
        ycc1r = a1r + b1r;  ycc1i = a1i + b1i;  ycc4r = a1r - b1r;  ycc4i = a1i - b1i;
        ycc2r = a2r + b2r;  ycc2i = a2i + b2i;  ycc5r = a2r - b2r;  ycc5i = a2i - b2i;
    }

    /* -- stage 3: z re compute (81 FP) -- */
    vd zor[13];
    Z_HALF(zx, zor);

    /* -- stage 4: y negacyclic half 1 + 6 output stores -- */
    {
        const vd kSN0 = ZK(SN0), kSN1 = ZK(SN1), kSN2 = ZK(SN2),
                 kSN3 = ZK(SN3), kSN4 = ZK(SN4), kSN5 = ZK(SN5);
        vd a0r,a1r,a2r, a0i,a1i,a2i;
#define YSA1(m, w0,w1,w2) do {                                                \
        vd tr_ = yvr[m], ti_ = yvi[m];                                        \
        a0r = tr_*(w0);  a1r = tr_*(w1);  a2r = tr_*(w2);                     \
        a0i = ti_*(w0);  a1i = ti_*(w1);  a2i = ti_*(w2);                     \
    } while (0)
#define YSA(m, s0,w0, s1,w1, s2,w2) do {                                      \
        vd tr_ = yvr[m], ti_ = yvi[m];                                        \
        a0r s0 tr_*(w0);  a1r s1 tr_*(w1);  a2r s2 tr_*(w2);                  \
        a0i s0 ti_*(w0);  a1i s1 ti_*(w1);  a2i s2 ti_*(w2);                  \
    } while (0)
        YSA1(0,     kSN0,    kSN1,    kSN2);
        YSA (1, +=, kSN1, +=,kSN2, +=,kSN3);
        YSA (2, +=, kSN2, +=,kSN3, +=,kSN4);
        YSA (3, +=, kSN3, +=,kSN4, +=,kSN5);
        YSA (4, +=, kSN4, +=,kSN5, -=,kSN0);
        YSA (5, +=, kSN5, -=,kSN0, -=,kSN1);
        YST( 1, ycc0r - a0i, ycc0i + a0r);  YST(12, ycc0r + a0i, ycc0i - a0r);
        YST( 2, ycc1r - a1i, ycc1i + a1r);  YST(11, ycc1r + a1i, ycc1i - a1r);
        YST( 4, ycc2r - a2i, ycc2i + a2r);  YST( 9, ycc2r + a2i, ycc2i - a2r);
#undef YSA
#undef YSA1
    }

    /* -- stage 5: z im transposing loads -- */
    for (int r = 0; r < 8; ++r) {
        const double *row = api + soff[r];
        q[r]     = VL_(row);
        q[8 + r] = VL_(row + 8);
    }
    TR8(q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],
        zx[0],zx[1],zx[2],zx[3],zx[4],zx[5],zx[6],zx[7]);
    TR8(q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],
        zx[8],zx[9],zx[10],zx[11],zx[12],zx[13],zx[14],zx[15]);

    /* -- stage 6: y negacyclic half 2 + 6 output stores -- */
    {
        const vd kSN0 = ZK(SN0), kSN1 = ZK(SN1), kSN2 = ZK(SN2),
                 kSN3 = ZK(SN3), kSN4 = ZK(SN4), kSN5 = ZK(SN5);
        vd a0r,a1r,a2r, a0i,a1i,a2i;
#define YSA1(m, w0,w1,w2) do {                                                \
        vd tr_ = yvr[m], ti_ = yvi[m];                                        \
        a0r = tr_*(w0);  a1r = tr_*(w1);  a2r = tr_*(w2);                     \
        a0i = ti_*(w0);  a1i = ti_*(w1);  a2i = ti_*(w2);                     \
    } while (0)
#define YSA(m, s0,w0, s1,w1, s2,w2) do {                                      \
        vd tr_ = yvr[m], ti_ = yvi[m];                                        \
        a0r s0 tr_*(w0);  a1r s1 tr_*(w1);  a2r s2 tr_*(w2);                  \
        a0i s0 ti_*(w0);  a1i s1 ti_*(w1);  a2i s2 ti_*(w2);                  \
    } while (0)
        YSA1(0,     kSN3,    kSN4,    kSN5);
        YSA (1, +=, kSN4, +=,kSN5, -=,kSN0);
        YSA (2, +=, kSN5, -=,kSN0, -=,kSN1);
        YSA (3, -=, kSN0, -=,kSN1, -=,kSN2);
        YSA (4, -=, kSN1, -=,kSN2, -=,kSN3);
        YSA (5, -=, kSN2, -=,kSN3, -=,kSN4);
        YST( 8, ycc3r - a0i, ycc3i + a0r);  YST( 5, ycc3r + a0i, ycc3i - a0r);
        YST( 3, ycc4r - a1i, ycc4i + a1r);  YST(10, ycc4r + a1i, ycc4i - a1r);
        YST( 6, ycc5r - a2i, ycc5i + a2r);  YST( 7, ycc5r + a2i, ycc5i - a2r);
#undef YSA
#undef YSA1
    }
#undef YLD
#undef YST

    /* -- stage 7: z im compute -- */
    vd zoi[13];
    Z_HALF(zx, zoi);

    /* -- stage 8: mix re (X = cc -+ i*SS, re part) + transposing store -- */
    {
        vd m[13];
        m[0]  = zor[0];
        m[1]  = zor[1] - zoi[7];   m[12] = zor[1] + zoi[7];
        m[2]  = zor[2] - zoi[8];   m[11] = zor[2] + zoi[8];
        m[4]  = zor[3] - zoi[9];   m[9]  = zor[3] + zoi[9];
        m[8]  = zor[4] - zoi[10];  m[5]  = zor[4] + zoi[10];
        m[3]  = zor[5] - zoi[11];  m[10] = zor[5] + zoi[11];
        m[6]  = zor[6] - zoi[12];  m[7]  = zor[6] + zoi[12];
        TR8(m[0],m[1],m[2],m[3],m[4],m[5],m[6],m[7],
            q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7]);
        TR8(m[8],m[9],m[10],m[11],m[12],m[12],m[12],m[12],
            q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15]);
        for (int r = 0; r < 8; ++r) {
            VS_(ur0 + doff[r],     q[r]);
            VS_(ur0 + doff[r] + 8, q[8 + r]);
        }
    }

    /* -- stage 9: mix im + transposing store -- */
    {
        vd m[13];
        m[0]  = zoi[0];
        m[1]  = zoi[1] + zor[7];   m[12] = zoi[1] - zor[7];
        m[2]  = zoi[2] + zor[8];   m[11] = zoi[2] - zor[8];
        m[4]  = zoi[3] + zor[9];   m[9]  = zoi[3] - zor[9];
        m[8]  = zoi[4] + zor[10];  m[5]  = zoi[4] - zor[10];
        m[3]  = zoi[5] + zor[11];  m[10] = zoi[5] - zor[11];
        m[6]  = zoi[6] + zor[12];  m[7]  = zoi[6] - zor[12];
        TR8(m[0],m[1],m[2],m[3],m[4],m[5],m[6],m[7],
            q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7]);
        TR8(m[8],m[9],m[10],m[11],m[12],m[12],m[12],m[12],
            q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15]);
        for (int r = 0; r < 8; ++r) {
            VS_(ui0 + doff[r],     q[r]);
            VS_(ui0 + doff[r] + 8, q[8 + r]);
        }
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
    double *ur[8], *ui[8]; /* plane buffers U[y][kz], 8-deep: the fused zy
                            * schedule consumes y lane-blocks up to ~4
                            * planes behind the z writes (4-deep was NOT
                            * enough: z block 17 already touches plane 11
                            * while plane 7's y is being consumed in the
                            * same call -- create() verifies the invariant) */
    double *sb[2];       /* interleaved staging planes for the batched path  */
    int fuse;            /* use the fused zy schedule (VW=8, panel_r9);
                          * B=1 ONLY since panel_r10: the node priced the
                          * fused schedule at -0.7% (B=1) but +4.2% (B=16)
                          * and +2.8% (B=512) against the unfused r8 code,
                          * so it keeps the one cell where it measured ahead
                          * and is rolled back where it lost (r9 VERDICT)    */
    int um;              /* U plane index mask: 7 when fused (the zy
                          * schedule consumes y up to ~4 planes late), 1
                          * unfused (r8's double buffer -- 20 KB less hot
                          * scratch in the streaming cells)                  */
    int zy[26];          /* y lane-block consumed inside fused z call b,
                          * or -1; block j is plane j>>1, lanes LBOFF[j&1]   */
    int ytail;           /* first y lane-block left for the trailing loop    */
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
/* Engaged-configuration string, filled by create(): the r9 round's lesson
 * ("print the engaged configuration before believing any A/B") made
 * fft3d_description() the panel's instrument; buffer pattern from
 * L13_direct. */
static char l13r_desc[240] =
    "Rader-13 split cyclic/negacyclic (186 FP/pt), X-first, "
#if VW == 8
    "512-bit";
#else
    "256-bit";
#endif
const char *fft3d_description(void) { return l13r_desc; }
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

/* Rewrite the z-pass dest-offset table for a given U index mask (7 = 8-deep
 * fused schedule, 1 = r8's double buffer).  kx and y are recovered from the
 * um-independent source table, so this is callable again at race time. */
static void l13r_zdum(fft3d_plan *p, int um)
{
    const long UD = 2*(long)LN*TR;
    p->um = um;
    for (int blk = 0; blk < p->znb; ++blk)
        for (int r = 0; r < 8; ++r) {
            long kx = p->zs[blk][r] / PS;
            long y  = (p->zs[blk][r] - kx*PS) / LN;
            p->zd[blk][r] = (kx & um)*UD + y*TR;
        }
}

static double __attribute__((unused)) l13r_now(void)   /* unused if -DL13R_AB=0 */
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return 1e9*(double)ts.tv_sec + (double)ts.tv_nsec;
}

fft3d_plan *fft3d_create(int L, int batch)
{
    if (L != LN || batch < 1) return NULL;
    fft3d_plan *p = calloc(1, sizeof *p);
    if (!p) return NULL;
    p->batch = batch;
    size_t nd = 2*(size_t)LN*PS + 20*(size_t)LN*TR + 2*352;
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
    }
    for (int h = 0; h < 8; ++h) {             /* ur/ui pairs, stride 2*LN*TR */
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
#ifndef L13R_FUSE
#define L13R_FUSE 1
#endif
    /* The fused zy schedule runs at B=1 only (see the plan comment); the
     * U mask must be fixed before the zd table is built. */
    const int want_fuse = (VW == 8) && L13R_FUSE && !p->ys && batch == 1;
    {
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
            }
        }
        l13r_zdum(p, want_fuse ? 7 : 1);
    }
    /* Fused zy schedule (panel_r9): consume one y lane-block per z call as
     * soon as its plane's U is complete (znc as of the PREVIOUS call).
     * Then verify the 4-deep U reuse invariant: any plane a z call writes
     * must have the plane 4 back fully consumed strictly BEFORE that call.
     * The schedule is a pure function of the block tables; the check is
     * belt-and-braces -- on violation fall back to the unfused loop. */
    {
        int cons = 0, ok = 1;
        for (int blk = 0; blk < p->znb; ++blk) {
            int avail = (blk ? 2*p->znc[blk-1] : 0) - cons;
            p->zy[blk] = (avail > 0) ? cons++ : -1;
        }
        p->ytail = cons;
        cons = 0;
        for (int blk = 0; blk < p->znb; ++blk) {
            for (int r = 0; r < 8; ++r) {
                long kxw = (p->zs[blk][r] / PS) - 8;   /* plane sharing buf */
                if (kxw >= 0) {
                    if (cons < 2*(int)kxw + 2) ok = 0;
                    if (p->zy[blk] >= 0 && (p->zy[blk] >> 1) == (int)kxw) ok = 0;
                }
            }
            if (p->zy[blk] >= 0) cons++;
        }
        p->fuse = want_fuse && ok;
    }

    /* ------- in-plan timed race of the node-undecidable knobs (panel_r11).
     * Pattern: L13_direct r10's ab instrument (<- L6_unrolled r9 <- L36_pfa
     * r8), plus L6_pfa's incumbency rule so it may also FLIP the pick: a
     * challenger is adopted only if it beats the sysconf incumbent by >3%
     * (min over 9 interleaved full-batch samples; node cell spread is
     * ~0.3%, so a real >3% effect is 10 sigma and the pick is stable, and
     * a sub-3% one keeps the deterministic incumbent -- no L36-style
     * lottery).  Raced pairs are output-BIT-IDENTICAL (fused/unfused
     * verified by cmp in r9; pw/pf are pure prefetches), so a flip can
     * never change results.  What it settles: at B=1, the monitor's
     * standing FUSE=0 ask (r10 verdict: "if it reads <=6.03, delete the
     * fused kernel"); at batch, the pw-at-B=16 question (my B16/B1 ratio
     * is 1.127 vs L13_direct's 1.051 and pw is the ONLY config difference
     * between my two cells -- three rounds of asking, never run), raced at
     * the FULL scored batch on private buffers so the cache regime is the
     * scored one.  -DL13R_AB=0 removes it; readings ship in the
     * description either way the pick lands. */
#ifndef L13R_AB
#define L13R_AB 1
#endif
#ifndef L13R_ABTH
#define L13R_ABTH 0.97
#endif
    char abuf[96] = "";
#if L13R_AB
    do {
        struct { int fuse, um, pw, pf; const char *tag; } v[3];
        int nv = 0;
        v[nv].fuse = p->fuse; v[nv].um = p->um; v[nv].pw = p->pw;
        v[nv].pf = p->pf; v[nv].tag = "i"; nv++;
        if (batch == 1) {
            if (!p->fuse) break;           /* nothing to race at B=1       */
            v[nv] = v[0]; v[nv].fuse = 0; v[nv].um = 1; v[nv].tag = "f0"; nv++;
        } else {
            if (batch > 2048) break;       /* bound the private buffers    */
            /* a FORCEd knob is the monitor's A/B: never race it back      */
#ifndef L13R_FORCE_PW
            v[nv] = v[0]; v[nv].pw = !p->pw; v[nv].tag = "pw!"; nv++;
#endif
#ifndef L13R_FORCE_PF
            if (p->pf) { v[nv] = v[0]; v[nv].pf = 0; v[nv].tag = "pf!"; nv++; }
#endif
            if (nv == 1) break;
        }
        size_t vn = (size_t)NVOL * batch;
        double _Complex *ri = 0, *ro = 0;
        if (posix_memalign((void **)&ri, 64, vn * 16)) break;
        if (posix_memalign((void **)&ro, 64, vn * 16)) { free(ri); break; }
        {   /* deterministic fill, values in [-4,4); no libm, no rand()    */
            unsigned long long s = 0x9e3779b97f4a7c15ull;
            double *rd = (double *)ri;
            for (size_t i = 0; i < 2*vn; ++i) {
                s = s*6364136223846793005ull + 1442695040888963407ull;
                rd[i] = (double)(long long)(s >> 12) * 0x1p-50;
            }
        }
        const int reps = (batch == 1) ? 16 : 1;
        /* Two independent trial blocks (laps 0-4 and 5-9), variant order
         * alternating per lap: a challenger is adopted only if it clears
         * the 3% bar in BOTH blocks.  A one-sided clock-mode window (the
         * wallaby bimodality that faked a >3% pw reading in dev) covers at
         * most one block for one variant; a real node effect covers both. */
        double b1[3] = { 1e30, 1e30, 1e30 }, b2[3] = { 1e30, 1e30, 1e30 };
        for (int t = -1; t < 10; ++t)      /* t = -1 is the warmup lap     */
            for (int j = 0; j < nv; ++j) {
                int k = (t & 1) ? nv - 1 - j : j;
                if (p->um != v[k].um) l13r_zdum(p, v[k].um);
                p->fuse = v[k].fuse; p->pw = v[k].pw; p->pf = v[k].pf;
                double t0 = l13r_now();
                for (int r = 0; r < reps; ++r) fft3d_execute(p, ri, ro);
                double dt = (l13r_now() - t0) / reps;
                if (t < 0) continue;
                double *bb = (t < 5) ? b1 : b2;
                if (dt < bb[k]) bb[k] = dt;
            }
        double best[3];
        for (int k = 0; k < nv; ++k) best[k] = b1[k] < b2[k] ? b1[k] : b2[k];
        int win = 0;
        for (int k = 1; k < nv; ++k)
            if (b1[k] < L13R_ABTH * b1[win] && b2[k] < L13R_ABTH * b2[win])
                win = k;
        if (p->um != v[win].um) l13r_zdum(p, v[win].um);
        p->fuse = v[win].fuse; p->pw = v[win].pw; p->pf = v[win].pf;
        {
            int off = snprintf(abuf, sizeof abuf, " ab[B%d]=", batch);
            for (int k = 0; k < nv && off < (int)sizeof abuf; ++k)
                off += snprintf(abuf + off, sizeof abuf - off, "%s%s:%.0f",
                                k ? "," : "", v[k].tag,
                                best[k] / (double)batch);   /* ns/transform */
            if (off < (int)sizeof abuf)
                snprintf(abuf + off, sizeof abuf - off, " pick=%s",
                         v[win].tag);
        }
        free(ri); free(ro);
    } while (0);
#endif
    snprintf(l13r_desc, sizeof l13r_desc,
             "Rader-13 split cyc/nega (186 FP/pt), X-first, %d-bit; "
             "fuse=%d um=%d ys=%d pf=%d pw=%d pace=%d znb=%d%s",
             VW * 64, p->fuse, p->um, p->ys, p->pf, p->pw,
             (int)L13R_PACE, p->znb, abuf);
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
    const int ys = p->ys, pw = p->pw, um = p->um;

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
        double *const _ur = p->ur[(kx) & um], *const _ui = p->ui[(kx) & um];  \
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
        /* pf slice + next-plane prefetchw, fired once per plane.  In the
         * unfused (batched) loop the two halves are now SPLIT around the
         * plane's y stores -- pfw keeps its one-plane lead ahead of the
         * stores, the input slice moves to AFTER them (between plane pl's
         * y and the next z blocks), which is where L13_direct's node-
         * winning pf exec issues its read-side prefetches.  -DL13R_PACE=0
         * restores the r8 joint placement. */
#define PFSLICE(pl) do {                                                      \
        if (nxt) {                                                            \
            const double *pp = nxt + 2*(long)(pl)*NPL;                        \
            for (int q0 = 0; q0 < 2*NPL; q0 += 8)                             \
                __builtin_prefetch(pp + q0, 0, 2);                            \
        }                                                                     \
    } while (0)
#define YPREP(pl) do {                                                        \
        PFSLICE(pl);                                                          \
        if ((pl) + 1 < LN) PWSTEP((pl) + 1);                                  \
    } while (0)
        if (p->fuse) {
            /* fused zy schedule: each z block carries one available y
             * lane-block inside the same instruction stream (see
             * zykern13_f); leftover y lane-blocks trail. */
            double *const ur0 = p->ur[0], *const ui0 = p->ui[0];
            const int znb = p->znb;
            for (int blk = 0; blk < znb; ++blk) {
                const int j = p->zy[blk];
                if (j >= 0) {
                    const int pl = j >> 1;
                    const long m0 = LBOFF[j & 1];
                    if ((j & 1) == 0) YPREP(pl);
                    zykern13_f(ar, ai, ur0, ui0, p->zs[blk], p->zd[blk],
                               p->ur[pl & 7] + m0, p->ui[pl & 7] + m0,
                               dst + 2*(long)pl*NPL, m0);
                } else {
                    zkern13_rows(ar, ai, ur0, ui0, p->zs[blk], p->zd[blk]);
                }
            }
            for (int j = p->ytail; j < 2*LN; ++j) {
                const int pl = j >> 1;
                const long m0 = LBOFF[j & 1];
                if ((j & 1) == 0) YPREP(pl);
                kern13(p->ur[pl & 7] + m0, p->ui[pl & 7] + m0, TR,
                       0, 0, 0, 0, dst + 2*(long)pl*NPL, m0, LN, 0, 1);
            }
        } else {
            double *const ur0 = p->ur[0], *const ui0 = p->ui[0];
            const int znb = p->znb;
            int ydone = 0;
            for (int blk = 0; blk < znb; ++blk) {
                zkern13_rows(ar, ai, ur0, ui0, p->zs[blk], p->zd[blk]);
                /* planes with all 13 z-rows now done through this block */
                const int ncomp = p->znc[blk];
                while (ydone < ncomp) {
#if L13R_PACE
                    if (ydone + 1 < LN) PWSTEP(ydone + 1);
                    YSTEP(ydone);
                    PFSLICE(ydone);
#else
                    YPREP(ydone);
                    YSTEP(ydone);
#endif
                    ydone++;
                }
            }
        }
        TSC_P(2);
#undef YPREP
#undef PFSLICE
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
