/* L45_mixedradix -- forward complex-double 3D DFT of a fixed 45^3 cube,
 * MULTICORE (32-thread) phase.
 *
 * SERIAL KERNEL (unchanged from the single-thread competition's r11 form;
 * see ../../geom/strategies/L45_mixedradix.md for its full history)
 *   Row-column (three 1-D passes) with a Good-Thomas / prime-factor 9x5 line
 *   transform, batch-vectorised across the *lines* of each pass, and blocked
 *   per x-plane.  45 = 9*5 with gcd(9,5)=1, so the PFA map applies and the
 *   inter-stage twiddle stage vanishes:
 *       n = (5*n1 + 9*n2) mod 45        (Ruritanian input map, n1<9, n2<5)
 *       k = (10*k1 + 36*k2) mod 45      (CRT output map)
 *   A 45-point DFT is 9 independent 5-point DFTs (FFTW n1_5 FMA DAG) then 5
 *   independent 9-point DFTs (genfft n1_9 FMA DAG) with no twiddles between,
 *   344 FMA-port ops + 78 shuffles per PW lines.  Two sweeps per volume:
 *     phase 1, per x-plane: 45 z-lines (register-transposed in/out of a
 *              padded plane scratch), then 45 y-lines into `out`;
 *     phase 2: x-lines, in place in `out`, tiled PW consecutive flat (y,z).
 *   45 is odd, so each pass runs NGF full-width groups plus one true PW=1
 *   xmm tail line (z/y passes) or one masked tail call (x pass).
 *
 * MULTICORE LAYER (new this phase, round mt_r1)
 *   * A persistent pthread pool is built in fft3d_create() (thread creation
 *     is setup): workers are pinned to the exact CPUs the harness's
 *     OMP_PROC_BIND=close / OMP_PLACES=cores mapping would give them (the
 *     mapping is read back from one throwaway OpenMP region), first-touch
 *     their own plane scratch after pinning (NUMA-local), and spin on an
 *     atomic generation counter.  fft3d_execute() publishes a job and
 *     release-stores gen+1; no thread is ever created on the execute path,
 *     and OpenMP is never entered after create().  Pool architecture,
 *     flag-array barriers and the GOMP-fork-costs-6-8us measurement that
 *     motivate it are BORROWED from L23_matrixsimd's mt_r1 record.
 *   * Barriers are per-thread padded arrival flags + one release word per
 *     group: an arrival is one uncontended store, the group leader scans
 *     (the loads overlap in the fill buffers) and publishes one release
 *     word.  Epochs derive from the job generation and the in-job volume
 *     index, so flags only ever increase and a varying participant set
 *     stays correct across tuner jobs of different shapes.
 *   * Four decompositions, raced at plan time:
 *       serial  -- the phase-1 kernel on the main thread (the B=1 floor:
 *                  if one volume does not parallelise, say so with the
 *                  pick, do not fake it);
 *       vol     -- thread t owns volumes [B*t/T, B*(t+1)/T) and runs the
 *                  full serial schedule on its own scratch; the only sync
 *                  is the join;
 *       grp     -- G threads share each volume: member m does x-planes
 *                  [45m/G, 45(m+1)/G) of phase 1, one group barrier, then
 *                  phase-2 tiles [NTT*m/G, NTT*(m+1)/G).  One barrier per
 *                  volume: a member may start volume b+1's phase 1 while a
 *                  sibling still runs volume b's phase 2 (disjoint data).
 *                  B=1 is grp with one group of T threads; at batch cells
 *                  grp G=2/4 also halves the number of volumes in flight,
 *                  i.e. the aggregate phase1-out -> phase2-in reuse window,
 *                  which is the L3-residency lever at B>=16.
 *       vnt     -- NEW round mt_r2, BORROWED from L45_pfa mt_r1's winning
 *                  "mtn" class (their node B=256 was 26.9 us/vol against
 *                  this entry's 79.1): volumes are claimed DYNAMICALLY off
 *                  one atomic counter; a thread runs BOTH phases inside its
 *                  private NUMA-local mid volume M (1.46 MB), then flushes
 *                  M -> out with one linear vmovntpd burst (16 B head peel:
 *                  volume bases rotate 16 B since NVOL*16 mod 64 = 16) and
 *                  one sfence per thread per job.  This touches the
 *                  caller's `out` exactly once, deleting both the phase-1
 *                  RFO and the phase-2 read-back+rewrite of `out` that made
 *                  the in-place schedule pay ~5 DRAM/UPI passes per volume
 *                  when 32 volumes in flight overflow the node's L3.  The
 *                  dynamic counter additionally load-balances the socket
 *                  whose share of the caller's pages is remote.
 *   * Memory mechanisms carried from the serial streaming winner, now as
 *     per-thread candidates: m0 none, m1 pfin+pfw (paced T1 prefetch of the
 *     next input plane + write-intent prefetch of the next output plane),
 *     m2 cpy (y pass into an L1-hot plane image, one ERMS rep-movsb per
 *     plane to `out` -- deletes the cold-out RFO, the biggest single-lever
 *     traffic cut at a 32-thread bandwidth wall), m3 = m1 + phase-2
 *     45-stream T0 poke.  The r11 odd-column (-oc) axis and the phase-2
 *     next-volume re-cover are dropped this round to shrink the surface;
 *     LDCOL/STCOL (the r9 forms) are the fixed odd-column path.
 *   * vns     -- NEW round mt_r3: vnt with STATIC contiguous volume blocks
 *                (thread t owns [B*t/T, B*(t+1)/T)), i.e. exactly L45_pfa's
 *                mtn shape.  Their impl_2 comment states the rationale this
 *                entry now adopts: "each thread touches the same volumes
 *                every call (prefetcher- and NUMA-stable)".  The mt_r2
 *                VERDICT (S5) measured two page-placement regimes on the
 *                node -- caller pages all socket-0 (~94 GB/s ceiling) vs
 *                spread (~200 GB/s) -- and AutoNUMA can only settle pages
 *                toward the far socket if the page->thread map is STABLE
 *                across the driver's ~48 calls.  My r2 dynamic claim
 *                counter destroyed that stability; vns restores it.  vnt
 *                stays fielded as the dynamic control.
 *   * PFNX (mt_r3, BORROWED from L45_pfa's r1/r2 pfi/pfw mechanisms, which
 *     won both streaming cells on the node): during phase 2, pre-cover the
 *     first ~63 KB of the NEXT volume's input with 2 paced T1 prefetch
 *     lines per tile.  Phase 2 + the NT flush leave the DRAM read stream
 *     idle for ~half the volume time; this overlaps the next volume's
 *     compulsory read into that hole.  Mechanisms m8 (pp+poke+pfnx, the
 *     analog of their winning B=16 g2-pfw), m9 (pfin+pfnx, their mtn-pfi),
 *     m10 (pfnx alone, for the node's m0-preferring vnt table).
 *   * Placement diagnostic (mt_r3, BORROWED from L8_fusedaxes mt_r2's
 *     governor, the instrument the mt_r2 VERDICT asked to be pointed at a
 *     32-thread streaming team): a read-only get_mempolicy scan of ~64
 *     sampled pages of the caller's in/out, run on the 1st and every 24th
 *     execute when an NT-staged mode is picked, published in the
 *     description as gov{nb,fr0,fr,sc,n}.  Pure read; no move_pages.
 *   * All decompositions assign whole lines to threads and change no
 *     per-line arithmetic or order, so every mode/width/mechanism is ONE
 *     bit class with the serial kernel (verified by the create()-time gate,
 *     which compares against a serial reference at 1e-13).
 *
 * ASSUMPTIONS
 *   * L == 45 only; fft3d_supports() refuses everything else.
 *   * `in` and `out` are 64-byte aligned and distinct (driver guarantees).
 *   * `out` doubles as the working buffer between phase 1 and phase 2; `in`
 *     is never written.
 *   * Never more than OMP_NUM_THREADS (<=32) threads, per the brief: the
 *     pool size is min(omp_get_max_threads(), 32) including the caller.
 *   * Three kernel widths: V0 = AVX2+FMA, V1 = AVX-512F (4 lanes),
 *     V2 = AVX-512VL (2 lanes, 32 ymm).  On AVX-512 build hosts V0
 *     compiles EVEX and folds into V2 (r9 audit); it is fielded only on
 *     hosts without AVX-512.
 */

#ifndef VAR
/* ======================= common (width-independent) part ================== */

#define _GNU_SOURCE             /* sched_getcpu, CPU_SET, pthread affinity */
#define _POSIX_C_SOURCE 200809L

#include <complex.h>
#include <immintrin.h>
#include <math.h>
#include <omp.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#include "fft3d_api.h"

#define LSIDE 45
#define NPLANE (LSIDE * LSIDE)          /* complex per x-plane = 2025    */
#define NVOL   (LSIDE * LSIDE * LSIDE)  /* complex per volume  = 91125   */
#ifndef PPITCH
#define PPITCH 52                       /* plane-scratch row pitch, complex:
                                           52*16 = 832 B = 13 cache lines,
                                           64-byte aligned rows, 13 coprime
                                           with the 64 L1 sets (see the
                                           phase-1 record; from L45_pfa r6) */
#endif
#define PFLINES 508                     /* 64B lines covering one x-plane     */

/* ---- pre-splatted constants, 8-double 64-byte rows usable as memory
   operands by both the 256-bit and 512-bit kernels. ---------------------- */
#define SPLAT8(v) { (v), (v), (v), (v), (v), (v), (v), (v) }
#define ALT8(v)   { (v), -(v), (v), -(v), (v), -(v), (v), -(v) }

static const double KC_HALF[8] __attribute__((aligned(64))) = SPLAT8(0.5);
/* sqrt(3)/2 alternating: swap(m) * this = -i*s*m in interleaved layout */
static const double KC_KS[8] __attribute__((aligned(64)))
    = ALT8(8.66025403784438646764e-01);
/* genfft n1_9 FMA-DAG constants (fftw-3.3.10 dft/scalar/codelets/n1_9.c;
 * vector transcription borrowed from L45_pfa r9).  A* rows are alternating
 * [v,-v,...] (their VPAIR), S852 is a plain splat. */
static const double KC_A176[8]  __attribute__((aligned(64))) = ALT8( 1.76326980708464973471e-01); /* tan(pi/18) */
static const double KC_A176N[8] __attribute__((aligned(64))) = ALT8(-1.76326980708464973471e-01);
static const double KC_A839[8]  __attribute__((aligned(64))) = ALT8( 8.39099631177280011763e-01);
static const double KC_A777[8]  __attribute__((aligned(64))) = ALT8( 7.77861913430206160028e-01);
static const double KC_A984[8]  __attribute__((aligned(64))) = ALT8( 9.84807753012208059367e-01);
static const double KC_A492[8]  __attribute__((aligned(64))) = ALT8( 4.92403876506104029683e-01);
static const double KC_A363[8]  __attribute__((aligned(64))) = ALT8( 3.63970234266202361351e-01); /* tan(pi/9) */
static const double KC_A954N[8] __attribute__((aligned(64))) = ALT8(-9.54188894138671133499e-01);
static const double KC_S852[8]  __attribute__((aligned(64))) = SPLAT8(8.52868531952443209628e-01);
/* 5-point, FFTW n1_5 FMA constants (borrowed from L45_pfa r6):
 * c1,c2 = -1/4 +- sqrt(5)/4 (cosine split), s2 = phi^-1 * s1 (scaled sine) */
static const double KC_Q4[8] __attribute__((aligned(64))) = SPLAT8(2.5e-01);
static const double KC_S5[8] __attribute__((aligned(64))) = SPLAT8(5.59016994374947424102e-01);
static const double KC_PHI[8] __attribute__((aligned(64))) = SPLAT8(6.18033988749894848205e-01);
static const double KC_S1[8] __attribute__((aligned(64))) = ALT8(9.51056516295153572116e-01);
/* one-complex tail mask for the AVX2 maskload/maskstore path */
static const long long KC_TMASK[4] __attribute__((aligned(32))) = { -1, -1, 0, 0 };

/* stage 1: nine 5-point DFTs; stage 2: five 9-point DFTs */
#define REP9(M) M(0) M(1) M(2) M(3) M(4) M(5) M(6) M(7) M(8)
#define REP5(M) M(0) M(1) M(2) M(3) M(4)

/* ---- the PFA 9x5 line codelet, as lazily-expanded macro text (r11, after
 * L45_pfa r10's lazy-expansion trick) so the SAME DAG instantiates at PW=1
 * for the 128-bit tail lines below.  ST1G/ST2G reference VD / VADD / VSWAP /
 * VLOAD / C_* ..., which resolve to whichever width's macro layer is in
 * scope at the expansion site. ---- */
#define C_HALF VLOAD(KC_HALF)
#define C_KS   VLOAD(KC_KS)
#define C_Q4   VLOAD(KC_Q4)
#define C_S5   VLOAD(KC_S5)
#define C_PHI  VLOAD(KC_PHI)
#define C_S1   VLOAD(KC_S1)

/* stage 1: nine 5-point DFTs over n2, PFA input map n = (5*n1 + 9*n2) mod 45.
 * FFTW n1_5's FMA DAG (borrowed from L45_pfa r6): 16 FMA-port ops + 2
 * shuffles.  Output y[k2] goes to TT[k2*9 + N1] so stage 2 reads contiguous
 * runs. */
#define ST1G(TT, LS, N1) {                                     \
    VD x0 = LS((5 * (N1) + 9 * 0) % 45);                       \
    VD x1 = LS((5 * (N1) + 9 * 1) % 45);                       \
    VD x2 = LS((5 * (N1) + 9 * 2) % 45);                       \
    VD x3 = LS((5 * (N1) + 9 * 3) % 45);                       \
    VD x4 = LS((5 * (N1) + 9 * 4) % 45);                       \
    VD t1 = VADD(x1, x4), t4 = VSUB(x1, x4);                   \
    VD t2 = VADD(x2, x3), t3 = VSUB(x2, x3);                   \
    VD te = VADD(t1, t2), ta = VSUB(t1, t2);                   \
    VD tm = VFNMADD(C_Q4, te, x0);                             \
    VD tp = VFMADD(ta, C_S5, tm);                              \
    VD tq = VFNMADD(ta, C_S5, tm);                             \
    VD u4 = VSWAP(t4), u3 = VSWAP(t3);                         \
    VD w1 = VFMADD(u3, C_PHI, u4);                             \
    VD w2 = VFMSUB(u4, C_PHI, u3);                             \
    TT[0 * 9 + (N1)] = VADD(x0, te);                           \
    TT[1 * 9 + (N1)] = VFMADD(w1, C_S1, tp);                   \
    TT[2 * 9 + (N1)] = VFMADD(w2, C_S1, tq);                   \
    TT[3 * 9 + (N1)] = VFNMADD(w2, C_S1, tq);                  \
    TT[4 * 9 + (N1)] = VFNMADD(w1, C_S1, tp);                  \
}
#define ST1(N1) ST1G(T, LSRC, N1)

/* stage 2: five 9-point DFTs over n1, PFA output map k = (10*k1+36*k2) mod 45.
 * genfft n1_9 FMA DAG transcription BORROWED from L45_pfa r9: 40 FMA-port
 * ops + 12 shuffles per call. */
#define ST2G(TT, SD, K2) {                                     \
    VD u0 = TT[(K2) * 9 + 0], u1 = TT[(K2) * 9 + 1];           \
    VD u2 = TT[(K2) * 9 + 2], u3 = TT[(K2) * 9 + 3];           \
    VD u4 = TT[(K2) * 9 + 4], u5 = TT[(K2) * 9 + 5];           \
    VD u6 = TT[(K2) * 9 + 6], u7 = TT[(K2) * 9 + 7];           \
    VD u8 = TT[(K2) * 9 + 8];                                  \
    VD s0 = VADD(u3, u6), e0 = VSUB(u3, u6);                   \
    VD S0 = VADD(u0, s0), a0 = VFNMADD(C_HALF, s0, u0);        \
    VD i0 = VSWAP(e0);                                         \
    VD s1 = VADD(u4, u7), e1 = VSUB(u4, u7);                   \
    VD S1 = VADD(u1, s1), a1 = VFNMADD(C_HALF, s1, u1);        \
    VD i1 = VSWAP(e1);                                         \
    VD p1 = VFMADD(i1, C_KS, a1), q1 = VFNMADD(i1, C_KS, a1);  \
    VD s2 = VADD(u5, u8), e2 = VSUB(u5, u8);                   \
    VD S2 = VADD(u2, s2), a2 = VFNMADD(C_HALF, s2, u2);        \
    VD i2 = VSWAP(e2);                                         \
    VD p2 = VFMADD(i2, C_KS, a2), q2 = VFNMADD(i2, C_KS, a2);  \
    VD sg = VADD(S1, S2), d3 = VSUB(S2, S1), id = VSWAP(d3);   \
    VD b0 = VFNMADD(C_HALF, sg, S0);                           \
    SD((10 * 0 + 36 * (K2)) % 45, VADD(S0, sg));               \
    SD((10 * 3 + 36 * (K2)) % 45, VFNMADD(id, C_KS, b0));      \
    SD((10 * 6 + 36 * (K2)) % 45, VFMADD(id, C_KS, b0));       \
    {   /* k = 1, 4, 7 */                                      \
    VD v1 = VFMADD(i0, C_KS, a0);                              \
    VD w2 = VFMADD(VSWAP(p2), VLOAD(KC_A176N), p2);            \
    VD w1 = VFMADD(VSWAP(p1), VLOAD(KC_A839), p1);             \
    VD uu = VFMADD(w1, VLOAD(KC_A777), VSWAP(w2));             \
    VD zz = VFMADD(VSWAP(w1), VLOAD(KC_A777), w2);             \
    SD((10 * 1 + 36 * (K2)) % 45, VFMADD(uu, VLOAD(KC_A984), v1)); \
    VD r1 = VFNMADD(uu, VLOAD(KC_A492), v1);                   \
    SD((10 * 4 + 36 * (K2)) % 45, VFMADD(zz, VLOAD(KC_S852), r1)); \
    SD((10 * 7 + 36 * (K2)) % 45, VFNMADD(zz, VLOAD(KC_S852), r1)); \
    }                                                          \
    {   /* k = 2, 5, 8 */                                      \
    VD v2 = VFNMADD(i0, C_KS, a0);                             \
    VD wA = VFMADD(q1, VLOAD(KC_A176), VSWAP(q1));             \
    VD wB = VFNMADD(VSWAP(q2), VLOAD(KC_A363), q2);            \
    VD uB = VFMADD(wB, VLOAD(KC_A954N), wA);                   \
    VD zB = VFMADD(VSWAP(wB), VLOAD(KC_A954N), VSWAP(wA));     \
    SD((10 * 2 + 36 * (K2)) % 45, VFMADD(uB, VLOAD(KC_A984), v2)); \
    VD rB = VFNMADD(uB, VLOAD(KC_A492), v2);                   \
    SD((10 * 5 + 36 * (K2)) % 45, VFNMADD(zB, VLOAD(KC_S852), rB)); \
    SD((10 * 8 + 36 * (K2)) % 45, VFMADD(zB, VLOAD(KC_S852), rB)); \
    }                                                          \
}
#define ST2(K2) ST2G(T, SDST, K2)

/* ---- true PW=1 tail lines (r11, BORROWED from L45_pfa r10): the same PFA
 * DAG at one complex per xmm vector -- no transposes, 16 B accesses that
 * never split a cache line, dual-issue on ports 0 and 1.  Compiled once
 * under target("fma"); called by every width variant. ---- */
#define VD               __m128d
#define VLOAD(p)         _mm_loadu_pd(p)
#define VADD(a, b)       _mm_add_pd((a), (b))
#define VSUB(a, b)       _mm_sub_pd((a), (b))
#define VFMADD(a, b, c)  _mm_fmadd_pd((a), (b), (c))
#define VFMSUB(a, b, c)  _mm_fmsub_pd((a), (b), (c))
#define VFNMADD(a, b, c) _mm_fnmadd_pd((a), (b), (c))
#define VSWAP(a)         _mm_shuffle_pd((a), (a), 1)

/* z-axis tail: the single y = 44 line of one x-plane; both the `in` row and
 * the plane row are contiguous, so there is nothing to transpose at PW=1. */
__attribute__((target("fma"), noinline, unused))
static void dft45_tz1(const double *src, double *dst)
{
    VD T[45];
#define LSRC(i)     VLOAD(src + (long)(i) * 2)
#define SDST(i, v)  _mm_storeu_pd(dst + (long)(i) * 2, (v))
    REP9(ST1)
    REP5(ST2)
#undef LSRC
#undef SDST
}

/* y-axis tail: the single z = 44 column (plane column walk -> out column
 * walk); 16 B accesses at 832 B / 720 B stride, never splitting a line. */
__attribute__((target("fma"), noinline, unused))
static void dft45_ty1(const double *src, double *dst)
{
    VD T[45];
#define LSRC(i)     VLOAD(src + (long)(i) * (PPITCH * 2))
#define SDST(i, v)  _mm_storeu_pd(dst + (long)(i) * (LSIDE * 2), (v))
    REP9(ST1)
    REP5(ST2)
#undef LSRC
#undef SDST
}

#undef VD
#undef VLOAD
#undef VADD
#undef VSUB
#undef VFMADD
#undef VFMSUB
#undef VFNMADD
#undef VSWAP

/* one ERMS rep-movsb per x-plane: with the cpy mechanism, the y pass stores
 * into an L1-hot plane image and this copies it to `out`.  ERMS full-line
 * writes skip the read-for-ownership of the destination, deleting the
 * 1.46 MB/volume cold-`out` fill read, while the lines stay in the cache
 * hierarchy for phase 2 to re-read (unlike NT stores). */
static inline void plane_copy(void *dst, const void *src, size_t n)
{
#if defined(__x86_64__)
    void *d = dst;
    const void *s = src;
    size_t c = n;
    __asm__ volatile("rep movsb" : "+D"(d), "+S"(s), "+c"(c) : : "memory");
#else
    memcpy(dst, src, n);
#endif
}

/* ---- vnt's staged flush: one linear non-temporal burst M -> out (mt_r2,
 * pattern BORROWED from L45_pfa mt_r1's mtn class, including their lesson
 * that vmovntpd demands 64 B alignment while out volume bases rotate 16 B:
 * NVOL*16 mod 64 = 16, hence the 16 B head peel and 16 B tail).  NT stores
 * delete the destination RFO and do not evict the L2/L3-resident mid
 * volume; the caller issues one sfence per job after its volume loop
 * (weakly-ordered stores must be fenced before the join's release store).
 * nd is a count of doubles; dst is always at least 16 B aligned here. ---- */
#if defined(__x86_64__)
__attribute__((target("avx512f"), noinline))
static void nt_flush_512(double *dst, const double *src, size_t nd)
{
    size_t i = 0;
    if ((uintptr_t)dst & 15) { memcpy(dst, src, nd * 8); return; }
    for (; ((uintptr_t)(dst + i) & 63) && i + 2 <= nd; i += 2)
        _mm_stream_pd(dst + i, _mm_loadu_pd(src + i));
#pragma GCC unroll 4
    for (; i + 8 <= nd; i += 8)
        _mm512_stream_pd(dst + i, _mm512_loadu_pd(src + i));
    for (; i + 2 <= nd; i += 2)
        _mm_stream_pd(dst + i, _mm_loadu_pd(src + i));
}
__attribute__((target("avx2"), noinline))
static void nt_flush_256(double *dst, const double *src, size_t nd)
{
    size_t i = 0;
    if ((uintptr_t)dst & 15) { memcpy(dst, src, nd * 8); return; }
    for (; ((uintptr_t)(dst + i) & 31) && i + 2 <= nd; i += 2)
        _mm_stream_pd(dst + i, _mm_loadu_pd(src + i));
#pragma GCC unroll 8
    for (; i + 4 <= nd; i += 4)
        _mm256_stream_pd(dst + i, _mm256_loadu_pd(src + i));
    for (; i + 2 <= nd; i += 2)
        _mm_stream_pd(dst + i, _mm_loadu_pd(src + i));
}
#else
static void nt_flush_512(double *dst, const double *src, size_t nd)
{ memcpy(dst, src, nd * 8); }
static void nt_flush_256(double *dst, const double *src, size_t nd)
{ memcpy(dst, src, nd * 8); }
#endif

/* ---- per-(width, mechanism) range kernels, instantiated below.
 * p1r: phase 1 (z pass + y pass) for x-planes [x0, x1) of one volume.
 * p2r: phase 2 (in-place x pass) for flat tiles [t0, t1); the last tile
 *      index (NTT-1) is the one masked tail call. ---- */
typedef void (*p1_fn)(const double *vin, double *vout, double *plane,
                      double *plane2, long x0, long x1,
                      const double *inend, double *outend);
typedef void (*p2_fn)(double *vout, long t0, long t1, const double *nx);

#define DECL_PR1(V, M)                                                        \
static void p1r_##V##_##M(const double *, double *, double *, double *,       \
                          long, long, const double *, double *);              \
static void p2r_##V##_##M(double *, long, long, const double *);
#define DECL_PR(V) DECL_PR1(V, 0) DECL_PR1(V, 1) DECL_PR1(V, 2) DECL_PR1(V, 3) \
                   DECL_PR1(V, 4) DECL_PR1(V, 5) DECL_PR1(V, 6) DECL_PR1(V, 7) \
                   DECL_PR1(V, 8) DECL_PR1(V, 9) DECL_PR1(V, 10)
DECL_PR(0)
DECL_PR(1)
DECL_PR(2)
#undef DECL_PR
#undef DECL_PR1

/* ======================= the persistent thread pool =======================
 * Architecture BORROWED from L23_matrixsimd mt_r1 (persistent pinned spin
 * pool; flag-array barriers; the 6-8 us GOMP fork/join measurement that
 * motivates it), re-spelt for this kernel's job shapes. */

#define MAXT 32

static inline void cpu_relax(void)
{
#if defined(__x86_64__)
    _mm_pause();
#endif
}

typedef struct {
    double *plane;   /* 45 x PPITCH-complex padded z->y handoff scratch */
    double *plane2;  /* one x-plane image for the cpy mechanism         */
    double *mid;     /* private staging volume for the vnt mode (mt_r2) */
    void *blk;
} mt_ctx;

struct fft3d_plan;
typedef void (*mt_workfn)(struct fft3d_plan *, int);

/* Per-thread arrival flag / per-group release word, one cache line each:
 * a barrier arrival is one uncontended store, not T serialized RMWs. */
typedef struct {
    _Atomic unsigned long v;
    char pad[56];
} mt_flag;

typedef struct {
    struct fft3d_plan *p;
    int tid;
    int cpu;
} mt_warg;

typedef struct {
    _Atomic long gen;
    char pad0[56];
    _Atomic long vidx;   /* vnt: next unclaimed volume (reset per job)   */
    char padv[56];
    mt_flag rel[MAXT];   /* per-group release words, indexed by group id */
    mt_flag arr[MAXT];   /* per-thread arrival flags                     */
    _Atomic int ready;
    _Atomic int fail;
    _Atomic int shutdown;
    char pad2[52];
    /* the job, published by plain stores BEFORE the gen release-store */
    mt_workfn workfn;
    const double *in;
    double *out;
    long batch;
    int tw;                 /* participants of the current job */
    int G;                  /* group size (grp mode)            */
    unsigned long emult;    /* epoch stride per generation      */
    long gen_local;
    int nthr;               /* pool size incl. main             */
    pthread_t th[MAXT];
    mt_warg wa[MAXT];
} mt_pool;

struct fft3d_plan {
    long batch;
    int mode;   /* 0 serial, 1 vol, 2 grouped, 3 vnt (NT dynamic), 4 vns
                   (NT static blocks, mt_r3) */
    int T, G;
    long ntt;   /* phase-2 tile count + 1 (tail) for the picked width */
    p1_fn p1;
    p2_fn p2;
    void (*ntcpy)(double *, const double *, size_t); /* vnt/vns flush */
    /* placement diagnostic (mt_r3, borrowed from L8_fusedaxes mt_r2):
     * read-only get_mempolicy scan of the caller's buffers, published in
     * the description.  Main-thread-only state. */
    long ncall;
    int main_node, nb_on, fr0, frl, nscan;
    mt_ctx ctx[MAXT];
    mt_pool *pl;
};

/* Percent of ~64 sampled in/out pages homed off the main thread's node --
 * a pure READ (get_mempolicy with MPOL_F_NODE|MPOL_F_ADDR mutates nothing;
 * the mt_r1 ruling banned move_pages and the mt_r2 VERDICT asked for
 * exactly this number under a 32-thread streaming team).  -1 if unknown. */
static int gov_scan_remote(const double *ip, const double *op, long batch,
                           int main_node)
{
#if defined(__linux__) && defined(SYS_get_mempolicy)
#ifndef MPOL_F_NODE
#define MPOL_F_NODE (1 << 0)
#endif
#ifndef MPOL_F_ADDR
#define MPOL_F_ADDR (1 << 1)
#endif
    if (main_node < 0) return -1;
    long step = batch / 64;
    if (step < 1) step = 1;
    int tot = 0, rem = 0;
    for (long b = 0; b < batch; b += step) {
        const void *pa[2] = { ip + (size_t)b * NVOL * 2 + 1024,
                              op + (size_t)b * NVOL * 2 + 1024 };
        for (int k = 0; k < 2; ++k) {
            const uintptr_t pg = (uintptr_t)pa[k] & ~(uintptr_t)4095;
            int nd = -1;
            if (syscall(SYS_get_mempolicy, &nd, NULL, 0UL, (void *)pg,
                        (unsigned long)(MPOL_F_NODE | MPOL_F_ADDR)) == 0 &&
                nd >= 0) {
                ++tot;
                rem += (nd != main_node);
            }
        }
    }
    return tot ? (int)(100L * rem / tot) : -1;
#else
    (void)ip; (void)op; (void)batch; (void)main_node;
    return -1;
#endif
}

static char g_gov[96];

/* JOIN (end of every job): workers post their flag and go straight back to
 * the generation spin; main cannot dispatch again (or return to the driver)
 * until it has seen every flag, so nobody waits on anybody else. */
static inline void mt_join(mt_pool *pl, int tid, int T)
{
    unsigned long e =
        (unsigned long)atomic_load_explicit(&pl->gen, memory_order_relaxed)
            * pl->emult + (pl->emult - 1);
    if (tid == 0) {
        for (int t = 1; t < T; ++t)
            while (atomic_load_explicit(&pl->arr[t].v, memory_order_acquire) < e)
                cpu_relax();
    } else {
        atomic_store_explicit(&pl->arr[tid].v, e, memory_order_release);
    }
}

/* volume-parallel: thread t owns volumes [B*t/T, B*(t+1)/T), full serial
 * schedule on its own scratch, join only. */
static void mt_work_vol(struct fft3d_plan *p, int tid)
{
    mt_pool *pl = p->pl;
    const int T = pl->tw;
    if (tid < T) {
        const long nb = pl->batch;
        const long lo = nb * tid / T, hi = nb * (tid + 1) / T;
        mt_ctx *cx = &p->ctx[tid];
        const double *inend = pl->in + nb * (long)NVOL * 2;
        double *outend = pl->out + nb * (long)NVOL * 2;
        for (long b = lo; b < hi; ++b) {
            const double *vin = pl->in + b * (long)NVOL * 2;
            double *vout = pl->out + b * (long)NVOL * 2;
            const double *nx = b + 1 < hi ? vin + (long)NVOL * 2 : NULL;
            p->p1(vin, vout, cx->plane, cx->plane2, 0, LSIDE, inend, outend);
            p->p2(vout, 0, p->ntt, nx);
        }
    }
    mt_join(pl, tid, pl->tw);
}

/* vns (mt_r3): static-block NT-staged -- L45_pfa's mtn shape, adopted with
 * attribution.  Thread t owns the CONTIGUOUS volumes [B*t/T, B*(t+1)/T),
 * runs both phases in its private NUMA-local mid volume M and flushes M ->
 * out with one linear NT burst.  Against vnt the only change is the volume
 * assignment: the same thread touches the same in/out pages on every call,
 * which is the stability AutoNUMA needs to migrate the caller's socket-0
 * pages toward the far socket over the driver's timed loop (mt_r2 VERDICT
 * S5: that regime is worth up to 2x at a streaming cell, and the min-over-
 * samples statistic catches the settled end of it). */
static void mt_work_vns(struct fft3d_plan *p, int tid)
{
    mt_pool *pl = p->pl;
    const int T = pl->tw;
    if (tid < T) {
        const long nb = pl->batch;
        const long lo = nb * tid / T, hi = nb * (tid + 1) / T;
        mt_ctx *cx = &p->ctx[tid];
        const double *inend = pl->in + nb * (long)NVOL * 2;
        double *outend = cx->mid + (long)NVOL * 2;
        for (long b = lo; b < hi; ++b) {
            const double *vin = pl->in + b * (long)NVOL * 2;
            double *vout = pl->out + b * (long)NVOL * 2;
            const double *nx = b + 1 < hi ? vin + (long)NVOL * 2 : NULL;
            p->p1(vin, cx->mid, cx->plane, cx->plane2, 0, LSIDE,
                  inend, outend);
            p->p2(cx->mid, 0, p->ntt, nx);
            p->ntcpy(vout, cx->mid, (size_t)NVOL * 2);
        }
#if defined(__x86_64__)
        if (hi > lo) _mm_sfence();
#endif
    }
    mt_join(pl, tid, pl->tw);
}

/* vnt (mt_r2): volumes claimed dynamically off pl->vidx; both phases run in
 * the thread's private mid volume M, then one linear NT burst M -> out.
 * The dynamic claim costs one uncontended-ish fetch_add per volume (<= 256
 * per job) and rebalances whichever socket is slowed by remote caller
 * pages; the NT flush touches `out` exactly once per volume.  Weakly-
 * ordered NT stores are fenced before the join's release store. */
static void mt_work_vnt(struct fft3d_plan *p, int tid)
{
    mt_pool *pl = p->pl;
    const int T = pl->tw;
    if (tid < T) {
        const long nb = pl->batch;
        mt_ctx *cx = &p->ctx[tid];
        const double *inend = pl->in + nb * (long)NVOL * 2;
        double *outend = cx->mid + (long)NVOL * 2;
        int did = 0;
        for (;;) {
            long b = atomic_fetch_add_explicit(&pl->vidx, 1,
                                               memory_order_relaxed);
            if (b >= nb) break;
            const double *vin = pl->in + b * (long)NVOL * 2;
            double *vout = pl->out + b * (long)NVOL * 2;
            p->p1(vin, cx->mid, cx->plane, cx->plane2, 0, LSIDE,
                  inend, outend);
            p->p2(cx->mid, 0, p->ntt, NULL); /* dynamic: next claim unknown */
            p->ntcpy(vout, cx->mid, (size_t)NVOL * 2);
            did = 1;
        }
#if defined(__x86_64__)
        if (did) _mm_sfence();
#else
        (void)did;
#endif
    }
    mt_join(pl, tid, pl->tw);
}

/* grouped: G threads per volume; member m does planes [45m/G, 45(m+1)/G),
 * one group barrier (arrival flags + the group's release word), then tiles
 * [NTT*m/G, NTT*(m+1)/G).  Only ONE barrier per volume: a member's phase 1
 * of volume b+1 touches only out[b+1] and its private scratch, so it never
 * conflicts with a sibling still in volume b's phase 2. */
static void mt_work_grp(struct fft3d_plan *p, int tid)
{
    mt_pool *pl = p->pl;
    const int T = pl->tw, G = pl->G;
    const int ng = T / G;
    if (tid < ng * G) {
        const int g = tid / G, m = tid % G;
        const long nb = pl->batch;
        const long lo = nb * g / ng, hi = nb * (g + 1) / ng;
        const unsigned long ebase =
            (unsigned long)atomic_load_explicit(&pl->gen,
                                                memory_order_relaxed)
                * pl->emult;
        mt_ctx *cx = &p->ctx[tid];
        const long x0 = (long)LSIDE * m / G, x1 = (long)LSIDE * (m + 1) / G;
        const long t0 = p->ntt * m / G, t1 = p->ntt * (m + 1) / G;
        const double *inend = pl->in + nb * (long)NVOL * 2;
        double *outend = pl->out + nb * (long)NVOL * 2;
        for (long b = lo; b < hi; ++b) {
            const double *vin = pl->in + b * (long)NVOL * 2;
            double *vout = pl->out + b * (long)NVOL * 2;
            /* pfnx target: this member's OWN phase-1 planes of the next
             * volume, [x0, x1) -- what it will read right after the tiles */
            const double *nx = b + 1 < hi
                ? vin + (long)NVOL * 2 + x0 * (long)NPLANE * 2 : NULL;
            p->p1(vin, vout, cx->plane, cx->plane2, x0, x1, inend, outend);
            {   /* group barrier: all of this volume's phase 1 before any of
                 * its phase 2.  Release/acquire chains through the leader,
                 * so every member's stores are visible to every reader. */
                unsigned long e = ebase + (unsigned long)(b - lo) + 1;
                if (m == 0) {
                    for (int j = 1; j < G; ++j)
                        while (atomic_load_explicit(&pl->arr[tid + j].v,
                                                    memory_order_acquire) < e)
                            cpu_relax();
                    atomic_store_explicit(&pl->rel[g].v, e,
                                          memory_order_release);
                } else {
                    atomic_store_explicit(&pl->arr[tid].v, e,
                                          memory_order_release);
                    while (atomic_load_explicit(&pl->rel[g].v,
                                                memory_order_acquire) < e)
                        cpu_relax();
                }
            }
            p->p2(vout, t0, t1, nx);
        }
    }
    mt_join(pl, tid, pl->tw);
}

/* Worker: pin to the harness-consistent CPU, allocate and FIRST-TOUCH this
 * thread's scratch (NUMA locality: the owner touches it, after pinning),
 * then spin on the pool generation. */
static int mt_ctx_alloc(mt_ctx *c)
{
    /* mid must be 64 B aligned for the NT flush's head-peel logic to match
     * the SOURCE loads too; round its offset up to a whole line */
    size_t moff = (((size_t)(LSIDE * PPITCH + NPLANE) * 2 + 7) / 8) * 8;
    size_t nd = moff + (size_t)NVOL * 2;
    void *b = NULL;
    if (posix_memalign(&b, 64, nd * sizeof(double)) != 0 || !b) return 1;
    memset(b, 0, nd * sizeof(double));   /* first touch by the owner */
    c->blk = b;
    c->plane = (double *)b;
    c->plane2 = (double *)b + (size_t)LSIDE * PPITCH * 2;
    c->mid = (double *)b + moff;
    return 0;
}

static void *mt_worker(void *arg)
{
    mt_warg *wa = arg;
    struct fft3d_plan *p = wa->p;
    mt_pool *pl = p->pl;
    const int tid = wa->tid;

    if (wa->cpu >= 0) {
        cpu_set_t s;
        CPU_ZERO(&s);
        CPU_SET(wa->cpu, &s);
        pthread_setaffinity_np(pthread_self(), sizeof s, &s);
    }
    if (mt_ctx_alloc(&p->ctx[tid]))
        atomic_store(&pl->fail, 1);
    atomic_fetch_add(&pl->ready, 1);

    long last = 0;
    for (;;) {
        while (atomic_load_explicit(&pl->gen, memory_order_acquire) == last)
            cpu_relax();
        ++last;
        if (atomic_load_explicit(&pl->shutdown, memory_order_relaxed)) break;
        pl->workfn(p, tid);
    }
    return NULL;
}

/* Run the plan's current configuration on (in, out): serial on the main
 * thread, or one pool job.  Used by execute() and by the plan-time tuner. */
static void mt_run(struct fft3d_plan *p, const double *in, double *out)
{
    if (p->mode == 0 || !p->pl || p->pl->nthr < 2 || p->T < 2) {
        const double *inend = in + p->batch * (long)NVOL * 2;
        double *outend = out + p->batch * (long)NVOL * 2;
        for (long b = 0; b < p->batch; ++b) {
            const double *vin = in + b * (long)NVOL * 2;
            double *vout = out + b * (long)NVOL * 2;
            const double *nx = b + 1 < p->batch ? vin + (long)NVOL * 2 : NULL;
            p->p1(vin, vout, p->ctx[0].plane, p->ctx[0].plane2,
                  0, LSIDE, inend, outend);
            p->p2(vout, 0, p->ntt, nx);
        }
        return;
    }
    mt_pool *pl = p->pl;
    pl->in = in;
    pl->out = out;
    pl->batch = p->batch;
    pl->tw = p->T;
    pl->G = p->G;
    pl->workfn = p->mode == 1 ? mt_work_vol
               : p->mode == 2 ? mt_work_grp
               : p->mode == 4 ? mt_work_vns : mt_work_vnt;
    atomic_store_explicit(&pl->vidx, 0, memory_order_relaxed);
    ++pl->gen_local;
    atomic_store_explicit(&pl->gen, pl->gen_local, memory_order_release);
    pl->workfn(p, 0);
}

/* instantiate the kernel once per (ISA, vector width) variant */
#define VAR 0
#include __FILE__
#undef VAR
#define VAR 1
#include __FILE__
#undef VAR
#define VAR 2
#include __FILE__
#undef VAR

/* (width, mech) tables; NTTW = phase-2 full tiles + 1 tail per width */
static const p1_fn P1T[3][11] = {
    {p1r_0_0, p1r_0_1, p1r_0_2, p1r_0_3, p1r_0_4, p1r_0_5, p1r_0_6, p1r_0_7,
     p1r_0_8, p1r_0_9, p1r_0_10},
    {p1r_1_0, p1r_1_1, p1r_1_2, p1r_1_3, p1r_1_4, p1r_1_5, p1r_1_6, p1r_1_7,
     p1r_1_8, p1r_1_9, p1r_1_10},
    {p1r_2_0, p1r_2_1, p1r_2_2, p1r_2_3, p1r_2_4, p1r_2_5, p1r_2_6, p1r_2_7,
     p1r_2_8, p1r_2_9, p1r_2_10},
};
static const p2_fn P2T[3][11] = {
    {p2r_0_0, p2r_0_1, p2r_0_2, p2r_0_3, p2r_0_4, p2r_0_5, p2r_0_6, p2r_0_7,
     p2r_0_8, p2r_0_9, p2r_0_10},
    {p2r_1_0, p2r_1_1, p2r_1_2, p2r_1_3, p2r_1_4, p2r_1_5, p2r_1_6, p2r_1_7,
     p2r_1_8, p2r_1_9, p2r_1_10},
    {p2r_2_0, p2r_2_1, p2r_2_2, p2r_2_3, p2r_2_4, p2r_2_5, p2r_2_6, p2r_2_7,
     p2r_2_8, p2r_2_9, p2r_2_10},
};
static const long NTTW[3] = {NPLANE / 2 + 1, NPLANE / 4 + 1, NPLANE / 2 + 1};

/* ------------------------------ the API ---------------------------------- */

const char *fft3d_name(void) { return "L45_mixedradix"; }

static char g_desc[512];
static char g_full[640];

const char *fft3d_description(void)
{
    if (!g_desc[0])
        return "row-column PFA 9x5, pinned pthread pool, "
               "vns/vnt/grp decomposition autotuned";
    snprintf(g_full, sizeof g_full, "%s%s", g_desc, g_gov);
    return g_full;
}

int fft3d_supports(int L) { return L == LSIDE; }

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

typedef struct {
    int mode, T, G, v, m;
    char tag[28];
} mt_cand;

static void cand_tag(mt_cand *c)
{
    static const char *mn[11] = {"m0", "pp", "cpy", "pfpp", "cpin", "cpinpf",
                                 "pf", "pfpk", "ppnx", "pfnx", "nx"};
    if (c->mode == 0)
        snprintf(c->tag, sizeof c->tag, "ser-v%d-%s", c->v, mn[c->m]);
    else if (c->mode == 1)
        snprintf(c->tag, sizeof c->tag, "vol%d-v%d-%s", c->T, c->v, mn[c->m]);
    else if (c->mode == 3)
        snprintf(c->tag, sizeof c->tag, "vnt%d-v%d-%s", c->T, c->v, mn[c->m]);
    else if (c->mode == 4)
        snprintf(c->tag, sizeof c->tag, "vns%d-v%d-%s", c->T, c->v, mn[c->m]);
    else
        snprintf(c->tag, sizeof c->tag, "grp%dx%d-v%d-%s",
                 c->T / c->G, c->G, c->v, mn[c->m]);
}

fft3d_plan *fft3d_create(int L, int batch)
{
    if (L != LSIDE || batch < 1) return NULL;

    fft3d_plan *p = (fft3d_plan *)calloc(1, sizeof *p);
    if (!p) return NULL;
    p->batch = batch;
    p->mode = 0;
    p->T = 1;
    p->G = 1;
    p->main_node = -1;
    p->nb_on = -1;
#if defined(__linux__) && defined(SYS_getcpu)
    {   /* the driver fills in/out on this thread; its node is "home" */
        unsigned cpu = 0, nodeid = 0;
        if (syscall(SYS_getcpu, &cpu, &nodeid, NULL) == 0)
            p->main_node = (int)nodeid;
    }
#endif
    {
        FILE *nf = fopen("/proc/sys/kernel/numa_balancing", "r");
        if (nf) {
            int v;
            if (fscanf(nf, "%d", &v) == 1) p->nb_on = v;
            fclose(nf);
        }
    }

    int have_512 = 0, have_vl = 0;
#if defined(__x86_64__)
    __builtin_cpu_init();
    have_512 = __builtin_cpu_supports("avx512f")
               && __builtin_cpu_supports("avx512dq");
    have_vl  = have_512 && __builtin_cpu_supports("avx512vl");
#endif
    const int defv = have_512 ? 1 : 0;
    p->p1 = P1T[defv][0];
    p->p2 = P2T[defv][0];
    p->ntt = NTTW[defv];
    p->ntcpy = have_512 ? nt_flush_512 : nt_flush_256;

    if (mt_ctx_alloc(&p->ctx[0])) { free(p); return NULL; }

    /* never take more threads than the harness grants (<= 32; on a raw
     * login shell omp_get_max_threads() can read 128 -- cap it hard) */
    int maxt = omp_get_max_threads();
    if (maxt > MAXT) maxt = MAXT;
    if (maxt < 1) maxt = 1;

    /* ---- read the harness's OMP thread->CPU mapping from one throwaway
     * OpenMP region, so the pool pins exactly where OMP_PROC_BIND=close /
     * OMP_PLACES=cores would put each thread.  This also pins the initial
     * thread (pool tid 0) to place 0.  (Borrowed from L23_matrixsimd
     * mt_r1, including the unbound-run guard.)  GOMP's own workers sleep
     * after their spin timeout and are never used again. ---- */
    int cpus[MAXT];
    for (int t = 0; t < MAXT; ++t) cpus[t] = -1;
#pragma omp parallel num_threads(maxt)
    {
        int t = omp_get_thread_num();
        if (t < MAXT) cpus[t] = sched_getcpu();
    }
    {
        int pin = getenv("OMP_PROC_BIND") != NULL;
        for (int a = 0; pin && a < maxt; ++a)
            for (int b2 = a + 1; b2 < maxt; ++b2)
                if (cpus[a] == cpus[b2]) pin = 0; /* unbound run: don't pin */
        if (!pin)
            for (int t = 0; t < MAXT; ++t) cpus[t] = -1;
    }

    /* ---- the persistent pool (PANEL_BRIEF: thread creation is setup) ---- */
    if (maxt > 1) {
        void *pb = NULL;
        if (posix_memalign(&pb, 64, sizeof(mt_pool)) != 0 || !pb) {
            fft3d_destroy(p);
            return NULL;
        }
        memset(pb, 0, sizeof(mt_pool));
        p->pl = (mt_pool *)pb;
        p->pl->nthr = maxt;
        p->pl->tw = maxt;
        p->pl->emult = (unsigned long)batch + 8;
        for (int t = 1; t < maxt; ++t) {
            p->pl->wa[t].p = p;
            p->pl->wa[t].tid = t;
            p->pl->wa[t].cpu = cpus[t];
            if (pthread_create(&p->pl->th[t], NULL, mt_worker,
                               &p->pl->wa[t]) != 0) {
                p->pl->nthr = t; /* join only what was spawned */
                break;
            }
        }
        maxt = p->pl->nthr;
        while (atomic_load(&p->pl->ready) < maxt - 1) cpu_relax();
        if (atomic_load(&p->pl->fail)) {
            fft3d_destroy(p);
            return NULL;
        }
    }

    /* regime: does the batch stream through one socket's LLC?  (The scored
     * node's caller buffers all live on socket 0 -- the driver freads and
     * memsets them on the main thread -- so its 22 MiB is the honest
     * denominator even with 32 threads on two sockets.) */
    long l3 = -1;
    {
#ifdef _SC_LEVEL3_CACHE_SIZE
        l3 = sysconf(_SC_LEVEL3_CACHE_SIZE);
#endif
        if (l3 <= 0) l3 = 22l << 20;
    }
    double foot = (double)batch * (double)NVOL * 16.0 * 2.0;
    int streaming = batch > 1 && foot > 1.25 * (double)l3;

    /* ---- candidate pools ---- */
    mt_cand cand[16];
    int nc = 0;
#define ADD(MO, TT, GG, VV, MM) do {                                          \
        if (nc < 16) {                                                        \
            mt_cand *c = &cand[nc];                                           \
            c->mode = (MO); c->T = (TT); c->G = (GG);                         \
            c->v = (VV); c->m = (MM);                                         \
            if (c->T > maxt) c->T = maxt;                                     \
            if (c->mode == 2) {                                               \
                if (c->G > c->T) c->G = c->T;                                 \
                c->T = (c->T / c->G) * c->G;                                  \
            } else c->G = 1;                                                  \
            if ((c->mode == 1 || c->mode == 4) && c->T > batch)               \
                c->T = batch;                                                 \
            if (c->T < 2) { c->mode = 0; c->T = 1; c->G = 1; }                \
            if ((c->v == 1 && !have_512) || (c->v == 2 && !have_vl))          \
                c->v = defv;                                                  \
            cand_tag(c);                                                      \
            int dup = 0;                                                      \
            for (int u = 0; u < nc; ++u)                                      \
                if (!strcmp(cand[u].tag, c->tag)) dup = 1;                    \
            if (!dup) ++nc;                                                   \
        }                                                                     \
    } while (0)

    /* env overrides for monitor A/Bs: any of FFT45_MODE/T/G/V/MECH forces a
     * single candidate (still correctness-gated). FFT45_VERBOSE prints. */
    {
        const char *e;
        int fmode = -1, fT = -1, fG = -1, fV = -1, fM = -1, any = 0;
        if ((e = getenv("FFT45_MODE"))) { fmode = atoi(e); any = 1; }
        if ((e = getenv("FFT45_T")))    { fT = atoi(e);    any = 1; }
        if ((e = getenv("FFT45_G")))    { fG = atoi(e);    any = 1; }
        if ((e = getenv("FFT45_V")))    { fV = atoi(e);    any = 1; }
        if ((e = getenv("FFT45_MECH"))) { fM = atoi(e);    any = 1; }
        if (any) {
            int mo = (fmode >= 0 && fmode <= 4) ? fmode
                                                : (batch == 1 ? 2 : 1);
            int T = fT > 0 ? fT : MAXT;
            int G = fG > 0 ? fG : (mo == 2 ? (batch == 1 ? T : 2) : 1);
            int v = (fV >= 0 && fV <= 2) ? fV : defv;
            int m = (fM >= 0 && fM <= 10) ? fM : 0;
            ADD(mo, T, G, v, m);
        } else if (batch == 1) {
            /* one volume, latency-bound: grouped intra-volume splits over a
             * T-sweep (barrier cost vs 45-plane / 507-tile granularity move
             * against each other), plus the serial floor.  mt_r2: T=28/20
             * added -- on the node the 32 threads span two sockets and the
             * best team size is a cross-socket-barrier question the wallaby
             * sweep cannot answer; widen it and let the node's race say. */
            ADD(2, 32, 32, 1, 0);
            ADD(2, 23, 23, 1, 0);
            ADD(2, 28, 28, 1, 0);
            ADD(2, 16, 16, 1, 0);
            ADD(2, 20, 20, 1, 0);
            ADD(2,  8,  8, 1, 0);
            ADD(2, 23, 23, 2, 0);
            ADD(2, 32, 32, 2, 0);
            ADD(0,  1,  1, 1, 0);
        } else if (!streaming) {
            /* small cached batch; one vnt probe (mt_r2) in case the cell is
             * closer to the wall than the L3 heuristic thinks; one m8 row
             * (mt_r3) so the pfnx mechanism is priced in the cached regime */
            ADD(1, 32,  1, 1, 0);
            ADD(1, 32,  1, 2, 0);
            ADD(2, 32,  2, 1, 0);
            ADD(2, 32,  2, 2, 0);
            ADD(2, 32,  2, 2, 8);
            ADD(2, 32,  4, 1, 0);
            ADD(2, 32, 32, 1, 0);
            ADD(3, 32,  1, 2, 6);
            ADD(0,  1,  1, 1, 0);
        } else {
            /* streaming, mt_r3 reworked around the node's r2 tables and the
             * VERDICT's placement finding.
             * B=256 there: my vnt32-v2-m0 52.2/51.5/53.5 vs L45_pfa's
             * static-block mtn 45.4 driver -- and their r1 mtn hit 26.9,
             * which per VERDICT S5 needs the spread-page regime that only a
             * CALL-STABLE volume->thread map lets AutoNUMA reach.  So the
             * static vns class leads (earliest-wins hysteresis: static and
             * dynamic tie in the pre-migration regime the arena races in,
             * and static is the one that can improve during the driver's
             * loop).  pfnx rows adopt L45_pfa's next-input pre-cover; m0
             * rows respect the node's dislike of my pfin pacing (52.2 vs
             * 56.8).  grp16x2 rows serve node B=16 (r2 winner shape) --
             * mt_r3 adds their winning poke+pfnx mech (m8) and restores the
             * v1 width (my r1 B=16 winner; L45_pfa's r2 win was pw4).
             * DROPPED (took no picks r1+r2, dead by >=10% in every node
             * table): all mode-1 vol rows, grp8x4, vnt16, the cpy family. */
            ADD(4, 32, 1, 2, 10);
            ADD(4, 32, 1, 2, 0);
            ADD(4, 32, 1, 2, 9);
            ADD(3, 32, 1, 2, 0);
            ADD(4, 32, 1, 1, 10);
            ADD(2, 32, 2, 2, 8);
            ADD(2, 32, 2, 2, 1);
            ADD(2, 32, 2, 1, 8);
            ADD(2, 32, 2, 1, 1);
            ADD(3, 32, 1, 2, 7);
        }
    }
#undef ADD

    /* ---- tuning arena.  Streaming arenas must actually stream on the
     * machine doing the tuning, and must have >= 32 volumes so a T=32
     * volume-parallel candidate is exercised at full width. ---- */
    long nt;
    if (batch == 1) nt = 1;
    else if (!streaming) nt = batch < 12 ? batch : 12;
    else {
        /* mt_r2: the r1 rule (4x L3, clamp [32,96]) gave the node a
         * 32-volume arena that priced the picked candidate at 25.4 us/vol
         * where the driver's B=256 measured 79.1 -- the same trap L45_pfa
         * mt_r1 hit and fixed by raising the arena to 128 volumes (their
         * arena and driver then agreed).  Adopt their number: as close to
         * the real batch as setup time affords, floor 32 so a T=32
         * volume-parallel candidate still runs at full width. */
        long arena = 128;
        nt = batch < arena ? batch : arena;
    }

    size_t nd = (size_t)NVOL * 2 * (size_t)nt;
    double *ti = NULL, *o0 = NULL, *ox = NULL;
    if (posix_memalign((void **)&ti, 4096, nd * sizeof(double)) == 0 &&
        posix_memalign((void **)&o0, 4096, nd * sizeof(double)) == 0 &&
        posix_memalign((void **)&ox, 4096, nd * sizeof(double)) == 0) {

        /* deterministic data, filled SERIALLY on purpose: the driver freads
         * `in` on its main thread, so the real buffers are socket-0
         * resident on the node; the arena must reproduce that placement. */
        uint64_t s = 0x9E3779B97F4A7C15ull;
        for (size_t i = 0; i < nd; ++i) {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            ti[i] = (double)(int64_t)(s >> 11) * (1.0 / 9007199254740992.0);
        }

        /* serial v0 reference for the correctness gate */
        long realb = p->batch;
        p->batch = nt;
        {
            int sm = p->mode, sT = p->T, sG = p->G;
            p1_fn s1 = p->p1; p2_fn s2 = p->p2; long sn = p->ntt;
            p->mode = 0; p->T = 1; p->G = 1;
            p->p1 = P1T[0][0]; p->p2 = P2T[0][0]; p->ntt = NTTW[0];
            memset(o0, 0, nd * sizeof(double));
            mt_run(p, ti, o0);
            p->mode = sm; p->T = sT; p->G = sG;
            p->p1 = s1; p->p2 = s2; p->ntt = sn;
        }

        /* gate every candidate at 1e-13 relative (they are all one bit
         * class with the reference, so this doubles as a threading check) */
        int ok[16];
        for (int k = 0; k < nc; ++k) {
            p->mode = cand[k].mode; p->T = cand[k].T; p->G = cand[k].G;
            p->p1 = P1T[cand[k].v][cand[k].m];
            p->p2 = P2T[cand[k].v][cand[k].m];
            p->ntt = NTTW[cand[k].v];
            memset(ox, 0, nd * sizeof(double));
            mt_run(p, ti, ox);
            double num = 0.0, den = 0.0;
            for (size_t i = 0; i < nd; ++i) {
                double d = ox[i] - o0[i];
                num += d * d;
                den += o0[i] * o0[i];
            }
            ok[k] = den > 0.0 && sqrt(num / den) < 1e-13;
        }

        /* time the survivors: interleaved rounds, per-candidate min (the
         * phase-1 fast/slow-window lesson: never read one window) */
        double best[16];
        for (int k = 0; k < nc; ++k) best[k] = 1e300;
        int reps = nt >= 16 ? 1 : (nt >= 4 ? 4 : 16);
        int rounds = nt >= 16 ? 5 : (nt >= 4 ? 6 : 10);
        for (int round = 0; round < rounds; ++round) {
            for (int k = 0; k < nc; ++k) {
                if (!ok[k]) continue;
                p->mode = cand[k].mode; p->T = cand[k].T; p->G = cand[k].G;
                p->p1 = P1T[cand[k].v][cand[k].m];
                p->p2 = P2T[cand[k].v][cand[k].m];
                p->ntt = NTTW[cand[k].v];
                mt_run(p, ti, ox);              /* warm / AVX-512 licence */
                double t0 = now_s();
                for (int r = 0; r < reps; ++r) mt_run(p, ti, ox);
                double dt = now_s() - t0;
                if (dt < best[k]) best[k] = dt;
            }
        }
        p->batch = realb;

        if (getenv("FFT45_VERBOSE"))
            for (int k = 0; k < nc; ++k)
                fprintf(stderr, "cand %-22s %s best %.3f us/vol\n",
                        cand[k].tag, ok[k] ? "ok " : "BAD",
                        best[k] * 1e6 / ((double)reps * (double)nt));

        /* hysteresis pick, earlier-candidate-wins: 3% cached (the wallaby
         * fast/slow toggle), 2% streaming */
        double hyst = streaming ? 0.98 : 0.97;
        int bk = -1;
        for (int k = 0; k < nc; ++k) {
            if (!ok[k]) continue;
            if (bk < 0 || best[k] < hyst * best[bk]) bk = k;
        }
        if (bk < 0) { /* nothing passed: serial default-width fallback */
            p->mode = 0; p->T = 1; p->G = 1;
            p->p1 = P1T[defv][0]; p->p2 = P2T[defv][0]; p->ntt = NTTW[defv];
            snprintf(g_desc, sizeof g_desc,
                     "MT PFA9x5; GATE FAILED, serial fallback (B=%d)", batch);
        } else {
            p->mode = cand[bk].mode; p->T = cand[bk].T; p->G = cand[bk].G;
            p->p1 = P1T[cand[bk].v][cand[bk].m];
            p->p2 = P2T[cand[bk].v][cand[bk].m];
            p->ntt = NTTW[cand[bk].v];
            int off = snprintf(g_desc, sizeof g_desc,
                               "MT PFA9x5 pool; pick=%s (B=%d nt=%ld str=%d "
                               "thr=%d nc=%d);", cand[bk].tag, batch, nt,
                               streaming, maxt, nc);
            for (int k = 0; k < nc && off > 0 && off < (int)sizeof g_desc - 1;
                 ++k)
                off += snprintf(g_desc + off, sizeof g_desc - (size_t)off,
                                " %s=%.1f", cand[k].tag,
                                ok[k] ? best[k] * 1e6 /
                                            ((double)reps * (double)nt)
                                      : -1.0);
        }
    }
    free(ti);
    free(o0);
    free(ox);
    return p;
}

void fft3d_execute(fft3d_plan *plan, const double _Complex *in, double _Complex *out)
{
    /* placement scan: 1st call and every 24th, only when an NT-staged mode
     * was picked (the streaming cells).  ~128 read-only syscalls ~ 0.1 ms,
     * i.e. ~1% of ONE 12 ms B=256 call out of every 24 -- invisible to the
     * driver's min-over-samples.  The JSON description is written after
     * the timed loop, so `fr` here is the POST-loop placement: fr0 > 0 or
     * frl > fr0 means the spread regime is live at this cell. */
    ++plan->ncall;
    if ((plan->mode == 3 || plan->mode == 4) && plan->main_node >= 0 &&
        (plan->ncall == 1 || plan->ncall % 24 == 0)) {
        int fr = gov_scan_remote((const double *)in, (double *)out,
                                 plan->batch, plan->main_node);
        if (plan->nscan == 0) plan->fr0 = fr;
        plan->frl = fr;
        ++plan->nscan;
        snprintf(g_gov, sizeof g_gov, " gov{nb=%d,fr0=%d,fr=%d,sc=%d,n=%ld}",
                 plan->nb_on, plan->fr0, plan->frl, plan->nscan, plan->ncall);
    }
    mt_run(plan, (const double *)in, (double *)out);
}

void fft3d_destroy(fft3d_plan *plan)
{
    if (!plan) return;
    if (plan->pl) {
        atomic_store(&plan->pl->shutdown, 1);
        ++plan->pl->gen_local;
        atomic_store_explicit(&plan->pl->gen, plan->pl->gen_local,
                              memory_order_release);
        for (int t = 1; t < plan->pl->nthr; ++t)
            pthread_join(plan->pl->th[t], NULL);
        free(plan->pl);
    }
    for (int t = 0; t < MAXT; ++t) free(plan->ctx[t].blk);
    free(plan);
}

#else /* ================= per-variant instantiation ======================== */

#define XCAT2(a, b) a##b
#define XCAT(a, b) XCAT2(a, b)
#define FN(n)     XCAT(XCAT(n##_, VAR), _0)
#define FNC(n, c) XCAT(XCAT(XCAT(n##_, VAR), _), c)

#if VAR == 1
/* ---- 512-bit: 4 complex lanes per zmm, 32 registers.  avx512dq is needed
 * for the insertf64x2/extractf64x2 odd-column forms; fft3d_create() gates
 * on it. ---- */
#pragma GCC push_options
#pragma GCC target("avx512f,avx512dq")
#define PW 4
#define VD __m512d
#define VLOAD(p)          _mm512_loadu_pd(p)
#define VSTORE(p, v)      _mm512_storeu_pd((p), (v))
#define VLOADT(p)         _mm512_maskz_loadu_pd((__mmask8)0x03, (p))
#define VSTORET(p, v)     _mm512_mask_storeu_pd((p), (__mmask8)0x03, (v))
#define VADD(a, b)        _mm512_add_pd((a), (b))
#define VSUB(a, b)        _mm512_sub_pd((a), (b))
#define VMUL(a, b)        _mm512_mul_pd((a), (b))
#define VFMADD(a, b, c)   _mm512_fmadd_pd((a), (b), (c))
#define VFMSUB(a, b, c)   _mm512_fmsub_pd((a), (b), (c))
#define VFNMADD(a, b, c)  _mm512_fnmadd_pd((a), (b), (c))
#define VSWAP(a)          _mm512_permute_pd((a), 0x55)
#define PWLIST(M)         M(0) M(1) M(2) M(3)
/* odd-column (z=44 / k=44) forms, r9: lane j of the vector <-> the 128-bit
 * complex at p + j*STR. */
#define LDCOL(p, STR)                                                     \
    _mm512_insertf64x2(_mm512_insertf64x2(_mm512_insertf64x2(             \
        _mm512_castpd128_pd512(_mm_loadu_pd(p)),                          \
        _mm_loadu_pd((p) + (STR)), 1),                                    \
        _mm_loadu_pd((p) + 2 * (STR)), 2),                                \
        _mm_loadu_pd((p) + 3 * (STR)), 3)
#define STCOL(p, STR, v) do {                                             \
    _mm_storeu_pd((p),               _mm512_castpd512_pd128(v));          \
    _mm_storeu_pd((p) +     (STR),   _mm512_extractf64x2_pd((v), 1));     \
    _mm_storeu_pd((p) + 2 * (STR),   _mm512_extractf64x2_pd((v), 2));     \
    _mm_storeu_pd((p) + 3 * (STR),   _mm512_extractf64x2_pd((v), 3));     \
} while (0)
/* 4x4 transpose of 128-bit complex lanes (involution) */
#define TRANSP(A, B) do {                                                 \
    VD z0 = _mm512_shuffle_f64x2((A)[0], (A)[1], 0x88);                   \
    VD z1 = _mm512_shuffle_f64x2((A)[0], (A)[1], 0xDD);                   \
    VD z2 = _mm512_shuffle_f64x2((A)[2], (A)[3], 0x88);                   \
    VD z3 = _mm512_shuffle_f64x2((A)[2], (A)[3], 0xDD);                   \
    (B)[0] = _mm512_shuffle_f64x2(z0, z2, 0x88);                          \
    (B)[2] = _mm512_shuffle_f64x2(z0, z2, 0xDD);                          \
    (B)[1] = _mm512_shuffle_f64x2(z1, z3, 0x88);                          \
    (B)[3] = _mm512_shuffle_f64x2(z1, z3, 0xDD);                          \
} while (0)
#else
/* ---- 256-bit: 2 complex lanes per ymm.  VAR 0 = "AVX2" (16 registers on a
   genuinely AVX2-only host), VAR 2 = EVEX/AVX-512VL (32 registers, no
   512-bit path).  KNOWN AND ACCEPTED (phase-1 r9 audit): on an AVX-512
   build host VAR 0 compiles EVEX and folds into VAR 2; v0 is fielded only
   on hosts without AVX-512. */
#pragma GCC push_options
#if VAR == 0
#pragma GCC target("avx2,fma")
#else
#pragma GCC target("avx512vl,avx512f,fma")
#endif
#define PW 2
#define VD __m256d
#define VLOAD(p)          _mm256_loadu_pd(p)
#define VSTORE(p, v)      _mm256_storeu_pd((p), (v))
#define VLOADT(p)         _mm256_maskload_pd((p), _mm256_load_si256((const __m256i *)KC_TMASK))
#define VSTORET(p, v)     _mm256_maskstore_pd((p), _mm256_load_si256((const __m256i *)KC_TMASK), (v))
#define VADD(a, b)        _mm256_add_pd((a), (b))
#define VSUB(a, b)        _mm256_sub_pd((a), (b))
#define VMUL(a, b)        _mm256_mul_pd((a), (b))
#define VFMADD(a, b, c)   _mm256_fmadd_pd((a), (b), (c))
#define VFMSUB(a, b, c)   _mm256_fmsub_pd((a), (b), (c))
#define VFNMADD(a, b, c)  _mm256_fnmadd_pd((a), (b), (c))
#define VSWAP(a)          _mm256_permute_pd((a), 0x5)
#define PWLIST(M)         M(0) M(1)
/* odd-column forms, 2-lane widths: plain AVX insert/extract */
#define LDCOL(p, STR)                                                     \
    _mm256_insertf128_pd(_mm256_castpd128_pd256(_mm_loadu_pd(p)),         \
                         _mm_loadu_pd((p) + (STR)), 1)
#define STCOL(p, STR, v) do {                                             \
    _mm_storeu_pd((p),         _mm256_castpd256_pd128(v));                \
    _mm_storeu_pd((p) + (STR), _mm256_extractf128_pd((v), 1));            \
} while (0)
/* 2x2 transpose of 128-bit complex lanes; involution */
#define TRANSP(A, B) do {                                        \
    VD z0 = _mm256_permute2f128_pd((A)[0], (A)[1], 0x20);        \
    VD z1 = _mm256_permute2f128_pd((A)[0], (A)[1], 0x31);        \
    (B)[0] = z0; (B)[1] = z1;                                    \
} while (0)
#endif

#define NB   ((LSIDE + PW - 1) / PW)   /* blocks per pass: 12 (PW=4), 23 (PW=2) */
#define NGF  (LSIDE / PW)              /* full blocks: 11 (PW=4), 22 (PW=2)     */
#define PFCH ((PFLINES + NB - 1) / NB) /* prefetch lines per paced call         */
#define NTFULL (NPLANE / PW)           /* full phase-2 tiles: 506 / 1012        */
/* phase 1's z and y subloops run NGF full groups + one PW=1 tail line;
 * -DFFT45_R10TAIL restores the r6-r10 overlap-recompute tails. */
#ifdef FFT45_R10TAIL
#define NGRP NB
#else
#define NGRP NGF
#endif

/* 45-point PFA line transform, y axis: reads the padded plane scratch,
 * writes `out` (or the cpy plane image) at 45-complex stride. */
static inline __attribute__((always_inline))
void FN(dft45_y)(const double *src, double *dst)
{
    VD T[45];
#define LSRC(i)     VLOAD(src + (long)(i) * (PPITCH * 2))
#define SDST(i, v)  VSTORE(dst + (long)(i) * (LSIDE * 2), (v))
    REP9(ST1)
    REP5(ST2)
#undef LSRC
#undef SDST
}

/* 45-point PFA line transform, x axis, in place: stride 2025 complex. */
static inline __attribute__((always_inline))
void FN(dft45_x)(double *base)
{
    VD T[45];
#define LSRC(i)     VLOAD(base + (long)(i) * (NPLANE * 2))
#define SDST(i, v)  VSTORE(base + (long)(i) * (NPLANE * 2), (v))
    REP9(ST1)
    REP5(ST2)
#undef LSRC
#undef SDST
}

/* masked tail of the x pass: the single line at flat index 2024 = (y,z) =
 * (44,44).  Only lane 0 is live; dead lanes compute on garbage and are
 * masked off at the store.  One call per volume. */
static inline __attribute__((always_inline))
void FN(dft45_xt)(double *base)
{
    VD T[45];
#define LSRC(i)     VLOADT(base + (long)(i) * (NPLANE * 2))
#define SDST(i, v)  VSTORET(base + (long)(i) * (NPLANE * 2), (v))
    REP9(ST1)
    REP5(ST2)
#undef LSRC
#undef SDST
}

/* 45-point PFA line transform, vector array in/out (z axis) */
static inline __attribute__((always_inline))
void FN(dft45_v)(const VD *X, VD *Y)
{
    VD T[45];
#define LSRC(i)     X[(i)]
#define SDST(i, v)  Y[(i)] = (v)
    REP9(ST1)
    REP5(ST2)
#undef LSRC
#undef SDST
}

/* phase 1 (z pass + y pass) for x-planes [x0, x1) of one volume.  The
 * always_inline body is instantiated with compile-time-constant mechanism
 * flags by the MKPR wrappers below. */
static inline __attribute__((always_inline))
void FN(p1body)(const double *vin, double *vout, double *plane,
                double *plane2, long x0, long x1,
                const double *inend, double *outend,
                const int pfin, const int pfw, const int cpy)
{
    for (long x = x0; x < x1; ++x) {
        const double *pin  = vin  + x * (long)NPLANE * 2;
        double       *pout = vout + x * (long)NPLANE * 2;
        /* the input/output planes one ahead of the ones in flight */
        const double *npf = pin  + (long)NPLANE * 2;
        double       *npw = pout + (long)NPLANE * 2;

        /* z pass: lanes = PW consecutive y-rows; NGF full groups, then ONE
         * true PW=1 (xmm) tail line for y=44 */
        for (long yb = 0; yb < NGRP; ++yb) {
            long y0 = yb * PW;
#ifdef FFT45_R10TAIL
            if (y0 > LSIDE - PW) y0 = LSIDE - PW;
#endif
            if (pfin) {
                /* paced positional cursor: cover one plane of in, one plane
                 * ahead, spread over this subloop's calls */
                long s = yb * PFCH, e = s + PFCH;
                if (e > PFLINES) e = PFLINES;
#pragma GCC unroll 8
                for (long i = s; i < e; ++i) {
                    const double *q = npf + i * 8;
                    if (q < inend)
                        _mm_prefetch((const char *)q, _MM_HINT_T1);
                }
            }
            VD Xv[LSIDE], Yv[LSIDE];
            const double *rows = pin + y0 * (LSIDE * 2);
            /* explicit unrolls: the monitor's build has no -funroll-loops,
             * and leaving these rolled cost 30% end to end (measured) */
#pragma GCC unroll 22
            for (long g = 0; g < NGF; ++g) {
                VD A[PW], B[PW];
#define LDR(j) A[j] = VLOAD(rows + (long)(j) * (LSIDE * 2) + g * (PW * 2));
                PWLIST(LDR)
#undef LDR
                TRANSP(A, B);
#define PUT(j) Xv[g * PW + (j)] = B[j];
                PWLIST(PUT)
#undef PUT
            }
            /* odd 45th column, z = 44: 128-bit column gather (r9 LDCOL) */
            Xv[44] = LDCOL(rows + 44 * 2, LSIDE * 2);
            FN(dft45_v)(Xv, Yv);
            double *prow = plane + y0 * (PPITCH * 2);
#pragma GCC unroll 22
            for (long g = 0; g < NGF; ++g) {
                VD A[PW], B[PW];
#define GET(j) A[j] = Yv[g * PW + (j)];
                PWLIST(GET)
#undef GET
                TRANSP(A, B);
#define PST(j) VSTORE(prow + (long)(j) * (PPITCH * 2) + g * (PW * 2), B[j]);
                PWLIST(PST)
#undef PST
            }
            /* odd column scatter: lane j of Yv[44] -> row j's z=44 slot */
            STCOL(prow + 44 * 2, PPITCH * 2, Yv[44]);
        }
#ifndef FFT45_R10TAIL
        if (pfin) {
            /* the pacing chunks the removed 12th group would have issued */
            long s = (long)NGF * PFCH;
#pragma GCC unroll 8
            for (long i = s; i < PFLINES; ++i) {
                const double *q = npf + i * 8;
                if (q < inend)
                    _mm_prefetch((const char *)q, _MM_HINT_T1);
            }
        }
        dft45_tz1(pin + 44 * (LSIDE * 2), plane + 44 * (PPITCH * 2));
#endif

        /* y pass: lanes = PW consecutive z.  With cpy the stores go to the
         * L1-hot plane image instead of cold `out`, and one ERMS rep-movsb
         * per plane moves the image out without the RFO read. */
        double *ydst = cpy ? plane2 : pout;
        for (long zb = 0; zb < NGRP; ++zb) {
            long z0 = zb * PW;
#ifdef FFT45_R10TAIL
            if (z0 > LSIDE - PW) z0 = LSIDE - PW;
#endif
            if (pfw) {
                long s = zb * PFCH, e = s + PFCH;
                if (e > PFLINES) e = PFLINES;
#pragma GCC unroll 8
                for (long i = s; i < e; ++i) {
                    double *q = npw + i * 8;
                    if (q < outend)
                        __builtin_prefetch(q, 1, 3);
                }
            }
            FN(dft45_y)(plane + z0 * 2, ydst + z0 * 2);
        }
#ifndef FFT45_R10TAIL
        if (pfw) {
            long s = (long)NGF * PFCH;
#pragma GCC unroll 8
            for (long i = s; i < PFLINES; ++i) {
                double *q = npw + i * 8;
                if (q < outend)
                    __builtin_prefetch(q, 1, 3);
            }
        }
        dft45_ty1(plane + 44 * 2, ydst + 44 * 2);
#endif
        if (cpy)
            plane_copy(pout, plane2, (size_t)NPLANE * 2 * sizeof(double));
    }
}

/* phase 2: x-lines, in place in `out`, tiles [t0, t1) of the FLAT (y,z)
 * index; tile NTFULL (the last, when t1 > NTFULL) is the single masked
 * tail line at flat index 2024. */
static inline __attribute__((always_inline))
void FN(p2body)(double *vout, long t0, long t1, const double *nx,
                const int pf, const int pfn)
{
    const double *pn = nx;      /* pfnx cursor over the next volume's input */
    long tf = t1 < NTFULL ? t1 : NTFULL;
    for (long t = t0; t < tf; ++t) {
        double *base = vout + t * (PW * 2);
        if (pf) {
            /* 45 x-streams, each advancing 64 B per tile: more than the L2
             * streamer tracks; poke them one line ahead */
#pragma GCC unroll 45
            for (int i = 0; i < LSIDE; ++i)
                _mm_prefetch((const char *)(base + (long)i * (NPLANE * 2) + 8),
                             _MM_HINT_T0);
        }
        if (pfn && pn) {
            /* pre-cover the NEXT volume's input head, 2 T1 lines per tile
             * (~63 KB over a full range) -- BORROWED from L45_pfa's PFNX;
             * phase 2 otherwise leaves the DRAM read stream idle */
            _mm_prefetch((const char *)pn, _MM_HINT_T1);
            _mm_prefetch((const char *)(pn + 8), _MM_HINT_T1);
            pn += 16;
        }
        FN(dft45_x)(base);
    }
    if (t1 > NTFULL)
        FN(dft45_xt)(vout + (long)(NPLANE - 1) * 2);
}

/* mechanism wrappers: m0 none, m1 pfin+pfw, m2 cpy, m3 pfin+pfw+p2 poke,
 * m4 cpy+pfin, m5 cpy+pfin+p2 poke, m6 pfin only, m7 pfin+p2 poke.  pfw is
 * never combined with cpy: with the y pass storing to the plane image, a
 * prefetchw on `out` would reintroduce exactly the RFO that cpy's
 * rep-movsb deletes.  m6/m7 (mt_r2) are the vnt mechanisms: the staging
 * volume M is cache-resident, so prefetching it is a pure uop tax
 * (L45_pfa mt_r1 / L23_matrixsimd mt_r1 lesson) -- only the DRAM `in`
 * stream is paced; m7 adds the phase-2 45-stream poke for the part of M
 * that spills the node's 1 MiB L2.
 * mt_r3, BORROWED from L45_pfa's node-winning pfi/pfw mechanisms:
 * m8 = m3 + pfnx (their B=16 g2-pfw analog: pfin+prefetchw+poke+next-input
 * pre-cover), m9 = pfin + pfnx (their B=256 mtn-pfi analog), m10 = pfnx
 * alone (the node's r2 vnt table preferred m0 over every pfin mechanism --
 * 52.2 vs 56.8 us/vol -- so the pre-cover is also fielded without pfin). */
#define MKPR(code, pfi, pfww, cpyy, pff, pfnn)                                \
static void FNC(p1r, code)(const double *vin, double *vout, double *plane,    \
                           double *plane2, long x0, long x1,                  \
                           const double *inend, double *outend)               \
{                                                                             \
    FN(p1body)(vin, vout, plane, plane2, x0, x1, inend, outend,               \
               pfi, pfww, cpyy);                                              \
}                                                                             \
static void FNC(p2r, code)(double *vout, long t0, long t1, const double *nx)  \
{                                                                             \
    FN(p2body)(vout, t0, t1, nx, pff, pfnn);                                  \
}
MKPR(0, 0, 0, 0, 0, 0)
MKPR(1, 1, 1, 0, 0, 0)
MKPR(2, 0, 0, 1, 0, 0)
MKPR(3, 1, 1, 0, 1, 0)
MKPR(4, 1, 0, 1, 0, 0)
MKPR(5, 1, 0, 1, 1, 0)
MKPR(6, 1, 0, 0, 0, 0)
MKPR(7, 1, 0, 0, 1, 0)
MKPR(8, 1, 1, 0, 1, 1)
MKPR(9, 1, 0, 0, 0, 1)
MKPR(10, 0, 0, 0, 0, 1)
#undef MKPR

#pragma GCC pop_options

#undef XCAT2
#undef XCAT
#undef FN
#undef FNC
#undef PW
#undef VD
#undef VLOAD
#undef VSTORE
#undef VLOADT
#undef VSTORET
#undef VADD
#undef VSUB
#undef VMUL
#undef VFMADD
#undef VFMSUB
#undef VFNMADD
#undef VSWAP
#undef PWLIST
#undef LDCOL
#undef STCOL
#undef TRANSP
#undef NB
#undef NGF
#undef PFCH
#undef NTFULL
#undef NGRP

#endif /* VAR */
