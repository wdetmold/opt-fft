/* =============================================================================
 * L23_rader -- 23^3 complex-double forward DFT; Rader realization for p = 23.
 * MULTICORE phase (mt_r2).  Phase-1 history: ../../geom/strategies/L23_rader.md.
 *
 * THE ARITHMETIC (settled in phase 1, untouched here):
 * p-1 = 22 = 2*11, so Rader's length-22 cyclic convolution splits, via
 * g^11 = -1 (g = 5 primitive root mod 23), into a CYCLIC-11 correlation with
 * the real kernel cos(2*pi*5^q/23) on the folded even part u_j = x_j+x_{23-j}
 * plus one with sin(2*pi*5^q/23) on w_j = -i(x_j - x_{23-j}).  Every
 * sub-quadratic realization of the length-11 convolution known to the corpus
 * exceeds the 121 fused FMAs of the direct circulant matvec on FMA hardware,
 * so the optimal Rader-23 IS the conjugate-folded direct form.  Per line:
 * 253 FMA + 44 add/sub = 297 vector FP ops; per volume 3*529 lines ->
 * 943 kflop (yardstick 5*N*log2(N) = 824 kflop).  The -i rotation's sign
 * lives in the sine tables (alternating {+s,-s} lanes -- phase-1 r10, from
 * L13_direct r9), so there is no vpxor anywhere.
 *
 * WHAT PHASE 1 SETTLED, KEPT VERBATIM:
 *   krn_ts    two-sweep pinned-constant kernel (11 cosines then 11 sines in
 *             registers), SIMD lanes = WC adjacent independent lines.
 *   X-first   pass order everywhere: X (in -> t1, plane stride padded to
 *             1064 doubles = 133 whole lines), then per plane x: Y (into a
 *             23x24-padded plane buffer), Z (transposing store into out).
 *   pf=2      in-pass X prefetch 4 chunks ahead (the 8464-B plane stride
 *             defeats page-local hardware prefetch); pw=1 write-intent
 *             prefetch of the out plane, half before Y, half between Y and Z
 *             ("hide the RFO, don't avoid it" -- VERDICT r5 4.5).  Node
 *             mt_r1 lesson: at 32-thread streaming BOTH knobs lose (pf0 pw0
 *             picked, 3.40 vs 5.51 us/t) -- heads updated, all combos raced.
 *   NT        staging-plane + streaming-store variant, kept raced: wallaby
 *             streaming winner mt_r1 (1.74 vs 2.00), NOT picked on the node.
 *   Retired for good (documented streaming nulls, do not rebuild): rp-t1,
 *   deferred-Z, uniform + tail-paced pipelining, pf=1, krn_il, P-parking,
 *   X-last.  See phase-1 record, VERDICT r10 7.
 *
 * WHAT IS NEW IN mt_r2 (threading layer only; per-volume DAG identical, so
 * output is bit-identical to mt_r1 and to the serial exec at every team
 * size):
 *   - PERSISTENT SPIN POOL instead of per-call OpenMP regions, adopted from
 *     L23_matrixsimd mt_r1 (their measurement: one GOMP fork+barrier+join
 *     costs 6.2-8.2 us at T=8..32 -- several times the 32-way-parallel
 *     compute of one volume).  fft3d_create() reads the harness's
 *     thread->CPU map from one throwaway OMP region (so the pool pins
 *     exactly where OMP_PROC_BIND=close / OMP_PLACES=cores puts threads),
 *     spawns 31 pthreads once, and execute() publishes a job and
 *     release-stores one generation word.  Barriers are per-thread padded
 *     arrival flags collected by tid 0 plus one release word (their
 *     flag-array design; a central fetch-add barrier measured ~2.5-5 us).
 *     Workers never sleep; OpenMP is never entered after create().
 *   - mode=batch / batchNT: thread t owns the contiguous volume block
 *     [nb*t/T, nb*(t+1)/T) on its own NUMA-local slot; one join flag per
 *     thread, no other sync.  Same range exec as mt_r1.
 *   - mode=fused (batch < 32): all batch*132 X items (the overlapping tail
 *     chunk is folded into the last item so no line is written by two
 *     threads), ONE mid barrier, all batch*23 (v,x) planes, join.  t1 now
 *     lives in a dedicated fused arena FIRST-TOUCHED BY THE POOL with the
 *     same static plane partition the plane loop uses (reader-owned pages,
 *     L23_matrixsimd's design), not in slot v -- at B=1 slot 0's pages all
 *     sat on one thread's node.
 *   - mode=serial kept and raced (the honest B=1 floor; note it now runs
 *     with 31 workers spinning, which is also what the scored run sees).
 *   - deterministic joint-cell hysteretic tuner unchanged in mechanism
 *     (cells = mode x width x team x pf x pw, two fixed-order sweeps,
 *     per-cell min, >2% to displace, 1.5 ms licence warmups).  Streaming
 *     head re-set to the NODE's mt_r1 pick (batch pf0 pw0); B<32 heads are
 *     fused-on-pool (L23_matrixsimd's measured mid-batch winner).
 *
 * WHAT IS NEW IN mt_r3 (dispatch + tuner surface only; DAG untouched, all
 * cells stay one bit class):
 *   - AGGREGATE-CACHE tuner arena (BORROWED: L23_matrixsimd mt_r2 /
 *     L6_pfa mt_r1): the streaming arena now exceeds nthr*L2 + nsock*L3
 *     (x3.5, cap 640 volumes), not 2.5x one L3.  mt_r2's node arena was
 *     nv=148 = 58 MiB, INSIDE the node's 76 MiB aggregate (32x1 MiB L2 +
 *     2x22 MiB L3): it priced out-RFOs at L3 speed, picked plain batch,
 *     and scored 7.17 us/vol at B=2048 where the rival's honest nv=640
 *     arena picked NT and scored 5.93.  The kernel was never the problem;
 *     the arena lied about the regime.
 *   - TWO-LEVEL (socket-tree) BARRIER, raced per-cell (bar=1): the far
 *     socket's lowest tid collects its socket's arrival flags locally and
 *     posts ONE flag to tid 0; releases relay through a second word
 *     (rel2) the far side spins on.  Cuts tid 0's remote flag reads from
 *     16 to 1 per episode on the node (both L23 entries' scored B=1 sits
 *     at fused T=16 because the flat 32-wide scan crosses UPI).  Package
 *     map from sysfs physical_package_id of the pinned CPUs (BORROWED:
 *     L23_matrixsimd mt_r2); nsock=1 (wallaby) degenerates to flat.
 *   - WEIGHTED near/far volume split for streaming (wt=1 -> 4:3, wt=2 ->
 *     5:3), raced: the driver first-touches in/out on thread 0, so the
 *     far 16 threads pay UPI in BOTH directions per volume; giving the
 *     near socket more volumes equalizes finish times if UPI, not DRAM,
 *     is the binding constraint.  Two-socket only; ownership-only change.
 *   - pf=2 wired into the fused X items (prefetch the NEXT item's 23
 *     strided lines): my r2 next-item 3, raced at B<32.
 *
 * WHAT IS NEW IN mt_r4 (memory schedule + idle-thread hygiene only; DAG
 * untouched, one bit class as always).  mt_r3's node verdict: both L23
 * entries sit at 65-67 GB/s streaming, the LOWEST of any geometry, while
 * the L=36 entries reach 137-151 GB/s with sequential-read discipline;
 * the tree barrier (bar) and weighted split (wt) lost every node cell and
 * their tuner rows are dropped (code kept for env-forced A/Bs only).
 *   - STAGED SEQUENTIAL INPUT (si knob, 0/1/2), range exec only: with si
 *     engaged a thread copies its volume's `in` (24334 doubles, 190 KiB)
 *     sequentially into slot-local scratch (vs) and runs the X pass off
 *     the copy.  Same compulsory bytes; ONE hardware-prefetchable stream
 *     instead of 23 interleaved 8464-B-strided page streams (32 threads x
 *     23 streams = 736 open DRAM pages was the suspected 65 GB/s wall).
 *     si=2 stages on the FAR socket only (near reads local DRAM where the
 *     strided walk is cheap).  Carried as my unbuilt next-item since
 *     mt_r1; the shape is the L=36 winners' (L36_pencilfused's paced read
 *     cursor, L36_mixedradix's sntp), ported per the mt_r3 verdict's
 *     explicit L=23 order; L23_matrixsimd mt_r4 built the same knob.
 *   - PACED NEXT-VOLUME PREFETCH (pv knob), range exec only: during
 *     volume v's compute-heavy plane phase, prefetcht1 volume v+1's `in`
 *     sequentially (~133 lines per plane, 3043 total).  Same stream
 *     conversion as si with ZERO extra stores; the X pass then reads
 *     mostly L2 hits.  Differs from the node-rejected pf=2, which issued
 *     the same strided pattern just-in-time against the X pass's own
 *     demand loads.
 *   - PARKED NON-PARTICIPANTS: once the pick is final, pool threads with
 *     tid >= team (which join no barrier) poll the generation word with
 *     nanosleep(100us) instead of pause-spinning.  The node's scored B=1
 *     pick (fused T=16) leaves 16 workers spinning all run; mt_r3 verdict
 *     4.4 collects three entries blaming busy spinners for all-core clock
 *     drag (clk512 2.29 vs 2.89 GHz).  Free by construction: parked
 *     threads are never participants, and a late generation read only
 *     delays their empty workfn call.  BORROWED: L23_matrixsimd mt_r4
 *     (their park design), L36_pfa mt_r3 (nap-after-1ms).
 *
 * ASSUMPTIONS: L == 23 only; in/out distinct, 64-byte aligned (driver);
 * gcc/clang vector extensions with -ffp-contract=fast; self-#include
 * instantiates the template at 512-bit (WC=4) and 256-bit (WC=2).
 * =============================================================================
 */
#ifdef L23R_TEMPLATE
/* ===========================================================================
 *  TEMPLATE BODY -- instantiated once per vector width.
 *  Inputs: WC (complex per vector), SUF(x) (name mangler).
 * ===========================================================================
 */
#define VDW (2 * WC) /* doubles per vector */

typedef double SUF(vd) __attribute__((vector_size(8 * VDW), aligned(8)));
typedef long long SUF(vi) __attribute__((vector_size(8 * VDW)));
#define VT SUF(vd)
#define IT SUF(vi)

#define VLD(p) (*(const VT *)(p))
#define VST(p, v) (*(VT *)(p) = (v))

#if defined(__clang__)
#  define SHUF1(a, ...) __builtin_shufflevector(a, a, __VA_ARGS__)
#  define SHUF2(a, b, ...) __builtin_shufflevector(a, b, __VA_ARGS__)
#else
#  define SHUF1(a, ...) __builtin_shuffle(a, (IT){__VA_ARGS__})
#  define SHUF2(a, b, ...) __builtin_shuffle(a, b, (IT){__VA_ARGS__})
#endif

/* -i*t on interleaved complex: SWAPRI only -- the im-lane negation lives in
 * the sine tables' alternating {+s,-s} lane signs (phase-1 r10, from
 * L13_direct r9).  s*(-x) == (-s)*x bitwise, so no bits change. */
#if WC == 4
#  define SWAPRI(x) SHUF1(x, 1, 0, 3, 2, 5, 4, 7, 6)
#else
#  define SWAPRI(x) SHUF1(x, 1, 0, 3, 2)
#endif

/* full-width read of entry i from a pre-splatted table */
#define CGET(base, i) VLD((base) + (size_t)(i) * VDW)

#if WC == 4
#  define CDIST(p) ((p)->cd8)
#  define SDIST(p) ((p)->sd8)
#else
#  define CDIST(p) ((p)->cd4)
#  define SDIST(p) ((p)->sd4)
#endif

/* ---- WCxWC transpose of complex elements (128-bit blocks) --------------- */
#if WC == 4
#  define TTILE(x, y)                                                          \
      do {                                                                     \
          VT q0 = SHUF2((x)[0], (x)[1], 0, 1, 2, 3, 8, 9, 10, 11);             \
          VT q1 = SHUF2((x)[2], (x)[3], 0, 1, 2, 3, 8, 9, 10, 11);             \
          VT q2 = SHUF2((x)[0], (x)[1], 4, 5, 6, 7, 12, 13, 14, 15);           \
          VT q3 = SHUF2((x)[2], (x)[3], 4, 5, 6, 7, 12, 13, 14, 15);           \
          (y)[0] = SHUF2(q0, q1, 0, 1, 4, 5, 8, 9, 12, 13);                    \
          (y)[1] = SHUF2(q0, q1, 2, 3, 6, 7, 10, 11, 14, 15);                  \
          (y)[2] = SHUF2(q2, q3, 0, 1, 4, 5, 8, 9, 12, 13);                    \
          (y)[3] = SHUF2(q2, q3, 2, 3, 6, 7, 10, 11, 14, 15);                  \
      } while (0)
#  define TILE(m0, e0, e1, e2, e3)                                             \
      do {                                                                     \
          VT xx[4], yy[4];                                                     \
          xx[0] = (e0); xx[1] = (e1); xx[2] = (e2); xx[3] = (e3);              \
          TTILE(xx, yy);                                                       \
          VST(dst + 0 * da + (m0) * 2, yy[0]);                                 \
          VST(dst + 1 * da + (m0) * 2, yy[1]);                                 \
          VST(dst + 2 * da + (m0) * 2, yy[2]);                                 \
          VST(dst + 3 * da + (m0) * 2, yy[3]);                                 \
      } while (0)
#else
#  define TTILE(x, y)                                                          \
      do {                                                                     \
          (y)[0] = SHUF2((x)[0], (x)[1], 0, 1, 4, 5);                          \
          (y)[1] = SHUF2((x)[0], (x)[1], 2, 3, 6, 7);                          \
      } while (0)
#  define TILE(m0, e0, e1)                                                     \
      do {                                                                     \
          VT xx[2], yy[2];                                                     \
          xx[0] = (e0); xx[1] = (e1);                                          \
          TTILE(xx, yy);                                                       \
          VST(dst + 0 * da + (m0) * 2, yy[0]);                                 \
          VST(dst + 1 * da + (m0) * 2, yy[1]);                                 \
      } while (0)
#endif

/* Store tail: X_0 = P_0, X_k = P_k + R_k, X_{23-k} = P_k - R_k.
 * tr=0: plain vector stores, output m at dst + m*db (lanes contiguous).
 * tr=1: WCxWC tile-transposed stores, outputs contiguous, lane stride da. */
#if WC == 4
#define L23R_STORE_TAIL()                                                      \
    do {                                                                       \
        if (!tr) {                                                             \
            VST(dst + 0 * db, P0);                                             \
            VST(dst + 1 * db, P1 + R1);    VST(dst + 2 * db, P2 + R2);         \
            VST(dst + 3 * db, P3 + R3);    VST(dst + 4 * db, P4 + R4);         \
            VST(dst + 5 * db, P5 + R5);    VST(dst + 6 * db, P6 + R6);         \
            VST(dst + 7 * db, P7 + R7);    VST(dst + 8 * db, P8 + R8);         \
            VST(dst + 9 * db, P9 + R9);    VST(dst + 10 * db, P10 + R10);      \
            VST(dst + 11 * db, P11 + R11); VST(dst + 12 * db, P11 - R11);      \
            VST(dst + 13 * db, P10 - R10); VST(dst + 14 * db, P9 - R9);        \
            VST(dst + 15 * db, P8 - R8);   VST(dst + 16 * db, P7 - R7);        \
            VST(dst + 17 * db, P6 - R6);   VST(dst + 18 * db, P5 - R5);        \
            VST(dst + 19 * db, P4 - R4);   VST(dst + 20 * db, P3 - R3);        \
            VST(dst + 21 * db, P2 - R2);   VST(dst + 22 * db, P1 - R1);        \
        } else {                                                               \
            TILE(0, P0, P1 + R1, P2 + R2, P3 + R3);                            \
            TILE(4, P4 + R4, P5 + R5, P6 + R6, P7 + R7);                       \
            TILE(8, P8 + R8, P9 + R9, P10 + R10, P11 + R11);                   \
            TILE(12, P11 - R11, P10 - R10, P9 - R9, P8 - R8);                  \
            TILE(16, P7 - R7, P6 - R6, P5 - R5, P4 - R4);                      \
            TILE(19, P4 - R4, P3 - R3, P2 - R2, P1 - R1);                      \
        }                                                                      \
    } while (0)
#else
#define L23R_STORE_TAIL()                                                      \
    do {                                                                       \
        if (!tr) {                                                             \
            VST(dst + 0 * db, P0);                                             \
            VST(dst + 1 * db, P1 + R1);    VST(dst + 2 * db, P2 + R2);         \
            VST(dst + 3 * db, P3 + R3);    VST(dst + 4 * db, P4 + R4);         \
            VST(dst + 5 * db, P5 + R5);    VST(dst + 6 * db, P6 + R6);         \
            VST(dst + 7 * db, P7 + R7);    VST(dst + 8 * db, P8 + R8);         \
            VST(dst + 9 * db, P9 + R9);    VST(dst + 10 * db, P10 + R10);      \
            VST(dst + 11 * db, P11 + R11); VST(dst + 12 * db, P11 - R11);      \
            VST(dst + 13 * db, P10 - R10); VST(dst + 14 * db, P9 - R9);        \
            VST(dst + 15 * db, P8 - R8);   VST(dst + 16 * db, P7 - R7);        \
            VST(dst + 17 * db, P6 - R6);   VST(dst + 18 * db, P5 - R5);        \
            VST(dst + 19 * db, P4 - R4);   VST(dst + 20 * db, P3 - R3);        \
            VST(dst + 21 * db, P2 - R2);   VST(dst + 22 * db, P1 - R1);        \
        } else {                                                               \
            TILE(0, P0, P1 + R1);                                              \
            TILE(2, P2 + R2, P3 + R3);                                         \
            TILE(4, P4 + R4, P5 + R5);                                         \
            TILE(6, P6 + R6, P7 + R7);                                         \
            TILE(8, P8 + R8, P9 + R9);                                         \
            TILE(10, P10 + R10, P11 + R11);                                    \
            TILE(12, P11 - R11, P10 - R10);                                    \
            TILE(14, P9 - R9, P8 - R8);                                        \
            TILE(16, P7 - R7, P6 - R6);                                        \
            TILE(18, P5 - R5, P4 - R4);                                        \
            TILE(20, P3 - R3, P2 - R2);                                        \
            TILE(21, P2 - R2, P1 - R1);                                        \
        }                                                                      \
    } while (0)
#endif

/* -----------------------------------------------------------------------
 *  krn_ts: two-sweep pinned-constant kernel.  cd holds the 11 distinct
 *  cosine magnitudes splatted; sd holds the 11 sine magnitudes with
 *  alternating {+s, -s} lane signs (the -i rotation, folded).  Every
 *  coefficient is a register constant.  All call sites pass compile-time-
 *  constant strides (keeps gcc from the runtime-offset lea-spill
 *  pathology L45_pfa r8 documented).
 * --------------------------------------------------------------------- */
static inline __attribute__((always_inline)) void
SUF(krn_ts)(const double *restrict src, long rs, double *restrict dst, long da,
            long db, const double *restrict cd, const double *restrict sd,
            int tr)
{
    VT P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11;
    VT R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11;

    { /* ---- cosine sweep: 12 accumulators, 11 pinned constants ---- */
        VT C1 = CGET(cd, 0), C2 = CGET(cd, 1), C3 = CGET(cd, 2),
           C4 = CGET(cd, 3), C5 = CGET(cd, 4), C6 = CGET(cd, 5),
           C7 = CGET(cd, 6), C8 = CGET(cd, 7), C9 = CGET(cd, 8),
           C10 = CGET(cd, 9), C11 = CGET(cd, 10);
        VT v0 = VLD(src);
        P0 = v0; P1 = v0; P2 = v0; P3 = v0; P4 = v0; P5 = v0;
        P6 = v0; P7 = v0; P8 = v0; P9 = v0; P10 = v0; P11 = v0;
        { VT u = VLD(src + 1 * rs) + VLD(src + 22 * rs);
          P0 += u; P1 += C1 * u; P2 += C2 * u; P3 += C3 * u;
          P4 += C4 * u; P5 += C5 * u; P6 += C6 * u; P7 += C7 * u;
          P8 += C8 * u; P9 += C9 * u; P10 += C10 * u; P11 += C11 * u; }
        { VT u = VLD(src + 2 * rs) + VLD(src + 21 * rs);
          P0 += u; P1 += C2 * u; P2 += C4 * u; P3 += C6 * u;
          P4 += C8 * u; P5 += C10 * u; P6 += C11 * u; P7 += C9 * u;
          P8 += C7 * u; P9 += C5 * u; P10 += C3 * u; P11 += C1 * u; }
        { VT u = VLD(src + 3 * rs) + VLD(src + 20 * rs);
          P0 += u; P1 += C3 * u; P2 += C6 * u; P3 += C9 * u;
          P4 += C11 * u; P5 += C8 * u; P6 += C5 * u; P7 += C2 * u;
          P8 += C1 * u; P9 += C4 * u; P10 += C7 * u; P11 += C10 * u; }
        { VT u = VLD(src + 4 * rs) + VLD(src + 19 * rs);
          P0 += u; P1 += C4 * u; P2 += C8 * u; P3 += C11 * u;
          P4 += C7 * u; P5 += C3 * u; P6 += C1 * u; P7 += C5 * u;
          P8 += C9 * u; P9 += C10 * u; P10 += C6 * u; P11 += C2 * u; }
        { VT u = VLD(src + 5 * rs) + VLD(src + 18 * rs);
          P0 += u; P1 += C5 * u; P2 += C10 * u; P3 += C8 * u;
          P4 += C3 * u; P5 += C2 * u; P6 += C7 * u; P7 += C11 * u;
          P8 += C6 * u; P9 += C1 * u; P10 += C4 * u; P11 += C9 * u; }
        { VT u = VLD(src + 6 * rs) + VLD(src + 17 * rs);
          P0 += u; P1 += C6 * u; P2 += C11 * u; P3 += C5 * u;
          P4 += C1 * u; P5 += C7 * u; P6 += C10 * u; P7 += C4 * u;
          P8 += C2 * u; P9 += C8 * u; P10 += C9 * u; P11 += C3 * u; }
        { VT u = VLD(src + 7 * rs) + VLD(src + 16 * rs);
          P0 += u; P1 += C7 * u; P2 += C9 * u; P3 += C2 * u;
          P4 += C5 * u; P5 += C11 * u; P6 += C4 * u; P7 += C3 * u;
          P8 += C10 * u; P9 += C6 * u; P10 += C1 * u; P11 += C8 * u; }
        { VT u = VLD(src + 8 * rs) + VLD(src + 15 * rs);
          P0 += u; P1 += C8 * u; P2 += C7 * u; P3 += C1 * u;
          P4 += C9 * u; P5 += C6 * u; P6 += C2 * u; P7 += C10 * u;
          P8 += C5 * u; P9 += C3 * u; P10 += C11 * u; P11 += C4 * u; }
        { VT u = VLD(src + 9 * rs) + VLD(src + 14 * rs);
          P0 += u; P1 += C9 * u; P2 += C5 * u; P3 += C4 * u;
          P4 += C10 * u; P5 += C1 * u; P6 += C8 * u; P7 += C6 * u;
          P8 += C3 * u; P9 += C11 * u; P10 += C2 * u; P11 += C7 * u; }
        { VT u = VLD(src + 10 * rs) + VLD(src + 13 * rs);
          P0 += u; P1 += C10 * u; P2 += C3 * u; P3 += C7 * u;
          P4 += C6 * u; P5 += C4 * u; P6 += C9 * u; P7 += C1 * u;
          P8 += C11 * u; P9 += C2 * u; P10 += C8 * u; P11 += C5 * u; }
        { VT u = VLD(src + 11 * rs) + VLD(src + 12 * rs);
          P0 += u; P1 += C11 * u; P2 += C1 * u; P3 += C10 * u;
          P4 += C2 * u; P5 += C9 * u; P6 += C3 * u; P7 += C8 * u;
          P8 += C4 * u; P9 += C7 * u; P10 += C5 * u; P11 += C6 * u; }
    }
    { /* ---- sine sweep: 11 accumulators, 11 pinned constants ---- */
        VT S1 = CGET(sd, 0), S2 = CGET(sd, 1), S3 = CGET(sd, 2),
           S4 = CGET(sd, 3), S5 = CGET(sd, 4), S6 = CGET(sd, 5),
           S7 = CGET(sd, 6), S8 = CGET(sd, 7), S9 = CGET(sd, 8),
           S10 = CGET(sd, 9), S11 = CGET(sd, 10);
        { VT w = SWAPRI(VLD(src + 1 * rs) - VLD(src + 22 * rs));
          R1 = S1 * w; R2 = S2 * w; R3 = S3 * w; R4 = S4 * w;
          R5 = S5 * w; R6 = S6 * w; R7 = S7 * w; R8 = S8 * w;
          R9 = S9 * w; R10 = S10 * w; R11 = S11 * w; }
        { VT w = SWAPRI(VLD(src + 2 * rs) - VLD(src + 21 * rs));
          R1 += S2 * w; R2 += S4 * w; R3 += S6 * w; R4 += S8 * w;
          R5 += S10 * w; R6 -= S11 * w; R7 -= S9 * w; R8 -= S7 * w;
          R9 -= S5 * w; R10 -= S3 * w; R11 -= S1 * w; }
        { VT w = SWAPRI(VLD(src + 3 * rs) - VLD(src + 20 * rs));
          R1 += S3 * w; R2 += S6 * w; R3 += S9 * w; R4 -= S11 * w;
          R5 -= S8 * w; R6 -= S5 * w; R7 -= S2 * w; R8 += S1 * w;
          R9 += S4 * w; R10 += S7 * w; R11 += S10 * w; }
        { VT w = SWAPRI(VLD(src + 4 * rs) - VLD(src + 19 * rs));
          R1 += S4 * w; R2 += S8 * w; R3 -= S11 * w; R4 -= S7 * w;
          R5 -= S3 * w; R6 += S1 * w; R7 += S5 * w; R8 += S9 * w;
          R9 -= S10 * w; R10 -= S6 * w; R11 -= S2 * w; }
        { VT w = SWAPRI(VLD(src + 5 * rs) - VLD(src + 18 * rs));
          R1 += S5 * w; R2 += S10 * w; R3 -= S8 * w; R4 -= S3 * w;
          R5 += S2 * w; R6 += S7 * w; R7 -= S11 * w; R8 -= S6 * w;
          R9 -= S1 * w; R10 += S4 * w; R11 += S9 * w; }
        { VT w = SWAPRI(VLD(src + 6 * rs) - VLD(src + 17 * rs));
          R1 += S6 * w; R2 -= S11 * w; R3 -= S5 * w; R4 += S1 * w;
          R5 += S7 * w; R6 -= S10 * w; R7 -= S4 * w; R8 += S2 * w;
          R9 += S8 * w; R10 -= S9 * w; R11 -= S3 * w; }
        { VT w = SWAPRI(VLD(src + 7 * rs) - VLD(src + 16 * rs));
          R1 += S7 * w; R2 -= S9 * w; R3 -= S2 * w; R4 += S5 * w;
          R5 -= S11 * w; R6 -= S4 * w; R7 += S3 * w; R8 += S10 * w;
          R9 -= S6 * w; R10 += S1 * w; R11 += S8 * w; }
        { VT w = SWAPRI(VLD(src + 8 * rs) - VLD(src + 15 * rs));
          R1 += S8 * w; R2 -= S7 * w; R3 += S1 * w; R4 += S9 * w;
          R5 -= S6 * w; R6 += S2 * w; R7 += S10 * w; R8 -= S5 * w;
          R9 += S3 * w; R10 += S11 * w; R11 -= S4 * w; }
        { VT w = SWAPRI(VLD(src + 9 * rs) - VLD(src + 14 * rs));
          R1 += S9 * w; R2 -= S5 * w; R3 += S4 * w; R4 -= S10 * w;
          R5 -= S1 * w; R6 += S8 * w; R7 -= S6 * w; R8 += S3 * w;
          R9 -= S11 * w; R10 -= S2 * w; R11 += S7 * w; }
        { VT w = SWAPRI(VLD(src + 10 * rs) - VLD(src + 13 * rs));
          R1 += S10 * w; R2 -= S3 * w; R3 += S7 * w; R4 -= S6 * w;
          R5 += S4 * w; R6 -= S9 * w; R7 += S1 * w; R8 += S11 * w;
          R9 -= S2 * w; R10 += S8 * w; R11 -= S5 * w; }
        { VT w = SWAPRI(VLD(src + 11 * rs) - VLD(src + 12 * rs));
          R1 += S11 * w; R2 -= S1 * w; R3 += S10 * w; R4 -= S2 * w;
          R5 += S9 * w; R6 -= S3 * w; R7 += S8 * w; R8 -= S4 * w;
          R9 += S7 * w; R10 -= S5 * w; R11 += S6 * w; }
    }
    L23R_STORE_TAIL();
}

/* chunk start offsets covering a 23-long index; the last chunk overlaps the
 * previous and rewrites bit-identical values (cheaper than masking --
 * phase-1 measured lesson). */
#if WC == 4
static const int SUF(off23)[6] = {0, 4, 8, 12, 16, 19};
#  define NOFF23 6
#else
static const int SUF(off23)[12] = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 21};
#  define NOFF23 12
#endif

#define PBROW (24 * 2) /* plane buffer row stride, doubles (23 padded to 24) */

/* X-pass chunk slots: slot i covers lanes i*WC..; the last is the
 * overlapping tail (529 % WC != 0 for both widths).  NXI23 = independent
 * pool items: the tail chunk is folded into item NXI23-1 so no t1 line has
 * two writers across threads. */
#define NX23 (529 / WC + 1)
#define NXI23 (NX23 - 1)
#define L23R_XF0(i) ((i) < 529 / WC ? (long)(i) * WC : (long)(529 - WC))

/* one X chunk: 23-point DFT along x, lanes = WC adjacent (y,z) pairs,
 * in (plane stride 1058) -> t1 (plane stride 1064 = 133 whole lines, so
 * every store is 64-byte aligned) */
static inline __attribute__((always_inline)) void
SUF(xchunk)(const double *restrict vin, double *restrict t1, long f0,
            const double *restrict cd, const double *restrict sd)
{
    SUF(krn_ts)(vin + 2 * f0, 1058, t1 + 2 * f0, 2, L23R_T1P, cd, sd, 0);
}

/* one (v,x) plane: Y (t1 rows, stride 46 -> pb, transposed) then Z (pb ->
 * out plane, transposed).  pw!=0 write-intent-prefetches the 133 out lines,
 * half before Y, half between Y and Z (phase-1 r7 pacing). */
static inline __attribute__((always_inline)) void
SUF(plane)(const double *restrict t1p, double *restrict pt,
           double *restrict pb, const double *restrict cd,
           const double *restrict sd, int pw)
{
    if (pw) L23R_PWB(pt, 0, 67);
    for (int t = 0; t < NOFF23; ++t) {
        long f0 = SUF(off23)[t];
        SUF(krn_ts)(t1p + 2 * f0, 46, pb + f0 * PBROW, PBROW, 2, cd, sd, 1);
    }
    if (pw) L23R_PWB(pt, 67, 133);
    for (int t = 0; t < NOFF23; ++t) {
        long f0 = SUF(off23)[t];
        SUF(krn_ts)(pb + 2 * f0, PBROW, pt + f0 * 46, 46, 2, cd, sd, 1);
    }
}

/* -----------------------------------------------------------------------
 *  Range exec: volumes [b0,b1) on ONE thread's scratch slot.  Plain
 *  X-first two-sweep, pf/pw as in phase 1; ntc!=0 stages each finished
 *  plane and streams it to out (NT stores skip the read-for-ownership).
 *  sieff!=0 (mt_r4) copies each volume's `in` sequentially into the
 *  slot's vs region first and runs the X pass off the L2-resident copy;
 *  p->pv!=0 (mt_r4) sequentially prefetcht1's the NEXT owned volume's
 *  `in` paced through the plane phase (~133 lines per plane).  Both are
 *  bit-exact: the X pass reads identical values in identical order.
 * --------------------------------------------------------------------- */
static void SUF(rng)(const fft3d_plan *restrict p,
                     const double _Complex *restrict in,
                     double _Complex *restrict out, int b0, int b1,
                     double *restrict slot, int ntc, int sieff)
{
    const double *restrict cd = CDIST(p);
    const double *restrict sd = SDIST(p);
    double *restrict t1 = slot + L23R_SL_T1;
    double *restrict pb = slot + L23R_SL_PB;
    double *restrict ps = slot + L23R_SL_PS;
    double *restrict vs = slot + L23R_SL_VS;
    const int pf = p->pf, pw = p->pw, pv = p->pv;
    for (int b = b0; b < b1; ++b) {
        const double *vin = (const double *)(in + (size_t)b * 12167);
        double *vout = (double *)(out + (size_t)b * 12167);
        if (sieff) {
            memcpy(vs, vin, 24334 * sizeof(double));
            vin = vs;
        }
        for (int i = 0; i < NX23; ++i) {
            if (pf == 2 && i + 4 < NX23) {
                /* pull the chunk 4 slots ahead: 23 lines, one per 8464-B-
                 * strided x-plane, where hardware prefetch restarts at
                 * every 4K boundary */
                const char *px = (const char *)(vin + 2 * L23R_XF0(i + 4));
                for (int q4 = 0; q4 < 23; ++q4)
                    __builtin_prefetch(px + (long)q4 * 8464, 0, 3);
            }
            SUF(xchunk)(vin, t1, L23R_XF0(i), cd, sd);
        }
        /* 190 KiB = 3043 lines incl. the 48-B volume-start misalignment;
         * 23 planes x 133 lines covers it with a harmless <64-B overrun */
        const char *pvn = (pv && b + 1 < b1)
                              ? (const char *)(in + (size_t)(b + 1) * 12167)
                              : (const char *)0;
        for (int x = 0; x < 23; ++x) {
            if (pvn) {
                long q0 = (long)x * 133, q1 = q0 + 133;
                if (q1 > 3043) q1 = 3043;
                for (long q6 = q0; q6 < q1; ++q6)
                    __builtin_prefetch(pvn + q6 * 64, 0, 2);
            }
            double *pt = vout + (long)x * 1058;
            if (!ntc) {
                SUF(plane)(t1 + (long)x * L23R_T1P, pt, pb, cd, sd, pw);
            } else {
                SUF(plane)(t1 + (long)x * L23R_T1P, ps, pb, cd, sd, 0);
                l23r_ntcopy(pt, ps, 1058);
            }
        }
    }
}

/* ---- pool jobs (tid >= tw returns immediately and joins no barrier, so
 * every barrier has exactly tw participants) ------------------------------ */

/* batch / batchNT job: thread t owns the contiguous volume block
 * [nb*t/T, nb*(t+1)/T) on its own NUMA-local slot; one join flag. */
static void SUF(work_rng)(const fft3d_plan *restrict p, int tid)
{
    l23r_pool *restrict pl = p->pl;
    const int T = pl->tw;
    if (tid >= T) return;
    const int nb = p->batch;
    int b0, b1;
    if (p->wt && pl->nsock == 2) {
        /* weighted near/far split: the driver first-touches in/out on
         * thread 0, so far-socket threads pay UPI both ways per volume;
         * near threads take proportionally more volumes.  Ownership-only
         * change: the per-volume DAG is untouched. */
        const int wn = p->wt == 1 ? 4 : 5, wf = 3;
        long tot = 0, pre = 0;
        for (int t = 0; t < T; ++t) {
            const int w = pl->far[t] ? wf : wn;
            if (t < tid) pre += w;
            tot += w;
        }
        const long wme = pl->far[tid] ? wf : wn;
        b0 = (int)((long)nb * pre / tot);
        b1 = (int)((long)nb * (pre + wme) / tot);
    } else {
        b0 = (int)((long)nb * tid / T);
        b1 = (int)((long)nb * (tid + 1) / T);
    }
    if (b1 > b0) {
        /* si=1: every thread stages; si=2: far-socket threads only (near
         * threads read local DRAM where the strided walk is cheap) */
        int sieff = p->si == 1 || (p->si == 2 && pl->far[tid]);
        SUF(rng)(p, pl->in, pl->out, b0, b1, L23R_SLOT(p, tid), p->ntc, sieff);
    }
    l23r_bar_join(pl, tid, T);
}

/* fused job (batch < 32): all volumes' X items into the fused t1 arena
 * (item NXI23-1 also runs the overlapping tail chunk, same thread), ONE mid
 * barrier (every t1 plane needs every X chunk of its volume), all volumes'
 * (v,x) planes on the executing thread's own pb, join. */
static void SUF(work_fused)(const fft3d_plan *restrict p, int tid)
{
    l23r_pool *restrict pl = p->pl;
    const int T = pl->tw;
    if (tid >= T) return;
    const double *restrict cd = CDIST(p);
    const double *restrict sd = SDIST(p);
    const int nb = p->batch;
    const double _Complex *in = pl->in;
    double _Complex *out = pl->out;
    double *restrict t1f = p->t1f;
    const int pf = p->pf;
    {
        const long nxi = (long)nb * NXI23;
        long g0 = nxi * tid / T, g1 = nxi * (tid + 1) / T;
        for (long g = g0; g < g1; ++g) {
            long b = g / NXI23;
            int i = (int)(g - b * NXI23);
            const double *vin = (const double *)(in + (size_t)b * 12167);
            if (pf == 2 && g + 1 < g1) {
                /* pull the NEXT item's 23 lines: one per 8464-B-strided
                 * x-plane, where hardware prefetch restarts every 4K */
                long b2 = (g + 1) / NXI23;
                int i2 = (int)(g + 1 - b2 * NXI23);
                const char *px =
                    (const char *)((const double *)(in + (size_t)b2 * 12167) +
                                   2 * L23R_XF0(i2));
                for (int q4 = 0; q4 < 23; ++q4)
                    __builtin_prefetch(px + (long)q4 * 8464, 0, 3);
            }
            double *t1v = t1f + (size_t)b * L23R_T1V;
            SUF(xchunk)(vin, t1v, L23R_XF0(i), cd, sd);
            if (i == NXI23 - 1)
                SUF(xchunk)(vin, t1v, L23R_XF0(NX23 - 1), cd, sd);
        }
    }
    l23r_bar_mid(pl, tid, T);
    {
        double *restrict pb = L23R_SLOT(p, tid) + L23R_SL_PB;
        const long npl = (long)nb * 23;
        long g0 = npl * tid / T, g1 = npl * (tid + 1) / T;
        for (long g = g0; g < g1; ++g) {
            long b = g / 23;
            int x = (int)(g - b * 23);
            SUF(plane)(t1f + (size_t)b * L23R_T1V + (size_t)x * L23R_T1P,
                       (double *)(out + (size_t)b * 12167) + (size_t)x * 1058,
                       pb, cd, sd, 0);
        }
    }
    l23r_bar_join(pl, tid, T);
}

/* ---- top-level exec variants (dispatch targets) ------------------------ */

static __attribute__((unused)) void
SUF(x_serial)(const fft3d_plan *restrict p, const double _Complex *restrict in,
              double _Complex *restrict out)
{
    SUF(rng)(p, in, out, 0, p->batch, L23R_SLOT(p, 0), p->ntc, p->si == 1);
}

static __attribute__((unused)) void
SUF(x_pool_rng)(const fft3d_plan *restrict p,
                const double _Complex *restrict in,
                double _Complex *restrict out)
{
    l23r_run_job(p, SUF(work_rng), in, out);
}

static __attribute__((unused)) void
SUF(x_pool_fused)(const fft3d_plan *restrict p,
                  const double _Complex *restrict in,
                  double _Complex *restrict out)
{
    l23r_run_job(p, SUF(work_fused), in, out);
}

#undef VDW
#undef VT
#undef IT
#undef VLD
#undef VST
#undef SHUF1
#undef SHUF2
#undef SWAPRI
#undef CGET
#undef CDIST
#undef SDIST
#undef L23R_STORE_TAIL
#undef TTILE
#undef TILE
#undef NOFF23
#undef PBROW
#undef NX23
#undef NXI23
#undef L23R_XF0

#else /* ================= main body ================= */

#define _GNU_SOURCE /* sched_getcpu, CPU_SET, pthread_setaffinity_np */

#include <complex.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "../fft3d_api.h"

#ifdef _OPENMP
#  include <omp.h>
#else
static inline int omp_get_max_threads(void) { return 1; }
static inline int omp_get_num_threads(void) { return 1; }
static inline int omp_get_thread_num(void) { return 0; }
#endif

#if defined(__x86_64__) || defined(__i386__)
#  define l23r_cpu_relax() __builtin_ia32_pause()
#else
#  define l23r_cpu_relax() ((void)0)
#endif

/* Streaming copy of one finished 23x23 plane (1058 doubles) into `out`:
 * NT stores skip the read-for-ownership of every output line.  Wallaby
 * streaming winner in mt_r1 (1.74 vs 2.00 us/t); the node did NOT pick it
 * (its pick: plain batch pf0 pw0).  Kept raced.  Plane starts are 16
 * (mod 64), hence the step-up loop. */
#if defined(__AVX512F__) || defined(__AVX__) || defined(__SSE2__)
#  include <immintrin.h>
static void l23r_ntcopy(double *restrict dst, const double *restrict src, size_t nd)
{
    size_t i = 0;
#if defined(__AVX512F__)
    while (i + 2 <= nd && (((uintptr_t)(dst + i)) & 63u)) {
        _mm_stream_pd(dst + i, _mm_loadu_pd(src + i));
        i += 2;
    }
    for (; i + 8 <= nd; i += 8) _mm512_stream_pd(dst + i, _mm512_loadu_pd(src + i));
#elif defined(__AVX__)
    while (i + 2 <= nd && (((uintptr_t)(dst + i)) & 31u)) {
        _mm_stream_pd(dst + i, _mm_loadu_pd(src + i));
        i += 2;
    }
    for (; i + 4 <= nd; i += 4) _mm256_stream_pd(dst + i, _mm256_loadu_pd(src + i));
#endif
    for (; i + 2 <= nd; i += 2) _mm_stream_pd(dst + i, _mm_loadu_pd(src + i));
    for (; i < nd; ++i) dst[i] = src[i];
    _mm_sfence();
}
#else
static void l23r_ntcopy(double *restrict dst, const double *restrict src, size_t nd)
{
    memcpy(dst, src, nd * sizeof *dst);
}
#endif

/* paced write-intent prefetch of [l0,l1) of the 133 out-plane lines */
#define L23R_PWB(base, l0, l1)                                                 \
    do {                                                                       \
        const char *pwp = (const char *)(base);                                \
        for (int q5 = (l0); q5 < (l1); ++q5)                                   \
            __builtin_prefetch(pwp + (long)q5 * 64, 1, 3);                     \
    } while (0)

#define L23R_MAXT 32

/* -----------------------------------------------------------------------
 *  Persistent spin pool (adopted from L23_matrixsimd mt_r1: one GOMP
 *  fork+barrier+join measured 6.2-8.2 us at T=8..32 on wallaby, several
 *  times the barrier'd work at B=1; a central fetch-add barrier measured
 *  ~2.5-5 us).  Per-thread padded arrival flags: an arrival is one
 *  uncontended store; main (tid 0) is always the collector -- it scans the
 *  flags (loads overlap in the fill buffers) and broadcasts one release
 *  word.  Epochs derive from the job generation (2*gen mid, 2*gen+1 join),
 *  so a varying participant count tw stays correct: flags and release only
 *  ever increase.  Workers never sleep: the driver's timing loop is
 *  back-to-back executes and the cores are ours.
 * --------------------------------------------------------------------- */
typedef struct {
    _Atomic unsigned long v;
    char pad[56];
} l23r_flag;

struct fft3d_plan;
typedef void (*l23r_workfn)(const struct fft3d_plan *, int);

typedef struct {
    struct fft3d_plan *p;
    int tid;
    int cpu;
} l23r_warg;

typedef struct l23r_pool {
    _Atomic long gen;
    char pad0[56];
    _Atomic unsigned long rel;
    char pad1[56];
    _Atomic unsigned long rel2; /* far-socket release relay (tree barrier) */
    char pad1b[56];
    l23r_flag arr[L23R_MAXT];
    _Atomic int ready;
    _Atomic int fail;
    _Atomic int shutdown;
    char pad2[52];
    l23r_workfn workfn;
    const double _Complex *in;
    double _Complex *out;
    long gen_local;
    _Atomic int parkfrom; /* tid >= parkfrom naps on the gen word (mt_r4);
                           * L23R_MAXT until the pick is final */
    int nthr; /* pool size incl. main */
    int tw;   /* active participants of the current job (<= nthr) */
    int fc;   /* far-collector tid of the current job; 0 = flat barrier */
    int nsock;                    /* packages under the pinned CPUs (1 or 2) */
    unsigned char far[L23R_MAXT]; /* tid pinned on a package != tid 0's */
    pthread_t th[L23R_MAXT];
    l23r_warg wa[L23R_MAXT];
} l23r_pool;

struct fft3d_plan {
    int L, batch;
    int team;   /* participants the next job runs with (<= pool size) */
    int pf;     /* in-pass X input prefetch: 0 off, 2 = 4 chunks ahead */
    int pw;     /* write-intent prefetch on the Z-group out stores */
    int ntc;    /* range exec stages planes + NT-streams them to out */
    int bar;    /* 1 = two-level socket-tree barrier (needs nsock == 2) */
    int wt;     /* weighted near/far volume split: 0 even, 1 = 4:3, 2 = 5:3 */
    int si;     /* staged sequential input: 0 off, 1 all threads, 2 far only */
    int pv;     /* paced sequential prefetch of the next owned volume's in */
    void (*exec)(const struct fft3d_plan *, const double _Complex *, double _Complex *);
    double *cd8, *sd8; /* 11 distinct cos / sin magnitudes, splatted 8x */
    double *cd4, *sd4; /* the same, splatted 4x */
    double *slab;      /* nslot per-thread scratch slots, page-aligned */
    int nslot;
    double *t1f;       /* fused t1 arena (batch < 32), reader-first-touched */
    void *t1f_blk;
    l23r_pool *pl;
    void *block;
    double _Complex *ti, *to; /* transient buffers for the plan-time tuner */
    size_t tn;
};

/* t1 plane stride, doubles: 23x23 complex = 1058, padded to 1064 so every
 * plane base (and so every X-pass store) is 64-byte aligned (8512 = 133
 * whole lines).  L23R_T1V = one padded volume in the fused arena. */
#define L23R_T1P 1064
#define L23R_T1V (23 * L23R_T1P)

/* per-thread scratch slot layout, in doubles.  Slots are PAGE-sized so a
 * slot's pages are NUMA-local to its first-touching thread and no two
 * threads' scratch shares a line. */
#define L23R_SL_T1 0                       /* 23 * 1064 = 24472 */
#define L23R_SL_PB 24472                   /* 23 * 48   =  1104 */
#define L23R_SL_PS 25576                   /* staging plane 1064 */
#define L23R_SL_VS 27136                   /* staged-input volume, 24334
                                            * doubles, page-aligned (mt_r4) */
#define L23R_SLOTSZ 51712                  /* 101 pages = 413696 B */

#define L23R_SLOT(p, t) ((p)->slab + (size_t)(t) * L23R_SLOTSZ)

/* MID barrier (between the fused X pass and plane phase): workers post
 * their arrival flag and spin on the release word; main collects and
 * releases.  Transitive release/acquire through main makes every worker's
 * X stores visible to every plane reader.
 *
 * TREE variant (pl->fc > 0, mt_r3): the far socket's collector fc scans
 * its own socket's flags locally, posts ONE flag tid 0 reads across UPI,
 * then relays the release through rel2 so the far side never spins on
 * tid 0's release line.  One remote flag read + one remote release read
 * per episode instead of 16 of each.  Ordering is transitive through fc
 * (far arrivals -> fc's flag -> tid 0; tid 0's rel -> fc -> rel2). */
static inline void l23r_bar_mid(l23r_pool *pl, int tid, int T)
{
    unsigned long e =
        2ul * (unsigned long)atomic_load_explicit(&pl->gen,
                                                  memory_order_relaxed);
    const int fc = pl->fc;
    if (fc > 0) {
        if (tid == 0) {
            for (int t = 1; t < T; ++t)
                if (!pl->far[t])
                    while (atomic_load_explicit(&pl->arr[t].v,
                                                memory_order_acquire) < e)
                        l23r_cpu_relax();
            while (atomic_load_explicit(&pl->arr[fc].v,
                                        memory_order_acquire) < e)
                l23r_cpu_relax();
            atomic_store_explicit(&pl->rel, e, memory_order_release);
        } else if (tid == fc) {
            for (int t = 1; t < T; ++t)
                if (pl->far[t] && t != fc)
                    while (atomic_load_explicit(&pl->arr[t].v,
                                                memory_order_acquire) < e)
                        l23r_cpu_relax();
            atomic_store_explicit(&pl->arr[fc].v, e, memory_order_release);
            while (atomic_load_explicit(&pl->rel, memory_order_acquire) < e)
                l23r_cpu_relax();
            atomic_store_explicit(&pl->rel2, e, memory_order_release);
        } else if (pl->far[tid]) {
            atomic_store_explicit(&pl->arr[tid].v, e, memory_order_release);
            while (atomic_load_explicit(&pl->rel2, memory_order_acquire) < e)
                l23r_cpu_relax();
        } else {
            atomic_store_explicit(&pl->arr[tid].v, e, memory_order_release);
            while (atomic_load_explicit(&pl->rel, memory_order_acquire) < e)
                l23r_cpu_relax();
        }
        return;
    }
    if (tid == 0) {
        for (int t = 1; t < T; ++t)
            while (atomic_load_explicit(&pl->arr[t].v, memory_order_acquire) < e)
                l23r_cpu_relax();
        atomic_store_explicit(&pl->rel, e, memory_order_release);
    } else {
        atomic_store_explicit(&pl->arr[tid].v, e, memory_order_release);
        while (atomic_load_explicit(&pl->rel, memory_order_acquire) < e)
            l23r_cpu_relax();
    }
}

/* JOIN (end of every job): workers post their flag and go straight back to
 * the generation spin -- they need not wait for each other, because main
 * cannot dispatch the next job (or return to the driver) until it has seen
 * every flag.  Tree variant: fc collects its socket's flags first, so
 * tid 0 reads one remote flag instead of 16. */
static inline void l23r_bar_join(l23r_pool *pl, int tid, int T)
{
    unsigned long e =
        2ul * (unsigned long)atomic_load_explicit(&pl->gen,
                                                  memory_order_relaxed) + 1;
    const int fc = pl->fc;
    if (fc > 0) {
        if (tid == 0) {
            for (int t = 1; t < T; ++t)
                if (!pl->far[t])
                    while (atomic_load_explicit(&pl->arr[t].v,
                                                memory_order_acquire) < e)
                        l23r_cpu_relax();
            while (atomic_load_explicit(&pl->arr[fc].v,
                                        memory_order_acquire) < e)
                l23r_cpu_relax();
        } else if (tid == fc) {
            for (int t = 1; t < T; ++t)
                if (pl->far[t] && t != fc)
                    while (atomic_load_explicit(&pl->arr[t].v,
                                                memory_order_acquire) < e)
                        l23r_cpu_relax();
            atomic_store_explicit(&pl->arr[fc].v, e, memory_order_release);
        } else {
            atomic_store_explicit(&pl->arr[tid].v, e, memory_order_release);
        }
        return;
    }
    if (tid == 0) {
        for (int t = 1; t < T; ++t)
            while (atomic_load_explicit(&pl->arr[t].v, memory_order_acquire) < e)
                l23r_cpu_relax();
    } else {
        atomic_store_explicit(&pl->arr[tid].v, e, memory_order_release);
    }
}

/* Dispatch one job to the pool and run it as tid 0; returns when the whole
 * job is done (every job ends in its own barrier). */
static void l23r_run_job(const struct fft3d_plan *p, l23r_workfn fn,
                         const double _Complex *in, double _Complex *out)
{
    l23r_pool *pl = p->pl;
    pl->tw = p->team;
    /* tree barrier only when asked, two packages present, and the team
     * actually reaches the far socket; tid 0 is near by construction */
    int fc = 0;
    if (p->bar && pl->nsock == 2)
        for (int t = 1; t < p->team; ++t)
            if (pl->far[t]) { fc = t; break; }
    pl->fc = fc;
    pl->in = in;
    pl->out = out;
    pl->workfn = fn;
    ++pl->gen_local;
    atomic_store_explicit(&pl->gen, pl->gen_local, memory_order_release);
    fn(p, 0);
}

/* ---- instantiate the kernel template: 512-bit and 256-bit ---- */
#if defined(__has_include)
#  if __has_include("L23_rader.c")
#    define L23R_SELF "L23_rader.c"
#  elif __has_include("impl/L23_rader.c")
#    define L23R_SELF "impl/L23_rader.c"
#  endif
#else
#  define L23R_SELF "L23_rader.c"
#endif

#ifdef L23R_SELF
#  define L23R_TEMPLATE 1
#  define WC 4
#  define SUF(x) x##_w4
#  include L23R_SELF
#  undef SUF
#  undef WC
#  undef L23R_TEMPLATE

#  define L23R_TEMPLATE 2
#  define WC 2
#  define SUF(x) x##_w2
#  include L23R_SELF
#  undef SUF
#  undef WC
#  undef L23R_TEMPLATE
#else
#  error "L23_rader.c must be able to #include itself"
#endif

/* Width-independent pool job: first-touch the fused t1 arena with the SAME
 * static plane partition the fused plane loop uses at full width, so a
 * plane's reader owns its pages (L23_matrixsimd's design). */
static void l23r_work_t1f(const fft3d_plan *p, int tid)
{
    l23r_pool *pl = p->pl;
    const int T = pl->tw;
    if (tid >= T) return;
    const long npl = (long)p->batch * 23;
    long g0 = npl * tid / T, g1 = npl * (tid + 1) / T;
    for (long g = g0; g < g1; ++g)
        memset(p->t1f + (size_t)g * L23R_T1P, 0, L23R_T1P * sizeof(double));
    l23r_bar_join(pl, tid, T);
}

/* Worker: pin to the harness-consistent CPU, FIRST-TOUCH this thread's
 * scratch slot (NUMA locality: the owner touches it, after pinning), then
 * spin on the pool generation. */
static void *l23r_worker(void *arg)
{
    l23r_warg *wa = arg;
    struct fft3d_plan *p = wa->p;
    l23r_pool *pl = p->pl;
    const int tid = wa->tid;

    if (wa->cpu >= 0) {
        cpu_set_t s;
        CPU_ZERO(&s);
        CPU_SET(wa->cpu, &s);
        pthread_setaffinity_np(pthread_self(), sizeof s, &s);
    }
    memset(L23R_SLOT(p, tid), 0, L23R_SLOTSZ * sizeof(double));
    atomic_fetch_add(&pl->ready, 1);

    long last = 0;
    for (;;) {
        /* mt_r4: once the pick is final, non-participants (tid >= parkfrom,
         * which every job returns immediately and which join no barrier)
         * nap-poll instead of pause-spinning -- verdict 4.4's clock-drag
         * hygiene.  A late generation read only delays their empty workfn
         * call; the ++last catch-up loop below tolerates any backlog. */
        if (tid >= atomic_load_explicit(&pl->parkfrom, memory_order_relaxed)) {
            while (atomic_load_explicit(&pl->gen, memory_order_acquire) ==
                   last) {
                struct timespec pts = {0, 100000}; /* 100 us */
                nanosleep(&pts, NULL);
            }
        } else {
            while (atomic_load_explicit(&pl->gen, memory_order_acquire) ==
                   last)
                l23r_cpu_relax();
        }
        ++last;
        if (atomic_load_explicit(&pl->shutdown, memory_order_relaxed)) break;
        pl->workfn(p, tid);
    }
    return NULL;
}

/* package id of one CPU from sysfs; -1 if unreadable (BORROWED:
 * L23_matrixsimd mt_r2's socket map) */
static int l23r_pkg_of_cpu(int cpu)
{
    char path[96];
    int pkg = -1;
    snprintf(path, sizeof path,
             "/sys/devices/system/cpu/cpu%d/topology/physical_package_id", cpu);
    FILE *f = fopen(path, "r");
    if (f) {
        if (fscanf(f, "%d", &pkg) != 1) pkg = -1;
        fclose(f);
    }
    return pkg;
}

static double l23r_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

const char *fft3d_name(void) { return "L23_rader"; }

static const char *g_desc =
    "Rader p=23 folded pair, two-sweep X-first, spin-pool batch/fused MT";

const char *fft3d_description(void) { return g_desc; }

int fft3d_supports(int L) { return L == 23; }

static int l23r_verbose(void)
{
    const char *e = getenv("L23R_VERBOSE");
    return e && *e && *e != '0';
}

static void l23r_tune_free(fft3d_plan *p)
{
    free(p->ti); free(p->to);
    p->ti = NULL; p->to = NULL; p->tn = 0;
}

/* Deterministic pseudo-random tuning data (realistic magnitudes suffice).
 * Filled single-threaded ON PURPOSE: the driver first-touches in/out
 * single-threaded, so the tuner's arena must live on one socket too. */
static int l23r_tune_alloc(fft3d_plan *p, int nv)
{
    size_t n = (size_t)nv * 12167;
    if (p->tn >= n) return 1;
    l23r_tune_free(p);
    if (posix_memalign((void **)&p->ti, 64, n * sizeof *p->ti) != 0) { p->ti = NULL; return 0; }
    if (posix_memalign((void **)&p->to, 64, n * sizeof *p->to) != 0) {
        free(p->ti); p->ti = NULL; p->to = NULL; return 0;
    }
    p->tn = n;
    unsigned sr = 20260821u;
    for (size_t i = 0; i < n; ++i) {
        sr = sr * 1103515245u + 12345u;
        double a = (double)(sr >> 8) / 8388608.0 - 1.0;
        sr = sr * 1103515245u + 12345u;
        double b = (double)(sr >> 8) / 8388608.0 - 1.0;
        p->ti[i] = a + b * (double _Complex)I;
    }
    memset(p->to, 0, n * sizeof *p->to);
    return 1;
}

/* Streaming-regime tuner arena: in + out together must exceed the
 * machine's AGGREGATE cache, nthr*L2 + nsock*L3 (x3.5, cap 640 volumes).
 * BORROWED: L23_matrixsimd mt_r2 (their fix) + L6_pfa mt_r1 (the original
 * lesson).  mt_r2's 2.5x-one-L3 sizing gave the node nv=148 = 58 MiB --
 * inside its 76 MiB aggregate -- so the arena priced out-RFOs at L3 speed,
 * picked plain batch over NT, and scored 7.17 us/vol at B=2048 against
 * the rival's honest-arena 5.93. */
static int l23r_tune_nv(const fft3d_plan *p, int batch)
{
    long l2 = -1, l3 = -1;
#ifdef _SC_LEVEL2_CACHE_SIZE
    l2 = sysconf(_SC_LEVEL2_CACHE_SIZE);
#endif
#ifdef _SC_LEVEL3_CACHE_SIZE
    l3 = sysconf(_SC_LEVEL3_CACHE_SIZE);
#endif
    if (l2 <= 0) l2 = 1l << 20;   /* CLX/SPR-scale fallbacks */
    if (l3 <= 0) l3 = 32l << 20;
    double agg = (double)p->nslot * (double)l2 +
                 (double)p->pl->nsock * (double)l3;
    double nv = 3.5 * agg / (2.0 * 12167.0 * 16.0);
    int cap = nv < 144.0 ? 144 : (nv > 640.0 ? 640 : (int)nv);
    return batch < cap ? batch : cap;
}

/* tuner cell: decomposition mode x width x team x prefetch knobs x
 * input schedule.  bar/wt lost every mt_r3 node cell; their code stays
 * (env-forceable) but no cell races them any more. */
typedef struct {
    signed char mode; /* 0 serial, 1 batch, 2 batch-NT, 3 fused */
    signed char w;    /* vector width in complex: 4 or 2 */
    signed char team;
    signed char pf, pw;
    signed char bar;  /* 1 = socket-tree barrier (no-op when nsock == 1) */
    signed char wt;   /* weighted volume split: 0 even, 1 = 4:3, 2 = 5:3 */
    signed char si;   /* staged sequential input: 0/1 all/2 far-only */
    signed char pv;   /* paced next-volume input prefetch */
} l23r_cell;

static const char *const l23r_mode_name[4] = {"serial", "batch", "batchNT",
                                              "fused"};

typedef void (*l23r_fn)(const fft3d_plan *, const double _Complex *,
                        double _Complex *);

static l23r_fn l23r_resolve(int mode, int w)
{
    switch (mode) {
    case 0: return w == 4 ? x_serial_w4 : x_serial_w2;
    case 1:
    case 2: return w == 4 ? x_pool_rng_w4 : x_pool_rng_w2;
    default: return w == 4 ? x_pool_fused_w4 : x_pool_fused_w2;
    }
}

static void l23r_apply(fft3d_plan *p, const l23r_cell *c)
{
    p->exec = l23r_resolve(c->mode, c->w);
    p->team = c->team;
    p->pf = c->pf;
    p->pw = c->pw;
    p->ntc = c->mode == 2;
    p->bar = c->bar;
    p->wt = c->wt;
    p->si = c->si;
    p->pv = c->pv;
}

/* Finish: env overrides (L23R_PF / L23R_PW / L23R_TEAM, for same-window
 * A/Bs), bake pick + tuner telemetry into the description (leaderboard
 * carries one line of the node's own tuner table -- phase-1 pattern). */
static char g_dbuf[240];
static double g_tpick, g_tinc; /* tuned us/transform: picked cell, list head */
static int g_tnv;              /* tuner arena volumes (0 = tuner never ran) */
static void l23r_tune_finish(fft3d_plan *p, const l23r_cell *c)
{
    const char *e = getenv("L23R_PF");
    if (e && *e) {
        int v = atoi(e);
        p->pf = v <= 0 ? 0 : 2;
    }
    e = getenv("L23R_PW");
    if (e && *e) p->pw = atoi(e) ? 1 : 0;
    e = getenv("L23R_TEAM");
    if (e && *e) {
        int v = atoi(e);
        if (v >= 1 && v <= p->nslot) p->team = v;
    }
    e = getenv("L23R_BAR");
    if (e && *e) p->bar = atoi(e) ? 1 : 0;
    e = getenv("L23R_WT");
    if (e && *e) {
        int v = atoi(e);
        p->wt = v < 0 ? 0 : (v > 2 ? 2 : v);
    }
    e = getenv("L23R_SI");
    if (e && *e) {
        int v = atoi(e);
        p->si = v < 0 ? 0 : (v > 2 ? 2 : v);
    }
    e = getenv("L23R_PV");
    if (e && *e) p->pv = atoi(e) ? 1 : 0;
    /* park the never-participating tail of the pool now that the team is
     * final (tid >= team joins no barrier and runs no work) */
    int park = 1;
    e = getenv("L23R_PARK");
    if (e && *e) park = atoi(e) ? 1 : 0;
    atomic_store_explicit(&p->pl->parkfrom,
                          park ? p->team : p->pl->nthr, memory_order_relaxed);
    int pk = park && p->team < p->pl->nthr;
    if (g_tnv > 0)
        snprintf(g_dbuf, sizeof g_dbuf,
                 "rader23 pool %s w%d team=%d pf=%d pw=%d si=%d pv=%d bar=%d wt=%d pk=%d ns=%d, tuner pick=%.2f inc=%.2f us/t nv=%d",
                 l23r_mode_name[(int)c->mode], (int)c->w, p->team, p->pf,
                 p->pw, p->si, p->pv, p->bar, p->wt, pk, p->pl->nsock,
                 g_tpick, g_tinc, g_tnv);
    else
        snprintf(g_dbuf, sizeof g_dbuf,
                 "rader23 pool %s w%d team=%d pf=%d pw=%d si=%d pv=%d bar=%d wt=%d pk=%d ns=%d",
                 l23r_mode_name[(int)c->mode], (int)c->w, p->team, p->pf,
                 p->pw, p->si, p->pv, p->bar, p->wt, pk, p->pl->nsock);
    g_desc = g_dbuf;
    l23r_tune_free(p);
}

fft3d_plan *fft3d_create(int L, int batch)
{
    if (L != 23 || batch < 1) return NULL;
    fft3d_plan *p = calloc(1, sizeof *p);
    if (!p) return NULL;
    p->L = L;
    p->batch = batch;

    /* never more threads than the harness gave us */
    int T = omp_get_max_threads();
    if (T > L23R_MAXT) T = L23R_MAXT;
    if (T < 1) T = 1;
    p->nslot = T;

    /* one page of tables + T page-aligned scratch slots (first touch of
     * each slot happens on its OWNING pool thread, below) */
    size_t bytes = 4096 + (size_t)T * L23R_SLOTSZ * sizeof(double);
    void *blk = NULL;
    if (posix_memalign(&blk, 4096, bytes) != 0 || !blk) {
        free(p);
        return NULL;
    }
    p->block = blk;
    memset(blk, 0, 4096); /* tables page: touched by the planning thread */
    p->cd8 = (double *)blk;
    p->sd8 = p->cd8 + 88;
    p->cd4 = p->cd8 + 176;
    p->sd4 = p->cd8 + 220; /* 264 doubles total, well inside the page */
    p->slab = (double *)((char *)blk + 4096);

    /* long double trig so the double table is correctly rounded.  SINE
     * entries carry the -i rotation's im-lane sign: alternating {+s,-s}. */
    const long double twopi = 6.283185307179586476925286766559005768394L;
    for (int m = 1; m <= 11; ++m) {
        double c = (double)cosl(twopi * (long double)m / 23.0L);
        double s = (double)sinl(twopi * (long double)m / 23.0L);
        for (int t = 0; t < 8; ++t) p->cd8[(m - 1) * 8 + t] = c;
        for (int t = 0; t < 8; ++t) p->sd8[(m - 1) * 8 + t] = (t & 1) ? -s : s;
        for (int t = 0; t < 4; ++t) p->cd4[(m - 1) * 4 + t] = c;
        for (int t = 0; t < 4; ++t) p->sd4[(m - 1) * 4 + t] = (t & 1) ? -s : s;
    }

    /* ---- read the harness's OMP thread->CPU mapping from one throwaway
     * OpenMP region, so the pool pins exactly where OMP_PROC_BIND=close /
     * OMP_PLACES=cores would put each thread (also pins the initial thread,
     * pool tid 0, to place 0).  GOMP's own workers go to sleep after their
     * spin timeout and are never used again.  (L23_matrixsimd mt_r1.) ---- */
    int cpus[L23R_MAXT];
    for (int t = 0; t < L23R_MAXT; ++t) cpus[t] = -1;
#ifdef _OPENMP
#pragma omp parallel num_threads(T)
    {
        int t = omp_get_thread_num();
        if (t < L23R_MAXT) cpus[t] = sched_getcpu();
    }
    {
        int pin = getenv("OMP_PROC_BIND") != NULL;
        for (int a = 0; pin && a < T; ++a)
            for (int b2 = a + 1; b2 < T; ++b2)
                if (cpus[a] == cpus[b2]) pin = 0; /* unbound run: don't pin */
        if (!pin)
            for (int t = 0; t < L23R_MAXT; ++t) cpus[t] = -1;
    }
#endif

    /* main thread (pool tid 0) first-touches its own slot */
    memset(L23R_SLOT(p, 0), 0, L23R_SLOTSZ * sizeof(double));

    /* ---- package map of the pinned CPUs: drives the tree barrier, the
     * weighted streaming split, and the aggregate-cache arena size.  Any
     * unreadable id degrades safely to nsock=1 (flat barrier, even split).
     * L23R_FAKESOCK pretends the top half of the team is a second package
     * -- a DEV knob so wallaby (one socket) can exercise the tree/weighted
     * code paths for correctness. ---- */
    int nsock = 1;
    unsigned char faru[L23R_MAXT];
    memset(faru, 0, sizeof faru);
    {
        const char *fake = getenv("L23R_FAKESOCK");
        if (fake && *fake && *fake != '0') {
            if (T > 1) {
                for (int t = T / 2; t < T; ++t) faru[t] = 1;
                nsock = 2;
            }
        } else if (cpus[0] >= 0) {
            int p0 = l23r_pkg_of_cpu(cpus[0]);
            int ok = p0 >= 0;
            for (int t = 1; ok && t < T; ++t) {
                int pk = cpus[t] >= 0 ? l23r_pkg_of_cpu(cpus[t]) : -1;
                if (pk < 0) ok = 0;
                else faru[t] = pk != p0;
            }
            if (!ok)
                memset(faru, 0, sizeof faru);
            else
                for (int t = 1; t < T; ++t)
                    if (faru[t]) { nsock = 2; break; }
        }
    }

    /* ---- the persistent pool (PANEL_BRIEF: thread creation is setup) ---- */
    {
        void *pb = NULL;
        if (posix_memalign(&pb, 64, sizeof(l23r_pool)) != 0 || !pb) {
            fft3d_destroy(p);
            return NULL;
        }
        memset(pb, 0, sizeof(l23r_pool));
        p->pl = (l23r_pool *)pb;
        /* nobody parks until l23r_tune_finish sets the final team */
        atomic_store_explicit(&p->pl->parkfrom, L23R_MAXT,
                              memory_order_relaxed);
        p->pl->nthr = T;
        p->pl->tw = T;
        p->pl->nsock = nsock;
        memcpy(p->pl->far, faru, sizeof p->pl->far);
        for (int t = 1; t < T; ++t) {
            p->pl->wa[t].p = p;
            p->pl->wa[t].tid = t;
            p->pl->wa[t].cpu = cpus[t];
            if (pthread_create(&p->pl->th[t], NULL, l23r_worker,
                               &p->pl->wa[t]) != 0) {
                p->pl->nthr = t; /* join only what was spawned */
                atomic_store(&p->pl->fail, 1);
                break;
            }
        }
        T = p->pl->nthr;
        p->nslot = T;
        while (atomic_load(&p->pl->ready) < T - 1) l23r_cpu_relax();
        if (atomic_load(&p->pl->fail)) {
            fft3d_destroy(p);
            return NULL;
        }
    }
    p->team = T;

    /* ---- fused t1 arena (batch < 32), first-touched by the pool with the
     * plane partition so a plane's reader owns its pages ---- */
    if (batch < 32) {
        if (posix_memalign(&p->t1f_blk, 4096,
                           (size_t)batch * L23R_T1V * sizeof(double)) != 0 ||
            !p->t1f_blk) {
            fft3d_destroy(p);
            return NULL;
        }
        p->t1f = (double *)p->t1f_blk;
        l23r_run_job(p, l23r_work_t1f, NULL, NULL);
    }

    /* ---- canonical cell list per regime.  Order is a policy statement;
     * heads follow the fastest-known-head rule (phase-1 r11): the head is
     * the fastest cell ANY scoring measurement supports, and a challenger
     * must beat the running incumbent by >2% (two fixed-order sweeps,
     * per-cell min) so near-ties resolve identically in every process.
     * All cells are one bit class: element ownership is disjoint (or
     * overlapping with identical values within one thread) and per-element
     * arithmetic is fixed, so the tuner's pick can never change the
     * output. ---- */
    l23r_cell cells[24];
    int ncell = 0;
#define ADDC(m_, w_, t_, pf_, pw_, si_, pv_)                                   \
    do {                                                                       \
        if (ncell < 24) {                                                      \
            cells[ncell].mode = (m_); cells[ncell].w = (w_);                   \
            cells[ncell].team = (signed char)(t_);                             \
            cells[ncell].pf = (pf_); cells[ncell].pw = (pw_);                  \
            cells[ncell].bar = 0; cells[ncell].wt = 0;                         \
            cells[ncell].si = (si_); cells[ncell].pv = (pv_); ++ncell;         \
        }                                                                      \
    } while (0)

    const int TB = batch < T ? batch : T;
    const int T16 = T < 16 ? T : 16;
    const int twosock = p->pl->nsock == 2;
    if (batch >= 32) {
        /* streaming regime: volumes across the full team.  Heads = the
         * node's mt_r3 picks (plain pf0 pw0 at B=128, batchNT static at
         * B=2048).  New this round: the si/pv input-schedule ladder --
         * the mt_r3 verdict names the sequential-read discipline as what
         * separates L=36's 137-151 GB/s from L=23's 65.  DROPPED on node
         * evidence: wt cells (wt=0 in every mt_r3 string), team=16 rows
         * (verdict 5: four independent narrow-vs-wide refutations; and
         * my own t16 rows never won a node table in three rounds). */
        ADDC(1, 4, T, 0, 0, 0, 0);
        ADDC(1, 4, T, 2, 1, 0, 0); /* node B=128 1/3, wallaby L3-resident */
        ADDC(2, 4, T, 0, 0, 0, 0); /* NT static: node B=2048 pick */
        ADDC(2, 4, T, 2, 0, 0, 0); /* NT+pf2: wallaby streaming winner */
        ADDC(2, 4, T, 0, 0, 1, 0); /* NT + staged sequential input */
        ADDC(2, 4, T, 0, 0, 0, 1); /* NT + paced next-volume prefetch */
        ADDC(2, 4, T, 0, 0, 1, 1); /* NT + si + pv (pv feeds the copy) */
        ADDC(1, 4, T, 0, 0, 1, 0); /* plain + si (the B=128 regime) */
        ADDC(1, 4, T, 0, 0, 0, 1); /* plain + pv */
        if (twosock) ADDC(2, 4, T, 0, 0, 2, 0); /* NT + far-only staging */
        ADDC(1, 2, T, 0, 0, 0, 0); /* non-AVX-512 fallback */
    } else if (batch == 1) {
        /* B=1: head = fused T=16 flat, the node's scored pick in mt_r2
         * AND mt_r3 (11.5-11.9 us, both L23 entries; my tree cells lost
         * 9/9 node strings, dropped).  The cell is latency-closed -- two
         * kernels and a purpose-built tree barrier all measure the same
         * floor -- so this round's only lever is parking the 16 idle
         * workers (always on, not a cell).  Serial stays as the floor. */
        ADDC(3, 4, T16, 0, 0, 0, 0);
        ADDC(3, 4, T, 0, 0, 0, 0);   /* flat, full team: wallaby's winner */
        ADDC(3, 4, T16, 2, 0, 0, 0); /* fused X prefetch, one socket */
        ADDC(0, 4, 1, 0, 0, 0, 0);
        ADDC(3, 2, T, 0, 0, 0, 0);
    } else {
        /* 2..31 volumes: fused-on-pool heads (uses all T threads);
         * one-volume-per-thread stays raced -- it keeps t1 L2-private. */
        ADDC(3, 4, T, 0, 0, 0, 0);
        ADDC(3, 4, T, 2, 0, 0, 0); /* fused X prefetch */
        ADDC(1, 4, TB, 0, 0, 0, 0);
        ADDC(1, 4, TB, 2, 1, 0, 0);
        ADDC(2, 4, TB, 0, 0, 0, 0);
        ADDC(0, 4, 1, 0, 0, 0, 0);
        ADDC(3, 2, T, 0, 0, 0, 0);
    }
#undef ADDC

    int bestc = 0;
    l23r_apply(p, &cells[0]);

    const char *force = getenv("L23R_FORCE");
    if (force && *force) {
        int v = atoi(force);
        if (v >= 0 && v < ncell) {
            l23r_apply(p, &cells[v]);
            if (l23r_verbose())
                fprintf(stderr, "[L23_rader] FORCED cell %d: %s w%d team=%d pf=%d pw=%d si=%d pv=%d\n",
                        v, l23r_mode_name[(int)cells[v].mode], cells[v].w,
                        cells[v].team, cells[v].pf, cells[v].pw,
                        cells[v].si, cells[v].pv);
            l23r_tune_finish(p, &cells[v]);
            return p;
        }
    }

    {
        double best[24];
        for (int c = 0; c < ncell; ++c) best[c] = 1e30;
        /* small batches tune at the TRUE batch (the decomposition is a
         * function of it); streaming batches on an L3-exceeding arena */
        int nv = batch < 32 ? batch : l23r_tune_nv(p, batch);
        if (l23r_tune_alloc(p, nv)) {
            int sb = p->batch;
            p->batch = nv;
            int inner = batch < 32 ? (32 + nv - 1) / nv : 1;
            int nrep = batch < 32 ? 5 : 3;
            /* TWO full fixed-order sweeps, per-cell min across both; each
             * cell gets a licence-honest warmup >= 1.5 ms (phase-1 rules:
             * one sweep on a drifting clock mis-ranks distant cells, and
             * 512/256-bit licence transitions need the dwell) */
            for (int sweep = 0; sweep < 2; ++sweep)
                for (int c = 0; c < ncell; ++c) {
                    l23r_apply(p, &cells[c]);
                    double tw = l23r_now();
                    do p->exec(p, p->ti, p->to);
                    while (l23r_now() - tw < 1.5e-3);
                    for (int r = 0; r < nrep; ++r) {
                        double t0 = l23r_now();
                        for (int q2 = 0; q2 < inner; ++q2)
                            p->exec(p, p->ti, p->to);
                        double dt = l23r_now() - t0;
                        if (dt < best[c]) best[c] = dt;
                    }
                }
            p->batch = sb;
            /* hysteretic walk: >2% to displace the running incumbent */
            for (int c = 1; c < ncell; ++c)
                if (best[c] < 0.98 * best[bestc]) bestc = c;
            g_tinc = best[0] * 1e6 / ((double)nv * inner);
            g_tpick = best[bestc] * 1e6 / ((double)nv * inner);
            g_tnv = nv;
            if (l23r_verbose())
                for (int c = 0; c < ncell; ++c)
                    fprintf(stderr,
                            "[L23_rader tune nv=%d] %-8s w%d team=%2d pf=%d pw=%d si=%d pv=%d %9.2f us/transform%s\n",
                            nv, l23r_mode_name[(int)cells[c].mode], cells[c].w,
                            cells[c].team, cells[c].pf, cells[c].pw,
                            cells[c].si, cells[c].pv,
                            best[c] * 1e6 / ((double)nv * inner),
                            c == bestc ? "  <== kept" : "");
        }
        l23r_apply(p, &cells[bestc]);
    }

    l23r_tune_finish(p, &cells[bestc]);
    return p;
}

void fft3d_execute(fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    p->exec(p, in, out);
}

void fft3d_destroy(fft3d_plan *p)
{
    if (!p) return;
    if (p->pl) {
        atomic_store_explicit(&p->pl->shutdown, 1, memory_order_relaxed);
        ++p->pl->gen_local;
        atomic_store_explicit(&p->pl->gen, p->pl->gen_local,
                              memory_order_release);
        for (int t = 1; t < p->pl->nthr; ++t) pthread_join(p->pl->th[t], NULL);
        free(p->pl);
    }
    l23r_tune_free(p);
    free(p->t1f_blk);
    free(p->block);
    free(p);
}

#endif /* template / main body */
