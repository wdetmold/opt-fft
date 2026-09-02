/* gen_rader -- Rader-class prime entry, round gen_r1.  Owns L=31.
 *
 * FOLDED RADER: the conjugate-pair fold (s_j = x_j + x_{31-j}, d_j = x_j - x_{31-j})
 * turns the 31-point DFT into
 *     X_k     = x_0 + E_k - i O_k,   X_{31-k} = x_0 + E_k + i O_k,
 *     E_k = sum_j cos(2pi jk/31) s_j,   O_k = sum_j sin(2pi jk/31) d_j,  j,k = 1..15.
 * Indexing both j and k through the multiplicative quotient group Z31* mod {+-1}
 * (cyclic of order 15, generator 3) makes the E system a CYCLIC-15 correlation of
 * s with the cos kernel and the O system a NEGACYCLIC-15 correlation of sign-twisted
 * d with the sin kernel; because 15 is odd, the negacyclic one converts to cyclic via
 * diagonal +-1 twists that fold entirely into the (compile-time) load/store index
 * tables.  Correlation -> convolution by reversing the (precomputed) kernel.
 *
 * Each cyclic-15 convolution is computed as Winograd-C3 (4 block products, 11 block
 * adds) NESTED OVER dense cyclic-5 blocks (25 FMA each): 100 FMA + 65 vector adds
 * per convolution, ALL constants real, so on interleaved complex data every op is a
 * plain zmm add/FMA -- no complex-multiply shuffles anywhere.  Per 4 pencils per
 * axis: ~435 zmm ops vs the folded-dense entry's ~555, with ~40 broadcast constants
 * instead of ~450 table loads.  (Dense C5 blocks beat Winograd C5 blocks on FMA
 * hardware: 25 FMA < 10 mul + 31 add.  The full-Winograd C15 at 40 mul + 179 add
 * would LOSE to this hybrid -- same lesson as ice L23_rader's "121 fused FMAs beat
 * any sub-quadratic length-11 convolution".)
 *
 * Chassis (pass order, z-row kernel, chain scheme, map) ADOPTED from
 * gen_dense_prime gen_r1 (itself from the ice-campaign records):
 *   P1 z rows contiguous (their zpass31 row-pair GEMM, verbatim), P2 x-axis
 *   inner=961, P3 y-axis per-plane in place; volume-resident fused chain;
 *   map = pair-compressed |w|^2, rsqrt14+2NR, ONE vdivpd per 8 points.
 * gen_r2: the chain runs on a fully PADDED private state (z-rows 31 -> 32
 * complex, planes -> 1148 complex == 124 mod 256) -- every access in every
 * pass 64B-aligned, the x-pass's exact 4K store->load aliases (row stride
 * 961*16: store row j+4 == next chunk's load of row j in low-12 bits) pushed
 * outside the 31-row system, z = 8 uniform quads/plane (row 32 is a zeroed
 * pad row), y and map tail-free.  See R31_ZP/R31_PP and r31_chain_volume.
 *
 * gen_r3: the class takes ANY odd prime 3 <= p <= 127 (the round-3 duty).
 * Primes != 31 run a GENERIC folded half-system engine (rp_*): the same
 * conjugate fold, C_k = x0 + sum_j cos(2pi jk/p) u_j, S_k = sum_j sin v_j,
 * X_k = C_k -+ iS_k, computed by a runtime-(p,h) column-chunk kernel with
 * k in quads (4 C + 4 S accumulators sharing each u_j/v_j load; all-real
 * broadcast constants, no complex-multiply shuffles) and the z axis through
 * the 4x4-complex transpose quad (r31_tp4 reused).  The chain runs fully in
 * place on the out volume (r1 form), map per plane.  The 31 fast path is
 * unchanged.  st/cpad for 31 now live in ONE 2MiB huge-page arena with the
 * c mirror at a +2048 B page phase (gen_layout gl_map_huge; gen_dense_prime
 * r3 found two same-phase ~500 KB aligned_allocs make the map's c loads
 * 4K-alias the y-pass state stores).
 *
 * gen_r4: PLANE CUSTODY in both chains (gen_layout r3 / gen_bluestein r4's
 * window idea): step s+1's z-pass is plane-local, so it runs right after each
 * plane's map while the plane is cache-hot -- one full-state read per step
 * deleted, bit-identical outputs (verified by cmp against the r3 binary).
 * Raced and rejected same-core (gen_batchlane r4 protocol): x-pass software
 * prefetch (-DR31_PFX, +0.7%) and the map fused into the z-quads' transpose-in
 * loads (-DR31_ZMAPF, +4% -- the panel's FIFTH map-fusion negative, first on
 * the load side).  -DR31_R3CHAIN / -DRP_R3CHAIN restore the r3 pass order.
 *
 * gen_r5, two changes.  (1) OUTER-C3 RADER for any prime whose quotient
 * group order is h = 3m, h odd, gcd(3,m) = 1 (43, 67, 79, 103): the r31
 * Winograd-C3-over-dense-blocks machinery at runtime tables (rp3_*), per-m
 * compile-time-instantiated chunk kernels -- 8m^2 + O(m) conv FMA vs the
 * dense engine's 18m^2.  Node: -34% at 43, -41% at 67, -31% at 79, -16% at
 * 103 (DRAM-hidden).  (2) the generic-prime chain moves onto a padded
 * huge-page arena for p <= RP_PAD_MAX = 61 (the r2 lesson at 31): z-rows p
 * -> mult-of-4 complex (row bases 64B-aligned, y tail-free), plane pitch
 * rp_pick_pp with PP/4 == 2 mod 4 (the +64B x-pass store->load alias
 * equation d*(PP/4) == 1 mod 64 becomes UNSOLVABLE), state + c mirror in
 * one gl_map_huge 2MiB arena (gen_layout, adopted as a LIB_ONLY include),
 * c at a +2048B page phase; outputs bit-identical to the flat chain
 * (lanewise ops only; cmp-verified).  -1.4% at 37, wash 13..61, and a
 * measured LOSS at DRAM sizes (-4% at 127: the per-volume c-mirror +
 * copy-out do not amortize at the small m large L implies) -- hence the
 * size gate.  -DRP_FLATCHAIN / -DRP_NOC3 restore the r4 arms.  The 31 path
 * is untouched (r4 showed it sits AT its issue-port model; this round's
 * -DR31_ZMIX port-5 rebalance race lost +2.5%/+7%, closing that item).
 *
 * gen_r6, two changes (the surprise-round generality round).  (1) EVEN-h
 * SPLIT RADER (rp2_*) for every prime p == 1 mod 4 in class (13 17 29 37 41
 * 53 61 73 89 97 101 109 113): the E (cos) system is cyclic-h for ANY h; the
 * O (sin) system is genuinely negacyclic when h = 2m is even (no odd-h twist
 * exists).  E splits by CRT z^{2m}-1 = (z^m-1)(z^m+1) into dense cyclic-m +
 * negacyclic-m products; O splits by y = z^2 (y^m = -1) into a 3-product
 * negacyclic-m Karatsuba: 5m^2 conv FMA vs the dense engine's 2h^2 = 8m^2.
 * Rows fold ONCE into stack U/V arrays at compile-time offsets (p = 4m+1
 * per instantiation) with a negated-V mirror so both fills are stack loads.
 * Node: -25..-41% vs the dense engine at every one of the 13 primes.
 * (2) CODEGEN: gcc 11 leaves the m >= 13 conv loops ROLLED with memory
 * accumulators (the r5 "register-resident" story was only true at m <= 11);
 * RP_UNROLL pragmas force full peeling, and above the spill ceiling an
 * 8-accumulator j-tiled blocked conv (RP2_CONV_BLK, the dense engine's
 * k-quad shape) takes over -- form chosen PER m by same-core node races
 * (rp2: unroll <= 9, blocked >= 10; rp3: unroll <= 13, blocked at 17).
 * The pragmas alone move the r5 outer-C3 primes -11..-38% (103: 17.8 -> 11.0
 * ms).  The 31 path is untouched (bit-identical, cmp-verified).
 *
 * gen_r7, three changes, all codegen/memory (an RP_PROF per-pass profile of
 * the flat chain drove every one; arithmetic untouched, every chain output
 * bit-identical to r6, cmp-verified).  (1) The RP2_CONV_BLK i loop is PINNED
 * rolled: -funroll-loops was peeling it 8x, compiling rp2_chunk_28 to 10820
 * instructions / ~75 KB PER CHUNK -- past L1I, so the passes were FRONT-END
 * bound even on cache-hot data (y-pass 6.7 ms/step at p=113 vs a ~2.4 ms
 * load-port model).  -7% at 89, -19% at 101, -8% at 113, -9..11% at 61 on
 * the node; m=13 (p=53) alone raced better with the gcc default and keeps it.
 * (2) The dead map-fused arms (RP_YMAPFUSE lost in r3; every caller passes
 * mapc=NULL) are compiled out via RP_MAPARM: ~57 never-executed rsqrt/rcp
 * ladders (~20 KB) deleted from every rp2/rp3 chunk; -0.6..-1.8% more at
 * 61/101/103/113.  (3) x-pass software prefetch (rp_pass pf arg): the x-pass
 * walks p+1 row streams at plane pitch -- past the L2 streamer -- so at
 * DRAM-resident sizes each chunk's row loads demand-miss; T0 one chunk ahead
 * is -8% on that pass at 113.  Gated to state+c > RP_PFMIN_KB (36 MiB, p >=
 * 107): raced wash-to-negative at 89/101, -1.5% at 113, -0.5% at 127.
 * Closed by analysis (see strategy record): x-band custody blocking is a
 * no-op (the custody chain already sits at the 5-sweep-per-step floor), and
 * gen_layout's NT-store win cannot transfer (in-place passes pay no RFO).
 *
 * gen_r8: PAIRED-COLUMN (2-wide) chunks -- the r7 next-step-1 lever spent.
 * The conv/k-quad inner loops were load-port bound with exactly-8 FMA chains
 * (zero latency slack: 2 pipes x 4-cyc latency); processing two zmm columns
 * per chunk shares every broadcast constant between the columns (gcc 11
 * additionally 4x-unrolls the pinned-rolled i loop with broadcast ROTATION --
 * the shifted kernel index reuses each broadcast across 4 iterations, so the
 * steady state is ~2 loads + 2 broadcasts + 16 FMA per column-pair step).
 * Node-raced (same-core interleaved, 3/3 per size): dense engine 2-wide
 * -25.5% at 127 (47.2 -> 35.2 ms, ~10% of it from the paired z-quad); the
 * fully-peeled 2-wide form loses EVERYWHERE (60 KB static at m=28 -- the r7
 * front-end lesson again); with fold/combine ROLLED the 2-wide wins vs r7 at
 * m >= 22, but the attribution race showed rolling those loops on the 1-WIDE
 * kernel is the better form until m = 25.  Shipped: 1-wide rolled at
 * m 15..24, 2-wide rolled + natural-store-order at m >= 25, dense pairing
 * everywhere dense runs.  Per-column arithmetic order is unchanged, so all
 * outputs stay bit-identical (cmp-verified at every prime).  Knobs for the
 * xarch race: RP_NOW2 / RP_W2MIN / RP_NOW2D / RP_NOW2Z / RP_W2FULL /
 * RP_W1ROLL / RP_W1FULL / RP2_NOSTORD.
 *
 * gen_r9 (developed model-side: both Ice Lake nodes queued-busy the whole
 * round; the monitor's NOTICE says develop with the analyzers + wallaby
 * correctness runs).  PAIRED-COLUMN RP3_WINO (rp3_chunk2_17) for p = 103 --
 * the r8 next-step-1 item -- built, verified bit-identical (30-prime cmp
 * battery vs the r8 binary, execute + chain), and shipped DEFAULT OFF
 * behind -DRP_W23: on the dev host (Gold 6448Y = SAPPHIRE RAPIDS, a graded
 * xarch machine) it LOSES +35% 3/3 while the node-proven rp2 pairing at 113
 * loses only +2.8% there, so the loss is a property of this kernel (rp3's
 * glue/conv ratio, see rp3_chunk2 comment), not dev-host bias.  The Ice
 * Lake race is staged in build/tryout/gen_rader/r9_ab.sh; adopt only on a
 * node win.  create() remains fully deterministic (no runtime races; the
 * PMU-audit avenue-1 duty is discharged by construction and verified by a
 * 5x create() battery on wallaby).
 *
 * gen_r10 (the staged r9 races run on the landed ICL hold, a81n2): (1) the
 * 2-wide RP3_WINO at 103 LOSES on Ice Lake too (+13% 3/3, 13.1 vs 11.4-11.6
 * ms/step) -- rp3 pairing is CLOSED on both graded architectures; the
 * glue-ratio boundary stands: pairing pays at glue:conv <~ 1:4 (dense, rp2
 * big-m), never at rp3's ~1:2.  (2) The r8 "97 sides with 89" interpolation
 * was wrong: 2-wide+stord WINS at m=24 (-4..6%, 3/3 non-overlapping), so
 * RP_W2MIN drops 25 -> 24 -- every rp2 boundary is now raced, not
 * interpolated.  (3) x-pass prefetch-only at 103 raced a wash (+-0.5%,
 * overlapping mins): RP_PFMIN_KB stays 36 MiB.  Everything else untouched;
 * outputs bit-identical to r8/r9 at every prime (2-wide preserves
 * per-column arithmetic order; cmp-verified on the node at 97/103).
 *
 * gen_r11 (all hands on L=100; this class's angle = its own DRAM-regime primes,
 * 101..127).  SHIPPED CODE UNCHANGED FROM r10 (this comment + description
 * only; instruction stream identical -- objdump diff shows only rodata
 * displacements shifted by the longer description string -- and chain
 * outputs cmp-identical on the node).  Built, verified, and REVERTED on node
 * evidence: (a) flat-chain prologue "memcpy + in-place z_0 -> direct z_0(x0)"
 * (-2 volume sweeps on paper) LOSES ~+0.6% at 127 m=2 -- glibc's 32 MB memcpy
 * writes NT (no RFO) and the in-place z_0 re-owns lines via its own loads, so
 * the "saved" sweep was cheaper than an out-of-place RFO'd z_0; (b) gen_layout
 * r11's zero-copy THP re-home (state -> gl_map_huge arena, last-step map writes
 * final_out directly) LOSES +4.1% at 127 m=2, 4/4: the in-place chain pays NO
 * RFO anywhere (r7), so the "zero-copy" exit adds a whole cold-destination RFO
 * sweep per chain, and the TLB prize is bounded MEASURED at 0.6% of cycles
 * (dtlb walk_active 117M/19.0G at 127; re-home cuts it to 48M, worth ~0.3%).
 * Per-chain fixed costs cannot amortize at the m ~ 4 the suite implies at
 * large L (the r5 arena lesson, re-derived in zero-copy clothing).  Both wash
 * at m=8.  Counter dashboards refreshed at 113/127 under gen_dense_prime r11's
 * corrected currency (no ~2.1 uop cap; 512-bit L1 accesses pool at ~1.12/cyc):
 * neither pool saturated at 113/127 (p0+p5 0.85-0.90, accesses ~0.5/cyc) --
 * the residue is chunk-local L1 thrash (measured ~1700 line fills per
 * chunk-pair at 113 vs an ~83 KB/chunk working set), latency- not
 * throughput-bound, consistent with r10's closure.
 *
 * gen_r13 (the benchFFT B=1 small-L round).  Probing the class found the same
 * hole at the TINY primes: MKL beat this entry at every p <= 13 at B=1 (13:
 * 8.4 vs 6.1 us; 5/7: 2x) while losing 3-10x at p >= 17.  Skip-knob timing at
 * 13 put every pass at ~3x its uop model -- at m = 3 the runtime-table
 * machinery is all fixed cost (dispatch + call per 4 columns, table-pointer
 * and index loads that cannot hoist, runtime j*rs address arithmetic).  Fix =
 * the r31 lesson in miniature: COMPILE-TIME engines for p <= 13, execute path
 * only.  p=13: rp13_* -- the rp2 m=3 arithmetic with the p=13 Rader tables
 * hardcoded (memcmp-verified against rp2_build at create(); generic path on
 * mismatch), chunks force-inlined, strides/masks/trip counts literal.
 * p=3/5/7/11 (dense): rpd_* -- rp_chunk is already always_inline with p/h as
 * parameters, so instantiating it at literal p/h constant-folds everything
 * for free.  Node B=1: 13: 8.39 -> 4.20 us (1.44x over MKL), 11: 5.26 -> 3.14
 * (MKL-4%), 7: 1.54 -> 0.66, 5: 0.67 -> 0.31, 3: 0.085 (all ahead of MKL but
 * 11).  Outputs bit-identical to r12 at 31 (exec + chain) and 37; all 31
 * gates identical digits.  Raced and REJECTED: paired-interleaved z-quads at
 * 11 (+4% -- the store-forward-stall theory is dead).  RP_NO13 / RP_NOD13
 * restore the generic dispatch; RP_SKIP{Z,X,Y} are timing-only dev knobs
 * (wrong output, self-check bypassed, never default).
 *
 * gen_r14 (the benchFFT B=1 execute() round).  Two moves, one shipped.
 * (1) execute()-onto-the-padded-arena at 31 (the round's named seam: the
 * chain has run on the r2 arena since r2, execute() stayed flat) BUILT,
 * RACED and REJECTED: B=1 90.8-104 vs 76.3-80 flat (+15-20%, 4/4), B=16
 * +14%.  Mechanism: execute's flat z is already OUT-OF-PLACE (src->dst, no
 * store->load aliasing) and y runs at small in-plane strides, so only the
 * in-place x-pass pays the r2 alias tax; the arena adds a third buffer
 * (st, 574 KB) pushing the per-call working set past L2.  Execute IS m=1 --
 * the r5/r11 "per-volume fixed costs do not amortize at small m" law from a
 * third direction.  Flat execute sits AT its ~73 us three-pass model.  Kept
 * opt-in (-DR31_PADEXEC); the r31_chunk/pass_core dest-stride (ds) refactor
 * it needed ships (all pre-r14 sites pass ds == rs; chain cmp-identical).
 * (2) DENSE Z-ROWS at p=11 (rpd11_*, the r13 next-step-1 item): the
 * transpose-quad z paid 12 tp4 shuffles + a staged stack round trip per 4
 * pencils; the dense form folds each row in 512-bit (one lane-reversed
 * mirror shuffle), accumulates the half-spectrum in one zmm (k=0..3) + one
 * ymm (k=4..5) per C/S system -- the ymm half co-issues on port 1, idle in
 * every kernel on this panel (PMU audit avenue 4) -- and stores the
 * conjugate side lane-reversed (R31_ZSTORE pattern).  4 rows share every
 * table load; duplicated-pair trig in ctd/std_ (31's fields, unused at
 * generic p).  Node: 11 B=1 2.663-2.674 vs 2.780-2.791 quad (-4.2%, 4/4);
 * MKL gap -5% -> -1.5% (2.63 vs 2.66), fftw3 beaten 1.6x.  -DRPD11_QZ
 * restores the quad z.
 *
 * create() SELF-CHECKS the fast engine against a dense reference volume at 1e-13
 * and falls back to the (slow, correct) dense-matrix path if the check fails --
 * a fast wrong answer scores nothing.
 */
#include <complex.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>

#ifdef __AVX512F__
#include <immintrin.h>
#endif

#include "../fft3d_api.h"

/* gen_layout library layer, adopted gen_r5: gl_map_huge/gl_unmap (2MiB THP
 * arena, prefaulted, heap fallback) back the generic-prime padded chain state */
#define GEN_LAYOUT_LIB_ONLY
#include "gen_layout.c"

typedef double _Complex cplx;

static const long double PIL = 3.141592653589793238462643383279502884L;

/* ---- index tables (derived + verified against a reference DFT offline) ----
 * slot n = 5*i + j maps to CRT point ((10i+6j) mod 15) of the cyclic-15 system.
 * JS:  E-sweep fill:  S[n] = x[JS] + x[31-JS]              (s fold, reversal baked in)
 * JDP/JDM: O-sweep fill: S[n] = x[JDP] - x[JDM]            (d fold; (-1)^q eps_q twist
 *                                                           baked into the p/m swap)
 * KP/KM: output rows: X[KP] = T - iO~, X[KM] = T + iO~     (eps_t (-1)^t twist baked
 *                                                           into the pair order)    */
static const int R31_JS[15]  = {1, 2, 4, 8, 15, 5, 10, 11, 9, 13, 6, 12, 7, 14, 3};
static const int R31_JDP[15] = {1, 2, 4, 8, 16, 5, 10, 20, 9, 18, 25, 19, 7, 14, 28};
static const int R31_JDM[15] = {30, 29, 27, 23, 15, 26, 21, 11, 22, 13, 6, 12, 24, 17, 3};
static const int R31_KP[15]  = {1, 16, 8, 4, 2, 25, 28, 14, 7, 19, 5, 18, 9, 20, 10};
static const int R31_KM[15]  = {30, 15, 23, 27, 29, 6, 3, 17, 24, 12, 26, 13, 22, 11, 21};

/* Padded chain-state layout.  z-rows padded 31 -> R31_ZP = 32 complex (512 B:
 * every z/y-pass row access 64 B-aligned; natural 496 B rows line-split 3 of 4
 * accesses).  Planes padded 992 -> R31_PP = 1148 complex, == 124 mod 256, so
 * the x-pass row stride is 1148*16 B == 31*64 B mod 4096: the nearest 4K
 * store->load alias sits at row distance 33 -- outside the 31-row system --
 * where the natural 961-pitch put a store to row j+4 at EXACTLY the low-12
 * bits of the next chunk's load of row j.  Pad slots are zeroed once at
 * create(); every pass maps zeros to zeros (columns never mix), so they stay
 * zero and cost only ~3% extra x-pass/map lanes -- bought back by tail-free
 * uniform chunks everywhere (992 = 124 x 8, 32 cols = 8 chunks, z = 8 quads). */
#define R31_ZP 32
#define R31_PP 1148

struct fft3d_plan {
    int L, batch;
    int h;               /* (L-1)/2 */
    int fast;            /* 1: AVX-512 Rader engine passed the self-check */
    double *ke, *ko;     /* transformed conv kernels: 4 blocks x 5 doubles each */
    double *ctd, *std_;  /* z-pass duplicated-pair trig tables [15][32] */
    double *gct, *gst;   /* generic-prime fold tables, k-major [h][h] (L != 31) */
    int m3;              /* outer-C3 Rader mode: m = h/3 when active, else 0 */
    int *j3;             /* 6 x h ints: js, p-js, jdp, jdm, kp, km (slot order) */
    double *k3e, *k3o;   /* transformed conv kernels: 4 stretched blocks x (2m-1) */
    int m2;              /* even-h split mode (gen_r6): m = h/2 when active, else 0 */
    int use13;           /* gen_r13: compile-time p=13 execute engine enabled */
    int *j2;             /* 6 x h ints: js, p-js, jdp, jdm, kp, km (conv order) */
    double *k2;          /* 5 stretched kernel blocks x (2m-1): kc kn kb0 kb1 kb2 */
    int zp, pp;          /* generic padded chain: row length / plane pitch, complex */
    int xpf;             /* x-pass prefetch distance in bytes (0 = off) */
    cplx *gs, *gc;       /* generic padded chain state + c mirror (in gmap) */
    gl_map gmap;         /* huge-page mapping backing gs+gc (gen_layout) */
    cplx *t1;            /* scratch volume */
    cplx *st;            /* padded chain state, 31 planes x R31_PP (pads zeroed) */
    cplx *cpad;          /* padded mirror of the chain's c volume, same layout */
    cplx *w;             /* dense LxL DFT matrix (self-check + fallback) */
    cplx *tmp;           /* fallback scratch volume */
    void *arena;         /* 2MiB huge-page arena backing st+cpad (may be NULL) */
    size_t alen;
};

const char *fft3d_name(void) { return "gen_rader"; }
const char *fft3d_description(void)
{
    return "Rader-class primes 3..127: at 31, conjugate fold -> cyclic-15 (cos) + "
           "negacyclic-15 (sin; odd-N sign-twist), Winograd-C3 x dense-C5 on a fully "
           "padded huge-page arena (64B-aligned, anti-4K pitch, c mirror phase-split); "
           "43/67/79/103 via OUTER-C3 Rader (the same Winograd-C3-over-dense-blocks "
           "at runtime tables, 8m^2 vs 18m^2 conv FMA); even-h primes (p=1 mod 4, "
           "13..113) via the E-side C2 CRT split + O-side negacyclic Karatsuba "
           "(5m^2 vs 8m^2); any other prime via the "
           "generic folded half-system engine; generic chain on a padded gl_map_huge "
           "arena for p<=61 (alias-free pitch, tail-free x/y), flat above; "
           "r7: blocked convs pinned rolled (front-end fix, -7..-21% at 61..113), "
           "dead map arms compiled out, x-pass stream prefetch at p>=107; "
           "r8: paired-column (2-wide) chunks share broadcast constants (dense "
           "engine -25.5% at 127; 2-wide+stord at m>=25), rolled fold/combine "
           "on the 1-wide kernels at m=15..24 (-1..-10% at 61/73/89); "
           "r9/r10: 2-wide RP3_WINO at p=103 CLOSED -- loses on BOTH graded "
           "architectures (SPR +35%, ICL +13%; rp3's ~1:2 glue:conv ratio); "
           "r10: rp2 pairing boundary fully raced, 2-wide+stord extended to "
           "m=24 (p=97, -4..6% on ICL); "
           "r11: unchanged (.text-identical) -- flat-chain memcpy deletion and "
           "gen_layout-r11 zero-copy THP re-home both raced and REJECTED at "
           "127 (+0.6% / +4.1% at m=2: in-place chains pay no RFO, NT memcpy "
           "is the optimal prologue, TLB prize measured 0.6% of cycles); "
           "r12: 4-wide dense chunks with an E/O phase split at p>=59 (table "
           "walk once per 16 columns at the 2-wide 512b load ratio; -12.2% at "
           "127, -9.6% at 107, -6..9% at 59..83 -- broadcast-uop deletion, "
           "port_2_3 -20%/step); "
           "r13: COMPILE-TIME tiny-prime execute engines (the benchFFT B=1 "
           "small-L round, applied to this class: MKL beat the runtime-table "
           "engines at every p <= 13 at B=1) -- p=13 rp2-m3 arithmetic with "
           "hardcoded Rader index tables (rp13_*, create()-verified vs "
           "rp2_build), p=3/5/7/11 via rp_chunk instantiated at literal p/h "
           "(rpd_*); strides, masks and trip counts all compile-time: "
           "-40..-57% (13: 8.4 -> 4.2 us B=1, 1.44x over MKL; 5/7 now ahead, "
           "11 at MKL-4%); outputs bit-identical to r12 at p >= 17 and 31; "
           "r14 (the B=1 execute() round): execute()-onto-the-padded-arena "
           "raced and REJECTED at 31 (+15-20% B=1, 4/4 -- execute IS m=1: the "
           "third buffer pushes src+st+dst past L2; flat execute's z is "
           "already out-of-place and sits AT its 3-pass model, 76 us vs "
           "libraries' 755+; kept opt-in as -DR31_PADEXEC); DENSE Z-ROWS at "
           "p=11 (r31_zrow shape at compile-time 11: one-zmm fold vs "
           "lane-reversed mirror, k=0..3 zmm + k=4..5 ymm halves co-issuing "
           "on idle port 1, ~1 port-5 shuffle/row vs the transpose-quad's "
           "~12): -4.2% at 11 B=1, MKL gap -5% -> -1.5%; "
           "self-check gated; s6 map from gen_dense_prime, arena from gen_layout";
}
static int rp_is_prime(int n)
{
    if (n < 2) return 0;
    for (int d = 2; d * d <= n; ++d)
        if (n % d == 0) return 0;
    return 1;
}
int fft3d_supports(int L) { return L >= 3 && L <= 127 && rp_is_prime(L); }

/* ---------------- plan-time tables ---------------- */

/* Winograd-C3 kernel-side transform of a cyclic-15 kernel K (long double in):
 * blocks H_i[j] = K[(10i+6j) mod 15]; out = { (H0+H1+H2)/3, (H0-H2)/3,
 * (H1-H2)/3, (H0-H1)/3 }, 20 doubles. */
static void r31_kernel_transform(const long double *K, double *out)
{
    long double H[3][5];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 5; ++j)
            H[i][j] = K[(10 * i + 6 * j) % 15];
    for (int j = 0; j < 5; ++j) {
        out[j]      = (double)((H[0][j] + H[1][j] + H[2][j]) / 3.0L);
        out[5 + j]  = (double)((H[0][j] - H[2][j]) / 3.0L);
        out[10 + j] = (double)((H[1][j] - H[2][j]) / 3.0L);
        out[15 + j] = (double)((H[0][j] - H[1][j]) / 3.0L);
    }
}

/* E kernel: CC_r = cos(2pi 3^r/31); O kernel: SN'_r = (-1)^r sin(2pi 3^r/31). */
static void r31_build_kernels(double *ke, double *ko)
{
    long double CC[15], SN[15];
    long g = 1;
    for (int r = 0; r < 15; ++r) {
        long double th = 2.0L * PIL * (long double)g / 31.0L;
        CC[r] = cosl(th);
        SN[r] = (r & 1) ? -sinl(th) : sinl(th);
        g = (g * 3) % 31;
    }
    r31_kernel_transform(CC, ke);
    r31_kernel_transform(SN, ko);
}

/* duplicated-pair trig layout for the z-pass (from gen_dense_prime):
 * row j-1 holds (w_{j,0}, w_{j,0}, ..., w_{j,15}, w_{j,15}) = 32 doubles */
static double *r31_trig_dup(int want_sin)
{
    double *t = aligned_alloc(64, 15 * 32 * sizeof(double));
    if (!t) return NULL;
    for (int j = 1; j <= 15; ++j)
        for (int k = 0; k <= 15; ++k) {
            long m = ((long)j * k) % 31;
            long double th = 2.0L * PIL * (long double)m / 31.0L;
            double w = want_sin ? (double)sinl(th) : (double)cosl(th);
            t[(size_t)(j - 1) * 32 + 2 * k]     = w;
            t[(size_t)(j - 1) * 32 + 2 * k + 1] = w;
        }
    return t;
}

/* padded z-row length for the generic chain: mult of 4 complex, so every row
 * base is 64B-aligned and the y-pass (ncols = zp -> 2*zp doubles, mult of 8)
 * is tail-free.  Plain round-up: a phase-spreading bump to avoid zp/4 == 0
 * mod 4 strides was RACED AND LOST on the node (127: zp 132 vs 128 +1.5%,
 * 61: zp 68 vs 64 +4% -- the extra pad lanes cost more than 4K-phase spread
 * buys; the set-conflict theory is dead, do not resurrect it) */
static int rp_zpc(int p) { return (p + 3) & ~3; }

/* padded-arena chain only where it measured win-or-wash: state + c mirror
 * L2/L3-resident.  Above this the chain is DRAM-resident, m is small by the
 * suite's construction (~0.4 s of MKL per case), and the arena's per-volume
 * overheads (c mirror fill + final copy-out, ~2 volume sweeps) do not
 * amortize: measured -4% at 127 m=4.  Boundary measured: win at 37, wash at
 * 61, loss at 127. */
#ifndef RP_PAD_MAX
#define RP_PAD_MAX 61
#endif

/* plane pitch for the padded generic chain: >= the live plane (p rows x zp),
 * mult of 8 complex (map window tail-free, plane bases 128B-aligned), and
 * PP == 8 mod 16 so PP/4 == 2 mod 4: the x-pass row stride is an EVEN number
 * of 64B units mod 4096, making the +64B store->load alias condition
 * d*(PP/4) == 1 (mod 64) UNSOLVABLE at every row distance d (the r2 closed
 * form at 31 only pushed the first solution outside the row system), while
 * consecutive planes' 4K phases still step by 2 lines -- no set pileup. */
static int rp_pick_pp(int live)
{
    int pp = (live + 7) & ~7;
    if (pp % 16 != 8) pp += 8;
    return pp;
}

/* generic-prime fold tables, k-major so a k-quad walks 4 linear rows:
 * t[(k-1)*h + (j-1)] = cos/sin(2pi jk/p), j,k = 1..h, exact long-double args */
static double *rp_trig(int p, int h, int want_sin)
{
    double *t = aligned_alloc(64, ((size_t)h * h * sizeof(double) + 63) & ~(size_t)63);
    if (!t) return NULL;
    for (int k = 1; k <= h; ++k)
        for (int j = 1; j <= h; ++j) {
            long m = ((long)j * k) % p;
            long double th = 2.0L * PIL * (long double)m / (long double)p;
            t[(size_t)(k - 1) * h + (j - 1)] =
                want_sin ? (double)sinl(th) : (double)cosl(th);
        }
    return t;
}

/* gen_r14: duplicated-pair trig for the p=11 DENSE Z-ROW kernel (the r31_trig_dup
 * layout at p=11): row j-1 holds (w_{j,0}, w_{j,0}, ..., w_{j,5}, w_{j,5}) = 12
 * doubles + 4 zero pad = 16 (rows 128B-aligned).  k=0 is included (cos row 1.0,
 * sin row 0.0) so X_0 accumulates in lane pair 0 for free. */
static double *rpd11_trig_dup(int want_sin)
{
    double *t = aligned_alloc(64, 5 * 16 * sizeof(double));
    if (!t) return NULL;
    memset(t, 0, 5 * 16 * sizeof(double));
    for (int j = 1; j <= 5; ++j)
        for (int k = 0; k <= 5; ++k) {
            long m = ((long)j * k) % 11;
            long double th = 2.0L * PIL * (long double)m / 11.0L;
            double w = want_sin ? (double)sinl(th) : (double)cosl(th);
            t[(size_t)(j - 1) * 16 + 2 * k]     = w;
            t[(size_t)(j - 1) * 16 + 2 * k + 1] = w;
        }
    return t;
}

/* ---------------- outer-C3 Rader tables for any prime with h = 3m, h odd,
 * gcd(3,m) = 1 (p in {43, 67, 79, 103}; 31 keeps its tuned compile-time path,
 * 7 is m = 1 = already dense, 19/127 have 3 | m and cannot CRT-split).
 * This is the r31 construction generalized to runtime tables -- the recipe
 * below regenerates the R31_JS/JDP/JDM/KP/KM constants exactly when run at
 * p = 31 (verified offline):
 *   a_r = g^r mod p (g a primitive root; quotient group Z_p* mod {+-1} = C_h),
 *   slot n = m*i + j <-> group index u(n) = (e3*i + em*j) mod h (CRT units),
 *   E fill (correlation -> convolution: data reversed, kernel forward):
 *     r_n = -u(n) mod h, js[n] = fold(a_{r_n}) = min(a, p-a),
 *   O fill: d''_r = (-1)^r (x[a_r] - x[p-a_r]) -- the negacyclic->cyclic
 *     twist for odd h baked into the jdp/jdm swap at odd r_n,
 *   outputs at t = u(n): X_{a_t} = x0 + E_t - i*(-1)^t O'_t, the (-1)^t baked
 *     into the kp/km pair order,
 *   kernels CC[r] = cos(2pi a_r/p), SN[r] = (-1)^r sin(2pi a_r/p), Winograd-C3
 *     kernel-transformed ((H0+H1+H2)/3 etc.) and each block STRETCHED to
 *     2m-1 doubles so the dense-Cm product indexes kb[m-1+j-i] mod-free. */

static int rp3_prim_root(int p)
{
    for (int g = 2; g < p; ++g) {   /* order test directly: p <= 127, trivial */
        long x = g;
        int ord = 1;
        while (x != 1) { x = x * g % p; ++ord; }
        if (ord == p - 1) return g;
    }
    return 0;
}

static int rp3_build(fft3d_plan *pl)
{
    const int p = pl->L, h = pl->h, m = h / 3, K = 2 * m - 1;
    const int g = rp3_prim_root(p);
    if (!g) return 0;
    pl->j3 = malloc((size_t)7 * h * sizeof(int));
    /* +7 zeroed tail slots: the blocked conv's 8-wide j-tiles may overread
     * the LAST block by up to 7 (inner blocks overread into the next block's
     * constants -- finite values that land in never-stored tile lanes) */
    pl->k3e = aligned_alloc(64, ((size_t)(4 * K + 7) * sizeof(double) + 63) & ~(size_t)63);
    pl->k3o = aligned_alloc(64, ((size_t)(4 * K + 7) * sizeof(double) + 63) & ~(size_t)63);
    int a[64], u[64];
    if (!pl->j3 || !pl->k3e || !pl->k3o || h > 64) return 0;
    memset(pl->k3e, 0, (size_t)(4 * K + 7) * sizeof(double));
    memset(pl->k3o, 0, (size_t)(4 * K + 7) * sizeof(double));
    long x = 1;
    for (int r = 0; r < h; ++r) { a[r] = (int)x; x = x * g % p; }
    const int e3 = ((m % 3) == 1) ? m : 2 * m;   /* == 1 mod 3, == 0 mod m */
    int s3 = 1;
    while ((3 * s3) % m != 1) ++s3;              /* em == 0 mod 3, == 1 mod m */
    const int em = 3 * s3;
    int *js = pl->j3, *js2 = js + h, *jdp = js2 + h, *jdm = jdp + h,
        *kp = jdm + h, *km = kp + h, *jv = km + h;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < m; ++j) {
            const int n = m * i + j;
            u[n] = (e3 * i + em * j) % h;
            const int rn = (h - u[n]) % h;
            const int ar = a[rn], at = a[u[n]];
            js[n] = ar <= h ? ar : p - ar;
            js2[n] = p - js[n];
            if (rn & 1) { jdp[n] = p - ar; jdm[n] = ar; }
            else        { jdp[n] = ar;     jdm[n] = p - ar; }
            /* signed V index for the one-pass fold (gen_r6): the kernel
             * builds V_j = x_j - x_{p-j} and a negated mirror V_{h+j};
             * x_{jdp} - x_{jdm} == +-V_{fold} exactly (-(a-b) == b-a). */
            jv[n] = jdp[n] <= h ? jdp[n] - 1 : h + jdm[n] - 1;
            if (u[n] & 1) { kp[n] = p - at; km[n] = at; }
            else          { kp[n] = at;     km[n] = p - at; }
        }
    long double CC[64], SN[64], H[3][22], B[4][22];
    for (int r = 0; r < h; ++r) {
        long double th = 2.0L * PIL * (long double)a[r] / (long double)p;
        CC[r] = cosl(th);
        SN[r] = (r & 1) ? -sinl(th) : sinl(th);
    }
    for (int e = 0; e < 2; ++e) {
        const long double *Kk = e ? SN : CC;
        double *out = e ? pl->k3o : pl->k3e;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < m; ++j)
                H[i][j] = Kk[u[m * i + j]];
        for (int j = 0; j < m; ++j) {
            B[0][j] = (H[0][j] + H[1][j] + H[2][j]) / 3.0L;
            B[1][j] = (H[0][j] - H[2][j]) / 3.0L;
            B[2][j] = (H[1][j] - H[2][j]) / 3.0L;
            B[3][j] = (H[0][j] - H[1][j]) / 3.0L;
        }
        for (int q = 0; q < 4; ++q)
            for (int v = 0; v < K; ++v)
                out[(size_t)q * K + v] =
                    (double)B[q][((v - (m - 1)) % m + m) % m];
    }
    return 1;
}

/* ---------------- even-h split Rader tables (gen_r6), any prime with h = 2m ----
 * (p == 1 mod 4: 13 17 29 37 41 53 61 73 89 97 101 109 113).  The E (cos)
 * system is a cyclic-h correlation for ANY h; the O (sin) system is genuinely
 * NEGACYCLIC-h when h is even (no odd-h +-1 twist exists).  Two splits, both
 * with dense length-m block products (the settled FMA doctrine):
 *   E: cyclic-2m -> CRT z^{2m}-1 = (z^m-1)(z^m+1): one dense cyclic-m product
 *      (folded data u_j + u_{m+j}, kernel (CC_j + CC_{m+j})/2) + one dense
 *      negacyclic-m product (u_j - u_{m+j}, (CC_j - CC_{m+j})/2); reconstruct
 *      E_j = Yc_j + Yn_j, E_{m+j} = Yc_j - Yn_j (the 1/2 lives in the kernels).
 *   O: negacyclic-2m -> y = z^2, y^m = -1: 3-product Karatsuba over
 *      negacyclic-m blocks (A0 = even slots, A1 = odd slots): M0 = A0*B0,
 *      M1 = A1*B1, M2 = (A0+A1)*(B0+B1); O_{2j} = M0_j + (y M1)_j (the y-shift
 *      is index renaming + one sign at j=0), O_{2j+1} = M2_j - M0_j - M1_j.
 * Total 5m^2 block FMA vs the dense engine's 2h^2 = 8m^2 (-37.5%).
 * Correlation -> convolution by data reversal; the negacyclic w-reversal sign
 * (w'_n = -w_{h-n}, n >= 1) is baked into the jdp/jdm swap.  Kernels are
 * stretched to 2m-1 doubles (negacyclic: negated on wrap) so the block
 * product indexes kb[m-1+j-i] mod-free -- RP3_CONV reused verbatim.
 * Verified against numpy in build/tryout/gen_rader/r6_proto.py (worst rel L2
 * 6.3e-16 over all 13 primes) before any C existed; create()'s self-check
 * gates the shipped tables at 1e-13 as always. */

static int rp2_build(fft3d_plan *pl)
{
    const int p = pl->L, h = pl->h, m = h / 2, K = 2 * m - 1;
    const int g = rp3_prim_root(p);
    if (!g || h > 56) return 0;
    pl->j2 = malloc((size_t)6 * h * sizeof(int));
    pl->k2 = aligned_alloc(64, ((size_t)5 * (K + 7) * sizeof(double) + 63) & ~(size_t)63);
    if (!pl->j2 || !pl->k2) return 0;
    int a[64];
    long x = 1;
    for (int r = 0; r < h; ++r) { a[r] = (int)x; x = x * g % p; }
    /* index tables against the ONE-PASS fold (gen_r6 kernel form): the kernel
     * folds U_j = x_j + x_{p-j}, V_j = x_j - x_{p-j} (j = 1..h) once into
     * stack arrays with a negated mirror V_{h+j} = -V_j, so every conv slot
     * is ONE stack load: js[n] = U index (1..h, reversal baked in), jv[n] =
     * signed V index (0..2h-1; the negacyclic reversal sign w'_n = -w_{h-n}
     * picks the mirror half).  -(a-b) == b-a exactly, so this is
     * bit-identical to the two-row-load form. */
    int *js = pl->j2, *jv = js + h, *kp = jv + h, *km = kp + h,
        *st2 = km + h;
    for (int n = 0; n < h; ++n) {
        const int q = a[(h - n) % h];        /* data reversal: slot n <- g^{-n} */
        js[n] = q <= h ? q : p - q;
        int dp, dm;
        if (n == 0) { dp = q;     dm = p - q; }
        else        { dp = p - q; dm = q;     }           /* w'_n = -w_{h-n} */
        jv[n] = dp <= h ? dp - 1 : h + dm - 1;
        kp[n] = a[n];
        km[n] = p - a[n];
        /* store-order table (gen_r8, -DRP2_STORD): output row min(a,p-a) in
         * natural order <- source slot n; bit 16 set when that row is the
         * km (X_{p-k}) side, which flips the +-SG combine */
        if (a[n] <= h) st2[a[n] - 1] = n;
        else           st2[p - a[n] - 1] = n | (1 << 16);
    }
    long double CC[64], SS[64];
    for (int r = 0; r < h; ++r) {
        long double th = 2.0L * PIL * (long double)a[r] / (long double)p;
        CC[r] = cosl(th);
        SS[r] = sinl(th);
    }
    const int KS = K + 7;   /* block stride: 7 zeroed pad slots so the blocked
                             * conv's 8-wide j-tiles may overread (x*0 into
                             * DST slots >= m that are never stored) */
    double *kc = pl->k2, *kn = kc + KS, *kb0 = kn + KS, *kb1 = kb0 + KS,
           *kb2 = kb1 + KS;
    memset(pl->k2, 0, (size_t)5 * KS * sizeof(double));
    long double Bc[32], Bn[32], B0[32], B1[32], B2[32];
    for (int j = 0; j < m; ++j) {
        Bc[j] = (CC[j] + CC[m + j]) / 2.0L;
        Bn[j] = (CC[j] - CC[m + j]) / 2.0L;
        B0[j] = SS[2 * j];
        B1[j] = SS[2 * j + 1];
        B2[j] = B0[j] + B1[j];
    }
    for (int v = 0; v < K; ++v) {
        const int d = v - (m - 1);           /* block-product shift j - i */
        const int dc = ((d % m) + m) % m;
        const long double sg = d < 0 ? -1.0L : 1.0L;    /* negacyclic wrap */
        kc[v]  = (double)Bc[dc];
        kn[v]  = (double)(sg * Bn[dc]);
        kb0[v] = (double)(sg * B0[dc]);
        kb1[v] = (double)(sg * B1[dc]);
        kb2[v] = (double)(sg * B2[dc]);
    }
    return 1;
}

#ifdef __AVX512F__

/* ---------------- the Winograd C3 x dense-C5 cyclic-15 convolution ----------------
 * S[15] in CRT slot order -> Y[15] same order; kt = 20 transformed kernel doubles.
 * If esum != NULL, also emits sum of the E block (= sum of all 15 inputs). */

#define R31_C5(DST, SRC, KB) do {                                              \
    __m512d k0 = _mm512_set1_pd((KB)[0]), k1 = _mm512_set1_pd((KB)[1]);        \
    __m512d k2 = _mm512_set1_pd((KB)[2]), k3 = _mm512_set1_pd((KB)[3]);        \
    __m512d k4 = _mm512_set1_pd((KB)[4]);                                      \
    DST[0] = _mm512_mul_pd(SRC[0], k0);                                        \
    DST[1] = _mm512_mul_pd(SRC[0], k1);                                        \
    DST[2] = _mm512_mul_pd(SRC[0], k2);                                        \
    DST[3] = _mm512_mul_pd(SRC[0], k3);                                        \
    DST[4] = _mm512_mul_pd(SRC[0], k4);                                        \
    DST[0] = _mm512_fmadd_pd(SRC[1], k4, DST[0]);                              \
    DST[1] = _mm512_fmadd_pd(SRC[1], k0, DST[1]);                              \
    DST[2] = _mm512_fmadd_pd(SRC[1], k1, DST[2]);                              \
    DST[3] = _mm512_fmadd_pd(SRC[1], k2, DST[3]);                              \
    DST[4] = _mm512_fmadd_pd(SRC[1], k3, DST[4]);                              \
    DST[0] = _mm512_fmadd_pd(SRC[2], k3, DST[0]);                              \
    DST[1] = _mm512_fmadd_pd(SRC[2], k4, DST[1]);                              \
    DST[2] = _mm512_fmadd_pd(SRC[2], k0, DST[2]);                              \
    DST[3] = _mm512_fmadd_pd(SRC[2], k1, DST[3]);                              \
    DST[4] = _mm512_fmadd_pd(SRC[2], k2, DST[4]);                              \
    DST[0] = _mm512_fmadd_pd(SRC[3], k2, DST[0]);                              \
    DST[1] = _mm512_fmadd_pd(SRC[3], k3, DST[1]);                              \
    DST[2] = _mm512_fmadd_pd(SRC[3], k4, DST[2]);                              \
    DST[3] = _mm512_fmadd_pd(SRC[3], k0, DST[3]);                              \
    DST[4] = _mm512_fmadd_pd(SRC[3], k1, DST[4]);                              \
    DST[0] = _mm512_fmadd_pd(SRC[4], k1, DST[0]);                              \
    DST[1] = _mm512_fmadd_pd(SRC[4], k2, DST[1]);                              \
    DST[2] = _mm512_fmadd_pd(SRC[4], k3, DST[2]);                              \
    DST[3] = _mm512_fmadd_pd(SRC[4], k4, DST[3]);                              \
    DST[4] = _mm512_fmadd_pd(SRC[4], k0, DST[4]);                              \
} while (0)

static inline __attribute__((always_inline))
void r31_wino15(const __m512d *S, const double *kt, __m512d *Y, __m512d *esum)
{
    __m512d E[5], A[5], B[5], C[5], M0[5], M1[5], M2[5], M3[5];
    for (int j = 0; j < 5; ++j) {
        E[j] = _mm512_add_pd(_mm512_add_pd(S[j], S[5 + j]), S[10 + j]);
        A[j] = _mm512_sub_pd(S[j], S[10 + j]);
        B[j] = _mm512_sub_pd(S[5 + j], S[10 + j]);
        C[j] = _mm512_sub_pd(S[j], S[5 + j]);
    }
    if (esum)
        *esum = _mm512_add_pd(_mm512_add_pd(_mm512_add_pd(E[0], E[1]),
                                            _mm512_add_pd(E[2], E[3])), E[4]);
    R31_C5(M0, E, kt);
    R31_C5(M1, A, kt + 5);
    R31_C5(M2, B, kt + 10);
    R31_C5(M3, C, kt + 15);
    const __m512d TWO = _mm512_set1_pd(2.0);
    for (int j = 0; j < 5; ++j) {
        __m512d t01 = _mm512_add_pd(M0[j], M1[j]);
        Y[j]      = _mm512_fnmadd_pd(M2[j], TWO, _mm512_add_pd(t01, M3[j]));
        Y[5 + j]  = _mm512_fnmadd_pd(M3[j], TWO, _mm512_add_pd(t01, M2[j]));
        __m512d t23 = _mm512_add_pd(M2[j], M3[j]);
        Y[10 + j] = _mm512_fnmadd_pd(M1[j], TWO, _mm512_add_pd(M0[j], t23));
    }
}

/* s6 map ladder pieces (arithmetic identical to map_volume; every ladder op is
 * elementwise, so where it runs cannot change per-point bits).  Pair form: one
 * vdivpd per two output vectors; single form for the lone X0 vector. */
/* hwdiv != 0: end the ladder in vdivpd (bit-identical to map_volume);
 * hwdiv == 0: divider-free rcp14 + 2 Newton (sub-ulp) -- the ice L23 r4
 * lesson: an eager STORE-side fused map must not end in vdivpd right before
 * the stores.  Compile-time constant at every call site. */
static inline __attribute__((always_inline)) __m512d r31_map_rec(__m512d m2,
                                                                 const int hwdiv)
{
    const __m512d ONE  = _mm512_set1_pd(1.0);
    const __m512d TH   = _mm512_set1_pd(1.5);
    const __m512d HALF = _mm512_set1_pd(0.5);
    const __m512d TINY = _mm512_set1_pd(1e-300);
    __m512d m2c = _mm512_max_pd(m2, TINY);
    __m512d r = _mm512_rsqrt14_pd(m2c);
    __m512d hm = _mm512_mul_pd(m2c, HALF);
    r = _mm512_mul_pd(r, _mm512_fnmadd_pd(_mm512_mul_pd(hm, r), r, TH));
    r = _mm512_mul_pd(r, _mm512_fnmadd_pd(_mm512_mul_pd(hm, r), r, TH));
    __m512d d = _mm512_fmadd_pd(m2c, r, ONE);                  /* 1 + |w| */
    if (hwdiv)
        return _mm512_div_pd(ONE, d);
    const __m512d TWO = _mm512_set1_pd(2.0);
    __m512d rec = _mm512_rcp14_pd(d);
    rec = _mm512_mul_pd(rec, _mm512_fnmadd_pd(d, rec, TWO));
    rec = _mm512_mul_pd(rec, _mm512_fnmadd_pd(d, rec, TWO));
    return rec;
}

#ifdef R31_FUSE_DIV
#define R31_FMDIV 1     /* store-side-fusion map style: raced, LOSES with div */
#else
#define R31_FMDIV 0
#endif

static inline __attribute__((always_inline))
void r31_map2(__m512d wa, __m512d wb, __m512d *oa, __m512d *ob, const int hwdiv)
{
    __m512d pa = _mm512_mul_pd(wa, wa), pb = _mm512_mul_pd(wb, wb);
    __m512d m2 = _mm512_add_pd(_mm512_unpacklo_pd(pa, pb),
                               _mm512_unpackhi_pd(pa, pb));
    __m512d rec = r31_map_rec(m2, hwdiv);
    *oa = _mm512_mul_pd(wa, _mm512_unpacklo_pd(rec, rec));
    *ob = _mm512_mul_pd(wb, _mm512_unpackhi_pd(rec, rec));
}

static inline __attribute__((always_inline)) __m512d r31_map1(__m512d w,
                                                              const int hwdiv)
{
    __m512d p = _mm512_mul_pd(w, w);
    __m512d m2 = _mm512_add_pd(p, _mm512_permute_pd(p, 0x55));
    return _mm512_mul_pd(w, r31_map_rec(m2, hwdiv));
}

/* one column chunk (up to 4 complex = 8 doubles wide) of a 31 x inner pass.
 * sx/dx already offset to the chunk; rs = row stride in doubles.  In-place safe:
 * within the chunk every load of a row precedes every store to it (E sweep writes
 * only row 0, which the O sweep never reads).  NO restrict here -- the y-pass runs
 * dst == src and the compiler must keep the load/store order.
 * mapc == NULL: plain FFT stores.  mapc != NULL: chain mode -- every output is
 * mapped ((X+c)/(1+|X+c|)) at the store, c chunk at mapc, same row strides. */
static inline __attribute__((always_inline))
void r31_chunk(const double *sx, double *dx, const double *mapc,
               const ptrdiff_t rs, const ptrdiff_t ds, int full,
               __mmask8 msk, const double *ke, const double *ko)
{
    /* gen_r14: ds = DEST row stride in doubles (loads stay on rs; mapc rows
     * are output-row-indexed so they ride ds).  Every call site passes
     * compile-time constants and all pre-r14 sites pass ds == rs, so their
     * codegen is unchanged (cmp-verified).  ds != rs only in the padded
     * execute()'s final y pass (padded loads, flat stores). */
    const __m512d SG = _mm512_setr_pd(1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0);
    __m512d S[15], Y[15], T[15];
#ifdef R31_ONEBODY
    (void)full;   /* single always-masked body: halves the hot code footprint */
#define R31_LD(px, off) _mm512_maskz_loadu_pd(msk, (px) + (off))
#define R31_ST(off, v)  _mm512_mask_storeu_pd(dx + (off), msk, v)
#else
#define R31_LD(px, off) (full ? _mm512_loadu_pd((px) + (off)) \
                              : _mm512_maskz_loadu_pd(msk, (px) + (off)))
#define R31_ST(off, v) do { if (full) _mm512_storeu_pd(dx + (off), v); \
                            else _mm512_mask_storeu_pd(dx + (off), msk, v); } while (0)
#endif
    __m512d x0 = R31_LD(sx, 0);
    for (int n = 0; n < 15; ++n) {
        const ptrdiff_t j = R31_JS[n];
        S[n] = _mm512_add_pd(R31_LD(sx, j * rs), R31_LD(sx, (31 - j) * rs));
    }
    __m512d esum;
    r31_wino15(S, ke, Y, &esum);
    __m512d X0 = _mm512_add_pd(x0, esum);
    if (mapc)
        X0 = r31_map1(_mm512_add_pd(X0, R31_LD(mapc, 0)), R31_FMDIV);
    R31_ST(0, X0);
    for (int n = 0; n < 15; ++n) T[n] = _mm512_add_pd(x0, Y[n]);
    for (int n = 0; n < 15; ++n)
        S[n] = _mm512_sub_pd(R31_LD(sx, (ptrdiff_t)R31_JDP[n] * rs),
                             R31_LD(sx, (ptrdiff_t)R31_JDM[n] * rs));
    r31_wino15(S, ko, Y, NULL);
    for (int n = 0; n < 15; ++n) {
        __m512d o = _mm512_permute_pd(Y[n], 0x55);              /* swap re/im */
        __m512d xp = _mm512_fmadd_pd(o, SG, T[n]);
        __m512d xm = _mm512_fnmadd_pd(o, SG, T[n]);
        if (mapc) {
            xp = _mm512_add_pd(xp, R31_LD(mapc, (ptrdiff_t)R31_KP[n] * ds));
            xm = _mm512_add_pd(xm, R31_LD(mapc, (ptrdiff_t)R31_KM[n] * ds));
            r31_map2(xp, xm, &xp, &xm, R31_FMDIV);
        }
        R31_ST((ptrdiff_t)R31_KP[n] * ds, xp);
        R31_ST((ptrdiff_t)R31_KM[n] * ds, xm);
    }
#undef R31_LD
#undef R31_ST
}

/* contract the slowest axis of a (31 x ncols) complex block whose rows sit at
 * `pitch` complex apart (pitch == ncols: flat; pitch > ncols: padded rows);
 * dst may equal src when c == NULL (plain).  c != NULL: map-fused stores
 * (dst must be distinct; c rows at the same pitch).
 * pf != 0: software-prefetch each row's line `pf` bytes ahead of the current
 * chunk's loads (gen_layout gen_r4: the fold's ~62 row streams outrun what
 * the DCU prefetcher tracks; T0 pays where the streams miss L1 -> L2). */
static inline __attribute__((always_inline))
void r31_pass_core(const cplx *src, cplx *dst, const cplx *c,
                   const ptrdiff_t ncols, const ptrdiff_t pitch,
                   const ptrdiff_t dpitch,
                   const int pf, const double *ke, const double *ko)
{
    const ptrdiff_t rs = 2 * pitch;
    const ptrdiff_t ds = 2 * dpitch;
    const ptrdiff_t nd = 2 * ncols;
    const double *sx = (const double *)src;
    const double *cx = (const double *)c;
    double *dx = (double *)dst;
#ifdef R31_ONEBODY
    for (ptrdiff_t d = 0; d < nd; d += 8) {
        __mmask8 msk = (nd - d >= 8) ? (__mmask8)0xFF
                                     : (__mmask8)((1u << (nd - d)) - 1);
        r31_chunk(sx + d, dx + d, cx ? cx + d : NULL, rs, ds, 0, msk, ke, ko);
    }
    (void)pf;
#else
    ptrdiff_t d = 0;
    for (; d + 8 <= nd; d += 8) {
        if (pf)
            for (int j = 0; j < 31; ++j)
                _mm_prefetch((const char *)(sx + (ptrdiff_t)j * rs + d) + pf,
                             _MM_HINT_T0);
        r31_chunk(sx + d, dx + d, cx ? cx + d : NULL, rs, ds, 1, (__mmask8)0xFF, ke, ko);
    }
    if (d < nd)
        r31_chunk(sx + d, dx + d, cx ? cx + d : NULL, rs, ds, 0,
                  (__mmask8)((1u << (nd - d)) - 1), ke, ko);
#endif
}

/* -DR31_SCHEDP: pre-RA pressure scheduling on the chunk instantiators only
 * (gen_batchlane r2 / gen_powp r1: pays on spill-bound bodies, as attribute
 * not global flags; this kernel spills ~29 moves/chunk). */
#ifdef R31_SCHEDP
#define R31_SCHED_ATTR __attribute__((optimize("schedule-insns", "sched-pressure")))
#else
#define R31_SCHED_ATTR
#endif

/* x-pass prefetch distance in bytes (0 disables; 128 = 2 chunks ahead).
 * Raced gen_r4 same-core: +0.7% LOSS at the graded cell (the 31 extra
 * port-2/3 uops/chunk cost more than the L2-hit latency they hide -- the
 * state is L2-resident, unlike gen_layout's DRAM-resident demo where the
 * same recipe won).  Default OFF; knob kept for the cross-arch race. */
#ifndef R31_PFX
#define R31_PFX 0
#endif

/* custody z step: quads per plane (8 = all rows via quads, the r4 form) */
#ifndef R31_ZMIX
#define R31_ZMIX 8
#endif

static R31_SCHED_ATTR
void r31_pass_x(const cplx *s, cplx *d, const double *ke, const double *ko)
{ r31_pass_core(s, d, NULL, 31 * 31, 31 * 31, 31 * 31, R31_PFX, ke, ko); }

/* x pass over the padded state: 992 columns per plane (31 pad columns of
 * zeros ride along -- tail-free), planes R31_PP apart */
static R31_SCHED_ATTR
void r31_pass_xp(cplx *st, const double *ke, const double *ko)
{ r31_pass_core(st, st, NULL, 31 * R31_ZP, R31_PP, R31_PP, R31_PFX, ke, ko); }

static R31_SCHED_ATTR
void r31_pass_y(const cplx *s, cplx *d, const double *ke, const double *ko)
{ r31_pass_core(s, d, NULL, 31, 31, 31, 0, ke, ko); }

/* y pass on one padded plane: rows at R31_ZP, 32 columns -> 8 full chunks */
static R31_SCHED_ATTR
void r31_pass_yp(cplx *pl, const double *ke, const double *ko)
{ r31_pass_core(pl, pl, NULL, R31_ZP, R31_ZP, R31_ZP, 0, ke, ko); }

/* gen_r14: y pass reading one PADDED plane and storing the FLAT output plane
 * directly (rows loaded at R31_ZP, stored at 31) -- the padded execute()'s
 * final pass, deleting the copy-out sweep the chain's last step pays.  Only
 * the 31 live columns run (7 full chunks + 6-double masked tail); the pad
 * column is never stored, so dst gets exactly the 961 output points. */
static R31_SCHED_ATTR __attribute__((unused))
void r31_pass_yf(const cplx *s, cplx *d, const double *ke, const double *ko)
{ r31_pass_core(s, d, NULL, 31, R31_ZP, 31, 0, ke, ko); }

/* y pass with the map fused at every store: src = x-pass output plane (t1),
 * dst = state plane, c = map-constant plane */
static __attribute__((unused))
void r31_pass_ym(const cplx *s, cplx *d, const cplx *c,
                 const double *ke, const double *ko)
{ r31_pass_core(s, d, c, 31, 31, 31, 0, ke, ko); }

/* ---------------- z-axis pass, Rader form: 4 rows via 4x4-complex transposes ----
 * Four contiguous rows are transposed (8 vshuff64x2 per 4x4-complex tile) into a
 * stack array where element j of the 4 pencils is one zmm at stride 8 doubles,
 * the SAME r31_chunk kernel runs on it (all offsets compile-time), and the result
 * transposes back.  ~110 arith/pencil + 32 shuffles/pencil vs the dense row-GEMM's
 * ~240 FMA/pencil.  In-place safe: all loads of the 4 rows precede all stores. */
static inline __attribute__((always_inline))
void r31_tp4(const __m512d a, const __m512d b, const __m512d c, const __m512d d,
             __m512d *o0, __m512d *o1, __m512d *o2, __m512d *o3)
{
    __m512d t0 = _mm512_shuffle_f64x2(a, b, 0x44);
    __m512d t1 = _mm512_shuffle_f64x2(a, b, 0xEE);
    __m512d t2 = _mm512_shuffle_f64x2(c, d, 0x44);
    __m512d t3 = _mm512_shuffle_f64x2(c, d, 0xEE);
    *o0 = _mm512_shuffle_f64x2(t0, t2, 0x88);
    *o1 = _mm512_shuffle_f64x2(t0, t2, 0xDD);
    *o2 = _mm512_shuffle_f64x2(t1, t3, 0x88);
    *o3 = _mm512_shuffle_f64x2(t1, t3, 0xDD);
}

/* srd/drd: source/dest row strides in DOUBLES (62 flat, 64 padded); always
 * called with compile-time constants so each wrapper folds its offsets. */
static inline __attribute__((always_inline))
void r31_zquad_core(const cplx *src, cplx *dst, const ptrdiff_t srd,
                    const ptrdiff_t drd, const double *ke, const double *ko)
{
    __attribute__((aligned(64))) double xt[32 * 8], yt[32 * 8];
    const double *x = (const double *)src;
    double *y = (double *)dst;
    for (int t = 0; t < 8; ++t) {           /* transpose in: tile t = elements 4t..4t+3 */
        __m512d r0, r1, r2, r3;
        if (t < 7) {
            r0 = _mm512_loadu_pd(x + 8 * t);
            r1 = _mm512_loadu_pd(x + srd + 8 * t);
            r2 = _mm512_loadu_pd(x + 2 * srd + 8 * t);
            r3 = _mm512_loadu_pd(x + 3 * srd + 8 * t);
        } else {                            /* elements 28..30 only */
            r0 = _mm512_maskz_loadu_pd(0x3F, x + 56);
            r1 = _mm512_maskz_loadu_pd(0x3F, x + srd + 56);
            r2 = _mm512_maskz_loadu_pd(0x3F, x + 2 * srd + 56);
            r3 = _mm512_maskz_loadu_pd(0x3F, x + 3 * srd + 56);
        }
        __m512d o0, o1, o2, o3;
        r31_tp4(r0, r1, r2, r3, &o0, &o1, &o2, &o3);
        _mm512_store_pd(xt + (4 * t) * 8, o0);
        _mm512_store_pd(xt + (4 * t + 1) * 8, o1);
        _mm512_store_pd(xt + (4 * t + 2) * 8, o2);
        _mm512_store_pd(xt + (4 * t + 3) * 8, o3);
    }
    r31_chunk(xt, yt, NULL, 8, 8, 1, (__mmask8)0xFF, ke, ko);
    for (int t = 0; t < 8; ++t) {           /* transpose out */
        __m512d o0, o1, o2, o3;
        r31_tp4(_mm512_load_pd(yt + (4 * t) * 8),
                _mm512_load_pd(yt + (4 * t + 1) * 8),
                _mm512_load_pd(yt + (4 * t + 2) * 8),
                (t < 7) ? _mm512_load_pd(yt + (4 * t + 3) * 8) : _mm512_setzero_pd(),
                &o0, &o1, &o2, &o3);
        if (t < 7) {
            _mm512_storeu_pd(y + 8 * t, o0);
            _mm512_storeu_pd(y + drd + 8 * t, o1);
            _mm512_storeu_pd(y + 2 * drd + 8 * t, o2);
            _mm512_storeu_pd(y + 3 * drd + 8 * t, o3);
        } else {
            _mm512_mask_storeu_pd(y + 56, 0x3F, o0);
            _mm512_mask_storeu_pd(y + drd + 56, 0x3F, o1);
            _mm512_mask_storeu_pd(y + 2 * drd + 56, 0x3F, o2);
            _mm512_mask_storeu_pd(y + 3 * drd + 56, 0x3F, o3);
        }
    }
}

static void r31_zquad(const cplx *src, cplx *dst, const double *ke, const double *ko)
{ r31_zquad_core(src, dst, 62, 62, ke, ko); }           /* flat, execute() */

static void r31_zquad_pp(const cplx *src, cplx *dst, const double *ke, const double *ko)
{ r31_zquad_core(src, dst, 2 * R31_ZP, 2 * R31_ZP, ke, ko); }  /* padded in place */

static void r31_zquad_fp(const cplx *src, cplx *dst, const double *ke, const double *ko)
{ r31_zquad_core(src, dst, 62, 2 * R31_ZP, ke, ko); }   /* step 0: flat x0 -> padded */

/* padded in-place z quad with the s6 MAP APPLIED AT THE TRANSPOSE-IN LOADS:
 * computes zquad(map(x + c)) for 4 rows, replacing map_volume + r31_zquad_pp
 * on those rows.  The map is elementwise and every ladder op is lanewise, so
 * per-element bits are IDENTICAL to the separate sweep (only which lanes share
 * a zmm changes); pad lanes/rows stay zero (maskz load -> w=0 -> map 0).
 * Unlike the four store-side map fusions this panel measured and killed
 * (r31 y-stores r1, dense_prime r2, rp y r3, batchlane epilogue r4), this is
 * LOAD-side fusion into a register-light transpose phase: ~10 live zmm before
 * the ladder, not ~30. */
static __attribute__((unused))
void r31_zquad_mp(const cplx *src, cplx *dst, const cplx *c,
                  const double *ke, const double *ko)
{
    __attribute__((aligned(64))) double xt[32 * 8], yt[32 * 8];
    const ptrdiff_t rd = 2 * R31_ZP;
    const double *x = (const double *)src;
    const double *cx = (const double *)c;
    double *y = (double *)dst;
    for (int t = 0; t < 8; ++t) {
        __m512d r0, r1, r2, r3;
        if (t < 7) {
            r0 = _mm512_add_pd(_mm512_loadu_pd(x + 8 * t),
                               _mm512_loadu_pd(cx + 8 * t));
            r1 = _mm512_add_pd(_mm512_loadu_pd(x + rd + 8 * t),
                               _mm512_loadu_pd(cx + rd + 8 * t));
            r2 = _mm512_add_pd(_mm512_loadu_pd(x + 2 * rd + 8 * t),
                               _mm512_loadu_pd(cx + 2 * rd + 8 * t));
            r3 = _mm512_add_pd(_mm512_loadu_pd(x + 3 * rd + 8 * t),
                               _mm512_loadu_pd(cx + 3 * rd + 8 * t));
        } else {                              /* elements 28..30 + pad col */
            r0 = _mm512_add_pd(_mm512_maskz_loadu_pd(0x3F, x + 56),
                               _mm512_maskz_loadu_pd(0x3F, cx + 56));
            r1 = _mm512_add_pd(_mm512_maskz_loadu_pd(0x3F, x + rd + 56),
                               _mm512_maskz_loadu_pd(0x3F, cx + rd + 56));
            r2 = _mm512_add_pd(_mm512_maskz_loadu_pd(0x3F, x + 2 * rd + 56),
                               _mm512_maskz_loadu_pd(0x3F, cx + 2 * rd + 56));
            r3 = _mm512_add_pd(_mm512_maskz_loadu_pd(0x3F, x + 3 * rd + 56),
                               _mm512_maskz_loadu_pd(0x3F, cx + 3 * rd + 56));
        }
        r31_map2(r0, r1, &r0, &r1, 1);      /* vdivpd: bit-identical to the
                                             * separate map_volume sweep */
        r31_map2(r2, r3, &r2, &r3, 1);
        __m512d o0, o1, o2, o3;
        r31_tp4(r0, r1, r2, r3, &o0, &o1, &o2, &o3);
        _mm512_store_pd(xt + (4 * t) * 8, o0);
        _mm512_store_pd(xt + (4 * t + 1) * 8, o1);
        _mm512_store_pd(xt + (4 * t + 2) * 8, o2);
        _mm512_store_pd(xt + (4 * t + 3) * 8, o3);
    }
    r31_chunk(xt, yt, NULL, 8, 8, 1, (__mmask8)0xFF, ke, ko);
    for (int t = 0; t < 8; ++t) {
        __m512d o0, o1, o2, o3;
        r31_tp4(_mm512_load_pd(yt + (4 * t) * 8),
                _mm512_load_pd(yt + (4 * t + 1) * 8),
                _mm512_load_pd(yt + (4 * t + 2) * 8),
                (t < 7) ? _mm512_load_pd(yt + (4 * t + 3) * 8) : _mm512_setzero_pd(),
                &o0, &o1, &o2, &o3);
        if (t < 7) {
            _mm512_storeu_pd(y + 8 * t, o0);
            _mm512_storeu_pd(y + rd + 8 * t, o1);
            _mm512_storeu_pd(y + 2 * rd + 8 * t, o2);
            _mm512_storeu_pd(y + 3 * rd + 8 * t, o3);
        } else {
            _mm512_mask_storeu_pd(y + 56, 0x3F, o0);
            _mm512_mask_storeu_pd(y + rd + 56, 0x3F, o1);
            _mm512_mask_storeu_pd(y + 2 * rd + 56, 0x3F, o2);
            _mm512_mask_storeu_pd(y + 3 * rd + 56, 0x3F, o3);
        }
    }
}

/* ---------------- z-axis pass: contiguous rows ----------------
 * ADOPTED VERBATIM from gen_dense_prime gen_r1 (their zpass31/zpass31_pair):
 * k dimension in 4 zmm per half-spectrum, u/v broadcast as 128-bit pairs,
 * rows in PAIRS sharing the 8 table loads per j. */
/* NO restrict on src/dst: the -DR31_ZMIX arm runs these in place (every load
 * of src precedes every store to dst -- safe only if the compiler keeps the
 * order, same rule as map_volume since r2) */
static void r31_zrow_pair(const cplx *src, cplx *dst,
                          ptrdiff_t sp, ptrdiff_t dp,
                          const double *restrict ctd, const double *restrict std)
{
    const __m512d SG = _mm512_setr_pd(1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0);
    const double *xA = (const double *)src;
    const double *xB = (const double *)(src + sp);
    double *yA = (double *)dst;
    double *yB = (double *)(dst + dp);

    __attribute__((aligned(64))) double ua[32], va[32], ub2[32], vb2[32];
    for (int j = 1; j <= 15; ++j) {
        __m128d a = _mm_loadu_pd(xA + 2 * j), b = _mm_loadu_pd(xA + 2 * (31 - j));
        _mm_store_pd(ua + 2 * j, _mm_add_pd(a, b));
        _mm_store_pd(va + 2 * j, _mm_sub_pd(a, b));
        __m128d c = _mm_loadu_pd(xB + 2 * j), e = _mm_loadu_pd(xB + 2 * (31 - j));
        _mm_store_pd(ub2 + 2 * j, _mm_add_pd(c, e));
        _mm_store_pd(vb2 + 2 * j, _mm_sub_pd(c, e));
    }
    __m512d xa0 = _mm512_broadcast_f64x2(_mm_loadu_pd(xA));
    __m512d xb0 = _mm512_broadcast_f64x2(_mm_loadu_pd(xB));
    __m512d CA0 = xa0, CA1 = xa0, CA2 = xa0, CA3 = xa0;
    __m512d CB0 = xb0, CB1 = xb0, CB2 = xb0, CB3 = xb0;
    __m512d SA0 = _mm512_setzero_pd(), SA1 = SA0, SA2 = SA0, SA3 = SA0;
    __m512d SB0 = SA0, SB1 = SA0, SB2 = SA0, SB3 = SA0;
    for (int j = 1; j <= 15; ++j) {
        const double *cr = ctd + (size_t)(j - 1) * 32;
        const double *sr = std + (size_t)(j - 1) * 32;
        __m512d c0 = _mm512_load_pd(cr + 0),  c1 = _mm512_load_pd(cr + 8);
        __m512d c2 = _mm512_load_pd(cr + 16), c3 = _mm512_load_pd(cr + 24);
        __m512d s0 = _mm512_load_pd(sr + 0),  s1 = _mm512_load_pd(sr + 8);
        __m512d s2 = _mm512_load_pd(sr + 16), s3 = _mm512_load_pd(sr + 24);
        __m512d uA = _mm512_broadcast_f64x2(_mm_load_pd(ua + 2 * j));
        __m512d vA = _mm512_broadcast_f64x2(_mm_load_pd(va + 2 * j));
        __m512d uB = _mm512_broadcast_f64x2(_mm_load_pd(ub2 + 2 * j));
        __m512d vB = _mm512_broadcast_f64x2(_mm_load_pd(vb2 + 2 * j));
        CA0 = _mm512_fmadd_pd(c0, uA, CA0); CA1 = _mm512_fmadd_pd(c1, uA, CA1);
        CA2 = _mm512_fmadd_pd(c2, uA, CA2); CA3 = _mm512_fmadd_pd(c3, uA, CA3);
        SA0 = _mm512_fmadd_pd(s0, vA, SA0); SA1 = _mm512_fmadd_pd(s1, vA, SA1);
        SA2 = _mm512_fmadd_pd(s2, vA, SA2); SA3 = _mm512_fmadd_pd(s3, vA, SA3);
        CB0 = _mm512_fmadd_pd(c0, uB, CB0); CB1 = _mm512_fmadd_pd(c1, uB, CB1);
        CB2 = _mm512_fmadd_pd(c2, uB, CB2); CB3 = _mm512_fmadd_pd(c3, uB, CB3);
        SB0 = _mm512_fmadd_pd(s0, vB, SB0); SB1 = _mm512_fmadd_pd(s1, vB, SB1);
        SB2 = _mm512_fmadd_pd(s2, vB, SB2); SB3 = _mm512_fmadd_pd(s3, vB, SB3);
    }
#define R31_ZSTORE(y, C0, C1, C2, C3, S0, S1, S2, S3) do {                     \
        __m512d T0 = _mm512_permute_pd(S0, 0x55);                              \
        __m512d T1 = _mm512_permute_pd(S1, 0x55);                              \
        __m512d T2 = _mm512_permute_pd(S2, 0x55);                              \
        __m512d T3 = _mm512_permute_pd(S3, 0x55);                              \
        _mm512_storeu_pd((y) + 0,  _mm512_fmadd_pd(T0, SG, C0));               \
        _mm512_storeu_pd((y) + 8,  _mm512_fmadd_pd(T1, SG, C1));               \
        _mm512_storeu_pd((y) + 16, _mm512_fmadd_pd(T2, SG, C2));               \
        _mm512_storeu_pd((y) + 24, _mm512_fmadd_pd(T3, SG, C3));               \
        __m512d h0 = _mm512_fnmadd_pd(T0, SG, C0);                             \
        __m512d h1 = _mm512_fnmadd_pd(T1, SG, C1);                             \
        __m512d h2 = _mm512_fnmadd_pd(T2, SG, C2);                             \
        __m512d h3 = _mm512_fnmadd_pd(T3, SG, C3);                             \
        _mm512_storeu_pd((y) + 32, _mm512_shuffle_f64x2(h3, h3, 0x1B));        \
        _mm512_storeu_pd((y) + 40, _mm512_shuffle_f64x2(h2, h2, 0x1B));        \
        _mm512_storeu_pd((y) + 48, _mm512_shuffle_f64x2(h1, h1, 0x1B));        \
        _mm512_mask_storeu_pd((y) + 56, 0x3F, _mm512_shuffle_f64x2(h0, h0, 0x1B)); \
    } while (0)
    R31_ZSTORE(yA, CA0, CA1, CA2, CA3, SA0, SA1, SA2, SA3);
    R31_ZSTORE(yB, CB0, CB1, CB2, CB3, SB0, SB1, SB2, SB3);
}

static void r31_zpass(const cplx *src, cplx *dst, size_t nrows,
                      ptrdiff_t sp, ptrdiff_t dp,
                      const double *restrict ctd, const double *restrict std)
{
    const __m512d SG = _mm512_setr_pd(1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0);
    size_t r = 0;
    for (; r + 2 <= nrows; r += 2)
        r31_zrow_pair(src + r * sp, dst + r * dp, sp, dp, ctd, std);
    for (; r < nrows; ++r) {
        const double *x = (const double *)(src + r * sp);
        double *y = (double *)(dst + r * dp);
        __attribute__((aligned(64))) double ub[32], vb[32];
        for (int j = 1; j <= 15; ++j) {
            __m128d a = _mm_loadu_pd(x + 2 * j);
            __m128d b = _mm_loadu_pd(x + 2 * (31 - j));
            _mm_store_pd(ub + 2 * j, _mm_add_pd(a, b));
            _mm_store_pd(vb + 2 * j, _mm_sub_pd(a, b));
        }
        __m512d x0 = _mm512_broadcast_f64x2(_mm_loadu_pd(x));
        __m512d C0 = x0, C1 = x0, C2 = x0, C3 = x0;
        __m512d S0 = _mm512_setzero_pd(), S1 = S0, S2 = S0, S3 = S0;
        for (int j = 1; j <= 15; ++j) {
            __m512d u = _mm512_broadcast_f64x2(_mm_load_pd(ub + 2 * j));
            __m512d v = _mm512_broadcast_f64x2(_mm_load_pd(vb + 2 * j));
            const double *cr = ctd + (size_t)(j - 1) * 32;
            const double *sr = std + (size_t)(j - 1) * 32;
            C0 = _mm512_fmadd_pd(_mm512_load_pd(cr + 0),  u, C0);
            C1 = _mm512_fmadd_pd(_mm512_load_pd(cr + 8),  u, C1);
            C2 = _mm512_fmadd_pd(_mm512_load_pd(cr + 16), u, C2);
            C3 = _mm512_fmadd_pd(_mm512_load_pd(cr + 24), u, C3);
            S0 = _mm512_fmadd_pd(_mm512_load_pd(sr + 0),  v, S0);
            S1 = _mm512_fmadd_pd(_mm512_load_pd(sr + 8),  v, S1);
            S2 = _mm512_fmadd_pd(_mm512_load_pd(sr + 16), v, S2);
            S3 = _mm512_fmadd_pd(_mm512_load_pd(sr + 24), v, S3);
        }
        __m512d T0 = _mm512_permute_pd(S0, 0x55);
        __m512d T1 = _mm512_permute_pd(S1, 0x55);
        __m512d T2 = _mm512_permute_pd(S2, 0x55);
        __m512d T3 = _mm512_permute_pd(S3, 0x55);
        _mm512_storeu_pd(y + 0,  _mm512_fmadd_pd(T0, SG, C0));
        _mm512_storeu_pd(y + 8,  _mm512_fmadd_pd(T1, SG, C1));
        _mm512_storeu_pd(y + 16, _mm512_fmadd_pd(T2, SG, C2));
        _mm512_storeu_pd(y + 24, _mm512_fmadd_pd(T3, SG, C3));
        __m512d h0 = _mm512_fnmadd_pd(T0, SG, C0);
        __m512d h1 = _mm512_fnmadd_pd(T1, SG, C1);
        __m512d h2 = _mm512_fnmadd_pd(T2, SG, C2);
        __m512d h3 = _mm512_fnmadd_pd(T3, SG, C3);
        _mm512_storeu_pd(y + 32, _mm512_shuffle_f64x2(h3, h3, 0x1B));
        _mm512_storeu_pd(y + 40, _mm512_shuffle_f64x2(h2, h2, 0x1B));
        _mm512_storeu_pd(y + 48, _mm512_shuffle_f64x2(h1, h1, 0x1B));
        _mm512_mask_storeu_pd(y + 56, 0x3F, _mm512_shuffle_f64x2(h0, h0, 0x1B));
    }
}
/* z-pass dispatcher: Rader-quad form by default, dense row-GEMM with -DR31_ZDENSE */
static void r31_zpass_main(const struct fft3d_plan *p, const cplx *src, cplx *dst,
                           size_t nrows)
{
#ifdef R31_ZDENSE
    r31_zpass(src, dst, nrows, 31, 31, p->ctd, p->std_);
#else
    size_t r = 0;
    for (; r + 4 <= nrows; r += 4)
        r31_zquad(src + r * 31, dst + r * 31, p->ke, p->ko);
    if (r < nrows)
        r31_zpass(src + r * 31, dst + r * 31, nrows - r, 31, 31, p->ctd, p->std_);
#endif
}

/* ---------------- generic odd-prime folded half-system engine (rp_*) ----------------
 * Round-3 class duty: any odd prime 3 <= p <= 127, p != 31 (which keeps the
 * tuned Winograd path above).  Same fold arithmetic as the r31 z-pass, in
 * column-chunk form with runtime loops: per chunk of 4 complex columns, all
 * p rows are loaded ONCE into stack u/v arrays (loads-all-then-stores => the
 * kernel is in-place safe for any dst == src), then k runs in QUADS of 4
 * (C,S) accumulator pairs sharing each u_j/v_j reload: per j per quad,
 * 2 stack loads + 8 broadcast constants feeding 8 FMAs.  ~2h^2 zmm FMA per
 * chunk per axis -- the settled folded-dense count on FMA hardware. */

#define RP_MAXH 63   /* (127-1)/2 */

/* gen_r7: the map-fused arms (`if (mapc)`) are DEAD in the default build --
 * RP_YMAPFUSE lost its race in gen_r3 and every pass calls with mapc = NULL --
 * yet they compiled to ~57 inline rsqrt/rcp ladders (~20 KB of never-executed
 * code) inside EVERY rp2/rp3 chunk, inflating the fetched footprint of
 * kernels the front-end already struggles to stream (see RP2_CONV_BLK note).
 * RP_MAPARM makes them a compile-time constant 0 unless the knob is on. */
#ifdef RP_YMAPFUSE
#define RP_MAPARM 1
#else
#define RP_MAPARM 0
#endif

static inline __attribute__((always_inline))
void rp_chunk(const double *sx, double *dx, const double *mapc,
              const ptrdiff_t rs, const int p, const int h,
              const double *ct, const double *st,
              int full, __mmask8 msk)
{
    const __m512d SG = _mm512_setr_pd(1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0);
    __m512d U[RP_MAXH + 1], V[RP_MAXH + 1];
#define RP_LD(px, off) (full ? _mm512_loadu_pd((px) + (off)) \
                             : _mm512_maskz_loadu_pd(msk, (px) + (off)))
#define RP_ST(off, v_) do { if (full) _mm512_storeu_pd(dx + (off), v_); \
                            else _mm512_mask_storeu_pd(dx + (off), msk, v_); } while (0)
    __m512d x0 = RP_LD(sx, 0);
    __m512d e0 = x0, e1 = _mm512_setzero_pd();
    for (int j = 1; j <= h; ++j) {
        __m512d a = RP_LD(sx, (ptrdiff_t)j * rs);
        __m512d b = RP_LD(sx, (ptrdiff_t)(p - j) * rs);
        U[j] = _mm512_add_pd(a, b);
        V[j] = _mm512_sub_pd(a, b);
        if (j & 1) e1 = _mm512_add_pd(e1, U[j]);
        else       e0 = _mm512_add_pd(e0, U[j]);
    }
    __m512d X0 = _mm512_add_pd(e0, e1);
    if (RP_MAPARM && mapc) X0 = r31_map1(_mm512_add_pd(X0, RP_LD(mapc, 0)), R31_FMDIV);
    RP_ST(0, X0);
/* X_k = C - iS, X_{p-k} = C + iS on interleaved lanes: o = swap(S),
 * X_k = C + SG*o, X_{p-k} = C - SG*o (SG = +1,-1,...) */
#define RP_OUT(KK, C, S) do {                                                  \
        __m512d o_ = _mm512_permute_pd(S, 0x55);                               \
        __m512d xp_ = _mm512_fmadd_pd(o_, SG, C);                              \
        __m512d xm_ = _mm512_fnmadd_pd(o_, SG, C);                             \
        if (RP_MAPARM && mapc) {                                               \
            xp_ = _mm512_add_pd(xp_, RP_LD(mapc, (ptrdiff_t)(KK) * rs));       \
            xm_ = _mm512_add_pd(xm_, RP_LD(mapc, (ptrdiff_t)(p - (KK)) * rs)); \
            r31_map2(xp_, xm_, &xp_, &xm_, R31_FMDIV);                         \
        }                                                                      \
        RP_ST((ptrdiff_t)(KK) * rs, xp_);                                      \
        RP_ST((ptrdiff_t)(p - (KK)) * rs, xm_);                                \
    } while (0)
    int k = 1;
    for (; k + 3 <= h; k += 4) {
        const double *c0 = ct + (size_t)(k - 1) * h, *c1 = c0 + h,
                     *c2 = c1 + h, *c3 = c2 + h;
        const double *s0 = st + (size_t)(k - 1) * h, *s1 = s0 + h,
                     *s2 = s1 + h, *s3 = s2 + h;
        __m512d C0 = x0, C1 = x0, C2 = x0, C3 = x0;
        __m512d S0 = _mm512_setzero_pd(), S1 = S0, S2 = S0, S3 = S0;
        for (int j = 1; j <= h; ++j) {
            __m512d uj = U[j], vj = V[j];
            C0 = _mm512_fmadd_pd(_mm512_set1_pd(c0[j - 1]), uj, C0);
            S0 = _mm512_fmadd_pd(_mm512_set1_pd(s0[j - 1]), vj, S0);
            C1 = _mm512_fmadd_pd(_mm512_set1_pd(c1[j - 1]), uj, C1);
            S1 = _mm512_fmadd_pd(_mm512_set1_pd(s1[j - 1]), vj, S1);
            C2 = _mm512_fmadd_pd(_mm512_set1_pd(c2[j - 1]), uj, C2);
            S2 = _mm512_fmadd_pd(_mm512_set1_pd(s2[j - 1]), vj, S2);
            C3 = _mm512_fmadd_pd(_mm512_set1_pd(c3[j - 1]), uj, C3);
            S3 = _mm512_fmadd_pd(_mm512_set1_pd(s3[j - 1]), vj, S3);
        }
        RP_OUT(k, C0, S0);
        RP_OUT(k + 1, C1, S1);
        RP_OUT(k + 2, C2, S2);
        RP_OUT(k + 3, C3, S3);
    }
    for (; k <= h; ++k) {
        const double *c0 = ct + (size_t)(k - 1) * h;
        const double *s0 = st + (size_t)(k - 1) * h;
        __m512d C0 = x0, S0 = _mm512_setzero_pd();
        for (int j = 1; j <= h; ++j) {
            C0 = _mm512_fmadd_pd(_mm512_set1_pd(c0[j - 1]), U[j], C0);
            S0 = _mm512_fmadd_pd(_mm512_set1_pd(s0[j - 1]), V[j], S0);
        }
        RP_OUT(k, C0, S0);
    }
#undef RP_OUT
#undef RP_LD
#undef RP_ST
}

/* gen_r8: PAIRED-COLUMN dense chunk (full chunks only; tails run rp_chunk).
 * The 1-wide k-quad is per j: 2 stack loads + 8 broadcasts + 8 FMA -- load-
 * bound AND exactly 8 accumulator chains (zero FMA-latency slack; the r7
 * record's "127 runs 2x above its load-port model" residue).  Two columns
 * share the 8 broadcasts: per j, 4 stack loads + 8 broadcasts + 16 FMA
 * (FMA-bound, 16 chains), and the 8 k-major table streams (63 KB at p=127)
 * are walked once per column PAIR instead of once per column. */
static void rp_chunk2(const double *sx, double *dx,
                      const ptrdiff_t rs, const int p, const int h,
                      const double *ct, const double *st)
{
    const __m512d SG = _mm512_setr_pd(1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0);
    __m512d U[2 * (RP_MAXH + 1)], V[2 * (RP_MAXH + 1)];
    __m512d x0a = _mm512_loadu_pd(sx), x0b = _mm512_loadu_pd(sx + 8);
    __m512d e0a = x0a, e0b = x0b;
    __m512d e1a = _mm512_setzero_pd(), e1b = _mm512_setzero_pd();
    for (int j = 1; j <= h; ++j) {
        const double *ra = sx + (ptrdiff_t)j * rs;
        const double *rb = sx + (ptrdiff_t)(p - j) * rs;
        __m512d a0 = _mm512_loadu_pd(ra), a1 = _mm512_loadu_pd(ra + 8);
        __m512d b0 = _mm512_loadu_pd(rb), b1 = _mm512_loadu_pd(rb + 8);
        U[2 * j] = _mm512_add_pd(a0, b0);
        U[2 * j + 1] = _mm512_add_pd(a1, b1);
        V[2 * j] = _mm512_sub_pd(a0, b0);
        V[2 * j + 1] = _mm512_sub_pd(a1, b1);
        if (j & 1) { e1a = _mm512_add_pd(e1a, U[2 * j]);
                     e1b = _mm512_add_pd(e1b, U[2 * j + 1]); }
        else       { e0a = _mm512_add_pd(e0a, U[2 * j]);
                     e0b = _mm512_add_pd(e0b, U[2 * j + 1]); }
    }
    _mm512_storeu_pd(dx,     _mm512_add_pd(e0a, e1a));
    _mm512_storeu_pd(dx + 8, _mm512_add_pd(e0b, e1b));
#define RP_OUT2(KK, Ca, Cb, Sa, Sb) do {                                       \
        __m512d oa_ = _mm512_permute_pd(Sa, 0x55);                             \
        __m512d ob_ = _mm512_permute_pd(Sb, 0x55);                             \
        const ptrdiff_t rp_ = (ptrdiff_t)(KK) * rs;                            \
        const ptrdiff_t rm_ = (ptrdiff_t)(p - (KK)) * rs;                      \
        _mm512_storeu_pd(dx + rp_,     _mm512_fmadd_pd(oa_, SG, Ca));          \
        _mm512_storeu_pd(dx + rp_ + 8, _mm512_fmadd_pd(ob_, SG, Cb));          \
        _mm512_storeu_pd(dx + rm_,     _mm512_fnmadd_pd(oa_, SG, Ca));         \
        _mm512_storeu_pd(dx + rm_ + 8, _mm512_fnmadd_pd(ob_, SG, Cb));         \
    } while (0)
    int k = 1;
    for (; k + 3 <= h; k += 4) {
        const double *c0 = ct + (size_t)(k - 1) * h, *c1 = c0 + h,
                     *c2 = c1 + h, *c3 = c2 + h;
        const double *s0 = st + (size_t)(k - 1) * h, *s1 = s0 + h,
                     *s2 = s1 + h, *s3 = s2 + h;
        __m512d C0a = x0a, C1a = x0a, C2a = x0a, C3a = x0a;
        __m512d C0b = x0b, C1b = x0b, C2b = x0b, C3b = x0b;
        __m512d S0a = _mm512_setzero_pd(), S1a = S0a, S2a = S0a, S3a = S0a;
        __m512d S0b = S0a, S1b = S0a, S2b = S0a, S3b = S0a;
        for (int j = 1; j <= h; ++j) {
            __m512d ua = U[2 * j], ub = U[2 * j + 1];
            __m512d va = V[2 * j], vb = V[2 * j + 1];
            __m512d kk;
            kk = _mm512_set1_pd(c0[j - 1]);
            C0a = _mm512_fmadd_pd(kk, ua, C0a); C0b = _mm512_fmadd_pd(kk, ub, C0b);
            kk = _mm512_set1_pd(s0[j - 1]);
            S0a = _mm512_fmadd_pd(kk, va, S0a); S0b = _mm512_fmadd_pd(kk, vb, S0b);
            kk = _mm512_set1_pd(c1[j - 1]);
            C1a = _mm512_fmadd_pd(kk, ua, C1a); C1b = _mm512_fmadd_pd(kk, ub, C1b);
            kk = _mm512_set1_pd(s1[j - 1]);
            S1a = _mm512_fmadd_pd(kk, va, S1a); S1b = _mm512_fmadd_pd(kk, vb, S1b);
            kk = _mm512_set1_pd(c2[j - 1]);
            C2a = _mm512_fmadd_pd(kk, ua, C2a); C2b = _mm512_fmadd_pd(kk, ub, C2b);
            kk = _mm512_set1_pd(s2[j - 1]);
            S2a = _mm512_fmadd_pd(kk, va, S2a); S2b = _mm512_fmadd_pd(kk, vb, S2b);
            kk = _mm512_set1_pd(c3[j - 1]);
            C3a = _mm512_fmadd_pd(kk, ua, C3a); C3b = _mm512_fmadd_pd(kk, ub, C3b);
            kk = _mm512_set1_pd(s3[j - 1]);
            S3a = _mm512_fmadd_pd(kk, va, S3a); S3b = _mm512_fmadd_pd(kk, vb, S3b);
        }
        RP_OUT2(k, C0a, C0b, S0a, S0b);
        RP_OUT2(k + 1, C1a, C1b, S1a, S1b);
        RP_OUT2(k + 2, C2a, C2b, S2a, S2b);
        RP_OUT2(k + 3, C3a, C3b, S3a, S3b);
    }
    for (; k <= h; ++k) {
        const double *c0 = ct + (size_t)(k - 1) * h;
        const double *s0 = st + (size_t)(k - 1) * h;
        __m512d Ca = x0a, Cb = x0b;
        __m512d Sa = _mm512_setzero_pd(), Sb = Sa;
        for (int j = 1; j <= h; ++j) {
            __m512d kk;
            kk = _mm512_set1_pd(c0[j - 1]);
            Ca = _mm512_fmadd_pd(kk, U[2 * j], Ca);
            Cb = _mm512_fmadd_pd(kk, U[2 * j + 1], Cb);
            kk = _mm512_set1_pd(s0[j - 1]);
            Sa = _mm512_fmadd_pd(kk, V[2 * j], Sa);
            Sb = _mm512_fmadd_pd(kk, V[2 * j + 1], Sb);
        }
        RP_OUT2(k, Ca, Cb, Sa, Sb);
    }
#undef RP_OUT2
}

/* gen_r12: FOUR-column dense chunk (32 doubles) with an E/O PHASE SPLIT
 * (full chunks only; tails fall through to rp_chunk2/rp_chunk).  The r11
 * differential counters at p=127 put the largest share of the 890 MB/step L1
 * fill in the kernel-table walk: 2h^2 doubles = 63.5 KB per column PAIR.
 * Four columns walk the tables once per 16 complex columns -- half the
 * 2-wide rate -- and halve the per-column broadcast uops again.  Keeping C+S
 * together at 4-wide would need k-steps of 2, DOUBLING per-column 512-bit
 * stack loads on the pooled ~1.12/cyc L1 access class (gen_dense_prime r11's
 * corrected currency); the phase split instead keeps k-quads of 4 with 16
 * accumulators per phase: the E phase (cos on U) stages C_k into the CS
 * stack array, the O phase (sin on V) reloads it for the combine.  Per-
 * column 512-bit load ratio identical to the 2-wide; the staging adds
 * 2 x 4h L1-resident zmm accesses per chunk (~2% of the conv loads).
 * Per-phase concurrent hot set = 16 KB (U or V) + 16 KB CS + one 2 KB table
 * row-group, under the ~40 KB fusion gate (gen_dense_prime r12).  Every
 * accumulator chain receives the same fmadds in the same j order as
 * rp_chunk2/rp_chunk => bit-identical outputs.  In-place safe: all source
 * loads happen in the fold, before any store. */
static void rp_chunk4(const double *sx, double *dx,
                      const ptrdiff_t rs, const int p, const int h,
                      const double *ct, const double *st)
{
    const __m512d SG = _mm512_setr_pd(1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0);
    __m512d U[4 * (RP_MAXH + 1)], V[4 * (RP_MAXH + 1)], CS[4 * (RP_MAXH + 1)];
    __m512d x0a = _mm512_loadu_pd(sx),      x0b = _mm512_loadu_pd(sx + 8),
            x0c = _mm512_loadu_pd(sx + 16), x0d = _mm512_loadu_pd(sx + 24);
    __m512d e0a = x0a, e0b = x0b, e0c = x0c, e0d = x0d;
    __m512d e1a = _mm512_setzero_pd(), e1b = e1a, e1c = e1a, e1d = e1a;
    for (int j = 1; j <= h; ++j) {
        const double *ra = sx + (ptrdiff_t)j * rs;
        const double *rb = sx + (ptrdiff_t)(p - j) * rs;
        __m512d a0 = _mm512_loadu_pd(ra),      b0 = _mm512_loadu_pd(rb);
        __m512d a1 = _mm512_loadu_pd(ra + 8),  b1 = _mm512_loadu_pd(rb + 8);
        __m512d a2 = _mm512_loadu_pd(ra + 16), b2 = _mm512_loadu_pd(rb + 16);
        __m512d a3 = _mm512_loadu_pd(ra + 24), b3 = _mm512_loadu_pd(rb + 24);
        U[4 * j]     = _mm512_add_pd(a0, b0); V[4 * j]     = _mm512_sub_pd(a0, b0);
        U[4 * j + 1] = _mm512_add_pd(a1, b1); V[4 * j + 1] = _mm512_sub_pd(a1, b1);
        U[4 * j + 2] = _mm512_add_pd(a2, b2); V[4 * j + 2] = _mm512_sub_pd(a2, b2);
        U[4 * j + 3] = _mm512_add_pd(a3, b3); V[4 * j + 3] = _mm512_sub_pd(a3, b3);
        if (j & 1) {
            e1a = _mm512_add_pd(e1a, U[4 * j]);
            e1b = _mm512_add_pd(e1b, U[4 * j + 1]);
            e1c = _mm512_add_pd(e1c, U[4 * j + 2]);
            e1d = _mm512_add_pd(e1d, U[4 * j + 3]);
        } else {
            e0a = _mm512_add_pd(e0a, U[4 * j]);
            e0b = _mm512_add_pd(e0b, U[4 * j + 1]);
            e0c = _mm512_add_pd(e0c, U[4 * j + 2]);
            e0d = _mm512_add_pd(e0d, U[4 * j + 3]);
        }
    }
    _mm512_storeu_pd(dx,      _mm512_add_pd(e0a, e1a));
    _mm512_storeu_pd(dx + 8,  _mm512_add_pd(e0b, e1b));
    _mm512_storeu_pd(dx + 16, _mm512_add_pd(e0c, e1c));
    _mm512_storeu_pd(dx + 24, _mm512_add_pd(e0d, e1d));
    /* E phase: cos correlation on U, k-quads of 4 x 4 columns */
    int k = 1;
    for (; k + 3 <= h; k += 4) {
        const double *c0 = ct + (size_t)(k - 1) * h, *c1 = c0 + h,
                     *c2 = c1 + h, *c3 = c2 + h;
        __m512d C0a = x0a, C0b = x0b, C0c = x0c, C0d = x0d;
        __m512d C1a = x0a, C1b = x0b, C1c = x0c, C1d = x0d;
        __m512d C2a = x0a, C2b = x0b, C2c = x0c, C2d = x0d;
        __m512d C3a = x0a, C3b = x0b, C3c = x0c, C3d = x0d;
        for (int j = 1; j <= h; ++j) {
            __m512d ua = U[4 * j],     ub = U[4 * j + 1],
                    uc = U[4 * j + 2], ud = U[4 * j + 3];
            __m512d kk;
            kk = _mm512_set1_pd(c0[j - 1]);
            C0a = _mm512_fmadd_pd(kk, ua, C0a); C0b = _mm512_fmadd_pd(kk, ub, C0b);
            C0c = _mm512_fmadd_pd(kk, uc, C0c); C0d = _mm512_fmadd_pd(kk, ud, C0d);
            kk = _mm512_set1_pd(c1[j - 1]);
            C1a = _mm512_fmadd_pd(kk, ua, C1a); C1b = _mm512_fmadd_pd(kk, ub, C1b);
            C1c = _mm512_fmadd_pd(kk, uc, C1c); C1d = _mm512_fmadd_pd(kk, ud, C1d);
            kk = _mm512_set1_pd(c2[j - 1]);
            C2a = _mm512_fmadd_pd(kk, ua, C2a); C2b = _mm512_fmadd_pd(kk, ub, C2b);
            C2c = _mm512_fmadd_pd(kk, uc, C2c); C2d = _mm512_fmadd_pd(kk, ud, C2d);
            kk = _mm512_set1_pd(c3[j - 1]);
            C3a = _mm512_fmadd_pd(kk, ua, C3a); C3b = _mm512_fmadd_pd(kk, ub, C3b);
            C3c = _mm512_fmadd_pd(kk, uc, C3c); C3d = _mm512_fmadd_pd(kk, ud, C3d);
        }
        CS[4 * k]           = C0a; CS[4 * k + 1]       = C0b;
        CS[4 * k + 2]       = C0c; CS[4 * k + 3]       = C0d;
        CS[4 * (k + 1)]     = C1a; CS[4 * (k + 1) + 1] = C1b;
        CS[4 * (k + 1) + 2] = C1c; CS[4 * (k + 1) + 3] = C1d;
        CS[4 * (k + 2)]     = C2a; CS[4 * (k + 2) + 1] = C2b;
        CS[4 * (k + 2) + 2] = C2c; CS[4 * (k + 2) + 3] = C2d;
        CS[4 * (k + 3)]     = C3a; CS[4 * (k + 3) + 1] = C3b;
        CS[4 * (k + 3) + 2] = C3c; CS[4 * (k + 3) + 3] = C3d;
    }
    for (; k <= h; ++k) {
        const double *c0 = ct + (size_t)(k - 1) * h;
        __m512d Ca = x0a, Cb = x0b, Cc = x0c, Cd = x0d;
        for (int j = 1; j <= h; ++j) {
            __m512d kk = _mm512_set1_pd(c0[j - 1]);
            Ca = _mm512_fmadd_pd(kk, U[4 * j], Ca);
            Cb = _mm512_fmadd_pd(kk, U[4 * j + 1], Cb);
            Cc = _mm512_fmadd_pd(kk, U[4 * j + 2], Cc);
            Cd = _mm512_fmadd_pd(kk, U[4 * j + 3], Cd);
        }
        CS[4 * k] = Ca; CS[4 * k + 1] = Cb; CS[4 * k + 2] = Cc; CS[4 * k + 3] = Cd;
    }
    /* O phase: sin correlation on V, combine with the staged C
     * (fmadd(o,SG,C) / fnmadd(o,SG,C) exactly as RP_OUT2 -- and the CS
     * store/reload is bit-preserving) */
#define RP_OUT4(KK, Sa, Sb, Sc, Sd) do {                                       \
        const ptrdiff_t rp_ = (ptrdiff_t)(KK) * rs;                            \
        const ptrdiff_t rm_ = (ptrdiff_t)(p - (KK)) * rs;                      \
        __m512d oa_ = _mm512_permute_pd(Sa, 0x55);                             \
        __m512d ob_ = _mm512_permute_pd(Sb, 0x55);                             \
        __m512d oc_ = _mm512_permute_pd(Sc, 0x55);                             \
        __m512d od_ = _mm512_permute_pd(Sd, 0x55);                             \
        _mm512_storeu_pd(dx + rp_,      _mm512_fmadd_pd(oa_, SG, CS[4 * (KK)]));    \
        _mm512_storeu_pd(dx + rp_ + 8,  _mm512_fmadd_pd(ob_, SG, CS[4 * (KK) + 1]));\
        _mm512_storeu_pd(dx + rp_ + 16, _mm512_fmadd_pd(oc_, SG, CS[4 * (KK) + 2]));\
        _mm512_storeu_pd(dx + rp_ + 24, _mm512_fmadd_pd(od_, SG, CS[4 * (KK) + 3]));\
        _mm512_storeu_pd(dx + rm_,      _mm512_fnmadd_pd(oa_, SG, CS[4 * (KK)]));    \
        _mm512_storeu_pd(dx + rm_ + 8,  _mm512_fnmadd_pd(ob_, SG, CS[4 * (KK) + 1]));\
        _mm512_storeu_pd(dx + rm_ + 16, _mm512_fnmadd_pd(oc_, SG, CS[4 * (KK) + 2]));\
        _mm512_storeu_pd(dx + rm_ + 24, _mm512_fnmadd_pd(od_, SG, CS[4 * (KK) + 3]));\
    } while (0)
    k = 1;
    for (; k + 3 <= h; k += 4) {
        const double *s0 = st + (size_t)(k - 1) * h, *s1 = s0 + h,
                     *s2 = s1 + h, *s3 = s2 + h;
        __m512d S0a = _mm512_setzero_pd(), S0b = S0a, S0c = S0a, S0d = S0a;
        __m512d S1a = S0a, S1b = S0a, S1c = S0a, S1d = S0a;
        __m512d S2a = S0a, S2b = S0a, S2c = S0a, S2d = S0a;
        __m512d S3a = S0a, S3b = S0a, S3c = S0a, S3d = S0a;
        for (int j = 1; j <= h; ++j) {
            __m512d va = V[4 * j],     vb = V[4 * j + 1],
                    vc = V[4 * j + 2], vd = V[4 * j + 3];
            __m512d kk;
            kk = _mm512_set1_pd(s0[j - 1]);
            S0a = _mm512_fmadd_pd(kk, va, S0a); S0b = _mm512_fmadd_pd(kk, vb, S0b);
            S0c = _mm512_fmadd_pd(kk, vc, S0c); S0d = _mm512_fmadd_pd(kk, vd, S0d);
            kk = _mm512_set1_pd(s1[j - 1]);
            S1a = _mm512_fmadd_pd(kk, va, S1a); S1b = _mm512_fmadd_pd(kk, vb, S1b);
            S1c = _mm512_fmadd_pd(kk, vc, S1c); S1d = _mm512_fmadd_pd(kk, vd, S1d);
            kk = _mm512_set1_pd(s2[j - 1]);
            S2a = _mm512_fmadd_pd(kk, va, S2a); S2b = _mm512_fmadd_pd(kk, vb, S2b);
            S2c = _mm512_fmadd_pd(kk, vc, S2c); S2d = _mm512_fmadd_pd(kk, vd, S2d);
            kk = _mm512_set1_pd(s3[j - 1]);
            S3a = _mm512_fmadd_pd(kk, va, S3a); S3b = _mm512_fmadd_pd(kk, vb, S3b);
            S3c = _mm512_fmadd_pd(kk, vc, S3c); S3d = _mm512_fmadd_pd(kk, vd, S3d);
        }
        RP_OUT4(k,     S0a, S0b, S0c, S0d);
        RP_OUT4(k + 1, S1a, S1b, S1c, S1d);
        RP_OUT4(k + 2, S2a, S2b, S2c, S2d);
        RP_OUT4(k + 3, S3a, S3b, S3c, S3d);
    }
    for (; k <= h; ++k) {
        const double *s0 = st + (size_t)(k - 1) * h;
        __m512d Sa = _mm512_setzero_pd(), Sb = Sa, Sc = Sa, Sd = Sa;
        for (int j = 1; j <= h; ++j) {
            __m512d kk = _mm512_set1_pd(s0[j - 1]);
            Sa = _mm512_fmadd_pd(kk, V[4 * j], Sa);
            Sb = _mm512_fmadd_pd(kk, V[4 * j + 1], Sb);
            Sc = _mm512_fmadd_pd(kk, V[4 * j + 2], Sc);
            Sd = _mm512_fmadd_pd(kk, V[4 * j + 3], Sd);
        }
        RP_OUT4(k, Sa, Sb, Sc, Sd);
    }
#undef RP_OUT4
}

/* ---------------- outer-C3 chunk kernel (gen_r5) ----------------
 * The r31_chunk shape at runtime table indices: fold via js/jdp/jdm, TWO
 * Winograd-C3-over-dense-Cm cyclic-h convolutions (4 block products of m^2
 * FMA + 11m block adds each, vs the dense engine's 2h^2 = 18m^2 FMA), output
 * rows via kp/km.  Instantiated per m (7/11/13/17) so every loop bound is a
 * compile-time constant and the block products unroll register-resident,
 * exactly like R31_C5.  In-place safe: all loads of a row precede all stores
 * (the E sweep stores only row 0, which the O sweep never reads). */

#define RP3_LD(px, off) (full ? _mm512_loadu_pd((px) + (off)) \
                              : _mm512_maskz_loadu_pd(msk, (px) + (off)))
#define RP3_ST(off, v_) do { if (full) _mm512_storeu_pd(dx + (off), v_); \
                             else _mm512_mask_storeu_pd(dx + (off), msk, v_); } while (0)

/* full-unroll pragmas: without these gcc 11 keeps the m >= 13 conv loops
 * ROLLED with stack-resident accumulators (load-fma-store per step, a
 * vmovapd storm -- seen in the gen_r6 objdump audit on rp3_chunk_17 too:
 * only ~430 of its 2312 design FMAs were straight-line).  The pragma forces
 * complete peeling; SRA then scalarizes the constant-index arrays. */
#define RP_UNROLL _Pragma("GCC unroll 64")

#define RP3_CONV(DST, SRC, KB, M) do {                                        \
    RP_UNROLL                                                                 \
    for (int j_ = 0; j_ < (M); ++j_)                                          \
        DST[j_] = _mm512_mul_pd(SRC[0], _mm512_set1_pd((KB)[(M) - 1 + j_]));  \
    RP_UNROLL                                                                 \
    for (int i_ = 1; i_ < (M); ++i_) {                                        \
        RP_UNROLL                                                             \
        for (int j_ = 0; j_ < (M); ++j_)                                      \
            DST[j_] = _mm512_fmadd_pd(SRC[i_],                                \
                          _mm512_set1_pd((KB)[(M) - 1 + j_ - i_]), DST[j_]);  \
    }                                                                         \
} while (0)

/* blocked conv for LARGE m (>= RP2_BLK_MIN): j in tiles of 8 register
 * accumulators, the i loop ROLLED -- per i: 1 stack load + 8 broadcast
 * loads + 8 FMA, the dense engine's proven k-quad shape on 5m^2 work.
 * The fully-unrolled RP3_CONV at m >= ~22 spills catastrophically (m = 25:
 * 7400 vmovapd vs 3900 FMA, +70% wall on the node-class sizes); this form
 * keeps ~10 live zmm.  Tiles overread the kernel block by up to 7 slots:
 * rp2_build zero-pads each block to K + 7 (0 lands in DST slots >= m,
 * which are never stored).
 * gen_r7: the i loop is PINNED rolled (unroll 1).  -funroll-loops was
 * unrolling it 8x behind the r6 race's back: rp2_chunk_28 compiled to
 * 10820 instructions / ~75 KB of straight-line code PER CHUNK (objdump:
 * 2443 stack vmovapd) -- past L1I, so every chunk refetched its own code
 * from L2.  That, not DRAM, was why the y-pass ran 6.7 ms/step on an
 * L2/L3-resident plane at p=113 (load-port model: ~2.4).  Rolled, the tile
 * body is ~15 instructions, DSB-resident; accumulation order unchanged =>
 * bit-identical.  Node-raced per m (same-core interleaved, 3 rounds):
 * rolled wins 61 -9..11%, 73 -2%, 103 -1%, 89 -7%, 101 -19%, 113 -8%;
 * WASH at 41; LOSES +1% at 53 (m=13, 3/3) -- so m=13 alone keeps the gcc
 * default via the RP2_BLK_IUNROLL redefinition at its instantiation. */
#define RP2_BLK_IUNROLL _Pragma("GCC unroll 1")
#define RP2_CONV_BLK(DST, SRC, KB, M) do {                                    \
    RP_UNROLL                                                                 \
    for (int jb_ = 0; jb_ < (M); jb_ += 8) {                                  \
        __m512d c0_, c1_, c2_, c3_, c4_, c5_, c6_, c7_;                       \
        {                                                                     \
            const double *kk_ = (KB) + (M) - 1 + jb_;                         \
            __m512d s_ = SRC[0];                                              \
            c0_ = _mm512_mul_pd(s_, _mm512_set1_pd(kk_[0]));                  \
            c1_ = _mm512_mul_pd(s_, _mm512_set1_pd(kk_[1]));                  \
            c2_ = _mm512_mul_pd(s_, _mm512_set1_pd(kk_[2]));                  \
            c3_ = _mm512_mul_pd(s_, _mm512_set1_pd(kk_[3]));                  \
            c4_ = _mm512_mul_pd(s_, _mm512_set1_pd(kk_[4]));                  \
            c5_ = _mm512_mul_pd(s_, _mm512_set1_pd(kk_[5]));                  \
            c6_ = _mm512_mul_pd(s_, _mm512_set1_pd(kk_[6]));                  \
            c7_ = _mm512_mul_pd(s_, _mm512_set1_pd(kk_[7]));                  \
        }                                                                     \
        RP2_BLK_IUNROLL                                                       \
        for (int i_ = 1; i_ < (M); ++i_) {                                    \
            const double *kk_ = (KB) + (M) - 1 + jb_ - i_;                    \
            __m512d s_ = SRC[i_];                                             \
            c0_ = _mm512_fmadd_pd(s_, _mm512_set1_pd(kk_[0]), c0_);           \
            c1_ = _mm512_fmadd_pd(s_, _mm512_set1_pd(kk_[1]), c1_);           \
            c2_ = _mm512_fmadd_pd(s_, _mm512_set1_pd(kk_[2]), c2_);           \
            c3_ = _mm512_fmadd_pd(s_, _mm512_set1_pd(kk_[3]), c3_);           \
            c4_ = _mm512_fmadd_pd(s_, _mm512_set1_pd(kk_[4]), c4_);           \
            c5_ = _mm512_fmadd_pd(s_, _mm512_set1_pd(kk_[5]), c5_);           \
            c6_ = _mm512_fmadd_pd(s_, _mm512_set1_pd(kk_[6]), c6_);           \
            c7_ = _mm512_fmadd_pd(s_, _mm512_set1_pd(kk_[7]), c7_);           \
        }                                                                     \
        DST[jb_] = c0_;                                                       \
        if (jb_ + 1 < (M)) DST[jb_ + 1] = c1_;                                \
        if (jb_ + 2 < (M)) DST[jb_ + 2] = c2_;                                \
        if (jb_ + 3 < (M)) DST[jb_ + 3] = c3_;                                \
        if (jb_ + 4 < (M)) DST[jb_ + 4] = c4_;                                \
        if (jb_ + 5 < (M)) DST[jb_ + 5] = c5_;                                \
        if (jb_ + 6 < (M)) DST[jb_ + 6] = c6_;                                \
        if (jb_ + 7 < (M)) DST[jb_ + 7] = c7_;                                \
    }                                                                         \
} while (0)

/* gen_r8: PAIRED-COLUMN (2-wide) blocked conv -- the r7 next-step-1 lever.
 * The 1-wide blocked tile is per i: 1 stack load + 8 broadcasts + 8 FMA =
 * 9 loads / 8 FMA -- load-port bound (4.5 vs 4 cyc on 2 ports) AND exactly
 * 8 accumulator chains = 2 FMA pipes x 4-cyc latency with ZERO slack (the
 * r7 profile's "y-pass 2x above its load-port model on L2-hot data").
 * Processing TWO zmm columns per chunk shares the 8 broadcasts: per i,
 * 2 stack loads + 8 broadcasts + 16 FMA = 10 loads / 16 FMA -- FMA-bound
 * with 16 independent chains (2x latency slack).  SRC/DST hold column pairs
 * interleaved: slot j's columns at [2j] and [2j+1].  Same K+7 kernel pad,
 * same overread rule, i loop pinned rolled (the r7 front-end lesson). */
#define RP2_CONV_BLK2(DST, SRC, KB, M) do {                                   \
    RP_UNROLL                                                                 \
    for (int jb_ = 0; jb_ < (M); jb_ += 8) {                                  \
        __m512d d0a_, d0b_, d1a_, d1b_, d2a_, d2b_, d3a_, d3b_;               \
        __m512d d4a_, d4b_, d5a_, d5b_, d6a_, d6b_, d7a_, d7b_;               \
        {                                                                     \
            const double *kk_ = (KB) + (M) - 1 + jb_;                         \
            __m512d sa_ = SRC[0], sb_ = SRC[1], k_;                           \
            k_ = _mm512_set1_pd(kk_[0]);                                      \
            d0a_ = _mm512_mul_pd(sa_, k_); d0b_ = _mm512_mul_pd(sb_, k_);     \
            k_ = _mm512_set1_pd(kk_[1]);                                      \
            d1a_ = _mm512_mul_pd(sa_, k_); d1b_ = _mm512_mul_pd(sb_, k_);     \
            k_ = _mm512_set1_pd(kk_[2]);                                      \
            d2a_ = _mm512_mul_pd(sa_, k_); d2b_ = _mm512_mul_pd(sb_, k_);     \
            k_ = _mm512_set1_pd(kk_[3]);                                      \
            d3a_ = _mm512_mul_pd(sa_, k_); d3b_ = _mm512_mul_pd(sb_, k_);     \
            k_ = _mm512_set1_pd(kk_[4]);                                      \
            d4a_ = _mm512_mul_pd(sa_, k_); d4b_ = _mm512_mul_pd(sb_, k_);     \
            k_ = _mm512_set1_pd(kk_[5]);                                      \
            d5a_ = _mm512_mul_pd(sa_, k_); d5b_ = _mm512_mul_pd(sb_, k_);     \
            k_ = _mm512_set1_pd(kk_[6]);                                      \
            d6a_ = _mm512_mul_pd(sa_, k_); d6b_ = _mm512_mul_pd(sb_, k_);     \
            k_ = _mm512_set1_pd(kk_[7]);                                      \
            d7a_ = _mm512_mul_pd(sa_, k_); d7b_ = _mm512_mul_pd(sb_, k_);     \
        }                                                                     \
        _Pragma("GCC unroll 1")                                               \
        for (int i_ = 1; i_ < (M); ++i_) {                                    \
            const double *kk_ = (KB) + (M) - 1 + jb_ - i_;                    \
            __m512d sa_ = SRC[2 * i_], sb_ = SRC[2 * i_ + 1], k_;             \
            k_ = _mm512_set1_pd(kk_[0]);                                      \
            d0a_ = _mm512_fmadd_pd(sa_, k_, d0a_);                            \
            d0b_ = _mm512_fmadd_pd(sb_, k_, d0b_);                            \
            k_ = _mm512_set1_pd(kk_[1]);                                      \
            d1a_ = _mm512_fmadd_pd(sa_, k_, d1a_);                            \
            d1b_ = _mm512_fmadd_pd(sb_, k_, d1b_);                            \
            k_ = _mm512_set1_pd(kk_[2]);                                      \
            d2a_ = _mm512_fmadd_pd(sa_, k_, d2a_);                            \
            d2b_ = _mm512_fmadd_pd(sb_, k_, d2b_);                            \
            k_ = _mm512_set1_pd(kk_[3]);                                      \
            d3a_ = _mm512_fmadd_pd(sa_, k_, d3a_);                            \
            d3b_ = _mm512_fmadd_pd(sb_, k_, d3b_);                            \
            k_ = _mm512_set1_pd(kk_[4]);                                      \
            d4a_ = _mm512_fmadd_pd(sa_, k_, d4a_);                            \
            d4b_ = _mm512_fmadd_pd(sb_, k_, d4b_);                            \
            k_ = _mm512_set1_pd(kk_[5]);                                      \
            d5a_ = _mm512_fmadd_pd(sa_, k_, d5a_);                            \
            d5b_ = _mm512_fmadd_pd(sb_, k_, d5b_);                            \
            k_ = _mm512_set1_pd(kk_[6]);                                      \
            d6a_ = _mm512_fmadd_pd(sa_, k_, d6a_);                            \
            d6b_ = _mm512_fmadd_pd(sb_, k_, d6b_);                            \
            k_ = _mm512_set1_pd(kk_[7]);                                      \
            d7a_ = _mm512_fmadd_pd(sa_, k_, d7a_);                            \
            d7b_ = _mm512_fmadd_pd(sb_, k_, d7b_);                            \
        }                                                                     \
        DST[2 * jb_] = d0a_; DST[2 * jb_ + 1] = d0b_;                         \
        if (jb_ + 1 < (M)) { DST[2 * (jb_ + 1)] = d1a_;                       \
                             DST[2 * (jb_ + 1) + 1] = d1b_; }                 \
        if (jb_ + 2 < (M)) { DST[2 * (jb_ + 2)] = d2a_;                       \
                             DST[2 * (jb_ + 2) + 1] = d2b_; }                 \
        if (jb_ + 3 < (M)) { DST[2 * (jb_ + 3)] = d3a_;                       \
                             DST[2 * (jb_ + 3) + 1] = d3b_; }                 \
        if (jb_ + 4 < (M)) { DST[2 * (jb_ + 4)] = d4a_;                       \
                             DST[2 * (jb_ + 4) + 1] = d4b_; }                 \
        if (jb_ + 5 < (M)) { DST[2 * (jb_ + 5)] = d5a_;                       \
                             DST[2 * (jb_ + 5) + 1] = d5b_; }                 \
        if (jb_ + 6 < (M)) { DST[2 * (jb_ + 6)] = d6a_;                       \
                             DST[2 * (jb_ + 6) + 1] = d6b_; }                 \
        if (jb_ + 7 < (M)) { DST[2 * (jb_ + 7)] = d7a_;                       \
                             DST[2 * (jb_ + 7) + 1] = d7b_; }                 \
    }                                                                         \
} while (0)

#define RP3_WINO(KT, M, WANT_ESUM, CONV) do {                                       \
    RP_UNROLL                                                                 \
    for (int j_ = 0; j_ < (M); ++j_) {                                        \
        E[j_] = _mm512_add_pd(_mm512_add_pd(S[j_], S[(M) + j_]),              \
                              S[2 * (M) + j_]);                               \
        A[j_] = _mm512_sub_pd(S[j_], S[2 * (M) + j_]);                        \
        B[j_] = _mm512_sub_pd(S[(M) + j_], S[2 * (M) + j_]);                  \
        C[j_] = _mm512_sub_pd(S[j_], S[(M) + j_]);                            \
        if (WANT_ESUM) esum = _mm512_add_pd(esum, E[j_]);                     \
    }                                                                         \
    CONV(M0, E, (KT), M);                                                 \
    CONV(M1, A, (KT) + (2 * (M) - 1), M);                                 \
    CONV(M2, B, (KT) + 2 * (2 * (M) - 1), M);                             \
    CONV(M3, C, (KT) + 3 * (2 * (M) - 1), M);                             \
    RP_UNROLL                                                                 \
    for (int j_ = 0; j_ < (M); ++j_) {                                        \
        __m512d t01 = _mm512_add_pd(M0[j_], M1[j_]);                          \
        Y[j_] = _mm512_fnmadd_pd(M2[j_], TWO, _mm512_add_pd(t01, M3[j_]));    \
        Y[(M) + j_] = _mm512_fnmadd_pd(M3[j_], TWO,                           \
                                       _mm512_add_pd(t01, M2[j_]));           \
        __m512d t23 = _mm512_add_pd(M2[j_], M3[j_]);                          \
        Y[2 * (M) + j_] = _mm512_fnmadd_pd(M1[j_], TWO,                       \
                                           _mm512_add_pd(M0[j_], t23));       \
    }                                                                         \
} while (0)

/* -DRP3_FOLD1 switches the rp3 kernels to the one-pass fold the rp2 kernels
 * use (rows loaded once, U/V + negated mirror on the stack).  RACED AND LOST
 * on the node (gen_r6): +15% at 43, +29% at 67, +39% at 103 -- the WINO
 * working set (S/Y/T + 8 block arrays) is already spill-saturated, so the
 * extra 9m stack slots and the fill indirection cost more than the 2h row
 * loads they save (the rows hit L2 and the two load ports absorb them).
 * Outputs bit-identical either way (cmp-verified at 67). */
#ifdef RP3_FOLD1
#define RP3_FOLD1_ 1
#else
#define RP3_FOLD1_ 0
#endif

#define RP3_DEFINE(M, CONV)                                                   \
static void rp3_chunk_##M(const double *sx, double *dx, const double *mapc,   \
                          const ptrdiff_t rs, const fft3d_plan *pl,           \
                          int full, __mmask8 msk)                             \
{                                                                             \
    enum { H_ = 3 * (M), P_ = 6 * (M) + 1 };                                  \
    const __m512d SG = _mm512_setr_pd(1.0, -1.0, 1.0, -1.0,                   \
                                      1.0, -1.0, 1.0, -1.0);                  \
    const __m512d TWO = _mm512_set1_pd(2.0);                                  \
    const int *js = pl->j3, *js2 = js + H_, *jdp = js2 + H_,                  \
              *jdm = jdp + H_, *kp = jdm + H_, *km = kp + H_,                 \
              *jv = km + H_;                                                  \
    __m512d S[H_], Y[H_], T[H_], UF[H_], VF[2 * H_];                          \
    __m512d E[M], A[M], B[M], C[M], M0[M], M1[M], M2[M], M3[M];               \
    __m512d esum = _mm512_setzero_pd();                                       \
    __m512d x0 = RP3_LD(sx, 0);                                               \
    if (RP3_FOLD1_) {                                                         \
        RP_UNROLL                                                             \
        for (int j_ = 1; j_ <= H_; ++j_) {                                    \
            __m512d a_ = RP3_LD(sx, (ptrdiff_t)j_ * rs);                      \
            __m512d b_ = RP3_LD(sx, (ptrdiff_t)(P_ - j_) * rs);               \
            UF[j_ - 1] = _mm512_add_pd(a_, b_);                               \
            VF[j_ - 1] = _mm512_sub_pd(a_, b_);                               \
            VF[H_ + j_ - 1] = _mm512_sub_pd(b_, a_);                          \
        }                                                                     \
        RP_UNROLL                                                             \
        for (int n = 0; n < H_; ++n) S[n] = UF[js[n] - 1];                    \
    } else {                                                                  \
        RP_UNROLL                                                             \
        for (int n = 0; n < H_; ++n)                                          \
            S[n] = _mm512_add_pd(RP3_LD(sx, (ptrdiff_t)js[n] * rs),           \
                                 RP3_LD(sx, (ptrdiff_t)js2[n] * rs));         \
    }                                                                         \
    RP3_WINO(pl->k3e, M, 1, CONV);                                            \
    __m512d X0 = _mm512_add_pd(x0, esum);                                     \
    if (RP_MAPARM && mapc)                                                    \
        X0 = r31_map1(_mm512_add_pd(X0, RP3_LD(mapc, 0)), R31_FMDIV);         \
    RP3_ST(0, X0);                                                            \
    RP_UNROLL                                                                 \
    for (int n = 0; n < H_; ++n) T[n] = _mm512_add_pd(x0, Y[n]);              \
    if (RP3_FOLD1_) {                                                         \
        RP_UNROLL                                                             \
        for (int n = 0; n < H_; ++n) S[n] = VF[jv[n]];                        \
    } else {                                                                  \
        RP_UNROLL                                                             \
        for (int n = 0; n < H_; ++n)                                          \
            S[n] = _mm512_sub_pd(RP3_LD(sx, (ptrdiff_t)jdp[n] * rs),          \
                                 RP3_LD(sx, (ptrdiff_t)jdm[n] * rs));         \
    }                                                                         \
    RP3_WINO(pl->k3o, M, 0, CONV);                                            \
    RP_UNROLL                                                                 \
    for (int n = 0; n < H_; ++n) {                                            \
        __m512d o = _mm512_permute_pd(Y[n], 0x55);                            \
        __m512d xp = _mm512_fmadd_pd(o, SG, T[n]);                            \
        __m512d xm = _mm512_fnmadd_pd(o, SG, T[n]);                           \
        if (RP_MAPARM && mapc) {                                              \
            xp = _mm512_add_pd(xp, RP3_LD(mapc, (ptrdiff_t)kp[n] * rs));      \
            xm = _mm512_add_pd(xm, RP3_LD(mapc, (ptrdiff_t)km[n] * rs));      \
            r31_map2(xp, xm, &xp, &xm, R31_FMDIV);                            \
        }                                                                     \
        RP3_ST((ptrdiff_t)kp[n] * rs, xp);                                    \
        RP3_ST((ptrdiff_t)km[n] * rs, xm);                                    \
    }                                                                         \
}

/* conv form per m, node-raced (gen_r6, re-raced gen_r7 against the FIXED
 * blocked codegen): the rp3 WINO makes 4 conv calls of m^2 each, so its
 * full-unroll spill ceiling sits higher than rp2's.  -DRP3_BLK11/-DRP3_BLK13
 * race the rolled-blocked form at 67/79. */
RP3_DEFINE(7,  RP3_CONV)      /* p = 43 */
#ifdef RP3_BLK11
RP3_DEFINE(11, RP2_CONV_BLK)
#else
RP3_DEFINE(11, RP3_CONV)      /* p = 67 */
#endif
#ifdef RP3_BLK13
RP3_DEFINE(13, RP2_CONV_BLK)
#else
RP3_DEFINE(13, RP3_CONV)      /* p = 79 */
#endif
RP3_DEFINE(17, RP2_CONV_BLK)  /* p = 103 */

/* ---------------- even-h split chunk kernel (gen_r6) ----------------
 * See rp2_build for the arithmetic.  Instantiated per m so every loop bound
 * is compile-time and the five dense m^2-FMA block products unroll register-
 * resident (the rp3/dense_prime exact-tile doctrine).  5m^2 + ~12m FMA-class
 * per chunk per system pair vs the dense engine's 8m^2.  Rows are folded
 * ONCE into stack U/V arrays (2h row loads at compile-time offsets, p = 4m+1
 * known per instantiation; the rp3 kernels load every row twice through
 * runtime index tables) with a negated V mirror so both conv fills are pure
 * stack loads.  In-place safe: the fold loads every row before the only
 * store the sweep makes to row 0; all other stores follow all loads. */

/* 1-wide fold/reconstruct/combine unroll, per m (gen_r8): the attribution
 * race showed ROLLING these loops (runtime index tables, ~20-instr bodies)
 * is a win on its own from m = 15 up (61 -1..4%, 73 -1..10%, 89 -3..8% vs
 * the peeled r7 form, 3/3 each) and a loss below (41 +8..14%, 53 +12..14%).
 * Default: peeled <= m 13, rolled >= 15 (redefined at the instantiation
 * boundary below).  -DRP_W1ROLL / -DRP_W1FULL force all-rolled/all-peeled
 * for the xarch race. */
#ifdef RP_W1ROLL
#define RP2_FC1UNROLL _Pragma("GCC unroll 1")
#else
#define RP2_FC1UNROLL RP_UNROLL
#endif

#define RP2_DEFINE(M, CONV)                                                   \
static void rp2_chunk_##M(const double *sx, double *dx, const double *mapc,   \
                          const ptrdiff_t rs, const fft3d_plan *pl,           \
                          int full, __mmask8 msk)                             \
{                                                                             \
    enum { H_ = 2 * (M), K_ = 2 * (M) - 1, KS_ = K_ + 7, P_ = 4 * (M) + 1 };  \
    const __m512d SG = _mm512_setr_pd(1.0, -1.0, 1.0, -1.0,                   \
                                      1.0, -1.0, 1.0, -1.0);                  \
    const int *js = pl->j2, *jv = js + H_, *kp = jv + H_, *km = kp + H_;      \
    const double *kc = pl->k2, *kn = kc + KS_, *kb0 = kn + KS_,               \
                 *kb1 = kb0 + KS_, *kb2 = kb1 + KS_;                          \
    __m512d UF[H_], VF[2 * H_], T[H_], O_[H_];                                \
    __m512d Q0[M], Q1[M], P0[M], P1[M], P2[M];                                \
    __m512d x0 = RP3_LD(sx, 0);                                               \
    RP2_FC1UNROLL                                                                 \
    for (int j_ = 1; j_ <= H_; ++j_) {       /* one-pass fold: rows at        \
                                              * COMPILE-TIME offsets */       \
        __m512d a_ = RP3_LD(sx, (ptrdiff_t)j_ * rs);                          \
        __m512d b_ = RP3_LD(sx, (ptrdiff_t)(P_ - j_) * rs);                   \
        UF[j_ - 1] = _mm512_add_pd(a_, b_);                                   \
        VF[j_ - 1] = _mm512_sub_pd(a_, b_);                                   \
        VF[H_ + j_ - 1] = _mm512_sub_pd(b_, a_);                              \
    }                                                                         \
    __m512d esum = _mm512_setzero_pd();                                       \
    RP2_FC1UNROLL                                                                 \
    for (int j_ = 0; j_ < (M); ++j_) {                                        \
        __m512d slo = UF[js[j_] - 1], shi = UF[js[(M) + j_] - 1];             \
        Q0[j_] = _mm512_add_pd(slo, shi);                /* mod z^m - 1 */    \
        Q1[j_] = _mm512_sub_pd(slo, shi);                /* mod z^m + 1 */    \
        esum = _mm512_add_pd(esum, Q0[j_]);                                   \
    }                                                                         \
    CONV(P0, Q0, kc, M);                                                      \
    CONV(P1, Q1, kn, M);                                                      \
    __m512d X0 = _mm512_add_pd(x0, esum);                                     \
    if (RP_MAPARM && mapc)                                                    \
        X0 = r31_map1(_mm512_add_pd(X0, RP3_LD(mapc, 0)), R31_FMDIV);         \
    RP3_ST(0, X0);                                                            \
    RP2_FC1UNROLL                                                                 \
    for (int j_ = 0; j_ < (M); ++j_) {                                        \
        T[j_] = _mm512_add_pd(x0, _mm512_add_pd(P0[j_], P1[j_]));             \
        T[(M) + j_] = _mm512_add_pd(x0, _mm512_sub_pd(P0[j_], P1[j_]));       \
    }                                                                         \
    RP2_FC1UNROLL                                                                 \
    for (int j_ = 0; j_ < (M); ++j_) {                                        \
        Q0[j_] = VF[jv[2 * j_]];                         /* A0: even slots */ \
        Q1[j_] = VF[jv[2 * j_ + 1]];                     /* A1: odd slots  */ \
    }                                                                         \
    CONV(P0, Q0, kb0, M);                                                     \
    CONV(P1, Q1, kb1, M);                                                     \
    RP2_FC1UNROLL                                                                 \
    for (int j_ = 0; j_ < (M); ++j_)                                          \
        Q0[j_] = _mm512_add_pd(Q0[j_], Q1[j_]);                               \
    CONV(P2, Q0, kb2, M);                                                     \
    O_[0] = _mm512_sub_pd(P0[0], P1[(M) - 1]);           /* y-wrap sign */    \
    RP2_FC1UNROLL                                                                 \
    for (int j_ = 1; j_ < (M); ++j_)                                          \
        O_[2 * j_] = _mm512_add_pd(P0[j_], P1[j_ - 1]);                       \
    RP2_FC1UNROLL                                                                 \
    for (int j_ = 0; j_ < (M); ++j_)                                          \
        O_[2 * j_ + 1] = _mm512_sub_pd(P2[j_],                                \
                                       _mm512_add_pd(P0[j_], P1[j_]));        \
    RP2_FC1UNROLL                                                                 \
    for (int n = 0; n < H_; ++n) {                                            \
        __m512d o = _mm512_permute_pd(O_[n], 0x55);                           \
        __m512d xp = _mm512_fmadd_pd(o, SG, T[n]);                            \
        __m512d xm = _mm512_fnmadd_pd(o, SG, T[n]);                           \
        if (RP_MAPARM && mapc) {                                              \
            xp = _mm512_add_pd(xp, RP3_LD(mapc, (ptrdiff_t)kp[n] * rs));      \
            xm = _mm512_add_pd(xm, RP3_LD(mapc, (ptrdiff_t)km[n] * rs));      \
            r31_map2(xp, xm, &xp, &xm, R31_FMDIV);                            \
        }                                                                     \
        RP3_ST((ptrdiff_t)kp[n] * rs, xp);                                    \
        RP3_ST((ptrdiff_t)km[n] * rs, xm);                                    \
    }                                                                         \
}

/* conv form per m, measured (see strategy record): full unroll wins while
 * the spill traffic stays under the FMA plateau, the 8-accumulator blocked
 * form wins once it does not. */
RP2_DEFINE(3,  RP3_CONV)      /* p = 13 */
RP2_DEFINE(4,  RP3_CONV)      /* p = 17 */
RP2_DEFINE(7,  RP3_CONV)      /* p = 29 */
#ifdef RP2_BLK9
RP2_DEFINE(9,  RP2_CONV_BLK)
#else
RP2_DEFINE(9,  RP3_CONV)      /* p = 37 */
#endif
RP2_DEFINE(10, RP2_CONV_BLK)  /* p = 41 */
#undef RP2_BLK_IUNROLL
#define RP2_BLK_IUNROLL       /* p = 53: the gcc-unrolled i loop raced better
                               * (+1% rolled, 3/3) -- keep the r6 codegen here */
RP2_DEFINE(13, RP2_CONV_BLK)  /* p = 53 */
#undef RP2_BLK_IUNROLL
#define RP2_BLK_IUNROLL _Pragma("GCC unroll 1")
/* raced boundary (gen_r8): the fold/combine loops roll from here up */
#if !defined(RP_W1ROLL) && !defined(RP_W1FULL)
#undef RP2_FC1UNROLL
#define RP2_FC1UNROLL _Pragma("GCC unroll 1")
#endif
RP2_DEFINE(15, RP2_CONV_BLK)  /* p = 61 */
RP2_DEFINE(18, RP2_CONV_BLK)  /* p = 73 */
RP2_DEFINE(22, RP2_CONV_BLK)  /* p = 89 */
RP2_DEFINE(24, RP2_CONV_BLK)  /* p = 97 */
RP2_DEFINE(25, RP2_CONV_BLK)  /* p = 101 */
RP2_DEFINE(27, RP2_CONV_BLK)  /* p = 109 */
RP2_DEFINE(28, RP2_CONV_BLK)  /* p = 113 */

/* ---------------- PAIRED-COLUMN (2-wide) even-h kernels (gen_r8) ----------------
 * The full chunk kernel over TWO adjacent zmm columns (16 doubles = 8 complex
 * columns): identical per-column op order (=> bit-identical outputs to two
 * 1-wide calls), all stack slot arrays interleaved [slot][2].  What it buys:
 * every broadcast constant (conv kernels; and the fold/combine index walks)
 * is loaded ONCE per column PAIR -- see RP2_CONV_BLK2.  Only defined for the
 * blocked-conv family (m >= 10); the full-unroll small-m kernels are already
 * register-resident FMA-bound and doubling them would only add spills.
 * Full chunks only (no mask arm -- the pass tail runs the 1-wide kernel);
 * no map arm (dead since r3).  In-place safe: all loads precede all stores.
 *
 * Fold/reconstruct/combine loops are ROLLED by default (runtime index
 * tables, ~15-25 instr bodies): the gen_r8 node race showed the fully-peeled
 * 2-wide form (60 KB static at m=28) LOSES at every rp2 size (+6..28%),
 * while rolled it wins at m >= 22 (-5.5..-7% at 89/101/113); -DRP_W2FULL
 * restores the peeled form for the xarch race.
 * -DRP2_STORD: stores in natural row order (rows 1..h ascending paired with
 * p-1..p-h descending) with the slot permutation moved to the T/O_ READ side
 * and the +-SG selection folded into a two-constant table -- the twice-queued
 * store-order race, bit-identical by fmadd(o,-SG,T) == fnmadd(o,SG,T). */
#ifdef RP_W2FULL
#define RP2_FCUNROLL RP_UNROLL
#else
#define RP2_FCUNROLL _Pragma("GCC unroll 1")
#endif

#define RP2_DEFINE2(M)                                                        \
static void rp2_chunk2_##M(const double *sx, double *dx,                      \
                           const ptrdiff_t rs, const fft3d_plan *pl)          \
{                                                                             \
    enum { H_ = 2 * (M), K_ = 2 * (M) - 1, KS_ = K_ + 7, P_ = 4 * (M) + 1 };  \
    const __m512d SG = _mm512_setr_pd(1.0, -1.0, 1.0, -1.0,                   \
                                      1.0, -1.0, 1.0, -1.0);                  \
    const int *js = pl->j2, *jv = js + H_, *kp = jv + H_, *km = kp + H_;      \
    const double *kc = pl->k2, *kn = kc + KS_, *kb0 = kn + KS_,               \
                 *kb1 = kb0 + KS_, *kb2 = kb1 + KS_;                          \
    __m512d UF[2 * H_], VF[4 * H_], T[2 * H_], O_[2 * H_];                    \
    __m512d Q0[2 * (M)], Q1[2 * (M)], P0[2 * (M)], P1[2 * (M)], P2[2 * (M)];  \
    __m512d x0a = _mm512_loadu_pd(sx), x0b = _mm512_loadu_pd(sx + 8);         \
    RP2_FCUNROLL                                                              \
    for (int j_ = 1; j_ <= H_; ++j_) {                                        \
        const double *ra_ = sx + (ptrdiff_t)j_ * rs;                          \
        const double *rb_ = sx + (ptrdiff_t)(P_ - j_) * rs;                   \
        __m512d a0 = _mm512_loadu_pd(ra_), a1 = _mm512_loadu_pd(ra_ + 8);     \
        __m512d b0 = _mm512_loadu_pd(rb_), b1 = _mm512_loadu_pd(rb_ + 8);     \
        UF[2 * (j_ - 1)]     = _mm512_add_pd(a0, b0);                         \
        UF[2 * (j_ - 1) + 1] = _mm512_add_pd(a1, b1);                         \
        VF[2 * (j_ - 1)]     = _mm512_sub_pd(a0, b0);                         \
        VF[2 * (j_ - 1) + 1] = _mm512_sub_pd(a1, b1);                         \
        VF[2 * (H_ + j_ - 1)]     = _mm512_sub_pd(b0, a0);                    \
        VF[2 * (H_ + j_ - 1) + 1] = _mm512_sub_pd(b1, a1);                    \
    }                                                                         \
    __m512d esa = _mm512_setzero_pd(), esb = _mm512_setzero_pd();             \
    RP2_FCUNROLL                                                              \
    for (int j_ = 0; j_ < (M); ++j_) {                                        \
        const int lo_ = js[j_] - 1, hi_ = js[(M) + j_] - 1;                   \
        Q0[2 * j_]     = _mm512_add_pd(UF[2 * lo_], UF[2 * hi_]);             \
        Q0[2 * j_ + 1] = _mm512_add_pd(UF[2 * lo_ + 1], UF[2 * hi_ + 1]);     \
        Q1[2 * j_]     = _mm512_sub_pd(UF[2 * lo_], UF[2 * hi_]);             \
        Q1[2 * j_ + 1] = _mm512_sub_pd(UF[2 * lo_ + 1], UF[2 * hi_ + 1]);     \
        esa = _mm512_add_pd(esa, Q0[2 * j_]);                                 \
        esb = _mm512_add_pd(esb, Q0[2 * j_ + 1]);                             \
    }                                                                         \
    RP2_CONV_BLK2(P0, Q0, kc, M);                                             \
    RP2_CONV_BLK2(P1, Q1, kn, M);                                             \
    _mm512_storeu_pd(dx,     _mm512_add_pd(x0a, esa));                        \
    _mm512_storeu_pd(dx + 8, _mm512_add_pd(x0b, esb));                        \
    RP2_FCUNROLL                                                              \
    for (int j_ = 0; j_ < (M); ++j_) {                                        \
        T[2 * j_]     = _mm512_add_pd(x0a, _mm512_add_pd(P0[2 * j_],          \
                                                         P1[2 * j_]));        \
        T[2 * j_ + 1] = _mm512_add_pd(x0b, _mm512_add_pd(P0[2 * j_ + 1],      \
                                                         P1[2 * j_ + 1]));    \
        T[2 * ((M) + j_)]     = _mm512_add_pd(x0a,                            \
                                    _mm512_sub_pd(P0[2 * j_], P1[2 * j_]));   \
        T[2 * ((M) + j_) + 1] = _mm512_add_pd(x0b,                            \
                                    _mm512_sub_pd(P0[2 * j_ + 1],             \
                                                  P1[2 * j_ + 1]));           \
    }                                                                         \
    RP2_FCUNROLL                                                              \
    for (int j_ = 0; j_ < (M); ++j_) {                                        \
        const int e_ = jv[2 * j_], o_ = jv[2 * j_ + 1];                       \
        Q0[2 * j_]     = VF[2 * e_];                                          \
        Q0[2 * j_ + 1] = VF[2 * e_ + 1];                                      \
        Q1[2 * j_]     = VF[2 * o_];                                          \
        Q1[2 * j_ + 1] = VF[2 * o_ + 1];                                      \
    }                                                                         \
    RP2_CONV_BLK2(P0, Q0, kb0, M);                                            \
    RP2_CONV_BLK2(P1, Q1, kb1, M);                                            \
    RP2_FCUNROLL                                                              \
    for (int j_ = 0; j_ < 2 * (M); ++j_)                                      \
        Q0[j_] = _mm512_add_pd(Q0[j_], Q1[j_]);                               \
    RP2_CONV_BLK2(P2, Q0, kb2, M);                                            \
    O_[0] = _mm512_sub_pd(P0[0], P1[2 * ((M) - 1)]);                          \
    O_[1] = _mm512_sub_pd(P0[1], P1[2 * ((M) - 1) + 1]);                      \
    RP2_FCUNROLL                                                              \
    for (int j_ = 1; j_ < (M); ++j_) {                                        \
        O_[2 * (2 * j_)]     = _mm512_add_pd(P0[2 * j_], P1[2 * (j_ - 1)]);   \
        O_[2 * (2 * j_) + 1] = _mm512_add_pd(P0[2 * j_ + 1],                  \
                                             P1[2 * (j_ - 1) + 1]);           \
    }                                                                         \
    RP2_FCUNROLL                                                              \
    for (int j_ = 0; j_ < (M); ++j_) {                                        \
        O_[2 * (2 * j_ + 1)]     = _mm512_sub_pd(P2[2 * j_],                  \
                          _mm512_add_pd(P0[2 * j_], P1[2 * j_]));             \
        O_[2 * (2 * j_ + 1) + 1] = _mm512_sub_pd(P2[2 * j_ + 1],              \
                          _mm512_add_pd(P0[2 * j_ + 1], P1[2 * j_ + 1]));     \
    }                                                                         \
    RP2_W2_COMBINE(M)                                                         \
}

#ifndef RP2_NOSTORD
/* natural-row-order stores (DEFAULT since the gen_r8 race: best 3/3 at 113,
 * ties 89/101/127 -- never worse; -DRP2_NOSTORD restores scattered stores):
 * st2[n] (built in rp2_build) = source slot for output row n+1, bit 16 =
 * "row n+1 took the km side" => flip SG.  fmadd with -SG is the exact
 * negation of the SG product, so results stay bit-identical to the
 * kp/km-scattered order. */
#define RP2_W2_COMBINE(M)                                                     \
    const int *st2 = js + 4 * H_;                                             \
    const __m512d SGN_ = _mm512_sub_pd(_mm512_setzero_pd(), SG);              \
    RP2_FCUNROLL                                                              \
    for (int n = 0; n < H_; ++n) {                                            \
        const int e_ = st2[n], s_ = e_ & 0xffff;                              \
        const __m512d sgv = (e_ >> 16) ? SGN_ : SG;                           \
        __m512d oa = _mm512_permute_pd(O_[2 * s_], 0x55);                     \
        __m512d ob = _mm512_permute_pd(O_[2 * s_ + 1], 0x55);                 \
        const ptrdiff_t rl_ = (ptrdiff_t)(n + 1) * rs;                        \
        const ptrdiff_t rh_ = (ptrdiff_t)(P_ - 1 - n) * rs;                   \
        _mm512_storeu_pd(dx + rl_,     _mm512_fmadd_pd(oa, sgv, T[2 * s_]));  \
        _mm512_storeu_pd(dx + rl_ + 8,                                        \
                         _mm512_fmadd_pd(ob, sgv, T[2 * s_ + 1]));            \
        _mm512_storeu_pd(dx + rh_,     _mm512_fnmadd_pd(oa, sgv, T[2 * s_])); \
        _mm512_storeu_pd(dx + rh_ + 8,                                        \
                         _mm512_fnmadd_pd(ob, sgv, T[2 * s_ + 1]));           \
    }
#else
#define RP2_W2_COMBINE(M)                                                     \
    RP2_FCUNROLL                                                              \
    for (int n = 0; n < H_; ++n) {                                            \
        __m512d oa = _mm512_permute_pd(O_[2 * n], 0x55);                      \
        __m512d ob = _mm512_permute_pd(O_[2 * n + 1], 0x55);                  \
        const ptrdiff_t rp_ = (ptrdiff_t)kp[n] * rs;                          \
        const ptrdiff_t rm_ = (ptrdiff_t)km[n] * rs;                          \
        _mm512_storeu_pd(dx + rp_,     _mm512_fmadd_pd(oa, SG, T[2 * n]));    \
        _mm512_storeu_pd(dx + rp_ + 8,                                        \
                         _mm512_fmadd_pd(ob, SG, T[2 * n + 1]));              \
        _mm512_storeu_pd(dx + rm_,     _mm512_fnmadd_pd(oa, SG, T[2 * n]));   \
        _mm512_storeu_pd(dx + rm_ + 8,                                        \
                         _mm512_fnmadd_pd(ob, SG, T[2 * n + 1]));             \
    }
#endif

RP2_DEFINE2(10)  /* p = 41 */
RP2_DEFINE2(13)  /* p = 53 */
RP2_DEFINE2(15)  /* p = 61 */
RP2_DEFINE2(18)  /* p = 73 */
RP2_DEFINE2(22)  /* p = 89 */
RP2_DEFINE2(24)  /* p = 97 */
RP2_DEFINE2(25)  /* p = 101 */
RP2_DEFINE2(27)  /* p = 109 */
RP2_DEFINE2(28)  /* p = 113 */

/* ---------------- PAIRED-COLUMN (2-wide) outer-C3 kernel (gen_r9) ----------------
 * The gen_r8 pairing applied to the RP3_WINO family -- the r8 next-step-1
 * item.  Only m = 17 (p = 103): it is the one rp3 prime on the BLOCKED conv
 * (whose 9-load/8-FMA tile with exactly-8 accumulator chains is what pairing
 * relieves: per i, 2 stack loads + 8 broadcasts + 16 FMA, 16 chains), and the
 * only DRAM-resident one (35 MB state+c), the regime where pairing won at
 * every dense size (113/127) and rp2 size (101/113).  The full-unroll rp3
 * kernels (m = 7/11/13) stay 1-wide: they are register-resident FMA-bound and
 * pairing them would only add spills (the r8 m=10..18 rp2 lesson).
 * Stack: S/Y/T [2H] + E/A/B/C/M0..M3 [2M] = 578 zmm slots = 37 KB -- smaller
 * than the shipped rp2_chunk2_28's 54 KB.  Fold/combine/WINO glue loops
 * ROLLED (runtime index tables; the r8 front-end doctrine at this size).
 * Per-column op order identical to rp3_chunk_17 => outputs bit-identical.
 * In-place safe: only row-0 store precedes the O-fold loads, and jdp/jdm
 * never index row 0.
 * DEFAULT OFF (-DRP_W23 enables): raced on wallaby -- which is a Gold 6448Y
 * SAPPHIRE RAPIDS, one of the graded xarch machines -- and LOST +35% 3/3
 * (9.59 vs 7.06 ms/step; RP_PROF: x +29%, y +35%, z +74%).  Calibration on
 * the same host: the node-proven rp2 2-wide win at 113 reads only +2.8%
 * here, so the generic dev-host anti-pairing bias (~3-11 points) cannot
 * explain 35.  Mechanism: rp3's glue/conv ratio is ~2x worse than rp2's
 * (8 conv calls of m^2=289 vs 5 of 784, glue scaling with H=51), so the
 * pairing's doubled stack arrays cost more than the shared broadcasts buy.
 * The Ice Lake verdict is unmeasured (nodes queued-busy all round); the
 * race is staged in r9_ab.sh -- adopt only on a node win. */
#define RP3_WINO2(KT, M, WANT_ESUM) do {                                      \
    RP2_FCUNROLL                                                              \
    for (int j_ = 0; j_ < (M); ++j_) {                                        \
        E[2 * j_] = _mm512_add_pd(_mm512_add_pd(S[2 * j_],                    \
                                                S[2 * ((M) + j_)]),           \
                                  S[2 * (2 * (M) + j_)]);                     \
        E[2 * j_ + 1] = _mm512_add_pd(_mm512_add_pd(S[2 * j_ + 1],            \
                                                    S[2 * ((M) + j_) + 1]),   \
                                      S[2 * (2 * (M) + j_) + 1]);             \
        A[2 * j_] = _mm512_sub_pd(S[2 * j_], S[2 * (2 * (M) + j_)]);          \
        A[2 * j_ + 1] = _mm512_sub_pd(S[2 * j_ + 1],                          \
                                      S[2 * (2 * (M) + j_) + 1]);             \
        B[2 * j_] = _mm512_sub_pd(S[2 * ((M) + j_)], S[2 * (2 * (M) + j_)]);  \
        B[2 * j_ + 1] = _mm512_sub_pd(S[2 * ((M) + j_) + 1],                  \
                                      S[2 * (2 * (M) + j_) + 1]);             \
        C[2 * j_] = _mm512_sub_pd(S[2 * j_], S[2 * ((M) + j_)]);              \
        C[2 * j_ + 1] = _mm512_sub_pd(S[2 * j_ + 1], S[2 * ((M) + j_) + 1]);  \
        if (WANT_ESUM) {                                                      \
            esa = _mm512_add_pd(esa, E[2 * j_]);                              \
            esb = _mm512_add_pd(esb, E[2 * j_ + 1]);                          \
        }                                                                     \
    }                                                                         \
    RP2_CONV_BLK2(M0, E, (KT), M);                                            \
    RP2_CONV_BLK2(M1, A, (KT) + (2 * (M) - 1), M);                            \
    RP2_CONV_BLK2(M2, B, (KT) + 2 * (2 * (M) - 1), M);                        \
    RP2_CONV_BLK2(M3, C, (KT) + 3 * (2 * (M) - 1), M);                        \
    RP2_FCUNROLL                                                              \
    for (int j_ = 0; j_ < (M); ++j_) {                                        \
        __m512d t01a = _mm512_add_pd(M0[2 * j_], M1[2 * j_]);                 \
        __m512d t01b = _mm512_add_pd(M0[2 * j_ + 1], M1[2 * j_ + 1]);         \
        Y[2 * j_] = _mm512_fnmadd_pd(M2[2 * j_], TWO,                         \
                                     _mm512_add_pd(t01a, M3[2 * j_]));        \
        Y[2 * j_ + 1] = _mm512_fnmadd_pd(M2[2 * j_ + 1], TWO,                 \
                                     _mm512_add_pd(t01b, M3[2 * j_ + 1]));    \
        Y[2 * ((M) + j_)] = _mm512_fnmadd_pd(M3[2 * j_], TWO,                 \
                                     _mm512_add_pd(t01a, M2[2 * j_]));        \
        Y[2 * ((M) + j_) + 1] = _mm512_fnmadd_pd(M3[2 * j_ + 1], TWO,         \
                                     _mm512_add_pd(t01b, M2[2 * j_ + 1]));    \
        __m512d t23a = _mm512_add_pd(M2[2 * j_], M3[2 * j_]);                 \
        __m512d t23b = _mm512_add_pd(M2[2 * j_ + 1], M3[2 * j_ + 1]);         \
        Y[2 * (2 * (M) + j_)] = _mm512_fnmadd_pd(M1[2 * j_], TWO,             \
                                     _mm512_add_pd(M0[2 * j_], t23a));        \
        Y[2 * (2 * (M) + j_) + 1] = _mm512_fnmadd_pd(M1[2 * j_ + 1], TWO,     \
                                     _mm512_add_pd(M0[2 * j_ + 1], t23b));    \
    }                                                                         \
} while (0)

#define RP3_DEFINE2(M)                                                        \
static void rp3_chunk2_##M(const double *sx, double *dx,                      \
                           const ptrdiff_t rs, const fft3d_plan *pl)          \
{                                                                             \
    enum { H_ = 3 * (M) };                                                    \
    const __m512d SG = _mm512_setr_pd(1.0, -1.0, 1.0, -1.0,                   \
                                      1.0, -1.0, 1.0, -1.0);                  \
    const __m512d TWO = _mm512_set1_pd(2.0);                                  \
    const int *js = pl->j3, *js2 = js + H_, *jdp = js2 + H_,                  \
              *jdm = jdp + H_, *kp = jdm + H_, *km = kp + H_;                 \
    __m512d S[2 * H_], Y[2 * H_], T[2 * H_];                                  \
    __m512d E[2 * (M)], A[2 * (M)], B[2 * (M)], C[2 * (M)];                   \
    __m512d M0[2 * (M)], M1[2 * (M)], M2[2 * (M)], M3[2 * (M)];               \
    __m512d x0a = _mm512_loadu_pd(sx), x0b = _mm512_loadu_pd(sx + 8);         \
    __m512d esa = _mm512_setzero_pd(), esb = _mm512_setzero_pd();             \
    RP2_FCUNROLL                                                              \
    for (int n = 0; n < H_; ++n) {                                            \
        const double *ra_ = sx + (ptrdiff_t)js[n] * rs;                       \
        const double *rb_ = sx + (ptrdiff_t)js2[n] * rs;                      \
        S[2 * n] = _mm512_add_pd(_mm512_loadu_pd(ra_),                        \
                                 _mm512_loadu_pd(rb_));                       \
        S[2 * n + 1] = _mm512_add_pd(_mm512_loadu_pd(ra_ + 8),                \
                                     _mm512_loadu_pd(rb_ + 8));               \
    }                                                                         \
    RP3_WINO2(pl->k3e, M, 1);                                                 \
    _mm512_storeu_pd(dx,     _mm512_add_pd(x0a, esa));                        \
    _mm512_storeu_pd(dx + 8, _mm512_add_pd(x0b, esb));                        \
    RP2_FCUNROLL                                                              \
    for (int n = 0; n < H_; ++n) {                                            \
        T[2 * n] = _mm512_add_pd(x0a, Y[2 * n]);                              \
        T[2 * n + 1] = _mm512_add_pd(x0b, Y[2 * n + 1]);                      \
    }                                                                         \
    RP2_FCUNROLL                                                              \
    for (int n = 0; n < H_; ++n) {                                            \
        const double *ra_ = sx + (ptrdiff_t)jdp[n] * rs;                      \
        const double *rb_ = sx + (ptrdiff_t)jdm[n] * rs;                      \
        S[2 * n] = _mm512_sub_pd(_mm512_loadu_pd(ra_),                        \
                                 _mm512_loadu_pd(rb_));                       \
        S[2 * n + 1] = _mm512_sub_pd(_mm512_loadu_pd(ra_ + 8),                \
                                     _mm512_loadu_pd(rb_ + 8));               \
    }                                                                         \
    RP3_WINO2(pl->k3o, M, 0);                                                 \
    RP2_FCUNROLL                                                              \
    for (int n = 0; n < H_; ++n) {                                            \
        __m512d oa = _mm512_permute_pd(Y[2 * n], 0x55);                       \
        __m512d ob = _mm512_permute_pd(Y[2 * n + 1], 0x55);                   \
        const ptrdiff_t rp_ = (ptrdiff_t)kp[n] * rs;                          \
        const ptrdiff_t rm_ = (ptrdiff_t)km[n] * rs;                          \
        _mm512_storeu_pd(dx + rp_,     _mm512_fmadd_pd(oa, SG, T[2 * n]));    \
        _mm512_storeu_pd(dx + rp_ + 8,                                        \
                         _mm512_fmadd_pd(ob, SG, T[2 * n + 1]));              \
        _mm512_storeu_pd(dx + rm_,     _mm512_fnmadd_pd(oa, SG, T[2 * n]));   \
        _mm512_storeu_pd(dx + rm_ + 8,                                        \
                         _mm512_fnmadd_pd(ob, SG, T[2 * n + 1]));             \
    }                                                                         \
}

#ifdef RP_W23
RP3_DEFINE2(17)  /* p = 103 */
#endif

/* chunk dispatch: even-h split / outer-C3 when the plan built tables, dense
 * otherwise (m2 and m3 are mutually exclusive by h's parity) */
static inline void rp_chunk_any(const double *sx, double *dx, const double *mapc,
                                const ptrdiff_t rs, const fft3d_plan *pl,
                                int full, __mmask8 msk)
{
    switch (pl->m2) {
    case 3:  rp2_chunk_3(sx, dx, mapc, rs, pl, full, msk);  return;
    case 4:  rp2_chunk_4(sx, dx, mapc, rs, pl, full, msk);  return;
    case 7:  rp2_chunk_7(sx, dx, mapc, rs, pl, full, msk);  return;
    case 9:  rp2_chunk_9(sx, dx, mapc, rs, pl, full, msk);  return;
    case 10: rp2_chunk_10(sx, dx, mapc, rs, pl, full, msk); return;
    case 13: rp2_chunk_13(sx, dx, mapc, rs, pl, full, msk); return;
    case 15: rp2_chunk_15(sx, dx, mapc, rs, pl, full, msk); return;
    case 18: rp2_chunk_18(sx, dx, mapc, rs, pl, full, msk); return;
    case 22: rp2_chunk_22(sx, dx, mapc, rs, pl, full, msk); return;
    case 24: rp2_chunk_24(sx, dx, mapc, rs, pl, full, msk); return;
    case 25: rp2_chunk_25(sx, dx, mapc, rs, pl, full, msk); return;
    case 27: rp2_chunk_27(sx, dx, mapc, rs, pl, full, msk); return;
    case 28: rp2_chunk_28(sx, dx, mapc, rs, pl, full, msk); return;
    default: break;
    }
    switch (pl->m3) {
    case 7:  rp3_chunk_7(sx, dx, mapc, rs, pl, full, msk);  return;
    case 11: rp3_chunk_11(sx, dx, mapc, rs, pl, full, msk); return;
    case 13: rp3_chunk_13(sx, dx, mapc, rs, pl, full, msk); return;
    case 17: rp3_chunk_17(sx, dx, mapc, rs, pl, full, msk); return;
    default:
        rp_chunk(sx, dx, mapc, rs, pl->L, pl->h, pl->gct, pl->gst, full, msk);
    }
}

/* gen_r8 pairing gates.  RP_NOW2 disables all pairing (control == the r7
 * engine bit for bit).  RP_W2MIN: smallest m2 that runs the 2-wide kernel.
 * Node-raced (3/3 per size, same-core interleaved, two sessions): the
 * 2-wide+rolled form wins vs the r7 control at m >= 22 (89 -5.5%, 101 -6%,
 * 113 -7%) -- but the second session's attribution race showed the 1-WIDE
 * kernel with the SAME rolled fold/combine beats the 2-wide at 89 (-3..-8%)
 * and ties it at 101, while sd (2-wide + store-order) is best 3/3 at 113.
 * gen_r10: the r8 interpolation "97 = m24 sides with 89" was WRONG -- the
 * staged ICL race (r9_ab.sh race97) reads 2-wide 11.07-11.21 vs 1-wide
 * 11.62-11.74 ms/step, -4..6%, 3/3 non-overlapping mins.  Boundary is now
 * RACED at every rp2 prime: 1-wide rolled at m = 15..22 (61/73/89), 2-wide+
 * stord at m >= 24 (97/101/109/113; no prime has m = 23).  RP_NOW2D: dense
 * engine stays 1-wide (dense pairing measured -25.5% at 127 -- the r8
 * round's biggest win; the paired z-quad alone carries ~10% of it, RP_NOW2Z
 * raced +11% at 127).  RP_NOW2Z: z quads stay 1-wide. */
#ifndef RP_W2MIN
#define RP_W2MIN 24
#endif
static inline int rp_w2_on(const fft3d_plan *pl)
{
#ifdef RP_NOW2
    return 0;
#else
    if (pl->m2 >= RP_W2MIN) return 1;
#ifdef RP_W23
    if (pl->m3 == 17) return 1;            /* gen_r9/r10: outer-C3 pairing at
                                            * p=103 -- CLOSED, loses on BOTH
                                            * graded architectures (ICL +13%
                                            * 3/3 r10, SPR +35% 3/3 r9); knob
                                            * kept only as a CLX curiosity */
#endif
#ifndef RP_NOW2D
    if (!pl->m2 && !pl->m3) return 1;      /* dense engine */
#endif
    return 0;
#endif
}

/* gen_r12 4-wide gate: dense engine only (rp2/rp3 stacks already exceed L1D
 * at large m -- the r10 residue -- and rp3 pairing is closed on both graded
 * architectures).  Node-raced boundary (a80n0, rotated interleaved rounds):
 * WINS 127 -12.2% (4/4), 107 -9.6% (4/4), 83 -8.6% (3/3), 59 -5.7% (3/3),
 * 71 -4..7% (2/3 + wash); WASH at 23/47 (overlapping min-sets) -- so 59 is
 * the default.  PMU attribution at 127: port_2_3 uops -20%/step with l1d
 * fills near-flat: the win is broadcast-dispatch deletion, the table-walk
 * fills were latency-covered behind the 16 chains (gen_layout r12's
 * accumulator doctrine confirmed from the winning side).  RP_NOW4 restores
 * the r11 dispatch. */
#ifndef RP_W4MIN
#define RP_W4MIN 59
#endif
static inline int rp_w4_on(const fft3d_plan *pl)
{
#ifdef RP_NOW4
    (void)pl;
    return 0;
#else
    return !pl->m2 && !pl->m3 && pl->L >= RP_W4MIN && rp_w2_on(pl);
#endif
}

/* one PAIR of adjacent full chunks (16 doubles).  For plans without a 2-wide
 * kernel this decays to two 1-wide calls -- identical to the r7 path. */
static inline void rp_chunk_any2(const double *sx, double *dx,
                                 const ptrdiff_t rs, const fft3d_plan *pl)
{
    switch (pl->m2) {
    case 10: rp2_chunk2_10(sx, dx, rs, pl); return;
    case 13: rp2_chunk2_13(sx, dx, rs, pl); return;
    case 15: rp2_chunk2_15(sx, dx, rs, pl); return;
    case 18: rp2_chunk2_18(sx, dx, rs, pl); return;
    case 22: rp2_chunk2_22(sx, dx, rs, pl); return;
    case 24: rp2_chunk2_24(sx, dx, rs, pl); return;
    case 25: rp2_chunk2_25(sx, dx, rs, pl); return;
    case 27: rp2_chunk2_27(sx, dx, rs, pl); return;
    case 28: rp2_chunk2_28(sx, dx, rs, pl); return;
    default: break;
    }
#ifdef RP_W23
    if (pl->m3 == 17) { rp3_chunk2_17(sx, dx, rs, pl); return; }
#endif
    if (!pl->m2 && !pl->m3) {
        rp_chunk2(sx, dx, rs, pl->L, pl->h, pl->gct, pl->gst);
        return;
    }
    rp_chunk_any(sx, dx, NULL, rs, pl, 1, (__mmask8)0xFF);
    rp_chunk_any(sx + 8, dx + 8, NULL, rs, pl, 1, (__mmask8)0xFF);
}

/* one p x ncols pass, rows `pitch` complex apart; in-place safe (dst may ==
 * src); c != NULL fuses the map at every store (c rows at the same pitch).
 * pf != 0: software-prefetch every row's line `pf` BYTES ahead of the current
 * chunk's loads (gen_layout gen_r4's fold-prefetch recipe).  The x-pass at
 * DRAM-resident p walks p+1 concurrent row streams at plane-pitch stride --
 * far past what the L2 streamer tracks (~32) -- so most rows' next line is a
 * demand miss every chunk; T0 prefetch issued a chunk early hides it on the
 * load ports the conv leaves partly idle.  At 31 the same recipe LOST (+0.7%,
 * gen_r4): the state there is L2-resident.  Off for y/z (row count is the
 * same but the streams live inside one L2-resident plane). */
/* gen_r12: x-pass prefetch hint class.  The default stays T0 (the r7-raced
 * form).  gen_layout gen_r12's rule: on a DRAM-resident stream a T0 prefetch
 * occupies an L1 fill buffer (~12 on ICX) for the full DRAM latency, so p+1
 * concurrent row streams serialize through the LFB pool; T1 fills L2 through
 * the deeper superqueue instead.  -DRP_PFT1 races the T1 form. */
#ifdef RP_PFT1
#define RP_PFHINT _MM_HINT_T1
#else
#define RP_PFHINT _MM_HINT_T0
#endif

static void rp_pass(const fft3d_plan *pl, const cplx *src, cplx *dst,
                    const cplx *c, const ptrdiff_t ncols, const ptrdiff_t pitch,
                    const int pf)
{
    const ptrdiff_t rs = 2 * pitch, nd = 2 * ncols;
    const int p = pl->L;
    const double *sx = (const double *)src;
    const double *cx = (const double *)c;
    double *dx = (double *)dst;
    ptrdiff_t d = 0;
    /* gen_r12: 4-wide dense main loop with the E/O phase split */
    if (!cx && rp_w4_on(pl)) {
        for (; d + 32 <= nd; d += 32) {
            if (pf)
                for (int j = 0; j <= p - 1; ++j) {
                    const char *pr =
                        (const char *)(sx + (ptrdiff_t)j * rs + d) + 2 * pf;
                    _mm_prefetch(pr, RP_PFHINT);        /* next 4-wide chunk: */
                    _mm_prefetch(pr + 64, RP_PFHINT);   /* four lines per row */
                    _mm_prefetch(pr + 128, RP_PFHINT);
                    _mm_prefetch(pr + 192, RP_PFHINT);
                }
            rp_chunk4(sx + d, dx + d, rs, pl->L, pl->h, pl->gct, pl->gst);
        }
    }
    /* gen_r8: paired-column main loop (map-fused arms stay 1-wide) */
    if (!cx && rp_w2_on(pl)) {
        for (; d + 16 <= nd; d += 16) {
            if (pf)
                for (int j = 0; j <= p - 1; ++j) {
                    const char *pr =
                        (const char *)(sx + (ptrdiff_t)j * rs + d) + pf;
                    _mm_prefetch(pr, RP_PFHINT);        /* next pair chunk: */
                    _mm_prefetch(pr + 64, RP_PFHINT);   /* two lines per row */
                }
            rp_chunk_any2(sx + d, dx + d, rs, pl);
        }
    }
    for (; d + 8 <= nd; d += 8) {
        if (pf)
            for (int j = 0; j <= p - 1; ++j)
                _mm_prefetch((const char *)(sx + (ptrdiff_t)j * rs + d) + pf,
                             RP_PFHINT);
        rp_chunk_any(sx + d, dx + d, cx ? cx + d : NULL, rs, pl,
                     1, (__mmask8)0xFF);
    }
    if (d < nd)
        rp_chunk_any(sx + d, dx + d, cx ? cx + d : NULL, rs, pl,
                     0, (__mmask8)((1u << (nd - d)) - 1));
}

/* z-axis: up to 4 contiguous rows via 4x4-complex transposes into a stack
 * pencil array, the SAME chunk kernel at rs = 8, transpose back (r31_zquad
 * generalized to runtime p and a row count 1..4).  All loads precede all
 * stores => in-place safe. */
static void rp_zquad(const fft3d_plan *pl, const cplx *src, cplx *dst,
                     int nrows, const ptrdiff_t srd, const ptrdiff_t drd)
{
    const int p = pl->L;
    /* 4*ceil(p/4) <= 128 pencil rows of 8 doubles each */
    __attribute__((aligned(64))) double xt[128 * 8], yt[128 * 8];
    const int nt = (p + 3) / 4;            /* 4-complex tiles per row */
    const int tail = p & 3;                /* complex in the last tile (0 = full) */
    const __mmask8 tmsk = tail ? (__mmask8)((1u << (2 * tail)) - 1) : (__mmask8)0xFF;
    const double *x = (const double *)src;
    double *y = (double *)dst;
    for (int t = 0; t < nt; ++t) {
        const int ft = (t < nt - 1) || !tail;
        const __mmask8 mk = ft ? (__mmask8)0xFF : tmsk;
        __m512d r0 = _mm512_setzero_pd(), r1 = r0, r2 = r0, r3 = r0;
        r0 = _mm512_maskz_loadu_pd(mk, x + 8 * t);
        if (nrows > 1) r1 = _mm512_maskz_loadu_pd(mk, x + srd + 8 * t);
        if (nrows > 2) r2 = _mm512_maskz_loadu_pd(mk, x + 2 * srd + 8 * t);
        if (nrows > 3) r3 = _mm512_maskz_loadu_pd(mk, x + 3 * srd + 8 * t);
        __m512d o0, o1, o2, o3;
        r31_tp4(r0, r1, r2, r3, &o0, &o1, &o2, &o3);
        _mm512_store_pd(xt + (4 * t) * 8, o0);
        _mm512_store_pd(xt + (4 * t + 1) * 8, o1);
        _mm512_store_pd(xt + (4 * t + 2) * 8, o2);
        _mm512_store_pd(xt + (4 * t + 3) * 8, o3);
    }
    rp_chunk_any(xt, yt, NULL, 8, pl, 1, (__mmask8)0xFF);
    for (int t = 0; t < nt; ++t) {
        const int ft = (t < nt - 1) || !tail;
        const __mmask8 mk = ft ? (__mmask8)0xFF : tmsk;
        __m512d o0, o1, o2, o3;
        /* rows >= p of yt are never written by rp_chunk; their lanes land in
         * masked-out columns (shuffles only -- no arithmetic on them) */
        r31_tp4(_mm512_load_pd(yt + (4 * t) * 8),
                _mm512_load_pd(yt + (4 * t + 1) * 8),
                _mm512_load_pd(yt + (4 * t + 2) * 8),
                _mm512_load_pd(yt + (4 * t + 3) * 8),
                &o0, &o1, &o2, &o3);
        _mm512_mask_storeu_pd(y + 8 * t, mk, o0);
        if (nrows > 1) _mm512_mask_storeu_pd(y + drd + 8 * t, mk, o1);
        if (nrows > 2) _mm512_mask_storeu_pd(y + 2 * drd + 8 * t, mk, o2);
        if (nrows > 3) _mm512_mask_storeu_pd(y + 3 * drd + 8 * t, mk, o3);
    }
}

/* gen_r8: EIGHT contiguous rows through one paired chunk -- two 4-row
 * transpose groups staged interleaved (element j of group g at
 * xt + j*16 + 8g), the 2-wide kernel at rs = 16, transpose back.  Same
 * per-column arithmetic and transposes as two rp_zquad calls =>
 * bit-identical.  Full 8-row groups only; tails run rp_zquad. */
static void rp_zquad2(const fft3d_plan *pl, const cplx *src, cplx *dst,
                      const ptrdiff_t srd, const ptrdiff_t drd)
{
    const int p = pl->L;
    __attribute__((aligned(64))) double xt[128 * 16], yt[128 * 16];
    const int nt = (p + 3) / 4;
    const int tail = p & 3;
    const __mmask8 tmsk = tail ? (__mmask8)((1u << (2 * tail)) - 1) : (__mmask8)0xFF;
    const double *x = (const double *)src;
    double *y = (double *)dst;
    for (int g = 0; g < 2; ++g) {
        const double *xg = x + 4 * g * srd;
        for (int t = 0; t < nt; ++t) {
            const int ft = (t < nt - 1) || !tail;
            const __mmask8 mk = ft ? (__mmask8)0xFF : tmsk;
            __m512d r0 = _mm512_maskz_loadu_pd(mk, xg + 8 * t);
            __m512d r1 = _mm512_maskz_loadu_pd(mk, xg + srd + 8 * t);
            __m512d r2 = _mm512_maskz_loadu_pd(mk, xg + 2 * srd + 8 * t);
            __m512d r3 = _mm512_maskz_loadu_pd(mk, xg + 3 * srd + 8 * t);
            __m512d o0, o1, o2, o3;
            r31_tp4(r0, r1, r2, r3, &o0, &o1, &o2, &o3);
            _mm512_store_pd(xt + (4 * t) * 16 + 8 * g, o0);
            _mm512_store_pd(xt + (4 * t + 1) * 16 + 8 * g, o1);
            _mm512_store_pd(xt + (4 * t + 2) * 16 + 8 * g, o2);
            _mm512_store_pd(xt + (4 * t + 3) * 16 + 8 * g, o3);
        }
    }
    rp_chunk_any2(xt, yt, 16, pl);
    for (int g = 0; g < 2; ++g) {
        double *yg = y + 4 * g * drd;
        for (int t = 0; t < nt; ++t) {
            const int ft = (t < nt - 1) || !tail;
            const __mmask8 mk = ft ? (__mmask8)0xFF : tmsk;
            __m512d o0, o1, o2, o3;
            r31_tp4(_mm512_load_pd(yt + (4 * t) * 16 + 8 * g),
                    _mm512_load_pd(yt + (4 * t + 1) * 16 + 8 * g),
                    _mm512_load_pd(yt + (4 * t + 2) * 16 + 8 * g),
                    _mm512_load_pd(yt + (4 * t + 3) * 16 + 8 * g),
                    &o0, &o1, &o2, &o3);
            _mm512_mask_storeu_pd(yg + 8 * t, mk, o0);
            _mm512_mask_storeu_pd(yg + drd + 8 * t, mk, o1);
            _mm512_mask_storeu_pd(yg + 2 * drd + 8 * t, mk, o2);
            _mm512_mask_storeu_pd(yg + 3 * drd + 8 * t, mk, o3);
        }
    }
}

/* z gating: pairing the quads doubles the transpose staging (32 KB) on top
 * of the kernel's slot arrays -- raceable separately from the x/y pairing */
static inline int rp_w2z_on(const fft3d_plan *pl)
{
#ifdef RP_NOW2Z
    (void)pl;
    return 0;
#else
    return rp_w2_on(pl);
#endif
}

/* ---------------- gen_r13: compile-time p=13 execute engine ----------------
 * The benchFFT round exposed B=1 small-L as the library's weak regime; probing
 * the class found the SAME hole at the tiny primes: MKL beat this entry at
 * every p <= 13 at B=1 (13: 8.4 vs 6.1 us) while losing 3-10x at p >= 17.
 * Skip-knob timing at 13 put each pass at ~3x its uop model: at m = 3 the
 * runtime-table rp2 machinery is all fixed cost -- switch dispatch + call per
 * 4 columns, plan-struct and kernel-table pointer reloads per call, runtime
 * js/jv/kp/km index loads and j*rs address arithmetic that cannot hoist out
 * of the column loop.  This is the r31 lesson in miniature: the fixed-size
 * engines won by making every index and stride a compile-time constant.
 * rp13_* is the rp2 m=3 arithmetic (identical op order, so outputs match
 * rp2_chunk_3 bitwise) with the p=13 Rader tables hardcoded (verified against
 * rp2_build at create(); rp13 only runs when they match), strides/masks/trip
 * counts compile-time, chunks force-inlined into the pass loops.  Kernel
 * DOUBLES stay in pl->k2 (broadcast cost is address-independent; no float
 * literals to transcribe).  Execute path only; the chain keeps rp_*. */
static const int RP13_TAB[24] = {
    1, 6, 3, 5, 4, 2,        /* js: U fold indices, reversal baked in   */
    0, 11, 8, 4, 9, 7,       /* jv: signed V indices (mirror half >= 6) */
    1, 2, 4, 8, 3, 6,        /* kp: X_k output rows                     */
    12, 11, 9, 5, 10, 7      /* km: X_{p-k} output rows                 */
};

static inline __attribute__((always_inline))
void rp13_chunk(const double *sx, double *dx, const ptrdiff_t rs,
                const double *k2, const int full, const __mmask8 msk)
{
    enum { M_ = 3, H_ = 6, KS_ = 12 };
    const int *js = RP13_TAB, *jv = js + H_, *kp = jv + H_, *km = kp + H_;
    const __m512d SG = _mm512_setr_pd(1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0);
    const double *kc = k2, *kn = kc + KS_, *kb0 = kn + KS_,
                 *kb1 = kb0 + KS_, *kb2 = kb1 + KS_;
    __m512d UF[H_], VF[2 * H_], T[H_], O_[H_];
    __m512d Q0[M_], Q1[M_], P0[M_], P1[M_], P2[M_];
    __m512d x0 = RP3_LD(sx, 0);
    RP_UNROLL
    for (int j_ = 1; j_ <= H_; ++j_) {
        __m512d a_ = RP3_LD(sx, (ptrdiff_t)j_ * rs);
        __m512d b_ = RP3_LD(sx, (ptrdiff_t)(13 - j_) * rs);
        UF[j_ - 1] = _mm512_add_pd(a_, b_);
        VF[j_ - 1] = _mm512_sub_pd(a_, b_);
        VF[H_ + j_ - 1] = _mm512_sub_pd(b_, a_);
    }
    __m512d esum = _mm512_setzero_pd();
    RP_UNROLL
    for (int j_ = 0; j_ < M_; ++j_) {
        __m512d slo = UF[js[j_] - 1], shi = UF[js[M_ + j_] - 1];
        Q0[j_] = _mm512_add_pd(slo, shi);
        Q1[j_] = _mm512_sub_pd(slo, shi);
        esum = _mm512_add_pd(esum, Q0[j_]);
    }
    RP3_CONV(P0, Q0, kc, M_);
    RP3_CONV(P1, Q1, kn, M_);
    __m512d X0 = _mm512_add_pd(x0, esum);
    RP3_ST(0, X0);
    RP_UNROLL
    for (int j_ = 0; j_ < M_; ++j_) {
        T[j_] = _mm512_add_pd(x0, _mm512_add_pd(P0[j_], P1[j_]));
        T[M_ + j_] = _mm512_add_pd(x0, _mm512_sub_pd(P0[j_], P1[j_]));
    }
    RP_UNROLL
    for (int j_ = 0; j_ < M_; ++j_) {
        Q0[j_] = VF[jv[2 * j_]];
        Q1[j_] = VF[jv[2 * j_ + 1]];
    }
    RP3_CONV(P0, Q0, kb0, M_);
    RP3_CONV(P1, Q1, kb1, M_);
    RP_UNROLL
    for (int j_ = 0; j_ < M_; ++j_)
        Q0[j_] = _mm512_add_pd(Q0[j_], Q1[j_]);
    RP3_CONV(P2, Q0, kb2, M_);
    O_[0] = _mm512_sub_pd(P0[0], P1[M_ - 1]);
    RP_UNROLL
    for (int j_ = 1; j_ < M_; ++j_)
        O_[2 * j_] = _mm512_add_pd(P0[j_], P1[j_ - 1]);
    RP_UNROLL
    for (int j_ = 0; j_ < M_; ++j_)
        O_[2 * j_ + 1] = _mm512_sub_pd(P2[j_], _mm512_add_pd(P0[j_], P1[j_]));
    RP_UNROLL
    for (int n = 0; n < H_; ++n) {
        __m512d o = _mm512_permute_pd(O_[n], 0x55);
        __m512d xp = _mm512_fmadd_pd(o, SG, T[n]);
        __m512d xm = _mm512_fnmadd_pd(o, SG, T[n]);
        RP3_ST((ptrdiff_t)kp[n] * rs, xp);
        RP3_ST((ptrdiff_t)km[n] * rs, xm);
    }
}

/* z quad at p = 13: rp_zquad with compile-time tiling (nt = 4, tail = 1) and
 * the inlined chunk at compile-time rs = 8.  nrows is compile-time at both
 * call sites.  Rows 13..15 of yt are never written by the chunk; the back
 * transpose moves their (uninitialized) lanes only through shuffles into
 * masked-out store columns -- same contract as rp_zquad. */
static inline __attribute__((always_inline))
void rp13_zquad(const cplx *src, cplx *dst, const int nrows, const double *k2)
{
    __attribute__((aligned(64))) double xt[16 * 8], yt[16 * 8];
    const double *x = (const double *)src;
    double *y = (double *)dst;
    RP_UNROLL
    for (int t = 0; t < 4; ++t) {
        const __mmask8 mk = t < 3 ? (__mmask8)0xFF : (__mmask8)0x03;
        __m512d r0 = _mm512_maskz_loadu_pd(mk, x + 8 * t);
        __m512d r1 = _mm512_setzero_pd(), r2 = r1, r3 = r1;
        if (nrows > 1) r1 = _mm512_maskz_loadu_pd(mk, x + 26 + 8 * t);
        if (nrows > 2) r2 = _mm512_maskz_loadu_pd(mk, x + 52 + 8 * t);
        if (nrows > 3) r3 = _mm512_maskz_loadu_pd(mk, x + 78 + 8 * t);
        __m512d o0, o1, o2, o3;
        r31_tp4(r0, r1, r2, r3, &o0, &o1, &o2, &o3);
        _mm512_store_pd(xt + (4 * t) * 8, o0);
        _mm512_store_pd(xt + (4 * t + 1) * 8, o1);
        _mm512_store_pd(xt + (4 * t + 2) * 8, o2);
        _mm512_store_pd(xt + (4 * t + 3) * 8, o3);
    }
    rp13_chunk(xt, yt, 8, k2, 1, (__mmask8)0xFF);
    RP_UNROLL
    for (int t = 0; t < 4; ++t) {
        const __mmask8 mk = t < 3 ? (__mmask8)0xFF : (__mmask8)0x03;
        __m512d o0, o1, o2, o3;
        r31_tp4(_mm512_load_pd(yt + (4 * t) * 8),
                _mm512_load_pd(yt + (4 * t + 1) * 8),
                _mm512_load_pd(yt + (4 * t + 2) * 8),
                _mm512_load_pd(yt + (4 * t + 3) * 8),
                &o0, &o1, &o2, &o3);
        _mm512_mask_storeu_pd(y + 8 * t, mk, o0);
        if (nrows > 1) _mm512_mask_storeu_pd(y + 26 + 8 * t, mk, o1);
        if (nrows > 2) _mm512_mask_storeu_pd(y + 52 + 8 * t, mk, o2);
        if (nrows > 3) _mm512_mask_storeu_pd(y + 78 + 8 * t, mk, o3);
    }
}

/* x pass in place, inner = 169 (rs = 338 compile-time): 42 full chunks + a
 * 1-column tail.  In-place safe: the chunk loads all 13 rows before storing. */
static void rp13_pass_x(cplx *v, const double *k2)
{
    double *dx = (double *)v;
    for (ptrdiff_t d = 0; d + 8 <= 338; d += 8)
        rp13_chunk(dx + d, dx + d, 338, k2, 1, (__mmask8)0xFF);
    rp13_chunk(dx + 336, dx + 336, 338, k2, 0, (__mmask8)0x03);
}

/* y pass in place per plane, inner = 13 (rs = 26): 3 full + 1-column tail */
static void rp13_pass_y(cplx *plane, const double *k2)
{
    double *dx = (double *)plane;
    rp13_chunk(dx,      dx,      26, k2, 1, (__mmask8)0xFF);
    rp13_chunk(dx + 8,  dx + 8,  26, k2, 1, (__mmask8)0xFF);
    rp13_chunk(dx + 16, dx + 16, 26, k2, 1, (__mmask8)0xFF);
    rp13_chunk(dx + 24, dx + 24, 26, k2, 0, (__mmask8)0x03);
}

static void rp13_volume(const fft3d_plan *pl, const cplx *src, cplx *dst)
{
    const double *k2 = pl->k2;
    for (size_t r = 0; r + 4 <= 169; r += 4)          /* 42 quads + 1 row */
        rp13_zquad(src + r * 13, dst + r * 13, 4, k2);
    rp13_zquad(src + (size_t)168 * 13, dst + (size_t)168 * 13, 1, k2);
    rp13_pass_x(dst, k2);
    for (int xpl = 0; xpl < 13; ++xpl)
        rp13_pass_y(dst + (size_t)xpl * 169, k2);
}

/* gen_r13, part 2: the same compile-time treatment for the DENSE tiny primes
 * (3/5/7/11 -- none has an rp2/rp3 mode, so rp_chunk is the kernel and it is
 * ALREADY always_inline with p/h as parameters: instantiating it with literal
 * constants makes every loop bound, row offset and mask compile-time for
 * free).  Trig stays in gct/gst (runtime base, broadcast cost unchanged).
 * LL = p^2 is 1 mod 4 for every odd p, so the z tail is always exactly one
 * row.  Execute path only, same as rp13. */
/* timing-only dev switches (compile-time constants; WRONG OUTPUT when 0) */
#ifdef RP_SKIPZ
#define RPD_DOZ 0
#else
#define RPD_DOZ 1
#endif
#ifdef RP_SKIPX
#define RPD_DOX 0
#else
#define RPD_DOX 1
#endif
#ifdef RP_SKIPY
#define RPD_DOY 0
#else
#define RPD_DOY 1
#endif

#define RPD_DEFINE(P, H)                                                       \
static inline __attribute__((always_inline))                                   \
void rpd_zquad_##P(const cplx *src, cplx *dst, const int nrows,                \
                   const double *ct, const double *st)                         \
{                                                                              \
    enum { NT_ = ((P) + 3) / 4, TL_ = (P) & 3 };                               \
    __attribute__((aligned(64))) double xt[4 * NT_ * 8], yt[4 * NT_ * 8];      \
    const double *x = (const double *)src;                                     \
    double *y = (double *)dst;                                                 \
    RP_UNROLL                                                                  \
    for (int t = 0; t < NT_; ++t) {                                            \
        const __mmask8 mk = (t < NT_ - 1 || !TL_)                              \
            ? (__mmask8)0xFF : (__mmask8)((1u << (2 * TL_)) - 1);              \
        __m512d r0 = _mm512_maskz_loadu_pd(mk, x + 8 * t);                     \
        __m512d r1 = _mm512_setzero_pd(), r2 = r1, r3 = r1;                    \
        if (nrows > 1) r1 = _mm512_maskz_loadu_pd(mk, x + 2 * (P) + 8 * t);    \
        if (nrows > 2) r2 = _mm512_maskz_loadu_pd(mk, x + 4 * (P) + 8 * t);    \
        if (nrows > 3) r3 = _mm512_maskz_loadu_pd(mk, x + 6 * (P) + 8 * t);    \
        __m512d o0, o1, o2, o3;                                                \
        r31_tp4(r0, r1, r2, r3, &o0, &o1, &o2, &o3);                           \
        _mm512_store_pd(xt + (4 * t) * 8, o0);                                 \
        _mm512_store_pd(xt + (4 * t + 1) * 8, o1);                             \
        _mm512_store_pd(xt + (4 * t + 2) * 8, o2);                             \
        _mm512_store_pd(xt + (4 * t + 3) * 8, o3);                             \
    }                                                                          \
    rp_chunk(xt, yt, NULL, 8, (P), (H), ct, st, 1, (__mmask8)0xFF);            \
    RP_UNROLL                                                                  \
    for (int t = 0; t < NT_; ++t) {                                            \
        const __mmask8 mk = (t < NT_ - 1 || !TL_)                              \
            ? (__mmask8)0xFF : (__mmask8)((1u << (2 * TL_)) - 1);              \
        __m512d o0, o1, o2, o3;                                                \
        r31_tp4(_mm512_load_pd(yt + (4 * t) * 8),                              \
                _mm512_load_pd(yt + (4 * t + 1) * 8),                          \
                _mm512_load_pd(yt + (4 * t + 2) * 8),                          \
                _mm512_load_pd(yt + (4 * t + 3) * 8),                          \
                &o0, &o1, &o2, &o3);                                           \
        _mm512_mask_storeu_pd(y + 8 * t, mk, o0);                              \
        if (nrows > 1) _mm512_mask_storeu_pd(y + 2 * (P) + 8 * t, mk, o1);     \
        if (nrows > 2) _mm512_mask_storeu_pd(y + 4 * (P) + 8 * t, mk, o2);     \
        if (nrows > 3) _mm512_mask_storeu_pd(y + 6 * (P) + 8 * t, mk, o3);     \
    }                                                                          \
}                                                                              \
static void rpd_volume_##P(const fft3d_plan *pl, const cplx *src, cplx *dst)   \
{                                                                              \
    const double *ct = pl->gct, *st = pl->gst;                                 \
    enum { LL_ = (P) * (P), ND_ = 2 * LL_, NP_ = 2 * (P) };                    \
    size_t r = 0;                                                              \
    if (RPD_DOZ) {                                                             \
        for (; r + 4 <= LL_; r += 4)                                           \
            rpd_zquad_##P(src + r * (P), dst + r * (P), 4, ct, st);            \
        rpd_zquad_##P(src + r * (P), dst + r * (P), 1, ct, st);                \
    } else {                                                                   \
        memcpy(dst, src, sizeof(cplx) * LL_ * (size_t)(P));                    \
    }                                                                          \
    double *dx = (double *)dst;                                                \
    ptrdiff_t d = 0;                                                           \
    if (RPD_DOX) {                                                             \
        for (; d + 8 <= ND_; d += 8)                                           \
            rp_chunk(dx + d, dx + d, NULL, ND_, (P), (H), ct, st,              \
                     1, (__mmask8)0xFF);                                       \
        if (d < ND_)                                                           \
            rp_chunk(dx + d, dx + d, NULL, ND_, (P), (H), ct, st,              \
                     0, (__mmask8)((1u << (ND_ - d)) - 1));                    \
    }                                                                          \
    if (RPD_DOY) for (int xpl = 0; xpl < (P); ++xpl) {                         \
        double *px = dx + (size_t)xpl * ND_;                                   \
        ptrdiff_t e = 0;                                                       \
        for (; e + 8 <= NP_; e += 8)                                           \
            rp_chunk(px + e, px + e, NULL, NP_, (P), (H), ct, st,              \
                     1, (__mmask8)0xFF);                                       \
        if (e < NP_)                                                           \
            rp_chunk(px + e, px + e, NULL, NP_, (P), (H), ct, st,              \
                     0, (__mmask8)((1u << (NP_ - e)) - 1));                    \
    }                                                                          \
}

#ifndef RP_NOD13
RPD_DEFINE(3, 1)
RPD_DEFINE(5, 2)
RPD_DEFINE(7, 3)
RPD_DEFINE(11, 5)

/* ---------------- gen_r14: DENSE Z-ROWS at p = 11 (the r13 next-step-1 item) ----
 * The transpose-quad z was the tiny-p residue r13 could not attribute: per 4
 * pencils it pays 12 tp4 shuffles (48 port-5 ops) + a staged stack round trip
 * before rp_chunk even starts.  This is the r31_zrow_pair dense form at
 * compile-time p=11: each contiguous 11-complex row folds ONCE in 512-bit
 * (u_{1..4}/v_{1..4} are one zmm add/sub against a lane-reversed mirror load,
 * u_5/v_5 one xmm pair), the half-spectrum accumulates in one zmm (k=0..3
 * duplicated pairs) + one ymm (k=4..5) per C/S system -- the ymm half
 * dispatches on port 1, which idles in every kernel on this panel (PMU audit
 * avenue 4) -- and the combine stores X_0..X_5 forward and X_6..X_10 as the
 * lane-reversed conjugate side (one shuffle each), exactly the R31_ZSTORE
 * pattern.  4 rows per call share every table load.  No transposes, no
 * staging: ~1 shuffle/row on port 5 vs the quad form's ~12. */
static inline __attribute__((always_inline))
void rpd11_zrow1(const double *x, double *y, const double *ctd, const double *std)
{
    __attribute__((aligned(64))) double uv[28];   /* u pairs at 0..9, v at 16..25 */
    __m512d A  = _mm512_loadu_pd(x + 2);                      /* x1 x2 x3 x4 */
    __m512d Br = _mm512_loadu_pd(x + 14);                     /* x7 x8 x9 x10 */
    __m512d B  = _mm512_shuffle_f64x2(Br, Br, 0x1B);          /* x10 x9 x8 x7 */
    _mm512_store_pd(uv,      _mm512_add_pd(A, B));            /* u1..u4 */
    _mm512_store_pd(uv + 16, _mm512_sub_pd(A, B));            /* v1..v4 */
    __m128d a5 = _mm_loadu_pd(x + 10), b5 = _mm_loadu_pd(x + 12);
    _mm_store_pd(uv + 8,  _mm_add_pd(a5, b5));                /* u5 */
    _mm_store_pd(uv + 24, _mm_sub_pd(a5, b5));                /* v5 */
    const __m512d x0 = _mm512_broadcast_f64x2(_mm_loadu_pd(x));
    __m512d CZ = x0, SZ = _mm512_setzero_pd();
    __m256d CY = _mm512_castpd512_pd256(x0), SY = _mm256_setzero_pd();
    RP_UNROLL
    for (int j = 0; j < 5; ++j) {
        const __m512d u = _mm512_broadcast_f64x2(_mm_load_pd(uv + 2 * j));
        const __m512d v = _mm512_broadcast_f64x2(_mm_load_pd(uv + 16 + 2 * j));
        CZ = _mm512_fmadd_pd(_mm512_load_pd(ctd + 16 * j), u, CZ);
        SZ = _mm512_fmadd_pd(_mm512_load_pd(std + 16 * j), v, SZ);
        CY = _mm256_fmadd_pd(_mm256_load_pd(ctd + 16 * j + 8),
                             _mm512_castpd512_pd256(u), CY);
        SY = _mm256_fmadd_pd(_mm256_load_pd(std + 16 * j + 8),
                             _mm512_castpd512_pd256(v), SY);
    }
    const __m512d SG  = _mm512_setr_pd(1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0);
    const __m256d SGY = _mm256_setr_pd(1.0, -1.0, 1.0, -1.0);
    __m512d TZ = _mm512_permute_pd(SZ, 0x55);
    __m256d TY = _mm256_permute_pd(SY, 0x5);
    _mm512_storeu_pd(y,     _mm512_fmadd_pd(TZ, SG, CZ));     /* X0..X3 */
    _mm256_storeu_pd(y + 8, _mm256_fmadd_pd(TY, SGY, CY));    /* X4 X5 */
    __m512d HZ = _mm512_fnmadd_pd(TZ, SG, CZ);
    __m256d HY = _mm256_fnmadd_pd(TY, SGY, CY);
    _mm256_storeu_pd(y + 12, _mm256_permute2f128_pd(HY, HY, 0x01)); /* X6 X7 */
    _mm512_mask_storeu_pd(y + 16, (__mmask8)0x3F,
                          _mm512_shuffle_f64x2(HZ, HZ, 0x1B));      /* X8..X10 */
}

/* 4 rows sharing every table load; rows are contiguous (22 doubles apart) */
static inline __attribute__((always_inline))
void rpd11_zrow4(const double *x, double *y, const double *ctd, const double *std)
{
    __attribute__((aligned(64))) double uv[4][28];
    __m512d CZ[4], SZ[4];
    __m256d CY[4], SY[4];
    RP_UNROLL
    for (int r = 0; r < 4; ++r) {
        const double *xr = x + 22 * r;
        __m512d A  = _mm512_loadu_pd(xr + 2);
        __m512d Br = _mm512_loadu_pd(xr + 14);
        __m512d B  = _mm512_shuffle_f64x2(Br, Br, 0x1B);
        _mm512_store_pd(uv[r],      _mm512_add_pd(A, B));
        _mm512_store_pd(uv[r] + 16, _mm512_sub_pd(A, B));
        __m128d a5 = _mm_loadu_pd(xr + 10), b5 = _mm_loadu_pd(xr + 12);
        _mm_store_pd(uv[r] + 8,  _mm_add_pd(a5, b5));
        _mm_store_pd(uv[r] + 24, _mm_sub_pd(a5, b5));
        const __m512d x0 = _mm512_broadcast_f64x2(_mm_loadu_pd(xr));
        CZ[r] = x0; SZ[r] = _mm512_setzero_pd();
        CY[r] = _mm512_castpd512_pd256(x0); SY[r] = _mm256_setzero_pd();
    }
    RP_UNROLL
    for (int j = 0; j < 5; ++j) {
        const __m512d cz = _mm512_load_pd(ctd + 16 * j);
        const __m512d sz = _mm512_load_pd(std + 16 * j);
        const __m256d cy = _mm256_load_pd(ctd + 16 * j + 8);
        const __m256d sy = _mm256_load_pd(std + 16 * j + 8);
        RP_UNROLL
        for (int r = 0; r < 4; ++r) {
            const __m512d u = _mm512_broadcast_f64x2(_mm_load_pd(uv[r] + 2 * j));
            const __m512d v = _mm512_broadcast_f64x2(_mm_load_pd(uv[r] + 16 + 2 * j));
            CZ[r] = _mm512_fmadd_pd(cz, u, CZ[r]);
            SZ[r] = _mm512_fmadd_pd(sz, v, SZ[r]);
            CY[r] = _mm256_fmadd_pd(cy, _mm512_castpd512_pd256(u), CY[r]);
            SY[r] = _mm256_fmadd_pd(sy, _mm512_castpd512_pd256(v), SY[r]);
        }
    }
    const __m512d SG  = _mm512_setr_pd(1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0);
    const __m256d SGY = _mm256_setr_pd(1.0, -1.0, 1.0, -1.0);
    RP_UNROLL
    for (int r = 0; r < 4; ++r) {
        double *yr = y + 22 * r;
        __m512d TZ = _mm512_permute_pd(SZ[r], 0x55);
        __m256d TY = _mm256_permute_pd(SY[r], 0x5);
        _mm512_storeu_pd(yr,     _mm512_fmadd_pd(TZ, SG, CZ[r]));
        _mm256_storeu_pd(yr + 8, _mm256_fmadd_pd(TY, SGY, CY[r]));
        __m512d HZ = _mm512_fnmadd_pd(TZ, SG, CZ[r]);
        __m256d HY = _mm256_fnmadd_pd(TY, SGY, CY[r]);
        _mm256_storeu_pd(yr + 12, _mm256_permute2f128_pd(HY, HY, 0x01));
        _mm512_mask_storeu_pd(yr + 16, (__mmask8)0x3F,
                              _mm512_shuffle_f64x2(HZ, HZ, 0x1B));
    }
}

/* rpd_volume_11 with the dense z (x and y passes identical to the macro's).
 * -DRPD11_QZ restores the transpose-quad z (control arm). */
static void rpd11_volume(const fft3d_plan *pl, const cplx *src, cplx *dst)
{
    const double *ct = pl->gct, *st = pl->gst;
    enum { P = 11, H = 5, LL_ = 121, ND_ = 242, NP_ = 22 };
    const double *sx = (const double *)src;
    double *dx = (double *)dst;
    if (RPD_DOZ) {
        size_t r = 0;
        for (; r + 4 <= LL_; r += 4)
            rpd11_zrow4(sx + r * 22, dx + r * 22, pl->ctd, pl->std_);
        rpd11_zrow1(sx + r * 22, dx + r * 22, pl->ctd, pl->std_);
    } else {
        memcpy(dst, src, sizeof(cplx) * LL_ * (size_t)P);
    }
    if (RPD_DOX) {
        ptrdiff_t d = 0;
        for (; d + 8 <= ND_; d += 8)
            rp_chunk(dx + d, dx + d, NULL, ND_, P, H, ct, st, 1, (__mmask8)0xFF);
        rp_chunk(dx + d, dx + d, NULL, ND_, P, H, ct, st,
                 0, (__mmask8)((1u << (ND_ - d)) - 1));
    }
    if (RPD_DOY) for (int xpl = 0; xpl < P; ++xpl) {
        double *px = dx + (size_t)xpl * ND_;
        ptrdiff_t e = 0;
        for (; e + 8 <= NP_; e += 8)
            rp_chunk(px + e, px + e, NULL, NP_, P, H, ct, st, 1, (__mmask8)0xFF);
        rp_chunk(px + e, px + e, NULL, NP_, P, H, ct, st,
                 0, (__mmask8)((1u << (NP_ - e)) - 1));
    }
}
#endif

/* forward 3D volume, generic prime: z rows (quads + tail), x in place
 * (inner = p^2), y in place per plane (inner = p) */
static void rp_volume(const fft3d_plan *pl, const cplx *src, cplx *dst)
{
    const int p = pl->L;
    const size_t LL = (size_t)p * p;
    size_t r = 0;
#ifdef RP_SKIPZ                 /* timing-only dev knob: WRONG OUTPUT */
    memcpy(dst, src, sizeof(cplx) * LL * (size_t)p);
    (void)r;
#else
    if (rp_w2z_on(pl))
        for (; r + 8 <= LL; r += 8)
            rp_zquad2(pl, src + r * p, dst + r * p, 2 * p, 2 * p);
    for (; r + 4 <= LL; r += 4)
        rp_zquad(pl, src + r * p, dst + r * p, 4, 2 * p, 2 * p);
    if (r < LL)
        rp_zquad(pl, src + r * p, dst + r * p, (int)(LL - r), 2 * p, 2 * p);
#endif
#ifndef RP_SKIPX                /* timing-only dev knob: WRONG OUTPUT */
    rp_pass(pl, dst, dst, NULL, (ptrdiff_t)LL, (ptrdiff_t)LL, pl->xpf);
#endif
#ifndef RP_SKIPY                /* timing-only dev knob: WRONG OUTPUT */
    for (int xpl = 0; xpl < p; ++xpl)
        rp_pass(pl, dst + (size_t)xpl * LL, dst + (size_t)xpl * LL, NULL, p, p, 0);
#endif
}

#endif /* __AVX512F__ */

/* ---------------- dense fallback / reference (the round-0 stub engine) ---------------- */

static void ref_contract(const cplx *w, int L, const cplx *in, cplx *out, int inner)
{
    for (int k = 0; k < L; ++k)
        for (int c = 0; c < inner; ++c) {
            cplx acc = 0.0;
            for (int j = 0; j < L; ++j)
                acc += w[(size_t)k * L + j] * in[(size_t)j * inner + c];
            out[(size_t)k * inner + c] = acc;
        }
}

/* full reference volume: src -> dst (src untouched; uses p->tmp) */
static void ref_volume(fft3d_plan *p, const cplx *src, cplx *dst)
{
    const int L = p->L;
    const size_t LL = (size_t)L * L;
    ref_contract(p->w, L, src, dst, (int)LL);
    for (int x = 0; x < L; ++x)
        ref_contract(p->w, L, dst + (size_t)x * LL, p->tmp + (size_t)x * LL, L);
    for (size_t row = 0; row < LL; ++row)
        ref_contract(p->w, L, p->tmp + row * L, dst + row * L, 1);
}

/* ---------------- one volume, forward 3D (fast path) ----------------
 * z rows: src -> dst (row-local, out-of-place or in-place safe); x and y IN
 * PLACE on dst (the chunk kernel loads every row before storing any). */

#ifdef __AVX512F__
/* gen_r14: execute()-onto-the-padded-arena, BUILT AND REJECTED on the node
 * (kept opt-in behind -DR31_PADEXEC for the xarch race).  The round's premise
 * was that execute() at 31 pays the r1 flat layout's 4K aliases the r2 arena
 * killed for the chain -- but the node race read B=1 90.8-104 vs 76.3-80 flat
 * (+15-20%, 4/4) and B=16 106 vs 92.8 (+14%).  Mechanism: execute's flat z is
 * already OUT-OF-PLACE (src -> dst, no aliasing) and its y runs at small
 * in-plane strides, so only the in-place x-pass pays the alias tax (~few %);
 * the arena route adds a whole third buffer (st, 574 KB), pushing the
 * per-call working set src+st+dst = 1.5 MB past the 1.25 MB L2.  Execute IS
 * m=1: the r5/r11 "per-volume fixed costs do not amortize at small m" law,
 * met again from a third direction.  Flat execute measures ~76 us vs a ~73 us
 * three-pass model -- already at model. */
static __attribute__((unused))
void r31_exec_pad(fft3d_plan *p, const cplx *src, cplx *dst)
{
    const size_t LL = 31 * 31;
    cplx *st = p->st;
    for (int x = 0; x < 31; ++x) {
        const cplx *sp = src + (size_t)x * LL;
        cplx *pl = st + (size_t)x * R31_PP;
        for (int q = 0; q < 7; ++q)
            r31_zquad_fp(sp + (size_t)q * 4 * 31, pl + (size_t)q * 4 * R31_ZP,
                         p->ke, p->ko);
        r31_zpass(sp + 28 * 31, pl + 28 * R31_ZP, 3, 31, R31_ZP,
                  p->ctd, p->std_);
    }
    r31_pass_xp(st, p->ke, p->ko);
    for (int x = 0; x < 31; ++x)
        r31_pass_yf(st + (size_t)x * R31_PP, dst + (size_t)x * LL,
                    p->ke, p->ko);
}
#endif

static void fast_volume(fft3d_plan *p, const cplx *src, cplx *dst)
{
#ifdef __AVX512F__
#ifndef RP_NO13
    if (p->use13) { rp13_volume(p, src, dst); return; }
#endif
#ifndef RP_NOD13
    if (!p->m2 && !p->m3) switch (p->L) {   /* compile-time tiny dense primes */
    case 3:  rpd_volume_3(p, src, dst);  return;
    case 5:  rpd_volume_5(p, src, dst);  return;
    case 7:  rpd_volume_7(p, src, dst);  return;
    case 11:
#ifndef RPD11_QZ
        if (p->ctd && p->std_) { rpd11_volume(p, src, dst); return; }
#endif
        rpd_volume_11(p, src, dst); return;
    default: break;
    }
#endif
    if (p->L != 31) { rp_volume(p, src, dst); return; }
#ifdef R31_PADEXEC
    if (p->st) { r31_exec_pad(p, src, dst); return; }
#endif
    const size_t LL = 31 * 31;
    r31_zpass_main(p, src, dst, LL);
    r31_pass_x(dst, dst, p->ke, p->ko);
    for (int x = 0; x < 31; ++x)
        r31_pass_y(dst + (size_t)x * LL, dst + (size_t)x * LL, p->ke, p->ko);
#else
    (void)p; (void)src; (void)dst;
#endif
}

void fft3d_execute(fft3d_plan *p, const cplx *in, cplx *out)
{
    const size_t vol = (size_t)p->L * p->L * p->L;
    for (int b = 0; b < p->batch; ++b) {
        if (p->fast)
            fast_volume(p, in + (size_t)b * vol, out + (size_t)b * vol);
        else
            ref_volume(p, in + (size_t)b * vol, out + (size_t)b * vol);
    }
}

/* ---------------- fused map chain (shape from gen_dense_prime / ice s6) ---------------- */

/* z and o may alias (in-place map): elementwise, loads precede the store per
 * point -- deliberately NOT restrict-qualified */
static void map_volume(const cplx *z, const cplx *restrict c,
                       cplx *o, size_t npts)
{
    const double *zp = (const double *)z;
    const double *cp = (const double *)c;
    double *op = (double *)o;
    size_t i = 0;
#ifdef __AVX512F__
    const __m512d ONE  = _mm512_set1_pd(1.0);
    const __m512d TH   = _mm512_set1_pd(1.5);
    const __m512d HALF = _mm512_set1_pd(0.5);
    const __m512d TINY = _mm512_set1_pd(1e-300);
    for (; i + 8 <= npts; i += 8) {
        __m512d w0 = _mm512_add_pd(_mm512_loadu_pd(zp + 2 * i),
                                   _mm512_loadu_pd(cp + 2 * i));
        __m512d w1 = _mm512_add_pd(_mm512_loadu_pd(zp + 2 * i + 8),
                                   _mm512_loadu_pd(cp + 2 * i + 8));
        __m512d p0 = _mm512_mul_pd(w0, w0), p1 = _mm512_mul_pd(w1, w1);
        __m512d m2 = _mm512_add_pd(_mm512_unpacklo_pd(p0, p1),
                                   _mm512_unpackhi_pd(p0, p1));
        __m512d m2c = _mm512_max_pd(m2, TINY);
        __m512d r = _mm512_rsqrt14_pd(m2c);
        __m512d hm = _mm512_mul_pd(m2c, HALF);
        r = _mm512_mul_pd(r, _mm512_fnmadd_pd(_mm512_mul_pd(hm, r), r, TH));
        r = _mm512_mul_pd(r, _mm512_fnmadd_pd(_mm512_mul_pd(hm, r), r, TH));
        __m512d d = _mm512_fmadd_pd(m2c, r, ONE);           /* 1 + |w| */
        __m512d rec = _mm512_div_pd(ONE, d);                /* the one divide */
        _mm512_storeu_pd(op + 2 * i,     _mm512_mul_pd(w0, _mm512_unpacklo_pd(rec, rec)));
        _mm512_storeu_pd(op + 2 * i + 8, _mm512_mul_pd(w1, _mm512_unpackhi_pd(rec, rec)));
    }
#endif
    for (; i < npts; ++i) {
        double re = zp[2 * i] + cp[2 * i];
        double im = zp[2 * i + 1] + cp[2 * i + 1];
        double sc = 1.0 / (1.0 + sqrt(re * re + im * im));
        op[2 * i] = re * sc;
        op[2 * i + 1] = im * sc;
    }
}

#ifdef __AVX512F__
/* env-gated per-pass profile of the flat generic chain (RP_PROF=1): dev-only
 * diagnostics on the DRAM-resident primes; zero timing calls when unset. */
static double rp_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}
static int rp_prof_on(void)
{
    static int v = -1;
    if (v < 0) v = getenv("RP_PROF") != NULL;
    return v;
}

/* One volume's whole fused chain on the padded private state (raw engine; the
 * caller gates on p->fast).  Step 0's z pass reads flat x0 straight into the
 * arena; steps run z (8 uniform quads/plane -- the 32nd "row" is the zeroed
 * pad row, DFT(0)=0), x across planes at the anti-alias pitch, then per plane
 * y + map (c from the padded mirror, filled once per volume).  The last step's
 * map stays in the arena and the rows are copied out flat. */
static void r31_chain_volume(fft3d_plan *p, const cplx *x0v, const cplx *cv,
                             cplx *outv, int m)
{
    const size_t LL = 31 * 31;
    cplx *st = p->st, *cp = p->cpad;
    for (int x = 0; x < 31; ++x)
        for (int y = 0; y < 31; ++y)
            memcpy(cp + (size_t)x * R31_PP + (size_t)y * R31_ZP,
                   cv + (size_t)x * LL + (size_t)y * 31, 31 * sizeof(cplx));
#ifdef R31_R3CHAIN
    /* gen_r3 pass order (control arm): z sweep / x sweep / y+map sweep --
     * step s+1's z re-reads the whole state from L2 after the map sweep. */
    for (int s = 0; s < m; ++s) {
        for (int x = 0; x < 31; ++x) {
            cplx *pl = st + (size_t)x * R31_PP;
            if (s == 0) {
                const cplx *sp = x0v + (size_t)x * LL;
                for (int q = 0; q < 7; ++q)
                    r31_zquad_fp(sp + (size_t)q * 4 * 31,
                                 pl + (size_t)q * 4 * R31_ZP, p->ke, p->ko);
                r31_zpass(sp + 28 * 31, pl + 28 * R31_ZP, 3, 31, R31_ZP,
                          p->ctd, p->std_);
            } else {
                for (int q = 0; q < 8; ++q)
                    r31_zquad_pp(pl + (size_t)q * 4 * R31_ZP,
                                 pl + (size_t)q * 4 * R31_ZP, p->ke, p->ko);
            }
        }
        r31_pass_xp(st, p->ke, p->ko);
        for (int x = 0; x < 31; ++x) {
            cplx *pl = st + (size_t)x * R31_PP;
            r31_pass_yp(pl, p->ke, p->ko);
            map_volume(pl, cp + (size_t)x * R31_PP, pl, 31 * R31_ZP);
            if (s == m - 1) {
                cplx *op = outv + (size_t)x * LL;
                for (int y = 0; y < 31; ++y)
                    memcpy(op + (size_t)y * 31, pl + (size_t)y * R31_ZP,
                           31 * sizeof(cplx));
            }
        }
    }
#else
    /* gen_r4 PLANE CUSTODY (gen_layout r3's window idea, gen_bluestein r4's
     * confirmation): z contracts within a plane, so step s+1's z-pass runs on
     * each plane RIGHT AFTER that plane's map, while it is still L1/L2-hot --
     * one full-state L2 read per step deleted vs the r3 order.  Identical
     * arithmetic, identical per-pass order within each plane => bit-identical.
     * -DR31_ZMAPF additionally fuses the map into the z-quads' transpose-in
     * loads (deletes the separate map sweep; vdivpd form => still
     * bit-identical). */
    for (int x = 0; x < 31; ++x) {          /* prologue: z_0 from flat x0 */
        cplx *pl = st + (size_t)x * R31_PP;
        const cplx *sp = x0v + (size_t)x * LL;
        for (int q = 0; q < 7; ++q)
            r31_zquad_fp(sp + (size_t)q * 4 * 31,
                         pl + (size_t)q * 4 * R31_ZP, p->ke, p->ko);
        r31_zpass(sp + 28 * 31, pl + 28 * R31_ZP, 3, 31, R31_ZP,
                  p->ctd, p->std_);
    }
    for (int s = 0; s < m; ++s) {
        r31_pass_xp(st, p->ke, p->ko);
        for (int x = 0; x < 31; ++x) {
            cplx *pl = st + (size_t)x * R31_PP;
            const cplx *cpl = cp + (size_t)x * R31_PP;
            r31_pass_yp(pl, p->ke, p->ko);
            if (s == m - 1) {
                map_volume(pl, cpl, pl, 31 * R31_ZP);
                cplx *op = outv + (size_t)x * LL;
                for (int y = 0; y < 31; ++y)
                    memcpy(op + (size_t)y * 31, pl + (size_t)y * R31_ZP,
                           31 * sizeof(cplx));
            } else {
#ifdef R31_ZMAPF
                for (int q = 0; q < 8; ++q)
                    r31_zquad_mp(pl + (size_t)q * 4 * R31_ZP,
                                 pl + (size_t)q * 4 * R31_ZP,
                                 cpl + (size_t)q * 4 * R31_ZP, p->ke, p->ko);
#else
                map_volume(pl, cpl, pl, 31 * R31_ZP);
                /* -DR31_ZMIX=q (< 8): rows 4q..30 through the dense zrow
                 * kernel instead of transpose quads -- moves z work from the
                 * quad's binding port 5 onto the load ports (r4 next-step 1).
                 * Default 8 = the all-quad r4 form, bit-identical. */
                for (int q = 0; q < R31_ZMIX; ++q)
                    r31_zquad_pp(pl + (size_t)q * 4 * R31_ZP,
                                 pl + (size_t)q * 4 * R31_ZP, p->ke, p->ko);
                if (R31_ZMIX < 8)
                    r31_zpass(pl + (size_t)R31_ZMIX * 4 * R31_ZP,
                              pl + (size_t)R31_ZMIX * 4 * R31_ZP,
                              (size_t)(31 - 4 * R31_ZMIX), R31_ZP, R31_ZP,
                              p->ctd, p->std_);
#endif
            }
        }
    }
#endif
}

/* generic-prime fused chain: fully in place on the out volume (the r1 form;
 * every rp pass and the map are in-place safe), map per plane right after its
 * y pass while the plane is cache-hot.  -DRP_YMAPFUSE instead fuses the map
 * into the y-pass stores (raceable; the r31 engine lost this one to register
 * pressure -- the generic kernel's combine is leaner, so it stays a knob). */
/* z-pass over one plane's p pencils, in place: quads + a per-plane tail;
 * rowp = row pitch in complex (p flat, zp padded) */
static void rp_zplane(const fft3d_plan *pl, cplx *plane, const ptrdiff_t rowp)
{
    const int p = pl->L;
    int r = 0;
    if (rp_w2z_on(pl))
        for (; r + 8 <= p; r += 8)
            rp_zquad2(pl, plane + (size_t)r * rowp, plane + (size_t)r * rowp,
                      2 * rowp, 2 * rowp);
    for (; r + 4 <= p; r += 4)
        rp_zquad(pl, plane + (size_t)r * rowp, plane + (size_t)r * rowp, 4,
                 2 * rowp, 2 * rowp);
    if (r < p)
        rp_zquad(pl, plane + (size_t)r * rowp, plane + (size_t)r * rowp, p - r,
                 2 * rowp, 2 * rowp);
}

static void rp_chain_volume(fft3d_plan *pl, const cplx *x0v, const cplx *cv,
                            cplx *stv, int m)
{
    const int p = pl->L;
    const size_t LL = (size_t)p * p, vol = LL * p;
    if (pl->gs) {
    /* gen_r5: PADDED ARENA custody chain (the r2 lesson at 31 paid at the
     * mid primes; create() only allocates the arena for p <= RP_PAD_MAX).
     * Rows p -> zp complex (64B-aligned, y tail-free), planes at the
     * alias-free pitch pp (see rp_pick_pp), state + c mirror in one
     * huge-page arena at a +2048B relative page phase.  Pad slots were
     * zeroed at create() and stay zero: no pass mixes columns, DFT(0)=0,
     * map(0+0)=0.  Pass order is the r4 custody order (prologue z, then per
     * step x, per plane y+map+z_next).  Per-point arithmetic is lanewise
     * everywhere, so outputs are BIT-IDENTICAL to the flat chain (cmp-
     * verified at 13/37/61). */
    const int zp = pl->zp;
    const size_t PP = (size_t)pl->pp;
    const ptrdiff_t live = (ptrdiff_t)p * zp;            /* mult of 4 complex */
    const size_t mpts = ((size_t)live + 7) & ~(size_t)7; /* map window, <= PP */
    cplx *st = pl->gs, *cp = pl->gc;
    for (int x = 0; x < p; ++x)                          /* c mirror, once per volume */
        for (int y = 0; y < p; ++y)
            memcpy(cp + (size_t)x * PP + (size_t)y * zp,
                   cv + (size_t)x * LL + (size_t)y * p, (size_t)p * sizeof(cplx));
    for (int x = 0; x < p; ++x) {                        /* prologue: z_0 flat -> padded */
        const cplx *sp = x0v + (size_t)x * LL;
        cplx *plane = st + (size_t)x * PP;
        int r = 0;
        if (rp_w2z_on(pl))
            for (; r + 8 <= p; r += 8)
                rp_zquad2(pl, sp + (size_t)r * p, plane + (size_t)r * zp,
                          2 * p, 2 * zp);
        for (; r + 4 <= p; r += 4)
            rp_zquad(pl, sp + (size_t)r * p, plane + (size_t)r * zp, 4,
                     2 * p, 2 * zp);
        if (r < p)
            rp_zquad(pl, sp + (size_t)r * p, plane + (size_t)r * zp, p - r,
                     2 * p, 2 * zp);
    }
    for (int s = 0; s < m; ++s) {
        rp_pass(pl, st, st, NULL, live, (ptrdiff_t)PP, pl->xpf);  /* x: planes at pitch PP */
        for (int xpl = 0; xpl < p; ++xpl) {
            cplx *plane = st + (size_t)xpl * PP;
            const cplx *cpl = cp + (size_t)xpl * PP;
#ifdef RP_YMAPFUSE
            rp_pass(pl, plane, plane, cpl, zp, zp, 0);
#else
            rp_pass(pl, plane, plane, NULL, zp, zp, 0);
            map_volume(plane, cpl, plane, mpts);
#endif
            if (s < m - 1) {
                rp_zplane(pl, plane, zp);
            } else {
                cplx *op = stv + (size_t)xpl * LL;
                for (int y = 0; y < p; ++y)
                    memcpy(op + (size_t)y * p, plane + (size_t)y * zp,
                           (size_t)p * sizeof(cplx));
            }
        }
    }
    return;
    }
    memcpy(stv, x0v, vol * sizeof(cplx));
#ifdef RP_R3CHAIN
    /* gen_r3 order (control arm): global z sweep per step */
    for (int s = 0; s < m; ++s) {
        size_t r = 0;
        for (; r + 4 <= LL; r += 4)
            rp_zquad(pl, stv + r * p, stv + r * p, 4, 2 * p, 2 * p);
        if (r < LL)
            rp_zquad(pl, stv + r * p, stv + r * p, (int)(LL - r), 2 * p, 2 * p);
        rp_pass(pl, stv, stv, NULL, (ptrdiff_t)LL, (ptrdiff_t)LL, pl->xpf);
        for (int xpl = 0; xpl < p; ++xpl) {
            cplx *plane = stv + (size_t)xpl * LL;
            const cplx *cpl = cv + (size_t)xpl * LL;
#ifdef RP_YMAPFUSE
            rp_pass(pl, plane, plane, cpl, p, p, 0);
#else
            rp_pass(pl, plane, plane, NULL, p, p, 0);
            map_volume(plane, cpl, plane, LL);
#endif
        }
    }
#else
    /* gen_r4 PLANE CUSTODY (same move as the r31 chain): step s+1's z runs
     * per plane right after that plane's map, while the plane is cache-hot.
     * At DRAM-resident sizes (p >= ~50, volume > L2) this deletes a whole
     * volume read per step.  Per-pencil arithmetic identical; only the quad
     * GROUPING changes at p % 4 != 0 (pencils are lane-independent =>
     * bit-identical outputs). */
    for (int xpl = 0; xpl < p; ++xpl)       /* prologue: z_0 per plane */
        rp_zplane(pl, stv + (size_t)xpl * LL, p);
    const int prof = rp_prof_on();
    double tx = 0, ty = 0, tmp_ = 0, tz = 0, t0 = 0, t1;
    for (int s = 0; s < m; ++s) {
        if (prof) t0 = rp_now();
        rp_pass(pl, stv, stv, NULL, (ptrdiff_t)LL, (ptrdiff_t)LL, pl->xpf);
        if (prof) { t1 = rp_now(); tx += t1 - t0; }
        for (int xpl = 0; xpl < p; ++xpl) {
            cplx *plane = stv + (size_t)xpl * LL;
            const cplx *cpl = cv + (size_t)xpl * LL;
            if (prof) t0 = rp_now();
#ifdef RP_YMAPFUSE
            rp_pass(pl, plane, plane, cpl, p, p, 0);
            if (prof) { t1 = rp_now(); ty += t1 - t0; t0 = t1; }
#else
            rp_pass(pl, plane, plane, NULL, p, p, 0);
            if (prof) { t1 = rp_now(); ty += t1 - t0; t0 = t1; }
            map_volume(plane, cpl, plane, LL);
            if (prof) { t1 = rp_now(); tmp_ += t1 - t0; t0 = t1; }
#endif
            if (s < m - 1)
                rp_zplane(pl, plane, p);
            if (prof) { t1 = rp_now(); tz += t1 - t0; }
        }
    }
    if (prof)
        fprintf(stderr, "[rp_prof p=%d m=%d] x %.3f ms/step  y %.3f  map %.3f  z %.3f\n",
                p, m, 1e3 * tx / m, 1e3 * ty / m, 1e3 * tmp_ / m, 1e3 * tz / m);
#endif
}
#endif /* __AVX512F__ */

void fft3d_chain(fft3d_plan *p, const cplx *x0, const cplx *c,
                 cplx *final_out, int m)
{
    const size_t LL = (size_t)p->L * p->L, vol = LL * p->L;
    for (int b = 0; b < p->batch; ++b) {
        cplx *stv = final_out + (size_t)b * vol;    /* state lives in the out volume */
        const cplx *cv = c + (size_t)b * vol;
        if (p->fast && m > 0 && p->L != 31) {
#ifdef __AVX512F__
            rp_chain_volume(p, x0 + (size_t)b * vol, cv, stv, m);
#endif
        } else if (p->fast && m > 0) {
#ifdef __AVX512F__
#if defined(R31_FLATCHAIN) || defined(R31_FUSEMAP)
            /* gen_r1 shipped form: all passes in place on the FLAT state in the
             * out volume (953 KB working set).  Kept for A/B (-DR31_FLATCHAIN);
             * the natural 961-complex plane pitch makes every x-pass access
             * line-split AND 4K-aliases each chunk's stores against the next
             * chunk's loads at row distance 4 (961*16*4 == 64 mod 4096). */
            memcpy(stv, x0 + (size_t)b * vol, vol * sizeof(cplx));
            for (int s = 0; s < m; ++s) {
                r31_zpass_main(p, stv, stv, LL);
                r31_pass_x(stv, stv, p->ke, p->ko);
#ifdef R31_FUSEMAP
                for (int x = 0; x < 31; ++x)
                    r31_pass_ym(stv + (size_t)x * LL, p->t1 + (size_t)x * LL,
                                cv + (size_t)x * LL, p->ke, p->ko);
                memcpy(stv, p->t1, vol * sizeof(cplx));
#else
                for (int x = 0; x < 31; ++x)
                    r31_pass_y(stv + (size_t)x * LL, stv + (size_t)x * LL,
                               p->ke, p->ko);
                map_volume(stv, cv, stv, vol);
#endif
            }
#else
            /* gen_r2 form: fully padded/aligned private state -- see
             * r31_chain_volume.  Working set st + cpad = 1.14 MB < L2. */
            r31_chain_volume(p, x0 + (size_t)b * vol, cv, stv, m);
#endif
#endif
        } else {
            memcpy(stv, x0 + (size_t)b * vol, vol * sizeof(cplx));
            for (int s = 0; s < m; ++s) {
                ref_volume(p, stv, p->t1);
                map_volume(p->t1, cv, stv, vol);
            }
        }
    }
}

/* ---------------- plan lifecycle ---------------- */

static void *xalloc(size_t bytes)
{
    return aligned_alloc(64, (bytes + 63) & ~(size_t)63);
}

/* deterministic pseudo-random volume; compare fast engine vs dense reference
 * for both execute AND one padded chain step (a pad/stride bug must fall back,
 * not ship).  Transcription bugs in the Rader tables would show at ~1e0; the
 * correct engines differ by rounding only (~1e-15). */
static int self_check(fft3d_plan *p)
{
#ifdef __AVX512F__
    const size_t vol = (size_t)p->L * p->L * p->L;
    cplx *a = xalloc(vol * sizeof(cplx));
    cplx *cc = xalloc(vol * sizeof(cplx));
    cplx *rf = xalloc(vol * sizeof(cplx));
    cplx *ff = xalloc(vol * sizeof(cplx));
    if (!a || !cc || !rf || !ff) { free(a); free(cc); free(rf); free(ff); return 0; }
    unsigned long long st = 0x9e3779b97f4a7c15ull;
    double *ad = (double *)a, *cd = (double *)cc;
    for (size_t i = 0; i < 2 * vol; ++i) {
        st ^= st << 13; st ^= st >> 7; st ^= st << 17;
        ad[i] = (double)(long long)(st % 2000001ull) / 1000000.0 - 1.0;
        st ^= st << 13; st ^= st >> 7; st ^= st << 17;
        cd[i] = (double)(long long)(st % 2000001ull) / 1000000.0 - 1.0;
    }
    ref_volume(p, a, rf);
    fast_volume(p, a, ff);
    long double num = 0, den = 0;
    for (size_t i = 0; i < vol; ++i) {
        cplx dd = ff[i] - rf[i];
        num += creal(dd) * creal(dd) + cimag(dd) * cimag(dd);
        den += creal(rf[i]) * creal(rf[i]) + cimag(rf[i]) * cimag(rf[i]);
    }
    int ok = den > 0 && sqrtl(num / den) < 1e-13L;
    if (ok) {
        for (size_t i = 0; i < vol; ++i) {      /* scalar-map the reference */
            double re = creal(rf[i]) + creal(cc[i]);
            double im = cimag(rf[i]) + cimag(cc[i]);
            double sc = 1.0 / (1.0 + sqrt(re * re + im * im));
            rf[i] = re * sc + I * im * sc;
        }
        int do_chain_leg = 1;
#if defined(R31_FLATCHAIN) || defined(R31_FUSEMAP)
        if (p->L == 31) do_chain_leg = 0;
#endif
        if (do_chain_leg) {
            if (p->L == 31) r31_chain_volume(p, a, cc, ff, 1);
            else            rp_chain_volume(p, a, cc, ff, 1);
            num = den = 0;
            for (size_t i = 0; i < vol; ++i) {
                cplx dd = ff[i] - rf[i];
                num += creal(dd) * creal(dd) + cimag(dd) * cimag(dd);
                den += creal(rf[i]) * creal(rf[i]) + cimag(rf[i]) * cimag(rf[i]);
            }
            ok = den > 0 && sqrtl(num / den) < 1e-13L;
        }
    }
    free(a); free(cc); free(rf); free(ff);
    return ok;
#else
    (void)p;
    return 0;
#endif
}

/* st + cpad in ONE 2MiB huge-page arena, c mirror at page phase +2048 B
 * (gen_layout gl_map_huge recipe; kills the map's c-load / y-store 4K alias
 * two same-phase aligned_allocs produce).  Heap fallback keeps r2 behavior. */
static void r31_arena_init(fft3d_plan *p)
{
    const size_t one = (size_t)31 * R31_PP * sizeof(cplx);
    const size_t stb = (one + 4095) & ~(size_t)4095;
    const size_t HP = (size_t)2 << 20;
    const size_t len = (stb + 2048 + one + HP - 1) & ~(HP - 1);
    void *raw = mmap(0, len + HP, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (raw != MAP_FAILED) {
        char *al = (char *)(((uintptr_t)raw + HP - 1) & ~(uintptr_t)(HP - 1));
        size_t head = (size_t)(al - (char *)raw);
        if (head) munmap(raw, head);
        size_t tl = (size_t)(((char *)raw + len + HP) - (al + len));
        if (tl) munmap(al + len, tl);
        madvise(al, len, MADV_HUGEPAGE);
        memset(al, 0, len);            /* prefault now: faults belong in create() */
        p->arena = al;
        p->alen = len;
        p->st = (cplx *)al;
        p->cpad = (cplx *)(al + stb + 2048);
        return;
    }
    p->st = xalloc(one);
    p->cpad = xalloc(one);
}

fft3d_plan *fft3d_create(int L, int batch)
{
    if (!fft3d_supports(L)) return NULL;
    fft3d_plan *p = calloc(1, sizeof *p);
    if (!p) return NULL;
    p->L = L;
    p->batch = batch;
    p->h = (L - 1) / 2;

    const size_t vol = (size_t)L * L * L;
    p->t1 = xalloc(vol * sizeof(cplx));
    p->w = xalloc((size_t)L * L * sizeof(cplx));
    p->tmp = xalloc(vol * sizeof(cplx));
    int okm = p->t1 && p->w && p->tmp;
    if (L == 31) {
        p->ke = xalloc(20 * sizeof(double));
        p->ko = xalloc(20 * sizeof(double));
        p->ctd = r31_trig_dup(0);
        p->std_ = r31_trig_dup(1);
        r31_arena_init(p);
        okm = okm && p->ke && p->ko && p->ctd && p->std_ && p->st && p->cpad;
        if (okm) {
            /* pad slots must be zero (not garbage/denormals) and then stay zero */
            memset(p->st, 0, (size_t)31 * R31_PP * sizeof(cplx));
            memset(p->cpad, 0, (size_t)31 * R31_PP * sizeof(cplx));
            r31_build_kernels(p->ke, p->ko);
        }
    } else {
        p->gct = rp_trig(L, p->h, 0);
        p->gst = rp_trig(L, p->h, 1);
        /* gen_r14: duplicated-pair trig for the p=11 dense z rows (ctd/std_
         * are 31-only otherwise; allocation failure just keeps the quad z) */
#if defined(__AVX512F__) && !defined(RP_NOD13) && !defined(RPD11_QZ)
        if (L == 11) {
            p->ctd = rpd11_trig_dup(0);
            p->std_ = rpd11_trig_dup(1);
        }
#endif
        /* outer-C3 Rader where the quotient group order is h = 3m, h odd,
         * gcd(3,m) = 1 (43/67/79/103): ~2.25x fewer conv FMA than the dense
         * half-system.  -DRP_NOC3 keeps the dense engine (control arm).
         * Correctness gated by the same create() self-check as everything. */
#ifndef RP_NOC3
        {
            const int hh = p->h, mm = hh / 3;
            if ((hh & 1) && hh % 3 == 0 && mm % 3 != 0 &&
                (mm == 7 || mm == 11 || mm == 13 || mm == 17) && rp3_build(p))
                p->m3 = mm;
        }
#endif
        /* even-h split Rader (gen_r6): 5m^2 vs dense 8m^2 conv FMA for every
         * p == 1 mod 4 in class (13..113).  -DRP_NO2 keeps the dense engine
         * (control arm).  Gated by the same create() self-check. */
#ifndef RP_NO2
        if (!p->m3) {
            const int hh = p->h, mm = hh / 2;
            if (!(hh & 1) &&
                (mm == 3 || mm == 4 || mm == 7 || mm == 9 || mm == 10 ||
                 mm == 13 || mm == 15 || mm == 18 || mm == 22 || mm == 24 ||
                 mm == 25 || mm == 27 || mm == 28) && rp2_build(p))
                p->m2 = mm;
        }
#endif
        /* gen_r13 compile-time 13 engine: only when rp2_build's tables match
         * the hardcoded ones (they are deterministic, but a mismatch must
         * select the generic path, not ship a wrong answer) */
#if defined(__AVX512F__) && !defined(RP_NO13)
        if (L == 13 && p->m2 == 3 && p->j2 &&
            !memcmp(p->j2, RP13_TAB, sizeof(RP13_TAB)))
            p->use13 = 1;
#endif
        /* padded chain arena: state + c mirror in one huge-page mapping
         * (gl_map_huge zeroes it -- the pad slots MUST start zero), c at a
         * +2048B relative page phase (the r3 anti-alias trick at 31).  Only
         * for p <= RP_PAD_MAX (see there); allocation failure falls back to
         * the flat chain (gs stays NULL), never fails create(). */
#ifndef RP_FLATCHAIN
        if (L <= RP_PAD_MAX) {
            p->zp = rp_zpc(L);
            p->pp = rp_pick_pp(L * p->zp);
            const size_t one = (size_t)L * p->pp * sizeof(cplx);
            const size_t stb = (one + 4095) & ~(size_t)4095;
            char *al = gl_map_huge(&p->gmap, stb + 2048 + one);
            if (al) {
                p->gs = (cplx *)al;
                p->gc = (cplx *)(al + stb + 2048);
            }
        }
#endif
        /* x-pass software prefetch: only where the chain state is not
         * cache-resident (the p+1 row streams then miss past the L2
         * streamer's tracking).  state + c = 2 * 16 L^3 bytes vs the 24 MB
         * LLC; RP_PFMIN_KB / RP_PFD are race knobs. */
#ifndef RP_PFD
#define RP_PFD 128
#endif
#ifndef RP_PFMIN_KB
#define RP_PFMIN_KB 36864       /* enable when state+c > 36 MiB (p >= 107):
                                 * node-raced -1.5% at 113, -0.5% at 127,
                                 * wash-to-negative at 89/101 */
#endif
        if (2 * vol * sizeof(cplx) > (size_t)RP_PFMIN_KB * 1024)
            p->xpf = RP_PFD;
        okm = okm && p->gct && p->gst;
    }
    if (!okm) {
        fft3d_destroy(p);
        return NULL;
    }
    for (int k = 0; k < L; ++k)
        for (int j = 0; j < L; ++j) {
            long double th = -2.0L * PIL * (long double)((k * j) % L) / (long double)L;
            p->w[(size_t)k * L + j] = (double)cosl(th) + I * (double)sinl(th);
        }
#if defined(RP_SKIPZ) || defined(RP_SKIPX) || defined(RP_SKIPY)
    p->fast = 1;                /* timing-only dev builds: WRONG OUTPUT, skip gate */
#else
    p->fast = self_check(p);
#endif
    return p;
}

void fft3d_destroy(fft3d_plan *p)
{
    if (!p) return;
    free(p->ke); free(p->ko); free(p->ctd); free(p->std_);
    free(p->gct); free(p->gst);
    free(p->j3); free(p->k3e); free(p->k3o);
    free(p->j2); free(p->k2);
    free(p->t1); free(p->w); free(p->tmp);
    gl_unmap(&p->gmap);
    if (p->arena) munmap(p->arena, p->alen);
    else { free(p->st); free(p->cpad); }
    free(p);
}
