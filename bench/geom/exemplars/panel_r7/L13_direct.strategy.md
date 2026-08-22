# L13_direct — strategy record

Geometry: **L = 13**, cube 13³ = 2197 complex doubles per volume (34.3 KiB),
forward, unnormalised, out-of-place, batched, single-threaded.
Implementation: `impl/L13_direct.c`. `fft3d_name()` → `L13_direct`.
Scored cases (cases.txt): B = 1, 16, 512.

First implementer for this geometry (wave-2 expansion). This entry is the
L17_matrixsimd design re-derived for L = 13, plus one genuinely new move that
only L = 13 permits (the fully register-resident matrix, below).

---

## Round panel_r6 (2026-08-21)

### Technique

Row-column: one dense length-13 DFT matrix per axis, three passes, conjugate
pair folded first (FFTW's dft-generic form — **adopted wholesale from
L17_matrixsimd rounds 1–5**, with attribution; their record has the full
derivation and every dead end, none of which were retried):

```
u_j = x_j + x_{13-j},  v_j = x_j - x_{13-j}          (j = 1..6)
P_k = x_0 + Σ_{j=1..6} cos(2π kj/13) u_j             (k = 0..6)
R_k =       Σ_{j=1..6} sin(2π kj/13) (-i v_j)        (k = 1..6)
X_k = P_k + R_k,  X_{13-k} = P_k - R_k,  X_0 = P_0
```

All surviving coefficients are REAL, so the driver's interleaved layout is the
SIMD layout: lanes = WC different lines (4 = zmm, 2 = ymm) from a contiguous
free index, every coefficient a lane-invariant broadcast, no cross-lane
arithmetic. `-i·v` = re/im swap + integer XOR of the imaginary sign bits (off
the FMA port). Tail chunks overlap the previous chunk and rewrite identical
bits. Passes and the two unavoidable fused plane transposes exactly as
L17_matrixsimd (Y: in→pb transposing, Z: pb→t1 transposing, X: t1→out plain,
Y/Z per 2.6 KiB L1-resident plane); X-first variant for batch ≥ 64 (their r3).

**The new thing — the whole matrix lives in registers.** The folded 13-point
transform has only **6 distinct cosine magnitudes** c_m = cos(2πm/13) (cos is
even, so cos(2πkj/13) = c_{fold(kj mod 13)} with no separate sign) and **6
distinct sines** s_m, with the sine signs (−1 when kj mod 13 > 6) compile-time
constants. All 12 constants are broadcast once per execute into 12 asm-opaque
registers (the pinning trick from L17_winograd r2 via L17_matrixsimd r3, with
attribution — extended here from 6-of-8 constants to *all* of them), and the
kernel is a **single fused sweep, fully unrolled**: per j it loads x_j and
x_{13−j} once, forms u and w = −i·v, and updates all 13 accumulators. Per
chunk: **13 line loads and ZERO coefficient loads**. Liveness: 13 accumulators
+ 12 pinned + a,b,u,w = 29 of 32 EVEX registers — this does not fit at L = 17
(8+8 constants + 17 accumulators), which is why their entry could pin only the
sines. The k=0 cosine row is 1.0, so P0 += u is a plain vaddpd (same port and
throughput as the FMA it replaces, per their r1 measurement — not
special-cased further).

The table-based kernel (pre-splatted cosine table read as full-width memory
operands, pinned sines, rolled j-loop — the literal L17 shape) is kept as
FORCE variants 0–5 for A/B; the all-pinned kernel is the shipped default.

**Mixed-width tail** (*adopted from L17_rader panel_r4 "512t" via
L17_matrixsimd panel_r5, with attribution*): the node's Gold 5218 has one
512-bit FMA unit but two 256-bit ports, so a 13-long free index costs 3 zmm
chunks (offsets 0,4,8) + 1 ymm tail at 11 instead of 4 zmm (0,4,8,9), and the
X pass 42 zmm + 1 ymm at 167 instead of 43 zmm. The ymm tail runs the
table-cosine w2 kernel with its own unpinned constants (pinning 24 registers
would starve the zmm body). The zmm group loops use the asm-opaque-bound trick
(L17_rader r4: gcc 11 ignores `#pragma GCC unroll 1` around an always_inline
callee).

**Determinism instead of a tuner.** No wall-clock tuning in create(): the exec
is a pure function of batch size and compile-time ISA (batch < 64: X-last;
≥ 64: X-first; mixed tail iff `__AVX512F__`), so repeated runs are
bit-identical by construction and the L17 bit-class machinery is not needed.
`-DL13_FORCE=0..9` pins any variant for experiments; `L13_XF_MIN` (default 64)
is the compile-time threshold, chosen so B=16 (cache-resident, X-last/X-first
measured as a tie, see below) stays X-last per the L17 r3 precedent and B=512
(past the node's 22 MB L3 at 36 MB in+out) gets X-first.

### Operation count

Per chunk (WC lines): 6 u-adds + 6 v-subs + 6 P0-adds + 36 cos FMA + 36 sin
FMA + 12 combine add/sub = **102 vector FP ops** (+ 6 shuffles + 6 XORs for
−i·v; + 32 shuffles/16 stores for the fused tile transpose when tr=1).
Per line: 360 real flop vs 1352 for the naive complex 13×13 matvec (3.8×).
Per volume (3·169 = 507 lines): 182.5 kflop; the driver's 5N log₂N yardstick
is 121.9 kflop, so reported GF/s × 1.5 ≈ real Gflop/s.

Chunks per volume: pure zmm 2·13·4 + 43 = **147**; mixed **120 zmm + 27 ymm**.
Node cycle floor at 1 zmm-op/cycle (ymm pairs on two ports): 120·102 + 27·51 =
**13 617 cycles = 4.7 µs at 2.9 GHz** (pure zmm: 15.0k = 5.2 µs, so the mixed
tail is worth −9% of port time on the node; on wallaby's two 512-bit units it
is FP-neutral and won anyway, see below).

### What was measured — wallaby (Gold 6448Y, shared and NOISY: sd within runs
up to 32%, single runs mis-ranked variants by 2×; min across ≥3 runs is the
only statistic, re-learning L17_matrixsimd r2's warning)

Forced variants, per transform, min of 3–5 runs:

| variant (FORCE) | B=1 | B=16 | B=512 |
|---|---|---|---|
| 512b table pure X-last (0) | 3.664 | — | — |
| 512b table + ymm tail X-last (4) | 3.677 | — | — |
| 512b all-pinned pure X-last (6) | 3.460 | — | — |
| **512b all-pinned + ymm tail X-last (8)** | **3.296** | 3.347 | 4.477 |
| 512b all-pinned + ymm tail X-first (9) | 4.235 | 3.338 | **3.935** |

MKL (mkl_dfti, same machine, same cases, best windows): B=1 3.567, B=16 3.59,
B=512 4.30 µs/t. **Unlike L=17, MKL is genuinely good at 13** (34 GF/s at B=1
— it clearly has a real small-prime codelet), so the margin is 1.08–1.15×,
not the 5× the L=17 entries enjoy. The node ratio may differ.

Shipped defaults (X-last all-pinned mixed at B=1/16, X-first at 512):
PASS at every batch, rel_l2 = 2.843e-16 … 2.866e-16, rel_max ≤ 4.3e-16,
bit-identical across re-runs at B = 1, 16, 512, 1024. AVX2 host (wombat,
Haswell, shared): PASS 2.851e-16, repeatable, 11.4 µs/t at B=8 (real-ymm w2
path, unscored fallback).

### What was tried and did NOT work

1. **The table-cosine kernel as the primary** (the literal L17 shape: 42
   pre-splatted zmm cosine operands per chunk + pinned sines). First
   measurement 7.05 µs at B=1; with min-of-3 discipline 3.66 vs the all-pinned
   kernel's 3.30 (**−10%**). Static analysis showed no spills (1027 insns, 33
   stack refs) — the cost is pure load traffic: 2.6 KiB of table per chunk
   against only 102 FP ops to hide it under. At L=17 the same table amortised
   over 148 FP ops and 42% more loads-per-FMA was tolerable; at L=13 it is
   the bottleneck. **Lesson for other small-L entries: below ~15 distinct
   constants, pin everything and unroll; the pre-splatted table is a
   register-pressure workaround, not a default.**
2. **X-first at B=1: 4.24 vs 3.30 µs (+28%)** — much worse than L17's +5%.
   Plausibly because at 13³ the X pass is 43 of 147 chunks (29%, vs 25% at 17³)
   and its strided 13-row read pattern gets no help from anything at B=1.
   X-first only pays past cache (B=512: −12%). Threshold stays 64.
3. **Single runs on wallaby today were worthless**: the same binary measured
   7.0, 8.5, 4.1, 3.3 µs at B=1 across windows (sd up to 33% *within* a run).
   Every decision above is min-of-3+ or it is not a decision.
4. **Not retried, on the strength of other entries' records** (attributed in
   place): scalar-constant + embedded broadcast (L17_matrixsimd r1 item 1,
   gcc 11 catastrophe), masked/narrow tail chunks (their r1 item 8 — overlap
   is free), letting gcc unroll rolled accumulator loops (r1 item 7),
   NT stores issued from pass stores (r1 item 10), interleaved A/B timing
   (r1 item 12), paired-volume lane packing (r2 item 3 — structural),
   alternating NT A/B (r1 item 11).

### Expectations for the node (monitor, please measure)

* B=1: if the kernel is FMA-port-bound like its L17 sibling (1.045 ops/cycle
  measured there), the mixed floor predicts **~4.7–5.5 µs**. Note the node is
  *slower* than wallaby here (one 512-bit FMA unit at 2.9 GHz vs two at 4.1).
* B=16 ≈ B=1 per transform (1.1 MB working set, cache-resident).
* B=512 streams (36 MB > 22 MB L3): X-first plain stores; expect the usual
  memory-bound overhead on top of the compute floor.
* MKL's node numbers at L=13 are the real unknown — its wallaby 3.57 µs will
  not transfer 1:1. If MKL lands ~6–8 µs the margin is comfortable; if its
  prime codelet holds up, this becomes the tightest geometry on the board.

### Next

1. Read the node numbers. If B=1 lands near 4.7 µs the kernel is port-bound
   and the only B=1 lever left is fewer FP ops: the primitive-root (g=2)
   cyclic reindexing splits the folded 6×6 cosine block as x⁶−1 =
   (x³−1)(x³+1) (cyclic-3 + negacyclic-3, 36→18 FMAs on that side, ~−6% ops
   overall). L17_matrixsimd r2 measured its −12% ops → −1.4% time on the
   node, so expect little; try only if B=1 is confirmed port-bound.
2. The overlapping 4th tile in the transposing store recomputes 3 columns
   (8 shuffles + 4 split stores per tr=1 chunk); a 4×16B extract-store of the
   13th column would halve that. Matters on wallaby's port scheme (512-bit
   shuffles compete with FMA on port 5 there), probably invisible on the node
   (shuffles hide under the single-FMA stream). Low priority.
3. If the node's B=512 shows a large gap over B=16: NT staging (L17's
   l17_ntcopy shape) and cross-volume pipelining (their r4) are the known
   levers, in that order of likely payoff — but the node rejected NT at L=17's
   nearly identical arithmetic intensity every round, so demand the number
   first.
4. If a rival L13 entry appears (L13_rader is stubbed): Rader-13 needs a
   length-12 convolution (12 = 4·3, composite — nicer than Rader-17's 16) but
   pays the gather + complex-constant pointwise stage; this entry's record is
   the baseline it has to beat at 3.30 µs wallaby-B=1.

---

## Round panel_r7 (2026-08-22)

Round panel_r6 was abandoned before its timing pass, so nothing here has ever
been node-scored; the r6 code carried over unchanged as the starting point.
Three changes this round, all measured pinned on wallaby; one built and
rejected.

### What changed

1. **Last-column extract-stores in the transposing store (`LASTCOL_`).** The
   tr=1 store previously covered the 13 output columns with 3 full WC×WC
   tiles plus a 4th *overlapping* tile that recomputed and rewrote 3
   identical columns just to deliver column m=12 (zmm: 8 shuffles + 4 wide
   stores). Column 12 is exactly one register (X_12 = P_1 − R_1), so it is
   now stored directly as WC 16-byte column stores. gcc 11 compiles the
   `(v2){c[2f],c[2f+1]}` constructors well: one `vsubpd` + 3 `vpermpd` + 4
   xmm stores per zmm chunk (net −5 port-5 ops and −192 stored bytes per
   tr=1 zmm chunk; same FP count, 102). Same treatment for the w2 tail's
   overlapping TILE2_. This was r6's "next" item 2, promoted to first
   priority once it also showed up batched (below).
2. **sysconf L2 gate replaces the fixed `batch >= 64` X-first threshold**
   (*adopted from L13_rader panel_r6*, their plan-time deterministic gates):
   X-first iff `32·B·2197 > sysconf(_SC_LEVEL2_CACHE_SIZE)`. Fixed
   thresholds tuned on wallaby's 2 MB L2 are wrong on the node's 1 MB. On
   wallaby the scored cases select exactly as before (B=1/16 X-last, B=512
   X-first); on the node B=16 now goes X-first (ws = 1.07 MiB > 1 MB), which
   wallaby measured as a tie in its own cache-resident regime (r6: 3.338 vs
   3.347) — low-risk bet that spilled-L2 stores prefer the plane window.
   Still no wall-clock tuner: sysconf is a fixed host property, runs stay
   bit-identical.
3. **Staged-Z variant built, measured, and NOT shipped** (`l13_exec_xsp_mx`,
   FORCE=10; *idea from L13_rader panel_r6 / L8_radix8 via L17_rader r4*):
   Z pass lands in a hot 2.7 KB plane, then one sequential 2704 B memcpy
   into `out`. On wallaby B=512 (out L3-resident, L2-streaming) it LOST:
   min 2209 µs vs plain X-first 1899 µs across 6 interleaved pinned runs
   (+16%). My transposing tile stores into a 2.7 KB plane window are already
   cheap — the burst copy pays for L13_rader because their unstaged pattern
   (interleaving stores into cold lines) was much worse, not because staging
   is generally right for this architecture. Kept as FORCE=10 **for the
   monitor**: the node's B=512 truly streams (36 MB > 22 MB L3), a regime
   wallaby cannot reproduce; if the node B=512 lags B=16 badly, A/B
   `-DL13_FORCE=10` there.

### Operation count

Unchanged FP: 102 vector FP ops per chunk, 120 zmm + 27 ymm chunks per
volume, node port floor still ~13.6k cycles ≈ 4.7 µs at 2.9 GHz. The
LASTCOL_ change removes per tr=1 zmm chunk: 8 shuffles + 4 zmm stores,
adds 3 vpermpd + 4 xmm stores; per volume (78 tr=1 zmm + 26 tr=1 ymm
chunks): ~−400 port-5 ops and ~−17 KB of store traffic.

### Measurement methodology (re-learned, sharpened)

wallaby today is **bimodal per process instance**: the same binary pinned to
the same core lands at either ~3.28 µs or ~4.35 µs at B=1 (within-run sd
0.03% both ways), with the fast mode appearing in roughly a third of runs.
Un-pinned tryout numbers are 2× wrong (8.5 µs for the same build). r6's
F8-vs-F6 ranking (3.296 vs 3.460) did NOT reproduce under min-of-3 — it took
**min over 8 interleaved pinned process instances** to recover it (F8 fast
mode 3.299, F6 fast mode 3.444). Every decision this round is min-of-≥6
pinned (`taskset -c 49`, alternating variants run-by-run). MKL alternated in
the same windows as the clock canary: B=1 3.571 µs, matching r6's 3.567,
so the machine state was comparable across rounds.

### What was measured — wallaby (Gold 6448Y, pinned `taskset -c 49`, min over ≥6 process instances, gcc 11.4, tryout flags)

| case | r6 code (re-measured today) | this round | MKL (same core, same day) |
|---|---|---|---|
| B=1 | 3.299 µs | **3.102 µs** (typical fast-mode 3.27) | 3.571 µs |
| B=16 | 54.09 µs/call (3.381/t) | **52.12 µs/call (3.257/t)** | 57.26 (3.579/t) |
| B=512 | 2004 µs (3.915/t) | **1864 µs (3.641/t)** | 1913 (3.736/t) |

Ahead of MKL in all three regimes on wallaby (1.15× / 1.10× / 1.03×). The
B=512 gain (−7%) is the LASTCOL_ change: less store traffic against a
streaming L2 (the exec selection there is unchanged). rel_l2 = 2.84e-16 …
2.87e-16 at B = 1, 3, 16, 512; bit-identical across re-runs at every batch;
AVX2 fallback verified on wombat (PASS, 9.5 µs/t at B=8, unscored).

### What was tried and did NOT work

1. **Staged-Z burst copy at B=512 on wallaby: +16%** (2209 vs 1899 µs,
   min-of-6 interleaved). See item 3 above — architecture-dependent, not a
   general win; shipped only as FORCE=10 for a node A/B.
2. **Not retried, per existing records**: pb double-buffering (L13_rader r6
   item 2 measured it a no-op — the OOO core already overlaps planes);
   NT stores (node rejected at L17 every round); primitive-root cyclic-3 +
   negacyclic-3 split of the cosine block — my kernel's version saves only
   ~6 ops of 102, and L17_matrixsimd r2 measured −12% ops → −1.4% time, so
   it stays parked until the node proves B=1 port-bound.

### Expectations for the node (monitor, please)

* First-ever node numbers for this geometry. B=1 prediction unchanged:
  **~4.7–5.5 µs** if port-bound (the node runs one 512-bit FMA unit at
  2.9 GHz — slower than wallaby here). B=16 now X-first (sysconf gate).
* If node B=512 per-transform lags B=16 by ≫10%: A/B `-DL13_FORCE=10`
  (staged Z) — the one lever wallaby could not evaluate honestly.
* MKL's node number at L=13 is still the unknown that decides the margin.

### Next

1. Node feedback first: port-bound or not at B=1 decides whether the
   arithmetic levers (cyclic split, −6% ops) are worth anything.
2. If the node's B=16 X-first bet loses, the gate constant is one line
   (`ws > l2c` → `ws > 2*l2c` puts B=16 back to X-last on the node).
3. The 3.10 µs B=1 fast-mode floor on wallaby is ~12.4k cycles ≈ 1.8× the
   two-FMA-port floor; remaining gap is load/store latency through pb/t1.
   The untried lever: interleave the last Y group of plane x with the first
   Z group reads (software pipelining across the pb dependency), analogous
   to L13_rader's zkern/ykern pipeline (+5% for them). Worth one experiment
   next round.
