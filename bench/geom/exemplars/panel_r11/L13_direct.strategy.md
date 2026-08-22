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

---

## Round panel_r8 (2026-08-22)

First round with node feedback (panel_r7): won all three L=13 cells (5.901 /
6.089 / 8.396 µs/t), but **B=512 is the thinnest margin on the board (1.08×
over mkl2026)** and the r7 verdict names it the round's priority. B=1 landed
at 5.901 = 17.1k cycles at 2.89 GHz against the 13.6k port floor (1.26×), so
B=1 is NOT purely port-bound — the arithmetic levers (cyclic split) stay
parked; the residue is load/store, which is what this round attacks. Three
changes shipped, all adopted from other entries' node-proven results.

### What changed

1. **t1 x-plane stride padded 338 → 344 doubles (`L13_T1P`, 2752 B = 43
   cache lines)** — *L23_rader panel_r6's padding rule ("odd number of
   cache lines, 64 B-aligned") applied to the one big stride this plan
   owns.* At the driver's natural 338 (2704 B = 42.25 lines), plane j of t1
   sits at byte 16·j (mod 64), so **3/4 of the X pass's zmm accesses to t1
   were cache-line-SPLITTING** — ~410 split loads per volume in X-last,
   ~410 split stores per volume in X-first (the X pass is 42 zmm chunks ×
   13 rows). At 344 they are all 64 B-aligned, and 2752's mod-4096 residue
   comb differs from in/out's 2704, so the X pass's t1 stream can no longer
   4K-alias the driver-buffer stream it runs against (the L=8 lesson: the
   in/out side is unreachable, but the scratch side is mine). The pad tail
   is never read; in-plane layout (169 contiguous complex) is unchanged.
2. **Streaming prefetch exec `l13_exec_xfp_pf_mx` (FORCE=11), selected only
   past this host's L3** — *the r7 verdict §4.5 rule ("hide the RFO with
   prefetchw, do not avoid it with NT stores", node-picked at every
   streaming cell at L=6/8/23/36/64), lead placement and L3 gate from
   L13_rader panel_r7 (their pw A/B at wallaby B=2048: −6%; their pfw at
   L3-resident B=512: ~+3% LOSS — hence the gate), read-side idea from
   L17_winograd r2's cross-volume prefetch (−4.4% streaming) and
   L64_radix8's node-picked slabpf.* Schedule: 43 `prefetchw` lines of out
   plane x+1 issued before plane x's Y group (lead ≈ one Y+Z group ≈ 700
   node cycles); plane 0's out lines paced one per X-pass chunk; 43
   `prefetcht1` lines of the NEXT volume's input per plane (slice x),
   issued between the Y and Z groups. Selection is now three-tier and still
   deterministic: ws ≤ L2 → X-last; L2 < ws ≤ L3 → X-first; ws > L3 →
   X-first+pf (sysconf both levels; node: B=1 X-last, B=16 X-first, B=512
   X-first+pf; wallaby scored cases unchanged, B≥1024 pf). `-DL13_PW=0` /
   `-DL13_PFIN=0` kill either half for node A/Bs; FORCE=12 is staged-Z+pf.
3. **`#pragma GCC optimize("unroll-loops")` at file top** — *build-flag
   parity, from L45_pfa panel_r7*: tryout builds carry `-funroll-loops`,
   the scored build has not always; pin the codegen the dev numbers were
   taken with. No-op on wallaby (flag already present); insurance on the
   node. The deliberately-rolled kernel loops are protected by asm-opaque
   bounds and unaffected. Also ran their scalar-instruction audit: the
   three hot execs are 998–1107 total insns with 67–113 scalar ops and
   53–79 stack refs — no offset-table pathology, nothing to fix.

### Operation count

Vector FP unchanged: 102 ops/chunk, 120 zmm + 27 ymm chunks/volume, node
port floor still ~13.6k cycles ≈ 4.7 µs at 2.89 GHz. The t1 pad deletes
~410 line-split penalties per volume from the X pass (loads in X-last,
stores in X-first — the latter hit the single store port on the node). The
pf exec adds ~1.1k prefetch + bookkeeping insns per volume, streaming
regime only.

### What was measured — wallaby (Gold 6448Y, pinned `taskset -c 49`, min
over ≥4 interleaved process instances, alternating variants run-by-run;
MKL canary 3.576 µs at B=1 matches r7's 3.571, so windows are comparable)

| case | r7 binary (same windows) | this round | Δ |
|---|---|---|---|
| B=1 | 3.272 µs | **3.179 µs** (typical fast-mode 3.19) | **−2.8%** |
| B=16 | 51.893 µs/call (3.243/t) | 51.837 µs/call (3.240/t) | tie |
| B=512 | 1863.7 µs/call (3.640/t) | **1685.5 µs/call (3.292/t)** | **−9.6%** |
| B=2048 (unscored streaming proxy, 144 MB > 60 MB L3) | 12322 µs (6.017/t) | **9390 µs (4.585/t)** | **−23.8%** |

Attribution of the deltas: B=1 and B=512 are the t1 pad (+ pragma, a no-op
here) — both defaults on wallaby run WITHOUT pf (36 MB < 60 MB L3). B=2048
isolates pf same-window: no-pf FORCE=9 12310 vs pf FORCE=11 **9390 (−23.7%)**,
5-round stable (9390–9515). The pf split (same window): read-half only
(`L13_PW=0`) 10956; pfw-half only (`L13_PFIN=0`) 13186 — **the input-side
prefetch carries most of the wallaby win and the two halves are strongly
synergistic** (SPR's hardware prefetchers already handle the out stream
well; the node's CLX has less MLP, so pfw may matter relatively more there —
the kill switches exist so the monitor can check cheaply if B=512 behaves
oddly). Default-gate sanity: default binary at B=2048 = 9432 ≈ FORCE=11 ✓.

Correctness: PASS rel_l2 = 2.84e-16 … 2.87e-16 at B = 1, 3, 8, 16, 512,
1024, 2048 (default and FORCE=11/12 and both kill-switch builds);
bit-identical across re-runs at every batch tried; AVX2 fallback PASS on
wombat (9.4 µs/t at B=8, unpinned, unscored).

### What was tried and did NOT work

1. **Staged-Z + pf (FORCE=12) at streaming B=2048: 10005 vs plain X-first+pf
   9409 (+6%)** — staging loses even with the RFOs hidden; r7's staged
   verdict stands with pf on top. Kept only as a FORCE for the node.
2. **pfw alone (PFIN=0) on wallaby streaming: 13186, i.e. no better than no
   prefetch at all (12310, cross-window)** — on THIS machine the out-stream
   RFO is not the binding cost; shipping pfw without the read half would
   have been a null. Recorded because it is the opposite balance to what
   the node evidence (L=6/8/23/36/64 picks) suggests for CLX — do not
   conclude from wallaby that pfw is dead on the node.
3. **Not retried, per existing records**: software pipelining across the pb
   junction (my r7 "next" item 3) — dropped without an experiment this
   round on the node's now-threefold rejection of scheduling attacks at
   L=17 (rader `ov` r5 0/4, rader `dz` r7 0/4, matrixsimd deferred-Z
   regressing its one selected cell) and the r7 verdict's L=17 synthesis
   "delete uops, don't reschedule them"; the cyclic-3/negacyclic-3 split
   stays parked (B=1 measured 1.26× floor on the node, not port-bound).

### Expectations for the node (monitor, please)

* **B=512 is the round's target**: pf exec activates (36 MB > 22 MB L3) on
  top of the t1 pad. Wallaby's honest proxy moved −24% (pf) and −10% (pad,
  L2-streaming regime); anchored on r7's measured 8.396, I expect
  **~6.5–7.5 µs/t**, which would take the MKL margin from 1.08× to ~1.3×.
  If it regresses instead, A/B `-DL13_PW=0` and `-DL13_PFIN=0` (one build
  each) — the halves behave very differently per machine (item 2 above).
* B=16 (X-first, t1-pad aligned X-pass stores onto the single store port):
  wallaby says tie, CLX store port says maybe −1–3%. B=1 (X-last, aligned
  X-pass loads): wallaby −2.8%; expect **~5.7–5.9 µs**.
* The r7 verdict's FORCE=10 ask is superseded: staged lost to plain+pf on
  the honest streaming proxy (item 1); if B=512 still lags, A/B FORCE=12
  (staged+pf) rather than 10.

### Next

1. Node numbers decide the pf balance (PW vs PFIN) on CLX; if B=512 lands
   ≥ −15% the streaming story is closed and the remaining gap is B=1's
   3.4k-cycle non-FP residue.
2. The one unexplored B=1 lever consistent with "delete uops": the Y pass
   reads `in` with 3/4 split loads (rows are 208 B — inherent to the
   driver layout, per-axis unfixable)… unless the Y and X passes swap lane
   axes so the split-heavy axis reads the aligned t1 instead. Sketch only;
   needs a full re-derivation of the pass strides before it is credible.
3. If a wallaby session shows the fast mode again (~3.18), a perf-counter
   run on the node (`ld_blocks.store_forward`, `ld_blocks_partial.
   address_alias`) would settle whether the remaining B=1 residue is the
   pb store→load junction — the monitor's standing L=6/L=8 counter recipe
   applies verbatim here.

---

## Round panel_r9 (2026-08-22)

r8 node results: won all three cells for the second round (B=1 5.725, B=16
6.039, B=512 8.110 µs/t), but B=512 stayed the thinnest margin on the board
(1.13× over mkl2026) and the r8 verdict's transfer study showed my wallaby
prefetch win (−24%) bought ≤2.5% on the node. The verdict's synthesis for
this round: only *deletions* transfer (and modestly); scheduling and memory
mechanisms do not. So this round is three pure deletions, all exactly
value-preserving (the shipped binary's outputs are **bit-identical to the r8
binary's** at B=1/16/512 — both kernel changes are IEEE-exact rewrites).

### What changed

1. **Removed the `#pragma GCC optimize("unroll-loops")`** shipped in r8
   (*directed by the r8 verdict §3c, on L17_rader's measurement*): the scored
   build has carried `-funroll-loops` all along (the r7 "build-flag gap"
   never existed), and the pragma form is a ~2% node TAX because
   `optimize()` rebuilds the whole per-function option set. On wallaby
   (where the flag is on the command line anyway) removal measured a wash
   (3.176 vs 3.184 at B=1, inside noise), exactly as expected — this item's
   payoff, if any, is node-side.
2. **The mixed execs' ymm tail chunks now run the all-pinned `chunk13p_w2`
   kernel** instead of the table-cosine `chunk13_w2`. The tail's 12 ymm
   constants (D0–5 cosines from ctd4, Q0–5 lane-signed sines from stab4) are
   loaded once per execute and deliberately NOT asm-pinned (pinning 24
   registers would starve the zmm body — r6's reasoning stands); gcc spills
   them and each tail chunk does ≤12 spill-reloads instead of **42 ctab4
   loads + 6 Q reloads**. Net ~−810 L1 loads per volume (27 tail chunks).
   Bit-exact swap: both kernels share the fold, accumulation order and store
   body (the table kernel's k=0 row is FMA(1.0,u,P0), which is exact).
3. **The −i sign moved from a per-chunk XOR into the sine tables.**
   MULI(t) was swap-re/im + `vpxor` with a sign mask on the imaginary lanes
   (6 XORs per chunk, every kernel). The sign pattern is per-LANE and
   constant, so stab8/stab4 now hold lane-alternating splats (s,−s,s,−s,…)
   and MULI degenerates to the swap alone. (−σs)·v = σs·(−v) exactly in
   IEEE-754, so this is bit-identical too. Deletes **882 vector XORs per
   volume** (720 zmm + 162 ymm) at zero register cost, no new constants, no
   new ops. On CLX, 512-bit `vpxorq` issues on p0/p5 — p0 is the one FMA
   port and p5 carries the tile shuffles, so every deleted XOR comes off a
   contended port. (This generalizes L17_winograd/L17_matrixsimd's
   "bake the signs in at compile time" from per-coefficient signs to the
   per-lane −i sign; any entry using the swap+XOR MULI idiom with
   lane-invariant real coefficients can apply it — the sine splat just
   stops being lane-invariant.)

### Operation count

Per chunk now **102 vector FP ops + 6 swaps + 0 XORs** (was +6 XORs); chunk
census unchanged at 120 zmm + 27 ymm per volume; node port floor unchanged
~13.6k cycles ≈ 4.7 µs at 2.89 GHz. Loads: tail chunks 13 line loads + ≤12
constant reloads (was 13 + 48).

### What was measured — wallaby (Gold 6448Y, pinned `taskset -c 49`, min
over ≥6 (10 at B=1) interleaved process instances, alternating variants
run-by-run; MKL canary 3.569 µs at B=1 vs 3.571/3.576 in r7/r8, so windows
are comparable across rounds)

Decomposition at B=1 (fast-mode minima, same windows):
r8 code 3.184 → pragma removed 3.176 (wash) → + all-pinned tail 3.105–3.120
(−2.0%) → + sign-folded tables **3.061 (−3.8% cumulative)**. The final
binary also lands in wallaby's fast mode 8/10 instances vs ~5/10 for r8 —
not load-bearing (mode selection is machine state), but consistent.

| case | r8 code (same windows) | this round | Δ |
|---|---|---|---|
| B=1 | 3.184 µs | **3.061 µs** | **−3.8%** |
| B=16 | 52.043 µs/call (3.253/t) | **49.946 µs/call (3.122/t)** | **−4.0%** |
| B=512 | 1678.5 µs/call (3.278/t) | **1600.2 µs/call (3.125/t)** | **−4.7%** |
| B=2048 (unscored streaming proxy, pf exec active) | 9541 µs | **9281 µs** | −2.7% |

Correctness: PASS rel_l2 = 2.843e-16 (B=1), 2.862e-16 … 2.872e-16 at
B = 8, 16, 512, 2048; **outputs bit-identical to the r8 binary's** at
B=1/16/512; bit-identical across re-runs at every batch; AVX2 fallback PASS
on wombat (B=8, unpinned, unscored); FORCE=2/11/12 and the `-DL13_PW=0`
`-DL13_PFIN=0` kill-switch builds all PASS and repeatable.

### What was tried and did NOT work

Nothing shipped failed this round; two things were considered and parked
with reasons rather than measured to death:

1. **Folding the remaining MULI swap away** — impossible: it is the one
   genuine cross re/im mixing in the kernel; moving it from the 6 w's to
   the 6 R's is count-neutral. Not attempted.
2. **A J-vector FMA combine** (X_k = fma(J,R',P) with J=(1,−1,…) pinned) as
   the XOR deletion — superseded by the table sign-fold, which needs no
   13th pinned register and no store-body fork. The table form is strictly
   better; do not revisit the J form.
3. Still parked per prior records: cyclic-3/negacyclic-3 split (B=1 is
   1.22× floor, not port-bound), software pipelining across pb (node
   rejected scheduling attacks 3× at L=17), staged Z (lost twice), NT
   stores (node-rejected every round).

### Expectations for the node (monitor, please)

Anchored on the r8 verdict's 1.80× wallaby→node band for this entry at B=1
(and the observed ~2.46× at streaming B=512): **B=1 ~5.4–5.6 µs, B=16
~5.7–5.9, B=512 ~7.6–7.9**. Standing asks, all one-build/one-run:

* **B=512 PW/PFIN split** (unchanged from r8, still the thinnest cell):
  `-DL13_PW=0` and `-DL13_PFIN=0`, one build each — the halves behaved
  oppositely on SPR vs what CLX evidence suggests.
* **B=16 X-first vs X-last**: `-DL13_FORCE=8` at B=16 — the sysconf gate's
  X-first bet at B=16 (ws=1.07 MiB > 1 MB L2) has never been A/B'd on the
  node; wallaby cannot price it (2 MB L2).
* The standing perf-counter request (`ld_blocks_partial.address_alias`,
  `ld_blocks.store_forward` at B=1) would settle whether the remaining
  ~2.9k-cycle B=1 residue is the pb junction or the in/out split accesses.

### Next

1. If the node confirms ~−3% at B=1, the residue is ~1.9k cycles over
   floor; the counters (ask 3) decide between the two remaining theories
   (pb store→load junction vs inherent in/out split accesses). If it is
   the junction, the one untried shape is having the Y pass emit the pb
   plane in Z-load order (column-major tiles) so Z's tail ymm loads are
   store-contained — sketched, not derived; needs the counter first.
2. If B=512 lands ≥7.9 with pf still ≤−2.5%, the streaming story on CLX is
   load-side, and the pfr43 slice pacing (43 lines/plane) should be
   re-tuned against the node's smaller L2 — one constant.
3. If MKL's node number moves again (its batched baselines have regressed
   twice), the B=512 margin may widen for free; do not spend the round on
   it before reading the leaderboard.

---

## Round panel_r10 (2026-08-22)

r9 node results: all three cells won for the third round (B=1 5.739, B=16
5.957, B=512 7.965 µs/t), margins 1.32×/1.28×/1.14× over mkl2026; B=512
still the thinnest cell on the board. The r9 verdict's inputs to this round:
(a) my wallaby deltas keep transferring at ≲half strength and only deletions
transfer at all; (b) perf counters are withdrawn cluster-wide — the panel's
instrument is now a timed in-plan A/B routed through `fft3d_description()`
(eleven entries did it in r9, every one produced a readable node number);
(c) the **one mechanism class with a positive node result** is association
order of a closed-arithmetic codelet (3.3–6.2% on CLX at L=6, sign-inverted
on SPR), and the verdict names `chunk13` as a propagation target. This round
ships one measured default change, one controlled twin experiment, and the
instrument to read both on the node.

### What changed

1. **The B=1 default flipped X-last → X-first: the cache-resident X-LAST
   tier is deleted.** While instrumenting the twins I re-measured the pass
   orders and found r6's "X-first +28% at B=1" — the entire justification
   for the X-last tier — is STALE: it was measured before the r8 t1 pad,
   whose whole point was to fix the split/4K-aliased t1 accesses of the X
   pass, i.e. exactly what penalised X-first. Driver-level, pinned
   (`taskset -c 49`), interleaved, min over 8 process instances on wallaby:
   **X-first 2.571 vs X-last 3.021 µs at B=1 (−15%)**, 41.6 vs 50.0 µs/call
   at B=16 (−17%), B=512 unchanged (already X-first). Selection is now:
   ws ≤ L3 → X-first plain, ws > L3 → X-first+pf (still sysconf, still
   deterministic, no tuner). **On the node this changes only the B=1 cell**
   (at B=16 the old L2 gate already picked X-first there; wallaby's 2 MB L2
   is why both cells moved here). FORCE=8 (X-last) is the rollback;
   `ab[B1]=xl…,xf…` on the node's own leaderboard line adjudicates
   immediately. The refactor around it is proven value-safe: the three
   scored default execs compile to instruction-identical asm modulo label
   numbering against the r9 exemplar source (checked with
   `-march=cascadelake` at the tryout flag set).
2. **Association twins of `chunk13p`** (*the r9 verdict §6 instruction;
   mechanism class from L6_pfa/L6_unrolled's codelet-order result, with
   attribution*). Same fold, same 13-loads-per-chunk, only the combine
   stage's DAG differs; the j=1..5 accumulation is one shared macro so the
   twins cannot drift from the default kernel:
   * `chunk13q` ("T1", FORCE 13/15): count-neutral. X_k = P_k + R_k as
     before, X_{13−k} = fma(−2, R_k, X_k) — half the outputs become
     store-feeding FMAs (the class that won on CLX at L=6) at the price of
     one latency link. Verified in asm: exactly +36 contracted FMAs per
     exec, no spill growth.
   * `chunk13s` ("TS", FORCE 14/16): the faithful transcription of the CLX
     winner's shape. The j=6 sine row is deferred past the combine: R runs
     j=1..5, A_k/B_k join early, and **every output is finished by one
     twiddle-carrying FMA of w6** (signs = the j=6 fold row). Costs +6
     vector FP ops/chunk (102→108) — a deliberate bet that B=1 is not
     port-bound (node-measured 1.22× over floor, three rounds).
   Both PASS everywhere with their own fingerprints (2.886e-16…2.923e-16),
   both repeatable.
3. **In-plan timed discriminator, INSTRUMENT ONLY** (*adopted from
   L6_unrolled r9's ab1 ← L36_pfa r8, with attribution*): create() races
   X-last, X-first, T1, TS at tb = min(batch,16) volumes on private
   buffers (interleaved round-robin, min of 9 trials, licence-warmed,
   ~10 ms, unscored) and appends `ab[B<tb>]=xl…,xf…,q…,s… ns/vol` to the
   description, which the driver emits into every leaderboard JSON. It
   NEVER changes the pick — the exec stays a pure function of
   batch/ISA/sysconf, so there is no L36-style pick lottery and re-runs
   stay bit-identical. At tb=16 the xl/xf reading is the standing B=16
   gate question measured on the node itself. `-DL13_AB=0` removes it.

### Operation count

Default kernels unchanged: 102 vector FP ops/chunk, 120 zmm + 27 ymm
chunks/volume, node port floor ~13.6k cycles ≈ 4.7 µs at 2.89 GHz. T1 is
count-neutral (12 combine ops stay 12, six become FMAs); TS is 108
ops/chunk = +5.9% port ops (the shared j=6 sine row is applied to A and B
separately) traded for an all-FMA output stage and a one-link-shorter R
chain.

### What was measured — wallaby (Gold 6448Y, pinned `taskset -c 49`, min
over ≥6 (8 at B=1) interleaved process instances; MKL canary 3.601 µs at
B=1 vs 3.569/3.571/3.576 in r7–r9, windows comparable)

Driver-level pass-order/twin race, B=1 (µs/t, min over 8 instances,
`-DL13_AB=0` builds): X-last(F8) 3.021, default-r9-shape 3.070, **X-first
(F9) 2.571**, X-first+T1 (F15) 2.644, X-first+TS (F16) 2.672. At B=16
(µs/call): X-last 50.0, **X-first 41.6**, T1 43.3, TS 43.5. At B=512:
X-last 2203, X-first 1616 (= r9 default path), T1 1652, TS 1660.

Shipped default vs r9, scored cases:

| case | r9 code | this round | Δ |
|---|---|---|---|
| B=1 | 3.061 µs | **2.577 µs** | **−15.8%** |
| B=16 | 49.946 µs/call (3.122/t) | **41.489 µs/call (2.593/t)** | **−16.9%** |
| B=512 | 1600–1620 µs/call | 1616–1874 (same binary path, asm-identical; window drift) | none (code unchanged) |

Correctness: PASS rel_l2 = 2.826e-16 (B=1), 2.872e-16 (B=16), 2.866e-16
(B=512); B=1/16 fingerprints legitimately differ from r9 (pass order is a
different rounding order), B=512 unchanged. Bit-identical across re-runs at
every batch; FORCE=13/14/15/16 PASS and repeatable; AVX2 fallback PASS on
wombat (2.862e-16 at B=8, unscored). Wallaby margin over MKL is now
1.40×/1.46×/1.15× (was 1.15×/1.10×/1.03× in r7).

### What was tried and did NOT work

1. **The association twins lose on wallaby/SPR, everywhere**: T1 +2.8% /
   TS +3.9% at B=1 driver-level (+0.3%/+3% by the in-plan instrument),
   +2–4% at B=16, +2.2%/+2.7% at B=512. This is the expected SPR side of
   the L6 result (SPR preferred the add-join there too, and CLX inverted
   it) — so the twins ship as FORCE variants + instrument readings, NOT as
   defaults. **The node's `ab[B*]` fields decide**; adopt next round only
   if q or s reads ≥2% under xf there.
2. **r6's X-first-at-B=1 penalty (4.24 vs 3.30, +28%) is retracted as a
   gate justification** — it described the pre-pad code. Lesson worth
   lending: after a layout fix (padding, de-aliasing), re-run the pass-order
   A/Bs that were decided on the broken layout; a structural verdict can be
   an artifact of the layout it was measured on.
3. Still parked per prior records: cyclic-3/negacyclic-3 split (B=1 still
   not port-bound), software pipelining across pb (3× node-rejected class),
   staged Z (lost twice), NT stores (node-rejected every round).

### Expectations for the node (monitor, please)

* **B=1 is the round's bet**: X-first replaces X-last. r9's wallaby→node
  ratio for this entry (1.87× at B=1) maps 2.577 → **~4.8 µs**; band
  4.8–5.6. Pre-registered: if the cell reads ≥5.74 (no better than r9),
  the flip bought nothing on CLX — A/B `-DL13_FORCE=8` (one build), and
  the `ab[B1]` field on the default line already contains the same answer
  for free. If it lands ≤5.4, the L2-resident pass-order lesson transfers
  and L13_rader should hear about it (their y→z→x order is my old X-last).
* **B=16 and B=512 defaults are byte-identical paths to r9** (X-first was
  already selected there by the old gate; asm diff confirmed) — expect
  5.9–6.0 and 7.9–8.0 unless the machine moves. The `ab[B16]` field now
  prints the node's own X-last-vs-X-first at B=16, closing the r8/r9
  standing ask without a forced build.
* **Twins**: read `ab[B1]`/`ab[B16]` q/s vs xf. CLX inverted the L6
  association result; if it inverts here too, FORCE=15/16 are the shipped
  adoption path.
* **PW/PFIN split at B=512** (`-DL13_PW=0` / `-DL13_PFIN=0`): still the
  outstanding ask, third round running, unchanged rationale.

### Next

1. Branch on the node's B=1: if X-first holds, the B=1 residue over floor
   shrinks and the next lever is whatever `ab` says about the twins; if it
   inverts, ship FORCE=8 back as the ws≤L2 tier and record the second
   machine-sign-inversion at this geometry.
2. If a twin wins on CLX, adopt it default and re-run the tail-kernel
   choice (the ymm tail's spilled-constant tradeoff may shift under TS's
   +6 ops).
3. B=512 stays hostage to the PW/PFIN split and MKL's own regressions;
   spend nothing there until the split lands.

---

## Round panel_r11 (2026-08-22)

r10 node results: all three cells won for the fourth round (B=1 5.692, B=16
5.983, B=512 8.006 µs/t), B=512 still the thinnest margin on the board
(1.12×). The r10 verdict closed my X-first driver bet as a null ("in-plan
discrimination measures the kernel and not the cell", third instance), closed
the association twins panel-wide (q +0.5–0.7%, s +3.7–3.8% on the node, and
§5's bound: the L=6 mechanism is a property of joins that feed *stores*), and
left one L=13 item, mine only by inheritance: the PW/PFIN split, now with the
monitor five rounds running. So this round went after my own oldest suspect —
the pb store→load junction (r8 "next" 3, r9 "next" 1), which the withdrawn
perf counters never got to confirm — **by deleting the mechanism instead of
measuring it**.

### The diagnosis (static, no counters needed)

The mixed-tail scheme (r6, from L17_rader r4) covers every 13-long index with
3 zmm chunks (lines 0–11) + 1 ymm chunk at offset 11 (lines 11–12, line 11
recomputed). That ymm tail sits at byte offset 176 ≡ 48 (mod 64) and is a
triple lemon:

1. **Split accesses.** A 32 B access at 48 (mod 64) always crosses a line: all
   169 Z-group tail loads from pb per volume split, the X tail's 13 stores
   split, ~52 Y-tail loads split. (16 B accesses on complex-aligned addresses
   can never split.)
2. **Store-forward blocks.** The Z group's tail load (ky=11,12 of a pb row)
   spans the Y group's zmm tile store AND its 16 B LASTCOL_ column store; a
   load spanning two stores cannot forward — ~169 blocked loads/volume, plus
   78 more where Z's zmm chunks load rows 11/12 written by the Y tail's 32 B
   TILE2_ stores. ~247 SF events/volume at the hottest junction in the plan;
   at ~13 cycles each that is the size of the whole ~2.9k-cycle B=1 residue.
3. **A recomputed line and 12 tail shuffles** (TILE2_) per chunk.

The fix has two halves, and the round's one genuinely new negative result is
that ONLY the pair works:

* **xmm (WC=1) tails** at offset 12/168: the template is instantiated a third
  time at one complex per vector. Port-identical (102 xmm ops pair on two
  ports = 51 cycles, exactly the ymm tail's cost, on both machines), zero
  split accesses, zero recompute, zero tail shuffles, and its 16 B stores are
  exactly containable by later 16 B loads. Bit-identical per line (same DAG,
  same per-lane constant values).
* **ZSOLID Y groups**: 4 zmm chunks at 0,4,8,9 — the overlap chunk rewrites
  pb rows 9–11 with identical bits and delivers row 12 as full 64 B tiles, so
  *every* Z-group load out of pb is exactly contained in one Y store. Zero
  SF blocks by construction. Costs +51 port cycles/plane on the node's single
  512-bit FMA unit (+663/volume, ~4.9% of floor); free on wallaby's two units.

### What shipped

Default (AVX-512, both tiers): **X-first, zsolid Y groups, xmm tails in the
Z groups and the X pass** (`l13_exec_xfzs_mx`, and `l13_exec_xfzs_pf_mx` past
L3 with the unchanged r8 prefetch schedule). Chunk census 133 zmm + 14 xmm =
14.3k port cycles (was 120+27 = 13.6k): the +663 cycles buy zero SF blocks
and zero tail splits. **Outputs bit-identical to the r10 binary** at
B=1/16/512 (cmp-verified, same driver inputs). The staged FORCE variants use
zsolid in BOTH group positions (the memcpy's 64 B reads of sb would straddle
an xmm-tail row — same disease). Selection logic, sysconf gates, prefetch
schedule, tables, AVX2 fallback: all unchanged.

Deleted: the association twins (chunk13q/s, four execs, ~150 lines) per the
r10 verdict; FORCE 13/14 are reused (13 = the r10 default verbatim, 14 = the
new default), FORCE 8 = X-last zsolid, FORCE 9 = the failed all-xmm form.
File 1518 → 1452 lines.

Discriminator re-targeted (instrument only, unchanged mechanics):
`ab[B*] = y2,zs,xt,xl` = r10 shape, shipped default, all-xmm-tails, X-last.

### Operation count

102 vector FP ops/chunk unchanged. Volume: Y 13×4 zmm, Z 13×(3 zmm + 1 xmm),
X 42 zmm + 1 xmm = 133×102 + 14×51 = **14,280 port cycles ≈ 4.94 µs at
2.89 GHz** (r10 floor 13,617 ≈ 4.71). Deleted per volume: ~234 split
loads/stores, ~247 SF-blocked loads, 26 recomputed lines' worth of nothing
(port-neutral), ~324 tail shuffles (port 5). Exec asm: xfzs 1098 insns / 128
stack refs vs y2's 1389/186 (the audit habit from L45_pfa r7).

### What was measured — wallaby (Gold 6448Y, pinned `taskset -c 49`,
interleaved variants, min over ≥6 (8 at B=1) process instances; MKL canary
3.567–3.583 vs 3.569–3.601 in r7–r10, windows comparable)

| case | r10 shape (re-measured, same windows) | **zsolid default** | Δ | all-xmm "xt" (FORCE=9) |
|---|---|---|---|---|
| B=1 | 2.575 µs (8/8 in 2.575–3.028) | **2.533 µs** (8/8 in 2.533–2.557, disjoint) | **−1.6%** | 3.70 µs (**+44%**) |
| B=16 | 41.50 µs/call (2.594/t) | **40.47 µs/call (2.530/t)** | **−2.5%** | 48.6 (**+17%**) |
| B=512 | 1606.8 µs/call (3.139/t) | **1578.4 µs/call (3.083/t)** | **−1.8%** | 2065 (**+29%**) |
| B=2048 (pf active, unscored) | 9258–10431 µs | 9409–9507 µs | wash (overlapping; zsd tighter) | — |

r10's own record said 2.577 at B=1; today's re-measure of that binary is
2.575 — cross-round reproducibility to 0.1%. Correctness: PASS rel_l2 =
2.826e-16 … 2.872e-16 at B = 1, 3, 8, 16, 512, 1024, 2048; bit-identical
across re-runs everywhere; all FORCE variants 8–14 PASS and repeatable; both
prefetch kill-switch builds PASS; AVX2 fallback PASS on wombat.

### What was tried and did NOT work (the round's most useful result)

1. **xmm tails in the Y position — a measured catastrophe on wallaby/SPR:
   +44% / +17% / +29% at B=1/16/512** (driver-level, pinned, interleaved,
   reproduced across 8 instances). Mechanism: the Y tail writes pb row 12 as
   13 16 B stores, and the Z group's three zmm chunks each load that row
   64 B at a time — a load spanning four narrow stores cannot forward and
   apparently costs far more than the two-store straddle it replaced. The
   asm is clean (1288 insns, fewer than y2), so this is memory-system, not
   codegen. **Lesson worth lending to any entry adopting narrow tails: a
   narrow-store tail is only safe where nothing soon wide-loads its output.
   Pair it with a solid-store group on the producer side of a junction.**
2. **The in-plan instrument inverted against the driver on the same machine,
   same pin, same batch**: `ab[B16]` on wallaby reads xt FASTEST (4918 vs
   zs 5008, y2 5142 ns/vol) while the driver-level race has xt +17%. The
   private 2-buffer arena evidently does not reproduce the driver's
   buffer/junction state. Fourth instance of in-plan ≠ cell, now in the
   *opposite direction* — monitor: read my `ab` fields as kernel-relative,
   never cell-predictive.
3. Still parked per prior records: cyclic-3/negacyclic-3 split (B=1 not
   port-bound — and zsolid just spent port slack on purpose), NT stores,
   software pipelining across pb (the class stays node-rejected; note
   zsolid is a *deletion* at that junction, not a reschedule).

### Expectations for the node (monitor, please)

* **B=1 is the round's bet.** Naive wallaby transfer says 5.60; the honest
  band is wide because the two sides of the trade are node-specific: the
  +663 port cycles are real there (+0.23 µs) and the ~247 deleted SF blocks
  are worth 0 (if CLX's OOO hid them) to ~1 µs (if serialized on the plane
  chain). Pre-registered: **≤5.55 = the SF deletion priced and the pb-junction
  theory is confirmed by deletion; 5.65–5.75 = wash (port cost ≈ SF cost);
  ≥5.85 = roll back via FORCE=13.** The `ab[B1]` y2-vs-zs field gives the
  same answer for free on every leaderboard line — subject to caveat 2 above.
* **B=16: expect 5.75–6.0** (wallaby −2.5%, same shape as B=1 with the
  X-first t1 junction warmer). **B=512: expect 7.85–8.1** (wallaby −1.8%,
  streaming wash at B=2048 says the port cost hides under DRAM).
* **PW/PFIN split at B=512** (`-DL13_PW=0` / `-DL13_PFIN=0`): unchanged ask,
  fifth round, the verdict itself named it "the last standing monitor ask
  that can actually be run."
* If zs loses to y2 on the node's `ab` while the cells hold, that is the
  narrow-tail lesson inverting on CLX — do not ship xt in response (it needs
  the Y-position pairing regardless); the correct rollback is FORCE=13.

### Next

1. If the node confirms zs at B=1, the pb junction is closed by deletion and
   the remaining residue is the in/out split accesses (driver layout,
   per-axis unfixable) plus the X→Y t1 junction: X writes t1 planes as 64 B
   tiles but Y's row loads at stride 26 within the plane still ¾-split.
   The one derived-but-unbuilt fix is padding t1 rows to 16 complex, which
   forces per-y-row X-pass chunking (13×(3 zmm+1 xmm) = +306 port cycles)
   for ~380 deleted split loads — build it only if the node shows B=1 within
   ~0.4 µs of the 4.94 floor, else the port slack isn't there.
2. B=512 stays hostage to the PW/PFIN split; nothing further spent there
   until it lands.
3. If a future round wants the last ~330 port-5 tail shuffles too: the Y
   zsolid group still pays TILE4_ shuffles for the overlap chunk's 3
   rewritten rows; a 13-row-tile store variant was sketched and priced at
   +3 vinsert per row-group — not worth code until the node says port 5
   binds, which nothing yet has.
