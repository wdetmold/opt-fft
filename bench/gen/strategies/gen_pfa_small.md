# gen_pfa_small — strategy record

Class: PFA of coprime pairs, small. Owned acceptance sizes: 10 (=2x5), 12 (=4x3),
15 (=3x5), 20 (=4x5). Graded cases: 10:64:1000, 12:64:600, 15:32:600, 20:32:256.

## Round gen_r1

### Where this entry started
The round-0 stub was the dense O(L^4) row-column floor. Everything below is new this
round; there is no earlier leaderboard.

### What was built
One C file, three cooperating structures:

1. **PFA (Good–Thomas) line codelets, no twiddles anywhere.** Each length-L transform
   is DFT_n1 (x) DFT_n2 over the coprime split, with the Ruritanian input map
   `(n2*j1 + n1*j2) mod L` and the CRT output map baked into the codelet's load/store
   index tables (verified against numpy: single-call rel L2 = 2.8–3.2e-16 at every
   size). Modules are exact-constant DFT2/3/4/5 in split-complex form, so every
   multiply-by-(±i) is a component swap folded into the following add, and every
   module constant is a broadcast FMA operand. Per-line op shape: 10 = 5xDFT2+2xDFT5,
   12 = 3xDFT4+4xDFT3, 15 = 5xDFT3+3xDFT5, 20 = 5xDFT4+4xDFT5.

2. **SoA-8 batch-lane engine for the graded path** (BORROWED: ice L45_pfa ice_r7's
   quad-volume "q4" arena, transitively rival v6_5a869e40's run4_ "SIMD lanes =
   volumes, no transposes, no tail lanes" — widened to 8 lanes/zmm for the two Ice
   Lake 512-bit FMA pipes; also the shape of the warm 3907 AoSoA seed). State is
   split re/im, point-major, 8 volumes in the lanes. All three axis passes are the
   SAME pencil codelet at compile-time strides (z: 8, y: 8L, x: 8L^2 doubles); every
   access is a full 64-byte vector; there are ZERO shuffle-class instructions inside
   the transform. Passes run in place (the codelet buffers a whole pencil in v8
   temps, so read-before-write is structural). Pack/unpack are 8x8 in-register
   transposes (24 shuffles per 4 points), once per execute or once per chain.

3. **fft3d_chain owns the whole graded m-step chain** (BORROWED: the ice campaign's
   universal endgame — the chain owns the inter-step state format, licence
   L17_winograd ice_r5). Per group of 8 volumes: pack x0 and c once, run all m steps
   inside the SoA arena, unpack once. The map is fused onto the z-pass stores of each
   step while the pencil is L1-hot (ice L45_pfa ice_r6: "map at the stores of the
   last axis"; a standalone staged map starves the ROB — their r5 lesson, not
   re-tested here). Groups are processed one at a time through all m steps, so the
   state stays L2-resident for the whole chain of that group.

4. **Fast map** — the single biggest win of the round. The graded map
   `s = 1/(1+sqrt(re^2+im^2))` written with vsqrtpd+vdivpd costs ~30 serial divider
   cycles per zmm and was ~half the whole chain step. Replaced with rsqrt14 + 2
   Newton steps (sqrt) and rcp14 + 2 Newton steps (reciprocal): ~1e-15 rel/point
   against the 1.5e-14/step contract, measured two-step gate 0.9–1.3e-15 (tol
   3e-14), chain ends 1.3–2.0x the honest anchor (tol 300x). `max(m, DBL_MIN)`
   guards rsqrt(0) making 0*inf = NaN at z = 0. Worth −24..−32% of the whole
   chain step at every size.

5. **4K-alias stagger**: the SoA component stride at L=12 and L=20 is an exact
   multiple of 4096, so re/im accesses would collide in the low 12 address bits at
   every point. +8 doubles (one line) of stagger between all arena components
   (ducc0's odd-stride rule via LITERATURE.md §08 §5.5). Cheap insurance; not
   A/B-isolated this round.

6. **B%8 remainders and B=1**: per-volume split-complex path reusing the same pencil
   codelets. Lanes are 8 consecutive inner points (x pass: flat (y,z) index — near
   zero waste; y pass: z-chunks with overlapped tails; z pass: y lanes via in-register
   8x8 transposes). Out-of-place ping-pong passes make overlapped tail chunks
   idempotent. Correct at every size, but SLOW (see "known gaps").

### Measured on the node (a80n0 Ice Lake, leased core 4, min over samples)

| case | this entry (chain us/xform) | MKL 2022 + driver map | ratio |
|---|---|---|---|
| L=10 B=64 m=1000 | **1.43–1.48** | 4.56 | **3.1–3.2x** |
| L=12 B=64 m=600  | **2.50–2.72** | 7.73 | **2.8–3.1x** |
| L=15 B=32 m=600  | **6.44–6.56** | 16.45 | **2.5x** |
| L=20 B=32 m=256  | **17.05–17.40** | 58.53 | **3.4x** |

(Ranges are run-to-run wobble with other implementers on neighbouring cores; sd
within a run is <0.1% except when a neighbour lands.) Before the fast map the same
structure measured 2.08 / 3.70 / 8.91 / 22.6 us — i.e. the FFT structure alone was
already ~2.1x MKL, the map cut another quarter to third.

Gates, all sizes: single call 2.8–3.2e-16; two-step 0.9–1.3e-15 (tol 3e-14);
chain-end 4.2e-14–1.8e-13 at 1.3–2.0x the honest anchor (tol 1e-10); bit-repeatable
across processes; B=12 mixed group+remainder and B=1 chains verified.

Plain execute (no map): L=10 B=64: 1.92 us vs MKL 1.51 (we LOSE — pack/unpack every
call is ~0.9 us of the 1.92); L=20 B=32: 25.9 vs 31.5 (we win). Not a graded number
this campaign (all our cases are map chains), but see next-round list.

B=1 plain: 3.75 / 5.30 / 15.06 / 30.94 us vs MKL 1.48 / 2.13 / 6.39 / 31.08 — we
lose B=1 badly at 10/12/15. Known, not scored in cases.txt, highest-priority gap.

### What did NOT work (with the number that killed it)
* **Software prefetch of the c field two pencils ahead in the fused-map z pass**
  (aimed at L=20, whose Q+CQ = 2 MB working set exceeds the 1.25 MB L2 so c streams
  from L3 every step): 19.5–20.1 us vs 17.05–17.5 control at L=20, i.e. **+14%**.
  The pass is issue-bound and the prefetch uops are pure tax — the same mechanism as
  ice L45_pfa ice_r8's "qc-pf next-plane pokes +20%" record, which I should have
  trusted without re-measuring. Reverted; do not rediscover.
* `--chain 1 --map` segfaults in the DRIVER for every implementation (pong buffer is
  only allocated when chain > 1, and both the fft3d_chain call and the fallback
  memcpy write it). Not our bug; the m=1 precision witness is covered by m=2.
* `tryout.sh` builds its `--cin` path from `$W` before `$W` is assigned, so its
  chain/map leg cannot work as shipped; I ran the identical build/run/check commands
  manually over ssh with a slot lease. (Also: `reserve.sh --status` needs `squeue`,
  which is not in PATH on wallaby, so tryout's reservation gate false-fails while
  the reservation heartbeat is in fact alive.)

### Operation count (per volume, per axis pass: L^2 pencils / 8 lanes)
Vector instructions per pencil (loads+stores+FP, all zmm): L=10 ~130, L=12 ~170,
L=15 ~210, L=20 ~330. No shuffles inside passes; pack/unpack ~6 shuffles/point/dir.
Map: 15 FMA-class ops per 8 points (no divider).

### What I would do next (ranked)
1. **B=1**: the split path loses 2.5x to MKL at 10/12/15. The waste is overlapped
   tail chunks (y pass at L=10 does 2 chunks where 1.25 are needed) plus ~3840
   port-5 shuffles/volume in the z-pass transpose sandwich. The known-good shape is
   ice L6_pfa's: interleaved complex, 2–4 complex per vector along adjacent-z lanes
   with a provably-minimal in-register lane turn for the z pass, plus half-width
   (ymm/xmm) tail codelets instead of overlap. Estimate 2.5–3x available.
2. **Fuse pack into the x pass and unpack into the z pass** for plain execute (saves
   one full round trip of the SoA arena per call; worth ~0.5 us/vol at L=10 B=64,
   would flip the plain-execute loss vs MKL). Chain already amortizes this.
3. **L=20 residency**: Q+CQ = 2 MB > L2. Prefetch failed (above). Candidates:
   interleave c INTO the state arena point-major (one stream instead of two), or
   accept L3 streaming. Measure `ld_blocks` off-node if PMU ever appears.
4. **Instruction diet in DFT5**: the e1/e2 pairs share `x0 + (a,b)` subexpressions;
   a 3-FMA lifting form (§08 §6.3) could cut ~10% of pencil FP at 15/20. Ice lesson:
   budget an hour, not a round (r2's 11.9% op cut bought 0.8% there).
5. **Round-3 generality**: the engine is already table-driven per (n1,n2); adding
   any coprime pair whose factors are in {2,3,4,5,7,8,9} is a table + one pencil
   function. Coordinate with gen_planner/gen_race on the create()-race when the
   driver starts asking for unlisted sizes.

### For other entries reading this
* The SoA-8 + PFA pencil structure here is generic: any line codelet that reads
  8-double vectors at a compile-time stride drops in. gen_powp/gen_pow2 could reuse
  the arena, pack/unpack, and fused-map chain scaffolding wholesale (twiddle stages
  slot in as extra broadcast-FMA operands between the two PFA-style stages).
* The fast map (rsqrt14/rcp14 + 2 Newton each) is worth −24..−32% of the whole
  graded step to ANYONE who owns their chain, and passes the two-step gate with 20x
  margin. Take it.
* Do not put prefetch uops in an issue-bound pass; two campaigns have now measured
  it at +14% and +20%.

## Round gen_r2

### Headline
Adopted gen_batchlane's engine structure wholesale (they beat me at 10/12/15 in r1
with the same PFA math), extended it to L=20 (which they do not cover), and added a
huge-page arena. Best chain numbers on a80n0 (leased core, min over runs):

| case | r1 shipped | gen_r2 now | vs batchlane r1 |
|---|---|---|---|
| L=10 B=64 m=1000 | 1.426 | **1.22–1.29** | 1.163 (still 5% ahead of me) |
| L=12 B=64 m=600  | 2.500 | **1.99**      | 1.995 (tied) |
| L=15 B=32 m=600  | 6.178 | **4.47–4.49** | 4.643 (now behind me) |
| L=20 B=32 m=256  | 16.92 | **13.39**     | (not covered by them) |

Gates: single call 2.6–3.2e-16; two-step m=2 gate 0.8–1.2e-15 (tol 3e-14); full
graded chains 1.2–1.8x the honest anchor (tol 300x/1e-10); B=1, B=9, B=12 mixed
group+remainder chains verified; bit-repeatable across processes. All measured by
hand-run check.py on the node (see harness notes).

### What changed (and what each piece was worth)
I A/B-ed my way to the rival design rather than transplanting blind; the deltas:

1. **Two-sweep step alone** (zy plane sweep + x pass instead of my three
   full-volume passes), on my old split-array engine: L=15 6.34 -> 6.15 (-3%),
   L=20 ~nil. NOT the main lever on its own.
2. **Map ladder swap** (my rcp14+2NR reciprocal -> batchlane's single vdivpd on
   the idle divider): L=15 6.15 -> 6.04, L=20 16.9 -> 16.6. Small alone.
   Diagnostic that mattered: **disabling the map entirely** measured the r1
   map-as-a-separate-reload-loop at 1.2 us/vol (L=15) and 5.4 us/vol (L=20) —
   20–32% of the whole step. The map's cost is placement, not arithmetic.
3. **In-register map fusion into x-pencil stores on the OLD split-array layout**:
   L=15 nil, L=20 +4% (c stream scattered into 20 extra L3 streams). The fusion
   only pays combined with the interleaved site layout below.
4. **The full engine rewrite** (all of: one interleaved site arena re[8]|im[8],
   128 B/site; padded plane stride PL=130/162/226/418 so plane bytes == 256 mod
   4096; in-place slot modules instead of tr[]/ti[] whole-pencil buffers;
   (C-S) == 2048 mod 4096; 34-instr Winograd DFT5 f +- (sqrt5/4)q; map8 fused
   in-register into the x-pass stage-2 stores): 1.22 / 2.25 / 4.61 / 14.06.
   This is where the round's gain lives. Note my r1 plane strides at L=12 and
   L=20 were == 2048 mod 4096 — the x-pass column loads stacked into TWO L1
   sets (L=20: 10+ lines vs 12 ways); the pad alone plausibly explains why
   L=20 improved 17% when 15 only matched batchlane.
5. **Huge-page arena** (2 MiB-aligned anon mmap + MADV_HUGEPAGE, node THP is
   madvise mode; faulted in by the create-time memset): L=12 2.25 -> 1.99,
   L=20 14.06 -> 13.39, and it removes physical page-coloring luck (see noise
   note). Plan-time cost ~zero; freed with munmap.

### The L=20 extension: in-place safety rule, written down
Stage-2 group c of a P-then-Q PFA reads slots {(Qc+Pb) mod L} and writes the CRT
slots; both sets are the full residue class {m == c mod P} iff **Q == 1 mod P**.
Then groups are mutually disjoint and every module is load-all-then-store-all:
fully in-place safe with no fusion. Holds for 10=2x5, 12 as 3-then-4, 20=4x5.
Fails for 15 (5 == 2 mod 3): groups c=1,2 read/write EQUAL slot sets and must be
one fused load-both-store-both codelet (batchlane's DFT5X2, taken verbatim, same
hazard). L=20 slot lists: stage 1 DFT4 on (0,5,10,15)+4b in place; stage 2 DFT5
reads (5c+4b), writes (5c+16d) — my r1 IN20/OUT20 tables, reused. This rule makes
round-3 generalization mechanical for any coprime P*Q.

### What did NOT work, with the number that killed it
* `optimize("schedule-insns","sched-pressure")` — batchlane's -13% at L=15 does
  NOT transfer to this engine: on my r1 engine L=15 +5.5%; on the new engine
  L=15 4.61 with vs 4.47–4.50 without, L=20 14.77–14.83 with vs 13.4–14.1
  without. Their gain is specific to their codelet's register structure.
  Default scheduler everywhere here. Do not re-litigate without a structural
  change to the codelets.
* In-register map fusion on a SPLIT-ARRAY (qr/qi) layout at L=20: +4% (above).
  Fusion requires the interleaved site layout to pay.
* Software prefetch: not re-tested (r1: +14%; ice: +20%). Standing rule.

### Borrowed, plainly
Nearly the whole SoA engine is **gen_batchlane gen_r1's** (transitively ice bl8,
rivals v5_cb7847fb / 8dc1a96d): interleaved site layout, plane-stride pad rule,
in-place slot algebra and the 10/12/15 slot lists verbatim, the DFT5X2 fusion and
its hazard analysis, the 34-instr Winograd DFT5, the map8 ladder and its in-x-pass
in-register placement, the (C-S) == 2048 mod 4096 offset. New and mine: the L=20
extension + the Q == 1 mod P safety rule, the huge-page arena (gen_layout's
territory — take it, it is 10 lines), the A/B numbers above showing which pieces
carry the win, and keeping the r1 split path for B%8/B=1.

### Operation count (per pencil per 8 volumes, FMA-contracted vector FP)
L=10: 5xDFT2(4) + 2xDFT5(34) = 88; L=12: 4xDFT3(12) + 3xDFT4(16) = 96;
L=15: 5xDFT3 + 3xDFT5 = 162; L=20: 5xDFT4 + 4xDFT5 = 216. Plus 4L ld + 4L st
per pencil (2 stages x L sites in-place). Map: 13 FMA-class + 1 vdivpd per site,
in-register at the x-pass stores. Two volume sweeps per step.

### Harness / measurement notes
* tryout.sh's $W bug is fixed, but its check.py chain leg still passes a
  LITERALLY-quoted `'$W/c.bin'` to the remote shell -> `--cin /c.bin` ->
  FileNotFoundError, and the `&&` chain then skips the repeatability check.
  The single-call gate and the timing are fine. I ran map-check and
  repeatability by hand over ssh with a slot lease (slot_lease.sh directly).
* wallaby still has no squeue; same PATH-shim workaround as r1 for
  reserve.sh --status.
* Run-to-run bimodality on the leased core (L=15: 4.47 vs 5.1–5.9, in-run sd
  0.05%): partially physical page coloring (huge pages fixed L=12/20),
  partially neighbor implementers' LLC/mesh traffic — other tryouts run
  concurrently. Trust the min; the scoring window is quiet.

### What I would do next (ranked)
1. **L=10 residual 5%** vs batchlane (1.22–1.29 vs 1.163): their remaining edge
   at the smallest size is likely loop/call overhead per tiny pencil; try
   unrolling two x-pass columns per iteration (their own "column pairing" idea)
   and a fully unrolled zy sweep at L=10.
2. **B=1 / B%8 split path**: untouched this round, still loses 2.5x to MKL at
   10/12/15 B=1. Unscored in cases.txt but round 6 may draw any batch. The
   lane-spatial plan from r1 stands.
3. **L=20 c-stream**: S+C = 2.3 MB > L2; c streams from L3 every step. Try
   interleaving c INTO the site (site = re|im|cre|cim, 256 B) so the x pass has
   one stream instead of two — measure, do not assume.
4. **Round-3 any-L**: the Q == 1 mod P rule + module set {2,3,4,5} covers
   any coprime pair with factors in {2,3,4,5} (14,21,30,33,35... need DFT7/11);
   coordinate with gen_planner on who serves what, and hand gen_layout the
   huge-page allocator.

## Round gen_r3

### Headline
The two obvious cross-entry borrows both measured as NON-TRANSFERS on this engine
(numbers below — the negative results are the round's cumulative contribution at the
tuned sizes), so the round's work went into the round-3 class duty: a GENERIC
runtime-table coprime-pair engine. supports() now also accepts 6, 14, 18, 21, 24,
28, 35, 36, 45, 56, 63 (any P*Q, gcd=1, modules in {2,3,4,5,7,8,9}), correct
through every gate, and 2.0x FASTER than MKL in the graded chain shape at L=14.
Tuned sizes are unchanged structurally and measure at r2 level or a hair better.

### Measured on the node (a80n0 leased core via tryout.sh, graded chains, min)

| case | r2 shipped | gen_r3 | MKL same window |
|---|---|---|---|
| L=10 B=64 m=1000 | 1.162 | **1.157–1.194** (slow-state runs 1.32; window bimodal all day) | 4.56–4.69 |
| L=12 B=64 m=600  | 1.931 | **1.970–1.975** | 7.93 |
| L=15 B=32 m=600  | 4.484 | **4.469–4.480** | 16.74 |
| L=20 B=32 m=256  | 13.39 | **13.24–13.53** | 58.1 |

(The r2-vs-r3 deltas at 10/12 are window noise, not code: the binary differs only
by the sched-pressure attribute (~0 at 10, −0.4% at 12) and dead-to-these-paths
generic code. The 1.16-vs-1.32 bimodality at L=10 persisted through the whole
session — batchlane r2's neighbor-turbo diagnosis; trust the min.)

Gates, all four tuned sizes, shipped build, run by hand on the node: single call
2.6–3.1e-16; two-step m=2 0.8–1.2e-15 (tol 3e-14); graded chains 4.6e-14–1.7e-13
at 1.2–1.8x the honest anchors (tol 300x/1e-10); bit-repeatable across processes.

### What did NOT transfer, with the numbers (same window, control second)

* **gen_batchlane gen_r2's rcp14+2NR map reciprocal** (their −8.1/−8.8/−4.7%):
  on THIS engine it LOSES everywhere — 10: 1.382 vs 1.325, 12: 2.042 vs 1.985,
  15: 4.615 vs 4.469, 20: 14.187 vs 13.842 (+2.4–4.3%). My x-pass keeps the FMA
  ports saturated (map8 sits between in-register stage-2 stores), so +5 FMA ops
  per site cost more than the single unpipelined vdivpd on the otherwise-idle
  divider. Ironic full circle: I taught them rcp14 in r1, adopted their vdivpd in
  r2, and their r2 swap back does not fit my codelet. LESSON, now measured twice
  in each direction: the div-vs-rcp choice is a property of the SURROUNDING
  CODELET's port pressure, not of the ladder — A/B it in place, never adopt on
  faith. `-DPS_RCPMAP` builds the ladder for the cross-arch reruns.
* **sched-pressure per-function attribute on the 10/12 families** (their −4.7/
  −6.4%): here ~0 at L=10 (fast-state runs 1.157 attr vs 1.162 without) and
  −0.4% at L=12 (1.970/1.973 vs 1.979/1.981, 2 pairs). Kept (free, tiny gain at
  12), `-DPS_NOSCHED1012` strips it. Their gain is specific to their codelet's
  spill structure, as my r2 already found at 15/20.
* **Consumption-order (column-major) c layout at L=20** (my own r2 next-step #3,
  the ice "tables in consumption order" doctrine): 14.39–15.13 vs 13.65–13.84
  control, +4–10%. WRONG ANALYSIS on my part: the natural plane-major c layout
  already streams perfectly — consecutive x-pass columns read ADJACENT 128 B
  blocks in each of the 20 planes, i.e. 20 concurrent sequential streams, which
  beats 1 sequential stream plus a scatter-pack. The doctrine applies to
  gathered tables, not to a layout that is already stream-linear per plane.
  Do not rediscover.

### NEW: the generic coprime-pair engine (round-3 class duty)

Structure (all in gen_pfa_small.c, ~250 lines):
* `gtabs_init` at create(): input map (Q*j1 + P*j2) mod L, CRT output map
  (A*k1 + B*j2) mod L with A = Q*inv(Q mod P, P), B = P*inv(P mod Q, Q) — the
  same algebra as the tuned slot lists (verified: reproduces the L=12 tables).
* `gpencil_body`: two-stage GT-PFA pencil on the SAME padded SoA-8 arena,
  BUFFERED through v8 temps — in-place safe for ANY pair, no Q ≡ 1 mod P
  constraint, map fused into stage-2 stores like the tuned STM path.
  always_inline, instantiated per (P,Q) with constant bounds (gcc unrolls all
  loops, resolves the module switch): worth 1.3–2.4x over runtime loops
  (14: 16.7→7.3, 24: 105.6→55.1, 63: 2611→2041 us B=8 execute).
* Modules: exact-constant 2/4/8 (8 = two DFT4s + W8 combine, sqrt(1/2) exact);
  ONE conjugate-pair-fold kernel for all odd n (3,5,7,9) with h*h cos/sin
  tables computed in LONG DOUBLE at create() (the brief's twiddle-exactness
  rule; h <= 4 so 16 doubles/table).
* Remainder volumes (B%8, B=1) lane-replicate the last volume (BORROWED:
  gen_batchlane gen_r1). Generic plane stride = L^2 padded to == 2 mod 32
  sites (the tuned pad rule, now a formula).
* Plan time ~0 (tables microseconds; same THP arena).

Correctness, all 11 generic sizes at B=8: single call 2.0–4.6e-16 vs numpy.
L=14: two-step 1.0e-15 (tol 3e-14), chain m=100 1.16e-14 vs anchor 1.24e-14,
bit-repeatable. L=21 B=3 (remainder path) chain m=50 1.08e-14 vs anchor
9.1e-15. B=1 and B=11 mixed groups verified at 14/21.

Speed (B=8 plain execute, us/xform, vs MKL same core): 6: 0.58; 14: 7.32 (MKL
4.90); 18: 22.1; 21: 40.3; 24: 55.1; 28: 96.3; 35: 206; 36: 252 (MKL 169);
45: 563; 56: 1252; 63: 2041. In the GRADED chain shape the picture flips:
L=14 B=8 m=100 chain 7.28 us/xform vs MKL 14.42 — 2.0x FASTER (chain owns
state residency + fused map; MKL pays a separate map pass per step). For
round 6: any coprime-pair draw in this set is covered at 2x-MKL chain speed,
vs Bluestein's 107–1315x existence fallback.

### Operation count
Tuned pencils unchanged (88/96/162/216 vector FP per pencil per 8 vols).
Generic pencil: stage-1 Q x DFT_P + stage-2 P x DFT_Q, buffered (2L v8 temp
round-trip per pencil); odd module n: ~4h + h*(4h+2) FMA-class vector ops
(h = n/2); map unchanged (13 FMA + 1 vdivpd per site).

### Harness / shared-state notes (READ THIS, monitor and gen_pfa_large)
* **Incident, resolved, flagged for gen_pfa_large**: at ~10:01 I ran `git stash`
  for an A/B, not realizing the round's `impl -> impl_3` SYMLINK is an
  uncommitted working-tree change — the stash reverted it to impl_2 for ~3
  minutes. During that window gen_pfa_large's agent saved an edit through the
  symlink and it landed on the FROZEN r2 file impl_2/gen_pfa_large.c (adds
  L=80 + DFT16 + wisdom, 65424 bytes, 10:02). I restored impl_2 to HEAD, the
  symlink to impl_3, and the monitor's logs/.rounds_state from the stash, and
  parked their orphaned edit VERBATIM at
  `impl_3/gen_pfa_large.c.RESCUED-see-pfa_small-r3-notes` — pfa_large: diff it
  against your working file; it may be newer than what you have (your impl_3
  file was last written 10:00). Standing rule for everyone: NEVER `git stash`
  in this repo; A/B with a file copy.
* tryout.sh works for chains now except the remote check.py leg (literal
  `'$W/c.bin'` → FileNotFoundError) and the repeatability cmp behind it; run
  both by hand (this round's gate numbers were). squeue PATH shim still needed
  on wallaby: `export PATH=/opt/software/slurm-19.05.8.1/bin:$PATH`.
* tryout.sh accepts any L (M defaults to 1 off-cases) — that is how the
  generic sizes were validated; MKL reference builds exist in
  build/a80n0/bin.

### What I would do next (ranked)
1. **Specialize the generic pencil harder for the round-6 favorites**: the
   buffered gpencil still loses ~1.5x to MKL at plain execute (chain wins
   2x). In-place slot codelets apply whenever Q ≡ 1 mod P (14=2x7 ✓, 18=2x9 ✓,
   21: 7≡1 mod 3 ✓, 28: 7 no... 7≡3 mod 4 ✗, 36: 9≡1 mod 4 ✓...) — the r2
   safety rule makes this mechanical; DFT5X2-style fusion covers the rest.
2. **B=1 lane-spatial engine** (third round on the list): still 2.5x behind
   MKL at 10/12/15 B=1. Coordinate with gen_batchlane — both records sketch
   the same ice L6_pfa shape; build it once, in whichever entry, and share.
3. **Planner handshake**: hand gen_planner the (P,Q) table + gpen_lookup so
   the trunk can route coprime composites here without probing supports().
4. **L=20 residual**: the only tuned size below 40 GF/s; C streams from L3 by
   capacity (S+C = 2.1 MB > L2). Prefetch failed (r1 +14%), c-interleave
   failed on split layout (r2 +4%), consumption-order failed (this round,
   +4–10%). The remaining idea is shrinking the footprint itself (half-plane
   c tiles processed twice per step = C L2-resident at 2x S re-reads) — an
   afternoon with a real chance of another dead end; budget accordingly.
