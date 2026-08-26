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

## Round gen_r4

### Headline
The round's marquee borrow — gen_batchlane gen_r3's register-explicit pencils —
was A/B-ed at all three candidate sizes and is a NON-TRANSFER at its most
important target: a wash at 10 and 12, and a +12.6% REGRESSION at 15 (numbers
below). 15 stays the r3 memory form. The positive work: the m-loop moved inside the
SCHED step function (batchlane's chainsteps shape, −1.3% at 20, the round's
one tuned-size win), and IPOK in-place pencils in the generic engine for
every coprime pair with Q ≡ 1 mod P (14, 18, 21, 36, 56), extending the r2
disjointness rule to the runtime-table path. Every gate passes.

### Measured on the node (a80n0 leased core via tryout.sh; fast-state minima,
### this session's windows were heavily bimodal — see method note)

| case | r3 shipped | gen_r4 | MKL same window |
|---|---|---|---|
| L=10 B=64 m=1000 | 1.156 | **1.155–1.157** | 4.67 |
| L=12 B=64 m=600  | 1.970 | **1.966–1.975** | 7.74–7.91 |
| L=15 B=32 m=600  | 4.469 | **4.444–4.463** | 16.73 |
| L=20 B=32 m=256  | 13.24 | **13.123** (13.30 before change 5) | 58.1–59.6 |

Gates, final shipped build, all run by hand on the node (tryout's map-check
leg still gets the unexpanded `'$W/c.bin'`): single call 2.6–3.1e-16; two-step
m=2 0.8–1.2e-15 (tol 3e-14); graded chains 4.6e-14–1.7e-13 at 1.2–1.8x the
honest anchors (tol 300x/1e-10); bit-repeatable at all four sizes; B=1 chains
at 10/20, mixed B=12 group+remainder at 12 all PASS. Generic engine: chains
PASS at 14 (B=8 m=100, B=1 m=50), 21 (B=3 m=50), 56 (B=3 m=8); single call
2.8–4.6e-16 at 14/18/21/36/56. B=1 chain timings (split path, structurally
unchanged since r1): 3.869 / 5.258 / 14.140 / 31.966 us at 10/12/15/20.

### What changed

1. **Register-explicit pencils at 10 and 12** (BORROWED: gen_batchlane gen_r3;
   stage 1 memory → named registers xr<k>/xi<k>, stage 2 registers → memory,
   exactly 2L ld + 2L st, map fused in the *_ipm stores). Measured a WASH:
   10: 1.157 vs r3 1.156 same window; 12: 1.966–1.974 vs 1.970. Explanation:
   their r3 gain was partly UNDOING their r2 out-lining (their pencils carried
   the optimize attr and compiled to calls); mine were always fully inlined,
   and gcc had already forwarded the stage-2 loads. Kept anyway — fewer
   stores, and the asm audit (their method, gen_pow2 r1's originally) now
   shows exactly the minimal 24 data stores + ~29 sched-pressure spills at 12.
2. **L=15 REVERTED to the r3 memory form after the A/B.** Register-explicit 15
   (identical in structure to batchlane's shipped 15) measured 5.018–5.045
   fast-state vs 4.444–4.463 for the memory form, same windows, +12.6%.
   Mechanism: 30 live site registers plus ~14 DFT5CORE temps per stage-2 group
   spill hard, while the memory form's stage-1 stores ride the 2-stores/cycle
   port at near-zero cost. Confirmed fast-state (a mixed-state run whose fast
   samples still read 5.02). NOTE for gen_batchlane: your r3 board number at
   15 was 4.771 vs your record's 4.456 — my memory-form 4.44–4.46 suggests
   your register-explicit 15 may be losing in scored conditions too; A/B the
   r2 memory form back at 15 before trusting the 4.456.
3. **Map tail re-raced on the new codelets** (the div-vs-rcp verdict is
   codelet-local — now measured on four codelet generations): vdivpd wins
   again everywhere. Paired runs at 12: div 1.969/1.971 vs rcp 2.005/2.010
   (+1.9%). A first rcp run read 2.281 — that was the slow node state, not
   the code; always pair control-first in the same window. Per-size knobs
   -DMT10/12/15/20=1 build the rcp ladder for the cross-arch race.
4. **sched-pressure re-raced**: keep on 10/12 (stripping costs +1.7% at 12:
   2.000 vs 1.966–1.974 fast-state); OFF at 15 memory form (unchanged r3
   verdict; on the REJECTED register-explicit 15 it helped ~1.5%, knob
   -DPS_SCHED15 kept for cross-arch); OFF at 20.
5. **m-loop moved INSIDE the SCHED step function** (soa_chain_L, batchlane's
   chainsteps shape; gcc hoists the map-ladder constants and base addresses
   across steps instead of reloading them per soa_step call). Bit-identical
   outputs. L=20: 13.123 (best of the session, vs 13.297–13.309 for the
   call-per-step form minutes earlier); 10/12/15 unchanged within noise
   (1.156 / 1.975 / 4.451). Cheap, kept everywhere.
6. **Generic engine IPOK** (round-6 insurance, my r3 next-step #1): when
   Q ≡ 1 mod P, the r2 disjointness rule holds for the runtime-table pencil
   too, so stage 1 writes back to its input slots and stage 2 reads them
   directly at inmap[j2*P + k1] — the 2L-vector tr/ti temp round trip is
   gone. Compile-time IPOK flag per GP_DEF instantiation, (Q % P == 1).
   Sizes 14, 18, 21, 36, 56. Same-window pairs: 36: 252.6 vs 257.4 buffered
   (−1.9%); 14: 8.125 vs 8.107 (wash, window was slow). Non-IPOK pairs keep
   the buffered form.

### What did NOT work, with the number that killed it
* **Register-explicit pencil at L=15**: +12.6% (5.02–5.05 vs 4.44–4.46, three
  runs each, fast-state confirmed). The technique's win is conditional on the
  live-register count fitting: 2L + module temps <= ~32. 10 and 12 fit (20/24
  live), 15 does not (30 live), 20 never could (40). WRITE THIS RULE DOWN
  before porting it anywhere else.
* **rcp14+2NR map tail at 10/12/15/20** (third time measured on this entry,
  now on the register-explicit form): +1.9% at 12, similar elsewhere. The
  x-pass FMA ports stay the binding resource in every form this engine has
  had; the divider stays free. Standing verdict for THIS engine only.
* **sched-pressure at 15/20**: unchanged losses; see 4.

### Method note (for everyone measuring on wallaby-leased cores this week)
The leased core spent most of this session flipping between two sustained
states ~13.5% apart (L=10: 1.157 vs 1.314; in-run sd 0.03–0.07% inside each
state, flips visible WITHIN an 8-sample run as median >> min). A single run
with small sd can be entirely in the slow state and look like a code
regression (my first rcp-tail run read +16%; the true paired delta was +1.9%).
Protocol that survived: control-first adjacent pairs, repeated until both
configs have shown their fast state; trust min-of-mins per config.

### Borrowed, plainly
* **gen_batchlane gen_r3**: the register-explicit pencil form (adopted at
  10/12, measured and rejected at 15) and the asm-audit-first method
  (transitively gen_pow2 r1).
* **gen_pow2 gen_r3**: the DSB/front-end checklist — audited my sweep and
  x-pass bodies (350–770 instrs, one pencil per iteration, no -funroll
  replication): all DSB-resident, nothing to fix here.
* My own r2 rule (Q ≡ 1 mod P) extended to the generic engine's IPOK.

### Operation count
Tuned pencils: unchanged FP (88/96/162/216 per pencil per 8 vols); 10/12 now
exactly 2L zmm loads + 2L zmm stores + spills (audited 53 stores at 12 incl.
sched-pressure spills), 15/20 unchanged 4L + 4L memory form. Map unchanged:
13 FMA + 1 vdivpd per site, fused in the x-pass stage-2 stores. Generic IPOK
pencil: drops 2L stores + 2L loads of temp traffic vs r3's buffered form.

### What I would do next (ranked)
1. **L=20 residency, still the only structural target** (S+C = 1.7 MiB > 1.25
   MiB L2; my 13.30 vs batchlane's 13.01 board). Prefetch (+14%), split-layout
   c-interleave (+4%), site-interleave (their +40%), consumption-order c
   (+4–10%) are ALL dead; the one untested idea is batchlane r3's bounce
   buffer (bulk-copy plane x+1's c during plane x's sweep — real work, not
   prefetch uops). My honest estimate: it loses (it adds a second write of C
   per step against an L2 that S already fills); measure only in a quiet
   window, budget two hours.
2. **B=1 lane-spatial engine** (fourth round on the list, still 2.5x behind
   MKL at 10/12/15 B=1). Round 6 draws unknown batches; coordinate with
   gen_batchlane — both r3 records sketch the same ice L6_pfa shape. Build it
   ONCE, in whichever entry moves first, and the other adopts.
3. **Non-IPOK generic pairs** (24, 28, 35, 45, 63): DFT5X2-style fused pairs
   would make them in-place too (the equal-slot-set groups come in pairs when
   Q mod P == P-1, e.g. 24=3x8, 45=5x9); mechanical but fiddly slot algebra.
4. **12 residual 2.8%** vs batchlane (1.966 vs 1.915): asm now near-identical
   (they carry MORE spills and rcp), and the chainsteps-owns-the-m-loop shape
   is adopted (change 5 -- it moved 20, not 12); their edge at 12 remains
   unexplained by structure. Suspect code alignment luck; a -falign-loops
   sweep via function attribute is the next cheap probe.

## Round gen_r5

### Headline
Adopted gen_batchlane gen_r4's same-core interleaved A/B protocol (one held
slot lease, all variants built side by side and ALTERNATED on one core with
--samples 4; gen_pfa_large r4 concurs) and used it to re-open every map-knob
verdict this entry had ever set through core-hopping tryout pairs. The result:
the 2.8% L=12 gap to gen_batchlane is CLOSED (1.917 vs their 1.915-1.919,
same core, same minutes), 15 and 20 each gained ~0.5%, and 10 ships
bit-identical to r4. The mechanism, found by diffing their source against
mine: my r2 adoption of the bl8 map ladder was a TRANSCRIPTION, not a copy --
one extra multiply per site in the rsqrt Newtons and set1-intrinsic constants
where theirs are static-const vectors -- and on the cheaper true-bl8 body the
div-vs-rcp verdict at 12 FLIPS.

### Measured on the node (a80n0 core 2, ONE slot lease held all session,
### interleaved minima; window was clean, in-run sd 0.01-0.25%)

| case | r4 ship (same-window control) | gen_r5 ship | gen_batchlane same-window | MKL same-window |
|---|---|---|---|---|
| L=10 B=64 m=1000 | 1.154-1.156 | **1.154-1.155** (bit-identical path) | 1.155-1.158 | 4.70 |
| L=12 B=64 m=600  | 1.973-1.979 | **1.916-1.920** (-2.8%) | 1.915-1.919 | 7.94 |
| L=15 B=32 m=600  | 4.438-4.454 | **4.418-4.425** (-0.5%) | 4.607-4.620 | 16.77 |
| L=20 B=32 m=256  | 13.699-13.739 | **13.614-13.681** (-0.6%) | 13.718-13.795 | 59.96 |

Gates, shipped build, all run on the node: single call 2.6-3.1e-16 at
10/12/15/20; two-step m=2 8.2e-16 / 9.2e-16 / 1.2e-15 / 1.2e-15 (tol 3e-14);
graded chains 1.615e-13 / 4.869e-14 / 5.208e-14 / 4.366e-14 vs anchors
1.081e-13 / 3.887e-14 / 4.784e-14 / 2.835e-14 (tol 300x/1e-10); repeatable
bit-identical at all four sizes; L=10 chain output bit-identical to the r4
ship (cmp). B=1 m=50 chains PASS at all four sizes (3.447 / 4.761 / 12.422 /
28.650 us); mixed B=12 group+remainder chain PASS. Generic engine: chains
PASS at 14 (B=8 m=100), 21 (B=3), 56 (B=3); singles at 36; L=14 graded-shape
chain now 6.17-6.27 us/xform (GMT race below).

### What changed

1. **map8 now carries TWO ladder bodies, selected per size like the div/rcp
   tail** (MT<L> encodes both: bit 0 = rcp tail, bit 1 = bl body; GMT for the
   generic engine). The bl body is gen_batchlane's map8 VERBATIM (bl8 r4
   lineage): hs = s/2 hoisted once and y *= (1.5 - hs*y^2) -- one multiply
   per site fewer than my r2-r4 form (0.5*r)*(3 - t*r), which recomputed the
   halving inside each Newton -- in vector-extension arithmetic with
   static-const vector constants instead of set1 intrinsics. Ship defaults:
   MT10=0 (legacy body + div, unchanged r4 arithmetic), MT12=3 (bl + rcp),
   MT15=2 (bl + div), MT20=2 (bl + div), GMT=2.
2. **The div-at-12 verdict flipped on the true-bl8 body**: with my old body,
   div beat rcp again this round (1.960 vs 1.933-1.956 was ALREADY rcp-
   favoring on the intermediate hs-form; on the verbatim body rcp 1.913-1.914
   vs div 1.953-1.957, five interleaved pairs, four clean). Ladder cost and
   tail choice are coupled: the r4 "div wins at 12" was measured on a ladder
   whose extra FMA-port ops were exactly what made rcp's +4 ops lose.
3. **Isolation of the r5 gain at 12**: hs-form alone -0.5% (1.960 vs 1.970);
   + verbatim body (static-const vectors, aligned deref) with rcp tail
   another -2.3% (1.913). sched-pressure stays ON at 10/12 (2x2 raced:
   div+nosched 1.945, rcp+nosched 2.015 -- rcp NEEDS the pressure scheduler
   here).
4. **L=15**: bl body + div -0.5% (4.418 vs 4.438-4.454). rcp tail still loses
   at 15 (+2.4%: 4.526-4.556) and PS_SCHED15 still loses (+10%: 4.893-5.003),
   both re-confirmed same-core on the new body.
5. **L=20**: bl body + div -0.6% (13.614-13.681 vs 13.699-13.739); rcp still
   loses (+2.6%: 13.986-14.001). Structurally untouched otherwise.

### Built, raced, and REJECTED: the L=15 hybrid sweep (kept as -DMEM15SW=1)

gen_batchlane's in-progress gen_r5 file ships BL_MEM15=2: register-explicit
pencils in the map-free zy sweep + MY memory-form fused-map x-pencil (they
adopted my r4 15 verdict for the x-pass). The combination my r4 A/B never
tried -- so I built it: dft15_ipr (X3L stage 1 to named registers, X5ST
stage 2 registers-to-memory, no DFT5X2 hazard) in the sweep only. Same-core,
five rounds, all three variants in every round: memory sweep 4.413-4.432 vs
hybrid 4.445-4.449 vs batchlane's shipped hybrid 4.602-4.620. REJECTED here:
my 30-site register sweep pencil spills even without map temps (~44 live v8
against 32 zmm), while the memory form's stage-1 stores ride the 2-store/
cycle port. NOTE for gen_batchlane: my memory-form 15 beats your hybrid by
4% same-core -- A/B the full memory form (my dft15_ip/ipm, D5X2SM hazard
pair included) before shipping BL_MEM15=2.

### What did NOT work / was declined, with the number or argument

* **rcp tail at 10 on any body** (1.162-1.164 vs div 1.159-1.160) and **bl
  body at 10** (+0.6%: 1.162-1.165 vs 1.154-1.156, three interleaved
  rounds): L=10's 20-live-register pencil leaves sched-pressure nothing to
  fix; 10 ships the r4 arithmetic bit-identical.
* **-falign-loops=32 / -falign-functions=64 at 12**: 1.974-1.975 vs base
  1.970-1.972 -- the r4 "alignment luck" hypothesis for batchlane's 12 edge
  is dead; the edge was the map ladder all along.
* **rcp+nosched at 12**: 2.013-2.018 (+5%). The rcp ladder's extra ops only
  pay under the pressure scheduler.
* **L=20 bounce buffer (r3/r4 next-step, batchlane r3's idea): DECLINED on
  an arithmetic argument this time, not a window**: the bounce copy cannot
  fix capacity -- BB(1.07 MiB) + S(1.07 MiB) still exceeds the 1.25 MiB L2,
  so bounced c lines must survive next to S exactly as un-bounced ones must,
  and the copy adds 2.14 MiB/step of extra L2 traffic. The only remaining
  L=20 lever is shrinking S+C below L2, which no layout in five rounds of
  records achieves. I consider the cell closed at ~13.6 on this node.
* **GMT race at 14** (bl vs legacy body in the generic engine): 6.174-6.220
  vs 6.212-6.270, 2 of 3 rounds favor bl, one flips -- a wash; bl kept for
  the op count.

### Borrowed, plainly

* **gen_batchlane gen_r4 / gen_pfa_large gen_r4**: the same-core interleaved
  A/B protocol. It re-decided TWO of my shipped defaults (MT12, and nearly
  MT15's sign) and killed the alignment hypothesis in one session. Every
  future default here gets set this way.
* **gen_batchlane (bl8 r4 lineage)**: the map8 body verbatim -- hs-form
  Newtons, static-const vector constants, aligned c deref. Full circle
  in four rounds: I taught them rcp14 (r1), took back vdivpd (r2), rejected
  their rcp swap on MY transcription (r3/r4), and now adopt their exact body
  + rcp at 12 because the transcription itself was the bug.
* **gen_batchlane gen_r5 (in progress)**: the BL_MEM15=2 hybrid idea --
  measured and rejected on my codelet (numbers above), knob kept.

### Operation count
Pencil FP unchanged (88/96/162/216 vector FP per pencil per 8 vols; 10/12
register-explicit 2L+2L, 15/20 memory 4L+4L). Map per site: 10: 13 FMA-class
+ 1 vdivpd (legacy, unchanged); 12: ~17 FMA-class + rcp14 seed, NO divider;
15/20: 12 FMA-class + 1 vdivpd (one mul/site less than r4). Two volume
sweeps per step, map fused in x-pass stage-2 stores, all unchanged.

### Method note
The first invocation of ANY binary in a session reads +10-15% (cold i-cache/
branch predictors survive taskset within it); every race here discards the
first sample of the first round. The window itself was the cleanest in three
rounds (no bimodality all session) -- the r4 protocol note stands regardless:
alternate within one lease and compare adjacent pairs.

### What I would do next (ranked)
1. **B=1 lane-spatial engine** (fifth round on the list; B=1 m=50 now
   3.45/4.76/12.4/28.7 us vs MKL ~1.5/2.1/6.4/31 -- still 2-2.3x behind at
   10/12/15). Round 6 draws unknown batches for the ASSEMBLED library;
   whoever moves first (me or batchlane) should build ice L6_pfa's
   interleaved-complex z-turn once and the other adopts. If nobody builds
   it, the planner should route B<8 coprime sizes to whichever engine's
   split path measures ahead (mine, currently).
2. **Generic-module widening for round 6** (modules 11/13 in gdftodd are a
   ~20-line h<=6 extension + gfactor entries: 22,26,33,39,44,52,55,65,77,
   88,91,99,104,117 become coverable at generic-engine speed instead of
   Bluestein's 107-1315x). Coordinate with gen_pfa_large (their r4 record
   queues the same modules) so only one of us serves each size class.
3. **Cross-arch (XARCH.md due after this round)**: race the MT<L>/GMT
   two-bit knobs and MEM15SW per host -- the r5 finding that ladder BODY
   and tail flip together means the CLX/SPR races must sweep all four MT
   values, not just the tail bit.
4. **L=20**: closed on this node (capacity argument above); revisit only if
   a layout layer materializes a sub-1.25 MiB S+C representation.

## Round gen_r6

### Headline
The surprise-round coverage widening. Tuned 10/12/15/20 are untouched and ship
BIT-IDENTICAL to r5 (verified by cmp on full graded chains, same-core paired
timings 1.156 / 1.916 / 4.42 / 12.99 vs r5's binary in the same minutes). All
the work went into the generic coprime-pair engine: the module set grew from
{2,3,4,5,7,8,9} to {2,3,4,5,7,8,9,11,13,15,16,17,19,21,23,25,27,29,31}, plus
NESTED twiddle-free GT-PFA for composite odd modules
(21,33,35,39,45,51,55,57,63). supports() now accepts 68 sizes: the 4 tuned +
64 generic = EVERY coprime P*Q in 14..127 with both factors in the module
set, except 50/80/100 which stay deliberately unclaimed (gen_pfa_large /
gen_powp scored cells). 53 sizes are new this round, including 72=8x9 which
was a plain omission in my r3 list (both modules existed for three rounds).
Every size passes every gate; the biggest wins are exactly where the brief
predicts round-6 draws hurt: composites with prime factors 17..31, where MKL
collapses (L=34 chain: 4.7x FASTER than MKL).

### What changed (all in the generic engine; ~250 new lines)

1. **Odd-module fold widened to h <= 15** (arrays 4 -> 15, tables long-double
   at create(), h*h <= 225 doubles/module): modules 11,13,15,17,19,21,23,25,
   27,29,31 all run through the ONE gdftodd conjugate-pair-fold kernel.
2. **Exact-constant DFT16** (two natural-order gdft8 halves + W16 combine,
   cos/sin(pi/8) literals): enables 48=3x16 and 112=7x16.
3. **Nested GT-PFA composite odd modules** (gmodpfa): module Q = q1*q2
   coprime runs stage-1 q2 x DFT_q1 buffered + stage-2 q1 x DFT_q2 IN PLACE
   on the buffer rows, with module-internal qin/qout maps built at create()
   by the same CRT algebra as the outer tables. Split set: 21,33,35,39,45,
   51,55,57,63. This is what makes 2 x {33..63} sizes (66,70,78,90,102,110,
   114,126) feasible at all -- their folds would need h up to 31.
4. **26 + 6 + 21 = 53 new (P,Q) instantiations + gfactor entries** (list in
   the file header). IPOK (Q==1 mod P, stage-2 reads stage-1's slots, no
   temp round trip) holds automatically at 18 of them via the existing
   compile-time (Qv % Pv == 1) test.
5. **Split per-volume buffers now allocated for tuned sizes only** -- the
   generic path never touches them; saves 6*L^3 doubles at create() (83 MB
   at L=120).
6. Encoding of gpen_lookup keys changed P*16+Q -> P*128+Q (Q now reaches 63).

### Raced same-core (held lease, alternated adjacent pairs, min-of-mins)

* **Nested-PFA vs flat fold, module 21**: nested WINS -12% at 42 (346-371 vs
  408-419) and -11% at 84 (5315-5361 vs 5912-6013). SHIPPED for 21 and all
  h>13 modules.
* **Nested-PFA vs flat fold, module 15**: nested LOSES +10% at 30 (109-117
  vs 100-111), +2-3% at 60/105/120 -- the smaller op cut (~450 -> ~280 incl.
  buffer moves) loses to the fold's straight-line FMA stream. Rewriting
  stage 2 to run in place on the buffer rows (saves 2n vector copies) did
  NOT flip it (114.6 vs fold 103.7). Module 15 ships as a FOLD;
  -DGM15PFA=1 rebuilds the split for the cross-arch races, -DGMODPFA=0
  strips all nesting (then 66..126 are declined -- fold tables only go to
  h=15).

### Measured on the node (a80n0 leased core, B=8 graded-shape chains, min;
### MKL 2022 same core, same window, adjacent runs)

| L | m | us/xform | MKL | ratio |
|---|---|---|---|---|
| 34 | 64 | **173.8** | 816.2 | **4.7x** |
| 26 | 100 | **60.4** | 189.0 | **3.1x** |
| 33 | 100 | **138.5** | 378.0 | **2.7x** |
| 62 | 32 | **2700.9** | 7335.7 | **2.7x** |
| 93 | 16 | **9730.3** | 25329.8 | **2.6x** |
| 46 | 50 | **944.6** | 2347.3 | **2.5x** |
| 22 | 200 | **40.5** | 92.9 | **2.3x** |
| 68 | 32 | **2715.4** | 6361.1 | **2.3x** |
| 124 | 8 | **25583** | 59555 | **2.3x** |
| 30 | 100 | **103.7** | 231.0 | **2.2x** |
| 102 | 16 | **11055** | 23123 | **2.1x** |
| 44 | 50 | **375.4** | 614.6 | 1.64x |
| 42 | 50 | **346-358** | 556-591 | 1.6x |
| 48 | 50 | **544.9** | 756.9 | 1.39x |
| 39 | 100 | **439.5** | 586.9 | 1.34x |
| 55 | 50 | **1241.8** | 1452.4 | 1.17x |
| 66/70/90/91/112/117/126 | | | | 1.02-1.10x |
| 65/77 | | | | 1.04-1.08x |
| 60/72/84/104/105/120 | | | | 0.95-0.97x (bandwidth-bound; S+C >> L2) |
| 78/88/99/110 | | | | 0.92-0.94x |
| 54 | 50 | 1691 | 1402 | 0.83x (module 27 fold) |
| 108 | 8 | 15697 | 11574 | 0.74x (module 27 fold) |

Gates: ALL 68 supported sizes pass single-call (2.1-5.0e-16, tol 1e-12) AND
the two-step m=2 gate (0.8-3.2e-15, tol 3e-14, >= 9x margin) on the final
build, B=8. Remainder path (B=1, B=3, B=5) verified at 22,34,48,62,66,70,90,
102,105,117,124,126 incl. m=2 chains; long chains vs honest anchor PASS at
22 (m=200), 34 (m=64), 54, 78, 102, 117; outputs bit-repeatable at every
size checked. Tuned sizes: bit-identical to r5 through full graded chains.
Setup stays trivial (0.001 s at 22; the L=124 arena is ~2 x 250 MB huge-page
memset, well under any budget).

### What did NOT work / bugs caught, with the number

* **gdft16 shipped briefly with cos/sin(pi/16) instead of pi/8**
  (0.98078/0.19509 instead of 0.92388/0.38268): single-call gate read
  1.7e-1 at 48 and 112, everything else clean -- even-k outputs exact, odd-k
  wrong, which localized it to the odd-half twiddles in one isolated-kernel
  test. The two-part gate catches a wrong CONSTANT instantly; trust it.
* **Nested-PFA module 15** (above): +10% at 30. The lesson generalizes the
  r5 ladder-body finding: an op-count cut only pays if it does not replace
  straight-line FMA streams with buffer traffic; below ~2x op ratio the
  fold's ILP wins.
* The first cross-window survey mislabeled 42/60/84 as regressions/washes --
  window drift (MKL itself moved 6% between surveys). Every shipped verdict
  above was re-taken as same-core adjacent pairs (gen_batchlane r4 protocol,
  standing).

### Borrowed, plainly
* **gen_batchlane gen_r4 / gen_pfa_large gen_r4**: the held-lease interleaved
  A/B protocol (standing method here since r5; decided the 15-vs-21 module
  verdicts this round).
* **gen_planner gen_r5's full-sweep coverage doctrine** (their L=2..128
  all-pass) is what convinced me coverage-at-speed, not existence, is the
  round-6 contest: the planner already guarantees existence everywhere, so
  a class entry only matters where it BEATS the planner's generic engine and
  the libraries. The prime-module cells (2-5x vs MKL) are that.
* The nested-PFA module, gdft16, and the prime-module widening are new here.

### Operation count
Tuned pencils unchanged (88/96/162/216 vector FP per pencil per 8 vols).
Generic: fold module n = ~8*h^2 FMA-class vector ops (h = n/2 <= 15);
gdft16 ~ 180; nested module q1*q2 = q2*cost(q1) + q1*cost(q2) + 2n buffer
stores + n gathers + n scatters (DFT21: ~850 -> ~530, DFT63: ~7700 -> ~1900).
Pencil = Q x mod_P + P x mod_Q + 2L (IPOK) or 4L (buffered) vector ld/st;
map unchanged (GMT=2: 12 FMA + 1 vdivpd per site, fused in x-pass stores).

### Notes for the panel (coordination, round 6)
* **gen_powp**: 54=2x27, 75=3x25, 108=4x27 are MY weakest cells (0.74-0.83x
  MKL) because modules 25/27 are O(h^2) folds here -- prime powers need your
  twiddled CT. If you claim them (the way you claimed 50=2x25), the trunk
  wins those cells; my entries remain as the race's fallback.
* **gen_pfa_large**: your r5 next-list queued modules 11/13 for 88/99/104/
  112/117 -- those are covered here now (0.94-1.03x MKL, bandwidth-bound).
  If your volume-major engine + pair-packed map beats that (it should at
  L >= 88 -- my SoA arena is 2 x 128 B/site streams), take the cells; the
  race decides. The remaining library holes in 14..127 are 96=3x32 (needs a
  DFT32 module -- ~40 lines on my gdft16 pattern, I stopped at 16), 98=2x49
  (yours/powp's, like 50), and 106/111/118/122/123 (prime factors 37..61:
  Rader-module territory, nobody's).
* **B=1 at generic sizes still lane-replicates (8x waste)** -- sixth round
  on the list. If round 6 draws B=1 at a generic size, the trunk should
  route to gen_planner's per-volume path or gen_pfa_large; my B=1 is
  correct but ~3-6x off my own B=8 rate. The ice L6_pfa lane-spatial shape
  remains unbuilt by anyone.

### What I would do next (ranked)
1. **DFT32 module** (2 x gdft16 + W32, cos/sin(pi/16) -- the constants I
   accidentally typed this round): closes 96, the last easy hole.
2. **Twiddled CT modules 25/27** (or just cede those sizes to gen_powp and
   delete my claims if their r6 covers them faster -- check the board).
3. **B=1 lane-spatial engine**, unchanged sixth-round entry.
4. **Cross-arch**: race GMODPFA/GM15PFA and the MT knobs per host; the
   module-15 fold-vs-nested margin (+10% ICL) is exactly the kind of
   port-pressure verdict CLX/SPR flip.

## Round gen_r7

### Headline
The round's mandate was spending the queued backlog: my class's assigned literature
item (stage-as-outer-product / the "dense GEMM wins at L<=16" crossover, lit 11
Tier 2) and the six-round-old B=1 gap. Both were spent on the same target -- the
split (B=1, B%8) path -- and the B=1 graded chains dropped 23-37%:

| case (B=1, graded m) | r6 ship | gen_r7 ship | MKL same core, same minutes | ratio |
|---|---|---|---|---|
| L=10 m=1000 | 3.877 | **2.572-2.582** | 4.34-4.93 | **1.7x** |
| L=12 m=600  | 5.305 | **3.357-3.375** | 7.32      | **2.2x** |
| L=15 m=600  | 14.031 | **10.761-10.791** | 15.84-16.68 | **1.5x** |
| L=20 m=256  | 32.330 | **21.690-22.133** | 55.05-55.24 | **2.5x** |

The split path had lost to MKL at B=1 plain execute since r1; in the graded chain
shape it now beats MKL 1.5-2.5x at every tuned size. ALL batched paths ship
BIT-IDENTICAL to r6 (cmp on full graded chains at 10 B=64 and 20 B=32 against a
fresh impl_6 build; same-session batched tryouts 1.155 / 1.917 / 4.436 / 12.953).
The dense-GEMM crossover claim was implemented honestly, measured at every size,
and is DEAD on AVX-512 (numbers below) -- being first to kill it in performant
code is this round's literature contribution.

### What changed (all in the split path; the batched engine is untouched)

1. **Fused-map ROTATION step** (step_<L>, SPLITZ<L>=1, the ship default).
   Three ping-pong passes: pass 1 = stride-L^2 dim, flat 8-lanes over the inner
   L^2 (4% overlap waste); pass 2 = stride-L dim, lanes = 8 contiguous stride-1
   positions (unchanged, see "closed" below); pass 3 = stride-1 dim with the
   volume stored ROTATED ([P0][P1][P2] -> [P2][P0][P1]) and the graded map fused
   into its stores. Two structural effects:
   - Because the rotated store writes site k of row R to k*L^2 + R, pass-3 lane
     blocks are 8 CONSECUTIVE rows R of the whole volume regardless of slab
     boundaries: ceil(L^2/8) blocks with one overlap tail instead of
     L*ceil(L/8) per-slab blocks (13 vs 20 at L=10, 18 vs 24 at 12, 29 vs 30 at
     15, 50 vs 60 at 20), and only the IN-transposes remain -- the r1-r6
     sandwich's back-transposes are gone. tr8 count per volume-step at L=10:
     160 -> 52 (-67% of the port-5 bill, the thing the r1 record flagged as
     "~3840 port-5 shuffles/volume").
   - The map rides pass-3's stores (map8c = map_span's exact ladder on
     pre-loaded c vectors), so the separate map_span pass -- a full state+c
     round trip per step, the r2 batched lesson never applied here -- is gone.
   Layout cycles with period 3 (natural -> [z][x][y] -> [y][z][x] -> natural);
   the chain builds c in all three rotations at chain start (scalar, once per
   chain) and un-rotates the final state once if m % 3 != 0. Overlapped blocks
   stay idempotent with the map fused because the map is a pure function of the
   recomputed pencil output and c.
2. **Isolation of the two effects, same-core interleaved** (one held lease,
   adjacent alternated runs, min-of-mins): per-slab half-turn (transpose-in
   only, inner-dim swap, fused map -- the intermediate form) beat the r6
   sandwich+map_span by 9-14% (10: 3.077 vs 3.385; 12: 4.121 vs 4.675; 15:
   10.763 vs 12.154; 20: 24.340 vs 28.295); the full rotation then beat the
   per-slab half-turn by another 10-26% where per-slab blocking wastes lanes
   (10: 2.592 vs 3.497; 12: 3.358 vs 4.126; 20: 21.766 vs 24.096) and is a wash
   at 15 (10.776 vs 10.698 -- 29 vs 30 blocks, nothing to save). Rotation ships
   everywhere (one maintained path).
3. **supports()/batched/generic engines untouched.** Plan struct grew the two
   rotated c copies (4 svol, tuned sizes only) and the dense DFT-matrix tables.

### The literature item: dense stage-matrix broadcast-FMA, measured and REJECTED

Implemented as dense3_slab (-DSPLITZ<L>=2, kept buildable): per stride-1 pencil,
broadcast each input scalar and FMA rows of the compiled DFT matrix (create()-time
long-double tables, zero-padded columns so masked-out lanes stay exactly 0 through
the map ladder), masked stores, map fused, ZERO shuffles. This is the
stage-as-outer-product form (lit 11 Tier 2) in its most favorable regime -- the
B=1 cross-lane pass, where the PFA alternative pays port-5 transposes and masked
lane waste. Same-core interleaved, min-of-mins, vs the half-turn (and vs the r6
sandwich):

| L | dense | half-turn | r6 sandwich | dense verdict |
|---|---|---|---|---|
| 10 | 4.165 | 3.077 | 3.385 | +35% / +23% |
| 12 | 6.585 | 4.121 | 4.675 | +60% / +41% |
| 15 | 15.370 | 10.763 | 12.154 | +43% / +26% |
| 20 | 39.727 | 24.340 | 28.295 | +63% / +40% |

The claim's arithmetic never closes: dense costs 4*L*ceil(L/8) FMAs/pencil (80 at
L=10) vs the PFA pencil's ~11 FMA + ~20 port-5 shuffles amortized -- moving work
off port 5 onto the FMA ports only pays until the FMA ports are the binding
resource, which they already are (the r3 rcp-ladder lesson, again). For the
BATCHED path I declined even building it: lanes are volumes there (no masking
waste, no shuffles anywhere), so dense = 4*L^2 vs 88 vector FP per pencil, a
4.5x op increase on an engine measured at 43-48 GF/s of the ~93 GF/s FMA peak --
the crossover cannot exist. Anyone tempted by DFT-by-GEMM on x86: these are the
numbers that kill it; the claim's home (Ascend/SME) has matrix units, we do not.

### What else did NOT work / was declined, with the number or argument
* **Two rotation passes per step** (pass 2's per-slab lane waste at 10/12 is 60%/
  33%; replacing pass 2 with a second rotation pass would make it ceil(L^2/8)
  blocks too): declined by arithmetic -- at L=10 it saves 7 pencils (~450 FMA-
  cycles) and adds 52 tr8 (~1250 port-5 cycles). The r6 module-15 lesson shape:
  do not replace straight-line FMA streams with shuffle traffic.
* **Half-width (ymm/xmm) tail pencils for pass 2** (the r1 next-step idea):
  declined -- instruction COUNT per pencil call is width-independent, so a v4/v2
  tail costs the same cycles as the overlapped v8 chunk it would replace. Pass-2
  waste at 10/12 is structural under 8-lane pencils; closed.
* The dense table above.

### Gates (ship build, all run on the node by hand; tryout's map-check leg still
### gets the unexpanded '$W/c.bin' -- unchanged harness bug)
Single call 2.6-3.2e-16 at 10/12/15/20 (B=1 and graded batched). Two-step m=2
gate at B=1: 1.222e-15 / 8.948e-16 / 1.145e-15 / 1.175e-15 (tol 3e-14, >= 25x
margin; m=2 ends in rotation p=2, so the gate exercises the un-rotate). Graded-m
B=1 chains: 2.233e-13 / 2.464e-14 / 4.575e-14 / 3.175e-14 vs anchors 1.779e-13 /
5.797e-14 / 2.405e-14 / 2.300e-14 (tol 300x/1e-10). m=3 chain (ends natural,
p=0) PASS at 10. Mixed-batch remainder-through-new-path: L=12 B=12 m=50 and
L=20 B=9 m=20 PASS. Repeatable bit-identical across processes. Batched graded
chains BIT-IDENTICAL to the r6 ship (cmp at 10 B=64 m=1000, 20 B=32 m=256).

### Borrowed, plainly
* My own r2 batched lesson (map placement, not arithmetic: fuse at the last
  axis's stores) -- five rounds late, applied to the split path.
* **gen_batchlane gen_r4 / gen_pfa_large gen_r4**: the held-lease same-core
  interleaved protocol, used for every verdict above (standing method).
* The rotation itself is the transpose-free ordering vein of lit 11 Tier 2
  (MDFFT column-order): cite it as partially validated here -- rotating the
  STORE side of the turn pass is what made the block count minimal.

### Operation count (split path, per volume per step)
Pass 1: ceil(L^2/8) pencils at stride L^2; pass 2: L*ceil(L/8) pencils at stride
L; pass 3: ceil(L^2/8) x [2*ceil(L/8) tr8 + 1 pencil + L fused map8c stores].
tr8 per step: 52/72/116/300 at 10/12/15/20 (r6: 160/192/240/720 + a full
map_span round trip). Map: ceil(L^2/8)*L v8 ladders (130 at L=10; minimum 125),
map_span's hs-form + one vdivpd, now in-register at the stores. Batched pencils
unchanged (88/96/162/216 vector FP per pencil per 8 vols).

### Notes for the panel
* **The rotation split step is generic and adoptable**: any entry whose split/
  remainder path still does sandwich-transposes + a separate map pass
  (gen_batchlane's B=1 gap now spans 8 sizes; the generic engines lane-replicate)
  can take DEF_STEP + interleave_rot + the three c copies wholesale. The pencil
  codelet plugs in unchanged -- it only ever sees (src, dst, stride).
* **B=1 at my GENERIC sizes still lane-replicates** (7th round on the list, now
  explicitly the remaining B=1 hole): the rotation step would work there too
  (gpencil is stride-agnostic), but needs an out-of-place gpencil variant; the
  round-6 surprise draw went to the trunk at B>=2, so I spent the round where
  the graded reply asks for numbers. Next round: port the rotation step to the
  generic engine if any evidence appears that small-B generic draws matter.
* Dense-GEMM-on-x86 is dead (table above); do not rediscover.

### What I would do next (ranked)
1. Port the rotation split step to the generic coprime engine (kills the 8x
   lane-replication waste at B<8 for 64 sizes; mechanical now that the tuned
   version exists).
2. Cross-arch: race SPLITZ<L> in {0,1} per host (the tr8-vs-FMA balance is
   exactly what CLX's port structure flips; dense=2 stays buildable but is
   30%+ behind on ICL -- only race it if a machine shows a port-5 famine).
3. The batched cells are saturated (r5/r6 verdicts stand); protect, don't chase.

## Round gen_r8

### Headline
Spent my r7 next-list #1, which is also what the brief's surprise-test addendum asks
this class for (the L=21/44-class small-batch gap): the r7 rotation fused-map split
chain is PORTED TO THE GENERIC COPRIME ENGINE. Remainder volumes (B%8, incl. B=1) at
the 64 generic sizes no longer lane-replicate into 8 zmm lanes; they run per-volume
through a three-pass rotation step with the graded map fused into the last pass's
rotated stores. B=1 graded-shape chains, same-core adjacent runs (min-of-mins, three
rounds, MKL in the same window):

| case (B=1) | r7 ship (lane-replicated) | gen_r8 ship | MKL | vs r7 | vs MKL |
|---|---|---|---|---|---|
| L=14 m=100 | 49.8 us | **11.10** | 12.65 | **4.5x** | **1.14x** |
| L=21 m=50  | 184.1   | **40.93** | 72.29 | **4.5x** | **1.77x** |
| L=34 m=64  | 1296.7  | **233.0** | 792.0 | **5.6x** | **3.40x** |
| L=44 m=20  | 2825.6  | **520.9** | 580.4 | **5.4x** | **1.11x** |

r7 LOST these cells 3-4x to MKL; every measured one now wins. ALL tuned paths --
batched engine and the r7 tuned split path -- ship BIT-IDENTICAL to r7 (cmp on full
graded chains at 10 B=64 m=1000, 20 B=32 m=256, 12 B=1 m=600, 15 B=9 m=600; official
tryout sanity 1.229/4.414/13.162 at 10/15/20 in a somewhat noisy window, single-call
gates 2.6-3.1e-16, MKL 4.67/16.55/59.62 same runs).

### What changed (all in the generic engine's B%8 path; ~150 net new lines)

1. **gstep_split**: DEF_STEP's three ping-pong passes with runtime L. Pass 1
   stride-L^2 (flat 8-lane blocks over the inner L^2, one overlap tail), pass 2
   stride-L per slab, pass 3 stride-1 via tr8 transpose-in, pencil on buffered
   lanes, and the graded map (map8c, the map_span ladder) fused into ROTATED
   contiguous stores (output k -> k*L^2 + R): ceil(L^2/8) row blocks, one overlap
   tail, no back-transposes -- the r7 minimal-block form, unchanged. Layout cycles
   period 3; the chain builds c in both extra rotations once per volume (scalar)
   and un-rotates once at chain end when m % 3 != 0 (interleave_rot was already
   runtime-L). cr==NULL skips the map (a plain FFT+rotation variant for a future
   execute() route; execute() itself still lane-replicates -- unscored, correct).
2. **gspencil_<P>_<Q>: out-of-place split-complex pencil instantiations** (same
   GT slot algebra, same gmod/gmodQ modules, always buffered -- out-of-place needs
   no IPOK rule). The (P,Q) instantiation list is now ONE X-macro (GP_LIST) that
   emits both pencil families and both lookup switches; the emitted gpencil set is
   unchanged in content and order (tuned+generic batched codegen identical, see
   bit-identity above). Build cost of doubling the instantiations: ~7s -> ~14s.
3. **GSPLIT_RMAX = 4 (raced)**: remainder groups r <= 4 go per-volume split;
   r >= 5 keep the one lane-replicated SoA group, whose cost is flat in r.
   Same-core race at L=21 m=50 (per-xform): r=4 split 29.86 vs replicated 31.19
   (-4%; also -3% at 34 r=4, wash at 14 r=4); r=6 split 31.2 vs 26.5 (+18%), and
   +21% at 34 r=6, +29% at 21 r=7. r=5 was state-flip muddy (split 30.7 fast-state
   vs replicated 39.2 slow-state reads); scaling the clean r=6 replicated number
   (26.5 * 14/13 = 28.5) puts replicated ahead at r=5 by ~8%, so 4 is the
   conservative cut. -DGSPLIT_RMAX=0 restores r7 behavior; re-race per host.
4. **Split per-volume buffers (10 svol doubles) now also allocated for generic
   plans** -- but ONLY when batch%8 is in 1..GSPLIT_RMAX and L >= 8, so the r6
   memory rule for big pure-batched draws still holds. L=6 (a slab row cannot
   feed 8 stride-1 lanes) always lane-replicates.

### Built, measured, and REPLACED: the site-buffer first cut
The first working version avoided new pencil instantiations by gathering each
pencil's L lanes into a site-format buffer (site = re[8]|im[8], st=16) and running
the EXISTING in-place gpen codelet on the buffer. It passed every gate and already
beat lane-replication 3.8-4.7x, but pays a 4L-vector gather/scatter round trip per
pass-1/2 pencil (~25-30% of pencil work at L=14/21). The dedicated out-of-place
pencils beat it 10-20% same-core (21: 42.5-47.5 vs 48.6-54.7; 14: 11.18 vs 12.44;
44: 537.7 vs 622.5; 34: 238-242 vs 300-305) and replaced it. Lesson (the r6
module-15 shape again, from the other side): buffer traffic that substitutes for
straight-line loads/stores is not free even when it enables code reuse; at ~25% of
pencil work it is worth 60 more instantiations and 7 s of compile.

### Gates (ship build, all run on the node by hand; tryout's map-check leg still
### gets the unexpanded '$W/c.bin' -- unchanged harness bug)
Single call 2.1-4.6e-16 at every size/batch exercised (6,14,18,21,26,34,42,44,48,
62,66,117 x B in {1,2,3,5,11}). Two-step m=2 gate at 21 B=1: 1.707e-15 (tol 3e-14,
17x margin; m=2 ends rotation p=2, exercising the un-rotate). Chains m=6..100
including m%3=0/1/2 all inside the honest anchor band (ratio 1.1-1.6x, tol 300x).
Mixed batches B=11 (r=3 split), B=5 (r=5 replicated), B=2/B=3 verified. All outputs
bit-repeatable across processes. Nested-PFA module path (66=2x33), prime-module
fold (34,62), IPOK (18), plain fold (117=9x13) all covered. Tuned sizes:
bit-identical to r7 (see headline).

### What did NOT work / notes
* The site-buffer reuse cut (above): replaced, numbers recorded.
* r=5..7 per-volume split: loses to the flat-cost replicated group (+18..+29%);
  threshold shipped at 4. This is the same crossover the tuned sizes have NEVER
  been given (their remainder path is per-volume for all r since r1) -- a tuned
  r=5..7 draw would likely also prefer a replicated SoA group, but graded batches
  are multiples of 8 and the tuned paths ship bit-identical; noted, not spent.
* The first-invocation-cold effect (+10-15%) and the two-state node windows both
  reappeared this session; every verdict above is adjacent-pair min-of-mins
  (standing protocol since r5).

### Borrowed, plainly
* My own r7 rotation step and c-rotation chain bookkeeping, ported verbatim
  (structure, overlap-tail idempotence argument, period-3 cycle, un-rotate).
* gen_batchlane gen_r4 / gen_pfa_large gen_r4: the held-lease same-core
  interleaved protocol (standing).
* The lit 11 Tier 2 transpose-free ordering vein, already cited in r7, now
  validated on 64 more sizes at runtime L.

### Operation count (generic split path, per volume per step)
Pass 1: ceil(L^2/8) pencils at stride L^2; pass 2: L*ceil(L/8) at stride L;
pass 3: ceil(L^2/8) x [2*ceil(L/8) tr8 + 1 pencil + L fused map8c stores].
Pencil = Q x mod_P + P x mod_Q module FMA + 2L vector loads + 2L stores (buffered
out-of-place; no gather/scatter round trip). Map: ceil(L^2/8)*L v8 ladders,
map_span's hs-form + one vdivpd, in-register at the stores. Lane waste is the r7
structural residue only: pass 2's per-slab blocks and the two overlap tails.

### Notes for the panel
* **The generic rotation split chain is adoptable**: any entry whose remainder
  path lane-replicates (both generic engines did until today) can take
  gstep_split + the GS_DEF pencil family + the chain block wholesale; the only
  per-entry piece is the out-of-place pencil. gen_pfa_large: your volume-major
  engine may not need it at your L >= 40 cells, but the trunk's small-batch
  draws at coprime sizes now route best through me at r <= 4.
* **gen_planner/gen_race**: at generic coprime sizes the small-batch verdict is
  now: B=1..4 (mod 8) -> my split chain; B=5..7 (mod 8) -> my replicated group.
  The wisdom cache can just race me twice if in doubt; create() cost is unchanged
  (~ms, the gspencil tables are the same gtabs).
* L=44's 1.11x vs MKL is the weakest of the measured wins (MKL's radix-11 at
  B=1 is decent); a Winograd/Rader DFT11 module would be the next lever there,
  but the fold-vs-structure op ratio (~130 -> ~90) is below the ~2x that r6
  showed is needed to beat a straight-line FMA stream -- do not spend a round
  on it without a port-model (llvm-mca) check first.

### What I would do next (ranked)
1. Route generic execute() remainders through gstep_split(cr=NULL) + one
   un-rotate (the plain-FFT variant already exists and is tested by nothing --
   wire it only with a measured win; the scalar un-rotate is ~vol stores, so at
   m=1 it may be a wash vs one replicated SoA fft).
2. Cross-arch: race GSPLIT_RMAX (and SPLITZ<L>, MT knobs) per host; the split/
   replicated crossover is a tr8-vs-FMA balance CLX will move.
3. Tuned batched cells: saturated (r5-r7 verdicts stand); protect, don't chase.

## Round gen_r9

### Headline
The context round: the one op-count cut my rival measured and I had not adopted.
gen_batchlane gen_r7's PHI-LIFTED DFT5 v-pair (lit 08 6.3, adapted by them:
sin(2pi/5) = phi * sin(pi/5) EXACTLY, so the scaled-reflection pair v1/v2 factors
through one shared u = sa - PHI5*sb) is now in BOTH of this entry's DFT5 forms:
D5CORE (tuned batched stage-2 at 10/15/20, incl. the D5X2 hazard pair and the
X5 register-explicit binds) and M_DFT5 (the split B=1/B%8 pencils at 10/15/20).
6 vector ops instead of 8 per DFT5; pencil FP per 8 vols 10: 88 -> 84, 15:
162 -> 156, 20: 216 -> 208 -- exactly the deltas their r7 shipped, which is
where their r8-board lead at my three DFT5 cells lived (10: 1.148 vs my 1.152;
15: 4.376 vs 4.416; 20: 12.770 vs 13.048). L=12 has no DFT5 and ships
bit-identical to r8 (cmp-verified, batched and B=1). Constants PHI5/KL5 exact
to the last bit (50-digit Decimal; KL5 - S51*PHI5 - S52 == 0 in double).

### NODE ACCESS CAVEAT (read before comparing boards)
The Ice Lake node was NOT reachable this session: both axxxl nodes were
allocated to another user's ~2-day jobs at 21:00 (our icehold 438851 was
CANCELLED mid-hold, the guard's re-queued 438854 sat PD "Resources" all
session; polled for 40+ min). Every timing below is therefore from WALLABY
(login host, Xeon Gold 6448Y = SAPPHIRE RAPIDS, taskset core 100, same-core
interleaved adjacent pairs, min over --samples 4, first invocation discarded)
-- a cross-arch signal, not a scored number -- plus the llvm-mca ICL model.
All correctness gates were run in full locally (wallaby has AVX-512; the
binary exercises the real paths). The monitor's next leaderboard is the ICL
measurement.

### What changed
1. **D5VPAIR / M5VPAIR**: the lifted pair
       u  = sa - PHI5*sb          (FMA)
       v2 = S52*u                 (mul)
       v1 = S51*u + KL5*sb        (FMA)   KL5 = 1.25/sin(pi/5)
   replacing 4 mul + 4 FMA with 2 FMA + 2 mul + 2 FMA per DFT5 (both
   components), zero latency change. NOT bit-transparent (same exact values,
   different rounding) -- all gates re-run, nothing recycled.
2. **Per-size knobs, the MT<L>/SPLITZ<L> pattern**: -DLIFT5=0 global,
   -DLIFT5_10/15/20=0 per size. Implemented as a per-function enum constant
   (D5LIFT_) so ONE macro body serves all sizes and the dead branch folds;
   the enum form was cmp-verified bit-identical to a #if-selected form and
   timing-identical (3.540 vs 3.541 at 15 batched, same core same minutes).
3. Nothing else. Batched engine structure, split rotation chain, generic
   engine (module 5 there is the table-driven fold, different arithmetic --
   untouched, cmp-verified bit-identical to r8 at 35/45 B=8 and 14 B=1).

### Measured on WALLABY (SPR; cross-arch signal only)
Same-core interleaved, r8-arithmetic control (bit-identical to the r8 ship,
proven below) vs lift, four rounds each, min us/xform:

| case | ctl (r8) | lift | delta |
|---|---|---|---|
| L=10 B=64 m=1000 | 0.914-0.919 | 0.920-0.921 | **+0.6% (lift LOSES on SPR)** |
| L=15 B=32 m=600  | 3.574-3.583 | 3.542-3.546 | **-0.9%** |
| L=20 B=32 m=256  | 8.740-8.755 | 8.634-8.665 | **-1.2%** |

B=1 raced too but the login node's neighbor noise (49 users; 2x state flips
within rounds) made every B=1 verdict unusable; not claimed.

### The ICL arbitration without the node (tools/TOOLS.md, this round's use)
uiCA is BROKEN in ext/tools (setup incomplete: no instrData/ -- flagged for
whoever owns tools). llvm-mca (LLVM 22, icelake-server) on the gcc
-march=icelake-server x-pass loop bodies extracted from soa_chain_<L>
(the loop containing the map vdivpd), 100 iterations:

| L | lift instrs/iter, cycles | nolift instrs/iter, cycles | model delta |
|---|---|---|---|
| 10 | 326, 27975 | 323, 28263 | **lift -1.0%** (despite 3 MORE instrs) |
| 15 | 532, 43957 | 527, 43963 | wash in the x-pass (15's DFT5 win lives in the sweep pencils) |
| 20 | 700, 58864 | 687, 59560 | **lift -1.2%** |

So the SPR loss at 10 is host-specific scheduling, NOT a property of the
lift: the ICL model agrees with gen_batchlane's measured ICL win at 10
(-0.8%, four clean same-core pairs, their r7 record) on the same codelet
shape. SHIP DEFAULT: lift ON at all three sizes; the SPR advisory race
should build -DLIFT5_10=0 (this is exactly the per-size-knob scenario).

### Gates (full graded shapes, run locally on wallaby AVX-512; node re-run
### belongs to the monitor's scoring pass)
Single call 2.9-3.5e-16 at 10/12/15/20 B=8 and B=1 (tol 1e-12). Two-step
m=2 B=1: 1.074e-15 / 8.948e-16 / 1.243e-15 / 1.332e-15 (tol 3e-14, >=22x
margin). Full graded chains: 10 B=64 m=1000 1.504e-13 (anchor 1.081e-13),
12 B=64 m=600 4.869e-14 (3.887e-14), 15 B=32 m=600 5.487e-14 (4.784e-14),
20 B=32 m=256 5.231e-14 (2.835e-14) -- 1.25-1.85x the honest anchor, same
tier as r8 (and 15/20 EQUAL gen_batchlane's r7 post-lift drift to the last
digit: converged engines drift identically). Mixed remainders B=12@12,
B=9@15, B=9@20 PASS; un-rotate parities m=3/4/5 at 10 B=1 PASS; generic
sanity 14 B=8 m=100, 21 B=1 m=50, 34 B=3 m=20 PASS; everything
bit-repeatable across processes.

Bit-identity matrix (cmp on chain outputs, local builds):
* -DLIFT5=0 vs the r8 ship: IDENTICAL at 10/15/20 batched + split -- the
  refactor is provably transparent, r8 remains one flag away.
* Default (lift) vs r8: L=12 IDENTICAL batched and B=1 (no DFT5 at 12);
  15/20/10 differ as expected.
* -DLIFT5_10=0: L=10 IDENTICAL to r8 (batched + B=1), L=15 IDENTICAL to
  the lifted build -- the per-size knob isolates correctly.
* Generic engine: IDENTICAL to r8 at 35/45/14 (module-5 fold untouched).

### Raced and NOT shipped
* **MEM15SW=1 (hybrid sweep) under the lift** -- gen_batchlane's r7 found
  the lift flips their 15 hybrid verdict (their BL_MEM15 1 -> 2, -0.25%);
  mine does NOT flip on SPR: mem 3.539-3.553 vs hybrid 3.536-3.556
  fast-state minima over five rounds -- a wash, outlier rounds both ways.
  Ship default stays MEM15SW=0; re-race on ICL/CLX with the knob.
* PMU avenue 1 (bank the picks) needs no work here: this entry has ZERO
  plan-time picks -- every internal choice (MT<L>, SPLITZ<L>, LIFT5_<L>,
  MEM15SW, GSPLIT_RMAX, GM15PFA) is a compile-time constant, create() is
  table-building only, and 5 consecutive create() cycles are trivially
  identical. Determinism was never at risk in this file; gen_race owns the
  cross-entry racing.

### Borrowed, plainly
* **gen_batchlane gen_r7**: the lifted DFT5 v-pair, taken whole -- the
  factoring, the constants' derivation, and the knowledge (their four
  same-core ICL pairs per size) that it WINS on the scored host. This
  round is the mirror image of r5: then their map-8 body, now their DFT5.
* **gen_batchlane gen_r4 / gen_pfa_large gen_r4**: held-core interleaved
  adjacent-pair protocol, applied on wallaby for want of the node.
* llvm-mca loop-extraction arbitration is the r8 tools mandate
  (tools/TOOLS.md discipline: model for RELATIVE choices, node for scores).

### Operation count
Pencil FP per 8 vols: 10: 84 (2xDFT5 lifted), 12: 96 (unchanged), 15: 156
(3xDFT5 lifted), 20: 208 (4xDFT5 lifted); loads/stores unchanged (10/12
register-explicit 2L+2L, 15/20 memory 4L+4L). Split pencils: same -4/-6/-8
vector FP at 10/15/20 via M_DFT5. Map, sweeps, rotation chain: unchanged.
Two new broadcast constants (PHI5, KL5).

### What I would do next (ranked)
1. **Confirm on the node** the two boards this round could not: (a) the
   lift's ICL deltas at 10/15/20 (expect ~-0.8/-0.7/-1.0% per batchlane's
   r7 and the mca model), (b) LIFT5_10 stays ON for ICL. If the next
   leaderboard shows 10 regressing instead, -DLIFT5_10=0 is the one-flag
   revert.
2. **Cross-arch races now have real per-size work**: SPR flips LIFT5_10
   (measured here, +0.6%); race LIFT5_<L> x MT<L> x MEM15SW per host.
3. **Port-1 co-issue** (PMU audit avenue 4) remains unspent in my cells:
   the concrete shape here would be running the B%8 remainder volume as a
   ymm 4-lane pair INTERLEAVED into the 8-lane group sweep instead of
   after it. Real restructure, needs the node's counters to validate
   (l1d + port_1 dispatch); do not attempt against a model only.
4. Batched cells otherwise remain saturated (r5-r8 verdicts stand);
   protect, don't chase.
