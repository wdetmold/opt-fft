# L8_fusedaxes — strategy record (ICE panel)

Full lineage (rounds panel_r1–r11 on CLX, mt_r1–r4 multicore) lives at
`bench/geom/strategies/L8_fusedaxes.md` and `bench/mt/strategies/L8_fusedaxes.md`.
This file starts at ice_r2 because the ice_r1 implementer agent was one of the
15 that died at launch (VERDICT §3) — ice_r1 scored the unmodified panel_r11
binary.

## Round ice_r2

### Where I stood

ice_r1: 0.556 µs/xform at the graded cell (L=8, B=64, chain m=2572, unitary),
second of three in a declared statistical tie (batchsimd 0.550, radix8 0.561),
1.13× ahead of MKL — the thinnest margin on the board.  The real regression
was **stability**: run spread 1.8% → 16.6% on unchanged code.  The three run
JSONs show the pick was identical all three times (`fused+pfs`); run 1 scored
0.649 while runs 2/3 scored 0.556/0.562 — a per-process placement lottery, not
a pick flip.  And in **all three** processes the plan-time arena ranked
`fusedAA2 ≤ fusedAA ≤ fused` (0.465/0.502/0.409 vs 0.473/0.500/0.415), but the
old 3% hysteresis toward the `fused` anchor discarded that ranking every time.

### What I changed (arithmetic untouched — output bit-identical to r11)

1. **Chain-shaped tuner** (`tune_chain`) for the graded band, ported from
   L17_matrixsimd's ice_r1 win — the one change in that round the VERDICT
   credits with a measurable result.  The old surrogate reused one (src,dst)
   pair with no scale pass: a 1 MiB working set, fully resident in this node's
   1.25 MiB L2, seeing exactly one placement.  The scored unit ping-pongs two
   dst buffers with the driver's unitary scale sweep between steps (~2 MiB+
   cycling through L2, both (src,dst) orderings, one deterministic alias
   relation per direction).  The new trial replays that unit exactly (step 0
   primes from `ti`, then ta↔tb with a vectorized ×1/√512 pass between; values
   cycle exactly since unitary FFT⁴ = id) and times only the kernel intervals.
   Calibration check on the node: in-plan `fused+pfs=0.563` vs driver-forced
   0.568 same window — the 22–34% optimism the VERDICT flagged (§4a) is gone.
2. **Clock-settle spin** (~150 ms dependent-FMA) before any tuning —
   L17_matrixsimd's fix for schedutil probing an unramped core.
3. **Anchor = fusedAA2** (v15 tiny band, v16 = +pfs L2 band), hysteresis 5%:
   the only shape whose alias behaviour cannot depend on where the driver's
   buffers landed, and it never lost an arena reading in ice_r1.  With ~1–2%
   observed margins between the four candidates, 5% hysteresis is a de-facto
   **pinned plan** (the VERDICT's §6 order) that still publishes the measured
   price table (`chain-arena{…}`) every run.  Candidate set now
   {fused, fused+pfs, fusedAA2, fusedAA2+pfs}: fusedAA (depth-1) dropped —
   AA2 dominated it in every r1 reading; NT and pfw stay out (VERDICT §4.8:
   NT catastrophic in the L3-resident chain; L13_rader measured pw at +7.4%
   in-chain on this node).
4. **PMC probe re-armed by default** (was off since panel_r10) and re-aimed at
   the VERDICT's §6 L=8 ask: count `ld_blocks_partial.address_alias` for the
   picked variant with out at (out−in) ≡ 0 mod 4096 (volume stride 8192 = 2
   pages ⇒ one residue governs every volume boundary; page-aligned buffers
   make 0 the realistic degenerate case) and at +32 lines, plus forced
   fusedAA2 at the degenerate placement.  **Under tryout it reports `pmc=na`:
   perf_event_open returns EACCES for the leader (ssh session, not the
   monitor's slurm context).**  The probe costs three ~2 ms measurements in
   create() and degrades cleanly; if the scoring window has the PMU access
   the brief claims, the three counter triples land in the description.
5. **Two-entry aa_setup cache**: the chain alternates (out,pong)/(pong,out)
   every call, so the single-pair cache recomputed ~30 scalar ops per call
   (≈1% at B=1 where a call is ~1.6k cycles).  Steady state is now a pointer
   swap.

### Operation count

Unchanged: per volume 1248 vector FP (Yavne 52-op codelet × 24 groups,
FMA-folded) + 896 shuffles + 256 loads + 256 stores, zero spills in every
shipping kernel (`vol_aa_s`, `vol_p_s`, `vol_aa` verified by objdump on the
icelake-server build).  ICX port floor: p0/p5 pool = (1248+896)/2 = 1072
cy/vol ≈ 0.370 µs at 2.90 GHz; quiet-window in-plan kernel price 0.437–0.457
µs = 1.18–1.24× floor.

### Measured (a80n0, tryout = the graded chain m=2572, unitary; node was
### CONTENDED for most of the session — other implementers' leases share L3 —
### so same-window A/B is the only meaningful comparison)

Quiet windows:

| config | B=64 min µs/xform | notes |
|---|---|---|
| tuned (pick=fusedAA2+pfs) | **0.553** | sd 1.45%; MKL same window 0.643 → **0.860×** |
| forced v16 fusedAA2+pfs | 0.545–0.562 | sd ≤0.12% when window quiet |
| forced v1 fused+pfs (old pick) | 0.568 best, **0.603/0.628 other processes**, one median 0.676 | the lottery, reproduced |
| forced v15 AA2 plain | 0.574 | pfs is worth ~2% in-chain |
| forced v0 fused plain | 0.645 | spread-t0 prefetch worth 12% in-chain — src is L2/L3-resident, not L1 |

In-plan chain arena, quiet window: `fused=0.452 fused+pfs=0.442 fusedAA2=0.452
fusedAA2+pfs=0.437` — AA2+pfs won both quiet-window tables; fused+pfs won one
contended table by 2.1% (0.563 vs 0.575), which is exactly the pick-flip the
5% hysteresis now absorbs.

B=1 (tuned, pick=fusedAA2): **0.554 µs/xform** min, sd 0.06%, vs MKL 0.546 —
at B=1 the driver's per-step scale pass and chain loop dominate both.
Correctness everywhere: rel_l2 = 2.267e-16 (B=64) / 2.269e-16 (B=1), chain
check rel_l2 = 1.390e-13 (tol 5.1e-11), repeatable bit-identical across runs,
AVX2-only build (-mno-avx512f) PASS at B=64 (1.733 µs, correctness net only).

Contended-window caveat, for whoever reads the next leaderboard: identical
final binaries measured 0.553 / 0.745 / 0.781 / 0.565 across ~an hour while
MKL held 0.632–0.660 — my 3 MiB chain working set is far more exposed to
co-tenant L3 traffic than MKL's.  Nothing in this table under ~5% separates
variants unless taken in the same window; the scored drained window is the
experiment.

### What did not work / was rejected, with the number that killed it

1. **PMC under tryout**: EACCES at the perf_event_open group leader — the
   reserved-node ssh session lacks the PMU access the brief describes for the
   monitor.  Kept armed (cost ~6 ms in create, `pmc=na` fallback) so the
   scored run can still answer the §4.5/§6 aliasing question.
2. **L8_PF_DIST=2** (prefetch two volumes ahead) at v16: 0.565 vs 0.556/0.545
   for dist 1 — same conclusion as CLX r2, now confirmed in the ICX chain
   regime.  Kept at 1.
3. **L17_rader's ymm-tile p5-relief transposes — analysed, not ported.**  They
   help when p5 is the lone bottleneck.  My kernel's p05 pool is balanced
   (1248 FP + 896 shuffle over two pipes, 1072 cy floor with p5 alone at
   896 < 1072), and on ICX a 256-bit shuffle on p1 steals the same slot the
   fused p0+p1 512-bit FMA pipe needs — halving width doubles op count for
   zero port relief at this instruction mix.  Recorded so nobody re-derives it.
4. **Boundary-alias scheduling fix — deferred with analysis.**  The in↔out
   volume-boundary relation (stride 8192 ≡ 0 mod 4096) cannot be removed by
   any k1/y permutation: every phase-A 16-line load window covers all 16
   mod-16 line classes, and each in-flight out-store iteration occupies
   exactly 2 of them, so the collision count is permutation-invariant.  The
   remaining lever is timing (load the ~2 colliding pencils of x=0 last, or a
   gated store-buffer drain at the boundary), worth building only if the
   scored PMC read shows `address_alias` firing at d0 and not d32.

### Borrowed, plainly

- **Chain-shaped tuner + clock-settle spin**: L17_matrixsimd (ice_r1), via the
  VERDICT's explicit port order.
- **pfw excluded in-chain**: L13_rader's measured +7.4% (ice_r1 record, §4d).
- **The pin-the-plan doctrine**: the ice_r1 VERDICT §6.  AA2 tables and the
  aa_setup machinery are my own lineage (panel_r7/r11).

### Next round

1. Read the scored description strings: if the PMU worked, `pick_d0` vs
   `pick_d32` vs `aa2_d0` settles §4.5 at L=8; if aa fires at d0 only, build
   the x=0 pencil-order rotation (8 precomputed DEINT orders keyed on the
   active (in−out) residue in aa_setup) — bounded, arithmetic-free change.
2. If the scored spread is now ≤3% and AA2+pfs held, the remaining gap to the
   1072-cycle floor is ~0.07 µs of load/store latency: try software-pipelining
   phase A of volume b+1 into phase B of volume b (the store buffer is idle
   during phase A and the load ports during late phase B).
3. If batchsimd still leads by <1%, note its FUSEDAA arena also beat its
   FUSED pick in all three r1 runs under a too-large hysteresis — the same
   trap I just removed; expect it to converge onto AA2-style pinning and plan
  for the tie-breaker to be the boundary-alias fix above.

## Round ice_r3

### Where I stood

ice_r2 scored: 0.544 µs/xform at the graded cell, a 0.12% statistical tie for
first with L8_batchsimd (0.544), radix8 0.565, MKL 0.628 → 1.15×, still the
thinnest margin on the board.  Run spread fixed (16.6% → 0.5%), prediction
error −1.6% (conservative).  The VERDICT's single L=8 order for this round:
the PMU is EACCES-blocked in both dev and scored contexts, so **answer the
§4.5 alias question by timing, not counters** — "if the time does not move,
the hypothesis is dead and L=8 can stop spending rounds on it."

### What I changed

**Built the boundary-deferral experiment ("fusedAA2b", variants 17/18) and
ran it to a clean null.**  This was my own r2 "next round" item 1 (the x=0
pencil-order rotation); L8_radix8's r2 record names the same channel as "the
one alias channel nobody handles."

Mechanism: at each volume boundary the next volume's phase-A in-loads issue
while the previous volume's last ~3 phase-B out-store iterations (48 of the
56 store-buffer entries) are still in flight; the volume stride is 8192 B =
2 pages so one residue d = (out−in)/64 mod 16 governs every boundary; my r7
proof says the collision *count* is permutation-invariant, but a load is only
falsely blocked while the matching store is still buffered, so loading the
colliding pencils of x=0 *last* (~30–40 cy later) should let the stores
drain.  Collision rule (from r7): pencil p collides with store iteration y
iff 2p+h′ ≡ 2y+h+d (mod 16); the in-flight iterations are the active
aa_perm2 row's [5],[6],[7], and c = d mod 8 by construction, so d alone keys
the order.  16 offline-solved load orders (table in the source), x=1..7 and
all arithmetic untouched — output bit-identical (rel_l2 2.267e-16 unchanged
to the digit, repeatable).

**Result — NULL, by the round's trustworthy instrument** (in-process
chain-arena, round-robin min-of-9; cross-invocation tryout numbers drifted
up to 7% on identical binaries this session, see below):

| window | fused+pfs | fusedAA2 | fusedAA2+pfs | fusedAA2b | fusedAA2b+pfs |
|---|---|---|---|---|---|
| quiet | 0.418 | 0.418 | **0.411** | 0.420 | 0.414 |
| contended | 0.491 | 0.492 | **0.481** | 0.495 | 0.485 |

The deferral gains nothing; the +0.7% on AA2b is the out-of-line dispatch
call (isolated with -DL8_AX0_NATURAL=1: natural-order body behind the same
call reads the same).  The quiet-window incumbent price 0.411 matches the r2
scored arena to the digit.  **Conclusion: the ~195-cycle residual over the
1072-cycle port floor is NOT volume-boundary 4K aliasing.**  Combined with
the r7 permutation-invariance proof, §4.5 at L=8 is now closed from both
ends by timing: the collisions cannot be scheduled away, and deferring them
buys nothing measurable, so their cost is under ~0.5% and L=8 should stop
spending rounds on aliasing.  Anchor and pick stay fusedAA2+pfs; v17/18 stay
raced in the graded band for ONE scored round so the drained-window arena
publishes this A/B with authority, then should be dropped from the sets.

**Candidate-set surgery** (less pick-flip exposure): bare `fused` (v0) leaves
the graded-band race — it lost the r2 in-chain A/B by 12% (0.645 vs 0.568;
chain src is L2/L3-resident, spread-t0 is not optional there).  Depth-1
fusedAA (v10) leaves the tiny race — dominated by AA2 in every arena reading
since panel_r11.  Graded band now {fused+pfs, AA2, AA2+pfs, AA2b, AA2b+pfs},
tiny {fused, AA2, AA2b}.

### Three codegen traps, measured on the node (gcc 11.4, -march=native) —
### recorded so nobody pays for them again

1. **Function-pointer dispatch of a 110-instruction body costs 1–3%** on the
   graded chain (forced v17-natural 0.565–0.574 vs inline v15 0.553, same
   session).
2. **A 16-arm always-inline switch spills catastrophically**: merging sixteen
   r[8]/q[8] bodies into one frame defeated the register allocator —
   1.149–1.165 µs/xform, 2× the incumbent.  This also stands as evidence
   against the cross-volume software-pipelining idea (my r2 item 2, radix8's
   item 2): gcc 11.4 cannot hold ~40 live zmm without spilling.
3. **gcc's memcpy idiom recognition eats contiguous ST loops in small
   standalone functions**: the `for (y) ST(...)` store loop became two
   memcpy PLT calls with r/q forced onto the stack — arena 0.679 vs 0.482
   (+40% kernel price) with dft8s also left out of line.  Cure: dft8s is now
   `always_inline` (a no-op everywhere it was already inlined) and the AX0
   store sequence is hand-unrolled.  Worth checking in ANY new small kernel
   function: objdump for `memcpy@plt` and rsp traffic, not just spills.

### Also established this session

- **Cross-invocation tryout numbers are worthless below ~7% even with sd
  0.1% within each run**: forced v16 read 0.571/0.574 in one pair of windows
  and 0.611/0.615 in another, identical binary, MKL steady (0.63–0.66).  The
  in-plan arena (one process, round-robin, min-of-9) is the only dev
  instrument that resolved 1% reliably.  This sharpens the VERDICT §4 rule.
- Default build, quiet windows: **B=64 min 0.543 µs/xform (sd 0.06%), MKL
  0.649 same window → 0.837×**; B=1 0.554 (sd 0.03%) vs MKL 0.545; B=2048
  streaming 1.249 vs MKL 1.770 (untouched paths); AVX2-only build PASS
  (1.804, correctness net).  Correctness everywhere: rel_l2 2.267e-16,
  chain 1.390e-13 (tol 5.1e-11), repeatable bit-identical.

### Operation count

Shipping variants unchanged: 1248 vector FP + 896 shuffles + 256/256
loads/stores per volume, zero spills (re-verified by objdump after the
always_inline change; ax0_d* bodies are 110 instructions, 0 rsp refs).
AA2b adds one predicted indirect call per volume, no new vector work.

### Borrowed, plainly

- The experiment's mandate (timing-not-counters) is the ice_r2 VERDICT §6
  L=8 order.  The channel definition matches L8_radix8's r2 "what I would do
  next" item 1 — executed here so radix8 need not.
- Nothing else transferred this round: L17's schedule-pragma win predicts a
  LOSS on my source shape (fully unrolled straight-line with independent
  chains — L23_matrixsimd measured +1.7%, L13_direct +5.2% for exactly that
  shape), so it was not attempted, per §4.6's own rule.

### What I would do next

1. Read the scored chain-arena strings: if AA2b confirms +0.5–1% behind AA2
   in the drained window, drop v17/18 from the candidate sets (keep compiled/
   forceable) — the aliasing chapter at L=8 is then fully closed.
2. The remaining ~18% over the port floor is now most plausibly front-end +
   dependency latency, not memory hazards.  The one untried structural lever
   compatible with gcc's register limits: split each phase-B iteration's
   two dft8s calls across ADJACENT y-iterations (a 2-stage software pipeline
   at codelet granularity, ~24 live zmm, not ~40) to cover the trans8 →
   dft8s → untrans latency chain.  Cheap to prototype; the arena will price
   it in one run.
3. If the harness ever unblocks perf_event_open, the dormant PMC probe
   (-DL8_PMC=1 already re-armed) prices DSB vs MITE delivery in one run —
   the front-end hypothesis above is exactly what idq.dsb_uops/mite_uops
   settles.

## Round ice_r4 (2026-08-23)

### The task changed: the graded step is the rivals' full map step

`state ← (z + c) / (1 + |z + c|)`, z = **raw** FFT(state) — verified in
driver.c/check.py: no unitary scale in map mode, c is a full per-batch field,
the chain gate is rel L2 < max(1e-12, 1e-13·m) = **2.57e-10 at L=8's m=2572**
against a numpy reference chain.  Without an exported `fft3d_chain` you are
timed through exec + a driver-side map; MKL through that fallback reads
2.10–2.16 µs/xform at the graded B=64.  This round built the chain entry
point; `fft3d_execute` and every FFT variant are untouched (the single-
transform gate still runs through them).

### What shipped

1. **`fft3d_chain`, EAGER in-register map.**  Phase B holds the transformed
   volume in split re/im registers right after the z-axis dft8s — point
   (kx=LANEX[l], ky=y, kz=j) — so the map runs THERE, before any interleave,
   and the stored volume is already the next state.  Each point is touched
   exactly twice per step (phase-A load, phase-B store), same as FFT-only.
   This is deliberately NOT the rivals' lazy map: L17_matrixsimd measured
   lazy losing (16.48 vs 13.26) because it parks the map's dependency chain
   in front of the next FFT; at the tail of phase B it hides behind stores.
   My own lazy arms (lz/lz+pfs, map after DEINT in the next step's phase A,
   raced on the theory that phase-A planes ~200 uops overlap 2-3 deep in the
   352-entry ROB where phase-B groups ~390 uops do not) **confirmed L17: lz
   +2.4–13% behind eager in every window** (e.g. 0.781/0.771 vs 0.753 same
   window).  The mechanism transfers even at plane granularity.
2. **Map form — one shared Newton ladder + ONE vdivpd per 8 points** (adopted
   from L17_matrixsimd's ice_r4 "s6" winner + corpus §10 §2 / 1000f989's
   mapF; my split layout gives the shared-denominator form natively, no
   deinterleave/expand shuffles at all).  Per 8 points: s = fma(wi,wi,
   fma(wr,wr,1e-300)) (the guard bias folded into the addend, L17's trick;
   rsqrt(0)=inf would NaN), vrsqrt14pd seed (~2.3 cyc on this node, the
   corpus's "microcoded" claim is a VM artifact), TWO quadratic Newtons,
   d = fma(s,y,1) = 1+|w|, one exact vdivpd, two mul-outs.  15 FMA-pipe ops
   + 1 divider op per pair.  **Precision arithmetic (brief's order): seed
   2^-14 → 2^-27.4 → 2^-54.2 + ~2 ulp iteration rounding on |w|; divide
   correctly rounded; ~3-4 ulp per application — the exact tier, mandatory
   at m=2572.  Measured whole-chain drift 1.660e-11 at B=64 (15× margin,
   at the FFT-reassociation noise floor ~3e-11), 7.134e-13 at B=1.**  The
   rivals' float-seed tier is never worth it here: the 14-bit seed costs
   the same and lands exact.
3. **In-place state in `final_out`** (adopted from L17_matrixsimd ice_r4):
   legal because phase A drains each volume into the L1 scratch before
   phase B's first store.  Working set = state 512K + csplit 512K ≈ 1 MiB,
   resident in the 1.25 MiB L2 at the graded B=64.  c is pre-split ONCE
   per (c, batch, placement) into the phase-B register layout
   (csp[b][y][j][ri][lane], lane l = kx = LANEX[l]) and cached in the plan.
4. **"slot" split-complex intermediate between MY OWN steps — the shipped
   pick** (direction from L64_blocked's ice_r4 item 3, executed here as a
   full bit-exchange derivation against trans8's structural map
   (reg a, lane l) → (reg l, lane [a1,a2,a0])).  Steps 1..m-1 store split
   vectors in a slot layout (regs (z2,x1,x2), lanes (z0,z1,x0), re/im at
   +0/+8); the next phase A loads them with ZERO shuffles (DEINT gone);
   phase B pays PRE(16) + trans8(48) + POST(32) = 96/group — the same 96
   the interleaved path pays — for a net **−128 shuffles/volume**.  The
   bitrev feeds keep the map/c layout identical, so slot and div arms are
   bit-identical.  Step 1 keeps the DEINT head (x0 is interleaved), step m
   keeps the untrans_interleave tail (final_out must be interleaved).
5. **Chain tuner in create()** (my tune_chain lineage, retargeted at the real
   map semantics): 10 arms {div, fma, div-pp, slot, lz} × {plain, +pfs},
   round-robin min-of-7, 2% hysteresis, anchor slot+pfs; plus a second race
   over the (csp − state) mod-4096 line residue (8 residues, min-of-5),
   realised in fft3d_chain by rebasing csp inside a 4 KiB slack.  Arena and
   rr published in the description.  All div/slot/pp/lz arms are
   bit-identical (same FP ops on the same values); only an fma pick would
   change bits, and it loses by 6-7% — far outside the hysteresis.

### Operation count (per volume-step, slot+pfs steady state)

1248 vector FP (FFT, unchanged) + 960 FP (map) + 64 vdivpd + **768 shuffles**
(was 896 interleaved) + 384 loads (256 state/scratch + 128 csplit) + 256
stores; zero spills (objdump: vol_sp_mid_s 553 instr, 1 rsp ref = frame;
vol_cm_d_s 561/1).  p0/p5 pool floor (1248+960+768)/2 = 1488 cyc ≈ 0.513 µs
at 2.9 GHz; measured 0.742-0.755 = 1.45× floor — the same ratio as the
FFT-only kernel's 1.42×, i.e. the map added its pool cost and nothing more.
Divider: 64×~17 ≈ 1100 cyc/vol fully hidden (proof: the fma arm removes ALL
divider work, adds 256 pool uops, and is 6-7% SLOWER).

### Measured on the node (tryout leases; same-window pairs only)

| cell | this round | MKL fallback same window | note |
|---|---|---|---|
| graded B=64 m=2572 | **0.742 / 0.749 / 0.750 / 0.755 / 0.759 min µs/xform** (5 processes, sd ≤0.04%) | 2.10–2.16 | **2.8×**; pick slot+pfs 3/3 |
| B=1 m=2572 | **0.801 / 0.809** (sd ≤0.05%) | 2.28 | pick slot (2048-step trials) |
| ice_r3 FFT-only B=64 | 0.544 | — | fused map costs +0.21 µs; the driver-side unfused map would cost ~1.6 |

Correctness: single rel_l2 2.267e-16 (B=64) / 2.269e-16 (B=1) bit-identical
to r11; **map-chain 1.660e-11 (B=64) / 7.134e-13 (B=1) vs tol 2.57e-10**;
.chain bit-identical across independent processes at both batches; AVX2-only
build (-mno-avx512f, exact sqrt+div map path) PASS at 3.917 µs, chain
6.754e-12.  Rivals' L=8 mark: 0.115 s ≈ 0.70 µs/vol-step; we are at ~0.75
with a 15×-margin exact map where their shipped code drifts 100× past our
gate.

### What did NOT work / negatives, with the number that killed it

1. **The rivals' lazy map (lz/lz+pfs): +2.4-13% behind eager** in every
   window (0.781-0.902 vs 0.753-0.770).  L17's latency mechanism confirmed
   on a second structure; nobody needs to try lazy at L=8 again.
2. **All-FMA map (rcp14 + 2 Newtons, zero divider): +6-7%** at B=64
   (0.814-0.831 vs 0.762-0.777) — with the divides interleaved into phase B
   at 8/group the divider is effectively free, matching L13_rader/L17 and
   opposite to L64's bunched-divide case.  Do not "save" the divider here.
3. **Ping-pong state buffers: +8-16% at B=64** (working set 1.5 MiB blows
   the 1.25 MiB L2), **null at B=1** (0.852 vs 0.848 — the in-place
   store-buffer boundary hazard is not measurable, consistent with r3's
   closed aliasing chapter).  In-place is strictly right.
4. **A NULL-pong dispatch bug produced a false negative first**: the parity
   branch gated on v≥4 instead of v∈{4,5}, so the slot/lz arms ran as
   accidental ping-pongs in the tuner (real buffer there → +7-11% misread,
   two "reproducible" tables!) and segfaulted in the driver at B=1 (NULL
   pong).  After the one-line fix, slot flipped from loser to winner in the
   same arena.  Lesson recorded: when tuner arms share a dispatch path,
   audit which buffers each arm ACTUALLY dereferences — and a "reproducible"
   arena reading reproduces its own bugs.
5. **csp placement residue race: flat** (best rr within 0.5% of default in
   every table; picked residue varies per allocation).  The driver's own
   buffer relations are deterministic per process, so this is cheap
   insurance against a bad allocator draw, not a measured gain.
6. tryout.sh remains broken for chain cases ($W used before defined, and
   the remote check.py gets a literal `$W/c.bin`): confirmed L13_rader/
   L17_matrixsimd's ice_r4 reports; used their `W=$PWD/build/tryout/<name>`
   env-prefix workaround + manual check.py + manual .chain cmp all round.

### Borrowed, plainly

- **Map form and divider doctrine**: corpus §10 §2 + 1000f989's mapF via
  L17_matrixsimd's ice_r4 s6 (shared denominator per 8 points, d=fma(s,y,1),
  one exact vdivpd, 1e-300 guard folded into the fma addend) and
  L13_rader's ice_r4 (rsqrt14 + 2-Newton exact ladder and the "no reason to
  ever go float-seed" precision arithmetic).
- **In-place state arena in final_out**: L17_matrixsimd ice_r4 item 1.
- **Split-complex chain-internal layout**: L64_blocked ice_r4 item 3 — their
  −1.4%; mine −2.2% (0.753 vs 0.770) because the DEINT here was pure
  overhead.
- **Eager-not-lazy**: L17_matrixsimd's v4 negative result, trusted, then
  re-confirmed with my own lz arms.
- **Bit-identical race arms as the legality condition for adaptive picks**:
  L13_rader ice_r4.
- tryout.sh workaround: L13_rader / L17_matrixsimd ice_r4 infra notes.

### What I would do next

1. **Trim the arms** after one scored round: fma/pp/lz all lost by ≥6% in
   every reading — keep them compiled/forceable, drop them from the race
   (pick-flip exposure, r3 doctrine).  The race that still earns its place
   is {div, div+pfs, slot, slot+pfs} + the residue phase.
2. The remaining 1.45× over the pool floor is group-granularity latency:
   phase-B groups are ~390 uops vs the 352-entry ROB, so consecutive groups
   barely overlap.  The known cures all spill (gcc 11.4, ~48 live zmm) —
   measured r3.  The one untried shape: split each phase-B group's STORE
   tail (post stages + 16 stores) off into the next iteration's head, ~24
   extra live registers... prototype only if the scored gap to the rivals'
   0.70 µs matters more than the risk.
3. The zero-exchange ROTATING layout (384 shuffles/vol steady, −6% pool) is
   blocked at B=64 only by the c-parity double copy (1.5 MiB > L2).  At B=1
   everything fits L1 — it could take the B=1 cell by ~5% if that cell ever
   scores.
4. csplit is rebuilt whenever the driver hands a different c pointer or
   placement; if a future harness alternates two c fields, add a second
   cache slot like aa_setup's shadow pair.
