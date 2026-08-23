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

## Round ice_r5 (2026-08-23)

### Where I stood

ice_r4 scored: **0.744 µs/xform at the graded cell — second, 1.32× behind
L8_radix8's 0.564**, ahead of batchsimd 0.779, MKL fallback 2.10 → 2.8×.
Radix8's record explains their win completely, and it is exactly the round's
cumulative mandate at work: their **volume-major** chain (each volume runs
all m=2572 steps L1-resident) is the enabler my own r4 item 3 said my
zero-exchange **rotating layout** (384 shuffles/step) was missing — I had
derived the rotation and shelved it because the batch-level c copies
(3 × 512 KiB) blow the 1.25 MiB L2; per-volume they are 24 KiB and live in
L1.  Their v1 (volume-major but interleaved state, 896 shuffles) measured
0.731–0.742 ≈ my batch-major slot 0.742-0.755 — so volume-major alone is
NEUTRAL at B=64 (1 MiB fits L2 either way); **the shuffle deletion is the
win, volume-major is what makes it legal.**

### What I built: "vm3", the volume-major rotating-axes chain

1. **Adopted radix8's structure** (attribution: L8_radix8 ice_r4, their v2),
   re-derived independently in my own kernel vocabulary: state between my
   own steps stays split-complex in the exact shape phase B ends in —
   vector (group u = kY, reg k = kL, lane = kS in LANEX order) stored at
   u*128 + k*16 + ri*8 inside the volume's 8 KiB slab of final_out
   (in place).  Axis roles rotate (L,S,Y) ← (S,Y,L), period 3; the lane
   order LANEX is STATIONARY (trans8's positional map is out(j)[l] =
   in(LANEX[l])[j], so lanes always come out LANEX-ordered), which makes
   every register feed compile-time: phase A and the phase-B S-axis dft8s
   feed natural, the post-trans8 L-axis feed is `lanex` itself (an
   involution).  896 → **384 shuffles/step**; phase A has ZERO shuffles.
   c relaid into 3 rotation-phase layouts — csp is reused verbatim as the
   (z,x,y) phase; crelA/crelB are new (24 KiB hot per volume).  Step 1 =
   classic DEINT head + rotated store; step m lands (reg=kz, lane=kx
   LANEX) = the classic untrans_interleave state, so the tail is verbatim
   and the interleaved output costs zero extra passes.  vm3 requires
   m ≡ 1 (mod 3); m=2572 ≡ 1.  Other m (verification-size runs only) fall
   back to the r4 slot+pfs arms, which stay compiled and forced-able.
2. **A period-2 rotation exists and I derived it first** — swap only L↔S
   each step (layout k*128 + u*16), needing TWO relays (16 KiB hot) and
   an m-parity fix in step 1 only.  **Rejected by line-residue analysis
   before building**: its phase-B stores are strided 1 KiB (every group
   hits all 8 state blocks), so the volume-boundary 4K-alias channel is
   unavoidable by any load order.  Period-3 stores are CONTIGUOUS per
   group (one 1 KiB block), which makes the boundary dodgeable (next
   item).  Recorded so nobody builds the 2-relay variant for its smaller
   footprint without pricing this.
3. **Boundary dodge, free**: at each step boundary the previous step's
   last ~3 store blocks (groups 5,6,7) 4K-alias the next phase A's slot
   loads j ≡ u (mod 4), i.e. j ∈ {1,2,3} (j ∈ {5,6,7} hit the same lines
   = store-forward, harmless).  Phase-A slot order (0,4,5,6,7,1,2,3)
   defers exactly the false-aliasing slots.  The same channel my r3 AA2b
   deferral could NOT fix in the fused shape (collision count was
   permutation-invariant there) becomes fixable here because the stores
   are block-contiguous.  Note slot j's true data dependence is only on
   GROUP j's stores, so the dodge order also puts maximum store-age
   distance on the critical path.
4. **Relay residue PINNED ≡ state (crres = 0)**: relay loads of group u
   then live in grid block (u mod 4) — exactly the one mod-64 16-line
   block that the in-flight state stores (groups u−1..u−3) never occupy.
   Alias-free by construction; radix8 leaves this to the allocator.  The
   r4 crres residue race is retired (it read flat, the pin is structural).
5. **"gs" grid scratch — the round's own structural find.**  The classic
   [kY][slot] scratch makes the phase-B (scratch load) vs (in-flight
   state store) collision count σ-INVARIANT: each group loads ~9
   consecutive lines whose grid block cycles with u, while the last ~3
   groups' stores always fill 3 of the 4 blocks — which is WHY the σ race
   read flat (0.676–0.678 over all seven residues).  Layout [u][ri][s]
   (each group's row = 16 contiguous lines at u*128 doubles) + σ = 0
   (scr ≡ state mod 4096) + natural group order puts every group's 16
   loads entirely inside grid block (u mod 4) — the free block.  Same
   trick as the relay pin, applied to the scratch.  sr/si land in
   disjoint lines of one block so the +520 skew is unnecessary there.
   Phase A's comb-vs-window channel is unchanged (one side of a transpose
   must stride — conservation of misery).  Node arena: gd=0.668 vs
   dn=0.678 (+1.5%); driver-forced same-window: gs 0.585-0.592 vs
   classic-scratch 0.593-0.597 (~−1%).  The tuner picked it.
6. **Phase-B unroll (adopted from L8_radix8's r4 NOUNROLL A/B)**: gcc 11.4
   at icelake-server leaves my 8-iteration phase loops ROLLED (467
   instr/kernel).  `#pragma GCC unroll 8` on phase B: interleaved A/B
   pairs read unrolled 0.585/0.585 vs rolled 0.593/0.593/0.593 (−1.3%,
   nothing like their +15% but real).  **Unrolling phase A too spills
   catastrophically (240 zmm-rsp moves/kernel)** — same gcc trap as my r3
   16-arm switch; phase-B-only is spill-free (2762 instr, 0 spills,
   objdump-verified).  -DL8_VM_UNROLL={0,1,2} for the A/B.
7. **Tuner surgery**: the r4 10-arm map race is retired (racing
   non-bit-identical arms against vm3 would reintroduce cross-process
   output flips; slot+pfs stays the pinned fallback arm).  The new race
   is all-bit-identical by construction — σ ∈ {0,8,16,24,40,48,56}, then
   {nn,dn,nr,dr,gn,gd} order/scratch arms — so a pick flip can never
   change output; verified by cmp of .chain files across forced arms on
   the node.  Setup 0.380 → 0.21 s.  Dev override L8_VM_FORCE="sig,so,go,gs"
   forces any combo without a rebuild (env propagates through manual ssh,
   not through tryout).

### Operation count (per volume-step, steady vm3)

1248 FFT FP + 960 map FP + 64 rsqrt + **384 shuffles** = 2656 p05 uops →
floor 1328 cy ≈ 0.458 µs at 2.9 GHz (0.402 at the 3.3 GHz clk512 probes
report); 64 vdivpd hidden beside it; 640 L1 loads/stores.  Was 2976 pool
uops + L2-resident batch sweep in r4.  Working set per volume-chain:
state 8 KiB (in-place in final_out) + scratch 8.3 KiB + relays 24 KiB ≈
40 KiB vs 48 KiB L1d.  Zero spills in all six vm kernels (objdump).

### Measured on the node (a80n0, leased core; every A/B same-window)

| cell | this round | r4 shipped | MKL fallback | note |
|---|---|---|---|---|
| graded B=64 m=2572 | **0.585–0.600 min µs/xform** (fast windows; tuned=forced-best) | 0.744 | 2.11 → **3.6×** | −21% on my r4; radix8's r4 scored 0.564 |
| B=1 m=2572 | **0.601 min / 0.664 median** (chain rel_l2 9.15e-13) | 0.801 | 2.27 | see negative #6: B=1-only mostly-slow mode |
| B=2 m=2572, same lease as a B=64 0.585 | **0.586** | — | — | batch-invariance confirmed at B ≥ 2 |
| forced r4 slot+pfs, same window | 0.752 | — | 2.13 | vm3 −9.4% same-window |

Correctness: single-transform rel_l2 2.267e-16 (bit-identical to r11 — the
execute paths are untouched); **map-chain 2.599e-11 vs tol 2.57e-10 (10×
margin)** — exactly radix8's number, as it must be: same rotation, same
reassociation, same harness seeds.  Local reference harness (naive DFT +
exact map, B=3): PASS at every m in {1,2,3,4,5,6,7,10,13,16,19,25} covering
all three rotation phases, both parities of the fallback, and the m<4
degenerate cases; repeat calls bit-identical; forced gs/std/rolled/unrolled
chain outputs bit-identical on the node (cmp of .chain).  AVX2-only build
(-mno-avx512f, exact sqrt+div map in the same vm3 structure): PASS at
1.771 µs, chain 2.211e-11.

### What did NOT work / negatives, with numbers

1. **σ is a dead knob in the classic scratch layout**: 0.676–0.678 across
   all 7 residues (two independent windows).  The line-residue model says
   why (σ-invariant total collision count) — and the fix is the gs layout,
   not a better σ.  Radix8's "re-sweep σ after any layout change" advice
   transferred as analysis, not as a knob.
2. **aa_perm row 0 as phase-B group order: +1.4%** (nr/dr 0.686 vs nn/dn
   0.678).  Their 14% order effect does not transfer to my layout — and
   gs REQUIRES natural group order (the free-block argument), so the aa
   rows are structurally incompatible with the winning arm.
3. **Full unroll (phase A + B): 240 zmm spills/kernel** — gcc 11.4
   interleaves the unrolled phase-A iterations and defeats the allocator.
   Kept compiled as -DL8_VM_UNROLL=2 (diagnostic).
4. **The first invocation after a rebuild in an ssh session often reads
   ~14% slow** (0.667-0.678 vs 0.585 immediately after, same binary,
   min=median sd<0.1% both; seen in three sessions — though a plain
   tryout's first run also read 0.586 once, so it is "often", not
   "always").  The r3/r4 "bimodal window" state is at least partly this
   cold state, not a random ambient mode.  Dev rule unchanged: discard
   the first invocation, A/B only inside one session.
5. tryout.sh chain plumbing still broken ($W expands empty in the remote
   check.py argument): confirmed again; W=... env prefix + manual check.py
   remains the drill (numbers above went through the manual gate).
6. **B=1 (not a graded cell) runs a mostly-slow per-process mode**: median
   0.664, min 0.601 only at samples=30, sd 1.7% — while B=2 in the same
   lease reads 0.586 = B=64's 0.585 (batch-invariance holds at B ≥ 2, as
   radix8 claimed).  Not the cold-lease effect (three back-to-back B=1
   invocations all read 0.663).  Suspect: at B=1 the driver's 8 KiB
   in/out/pong heap allocations land at per-process 64 B residues instead
   of page-ish alignment, changing some absolute 4K relation my relative
   pins cannot control.  Worth one PMU look ONLY if B=1 ever scores.

### Borrowed, plainly

- **Volume-major L1-resident chain + rotating split state**: L8_radix8
  ice_r4 (their v2, 0.565).  I re-derived the bit-exchange bookkeeping in
  my own kernel's terms (their SW = my LANEX) rather than porting code.
- **Phase-B unroll pragma**: radix8's NOUNROLL A/B, re-priced here (+1.3%,
  not +15% — kernel-shape dependent).
- **Map ladder, in-place state, relay-per-phase**: my own r4 lineage (in
  turn from L17_matrixsimd/corpus §10), unchanged.
- The gs grid scratch, the relay-residue pin, the period-2 analysis and
  its rejection, and the slot-order boundary dodge are this file's own.

### What I would do next

1. The ~3-4% gap to radix8's best dev readings (0.585 vs their 0.565) with
   identical op counts is unexplained.  Candidates: their phase-A load/
   store schedule, DSB vs MITE delivery of the 2.7k-instr unrolled body
   (the dormant -DL8_PMC=1 probe prices idq.dsb_uops/mite_uops in one run
   if the scored context ever has PMU access), or simply their session's
   window.  The scored drained window is the real A/B — read both entries'
   numbers this round before spending kernels on it.
2. Phase A's comb-store channel (~2 lines per in-flight slot) is the one
   remaining modeled alias cost.  A "double-buffered row" scratch (write
   group rows round-robin into 5 of 8 slots' worth of extra lines) could
   free it at +4 KiB scratch; price only if the PMU confirms the channel.
3. If a future round changes m off ≡1 (mod 3): the m≡2 case needs only the
   step-1 tail swap (S1=z via one extra trans8 pair, already designed, see
   the r5 header notes); m≡0 additionally needs a y-strided untrans tail.
   Neither costs steady-state work; build them only if graded m changes.
4. The fallback arms (r4 slot/div/lz) and their tuner are now dead weight
   at every graded cell; keep exactly slot+pfs and delete the rest from
   the dispatch after one more scored round confirms vm3.

## Round ice_r6 (2026-08-23)

### Where I stood

ice_r5 scored: 0.585 µs/xform at the graded cell, second — L8_radix8 0.570
(spread 1.7%), me 0.585 (spread 0.2%), batchsimd 0.596, MKL fallback 2.095 →
3.67×.  My r5 "next" item 1 (the unexplained ~3% gap to radix8 at identical
op counts) is this round subsumed by a structural change that put me under
their number instead of chasing theirs.

### What I built: "hp", the half-pass phase-B split — THE ROUND'S WIN

Executed **L8_radix8's ice_r5 "what I would do next" item 1** — the one
structural idea left open at L=8 after their v3 measured that fusing MORE
work into a group loses 15% to ROB overflow.  Their framing: the vm3
phase-B group is ~380 uops against the 352-entry ROB, so consecutive groups
barely overlap and each group's rsqrt→2-Newton→div→store tail serializes.
hp probes the same group-size law from the other side — split phase B into
two passes through a SECOND grid scratch:

- **B1** (per u): load scr row u, S-axis dft8s, trans8 ×2, store the 16
  registers to scr2 row u ([u][ri][k], 16 contiguous lines).  ~150 uops.
- **B2** (per u): load scr2 row u with the lanex feed COMPOSED INTO THE
  LOAD ADDRESSES (zero shuffles, zero extra ops — lanex is compile-time),
  L-axis dft8s, map, state store.  ~240 uops.

Cost: +128 stores +128 loads per step (640 → 896 L1 ops), all on ports
2/3/4 far under the p05 floor; **p05 uop count unchanged at 2656**, so hp
is bit-identical to the fused mids.  Alias-freedom is my own gs frame, not
radix8's allocator-lottery version: scr, scr2, state and relay all pinned
≡ 0 mod 4096, gs comb phase A + dodge slot order — B1(u) loads scr block
(u mod 4) while B1(u−1..u−3)'s scr2 stores fill the other three blocks;
B2(u) loads scr2 block (u mod 4) against B2(u−1..u−3)'s state stores in
the other three; relay loads live in the free block (r5 pin).  Every pass
boundary alias-free by construction.

Also built **"hpf"** — the cut one stage later (B1 = both codelets, B2 =
pure map pass of 64 independent ~22-uop units) — as the A/B on cut
placement.  Both raced as tune_vm arms 6/7 (8-arm bit-identical pure-min
race, publishes vm-arena{…,hp=,hf=}); pick ships as gs=2/3.

### Measured (a80n0; same-lease interleaved A/B, 3 rounds, warm-discarded)

| arm | B=64 m=2572 min µs/xform | note |
|---|---|---|
| classic dn σ=16 (r5 ship) | 0.587–0.588 | sd ≤0.04% |
| gd (grid scratch) | 0.587–0.588 | |
| **hp** | **0.555–0.562** | **−5.5% on the incumbent; radix8's r5 scored 0.570** |
| hpf | 0.644–0.646 | +10%, see negatives |
| MKL same window | 2.142 | hp → **3.86×** |

Tuner (two independent processes, same lease): arena σ race flat
0.583–0.586 (as since r5), orders nn..gd 0.583–0.588, **hp=0.556/0.557,
hf=0.646 → pick sig=0 so=1 go=0 gs=2 both times**; tuned driver runs read
**0.555 / 0.555 (sd ≤0.02%)**.  B=1 **0.556** (first invocation in the
lease read the known cold 0.633; r5's "B=1 mostly-slow mode" did not
reappear); B=2 0.559 — batch-invariant.  Setup 0.21 s.

Correctness: single rel_l2 2.267e-16 (bit-identical to r11, execute paths
untouched); **map-chain 2.599e-11 vs tol 2.57e-10 at B=64 — the exact r5
number, because hp is bit-identical**: forced gs∈{0,1,2,3} .chain outputs
byte-identical locally (B=3, m=13) AND on the node (B=64, m=2572, cmp);
B=1 chain 1.797e-13, B=2 1.024e-12; local harness PASS at m ∈ {1,2,3,4,5,
7,10,13,16,19,25}; repeatable identical across runs; AVX2-only build
(-mno-avx512f) PASS 1.775 µs, chain 2.211e-11 (r5's exact number).

### Operation count (per volume-step, hp steady)

Unchanged p05: 1248 FFT FP + 960 map FP + 64 rsqrt + 384 shuffles = 2656 →
floor ~1328 cy ≈ 0.458 µs at 2.9 GHz; 64 vdivpd hidden; L1 traffic now 896
ld/st (was 640).  0.555 µs = **1.21× the pool floor** (was 1.28×) — the
split recovered about a third of the residual, confirming allocation-stall
(group-granularity) as the dominant non-port cost, exactly as radix8's v3
mechanism predicted from the other direction.  Kernels: hp_b1_pass 1097
instr / hp_b2_pass 1823 / vol_vm_hpf 2580, all 0 spills, 0 memcpy
(objdump, icelake-server).

### What did NOT work / negatives, with numbers

1. **hpf (map-only B2): +10%** (0.644–0.646 vs hp 0.555, incumbent 0.588).
   Bunching all 64 divides/rsqrts into a pass with no FFT work to hide
   behind loses more than the small-group ROB win gains — L64_blocked's
   bunched-divide lesson reproduced inside a single volume step.  The map
   must stay interleaved with FFT work.  Cut placement is NOT free.
2. **Inlining both unrolled hp loops in one frame: 251 zmm spill moves**
   (3412-instr kernel) — gcc 11.4 cross-schedules the two loops.  The r3
   16-arm-switch trap in a new costume; radix8's r5 noinline cure applied
   (two direct calls per step, bodies ~1500 instr each, no measurable
   cost).  Partial-unroll alternatives (B1 rolled 2072 instr / B2 rolled
   1451 instr, both spill-free) exist but were not raced once noinline
   gave full unroll at 0 spills.
3. **Pair-interleaved volume chains ("pi", from L6_unrolled) — rejected by
   line-residue analysis before building**: at each step boundary volume
   B's phase-A slot loads j ∈ {1,2,3,5,6,7} ALL 4K-alias volume A's
   in-flight B2 state stores (the 8 KiB volume stride keeps residues equal
   while making addresses differ — within one volume slots 5,6,7 are
   same-address store-forwards, across volumes they become false aliases),
   only slots 0/4 are clean, no order fixes it; and the hot set doubles to
   ~49 KiB ≥ 48 KiB L1d.  At L=6 the volumes are small enough to pair; at
   L=8 they are not.  Recorded so nobody builds it.
4. The driver's `--chain 1 --map` NULL-pong hole (radix8 r4 note) also
   crashed MY entry via tryout; now guarded (`!final_out` return).  m=1
   never scores; the guard is insurance.

### Borrowed, plainly

- **The half-pass split direction**: L8_radix8 ice_r5 "next" item 1,
  executed on my gs alias-free frame with my grid-block pinning instead of
  their σ/perm knobs.  Their v3 (+15% for MORE fusion) and my hp (−5.5%
  for LESS) now bracket the group-size law at L=8 from both sides:
  ~150–240-uop groups beat ~380-uop groups beat ~420-uop groups, and +256
  L1-resident mem ops are cheaper than one ROB-overflowing group.  That is
  the transferable sentence.
- **The noinline spill cure**: L8_radix8 ice_r5.
- **The chain-1 guard**: L8_radix8 ice_r4 infra note.

### What I would do next

1. The remaining 1.21× over the pool floor: the step is now ~5.5k
   instructions through the front-end per 8 KiB volume — DSB capacity on
   ICX (~2.3k uops) cannot hold it, so MITE delivery is the next suspect;
   the dormant -DL8_PMC=1 probe (idq.dsb_uops/mite_uops) answers it in one
   run if any context ever gets PMU access.  A cheaper timing-only probe:
   force VM_UNROLL=0 on the hp passes (rolled loops fit the DSB) and see
   whether the rolled penalty flips sign inside hp.
2. The untried third cut (before trans8, radix8's original sketch:
   B1 ~116 / B2 ~290) is strictly less balanced than hp's 150/240 and hpf
   showed unbalanced cuts lose; not worth a round unless hp's win erodes
   in the scored window.
3. Expect radix8 to take hp back (it is their idea executed); if parity
   resumes, the next differentiator is likely their remaining σ=8/row-0
   constants re-swept INSIDE the hp shape, and the phase-A comb channel
   (my r5 item 2) which is now the only modeled alias cost left.
4. Trim: hpf stays compiled/forceable as the documented negative; drop it
   from the race after one scored round confirms (pick-flip hygiene).
