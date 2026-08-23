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
