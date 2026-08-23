# L8_radix8 — strategy record (ICE panel)

Pre-ICE history (panel rounds r1–r11 on the Cascade Lake panel) is documented in
the header of `impl/L8_radix8.c`: 52-instruction split-complex radix-8 codelet
(56 flops, Yavne minimum, w8 twiddles FMA-folded), 2p / fused-1f / 3p kernel
shapes, spread-t0 + prefetchw streaming, single-bit-class regime-gated tuner.

## Round ice_r1 (no record was written; reconstructed from the leaderboard)

Shipped the panel_r11 CLX state unchanged. Node result on the graded chain
(L=8, B=64, m=2572, unitary, 1.00 MiB working set): **0.561 us/transform**,
third of three behind L8_batchsimd 0.550 and L8_fusedaxes 0.556; MKL 0.623.
Pick was avx512-1f-pfs (default), node arena
`{1f-pfs=0.458 1f520-pfs=0.457 3p-pfs*=0.457 2p*=0.495}`. The rivals' arenas
on the same cell: batchsimd `FUSEDAA/s0=0.426, FUSED/s0=0.429`; fusedaxes
`fusedAA2+pfs=0.409, fusedAA+pfs=0.412, fused+pfs=0.415` — both fused-family
files read their anti-aliased variants ahead, and both were ahead of me.

## Round ice_r2

### What I changed

1. **Ported the fusedAA2 anti-alias schedule from L8_fusedaxes** (their
   panel_r7 fusedAA + panel_r11 depth-3 rows; `aa_perm2_tab` taken verbatim,
   with attribution in the source) into my 1f fused shape as new kernels
   `1faa` / `1faa-pfs`:
   - phase A stores contiguously per x-plane, layout [x][ri][k1], instead of
     my split-planar [k1][x] comb (whose 512-B row stride puts 2 stores in
     every 16-line input-load window at ANY base — their line-granularity
     model says ~14 falsely blocked loads/volume, structural);
   - the 8-KiB scratch window is chosen at execute time from a 4-KiB slack so
     sigma = (scr − in)/64 ≡ 48 (mod 64), making phase-A loads alias-free
     against its own in-flight stores;
   - phase B iterates k1 in a permuted order indexed by
     c = (out − scr)/64 mod 8, collision-free at store-buffer depths 1–3.
   `aa_setup()` recomputes the window per (in,out) pair, which matters here
   because the graded chain ping-pongs two buffer pairs every step (cost ~15
   scalar ops per 64-volume execute). One choice covers the whole batch since
   a volume is exactly two 4-KiB pages.
2. **Verified bit-identity** of 1faa vs 1f by `cmp` of forced-candidate driver
   outputs (identical bytes), so 1faa joins the mid-regime installable bit
   class legally under the r11 single-bit-class rule. Mid pool is now
   {1faa-pfs (default), 1f-pfs, 1faa}, probe {3p-pfs}; 1f520-pfs and the 2p
   probe leave the pool (kernels stay compiled).
3. **B=1 hardwire flipped 2p → 1faa.** The 2p hardwire was a CLX verdict. On
   the bare-metal ICE node the create-arena read 1faa = 0.4684 us vs
   2p = 0.5236 / 1f = 0.5212 (−10 %), and forced driver-level chain A/Bs
   confirmed it: 1faa 0.566/0.567 vs 2p 0.576/0.579 us/transform with MKL
   steady at 0.619 — a reproducible ~1.8 %. Still no B=1 tournament.
4. **Tooling:** run functions now take an `l8aa_t` (scratch window + perm);
   added compile-time `-DL8R_FORCE_NAME='\"<cand>\"'` and `-DL8R_DEBUG=1`
   because tryout.sh builds/runs over ssh where env vars don't propagate —
   this is how all the A/Bs below were forced.

### Operation count

Unchanged: 1248 vector FP (8-wide) + 896 shuffles + 256/256 loads/stores per
volume for the 1f/1faa shape; the AA variant changes only addresses and store
order. Port floor on ICX (shuffles port-5-only, FMA ports 0+5):
(1248+896)/2 = 1072 cycles/volume.

### What I measured (reserved ICE node a80n0 via tryout.sh, leased core)

Graded case L=8 B=64 m=2572 unitary chain:
- default build ships pick=avx512-1faa-pfs; driver min 0.575–0.582,
  rel_l2 = 2.267e-16, chain check passes (1.390e-13 vs tol 5.1e-11),
  repeatable byte-identical across runs.
- Dev-core noise is severe: identical binaries swing 0.558–0.727 us
  run-to-run, but the swings track MKL (0.618–0.674) almost perfectly, so I
  normalized every A/B by same-run MKL. Stable-slow outlier runs
  (min=median, sd<0.1 %, e.g. 0.727 with MKL 0.674) are environmental.
- MKL-normalized at B=64 chain: 1faa-pfs 0.869–0.891, 1f-pfs 0.858–0.909,
  3p-pfs 0.898/0.901 — **parity between 1faa-pfs and 1f-pfs** (0.890 vs
  0.891 on the clean back-to-back pair).
- In-plan arena (nb=64): 1faa-pfs=0.4905, 1f-pfs=0.4895, 1faa=0.4992,
  3p-pfs=0.4882 — a four-way tie at dev-core noise.
- B=1 chain: 1faa 0.566/0.567 vs 2p 0.576/0.579 (MKL 0.619 in 3 of 4 runs).
- B=2048 streaming regression: pick 3p-pfs-pfw unchanged, 1.208 us min,
  passes; streaming paths were not touched.

### What did not work / null results

- **AA bought no measurable mean at the graded B=64 chain cell** (parity at
  0.890 vs 0.891 ×MKL), despite fusedaxes' quiet-window ice_r1 arena reading
  AA2 +1.5 % over fused+pfs. I ship it anyway: at parity the depth-3 schedule
  still deletes the (out−scr) allocation lottery, i.e. it buys variance, not
  mean — and 1f-pfs stays installable so the quiet-window arena can take the
  pick back with a >2 % win.
- **1faa plain (no software prefetch) is not better on bare metal** at B=64
  chain: 0.577 (MKL 0.661) and one 0.727 outlier vs 1faa-pfs 0.564/0.574;
  arena 0.499 vs 0.491. The hypothesis that bare-metal HW prefetchers make
  spread-t0 redundant (batchsimd is probing the same question this round) is
  answered no for my shape; pfs keeps ~2 %.

### Borrowed

- `aa_perm2_tab`, the sigma=48 execute-time base selection, and the whole
  anti-alias analysis: **L8_fusedaxes** (fusedAA/fusedAA2). Noted that
  **L8_batchsimd** independently moved its mid anchor to FUSEDAA2+s0 this
  round — all three L=8 entries have now converged on the same schedule.

### What I would do next

1. The one alias channel nobody handles: phase-A input loads of volume b+1
   vs the previous volume's in-flight phase-B out-stores at each volume
   boundary. The residue (in − out)/64 mod 64 is a driver-allocation lottery,
   but it is KNOWN at aa_setup time, and phase-A iterations write disjoint
   scratch slabs, so a rotated/permuted phase-A order could dodge it — same
   trick, third application (~8 rotations to brute-force against the last
   2–3 phase-B store windows).
2. Cross-volume software pipelining (interleave phase A of volume b+1 into
   phase B of volume b) to smooth the port-5 shuffle bursts and the
   A→B store-to-load wall; risk is GCC spills at ~40 live zmm.
3. Ask the monitor for per-candidate quiet-window arena numbers: dev-core
   leases cannot resolve <2 % on this node (MKL itself swings 7 % across
   leases), so round-over-round kernel decisions here should come from the
   scored window's arena strings, not tryout minima.

## Round ice_r4

### The task changed: the graded step is now FFT + the nonlinear map

`state <- (FFT(state)+c)/(1+|FFT(state)+c|)`, and the driver detects an
optional `fft3d_chain` weak symbol.  Unfused entries pay a driver-side map
pass — MKL went from 0.63 to **2.12–2.28 µs/xform** under it on this cell.
Everything this round went into owning the chain; the FFT-only tuner paths
are untouched (they still back the single-transform gate).

### What I built

1. **`fft3d_chain`, volume-major.**  The B volumes' chains are independent
   (FFT per-volume, map pointwise), so each volume runs all m=2572 steps
   while its working set stays **L1-resident** (state 8 KiB in-place + 8 KiB
   scratch + relaid c), instead of sweeping the 1 MiB batch through L2 every
   step — corpus §10 §3's "iterate each volume through all m steps while
   cache-resident".  Per-transform time is batch-invariant by construction
   (measured: 0.565 at B=1, B=64, and B=2048 alike).
2. **The map, fused into phase B in SPLIT-complex form** (on the (zr[j],
   zq[j]) register pairs after the third codelet), so |w|² = wr²+wi² is one
   mul + one FMA with no horizontal shuffle.  Ladder = the 1.00-scoring
   rival's `mapc` **taken verbatim from `ext/reference/fft_v4_solutions/
   1000f989_score1.00`**: `vrsqrt14pd` seed + 2 Newton in rsqrt space
   (1.5·(1.5·2⁻²⁸)² = 4.7e-17 — full double precision, unlike their bulk
   float-seed tier at ~1e-12/step, which the brief's arithmetic makes
   ILLEGAL at our m=2572: budget 1e-13/step) + |w| = s·q + **one exact
   `vdivpd` per point**; the divider runs beside the FMA pipes and 64
   divs/volume (~550 cy) hide under the ALU floor.  `+1e-300` folded into
   the first FMA kills the w=0 → NaN corner (their trick).  15 vector-ALU
   ops + 1 rsqrt + 1 div per 8 points.
3. **v2, the shipping shape: SPLIT-STATE chain.**  The inter-step state is
   only ever read by the next step's phase A, so it does not need to be
   interleaved-natural.  Keeping it split-complex in the exact register/lane
   shape phase B ends with deletes **ZUNTRI (384 shuffles) AND the phase-A
   deinterleave (128 shuffles): 896 → 384 shuffles/step**, the port-5-only
   class.  Tag-run established `ZTR8F out[j][l] = in[SW(l)][j]`, SW =
   (0,1,4,5,2,3,6,7), an involution — so state lanes carry one axis in SW
   order, the third codelet's feed permutation is SW itself, and EVERY
   permutation is compile-time.  The axis roles rotate (A,B,C)→(B,C,A) per
   step, period 3: one uniform kernel text serves all steps; c is pre-relaid
   per volume into 3 rotation-phase variants (24 KiB, hot one L1-resident);
   step t (0-based) uses variant (t+2) mod 3; the final state converts to
   natural interleaved once per volume (1024 scalar moves per 2572 steps).
   First step = unpack-input variant with the fpiinv feed (lanes carry z in
   FPI order there); phase-B tail identical.
4. **v1 (kept for A/B, `-DL8R_CHAIN_NAME='"ilv"'`):** interleaved state,
   ZUNTRI kept, map before ZUNTRI with c relaid via a create-time **tag run
   of the real ZUNTRI+f_off network** (`build_chain_cidx`, self-verifying —
   no hand-derived residue can silently be wrong).  Node: 0.731–0.742.
5. **Alias re-tuning for the new layout** (both knobs compile-time A/B-able):
   - scratch offset σ = (scr−state)/64 mod 64: the fusedAA **σ=48 is wrong
     for split-state** (it was tuned for 16-line interleaved input windows;
     split phase A loads 8+8 lines).  Node sweep: σ=8/16 = **0.565**, 24 =
     0.568, 40 = 0.574, 48/56 = 0.582.  Shipped σ=8 (`-DL8R_CHSCR=8704`).
   - phase-B iteration order `chs_perm`: with the new contiguous row stores
     the orders split into two clean classes — aa rows c∈{0,1,7} = 0.582
     and natural / c=3 = 0.662 (at σ=48).  Shipped row 0 (good at both σ).
   - FTZ/DAZ set inside fft3d_chain only, saved/restored (corpus §10 §2).

### Operation count (steady split step, per volume = per transform+map)

Vector-ALU: 1248 FFT FP + 384 shuffles + 960 map FP + 64 rsqrt = 2656 →
port floor ~1328 cy ≈ 0.46 µs at 2.9 GHz; 64 vdivpd (~550 cy) and 640
loads/stores ride the other ports.  v1 for comparison: 3168 → ~1584 cy.

### Measured (a80n0, leased core, tryout; map chain m=2572 unless noted)

| case | result |
|---|---|
| B=64 graded chain | **0.565–0.568 µs/xform** (mode; many runs), MKL fallback 2.12–2.17 → **3.8×**; rivals' external best ≙ 0.699 → **19 % under** |
| B=1 / B=2048 | 0.565–0.582 / 0.565 (volume-major ⇒ batch-invariant) |
| correctness | single 2.267e-16; **map-chain 2.599e-11 vs tol 2.572e-10** (10× margin; v1 reads 1.660e-11 — the difference is pure reassociation from the rotated DFT axis order, exactly the corpus's ~3e-11 exact-tier scale); m=2,3,4,5 chains PASS (~1e-15) covering all three rotation phases; repeatable bit-identical; AVX2 fallback chain PASS locally (1.27e-15 at m=7) |
| dev noise | bimodal windows: identical binary reads 0.565 or ~0.644 (min=median, sd<0.4 %, MKL steady) — same environmental stable-slow state my ice_r2 record documented.  All A/B calls above use the 0.565-mode readings, which reproduced ≥3× each. |

### What did not work / nulls, with numbers

- **Natural phase-B order** with the split stores: 0.662 vs 0.582 (σ=48) —
  the iteration order still matters ~14 % even though the aa_perm2 rows'
  original f_off-scatter rationale is gone.  aa rows c∈{0,1,7} tie; c=3 is
  as bad as natural.  I did not find the structural explanation this round;
  my line-residue model says the collision COUNT is order-invariant here, so
  it is a store-buffer timing effect — worth a PMU look next round.
- **Rolled phase-B loop** (`-DL8R_NOUNROLL`): 0.687 vs 0.582 — the unroll-8
  pragma (constant-folded perms/offsets) is worth ~15 %; the DSB holds it.
- **σ=48 carried over blindly**: +3 % (0.582 vs 0.565).  Alias constants do
  not transfer across layout changes; re-sweep after any load-pattern change.
- `--chain 1 --map` segfaults in ANY entry: the driver passes pong=NULL as
  final_out (it only allocates pong for chain>1) — its own fallback would
  crash too.  fft3d_chain now guards; recorded so nobody debugs it again.
- tryout.sh currently dies at `set -u` on `$W` (used in the CH= line before
  assignment) and skips the map-check/repeatability tail; workaround:
  `W=$PWD/build/tryout/<name> ./tryout.sh ...` and run check.py by hand.

### Borrowed, plainly

- Map ladder: **rival 1000f989** (`mapc`, verbatim structure); one-divide+
  Newton shape and lazy-map/relay-c ideas: corpus §10 §2/§3.
- Volume-major cache-residency: corpus §10 §3 (all seven rivals converged).
- 1f520-style re/im +1-line skew in the split state; the fpiinv feed trick
  (now generalized to SW): **L8_fusedaxes** lineage.
- The "derive permutations by tag-run, never by hand" method is my own r2
  L8R_FORCE-style hygiene applied to shuffle networks; recommended.

### What I would do next

1. **PMU the 0.565 residual** (bare-metal counters work here): 2656 ALU /
   0.565 µs @2.9 GHz = 1.62 uops/cy on ports 0+5 — 81 % of ceiling.  Where
   are the other 19 %: 4K false aliases (the perm effect says they exist),
   div/rsqrt latency chains at iteration tails, or the per-step serial
   boundary (last phase-B row → first phase-A load)?
2. The phase-B order and σ interact; a joint brute-force over (σ, perm) with
   a store-buffer timing model — or just 64 quiet-window probes — may find
   another 1–3 %.
3. If anyone needs B-major back (huge B, state > L3): the split-state kernel
   works unchanged with a rotating 3-buffer whole-batch layout; not needed
   at any graded cell.
4. The bimodal 0.565/0.644 dev-window state deserves a monitor-side look:
   it is not frequency (MKL steady) and survives within-run (sd 0.02 %).

## Round ice_r5

### Where I stood

ice_r4 leaderboard: **first at L=8, 0.564 µs/xform** (fusedaxes 0.744,
batchsimd 0.779, MKL 2.102 → 3.72×), and 19 % under the rivals' external
0.699 mark.  Rivals' r4 records contain nothing my v2 split-state chain
lacks (they are at 768/896 shuffles per step; I am at 384).  So this round
went after my own r4 "next" list, headlined by the one big structural idea
left: deeper fusion, per the ROOFLINE.md directive ("L=8 is L1-traffic-
bound — only traffic reduction, deeper fusion, fewer passes").

### What I tried: v3 "fused-pass" chain — and why it LOST

**The idea.**  In v2, phase-B iteration k1 ends with the complete new-state
row k1 in registers (zr/zq = cols, lanes already in the SW order the next
phase A wants), and the next step's phase A does nothing with that row but
reload it and run one elementwise RADIX8.  So fuse: after CH_MAP8, run the
next step's phase-A RADIX8 on the registers and store its result straight
into the next step's [row][ri][col^] scratch.  The steady step becomes ONE
pass — loads 128 scr + 128 c, stores 128 scr' — deleting the 8 KiB state
round-trip, i.e. **256 of 640 mem ops/step (40 % of L1 traffic)**, plus the
serial store→load wall at the step boundary.  Ping-pong scratches are
required (iteration k1 reads column k1 across all rows but writes row k1
whole); 2×8 KiB scr + 24 KiB clay = 41 KiB still fits L1.  ALU count is
unchanged (the phase-A codelet moves, it does not vanish).

**The numbers that killed it** (a80n0, B=64 m=2572 graded chain, all
correctness gates PASS at every variant):

| variant | µs/xform | window control |
|---|---|---|
| v3 first cut (inlined, 156 zmm spills) | 0.659 | MKL 2.100 |
| v3 noinline kernels + peeled step-0 branch (0 spills) | 0.654 | MKL 2.144 |
| v3 rolled (`-DL8R_NOUNROLL`) | 0.649 | MKL 2.113 |
| **v2, same window/core, back-to-back with v3's 0.654** | **0.570** | sd 0.00 % |

v3 loses **+15 %**, robust to spills, unroll, and the step-loop branch.
Note the rolled v3 did NOT pay v2's rolled 15 % penalty — consistent with
v3 being latency/allocation-bound, not fetch-bound.

**Mechanism (the transferable lesson).**  The fused iteration is ~420 uops
against the 352-entry ROB, so consecutive iterations stop overlapping at
allocation — AND the work it absorbed was exactly v2's phase-A pass: 8
small (~90 uop) independent iterations that the OOO engine uses as ILP
filler between phase B's long-latency groups (rsqrt→2 Newton→div chains).
Fusion removed the filler and lengthened the groups: both directions
wrong.  This is the same group-granularity wall L8_fusedaxes' r4 item 2
describes; my v3 is the measured proof that "fewer passes" REVERSES sign
once the chain is already cache-resident.  The ROOFLINE traffic-bound
call applies to the UNFUSED map configuration, not inside a fused
L1-resident chain.  Nobody at L=8 needs to try tail-fusing phase A again.

**A subtle find while verifying: v2 and v3 are NOT bit-identical, and
cannot be.**  Both pass everything (v2 chain 2.599e-11 — the exact r4
number, v3 2.391e-11, tol 2.6e-10; both ~8e-16 at m=2..4), but a checkpoint
harness (build/tryout/L8_radix8/test_v23.c) showed the first divergence
inside the fused kernel, and the disassembly explains it: GCC's default
`-ffp-contract=fast` treats `_mm512_mul_pd`/`_mm512_add_pd` as generic
vector ops and CONTRACTS the map's `w*sc` output muls into the tail
codelet's stage-1 adds — 8 extra FMAs/iteration (chba has 48 muls/iter vs
chblast's 56) — a fusion that v2's store/load boundary structurally
forbids.  Rounding-level, benign, MORE accurate if anything.  Rule for
everyone: **`cmp` is only a valid A/B gate between kernels whose mul→add
producer/consumer boundaries are identical; across a fusion boundary, use
the map-check.**  (This also retro-explains why "bit-identical by
construction" arguments must name the compiler contraction state.)

### What shipped

**v2, unchanged as the default** (chain rel_l2 2.599e-11 matches r4 to all
printed digits — the shipped chain is bit-identical to the r4 scored one).
The v3 kernels stay compiled and forceable (`L8R_CHAIN=v3` /
`-DL8R_CHAIN_NAME='"v3"'`) as the documented negative, noinline
(`KCHFN`) so they never perturb the v2 code generation (first cut inlined
them into fft3d_chain and GCC 11.4 spilled 156 zmm moves around the step
loop — the noinline cure is worth remembering).  Arena grew 48→64 KiB for
the ping-pong scratches; v1/v2 offsets untouched.

### Joint (σ, phase-B perm) sweep — r4 next-item 2, answered: FLAT

One lease, interleaved 3×, driver minima (µs), clean 0.570-mode window:
σ=8/row0 0.570-0.571, σ=16/row0 0.570-0.578, σ=24/row0 0.573-0.575,
σ=8/row1 0.570-0.571, σ=8/row7 0.569-0.570.  Nothing outside noise;
σ=8/row0 stays.  The r4 single-axis optima were already joint optima.

### Measured (a80n0; the graded chain unless noted)

| case | result |
|---|---|
| B=64 chain | **0.570 µs/xform** steady in-lease (many readings); MKL 2.11-2.15 → 3.7× |
| B=1 / B=2048 chain | 0.570 / 0.581 (volume-major, batch-invariant as designed) |
| correctness | single 2.267e-16 (B=64) / 2.269e-16 (B=1) / 1.914e-16 (B=2048); map-chain m=2572 2.599e-11 vs tol 2.572e-10 (10×); m=2,3,4 ~1e-15; repeatable byte-identical |

### Dev-window methodology notes (for whoever leases next)

- The stable-slow state (everything reads 0.649, sd 0.02 %, MKL steady)
  ate one whole sweep before I caught it: a σ=16 "regression" to 0.649 was
  reproduced by the σ=8 baseline a minute later.  **Never compare across
  leases; put all arms of an A/B in ONE lease, interleaved, with a baseline
  re-run**, driving the driver directly over ssh (the pattern is in this
  round's sweep script).  Same-lease A/Bs resolved 0.570 vs 0.654 with
  sd < 0.1 %.
- The FIRST driver invocation in a fresh lease often reads ~0.649 where
  the second reads 0.570 (reproduced 3×) — warm up the lease before
  believing anything, including tryout.sh output (tryout = one fresh
  lease, one invocation: its minima this round read 0.649 for binaries
  that steady-lease at 0.570).
- tryout.sh still needs `W=$PWD/build/tryout/<name>` prefixed and the
  map-check/repeatability run by hand (r4's `$W` bug, still present).

### Borrowed

- The ROB/group-size framing that explains the v3 negative:
  **L8_fusedaxes ice_r4** ("what I would do next" item 2).  Their untried
  store-tail-split idea is now half-answered: moving MORE work into the
  group (my v3) loses; the reverse direction (splitting groups) is the
  only one still open.
- Same-lease interleaved A/B discipline: L8_fusedaxes' "same-window pairs
  only" protocol, taken further because the window states got worse.

### What I would do next

1. **Split phase B into two half-passes** (B1: x-codelet, store regs=B^;
   B2: reload, ZTR8F, z-codelet, map, store): +256 mem ops but groups of
   ~116/~260 uops instead of ~380.  My v3 negative says group size beats
   traffic in this regime; this probes the same law from the other side.
   50/50, and the only structural idea left at 0.570.
2. PMU the 0.570 residual (counters work on bare metal, perf tool absent —
   needs a perf_event_open harness): 2656 p05 uops / 0.570 µs @2.9 GHz =
   1.61 uops/cy = 80 % of the 2-port ceiling.  Where the other 20 % goes
   (allocation stalls vs port conflicts vs the div tail) decides whether
   item 1 is worth it.
3. If the monitor can, score-window arena strings for chain shapes (v2 vs
   v3 there) would settle whether the dev-lease 0.570/0.649 bimodality
   exists in the drained window at all.

## Round ice_r6

### Where I stood

ice_r5 leaderboard: first at L=8, 0.570 µs/xform (fusedaxes 0.585, batchsimd
0.596, MKL 2.095 → 3.67×).  Both rivals adopted my v2 volume-major
split-state shape last round; my remaining margin is small and structural
ideas were down to two: the corpus lazy map (which I had never tried) and my
own r5 next-item 1, the half-pass phase-B split.  This round built and
measured BOTH, plus two map-ladder probes — all four are negatives, but the
probes turned into the round's real product: a measured resource map of the
0.570 step.  One micro-win shipped (RADIX8J in the chain, −0.4%).

### What I tried, with the numbers that killed it

All A/Bs same-lease, interleaved, env-forced from one binary (L8R_CHAIN=…)
or twin binaries in one lease; MKL bracketed every session (2.10–2.15
throughout — every number below is from fast-mode windows).

1. **v4 "lz", the corpus lazy map (s10 s2; L13_direct and L36_pencilfused
   ship variants of it): LOST 0.632–0.633 vs v2 0.570–0.571 (+11%), 4/4
   pairs.**  Inter-step state stays RAW; the map of step t runs in step
   t+1's phase A fused ahead of the phase-A codelet (identical per-point
   arithmetic and lane grouping, clay re-indexed natural-row, last step's
   map as a standalone 8 KiB pass).  Total ops and traffic IDENTICAL to v2
   — only group shapes change (~228/~184 uops vs 84/330).  My ROB
   prediction was wrong, and the mechanism is instructive: in v2 the
   rsqrt→2NR→vdivpd ladder sits at the very END of phase B feeding only
   stores — off every critical path; lz moves it to the HEAD of phase A
   where the divide gates the entire next codelet, and it concentrates the
   divider's ~550 cy/step of occupancy into the ~690-cy phase-A pass
   instead of spreading it across the whole step.  Lazy map is the right
   call only where the map would otherwise be a separate PASS (the unfused
   configurations it was invented for, and L13/L36's shapes); inside an
   already-fused step it strictly worsens the critical path.  Nobody at
   L=8 needs to rediscover this.
2. **v5 "sb", half-pass phase-B split (my r5 next-item 1): LOST 0.627–0.632
   vs v2 0.572 (+10%), 4/4 pairs.**  B1 = comb-load + x-codelet + register
   spill to a row scratch (~84 uops/iter); B2 = contiguous reload + ZTR8F
   + z-codelet + map + state store (~260 uops/iter); map stays at the tail;
   +128 loads +128 stores/step; scr2 pinned at state+12 pages exactly and a
   mod-4-clean B2 order {2,3,0,1,6,7,4,5} so the new pass adds no depth-3
   4K aliases.  The +10% ≈ the extra traffic plus the extra pass boundary
   — "mem ports at 25% occupancy" does NOT make an extra 8 KiB round trip
   free.  Together with v3 (+15%, r5) and lz (+11%), every deviation from
   v2's two-pass shape now measures +10–15%: **v2 is a genuine local
   optimum, and the structural search around it is CLOSED.**
3. **Map probe A, all-FMA reciprocal (rcp14 + 2 Newton replacing the one
   vdivpd; the ladder L64_radix8 ships): +6.7% (0.610–0.611 vs 0.572, 4/4).**
   Diagnostic: the added +256 p05 ops cost almost exactly their +128-cy
   throughput floor — **the p05 pool is the binding resource at the margin;
   the divider (~1/3 busy) contributes nothing at the margin.**  Correct
   (m=2572 chain 1.832e-11), kept compiled as -DL8R_MAPRCP=1.
4. **Map probe B, two direct divides (wr/dn, wi/dn deleting the recip's
   2 muls, −128 p05 ops/step): +46% (0.834 vs 0.572, 5/5).**  128
   vdivpd/step saturates the divider (16 back-to-back per iteration ≈
   190 cy against a ~165-cy iteration).  With probe A this brackets the
   design: 0 divs +6.7%, 64 divs fastest, 128 divs +46% — the shipped
   rsqrt14+2NR+1div+2mul ladder sits exactly at the divider/pool sweet
   spot.  Correct (2.175e-11), kept as -DL8R_MAP2DIV=1.

### What shipped

**v2 unchanged in shape, with RADIX8J as the chain phase-A codelet**
(-DL8R_CHJ, default 1): the 16 store-feeding output add/subs issue as
FMA/FNMA with broadcast 1.0 — round(1.0·x+y) = round(x+y), so the output is
bit-identical (cmp-verified on the node twice: J vs base and DJ vs D driver
outputs byte-equal; the shipped graded chain rel_l2 is 2.599e-11, the exact
r4/r5 scored number).  Node A/B: **J won or tied 6/6 interleaved pairs,
0.569–0.570 vs base 0.572–0.581 (~−0.4%)**, same sign 5/5 in the D-race
twins.  This is the L=6 panel_r9 "store-feeding FMAs beat store-feeding
adds" result finally propagated here, as the r9 VERDICT s6 asked.
Everything else (tuner pools, execute paths, streaming kernels, v1/v3
probes) is untouched; lz and sb stay compiled and forceable (L8R_CHAIN=lz /
sb) as documented negatives.

### Operation count (steady step, per volume = per transform+map)

Unchanged totals: 1248 FFT FP + 384 shuffles + 896 map ALU + 64 rsqrt =
2592 p05-pool ops → floor 1296 cy ≈ 0.447 µs at 2.9 GHz; 64 vdivpd and 640
loads/stores beside it.  CHJ converts 128 of the FFT add/subs per step to
FMA-class (count identical).  Measured 0.570 ≈ 78% of the pool floor, and
probe A says the pool is what binds at the margin — the remaining ~360 cy
is boundary/replay overhead that none of my structural rearrangements
reduced.

### Measured (a80n0, leased cores; graded chain m=2572 unless noted)

| case | result |
|---|---|
| B=64 graded chain, shipped default | **0.570 µs/xform** (warm lease, 4/4 readings; 0.569–0.570 across the J A/Bs); MKL 2.10–2.15 → **3.75×** |
| B=1 chain | 0.570 (volume-major, batch-invariant), chain rel_l2 9.154e-13 |
| correctness | single 2.267e-16; map-chain m=2572 **2.599e-11** vs tol 2.572e-10 (10×, bit-identical to the r5 scored chain); m=2,3,4,5 ≈ 1e-15 (all rotation phases); repeatable byte-identical |
| dev noise | fresh-lease first invocations read 0.648–0.651 (min=median, sd<0.1%, MKL normal) — the documented stable-slow state, 3× this round; every decision came from warm interleaved pairs |

### Borrowed, plainly

- Lazy map: corpus s10 s2 via **L13_direct** (lazy-pairmap) and
  **L36_pencilfused** (lazymap2) — adopted, measured, REJECTED at L=8
  (+11%; see negative 1 for why their geometries differ).
- All-FMA reciprocal probe: **L64_radix8**'s "all-FMA recip" — rejected
  here (+6.7%) but its failure is what proved the p05-pool diagnosis.
- RADIX8J association-order trick: **L6_pfa/L6_unrolled** panel_r9, via my
  own dormant r10 probe kernel — now shipped in the chain.

### What I would do next

1. The shape space is closed and the pool is at instruction minimum
   (Yavne-count codelets, 24-shuffle transposes, sweet-spot map).  The only
   instrument left for the residual ~20% over pool floor is the **PMU**
   (bare-metal counters work; perf tool absent — needs a perf_event_open
   harness): p05 utilization, ld_blocks.partial.address_alias, and
   rob/rs stall counts would say whether ANY of it is recoverable or
   whether 0.570 is this shape's floor.  Do this before building anything.
2. Priced and rejected without building: offloading one phase-A row-pair
   to ymm halves on port 1 (ICX runs 256-bit FP on p1 beside the 512-bit
   p0/p5 pipes. Balance point ~7 zmm rows : 1 ymm row → ≤1.6%/step best
   case, before spill risk and the extra code path; revisit only if the
   PMU shows p05 genuinely saturated end-to-end.
3. Defense: rivals copying v2+J converge to 0.569–0.570.  If someone finds
   a real win below that, it will NOT come from regrouping v2's passes or
   the map ladder — this round bought that certainty for everyone.

## Round ice_r7

### Where I stood

ice_r6 leaderboard: SECOND at L=8, 0.569 µs/xform — **L8_fusedaxes took the
cell at 0.555** by executing my own r5 "next" item 1 (the half-pass phase-B
split) with two things my rejected v5 "sb" (+10%) got wrong, and their record
politely expects me to take it back ("it is their idea executed").  This
round did exactly that, plus one new micro-win their record suggested but
nobody had run.

### What I built: v6 "hp", the block-scheduled half-pass split (new default)

Ported **L8_fusedaxes' ice_r6 "hp"** into my chain as mode 6 (L8R_CHAIN=v2
reverts; v2 kernels untouched).  The two corrections to my sb, per their
measurements and my re-derivation:

1. **Cut placement.**  The split sits AFTER the ZTR8F pair: B1 = comb x-load
   + x-codelet + both transposes + contiguous store to scr2 (~150 uops); B2 =
   contiguous reload **with the feed permutation (swp/fpiinv) composed into
   the load addresses — zero shuffles in B2** — + z-codelet + map + state
   store (~240 uops).  sb cut before the transpose (84/260); their hpf cut
   after the z-codelet (bunched divides).  Both unbalanced cuts lose ~10%;
   the balanced one wins.  Group-size law at L=8, now bracketed from all
   sides: 150–240-uop groups < 330–380 < 420.
2. **The grid frame instead of σ/perm knobs.**  State rows are re-laid as 16
   CONTIGUOUS lines ([row][col][ri]: re of (row,col) at row*128 + col*16, im
   at +8 — not v2's split-planar 8+8-line rows), and state, scr, scr2 and
   all three 8-KiB clay variants sit ≡ 0 mod 4096 (arena offsets 0 / 8192 /
   16384 / 24576, overlapping the exclusive v2/v3/v5 buffers).  Phase A
   comb-STORES scr[k][ri][a] — the transpose scatter moves to the store
   side, which the store buffer absorbs; v2 paid it as comb LOADS.  Every
   load in the step is then a contiguous 16-line block, and with natural
   B1/B2 order iteration u's loads sit wholly in grid block (u mod 4) — the
   one block iterations u−1..u−3 never store to.  Alias-free at store-buffer
   depth 3 by construction; no allocation lottery, no perm table.  Phase A
   runs their boundary-dodge order {0,4,5,6,7,1,2,3} (rows 1,2,3 4K-alias
   the previous step's last B2 stores at different lines, so they load
   last); A/B vs natural order read a tie (0.555/0.556 both), dodge kept as
   the analyzed choice.
3. Kept MY phase-A codelet CH_R8A = RADIX8J (store-feeding FMA joins,
   ice_r6, ~−0.4%), which their hp lacks.  Passes are KCHFN noinline (their
   251-spill cross-scheduling incident = my v3's, same cure).

Values and FP order are IDENTICAL to v2 — addressing only — and the node
confirmed it: forced-v2 vs hp driver outputs **byte-identical** (cmp), chain
rel_l2 = 2.599e-11, the exact r4/r5/r6 scored number.

### The new micro-win: hp runs ROLLED (front-end answer, shipped default)

L8_fusedaxes' r6 "next" item 1 suggested a timing probe nobody ran: force
the hp passes rolled and see if the rolled penalty flips sign (the unrolled
step is ~5.5k instructions/volume-step, over the ~2.3k-uop DSB → MITE-fed).
**It flips: rolled 0.552–0.554 vs unrolled 0.555–0.556, 5/5 warm interleaved
pairs, byte-identical outputs.**  v2's big fused group needed the unroll
(+15% rolled, ice_r4) to keep the ROB fed from constant-folded code; hp's
small groups fit the DSB rolled, and front-end supply wins.  Shipped as the
default (-DL8R_HPUNROLL=1 reverts; L8R_NOUNROLL still rolls everything).

### Operation count (steady hp step, per volume = per transform+map)

p05 pool unchanged: 1248 FFT FP + 384 shuffles + 896 map ALU + 64 rsqrt =
2592 → floor 1296 cy ≈ 0.447 µs at 2.9 GHz; 64 vdivpd beside it.  L1
traffic 640 → 896 loads+stores (+256, ports 2/3/4, ~35% occupancy).
Measured 0.553 = 1.24× pool floor (v2 was 1.28×): the split recovered about
a third of v2's residual allocation-stall cost, exactly as fusedaxes
measured on their frame.

### Measured (a80n0, leased core 3, graded chain B=64 m=2572 unless noted)

| case | result |
|---|---|
| B=64 graded chain, shipped | **0.553–0.554 µs/xform** (5 warm readings, sd ≤0.2%); forced v2 same lease 0.572×3; MKL 2.100 → **3.80×**; fusedaxes' scored hp = 0.555 |
| B=1 chain | 0.555 (fast window; one 0.636 stable-slow window), chain rel_l2 1.797e-13 |
| correctness | single 2.267e-16; map-chain m=2572 **2.599e-11** vs tol 2.6e-10 (bit-identical to every scored chain since r4, cmp-verified vs forced v2 on the node); m=2,3,4,5 at B=2 all ≈1e-15 (all rotation phases); B=2048 m=40 4.78e-15; repeatable byte-identical across runs |
| FFT-only execute (untouched paths) | 0.525 µs B=64, 2.267e-16, unchanged |

### What did not work / nulls

- **Phase-A order natural vs dodge: tie** (0.555/0.556 vs 0.555/0.556, 3
  pairs).  With RADIX8J in phase A the boundary channel is not resolvable at
  dev-lease precision; kept dodge (the analyzed order, and their raced pick).
- Infra note: tryout.sh's `$W`-before-assignment bug (r4) is still there;
  additionally reserve.sh --status false-negatives when slurm tools are not
  on the caller's PATH (squeue missing → "job not running" although the
  heartbeat is 41 s old).  Workaround: check RESERVATION.heartbeat age, then
  drive slot_lease.sh + ssh by hand exactly as tryout.sh does.

### Borrowed, plainly

- **The whole v6 shape: L8_fusedaxes ice_r6 "hp"** — cut placement, the
  mod-4096 grid frame, comb-store phase A, feed-composed B2 loads, the
  boundary-dodge slot order, and the noinline pass split.  (Which was itself
  my r5 next-item 1 executed on their gs frame — round-tripped, as they
  predicted.)  My additions on top: RADIX8J phase A (ice_r6) and the rolled
  default (their suggested probe, run and shipped by me).
- The rolled-flips-sign mechanism confirms their DSB/MITE hypothesis
  (their r6 next-item 1) — recorded here so nobody PMUs it from scratch.

### What I would do next

1. 0.553 = 1.24× pool floor.  The remaining ~290 cy/step: with groups small,
   aliasing engineered away, and the front-end now DSB-resident, the leading
   suspects are the B1→B2 store-forward latency (B2(u) reloads lines B1(u)
   stored ~1 pass earlier — fine) and the per-pass call/drain boundaries.
   A PMU harness (perf_event_open; counters work on bare metal) measuring
   idq.dsb_uops vs mite_uops and ld_blocks.no_sr would size what is left.
2. Granular unroll (roll B2 only, unroll phase A) — 3 builds, maybe 0.5%.
3. If fusedaxes copies RADIX8J + rolled back, we converge again at ~0.552;
   the next real differentiator at L=8 is probably nothing inside the step —
   both files are at the same instruction minimum on the same schedule.
   Defense of the cell now rests on the quiet-window tie-break.
