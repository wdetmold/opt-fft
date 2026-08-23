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
