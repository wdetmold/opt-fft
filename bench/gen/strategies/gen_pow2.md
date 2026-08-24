# gen_pow2 — strategy record (GENERALIZE panel, 2^k-axes class)

Class: 2^k axes.  Scored acceptance size: **32** (B=8, m=250 graded map
chain).  Accepted unscored: 16, 64, 128.  From round 3 the entry must take
any 2^k the driver asks for.

## Round gen_r1 — from the dense stub to a custody-chain L=32 in one round

### Starting point

The entry was the harness-validation stub (dense L x L matrix per axis,
O(L^4)).  No prior gen rounds, no leaderboard.  The seed material named in
the brief (L8_radix8, L64_blocked) lives in `bench/ice`; the L64_blocked
strategy record (ice_r2..r8) is effectively a finished playbook for this
class and I executed it rather than rediscovering it.

### What was built (impl/gen_pow2.c, complete rewrite)

**L=32 fast path — the ice custody-chain structure re-derived for G = 4
slots:**

* **Custody layout.**  Chain state lives split-complex at slot
  `x*XS + y*KS + g*16` doubles (re +0, im +8, lanes = 8 ADJACENT z, z=8g+l).
  KS = 72 doubles (9 lines), XS = 32*KS+8 (289 lines) — both odd line counts
  (the standing 4K-aliasing proofing).  One volume 578 KB; state + custody c
  = 1.16 MB ≈ L2-resident.  **Each volume runs all m=250 steps while
  cache-hot** (per-volume chain iteration), natural layout touched only at
  the chain ends (custody<->natural is a pure interleave, zero transposes).
* **Lazy map** (buffers stay RAW z between steps; the graded
  `state <- (z+c)/(1+|z+c|)` applies at the START of the next step) with the
  ice_r8 **EXACT all-FMA MAP8V ladder** (rsqrt14 + 2 quadratic Newtons,
  rcp14 + 2 quadratic Newtons, 18 FMA-port ops + 2 seeds per 8 points, no
  divider, 1e-300 bias).  Shipped placement: a **per-plane map PREPASS**
  in place in custody (see "what did not work" — the register-fused form
  spilled).  Step 1 maps nothing (transforms x0); an epilogue maps step m's
  raw z into final_out.
* **z-pass: line-PAIR codelet.**  32 = 4 slots x 8 lanes does not fill an
  8x8 transpose, so TWO adjacent y-rows share one TR8:
  per row DFT4 over slots (k mod 4), lane twiddle W32^{l*k2} (CTWV, 3
  vector-constant pairs), stack both rows' 4 vectors, TR8 (re,im), one DFT8S
  over the former lane axis (k div 4), then X[8m+j] sits in lane (j mod 4)
  of O[2m + (j>=4)] — one vshuff64x2 per output slot component re-forms
  custody rows.  Custody form in, custody form out, stable across steps, no
  bit reversal.  108 FMA-port + 64 shuffle ops per 64 points.
* **y/x passes: vertical 32-point FFT** (vfft32), pure vertical SIMD:
  X[k2+8k1] = DFT4_b(W32^{b*k2} * DFT8_a(x[4a+b])), two passes through a
  4-KB stack line buffer (the FFT64S shape).  420 FMA-port ops per line of
  32 vector-pairs.  x-pass blocked per y-row: the 32 touched rows (one per
  plane, 18 KB) stay L1-hot across the 4 slots.
* **Everything always_inline** from the start (the ice_r7 lesson: gcc 11
  declines plain `inline` on 350-uop bodies; that bug cost them 3.5% for
  three rounds).
* **fft3d_execute** = deinterleave -> the same step (no map) -> interleave,
  so the checked path is the timed arithmetic.  **fft3d_chain** exported;
  for L != 32 (and any build without AVX-512) it falls back to a generic
  exact chain (execute + scalar sqrt/divide map).
* **16/64/128**: generic scalar radix-2 DIT per axis (gather/scatter pencil,
  long-double twiddles).  Correct, slow (L=64: 6.1 ms/xform), unscored —
  placeholder until the custody engine is generalized over G.

### Operation count (per step-volume, AVX-512 vector ops, L=32)

map 4096 x (18+2) = 74K FMA + 8K seeds; z 512 pairs x 108 = 55K FMA + 33K
shuffles; y 128 lines x 420 = 54K; x 54K.  Total **~237K FMA-port + 33K
shuffles + 8K seeds**.  Traffic per step ~4 MB of L1/L2 movement, nothing
past L3.  Quiet-window profile (rdtsc, -DGP2_PROF=1): map 61.6K cyc,
z 51.8K, y 37.5K, x 39.8K = ~191K cyc/step-vol; phase floors put ~45 us as
this structure's ceiling-of-merit (current 63 us, ~75-85% port efficiency
everywhere).

### Measured on the node (a80n0 leased core via tryout.sh; same-window
### MKL pairs; MKL timed through the driver's unfused execute+map fallback)

| config | gen_pow2 us/step-vol (min) | same-window mkl_dfti | ratio |
|---|---|---|---|
| graded B=8 m=250 | **63.6** (quiet-window best 62.8, sd 0.02%) | 171.0 | **0.372** |
| B=1 m=250 | **63.5** (sd 0.04%) | 155.7 | 0.408 |

Correctness, final shipped build, all on the node:

| gate | B=8 | B=1 | tolerance |
|---|---|---|---|
| single call vs numpy | 2.876e-16 | 2.872e-16 | 1e-12 |
| two-step fused chain (m=2) | **1.329e-15** | 1.336e-15 | 3e-14 (22x margin) |
| chain end m=250 | 2.913e-14 | 2.770e-14 | 1e-10 (anchor floor) |
| repeatability | chain end bit-identical across independent runs | | |

16/64/128 single-call: PASS at 2.5e-16 / 3.5e-16 / 4.1e-16.  Setup ~0 s
(tables + two buffer allocs; no plan-time race yet).

### What did not work, with the numbers that killed it

* **Register-fused lazy map inside the z-pair codelet** (the ice sweep-A
  shape, my first version): gcc must hold 16 result zmm + 8 concurrent
  26-op map ladders -> **80 spill stores + 35 reloads per pair** (asm
  count), map cost 76K cyc/step-vol.  Moving the map to a per-plane
  in-place PREPASS (tiny loop body, 128 independent ladders for the OOO
  core, plane stays L1-hot for the z-lines that follow) cut it to 63K:
  **66.6 -> 62.8 us/step-vol**.  Lesson: input-side fusion is not free when
  the consumer already saturates the register file; ice got away with it at
  L=64 because their z-row body consumed values slot-by-slot.
* **GP2_PF=1 (T0-prefetch next y-pair's state rows in the z-phase)**:
  ON 68.8 vs OFF 66.6 same-window — a 3% LOSS.  L2-resident working set;
  prefetches are pure issue overhead (ice_r5's "software prefetch mostly
  loses" reconfirmed at 1/8 the working set).  Default 0.
* **GP2_PFC=1 (T0-prefetch c rows in the map prepass)**: 63.7 vs 63.2 —
  wash-to-loss.  The prepass streams c near-sequentially; the L2 streamer
  already has it.  Default 0.
* **vfft32x2 (two slots per vertical-FFT call for ILP)**: y-pass 37.5K ->
  56.3K cyc, x-pass 39.8K -> 45.2K; 72.4 vs 63 us overall.  Eight live
  v8d[8] arrays overflow the 32-register file — the exact failure mode that
  killed my fused map, one experiment later.  Removed.
* **GP2_PREMAP=0 (output-side map fusion into the x-pass stores of steps
  1..m-1, the ice ZMS=1 shape)**: bit-identical to the prepass scheme,
  raced alternating same-window: xmap 63.4/64.9 vs premap 63.6/64.4 —
  a WASH (the map's ~60K cyc cost is placement-invariant; the x-pass
  balloons to 99.8K when it eats the map).  Prepass ships by simplest-wins;
  xmap kept compilable as the raced control.

### Borrowed / attribution

* The entire architecture — custody split-complex layout, per-volume chain
  residency, lazy map, odd strides, materialize-only-at-the-ends — from
  **bench/ice L64_blocked** (ice_r5..r8 record + exemplar source), itself
  descended from rival pipeline 1000f989's run64_zsplit and L64_radix8's
  split-state observation (attributions in that record).
* Verbatim code lifts from the ice_r8 exemplar: DFT8S, TR8, CTWS/CTWV,
  MAP8V (MAPDIV=3 exact tier), DEIN/ILV macros, the FFT64S two-stage
  register-array shape (as vfft32).
* The 4x8 line-pair z-codelet with the vshuff64x2 slot re-form, and the
  DFT4S codelet: this entry (new — the G=4 adaptation the L64 code could
  not express).
* Method rules applied from the ice record: same-window-pairs-only,
  always_inline + count-the-spills asm audit, measure-then-restructure
  (rdtsc phase profiler before every structural bet), simplest-wins
  hysteresis on washes.

### Notes for the monitor

* tryout.sh has the ice-era `$W` bug at line 36 (`--cin '$W/c.bin'` is
  built before `W=` is assigned; under `set -u` the script dies for any
  m>1 case unless W is pre-seeded in the env), and line 49's command
  substitution hands check.py a remotely-unexpanded `$W/c.bin`, so the
  map-check errors with `/c.bin` not found.  Workaround used throughout:
  `W=$PWD/build/tryout/gen_pow2 ./tryout.sh ...` and run check.py manually
  on the node.  gen's check.py itself is fine (the ice m>2 `math` import
  bug is fixed here).
* No MKL binaries under build/a80n0/bin yet; I compiled sota/mkl_dfti.c
  into my own tryout dir for the same-window pairs above (MKL 2022.0.2,
  sequential, exact Makefile link line).

### Next round

1. **Generalize the custody engine over G = L/8** (needed by round 3's
   any-2^k rule and round 6's library assembly): G=8 is the ice single-line
   z-codelet verbatim; G=2 (L=16) packs 4 rows per TR8; G=16 (L=128) needs
   two TR8s per line-half and an L2-blocked x-pass (volume 4.6 MB is no
   longer L2-resident — expect the ice L3-regime lessons, incl. prefetch
   flipping back to a win).  The vertical pass generalizes as radix-8/4
   stages; keep the scratch-line two-pass shape.
2. **Map ladder scheduling**: 15 cyc/MAP8V vs ~10 floor; the 55-cyc serial
   ladder wants more in-flight independents.  Try 2-plane interleaving in
   the prepass, or the rivals' row-ahead software pipeline staged through
   the c-row registers (ice_r8 next-note).
3. **vfft32 join bubble**: pass 2 cannot start until pass 1's last b-group
   stores; ~25K cyc/step across y+x.  A k2-majority store order for pass 1
   (finish k2=0..1 across all b first) could let pass 2 start early —
   needs care to keep stores contiguous.
4. Adopt gen_planner/gen_race/gen_twiddle/gen_layout layers as they land;
   this entry currently has zero plan-time race (nothing to race yet) and
   trivially meets the 60-s budget (setup ~0 s).
5. Cross-arch guard rounds: the custody scheme has no NT stores and no
   Ice-Lake-specific constants; expect it to transfer, but re-race
   GP2_PREMAP on SPR (the wash may break either way).
