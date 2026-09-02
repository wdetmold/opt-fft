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

## Round gen_r2 — two combination wins the single-knob races missed

Standings into the round: led L=32 at 63.68 us (2.69x MKL 2022, r1
leaderboard).  Same architecture as r1; this round changed the map's FORM +
PLACEMENT as one move, and the z/y phase SCHEDULING.  Everything below was
raced on a80n0 via tryout.sh (leased core; the `$W` bug is FIXED this round,
tryout runs chains unaided — but its remote check.py leg still gets a
literal `'$W/c.bin'`, so all map-gates below were run by hand with check.py
on wallaby over the shared FS, exactly as gen_layout r2 describes).

### What changed (ships as the new defaults)

1. **GP2_MAPDIV=1 + GP2_PREMAP=0 (xmap), adopted as a PAIR.**  The map's
   1/(1+sqrt) tail is now ONE vdivpd (rsqrt14 + 2 quadratic Newtons still
   produce the sqrt; the rcp14 + 2-Newton reciprocal ladder is deleted:
   16 -> 12 FMA-port ops + 1 divider op per 8 points), and the map is fused
   into each step's x-pass stores (vfft32m), deleting the r1 map prepass's
   1.16 MB/step L2 round trip.  The reasoning that found it: my r1 profile
   plus this round's div-vs-ladder wash showed the prepass is
   L2-TRAFFIC-bound, not port-bound — so the div map alone did nothing
   (60419 vs 65143 map cyc across two windows whose OTHER phases moved by
   the same 7%: a pure window artifact, map/z ratio identical at 1.20), and
   r1's xmap-with-ladder was a wash because the ladder's 20 ops clogged the
   FMA-saturated x-pass.  Combined, the traffic saving survives and the
   1/den rides the otherwise-idle Ice Lake divider (~8 cyc/zmm, 2x faster
   than SKX): **63.9/64.3/65.4 -> 60.5/62.1/62.5 us** interleaved
   same-day pairs, ~-4%.
2. **GP2_ZYIL=1: skewed z/y plane pipeline.**  z-pairs are port-5-bound
   (1 shuffle/point: TR8 + slot re-form; port floor ~86 cyc/pair with FMA
   rebalanced), y-lines are pure-FMA; running them as separate per-plane
   phases leaves each phase's idle port dark.  The step now interleaves at
   CODELET granularity — 4 z-pairs of plane x, then one y-slot vfft32 of
   plane x-1 — so every ~350-uop OOO window spans a shuffle-heavy/FMA-heavy
   boundary.  Bit-identical output (same per-line arithmetic and order).
   Raced: **59.4/59.4/59.8 vs 60.3/61.1 us**, ~-2%.  Profile: fused z+y
   82.6K cyc/step vs 52.0K + 37.5K = 89.5K split.
3. Comment/knob hygiene: all r1 knobs kept compilable (GP2_PREMAP=1 is the
   raced control, MAPDIV=0 the all-FMA fallback for a divider-poor host —
   the cross-arch race can flip them per host).

### Shipped numbers (a80n0, leased core, graded case L=32 B=8 m=250)

| config | us/step-vol (min, quiet windows) | same-window MKL 2022 | ratio |
|---|---|---|---|
| r1 ship (prepass + ladder) | 63.6-65.4 this round | 171-187 | 0.372 (r1) |
| + xmap + div map | 60.5-62.5 | 176-187 | ~0.34 |
| + z/y skew (SHIPS) | **58.4-59.8** (best 58.39, B=1 58.84 sd 0.3%) | 171-187 | **~0.33** |

Phase profile (rdtsc, shipped build): z+y fused 82.6K cyc/step, x-pass (now
carries the map) 97.9K, total ~180K cyc/step-vol.  Op count per step-volume:
z 55.3K FMA + 32.8K shuffles; y 53.8K FMA; x 128 cols x (420 + 32x14) =
111.1K FMA-port + 4096 vdivpd + 4096 rsqrt14 seeds.  Port floor
(220K+33K)/2 = 126.5K cyc = ~44 us — shipped is ~75% port efficiency, and
the x-pass (97.9K vs ~58K floor) is now the single dominant phase.

Gates (final build, node, all by-hand check.py): single call 2.876e-16
(B=8) / 2.863e-16 (B=1) vs numpy, tol 1e-12; two-step fused m=2 gate
**1.334e-15 / 1.284e-15** (tol 3e-14, 22x margin — the vdivpd tail is
exactly rounded, strictly more accurate than the rcp ladder it replaced);
chain end m=250 2.914e-14 / 2.330e-14 (tol 1e-10); chain output
bit-identical across independent runs AND bit-identical to the r1-arithmetic
xmap build (the skew is pure reordering).  16/64/128 generic path untouched,
re-verified PASS (2.5e-16 / 3.5e-16).

### What did NOT work, with the number that killed it

* **Div map ALONE (in the r1 prepass)**: a wash — see the window-artifact
  analysis above.  Placement had to move with it; neither knob wins alone.
* **GP2_SCHED=1** (`optimize("schedule-insns","sched-pressure")` on the step
  bodies, gen_batchlane's SCHED15 / gen_powp's 25-family trick): **84.4 vs
  64.2 us, +31%** — y-pass 35.9K -> 65.8K cyc, x-pass 38.5K -> 64.3K.  The
  vfft32 line buffer is a deliberate 4-KB stack round trip, and pre-RA
  scheduling across it explodes live ranges into spills.  Their rule
  ("pays only on spill-bound bodies") holds; my bodies are
  spill-STRUCTURED, not spill-bound.  Knob kept for cross-arch evidence.
* **GP2_PF=1 re-race under xmap** (z-rows now L2-cold since the prepass is
  gone — the one scenario where r1's prefetch verdict might flip): 61.7/62.5
  vs 61.4 — still a wash-to-loss.
* **GP2_PFXC=1** (NEW: T0-prefetch the x-pass's own-column c rows in pass 1,
  motivated by c's XS = 18.5 KB stride defeating the L2 streamer and S+C
  riding the 1.25 MB L2 edge): **64.0 vs 61.5 us, +4%** in adjacent quiet
  windows.  That is FIVE independent software-prefetch losses across this
  campaign's records (bl8, gen_pfa_small, gen_pow2 r1 x2, this) — issue
  slots are always worth more than the latency hidden.  Closed permanently
  for L2-resident working sets.

### Borrowed / attribution (gen_r2)

* **gen_dense_prime r1**: the div-map shape (their standalone-map-pass
  measurement was the pointer; the twist here is that it only pays FUSED).
* **gen_batchlane r2 / gen_powp r1**: the sched-pressure attribute pattern
  (tested honestly, lost big here — their spill-bound precondition is the
  operative clause, now with a +31% counterexample from a spill-structured
  body).
* **gen_layout r2 harness notes**: the exact by-hand check.py invocation.
* The z/y codelet-granularity skew: this entry (new; generalizes
  gen_batchlane's divider-hiding idea from within-column to across-phases).

### What I would do next

1. **The x-pass is now the whole game**: 97.9K cyc vs ~58K floor.  The 64
   stack-buffer stores + 64 reloads per column and the pass-1/pass-2 join
   are the slack; a k1-blocked two-column software pipeline died once
   (vfft32x2, r1) but a 3/4-column STORE-side skew (start column n+1's
   pass 1 loads between column n's pass-2 store groups) has the same port
   logic as GP2_ZYIL and none of the register pressure.  Try that first.
2. **Generalize the custody engine over G = L/8** (round-3 any-2^k rule,
   round-6 library): unchanged plan from r1 — G=8 is the ice codelet
   verbatim, G=2 packs 4 rows/TR8, G=16 needs an L2-blocked x-pass.  The
   z/y skew and the fused div map carry over unchanged.
3. Adopt gen_planner/gen_race routing when it lands; expose GP2_PREMAP /
   GP2_MAPDIV / GP2_ZYIL as the race's candidate axes on CLX/SPR (the div
   map in particular is an Ice-Lake-divider bet — SKX-class hosts should
   re-race MAPDIV=0).
4. gen_layout adoption for the custody block stays deferred: 1.16 MB = 145
   4K pages sits in the STLB, and batchlane's r2 THP A/B on an L2-resident
   set measured a null; re-evaluate at G=16 (4.6 MB/volume).

## Round gen_r3 — the x-pass was FRONT-END bound: DSB-resident loop bodies

Standings into the round: led L=32 at 58.13 us (2.95x MKL 2022, r2
leaderboard).  This round's plan was my r2 next-step #1 (a cross-column
store-side skew for the x-pass); that experiment FAILED but its profile
numbers exposed the real limiter, which nobody in any corpus record has
named yet: the **front end**.  Also shipped the round-3 mandate: supports()
now takes ANY 2^k in 2..128 (2/4/8 join 16/64/128 on the generic path).

### The diagnosis chain (worth recording as method)

1. GP2_XSK (x-pass column skew, pass 1 of column n+1 software-pipelined
   between pass 2 codelets of column n through ping-pong H buffers,
   bit-identical): **x-phase 101.5K -> 103.3K cyc — a wash-to-loss.**
   So the x-pass join bubble was already covered by the OOO core, i.e. the
   pass is NOT latency-bound.
2. It is not port-bound either: 793 cyc/column measured vs ~434 FMA-port
   floor (r2 numbers).
3. What remains is the front end: the r2 x-pass loop body was the g-loop
   unrolled 4x = ~4 fully-inlined columns ~ 5K uops of 7-byte EVEX code —
   more than 2x the ~2.3K-uop DSB (uop cache), so every iteration streamed
   from the legacy decoder at ~16 code bytes/cyc ≈ 2.3 EVEX instr/cyc.
   The z/y q-body (~2K uops) had the same disease, and the XSK body
   (~2.6K uops) made it slightly worse — consistent with everything above.

### What changed (ships)

* **GP2_XU=1: ONE column per x-pass loop iteration** (~1.3K-uop body,
  DSB-resident; asm audit of the final binary: 1561 insns / 8989 bytes for
  the loop body, fits the 384-line DSB).  Same-window A/B:
  **x-phase 101.9K -> 92.0K cyc, 59.05 -> 57.55 us/step-vol**, bit-identical.
  Quiet-window x-phase settles at 88.5-89.3K.
* **GP2_ZU1=1: keep the 4-zpair sub-loop of the z/y skewed phase rolled**
  (q-body ~2K -> ~1.3K uops).  Small but consistent: z-phase 77.0-77.7K
  (ZU1) vs 77.4-82.0K (unrolled) across four windows.  Bit-identical.
* supports(): any 2^k in [2,128] (round-3 rule).  L=2/4/8 PASS at
  0/0/1.5e-16 on the generic path.
* Description string updated; all r1/r2 knobs kept compilable.

### Measured on the node (a80n0, leased core via tryout.sh, graded case)

| config | us/step-vol (min, quiet windows) | same-window MKL 2022 | ratio |
|---|---|---|---|
| r2 ship (XU=0) | 58.8-59.4 this round | 183-193 | ~0.32 |
| + XU=1 + ZU1=1 (SHIPS) | **57.4-57.8** (best 57.41; B=1 **58.0-58.2**) | 183-193 | **~0.31** |

Busy windows this round read 62-73 us with median >> min (neighbors'
AVX-512 load through the all-core turbo bin — gen_batchlane r2's bimodal
observation reconfirmed); all A/Bs above are control-first adjacent pairs,
several in sd<0.1% windows.  Phase profile (shipped, quiet): z+y fused
77.2-77.7K cyc/step, x-pass 88.5-89.3K, total ~167K cyc/step-vol (was ~180K).

Gates (final ship build, node, check.py by hand on the shared FS):
single call 2.876e-16 (B=8) / 2.863e-16 (B=1), tol 1e-12; two-step fused
m=2 **1.334e-15 / 1.284e-15** (tol 3e-14, 22x margin); chain end m=250
2.914e-14 / 2.330e-14 (tol 1e-10); chain output bit-identical across
independent runs AND bit-identical to the r2 arithmetic (both new knobs are
pure reordering/recompilation).  2^k regression: L=2/4/8/16/64/128 single
call PASS at 0 / 0 / 1.5e-16 / 2.5e-16 / 3.5e-16 / 4.1e-16.

### What did NOT work, with the number that killed it

* **GP2_XSK=1 (x-pass cross-column codelet skew, this round's plan A)**:
  63.0 vs 58.1 us, x-phase 107.6K vs 89.0K cyc (with the skew loop rolled;
  101.5->103.3K in the r2-shaped build).  The OOO core already spans the
  pass-1/pass-2 join, and the ping-pong H buffers double the stack traffic.
  Kept compilable as the raced control.  Together with r1's vfft32x2 and r2's
  GP2_SCHED, that is three independent losses for manual cross-codelet
  scheduling in this engine — the machine schedules better; feed its FRONT
  END instead.
* **GP2_XU=2 (fully rolled vp1/vp2 codelet loops, ~400-uop body)**: x-phase
  97.3K vs 89.3K (+7K cyc) — loop overhead plus runtime-k2 twiddle loads.
  The sweet spot is the LARGEST body that fits the DSB, not the smallest.
* **GP2_ZYF=1 (finer z/y interleave: single zpairs alternating with single
  vp1 y-codelets inside each q-group)**: 58.97 vs 57.69 us clean-window,
  z-phase 79.0K vs 77.7K.  Same lesson as XSK, from the other side: once
  DSB-fed, codelet-granularity phase mixing (GP2_ZYIL) is already enough;
  going finer only adds overhead.
* **GP2_MAPDIV=0 re-race under XU=1** (front-end fix could have shifted the
  div-vs-ladder balance): 61.1 vs 58.7 us, x-phase 97.3K vs 89.3K — the
  vdivpd map still wins; the divider is not the binding port even now.
* **GP2_ZYIL=0 re-race under XU=1**: z+y split 50.3K+37.3K = 87.6K vs fused
  77-79K — the r2 skew still pays.
* -falign-loops=32: wash (58.34 vs 58.42 min in adjacent windows); moot
  anyway since the monitor builds with the standard flag line.

### Borrowed / attribution (gen_r3)

* The diagnosis METHOD is the ice discipline (rdtsc phase profile before
  every structural bet + count-the-uops asm audit); the DSB-residency
  finding itself is new this round — no prior record (ours, ice, or rivals)
  mentions the front end.  Peers with fully-unrolled multi-thousand-uop hot
  loops (gen_powp's 25/27 pencils at 404-436 vector ops x4 sites unrolled,
  gen_pfa_large's DFT25M bodies, gen_batchlane's L=15 DFT5X2) should check
  their body sizes against the 2.3K-uop DSB; my +31% GP2_SCHED disaster in
  r2 may even have been partly this (sched-insns inflates code size).
* gen_batchlane r2: the bimodal-window observation (min-vs-median discipline
  under neighbor load), used to keep this round's A/Bs honest.
* gen_powp r2 / gen_layout r2: the by-hand check.py + cmp repeatability
  procedure (tryout's remote map-check leg still dies on the unexpanded
  '$W/c.bin'; its $W build bug is fixed).

### What I would do next

1. **Generalize the custody engine over G = L/8** (16/64/128 currently ride
   the O(L^4)-ish generic radix-2 path: 81 us / 7.4 ms / 63 ms per volume —
   a round-6 liability if a 2^k is drawn; 16, 32, 64 are all in 14..127).
   G=8 is the ice L64_blocked z-codelet verbatim, G=2 packs 4 rows per TR8,
   G=16 needs an L2-blocked x-pass.  Apply the DSB rule from the start: one
   column/line per loop body, never more.
2. **The remaining x-pass slack (~89K vs ~450/col port floor) is now most
   likely the S+C working set riding the 1.25 MB L2 edge** (1.183 MB + code
   + stack): try shaving KS padding (KS=68 keeps odd-line 4K-proofing at
   17 lines/row? — must re-derive), or accept it.
3. Re-race GP2_XU/GP2_ZU1 on CLX/SPR in the cross-arch guard (DSB sizes
   differ: SKX 1.5K uops, SPR 4K+ — the sweet spot moves; the knobs are the
   race axes).
4. Library layers: still zero-cost create() (~0 s, trivially inside budget);
   adopt gen_race wisdom keys when the custody engine grows real candidates
   (the XU/ZU1/MAPDIV knobs are exactly what its per-host race is for).

## Round gen_r4 — c-transpose on the scored size; the custody engine
## generalized over G = L/8 (L=16 and L=64 leave the generic path)

Standings into the round: led L=32 at 57.32 us (3.07x MKL 2022, r3
leaderboard).  Two thrusts this round: one scored-size win (GP2_CT) and the
structural mandate both my earlier records promised — G-generalization, the
round-6 insurance (16 and 64 are drawable in 14..127; the generic path
served them at 81 us / 7.4 ms per volume).

### Harness note (monitor, please read)

`./reserve.sh --status` (and therefore `./tryout.sh`) is broken on the login
host this session: `sbatch`/`squeue` do not exist there, so the reservation
looks dead while node a80n0 is alive and serving other implementers
(logs/reserve.log is a wall of "sbatch: command not found" from the cron
re-arm).  I replicated tryout.sh's exact steps over ssh — same gcc line,
same gen_input/check.py invocations, a proper slot lease (slot 1, core 3)
held for the whole session and released at the end.

### What changed (ships)

1. **GP2_CT=1: the custody c volume is stored X-FASTEST** — slot (y, g, x)
   at y*CYS + g*CGS + x*16 doubles, CGS = 520 (65 lines, odd), CYS = 4*CGS.
   Only the x-pass (vfft32m) and the epilogue read c, and both consume it
   along x: each x-pass column's 32 c-slots become ONE contiguous 4-KB
   stream instead of 32 touches at XS = 18.5 KB stride (which defeats the
   L2 streamer — the disease my r2 GP2_PFXC prefetch attack diagnosed
   correctly and treated wrongly: +4% issue overhead then, a free layout fix
   now).  Same values at every read => bit-identical output (cmp-verified
   against the r3-arithmetic control in every window).  Raced in THREE
   same-window interleaved sets: busy window 65.86/66.07/65.47 (ct0) vs
   65.01/64.17/64.51 (ct1); first quiet window 58.70 vs 57.77/57.38; final
   quiet window 55.97/55.96 vs 55.21/55.85.  Every pair favors CT, -0.5 to
   -2%; x-phase profile 108.2K -> 105.0K cyc same-window.  The conversion
   cost (nat_to_cust_c scatters at x*16) is once per chain volume, amortized
   over m=250 steps.
2. **The custody engine generalized over G = L/8** — the same architecture
   (custody split-complex, per-volume chain residency, lazy map fused into
   x-pass stores, z/y port-profile skew, DSB-rolled bodies, x-fastest c)
   instantiated per G:
   * **L=64 (G=8)**: strides K64=136/X64=8712 doubles (17/1089 lines, both
     odd — the ice SCKS/SCXS numbers); volume 4.46 MB, S+C = 8.7 MB: the L3
     REGIME.  z-row = the ice L64_blocked st=3 z-line VERBATIM in shape
     (DFT8 over slots, lane twiddle W64^{l*k2}, TR8 pair, DFT8 over lanes,
     direct slot store — G=8 fills the transpose from ONE row, no re-form
     shuffles).  Vertical vfft64 = DFT8_b(W64^{b*k2} DFT8_a), H[64] (8 KB)
     line buffer, pass loops rolled (a full column is ~2.5K uops, over the
     DSB — the r3 rule applied from birth).
   * **L=16 (G=2)**: K16=40/X16=648 (5/81 lines, odd); volume 81 KB,
     everything L2-resident.  z-codelet NEW: FOUR rows share one TR8 — per
     row DFT2 over the 2 slots, lane twiddle W16^l on the odd branch, stack
     4x2 vectors (lane j = 2r+k2), TR8 pair, DFT8 over lanes; the output
     slot re-forms from 128-bit granules of 4 O-vectors, 3 shuffles per
     slot-component (96 shuffles/64 points vs the zpair's 64 — acceptable,
     unscored size).  Vertical vfft16 = DFT2_b(W16^{b*k2} DFT8_a), H[16].
     No z/y skew (not worth complexity at 81 KB; plain per-plane z then y).
   * Routing: fft3d_create picks the engine for L in {16,32,64}; 2/4/8/128
     stay generic (128 cannot be drawn in round 6's 14..127).  create() is
     still ~0 s (tables + one posix_memalign).

### Measured on the node (a80n0 core 3, same-window pairs, check.py by hand)

| case | gen_pow2 | same-window MKL 2022 | ratio |
|---|---|---|---|
| L=32 B=8 m=250 (graded) | **55.27-55.52 us** (quiet, sd 0.02-4.8%) | 172.6 | **0.322** |
| L=32 B=1 m=250 | **55.34** (sd 0.02%) | — | |
| L=64 B=2 m=64 chain | **678.7 us/step-vol** (B=1 677.4) | 2036 (guru 3015) | **0.333** |
| L=64 single call | 1219 | — | (was 7.4 ms generic: 10.9x) |
| L=16 B=8 m=300 chain | **7.58 us/step-vol** (single 7.43) | 18.85 (guru 29.77) | **0.402** |

Note the node ran ~3% faster in this round's quiet windows than r3's (ct0
control measured 55.96 today vs its own 57.4 in r3) — cross-round absolute
comparisons are meaningless as always; the CT gain is the paired delta.

Gates (ship build, all PASS): L=32 single 2.876e-16 (B=8) / 2.872e-16 (B=1);
two-step fused m=2 **1.334e-15 / 1.326e-15** (tol 3e-14); chain end m=250
2.914e-14 / 2.762e-14; repeatable and bit-identical to the r3 arithmetic.
L=64: single 3.214e-16, m=2 **1.723e-15**, chain m=64 2.342e-14, repeatable.
L=16: single 2.397e-16, m=2 **9.855e-16**, chain m=300 2.321e-14.
2^k regression L=2/4/8/128: PASS at 0 / 0 / 1.3e-16 / 4.1e-16.

### L=64 physics (why 677 us is near this structure's wall)

Profile (rdtsc): z+y fused 1.042M cyc/step-vol, x-pass 1.266M, total ~2.3M.
FMA-port floor is ~750K cyc (~1.5M vector ops / 2 pipes) — the phase is
MEMORY-bound, not port-bound: per step the structure moves ~22 MB of L3
traffic (S read+write in z/y, S read+write in x, C read once = the
irreducible set for a 3-axis in-place step), and 22 MB / 677 us = 32.5 GB/s,
which IS this node's practical single-core L3 bandwidth.  Improving L=64
now requires cutting traffic below that set (cross-step fusion — the ice
ZMS idea that lost there), not scheduling.

### What did NOT work, with the number that killed it

* **GP64_PFS=1 (x-pass slab-burst prefetch, 128 T0/column covering row y+1
  across all 64 planes)**: 721-753 vs 679 us — too bursty; prefetch loss #6.
* **GP64_PFS=2 (next-column T0 fused into each pass-1 load — the ice
  L64_radix8 sc_pass23 hint, +12% for THEM)**: a WASH here, 677.5/677.7 vs
  677.4/679.2.  Their x-stage read mid cold every sweep; my custody x-pass
  re-reads data its own z/y phase wrote 4.4 MB ago, so half the slab is
  still L2/L3-warm.  Kept compilable; prefetch wash/loss #7.
* **GP64_HP=1 (2-MB-aligned block + MADV_HUGEPAGE + touch, the ice_r7
  hugepage move)**: LOSES ~1.7% (688-691 vs 677.7 x3, THP confirmed active,
  [madvise] mode).  The phase is L3-bandwidth-bound, not TLB-bound, and 2-MB
  frames flatten the 4-KB page-color scatter the odd-line padding relies on.
  gen_layout's r2 THP-null on L2-resident sets extends to an outright loss
  in the L3 regime.  Kept as a control.
* **GP2_PREMAP=1 at L=64** (map as its own per-plane prepass): 810/813 vs
  the same-window xmap's 721-753 — in the L3 regime the prepass's extra
  volume round trip is pure L3 traffic on top of an already
  bandwidth-saturated step.  The r2 fusion verdict transfers with a bigger
  margin.

### Borrowed / attribution (gen_r4)

* **ice L64_blocked (st=3, sc_pass23)**: the G=8 z-line codelet shape,
  taken verbatim; the strides K64/X64 = their SCKS/SCXS; the next-column
  load prefetch (raced: wash) and the hugepage mapping (raced: loss) — both
  honestly tested against their record's claims, both die in the custody-
  chain regime their 2-sweep streaming structure did not have.
* **GP2_CT (x-fastest c)**: this entry, new — no corpus record stores a
  chain operand in a different custody orientation than the state.  Peers
  with a read-only per-step operand consumed along one axis (every entry's
  fused chain reads c) should check whether their c layout matches their
  LAST pass's walk order; mine did not, for three rounds.
* The G=2 four-rows-per-TR8 z-codelet with granule scatter: this entry, new.

### What I would do next

1. **L=32 z+y phase re-profile under CT** (77K r3 -> today's quiet windows
   suggest ~74K): the x-pass took three rounds of wins (fusion, DSB, CT);
   the z/y skew phase has had one.  The zpair's 16 vshuff64x2 re-forms per
   pair might fold into the following y-pass loads (the y-pass reads the
   same rows the z-pairs just wrote).
2. **L=64 traffic cut**: the only remaining lever is structural — fuse the
   x-pass of step s with the z-pass of step s+1 per ky-slab (state never
   returns to L3 between them).  Ice's ZMS lost at this, but their loss was
   sweep-A overlap, not traffic; in my skewed structure the x-pass already
   ends the step.  High risk, one round of work, ~25% ceiling.
3. **L=128 (G=16)**: outside round 6's draw, but the library should not
   ship a 63-ms hole; two TR8s per line-half, L2-blocked x-pass.
4. Cross-arch: GP2_CT should transfer (it deletes a stride pathology, adds
   none); GP64_HP/GP64_PFS are exactly the wisdom-race axes for CLX/SPR.

## Round gen_r5 — the dual-select FMA twiddle fold validated (a wash that
## ships), the row-pad question settled, and the structural wall named

Standings into the round: led L=32 at 55.746 us (3.07x MKL 2022, r4
leaderboard).  This round ran three experiments on the scored size; one
ships (as a knob default, not a wall-clock win), two are negative results
worth their numbers.  Protocol upgrade first: adopted the one-lease-one-core
discipline from **gen_dense_prime gen_r5 / gen_batchlane gen_r4** (slot 2,
core 4, held for the whole session; every A/B alternates the same binaries
on that core).  New wrinkle observed: in fixed-order A/B/C loops the
FIRST-position binary reads consistently better once windows drift —
ROTATE the order every round (my 3-way race flipped verdict when rotated;
the fixed-order session was discarded).

### What shipped: GP2_FTW=1 — dual-select FMA-folded pass-2 twiddles

The literature 11 Tier 1 item (Bergach, arXiv:2604.00567 / Linzer-Feig;
"2^k codelets first" — this entry is that codelet family).  In vfft32 /
vfft32m / vp2_k2, each pass-2 twiddle product x = W32^j·h is factored
x = f·m: m computed by two FMAs from a stored ratio (c-form t = s/c when
|c|>=|s|, else s-form u = c/s — dual-select keeps every stored ratio <= 1),
and the scale f folded into the following DFT4 stage-1 butterfly (FMA in
place of add/sub).  Twiddle+butterfly drops 8 -> 6 ops per site; the
j = 8 site (W32^8 = -i, the b=2 arm of k2=4) needs no multiply at all
(4 pure adds).  Per 32-point line: 420 -> 390 FMA-port ops (-7.1%), in
BOTH the y- and x-passes.  Tables long-double, ratio rounded once; the
form select is compile-time (k2 constant after unrolling), so zero runtime
cost — exactly the "table-generation policy" the citation promises.

**Result: correct, accuracy-neutral, wall-clock WASH on Ice Lake.**
16 interleaved same-core rounds across three sessions: ftw1 55.42-55.81 vs
ftw0 55.44-56.32 min, every paired delta within ±0.3%.  Same-window prof
pair says the ops really do leave (x-phase 87.8K -> 82.3K cyc, z+y 78.6K
-> 76.4K) but the wall does not move — the engine is not FMA-port-bound
(the r3 verdict, now proven by subtraction: removing 7% of the port ops
changes nothing).  Gates: single 2.902e-16 (B=8), m=2 fused **1.338e-15**
(B=8) / 1.393e-15 (B=1) vs r4's 1.334e-15 — accuracy statistically
unchanged; chain end 3.328e-14 / 2.948e-14; bit-repeatable x3.
SHIPPED as default anyway: the op headroom is free, the accuracy claim of
the citation is confirmed at fp64 (no regression at ratio<=1), and on a
port-bound host (CLX's downclocked 512-bit units) the 7% should cash —
GP2_FTW is a first-class cross-arch race axis; =0 is the r4 arithmetic,
kept compilable.  First validation of this citation in performant code,
for whatever the campaign credit is worth: the honest verdict is
"free, not faster, here."

### What did NOT work, with the number that killed it

* **GP2_KS=64 (drop the custody row pad: S 578 -> 526 KB, S+C 1.11 ->
  1.06 MB vs the 1.25 MB L2)**: bit-identical output, and consistently
  +0.3-0.5% SLOWER in every rotated-order round (55.78-56.30 vs ftw1's
  55.53-55.81).  Two lessons.  (1) The L2-capacity-edge theory of the
  x-pass slack is DEAD — shrinking the set made it worse.  (2) I
  re-derived the hazard audit before building: with KS=64 this engine has
  NO store->load pair at equal addr mod 4K in any phase — the odd-line
  row rule's value at L=32 is not the ice 4K store-load proofing, it is
  **L1-SET UNIFORMITY of the y-pass**: stride 9 lines walks all 64 L1
  sets (gcd(9,64)=1); stride 8 hits 16 sets at 4 lines each and loses to
  conflict pressure from H/c/stack sharing those sets.  KS=72 stays; the
  knob remains for the cross-arch race (bigger-L2 hosts may flip it).
* **GP2_ZYIL=0 re-race under FTW** (the y-lines lost 7% of their FMA ops,
  so the z/y port balance shifted): 58.97-59.70 vs 56.65-57.23 — the skew
  still pays ~3.5%.  Keep =1.

### The wall, named (analysis, so nobody spends r6 rediscovering it)

Phase accounting at L=32: per step-volume the port floors are ~110K cyc
(220K FMA-port ops / 2 pipes; 33K shuffles hide under them) and the L2
transfer is ~44K cyc (z/y sweep 578 KB r+w, x sweep 578 KB r+w + 532 KB c
read, at ~64 B/cyc).  Measured ~158-166K = floors + modest overlap
residual.  Cutting ops 7% did not move the wall; cutting the set 5% made
it worse.  The two-sweep step structure is traffic-minimal for 3D (z and
y already share a sweep; x must cross planes), so **L=32 is at its
structural ceiling short of a different factorization of the whole step**.
Corollary for L=64, killing my own r4 next-step #2: the proposed
x(s)+z(s+1) cross-step fusion is traffic-NEUTRAL — z already rides the
z/y sweep, and unfusing it from y to fuse with x leaves y needing its own
sweep; same L3 crossings (~22 MB/step), same 32.5 GB/s wall.  Verified on
paper this round; do not build it.

### Measured on the node (a80n0 core 4, quiet windows, same-core pairs)

| case | gen_pow2 (ship) | same-window MKL 2022 | ratio |
|---|---|---|---|
| L=32 B=8 m=250 (graded) | **55.27-55.42 us** (min 55.274, sd 0.06%) | 175.3 | **0.315** |
| L=32 B=1 m=250 | **55.45** (sd 0.03%) | — | |
| L=16 B=8 m=300 chain | 7.60 us/step-vol (path untouched) | — | |
| L=64 B=2 m=64 chain | 685.6 us/step-vol (path untouched; window +1%) | — | |

Gates (ship build = flagless default, verified bit-identical to the raced
ftw1 binary): L=32 single 2.902e-16 / B=1 2.902e-16 (tol 1e-12); two-step
fused m=2 1.338e-15 / 1.393e-15 (tol 3e-14, 21x margin); chain end m=250
3.328e-14 / 2.948e-14 (tol 1e-10); bit-identical across 3 independent
runs.  L=16: m=2 9.855e-16, chain 2.321e-14 — code path untouched, values
identical to r4.  L=64: m=2 1.723e-15, chain 2.342e-14 — same.  2^k
regression L=2/4/8/16/64/128 single call: PASS at 0 / 0 / 1.3e-16 /
2.3e-16 / 3.2e-16 / 4.1e-16.

### Borrowed / attribution (gen_r5)

* **Literature 11 Tier 1 (Bergach arXiv:2604.00567, Linzer-Feig
  factorizations)**: the dual-select FMA twiddle fold, implemented and
  validated here first; verdict "free accuracy-safe op cut, wall-neutral
  on a non-port-bound engine."
* **gen_dense_prime gen_r5 / gen_batchlane gen_r4**: the one-lease-same-
  core A/B protocol.  Added the rotate-the-order corollary from my own
  fixed-order artifact.
* gen_powp r4 / gen_pfa_large r4's volume-major chain: checked, already
  native to this entry since r1 (per-volume custody residency).

### What I would do next

1. **L=128 (G=16) custody engine** — the last real hole in the class
   (63 ms generic; outside round 6's draw but not outside the library's
   dignity).  Two TR8s per line-half, L2-blocked x-pass; the FTW fold and
   DSB rules apply from birth.
2. **Cross-arch races when XARCH.md lands**: the axis set is now
   {FTW, MAPDIV, PREMAP, XU, ZU1, ZYIL, KS} — FTW and MAPDIV are the two
   most likely to flip on CLX (port-bound downclock, slower divider).
3. If anyone wants L=32 faster, the remaining lever is a different STEP
   factorization (e.g. two-axes-per-pass with in-register y×z tiles — the
   literature 11 Tier 2 shape), not scheduling: the current structure is
   at [port ∥ L2] with <8% residual.
4. FTW for vfft64i/vfft16i: op cut is real (-10% at L=64 pass 2) but
   L=64 is L3-bound and L=16 unscored — do it only as library hygiene in
   a quiet round.

## Round gen_r6 — the library-assembly round: L=16 gets the skew + the fold
## (-3.5%); L=32/64 verified at their walls and left alone

Standings into the round: led L=32 at 56.472 us (3.05x MKL 2022, r5
leaderboard; 2.28x ahead of the next entry).  Round 6 scores the ASSEMBLED
library on three surprise sizes in 14..127, so this entry's exposure is
16/32/64.  My r5 record names L=32's structural ceiling ([port ∥ L2],
<8% residual) and L=64's L3-bandwidth wall (32.5 GB/s); the one path with
real headroom was L=16 (ratio 0.402 vs MKL — my weakest — and it had never
received the r2 z/y skew or the r5 twiddle fold).  This round closed that
gap.  Session protocol: slot lease 2 / core 4 held for the whole session on
a80n0 (ssh replication of tryout.sh's exact steps — reserve.sh --status
still dies on the login host, the r4-documented breakage; ls of leases/
shows three other implementers doing the same), rotated-order interleaved
A/Bs throughout (my r5 rule).

### Cross-arch check first (XARCH SPR, results/xarch_spr_r5)

gen_pow2 L=32 B=8 on wallaby (SPR Gold 6448Y): chain per-call 82.6 ms vs
MKL 276.7 / fftw3_measure 323.5 → ratio 0.30, BETTER than the Ice Lake
0.32; gates pass (single 2.9e-16, chain 3.9e-14).  No portability flag, no
knob action needed; the custody engine transfers as predicted in r4.

### What changed (ships)

1. **GP16_ZYIL=1: the z/y port-profile skew instantiated at G=2.**  Same
   shape as the L=32 GP2_ZYIL (r2): plane x's zquads interleave at codelet
   granularity with plane x-1's y-lines, 2 zquads + 1 vfft16i per q-group.
   The case for it was stronger at 16 than at 32: the zquad16 codelet costs
   96 port-5 shuffles per 64 points (vs the zpair's 64) against only ~92
   FMA-class ops, so the unskewed z-phase is outright port-5-bound while
   the y-phase leaves port 5 dark.  Bit-identical output (cmp-verified
   against the r5-arithmetic base at m=300), -1..-2.8% alone, -2..-2.8%
   measured on top of the fold (adjacent same-state pairs).
2. **GP16_FTW=1: the dual-select FMA twiddle fold in vfft16i's pass 2**
   (literature 11 Tier 1, my r5 GP2_FTW carried down to the DFT2 sites).
   x = W16^k2·h factored f·m, m by two FMAs from a stored ratio <= 1
   (fwt16/fwf16, long double, rounded once), f folded into the DFT2
   add/sub as 4 FMAs; k2=4 (W16^4 = -i) is 4 pure adds.  Pass-2
   twiddle+butterfly 60 -> 44 ops per 16-point line (line total 164 -> 148,
   -10%), in BOTH the y- and x-passes.  Unlike at L=32 (where FTW was a
   proven wash), at L=16 the fold is a real wall-clock win: -2.6..-2.8% in
   3/3 clean adjacent pairs vs skew-only builds' -1..-2.8, and ftw+skew
   beat base 4/4 rotated rounds by 2.7-3.9%.  The difference from L=32 is
   honest and worth recording: vfft16i's butterfly (DFT2) is trivial, so
   the twiddle multiply is a much larger fraction of the line, and the
   16-point y/x phases sit closer to their port floor than the 32-point
   engine ever did.
3. L=32, L=64, generic 2^k paths: UNTOUCHED — chain outputs cmp-identical
   to the r5 binary, and the m=2/chain gate values reproduce r5's exact
   digits (1.338e-15 / 3.328e-14 at 32; 1.723e-15 / 2.342e-14 at 64).
   Code-layout hazard check (gen_twiddle gen_r5's case-bloat lesson: new
   code can move sizes that never execute it): three same-window L=32
   A/B pairs r6-vs-r5base read 56.73/56.77, 56.51/56.49, 56.61/56.45 —
   a wash, no tax.

### Operation count (L=16, per step-volume)

z 64 quads x (92 FMA + 96 shuffles) = 5.9K FMA + 6.1K shuffles (unchanged);
y 32 lines x 148 = 4.7K; x 32 cols x (148 + fused map 16x(12+2) + adds) ≈
12.0K FMA + 512 vdivpd + 512 seeds.  Fold saves ~1.0K FMA-port ops/step-vol;
skew moves nothing, it only overlaps the z-phase's port-5 serialization
with y-phase FMAs.

### Measured on the node (a80n0 core 4, same-core rotated pairs)

The L=16 windows this session were BIMODAL (~6.4-6.6 vs ~7.3-7.6 us states,
flipping mid-round twice — gen_batchlane r2's observation, sharpest yet);
every verdict above is from adjacent same-state pairs.

| case | r5 path (same window) | gen_r6 ship | same-window MKL 2022 | ratio |
|---|---|---|---|---|
| L=16 B=8 m=300 chain | 7.60-7.62 / 6.66 (fast state) | **7.31-7.35 / 6.41-6.48** | 18.97-21.60 | **~0.32-0.34** (was 0.402) |
| L=16 B=1 m=300 chain | — | **6.410** (sd 0.01%) | — | |
| L=32 B=8 m=250 (graded) | bit-identical path | **55.40** (quiet, sd 0.02%) | 184.2 | **0.301** |
| L=32 B=1 m=250 | — | **56.32** (sd 0.02%) | — | |
| L=64 B=2 m=64 chain | bit-identical path | 766 this window (L3-bound size; neighbors' load visible, r4 quiet floor was 678) | 1883 | 0.41 this window |

Gates (ship build = flagless defaults, node runs + check.py by hand on
wallaby over the shared FS — tryout's remote map-check leg still dies on
the literal '$W/c.bin'): L=16 single 2.367e-16 (B=8) / 2.357e-16 (B=1),
tol 1e-12; two-step fused m=2 **9.376e-16 / 8.650e-16** (tol 3e-14, 32x
margin; r5 was 9.855e-16 — the dual-select fold is accuracy-NEUTRAL-to-
better here too); chain end m=300 2.351e-14 / 1.657e-14 (tol 1e-10);
bit-repeatable across independent runs.  L=32: single 2.902e-16 (B=8) /
2.915e-16 (B=1), m=2 1.338e-15 / 1.393e-15, chain 3.328e-14 / 2.948e-14 —
all EXACTLY r5's digits.  L=64: 3.214e-16, m=2 1.723e-15, chain 2.342e-14 —
same.  2^k regression L=2/4/8/128 singles: PASS at 0 / 0 / 1.4e-16 /
4.1e-16.

### What was considered and deliberately NOT built, with the citations

* **L=64 FTW fold (vfft64i pass 2, ~-11% ops)**: skipped.  L=64 is
  L3-bandwidth-bound at 32.5 GB/s (r4 profile), and r5 proved by
  subtraction at L=32 that op cuts do not move a non-port-bound wall.  The
  op headroom argument that justified shipping FTW at 32 ("free for
  cross-arch") does not pay for new folded-DFT8 code on a path where even
  the port floor is 40% below the memory wall.
* **Paired-vdivpd map (one divider op per two sites)**: skipped on two
  entries' r5 evidence rather than re-raced — gen_layout gen_r5 measured
  gl_map16 at +9.7% (L=31) / +2% (L=100) in a REGISTER-FUSED exit (mine is
  exactly that shape), while gen_powp gen_r5's -1.3..-1.6% win was in an
  FMA-SATURATED x-pass; my x-pass is measured NOT port-bound (r5's FTW
  subtraction), and MAPDIV=0 racing showed the divider is nowhere near
  binding.  Both boundary conditions point to wash-to-loss here.
* **L=32 structural work**: nothing this round; the r5 wall analysis
  stands (two-sweep step is traffic-minimal, [port ∥ L2] with <8%
  residual).  The literature 11 Tier 2 two-axes-per-pass rewrite remains
  the only lever and is a full-round bet nobody should take while the cell
  leads by 2.3x.
* **L=128 (G=16) custody engine**: still parked — outside the 14..127
  draw; the 63-ms generic path remains a documented wart, not a scoring
  risk.

### Borrowed / attribution (gen_r6)

* **gen_twiddle gen_r5**: the case-bloat/code-layout hazard discipline —
  the reason the L=32 same-window A/B pairs above exist (they cleared it).
* **gen_layout gen_r5 + gen_powp gen_r5**: the paired-div boundary
  (adopted as a decision not to build; both records cited above).  Also
  gen_layout gen_r6 folded my r5 L1-set-uniformity mechanism into their
  gl_pad_stride doctrine — adoption noted for the monitor's ledger.
* **GP16_ZYIL / GP16_FTW**: my own r2/r5 techniques instantiated at G=2;
  the observation that the fold's wall-clock value INVERTS with radix size
  (wash at 32, -2.7% at 16, because DFT2's butterfly-to-twiddle ratio is
  the smallest possible) is this round's contribution to the FTW record.
* Session protocol: rotated-order same-core interleaved pairs (my r5),
  held-lease discipline (gen_dense_prime r5 / gen_batchlane r4).

### What I would do next

1. **If round 7 exists and a 2^k was drawn**: read the round-6 library
   numbers first — if the trunk routed 16/32/64 through this entry, the
   per-host wisdom race (gen_race) over {FTW, MAPDIV, PREMAP, XU, ZU1,
   ZYIL, GP16_*} is the remaining upside; the knobs are all compilable
   controls already.
2. **L=16 residual**: after skew+fold, ~21.5K cyc/step-vol vs ~15K
   summed port floors — the gap is now mostly the zquad's 96-shuffle
   scatter; folding the ZQ_SCAT granule re-form into the y-pass loads has
   the same shape as my r4 next-step #1 at 32 (never built there because
   ZYIL already overlaps the shuffles — same likely verdict here now that
   16 is skewed).  Only worth a window if 16 turns out to matter.
3. **L=64**: the wall stands; the only untested idea in any record that
   touches it is literature 11 Tier 2 stage-as-matrix restructuring.
   Coordinate with gen_powp/gen_pfa_large before anyone burns a round.
4. L=128 custody engine as library hygiene if the campaign continues past
   the draw.

## Round gen_r7 — the backlog spent: constant-per-site twiddle routing LOSES
## on this engine (-1%, both arms), the TLB theory of the x-pass residual
## dies, and the ship stays bit-identical to r6

Standings into the round: led L=32 at 56.524 us (3.04x MKL 2022, r6 board;
1.87x ahead of gen_race, the next entry).  The rounds-7/8 brief assigns the
queued literature backlog; its item 2 — **constant-per-site twiddle routing
(Garrido, literature 11 Tier 1), "try it on one mid-size cell (25/27/32)
where twiddle loads are measurable"** — names my scored size, and no other
record had touched it (grep confirmed).  My r5 wall analysis predicted a
wash; the honest thing was to build the strongest feasible form and measure.
Second thrust: one genuinely untested lever on the x-pass residual (TLB).
Session protocol: slot lease 2 / core 4 on a80n0 held for the session (ssh
replication of tryout.sh's exact steps — reserve.sh --status still dies on
wallaby, the r4-documented breakage), rotated-order same-core interleaved
pairs throughout, asm audit before racing.

### GP2_CPS: constant-per-site twiddle routing, the strongest x86 form

What the idea can even mean here: the site->constant BINDING has been
compile-time since r5's FTW (every unrolled site indexes tables with a
constant j); what remained runtime was the table VALUES.  gcc cannot prove
heap stores (S comes from posix_memalign, not a tracked allocator) never
alias a filled static double array, so every k2-group's broadcast twiddles
are re-loaded per column and nothing hoists.  GP2_CPS=1 compiles the L=32
hot-path tables in as LITERAL static const arrays: %a-exact doubles
generated on the node by the exact fill_fast_tables expressions (long-double
cosl/sinl, dual-select ratios), verified equal to the runtime fill by a
create-time memcmp (it passes: chain output is bit-identical to the r6
arithmetic).  On x86 that is the whole idea: no 64-bit vector immediates
exist, so a .rodata broadcast IS the compiled-in constant; the only thing
routing can add is aliasing-free hoist/CSE freedom.  =2 scopes it to the
pass-2 scalars only (z-pass vector tables stay runtime).

**Asm audit (the reason the race happened at all)**: fused step body
fft_step_xmap 2350 -> 2243 insns (-4.6%), vbroadcastsd 33 -> 7, packed
loads 289 -> 240, FMA count IDENTICAL (428) — gcc really does hoist and
share the constants once the aliasing hazard is gone.  Front-end-wise this
looked like free money (my own r3 DSB result), so it was raced properly.

**Result: a consistent LOSS, both arms.**
* =1: +0.7-1.6% in 5/5 clean rotated same-core pairs (57.1-57.9 vs ctl
  56.0-56.8 us adjacent; round-0 discarded, ctl median 74.5 sd 11.7% —
  neighbor burst).
* =2 (scalars only): +1.1-2.7% in 6/6 (56.6-57.3 vs 55.5-56.7).  The loss
  is IN the scalar folding, not the z-table hoist.

Mechanism, recorded because it inverts a naive front-end intuition: the
runtime tables' per-iteration broadcast re-loads were L1-hot issues on
half-idle load ports — free by r5's non-port-bound verdict — and they kept
every twiddle constant's live range ONE k2-group long: compiler-enforced
REMATERIALIZATION.  Compiled-in constants let gcc stretch those live ranges
across the column body (fewer instructions, more concurrent live values),
and in a body that already runs the register file at the edge (r1 vfft32x2
+15%, r7 confirms from the other side) the pressure costs more than the
loads ever did.  Garrido's routing pays where twiddles are per-butterfly
TRAFFIC; this engine converted them to site-constant L1 broadcasts in r5,
after which the loads themselves are the cheap half of the bargain.
Verdict for the campaign record: **constant-per-site twiddle routing is
already structurally present in any fully-unrolled FTW-style codelet; the
literal-constant final step is a small loss on ICL.**  Ships DEFAULT 0 —
bit- AND codegen-identical to the r6 ship; both arms kept compilable as
cross-arch race axes (CLX's port-bound downclock could flip the trade).

### GP2_HP32: the TLB theory of the x-pass residual, tested and dead

The x-pass's pass 1 touches 32 SCATTERED 4-KB pages per column (XS = 18.5 KB
stride; 145 state pages cycled by 128 columns against the 64-entry DTLB) —
a TLB profile neither batchlane's r2 THP-null (sequential streams) nor my r4
GP64_HP loss (L3-bandwidth regime) ever tested, and the S+C block is 1.11 MB
= ONE 2-MB frame.  GP2_HP32=1: 2-MB-aligned block + MADV_HUGEPAGE + touch.
Bit-identical output.  **A WASH: +-0.3%, mixed signs, 6 rotated rounds**
(e.g. 56.93/56.86/56.53/55.94/55.94/56.48 vs ctl 57.43/56.83/56.41/56.03/
55.64/56.12).  The OOO core hides the STLB-hit latency, same as it hides
everything else ever thrown at this residual.  With the r5 L2-capacity-edge
theory and seven prefetch results, every cheap theory of the ~50 cyc/column
x-pass slack is now dead: the r5 structural-ceiling verdict stands
unqualified.  Default 0 by simplest-wins; knob kept (CLX's smaller STLB is
the one host that might flip it).

### Ship state and measured numbers (a80n0 core 4, same-core windows)

Ship = flagless default = the r6 arithmetic and codegen exactly (chain
output cmp-identical to a -DGP2_CPS=0 control build and to r6's).

| case | gen_r7 ship | same-window MKL 2022 | ratio |
|---|---|---|---|
| L=32 B=8 m=250 (graded) | **55.51-56.80 us** (best 55.51, typical-window sd 0.01%) | 171.2-171.5 | **0.325** |
| L=32 B=1 m=250 | **55.82-56.19** (best 55.82) | — | |
| L=16 B=8 m=300 chain | 6.428 (path untouched) | — | |
| L=64 B=2 m=64 chain | 678.0 (path untouched; r4 quiet floor 678) | — | |

Gates (ship build, node, check.py on the node this round — its map-check leg
works when invoked by hand with explicit --cin): L=32 single 2.902e-16 (B=8)
/ 2.915e-16 (B=1), tol 1e-12; two-step fused m=2 **1.338e-15 / 1.393e-15**
(tol 3e-14, 21x margin); chain end m=250 3.328e-14 / 2.948e-14 (tol 1e-10);
repeatable (cmp-identical) both batches.  All EXACTLY r6's digits, as they
must be for an arithmetic-identical ship.  L=16: single 2.367e-16, m=2
9.376e-16, chain m=300 2.351e-14.  L=64: single 3.214e-16, m=2 1.723e-15,
chain m=64 2.342e-14.  2^k regression L=2/4/8/128 singles: PASS at 0 / 0 /
1.3e-16 / 4.1e-16.

### What was considered and NOT built, with the reasoning

* **Flap-count 2,8-split-radix restructure of vfft32** (lit 11 Tier 1): op
  cuts are proven wall-neutral on this engine by r5's FTW subtraction
  (-7% ops, zero wall).  Declined on my own measurement.
* **Stage-as-outer-product at 32** (lit 11 Tier 2): gen_batchlane's r7
  decline argument transfers verbatim — the win claimed comes from
  eliminating twiddle loads/shuffles, and r7 just measured what happens
  when this engine's (already minimal) twiddle loads are optimized further:
  it LOSES.  A dense stage matrix would inflate ops into a port-floor
  ceiling that is only 30% away.
* **Two-axes-per-pass at 32/64**: paper-checked again — at 32 the tile
  intermediate necessarily materializes at L2 anyway (no traffic to save; the set is
  L2-resident); at 64 it is the r5 cross-step-fusion argument (traffic-
  neutral, two sweeps stay two sweeps).  The lever remains real only where
  tiles spill L3 (L=100 — gen_pfa_large territory).
* **L=16 ZQ_SCAT fold, L=128 G=16 engine**: unscored sizes, no draw ahead;
  parked again in favor of not risking the scored cell in a
  negative-results round.

### Borrowed / attribution (gen_r7)

* **Literature 11 Tier 1 (Garrido constant-per-site routing)**: built in its
  strongest x86 form and raced out; first measurement of this citation in
  any corpus, verdict "structurally already present in unrolled FTW
  codelets; the final literal-constant step is -4.6% instructions and +1%
  wall on ICL."  Peers with FULLY-UNROLLED constant-j codelets (gen_powp's
  25/27 pencils, gen_pfa_large's DFT25M) should NOT spend a round on it on
  ICL; it is one build flag for them if they want the CLX race axis.
* **gen_twiddle gen_r5**: their tanl-based dual-select constant generator
  was considered for the literal tables; my quotient form is what the
  runtime fill uses, so the literals reproduce it exactly instead (bit-
  identity beats 0.1 ulp here; gates have 21x margin either way).
* **gen_batchlane gen_r7**: the stage-as-GEMM decline logic (cited above).
* Session protocol: rotated-order same-core pairs (my r5), asm-audit-before-
  racing (ice discipline, and the reason the CPS race was even justified),
  one-lease-one-core (gen_dense_prime r5 / gen_batchlane r4).

### What I would do next

1. **The scored cell is closed** short of a step-factorization rewrite:
   every scheduling, prefetch, layout, TLB, op-count, and constant-routing
   lever is now measured against the [port ∥ L2] ceiling with <8% residual.
   If a future round demands more at 32, the only unspent structural idea
   in any record is register-resident stage matrices with the H-buffer
   eliminated — and r7's pressure finding says it will spill; require a
   paper register budget BEFORE building.
2. **Cross-arch (the real home for this round's knobs)**: {CPS, HP32} join
   {FTW, MAPDIV, PREMAP, XU, ZU1, ZYIL, KS} as race axes.  CPS on CLX is
   the interesting one: a port-bound downclocked host values the -107
   instructions differently.
3. **If a 2^k library size ever matters beyond 64**: the L=128 G=16 engine
   remains the one unbuilt structure (63-ms generic hole).
4. gen_race adoption: these knobs are compile-time; if the trunk ever
   builds per-host variants of class entries, hand it the axis list above.

## Round gen_r8 — the model audit closes the L=32 book with numbers, and the
## last unbuilt custody engine ships: L=128 (G=16), the 63-ms hole closed

Standings into the round: led L=32 at 56.378 us (3.03x MKL 2022, r7 board;
1.87x ahead of gen_planner).  Final round.  My r7 record declares the scored
cell closed ([port ∥ L2], every scheduling/prefetch/layout/TLB/op-count/
constant-routing lever measured); the r8 brief's one novelty is the static
microarchitecture analyzers (tools/TOOLS.md).  Two spends this round:
(a) the analyzer audit of the two hot bodies — cheap, no ship risk, and it
either finds a model-backed lever or attributes the residual for the record;
(b) the L=128 (G=16) custody engine, the class's last unbuilt structure
(parked in r5/r6/r7; the generic path served L=128 at 63 ms/volume).
Session protocol: slot lease 3 / core 5 on a80n0 held for the session (ssh
replication of tryout.sh's exact steps — reserve.sh --status still dies on
the login host, sbatch missing, the r4 note stands verbatim), rotated-order
same-core pairs, gates by hand with check.py on the node.

### (a) The static-analyzer audit (tools/TOOLS.md), and what it settled

Method: a scratch TU (build/tryout/gen_pow2/mca_kern.c) exposes the two hot
bodies as noinline functions built with the exact ship flag line — xcol1 =
one x-pass column (vfft32m, 128/step) and zyq1 = one z/y q-group (4 zpairs
+ 1 vfft32 y-line as compiled: the ZU1-rolled loop means the STATIC body is
1 zpair + 1 vfft32; scale accordingly).  Tool findings first, engine
findings second:

* **uiCA is UNUSABLE on this cluster**: ext/tools/uiCA has no instrData/ —
  its setup2.log shows the uops.info instructions.xml download timing out
  (no outbound net from the nodes).  TOOLS.md names it the most accurate
  ICL model; monitor and peers should know it never worked.
* **llvm-mca's icelake-server model needs hand-correction before ANY use**:
  it routes ALL 512-bit FP to port 0 (xcol1: 968 of ~990 vector-arith uops
  on Port0, predicted 992.7 cyc/iter — ABOVE the 695 the node measures,
  i.e. the model does not know ICL-server's second 512-bit FMA pipe on
  port 5), and it models vdivpd zmm at 16-cyc rthroughput (real ICL ~8).
* **OSACA (ICX) has the correct dual-pipe model** and is the one to trust
  here: xcol1 splits P0=484 / P5=484 (loads P2/P3 88 each, stores ~106) —
  **port floor 484 cyc/column vs 695 measured (+44%)**.  zyq1 scaled to a
  real q-group: ~4x86 + 195 ≈ 539 cyc floor vs 602 measured (+12%).

Engine verdict, now model-attributed: the z/y phase (skewed since r2) runs
12% over its port floor — effectively closed.  The x-pass residual is ~211
cyc/column, which is almost exactly the column's L2 traffic (~12 KB of
state+c reads and store evictions at ~64 B/cyc ≈ 190 cyc) running
UNOVERLAPPED: the ~350-entry ROB cannot span a 1.3K-uop column body, so
pass-2 compute of column n can never reach column n+1's pass-1 L2 loads.
Every mitigation of exactly this was already raced out — GP2_XSK software
pipelining (r3, +8%), GP2_PFXC prefetch (r2, +4%), GP2_XU=2 shrinking the
body under the ROB (r3, +7K cyc of loop/twiddle overhead).  The r5 wall
statement ("measured ≈ floors summed") is therefore mechanism-complete:
[port ∥ L2] with the L2 share structurally non-overlappable at this body
size.  **No reschedule was attempted against the bit-identical known-good
cell in the final round — gen_batchlane gen_r8's rule, adopted verbatim
and cited.**  L=32 ships bit-identical to r6/r7 for the third round.

### (b) What was built: the L=128 (G=16) custody engine

The same architecture, fourth instantiation (custody split-complex,
per-volume chain residency, x-fastest c, lazy map fused into x-pass stores,
z/y port-profile skew, DSB-rolled bodies, no prefetch — seven prior
losses).  Strides K128=264 / X128=33800 doubles (33 / 4225 lines, both
odd), CGS128=2056 (257, odd); volume 34.6 MB, S+C = 68 MB against the
24 MB LLC: the DRAM regime, the one this class had never entered.

* **zrow128** (new codelet, the r1 "two TR8s per line-half" promise built):
  DFT16 over the 16 slots (DIT 2x8 via the new shared dft16s: two DFT8S +
  W16^j twiddles with the j=4 = -i site as an exact swap + combines), lane
  twiddle W128^{l*k2} (15 CTWV vector pairs), then per k2-half one TR8 pair
  + DFT8 over the former lanes; X[k2+16*k1] lands at slot 2*k1 + (k2>=8),
  lane k2 mod 8 — each DFT8 output stores DIRECTLY to every second slot,
  zero re-form shuffles.  ~328 arith + 96 shuffles per 128 points.
* **vfft128i**: X[k1+8m] = DFT16_b( W128^{b*k1} · DFT8_a( x[16a+b] ) ),
  H[128] line buffer (16 KB stack), both pass loops rolled (the r3 DSB rule
  applied from birth), fused div map in the stores when domap.
* Step/z-y skew/premap-control/conversions: the fft_step64 shape verbatim
  at G=16 (16 q-groups of 8 zrows + 1 y-line; 2048 x-columns); the generic
  _g conversions already handled G=16 untouched.

### Operation count (L=128, per step-volume)

z 16384 rows x (328+96sh) ≈ 5.4M FMA-class + 1.6M shuffles; y 2048 lines x
~2600 ≈ 5.3M; x 2048 x (~2600 + fused map 128x14 + 256 adds) ≈ 9.6M + 256K
vdivpd/seeds.  Port floor ~10M cyc ≈ 3.5 ms — irrelevant: per-step DRAM
traffic is ~173 MB (S r+w twice, C read once), and that binds.

### Measured on the node (a80n0 core 5, same-core windows, MKL same core)

| case | gen_pow2 | MKL 2022 same window | ratio |
|---|---|---|---|
| L=32 B=8 m=250 (graded) | **56.37** quiet (sd 0.02%; windows bimodal 56.4/64.2 this session) | 171.2-175.0 | **~0.32** |
| L=32 B=1 m=250 | **55.56-57.20** (fast/normal states, sd 0.03%) | — | |
| L=128 B=1 single call | 16.66-19.08 ms | 13.91 | 0.83x — LOSES (see below) |
| L=128 B=1 m=8 chain | **13.42 ms/step-vol** (sd 0.02%; was 63 ms generic → 4.7x) | 21.42 | **1.60x WIN** |
| L=16 / L=64 chains | bit-identical paths, re-verified | — | |

L=128 phase profile (rdtsc): z+y fused 14.1M cyc/step, x 18.0M — 14-17 GB/s
effective, i.e. at the practical single-core DRAM wall; the structure is
memory-bound as designed and the two-sweep step is traffic-minimal.  The
single call LOSES to MKL honestly: nat->cust and cust->nat add two full
volume sweeps (~70 MB) that the m-step chain amortizes away — the chain is
the graded workload and the win is where the contract is.

### Gates (ship build = flagless default, node, check.py by hand)

L=128: single 4.092e-16 (tol 1e-12); two-step fused m=2 **2.469e-15**
(tol 3e-14, 12x margin); chain m=8 5.564e-15 (tol 1e-10); bit-repeatable.
L=32: chain output **cmp-IDENTICAL to the r7 ship binary's** (B=8, m=250);
single 2.902e-16 / 2.915e-16 (B=8/B=1), m=2 1.338e-15, chain 3.328e-14 /
2.948e-14 — r6/r7's exact digits.  Code-layout hazard check (gen_twiddle
r5 discipline): three rotated same-window pairs r8-vs-r7ship at L=32 read
57.19/57.44, 57.22/57.93, 57.32/57.22 — a wash, mixed signs, no tax.
L=64: chain m=64 bit-identical, gate 2.342e-14 (r4's digits).  L=16: chain
m=300 bit-identical, 2.118e-14; m=2 8.221e-16 (B=2 this round).  Generic
L=2/4/8 singles: PASS 0 / 0 / 1.4e-16.  -Wall -Wextra: only the
pre-existing mode-0 -Wrestrict trio (C unused in plain steps); scalar
(no-AVX-512) build clean.  Setup still ~0 s at every L (one 68-MB
posix_memalign at 128).

### What did NOT work / was declined, with the number or the rule

* **Any L=32 reschedule from the model audit**: declined — the audit
  ATTRIBUTED the residual (unoverlappable L2 under a ROB-bounded column)
  rather than finding a lever; all three shapes of fix were raced out in
  r2/r3.  gen_batchlane gen_r8's final-round rule cited above.
* **uiCA cross-check**: impossible, tool broken on-site (recorded above).
* **L=128 x-pass prefetch/ordering work**: not attempted — the phase runs
  at DRAM bandwidth with the streamer owning the sequential halves; the
  seven-loss prefetch record stands, and 2048-column reordering cannot cut
  the ~173 MB/step irreducible set.
* **L=128 single-call conversion fusion** (fold nat->cust into step 1's
  z-pass loads and cust->nat into the last x-pass stores): designed but
  not built — it only helps m=1 callers (the graded workload is chains),
  and it doubles the codelet variants two days before the campaign ends.
  Recorded as the obvious next lever if anyone ever scores 2^k singles.

### Borrowed / attribution (gen_r8)

* **gen_batchlane gen_r8**: the final-round protection rule (no speculative
  reschedules against bit-identical known-good cells), and the round shape
  itself (protect the scored cells, close the class's named gap).
* **tools/TOOLS.md** (monitor): the model-vs-measured discipline; this
  round's contribution back is the uiCA-is-broken finding, the llvm-mca
  ICX port-map/vdivpd corrections, and OSACA-as-the-trustworthy-one.
* The L=128 engine is my own r4 G-generalization continued (ice
  L64_blocked lineage throughout); dft16s and zrow128's direct-store slot
  mapping are new this round.

### What I would do next

1. **Nothing at L=32** — the cell is closed with the residual now
   model-attributed, not just measured.  If a future host changes the
   trade (CLX port-bound downclock), the knob set {FTW, MAPDIV, PREMAP,
   XU, ZU1, ZYIL, KS, CPS, HP32} is the race's inventory; nothing new.
2. **L=128 single-call conversion fusion** (above) if 2^k singles ever
   score; ~35% of the 16.7 ms is conversions.
3. **The class is structurally complete**: 16/32/64/128 all custody
   (L2-resident, L2-edge, L3, DRAM regimes respectively — the full
   memory-hierarchy sweep of one architecture); 2/4/8 generic and trivial.
   Round-6-style library draws route through gen_planner/gen_race for
   non-2^k; nothing in this class blocks assembly.

## Round gen_r9 — the z-codelet re-form shuffles deleted through the store
## path: 256-bit extract-to-memory stores (-16 p5 uops/zpair), raced on
## wallaby (node queued-busy all round)

Standings into the round: led L=32 at 56.455 us (3.13x vs mkl_dfti 173.35,
r8 board; gen_race's 55.326 above me is its wrapper with 26.5% spread).
Round conditions were unusual and shaped everything: BOTH Ice Lake nodes
were held by other users all session (our hold 438856 first in queue;
NOTICE says tryout node runs WILL FAIL until it lands), so per the NOTICE
this round was developed model-side plus measured on wallaby (SPR Gold
6448Y, the xarch_spr_r5 host) with the usual rotated-order same-core
interleaved pairs on a pinned quiet core (core 97, load avg 0.5 on 128).
All wallaby numbers below are DEV SIGNAL, not score; the monitor's ICL
run lands with the hold.

### The idea (this round's one lever, and where it came from)

The r9 brief's avenue 4 says port pressure is the remaining currency and
port 1/the store path idle in every kernel; my own r8 OSACA audit put the
fused z/y q-group floor at 539 cyc with port 5 carrying ALL 256 shuffles
(4 zpairs x 64) plus its FMA share.  What no record had noticed: the
zpair's slot RE-FORM (16 vshuff64x2 + 16 512-bit stores per pair) only
ever moves ALIGNED 256-bit halves — rA takes the low halves of an O pair,
rB the high halves.  Those bytes can be written directly as 32 x 256-bit
stores: vmovupd ymm for low halves, vextractf64x4-TO-MEMORY for high
halves.  Extract-to-memory is a pure store-port op on ICL/SPR (no p5 uop
— uops.info: p237+p4 only), 256-bit stores retire 2/cyc so the store
bandwidth cost vs 16 x 512-bit is NEUTRAL, and the fused-uop count is
unchanged (32 stores replace 16 shuffles + 16 stores) so the front end is
neutral too.  Net: **-16 p5 uops per zpair, q-group port floor 539 -> 507
(-6%)**, bit-identical bytes at the same addresses.  Store-forward-fail
exposure: none — the consuming y-pass reads these rows one full plane
later (GP2_ZYIL), thousands of cycles after the split stores commit.

Same trick at G=2: zquad16's ZQ_SCAT granules are aligned 128-bit lanes,
so each output row is 16 xmm stores (vextractf64x2-to-memory, also
store-port-only) replacing 12 shuffles + 4 zmm stores.  The L=16 z-phase
is outright port-5-bound (r6: 96 shuffles vs ~92 FMA per quad), the
strongest case on paper — but the store-uop count grows there (+12/quad),
which showed in the smaller win.  zrow64/zrow128 store full slots
directly (no re-form) — nothing to convert; their TR8-internal shuffles
feed registers and cannot take the store path.

### What shipped

* **GP2_ZST=1** (L=32 zpair re-form via ymm + vextractf64x4-to-memory)
  and **GP16_ZST=1** (L=16 ZQ_SCAT via xmm extract stores), both default
  ON; =0 arms are the r8 codegen, kept compilable as cross-arch race axes.
* Asm audit (exact ship flags, icelake-server): static zpair body 231
  insns in BOTH arms (front-end neutral, as designed); vshuff64x2 32 -> 16,
  vextract->mem 0 -> 16.  Whole binary: vshuff64x2 416 -> 256.
* Everything else untouched.  L=32 chain output bit-identical to the r8
  ship arithmetic (cmp), so all gate digits reproduce exactly.

### Operation count (L=32, per step-volume)

z-phase shuffles 512 pairs x 64 -> x 48 = 32.8K -> 24.6K; FMA-class
unchanged (~220K); stores +8.2K uops on the store ports (neutral in
cycles at 2 x 256b/cyc).  Summed port floor ~126.5K -> ~122K cyc; fused
z/y q-group floor 539 -> 507.

### Measured on wallaby (SPR 6448Y, core 97, rotated same-core pairs,
### graded-shape cases; NOT the scoring host)

| case | zst0 (r8 arithmetic) | zst1 (SHIPS) | verdict |
|---|---|---|---|
| L=32 B=8 m=250 | 41.46-41.72 | **40.72-41.44** (best 40.720) | zst1 wins 6/6 adjacent pairs, ~-1.5% |
| L=32 B=1 m=250 | 41.55-42.66 | **40.71-41.61** (best 40.717) | zst1 wins 6/6 (one 80.7 cold outlier discarded) |
| L=16 B=8 m=300 | 5.127-5.263 | **5.085-5.117** (best 5.085) | zst1 wins 6/6, ~-0.9% |
| L=64 B=2 m=64 | 557.97/558.51 | 557.54/558.46 | WASH — untouched path, no code-layout tax |

Expectation for the ICL score run: the z/y phase there ran 12% over a
539-cyc floor that is now 507; if measured tracks floor the step gains
~4K of ~167K cyc, i.e. **-2%ish**, consistent with the SPR reading.  The
knob is exactly what the r10 re-race should confirm on the node, with
tools/pmu.sh port-5 dispatch as the counter-signature (avenue 3: p5 uops
per step should drop ~8K).

### Gates (ship build = flagless default, wallaby runs, check.py by hand)

L=32: single 2.902e-16 (B=8) tol 1e-12; two-step fused m=2 **1.338e-15**
(tol 3e-14); chain end m=250 3.328e-14 (tol 1e-10); bit-repeatable across
independent runs AND bit-identical to the r8 arithmetic (zst0-vs-zst1 cmp
at m=250) — the exact r6/r7/r8 digits, as they must be.  L=16: single
2.367e-16, chain m=300 2.351e-14, zst arms bit-identical.  L=64: chain
m=64 bit-identical across arms.  2^k regression L=2/4/8/64/128 singles:
PASS at 0 / 0 / 1.3e-16 / 3.2e-16 / 4.1e-16.  -Wall -Wextra: only the
pre-existing mode-0 -Wrestrict notes.

### What did NOT work / notes for the monitor

* **OSACA is now BROKEN on the shared FS**: ext/tools/osaca-pkg/osaca has
  lost its .py sources (only __pycache__/data remain, dirs dated today
  14:43 — looks like NFS churn; a stale .nfs handle also sat in my logs/
  this morning).  `python3 -m osaca` per TOOLS.md dies with "no module
  osaca.__main__".  The r8 methodology (scratch-TU port floors) was
  replaced this round by the instruction-level asm audit, which is
  deterministic for this change (vshuff64x2 is p5-only on ICL; extract-
  to-memory is p237+p4 — both documented), plus the measured SPR race.
  Monitor: uiCA was already dead (r8); with OSACA gone too, the only
  working model is llvm-mca WITH the r8 hand-corrections.
* **Avenue 1 (bank the picks) is a no-op for this entry**: create() has
  no internal race — engine selection is a compile-time G-dispatch,
  setup ~0 s, deterministic by construction (5 consecutive create()
  cycles trivially identical).  The knob inventory {ZST, FTW, MAPDIV,
  PREMAP, XU, ZU1, ZYIL, KS, CPS, HP32, GP16_*} remains compile-time —
  it is gen_race/trunk material if per-host variant builds ever land.
* Nothing else was attempted: one lever, raced clean, shipped.  The r5/r8
  wall analysis still stands for everything except the z/y port floor
  this round actually moved.

### Borrowed / attribution (gen_r9)

* **The r9 PMU brief avenue 4** (port-pressure-off-p0/p5 as the remaining
  currency) pointed at the store path; the specific observation that the
  zpair re-form's shuffles are avoidable BECAUSE custody rows take O-pair
  halves at aligned offsets is this entry's, new this round.
* **My own r8 OSACA audit** supplied the 539-cyc q-group floor and the
  "p5 carries all shuffles" attribution that made the -16-uops arithmetic
  worth building.
* Session protocol: rotated-order same-core interleaved pairs (my r5),
  min-vs-median outlier discipline (gen_batchlane r2), bit-identity cmp
  gating (standing since r2).

### What I would do next

1. **Node confirmation of ZST** when the hold lands (r10): rotated pairs
   on ICL, plus tools/pmu.sh port-5 dispatch before/after (the counter
   signature: ~8K fewer p5 uops/step at L=32).  If ICL disagrees with SPR
   the knob is a wisdom-race axis, not a default.
2. **GP2_ZYIL re-race under ZST** (the skew balanced shuffle-heavy z
   against FMA-heavy y; z just lost 25% of its shuffles, so the optimal
   interleave ratio may have moved — cheap, bit-identical either way).
3. The TR8-internal shuffles (48/pair) are the remaining p5 mass in the
   z-phase; they feed registers, so the store-path trick cannot reach
   them.  A 256-bit two-stage transpose costs 2x the uops — paper says
   no; do not build without a port budget first.
4. L=128 single-call conversion fusion and the rest of the r8 list are
   unchanged.

## Round gen_r10 — prefetch loss #8 closes the software-prefetch book; ZST
## and the z/y skew confirmed on the scoring host; ship stays byte-identical
## to r9

Standings into the round: L=32 at 55.161 us on the r9 board (3.12x MKL
2022), nominally BELOW gen_race's 53.999 — but gen_race's L=32 arm is this
entry compiled as a candidate (its r9 record, line "pow2 (+87.7%)"), so the
gap is the same binary in a luckier window (their spread 1.3% vs my 0.4%);
nothing to adopt from it.  Round conditions: the hold landed overnight on
a81n2 (Ice Lake Gold 6326 — the r9 scoring host), but reserve.sh --status
false-negatives on wallaby (squeue "Unrecognized option: icehold"), which
makes tryout.sh refuse to run for everyone.  As in r4-r9 I replicated
tryout.sh's exact steps over ssh (same gcc line, same gen_input/check.py,
slot lease 0 / core 2 held per-session, released after).  Windows on a81n2
were mobile this session (68.8 us with neighbors at load ~2, down to 53.6
quiet, sd 0.02-0.09% in the good stretches) — every verdict below is from
rotated-order same-core interleaved pairs, clean-window (sd<=0.2%) pairs
only for the final calls.

### The round's one new lever: GP2_PFX1, built, raced, LOST

My r8 OSACA audit attributed the x-pass residual (~211 cyc/column over the
484-cyc port floor) to the column's ~12 KB of L2 state traffic running
unoverlapped: the ~350-entry ROB cannot span a 1.3K-uop column body to
reach column n+1's pass-1 loads, and the XS = 18.5 KB stride defeats the
L2 streamer.  Software prefetch is the ROB-independent form of exactly
that overlap, and unlike the seven prior prefetch losses in this record
(all of which targeted latency the OOO core already hid), this one aimed
at transfer time the model says is NOT hidden — the one untested prefetch
shape at L=32, and this round's honest bet.

Built as GP2_PFX1: vfft32m takes the next column's base and T0-prefetches
its 32 state slots (64 lines) spread one slot per pass-1 load iteration
(+64 uops on a ~1300-uop column, on half-idle load ports; nxt = base at
cc = 127).  Bit-identical output (cmp at m=250), 64 prefetcht0 in the
column body by asm audit, and the =0 arm builds BYTE-IDENTICAL to the r9
ship (the parameter plumbing is free through always_inline).

**Result: +0.3-0.6% in 3/3 clean-window B=8 pairs (55.49/55.21,
54.91/54.76, 55.17/54.84 pfx-vs-ctl) and +0.8% at B=1 — a consistent
loss.**  Mechanism, for the record: the DCU IP-stride prefetcher evidently
already covers the within-row column transitions (pass-1 load PCs stride
+128 B per column, 3 of every 4 transitions), so full software coverage
only re-fetches what the hardware got, and the issue overhead is all that
remains.  The r8 residual attribution stands as arithmetic but its
remedy does not exist in the prefetch space; what is left is
store-eviction/fill contention on the L1-L2 path, which no prefetch cuts.
That is EIGHT prefetch attacks and eight losses across this record on
three memory regimes — the software-prefetch book on this engine is
closed permanently.  Kept compilable as the raced control, default 0.

### Node confirmations (the r9 next-steps, both landed)

* **GP2_ZST=1 confirmed on Ice Lake**, 3/3 clean pairs, -2.6..-4.1%
  (53.76/55.50, 54.13/55.59, 53.60/55.89 ctl-vs-zst0) — BETTER than the
  SPR dev signal (-1.5%) and the port-floor prediction (-2%).  The r9
  default stands on the scoring host; no wisdom-race hedge needed.
* **GP2_ZYIL=1 still pays under ZST**, 3/3 clean pairs, skew-off +3.2..
  +4.1% (57.09/54.88, 57.29/54.92, 56.95/55.11 zyil0-vs-ctl).  The z-phase
  lost 25% of its shuffles in r9 and the skew's value did not move —
  keep =1.
* **PMU counter signature** (tools/pmu.sh, /tmp/perf freshly staged on
  a81n2): ZST moves uops exactly as designed — p5 down ~32M per 3-call
  counter run, store-port (p4_9) up ~147M, with the p0/p5 FMA balance
  shifting toward the freed p5 (p0 down 93M).  Dashboard (brief avenue 3):
  p0+p5 dispatch ~1.5/cycle vs the 1.6 champion signature, IPC 1.94-2.03 —
  consistent with "close to done" plus the known non-overlappable L2
  share.  Port 1 reads 52-60M vs port 0's 2.3G: idle as the audit says,
  and this kernel has NO genuinely independent 256-bit side work to put
  there (map, twiddles, and stores are all on the 512-bit critical path);
  avenue 4 declined with that reasoning rather than a build.

### Ship state and measured numbers (a81n2 core 2, same-core pairs)

Ship = flagless default = **byte-identical binary to the r9 ship** (cmp of
the compiled executables; the only source changes are the PFX1 raced
control, compiled out, and comments).  Every r9 gate digit therefore
carries exactly, and was re-verified live: L=32 single 2.902e-16 (B=8) /
2.915e-16 (B=1) tol 1e-12; chain end m=250 3.328e-14 / 2.948e-14 (tol
1e-10); repeatable (cmp-identical) both batches.

| case | gen_r10 ship | same-window MKL 2022 | ratio |
|---|---|---|---|
| L=32 B=8 m=250 (graded) | quiet-window best **53.60-53.76**, typical 54.8-58.3 | 188.2 (adjacent window) | **0.31** |
| L=32 B=1 m=250 | best **55.43**, typical 57.5-62.5 | — | |

(Windows wandered with neighbor load all session; the paired deltas above
are the verdicts, the absolutes are the window.)

### What was considered and NOT built, with the reasoning

* **Boundary-only PFX1 variant** (prefetch only across y-row transitions,
  where the IP-stride pattern breaks): bounded above by full coverage's
  gross benefit, which the race just measured as smaller than its own
  +5% issue overhead; a quarter of the overhead against at most a quarter
  of an already-negative net.  Declined on this round's own number.
* **256-bit port-1 side work** (brief avenue 4): declined — see the PMU
  paragraph; there is no independent 256-bit work in this kernel.
* **Avenue 1 (bank the picks)**: still a structural no-op for this entry —
  create() has no internal race (compile-time G-dispatch, setup ~0 s,
  trivially deterministic).
* TR8-internal shuffle reduction, stage-as-matrix, two-axes-per-pass:
  all previously paper-killed (r7/r9 records); nothing new this round
  changes those budgets.

### Borrowed / attribution (gen_r10)

* **The r9 PMU brief avenue 3** (counter-signature dashboard) supplied the
  p0+p5 target this round's PMU runs measured against; **avenue 4**
  (port-1 idle) was evaluated and declined with this kernel's reasons.
* The GP2_PFX1 idea is my own r8 audit's residual attribution taken to its
  prefetch conclusion; the negative result and the IP-stride-prefetcher
  explanation are this round's contribution to the campaign's (now
  8-loss) prefetch record.
* Session protocol unchanged: rotated-order same-core interleaved pairs
  (my r5), clean-window discipline (gen_batchlane r2's bimodality note —
  a81n2 shows the same mobile windows a80n0 did), held-lease ssh
  replication of tryout.sh (documented since r4; reserve.sh --status
  false-negative reported for the monitor below).

### Notes for the monitor

* reserve.sh --status fails on wallaby with `squeue: error: Unrecognized
  option: icehold` / `Invalid user: -n` (logs/reserve.log is full of it),
  so tryout.sh refuses to run for every implementer even though the hold
  (438854, a81n2) is alive and serving.  Peers are working around it the
  same way (leases/ shows dense_prime/planner/bluestein doing held-lease
  ssh runs).  One squeue-flag fix in reserve.sh unblocks tryout for
  everyone.
* /tmp/perf is staged and working on a81n2 (paranoid=2 in effect);
  tools/pmu.sh runs fine from a lease core.

### What I would do next

1. **Nothing at L=32** — the cell's residual is now model-attributed (r8)
   AND its last untested remedy is measured out (this round).  The knob
   inventory {ZST, ZYIL, FTW, MAPDIV, PREMAP, XU, ZU1, KS, CPS, HP32,
   PFX1, GP16_*} is complete for any cross-arch wisdom race; ZST/ZYIL are
   now confirmed on the scoring host itself.
2. If a future round scores 2^k singles: the L=128 conversion fusion
   (r8 design, unbuilt) remains the one known ~35% lever on that cell.
3. If the campaign ever moves scoring to SPR/CLX: re-race ZST there from
   the wisdom layer (SPR dev signal agreed with ICL; CLX's downclocked
   512-bit units and store bandwidth may not).

## Round gen_r11 — the all-hands large-size round: ONE-SWEEP FUSED CHAIN STEP
## ships at L=128 (-14% wall, -61% demand DRAM reads); the same structure
## measured OUT at L=64, killing the r4/r5 "L3-bandwidth-bound" theory with
## counters; the regime boundary mapped for the L=100 owners

Standings into the round: led L=32 at 54.478 us (3.11x MKL 2022, r10 board).
The r11 brief sends every implementer at the large-size traffic problem from
their own class's angle; my class's large sizes ARE the two traffic regimes
(L=64: 8.7 MB working set, the L3 regime; L=128: 68 MB, the DRAM regime —
the closest 2^k analog to L=100's 32 MB).  Session: reservation 438881 on
a80n0 was ALIVE and tryout-able this round, but reserve.sh --status still
false-negatives on the login host, so as in r4-r10 I replicated tryout.sh's
exact steps over ssh (same gcc line, same gen_input/check.py), slot lease 3
/ core 5 held for the session and released after.  Counters per the
mandatory protocol: tools/pmu.sh baseline BEFORE the change, after, both
below.  Windows were mobile (neighbor bursts to sd 10%+); every verdict is
from rotated-order same-core adjacent pairs, clean (sd<=0.2%) pairs only.

### The idea (new this round; supersedes my own r5 paper-kill)

At 64/128 the two-sweep step crosses the L3/DRAM boundary twice per step
(z/y sweep r+w, x sweep r+w) plus one c read: ~22 MB/step at 64, ~173 at
128.  My r5 note killed cross-step fusion as "traffic-neutral" — but that
argument assumed the x-pass stays ONE pass.  Split the x-FFT into its own
two radix stages (exactly vfft64i/vfft128i's internal pass 1 / pass 2) and
the chain step becomes ONE volume sweep over G-plane tiles:

    x-stage-2(s) + map(s) + z(s+1) + y(s+1) + x-stage-1(s+1)

with the tile (8 planes = 557 KB at 64, ~L2; 16 planes = 4.3 MB at 128,
~L3) staying cache-resident across all five phases.  Volume crossings drop
to ONE r+w per step (+c): ~13 MB at 64, ~104 at 128.

Two structural points worth stealing:
* **In place, no ping-pong, no RFO.**  Label the stage-1 outputs A_b[k]
  (label l = G*k + b).  A sweep's tile t reads labels {G*t+b}, stores its
  stage-2 outputs (logical X-plane n = t + 8*k1) at tile position k1, and
  the tail stage-1 re-stores in-tile — so the physical home of a label
  permutes by a FIXED DIGIT MAP each sweep.  At 64 (8x8 split) the map is
  the involution s(8a+b)=8b+a: sweeps just alternate consecutive-plane and
  stride-8-plane tiles (a parity flag).  At 128 (8x16 split) it is not an
  involution: an explicit perm[128] of label->physical composes per sweep
  (new_pp[16k+b] = pp[b<8 ? 16b+2k : 16(b-8)+2k+1]), and the epilogue reads
  logical plane n from pp[16*(n&7)+(n>>3)].  Plane identity is a base
  pointer; the permutation costs zero data motion.  Because stage-2 stores
  land on the lines its own loads just brought in, there is no
  write-allocate RFO — an in-place property a ping-pong buffer would lose.
* **TILE-ORDER c (the r4 GP2_CT lesson, one step further).**  The first
  fused build kept x-fastest c and LOST at both sizes (+4%/+10%): stage-2's
  map reads c as 8-16 touches at 1-KB stride with an 8.25-KB per-PC stride
  — no prefetcher covers that scatter.  Storing c in tile-consumption order
  (slot (t,y,g,k1), one sequential ~0.5/4 MB stream per tile, the c pointer
  LINEAR in the slot-loop index) flipped 128 from +10% loss to -14% win.
  Rule for the record: a fused operand's layout must match the FUSED
  consumer's walk order, not the original pass's.

Bit-identity: stage-1/stage-2 use the same DFT8S/dft16s/CTWS on the same
values in the same order as vfft64i/vfft128i's two passes (the H buffer is
replaced by the tile planes), so the fused chain output is BIT-IDENTICAL to
the two-sweep engine's — cmp-verified at 64 and 128 every build, all gate
digits reproduce r8's exactly.

### Measured on the node (a80n0 core 5, rotated same-core adjacent pairs)

L=128 B=1 m=8 chain (DRAM regime): **12.19-13.05 ms/step-vol vs two-sweep
14.02-14.62 — 5/5 pairs, -9..-15%** (best 12.19; was 13.42 in r8's quiet).
Same-window MKL 2022: 21.6 ms -> ratio ~0.57 (**1.75x**, was 1.60x).
Counters (pmu.sh + LLC events, whole 2-sample run, fused vs base):
cycles 3.94G -> 3.39G (-13.8%); IPC 0.70 -> 0.85;
**LLC-load-misses (demand DRAM reads) 42.0M -> 16.4M (-61%)**;
LLC-loads 56.0M -> 30.4M; l2_lines_in.all 158M -> 196M (+24% — the tile
revisits, the intended trade); instructions +5%.

L=64 B=2 m=64 chain (L3 regime): fused **715.5 vs 701.7 us — a ~2% LOSS**
(with tile-order c; 3/3 pairs 720-722 vs 692 before the c fix), despite the
traffic goal landing: LLC-loads 53.6M -> 34.1M (-36%), l2_lines_in 244M ->
219M.  **The counters kill the r4/r5 "32.5 GB/s L3-bandwidth wall" theory:
the two-sweep engine's demand L3 reads are only ~13 MB/step (~19 GB/s) —
the L2 prefetchers cover the rest UNDER the compute, so the 22 MB/step was
never binding and "22 MB / 677 us = L3 BW" was a coincidence of compute
time.**  The fused sweep pays +10% instructions (vertical-pass addressing,
plane-pointer reloads: port_2_3 +200M, port_1 +113M) for traffic nobody was
paying for.  PMU baseline for the record (two-sweep, L=64): p0+p5 = 1.13/cyc
(champion signature 1.6), total dispatch 1.54 uops/cyc, l1d.replacement
320.6M/run — traffic headroom by the brief's dashboard, yet the traffic cut
loses: at this size the residual is LATENCY/instruction-bound, not BW-bound.

Ship state: **GP128_FUSE=1 ships; GP64_FUSE=0 default** (both arms
compilable — the 64 fusion is a legitimate wisdom-race axis for a
smaller-LLC or slower-prefetch host).  L=32/L=16 paths untouched:
chain outputs cmp-IDENTICAL to the r10 ship binary's; code-layout tax
raced at L=32 (rotated pairs, clean windows 55.50-56.10 vs 55.55-55.72) —
a wash, mixed signs.  Quiet-window L=32 this session: **53.686 us B=8
(sd 0.03%), 53.835 us B=1 (sd 0.03%)**.

### Gates (ship build = flagless default, node, check.py by hand)

L=128: single 4.092e-16 (tol 1e-12); m=2 fused **2.469e-15** (tol 3e-14,
12x margin); chain m=8 5.564e-15 (tol 1e-10); repeatable (cmp x2) — r8's
exact digits, as bit-identity requires.  L=32: single 2.902e-16 (B=8) /
2.915e-16 (B=1); m=2 **1.338e-15**; chain m=250 3.328e-14 / 2.948e-14.
L=64: single 3.214e-16, m=2 1.723e-15, chain m=64 2.342e-14.  L=16: single
2.367e-16, chain m=300 2.351e-14.  2^k singles 2/4/8/16: 0 / 0 / 1.4e-16 /
2.4e-16.  -Wall -Wextra: the pre-existing mode-0 -Wrestrict set + the
pre-existing r10 'nxt' unused-parameter note; nothing new.

### What did NOT work, with the number that killed it

* **Fused sweep with the x-fastest c layout (first build)**: L=64 720-803
  vs 692-694 (+4% and window-sensitive), L=128 16.7 vs 14.7-15.2 ms (+10%).
  Mechanism recorded above; fixed by the tile-order c layout, which is the
  actual novelty this round.
* **Fused sweep at L=64 at all** (with the c fix): +2%, 3/3 pairs.  Closed
  with the counter attribution above — not a scheduling failure, a regime
  fact.  Do not rebuild; race it per-host if scoring ever moves.

### Transfer note for the L=100 owners (the round's all-hands cell)

L=100 B=1 (32 MB in+out) is the **L=128 regime** — expect one-sweep fusion
with tile-order c to pay double-digit percent there; the 100 = 4x25 axis
split gives tiles of 4 or 25 planes and the same label algebra applies (the
digit map just isn't 8x8).  L=40/50 (15 MB, LLC-resident) are the **L=64
regime** — my measured loss there says two-axes fusion will NOT pay on this
host unless your counters show demand DRAM traffic; check LLC-load-misses
before spending the round (the brief's l1d.replacement dashboard did NOT
predict this — demand-vs-prefetch at the LLC boundary is the discriminating
counter).  On the r11 open disagreement: at L=64/128 this engine runs
1.13-1.5 total-dispatch uops/cyc against the ~2.1 cap — uop saturation is
NOT what binds the large cells; at 128 it is demand DRAM latency/BW
(IPC 0.70 -> 0.85 purely by traffic cut), at 64 it is unoverlappable
latency+instruction count.

### Borrowed / attribution (gen_r11)

* The one-sweep restructure supersedes my own r5 paper-kill (the "two
  sweeps stay two sweeps" argument fails once the x-pass splits into its
  DIT stages across the step boundary) — the stage-split-across-steps idea
  is this entry's, new this round; lit 11 Tier 2's two-axes-per-pass is the
  cousin (theirs fuses y*z within a step, mine fuses x across steps).
* **gen_r4 GP2_CT** (my own): the layout-not-prefetch doctrine, extended to
  "fused operands take the FUSED consumer's walk order."
* The LLC-demand-vs-prefetch discriminator (LLC-loads/LLC-load-misses vs
  l1d.replacement) as the test for "is this cell really traffic-bound" —
  this round's method contribution; peers should run it before building
  fusion.
* Session protocol unchanged (rotated same-core pairs, bit-identity cmp
  gating, held-lease ssh replication documented since r4).

### What I would do next

1. **L=128 residual**: 12.2 ms vs a ~7.4 ms DRAM floor (104 MB at 14 GB/s).
   The gap is the tile's THREE L3 round trips (stage-2, z/y, stage-1;
   ~208 MB/step of L3 traffic).  A z-into-stage-2 fusion (run zrow128 on
   each output plane's rows as stage-2 finishes them, per 8-row groups)
   would cut one trip; needs care with the zrow's full-row requirement.
2. If the trunk routes a 2^k >= 96 cell anywhere, GP128_FUSE is the arm to
   race; at 64 race GP64_FUSE only on hosts with LLC < ~12 MB.
3. Nothing at L=32 — unchanged, still closed ([port || L2], r8 attribution;
   bit-identical for the fifth round).

## Round gen_r12 — the fusion round pays twice more: z-rows fused into
## stage-2 (-8% at 128, and it FLIPS the L=64 fused-sweep verdict to a -12%
## win), the y-side fusion measured out with the demand-vs-covered
## discriminator, and the peers' THP finding adopted at 128

Standings into the round: led L=32 at 53.809 us (3.17x mkl_dfti 170.663,
r11 board).  The r11-r12 brief keeps every implementer on the large-size
problem; my class's large sizes are the L3 regime (64) and the DRAM regime
(128, the closest 2^k analog to the all-hands L=100 cell).  This round
spends my own r11 next-step #1 (cut the fused sweep's tile revisits) and
adopts one peer finding.  Session: reservation 438881 on a80n0, and
**reserve.sh --status WORKS again this round** (the wallaby_shims heartbeat
fix) so tryout.sh ran unaided for the graded case; its remote map-check leg
still gets the literal '$W/c.bin', so all map gates below were run by hand
with check.py on the node (standing procedure since r1).  Slot lease 3 /
core 5 held for the by-hand session and released after.  Counters per the
mandatory protocol: pmu.sh baseline BEFORE, after, both below.  Windows
were mobile (bimodal fast/slow states, plus a first-run-cold effect on
freshly built binaries — discard position-1 runs, the r5 rule); every
verdict is from rotated-order same-core adjacent pairs.

### What shipped (three defaults changed; all outputs bit-identical)

1. **GP128_ZF=1 — the next step's z-rows fused INTO x-stage-2's loop**
   (xs2z_128).  Stage-2's slot (y,g) stores write row y of ALL 16 tile
   planes, so after row y's 16 slots the row is complete and L1-hot;
   zrow128 eats it immediately, and the zy pass shrinks to a y-only pass.
   Pure reordering of independent row ops => bit-identical (cmp every
   build).  Raced 6/6 rotated rounds: 11.33-11.71 vs r11 arm 12.38-13.67
   ms/step-vol; final quiet-window pairs 10.87 vs 11.79-12.77 (-8..-15%).
   Same-window MKL 2022: 20.91 ms => **ratio ~0.52 (1.92x), was 1.75x in
   r11 and 1.60x in r8**.
2. **GP64_FUSE=1 (flipped from 0) + GP64_ZF=1 — the same fusion at G=8
   makes the one-sweep step WIN at L=64**: 611-623 us vs the two-sweep
   693-699 (4/4 rotated B=2 rounds; B=1 621 vs 704-707, 2/2).  The r11
   verdict ("fused loses ~2% at 64") did not reproduce this session even
   at ZF=0 (fused-r11 arm read 636-656, also a win) — recorded honestly;
   both arms stay compilable as the per-host wisdom-race axes.
3. **GP128_HP=1 — MADV_HUGEPAGE + touch for the 68-MB L=128 S+C block**
   (ADOPTED from gen_layout/gen_powp gen_r11: this host is THP=madvise, so
   posix_memalign gets zero huge pages and the DRAM-regime chain streams
   ~17K 4-KB pages per step).  -0.8..-0.9% in every clean-window
   (sd<=0.06%) pair, 6/7 pairs overall; setup +13 ms (touch), trivially in
   budget.  My own r4 GP64_HP loss (page-color flattening) does NOT
   transfer to the DRAM regime — TLB reach wins there.
4. L=32 / L=16 paths untouched: chain outputs cmp-IDENTICAL to the r11
   ship binary's, layout-tax race a wash (rotated pairs 54.23-54.81 vs
   54.42-55.03, mixed signs).

### The counters (mandatory protocol; 2-sample m=8 runs at L=128, core 5)

| counter | r11 arm | ZF (ships) | ZF+YF (raced out) |
|---|---|---|---|
| instructions | 2.81G | **2.55G (-9.3%)** | 2.54G |
| l1d.replacement | 266M | **236M (-11%)** | 235M |
| l2_lines_in.all | 197M | 196M (flat) | 162M (-18%) |
| LLC-loads / LLC-load-misses | 29.5M / 18.1M | 30.4M / 18.0M (flat) | **35.8M / 24.1M (+33%)** |

**The r11 next-step's theory of ZF was WRONG in mechanism and right in
value**: the tile trip count is unchanged at three (the y pass still
re-reads the tile) — the -8% is the zy pass's z-side loop overhead and L1
reloads deleted (instructions -9.3%, L1 fills -11%), not an L3-trip cut.
At L=64 the counters are starker: LLC-loads 51.4M -> 7.4M (-86%),
l2_lines_in -29% against +9% instructions — with ZF the demand-traffic
elimination now dominates the fused sweep's addressing tax, which is what
flips the r11 verdict.

### What did NOT work, with the number that killed it

* **GP128_YF=1 (y fused with x-stage-1 per 128-KB column slab)**: alone
  -4% vs r11 (11.87-12.28), but a +2% LOSS on top of ZF (ship-both
  11.30-12.03 vs ZF-only 11.32-11.66, ZF wins/washes 6/6).  The counters
  above are the post-mortem: the slab fusion DOES cut L3->L2 traffic
  (l2_lines_in -18%) but its walk is 128-B touches at 2112-B stride — no
  streamer covers it, and demand LLC misses rise +33%.  **The r11 L=64
  lesson, one level down the hierarchy: covered traffic is nearly free;
  only DEMAND misses are worth restructuring against.  Run the
  LLC-loads/LLC-load-misses discriminator BEFORE building any fusion.**
  Kept compilable (all four ZF/YF combos build; (0,0) is the exact r11
  codegen).

### Measured on the node (a80n0 core 5, rotated same-core adjacent pairs)

| case | r11 arm (same window) | gen_r12 ship | verdict |
|---|---|---|---|
| L=128 B=1 m=8 chain | 11.79-12.77 ms | **10.74-10.88 ms** (best 10.744, sd 0.04%) | -8..-15%, 6/6 + 4/4 pairs; MKL 20.91 => **1.92x** |
| L=128 B=2 m=8 chain | — | 12.36 (one window) | gates pass |
| L=64 B=2 m=64 chain | two-sweep 692-699 us | **611-623** (best 609.1) | -12%, 4/4 |
| L=64 B=1 m=64 chain | two-sweep 704-707 | **621** (final ship 608.4) | -12%, 2/2 |
| L=32 B=8 m=250 (graded) | bit-identical path | **54.14-54.93** this session (sd 0.01-0.03%) | no layout tax |
| L=16 B=8 m=300 chain | bit-identical path | 6.58 (fast state) | untouched |

SPR dev signal (wallaby core 97, r9 protocol): L=64 fused+ZF 478.6 vs
two-sweep 555.4 (**-14%**, 2/2); L=128 ZF 7.02 vs r11 7.43-7.48 ms
(**-5.5%**, 2/2).  Both new defaults transfer; no wisdom-race hedge needed
on the hosts we can see.

### Gates (ship build = flagless default, node, check.py by hand)

All chain outputs bit-identical to the r11 ship binary's at every size
(cmp), so every digit reproduces: L=32 single 2.902e-16 (B=8) / 2.915e-16
(B=1), m=2 fused **1.338e-15 / 1.393e-15** (tol 3e-14), chain m=250
3.328e-14 / 2.948e-14; L=64 single 3.214e-16, m=2 **1.723e-15**, chain
m=64 2.342e-14 (now via the FUSED path — verified identical to the
two-sweep output); L=128 single 4.092e-16, m=2 **2.469e-15**, chain m=8
5.564e-15, B=2 chain 5.938e-15; L=16 chain m=300 2.351e-14.  2^k
regression 2/4/8 singles 0 / 0 / 1.4e-16.  Repeatable (cmp-identical
across independent runs) at 128 and 64.  -Wall -Wextra: only the
pre-existing 5 notes (mode-0 -Wrestrict quartet + r10 'nxt').

### Borrowed / attribution (gen_r12)

* **gen_layout gen_r11 + gen_powp gen_r11**: the THP-on-madvise finding
  and its adoption recipe — GP128_HP is their mechanism applied to this
  entry's one DRAM-regime block, raced honestly against my own r4/r7
  anti-hugepage results (which stand in their own regimes).
* **My own r11**: the one-sweep fused sweep and the demand-vs-covered LLC
  discriminator; this round extends the discriminator from "should you
  fuse" to "which SIDE of a fusion pays" (ZF yes, YF no) and corrects the
  r11 trip-count theory with the instruction/L1 attribution.
* The z-into-stage-2 row fusion (xs2z_128/xs2z_64) and the column-slab
  y+stage-1 fusion (raced out): this entry, new this round.

### Transfer note for the L=100 owners

The winning shape generalizes: **fuse the next step's FIRST post-map pass
into the current step's LAST pass at the granularity where its input
becomes complete in L1** (here: z-rows into stage-2's row completion) —
it cost -9% instructions at 128 and flipped the L3-regime verdict at 64.
The losing shape is its dual: do NOT fuse a pass whose fused-order walk
is streamer-hostile (my column slabs at 2112-B stride) — check
LLC-load-misses before building; l2_lines_in going DOWN while wall goes
UP is the signature.  At L=100, gen_powp/gen_pfa_large's prepass would be
the fusion target, and their THP re-home already ships.

### What I would do next

1. **L=128 residual**: 10.74 ms vs the ~7.4 ms DRAM floor.  The remaining
   structure is the y-pass + stage-1 tile revisits; the YF loss says the
   fix is NOT slab fusion.  The one untested shape is reordering stage-1
   to consume y-line outputs at y-line granularity (stage-1 column q needs
   only column q of all 8 group planes — the y pass could emit columns
   round-robin across planes instead of plane-major, keeping walks
   sequential per plane).  Needs a paper walk-order audit first.
2. **L=64 at B=1/B=2 is now fused**: if the trunk ever routes a 2^k in
   the L3 regime, GP64_FUSE/GP64_ZF are live defaults; re-race on CLX
   (smaller LLC moves the boundary toward the r11 verdict).
3. Nothing at L=32 — bit-identical for the sixth round ([port || L2], r8
   attribution unchanged).
4. Knob inventory for the wisdom race grows by {GP128_ZF, GP128_YF,
   GP64_ZF, GP128_HP} on top of the r10 list.

## Round gen_r13 — the B=1 single-call round: execute-path conversion fusion
## ships (-19/-19/-24/-13% at 16/32/64/128 singles), the r8 "if anyone ever
## scores 2^k singles" lever finally spent, and a TLB inversion at 128 worth
## every peer's attention

Standings into the round: led L=32 at 54.851 us (3.13x mkl_dfti 171.758, r12
board).  The r13 brief is the benchFFT round: the community harness times
REPEATED B=1 fft3d_execute() CALLS (their calibrated min-timing, 5N log2 N
"mflops"), and the two new scored cells (10:1, 12:1) are other classes'
problems — my instruction was "protect your cells."  But the benchFFT curve
(results/benchfft/) runs straight through this class at 2^k, and it showed
exactly the disease my own r8 record predicted and postponed: the trunk's
16/32/64/128 single-call points were 1.39x / 1.32x / 1.16x / **1.03x** over
fftw3_measure — the custody conversions (nat_to_cust + cust_to_nat, two full
volume sweeps) are unamortized at m=1, ~20% of a 32 call and ~35% of a 128
call, and L=128 sat one bad window from a community LOSS.  This round built
the r8-designed fix.  The scored chain paths are UNTOUCHED (verified, below).

### What shipped: GP2_XFE=1 — conversion fusion on the execute path

The z-pass is the first consumer of the input and the x-pass pass-2 store is
the last producer of the output, and the custody conversion at 2^k is a pure
interleave (lanes = 8 adjacent z = one natural row segment; NO transpose
exists in the conversion).  So:
* **Load side**: per-engine z-codelet variants (zpair_n / zquad16_n /
  zrow64_n / zrow128_n) read the NATURAL volume directly, deinterleaving
  with the same 2 DEIN shuffles per slot the conversion sweep paid anyway,
  and write custody S as usual.  Deletes one custody-volume store + load.
* **Store side**: vertical-FFT variants (vfft32e / vfft16e / vfft64e /
  vfft128e) whose pass-2 stores ILV-interleave straight to the natural
  output at stride 2·L·L doubles — custody S is never written by the last
  pass at all.  Deletes another custody-volume store + load.
* Four new top-level steps (fft_exec16/32/64/128) instantiate the plain
  two-sweep step shape (z/y skew as shipped, DSB-rolled bodies) with those
  variants; fft3d_execute routes per size.  All codelet arithmetic is the
  byte-for-byte shape of the unfused variants, so the fused output is
  **BIT-IDENTICAL to nat_to_cust + step_plain + cust_to_nat** — cmp-verified
  at every size, which is also why every gate digit below is the standing
  one.  Phase ordering (all natural reads in the z-phase, all natural writes
  in the x-phase) makes it in-place-safe; nin/nout deliberately carry no
  restrict against each other.  GP2_XFE=0 restores the r12 execute as the
  raced control; chain paths never see any of it.

### The L=128 finding: huge pages invert the fusion (GP128_XF)

Full fusion at 128 was a WASH (17.2-17.4 vs ctl 17.1-18.2 ms, mixed signs,
3 rotated rounds) despite deleting ~138 MB of the call's ~240 MB of DRAM
traffic.  Mechanism: the custody block is huge-paged since r12 (GP128_HP),
so the unfused x-pass stores at X128 stride (270 KB) cost ~8 stores per 2-MB
page — but the fused stores land in the DRIVER'S natural buffer, 4-KB pages,
at 256-KB stride: ONE PAGE WALK PER STORE LINE, and the TLB cost eats the
entire traffic win.  **GP128_XF=1 ships the load-side-only form** (z reads
natural, x-pass keeps its huge-paged S stores, the sequential cust_to_nat
sweep stays): 15.08-15.67 vs ctl 17.22-18.16 ms, 3/3 rotated rounds, ~-13%.
=2 (full) and =0 (unfused) stay compilable as the race axes.  Rule for the
record, the r11 "fused operands take the fused consumer's walk order" lesson
extended one level down: **a fusion that redirects strided STORES from your
own (huge-paged) buffer to a caller's 4-KB-paged buffer can lose to page
walks everything it wins in traffic — check the store-target's page size
before fusing, not just the walk order.**  At 16/32/64 the natural stride is
16 lines/page or the set is L2-resident: full fusion is clean there.

### Measured on the node (a80n0 core 6, slot lease 4, rotated same-core
### adjacent pairs; fused vs GP2_XFE=0 ctl, same day, same core)

| case (B=1 single call) | ctl (unfused) | gen_r13 ship | verdict |
|---|---|---|---|
| L=16 | 7.445-8.493 us | **6.016-6.808** (best 6.016) | -19..-21%, 3/3 |
| L=32 | 75.86-76.04 | **61.60-61.63** (sd 0.04%) | **-19%**, 3/3 |
| L=64 | 1047-1057 | **793.4-799.6** (best 793.4) | **-24%**, 3/3 (r8's 1219 us single is now 793) |
| L=128 | 17.22-18.16 ms | **15.08-15.67** (GP128_XF=1) | **-13%**, 3/3; full form wash (TLB, above) |

benchFFT-convention equivalents (5N log2 N / t, node): 16: 32.7k -> ~40.9k;
32: 31.9k -> ~39.9k (fftw3 24.2k); 64: 21.4k -> ~29.7k (fftw3 18.5k);
128: 13.0k -> **~14.6k** (fftw3 12.6k — the 1.03x cell moves to ~1.16x).
wallaby dev signal (SPR core 97, load 148 — noisy, footprint kept minimal):
L=32 B=1 single fused 60.5-62.9 vs unfused 76.5-86.6, 3/3, same story.

### Protection (the round's standing order), all on the node

* Chain outputs **cmp-IDENTICAL to the r12 ship binary** at L=32 (B=8
  m=250), L=16 (B=8 m=300), L=64 (B=1 m=64), L=128 (B=1 m=2).
* Graded case timing: 54.813 us min via tryout.sh (MKL same-window 185.2),
  56.18/56.23 in the by-hand same-window pair vs the r12 binary — code-
  layout tax a wash (0.07%), the gen_twiddle r5 hazard cleared.
* Gates, ship build, standing digits exactly: L=32 single 2.902e-16 (B=8) /
  2.915e-16 (B=1); m=2 fused **1.338e-15**; chain m=250 3.328e-14 (B=8) /
  2.948e-14 (B=1); L=32 B=1 chain 57.80 us this window.  Singles vs numpy:
  L=16 2.337e-16, L=64 3.214e-16, L=128 4.092e-16 (r8's exact digit).
  Generic 2/4/8: 0 / 0 / 1.4e-16.  B=8 fused execute PASS (2.902e-16).
* -Wall -Wextra: only the pre-existing 'nxt' note; scalar (no-AVX-512)
  build clean.  tryout.sh runs unaided again; its remote map-check leg
  still dies on the literal '$W/c.bin' (standing since r1), map gates by
  hand as always.

### What did NOT work, with the number that killed it

* **Full store-side fusion at L=128** (GP128_XF=2): 17.34-17.95 vs ctl
  17.22-18.16 ms — wash, 3 rounds, mechanism above.  Kept compilable; on a
  host where the driver's buffers are THP-backed it should flip, which is
  exactly a wisdom-race axis.

### Borrowed / attribution (gen_r13)

* The fusion design is my own r8 "single-call conversion fusion" note,
  built at last because the round made 2^k singles visible (benchFFT);
  the monitor's benchfft_ours wiring supplied the target numbers.
* **gen_layout/gen_powp gen_r11's THP finding**, which I adopted in r12 as
  GP128_HP, is the direct CAUSE of the L=128 inversion — an honest case of
  one optimization changing the sign of the next.  The page-size boundary
  condition on store-side fusion is this round's contribution back.
* Protection protocol: gen_batchlane gen_r8's final-round rule (chain paths
  untouched, bit-identity cmp against the shipped binary), gen_twiddle
  gen_r5's code-layout hazard race, rotated-order same-core pairs (my r5),
  one-lease-one-core (gen_dense_prime r5 / gen_batchlane r4).

### What I would do next

1. **L=128 single residual**: 15.1 ms = z-fused sweep + x sweep + the
   sequential cust_to_nat.  The remaining lever is applying the r11/r12
   FUSED-SWEEP structure (one tile-resident sweep + ZF) to the execute
   path too — the m=1 case of the chain engine, with the natural-load z
   variants feeding stage-1 and cust_to_nat_g on the permuted planes.
   Worth ~2-3 ms on paper; a full round of careful permutation work.
2. **If 10:1/12:1-style B=1 cells ever reach 2^k** (16:1 is the obvious
   candidate): the fused execute IS the B=1 chain step at m=1; a fused
   B=1 CHAIN at small L would additionally want the c volume read
   natural-side, the same shape gen_pfa_small is building this round.
3. Wisdom-race axes grow by {GP2_XFE, GP128_XF}.  Cross-arch note: on CLX
   (no THP'd custody block? smaller STLB) GP128_XF may flip either way;
   the SPR signal agrees with ICL on the 16/32/64 wins.
4. Nothing at L=32 chain — bit-identical for the seventh round.

## Round gen_r14 — the tiled one-sweep EXECUTE ships at 64/128 (-28% at the
## weakest B=1 cell), NT emit for aligned callers, and the round's most
## important find: every r13 fused execute path SEGFAULTED on benchFFT's
## 16-byte-aligned buffers — fixed, and every peer should check for the same

Standings into the round: the r14 brief names the B=1 execute() curve as the
systemic seam; my class's numbers in it (32: 1.32x, 64: 1.16x, 128: 1.03x
over fftw3) are the PRE-r13 trunk's — results/benchfft/ was measured against
impl_12, before my r13 GP2_XFE conversion fusion (my r13 record's "benchFFT-
convention equivalents" already put 16/32/64 at 1.6-1.7x and 128 at ~1.16x).
So this round's real work at my sizes was (a) the L=128 residual — my r13
next-step #1, the r11/r12 fused-sweep structure applied to execute() — and
(b) whatever the benchFFT harness itself would expose.  (b) turned out to be
a latent crash that would have zeroed the class in this round's scoring.
Session: reservation 439820 on a80n0, slot lease 2 / core 5-2=4 held for the
session; rotated-order same-core adjacent pairs; first-position runs
discarded (the r12 cold rule); all gates by hand with check.py on the node.
Node sshd dropped connections late in the session; the last hygiene checks
(scalar build, misaligned sanity sweep, final-source bit-identity) ran on
wallaby as dev signal.  Monitor note: after ~23:00 BOTH Ice Lake nodes
reject ssh with "pam_slurm_adopt: you have no active jobs on this node"
while ./reserve.sh --status still reads ALIVE — the wallaby_shims squeue
answers 'R' whenever RESERVATION.heartbeat is <300 s fresh, and something
is still refreshing that file, so the shim now produces a false POSITIVE
(the mirror image of the r10 false negative).  Lease released cleanly.

### (1) THE CRASH: caller-buffer alignment (fix shipped, peers take note)

benchFFT's bench_malloc returns malloc+16 — buffers 16-B aligned, never 64.
Every natural-side access in my r13 execute fusion (and the chain-end
conversions) was a `*(v8d *)` deref, which LICENSES gcc to assume 64-B
alignment; our own driver 64-aligns everything so five rounds of testing
never noticed, but a deliberate alignment-offset harness (scratch TU
build/tryout/gen_pow2/al16bench.c, offsets 8/16/32 at every size) got an
immediate SIGSEGV at L=128 off=16.  The entry would have CRASHED in the
round's own scoring metric.  Fix: `v8du` (vector_size(64), aligned(8)) with
LDU/STU wrappers on every caller-buffer access — nat loads in zpair_n/
zrow64_n/zquad16_n/zrow128_n, nat stores in vfft{16,32,64,128}e and the
xs2e emits, and all nat_to_cust*/cust_to_nat*/cust_map_to_nat* variants.
Custody S/C accesses stay `*(v8d *)` (we allocate them, 64-B by
construction).  Outputs at every size/offset byte-identical (fnv cmp), no
measurable cost on aligned buffers (all races below are post-fix).
**Peer warning: any entry that vector-loads the driver's/benchFFT's in, out,
x0 or c pointers with an aligned deref has the same latent fault — the gen
driver hides it, benchFFT does not.  Audit for it; the misaligned harness is
in my tryout dir and takes a minute per size.**  Measured line-split cost of
honest unaligned access (al16, off=16 vs off=0, regular stores): FREE at 128
(12.07 vs 12.05 ms — DRAM latency hides splits), ~+11% at 64, ~+26% at 32,
~+44% at 16 (compute-bound sizes feel the doubled L1 accesses).  fftw3's
benchFFT numbers were measured under the same 16-B handicap, so the
comparison stays fair; a permute-realign load path for small-L misaligned
callers is a real open lever, declined this round (all cells win anyway).

### (2) GP128_XT / GP64_XT: the tiled one-sweep execute (ships, both =1)

The r11/r12 chain structure at m=1: sweep 1 = z (NATURAL loads) + y skewed
+ x-stage-1 per stride-G plane group (the fft_prologue shape with zrow*_n);
sweep 2 = x-stage-2 per tile of G consecutive planes (identity placement —
no perm composition at m=1), with the stage-2 outputs EMITTED straight to
natural out (ILV, output plane n = t + 8*mm).  The tile is read-only in
sweep 2, so both the in-place x-pass custody write AND the cust_to_nat read
are deleted: at 128 that is ~69 MB of the call's ~205 MB DRAM set.  Why
this dodges the r13 GP128_XF=2 TLB inversion: the unsplit x-pass touched
all 128 natural planes per COLUMN (a page walk per store line); the tiled
stage-2 touches 16 planes per TILE, each as sequential 2-KB row chunks —
16 forward streams, DTLB- and streamer-clean.  Arithmetic is vfft64i/
vfft128i's own two passes on the same values in the same order (the r11
stage split), so output is BIT-IDENTICAL to the r13 execute (cmp, every
size, every build).

### (3) GP128_NTE: non-temporal zmm emit for 64-B-aligned callers (ships)

The natural write at 128 is a 33.5-MB pure stream; regular stores pay an
RFO read per line.  vmovntpd deletes those reads (gen_layout gen_r4's
NT-on-DRAM-resident precedent): raced -9%, 3/3 (10.44-10.53 vs 11.50-11.62
ms same-window).  Runtime-gated on 64-B alignment of nout; sfence after the
sweep.  For benchFFT's 16-B callers the gate falls back to regular
unaligned stores — see what did NOT work.

### Measured on the node (a80n0 core 4, rotated same-core adjacent pairs,
### ship = final build, ctl = the r13-shape two-sweep fused execute)

| case (B=1 single call) | ctl (r13 path) | gen_r14 ship | verdict |
|---|---|---|---|
| L=128 | 14.83-16.57 ms | **10.41-11.04 ms** (best 10.409) | **-28..-30%, 6/6** (tiled -20..-23%, NT -9% on top) |
| L=64 | 806-812 us (736-750 fast state) | **787-789 (732-738)** | -1..-4%, 4/4 mins, medians -3..-4% |
| L=32 | 60.6-72.0 us | 60.8-66.3 | untouched path, wash |
| L=16 | 6.04-6.86 us | 6.13-6.20 | untouched path, wash |

benchFFT-convention equivalents (5N log2N / t; driver GF/s x1000 = the same
number): L=128 aligned 10.41 ms -> **21.2k mflops**; at benchFFT's 16-B
alignment (al16 harness, regular-store fallback) ~12.06 ms -> **~18.3k vs
fftw3's 12,604 = ~1.45x** (was 1.03x pre-r13, ~1.16x post-r13).  L=64:
732 us -> 32.2k aligned, ~27k at 16-B vs fftw3 18,487.  L=32: ~40k aligned,
~32k at 16-B vs 24,163.  L=16: ~40k aligned, ~28k at 16-B vs 23,510.  Every
2^k B=1 cell beats fftw3_measure under benchFFT's own alignment handicap.

### Protection (all on the node, final ship build)

Chain paths untouched: graded L=32 B=8 m=250 chain output BIT-IDENTICAL to
the r13 arithmetic (cmp vs ctl build, which is the r13 codegen); m=2 chains
at 64/128 cmp-identical across arms.  Code-layout tax raced properly after
one scare: the first ship-vs-ctl graded pair read +11% — that was the
FIRST-RUN-COLD artifact (r12 rule: discard position-1 runs); 11 subsequent
clean rotated pairs gave deltas {+0.77,+0.64,+1.1,+0.16,-0.25}% then
{-0.08,-0.5,+0.7,+0.5,-0.6}% — a wash, mixed signs.  The tiled-exec bodies
were additionally moved to their own .text section (gen_twiddle r5 hazard
hygiene; the wash held before and after, so it ships as insurance, not as a
measured win).  Gates, ship build, standing digits exactly: singles
2.357e-16 / 2.915e-16 / 3.209e-16 / 4.092e-16 (16/32/64/128, B=1, tol
1e-12); B=8 exec 2.903e-16; m=2 fused **8.650e-16 / 1.393e-15 / 2.005e-15 /
2.469e-15** (tol 3e-14); graded chain m=250 2.902e-16 single-equiv, chain
end 3.328e-14 (tol 1e-10), repeatable (cmp x2).  -Wall -Wextra: only the
pre-existing 'nxt' note; scalar (-mno-avx512f) build clean, 0 warnings.

### What did NOT work, with the number that killed it

* **Interleaved xmm NT stores for 16-B-aligned callers** (first attempt at
  keeping NT under benchFFT alignment): **16.7 ms vs 12.1 regular — a
  disaster.**  16 concurrent output streams each leaving partial WC lines
  (128-B visits at 16-mod-64 alignment) thrash the ~dozen WC buffers into
  partial-line NT flushes.  Recorded so nobody retries the naive form.
* **Staged shifted-NT rows (GP128_NTS, kept compilable, default 0)**: stage
  row y of all 16 planes in L1 (32 KB), then per plane one vpermt2pd per
  vector + aligned zmm NT to the interior, scalar edges.  Fixes the WC
  thrash but the staging round trip + permutes eat the RFO saving: 12.06 vs
  regular 11.4-12.1 ms, 3 rotated rounds — wash-to-loss.  Simplest-wins:
  misaligned callers take plain unaligned stores.  A CLX/SPR host may
  re-race the knob (SPR's WC/store path differs).
* **The first graded-case pair scare** (+11%): first-run-cold, not layout —
  see Protection.  Lesson re-learned: never act on a position-1 run.

### Borrowed / attribution (gen_r14)

* **gen_layout gen_r4**: the NT-full-line-stores-on-DRAM-resident-volumes
  precedent behind GP128_NTE.
* **My own r11/r12** one-sweep fused structure and r13 conversion fusion —
  this round is their composition at m=1; the identity-placement
  simplification (no perm needed at m=1) is what makes it cheap.
* The alignment audit and the al16bench harness: this entry, new; prompted
  by reading benchFFT's libbench/util.c allocator while checking whether
  the NT gate would ever fire in the scoring harness.
* Session protocol unchanged (rotated same-core pairs, first-run discard,
  bit-identity cmp gating, one-lease-one-core, held-lease ssh).

### What I would do next

1. **L=128 residual**: 10.4 ms vs a ~9.7 ms paper floor for the 136-MB set
   at ~14 GB/s — the structure is now within ~7% of its own traffic bound;
   further gains need the set cut (a z+y+xs1 sweep that reads natural AND
   emits stage-2 in one pass does not exist — stage-2 needs all groups'
   stage-1 done).  Probably closed.
2. **Misaligned small-L execute** (the +26%/+44% split penalties at 32/16):
   permute-realign loads (two aligned loads + vpermt2pd feeding DEIN) would
   trade L1 accesses for p5 uops; only worth it if a scored cell ever runs
   misaligned-compute-bound.  benchFFT does exactly this — if the r14 board
   shows 16/32 thinner than the aligned numbers predict, this is why.
3. **Peers**: run the alignment audit (see (1)).  gen_race/trunk: when
   forwarding execute() to class .so's, the caller's buffers pass through
   verbatim — the trunk inherits every class's alignment assumptions.
4. Knob inventory grows by {GP128_XT, GP64_XT, GP128_NTE, GP128_NTS}; the
   NT gate is runtime (per-call alignment), the rest compile-time race axes.
