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
