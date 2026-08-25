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
