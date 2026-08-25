# gen_batchlane -- SoA 8-vol/zmm batch-lane engine (L = 10, 12, 15 at B >= 8)

## Round gen_r1 -- from the dense-floor stub to a full batch-lane PFA chain engine

### What shipped

Replaced the stub with the real class engine, generalizing ice_r7/r8's **bl8**
chain (L8_fusedaxes.c, itself adopted from rivals v5_cb7847fb / 8dc1a96d) from
L=8 to 10/12/15:

1. **Batch-lane SoA**: 8 volumes fill the zmm lanes; every axis DFT is
   elementwise across registers -- zero shuffles in the arithmetic, the only
   transposes are pack/unpack at the chain ends. Scratch = split-complex site
   vectors (re[8] | im[8], 128 B/site), natural site order, plane stride padded
   to PL2 = 130/162/226 site-vectors so plane bytes == 256 (mod 4096) (bl8's
   anti-alias pad, kept untested-against on this engine -- candidate A/B next
   round).
2. **Twiddle-free two-stage Good-Thomas PFA pencils**: 10 = 2x5, 12 = 3x4,
   15 = 3x5 are coprime splits, so there is NO twiddle table anywhere: input map
   n = (Qa+Pb) mod L and CRT output map k are baked into compile-time slot
   offsets. Codelet costs per pencil per 8 volumes (FMA-contracted vector
   instrs): L=10: 5xDFT2+2xDFT5 = 88; L=12: 4xDFT3+3xDFT4 = 96; L=15:
   5xDFT3+3xDFT5 = 162 (DFT5 is the 32a+12m Winograd split, 34 instrs).
   In-place safety was proven per c-group; the ONE exception is L=15 stage 2,
   where groups c=1 and c=2 have EQUAL read/write slot sets ({slots != 0 mod 3}),
   so they are a single fused load-both-then-store-both codelet (DFT5X2). Get
   this wrong and you silently corrupt slot 10 first -- the slot tables in the
   file header comment are the reference.
3. **Two-sweep step**: zy sweep per x-plane (12.8-28.9 KiB, L1-resident:
   z-pencils stride 1 site, y-pencils stride L), then x-pass per column (stride
   PL2 sites, an L2 stream). No software prefetch (bl8 measured pf as pure loss;
   not re-litigated).
4. **fft3d_chain owns the graded m-step map chain**: pack x0 and c to SoA once
   per group, run all m steps in place in SoA (state + c together <= 848 KiB at
   L=15, L2-resident; (C - S) == 2048 mod 4096, bl8's de-alias offset), unpack
   once. Map is EAGER and in-register, fused into the x-pass stores
   (bl8/L17_matrixsimd lesson: lazy map loses ~24% -- it fronts the next step's
   critical path). Map form is bl8's exact r4 ladder verbatim: s = wr^2+wi^2 +
   1e-300, vrsqrt14pd, two quadratic Newtons, d = fma(s,y,1), one vdivpd.
5. **Pack/unpack**: bl8's trans8 (24 lane permutes / 4 sites) and
   untrans_interleave (48 ops / 8 sites) verbatim; I re-derived the index
   algebra: trans8 leaves lane l = volume lanex[l] (lanex = swap bits 1,2,
   self-inverse) and untrans_interleave's UO_V/UO_H output table composes it
   away. Tails (L^2 % 4, % 8 != 0) redo a full overlapping block -- stores are
   idempotent, so overlap is safe.
6. **Any B >= 1**: remainder group (incl. B = 1) replicates the last volume into
   unused lanes, unpacks only real ones. Correct, but pays up to 8x on the
   remainder group.
7. **SCHED15**: `optimize("schedule-insns","sched-pressure")` as a function
   attribute (bl8 ice_r8's trick) on the **L=15 family only** -- see the numbers.

### Measured on the reserved Ice Lake node (a80n0) via tryout.sh, graded chain

| case | this engine (min us/xform) | MKL 2022 same case | ratio |
|---|---|---|---|
| L=10 B=64 m=1000 | **1.163** (typ 1.17) | 4.646 | 4.0x |
| L=12 B=64 m=600  | **1.995** | 7.877 | 3.9x |
| L=15 B=32 m=600  | **4.643** (one 4.497 seen) | 16.734 | 3.6x |

Gates: single-call rel L2 = 2.6e-16 / 2.9e-16 / 3.1e-16 (tol 1e-12); full graded
chains rel L2 = 1.6e-13 (m=1000), 4.8e-14, 5.2e-14 vs honest anchors 1.1e-13,
3.9e-14, 4.8e-14 (tol 1e-10) -- drift is at the anchor, i.e. exact-tier; the
m=2 gate (3e-14) has orders of magnitude of margin. Repeatable bit-identical.
L=15 run-to-run spread on the leased core is ~3% (other implementers share the
node's mesh/LLC); L=10/12 are ~0.1%.

### What did NOT work, with the number that killed it

- `-fschedule-insns -fsched-pressure` globally: L=15 5.168 -> 4.497 (-13%, the
  DFT5X2 pair is register-pressure-bound and this cuts its spills) but L=10
  1.163 -> 1.362 (+17%) and L=12 1.995 -> 2.317 (+16%). Hence the attribute on
  the 15-family only; 4.497 itself did not reproduce (4.64-4.77 typical) -- it
  was a good-luck run, not the attribute-vs-flags difference.
- B=1 through the padded batch-lane path: 10.6 / 18.0 / 41.3 us vs MKL B=1
  4.5 / 7.4 / 16.3 -- 2.3-2.5x SLOWER. Expected (7 of 8 lanes wasted). No
  scored case has B < 32, and the roster scopes this class to B >= 8, so I
  shipped correctness and stopped, but see next steps.

### Borrowed, plainly

Nearly everything structural is from **L8_fusedaxes.c's bl8** (ice_r7/r8, which
itself credits rivals v5_cb7847fb and 8dc1a96d): the batch-lane idea, trans8 /
untrans_interleave networks and their index maps, the rsqrt14+2NR+1vdivpd map
ladder with the 1e-300 guard, eager in-register map placement, in-place SoA
chain with pack-once/unpack-once, plane-stride 256 mod 4096 and (C-S) 2048 mod
4096 pads, no-software-prefetch, and the sched-pressure attribute. The PFA slot
algebra and the 10/12/15 codelets are new this round.

### Harness notes for whoever reads this next (monitor included)

- wallaby has no slurm client, so `reserve.sh --status` (and therefore
  tryout.sh) reports a LIVE reservation as dead. I ran with a PATH shim that
  answers `squeue -o %t` with R while the heartbeat is < 300 s old.
- tryout.sh line 36 uses `$W` before line 38 defines it: with a chained case
  (all of mine) it dies with `W: unbound variable` under `set -u`. Workaround:
  export W=<gen>/build/tryout/<name> in the environment first.
- tryout.sh's remote check.py line quotes `'$W/c.bin'` inside `$(...)`, which
  reaches the remote shell unexpanded -> `--cin /c.bin` -> FileNotFoundError.
  The single-call gate still runs; I ran the map-check by hand on wallaby
  (shared FS) with the same arguments -- that is what the chain numbers above
  are from.
- No node-built baselines existed yet; I built MKL (sota/mkl_dfti.c, MKL
  2022.0.2, sequential) into my own build/tryout dir for the reference column.

### Next round, in order of expected value

1. **B=1 / remainder path**: a lane-spatial engine (x/y axes vectorize over 8
   contiguous sites with masked tails; z axis via an in-L1 plane transpose or
   the y-lane gather) should land near MKL B=1 rather than 2.4x behind. Only
   matters if the monitor scores off-case batches or round 6 draws one.
2. **x-pass column pairing** for L=15: process two columns per iteration to
   overlap the map's divider latency with the second column's FFT work
   (bl8-style divider hiding is currently within one column only).
3. A/B the mod-4096 pads ON this engine (inherited on faith from bl8).
4. From round 3 the driver may ask any class size: the engine generalizes to
   any L with a coprime split (14=2x7, 20=4x5, 21=3x7, 30, 33, 35...) by
   emitting the same two-stage PFA with a DFT7/DFT11 module; coordinate with
   gen_pfa_small/gen_planner on who serves what at which B.

## Round gen_r2 -- kill the divider, spread sched-pressure, adopt the layout layer

Standings into the round: led all three owned sizes (1.174 / 1.995 / 4.686 us on
the r1 leaderboard; gen_pfa_small 21-32% behind). This round is cumulative, so
the work was: take what peers proved, re-test my own r1 conclusions that their
records contradicted, and not touch what works.

### What changed (three things, in the order they landed)

1. **Divider-free map ladder** (ADOPTED from **gen_pfa_small r1**, corroborated
   by gen_powp/gen_pow2 -- three records now agree vdivpd/vsqrtpd zmm are the
   enemy). The map's final `t = 1/d` was one vdivpd per output vector; zmm
   vdivpd is unpipelined (~16 cyc throughput), and the fused x-pass issues one
   per site-vector -- at L=15 that is 15 divides x 16 cyc = 240 cyc of divider
   occupancy per pencil vs ~160 cyc of FMA work. Replaced with vrcp14pd + TWO
   residual Newtons (t += t*(1 - d*t), 2^-14 -> 2^-28 -> full double), +5
   FMA-port ops, -1 divide. Measured (same window, same core, control first):
   L=10: 1.333 -> 1.225 (**-8.1%**), L=12: 2.274 -> 2.073 (**-8.8%**),
   L=15: 4.707 -> 4.484 (**-4.7%**). The rsqrt half of the ladder is unchanged
   bl8 r4.

2. **sched-pressure as a per-function attribute on ALL pencil families, not
   just 15** -- this REVISES my r1 conclusion. r1 measured the GLOBAL flags
   at +17%/+16% for 10/12 and shipped the attribute on the 15-family only.
   gen_powp's r1 record (same flags, +48% at their L=100, -5% at 25) made me
   suspect the r1 loss was the flags rescheduling the shuffle-heavy
   pack/unpack and trans8 networks, not the pencils. Re-tested as a
   per-FUNCTION attribute on the 10/12 pencil/sweep/chainstep families only:
   L=10: 1.222 -> 1.165 (**-4.7%**), L=12: 2.068 -> 1.936 (**-6.4%**).
   Also re-confirmed SCHED15 with the new longer ladder: stripping it costs
   +4.3% (4.488 -> 4.683). Lesson recorded: "global flags lose" does not imply
   "the attribute loses" -- the r1 experiment conflated codelet scheduling
   with pack/unpack scheduling. Dev knobs -DBL_NOSCHED1012 / -DBL_NOSCHED15
   left in for the monitor's cross-arch reruns.

3. **gen_layout adoption: THP 2MiB arena** (`gl_map_huge`) for the SoA
   scratch, replacing posix_memalign; phases (256-mod-4096 plane stride,
   (C-S) == 2048 mod 4096) preserved exactly. Measured a WASH at all three
   sizes (1.225 -> 1.222, 2.073 -> 2.068, 4.484 -> 4.488 -- all within the
   ~0.5% window noise); the state walks its ~420 4K pages through the 64-entry
   L1 dTLB but they all sit in the 2048-entry STLB, so there was never a page
   walk to save. Kept anyway: zero measured cost, posix_memalign fallback
   inside the layer, and it future-proofs larger class sizes (r3+: L=21, 30,
   33... where the group grows past STLB reach). gen_layout's honest-adoption
   request is satisfied: this is a real null result on a real class kernel.

### Measured on the node (a80n0 via tryout.sh, graded chains, min over clean windows)

| case | r1 shipped | r2 shipped | same-window MKL 2022 | ratio |
|---|---|---|---|---|
| L=10 B=64 m=1000 | 1.163 (this window: 1.333) | **1.162** | 4.67 | **4.0x** |
| L=12 B=64 m=600  | 1.995 (this window: 2.274) | **1.931** | 7.91 | **4.1x** |
| L=15 B=32 m=600  | 4.643 (this window: 4.707) | **4.484** | 16.73 | **3.7x** |

Cross-window note for the monitor: wallaby-leased cores this round showed
BIMODAL sustained states (L=10 pinned runs of 1.16 vs 1.32 us with sd < 0.1%
inside each; MKL moved < 1% between the same windows) -- a compute-bound
kernel at 42+ GF/s sees neighbors' AVX-512 load through the all-core turbo
bin; a memory-bound baseline does not. All A/Bs above were control-first in
the same window; the r1-vs-r2 delta at L=10 is therefore quoted from the
clean-window pair, not across rounds.

Gates (final shipped build, all verified by hand on the node -- the tryout
map-check leg still dies on the unexpanded `$W`, see r1 harness notes):
single call 2.6/2.9/3.1e-16 (tol 1e-12); two-step m=2 gate 8.2e-16 / 9.2e-16 /
1.2e-15 (tol 3e-14, ~25x margin -- the rcp14 ladder costs nothing measurable
in precision); graded chains 1.687e-13 / 4.869e-14 / 5.249e-14 vs honest
anchors 1.081e-13 / 3.887e-14 / 4.784e-14 (1.6x / 1.25x / 1.1x, tol 1e-10);
chain outputs bit-identical across independent runs. B=1: correct
(2.6-3.1e-16), 10.50 / 17.54 / 40.88 us vs MKL 4.96 / 8.32 / 18.03 --
unchanged gap, see below.

### What did NOT work / was declined, with the number or the record that killed it

- **Nothing failed outright this round** -- but two candidate experiments were
  declined on peer evidence rather than re-run, which is the point of the
  cumulative round: (a) x-pass column PAIRING at L=15 (my own r1 next-step #2,
  originally aimed at hiding vdivpd latency): the divider it was meant to hide
  is now gone, and gen_pow2's structurally identical vfft32x2 experiment
  measured +15% from register-file overflow (eight live v8d[8] arrays); the
  DFT5X2SM pair already runs pressure-bound. Declined. (b) Software prefetch
  anywhere: now FOUR records agree it loses in issue-bound passes
  (bl8, gen_pfa_small +14%, gen_pow2 +3%, ice L45 +20%). Closed permanently
  for this engine.
- THP arena as a SPEEDUP: null (numbers above). Kept as insurance, but recorded
  so nobody re-measures TLB effects on an STLB-resident working set.

### Borrowed, plainly

- **gen_pfa_small r1**: the rcp14 + 2-Newton reciprocal (their "fast map"
  finding, worth -24..-32% to them; worth -5..-9% here because my r1 ladder
  already had the rsqrt half and only one divide remained).
- **gen_powp r1**: the counter-evidence on sched flags that triggered the
  per-function re-test (their attribute-on-25-family-only pattern, itself
  borrowed from my r1 SCHED15 -- the technique made a round trip and came
  back better).
- **gen_layout r1**: gl_map_huge/gl_unmap verbatim via the documented
  GEN_LAYOUT_LIB_ONLY include.
- **gen_pow2 r1**: the vfft32x2 register-overflow number that killed my
  column-pairing plan before I spent a day on it.

### Operation count

Unchanged per-pencil DFT costs (88 / 96 / 162 vector instrs per 8 volumes).
Map ladder is now ~20 FMA-port ops + 2 seed ops (vrsqrt14pd, vrcp14pd) per
site-vector, zero divider ops in the whole binary's hot path.

### What I would do next (ranked)

1. **B=1 / small-batch lane-spatial engine** (unchanged from r1, still the
   biggest hole: 2.1-2.3x behind MKL at B=1). The known-good shape is ice
   L6_pfa's interleaved-complex z-lane turn; gen_pfa_small's r1 has the same
   gap and their record sketches the same fix -- coordinate rather than build
   it twice. Matters for round 6 (library assembly at arbitrary B).
2. **Round-3 generality**: emit the two-stage PFA for any coprime split with
   factors in {2,3,4,5,7,8,9} (14, 20, 21, 30, 33, 35...) -- the codelet
   macros are already length-generic; what is missing is a DFT7 module and a
   slot-table generator instead of hand-baked tables. Coordinate with
   gen_planner on routing and with gen_pfa_small on who serves which B.
3. **L=15 op diet**: the 3-FMA lifting DFT5 form (LITERATURE 08 6.3) could cut
   ~10% of pencil FP; ice measured 11.9% op cut -> 0.8% time, so budget an
   hour, not a round.
4. If the monitor's cross-arch rerun flags SPR/CLX: re-race the SCHED knobs
   there (the -DBL_NOSCHED* build flags exist for exactly that).

## Round gen_r3 -- register-explicit pencils, L=20 adopted, per-size map tails

Standings into the round: led or tied all three owned sizes (r2 board: 10 at
1.162 tied with gen_pfa_small, 12 at 1.933 leading, 15 at 4.478 a hair behind
their 4.466). gen_pfa_small had adopted this engine wholesale in their r2, so
the tie is structural -- this round had to find something neither of us was
doing.

### The finding that drove the round: the compiler was wasting our stores

An asm audit of the r2 binary (gcc 11, -O3 -march=icelake-server, the exact
tryout flags) found two things the r1/r2 A/Bs never isolated:

1. **The in-place slot macros compiled to dead stores.** Each pencil ran two
   PFA stages through L1: gcc forwarded the stage-2 LOADS in registers (loads
   per pencil were already the minimal 2L), but its DSE failed to kill most of
   the stage-1 stores it had just made redundant: 27/39/60 zmm stores per
   pencil at L=10/12/15 where 20/24/30 suffice. At L=15 that is 30 pure-waste
   stores against 168 FP ops (2 stores/cycle on ICL: ~15 wasted cycles per
   ~84-cycle pencil).
2. **The sweep pencils were not inlined at all.** `dftN_pencil` carried the
   r2 `optimize("schedule-insns","sched-pressure")` attribute; gcc out-lined
   them as `dftN_pencil.constprop` and the zy sweeps paid call/ret per pencil
   (the map pencils DID inline into chainsteps). So r1/r2's "sched attribute"
   measurements were partly measuring inline-vs-call differences -- which is
   also why gen_pfa_small could not reproduce the attribute's win on their
   (fully inlined) build. Their r2 "does not transfer" note was the tell.

### What shipped

1. **Register-explicit pencils for 10/12/15** (R2L/R3L stage-1 macros read
   memory and write NAMED registers xr<k>/xi<k>; R4ST/R5ST/R4STM/R5STM stage-2
   macros read registers and store straight to memory, map fused). Exactly 2L
   zmm loads + 2L zmm stores per pencil, no store-forward round trip, no
   calls (`always_inline`, and the optimize attribute now lives ONLY on the
   sweep/xpass/chainsteps wrappers, so inlined bodies schedule under the
   caller's flags). Bonus: the L=15 stage-2 equal-slot-set hazard (r2's fused
   DFT5X2 codelet) is GONE -- stage 2 never reads memory, so groups c=1/c=2
   need no ordering. The chain outputs are bit-identical to r2 (same module
   arithmetic order), so all r2 error numbers carry over exactly.
2. **L=20 = 4x5 adopted** (from gen_pfa_small's r2 extension of this engine;
   slot algebra re-derived: input n=(5a+4b)%20, stage-1 DFT4 over a in place
   per b, stage-2 DFT5 over b reads (5c+4b) writes CRT (5c+16d); their
   Q==1 mod P disjointness rule makes it plain in-place). 40 live site
   registers will not fit, so L=20 keeps the r2 memory-round-trip form
   (M4IP/M5ST/M5STM macros). PL2=418 sites, plane bytes == 256 mod 4096.
3. **Per-size map tail** (re-raced on the new codelets): 10/12/15 keep the r2
   rcp14+2NR ladder; L=20 takes ONE exact vdivpd (gen_powp r2's verdict on
   their SoA x-pass transfers: the L=20 x-pass is memory-bound enough that
   the idle divider is free, and dropping the ladder's 5 FMA-port ops pays).
   Same-window pairs: 20: 13.72 div vs 14.35 ladder (-4.4%); 15: 4.92 div vs
   4.46 ladder (+10% -- ladder stays); 10/12: wash. Knobs -DBL_MAPDIV /
   -DBL_MAPRCP force one form everywhere for the cross-arch race.
4. **Per-size sched-pressure**: still ON for the 10/12/15 families -- on the
   new structure stripping it costs +5.7/+10/+5.3% (so the attribute's win was
   NOT only the inline confound; on register-explicit codelets it is real and
   bigger). OFF for L=20 (+4..8% when on, matching gen_pfa_small's r2 verdict
   on this codelet shape). Knobs: -DBL_NOSCHED1012/-DBL_NOSCHED15/-DBL_SCHED20.

### Measured on the node (a80n0 via tryout.sh, graded chains, control-first
### same-window pairs; quiet-window minima quoted)

| case | r2 shipped | gen_r3 now | same-window pair | MKL 2022 | ratio |
|---|---|---|---|---|---|
| L=10 B=64 m=1000 | 1.162 | **1.156** | 1.167 -> 1.156 (quiet); 1.328 -> 1.157 (slow-state ctl) | 4.55-4.67 | 4.0x |
| L=12 B=64 m=600  | 1.931 | **1.919** | 1.936 -> 1.950 busy pair, 1.919-1.920 quiet | 7.79-7.92 | 4.1x |
| L=15 B=32 m=600  | 4.484 | **4.456** | 4.529 -> 4.456 | 16.7 | 3.75x |
| L=20 B=32 m=256  | (new) | **12.99-13.17** | vs gen_pfa_small r2 13.39 | 58.9-60.8 | 4.5x |

Gates (final shipped build, run by hand on the node -- tryout's map-check leg
still gets the unexpanded `'$W/c.bin'`): single call 2.6/2.9/3.1/3.0e-16;
two-step m=2 gate 8.2e-16 / 9.2e-16 / 1.2e-15 / 1.2e-15 (tol 3e-14, ~25x
margin); graded chains 1.687e-13 / 4.869e-14 / 5.249e-14 / 4.366e-14 vs honest
anchors 1.081e-13 / 3.887e-14 / 4.784e-14 / 2.835e-14 (tol 1e-10); repeatable
bit-identical at all four sizes; B=1 and mixed B=12 group+remainder single and
m=2 chains PASS at all sizes (script: build/tryout/gen_batchlane/final_check.sh).
B=1 chains: 10.52 / 17.42 / 44.2 / 119.2 us -- the remainder-lane gap is
unchanged (see next steps).

Window health note for the monitor: this round the leased cores spent long
stretches in a slow all-core-turbo state (every compute-bound number +8..20%
with in-run sd < 0.1% while MKL moved < 1%: L=15 read 4.78/4.82/5.44 across
adjacent invocations of the SAME binary, L=12 read 2.19). All A/Bs above were
control-first adjacent pairs; the quiet-state minima are the honest numbers
and reproduce whenever the node calms down.

### What did NOT work, with the number that killed it

* **Interleaving c INTO the site at L=20** (site = re8|im8|cre8|cim8, 256 B,
  one x-pass stream instead of two; the fix gen_pfa_small's r2 record queued
  for the S+C = 2.05 MiB > L2 residency problem): **18.39 vs 13.17 control
  (+40%)** in a window only ~4% busier by MKL. Mechanism, best reading: the
  zy sweep's working span doubles (state lines interleave with c lines it
  never touches, 2.05 MiB span vs 1.02 MiB dense), thrashing L2 far worse
  than the second x-pass stream ever did. The idea is now measured and dead
  for this engine shape -- gen_pfa_small, do not build it.
* **vdivpd map tail at 15** (+10%) and **sched-pressure at 20** (+4..8%):
  numbers above, both per-size defaults set accordingly.
* Register-explicit form at L=20 was not attempted: 40 live site registers
  against 32 zmm is a guaranteed heavy spill; the memory-round-trip form's
  "dead stores" are not dead there (stage 2 genuinely reloads).

### Borrowed, plainly

- **gen_pfa_small r2**: the L=20 size itself, the Q == 1 mod P in-place
  disjointness rule, PL2=418, and their sched-doesn't-transfer observation
  (which is what made me audit the asm and find the out-lining).
- **gen_powp r2**: the map div-vs-ladder is engine-specific and must be
  re-raced per x-pass -- their verdict transfers to my L=20, not to 10/12/15.
- **gen_pow2 r1** (method): always_inline + count-the-stores asm audit before
  structural bets. The audit WAS this round's headline; do it every round.

### Operation count

Per pencil per 8 volumes (vector FP, FMA-contracted): 88 / 96 / 162 / 216 at
L=10/12/15/20, now with exactly 2L zmm loads + 2L zmm stores at 10/12/15
(L=15 spills a handful under pressure scheduling) and 4L + 4L at L=20.
Map: ~19 FMA + 2 seeds per site at 10/12/15; ~14 FMA + 1 seed + 1 vdivpd at
20. Zero shuffle-port ops inside the transform; pack/unpack unchanged.

### What I would do next (ranked)

1. **B=1 / small-batch lane-spatial engine** (third round on the list; still
   2.3x behind MKL at B=1 and now also 119 us at L=20 B=1). Matters only if
   round 6 draws small batches; coordinate with gen_pfa_small before building
   it twice.
2. **Round-3+ generality recipe** (not built this round, written down so the
   next round can): the register-pencil generator is mechanical for any
   coprime P*Q with P in {2,3,4} and Q in {5,7} up to ~28 sites (2 x 14 = 28
   site registers at L=14=2x7, 21=3x7, 28=4x7 needs the L=20 memory form);
   modules needed: DFT7 (Winograd, 36+16 form). Sizes 14/21/28/35 would then
   cost one slot-table derivation each; the in-place rule decides fused vs
   plain per size. Bluestein/planner cover existence meanwhile.
3. **L=20 residency, the honest version**: c streaming from L3 is ~40% of the
   step there. Interleaving is dead (above); the remaining candidates are a
   c-plane software pipeline staged through the pack buffer (prefetch is
   banned by five records, but a bulk 128B-granule copy of plane x+1's c into
   an L2-resident bounce buffer during plane x's sweep is not a prefetch uop
   tax, it is real work with real reuse) -- measure, do not assume.
4. If the xarch guard flags SPR/CLX: race -DBL_MAPDIV/-DBL_MAPRCP and the
   three sched knobs; the defaults encode Ice Lake verdicts only.

## Round gen_r4 -- the A/B protocol was broken: same-core interleaving flips the L=15 sched verdict

Standings into the round (r3 board): led 12 (1.915 vs pfa_small 1.970) and 20
(13.011 vs 13.257), tied 10 (1.157 vs 1.156), and SCORED 7% behind at 15
(4.771 at 2.3% spread vs their 4.464 -- my r3 dev number was 4.456, so that
cell smelled like window noise, and this round found the mechanism).

### The finding that drove the round: tryout.sh A/B pairs hop cores

`tryout.sh` acquires a FRESH slot lease per invocation. Consecutive
invocations of an A/B pair therefore often run on DIFFERENT leased cores, in
different turbo/neighbor states -- and this session the states differed by
10-25% while MKL (memory-bound) moved <2%. Every "control-first adjacent
pair" in my r1-r3 records (and, I suspect, in several peers') carries this
confound on top of the known time-bimodality. The fix, used for everything
below: hold ONE slot lease, build all variants side by side, and alternate
`--samples 4` invocations of the SAME binaries on the SAME core
(literature 10's interleaved best-of-N, applied at invocation granularity).
Six-pair interleaves gave sub-1% discrimination in windows where
cross-invocation tryout pairs contradicted themselves by 8%.

### What shipped (one real change; chain outputs bit-identical to r3)

**SCHED15 default flipped OFF.** Same-core interleave, six consecutive
pairs, fused map, all wins: stock scheduler 4.567-4.605 vs sched-pressure
attribute 4.767-4.880 us (**-4.3%**). r3 shipped the attribute at 15 on a
"+5.3% when stripped" cross-invocation measurement -- backwards. The r2/r3
history of this attribute at 15 (r1: -13% as global flags, r2: +4.3% when
stripped, r3: +5.3% when stripped) is now explained as core-state luck
stacked on the inline confound r3 found. `-DBL_SCHED15` re-enables it for
the cross-arch races. 10/12 RE-CONFIRMED the attribute honestly wins
(10: 1.159 attr vs 1.218 without, 12: 1.921 vs 2.037; settled-state
same-core pairs), and 20 re-confirmed OFF (13.29-13.50 vs 14.21-14.81 with
`-DBL_SCHED20`). Map tails re-confirmed same-core: rcp14 ladder at 15
(4.77 vs div 4.94-4.96), vdivpd at 20 (13.29 vs rcp 13.74-14.01). So every
r3 verdict EXCEPT SCHED15 survives the honest protocol.

### Built, raced, and rejected: the map_col epilogue (kept as BL_EPI knobs)

The r4 asm audit found the fused x-pencil at 15 spills hard (27 spill
stores + 15 spill loads + 12 folded rsp operands per pencil; L=12: 31 rsp
touches) while the map-free sweep pencils spill ZERO -- the fused map's ~7
temps + 4 constants on top of 30 live site registers are what overflow the
32 zmm. The targeted fix: `map_col`, a per-column epilogue loop right after
a spill-free plain DFT pencil (outputs still L1-hot/store-forwarded,
per-site arithmetic unchanged, outputs bit-identical -- verified by cmp).
Cross-invocation tryout pairs said it WON at 15 (-4.5%!) and 20 (-8.7%!),
which is what exposed the protocol problem: under same-core interleaving it
LOSES everywhere -- 15: 5.22-5.35 vs fused 4.78-4.92 (six pairs, +9%); 15
with the new no-sched default: 4.86-4.89 vs 4.58 (+6%); 12: 2.225 vs 2.184;
10: 1.31-1.49 vs 1.16; 20: fused wins 3 of 4 pairs. gcc's spill placement
inside the fused pencil beats a clean split plus 2L extra L1 round trips.
The knobs stay compilable (`-DBL_EPI10/12/15/20=1`) as cross-arch race
candidates -- CLX's smaller L1/turbo behavior could flip it.

### What else did NOT work

- **Scalar constants for embedded-broadcast FMA operands** (literature 10's
  pressure cure): gcc 11 generated a BYTE-IDENTICAL binary (zero `{1to8}`
  operands emitted from vector-extension code). Reverted; it needs
  intrinsics or a newer gcc, not worth the rewrite. Do not re-try on this
  toolchain.
- **gen_pow2's DSB/front-end check** (their r3 warning to peers with big
  unrolled bodies): audited, NOT my disease -- hot loop bodies are ~630
  instrs at 15 (x-pencil+map), ~380 at 20, all under the ~2.3K-uop DSB.
  Clean verdict, 15 minutes, worth repeating after any unroll change.

### Measured on the node (a80n0 core 4 via held lease, graded chains, same-core minima; this session's windows never reached r3's quietest state)

| case | r3 ship (same-session control) | r4 ship | same-window MKL 2022 | ratio |
|---|---|---|---|---|
| L=10 B=64 m=1000 | 1.158 | **1.157** (unchanged path) | 4.59 | 4.0x |
| L=12 B=64 m=600  | 1.921 | **1.919** (unchanged path) | 7.78 | 4.1x |
| L=15 B=32 m=600  | 4.767-4.880 | **4.567-4.589** (-4.3%) | 16.52 | 3.6x |
| L=20 B=32 m=256  | 13.17-13.50 | **13.394** (unchanged path) | 60.5 | 4.5x |

Gates (final shipped build, all run by hand on the node; identical error
numbers to r3 because the arithmetic is bit-identical): single call
2.6/2.9/3.1/3.0e-16; two-step m=2 8.2e-16/9.2e-16/1.2e-15/1.2e-15 (tol
3e-14); graded chains 1.687e-13/4.869e-14/5.249e-14/4.366e-14 vs anchors
1.081e-13/3.887e-14/4.784e-14/2.835e-14 (tol 1e-10); repeatable
bit-identical; B=1 and B=12 mixed group+remainder single and m=2 chains
PASS at all four sizes. B=1 chains 10.52/17.44/41.66/116.5 us (unchanged
gap, four rounds on the list now).

### Borrowed, plainly

- **gen_pow2 gen_r3**: the front-end/DSB audit request (run, clean) and the
  min-vs-median window discipline.
- **Literature 10**: the interleaved best-of-N timing protocol -- the
  round's real adoption; it re-decided one shipped default and killed a
  false positive that cross-invocation tryout numbers would have shipped.
- **gen_pfa_large gen_r3 / gen_powp gen_r3**: their map-placement negative
  results (ipe/ipm) framed the epilogue experiment as pencil-LOCAL rather
  than pass-global; the local version still lost, which closes map
  placement at ALL granularities for this engine on this node.

### Operation count

Unchanged from r3 everywhere (88/96/162/216 vector FP per pencil per 8
volumes; map ~19 FMA + 2 seeds per site at 10/12/15, ~14 FMA + 1 seed +
1 vdivpd at 20). This round moved no arithmetic -- it fixed a default that
scheduling luck had set wrong and repaired the measurement procedure.

### What I would do next (ranked)

1. **Tell the monitor / peers about the core-hop confound** (done here in
   the record): any per-size default in ANY entry that was set by a <8%
   cross-invocation tryout margin deserves a same-core re-race. My own
   remaining suspects were all re-raced this round.
2. **B=1 lane-spatial engine**: still the biggest hole (2.3x behind MKL),
   still only round-6-relevant, still better built once with gen_pfa_small
   (their split path already beats my replicated path 2.8x at B=1 --
   coordinate, adopt, or concede that batch regime to them in the planner
   routing).
3. **L=15 residual vs the 4.46 quiet floor**: with sched luck removed, the
   honest gap to the port-utilization floor (~3.8 us) is spills (~54 rsp
   ops/x-pencil) + seeds' port-0 tax. A hand-scheduled two-block stage-2
   (a_/b_ in regs, c_ spilled deliberately) might shave a few of the 27
   spill stores; expected value ~1-2%, an afternoon.
4. **Cross-arch races**: the knob set now covers map form, map placement,
   and all three sched attributes; when XARCH.md lands, race exactly those
   six axes per host rather than re-deriving.

## Round gen_r5 -- L=15 goes back to the memory form, and its map tail flips to vdivpd

Standings into the round (r4 board): tied 10 (1.154 vs pfa_small 1.153; my 7.9%
spread was neighbor noise), led 12 (1.912 vs 1.969), TRAILED 15 by 3.1% (4.566
vs 4.429), trailed 20 by 0.7% (13.145 vs 13.059). gen_pfa_small's r4 record
addressed me directly: their same-core A/B measured MY register-explicit 15
form at +12.6% against their r2-lineage memory form (5.02 vs 4.44-4.46), with
the rule "register-explicit pays only when 2L site regs + module temps fit
~32 zmm". This round is the audit of that claim on my own engine, plus the
map-tail race it exposed.

### What shipped (two changes at L=15; 10/12/20 paths untouched, outputs there
### bit-identical to r4)

1. **L=15 pencils REVERTED to the r2 memory form** (in-place M3IP stage-1
   DFT3s; stage-2 memory DFT5 with the fused load-both-store-both DFT5X2 pair
   for the equal-slot-set groups c=1,2). ADOPTED BACK on gen_pfa_small gen_r4's
   evidence -- their rule is confirmed on my build: 15's 30 live site registers
   + DFT5CORE/map temps spill past 32 zmm, and the memory form's stage-1 stores
   ride ICL's 2-stores/cycle ports at near-zero cost. Same arithmetic order, so
   the form knob is bit-transparent (verified by cmp at single and m=2).
   Knob: -DBL_MEM15=0/1/2 (register / memory / hybrid).
2. **L=15 map tail flipped to ONE exact vdivpd** (MAPTAIL_15, new). On the
   memory-form pencil the rcp14+2NR ladder's 5 extra FMA-port ops cost more
   than the unpipelined divide the pencil leaves idle -- exactly
   gen_pfa_small r3's "the div-vs-rcp verdict is a property of the SURROUNDING
   CODELET" lesson, now measured in both directions on my own entry.

### The same-core race that decided it (held slot lease core 3, alternating
### --samples 4 invocations, 4-6 rounds each; first invocation of each session
### discarded as warmup -- it reads +10-14% every time)

| variant (L=15 B=32 m=600) | min-of-mins | verdict |
|---|---|---|
| register + rcp ladder (r4 ship, control) | 4.567 | baseline |
| register + vdivpd | 4.419 | -3.2% (r4's opposite verdict did not reproduce) |
| memory + rcp ladder | 4.582 | wash vs control |
| **memory + vdivpd (SHIPPED)** | **4.411** | **-3.4%, wins 3 of 4 head-to-heads vs register+div** |
| hybrid (reg sweeps + mem map x-pass) + div | 4.427 | +0.35% vs shipped -- rejected |
| memory + div + sched-pressure | 4.892 | +10.9% -- pfa_small's r2 verdict confirmed |

The FORM was never the whole 3%: register and memory tie at equal map tails
(4.419 vs 4.411). The board gap was mostly the MAP TAIL, which r4's same-core
race got backwards at 15 (it measured register+div 4.94 vs register+rcp 4.77
on core 4; today's core 3 reads register+div 4.42 vs register+rcp 4.57, six
consecutive pairs, sd < 0.1%). Two honest same-core sessions, opposite
verdicts: the div/rcp margin at 15 is CORE-STATE-DEPENDENT on this node.
Shipped memory+div anyway because (a) it wins today's window outright, (b) it
is the exact form gen_pfa_small scored 4.429 with under real scoring
conditions, and (c) BL_MAPRCP stays raceable if XARCH or a future window
disagrees. The hybrid's loss is a real (small) negative result: the register
sweep pencil's halved load/store count does NOT beat the memory sweep pencil's
ILP on the L1-resident zy planes.

### Re-raced and kept (same session, same core)

- **10/12 map tail**: rcp ladder confirmed AGAIN (10: 1.156 rcp vs 1.167 div;
  12: 1.916 rcp vs 1.957 div, +2.1% -- four pairs each). The register pencils
  keep the FMA ports saturated; the memory pencils do not. Per-size defaults
  now: rcp at 10/12, div at 15/20 -- consistent with the form split.
- 10/12 register-explicit + sched-pressure, 20 memory + stock + div: untouched,
  end-of-session confirmations 1.157 / 1.915 / 13.04-13.09.

### Measured on the node (a80n0 core 3, quiet window, min over 3 end-of-session
### runs of the shipped binary; MKL 2022 same core same window)

| case | r4 ship (board) | gen_r5 ship | MKL | ratio |
|---|---|---|---|---|
| L=10 B=64 m=1000 | 1.154 | **1.157** (unchanged path) | 4.552 | 3.9x |
| L=12 B=64 m=600  | 1.912 | **1.915** (unchanged path) | 7.738 | 4.0x |
| L=15 B=32 m=600  | 4.566 | **4.410-4.413** (-3.4%) | 16.461 | 3.7x |
| L=20 B=32 m=256  | 13.145 | **13.043-13.091** (unchanged path) | 58.883 | 4.5x |

Gates (shipped build, run by hand on the node; tryout.sh's map-check leg still
passes the literal '$W/c.bin'): single call 2.6/2.9/3.1/3.0e-16; two-step m=2
8.2e-16 / 9.2e-16 / 1.198e-15 / 1.2e-15 (tol 3e-14, 25x margin -- the 15 div
tail moved it from 1.196e-15, i.e. nothing); graded chains 1.687e-13 /
4.869e-14 / 5.208e-14 / 4.366e-14 vs anchors 1.081e-13 / 3.887e-14 /
4.784e-14 / 2.835e-14 (tol 1e-10); bit-repeatable at all four sizes; 10/12/20
chain outputs bit-identical to r4. Remainder paths: B=1 single + m=2 PASS at
all four sizes, B=12 mixed group+remainder at 12 PASS, B=9 at 15 (memory-form
remainder lanes) PASS. B=1 chains (m=64): 10.65 / 17.59 / 40.53 / 133.97 us
-- the five-round-old gap, unchanged.

### What did NOT work, with the number that killed it

- **Hybrid form at 15** (register sweeps + memory map x-pass, my one new
  structural idea this round): 4.427-4.442 vs memory 4.411-4.430, +0.35%,
  five consecutive interleaved rounds. Cheap to build off the knob, honestly
  dead.
- **sched-pressure on the memory-form 15**: 4.892-4.910 vs 4.411 (+10.9%).
  Now measured on BOTH engines (pfa_small r2: +3-10%); closed.
- **rcp ladder on the memory-form 15**: +4.1% (4.59 vs 4.41). Closed with the
  form.

### Borrowed, plainly

- **gen_pfa_small gen_r4**: the round's headline -- their register-fits-32-zmm
  rule and their direct "A/B the memory form back at 15" instruction, both
  correct; and their r3 codelet-local div-vs-rcp lesson, which is what made me
  race the tail ON the new form instead of porting the r4 verdict.
- **Literature 10 / my own r4**: the held-lease interleaved protocol, used for
  every number above.
- Checked and NOT taken: gen_pow2 r4's GP2_CT (c in last-pass walk order) --
  gen_pfa_small r3 already measured consumption-order c at +4-10% on this
  engine shape (the natural plane-major c already streams as L sequential
  per-plane streams); gen_powp/gen_pfa_large r4's volume-major chain -- my
  chain has been group-major (8 volumes through all m steps) since r1, same
  residency effect.

### Operation count

FP unchanged (88/96/162/216 vector instrs per pencil per 8 volumes). L=15 is
back to 4L ld + 4L st per pencil (memory form; the 2L + 2L register form ties
at best and spills under the map). Map at 15 is now ~14 FMA + 1 rsqrt14 seed +
1 vdivpd per site (was ~19 FMA + 2 seeds); 10/12 unchanged rcp ladder; 20
unchanged div.

### What I would do next (ranked)

1. **The div/rcp core-state dependence deserves a wisdom-race axis**: two
   same-core sessions on different cores gave opposite 15-tail verdicts at
   the +/-3% level. If gen_race can afford one extra candidate per size, the
   map-tail knobs (-DBL_MAPDIV/-DBL_MAPRCP, -DBL_MEM15) are exactly the axes
   to race per host -- this is what the round-6 surprise sizes will hit.
2. **B=1 lane-spatial engine** (fifth round on the list, unchanged): 2.4x
   behind MKL at B=1, 10/12/15. Round 6 draws unknown batches. gen_pfa_small
   and I keep sketching the same ice L6_pfa shape at each other -- whoever
   moves first, the other adopts.
3. **L=15 residual vs the ~3.8 us port floor**: with the form and tail settled
   at 4.41, the remaining gap is the DFT5's op count. The 3-FMA lifting form
   (literature 08 6.3) cuts ~10% of DFT5 FP; ice measured 11.9% ops -> 0.8%
   time, so budget an hour, expect ~0.5%, verify the 1.5e-14/step budget
   (reassociation changes rounding).
4. **L=20**: identical structure to gen_pfa_small's, cell decided by window
   luck at the 0.5% level. The only untested idea remains the c bounce buffer
   (bulk-copy plane x+1's c during plane x's sweep); both records predict it
   loses; only worth a quiet-window hour if the cell matters for the final
   geomean.
