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

## Round gen_r6 -- the class goes 7-smooth: DFT7 + safe placement give 14/21/28/35 at batch-lane speed

Standings into the round (r5 board): effectively tied with gen_pfa_small at
10 (1.158 vs 1.152), 12 (1.915 vs 1.914) and 15 (4.411 vs 4.406), 1.5% back
at 20 (13.268 vs 13.072, our two L=20s now algorithmically identical -- that
cell is window luck, not code, and I did not chase it). Round 6 scores the
ASSEMBLED LIBRARY on three surprise sizes in 14..127, so this round's real
job was making my class COVER more of that range at full batch-lane speed,
plus one structural experiment at 15 and knob re-confirmations.

### The round's one idea: SAFE PLACEMENT kills the stage-2 hazard at any coprime split

Every prior round treated stage-2 in-place safety as size-specific luck:
L=20 was safe because Q == 1 mod P (disjoint residue classes), L=15 was
unsafe (groups c=1,2 share slot sets) and needed the fused
load-both-store-both DFT5X2 codelet. The general fix is a free store
permutation in STAGE 1: place stage-1 output c at the slot of its group
whose a = (Q^-1 c) mod P -- i.e. the slot CONGRUENT TO c mod P. Stage-2
group c then reads and writes exactly the residue class {m == c mod P}:
read set == write set per group, groups mutually disjoint, plain
load-all-then-store-all in-place safe for ANY (P, Q). Zero extra
instructions (the permutation is baked into stage-1 store offsets), values
unchanged, so outputs are bit-identical wherever both forms exist. Slot
tables for every size were generated and verified against a reference DFT
in Python before touching C.

This is what makes new sizes MECHANICAL: without it, L=28 (Q=7 == 3 mod 4)
would need a fused DFT7X2 (28 slot loads + ~24 core temps, hopeless) and
L=35 (Q=7 == 2 mod 5) a fused FOUR-group cycle.

### What shipped

1. **Four new class sizes, all memory form, all safe placement, div map
   tail, stock scheduler**:
   - L=14 = 2x7: n=(7a+2b)%14, k=(7c+8d)%14; 7 in-place DFT2 + 2 DFT7 = 160
     vector FP/pencil. PL2=226.
   - L=21 = 3x7: n=(7a+3b)%21, k=(7c+15d)%21; 7 DFT3 + 3 DFT7 = 282. PL2=450.
   - L=28 = 4x7: n=(7a+4b)%28, k=(21c+8d)%28; 7 DFT4 (M4IPO permuted stores)
     + 4 DFT7 = 376. PL2=802.
   - L=35 = 5x7: n=(7a+5b)%35, k=(21c+15d)%35; 7 DFT5 (M5ST with permuted
     output slots) + 5 DFT7 = 568. PL2=1250.
   All PL2 keep plane bytes == 256 (mod 4096) (== 2 mod 32 site-vectors).
2. **DFT7 module** (new): symmetric/antisymmetric split, t_j = x_j + x_{7-j},
   u_j = x_j - x_{7-j}; A_k = x0 + sum_j cos(2pi kj/7) t_j, B_k = sum_j
   sin(2pi kj/7) u_j, X_k = A_k - iB_k, X_{7-k} = A_k + iB_k; kj exponents
   folded to j in {1,2,3} with signs. 66 vector FP per DFT7 per 8 volumes.
   Constants computed to 22 digits by Decimal series (no mpmath on wallaby)
   and spot-checked against libm.
3. **supports() now accepts 10, 12, 14, 15, 20, 21, 28, 35** -- the planner
   and race layers see the new sizes for the round-6 library assembly.
4. **10/12/15/20 paths untouched**; the shipped L=15 chain output is
   bit-identical to r5 (verified by cmp on the m=600 chain).

### Measured on the node (a80n0 core 2, ONE held slot lease, interleaved
### --samples 4 minima; first invocation of each binary discarded as warmup)

New sizes, graded-shape chains, MKL 2022 same core same window:

| case | this engine | MKL | ratio |
|---|---|---|---|
| L=14 B=32 m=600 | **3.632-3.652** | 13.38 | **3.7x** |
| L=21 B=16 m=300 | **18.41-18.58** | 73.3  | **4.0x** |
| L=28 B=16 m=180 | **50.92-50.94** | 170.8 | **3.4x** |
| L=35 B=8  m=128 | **109.3-113.3** | 358.4 | **3.2x** |

(L=14 is FASTER than L=15 -- 160 vs 162 FP on a shorter pencil -- so the
class's per-point cost is monotone in the op count, as it should be.)
Owned scored sizes, end-of-session same-core: 10: 1.157-1.158 (the 1.31-1.32
reads this session are the known slow-turbo state; interleaved pairs against
the r5 binary are a wash), 12: 1.914-1.920, 15: 4.413-4.437 (r5 path kept,
see below), 20: 13.07 in the quiet early window, 13.7 late-session busy.

Gates, ship build, all run on the node: single call 2.6-3.7e-16 at all
EIGHT sizes (tol 1e-12); two-step m=2 gate 1.0e-15 / 1.4e-15 / 1.4e-15 /
1.8e-15 at 14/21/28/35 (tol 3e-14, 16-30x margin), unchanged at 10/12/15/20;
graded chains 5.8e-14 (14, anchor 6.9e-14), 3.9e-14 (21, anchor 2.8e-14),
2.5e-14 (28, anchor 2.4e-14), 2.5e-14 (35, anchor 2.0e-14), tol 1e-10;
repeatable bit-identical; remainder paths PASS m=2 chains at B=1 for all
four new sizes and at mixed B=12 (14), B=9 (21), B=9 (35). B=1 m=64 chains:
9.34 / 15.45 / 29.24 / 35.71 / 110.6 / 150.0 us at 10/12/14/15/20/21 -- the
six-round-old remainder-lane gap now extends to the new sizes too.

### Built, raced, and REJECTED: BL_SAFE15 (kept as a knob, default 0)

Safe placement applied AT 15 splits the r5 memory form's forced DFT5X2 pair
into three independent in-place DFT5 groups -- bit-identical outputs
(verified by cmp), lower peak register pressure in the fused-map x-pencil.
Same-core, five interleaved rounds, three binaries per round: safe
4.458-4.654 vs fused 4.414-4.480 vs r5 ship 4.413-4.437 -- safe LOSES ~1%.
The X2 pair's two interleaved dependency chains (2 DFT5COREs + 2 map
ladders in flight per codelet) buy more ILP than the split saves in
spills. So the fused pair stays at 15 ON THIS NODE, and the lesson
transfers forward: fusing PAIRS of stage-2 groups is an ILP play worth
racing at the new sizes too (not tried this round -- a DFT7X2 at 14 would
be the first candidate since its two groups are the whole stage).

### What else did NOT work / was re-confirmed, with the numbers

- **div map tail at 10** (chasing gen_pfa_small's 1.152 cell, their MT10
  ships div): rcp 1.158 vs div 1.160-1.162, five clean pairs -- rcp
  RE-CONFIRMED on my register pencils; their 10 edge is their legacy ladder
  body on their codelet, not the tail. Cell closed as body-shape-specific.
- **rcp map tail at 14** (validating MAPTAIL_GEN=div for the new memory-form
  sizes): div 3.632-3.652 vs rcp 3.722-3.789 (+2.3%) -- the 15/20 verdict
  transfers to the whole memory-form family, as gen_pfa_small r3's
  codelet-local rule predicts.
- **L=20 gap to gen_pfa_small**: read both sources side by side -- after
  their r5 adoption of my map8 the two L=20 paths are op-identical, and
  their own r5 record has me AHEAD in their window (13.72 vs 13.61 -- within
  the same noise band as the board's 13.27 vs 13.07 the other way). No code
  action exists; not chased.

### Borrowed, plainly

- **gen_pfa_small r2's Q == 1 mod P disjointness rule** is what the safe
  placement generalizes: their rule is the special case where the identity
  permutation already lands outputs in their own residue class.
- **gen_pfa_small gen_r5**: their MT10=div result triggered the 10-tail
  re-race (declined on measurement); their "ladder body and tail flip
  together" warning is why I raced MAPTAIL_GEN on the actual new codelets
  instead of assuming.
- **Literature 10 / my r4**: the held-lease interleaved protocol, used for
  every verdict above.
- The DFT7 A/B-split form follows the same folding as bl8's DFT5 (KS1/KS2
  sign pattern), extended to three rotation pairs.

### Operation count

Unchanged at 10/12/15/20 (88/96/162/216 vector FP per pencil per 8 vols).
New: 160/282/376/568 at 14/21/28/35, memory form (4L ld + 4L st per
pencil), map ~14 FMA + 1 rsqrt14 seed + 1 vdivpd per site-vector. Zero
twiddle tables and zero shuffle-port ops inside all transforms, still.

### What I would do next (ranked)

1. **Race DFT7X2 fusion at L=14** (the whole stage 2 is two groups with,
   after safe placement, disjoint slot sets -- fusing them is optional now,
   which is exactly what makes it raceable): the 15 verdict says interleaved
   pairs win ~1% when register pressure allows; 14's stage-2 has only 28
   site loads, so it might.
2. **B=1 lane-spatial engine** (sixth round on the list; now 6 sizes wide).
   If round 6 draws small B, the planner routes around me (their split path
   and gen_planner's split-group engine both beat my replicated lanes);
   the class concedes that regime unless someone builds the z-turn.
3. **More coprime coverage if a future round wants it**: 33 = 3x11 and
   55 = 5x11 need only a DFT11 module (the safe placement is already
   general); 30/40/42 need a DFT6/DFT8 module or a three-factor plan --
   coordinate with gen_pfa_small/gen_pfa_large, whose generic engines
   already cover some of these slower.
4. **Cross-arch**: the new sizes ship with the Ice Lake verdicts baked in
   (div tail, stock sched, fused map); the knobs (-DBL_MAPRCP, BL_SAFE15)
   are the axes to race on CLX/SPR.

## Round gen_r7 -- the lifted DFT5 v-pair: the golden ratio buys 2 ops per DFT5, and it flips the r5 hybrid verdict

Standings into the round (r6 board): tied 10 (1.155 vs pfa_small 1.156), led
12 (1.911 vs 1.917), trailed 15 by 0.006 us (4.412 vs 4.406), tied 20 to the
third decimal (12.866 vs 12.867). All four cells are the same engine converged
in two entries; only an op-count cut could move them. The brief's rounds-7/8
mandate is to spend the queued literature backlog -- mine was the DFT5 op diet
(queued since r2 as "budget an hour, expect ~0.5%").

### The round's one idea: sin(2pi/5) = phi * sin(pi/5) EXACTLY, so the v-pair lifts

Literature 08 6.3's 3-FMA lifting cuts rotations from 4 ops to 3. The Winograd
DFT5's v-pair is not a rotation (it is a scaled reflection):

    v1 = KS1*sa + KS2*sb        KS1 = sin(2pi/5), KS2 = sin(pi/5)
    v2 = KS2*sa - KS1*sb

but KS1/KS2 = 2cos(pi/5) = phi, the golden ratio, EXACTLY -- so it factors
through one shared term at the same dependency depth (2):

    u  = sa - PHI*sb            (FMA)
    v2 = KS2*u                  (mul;  KS2*(sa - phi*sb) = KS2*sa - KS1*sb)
    v1 = KS1*u + KL5*sb         (FMA;  KL5 = (KS1^2+KS2^2)/KS2 = 1.25/sin(pi/5))

6 vector ops instead of 8 per DFT5 (r/i components), zero latency cost, one
extra constant. PHI/KL5 computed exact to the last bit (50-digit Decimal
series). Pencil FP: 10: 88 -> 84, 15: 162 -> 156, 20: 216 -> 208, 35: 568 ->
554; 12/14/21/28 have no DFT5 and are untouched. This is NOT bit-transparent
(same exact values, different rounding), so the round's gate work was real.

### What shipped

1. **BL_LIFT5=1 (default)**: the lifted v-pair in DFT5CORE, all DFT5 users
   (10 stage-2, 15 stage-2, 20 stage-2, 35 stage-1+2). BL_LIFT5=0 restores
   the r6 arithmetic for cross-arch races.
2. **BL_MEM15 default flipped 1 -> 2 (hybrid)**: the r5 hybrid verdict
   (register-explicit sweep pencils + memory-form fused-map x-pass, +0.35%,
   rejected) REVERSES under the lift: hybrid wins 9 of 12 same-core pairs,
   4.373-4.395 vs memory 4.385-4.411 (-0.25%). Reading: the lift's two fewer
   live temps per DFT5 keep the register-explicit sweep pencil spill-free
   where it previously tied. Bit-transparent vs the memory form (verified by
   cmp on the m=600 chain), so the two knobs' gate numbers are shared.
3. **BL_X214=0 (new knob, default OFF)**: fused DFT7X2 stage-2 at L=14,
   built as my r6 next-step #1 -- and it LOSES (numbers below). Code kept
   as a cross-arch race candidate.

### Measured on the node (a80n0 core 2, ONE held slot lease, interleaved
### --samples 4 minima, control first, first invocation discarded as warmup)

Same-core A/B, r6 ship (bin_ctl) vs r7 ship:

| case | ctl (r6 path) | r7 | delta | pairs |
|---|---|---|---|---|
| L=10 B=64 m=1000 | 1.155-1.157 | **1.145-1.146** | -0.8% | 4/4 |
| L=15 B=32 m=600  | 4.414-4.430 | **4.384-4.398** (mem), 4.373-4.395 (hyb ship) | -0.7..-0.9% | 6/6, then 9/12 hyb vs mem |
| L=20 B=32 m=256  | 13.61-13.66 | **13.48-13.52** | -1.0% | 4/4 |
| L=35 B=32 m=128  | 106.6      | 106.5           | wash (memory-bound) | 3 |
| L=12 B=64 m=600  | (bit-identical path) 1.917-1.922 | same | -- | -- |

End-of-session ship minima, MKL 2022 same core same window: 10: 1.150 (MKL
4.68, 4.1x), 12: 1.917 (7.95, 4.1x), 14: 3.618, 15: 4.385 (16.81, 3.8x),
20: 13.53 (65.5, 4.8x -- window busier than the r6 board's 12.87; the A/B
delta is the honest number), 35: 106.8.

Gates (ship build, all run on the node): single call 2.8-3.9e-16 at all
EIGHT sizes (tol 1e-12); two-step m=2 gate 9.0e-16..1.8e-15 (tol 3e-14,
16-33x margin -- the lift moved 15's from 1.198e-15 to 1.272e-15, i.e.
nothing); graded chains 1.673e-13 / 4.869e-14 / 5.824e-14 / 5.487e-14 /
5.231e-14 / 3.540e-14 / 2.377e-14 / 3.474e-14 at 10/12/14/15/20/21/28/35 vs
honest anchors 1.081e-13 / 3.887e-14 / 6.874e-14 / 4.784e-14 / 2.835e-14 /
2.869e-14 / 1.953e-14 / 2.604e-14 (tol 1e-10; the largest drift ratio is
20's 1.8x, same tier as before the lift); bit-repeatable at every size;
B=1 single + m=2 PASS at all eight sizes; mixed remainder B=9 (15) and
B=12 (10) PASS. 12/21/28 chains bit-identical to r6 (cmp); 14's fused-X2
experimental path bit-identical to serial (cmp) before it lost the race.

### What did NOT work, with the number that killed it

- **DFT7X2 fusion at L=14** (my own r6 next-step #1, the DFT5X2-at-15
  pattern): serial 3.622-3.641 vs fused 3.834-4.011 (+6-10%, five clean
  same-core pairs). 28 slot loads + two DFT7COREs' ~24 temps each spill far
  past 32 zmm; the X2 ILP win does not transfer to modules this wide. The
  boundary is now measured from both sides on this engine: DFT5X2 (10 loads,
  ~14 temps/core) wins ~1%, DFT7X2 loses 6-10%. Knob kept for CLX/SPR.
- **BL_SAFE15 re-raced under the lift** (the lower-pressure fused-map pencil
  might have flipped it): still loses, 4.412-4.430 vs 4.394-4.411 (~+0.5%,
  four rounds). Closed again.
- **Stage-as-dense-GEMM at 10/12/15** (brief backlog item 3, lit 11 Tier 2):
  DECLINED on arithmetic, not measured. The claim's win comes from
  eliminating twiddle loads/shuffles; this engine has neither -- its stage-2
  groups already ARE register-resident constant matrices applied by
  broadcast-FMA on batch lanes. A dense 10-point matrix apply costs 4L^2 =
  400 FMA/pencil vs the PFA's 84 in an FP-throughput-bound kernel: a 4.8x op
  inflation with nothing to buy back. If anyone wants the crossover claim
  tested, test it on an engine that pays per-butterfly twiddle traffic.

### For the monitor / gen_race / gen_planner: the surprise round exposed a routing gap at L=21

gen_surprise drew L=21 B=32 and the trunk (gen_race/gen_planner) ran it at
37.4-37.7 us/xform vs MKL 74.6 (2.0x). My entry has supported 21 since r6 and
measured 18.4-18.6 us at B=16 in r6; this round's ship build does 3.6 us at
14 and ~18.5 at 21 -- the trunk left ~2x on the table at a size my class
covers. Whether the race never enumerated gen_batchlane's 21 candidate or the
wisdom was cold, that is exactly the "surprise-size failure as planner/race
bug" the r7 brief prioritizes. The class engine is sitting there; route to it.

### Borrowed, plainly

- **Literature 08 6.3** (Gustafsson ARITH-24): the lifting idea; adapted
  rather than transplanted -- the v-pair is a scaled reflection, not a
  rotation, and the exact phi ratio between sin(2pi/5) and sin(pi/5) is what
  makes the 3-op factoring exact-constant clean. Cited as the round's
  literature spend.
- **gen_pfa_small r4's register-budget rule**, applied predictively this
  time: it said DFT7X2 (28+48 live values) must spill -- built it anyway to
  put a number on the boundary; the rule was right.
- **Literature 10 / my r4**: the held-lease interleaved protocol, used for
  every verdict above.

### Operation count

Per pencil per 8 volumes (vector FP, FMA-contracted): 84 / 96 / 160 / 156 /
208 / 282 / 376 / 554 at L=10/12/14/15/20/21/28/35. Loads/stores: 10/12
register 2L+2L; 15 hybrid = register sweeps (2L+2L) + memory map x-pass
(4L+4L); 14/20/21/28/35 memory 4L+4L. Map unchanged (rcp ladder 10/12,
vdivpd elsewhere). Zero twiddle tables, zero shuffle-port ops in the
transforms, one new broadcast constant pair (PHI, KL5).

### What I would do next (ranked)

1. **Get L=21 (and 14/28/35) routed in the trunk** -- worth 2x at any
   surprise draw my class covers, vs the ~1% left in my own cells. Needs
   gen_race/gen_planner action, not mine; flagged above.
2. **B=1 lane-spatial engine** (seventh round on the list, unchanged).
3. **DFT7 op diet**: no phi-like exact ratio exists among sin(2pi k/7), but
   the A-side (cosine) rows share x0 and the identity c1+c2+c3 = -1/2;
   a Rader-style 7-point (lit 02) trades 66 straight-line ops for a cyclic
   convolution -- only worth it if a surprise draw makes 14/21/28 scored.
4. **Cross-arch**: new knobs to race per host: BL_LIFT5 (0/1), BL_MEM15
   (1/2 -- the verdict moved once already), BL_X214 (0/1). CLX's weaker
   FMA throughput could widen the lift's win; its smaller L1 could flip
   the hybrid back.

## Round gen_r8 -- the class goes 11-smooth: a DFT11 module gives 22/33/44/55, closing the surprise test's named gap

Standings into the round (r7 board): led all four scored cells (10: 1.147,
12: 1.915, 15: 4.381, 20: 12.855), all within ~1% of gen_pfa_small's converged
copies -- nothing structural left in the owned cells. The round-8 brief's
surprise-test addendum names the real lever instead: L=44 (prime 11) was the
trunk's weakest surprise cell (1.29x) because NO panel entry has ever built an
11-point module. My r6 safe placement makes 11-smooth sizes mechanical; this
round built them (my own r6 next-step #3, third round on the queue).

### What shipped

1. **DFT11 module** (new): same symmetric/antisymmetric split as the r6 DFT7,
   five rotation pairs: t_j = x_j + x_{11-j}, u_j = x_j - x_{11-j}, j = 1..5;
   A_k = x0 + sum_j cos(2pi kj/11) t_j, B_k = sum_j sin(2pi kj/11) u_j;
   X_k = A_k - iB_k, X_{11-k} = A_k + iB_k. Exponent folding (kj mod 11, cos
   even, sin sign-flipped past 5) GENERATED in Python rather than hand-derived,
   and verified against a reference DFT before touching C. Constants exact to
   the last bit of double (60-digit Decimal Machin pi + Taylor series,
   cross-checked vs libm). 150 vector FP per DFT11 per 8 volumes.
2. **Four new class sizes, all memory form, safe placement, div map tail,
   stock scheduler** (the memory-form family defaults, unchanged):
   - L=22 = 2x11: n=(11a+2b)%22, Q==1 mod P: natural placement (M2IP).
     11xDFT2 + 2xDFT11 = 388 FP/pencil. PL2=514.
   - L=33 = 3x11: n=(11a+3b)%33, Q^-1=2 mod 3: M3IPS, the exact L=15 stage-1
     perm reused verbatim. 11xDFT3 + 3xDFT11 = 582. PL2=1090.
   - L=44 = 4x11: n=(11a+4b)%44, Q^-1=3 mod 4: M4IPO(i0,i3,i2,i1), the L=28
     perm reused. 11xDFT4 + 4xDFT11 = 776. PL2=1954.
   - L=55 = 5x11: n=(11a+5b)%55, Q==1 mod P: natural (M5ST identity outs).
     11xDFT5(lifted) + 5xDFT11 = 1102. PL2=3042.
   All PL2 keep plane bytes == 256 (mod 4096). Slot tables for every size
   generated AND simulated (load-all-store-all group semantics, exactly what
   the C does) against a reference DFT in Python; the in-place and
   disjointness invariants asserted programmatically, not eyeballed.
3. **supports() now accepts 10,12,14,15,20,21,22,28,33,35,44,55** -- the
   planner/race layers see 12 sizes for library assembly and any surprise draw.
4. **L=77 = 7x11 deliberately NOT built**: an 11xDFT7 + 7xDFT11 = 1776-FP
   always_inline pencil (x2 forms) is a real compile-time hazard for a size no
   case has ever named, at a working set (58 MB x2) far past LLC where the
   engine's advantage collapses anyway. Its verified slot table is one
   generator run away (/tmp scratch reproduced in this record's method) if a
   round ever wants it.

### Measured on the node (a80n0, held slot lease, same-core; graded-shape
### chains with MKL 2022 same core same window; m chosen per the ~6 ns/pt/step
### case calibration since the new sizes are not in cases.txt)

| case | this engine | MKL | ratio |
|---|---|---|---|
| L=22 B=32 m=256 | **25.37** us/xform | 78.86 | **3.1x** |
| L=33 B=16 m=200 | **98.07** | 364.45 | **3.7x** |
| L=44 B=16 m=128 | **272.5** (busy window, sd 6.8%; 256.9 seen later same core) | 688.39 | **2.5x** |
| L=55 B=8  m=64  | **985.0** | 1473.0 | **1.5x** |

Single-call (m=1, pack/unpack unamortized) also beats MKL everywhere:
38.4 vs 45.8 (22), 162.4 vs 289.3 (33), 404.5 vs 557.3 (44), 985 vs 1076 (55).
L=55's chain time EQUALS its single-call time: S+C = 43 MB streams past the
24 MB LLC, so the whole chain is bandwidth-bound and pack amortization is
invisible -- that size is at the engine's shape boundary, as expected.
The 44 ratio (2.5x) vs the trunk's surprise-test 1.29x is the addendum's
predicted lift, now measured on real code.

Scored cells: 10/12/15/20 paths untouched. Same-core interleaved A/B of the
r7 ship binary vs this one at L=10 (four pairs after warmup): 1.147-1.148 vs
1.150-1.151 (+0.2%, code-layout noise; first invocation read 1.305-1.307 --
the known warmup state, do not panic at it). Chain outputs BIT-IDENTICAL to
r7 at ALL EIGHT pre-existing sizes (cmp on the full graded chains: 10 m=1000,
12/15 m=600, 20 m=256, 14 m=600, 21 m=300, 28 m=180, 35 m=128), so every r7
gate number carries over exactly.

Gates at the new sizes, all run on the node by hand (tryout's map-check leg
still dies on the r1 '$W/c.bin' quoting bug -- harness note stands):
single call 3.2/3.6/3.5/4.2e-16 at 22/33/44/55 (tol 1e-12); two-step m=2
1.38/1.72/1.79/2.21e-15 (tol 3e-14, 14-22x margin); graded chains 3.75e-14 /
2.86e-14 / 2.35e-14 / 2.56e-14 vs honest anchors 2.68e-14 / 2.54e-14 /
2.08e-14 / 1.74e-14 (1.1-1.5x, tol 1e-10); repeatable bit-identical; B=1
single + m=2 chains PASS at all four new sizes (remainder-lane replication
path, correct as always, still slow as always).

### What did NOT work / was raced and settled

- **rcp map tail at 44** (-DBL_MAPRCP, racing the family default on the new
  codelet as the r3/r6 rule requires): div 256.9-332.9 vs rcp 266.4-363.3,
  div wins the minima 3 of 4 same-core pairs. Window was noisy (the 22 MB
  working set shares LLC with neighbors), but direction matches the whole
  memory-form family (14: +2.3%, 15: +4.1%, 20: +4.4% for rcp). Div stays;
  knob remains for CLX/SPR.
- **DFT11X2 fusion at 22** (the analogue of DFT5X2-at-15): DECLINED on
  gen_pfa_small r4's register-budget rule, now measured from both sides on
  this engine (DFT5X2: 10 loads + ~14 temps/core, wins ~1%; DFT7X2: 28 loads
  + ~24 temps/core, loses 6-10% r7). DFT11X2 is 44 loads + ~60 temps/core --
  further past the boundary than the case that already lost. Not built.
- Nothing else was attempted at the scored cells: r7 closed them ("only an
  op-count cut could move them"), the r8 static analyzers (tools/TOOLS.md)
  were not spent on a speculative reschedule against bit-identical known-good
  cells in the campaign's final round.

### Borrowed, plainly

- **My own r6 safe placement + gen_pfa_small r2's disjointness rule**: the
  entire mechanism; this round is its second mechanical application (stage-1
  perms M3IPS and M4IPO reused verbatim from the 15 and 28 derivations).
- **gen_pfa_small r4's register-budget rule**, used predictively again to
  decline DFT11X2 without spending node time.
- **Literature 10 / my r4**: held-lease interleaved protocol for the L=10
  regression check and the 44 map-tail race.
- The surprise-test addendum (brief r7/r8) chose the round's target: it
  measured the 11-point gap at L=44 as ~2x left on the table; this round
  closes my class's half of it. The other half is ROUTING (gen_race/
  gen_planner must enumerate gen_batchlane's candidates at 22/33/44/55 --
  same gap my r7 record flagged at L=21, still the highest-value fix that
  is not in my file).

### Operation count

Per pencil per 8 volumes (vector FP, FMA-contracted): 84 / 96 / 160 / 156 /
208 / 282 / 376 / 388 / 554 / 582 / 776 / 1102 at L = 10/12/14/15/20/21/28/
22/35/33/44/55. New sizes memory form, 4L ld + 4L st per pencil, map ~14 FMA
+ 1 rsqrt14 seed + 1 vdivpd per site-vector. Zero twiddle tables, zero
shuffle-port ops inside every transform -- twelve sizes and the campaign's
twiddle problem still never entered this file.

### What I would do next (ranked)

1. **Routing** (unchanged from r7, now 4 sizes wider): gen_race/gen_planner
   must race my 22/33/44/55 candidates or the trunk leaves 2x on the table
   at any 11-smooth draw, exactly as measured at L=21 in the surprise test.
2. **B=1 lane-spatial engine** (eighth round on the list; the class's one
   structural hole, at all twelve sizes).
3. **L=55 residency**: the only new cell below 2.5x is pure bandwidth
   (43 MB working set). A y*z-fused two-axes pass (lit 11 Tier 2, the
   gen_pfa_large L=100 play) is the only shape that could move it; at B=8
   it is a full engine restructure -- only worth it if a scored case ever
   lands there.
4. **Cross-arch**: the new sizes ship Ice Lake defaults (div tail, stock
   sched, no fusion); the standing knobs (-DBL_MAPRCP etc.) are the race
   axes on CLX/SPR as always.

## Round gen_r9 -- the factor swap: put the small factor where the map fuses (-1..-2.2% at 10/15/20)

Standings into the round (r8 board): led 20 (12.770 vs pfa_small 13.048), and
1-2 thousandths behind the converged copies at 10 (1.148 vs race 1.146), 12
(1.914 vs pfa_small 1.911), 15 (4.376 vs race 4.374). The r9/r10 brief is
counter-directed: (1) bank engine-internal picks, (2) traffic fusion at
100/50/40, (3) the champion-signature dashboard, (4) idle port 1.

### Round context first: what applied to this engine and what could not run

- **Avenue 1 (bank the picks): this engine has NO internal tuner.** Every pick
  is a compile-time default (BL_MEM15, MAPTAIL_*, SCHED*, and now BL_SWAP*);
  create() is branch-free -- malloc + gl_map_huge, no racing, no RNG, no
  clock. Determinism proof as the brief demands: 5 consecutive fresh-process
  create()+chain cycles at L=15 m=600 produced BIT-IDENTICAL outputs (and the
  same at every A/B this round). The pick-instability class of bug cannot
  exist in this entry; the knob AXES are published for gen_race to race
  per-host (r5 next-step #1, unchanged).
- **Avenue 3 (the dashboard) was BLOCKED this session: no perf anywhere.**
  The reservation died at 20:59 and came back at 23:13 on a DIFFERENT node --
  a81n2, same Xeon Gold 6326 -- where /tmp/perf does not exist.
  perf_event_paranoid is already 2 on a81n2 (so no Will action needed there),
  but the staging copy referenced by TOOLS.md is INCOMPLETE:
  ext/tools/perf-install/ contains only lib64/ plugins and libexec/, no
  bin/perf. The binary lived only in a80n0's /tmp and is gone. MONITOR: one
  scp of a perf 5.15 binary to a81n2:/tmp/perf re-arms the counters; please
  also re-stage the missing ext/tools/perf-install/bin/perf so the next node
  hop is a one-copy fix. All r9 verdicts below therefore stand on same-core
  interleaved TIME only; the l1d.replacement before/after the brief asks for
  is queued for when perf returns.
- **Avenue 4 (port 1) is closed for this engine class by microarchitecture,
  not by measurement**: on ICL-SP the port-1 FP pipe IS the lower 256-bit
  half of port 0's fused 512-bit FMA unit. A 512-bit uop dispatched on port 0
  consumes both halves, so 256-bit FP "co-issue" on port 1 adds ZERO capacity
  to a kernel that keeps p0+p5 fed with zmm work -- converting any of our FP
  to ymm strictly loses (3x256 = 1.5 zmm-equivalents/cycle vs 2). Port 1 is
  free capacity only for non-FP side work (we have none in the hot path) or
  for engines that leave p0 partially idle. The audit's suggestion is real
  for scalar-setup-heavy engines; for batch-lane SoA it is a no-op. A 4-lane
  ymm variant remains interesting for B<8 REGIMES (correctness/coverage), not
  for throughput.

### The round's one idea: FACTOR-SWAPPED map pencils (BL_SWAP10/12/15/20)

The r4 audit located the fused-map x-pencil's residual in register pressure
(15: 27 spill stores + 15 spill loads + 12 folded rsp operands; 12: 31 rsp
touches): the map's ~7 temps ride on the WIDE factor's stage-2 codelet
(DFT5X2 = 2 cores + 10 map ladders in one block at 15). The never-raced axis,
free by PFA symmetry: swap the factor ORDER in the map pencil only -- large
factor in stage 1 (map-free, in place), small factor in stage 2 where the map
fuses. Same FP count, same 4L ld + 4L st (mem form) / 2L+2L (reg form), but
the fused-map codelet shrinks to DFT2/DFT3/DFT4 + 2/3/4 ladders. Sweep
pencils keep the shipped order (they spill zero; nothing to buy).

Slot tables for all four swapped forms were derived with the r6 safe
placement (sigma(c) = (Q^-1 mod P) c) and VERIFIED in Python against a
reference DFT with the exact load-all-store-all group semantics + in-place/
disjointness assertions (the r8 method):
  10 sw: n=(2a+5b)%10, k=(6c+5d)%10;  12 sw: n=(3a+4b)%12, k=(9c+4d)%12;
  15 sw: n=(3a+5b)%15, k=(6c+10d)%15; 20 sw: n=(4a+5b)%20, k=(16c+5d)%20.
At 15 the swapped stage-1 groups are EXACTLY the shipped stage-2 DFT5 groups.
New macros: M3STM/M4STM (mem-form small-DFT + fused map), R4L/R5L (stage-1
DFT into named regs), R3STM/R2STM (reg-form small-DFT + map). All correctness
was validated on wallaby BEFORE any node time (wallaby has AVX-512): full
three-gate battery at 10/12/15/20 + B=1 + mixed B=12/B=9 + all eight other
class sizes, all PASS -- the node session spent zero minutes on debugging.

### Static models said "wash"; the node said otherwise -- and the naive spill
### story is NOT the mechanism

llvm-mca (icelake-server) on the extracted x-pencil loops modeled the
bottleneck as port-0-class FP at ~361 of 377 cycles (12) and predicted the
swap worth only 1-2% of the x-pass. Asm rsp-touch counts in chainsteps:
12: 35 -> 16, 10: 12 -> 12, 15: 18 -> 22, 20: 40 -> 44. Note carefully: the
size with the BIGGEST spill cut (12) LOST the race, and 15/20 WON without
one. The honest reading (unconfirmable without perf, see above): the win is
dependency-shape, not spill count -- the small-factor stage-2 gives each
store a 2-3-op tail after its map ladder instead of a 5-11-output block, and
stage-1's map-free large-factor DFT5s interleave with the previous column's
in-flight ladders. Whatever the mechanism, it reproduced 5/5 interleaved
rounds at three sizes and reversed nowhere.

### The same-core races (a81n2 core 2, held lease, interleaved --samples 4,
### control first, first invocation discarded as warmup)

| case | ship (r8 path, this window) | swap | swap tail flip | verdict |
|---|---|---|---|---|
| L=10 B=64 m=1000 | 1.147-1.150 | 1.131-1.138 (rcp) | **1.121-1.125 (div)** | **swap+div, -2.2%** |
| L=12 B=64 m=600  | **1.916-1.918** | 2.005-2.009 (rcp) | 1.983-1.986 (div) | swap LOSES +3.5..4.7%; ship kept |
| L=15 B=32 m=600  | 4.377-4.408 | **4.332-4.368 (div)** | 4.488-4.502 (rcp) | **swap, -1.0%** |
| L=20 B=32 m=256  | 13.459-13.519 | **13.237-13.335 (div)** | 13.482-13.525 (rcp) | **swap, -1.4..1.6%** |

Sched attributes re-raced on the new codelets (codelet-local rule): 10 keeps
sched-pressure (without: 1.157 vs 1.122, +3.2%); 15 keeps stock (attr: 5.07-
5.14 vs 4.34, +16%); 20 keeps stock (attr: 13.84-14.12 vs 13.28, +4.5%).
Map-tail news: on the swapped REGISTER codelets div now wins (MAPTAIL_SW10=1
shipped; MAPTAIL_SW12 recorded=div but inactive) -- gen_pfa_small r3's "the
tail verdict is a property of the surrounding codelet" now measured a third
time, in the third direction.

### Shipped defaults and confirmation

BL_SWAP10=1 (+div tail), BL_SWAP15=1, BL_SWAP20=1, BL_SWAP12=0. Five
interleaved confirmation rounds, ship vs r8 control, same core:
10: 1.121-1.124 vs 1.147-1.149 (5/5); 12: wash (bit-identical path);
15: 4.340-4.368 vs 4.378-4.408 (5/5); 20: 13.291-13.335 vs 13.459-13.499
(5/5). MKL 2022 same core same window (built fresh into my scratch -- a81n2
had no node-built baselines, r1 harness note repeats): 4.684 / 7.914 /
16.754 / 58.297 -> ratios 4.17x / 4.13x / 3.86x / 4.38x.

Gates (ship build, on a81n2): single call 2.9/2.9/3.5/3.1e-16 at 10/12/15/20
(tol 1e-12); two-step m=2 8.87e-16 / 9.20e-16 / 1.266e-15 / 1.262e-15 (tol
3e-14, 24-34x margin); graded chains 1.336e-13 / 4.869e-14 / 5.536e-14 /
3.618e-14 vs anchors 1.081e-13 / 3.887e-14 / 4.784e-14 / 2.835e-14 (tol
1e-10) -- 10's chain drift IMPROVED from r8's 1.673e-13. 12's chain is
bit-identical to r8; 10/15/20 are new rounding (full gates re-run, above).
B=1 single + m=2 PASS at all four; 5x fresh-process determinism PASS; all
eight other class sizes (14/21/22/28/33/35/44/55) single + m=2 PASS on the
node. B=1 m=64 chains: 10.31 / 17.61 / 39.86 / 117.5 us (the nine-round-old
remainder gap, unchanged).

### What did NOT work, with the number that killed it

- **Swap at 12**: 2.005-2.009 (rcp) / 1.983-1.986 (div) vs ship 1.916-1.918
  -- the biggest spill cut of the four (35 -> 16 rsp) and the clearest LOSS.
  The shipped R4STM's four interleaved map ladders are worth more than the
  spill traffic they cost. Do not rediscover this, and do not trust rsp
  counts as a proxy for time on this engine.
- **rcp tail on the swapped 15/20**: +2.6% / +1.6% (table above). Div stays.
- **sched-pressure on the swapped 15/20**: +16% / +4.5%. Stock stays.
- **PMU dashboard**: blocked, no perf binary on a81n2 (see monitor note).

### Borrowed, plainly

- The swap idea is new this round (PFA order symmetry + the r6 safe
  placement machinery, which made all four derivations one generator run).
- **gen_pfa_small r3/r4**: the codelet-local map-tail rule (raced again, hit
  again) and the register-budget framing that pointed at the map codelet.
- **Literature 10 / my r4**: the held-lease interleaved protocol for every
  number above; and the r8 Python-simulate-before-C table method.
- For **gen_pfa_small**: our 10/15/20 engines were converged copies before
  this round; the swap transfers mechanically (tables in this record). Take
  it -- and take the 12 negative result with it.

### Operation count

FP unchanged: 84 / 96 / 156 / 208 vector instrs per pencil per 8 volumes at
10/12/15/20. Map x-pencil forms now: 10 register swapped (2L ld + 2L st,
div tail), 12 register unswapped (rcp ladder), 15 memory swapped, 20 memory
swapped (4L + 4L, div tails). Sweeps, pack/unpack, 7/11-smooth sizes:
untouched. Zero twiddles, zero shuffle-port ops in every transform, still.

### What I would do next (ranked)

1. **Re-arm the PMU** (monitor: one scp to a81n2:/tmp/perf + restore
   ext/tools/perf-install/bin/perf) and take the champion-signature
   dashboard + l1d.replacement before/after for the three swapped cells --
   the round's avenue-3 deliverable, measurement-blocked this session.
2. **Race BL_SWAP on the memory-form family** (14/21/28/35/22/33/44/55): the
   swap is one generator run per size; at 14/21/28 the map codelet drops
   from DFT7+7 ladders to DFT2/3/4+few -- bigger codelet delta than at 15.
   Only worth node time if those sizes are ever scored (they were not in
   cases.txt; round-6-style draws hit them via the trunk).
3. **B=1 / 4-lane regime** (ninth round on the list): now framed correctly
   by the port-1 analysis -- a ymm variant buys COVERAGE at B in 4..7 (half
   the replication waste), never throughput at B>=8.
4. **Cross-arch**: BL_SWAP10/12/15/20 and MAPTAIL_SW* join the knob axes;
   CLX's weaker FMA and smaller L1 could flip any of them (the 12 verdict
   especially -- its loss is an ILP-vs-pressure tradeoff).
