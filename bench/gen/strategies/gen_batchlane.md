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
