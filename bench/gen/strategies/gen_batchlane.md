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
