# gen_rader — strategy record (GENERALIZE campaign)

Class: Rader for primes.  Owns L=31 (cases.txt 31:16:140) — the dense-vs-Rader
crossover fight against gen_dense_prime.

## Round gen_r1

### Where this round started

The round-0 stub (unfolded dense L x L matrix, scalar, O(L^4)) — the same code
gen_dense_prime's record clocks at **427.9 us/step** through the driver fallback.
No prior record existed.  Read first: gen_dense_prime's gen_r1 record and source
(my direct rival at 31, already rebuilt this round to **177.9 us/step**), the ice
L13/17/23_rader and L17/L23_matrixsimd records, and the ice-r8 chain lessons.

### The arithmetic: why Rader-31 is NOT dense, and what form wins on FMA hardware

The conjugate fold (s_j = x_j + x_{31-j}, d_j = x_j - x_{31-j}) reduces the
31-point DFT to X_k = x0 + E_k -+ iO_k with E = cos-matrix * s, O = sin-matrix * d
(15x15 real matrices; that far it is exactly gen_dense_prime, 4h^2 = 900 FMAs).
Rader's multiplicative reindexing applied to the FOLDED system (quotient group
Z31*/{+-1}, cyclic of order 15, generator 3) turns E into a CYCLIC-15 correlation
and O into a NEGACYCLIC-15 correlation.  Because 15 is ODD, negacyclic converts to
cyclic by diagonal (-1)^q / (-1)^t twists — and every permutation, reversal and
sign lands in compile-time load/store index tables (verified in a Python prototype
against a reference DFT before any C was written; worst rel L2 7.4e-15 over 20
trials, reference-limited).

The convolution algorithm was CHOSEN BY FMA ARITHMETIC, not by multiplication
count:
- full Winograd C15 (C3xC5, 40 mul + 179 add = 219 slots) LOSES to
- **Winograd-C3 nested over DENSE cyclic-5 blocks**: 4 block products of 25 FMA
  each + 11 block adds = 100 FMA + 65 add = **165 instruction slots** per
  convolution, because a dense C5 block is 25 FMA while a Winograd C5 block is
  10 mul + 31 add = 41 slots.  This is the ice L23_rader lesson ("121 fused FMAs
  beat any sub-quadratic length-11 convolution") applied one level up: use
  Winograd ONLY where the multiplications it removes are block products, keep
  the blocks dense.
- All constants are REAL, so on interleaved complex data every op is one zmm
  add/FMA — no complex-multiply shuffles, same trick as the dense entry.
Per 4 pencils per axis: fold 30 + 2x165 conv + X0 5 (summed from the E block,
not the 15 s values) + T 15 + combine 45 = **~430 zmm ops vs dense's ~555**, and
~40 broadcast constants (4 blocks x 5 per kernel) vs their ~465 table loads.
Kernel transforms ((H0+H1+H2)/3 etc.) computed at plan time in long double.

### What was built (complete rewrite of impl/gen_rader.c)

1. **r31_chunk**: one always_inline kernel = E sweep (load 30 rows via JS table,
   fold s, conv15 with cos kernel, store X0, stash T_t = x0+E_t on stack) + O
   sweep (reload rows via JDP/JDM tables — sign of the fold IS the table order,
   conv15 with twisted sin kernel, combine X = T -+ iO via one vpermilpd + two
   sign-vector FMAs, store rows from KP/KM tables).  In-place safe by
   construction: within a chunk every load of a row precedes every store to it.
   Instantiated with compile-time strides for inner=961 (x pass) and inner=31
   (y pass); masked-tail chunk instantiation for the ragged columns.
2. **z pass, Rader form (r31_zquad)**: 4 contiguous rows are transposed through
   8 vshuff64x2 per 4x4-complex tile into a stack array where element j is one
   zmm at stride 8, the SAME r31_chunk runs on it (all offsets compile-time),
   transpose back.  961st row via the borrowed dense single-row z kernel.
   Beat the borrowed dense row-GEMM z pass 3/3 same-core alternating:
   **110.2-112.8 vs 118.1-124.7 us/step** (-8 us).  -DR31_ZDENSE restores it.
3. **Fully IN-PLACE chain step** — the round's biggest single win: because
   r31_chunk is in-place safe, z, x AND y all run in place on the state volume,
   then the map in place (state + c -> state).  t1 exists only for execute()'s
   out-of-place contract.  Chain working set fell from state+t1+c = 1.43 MB
   (over the 1.25 MB L2 — the exact boundary gen_dense_prime's record flags)
   to **state + c = 953 KB, L2-resident: 110.4 -> 94.3 us/step (-16 us)**.
4. **Volume-resident fused chain + s6 map** adopted from gen_dense_prime
   (their map_volume verbatim: pair-compressed |w|^2, 1e-300 clamp, rsqrt14 +
   2 Newton, ONE vdivpd per 8 points; their volume-major chain scheme).
5. **create()-time self-check**: full random volume through the fast engine vs
   the dense-matrix reference; rel L2 must be < 1e-13 or the plan falls back to
   the (slow, correct) stub path.  A transcription bug in the Rader tables would
   read ~1e0; the check makes a fast wrong answer structurally impossible to
   ship (the ice L17_rader r5 gate pattern).

### Measured on the node (a80n0 leased cores via tryout.sh, graded cell L=31 B=16 m=140)

| configuration | us/step |
|---|---|
| round-0 stub (dense_prime's measurement of the same code) | 427.9 |
| Rader x/y passes + borrowed dense z, separate map | 119.0 |
| + Rader-quad z pass | 110.4 |
| + everything in place, L2-resident chain | **94.0 / 94.3 / 94.7 / 95.0 / 96.1 across 5 windows, sd <= 0.13%** |
| gen_dense_prime (their record, same cell) | 177.9 |
| MKL (their record, same cell; no MKL binary in gen build dir to rerun) | 859.5 |
| **B=1** | **93.1-93.6** (chain is volume-major, batch-invariant) |

Correctness (all on the shipped binary): single rel_l2 **4.059e-16** (B=16) /
4.073e-16 (B=1); map-chain m=140 rel_l2 **2.545e-14** (B=16, anchor 2.312e-14,
tol 1.0e-10) / 2.059e-14 (B=1); **two-step gate 1.697e-15** (tol 3.0e-14, the
1.5e-14/step contract met 17x over); chain outputs bit-identical across
processes (manual cmp — tryout's own cmp still dies with its check.py c.bin
quoting bug).

### What did NOT work, with the number that killed it

- **Map fused into the y-pass stores (eager Z-store fusion, ice recipe):
  121.9 us/step with the hw divide, 127.3 with the rcp14+2NR ladder, vs 111.3
  separate.**  Both reciprocal styles lose, so the mechanism is NOT the divider
  (ice L23 r4's div-at-end-of-chain story): the kernel already runs ~30 live
  zmm, and the ladder's temps + constants push it over — same mechanism as
  L17_rader r4's fd negative ("the ladder burst displaces slots; the deint is
  not where the slack is").  Code kept behind -DR31_FUSEMAP; do not retry
  without first freeing registers in the combine loop.
- **Single always-masked kernel body (-DR31_ONEBODY), to halve the ~30 KB hot
  code footprint: 99.6/100.9 vs 94.7/95.0 same-core alternating (2/2).**
  Masked loads/stores + the in-loop mask computation cost more than the L1I
  pressure they save.
- **Map per plane right after each y-plane (L1-hot idea): 111.5 vs 110.4** in
  adjacent windows — a wash (measured before the in-place rewrite; the volume
  sweep was already L2-resident).  Not retried after; the in-place rewrite
  changes the calculus, worth one re-A/B next round.
- An overlapped (recompute-3-columns) tail chunk to delete the masked
  instantiation is IMPOSSIBLE in this engine: the passes run in place, so the
  overlap columns are already transformed when the tail would re-read them.
  Written down so nobody re-derives it.

### Borrowed this round, named

- **gen_dense_prime gen_r1**: the whole chassis — transpose-free pass order,
  z-row dense GEMM kernel (now only the 1-row tail + -DR31_ZDENSE arm), s6
  map_volume verbatim, volume-resident 3-buffer chain scheme (now collapsed to
  2 buffers), the masked-tail column blocking style, and their negative on
  axpy-form contractions (I went straight to register-resident forms).
- **ice L23_rader** (panel_r1-r2): the settled arithmetic that folded-prime
  kernels are FMA-bound, which drove the C3-Winograd/dense-C5 hybrid choice;
  (r4): the div-at-end-of-chain warning, raced here in both styles.
- **ice L17_rader r5**: the create()-time numerical self-check gate before an
  engine may ship.
- **ice L17_matrixsimd r4 / L23_matrixsimd r7**: the objdump-histogram
  discipline (used to confirm the kernel compiled to design: 81 broadcasts,
  ~440 FMA-class, ~29 spill moves per chunk — no pathology).

### Next round

1. **The remaining gap**: ~94 us vs a ~65-70 us port-floor estimate (FMA-class
   ~430/4-pencils x 3 passes + map).  The x-pass streams 31 rows at 15 KB
   stride; latency/AGU behavior there is the most likely residue.  Candidates:
   software-pipeline two column chunks (independent chains), or spill-count
   reduction in wino15 (write Y over S; ~29 spill moves/chunk today).
2. **Fused map, retried properly**: free registers first (e.g. run the combine
   loop in two half-passes of 8 pairs), then re-race -DR31_FUSEMAP.  The
   in-place rewrite means the map sweep now costs a full extra state+c read +
   state write per step — the prize got bigger (~8-10 us).
3. **The crossover verdict so far: Rader 94.0 vs dense 177.9 at L=31** — but
   ~25-30 us of that gap is engine engineering (in-place L2-resident chain),
   not arithmetic.  If gen_dense_prime adopts the in-place chain (they should
   — their z and y passes are already in-place safe; their x-pass GEMM is not,
   it would need my chunk-local structure), expect them near ~130-140; the
   arithmetic edge (430 vs 555 zmm per 4 pencils) is the durable ~20%.
4. **From round 3 the class must take ANY prime**: the machinery generalizes
   (p-1 = 2h, h odd -> cyclic-h + twist; h composite -> nested Winograd with
   dense blocks; h prime -> dense half-system IS the right Rader form, per the
   ice L23 lesson).  Needs a small generator for the index/sign tables and a
   per-h conv plan; the create()-time self-check already guards any p.
5. Harness notes: tryout.sh's remote check.py still receives literal '$W/c.bin'
   (run the map-check manually on the node); reserve.sh needs
   /opt/software/slurm-19.05.8.1/bin on PATH from this dev host; tryout
   regenerates in.bin/c.bin at the batch you pass (keep per-batch copies for
   manual A/Bs — in16/c16, in1/c1 under build/tryout/gen_rader/).

## Round gen_r2

### Where this round started

r1 leaderboard: **94.17 us/step** at the graded cell (L=31 B=16 m=140), leading
the crossover fight (gen_dense_prime 175.6, MKL 848-860).  This window's
control re-read of the r1 binary: 95.95 / 96.44 us (slightly warm window; all
deltas below are same-window A/Bs).  My own r1 next-steps list said the x-pass
"latency/AGU behavior" was the most likely residue; the cumulative context
supplied the missing mechanism.

### The diagnosis: the natural 31^3 layout is a 4K-aliasing and line-split trap

Two facts about the in-place chain nobody had put together in r1, both pure
address arithmetic:

1. **Exact 4K store->load aliasing in the x-pass.**  Row stride = plane =
   961 complex = 15376 B == 3088 mod 4096, and 4 x 3088 == 64 mod 4096.  So the
   chunk at column d stores row j+4 at EXACTLY the low-12 address bits of the
   next chunk's load of row j (at column d+64B).  With ~31 stores in flight at
   every chunk boundary and ~27 of the next chunk's 61 loads landing on
   stored-row+4 addresses, that is thousands of memory-disambiguation replays
   per pass, every step.  This is the L23_rader ice "+27% alias hole" mechanism
   (gen_pfa_large used the same rule to move L=100 13.4 -> 12.0 ms in r1).
2. **Line splits everywhere.**  15376 mod 64 = 16 and 496 mod 64 = 48: every
   x-pass access and 3 of 4 y/z-pass row accesses straddled a cache line.

### What was built: the chain state moved to a fully padded, fully aligned private arena

- z-rows padded 31 -> 32 complex (R31_ZP, 512 B: all y/z row bases 64B-aligned),
  planes padded to 1148 complex (R31_PP; == 124 mod 256, chosen so the x-pass
  row stride is 31*64 B mod 4096: with 31 = 31^-1 mod 64, a store and a later
  load first share low-12 bits at row distance 33 / 2-chunks-retired -- outside
  the 31-row system).  State + padded c mirror = 2 x 570 KB = 1.14 MB, still
  L2-resident (the r1 in-place-chain win is kept).
- Pad slots zeroed once at create() and PROVABLY stay zero (no pass mixes
  columns; DFT(0)=0; map(0+0)=0) -- no NaN/denormal assists, and the passes
  become tail-free: z = 8 uniform quads per plane (the 32nd "row" is the zero
  pad row), y = 8 full chunks (32 cols), x = 248 full chunks (992 cols, no
  masked tail instantiation in the hot path at all), map = 124 exact vector
  groups (992 = 124 x 8; the per-plane scalar tail point is gone too).
- Step 0's z-pass reads the flat x0 directly into the arena (7 flat->padded
  quads + a strided 3-row tail; the old full-volume memcpy is deleted); the
  last step's map stays in the arena and 961 rows are memcpy'd out flat
  (1 of 140 steps, negligible).  c is copied into a padded mirror once per
  volume (466 KB per 140 steps, negligible).
- create() self-check EXTENDED: it now also runs one padded chain step against
  dense-reference + exact scalar map at 1e-13, so a pad/stride bug falls back
  to the slow correct path instead of shipping a fast wrong answer.
- Cost of the padding: 31 zero pad columns ride through the x-pass and map
  (+3.2% lanes), bought back several times over by alignment + no tails.

### Measured on the node (a80n0 leased cores via tryout.sh, graded cell, same-window A/Bs)

| configuration | us/step |
|---|---|
| r1 binary, this round's windows (control) | 95.95 / 96.44 |
| + plane pad only (planes -> 1148, rows still 31): x-pass aliases+splits gone | 91.30-91.64 (**-5.0%**) |
| + z-row pad -> 32 (full alignment, tail-free passes, shipped) | **86.61 / 87.06 / 87.30, sd 0.14%** |
| gen_dense_prime (r1 leaderboard, same cell) | 175.6 |
| MKL 2022 (same window, same core) | 848.6 |
| **B=1** | **99.7** (sd 0.21%) |

B=1 note: the identical batch-invariant code path read 93.1-93.6 in r1's
windows; 99.7 here is the documented B=1 short-unit core-ramp signature
(~12 ms units under schedutil -- gen_dense_prime ice_r8 diagnosis, also seen
by gen_layout r2 at +20%).  Expect the quiet scoring window to read lower.

Correctness (shipped binary, all checked on the node by hand -- tryout's
check.py chain leg still dies on the unexpanded `$W`): single rel_l2
**4.059e-16** (B=16) / 4.073e-16 (B=1); map-chain m=140 **2.559e-14** (B=16,
anchor 2.312e-14) / 1.923e-14 (B=1, anchor 1.178e-14), tol 1e-10; **two-step
gate 1.633e-15** (tol 3e-14, 18x margin); chain outputs bit-identical across
independent node processes.  objdump audit of the shipped x-pass chunk: 410
arith (220 FMA-class), 41 broadcasts, 15 vpermil, 17 spill stores -- compiled
to design, no pathology.

### What did NOT work, with the number that killed it

- **sched-pressure as a function attribute on the pass instantiators
  (-DR31_SCHEDP): 131.6 vs 91.3 us/step -- +44%, the round's biggest loss.**
  gen_batchlane r2 / gen_powp r1 measured it paying on THEIR spill-bound
  codelets; on this monolithic ~500-op chunk the pressure scheduler serializes
  the four independent C5 block products.  gen_pfa_small r2 saw the same
  non-transfer on their new engine.  Lesson sharpened: the attribute pays on
  SMALL pressure-bound codelets, not on large software-pipelined bodies with
  deliberate parallel accumulator structure.  Knob left in for cross-arch.
- **Fused map (R31_FUSEMAP) not retried**, on my own r1 numbers (121.9/127.3
  vs 111.3): padding does not free the registers that killed it, and the
  per-plane map is now L1-hot so the prize shrank.  The flat-chain arm
  (-DR31_FLATCHAIN) keeps both old paths raceable.
- The old in-place map_volume(state, c, state) call was a latent `restrict`
  violation (UB that happened to compile right in r1); the r2 map drops
  restrict on the aliasing pointers.  Recorded so nobody reintroduces it.

### Borrowed this round, named

- **gen_pfa_large gen_r1** (transitively ice L23_rader): the odd-pitch /
  mod-4096 anti-aliasing rule and their 13.4 -> 12.0 ms confirmation that it
  is worth real time -- this round's central lever.  The exact-pitch analysis
  (124 mod 256 so 31^-1 mod 64 pushes the first collision to row 33) is mine;
  gen_layout r2's gl_pick_pitch4k would have found a pitch by measurement --
  I solved this one closed-form but their audit API is the right tool the
  moment the class goes any-prime (r3).
- **gen_batchlane r2 / gen_powp r1**: the sched-pressure-as-attribute recipe
  (measured; lost here -- negative result contributed back above).
- **gen_dense_prime r1**: the "za-style padding 31 -> 32" next-step idea from
  their record (their item 2), realized here on the Rader engine first.
- **gen_layout r2 / gen_pfa_small r2**: harness notes (tryout's $W bug now
  fixed; the remote check.py chain leg still broken -- run map-check by hand).

### Operation count (shipped, per chain step per volume)

z: 248 quads (8/plane) x ~500 zmm ops; x: 248 full chunks x ~475; y: 248 full
chunks x ~475; map: 3844 groups x (~21 ops + 1 vdivpd).  All accesses
64B-aligned, zero masked-tail work in steps >= 1, zero scalar map points.
~240k zmm FMA-class + ~90k adds per volume vs r1's same arithmetic on worse
addresses.  Port floor still ~65-70 us; shipped 86.6 => ~1.27x above floor
(was 1.4x).

### What I would do next

1. **The remaining ~17-20 us over floor**: with addresses clean the residue is
   issue/feed shape inside the chunk -- the two r1 candidates stand: write Y
   over S in wino15 (cuts the 45-live-value plateau), and software-pipelining
   the E-fold of chunk n+1 under the O-combine of chunk n.
2. **Fused map, only after registers are freed** (two half-passes of the
   combine loop); the padded layout makes the c operand aligned now, which
   removes one of the r1 objections.
3. **Any-prime generalization (round 3)**: p-1 = 2h; h odd -> cyclic-h +-
   twist; the pitch rule generalizes (pick plane pitch with (h+1)^-1-style
   closed form or gl_pick_pitch4k / gl_stream_audit4k at plan time -- adopt
   gen_layout there).  The index/sign table generator and per-h conv plan are
   the work items; the extended create() self-check already gates any p.
4. **For gen_dense_prime**: the diagnosis transfers verbatim -- their x/y
   fold_pass GEMM streams the same 15376 B rows in place; their planned
   31->32 row pad should use plane pitch == 124 mod 256 complex (not just
   row-rounding) or the x-pass store->load aliases survive at row distance 4.

## Round gen_r3

### Where this round started

r2 leaderboard: **86.913 us/step** at the graded cell (L=31 B=16 m=140), leading
the crossover (gen_dense_prime 124.3, MKL 849-883).  The round-3 rule activates:
the class must accept ANY size in class — for this entry, any odd prime.  This
round's control re-measure of the r2 binary: 86.61 / 86.84 / 86.93 (interleaved
with the new build; the node is ~5% bimodal this round per gen_dense_prime's r3
protocol — all A/Bs below are 2-3 interleaved process instances per arm).

### What was built

1. **Any-prime class duty: supports() = every odd prime 3 <= p <= 127.**
   Primes != 31 run a NEW generic engine (rp_*): the conjugate-folded half
   system (C_k = x0 + sum_j cos(2pi jk/p) u_j, S_k = sum_j sin v_j, X_k =
   C_k -+ iS_k — folded dense, the settled ~2h^2-FMA count; NOT the Winograd
   nesting, which is 15-specific) in a runtime-(p,h) column-chunk kernel:
   all p rows loaded ONCE into stack u/v arrays (loads-all-then-stores => in-
   place safe for any dst==src), then k in QUADS of 4 (C,S) accumulator pairs
   sharing each u_j/v_j reload — per j per quad, 2 stack loads + 8 broadcast
   constants feeding 8 FMAs, all-real constants on interleaved zmm.  z axis
   through the r31_tp4 4x4-complex transpose quad generalized to runtime p and
   1..4 rows (zero-padded missing rows; garbage lanes only ever cross shuffles
   and die in masked stores).  Chain fully in place on the out volume (my r1
   form), map per plane while the plane is cache-hot.  Tables k-major exact
   long-double sincos.  The existing create() self-check (execute + one chain
   step vs dense reference at 1e-13) gates every p; the dense-matrix fallback
   ships if it fails.
2. **L=31: st + cpad moved into ONE 2MiB huge-page arena, c mirror at page
   phase +2048 B** (gl_map_huge recipe from gen_layout, verbatim; heap
   fallback kept).  gen_dense_prime's r3 found two same-phase ~500 KB
   aligned_allocs make the map's c loads 4K-alias the y-pass state stores —
   my r2 had exactly that layout.  Also puts the 1.14 MB chain working set on
   one TLB entry and makes layout luck deterministic.

### Measured on the node (a80n0 leased cores; graded cell L=31 B=16 m=140)

| configuration | us/step |
|---|---|
| r2 binary, this round (control, interleaved) | 86.61 / 86.84 / 86.93 |
| r3 arena build, same windows | **85.50 / 85.57 / 85.62 (-1.4%, 3/3)** |
| **B=1** | **85.34 (sd 0.09%)** — r2's window read 99.7; this window is quiet, so part of the delta is the documented B=1 ramp signature, but B=1 now reads BELOW B=16 |
| MKL 2022 same case/core | 849.6 |

Correctness at 31 (shipped binary, all by hand on the node — tryout's remote
map-check still dies on its unexpanded '$W/c.bin'): single rel_l2 4.059e-16
(B=16) / 4.073e-16 (B=1); map-chain m=140 2.559e-14 (B=16, anchor 2.312e-14) /
1.923e-14 (B=1); **two-step gate 1.784e-15** (tol 3e-14, 17x margin); outputs
bit-identical across independent runs.

Generic primes (execute() B=2 unless noted; MKL same core where measured):

| p | us/xform | MKL | setup |
|---|---|---|---|
| 3 / 5 / 7 / 11 | 0.19 / 0.72 / 1.6 / 6.6 | — | ~0 |
| 13 (B=4) | 10.96 | 6.02 | ~0 |
| 17 / 23 | 25.5 / 84.3 | — | ~0 |
| 29 (chain m=8) | 177.5/step | — | 0.005 s |
| 37 (chain m=8, B=2) | 418-472/step (window-dependent) | 1599 (execute) => **3.7x** | 0.014 s |
| 43 | 839 | — | 0.03 s |
| 61 | 3067 | 16004 => **5.2x** | 0.11 s |
| 101 | 24158 | — | 1.24 s |
| 127 | 48668 | 49748 => **1.02x** | 2.92 s |

All pass single rel_l2 at 1e-12 (worst 1.04e-15 rel_max at 127); chains at 29
and 37 pass map-chain AND the m=2 two-step gate (1.84e-15 at 29); repeatable.
Every setup is far under the 60 s cold budget.  Primes 37..127 previously had
ONLY Bluestein coverage in the library — that is the r6 insurance this buys.

### What did NOT work, with the number that killed it

- **-DRP_YMAPFUSE (map fused into the generic y-pass stores): LOSES 2/2 at
  both probe sizes — L=37: 480.2/482.0 vs 469.7/471.6 (+2.3%); L=13: 13.8/14.5
  vs 13.1/13.2 (+5-9%).**  Third engine on this panel where eager map fusion
  into a pass loses (my r31 r1 numbers, gen_dense_prime's r2 plane-map, now
  the lean generic kernel too).  gen_pfa_small's r3 lesson confirmed again:
  the div-vs-ladder/fusion choice is a property of the surrounding codelet —
  A/B in place, never adopt on faith.  Knob kept for cross-arch.

### Borrowed this round, named

- **gen_layout r1/r2**: gl_map_huge (2MiB THP arena, align-trim-madvise-
  prefault) adopted verbatim into create(); their phase-stagger doctrine.
- **gen_dense_prime gen_r3**: the same-phase-aligned_alloc 4K-alias diagnosis
  (their cp/sp finding transferred cleanly — my engine's map DOES abut the
  y-pass store drain, which is exactly the geometry their record says the fix
  pays on), and the bimodal-node measurement protocol (interleave 3+ process
  instances per arm, compare min-sets).
- **gen_pfa_small gen_r3**: the shape of the round-3 duty — a generic
  runtime-table engine beside the tuned paths, gated by the same self-check.

### Operation count

31: unchanged from r2 (~240k zmm FMA-class + ~90k adds per volume step; all
accesses 64B-aligned, tail-free).  Generic p: ~2h^2 zmm FMA per 4-complex
chunk per axis (k-quads at 10 loads / 8 FMAs — mildly load-bound), z via
transpose quads (+16 shuffles/4 pencils), chain adds one in-place map sweep
per plane.  Flat layout: line-splits and (at some p) 4K aliases are UNPAID
DEBT at the generic sizes — see next.

### What I would do next

1. **True generalized Rader for 3|h** (p in {7,19,43,67,79,103,127}): the
   Winograd-C3-over-dense-cyclic machinery generalizes (4 block products of
   (h/3)^2 FMA vs h^2 dense — ~45% FMA cut at 127, where the engine is only
   at MKL parity).  Needs a runtime index/sign table generator for the
   quotient-group reindexing; the self-check already gates any p.
2. **Padded arena chain for generic primes** (rows -> mult-of-4 complex,
   anti-alias plane pitch = the r2 closed form generalized, or
   gl_pick_pitch4k at plan time): the r2 win at 31 was 9.5%; the generic
   sizes still run the r1-era flat layout.
3. **Large p (>= 61) chain is DRAM-resident** (127^3 = 32 MB volume): the
   in-place full-volume passes stream it 4x per step.  A plane-blocked step
   order (z+x on a plane band, y+map behind it) could cut that ~2x.
4. 31 residue unchanged: wino15 spill cut and two-chunk software pipelining
   still untried (gcc lotteries; this round's noise budget went to the A/Bs
   above).
5. Harness: tryout's remote map-check still needs the by-hand run; keep
   in16/c16, in1/c1, in37/c37 etc. under build/tryout/gen_rader/ current.
