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

## Round gen_r4

### Where this round started

r3 leaderboard: **85.088 us/step** at the graded cell (L=31 B=16 m=140), leading
the crossover (gen_dense_prime 123.8, gen_bluestein 301.3, MKL/FFTW 849-883).
This round's control read of the r3 binary via tryout: 85.046 — a quiet window,
board-consistent.  My r3 next-list: wino15 spill cut, two-chunk pipelining,
generic-prime arena, large-p custody blocking.

### The round's main negative finding: the engine is AT its issue-port model

Before touching code I costed the shipped step against the port budget
(2 FMA pipes, shuffles on port 5, ICX): x/y chunks ~470 FMA-class / 2 ports
= ~235 cyc per 4 columns; z quads ~(470 arith + 128 shuffles) with port 5 the
binder at ~363 cyc/quad; map 3844 groups x ~21 FMA-port ops + 1 vdivpd
(divider ~31k cyc, overlapped).  Sum ≈ 250k cycles ≈ 86 us at 2.9 GHz —
**the measured 85 IS the model**.  The "65-70 us floor" in my r1/r2 records
undercounted the z-pass's port-5 shuffle bill and the map's FMA bill.  Spills
(~32 stack ops/chunk vs 134 total port-2/3 ops = 67 cyc, under the 235-cyc FMA
plateau) are NOT binding, so the r2 "wino15 spill cut" and "write Y over S"
items are dead on arrival — struck from the list with this arithmetic instead
of a burned window.  Consequences of the model, all consistent with this
round's measurements: only deleted WORK can move this kernel now, and the
work is already the settled minimum (C3xC5 hybrid, ice L23 lesson).

### What shipped: PLANE CUSTODY in both chains (bit-identical, cmp-verified)

Step s+1's z-pass contracts within a plane, so it does not need to wait for
the map sweep of the whole volume: it now runs per plane immediately after
that plane's map, while the plane is L1/L2-hot (z_0 moves to a prologue; the
last step keeps map + copy-out).  One full-state read per step deleted from
the pass sequence.  Arithmetic order within every pass unchanged =>
**chain outputs bit-identical to the r3 binary** (verified by cmp at 31, and
at 23/61 for the generic engine where the per-plane regrouping changes quad
boundaries but pencils are lane-independent).  Same move applied to
rp_chain_volume (rp_zplane: per-plane quads + tail).  -DR31_R3CHAIN /
-DRP_R3CHAIN keep the r3 order raceable.

### Measured on the node (a80n0; SAME-CORE interleaved A/Bs — see protocol note)

L=31 B=16 m=140, min over 6 interleaved rounds of 4 arms, one held lease,
core 4:

| arm | us/step |
|---|---|
| r3 control (impl_3 rebuild) | 85.024 |
| **custody (shipped)** | **84.725** (-0.35%, never worse across rounds) |
| custody + x-pass prefetch (-DR31_PFX=128) | 85.648 (+0.7% — REJECTED) |
| custody + map-in-z-loads (-DR31_ZMAPF) | 88.468 (+4.0% — REJECTED) |
| **B=1 (shipped)** | **85.663** (sd 0.10%) |
| MKL 2022 same window | 849.6 |

Generic primes, custody vs r3 order (interleaved, same core): p=61 m=8 a WASH
(2649 both arms — 3.6 MB volume sits in the 24 MB LLC; the deleted read was
cheap); **p=127 m=4: 47372-47398 vs 47785-47899, -0.9%, 3/3** (32 MB volume,
DRAM-resident — the regime the change is for; round-6 insurance).

Gates (shipped default build, by hand on the node — my wrong check.py
invocation mid-round produced scary FAIL lines; --output wants the BASE file,
it appends .chain itself): single rel_l2 4.059e-16 (B=16) / 4.073e-16 (B=1);
map-chain m=140 2.559e-14 (B=16, anchor 2.312e-14) / 1.923e-14 (B=1);
**two-step gate 1.784e-15** (tol 3e-14); bit-repeatable; p=23/37 chains and
p=13 execute PASS.  All error numbers identical to r3 — as they must be, the
outputs are bit-identical.

### What did NOT work, with the number that killed it

- **x-pass software prefetch (gen_layout r4's recipe): +0.7% (85.65 vs
  85.02).**  Their -2.6% at L=31 was on a DRAM/L3-resident demo engine; my
  state is L2-resident, the loads hit L2 and OoO already hides them, so the
  31 extra port-2/3 uops per chunk (~1.3 us/step) are pure cost.  Boundary
  written down: prefetch pays where the stream MISSES the cache the kernel
  can hide, not where it hits L2.  Knob kept (-DR31_PFX) for CLX/SPR.
- **Map fused into the z-quads' transpose-in loads (-DR31_ZMAPF): +4.0%
  (88.47 vs 84.73).**  The panel's FIFTH map-fusion negative and the first
  LOAD-side one (r31 y-stores r1 +10%, dense_prime r2, rp y r3 +2-9%,
  batchlane epilogue r4 +6-9%, now this).  I built it because the load-side
  geometry dodges the register-pressure mechanism that killed the other four
  (~10 live zmm at the fusion point, not ~30) — and it STILL lost: the map's
  FMA/unpack ops land on the z-quad's already-binding port 5, and the vdivpd
  goes serial on the transpose's critical path.  With the port model above,
  fusion cannot win here: it relocates work onto the binder.  Map placement
  at L=31 is now closed at every granularity.  (Kept the vdivpd variant
  bit-identical to map_volume via the new hwdiv parameter on r31_map_rec, so
  the knob remains a clean cross-arch candidate.)
- **wino15 spill reduction / Y-over-S / two-chunk pipelining: struck without
  a window** — spills are ~67 port-2/3 cycles under a 235-cycle FMA plateau
  (see model).  Do not resurrect these without first showing the model wrong.

### Borrowed this round, named

- **gen_batchlane gen_r4**: the same-core A/B protocol — tryout.sh acquires a
  FRESH lease (fresh core) per invocation, so cross-invocation pairs carry a
  10-25% core-state confound.  Everything above was measured holding ONE
  lease and alternating the same binaries on the same core.  My r1-r3
  verdicts re-examined under this light: the big ones (in-place chain -16us,
  padding -9.5%, quad-z -8us) are far above the confound; no re-race needed.
- **gen_layout gen_r3 / gen_bluestein gen_r4**: the plane-custody idea (their
  window/blocked custody); my contribution is noticing z is plane-local in
  MY pass order so custody costs zero new code structure and stays
  bit-identical.  Adopted their "test at the size regime it targets"
  discipline: the win is at DRAM-resident p, not at the L2-resident 31.
- **gen_layout gen_r4**: the prefetch recipe (measured, lost here — boundary
  contributed back above).

### Operation count

Unchanged from r3 at 31 (~240k zmm FMA-class + ~90k adds per volume step);
the custody change moves no arithmetic, deletes one full-state read per step
from the pass schedule.  Generic p: identical per-pencil ops; per-plane z
grouping adds one short-quad tail per plane at p%4!=0 (measured: a wash at
61 even so).

### What I would do next

1. **The 31 cell is port-model-saturated.**  The only untried lever that
   DELETES port-5 work is replacing some z-quads with dense zrow_pair rows to
   rebalance port 5 vs the load ports (interleave the two forms within a
   plane); expected value small, OoO windows may not span the call boundary.
   Model first, budget one window.
2. **Large-p chain, the real remaining meat (r6 insurance): block the X pass.**
   Custody only deleted z's read; at 127 the x-pass still streams 32 MB from
   DRAM per step.  A z-block custody (x-pass restricted to a block of
   (y,z)-columns across all planes, then y+map on those columns) is the
   gen_bluestein r4 shape; at 31 it is not worth it (L2-resident), at >= 61
   it is the difference between 1 and 2 DRAM sweeps per step.  Pair it with
   gen_layout's NT stores (their -19% at L=100) for the map stores at
   DRAM-resident sizes.
3. **Generic-prime padded arena** (rows -> mult-of-4, anti-alias pitch via
   gl_pick_pitch4k or the r2 closed form) — still unpaid debt from r3, gated
   by gl_alias_drained4k first (gen_layout r4 built that gate from
   gen_dense_prime's negative: my rp chunks run back-to-back, so the alias
   fix should actually pay here, unlike theirs).
4. **True generalized Rader for 3|h primes** (7,19,43,67,79,103,127): ~45%
   FMA cut at 127 where the engine is only at MKL parity — the biggest
   arithmetic lever left anywhere in this entry.
5. Harness: check.py --output takes the BASE filename (it appends .chain);
   in16/c16, in1/c1, in23/37/61/127 + c mirrors kept current under
   build/tryout/gen_rader/.

## Round gen_r5

### Where this round started

r4 leaderboard: **84.603 us/step** at the graded cell (L=31 B=16 m=140), leading
the crossover (gen_dense_prime 120.5, gen_bluestein 299.4, MKL/FFTW 830-883).
My r4 record's port model says the 31 cell is saturated; my r3/r4 next-lists
say the round belongs to the generic primes (r6 insurance): the padded-arena
debt and the generalized Rader arithmetic.  That is what this round did.  All
node numbers below: a80n0, ONE held lease (slot 0, core 2), interleaved arms
(the gen_batchlane r4 protocol).

### What shipped, part 1: OUTER-C3 RADER for h = 3m primes (the round's headline)

The r31 Winograd-C3-over-dense-blocks machinery generalized to runtime tables,
for any prime whose quotient-group order h = (p-1)/2 is odd, divisible by 3,
with gcd(3, h/3) = 1 — **p in {43, 67, 79, 103}** (7 is m=1 = already dense;
19 and 127 have 3 | m and cannot CRT-split; h-even primes need a negacyclic
split, see next-steps).  The construction (rp3_build) is the r31 table recipe
verbatim at runtime — a_r = g^r mod p, slot n = m*i+j <-> CRT index
(e3*i + em*j) mod h, correlation->convolution by data reversal, the odd-h
negacyclic sign twist baked into the jdp/jdm swap, the output (-1)^t twist
baked into the kp/km pair order, kernels Winograd-C3-transformed in long
double and STRETCHED to 2m-1 doubles so the dense-Cm block product indexes
kb[m-1+j-i] mod-free.  I re-derived the recipe from scratch and cross-checked
it against the R31_JS/JDP/JDM/KP/KM constants before writing any C.

The chunk kernel (rp3_chunk_M) is macro-instantiated PER m (7/11/13/17) so
every loop bound is compile-time and the four m^2-FMA block products unroll
register-resident, exactly like R31_C5 — the gen_dense_prime r2/r5 exact-tile
doctrine.  Per 4 columns per system: 4m^2 + ~11m FMA-class vs the dense
engine's 9m^2 (2h^2 = 18m^2 for both systems).  Dispatch is one switch on
plan->m3 per chunk (rp_chunk_any); the dense engine is untouched and remains
the path for all other primes.

**Measured (same-core interleaved, min us/step, chain B/m as shown):**

| p | dense (-DRP_NOC3) | outer-C3 | delta |
|---|---|---|---|
| 43 (B=2 m=8) | 697.6 / 701.2 | **456.7 / 467.5** | **-34.5%** |
| 67 (B=1 m=8) | 3778 / 3788 | **2233 / 2251** | **-40.9%** (3/3) |
| 79 (B=1 m=8) | 7751 / 8138 | **5343 / 5351** | **-31%** (2/2) |
| 103 (B=1 m=4) | 21250 / 21276 | **17829 / 17963** | **-16%** (33 MB volume, DRAM-resident — the FMA cut is partially hidden behind memory; see next-steps) |

Setup: 0.02 s (43) / 0.14 (67) / 0.3 (79) / 1.15 (103) — far under the 60 s
budget.  Gates on the node at 43/67/79/103: single rel_l2 4.1e-16..9.1e-16,
**two-step 2.1e-15 / 2.5e-15 / 2.8e-15 / 5.0e-15** (tol 3e-14).  A table bug
cannot ship: the create() self-check (execute + one chain step vs dense
reference at 1e-13) gates every plan, same as always.

### What shipped, part 2: padded huge-page arena chain for generic primes p <= 61

The r2 lesson at 31 paid at the other primes, with a measured size gate.
Rows p -> mult-of-4 complex (all row bases 64B-aligned, y-pass and x-pass
tail-free), plane pitch by rp_pick_pp: PP/4 == 2 mod 4 makes the +64B x-pass
store->load alias equation d*(PP/4) == 1 mod 64 UNSOLVABLE at every row
distance — stronger than my r2 closed form, which only pushed the first
solution outside the row system.  State + padded c mirror in ONE gl_map_huge
2MiB arena (gen_layout, now adopted as a real GEN_LAYOUT_LIB_ONLY include,
not a copied recipe), c at a +2048B page phase (my r3 trick).  Map runs on a
mult-of-8 window whose pad lanes are provably zero.  Custody pass order kept.
**Outputs are BIT-IDENTICAL to the flat chain** (every op is lanewise;
cmp-verified at 13/37/61) — the whole numerics story, borrowed from
gen_dense_prime r5.

Measured: **37: 423.9/424.3 flat vs 418.6/417.4 padded (-1.4%, 2/2** after a
warmup round); 13/43/53/61: wash (43/53 in a noisy sd 2-7% window, min-of-mins
overlap); **127: padded LOSES -4% (48.0k flat vs 49.9-50.3k, 2/2 clean)**.
Diagnosis of the 127 loss: at DRAM-resident sizes the arena's per-VOLUME
overheads (c-mirror fill + final copy-out ~ 2 volume sweeps = ~32 MB/step at
m=4) are exactly the gap, and large L implies small m by the suite's
construction (~0.4 s of MKL per case), so they never amortize.  Hence
RP_PAD_MAX = 61: padded where it wins or washes, flat where it loses.
create() falls back to the flat chain if the mapping fails; never fails.

### What did NOT work, with the number that killed it

- **zp phase-spreading bump** (skip row strides == 0 mod 4 in 64B units, to
  fix the "zp=128 at p=127 is a 2048B stride = two 4K phases for 127 rows"
  theory): **127: zp 132 vs 128 = +1.5% (3/3); 61: zp 68 vs 64 = +4% (3/3).**
  The extra pad lanes cost more than 4K-phase spread buys.  The set-conflict
  theory is DEAD at these kernels — do not resurrect it without new evidence.
- **-DR31_ZMIX (my r4 next-step 1: replace trailing z-quads with dense
  zrow_pair rows to move z work off the quad's binding port 5): ZMIX=6 =
  88.0-89.0 vs 85.7-86.4 (+2.5%, 3/3); ZMIX=4 = 91.7-92.3 (+7%).**  The dense
  rows' table loads + fold traffic cost more than the port-5 relief; the
  all-quad r4 form is confirmed optimal and the r4 port model's "only deleted
  work moves this kernel" holds.  31 is closed.  (Dropping `restrict` from
  r31_zpass/zrow_pair src/dst was required for the in-place ZMIX arm — kept,
  since the r1 flat arm ran the same latent violation; no timing effect.)
- Bookkeeping: the padded chain at 43 combines with outer-C3 (43 <= 61), so
  the 43 numbers above are arena+C3 vs arena+dense — the C3 delta is clean.

### Borrowed this round, named

- **gen_layout** (adoption, not transcription): gl_map_huge/gl_unmap via
  GEN_LAYOUT_LIB_ONLY include back the generic arena; their r5 "one code path
  unless a scored regime measurably loses" doctrine shaped the RP_PAD_MAX
  gate; their r5 result that the fusion/locality money lives in the L2/L3
  MIDDLE (not at DRAM sizes) predicted my 127 finding before I measured it.
- **gen_dense_prime gen_r5**: the exact-tile/tail-free discipline (their
  -17% at 17) is the shape of the padded layout; the bit-identity-by-lanewise
  argument + cmp-at-every-size verification protocol, used verbatim.
- **gen_batchlane gen_r4**: the one-lease same-core interleave protocol, again.
- **ice L23_rader / my r1**: the "keep the blocks dense, Winograd only at the
  outer level" arithmetic that makes outer-C3 the right generalization.
- **docs/literature 11 (rounds 5-6 brief)**: "GT/Rader-as-vectorization-first"
  (Tier 1) — this round is the first performant realization of the
  generalized-quotient-group Rader in this campaign; cited per the brief.

### Operation count (shipped)

31: unchanged (~240k zmm FMA-class + ~90k adds per volume step); chain output
bit-identical to the r4 binary (cmp on the node).  Outer-C3 primes: per 4
columns per axis, 2 x (4m^2 + ~11m) FMA + fold 2h + combine ~4h, vs dense
2h^2 = 18m^2 — a 2.1-2.25x conv-FMA cut; z via the same transpose quads.
Padded chain (p <= 61): identical arithmetic to flat on padded addresses,
plus <= 8% pad lanes, minus all masked tails and line splits.

### Measured summary (the reply line)

L=31 B=16 m=140: **85.27 / 85.57 / 86.05** (this window; r4 board 84.603 —
same binary path, bit-identical output); B=1: **85.54**.  Single rel_l2
4.059e-16, map-chain m=140 2.559e-14 (anchor 2.312e-14), two-step 1.784e-15,
bit-repeatable.  Class duty: 43 **457**, 67 **2233**, 79 **5343**, 103
**17829** us/step — -31..-41% vs r4's engine (-16% at 103).

### What I would do next

1. **h-even primes (37, 41, 53, 61, 73, 89, 97, 101, 109, 113)**: the E
   (cos) system is still a cyclic-h correlation for ANY h — only the O (sin)
   system goes negacyclic when h is even, and a negacyclic C2 split is a
   3-block-product Karatsuba (not 2).  E-side C2 split alone is a ~25% FMA
   cut at h == 2 mod 4; worth one round.  127 (h = 63 = 7x9) needs an
   Agarwal-Cooley C7xC9 or a two-level plan — the last big-prime hole.
2. **103/127 are now (more) memory-bound**: the x-band custody blocking
   (gen_bluestein r4 shape, my r4 next-step 2) is the lever the C3 cut just
   sharpened — at 103 the dense engine hid the memory time, the C3 engine
   exposes it (16% vs 35-40% at cache-resident sizes).
3. The rp3 chunk's kp/km stores are index-scattered; a store-order pass
   (contiguous row order, indices on the LOAD side only) is a cheap race.
4. Two-level outer Winograd (C3 x C3 does not CRT-split, but C2 x C3 = C6
   does for h == 6 mod 12, e.g. 61's h = 30 -> m = 5 dense): after item 1.
5. Harness: check.py --output takes the BASE filename; r5_ab.sh under
   build/tryout/gen_rader/ holds the whole session protocol (build / generic /
   zpfix / mid / l31 / c3 / final phases); in/c pairs for 13,31(B1,B16),37,43,
   53,61,67,79,103,127 kept current there.

## Round gen_r6

### Where this round started

r5 leaderboard: **84.668 us/step** at the graded cell (L=31 B=16 m=140), leading
the crossover (gen_dense_prime 120.3, gen_bluestein 289.3, MKL/FFTW 833-883).
Round 6 is the surprise round: three never-announced sizes in 14..127 score the
ASSEMBLED library, so this round belongs entirely to the class duty -- make
arbitrary primes fast.  My r5 next-list item 1 (h-even primes) is the round's
headline; the codegen audit it forced turned out to matter as much as the
arithmetic.  All node numbers: a80n0, ONE held lease (slot 2, core 4),
interleaved arms (the gen_batchlane r4 protocol, as always).

### What shipped, part 1: EVEN-h SPLIT RADER (rp2_*) for all p == 1 mod 4

Covers 13 17 29 37 41 53 61 73 89 97 101 109 113 -- every prime in class whose
quotient-group order h = (p-1)/2 = 2m is even.  These all ran the dense 2h^2
engine before this round.  The arithmetic (validated in numpy FIRST --
build/tryout/gen_rader/r6_proto.py, worst rel L2 6.3e-16 over all 13 primes --
then gated by the create() self-check as always):

- The E (cos) system is a cyclic-h correlation for ANY h (cos is even, so the
  quotient-group reindexing never produces signs).  For even h there is NO
  odd-h negacyclic->cyclic twist for the O (sin) system -- it is genuinely
  NEGACYCLIC-h.  This is why r3-r5 left these primes dense.
- E: cyclic-2m -> CRT z^{2m}-1 = (z^m-1)(z^m+1): one dense cyclic-m product
  (data u_j + u_{m+j}, kernel (CC_j+CC_{m+j})/2) + one dense negacyclic-m
  product (u_j - u_{m+j}, (CC_j-CC_{m+j})/2); E_j = Yc_j + Yn_j,
  E_{m+j} = Yc_j - Yn_j (the 1/2 lives in the kernels).
- O: negacyclic-2m -> y = z^2 with y^m = -1: 3-product KARATSUBA over dense
  negacyclic-m blocks (A0 = even slots, A1 = odd slots; M0 = A0 B0,
  M1 = A1 B1, M2 = (A0+A1)(B0+B1)); O_{2j} = M0_j + (y M1)_j -- the y-shift
  is pure index renaming plus ONE sign at the wrap -- and
  O_{2j+1} = M2_j - M0_j - M1_j.
- Total 5m^2 conv FMA vs dense 2h^2 = 8m^2 (-37.5%), all blocks dense (the
  settled ice-L23/r1 doctrine: Winograd/CRT only at the outer level).
- Correlation -> convolution by data reversal; the negacyclic reversal sign
  (w'_n = -w_{h-n}) bakes into a SIGNED V-INDEX table: the kernel folds
  U_j = x_j + x_{p-j}, V_j = x_j - x_{p-j} ONCE into stack arrays with a
  negated mirror V_{h+j} = -V_j (exact: -(a-b) == b-a), so both conv fills
  are single stack loads and every row is loaded once per chunk at a
  COMPILE-TIME offset (p = 4m+1 per instantiation).  Kernels stretched to
  2m-1 (negacyclic: sign on wrap) so block products index kb[m-1+j-i]
  mod-free -- RP3_CONV reused verbatim.

### What shipped, part 2: the codegen finding (worth as much as the arithmetic)

The r5 record's "compile-time instantiation => block products unroll
register-resident" was FALSE above m = 11: gcc 11 leaves the conv loops
ROLLED with memory-resident accumulators (objdump: rp3_chunk_17 had only
~430 of its 2312 design FMAs straight-line, 855 vmovapd; every accumulation
step round-trips the stack).  Two fixes, chosen PER m by node races:

1. RP_UNROLL (#pragma GCC unroll) on every conv/fill loop forces complete
   peeling; SRA then scalarizes.  Wins while the spill traffic stays under
   the FMA plateau.
2. Above that ceiling, full unroll is a catastrophe (m=25: 7400 vmovapd vs
   3900 FMA, +73% on wallaby-class memory) -- RP2_CONV_BLK instead: j in
   tiles of 8 REGISTER accumulators, i-loop rolled, per i: 1 stack load +
   8 broadcasts + 8 FMA (the dense engine's proven k-quad shape on 5m^2
   work).  Kernel tables get 7 zeroed pad slots so tiles may overread.

Node-raced boundary: rp2 unroll at m <= 9, blocked at m >= 10 (37: unroll
wins 261 vs 271; 41: 384-391 blocked vs 392-455; 53: blocked -13%).
rp3 unroll at m <= 13, blocked at 17 (67/79: blocked LOSES +2-6%; 103:
blocked wins -3% -- rp3's WINO makes 4 smaller conv calls, so its unroll
ceiling sits higher).  The pragmas alone (no other change) moved the r5
outer-C3 primes: **43 -20%, 67 -11%, 103 -38%**.

### Measured on the node (interleaved final binary vs r5 binary, min us/step, chain)

| p | r5 | gen_r6 | delta | engine |
|---|---|---|---|---|
| 37 (B=2 m=8) | 426 | **265** | -38% | rp2 unroll m=9 |
| 41 (B=2 m=8) | 588 | **376** | -36% | rp2 blk m=10 |
| 43 (B=2 m=8) | 440 | **356** | -19% | rp3 unroll (pragma) |
| 53 (B=2 m=8) | 1589 | **944** | -41% | rp2 blk m=13 |
| 61 (B=2 m=8) | 2683 | **1722** | -36% | rp2 blk m=15 |
| 67 (B=1 m=8) | 2224 | **1912** | -14% | rp3 unroll (pragma) |
| 73 (B=1 m=8) | 5262 | **3560** | -32% | rp2 blk m=18 |
| 79 (B=1 m=8) | 5048 | **3496** | -31% | rp3 unroll (pragma) |
| 89 (B=1 m=4) | 12009 | **9021** | -25% | rp2 blk m=22 |
| 101 (B=1 m=4) | 20164 | **17284** | -14% | rp2 blk m=25 |
| 103 (B=1 m=4) | 18863 | **11448** | -39% | rp3 blk m=17 |
| 113 (B=1 m=4) | 31438 | **23598** | -25% | rp2 blk m=28 |

L=31 (graded cell): **84.82 / 84.95 / 86.02** B=16, **84.80** B=1 -- a wash vs
r5 (85.0-85.1 interleaved control), chain output BIT-IDENTICAL to the r5
binary (cmp).  Setup at the largest new-engine prime (113): 1.85 s, far under
the 60 s budget.

Correctness (final binary, all on the node): L=31 single 4.059e-16, map-chain
m=140 2.559e-14 (anchor 2.312e-14), two-step 1.784e-15, repeatable-identical.
Every rp2/rp3 prime: single 4.1e-16..9.8e-16, **two-step 1.9e-15..5.3e-15**
(tol 3e-14; worst margin 5.6x at 113).  Dense-path primes (23 47 59 71 83
107 127) and small primes (3..29) all PASS, untouched.

### What did NOT work, with the number that killed it

- **rp3 one-pass fold (the rp2 fold ported to the outer-C3 kernels):
  43 +15% (406 vs 352), 67 +29% (2447 vs 1892), 103 +39% (15300 vs 10989),
  3/3 each.**  The WINO working set (S/Y/T + 8 block arrays, ~26m slots) is
  already spill-saturated; the extra 9m stack slots + fill indirection cost
  more than the 2h row loads they save -- those rows hit L2 and the two load
  ports absorb them.  Outputs bit-identical (cmp at 67); knob kept
  (-DRP3_FOLD1).  Lesson: fold-once pays in the LEAN kernel (rp2, ~15m
  slots), not the fat one -- same register-budget mechanism as the panel's
  five map-fusion negatives.
- **Blocked conv at rp2 m=9 (+4%, 3/3) and rp3 m=11/13 (+2..6%)** -- the
  unroll/blocked boundary is real and per-family; wrote the raced boundary
  into the instantiation table.
- **Wallaby-local timing at DRAM sizes actively misleads**: local runs said
  rp2 LOSES at 89/101/113 (+4..17%); the node says it WINS -16..-27%.  Two
  different memory systems.  Never gate a size-regime decision on the dev
  host -- the r5 "test at the size regime it targets" discipline needs
  "...on the machine that scores it" appended.

### Borrowed this round, named

- **My own r3 dense engine's k-quad accumulator shape** became RP2_CONV_BLK
  (8 register accumulators sharing one data load per step) -- reused as the
  large-m conv form on 5m^2 work.
- **gen_dense_prime r2/r5**: the exact-tile doctrine (per-m instantiation)
  and the bit-identity-by-lanewise + cmp verification protocol, again.
- **gen_batchlane gen_r4**: the one-lease same-core interleave protocol.
- **ice L23_rader / my r1**: blocks stay dense, structure only at the outer
  level -- the C2-CRT/Karatsuba split is that lesson at even h.
- **docs/literature 11 Tier 1 (GT/Rader-as-vectorization-first)**: this
  round completes the program -- every prime 13..113 with h even or h = 3m
  (odd, 3 coprime m) now runs a sub-dense-count Rader in performant code;
  cited per the brief.

### Operation count (shipped)

31: unchanged (~240k zmm FMA-class + ~90k adds per volume step, bit-identical
to r4/r5).  Even-h primes: per 4 columns per axis, fold 2h loads + 3h add/sub,
5m^2 + ~12m conv FMA (2 CRT products + 3 Karatsuba products, all dense),
combine ~4h -- vs dense 2h^2 = 8m^2.  Outer-C3 primes: unchanged 8m^2 + O(m)
arithmetic, now actually compiled straight-line (<= m=13) or 8-acc blocked
(m=17).  Dense engine: unchanged, still the path for 19 23 47 59 71 83 107 127.

### What I would do next

1. **19 and 127 are the last sub-dense holes** (h = 9 and 63, both 3 | m so
   no C3 CRT; 63 = 7x9 wants Agarwal-Cooley C7xC9 or a two-level plan).  127
   is DRAM-resident so the FMA cut is partially hidden -- pair it with the
   x-band custody blocking (r4 next-step 2, still unpaid) which is now the
   dominant term at 101/113 too (the m=25/28 blocked kernels leave the
   engine clearly memory-shaped: 101 gains only -14% from a -37.5% FMA cut).
2. **Store-order pass for rp2/rp3 combines** (kp/km stores are row-scattered;
   natural-order stores with table indices moved to the O_/T reads) -- the
   r5 item, still cheap, now applies to two engines.
3. **The rp2 blocked kernels' UF/VF fills are the next spill frontier**: at
   m >= 22 the fold + fills are ~9h stack ops outside the products; a fused
   fill-into-first-product form could delete a third of them.
4. Wisdom/race layer: the per-m conv-form table is a per-HOST truth (the
   wallaby/node flip proves it) -- expose -DRP2_FORM overrides so gen_race
   can race both forms per size on CLX/SPR instead of trusting my Ice Lake
   boundary.
5. Harness: r6_ab.sh under build/tryout/gen_rader/ holds the whole session
   (build/even/c3/l31/fold3/gates phases); in/c pairs now cover 13..127 incl.
   41 53 73 89 97 101 109 113 at the B the suite would use.

## Round gen_r7

### Where this round started

r6 leaderboard: **84.801 us/step** at the graded cell (L=31 B=16 m=140), leading
the crossover (gen_dense_prime 120.7, gen_bluestein 288.6, MKL/FFTW 833-883).
The r6 surprise test proved the library generalizes; the rounds 7-8 brief says
spend the queued backlog.  My queue: 19/127 sub-dense arithmetic, x-band custody
blocking at DRAM sizes, store-order pass, UF/VF fill spills.  This round's first
move was neither: I instrumented before choosing (env-gated RP_PROF per-pass
timer in the flat chain, zero-cost when unset), because my own compute model
said p=113 should cost ~8 ms/step and the node said 23.6 -- a 3x gap nobody had
localized.  All node numbers: a80n0, ONE held lease (slot 2, core 4),
interleaved arms (the gen_batchlane r4 protocol).

### Closed BY ANALYSIS before spending windows (do not resurrect without new evidence)

1. **x-band custody blocking is a structural no-op.**  With three contraction
   axes + map, z/y/map are plane-local and x needs all planes, so ANY grouping
   fuses the four ops into at most two memory-sweep groups; the floor is
   2 state reads + 2 writebacks + 1 c read per step, and the r4 custody order
   already sits ON that floor (verified by the RP_PROF numbers: map runs at
   ~12 GB/s, the c stream).  Banding x by z-columns just makes z_next the lone
   cold sweep -- same 5.  The r4/r5 next-step item is dead, closed-form.
2. **gen_layout's NT-store -19% at L=100 cannot transfer here.**  Their win
   killed RFO reads on OUT-OF-PLACE scratch streams.  My chain is fully in
   place: every store hits a line the pass's own load just brought in E state
   -- there is no RFO to eliminate.  (Their r4 attribution table also shows
   prefetch was a WASH at their DRAM size; mine pays -- see below -- because my
   in-place passes leave the LFBs free, theirs were store-occupied.)
3. **The h = 3m, 3|m "last sub-dense holes" (p = 19, 127) stay dense.**  The
   only split that exists is z^3m - 1 = (z^m - 1)(z^2m + z^m + 1); the Phi-part
   product needs LINEAR length-m convolutions (2m-1 outputs = 2m^2 - m FMA each
   in stretched-kernel form, NOT m^2), so Karatsuba gives m^2 + 3(2m^2 - m) =
   7m^2 - 2m vs dense 9m^2 -- a 22% cut buried under ~19m reconstruction adds.
   At p=19 (m=3) dense 81 beats the split's 63+57 outright.  Agarwal-Cooley
   C7xC9 at h=63 needs a Winograd C7/C9 outer (16+ block products, add-heavy,
   high table risk) for a DRAM-shaped size -- not worth a round.  Written down
   with the op count so nobody re-derives it.

### The profile, and the real finding: the large-m kernels were FRONT-END bound

RP_PROF at p=113 m=4 (r6 binary, 23.5 ms/step): **x 7.6 / y 6.7 / map 2.0 /
z 5.1 ms**.  The y-pass streams a 204 KB plane (L2-resident, 23 MB/step = 3.4
GB/s -- nowhere near bandwidth) yet runs 2.8x above its ~2.4 ms load-port
model.  Not memory.  objdump: **rp2_chunk_28 compiled to 10820 instructions /
~75 KB PER CHUNK** (2443 stack vmovapd) -- `-funroll-loops` had peeled the
blocked conv's inner i loop 8x behind the r6 race's back.  75 KB of
straight-line code per chunk is past L1I (32 KB): every chunk re-fetched its
own code from L2.  The r6 record's unroll-vs-blocked boundary was raced
against MANGLED blocked codegen.

### What shipped (arithmetic untouched; every chain output bit-identical to r6, cmp-verified at 31/89/113/127)

1. **RP2_CONV_BLK's i loop pinned rolled** (`#pragma GCC unroll 1`): the tile
   body becomes ~15 instructions, DSB-resident; accumulation order unchanged.
   Node: **-7% at 89, -19% at 101, -8% at 113, -9..11% at 61, -2% at 73, -1%
   at 103**; wash at 41; **+1% at 53 (3/3)** -- so m=13 alone keeps the gcc
   default via an RP2_BLK_IUNROLL redefinition at its instantiation.
2. **Dead map-fused arms compiled out (RP_MAPARM)**: RP_YMAPFUSE lost its race
   in r3 and every caller passes mapc = NULL, yet ~57 never-executed
   rsqrt/rcp ladders (~20 KB) sat inline in every rp2/rp3 chunk (64 KB -> 40
   KB at m=28).  -0.6..-1.8% more at 61/101/103/113, wash at 89/53.  The r31
   kernels keep their arms (31 is port-bound, closed, bit-frozen).
3. **x-pass software prefetch** (rp_pass pf arg, gen_layout r4's fold-prefetch
   recipe): the x-pass walks p+1 row streams at plane pitch -- past the L2
   streamer's ~32 -- so each chunk's row loads demand-miss at DRAM sizes.
   T0 one chunk (RP_PFD=128 B) ahead: x 7.6 -> 7.0 ms at 113 (-8% on the
   pass).  Gated RP_PFMIN_KB = 36 MiB of state+c (p >= 107): raced
   wash-to-NEGATIVE at 89/101, -1.5% at 113, -0.5% at 127.  Bit-identical
   pf/nopf (cmp).

### Measured on the node (same-window interleaved vs the r6 binary, min us/step, graded chain)

| p | r6 (this window) | gen_r7 | delta | what moved it |
|---|---|---|---|---|
| 31 B=16 | 85.07-85.48 | **84.82-85.63 (wash; tryout 84.818)** | 0 | untouched, bit-identical |
| 31 B=1 | -- | **85.63** held-lease (96.2 fresh-lease tryout = the documented ramp signature) | 0 | |
| 61 (B=2 m=8) | 2044-2316 | **1758** | -9..14% | rolled conv + map-arm |
| 73 (B=1 m=8) | 3894-3966 | **3802** | -2% | rolled conv |
| 89 (B=1 m=4) | 8992-10027 | **8275** | -7% | rolled conv |
| 101 (B=1 m=4) | 17176-17345 | **13546** | **-21%** | rolled conv + map-arm |
| 103 (B=1 m=4) | 11426-12376 | **11021** | -3% | map-arm (conv already blocked) |
| 113 (B=1 m=4) | 22777-23259 | **20170** | **-11%** | rolled + map-arm + pf |
| 127 (B=1 m=4) | 48947-49002 | **48342** | -1% | pf only (dense engine) |

Post-fix profile at 113: x 6.3 / y 5.9 / map 2.0 / z 4.8 ms (was 7.6/6.7/2.0/5.1).

Correctness (final binary, all on the node by hand): single + TWO-STEP gates
PASS at 13 29 37 41 43 53 61 67 73 79 89 101 103 109 113 127 (worst two-step
5.3e-15 at 113 vs tol 3e-14); L=31 single 4.059e-16 (B=16) / 4.073e-16 (B=1),
chain m=140 output **bit-identical to the r6 binary** (cmp), generic chains
bit-identical and repeatable at 89/113/127.  Setup worst case 3.0 s (127).

### What did NOT work, with the number that killed it

- **Rolled-blocked conv where full-unroll rules (re-race of the r6 boundary
  against FIXED blocked codegen)**: 37 +27% (330 vs 260), 67 +5..7% (2007 vs
  1913), 79 +13..16% (3876 vs 3385), 3/3 each.  The r6 unroll-side boundary
  SURVIVES; only the blocked side was mismeasured.  Boundary unchanged: rp2
  unroll <= 9 / blocked >= 10 (m=13 on gcc-default), rp3 unroll <= 13.
- **Prefetch below the DRAM gate**: 89/101 wash-to-negative (89: 8349 nopf vs
  8377-8406 pf) -- their state+c (11-16 MB x2) still part-fits the 24 MB LLC.
  Gate set at 36 MiB, knobs RP_PFD/RP_PFMIN_KB kept for the xarch race.
- The 127 dense engine still runs ~2x above its load-port model with hot data
  and small code (rp_chunk_any = 4.5 KB, L1I-resident) -- NOT a code-size
  problem; suspicion is the 8 k-major table streams (63 KB re-walked per
  chunk) plus the exactly-8-accumulator FMA latency balance.  Unspent; see
  next steps.

### Borrowed this round, named

- **gen_layout gen_r4**: the fold-prefetch recipe (adopted, pays at my DRAM
  sizes where their attribution said loads were NOT their binder -- the
  difference is my in-place stores leave the LFBs free); their NT-store
  eligibility analysis (used to PROVE non-transfer here, saving the window).
- **gen_batchlane gen_r4**: the one-lease same-core interleave protocol.
- **gen_dense_prime r3**: the "never conclude from one window" discipline --
  the 89/53 verdicts above took three windows each.
- My own r6 codegen lesson, extended: "compile-time instantiation =>
  register-resident" was false above m=11; NOW also "blocked form => small
  code" was false under -funroll-loops.  Codegen claims get an objdump audit
  EVERY round; the compiler re-litigates them per flag set.

### Operation count (shipped)

Identical to r6 everywhere (the round moved zero arithmetic).  Deltas are
instruction-stream and memory-schedule only: m>=15 blocked kernels ~40 KB ->
~26 KB hot path with rolled tiles (~15-instr loop bodies), all rp2/rp3 chunks
lose ~20 KB of dead map arms, x-pass adds p+1 T0 prefetches per chunk at
p >= 107.

### What I would do next

1. **The dense-engine residue at 127** (x 15.7 / y 14.5 ms vs ~7 ms model,
   both 2x): race a j-major interleaved table layout (per j-step the 8
   broadcast operands land on 2 lines instead of 8 streams), and/or 2-wide
   column chunks (2 SRC loads + 8 broadcasts + 16 FMA -> load ratio 10/16,
   FMA-bound) -- the latter also applies to the rp2 blocked kernels (9/8 ->
   10/16), est. -10..20% on every conv pass, at the cost of doubling the
   stack pencil arrays (watch 48 KB L1D at m >= 22).
2. **Rolled fold/combine loops for m >= 22** (static hot path 26 KB -> ~8 KB,
   fully DSB-resident); the index tables are already runtime arrays, so the
   loops roll without new structure.
3. Store-order pass: STILL unraced (two rounds queued); in-place E-state
   stores make it low-expectation -- race once, then close it either way.
4. The wisdom/race layer should own the RP_PFD/RP_PFMIN_KB and per-m conv-form
   knobs on CLX/SPR (CLX's 1 MB L2 moves both the front-end story and the
   prefetch gate).
5. Harness: r7_ab.sh / r7_race2.sh / r7_race3.sh / r7_gates.sh / r7_final.sh
   under build/tryout/gen_rader/ hold the whole session; in/c pairs now
   include 109b1.  reserve.sh --status needs the slurm PATH prefix from the
   dev host or it false-reports a dead reservation.

## Round gen_r8

### Where this round started

r7 leaderboard: **84.544 us/step** at the graded cell (L=31 B=16 m=140), leading
the crossover (gen_dense_prime 120.3, gen_bluestein 290.0, MKL/FFTW 833-883).
The 31 cell has been port-model-saturated since r4 (this round: wash, output
bit-identical, untouched).  My r7 next-list: (1) 2-wide column chunks -- the
conv/k-quad inner loops are ~9 loads per 8 FMA AND exactly 8 accumulator
chains = 2 FMA pipes x 4-cycle latency with zero slack; (2) rolled
fold/combine for m >= 22; (3) the twice-queued store-order race.  This round
spent all three.  New tools arrived (llvm-mca/uiCA/OSACA at round start, live
PMU counters mid-round via tools/pmu.sh) and were used as designed: models to
choose, node to score, counters to attribute.  All node numbers: a80n0, ONE
held lease per session (slot 0, core 2), interleaved arms (the gen_batchlane
r4 protocol); every arm's chain output cmp-verified BIT-IDENTICAL to the r7
binary at every prime tested (30 primes locally, key sizes on the node) --
per-column arithmetic order is never changed anywhere in this round.

### What shipped: PAIRED-COLUMN (2-wide) chunks, gated per engine and per m

One chunk now processes TWO adjacent zmm columns (8 complex columns), so
every broadcast constant is loaded once per column PAIR:

1. **Dense engine (rp_chunk2 + paired z-quads), all dense primes**: the
   k-quad becomes 16 accumulators sharing 8 broadcasts (per j: 4 stack loads
   + 8 broadcasts + 16 FMA vs 1-wide's 10 loads / 8 FMA), and the 8 k-major
   table streams (63 KB at p=127) are walked once per pair.  **127: 47.2 ->
   35.2 ms/step, -25.5% (3/3)** -- the round's biggest win and the r7
   record's "127 runs 2x above its load-port model" residue largely closed.
   Also **107: -25%, 59: -20%, 23: -15%** (2/2 each).  PMU attribution at
   127: port_2_3 load uops -23% (7.04e9 -> 5.39e9), l1d.replacement -19%,
   512-bit license cycles -26% -- the deleted work is exactly the design's.
   The paired z-quad (two 4-row transpose groups staged interleaved, one
   2-wide chunk at rs=16) carries ~10 points of the 127 win on its own
   (-DRP_NOW2Z raced +11%, 3/3).
2. **rp2 blocked kernels, m >= 25 only (101/109/113), + STORE-ORDER default**:
   2-wide with fold/combine loops rolled, plus the natural-row-order combine
   (stores walk rows 1..h ascending / p-1..p-h descending; the slot
   permutation moves to the T/O_ read side, the +-SG choice folds into a
   two-constant select -- bit-identical because fmadd(o,-SG,T) ==
   fnmadd(o,SG,T) exactly).  vs r7: **101 -5..12%, 113 -8..9% (3/3 each)**.
   Store-order was best 3/3 at 113 and never worse anywhere -- the r5/r7
   queued item finally raced, and it PAYS (mildly) where the kernel is
   biggest.  The final-session race also confirmed 2-wide+stord beats the
   1-wide-rolled form 3/3 at 101 and 113, fixing the m >= 25 boundary.
3. **1-wide kernels with ROLLED fold/reconstruct/combine loops, m = 15..24
   (61/73/89/97)**: the attribution arm (-DRP_W1ROLL, ctl = true r7 build)
   showed rolling those loops is a win ON ITS OWN: **61 -1..4%, 73 -1..10%,
   89 -9..14% (3/3 each; 89 is window-bimodal, 7.2-8.5 ms shipped vs
   8.3-9.5 r7)** -- and at 89 the 1-wide rolled form BEATS the 2-wide rolled
   form.  Below the boundary the roll loses (41 +8..14%, 53 +12..14%, 3/3)
   -- same shape as every unroll boundary this entry has raced: the peeled
   form wins while it fits the front end.

### The codegen finding (this round's lesson worth exporting)

gcc 11 + -funroll-loops turns the PINNED-ROLLED blocked-conv i loop into a
4x unroll with **broadcast rotation**: because the kernel index kb[M-1+jb+j-i]
shifts by one per i, the broadcast loaded for (j, i) is REUSED for (j+1, i+1)
-- the steady-state 2-wide tile body is 85 instructions carrying 64 FMA with
only 8 broadcasts + 8 stack loads (~2 loads + 2 broadcasts per 16 FMA per
column-step), zero spills, DSB-resident.  The same rotation exists in the r7
1-wide loops (49-instr body, 32 FMA) -- so the r7 record's "9 loads / 8 FMA"
static model OVERSTATED the 1-wide load bill; the real 1-wide deficit is the
8-chain latency wall, which is exactly what the counters show the pairing
relieving.  Model notes: llvm-mca (icelake-server) predicted per-column
parity between the two forms because it dispatches ALL zmm FMA to port 0
(one 512-bit pipe); this SKU (Gold 6326) has two, and the r4 port model +
the measured wins say so -- recorded as a model blind spot, uiCA was not
runnable (ext/tools/uiCA is missing its instrData; use llvm-mca + counters).

### What did NOT work, with the number that killed it

- **2-wide with fully-peeled fold/combine: LOSES at every rp2 size** --
  41 +12%, 53 +31%, 61 +13%, 73 +14%, 89 +10%, 101 +8%, 113 +6% (3/3 each).
  rp2_chunk2_28 compiles to 60.9 KB static; the r7 front-end lesson
  re-taught at 2x width.  Knob kept (-DRP_W2FULL).
- **2-wide rolled at m = 10..18**: 41 +8..11%, 53 +28%, 61 +7%, 73 +4% --
  the pairing's doubled stack slot arrays (UF/VF/T/O_ + staging) cost more
  than the shared broadcasts buy while the 1-wide working set still fits.
  Hence RP_W2MIN = 25, with m = 15..24 taking only the rolled-loops half of
  the change.
- **z-pairing off (-DRP_NOW2Z) at 127: +11% (3/3)** -- the paired z is
  load-bearing in the dense win; knob kept for CLX/SPR where 1 MB L2 may
  flip it.
- 103 (rp3, m=17) untouched: pl->m3 plans take the r7 1-wide path verbatim
  (a 2-wide RP3_WINO needs its own macro surgery -- see next steps).

### Correctness (shipped binary, all on the node by hand)

Single + map-chain + TWO-STEP gates PASS at 31(B16,B1) 37 41 43 53 61 67 73
79 89 101 103 109 113 127 13 29 (worst two-step 5.3e-15 at 113, tol 3e-14;
L=31 numbers identical to r7 to the last digit -- outputs are bit-identical
by construction, cmp-verified).  L=31: **85.27-85.88 us/step B=16 (interleaved
wash vs control), 85.21 B=1**, setup 0.007 s.  Worst setup 3.2 s (127) vs the
60 s budget.  Local bit-identity battery: all 30 class primes, every arm
(default / STORD / W2ROLL / W1ROLL / controls) identical to the r7 engine.

### Borrowed this round, named

- **gen_batchlane gen_r4**: the one-lease same-core interleave protocol.
- **gen_dense_prime r3/r5**: the bit-identity-by-lanewise + cmp discipline
  (this round's whole safety story) and never-conclude-from-one-window (the
  89 cell is bimodal across windows; three windows raced).
- **My own r6/r7 codegen lessons, extended**: "objdump audit every round"
  caught the 60 KB peeled 2-wide kernels BEFORE any node window was spent on
  a bad default; the discovered broadcast-rotation unroll revises the r7
  static load model (above).
- **tools/TOOLS.md discipline (round-8 brief)**: llvm-mca to compare loop
  shapes (with its port-0-only FMA blind spot recorded), PMU counters
  (NOTICE, mid-round) to attribute the 127 win to deleted load uops.

### Operation count (shipped)

Arithmetic identical to r6/r7 everywhere (this round moved zero FLOPs; all
deltas are instruction-stream, broadcast-traffic and store-order only).
Dense: per column-pair per j, 4 row loads + 8 broadcasts + 16 FMA (halved
broadcast + table traffic per column); z quads paired the same way.  rp2
m >= 25: five 2-wide blocked convs (rolled tiles, broadcast-rotated by gcc),
rolled fold/combine, natural-order stores.  rp2 m = 15..24: r7 arithmetic
with rolled fold/combine loops (~20-instr bodies, 7.5-9 KB kernels vs
r7's 20-30 KB).

### What I would do next

1. **2-wide RP3_WINO for the outer-C3 kernels** (43/67/79/103): the same
   pairing mechanically applied to the 4-block-product WINO; 103 (DRAM-edge,
   m=17 blocked) is the natural first target -- expect a fraction of the
   dense engine's win since conv broadcasts are already amortized 4 ways.
2. **The 89 bimodality**: the same binary reads 7.2-9.5 ms across windows
   (sd inside a window 0.1-0.6%).  11.5 MB state+c sits at the 24 MB LLC
   boundary with neighbors active; a plane-blocked custody variant could
   shrink the resident set.  Measure with pmu.sh l1d/llc counters across
   modes before building anything.
3. **Dense-prime table layout** (r7 item 1, still unspent): j-major
   interleaved tables would cut the dense engine's 8 k-major streams to 2;
   the 2-wide already halved that traffic, so re-model with counters first.
4. The per-m form table is now three-way (peeled-1w / rolled-1w / 2-wide+sd)
   and per-HOST: expose to gen_race on CLX/SPR via the knob set
   (RP_NOW2 / RP_W2MIN / RP_NOW2D / RP_NOW2Z / RP_W2FULL / RP_W1ROLL /
   RP_W1FULL / RP2_NOSTORD).
5. Harness: r8_ab.sh under build/tryout/gen_rader/ holds the whole session
   (build/w2/build2/knobs2/mid2/build3/final/pmu/l31/gates phases); note the
   final-phase "ctl" was rebuilt from edited source, so its 61/73/89 rows
   compare shipped-vs-shipped (noise) -- the true r7 controls for those
   sizes live in the w2/knobs2/mid2 phases.  in/c pairs now include
   23b2/59b1/107b1.  uiCA is broken (missing instrData); pmu.sh works and
   needs no lease discipline beyond the usual same-core rule.

### Measured summary (the reply line)

L=31 B=16 m=140: **85.27 us/step** (wash vs r7's 84.5-84.8 board, bit-identical
output); **B=1: 85.21**.  Single rel_l2 4.059e-16, two-step 1.784e-15,
map-chain m=140 2.559e-14 -- all identical to r7.  Class duty vs r7, node
same-window: **127: -25.5% (35.2 ms), 107: -25%, 59: -20%, 23: -15%,
113: -8..9%, 101: -5..12%, 89: -9..14%, 73: -1..10%, 61: -1..4%**; 31/37/41/43/
53/67/79/103 unchanged.


## Round gen_r9

### Where this round started

r8 leaderboard: **84.745 us/step** at the graded cell (L=31 B=16 m=140), leading
the crossover (gen_dense_prime 113.5, gen_bluestein 292.6, MKL/FFTW 833-883).
The rounds 9-10 brief is counter-directed; the PMU audit names gen_rader at 31
THE champion signature (IPC 2.15, p0+p5 = 1.60/2.0 -- "what done looks like"),
so the graded cell stays bit-frozen.  **Constraint that shaped the whole round:
both Ice Lake nodes were held by other users from round start (our hold first
in queue, jobs 438854/438856 PD, ssh denied; scheduler worst-case Aug 27); the
monitor's NOTICE says develop model-side with the analyzers and wallaby
correctness runs.  NO node window existed this round -- every timing number
below is a WALLABY number and labeled so.**  Wallaby fact established this
round that upgrades all its numbers: it is a **Xeon Gold 6448Y = SAPPHIRE
RAPIDS** -- one of the three graded architectures, not a random dev box.

### The four brief avenues, triaged for this entry

1. **Bank the picks: nothing to bank, verified.**  create() contains no timing
   race and no host-dependent branch -- engine choice (m2/m3/dense), conv
   form, pairing gates, arena and prefetch gates are all pure functions of
   (L, compile flags).  Demonstrated, not just argued: 5 consecutive
   create()+chain cycles at 103 on wallaby produce byte-identical outputs.
   The L=25 instability mechanism structurally cannot happen in this entry.
2. **Two-axes fusion**: gen_pfa_large/gen_powp cells, not my geometry (and
   for MY chain the r7 closed-form stands: custody already sits on the
   5-sweep floor).
3. **Champion dashboard**: 31 closed by counter evidence; the dashboard duty
   for the other primes is staged as the pmu phase of r9_ab.sh (needs the
   node).
4. **Port-1 co-issue**: not attempted -- every candidate side-task in this
   engine (map, fold tails) is data-dependent on the 512-bit stream, and the
   panel's five map-fusion negatives say relocated work loses when it
   lengthens the critical path.  Left for a round with node time.

### What was built: PAIRED-COLUMN RP3_WINO at p = 103 -- and then DEFAULTED OFF by measurement

rp3_chunk2_17 + RP3_WINO2: the r8 pairing applied to the outer-C3 kernel
family, m = 17 only (the one rp3 prime on the blocked conv and the one that
is DRAM-resident; the full-unroll rp3 kernels 43/67/79 stay 1-wide -- the r8
m=10..18 rp2 lesson).  Fold via js/js2 two-row loads (RP3_FOLD1 lost in r6
and pairing doubles its stack bill), glue loops rolled, combine kp/km-
scattered, dispatch through rp_w2_on/rp_chunk_any2, z pairs automatically
through rp_zquad2.  Stack 578 zmm slots = 37 KB.  Knob: **-DRP_W23 enables;
DEFAULT OFF** -- the default build is the r8 engine bit for bit.

Why OFF, when the r8 evidence said pairing wins at DRAM sizes -- the round's
main measured finding:

- Raced on wallaby (chain B=1 m=4, 3 interleaved rounds, sd 0.1-0.5%):
  1-wide 7062/7091/7104 vs 2-wide **9587/9593/9593 us/step = +35%, 3/3**.
- CALIBRATION on the same host, same session: the r8-shipped rp2 2-wide at
  113 -- which the NODE proved wins -8..9% -- reads only **+2.8%** on wallaby
  (16251-16300 vs 15717-15882, 3/3).  So the known dev-host anti-pairing bias
  is ~3-11 points on this kernel family; it CANNOT explain +35.  The loss is
  a property of THIS kernel.
- RP_PROF attribution: x 2.03 -> 2.62 ms (+29%), y 1.92 -> 2.59 (+35%),
  z 1.32 -> 2.30 (+74%).  Every paired pass loses; the paired z (69 KB
  staging + slots) loses worst.
- Mechanism, consistent with all of it: rp3's glue-to-conv ratio is ~2x
  rp2's (8 conv calls of m^2 = 289 vs rp2's 5 of 784, with fold/WINO-
  build/reconstruct/T/combine glue scaling with H = 51 and M).  The pairing
  buys shared broadcasts INSIDE the conv tiles and pays doubled stack arrays
  across ALL the glue -- at this glue ratio the price exceeds the prize, at
  least on SPR.  And SPR is a graded xarch machine: even if ICL flips the
  verdict, a 35% SPR loss is exactly the "wins only on Ice Lake" flag the
  brief warns about, which is gen_race's knob to arbitrate, not a default.
- The ICL verdict stays unmeasured this round; r9_ab.sh phases build/cmp/
  race103/race97/pmu/l31/gates run it in one command when the hold lands.
  Adopt only on a clean 2/3-min ICL win AND a defensible xarch story.

### Verified this round (wallaby -- correctness, plus the timing above)

- **Bit-identity battery, all 30 class primes**: default build vs the TRUE r8
  binary (bin_r8_local, built last round from r8 source) AND vs the -DRP_W23
  build: all identical, execute; chain m=4 at 103 identical across all three.
- **Gates at 103** (both arms): single rel_l2 9.085e-16 (tol 1e-12),
  map-chain m=4 7.410e-15, **two-step 5.006e-15** (tol 3e-14); L=31 B=16
  single 4.059e-16 -- identical to r8 to the last digit, as they must be.
- **Codegen audit**: rp3_chunk2_17 = 3510 instructions / 20.3 KB static
  (SMALLER than the 1-wide's 32.5 KB peeled form), 564 vmovapd vs the r7
  pathology's 2443, conv i loops rolled as pinned.  The static shape is
  clean -- the loss is dynamic (stack traffic volume), which no static
  histogram shows: static audits catch code-SIZE pathologies, not
  working-set ones.
- llvm-mca NOT used to adjudicate: its ICL model dispatches all 512-bit FMA
  to port 0 (the r8-recorded blind spot) -- exactly the quantity pairing
  changes.  Measured instead (see above), which is what settled it.

### What did NOT work, with the number that killed it

- **The 2-wide RP3_WINO as a default: +35% on wallaby/SPR, 3/3** (the whole
  story above).  Code kept behind -DRP_W23 with the ICL race staged; the r8
  extrapolation "pairing wins at DRAM sizes" is now bounded: it held where
  conv dominates (dense: 100% conv; rp2: 5 big products), it FAILS where the
  glue ratio is high (rp3 WINO).  Written down so the 2-wide idea is not
  re-applied to glue-heavy kernels on faith.
- **A harness trap that invalidated my first A/B, written down so nobody
  repeats it: `impl` is a SYMLINK to `impl_N` for the current round.**  My
  "control built from impl_9/gen_rader.c" WAS my edited file; the first
  30-prime battery and first timing A/B compared the new engine against
  itself (the timing read a perfect wash -- both arms ~9.8 ms -- which is
  what quietly flagged it).  Every verdict above was re-established against
  bin_r8_local (the r8 session's binary, mtime Aug 25 14:48) and a -DRP_W23=
  off build.  Rule: a control must be a BINARY from the previous round or a
  flag-disabled build -- never "the snapshot directory", which is live.
- **97 (m=24) 2-wide, prefetch-at-103 (RP_PFMIN_KB=34816)**: not decidable
  without the node; staged as bin_r9w24 / bin_r9pf in r9_ab.sh.

### Borrowed this round, named

- **My own gen_r8 machinery**: RP2_CONV_BLK2 verbatim as the paired conv;
  the interleaved-slot layout and rolled-glue doctrine; the RP_* knob
  taxonomy.
- **gen_dense_prime r3/r5**: bit-identity-by-lanewise + cmp battery -- this
  round it was the ONLY cross-round safety story, and it caught the symlink
  trap when re-run honestly.
- **gen_batchlane r4 protocol, adapted**: same-core interleaved arms, 3
  rounds, min-sets -- run on wallaby cores since no lease exists; wallaby
  verdicts labeled as SPR data, not ICL proxies.
- **results/PMU_AUDIT.md**: the champion-signature framing and the avenue-1
  determinism duty (discharged: deterministic by construction, 5x verified).
- **The r6 lesson, sharpened by calibration**: "wallaby misleads at DRAM
  sizes" is now quantified (~3-11 points anti-pairing on this family via the
  113 control experiment) -- which is what made +35% readable as a real
  loss rather than bias.  Calibrate the proxy with a node-proven case before
  trusting any proxy verdict; that calibration was this round's best trick.

### Operation count (shipped)

The shipped default moves ZERO instructions anywhere: it is the r8 engine bit
for bit (30-prime cmp).  The -DRP_W23 arm (off): at 103 per column-pair per
conv tile, 2 stack loads + 8 broadcasts + 16 FMA vs 2 x (1 + 8 + 8) 1-wide;
kernel-table broadcasts halved; glue stack arrays doubled (37 KB slots + 32 KB
z staging) -- the doubling is what the SPR race says dominates.

### What I would do next

1. **Run r9_ab.sh when the hold lands** (build / cmp / race103 / race97 /
   pmu / l31 / gates).  Expected: ctl ~11.0 ms at 103; if w23 wins on ICL
   despite SPR, it is a race-layer (per-host wisdom) candidate, not a
   default -- flag it to gen_race with both numbers.
2. **If w23 loses on ICL too, close the rp3-pairing item permanently** and
   record the glue-ratio boundary next to the r8 m-boundaries: pairing pays
   at glue:conv below ~1:4 (dense, rp2 m>=25), not at rp3's ~1:2.
3. **The 89 bimodality** (r8 item 2, still unspent): pmu.sh l1d/llc counters
   across windows before building anything.
4. **Dense-engine j-major table layout** (r7 item, thrice queued): re-model
   with counters at 127 first.
5. Harness: r9_ab.sh under build/tryout/gen_rader/ holds the session; its
   build phase generates in97b1/c97b1; reserve.sh --status needs the slurm
   PATH prefix on wallaby; **impl is a symlink to the live impl_N -- controls
   come from prior-round binaries or flag-off builds, never from impl_N**.

### Measured summary (the reply line)

No Ice Lake window this round (nodes queued-busy; NOTICE-directed model-side
round); the shipped binary is bit-identical to r8 everywhere, so the board
numbers stand: **L=31 B=16 m=140: 84.745 us/step, B=1: 85.21; single rel_l2
4.059e-16, two-step 1.784e-15**.  New: 2-wide RP3_WINO at 103 built and
verified (gates 9.085e-16 / 5.006e-15; 30-prime bit-identity), shipped
DEFAULT OFF on a calibrated SPR loss (+35% vs the +2.8% bias ceiling); ICL
race staged in r9_ab.sh.

## Round gen_r10

### Where this round started

r9 leaderboard: **85.210 us/step** at the graded cell (L=31 B=16 m=140; gen_race
84.694 rides my engine), leading the crossover (gen_dense_prime 109.955,
gen_bluestein 272.6, MKL/FFTW 848-883).  The r9 round was node-less; its entire
output was a staged Ice Lake decision script (r9_ab.sh) plus two SPR data
points.  This round the hold LANDED (a81n2, Gold 6326 Ice Lake — the same host
that scored r9), so the round was spent exactly as the r9 next-list said: run
the staged races, adopt what wins, close what loses.  All node numbers: a81n2,
ONE held lease (slot 1, core 3), interleaved arms (the gen_batchlane r4
protocol); /tmp/perf was re-staged by the monitor, so pmu.sh works on a81n2.

### The three staged verdicts, all decided on the node

1. **2-wide RP3_WINO at p=103 (-DRP_W23): LOSES on Ice Lake too — +13%
   (13.10-13.17 vs ctl 11.44-11.62 ms/step, 3/3 non-overlapping).  CLOSED
   PERMANENTLY.**  With r9's SPR +35% this kernel now loses on BOTH graded
   architectures, so it is not even a race-layer candidate.  The glue-ratio
   boundary is confirmed as a real law, not an SPR artifact: column-pairing
   pays where conv products dominate the chunk (dense engine: 100% conv,
   -25.5% at 127; rp2: 5 products of 4m^2, -5..9% at m>=24) and loses where
   glue is ~half the work (rp3 WINO: 8 products of m^2 vs fold/build/
   reconstruct/combine glue scaling with H=51 and M).  Do not re-apply
   pairing to glue-heavy kernels; the knob stays only as a CLX curiosity.
2. **rp2 2-wide+stord at m=24 (p=97): WINS — RP_W2MIN default 25 -> 24, the
   round's one adoption.**  First race (vs the -DRP_W23 binary as the 1-wide
   arm): 11.07-11.21 vs 11.62-11.74 ms/step, -4..6%, 3/3 non-overlapping.
   A later noisy window (neighbor compile storm, sd 5-12%) read the
   OPPOSITE ordering on min-sets — re-raced 4 arms x 4 rounds in a clean
   window with BOTH binaries per side (r9ctl+r9 1-wide vs r10+r9w24 2-wide)
   to exclude the code-layout confound: 1-wide 11.54-11.97, 2-wide
   10.92-11.22, 4/4, both binaries per side agree.  The r8 "97 = m24 sides
   with its 89/101 neighbors" INTERPOLATION was wrong; every rp2 boundary
   is now raced, not interpolated: 1-wide rolled m=15..22 (61/73/89),
   2-wide+stord m>=24 (97/101/109/113; no prime has m=23).
3. **x-pass prefetch-only at 103 (RP_PFMIN_KB=34816 without W23 — the
   staged pf arm had W23 baked in, so a fresh arm was built): WASH**
   (10.86-10.99 vs ctl 10.93-11.05, overlapping min-sets, +-0.5%).
   RP_PFMIN_KB stays 36 MiB (p>=107).  Item closed.

### Verified (node, final binary)

- **Bit-identity**: 30-prime execute battery vs bin_r9ctl — all identical
  (the change only affects which kernel form runs at m2=24, and the 2-wide
  form preserves per-column arithmetic order); chain outputs identical at
  31(B16)/89/97/103/113.
- **Gates**: L=31 B=16 single 4.059e-16, two-step 1.784e-15, map-chain
  m=140 2.559e-14 (anchor 2.312e-14) — identical to r4-r9 to the last
  digit, as a bit-identical binary must read.  L=97: single 5.548e-16,
  two-step 3.140e-15 (tol 3e-14).
- **Graded cell**: 85.20-86.70 us/step B=16 (min 85.20, sd 0.04%), 85.31-85.62
  B=1 — the usual; the cell remains bit-frozen at its port model (since r4).
- **Class duty timings this window (bin_r10, chain B=1 m=4)**: 89 7.14 ms
  (the fast mode of its bimodal 7.2-9.5), 97 10.57-11.22 (was ~11.6-11.7
  1-wide), 103 11.02, 113 18.50 (board 23.6 — window/node variance on a
  bit-identical binary; a81n2 reads this size faster than a80n0 did),
  127 34.81 (board 35.2, consistent).

### PMU dashboard (brief avenue 3, the champion-signature duty — first counters at the big primes)

tools/pmu.sh on bin_r10, chain B=1 m=4, whole-process counters (NOTE: these
include create()'s dense-reference self-check, which at these L is 0.4-2.9 s
of pure dense FMA vs a 0.1-0.6 s timed region — so the true chain-only
p0+p5/cycle ratios are LOWER than these):

| p | p0+p5 per cycle | IPC | l1d.replacement |
|---|---|---|---|
| 89 | 1.21 | 2.48 | 116M |
| 97 | 0.99 | 1.94 | 323M |
| 103 | 0.90 | 1.82 | 170M |
| 113 | 0.89 | 1.73 | 563M |
| 127 | 0.91 | 1.84 | 854M |

Every large prime sits at ~0.9 vs the champion 1.60 with heavy L1 fill —
per the brief's rule, traffic headroom everywhere at 97..127.  This is the
counter-side confirmation of the r7 profile: the custody chain is on its
5-sweep DRAM floor (closed-form, r7), so the remaining traffic is CHUNK-
LOCAL — at m>=24 the 2-wide stack arrays (54 KB at m=28) exceed the 48 KB
L1D, so every chunk round-trips its own stack through L2.  A lever would
have to shrink the per-chunk working set, not the volume sweeps; nothing
cheap presents itself (splitting the five conv products re-reads the fold
arrays; narrower pairing gives back the broadcast sharing).  Recorded as
the shape of the residue, not an action item.

### What did NOT work, with the number that killed it

- **RP_W23 on ICL: +13%, 3/3** (closed above — the r9 build/verify cost is
  now fully amortized into a two-architecture negative law).
- **pf-only at 103: wash** (closed above).
- **The official tryout.sh path mid-round read 172 us at the graded cell
  with MKL at 1716 — both exactly 2x their canonical values**: a neighbor's
  compile storm (gen_planner races candidate builds) had the node's
  uncontended-core assumption broken.  Held-lease interleaved runs on core 3
  in the same session read 85.2/849-canonical.  Lesson re-sharpened: when
  every backend in a window scales by the SAME factor, it is the window,
  not the code — check a library reference before reacting.  Also: a noisy
  window can INVERT a 5% verdict on min-sets (the 97 scare above); the
  4-arm/both-binaries re-race is the antidote and is now my standard form
  for adopting anything under ~8%.

### Borrowed this round, named

- **gen_batchlane gen_r4**: the one-lease same-core interleave protocol, as
  every round.
- **gen_dense_prime gen_r3**: never-conclude-from-one-window — this round it
  saved the 97 adoption from a noise inversion (and nearly saved me from
  myself in the other direction).
- **My own r9 staging discipline**: the entire round executed a script
  written blind a round earlier; total time from lease to all-verdicts was
  under an hour.  Staging decision scripts during node-less rounds is now a
  proven pattern for this panel.
- Read gen_dense_prime/gen_layout/gen_race r9 records for adoptables:
  gen_dense_prime's 24-accumulator z applies to their GEMM-form z pass (mine
  is the transpose-quad form at its port-5 model, closed since r4);
  gen_layout's new gl_map4/ymm co-issue primitives are the avenue-4 shape I
  struck in r9 for data-dependence reasons — both correctly not taken.

### Operation count (shipped)

Arithmetic identical to r6-r9 everywhere.  The only behavioral delta vs r9:
p=97 runs rp2_chunk2_24 (2-wide+stord: five 2-wide blocked convs, rolled
glue, natural-order stores) instead of the 1-wide rolled kernel — per
column-pair per conv-tile step, 2 stack loads + 8 broadcasts + 16 FMA vs
2 x (1 + 8 + 8); outputs bit-identical.

### What I would do next

1. **The class is converged on this hardware.**  31 has been at its port
   model since r4; the counter dashboard now shows every large prime on its
   traffic floor with the chunk-local stack residue named (m>=24 stacks >
   L1D).  The one open measurement is the **89 bimodality** (7.2-9.5 ms
   across windows, r8 item): catch a SLOW window with pmu.sh l1d/llc
   counters and diff against this round's fast-mode baseline (116M
   l1d.replacement) — it needs luck with window timing, which is why it
   keeps surviving rounds.
2. **For gen_race**: the per-m form table now has three raced boundaries on
   ICL (unroll<=9 / 1w-rolled 10..22 / 2w+sd >=24 for rp2; 13 / 17 for
   rp3) and ONE SPR calibration point (113 2-wide +2.8%).  On CLX/SPR the
   right move is racing RP_W2MIN in {22, 24, 25, 99} per host — the knobs
   are all exposed.
3. If a future round reopens arithmetic: the only untried sub-dense idea
   left in class is a two-level split at h=63 (p=127), and the r7 op-count
   analysis says it loses; re-derive only against new evidence.
4. Harness: reserve.sh --status from the dev host needs
   PATH=/opt/software/slurm-19.05.8.1/bin prefixed or tryout.sh
   false-reports no reservation; tryout's remote chain-leg check.py still
   gets an unexpanded '$W/c.bin' (run map-checks by hand, as every round);
   /tmp/perf is staged on a81n2 and pmu.sh works there.

### Measured summary (the reply line)

L=31 B=16 m=140: **85.20 us/step min (85.2-86.7 across rounds), B=1 85.31**;
single rel_l2 4.059e-16, two-step 1.784e-15, map-chain 2.559e-14 — bit-identical
to r8/r9.  Adopted: RP_W2MIN=24 (p=97: **~10.9-11.2 vs 11.5-12.0 ms/step,
-4..6%, 7/7 clean-window rounds**; gates 5.548e-16 / 3.140e-15).  Closed on
node evidence: rp3 pairing (+13% ICL / +35% SPR), prefetch-at-103 (wash).

## Round gen_r11

### Where this round started

r10 leaderboard (a81n2): **84.838 us/step** at the graded cell (L=31 B=16
m=140; gen_race 84.519 rides this engine), leading the crossover
(gen_dense_prime 109.865, gen_bluestein 272.9, MKL/FFTW 848-883).  The r11
brief is ALL HANDS ON L=100 with a MANDATORY counter protocol; my class has
no composite cells, so this class's angle on the large-size round is its own
DRAM-regime primes (101..127 -- same traffic-bound regime as 100) plus the
measurement duties.  Node: a80n0 (job 438881), ONE held lease (slot 0,
core 2), interleaved arms as always.  NODE-IDENTITY NOTE (gen_dense_prime
r11 saw the same): a80n0 is NOT a81n2 -- same-node contrasts only.

### Read first, and what it changed (the cumulative-round duty)

- **gen_dense_prime r11 SETTLED the brief's open disagreement** with a
  calibration microbench (r11dev/ubcap.c -- run it on any new host): there
  is NO ~2.1 total-uop cap; the node sustains 3.0 total vector uops/cyc.
  The real wall is a **512-bit L1 access ceiling of ~1.12/cyc, loads and
  stores POOLED** (ymm and 8-byte broadcasts have their own 2/cyc class);
  zmm FMA saturation is only reachable below 0.56 zmm loads/FMA.  This
  retroactively explains my r7 "dense engine runs 2x above its load-port
  model" mystery: every port floor this entry ever computed assumed 2x512b
  loads/cyc -- the actual SKU does half that.
- **gen_batchlane r11 took L=100** (4059 us, 2.14x) with within-volume SoA;
  **gen_layout r11** measured the host at THP=madvise (driver posix_memalign
  buffers get ZERO huge pages) and shipped a zero-copy chain-state re-home
  (~-0.5% at L=100 m=64, dtlb walks -96%); **gen_pfa_small r11** recommends
  the DIFFERENTIAL counter method (samples=4 minus samples=2 cancels
  create/driver pollution).

### What was built -- and REVERTED on node evidence (the shipped file is the r10 engine, comment/description-only delta, instruction stream identical)

Two zero-copy changes to the FLAT chain (p >= 67: 67 71 73 79 83 89 97 101
103 107 109 113 127), both borrowed from gen_layout r11's recipe:

1. **memcpy deletion**: the flat chain opened with memcpy(out, x0, vol) +
   an IN-PLACE prologue z; the padded path (p <= 61) has read x0 directly
   since r5.  Rebuilt the prologue as out-of-place z-quads x0 -> state
   (-2 volume sweeps per chain call on paper).
2. **Zero-copy THP re-home**: chain state in a gl_map_huge arena at a
   +2048 B page phase (the r3 anti-alias trick vs the page-aligned driver
   c volume); prologue z reads x0 directly, LAST step's map writes
   final_out directly -- zero extra copies claimed.

Verified before racing: outputs BIT-IDENTICAL to the r10 binary at 13 37 61
67 89 97 103 113 127 (execute + chain m=4, wallaby battery), 31 B=16 m=140,
89 B=2 (shared-arena batch loop), repeatable; node cmp identical at 113/127;
control = a binary built from impl_10 source (the r9 symlink-trap rule).

**Node races (a80n0 core 2, held lease, interleaved arms, min us/step):**

| case | ctl (r10) | nrh (memcpy-fix only) | ship (fix + re-home) |
|---|---|---|---|
| 127 m=4, 3 rounds | 34977-35993 | 35521-36526 | 35521-36561 |
| 127 m=8, 2 rounds | 34394/35163 | 34401/34484 | 34640/34594 (wash) |
| **127 m=2, 4 rounds** | **36091-36225** | 36305-36924 (**+0.6%, 4/4**) | 37586-38054 (**+4.1%, 4/4 non-overlapping**) |
| 113 m=2, 3 rounds | 19699-20451 | 19150-21197 | 19539-21268 (inconclusive window) |

m=2 weights the per-chain fixed cost 4x heavier than m=8; the arms converge
at m=8 and separate cleanly at m=2 => ALL deltas live in per-chain fixed
costs, and both changes LOSE there.  **REVERTED both.  Mechanisms, so
nobody re-derives this:**

- **An in-place custody chain pays no RFO anywhere** (r7's NT-store
  analysis, now paid forward): the "zero-copy" re-home exit is NOT zero
  cost -- the last-step map writing a cold final_out adds a whole 32 MB
  RFO read per chain that the in-place exit never paid (~2.7 ms at 127 =
  exactly the ship-minus-nrh gap).
- **glibc's 32 MB memcpy writes NT (no RFO) and the in-place z_0 re-owns
  lines via its own loads** -- so memcpy + in-place z is CHEAPER than one
  out-of-place RFO'd z_0 despite 64 MB more nominal traffic.  "Fewer
  sweeps" accounting that ignores RFO and NT gets the sign wrong.
- **The TLB prize was measured, and it is small for THIS engine**: dtlb
  walk_active at 127 m=8 = 117M cycles of 19.0G (0.6%); the re-home cuts it
  to 48M (-59%) = a ~0.3% prize.  My x-pass touches p pages per chunk but
  consecutive chunks re-touch the SAME pages, so the STLB absorbs it --
  unlike gen_layout's L=100 demo geometry.  Their recipe pays on
  out-of-place/staged chains at large m (their cell: m=64); it does NOT
  transfer to in-place custody chains at the m ~ 4 the suite implies at
  large L.  This is my r5 arena lesson ("per-volume overheads do not
  amortize at small m") re-derived in zero-copy clothing -- the boundary
  is now quantified from both sides.

### The counter protocol (mandatory) -- dashboards refreshed in the corrected currency

Whole-process (tools/pmu.sh, ctl binary, chain B=1 m=4; create()'s dense
self-check pollutes ~30-60% of these totals -- r10 caveat holds):

| p | p0+p5/cyc | TOTAL vector/cyc | l1d.repl | IPC |
|---|---|---|---|---|
| 113 | 0.90 | 1.45 | 573M | 1.76 |
| 127 | 0.85 | 1.48 | 867M | 1.73 |

Differential per chain step (samples=4 minus samples=2, the gen_pfa_small
r11 method; 8 steps per differential):

| p | p0 | p5 | p2_3 | p4_9 | l1d lines/step | ~cycles/step |
|---|---|---|---|---|---|---|
| 113 | 29M | 29M | 22M | 9.5M | **9.2M (~590 MB)** | ~55M |
| 127 | 57M | 58M | 92M | 8M | **13.9M (~890 MB)** | ~112M |

Reading: neither pool is anywhere near saturation (p0+p5 ~1.05/cyc, 512b
accesses ~0.5/cyc vs the 1.12 ceiling) -- these kernels are LATENCY-bound on
chunk-local L1 thrash, exactly the r10 residue: at 113 the measured 9.2M
line fills/step match ~1700 fills per chunk-pair against an ~83 KB/chunk
working set (54 KB slot arrays + 29 KB source rows) that cannot fit 48 KB
L1D at m=28; at 127 the dense engine's per-chunk 63.5 KB table walk
dominates (890 MB/step through L1 vs a 160 MB sweep floor).  My whole-chain
numbers corroborate gen_dense_prime's verdict: nothing here is uop-capped;
deleting cross-L2 round trips, not port surgery, is the only lever -- and
the r8 pairing already took the accessible half of it.  CAVEAT for the
differential method at small timed regions: the CYCLES differential at 113
came out negative (create-time variance exceeds 2 chains of work) -- port
and l1d counters difference cleanly, cycles need the timing runs instead.

### Struck at the whiteboard (gen_dense_prime r11's costing discipline)

A Rader cross-class entry at L=100: the only Rader-shaped angle is
Rader-25 on Z_25* (C20, even h=10 -> my rp2 machinery at m=5) inside a
PFA(4x25) axis -- but a 25-point via two twiddled 5-stages costs ~404
vector FP (gen_pfa_small r11 measured form) vs the fold+conv+glue of a
Rader-25 at comparable count with far worse glue ratio, and the incumbent
cell (batchlane 4059 us) is traffic-, not arithmetic-, limited.  No Rader
entry can win 100; recorded so nobody re-costs it.

### What did NOT work, with the number that killed it

- The round's whole build: memcpy deletion **+0.6% at 127 m=2 (4/4)**,
  zero-copy THP re-home **+4.1% at 127 m=2 (4/4)**, both wash at m=8 --
  reverted; mechanisms above.  The staged sources/logs live in
  build/tryout/gen_rader/r11/ (race1/race_m/race_m2/race113_m2/pmu_*.log).
- First local battery read all-FAIL: gen_input.py silently writes nothing
  without the env sourced -- source env.sh in EVERY shell, check inputs
  exist before trusting a cmp verdict.
- Driver `--chain 1` is a silent no-op (no output line): use m=2 vs m=8 to
  decompose per-chain fixed costs, never m=1.

### Borrowed this round, named

- **gen_layout gen_r11**: the zero-copy re-home recipe + gl_thp_bytes THP
  finding -- adopted, raced, and REJECTED here with the regime boundary
  contributed back (theirs stands at L=100 m=64 out-of-place; mine closes
  in-place custody chains at small m).
- **gen_dense_prime gen_r11**: the corrected machine currency (512b access
  pool 1.12/cyc, no total-uop cap) -- adopted as this entry's accounting
  basis; their a80n0-vs-a81n2 node-identity warning; the whiteboard-costing
  discipline; ubcap.c noted for any future host.
- **gen_pfa_small gen_r11**: the differential counter method (with the
  cycles caveat found here).
- **gen_batchlane gen_r4**: the one-lease same-core interleave protocol.
- My own r5/r7/r9 lessons, applied: control from a prior-round binary
  (symlink trap), per-volume overheads vs small m, in-place = no RFO.

### Operation count (shipped)

Identical to r10 at every size: the shipped source differs from r10 by the
header comment and description string only; objdump shows an identical
instruction stream (rodata displacements shift by the longer string), and
chain outputs are cmp-identical on the node at 113/127 and at the graded
cell.  All r10 gate digits inherited exactly (re-verified fresh on a80n0:
31 B=16 single 4.059e-16, two-step 1.784e-15, map-chain m=140 2.559e-14
anchor 2.312e-14; 113 single 9.758e-16, two-step 5.335e-15, chain m=4
7.210e-15).

### Measured summary (the reply line)

L=31 B=16 m=140 on a80n0: **85.448 us/step min (sd 0.08%), MKL 848.6 same
window**; B=1 96.1-96.7 this window (the documented short-unit ramp
signature on a fresh a80n0 core; a81n2 board form: 85.2-85.3).  Single
rel_l2 4.059e-16 (B=16) / 4.073e-16 (B=1), two-step 1.784e-15, map-chain
2.559e-14 -- bit-identical lineage since r4.  Class duty this window:
113 18.48-18.94 ms, 127 34.98-35.25 ms (ctl/final forms, same engine).

### What I would do next

1. The class remains CONVERGED on this hardware (r10 verdict, now with the
   corrected-currency dashboards agreeing): the residue at 97..127 is
   chunk-local L1 thrash from working sets that exceed L1D by construction;
   shrinking them meaningfully (E/O phase split, VF-mirror deletion) saves
   only ~3-13% of the per-chunk footprint against a 2x overshoot -- modeled
   this round, not worth a window until someone shows a >30% footprint cut.
2. The **89 bimodality** (r8 item) is still the one open measurement --
   needs a slow window with pmu.sh, still luck-gated.
3. For gen_race on CLX/SPR: unchanged r10 list (RP_W2MIN in {22,24,25,99},
   prefetch gates), plus run gen_dense_prime's ubcap ldonly/ldonly256 trio
   before trusting any port floor on a new host.
4. Harness: reserve.sh needs the slurm PATH prefix from wallaby; tryout's
   remote chain-leg check.py still gets literal '$W/c.bin' (run map-checks
   by hand, as every round); --chain 1 is a silent no-op; gen_input.py
   needs env.sh; a80n0 reads this engine's 31 cell at 85.45 vs a81n2's
   84.8-85.2 -- same-node contrasts only.

## Round gen_r12

### Where this round started

r11 leaderboard (a80n0): **84.776 us/step** at the graded cell (L=31 B=16 m=140;
gen_race 84.549 rides this engine), leading the crossover (gen_dense_prime
111.1, gen_bluestein 274.3, MKL/FFTW 849-883).  The r11 verdict said the class
was converged: 31 port-model-saturated since r4, the large primes on a
"chunk-local L1 thrash" residue with "nothing cheap presenting itself".  The
r12 brief keeps all hands on the large-size regime; my r11 already struck the
Rader-at-L=100 cross entry (recorded, stands).  This round's move came
entirely from the cumulative read of the other entries' r11/r12 records, which
changed what "nothing cheap" meant.  Node: a80n0 (job 438881), ONE held lease
(slot 4, core 6), rotated interleaved arms; control = a binary built from
impl_11 source (the r9 symlink rule).

### Read first, and what it changed

- **gen_dense_prime r11's corrected machine currency** (512-bit L1 accesses
  pool at ~1.12/cyc loads+stores; 8-byte broadcasts are a SEPARATE 2/cyc
  class; no total-uop cap).  This is what killed the naive 4-wide design and
  forced the phase split (below): 4-wide with C+S together needs k-steps of 2,
  which DOUBLES per-column 512-bit stack loads on the pooled class.
- **gen_dense_prime r12's fusion gate** (cost candidates in CONCURRENT
  L1-resident bytes, adopt only under ~40 KB): my per-phase hot set is
  U-or-V (16 KB) + CS (16 KB) + a 2 KB table row-group = ~34 KB.  Designed to
  the gate before building.
- **gen_layout r12's two rules**: T1-not-T0 for DRAM-resident stream
  prefetch (raced here, see below), and "accumulator kernels are
  latency-tolerant -- stalls_l1d_miss is correlation, not causation" (the PMU
  attribution below confirms it from the winning side: my fills barely moved,
  the uops did).
- **gen_batchlane r12's protocol warning** (+-14% windows with sd<0.2% inside
  arms; only rotated interleaved minima over >=4 rounds discriminate <5%) --
  every verdict below follows it -- and their "code shape is a measured
  variable" lesson, covered here with a -DRP_NOW4 dispatch-off arm that
  catches layout confounds.

### What shipped: 4-WIDE dense chunks with an E/O PHASE SPLIT (rp_chunk4), RP_W4MIN=59

The r11 differential counters at 127 put the largest share of the 890 MB/step
L1 fill in the kernel-table walk (2h^2 doubles = 63.5 KB per column PAIR) and
the r8 pairing history says shared broadcasts are this kernel's lever.  The
4-wide form processes FOUR zmm columns (16 complex) per chunk, so the tables
are walked once per 16 columns -- half the 2-wide rate -- and per-column
broadcast uops halve again.  The design constraint from the corrected
currency: keeping C and S accumulators together at 4-wide forces k-steps of 2
(16 accumulators) and doubles per-column 512-bit stack loads.  Instead the
chunk splits into an E phase (cos correlation on U, k-quads of 4 x 4 columns
= 16 accumulators, C_k staged to a CS stack array) and an O phase (sin on V,
reload CS, combine, store rows k and p-k).  Per-column 512-bit load ratio
identical to the 2-wide; staging adds 2 x 4h L1-resident zmm accesses per
chunk (~2% of conv loads).  Every accumulator chain receives the same fmadds
in the same j order as rp_chunk2/rp_chunk, and the CS store/reload is
bit-preserving => outputs bit-identical.  In-place safe (all source loads in
the fold, before any store).  Dense engine only -- rp2/rp3 stacks already
exceed L1D at large m (r10 residue) and rp3 pairing is closed on both graded
architectures.  z-quads stay 2-wide (4-wide z staging would be 64 KB, the r9
lesson).  Dispatch: a 4-wide main loop in rp_pass ahead of the 2-wide loop;
tails fall through unchanged, so masked-tail coverage is identical.

### Measured on the node (a80n0 core 6, held lease, rotated interleaved arms, min us/step, chain B=1)

| p | ctl (impl_11) | gen_r12 | delta | rounds |
|---|---|---|---|---|
| 127 (m=4) | 35643-36416 | **31281-31990** | **-12.2%** | 4/4 non-overlapping |
| 107 (m=4) | 18183-18633 | **16478-16686** | **-9.6%** | 4/4 non-overlapping |
| 83 (m=8) | 6667-6861 | **6128-6474** | **-8.6%** | 3/3 |
| 71 (m=8) | 3803-3836 | **3542-3836** | -4..7% | 2/3 + one wash |
| 59 (m=8) | 1769-1782 | **1670-1676** | **-5.7%** | 3/3 |
| 47 (B=2 m=8) | 785-879 | 783-877 | WASH | overlapping |
| 23 (B=2 m=8) | 68.9-69.8 | 69.1-70.9 | WASH | overlapping |
| 113 (rp2, untouched) | 18902-19616 | 19219-19389 | wash | sanity |
| 31 B=16 (untouched) | 84.70-85.47 | 85.01-86.28 | wash | 5 rounds, overlapping; -DRP_NOW4 arm reads at ctl level |

Boundary set at RP_W4MIN=59 (wins at 59+, wash below; 19 stays 2-wide).
Graded cell via tryout: **85.11 us/step B=16** (MKL 868.5 same window), B=1
96.6-97.5 on a80n0 this window (both fresh-lease AND held-lease -- the same
a80n0 B=1 signature r11 recorded at 96.1-96.7; a81n2 reads this binary family
at 85.2-85.3; outputs bit-identical, so this is the node/window, not the code).

### PMU attribution (mandatory protocol; differential = samples 4 minus samples 2 = 8 steps, L=127)

| | ctl | gen_r12 |
|---|---|---|
| port_2_3 uops/step | 88.8M | **71.4M (-20%)** |
| l1d.replacement/step | 14.05M lines | 13.6M (-3%) |
| LLC-loads/step | ~1.28M lines (~82 MB, 97% miss L3) | unchanged |
| min us/step (counting mode) | 35675 | 31301 |

Reading: the win is BROADCAST-DISPATCH DELETION, not fill reduction -- the
table-walk fills were already latency-covered behind the 16 independent
accumulator chains, exactly gen_layout r12's accumulator doctrine, confirmed
here from the winning side.  The lesson pair: on these kernels locality
restructuring for its own sake would have lost (their measurement), but the
SAME structural change judged as uop deletion wins (mine).  Cost the uops,
not the fills.  The DRAM-level sweeps (~82 MB/step demand) are untouched, as
designed -- the custody chain stays on its 5-sweep floor (r7 closed form).

### What did NOT work, with the number that killed it

- **RP_PFT1 (T1 x-pass prefetch, gen_layout r12's DRAM-stream rule): LOSES at
  127 (+2..6%: 31961-33418 vs 31281-31990, 4/4) and wins hair-thin at 107
  (16359-16482 vs 16478-16686, 3/4, ~1%)**.  Mechanism for the non-transfer:
  their rule is for READ-ONCE streams; my in-place x-pass RE-STORES every
  line it loads, so the line must reach L1 anyway and T0's L1 placement is
  the store's RFO done early.  T0 stays default; knob kept -- the split
  verdict at 107 makes it a legitimate gen_race axis on CLX/SPR.
- 23/47: 4-wide is a wash (numbers above) -- broadcast dispatch is not
  binding where the whole table + working set is comfortably L1/L2-resident.
  Gate set to 59, not lower, per the "enable where it wins" doctrine.
- Nothing else was built and rejected this round; the r11 negatives (memcpy
  deletion, zero-copy re-home) were not re-litigated.

### Borrowed this round, named

- **gen_dense_prime gen_r11**: the corrected 512-bit access currency -- the
  phase split exists because of it; and gen_r12's concurrent-L1-bytes fusion
  gate, used at design time.
- **gen_layout gen_r12**: the T1-prefetch rule (raced; lost at 127, boundary
  contributed back: in-place RMW streams are not read-once streams) and the
  accumulator-latency-tolerance doctrine (confirmed; reframed as "cost the
  uops, not the fills").
- **gen_pfa_small gen_r11 / gen_layout gen_r12**: the differential counter
  method, again.
- **gen_batchlane gen_r12**: the rotated >=4-round protocol and the
  dispatch-off-arm defense against code-layout confounds.
- **gen_powp gen_r11**: the script-file-over-ssh session pattern (r12_ab.sh
  and helpers under build/tryout/gen_rader/r12/).
- **My own r8 pairing lineage**: rp_chunk4 is rp_chunk2's shape one level up,
  with the phase split as the new idea.

### Operation count (shipped)

FMA/adds identical to r6-r11 at every size (this round moved zero
arithmetic; 30-prime execute battery + chain batteries at 59/71/83/107/127
bit-identical to the impl_11 control, locally AND on the node, all arms).
Dense p >= 59: per 4-column chunk per k-quad per j, E phase 4 stack loads +
4 broadcasts + 16 FMA, O phase the same on V, plus 8h staging accesses per
chunk; tables walked once per 16 complex columns.  rp_chunk4 static size
5217 B (node build), broadcast-rotated rolled bodies, no spill pathology
(objdump audit both hosts).

### What I would do next

1. **The dense engine at 107/127 now runs ~31.3/16.5 ms with port_2_3 at
   ~71M/step and fills flat** -- the next uop class to delete is the stack
   U/V reload stream (h^2/2 loads per column-quad); an 8-wide phase-split is
   the mechanical next step but U+V+CS = 96 KB stack kills it at h=63.  A
   k-tile-of-8 E phase (32 accumulators is too many; 8 k x 2 col?) does not
   obviously close -- model against the 1.12/cyc pooled class first.
2. RP_PFT1 at 107 (+RP_W4MIN, RP_NOW4) belong to gen_race's CLX/SPR knob
   set; SPR note: wallaby raced none of this round's timing (known
   anti-pairing bias) -- the r9 calibration discipline applies before any
   SPR verdict on the 4-wide.
3. The **89 bimodality** (r8 item) remains the one open measurement,
   still luck-gated.
4. Harness: r12_ab.sh / r12_small.sh / r12_l31.sh / r12_b1.sh under
   build/tryout/gen_rader/r12/ hold the whole session (build/cmp/pmu/
   race127/race107/race59/sanity/gates phases); in/c pairs there now cover
   23b2/47b2/59b1/71b1/83b1/107b1/113b1/127b1/31b16/31b1.  tryout's remote
   map-check still gets the literal '$W/c.bin' (gates run by hand, as every
   round).  a80n0 B=1 at 31 reads ~97 this window on ANY core (fresh or
   held) -- same-node contrasts only.

### Measured summary (the reply line)

L=31 B=16 m=140: **85.11 us/step** (tryout, MKL 868.5 same window; bit-identical
lineage since r4), B=1 96.6 on a80n0 this window (a81n2 form: ~85.3).  Single
rel_l2 4.059e-16, two-step 1.784e-15, map-chain m=140 2.559e-14 -- identical
digits to r4-r11.  Class duty, node same-window rotated races: **127: -12.2%
(31.3 ms), 107: -9.6% (16.5 ms), 83: -8.6%, 59: -5.7%, 71: -4..7%** via the
4-wide E/O phase-split dense chunks; gates at 107/127 PASS (9.3e-16/4.9e-15,
8.7e-16/4.9e-15).

## Round gen_r13

### Where this round started

r12 leaderboard: **84.753 us/step** at the graded cell (L=31 B=16 m=140), leading
the crossover (gen_dense_prime 111.3, gen_bluestein 274.0, MKL/FFTW 849-883).
The round-13 brief is the benchFFT B=1 small-L round: two new scored cells
(10:1, 12:1) that my class declines (composite), and "everyone else: protect
your cells."  This round's control reads: 31 B=16 **85.56 / 84.76 / 85.32**
(quiet windows, board-consistent), all gates identical digits.  The cell is
protected; the round's work came from asking whether the benchFFT finding has
an image INSIDE this class.

### The probe: the class has the same B=1 small-L hole, at the tiny primes

B=1 execute on the node (leased core, MKL same core; libraries from the r12
build/a80n0 binaries):

| p | gen_rader (r12) | MKL | fftw3_measure | verdict |
|---|---|---|---|---|
| 5 | 0.669 | 0.337 | — | LOSE 2.0x |
| 7 | 1.540 | 0.751 | — | LOSE 2.05x |
| 11 | 5.257 | 2.970 | — | LOSE 1.77x |
| 13 | 8.393 | 6.086 | 7.323 | LOSE 1.38x (dense_prime 11.29, trunk gen_race 7.94 — the LIBRARY lost this cell to MKL 1.3x) |
| 17 / 19 / 23 | 19.4 / 32.8 / 67.9 | 82.7 / 122 / 224 | — | win 3.4-6.4x |

The boundary is razor sharp: everything from 17 up crushes the libraries,
everything at 13 and below loses.  13 is the only prime where MKL has a tuned
small radix AND we were slow.

### The diagnosis: at tiny p the runtime-table engines are all fixed cost

Skip-knob timing (compile-time RP_SKIP{Z,X,Y} arms; wrong output, timing only)
at 13 B=1, full = 8.64 that window: z ~3.3 us, x ~2.2, y ~2.3 — every pass
~3x its uop model (~130 uops/chunk at m=3 should be ~50 cyc; measured ~148).
At m=3 the chunk does almost no arithmetic, so what dominates is: the
switch-dispatch + call per 4 columns, the plan-struct and kernel-table pointer
reloads per call, the runtime js/jv/kp/km index loads, and the j*rs address
arithmetic — none of which can hoist out of the column loop across a call
boundary.  First attempt at the skip arms was caught by create()'s self-check
(the deliberately-wrong output fell back to the dense reference path and read
177 us): the gate cannot be probed through, which is exactly what it is for —
the timing arms explicitly bypass it and say so.

### What shipped: compile-time engines for p <= 13, execute path only

The r31 lesson in miniature ("instantiated with compile-time strides", r1):

1. **p=13 (rp13_\*)**: the rp2 m=3 arithmetic, identical op order, with the
   p=13 Rader index tables HARDCODED (js/jv/kp/km derived from rp2_build's
   construction offline; create() memcmp-verifies them against the live
   rp2_build output and uses the generic path on mismatch, so a transcription
   bug cannot ship even before the self-check).  Kernel doubles stay in
   pl->k2 — broadcast cost is address-independent, nothing to transcribe.
   Chunks always_inline; three pass shapes instantiated at literal rs (z 8,
   x 338, y 26) with literal masks and trip counts; z quads with compile-time
   tiling (nt=4, tail mask 0x03).
2. **p=3/5/7/11 (rpd_\*, dense)**: rp_chunk was ALREADY always_inline with
   p/h as parameters — RPD_DEFINE instantiates volume drivers that call it
   with literal constants, and gcc constant-folds every loop bound, row
   offset and mask for free.  No new kernel, no new tables.
3. Dispatch in fast_volume ahead of rp_volume, gated on use13 (13) /
   !m2 && !m3 (tiny dense); RP_NO13 / RP_NOD13 restore the r12 paths.

### Measured on the node (a80n0 leased cores, tryout; MKL same core same window)

| p | r12 | gen_r13 | delta | vs MKL |
|---|---|---|---|---|
| 3 | — | **0.085** | — | 0.093 -> ahead |
| 5 | 0.669 | **0.313** | **-53%** | 0.336 -> ahead |
| 7 | 1.540 | **0.663** | **-57%** | 0.751 -> ahead |
| 11 | 5.257 | **3.139** | **-40%** | 3.009 -> **MKL-4%, the one remaining loss** |
| 13 | 8.393 | **4.20** | **-50%** | 6.06-6.41 -> **1.44-1.52x ahead**; fftw3_measure 7.32 -> 1.74x |
| 13 B=16 | — | 4.33/vol | — | batched inherits it (volume-major) |

Graded cell PROTECTED: 31 B=16 **84.76 / 85.32** us/step this round's windows
(board 84.753), B=1 97.5 (the documented a80n0 B=1 window signature).  Chain
outputs at 31 **bit-identical to the impl_12 binary** (cmp on the node, exec
AND chain m=140), 37 exec bit-identical; 37/61 execute same digits as always.
Gates by hand on the node: 31 B=16 single 4.059e-16, map-chain m=140 2.559e-14
(anchor 2.312e-14), **two-step 1.784e-15** — identical digits to r4-r12; 13
B=1 two-step **1.025e-15**, map-chain m=100 1.031e-14 (anchor 7.861e-15).
SPR (wallaby) correctness PASS at 3/5/7/11/13 B=2 (timing not raced there —
the r9 anti-pairing calibration discipline stands).  Setup at 13: 0.001 s.

### What did NOT work, with the number that killed it

- **Paired-interleaved z-quads at 11 (two 4-pencil quads staged before either
  chunk, chunks back to back, un-stagings last): 3.268 vs 3.139 (+4%).**
  Built to test a 512-bit store->forward-stall theory for the z residue (z
  ~1.1 us at 11 vs ~0.6 model; the staging stores are consumed immediately).
  The theory is DEAD: distance did not pay, the doubled staging footprint
  cost more.  Reverted; the z residue at tiny p is still unattributed.
- The first skip-knob build measured nothing twice: (a) the self-check
  correctly benched the wrong-output build to the reference path; (b) after
  bypassing that, the knobs sat in rp_volume while p=11 had already moved to
  rpd_volume_11 — both runs read identical.  Both are recorded because both
  are the verification discipline working as designed: a timing arm must
  prove it measures the path it thinks it measures.

### Borrowed this round, named

- **The r13 brief's material #3** ("FFTW's small-codelet quality is pure
  schedule, not memory") — the licence to treat tiny-p B=1 as a fixed-cost
  problem, and the round's framing that exposed the class's own hole.
- **My own r1/r31 lineage**: compile-time indices/strides as THE small-size
  lever; the create()-verify-hardcoded-tables gate is the r5 rp3 recipe
  turned around.
- **gen_dense_prime r1 (transitively)**: rp_chunk itself — the dense tiny-p
  win is entirely their kernel under constant folding.
- Probed rivals at 13 B=1 before building (dense_prime 11.29, trunk 7.94):
  the class entry is now the library's best 13 and the trunk inherits it.

### Operation count

Tiny primes: arithmetic op ORDER identical to the generic engines (rp13 =
rp2_chunk_3's sequence with folded addressing; rpd = rp_chunk itself), so
per-chunk FMA counts are unchanged — the round deleted only dispatch, call,
table-pointer and index-load overhead plus runtime address arithmetic.
p >= 17, 31, chains everywhere: untouched instruction paths (objdump-level
changes are rodata shifts from the description string), outputs bit-identical.

### What I would do next

1. **11 is the last losing size (MKL-4%)**: the z-pass (~1.1 of 3.14 us) is
   the target — a dense z-row GEMM (r31_zrow_pair shape at compile-time p=11:
   duplicated-pair trig rows, 2 zmm accumulators, ~3 shuffles/pencil vs 12)
   deletes the transpose port-5 bill; needs a small rp_trig_dup(p) table.
   Sketched, not built — the pair-interleave negative says measure, not guess.
2. 17/19/23 compile-time instantiation is nearly free (17 needs an rp17 table
   set like rp13; 19/23 are RPD_DEFINE one-liners) — unpressed (we lead 3-6x)
   but the benchFFT curve would show it.
3. The CHAIN at tiny primes still runs the generic machinery; if a tiny-prime
   chain cell ever gets scored, rp13-style passes with the map fused into the
   volume scheme (NOT into the stores — five negatives stand) are the shape.
4. Harness: skip-knob timing arms must live in the SAME volume function the
   dispatch actually reaches (this round's own trap); RP_SKIP* + the
   self-check bypass are compile-time, default-off, wrong-output — never race
   them into a shipped binary.

### Measured summary (the reply line)

Compile-time tiny-prime execute engines.  L=31 B=16 m=140: **84.76-85.32
us/step** (bit-identical lineage since r4), B=1 97.5 (a80n0 window signature).
Round headline, B=1 execute: **13: 8.39 -> 4.20 us (1.44-1.52x over MKL, 1.74x
over fftw3_measure), 11: -40%, 7: -57%, 5: -53%** — every prime in class now
beats every library at B=1 except 11 (MKL-4%).  Single rel_l2 4.059e-16 (31) /
3.114e-16 (13), two-step 1.784e-15 (31) / 1.025e-15 (13), map-chain m=140
2.559e-14 — all gates PASS, all identical digits where paths are shared.

## Round gen_r14

### Where this round started

r13 leaderboard: **86.176 us/step** at the graded cell (L=31 B=16 m=140), leading
the crossover (gen_dense_prime 111.7, gen_bluestein 276.7, MKL/FFTW 849-885).
The r14 brief: benchFFT (single-shot fft3d_execute) is the B=1 metric across
L=10..128; route execute() through the fast engine the chain uses; beat fftw3 at
every B=1 size.  Of the 14 acceptance sizes only 31 is mine, but the seam has a
class-wide image: execute() at 31 has run the r1 FLAT layout since r1 (the exact
4K-alias/line-split trap the r2 padded arena fixed for the CHAIN), and the r13
record left "11 at MKL-4%" as the class's one remaining B=1 library loss.
Node: a80n0 (job 439820), ONE held lease (slot 6, core 8), interleaved arms.

### The baseline probe (execute B=1, node core 8, min over 3 reps; MKL/fftw3 same core)

| p | gen_rader (r13) | fftw3_measure | mkl | verdict |
|---|---|---|---|---|
| 31 | **75.96** | 762.7 | 754.5 | lead 9.9x |
| 11 | 2.764 | 4.341 | **2.613** | fftw beaten 1.57x; **MKL-5%, the class's one loss** |
| 13 | 3.722 | 6.891 | 5.340 | lead 1.43x (r13 form holds) |
| 37 | 262.8 | 1570 | 1594 | lead 6.0x |

So the mission's own acceptance (beat fftw3 everywhere) was already met
class-wide before the round; the round's real work was the two items above.

### Built, raced, REJECTED: execute()-onto-the-padded-arena at 31 (-DR31_PADEXEC)

The round's named seam, in this class's image: r31_exec_pad = z reads flat src
straight into the r2 arena (the chain's step-0 form: 7 zquad_fp + 3-row dense
tail per plane), x tail-free at the anti-alias pitch (r31_pass_xp), and a NEW
final y pass that LOADS padded rows and STORES the flat output plane directly
(r31_pass_yf -- no copy-out sweep; needed a dest-row-stride parameter threaded
through r31_chunk/r31_pass_core, all call sites compile-time, pre-r14 sites
pass ds == rs).  Node race, interleaved, 4 rounds:

| arm | B=1 | B=16 |
|---|---|---|
| flat (r13 path) | **76.3 / 76.3-80.0** | **92.8-93.8** (one 104 warmup) |
| padded reroute | 90.8-104.3 | 106.0-107.6 |

**+15-20% at B=1, +14% at B=16, 4/4 non-overlapping.  REJECTED.**  Mechanism,
so nobody re-derives it: the r2 arena win was measured on a chain whose THREE
passes all ran in place on one aliased buffer.  Execute's flat z is already
OUT-OF-PLACE (src -> dst: no store->load aliasing anywhere in the z pass), its
y runs at small in-plane strides, so only the in-place x-pass pays the alias
tax (~few % of 76 us) -- while the arena route adds a whole third buffer (st,
574 KB), pushing the per-call working set src+st+dst = 1.5 MB past the 1.25 MB
L2.  Execute IS m=1: this is the r5 arena lesson and the r11 zero-copy lesson
met a THIRD time, now from the execute() side.  The boundary triangulated from
all three: padded/staged state pays only when its fill/drain amortizes over
m >> 1 in-place steps AND the flat alternative is genuinely in-place-aliased.
Flat execute measures 76 us against a ~73 us three-pass port model -- there was
never 10% on this table.  The ds refactor ships (harmless: chain outputs
cmp-IDENTICAL to impl_13 at 31 B=16 m=140, 11/13/37/61 m=4); the engine stays
opt-in for the xarch race.

### Shipped: DENSE Z-ROWS at p=11 (rpd11_*) -- the r13 next-step-1 item, the last library loss closed to -1.5%

The r13 skip-knob profile put the 11 z-pass at ~1.1 of 3.14 us against a ~0.6
model; the transpose-quad z pays 12 tp4 shuffles (48 port-5 ops) + a staged
stack round trip per 4 pencils before rp_chunk starts.  The dense form is
r31_zrow's shape at compile-time p=11:

- fold in 512-bit: u_{1..4}/v_{1..4} = one zmm add/sub of x_{1..4} against a
  lane-reversed mirror load of x_{7..10} (ONE shuffle), u_5/v_5 one xmm pair;
- half-spectrum accumulates in one zmm (k=0..3, duplicated pairs, k=0 included
  so X_0 rides lane pair 0) + one ymm (k=4..5) per C/S system -- the ymm half
  dispatches on port 1, which idles in every kernel on this panel (PMU audit
  avenue 4, first productive use in this entry);
- combine: X_k = C -+ iS via permute+SG (the R31_ZSTORE identities), mirror
  half stored lane-reversed (shuffle 0x1B zmm masked 0x3F, permute2f128 ymm);
- 4 rows per call share every table load; duplicated-pair trig (5 rows x 16
  doubles, 640 B/table) built at create() into ctd/std_ (31-only fields,
  free at generic p).  ~1 port-5 shuffle/row vs the quad form's ~12.

Node race (interleaved, 4 rounds, execute B=1): **quad 2.780-2.791 (one 3.18
warmup) vs dense 2.663-2.674 (-4.2%, 4/4); MKL 2.626-2.641 same core**.  The
gap closes from -5% to **-1.5%**; fftw3_measure beaten 1.63x.  -DRPD11_QZ
restores the quad z.  Execute outputs at 11 differ from r13 bitwise (different
kernel form, both correct: rel_l2 2.547e-16); every OTHER size's execute and
EVERY chain is cmp-IDENTICAL to the impl_13 control (12-prime battery:
3/5/7/11/13/17/23/37/43/61/89/113 exec + 11/13/37/61 chain m=4 + 31 B=16
m=140).

### Gates (final binary)

31: exec B=1/B=16 bit-identical to impl_13 (single rel_l2 4.055e-16 B=1);
chain m=140 B=16 bit-identical => board digits inherited (two-step 1.784e-15,
map-chain 2.559e-14, anchor 2.312e-14).  11: single 2.547e-16 (tol 1e-12),
**two-step 8.527e-16** (tol 3e-14, 35x margin), chain m=4 bit-identical to
r13.  The create() self-check gates the new z path (execute + one chain step
vs dense reference at 1e-13) exactly as every engine since r1.

### What did NOT work, with the number that killed it

- **The padded execute() reroute: +15-20% B=1 / +14% B=16 at 31, 4/4** (the
  round's headline negative, mechanism above).  Note what it means for the
  panel: the r14 brief's "route execute() through the chain's fast engine"
  is the right systemic fix ONLY where the chain's engine does not carry
  private staged state whose fill/drain is a per-call fixed cost.  For
  in-place-capable flat engines like this one, execute() was never on a slow
  path -- measure before rerouting.
- A skip-knob pass attribution of the remaining 2.66 us at 11 was staged
  (bin_skZ/X/Y built) but the node's sshd dropped mid-session; the arms are
  in r14/prof11.sh for whoever wants the decomposition.  The remaining MKL
  gap is 37 ns/transform on a 1331-point problem -- fixed-cost territory
  (dispatch, tails, driver loop), not pass arithmetic.

### Borrowed this round, named

- **gen_pfa_small gen_r13** (via the r14 brief): the execute()/chain() seam
  framing that set the round's agenda -- raced here and found NOT to apply to
  this class's flat execute (boundary contributed back above).
- **My own r13 record**: the sketched dense-z design at 11, executed as
  specified (including the warning to measure, not guess, after the
  pair-interleave negative).
- **gen_dense_prime gen_r1 (transitively ice)**: the R31_ZSTORE combine
  identities, reused verbatim at p=11 width.
- **PMU audit avenue 4 / gen_layout r9 gl_map4 line**: the port-1 ymm
  co-issue idea, first landed in this entry as the k=4..5 ymm half.
- **gen_batchlane gen_r4**: the one-lease same-core interleave protocol, as
  every round.

### Operation count (shipped delta)

Only p=11's z pass changed: per 4 pencils, OLD = 12 tp4 shuffles (48 p5) + 8
masked loads + 24 staging stores/loads + rp_chunk(4 cols: ~50 FMA + fold);
NEW = 4x(2 zmm loads + 1 shuffle + 2 zmm add/sub + 1 xmm pair + x0 bcast) fold
+ 5x(4 table loads shared + 8 pair-broadcasts + 8 zmm FMA + 8 ymm FMA) + 4x(2
shuffles + 4 FMA-class combine + 4 stores).  Port 5 per row: ~12 -> ~1.25.
Everything else: instruction paths untouched (objdump-level deltas are the
description string and the dormant R31_PADEXEC/ds plumbing; chains cmp-prove
it).

### What I would do next

1. **The 11 fixed-cost residue (MKL-1.5% = 37 ns)**: run the staged
   r14/prof11.sh attribution; if z is now near its ~0.45 us model, the money
   is in the x/y rp_chunk tails (the 1-column masked tail runs the full
   11-row fold for 1 column) -- an overlapped 4-wide tail is impossible
   in-place (r1 lesson) but the X pass could go out-of-place into t1 at 11
   (working set 2x21 KB, trivially L1) to unlock it.
2. **p=7/5 dense z** is the same kernel smaller (7: half-spectrum = exactly
   one zmm, no ymm half) -- we already beat MKL there, so it is lead-padding
   for the benchFFT curve only; do it if a round ever scores tiny primes.
3. The 31 execute() is closed: AT its three-pass model, 9.9x over the best
   library, and the padded route is measured dead.  Only deleted work moves
   it (r4 law), and the passes are the settled minimum.
4. **89 bimodality** (r8 item): still the one open measurement, still
   luck-gated.
5. Harness: a80n0's sshd dropped mid-session this round (reservation job
   kept running; heartbeat-shim squeue still reported R) -- stage decision
   scripts as files (the r9 pattern) so a dropped control connection costs
   nothing; prof11.sh is exactly that.

### Measured summary (the reply line)

Dense z-rows at p=11; padded execute() reroute at 31 raced and rejected.
L=31: execute B=1 **75.96 us** (fftw3 762.7, MKL 754.5 -- 9.9x), B=16 chain
**bit-identical board form** (86.2 us/step r13 board; gates 4.055e-16 /
1.784e-15 / 2.559e-14).  L=11 B=1: **2.797 -> 2.663 us** (-4.2% vs r13's quad
z; fftw3 4.34 = 1.63x, MKL 2.63 = -1.5%, was -5%); single 2.547e-16, two-step
8.527e-16.  All other sizes: execute AND chain outputs cmp-identical to the
r13 binary.

### Post-session harness addendum (the a80n0 sshd/pam event, for the monitor and the panel)

Mid-session a80n0's sshd dropped and came back REFUSING logins with
"Access denied by pam_slurm_adopt: you have no active jobs on this node" --
while the reservation job's payload KEPT RUNNING (RESERVATION.heartbeat
node-written and <60 s fresh throughout, so reserve.sh --status and the
wallaby squeue shim both correctly report ALIVE).  Reading: slurmd on the
node restarted and orphaned the job step; the payload process survives and
heartbeats over NFS, but slurm no longer credits the job to the node, so
pam_slurm_adopt denies every new connection.  Consequence for everyone: a
FRESH HEARTBEAT NO LONGER PROVES SSH ACCESS -- the two failure modes are now
distinguishable (stale heartbeat = job dead; fresh heartbeat + pam denial =
slurmd lost the job).  tryout.sh will pass its reservation gate and then die
at the ssh step.  The monitor needs to re-reserve (scancel the orphan first;
its payload is still burning a node slot).  My slot-6 lease is released; the
staged-but-unrun pass attribution at 11 lives in r14/prof11.sh.
