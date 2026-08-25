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
