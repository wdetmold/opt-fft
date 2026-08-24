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
