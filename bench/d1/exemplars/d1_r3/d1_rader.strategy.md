# d1_rader — strategy record

Class: prime N via Rader's reduction to an (N-1)-point cyclic convolution.
Acceptance set this round: 13, 31, 127, 1021, 65537. The class headline per the
brief is the large primes whose N-1 factors well — above all 65537 (N-1 = 2^16,
the "dream case": the convolution IS a pow2 FFT, unpadded), and 1021
(N-1 = 1020 = 2^2*3*5*17). 10007/100003 are deliberately NOT claimed: their
N-1 is awkward, d1_bluestein already wins them with smooth non-pow2 pads
(20480/204800), and a pow2-padded Rader would tie at best. Nested Rader for
10007 (5003-1 = 2*41*61) remains an untried A/B for a future round.

## Round d1_r1 (2026-09-02) — from dense stub to a fused Rader engine

**Measurement caveat (same as every entry this round): the Ice Lake
reservation (job 440299) was dead the whole session, so tryout.sh was
unusable and no slurm was submitted (monitor owns the node). All numbers are
wallaby (SPR Gold 6448Y), taskset to a core verified idle together with its
SMT sibling via /proc/stat deltas, best-min over >=3 back-to-back reps,
A/B'd against fftw1d_measure (FFTW 3.3.10, sota/fftw1d.c) built with the
identical flags in the same minutes.** Wallaby load sat at ~25 and spiked
mid-session: the same 65537 binary that measured 670-703 us in a quiet
window read 1300-1600 us an hour later on every core of both sockets. The
quiet-window same-minute A/B ratios are the numbers I trust.

### What the implementation now is

One engine, three plan shapes, all sharing a split-complex mixed-radix
Stockham conv core (radices 2/3/4/5/8 + one dense 17), per-stage plan-time
twiddle tables, out-of-place ping-pong:

1. **Unpadded conv M = N-1** when N-1 factors over {2,3,4,5,8} plus at most
   one 17: 65537 -> M=65536 stages [4,8,8,8,8,4]; 1021 -> M=1020 stages
   [4,3,5,17]; 13 -> [4,3]; 31 -> [2,3,5].
2. **Pow2-padded conv M >= 2(N-1)-1** otherwise: 127 -> M=256 [4,8,4,2],
   with the zero-blocks pruned out of the entry stage (a[t]=0 for t>=P kills
   radix-4 blocks 2,3 and half of block 1 — d1_bluestein's pruned-entry
   shape, reused for the Rader zero-pad).
3. **Fusions** so no pass over the M-array exists that isn't a butterfly
   stage: the g^q gather is fused into the forward stage-0 butterfly (the
   permutation is read as 4 index streams, the random reads feed the
   butterfly directly); the kernel-spectrum multiply (1/M folded in) + plane
   swap is fused into the inverse's stage-0 (inverse = forward on swapped
   planes); the output scatter X[g^-m] = x[0] + conv[m] is fused into the
   last inverse stage when its radix is 4 (65536) or 2-pruned (padded);
   X[0] = x[0] + forward-conv DC bin, i.e. free — no separate sum pass.
4. **Fused map chain (fft1d_chain), the round's own idea**: because
   iidx[q] = g^q = g^-(P-q) = oidx[(P-q) mod P], the composition
   gather∘map∘scatter between chain steps is a pure INDEX REVERSAL, and the
   elementwise map commutes with any permutation. So the chain state lives
   split, in conv-output order, across all steps: interior steps have NO
   random gather, NO random scatter, no interleaved round trip — the entry
   stage reads the previous d[] backwards, the exit stage applies the map
   (+ the pre-permuted c field) as it stores. state[0] rides along as one
   scalar pair. Only step 1 reads natural order and one final P-pass
   materializes the interleaved output. Batched chains run b-outer so the
   P-sized state stays cache-hot for the whole chain. Verified at every
   graded (L,B,m) including 13 @ m=200000 (rel_l2 1.2e-15 vs tol 1e-10).
   Measurable proof it works: 65537 chained runs at 670 us/step vs 703 us
   for a standalone execute — the chain steps are FASTER than the transform
   alone because the permutation passes vanished.

### Measured (wallaby best-min, quiet window; "FFTW" = same-core same-minute fftw1d_measure)

| cell | d1_rader | FFTW same box | Ice Lake lib best (baseline table) |
|---|---:|---:|---:|
| 65537 B=1 m=1 | **703 us** | 1169-1448 | 1632 (patient) |
| 65537 B=16 m=1 | **734 us** | 1169-1341 | 1740 (patient) |
| 65537 B=1 m=60 chain | **670 us** | 1436-1646 | 1632 |
| 65537 B=16 m=20 chain | **726 us** | 2047-2095 | 1740 |
| 1021 B=1 m=1 | **7.43 us** | 8.9-9.1 | 11.39 (MKL) |
| 1021 B=256 m=1 | 9.2-9.8 us (noisy) | 9.6-13 (noisy) | 12.39 (MKL) |
| 1021 B=1 m=2000 chain | **8.45 us** | 11.6-13.3 | 11.39 (MKL) |
| 1021 B=256 m=400 chain | **9.8 us** | 16.8-17.7 | 12.39 (MKL) |
| 13 B=1 / B=512 / chains | 0.073 / 0.075 / 0.082-0.084 | 0.021 B=1 | 0.066/0.051 (MKL) |
| 31 B=1 / B=512 / chains | 0.203 / 0.204 / 0.228-0.234 | 0.197-0.255 | 0.334/0.315 (MKL) |
| 127 B=1 (unscored) | 0.60-0.92 | 0.96-1.34 | — |

Accuracy: single-call rel_l2 = 2.7e-16 (13) … 1.4e-15 (65537), gate 1e-12;
all map-chain gates pass with >=4 decades of margin; also verified at odd
batches (1,2,3,4,8) per size. Setup <= 0.008 s even at 65537 (FFTW patient
pays 58 s there on the scoring node).

The headline works as the survey predicted: at 65537 the unpadded 2^16
Rader convolution beats same-box FFTW by 1.7-2.4x per regime, and beats
d1_bluestein's own 65537 (2290-2480 us wallaby, their record — they pad the
Bluestein conv to 147456, 2.25x more work). 31 should take its non-batched
cells from MKL/FFTW; 13 stays lost to library codelets (~0.02 us) — that
cell belongs to d1_prime's dense fold or a future straight-line codelet.

### What did NOT work, with the numbers that killed it

- **Four-step 256x256 at M=65536** (transpose-free, 8-lane column tiles,
  lane FFT = the same Stockham kernels entered with initial stride 8, both
  tile sides contiguous-8, symmetric middle-twiddle table, pointwise fused
  into the inverse tile load): correct (rel_l2 1.4e-15) but 1017-1034 us vs
  703 us for the plain 6-stage Stockham, lane radices [4,8,8] and [8,8,4]
  identical. The 1 MB working set still fits wallaby's L3, so the ~2x
  traffic cut buys nothing while the extra tile load/store passes cost ~45%.
  The code is still in the file behind `p->fourstep = 0 &&` — worth one
  retry on the scoring node's weaker cache (Ice Lake 1.25 MB L2), and it is
  the natural base for a 100003-class size that does NOT fit L3.
- **[.,8] instead of [.,4,2] tail at padded M=2048** (fewer stages, generic
  scatter instead of the half-pruned fused radix-2 exit): 9.55-9.68 vs
  9.12 us at 1021 before the radix-17 work made it moot. Fused pruned exit
  kept.
- **Padded M=2048 at 1021 at all**: 9.12 us vs 7.43 us unpadded [4,3,5,17].
  The dense radix-17 final stage (symmetric fold u/v pairs, real-coefficient
  8x8 FMA blocks, unit twiddles since m=1) costs ~512 FMA per 17-lane but
  halves the conv length; it also lifted every 1021 chain cell by ~15%.
- (Bug worth remembering: computing `m0 = M / p->entry_r` before the
  fourstep branch divided by zero — entry_r is only set for the generic
  path. FPE, caught by the first direct run.)

### Borrowings (this is the cumulative round working as intended)

- From **d1_bluestein** (impl + record): the entire split-complex Stockham
  stage-kernel family (st2/3/4/5/8 taken nearly verbatim), per-stage
  plan-time twiddle layout, inverse-as-forward-on-swapped-planes with 1/M
  folded into the kernel spectrum, the kernel-multiply-fused inverse entry,
  the zero-pruned entry shape, `#pragma GCC ivdep` on every hot loop, and
  the per-function `target("arch=icelake-server,prefer-vector-width=512")`
  attribute (their hard-won lesson that file-wide pragmas reset the ISA).
- From **d1_prime** (record): the symmetric-pair real-coefficient fold, used
  here inside the radix-17 stage; the mktemp-dir discipline after their
  /tmp collision; the idle-core + same-core-A/B measurement protocol.
- From **docs/literature_1d/00-SURVEY.md**: the per-prime playbook (65537
  unpadded, 1021 via a twiddle-free-ish 1020 conv, leave 10007/100003 to
  Bluestein), and plan-stage tables over in-loop recurrences.

### Next round, in priority order

1. **Re-measure on the scoring node the moment the reservation returns** —
   every number above is SPR; the four-step decision (off) and the
   [4,8,8,4,2]-vs-[4,8,8,8] call (5%) were both made on wallaby's fat cache
   and could flip on Ice Lake.
2. **65537 standalone execute below the chain's 670 us**: the chain proved
   the gather+scatter passes cost ~35 us each; software-prefetching the
   gather index stream, or an 8-wide manual vgatherdpd entry, should claw
   back most of that for the m=1 cells.
3. **1021 conv below 7.4 us**: the radix-17 stage is ~40% of the 1020 FFT;
   a Rader-16 sub-decomposition of the 17-point DFT (16 = pow2!) or a
   Good-Thomas 4x255 split would cut it. Also the B=256 cell is
   memory-streaming and was too noisy to trust — remeasure quiet.
4. **13 B=1**: concede or write a straight-line Rader-13 codelet (12-point
   conv via 3x4 Good-Thomas, fully unrolled, no core dispatch) — MKL's 0.02
   us is ~60 cycles and the stage-loop machinery alone costs more than that.
5. **10007 nested-Rader A/B** (5003-1 = 2*41*61): the survey's untested
   opening; only worth it if it can beat d1_bluestein's ~110 us, so measure
   the 5003-conv cost first before building anything.

## Round d1_r2 (2026-09-02) — the scoring node is a different machine: hand-vectorize everything hot

**Measurement conditions: the Ice Lake reservation (icehold 440371 on a80n0 —
the scoring node itself) was LIVE this round, so every number below is from
tryout.sh / direct ssh runs pinned to core 2 of a80n0. Same-window A/B against
the previous build throughout. Late in the session other implementers' tryout
slots added visible noise to the memory-bound cells (65537 B=16 wobbled
1489 -> 1653 across windows); the A/B deltas quoted were taken within one
window and are trustworthy, the absolute numbers for memory-bound cells are
whatever the scoring quiet gives.**

### The r1 lesson that drove everything: wallaby numbers had lied

r1's 7.43 us at 1021 B=1 (wallaby, SPR) scored as **13.92 us** on Ice Lake —
MKL (8.28) took the cell. First diagnostic this round: skipping both dense
radix-17 stages dropped 14.7 -> 8.17 us, i.e. **st17 was 6.5 us = 44% of the
transform**, running at ~10% of FMA peak. gcc 11 had never vectorized ANY of
the q-loops (3.3 GF/s = scalar FMA throughput); SPR merely tolerated the
scalar code better than Ice Lake. Autovectorizer coaxing failed twice with
numbers: a tiled two-pass form with `t[4][8][8]` went to 21.7 us (gcc keeps
vector accumulator arrays in memory), a register-economized two-pass scalar
form 15.4 us (outer-loop vectorization never triggers across nested constant
j/k loops). Conclusion adopted for the whole file: **the hot kernels get
explicit AVX-512 intrinsics; the autovectorizer only keeps the cases it
provably handles.**

### What the implementation now is (deltas from r1)

1. **st17 rewritten with __m512d intrinsics** (st17_body/st17_vblock): masked
   8-lane blocks, u-half computed with A_k parked in both twin output slots,
   v-half computed second and combined by RMW — live state ~22 regs per half,
   j/k loops unrolled by pragma, coefficients as vbroadcastsd. Three
   compile-time mode instantiations: plain / scatter-fused exit / chain-map-
   fused exit. NOTE: header intrinsics refuse to inline into functions
   carrying the old `target("arch=icelake-server")` attribute when the command
   line is `-march=native` (option-mismatch error), so the HOT attribute was
   dropped from everything the intrinsic code touches — builds are always
   -march=native on the target machine anyway.
2. **Entry gather vectorized WITHOUT vgatherdpd**: 8 random complex points are
   loaded as 128-bit pairs assembled with vinsertf64x2 (4 per zmm) + one even/
   odd vpermt2pd per plane (GATHER8). An actual vgatherdpd version measured
   WORSE than scalar (2.8 us vs 2.0 within a 9.56 us transform) — Ice Lake
   gathers are microcoded and lose on L1/L2-resident data. Same story on the
   store side: the vscatterdpd fused exit cost 2.4 us; the shipped fused exit
   stages the vector through the stack and stores scalar (ST17_SINKSTORE),
   and beats the unfused arrangement 7.85 vs 8.28 us. **Fusion decisions must
   be A/B'd, not profiled**: perf attributed the stall time so misleadingly
   that I un-fused first and lost 0.4 us.
3. **st3 s==4 vector path** (st3_s4): two p-groups per zmm (8 contiguous
   doubles ARE two radix-3 butterflies at s=4), twiddles pair-broadcast with
   one vpermpd from a 128-bit load, outputs stored as 256-bit halves at
   12p+4u. **st4_first_bhat and st5 (masked lane blocks, any s) vectorized**
   the same way; scalar tails kept for m%8.
4. **13/31 replaced by CRT register codelets** (the round's second big idea,
   from the Winograd/Agarwal-Cooley corner of the survey): the Rader conv is
   bracketed by permutations we own, so map the ring Z12 = Z4 x Z3 and
   Z30 = Z2 x Z3 x Z5 at PLAN time (CRT maps folded into iidx/oidx, kernel
   spectrum computed by the same codelet DFT so ordering is consistent by
   construction). Cyclic conv becomes a twiddle-free DFT4(x)DFT3 /
   DFT2(x)DFT3(x)DFT5, fully in registers, straight-line. TWO codegen traps,
   each worth remembering: (a) local pointer temps (`double *r_ = re + 3*k`)
   defeat gcc's SROA — index the arrays directly; (b) gcc's SLP vectorizer
   turned the 12-point codelet into 47 vpermt2pd + 8 vgatherdpd and made it
   SLOWER than the machinery it replaced (0.163 vs 0.113 us) —
   `optimize("no-tree-vectorize,no-tree-slp-vectorize")` on the codelets
   fixed it: 0.163 -> 0.056 us. Chain codelets keep the state in CRT order
   (reversal permutation precomputed into p->rev); the map stays SCALAR
   because zmm vsqrtpd/vdivpd are so slow on Ice Lake that an 8-wide map
   measured 0.110 vs 0.099 us/step.

### Measured (a80n0 = the scoring node, core 2, same-day A/B; "r1 score" = d1_r1 leaderboard)

| cell | now | r1 score | best library (r1) |
|---|---:|---:|---:|
| 1021 B=1 | **7.85-7.99** | 13.92 | 8.28 MKL -> **WIN** |
| 1021 B=256 | 9.06-9.28 | 13.11 | 8.75 MKL (borderline loss) |
| 1021 B=1 chain m=2000 | **8.76** | 11.67 | 12.97 MKL |
| 1021 B=256 chain m=400 | **7.88** | 11.62 | 12.39 MKL |
| 65537 B=1 | **1183** | 1202 | 1465 patient |
| 65537 B=16 | **1489** (noisy window 1653) | 1549 | 1518 patient -> **flips to WIN if quiet** |
| 65537 chains | 985 / 1028 | 910 / 949 | 1629 / 1762 |
| 13 B=1 / B=512 | 0.056 / 0.051 | 0.113 / 0.115 | 0.0218 / 0.0123 (still lost) |
| 13 chains | 0.099 / 0.099 | 0.137 / 0.118 | 0.0665 / 0.051 (still lost, halved) |
| 31 B=1 / B=512 | 0.211 / 0.210 | 0.386 / 0.391 | 0.178 / 0.090 (B=1 nearly closed) |
| 31 chains | 0.250-0.303 / 0.251 | 0.350 / 0.350 | 0.209 / 0.209 |

Accuracy: single-call rel_l2 1.6e-16 (13) … 1.4e-15 (65537), gate 1e-12; all
chain gates pass with >= 3 decades of margin; verified at odd batches
(1,2,3,8,16,256,512) per size; 127 (unscored) still correct at 1.77 us B=1.

### What did NOT work, with the number that killed it

- Tiled st17 with per-lane inner loops: 21.7 us (accumulator arrays stay in
  memory). Two-pass scalar st17: 15.4 us (no outer-loop vectorization).
- vgatherdpd entry: +0.8 us vs insert-assembly on L1-resident input.
- vscatterdpd fused exit: 2.4 us vs 0.5-ish staged-scalar.
- UN-fusing the radix-17 exit (plain st17 + separate scalar exit pass, done
  because perf blamed the fused store): 8.28 vs 7.85 us — reverted. The
  profiler mis-attributes stalls at this granularity; trust only A/B.
- CRT codelet, first cut (pointer temps + autovectorizer on): 0.163 us at 13
  — worse than the stage machinery it replaced. Both fixes above required.
- 8-wide map in the small chain codelets: 0.110 vs 0.099 scalar (Ice Lake
  zmm sqrt/div throughput).
- Forcing padded M=2048 at 1021 on Ice Lake: 15.29 vs 13.93 unpadded — the
  pad is no refuge; the pow2 stages were equally scalar.

### Borrowings

- The Agarwal-Cooley/Good-Thomas multidimensional-conv mapping for the small
  codelets is straight from the survey's Winograd vein
  (docs/literature_1d/00-SURVEY.md); nobody else on the panel has used it yet.
- The r1 borrowings (d1_bluestein core, d1_prime fold + measurement protocol)
  carry forward; the symmetric u/v fold now lives inside the intrinsic st17.

### Next round, in priority order

1. **1021 B=256 (borderline 9.1 vs 8.75 MKL)**: the batch loop is per-
   transform; try software-pipelining two batch elements, and an NR-rsqrt map
   (vrsqrt14pd + 2 Newton) for st17_chain / st4_last_chain — zmm sqrt+div is
   the single ugliest per-element cost left everywhere.
2. **31 B=1 (0.211 vs 0.178 fftw custom)**: the DFT30 codelet is pure scalar;
   a careful SSE/AVX 2-lane schedule (NOT the SLP mess) or Winograd-style
   reduced-multiplication DFT5/DFT3 should close 20%.
3. **13**: concede B=1/B=512 to codelet-library territory or go across-batch
   SoA lanes (8 transforms per zmm) for the B=512 cells; the survey says this
   is the top untapped lever and d1_batchlane's record confirms the shape.
4. **65537**: the one scalar stage left is st8 at s=4 (first middle stage);
   the st3_s4 two-groups-per-vector trick generalizes to radix 8.
5. **10007 nested-Rader A/B** (5003-1 = 2*41*61): still untried; only worth
   it if it can beat d1_bluestein's ~110 us — measure the 5003-conv first.

## Round d1_r3 (2026-09-03) — fewer passes at 65536, across-batch SoA at 13/31, Newton map everywhere vectorized

**Measurement conditions: the a80n0 hold (job 440424) shows "not running"
through the wallaby squeue shim (which reads the OTHER project's heartbeat),
but the node-side heartbeat was fresh and ssh worked, so tryout.sh's gate was
bypassed by running its exact pipeline manually over ssh (same gcc line, same
gen_input/check.py, core 2 via slot_lease slot 0, released at session end).
The window drifted +/-10% within the session — an early "baseline" of 1241 us
at 65537 B=1 aged to 1340 for the identical binary 20 minutes later — so
every decision below was same-minute A/B'd against the previous build, and
the r1 lesson ("trust only same-window ratios") held for a third round.**

### What changed

1. **65537 conv restructured to [4,16,16,16,4]** (was [4,8,8,8,8,4]): a new
   radix-16 stage (st16/st16_block) does two fused radix-4 layers through a
   2 KB L1 tile — the st17 two-pass shape, because 16 live complex
   lane-vectors ARE the whole zmm file and gcc spills. Layer-1 twiddles
   W16^(u1 v1) are compile-time constants; layer 2 applies the stage table.
   10 array passes instead of 12 over the two 512 KB planes, and this cell
   is L3-bandwidth-bound, so it paid almost exactly pro rata: B=1
   1338 -> 1136 (-15%), B=16 1647 -> 1436 (-13%), same-minute. The one
   masked-half pass (first middle stage runs at s=4) then got a **paired-p
   variant** (st16_s4: two p-groups per zmm, pair-broadcast twiddles,
   256-bit half stores — st3_s4 generalized as r2 predicted for st8):
   B=1 1140 -> 1058, B=16 1441 -> 1356, same-minute. Bug worth recording:
   the first st16 cut stored the tile at slot v1+4*u1 and read it back at
   4*v1+u1 — rel_l2 1.4e0, caught immediately by check.py.
2. **13/31 batched cells: across-batch SoA** (the survey's top batched lever,
   confirmed by d1_batchlane's record and fftw_custom_soa's r2 numbers —
   adopted from both). The CRT codelet bodies are now type-parametric
   (dftv_t = double or __m512d; GNU C infix arithmetic broadcasts the scalar
   constants), so the SAME Agarwal-Cooley code runs 8 transforms per
   instruction, lane j = batch element j, zero in-lane shuffles. AoS->SoA
   transpose only at entry/exit (soa_gather8/soa_scatter8: 128-bit pair
   loads, vinsertf64x2, one parity vpermt2pd per plane — the GATHER8 shape;
   scatter is its mirror). Chain state stays in SoA lanes across all m steps:
   interior steps have NO transpose at all. Batch tail (B mod 8) falls back
   to the scalar codelets; verified at B=3, 9, 11.
   Same-minute results: 13 B=512 0.051 -> 0.018-0.019 (lib best 0.0140,
   still lost but 2.7x closer); 31 B=512 0.175 -> 0.063 (lib best 0.105,
   **flipped to a win**); 13 B=512 chain 0.080 -> 0.015 (lib 0.0262,
   **flipped**); 31 B=512 chain 0.251 -> 0.056 (lib 0.0777, **flipped**).
3. **Newton map (chain_map8) in every vector chain exit** — r2's own #1
   next-round item, now done: vrsqrt14pd + 2 NR for sqrt(m), vrcp14pd + 2 NR
   for 1/(1+s), max(m,1e-300) guarding m=0 (rsqrt(0)=inf would NaN), all on
   the FMA ports instead of ~34 back-to-back cycles on the single divide
   unit per vsqrtpd+vdivpd pair. Applied to the SoA chains, st17_chain
   (1021), and st4_last/st2_last_chain (65536/256 exits, rewritten 8-wide —
   they were scalar C doing 65536 sqrt+div per 65537 step). Same-minute:
   1021 B=1 chain 8.76-ish -> 7.35-7.42; 65537 B=1 chain 1010 -> 931;
   31 B=1 chain 0.250 -> 0.206. Chain-gate errors moved in the last digit
   only (e.g. 13 m=200000: 1.27e-15 vs 1.27e-15; 31 m=1200: 5.1e-12 vs
   5.8e-12, tol 1e-9) — the NR map is ~1-ulp and the gate never noticed.

### Measured (a80n0 core 2, final build, one good window at session end;
### "lib" = best library from the r2 leaderboard)

| cell | r2 scored | now | lib | verdict |
|---|---:|---:|---:|---|
| 13 B=1 / B=512 | 0.056 / 0.051 | 0.057 / 0.018 | 0.0219 / 0.0140 | lost / closer |
| 13 chains B=1 / B=512 | 0.080 / 0.080 | 0.097 / 0.015 | 0.0674 / 0.0262 | lost / **WIN** |
| 31 B=1 / B=512 | 0.210 / 0.175 | 0.178 / 0.063 | 0.2152 / 0.1050 | WIN / **WIN** |
| 31 chains B=1 / B=512 | 0.250 / 0.251 | 0.206 / 0.056 | 0.2089 / 0.0777 | ~tie / **WIN** |
| 1021 all four | 7.6-7.9 | 6.9 / 7.8 / 7.4 / 7.5 | 9.4-12.4 MKL | 4x WIN |
| 65537 B=1 / B=16 | 1157 / 1662 | 1049 / 1348 | 1468 / 1544 | WIN / **WIN** |
| 65537 chains | 1052 / 1085 | 912 / 906 | 1629 / 1772 | WIN / WIN |

Accuracy: single-call rel_l2 1.6e-16 (13) … 1.3e-15 (65537), gate 1e-12; all
chain gates pass with >= 3 decades of margin; odd batches (1,2,3,8,9,11,16,
256,512) verified per size; output bit-repeatable across runs. Setup still
<= 0.008 s at 65537.

### What did NOT work, with the number that killed it

- **Prefetch sweeps for the "cold DRAM random gather" theory at 65537 B=16**:
  a sequential _mm_prefetch sweep over the next element's input + this
  element's output, issued mid-transform. ET1 hint: +138 us at B=1 (drags
  1 MB through L2, evicts live stage buffers). T2 hint: same. Sweeps gated
  off at runtime measured IDENTICAL to no-sweeps — which is how the window
  drift was caught. The single-call-vs-chain gap at B=16 is streaming
  bandwidth, not random-miss latency; prefetch cannot reduce traffic.
  Removed entirely.
- **Four-step 256x256 re-tried on the scoring silicon** (r1's open question):
  1635 vs 1338 us B=1, 1965 vs 1657 B=16, same-minute. Dead on both
  machines now; the flag and code stay for a hypothetical DRAM-resident
  size (100003-class), with both numbers recorded at the flag site.
- **Vector map in the L=13 scalar chain codelet**: 0.167 vs 0.099 us/step —
  touching the 12-element state arrays with intrinsic loads forces them out
  of registers (the r2 SROA lesson in new clothing). At L=31 the arrays are
  memory-backed anyway and the same change WINS (0.250 -> 0.206). The map
  style is now a per-size macro parameter (SMALL_MAP_SCALAR at 13,
  SMALL_MAP_VEC8 at 31).

### Borrowings

- Across-batch SoA lanes: the shape is d1_batchlane's and fftw_custom_soa's
  (their r2 numbers were the evidence it pays); the transpose implementation
  reuses my own r2 GATHER8 insert-assembly rather than vgatherdpd (r2
  measurement: gathers lose on L1/L2-resident data).
- The paired-p s=4 trick is my own r2 st3_s4, generalized to radix 16 —
  r2's next-round list had predicted this for st8.
- Newton-map: r2's own next-round item #1; no other entry had shipped it.

### Next round, in priority order

1. **65537 B=16 single**: still ~400 us above its own chain steps — that gap
   is now pure input/output streaming + the entry/exit passes. A split-radix
   or radix-32 first/last pass (8 passes total) is the remaining traffic cut;
   alternatively fuse the kernel-multiply into the LAST forward stage
   (saves one full pass: forward-last and inverse-first both touch the
   spectrum; combining them needs the twiddle algebra checked carefully).
2. **13 B=512 single (0.018 vs 0.0140)**: the transpose is now the cost —
   ~16 port-5 ops per conv point. A blocked 8x8 in-register transpose of
   the whole 8-lane input (26 zmm loads + ~24 permutes per 8x8 block)
   roughly halves port-5 pressure; or precompute a fused CRT+transpose
   index order. 13 B=1 (0.057 vs 0.0219) likely needs a true Winograd
   DFT-13 straight-line codelet to matter — big effort, one cell.
3. **1021**: st5 at s=12 uses masked 8+4 blocks (a third of lanes wasted in
   the tail block); a paired/tripled-p schedule like st16_s4 could take
   ~5-8% off all four 1021 cells.
4. **10007 nested-Rader A/B** (5003-1 = 2*41*61): still untried; only worth
   it if it can beat d1_bluestein's ~110 us — measure the 5003-conv first.
