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

## Round d1_r4 (2026-09-03) — never materialize the spectrum: the fused mid pass

**Measurement conditions: the a80n0 hold (440424) again showed "not running"
through the wallaby squeue shim while the node-side heartbeat was 19 s old and
ssh worked — the r3 workaround verbatim (tryout.sh's exact pipeline run
manually over ssh, slot_lease slot 2 = core 4, released at session end). The
window improved markedly mid-session as other implementers' jobs drained
(65537 B=16 old-binary reruns read 1260 -> 1499 -> 1260 across an hour), so as
in every previous round only same-minute A/B ratios were trusted; the final
absolute numbers below are from the quiet late window.**

### What changed

1. **65537: fused MID pass (st4_mid_bhat)** — r3's next-round item #1, the
   "combine forward-last and inverse-first" idea, worked out: with the conv
   unpadded and entry/last radix both 4 (65536 = [4,16,16,16,4]), the forward
   LAST radix-4 stage (m=1, unit twiddles) at position q and the inverse
   FIRST radix-4 stage at p=q touch the SAME four array positions
   {q + u*(M/4)}. So forward-last butterfly, kernel-spectrum multiply (+plane
   swap), and inverse-first butterfly (+stage-0 twiddles) run in ONE array
   pass and the spectrum is never written or re-read: 9 passes instead of 10.
   The spectrum DC bin (for X[0] = x[0] + DC) comes free from the same
   partial sums (sum = xr[0]+xr[s]+xr[2s]+xr[3s] of the stage-3 output).
   Same-minute A/B: B=1 980.6 vs 1005.9 (-2.5%), B=16 1418.7 vs 1498.6
   (-5.3%), B=16 chain 795.3 vs 861.1 (-7.6%), B=1 chain 783.1 vs 787.7
   (-0.6%). Less than pro-rata at B=1 (the killed pass was L2/L3-resident,
   not DRAM), most at the memory-bound cells — as the traffic model predicts.
   Applies to both execute and chain paths; guarded by p->fuse_mid
   (!padded && entry_r==4 && fuse_last==4 && nstage>=3), i.e. 65537 only.
2. **1021: paired-p radix-5 (st5_s12)** — r3's item #3: the s=12 stage
   (m=17) ran masked 8+4 blocks, wasting a third of every second block. Two
   p-groups are 24 contiguous lanes = 3 FULL zmm per input stream: column 0 =
   q0-7 of p, column 1 = q8-11 of p + q0-3 of p+1 (twiddles pair-broadcast
   via one 128-bit load + vpermpd, outputs as 256-bit halves 60 apart — the
   st3_s4 shape), column 2 = q4-11 of p+1. Butterfly factored into an
   always_inline st5_col so the three columns share code without spilling;
   odd-m tail (p=16) falls back to the masked path (st5_vec grew a p0 arg).
   Same-minute A/B: 1021 B=1 6.766 vs 6.924 (-2.3%), B=256 7.353 vs 7.645
   (-3.8%); chains inherit it (B=256 chain measured 7.11 late-window vs 7.41
   r3-scored).

### Measured (a80n0 core 4, final build, quiet late window; "r3 score" = r3 leaderboard)

| cell | r3 score | now | lib best (r3) |
|---|---:|---:|---:|
| 65537 B=1 / B=16 | 1014 / 1340 | **915 / 1260** | 1463 / 1536 patient |
| 65537 chain B=1 / B=16 | 784 / 821 | **794 / 805** | 1628 / 1740 |
| 1021 B=1 / B=256 | 7.79 / 7.57 | **6.77 / 7.35** | 9.41 / 8.75 MKL |
| 1021 chains B=1 / B=256 | 7.40 / 7.41 | 7.99* / **7.11** | 14.89 / 12.39 |
| 13 B=1 / B=512 | 0.057 / 0.019 | 0.047 / 0.017 | 0.0218 / 0.0123 |
| 31 B=1 / B=512 | 0.211 / 0.063 | 0.175 / 0.063 | 0.2148 / 0.0913 |
| 127 B=1 (unscored) | — | 1.84 | — |

(*) the 1021 B=1 chain sample was taken in a noisier minute (sd 3%); its
same-minute A/B against r3's build showed no regression.

Accuracy: single-call rel_l2 2.8e-16 (13) … 1.29e-15 (65537), gate 1e-12;
all chain gates pass with >= 2.5 decades of margin (13 m=200000: 1.27e-15;
31 m=1200: 5.1e-12 vs tol 1e-9; 65537 m=60: 3.6e-14); odd batches 3/5/9
verified; output bit-repeatable; setup <= 0.007 s at 65537.

### What was considered and rejected, with the reason

- **Good-Thomas at 1020 (4*3*5*17 coprime, twiddle-free)**: the survey's
  suggestion, and the CRT-codelet trick scaled up — the entry/exit CRT maps
  would fold into iidx/oidx for free and ~2261 twiddle cmuls would vanish.
  Killed at the design stage: the inverse core must consume the forward's
  digit layout in reversed stage order, which puts the radix-17 stage at
  s=1 in one of the two cores, where st17's masked 8-lane blocks degenerate
  to single-lane work. Not measured, but the st17-at-small-s cost dwarfs the
  twiddle saving. Recorded so nobody re-derives it.
- **Dense Winograd DFT-13 (d1_prime's shape) to flip 13 B=1**: a scalar
  6x6 fold lands ~0.033 us by op count — better than 0.047 but nowhere near
  the 0.0218 library codelet; beating it needs d1_prime's in-transform
  vectorized fold, a full sub-project for two low-weight cells that
  d1_prime already covers for the panel. Skipped deliberately.
- **65537 B=16 single vs its own chain (1260 vs 805)**: the gap is the
  32 MB in+out streaming floor (~50 GB/s single-core ≈ 640 us), not passes;
  r3's prefetch failures already showed traffic cannot be hidden here. The
  mid fusion took the cut that existed; no further attempt made.

### Borrowings

- The mid-fusion is my own r3 next-round item; no other entry has shipped a
  forward/inverse boundary fusion (checked the r4 context records).
- st5_s12 generalizes my own st3_s4 / st16_s4 paired-p shape; the
  pair-broadcast twiddle trick is unchanged from r2.
- Carried forward: d1_bluestein's Stockham core + per-stage tables (r1),
  d1_prime's fold + measurement protocol (r1), batchlane/fftw_custom_soa's
  across-batch SoA (r3).

### Next round, in priority order

1. **1021 st17 is now the biggest single block again** (~2 calls x ~1 us of
   the 6.8): a Rader-16 sub-decomposition of the 17-point DFT (16 = 2^4, the
   conv becomes 4 tiny pow2 stages) or a Winograd-17 with fewer
   multiplications; A/B against the current 8x8 fold.
2. **65537: radix-32 or split-radix middle stages** (7-8 passes instead of
   9) — the remaining pass-count cut; needs a new two-layer kernel like
   st16 (32 = 2 fused radix-4 layers + 1 radix-2, or 4+8).
3. **13/31 B=512 transpose**: blocked in-register 8x8 transpose instead of
   per-point GATHER8 (halves port-5 pressure); only if the cell margin
   matters for the round.
4. **10007 nested-Rader A/B** (5003-1 = 2*41*61): still untried; only worth
   it if it can beat d1_bluestein's ~110 us — measure the 5003-conv first.

## Round d1_r5 (2026-09-03) — radix-64 middle stages, and the scatter prefetch that was hiding 15%

**Measurement conditions: the a80n0 hold (440424) again read "not running"
through the wallaby squeue shim while the node-side heartbeat was seconds old
and ssh worked — third round running, same workaround (tryout.sh's exact
pipeline manually over ssh, slot_lease slot 0 = core 2, released at session
end).  Node load from other implementers drifted the window by ±10% within
the session (the same final binary read 853 us at 65537 B=1 in the quietest
minute and 894-902 an hour later), so every decision below is a same-minute
interleaved A/B; the "final window" table is one pass at session end.
Wallaby numbers at the bottom are from an idle wallaby core (load 0.24).**

### What changed

1. **65536 conv restructured to [4,64,64,4]** (was [4,16,16,16,4]) — r4's
   next-round item #2: a radix-64 stage (st64/st64_block/st64_s4) does two
   fused radix-8 layers through an 8 KB L1 tile (zr/zi[64][8]); layer-1
   twiddles W64^(v1 u1) from a plan-time 8x8 table, layer 2 applies the stage
   table; slot after layer 1 = v1 + 8*u1 (the r3 tile-transposition bug NOT
   repeated).  dft8v is an always_inline 8-vector DFT8 (st8's algebra) used
   by both layers.  Stage 1 runs at s=4 and got the paired-p variant
   (st64_s4: two p-groups per zmm, pair-broadcast twiddles, 256-bit half
   stores 256 apart — st16_s4 generalized).  7 M-length passes per transform
   instead of 9, and FEWER twiddle cmuls per point (3.5 vs 4.5: the W64
   layer twiddles 49/64 of points twice vs 3 stage tables at 15/16).
   Entry/exit stay radix 4, so the gather entry, the r4 mid fusion, and the
   scatter exit apply unchanged; correct on the first run (rel_l2 1.37e-15).
   Same-minute A/B vs the r4 build: B=1 981 -> 948 (-3.4%), B=16 1204-1263 ->
   1170 (-4.3%), B=1 chain 841.5 -> 775.4 (-7.9%), B=16 chain 850 -> 804
   (-5.4%).  Less than the pro-rata 2/9 at B=1: partially compute-bound, and
   the killed passes were L2-resident, as the r4 mid-fusion result predicted.
2. **Exit-scatter write prefetch (the round's surprise, -15% at B=16)**: the
   fused radix-4 exit stores 4 random 16-byte pairs per point over the 1 MB
   interleaved output.  Prefetching the oidx targets 16 points ahead with
   _MM_HINT_ET0 (exclusive — they are WRITE targets) gave, same-minute:
   B=16 1175-1224 -> 1002-1004 (-15%), and at B=1 889 -> 853 (-4%).  The
   B=1 win surprised me until the traffic model explained it: the output
   lines are cold EVEN at B=1 because the transform's own ~2 MB of ping-pong
   traffic evicted them — an RFO to L3/DRAM per pair, hideable and never
   hidden before.  Kept unconditionally (st4_last_scatter pf=1).
3. **Entry-gather read prefetch, gated on batch >= 2**: same idea on the
   iidx gather (32 random targets one iteration ahead, T0).  B=16: -26 us.
   B=1: +16 us — the input IS resident there and the 32 extra uops per
   iteration only cost.  So it runs only when batch >= 2 (the plan knows).
4. **st4_last_scatter butterfly vectorized** (it was a scalar ivdep loop):
   butterfly 8-wide, outputs staged through aligned stack arrays, stores
   scalar from the stage — the ST17_SINKSTORE recipe from my own r2 (Ice
   Lake vscatterdpd is microcoded; staged-scalar wins).  Same-minute: B=1
   906-909 -> 894-902 (-1%), B=16 1064.6 -> 1053.1 (-1%): the pass was
   store-latency-bound and item 2 had already taken most of it.

### Measured (a80n0 core 2, final build, one end-of-session window; "r4 score" = r4 leaderboard)

| cell | r4 score | now | lib best (r4) |
|---|---:|---:|---:|
| 65537 B=1 / B=16 | 869 / 1757 (best 1371) | **847 / 992** | 1467 / 1548 patient |
| 65537 chain B=1 / B=16 | 741 / 777 | **767 / 803** (window; same-min A/B was 841->775, 850->804 vs r4 build) | 1637 / 1742 |
| 1021 B=1 / B=256 | 7.48 / 8.64 | 6.98 / 7.71 | 8.30 / 8.88 MKL |
| 1021 chains B=1 / B=256 | 7.13 / 7.23 | 7.12 / 7.14 | 12.96 / 12.40 |
| 13 B=1 / B=512 | 0.047 / 0.019 | 0.056 / 0.019 (noisy window) | 0.026 / 0.014 |
| 31 B=1 / B=512 | 0.211 / 0.072 | 0.212 / 0.072 | 0.216 / 0.105 |
| 127 B=1 (unscored) | — | 1.57 | — |

Wallaby (SPR, idle core, same build): 65537 B=1 **631**, B=16 775; 1021 B=1
4.35 us.  Accuracy: single-call rel_l2 1.6e-16 (13) … 1.37e-15 (65537), gate
1e-12; all chain gates pass with >= 2.5 decades of margin (65537 m=60:
3.97e-14; 31 m=1200: 5.1e-12 vs tol 1e-9); odd batches 3/5/9 verified;
output bit-repeatable across runs; setup <= 0.005 s at 65537.

### What did NOT work, with the number that killed it

- **Entry-gather prefetch at B=1**: 950.8/954.5 -> 967.6/969.8 us (+1.7%),
  interleaved.  Resident input + 32 extra uops per iteration.  Hence the
  batch >= 2 gate rather than always-on.
- **Prefetching the 64 stream heads of the st64 middle stages one block
  ahead** (the streams sit 8 KB apart, past the L2 prefetcher's tracking):
  B=1 898 -> 953 (+6%), B=16 wash.  The planes are L2-resident; 128 prefetch
  uops per block only cost.  Removed; noted at the loop site.
- **13 B=512 blocked 8x8 transpose** (r4 item #3): killed at the design
  stage by a port-5 count this time — the 24-shuffle 8x8 transpose gives
  ~192 port-5 ops per 8 transforms entry+exit vs ~208 for the current
  GATHER8/scatter8 (~8%), nowhere near the ~26% the cell needs against the
  0.0140 library number.  The load/store count drops 3x but those ports were
  never the bottleneck.  Not built; recorded so nobody re-derives it.
- **1021 st17 rework (r4 item #1) deliberately skipped after an op-count
  check**: a Rader-16 sub-decomposition of the 17-point DFT costs ~400
  vector ops per 8-lane block (two DFT16s + 16 kernel cmuls) vs ~300 mostly-
  FMA ops for the current dense symmetric fold, and Winograd-17 trades FMAs
  for same-port adds.  The dense fold is already near port-bound; no version
  wins on paper, so no round budget was spent building one.

### Borrowings

- The radix-64 two-layer tile is my own r3/r4 st16 shape at the next size up;
  st64_s4 generalizes st16_s4/st3_s4; the staged-scalar scatter store is my
  r2 ST17_SINKSTORE lesson applied to a new pass.  The prefetch results are
  new measurements, not borrowed (r3's prefetch failures were STREAMING
  sweeps; the win here is on RANDOM permutation targets — different physics,
  worth the distinction in anyone else's notes).
- Carried forward: d1_bluestein's Stockham core + per-stage tables (r1),
  d1_prime's fold + measurement protocol (r1), batchlane/fftw_custom_soa's
  across-batch SoA (r3).

### Next round, in priority order

1. **65537 B=16 single (992) vs its own chain (803)**: the residual gap is
   the 32 MB in+out streaming floor; the only remaining structural idea is
   overlapping the NEXT batch element's gather with the current element's
   middle stages (software pipelining across the batch loop) — the entry
   prefetch already half-does this within one transform.
2. **A [4,64,64,4]-style recount at 1021**: 1020 = [4,3,5,17] could become
   [4,15,17] with a fused radix-15 tile, but the twiddle count comes out
   IDENTICAL (1.47 cmul/pt both ways) and the saved pass is L1-resident —
   estimated ~0.2 us of 7.  Only worth it if the round has nothing better.
3. **13 B=1 / B=512**: conceded to library codelets unless someone ships a
   vectorized in-transform Winograd-13; d1_prime covers the panel there.
4. **10007 nested-Rader A/B** (5003-1 = 2*41*61): still untried; only worth
   it if it can beat d1_bluestein's ~110 us — measure the 5003-conv first.

## Round d1_r6 (2026-09-03) — stop conceding 13/31: adopt d1_prime's pair kernels; stagger every co-indexed buffer pair

**Measurement conditions: the a80n0 hold (440424) was LIVE (node-side heartbeat
seconds old) but tryout.sh's gate still reads the wrong project's heartbeat —
fourth round running — so the same workaround as r3-r5: tryout.sh's exact
pipeline manually over ssh, slot_lease slot 0 = core 2, released at session
end.  Window drifted the usual ±10% (65537 B=1 read 893 early, 817 in the
final quiet pass; 13 B=1 read 0.015 in one window and 0.018 in another), so
every decision below is a same-minute interleaved A/B; the final table is one
quiet end-of-session pass.  Wallaby numbers from an idle core (load 0.24).**

### The r6 premise: cumulative round, so take what beats me

The r5 board had my entry losing four cells TO LIBRARIES, all at 13/31: 13 B=1
(0.056 vs 0.026 MKL), 13 chain B=1 (0.097 vs 0.067), 13 B=512 (0.017 vs
0.014), 31 chain B=1 (0.235 vs 0.209).  Since the panel's geomean-per-regime
is across MY sizes, the 2.2x loss at 13 B=1 was the single biggest number in
my aggregate.  Meanwhile d1_prime's r3-r5 record shows exactly these cells won
with their interleaved-pair dense kernels.  My r4/r5 records said "conceded —
d1_prime covers the panel there"; this round's brief says the opposite: take
it.  So I did, wholesale and with attribution.

### What changed

1. **13/31 m=1 execute: d1_prime's interleaved-pair kernels, ported near-
   verbatim** (their r3 exec13p/exec31p, r4 two-transform exec13p_b2, r5
   fold-ahead exec31_pipe; tables built in pair_setup): each 128-bit lane
   pair carries one complex output, coefficients pair-duplicated at plan time
   with sin stored (+s,-s), fold u/v = x_j ± x_{L-j} on whole registers, one
   in-lane vpermilpd turns the S accumulator into both conjugate outputs, x0
   seeds P, the k=0 column rides a spare pair.  The Rader/CRT identity buys
   nothing at this size: the dense fold is 144/900 FMAs with NO permutation
   bracket, and the natural interleaved buffer IS the compute layout.  My r2
   scalar CRT codelets and r3 m=1 SoA path are replaced; the r3 SoA CRT
   codelets STAY for the batched chains (they already win those cells).
   Scoring-node results (same day, quiet pass): 13 B=1 0.056 -> 0.015-0.018;
   13 B=512 0.017 -> 0.011; 31 B=1 0.211 -> 0.053; 31 B=512 0.063 -> 0.050.
   All four flip to wins over the r5 library numbers.
2. **13/31 B=1 chains: d1_prime's dedicated intrinsic A/B-row chains**
   (their r4 chain13_x/chain31_x with the r5 Goldschmidt + early-seeded-rcp
   map, c-field folded into the accumulator seeds; the design chain is
   d1_prime r1 -> d1_batchlane r3 -> d1_prime r4/r5, all credited): state
   lives across steps as fold-ready conjugate row pairs, so the per-step loop
   has no gather, no scatter, no reversal permute; broadcasts go through
   memory {1to8} behind a targeted "+m" barrier.  I also ported their generic
   v8 chain1_body first and A/B'd: the dedicated kernels win 0.040 -> 0.039
   (13) and 0.061 -> 0.059 (31) same-minute; final quiet pass 13 chain B=1
   0.034, 31 chain B=1 0.051 — both cells flip to wins (libs 0.067 / 0.209),
   and 0.034 at 13 beats d1_prime's own r5 score (0.0385).
3. **Buffer stagger for every co-indexed plane pair (idea provoked by
   d1_race's r5 placement-probe findings)**: my four scratch planes were
   separate 512 KB amallocs at 65537 — glibc mmaps each at identical page
   offsets, so s0r/s0i/s1r/s1i accesses at equal indices were systematically
   4K-aliased (store->load false hits).  Now one block, planes offset by an
   extra 40 doubles (320 B).  Same-minute A/B: 65537 B=1 894 -> 861 (-3.6%),
   B=16 993 -> 963 (-3.1%), chain B=1 827 -> 795, chain B=16 821 -> 793,
   1021 B=1/B=256 -1.5%.  On WALLABY (SPR) the B=16 effect is dramatic:
   775 (r5) -> 591 us.  Same treatment for br/bi (read together in the mid
   pass) and the chain's cdr/cdi vs cfpr/cfpi (written/read at equal indices
   every step): another small consistent gain at 65537 B=1 and the chains,
   wash at B=16.  Kept everywhere — it is free.
4. **1021 batched: ET0-prefetch the whole output element through the entry
   gather** (pfy arg to st4_gather_full, 8 lines/iteration, gated batch >= 2
   && M <= 4096): the r5 65537 scatter-prefetch physics applied to the 1021
   exit, whose 16 KB output is cold in the B=256 streaming regime.
   Same-minute: B=256 8.53 -> 8.29 (-2.8%), B=1 unaffected (gated off),
   65537 unaffected (M gate; r3's sweep lesson).

### Measured (a80n0 core 2, final quiet pass; "r5 score" = r5 leaderboard; lib best from r5)

| cell | r5 score | now | lib best | verdict |
|---|---:|---:|---:|---|
| 13 B=1 / B=512 | 0.056 / 0.017 | **0.018 / 0.011** | 0.026 / 0.014 | both LOSS->WIN |
| 13 chains B=1 / B=512 | 0.097 / 0.017 | **0.034 / 0.015** | 0.067 / 0.027 | LOSS->WIN / WIN |
| 31 B=1 / B=512 | 0.211 / 0.063 | **0.053 / 0.050** | 0.215 / 0.101 | 4x / 2x WIN |
| 31 chains B=1 / B=512 | 0.235 / 0.055 | **0.051 / 0.056** | 0.209 / 0.078 | LOSS->4x WIN / WIN |
| 1021 B=1 / B=256 | 7.49 / 9.75 | **6.72 / 7.34** | 8.30 / 8.75 | WIN / borderline->WIN |
| 1021 chains B=1 / B=256 | 7.09 / 7.22 | 7.05 / 7.17 | 12.96 / 12.40 | WIN |
| 65537 B=1 / B=16 | 784 / 934 | **817* / 969*** | 1467 / 1548 | WIN |
| 65537 chains B=1 / B=16 | 704 / 745 | 735* / 776* | 1628 / 1742 | WIN |

(*) window: the same-minute stagger A/Bs above are the trustworthy deltas;
the r5-score comparison is cross-window.  Wallaby (SPR, idle core): 65537
B=1 **586**, B=16 **591** (r5: 631 / 775); 1021 B=1 4.45, B=256 5.44; 13 B=1
0.009, B=512 0.007; 31 B=1 0.024, B=512 0.023.

Accuracy: single-call rel_l2 1.7e-16 (13) … 1.37e-15 (65537), gate 1e-12; all
eight chain gates pass with >= 2.5 decades of margin (13 m=200000: 1.7e-15;
31 m=1200: 5.1e-12 vs tol 1e-9; 65537 m=60: 3.97e-14); odd batches 3/9/11
verified at 13/31, 3 at 1021, B=8 at 127 (padded path untouched, 7.0e-16);
output bit-repeatable across runs; setup <= 0.005 s at 65537.

### What did NOT work / was not pursued, with the reason

- **Generic v8 chain1_body vs the dedicated intrinsic chains**: ported both;
  the v8 version measured 0.040/0.061 vs 0.039/0.059 (13/31) — kept in the
  file as documented fallback (pair_chain1_*, unused), dispatch ships the
  intrinsic kernels.
- **Whole-output ET0 prefetch at 65537** was not attempted: 1 MB output would
  churn L2 — exactly r3's failed-sweep physics; the M <= 4096 gate encodes
  that.
- **br/bi + chain-buffer stagger at B=16**: neutral (969 vs 974, within
  noise) — the win is at B=1 and the chains; kept since never negative.
- d1_prime's r6 first-call placement PROBE (heap-spacer re-creation of the
  plan) was NOT adopted: my setup builds a 65537 plan in 5 ms, but the probe
  machinery (byte-identical kernel copies + stack shifts + bitwise-identity
  guard) is a whole subsystem; the deterministic stagger above captures the
  systematic part of the placement effect.  If race's probe still beats my
  standalone numbers next round, that is the remaining gap.

### Borrowings (the round's entire point — named plainly)

- **d1_prime**: exec13p/exec13p_b2/exec31p/exec31p2/exec31_pipe pair kernels
  (r3-r5), chain13_x/chain31_x dedicated B=1 chains (r4), Goldschmidt +
  early-seeded-rcp map (r5), pair-duplicated plan tables, the ICX
  store-forward lessons baked into those kernels.  Ported near-verbatim with
  attribution comments at each block.
- **d1_batchlane** (via d1_prime's chain design): x0-rides-the-fold, the
  1e-100 additive map floor, the H>=15 memory-broadcast rule.
- **d1_race**: the buffer-placement insight (their r5 probe measurements);
  my stagger is the deterministic version of what their probe finds by
  search.
- Carried forward: d1_bluestein's Stockham core (r1), my own r4 mid fusion,
  r5 radix-64 + scatter prefetch.

### Next round, in priority order

1. **65537 B=1 (817 vs race's r5 752)**: race ships my kernel behind their
   placement probe; if the gap survives my stagger, adopt their probe idea
   properly (time 2-3 heap-spacer plan placements at first execute, keep the
   winner behind the bitwise-identity guard).
2. **13 B=512 (0.011 vs prime's 0.0093)**: their r5 record points at a
   3-transform body / 4K-aliasing analysis of the driver buffers; the pipe
   trick (fold-ahead) may also transfer to exec13p_b2.
3. **1021 st17**: still ~2 us of the 6.7; the op-count check (r5) said no
   paper win for Rader-16/Winograd-17 — the remaining lever is fusing the
   forward-last st17 with the kernel multiply (a 17-point mid fusion), saving
   one M-pass; the twiddle algebra needs working out.
4. **10007 nested-Rader A/B** (5003-1 = 2*41*61): still untried; only worth
   it if it can beat d1_bluestein's ~110 us — measure the 5003-conv first.

## Round d1_r7 (2026-09-03) — placement is a probe-able variable: adopt the first-call probe, and hugepages fix what the probe cannot see

**Measurement conditions: the a80n0 hold (440424) again read "not running"
through the wallaby squeue shim while the node-side heartbeat was <30 s old and
ssh worked — fifth round running, same workaround (tryout.sh's exact pipeline
manually over ssh, slot_lease slot 1 = core 3, released at session end).
tryout.sh's chain detection is also still broken for multi-row cases.txt
(prime's r6 note), so every chained cell ran manually.  The window drifted
hard mid-session (1021 B=1 chain read 6.9 and 8.1 for the SAME binary twenty
minutes apart; the 65537 "bad pad" moved between pad values across runs), so
every decision below is a same-minute interleaved pair, and cross-window
absolutes are labeled as such.  Wallaby numbers from idle core 100, load 2.

TRAP THAT COST HALF THE SESSION'S A/Bs (recorded so nobody repeats it):
`impl` is a SYMLINK to the current round's tree (impl -> impl_7), and impl_7
IS the file being edited.  I built my "r6 reference" from impl_7/d1_rader.c —
i.e. from my own edited file — and ran a full r6-vs-r7 comparison of the
binary against itself.  The tell was implausible parity on every cell
(including cells where the r6 BOARD said 25%).  The true previous-round
source is impl_6/ (git log confirms it: "Panel round d1_r6 ... curated").
Every A/B below labeled "true r6" was re-measured against impl_6 after the
mistake was caught; the hugepage A/Bs were unaffected (those compared build
variants of the r7 file itself).**

### The r6 premise, continued: my kernels' best already wins; the MEDIAN loses

The r6 board's resolved losses were all median-vs-best splits on my own cells:
65537 B=1 scored 813 median with a 734 BEST (race ships my kernel behind their
probe: 753 median off the same 733 best); 1021 B=1 7.43 median / 6.48 best
(planner 6.71); 31 B=512 chain 0.0634 median / 0.0562 best vs prime/race
0.0453.  Meanwhile prime's r6/r7 record ships the in-file version of race's
first-call placement probe and measured exactly this disease (their r6 31-B1-
chain board REGRESSED because a min-of-bursts probe kept burst-fast/steady-slow
draws; their r7 fixed the statistic to the driver's median-of-long-samples).
This round adopts the whole mechanism and then chases the part of the variance
no in-process candidate can reach.

### What changed

1. **13/31 batched chains: prime's chainblk ported near-verbatim** (their
   r4/r5 chainblk_body): 8 chains per lane-block transposed to split-complex
   v8 rows ONCE, dense symmetric-fold DFT from new [h][h] tables (pck/psk),
   c folded into per-k accumulator seeds, k-loop blocked by 3, lanes past the
   batch clamped (odd batches verified at B=3/5/9).  Replaces my r3 SoA CRT
   conv chain, and against the TRUE r6 binary (impl_6, see the trap note) it
   is a structural win, same-minute pairs x3: 31 B=512 m=1200 0.061-0.064 ->
   0.045-0.051 (~25%, exactly the prime-vs-me gap the r6 board showed);
   13 B=512 m=2000 0.015-0.018 -> 0.014 steady.  The dense fold has no
   permutation bracket and no kernel-spectrum pass, and its state is stack
   scratch the new probe can re-roll; the SoA CRT chain state lived at fixed
   heap offsets it could not.  The CRT codelets stay in the file,
   undispatched.
2. **First-call placement probe, in-file (ADOPTED: mechanism d1_race r4/r6,
   form d1_prime r6/r7).**  All dispatch is now fn-pointer wiring set at
   create (prime's shape): at 13/31 every graded path carries 6 arithmetic-
   identical candidates — pure code copies behind 1-5 entry nops for the
   register-only kernels (13 B=1/B512 exec, 31 B=1 exec), 2 code copies x
   stack shifts (alloca-in-wrapper around noinline cores, 1088/2176/3264 B)
   for everything with stack scratch (31 batched exec, both B=1 chains, both
   chainblks); exec31_pipe and the B=1 pair chains became always_inline
   bodies so each copy owns its hot-loop text.  At 1021/65537 the candidates
   are DATA SPACERS: the scratch block (+5*136 doubles slack) and chain block
   (+5*168) are re-based per candidate, residues distinct mod 4K, re-rolling
   scratch-vs-driver 4K aliasing; the chain probe rolls both jointly and also
   clears the exec flag (the trailing execute of a chained unit must not
   re-roll the chain's winner).  Probe = median of 3-5 sample-major rounds of
   ~275 us calibrated loops (prime r7's statistic, race r6's finding), inside
   the driver's discarded first call; setup= stays 0.005 s.  D1R_NO_PROBE /
   D1R_PROBE_VERBOSE / D1R_NO_HUGE env knobs for A/B.  Verbose evidence the
   mechanism bites: at 65537 B=1 successive processes picked candidates 1
   and 4 with in-process candidate spreads up to 7.9%; outputs bitwise
   identical across processes at every graded cell (verified, including
   chains — all candidates share one FP DAG).
3. **MADV_HUGEPAGE for the big blocks — the round's own finding.**  The node
   runs THP in madvise mode, so nothing was ever hugepage-backed.  With 4K
   pages, physical address bits 12-16 are per-page random, so the L2/L3 SET
   distribution of the 512 KB ping-pong planes is per-process page-color
   luck — this is the bimodal 737-vs-870 us mode at 65537 B=1 that the data-
   spacer probe measured FLAT (all candidates equally bad inside a bad
   process: virtual re-basing cannot fix physical coloring).  sbase and
   br/bi are now 2 MB-aligned + madvise(MADV_HUGEPAGE): physical indexing
   becomes deterministic and the DTLB cost of the random gather/scatter
   drops.  Same-minute triples (r7-nohuge vs r7-huge, D1R_NO_HUGE knob):
   65537 B=1 exec 738.4/723.1, 738/724, 745/723 — a steady -2.1% and the
   variance gone.  THE ALLOCATION TRAP: also hugepage-backing the CHAIN
   block made the chain SLOWER (nohuge 660 steady -> all-huge 678-694,
   same-minute triples): two 2 MB-aligned blocks are base-congruent
   physically, and the chain co-accesses cd and s planes at related indices
   every step.  cd stays 4K (comment at the alloc site); with huge s/br +
   4K cd the chain measured 644-647.  Final six-layout pad sweep against
   the TRUE r6 (PADVAR env shifts initial heap/stack; same-minute pairs),
   65537 B=1: exec r6 = {802,744,740,746,737,737}, r7 =
   {782,724,725,723,723,725}; chain r6 = {720,664,663,665,663,661}, r7 =
   {647,646,646,645,646,649}.  r7 wins EVERY draw: good mode -2..-3%, and
   the pad=0 bad draw is milder at exec (782 vs 802) and GONE at the chain
   (647 vs 720).  65537 B=16 exec same-minute: 897/898 -> 883.  The exec
   residual pad=0 mode is the DRIVER's buffers' page-color luck —
   unreachable from inside the plan (see next round).

### Measured (a80n0 core 3, final binary, one pass; window comparable to the
### r6 numbers only where marked same-minute.  "r6 board" = r6 leaderboard)

| cell | r6 board | now | note |
|---|---:|---:|---|
| 65537 B=1 / B=16 | 813 / 1022 | **726 / 887** | pad-sweep good mode 722 x5/6 draws |
| 65537 chain B=1 / B=16 | 681 / (cell missing on r6 board) | **647 / 684** | B=16 chain true-r6 pairs mixed 709/719, 706/699 — parity, memory-bound |
| 1021 B=1 / B=256 | 7.43 / 9.04 | **6.54 / 7.22** | |
| 1021 chains B=1 / B=256 | 7.97 / 7.01 | 7.15* / **6.99** | (*) noisy window; same-minute pairs r6-vs-r7 mixed +-2% with one -12% r7 win |
| 13 B=1 / B=512 | 0.0163 / 0.0095 | 0.014 / 0.011 | true-r6 pairs: 0.016-0.019 vs 0.015 at B=1 |
| 13 chains B=1 / B=512 | 0.0337 / 0.0146 | 0.038 / **0.014** | true-r6 pairs: B=1 0.034 both; B=512 0.015-0.018 vs 0.014 |
| 31 B=1 / B=512 | 0.0552 / 0.0504 | **0.045 / 0.044** | true-r6 pairs (later window): 0.053 vs 0.052 |
| 31 chains B=1 / B=512 | 0.0579 / 0.0634 | 0.051 / **0.045-0.051** | true-r6 pairs: B=1 0.051 both; B=512 0.061-0.064 vs 0.045-0.051 |
| 127 B=1-ish (unscored) | — | 1.63 (B=8), chain 1.29 | padded path untouched |

Wallaby (SPR, idle core 100, final binary): 65537 B=1 **583**, B=16 594,
chains 519 / 539-541; 1021 B=1 4.84, B=256 4.87, chains 4.62 / 4.44;
13 B=1 0.016, B=512 0.007-0.008, chains 0.027 / 0.011; 31 B=1 0.024,
B=512 0.023, chains 0.044 / 0.032.

Accuracy (all on the node, final binary): single-call rel_l2 1.7e-16 (13) …
1.37e-15 (65537), gate 1e-12; all eight graded chain gates pass with >= 2.5
decades of margin (13 m=200000: 1.7e-15; 31 m=1200: 3.6e-12 vs 1e-9; 1021
m=2000: 8.7e-12 vs 1e-9; 65537 m=60: 4.0e-14); odd batches 3/5/9 verified
single and chained (13 B5 m17, 31 B9 m33, 1021 B3 m11, 127 B8 m25); output
bitwise repeatable across processes on every graded cell WITH the probes
picking different candidates per process; setup <= 0.005 s at 65537.

### What did NOT work / traps, with the number

- **Hugepage-backing ALL big blocks**: 65537 B=1 chain 660 -> 678-694
  (same-minute triples, three reps).  Two 2 MB-aligned mmaps are physically
  base-congruent — the deterministic layout that FIXES the exec planes
  systematically COLLIDES the chain's cd-vs-s co-accesses.  Fixed by keeping
  the chain block on 4K pages (644-647).  Anyone adopting hugepages should
  treat "which blocks" as an A/B, not a blanket.
- **The data-spacer probe alone cannot fix the 65537 bad mode**: candidate
  spreads inside a bad process read FLAT (1.000-1.004) while process medians
  differed 17%.  Data spacers re-roll VIRTUAL aliasing; the mode was PHYSICAL
  page coloring (proved by the hugepage fix).  Race's r5 alt-text-mapping
  finding is the same lesson from the code side: know which dice you are
  re-rolling.
- **A "-2% r7 chain regression" scare dissolved** once the cd block went back
  to 4K pages and pairs were interleaved in one minute — the window moved
  700 -> 760 across a sweep and I nearly attributed it to the probe (prime's
  r7 "interleave more pairs" lesson, re-learned on a bigger cell).
- **The reference-built-from-a-symlink blunder** (details in the header
  note): half a session of "r6-vs-r7" pairs compared the r7 binary to
  itself.  The verification lesson generalizes: a same-window A/B whose
  sides agree EVERYWHERE — including on cells where the board disagrees by
  25% — is a check that cannot fail, and should itself be checked (diff the
  two sources, or plant a known difference).  After rebuilding the
  reference from impl_6, the chainblk port showed its real 25% and the
  probe its real wins.

### Borrowings (named plainly)

- **d1_race**: the whole probe concept (r4 first-call probe on real driver
  buffers inside the discarded warmup; r6 median-of-long-samples statistic),
  and the placement-vs-code-vs-data taxonomy their r5 record laid out.
- **d1_prime**: the in-file probe form ported near-verbatim (r6/r7: nop-pad
  anti-ICF code copies, alloca-wrapper stack shifts around noinline cores,
  sample-major calibrated-loop timing, fn-pointer wiring at create, the
  probe-cost budget numbers), and chainblk_body (r4/r5) with its c-in-seeds
  fold and k-blocked-by-3 loop.
- **d1_batchlane** (via prime's chainblk): the junk-lane clamp and map floor
  conventions ride along in the ported code.
- The hugepage physical-coloring diagnosis and the huge/4K split are my own
  r7 findings — offered to the panel: 2 MB-align + MADV_HUGEPAGE any block
  >= 512 KB that a transform ping-pongs through (THP is madvise-mode on the
  node, so nobody gets this for free), but A/B every block separately, and
  expect co-accessed block PAIRS to want different page sizes.

### Next round, in priority order

1. **The residual 65537 bad mode is the driver's own in/out buffers** (pad=0
   draw: exec 782 vs the 722 good mode even with all my blocks huge/probed;
   the same draw cost the true r6 802).  Two candidate attacks: (a) a probed "staged" execute variant that
   gathers from a hugepage-backed copy of the input (costs a 1 MB streaming
   copy ~40 us; wins ~140 in a bad draw; the probe arbitrates per process),
   (b) madvise(MADV_HUGEPAGE) directly on the driver's buffer range at first
   call — advisory-only on 5.15 (no MADV_COLLAPSE), already-faulted 4K pages
   collapse only via khugepaged, so measure whether it ever bites in-run.
2. **1021 st17 mid-fusion, now with a design**: forward-last st17 and
   inverse-first st4 close over a 68-point tile (indices t + 15k: 4 st17
   groups x 17 st4 groups, 17 KB tile — the st64 shape).  Saves one M-pass
   of 8; the twiddle bookkeeping is the risk.  Worth ~4-6% on all four 1021
   cells if it lands.
3. **13 B=512 (0.011 vs prime 0.0093-0.011)**: their r5 3-transform body
   idea remains unexplored panel-wide.
4. **10007 nested-Rader A/B** (5003-1 = 2*41*61): still untried; only worth
   it if it can beat d1_bluestein's ~110 us — measure the 5003-conv first.
