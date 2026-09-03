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
