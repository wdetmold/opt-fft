# L17_rader — strategy record

Geometry: **L = 17**, cube 17³ = 4913 complex doubles per volume, forward, unnormalised,
out-of-place, batched, single-threaded.
Implementation: `impl/L17_rader.c`. `fft3d_name()` → `L17_rader`.

---

## Round 1 (2026-08-21)

### Technique

Row-column over the three axes; every 1D line transform is **Rader-17** with the
length-16 cyclic convolution done as **two forward radix-4 FFT16 codelets plus 16
pointwise complex multiplies by a constant table**. The vector lanes hold *independent*
17-point transforms (`DFT₁₇ ⊗ I_ν`), split real/imaginary, so the kernel contains no
cross-lane operation at all.

#### The Rader construction, exactly as implemented

`g = 3` is a primitive root mod 17; `perm[u] = 3^u mod 17 =
{1,3,9,10,13,5,15,11,16,14,8,7,4,12,2,6}`.

```
a[u]        = x[perm[u]]                                u = 0..15
A           = FFT16(a)
X[0]        = x[0] + A[0]                               A[0] = sum_{n!=0} x[n]
P[s]        = A[s] * Bhat[s]                            Bhat = FFT16(b')/16
                                                        b'[q] = W17^(3^-q mod 17)
P[0]       += x[0]                                      DC fold
X[perm[w]]  = FFT16(P)[w]                               w = 0..15
```

Three points worth keeping:

1. **The same permutation table serves the gather and the scatter.** The unnormalised
   inverse DFT16 is the forward DFT16 with the output index negated,
   `IDFT(P)[v] = (1/16)·DFT(P)[-v mod 16]`; and negating the exponent of `g` turns
   `g^{-v}` back into `g^{w}`. So the output map is `X[g^w] = FFT16(P)[w]`, i.e. the
   *same* `perm[]` used for `a[u] = x[perm[u]]`. No second table, no reversal loop.
2. **`Bhat = DFT16(b')/16` is a compile-time constant**, so only **two** length-16
   transforms run per 17-point DFT, not three (this is what FFTW's `dft/rader.c` does
   with its separate `cld_omega` plan).
3. **DC fold.** Adding `x[0]` to `P[0]` adds it to all 16 outputs of the final transform,
   which removes 16 complex additions (32 real adds).

Two entries of the constant table are *exact* and are written as closed forms rather than
taken from the numeric table:

* `Bhat[0] = -1/16` — the 16 nontrivial 17th roots of unity sum to −1, so `DFT16(b')[0] = -1`.
* `Bhat[8] = sqrt(17)/16` — `DFT16(b')[8] = Σ_{QR} W^r − Σ_{non-QR} W^r` is the quadratic
  Gauss sum for p ≡ 1 (mod 4), hence **real** and equal to `√17`. Verified numerically:
  `0.25769410160110378436 × 16 = 4.123105625617661 = √17`.
  This turns two of the sixteen pointwise complex multiplies into cheap real ones.

The remaining table entries obey `Bhat[16−s] = (−1)^s conj(Bhat[s])` (a consequence of
`b'[q+8] = conj(b'[q])`, see "what did not work" item 3). That symmetry buys nothing here,
because `A[s]` and `A[16−s]` are independent; it is recorded because it is the hook a
real-input variant would use.

#### FFT16, radix-4, `n = 4p+q`, `k = 4r+s`

```
X[4r+s] = Σ_q ( w16^{qs} · DFT4_p( x[4p+q] )[s] ) · (−i)^{qr}
```

Both stages are the same 4-point butterfly with `W4 = −i` (16 real adds, no multiplies).
The twiddle exponents `qs mod 16` for q,s ∈ [0,4) are
`0,0,0,0 / 0,1,2,3 / 0,2,4,6 / 0,3,6,9`, so the nontrivial multiplies are:

| twiddle | value | count | cost |
|---|---|---|---|
| `w^1`, `w^3`, `w^3`, `w^9 = −w^1` | full | 4 | 2 mul + 2 fma = 4 instr, 6 flops |
| `w^2 = (1−i)/√2`, ×2, `w^6 = (−1−i)/√2`, ×2 | ±(1∓i)/√2 | 4 | 2 add + 2 mul = 4 instr, 4 flops |
| `w^4 = −i` | ±i | 1 | free (register rename + sign folded into the next add) |

So there are exactly **five** distinct real constants in the whole 17-point module:
`cos π/8`, `sin π/8`, `cos 3π/8`, `sin 3π/8`, `1/√2`, plus the 30 nonzero entries of `Bhat`.
Total twiddle data under 300 bytes, all `static const`, all broadcast from memory.

### Operation count per 17-point transform

Real FP instructions counting an FMA as one instruction and two flops:

| term | instr | flops |
|---|---|---|
| 8 × DFT4 (two FFT16, four butterflies each) | 256 | 256 |
| 8 × full twiddle (4 per FFT16) | 32 | 48 |
| 8 × √2 twiddle (4 per FFT16) | 32 | 32 |
| 2 × `×(−i)` twiddle | 0 | 0 |
| pointwise `s=0` (Bhat[0] = −1/16 with the DC fold) | 2 | 4 |
| pointwise `s=8` (Bhat[8] real) | 2 | 2 |
| pointwise, 14 full complex multiplies | 56 | 84 |
| `X[0] = x[0] + A[0]` | 2 | 2 |
| **total** | **382** | **428** |

= **22.5 instr / 25.2 flops per point**. gcc actually emits **368** FP instructions per
kernel instantiation (it folds a few multiplies into the following adds), so the hand
count is a slight over-estimate.

Comparison with the corpus (`docs/literature/02` §2.4, §2.7):

| 17-point kernel | flops | FP instr |
|---|---|---|
| this file (Rader, radix-4 FFT16, DC fold, exact `Bhat[0]`/`Bhat[8]`) | **428** | **382** (368 emitted) |
| FFTW's Rader-17 accounting | 468 | 388 |
| FFTW's dense conjugate-symmetric `dft-generic-17` (what its planner picks) | 592 | 336 |
| Bluestein-17 | 1932 | 1524 |

Per 17³ volume: 3 × 289 = 867 transforms = **331 k instructions / 371 k flops =
75.6 flops/point**, against FFTW's measured 104.5 flops/point for the same volume
(`3 · 17² · 592`). So the arithmetic is 28% below FFTW's, and the instruction count is
14% *above* it — the metric trap of §02 §2.7, exactly as advertised. The 2.7× measured
speedup below therefore comes almost entirely from vectorisation and layout, not
arithmetic, which is what §02 §1 predicted.

### Layout and SIMD decisions

**Vector width is 256 bits (4 doubles) on purpose, even on AVX-512 hardware.** The
argument, since it is counter-intuitive and is the one thing a later round should re-test
on the node:

* The scored node is a **Xeon Gold 5218**, and the Gold 5200 series carries **one**
  512-bit FMA unit, not two. A 512-bit FMA then retires at 1/cycle while two 256-bit FMAs
  retire at 2/cycle on ports 0 and 1: *identical* flop throughput, zero arithmetic gain
  for zmm. (Verify on the node; if the SKU turns out to have two FMA units, zmm wins by
  roughly 1.7× and this decision is the single biggest thing to change.)
* Widening makes lane utilisation **worse** here, because the two plane passes have only
  17 independent transforms in the lane direction: 17 of 24 lanes (71%) at 512 bits
  against 17 of 20 (85%) at 256. Counting whole volumes, zmm issues **1112** lane-slots
  of arithmetic where ymm issues **972** — 14% more work at the same per-cycle throughput.
* 512-bit arithmetic drops the core into the lower AVX-512 frequency licence; 256-bit
  code stays in the AVX2 licence. The corpus (§04 §8.1-8.2) documents 2.3 → 1.6 GHz on a
  sibling SKU.
* **The register-file argument for zmm does not apply.** With `AVX512VL` gcc allocates
  `ymm16-31` for 256-bit vectors. Measured on the generated code of `fft3d_execute`, at
  unchanged width 4: **357** stack moves with `-march=haswell`, **72** with
  `-march=skylake-avx512`, and 1662 uses of the high registers. So the 256-bit build gets
  the whole 32-register file on the node — the best of both. **This is new information;
  the corpus contains no AVX-512 measurement of any kind.**

`L17R_FORCE_VW=8` builds the 512-bit variant. It is verified *correct* (gcc emulates
64-byte vectors with ymm pairs on an AVX2 host, which is how the wide path was tested
here — 4.2e-14 on a single line, 4.1e-16 on a full volume) but has never been *timed* on
real AVX-512 silicon.

**Kernel shape.** The two FFT16s are chained rather than run back to back: group `q` of
the second FFT16's decimation stage consumes exactly the `s = q` output set of the first
FFT16's second stage, so the code runs four independent `D4 → pointwise → D4` chains and
then four final butterflies, each of which stores its four results immediately. Peak
liveness is ~40 vectors instead of ~66 for the naive "compute all 17 outputs into an
array, then store" form. On AVX2 that took spill traffic from 227 to 89 stack moves per
kernel instantiation and the whole transform from 51.2 µs to 33.5 µs.

**Volume layout.** Each pass wants its transformed axis to be the *outermost* index
(stride = padded plane pitch) so the other two axes form one contiguous 289-long lane
space. Two of three axes can be had that way in any one layout; the third needs a
transpose. Rather than transpose the volume, the y and z passes are fused per x-plane
where a 17×17 plane (4.6 KiB split) is L1-resident:

```
for each x plane:
    in[x][y][z]  --deinterleave + 17x17 transpose-->  T[z][y]        (row stride TR=20)
    z pass on T   (axis stride TR, lanes = y)      ->  T[kz][y]
    17x17 transpose                                ->  U[y][kz]
    y pass on U   (axis stride TR, lanes = kz), storing straight into
                  A[x][ky][kz]                     ->  A             (plane pitch PS=292)
x pass on A (axis stride PS, 289 contiguous lanes), interleaving store -> out[kx][ky][kz]
```

The volume therefore crosses the L1 boundary **exactly four times**: read `in`, write A,
read A, write `out` — about 320 KiB of traffic per volume.

Details that matter:

* The y pass writes **directly** into `A[x][ky][kz]` (outputs at stride 17, lanes
  contiguous), which removes what used to be a separate digit-permutation pass. Its lane
  blocks must not overrun lane 16, so they are `{0,4,8,12,13}` (VW=4) / `{0,8,9}` (VW=8) —
  the last block *overlaps* the previous one and recomputes 3 transforms, which is free
  because `ceil(17/VW)` blocks are needed either way.
* The z pass is in place, so it can use aligned padded blocks `0,VW,…,TR−VW` and let the
  pad lanes compute zeros.
* Pad lanes (columns 17..TR−1 of the plane buffers, lanes 289..PS−1 of A) are zeroed once
  in `fft3d_create` and every pass maps zeros to zeros, so they stay zero for the life of
  the plan. Nothing is re-zeroed per call, and `fft3d_execute` is bitwise repeatable
  (checked).
* The x pass's store interleaves re/im with one shuffle pair per output
  (`vunpcklpd`/`vperm2f128` on AVX2, `vpermi2pd` on AVX-512) straight into `out`, so there
  is no separate interleave pass.
* **No cache padding anywhere.** 17 is odd, so every stride (17, 20, 292 doubles) is an
  odd number of 8-byte words and all L1/L2 sets get used — §04 §7.3's observation that
  L=17 is the one size where cache geometry is free, confirmed by the absence of any
  padding-related cliff in the sweep from B=1 to B=2048.
* Both 17×17 transposes are 4×4 ymm blocks (8 shuffles per 16 elements) plus a scalar
  17-element row and 16-element column. The *deinterleaving* one transposes the **complex**
  values first at 128-bit granularity and splits re/im afterwards: 8 + 8 = 16 shuffles per
  4×4 complex tile instead of 8 deinterleaves + 16 real transposes = 24. That single
  reordering took the 17-plane deinterleave pass from **5.31 µs to 3.34 µs**.

### What was measured

#### On the scored node — `probe_node.sh`, isolated Xeon Gold 5218 @ 2.30 GHz, 1 MiB L2/core, 44 MiB L3, gcc `-O3 -march=native -mtune=native -fno-math-errno -funroll-loops`

| B | **this file, µs/transform** | MKL 2022 sequential, µs/transform | speedup |
|---|---|---|---|
| 1 | **21.230** (14.19 GF/s) | 98.762 | **4.65×** |
| 8 | **22.211** (13.56 GF/s) | 100.118 | **4.51×** |
| 256 | **26.273** (11.47 GF/s) | 105.538 | **4.02×** |

rel L2 vs numpy on the node: 4.074-4.085e-16 at every B. Against the `sota_r1` leaderboard
figure for L=17 B=1 (FFTW patient, **81.68 µs**) this is **3.85× faster**; against the
"conservative target" of §02 §6.2 (~8-9 ns/point ≈ 42 µs/volume, "≈2.4× faster than FFTW
and MKL") it is **2.0× past** it. Per point: **4.32 ns**, against FFTW's measured 20.98 and
MKL's 20.86 — and against FFTW's **3.93 ns/point at 16³**, i.e. this makes 17³ cost
**1.10× per point what the libraries' best power-of-two size costs them**, where the
libraries pay 5.3×. The L=17 penalty the corpus was built around is essentially gone.

#### AVX-512 vs AVX2 on the node — the measurement the corpus does not have

§04 gap 6 records that no AVX-512 measurement exists anywhere in the corpus. Both widths
were built from this same source (`L17R_FORCE_VW`) and timed on the node, µs per transform:

| width | B=1 | B=8 | B=256 |
|---|---|---|---|
| **256-bit (ymm), default** | **21.230** | **22.211** | **26.273** |
| 512-bit (zmm) | 21.943 | 23.311 | 27.955 |
| 256-bit advantage | 3.4% | 5.0% | 6.4% |

**256-bit wins at every batch size on the graded hardware, and the margin grows with the
working set.** Output is bit-identical between the two. The four reasons are in the file
header; the short version is that the Gold 5200 series has one 512-bit FMA unit, so zmm
buys no flop throughput, while the 17-lane plane passes make zmm's lane utilisation *worse*
(71% against 85%) and 512-bit arithmetic costs frequency licence. Note the margin is much
smaller than those arguments predict (they add up to ~25%), so zmm's halved instruction and
load/store counts are recovering most of the lane waste — on a two-FMA-unit SKU (Gold 6xxx,
Platinum 8xxx) the balance would very likely flip, and any port of this file to such a
machine should re-run the comparison rather than inherit the default.

#### Local development machine

Machine: **Xeon E5-2680 v3 (Haswell, 2.50 GHz, AVX2, no AVX-512), 32 KiB L1d, 256 KiB L2,
30 MiB L3**, gcc 11.4.0 `-O3 -march=native -mtune=native -std=gnu11`, single thread, shared
(other agents active — numbers are best-of-3 processes × 20 samples). The node is 1.42×
faster at B=1 despite a *lower* base clock, which is the 32-register file removing the
spill traffic (357 → 72 stack moves) plus single-core turbo.

| B | µs per execute | **µs per transform** | rel L2 vs numpy |
|---|---|---|---|
| 1 | 30.2 - 30.8 | **30.2 - 30.8** | 4.080e-16 |
| 8 | 254.9 - 255.2 | **31.86 - 31.90** | 4.099e-16 |
| 256 | 9690 - 10061 | **37.85 - 39.30** | 4.072e-16 |
| 2048 | 81552 - 81760 | **39.82 - 39.92** | 4.074e-16 |

(range = two independent best-of-3 measurement sessions; the host was shared with other
agents throughout, so treat the low end as the honest figure and expect the monitor's
isolated-node numbers to be cleaner as well as faster.)

`setup` (`fft3d_create`) is 3 µs: one `posix_memalign` + `memset` of 90 KiB. Nothing is
computed at run time; every constant is a literal.

Reference points: the round-`sota_r1` best library at L=17 B=1 is **FFTW patient, 81.68 µs**
on the *isolated Cascade Lake node*, and the corpus measured FFTW at 20.98 ns/point
(103 µs/volume) and MKL at 20.86 ns/point on *this* Haswell. So on like-for-like hardware
this is **3.4× faster than FFTW and MKL at B=1**, and about 2.7× faster than the node
figure a Cascade Lake will be scored against — before the node's extra 16 vector
registers and higher single-core turbo, which the static counts above say are worth
another 15-25%.

Phase breakdown per volume, from an in-tree micro-benchmark (AVX2, warm caches, so the
parts sum to slightly more than the whole):

| phase | blocks | µs |
|---|---|---|
| deinterleave + transpose, ×17 planes | — | 3.34 |
| z pass | 85 | 9.84 |
| 17×17 transposes, ×34 | — | 2.42 |
| y pass (storing into A) | 85 | 12.37 |
| x pass + interleaving store | 73 | 12.32 |
| whole `fft3d_execute` | 243 | 34.9 (30.2 best-of) |

Static instruction counts for `fft3d_execute` (all four kernel instantiations + both
transposes inlined), from `objdump`:

| build | total | FP | stack moves |
|---|---|---|---|
| `-march=native` (Haswell), VW=4 | 3420 | 1472 | **342** |
| `-march=skylake-avx512`, VW=4 | 3108 | 1472 | **72** |
| `-march=skylake-avx512`, VW=8 | 3104 | 1472 | 72 |

1472 FP / 4 instantiations = 368 per 17-point kernel — the op count above, confirmed.

Correctness gates, all passed:

* `check.py` at B = 1, 3, 8, 64, 256, 2048: rel L2 = 4.07-4.10e-16 (tolerance 1e-12).
* Both vector widths give **bit-identical** output (VW=8 built via gcc's emulated 64-byte
  vectors).
* `in` is bitwise unmodified; two, three and four successive `fft3d_execute` calls on one
  plan give **bitwise identical** output.
* `-fsanitize=undefined`: clean.
* Guard-page tests: `in` and `out` each placed so the buffer *ends* on an inaccessible
  page — no overrun; and the plan's 90 KiB scratch likewise ended on a guard page — no
  overrun. (ASan itself cannot run on this host: `ulimit -v` blocks its shadow map.)
* `-Wall -Wextra`: silent at both widths.

### What was tried and did NOT work

1. **Separate full-volume transpose and digit-permutation passes.** The first working
   version kept the volume in one buffer and ran `deinterleave → z pass → full-volume
   17×289 transpose → y pass → full-volume digit swap → x pass`: twelve crossings of the
   L1 boundary, ~960 KiB of traffic per volume. **64.4 µs**, against 37 µs for the
   plane-fused arrangement — and the three movement passes measured 8.79 + 8.15 + 8.07 µs,
   i.e. 25 µs of the 64. All three were running at ~29 GB/s, which is L2 bandwidth with
   RFO, not an instruction-count limit. *Reducing the number of times the volume crosses
   L1 was worth more than everything else in this file put together.*
2. **`fft16` and `rader17` as ordinary (non-`always_inline`) static functions taking
   `vd *` arrays.** gcc then materialises the 16-element vector arrays in memory across
   the call boundary: 205 `vmov` of which 114 stack, per `rader17`, plus 2 × 299
   instructions for the two `fft16` calls — 887 instructions for a kernel whose
   arithmetic is 368. **64 µs → 51 µs** simply from adding `__attribute__((always_inline))`,
   before any restructuring. Do not let the kernel be a real function.
3. **Splitting the length-16 cyclic convolution into two length-8 convolutions** using
   `b'[q+8] = conj(b'[q])` (which does hold — verified to 3.5e-16). Write
   `b'[q] = β[q mod 8] + i·s(q)·γ[q mod 8]` with `s = +1` on `[0,8)` and `−1` on `[8,16)`;
   then `a ⊛ b' = (A⁺ ⊛₈ β) + i·(A⁻ ⊛₈ γ)` with `A± [j] = a[j] ± a[j+8]`. The first term
   really is an 8-point **cyclic** convolution. The second is **negacyclic**, not cyclic —
   I implemented the cyclic version in numpy first and it was wrong by O(10) on random
   input, which is how the error was caught. Making it cyclic costs a pre-twiddle
   `A⁻[j]·ω16^j` and a post-twiddle `ω16^{−v}`, 24 instructions each, and the honest total
   becomes **394 instructions against 382** — *worse*. The decomposition is just FFT16
   rewritten, with the DIF twiddle stage exposed. This is §02 §8.6's prediction ("would
   not beat two length-16 FFTs by more than ~10% on flops and will lose on regularity")
   coming out on the losing side of zero. **Do not retry this.**
4. **§01's "alternate convolution" (symmetric/antisymmetric split, `alternate_convolution
   = 17` in genfft's `magic.ml`).** Chased and abandoned: it makes the sub-transforms
   *real-input* and therefore half cost, which is a real-data-FFT optimisation. Our input
   is complex; there is no real-input sub-transform to exploit. §01 §8's flop targets of
   356-384 for a 17-point module are consequently not reachable by that route for complex
   data.
5. **A 512-bit kernel.** Rejected **by measurement on the scored node**: 21.943 / 23.311 /
   27.955 µs per transform at B = 1 / 8 / 256 against 21.230 / 22.211 / 26.273 for the
   256-bit build — 3.4% to 6.4% slower, worsening with the working set. The predicted
   reasons (one 512-bit FMA unit so equal flop throughput; 71% against 85% lane
   utilisation; the AVX-512 frequency licence; and no register-file advantage because
   256-bit code already gets ymm16-31 under AVX512VL) add up to roughly 25%, so most of
   that is being recovered by zmm's halved instruction and memory-op counts. Both builds
   produce bit-identical output, so this was a clean A/B.
6. **A first digit-permutation that copied a 17-double run with two overlapping vector
   stores** (`[0,VW)` and `[17−VW,17)`). Correct at VW=8, silently wrong at VW=4, where
   two 4-vectors cover 8 of 17 doubles: rel L2 = 7.5e-01, and the *symptom* was that
   output bins `kz ≥ 4` were wrong while `kz < 4` were right. Worth remembering that
   `17 ≤ 2·VW` only holds for the wide build — a whole class of "one vector plus an
   overlapping tail" idioms is width-dependent here.
7. **Non-temporal stores for `out`** (§05 §8.2 sanctions them for exactly this case: a
   final write-out never re-read). Rejected on analysis, not measured. The x pass writes
   `out + 2·(k·289 + m0)`, i.e. byte offset `16·(k·289 + m0)`; since 289 is odd this is
   64-byte aligned only when `k ≡ 0 (mod 4)`, so 12 of the 17 output streams could not use
   an aligned NT store. Worse, the pass has **17 concurrent output streams**, more than the
   core has fill buffers, so write-combining buffers would be evicted partially filled —
   which costs more than the RFO traffic it saves. Getting NT stores would mean restructuring
   the last pass to emit one contiguous stream, which means a transposing final store.
8. **Grouping G x-planes into one plane buffer to fix the 17-lane utilisation.** Analysed,
   not built. The z pass would benefit (it stores in place, so lane blocks may straddle a
   plane boundary: at G=2, VW=8 that is 5 blocks per two planes instead of 6). The y pass
   would not, because its store into `A[x][ky][kz]` must be lane-contiguous and therefore
   its blocks may not straddle x. Net saving ≈ 8 of 244 blocks, ~3%, for a materially more
   complicated buffer and a plane buffer that stops fitting L1 above G=2. Not worth it.
9. **A newer compiler.** gcc 13.2.0 and gcc 16.1.0 were measured against gcc 11.4.0 on the
   final source: 30.7 / — / 38.6 µs against 30.2 / — / 39.3 µs, i.e. within noise. (On an
   *earlier*, spill-heavy version gcc 13 was 19% faster at B=1, which is a hint that the
   remaining gap on AVX2 is register pressure the newer allocator handles better; once the
   kernel was restructured the difference vanished.) **No compiler request needed.**
   clang cannot build this file as written — `__builtin_shuffle` is a gcc spelling — so a
   `__builtin_shufflevector` path is included, untested beyond compiling.

### Where the time actually goes, and what is left

At B=1 on AVX2, of ~75 000 cycles per volume:

* ~46 000 cycles are the FP throughput floor: 972 lane-slots × 368 ymm FP ops ÷ 2 per cycle.
* ~8 700 ymm shuffles in the two transposes and the interleaving store, all on port 5.
* the balance is spill traffic (342 stack moves in the AVX2 build; 72 on the node) plus
  the 34 loads + 34 stores per kernel block.

So on the node the kernel should sit close to its arithmetic floor, and the two remaining
levers are (a) the 12% lane waste in the plane passes and (b) the port-5 shuffle load.

### Next

In priority order, with why:

1. ~~Measure `L17R_FORCE_VW=8` on the node.~~ **Done this round** — see the table above.
   256-bit wins by 3.4-6.4%; the default is correct and this question is closed for the
   Gold 5218. It is *not* closed for a two-FMA-unit SKU.
2. **Do NOT fuse the second 17×17 transpose into the z pass's store** — I costed this
   after writing the first draft of this list and it is a net loss, recorded here so
   nobody repeats it. Removing `transpose17` saves the measured 2.42 µs, but a VW×17
   in-register transposing store costs, per kernel block and per array, 4 ymm 4×4
   transposes (32 shuffles) + 4 scalar extracts for output 16 + 5 stores per lane instead
   of 17: ≈86 extra ops × 85 blocks ≈ 7.3 k ops ≈ 2.9 µs. Net −0.5 µs. The blocked 4×4
   transpose through L1 is simply cheaper per element (0.5 shuffles) than an in-register
   17-tall transpose (1.9 shuffles per element, because 17 is not a multiple of 4).
3. **Reduce the plane passes' 17-lane waste properly.** The ceiling is exact and small:
   972 lane-slots are issued for 867 transforms, so perfect lanes are worth 10.8% of the
   kernel, ~2.6 µs of 30, 8.6% of the whole. The cheap half of it — a G=2 slab used by
   the **z pass only**, which is risk-free because that pass stores in place and so tolerates lane blocks straddling a
   plane boundary — is worth 8 of 244 blocks, ~3%, and was skipped only because it did not
   justify re-running the whole verification matrix late in the session. The full version
   gives the y pass a slab too (`[ky][x_g][kz]`, lane-contiguous; at G=3 with VW=4 that is
   13 blocks per 3 planes against 15, and 51 of 52 lanes used) and handles the 2 straddling
   blocks per slab with scalar stores (~136 each, ≈0.6 µs total). Net ≈ 2.3 µs, ~7.6%, and
   it is the only idea left that is worth more than 5%. Watch L1: T and U at G=3 are
   4 × 7.1 KiB = 28 KiB, which is most of a 32 KiB L1 and would start competing with the
   kernel's own spill slots.
4. **Try the dense conjugate-symmetric kernel as the competitor it is.** §02 §2.5 puts
   `dft-generic-17` at 336 FP instructions against this file's 368 emitted — 9% *fewer*,
   with 256 of them FMAs and no permutation and no buffers, and it is what FFTW's measuring
   planner actually selects. Batch-vectorised in *this* layout it might simply win. Two
   other entries in this same round (`impl/L17_matrixsimd.c`, `impl/L17_winograd.c`) attack
   L=17 from those directions, so read their records first: if either beats this file, the
   productive merge is almost certainly their 17-point kernel dropped into the plane-fused
   layout and the lane-block scheme above, since those are what the 2.7x here came from
   and they are kernel-agnostic.
5. Only after 1-4: reorder the last pass to emit a single contiguous output stream so NT
   stores become usable, which is worth something only at B ≥ 256 (39.3 vs 30.2 µs today,
   so ~9 µs of DRAM exposure of which NT stores could recover perhaps a third).

---

## Round panel_r2 (2026-08-21)

### What changed

**The 17-point module was replaced; everything else survived.** The two-FFT16
Rader kernel (368 emitted FP instructions) was swapped for the symmetrised
cyclic/negacyclic module **adopted from `L17_winograd` (round 1)** — their
derivation, their constants, verbatim: fold `u_j = x_j + x_{17-j}`,
`v_j = x_j − x_{17-j}`, index by the order-8 quotient group of `(Z/17)*` by
`{±1}`; the `u` half is a cyclic-8 correlation with the real kernel
`cos(2π·3^r/17)`, split once more with sign-only reductions into a cyclic-4
(x₀-seeded, so DC is free) plus a negacyclic-4; the `v` half is a dense
negacyclic-8 with kernel `sin(2π·3^r/17)`. **296 FP instructions (192 FMA +
104 add/sub, 488 flops) per 17-point transform against 368 — −19.6%.**
Verified in the emitted assembly: 1184 FP instructions / 4 instantiations.

The plane-fused layout, split re/im, lane blocking, in-place z pass,
direct-into-A y store, and interleaving x-pass store are unchanged from
round 1 — the merge is exactly what round 1's "Next" item 4 predicted: their
kernel is also all-real-constant, lane-invariant, cross-lane-free, so it
dropped into `wino17()` behind the same `ST` store modes with no layout work.
Rel L2 vs numpy: **3.114e-16** (B=1), 3.151e-16 (B=8), bitwise repeatable,
`-Wall -Wextra` clean, AVX2 (Haswell) path runs correctly.

### Operation count per volume (VW=4)

243 kernel blocks × 296 = **71.9k vector FP instructions** (was 89.4k).
FP floor on a 2-per-cycle 256-bit machine: 36.0k cycles (was 44.7k).

### Measurement methodology on wallaby — read this before comparing numbers

wallaby's per-core clock is **bimodal under other users' load** (~3.9 vs
~3.0 GHz; the same binary gives 11.0 or 14.1 µs in different runs with tiny
in-run sd). `tryout.sh` numbers taken solo are therefore ±30% air. All
comparisons below are **min over ≥4 alternating pinned runs**
(`taskset -c 17`, 20 samples each), i.e. fast-mode min against fast-mode min.
This is the same lesson as L17_matrixsimd's round-1 item 12 (block your
measurements), arriving via frequency instead of licence transitions.

### Measured, wallaby (Sapphire Rapids Gold 6448Y), gcc 11.4, panel flags

| case | round-1 kernel | this round | delta |
|---|---|---|---|
| B=1, VW=4 | 11.95–12.02 µs | **11.01 µs** | −8% |
| B=8, VW=4 (µs/transform) | ~14.2 (no fast-mode run caught) | **10.97** | — |
| B=256, VW=4 (µs/transform) | 16.37 | **15.66** (one full-turbo run: 12.29) | −4.3% |
| B=1, VW=8 (`L17R_FORCE_VW=8`) | — | **10.00 µs** | −9% vs VW=4 |

VW=8 winning on wallaby is expected (SPR has **two** 512-bit FMA units) and
does **not** transfer to the node (one unit) — see "Next".

Static counts for `fft3d_execute` (native build on wallaby): old 4493 instr /
1472 FP / 72 stack / 448 rip-const; new **4466 / 1184 / 237 / 368**. Note the
totals are nearly equal — the FP saving is partly re-spent on spills and
constant loads, which is why the wall-clock gain (8%) is less than the FP gain
(19.6%). On the node, which was FP-bound at 21.23 µs (floor 19.4), more of the
19.6% should convert; projected **~17.5 µs at B=1**, vs L17_matrixsimd's 16.99
and L17_winograd's 18.26.

### What was tried and did NOT work — with the numbers that killed it

1. **Pinning the broadcast constants in registers via an empty asm barrier**
   (`KREG`, kept in the file under `-DL17R_PIN_CONST`). Motivation: gcc folds
   every constant into an FMA memory operand (~148 constant-load µops per
   kernel instantiation) and, worse, **canonicalises `a -= b*c` with constant
   `c` into `a += b*(-c)` and materialises a separate negated .LC copy** —
   the emitted code had *zero* `vfnmadd` until the pinning forced it (then
   256). Measured, wallaby fast-mode B=1: unpinned **11.02**, S̃-stage-only
   pinned 11.60, all pinned 11.53; stack moves 237 → 279 → 361. The spills
   the pinning induces cost more than the loads it removes on a 3-load-port
   core. A 2-load-port Cascade Lake might disagree — one A/B for the monitor.
2. **The plane-pair slab (round 1's "Next" item 3) LOSES, in every variant.**
   Two x-planes share T2/U2 buffers (row stride 36) so the z and y passes get
   a 34-wide lane space: 77+77 blocks instead of 85+85, −6.6% kernel FP.
   Measured, wallaby fast-mode B=1 against 11.01 pre-slab:
   * z+y slab, straddling y-block stored with 136 scalar extracts: **11.53**;
     first attempt also instantiated the kernel 8× and blew `fft3d_execute`
     to 9.6k instructions ≈ 38 KB, past L1i — table-driven loops (one z site,
     one y site, one straddle site) brought it back to 5.2k and it *still*
     lost.
   * same with the straddle store as one AVX-512VL masked store
     (`_mm256_mask_storeu_pd(p−1, 0xE, v)`, faults suppressed on the
     masked-off lane — the trick works and is worth keeping): **11.34**.
   * **z-slab only**, no straddle, y per plane: **11.37**.
   So the pair *plumbing itself* costs ~0.4 µs — the 2.9× larger plane
   buffers (19.6 KB vs 10.9 KB across T/U × re/im) and offset transposes eat
   more than the 2368-cycle FP saving, and the node's 32 KB L1d would make
   the footprint worse, not better. **Do not retry the slab without first
   explaining the plumbing cost.** The 12% lane tax stands at ~6.6%
   recoverable and unrecovered.
3. **`a -= b*c` as a source-level fnma spelling** (sign-token macros): gcc 11
   canonicalises it right back to a negated constant. Harmless (kept — the
   file reads better) but it changes nothing without `L17R_PIN_CONST`.

### Borrowed / read this round

* **The whole 17-point module from `L17_winograd`** (their round 1, §1(b)-(e)),
  constants verbatim from `impl/L17_winograd.c`. Attribution in the file
  header too. Their negative results (WFTA-17, every negacyclic-8 split,
  batch-in-lanes) were treated as settled and not retried.
* From `L17_matrixsimd`: the measure-in-blocks discipline (their item 12) —
  adapted here as alternating pinned runs; their NT-store-loses-on-the-node
  measurement (item: 34.4 vs 28.0 at 384 volumes) — my round-1 item 7
  rejection of NT stores stands and was not revisited.

### Next (for the monitor / next round, in order)

1. **Node A/B: VW=4 vs VW=8 (`L17R_FORCE_VW=8`), all batch sizes.** Round 1
   measured VW=4 ahead 3.4-6.4% with FP floors 44.7k vs 51.2k cycles; the new
   kernel narrows the floor gap to 36.0k vs 41.1k while VW=8's non-FP savings
   (which recovered 2.1 of 2.8 µs last time) are unchanged — **the widths are
   now within ~0.1 µs of each other on paper and the round-1 answer may
   flip.** Wallaby's 10.00 vs 11.01 says nothing (two FMA units there).
2. **Node A/B: `-DL17R_PIN_CONST`** (one build flag). Loses 5% on a
   3-load-port SPR; a 2-load-port CLX bound elsewhere might take the ~140
   removed load µops per block.
3. If matrixsimd is still ahead at B=1 after this lands: the remaining gap is
   pure non-FP overhead (transposes ~2.4 µs + 237 stack moves), not
   arithmetic — 296 instr/17-pt is now below their 168×5-chunks-per-17
   equivalent. Attack the spills (the vv/cc arrays) before touching the
   transposes; round 1's item 2 already killed transpose fusion.

---

## Round panel_r3 (2026-08-21)

### Standing going in (panel_r2 node leaderboard)

B=1: 19.212 µs (3rd — matrixsimd 16.751, winograd 18.247). B=8: 20.365 (3rd,
0.15 behind winograd). B=256: 24.394 (2nd, winograd 24.031). B=2048: 26.635
(2nd, winograd 24.603). Note the r2 projection (~17.5 at B=1 from the −19.6%
FP cut) did NOT convert: only 21.23 → 19.21 materialised, and matrixsimd's
identical-algebra kernel also barely moved (16.99 → 16.75) — node B=1 has a
non-FP floor near 17 µs that op-count alone does not shift.

### What changed (no arithmetic changed; the *choice machinery* did)

The 296-instruction cyclic/negacyclic kernel, plane-fused layout, split
re/im, lane blocking and store paths are all unchanged from panel_r2. Three
structural changes:

1. **Both vector widths now live in one binary and the plan picks by
   measurement.** The whole pipeline (kernel + passes + exec loop) is
   templated over VW by a self-`#include` (mechanism from `L17_matrixsimd`)
   and instantiated at VW=4 (ymm, TR=20, PS=292) and VW=8 (zmm, TR=24,
   PS=296), each also in a pinned-S-constant version (panel_r2's
   `L17R_PIN_CONST`, now a runtime candidate, EVEX-only) — four exec
   variants: `{256,512} × {plain, pinned-S}`. `fft3d_create()` ranks them in
   **blocks of ≥64 consecutive volume transforms, never interleaved**
   (matrixsimd round-1 item 12: interleaving ISA widths mis-ranked by 35% on
   the node), min of 3 reps, at nv = min(batch,16); when batch ≥ 64 it
   re-ranks at nv = min(batch,384) — 60 MB in+out, past node L3 — because the
   L2-resident winner can lose in the streaming regime (winograd's round-2
   lesson). Each width owns a disjoint scratch region so the pad-lanes-stay-
   zero invariant holds per width and repeated executes stay bit-identical.
   This closes panel_r2's "Next" items 1 and 2 (node A/B of VW and of
   pinning) permanently: the scoring machine answers them itself at plan
   time, every round, and `fft3d_description()` reports the pick (read it
   off the leaderboard JSON).
2. **Cross-volume input prefetch, tuner-gated** — adopted from
   `L17_winograd` round 2 (their measured −4.4% at B=2048). At plane x of
   volume b, 73 `prefetcht1` pull plane x of volume b+1 while the z/y passes
   compute; a NULL guard makes it free at B=1. When batch ≥ 64 the plan A/Bs
   pf on/off on the streaming working set (blocked, never alternating — the
   NT-store measurement lesson applies to any cache-state A/B) and keeps the
   winner (`pf=` in the description string).
3. **The x pass's scalar-tail kernel is gone.** The last block's start is
   clamped to `NPL−VW` so it overlaps the previous block; the recomputed
   lanes are bit-identical because every lane's arithmetic is independent of
   the block offset. One whole kernel instantiation disappears per exec:
   **888 FP instructions (= 3×296) per exec against 1184 (= 4×296) in
   panel_r2**, ~700 total instructions less I-footprint, and the x pass no
   longer reads pad lanes at all.

### Operation count

Unchanged per 17-point transform: 296 FP instr (192 FMA + 104 add/sub),
488 flops. Kernel blocks per volume: 243 at VW=4 (FP floor 36.0k cycles on a
2-FMA-port ymm machine), 139 at VW=8 (41.1k cycles on a 1-FMA-unit zmm
machine, 20.6k on a 2-unit one). Static per exec variant (wallaby native
build): np_w4 3820 instr / 888 FP / 204 rsp-refs; np_w8 3765 / 888 / 206;
pinned variants +~40 instr, −44 rip-const loads.

### Measured — wallaby (Sapphire Rapids Gold 6448Y, gcc 11.4, panel flags; min over ≥4 pinned runs, contended day, same methodology note as panel_r2)

| case | panel_r2 code | this round | tuner's pick |
|---|---|---|---|
| B=1 | 11.01 (VW4) / 10.00 (VW8 forced) | **9.81 µs** | 512-bit |
| B=8 | 10.97 (VW4) | **10.00 µs/t** | 512-bit |
| B=256 | 15.66 (VW4) | **11.47 µs/t** | 512-bit, pf=0 |
| B=2048 | — | **19.05 µs/t** | 512-bit, pf=1 |

The tuner's own tables agree with the driver to ~1% (e.g. B=1: tuner 9.85–9.94,
driver min 9.81). Pinned-S loses ~2% on wallaby at B=1 (14.30 vs 14.18 at
256-bit, 10.08 vs 9.91 at 512-bit), consistent with panel_r2 — but at the
streaming nv=384 stage it *won* once (13.99 vs 14.20), so leaving it in the
candidate set costs nothing and may pay somewhere. pf: −0.8% to −5.6% at
nv=384 on wallaby, +2–4% at nv=256 (40 MB still fits wallaby's 60 MB L3 —
exactly why the decision is measured per machine, in the scored regime).

Correctness: rel L2 vs numpy 3.114e-16 (B=1), 3.151e-16 (B=8), 3.153e-16
(B=256), 3.155e-16 (B=2048); bit-identical output across repeated executes at
every batch size; AVX2 host (wombat) path verified at B=64 (3.158e-16, the
emulated-zmm and pinned candidates correctly eliminate themselves there);
`-Wall -Wextra` silent; `-fsanitize=undefined` clean at B=8;
`-DL17R_FORCE`/`-DL17R_FORCE_PF` dev overrides compile.

### What was tried and did NOT work / what to know

1. **Nothing new was tried and rejected on numbers this round** — the round
   was spent building the selection machinery rather than new kernels; every
   previously-killed idea (slab lane-packing, transpose fusion, NT stores,
   negacyclic splits, same-volume prefetch) stays killed per the earlier
   sections.
2. **wallaby's B=256 pf verdict does not transfer to the node.** 256 volumes
   (40 MB in+out) fit wallaby's 60 MB L3 but not the node's 22 MB, so the
   node's stage-2 tuner sees a streaming regime at B=256 where wallaby sees a
   cache-resident one. Expect pf=1 on the node at B≥256; do not be surprised
   if the description strings differ between machines.
3. **The stage-1/stage-2 cap (nv ≤ 384) can under-serve B=2048 on huge-L3
   machines**: wallaby's tuner at nv=384 (60 MB ≈ its L3) reported 12.4–14
   µs/t where the B=2048 steady state is 19. On the node (22 MB L3) nv=384 is
   genuinely streaming, so the ranking should hold there. If a future machine
   has >60 MB of L3, raise the cap.

### Borrowed this round (attribution)

* **L17_matrixsimd**: the entire plan-time tuning design — self-#include
  width templating, blocked never-interleaved candidate timing, tuner scratch
  with deterministic pseudo-random fill, reporting the pick through
  `fft3d_description()`, and the `L17_FORCE`-style dev override.
* **L17_winograd**: cross-volume `prefetcht1` of the next volume's input
  during a compute-bound phase (their round-2 item 3, measured −4.4% at
  B=2048), and the tune-in-the-scored-regime rule (their round-2 item 2).

### Next (in order)

1. **Read the node's description strings off the panel_r3 leaderboard** —
   they carry the answers to VW (4 vs 8), pinning, and pf per batch size on
   the actual scoring machine. Whatever they say supersedes every projection
   in this file.
2. **If B=1 is still ~19 µs after the node picks its best width**: the non-FP
   floor (two 17×17 transposes + deinterleave ≈ 2–3 µs serialized, plus ~60
   stack moves and ~90 constant loads per kernel) is now the whole gap to
   matrixsimd, whose transposes hide under the FMA stream inside its chunk
   stores. The untried idea: fuse the T→U transpose into the y pass's *load*
   stage as per-block 4×4 in-register tile transposes, so the shuffle work
   sits on port 5 under the kernel's FMA stream instead of in its own
   serialized loop. Costed at roughly µop-parity with the separate transpose
   (5×5 tiles per plane vs 16+edges, minus U's store+reload), so it wins only
   through port overlap — wallaby (2 FMA units, less port pressure) will
   understate the node benefit; it needs a node A/B, i.e. build it as a fifth
   exec variant and let the tuner judge.
3. **If winograd still leads at B=2048 with pf=1**: the remaining lever is
   output-side traffic. Direct NT stores stay dead (r1 item 7, matrixsimd's
   node measurement); the staged-NT variant (finish a volume in scratch, one
   aligned streaming copy out) is a tuner candidate away — matrixsimd
   measured it losing on the node at 384 volumes, so only bother if the node
   pf verdict shows the input side saturated.

---

## Round panel_r4 (2026-08-21)

### Standing going in (panel_r3 node leaderboard)

3rd in all four cells. B=1: 18.491 (matrixsimd 16.386, winograd 18.169).
B=8: 19.792 (2nd actually, matrixsimd 17.930). B=256: 26.205 (matrixsimd
21.444, winograd 23.905) — a **+7.4% regression** vs my own r2, which the
monitor's VERDICT attributed to my tuner picking `pf=1` at B=256 (both rivals
picked pf=0; the node description strings confirm my pick was `512-bit, pf=1`
in all three runs). B=2048: 27.114. The r3 VERDICT's L=17 findings that shape
this round: arithmetic is CLOSED (matrixsimd's −11.9% FP ops bought −0.8%),
store *order* is what pays (their X-first reorder bought −10.8% at B=256,
"more than every fusion attempt on the board combined"), and my node B=1
(18.49) sits ~3% above the pure-512-bit FP floor (139 blocks × 296 = 41.1k
cycles = 17.9 µs at 2.30 GHz) — i.e. the node B=1 is FP-PORT-BOUND AT VW=8,
and the gap to matrixsimd is exactly my worse 512-bit lane tax (their zmm
holds 4 complex → 17 lanes cost 5×4 slots = 36.0k cycles/volume; my split
re/im zmm holds 8 lanes → 17 costs 3×8 = 41.1k).

### What changed (kernel arithmetic untouched: still 296 FP / 488 flops per 17-pt)

1. **Mixed-width tail blocks ("t" tuner candidates, the round's headline
   bet).** On the Gold 5218 (ONE 512-bit FMA unit, TWO 256-bit FMA ports) a
   ymm kernel block retires 296 FP ops in ~148 cycles where a zmm block needs
   ~296. So each 17-lane plane pass becomes 2 zmm blocks + **one ymm tail**
   (the already-instantiated `wino17_w4`, called from the w8 exec): 740
   cycles instead of 888 — the pure-VW=4 floor with VW=8's instruction/load
   counts. z pass: zmm {0,8} + ymm at 16 (one real lane + 3 pad lanes; safe
   in place because nothing overlaps). y pass: zmm {0,8} + ymm at 13
   (overlap lanes 13–15 recomputed bit-identically). x pass: 36 zmm + ymm at
   285. Saving: 148 cycles × 35 tail blocks = **5.2k cycles ≈ 2.2 µs/volume
   at 2.30 GHz — the whole B=1 gap to matrixsimd on paper** (predicted floor
   35.9k cycles = 15.6 µs ≈ theirs). On wallaby (TWO 512-bit units) the mix
   is FP-neutral and measures 3–4% WORSE (tail adds a second inlined kernel
   + w4's denser instruction stream): B=1 xl-512t 10.24 vs xl-512 9.84. **So
   wallaby cannot confirm this; it is a pure node bet and ships only as a
   plan-time tuner candidate** — if the port arithmetic is wrong the node
   tuner discards it and nothing is lost. Verified bit-identical to the pure
   widths (md5 of full outputs, all 5 class-A candidates, B=8 and B=256).

2. **X-first pass order: built, measured, REJECTED — the round's main
   negative result.** Adopted the idea from L17_matrixsimd's panel_r3 reorder
   (their −10.8%/−7.7% node gains at B=256/2048). Implementation: the kernel
   gained an interleaved-load mode (lmode=1: two vector loads + DLE/DLO
   deinterleave shuffles per input, so the x pass reads the caller's `in`
   directly), x pass writes split A[kx][y][z], then per-kx-plane
   transpose→z→transpose→y with an interleaving stride-17 store straight
   into `out`. Wallaby, pinned, alternating same-window A/B vs X-last:
   * B=256: xf 12.40 vs xl **11.17 µs/t** (xf +11%)
   * B=2048: xf 26.9 vs xl **16.9 µs/t** (xf +60%!)
   Phase isolation (skip-ifdef probe builds, B=2048/volume): xf x-pass alone
   5.9 µs (DRAM read + A write — fine); xf plane phase alone **22.1 µs** vs
   xl plane phase 10.3 µs and xl x-pass-alone (the burst `out` write) 6.0 µs.
   Diagnosis: my y pass emits DRAM-destined output through 17-row strided,
   16-byte-aligned partial-line stores *interspersed with compute*, and that
   costs ~2× what the X-last dedicated burst costs; matrixsimd's X-first
   works because their chunk store writes a finished plane densely. I then
   tested the L8_radix8-style fix — y pass stores split into the dead T
   buffer, one tight sequential interleaving copy per plane to `out`
   (bit-identical, verified by cmp): improves xf to 23.3 µs/t at B=2048 but
   **still loses to xl at 20.1 in the same window**. Conclusion: for THIS
   pass structure X-last is simply right on wallaby; the 78.6 KB burst store
   at the volume end is already near the machine's streaming rate. The xf
   code ships in the file but the class cut defaults to disabled
   (`-DL17R_XF_CUT=64` re-enables). **Monitor: a node A/B of
   `-DL17R_XF_CUT=64` at B=256/2048 would settle whether CLX (22 MB L3, so
   B=256 truly streams; different store-buffer behaviour) flips this** — I
   could not test that from here, and matrixsimd's node numbers say the
   reorder is worth 10% for a structure only modestly different from mine.

3. **The pf mis-pick is fixed two ways**: the stage-2 prefetch A/B now
   requires a **3% margin** to switch pf on (r3's B=256 pick was a near-tie
   that cost 7.4% steady-state), and the streaming re-rank uses 4 reps
   instead of 3. Wallaby now picks pf=0 everywhere (e.g. B=256: off 12.42 /
   on 12.48 → pf=0, where r3's code had picked pf=1 on the node).

4. **Bit-class discipline formalised** (from L17_matrixsimd): X-last and
   X-first are separate bit-equivalence classes; the class is a pure function
   of batch size, the tuner selects freely only within a class. All five
   X-last candidates {256, 512, 512t, 512pin, 512t-pin} verified bit-identical
   by md5 at B=8 and B=256; all four xf candidates likewise at B=64.

5. **I-footprint control**: gcc 11 completely unrolls a 2-trip kernel loop
   even under `#pragma GCC unroll 1` (the mixed exec hit 8 inlined kernels,
   7.4k instr ≈ 29 KB, uncomfortably near the 32 KB L1i — r2 already showed
   38 KB kills). Fix: an opaque loop bound (`__asm__("" : "+r"(mlim))`)
   keeps the 2-trip zmm loops rolled → 6 kernel instantiations, 6.2k instr
   ≈ 25 KB. **Pragma unroll is not a guarantee; asm-opaque bounds are.**

### Operation count

Per 17-point transform unchanged: 296 FP instr (192 FMA + 104 add/sub).
Node-cycle floors per volume at 2.30 GHz: pure w4 36.0k (243 blocks, 2/cyc),
pure w8 41.1k (139 blocks, 1/cyc), **mixed w8+ymm-tail 35.9k** (104 zmm ×
296 + 35 ymm × 148). Static per exec (wallaby build): np_w8 3766 instr /
888 FP; npm_w8 6206 / 1776 (6 kernel copies); xf_w8 4487 / 888.

### Measured — wallaby (Gold 6448Y, gcc 11.4, panel flags; min over ≥3 pinned runs; clock bimodal, fast-mode figures)

| case | panel_r3 code | this round | pick |
|---|---|---|---|
| B=1 | 9.81 | **9.83** (no change expected: mixed is node-only) | xl-512, pf=0 |
| B=8 | 10.00 | **9.81 µs/t** | xl-512 |
| B=256 | 11.47 | **11.08 µs/t** | xl-512(±pin), pf=0 |
| B=2048 | 19.05 | **17.22 µs/t** (unpinned tryout best; pinned slow-window 20.1) | xl-512, pf=0 |

Correctness: rel_l2 3.114e-16 (B=1), 3.151e-16 (B=8), 3.153e-16 (B=256),
3.155e-16 (B=2048); bit-repeatable across processes at every batch; AVX2
host (wombat) verified end to end (30.1 µs/t at B=8, emulated-zmm candidates
self-eliminate); `-Wall -Wextra` clean.

### What was tried and did NOT work — numbers that killed it

1. **X-first at batch, both store flavours** — see item 2 above: +11% (B=256)
   and +60%/+16% (B=2048, strided/sequential store) on wallaby. Disabled by
   default; do not re-enable without a node measurement.
2. **`#pragma GCC unroll 1` on a 2-trip loop containing an always_inline
   kernel** — silently ignored by gcc 11's early complete unroller; 29 KB of
   exec body. Use an asm-opaque bound.
3. (Inherited dead ends all stay dead: slab lane-packing, transpose fusion,
   NT stores, negacyclic splits, same-volume prefetch, non-inline kernels.)

### Borrowed this round (attribution)

* **L17_matrixsimd**: the X-first idea itself (their panel_r3 reorder) and
  the bit-equivalence-class discipline for structurally different variants;
  also their store-order finding is what my phase probes confirmed from the
  other side.
* **L8_radix8** (panel_r3): "make output stores sequential even at the cost
  of an extra L1 round trip" — tested here as the seq-store xf variant;
  correct diagnosis, insufficient cure for my structure.
* **Monitor's panel_r3 VERDICT**: the pf=1 diagnosis (→ 3% hysteresis) and
  the "add candidates, do not replace structures" rule (→ xf shipped
  disabled rather than swapped in).

### Next (in order)

1. **Read the node's panel_r4 description strings.** The single question this
   round poses the node: does `xl 512t` (mixed ymm tail) win B=1/B=8 as the
   port arithmetic predicts (~16.3 µs at B=1, i.e. level with matrixsimd)?
   If yes, the same 5.2k cycles help every batch cell too. If the node keeps
   pure `xl 512`, the one-FMA-unit premise is wrong for interleaved ymm/zmm
   streams and the file's header table should be corrected.
2. **Ask the monitor for one forced A/B: `-DL17R_XF_CUT=64` at B=256/2048.**
   Costs one build flag; settles whether the X-first verdict flips on CLX
   where B=256 actually streams (wallaby's 60 MB L3 made B=256
   cache-resident, which is exactly the regime difference the r3 VERDICT
   warns about).
3. **Software-pipeline across volumes WITHIN X-last** — the bit-class-safe
   version of write-spreading that this round's negative result motivates:
   interleave volume b's x-pass burst (37 chunks writing `out`) with volume
   b+1's plane phase (compute on L2 scratch), using ping-pong A buffers.
   Volumes are independent, so this is bit-identical to plain xl and can be
   a free tuner candidate; it attacks the same serialisation X-first was
   aimed at without moving the strided stores into the compute loop. This is
   also the monitor's named remaining lever for L=17 batched (~1.39× of
   un-overlapped memory time at B=2048).
