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

---

## Round panel_r5 (2026-08-21)

### Standing going in (panel_r4 node leaderboard)

2nd at B=1 (17.742 vs matrixsimd 16.431) and B=8 (19.293 vs 18.008); 3rd at
B=256 (25.202 vs 21.626/24.032) and B=2048 (25.704 vs 22.290/24.221). The
node tuner selected the mixed-width tail in ALL FOUR cells (`xl 512t pin` at
B=1, `xl 512t` elsewhere) — the r4 bet paid −2.5..−5.2% across the board and
the VERDICT (§5b) records it as the first confirmed port-level lever on this
part. Two r4 VERDICT facts reshape this round:

1. **The node's sustained AVX2 clock is 3.89 GHz, not 2.30** (L6_unrolled's
   probe). Re-derived: my B=1 of 17.742 µs is ~69k cycles against the mixed
   shape's ~35.9k-cycle FP floor — **the node B=1 is NOT FP-bound; roughly
   half the runtime is non-arithmetic**. The r4 claim "node B=1 sits ~3%
   above the FP floor" was an artifact of the wrong clock. Prime suspect:
   the per-plane transpose loops (deint + 2× transpose17 ≈ 1.1k
   port-5/load/store µops per plane, ~19k µops per volume) which run
   SERIALIZED between kernel calls, plus ~90 constant loads and ~60 stack
   moves per kernel block.
2. The monitor's L=17 asks: measure the AVX-512 licence clock and report it
   in the description string; and run the `-DL17R_XF_CUT=64` node A/B.

### What changed (kernel arithmetic untouched: 296 FP / 488 flops per 17-pt)

1. **Overlapped-shuffle exec variants (`ov`, mixed-width only) — the round's
   headline bet, aimed at the node's non-FP half.** Mechanism: on the Gold
   5218's single 512-bit FMA unit a zmm kernel block drains ~296 port-0 µops
   at 1/cycle while allocation (4/cycle) runs far ahead, so younger
   INDEPENDENT shuffle/load µops issue on ports 5/2/3 essentially for free —
   but only if they are emitted in the block's shadow instead of in their own
   serialized loop. `exec_ov_body` reorders the SAME operations so every
   shuffle burst sits in a zmm drain:
   * z pass runs its ymm tail FIRST, zmm blocks LAST → the T→U transpose
     that follows lands in a zmm drain;
   * `transpose17` is split into halves by destination-column range
     (`transpose17_part`): cols 0..7 (all that the y pass's first zmm block
     reads) emitted before the y loop, cols 8..16 emitted right after the
     first y kernel block;
   * the NEXT plane's `deint_transpose17` (T is dead once T→U completes) is
     split likewise and slotted after the y pass's second zmm block and its
     ymm tail; plane 0's deinterleave runs at the top of the volume in the
     previous volume's x-pass shadow.
   All moved pieces touch regions disjoint from anything concurrently live,
   so ov is **bit-identical to every other class-A candidate** (cmp-verified
   at B=8 for both ov and ov-pin against 512t). Two new tuner candidates
   (`xl 512t ov`, `xl 512t ov pin`); class A is now 7 candidates on EVEX.
   I-footprint checked: exec_ovm_w8 = 7355 instr ≈ 29 KB (still 6 kernel
   copies, 1776 FP — the branchy rolled y-loop did NOT unroll; the r4
   asm-opaque-bound trick holds), below the 38 KB kill line from r2.
   **On wallaby ov loses ~4.7% at B=1 (10.19 vs 9.72) — expected: two
   512-bit FMA units make the drains half as long, so there is half the
   shadow.** This is a pure node bet shipped as a tuner candidate, exactly
   the r4 mixed-tail play; notably wallaby's occasional full-turbo sessions
   DID pick ov at B=1 once, so the candidates are close even there.
2. **Dual-width clock probe in the description string** (monitor's ask):
   `clk256=`/`clk512=` from serially dependent FMA chains (latency 4 at both
   widths on CLX/SPR), best of 5 trials, run after the tournament, ~20 ms,
   unscored. ADOPTED FROM L6_unrolled panel_r4, extended with the 512-bit
   chain. Verified the chain contracts to vfmadd (objdump: no separate
   mul/add). On wallaby it reports 4.10 GHz in normal sessions and 2.10 in
   throttled ones — independently confirming L6's "clock lottery"; SPR shows
   clk256 == clk512 (no licence gap), and **the node's clk512 is the number
   nobody has ever measured** — read it off the panel_r5 JSONs.
3. **The X-first class choice is now MEASURED at plan time** when batch ≥ 64:
   both classes ranked on the streaming arena, X-first must beat the X-last
   incumbent by >3% to be selected. This runs r4's requested node A/B at
   every plan, forever. Per-plan determinism (rule 4) holds; a cross-process
   class flip would change output bits, which the margin + X-last default
   confine to genuinely-winning cases. `-DL17R_XF_CUT` still force-selects.
   On wallaby the measurement keeps X-last everywhere (B=256: xl 11.79 vs
   xf 12.58; nv=978: xl 16.27 vs xf 21.09) — consistent with r4.
4. **L3-scaled tuner arena** (ADOPTED FROM L17_matrixsimd panel_r4, itself
   from L36_mixedradix): nv = min(batch, clamp(2.5·L3/157KB, 384, 1024)) via
   sysconf. Node (22 MB) → 384, bit-for-bit r4 behaviour; wallaby (60 MB) →
   978, which finally makes its streaming stage actually stream — and its
   pf decision flipped to pf=1 there (14.95 vs 16.01 at nv=978, an honest
   −6.6%), worth ~1 µs/t at B=2048 on wallaby.
5. **Clock-settle spin before ranking** (~150 ms of a real zmm exec): without
   it the first two or three candidates in a rank are measured on a
   still-ramping clock. Seen directly on wallaby at nv=256: `xl 512t` timed
   at 20.96 µs/t when ranked third and 11.88 as `512t pin` two slots later —
   bit-identical work, 76% apart, purely table order. With the settle the
   same table spans 11.5–12.1. Any tuner that ranks candidates in a fixed
   order needs this or its early slots are handicapped.

### Operation count

Unchanged per 17-point transform: 296 FP instr (192 FMA + 104 add/sub).
ov moves zero arithmetic. Cycle floors per volume, restated at the REAL
clocks (r4 VERDICT): mixed w8 ≈ 35.9k cycles = 9.2 µs at 3.89 GHz (if zmm
clock = AVX2 clock; the probe will say). Node B=1 measured 17.742 = ~69k
cycles → ~33k non-FP cycles/volume is the prize ov attacks; the transposes
alone are ~19k µops of it.

### Measured — wallaby (Gold 6448Y, gcc 11.4, panel flags; min over ≥3-4
### alternating pinned runs, taskset -c 17; clock bimodal as always)

| case | panel_r4 code | this round | pick (settled tuner) |
|---|---|---|---|
| B=1 | 9.83 | **9.465 µs** | xl 512 / 512t (within 0.002) |
| B=8 | 9.81 | **9.73 µs/t** | xl 512t |
| B=256 | 11.08 | **10.95 µs/t** | xl 512 pin, pf=0 |
| B=2048 | 17.22 | **16.26 µs/t** | xl 512, pf=1 (L3-scaled arena) |

Correctness: PASS rel_l2 = 3.114e-16 (B=1), 3.151e-16 (B=8), 3.158e-16
(B=64), 3.153e-16 (B=256), 3.155e-16 (B=2048); bitwise repeatable across
runs at every batch size; ov/ov-pin cmp-identical to 512t; `-Wall -Wextra`
silent; `-fsanitize=undefined` clean at B=8; AVX2 host (wombat) verified
end-to-end (30.2 µs/t at B=8, emulated-zmm candidates self-eliminate).
Setup: 0.18 s (B=1) to 1.9 s (B=2048 on wallaby's 978-volume arena; node
arena stays 384 → ~0.5 s).

### What was tried and did NOT work / caveats

1. **ov on a two-FMA-unit machine**: 10.19 vs 9.72 µs at B=1 on wallaby
   (−4.7%) — the shadow is half as long and the split transposes cost branch
   and code-layout overhead. NOT a failure of the mechanism, but the reason
   it ships as a tuner candidate rather than a replacement. If the node
   rejects it too, the non-FP 33k cycles are NOT hiding in alloc-shadow-able
   shuffle bursts and the next lever is the kernel's own ~90 constant loads
   + 60 stack moves per block (attack the vv/cc spills), or the zmm licence
   clock is far below 3.89 and B=1 really is FP-bound — the probe
   disambiguates these two BEFORE anyone spends a round on either.
2. **Unfixed-order ranking without a settle spin mis-ranks by up to 76% on a
   ramping clock** (numbers in item 5 above). This likely polluted my r2-r4
   wallaby tuner tables too (first-slot candidates handicapped); node tables
   (exclusive, warm) were probably fine.
3. (Inherited dead ends stay dead: X-first on wallaby, slab lane-packing,
   transpose fusion INTO stores, NT stores, negacyclic splits, same-volume
   prefetch, non-inline kernels, pragma-unroll-on-2-trip-loops.)

### Borrowed this round (attribution)

* **L6_unrolled** (panel_r4): the FMA-chain clock probe, verbatim mechanism,
  extended to 512-bit.
* **L17_matrixsimd** (panel_r4): the L3-scaled tuner arena (their item 3,
  via L36_mixedradix), and the general "candidates must be measured jointly
  in the scored regime" discipline the class-choice measurement follows.
* **Monitor's r4 VERDICT**: the 3.89 GHz re-derivation that motivated ov,
  and both L=17 asks (clock probe, XF A/B) executed this round.

### Next (in order)

1. **Read the node's panel_r5 description strings**: (a) clk256/clk512 — the
   licence-clock question, closed for good; (b) whether `xl 512t ov` was
   picked at B=1/B=8 (the port-shadow bet), and whether the measured class
   choice kept X-last at B=256/2048 (I expect it does; matrixsimd's X-first
   advantage is their dense chunk store, which my pass structure lacks).
2. **If ov wins at B=1**: extend the same treatment to the x pass at batch
   (interleave the next volume's plane-0 deint into the x-pass tail blocks —
   at B=1 there is nothing to overlap there, but at batch it is the one
   remaining serialized burst).
3. **If ov loses AND clk512 ≈ 3.9**: the non-FP gap is in the kernel blocks
   themselves — attack the vv/cc array spills (~60 stack moves/block) by
   splitting the 8-accumulator negacyclic into two 4-accumulator passes over
   vv (halves peak liveness; costs re-loading vv once — 16 extra loads vs
   ~30 saved spill round-trips, worth trying only with the node's numbers in
   hand).
4. **If clk512 << 3.9 on the node**: B=1 is closer to FP-bound than the 3.89
   arithmetic says; the mixed tail is already the right shape, and the next
   win is lane-tax reduction, which r2 item 2 killed at plane-pair
   granularity but which a kx-blocked x pass (2 volumes' x passes fused to
   fill 512-bit lanes) has never been costed for.

---

## Round panel_r6 (2026-08-21)

### Standing going in (panel_r5 node leaderboard)

2nd at B=1 (17.098 vs matrixsimd 15.223) and B=8 (18.605 vs 16.658); 3rd at
B=256 (24.843 vs 21.198 / winograd 23.933) and B=2048 (25.500 vs 21.983 /
24.567). Node picks: `xl 512t, pf=0` in ALL FOUR cells — the r5 `ov` bet was
selected in zero cells (r5 item 1's losing branch), and my clock probe
answered the licence question for good: **clk256 = 3.89, clk512 = 2.89 GHz**
(confirmed by four of five probes panel-wide; the r5 VERDICT's synthesis is
"the cost is the licence *transition*, not the width — once inside the
512-bit licence, mix widths freely"). Re-derived at 2.89 GHz, my node B=1 is
49.4k cycles against the ~35.9k mixed-shape floor: ~13.5k cycles non-FP.
The r5 VERDICT's two batched facts that shaped this round: the node-winning
streaming mechanism of the round was **write-intent prefetch** (L8_fusedaxes
`fused+pfs+pfw` B=2048 −31%; L36_pfa `inplace pf=2` B=256 −16.6% — "hide the
RFO, don't avoid it"; NT stores lost for the fourth round running), and my
own r4 "Next" item 3 (cross-volume software pipelining within X-last) was
still unbuilt.

### What changed (kernel arithmetic untouched: 296 FP / 488 flops per 17-pt; three additions, all tuner-gated)

1. **Software-pipelined X-last ("xl 512t sp"), the round's structural bet.**
   Volume b−1's x pass (37 mixed-width blocks writing 78.6 KB of
   DRAM-destined `out` in one burst, FMA idle) is interleaved into volume
   b's plane phase: after each plane's y pass, 2–3 x-pass blocks of the
   previous volume run (⌈37/17⌉ pacing, `((x+1)*37)/17 − (x*37)/17`),
   reading a **ping-pong A pair** (`ar2_w8/ai2_w8`, +80 KB scratch, pads
   zeroed once) so A(b−1) stays live while the plane phase fills A(b).
   Per-volume kernel calls and operands are exactly `exec_body(0,0,1)`'s —
   only the global order moves across volume boundaries and volumes are
   independent — so the output is **bit-identical to every class-A
   candidate** (cmp-verified on full outputs vs forced `512t` at B=8 and
   B=64 on wallaby). I-footprint discipline: the x block runs behind a
   **noinline+noclone helper `xblk_run`** (one shared instantiation of the
   zmm kernel + ymm tail, 1580 instr), so the pipeline adds ZERO inlined
   kernel copies; exec_spm_w8 = 5262 instr ≈ 21 KB + 6.3 KB helper, under
   r2's 38 KB kill line. This is the class-A-safe version of what
   matrixsimd's X-first chunk store gets structurally: output writes spread
   across compute.

2. **Paced write-intent prefetch ("pfw") on the x pass's `out` stores** —
   ADOPTED FROM L8_fusedaxes (panel_r5 pfw pick, B=2048 −31%) and L36_pfa
   (panel_r5 `pf=2`, B=256 −16.6%), the round's only cross-geometry
   node-proven mechanism. Before x-pass block k runs, the 17 destination
   row regions of block k+2 are prefetched with write intent (3 lines per
   zmm row region — the rows are 16-B-aligned 128-B strided stores — 2 for
   the ymm tail; block 35's region covers the tail's rows so the pacing
   closes). Wired into the mixed x pass and `xblk_run`; plan flag `pfw`,
   reported in the description string. **Gated by measurement at BOTH
   stages** now: the streaming stage ranks (pf, pfw) jointly over 4 configs
   (blocked, min-of-4, 3% margin vs (0,0) — r3's pf mispick and L36_pfa's
   +13%-on-resident-lines both say never default this on), and the
   small-batch stage got its own pfw A/B (3% margin) after wallaby measured
   pfw **−5 to −7% at B=1** (below).

3. **Per-candidate licence warmup in the tuner** (honesty fix, from the
   clock result + pencilfused's predecessor-state lesson): each candidate
   now warms itself for ≥1.5 ms (not just 2 execs) before being timed.
   With clk256=3.89 vs clk512=2.89 and Intel's ~670 µs licence-up dwell, a
   ymm candidate ranked after a zmm one (or after the zmm settle spin) at
   the B=1 stage (one exec ≈ 200–350 µs) was measured inside the AVX-512
   licence — the only way the tuner could ever discover a licence-clock win
   for `xl 256` is to warm past the dwell. Costs ~10 ms of plan time.

### Operation count

Unchanged per 17-point transform: 296 FP instr (192 FMA + 104 add/sub).
sp moves zero arithmetic; pfw adds ~600 prefetchw µops per volume (37
blocks × 17 rows × 2–3 lines ≈ 0.1k cycles equivalent — noise against the
~8 µs/volume of memory exposure it attacks at B=2048). Static (wallaby
build): exec_spm_w8 5262 instr, xblk_run_w8 1580 (single shared copy,
verified in objdump — noclone held), exec_npm_w8 6206 → 6734 (+pfw wiring).

### Measured — wallaby (Gold 6448Y, gcc 11.4, panel flags; same-window
### alternating pinned runs `taskset -c 17`, min over ≥3; clock bimodal as always)

Same-window forced A/B (t = xl 512t, +w = pfw=1), µs/transform:

| case | t | t+pfw | sp | sp+pfw |
|---|---|---|---|---|
| B=1 | 9.52–9.59 | **9.06–9.08 (−4.9%)** | 9.64 | 9.17 |
| B=256 | 10.30 | 10.35 | 10.76 | 10.79 |
| B=2048 | 14.48 | 14.53 | 15.25 | 15.47 |

So on wallaby: **pfw wins B=1 clearly** (the strided 128-B partial-line
stores expose the out RFO even L2-resident — opposite sign to L36_pfa's
structure, which is why it ships measured-gated, not defaulted), is neutral
batched; **sp loses ~1% (B=1) to ~5% (B=2048) forced** — two 512-bit FMA
units and DDR5 mean wallaby's x-pass burst never stalls, so the pipeline
only adds overhead there. BUT in the one streaming-arena tuner table caught
verbose (nv=978, contended), `xl 512t sp` RANKED FIRST (16.66 vs 512:16.75)
and the joint prefetch A/B chose pf=1 pfw=1 (14.0 vs 20.3 in-arena) — the
candidates are close enough that only the node's stable clock can rank them.
Like r4's mixed tail and r5's ov, **sp is a node bet shipped as a tuner
candidate**; unlike ov, its mechanism (spreading a store burst that STALLS)
is exactly what the node's 1.39× un-overlapped memory time at B=2048 is
made of, and the node's slower DRAM + single FMA unit both push its way.

Autotuned end-to-end (tryout.sh, best windows):

| case | panel_r5 code | this round |
|---|---|---|
| B=1 | 9.465 | **8.838 µs** (pick: 512t pin, pfw=1) |
| B=8 | 9.73 | **9.15 µs/t** |
| B=64 | — | 10.84 µs/t |
| B=256 | 10.95 | **11.03 µs/t** (window-limited; forced-t same-window 10.30) |
| B=2048 | 16.26 | **15.08 µs/t** |

Correctness: PASS rel_l2 = 3.114e-16 (B=1), 3.151e-16 (B=8), 3.158e-16
(B=64), 3.153e-16 (B=256), 3.155e-16 (B=2048) — same fingerprints as r5, so
the bit class is preserved; bitwise repeatable across runs at every batch
incl. forced sp+pfw at odd batches (B=1, B=3); sp/sp+pfw/t+pfw all
cmp-identical to 512t on full outputs (B=8, B=64); `-Wall -Wextra` silent;
`-fsanitize=undefined` clean at B=8; AVX2 host (wombat) verified end-to-end
(30.6 µs/t at B=8; EVEX-only candidates self-eliminate). Setup cost grew
~10 ms (licence warmups) + one extra candidate: 0.19 s at B=1, 2.0 s at
B=2048 on wallaby (node arena stays 384).

### What was tried and did NOT work / caveats

1. **sp forced on wallaby loses at every batch size** (numbers above) —
   recorded so nobody deletes it for the wrong reason: the mechanism needs
   a store burst that actually stalls, which wallaby's memory system
   doesn't provide. The node A/B is the point (`-DL17R_FORCE=7` forces sp;
   `-DL17R_FORCE_PFW=1` forces pfw).
2. **pfw at batch on wallaby is neutral** (14.48 vs 14.53 at B=2048 forced)
   even though B=2048 streams there too — wallaby's RFO cost is simply
   lower; the L8/L36 evidence for pfw is node evidence, which is what
   matters. Do not extrapolate wallaby's B=1 pfw win to the node either:
   the gate measures it per machine, per batch.
3. **Nothing else was tried and killed this round**; every inherited dead
   end stands (X-first on this structure, slab lane-packing, transpose
   fusion into stores, NT stores, negacyclic splits, same-volume prefetch,
   non-inline *kernels* — note xblk_run is a noinline *block wrapper*, the
   kernel inside it is still always_inline, which is why it does not
   reproduce r1 item 2's 887-instruction disaster).

### Borrowed this round (attribution)

* **L8_fusedaxes + L36_pfa (panel_r5, via the VERDICT §4.5 synthesis)**: the
  paced prefetchw mechanism and its gating discipline (prefetchw on
  resident lines is µop tax — measure, don't default).
* **L36_pencilfused (panel_r5)**: per-candidate self-warming in the tuner,
  here extended to cover the AVX-512 licence dwell specifically.
* **L17_matrixsimd**: the store-order finding (r3/r4) that motivated
  spreading the out burst; the bit-class discipline the sp candidate obeys.
* **Monitor's r5 VERDICT**: the licence-transition synthesis, the 2.89 GHz
  re-derivation, and the L=17 batched memory-time figure sp attacks.

### Next (in order)

1. **Read the node's panel_r6 pick strings**: whether sp took the batched
   cells (the bet), whether pfw was selected anywhere (B=1 included), and
   whether the licence-honest warmup changed any width pick. Each answer is
   a mechanism settled on the scoring machine.
2. **If sp wins batched but the gap to matrixsimd remains**: deepen the
   pipeline one step (issue volume b−1's x blocks TWO planes behind, i.e.
   start them during plane 2 of volume b, giving the store stream a longer
   window) — stays in class A, one more candidate.
3. **If sp loses on the node too**: the batched gap is input-side, not
   output-side (pf=0 was picked at every batch in r5, so the input stream
   is NOT the issue either) — then the remaining lever is the L2 working
   set: A + T + U + pads ≈ 250 KB/volume of scratch traffic, and the
   kx-blocked two-volume x pass (r5 item 4) becomes the only uncosted idea
   left on the list.
4. **B=1 structural**: if pfw is picked at B=1 on the node, the ~13.5k
   non-FP cycles shrink by whatever the RFO share was; what remains is the
   serialized transposes (~19k µops/volume), and the monitor's forced-ov
   measurement (r5 VERDICT §6 L=17 ask) will say whether shadow-scheduling
   can ever recover them or whether the kernels' ~90 constant loads + ~60
   stack moves per block are the real residue.

---

## Round panel_r7 (2026-08-22)

### Standing going in

panel_r6 was ABANDONED between development and timing (stale-runner
retirement; see results/panel_r6_abandoned_no_timing/WHY.md), so there is no
r6 leaderboard: the standings are still panel_r5 (2nd at B=1 17.098 and B=8
18.605 behind matrixsimd 15.223/16.658; 3rd at B=256 24.843 and B=2048
25.500 behind matrixsimd 21.198/21.983 and winograd 23.933/24.567), and my
r6 additions (sp pipeline, pfw, licence-honest warmup) have NEVER been
node-measured.  This round's leaderboard therefore judges r6's and r7's
bets together.

### What changed (kernel arithmetic untouched: 296 FP / 488 flops per 17-pt)

1. **Deferred-junction plane schedule ("dz", "dz pin", "dzsp") — the
   round's structural bet, ADOPTED FROM L17_matrixsimd panel_r6's
   deferred-Z** (their group-level deferral with a double-buffered plane
   buffer: wallaby −3.0% at B=1, −5.7% at B=8 for their structure — the only
   new L=17 mechanism of r6 with a positive same-structure measurement).
   My plane phase had three store→load junctions per plane, all
   back-to-back: deint(x)→z(x) on T, z(x)→transpose(x) on T (in place),
   transpose(x)→y(x) on U.  Each is a group tail whose stores are
   immediately re-read with no independent work behind them — on the node's
   224-entry ROB (2.3× shallower than wallaby's) that exposure is a prime
   suspect for the ~13.5k non-FP cycles/volume at B=1 (r5 accounting at the
   measured clk512=2.89).  `exec_dz_body` software-pipelines the SAME groups
   one plane deep with T double-buffered by parity (U stays single — its
   producer and consumer are one kernel group apart in the same iteration):

       deint(0)→T0; for x=0..17: { transpose(x−1); z(x); y(x−1); deint(x+1) }

   Every junction now has ≥1 full independent group between producer and
   consumer (z→transpose has two); the only exposed ones left are plane 0's
   deint→z and the final transpose(16)→y(16).  Unlike r5's ov — which chased
   the same cycles by SPLITTING the transpose loops and lost to its own
   plumbing — dz moves whole groups and adds ZERO instructions.  Same kernel
   calls, same operand values, same within-pass order → **bit-identical to
   every class-A candidate** (cmp-verified vs forced 512t at B=8 and B=3;
   B=3 exercises the ping-pong parity).  Cost: one extra T pair (+6.5 KB
   scratch; plane-phase footprint T0+T1+U = 19.6 KB of the node's 32 KB
   L1d).  "dzsp" composes the r6 cross-volume x-block pipeline (via the
   shared noinline xblk_run) into the same schedule, for the batched cells.
   Three new class-A tuner candidates (11 total); L17R_FORCE indices:
   8=dz, 9=dz pin, 10=dzsp (0–7 unchanged, sp stays 7).

2. **Two-pass candidate ranking (tuner honesty fix #3).**  Even with the r5
   settle spin and the r6 per-candidate licence warmups, one wallaby verbose
   table this round spanned 27 → 10 µs/t MONOTONICALLY down the table for
   near-identical work — the machine ramped over the whole ~50 ms
   tournament and the settle spin didn't cover it.  `l17r_rank` now runs two
   full fixed-order sweeps and keeps the per-candidate min across both
   (second sweep runs on the settled clock, halving order bias); each
   candidate is still timed in its own contiguous block within a sweep
   (matrixsimd item-12 discipline).  The B=1 pfw A/B and the batched
   (pf,pfw) grid got the same two-sweep treatment at unchanged total exec
   count.  Verified: a later verbose table on a contended window is
   internally consistent (~5% spread, no ramp), matching the forced A/Bs.
   Costs ~2× rank time; setup is now 0.26–0.30 s at B=1 and 4.2 s at B=2048
   on wallaby's 977-volume arena (node arena stays 384 → roughly half that).

### Operation count

Unchanged per 17-point transform: 296 FP instr (192 FMA + 104 add/sub).
dz moves zero arithmetic and zero data; it re-orders whole groups only.
I-footprint (objdump, wallaby native): exec_dzm_w8 7629 instr ≈ 30.5 KB,
exec_dzmpin_w8 7676, exec_dzspm_w8 6136 ≈ 24.5 KB — all under r2's 38 KB
kill line; the rolled 2-trip zmm loops held (asm-opaque bounds).

### Measured — wallaby (Gold 6448Y, gcc 11.4, panel flags; forced pairs are
### same-window alternating pinned runs `taskset -c 17`, min over 4 reps)

Forced A/B at B=1 (F2 = xl 512t vs F8/F9 = dz/dz pin), µs:

| config | pfw=0 | pfw=1 |
|---|---|---|
| xl 512t | **9.459** | **8.874** |
| xl 512t dz | 9.948 (+5.2%) | 9.269 (+4.5%) |
| xl 512t dz pin | — | 9.183 (+3.5%) |

Forced A/B at B=2048 (pf=0, pfw=0, slow-ish window), µs/t:
512t 18.07, sp **17.94**, dzsp 18.44.  **dz loses 3.5–6% on wallaby in every
cell** — the same shape as r4's mixed tail (wallaby −3–4%, node picked it
everywhere at −2.5–5.2%) and for the same reason: two 512-bit FMA units and
a 512-entry ROB already hide these junctions there.  matrixsimd's deferral
won on wallaby because their junction is a store-forwarding chokepoint
between much shorter (74-cycle) chunks; mine sit between 296-cycle drains.
So dz is a **pure node bet shipped as tuner candidates**, riding on the
node's shallow ROB + single FMA unit; if the port arithmetic is wrong the
node tuner discards it and nothing is lost.

Autotuned end-to-end (tryout.sh, mixed windows):

| case | r6 code (best windows) | this round |
|---|---|---|
| B=1 | 8.838 | **8.889 µs** (sd 0.04%) |
| B=8 | 9.15 | 9.33 µs/t |
| B=256 | 11.03 | 11.51 µs/t |
| B=2048 | 15.08 | 16.74 µs/t (setup 4.2 s) |

(Window-limited; no wallaby regression signal — forced same-window pairs
above are the honest statistic, and the incumbents are unchanged code.)

Correctness: PASS rel_l2 = 3.114e-16 (B=1), 3.151e-16 (B=8), 3.138e-16
(B=3), 3.153e-16 (B=256), 3.155e-16 (B=2048); dz/dz-pin/dzsp all
cmp-bit-identical to forced 512t on full outputs (B=8, B=3); repeatable
across runs at every batch incl. forced dzsp at B=3;
`-fsanitize=undefined` clean (forced dzsp+pfw, B=8); `-Wall -Wextra`
silent; AVX2 host (wombat) verified end-to-end (PASS 3.151e-16, 30.4 µs/t
at B=8, EVEX-only candidates self-eliminate).

### What was tried and did NOT work / caveats

1. **dz forced on wallaby loses everywhere** (table above) — recorded so
   nobody deletes it for the wrong reason; the mechanism targets the node's
   224-entry ROB and 296-cycle single-unit drains, which wallaby does not
   have.  Node A/B via `-DL17R_FORCE=8/9/10`.
2. **The one-sweep tuner mis-ranked by up to 2× on a ramping wallaby
   window** (verbose table 27→10 µs/t monotone; the same run's driver
   steady-state was 9.33).  This is the same failure r5 item 5's settle
   spin was meant to fix, at a longer time scale.  Two-sweep min fixed it;
   any fixed-order tuner on a powersave-governor machine needs this.
3. (All inherited dead ends stand: ov's split-loop overlap, X-first on this
   structure, slab lane-packing, transpose fusion into stores, NT stores,
   negacyclic splits, same-volume prefetch, non-inline kernels,
   pragma-unroll on 2-trip loops.)

### Borrowed this round (attribution)

* **L17_matrixsimd (panel_r6)**: the deferred-junction mechanism itself
  ("deferred-Z": group-level deferral + double-buffered plane buffer, zero
  extra instructions), here generalised to a one-plane-deep software
  pipeline covering all three of my junctions; also their raw-ssh
  measurement trap note (hit it again this round — always `cd` in the
  remote command).
* **L17_winograd / L36_pfa / L8_fusedaxes**: nothing new taken this round;
  their pfw mechanism ships unchanged from my r6 adoption.

### Next (in order)

1. **Read the node's panel_r7 pick strings** — they answer, in one shot:
   r6's bets (sp at batch? pfw where?) and r7's (dz/dz-pin at B=1/B=8?
   dzsp at batch?).  Each is a mechanism settled on the scoring machine.
2. **If dz is picked at B=1/B=8**: the junctions were real — deepen the
   pipeline (defer y two planes, or start the x pass's first blocks inside
   the last planes' drains, both still class A) and expect the same
   treatment to pay in matrixsimd-style at B=256+.
3. **If dz is rejected along with ov (r5)**: two independent scheduling
   attacks on the non-FP residue will have failed on the node — the
   remaining suspects are per-block effects (window turnover at kernel
   boundaries, the ~90 constant loads per block), and the honest next step
   is a kernel-shape change: matrixsimd's 148-op half-size chunks give the
   OoO window 2× more boundaries per FP op, which their 15.2 vs my 17.1
   at B=1 says matters more than my 8% lower lane-slot count.  That is a
   big rewrite; do it only on this evidence.
4. **If B=2048 is still 3rd after sp/pfw/dzsp land**: the write path is
   exhausted; the read side (A + T + U ≈ 250 KB of L2 scratch traffic per
   volume) and the kx-blocked two-volume x pass (r5 item 4) are the only
   uncosted ideas left.

---

## Round panel_r8 (2026-08-22)

### Standing going in (panel_r7 node leaderboard)

3rd in all four cells and flat: B=1 17.156 (matrixsimd 15.227, winograd
16.567), B=8 18.586, B=256 24.467, B=2048 25.537.  The node picked
`xl 512t, pf=0, pfw=0` in ALL FOUR cells — meaning every r6 and r7 bet was
rejected at once: sp (r6), pfw (r6), dz / dz pin / dzsp (r7) all got zero
picks.  The r7 VERDICT's L=17 synthesis is one-directional and this round
executes it: the node has now declined THREE scheduling attacks at this
geometry (my ov r5 0/4, my dz r7 0/4, matrixsimd's deferred-Z at the small
cells 0/2), while the only mechanism that moved an L=17 cell in three
rounds is uop DELETION (L17_winograd's g8: delete a transposed store,
−8..−10% in all four cells).  Delete uops, don't reschedule them.

### What changed (kernel arithmetic untouched: 296 FP / 488 flops per 17-pt)

**512-bit plane transposes in the w8 pipeline.**  The per-plane
transpose/deinterleave loops — the bulk of my ~13.5k-cycle non-FP residue
at B=1 (at clk512=2.89) — were still round-1's 4x4 ymm tiles even inside
the 512-bit pipeline.  Replaced, in the w8 template only, with 8x8 zmm
blocks:

* `tz8x8`: the classic 3-stage network (vunpck{l,h}pd / 128-bit-granule
  permute / 256-bit-granule permute) = 8 loads + 24 shuffles + 8 stores
  per 64 elements, against 64 uops for the same elements at 4x4 ymm.
* `dz8x8`: interleaved-complex 8x8 tile, same scheme as the ymm one
  (transpose complex at 128-bit granularity first — two 4x4 granule
  transposes per register half — split re/im after) = 16 + 48 + 16 = 80
  uops per 64 complex against 128.
* `transpose17z_part` / `deint_transpose17z_part` keep the exact part
  contract of the ymm versions (part 0 writes dest cols 0..7, part 1 cols
  8..16, same scalar edges), so all six exec bodies (plain/ov/sp/dz, both
  classes) took the swap through two width-selected macros (TP17/DT17)
  with no restructuring.  w4 keeps the ymm tiles.

Movement-uop count per volume (deint x17 planes + transpose17 x34 arrays):
~20.7k uops at 4x4 → ~14.3k at 8x8, i.e. **~6.4k uops deleted (−31%),
about half of them load/store slots** — which is where the node hurts most
(2 load ports on CLX vs 3 on SPR; T/U/A rows are all 64-B aligned at w8
strides TR=24/PS=296, so the zmm accesses split no cache lines except the
deint's source loads from `in`).  Emitted code verified in objdump: the
transposes land as single-uop vpermt2pd/vpermi2pd/vshuff64x2/vunpck with
the index constants hoisted (38 vmovdqa64 in the whole exec body).

**Pure data movement — identical values to identical places — so every
class-A candidate stays bit-identical by construction**, and the outputs
are bit-identical to panel_r7's too (fingerprints at every batch match
r5/r6/r7 exactly).

### What was measured — wallaby (Gold 6448Y, gcc 11.4, panel flags; forced
### pairs are same-window alternating runs, min over >= 2 reps)

Forced `512t` (F2), same-window alternating old (=impl_7) vs new:

| case | old | new | delta |
|---|---|---|---|
| B=1 (pfw auto) | 9.059 / 9.063 | **8.828 / 8.800** | **−2.8%** |
| B=256 (pf=pfw=0 forced), us/t | 11.37 / 11.05 | **10.87** / 11.04 | −1.6% best-vs-best |

Autotuned end-to-end (mixed windows; wallaby clock bimodal as always):

| case | panel_r7 code (best windows) | this round |
|---|---|---|
| B=1 | 8.889 | **8.660** (pinned direct; tuner session 8.808, pick `xl 512, pfw=1`) |
| B=8 | 9.33 | **9.19 us/t** |
| B=64 | 10.84 (r6) | 11.02 us/t (window-limited) |
| B=256 | 11.51 | **10.87 us/t** (forced window; autotuned 10.90–11.34) |
| B=2048 | 16.74 | **16.34 us/t** |

Correctness: PASS rel_l2 = 3.114e-16 (B=1), 3.171e-16 (B=3, forced dz and
dzsp), 3.151e-16 (B=8), 3.158e-16 (B=64), 3.153e-16 (B=256), 3.155e-16
(B=2048) — the same fingerprints as r5–r7 at every batch, so the bit class
is preserved across rounds; bitwise repeatable at every batch size;
full-output cmp at B=8: forced 512t ≡ dz ≡ sp ≡ **xl 256** (the last one
matters: the w4 path still uses the ymm tiles, so the cmp also proves the
zmm transposes move identical values); `-Wall -Wextra` silent;
`-fsanitize=undefined` clean at B=8; AVX2 host (wombat) verified
end-to-end (30.0 us/t at B=8, PASS, repeatable — gcc emulates the 64-byte
shuffles correctly and the EVEX candidates self-eliminate).

### What was tried and did NOT work — with the numbers that killed it

1. **`#pragma GCC optimize("unroll-loops")` — the L45_pfa panel_r7
   build-flag fix — tested and REJECTED.**  Their finding: the scored build
   lacks tryout.sh's `-funroll-loops` and that cost them 10%.  Same-window
   3-way on wallaby (forced 512t, pfw=0, B=1): unroll ON via command line
   9.29/9.37 us, unroll OFF 9.36/9.49, **pragma form 9.55/9.62**.  So (a)
   the flag itself is worth <1% for THIS file — the hot tile loops fully
   unroll under -O3's complete peeling anyway, and the 2-trip kernel loops
   are deliberately kept rolled by asm-opaque bounds; and (b) the pragma
   form actively costs ~2%: the optimize attribute perturbs codegen beyond
   the named flag (gcc rebuilds the whole per-function option set).  The
   node build (no -funroll-loops) is therefore ~fine for this entry as-is.
   Do not add the pragma back; the file header carries the numbers.
2. A first A/B of the autotuned builds (not forced) read new SLOWER at
   B=1 (9.24–9.63 vs old best 8.888) — that was pick/window variance, not
   the transposes; the forced same-window pairs above and the later
   8.660/8.808 sessions are the honest statistic.  Recorded as a reminder:
   **never judge a mechanism from autotuned runs when the tuner's pick can
   differ between builds.**
3. (All inherited dead ends stand.  In particular this round did NOT
   revisit transpose-fusion-into-stores (r1 item 2) — the 8x8 blocks make
   the standalone transpose cheaper, which moves the trade FURTHER against
   fusion.)

### Borrowed this round (attribution)

* **L17_winograd (panel_r7, via the VERDICT's L=17 synthesis)**: the
  delete-uops-don't-reschedule lesson that selected this round's mechanism.
* **L45_pfa (panel_r7)**: the build-flag gap finding and the scalar-audit
  discipline — tested; the pragma fix does not transfer to this structure
  (numbers above); their objdump-the-hot-function habit found my transpose
  index constants properly hoisted, so no scalar-junk fix was needed.

### Next (in order)

1. **Read the node's panel_r8 picks and the B=1 number.**  The transposes
   benefit every class-A candidate equally, so the pick strings should
   stay `xl 512t pf=0 pfw=0`-shaped; the question is the size of the B=1
   move.  Wallaby says −2.8% on two 512-bit FMA units and 3 load ports;
   the node's single FMA unit means the FMA stream hides LESS of the
   movement work and its 2 load ports price the deleted load/store slots
   HIGHER — both push the node gain above wallaby's.
2. **If B=1 is still >1 us behind matrixsimd after this lands**: the
   remaining residue is inside the kernel blocks (~90 constant loads + ~60
   stack moves per block) — the honest next step is the r7 item-3 kernel
   reshape (matrixsimd's half-size chunks), which is a big rewrite; the
   alternative one-round lever is splitting the negacyclic-8 into two
   4-accumulator passes over vv (r5 next-item 3) to cut the vv/cc spills.
3. **Scalar edges**: the 17th row/column of every transpose is still ~66
   scalar movsd per array (~4.5k/volume).  An AVX-512 masked-store version
   is a small, class-safe uop deletion (~2k uops) if the node shows the
   movement passes still matter after item 1's answer.

---

## Round panel_r9 (2026-08-22)

### Standing going in (panel_r8 node leaderboard)

3rd in all four cells: B=1 16.638 (matrixsimd 15.182, winograd 16.527), B=8
18.073, B=256 24.875 (matrixsimd 21.311, winograd 21.382), B=2048 25.890
(winograd 21.645).  The r8 zmm transposes delivered −3.0%/−2.8% at B=1/B=8 as
predicted but REGRESSED both batched cells (+1.7%/+1.4%) — the entry was not
promoted, for the fourth consecutive round of node-rejected or negative
headline mechanisms.  The r8 VERDICT's L=17 synthesis that frames this round:
FOUR mechanism classes are now falsified at B=1 (rescheduling, address
alignment, spill deletion, movement-uop deletion — all selected by node
tuners, all ~0), B=1 has read 15.18–15.23 for five rounds at 1.31× the 33.4k-
cycle floor, and the verdict's instruction is "stop optimizing and measure,
or rewrite."  My B=1 is at the same wall as everyone's; **my batched cells
are NOT — they trail the fused rivals 15–20%, which is proven headroom.**
This round is therefore batch-only, and it ships measurement before mechanism.

### What changed (kernel arithmetic untouched: 296 FP / 488 flops per 17-pt)

1. **"dy" candidates — ymm deint inside the otherwise-zmm w8 pipeline**
   (`xl 512t dy` = FORCE 11, `xl 512t sp dy` = FORCE 12).  Diagnosis of the
   r8 batch regression: of everything r8 changed, the only stage whose
   behaviour differs between B=1 and batch is the deinterleave — its source
   loads from `in` are the one unaligned zmm stream r8 added, and at batch
   they hit COLD lines.  A 64-byte load on the 16-byte alignment classes of
   an odd-length complex plane splits a cache line 3/4 of the time (the two
   loads of an 8-complex row piece touch 3 lines with 4 line-accesses),
   against ~1/2 for the ymm tile's 32-byte loads; a split on a cold line
   holds LFB entries for two fills on a 2-load-port CLX.  T/U/A accesses are
   64-B aligned and L1-hot at every batch, so they are exonerated — the
   batch-only regression signature points at the deint source alone.  dy
   keeps every zmm transpose and swaps only the deint tile back to ymm.
   Pure data movement, bit-identical (cmp-verified, below), tuner-ranked
   per cell; expected outcome is dy at batch and plain zmm at B=1/B=8, i.e.
   both regimes keep their r8 win.

2. **Joint (variant, pf, pfw) grid at batch — ADOPTED FROM L23_rader
   panel_r8** (their joint grid found plain-xf+pf=2+pw=1, a combination
   stage-1-then-grid ranking can never select, and it took their B=128 cell;
   the r8 VERDICT names racing knobs only on the stage-1 winner "a
   documented, repeated mistake").  The batched stage now runs the 4-config
   (pf,pfw) grid over TWO variants: the stage-1 winner and an sp-flavoured
   partner (7↔2, 11↔12), 3% margin vs the incumbent at (0,0), two sweeps,
   blocked.  Rationale: sp's r7 node rejection was measured at (0,0) only,
   and its mechanism — spread the 78.6 KB out burst across the next volume's
   compute — composes with pfw (prefetchw the rows it is about to scatter).
   Class-A only, so a measured variant switch cannot change output bits;
   disabled under L17R_FORCE so forced A/Bs stay clean.

3. **Streaming decomposition probe in fft3d_create(), reported in the
   description string** as `probe ph/xp/fu` (µs/volume on the streaming
   arena at (pf,pfw)=(0,0)) — the pattern the r8 VERDICT says should become
   the panel default, ADOPTED FROM L36_pfa panel_r8's in-plan node probe.
   ph = plane phase alone (cold `in` reads + A fill: the input-side
   exposure); xp = x pass alone (hot A reads + the `out` burst: the
   output-side exposure); fu = full plain exec of the same family.  Probe
   functions are exec_body instantiations gated by a compile-time `ph`
   flag and are NEVER candidates (xp reads whatever A holds).  ~0.1 s of
   unscored plan time.  On the node this attributes the ~4 µs/volume
   batched gap in one leaderboard line: xp far above its ~4 µs compute
   share means the out-burst RFO is the gap (respread the burst); ph
   carrying the excess means the input side (restage the reads); and
   fu − ph − xp ≫ 0 means the phases contend (attack the L2 footprint).

### Operation count

Unchanged per 17-point transform: 296 FP instr (192 FMA + 104 add/sub),
488 flops.  dy moves zero arithmetic (the ymm deint tile is ~128 movement
uops per 64 complex vs 80 at zmm — dy deliberately re-adds ~3.2k movement
uops per volume to delete cold split-loads; it pays only where the split
cost exceeds the uop cost, which is the batch regime on a 2-load-port
machine).  Probes and grid are plan-time only.  Static (wallaby build):
exec_npmdy_w8 6403 instr (vs npm_w8 6510 — the ymm deint bodies are
smaller), exec_spmdy_w8 4897, probes ph/phdy/xp 4464/4358/1868; everything
under r2's 38 KB kill line.

### Measured — wallaby (Gold 6448Y, gcc 11.4, panel flags; clock bimodal as
### always, same-window comparisons only)

Autotuned end-to-end (tryout.sh):

| case | r8 best (same machine) | this round | pick |
|---|---|---|---|
| B=1 | 8.660 | 9.41 session / 8.87 pinned best (window-limited; B=1 path unchanged) | xl 512 |
| B=8 | 9.19 | 9.23 µs/t | xl 512 |
| B=64 | 11.02 | **10.38 µs/t** | — |
| B=256 | 10.87 | **10.59** / 10.83 µs/t (two windows) | w1: xl 512 pf0 pfw0; w2: **xl 512t dy pf0 pfw1** |
| B=2048 | 16.34 | **15.01 µs/t (−8.1%)** | **xl 512t sp pf=1 pfw=1** (joint grid) |

**The joint grid is the round's result.  B=2048, nv=978 arena, same window:**

| variant | (0,0) | pf=1 | pfw=1 | pf=1,pfw=1 |
|---|---|---|---|---|
| xl 512 (stage-1 winner) | 15.554 | 14.805 | 15.608 | 14.801 |
| xl 512t sp (partner) | 15.835 | 15.521 | 14.406 | **13.631** |

sp had LOST stage 1 (16.09 vs 15.91) and pfw had been near-neutral on the
incumbent for three rounds — but sp+pf+pfw together is −12.4% vs the
incumbent at (0,0) and −7.9% vs the incumbent's own best config.  This is
exactly the L23_rader shape: the winning combination is reachable only by
racing the knobs jointly across variants.  At B=256 (nv=256, L3-resident on
wallaby) the grid read sp+pfw 10.646 vs incumbent 10.766 — under the 3%
margin, correctly kept the incumbent; the node's B=256 arena (40 MB vs
22 MB L3) actually streams, where the B=2048 shape should apply.

**Probe readings (wallaby):** nv=978 streaming: ph=8.91 xp=5.39 fu=15.98 —
fu − ph − xp = +1.68 µs, i.e. at streaming the phases do NOT simply add
even on wallaby (in-read + out-burst contend); nv=256 L3-resident: ph=7.21
xp=3.57 fu=10.90, sum ≈ fu − 0.1 — phases add cleanly when nothing
streams.  The node's numbers are the ones that matter; read them off the
r9 leaderboard descriptions.

**dy on wallaby:** stage-1 nv=256: dy 10.930 vs 512t 10.909 (wash);
nv=978 streaming: dy 16.684 vs 512t 16.433 (−1.5% for dy).  Expected:
wallaby has 3 load ports and DDR5, so split loads are cheap there and dy's
extra uops show.  dy is a node bet (2 load ports, slower DRAM, and the
node is where the +1.7%/+1.4% regression lives); it ships as candidates
the node ranks itself.  In one B=256 window the node-side logic already
picked `dy pfw=1` on wallaby and produced this entry's best-ever wallaby
B=256 (10.59).

Correctness, all PASS with the r5–r8 fingerprints preserved at every batch:
rel_l2 = 3.114e-16 (B=1), 3.171e-16 (B=3), 3.151e-16 (B=8), 3.158e-16
(B=64), 3.153e-16 (B=256), 3.155e-16 (B=2048); bitwise repeatable at every
batch; full-output cmp: forced dy (F11) ≡ forced 512t (F2) ≡ forced sp dy +
pfw (F12) at B=8, and forced sp dy + pf + pfw ≡ 512t at B=3 (odd batch
exercises the sp ping-pong parity); `-Wall -Wextra` silent on both hosts;
`-fsanitize=undefined` clean at B=8; AVX2 host (wombat) verified end-to-end
(30.1 µs/t at B=8, PASS, repeatable, EVEX candidates self-eliminate).
Setup: 0.26–0.31 s at B=1, ~1.0 s at B=256, 4.8 s at wallaby B=2048
(977-volume arena; node arena stays 384 → roughly half).

### What was tried and did NOT work / caveats

1. **The raw-ssh missing-`cd` trap fired a FOURTH time, on me** (matrixsimd
   documented three).  Three identical remote commands failed before I
   noticed the `cd` was missing; the durable fix used here: call
   `tryout.sh` by ABSOLUTE PATH — it `cd`s to its own directory itself, so
   the command cannot be wrong.  Recommend every entry adopt that form.
2. **dy forced on wallaby streaming loses ~1.5%** (16.684 vs 16.433 at
   nv=978) — recorded so nobody deletes it for the wrong reason: the
   mechanism prices cold split-loads on a 2-load-port CLX, which wallaby
   does not have.  The node's r8 batch regression is the evidence it rides.
3. **Nothing else was killed by measurement this round** — the round built
   selection machinery and a probe rather than new kernels; every inherited
   dead end stands (X-first on this structure, slab lane-packing, transpose
   fusion into stores, NT stores, negacyclic splits, same-volume prefetch,
   non-inline kernels, pragma-unroll forms, ov's split-loop overlap).
4. Caveat on the wallaby B=2048 pick: the tuner's arena (978 vols, 154 MB)
   streams past wallaby's L3, but the driver's steady state at B=2048
   (314 MB in+out) is harsher; the picked sp+pf+pfw still delivered −8.1%
   end-to-end, so the arena ranking held.  On the node the arena is 384
   vols = 60 MB vs 22 MB L3 — honestly streaming, as in r5–r8.

### Borrowed this round (attribution)

* **L23_rader (panel_r8)**: the joint (variant, knob) grid — their B=128
  node win is the direct precedent for racing sp jointly with (pf, pfw).
* **L36_pfa (panel_r8)**: the in-plan decomposition probe routed through
  fft3d_description(), endorsed by the r8 VERDICT as the panel default;
  ph/xp/fu is their p1/p2w/fu pattern mapped onto my pass structure.
* **L17_matrixsimd**: the raw-ssh trap documentation (which I re-triggered
  anyway — see item 1) and, standing, the blocked-measurement discipline.

### Expectations for the node, and what to read off the r9 leaderboard

* **B=256/B=2048 are the round's bet.**  If the node's joint grid finds
  what wallaby's found, the pick strings read `xl 512t sp` (or `sp dy`)
  with `pf=1 pfw=1`, and the cells should move from 24.9/25.9 toward
  22.5–24.5 (the wallaby −8% shape, damped by the node's slower DRAM).
  If the node instead picks `dy` alone with pfw, the r8 regression is
  recovered (~24.4/25.5) and the split-load story is confirmed at reduced
  magnitude.
* **The probe values in the description strings are the round's real
  deliverable**: node ph vs xp vs fu at nv=384.  xp ≫ ~4 µs ⇒ the out
  burst is the gap and r10 deepens the sp pipeline (start x blocks two
  planes earlier, or pace 3-wide); ph carrying the excess ⇒ the input side
  and r10 stages `in` through L2; fu − ph − xp ≫ 0 ⇒ phase contention and
  r10 attacks the A/T/U footprint.
* **B=1/B=8 should be flat** (~16.6/18.1): the paths are unchanged and the
  whole panel is at the structural wall there; this entry deliberately
  spent nothing on it.

### Next (in order)

1. Read the node's picks AND the probe ph/xp/fu off the r9 descriptions;
   the r10 move reads directly off which of the three branches fired.
2. If sp+pfw is picked and the gap to the fused rivals narrows but
   persists: deepen the pipeline (issue the previous volume's x blocks two
   planes behind instead of one, giving each store burst a longer drain
   window — still class A, one more candidate), and consider extending the
   joint grid to a third variant slot (dz-family) now that the grid
   machinery exists.
3. If the probe shows fu − ph − xp large on the node: shrink the streaming
   L2 footprint — A is 2×296×17×8 = 80.5 KB; a kx-blocked x pass that
   consumes A in two halves (r5 item 4, still uncosted) would halve the
   live window.
4. If everything is flat again: the honest position, per the r8 VERDICT,
   is that this structure has met its wall at both regimes, and the only
   remaining move is the interleaved-complex rewrite both rivals costed at
   "likely a wash" — recommend the slot be judged on that basis rather
   than spending r10 building it.

---

## Round panel_r10 (2026-08-22)

### Standing going in (panel_r9 node leaderboard)

3rd in all four cells: B=1 16.543 (matrixsimd 14.866, winograd 16.460), B=8
18.063, B=256 24.610 (matrixsimd 21.073, winograd 21.512), B=2048 24.983
(winograd/matrixsimd tied 21.716/21.718).  Node picks: `xl 512t` at B=1,
`xl 512t dy` at B=8/256/2048, pf=0 pfw=0 everywhere — so r9's dy bet was
PICKED at batch and recovered the r8 regression (B=2048 −2.8%, B=256 −1.5%
on medians), while the joint (variant,pf,pfw) grid's sp bet was DECLINED 6/6
(the r9 VERDICT prices it as "the round's largest single divergence":
wallaby −12.4%, node 0).  L17_winograd's q+pfw died identically (declined
6/6), so the VERDICT's synthesis is one-directional: **the L=17 write path
does not respond to cross-volume spreading plus prefetchw on this machine;
fund traffic DELETION at B=256/2048 or declare it closed.**  B=1 is formally
closed panel-wide (matrixsimd's b1dec: 1.30–1.34× port floor from L1, phases
additive; winograd's rewrite gate read 37.4% < 42%).  My r9 probe, node:
ph/xp/fu = 14.93/5.48/22.79 (B=256) and 16.38/6.77/23.46 (B=2048) µs/vol —
input-side excess ~+4.9, output-side ~+2.8 over its ~4 µs compute share,
contention +2.4 (B=256) / +0.5 (B=2048).  The VERDICT also names this entry
the consolidation donor; this round is the counter-argument or the
confirmation.

### The diagnosis this round acts on

The rivals and I move the same compulsory DRAM traffic per volume
(~236 KB: read `in`, RFO + write back `out`), yet at B=2048 they run at
10.9 GB/s effective and I run at 9.4.  The one traffic term unique to me:
my x pass emits `out` through 17 CONCURRENT k-row streams of
16-byte-aligned 128-byte pieces strided 4624 B apart.  A piece at 16-B
misalignment touches 3 cache lines; the boundary lines are completed only
by the same stream's NEXT kernel block (~300 cycles later); and 17 live
streams exceed the ~12 fill buffers — so write-combining entries are
evicted half-filled and the same `out` line is RFO'd and written back MORE
THAN ONCE.  Both rivals write finished planes densely (matrixsimd's chunk
store, winograd's g8) and pay none of this.  ~30–50 KB/vol of refetch waste
closes my 9.4-vs-10.9 GB/s gap arithmetically.  Deleting it is exactly the
funded direction.

### What changed (kernel arithmetic untouched: 296 FP / 488 flops per 17-pt)

1. **"st" / "st dy" (L17R_FORCE 13/14): staged dense out flush.**  The
   mixed x pass stores into a 78.6 KB L2-resident staging volume `vo`
   (same kernel, same operands, base pointer swapped), then `vo_flush`
   emits the volume to `out` as ONE sequential dense stream of full lines
   (1228 zmm load/store pairs + 2 scalars; pfw composes as one prefetchw
   per line, 512 B ahead).  Costs ~157 KB of extra L2 round trip and ~2.5k
   movement µops; deletes the partial-line refetch waste.  Values and
   final addresses identical → bit-identical, class A.

2. **"stp" / "stp dy" (L17R_FORCE 15/16): the flush paced under compute.**
   The first forced wallaby A/B showed the immediate flush fully EXPOSED
   (nothing behind it): st +23% at B=2048 streaming.  stp keeps the staged
   x pass and flushes volume b−1 across volume b's plane phase, one 4.5 KB
   sequential chunk (`vo_flush_chunk`, 72 zmm) after each plane's y pass,
   with vo ping-ponged (+78.6 KB scratch).  NOT a rerun of the falsified
   sp: sp's paced pieces were kernel blocks with 17-stream scattered
   partial-line stores; stp's are single dense sequential bursts — the
   pattern prefetchers and LFBs handle at line rate.  Bit-identical (out
   writes only reordered across independent volumes; cmp-verified).

3. **Grid partner slot goes to the staged family; sp demoted.**  Stage 1
   ranks all four staged shapes on the streaming arena; the joint
   (variant, pf, pfw) grid races the best of them (by stage-1 time)
   against the incumbent, or the unstaged twin if a staged shape already
   won stage 1.  sp stays a stage-1 candidate but loses its grid
   privilege (node: declined 6/6 in r9).

4. **Flush loops left unrollable** (no `#pragma GCC unroll 1` — they are
   simple 2-op loops that -funroll-loops handles well; the pfw branch is
   hoisted into loop twins so the hot loop is branch-free).

### Operation count

Unchanged per 17-point transform: 296 FP instr (192 FMA + 104 add/sub),
488 flops.  st/stp add zero FP: ~2456 movement µops + ~157 KB L2 traffic
per volume (~0.4–0.9 µs equivalent), against the targeted ~30–50 KB/vol of
DRAM refetch waste (~2.5–4 µs at node streaming bandwidth).  I-footprint
(objdump, wallaby): exec_stm_w8 6326 instr ≈ 25 KB, exec_stpm_w8 6687 ≈
27 KB, dy twins slightly smaller — all under r2's 38 KB kill line;
exec_npm/npmdy byte-comparable to r9 (6507/6401).

### Measured — wallaby (Gold 6448Y, gcc 11.4, panel flags; clock bimodal as
### always; forced tables are same-window pinned alternations, min over 3)

Forced B=2048 (`taskset -c 17`, µs/call over 2048 vols; in+out 314 MB):

| config | rep1 | rep2 | rep3 |
|---|---|---|---|
| F2  (xl 512t, incumbent) | **30144** | **29827** | **29753** |
| F13+pfw (st, pfw=1) | 36572 | 36551 | 36722 |
| F15 (stp) | 37429 | 37822 | 37540 |
| F15+pfw | 38778 | 37608 | 37434 |
| F15+pf+pfw | 37910 | 37450 | 37490 |

**Every staged shape loses ~22–27% forced on wallaby streaming.**  Expected
direction, surprising magnitude: wallaby (DDR5, deeper LFB pool, 3 load
ports) pays no partial-line tax, so staging is pure overhead there — but
the overhead reads ~3.5 µs/vol, larger than the µop+L2 estimate.  In the
autotuned B=2048 grid the same window read incumbent 15.32 (pf=1: 14.59)
vs stp 18.30 → stp+pf+pfw 16.48 — i.e. **pfw is worth −9.5% ON the staged
flush** (the dense stream responds to write-intent prefetch exactly as the
mechanism predicts), it is the staging cost wallaby refuses to fund.  At
B=8 (cache-resident) staging reads +1.7 µs/vol (85.97 vs 72.67 forced).

Autotuned end-to-end (tryout.sh; picks in parentheses):

| case | r9 best (same machine) | this round |
|---|---|---|
| B=1 | 8.87 | **8.847** (xl 512; B=1 path untouched) |
| B=8 | 9.23 | 9.25–9.56 µs/t (window spread) |
| B=64 | 10.38 | 10.68 µs/t |
| B=256 | 10.59 | **10.78** µs/t (xl 512 pf=0 pfw=0; grid kept incumbent 10.78 vs st+pfw 12.44) |
| B=2048 | 15.01 | **16.30** µs/t contended window (xl 512 pf=1 pfw=0; grid kept incumbent 14.59 vs stp+pf+pfw 16.48 in-arena) |

The tuner machinery behaves exactly as designed on a machine where the bet
is wrong: all four staged candidates ranked, the best raced through the
grid with (pf, pfw), the incumbent kept, defaults unchanged — **zero
regression risk if the node also declines.**

Correctness, all PASS with the r5–r9 fingerprints preserved: rel_l2 =
3.114e-16 (B=1), 3.171e-16 (B=3), 3.151e-16 (B=8), 3.158e-16 (B=64),
3.153e-16 (B=256), 3.155e-16 (B=2048); bitwise repeatable at every batch;
full-output cmp vs forced 512t: st (F13) and st dy (F14) at B=8, stp (F15)
and stp dy+pfw (F16) at B=8, stp+pfw at B=3 (odd batch exercises the vo
ping-pong parity), UBSan build of stp+pfw at B=8 — all identical;
`-fsanitize=undefined` clean; `-Wall -Wextra` silent (one unused-parameter
warning found and fixed during the round); AVX2 host (wombat) verified
end-to-end (30.2 µs/t at B=8, PASS, repeatable, EVEX candidates
self-eliminate).  Setup: 0.29 s at B=1, 1.1–1.3 s at B=256, 5.9 s at
wallaby B=2048 (977-vol arena; node arena stays 384 → roughly half).

### What was tried and did NOT work — with the numbers

1. **The immediate (un-paced) flush is a fully exposed serial burst**:
   forced st at wallaby B=2048 35.5–36.6 ms vs incumbent 28.7–29.9 — the
   scattered stores it replaces were at least riding under the x-pass FMA
   stream.  stp exists because of this measurement.
2. **No staged shape wins anywhere on wallaby** (tables above) — recorded
   so nobody deletes them for the wrong reason: the machine that ranks
   them honestly is the node, whose 2-load-port/12-LFB/DDR4 memory system
   is where the deleted waste lives.  Note the precedent record is mixed:
   r4's mixed tail was a wallaby-loser the node took everywhere; ov (r5),
   dz (r7) and sp (r9) were wallaby-losers the node also declined.  This
   one differs from ov/dz/sp in kind: it deletes traffic rather than
   rescheduling work, which is the only mechanism class with L=17 node
   wins (winograd g8, rader r8 transposes at B=1/B=8).
3. **The raw-ssh missing-`cd` trap fired THREE more times, on me, in one
   round** — including once with a STALE /tmp binary silently executing
   instead of the fresh build (caught because setup=0.014 s was
   impossible), and twice while my own message text said "cd first".
   matrixsimd's r9 observation ("re-reading your own composed command is
   unreliable once you believe it is fixed") is exactly right.  Durable
   fix used for everything afterwards: write the command sequence to a
   script file on the shared filesystem with `cd` baked in, run the
   script by absolute path.  Recommend the panel adopt the script form
   for ALL remote build/measure sequences, not just tryout.sh.

### Borrowed this round (attribution)

* **L17_matrixsimd**: the dense chunk-store principle itself — their r3
  X-first reorder (−10.8% at B=256 on the node) is the standing proof
  that dense finished-region stores beat scattered ones on this machine;
  st/stp is that principle retrofitted to my X-last structure without
  changing bit class.
* **L17_winograd**: g8 (delete a transposed store, −8–10% all cells) as
  the other half of the "dense stores win" evidence; and their r9 q+pfw
  null, which (with my sp null) is why this round deletes instead of
  spreads.
* **Monitor's r9 VERDICT**: the funded direction ("traffic deletion at
  B=256/2048"), the 10.9 GB/s effective-bandwidth framing that the
  diagnosis rests on, and the donor warning this round answers.

### Expectations for the node (pre-registered)

* **The bet fires**: the grid picks `st`/`st dy`/`stp`/`stp dy` (most
  likely with pfw=1 — the wallaby grid already shows pfw pays ON the
  flush) at B=256 and/or B=2048.  Arithmetic: waste deleted ~2.5–4 µs/vol
  minus staging cost ~1.5–2 µs/vol → **B=2048 from 25.0 toward
  22.5–23.5; B=256 similarly**.  That would be the first node-positive
  batched mechanism for this entry since r4 and closes half the gap to
  the tied leaders.
* **The bet dies**: picks stay `xl 512t dy pf=0 pfw=0` and the cells stay
  ~24.6/25.0.  Then the partial-line-waste theory joins spreading in the
  falsified list, the 9.4-vs-10.9 GB/s gap must be input-side or
  intrinsic to the extra pass structure (my A round trip), and the honest
  position is the r9 VERDICT's: this slot is the donor.  Say so.
* **B=1/B=8 flat by design** (~16.5/18.1; paths untouched).  The B=1
  stage now ranks 4 more candidates (~+25 ms plan time); picks should be
  unchanged.

### Next (in order)

1. Read the node's r10 picks and grid tables.  If st-family is picked
   with pfw=1, try upgrading the flush to `vmovntpd` on the 64-B-aligned
   interior (NT is dead for SCATTERED stores 4 rounds running, but a
   single dense aligned stream is the one shape it was designed for —
   worth exactly one gated candidate, no more).
2. If st-family is picked at B=2048 but not B=256: the B=256 contention
   term (fu−ph−xp = +2.4) is the difference; consider staging `in` reads
   the same way (dense per-volume copy-in) only then.
3. If everything is declined again: recommend the panel take the r9
   VERDICT's consolidation (this slot to L=64 or L=13) — four rounds of
   node-declined batched bets from this structure is the evidence, and
   the leaders sit at the bandwidth floor where my remaining deficit
   lives in structure, not knobs.

---

## Round panel_r11 (2026-08-22)

### Standing going in (panel_r10 node leaderboard)

3rd in all four cells: B=1 16.515 (matrixsimd 14.995, winograd 16.451), B=8
18.137, B=256 23.874 (matrixsimd 21.001, winograd 21.526), B=2048 24.608
(matrixsimd/winograd tied 21.62/21.65).  Node picks: `xl 512t` at B=1,
`xl 512t dy` elsewhere, pf=0 pfw=0 -- the r10 st/stp staged-flush bet was
DECLINED 6/6, exactly its own "bet dies" branch, the fifth consecutive round
of node-declined batched mechanisms from this structure, and the r10 VERDICT
names this entry the donor.  But three r10 VERDICT facts re-arm the round:

1. **The batched L=17 cells are NOT bandwidth-closed.**  matrixsimd's sbw
   probe: the node moves this traffic mix at 17.9 GB/s (cp = 13.2 us/vol)
   where the cells run at 10.9 -- ~8 us/vol of headroom, localized to the
   write/overlap side.  The monitor calls it "the highest-value item on the
   whole board."
2. **My stp was the right construction on the wrong baseline.**  VERDICT §6:
   the staged+paced dense flush is worth one candidate *in a dense-store
   structure*; my X-last x pass emits 17 concurrent misaligned streams, so
   staging behind it deleted a waste term the node does not charge (its
   s17/rd = 0.81-0.83: interleaved streams are CHEAPER than sequential
   there -- the sign inversion vs wallaby's 1.28-1.31).
3. **My own r9/r10 probe localizes MY batched excess on the input side**:
   node ph/xp/fu = 14.6/5.4/22.2 (B=256), 16.2/6.8/23.3 (B=2048) us/vol --
   the plane phase carries the cold `in` read fully UNOVERLAPPED (its
   deint is a pure movement loop with no FMA stream to hide misses under),
   while the x pass's scattered writes ride fine.

### What changed (kernel arithmetic untouched: 296 FP / 488 flops per 17-pt)

**The X-first class was rebuilt around a staged dense per-plane flush
("xfs") -- the leaders' structure retrofitted into class B.**  X-first
attacks exactly both measured exposures at once: the cold `in` reads move
into the x-pass kernel's deinterleaving loads, where they hide under
296-cycle FMA drains (and land on the read shape the node prices at 0.82x
sequential), and the output leaves as dense finished planes (the only store
shape that has ever won an L=17 batched cell on the node -- matrixsimd r3
X-first -10.8%, winograd g8 -8..-10%).  My shipped xf had the RIGHT read
side since r4 but the WRONG store: the y pass wrote out[kx] directly
through 17-row strided 16-B-aligned 128-B partial-line pieces from inside
the compute loop, and with that store it lost the plan-time class race six
rounds running.  Three changes:

1. **Staged y store + deferred dense flush.**  The y pass now stores its
   interleaved output into an L1-hot 4.6 KB staging plane (double-buffered
   by kx parity, carved from the existing vo scratch at +0/+8 KB -- no new
   allocation); the previous plane's staged output is flushed to out as ONE
   sequential dense 4.6 KB stream (72 zmm pairs + 2 scalars, `pl_flush`) at
   the TOP of the next plane's compute, so its RFO misses resolve under
   ~1.5 us of independent kernel work.  This is r10's stp construction at
   plane granularity, applied where dense finished regions exist.  Values
   and final addresses unchanged: bit class B is preserved, all class-B
   candidates cmp-identical (verified, below).  The strided-store xf is
   deleted -- six rounds of losses is enough; the class race now races the
   right baseline.
2. **pfw = one-plane-ahead write-intent prefetch** (74 prefetchw covering
   plane kx's out region, issued at the top of plane kx's compute, one full
   plane ahead of that region's flush) -- the L8_fusedaxes/L36_pfa pacing
   discipline mapped onto the plane pipeline.  Rides the existing pfw knob.
3. **Class B rides the joint (variant, pf, pfw) grid** (panel_r9's L23_rader
   lesson one level up): stage 1's class race at (0,0) can never find
   xfs+pf+pfw, which is the DESIGNED combination -- and on wallaby streaming
   it is indeed xfs's best config (17.0 vs 17.4 plain).  The grid now races
   incumbent-A, staged-A partner, and stage-1's best class-B candidate at 4
   configs each (12 cells, 2 sweeps, blocked); a cross-class pick needs the
   3% margin against the best CLASS-A config, so bits change only when xfs
   wins the honest joint race.  Off under L17R_FORCE.
4. **The stage-1 class race is now published**: `xrace xl/xfs=<a>/<b>` in
   the description string -- six rounds of silent declines carried zero
   information; from this round the monitor (and r12) can read the margin.

### Operation count

Unchanged per 17-point transform: 296 FP instr (192 FMA + 104 add/sub),
488 flops.  xfs adds pure movement: 17 staged-plane round trips = 78.6 KB/vol
of L1 traffic + ~2.5k movement uops (74 zmm+2 scalar loads/stores x 17), and
deletes the partial-line scatter into DRAM-destined out.  I-footprint
(objdump, skylake-avx512 build): exec_xfm_w8 32633 B (r10: 32183), xfpinm_w8
33062 (32543), xf_w8 20450, class-A bodies byte-identical -- all under r2's
38 KB kill line.

### Measured -- wallaby (Gold 6448Y, gcc 11.4, panel flags)

Correctness, all PASS:

* Class A untouched: fingerprints 3.114e-16 (B=1), 3.151e-16 (B=8),
  3.153e-16 (B=256), 3.155e-16 (B=2048) -- identical to r5-r10 at every
  batch; autotuned picks keep class A on wallaby (correct there);
  bitwise repeatable everywhere.
* Class B (forced via -DL17R_XF_CUT): PASS 3.258e-16 (B=8), 3.270e-16
  (B=3, odd batch exercises the staging parity + final flush); repeatable;
  **full-output cmp: xfs 256 = xfs 512 = xfs 512t = xfs 512t pin = xfs+pfw
  = xfs+pf, all IDENTICAL** (6-way, B=8).
* `-fsanitize=undefined` clean (forced xfs+pf+pfw, B=8); `-Wall -Wextra`
  silent at native and skylake-avx512; AVX2 host (wombat) verified
  end-to-end both classes (30.7/33.7 us/t at B=8, PASS, repeatable).

Performance (same-window comparisons; wallaby clock bimodal as always):

| case | class A (unchanged) | xfs forced, best config | note |
|---|---|---|---|
| B=8 (L2-resident) | 9.37-9.51 us/t | 9.61 (xfs 512) | old strided xf: +11-60% |
| nv=256 stage-1 (L3-res.) | 10.76 (xl 512) | 11.56 (xfs 512t pin) | -7%, was -11% in r4 |
| B=1024 streaming, pinned | **14.40-14.44 (xl 512t)** | 17.0 (xfs 512t +pf+pfw) | xfs -18% on wallaby |
| B=1 / B=256 / B=2048 autotuned | 9.84 / 10.84 / 16.5 us/t | -- | class A kept, no regression |

**Wallaby says xl; wallaby cannot rank this shape** (see the XF_CUT comment
in the file): the xfs x pass reads `in` through the 17-row-stream shape that
wallaby's prefetchers price 1.28-1.31x adverse and the node measures at
0.81-0.83x FAVORABLE (matrixsimd's sbw, the largest cross-machine inversion
the panel has measured), and the dense finished-plane store is the shape
that won both rivals' node cells while losing on wallaby.  Like r4's mixed
tail (wallaby -3%, node picked it 4/4 at -2.5..-5.2%), this ships as a
measured node bet: the class race + grid run on the scoring machine at plan
time, the 3% margin protects the incumbent, and zero regression is possible
if the node declines.  Unlike six rounds of silent declines, the margin now
comes back in the description string.

### What was tried and did NOT work / caveats

1. **pfw forced on xfs at cache-resident B=8 on wallaby: +108%** (169 vs
   81 us/call, sd 0.08%, reproducible) -- prefetchw across a resident
   working set is far worse than L36_pfa's +13% uop-tax figure suggests on
   SPR.  At streaming it is mildly positive (17.38 -> 17.17 with pf).  No
   exposure: class B is never used below batch 64 and the grid measures pfw
   only on the streaming arena; recorded so nobody wires pfw into a
   resident regime unmeasured.
2. **xfs on wallaby loses everywhere** (numbers above) -- recorded so
   nobody deletes it for the wrong reason: the two mechanisms it rides
   (favorable 17-stream reads, dense-store advantage) are node properties
   wallaby measurably inverts.
3. The raw-ssh missing-`cd` trap fired a FIFTH time (B=3 verification run
   silently produced nothing).  The absolute-path tryout.sh form fixed it
   again; the durable fix remains "scripts with cd baked in, run by
   absolute path."
4. All inherited dead ends stand (st/stp on the X-last baseline -- now
   understood via the r10 VERDICT as deleting a waste term the node does
   not charge; sp/ov/dz scheduling; NT stores; slab lane-packing;
   transpose fusion into stores; negacyclic splits; non-inline kernels).

### Borrowed this round (attribution)

* **L17_matrixsimd**: the whole bet rests on their panel_r10 sbw
  measurement (cp=13.2 -> the cells are open; s17/rd=0.82 -> the xfs read
  shape is favorable ON THE NODE) and their r3 X-first + dense chunk-store
  node win; the finished-plane store shape is theirs (and L17_winograd
  g8's) structurally.
* **L23_rader (panel_r8)**: the joint-grid principle, extended this round
  across the class boundary (xfs+pf+pfw is findable only jointly).
* **Monitor's r10 VERDICT §6**: the "right construction, wrong baseline"
  reading of my stp, inverted here into "rebuild the baseline, keep the
  construction."

### Expectations for the node (pre-registered)

* **The bet fires**: `xrace` shows xfs within or past the 3% margin at
  nv=384, the grid picks `xfs 512t (pin)` with some (pf,pfw) at B=256
  and/or B=2048, fingerprints at those cells change to the class-B values
  (3.2-3.3e-16 range -- the correctness check still passes, same tolerance),
  and the cells move from 23.9/24.6 toward ~21-22.5: the input-side
  exposure (~+4.9 us/vol, my probe) is the prize, minus the staging cost.
* **The bet dies**: picks stay `xl 512t (dy)`, cells stay ~23.9/24.6 --
  but the description now carries `xrace xl/xfs=...`, so r12 learns the
  actual margin instead of a sixth silent decline.  If xfs loses by <10%
  on the node (vs 18% on wallaby), the structure is right and the residue
  is my plane phase's cost (2 transposes/plane vs the rivals' fused
  chunks); if it loses by ~18% too, the s17/rd argument does not transfer
  to MY x pass (its 17 streams carry 2 loads per row against sbw's 1) and
  the donor recommendation stands with evidence.
* **B=1/B=8 flat by design** (~16.5/18.1): class A is byte-identical, the
  B<64 tuner path is untouched, B=1 stays closed panel-wide.

### Next (in order)

1. Read the node's r11 picks, `xrace`, and (if xfs was picked) the new
   fingerprints.  Every branch above is decidable from the leaderboard
   JSONs alone.
2. If xfs wins batched: fuse the xfs plane phase's two transposes the way
   the read side now permits (T is fed from A, not from cold `in`, so the
   dz-style one-plane software pipeline -- declined on the X-last baseline
   -- has a different cost structure here); and try the staged flush at
   TWO-plane granularity (9.2 KB bursts, half the flush-loop overhead).
3. If xfs loses by <10%: attack the xfs plane phase cost directly (it is
   pure L2-hot work; the 8x8 zmm transposes already run there) before
   conceding the slot.
4. If xfs loses by ~18%: recommend the consolidation (r9/r10 VERDICTs) --
   at that point every structural direction the leaders' node wins point
   to will have been priced in this structure and declined.
