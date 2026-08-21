# L8_batchsimd — strategy record

Geometry: **L = 8** (8³ = 512 complex doubles per volume). Assigned strategy:
**batch-major SIMD — vectorise ACROSS volumes**, and measure what the repack costs.

---

## Round 1 (panel round 1)

### Headline answer to the assignment's question

> "Repack from the driver's volume-major layout into lane-major and back; measure that
> repack cost separately and report it, because it is the whole question."

**The repack costs zero cycles, if you schedule it against the arithmetic.** At L = 8 the
whole kernel is arithmetic-port bound, the repack is entirely shuffle work, and on
Skylake-SP/Cascade Lake shuffles issue on port 5 while 512-bit FP issues on port 0
(one-FMA-unit SKUs) — different ports. The repack is 768 shuffle uops per volume against
1248 vector FP instructions per volume, so it fits underneath with 480 port-5 slots to
spare. **What is *not* free is getting the shuffles into the same loop bodies as the
arithmetic**: a naive three-pass-then-repack-out structure spends 48 port-0-idle cycles
per (y,z) in a shuffle-only output pass, and that costs ~25%. Fixing it (see
"software pipelining" below) is the single largest structural win in this entry.

### Technique

Split-complex (SoA) straight-line radix-8 codelet applied as `DFT_8 ⊗ I_W`
(§04 §3.1 vector terminal): every scalar operation is exactly one W-wide vector
instruction, zero cross-lane operations *inside* the transform, for all three axes.
Two lane assignments are compiled; `fft3d_create()` times both on a surrogate batch and
keeps the winner (a plan-time decision, so `fft3d_execute()` stays branch-free and
repeatable).

**BATCH** (the assigned strategy, `MODE_BATCH`) — lanes = W consecutive volumes.
* **phase 1** — transposing load fused with the x pass. For each (y,z) of a block of W
  volumes, load the W volumes' 8-complex x-rows (16 zmm), run two 8×8 double transposes
  (48 shuffles), which simultaneously *deinterleaves* re/im and transposes
  volume↔element, then run the radix-8 along x (52 FP) and store. 48 shuffles under
  52 FP → balanced.
* **phase 2a** — y pass, one L1-resident x-plane (10 KiB) at a time. 52 FP, 0 shuffles.
* **phase 2b** — z pass, **software-pipelined against the transposing store of the
  previous y**: `ZPASS(x,y); TSTORE(y-1,x)`. The z codelet is 52 port-0 instructions with
  no shuffles; the transposing store is 48 port-5 shuffles with no arithmetic. Emitted
  adjacent and mutually independent they hide inside each other (both fit in the
  224-entry ROB together). A one-y prologue and a one-y epilogue drain the pipeline.

**LANEX** (`MODE_LANEX`, W = 8 only, because the lane count must equal L) — lanes = the
8 x-values of ONE volume. Needs no batch at all, so it is the path for B < W and the
insurance policy if batch-major loses on the node.
* pass 1: transposing load (the same 8×8 network — it does deinterleave *and* x↔y in one
  go) fused with the **x** transform. 48 shuffles / 52 FP.
* pass 2: **z** transform. Lanes are y, so shuffle-free; the 8 z of one x are 8
  consecutive vectors.
* pass 3: one 8×8 transpose per component turns lanes y into lanes x, the **y** transform
  is then free, and the interleaving store is 2 `vpermt2pd` per row. 64 shuffles / 52 FP
  — the one unbalanced pass in the design.

**The 8×8 transpose network** (24 one-uop port-5 instructions, no index registers beyond
two): treat an element as `(b, j)` = (register index, lane index) and swap the three bit
pairs in turn.
* level 1, `b0 ↔ j0`: `vunpcklpd/vunpckhpd` on register pairs — 8 ops, immediates only.
* level 2, `b1 ↔ j1`: `vpermt2pd` with two index vectors
  `A = [0,1,8,9,4,5,12,13]`, `B = [2,3,10,11,6,7,14,15]` — 8 ops.
  Output `k` uses sources `t[k&~2], t[k|2]` and index `A` if `(k>>1)&1 == 0` else `B`.
* level 3, `b2 ↔ j2`: `vshuff64x2(u[k&~4], u[k|4], 0x44 or 0xEE)` — 8 ops, immediates.
Verified element-by-element by hand and then numerically (see "verification").
The same primitive serves the transposing load, the transposing store, and LANEX's
internal x↔y swap — one network, three call sites.

### Derivation and operation count

Radix-8 by even/odd decimation: `X_k = E_k + W^k O_k`, `X_{k+4} = E_k − W^k O_k`,
`E = DFT4(x0,x2,x4,x6)`, `O = DFT4(x1,x3,x5,x7)`, `W = e^{−2πi/8}`,
`W^0 = 1`, `W^1 = c(1−i)`, `W^2 = −i`, `W^3 = −c(1+i)`, `c = 1/√2`.

| piece | real instructions |
|---|---|
| DFT4 on the evens (4 ± pairs, then 2 ± pairs and 2 free ×(∓i) pairs) | 16 |
| DFT4 on the odds | 16 |
| `X0/X4` from `E0,O0` | 4 |
| `X2/X6` — `W^2 = −i` is a rename plus a sign folded into the ± | 4 |
| `s1 = O1r+O1i`, `d1 = O1i−O1r`, then 4 FMA/FNMA for `X1,X5` | 6 |
| `s3`, `d3`, then 4 FMA/FNMA for `X3,X7` | 6 |
| **total** | **52 = 44 add/sub + 8 fma** |

This is Burrus T7.1 / T9.1 / FFTW `n1_8`'s **4 real mul + 52 real add** (56 ops), with the
two `c(1∓i)` twiddles folded into the closing butterfly as `fmadd`/`fnmadd`. Folding
replaces `2 mul + 4 add` with `4 fma` per twiddle, i.e. **56 → 52 instructions**, and on
this machine instructions are the currency (LITERATURE §2.3), so 52 is the number that
matters. Multiplication by ±i is free in split layout (§3.3 item 4) and radix-8 is full of
them: 3 of the 7 nontrivial twiddle multiplies vanish entirely.

Per volume: 3 axes × 64 pencils × 52 = **9984 real FP ops** → **1248 vector FP
instructions at W = 8**. `python/fft3d.py line_cost()`'s yardstick and the driver's
nominal `5·N·log2 N` = 23040 flops per volume are both ~2.3× this; the real arithmetic is
9984, so quoted GF/s should be read as a comparison ratio only.

Data movement per volume: BATCH **768 shuffles, 512 loads, 512 stores**;
LANEX **896 shuffles, 384 loads, 384 stores**.

### Port model, and why the design looks the way it does

Xeon Gold 5218 is a **one-512-bit-FMA-unit** SKU (Gold 52xx / Silver 42xx family), so all
512-bit FP arithmetic issues on port 0 at 1/cycle = **8 doubles/cycle**, while all 512-bit
shuffles issue on port 5 at 1/cycle. Note that 256-bit code on the same part also gets
8 doubles/cycle (ports 0+1), so **the AVX-512 win at L = 8 is not FP throughput** — it is
(a) half the instruction count, (b) half the shuffles per byte repacked, and (c) 32
registers instead of 16, which is what removes the spills (see below).

Consequently **9984 real ops / 8 per cycle = 1248 cycles/volume is a hard floor** for
row–column radix-8 on this part. That is 543 ns at the 2.3 GHz base clock and ~390 ns at a
~3.2 GHz single-core AVX-512 turbo, against MKL's measured 0.653 µs at B = 1. There is no
arithmetic left to remove (52 adds is Yavne's minimum for an 8-point complex DFT and
FMA makes the 4 multiplies free), so the entire engineering problem at L = 8 is *not
wasting port-0 slots* — hence the pipelining, and hence the refusal to spend port 0 on
anything that could go on port 5 or the load ports.

### What was measured

**Machine A — the dev machine, Haswell Xeon E5-2680 v3, AVX2 only, W = 4, shared
48-thread node, gcc 11.4.0, `-O3 -march=native -mtune=native -fno-math-errno
-funroll-loops`.** Min over 3 independent processes × 12 samples, per transform:

| B | per transform | notes |
|---|---|---|
| 1 | **6.653 µs** | AVX2 has no LANEX (needs 8 lanes), so B=1 goes through BATCH + the zero-padded staging block: 1 useful lane of 4, plus spills. Not representative of the graded build. |
| 8 | **1.448 µs** | one full block, everything L1/L2 resident |
| 64 | **1.574 µs** | |
| 2048 | **1.635 µs** | NT stores on |
| 16384 | **1.902 µs** | NT stores on |

Correctness, every case: `rel_l2 = 1.31e-16 … 1.34e-16` (BATCH),
`1.90e-16 … 2.30e-16` (LANEX, checked through the emulated build). Tolerance 1e-12.

**Machine A, NT stores forced on/off** (this is the bandwidth experiment):

| B | ordinary stores | NT stores | gain |
|---|---|---|---|
| 64 | 108.6 µs | 107.2 µs | none (1 MiB working set — NT correctly not selected) |
| 2048 | 4143 µs | 3481 µs | **1.19×** |
| 16384 | 38412 µs | 33625 µs | **1.14×** |

The mechanism is the write-allocate/RFO read on `out`: 8 KiB read + 8 KiB RFO + 8 KiB
written per volume becomes 8 + 8, a 1.5× traffic reduction. Machine A only realises
1.14–1.19× of that because at 1.6 µs/volume it is still partly compute-bound. On the
scored node, where compute should be ~0.5 µs/volume, the large-B case is squarely
bandwidth-bound (MKL's own 1.349 µs at B=2048 works out to 24 KiB / 1.349 µs = 17.8 GB/s,
i.e. exactly a single-core DRAM stream), so the NT win there should be much closer to the
full 1.5×. Every output store is a full 64-B-aligned cache line, which is Drepper's
condition for write-combining to succeed; NT is never used on the L1-resident
intermediate, and never below a 12 MiB in+out working set.

**Machine A, the pipelining change** (fusing the transposing store into the z pass instead
of running it as a separate shuffle-only pass): B=64 106.96 → 99.70 µs, B=2048
3466 → 3258 µs, i.e. **6–7%** here. On AVX-512 the predicted gain is much bigger
(~1630 → ~1300 cycles/volume, 20%) because on machine A the bottleneck is spills, not
ports.

**Machine B — the scored part, modelled with `llvm-mca -mcpu=cascadelake`** on the actual
`gcc -march=cascadelake` output, with `LLVM-MCA-BEGIN/END` markers around each hot loop.
This is the only quantitative AVX-512 information I could obtain without the hardware.
Note llvm-mca's `cascadelake` model assumes **two** 512-bit FMA units, which a Gold 5218
does not have, so I report both its simulation and my hand recomputation for one FMA unit.

| region | insn/iter | mca cycles/iter | mca p0 | mca p5 |
|---|---|---|---|---|
| `B_phase1` (per y) | 148 | 73.1 | 41.0 | **72.0** |
| `B_phase2a` (per z) | 92 | 28.1 | 28.0 | 28.0 |
| `B_phase2b_fused` (per x) | 183 | 66.1 | 55.0 | **66.1** |
| `L_pass1` (per z) | 144 | 73.1 | 41.0 | **72.0** |
| `L_pass2` (per x) | 88 | 28.1 | 28.0 | 28.0 |
| `L_pass3` (per z) | 168 | 83.1 | 55.0 | **83.0** |

* Two-FMA model (mca as-is): **BATCH 1338 cycles/volume, LANEX 1474.** BATCH wins by 10%,
  and port 5 is the bottleneck everywhere because 512-bit FP shares p5 with shuffles.
* One-FMA model (all 512-bit FP forced onto p0, shuffles on p5, register moves balanced
  across both): **BATCH ≈ 1423 cycles/volume, LANEX ≈ 1428.** = 619 ns at 2.3 GHz,
  445 ns at 3.2 GHz.

Both models agree within 7%, which is the point: the design is balanced enough that the
FMA-unit count barely moves it. Predicted vs MKL (0.653 µs at B=1, 0.708 at B=64,
1.349 at B=2048, 1.812 at B=16384): roughly 1.1–1.5× at small/medium B and 1.3–1.5× at
large B. **AVX-512 timings on the node are new information — no AVX-512 measurement exists
anywhere in the corpus (LITERATURE §4.8 gap 6) — so record whatever the monitor reports
here.**

### Verification

* `check.py` PASS at B = 1, 2, 3, 5, 7, 8, 9, 11, 16, 17, 64, 2048, 16384 — i.e. below a
  full vector, exactly one, and with a remainder, on both paths.
* **The AVX-512 code path is verified without AVX-512 hardware** via a `-DL8_EMU8` build:
  `vd` becomes `struct { double d[8]; }` and `vunpcklpd`, `vunpckhpd`,
  `vpermt2pd` (`permutex2var`) and `vshuff64x2` are emulated *individually and faithfully
  to their Intel pseudocode*, so the composed transpose network, the index vectors, the
  lane-order assumptions and the LANEX pass structure are all exercised. Both
  `-DL8_FORCE_MODE=0` and `=1` pass. A `-DL8_SCALAR` build (W = 1, NGRP = 16) exercises
  the generic index arithmetic.
* Repeatability: an explicit harness calls `fft3d_execute` three times on one plan and
  `memcmp`s the outputs — **bit-identical** at B = 1, 7, 64, 1000, both paths, NT on and
  off. (The driver's own warmup + calibration + 20 samples + final execute already proves
  this over thousands of calls.)
* **Spill audit (§07 §7.8's check):** `grep` for `vmov*` against `%rsp`/`%rbp` in
  `-march=cascadelake` output — **zero** in both `batch_run` and `lanex_run`. On AVX2 the
  same source spills (see below).
* Warning-free under `-Wall -Wextra` on all four instantiations.
* `fft3d_supports()` returns false for 6, 17, 36 and the driver reports it cleanly.

### What was tried and did NOT work

1. **A separate shuffle-only output pass (the obvious structure).** 48 port-5 shuffles per
   (y,z) with nothing on port 0 → 384 wasted port-0 cycles per volume. Hand model 1632
   cycles/volume vs 1296 for the pipelined version; measured 6–7% even on the
   spill-limited AVX2 machine. **Killed by: 25% of the port-0 budget doing nothing.**
2. **LANEX with the original pass order** (cheap deinterleave + y pass, z pass, then
   transpose + x pass + transposing store). The last pass carried 96 shuffles against
   52 FP → 8 × 96 = 768 cycles for one pass, 1600 cycles/volume total. Re-ordering so the
   *transposing* load does double duty (deinterleave AND x↔y in one 8×8 network) and the
   x pass runs first brings it to 1344/1474. **Killed by: 96 vs 52 — a 44-cycle port-0
   hole per z-plane.**
3. **Interleaved complex in the lanes** (one zmm = 4 volumes × (re,im), so a 4×4
   128-bit-block transpose costs 2 shuffles/vector instead of 3). Op count: a complex
   add is 1 vector op for 4 volumes = the same 4 volumes/op as split's 2 ops for 8, but
   ×(±i) costs 2 ops instead of 0 and ×(1∓i)/√2 costs 3 for 4 volumes instead of 4 for 8
   → **36 vector ops per 4 volumes = 9/volume-codelet against split's 7** (+29% FP) to
   save 256 shuffles/volume. In the 1-FMA model that trades 384 port-0 cycles for 256
   port-5 cycles on the port that has slack. **Killed by: +29% on the bottleneck port.**
   This is LITERATURE §4.4's open question, decided by op count in favour of split.
4. **In-register radix-8 along the lane axis** (butterflies with `vshuff64x2`/masked
   add-sub instead of a transpose pair). Costs ~18–24 ops per single 8-point DFT versus
   7 ops/volume-codelet batch-major, i.e. ~160 ops to do what 52 + 96 shuffles does.
   **Killed by: 3× the instruction count.**
5. **Doing the x pass in the interleaved domain before deinterleaving** (radix-2 stage is
   free — `A±B` on the two zmm of a row — then two in-register 4-point DFTs).
   ~5 shuffles + 17 ops per *single* volume-row = ~22 uops/volume-codelet against 7.
   **Killed by: 3× the instruction count**, same reason as (4).
6. **Hardware gather for the deinterleave.** `vgatherqpd zmm` is ~5 uops and ~4 cycles
   throughput per output vector → 512 cycles/volume on the load ports, against 384 port-5
   cycles for the shuffle network — and Intel's own AOS→SOA table has shuffles at 4.9×
   gather on Skylake (§04 §2.5). **Killed before writing: not even close.**
7. **Padding the BATCH scratch (`BPY = BPZ = 9` vs `8`), measured.** At W = 4 (a 32-byte
   granule) it is worth **nothing** — 1.473/1.579/1.550 µs padded vs
   1.491/1.500/1.531 µs unpadded at B = 8/64/2048, i.e. unpadded is marginally *faster*
   and both are inside the ±2% run spread. This is a partial answer to LITERATURE §4.5:
   §04's "padding is mandatory at L=8" is derived for a **64-byte** batch granule, where
   the unpadded x-stride is exactly 64 cache lines ≡ 0 mod 64 sets — one L1 set for all
   8 x values. At a 32-byte granule the strides are half-lines and the hazard is diluted;
   the per-pass working sets (4 KiB per x-plane at W=4) are small enough that 8-way
   associativity absorbs what is left. **I kept the padding**, because it is free here
   (81 KiB of L2 instead of 64 KiB, and every pass touches only a ≤10 KiB slice) and the
   argument does bite at the W = 8 granule that gets graded. **This needs re-measuring at
   W = 8 on the node — it is a two-line `sed` experiment.**
8. **A 256-bit (ymm-under-AVX512VL) variant of the whole kernel, to dodge AVX-512
   licence downclocking.** Not built. The arithmetic: 256-bit on a 1-FMA part gets the
   same 8 doubles/cycle (ports 0+1) but needs 2× the FP instructions, 1024 shuffles
   instead of 768, and 5568 uops/volume against 3040, so the front end (4-wide) becomes
   the bound at ~1392 cycles versus 512-bit's ~1250–1420. Published Cascade Lake 1–2-core
   turbos are ~3.7 GHz AVX2 vs ~3.5 GHz AVX-512, a 5% gap — not the 10%+ that would be
   needed. **Judged a ≤5% question and not worth the self-include machinery**; noted as a
   next-round item since nobody in the corpus has measured it.

### Findings worth carrying forward

* **LITERATURE §4.1 is answered, with numbers.** The batch-major L = 8 codelet's peak
  liveness is 16 data vectors plus temporaries, which is over the AVX2 file and inside the
  AVX-512 file, and the consequence is large. Micro-benchmark (L1-resident, 64
  independent codelets per pass): the AVX2 W=4 codelet takes **18.74 ns = 54.3 cycles at
  2.9 GHz against a 26-cycle port bound**, and its 99-instruction loop body contains
  10 `%rbp` references = **5 spill/reload pairs**, whose 32-byte store-forwarding latency
  lands on the dependency chain. The identical source at W = 8 compiles to **zero** stack
  traffic. So: §01's "for AVX2 the batch-major form of even n = 6 and n = 8 will spill a
  little" is right about the mechanism and much too mild about the cost — it is
  **2.1× on the codelet**, not "a little". The corollary for L = 8 is §3.3 item 3(a) in
  LITERATURE: run it on AVX-512, and treat any AVX2 measurement of an L=8 batch-major
  kernel as a lower bound on quality.
* **Port 5 vs port 0 is the whole game at L = 8, and which one binds depends on the SKU.**
  On a 2-FMA part 512-bit FP shares p5 with shuffles and p5 binds (mca: 72 of 73 cycles in
  phase 1); on a 1-FMA part all FP is on p0 and p0 binds. A design that balances shuffles
  against arithmetic *within each loop body* is within 7% of optimal under both models,
  which is why the autotune barely has to work.
* **Register moves from the destructive `vpermt2pd` are the visible remaining waste.**
  `_mm512_permutex2var_pd(a, idx, b)` overwrites `a`, and level 2 of the transpose uses
  each `t` register as `a` twice, so GCC emits 4 `vmovapd zmm,zmm` per 8×8 transpose. mca
  attributes 13–29 move uops per loop iteration to p0/p5 — **~10% of the 1-FMA bound**.
* **`llvm-mca` is a usable substitute for the missing hardware.** `-mcpu=cascadelake`
  plus `LLVM-MCA-BEGIN/END` inline-asm markers on a scratch copy of the source gives
  per-loop port pressure and dependency analysis, and it is how the pipelining win was
  confirmed before it could be measured. `/opt/software/llvm-18.1.2/bin/llvm-mca`. Any
  implementer targeting the AVX-512 node from the Haswell login node should use it.
* Cross-check on the emulated build: **the LANEX pass order changes the last digit**
  (rel_l2 1.9e-16 vs BATCH's 1.3e-16) purely because the axes are transformed in a
  different order (x,y,z vs y,z,x vs numpy's own). Both are 4 orders of magnitude inside
  tolerance; neither is a problem, but do not read a 1.5× rel_l2 difference between two
  entries as a quality signal at this level.

### Next

In priority order, with why.

1. **Kill the `vpermt2pd` copies.** ~10% of the 1-FMA bound. The clean trick is to make
   the compiler use the `vpermi2pd` encoding (destination = the *index* register) for one
   of each pair, with the index rematerialised by a load — loads are on p2/p3, which are
   at 12 of 58 slots, i.e. free. Getting GCC to choose that encoding is the work; a
   fallback is a tiny inline-asm transpose. Check with `llvm-mca` before and after.
2. **Re-measure the padding at W = 8 on the node** (`sed 's/#define BPY 9/#define BPY 8/'`
   etc., three defines). Two builds, five minutes, and it settles LITERATURE §4.5 for the
   granule that actually gets graded. My W=4 measurement says 0%; the theory says the
   unpadded W=8 x-stride is the `E = 1/64` worst case.
3. **Drain the BATCH pipeline across blocks.** The 8 epilogue `TSTORE`s run with port 0
   idle: 384 of 10992 cycles = 3.5%. Double-buffer the scratch (2 × 81 KiB still fits L2)
   and pair block k's epilogue with block k+1's phase 1.
4. **Rebalance LANEX pass 3** (64 shuffles against 52 FP, the last unbalanced pass, ~7%).
   The only route is software-pipelining pass 3 of volume v against pass 2 of volume v+1,
   which needs two scratch buffers. Worth it only if the node says LANEX beats BATCH.
5. **Build the 256-bit variant and settle item 8 above with a number**, because it is a
   documented corpus gap (§04 §8.1/§8.2, §4.8 gap 6) and it is the one thing that could
   invalidate the "use AVX-512" conclusion for every entry in this panel, not just this
   one. Request Intel SDE (or just measure on the node) so the two widths can be compared
   in one place.
6. **Huge pages for the large-B cases.** Not attempted: the buffers belong to `driver.c`,
   so `madvise(MADV_HUGEPAGE)` is not mine to call. If a future round lets the plan see
   the pointers, an 8 KiB volume is two 4 KiB pages and the L2 streamer restarts at every
   page boundary — the software prefetch in this entry is a workaround for exactly that,
   and 2 MiB pages would be the real fix.
