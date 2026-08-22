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

---

## Round panel_r2

### Where round 1 left me (node numbers, panel_r1)

B=1 **0.573 µs** (three-way statistical tie with L8_fusedaxes 0.570 and L8_radix8 0.576,
all ahead of MKL 0.651); B=64 0.640 (2nd); B=2048 **1.432, losing to MKL's 1.338**;
B=16384 1.782 (tied with mkl2026 1.772). So: compute-bound regimes fine, large-batch
bandwidth regime the weak spot.

### What changed this round

**1. Non-destructive 8×8 transpose network — borrowed from L8_fusedaxes round 1.**
My round-1 "Next" item 1 (kill the `vpermt2pd` register copies, ~10% of the port bound)
turned out to be already solved in L8_fusedaxes's record: there is no non-destructive
AVX-512 encoding of a straight r1↔l1 middle level (`vshuff64x2` can only route the
source-register bit to lane bit 2), but the 3-cycle r1→l2→l1→r1 IS encodable, and

```
stage A  r2 <-> l2        vshuff64x2 imm 0x44 / 0xEE   (pairs (i,j),(i,j+4))
stage B  r1 -> l2 -> l1   vshuff64x2 imm 0x88 / 0xDD   (pairs (u,k),(u,k+2))
stage C  r0 <-> l0        vunpcklpd / vunpckhpd        (pairs (w,k),(w,k+1))
```

is a transpose up to the fixed lane permutation **SW = swap(lane bits 1,2)** =
(0,1,4,5,2,3,6,7).  Same 24 shuffle uops, but *all* two-source non-destructive forms with
immediate control: zero index vectors, zero `vpermt2pd`, zero compiler `vmovapd` copies.
I verified the bit algebra by tracking (r2,r1,r0|l2,l1,l0) through the three stages:
element (r|l) of the input lands at register (l2,l1,l0), lane (r1,r2,r0), i.e.
`o[k][l] = i[SW(l)][k]` with the register side the identity.  SW is absorbed at
compile time: BATCH's TSTORE feeds doubles pre-permuted by SW and stores register k to
volume SW(k); LANEX pass 3 renames its transpose outputs `pr[SW(k)] = out[k]` (free —
register renaming) and the final interleave's `vpermt2pd` index vectors are composed
with SW ((0,8,1,9,4,12,5,13) and (2,10,3,11,6,14,7,15)).  Correctness: PASS at
B = 1,2,3,5,7,8,17,64,2048,16384 with rel_l2 1.85–1.93e-16, bit-identical re-runs; the
emulated `-DL8_EMU8` build (which emulates vshuff64x2 per Intel pseudocode) and the AVX2
W=4 build (SW = identity there) also PASS.

**Measured effect (wallaby, Sapphire Rapids, min over repeated runs): LANEX B=1 went
0.629 → 0.306 µs — 2.05×.**  Far beyond the ~10% the copy count predicted; GCC 11's
handling of the index-vector `vpermt2pd` network was much worse than llvm-mca modelled.
Forced-mode measurements after the change: LANEX 0.31 µs/vol at B=8 and B=64 against
BATCH's 0.47–0.49, and at B=2048 LANEX+NT 0.49 vs BATCH+NT 1.21.  **LANEX now wins every
regime on wallaby**, so it is the plan default for all B (BATCH stays as the fallback and
the W=4 path).  The op counts did not change: 1248 vector FP + 896 shuffles + 384 loads +
384 stores per volume (LANEX); BATCH 768 shuffles.

**2. Fused the BATCH y pass into phase 1's z loop** (each 8 KiB z-plane of scratch is
y-transformed while L1-hot, saving one 81 KiB scratch round trip through L2 per block).
A/B forced-BATCH: B=8 3.77 vs 8.03 µs, B=64 31.1 vs 33.6 — kept.  Same instruction count.

**3. Autotune made drift-robust.**  The round-1 tuner timed candidates sequentially
(5 trials each); one noisy block was enough to invert the mode choice — I watched it pick
BATCH at B=64 when a forced run showed LANEX 1.5× faster.  Now the trials are
round-robin-interleaved across all (mode,nt) candidates (min of 7 per candidate — the
idea is from L8_radix8's tuner), hysteresis widened to 3% in favour of the default, and
the default is LANEX/nt-threshold.  There is an env-gated debug print
(`L8_TUNE_DEBUG=1`) that reports all four candidate times per create.

### What was measured (wallaby, Xeon Gold 6448Y, min over ≥3 runs, per transform)

| B | round 1 code | this round | plan chosen |
|---|---|---|---|
| 1 | 0.629 µs | **0.306 µs** | LANEX, no NT |
| 8 | — | **0.311 µs** | LANEX, no NT |
| 64 | 0.599 µs | **0.307 µs** | LANEX, no NT |
| 2048 | 0.453 µs | **0.452 µs** | LANEX, NT |
| 16384 | — | 0.640 µs (1 run) | LANEX, NT |

B=2048/16384 are DRAM-bound on wallaby, so the compute halving does not show there.
rel_l2 = 1.85–1.93e-16 everywhere (tolerance 1e-12).  Spill audit at
`-march=cascadelake`: **zero stack traffic in both kernels** (the only `%rsp` traffic is
in `fft3d_create`); `vpermt2pd` count is down from 128+64 to the 48 unavoidable
interleaves in LANEX pass 3.

**Beware wallaby's bimodality.**  Runs land in two clean modes almost exactly 2× apart
(0.306 vs 0.599 at B=1, 19.6 vs 38.4 at B=64), each with sd < 0.2%.  It is the machine
(SMT sibling / frequency placement on the shared node), not the plan: a debug run showed
the tuner correctly choosing LANEX while the whole process ran 2× slow, tuner
measurements included.  Take the min over repeated runs; the exclusive benchmark node
should not show this.

### What was tried and did NOT work

1. **Double-buffering the LANEX scratch** (alternating 9 KiB halves so volume v+1's
   pass-1 stores don't collide with volume v's pass-3 loads and block cross-volume OoO
   overlap): B=8 2.57 vs 2.49 µs, B=64 20.3 vs 20.0 — neutral-to-worse, reverted.  A
   volume is ~1100+ cycles of work, far beyond the scheduler window, so the "false
   dependency" it removes was never the limiter.
2. **The round-1 sequential autotuner** — see above; replaced, with the failure mode
   documented (picked a 1.5× slower mode at B=64 from one noisy measurement).
3. Not retried, per the records: interleaved-complex lanes, in-lane butterflies, gather
   deinterleave (my round 1); a fused 16-register untranspose+interleave for LANEX
   pass 3 was costed at −128 shuffles/volume but shuffles (896) already sit under the
   node's p0 FP bound (1248), so it can only help 2-FMA parts like wallaby, not the
   scored Gold 5218 — skipped as unscoreable.

### Borrowed / lent

* Borrowed from **L8_fusedaxes**: the entire non-destructive transpose idea, including
  the observation that r↔l1 has no non-destructive encoding and the T3 3-cycle
  workaround.  Their record's mca estimate (−12%) undersold it on real GCC-compiled
  code: −51% on the LANEX path end to end.
* Borrowed from **L8_radix8**: interleaved-trial autotuning to resist frequency/licence
  drift during create.
* For others to take: the SW-absorption trick (any residual lane permutation of a
  shuffle network can be cancelled by compile-time relabels at the ends — it costs
  literally nothing); the tuner-noise failure mode; the wallaby 2× bimodality warning.

### Prediction for the node, and next

On the 1-FMA Gold 5218 the p0 floor is 1248 cycles/volume = 0.54 µs at 2.3 GHz base,
~0.40 at a ~3.1 GHz AVX-512 turbo; round-1 code scored 0.573, so with the copies and
index-vector traffic gone I expect **~0.42–0.50 µs at B=1/B=64** — ahead of the
fusedaxes/radix8 tie if they stand still.  B=2048 is the open question: LANEX+NT streams
16 KiB/volume, and whether that beats MKL's 1.338 depends on the node's achievable
single-core DRAM rate, which wallaby cannot predict.  Next round: (a) if the node
tuner report shows BATCH never winning anywhere, delete the mode machinery and spend the
register budget on a 2-volume LANEX variant; (b) measure whether the node's B=2048 case
wants NT off (32 MiB is only 1.5× L3 — the tuner can already flip it, check what it
chose); (c) the BPY/BPZ=9 padding question is now BATCH-only, i.e. dead unless (a) goes
the other way.

---

## Round panel_r3

### Where round 2 left me (node numbers, panel_r2)

B=64 **0.636 — first**, B=2048 **1.205 — first, ahead of MKL's 1.325** (the first time
the panel took that cell), B=16384 **1.557 — first**.  B=1 **0.598 — third and a real
regression** (+4.4% on my own r1 0.573, against run spreads of 2.2/0.8%), behind
L8_fusedaxes 0.573 and L8_radix8 0.583.  The monitor's verdict on L=8: three entries
stuck within 1.2% for two rounds, instruction-cutting demonstrably buys nothing at B=1,
and my wallaby-2× transpose win transferred to the node as a small loss.

### What changed this round

**1. LANEX restructured from THREE passes to TWO ("LANEX2") — structure adopted from
L8_fusedaxes.**  Their round-1 record measured the exact defect my 3-pass form had:
a separate middle pass costs one full scratch round trip through L1, worth ~2.7% at B=1
(their `fused` vs `mode 1` A/B).  New shape, per volume:

* **pass A, per slow plane (8×):** transposing load (deinterleave AND
  contiguous-axis→registers in one 8×8 network, 48 shuffles) → x DFT → ONE in-register
  transpose pair (48 shuffles, SW residue absorbed by the free rename
  `yr[SW(k)] = u[k]`) → y DFT → 16 stores to the 9 KiB scratch at `(y*9 + s)`.
  104 FP : 96 shuffles — balanced.
* **pass B, per y (8×):** 16 contiguous loads → z DFT (shuffle-free) → copy-free
  interleave (2 permutes/row) → 16 stores straight into the driver's layout.
  52 FP : 16 shuffles.

Per volume: **1248 FP + 896 shuffles + 256 loads + 256 stores** (round 2: 384 + 384 mem
ops).  The fusion deletes 128 loads + 128 stores and ~250 total uops.  Note
L8_radix8's round-2 code is *isomorphic* to this shape (two passes, two transposes per
plane, 52-op codelet) and scored **0.583** on the node against my 3-pass 0.598 — that
isomorphism is the strongest evidence the structure, not the arithmetic, was my B=1
deficit.

**2. Copy-free interleave operand swap — borrowed from L8_radix8 round 2.**  The HIGH
interleave now takes its operands swapped (`VILVHI2(im, re)`) with the index vector
rewritten for the swapped order (`{10,2,11,3,14,6,15,7}` composed with SW), so the two
permutes of a pair destroy *different* sources.  Honest audit of the emitted
cascadelake asm: it did **not** reach radix8's claimed zero — gcc still emits 1 copy per
pair on the `lo` side (~8/volume; it declines the vpermi2pd-with-index-reload encoding)
plus ~16 C-constant copies into FMA destinations that have been there since round 1
(cost parity with the classical 4-mul form: 3 uops per twiddle pair either way).
Total ~24 copies/volume ≈ ≤2% — known, bounded, not worth inline asm this round.

**3. Prefetch hint (t0 vs t1) is now a create-time tuned choice — reacting to
L8_fusedaxes round 2.**  They measured t1 ≫ t0 on wallaby/SPR (t0 fights NT fill
buffers there) while my t0 demonstrably worked on the CLX node (B=2048 1.205).  Both
hints are compiled (5 specialised runners over (nt, pf)) and the tuner measures
LANEX×{nt0/none, nt0/t0, nt1/none, nt1/t0, nt1/t1} (NT candidates only past L3 scale) +
BATCH×{nt0, nt1}.  Prefetch is clamped at the last volume (round-2 code prefetched 8 KiB
past the input mapping — harmless architecturally, now gone anyway).

**4. Tuner pick plumbed into `fft3d_description()`** — the VERDICT's cross-cutting
request #2, the highest-value-per-line change of round 2 by the monitor's own reading.
`fft3d_create()` snprintf's the chosen `mode/nt/pf` and the batch it was tuned for into
the description, so the node's per-case picks are readable from the raw `t_*.json` even
though `leaderboard.py` collapses them to one.

**5. Autotune surrogate cap raised 2048 → 4096 volumes** (64 MiB), so the B=16384 store
policy is tuned on a clearly-streaming arena on the node (22 MiB L3) instead of a
marginal 1.5×-L3 one — this was flagged in round 2's record and it is the failure mode
L36_pfa documented (16-volume arena chose cached stores, the real run wanted NT).

### What was measured (wallaby, Xeon Gold 6448Y SPR, fast-state min over repeated runs;
the 2× bimodality from the round-2 record is still present — mins only)

| B | round 2 code | this round | tuner pick (wallaby) |
|---|---|---|---|
| 1 | 0.306 µs | 0.321 µs (**+5%, see below**) | LANEX nt0 |
| 8 | 0.311 | 0.322 | — |
| 64 | 0.307 | 0.324 | LANEX nt0 pf=t0 |
| 2048 | 0.452 | **0.433** (−4%) | LANEX nt1 pf=t0 |
| 16384 | 0.640 | **0.597** (−7%) | (nt1 path) |

Correctness: rel_l2 = 1.26–1.34e-16 at B = 1, 3, 5, 7, 8, 17, 64, 2048, 16384
(tolerance 1e-12; the value moved from round 2's 1.9e-16 because the axis order
changed), bit-identical re-runs everywhere.  The `-DL8_EMU8` build (which emulates
`vpermt2pd` per Intel pseudocode, so the new swapped index vector is exercised in plain
C), the AVX2 W=4 build and the `-DL8_SCALAR` build all PASS.  Assembly audit at
`-march=cascadelake`: 1 rsp reference per runner (callee-save), 0 spills, 32 vpermt2pd
per runner = exactly the 16 interleaves × 2 inlined bodies, 48 vmovapd copies
(the ~24/volume itemised above).

**The compute-bound cells got 5% SLOWER on wallaby, and that is expected, not noise.**
On 2-FMA SPR the binding resource is combined p0+p5: (1248 FP + 896 shuffles)/2 ports =
1072 cycles/volume for the 2-pass and the 3-pass alike — the deleted 256 memory ops sat
on ports that had slack, so the fusion buys nothing there, and the longer pass-A bodies
(~250 uops, vs the ROB) cost a little cross-iteration overlap.  On the 1-FMA node the
arithmetic is different: p0 = 1248 alone is the floor, my round-2 B=1 sat at ~110% of it
(0.598 µs ≈ 1375 cy at 2.3 GHz) while fusedaxes' fused 2-phase form sat at ~102% — the
overhead was structural, and this round deletes the structural difference.  **Node
prediction: B=1 0.57–0.59** (radix8's isomorphic shape scored 0.583; I carry 48 fewer FP
instructions and the same shuffle count), **B=64 ~0.62, B=2048 ≤ 1.20, B=16384
1.50–1.56** with the tuner free to flip nt/pf on the machine that scores.  If B=1 lands
at 0.59+ anyway, the structure hypothesis is dead too and the residue is the clock (see
Next).

### What was tried and did NOT work

1. **Getting gcc to zero interleave copies via the operand swap alone** — see above:
   8 copies/volume remain (gcc prefers a vmovapd over vpermi2pd + index reload).  The
   fix would be inline asm forcing `vpermi2pd`; costed at ~0.6% of the node floor and
   skipped as not worth the fragility this round.
2. **Chasing the wallaby 5% back.**  Verified it is port-arithmetic, not a bug: the
   2-pass and 3-pass have identical (FP+shuffle)/2 bounds on SPR, and the measured gap
   (0.321 vs 0.306) is consistent with reduced cross-plane overlap in the bigger pass-A
   bodies.  Reverting to 3-pass to please the dev machine would optimise for the wrong
   part — the monitor's r2 verdict says wallaby wins at L=8 B=1 do not transfer.
3. **t1-everywhere** (fusedaxes' SPR result, applied blindly): my own tuner run at
   wallaby B=2048 measured t0 0.721 vs t1 0.832 µs/vol (same state, interleaved trials)
   — t1-wins does not even transfer to a different kernel *on the same machine* at a
   different residency.  Hence hint-as-tuned-candidate rather than hint-as-constant.
4. Not retried, per the records: double-buffered LANEX scratch (my r2: neutral),
   separate shuffle-only output pass (r1: −25%), interleaved-complex lanes, in-lane
   butterflies, gather deinterleave, batch-in-lanes (all killed on counts in r1/r2, and
   fusedaxes/radix8 records agree).

### Borrowed / lent

* **L8_fusedaxes**: the 2-pass fused structure (their r1 fused-vs-3-pass A/B is the
  measurement this round acts on); the prefetch-hint inversion finding that motivated
  tuning the hint; the last-volume prefetch clamp.
* **L8_radix8**: the interleave operand-swap (partial success here, see failures);
  their r2 node score is the isomorphism argument for the whole round.
* For others: the honest copy audit (the swap does NOT reach 0 copies under gcc 11 —
  check your own asm before quoting radix8's number); the surrogate-size rule (tune the
  store policy on an arena ≥ 2× the scoring node's L3, not the dev node's); the
  observation that a fused structure can measure *worse* on a 2-FMA part and still be
  the right choice for a 1-FMA part — port arithmetic, not wallaby timings, decides.

### Next

1. **Read the node's tuner picks out of `t_*.json`** (descriptions now carry them).
   If BATCH is never picked at any B, delete the whole BATCH/staging machinery next
   round (~40% of the file) — keep the W=4 path only for local sanity.
2. **If B=1 is still ≥ 0.59 µs**: the structure hypothesis is exhausted; the open
   variable is the actual core clock under this code (powersave governor, licence
   levels).  Ask the monitor for one `perf stat -e cycles,ref-cycles` on an L=8 B=1 run
   — the same measurement VERDICT §6 already wants for L=6, and it decides whether
   0.54 µs (2.3 GHz × 1248) is the floor or there is 30% of turbo headroom nobody is
   getting.
3. **Interleave copies via inline-asm vpermi2pd** if the node shows p0/p5 saturated
   (~0.6%, last arithmetic lever left).
4. **B=2048/16384**: if MKL retakes either, the remaining lever is a 2-volume software
   pipeline (pass B of volume v against pass A of v+1) with double-buffered scratch to
   put demand loads under compute — fusedaxes' r2 "Next" item 1; my r2 double-buffer
   null result was for the 3-pass shape, so it is worth one retry in the streaming
   regime only.
