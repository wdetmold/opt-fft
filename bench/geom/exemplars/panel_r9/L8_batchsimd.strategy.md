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

---

## Round panel_r4

### Where round 3 left me (node numbers, panel_r3)

B=1 **0.570 — first** (fusedaxes 0.577, radix8 0.583, MKL 0.652); the restructure won
the cell back exactly as predicted.  But the 2-pass **lost all three batched cells I had
held in r2**: B=64 0.663 (2nd, fusedaxes 0.642), B=2048 1.283 (2nd, radix8 1.243),
B=16384 1.748 (3rd, fusedaxes 1.580, +12.3% on my own r2 1.557).  The VERDICT's
diagnosis, which I accept in full: I shipped the 2-pass as a *replacement* instead of a
*candidate*, so the node had no way to keep the 3-pass where the 3-pass was winning; and
the underlying mechanism is **output store order** — my r2 3-pass writes the volume
front-to-back sequentially, the r3 2-pass writes it as 8 interleaved 1-KiB-strided
streams, and L8_radix8's opposite move (2→3 passes to make stores sequential, −18.5% at
B=2048) plus §4.3's verdict ("what pays is the order traffic is issued in, not its
volume") nail it.  The VERDICT's §6 instruction for L=8 is explicit: ship both
structures and let the node choose — "the single largest guaranteed gain available
anywhere on the board", since both targets (1.205 at B=2048, 1.557 at B=16384) are my
own already-measured node numbers.

### What changed this round

**1. Three structures compiled, tuner-selected — the VERDICT's process lesson applied
literally ("add candidates; do not replace structures").**

* **LANEX2** — the r3 2-pass, byte-identical hot path.  Node-verified 0.570 at B=1;
  strided output stores.  Stays the B=1 default.
* **LANEX3** — the r2 3-pass, resurrected from `exemplars/panel_r2/L8_batchsimd.c` and
  re-expressed with the current (copy-free, SW-composed) interleave macros: pass 1 per
  slow plane = transposing load + fast-axis DFT to scratch `(c*9+a)`; pass 2 per c =
  slow-axis DFT **in place** in the scratch column, shuffle-free; pass 3 per
  slow-spectral plane s = transpose pair + mid-axis DFT + interleave + **sequential**
  1-KiB plane stores, s ascending.  Node-verified 1.205 µs at B=2048 / 1.557 at
  B=16384 (panel_r2) — still the best numbers ever taken in those cells.
* **LANEX2S** — new this round: a 2-pass with the axis roles swapped so it gets
  LANEX3's sequential stores at LANEX2's memory-op count.  Pass A per **mid row b**:
  transposing load of the 8 rows `(a=0..7, b)` (reads strided 1 KiB apart), fast-axis
  DFT, one transpose pair, **slow-axis** DFT, store scratch `(s*9+b)`.  Pass B per
  slow-spectral plane s: mid-axis DFT (shuffle-free), interleave, sequential stores.
  The strided traffic moves to the *read* side, which pays no RFO and sits under the
  next-volume software prefetch that the batched regime runs anyway.  Same op count as
  LANEX2 (1248 FP + 896 shuffles + 256 loads + 256 stores); the SW-residue algebra
  needed no new pieces (the rename absorbs it identically, and VILVLO/VILVHI2's
  SW-composed indices already match pass B's lane state).

Per-volume counts: LANEX2/LANEX2S **1248 FP, 896 shuffles, 256+256 mem ops**;
LANEX3 **1248 FP, 896 shuffles, 384+384** (the extra 256 are 9-KiB-L1-scratch traffic).
DRAM-facing traffic identical (8 KiB in + 8 KiB out, 16 KiB with NT); only the order
differs.

**2. BATCH deleted from the W=8 build entirely** (my r3 "Next" item 1): the node tuner
never picked it in any cell, 3/3 runs × 4 batches, readable from the r3 description
strings.  It remains the W=4/scalar path (`#if VW != 8`), so the AVX2 and portable
builds still work; the W=8 candidate list halves and the staging machinery disappears
from the graded binary.

**3. Tuner: one untimed state-setting pass per candidate per trial — borrowed from
L8_fusedaxes round 3.**  Plain and NT candidates leave different cache states behind, so
timing candidate c right after c′ biases the sample; each candidate now runs once
untimed immediately before its timed block.  Candidates remain round-robin-interleaved,
min of 7 trials, 3% hysteresis toward the default.  Candidate list: B=1 → the three
structures plain; B>1 → + each with pf=t0; working set > 6 MiB → + nt1×t0 for all
three and nt1×t1 for the two sequential-store structures (11 total).

**4. Defaults are node-verified configurations.**  B=1 → LANEX2/plain (0.570 measured);
streaming (ws > 12 MiB) → **LANEX3/nt/t0** — the exact configuration that measured
1.205/1.557 on the node in r2 — so with 3% hysteresis the node keeps r2's numbers
unless something beats them by a clear margin; in between → LANEX2S/plain/t0 (the
on-paper best; candidates cover both verified shapes if it disappoints).

### What was measured (wallaby, Xeon Gold 6448Y SPR, fast-state min over repeated runs;
the 2× bimodality documented in r2 is still present and the in-tuner candidate ordering
is preserved inside the slow state, so the picks are state-robust)

| B | r3 code | this round | tuner pick (wallaby) | in-tuner runners-up |
|---|---|---|---|---|
| 1 | 0.321 µs | **0.305** | LANEX2 plain | L2S 0.334, L3 slower |
| 64 | 0.324 | **0.306** | LANEX3 nt0/t0 (0.306) | L2 t0 0.322, L2S t0 0.323 |
| 2048 | 0.433 | 0.442 | LANEX2 nt1/t0 (0.440) | L2S nt1/t0 0.448, L3 nt1/t0 0.497 |
| 5632 (node-B2048 analog, 1.5×L3) | — | **0.438** | LANEX2 nt1/t0 (0.449) | L2S 0.455, L3 0.533 |
| 16384 | 0.597 | **0.597** | LANEX2 nt1/t0 (0.446) | L2S 0.454, L3 0.497 |

Correctness: rel_l2 = 1.32–1.95e-16 (tolerance 1e-12) across B = 1, 3, 5, 7, 33, 64,
2048, 5632, 16384; bit-identical re-runs everywhere; all three structures forced and
PASSed individually (LANEX2S/LANEX3 share the C,A,B axis order and give identical
rel_l2 = 1.900e-16 at B=7 — the axis-order last-digit effect from my r1 record); forced
NT+t1 and NT+t0 paths PASS; the `-DL8_EMU8`, AVX2 (wombat) and `-DL8_SCALAR` builds all
PASS.  Warning-free under `-Wall -Wextra` on all four instantiations.  cascadelake asm
audit: **0 spills in all 15 runners**, 32 vpermt2pd per runner (= the 16 interleaves ×
2 inlined volume bodies, unchanged from r3).

### What this round's wallaby numbers do and do not say

Wallaby (2-FMA SPR, 60 MiB L3, full-clock AVX-512) picked LANEX2 — the *strided*-store
structure — in every streaming cell, while the node evidence (r2 vs r3, and radix8's
r3 experiment) says sequential stores win there by 6–12%.  So **wallaby cannot decide
the node's streaming structure**, which is exactly why the streaming default is the
node-verified LANEX3/nt/t0 rather than wallaby's favourite: on the node the tuner
starts from the configuration that measured 1.205/1.557 and must be beaten by >3% in
its own measurement to move off it.  LANEX2S beat LANEX2 on wallaby in the nt0/t0
column at every large batch (0.657 vs 0.691 at 5632, 0.674 vs 0.753 at 16384) and lost
narrowly in the nt1 column — on the node's 1-FMA, RFO-expensive memory system its
sequential writes should be worth more than on wallaby.  Whether LANEX2S or LANEX3
wins the node's streaming cells is the round's real experiment; both are candidates.

### What was tried and did NOT work

1. **LANEX2S as an outright wallaby win in the streaming regime**: it trails LANEX2 by
   1–2% under nt1 there (0.454 vs 0.446 at B=16384 in-tuner) — the sequential-store
   advantage does not show on SPR's memory system, consistent with every store-order
   observation in this file transferring badly *toward* wallaby.  Not a failure of the
   structure; the node decides.  (Numbers above.)
2. **LANEX3 at B=1 on wallaby** measured 0.595 forced vs LANEX2's 0.305 in an
   adjacent-but-possibly-different machine state — radix8's r3 record has the 3-pass
   *winning* B=1 on the same machine, so treat this single number as unresolved wallaby
   noise rather than a fact; the B=1 default (LANEX2) is node-verified anyway.
3. Not retried, per the records: everything in the r1–r3 failure lists (separate
   shuffle-only output pass, interleaved-complex lanes, in-lane butterflies, gather,
   batch-in-lanes, double-buffered LANEX scratch at L1 residency).

### Borrowed / lent

* **The panel_r3 VERDICT §6**: the whole shape of this round — both structures as
  candidates, defaults = node-verified configs.
* **L8_radix8 r3**: the store-order diagnosis (their 2p→3p A/B is the controlled
  experiment my r2→r3 regression mirrors), and their finding that the sequential-store
  3-pass can win even compute-bound cells on some machines — hence LANEX3 is a
  candidate at *every* batch size, not just streaming.
* **L8_fusedaxes r3**: the per-candidate untimed state-setting pass in the tuner.
* For others: **LANEX2S** — sequential output stores are compatible with the 2-pass op
  count if you fuse (fast, slow) in pass A and do the mid axis last; the residue
  algebra is unchanged.  If it wins on the node, the 3-pass's extra 256 L1 ops were
  never necessary for the store-order win.

### Prediction for the node

* B=1 **0.570 stands** (same default, same code path, hysteresis protects it).
* B=64: LANEX3/t0 and LANEX2S/t0 now available; radix8's isomorphic 3p-pf scored 0.671
  there, my r3 LANEX2/t0 0.663, fusedaxes' plain 0.642 — expect **0.62–0.66**, pick
  uncertain.
* B=2048: default = r2's exact winning config → **≤1.21** unless the node tuner finds
  LANEX2S >3% better, in which case lower.  MKL is at 1.335.
* B=16384: default = r2's config → **≤1.56**; LANEX2S is the upside case (fusedaxes
  holds 1.580; the bandwidth bound argument in the VERDICT puts the floor near 1.37).

### Next

1. **Read the node's picks off the descriptions** (`pick[B=…]: mode=… nt=… pf=…` is in
   every t_*.json).  The decisive datum is B=2048/16384: LANEX3 vs LANEX2S settles
   whether the 3-pass's extra L1 round trip costs anything once store order is equal.
2. If LANEX2S wins streaming on the node, delete LANEX3 next round (it is then strictly
   dominated) and spend the freed tuner budget on nt0/t1 columns at mid batch.
3. The un-overlapped-memory lever (2-volume software pipeline, pass B of volume v
   against pass A of v+1, double-buffered scratch) remains the only untried structural
   idea for streaming; the VERDICT's bandwidth arithmetic says B=16384 is within 16% of
   the stream bound, so it pays at most ~0.2 µs — attempt only if the cells are still
   lost after this round.
4. The `perf stat -e cycles,ref-cycles` clock question (r3 Next item 2) is still open
   and still owned by the monitor; it bounds what B=1 work is worth doing at all.

---

## Round panel_r5

### Where round 4 left me (node numbers, panel_r4)

B=1 **0.570 — tied first** with L8_radix8 (fusedaxes 0.579, MKL 0.654); LANEX2/plain,
3/3 picks, exactly as designed.  Every batched cell lost: B=64 **0.665 — second**
(fusedaxes 0.623 with plain/no-pf), B=2048 **1.215 — second** (radix8 1.136), B=16384
**1.642 — third** (radix8 1.418, fusedaxes 1.585).  My node picks (r4 t_*.json):
B=64 LANEX2/2S + nt0 + **burst t0** (the pick flipped 2/2S between runs, cost ≈ spread);
B=2048 and B=16384 **LANEX3 + nt0 + burst t0, 3/3** — the tuner correctly abandoned the
NT default I shipped (so "NT loses on this node's streaming cells" is now MY OWN tuner's
measured verdict too, not just radix8's r3 reading), but the only prefetch shapes I gave
it were burst t0 / burst t1 / none.  Meanwhile L8_radix8 won both streaming cells with
`avx512-3p-pfs` = **spread prefetch + plain stores** (their kernel is isomorphic to my
LANEX3 — the VERDICT §3c even shows our outputs are bit-identical), 3/3 picks, beating
their own burst-plain candidate on the node itself.  So the whole streaming gap
(1.215 vs 1.136, 1.642 vs 1.418) is attributable to ONE variable: prefetch placement.
The VERDICT's promotion note says my instructive datum was that shipping r2's winning
config as default still landed 5.5% short at B=16384 — the residue was never the store
policy, it was the 16-line prefetch bursts stalling the pass-1 demand loads.

### What changed this round (one mechanism + the plumbing it needs)

**1. SPREAD prefetch — borrowed from L8_radix8 round 4, including the fill-buffer
rationale from their record.**  A new plan-time prefetch mode `PF_S0`: the same 128
cache lines of volume v+1 are issued a few per loop iteration across the WHOLE volume
instead of a 16-line plane burst at the top of each pass-A/pass-1 iteration —
LANEX3: 6/5/5 lines per iteration of passes 1/2/3 (48+40+40, each line exactly once,
radix8's exact schedule); LANEX2/LANEX2S: 8 lines per iteration of both passes (64+64).
~1 prefetch per 10–12 cycles, never competing with the transposing load's own 16 demand
loads or with an NT drain for fill buffers.  Verified in the cascadelake asm: the
PF_LINES bundles are fully unrolled (6/5/5 and 8/8 static `prefetcht0` per rolled pass
iteration), zero spills in all 15 runners.

**2. Runner set restructured: burst+NT deleted, t1 deleted.**  Per structure the
compiled runners are now plain × {none, burst t0, spread t0} and NT × {none, spread t0}
(15 total, same count as r4).  Burst+NT is the documented fill-buffer clog (radix8 r4);
t1 was never picked by the node tuner in any cell in rounds 3–4.

**3. Regime-gated candidate sets (gate idea from radix8 r4) with an L3-relative gate.**
`sysconf(_SC_LEVEL3_CACHE_SIZE)` (fallback 22 MiB); NT candidates only when
in+out > 0.9×L3.  B=1: three structures plain (default LANEX2, node-verified twice).
Mid batch (B=64 cell): {L2S,L2,L3} × spread + LANEX2/burst (my r4 node config) +
{L2S,L2,L3} × none (fusedaxes' winning B=64 policy) — 7 candidates, default LANEX2S/spread.
Streaming: LANEX3+plain+spread (default — the twin of radix8's node winner),
LANEX3+plain+burst (my r4 incumbent, 1.215/1.642), L2S/L2+plain+spread,
LANEX3/L2S+NT+spread, LANEX3+NT+none — 7 candidates.  On the node no scored cell sits
in the excluded 0.25–0.9×L3 band (B=64 = 0.045×L3, B=2048 = 1.45×L3).

**4. Tuner arena machine-relative: 4×L3 of volumes clamped [4096, 8192]** (radix8 r4,
who took it from fusedaxes r3) — 5632 on the node, 8192 on wallaby — so the streaming
policy is tuned on a faithful ≥4×L3 stream on both machines (my old fixed 4096 was only
1.07×L3 on wallaby).

**5. Defaults are all plain-store now**: B=1 LANEX2/plain (0.570 twice-verified),
mid LANEX2S/plain/spread, streaming LANEX3/plain/spread.  NT is a candidate, never a
default — three consecutive rounds of node evidence against it (my r4 picks, radix8's
r4 picks, and the r4 VERDICT's "the NT variant lost again").

Arithmetic, transposes, interleaves, scratch layout: untouched.  1248 vector FP +
896 shuffles per volume (LANEX2/2S: 256+256 mem ops, LANEX3: 384+384); the spread
runners issue the identical 128 prefetcht0 uops per volume, only distributed.

### What was measured (wallaby, Xeon Gold 6448Y SPR, fast-state min over repeated runs;
the 2× bimodality documented since r2 is still present)

| B | r4 code | this round | wallaby tuner pick | notes |
|---|---|---|---|---|
| 1 | 0.305 µs | **0.305** | LANEX2 plain | byte-identical path |
| 64 | 0.306 | **0.306** | LANEX3 nt0/**s0** | in-tuner: L3/s0 0.597 < L3/none 0.613 < L2S/s0 0.626 |
| 2048 (0.53×L3 → mid set here) | 0.442 | 0.451 | LANEX2 nt0/t0 (0.806 vs s0 0.821) | wallaby-only regime, see below |
| 5632 (1.47×L3, node-B2048 analog) | 0.438 | **0.429** | LANEX3 **nt1/s0** (0.440; nt1/none 0.485, L2S nt1/s0 0.455) | spread worth 9% on top of NT here |
| 16384 (4.4×L3) | 0.597 | **0.597** | LANEX3 nt1/s0 (0.643; nt1/none 0.802, plain 1.2–1.36) | |

Correctness: rel_l2 = 1.32–1.92e-16 (tolerance 1e-12) at B = 1, 3, 7, 17, 64, 2048,
5632, 16384; bit-identical re-runs everywhere.  All new runner combinations forced and
PASSed individually (each structure × spread, NT+spread for all three, burst for L3).
`-DL8_EMU8`, AVX2 (wombat) and `-DL8_SCALAR` builds PASS.  Warning-free under
`-Wall -Wextra`; cascadelake asm audit: 0 spills in all 15 runners.

### What this round's wallaby numbers do and do not say

* **Wallaby's plain column prefers BURST over spread** (L3: t0 0.884 vs s0 0.985 at
  B=5632; t0 1.200 vs s0 1.347 at B=16384) — the opposite sign to radix8's r4 node
  tuner, which picked spread-plain over burst-plain 3/3 in both streaming cells and won
  them with it.  Yet under NT, wallaby says spread ≫ none (0.440 vs 0.485).  So
  "spread-plain beats burst-plain" is a NODE fact (measured there by radix8's tuner, on
  a kernel bit-identical to mine), not a universal one — one more entry for the
  store-order/prefetch family of results that do not transfer toward wallaby.  Both
  shapes are candidates and the node's own tournament decides; the default (spread) is
  the node-verified side, hysteresis 3%.
* Wallaby's B=2048 sits in the mid set by the 0.9×L3 gate (0.53×L3 there), so it never
  sees NT candidates on the dev machine — same accepted trade as radix8 r4: on the node
  that cell is 1.45×L3 and gets the full streaming set.

### What was tried and did NOT work

1. Nothing new failed this round; it was deliberately a single-mechanism round.  The
   near-miss worth recording is above: my spread-plain does NOT win on wallaby's plain
   column, so if the node pick comes back burst-plain (i.e. radix8's result does not
   reproduce under my pass structure), the difference to chase is that LANEX3's pass 2
   is pure L1 compute with idle load ports — radix8's r4 "Next" suggested issuing ALL
   spread prefetches from pass 2; that variant is unbuilt.
2. Not retried, per the records: everything in the r1–r4 failure lists (burst+NT —
   radix8 r4's clog; separate shuffle-only output pass; interleaved-complex lanes;
   in-lane butterflies; gather; batch-in-lanes; double-buffered scratch; cross-volume
   software pipelining — five schemes built panel-wide in r4, zero selected by the node).

### Borrowed / lent

* **L8_radix8 r4**: the spread prefetch schedule (6/5/5 across the three passes), the
  burst+NT fill-buffer clog (their candidate deletion, adopted as a runner deletion),
  the L3-relative candidate gate, and the 4×L3 arena clamp.  Their node win on an
  isomorphic kernel is the entire justification for this round's default.
* **L8_fusedaxes** (via radix8): the original compute-embedded prefetch idea and the
  arena cap; their plain/no-pf B=64 win is why no-pf is a mid-set candidate.
* For others: the burst-vs-spread sign INVERTS between wallaby and the node in the
  plain-store column (numbers above) — do not tune prefetch placement for plain stores
  on wallaby; and the L2/L2S/L3 structure choice at B=64 keeps flip-flopping within
  ~2% on both machines, so treat any single B=64 structure pick as noise.

### Prediction for the node

* B=1 **0.570 stands** (byte-identical path, protected default).
* B=64: spread replaces burst in the pick; radix8's r4 evidence says placement is worth
  little at cache residency, fusedaxes' 0.623 is the target.  Expect **0.62–0.66**;
  structure pick uncertain (2/2S/3 within noise everywhere).
* B=2048: default = the exact winning configuration of the cell (radix8's 1.136 was
  spread+plain on bit-identical arithmetic; my r4 burst+plain measured 1.215).  Expect
  **1.13–1.18**; anything above 1.20 means my pass structure interacts with spread
  differently than radix8's and the pass-2-only placement is the next experiment.
* B=16384: same argument, target **1.42–1.50** (radix8 1.418; my burst 1.642).

### Next

1. Read the node's streaming picks: if spread-plain is picked and ≈1.14/1.42, the
   burst→spread attribution is confirmed on my structure and the L=8 batched story is
   at the radix8 frontier; the remaining ~4% to the 1.365 bandwidth floor at B=16384 is
   read-side scheduling (pass-2-only prefetch placement, unbuilt).
2. If the node keeps burst-plain: the wallaby inversion transferred, radix8's win is
   specific to their pass structure, and the pass-2-only variant is the cheap next probe.
3. B=64 remains structural (fusedaxes' 256+256 single-fused-pass at 0.623); if this
   round's spread does not close it, the honest move next round is adopting their fused
   shape as a fourth structure — a rewrite, only worth it if the cell still matters.
4. The `perf stat` clock question (open since r3) is partially answered by L6_unrolled's
   probe (node AVX2 clock = 3.89 GHz); the AVX-512 licence clock is still unmeasured and
   still bounds what B=1 work is worth doing.

---

## Round panel_r6

### Where round 5 left me (node numbers, panel_r5)

B=1 **0.574 — third in the standing three-way tie** (radix8 0.570, fusedaxes 0.573; all
inside spread).  Every batched cell lost to L8_fusedaxes, and by a lot in streaming:
B=64 0.610 reported (2nd) **but read down by the VERDICT to ≈0.655 and fourth** — the
0.610 came from the one run in three whose tuner picked LANEX3+s0; the two runs that
kept my shipped LANEX2S default measured 0.655/0.660 (fusedaxes 0.594, radix8 0.619).
B=2048 **1.096 vs fusedaxes' 0.910** (−17%); B=16384 **1.388 vs 1.254** (−10%).  My own
round worked exactly as designed — spread prefetch was picked 3/3 in both streaming
cells and beat all three of my predictions (VERDICT §4 scores it "correct and slightly
conservative") — but fusedaxes' `fused+pfs+pfw` moved the frontier further: the first
node selection of **write-intent prefetch** at L=8, confirmed independently the same
round by L36_pfa's `pf=2` (−16.6% at B=256).  The VERDICT's §6 instruction for L=8 is
explicit and is this round's brief: fusedaxes changed pass count AND store policy in one
round, so **add a pfw candidate to L8_batchsimd's LANEX3+s0** and let the node separate
the fusion win from the pfw win.  Also: NT lost on the node for the fourth consecutive
round, everywhere; burst t0 was picked nowhere in r5.

### What changed this round (one mechanism + one default fix)

**1. PF_SW — spread read prefetch + spread WRITE-INTENT prefetch, borrowed from
L8_fusedaxes round 5** (who took prefetchw from L6_unrolled/L6_pfa; L36_pfa confirmed it
independently).  With plain stores every output line pays an RFO read; NT avoids the RFO
and loses on the node (4 rounds of picks), so the winning move is to HIDE it:
`__builtin_prefetch(p, 1, 3)` (emits `prefetchw` on CLX/SPR) issues the next volume's
128 OUTPUT lines one volume early, interleaved at the exact cadence of the existing s0
read prefetch — LANEX3: 6/6, 5/5, 5/5 read/write lines per iteration of passes 1/2/3;
LANEX2/2S: 8/8 per iteration of both passes.  Three new plain-store runners (l2/l2s/l3
`_psw`); NT+pfw is not instantiated (NT avoids the RFO that pfw hides — nonsense by
construction).  prefetchw on cache-resident output is pure uop tax (L36_pfa +11–13%
in-arena, fusedaxes +3%), so PF_SW is offered ONLY in the streaming candidate set.
Streaming set is now: **LANEX3+s0w (default), LANEX3+s0 (my node-verified 1.096/1.388),
LANEX2S+s0w, LANEX2+s0w, LANEX2S+s0, LANEX3+nt+s0 (insurance), LANEX3+none** — with
LANEX2S+s0w as the "fusion arm": LANEX2S is my 2-pass sequential-store twin of
fusedaxes' fused shape, so the node tournament L3+s0w vs L2S+s0w vs L3+s0 is exactly
the VERDICT's isolating experiment run inside one entry.

**2. Mid-batch (B=64) default flipped LANEX2S → LANEX3+s0, on the node's own r5 numbers**
(LANEX3+s0 0.610 measured vs LANEX2S+s0 0.655/0.660 — a 7% gap my wallaby tuner calls a
coin flip; the r2–r5 records say the 2/2S/3 choice at B=64 is noise on wallaby and it
demonstrably is NOT on the node).  Mid set shrunk to 6 (three structures × {s0, none});
burst t0 deleted from every candidate set (zero r5 node picks; s0 won 3/3 streaming);
streaming NT candidates cut to one.

Arithmetic, transposes, interleaves, scratch, B=1 path: untouched — 1248 vector FP +
896 shuffles per volume (LANEX2/2S 256+256 mem ops, LANEX3 384+384).  PF_SW adds 128
`prefetchw` uops/volume on ports 2/3 (+7 with slack); asm audit at `-march=cascadelake`:
warning-free, 16 static prefetchw per rolled psw runner (6/5/5 ✓), **zero vector spills
in all 18 runners** (two 8-byte scalar pointer saves per volume loop in l3_run_psw —
integer bookkeeping, ~4 uops per 1250-cycle volume, ignored).

### What was measured (wallaby, Xeon Gold 6448Y SPR; the documented 2× clock lottery is
still present — same-process in-tuner tables are the only numbers I trust, driver mins
quoted from the fastest window seen)

| B | r5 code | this round | wallaby pick | key same-process in-tuner comparison |
|---|---|---|---|---|
| 1 | 0.305 µs | **0.305** | LANEX2 plain | byte-identical path |
| 64 | 0.306 | **0.306** | LANEX3 s0 (new default, won in-tuner) | L3/s0 0.431 < L2/s0 0.453 < L2S/s0 0.454 < L3/none 0.443* (*ordering stable, L3/s0 first 2/2 runs) |
| 2048 (0.53×L3 → mid set here) | 0.451 | 0.467 | LANEX2 s0 | wallaby-only regime; no pfw by design |
| 16384 (4.4×L3) | 0.597 | **0.540** | LANEX3 **nt1**/s0 (SPR keeps NT, as every round) | **plain column: L3/s0w 0.754–0.764 vs L3/s0 1.066–1.093 → pfw −29/−31%**; L2/s0w 0.725–0.734, L2S/s0w 0.743–0.751 |

The B=16384 in-tuner table, twice reproduced, is the round's result: **on wallaby's
plain-store column pfw is worth −29 to −31% on my structure**, the same sign and
comparable size to fusedaxes' r5 wallaby measurement (plain+pfs+pfw 0.637 vs plain+pfs
1.107, −42%) that correctly predicted their node win.  Wallaby's *final* pick is still
NT (0.452–0.485), which four rounds of node evidence say does not transfer — on the node
the plain column is the contest and s0w owns it here by a wide margin.  All three
2-pass/3-pass s0w variants sit within 4% of each other on wallaby (L2 0.725 < L2S 0.743
< L3 0.754), so wallaby cannot rank the structures under pfw; the node tournament will.

Correctness: rel_l2 = 1.30–1.92e-16 (tolerance 1e-12) at B = 1, 5, 7, 17, 64, 2048,
5632, 16384; bit-identical re-runs everywhere.  Every new runner forced and PASSed
individually (LANEX3+s0w at B=5632/16384, LANEX2S+s0w at B=5632, LANEX2+s0w at B=2048);
the `-DL8_EMU8` build forced through PF_SW on all three structures PASSes (the PFW
macro is a no-op there, so this exercises the branch/index logic in plain C); AVX2
(wombat) and `-DL8_SCALAR` builds PASS.  Warning-free under `-Wall -Wextra`.

### What was tried and did NOT work

1. Nothing new failed; this was again a single-mechanism round by design.  One
   measurement worth flagging as UNTRUSTWORTHY rather than failed: my forced
   cross-process A/B of LANEX3+s0 vs LANEX3+s0w at B=5632 read 1.100 vs 0.574 µs/vol —
   a 1.9× "win" that is mostly the clock lottery (the two runs' MKL companions differ
   1.8×).  Per L17_winograd's r5 proof (2.10 ↔ 4.10 GHz windows), only the same-process
   in-tuner numbers above are quotable.
2. Not retried, per the records: everything in the r1–r5 failure lists (burst+NT clog,
   NT as default, separate shuffle-only output pass, interleaved-complex lanes, in-lane
   butterflies, gather, batch-in-lanes, double-buffered scratch, cross-volume software
   pipelining).

### Borrowed / lent

* **L8_fusedaxes r5**: the entire pfw mechanism — `__builtin_prefetch(p,1,3)`, the
  one-volume distance, pairing it with plain stores + spread t0 reads, and the gate
  that keeps it out of cache-resident regimes.  Their node numbers (0.910/1.254,
  fused+pfs+pfw 3/3) are the existence proof this round acts on.
* **L6_unrolled / L6_pfa** (via fusedaxes) and **L36_pfa** (independent confirmation,
  `pf=2` −16.6%): the RFO-hiding idea itself — the r5 VERDICT elevates it to the rule
  "hide the RFO (prefetchw) rather than avoid it (NT)" for Cascade Lake.
* **panel_r5 VERDICT §3a**: the B=64 default fix — reading my own two-of-three node
  runs as the honest value instead of the lucky min.
* For others: the s0w cadence generalises trivially to any structure that already has
  a spread read prefetch (add the output line at the same index); and one more entry
  for the transfer-warning file — wallaby's tuner still terminates at NT in every
  streaming cell, so a wallaby *pick* is uninformative at L=8 streaming even when the
  in-tuner plain column is decisive.

### Prediction for the node (stated to be scored)

* B=1: **0.570–0.574 stands** (byte-identical path, protected default).
* B=64: the default now IS the configuration that measured 0.610 in r5's lucky run,
  and the set around it is smaller.  Expect **0.60–0.62** with pick LANEX3/s0 3/3; the
  instability cost (VERDICT 3a) should disappear.  fusedaxes at 0.594 stays favourite.
* B=2048: if pfw carries fusedaxes' win, LANEX3+s0w (or L2S+s0w) lands **0.91–1.00**;
  if fusion contributed materially, I land 1.00–1.09 and the residue is the shape.
  Either outcome answers §6's question — the pick string and the gap to 0.910 are the
  two numbers to read.
* B=16384: same logic from 1.388 vs their 1.254: **1.25–1.35** if pfw carries it.

### Next

1. Read the node's streaming pick strings: (a) s0w picked + gap to fusedaxes closed →
   pfw carried it, §4.3's fusion story at L=8 is dead, and the remaining streaming gap
   (if any) is bandwidth-floor territory; (b) s0w picked but 5–10% short of fusedaxes →
   fusion (or their 16-iteration shape's interaction with the prefetch stream) is real,
   and the honest next move is porting the fused phase structure as a fourth mode, as
   radix8 already did in r5 (their `1f`); (c) s0w NOT picked → the pfw win is
   structure-specific, which would be new information for the corpus.
2. B=64: if LANEX3/s0 3/3 lands ≈0.61 and fusedaxes holds 0.594, the gap is their
   fused shape's shuffle placement (16 light shuffles in the load pass, heavy networks
   against L1 scratch — radix8's r5 analysis); the `1f` port covers that cell too.
3. B=1 remains frozen: 0.570 = 1648 cycles at the now-measured 2.89 GHz clk512 vs the
   1248-cycle p0 floor (1.32×).  The one untried compute lever on this part is mixed
   width — move part of the codelet to ymm to co-issue on port 1 (the licence is
   already paid; L17's mixed shapes won −7.4% on exactly this argument).  On paper the
   balanced split is ~832 cycles of FP.  Big rewrite, only worth it if the monitor's
   clock/port data says B=1 is actually port-bound rather than latency-bound.

---

## Round panel_r7

### Where round 6 left me

Nowhere new: **panel_r6 was abandoned between its development and timing phases**
(`results/panel_r6_abandoned_no_timing/WHY.md` — a stale runner was retired; the code
survived as `impl_6/` and became panel_r7's starting point).  So my round-6 work — PF_SW
(spread prefetchw of the next volume's output lines) and the B=64 default fix — was never
node-timed, and the standings are still panel_r5's: B=1 0.574 (third in the three-way
tie), B=64 ≈0.655 read-down (fourth), B=2048 1.096 and B=16384 1.388, both second to
L8_fusedaxes' fused+pfs+pfw (0.910 / 1.254, picked 3/3).  Both L=8 rivals stood still in
round 6 (their impl_6 files are byte-identical to their r5 exemplars), so the frontier is
unchanged and the r5 VERDICT's reading still holds: the fused shape won every batched
cell, and my r6 pfw-on-LANEX arm answers only half the question — on wallaby r6 their
fused+pfs+pfw measured 0.637 against my best LANEX+s0w 0.725, so ~12% of their win is the
shape, not the prefetch.

### What changed this round: the fused shapes are now inside my tournament

**1. MODE_FUSED — L8_fusedaxes' fused structure, ported** (the shape that won node
B=64/2048/16384 in r5; L8_radix8 ported the same thing in r5 as `1f` and fell
0.680 → 0.619 at B=64).  Lane = z (the contiguous axis) throughout, so the DRAM-facing
load pass needs NO transposing network — one z-pencil is 8 contiguous complex and a
single vunpcklpd/vunpckhpd pair splits it re/im (lane l holds z = PI[l],
PI = 0,4,1,5,2,6,3,7):

* pass A per slow plane x (8×): 16 loads, 16 unpck shuffles TOTAL, y-axis DFT across
  registers (shuffle-free), 16 stores to scratch [ky][x].
* pass B per ky (8×): 16 scratch loads, x-axis DFT across registers, transpose pair
  (2×24 non-destructive permutes: their T2/T3/T1 bit-swap network = my
  VSH44/VSHEE, VSH88/VSHDD, VUNPLO/VUNPHI), free `piinv` register relabel, z-axis DFT,
  then their 48-op fused untranspose+re-interleave over all 16 registers, and 16
  table-driven stores (offset kx*128 + half*8 within the ky-row).

The point (radix8's r5 analysis, confirmed by my r6 wallaby gap): the pass that faces
DRAM carries 16 shuffles instead of my LANEX transposing load's 96, and both heavy
networks run against L1 scratch.  **I substituted my 52-op FMA codelet (`r8`) for their
56-op 4-mul `dft8s`** — both are natural-order in/out, so the surrounding index algebra
transfers verbatim — which makes my port 1248 FP + 896 shuffles + 256/256 mem ops per
volume, 48 FP instructions fewer than the original that won the cells.

**2. MODE_FUSED3 — their `seq3` variant, same port** (fused with the kx-axis DFT moved
to a zero-shuffle middle pass through a second 8 KiB L1 scratch, so pass B2's 16 stores
land in ONE kx plane in ascending address order — fully sequential write stream at
+128/+128 L1-resident mem ops).  Included because their B=64 node win (0.594) may have
been this variant, not fused (their own wallaby pick at B=64 was seq3+pfs and their
prediction bracketed 0.59–0.61), and it differs from my LANEX3 in exactly the
shuffle-placement way.  Scratch arena grown 1216 → 2112 doubles to hold scratch2.

**3. Candidate sets rebuilt around them** (all four r6 mechanisms retained):
B=1 {LANEX2 (default, node 0.570 four rounds), FUSED, LANEX2S, LANEX3} plain;
mid/B=64 {FUSED+s0 (default), FUSED3+s0, LANEX3+s0 (my node 0.610), FUSED+none,
FUSED3+none, LANEX3+none}; streaming {FUSED+s0w (default — the exact winning node config
of r5 on the winning shape), FUSED+s0, FUSED3+s0w, LANEX3+s0w, LANEX3+s0 (my
node-verified 1.096/1.388), FUSED+nt+s0 (insurance), LANEX3+none}.  LANEX2S and LANEX2
leave the batched sets (three rounds of node numbers never had them first); they remain
compiled and in the B=1 set.  The streaming tournament is now the r5 VERDICT's isolating
experiment run properly inside one entry: fusion×pfw fully crossed.

### What was measured (wallaby, Xeon Gold 6448Y SPR; the 2× clock lottery documented
since r2 is still present — same-process in-tuner tables and fast-state driver mins)

| B | r6 code | this round | notes |
|---|---|---|---|
| 1 | 0.305 µs | **0.305** | LANEX2 plain, byte-identical path (2 of 3 runs fast-state) |
| 8 | 0.311 | **0.310** | |
| 64 | 0.306 | **0.316** | in-tuner (one noisy state): L3/none 0.350, FUSED/s0 0.363, FUSED3/none 0.365 — order flips run to run, the standing "B=64 on wallaby is noise" lesson |
| 2048 (0.53×L3, mid set here) | 0.451 | **0.465** (fast window) | forced plain A/B: FUSED+s0w 0.899, FUSED3+s0w 0.994 µs/vol |
| 16384 (4.4×L3) | 0.540 | 0.698 (one run, NT pick as every round) | **in-tuner plain column: FUSED/s0w 0.759 ≈ LANEX3/s0w 0.762 ≈ FUSED3/s0w 0.773**, vs s0-only 1.10–1.18 (pfw −31–35%, r6 reproduced) |

Wallaby CANNOT rank the three s0w structures (2% spread) — same non-transfer it showed
for every store-order/prefetch family since r3.  The node tournament decides; the default
is the node-proven config (fused+pfs+pfw ≡ FUSED+s0w).

Correctness: rel_l2 = 1.87–2.27e-16 (tolerance 1e-12) at B = 1, 3, 5, 7, 17, 64, 2048,
16384; bit-identical re-runs everywhere.  FUSED and FUSED3 forced individually at B=1, 7,
2048 including PF_SW (rel_l2 2.270e-16 — identical to fusedaxes' value, as expected: same
axis order); the `-DL8_EMU8` build forced through both new modes PASSes (the T2/T3/T1
networks and the FOUT/piinv tables exercised in plain C per Intel pseudocode); AVX2
(wombat) and `-DL8_SCALAR` builds PASS; warning-free under `-Wall -Wextra`.
cascadelake asm audit: **0 vector spills and 0 zmm register copies in all six new
runners** (the BF primitives are all 2-source non-destructive forms — the property
fusedaxes' r1 record designed for, preserved through my macro layer); 16 static
prefetchw in `f_run_psw` (8+8 ✓), 224 static shuffles per runner (112/volume × 2
inlined bodies, loops rolled ✓).

### What was tried and did NOT work

1. Nothing failed outright; the port worked first try (rel_l2 2.27e-16 on the first
   forced run).  The near-miss to record: **on wallaby the ported FUSED does not
   reproduce fusedaxes' r6-observed 12% margin over my LANEX3+s0w** (0.759 vs 0.762
   here, vs their 0.637 against my 0.754 in r6 — different processes, different clock
   states).  Either the margin was wallaby state-lottery all along, or it is
   node-specific.  Both explanations say the same thing: ship both, read the node.
2. Not retried, per the records: everything in the r1–r6 failure lists (burst+NT clog,
   NT as default, separate shuffle-only output pass, interleaved-complex lanes, in-lane
   butterflies, gather, batch-in-lanes, double-buffered scratch, cross-volume software
   pipelining, prefetch distance 2).

### Borrowed / lent

* **L8_fusedaxes**: the entire FUSED and FUSED3 structures — the PI lane order, the
  T2/T3/T1 non-destructive butterfly decomposition, the piinv relabel, the fused
  untranspose+interleave network, the out_off/FOUT store tables, and the 8+8 / 6-5-5
  prefetch cadences.  This is the largest single borrow in this file's history and it
  is their r5 node win (0.594/0.910/1.254) being adopted wholesale.
* **L8_radix8 r5**: the precedent that the port transfers (`1f`, −9% at B=64).
* Lent / for others: the two shapes compose — my 52-op FMA codelet drops into their
  index algebra with no other change (both codelets are natural-order in/out), so any
  entry can mix the best codelet with the best structure; and the EMU8 harness now
  emulates their whole network family in plain C, which is the cheap way to verify a
  port of it on a machine without AVX-512.

### Prediction for the node (stated to be scored)

* B=1: **0.570–0.574 stands** (byte-identical LANEX2 path, protected default; FUSED
  joins the candidate table — fusedaxes' B=1 is 0.573–0.579, so no change expected).
* B=64: default FUSED+s0, full coverage of the winning family.  If the r5 winner was
  the shape, **0.59–0.61**; the pick string (FUSED vs FUSED3 vs LANEX3) is the round's
  B=64 datum.
* B=2048: the r5 winning configuration is now my default with 48 fewer FP
  instructions/volume: **0.90–0.95** if the shape carries at face value; anything at
  1.00+ means the win does not survive the codelet/macro-layer translation, which
  would be new information.
* B=16384: same logic from 1.254: **1.22–1.30**.
* Streaming pick strings to read next round: FUSED+s0w vs LANEX3+s0w separates fusion
  from pfw on the node at last; FUSED3+s0w vs FUSED+s0w separates store order within
  the fused family.

### Next

1. Read the node's pick strings (descriptions carry them).  (a) FUSED+s0w picked and
   ≈0.91/1.25 → converged with fusedaxes; the remaining lever is the bandwidth floor
   (~1.37 at B=16384 per the r4 VERDICT arithmetic) and the honest next step is
   idle-port scheduling in pass A (more prefetch under the shuffle-light pass), worth
   at most ~5%.  (b) LANEX3+s0w picked → fusion does NOT transfer through my macro
   layer; diff the emitted asm of `f_run_psw` against fusedaxes' `vol_p_sw` next
   round.  (c) FUSED3 picked anywhere → sequential stores still matter even under
   pfw; propagate to the streaming default.
2. B=1: still frozen at 0.570 = 1.32× the 1248-cycle p0 floor at the measured
   2.89 GHz clk512.  The untried levers remain mixed-width (ymm halves of the codelet
   co-issued on port 1) and cross-volume pipelining at small batch — both big
   rewrites, neither justified until the monitor's port data says B=1 is
   throughput-bound rather than latency-bound.
3. If FUSED wins the batched cells, next round should delete LANEX2S entirely and
   consider retiring LANEX2 to a B=1-only compile (the file is at 6 structures × 5
   runner variants and the tuner budget is better spent on prefetch cadence variants
   for the winning shape).

---

## Round panel_r8

### Where round 7 left me (node numbers, panel_r7)

The port round worked: **B=1 0.558 — first, the first movement in that cell in six
rounds** (fusedaxes 0.571, radix8 0.572, MKL 0.655), won with the ported FUSED shape
(picks LANEX3/FUSED/FUSED at 0.5935/0.5577/0.5647 — the FUSED runs are the two fastest
B=1 numbers ever taken at L=8).  **B=16384 1.232 — first** (fusedaxes 1.234), FUSED+s0w
3/3.  B=64 **0.588 — second by 0.001 µs** (fusedaxes 0.587; my picks were LANEX3+s0 at
0.6048/0.6135 and FUSED+s0 at 0.5880 — the run that kept my shipped default was the
fastest).  B=2048 0.945 reported — second — but the VERDICT (§3b) reads it honestly at
≈0.984 (0.9450/0.9842/0.9876; the min is a 4.2% outlier), FUSED+s0w 3/3, against
fusedaxes' 0.930 (itself +2.2% on r5 with a 4.6% spread).  The r7 VERDICT's L=8
instructions: streaming is converged (three entries within 3.6% on one technique) —
**stop tuning it**; the B=1 residue (1612 cycles vs the 1248–1296 floor) hinges on a
`ld_blocks_partial.address_alias` counter run that only the monitor can do, and the
aliasing, if real, is between the driver's own in/out buffers — likely unreachable from
inside a plan.  Two panel-wide r7 lessons directly actionable by me: L45_pfa's
build-flag gap (the scored build lacks `-funroll-loops`, which tryout.sh has — worth
10% to them) and its scalar-instruction audit (gcc materialising offset tables).

### What changed this round: no kernel code — the node's own r7 verdicts written into the plan machinery

Round 8 is deliberately a consolidation round.  Every hot path is byte-identical to the
r7 exemplar; what changed is which configuration each regime *defaults to* and what the
tuner is allowed to consider:

1. **B=1 default LANEX2 → FUSED.**  The node displaced my shipped LANEX2 default in all
   three r7 runs (so LANEX2 lost by >3% in-arena every time) and the two runs that
   picked FUSED produced 0.5577/0.5647 while the LANEX3-picking run cost 0.5935 (+6%).
   With FUSED as the hysteresis-protected default, the pick instability that cost run 1
   its 6% should disappear.  LANEX2S leaves the B=1 candidate set (never picked at B=1
   on the node in any round); LANEX2 and LANEX3 stay as candidates.
2. **Mid-regime (B=64) hysteresis widened 3% → 6%.**  The r7 create()-arena ranking
   inverted the driver-level ranking twice at exactly the old 3% band: the arena said
   LANEX3+s0 beat FUSED+s0 by >3% (two runs of three), but the driver measured the
   FUSED+s0 run fastest (0.588 vs 0.605/0.613).  The default (FUSED+s0, unchanged) is
   the driver-verified configuration, so the arena must now beat it by a clear 6% to
   move off it.  Applied to the mid regime only; B=1 and streaming keep 3%.
3. **FUSED+s0w and LANEX3+s0w join the mid candidate set.**  At B=64 the working set
   (1.00 MiB) is exactly the node's 1 MiB L2, so output lines are typically L2-evicted
   when re-stored and the RFO goes to L3 (~44 cycles); prefetchw one volume ahead can
   hide that.  Wallaby cannot measure this — its 2 MiB L2 holds the whole working set —
   so the arm is offered and the node's own tournament decides.  This is the only cell
   I lost by a margin worth attacking (0.001 µs), and the s0w runners already exist.
4. **FUSED3 leaves every candidate set** (zero picks in 12/12 r7 node runs, both mid
   and streaming; the code stays compiled for forced runs).  Streaming set otherwise
   untouched per the VERDICT's "stop tuning it": FUSED+s0w default, same insurance arms.

Operation counts unchanged: 1248 vector FP + 896 shuffles per volume (FUSED 256+256 mem
ops, LANEX3 384+384).

### Two panel-wide r7 findings tested here, both nulls for this entry

* **Build-flag gap (borrowed from L45_pfa r7): NULL for this code.**  Alternating
  same-session A/B on wallaby, default tryout flags (with `-funroll-loops`) vs
  `-fno-unroll-loops` (simulating the node build): B=1 fast-state min **0.305 vs
  0.307 µs** (≤1%, inside spread); B=16384 differences straddle zero (0.650/0.656 vs
  0.603/0.705 µs/vol across two pairs).  My hot loops are all compile-time-constant
  trip counts that gcc rolls/unrolls identically under both flags (the r7 asm audit's
  "loops rolled" holds in both).  No pragma shipped — it could only perturb codegen
  that four rounds of node numbers validate.  Recorded so nobody re-tests it at L=8.
* **Scalar-instruction audit (borrowed from L45_pfa r7): PASS, nothing to fix.**  At
  node flags (`-O3 -march=cascadelake`, no unroll): f_run_p0 = 742 instructions of
  which **51 scalar** (all loop/pointer bookkeeping — leaq/addq/cmpq), f_run_psw 72,
  l3_run_ps0 74.  Zero `movsx`/table reloads: the FOUT store-offset table and fpiinv
  relabel fold to immediates.  Nothing like L45_pfa's 758-scalar pathology exists here.

### What was measured (wallaby, Xeon Gold 6448Y SPR; the 2× clock lottery documented
since r2 is still present — fast-state mins over repeated runs)

| B | r7 code | this round | notes |
|---|---|---|---|
| 1 | 0.305 µs | **0.305** | byte-identical kernel, FUSED now default |
| 64 | 0.316 | **0.306** (19.585/64) | |
| 2048 (0.53×L3, mid set here) | 0.465 | **0.457** | |
| 16384 (4.4×L3) | 0.698 | 0.668 (one run) | wallaby keeps NT, as every round |
| 7 (tail) | — | 4.211/7 = 0.602 slow-state | correctness case |

Correctness: rel_l2 = 1.87–2.27e-16 (tolerance 1e-12) at B = 1, 7, 17, 64, 2048,
16384; bit-identical re-runs everywhere.  Forced FUSED+s0w at B=64 (the new mid arm)
PASSes, rel_l2 2.267e-16.  The `-DL8_EMU8` build PASSes (B=17); `-DL8_SCALAR` and the
cascadelake build compile warning-free under `-Wall -Wextra`.  No kernel bytes changed,
so the r7 spill/copy audits stand.

### What was tried and did NOT work

1. **`-funroll-loops` parity** — the null above, with the numbers.  The r7 VERDICT
   flagged it as applying "to every entry"; for this entry it does not.
2. Nothing else was attempted, deliberately.  The VERDICT's reading stands: streaming
   is converged, B=1's residue needs the monitor's alias-counter run before any kernel
   responds to it, and the L=6 falsification (17–25% uop cuts, licence-fair, zero
   picks) says the compute-side levers I have left (mixed width, in-lane butterflies)
   do not pay at B=1-like residency.  Not retried, per the records: everything in the
   r1–r7 failure lists.

### Borrowed / lent

* **L45_pfa r7**: the build-flag A/B protocol and the scalar-instruction audit — both
  run to completion here, both nulls, both recorded with numbers.
* **panel_r7 VERDICT §3b/§6**: the honest-minimum reading of my own B=2048 (≈0.984,
  not 0.945 — I will not quote the outlier), and the per-run pick-string analysis that
  drives every change this round.
* Lent / for others: the arena-vs-driver inversion at B=64 (a create()-time tuner can
  reproducibly rank two near-tied candidates opposite to the driver's 20-sample
  measurement — if your default is driver-verified, widen the hysteresis rather than
  trust the arena); and the L2-exact working-set argument for offering pfw at mid
  batch even though it is a documented uop tax at cache residency.

### Prediction for the node (stated to be scored)

* B=1: **0.552–0.565, pick FUSED 3/3.**  The kernel that measured 0.5577/0.5647 is now
  the protected default; the gain over 0.558 is only the removal of the LANEX3-pick
  lottery, so the min moves little but the spread tightens.
* B=64: **0.583–0.592** if s0w is a wash (default FUSED+s0 = the r7 best run, now
  6%-protected); **0.57–0.58 with pick FUSED+s0w** if the RFO-hiding argument is real
  at L2 scale.  The pick string is the round's B=64 datum either way.
* B=2048: **0.93–0.99** (unchanged config; the honest r7 value is ≈0.984 and fusedaxes'
  ≈0.95 with both spreads overlapping — converged, per the VERDICT).
* B=16384: **1.22–1.26** (unchanged config, 6/6 pick stability expected).

### Next

1. Read the node picks.  (a) B=1 FUSED 3/3 and ≤0.558 → the consolidation worked; the
   remaining B=1 residue belongs to the monitor's `ld_blocks_partial.address_alias`
   run (VERDICT §6) — do not write kernels against it until that number exists.
   (b) B=64 s0w picked → propagate: offer s0w at whatever regime the working set is
   1–2× L2.  (c) any regression → the hysteresis widening was wrong; revert to 3%.
2. If fusedaxes moves B=1 (they will read the same r7 data), the next real lever is
   theirs to find; my honest assessment is that FUSED at B=1 is within ~1.25× of a
   floor whose residue no candidate in the current design space touches.
3. The B=2048 cell should be treated as tied until either entry's three-run spread
   drops under 2%; the VERDICT's "stop tuning streaming" applies to me verbatim.

---

## Round panel_r9

### Where round 8 left me (node numbers, panel_r8)

B=2048 **0.912 — first** (honest ≈0.931 per the VERDICT §3e; fusedaxes ≈0.946) — the
cell recovered exactly as the unchanged config predicted.  B=16384 1.241 vs fusedaxes'
1.236 — a tie inside spread, third round converged.  B=64 0.589 — second on minima
(fusedaxes' 0.575 min sits on a 9.1% spread; **on medians the cell is a tie**, VERDICT
§3e).  B=1 **0.564 — second**, and this one is a real loss: fusedaxes took 0.552 with
runs 0.5516/0.5569/0.5572, *entirely below* my 0.5643/0.5647/0.5935 — they adopted my
52-op codelet into their fused shape, went four-for-four on their prediction, and the
r8 VERDICT §3b declares all three L=8 entries **bit-identical in all four cells**: the
geometry is one algorithm implemented three times, and the 3.3% B=1 spread
(0.552/0.564/0.570) is a pure measurement of non-arithmetic cost — prefetch branch
shape, scratch placement, code layout.  My own r8 mechanisms: the B=1 FUSED default
still lost the pick lottery once (LANEX3 fired in one of three runs at 0.5935, +5%,
despite 3% hysteresis — and the raw r8 result JSONs show **LANEX3 picked in 4 of 6
B=1 creates**); s0w at B=64 was offered and rejected (s0 picked); both borrowed audits
were documented nulls.

### What changed this round: no new kernel — retire the arena where it mis-ranks, and
### two free structural cleanups

1. **B=1 runs NO tournament.**  `fft3d_create()` at batch==1 (W=8) skips `autotune()`
   entirely and hardwires FUSED/plain/no-pf — the configuration the DRIVER itself
   measured fastest in every FUSED-picked node run of r7 and r8
   (0.5577/0.5647/0.5643/0.5647), against LANEX3-pick runs of 0.5935/0.5935.  The
   create()-arena has now been shown three times to mis-rank near-ties that the
   driver's 20-sample measurement ranks consistently (r7 B=64 inversion, r8 B=1
   4-of-6 LANEX3 picks, r8 B=64 one LANEX3 pick = the 0.6151 run).  Hysteresis
   widening (my r8 fix) demonstrably did not eliminate the lottery; removal does.
   This is the honest generalisation of my own r8 lesson — "if your default is
   driver-verified, widen the hysteresis rather than trust the arena" was still too
   much trust.
2. **Mid regime (B=64): structure fixed at FUSED, only the prefetch policy is raced**
   ({s0 (default), s0w, none}).  LANEX3's arena wins at B=64 were driver-level losses
   in BOTH r7 (0.605/0.613 vs 0.588) and r8 (0.6151 vs 0.5886/0.5910).  s0w stays a
   candidate: rejected once (r8) but the working-set-equals-L2 argument stands and
   wallaby structurally cannot test it.
3. **Streaming: byte-identical candidate sets and defaults** (FUSED+s0w, picked 6/6 in
   r8) — the VERDICT's "stop tuning it" applied verbatim, second consecutive round.
4. **FUSED/FUSED3 scratch de-aliased: SI moved from scr+512 to scr+520 doubles**
   (`-DL8_SI_OFF=512` restores the old layout for a forced A/B).  SR and SI were
   EXACTLY 4096 B apart, so at the pass A→B boundary every pass-B load of an SR line
   falsely aliases (address bits 11:6) any still-in-flight pass-A store to the
   same-index SI line — the `ld_blocks_partial.address_alias` mechanism, applied to
   my own layout via **L8_fusedaxes' r7 line-granularity alias model** (borrowed).
   +64 B breaks the relation for every same-index pair; FUSED3's S2 moves to
   scr+1040 (2 lines mod 4096 from SR, 1 from SI).  Honesty note: fusedaxes' own
   classic fused keeps the 4096 relation and still runs 0.552, so this is NOT the
   inter-entry gap — it is simply free to remove, and it gives the monitor's pending
   alias-counter run one clean forced pair inside one binary.
5. **Arena page-aligned (posix_memalign 4096, was 64).**  The scratch frame offset
   mod 4096 is now a fixed 0 instead of a glibc placement draw — per
   **L17_matrixsimd's r8 finding** (borrowed) that heap offsets are a fixed property
   of a build, this makes my own round-over-round comparisons meaningful and removes
   my half of the "allocation lottery" as a confound.

### Operation count

Unchanged since r7: 1248 vector FP (52-op codelet × 3 axes × 8) + 896 shuffles per
volume; FUSED 256+256 mem ops, LANEX3 384+384.  This round changes zero instructions
on the hot path — asm audit at node flags (`-O3 -march=cascadelake -funroll-loops`,
`-Wall -Wextra` clean, scalar build clean): f_run_p0 is 742-instruction-equivalent
(744 objdump lines) exactly as the r8 audit recorded, zero vector spills (3 integer
rsp refs, the known pointer bookkeeping).  Only the SI base displacement and the
allocation alignment differ.

### What was measured (wallaby, Xeon Gold 6448Y SPR; the documented ~2× state lottery
is in full force this session — fast/slow medians 0.330/0.645 at B=1, ratio 1.95)

| B | r8 code (r8 session) | this round | notes |
|---|---|---|---|
| 1 | 0.305 µs | **0.330** (fast-state min, many runs) | setup 0.000 s — tuner gone; this session's fast state is ~8% slower than r8's session across the board, see below |
| 7 (tail) | 0.602 | 0.324/vol (2.267 µs fast) | correctness case, forced FUSED3 also PASS |
| 64 | 0.306 | **0.328** (20.981/64) | consistent with the session state shift |
| 2048 | 0.457 | 0.446 (913.98/2048) | |
| 16384 | 0.668 | 0.662 (10840/16384) | wallaby keeps NT as every round |

**SI=520 vs SI=512, alternating same-session A/B at B=1 (3 pairs): 0.330/0.330,
0.645/0.645 (slow-state pair), 0.330/0.330 — dead even.**  Expected: wallaby has never
resolved a store-order/alias-family effect (r3–r8 records), SPR's alias penalty is
smaller, and the node's heap offsets differ anyway.  The wallaby measurement
establishes only "no regression"; the node decides the sign.  The uniform ~8% offset
of this session's fast state vs the r8 session (0.330 vs 0.305 at B=1, 0.328 vs 0.306
at B=64, identical binaries' companion MKL also shifted) is the machine, not the code
— one more entry for the never-compare-across-sessions file.

Correctness: rel_l2 = 2.267–2.280e-16 (tolerance 1e-12) at B = 1, 7, 17, 64, 2048,
16384; bit-identical re-runs everywhere; forced FUSED3 at B=7 PASSes (its S2 offset
changed); the `-DL8_EMU8` build at B=17 PASSes (the new offsets exercised in plain C);
`-DL8_SCALAR` and cascadelake builds compile warning-free under `-Wall -Wextra`.

### What was tried and did NOT work / was deliberately not done

1. **fusedAA was NOT ported**, and this is a decision, not an omission: fusedaxes
   offered their address-aware variants (execute-time scratch base vs `in`, permuted
   pass-B order vs `out`) at B=1 in r7 and r8 and **the node's own tournament
   declined them both times while plain fused won the cell at 0.552**.  Their comb
   analysis also shows my pass-A store pattern (8 lines at 16-line spacing) collides
   with a 16-line load window at every base offset, so base choice alone cannot fix
   my layout — only their full layout change could, and the node has not endorsed it
   in their own file.  If the monitor's alias counter (asked in r7, r8, and now by
   the r8 VERDICT §6 as the single L=8 item) shows variant-0-vs-12 moving, that
   evidence changes; until then this stays unbuilt.
2. **No runner pruning / no code-layout chasing.**  The remaining B=1 gap to 0.552 at
   bit-identical arithmetic is, by elimination (same shape, same codelet, same plain
   config, alias relations now shown not to explain it in their file), code layout
   and driver-buffer offsets — the class where the r8 VERDICT names four panel-wide
   instances of refactors around untouched hot paths regressing 1–2%.  Deleting my
   never-picked runners re-rolls that die with no model of the outcome; declined.
3. Not retried, per the records: everything in the r1–r8 failure lists.

### Borrowed / lent

* **L8_fusedaxes r7**: the line-granularity 4K-alias model — applied here to find and
  remove my own SR↔SI 4096-byte relation (their record's "26–30 blocked
  loads/volume" arithmetic is the motivation, their sigma/comb analysis the tool).
* **L17_matrixsimd r8**: heap offsets are per-build constants, not a lottery — the
  page-pinning rationale.
* **panel_r8 VERDICT §2/§3e**: the per-run pick-string reading that convicts the
  arena at B=1 and B=64; §6's "stop tuning streaming".
* Lent / for others: the cleanest statement yet of the arena-vs-driver failure mode —
  across r7+r8 my arena picked LANEX3 in 5 of 9 creates at cells where every
  LANEX3-picked driver run was +3–6%; if your default is driver-verified across two
  rounds, the right amount of arena trust at that cell is ZERO, not a wider band.
  Also: `-DL8_SI_OFF=512/520` is a one-flag alias-counter A/B inside one binary,
  ready for the monitor's pending perf run.

### Prediction for the node (stated to be scored)

* B=1: **pick string reads "fixed, no tuner"; three runs all on FUSED/plain.**
  Central estimate **0.560–0.566 min, spread < 1.5%** (the 0.5935-class tail is
  structurally gone; min moves only if SI=520 or page-pinning has node value).  If
  the SI de-alias is worth what the blocked-load arithmetic allows (up to ~8 loads ×
  ~10 cy at the pass boundary ≈ 0.5%), **0.558–0.562**.  I do not predict taking the
  cell: fusedaxes' 0.552 residue is code-layout, which I have deliberately not
  re-rolled.
* B=64: FUSED+s0 3/3 (no structure rival left in the set), **0.583–0.592 min,
  median ≈0.590** — the r8 0.6151 LANEX3 run has no path to recur.  s0w pick would
  mean the L2-RFO story is real after all.
* B=2048: unchanged config, **0.91–0.95** (honest ≈0.93, the converged value).
* B=16384: unchanged config, **1.23–1.26**.

### Next

1. Read the node's B=1 three-run spread.  (a) All three ≈0.564 → lottery closed, and
   the remaining −1.5% to fusedaxes is confirmed pure code-layout/driver-offset; the
   only principled attack left is the monitor's alias counter
   (`-DL8_SI_OFF=512` vs `520`, and my binary vs theirs), which is already the
   VERDICT's single L=8 ask.  (b) Min drops ≤0.558 → the SI/page changes had node
   value; report the delta to the corpus since fusedaxes' classic fused still
   carries the 4096 relation.  (c) Any run ≥0.58 → something outside the plan moved
   (the fixed path cannot pick wrong by construction) — flag it as a machine datum.
2. If the alias counter never runs, the L36_pfa in-plan pattern could carry it: an
   execute-lazy variant is NOT possible for a counter, but a create()-time timed
   A/B of SI=512-vs-520 **on the create arena** is one honest step (arena offsets
   differ from the driver's, stated limitation) — only worth building if (1b).
3. The geometry-level question is now resourcing, not engineering: the r8 VERDICT
   says L=8 does not need three slots for one algorithm and names radix8 the donor.
   If the panel consolidates, the surviving entries should keep both the
   convergence story and the pick-lottery lesson in their records.
