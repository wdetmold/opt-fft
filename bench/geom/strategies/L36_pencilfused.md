# L36_pencilfused — strategy record

Geometry: **L = 36** (46 656 complex doubles = 729 KiB per volume).
File: `impl/L36_pencilfused.c`. `fft3d_name()` = `L36_pencilfused`.

---

## Round 1 (panel round 1)

### Technique

Two-pass **pencil/tile-fused** row–column transform. Every 36-point line is a
**Good–Thomas (PFA) 4×9** kernel; SIMD lanes always hold a *spectator* axis (never the
transform axis), split-complex (SoA) inside the kernel, interleaved only at the two
buffer boundaries the driver owns.

```
pass 1   x-transform of the whole volume.
         lanes  = TW consecutive positions of the flat (y,z) index (1296, divisible by
                  both 4 and 8, so no masking and no wasted lanes)
         read   = in[x][flat]      contiguous, 2 loads + de-interleave per element
         write  = A[kx][y][z]      split-complex scratch, 2 x 373 KB
pass 2   for each of the 36 kx-planes (20.25 KiB, L1-resident):
   2a    y-transform, lanes = z (36 wide -> 9 groups at TW=4, 5 at TW=8)
         output transposed in TWxTW register blocks into a plane buffer P[z][ky]
   2b    z-transform, lanes = ky (reads P down its stride-36 column: contiguous vectors)
         output transposed back, re-interleaved, stored straight into out[kx][ky][kz]
         (non-temporal when the batch has left L3)
```

The grid is crossed **twice**, not three times — Liu et al.'s two-pass 3D FFT shape
(corpus 05 §3.4) — and there is **no materialised transpose anywhere**: only TW×TW
in-register transposes on an L1-resident plane (corpus 05 §10.2). `A` is 729 KB, which
is what the Cascade Lake target's 1 MiB L2 holds, so on the scored machine the only
L3/DRAM traffic is *read `in`* + *write `out`*, which is the information-theoretic
minimum for any out-of-place transform. On the 256 KiB-L2 Haswell dev box `A` spills to
L3 and that costs about 30 µs per pass — see "what didn't work" #7.

#### Why exactly one transpose pair is unavoidable (worth writing down)

Let *l* be the lane axis of a pass. For a pass to be shuffle-free, *l* must be the
contiguous axis of both its input and output buffer, and the transform axis must differ
from *l*. Pass *i*'s output buffer is pass *i+1*'s input, so all passes share one *l*.
Three passes must transform three different axes, so exactly one of them has transform
axis = *l*, and that one needs a transpose on each side. Two volume-transposes is the
floor for a fixed interleaved-complex in/out layout; the only freedom is *where* they
land. Putting both inside pass 2, where the plane is in L1, is the cheap placement.

### Derivation and operation count

Ruritanian input map / CRT output map for 36 = 4·9 (`[9^-1]_4 = 1`, `[4^-1]_9 = 7`):

```
n = (9 n1 + 4 n2) mod 36      n1 = 0..3, n2 = 0..8
k = (9 k1 + 28 k2) mod 36     k1 = 0..3, k2 = 0..8
=> n k = 81 n1k1 + 252 n1k2 + 36 n2k1 + 112 n2k2
       = 9 n1k1 + 4 n2k2   (mod 36)     since 252 = 7*36 and 112 = 3*36 + 4
=> W36^{nk} = W4^{n1k1} * W9^{n2k2}     — the twiddle stage vanishes identically
```

Verified numerically against numpy at 3.8e-16.

| module | form used here | flops | source ops |
|---|---|---|---|
| DFT-4 | radix-2 butterflies, `×(∓i)` folded into the adds | 16 | 16 add |
| DFT-3 | `s,d`; `u = a0 − s/2`; `t = (√3/2)·d`; `y1,2 = u ± t` | 16 | 10 add + 2 fma + 2 mul = 14 |
| cmul (W9) | 2 mul + 2 fma | 6 | 4 |
| DFT-9 | CT 3×3 = 6 DFT-3 + 4 nontrivial W9 twiddles | 120 | 100 |
| **PFA-36** | 9 DFT-4 + 4 DFT-9 | **624** | **544** |

Per volume: 3 × 1296 lines × 624 = **2 426 112 real flops** in 2 115 072 source-level FP
ops → 528 768 vector FP instructions at TW=4. gcc re-contracts many add/mul pairs into
FMAs; the count *emitted* under `-march=skylake-avx512` is ~492 FP instructions per line.

Comparison to the corpus (02 §5.4): PFA 4×9 with FFTW's hand-scheduled modules is
688 flop / 464 instr; FFTW's actual mixed-radix 3/6/12 plan charges itself 756 / 552.
So this file is below FFTW's plan on both counts, and ~6% above the ideal PFA
instruction count, entirely inside the 9-point module (100 ops here vs `n1_9`'s 80).
**That 8-ops-per-DFT-9 gap is the single largest identified arithmetic saving left: ~6%.**

Data movement per volume, TW=4: 2 × 36×36 register transposes per plane
(0.5 shuffles/double) + de-interleave of `in` + re-interleave of `out`
≈ 187 000 shuffle instructions against ≈ 479 000 FP ones. On Cascade Lake the shuffles
are port-5 and the FP work ports 0+1, so they overlap; the FP side is the binding port.

### Layout and SIMD decisions

* **Split-complex (SoA) inside the transform, interleaved only at `in`/`out`.** Confirms
  corpus 04 §2: zero shuffles in the arithmetic, `×(±i)` free (a sign folded into the
  next add), constants are lane-invariant broadcasts. The interleave/de-interleave at the
  boundaries costs 1 permute per TW doubles on AVX-512 (`vpermt2pd`) and 2 on AVX2
  (`vunpck` + `vpermpd`) — verified in the disassembly.
* **Lanes = spectator axis, never the batch.** At B = 1 there is no batch to vectorise
  over, and B = 1 is scored separately, so the whole design has to work at B = 1. It
  does: the lane axis is always another spatial axis. Batch-major repacking was rejected
  on traffic grounds (see #4 below).
* **Two instantiations from one source text**, by `#include __FILE__` under different
  `TW`/target macros: 256-bit with *no* target attribute (so it inherits `-march=native`
  and on the scored node gets EVEX-encoded 256-bit ops, 32 registers and `vpermt2pd`,
  without paying the 512-bit frequency licence) and 512-bit under
  `target("avx512f,avx512dq,avx512vl")`. Verified to compile under `-O3` bare,
  `-march=core2`, `-march=haswell`, `-march=skylake-avx512`, `-O2`, and from a foreign
  cwd with an absolute path.
* **The generic W×W transpose** is a butterfly of `__builtin_shuffle`s with masks built
  by a compile-time loop: for block size `s`, `lo[j] = (j/s odd) ? W+(j/s−1)s+j%s :
  (j/s)s+j%s`, `hi[j] = lo[j]+s`, applied for `s = 1,2,…,W/2`. It lands `a[i] =
  column i` with no fix-up permutation. gcc folds the masks to immediates:
  8 instructions for 4×4 (`vunpcklpd/vunpckhpd` + `vinsertf128/vperm2f128`), 24 for 8×8
  (`vunpcklpd/vunpckhpd`, `vpermt2pd`, `vshuff64x2`). One expression, both widths.
* **Tail handling without masks.** 36 is not a multiple of 8. Instead of masked
  loads/stores the last lane group is *shifted to overlap* the previous one
  (`ZOFF = {0,8,16,24,28}` for TW=8, `{0,4,…,32}` for TW=4). Every access stays inside
  the volume — no over-read past the last volume of the batch, which a naive
  `z0 = 32, width 8` group would do by 8 doubles — and the duplicated lanes are written
  with identical values, so it is idempotent. Costs 11% extra work at TW=8 on the two
  lane axes that are 36 wide (none at TW=4, and none on pass 1 at either width because
  its lane axis is the flat 1296).
* **Alignment.** Every NT store is a full 64-byte line at a 64-byte-aligned address:
  byte offset is `20736·kx + 576·ky + 16·c0` with `c0 ∈ {0,4,…}` (TW=4) or
  `{0,8,16,24,28}` (TW=8), all multiples of 64.

### Self-tuning in `fft3d_create()` (and the correctness interlock)

Four configurations — {256-bit, 512-bit} × {cached stores, NT stores} — are timed on a
dummy batch of `min(B,24)` volumes, warmed up first (a cold single shot mis-ranks NT).
**Every configuration must reproduce the 256-bit cached-store answer on this machine to
1e-11 relative before it is eligible.** That interlock matters: the 512-bit path cannot
be *executed* on the AVX2 dev box, so a latent bug in it would otherwise only appear on
the graded node. With the interlock, a wrong AVX-512 kernel can cost speed but never
correctness. Setup cost 30–130 ms, excluded from the score.

### What was measured

**Dev machine** (Xeon E5-2680 v3 Haswell, AVX2 only, 2.5 GHz base / 3.3 turbo,
32 KiB L1d / **256 KiB L2** / 30 MiB L3), gcc 11.4, `-O3 -march=native -mtune=native
-std=gnu11 -fno-math-errno -funroll-loops`. µs **per transform**, best of 3 driver runs
of 10 samples each. MKL 2022 and FFTW 3 patient built from the panel's own `sota/`
sources against the same `driver.o`:

| B | **L36_pencilfused** | MKL 2022 | FFTW patient | speedup vs MKL |
|---|---|---|---|---|
| 1 | **242.9** | 365.1 | 359.2 | **1.50×** |
| 4 | **244.2** | 367.9 | 365.0 | 1.51× |
| 8 | **245.4** | 367.5 | 363.0 | 1.50× |
| 32 | **259.7** | 407.9 | 506.0 | 1.57× |
| 128 | **347.7** | 413.1 | 433.0 | 1.19× |
| 256 | 348 (single run) | — | — | — |

Correctness (`check.py` vs numpy): `rel_l2 = 3.83e-16` at B = 1, 4, 8, 32, 128, 256;
`rel_max ≤ 5.2e-16`. Tolerance is 1e-12, so ~3 400× inside it. Repeatability is
exercised by the driver itself — it runs warmup + a calibration loop + 10–20 timed
samples + one final scored execute, and it is that last execute's output that is
checked, so the transform is verified to be idempotent across hundreds of calls on one
plan. The 512-bit path was verified **on this AVX2 machine** by re-instantiating the
same template at `TW=8` with the target attribute removed, so gcc lowers 64-byte vectors
to ymm pairs: `rel_l2 = 3.83e-16` at B = 1, 4, 32.

Pass breakdown at B = 1 on the dev box (individual passes compiled out): pass 1 = 85 µs,
y = 77 µs, z = 81 µs, i.e. essentially balanced. Isolating memory by rewiring pass 1 to
read/write an L1-resident window: instructions alone = 57 µs, the L3 read of `in` adds
10 µs, the L3 write of `A` adds 17 µs. So on **this** box we are ~2/3 arithmetic-and-
shuffle-bound and ~1/3 memory-bound; the memory third should largely disappear on the
scored node, where `A` fits L2.

**Achieved bandwidth vs roofline** (asked for explicitly). Traffic per volume is
4 × 729 KB = 2.92 MB (read `in`, write `A`, read `A`, write `out`) plus a 729 KB RFO on
the `A` write, and no RFO on `out` when NT stores are on. At B = 1, 243 µs → 15.0 GB/s
against ~25 GB/s of single-core L3 read bandwidth on this part: **not at the bandwidth
roofline**. At B = 128, 348 µs → 10.5 GB/s of which only 1.46 MB/volume (4.2 GB/s) is
DRAM: also not at the DRAM roofline. Arithmetic roofline: 2.43 Mflop / 243 µs =
10.0 Gflop/s against 2 × 4 × 2 × ~3.0 GHz = 48 Gflop/s of AVX2 FMA peak = **21% of
peak**. So on the dev box this kernel is **instruction-issue bound, not bandwidth bound**
— measured 531 cycles per 4-line group against a 298-cycle port-limit model (Haswell
puts `vaddpd`/`vsubpd` on port 1 only, and 288 of the 496 emitted FP ops per line are
plain adds, so Haswell is a *pessimistic* proxy: Cascade Lake runs FP adds on ports 0
**and** 1, which alone should lift the FP ceiling by ~1.16× and remove the add-port
serialisation).

**Non-temporal stores** (corpus 05 §8): measured directly by forcing the flag.

| B | cached stores | NT stores | |
|---|---|---|---|
| 1 | 257 µs | 255 µs | neutral (`out` is L3-resident across calls) |
| 4 | 258 | 249 | neutral/small |
| 32 | **421** | **276** | **1.53× — the write-allocate elimination, exactly as predicted** |

This is the clearest single result in the file: at L = 36 a batch of 32 volumes is
already DRAM-streaming (48 MB against 30 MiB L3) and removing the RFO on the final write
is worth 53%. It costs nothing at B = 1, so the tuner is free to pick it either way.

### What was tried and did NOT work — with the number that killed it

1. **Forcing the PFA stage-boundary buffer (`u`, 72 vectors) into explicit memory instead
   of letting gcc spill it.** 363 µs vs 256 µs — **42% worse**. With a per-line buffer
   (`NGT=1` in the tiled variant) pass 1 alone went 86 → **154 µs**. gcc's register
   allocator, spilling ~40 of the 72 live vectors, beats hand-written staging decisively.
   Do not "help" it here.
2. **Splitting the two PFA stages into separate sub-passes over an L1 tile** (DFT-4 stage
   for NGT lines → tile buffer, then DFT-9 stage), the textbook cure for register
   pressure. Pass 1: 85.7 µs at NGT=4 vs 85.9 µs unsplit — **dead even** (NGT=2: 88,
   NGT=8: 95, NGT=16: 102). The 72-vector stage boundary is not what is costing us.
3. **Software prefetch.** `__builtin_prefetch` on pass 1's 36 source streams at distances
   2/4/8 groups: 252/259/257 µs vs 249 baseline — **all worse**. Prefetching the next
   kx-plane of `A` during the y-pass: 269 µs, and both together 284 µs. The hardware
   streamers already have this; the extra µops are pure cost.
4. **Vectorising over the batch** (lanes = TW different volumes), which would remove
   every transpose and every de-interleave. Rejected on traffic before coding: the batch
   stride is 729 KB, so building a lane vector is TW scalar loads plus inserts unless the
   batch is first repacked to SoA — and repacking costs 2 extra volume crossings in and
   2 out, i.e. 4 extra passes over DRAM at large B against the 2 the whole algorithm
   currently uses. It also does nothing at B = 1, which is separately scored.
5. **Slab-gathering pass 1** (read NC groups × 36 planes into an 18 KB L1 tile with the
   de-interleave done once, then run NC kernels out of the tile) to turn 36 short read
   streams into 36 long ones. B = 128 improved 345 → 324 µs at NC=16, but B = 1
   regressed 243 → 272 µs. Net loss on the headline metric; rejected.
6. **Compiler flag/pragma hunting on the instruction-bound inner loop**: `-fsched-pressure`
   58.4, `-fira-algorithm=priority` 57.4, `-fno-schedule-insns2` 58.8,
   `-fira-region=one` 57.5, `-fno-gcse` 57.3, `-fno-tree-slp-vectorize` 56.8, baseline
   **57.3 µs**. All inside noise except `-fschedule-insns -fsched-pressure`, which is a
   **disaster at 94.6 µs**. There is no compiler flag to buy here.
7. **Unrolling the transpose-tile loops so the lane offsets are compile-time constants**
   (letting the 36-vector output buffer stay in registers): 244.0 vs 241.7 µs — no gain,
   more code. Reverted.
8. **The σ-permutation trick for a cheaper AVX2 de-interleave** — take `vunpcklpd`'s
   natural lane order `(0,2,1,3)` and compensate with compile-time index permutations,
   halving the de-interleave from 4 shuffles to 2 per element. Designed, then dropped
   *without* implementing: on the scored node the exact mask compiles to a single
   `vpermt2pd` anyway (verified: 72 `vpermt2pd` for 36 elements under
   `-march=skylake-avx512`), so the trick buys ~6% only on machines with no AVX-512,
   i.e. only on the dev box, which is not scored. It also breaks at TW=8, where pass 1's
   flat blocks (multiples of 8) do not align with the y-pass's `(y, z-group)` blocks
   (`y·36 mod 8 ∈ {0,4}`).
9. **`madvise(MADV_HUGEPAGE)` on the driver's buffers** (corpus 05 §7 recommends it
   unconditionally at L = 36, where every element of a z-pencil is on its own 4 KiB
   page). Not done: THP is `madvise`-mode here, and the driver has already faulted in and
   populated `in` and `out` with 4 KiB pages before `fft3d_create()` runs, so collapsing
   them depends on khugepaged, whose default scan rate (4096 pages / 10 s) is orders of
   magnitude too slow to matter inside a benchmark process. The corpus's advice is sound
   but only actionable by whoever *allocates* the buffers, which is the driver.
10. **Vector radix** (2×2×2 decimation of all three axes at once, corpus 03 §3.5). Not
    attempted, on the corpus's own advice (03 §10.3): its advantage is multiply count,
    which is the wrong currency, and a fully unrolled 3D butterfly has the register
    pressure of O(L³) rather than O(L). Our measurements support that judgement
    indirectly: a *36-point* line already needs 72 live vectors and spills on 32
    registers; a 3D kernel would be hopeless.

### Next

In priority order, with why I expect each to pay:

1. **Replace the 9-point module.** 100 source ops here vs FFTW's `n1_9` at 80 instructions.
   The DFT-9 is 4/13 of the line's op count weighted by size — closing the whole gap is
   **~6% of the transform**. The lever is the second CT stage: `m = 1, 2` each cost 20 ops
   (2 cmul + DFT-3) and I could not get below that by hand; genfft's schedule does, so
   either transcribe `n1_9`'s DAG or run a small search over associations of
   `s = W9^m A1 ± W9^{2m} A2`.
2. **Settle 256-bit vs 512-bit on the node from the tuner's own verdict** and record it.
   This is new information for the corpus (04 §8.1 has no AVX-512 measurement). My model
   says 256-bit should win on a **Gold 5218 specifically**: that SKU has **one** 512-bit
   FMA unit, so 512-bit FP throughput is 8 doubles/cycle — *identical* to two 256-bit
   FMAs — while 512-bit additionally pays the frequency licence and 11% of lane waste
   because 36 is not a multiple of 8. Predicted vector-FP cycles per volume: 264k at
   TW=4 (2/cycle) vs 264k at TW=8 (1/cycle), i.e. a wash before the clock penalty. If
   the node instead picks 512-bit, that falsifies the model and is worth knowing.
3. **Find the real issue-bound.** 531 cycles per 4-line group against a 298-cycle port
   model is a 1.8× gap that neither spilling (#1, #2) nor scheduling flags (#6) explain.
   The remaining suspects are the ~1100-µop loop body against Haswell's 192-entry ROB
   and 1536-µop DSB. The experiment: instantiate two independent line-groups in the same
   loop body at TW=2 (halving register need per group) so a ROB window spans complete
   independent chains, and compare. If that is the cause it is worth much more than
   item 1.
4. **Large-batch regime.** B = 128 (348 µs) is 43% worse per volume than B = 8 (245 µs)
   even with NT stores, and the DRAM traffic (4.2 GB/s) says it is *not* bandwidth. It is
   most likely the 36 concurrent read streams of stride 20 736 B in pass 1 plus 4 KiB-page
   TLB pressure (183 pages/volume). The slab gather (#5) fixes streaming but costs B = 1;
   the right answer is probably to select between the two structures in the tuner by
   batch size, since the tuner already exists.
5. **Fold pass 1's `A` write into a layout where `re` and `im` of one lane block share a
   cache line** (`A[kx][flatblock][2·TW]` instead of two separate arrays). Halves the
   number of open streams in both pass 1 and the y-pass. Free at TW=4; breaks the lane
   alignment at TW=8 (see #8), so it needs pass 1 to switch to `(y, z-group)` grouping
   there and pay the 11%.

---

## Round panel_r2

### Where round 1 landed, and the diagnosis

Node (Gold 5218, panel_r1): **194.2 µs at B=1** against **L36_mixedradix's 118.4** —
1.64× behind, on the same PFA-4×9 arithmetic. Reading their strategy record against my
own pass breakdown, the gap decomposes into exactly the things their design does and
mine did not:

1. **Split-complex was the wrong layout for a batch-of-one 36-cube.** My PFA stage
   boundary was 72 live vectors (36 re + 36 im); theirs is 36 interleaved-complex
   vectors. On 32 registers mine spilled ~40 vectors per line-group (201 stack `vmov`s
   per body in the old disassembly), theirs barely spills. Their counted argument —
   which I verified independently — is that interleaved complex costs the *same*
   FMA-port pressure at every module (DFT4 2 ops/complex-lane, DFT3 1.5, CMUL 0.5,
   using swap+sign-constant FMAs for the ±i and cmul), and its extra shuffles sit on
   port 5, which the FP work leaves ~80% idle. Their 248 FMA-port ops per PW lines vs
   my 544 source ops per TW lanes is *fewer ops at equal width* once you count the
   de/re-interleave I also had to pay at the buffer boundaries and they don't.
2. **The separate 746 KB split-complex scratch was pure loss.** Resident set was
   in + A + out = 2.2 MB against their in + out = 1.46 MB; on the 1 MiB-L2 node that
   is an extra full-volume round trip through L3 (+RFO) per transform.
3. **36/8 lane waste**: my TW=8 path wasted 11% on two of three axes. Interleaved
   lanes hold complex *pairs*, and 36 is divisible by both PW=2 and PW=4, so the
   waste vanishes identically.

### Technique (rewrite, borrowed core acknowledged)

**Adopted from L36_mixedradix round 1** (this is the round's headline borrow, stated
plainly): interleaved-complex end to end, a vector lane = one 128-bit re/im pair = one
*line* of the pass; the ±i and constant-cmul realized as CSWAP (in-lane `vpermilpd`)
plus sign-alternating constant FMAs; 248 FMA-port ops + 49 port-5 shuffles per PW
lines; and the in-place discipline that keeps the resident set at in + out. Also
adopted: their **interleaved-rounds autotune protocol** (see below) and their
observation that pass-1-style access patterns (36 concurrent read streams of stride
20 736 B) exceed the L2 streamer's tracking, making a one-line-ahead software prefetch
worth offering to the tuner (their measured best distance; my round-1 prefetch
experiments had only tried 2–8 *groups*, i.e. 128–512 B, and wrongly concluded
prefetch never helps).

**Kept from my round 1** (what makes this file not a copy):

* The **pencil-fused two-pass order**: pass 1 = x-transform streaming in→mid with the
  contiguous flat (y,z) index in the lanes — with interleaved complex this pass now has
  **zero shuffles of any kind** (their phase 1 carries the z-axis lane transposes; mine
  puts the transpose-carrying axis inside the L1-resident plane pass). Pass 2 fuses y
  and z per kx-plane (20.25 KiB): y-transform shuffle-free with lanes = z, PW×PW
  complex-lane register transposes into a plane buffer P[z][ky], z-transform with
  lanes = ky reading P contiguously, transposed back on the store to out. Both
  unavoidable transposes (the round-1 proof still applies) act on L1-resident data.
* The **{width} × {store mode} self-tuning plan** with the correctness interlock
  (every candidate must reproduce the PW=2 INPLACE reference to 1e-11 before it is
  eligible), extended this round to a third axis (prefetch) and a smarter protocol.
* The **SCRATCH+NT large-batch mode**, which mixedradix does not have: pass 1 writes a
  one-volume scratch `mid` that is *reused across the batch* (so it stays cache-resident
  for the whole call), and pass 2b's final store is non-temporal. DRAM traffic per
  volume ≈ read-in + NT-write-out ≈ 1.5 MB, against INPLACE's ~2.9 MB with the RFO.
  Round-1 measurement (NT 276 vs cached 421 µs at B=32 on the dev box) predicted this;
  it is the reason for the B≥32 numbers below.

Modes: **INPLACE** (mid = out; pass 2 rewrites each plane of out in place — safe because
2a fully drains a plane into P before 2b overwrites it) for batches whose in+out
footprint can live in cache, **SCRATCH+NT** for streaming batches.

### Operation count

Per 36-point line over PW lanes: 9 DFT4 × (8 FMA-port + 1 swap) + 24 DFT3 × (6 + 1)
+ 16 CMUL × (2 + 1) = **248 FMA-port ops + 49 port-5 shuffles**, identical to
mixedradix's count (verified in my disassembly: exactly 744 FP instructions across the
three loop bodies, model exact). Per volume at PW=4: 241k FMA-port instructions vs
round 1's 528k at TW=4 — **2.2× fewer**. Register transposes add 8 shuffles per PW
vectors = 18/line at PW=4, port 5. Stack traffic collapsed from 201 `vmov (rsp)` per
body to ~52.

Floor: 62 cycles/line on one 512-bit FMA pipe (PW=4) or two 256-bit pipes (PW=2) —
identical on the Gold 5218 by construction, which is why both widths are still built
and measured rather than assumed (the corpus's AVX-512 licence-downclocking question,
still unanswered for our sizes; the node's pick this round will answer it).

### The tuner lesson this round (measured, worth keeping)

The round-1 tuner timed each candidate in one contiguous block. On a shared machine
that is a trap: three consecutive tryout runs picked configs whose steady-state times
were 63.5 / 78.6 / 112.9 µs — each run internally stable to <0.4%, i.e. the tuner
locked a bad config in and the whole scored run paid for it. Two fixes, both verified:

1. **Interleaved rounds** (adopted from L36_mixedradix): round-robin all candidates
   (12 rounds × 6 execs at B≤2, fewer at larger tb) and keep each candidate's minimum,
   so a load spike hits every candidate equally instead of assassinating whichever one
   it landed on. Caveat in fairness: some of the "mispick" evidence above is
   contaminated by the placement effect described under measurement caveats, so the
   size of this fix's benefit on wallaby is not cleanly separable; it is kept because
   it is strictly more robust for the same setup cost (~20–80 ms, unscored).
2. **A physics gate on NT**: the 110-µs outliers were NT+scratch being selected at
   B=1, where its forced 746 KB DRAM write per call costs ~50 µs and it can never win.
   NT candidates are now admitted only when batch × 1.46 MB > 16 MB (in+out beyond
   ~L3), i.e. B≥12; below that the tuner cannot be noise-tricked into it. Round-1 data
   says the gate loses nothing (NT was neutral at B≤4).

### What was measured (wallaby, Xeon Gold 6448Y Sapphire Rapids, shared and noisy
today — sd up to 30% between runs; numbers are min-of-many-runs, µs per transform)

| B | round 1 code | this round | L36_mixedradix (same day) | MKL (same day) |
|---|---|---|---|---|
| 1 | 103.7 | **55.4** | 50.5 | 76.4 |
| 4 | — | **70.0** | — | — |
| 32 | — | **69.5** | 84.0 | 100.9 |
| 256 | 127.7 (pre-tuner-fix) | **83.0** | — | 198.4 |

Correctness: `rel_l2 = 3.83e-16` at B = 1, 4, 32, 256 (tolerance 1e-12); repeatability
(bit-identical re-run) verified by tryout on every run above. Both widths validated on
wallaby natively (it has AVX-512), and the interlock re-validates every candidate at
plan time on the node.

B=1 is 1.87× faster than round 1 on the same machine; B=32 beats mixedradix by 17%
on wallaby thanks to the NT mode. At B=1 mixedradix's best min (50.5) is still ~9%
ahead of mine (55.4) on this noisy box; on the quiet node the ordering is genuinely
open — their node/wallaby ratio in round 1 was 2.34, mine 1.87.

Measurement caveats learned the hard way, both of which produce a *rock-stable wrong
number* rather than visible noise:

* **Never run two tryouts concurrently on wallaby.** A B=256 run streams ~373 MB and
  thrashes the shared L3; B=1 runs launched alongside it sat at a stable 113 µs
  (sd 0.07%) — exactly 2× the true number.
* **Wallaby has a bimodal ~2× state even for sequential runs** (roughly 1 process in 5
  today): stable ~112 µs at sd 0.3%. It hits every implementation the same way
  (mixedradix's same-day medians show it too), so it is process placement on a busy
  core/hyperthread sibling, not a plan-time mispick. Take min over ≥4 tryout
  invocations, not over samples within one, and distrust any single wallaby run —
  however stable — that is ~2× another.

### What was tried and did NOT work — with the number that killed it

1. **8 candidates timed in contiguous blocks** (the round-1 tuner protocol, just with
   more configs): 3 runs picked 63.5 / 78.6 / 112.9 µs steady states — up to 2× left
   on the table from one bad pick. Fixed by interleaved rounds + the NT gate; see above.
   Rule: *every candidate you add to a self-tuner is a new way for timing noise to hurt
   you; either interleave the measurement or gate candidates on physics.*
2. **A cheaper 9-point module (round-1 "next" item 1) — investigated and dropped.**
   In interleaved-vector currency my DFT-9 is 44 FMA-port ops (6 DFT3 + 4 cmul).
   A Winograd-style 9-point trades multiplies for additions: ~44 vector adds + 10 muls
   plus the swap overhead ≈ 54 ops — *worse* in the only currency that binds. This
   confirms L36_mixedradix's counted rejection (their round-1 item 5) from the other
   direction; the 6% "n1_9 gap" from my round-1 record was an artifact of split-complex
   accounting and does not survive the layout change. Do not revisit.
3. **Split-complex itself** — the entire round-1 design — is this round's headline
   documented dead end for L=36: identical FMA-port floor, +100% stage-boundary
   register pressure, +11% lane waste at 512-bit, plus two boundary permute passes.
   At L=8 (batch-major, 8|width) the split layout wins per L8_batchsimd's record;
   at L=36 with a batch of 1 it loses. The layout decision is geometry-specific.

### Next

1. **Read the node's tuner verdict** (width and prefetch choice at each batch) off the
   leaderboard timings if possible; PW=2-EVEX vs PW=4 on a one-FMA-unit Cascade Lake
   with the licence clock penalty is still the corpus's open gap 6, and this file now
   generates a clean A/B for it.
2. **Software-pipeline two zg-groups in pass 2a at PW=2** (32 EVEX ymm, ~18 live →
   headroom for 2 groups): the u[36] stage barrier serializes each group; two
   independent groups in flight should cover the DFT3 latency chain. Expected worth:
   the difference between my 55.4 and mixedradix's 50.5 on wallaby, if it is real.
3. **B=4 regime**: INPLACE is forced by the NT gate there (footprint 5.8 MB < L3);
   if the node's B=4 number looks bandwidth-shaped anyway, try SCRATCH+cached-stores
   as a third mode candidate — mid stays L2-resident across volumes and out is written
   once, which is one fewer volume crossing than INPLACE when out no longer fits L2.
4. **If mixedradix stays ahead at B=1 on the node**, the remaining structural
   difference is their per-x-plane fusion of the z-transposes with the streaming read
   of `in` (latency hiding) vs my dedicated shuffle-free streaming pass; port the
   winner's shape onto the loser's tuner.

---

## Round panel_r3

### Where round r2 landed, and the diagnosis

Node (Gold 5218, panel_r2): third of three in all four cells — B=1 **125.1** vs
L36_pfa's 119.3 (−5%), B=4 **151.5** vs 128.5 (−18%), B=32 **241.4** vs 202.7,
B=256 **283.9** vs 238.8 (both −19%). The r2 rewrite fixed the arithmetic (my
op count, 248 FMA-port ops/line, is identical to both rivals') but kept the
passes in the wrong ORDER: my strided x-pass ran FIRST, so its 36 concurrent
20736-B read streams hit DRAM-cold `in`, while both rivals run the plane-fused
pass first (sequential read of the only cold buffer) and point the strided pass
at the cache-warm intermediate. The monitor's VERDICT additionally named the
round's biggest open target: at B=256 the panel runs at ~6.25 GB/s single-core
when the same node demonstrably streams at 12.1 (L6_unrolled), i.e. reads and
compute do not overlap; ceiling ≈ 123 µs/vol against the measured 238.8.

### Technique (round r3)

Same PFA-4×9 interleaved-complex line kernel and PW∈{2,4} template as r2
(operation count unchanged: 248 FMA-port ops + 49 port-5 shuffles per line over
PW lanes; 241k FMA-port vector ops/volume at PW=4, floor ~105 µs at 2.3 GHz on
the node's single FMA pipe). Three structural changes:

1. **Pass order swapped** (adopted from **L36_pfa round r2**, who adopted it
   from **L36_mixedradix round 1**). Pass A fuses two axes per x-plane of `in`
   (20.25 KB, L1-resident) and writes mid[x][ky][kz]; pass B is the strided
   x-transform mid→out with an unconditional one-line-ahead prefetch on its 36
   read streams (pfa measured dropping that prefetch costs 14% at B=1; my r2
   tuner had it as a flag, now folded to always-on). Modes: INPLACE (mid=out),
   SCRATCH (reused one-volume scratch, cached stores), SCRATCH+NT (pass-B
   stores non-temporal; PW=2 pairs two flat groups so every NT write completes
   a 64-B line — pairing from L36_pfa/L36_mixedradix r2).

2. **Two pass-A variants, selected by the mode the tuner already ranks.**
   *INPLACE (small batch):* y-transform first, lanes = z — the kernel loads
   straight off the warm plane with no staging array and no load-side
   shuffles; this is exactly my r2 plane pass and it is the register-pressure-
   friendly order. *SCRATCH modes (streaming):* z-transform first with
   PW×PW transpose-on-load (pfa's phase-1 load discipline, attributed), so the
   cold `in` reads are PW sequential streams instead of 36 stride-576B ones,
   then the y-transform runs shuffle-free out of the plane buffer and stores
   straight to mid through 36 scattered 64-B store streams (stores scatter for
   free — the store buffer absorbs them; scattered *loads* were the r2
   mistake). Shuffle count is identical in both variants (one unavoidable
   transpose pair per plane element); only the placement against cold memory
   differs. Measured motivation below.

3. **Cross-volume software pipeline (XV), new this round** — the monitor's
   named target. Mode SCRATCH+NT+XV: while pass B of volume v drains NT
   stores (store-bound, load ports idle), it prefetches volume v+1's `in`
   into L3 (`prefetcht2`, 36 lines per line-group × 324 groups = exactly one
   volume). On top of that, pass A prefetches plane x+1 into L2
   (`prefetcht1`, 324 lines spread over the 9 (PW=4) z-kernel groups) while
   plane x computes — a two-stage stage-in: volume-ahead into L3 under pass B,
   plane-ahead L3→L2 under pass A.

Tuner: same interleaved-rounds + correctness-interlock protocol as r2, now over
{PW=2,4} × {INPLACE, SCRATCH, NT, NT+XV} (NT/XV admitted only when
batch × 1.46 MB > 16 MB), and the arena cap raised 24 → **64 volumes** after
re-learning L36_pfa's r2 arena lesson at a larger size (below). The tuner's
pick is now written into `fft3d_description()` (VERDICT r2 cross-cutting
item 2) so the monitor can read pw/mode per case out of the raw t_*.json.
Diagnostics for control runs: `-DFFT36PF_FORCE_PW=2|4`,
`-DFFT36PF_FORCE_MODE=0..3`, `-DFFT36PF_NOPAPF` (drop plane-ahead prefetch),
`-DFFT36PF_SKIPA/SKIPB` (phase timing, wrong answers), env `FFT36PF_VERBOSE=1`
(tuner table on stderr).

### What was measured (wallaby, Gold 6448Y; same-window A/Bs are the evidence,
cross-window numbers are context; µs per transform, driver min)

Best fast-state numbers, this round's final code:

| B | r3 final | r2 code (r2 round, same host) | L36_pfa same day | MKL same day |
|---|---|---|---|---|
| 1 | **51.6** | 55.4 | 51.1–53.2 | 113.0 |
| 4 | **71.7** | 70.0 | **126.2** (sd 0.06%) | 96.5 |
| 32 | **80.2** | 69.5 (different day) | 78.3 | 107.0 |
| 256 | **109.5** | ~135 (this code pre-change) | 104.3–109.5 | 153.4 |

Correctness: rel_l2 = 3.65–3.84e-16 at B = 1, 2, 4, 32, 256 (tolerance 1e-12);
bit-identical re-runs on every tryout above; AVX2-only pw2 path verified end to
end on the Haswell login node (B=2, 3.82e-16); clean builds under
`-march=cascadelake` and bare `-O2`. So: parity with L36_pfa at B=1/32/256 on
wallaby, and **1.76× ahead at B=4** (their 126.2 at sd 0.06% is their steady
state here, matching their own r2 record's "~120 noisy"). The node may order
B=1/256 either way — wallaby's 2-FMA units and bandwidth flatter different
things than the node's single FMA + 1 MB L2.

The same-window A/B chain that produced the design (all B=256, forced modes,
back-to-back):

* Pass order swap alone (old y-first pass A): cached 146.4 / NT 130.6 /
  **NT+XV 119.2** — XV is worth 8.7% over NT even before the pass-A fix.
* Phase split, cold: **my old pass A 101.0 µs/vol vs pfa's phase 1 at 58.1**
  (their sequential transpose-on-load reads vs my 36 stride-576B load streams —
  36 streams exceed the L2 streamer and the demand misses saturate the fill
  buffers, so even an explicit plane-ahead prefetch could not rescue it: with
  prefetch the full run only reached 119.8). Pass B side: mine 64.1 with XV's
  extra 746 KB/vol read traffic vs their 44.1 without — the XV cost is real but
  it buys back more in pass A.
* z-first pass A (transpose-on-load) + XV + plane-ahead: **110.6**, and
  `-DFFT36PF_NOPAPF` shows the plane-ahead L3→L2 stage is worth 8.5 µs/vol
  (119.1 without). Final 109.5.
* But z-first alone at B=1: 55.9–58.5 vs the y-first 51.6–53.4 — the Zv/Wv
  staging arrays (72 live vectors) spill where the y-first kernel feeds
  straight from L1. Hence the two variants keyed off the mode.

### What was tried and did NOT work — with the number that killed it

1. **A 48-volume tuner arena for the NT ranking.** tb=48 is 71.7 MB in+out,
   which still half-fits wallaby's 60 MB L3: the tuner said cached-scratch
   80.8 vs NT+XV 95.4 and picked cached; the full 382 MB run said cached
   146.4 vs NT+XV 119.2. This is L36_pfa's r2 arena lesson recurring at a
   size they had already published (they use 64 volumes); I re-learned it for
   one build. Arena now 64 volumes. *Rule restated: an NT-vs-cached decision
   is only measurable on a working set that streams past the largest L3 the
   code will meet.*
2. **Plane-ahead software prefetch as a cure for the scattered-load pass A.**
   With the y-first (36-stream) load order, adding a sequential t1 prefetch of
   plane x+1 moved B=256 only 119.0 → 119.8 (nothing). The fill-buffer-
   saturation mechanism means prefetches issued behind 36 demand-miss streams
   just get dropped; the load ORDER had to change (z-first), after which the
   same prefetch became worth 8.5 µs/vol. Prefetch placement is downstream of
   access order, not a substitute for it.
3. **One pass-A variant for all regimes.** z-first everywhere costs ~4 µs at
   B=1 (55.9 vs 51.6: staging-array spills); y-first everywhere costs ~43
   µs/vol at streaming B (the 101-vs-58 split). Neither dominates; the mode
   switch keeps both. This mirrors the r2 lesson that layout decisions are
   regime-specific, one level down.

### Attribution summary

Pass-order swap and the strided-pass-on-the-warm-buffer principle:
**L36_pfa r2** (originally **L36_mixedradix r1**). Transpose-on-load
sequential-read plane pass: **L36_pfa r2 phase 1**. PW=2 NT line-pairing:
**L36_pfa/L36_mixedradix r2**. Always-on 36-stream prefetch in the strided
pass: **L36_pfa r2** (their −14% measurement). 64-volume tuner arena:
**L36_pfa r2**. Cross-volume XV pipeline, plane-ahead L3→L2 stage-in,
mode-keyed pass-A variants, and the tuner-pick description plumbing: this
file, this round.

### Next

1. **Read the node's picks off the t_*.json descriptions** (now plumbed). My
   prediction, stated so it can be scored: pw=4 everywhere; INPLACE at B=1
   and B=4; NT+XV at B=32 and B=256. XV should be worth *more* on the node
   than wallaby's 8.7% (lower single-core bandwidth, so unoverlapped reads
   cost relatively more); if B=256 lands near max(compute ≈ 120, memory ≈
   124 µs) ≈ 125–160 rather than r2's 238.8, the monitor's overlap gap is
   substantially closed. If the node picks NT *without* XV, the t2 prefetch
   is fighting the smaller L2/L3 there — try locality 0 (NTA) next.
2. **B=1 has ~10% of headroom left against the ~105 µs port floor** (node
   119–125 across all three entries). The one structural candidate left:
   software-pipeline two y-groups/kernels in pass A so the DFT3 latency chains
   of group g+1 overlap the stores of group g (mixedradix r1 item 2, still
   untried by anyone). Worth it only if the node shows the three entries still
   bunched at ~119.
3. **The XV budget is fixed at one volume ahead.** At B=256 the L3 (22 MB)
   could hold ~10 volumes ahead; if the node still shows read stalls, a
   2-volumes-ahead cursor is a 3-line change (split the 36 lines/group between
   v+1 and v+2).
4. If pfa adopts XV (they should — it composes with their structure exactly as
   with mine), the differentiator becomes the B=4 regime, where this file is
   now 1.76× ahead on wallaby; understand why before they do: my INPLACE
   y-first pass A appears to cost far less than their phase 1 when the volume
   set cycles through L3.

---

## Round panel_r4

### Where round r3 landed, and the diagnosis

Node (Gold 5218, panel_r3): **first at B=4** (127.3 vs pfa 129.2, mixedradix
128.8), third-of-three-but-inside-spread at B=1 (121.5 vs 118.6/120.0), second
at B=32 (221.2 vs pfa 218.4) and B=256 (236.8 vs pfa 227.5). The r3 headline —
pass-order swap + mode-keyed pass-A variants — was the round's largest
improvement (−16.6% at B=256) and it moved this file from last everywhere to
contention everywhere. Predictions scored: 3 of 4 right (pw=4 everywhere,
INPLACE at B=1/B=4, NT at B=32/B=256 — but **without XV**, which the node's
own tuner rejected).

The open prize is unchanged and quantified by the monitor twice now: at B=256
the compulsory traffic is 1.49 MB/volume ≈ 124 µs at the node's demonstrated
12 GB/s, the compute floor is ~119 µs, and the measured best is 227.5 —
**1.83×, zero read/write-compute overlap**. Three entries threw prefetch
variants at it in r3 and collectively bought 4.7% in one cell while losing
7.7% in another. Why prefetch fails on the node but wins on wallaby, best
current theory (mine, stated in the code header): pass B's NT drains hold the
~12 fill buffers, and software prefetches issued behind full LFBs are simply
**dropped** — the one circumstance where they were needed. (Secondary: on
Skylake-SP-family parts `prefetcht2` fills L2, not L3, so my r3 XV was also
fighting the 1 MB L2 for space against the 746 KB scratch.)

### Technique (round r4): overlap by REAL work, not droppable prefetches

Same PFA-4×9 interleaved-complex line kernel, same PW∈{2,4} template, same
modes 0–3, same INPLACE small-batch path (byte-identical — B=1/B=4 are cells I
hold or tie and they were not touched). Pass A (streaming variant) and pass B
were refactored into per-template functions so different volume schedules can
share them; three NEW modes were added as ADDITIONAL tuner candidates — the r3
verdict's process lesson ("on this hardware pair, add candidates; do not
replace structures", proven by L8_fusedaxes/L8_radix8 vs L8_batchsimd):

* **Mode 4 PIPE** — the true cross-volume ping-pong pipeline (designed and
  deferred by **L36_pfa r3** item 5; their own record says "if B=256 lands
  ≥ 180 µs on the node, build it" — it landed 227.5). Two mid buffers; pass A
  of volume v+1 interleaves with pass B (NT) of volume v at plane granularity
  (1 pass-A plane, then 9 pass-B units, ×36). Demand loads cannot be dropped
  the way prefetches can, so volume v+1's DRAM reads are FORCED to execute
  between volume v's NT store bursts. Cost: the older mid is LRU-demoted, so
  pass B's mid reads become L3 hits (pfd bumped 8→16 doubles there).
* **Mode 5 SEQNT** — store-ORDER discipline: pass B runs **in place on mid**
  (cached stores; PFA36 loads a whole line group before its first store, so
  in-place is well-defined — same property INPLACE mode already relies on),
  then one perfectly **sequential** NT copy mid→out. Rationale: modes 2–4
  drain NT through 36 concurrent streams of stride 20 736 B — a DRAM
  row-buffer-thrash pattern — while the node's demonstrated 12.3 GB/s
  single-core stream (L6_unrolled) is sequential. Adopted from **L8_radix8
  r3** (2-pass → 3-pass to sequentialize output stores, −18.5% at B=2048) and
  the monitor's §4.3 verdict ("what actually pays at these sizes is the
  *order* traffic is issued in, not its volume"). Cost: one extra
  cache-resident volume round trip (746 KB through L2/L3; +23 328 vector
  memory ops/volume at PW=4, zero extra FMA-port work).
* **Mode 6 PIPESEQ** — mode 5 with the copy pipelined across volumes: the
  sequential NT copy of volume v−1 is interleaved into pass B of volume v
  (36 copy vectors per line group; the copy finishes exactly with the pass),
  because pass B in-place on mid is pure cache-resident compute with an idle
  memory system — the natural slot to hide the entire NT drain under. Pass
  A's cold reads hide under its own compute via the existing plane-ahead T1
  prefetch. Ideal node schedule: ~65 (pass A) + max(~60 compute, ~62 drain)
  ≈ 130 µs/volume against the 124 ceiling. Ping-pong mids as in PIPE; last
  volume's copy runs bare (amortized: 1/B).

Also new: the SCRATCH-cached mode's pass B now issues a **write-intent
prefetch** (`__builtin_prefetch(dst…, 1, 3)`, 4 lines ahead) on its 36 cold
output streams, converting demand-RFO stalls into prefetched-exclusive lines —
adopted from **L6_unrolled**'s `prefetchw` (the only prefetch variant a node
tuner actually selected in r3). INPLACE's pass B is untouched (its dst lines
are already M-state from pass A).

Tuner: same interleaved-rounds + 1e-11 correctness interlock, now over
{PW 2,4} × {modes 0–6} = 14 candidates; modes ≥ 2 still gated to streaming
batches (batch × 1.46 MB > 16 MB). Setup cost 0.4–0.6 s on wallaby at B=256
(unscored). Diagnostics unchanged, `FFT36PF_FORCE_MODE` now 0..6.

### Operation count

Line kernel unchanged: 248 FMA-port ops + 49 port-5 shuffles per 36-point line
over PW lanes; 241k FMA-port vector ops/volume at PW=4, port floor ~105 µs at
2.3 GHz. Modes 5/6 add 11 664 vector loads + 11 664 NT vector stores (the
sequential copy) and retarget pass B's 11 664 stores from out to mid; no FP
change. Mode 6's copy rides load/store ports during FMA-bound compute.

### What was measured (wallaby, Gold 6448Y; same-window forced-mode A/Bs are
the evidence; µs per transform, driver min; rel_l2 3.65–3.84e-16 and
bit-identical re-runs on EVERY run listed)

Same-window A/B chain, B=256, FORCE_PW=4, back-to-back:

| mode | µs/vol | |
|---|---|---|
| 2 scratch+nt (r3 shipping config) | 110.3 | baseline |
| 3 +xv | 107.9 | prefetch XV: −2% here, rejected by node in r3 |
| 4 pipe | 118.1 | **loses at pw4 on wallaby** (see below) |
| 5 seqnt, phase-serial | 168.2 | sequential stores ALONE lose 53% |
| 6 pipeseq | **97.8** | **−11.3% vs mode 2** |

The 5-vs-6 pair is the round's cleanest decomposition: the sequential-store
rewrite is *worthless* (−53%) until the drain is hidden under compute, then it
is the best thing on the board. Store order and drain placement are one
lever, not two.

Other cells, same windows unless noted:

* pw2 B=256: mode 2 = 132.6, mode 4 = 126.8 (PIPE **wins** at pw2), mode 6 =
  108.2. pw4 dominates pw2 in every mode.
* B=32 pw4: mode 2 = 95.7, mode 4 = 99.4; mode 6 = 90.6 (adjacent window).
* Auto-tuned full runs: **B=256 = 97.9** (tuner chose pw4/pipeseq; in-arena
  table: pipeseq 81.6, scratch-cached 82.4, xv 95.7, pipe 97.2, nt 100.2),
  **B=32 = 71.6** (tuner chose pw4/scratch-cached — correct on wallaby, where
  46 MB half-fits the 60 MB L3; the node's 22 MB L3 will re-rank this, and
  the 32-volume arena streams there), **B=1 = 52.7** in a fast window
  (r3: 51.6 — INPLACE path untouched, parity confirmed), B=4 = 129.4 (slow
  window; path untouched). vs r3 bests on this host: B=256 109.5 → 97.9
  (−11%), B=32 80.2 → 71.6 (−11%).
* Wallaby was in its documented bimodal ~2× slow placement state for most of
  the session (B=1 100–106 µs, MKL 148 µs); all A/Bs above are within-window.
* AVX2-only path exercised end-to-end on the Haswell login node (B=1, B=32
  with all modes admitted): PASS, bit-repeatable. Clean builds under
  `-march=cascadelake`, `-march=skylake-avx512`, `-march=haswell`, bare `-O2`.

### What was tried and did NOT work — with the number that killed it

1. **Mode 5 (sequential NT stores) as a phase-serial structure**: 168.2 vs
   110.3 µs/vol at B=256/pw4 — adding an unhidden 62 µs drain phase loses far
   more than store sequentiality gains on wallaby's DRAM. Only the pipelined
   form (mode 6) wins. Kept as a tuner candidate anyway: it is the control
   that separates "store order" from "drain hiding" in the node's data, and
   on a controller where strided NT is truly row-thrash-bound it could rank
   differently.
2. **Mode 4 (PIPE) at pw4 on wallaby**: 118.1 vs 110.3 — interleaving pass A
   demand reads into the NT pass costs more (mid demoted to L3, 2 MB L2
   overflowed by two mids + streams) than it recovers on a machine whose
   prefetches already survive. At pw2 it wins (126.8 vs 132.6). Kept: its
   mechanism (undropable demand loads) targets the node's specific failure,
   which wallaby cannot exhibit.
3. Attempted nothing at B=1 this round (the software-pipeline-two-groups idea
   remains untried by anyone); the INPLACE path ships byte-identical, so the
   B=1/B=4 cells cannot regress from this round's changes — only the tuner's
   candidate list grew, and modes ≥ 2 are physics-gated out at those batches.

### Attribution summary

Sequentialized output stores via an extra cache-resident pass: **L8_radix8
r3** (+ the monitor's §4.3/§5 store-order verdict). Hide-the-drain-under-
compute cross-volume pipelining: the monitor's r2/r3 §6 ceiling analysis;
ping-pong two-scratch structure: **L36_pfa r3** (designed, deferred, and
explicitly bequeathed at their measured 227.5). Write-intent prefetch on cold
output streams: **L6_unrolled** (node-validated in r3). Add-candidates-never-
replace process rule: **r3 VERDICT §4** (L8_fusedaxes/L8_radix8 vs
L8_batchsimd evidence). The passB-in-place-on-mid trick and the
copy-interleaved-into-passB schedule: this file, this round.

### Predictions for the node (stated so they can be scored)

* Picks: B=1/B=4 pw4 INPLACE (unchanged). B=32 and B=256: **pw4 pipeseq** if
  the strided-NT row-thrash theory is right; pw4 scratch+nt if it is not. If
  the node picks scratch-CACHED at B=32 like wallaby did, that is a mis-tuned
  arena and I want to know (arena = 32 vols = 46 MB > 22 MB L3, so it should
  stream correctly there).
* B=256: **175–215 µs** (from 236.8). Mode 6 removes the exposed drain phase;
  the node pays more than wallaby for the extra mid round trip (1 MB L2) but
  also gains more from sequentializing NT (weaker memory controller). Under
  175 would mean the drain really was the whole story; over 215 means the
  exposed term is pass A's reads and the next lever is below.
* B=32: **185–215 µs** (from 221.2, and pfa's 218.4 is the cell target — this
  is the cell where the panel's margin over MKL fell to 1.01×).
* B=1: 118–125, B=4: 125–130 (unchanged paths, spread only).

### Next

1. **Read the node's picks per cell** (plumbed since r3). The five-way mode
   menu {nt, xv, pipe, seqnt, pipeseq} is itself the forced-variant
   instrument the monitor asked for — whichever wins, the in-arena ranking
   answers which mechanism the node actually lacks.
2. **If pipeseq wins but lands > 175 at B=256**: pass A's cold reads are the
   remaining exposed term; spread a fraction of the copy into pass A (copy
   chunks between planes) so the drain also covers pass A's read stalls, or
   revisit paced-prefetch distances inside pass A now that pass B no longer
   competes for LFBs.
3. **B=4 regime**: INPLACE wins it, but a cached-store variant of pipeseq
   (copy with plain stores, no NT) would give B=4 the same drain-hiding
   without the NT-at-small-batch penalty; worth one candidate if the node
   shows B=4 memory-shaped.
4. **B=1** remains a three-way tie 13% above the port floor; the only untried
   structural lever is software-pipelining two line-groups (mixedradix r1
   item 2). Somebody should finally measure it, even to kill it.

---

## Round panel_r5

### Where round r4 landed, and the diagnosis

Node (Gold 5218, panel_r4): third at B=1 (122.5 vs mixedradix 119.0), third at
B=4 (132.4 vs 129.9 — lost the cell r3 held), **third at B=32 (219.6 vs
L36_pfa's 174.2, −26%)** and third at B=256 (242.2 vs pfa's 218.9, and a real
+2.3% regression against my own r3 with the same `scratch+nt` pick). The
r4 headline machinery (PIPE/SEQNT/PIPESEQ) was selected by the node tuner in
**zero cells** — the verdict promoted this file explicitly as the round's
*instructive failure*.

The decisive information in the r4 data is not my regression, it is **how
L36_pfa won both streaming cells: `pw=4 mode=inplace pf=1`** — no scratch, no
NT stores, no pipeline. Sequential pass A directly into `out`, strided pass in
place on `out` with plain cached stores (RFO and all), one-line-ahead prefetch
on the 36 strided streams, paced software prefetch of the `in` read stream,
and ~62 KB pre-coverage of the *next* volume's input. The node rejected every
NT variant on the board (also L8: `avx512-3p-pfs`, plain stores, won both
streaming cells). Meanwhile MY tuner could never discover this: my INPLACE
(mode 0) is hardwired to the y-first pass A, whose 36 stride-576B load streams
on DRAM-cold input are this file's own documented 2× loss (101 vs 58 µs/vol,
r3). So at B≥32 my candidate set contained *no viable in-place option at all*
— the structural reason the tuner fell back to `scratch+nt` and lost by 26%.

### Technique (round r5): mode 7 ISTREAM + two tuner protocol fixes

Line kernel, both widths, modes 0–6, INPLACE/B=1 path: all byte-identical to
r4. Three changes, all additive:

1. **Mode 7 ISTREAM** (adopted from **L36_pfa r4**, their node-winning
   `inplace pf=1` configuration, translated onto my kernels — stated plainly:
   this is the round's headline borrow). Per volume: `passA_plane` (z-first
   transpose-on-load, sequential cold reads, plane-ahead T1 prefetch) writes
   straight into `out`; then `passB_cached` runs the x-transform **in place
   on `out`** (PFA36 loads a whole line group before its first store, the
   same property modes 0/5/6 already rely on), with its existing
   one-line-ahead prefetch on the 36 strided read streams, plus **3 lines of
   the next volume's input prefetched per line group** (T1; 324 groups × 3
   lines ≈ 62 KB at PW=4 — L36_pfa's PFNX trick, so the next volume's pass A
   never starts cold). No new kernels: the mode is ~10 lines of dispatch.
   Admitted at every batch size (not NT-gated). DRAM traffic per volume =
   read `in` + RFO `out` + write `out` ≈ 2.2 MB against scratch+nt's
   nominal 1.5 MB — the node's r4 verdict is that the *order* wins over the
   volume: 36-stream strided NT is a row-buffer-thrash pattern, and cached
   stores let the L2/LLC absorb and re-order the drain.

2. **Self-warming tuner measurements** (new, this file). Found while
   validating mode 7: with `inner=1` at large tb, each candidate's in-arena
   measurement inherits the *previous* candidate's cache state. istream sits
   right after pipeseq in the rotation; pipeseq ends with an NT sweep that
   flushes `dout`, so istream was charged full RFO misses its own steady
   state never sees — **167.4 µs/vol in-arena vs 89.8 end-to-end at B=32
   (pw4, wallaby)**, an 86% phantom penalty that would have buried the new
   candidate on the node too. Fix: one untimed exec of the candidate before
   each timed rep (tuner cost ×2, still unscored ~1 s at B=256). After the
   fix: istream 90.2 in-arena = its end-to-end number. *Rule for the record:
   an interleaved-rounds tournament is only fair if every candidate is timed
   from its own steady-state cache; NT-mode candidates poison whoever runs
   next.*

3. **3% simplest-wins hysteresis** (adopted from **L36_pfa r4**,
   node-validated): among correct candidates within 3% of the best in-arena
   time, install the structurally simplest mode (inplace < istream < scratch
   < nt < seqnt < xv < pipe < pipeseq); at equal mode keep the faster width
   (width is not a complexity axis — on the node's single 512-bit FMA unit
   the two widths can genuinely tie, and B=1 is decided by ~3%).

### Operation count

Unchanged: 248 FMA-port ops + 49 port-5 shuffles per 36-point line over PW
lanes; 241k FMA-port vector ops/volume at PW=4. Recomputed floor with the
r4 verdict's measured 3.89 GHz sustained clock (not 2.30): **~62 µs/volume**
on one 512-bit FMA pipe — B=1's 122.5 is 2.0× the floor, not 1.16×; the whole
panel has more B=1 headroom than every prior record assumed. Mode 7 adds 972
prefetch µops per volume and zero FP.

### What was measured (wallaby, Gold 6448Y; rel_l2 3.65–3.84e-16 and
bit-identical re-runs on every run; µs per transform, driver min)

Calibration A/B — my mode 7 vs the actual node winner, **L36_pfa r4 exemplar
forced to `inplace pf=1`**, built privately from `exemplars/panel_r4/`, same
windows, B=256: **pfa 157.1, mine 156.6** (and 40088 vs 40219 µs/call in the
adjacent window — parity). The translation is faithful; on the node their
config measured 218.9 (B=256) and 174.2 (B=32), which is where mode 7 should
land.

Forced end-to-end, pw4, same window: B=256 mode 0 (inplace y-first) 184.6 vs
**mode 7 161.2** (−13% — the z-first pass A on cold input, again); B=32
mode 0 88.1 vs mode 7 89.8 (parity — B=32's 46 MB is L3-marginal on wallaby,
so the cold-read order cannot matter here; it matters on the node's 22 MB L3).
Same-window NT ladder at B=256/pw4 for context: mode 2 114.3, mode 6
(pipeseq) 98.9, mode 7 156.6 — wallaby still loves NT, exactly as in r4, and
the node demonstrably does not; both stay in the candidate set and the node's
own (now self-warming) tournament decides.

In-arena tables after the self-warming fix (B=32/pw4): inplace 102.1, scratch
78.1, nt 101.5, seqnt 91.7, pipeseq 93.2, **istream 90.2**; wallaby's tuner
picks scratch — legitimate *on wallaby*, where the 64-volume arena half-fits
the 60 MB L3 (the documented r3/r4 arena artifact; on the node the same arena
is 4.4× L3 and streams).

Auto-tuned full runs (all PASS, bit-repeatable): B=1 **51.45** (best this
file has ever recorded on wallaby; INPLACE path untouched), B=4 72.7 (fast
window; 81.4 in a 25%-sd window), B=32 72.5–76.7, B=256 105.0–106.4. AVX2
path verified end-to-end on the Haswell login node (B=2, 3.646e-16, PASS,
repeatable); clean builds under `-march=cascadelake` and bare `-O2`.

### What was tried and did NOT work — with the number that killed it

1. **Trusting the r4 tuner protocol for the new candidate.** In-arena istream
   read 167.4 µs/vol at B=32 against a true 89.8 — not noise but a
   deterministic predecessor-state bias (it ran after pipeseq's NT flush).
   Any conclusions drawn from earlier in-arena tables where a candidate
   followed an NT mode carry this taint; the r4 in-arena ranking that
   preferred scratch-cached over istream-like shapes is exactly the artifact.
   Fixed by self-warming (above), verified: 167.4 → 90.2 = end-to-end.
2. **Nothing else was touched.** B=1/B=4 paths ship byte-identical; modes
   3–6 remain as candidates (add-don't-replace, r3 verdict) but the r4 node
   data says they will not be picked, and that is fine — mode 7 is the bet.

### Attribution summary

ISTREAM structure (in-place streaming two-pass with cached stores + paced
input coverage) and the PFNX next-volume pre-coverage: **L36_pfa r4** (their
node-winning `inplace pf=1`; originally the in-place discipline is
**L36_mixedradix r1**). 3% simplest-wins hysteresis: **L36_pfa r4**. The
self-warming tournament fix and the predecessor-poisoning diagnosis: this
file, this round. Clock re-derivation input (3.89 GHz probe): **L6_unrolled
r4** via the r4 VERDICT §5.

### Predictions for the node (stated so they can be scored)

* Picks: B=1 pw4 inplace (unchanged), B=4 pw4 inplace, **B=32 pw4 istream,
  B=256 pw4 istream** (istream should beat scratch+nt by ≫3% there, as
  pfa's identical structure did by 26% and 10%).
* B=32: **170–185 µs** (pfa's same-structure 174.2 ± my pass-A/pacing
  differences). B=256: **210–225** (their 218.9 ±). B=1: 119–124 and B=4:
  128–133 (untouched paths, spread only; the hysteresis may flip B=4 pw4→pw2
  only if they truly tie, costing ≤3% by construction).
* If the node still picks scratch+nt at B=32/256, the self-warming fix
  changed the in-arena ranking in the wrong direction on that machine and I
  want the forced `FFT36PF_FORCE_MODE=7` number the monitor can take.

### Next

1. **If istream lands as predicted**, the remaining streaming gap to the
   ~124 µs traffic ceiling is the RFO on `out`: try a *paced* `prefetchw`
   (write-intent, L36_pfa's r4 verdict shows L6's prefetchw was the one
   prefetch the node ever picked) on pass A's `out` stores in istream mode —
   it converts the RFO stall into a prefetched-exclusive line without
   changing store order.
2. **B=1 is 2.0× the recomputed 62 µs floor** (3.89 GHz, single 512-bit FMA).
   The untried lever remains software-pipelining two line-groups so DFT3
   latency chains overlap across groups (mixedradix r1 item 2, still nobody
   has measured it, four rounds later). At 2× the floor this is now the
   largest identified prize on the whole L=36 board and it is compute-side,
   so wallaby CAN model it — do it first thing next round.
3. **B=4 regime**: mixedradix holds it (129.9 vs my 132.4). Both entries run
   inplace there; the difference is inside the small-batch pass A. Diff their
   plane pass against mode 0's y-first order before spending a round.

---

## Round panel_r6

### Where round r5 landed, and the diagnosis

Node (Gold 5218, panel_r5): 2nd at B=1 (121.255 vs pfa 120.358 — but §3b of the
verdict flags my three runs as 134.1/125.6/121.3, so the placing rests on one
run), 3rd at B=4 (132.656), **3rd at B=32 (186.903 vs pfa 168.565) and 3rd at
B=256 (230.243 vs pfa 182.598, −26%)**. The node picked my modes exactly as
predicted (pw4 inplace at B=1/B=4, pw4 istream at B=32/B=256), so the loss is
not a mispick: the istream *structure* is behind. Two named causes in the r5
verdict:

1. **L36_pfa's r5 headline: `pf=2`** — a paced write-intent prefetch
   (`prefetchw`) over phase 1's in-place out store stream — was selected by the
   node 3/3 at BOTH streaming cells and took them 174.2→168.6 and 218.9→182.6.
   Their in-arena decomposition: inplace-pf2 90.5 vs inplace-pf1 156.6 at B=256
   (**−42%** — the cold-out RFO was the dominant exposed cost of precisely the
   mode the node runs me in). My istream was their pf=1 twin; my own r5 "Next"
   item 1 predicted this lever and this round ships it.
2. **The residual gap**: my istream measured wallaby-parity with their exemplar
   (156.6 vs 157.1) yet landed 10.9% behind their pf=1 on the node — the
   verdict's "most precisely localised unexplained number on the board", to be
   closed by diffing the two files, not by writing new mechanisms. The diff
   (done this round, mechanism level): their phase-1 read prefetch (PFIN) is a
   **paced cursor** 4096 doubles (32 KB) ahead, advanced 18 lines per loop
   iteration through BOTH subloops, so the DRAM read stream stays busy during
   the y-transform half of every plane; my r5 scheme issued the whole next
   plane (324 lines) during the FIRST subloop only, then went silent for the
   second — on wallaby's bandwidth that difference is invisible, on the node's
   it plausibly is the 10.9%. Everything else on the monitor's candidate list
   (PFNX depth 3 lines/group, phase-2 one-line-ahead pattern, in-place pass-B
   discipline) was already identical, and the monitor's own fingerprint check
   confirms arithmetic-order identity at B=32/B=256.

### Technique (round r6): mode 8 ISTREAM+PFW + the byte-faithful PFIN pacing

Line kernel, both widths, modes 0–7, the INPLACE/B=1 path: all unchanged.
Three additive changes:

1. **Mode 8 ISTREAM+PFW** (adopted from **L36_pfa r5**, their node-winning
   `inplace pf=2` — stated plainly, this is the round's headline borrow, and
   it is the second round running I port their node winner one round behind
   them). Mode 7 plus a paced write-intent cursor over pass A's out stores:
   `__builtin_prefetch(p,1,3)` (emits `prefetchw`), FFT36PF_PFWD = 2592
   doubles = one plane ahead, 18 lines per iteration through both pass-A
   subloops — pacing arithmetic identical to the read cursor, values
   identical to pfa's node-selected configuration. Gated with the streaming
   modes (batch × 1.46 MB > 16 MB): prefetchw on cache-resident lines is pure
   µop tax (pfa in-arena +13%/+11% at B=1/B=4, L6_unrolled +17%). Mode 7
   stays in the candidate set as the pf=1 control. Hysteresis rank: inplace <
   istream < istream+pfw < scratch < … (a 3%-tie falls to the simpler mode).
2. **Pass A's read prefetch is now pfa's PFIN, byte-faithful**: paced cursor
   FFT36PF_PFD = 4096 doubles ahead of the plane being consumed, advanced
   PFSTEP = 36·PW doubles (18 lines at pw4) per iteration in BOTH subloops
   (2·(36/PW) iterations × PFSTEP = exactly one plane per plane), hint T1,
   cursor recomputed per plane (`inv + PFD + x·2592`) so pipe modes can still
   call planes individually. Replaces the r5 next-plane-dump in ALL streaming
   pass-A callers (modes 1–8; mode 0's y-first pass A untouched). Cursors run
   harmlessly past the volume end; the next volume's first 62 KB is still
   pre-covered by the PFNX 3-lines-per-group in pass B (modes 7/8).
3. **PFA36X2, a diagnostic** (`-DFFT36PF_PAIRB`): two independent line-groups
   stage-interleaved in pass B, to finally measure the software-pipeline-two-
   groups idea (mixedradix r1 item 2, on every L=36 "Next" list for five
   rounds, named again in the r5 verdict §6). Result: **dead** — see below.

Tuner: same interleaved-rounds + self-warming + 1e-11 interlock protocol, now
{pw 2,4} × {modes 0–8} = 18 candidates. `FFT36PF_FORCE_MODE` now 0..8;
`FFT36PF_PFD` / `FFT36PF_PFWD` are `-D`-overridable for the monitor's sweeps.

### Operation count

Unchanged: 248 FMA-port ops + 49 port-5 shuffles per 36-point line over PW
lanes; 241k FMA-port vector ops/volume at PW=4; ~83 µs/volume single-FMA-pipe
floor at the measured 2.89 GHz AVX-512 licence clock (r5 verdict §5). Mode 8
adds 11 664 `prefetchw` µops/volume (one per out line) and zero FP. The
cascadelake disassembly carries the prefetchw sites (verified, 22 rolled
sites) alongside the read-prefetch population.

### What was measured (wallaby, Gold 6448Y; the r5 verdict's clock-lottery
finding applies — only same-window A/Bs quoted as evidence; µs per transform,
driver min; rel_l2 3.654–3.836e-16, rel_max ≤ 4.9e-16, bit-identical re-runs
on every run listed)

Forced end-to-end A/Bs, same window, alternating runs (the round's evidence):

| cell | mode 7 (istream) | mode 8 (+pfw) | Δ |
|---|---|---|---|
| B=256 pw4 | 155.1 / 156.1 / 156.7 | **105.1 / 106.1 / 110.7** | **−31%** |
| B=32 pw4 | 86.0 | **77.9** | −9% (B=32 is L3-marginal on wallaby; the node's 22 MB L3 should show more) |

In-arena tables (self-warming tuner, one tournament each): B=256 pw4
istream+pfw **87.0** vs istream 136.0 (−36%), pipeseq 85.4, scratch 81.1;
B=32 pw4 istream+pfw **75.7** vs istream 91.4 (−17%), scratch 72.0. Wallaby's
tuner picks scratch at both (its documented 60 MB-L3 arena artifact, r3/r5
records); the node's own tournament decides there, and the node has rejected
scratch modes in every round since r3 while picking exactly this in-place
shape 3/3.

Auto-tuned full runs (all PASS): B=1 **52.3** fast-window (r5: 51.45 — path
untouched, parity), B=4 87.4 (noisy window, path untouched), B=32 72.7,
B=256 100.1. AVX2-only path verified end-to-end on wombat (B=2 auto PASS
3.818e-16; B=32 forced mode 8 PASS 3.654e-16, bit-repeatable — on Haswell,
which lacks PRFCHW, gcc lowers the write-intent hint safely). Clean builds:
`-march=native` (Haswell), `-march=cascadelake`, bare `-O2`, and with
`-DFFT36PF_PAIRB`.

### What was tried and did NOT work — with the number that killed it

1. **Software-pipelining two line-groups (PFA36X2), measured at last after
   five rounds on the panel's Next lists.** Forced mode 0, B=1, alternating
   same-window runs on wallaby: pw4 paired 54.9–55.1 vs baseline 53.4–54.8
   (**+1–3%, a loss**); pw2 paired 94.0 vs baseline 80.2 (**−17%,
   decisively worse** — the doubled live set spills exactly as the r1
   register arithmetic predicted). Conclusion for the whole L=36 board: the
   out-of-order window already covers cross-group overlap at PW=4 (a ~400-µop
   group body against a 224-entry ROB leaves nothing for software pipelining
   to add), and at PW=2 the register cost is ruinous. The B=1 gap to the
   floor is NOT latency-chain serialization. Kill it from the Next lists; the
   diagnostic stays in the file (`-DFFT36PF_PAIRB`) if anyone wants the node
   number.
2. Nothing else regressed or was reverted; the round was deliberately narrow
   (one borrowed node-proven mechanism, one pacing port, one measurement).

### Attribution summary

Write-intent paced cursor on the in-place out stream (mode 8) and its exact
constants (PFWD 2592, 18 lines/iteration, T0 hint on the w-side): **L36_pfa
r5** (`pf=2`, node-selected 3/3 at both streaming cells; ultimately from
**L6_unrolled r3**'s prefetchw, node-confirmed via **L6_pfa r4**). Read-side
paced-cursor pacing through both subloops (PFD 4096, T1): **L36_pfa r3
PFIN**, ported byte-faithfully per the r5 verdict §6's file-diff instruction.
The PFA36X2 measurement vehicle and the kill of the two-group pipeline
hypothesis (originally **L36_mixedradix r1** item 2): this file, this round.

### Predictions for the node (stated so they can be scored)

* Picks: B=1 pw4 inplace, B=4 pw4 inplace (both unchanged), **B=32 pw4
  istream+pfw, B=256 istream+pfw** (width open at B=256 — the node took
  pfa's pw2 there in r5; both widths of mode 8 are candidates).
* **B=256: 180–200 µs** (from 230.2; pfa's same-mechanism 182.6 is the
  target, and my pass A now carries their exact pacing — if the pacing port
  closed the r5 residual, I land at 182±5; if only the RFO part transfers,
  nearer 195).
* **B=32: 165–178** (from 186.9; pfa's 168.6 ± the same residual logic).
* B=1: 119–125, B=4: 129–134 (untouched paths, spread only).
* If mode 8 is picked but B=256 still lands >205, the remaining exposed term
  is pass A's read stream despite the pacing port, and the next lever is the
  monitor's `FFT36PF_PFD` sweep (2048/4096/8192) — one env-free `-D` A/B per
  value.

### Next

1. **If the pacing port closed the residual**: parity with pfa at streaming
   batch; the differentiator moves to B=1/B=4, where all three entries sit
   1.44× above the 2.89 GHz port floor and the two-group pipeline is now a
   documented dead end. The remaining B=1 suspects, in order: front-end (the
   ~1100-µop unrolled plane bodies vs the 1.5k-µop DSB — measurable only via
   the monitor's `perf stat -e idq.dsb_uops,idq.mite_uops`), and the pp
   plane-buffer round trip in pass A (fusable in principle by keeping the
   9 Wv blocks of a yb-group in registers through the y-transform at pw2 —
   32 EVEX ymm make 36+9 live vectors feasible; unverified).
2. **If mode 8 is rejected on the node** despite pfa's identical-mechanism
   3/3 selection, something in my pacing differs from theirs in a way the
   file diff missed; ask the monitor for forced `FFT36PF_FORCE_MODE=7` vs
   `=8` node numbers at B=256 — one pair settles it.
3. **B=4** remains the cell nobody has attacked structurally (all three
   entries run plain inplace there); if the node shows B=4 memory-shaped,
   the gated-out mode 8 at B=4 (5.8 MB working set, streams on nothing) is
   correctly excluded and the lever is elsewhere — probably the same
   front-end story as B=1.
