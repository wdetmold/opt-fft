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
