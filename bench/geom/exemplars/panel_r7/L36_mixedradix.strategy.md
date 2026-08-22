# L36_mixedradix — strategy record

Geometry: `L = 36`, cube `36^3 = 46656` complex doubles = 746 496 B = 729 KiB per volume.
File: `impl/L36_mixedradix.c`. `fft3d_name()` = `L36_mixedradix`.

---

## Round 1

### Technique

Row–column: three 1-D passes, each pass a **Good–Thomas (prime-factor) 4×9**
36-point line codelet, **batch-vectorised across the lines of the pass** (a vector
lane = one line, never a point inside a line), interleaved complex kept end to end,
and the passes fused into **two sweeps over the volume** rather than three.

#### The line transform: PFA 4×9, no twiddles

36 = 4·9 with gcd(4,9) = 1, so Good–Thomas applies. With the Ruritanian input map
and the CRT output map

```
n = (9*n1 + 4*n2) mod 36            n1 in [0,4), n2 in [0,9)
k = (9*k1 + 28*k2) mod 36           [9^-1]_4 = 1,  [4^-1]_9 = 7
```

then

```
n*k = 81 n1 k1 + 252 n1 k2 + 36 n2 k1 + 112 n2 k2
    ==  9 n1 k1 +       0  +       0  +   4 n2 k2   (mod 36)
so  W36^{nk} = W4^{n1 k1} * W9^{n2 k2}
```

i.e. `X[(9k1+28k2) mod 36] = sum_{n1} sum_{n2} x[(9n1+4n2) mod 36] W4^{n1k1} W9^{n2k2}`
— **nine 4-point DFTs then four 9-point DFTs, with zero twiddle multiplies
between the stages**, only a compile-time index permutation. Both permutations are
literal constants inside the unrolled macros (`(9*0+4*N2)%36` etc.), and since every
pass is already a strided gather/scatter, the permutation costs *nothing* — it
changes displacements, not the number of memory operations. This is exactly the
`LITERATURE.md` §3.5 item 1 / §02 §5.4 recommendation.

The 9-point module is Cooley–Tukey 3×3 (9 = 3² is a prime power, so PFA cannot help
inside it): `A[b][r] = DFT3_a(u[3a+b])`, twiddle by `W9^{br}` (nontrivial only for
`(b,r) ∈ {(1,1),(1,2),(2,1),(2,2)}` → `W9^1, W9^2, W9^2, W9^4`), then
`v[r+3s] = DFT3_b(B[b][r])`.

A pleasant identity that fell out and is worth recording: `9k1 + 28k2 ≡ k1 (mod 4)`,
so **DFT9 number `k1` writes exclusively to output slots congruent to `k1` mod 4**.
With a 4-complex-lane vector that means each DFT9 fills exactly one lane position of
all nine output transpose groups. (It does *not* let you fuse the output transpose —
a group still needs all four `k1` — but it is the reason the z-pass output buffer is
exactly 36 vectors and not more.)

#### Operation count, as vector instructions

Counted per 36-point line, over PW complex lanes, all on interleaved complex:

| module | FMA-port ops | port-5 shuffles | count | subtotal |
|---|---|---|---|---|
| DFT4 | 8 | 1 | ×9 | 72 + 9 |
| DFT3 | 6 | 1 | ×24 | 144 + 24 |
| CMUL | 2 | 1 | ×16 | 32 + 16 |
| **36-point line** | **248** | **49** | | per PW lines |

Derivations (interleaved complex, `s = sqrt(3)/2`):

* **DFT4** — `t0=x0+x2, t1=x0-x2, t2=x1+x3, t3=x1-x3` (4), `y0=t0+t2, y2=t0-t2` (2),
  `sw = swap(t3)` (1 shuffle), `y1 = vfmsubadd(t1,1,sw)`, `y3 = vfmaddsub(t1,1,sw)` (2).
  The `×(−i)` is absorbed into the alternating-sign FMA, so it costs no extra
  instruction, only two wasted flops per lane.
* **DFT3** — `t=x1+x2, m=x1-x2` (2), `y0=x0+t` (1), `a=vfnmadd(0.5,t,x0)` (1),
  `ms=swap(m)` (1 shuffle), `y1=vfmadd(ms,KS,a)`, `y2=vfnmadd(ms,KS,a)` (2), where
  `KS = [s,-s,s,-s,...]` so that `ms*KS = -i*s*m` exactly. **6 ops.** Folding
  `d = s*m` into the two output FMAs instead of computing it once trades 2 flops for
  1 instruction — the right trade per §02 §2.7 / §04 §5.1. 6 is the floor: you need
  2 for `t,m`, and 1 per output.
* **CMUL by a constant `(c+id)`** — `q = swap(v)*D` (1 shuffle + 1 mul),
  `r = vfmaddsub(v,C,q)` (1). **2 FMA-port ops.**

Real flops: `9*20 + 4*(6*18 + 4*6) = 180 + 528 = 708` per line, vs the corpus
reference PFA-4×9 count of **688** flops / 464 scalar instructions (§02 §5.4). The 20
extra flops are the two unit-multiplier FMAs in DFT4; they buy one instruction each.
Per volume `3 * 36^2 * 708 = 2 752 704` flops over **3888 line transforms**. The
driver's nominal yardstick is `5*46656*log2(46656) = 3.618` Mflop, so the nominal
GF/s figure is ~1.31× the true one.

#### Layout, lanes, and the one unavoidable transpose

Interleaved complex, unchanged from the driver's layout — **no de-interleaving pass
exists at all**. A vector holds `PW` complex numbers = `PW` consecutive `z` values =
`PW` different *lines* of the y and x passes. Consequences:

* Every twiddle is lane-invariant → all constants are pre-splatted 64-byte rows in
  `.rodata`, used as memory operands, costing zero registers (`KC_*` arrays).
* **Zero cross-lane operations inside the transform** for the y and x passes: their
  transform axis is strided (36 or 1296 complex) and the lane axis is contiguous.
* The **z pass is the exception** — its transform axis *is* the contiguous one — so
  it needs a `PW×PW` transpose of 128-bit complex lanes on the way in and on the way
  out: 2 `vperm2f128` per 2 vectors at 256 bits, 8 `vshuff64x2` per 4 vectors at
  512 bits. Both are involutions, so one macro serves both directions.

I measured this cost: the z pass runs at 182 cycles/line against 113 (y) and 108 (x)
on the dev machine, i.e. **the transposes cost +65% on one of three passes, +22%
overall**. I convinced myself it is irreducible — see "what did not work" #3.

Interleaved vs split complex (§4.4 of `LITERATURE.md`, flagged there as
"strongly motivated but unproven"): I counted both and they are **exactly equal on
FMA-port pressure** at every module — DFT4 2 ops/lane both ways, DFT3 1.5, CMUL 0.5.
Split saves the shuffles, but the shuffles sit on port 5 which is only ~20% loaded,
so they are free. Interleaved wins on everything else here: no de/re-interleave pass,
and `36/8` is not an integer so a split 512-bit vector (8 real lanes) cannot tile the
z axis, while `36/4` is. **Conclusion for this size: interleaved, and the §04
recommendation of split-complex does not transfer to a batch-of-one 36-cube.**

#### Blocking: two sweeps, not three

```
phase 1, per x-plane (36x36 complex = 20.25 KiB, fits the 32 KiB L1):
    for each block of PW y-rows:  load PW rows (contiguous),
        transpose into lanes, 36-point z transform, transpose back,
        store into a 20.25 KiB `plane` scratch
    for each block of PW z:  36-point y transform, plane -> out   (stride 72 doubles)
phase 2:
    for each y, for each block of PW z:  36-point x transform, in place in `out`
        (stride 2592 doubles = 20736 B)
```

* `out` doubles as the inter-phase buffer, so the resident footprint is `in` + `out`
  = 1.46 MB rather than 2.2 MB with a separate scratch. On the 1 MiB-L2 target part
  the 729 KiB volume *fits L2*, so phase 2 should run out of L2 and the DRAM/L3
  traffic collapses to the compulsory `read in + write out`. (The dev Haswell has
  only 256 KiB L2, so locally phase 2 goes to L3 — see "what was measured".)
* Phase 2's tile is `PW` consecutive z = exactly one 64-byte line per x at PW=4, so
  **every touched cache line is consumed in full**, and as `zb` advances each of the
  36 x-streams walks forward one line at a time — 36 sequential streams, more than
  the L2 streamer tracks, hence the explicit prefetch (below).
* Loop order `(y outer, zb inner)` means for fixed `(x,y)` the nine `zb` tiles are
  nine *consecutive* cache lines. Confirmed: no reordering beat it.
* **No padding.** The natural strides are already benign: x-stride 20736 B = 324
  lines, `324 mod 64 = 4`, so the 36 x's spread over 16 of the 64 L1 sets at 2.25
  per set against 8-way associativity; in L2 `gcd(324,1024) = 4` gives a 256-long
  orbit. y-stride 576 B = 9 lines, `gcd(9,64) = 1`, all 64 sets. So the `Nx=Ny=37`
  padding that §04 §7.3 prescribes is **unnecessary in the driver's interleaved
  layout** — that recommendation is for a batch-minor layout with a 64-byte granule,
  which is not what this file uses. Skipping it also keeps every access naturally
  64-byte aligned: `(x*1296 + y*36 + zb*PW)*16` with `4 | 1296` and `4 | 36`.

#### Three kernels, autotuned on the node

`AVX-512 is not automatically a win on this part, and you must measure it` — so I
did not guess. The kernel source is instantiated three times by self-inclusion
(`#include __FILE__` guarded on `VAR`), each under its own `#pragma GCC target` so
all three compile even on an AVX2-only dev box:

| variant | ISA | lanes | registers | why it might win |
|---|---|---|---|---|
| V0 | `avx2,fma` | 2 complex / ymm | 16 | no AVX-512 frequency licence at all |
| V1 | `avx512f` | 4 complex / zmm | 32 zmm | **half the dynamic instructions per line** |
| V2 | `avx512vl,avx512f` | 2 complex / ymm16-31 | 32 ymm | 32 registers *and* 256-bit licence |

crossed with prefetch distance ∈ {off, 1 line, 4 lines} = **9 candidates**.
`fft3d_create()` (a) runs each AVX-512 candidate and rejects it unless its output
matches V0's to 1e-13 relative, (b) times all survivors over 4 interleaved rounds on
`min(batch,4)` volumes and keeps the per-candidate minimum, (c) installs the winner
as a function pointer. Setup costs 30–60 ms and is excluded from the score.

Why V1 is expected to win, from static counts (`objdump`, one `exec` body):

| | total instrs | arith | shuffles | stack `vmov` | uses ymm/zmm16-31 |
|---|---|---|---|---|---|
| V0 (AVX2, 256b) | 1525 | 744 | 151 | **201** | no |
| V1 (AVX-512, 512b) | 1572 | 744 | 293 | **88** | yes |
| V2 (AVX-512VL, 256b) | 1445 | 744 | 155 | **123** | yes |

All three contain exactly `3 × 248 = 744` arithmetic ops — the model is exact. V1's
loop trip counts are 9 where V0/V2's are 18, so V1's *dynamic* count is roughly half
per line, and its spill traffic is 2.3× lower because 36-entry intermediate arrays
fit far better in 32 registers. On a Gold 5218 (one 512-bit FMA unit) 512-bit and
256-bit have **identical FMA-port peak** — 248 ops/cycle-limited either way, 62
cycles/line — so the win is entirely front-end, and the loser is whichever path
runs out of issue bandwidth first. That prediction is what the autotune is for.

### What was measured

Dev machine: **Xeon E5-2680 v3 (Haswell, AVX2 only), 2.50 GHz measured** (an
independent 10-way-unrolled `vfmadd` loop gives 4.991 ops/ns = exactly 2 FMA/cycle at
2.50 GHz), 32 KiB L1d, **256 KiB L2**, 60 MiB L3, and *shared with 11 other
implementer agents*, so run-to-run spread is 5–25%. Only V0 (AVX2) ever executes
here; the AVX-512 variants are compiled and dispatch-guarded.

Per transform, best of many runs (`--samples 15..30`, driver's own minimum):

| batch | µs / transform | nominal GF/s | rel_l2 vs numpy |
|---|---|---|---|
| B = 1 | **191.3** | 18.9 | 3.966e-16 |
| B = 4 | 189.2 | 19.1 | 3.963e-16 |
| B = 16 | 203.3 | 17.8 | 3.963e-16 |
| B = 64 | 289.3 | 12.5 | 3.961e-16 |

Correctness: **3.96e-16 relative L2 at every batch size tested (1, 3, 4, 16, 64)** —
five orders of magnitude inside the 1e-12 gate. Repeatability is exercised by
construction: the driver does `warmup + inner*samples` executes before the checked
one, and phase 1 rewrites every element of `out` before phase 2 reads it, so the
in-place phase 2 is idempotent across calls.

**The AVX-512 path is numerically verified too, without an AVX-512 machine.** I built
a scratch copy in which the eleven `_mm512_*` primitives the V1 kernel uses are
replaced by scalar C functions written from the Intel SDM pseudocode — including the
two easy-to-invert ones (`VFMADDSUB` = subtract on even lanes, `VFMSUBADD` = add on
even lanes), `VPERMILPD` with `imm=0x55`, and `VSHUFF64X2`'s 128-bit-lane select
(low half from `a`, high half from `b`). Linked against the real driver, the
emulated V1 kernel gives **rel_l2 = 3.959e-16 at B=3 and 3.966e-16 at B=1** — bit-
comparable to V0. That validates the 4×4 lane transpose, the `PW=4` index
arithmetic, and the FMA-variant sign conventions. V2 shares V0's source text
verbatim (only the target pragma differs), so V0's run validates it.

Per-pass breakdown (V0, B=1, separately compiled phases, cycles per 36-point line at
2.50 GHz):

| pass | µs | cycles/line | uops/line (counted) |
|---|---|---|---|
| z (contiguous axis, 2 transposes) | 94.4 | **182** | 328 |
| y (stride 72) | 58.8 | **113** | 220 |
| x (stride 2592, in place) | 55.9 | **108** | 220 |

So we run at ~2.0 uops/cycle and ~1.8× the FMA-port floor (126 cycles per 2-line
call on Haswell, where `vaddpd` is port-1-only). The isolated y-codelet in a pure-L1
loop hits **162 cycles/call = 81 cycles/line = 1.29× the floor**, so roughly half the
in-situ gap is memory, not issue.

**Memory is what limits the dev machine, and the target part is different.** Total
traffic is `read in (729 KiB) + write out (+RFO) + read out + writeback` ≈ 3 MB per
transform; at 191 µs that is ~16 GB/s, right at single-core L3 bandwidth on a
contended Haswell. Because the 729 KiB volume does *not* fit this machine's 256 KiB
L2 but *does* fit the Gold 5218's 1 MiB L2, phase 2 should become an L2-resident
sweep on the scoring node and the local numbers are pessimistic. This is exactly the
"do not tune to this node's L2" warning of `LITERATURE.md` §3.5 item 9, from the
other direction.

### What was tried and did NOT work

1. **Making the line strides runtime arguments** (one generic `dft36_ss(src,ss,dst,ds)`).
   The *standalone* function spends **67 integer `add` + 4 `imul` + 3 `shl`** on
   address arithmetic out of 584 instructions. Specialising to literal strides 72 and
   2592 was worth **0%** end-to-end (194.2 → 194.5 µs, inside noise) because gcc
   already forward-propagates the constants at the inlined call sites. Kept anyway
   (it costs nothing and guarantees the property), but do not expect a win from it.

2. **AVX-512 assumed rather than measured.** Deliberately not done. On a Gold 5218
   the single 512-bit FMA unit makes 512-bit and 256-bit *arithmetically identical*
   (248 ops/cycle-limited, 62 cycles/line either way) while 512-bit costs a frequency
   licence, so "wider is faster" is not derivable — hence the 9-way runtime autotune
   with a numerical admission test. **This is new information the corpus does not
   have (§4.8 gap 6): put whatever the node reports into the next record.**

3. **Eliminating one of the z-pass's two transposes.** Four rearrangements were
   worked through on paper and every one is a wash, so do not redo them:
   * *z-pass stores transposed (`P[z][y]`), y-pass transposes on input* —
     total 2 transposes and total 432 memory ops per PW lines, **identical** to the
     current 2-in-the-z-pass arrangement, because the buffer round-trip just moves
     from the z pass to the y pass.
   * *explicit whole-plane 36×36 transposes around a stride-72 z codelet* —
     **290 uops/line vs 182**, because a plane transpose moves every element (36
     ld + 36 st per line) whereas the in-register version moves only PW lines' worth
     (9 + 9 per line at PW=4). Strictly worse.
   * *fusing the transpose into stage 1's loads* — impossible: transpose group `g`
     position `p` feeds `ST1(n2 = (g − 2p) mod 9)`, so each group's PW outputs go to
     PW *different* DFT4s and all 36 must be materialised first.
   * *doing the z axis with lanes = z (an in-register 36-point FFT)* — the PFA-4×9
     stage-1 elements sit 9 apart, so DFT4 becomes a cross-lane butterfly: ~36
     shuffles + 36 arith per line for stage 1 alone (vs 18 arith batched) and stage 2
     goes from 44 ops per 4 lines to 44 per line. **~11× worse.** Confirms
     `LITERATURE.md` §2 item 2 emphatically.

4. **Cooley–Tukey 6×6 instead of PFA 4×9.** Counted: 12 DFT6 (18 vector ops each,
   itself PFA 2×3) + 24 nontrivial twiddle CMULs = **264 vector ops vs PFA's 248**,
   and it needs a twiddle table. PFA 4×9 wins. Consistent with §01 §8's flop numbers
   (CT 4×9 ≈ 704, CT 6×6 ≈ 678, PFA 4×9 = 688 by their accounting).

5. **4K-aliasing avoidance by shifting the `plane` scratch per x-plane.** Real
   effect, but not worth the machinery. An offset scan of the isolated y-codelet
   showed **162 → 268 cycles/call purely from the source buffer's page offset**
   (a 1.65× swing — worth knowing about). Because `pout` advances 20736 B ≡ 256 B
   (mod 4096) per plane, locking the relative offset needs an x-dependent plane
   pointer `plane + ((x & 15)*32) + off`. End to end that bought **194 → 191 µs
   (~2%)**, and with prefetching enabled the sensitivity almost vanished
   (185–190 µs across all offsets). Removed as not worth the risk of tuning an
   offset against `out`'s alignment, which is unknown at `fft3d_create()` time.
   *If someone revisits this, the diagnostic is the win: a page-offset sweep of a
   single codelet exposes 4K aliasing instantly.*

6. **Prefetch distance beyond one line.** {0,1,2,4,8,16} lines ahead measured; 1 line
   was best or tied in the quiet runs and the rest were inside the machine's noise
   floor. Rather than pick, distances {0,1,4} went into the autotune. Prefetching is
   worth 3–20% locally depending on machine load; on an L2-resident node it may well
   tune to *off*, which is fine — that is what the candidate list is for.

7. **A separate full-volume scratch instead of working in place in `out`.**
   Identical traffic (write scratch + read scratch + write out vs write out + read
   out + write out) but 2.2 MB resident instead of 1.46 MB, which is the difference
   between fitting and not fitting the target's 1 MiB L2. Rejected on that basis.

8. **Split-complex layout** (the §04 recommendation). Not built, and the counting
   above is why: identical FMA-port pressure at every module, the shuffles it saves
   are on an idle port, `36/8` is not an integer, and it would add a de-interleave
   and a re-interleave pass that interleaved complex does not need. Recorded as a
   *counted*, not measured, rejection.

### Next

1. **Read the node's autotune choice and hard-code the ranking.** The first thing to
   learn from the scoring run is which of the nine candidates won and by how much;
   that settles §4.8 gap 6 for this project. If V1 wins big, drop V0/V2 from the
   candidate list and spend the register headroom (32 zmm) on item 2.
2. **Software-pipeline two line transforms in the V1 kernel.** V1 spills only 88
   times where V0 spills 201, i.e. it has register headroom that is currently idle,
   and the codelet's own critical path (stage 1 must fully complete before stage 2's
   first DFT3, ~36 + 88 cycles) is longer than its port time. Interleaving two
   independent 36-point transforms should hide the stage boundary. Expected 10–20% on
   the two-thirds of the work that is *not* the z pass; it will not help V0, which is
   already spilling.
3. **Fuse phase 2 into phase 1 at z-block granularity for large batches.** B=64
   measures 289 µs/transform against B=1's 191 — the volume plus streams stop fitting
   the cache. Restructuring phase 1 to emit nine `[x][y][PW z]` slabs of 82.9 KiB
   each, then running phase 2 slab by slab, keeps the phase-2 working set in L2
   independently of L2 size. Same instruction count, better locality; the 1.5×
   batched/non-batched gap is the size of the prize.
4. **Huge pages.** Not attempted: the driver owns the buffers, so `madvise()` has
   nowhere to go at `fft3d_create()` time. `LITERATURE.md` §3.5 item 6 argues this
   matters at L=36; against that, phase 1 is sequential and phase 2 touches only 36
   pages per tile (inside a 64-entry L1 dTLB), so I expect it to be small here. Worth
   one `perf stat -e dTLB-load-misses` on the node before spending effort.
5. **A cheaper 9-point module.** The four DFT9s are 176 of the 248 ops per line
   (71%). §02 §5.4 points at Johnson–Burrus / Temperton hand-derived 9-point modules,
   but they are minimum-*multiplication* constructions and 9 = 3² admits no
   Good–Thomas, so per §02 §2.7 expect this to lose on instructions. Low priority;
   item 2 is the better use of a day.

---

## Round panel_r2

### Where round 1 landed (node, panel_r1)

Won L=36 outright at B=1 (118.4 µs vs MKL 163.6, 1.38×) and B=4 (128.4 vs 174.4),
kept a thin lead at B=32 (204.7 vs 220.6), and **lost B=256 to MKL by a hair**
(247.4 vs 246.3). So the batched regime was the whole round-2 target: the 2.1×
slide from B=1 to B=256 is memory, not arithmetic — the codelets don't know what
B is.

### What changed: a streaming store path for large batches

The arithmetic, PFA 4×9 line codelet, lane discipline, and two-phase structure
are all unchanged. What changed is where phase 1 lands and how phase 2 stores:

```
cached path (unchanged, small batch):   NT path (new, large batch):
  phase 1:  in -> out  (per x-plane)      phase 1:  in -> vol scratch (729 KiB,
  phase 2:  out -> out, in place                     reused every volume)
                                          phase 2:  vol -> out, non-temporal
                                                    stores, then one sfence
```

Why this is the right shape: at large B, `out`'s volume is cold at every phase-1
write, so the cached path pays *read in (729K) + RFO on out (729K) + eventual
writeback of out (729K)* ≈ 2.19 MB of DRAM per volume. With phase 1 landing in a
**reused** one-volume scratch (cache-resident after the first volume, so its RFO
is an L2/L3 hit) and phase 2 streaming the final result, DRAM traffic drops to
*read in + NT write out* = 1.46 MB — the compulsory minimum for an out-of-place
transform. Predicted 1.5×; measured 1.39× at B=256 (below).

**Borrowed, with attribution:**
* The NT-stores-on-the-final-write idea and its magnitude are from
  **L36_pencilfused round 1** (measured 421 → 276 µs at B=32, 1.53×, on the dev
  Haswell). My structure needed the volume scratch to make the final write the
  *only* touch of `out`; theirs already had one.
* The **threshold form of the decision** is from **L8_fusedaxes** (NT iff the
  batch working set exceeds L3, read via `sysconf(_SC_LEVEL3_CACHE_SIZE)`,
  fallback 22 MiB): here `use_nt = batch * 1.49 MB > 1.25 * L3`.
* The reason it is a threshold and *not* a tuner comparison is **L6_pfa's round-1
  record**: their timing tuner picked NT at B=8 from a small tuning buffer and
  lost 21%. My tuning buffers are ≤ 4 volumes (always L3-resident), so a timed
  NT-vs-cached comparison there would be systematically wrong in the same way.
  The threshold picks the *policy*; the tuner only ranks kernels within it.

Mechanics worth recording:

* **512-bit NT is trivial** — every phase-2 store is a full 64-byte line at a
  64-byte-aligned address (`(y*36 + zb*4)*16` with `64 | y*576, 64 | zb*64`, plus
  `k*20736` with `64 | 20736`), so `_mm512_stream_pd` drops in.
* **256-bit NT is not**: 32-byte NT stores are half a line, and a phase-2 tile
  touches 36 *different* lines — far beyond the ~10–12 write-combining buffers —
  so naive `_mm256_stream_pd` would evict partial lines constantly. Fix: two
  adjacent z-tiles are transformed into a 36×2-vector stack stage (2.3 KiB,
  L1-hot), then flushed with two back-to-back 32-byte NT stores per line, which
  the WC buffer merges into one full-line write. Costs 72 extra L1 stores + 72
  loads per tile-pair (~9% more memory µops on the 256-bit NT path only).
* The NT path exists per kernel as a fourth and fifth body (`exec_<V>_3` plain,
  `exec_<V>_4` with a one-line `prefetcht0` on the phase-2 scratch reads — on the
  node's 1 MiB L2 the 729 KiB scratch is partly evicted to L3 by phase 1's `in`
  stream, which wallaby's 2 MB L2 cannot reproduce, so the choice is left to the
  node's own tuner). All are admitted through the same 1e-13 numerical gate
  against `exec_0_0`, with fallback to the cached candidates if allocation or
  admission fails. A diagnostic `FFT36_NT=0|1` override (read once in
  `fft3d_create`, so executes stay repeatable) forces the policy for A/B runs.

### Operation count

Unchanged: 248 FMA-port vector ops + 49 shuffles per 36-point line over PW lanes,
708 real flops/line, 2 752 704 flops/volume. The NT path adds no arithmetic at
PW=4 and only the stage-flush traffic above at PW=2.

### What was measured (wallaby, Sapphire Rapids Gold 6448Y, AVX-512, 2 MB L2, 60 MB L3)

µs per transform, driver minimum. Wallaby's load varied hugely during this round
(sd 0.2% in quiet windows, 5–26% when other implementers piled on), so the
paired A/B runs are the trustworthy rows:

| batch | path | this round | forced NT=0, same conditions | MKL same run |
|---|---|---|---|---|
| 1 | cached (unchanged) | **102.2** (sd 0.03%, quiet) | — | 80.7 |
| 4 | cached (unchanged) | **73.9** (one noisy-window run; treat as soft) | — | 97.4 |
| 32 | cached (footprint 47.8 MB < 75 MB threshold here) | **84.7** (sd 0.2%) | — | 107.2 |
| 64 | **NT** | **90.9** | — | 174.1 |
| 128 | **NT** | **170.5** (sd 0.4%) | 179.7 (sd 12%, noisy window) | 180.6 |
| 256 | **NT** | **126.6–129.3** (sd ≤ 2%) | **179.6** (sd 0.5%) | 155.1 |

(One B=1 run in a noisy window printed min = 71.8 µs against its own median of
103.7 — a glitchy minimum, not a real speed; the quiet-window 102.2 with
matching median is the number to trust. Same lesson as observation 1 below.)

The B=256 A/B is the round's result: **179.6 → 126.6 µs, 1.4× from the store
policy alone**, against the 1.5× traffic-model bound. Correctness at every batch
tried (1, 4, 32, 64, 128, 256): rel_l2 = 3.95–3.96e-16, bit-identical re-runs.
The 256-bit staged-flush NT path was also exercised *end to end* on the AVX2 dev
box (it is the only NT candidate there): B=64 = 256.3 µs vs 289.3 in round 1
(−11% even on Haswell), PASS at 3.96e-16, repeatable.

On the scoring node the threshold puts B=32 and B=256 on the NT path (footprints
47.8 / 382 MB against 27.5 MB threshold), B=1 and B=4 on the unchanged cached
path. Node prediction: B=256 from 247 µs to ~170–185 µs (the in-read of 187 MB
is uncacheable there, unlike wallaby's B=64 case); B=32 from 204.7 to ~150–165
(its `in` is 23.9 MB against a 22 MiB L3, so partially resident); B=1/B=4
unchanged.

### What was tried / observed that did NOT work or nearly misled

1. **Trusting single-shot wallaby baselines.** My pre-change "baselines"
   (B=32 = 124.9, B=128 = 177.9) were taken in a noisy window; re-measuring the
   *identical* code path at B=32 later gave 84.7 µs. Any conclusion drawn from
   comparing those two numbers across runs would have been fiction. The habit
   that survives contact: only paired A/B runs (the `FFT36_NT` override exists
   for exactly this), taken back-to-back, count as evidence on a shared machine.
2. **B=128 vs B=256 per-volume anomaly** (170.5 vs 129.3, both NT, both tight
   sd): unexplained — same code path, same per-volume traffic. Both buffers
   exceed wallaby's 60 MB L3, so it is not residency of `in`. Suspects: NUMA
   placement of the 191 vs 382 MB allocations on the 2-socket wallaby, or THP.
   Not chased because the scoring node is the monitor's measurement anyway; do
   not burn a round on wallaby-specific memory topology.
3. **Letting the tuner choose the store policy** — not tried, deliberately, on
   the strength of L6_pfa's documented mis-pick (their B=8, −21%). Recorded here
   so nobody re-litigates it: with ≤ 4-volume tuning buffers the comparison is
   structurally biased against NT.

### Next

1. **Read the node's verdict on 512-bit vs 256-bit NT** (and cached): still the
   open §4.8-gap-6 question, now doubled — Cascade Lake's single 512-bit FMA and
   frequency licence vs the staged-flush overhead of the 256-bit NT path. Ask
   the monitor to report which candidate won if the leaderboard doesn't say.
2. **Software-pipeline two line transforms in the V1 kernel** (round-1 Next #2,
   still undone): B=1 is ~1.13× the FMA-port floor on the node (118.4 vs ~105 µs
   at 2.3 GHz), so the ceiling is ~10%. Worth a day only if B=1 leadership comes
   under threat.
3. **`prefetchnta` on the phase-1 `in` reads** (cf. L36_pfa's Next #3), so the
   input stream stops evicting the 729 KiB `vol` scratch from the node's 1 MiB
   L2 in the first place. The reactive half of this (a `prefetcht0` NT variant
   for phase 2, `exec_<V>_4`) is already in the candidate set this round; the
   NTA half needs the node to show whether phase-2 L2 misses are actually what
   remains. Wallaby's 2 MB L2 cannot see any of this.
4. **B=4 is the weakest remaining ratio** (128.4 vs B=1's 118.4 on the node,
   footprint 5.9 MB, all-cached): likely L2 conflict between `in`, `out` and the
   plane scratch as volumes rotate. A small experiment with per-volume plane
   offsets could recover a few percent; low priority.

---

## Round panel_r3

### Where round panel_r2 landed (node, p55n3 — NOTE: different physical node than r1)

Lost all four cells to `L36_pfa` for the first time: B=1 120.3 vs their 119.3
(inside spread), B=4 129.7 vs 128.5 (inside spread), B=32 **231.4 vs 202.7**,
B=256 **261.5 vs 238.8**. My own four cells regressed vs r1
(+1.6/+1.1/+13.1/+5.7 %), but the MKL control moved +17.8 %/+24.9 % in exactly
the B=32/B=256 cells on the new node, so the monitor could not attribute my NT
regression and asked for an `FFT36_NT=0|1` control run (VERDICT §6a — still
wanted, the override is still in). The VERDICT also quantified the round's
biggest open prize: at B=256 the panel runs 238.8 µs against a
`max(compute 119, 1.49 MB / 12.1 GB/s)` ≈ **123 µs ceiling — ~1.9× sitting in
un-overlapped memory time**, and noted nobody has built a cross-volume pipeline.
That is what this round builds.

### What changed: the cross-volume pipeline (code 5, "xv"), plus two fixes

**1. Cross-volume input prefetch on the NT path** (`exec_<V>_5` =
pf1 + xv). Observation: with the reused-scratch NT structure, volume b's
phase 2 is NT-store-drain-bound and issues *no DRAM reads* (scratch is
cache-resident), while volume b+1's phase 1 then pays a serial DRAM read of
`in`. So phase 2 now prefetches volume b+1's `in` with `_mm_prefetch(..., T1)`
(into L2, sparing L1), paced **one 36-line granule per phase-2 tile**:
324 tiles × 36 lines = 11 664 lines = exactly one 746 KB volume, and the pace
matches the rate at which phase 2 retires scratch lines, so on an LRU L2 the
incoming stream tends to replace dead scratch. Cost: 11 664 prefetch
instructions per volume ≈ 2.4 % more instructions, phase 2 only, no arithmetic
change. Same granule logic in the PW=2 staged path (tile-pair = same 36-line
granule). Idea and magnitude estimate: **the monitor's panel_r2 VERDICT §6
(L=36 item b)** — this is that suggestion, implemented.

**2. The NT tuner arena now must stream on the machine doing the tuning.**
My r2 tuner ranked NT candidates on a ≤4-volume (6 MB) arena — fine for
*policy* (that is threshold-decided) but it ranks pf/xv variants under
cache-resident conditions. First attempt was a fixed 32-volume arena; wallaby
promptly demonstrated the bug: 47.8 MB fits its 60 MB L3, the tuner dropped
xv, and the auto run scored 129.3 µs/vol while a forced `FFT36_XV=1` run of
the same binary scored 112.6. Fix: arena = in+out footprint of **2.5× this
machine's L3** (sysconf, fallback 22 MiB), clamped to [32, 128] volumes and to
the batch. On wallaby that is 106 volumes (158 MB/call, streams); on the node
37 (55 MB/call, streams). After the fix the auto-tuner picks xv on wallaby and
lands at 113.9. This is **L36_pfa's round-2 lesson #1** ("a tuning arena must
actually stream on the machine with the largest L3 you will meet"), borrowed
with attribution and made machine-relative instead of hard-coded. Setup cost
grew to ~1.4 s at B=256 on wallaby (~0.7 s expected on the node) — excluded
from the score and comparable to L17_winograd's 1.2 s.

**3. Tuner picks are now readable off the leaderboard.** `fft3d_create()`
snprintf's the winning candidate into a static buffer that
`fft3d_description()` returns, e.g.
`"PFA 4x9 2-sweep, lanes=lines; pick=v1-nt-pf1-xv (B=256, arena=106 vol,
ntpolicy=1, 9 cand)"` — verified in the driver's `--json` output at B=1 and
B=256. Mechanism from **L6_pfa** (r2's only fully closed prediction loop);
requested by VERDICT cross-cutting item 2. A new `FFT36_XV=0|1` override
(read once at plan time, mirrors `FFT36_NT`) forces the xv candidates out/in
for paired A/B runs.

### Operation count

Unchanged: 248 FMA-port vector ops + 49 shuffles per 36-point line over PW
lanes, 708 real flops/line, 2 752 704 flops/volume. xv adds 11 664 `prefetcht1`
per volume (+2.4 % instructions) on the NT path only; nothing else moved.

### What was measured (wallaby, Sapphire Rapids Gold 6448Y, 2 MB L2, 60 MB L3)

Wallaby again toggled fast/slow states between runs (same as r2); paired
back-to-back A/Bs are the evidence, single numbers are context only.

B=256 (streams everywhere: 382 MB), µs per transform, driver min, alternating
`FFT36_NT` unset (auto=NT), 3 pairs:

| run | XV=0 | XV=1 |
|---|---|---|
| pair 1 | 129.8 (sd 0.23 %) | **112.6** (sd 0.15 %) |
| pair 2 | 147.3 (sd 0.22 %) | **112.8** (sd 0.09 %) |
| pair 3 | 145.9 (sd 0.36 %) | **112.8** (sd 0.13 %) |

xv is **−13 % against XV=0's best window and −23 % against its typical one**,
and it flattens the window-to-window variance (the prefetched read is
insensitive to whatever was perturbing the demand-read runs). Auto-tuned run
after the arena fix: 113.9 µs/vol, pick=`v1-nt-pf1-xv`. r2's number on this
machine was 126.6–129.3, so the round is −11 % to −13 % on wallaby at B=256.
Correctness at every batch tried (1, 4, 32, 256): rel_l2 = 3.95–3.96e-16,
bit-identical re-runs, all four PASS.

B=32 forced NT (real B=32 on wallaby is cached by the threshold), 2 pairs:
XV=0 84.5 / 86.1 µs/vol, XV=1 94.0 / 96.9. **xv loses ~12 % when `in` is
L3-resident** — there is no DRAM read to hide and 11 664 prefetches are pure
overhead plus L2 pollution. Expected and acceptable: the tuner sees exactly
these conditions (arena = min(batch, stream-size) = the real footprint when
the batch itself does not stream) and will keep xv off wherever it loses.

B=1: 54.05 µs, PASS (cached path — code untouched this round). For the
record: that is half of r2's quiet-window 102.2 on the same machine, while
MKL in the same two runs went the other way (80.7 → 150.6). Same binary
semantics, different day — one more instance of this record's rule 1
(cross-run wallaby comparisons are fiction; only the paired A/Bs above count).

### What was tried / observed that did NOT work

1. **A fixed 32-volume tuning arena** (first version of fix 2): on wallaby it
   sits inside L3, the tuner dropped xv, and the auto run gave 129.3 µs/vol vs
   112.6 forced — a 15 % mis-pick from ranking a streaming decision on a
   cache-resident arena, reproducing L36_pfa's lesson on my own tuner one
   abstraction level down (they mis-ranked the *policy*; I mis-ranked the
   *variant within the policy*). Machine-relative sizing fixed it; the number
   pair above is the receipt.
2. **xv at L3-resident batch sizes** (wallaby B=32 forced-NT): −12 %, numbers
   above. Not a code failure — a regime boundary the tuner now handles — but
   recorded so nobody hard-enables xv unconditionally.

### Predictions for the node (stated so they can be scored)

* Expected picks: B=1/B=4 `v1-cached-pf*`; B=32 and B=256 NT by threshold,
  xv's fate per-case readable from the description string.
* B=256: `in` streams from DRAM (382 MB), same shape as wallaby B=256, but the
  node's 1 MB L2 must hold the 729 KB scratch *and* the incoming prefetch
  stream, so some prefetched lines will be evicted before phase 1 uses them —
  expect a smaller xv win than wallaby's 13–23 %. From r2's 261.5: **~205–235
  µs** if the node's L36-batched anomaly (MKL +25 %) persists, ~180–210 if it
  was the other node. Beating L36_pfa's 238.8 is the target.
* B=32: `in` is 23.9 MB against 22 MB L3 — barely streaming; xv should be a
  small win or a tuner-rejected tie. From r2's 231.4: **~200–225 µs**.
* The B=32/B=256 `FFT36_NT=0|1` control the VERDICT asked for is still wanted,
  and `FFT36_XV=0|1` now exists for the same purpose on this round's change.

### Next

1. **Read the node's per-case pick strings** (now plumbed); if xv lost at
   B=256 there, the L2-competition explanation above is the suspect — try a
   half-volume pacing (prefetch only during the last 162 tiles, halving
   residency time) or `prefetchnta` on the *scratch* reads so scratch does not
   need L2 retention and `in` can have it.
2. **Extend the pipeline to phase 1** (prefetch the tail of in(b+1) during
   phase 1(b+1) itself, a few KB ahead): covers whatever xv's L2 residency
   loses, and the first volume of every call, which xv cannot touch.
3. **B=1 software pipelining of two line transforms** (round-1 Next #2, still
   undone): B=1 is now a three-way 1 % race on the node (119.3/120.3/125.1);
   ~10 % of front-end headroom is documented in round 1's static counts.
4. If the node's L36-batched anomaly persists on p55n3, ask the monitor to pin
   the node or run one `perf stat -e cycles,ref-cycles` L36 B=256 job — MKL
   +25 % on a fixed binary is a machine effect, not a code effect, and it caps
   what any B=256 number can mean.

---

## Round panel_r4

### Where round panel_r3 landed (node, p55n3)

Held B=1 (118.626 µs, first — but all three L36 entries within 1.2%, inside
spread), second at B=4 (128.794 vs pencilfused 127.304), and **lost both
batched cells, falling behind MKL itself**: B=32 233.434 (L36_pfa 218.351,
MKL 220.506), B=256 264.531 (L36_pfa 227.497, pencilfused 236.824, MKL
247.568). The node's tuner **rejected my xv candidates** at B=32 and B=256
(picked `v1-nt-pf1`) — the cross-volume full-volume prefetch that won 13–23%
on wallaby does not survive the node's 1 MB L2. The VERDICT also named my
tuner unstable twice: B=1 picks flipped pf4/pf0/pf1 across the three runs
(118.6/119.7/123.2, a 3.9% self-inflicted spread), and B=32/B=256 flipped
pf0/pf1. Meanwhile the round's ranking mechanism at L=36 was clear: L36_pfa's
**paced phase-1 input prefetch** was the only thing that moved a batched cell
on the node (−4.7% at B=256), and the phase split published by pencilfused
(their pass A cold: 101.0 µs with 36-stream loads vs 58.1 with sequential
loads) plus the monitor's 6.25 GB/s effective-rate arithmetic all point at the
same mechanism: **phase-1 `in` reads are LFB-limited demand misses**
(~10 outstanding × 64 B / ~100 ns ≈ 6.4 GB/s single-core), and my code had no
input prefetch at all inside phase 1 — xv prefetched only the *next* volume,
from phase 2, a full volume ahead of use.

### What changed: paced phase-1 input prefetch ("pfin", code 6), and a stable tuner

**1. `exec_<V>_6` = nt-pf1-pfin — borrowed from L36_pfa round panel_r3
(their PFIN + PFNX), with attribution.** A T1 cursor runs `PFIN_D` = 4096
doubles = 32 KB ahead of the plane phase 1 is consuming; each of the
2·(36/PW) codelet calls per x-plane issues `PFIN_L` = 36·PW/8 line-prefetches
(18 at PW=4, 9 at PW=2) and advances, so exactly one plane of prefetches
issues per plane processed — 324 lines/plane, 11 664 per volume, +3.6%
instructions on the NT path only, zero arithmetic change. The yb subloop
consumes `in` at 2× the cursor rate (the zb y-subloop touches no `in` bytes),
so the cursor falls up to 10.4 KB behind mid-plane; 32 KB of distance absorbs
that (pfa measured the 16–32 KB plateau; I shipped their 32 KB). At each
volume boundary the cursor is naturally 32 KB into `in[b+1]`; because
phase 2's 729 KB of scratch reads would evict that L2 window before phase 1
returns to it, phase 2 re-covers it with 3 T1 lines per tile (324 tiles
= 62 KB, pfa's PFNX shape). Cursor clamps at the batch end; prefetches are
architecturally side-effect-free so all candidates still pass the 1e-13
admission gate against `exec_0_0`, and `FFT36_PFIN=0|1` (read once at plan
time) mirrors `FFT36_NT`/`FFT36_XV` for paired A/Bs. xv candidates are
**kept**, ordered after pfin — the r3 VERDICT's process lesson ("add
candidates; do not replace structures") applied literally.

**2. Tuner hysteresis + more sampling.** Candidates are now listed
simplest-first per kernel (pf0, pf1, pfin, xv) and a later candidate must
beat the incumbent by **>1%** to be installed; small arenas get more work
(reps 16 at nt<4, 4 at nt<16; rounds 6 below nt=16, 4 above). This directly
attacks the VERDICT §3 finding: near-tied candidates flipping between runs
now resolve to the simplest one, and a spurious win by an exotic candidate
must clear a 1% bar. Verified on wallaby: three consecutive B=1 plans all
picked `v1-cached-pf1` (r3's code flipped picks run to run).

**3. Latent build break fixed.** The V2 variant's pragma was
`target("avx512vl,avx512f")`, which does not imply FMA — under a bare `-O2`
build (no `-march`) every `_mm256_fmadd_pd` in V2 failed with
"target specific option mismatch". Never bit on the node (it builds
`-march=native`), but the pragma now says `avx512vl,avx512f,fma`; verified
clean under `-O3 -march=cascadelake -mtune=cascadelake` and bare `-O2`.

### Operation count

Unchanged: 248 FMA-port vector ops + 49 shuffles per 36-point line over PW
lanes, 708 real flops/line, 2 752 704 flops/volume. pfin adds 11 664
`prefetcht1` in phase 1 + 972 in phase 2 per volume (NT path only, when
selected).

### What was measured (wallaby, Sapphire Rapids Gold 6448Y, 2 MB L2, 60 MB L3)

Paired back-to-back A/Bs at B=256 (382 MB, streams; µs/volume, driver min;
same binary, forced candidate sets):

| pair | pf1 only (`FFT36_PFIN=0 FFT36_XV=0`) | pfin (`FFT36_PFIN=1`) | xv (`FFT36_XV=1`) |
|---|---|---|---|
| 1 | 128.6 (sd 0.34%) | **99.7** | 114.2 (sd 0.92%) |
| 2 | 126.5 (sd 0.49%) | **97.5** (sd 0.60%) | 123.6 |

**pfin is −23% against pf1 and beats xv (r3's winner here) by 13–21%.** The
full 12-candidate auto tournament picks `v1-nt-pf1-pfin` and the quiet-window
end-to-end run gives **98.4 µs/vol (sd 0.08%)** — r3's best on this machine
was 112.8. MKL in the same window: 212 µs/vol.

B=32 forced-NT (the marginal regime where xv lost 12% in r3; 47.8 MB fits
wallaby's L3 so `in` is L3-resident): pf1 102.2 / 87.0 vs pfin 94.4 / 88.1
across wallaby's fast/slow toggling — i.e. **pfin is neutral-to-positive
where xv was a 12% loss**, because it prefetches 32 KB ahead of use instead
of parking a whole volume in L2. The 1% hysteresis decides the true ties.

B=1: **50.4 µs** (quiet window, sd 0.36%), pick `v1-cached-pf1`, stable
across three plans. B=4: 72.7 µs/vol best (noisy window), `v1-cached-pf1`,
ntpolicy=0 — cached path untouched this round.

Correctness: rel_l2 = 3.95–3.96e-16 at B = 1, 4, 32, 64, 256; bit-identical
re-runs everywhere; the PW=2 (AVX2 V0) pfin path exercised end-to-end on the
Haswell login node at B=64 (auto pick `v0-nt-pf1-pfin`, PASS 3.962e-16,
output bit-identical to the reference pick). Setup cost: 1.73 s at B=256 on
wallaby (~0.9 s expected on the node's 37-volume arena), excluded from score.

### What was tried / observed that did NOT work

1. **xv is now a dominated candidate on both machines I can see.** Direct
   A/B above: pfin beats xv by 13–21% at wallaby-B=256, and the node already
   rejected xv twice. Kept in the list (it costs only tuner time and must
   now also clear the 1% hysteresis bar) but nobody should extend the
   full-volume-ahead approach; the paced-cursor form supersedes it.
2. Nothing else was removed or replaced; this round was deliberately narrow.

### Predictions for the node (stated so they can be scored)

* Picks: B=1/B=4 `v1-cached-pf*` (stable across runs now); B=32/B=256 NT by
  threshold, and `v1-nt-pf1-pfin` should be selected in both if the paced
  prefetch lifts the node's ~6.25 GB/s demand-read rate the way it lifted
  wallaby's.
* B=256: from 264.5 → **~200–230 µs**. pfa's PFIN bought them only −4.7%
  (238.8→227.5) on the node, but my baseline carries more exposed read
  latency (I had no phase-1 prefetch at all), so my delta should be larger;
  parity with L36_pfa (~227) is the target, and anything under 247 retakes
  MKL.
* B=32: **~205–225 µs**. The arena is exactly 32 volumes (= the real
  regime); if pfin does not clear 1% over pf1 there, the hysteresis keeps
  pf1 and I land ~230 — either way no repeat of pfa's +7.7% B=32 surprise.
* B=1: unchanged 118–121, but the *median* pick should now match the min
  (r3 shipped a 123.2 pf4 plan in one run of three).

### Next

1. **Transposed-mid phase 2** (not built this round, costed on paper): store
   phase 1's output as mid[ky][x][z-vector] (chunk stride 36 complex for x,
   plane stride 36² for ky) instead of mid[x][ky][z]. Phase 1's y-line
   stores become 36 scattered 576-B streams (stores scatter for free — the
   store buffer absorbs them; pencilfused measured exactly this asymmetry),
   and phase 2's loads become one **contiguous 20.25 KB block per y** instead
   of 36 concurrent 20736-B-stride streams. That removes the last
   demand-miss stream structure in the transform. Cost: the cached B=1 path
   can no longer run phase 2 in place in `out` (mid layout ≠ out layout), so
   it would need the volume scratch at every batch size — 2.2 MB resident at
   B=1, the thing round 1 rejected. Ship as *additional* NT-path candidates
   first (`xnt` reads sequential, NT stores scatter at 20736 B — still full
   64-B lines at PW=4).
2. **If the node again shows B=256 ≥ 40 µs above L36_pfa** with the same
   pfin mechanism, the difference is no longer prefetch and a phase-split
   timing on the node (their SKIPA/SKIPB trick) is the only way forward —
   ask the monitor.
3. **B=1 software pipelining of two line transforms** (round-1 Next #2,
   still undone, ~10% front-end headroom documented): now the only lever
   left at B=1, where three entries sit within 1.2%.

---

## Round panel_r5

### Where round panel_r4 landed (node, p55n3)

Held **B=1 (119.021 µs, first)** and **B=4 (129.921, first)** — both picks
stable at `v1-cached-pf0` across all three runs, so the r4 tuner fix worked.
pfin recovered the batched cells vs r3 (B=256 264.5 → 228.7, B=32 233.4 →
221.6) but **both still lost to L36_pfa, and B=32 lost by 27%** (174.226 vs
221.602). The decisive datum is in the r4 pick strings: L36_pfa won BOTH
batched cells with **`mode=inplace pf=1` — cached in-place stores, no NT, no
volume scratch, paced input prefetch** — while my picks were `v1-nt-pf1-pfin`.
The r4 VERDICT also settled the clock question: the node runs ~3.89 GHz
sustained (L6_unrolled's in-plan probe), not the 2.3 GHz three rounds of
cycle accounting assumed.

### What changed: the cached path gets pfin, and the store policy becomes a
### tournament decision at streaming batch sizes

My candidate list had a structural hole that the threshold policy made
invisible: `pfin` (the paced phase-1 input prefetch that is worth −23% on
wallaby and −13.5% on the node) existed **only as an NT-path candidate**, and
at B≥32 the working-set threshold excluded every cached candidate from the
tournament. So the exact configuration that won both of L36_pfa's batched
cells on the node — cached in-place phase 2 + paced input prefetch — was
never even timed by my tuner. This round:

1. **New execute body, code 7 = `cached-pf1-pfin`** (all three kernels):
   phase 1 `in -> out` with the 32 KB paced T1 cursor, phase 2 in place in
   `out` with cached stores and the one-line 36-stream prefetch, plus the
   PFNX-style cold-window pre-coverage of `in[b+1]` (3 T1 lines per 36-line
   tile group = 62 KB) now issued from the *cached* phase 2 as well (at PW=2
   it fires on even z-tiles with the pair index, same 62 KB coverage).
   **Borrowed, with attribution: this is L36_pfa's `inplace pf=1`, the
   panel_r4 node winner at B=32 (174.2) and B=256 (218.9), reproduced inside
   my two-sweep structure.** No arithmetic change anywhere — still 248
   FMA-port vector ops + 49 shuffles per 36-point line, 708 real flops/line,
   2 752 704 flops/volume.
2. **Mixed tournament in the streaming regime.** The threshold now decides
   only whether NT candidates (and the volume scratch) are *in play*; it no
   longer excludes cached candidates. Per kernel, simplest first:
   `cached-pf1, cached-pf1-pfin, nt-pf0, nt-pf1, nt-pf1-pfin, nt-pf1-xv`
   = 18 candidates at streaming batch (12 at cached-only sizes, where
   `cached-pf1-pfin` is also added). The r2-era rule "the threshold picks the
   policy, the tuner only ranks within it" was correct **when tuning arenas
   were ≤4 volumes and could not rank store policies honestly (L6_pfa's
   documented mis-pick)**; it has been obsolete since r3 made the arena
   machine-relative (2.5× this machine's L3, 32–128 volumes — it genuinely
   streams on both wallaby and the node), and r4 proved the cost: the
   threshold locked me onto NT on the exact machine where cached wins.
3. **Hysteresis widened 1% → 3%** (borrowed from L36_pfa r4: they measured
   the coin-flip zone at 2.4%, and every genuine win on this board is ≥10%).
   With cached listed before NT, a near-tie now resolves to the store policy
   the node measured as the winner.
4. **Override semantics** (plan-time, execution stays repeatable):
   `FFT36_NT=0` = cached candidates only, `FFT36_NT=1` = NT candidates only
   (the old A/B semantics preserved); unset = threshold decides whether the
   pool is mixed. `FFT36_PFIN`/`FFT36_XV` filter as before, now across both
   store policies.

### What was measured (wallaby, Sapphire Rapids Gold 6448Y, 2 MB L2, 60 MB L3)

All numbers driver min, µs per transform; rel_l2 3.954–3.968e-16 at every
batch tried (1, 4, 8, 32, 256), bit-identical re-runs everywhere.

* **B=1: 51.59 (sd 0.06%, quiet window)**, pick `v1-cached-pf1` from the
  12-candidate pool, stable across plans. MKL same window: 142.4.
* B=4: 72.5/vol, pick `v1-cached-pf1`.
* B=32: 83.8/vol (cached pool on wallaby — 47.8 MB fits its L3).
* **B=256: 101.2/vol (sd 2.1%)**, the 18-candidate mixed pool picks
  `v1-nt-pf1-pfin` — correctly for this machine, see the A/B. MKL same
  window: 163.6/vol.
* Paired A/B at B=256, forced pools, back-to-back, two pairs:
  `cached-pf1-pfin` **154.4 / 165.8** vs `nt-pf1-pfin` **101.9 / 101.5** —
  on wallaby NT wins by ~35%, i.e. wallaby CANNOT reproduce the node's
  cached-wins result (its DRAM absorbs the RFO stream; same lesson as r4's
  in-arena inversion). The point of this round is precisely that the node's
  own tournament now gets to make that call with both policies present.
* The new `v1-cached-pf1-pfin` body was verified against numpy end-to-end
  at B=256 (PASS 3.961e-16) and the PW=2 variant (`v0-cached-pf1-pfin`,
  including the even-tile cold-window indexing) end-to-end on the Haswell
  login node at B=8 (PASS 3.957e-16).
* Setup cost: 3.0–3.4 s at B=256 on wallaby (106-volume arena × 18
  candidates); ~1.5–2 s expected on the node's 37-volume arena. Excluded
  from the score.

### What was tried / observed that did NOT work

1. Nothing was removed; this round is deliberately additive-only (the r3
   VERDICT's "add candidates, do not replace structures", third application).
   The xv candidates remain listed last and are still dominated.
2. **Wallaby cannot arbitrate cached-vs-NT for the node** — the forced A/B
   above (NT −35% on wallaby, while the node's r4 pick strings show cached
   winning there by 20% at B=32) is the cleanest cross-machine store-policy
   inversion the record has. Do not tune store policy on wallaby, ever.

### Predictions for the node (stated so they can be scored)

* **B=32: the big one.** The mixed pool's arena at B=32 *is* the real regime
  (arena = batch = 32 volumes, 45.6 MB against 22 MB L3). If the cached+pfin
  mechanism transfers, `v1-cached-pf1-pfin` is picked and lands **~175–195 µs**
  (L36_pfa's identical-mechanism cell measured 174.2). If the tuner keeps NT,
  something about my two-sweep differs from their structure and the pick
  string will say so.
* **B=256:** pick `v1-cached-pf1-pfin` or `v1-nt-pf1-pfin` within 3% of each
  other; either way **~210–230 µs** (pfa's cached cell: 218.9; my NT r4:
  228.7). Parity with L36_pfa is the target.
* **B=1/B=4: unchanged** (118–121 / 128–131), picks `v1-cached-pf0/pf1` —
  the pool at those sizes only gained one candidate (cached-pf1-pfin), which
  should lose in a 1–4 volume arena and is behind a 3% bar.
* Setup ≤ 2.5 s in every cell.

### Next

1. **Read the node's B=32/B=256 pick strings first.** If cached-pfin won:
   the store-policy question is closed for L=36 and the remaining batched gap
   (if any) vs L36_pfa is structural — diff their phase-2 loop against mine
   (they run in-place in `out` too; candidate differences are their PFNX
   depth, their 36×5-site phase-2 prefetcht0 pattern, and plane-scratch
   handling). If NT-pfin won instead and the cells still trail pfa, ask the
   monitor for one forced `FFT36_NT=0` B=32 run — that isolates policy from
   structure in one measurement.
2. **B=1 software pipelining of two line transforms** (round-1 Next #2,
   still undone): with the node at 3.89 GHz, B=1's 119 µs = ~463k cycles
   against a ~241k-cycle single-512-bit-FMA port floor — the front-end/latency
   headroom is real (~10% was the static estimate) and B=1 is a three-entry
   race within 1.2%.
3. **If both batched cells land on cached**, consider retiring the xv bodies
   and the PW=2 staged-NT machinery next round to halve the candidate count
   and setup time — but only after the node confirms, not before.

---

## Round panel_r6

### Where round panel_r5 landed (node, p55n3)

Lost B=1 for the first time in three rounds (122.755 vs L36_pfa 120.358 and
pencilfused 121.255 — the round's only cell regression, +3.1% on an identical
pick string, VERDICT suspect: code layout as the pool grew 9→12). Tied B=4
(129.742 vs pfa 129.295). Lost both streaming cells again: B=32 177.726 vs
pfa 168.565, B=256 215.882 vs pfa 182.598. The decisive datum is in the pick
strings: my node picks at B=32/B=256 were `v1-cached-pf1-pfin` (3/3 stable) —
which is exactly L36_pfa's *losing* `inplace pf=1` configuration — while pfa
won both cells with `inplace pf=2`: the same structure plus a **paced
`prefetchw` on the phase-1 cold-`out` store stream**. Their in-arena
decomposition put a number on what I was missing: inplace-pf2 90.5 vs
inplace-pf1 156.6 µs/vol at B=256 (−42%). The r5 VERDICT's store-policy
verdict is now categorical: hide the RFO (`prefetchw`), don't avoid it (NT
stores lost on the node for the fourth consecutive round, at every geometry,
in every entry's own tournament).

### What changed

**1. `pfw` — paced write-intent prefetch on phase 1's store stream (new
exec code 4 = `cached-pf1-pfin-pfw`, all three kernels). Borrowed, with
attribution: L36_pfa round panel_r5's PFWMID (their `pf=2`, node-selected at
both B=32 and B=256), which in turn adopted it from L6_unrolled r3's
`fused_pfw`.** A cursor of `__builtin_prefetch(p, 1, 3)` (emits `prefetchw`
on PRFCHW parts — Cascade Lake and Sapphire Rapids both; degrades to
`prefetcht0` on Haswell) runs one plane (PFW_D = 2592 doubles = 20.25 KB)
ahead of the plane phase 1 is storing, advancing PFIN_L lines per codelet
call — 18 calls/plane at PW=4, 36 at PW=2, either way exactly one plane of
prefetchw per plane stored, the same pacing arithmetic as pfin. The cursor is
per-volume (`vout + PFW_D`), clamped at the end of `out`. At streaming batch
every one of the 11 664 output lines per volume otherwise costs an
un-overlapped demand RFO from DRAM; prefetchw acquires the line exclusive
ahead of the store while keeping the normal-store shape the node prefers.
Offered **only in the streaming pool**: pfa and L6_unrolled both measured
prefetchw at +11–17% on cache-resident volumes, so it is not re-litigated at
B=1/B=4 in every plan.

**2. The NT and xv machinery is retired** (exec codes 3/4/5/6 of r2–r5, the
one-volume scratch, the PW=2 staged-flush NT path, `FFT36_NT`/`FFT36_XV`).
This executes my own r5 "Next #3", whose stated condition was met: the node
picked cached at both streaming cells *with the full NT candidate set in the
pool* (ntpolicy=1, 18 cand), and NT has now lost every tournament on the node
for four rounds panel-wide. xv has been dominated by pfin since r4 on both
machines. Effects: candidate pool 18 → 9 at streaming sizes and 12 → 15 at
cached sizes (sp2 added, see below), setup time halved (3.0–3.4 s → 1.44 s at
B=256 on wallaby), and the compiled footprint drops by roughly half its
bodies — relevant because the r5 VERDICT's suspect for my +3.1% B=1
regression was code layout.

**3. sp2 — the five-rounds-untried B=1 lever, finally measured.** New exec
code 5 = `cached-pf0-sp2`: two *independent* 36-point line transforms
interleaved at source level (DFT4-by-DFT4 in stage 1, DFT9-by-DFT9 in stage
2) in both the y-pass and the x-pass, via generic `ST1G/ST2G` macros
parameterised over the temp array and load/store macros (the single-line
codelets expand through the same macros, so their codegen is unchanged).
Output is bit-identical to sequential calls — same operations, same
per-transform association. **Result: it loses.** Forced `FFT36_SP2=1` on
wallaby at B=1: 55.5–55.6 µs vs the full pool's 51.58 (**+7.7%**); on the
Haswell login node (V0, 16 ymm) forced sp2 at B=8: 209.6 vs ~200 µs/vol
(+5%). The mechanism doubles live vector state (72 T-array values against 32
zmm) and the extra spill traffic costs more than the latency overlap buys —
on machines whose OOO window already overlaps adjacent loop iterations.
Recorded as measured-and-rejected at this granularity; kept in the pool as a
gated candidate (the node's narrower Cascade Lake OOO window gets to vote,
and the numerical gate admits it every plan), listed last so it must beat the
incumbent by 3%.

**4. Tuner hardening after an observed noisy-window mis-pick.** With the r5
ordering (v0 kernels listed first), one wallaby plan taken during a load
spike picked `v2-cached-pf4` and executed at 75.6 µs (sd 15.9%) — a 40%
mis-pick. Candidates are now ordered **V1 first** (V1 has won every node cell
in every round since r1), then V2, then V0, all through the same
probe-and-admit path, so a 256-bit kernel now needs a >3% fake win over V1's
best-of-rounds minimum to be installed rather than a coin flip. Four
subsequent plans in windows with sd up to 33% all picked v1 (pf0/pf1 flipped
between those near-identical twins, which is harmless). Sampling at tiny
arenas also raised: rounds 6 → 10 at nt < 4.

Overrides now: `FFT36_PFIN=0|1`, `FFT36_PFW=0|1`, `FFT36_SP2=0|1` (read once
at plan time; execution stays repeatable). `FFT36_NT`/`FFT36_XV` are gone.

### Operation count

Unchanged since round 1: 248 FMA-port vector ops + 49 shuffles per 36-point
line over PW lanes, 708 real flops/line, 2 752 704 flops/volume. pfw adds
11 664 `prefetchw` per volume (one per output line), phase 1 only, streaming
candidates only. sp2 adds zero operations (same ops, reordered).

### What was measured (wallaby, Sapphire Rapids Gold 6448Y, 2 MB L2, 60 MB L3)

All PASS vs numpy, bit-identical re-runs. rel_l2 = 3.954e-16 (B=1),
3.957e-16 (B=8, wombat V0), 3.961e-16 (B=32 wombat, B=256 wallaby).

* **The round's result — paired A/B at B=256** (382 MB, streams; same
  binary, forced pools, back-to-back, two pairs):
  `pf1-pfin` **152.2 / 153.1** vs `pf1-pfin-pfw` **102.8 / 104.5** µs/vol —
  **pfw is −33%**, against pfa's −42% in-arena for the identical mechanism.
  Auto tournament picks `v1-cached-pf1-pfin-pfw` and lands **101.4–103.7
  µs/vol** end-to-end (sd 0.07–0.13% in quiet windows) — the same speed the
  retired NT path measured in r5 (101.2), now in the cached-store shape the
  node actually selects. MKL same windows: 168.5–170.2 µs/vol.
* B=1: **51.58 µs** quiet-window (pick `v1-cached-pf1`), matching r5's 51.59
  — the restructure (NT removal, probe-path unification, V1-first) cost
  nothing on the fast path.
* B=4: 78.2–79.7 µs/vol, pick `v1-cached-pf0`. B=32 (cached regime on
  wallaby's 60 MB L3): 82.6, pick `v1-cached-pf1-pfin`.
* V0 pfw path end-to-end on the Haswell login node, where B=32 *is*
  streaming (45.6 MB > 1.25 × 30 MB): pick `v0-cached-pf1-pfin-pfw`, PASS,
  repeatable — exercises the PW=2 pacing (9 lines/call, 36 calls/plane) and
  the builtin's graceful degradation (no PRFCHW on Haswell).
* Build hygiene (the r4 latent-break lesson): clean under the node's
  `-O3 -march=cascadelake -mtune=cascadelake` (objdump shows 34 real
  `prefetchw` sites) and under bare `-O2`.
* Setup: 1.44–1.60 s at B=256 (halved from r5), 0.18 s at B=4, ≤0.35 s at B=1.

### What was tried and did NOT work — with the number that killed it

1. **sp2 (source-interleaved transform pairs) at B=1: +7.7% on wallaby
   (55.6 vs 51.6), +5% on Haswell V0.** The five-round "Next" item is now a
   measured rejection at this granularity. Anyone revisiting B=1 compute
   should try a *stage-level* pipeline instead (one T array live while the
   next tile's stage 1 fills a second — half the extra live state), not
   re-run the full-pair form.
2. **V0-first candidate ordering under load: a real 40% mis-pick**
   (`v2-cached-pf4`, 75.6 µs, sd 15.9%). Fixed structurally (V1 first + 3%
   bar); four noisy-window plans since all picked v1.

### Predictions for the node (stated so they can be scored)

* **B=32** (streams there: 45.6 MB vs 27.5 MB threshold): pick
  `v1-cached-pf1-pfin-pfw`. pfa's identical-mechanism cell measured 168.565;
  wallaby says my two-sweep is at parity with their structure once pfw is in.
  **Predict 163–175 µs** (from 177.726).
* **B=256**: pick `*-cached-pf1-pfin-pfw`; **predict 180–200 µs** (from
  215.882). Note pfa's winning node pick here was **pw=2** inplace-pf2 — if
  my v2/v0 pfw candidates beat v1 on the node the tournament can follow, and
  the pick string will say so.
* **B=1**: pool semantics unchanged (sp2 should be rejected in-arena);
  the halved code footprint is the only layout change — if the r5 VERDICT's
  layout theory is right, B=1 returns toward 119–121; if it stays ~123 the
  theory needs a different suspect. **Predict 119–123**, pick
  `v1-cached-pf0/pf1`.
* **B=4**: 128–131, pick `v1-cached-pf0`.
* Setup ≤ 1 s in every cell (37-volume arena, 9 candidates at streaming).

### Next

1. **Read the node's B=256 pick width.** If v1-pfw lands >5% behind pfa's
   pw=2-pfw cell, the gap is the 512-bit kernel's L1/L2 behaviour under a
   live prefetchw stream — try lowering the pfw batch per call (9 lines
   twice as often) before touching structure.
2. **PFW_D sweep** (1296 / 2592 / 5184 doubles): shipped 2592 = pfa's
   default, which their own record calls pacing arithmetic, not measurement.
   One forced A/B per value at B=256 on the node settles it (pfa's r5 Next
   #1, still open — worth coordinating via the monitor).
3. **B=1 stage-level software pipeline** (see failure #1): the only
   remaining compute lever; the full-pair form is dead, the halved-live-state
   form is not.
4. If B=1 still reads ~123 with the smaller binary, ask the monitor for one
   `perf stat -e icache_64b.iftag_miss,frontend_retired.itlb_miss` B=1 run —
   settle the layout theory with a counter instead of a third guess.

---

## Round panel_r7

### Where round panel_r6 landed

Nowhere: panel_r6 was **abandoned between development and timing** (a stale
runner was retired; see `results/panel_r6_abandoned_no_timing/WHY.md`), so the
r6 code — pfw, sp2, the NT/xv retirement — was never node-scored and the
standings are still panel_r5 (B=1 122.755 3rd-of-3 by 2%, B=4 tied, B=32
177.726 and B=256 215.882 both lost to L36_pfa's pfw, which my r6 then
adopted). The r6 predictions stand unscored and carry over. What is new this
round is the two rivals' r6 records:

* **L36_pfa r6** made the board's best B=1 diagnosis: the node's B=1 gap
  (~120 µs vs an ~83 µs port floor at the 2.9 GHz licence clock) is **L2
  thrash** — in+out = 1.5 MB against 1 MB L2, so the phase-1 in-read evicts
  `out` mid-execute, every in-place store RFOs from L3 and phase 2 re-reads
  from L3. All three L36 entries share the two-sweep structure, hence all
  cluster at 120–123. Their fix (their pf=4): a **constant-lead NTA prefetch
  of the in-read** so the single-use stream fills L1, bypasses L2, and `out`
  stays L2-modified across executes. Never node-tested (round abandoned).
  They also measured pfw at **B=4 −8%** in a quiet window (70.7 vs 76.9) —
  r5's B=4 rejection of pfw was a noisy-window artifact.
* **L36_pencilfused r6** killed the two-group software pipeline for the whole
  board (their PFA36X2: +1–3% pw4, −17% pw2), independently confirming my r6
  sp2 rejection (+7.7%).

### What changed (all additive, no arithmetic change anywhere)

1. **nta — exec code 6, `cached-pf1-nta`, all three kernels. Borrowed, with
   attribution: L36_pfa round panel_r6's pf=4 design, including their
   512-double (4 KB) lead, the best of {128,256,512} in their wallaby sweep.**
   A stateless constant-lead cursor: each yb-iteration of phase 1 (the only
   consumer of `in`) issues 9·PW `prefetchnta` for the 9·PW in-lines it will
   read one lead ahead, so the lead never swings (their r6 lesson: a swinging
   cursor is fine for L2-bound T1, fatal for L1-resident NTA lines). No pfin
   cursor, no PFNX cold-window (the target is cache-resident `in` at B=1; the
   next-volume window is not the point). Lead is `-DFFT36_NTA_DIST`
   overridable. Offered in the **cached pool only** — pfa measured NTA on
   DRAM-rate streams at +14% over *no prefetch at all* (their wallaby B=32),
   so at streaming sizes it appears only under an explicit `FFT36_NTA=1`
   force (never a silent fallback). New override `FFT36_NTA=0|1` mirrors the
   others. Admission gate unchanged (1e-13 vs exec_0_0; prefetches are
   side-effect-free).
2. **pfw joins the cached pool at batch ≥ 2** (same body, exec code 4 — no
   new code, just candidate placement). Rationale: pfa's B=4 quiet-window
   finding above; at B≥2 the `out` volumes cycle through L2/L3 so phase 1's
   store stream RFOs are exposed even though the batch does not stream. At
   B=1 it stays excluded (steady-state-resident `out`; prefetchw measured
   +11–17% tax by pfa and L6_unrolled).
3. **PFW_D is now `-DFFT36_PFW_DIST`** (default unchanged, 2592 doubles = one
   plane) so the monitor's 1296/2592/5184 node sweep (open since pfa r5)
   needs no source edit.

Candidate pools: cached 15 → 21 (nta ×3, pfw ×3 at B≥2), streaming 9
(unchanged; +3 only under forced FFT36_NTA=1). Hysteresis order per kernel:
plain pf0/pf1/pf4, pfin, **nta**, pfw (B≥2), sp2 — V1 first as since r6.

### Operation count

Unchanged since round 1: 248 FMA-port vector ops + 49 shuffles per 36-point
line over PW lanes, 708 real flops/line, 2 752 704 flops/volume. nta adds
11 664 `prefetchnta` per volume (one per input line, phase 1 only) when
selected; zero FP.

### What was measured (wallaby, Sapphire Rapids Gold 6448Y, 2 MB L2, 60 MB L3)

All PASS vs numpy, bit-identical re-runs everywhere. rel_l2 = 3.954e-16
(B=1), 3.963e-16 (B=4 forced-nta), 3.961e-16 (B=32, B=256).

* **B=4, the round's measurable win — forced A/B pairs, back-to-back**
  (µs/volume): plain (all mechanisms off) **84.5 / 83.9** vs pfw
  **71.5 / 73.6** — **pfw is −13 to −15% at B=4**, larger than pfa's −8%.
  Auto tournament picks `v1-cached-pf1-pfin-pfw` and lands **73.8 µs/vol**
  (r6 auto: 78.2–79.7, pick pf0).
* **B=32** (cached regime on wallaby's 60 MB L3): auto picks
  `v1-cached-pf1-pfin-pfw`, **71.4 µs/vol** (sd 0.15%) vs r6's 82.6 — the
  same placement win. (On the node B=32 streams, pool unchanged from r6.)
* **B=1: 54.2–55.2 µs** across quiet windows (sd 0.04–0.17%), picks
  `v1-cached-pf0/pf1` stable over three plans — the fast path is untouched
  and nta is never mis-picked into it here.
* **B=256: 105.5 µs/vol** auto (sd 0.98%), pick `v1-cached-pf1-pfin-pfw` —
  parity with r6's 101.4–103.7 windows; this round did not target it.
* **nta forced at B=1 is bimodal on wallaby**: one quiet-window pair 88.7 vs
  plain's 54.8 (**+62%**), another 53.6 vs 54.3 (−1.3%), a third window 51.0
  (sd 7%). Exactly pfa's expectation: wallaby's 2 MB L2 holds in+out at B=1,
  so there is no prize, only the L1 quick-evict tax showing raw — **wallaby
  cannot price nta's node bet either way** (the r5 store-policy lesson, third
  appearance). What wallaby *can* certify: the full-pool tuner rejects it
  reliably (3/3 plans), and the path is numerically correct at B=1 and B=4.
* **PFW_D sweep at B=256, forced pfw, three binaries interleaved round-robin**
  (µs/vol, two rounds): 1296 → 106.1/100.7, 2592 → 105.0/104.3,
  5184 → 105.6/109.2. **Flat within wallaby noise; default stays 2592**
  (pfa's node-selected constant). The node sweep is now one `-D` per run.
* Build hygiene: clean under `-O3 -march=cascadelake -mtune=cascadelake` and
  bare `-O2`; disassembly carries the prefetchnta sites (gcc's ICF folds
  exec_2_6 into exec_0_6 — same PW=2 code, harmless).

### What was tried and did NOT work — with the number that killed it

1. **nta on wallaby**: +62% in one quiet window (88.7 vs 54.8) — see above.
   Not a rejection of the mechanism (wallaby structurally lacks the prize);
   recorded so nobody reads a wallaby nta number as the node's answer.
2. Nothing was removed; sp2 stays as a gated candidate (now doubly
   condemned locally — mine r6, pencilfused r6 — but the node has never
   voted on it).

### Predictions for the node (stated so they can be scored)

* **B=1, the round's bet.** If pfa's L2-thrash diagnosis is right and the
  NTA tax stays under the L3-round-trip savings on Cascade Lake, the nv=1
  steady-state arena picks `v1-cached-pf1-nta` and B=1 lands **95–115 µs**.
  If the wallaby pathology transfers, the pick stays `v1-cached-pf0/pf1` at
  **119–123** — a clean null that closes NTA for L=36 with one pick string.
  Same bet as L36_pfa's pf=4; whoever's tuner prices it better wins the cell.
* **B=4**: pick `v1-cached-pf1-pfin-pfw`; if wallaby's −13% transfers even
  half, **118–128 µs** (from 129.9). If the node's B=4 working set behaves
  like its B=1 (out re-resident across the 4-volume cycle), pfw is rejected
  in-arena and B=4 stays 128–131.
* **B=32**: streaming pool, unchanged r6 code → pick
  `v1-cached-pf1-pfin-pfw`, **163–175 µs** (r6 prediction, still unscored).
* **B=256**: same pick, **180–200 µs** (r6 prediction, still unscored).
* Setup ≤ 2.5 s at B=256 (21-candidate cached pools only exist at B<32).

### Next

1. **Read the node's B=1 pick string first** — it settles the NTA question
   for the whole L=36 board either way. If nta won: sweep FFT36_NTA_DIST
   (256/512/1024) on the node, and consider an nta+pfw composite for B=4.
2. If nta was rejected and B=1 still reads ~120: the memory theory survives
   only via a counter — ask the monitor for one
   `perf stat -e l2_rqsts.all_demand_miss,LLC-loads` B=1 run to size the L3
   term; if it is small, the residual is front-end and the lever is codelet
   scheduling, not caching.
3. **Node PFW_D sweep** (`-DFFT36_PFW_DIST=1296/2592/5184`, forced
   `FFT36_PFW=1`, B=256): wallaby is flat, the node may not be.
4. If B=32/B=256 land >5% behind L36_pfa with identical mechanisms, diff the
   pfw pacing against their pf=2 (mine: PFIN_L lines per codelet call in
   phase 1 only; theirs: 18 lines/iteration through both subloops) — that is
   the only remaining structural difference in the streaming path.
