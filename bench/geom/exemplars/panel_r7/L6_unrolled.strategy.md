# strategies/L6_unrolled.md — L = 6 (6^3 = 216 complex doubles)

Implementation: `impl/L6_unrolled.c`   ·   `fft3d_name()` = `L6_unrolled`

---

## Round 1 (panel round 1, dev machine = Haswell E5-2680 v3, 2.5 GHz, AVX2, 16 ymm)

### Technique

One fully unrolled straight-line **6-point PFA (Good–Thomas) 2×3 codelet**, applied
row–column along all three axes, batch-looped, with the y- and z-axis passes fused in
registers. No twiddle table, no runtime index table, no loops inside the codelet, no
data-dependent branches, no library call.

### Derivation (the whole thing, so the next implementer need not redo it)

6 = 2·3 and gcd(2,3)=1, so Good–Thomas applies and **every twiddle disappears**. With

```
j = (3*j1 + 2*j2) mod 6      j1 in {0,1},  j2 in {0,1,2}
k = (3*k1 + 4*k2) mod 6
```

both maps are bijections on {0..5}, and

```
j*k = 9*j1*k1 + 12*j1*k2 + 6*j2*k1 + 8*j2*k2
    ≡ 3*j1*k1 + 2*j2*k2   (mod 6)
=>  w6^(jk) = (-1)^(j1 k1) * w3^(j2 k2),      w3 = exp(-2*pi*i/3)
```

so `DFT6 = DFT2 ⊗ DFT3` exactly, joined by a pure index permutation. Written out:

```
p0 = x0+x3   p1 = x2+x5   p2 = x4+x1        (DFT2 stage, k1 = 0)
q0 = x0-x3   q1 = x2-x5   q2 = x4-x1        (DFT2 stage, k1 = 1)
DFT3(p0,p1,p2) -> X0, X4, X2
DFT3(q0,q1,q2) -> X3, X1, X5
```

and the 3-point module, with s = sqrt(3)/2 the *only* irrational constant in the
whole transform (1/2 is the other constant):

```
a = y1+y2 ;  b = y1-y2
Y0 = y0 + a
m  = y0 - 0.5*a                       (1 FNMADD)
Y1 = m - i*s*b   =  m + K*swap(b)     (1 shuffle + 1 FMA)     K = (+s,-s,+s,-s)
Y2 = m + i*s*b   =  m - K*swap(b)     (1 FNMADD)
```

`swap(b)` is one `vpermilpd $5` on interleaved complex; the sign pattern is folded
into the constant vector K, so **there is no complex-multiply routine anywhere** and
every constant is written literally. Verified against a direct O(n^2) DFT to 3e-14
before a line of SIMD was written (`t6.c` in scratch).

### Operation count

| unit | flops | instructions |
|---|---|---|
| DFT2 (complex) | 4 | 4 |
| DFT3 (complex) | 18 | 12 (6 add/sub + 2 FNMADD + 4 FMA, + 1 swap per SIMD vector) |
| **DFT6 = 3·DFT2 + 2·DFT3** | **48** | **36** |
| 6^3 volume = 3 axes × 36 lines = 108 line-DFTs | **5184** | **3888** |

48 flops / 36 instructions is exactly what `LITERATURE.md` §3.1 calls the provably
optimal Good–Thomas count and exactly what FFTW's `n1_6` attains, so **the arithmetic
is closed; do not spend time here.** (`python/fft3d.py`'s `line_cost()` model, 5 n log2 n
= 38.75 flop/point, is a *pessimistic* yardstick: the real count is 24 flop/point.)

Vectorised (one `__m256d` = 2 complex): 54 codelet instances per volume =
**972 arithmetic vector uops + 108 in-codelet shuffles** per 6^3 volume.

### Layout and SIMD decisions

**Interleaved complex, no repack.** This is the one place where I deviate from the
corpus's Tier-1 recommendation (split-complex, batch-minor) and the reason is a
head-to-head instruction count:

* interleaved, 2 complex/ymm: 18 arith + 2 shuffles per vector codelet = **9 arith +
  1 shuffle per 6-point line**
* split, 4 doubles/ymm: 36 arith + 0 shuffles per vector codelet = **9 arith + 0
  shuffles per 6-point line**

Identical arithmetic. Split only saves the 108 in-codelet shuffles per volume, and it
*costs* a deinterleave/interleave (>= 216 shuffles + a full extra read/write pass), and
at L=6 the split lane dimension is 6 — not a multiple of 4 — so the y-axis pass runs at
75 % lane efficiency. Interleaved wins here; the corpus's argument is written for a
*batch-minor* layout, and batch-minor at L=6 needs a cross-volume transpose (a gather of
one double from each of 4 volumes 3456 B apart) that costs ~432 extra uops per volume —
more than everything it saves. See "did not work" below.

Per-axis lane structure inside a volume (element (x,y,z) at complex index 36x+6y+z):

| pass | axis stride | lanes | shuffles |
|---|---|---|---|
| x | 36 complex | 2 adjacent (y,z) positions, 18 groups | 0 |
| y | 6 complex | 2 adjacent z, 3 groups per x-plane | 0 |
| z | 1 complex | 2 whole z-pencils, brought to z-major in registers | 6 in + 6 out per pair |

The z pass is the only awkward one (the axis is the contiguous direction). Two pencils
(y, y+1) are loaded as 6 `__m256d`, turned into 6 z-major vectors with 6
`vperm2f128`, transformed, and turned back with 6 more. The 12 results of the pair then
land as **6 consecutive 32-byte stores covering exactly three whole 64-byte cache
lines** — 96·y and 576·x are both multiples of 64 — which is what makes streaming
stores clean (no partial-line write-combining).

**Every load and every store in the transform is 32-byte aligned by construction**
(volume 3456 B = 54·64, plane 576 B, pencil 96 B), so no unaligned penalties and no
`loadu` anywhere.

**Register budget.** The x pass keeps 6+6 vectors live (fits AVX2 easily). The fused
y+z stage holds a whole 6×6 (y,z) plane = **18 `__m256d` plus ~8 codelet temporaries
≈ 26 live vectors**. That is the measured resolution of `LITERATURE.md` §4.1's open
question at L=6: compiled `-march=cascadelake` GCC 11 gives **148 references to
ymm16–ymm31 and zero spill slots** (checked by grepping `(%rsp)`/`(%rbp)` in the
generated assembly: the single hit is the callee-saved `%rbx` restore). Compiled for
Haswell (16 ymm) the same function spills 17 times. So: *the fused variant is an
AVX-512VL machine's variant*, and 256-bit code on an AVX-512 machine gets the 32-register
file for free without ever issuing a 512-bit instruction. §04's "12 of 16 ymm" data-only
figure and §01's "17 including temporaries" are both right; the answer for a *fused two
axes* kernel is 26.

**Why 256-bit and not 512-bit — the AVX-512 measurement the corpus asks for.** I
deliberately shipped no zmm path, on this arithmetic:

* Xeon Gold 5218 is a Gold-52xx part, i.e. **one** AVX-512 FMA unit. On such SKUs all
  512-bit FP arithmetic issues on the fused port 0+1 at **1/cycle**, while 256-bit FP
  arithmetic issues on ports 0 *and* 1 at **2/cycle**. Both deliver 8 doubles/cycle:
  512-bit buys exactly **zero** arithmetic throughput here.
* Our bottleneck *is* that arithmetic: 972 vector arith uops / 2 per cycle = **486
  cycles per volume**, against 216 loads (108 cy), 216 stores (216 cy) and 324 port-5
  shuffles (324 cy) — all comfortably underneath.
* 512-bit code additionally moves the core into frequency licence 2. Gold 5218's
  1–2-core turbo is ~3.9 GHz non-AVX / ~3.6 AVX2 / ~3.2 AVX-512, so zmm would cost
  ~10–15 % clock for no instruction-level gain.
* Also checked: a zmm z-pass needs a 4-line × 6-z transpose of 128-bit granules,
  ~3 `vpermt2pd` per output vector (36 shuffles per 4 lines = 9 port-5 cycles/line)
  against the ymm pairing's 7 cycles/line. Wider is *worse* on the transpose too.

Net: **on a 1-FMA-unit Cascade Lake, AVX-512 has nothing to offer this kernel.** The
one thing that would overturn this is a 2-FMA SKU (Gold 61xx/62xx, Platinum), where
512-bit would be a genuine 2× on the arithmetic floor. First AVX-512 datum in the
corpus; see "Next".

### Structure, and the plan-time race

`fft3d_execute` runs, per volume: x pass (`in` -> `t1`, 3456 B of L1 scratch), then the
fused y+z stage (`t1` -> `out`, plane-at-a-time in registers). Everything for one volume
is L1-resident (3.4 KB volume, 3.4 KB scratch), so all three axes are done on one
load-once/store-once trip exactly as §3.2 item 3 asks.

Four kernels are compiled: `{3pass, fused} × {normal stores, NT stores}`.
`fft3d_create()` **validates all four against the scalar reference on random data
(disqualifying any that disagrees by > 1e-11 relative) and then races the survivors at
the real batch size**, min of 7 trials, ordered safest-first with a 1.5 % margin needed
to take over. This is what makes an untestable code path safe, and it is what
automatically resolves the NT-store question, which is *batch-size dependent* and
*machine dependent* (L2 size decides it). The winner is reported in
`fft3d_description()` as `variant=...`, so the leaderboard line records which kernel
actually ran on the node.

**4K-aliasing defence.** A store to S followed by a load from L with (S−L) ≡ 0 mod 4096
false-aliases and replays the load. The scratch is carved out of a 4 KiB-oversized arena
and positioned once (at first execute, when `in`/`out` are known) to maximise the cyclic
distance from 0 mod 4096 of every store->load delta the kernels can produce. Measured
motivation: sweeping the scratch offset over 4 KiB at B=1 gave 0.254–0.314 µs — the bad
end being the configuration where `out − t2 ≡ 64 (mod 4096)`, i.e. **+22 % for a purely
accidental malloc address**. This changes addresses only, never arithmetic or output.

### What was measured

Dev machine: Haswell E5-2680 v3, 2.5 GHz, AVX2 only, 16 ymm, **shared login node with
11 other agents running** — so these are best-of-4-process minima and the noise floor is
real (one in five runs lands 2× high from SMT contention; those are discarded, and the
scored node is `--exclusive`). Flags as in the Makefile:
`-O3 -march=native -mtune=native -std=gnu11 -fno-math-errno -funroll-loops`.

| B | per transform | GF/s (nominal 5NlogN) | raced variant | rel L2 vs numpy |
|---|---|---|---|---|
| 1 | **0.261 µs** | 32.2 | fused / 3pass | 2.31e-16 |
| 8 | 0.269 µs | 31.1 | fused | 2.39e-16 |
| 64 | 0.310 µs | 27.0 | fused_nt | 2.40e-16 |
| 512 | 0.338 µs | 24.8 | fused / fused_nt | 2.43e-16 |
| 4096 | 0.344 µs | 24.4 | fused_nt | 2.43e-16 |
| 32768 | 0.631 µs | 13.3 | fused_nt | 2.42e-16 |

Reference points: MKL 2022 on the *scored* node is 0.370 µs at B=1. The portable
non-AVX2 fallback in the same file runs 0.80 µs/transform, i.e. the SIMD path is 3.1×
the scalar codelet.

**The B=1 number is at the Haswell hardware floor and I can prove it.** On Haswell FP
add/sub issue on port 1 *only*; our 54 codelet instances per volume contain 12 add/sub
each = 648 port-1 uops, so the floor is 648 cycles. Measured 0.261 µs × 2.5 GHz =
**653 cycles**. There is nothing left to win locally, which is also why local numbers
cannot guide further tuning. On Skylake/Cascade Lake, FP add moved to ports 0 *and* 1,
so the same code's floor drops to **486 cycles** — expect ≈0.21 µs at 2.3 GHz and less
if turbo is live, i.e. ~1.7× MKL. That port-scheme difference, not the clock, is the
reason the node should beat the dev machine.

Large-batch behaviour is bandwidth, not arithmetic: at B=32768 (226 MiB) the traffic is
the irreducible 3456 B read + 3456 B written per volume and 0.631 µs/volume = 10.9 GB/s
on the contended dev node. NT stores are worth **1.55×** there (0.344 vs 0.517 µs at
B=4096) purely by removing the write-allocate read; the write side is already
full-cache-line aligned, which is what makes them legal.

### What was tried and did NOT work (with the number that killed it)

1. **Software prefetch of the next volume's input** (`_mm_prefetch` T0, 9 lines ahead
   of the x pass). B=1 0.382 vs 0.334 µs, B=4096 0.353 vs 0.346 µs — *worse or neutral
   everywhere*. The L2 streamer already has a perfectly sequential 3456-B-per-volume
   stream; the extra uops cost more than they buy. **Dropped.**
2. **Full `#pragma GCC unroll` of all three passes into one ~1900-instruction
   straight-line body.** B=1 0.2640 vs 0.2628 µs, B=4096 0.584 vs 0.514 µs — neutral at
   B=1, **14 % worse** at large batch. The loop bodies as written (32–44 uops) sit inside
   the uop cache; the fully unrolled body does not. "Fully unrolled" should mean the
   *codelet*, not the 36-line loop. **Dropped.**
3. **Two volumes in flight with two scratch sets**, to break the write-after-read
   dependence on the shared `t1`/`t2` between consecutive volumes. B=1 0.2643,
   B=512 0.3633 vs 0.3751, B=4096 0.5363 vs 0.5363 µs — inside noise. The out-of-order
   engine already covers the pass boundary (each pass offers 18 independent groups), so
   there was no serialisation to remove. **Dropped.**
4. **Split-complex batch-minor layout (the corpus's Tier-1 recommendation) at L=6.**
   Not shipped, killed on an instruction count before coding: interleaved and split have
   *identical* arithmetic per line here (9 vector arith ops), so split's entire prize is
   108 shuffles per volume — while the AoS -> batch-minor repack is a 4×4 double
   transpose per 2 points, 108 × (4 loads + 8 shuffles + 4 stores) = **1728 uops per 4
   volumes = 432/volume, twice** (in and out) against a 1728-uop kernel. Net ≈ +50 %.
   Batch-minor is right when the transform has non-trivial twiddles or an awkward lane
   count; PFA-6 has neither.
5. **512-bit AVX-512** — not shipped; the port/licence arithmetic above (equal doubles
   per cycle on a 1-FMA SKU, worse shuffles, lower clock) says it cannot win. Recorded
   here so nobody re-derives it.
6. **3D PFA `DFT_{6^3} ≅ DFT_{2^3} ⊗ DFT_{3^3}`** (`LITERATURE.md` §3.2 item 5, flagged
   there as the most promising structural idea for L=6). Counted, not built: the 2^3 part
   is 27 blocks × 12 DFT2 = 648 instructions, the 3^3 part is 8 blocks × 27 DFT3 = 1296
   arith + 216 shuffles — **1944 arith + 216 shuffles, byte-for-byte identical to
   row-column PFA-6.** Of course it is: PFA does not change the count, it only removes
   twiddles, and row-column PFA-6 has already removed them. It would only *cost* index
   irregularity. **This closes §03 §9.4 / §05 §10.3 negatively for L=6.**
7. **Compiler-flag search** — pointless here: `sweep.sh` builds with fixed
   `-O3 -march=native -mtune=native -fno-math-errno -funroll-loops`. Checked that `-O2`
   and `-O3` are within noise of each other and that neither spills; nothing to tune.

### Next (in the order I would do it)

1. **Read the node's raced variant off the leaderboard `description` string.** If the
   node picks `3pass` over `fused` at B=1, the register analysis is wrong somewhere and
   the assembly should be re-grepped on the node.
2. **Confirm the SKU's AVX-512 FMA-unit count on the node** (`lscpu`, or time a
   512-bit FMA chain). If it is a 2-FMA part, the arithmetic floor halves to 243 cycles
   and a zmm kernel is worth writing — but note the z-pass transpose gets *worse* with
   zmm, so the right shape is "zmm for the x and y passes, ymm pairing for the z pass",
   a mixed-width kernel. This is the only remaining ≥1.3× on the table for B=1.
3. **Huge pages at B >= 4096.** Not tried. 3456 B/volume means a volume straddles
   4 KiB pages; `madvise(MADV_HUGEPAGE)` on `in`/`out` at first execute is legal from
   inside the plan and costs nothing to test. Expect single-digit percent (we are already
   at minimal traffic), which is why it is below (2).
4. **Widen the plan-time race to include the scratch offset.** The 4K-aliasing sweep
   showed a 12 % spread that the current analytic placement rule does not fully capture
   (max-min-cyclic-distance picked r=1792 -> 0.273 µs where r=512 measured 0.255 µs).
   Racing 8 candidate offsets against the *real* `in`/`out` on the first execute would
   capture it; the output is identical either way, so it is safe — I judged the added
   machinery not worth 5 % this round.

---

## Round panel_r2 (dev machine = wallaby, Sapphire Rapids Gold 6448Y, full AVX-512)

### Where round 1 left me

panel_r1 node numbers: **B=1 0.219 µs (tied with L6_pfa for first)**, but at large batch I
was losing badly — B=4096 **0.514 µs vs L6_pfa's 0.392** (31% behind), B=32768 **0.760 vs
0.631** (20% behind, also behind both MKLs). The arithmetic is closed (48 flops/36 instrs
per line is the Good–Thomas optimum), so the entire round went into the batched regime,
which is memory behaviour, not arithmetic.

### What I changed (two ideas, both borrowed from L6_pfa — attribution up front)

1. **Software prefetch of the next volume's input, hooked into the x pass**
   (borrowed from **L6_pfa**, whose `v8 = fused+nt+prefetch` won every large-batch case in
   panel_r1). Each of the 18 x-pass groups issues 3 `prefetcht0`s into the *next* volume =
   all 54 lines of its 3456 B, one volume (~800 cycles) of lead time. My own round-1 record
   says prefetch "did not work" — that experiment was 9 lines ahead within the *current*
   volume on a contended Haswell; L6_pfa's version (whole next volume, all 54 lines) is the
   right shape and it is worth **1.4× at B=4096 on wallaby** (0.27 µs/vol without, 0.19
   with). My round-1 dead-end entry stands corrected on the number, not the idea:
   prefetch *distance and coverage* were what was wrong.
2. **Round-robin plan-time tournament** (borrowed from **L6_pfa**'s "what did not work"
   item 4: their sequential per-candidate race mis-picked by 21% under drifting background
   load). Every round times each surviving candidate once; each keeps its own minimum over
   7 rounds. The safest-first ordering with the 1.5% takeover margin is retained.

Two refinements of my own on top:

3. **The candidate grid is now {3pass, fused} × {normal, NT stores} × {no pf, pf dist 1,
   pf dist 2} (11 kernels shipped)**, because the winner genuinely changes with batch size
   — measured picks on wallaby: B=1 `3pass`, B=64 `fused`, B=4096 `fused_pf`, B=32768
   `3pass_nt_pf`. Note the inversion at DRAM sizes: **3pass+NT+pf beats fused+NT+pf by
   1.6×** (0.196 vs 0.317 µs/vol at the 113 MiB race size). Best reading: the separate z
   pass emits its 108 NT stores in tight back-to-back bursts (good write-combining), while
   the fused variant scatters them between codelet work.
4. **The race's batch truncation cap went 4096 → 16384 volumes.** 4096 volumes = 27 MiB is
   L3-*resident* on wallaby (60 MiB) and marginal on the node (22 MiB), so a race at that
   size mis-picks a normal-store kernel for a DRAM-bound real batch. 16384 volumes =
   113 MiB is unambiguously DRAM on both machines. Setup cost grows to ~0.6 s at B=32768,
   still excluded from the score.

Also added: `L6_VERBOSE=1` makes `fft3d_create` print the full per-candidate race table to
stderr (dev tool; silent by default).

### Operation count

Unchanged: PFA 2×3, 48 flops / 36 instructions per line, 972 vector arithmetic uops per
volume. The prefetch adds 54 `prefetcht0` uops per volume (~5% uop overhead) — pure
memory-level-parallelism play; at B=1 the raced no-pf kernels win, so it costs nothing
where it does not pay.

### What was measured (wallaby, quiet; min over driver samples; rel L2 2.3–2.4e-16 everywhere)

| B | before (this file, r1 code) | after | winner picked | note |
|---|---|---|---|---|
| 1 | 0.130 µs | **0.130 µs** | 3pass | unchanged, as intended |
| 64 | — | 0.131 µs/vol | fused | |
| 512 | — | 0.272 µs/vol | (nt=512 race) | |
| 4096 | 0.368 µs/vol | **0.190 µs/vol** | fused_pf | **1.9×** |
| 32768 | 0.341 µs/vol | **0.234–0.263 µs/vol** | 3pass_nt_pf | **1.3–1.5×** |

wallaby B=1 note: the driver bimodally reports 0.130 or 0.254 µs min across invocations
(exactly 2×, sd inside a run 0.02%) — that is a wallaby clock-state artifact, not a code
difference (same variant chosen both ways). Also: wallaby under load (load avg ~9) inflates
B=4096 to 0.35 µs/vol; every number above was reproduced on a quiet machine.

Prediction for the node: the race runs *on the node* at create time with the corrected
truncation, so it will pick per-regime winners there. B=4096 (27 MiB) exceeds the node's
22 MiB L3, so I expect an NT+pf variant to win there rather than wallaby's `fused_pf`,
and the score to land near or below L6_pfa's 0.392 µs. B=1 should stay 0.219 µs (nothing
touched on that path; the no-pf kernels win the race there).

### What was tried and did NOT work (with the number)

1. **`prefetchnta` on the input stream** (theory: read-once data should skip cache
   pollution): 0.53–0.65 µs/vol at every batch size vs 0.19–0.20 for `prefetcht0` —
   catastrophically worse, worse even than no prefetch. NTA on these parts evicts through
   a single L3 way and apparently loses the line before the x pass returns to it. Removed
   from the candidate set entirely. **Do not re-try NTA at L=6.**
2. **Prefetch distance 2 volumes**: within noise of distance 1 everywhere on wallaby
   (0.1955 vs 0.1956 at the 113 MiB race size). Kept in the grid because it is free and
   the node's shorter L2/longer DRAM latency may separate them; the tournament decides.
3. **`madvise(MADV_HUGEPAGE)` on the driver's buffers at first execute** (my round-1
   "Next" item 3): checked and rejected without shipping. Both wallaby and the node run
   THP in `madvise` mode, but the driver's buffers are already *faulted in* before the
   plan ever sees the pointers, so collapse would rely on khugepaged, which scans ~8 MiB
   per 10 s — orders of magnitude too slow for the benchmark window. A one-line dead end;
   recorded so nobody re-derives it. (L8_batchsimd's record reached the same conclusion
   from a different direction.)

### Next

1. **Read the node's chosen variants off the leaderboard description string** — the
   interesting datum is whether B=4096 picks `fused_pf` (L3-marginal) or `3pass_nt_pf`
   (DRAM), and whether pf2 ever separates from pf1 on Cascade Lake.
2. **B=1 is structurally stuck at ~0.219 µs on the node** (tied with L6_pfa two rounds
   running). If the node clock is actually turbo (~3.9 GHz) that is ~850 cycles against a
   486-cycle port floor, i.e. there IS headroom, and the blocker is the pass-boundary
   store→load chain. The one untried idea with real upside: software-pipeline pass 1 and
   the fused y+z stage across volumes or across x-planes (L6_pfa's Next item 2 proposes
   the same thing). Big rewrite, uncertain payoff, right thing to attempt in a round where
   batched is settled.
3. **If the node's B=32768 still trails L6_pfa**: their remaining edge would be the z-pass
   store *ordering* interacting with write-combining buffers; try emitting the 3pass z
   pass plane-by-plane in strictly ascending address order (it already is) and grouping
   the 6 NT stores of each pencil-pair back-to-back (they already are) — then diff the
   generated assembly against theirs rather than guessing.

---

## Round panel_r3 (dev machine = wallaby, Sapphire Rapids Gold 6448Y)

### Where round panel_r2 left me

First in all four L=6 cells on the node (B=1 0.218, B=64 0.214, B=4096 0.384,
B=32768 0.572 µs), but narrowly — L6_pfa is 1–8% behind everywhere. The decisive r2
node facts, from the VERDICT and the description strings: **NT stores lose on the node
at every batch size** (both L=6 entries independently; my node picks were `fused` at
B=1, `fused_pf` at B=64/4096, `3pass_pf` at B=32768 — all normal stores), and B=1 at
base clock is ~501 cycles against the 486-cycle FP-port floor, i.e. ≤3% headroom until
the monitor's `perf stat` says whether the node turbos. So this round went entirely
into the batched regime, and specifically into the one cost nobody had attacked: with
normal stores winning on the node, every output line pays a **write-allocate RFO**
that the store buffer eats at demand time.

### What I changed

1. **Write-intent prefetch of the next volume's output (`prefetchw`) — new idea, the
   round's headline.** Six new kernels `{3pass,fused} × {pfw, pfw2, pft1w}`: the x-pass
   prefetch hook now also touches the *next* volume's 54 output lines with
   `__builtin_prefetch(p, 1, 3)` (emits `prefetchw` on any PRFCHW machine — Cascade
   Lake and Sapphire Rapids both; falls back to `prefetcht0` elsewhere, and compiles
   everywhere). Rationale: NT stores would avoid the RFO entirely but the node rejects
   them; `prefetchw` keeps the normal-store shape the node likes and merely issues the
   RFO one volume (~0.5 µs at DRAM rates) early, off the critical path.
   **Measured on wallaby at the 113 MiB race size (the DRAM regime, B=32768): 1.41×
   in the normal-store rows** — `3pass_pf` 0.4435 → `3pass_pfw` **0.3150 µs/vol**,
   `fused_pf` 0.4650 → `fused_pfw` **0.3114**. First positive `prefetchw` datum in the
   corpus.
2. **`prefetcht1` on the input as a third hint (borrowed from L6_pfa's panel_r3
   record**, which measured T1 beating T0 by 7% in wallaby's NT rows): kernels
   `3pass_pft1`, `fused_pft1`, `3pass_nt_pft1`, `fused_nt_pft1`, plus the T1+W combos.
   On my wallaby runs T1 is only within-noise ahead of T0 in the NT rows (0.1974 vs
   0.1988 at 113 MiB) and behind in normal-store rows at 27 MiB (0.394 vs 0.378) —
   kept because it is free and the node tournament decides.
3. **Grid grown 11 → 22 kernels** (`{3pass,fused} × {none,pfT0·1,pfT0·2,pfT1,
   pfT0+W·1, pfT0+W·2, pfT1+W} × normal` + the 8 NT kernels). Same correctness gate
   (every kernel validated against the scalar reference at plan time, >1e-11 rel
   disqualifies), same round-robin tournament, same safest-first order with the 1.5%
   takeover margin — the no-prefetch kernels still come first, so B=1 cannot be
   noise-stolen by a prefetch variant. Setup grows to ~0.5 s at B=1, ~1.1 s at
   B=32768; unscored.

### Operation count

Arithmetic untouched and still closed: PFA 2×3, 48 flops / 36 instructions per line,
972 vector FP uops per volume, 486-cycle port floor on the node. The W hooks add 54
`prefetchw` uops per volume (on top of pf's 54 `prefetcht0`) in the W variants only —
~10% uop overhead, DRAM-regime-only by construction of the tournament.

### What was measured (wallaby; race-table numbers are the trustworthy statistic —
see the r2 clock-lottery warning; driver minima quoted with the invocation's clock mode)

Race table at the 113 MiB race size (B=32768 case), µs/vol, base-clock invocation:

| shape | pf0 | pfT0·1 | pfT0·2 | pfT1 | pfT0+W | pfT0+W·2 | pfT1+W |
|---|---|---|---|---|---|---|---|
| 3pass | 0.4738 | 0.4435 | 0.4379 | 0.4391 | **0.3150** | 0.3148 | 0.3168 |
| fused | 0.4760 | 0.4650 | 0.4520 | 0.4612 | **0.3114** | 0.3120 | 0.3190 |
| 3pass_nt | 0.3315 | 0.1988 | 0.1977 | **0.1974** | — | — | — |
| fused_nt | 0.3330 | 0.3083 | 0.3132 | 0.2998 | — | — | — |

Driver numbers: B=1 **0.130 µs** (turbo) / 0.254 (base), picks `3pass` — unchanged
from r2. B=64 **0.131 µs/vol**, unchanged. B=4096 **0.194 µs/vol** (turbo invocation;
base-clock race picks `3pass_nt_pf` at 0.326 on wallaby). B=32768 **0.239 µs/vol**,
picks `3pass_nt_pf(t1)` on wallaby (NT still rules wallaby's DRAM). rel L2
2.34–2.43e-16 at B ∈ {1, 64, 4096, 32768}, bit-identical across re-runs, all PASS.
Cross-compile check at `-march=cascadelake`: `prefetchw` and `prefetcht1` both
emitted, all 21 vector kernels spill-free (0 stack refs; `3pass_nt` has 3 = the
callee-save frame; the 168 are the scalar fallback).

**Node prediction, falsifiable via the description string:** the node rejected NT at
113 MiB in r2, so I expect B=32768 to pick `3pass_pfw` or `fused_pfw` and land near
**0.41–0.45 µs** (r2: 0.572) if the node's normal-store rows respond to W like
wallaby's; B=4096 (27 MiB, L3-marginal there) to pick `fused_pfw` or stay `fused_pf`;
B=1/B=64 unchanged (`fused`/`fused_pf`, ~0.218/0.214). If the node picks an NT variant
anywhere, the r2 NT story was wrong and the cell drops further still.

### What was tried and did NOT work (with the number)

1. **`prefetchw` at cache-resident sizes**: at wallaby's 27 MiB race (L3-resident
   there), `fused_pfw` 0.3996 vs `fused_pf` 0.3410 µs/vol — **17% worse**. When the
   RFO hits L3 the early-issue buys nothing and the extra 54 uops/vol + L1 fills cost.
   Exactly why W lives behind the tournament and not a heuristic; expect the node's
   27 MiB case to be the interesting borderline.
2. **`prefetcht1` on the input in normal-store rows**: 0.3942 vs T0's 0.3775 (27 MiB),
   0.4391 vs 0.4379 (113 MiB) — never better than T0 for me; L6_pfa's 7% T1 win
   appears only in their NT rows. Kept only because the node may differ.
3. Consciously not attempted, per the r2 VERDICT's instruction: any B=1 structural
   work (software-pipelining the pass boundary) before the monitor's
   `perf stat -e cycles,ref-cycles` on an L=6 B=1 run settles whether the node turbos.
   At base clock there is ≤3% on the table; the request stands.

### Next

1. **Read the r3 `variant=` strings.** If B=32768 = `*_pfw` at ≤0.45, RFO-pipelining
   is confirmed as the normal-store answer and should be propagated to L=8/L=36
   (L36_pfa's B=256 cell is 6.25 GB/s against my 12+ GB/s — same disease, bigger
   organ). If the node picks NT after all, delete the W story and keep the grid.
2. **Still outstanding: the node clock question** (`perf stat` at B=1). Decides
   whether B=1 has ~3% or ~40% left.
3. If W wins on the node, try **W-distance tuning** (currently 1–2 volumes) and
   **W on the t1 scratch's first touch** (probably nothing — scratch is L1-resident).

---

## Round panel_r4 (dev machine = wallaby, Sapphire Rapids Gold 6448Y)

### Where round panel_r3 left me

First or tied-first in all four L=6 cells on the node (B=1 0.220, B=64 0.214,
B=4096 0.392, B=32768 0.563 µs; 1.25–1.83× MKL), with the VERDICT declaring the
geometry essentially finished: B=1 is 1.04× its own 486-cycle FP-port floor *at base
clock*, and B=32768 is the fastest single-core stream on the board (12.3 GB/s
compulsory). The r3 node picks, stable across all three runs per cell (`fused` /
`fused_pf` / `fused_pfw` / `fused_pfw`), confirmed prefetchw was selected at the DRAM
sizes but bought only 1.6% at B=32768 and *cost* 2% at B=4096 — a race-vs-driver
mis-pick in the L3-marginal regime. The monitor's #1 L=6 ask, two rounds running: the
node clock measurement that decides whether B=1 has ~4% or ~40% headroom.

### What I changed (three things)

1. **A core-clock probe inside `fft3d_create()`, reported via the description
   string — the round's headline, and it answers the panel's standing question
   without needing the monitor's `perf stat`.** A serially dependent 256-bit FMA
   chain (latency 4 cycles on SKX/CLX/ICL/SPR) timed after the tournament has warmed
   the core: freq = iters×4/time, best of 5 trials, ~10 ms, unscored. The result is
   formatted into `fft3d_description()` as `clk=X.XXGHz` next to `variant=`, so the
   r4 leaderboard JSON carries the node's *actual sustained AVX2 clock in every L=6
   cell*. Validation on wallaby: reports **4.10 GHz** (the Gold 6448Y's exact max
   turbo) in normal invocations and **2.10 GHz** in occasional ones — which finally
   *explains* the "clock lottery" my r2 record flagged (bimodal 0.130/0.254 µs at
   B=1, exactly 2×): wallaby really does pin some sessions at half clock. Caveat
   recorded in the source: Haswell FMA latency is 5, so on wombat the probe would
   over-read by 25%; both wallaby and the node are 4-cycle parts.
2. **Split-z-store kernel shapes (`_s`), my own idea, the only uop-mix lever left.**
   The z-pass output permutes are pure data movement: w_k = (A_k | B_k) with pencil A
   wanted at D+0..11 and B at D+12..23. Instead of 6 `vperm2f128` + 6 ymm stores per
   pencil pair, store each half directly: 6 xmm stores of the low halves + 6
   `vextractf128`-to-memory of the highs (2 uops each on SKX: store pipes only, **no
   port-5 shuffle**). The 3pass form also splits the *loads* (`vinsertf128` from
   memory: load + p015 blend, again no p5-only shuffle). Port-5 pressure per volume:
   fused 324 → 216, 3pass 324 → **108**; stores rise 216 → 324 (p4 has headroom
   against the 486-cycle FP floor). Verified in the cascadelake cross-compile: GCC
   emits `vextractf64x2 $1, %ymm, mem` / `vinsertf64x2` from memory, all new kernels
   spill-free. Five new tuner candidates: `3pass_s`, `fused_s`, `fused_s_pf`,
   `3pass_s_pfw`, `fused_s_pfw` (normal stores only — NT needs full-line 32-byte
   bursts, which half-stores give up by construction).
3. **Tournament hardening: takeover margin 1.5% → 2.5%, and 3 dominated candidates
   pruned** (`fused_nt_pf/pf2/pft1` — beaten by their `3pass_nt_*` twins in every r2/r3
   measurement on both machines). The margin raise is aimed at the r3 B=4096 mis-pick:
   the race promoted `fused_pfw` on a <2% race win that the driver then measured as a
   2% loss. Grid is now 24 kernels; same correctness gate, same round-robin race.

### Operation count

FP arithmetic untouched and still closed: PFA 2×3, 48 flops / 36 instructions per
line, 972 vector FP uops per volume, 486-cycle two-port floor on the node. The `_s`
shapes change only the uop *mix*: per volume, −216 port-5 shuffles / +108 store uops
(3pass_s, which also converts 108 ymm loads into 216 half-width load uops) or
−108 p5 / +108 stores (fused_s). Frontend roughly neutral (extract-to-mem is 2 fused
uops against permute+store's 2).

### What was measured (wallaby, quiet, turbo-state invocations; race tables quoted
where they are the trustworthy statistic)

| B | r3 code | r4 code | picked | note |
|---|---|---|---|---|
| 1 | 0.130 µs | **0.114 µs** | **3pass_s** | **−12%**; sd 0.05%; race: 3pass_s 0.1133 vs 3pass 0.1290, fused_s 0.1166 vs fused 0.1286 — every split shape beats every unsplit one |
| 64 | 0.131 µs/vol | 0.131 µs/vol | fused | unchanged; split noisy here (0.129–0.172 across runs) and not chosen |
| 4096 | 0.194–0.196 | 0.196 µs/vol | fused_pf(t1) | unchanged |
| 32768 | 0.239 (driver) | 0.251–0.260 (driver, contended) | 3pass_nt_pf | race 0.1961; NT still owns wallaby's DRAM; **in the normal-store rows the node actually picks, fused_s_pfw 0.3140 now leads fused_pfw 0.3220 (−2.5%)** |

rel L2 vs numpy: 2.34e-16 (B=1), 2.43e-16 (B=64), 2.43e-16 (B=4096), 2.42e-16
(B=32768); bit-identical across re-runs; all PASS. Setup 0.23 s (B=1) to 1.37 s
(B=32768) — *down* from r3's 2.0 s despite the larger grid, because the pruned NT
kernels were the slow ones to race. MKL same-host reference: 0.351 µs at B=1,
0.521 µs/vol at B=32768.

Why split-z wins 12% at B=1 on wallaby when the FP ports are supposedly the floor:
the extract-store starts the moment w_k retires from VD6 (no shuffle between codelet
and store), and the split loads of the next pair issue earlier — it shortens the
z-pass dependency tail, it does not change throughput. That mechanism is not
SPR-specific, so some of it should transfer to CLX; how much depends on the node
clock, which the probe will now report.

### Node prediction, falsifiable via the description strings

* Every L=6 cell's description will carry `clk=X.XXGHz`. **If clk ≈ 2.3, B=1 is
  ≤4% from its floor and 0.211–0.220 is the end state; if clk ≥ 3.0, B=1 has real
  headroom and I expect 3pass_s/fused_s to take part of it — 0.19–0.21 µs.** This
  number should also settle, once and for all, how to read every past and future
  L=6 node time in cycles.
* B=64: `fused` or `fused_pf`, ~0.214, unchanged.
* B=4096: with the wider margin I expect the pick to revert to `fused_pf` and the
  cell to return to ~0.384 (r2's number), recovering the r3 mis-pick.
* B=32768: `fused_s_pfw` or `fused_pfw`, 0.54–0.563. If `*_s_pfw` is picked and the
  cell moves below 0.55, the split-store mechanism transfers; if the cell stays at
  0.563 the geometry really is bandwidth-closed.

### What was tried and did NOT work (with the number)

1. **Split-z at B=64 on wallaby**: unstable (fused_s 0.1289–0.1692 µs/vol across
   invocations vs fused's steady 0.131) and never chosen. L2-resident batches
   apparently expose the doubled store-uop count to whatever else contends for the
   store pipes; the tournament handles it, nothing shipped differently.
2. **NT variants of the split shapes**: not built, by design — `_mm_stream_pd` on
   16-byte halves defeats the full-cache-line write-combining that makes NT legal
   here (my r1 layout note), so the combination is structurally wrong, not untested.
3. Nothing else failed; the round was deliberately narrow (the r3 VERDICT told L=6
   to stop optimising and measure the clock, so the only kernel change is the one
   with a mechanism argument behind it).

### Borrowed / lent

Nothing borrowed this round; the split-z stores and the clock probe are mine. Both
are lendable: the probe is ~30 lines, answers the machine-state question every
geometry keeps asking (L36's and L17's records both want the node clock), and any
entry can paste it; the split-store trick applies wherever a kernel pays a shuffle
purely to marshal halves for a store (L8_radix8's sequential-store z-pass is the
obvious candidate).

### Next

1. **Read `clk=` off the r4 leaderboard.** Then convert every L=6 cell to cycles and
   re-derive the true B=1 headroom. If clk ≈ base and B=1 sits at ≤4%, propose
   redeploying one L=6 implementer to L=36 (the VERDICT's own suggestion; L=36 B=32
   is at 1.01× MKL and needs bodies).
2. If `fused_s_pfw` wins B=32768 on the node, try the same split-store form for the
   x-pass outputs (t1 is L1-resident scratch so the RFO story is different, but the
   dependency-tail argument still applies).
3. If the node's clk shows turbo and B=1 still does not move: the remaining blocker
   is the pass-boundary store→load chain, and the next lever is issuing the fused
   stage's first plane loads *before* the x-pass finishes its last groups — a
   software-pipelined variant, worth building only with a measured cycle budget.

---

## Round panel_r5 (dev machine = wallaby, Sapphire Rapids Gold 6448Y)

### Where round panel_r4 left me — the clock probe changed everything

r4 node results: first or tied-first in all four cells (B=1 0.219, B=64 0.214,
B=4096 0.397, B=32768 0.566), picks `fused`/`fused_pf`/`fused_pf(w)`/`fused_pfw`,
and **every cell carries `clk=3.89GHz`** — the node turbos to its 3.9 GHz maximum,
not the 2.3 GHz base that three rounds of my own cycle accounting assumed. So B=1
is **852 cycles against the 486-cycle FP-port floor: 1.75×, with ~366 cycles/volume
unaccounted for** (the r4 VERDICT §5a re-derivation, which reopened the geometry).
Two more r4 facts drove this round: (a) my split-z `_s` shapes, worth 12% on wallaby,
were never selected on the node in 12 invocations — so the bottleneck on Cascade Lake
is NOT port 5; (b) L17_rader's mixed-width zmm+ymm passes were selected by the node
tuner in all four L=17 cells (+2.5–5.2%) — the first direct proof that trading
instruction count for 512-bit width pays on this 1-FMA-unit part.

### What I changed (three things)

1. **Mixed-width AVX-512 kernels — the round's headline.** Eight new tournament
   candidates (`z2p`, `z2s`, `z3t`, and `_pf`/`_pfw` variants), guarded by
   `__AVX512F__ && __AVX512VL__ && __AVX512DQ__`, appended LAST in the safest-first
   order so a licence-downclocked node silently keeps its ymm incumbents.
   The structural idea is adopted from **L17_rader** (mixed zmm+ymm passes, node-
   confirmed) with the mechanism argument from **L8_batchsimd**'s round-1 record
   (on a 1-FMA Gold 5218 zmm buys zero FP throughput — 8 doubles/cycle either way —
   so the win is instruction count, shuffle count, and register count):
   * **x-pass zmm**: lanes = 4 adjacent (y,z) sites, 9 groups instead of 18,
     everything 64-byte aligned by construction. 288 uops vs 468.
   * **y-pass zmm+ymm**: per x-plane one zmm codelet on z=0..3 (`loadu`; the
     12-double row stride makes half the 64-B accesses line-split — accepted, see
     "did not work" for the padding analysis) plus one ymm codelet on the z=4,5
     tail (32-B aligned). 384 uops vs 468.
   * **z-pass, three raced options**: existing ymm pairing (`z2p`), existing
     split-store form (`z2s`), and a new all-zmm form (`z3t`): 4 whole z-pencils
     per iteration = 6 aligned zmm loads, in-register 4×6-complex transpose
     (12 `vpermt2pd` + 6 mask-blends each way; forward needs only 4 index vectors
     because the lo/hi 256-bit halves share patterns, inverse needs 6), one zmm
     codelet, 6 aligned stores. 594 uops vs 792.
   * Volume totals: ymm fused 1728 uops → `z2s` 1464 (−15%) → `z3t` 1266 (−27%).
     FP-port floor unchanged at 486 cycles on the node (1 FMA unit) — this attacks
     the unexplained 366 cycles, not the floor.
2. **`L6_FORCE` variant-forcing switch** — the r4 VERDICT's explicit ask ("force
   the `_s` shapes on the node and A/B them"). `L6_FORCE=<name>` (env var, or
   `-DL6_FORCE_DEFAULT='"name"'` at compile time) selects that candidate
   unconditionally, skips the race (fast setup), still passes the correctness
   gate, and reports `variant=<name>!` — the bang marks a forced pick so it can
   never be mistaken for a tournament result. Verified: `L6_FORCE=3pass_s` →
   `variant=3pass_s!`; unknown names fall back to the normal race.
3. **Dual clock probe**: `fft3d_description()` now reports
   `clk=<256-bit>/<512-bit>GHz` — the second number is a serially dependent
   512-bit FMA chain (latency 4, same as ymm, so directly comparable) and is the
   AVX-512 licence-clock measurement the r4 VERDICT asked for (§5b/§6, requested
   for L=17 but every geometry wants it). Wallaby validation: **4.10/4.10 GHz** —
   Sapphire Rapids runs 512-bit heavy FMA at full turbo, confirming the brief's
   claim with a number. The node's second figure in the r5 leaderboard will be the
   first AVX-512 licence clock ever measured on the scoring machine.

### Operation count

Arithmetic unchanged and still closed (PFA 2×3, 48 flops/36 instrs per line, 4 real
mul + 4 FMA + 40 add-class flops per line). What changes is width and uop count:
`z2s` does 270 zmm-FP + 432 ymm-FP uops per volume (arith), `z3t` 324 zmm-FP +
216 ymm-FP + 162 zmm p5 (transpose+codelet swaps). On the node (512-bit FP at
1/cycle on the fused port 0+1, 256-bit at 2/cycle) every variant's FP floor is the
same 486 cycles; on wallaby (2×512-bit units) the `z2s` floor is ~351 cycles.

### What was measured (wallaby, quiet, full-clock invocations; race table = same
process, same clock; rel L2 2.34–2.43e-16 everywhere, bit-identical re-runs, PASS)

B=1 race (µs/vol): **z2s_pf 0.1080 < z2s_pfw 0.1082 < z2s 0.1115 < 3pass_s 0.1126
< fused_s 0.1163 < z3t 0.1209 < z2p 0.1258 < 3pass 0.1269 < fused 0.1287**.

| B | r4 code | r5 code | picked | note |
|---|---|---|---|---|
| 1 | 0.114 µs | **0.108 µs** | **z2s_pf** | −5.3%; sd 0.04%; 443 cycles at 4.10 GHz, under the ymm 486 floor (2-FMA machine) |
| 64 | 0.131 µs/vol | 0.138 µs/vol | 3pass | base-clock race invocation; B=64 races remain noisy on wallaby (r4 note stands); z3t won the raw table (0.2470 vs 0.2533) but inside the 2.5% margin |
| 4096 | 0.196 µs/vol | 0.191 µs/vol | 3pass_nt_pf | wallaby DRAM/L3 still NT country |
| 32768 | 0.249–0.260 | 0.249 µs/vol | 3pass_nt_pf | unchanged; zmm irrelevant at bandwidth |

MKL same-host reference: 0.351 µs at B=1 (full clock), 0.722 µs/vol at B=32768.
Cross-compile at `-march=cascadelake`: all four zmm kernels **spill-free** (0 stack
refs), 72 `vpermt2pd` and 36 `prefetchw` sites emitted as intended.

### Node prediction (falsifiable via the description strings)

* Every cell now reports `clk=A/B`; **B is the node's AVX-512 licence clock** — the
  panel-wide unknown. If B ≈ 3.5–3.9, zmm is near-free on this SKU and I expect
  `z2s_pf` or `z3t(_pf)` to take B=1 at **0.18–0.21 µs** (uop scaling: 852 →
  725/623 cycles if the non-port limiter tracks uop count). If B ≤ 3.2, the 2.5%
  margin plus the clock loss will keep `fused` and the cell stays ≈0.219 — and the
  measured B still converts every past L=17 zmm number into cycles, which is worth
  the round by itself.
* B=64: same logic; z3t narrowly won wallaby's (noisy) race, so a node pick of any
  `z*` variant here is a genuine signal, not noise.
* B=4096/32768: bandwidth-bound; expect `fused_pf(w)` unchanged ≈0.39/0.56. zmm
  variants racing there cost nothing.
* If the monitor wants the r4 VERDICT's A/B: `L6_FORCE=3pass_s` (and `z2s`,
  `z3t`) with `perf stat -e uops_issued.any,cycles` on B=1 now takes one command
  per variant and ~0.1 s of setup each (forced picks skip the race).

### What was tried and did NOT work (with the number)

1. **`z2p` (zmm x/y + ymm permute z)**: 0.1258 at B=1 on wallaby — beaten by z2s
   (0.1115) and even by plain 3pass_s. With the zmm front half, the z-pass's 216
   port-5 `vperm2f128` become the visible tail; the split-store z is the right
   partner for zmm passes. Kept in the grid only because the node's store port
   count differs (1 vs SPR's 2) and could invert z2s/z2p there.
2. **`z3t` on wallaby**: 0.1209 vs z2s 0.1115 at B=1 — the fully-zmm z-pass LOSES
   on SPR despite the lowest uop count (1266). The transpose's vpermt2pd→blend→
   codelet→vpermt2pd→blend chain adds ~6 p5-class latencies to every pencil quad's
   critical path, and SPR had no uop-supply problem to relieve. On CLX, where the
   working hypothesis IS a uop/window limit, the trade may flip — that is exactly
   what the tournament is for. Do not delete it on wallaby evidence.
3. **Padding t1's y-rows 12 → 16 doubles to make the y-pass zmm loads aligned**:
   killed on paper before coding. The x-pass writes t1 in (y,z)-site order, so
   2/3 of its store groups would straddle the padded row boundary and split
   6 zmm stores into 12 half stores (+36 store uops/volume) to save ~18 line-split
   load penalties. Net negative; `loadu` on the 96-byte stride is the cheaper evil.
4. **Not attempted, per documented dead ends elsewhere**: cross-volume software
   pipelining (five schemes built panel-wide in r4, node selected none — VERDICT
   §5), NT stores on the node (rejected three rounds running), prefetchnta (my r2,
   catastrophic), batch-minor relayout (my r1, +50% on uops).

### Borrowed / lent

Borrowed: the mixed-width zmm+ymm pass structure from **L17_rader** (node-proven in
r4), with **L8_batchsimd**'s round-1 port arithmetic as the sizing argument.
Lent: the `L6_FORCE` switch pattern and the dual `clk=A/B` probe are both ~30 lines
and generic — any entry can paste them; L=17 explicitly wants the 512-bit clock
(their r4 cycle model is built on a guessed clock), and L=36/L=8 can use the forcing
switch for their own node A/Bs.

### Next

1. **Read the r5 `clk=A/B` strings and the B=1 pick.** Three outcomes: (i) node
   picks a `z*` variant → the missing 366 cycles were (partly) a uop/window limit;
   push further with a fused zmm y+z plane stage (in-register 4×6 transpose per
   plane, saves the t2 round trip, ~150 more uops off). (ii) node keeps `fused`
   and B≥3.5 → width is free but uops were not the limiter; the remaining suspects
   are 4-wide rename and the pass-boundary store→load latency, and the next probe
   is the monitor's `perf stat` (uops_issued vs cycles decides rename-bound in one
   number). (iii) B ≤ 3.2 → downclock dominates; record it and stop pursuing zmm
   at L=6.
2. If z2s_pf wins B=1 on the node, race a **zmm z-pass with split half-stores**
   (zt's load side + z2s's store side) — the two mechanisms are independent.
3. The B=64 wallaby race noise (r4 and r5) deserves one diagnostic round: race at
   nt=64 exactly (not the 16384 cap) with 15 rounds instead of 7, and see whether
   the pick stabilizes; if the node's B=64 pick flaps in r5, raise per-candidate
   rounds for L2-resident sizes only.

---

## Round panel_r6 (dev machine = wallaby, Sapphire Rapids Gold 6448Y)

### Where round panel_r5 left me — three node facts set this round's agenda

1. **All eight zmm candidates rejected, all four cells, all three processes** (picks:
   `fused` / `fused_pf` / `fused_pf|pfw` / `fused_pfw`), and the probe pair read
   `clk=3.89/2.89GHz` — clk512 = 2.89 is now settled panel-wide (four probes, three
   entries). The r5 VERDICT synthesis: **the cost is the licence transition, not the
   width**. zmm at L=6 is dead; my own r5 fork resolved on its falsifying branch.
2. **My B=1 regressed 0.219 → 0.227 typical (0.227/0.227/0.220) with an IDENTICAL
   pick string**, while L6_pfa read 0.219 in all three runs — the VERDICT scores them
   as reproducibly owning the non-batched cell by ~3.5%. Same disease the VERDICT
   names at L36_mixedradix B=1 (+3.1%, identical pick): a regression that arrives
   with code that the cell never executes.
3. **The one number that decides whether B=1 is finished is unsettled: clk256.**
   Five node probes read 3.89 / 3.89 / 3.89 / 3.27 / **2.89**; the structured reading
   is that sparse chains never engage the AVX2 licence and read the non-AVX clock.
   At 3.89 GHz my B=1 is 852 cycles vs the 486 FP floor (1.75×, real headroom); at
   2.89 it is 633 (1.30×, nearly closed). The VERDICT's L=6 instruction is explicit:
   *"settle clk256, then profile. Stop shipping kernels."* This round obeys it.

### What I changed (no new kernels — a measurement round plus two mechanism fixes)

1. **Clock-density ladder, the round's headline.** Five probes, one process, reported
   in every leaderboard line: `clk256=<sparse,mid,sat> clk512=<sparse,sat>` =
   {1, 5, 8} × 256-bit and {1, 4} × 512-bit independent latency-4 FMA chains =
   0.25 / 1.25 / 2.0 FMA-per-cycle at 256-bit and 0.25 / 1.0 at 512-bit. Every chain
   count satisfies C ≤ 4·throughput on both machines, so each probe is latency-bound
   at exactly 4 cycles per iteration and freq = iters·4/dt uniformly. The saturating
   design and the 256-before-512 issue order are **adopted from L17_winograd** (whose
   2.89/2.89 reading is the crux of the dispute); the sparse chain is my r4 probe, so
   the disagreeing designs finally run back to back in one process. `mid` (1.25/cy)
   is new and is the decision-relevant point: it brackets the real kernel's FP density
   (972 FP uops over 633–852 cycles ≈ 1.1–1.5/cy). Ladder runs twice, per-probe max
   (wallaby can spend ~1 s of a session ramping; validated below).
2. **Licence-tail fix — my mechanism hypothesis for r5's B=1 regression, and it fits
   every symptom.** r5's `create()` ENDED with the 512-bit probe, returning to the
   driver with the core in the AVX-512 licence. At B=1 the driver's whole sample set
   is ~0.5 ms of work — small enough to complete inside the licence-recovery window;
   at large B a single sample exceeds it, which is why only small-B cells got noisy
   (B=1 spread 3.1%, B=64 3.3% in r5, vs 0.5–1.2% before the 512 probe existed in
   r4... where my B=1 spread was already visible the round I added the *256-bit*
   probe — consistent, since that probe is harmless). Fix: `create()` now ends with
   ~20 ms of active 256-bit FMA so the 512 licence expires before the driver ever
   times. Falsifiable: **if the mechanism is right, my B=1 returns to ≈0.219–0.220
   in ALL THREE processes; if it still reads ≈0.227 typical, the hypothesis is dead.**
3. **Grid pruned 32 → 10, and every kernel entry pinned to a 64-byte boundary** —
   the layout defence, the second suspect for (2). Kept: {3pass,fused} ×
   {plain, pfT0·1, pfT0+W} (the only shapes the node has EVER picked, 4 rounds of
   stable pick strings), `3pass_s`/`fused_s` unprefetched (wallaby dev references;
   0-for-12 on the node), `3pass_nt_pf` (single NT representative; NT 0-for-4
   rounds), and `z2s` as the one zmm survivor — kept ONLY as the `L6_FORCE` target
   for the perf-counter A/B the r5 VERDICT §6 asks the monitor to run. Deleted:
   pf2/pft1/pft1w/pfw2 rows, 4 NT kernels, `fused_s_pf`, `*_s_pfw`, `z2p*`, `z3t*`
   (the whole 4×6 zmm transpose pass), and the zmm prefetch hooks. Setup drops
   0.60 → ~0.3–0.55 s.
4. **Settle spin before the tournament** (100 ms of 256-bit FMA) — adopted from
   **L17_rader**'s r5 finding (a fixed-order table on a ramping clock mis-ranked
   bit-identical work by 76%; their fix alone was worth −3.6% at B=1). My round-robin
   + per-candidate-minimum already blunts drift, but round 0 was still rankable on a
   cold clock and the spin costs nothing scored.

### Operation count

Unchanged and closed since round 1: PFA 2×3, 48 flops / 36 instructions per line,
972 vector FP uops per 6³ volume, 486-cycle two-port floor on the node. This round
adds zero uops to any kernel; everything above is plan-time or address-level.

### What was measured (wallaby; same-window statistics quoted per the r5 VERDICT rule)

* **B=1: 0.108 µs** (full-clock window; picks `z2s`, which on SPR is legitimately
  fastest and on the node will keep losing the race by the licence margin — by
  design). Forced `fused` in a half-clock window: 0.2635 µs with the ladder reading
  2.10 across — kernel and probes now agree in BOTH clock states.
* **The headline validation: driver median collapsed onto the min.** Before this
  round: min 0.114 / median 0.222 / sd 29.5% at B=1. After the licence-tail fix +
  settle spin: **min 0.108–0.114, median = min, sd 0.04%** across invocations in the
  same windows. The bimodal-median artifact my r2 record first flagged is gone from
  the B=1 driver statistics on wallaby.
* B=64: 8.365 µs/call = **0.1307 µs/vol** (sd 0.05%), matches r4/r5.
* B=4096: **0.190 µs/vol**, matches r5's 0.191.
* B=32768: driver 0.275 µs/vol this session (r5: 0.249); the race table matches r5
  (`3pass_nt_pf` 0.254, `fused_pfw` 0.321, `3pass_pfw` 0.325) so I read the driver
  delta as session state, not code. Node-relevant rows unchanged.
* rel L2 vs numpy: 2.342e-16 (B=1), 2.428e-16 (B=64), 2.425e-16 (B=4096), 2.424e-16
  (B=32768); bit-identical across re-runs; all PASS.
* Cross-compile `-march=cascadelake`: all 10 vector kernels spill-free (0 stack refs
  except `3pass_nt_pf`'s callee-save frame), every kernel entry at a 64-byte-aligned
  address (verified in the object's symbol table), `prefetchw` emitted.

### What was tried and did NOT work (with the number that killed it)

1. **First ladder implementation kept the FMA chains in a local ARRAY (`__m256d
   x[C]`), and gcc kept the array in memory** — every FMA gained a store-forward
   round trip and the probe under-read the clock by an exact ~2×: it reported
   **2.10 GHz in the same process whose kernel timing implied ≥3.9** (0.108 µs B=1
   needs ≥3.2 GHz even at wallaby's 351-cycle zmm floor). Identical readings across
   runs made it look like a real clock state; it was a codegen artifact. Fixed by
   writing every chain as a named local. **Lesson for anyone porting the ladder:
   chains must be named variables, never arrays — and validate the probe against a
   kernel whose cycle count you know.**
2. Nothing else was attempted, deliberately: the r5 VERDICT told L=6 to stop
   shipping kernels, and both structural hypotheses (uop count via zmm; OoO window
   via L6_pfa's `fused_sp`) are already node-falsified at B=1.

### Borrowed / lent

Borrowed: the saturating multi-chain probe design and the 256-before-512 issue order
from **L17_winograd** (r5); the pre-race settle spin from **L17_rader** (r5). Lent /
lendable: the full ladder (5 probes + spin, ~120 lines, self-validating against the
known-answer machine) — L=17 and L=36 both build cycle models on the contested
clk256, and any entry can paste it; also the licence-tail warning: **any entry whose
`create()` ends with 512-bit work (a zmm probe, a zmm tournament candidate racing
last) hands the driver a licence-degraded core, and at small B the whole measurement
can fit inside the recovery window.** Check your own create() tail.

### Node predictions (falsifiable via the description strings)

* Every L=6 cell carries `clk256=a,b,c clk512=d,e`. Expected: a≈3.89 (sparse, the
  non-AVX clock), c≈2.89 if L17_winograd's reading is right; **b (mid, 1.25 FMA/cy)
  is the decision number.** If b≈3.89: dense-enough-for-L=6 ymm runs at 3.89, B=1 is
  852 cycles = 1.75× floor, and the 366-cycle gap is real — next round asks the
  monitor for `perf stat` (uops_issued vs cycles) on forced `fused` vs `z2s`, which
  both switches already support. If b≈2.89: B=1 is 633 cycles = 1.30× floor, the gap
  is 147 cycles, and I will argue L=6 B=1 is within ~1.3× of closed and support
  redeploying effort. If e≠2.89 the r5 consensus itself is in trouble (not expected).
* **B=1: 0.219–0.220 in all three processes** (licence-tail + layout fixes restore
  the r4 distribution). Pick `fused`. If it still reads 0.227 typical, both of my
  regression hypotheses are wrong and the remaining suspect is something in the
  driver's process state that only the monitor's perf counters can see.
* B=64 `fused_pf` ≈0.214; B=4096 `fused_pf(w)` ≈0.39; B=32768 `fused_pfw` ≈0.566
  (unchanged; no kernel differences on the node's picks).

### Next

1. Read the ladder off the r6 leaderboard; branch on `mid256` as above. Either way
   the clk256 dispute ends this round — same process, both probe designs, plus the
   density point that actually matters.
2. If B=1 lands back at 0.219×3, propose the licence-tail check to L6_pfa and any
   entry with end-of-create 512-bit work; if it stays 0.227, request the monitor's
   perf A/B (`L6_FORCE=fused` vs `z2s`, B=1) — the switches are shipped and the
   grid is small enough that setup is fast in forced mode.
3. The B=32768 cell remains at the compulsory-traffic floor for both entries; per
   my r5 note and L6_pfa's, nobody should spend another round there.

---

## Round panel_r7 (dev machine = wallaby, Sapphire Rapids Gold 6448Y)

### Where round panel_r6 left me — a round that was never scored

panel_r6 was halted before its timing pass (see `results/panel_r6_abandoned_no_timing/WHY.md`),
so nothing in my r6 section has node numbers: the licence-tail fix, the 64-byte kernel
pinning, the pruned grid and the clock-density ladder all ship *this* round untested.
Standings are still panel_r5: L6_pfa reproducibly owns B=1 (0.219 in all three runs vs my
0.227/0.227/0.220 — the regression my r6 fixes target); I own B=64 (0.215) and B=32768
(0.566); they hold B=4096 (0.391 vs my 0.397).

**The r7 brief carries a correction that reopens the biggest closed question at L=6.**
Intel's Specification Update 338848-028US (now §08 of the corpus): on the Gold 5218 at
1–8 active cores the AVX2 and AVX-512 licence clocks are **identical, 2.9 GHz** (non-AVX
3.9). The r5 zmm rejection (eight candidates, 0 picks in 12 invocations) was read as a
licence penalty; if dense ymm code actually runs at 2.9 too, that reading is wrong and
the rejection needs a different cause — the r5 shapes' actual defects (stride-12 `loadu`
in the y-pass, half of which straddle cache lines, and the t2 round trip) are the
suspects. The brief says plainly: prefer 512-bit. The r5 VERDICT's "stop shipping
kernels" instruction was written under the licence theory this correction overturns, so
this round ships kernels again — behind the tournament's safety interlocks.

### What I changed (two new zmm shapes, one tournament fix, one adopted probe, one prune)

1. **`zxf` — zmm x-pass + the node-proven ymm fused y+z.** The x-pass is the one pass
   that is perfectly 512-bit-shaped: 64-byte-aligned loads/stores by construction,
   stride 72 doubles, zero shuffles, 9 groups instead of 18. 288 uops replace 576; the
   second stage is token-identical to `fused`'s (the plane loop is now a shared macro,
   `L6_FUSED_YZ`). ~1440 uops/volume vs fused's 1728 (−17%). Minimal-risk probe of
   "does 512-bit width pay at all on the node when the rest of the kernel is unchanged".
2. **`zff` — zmm x-pass + fully fused zmm/ymm y+z per plane; fixes both defects of
   r5's `z2s`.** Every (y,z) plane of t1 is 72 doubles = 9 whole cache lines, 64B-aligned,
   so it loads as **9 aligned zmm** (z2s used 6 stride-12 `loadu`, half line-split). Row
   registers are built in-register: rows 0/2/4 are v0/v3/v6 directly; rows 1/3/5 are one
   `valignq(v_{k+1}, v_k, 4)` each; the (z4,z5) ymm tails are 3 free casts + 3
   `vextractf64x4`. One VD6Z (zmm) + one VD6 (ymm) is the whole y-DFT of the plane. The
   z-DFT runs per row pair: 4 `vpermt2pd` gather lane z of both rows into z-major ymm
   (z=0..3), 2 `vperm2f128` handle z=4,5 from the tails, then the standard ymm codelet
   and the `fused`-style full-width store tail (deliberately NOT split stores — 0-for-12
   on the node). **No t2 round trip** (z2s pays 216 complex stored + reloaded per
   volume). ~1302 uops/volume (−25% vs fused). FP-port floor is unchanged at 486 node
   cycles by design (270 zmm-FP at 1/cy + 432 ymm-FP at 2/cy) — this attacks the
   ~147-or-366-cycle overhead, not the floor.
3. **Per-candidate licence warm-up in the tournament (LITERATURE §08 §4.3).** The
   round-robin race interleaves ymm and zmm candidates, and CLX licence state persists
   ~670 µs — comparable to a whole 2 ms trial, so a ymm candidate timed right after a
   zmm one ran up to a third of its trial at the lower clock, a *systematic table-order
   bias* (it hits the same candidates every round, so per-candidate minima do not wash
   it out). Every candidate now runs itself untimed for ~0.7 ms immediately before each
   timed trial: it establishes its own licence and pays its own transition outside the
   timing. This also makes the r5 zmm rejection data cleaner to reinterpret if this
   round's picks differ.
4. **`kclk` probe — ADOPTED FROM L6_pfa (their panel_r6 round).** Dwell ~2 ms in the
   *chosen* kernel, then immediately time a ~150 µs sparse ymm FMA chain (which never
   raises the licence by itself, so it reads the licence the kernel established); median
   of 9 pairs, reported as `kclk=X.XX` in the description of every cell. This is the
   number that converts my node times into cycles — better than my r6 `mid256` density
   probe, which brackets the kernel's FP density instead of measuring the kernel itself.
   Both now ship side by side, so the r7 leaderboard settles the clock question
   redundantly.
5. **Pruned:** the ymm split-store shapes `3pass_s`/`fused_s` and the `L6_ZPAIR_S`
   macro are deleted (0 picks in 12 node invocations, SPR-only mechanism; they were
   "wallaby dev references" and wallaby's z2s pick now serves that purpose). Grid:
   10 → 12 (7 ymm + 5 zmm), all entries still 64B-pinned, zmm strictly after every ymm
   incumbent so a zmm pick must clear the 2.5% margin against the best ymm time.

### Operation count

Per-line arithmetic unchanged and closed since round 1 (PFA 2×3, 48 flops / 36 instrs,
no twiddles). Volume uop totals (fused-domain, approximate): `fused` 1728, `zxf` ~1440
(−17%), `zff` ~1302 (−25%), `z2s` ~1464. All have the identical 486-cycle FP-port floor
on the node (1 × 512-bit FMA unit ⇒ zmm FP at 1/cy, ymm at 2/cy, 8 doubles/cy either
way); the zmm shapes trade nothing at the floor and remove load/store/front-end uops
above it.

### What was measured (wallaby, quiet, full-turbo session — `clk256=4.10,4.10,4.10
clk512=4.10,4.10 kclk=4.10`; rel L2 2.342e-16 (B=1), 2.428e-16 (B=64), 2.425e-16
(B=4096), 2.424e-16 (B=32768); bit-identical re-runs; all PASS)

B=1 race table (µs/vol): **z2s 0.1091 (chosen)** < 3pass 0.1270 < fused_pf 0.1285 <
fused 0.1287 < fused_pfw 0.1290 < 3pass_pf 0.1297 < zxf_pf 0.1309 < zxf 0.1312 <
3pass_pfw 0.1355 < **zff 0.1445 < zff_pf 0.1467** < 3pass_nt_pf 0.3453.

| B | r6 code (wallaby) | r7 code (wallaby) | picked | note |
|---|---|---|---|---|
| 1 | 0.108 µs | **0.108 µs** (sd 0.03%) | z2s | unchanged, SPR still loves the split-store z |
| 64 | 0.131 µs/vol | 0.137 µs/vol | (session state) | within the B=64 wallaby noise band flagged since r4 |
| 4096 | 0.190 µs/vol | 0.198 µs/vol (sd 2.3%) | — | session state, race rows match r5/r6 |
| 32768 | 0.249–0.275 µs/vol | 0.263 µs/vol | — | unchanged regime |

Forced-variant cross-checks on one input: `L6_FORCE=zff`, `zxf`, `fused` and the raced
pick produce **bit-identical output files** (expected — every kernel performs the same
FP operations in the same per-line order), and forced `zff` PASSes numpy directly at
2.342e-16. Cross-compile at `-march=cascadelake`: **all 12 kernels spill-free** (0
rsp/rbp references inside every kernel body) and **all 12 entry points 64B-aligned**
(checked in the object's symbol table). Haswell (`-march=haswell`, no AVX-512) and CLX
`-Wall -Wextra` builds are clean. Setup 0.57–0.67 s (B≤4096), 1.66 s (B=32768).

### What was tried and did NOT work (with the number)

1. **`zff` on wallaby loses badly: 0.1445 vs z2s 0.1091 (−32%) and vs plain ymm fused
   0.1287 (−12%), despite the lowest uop count in the file.** Reading: on SPR the
   split-store z-pass dominates every other z-form (consistent every round since r4),
   and zff's z-gather chain — VD6Z result (zmm) → `vpermt2pd` → ymm codelet →
   `vperm2f128` → store — is longer than z2s's extract-to-memory tail. This is an
   *SPR statement, not a node statement*: wallaby has inverted the node's ranking of
   z-forms in every round where both were measured (my `_s` shapes won 12% on wallaby
   and went 0-for-12 on the node). The node race decides; recorded so the wallaby
   number does not get mistaken for a kill.
2. **Not built, consciously: a zff twin with the split-store tail** (`zff_s`). It would
   win wallaby (see 1) but the node has rejected split stores in 12 straight
   invocations, and grid discipline beats collecting SPR trophies.
3. **Not re-tried, per documented dead ends:** NT stores on the node (0-for-4 rounds),
   prefetchnta (r2, catastrophic), batch-minor relayout (r1, +50% uops), cross-volume
   software pipelining (r4, node-rejected panel-wide), t1 row padding 12→16 doubles to
   align the z2s y-pass (r5, killed on paper: +36 store uops to save ~18 splits).

### Borrowed / lent

Borrowed: **`kclk` kernel-context licence probe from L6_pfa (r6)** — their design,
credited above, now reporting in every cell of mine; **the per-candidate licence
warm-up is LITERATURE §08 §4.3's recommendation**, first implemented at L=6 here. The
zxf/zff shapes are mine (zff's aligned-load construction — 9 aligned zmm + 3 `valignq`
+ 3 `vextractf64x4` replacing line-splitting `loadu` — is the reusable piece: any
geometry whose plane rows are not zmm-multiples pays the same split tax; L=17's
records describe similar stride pain). Lent/standing: the licence-tail warning from my
r6 record (any `create()` that ends with 512-bit work hands the driver a
licence-degraded core) — with zmm candidates racing again panel-wide, every entry
adding them should check its create() tail.

### Node predictions (falsifiable via the description strings)

* **Every cell reports `kclk` next to the r6 ladder.** kclk is the decision number:
  2.89 ⇒ B=1 fused = 633 cycles = 1.30× the 486 floor (~147 cycles of overhead);
  3.89 ⇒ 852 = 1.75× (~366). My bet, following L6_pfa's r6 reasoning and Intel's
  table: **kclk = 2.89**.
* **If kclk = 2.89, a zmm pick at B=1 is live**: zxf/zff run at the same clock and the
  same FP floor with 17–25% fewer uops; if the B=1 overhead is uop/front-end-side,
  zff should take the cell at ~0.19–0.21 µs. If the overhead is latency joints
  (pass-boundary store→load), the uop reduction buys nothing and the incumbents hold —
  that outcome, *with kclk known*, finally kills the uop theory at L=6 for good.
* **If kclk = 3.89**, zmm needs its whole 25% uop saving to convert to cycles just to
  break even against the licence drop; expect `fused` to hold everywhere and the zmm
  rows to lose by ~10–20% — same result as r5, but this time correctly attributed.
* B=1 also tests the r6 licence-tail + layout fixes: **0.219–0.220 in all three
  processes** (vs r5's 0.227/0.227/0.220) even if no zmm shape is picked.
* B=64: `fused_pf` ≈ 0.214, or `zxf_pf` if width pays (B=64 runs at B=1 speed on the
  node, so the B=1 logic transfers). B=4096: `fused_pf(w)` ≈ 0.39. B=32768:
  `fused_pfw` ≈ 0.566 — both bandwidth-closed, no zmm effect expected.

### Next

1. **Branch on kclk and the B=1 pick.** (i) zmm picked → iterate on the winner: the
   next uops to remove are the x-pass's (try a 2-volume-interleaved zmm x-pass at
   B≥2, and a zff variant whose z-gather uses `vpermt2pd` to build zmm pairs feeding
   a zmm z-codelet). (ii) zmm rejected AND kclk=2.89 → the uop theory is dead with
   clean attribution; request the monitor's
   `perf stat -e uops_issued.any,cycles,cycle_activity.stalls_mem_any` on
   `L6_FORCE=fused` vs `zff` at B=1 (both names work in the switch) — the remaining
   suspect is the t1 store→load joint, which no uop count sees. (iii) kclk=3.89 →
   licence story confirmed for real this time; stop zmm at L=6 permanently and record
   it as closed in LITERATURE terms.
2. If B=1 still reads 0.227-typical with identical picks, both r6 mechanism fixes are
   falsified and the cell's disease is outside this file (driver process state);
   hand it to the monitor with the forced-pick switch.
3. B=32768 remains bandwidth-closed; nobody should spend a round there (third round
   this note survives).
