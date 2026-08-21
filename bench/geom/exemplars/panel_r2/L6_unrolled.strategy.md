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
