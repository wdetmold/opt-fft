# L45_pfa — Good–Thomas PFA 9×5 per axis, two-sweep plane fusion, odd-L tails

`impl/L45_pfa.c`, `fft3d_name() == "L45_pfa"`, L = 45 only.

---

## Round panel_r6 (first implementation — L = 45 is new this round)

### Technique

Row–column 3D DFT; every 45-point line is a **Good–Thomas / prime-factor 9×5
codelet** on interleaved-complex vectors whose lanes are a spectator axis.
gcd(9,5) = 1, so with

```
input  (Ruritanian): n = (5*n1 + 9*n2) mod 45      n1 in [0,9), n2 in [0,5)
output (CRT):        k = (10*k1 + 36*k2) mod 45    10 = 5*[5^-1]_9, 36 = 9*[9^-1]_5
```

`W45^{nk} = W9^{n1 k1} · W5^{n2 k2}` exactly (the two cross terms carry a
factor 45 and drop; `50 ≡ 5`, `324 ≡ 9 (mod 45)`), so the 45-point DFT is **5
DFT9s then 9 DFT5s with zero twiddles in between**, both index maps folded into
compile-time addressing. Modules:

* **DFT9 = Cooley–Tukey 3×3** (9 = 3² is a prime power, PFA cannot enter):
  6 × DFT3 (6 FMA-port ops + 1 swap) + 4 twiddle CMULs (2 ops + 1 swap)
  = **44 FMA-port ops + 10 swaps**. Same module family as L36_pfa; their round-1
  record shows three failed attempts to close the 88→80 gap vs genfft's n1_9 —
  not re-attempted.
* **DFT5 = FFTW n1_5's FMA DAG**: the √5/4 cosine split
  (`tm = x0 − te/4; tp,tq = tm ± (√5/4)·ta`) and the scaled-sine trick
  (`sin(4π/5) = 0.618… · sin(2π/5)`, so `tv = t4 + 0.618·t7` needs one FMA and
  one shared constant) give **16 FMA-port ops + 2 swaps** — 2 ops/DFT5 fewer
  than the plain two-rotation form my rival counted (their 18; see
  attribution).

Per 45-point line over PW lanes: 5·44 + 9·16 = **364 FMA-port vector ops + 68
swaps**. (L45_mixedradix counts 382 for the same maps — the whole difference
is the DFT5 form.) A plain CT 9×5 with twiddles would add 32 nontrivial CMULs
= +64 ops/line; PFA deletes them all.

Pass structure = **the two-sweep plane fusion that won L=36** (adopted from
L36_pfa r2, transitively L36_mixedradix r1):

```
phase 1, per x-plane:
    z transform  lanes = PW y-rows, PW×PW complex-granule register transposes
                 on load and store, into plane scratch pl[y][kz]
                 (heap, plan-owned, row pitch 52 complex — see below)
    y transform  lanes = PW kz (contiguous in pl), store to mid[x][ky][kz]
phase 2:
    x transform  lanes = PW kz, 45 streams at stride 4050 doubles, mid -> out
                 (in place when mid == out)
```

**Odd-L tails** (45 = 11·4 + 1 at PW=4; no vector width divides 45):

* Out-of-place subpasses (z: in→pl, y: pl→mid) **overlap** the last group
  (starts at 45−PW); recompute is idempotent, costs one extra group in 12
  (+6.7% of those passes), needs no second codelet.
* The **in-place x pass cannot overlap** (it would re-read already-transformed
  columns — this is a correctness trap, not a tuning choice). Its tail column
  kz = 44 runs a **PW=1 instantiation of the same codelet template** (one
  complex per 128-bit vector), 45 lines/volume ≈ 3% of the arithmetic. On the
  1-FMA-unit node this beats the rival's masked-full-width tail on paper:
  364 SSE ops dual-issue on ports 0+1 (~190 cyc) vs 382 zmm ops at 1/cyc.
* All in/out/mid vector accesses are **unaligned** (`aligned(8)` vector
  typedef): 45 complex = 720 B row stride rotates alignment mod 64. Odd-L toll,
  ~50–75% of those 64 B accesses split a cache line; both L=45 entries pay it.

**Modes and prefetch** (create-time tournament, 3% simplest-first hysteresis,
`FFT45_PW/MODE/PF` env forcing, pick reported in `fft3d_description()` — all
L36_pfa r3–r5 machinery):

* modes: `inplace` (mid = out) and `scratch` (one reused plan-owned volume).
  **No NT-store or pipe modes**: the node rejected NT at L=36 four rounds
  running, and 720 B row alignment rotation makes 64 B-aligned stream stores
  structurally impossible at L=45 anyway.
* pf=1: paced T1 read-prefetch of phase 1's linear in-stream (`FFT45_PFD` =
  32 KB, 2·NGRP steps/plane, ceil-covered) + phase-2 pre-coverage of the next
  volume's first ~63 KB (`FFT45_PFN` = 2 lines/tile).
* pf=2: pf=1 + write-intent `prefetchw` on whichever pass stores cold lines
  (phase-1 mid-stream when inplace, phase-2 out-streams when scratch).
* Phase 2 always hand-prefetches its 45 read streams one line ahead (more
  streams than the L2 streamer tracks; L36 measured −14% without it).

Correctness gate: every candidate's volume-0 output is checked at 1e-13
relative against a **scalar O(45²)-per-line reference DFT built into
`fft3d_create()`** — an independent ground truth, not another vector path.

### Operation count

Per volume: 3·45² = 6075 lines. At PW=4: 45 planes × (12 z-groups + 12
y-groups) + 45·11 phase-2 tiles = 1575 vector codelets × 364 =
**573,300 zmm FMA-port ops** + 45 PW1 tail lines (16.4k SSE ops) + ~211k
port-5 shuffles (68/codelet + 192 transpose shuffles per z-group). Node
port-0 floor ≈ 573k cycles (one 512-bit FMA unit). Verified in the object
code under the monitor's flags: each codelet site compiles to exactly
191 FMA + 153 add/sub + 20 mul + 68 shuffles — the model is exact, no spill
bloat (44 stack movs in the whole phase-2 body).

### What was measured (wallaby, Gold 6448Y — the host toggled fast/slow ~2×
windows all session, same as L36_pfa r2/r5 reported; only same-window
comparisons are meaningful, and MKL on the same case is quoted as the window
sentinel)

Driver min per transform, rel_l2 vs numpy **4.04e-16** at every batch tried
(1, 2, 4, 5, 8, 32, 64), bit-identical re-runs, `in` unmodified. AVX2-only
build verified end-to-end on wombat (PASS 4.038e-16, B=4, 580.7 µs/vol vs MKL
869.7 — wins there too).

| B | µs/vol (best quiet window) | MKL same window | ratio | tuner pick |
|---|---|---|---|---|
| 1 | **204.0** | 305.4 | 1.50× | pw4, inplace, pf=0 |
| 4 | **213.7** | 315.3 | 1.48× | pw4, inplace, pf=0/2 (3% band) |
| 8 | **215.4** | 335.0 | 1.56× | pw4, inplace, **pf=2** |
| 32 | **229.7** | 398.4 | 1.73× | pw4, inplace, **pf=2** |
| 64 | **265.3** | 405–418 | 1.57× | pw4, inplace, **pf=2** |

In-arena (back-to-back, load-immune): at B=64, inplace-pf2 **225.4** vs
inplace-pf1 321.7 vs inplace-pf0 355.6 — the write-intent prefetch is worth
−37% in the streaming regime, exactly the L36_pfa r5 / L6_unrolled r3
mechanism transplanted. At B=8: pf2 215.0 vs pf0 256.5 (−16%). At B=1 pf is
flat (391.9/395.8/394.9 in one arena) and the hysteresis keeps pf=0.
Scratch loses to inplace everywhere on wallaby (B=1 forced end-to-end:
scratch ~350 vs inplace ~210).

Phase split at B=1 (diagnostic builds, `FFT45_SKIP1/2`): phase 1 ≈ 190 µs,
phase 2 ≈ 38 µs — **phase 1 is ~85% of the transform**, and it is
data-movement-bound, not arithmetic-bound: replacing the z-subloop's codelet
with a copy (`FFT45_ZCOPY`) changed nothing (191 vs 190 µs), and replacing
both codelets (`FFT45_ZCOPY + YCOPY`) still cost 145–165 µs. The transposes,
the pl round-trip, and the split unaligned accesses ARE the B=1 time; codelet
op-count tuning is second-order here (it will matter more on the node's
single FMA port).

### What was tried and did NOT work — with the number that killed it

1. **Plane-scratch row pitch 48 complex (the "obvious" 64B-aligned pad,
   and what L45_mixedradix uses)**: paired A/B/A/B runs gave 281.0/242.0 µs
   (pitch 48) vs 204.0/199.4 (pitch 52) at B=1 — **pitch 52 is 15–20%
   faster**. Theory: 48 complex = 12 lines/row, and gcd(12,64) = 4 lands the
   y-pass column walk (45 rows, one line each) on only 16 L1 sets; 13
   lines/row (52 complex = 832 B, still 64B-aligned rows) is coprime with 64
   and spreads over all sets. The rival runs pitch 48 and is still faster
   overall, so this is not the whole story of the y-pass — but it is free and
   it measured decisively. **PPITCH is a `-D`-overridable constant; the node
   should re-check 48 vs 52 once** (its L1 is 32 KB, 64 sets, 8-way — same
   set geometry, smaller capacity).
2. **L1-prefetching the next pl column in the y-subloop (`FFT45_PLPF`)**:
   219.1/382.6 vs 204/199 µs — never better, sometimes much worse. Dropped.
3. **Scratch mode at B=1** (phase 1 → S, phase 2 S → out): ~350 vs ~210 µs
   forced end-to-end — the extra volume round-trip loses at every batch on
   wallaby, matching the node's four-round preference for inplace at L=36.
4. **Timing the subloops by deleting them**: `SKIPY` alone lets gcc
   dead-store-eliminate the entire z-subloop (0.354 µs "measurement");
   `SKIPZ` alone makes the y-subloop read uninitialized stack — **denormals,
   1051 µs, 5× slower than doing the work**. Diagnostic numbers from
   dead-code builds are garbage unless the data stays live and finite; the
   `ZCOPY/YCOPY` copy-instead-of-codelet variants are the honest probe.
5. **Stack-ASLR 4K-aliasing hypothesis for the bimodal B=1 runs** (some
   whole processes ran ~2× slow, e.g. 444 µs with sd 0.34%): moved the plane
   scratch from the stack to a page-aligned plan-owned heap buffer. The
   change is kept (deterministic addressing, 37 KB off the stack, zero cost)
   but it did **not** cure the bimodality; 12 subsequent processes were all
   fast, so the slow processes were most plausibly wallaby load bursts
   lasting a whole process. Do not chase this again without hardware
   counters; `perf` is unavailable on wallaby (`perf_event_paranoid = 4`).
6. **NT stores / pipe mode**: not built, by design — see technique. At L=45
   the y-pass and x-pass store at 720 B row stride, so 64-byte-aligned
   `vmovntpd` cannot be issued at all without staging through an aligned
   bounce buffer, which is exactly the extra pass the node kept refusing.

### Attribution

* **Two-sweep plane-fused structure, interleaved-complex spectator lanes,
  6-op DFT3 / 2-op CMUL forms, TRNC granule transpose, PFIN/PFNX paced read
  prefetch, pf=2 prefetchw, tuner + hysteresis + env forcing + pick
  reporting, tuning-arena-must-stream rule**: L36_pfa (rounds 2–5),
  transitively L36_mixedradix r1 (structure) and L6_unrolled r3 (prefetchw).
* **Overlap tails for out-of-place subpasses**: converged independently with
  L45_mixedradix this round; read their header mid-design — their masked
  z-tail granule (4×16 B masked loads instead of an overlapped unaligned
  granule) is slightly cleaner and worth stealing next round.
* **16-op DFT5**: FFTW n1_5's DAG from the corpus (√5/4 split + s2 = φ⁻¹·s1).
* **Same-window rival comparison** (both entries measured back-to-back):
  L45_mixedradix 174.9–177.8 µs vs my 204–224 at B=1; 250.7 vs 265.5 µs/vol
  at B=64. They lead by ~6–15% on wallaby despite 18 more FMA-port ops/line,
  with a structurally near-identical program — the difference is in phase-1
  data movement details I did not fully isolate this round.

### Node predictions (stated so they can be scored)

* **B=1: 240–300 µs** (arith floor 573k cycles ≈ 150–200 µs at 2.9–3.9 GHz,
  plus the phase-1 movement overhead that dominated wallaby, minus wallaby's
  2-FMA advantage which the node lacks — the op-count gap vs the rival
  (364 vs 382) matters more on one FMA port, so the node ordering may invert
  vs wallaby's).
* **MKL on the node at L=45 B=1: ~450–700 µs** (it ran 12.9–26 GF/s on
  wallaby; 45 = 3²·5 is a decent MKL case but nowhere near its L=36 form).
  Expect to beat every library at every batch.
* **Pick: pw4-inplace-pf0 at B=1/B=4, pw4-inplace-pf2 at streaming batches.**
  If the node picks scratch anywhere, that is new information (wallaby never
  did).

### Next

1. **Close the phase-1 movement gap.** It is 85% of B=1 and none of it is
   arithmetic. Candidates in order: (a) steal L45_mixedradix's masked z-tail
   granule (kills the overlapped split loads in every z-group's 12th
   granule); (b) try their stage order (DFT5s first, DFT9s second — their
   T-array write pattern is stride-9×64B vs my stride-5×64B, and their DFT9
   then reads 9 contiguous slots hot; mine reads 5); (c) merge the
   z-store-transpose directly into pl with masked 16 B stores for the tail
   column. Each is a ≤5% candidate; wallaby's windows can resolve none of
   them reliably — put them in as tuner-gated variants and let the node
   tournament decide, or ask the monitor for one `perf stat` pair.
2. **PPITCH 48 vs 52 on the node** (one `-DPPITCH=48` control run).
3. If the node's streaming cells land far above the ~4.4 MB/vol × node
   bandwidth floor, revisit pacing constants (`FFT45_PFD/PFWD` sweep) —
   wallaby was too noisy to tune them beyond L36's carried-over values.
4. The 88→80 DFT9 instruction gap via transcribing genfft's n1_9 DAG (~3%):
   only via the actual DAG, not by hand — L36_pfa burned three attempts.
