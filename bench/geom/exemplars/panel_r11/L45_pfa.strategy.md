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

---

## Round panel_r7

### Where this started

No node numbers exist for L=45 (r6 was halted before timing). r6 wallaby
standings: me 204.0 µs at B=1 / 265.3 µs/vol at B=64, rival L45_mixedradix
172.8 / 251.0. Their r7 entry (written before mine this round) improved to
167.7 / 249.1 and borrowed my DFT5 and pitch-52; I owed them a read of their
code. That read paid for the whole round.

### What changed (five things, in order of measured impact)

1. **Single-base addressing in phase 1 — the round's main find, −14% at
   B=1.** An objdump diff against L45_mixedradix's kernel (compiled with the
   node's flags) showed my `phase1_pw4` carrying **758 scalar instructions
   against 42 in their entire transform**. Cause: my granule accesses were
   written as `px + ((yb+j)*L + zb)*2` with runtime `yb`; gcc precomputed a
   48-entry offset table, spilled it, and **reloaded an offset into the same
   GPR (%rbx) before every vector load** — an extra load plus a serial
   register dependency through the whole z-gather. Fix, copied from the shape
   of their code: hoist ONE base pointer per block (`rows = px + yb*90`,
   `prow = pld + yb*2*PPITCH`, and in the y-subloop `pcol/mcol = base + 2*zb`)
   so every access is base + compile-time constant. phase1_pw4: 2574 → 2037
   instructions, scalar 758 → 217. Wallaby B=1, same window: 203.5 → 175.6.
   **Lesson for every entry with runtime-offset codelet macros: check the
   scalar-instruction count of your hot function; gcc will not fold
   `(runtime + const)*const` addressing for you.**
2. **`#pragma GCC optimize("unroll-loops")` — monitor-build parity, worth
   ~10% on the node.** L45_mixedradix r6 discovered the scored build lacks
   tryout.sh's `-funroll-loops`; unlike them (whose no-unroll build got
   *faster* after explicit loop pragmas), my code lost 10% without it
   (202 → 222–234 µs paired A/B/A/B, MKL sentinel flat at ~305) — the gain
   is mostly `-frename-registers`, which `-funroll-loops` implies, on the big
   straight-line codelet bodies. The file-level pragma pins the fast codegen:
   with `-fno-unroll-loops` forced, 202.3 µs (was 222–234). Brief explicitly
   permits compiler-specific pragmas.
3. **Flat phase-2 tiling — borrowed from L45_mixedradix r7.** x pass tiles
   the flat (y,z) index: 2025 = 506·4 + 1 → 506 full tiles + ONE masked tail
   call per volume (`_mm512_maskz_loadu_pd(0x03)` / mask-store, dead lanes
   compute on zeros), replacing 495 tiles + 45 PW=1 tail lines. The PW=1
   codelet instantiation and its 45-prefetch-per-call preamble are deleted.
4. **Masked z-column tail — borrowed from L45_mixedradix r6, implemented
   with 128-bit inserts/extracts instead of their masked-load TRNC.** The
   z-pass granule loop now runs 11 full granules + GCOL/SCOL (4×16 B column
   gather/scatter for z=44; complex elements are 16B-aligned so these never
   split a line), replacing the overlapped 12th granule whose unaligned 64 B
   accesses split cache lines ~75% of the time.
5. **pf became a 4-level ladder and phase 2's 45-stream poke is now level 1
   instead of always-on.** A/B at B=1: compiling PF45 out gained ~2% on
   wallaby (172.2 vs 175.8, mixed windows) — on a 2 MB-L2 machine at B=1 the
   volume is L2-resident and 22,770 prefetches/volume are pure overhead. But
   the node's L2 is 1 MB (volume does NOT fit) and L36 measured −14% without
   the analogue, so this is machine-dependent: runtime-gated, tournament
   decides. Ladder: 0 = nothing, 1 = PF45, 2 = +PFIN/PFNX, 3 = +prefetchw.
   16 candidates ({pw2,pw4} × {inplace,scratch} × pf0..3), same 3% hysteresis.

Changes 3+4 were op-neutral on wallaby B=1 (202.3 both before and after, same
window) — they cut instructions retired and split accesses, which should
matter more on the node's single 512-bit FMA port and smaller caches.

### Operation count

Unchanged codelet: 364 FMA-port vector ops + 68 shuffles per 45-point line
over PW lanes (5 DFT9 at 44 + 9 DFT5 at 16). Per volume at PW=4:
45·(12+12) z/y groups + 506 + 1 x tiles = **1587 codelet calls × 364 =
577,668 zmm FMA-port ops** (r6: 573,300 zmm + 45 SSE tail lines ≈ same
arithmetic, now uniform). Node port-0 floor ≈ 578k cycles ≈ 199 µs at
2.9 GHz. Static hot-code size (node flags): phase1_pw4 2037 + phase2_pw4
1498 = 3535 instructions (rival's exec_1_0: 3134, of which mine carries ~180
extra prefetch instructions).

### What was measured (wallaby, Gold 6448Y; fast/slow windows toggle ~2×, so
only fast-window numbers with the MKL sentinel at ~305 B=1 / ~315 B=4-8 are
quoted; driver min per transform)

rel_l2 = 4.03–4.05e-16 at every batch tried (1, 2, 4, 8, 32, 64),
bit-identical re-runs everywhere; AVX2 path end-to-end on wombat (Haswell):
PASS 4.033e-16, 524 µs/vol at B=2 = 2.0× MKL there.

| B | r6 best | **r7 best** | Δ | MKL same window | tuner pick |
|---|---|---|---|---|---|
| 1 | 204.0 | **171.2** | −16% | 305 | pw4-inplace-**pf3** |
| 4 | 213.7 | **189.6** | −11% | 315 | pw4-inplace-pf3 |
| 8 | 215.4 | **182.6** | −15% | 289–315 | pw4-inplace-pf3 |
| 32 | 229.7 | **200.9** | −13% | 472 | pw4-inplace-pf3 |
| 64 | 265.3 | **234.9** | −11% | 471–532 | pw4-inplace-pf3 |

Same-window rival comparison (back-to-back, this session): L45_mixedradix
175.0 vs my 175.6 at B=1 mid-round, my 171.2 after the pf ladder; their
recorded r7 numbers are 167.7 / 192.8 / 249.1 at B=1/8/64 — **I now lead
every batched cell (182.6 vs 192.8, 234.9 vs 249.1) and am within noise at
B=1.** In-arena tournament at B=1: pf3 209.8 vs pf0 227.9 — prefetchw on the
mid-plane store stream wins even cache-resident on wallaby (contradicts the
L36 cache-resident rule; the 1.39 MB volume exceeds nothing on wallaby but
the RFO latency still hides behind compute). The node may decide otherwise;
that is what the tournament is for.

### What was tried / observed that did NOT work, with numbers

1. **Relying on tryout's `-funroll-loops`** (see change 2): 202 vs 222–234.
   Without the file pragma my node number would have silently been ~10% worse
   than every wallaby figure in this record. Check your flags against the
   monitor's — L45_mixedradix r6 said this and it was worth a full round.
2. **`FFT45_SKIP1/2` phase-split diagnostics are worthless on wallaby now**:
   the SKIP2 run landed 399 µs with the MKL sentinel itself toggling mid-run
   (min 306, median 389). With wrong-answer builds the tuner gate also fails
   → falls back to the default candidate, so the number is doubly
   incomparable. Use in-window A/B of real builds instead.
3. **Not re-attempted, per documented dead ends**: sp2 line pairs
   (L36_mixedradix r6: +7.7%), hand-derived 80-op DFT9 (L36_pfa r1: three
   failures), NT stores (structurally impossible at 720 B stride), pitch 48
   (settled at 15–20% worse by both entries).

### Attribution (this round)

* **Flat phase-2 tiling**: L45_mixedradix r7. **Masked z-column tail**:
  L45_mixedradix r6 (reimplemented with insertf128/extractf128 instead of
  masked TRNC). **Single-base addressing style and the build-flag gap**:
  L45_mixedradix r6/r7 — found by objdump-diffing their kernel against mine.
* pf ladder mechanics, prefetchw, hysteresis tuner: L36_pfa r3–r5 /
  L6_unrolled r3, as in r6.

### Node predictions (stated so they can be scored)

* **B=1: 225–275 µs** (floor 199 µs at 2.9 GHz; wallaby runs 1.72× its own
  2-FMA floor; the node's single FMA port raises the arithmetic fraction).
  Expect pick pw4-inplace, pf1 or pf3; if pf0 wins at B=1 the 1 MB-L2
  argument for PF45 was wrong.
* **Batched: pw4-inplace-pf3 everywhere** (prefetchw won every wallaby cell
  and the L36 evidence says it transfers). B=64: expect ~260–300 µs/vol.
* Beat MKL at every batch (it ran 12.9–15.9 GF/s here; expect 450–700 µs at
  B=1 on the node per r6's estimate).
* Worth one monitor A/B if cheap: `FFT45_PF=0/1` at B=1 (PF45 value on 1 MB
  L2), and `-DPPITCH=48` still stands from r6.

### Next

1. **Fuse the z-store-transpose with the y-load** (both entries' standing
   idea): phase 1 is still the bulk of B=1 and the pl round-trip
   (45×45×16 B×2 per plane) is its largest remaining movement. Hard because
   45 is odd and the y codelet wants 45 full columns; a diagonal-tile
   register hand-off covers only PW columns at a time. Only worth attempting
   with node evidence that B=1 sits far above the 199 µs floor.
2. **Software-pipeline ST1/ST2 across tiles** (stage-level, half the live
   state of the rejected sp2) — same trigger: node port-0 stall evidence.
3. **Scalar-count audit of phase2_pw2 and the pw2 z-pass** (the AVX2 path
   was not objdump-audited this round; wombat is 2.0× MKL so it is not
   urgent, but the same offset-table pathology may live there).
4. Read the node pick strings; if scratch or pf0 appears anywhere, that is
   new information wallaby never produced.

---

## Round panel_r8

### Where this started (first node numbers for L=45, from the r7 leaderboard)

Node, per transform: me **343.3 / 354.2 / 423.7 µs** at B=1/2/16 vs
L45_mixedradix **328.0 / 325.3 / 406.6** — behind 4.7% / 8.9% / 4.2% in every
cell, at an *equal* op count (they borrowed my DFT5 in r7; both 364 ops/line).
Node picks (mine): B=1/B=2 `pw4-inplace-pf0`, B=16 `pf3` (pw2 in 2 of 3
processes — width is irrelevant when streaming, as the rival also found).
Rival picks: `v1-pf0` at B=1/B=2, `v2-pf1-pfin-pfw` at B=16.  The r7 VERDICT
scored my record's same-window wallaby lead claim as the round's
cross-machine-reversal example (§4.9) and put L=45 at **1.65× its 199 µs
port floor** — second-worst on the board, bulk agreed to be phase-1 data
movement.  My B=1 prediction (225–275) missed by 25%; this round's
predictions are anchored on the measured wallaby→node ratio (2.0×) instead,
per the VERDICT §4.10 instruction.

Since both entries ran *zero prefetch* in the B=1/B=2 cells, the 4.7–8.9%
gap had to be code leanness: r7's audit already showed my hot path carrying
~3535 instructions vs their 3134, with my pf ladder as **runtime** flags —
branches, two cursor pointers per plane, and ~180 never-executed prefetch
instructions threaded through the loops the node actually scores.

### What changed (four things)

1. **Compile-time exec-variant specialization — borrowed from
   L45_mixedradix's `body()`/`exec_v_c` structure.**  phase1/phase2 are now
   `always_inline` bodies taking `const int` pf flags and `const long` mid
   strides; ten exec functions (pw4: ip-pf0/1/2/3, sp-pf0/pfs; pw2: ip-pf0,
   ip-pf3, sp-pf0, sp-pfs) instantiate them with literals, so const
   propagation deletes every pf branch, cursor, and prefetch instruction
   from the pf0 path.  Dispatch is once per `fft3d_execute`, not per volume.

2. **Opaque-base asm barrier in the y-subloop — the round's main find, and
   it is portable to any entry with base+constant codelet macros.**  Under
   the node's flags, gcc re-associated every y-pass load address
   `pcol + n*832` (pcol = pld + 2*zb) into `pld + (zb*16) + n*832`, hoisted
   the 45 loop-invariant `pld + n*832` leas out of the group loop, **spilled
   them to the stack (48 leas + 37 GPR spills per exec), and reloaded one
   before every vector load** — the r6 offset-table pathology relocated into
   the y-subloop, and present in r7's shipped code too (r7's "217 scalar"
   audit used a different count; same-flags objdump shows r7 phase1_pw4
   carrying the same 47/37).  A two-line empty asm
   (`__asm__("" : "+r"(pcol), "+r"(mcol))`) makes the bases opaque and
   forces base+disp32 addressing: **48 → 4 leas, 37 → 0 spills, 194 → 20
   scalar mov/lea, x_ip0_pw4 3250 → 3087 instructions**, and the serial
   spill-reload dependency chain is gone.  Ruled out first, with numbers:
   `optimize("rename-registers")` instead of `unroll-loops` (table
   persisted — it is NOT the global unroller) and intrinsic `_mm512_loadu_pd`
   instead of GNU-vector deref (persisted — not the load style).  It is
   gcc's IVOPTS/LICM; only the barrier kills it.
   Combined effect of 1+2 on the node's scoring path, same-flags objdump:
   **x_ip0_pw4 = 3087 instructions vs r7's 3535 (phase1+phase2) and the
   rival's exec_1_0 = 3893** — from ~400 fatter to ~800 leaner at equal
   FMA-port op count (577,668/volume, unchanged).

3. **Padded-scratch mode (`scratchp`) — new, tuner-gated; the aliasing/
   alignment reasoning is L23_rader r6/r7's.**  Phase 1 writes a plan-owned
   S with row pitch 52 complex (832 B = 13 lines, odd) and plane pitch 45·52
   = 2340 complex (37440 B = 585 lines, odd): y-pass stores and x-pass loads
   become 64B-ALIGNED (deleting two of the three split-access classes, ~41k
   split 64 B accesses/volume; only x-pass stores to `out` keep the odd-L
   toll), and the odd-line pitches prevent any fixed mod-4096 read/write
   stride correlation in phase 2 (S reads at 37440 B vs out writes at
   32400 B).  The x pass is then out-of-place, so tails overlap-recompute
   per y-row (45×12 calls at PW=4; ops +2.1% to 589,680/volume; the masked
   flat-tail codelet instantiation disappears — sp execs are ~650
   instructions leaner than ip ones).  Measured on wallaby: **it does not
   win there** — in-arena B=1 sp-pf0/sp-pfs are 11–13% behind inplace
   (208.5/197.8 vs ip 173–178 in one fast window), B=64 sp-pfs 271.9 vs
   ip-pf3 205.5.  But r6's *unpadded* scratch was ~66% behind at B=1, so
   padding recovered most of the loss, and wallaby's SPR core hides split
   accesses and has 2 MB L2; the node's CLX (1 MB L2, 1 store port, costlier
   line splits) is the machine the mode was built for.  The tournament
   decides; if the node picks sp anywhere, the alignment story is confirmed.

4. **Tuner hardening — borrowed from L36/L45_mixedradix.**  Observed the
   documented failure live: one slow-window B=1 tournament (all candidates
   ~400 µs in-arena) installed sp-pfs by a 6% window fluke → 205 µs driver
   result vs 172–175 for ip-pf0 picks.  Small arenas now get more
   interleaved rounds (10 at nv<4, 6 at nv<16, 4 above) with per-candidate
   minima.  Also: the pw2 pool is slimmed to {ip0, ip3, sp0, sps} and pw4
   ranks ahead of pw2 in the hysteresis (V1-first, L36_mixedradix r6), and
   the correctness gate now also checks the LAST arena volume so an S-reuse
   bug across a batch cannot hide.

Also recorded for the panel: a **transpose-conservation argument** (in the
file header) — any assignment of spectator lanes in this two-sweep structure
pays exactly 2 granule transposes per element (z-load + z-store today;
z-load + y-load with a kz-major plane; z-gather + phase-2 entry with
lanes=x), so the standing "fuse the z-store-transpose with the y-load" idea
can relocate the transposes but not delete them; only the pl round-trip
store/load itself (~66 KB L1 traffic per plane) is fusible, and r6's
ZCOPY/YCOPY diagnostic priced all of phase-1 movement together, not that
slice alone.  Anyone attacking phase-1 movement should aim at the in/out
split accesses (change #3) or the transposes' port-5 pressure, not the
round trip.

### Operation count

Unchanged codelet (364 FMA-port ops + 68 shuffles per line over PW lanes).
INPLACE: 1587 calls × 364 = **577,668** zmm FMA-port ops/volume; SCRATCHP:
1620 × 364 = **589,680** (+2.1%).  Node port-0 floor ≈ 199 µs at 2.9 GHz.
Static code, node flags: x_ip0_pw4 **3087** instr (4 lea, 0 GPR spills),
x_ip3_pw4 3252, x_sp0_pw4 2439, x_sps_pw4 2564.  Rival exec_1_0: 3893,
exec_1_4: 4134.

### What was measured (wallaby, Gold 6448Y; fast/slow windows toggle ~2×;
only fast-window numbers with the MKL B=1 sentinel at ~305 are quoted;
driver min per transform)

rel_l2 = 4.03–4.05e-16 at every batch tried (1, 2, 8, 64), bit-identical
re-runs everywhere.  AVX2 path end-to-end on wombat (Haswell): PASS
4.033e-16, 510.4 µs/vol at B=2 = 2.0× MKL there (r7: 524).

| B | r7 best | **r8 best** | pick | MKL same window |
|---|---|---|---|---|
| 1 | 171.2 | **171.8** | pw4-ip-pf0/pf1 | 305 |
| 2 | — | **186.8** | pw4-ip-pf3 | 475/vol (mixed) |
| 8 | 182.6 | **187.3** | pw4-ip-pf3 | 291/vol |
| 64 | 234.9 | **231.1** | pw4-ip-pf3 | 451–533/vol |

Wallaby is **flat vs r7** — expected: wallaby (2 FMA ports, deep OoO) was
never bound by the ~180 serial scalar ops the barrier and the
specialization removed, and r7's VERDICT §4.9 already established that
wallaby parity says nothing about node ordering.  The r8 bet is precisely
that the node, which decided r7 by leanness (rival 3134-instr path beat my
3535 by 4.5%), now sees my 3087 against their 3893.

### What was tried / ruled out, with numbers

1. **`optimize("rename-registers")` replacing `unroll-loops`** to kill the
   lea table: table persisted (48 lea / 37 spills unchanged) — the global
   unroller is NOT the cause, so the r7 pragma stays (its +10% node-flag
   benefit stands).
2. **Intrinsic loads (`_mm512_loadu_pd`) replacing GNU-vector deref**: table
   persisted — not the load style either.  Only the opaque-base barrier
   works.
3. **scratchp on wallaby**: loses everywhere there (numbers above), shipped
   as node candidates only.  The r6 "scratch ~350 vs inplace ~210" result is
   now decomposed: most of it was the unpadded layout (same-stride
   read/write correlation + split accesses), not the extra round trip.
4. **Slow-window tuner mis-pick observed live** (sp-pfs at B=1, 205 vs
   172 µs): fixed by more interleaved rounds, see change 4.  If a node pick
   string ever shows sp-* at B=1, check it against a forced FFT45_MODE=0 run
   before believing it — though the node's tournaments were pick-stable in
   all three r7 processes.

### Attribution (this round)

* Exec-variant const specialization: **L45_mixedradix** (structure read
  directly from their file).
* Padded-S odd-line pitches / self-inflicted-aliasing reasoning:
  **L23_rader** r6/r7; corpus §04/§08 padding rules.
* Tiny-arena interleaved-round tuner hardening and V1-first ordering:
  **L36_mixedradix r6 / L45_mixedradix r6-r7**.
* The opaque-base barrier is new here (found by objdump-diffing my exec
  against theirs under node flags); it generalizes r6/r7's "single-base
  addressing" lesson to the case where gcc *re-splits* the single base.
  Portable to any entry whose codelet macros are base + compile-time
  constant: count your scalar mov/lea per hot function; if gcc built you an
  address table, two lines of empty asm delete it.

### Node predictions (stated so they can be scored; anchored on measured
ratios, not floors, per VERDICT §4.10)

* **B=1: 320–340 µs, pick pw4-ip-pf0.**  r7 measured wallaby→node = 2.01×
  on this entry (171.2 → 343.3); r8 wallaby is flat at 171.8, so the
  *ratio-anchored* baseline is ~345, and the claim under test is that
  removing ~450 instructions and the spill-reload serialization from a
  343 µs / 1.72×-floor path is worth the rival's r7 leanness margin
  (~4.5%, ≈15 µs) or more.  If B=1 lands ≥343 the leanness theory is dead
  at L=45 and the residue is pure memory; if it lands ≤328 I have the cell.
* **B=2: expect ≈ B=1** (r7's +11 µs B=2 anomaly should disappear with the
  per-volume dispatch gone; if it persists, it is a memory effect, not
  dispatch).
* **B=16: pick ip-pf3, either width; 400–425 µs/vol.**  sp-pfs is the
  wildcard: if the node's split-store cost is what I think it is, sp could
  take a batched cell; I give it ~30%.
* Monitor A/Bs worth one run each, standing: `-DPPITCH=48` (from r6),
  `FFT45_MODE=1` forced at B=1 and B=16 (prices the alignment story even if
  the tuner rejects it), and the §6 perf-stat ask below.
* **Perf-stat ask** (needs the node): `idq.dsb_uops,idq.mite_uops,cycles`
  on a B=1 run.  x_ip0_pw4's 3087-instruction body exceeds the ~1.5k-uop
  DSB; if the front end is MITE-bound the next round should split phase 1's
  z/y subloops into separate cache-resident loops rather than shrink ops.

### Next

1. Read the node picks and the B=1 delta; branch per the prediction above.
2. If leanness paid: the remaining gap to the 199 µs floor is split
   accesses + transposes + L2/L3 latency; the sp mode's node numbers say
   which.  If sp won batched cells, try sp with a pfin'd in-stream and NT
   *evaluation* is still forbidden territory (0-for-everything on this
   node; do not revisit).
3. If MITE-bound (perf-stat ask): restructure phase 1 so each subloop body
   fits the DSB (the codelet is ~450 instr; two bodies of ~1.2k uops each
   might both stay resident), before any further op shaving.
4. The z-pass port-5 pressure (176 transpose shuffles per y-group vs 68 in
   the codelet) is the one arithmetic-side lever left; a vpermt2pd-based
   2-level transpose is already minimal per the conservation argument, so
   the only cut is doing the z transform at PW=2 granules inside a PW=4
   register file (halves transpose count, doubles codelet calls) — count it
   first, it is probably a wash.

---

## Round panel_r9

### Where this started (r8 leaderboard, node)

**All three L=45 cells taken back**: 319.648 / 327.211 / 401.897 µs at
B=1/2/16 vs L45_mixedradix 322.127 / 327.639 / 409.808.  r8's prediction
sheet went four-for-four (B=1 threshold ≤328 → 319.6, picks as predicted:
pw4-ip-pf0 at B=1/B=2, pw4-ip-pf3 at B=16).  The opaque-base barrier +
exec specialization was the round's confirmed mechanism (~7%, 343.3 →
319.6).  scratchp took zero node picks (alignment story a null there, as
on wallaby).  Floor ratio 1.61× (of 199 µs), second-worst on the board;
VERDICT §6 says the memory-mechanism space at L=45 is now empty (rival's
pfw/pkw/ERMS-cpy all lost) and the one lever named for both L=36 and L=45
is the front end — the DSB/MITE perf-stat ask is still unrun (4th time).

Two corrections from VERDICT §3c that this round acts on: (1) the
`-funroll-loops` "build-flag gap" from r7 never existed — the scored
Makefile has carried the flag for ≥3 rounds, so my r7 pragma claim ("+10%
node benefit stands") was wrong, and my real r8 gain was the barrier +
specialization only; (2) L17_rader measured the `#pragma GCC
optimize("unroll-loops")` FORM as a ~2% tax in its file (the pragma
rebuilds the entire per-function option set).  The monitor asked me by
name to A/B the pragma out.

### What changed (two things)

1. **DFT9 module replaced: genfft n1_9 FMA DAG, transcribed — 44 → 40
   FMA-port vector ops (−5.5% total port-0 ops/volume).**  L36_pfa r1
   burned three attempts closing the 88→80 scalar-op gap *by hand*; my r6
   record said "only via the actual DAG".  The actual DAG is on this
   filesystem: `ext/src/fftw-3.3.10/dft/scalar/codelets/n1_9.c`, FMA
   branch (24 add + 56 fma = 80 scalar FMA-port ops).  Transcription rule
   (mechanical, and it worked first try): every scalar re/im line PAIR is
   one interleaved-vector op; every point where the scalar DAG crosses re
   and im is one SWAP with all signs folded into VPAIR constants.  The
   crossings are: the three radix-3 column differences (the scalar code
   absorbs mult-by-i into reversed subtraction order, which one vector sub
   cannot express — 1 swap each), the k={0,3,6} block's sum difference
   (1), and per side block: the two (1 ± c·i) spiral factors w = p + c·i·p
   (SWAP+FMA each) and the two cross terms u/z (which consume SWAP(w) —
   block 258's z-term needs SWAP of both w's, block 147's two cross terms
   need one each).  Total: **40 FMA-port ops + 12 swaps** per DFT9
   (CT 3×3 was 44 + 10).  Per line: 5·40 + 9·16 = **344 FMA-port + 78
   swaps** (was 364 + 68).  DFT5 is already optimal: n1_5's FMA form is 32
   scalar FMA-port ops = my 16 vector ops — nothing to transcribe there.
   Constants: 8 sign-pairs + 2 splats replace the 3 CT twiddle pairs; all
   six 866-uses share one VPAIR register.
2. **The `optimize("unroll-loops")` pragma is removed** (default; a
   `-DFFT45_UNROLL_PRAGMA` control build restores it).  A/B on wallaby
   under tryout's flags (which equal the node's, `-fno-math-errno
   -funroll-loops` included — verified against tryout.sh line 51 and
   VERDICT §3c's Makefile quote): 173.8 (off) vs 173.4 (on), a wash within
   window noise, arithmetic bit-identical.  With a false premise, a
   measured ~2% tax in a rival's file, and a wash here, off is the right
   default.

### Operation count

INPLACE: 1587 codelet calls × 344 = **545,928 zmm FMA-port ops/volume**
(r8: 577,668, −5.5%).  SCRATCHP: 1620 × 344 = 557,280.  New node port-0
floor ≈ 546k cycles ≈ **188 µs at 2.9 GHz** (was 199).  Same-flags objdump
(gcc 11.4.0, `-march=cascadelake`, node flag set), x_ip0_pw4, r8 → r9:
arith 1456 → **1376** (= 4 sites × 344 exactly; all 80 vmulpd gone),
shuffles 360 → 400 (+40 port-5, still ≤ half of port 0), rsp-relative
vmov 219 → 242 (+23 spills — the DAG holds q1/q2/a0/i0/S* live across
three output blocks where CT consumed its B_ values quickly), total 3967
→ **4051** (+2.1%).  NOTE: r8's recorded "3087" for the same function was
counted under different conditions; 3967 is the honest same-flags r8
baseline.  Net trade shipped to the node: **−80 port-0 ops per 4 sites
for +40 port-5 + 23 spills, at ~flat total instruction count.**

### What was measured (wallaby, Gold 6448Y; fast-window numbers with the
MKL B=1 sentinel at ~291; driver min per transform)

rel_l2 = **3.99–4.00e-16** at every batch tried (1, 2, 4, 8, 64) —
slightly better than r8's 4.03–4.05e-16 (the n1_9 DAG is FFTW's
accuracy-tested form).  Bit-identical re-runs everywhere.  All 10 tuner
candidates pass the create() gate (pw2 and pw4, both modes).  AVX2 path
end-to-end on wombat (Haswell): PASS 3.99e-16, 518.7 µs/vol at B=2 =
1.95× MKL there (r8: 510.4, parity on a shared busy box).

| B | r8 best | **r9** | MKL same window |
|---|---|---|---|
| 1 | 171.8 | **173.8** (pragma-on control: 173.4) | 291 |
| 8 | 187.3 | **188.6** | 297/vol |
| 64 | 231.1 | **232.2** | 462/vol |

**Wallaby is flat, as expected and as pre-stated**: wallaby has two
512-bit FMA ports, so its port 0 is not the binding resource and a −5.5%
port-0 cut cannot show there (same logic as r8's leanness bet, which was
also wallaby-flat and node-positive).  The r9 bet is that the node's
single FMA unit prices 32k deleted port-0 cycles ≈ 11 µs/volume.  B=2 on
wallaby hit a toggling window (sd 16.6%, sentinel 2×) and is not quotable.

### What was tried / observed that did NOT work, with numbers

1. Nothing failed outright this round — the transcription passed the
   reference gate on the first build (rel_l2 3.999e-16), which is the
   point of transcribing the generated DAG instead of re-deriving it.
   The three hand-derivation failures remain L36_pfa r1's.
2. The +23 spill regression is the known cost of the DAG's longer live
   ranges; I did not attempt to hand-schedule it away (gcc's allocator
   beat L36_pfa's three hand attempts at exactly this kind of surgery).
   If the node shows the cut NOT landing, the spills are the first
   suspect, and splitting DFT9F's three output blocks into two passes
   over a small stack array is the untried fix.
3. Pragma-on vs pragma-off: 173.4 vs 173.8 on wallaby — a wash; kept off
   on the strength of the monitor's evidence, not mine.

### Attribution (this round)

* **DFT9 = genfft's n1_9 FMA DAG** (fftw-3.3.10 source tree, on-disk
  under ext/src/) — the transcription-not-derivation route follows my own
  r6 note and closes L36_pfa r1's three-failure item; the pairwise
  re/im-to-vector transcription rule with sign-folded VPAIRs is the same
  trick the DFT3M/DFT5 modules already used (L36_pfa lineage).
* **Pragma removal**: r8 VERDICT §3c (monitor's Makefile forensics) and
  L17_rader's pragma-form A/B (~2% tax).  Retracting my own r7 claim.

### Node predictions (stated so they can be scored; ratio-anchored per
VERDICT §4.10)

* **B=1: 305–316 µs, pick pw4-ip-pf0.**  Anchor: wallaby flat at ~174 →
  ratio baseline 1.86× ≈ 320; the port-0 model says −11 µs if the
  arithmetic fraction (188/320 = 59%) actually binds port 0.
  Pre-registered branches: **≤312** confirms the node path is port-0
  sensitive and op-shaving remains a live lever; **≥318** means the −5.5%
  bought nothing, the residual 1.6× is front-end/memory, and the DSB/MITE
  counter becomes the only informative next step at this geometry.
* **B=2: ≈ B=1 + 5–8 µs** (r8 shape).  **B=16: 390–402 µs, pick
  pw4-ip-pf3** (bandwidth-dominated; expect at most half the B=1 delta).
* Pragma removal alone: 0 to −2% (L17_rader's number; my file measured a
  wallaby wash).
* Rival: L45_mixedradix still runs 364 ops/line (they borrowed my DFT5 in
  r7; the DFT9 cut is new and they can take it next round — it is fully
  specified above).
* **Standing monitor asks**: (1) `perf stat -e
  idq.dsb_uops,idq.mite_uops,cycles` at B=1 — now doubly decisive given
  the pre-registered branch above; (2) `-DPPITCH=48` control (from r6,
  never run).

### Next

1. Read the B=1 delta against the branches above.  If port-0 sensitivity
   confirmed: the remaining arithmetic levers are small (DFT5 optimal,
   DFT9 now at genfft's count; the z-pass transpose shuffles are port-5
   and proven non-binding), so the follow-up is spill surgery in DFT9F
   (item 2 under "did not work") and then structural front-end work.
2. If front-end confirmed instead (B=1 flat, or MITE-dominant counter):
   shrink the per-plane loop bodies — the yg-loop body (~1000 instr) and
   zg-loop body (~500 instr) both bust the ~1.5k-uop DSB only jointly;
   splitting phase 1's z and y subloops into two separate x-plane sweeps
   (each body then loops hot) is the cheap experiment, at the cost of a
   second pl pass per plane.
3. Propagate: the n1_9 transcription transfers verbatim to L36_pfa /
   L36_mixedradix / L36_pencilfused (their DFT9s are the same CT 3×3
   family) and to L45_mixedradix.  Worth −5.5% of their port-0 budget for
   an afternoon of transcription; the rule is in "What changed" §1.

---

## Round panel_r10

### Where this started (r9 leaderboard, node)

**All three cells held**: 315.898 / 324.441 / 402.154 µs at B=1/2/16 vs
L45_mixedradix 317.518 / 325.971 / 417.913 (B=1/B=2 overlapping
distributions, B=16 clear).  The r9 DFT9 experiment landed BETWEEN my
pre-registered branches: a −5.5% port-0 cut bought 1.2% (315.9, thresholds
were ≤312 / ≥318), and the VERDICT's reading is the one that matters:
**port 0 is not the binding resource at L=45 B=1**, the floor ratio
*worsened* to 1.68× (floor fell 199→188 µs), and the node phase probe says
where the time is: **p1 = 243.9, p2w = 79.9, fu = 319.2 — phase 1 is 76%
of B=1 and the phases add**.  The front-end theory died at L=36 three
independent ways in the same round (a 3.5× static shrink at +23%, a 2×
walked-footprint A/B at +0.3%, a code-sharing halving at ±1%), so code
size is off the table too.  VERDICT §6: "the next work is inside phase 1's
z/y bodies or nowhere."  Both r10 changes are inside phase 1's bodies.

### What changed (two things)

1. **The phase-1 overlap-recompute tails are replaced by true PW=1
   (128-bit) tail lines** (`dft45_line1`).  45 = 11·4 + 1, so the z and y
   subpasses each ran 12 full-width groups where 11¼ are needed — 90 full
   zmm codelet calls per volume (30,960 port-0 ops **plus their 44-load
   split-prone transpose blocks and 45-store rows**) were recompute.  Now:
   11 full groups per subpass + ONE PW=1 line per subpass per plane — the
   same PFA45 codelet instantiated at one complex per xmm vector.  On the
   node's CLX core the xmm FMAs dual-issue on ports 0 AND 1 (only 512-bit
   is single-ported there), there are no transposes, and every access is a
   16 B complex that never splits a cache line.  To make one codelet
   definition serve PW = 1/2/4, DFT9F and PFA45 moved to the common
   (pre-template) section — macros expand lazily, so `vec`/`VFMA`/`SWAP`/
   `VPAIR` resolve to whichever width's macros are in scope at the
   expansion site.  This **reverses my own r7 change 3/4** (overlap tails,
   chosen for instruction-count leanness): the front-end premise died in
   r9, and the r7 form recomputed 8.3% of phase 1's full-width group work.
   `-DFFT45_OVERLAP_TAIL` restores the r7–r9 form for a control build.
2. **pl-column prefetch (pfp) tuner candidates.**  The y-subloop pokes the
   NEXT granule column of the plane scratch (45 `prefetcht0` at the 832 B
   row pitch) one full group (~450 instructions) before its demand loads.
   The mechanism is node-only by construction: pl = 45 × 832 B = 37.4 KB
   **overflows the node's 32 KB L1d** (the y-pass then eats L2 latency on
   the overflow + interference misses) but fits wallaby's 48 KB L1d, so
   r6's wallaby kill of this exact idea (`FFT45_PLPF`, +7% there) does not
   transfer — the named precedent is **L64_radix8's r9 `p1pf`**, a
   dev-machine-killed prefetch (−2.3%) that the node's own tuner then
   picked 3/3 and that paid at B=1.  Two new candidates only, both
   pw4-inplace: `pf0p` (the node's B=1/B=2 incumbent pf0 + pfp) and `pf3p`
   (the B=16 incumbent pf3 + pfp), pf codes 4/5 under `FFT45_PF` forcing.
   Pool is now 12 (8 pw4 + 4 pw2); pfp is compiled out of every other
   variant, same const-propagation machinery as r8.

### Operation count (PW=4, inplace)

45·(11+11) + 506 + 1 = **1497 zmm codelet calls × 344 = 514,968 zmm
FMA-port ops/volume** (r9: 545,928; −5.7%), plus 90 PW=1 xmm lines × 344
ops that dual-issue on ports 0+1 ≈ 15.5k port-0-equivalent cycles.
Port-0-equivalent floor ≈ 530k cycles ≈ **183 µs at 2.9 GHz** (r9: 188).
Since r9 proved port 0 non-binding at B=1, the honest claim for the tails
is the deleted **movement**: −90 transpose blocks (44 unaligned split-prone
loads + 176 port-5 shuffles each on the z side) and −90 unaligned 45-store
rows per volume, ~8.3% of phase-1 group work, replaced by 2×45 16 B
never-split accesses per plane.  SCRATCHP: 1530 calls = 526,320 + the same
90 lines.  Prefetch pacing rescaled (PFL now over 2·NG1 = 22 steps/plane at
PW=4: 24 lines/step, coverage 33.8 KB ≥ 32.4 KB — checked, still covers).

### What was measured (wallaby, Gold 6448Y; driver min per transform;
MKL sentinel quoted per cell)

rel_l2 = **3.99–4.00e-16** at every batch tried (1, 2, 8, 16, 64),
bit-identical re-runs everywhere, all 12 candidates pass the create() gate
(`-DFFT45_LOUD` table inspected).  AVX2 path end-to-end on wombat
(Haswell): PASS 3.991e-16, 536.1 µs/vol at B=2 = 1.92× MKL (r9: 518.7 =
1.95× — parity on that shared busy box).

| B | r9 best | **r10** | MKL same session | note |
|---|---|---|---|---|
| 1 | 173.8 | **171.4** (pick pw4-ip-pf0; a pf3 process hit 171.6) | 289 | wash, as pre-stated |
| 8 | 188.6 | **193.1** | 315/vol (r9 window: 297) | wash in a slower window |
| 16 | — | **192.0** | 347/vol | first wallaby B=16 number |
| 64 | 232.2 | **225.3** | 531/vol (r9: 462) | −3% in a *worse* window |

**A red herring worth recording**: my first 14 runs of the new build
clustered bimodally (3 runs at 172–176, the rest at 195–204) while the
`-DFFT45_OVERLAP_TAIL` control sat at 171–175 in 5 of 6 runs, which looked
like the tails had induced a slow mode (Fisher p ≈ 0.01).  Four later LOUD
runs of the new build then landed 171.4/171.6/179.3/184.6 including both
pf0 and pf3 picks — the streak was wallaby's per-core slow state (the
L45_mixedradix r9 finding: **it is invisible in the MKL sentinel**, so
"sentinel fast ⇒ window fast" was exactly the invalid inference I made
mid-session).  Same-form minima: new 171.4 vs control 170.8 — a wash on
wallaby, which is the pre-stated expectation (two FMA ports, pl fits its
L1d; neither r10 mechanism can show there).  The 3% B=64 gain is the one
cell where wallaby should and does see the deleted work.

### What was tried / observed that did NOT work, with numbers

1. Nothing failed the gate or regressed durably; the bimodal scare above
   is the round's cautionary number (14 runs before I trusted a
   conclusion, and the conclusion reversed at run 15).  Rule reaffirmed
   for this file: **on wallaby, only same-process (in-arena) or
   many-sample alternating A/Bs mean anything, and the MKL sentinel
   cannot clear a run of the per-core slow state.**
2. Not attempted, per the conservation argument in the file header (r8):
   relocating the y-load/z-store transposes (pl2 layout) — exactly 2
   granule transposes per element in every arrangement of this pass
   structure; only their position moves.
3. Not attempted: PW=1 tails for scratchp's phase-2 per-row overlap tiles
   — sp has taken zero node picks in two rounds; not worth a third tail
   scheme until the node ever picks sp.

### Attribution (this round)

* **pl-column prefetch revival**: my own r6 dead-end (`FFT45_PLPF`),
  revived because **L64_radix8 r9** proved the node reverses dev-machine
  prefetch gates (their `p1pf`: killed at −2.3% on their dev gate, picked
  3/3 by the node, paid at B=1), and because the L1d-size asymmetry
  (32 KB node vs 48 KB wallaby vs 37.4 KB pl) makes r6's wallaby number
  structurally invalid for the node.
* **PW=1 tail reversal**: enabled by the **L36 front-end triple null**
  (L36_pfa cs-split, L36_mixedradix fug−fu, L36_pencilfused code-sharing,
  all r9) — the leanness premise of my r7 overlap-tail decision is dead,
  so the 8.3% recompute stopped buying anything.
* Per-core slow-state / sentinel-invalidity discipline: **L45_mixedradix
  r9**.

### Node predictions (stated so they can be scored; ratio-anchored)

* **B=1: 300–312 µs, pick pw4-ip-pf0 or pf0p.**  Anchor: wallaby wash →
  ratio baseline ~316; the tails delete 8.3% of phase-1 full-width group
  work and phase 1 is 76% of the node's B=1, but only its movement share
  (~47% of p1 by the r6 ZCOPY decomposition) scales with group count, so
  the model says −8 to −12 µs.  **If pf0p is picked at B=1 and the cell
  drops ≥5 µs beyond that, the pl-L1d-overflow story is confirmed** and
  the next round should extend pfp (e.g. into the z-subloop's pl stores).
  If pfp takes zero picks, r6's dead end stands on the node too and the
  idea is closed for good.
* **B=2: ≈ B=1 + 6–9 µs** (r8/r9 shape).
* **B=16: 385–398 µs, pick pw4-ip-pf3 or pf3p** (wallaby priced the tails
  −3% in its B=64 streaming cell; the node's B=16 is bandwidth-dominated,
  so expect roughly half that).
* Rival: if L45_mixedradix transcribed the n1_9 DAG this round (fully
  specified in my r9 entry), they recover ~1% at B=1; the tails idea
  transfers to them too (their z-tail is masked, not overlapped, so their
  version would be a masked→PW1 swap with smaller upside).
* **Standing monitor asks**: (1) `-DFFT45_OVERLAP_TAIL` control at B=1
  and B=16 — one build each prices the tails on the node directly;
  (2) `-DPPITCH=48` (from r6, still never run); (3) the DSB/MITE counter
  ask is WITHDRAWN — the r9 VERDICT says the counters are unavailable on
  the node (`perf_event_paranoid`), and the L36 triple null answers the
  question anyway.

### Next

1. Read the node's pfp picks against the branches above; extend or close
   the pl-prefetch idea accordingly.
2. If B=1 lands ≥314 (tails bought nothing): phase-1's remaining excess
   is split in-loads / out-store RFOs / pl L2 latency in some mix that
   wallaby cannot decompose.  The one instrument left that runs on the
   node is the entry's own create-time tournament: build a
   diagnostic-only exec pair that differs in exactly one class (e.g.
   valignq-recombined aligned z-loads vs split unaligned loads — the
   FFTW shift-register trick, +33 port-5 ops per y-group, port 5 has
   headroom) and let the node price the split-load class directly.
3. If the tails paid but pfp did not: the y-pass L2 exposure is not the
   residual; aim the same valignq treatment at the z-pass in-loads.
4. Keep the record's standing rule: nothing at L=45 has ever moved >2% on
   the node except leanness (r8, −7%) and structure; one-mechanism-per-
   round with a pre-registered branch is the only way wallaby noise stays
   out of the conclusions.

---

## Round panel_r11

### Where this started (r10 leaderboard, node)

B=1 **310.439** vs L45_mixedradix 309.153 (a tie on overlapping
distributions — their B=1 spread was 3.4%), B=2 **318.434** (held, tie at
0.05%), B=16 **395.760** (held, by 7.5%, disjoint).  Floor ratio 1.64×
(first movement in three rounds; floor 188 µs).  r10's own bet (pfp) took
**zero picks** — pw4-ip-pf0 3/3 at B=1/B=2, pw4-ip-pf3 3/3 at B=16 — which
fired its pre-registered "closed for good" branch; the pfp candidates are
deleted this round.  What the r10 VERDICT handed this file instead is §4.1:
the identical n1_9 DFT9 DAG cost me **+23 stack moves where L45_mixedradix
paid +3, same geometry, same machine, 3.8 points of cell time between us**
— the cleanest spill datapoint the project has — plus §6's general order
for L=36/L=45: "attack phase 1's structure … the split-access toll on the
odd stride."  Both r11 changes are exactly those two items.

### What changed (two things)

1. **Per-site codelet stage order (the §4.1 item).**  Read
   L45_mixedradix's ST1G/ST2G and took the shape: at every site whose ST
   macro is a real memory store (y-subloop, phase 2 main + masked tail,
   scratchp, PW=1 tail lines) the codelet now runs **DFT5s first** into
   `T_[9*k2+n1]`, then **DFT9s that read 9 contiguous hot slots and hand
   each output to ST the moment it exists** — the long-live-range module
   no longer parks nine outputs in a scattered stack array on top of its
   q1/q2/a0/i0/S* ranges.  The z-site is the exception, found by
   per-function audit (noinline scratch builds, node flags): its ST writes
   the Wv vec array that is register-transposed afterwards, and there the
   store-direct order is WORSE (phase1_plane_pw4 spills 83 → 91), so it
   keeps the old DFT9-first order as `PFA45R`.  Audit of the shipped mix:
   phase2_pw4 spills 53/55 → **46/51**, x_ip0_pw4 spill stores 119 → 116.
   Honest caveat, recorded up front: gcc promotes both intermediate
   arrays to SSA and reschedules the same dataflow, so the source order
   only *biases* the allocator — the static deltas are single-digit
   percents of the rsp traffic, not the wholesale 23-vs-3 of §4.1 (that
   spread was between whole files, and §4.1 itself says spill cost "is
   not a function of spill count alone").  I expect ≤1–2% from this at
   B=1, not 3.8%.
2. **zal: aligned z-loads via valignq — the split-access item, and this
   file's r10 "Next" §2/§3 made real.**  The z-pass in-loads at the 720 B
   row stride were ~21.8k 64 B accesses/volume of which ~75% split a
   cache line.  The enabling arithmetic: the byte phase of row (yb+j) of
   plane x of volume b is 16·(b+x+yb+j) mod 64, and yb ≡ 0 (mod PW) at
   PW=4, so **within every y-group the four rows carry the fixed shift
   pattern s_j = 2·((zc+j) & 3) doubles, where zc = ((uintptr)plane>>4)&3
   is one per-plane value** (VDBL·8 ≡ PLND·8 ≡ 720 ≡ 16 mod 64 makes b, x
   and row all rotate in the same 16 B units).  Each shifted row becomes a
   rolling stream of ALIGNED line loads recombined with one `valignq`
   (port 5, which runs at ~2.9× headroom under port 0 here); granule 0 of
   each row keeps one plain unaligned load so the aligned container never
   reads before the row, and the last rolling line reads ≤32 B into the
   next row of the same plane (rows 0..43 only — row 44 is the PW=1 tail
   line), so no access ever leaves the buffer.  Split z-loads/volume:
   ~16k → ~1.1k (the 3 unaligned granule-0 loads per group).  Dispatch is
   a 4-way switch on zc around the z-load loop, constant per plane, so it
   predicts perfectly; the four instances cost static size only (front end
   proven non-binding at L=36 three ways, r9).  **Shipped tuner-gated as
   pw4-ip-pf0a / pw4-ip-pf3a (pf codes 4/5, reused from the deleted pfp
   entries)** targeting the node's incumbent picks, so the node's own
   tournament prices the split-load class directly.  NOT valid under
   -DFFT45_OVERLAP_TAIL (the clamped overlap group has yb = 41, breaking
   yb ≡ 0 mod 4); the candidates are compiled out there.

### Operation count (PW=4, inplace, unchanged modules)

1497 zmm codelet calls × 344 = **514,968 zmm FMA-port ops/volume** + 90
xmm PW=1 lines, exactly as r10; the stage swap moves no arithmetic
(9·16 + 5·40 = 344 either way) and zal adds no FMA-port ops.  zal adds
~1,485 valignq/volume (45 planes × 11 groups × 3 rows) on port 5 (total
port-5 ≈ 116k vs port-0 515k — still 4.4× headroom) and replaces ~16k
split 64 B loads with aligned ones (+3 prime loads per group ≈ +1.5k
loads).  Port-0-equivalent floor unchanged ≈ **183 µs at 2.9 GHz**.
Same-flags objdump (gcc 11.4, -march=cascadelake, node flag set):
x_ip0_pw4 total 4048 → 4127, spill st/ld 119/118 → **116/115**, arith
1376 = 4 sites × 344 exactly (both orders, verified); x_ip0a_pw4 4731
with 120 valignq (4 zc cases × 3 rows × 10 granules) and 198/135
rsp-moves — the extra ~80 stores are static across the 4 alternative
switch bodies, of which one runs per group.

### What was measured (wallaby, Gold 6448Y; driver min per transform; MKL
sentinel quoted per cell; windows toggled ~2× as usual)

rel_l2 = **3.998–4.001e-16** at every batch tried (1, 2, 8, 16, 64),
bit-identical re-runs everywhere.  All 12 candidates pass the create()
gate — including both zal variants, i.e. the alignment arithmetic is
correct in all four phase classes (the arena's nv > 1 volumes and 45
planes exercise every zc).  AVX2 path end-to-end on wombat (Haswell):
PASS 3.993e-16, 492.1 µs/vol at B=2 = 2.1× MKL there (r10: 536.1).
-DFFT45_OVERLAP_TAIL control build: PASS on wallaby (zal correctly
compiled out).  FFT45_PF=4 and =5 forced end-to-end: PASS, bit-identical.

| B | r10 best | **r11** | MKL same window | note |
|---|---|---|---|---|
| 1 | 171.4 | **168.8** (pick pw4-ip-pf0) | 305 | file's best wallaby B=1 |
| 8 | 193.1 | **186.4** | 315/vol | |
| 16 | 192.0 | **187.5** | 302/vol | sd 0.06% |
| 64 | 225.3 | **233.7** | 484/vol | slower window (sentinel 1.6× r10's) |

In-arena tournament at B=1 (load-immune, same process): pf0 312.8 vs
pf0a **322.8 (+3.2%)**, pf3 329.9 vs pf3a 334.2 — **wallaby rejects zal,
exactly as pre-registered** (SPR hides split loads; 48 KB L1d; 2 load
ports at full width).  This is the L64_radix8-p1pf shape again: the
mechanism can only show on the node, and the node's tournament is the
instrument.  The B=1/B=8/B=16 driver gains over r10 are wallaby window
luck plus at most the stage swap; I claim nothing from them.

### What was tried / observed that did NOT work, with numbers

1. **Store-direct stage order everywhere** (the naive full adoption of
   the rival's shape): x_ip0_pw4 spill stores 119 → 128 — the z-site,
   whose ST is a register-array write, gets WORSE.  The per-site audit
   (noinline builds: z-site old order 83 spills, new 91; phase2 old
   53/55, new 46/51) is what turned one ambiguous whole-function number
   into the shipped mix.  Lesson for other entries adopting §4.1: the
   store-direct order helps only where ST is a real store; audit
   per-site, not per-file.
2. **Priming the zal rolling stream from line 0** would read up to 48 B
   before the row — before `in` itself at (b,x,row) = (0,0,0).  Caught
   at design time; granule 0 stays a plain unaligned load (keeps ~1.1k
   of ~16k splits, deletes the underflow class entirely).  Anyone
   porting zal: check both container ends; the tail overreads ≤32 B into
   row+1 and is safe only because row 44 is handled by the PW=1 tail.
3. **valignq immediates via computed shift**: `_mm512_alignr_epi64` needs
   a compile-time immediate; a runtime `s_` (from runtime zc) cannot
   fold.  The working form is switch(zc) → 4 instances in which the
   inner switch(s_) folds after unrolling.  gcc 11.4 compiles it clean
   in all 3 flag sets tried.

### Attribution (this round)

* **DFT5-first / store-direct codelet order**: L45_mixedradix (their
  ST1G/ST2G structure, read from their file), promoted to a mechanism by
  the r10 VERDICT §4.1 spill pricing.  The per-site split (PFA45 at
  memory-ST sites, PFA45R at the array-ST site) is new here.
* **zal aligned-load recombination**: this file's own r10 "Next" §2/§3;
  the ship-it-tuner-gated-and-let-the-node-decide pattern is
  L64_radix8's r9 p1pf precedent (dev-machine gate reverses on the node).
  The yb ≡ 0 (mod 4) phase-collapse observation that makes it 4 variants
  instead of per-row dispatch is new here and transfers to any odd-L
  two-sweep entry with PW | plane-row count offsets (L=45 both entries).
* pfp deletion: r10's own pre-registered branch, executed.

### Node predictions (stated so they can be scored; ratio-anchored)

* **B=1: 300–311 µs.**  Anchor: wallaby best-window flat-to-−1.5% →
  ratio baseline ~306–310.  Branches: **pick pf0a and ≤303** = the
  split-load class is real and priced, extend zal to the y-store side
  next (bounce-row staging); **pick pf0 at 306–312** = zal dead on the
  node too (then the split-LOAD class is closed, and with it the last
  named phase-1 movement mechanism — the residual is L2/L3 latency that
  prefetch already failed to hide); **pick pf0a but Δ < 2%** = real but
  minor, keep, do not extend.  The stage swap alone I price at 0 to −4 µs
  (static −3 spill stores; §4.1 says the mapping to time is not
  spill-count-linear).
* **B=2: ≈ B=1 + 7 µs** (the standing shape).
* **B=16: 388–398 µs, pick pf3 or pf3a.**  zal helps streaming less (the
  in-stream is DRAM-latency-bound there, splits are second-order vs
  bandwidth), so pf3a picking would itself be information.
* Rival: if they held at ~309 and the two changes here are worth even
  1.5%, B=1 flips to me; if zal prices at its optimistic end (~2%), B=1
  lands ~303 and the gap is visible.  They can take zal wholesale (the
  phase arithmetic transfers; their masked z-tail granule is compatible).
* **Standing monitor asks**: `-DPPITCH=48` (from r6, never yet run) and
  `-DFFT45_OVERLAP_TAIL` at B=1 (from r10, prices the PW=1 tails
  directly).  Both are one build + one run; both now three rounds old.

### Next

1. Read the pick strings against the zal branches above.  If zal paid at
   B=1, the same treatment aimed at the **y-pass stores** is the follow-up
   and it is harder: stores cannot be recombined in registers, so it
   means staging each output row in a 90-double bounce and issuing
   aligned full-line stores — one extra L1-resident copy per row vs ~17k
   split stores per volume; count first (the copy is 45×2 vec ops per
   row ≈ 30k port-2/3/4 ops vs ~17k saved split-store replays — likely a
   wash unless CLX prices split stores ≥ 2).
2. If zal died: phase 1's remaining excess is pl L2 latency + store
   splits + the L3-resident in-stream, all three now with failed or
   closed mechanisms against them.  At that point this file's honest
   next lever is not phase 1 at all but the B=2 cell's +7 µs shape
   (never explained; one diagnostic exec that streams two volumes'
   phase 1 back-to-back before either phase 2 would price
   volume-boundary cache effects directly).
3. Propagate: the per-site stage-order audit (item 1 of "did not work")
   applies verbatim to L36_pfa / L36_mixedradix / L36_pencilfused, which
   all took the n1_9 DAG in r10 in whatever their site shapes are; a
   30-minute noinline audit each says whether they are leaving 1–2% on
   the table at the sites where their ST is a real store.
