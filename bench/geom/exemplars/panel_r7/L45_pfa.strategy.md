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
