# L45_mixedradix — strategy record

Geometry: `L = 45`, cube `45^3 = 91125` complex doubles = 1 458 000 B = 1.39 MiB per
volume. File: `impl/L45_mixedradix.c`. `fft3d_name()` = `L45_mixedradix`.

---

## Round panel_r6 (first round for this geometry)

### Where this started

Nothing existed: L=45 is a new geometry this round and the file was a stub. The stub
suggested Cooley–Tukey 3×3×5 "twiddles kept, no permutation" as the counterpart to
L45_pfa. I did not build that, for a counted reason (below); what I built is the
architecture that has led L=36 since panel_r1.

### Technique

**Borrowed wholesale, with attribution: this is L36_mixedradix's architecture
(row–column, two sweeps, lanes = lines, PFA line codelet, kernel-width × prefetch
tournament in `fft3d_create()`), transplanted to 45 and adapted for oddness.**
Everything below that is not marked as new is theirs.

* **Line transform: Good–Thomas PFA 9×5, zero inter-stage twiddles.**
  With the Ruritanian input map and CRT output map

  ```
  n = (5*n1 + 9*n2) mod 45        n1 in [0,9), n2 in [0,5)
  k = (10*k1 + 36*k2) mod 45      10 = 5*[5^-1 mod 9] (=2), 36 = 9*[9^-1 mod 5] (=4)
  n*k == 5*n1*k1 + 9*n2*k2 (mod 45)   =>   W45^{nk} = W9^{n1 k1} * W5^{n2 k2}
  ```

  so a 45-point DFT is nine 5-point DFTs (over n2, one per n1), then five 9-point
  DFTs (over n1, one per k2), with only a compile-time index permutation between
  them — free, since every pass is already a strided gather/scatter. Both maps were
  verified against numpy in a scalar Python model (2.9e-16) before any C existed.
  The 9-point module is L36's Cooley–Tukey 3×3 verbatim (3 DFT3 + 4 CMUL by W9
  constants + 3 DFT3; 44 FMA-port ops + 10 shuffles). The 5-point module is the
  classic two-rotation form: `t1=x1+x4, t4=x1-x4, t2=x2+x3, t3=x2-x3`, two cos-FMA
  chains for the real parts (`x0 + c1*t1 + c2*t2` and `x0 + c2*t1 + c1*t2`), two
  `-i*sin` chains built with the swap-and-alternating-sign-constant trick
  (`swap(t)*[s,-s,...] = -i*s*t`), then four adds. **18 FMA-port ops + 2 shuffles.**
  Stage-1 outputs land at `T[k2*9 + n1]` so each stage-2 DFT9 reads a contiguous
  9-vector run.

* **The name-vs-technique decision, stated plainly:** the stub's literal suggestion
  (CT 9×5, twiddles kept) was counted and rejected without building it: the
  inter-stage twiddle matrix `W45^{n2*k1}` has 32 nontrivial entries = 32 CMULs =
  +64 FMA-port ops per line (446 vs 382, +17%), plus a twiddle table. This is the
  same verdict L36_mixedradix round 1 reached against CT 6×6 (264 vs 248), one size
  up. At L=36 the "mixedradix vs pfa" distinction that mattered was the *3D
  structure* (row–column line codelets vs 3D index mapping), not the line codelet's
  internals — and row–column won. That is the distinction this entry carries.

* **Two sweeps, lanes = lines, interleaved complex end to end** (L36_mixedradix):
  phase 1 per x-plane does 45 z-lines (in-register PW×PW transposes of 128-bit
  complex lanes on the way in and out, into a plane scratch), then 45 y-lines
  straight into `out`; phase 2 does the x-lines in place in `out`. Every constant
  is lane-invariant (pre-splatted 64-byte `.rodata` rows used as memory operands);
  there is no cross-lane operation anywhere except the z-pass transposes.

* **NEW — odd-L tail discipline (45 = 11·4 + 1, so PW never divides 45):**
  * z pass and y pass are out-of-place (`in → plane`, `plane → out`), so their last
    block **overlaps** the previous one (starts at `45-PW`): the overlapped lines
    are recomputed with bit-identical results and double-written. Costs one extra
    block in 12 at PW=4 — the same instruction count as a masked tail (a masked
    call burns the same 382 vector ops for fewer live lanes), and needs no extra
    codelet instantiation.
  * the x pass is in place, so overlap would re-transform already-transformed
    lines; its tail (the single z=44 column) is a **masked-load/masked-store
    instantiation of the same codelet macros** (`_mm512_maskz_loadu_pd(0x03,·)` /
    `_mm256_maskload_pd`), one call per y with one live lane.
  * the z-pass gather/scatter handles the odd 45th *column* through the same PW×PW
    transpose with masked column loads/stores (only `B[0]` is live on the way in;
    on the way out `TRANSP` of PW copies of `Yv[44]` puts lane j into `B[j]`'s
    lane 0).
  * **NEW — plane scratch padded to a 48-complex row pitch** (45×48 complex =
    33.75 KiB), so every plane access in the z-scatter and y-load is 64-byte
    aligned. Accesses to `in`/`out` rotate alignment mod 64 (stride 720 B ≡ 16),
    which is an odd-L fact of life the driver's layout forces.

* **Memory mechanisms — all tournament-gated candidates, never defaults, all
  borrowed with attribution:**
  * `pf` (phase-2 45-stream prefetch, 1 or 4 lines ahead): L36_mixedradix r1.
  * `pfin` (paced T1 prefetch of the phase-1 input): L36_pfa r3's PFIN via
    L36_mixedradix r4, **reshaped**: instead of a free-running cursor with a
    lines-per-call quota (which cannot pace exactly here — one plane is 506.25
    lines against 24 codelet calls, and the r4-style fixed quota would drift ~6.5 KB
    per volume, unbounded over a batch), the cursor is **positional**: during plane
    x's z-subloop the code prefetches plane x+1's 508 lines in `ceil(508/NB)`-line
    chunks indexed off the loop counter. Exact by construction, crosses volume
    boundaries by itself (`in` is contiguous), cannot drift. The PFNX-style
    re-cover of `in[b+1]`'s first plane (evicted by phase 2's traffic) issues 1
    line per phase-2 call, 508 lines total.
  * `pfw` (paced write-intent prefetch, `__builtin_prefetch(p,1,3)` → `prefetchw`,
    over phase 1's cold-`out` store stream, one plane ahead, spread over the
    y-subloop): L36_pfa r5's PFWMID, ultimately L6_unrolled r3. Same positional
    pacing as pfin.
  * **No NT-store path at all**: the node rejected NT stores in every L=36
    tournament for four consecutive rounds (panel_r5 VERDICT; L36_mixedradix retired
    its NT machinery in r6). I did not build what the node has consistently refused.
  * Tuner: three kernels (V0 = AVX2 2-lane, V1 = AVX-512F 4-lane, V2 = AVX-512VL
    2-lane/32-reg) × mechanism, 1e-13 admission gate against `exec_0_0`, min over
    interleaved rounds, **3% simplest-first hysteresis** (L36_mixedradix r5 /
    L36_pfa r4). Streaming pool (`pf1`, `pfin`, `pfw`) vs cache pool (`pf0/1/4`,
    `pfin`; no pfw — two entries measured prefetchw at +11..17% on cache-resident
    volumes). Arena: 2.5× this machine's L3, clamped [12,64] volumes (the
    "arena must stream" lesson, L36_pfa r2). Overrides read once at plan time:
    `FFT45_PFIN=0|1`, `FFT45_PFW=0|1`, `FFT45_V=0|1|2` (kernel-width filter, new —
    lets the monitor A/B widths in one window), `FFT45_VERBOSE=1` (prints the
    per-candidate tournament times to stderr).

### Operation count

Per 45-point line over PW lanes: `9*DFT5 + 5*DFT9 = 9*18 + 5*44 = 382` FMA-port
vector ops + `9*2 + 5*10 = 68` shuffles. Real flops `9*48 + 5*132 = 1092` per line,
`3 * 45^2 * 1092 = 6 633 900` per volume in 6075 line transforms. The tail
discipline executes 1620 codelet calls per volume against an ideal 1518.75
(45²·3/PW at PW=4), i.e. **+6.7% overlap/masked overhead** — 618 840 FMA-port ops
per volume total. The driver's nominal yardstick is `5*91125*log2(91125) = 7.507`
Mflop, 1.13× the true count.

### What was measured (wallaby, Sapphire Rapids Gold 6448Y, 2 MB L2, 60 MB L3)

Driver min, µs per transform, `-fno-unroll-loops` builds (= the monitor's effective
codegen, see failure #1). Correctness at every batch tried (1, 2, 3, 8, 64):
**rel_l2 = 4.107–4.114e-16**, bit-identical re-runs everywhere.

| batch | regime | best (quiet window) | pick | MKL same session |
|---|---|---|---|---|
| 1 | cached | **172.8** (sd 0.39%) | `v1-pf4` / `v1-pf0` (ties) | 288.4 |
| 8 | cached | **199.4/vol** | v1 cached | 314.9/vol |
| 64 | streaming (187 MB) | **251.0/vol** (sd 0.02%) | `v0-pf1-pfin-pfw` | 464.8/vol |

* Paired kernel-width A/B at B=1 in one window (`FFT45_V=`0/1/2): **V0 253.1,
  V1 176.2, V2 190.1** — the 512-bit kernel wins by 30% on wallaby, same shape as
  L=36 (halved instruction count; T[45] spills far less in 32 zmm).
* Streaming mechanism ladder at B=64, one window, in-arena: `pf1` 398 → `pfin`
  334 (−16%) → `pfin+pfw` **254 (−36% total)**. Both paced-prefetch mechanisms
  transfer from L=36 at full strength. Kernel width is nearly irrelevant there
  (254.1/258.3/254.2 for v0/v1/v2) — memory-bound, as expected.
* The AVX2 V0 path (including both maskload tails) was exercised end to end on the
  Haswell login node: B=2 515.4 µs/vol, PASS 4.113e-16, repeatable — 2.0× MKL's
  1030 µs/vol there.

### What was tried / observed that did NOT work

1. **Relying on the harness's `-funroll-loops`.** tryout.sh adds it; the monitor's
   scored build does not. Without it the rolled z-pass gather/scatter transpose
   loops cost **237 µs vs 182 µs at B=1 (+30%)**. Fix: explicit
   `#pragma GCC unroll` on the gather/scatter/prefetch loops. After the fix the
   no-unroll build is actually the *faster* one (172.8 vs ~182): gcc's global
   unroller was bloating something. **Anyone porting this architecture to a new
   size: check your build flags against the monitor's, the difference is not
   cosmetic.**
2. **CT 9×5 line codelet** — rejected by count, not built: +64 FMA-port ops per
   line (+17%) for the 32 nontrivial inter-stage twiddles. See Technique.
3. **Masked tails where overlap is legal** — rejected by count: a masked call costs
   the same 382 vector ops as a full call, so recompute-overlap is never worse and
   saves the extra codelet instantiations. Only the in-place x pass needs masks.
4. **Wallaby's fast/slow toggle can straddle the create()-time tournament and
   mis-rank kernels.** Observed directly: one auto plan picked `v0-pf0` and scored
   230 µs while `FFT45_V=1` in the same window gave 176. This is a dev-machine
   artifact (the L36 record documents the same toggling; the exclusive node with
   its powersave governor showed stable picks for four rounds), but I widened tiny
   arenas to 10 interleaved rounds so every candidate sees at least one fast
   window. If the node's B=1 pick string ever shows `v0-*`, distrust it and ask for
   an `FFT45_V=0|1` A/B.
5. **Free-running prefetch cursor with a fixed lines-per-call quota** (the r4 L36
   shape) — rejected on paper for 45: 506.25 lines/plane over 24 calls has no
   integer quota; any fixed quota drifts ~6.5 KB/volume, which compounds over a
   256-volume batch to megabytes of mis-aimed prefetch. The positional per-plane
   cursor replaces it exactly. (This is a genuine odd-size trap: every L36-family
   pacing constant assumed divisibility.)

### Predictions for the node (stated so they can be scored)

* B=1: pick `v1-pf0/pf1/pf4` (near-ties). The node has one 512-bit FMA unit, so
  V1's 30% wallaby margin shrinks to whatever the halved instruction count buys
  against V0's port-equal arithmetic — expect V1 by 5–15%. Absolute: the FMA-port
  floor is 618 840 ops ≈ 213 µs at 2.9 GHz; wallaby runs 1.6× its own floor, so
  **~270–330 µs**. MKL's wallaby/node ratio elsewhere suggests MKL lands ~450–550.
* B≥10 streams (2.92 MB/volume against 27.5 MB threshold): picks `*-pf1-pfin-pfw`.
  If the node's demand-RFO arithmetic matches L=36's, pfw is the biggest single
  lever there too.
* Setup ≤ ~2 s at streaming batches (54-volume arena × 9 candidates), well under
  a second at B=1.

### Next

1. **Batch the x-pass tail.** The 45 masked tail calls per volume each burn 382
   ops on one live lane; gathering 4 tail lines (four y's at z=44, lane stride 45
   complex) into full vectors via 128-bit loads + inserts would cut phase-2 call
   count 540 → 519 (~2% of volume ops, minus gather overhead). Small, real,
   fiddly.
2. **sp2 software-pipelined line pairs** (L36_mixedradix r6 added them; their node
   verdict is not in yet): at L=45 a pair needs 90 vector temps, so on 32 zmm it
   may just spill — read their r6 node result before spending a day.
3. **Read the node's pick strings** (plumbed through `fft3d_description()` exactly
   like L36). If B=1 lands `v0-*`, that is failure-mode #4 above — request the
   `FFT45_V` A/B rather than believing it.
4. **If L45_pfa's 3D-mapping entry beats this at any cell**, diff structures: the
   likely differentiators are their lane-filling strategy (the stub flags batch-major
   as "the obvious answer" — at B=1 that answer does not exist) and their handling
   of the same odd-size tails.

---

## Round panel_r7

### Where this started

panel_r6 was halted before its timing phase (stale-runner retirement; see the
repo log), so there are **no node numbers for L=45 yet** — r6's wallaby
standings are the only cross-entry data: this entry 172.8 µs at B=1 / 251.0
µs/vol at B=64, vs L45_pfa's 204.0 / 265.3. This round is three targeted
changes, two of them borrowed, all measured.

### What changed

1. **16-op DFT5 — borrowed from L45_pfa r6 (FFTW n1_5's FMA DAG, from the
   corpus).** Replaced my classic two-rotation 5-point module (18 FMA-port
   ops) with the √5/4 cosine split (`tm = x0 − te/4; tp,tq = tm ± (√5/4)·ta`)
   plus the scaled-sine trick (`s2 = φ⁻¹·s1`, so both −i·sin chains share one
   ALT(s1) constant: `w1 = swap(t4) + φ⁻¹·swap(t3)`, `w2 = φ⁻¹·swap(t4) −
   swap(t3)`, then FMADD/FNMADD into tp/tq). **16 FMA-port ops + 2 shuffles**;
   needed one new macro (VFMSUB) and swapped 4 constant rows. Line codelet:
   382 → **364 FMA-port ops** (−4.7%), same 68 shuffles. Their record proved
   the numerics are fine (4.04e-16); mine confirms (4.10–4.11e-16, unchanged
   from r6 to the second digit).

2. **Phase-2 flat tiling — my own r6 "Next #1", in a cleaner form than
   planned.** The x-pass previously tiled per y (11 full tiles + 1 masked
   tail per y = 45 masked calls/volume, each burning full-codelet ops on one
   live lane). But all 2025 x-lines share the same stride and are contiguous
   in the flat (y,z) index, and the tiles are disjoint in-place, so nothing
   forces tiles to respect y boundaries: tile flat 0..2024 as **506 full
   PW=4 tiles + ONE masked call per volume** (1012+1 at PW=2). Kills 33
   full-width calls per volume vs r6 (540+45 → 506+1 calls in phase 2), no
   gather overhead, *simpler* code than the per-y form — strictly dominates
   the "gather 4 tail lines into full vectors" plan r6 sketched. Per-volume
   codelet calls 1620 → 1587; total **577,668 FMA-port ops/volume** (−6.7%
   vs r6's 618,840); overlap/tail overhead now +4.5% over the 1518.75-call
   ideal (was +6.7%).

3. **Plane-scratch row pitch 48 → 52 complex — borrowed from L45_pfa r6
   (their failed-experiment #1, which was a *finding*: pitch 52 beat 48 by
   15–20% at B=1).** 52 complex = 832 B = 13 cache lines per row;
   gcd(13,64) = 1 so the y-pass column walk (45 rows, stride PPITCH) spreads
   over all 64 L1 sets, where 48 = 12 lines (gcd 4) hits only 16 sets. Rows
   stay 64B-aligned (832 = 13·64). Confirmed on MY structure with a paired
   A/B on wallaby at B=1: **-DPPITCH=48 220.4 µs vs pitch-52 187.7 µs in
   adjacent windows** (MKL sentinel ~289 in both) — same ~15% the rival
   measured. `PPITCH` is now `#ifndef`-guarded so the node can re-check with
   `-DPPITCH=48`; scratch grows 33.75 → 36.6 KiB (still L2-resident on both
   machines).

4. **V1-first tuner ordering — borrowed from L36_mixedradix r6 (their tuner
   hardening after a measured 40% mis-pick under load; my own r6 failure #4
   was the same mode).** Candidate pools now list V1, then V2, then V0, and
   the V0 candidates go through the same probe-and-admit path as the rest
   (exec_0_0 trivially passes its own gate), so the 3% hysteresis now
   protects the kernel that has won every AVX-512 measurement instead of the
   AVX2 fallback. A noisy window can no longer install `v0-*` by coin flip.

### Operation count

364 FMA-port vector ops + 68 shuffles per 45-point line over PW lanes
(9×16 + 5×44 = 364; real flops unchanged at 1092/line, 6,633,900/volume).
Per volume at PW=4: 540 z + 540 y + 507 x codelet calls = 1587 × 364 =
**577,668 FMA-port ops** (r6: 618,840). Node port-0 floor at 2.9 GHz ≈
199 µs/volume (one 512-bit FMA unit).

### What was measured (wallaby, Sapphire Rapids Gold 6448Y, 2 MB L2)

Driver min, µs per transform. Correctness at every batch tried (1, 2, 8,
64): **rel_l2 = 4.096–4.112e-16**, bit-identical re-runs everywhere.

| batch | r6 best | **r7 best** | Δ | MKL same session |
|---|---|---|---|---|
| 1 | 172.8 | **167.7** (sd 0.01%) | −3.0% | 286–305 |
| 8 | 199.4/vol | **192.8/vol** (sd 0.45%) | −3.3% | 292/vol |
| 64 | 251.0/vol | **249.1/vol** (sd 1.8%) | −0.8% | 478/vol |

* The B=1/B=8 gains match the op-count cut (−6.7% ops → −3% time: wallaby
  B=1 is roughly half arithmetic, half data movement, consistent with
  L45_pfa's phase-split diagnosis that phase-1 movement dominates).
* B=64 is memory-bound; the op cut is invisible there, as predicted.
* Pitch A/B at B=1: 48 → 220.4, 52 → 187.7 (adjacent windows, MKL sentinel
  flat) — the rival's finding transfers to this structure at full strength.
* V0/AVX2 path (new DFT5 + flat phase 2 + both masked tails) exercised end
  to end on the Haswell login node: B=2 496.3 µs/vol, PASS 4.096e-16,
  repeatable — improved from r6's 515.4 there, and 1.7× MKL's 855/vol.

### What was tried / decided against, with the reason

1. **sp2 (source-interleaved line pairs)** — NOT built: L36_mixedradix r6
   measured the full-pair form at **+7.7% on wallaby, +5% on Haswell**
   (their failure #1); at L=45 the live state is worse (90 temps vs their
   72). Documented dead end, not rediscovered. Their suggested stage-level
   half-live-state variant remains open (below).
2. **Gathering the x-tail lines into full vectors** (r6's Next #1) —
   superseded before building: flat tiling achieves strictly more (33 of 34
   saved calls) with zero gather shuffles. The remaining single masked call
   per volume is 0.06% of ops; there is nothing left in the tail.
3. Nothing else was attempted and rejected this round; the three changes are
   all keepers.

### Predictions for the node (stated so they can be scored)

* **B=1: 255–310 µs**, pick `v1-pf0/pf1/pf4` (near-ties). Floor is now
  199 µs at 2.9 GHz; wallaby runs 1.9× its own (2-FMA-port) floor, and the
  node's single FMA unit raises the arithmetic fraction, so the op-count
  cut should matter *more* there than the 3% seen on wallaby.
* **B≥10 (streaming)**: pick `*-pf1-pfin-pfw`; the mechanisms are unchanged
  from r6. Expect the pitch-52 effect to persist (same 64-set L1 geometry,
  8-way, on the 5218).
* If any pick string shows `v0-*`, the V1-first ordering failed — but it
  now requires a >3% real deficit, not a noisy window.
* Worth one monitor A/B if cheap: `-DPPITCH=48` vs default at B=1 (the node
  L1 is 32 KB/8-way vs wallaby's 48 KB/12-way; the set-conflict argument
  says 52 still wins, but it is a one-run check).

### Next

1. **Phase-1 movement is now the whole game at B=1** (L45_pfa's ZCOPY/YCOPY
   diagnosis: the transposes, plane round-trip, and split unaligned loads
   dominate). The one structural idea not yet tried at L=45: fuse the
   z-store-transpose with the y-load by keeping PW y-lines' worth of stage-1
   z-output in registers and feeding the y codelet directly for the diagonal
   tile — hard (45 is odd, tiles don't align), probably a rewrite; only
   worth it if the node shows B=1 far above the 199 µs floor.
2. **Stage-level software pipeline** (L36_mixedradix r6's Next #3): overlap
   ST1 of tile t+1 with ST2 of tile t — half the extra live state of the
   rejected sp2. Try only on node evidence of port-0 stalls.
3. **Read the node pick strings and the PPITCH A/B** once the r7 leaderboard
   lands — L=45 has never been measured on the scoring machine, so every r6
   and r7 prediction is still open.

---

## Round panel_r8

### Where this started (first node numbers for L=45)

panel_r7 put this entry **first in all three node cells**: B=1 **328.0 µs**,
B=2 **325.3**, B=16 **406.6 µs/vol** — 1.85×/1.85×/1.67× ahead of MKL, and
promoted. Node picks were perfectly stable across all 9 runs per cell:
**`v1-pf0` at B=1 and B=2, `v2-pf1-pfin-pfw` at B=16.** But B=1 sits at
**1.65× the 199 µs port floor** (second-worst ratio on the board), my r6
prediction band (255–310) missed by 6%, and the verdict's L=45 line says both
entries agree the bulk is data movement. The verdict also flags: "the PF45
poke is pure overhead on the scoring machine."

**The diagnosis that drove this round: the node cells look bandwidth-bound,
not latency- or port-bound.** Traffic model at B=1 (steady state: in + out =
2.9 MB live in the 22 MB L3, out exceeds the 1 MB L2): sweep 1 reads `in`
(1.46 MB) and writes cold `out` (RFO read 1.46 + writeback 1.46); sweep 2
re-reads `out` (1.46, it was evicted to L3 by its own size) and writes it
back (1.46). Total ≈ **7.3 MB of L3 traffic per transform → 22 GB/s at
328 µs**, which is about what one CLX core can pull from L3. Same arithmetic
at B=16 gives ≈ 4.2 MB of DRAM traffic/vol → 10.3 GB/s at 406 µs, near one
core's DRAM limit. If that model is right, op-count tuning is nearly
worthless here, prefetch distance tuning is nearly worthless, and the only
levers are (a) issuing the *same* traffic with better overlap, and (b)
**deleting traffic** — the RFO read of cold `out` is the only deletable term
(1.46 of 7.3 MB at B=1; 1.46 of 4.2 MB DRAM at B=16).

### What changed (no arithmetic change; the codelet is byte-for-byte r7's)

1. **The memory mechanisms are now ORTHOGONAL tournament candidates instead
   of a cumulative ladder.** In r7, pfin and pfw were only offered bundled
   with the phase-2 poke (`pf1`), which the node's verdict declared pure
   overhead — so the node rejected `pf1`, `pf4`, `pf1+pfin` at B=1/B=2 and
   never saw pfin or pfw alone. New exec codes: pf0 / pf1 / pfin / pfw /
   pfin-pfw / pkw / pkw-pfin-pfw / pf1-pfin-pfw / cpy / cpy-pfin (10 codes ×
   3 kernel widths = 30 exec variants; monitor-flag compile is 5 s).
   Rationale for fielding pfw in the CACHED pool, against the L36
   "cache-resident volumes reject prefetchw" rule: that rule was measured on
   volumes that fit the node's L2 (746 KB < 1 MB). At L=45 the 1.39 MB
   volume does NOT fit, so phase 1's out-stores RFO from L3 even at B=1 —
   the mechanism's target exists here. (L45_pfa's r7 wallaby measurement
   pointed the same way; their bundled pf3 was untestable on the node.)
2. **`pkw` — the phase-2 poke re-cast as PREFETCHW.** Phase 2 is an in-place
   read-modify-write of 45 streams; the old T0 poke fetched each line for
   reading and left the store's ownership upgrade unpaid. One prefetchw does
   both jobs. **Borrowed shape: L64_radix8's `pfw` (prefetchw of the output
   rows ahead), which the node picked at L=64 B=2/B=8** — the closest
   geometry analogue (volume ≫ L2, in-place strided pass).
3. **`cpy` — NEW, mine: RFO deletion via ERMS.** The y pass stores into an
   L1-hot plane image (2025 complex, laid out exactly as one x-plane of
   `out`; the plan's scratch allocation grew by 32.4 KB), and one
   **`rep movsb`** per x-plane copies it to `out`. Intel's fast-string
   implementation performs full-cache-line writes **without reading the
   destination for ownership**, and — unlike NT stores, which the node has
   rejected four straight rounds at L=36 and which would force sweep 2 to
   re-read from DRAM — the lines land in the normal cache hierarchy. This
   *deletes* the 1.46 MB/volume RFO term instead of overlapping it. Cost: a
   plane-sized L1/L2 round trip (~2.9 MB/volume of cheap near-cache traffic)
   plus ~45 × ~600 cycles of rep-movsb. Also sidesteps the odd-L split-store
   toll: the codelet's rotating-alignment 64 B stores now hit the L1-hot
   image, and the copy is sequential.
4. **Pool ordering: incumbents first, then singles before combos.** The 3%
   hysteresis resolves within-band ties to the EARLIER candidate, so each
   pool leads with the node's 9/9-stable r7 pick (`v1-pf0` cached,
   `v2-pf1-pfin-pfw` streaming) — a new mechanism must beat the proven one
   by >3% on the node itself to displace it; and singles precede combos
   (measured consequence at B=8 below). `pf4` is retired (node-rejected at
   B=1/B=2 twice, never fielded streaming). Cached pool: v1 × {pf0, cpy,
   pfin, pfw, pkw, cpy-pfin, pfin-pfw, pkw-pfin-pfw} + thin v2/v0 slates
   (13 candidates); streaming: v2 × {pf1-pfin-pfw, cpy-pfin, pkw-pfin-pfw,
   pfin-pfw} + v1/v0 slates (9).
5. **New env overrides for monitor A/Bs**: `FFT45_PF=0|1|w` (phase-2 poke:
   none / T0 / prefetchw), `FFT45_CPY=0|1`, joining `FFT45_V/PFIN/PFW/
   VERBOSE` and `-DPPITCH`. All read once at plan time; execution stays
   repeatable.

### Operation count

Unchanged from r7: 364 FMA-port vector ops + 68 shuffles per 45-point line;
1587 codelet calls/volume at PW=4 = **577,668 zmm FMA-port ops**; port-0
floor ≈ 199 µs at 2.9 GHz. `cpy` adds no FP work — 45 rep-movsb of 32,400 B
each, pure load/store, and only when the tournament installs it.

### What was measured (wallaby, Gold 6448Y; fast/slow windows toggle, so
in-arena candidate tables — one process, interleaved rounds, min — carry the
comparisons; MKL is the window sentinel)

Correctness at every batch tried (1, 2, 8, 16, 64): **rel_l2 =
4.096–4.112e-16**, bit-identical re-runs everywhere. AVX2/V0 path (including
cpy and both masked tails) end-to-end on the Haswell login node: PASS
4.096e-16, 532.7 µs/vol at B=2 = 1.9× MKL there.

Driver mins (node-parity `-fno-unroll-loops` builds), best quiet window:

| batch | r7 best | r8 | pick | note |
|---|---|---|---|---|
| 1 | 167.7 | **166.8** | v1-pf0 | refactor is time-neutral, as intended |
| 2 | — | 182.5/vol | v1-pf0 | 4.5% window |
| 8 | 192.8/vol | **178.6/vol (−7.4%)** | **v1-pfw** | unbundling pays already on wallaby |
| 16 | — | 223.0/vol | v1 cached | |
| 64 | 249.1/vol | 245.2/vol | v2-pf1-pfin-pfw | |

In-arena candidate tables (same process, µs/vol):

* **B=8 (cached pool, out volumes cycle L2):** pf0 201.9 / **pfw 187.1
  (−7.3%)** / pkw-pfin-pfw 188.8 / pfin-pfw 190.7 / pkw 199.9 / cpy 203.0 /
  pfin 205.9. The write-intent prefetch alone is the whole effect; pfin adds
  nothing on top of it here.
* **B=1 (volume fits wallaby's 2 MB L2):** pf0 173.1 / pkw 171.7 / pfw 171.9
  / pfin-pfw 175.6 / cpy 187.8 — everything within or near the 3% band,
  incumbent holds. Expected: on wallaby the mechanism targets barely exist
  at B=1; the node (1 MB L2) is the real test.
* **B=64 (streaming):** incumbent pf1-pfin-pfw 240.0 / pkw-pfin-pfw 239.9 /
  pfin-pfw 241.4 / **cpy-pfin 275.0 (+15%)**.

### What was tried / observed that did NOT work, with numbers

1. **`cpy` on wallaby: loses everywhere** — B=1 187.8 vs 173.1 (+8.5%),
   B=8 203.0 vs 187.1 (+8.5%), B=64 275.0 vs 239.9 (+15%). Understood, and
   why I shipped it anyway: on wallaby the RFO it deletes is nearly free
   (B=1: out fits the 2 MB L2; B=64: SPR's DDR5 gives one core far more
   bandwidth headroom than CLX DDR4), so only the plane-image overhead
   shows. The mechanism's target — RFO fill bandwidth as the binding
   resource — exists only on the node, per the traffic model above. This is
   exactly the L64_radix8 r7 pattern (their pfw lost to NT on wallaby by
   5–37% and they shipped it for the node, which then picked it). If the
   node ALSO rejects cpy in all three cells, the ERMS-no-RFO idea is dead on
   this machine and the record should say so next round.
2. **`-frename-registers` (the mechanism behind L45_pfa's +10% from
   `unroll-loops`)** — A/B/A/B on wallaby at B=1, node-parity flags:
   rename 176.4 vs plain 166.8 (fast window), rename 209.9 vs plain 207.0
   (slow window). Loses both pairings on my code; consistent with my r6
   finding that the global unroller bloats this kernel. Not adopted; their
   pragma is right for their code and wrong for mine — build-flag findings
   do not transfer between entries.
3. **Combos ahead of singles in the pool ordering** — first cut listed
   pfin-pfw before pfw; at B=8 the hysteresis then installed pfin-pfw
   (191.8) over the better pfw (187.6) because 187.6 was not >3% under
   191.8. Reordered singles-first; the pick moved to pfw. Lesson for anyone
   using the hysteresis tuner: within-band ties go to the earlier candidate,
   so list order IS a policy statement, and simpler-single-mechanism should
   outrank combo at equal evidence.
4. **Not re-attempted, per documented dead ends**: NT stores (4 node
   rejections at L=36; also structurally impossible at 720 B stride without
   the bounce buffer cpy now provides — but cpy deliberately uses ERMS, not
   NT, so sweep 2 re-reads from cache); sp2/pipelining (+7.7% L36 r6);
   pf4 (node-rejected 2 cells × 3 runs); scalar-count pathology (audited
   exec_1_0 under monitor flags: ~55 scalar instructions against 3067 vector
   — clean, no runtime-offset regression).

### Attribution

* **pkw**: the prefetchw-on-RMW-stream shape is L64_radix8's `pfw`
  (node-picked at L=64 B=2/B=8); re-aimed at my phase-2 poke.
* **Unbundling rationale**: the r7 VERDICT's "PF45 poke is pure overhead"
  finding plus L45_pfa's r7 wallaby pf3-vs-pf0 in-arena result.
* **cpy (ERMS rep-movsb no-RFO plane copy)**: mine, new this round.
* Everything else (structure, codelets, tuner) unchanged from r7; original
  attributions stand.

### Predictions for the node (anchored on r7's measured ratios, per the
verdict's §5 instruction — not re-derived from floors)

* **B=1/B=2**: if no new mechanism clears the 3% bar, pick stays `v1-pf0`
  and the cells stay ~325–330 (the refactor is time-neutral). If the RFO
  exposure is real, `v1-pfw` or `v1-pkw` installs and B=1 lands **300–320**.
  If ERMS no-RFO genuinely deletes traffic on CLX, `v1-cpy` installs and
  B=1 lands **280–305**. I genuinely do not know which branch fires — that
  is what the tournament is for, and any of the three outcomes closes a
  question.
* **B=16**: pick `v2-*-pfin-pfw` (pkw or pf1 flavour, likely a near-tie);
  ~400–410. If `v2-cpy-pfin` installs instead, the DRAM-RFO model was right
  and it should read **≤380**.
* **Falsification branch, pre-registered**: if the node keeps `v1-pf0` /
  `v2-pf1-pfin-pfw` in all cells AND cpy/pkw/pfw all lose, then the
  bandwidth model is wrong in its actionable part (the traffic is real but
  already optimally overlapped), and the next lever at L=45 is not memory
  mechanisms at all — it is the fuse-z-store-with-y-load rewrite or nothing.
* Worth one monitor A/B each, all one-flag: `-DPPITCH=48` (standing from
  r6/r7), `FFT45_CPY=1` forced end-to-end at B=1 (separates "cpy loses the
  tournament" from "cpy loses by <3%"), and if cheap a
  `perf stat -e offcore_requests.demand_rfo,LLC-load-misses` pair on
  `FFT45_CPY=0/1` — one counter pair settles whether ERMS no-RFO exists on
  CLX for L3-resident destinations, which generalises to every entry with a
  cold store stream.

### Next

1. Read the node picks. Each of the 10 mechanisms now resolves
   independently; whatever survives becomes the r9 baseline and the rest
   get retired from the pools (they cost tournament time).
2. If B=1 stays ≥320 with everything rejected: the remaining ideas are
   structural (z/y fusion around the plane scratch — hard, 45 is odd) or
   nothing; the traffic model says the compulsory 5.8 MB/volume (sans RFO)
   at ~22 GB/s bounds B=1 at ~265 µs regardless.
3. If cpy wins anywhere: try extending it to a half-volume image so phase 2
   also stores through it (deletes the second RFO-equivalent upgrade), and
   sweep `rep movsb` vs 4-plane batched copies.
