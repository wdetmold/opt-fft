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
