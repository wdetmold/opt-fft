# L23_matrixsimd — strategy record

Entry: dense conjugate-symmetric 23x23 DFT matrix per axis, SIMD across lines.
Geometry: L = 23 (prime), 12167 points, 190.1 KiB per volume (one volume ≈ 6x
the node's 32 KiB L1d, 19% of its 1 MiB L2; a 23x23 plane is 8.3 KiB and
L1-resident).  First implemented in round panel_r6 — there are no earlier
rounds for this entry.

## Round panel_r6 (first implementation)

### Technique

Row-column 3D: one dense length-23 DFT matrix applied along each axis, with
the j <-> 23-j conjugate pair folded first (FFTW's dft-generic form):

    u_j = x_j + x_{23-j},  v_j = x_j - x_{23-j}          (j = 1..11)
    P_k = x_0 + sum_{j=1..11} cos(2pi kj/23) u_j         (k = 0..11)
    R_k =       sum_{j=1..11} sin(2pi kj/23) (-i v_j)    (k = 1..11)
    X_k = P_k + R_k,  X_{23-k} = P_k - R_k,  X_0 = P_0

All coefficients are REAL, so the driver's interleaved complex layout is
already the right SIMD layout (lanes = WC independent lines from a contiguous
run of a free index; a coefficient is a lane-invariant operand; the only
permutes are the re/im swap in (-i v), done as shuffle + XOR on port 5, and
the WCxWC tile transposes feeding axes 2 and 3).  Passes: Y and Z run plane
by plane through a padded 23x24-complex plane buffer with transposing stores;
the X pass runs lanes over the 529 contiguous (ky,kz) points with plain
stores.  A 23-long index is covered by 6 zmm chunks at offsets 0,4,8,12,16,19
(the tail recomputes 1 line, bit-identically); the 529-long X index by 132
zmm chunks + 1 tail at 525 (recomputes 3).

**The architecture is adopted wholesale from L17_matrixsimd** (attribution:
its rounds 1–5 records): pre-splatted coefficient tables, fused transposing
stores, plane-buffer padding, k-blocked kernel option, X-first pass order for
batch, cross-volume software pipelining (its exec18), plane-wise NT-staged
output (its exec20/l17_ntcopy), the licence-clock probe, and the bit-class
tuner discipline.  Everything was re-derived and re-measured at L = 23.

### The one genuinely new idea at this size: pin the WHOLE matrix (22 constants)

The 23x23 DFT matrix has only **11 distinct cosine and 11 distinct sine
magnitudes**: cos(2pi r/23) = cos(2pi (23-r)/23) with NO sign change, and
sin(2pi r/23) = -sin(2pi (23-r)/23), where the sign is a compile-time choice
between vfmadd and vfnmadd (free).  So a fully-unrolled kernel (generated,
253 statements) runs the entire matrix from 22 register-resident constants,
and the ~253 coefficient-table loads per chunk of a rolled kernel disappear
(~298 loads/chunk -> ~90).  This generalises L17_matrixsimd r3's "pinned sine
constants" (which could pin only the 8 sines of its nested kernel) to the
whole dense matrix — at L=17 the same fold gives 8+8 = 16 constants and would
likely pay there too (note for the L17 entries).  No asm-pinning is used: 22
pinned + 12 accumulators exceeds even the EVEX file, so the constants are
plain locals loaded once per chunk and gcc keeps what fits (measured: better
than explicitly parking P in L1, see below).

### Operation count

Per line (594 real flop): 253 vector FMA per chunk = (12 cos accumulators +
11 sin accumulators) x 11 rank-1 updates, + 22 butterfly + 22 combine
add/subs.  Naive complex 23x23 matvec is 4232 flop/line: 7.1x more.
Per volume: 3*529 lines * 594 = 943 kflop (driver yardstick 5 N log2 N = 824
kflop, so reported GF/s ≈ real GF/s).  As vector work: 409 zmm chunks/volume
(2 passes * 23 planes * 6 + 133), 297 FP-port ops each = 121.5k FP-port ops.
Node floor (one 512-bit FMA unit, 2.9 GHz): **41.9 us at B=1**.  Wallaby
floor (two units, ~3.7 GHz): ~16.4 us.

### What was measured (wallaby, Xeon Gold 6448Y, SPR; MKL on same data)

wallaby was intermittently contended during this round (MKL's own B=1 number
swung 143 <-> 279 us between runs); numbers below are best-of-several.

| case | this entry | us/vol | MKL best seen | ratio |
|---|---|---|---|---|
| B=1    | 24.5 us   | 24.5 | 142.7 us | 5.8x |
| B=8    | 227.8 us  | 28.5 | 1150.9 us | 5.1x |
| B=64   | 1824.6 us | 28.5 | 9234.4 us | 5.1x |
| B=256  | 8681.1 us | 33.9 | 38269.0 us | 4.4x |
| B=512  | 17705.2 us| 34.6 | 78576.4 us | 4.4x |
| B=2048 | 75938.4 us| 37.1 | 318.0 ms | 4.2x |

rel L2 error 3.8e-16 in every regime; repeatable (bit-identical across
processes) after the bit-class fix described below.  AVX2-only build verified
on wombat (Haswell): 94 us/vol at B=2, correct, 3.2x MKL there.

Streaming-regime tuner table (nv=403 volumes, >L3), the interesting rows:
512-bit pinned X-last 46.8 / X-first 43.4 / X-first+pipelined 44.2 /
**X-first+NT planes 34.1 (kept)** / X-first+pipelined+NT 35.3.  256-bit
pinned X-first 43.2 — the widths nearly tie in the DRAM-bound regime, as
expected.

### What did NOT work / negative results (with numbers)

* **Rolled table kernels lose to the pinned kernel everywhere.**  512-bit
  unblocked table kernel: 38.1 us at B=1 on wallaby vs 24.5 pinned (-36%).
  In the same tuner table (equal conditions): 52.6 vs 46.8 units X-last
  streaming, 48.8 vs 43.4 X-first streaming.  Cause: ~253 coefficient-table
  loads per chunk saturate the load ports; the FMA ports never see them.
* **k-blocking (register-pressure relief) is not needed at this size on
  AVX-512**: k-blocked 56.7 vs unblocked 52.6 tuner units at B=1 (-7% for
  blocking); k-blocked + L1-parked butterflies worse still (60.0).  The
  unblocked kernel's 23 accumulators fit the EVEX file; do not retry.
* **Explicitly parking P0..P11 in L1 across the R sweep (pc=1) does not beat
  letting gcc spill what it wants**: 59.7 vs 57.8 (B=1), 43.32 vs 43.39
  (streaming) — a wash at best.  Both stay as tuner candidates.
* **The pinned kernel is NOT bit-identical to the table kernels** (cmp on
  full outputs; gcc contracts the two differently) — exactly the failure
  L17_matrixsimd r3 documented.  First submission tripped the panel's
  repeatability check because two processes' tuners picked across the
  boundary.  Fix: only the pinned family is selectable; within it, width,
  P-parking, pipelining and NT staging are cmp-verified bit-identical
  (12 forced-variant cmp runs at B=4 and B=64 on wallaby).  X-last and
  X-first reassociate the triple sum and are separate classes, chosen
  deterministically by batch (<64: X-last, >=64: X-first).
* **Cross-volume pipelining did not add on top of NT staging on wallaby**
  (35.3 vs 34.1) and was ~flat without it (44.2 vs 43.4).  Kept as tuner
  candidates because L17_matrixsimd r4 measured the OPPOSITE interplay on
  the node (NT lost without pipelining, won 14% with it) — the node tuner
  must judge this itself on its 1 MiB L2 / 22 MiB L3.
* **Cross-volume input prefetch (pf) lost at streaming** with the NT winner:
  33.9 (pf=0) vs 36.7 (pf=1) us/vol.  A/B'd at plan time; picked per machine.
* **Mixed-width ymm tail (L17_rader's "512t", via L17_matrixsimd r5):
  evaluated on paper, not built.**  Covering 23 needs 6 zmm, or 5 zmm + 2 ymm
  = 6 equivalents on the node's single 512-bit unit (a tie), and 3.5 vs 3.0
  equivalents on wallaby's two units (a loss).  23 = 4*5+3 lacks L17's lucky
  4*4+1 shape; there is nothing to win.  Do not retry.

### What worked, in order of impact

1. **Pinned 22-constant fully-unrolled kernel**: -36% at B=1 vs the rolled
   dense kernel.
2. **NT-staged plane output** (adopted from L17_matrixsimd exec20): -21% in
   the streaming regime (43.4 -> 34.1 us/vol at nv=403).
3. **X-first pass order at batch** (adopted from L17_matrixsimd r3): -7%
   streaming before NT (46.8 -> 43.4); X-last stays better under batch 64.
4. 512-bit everywhere on wallaby at B=1 (-14% vs 256-bit); near-tie streaming.

### Where this stands / the crossover question this entry exists to answer

The brief asks where dense-matrix loses to Rader at a bigger prime.  This
round's answer so far: the dense approach lands at 24.5 us B=1 on wallaby =
1.5x its FP-port floor there, and ~5.8x faster than MKL; the node floor is
41.9 us.  L23_rader is still a stub, so the crossover cannot be measured yet.
For the comparison when it lands: dense-folded spends 594 flop/line; Rader-23
(length-22 cyclic convolution, = 2 x length-11, each needing Rader-11 or
Winograd-11 again) has a much lower flop count on paper but pays gathers for
the index permutation and needs two levels of prime machinery.  At L=17 the
same fight ended 15.2 vs 17.1 us in dense's favour.

### Next round

1. **Nested cyclic-11 split of the cosine side.**  With j,k reindexed by
   powers of 5 mod 23 (quotient by +-1), the cosine half is a length-11
   CYCLIC correlation and the sine half a length-11 negacyclic one.  Unlike
   L17's length-8 case there are no sign-only splits (x^11-1 = (x-1)*Phi11),
   but the (x-1) factor still peels the DC term, and a Winograd-style
   cyclic-11 by 2x Toom/Karatsuba-ish halves could cut the 253 FMA/chunk
   meaningfully.  Count the adds first — at 512-bit they share the FMA port,
   which is what killed every such split for L17_winograd.
2. **Node feedback**: check which variant the node tuner picked per batch
   cell (description string carries it plus measured clk512/256), especially
   the NT-vs-pipelined interplay, and whether 41.9 us B=1 floor + measured
   wallaby 1.5x-of-floor scheduling overhead predicts the node number
   (~55-60 us expected).
3. **B=1 gap**: still 1.5x above the wallaby FP floor.  Suspects: the X pass
   writes 23 strided streams (rows are 16-mod-64 aligned so half the vector
   stores split cache lines), and the Y-pass loads have 368-byte row stride.
   A line-splitting fix (pad t1's X-pass layout to 64-byte rows) is worth one
   experiment.
4. If the monitor can spare it: `perf stat -e cycles,ref-cycles` on the node
   run to confirm the 2.9 GHz licence clock assumption for this kernel mix
   (the clk512/256 probe in the description string reports it already).

## Round panel_r7 (2026-08-22)

### Standing going in

No node numbers exist yet for L=23 (panel_r6 produced no leaderboard), so the
standing is wallaby-only: statistical parity with L23_rader (23.4–23.6 µs B=1
under r6 conditions), both ~5–6× ahead of MKL.  L23_rader's r6 record settled
the algorithm question by operation count: no realization of the cyclic-11
convolution pair (Winograd/CRT, Karatsuba, FFT-22) beats the conjugate-folded
direct form on hardware where FMA = add = one FP-port cycle.  My r6 "next"
item 1 (nested cyclic-11 split) is therefore DEAD BY COUNTING — I did not
implement it, per the brief's rule about documented dead ends.  This round is
scheduling only; the arithmetic is unchanged at 297 vector FP ops per chunk,
409 zmm chunks/volume, node floor 41.9 µs at 2.9 GHz.

### What changed (five things, four borrowed)

1. **X-first pass order at ALL batch sizes** — adopted from **L23_rader r6**,
   which measured X-first winning B=1 by ~5% on wallaby (scattered X-pass
   stores then go to the hot t1 scratch, not a cold `out`).  This collapses
   the two bit classes of r6 into ONE (pinned X-first family); X-last and the
   table kernels are still timed for the record but are never selectable.
2. **t1 plane stride padded 1058 → 1064 doubles** (`L23_T1P`, 8512 B = 133
   whole cache lines) — my own r6 "next" item 3.  At stride 1058 the X-pass
   stores into t1 cycle 4 alignment classes and 3 of 4 zmm accesses split a
   cache line; at 1064 every X-pass store and every plane-phase reload of t1
   is line-aligned (the one overlapping tail chunk per index stays split).
   The driver-owned `in`/`out` strides cannot be fixed.
3. **Deferred-Z plane schedule** (`dz` variants 34–39) — adopted from
   **L17_matrixsimd r6** (its deferred-Z / `xld`): the plane buffer is
   double-buffered (pb/pb2) and Y(x+1) runs between Y(x) and Z(x), so a Z
   group's loads never issue directly behind its own Y group's stores to the
   same L1 plane buffer (a store→load-forwarding junction once per plane,
   23 per volume).  Pure scheduling: same chunks, same operands, same
   per-value order.
4. **Paced write-intent prefetch (`pw`)** — adopted from the **panel_r5
   VERDICT §4.5** rule ("hide the RFO with prefetchw, do not avoid it with NT
   stores"; NT lost on the node four rounds running) via **L17_matrixsimd
   r6**, originally **L8_fusedaxes r5 / L36_pfa r5**.  All non-NT X-first
   variants can issue `__builtin_prefetch(line, 1, 3)` on the 133 lines of
   the `out` plane the Z group is about to store, in two half-bursts around
   the Y group.  Runtime flag, judged JOINTLY with `pf` as a 2×2 grid on the
   stage-1 winner with a 3% margin against (0,0) (L36_pfa: prefetch on
   resident lines is pure µop tax — never default a prefetch on).
5. **Per-candidate licence warmup in the tuner** — adopted from **L17_rader
   r6**: every candidate is warmed for ≥1.5 ms before being timed, so a ymm
   candidate is never measured inside a predecessor's not-yet-decayed AVX-512
   licence window (~670 µs).  Protects the 256-bit candidates' ranking on the
   node, where clk256 (sparse) reads 3.89 vs clk512 2.89 GHz.

### Operation count

Unchanged from r6: 594 real flop per line, 943 kflop per volume, 297 vector
FP ops per chunk, 409 zmm chunks per volume.  Deferred-Z adds zero
instructions (one extra 8.6 KiB L1 plane buffer); padding adds nothing; pw
adds ~133 prefetchw µops per plane when selected.

### What was measured (wallaby, Xeon Gold 6448Y; best of the quiet windows)

| case | r7 | r6 | delta |
|---|---|---|---|
| B=1   | **21.32 µs**      | 24.5 µs  | −13% |
| B=8   | 25.11 µs/vol      | 28.5     | −12% |
| B=64  | 25.63 µs/vol      | 28.5     | −10% |
| B=512 | 31.99 µs/vol      | 34.6     | −7.5% |

rel L2 3.8e-16 everywhere; bit-repeatable; AVX2 host (wombat) PASS at
81.4 µs/vol B=2 (was 94).  For cross-entry reference: L23_rader's r6 wallaby
numbers were B=1 23.42, B=8 29.7, B=64 28.6 µs/vol.

Tuner picks on wallaby: **B=1 kept "512-bit pinned X-first pipelined
deferred-Z" with pw=1** — the pw A/B at B=1 read 22.21 (pw=1) vs 23.94
(pw=0), −7.2%, mirroring L17_rader's r6 wallaby pfw result at B=1.  In the
same (noisy) B=1 table the deferred-Z variants sat at 24.1–24.7 against
43–45 for their non-deferred twins; the table was contended (early candidates
drifted 2×), so treat the end-to-end 21.3 µs and the A/B pairs as the signal,
not the raw table.  **Streaming (nv=256, >L3): kept "512-bit pinned X-first
NT planes" at 30.77 µs/vol**, deferred-Z+NT 31.80, plain deferred-Z 40.0 —
deferred-Z buys nothing once the plane phase is DRAM-bound, exactly as
L17_matrixsimd r6 found at its B=256.  pf=0 kept at streaming; pw correctly
not explored under an NT winner.

### Bit-class verification (the r6 lesson, redone because every call site
was recompiled)

Single class = pinned X-first family, 18 selectable variants.  Forced-variant
cmp on full outputs, wallaby: 13 variants at B=4 (8, 9, 13, 23, 28, 30, 32,
and all six new 34–39) all IDENTICAL; spot re-check at B=64 (8, 34, 35, 36,
38) all IDENTICAL.  Each forced variant also PASSed check.py independently.
X-last and table kernels are outside the class and unreachable by the tuner.

### What did NOT work / caveats (with numbers)

* **Deferred-Z at streaming batch on wallaby: worse or flat.**  Plain dz 40.0
  vs plain 36.3–38.4 µs/vol; dz+NT 31.8 vs NT 30.8.  The junction it removes
  is already hidden under DRAM misses there.  It stays selectable because the
  node's cache-resident cells (B=1, B=8) are where it should pay, and the
  node tuner decides per cell.
* **256-bit did not win anywhere on wallaby even with the licence-warmup fix**
  (best 256-bit B=1: pipelined dz 30.9 vs 512-bit 24.1).  On the node the
  1-FMA-unit/2-half-width-port balance plus clk256=3.89 GHz could differ —
  the candidates are ranked fairly there now, which is all this round can do.
* **The B=1 tuner table on a loaded wallaby drifts ~2× across the sweep**;
  min-of-3 with 1.5 ms warmups was not enough to make the raw table
  comparable across candidates far apart in time.  The within-class pick was
  still right (verified end-to-end); do not read cross-candidate deltas from
  a single contended table.

### Where this stands

wallaby B=1 is now 21.3 µs ≈ 1.37× the ~15.6 µs two-FMA-unit wallaby floor
(was 1.5×).  Scaling the same scheduling ratio to the node's 41.9 µs
one-FMA-unit floor predicts **~53–58 µs at B=1** on the node; streaming cells
depend on whether the node keeps NT (wallaby's pick) or flips to pw as it did
at L=8/L=36 in panel_r5 — both are in the tournament, plus deferred-Z for the
resident cells.  First node numbers for L=23 arrive with this round's
leaderboard.

### Next

1. **Node feedback first**: which variant per cell (description string
   carries it plus pf/pw and clk512/256), does pw beat NT at batch on the
   1 MB L2 / 22 MB L3 node, does deferred-Z hold at B=1/B=8, and does the
   ~53–58 µs B=1 prediction land.  If B=1 sits well above it, ask the
   monitor for `perf stat -e cycles,ref-cycles` on the B=1 cell.
2. **Y-pass load alignment**: the plane-phase Y loads from t1 stride 46
   doubles (368 B), so half the vector loads split lines and are unfixable by
   plane padding alone.  A row-padded t1 (rows 23→24 complex) would fix it
   but breaks the flat-529 X pass; the X pass would need per-row chunking
   (138 vs 133 chunks, +3.8% X-pass work) — worth one forced experiment if
   the node shows the plane phase load-bound.
3. **If the node picks pw**: try pacing variants (per-chunk vs half-bursts,
   one plane ahead) — L36_pfa's record says pacing granularity mattered.
4. The arithmetic is settled (L23_rader r6's counting).  If the panel wants
   a genuinely new experiment at a prime, L=13 (p−1 = 4·3) is the
   interpolating case — noted for the panel, not for this entry.

## Round panel_r8 (2026-08-22)

### Standing going in (first node numbers, r7 leaderboard + verdict)

Node: **B=1 47.717 µs (cell best, 1.14× the 41.9 µs port floor — the
tightest floor ratio on the board), B=4 49.440, B=128 65.593** — a
statistical tie with L23_rader in all three cells (47.854 / 49.232 /
65.197; rader holds B=4 and B=128 on the minima).  My r7 B=1 prediction
(53–58 µs) was beaten from the right side.  Two verdict findings define
this round:

1. **Not promoted.**  The verdict (§3, §7) found L23_matrixsimd ≡
   L23_rader bit-identically at every batch size — L=23 is one algorithm
   implemented twice — and promoted rader, in part because **my entry was
   "the board's worst timed-≠-checked offender": all three cells reported
   a time from a tuner pick that run 3 (the checked run) did not make.**
   The picks differed across processes because 18 near-tied variants were
   ranked by min-of-noise.  My cmp discipline means the numbers are almost
   certainly right, but the harness could not prove them.  That is a
   protocol defect in MY plan code, and fixing it is worth more than a
   kernel this round.
2. The verdict's remaining kernel lever for L=23: "row-padded t1 to fix
   the Y-pass load splits, at +1.2% volume FP — one tuner-gated
   experiment."  (My own r7 next-item 2.)

### What changed (protocol first, one gated kernel experiment)

1. **Deterministic tuner: canonical-order hysteresis.**  The selectable
   set (now 26) is listed in a canonical preference order (512-bit cached
   family first — the node's r7 pick family — then 512-bit NT, then
   256-bit); walking the list, a candidate displaces the incumbent only if
   it beats it by **>2%**.  Near-ties now resolve to the same variant in
   every process; a real mechanism still wins (wallaby's NT-at-streaming
   is +13%, far above the margin).  Verified: 4/4 independent plan
   creations at B=1 picked the same variant on a *contended* wallaby
   (r7's tuner flipped constantly under the same conditions).
2. **Two-sweep ranking + selectables only** (adopted from **L17_rader
   r6**): two full fixed-order sweeps with per-candidate minima, and the
   14 never-selectable variants (X-last, table kernels) are no longer
   timed at plan time at all — they exist for forced experiments only.
   Cuts tuner licence churn and plan time roughly in half.
3. **za variants (the verdict's named experiment), tuner-gated.**  The
   z-extent of t1 is padded 23 → 24 complex per row (row stride 48
   doubles, plane 1104 doubles = 138 lines, 640 mod 4096 — decorrelated
   from the driver's 272): **every plane-phase Y load from t1 becomes
   64-byte aligned** (flat layout: ~3/4 split a cache line, ~2400 split
   loads/volume).  Cost: the X pass goes row by row — 138 chunks instead
   of 133 (+1.2% volume FP).  Construction detail: rows y<22 use exactly
   WC-aligned offsets {0,4,8,12,16,20}; lane 23 of the last chunk reads
   input element (y+1,0) — real and in-bounds — and writes a pad column
   that flows only into plane-buffer row 23, which the Z pass never
   reads.  Row y=22 keeps the overlapping offsets so nothing reads past
   the volume, and its pad element is never written (buffer zeroed at
   create, stays finite even after other variants scribble t1 — vector
   ops are lanewise, so the garbage lane cannot contaminate lanes 20–22).
   Eight variants (plain/pipelined/deferred-Z/NT × width), candidates
   40–47.
4. **Two checked nulls from L45_pfa r7's findings** (so nobody re-runs
   them here): (i) the `-funroll-loops` build-flag gap **does not apply
   to this entry** — same-window alternating A/B with node flags: unroll
   {21.91, 21.49, 22.23} vs no-unroll {21.73, 21.82, 21.17} µs at B=1, a
   wash with bit-identical outputs (hot kernels are hand-unrolled; the
   rolled loops carry an intentional `unroll 1`).  My first cross-window
   reading said "6.3% gap" — that was wallaby's clock-state swing, not
   the flag.  No pragma shipped.  (ii) Scalar-instruction audit: 142
   scalar instructions in the whole hot exec (loop control + addressing),
   1482 vector; no offset-table materialisation.  Nothing to fix.

### Operation count

Unchanged in the canonical variants: 594 real flop/line, 943 kflop/volume,
297 vector FP ops per chunk, 409 zmm chunks.  za variants: 414 chunks
(+1.2%); node FP floor for za ≈ 42.4 µs vs 41.9 flat.

### Bit-class verification (every call site recompiled, so redone in full)

One class, pinned X-first, now 26 selectable variants.  Forced-variant cmp
on full outputs, wallaby, every run asserting its own PASS: **all 26 at
B=4 IDENTICAL** to the class representative (v8), including all eight new
za variants; spot re-check at B=64 (8, 13, 35, 40, 41, 42, 43, 45) all
IDENTICAL.  AVX2 host (wombat): PASS at 85.1 µs/vol B=2, repeatable.

### What was measured (wallaby, Xeon Gold 6448Y; contended day, best of
quiet windows; full-tuner end-to-end via tryout.sh)

| case | r8 | r7 | pick |
|---|---|---|---|
| B=1   | **21.03 µs**  | 21.32 | 512-bit pinned X-first (canonical), pw per grid |
| B=8   | 24.87 µs/vol  | 25.11 | canonical |
| B=64  | 24.56 µs/vol  | 25.63 | canonical |
| B=256 | 30.27 µs/vol  | 30.77 | 512-bit pinned X-first NT planes |

rel L2 3.8e-16 everywhere, bit-repeatable.  Differences from r7 are within
wallaby's window-to-window swing — expected: the arithmetic is unchanged
and this round was protocol + one gated experiment.

**za A/B (forced, same-window alternating, pf=pw=0):** B=1 mins over six
rounds: za 21.16 vs flat 21.17; pipelined-za 21.12 vs pipelined 21.18.
Streaming B=256: za 33.59 vs flat 33.77 µs/vol cached; za-NT 29.28 vs NT
29.44.  **A wash everywhere on wallaby** — the +1.2% FP does not show
(two FMA units halve its cost here) and the split-load saving does not
show either.  On the node (one 512-bit FMA unit: +1.2% FP costs the full
1.2%; but split economics also differ) it stays selectable and the 2%
hysteresis means it is picked only on a real win.  Honest expectation:
the node keeps the flat canonical variant and za is a documented null —
the counter (`ld_blocks_partial.address_alias`, verdict §6) would settle
*why*.

### What did NOT work / caveats (with numbers)

* **The unroll-flag "gap" was a clock artifact** — see checked null above.
  Do not trust a cross-window wallaby comparison for a ≤6% effect, even
  with sd < 1% in both runs; the clock state shifts *between* runs.
* **Deferred-Z at B=1 on wallaby no longer shows its r7 win**: quiet-window
  forced mins 21.74 (dz) / 21.41 (dz-za) vs 21.17 (plain).  r7's "24.1 vs
  43–45" table was contended; the node never picked dz at B=1/B=4 either
  (r7 picks: plain and pipelined).  dz stays selectable but behind the 2%
  margin it will likely never be picked — consistent with the L17 story
  (three scheduling attacks rejected on the node, verdict §6).
* **One pick flip under heavy contention remains possible**: in 3 plan
  creations at B=256 on the worst window, NT-za once displaced NT (true
  ~0.5% tie; one bad draw of the incumbent across both sweeps).  Stage 1b
  now takes 3 reps × 2 sweeps.  On the node (0.3% spread) a 2% margin is
  many sigma; the exposure should be closed there.
* wallaby was heavily contended all session (same-variant B=1 mins swung
  21–41 µs between windows); every number above is min-of-alternation or
  a quiet-window tryout, per the r5/r7 measurement lessons.

### Where this stands

Node prediction: **B=1 ≈ 47.7 µs unchanged** (arithmetic identical;
canonical pick = r7's timed pick), B=4 ≈ 49.4, B=128 ≈ 65.6 with pw per
grid — but now **the same variant in every process**, so timed = checked
and the §3a exposure is gone.  za: expected null on the node (picked only
if it wins >2%; wallaby says it will not).  If the panel wants the L=23
slot decided on substance rather than protocol, this round removes the
protocol objection; the algorithms remain identical, which is the
verdict's real point.

### Next

1. **Node feedback**: confirm timed pick == checked pick in all three
   cells (the round's purpose); note whether pw survives at B=128 under
   the deterministic variant.
2. If the monitor runs the outstanding counter
   (`perf stat -e ld_blocks_partial.address_alias,cycles`, verdict item 1)
   the za pair (`-DL23_FORCE=8` vs `40`) is the cleanest split-load A/B on
   the board: identical arithmetic ±5 chunks, both in the shipped binary.
3. The streaming cells (B=128, 65.6 µs vs a ~47–50 µs bandwidth-overlap
   bound) are the only place with real headroom left; the node rejected
   pf, pipelining and NT there in r7, so the next mechanism would have to
   be a genuinely better overlap schedule (e.g. pacing the X pass of
   volume b+1 into the *tail* planes only).  Wallaby cannot rank this
   (different L2/L3 and BW); it needs node A/Bs or the counter first.
4. The arithmetic remains settled (L23_rader r6 counting; verdict §5
   confirms dense wins at 13, 17 and 23).

## Round panel_r9 (2026-08-22)

### Standing going in (r8 leaderboard + verdict)

Node: B=1 48.184 µs / B=4 49.701 / B=128 66.223 — second to L23_rader in all
three cells (47.688 / 49.557 / 64.835), with the B=128 min flagged by the
verdict as 2.2% low against its own other runs (66.2 / 67.7 / 67.9).  The two
entries remain bit-identical at every batch size; L=23 is one algorithm twice,
L23_rader is the promoted arm.  My r8 protocol fix WORKED where it was
deployed: B=1 and B=4 picked the same variant in all 3 processes
(timed = checked, exposure gone) — but **B=128 still flipped across
processes** (pinned+park / pinned / pinned+pw=1), i.e. streaming tuner noise
on the node exceeds the 2% hysteresis margin.  The verdict's instruction for
L=23 (§6): the only cell with headroom is B=128 (1.55× floor), the only thing
that has ever moved it is **L23_rader's joint-grid combination "rp-t1 + pf=2 +
pw=1" (−1.0%, picked in one process of three)**, and the next round should
"make it the incumbent so it is picked 3/3".  za (my r8 experiment) is the
same layout as rader's rp; what I lacked was pf=2 and the joint grid.

### What changed (three things, all borrowed, all streaming-side)

1. **pf=2, in-pass X prefetch** — adopted from **L23_rader r8**: before
   issuing X chunk i, prefetch the 23 source lines of chunk i+4 (one line per
   8464-B-strided x-plane of `in`).  Rationale (rader's): at streaming batch
   `in` misses LLC and the hardware streamer restarts at every 4K boundary of
   all 23 concurrently-read planes.  Wired into the flat and za X passes of
   all non-pipelined X-first variants (the pipelined variants' interleaved X
   chunks ARE the input prefetch — also rader's rule).  The old pf test
   became `pf == 1` exactly.  Changes no bits (prefetch only; cmp-verified
   anyway, below).
2. **Streaming tuner (batch ≥ 64) is now ONE deterministic canonical walk
   over 24 (variant, pf, pw) COMBOS** — the joint-grid discipline from
   **L23_rader r8** (itself L17_matrixsimd r4/r6: interacting knobs must be
   raced as combinations, never knobs-on-the-stage-1-winner), fused with my
   r8 canonical-order hysteresis.  Two policy changes: the list head
   (incumbent) is **za + pf=2 + pw=1**, the node-proven winner, per the
   verdict; and the displacement margin at streaming is **4%** (my r8 B=128
   flips prove node streaming tables are noisier than 2%; every real
   streaming effect measured so far on either L=23 entry is smaller than 4%,
   so the honest policy is: pin the node-proven combo, let only a genuinely
   new mechanism displace it).  pf=1 (cross-volume plane-phase prefetch) is
   excluded from the walk — it lost every streaming grid on both entries
   three rounds running (rader r8 documents the third).  256-bit keeps two
   sanity rows.  Resident cells (batch < 64) keep the r8 two-stage tuner
   unchanged except the knob grid gains pf=2 rows (canonical walk
   {00, 01, 20, 21}, 3% margin, prefetch never defaulted on).
3. **Env overrides `L23_PF` / `L23_PW`** — from **L23_rader r7**: forced
   knob A/Bs without a recompile; applied after tuning/FORCE, baked into the
   description string, never set by the harness.

### Operation count

Unchanged: 594 real flop/line, 943 kflop/volume, 297 vector FP ops/chunk,
409 zmm chunks (414 for za, +1.2%).  pf=2 adds 23 prefetch µops per X chunk
when selected (~3.1k/volume flat, ~3.2k za); pw adds ~133 prefetchw per
plane, as before.

### Bit-class verification (every call site recompiled → redone in full)

One class, pinned X-first.  Forced-variant cmp on full outputs, wallaby:
all 10 combo-walk variants {9,13,23,30,34,40,41,42,43,44} ≡ representative
v8 at B=4; spot re-check {13,30,34,40,42} at B=64 all IDENTICAL.  Knobs: all
five non-zero (pf,pw) cells on v8 and on v40 at B=4, and (2,1) on both at
B=64 — all IDENTICAL.  check.py PASS asserted on the representative at both
batches.  tryout repeatability PASS at B=1, 4, 8, 64, 128, 256.  AVX2 host
(wombat): PASS at 84.1 µs/vol B=2 (r8: 85.1), repeatable.

### What was measured (wallaby, Xeon Gold 6448Y; ordinary-contention day)

| case | r9 | r8 | pick (this round) |
|---|---|---|---|
| B=1   | 21.25 µs      | 21.03 | plain pinned X-first, pf=0 pw=0 |
| B=4   | 22.41 µs/vol  | —     | (resident path, unchanged) |
| B=8   | 23.59 µs/vol  | 24.87 | resident canonical |
| B=64  | 26.12 µs/vol  | 24.56 | za pf=0 pw=0 (walk table: 24.17 vs plain 24.14 — a true tie) |
| B=128 | 25.86 µs/vol  | —     | za pf=2 pw=1, 4/4 processes |
| B=256 | 28.5–29.1 µs/vol | 30.27 | za pf=2 pw=1 (4/4; one earlier process: plain pf=2 pw=1) |

rel L2 3.767e-16 – 3.808e-16 everywhere.  B=256 tuner table (nv=256, one
window): plain pf0pw0 32.83 / plain pw1 28.81 / plain pf2pw1 29.13 / za
pf0pw0 34.11 / za pw1 29.74 / za pf2pw1 30.50 / dz pf2pw1 28.66 / NT 30.44 /
za-NT 30.17 / 256-bit 39.0.  Readings: **pw=1 at streaming is now worth
~10–12% on wallaby** and beats NT outright (28.81 vs 30.44 — r8's wallaby NT
pick is stale); pf=2 is ~1% NEGATIVE on wallaby (SPR's streamer covers it;
the node's grid said the opposite in rader r8 — exactly why the incumbent is
set from node evidence, not wallaby); dz+pf2+pw1 is wallaby's raw best but
by <4%, so it cannot displace — if the node disagrees it must say so by >4%.

### Determinism check (the round's protocol point)

4 independent plan creations at B=128 and at B=256 on wallaby: **8/8 picked
"za, pf=2, pw=1"**.  Caveat honestly recorded: wallaby's za-vs-plain gap
(~4.5% in one window) sits near the 4% margin, and one earlier same-day
process picked plain+pf2+pw1 instead — wallaby can flip across the margin
boundary.  On the node this exposure should close: rader r8's node grid
measured its rp (≡ za) rows within ~1% of plain (incumbent survives) and
node tuner spread is ~0.3%.

### What did NOT work / caveats (with numbers)

* **pf=2 on wallaby at streaming: −1% (loses)** — plain pf2pw1 29.13 vs
  pf0pw1 28.81; za 30.50 vs 29.74.  Kept as the incumbent's knob anyway on
  the node's evidence (rader r8 B=128: the ONE process that gridded pf=2
  picked it and won the cell).  If the node r9 leaderboard shows my B=128 at
  or above r8's 66.2, the combo transferred badly and the incumbent should
  be demoted to (za, 0, 1) next round.
* **B=64 on wallaby reads 26.1 end-to-end vs 24.2 in its own tuner table**
  — window drift between plan time and timing loop, not a pick error (the
  table's plain row read 24.14, same 2 µs off the end-to-end number).  Node
  B=64 is not a scored cell; not chased.
* **The raw-ssh missing-`cd` trap fired for me this round — four times in a
  row on the same command** (documented panel-wide three rounds running; my
  own r7 record warns about it).  A bare `ssh wallaby 'python3
  gen_input.py…'` lands in $HOME and everything downstream 'fails missing'.
  What finally worked, and what I will do from now on: never inline the
  remote command — write a helper script with the cd + env inside and
  `ssh wallaby 'bash /full/path/script.sh'`.
* The FP arithmetic remains settled (L23_rader r6 counting; r8 verdict §4.2:
  dense wins at 13, 17, 23).  No kernel work attempted, per the verdict's
  "algorithmically finished at 1.14× floor".

### Where this stands / node prediction

B=1 ≈ 48.0 and B=4 ≈ 49.6 unchanged (identical pick, identical arithmetic;
deterministic 3/3 again).  B=128: pick will be **za + pf=2 + pw=1 in all
three processes** (that is the round's design goal); expected ≈ **64.5–65.5
µs** — rader's identical-arithmetic combo measured 64.835 in its one r8
process — versus my r8 66.2, and timed = checked for the first time in this
cell.  If it lands there, the B=128 gap to the ~47–50 µs bandwidth-overlap
bound is pure DRAM overlap and the schedule space is close to empty.

### Next

1. **Node feedback**: (a) 3/3 same pick at B=128 (the protocol goal);
   (b) does pf=2 transfer (B=128 ≤ 65.5) or backfire (≥ 66.2 → demote the
   incumbent to (za, 0, 1)); (c) the alias counter ask stands a third round:
   `perf stat -e ld_blocks_partial.address_alias` on `-DL23_FORCE=8` vs `40`
   at B=1 — still the cleanest split-load A/B on the board.
2. If the monitor stays counter-less, adopt **L36_pfa's create()-side
   self-measurement** (verdict §6's recommended default): time forced
   A/B pairs inside fft3d_create() and route the numbers out through the
   description string.  One round of that would settle pf=2-on-node and
   za-vs-plain-on-node without any monitor time.
3. The only untried streaming schedule: pace volume b+1's X pass into the
   TAIL planes only (my r8 next-item 3) — wallaby cannot rank it (different
   L2/L3/BW); build it only if (1b) shows the current overlap is the
   bottleneck rather than raw DRAM bandwidth.
4. If the panel consolidates L=23 to one arm (r8 verdict floated it), this
   entry's transferable assets are the canonical-order hysteresis (already
   lifted into the verdict's recommendation for rader) and the 4%-margin
   streaming policy introduced here.

## Round panel_r10 (2026-08-22)

### Standing going in (r9 leaderboard + verdict)

Node: B=1 47.945 / B=4 49.632 / B=128 65.112 µs — statistical ties with
L23_rader in all three cells (47.795 / 49.524 / 64.882); the verdict calls
L=23 **closed** (1.14× floor at B=1, one algorithm implemented twice) and
keeps L23_rader as the promoted arm.  Two r9 findings drive this round:

1. **My B=128 determinism goal failed and my incumbent was wrong.**  With
   za+pf2+pw1 at the list head, the pick flipped (flat in run 1, za in
   runs 2–3): the node's za-vs-flat gap (~3.4%) straddles my 4% margin from
   the head slot.  Worse, the gap has the OPPOSITE sign to what the r8
   evidence suggested: L23_rader's r9 telemetry shows the node's own arena
   displacing rp (≡ za) with FLAT 3/3, pick=59.54–60.20 vs inc=61.63–62.36
   µs/t — flat is 3.4% FASTER at streaming on the node.  Verdict: "the r8
   B=128 win was the knobs (pf=2, pw=1), not the folded-pair layout."  The
   hysteresis lesson, sharpened: **the head must be the fastest known cell,
   not a hoped-for one** — a wrong head converts real speed differences
   into cross-process flips.
2. The only cell with headroom is B=128 (65 µs vs a ~47–50 µs bandwidth-
   overlap bound); the node has now rejected NT, deferred-Z, pf=1 and
   uniform pipelining there.  The one streaming schedule never raced on the
   node is my own r8 next-item 3: pace volume b+1's X pass into the TAIL
   planes only.

### What changed (all streaming-side; B=1/B=4 untouched and closed)

1. **Streaming incumbent demoted to flat: list head = (v8 pinned X-first,
   pf=2, pw=1)** — the node's r9 pick (rader 3/3, my run 1), za rows moved
   below the flat family.  With flat at the head, za must now be >4%
   *faster* to be picked; the node measured it 3.4% slower, so 3/3 flat
   should follow there.
2. **Tail-paced pipeline (new variant 48, X0=12)** — my r8 next-item 3,
   finally built: cross-volume pipelining (t1 double-buffered) but volume
   b+1's whole X pass is issued during planes 12..22 of volume b's plane
   phase only (~12 chunks per plane over the tail 11 planes; the classic
   pipeline spreads ~6 over all 23).  Rationale: plain (no overlap of the
   next in-read with the plane phase) beats uniform pipelining on the node,
   and v48 is the midpoint of that axis — the in-read overlap is confined
   to the tail, where cur-t1 planes are dying at the rate t1b fills, and
   the front planes' out-store stream runs uninterfered.  Implemented as a
   pacing-window parameter X0 on the existing pipelined macro (X0=0
   reproduces the old schedule exactly).  Walk rows (48,2,1), (48,0,1),
   (48,0,0), gated behind the incumbent by the 4% margin.
3. **pf=2 in-pass X prefetch wired into the pipelined family** (prologue
   and both insertion loops), so pipelined rows race the incumbent with the
   same prefetch help it enjoys; row (13,2,1) added.  Prefetch only —
   changes no bits.
4. **Tuner telemetry in the description string** — adopted from
   **L23_rader r9** (itself **L36_pfa r8**'s in-plan probe pattern):
   `tune[pick=… inc=… tp=… us/t nv=…]` — the picked cell's arena time, the
   canonical head's, and the best tail-paced row's.  Every leaderboard run
   now reports the node's own create()-time numbers for the round's
   experiment, picked or not — this is how the tail-paced question gets a
   node answer without any monitor time.

### Operation count

Unchanged: 594 real flop/line, 943 kflop/volume, 297 vector FP ops/chunk,
409 zmm chunks flat (414 za).  v48 is the flat layout: 409 chunks, zero
extra FP; its only cost is t1 double-buffering (+195 KiB scratch footprint,
same as the uniform pipeline).  pf=2 adds 23 prefetch µops per X chunk when
selected, as before.

### Bit-class verification (every call site recompiled → redone in full)

One class, pinned X-first, now 28 selectable streaming combos over 11
variants + 26 resident selectables.  Forced-variant cmp on full outputs,
wallaby: **{9, 13, 30, 34, 40, 41, 42, 43, 48, 49} ≡ v8 at B=4** (including
both new tail-paced widths); spot re-check {13, 40, 48} at B=64 IDENTICAL;
knob grid {(0,0),(2,0),(0,1),(2,1)} on v8 AND on v48 at B=64 all IDENTICAL.
check.py PASS asserted on v48's outputs at B=4 and B=64.  tryout
repeatability PASS at B=1, 4, 8, 128, 256.  4 independent full-tuner
processes at B=256 produced bit-identical outputs (picks differed — see
below — but every pick is inside the class, so outputs cannot differ).
AVX2 host (wombat): PASS at 85.3 µs/vol B=2, repeatable.

### What was measured (wallaby, Xeon Gold 6448Y; mixed-quality windows)

| case | r10 | r9 | pick |
|---|---|---|---|
| B=1   | 21.72 µs      | 21.25 | flat canonical, pf=0 pw=0 |
| B=4   | 21.75 µs/vol  | 22.41 | flat canonical |
| B=8   | 24.26 µs/vol  | 23.59 | flat + pf=2 pw=1 (knob grid: 21 = 24.01 vs 00 = 27.80) |
| B=128 | 25.50 µs/vol  | 25.86 | flat + pf=2 pw=1 (that process; see flips below) |
| B=256 | 28.24 µs/vol  | 28.5–29.1 | za or flat + pf2 pw1 (process-dependent) |

rel L2 3.767e-16 – 3.808e-16 everywhere.  B=256 walk table (nv=256, one
window): flat pf2pw1 29.29 / flat pw1 29.32 / **za pf2pw1 28.14** / za pw1
28.64 / park pf2pw1 29.22 / dz pf2pw1 29.37 / dz-za pf2pw1 28.60 /
**tail-paced pf2pw1 28.62** / tail-paced pw1 29.15 / uniform-pipe pf2pw1
29.56 / NT 29.52 / za-NT 29.28 / 256-bit 39.1.  Readings: pw=1 is worth
~12% on wallaby streaming and pf=2 is ~neutral (both repeat r9); the
tail-paced schedule beats the uniform pipeline (28.62 vs 29.56) and flat
(29.29) but not za — **on wallaby**.  One B=128 arena read pick=25.43
(flat) with tp=26.14.

### Determinism check, honestly recorded

4 independent plan creations at B=256 on wallaby: **v40 (za) 3/4, v8
(flat) 1/4** — on wallaby za genuinely IS ~4% faster at streaming (DDR5,
2 MB L2: the same machine-inversion r8/r9 documented), so the za-vs-flat
gap straddles the 4% margin HERE just as it straddled it on the node in r9
with the heads reversed.  This is unavoidable with a fixed margin when two
machines order the same two cells oppositely by ~the margin size; the fix
is that the head now matches the SCORING machine's own 3/3 evidence
(rader's r9 node telemetry), so on the node — spread 0.3%, flat 3.4%
ahead — the head should hold 3/3.  Wallaby flips are cosmetic: outputs are
bit-identical across every pick in the class.

### What did NOT work / caveats (with numbers)

* **Tail-paced did not win on wallaby**: 28.62 vs za's 28.14 at B=256,
  tp=26.14 vs pick=25.43 in the one B=128 arena.  But wallaby cannot rank
  node streaming (it inverted za-vs-flat, its B=128 half-fits L3); the
  round's design is that the node's tp= telemetry answers this cell-by-cell
  in every process.  Expectation set in advance: if tp lands within 4% of
  pick, the tail schedule joins NT/dz/pf1/uniform-pipe as a documented
  streaming null and the schedule space is empty.
* **A wallaby tryout window can read +27% on a resident cell**: B=8 read
  30.88 µs/vol in one window and 24.26 in the next with the same binary and
  pick — wallaby's per-core slow state (L45_mixedradix r9: invisible to an
  MKL sentinel).  Do not read a single tryout level as a regression;
  re-run before reacting.
* The raw-ssh missing-`cd` trap fired AGAIN despite my own r9 warning
  (fourth round it has bitten someone on the panel).  The helper-script
  rule works; it is now the only way I run remote commands.

### Where this stands / node prediction

B=1 ≈ 47.9 and B=4 ≈ 49.6 unchanged (identical pick and arithmetic).
B=128: pick = **flat + pf=2 + pw=1 in all three processes** (the round's
protocol goal, now with the head on the node's own evidence), level ≈
**64.5–65.0 µs** (rader measured 64.882 with the identical cell).  The
description string will carry pick/inc/tp for every cell: tp is the round's
one real question — the last untried streaming schedule at this geometry.
If it shows nothing, L=23's honest status is: closed at every cell, both
arms identical, 1.14× floor at B=1, B=128 at the DRAM-overlap wall.

### Next

1. **Read tp= off the r10 leaderboard descriptions.**  tp < pick by >4%:
   promote the window start X0 to a tuned parameter (8/12/16) next round.
   tp ≈ pick: record the streaming schedule space as EXHAUSTED — plain,
   uniform-pipe, tail-paced, deferred-Z, NT, pf1 all raced on the node,
   knobs (pf=2, pw=1) are the only movers — and stop spending rounds here.
2. Confirm 3/3 flat at B=128 (protocol goal).  If za somehow displaces
   flat on the node this round, the r9 telemetry was window luck and the
   4% margin needs a per-machine head table instead — but rader's 3/3 ×
   3.4% makes that unlikely.
3. If the panel consolidates L=23 (verdict has floated it twice), carry to
   the survivor: the fastest-known-head rule, the pick/inc/tp telemetry,
   and the tail-paced result whichever way it lands.

## Round panel_r11 (2026-08-22)

### Standing going in (r10 leaderboard + verdict)

Node: B=1 47.733 / B=4 49.216 / B=128 64.874 µs — all three cells remain
statistical ties with L23_rader (47.469 / 49.678 / 64.793); I hold the B=4
minimum.  Everything this entry set out to prove in r10 landed:

1. **Protocol goal met**: B=128 picked "flat + pf=2 + pw=1" in ALL THREE
   processes (arena pick=59.79–60.90 µs/t), timed = checked everywhere.
   The pick lottery has moved to L23_rader (3 picks in 3 runs); the verdict
   says if rader survives consolidation it needs my fastest-known-head fix.
2. **The tail-paced schedule is dead**: tp=62.28/63.29/63.17 vs
   pick=60.90/59.87/59.79 — 2.3–5.7% slower in all three processes, exactly
   the pre-registered "null" branch.  With that, the verdict (§6) declares
   **the L=23 streaming schedule space EXHAUSTED** (plain, uniform-pipe,
   tail-paced, deferred-Z, NT, pf=1 all raced and rejected on the node;
   knobs pf=2/pw=1 are the only movers) and rules the geometry **closed:
   "stop funding it"** — 1.13× floor at B=1, tightest on the board.

But the same verdict's §5 headline cuts the other way: L17_matrixsimd's
`sbw` probe showed the L=17 batched cells — which had been declared at a
bandwidth wall by the very same schedule-exhaustion reasoning — are in fact
**39% above the machine's own copy speed for their own traffic**, with the
residual localized to the write/overlap side.  L=23's B=128 cell was closed
by the SAME argument (schedules exhausted) WITHOUT the measurement: nobody
has ever measured what this node can actually move at 190.1 KiB volumes.
The verdict also asked for "a second sbw four-tuple" to make the bandwidth
model panel-wide.

### What changed (instrument only; every exec, pick and bit is frozen)

**sbw — in-plan streaming bandwidth decomposition, adopted from
L17_matrixsimd r10** (structure copied verbatim, re-derived at L=23's
strides).  At batch ≥ 64, create() times four pure memory patterns on the
>L3 tuner arena (min of 3 after a discarded warmup rep) and routes them out
through the description string, so the node's own numbers arrive with every
leaderboard run at zero monitor cost:

* `rd`  — sequential zmm read of a volume (24334 doubles, unit stride)
* `wr`  — sequential zmm write (RFO + writeback, the exec's plain stores)
* `cp`  — per-volume read burst then write burst: the X-first exec's own
  phase alternation with the compute deleted (X pass reads a volume, plane
  phase writes one)
* `s23` — the X pass's ACTUAL read pattern: 23 interleaved plane streams,
  one 64 B line per plane per step, planes 8464 B apart (132 columns × 23
  rows of 8-double vector loads)

Reported as `sbw[rd/wr/cp/s23]=…` in µs per 190.1 KiB volume.  Nothing else
changed: no new exec variants (building another schedule would contradict
both the verdict's ruling and my own r10 exhaustion record), no tuner
changes, B=1/B=4 paths byte-identical to r10.

### Operation count

Unchanged: 594 real flop/line, 943 kflop/volume, 297 vector FP ops/chunk,
409 zmm chunks flat (414 za).  The probe adds ~25 ms to create() at B=128
(setup is excluded from scoring) and zero instructions to any exec.

### The accounting the probe will settle (pre-registered)

Compulsory DRAM traffic per volume at streaming batch, no NT: 190.1 KiB in
read + 190.1 out RFO + 190.1 out writeback = 570.3 KiB.  Scaling
L17_matrixsimd's measured node rates (rd 16.3, wr 19.8, cp 17.9 GB/s):

* predicted node rd ≈ 11–13 µs/vol, wr ≈ 18.5–20.5, **cp ≈ 31–34**
  (caveat: wallaby measures cp ≈ 1.35×(rd+wr) at this volume size where
  L17's node measured 1.13× — if the alternation penalty grows with burst
  size on CLX too, cp could reach ~37)
* predicted s23/rd ≲ 1.0 (the node read s17/rd = 0.81–0.83 at L=17; even
  wallaby, which punished L17's interleaved reads at 1.28–1.31, reads only
  1.12 at L=23)

Cell = 64.9, B=1 compute = 47.7.  Zero-overlap bound = 47.7 + cp; perfect-
overlap bound = max(47.7, cp) = 47.7.  **Ledger:**

1. **cp lands 30–37 and s23 ≈ rd** (expected): the cell sits ~17 µs above
   the perfect-overlap bound with only ~45% of its traffic hidden under a
   compute phase long enough to hide all of it.  L=23 is then NOT
   bandwidth-closed — it is schedule-closed with measured overlap headroom,
   the same verdict L=17 just got.  Record it that way; the next mechanism
   is whatever the L=17 write-side experiment (staged output flush paced
   under compute, named in r10 verdict §6) proves out on the node, ported
   here — do not build it before the L=17 arm prices it.
2. **cp lands ≥ ~45**: compute + unavoidable alternation accounts for the
   cell; B=128 is genuinely at this machine's own bound for this traffic —
   the r10 "closed" ruling gets its measured proof, write that and stop.
3. **s23 ≥ 1.3×rd** (not expected): the 23-stream X read shape is the
   recoverable share; a staged-input variant (L17_matrixsimd's staged
   twins) would be the matching fix.

Consistency note for branch 1: NT stores delete a third of the traffic
(the RFO) yet lost on the node four rounds running — the cell is not
limited by traffic VOLUME, which is exactly what "unhidden, not
undersized" predicts.

### What was measured (wallaby, Xeon Gold 6448Y; ordinary windows)

| case | r11 | r10 | notes |
|---|---|---|---|
| B=1   | 21.88 µs      | 21.72 | pick flat canonical, pf=0 pw=0 (unchanged) |
| B=4   | 21.86–22.89 µs/vol | 21.75 | one slow-state window read 33.0 first — re-run before reacting (r10 lesson, again) |
| B=128 | 24.66–25.10 µs/vol | 25.50 | pick flat + pf=0/2 + pw=1 (walk near-ties, head holds) |
| B=256 | 28.2 µs/vol   | 28.24 | pick za pf=0 pw=1 in one window (za-vs-flat straddle, cosmetic — bit-identical class) |

rel L2 3.767e-16 – 3.803e-16 everywhere; repeatable (bit-identical across
runs) at B=1, 4, 64, 128, 256.  AVX2 host (wombat): PASS at 87.7 µs/vol
B=64, repeatable, probe path runs correctly there (generic vectors → 2×ymm).

**sbw on wallaby** (the probe's own first data at this geometry):

* nv=128 (arena 47.5 MiB — FITS wallaby's 60 MB L3; L3-flavored, listed
  only as a caveat): rd=4.67 wr=5.52 cp=10.77 s23=4.81
* nv=256 (arena 95 MiB, genuinely streaming; two runs): rd=4.98/4.98
  wr=6.78/6.63 cp=15.80/15.73 s23=5.60/5.62 → rd 39 GB/s, cp/(rd+wr)=1.35,
  s23/rd=1.12.  Wallaby cell 28.2 vs compute 21.9 + cp 15.8: ~9.4 of 15.8
  µs of traffic hidden — the same "roughly half the traffic unhidden"
  shape expected on the node.  **Caveat carried from r9/r10: the node's
  B=128 arena (l23_tune_nv caps at ~148 volumes there, 2.5×L3) does NOT
  fit its 22 MB L3 — the node's sbw at B=128 is a true streaming number
  even though wallaby's B=128 one is not.**

### What did NOT work / caveats (with numbers)

* Nothing was built to fail this round by design — the r10 verdict closed
  the schedule space and the honest response is to measure, not to build a
  new schedule for the tuner to reject.  The probe is the round.
* The wallaby slow-state window bit AGAIN (B=4 first read 33.0 µs/vol,
  re-runs 21.9–22.9, identical binary and pick) — third round this trap
  appears in my record.  The rule stands: never read one tryout level.
* Description-string length: now ~250 chars with tune[] + sbw[] + clk;
  driver has no cap (raw printf into JSON), clk buffer bumped 352→448 so
  nothing truncates.

### Where this stands / node prediction

Picks and cells should reproduce r10 exactly (identical exec code paths,
identical canonical walks): B=1 ≈ 47.7–48.1, B=4 ≈ 49.2–50.0, B=128 ≈
64.8–65.0 with flat + pf=2 + pw=1 3/3.  The deliverable is the sbw
four-tuple in every batched JSON.  Branch 1 of the ledger is the expected
outcome and would give the panel two geometries' worth of evidence that
"schedule-exhausted" and "bandwidth-closed" are different states, plus the
number (cell − max(compute, cp)) that a future write-side mechanism at
L=23 has to collect against.

### Next

1. **Read the node's sbw four-tuple off the r11 leaderboard descriptions**
   and settle the ledger branch.  If branch 1: L=23's honest status becomes
   "closed pending the L=17 write-side result" — port that mechanism the
   round after it wins at L=17, not before.
2. If the panel consolidates L=23 to one arm, carry to the survivor: the
   fastest-known-head rule + 4% streaming margin (rader's B=128 pick
   flipped 3-ways in r10 while mine held 3/3), the pick/inc/tp telemetry,
   and now the sbw instrument.
3. B=1/B=4 remain closed (1.13× floor, arithmetic settled by L23_rader
   r6's counting; three scheduling attacks rejected).  No kernel work is
   warranted at any cell until the sbw ledger says where the 17 µs lives.
