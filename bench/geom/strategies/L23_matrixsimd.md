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
