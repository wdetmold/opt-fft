# gen_layout — allocation & layout library layer

Scored by ADOPTION. This record doubles as the layer's user manual: if you own a
class entry, the section "How to adopt" is written for you.

## Round gen_r1

### What shipped

`impl/gen_layout.c` was rewritten from the dense stub into two parts:

1. **The library** (top of the file): all-`static`, `gl_`-prefixed, zero link
   footprint. Class owners adopt it with

   ```c
   #define GEN_LAYOUT_LIB_ONLY
   #include "gen_layout.c"        /* impl/ is the include dir */
   ```

   and take only what they need. Compiles clean with `-Wall -Wextra` on both
   the AVX2 login host (wallaby) and the Ice Lake node (a80n0); every SIMD
   utility has a scalar fallback so an adopting entry still builds anywhere.

2. **A demo entry** (below the `GEN_LAYOUT_LIB_ONLY` guard, compiled only when
   this file is the entry TU): a generic dense row-column DFT, any 2 ≤ L ≤ 128,
   split-complex broadcast-FMA staged through pencil-SoA blocks — deliberately
   the same O(L^4) algorithm as `baseline_matrix` so the delta over the floor
   is exactly "vectorization + this layer". It is the layer's living test
   bench and the panel's any-L vectorized floor, NOT a contender.

### The API (what each piece is, and the lesson it encodes)

| function | what | lesson / source |
|---|---|---|
| `gl_map_huge / gl_unmap` | 2 MiB-aligned THP mapping, `MADV_HUGEPAGE`, prefaulted (memset) so first-touch faults land in create(), falls back to `posix_memalign` | warm 00291a90 `prelude_c.py alloc_huge`; L17_winograd ice_r6 THP arenas; LITERATURE §05 §7 |
| `gl_arena_init/take/destroy` | arena on one THP map; each `take` lands on a different mod-4096 phase (4672 B = 73-line rotation, 576 B/step, gcd(9,64)=1 walks all 64 line phases) | warm 00291a90 `alloc_huge_st`; two page-aligned buffers is the 4K store→load alias worst case (LITERATURE §08 §1.8, §10) |
| `gl_arena_take_phase` | take at an EXACT mod-4096 phase | deterministic placement beats hoping |
| `gl_phase4k` | phase of any pointer | inspect the driver's buffers at execute() |
| `gl_far_phase4k` | line-aligned phase maximizing min circular distance to a set of phases | L6_pfa ice: re-place scratch when the driver's buffer-pair residues change |
| `gl_alias_pairs4k` | count of (load row, store row) pairs within the ±64 B 4K window | the collision MODEL: measure, don't guess |
| `gl_best_lineshift4k` | best 0..63-line shift of one stream against fixed streams | generalized from L17_matrixsimd ice_r8 `l17_as_build` (their shipped 256-entry astab) |
| `gl_walk_min_gap4k` | min circular gap of two walking streams over n steps | L23_rader ice_r5 post-mortem: "a deterministic stagger is only as good as the stride analysis behind it" — differing strides can collide on ONE plane no base offset fixes |
| `gl_pad_stride` | round a stride to an ODD line count (rotates all 64 line phases) | but see L17 ice_r8: 73-line padding scored 31 weighted collisions vs ≤4 for 80 lines (1.25 pages) — when you know both strides, use `gl_best_lineshift4k` instead |
| `gl_deint8 / gl_int8` | interleaved↔split for 8 complex, one `vpermt2pd` per output zmm | LITERATURE §04: split layout deletes permutations from the hot loop |
| `gl_tr8x8` | in-register 8×8 double transpose, 24 shuffles | standard unpack/shuffle_f64x2 network |
| `gl_pack8 / gl_unpack8` | 8 interleaved streams (stride in elements) ↔ lane-SoA `[site][2][8]` | gen_batchlane's 8-vol lanes (stride = L^3), pencil SoA (stride = pencil pitch); L8_fusedaxes bl8, L13_rader soa8 |
| `gl_selftest` | one-shot exactness audit of the SoA converters | run it in your create(); it is how I know the shuffle networks are right on your build |

Sanity numbers from the adopter test (both hosts, identical): the L17 geometry
(17×17 rows, strides 5120 vs 4624 B) has 9 alias pairs at shift 0;
`gl_best_lineshift4k` finds shift 5 → 0 pairs. `gl_walk_min_gap4k` on
different-stride walks correctly reports gap 0 (the L23 pathology).

### How to adopt (60 seconds)

```c
#define GEN_LAYOUT_LIB_ONLY
#include "gen_layout.c"

/* in create(): */
gl_arena ar;                                   /* one THP arena per plan      */
gl_arena_init(&ar, total_bytes + n_bufs*4096 + (64<<10));
double *tw  = gl_arena_take(&ar, tw_bytes);    /* staggered automatically     */
double *scr = gl_arena_take(&ar, scr_bytes + 4096 + 64);  /* + placement slack */

/* in execute(), when the driver's (in,out) pair changes: place scratch far
   from their phases (legal iff scratch is fully rewritten before read):     */
unsigned ph[2] = { gl_phase4k(in), gl_phase4k(out) };
double *scr_now = (double *)((char *)scr +
    ((gl_far_phase4k(ph, 2) + 4096u - gl_phase4k(scr)) & 4095u));

/* in destroy(): */ gl_arena_destroy(&ar);
```

If your kernel has two streams with DIFFERENT strides (e.g. padded scratch rows
vs the contract's row pitch), run `gl_walk_min_gap4k` at plan time; if it
reports < 64, pick the padding with `gl_best_lineshift4k`, not by rounding.

### What I measured on the node (a80n0, Ice Lake Xeon Gold 6326, pinned core, graded chain workload, min per transform)

Demo entry, default build (THP arena + stagger + far-phase placement):

| L | B | m | µs/xform | rel L2 | vs baseline_matrix |
|---|---|---|---|---|---|
| 2 | 5 | 1 | 0.098 | 9.5e-17 | |
| 3 | 5 | 1 | 0.241 | 1.9e-16 | |
| 7 | 5 | 1 | 1.871 | 2.6e-16 | |
| 10 | 64 | 1000 | 9.85 | 2.9e-16 | |
| 12 | 64 | 600 | 16.15 | 3.1e-16 | 112.3 → **7.0×** |
| 12 | 1 | 600 | 15.71 | 3.1e-16 | |
| 25 | 16 | 256 | 235.9 | 4.1e-16 | |
| 31 | 16 | 140 | 464.4 | 4.4e-16 | |
| 32 | 8 | 250 | 500.0 | 3.9e-16 | 5757.5 → **11.5×** |
| 32 | 1 | 250 | 478.5 | 3.9e-16 | |
| 50 | 4 | 128 | 2940.4 | 5.3e-16 | |
| 100 | 1 | 64 | 42495.7 | 6.5e-16 | |

Map-chain gate at L=12 B=64 m=600: rel L2 5.10e-14 (honest anchor 3.89e-14,
tol 1e-10) — PASS. Repeatability: bit-identical across runs. Setup ≤ 15 ms at
L=100 (table fill is L² long-double sincos), far under the 60 s budget.
The demo runs ~54–61% of the 2-FMA-pipe peak (real flops, not the 5NlogN
yardstick); it is O(L) more arithmetic than a true FFT, as intended.

### What did NOT show an effect — read this before you cargo-cult layout

A/B knobs on the demo, same node, same core:

- **THP + stagger vs plain posix_memalign** (`-DGL_DEMO_PLAIN=1`), L=100 B=1
  (biggest working set, 4×8 MB split scratch + 32 MB driver buffers):
  42495.7 vs 42642.3 µs — **0.3%, inside the 0.8–2% run spread**.
- **Dynamic placement off** (`-DGL_DEMO_NOPLACE=1`), L=100 B=1: 42319.6 µs —
  a wash.
- **Adversarial collision** (`-DGL_DEMO_COLLIDE=1`, all four scratch buffers
  placed at `out`'s exact page phase), L=32 B=8: 499.9 vs 500.0 µs
  (identical); L=12 B=64: 16.28 vs 16.15 µs (**+0.8%**).

Interpretation, and it matters for adoption honesty: a compute-dense kernel
(0.5 FMA-instruction per cmac, everything staged through an L1-resident pencil
block) hides the 4K hazard almost completely — the staging decouples the load
and store streams that would otherwise alias. The ice campaign's kernels where
placement WAS worth double digits (L23_rader's +27% alias hole, L17's ~8
aliasing pairs per chunk on every volume, L6_pfa's per-pair re-placement) are
5NlogN-class kernels with direct strided load/store interleaving in the hot
loop — which is exactly what the class entries of THIS campaign will be once
they stop being dense stubs. So: the layer's placement tools cost you zero
cycles and one line at execute(); take them as insurance, and VERIFY with the
collision model instead of assuming either way. I will re-run the A/B ladder
on an adopting entry's real kernel the moment one exists — that number, not
the demo's, is the honest price of skipping this layer.

### Harness notes every implementer should know (found this round)

- `tryout.sh` line 36 uses `$W` two lines before it is defined: with `set -u`
  every m>1 case dies with "W: unbound variable". Workaround until the monitor
  fixes it: `export W=/home/lqcd/wdetmold/fft/bench/gen/build/tryout/<name>`
  before calling. The same expansion hole makes the remote `check.py` see
  `--cin '/c.bin'`, so the map-check and the MKL reference lines of tryout
  never run; run the map-check yourself afterwards (shared fs):
  `python3 check.py --input $W/in.bin --output $W/out.bin --L <L> --batch <B>
  --map-check <m> --cin $W/c.bin`.
- `reserve.sh --status` (and therefore tryout's gate) needs slurm on PATH; on
  wallaby: `export PATH=/opt/software/slurm-19.05.8.1/bin:$PATH`. The
  reservation was alive the whole round (monitor heartbeat ~60 s); "job not
  running" from a bare shell is just the missing squeue.
- No sota baselines are built for a80n0 yet (`build/wombat` only), so there is
  no MKL reference number in this record.

### What I tried that did not work (with the number that killed it)

- Trying to demonstrate a THP/stagger/placement win ON THE DEMO at any of
  L ∈ {12, 32, 100}: every knob within noise (numbers above). The demo is the
  wrong vehicle by construction; kept as a test bench, and the null result is
  itself recorded so nobody re-measures layout knobs on a compute-bound dense
  kernel expecting a win.

### Borrowed from other entries

- warm 00291a90 `prelude_c.py`: `alloc_huge` / `alloc_huge_st` (THP map +
  4672 B stagger) — generalized into `gl_map_huge` / `gl_arena_take`.
- L17_matrixsimd ice_r8: the `l17_as_build` collision model and its
  73-vs-80-line stride discovery — generalized into `gl_alias_pairs4k` /
  `gl_best_lineshift4k`.
- L23_rader ice_r4/r5: the mod-4096 phase-walk audit → `gl_walk_min_gap4k`.
- L6_pfa ice: execute-time re-placement when the driver's buffer pair changes
  → the `dm_placement` pattern in the demo, `gl_far_phase4k` in the library.

### What I would do next (gen_r2)

1. Wire the layer into the first class entry that wants it (gen_batchlane's
   lane packing via `gl_pack8`, or any owner's scratch via the arena) and
   re-run the A/B ladder on THEIR kernel — the first real adoption number.
2. Per-volume placement tables (the full L17 astab, not just base far-phase)
   once a memory-bound class kernel exists to test against; the batched driver
   buffers rotate phases by `(16·L³ mod 4096)` per volume, and one static
   phase cannot serve all volumes at every L.
3. A `gl_stream_audit(plan)` convenience that takes the kernel's declared
   streams and prints/returns the full collision report, so class owners can
   assert "0 alias pairs" in create() the way they assert twiddle exactness.
4. Non-temporal store guidance is deliberately ABSENT this round: LITERATURE
   §08 documents Intel's Skylake-Server warning, and no measurement here yet
   justifies a helper. Measure before shipping one.

## Round gen_r2

### What changed

**Library (new, section 4b of the file):**

- `gl_stream_audit4k(streams, n, window, &report)` — the one-call collision
  report promised in r1 item 3: total alias pairs over all unordered stream
  pairs, the worst pair, and the min walking gap (`gl_walk_min_gap4k`) over
  every differing-stride pair. Assert `== 0` in your create() the way
  gen_twiddle has you assert table exactness.
- `gl_pick_pitch4k(row_bytes, nrows, base_phase, fixed, nfixed, window,
  max_extra_lines, &score)` — pick a row/plane pitch by MEASURED collision
  score over candidates `lineround(row_bytes) + 0..max_extra_lines` lines,
  ties to the smallest. This is the L17 ice_r8 73-vs-80-line discovery as an
  API; gen_dense_prime's r1 item 2 (pad 31→32 rows?) is one call now.
- Both compile warning-free under `-Wall -Wextra` in `GEN_LAYOUT_LIB_ONLY`
  mode on AVX-512 and plain x86-64 hosts (verified both, this round).

**Demo entry (the layer's test bench, still O(L⁴)/axis dense class by design):**

1. **Conjugate-pair folded arithmetic, all three axes** — BORROWED from
   gen_dense_prime gen_r1 (transitively ice L13_direct / L17 / L23_matrixsimd):
   u_j = x_j + x_{L−j}, v_j = x_j − x_{L−j}; C = cos-matrix·u, S = sin-matrix·v;
   X_k = C_k − iS_k, X_{L−k} = C_k + iS_k. All constants REAL, so in my
   pencil-SoA form every op stays a broadcast-FMA and the per-pencil FMA count
   drops ~4× (L=31: 900-ish vs 3844 vector FMAs per 8 pencils; general L² + O(L)
   vs 4L²). Even L folds the lone x_{L/2} row as an extra u column whose cos
   weights are ±1, and k=0 / k=L/2 are C-only table rows — one engine, any L,
   both parities (edge cases L=2,3,4 verified against numpy).
2. **Axis-2 rewritten from `rowfast` to a transpose-staged 8-pencil block
   kernel**: stage 8 contiguous split pencils via `gl_tr8x8` (lane = pencil),
   run the SAME folded kernel as axes 0/1 into an output staging block, then
   transpose back + `gl_int8` to interleaved out. The r1 rowfast form was
   load-bound (4 loads / 4 FMAs) and streamed the whole table per SINGLE
   pencil (L·Lp·16 B from L2, ~166 KB/pencil at L=100); now every table
   broadcast feeds 8 pencils.
3. **Owned `fft3d_chain`** — BORROWED wholesale: the volume-major chain (each
   volume runs all m steps cache-hot — corpus consensus via gen_dense_prime /
   gen_rader), and the campaign-standard map ladder (gen_pfa_small's "take it"
   note / gen_dense_prime's s6 shape): pair-compressed |w|² via unpacklo/hi,
   1e-300 guard, vrsqrt14 + 2 Newton for the sqrt, ONE exact vdivpd per 8
   complex (their measurement that div beats an rcp ladder on a standalone
   map pass — not re-litigated). Step 1 reads x0 directly so the fallback's
   per-step memcpy is gone too. Kernel body audit: 1 zmm stack spill in the
   whole file (objdump discipline, ice L17/L23) — not pressure-bound, so the
   SCHED15-style attribute was not tried (gen_batchlane/gen_powp both measured
   it only pays on spill-bound bodies).

### Operation count (vector ops per 8-pencil block, per axis)

Fold: 4h ALU + 12h ld/st. Kernel: 16 FMA + 12 loads per j per 4 output pairs
→ ~4·h·hu FMA per block (+2·hu per C-only row, ≤2 rows), combine 6 ALU + 4
masked stores per pair. h = ⌊(L−1)/2⌋, hu = h+(L even). Per volume ≈
3·(L³/8)·4h·hu FMAs ≈ ¼ of r1's 3·L³·4L²/8. Map: ~22 FMA-class + 1 vdivpd
per 8 complex. Axis-2 staging: 48 shuffles per 8×8 site block each way.

### Measured on the node (a80n0, leased core via tryout.sh, graded chain, min µs/xform; MKL 2022 same window)

| L | B | m | gen_r1 | gen_r2 | speedup | MKL (same window) |
|---|---|---|---|---|---|---|
| 10 | 64 | 1000 | 9.72 | **5.32** | 1.83× | 4.67 |
| 12 | 64 | 600 | 15.95 | **8.61** | 1.85× | 7.90 |
| 15 | 32 | 600 | 33.63 | **19.92** | 1.69× | 16.78 |
| 20 | 32 | 256 | 93.16 | **43.68** | 2.13× | 58.23 |
| 25 | 16 | 256 | 229.26 | **100.67** | 2.28× | 120.83 |
| 27 | 16 | 200 | 291.97 | **141.65** | 2.06× | 145.29 |
| 31 | 16 | 140 | 466.30 | **264.39** | 1.76× | 868.86 |
| 32 | 8 | 250 | 493.85 | **241.44** | 2.05× | 183.05 |
| 40 | 8 | 128 | 1114.32 | **551.55** | 2.02× | 418.34 |
| 50 | 4 | 128 | 2918.74 | **1358.01** | 2.15× | 954.00 |
| 100 | 1 | 64 | 42345.9 | **23120.0** | 1.83× | 7954.14 |

The DENSE FLOOR now beats MKL at L = 20, 25, 27 and 31 (3.3× at 31). B=1:
L=12 10.49, L=31 335.4 — the code path is batch-invariant (volume-major
chain); the elevation is the known B=1 short-unit core-ramp signature
(gen_dense_prime ice_r8 diagnosis). Decomposition of the round at L=12 B=64:
folded engine execute-only 8.77 (r1 ≈ 11.5 through the 15.95 chain), chain
ownership takes the graded step to 8.61 — the owned chain step (FFT+map) is
CHEAPER than raw execute alone because the per-volume 600-step loop stays
L1/L2-resident (28 KB state) where execute-only sweeps 3.4 MB of driver
buffers per call. Intermediate checkpoints (folded engine, fallback chain):
12: 13.16, 25: 148.6, 31: 326.1, 50: 1571.3, 100: 23760.

Gates, all 11 sizes: single-call rel L2 2.8–5.6e-16 (tol 1e-12); two-step
9.8e-16 (tol 3e-14, 30× margin); full graded chains 2.8e-14–1.8e-13 at
1.15–2.7× the honest anchors (tol 1e-10); single AND chain outputs
bit-identical across independent node processes (manual cmp — tryout's
repeatability leg still dies before running for chain cases, see below).
Setup ≤ 13 ms at L=100. Scalar (non-AVX-512) build verified correct at
3, 4, 31, 64, 100, 127, 128 including full chains.

### The A/B ladder, re-run on the rounder kernel (r1 promise)

L=32 B=8 (power-of-2 strides, batched volume phase rotation ≡ 0 mod 4096 —
the 4K worst case), same window, min µs/xform: default (THP + stagger +
far-phase placement) **251.2**, `-DGL_DEMO_COLLIDE=1` 252.0, `-DGL_DEMO_PLAIN=1`
252.9 — still a null result (≤0.7%, ~noise). Same interpretation as r1, now
with the chain's extra state/c/zt streams in play: a kernel that stages
everything through L1-resident blocks decouples its load/store streams and
hides the 4K hazard. The placement tools remain free insurance; the entries
that need the collision MODEL are the ones with direct strided hot loops
(the class engines), which is exactly what 4b's audit call is for.

### What did NOT work / was deliberately skipped, with the number

- **tryout.sh note (update to r1's)**: the `$W`-before-definition bug is FIXED
  this round (W is set at line 36 before CH uses it), so tryout runs chains
  unaided. STILL broken: the remote check.py leg receives a literally-quoted
  `'$W/c.bin'` (single quotes inside `$(...)`) → FileNotFoundError '/c.bin',
  and because of the `&&` chain the repeatability cmp then never runs for any
  chain case. Run `check.py --map-check m --cin $W/c.bin` yourself (wallaby,
  shared FS — numpy-only) and cmp the two `.chain` outputs on the node.
- **sched-pressure attribute**: not tried, on evidence — the shipped kernel
  compiles to 1 zmm spill total, and gen_batchlane (+17% at L=10/12) /
  gen_powp (+48% at L=100) both measured the flag as a loss on
  non-pressure-bound bodies. Recorded so nobody burns a window on it.
- B=1 chain reads ~20% above batched at 12/31 despite an identical code path
  — consistent with the documented core-ramp artifact, not a code difference;
  don't restructure for it.

### Borrowed this round, named

- **gen_dense_prime gen_r1**: the conjugate-pair folded dense arithmetic and
  its op-count doctrine; the volume-major chain scheme; the s6 map shape and
  their div-vs-rcp race verdict (taken at face value, not re-raced); the
  objdump spill-audit discipline.
- **gen_pfa_small gen_r1**: the "fast map is worth −24..−32% to ANYONE who
  owns their chain — take it" directive that triggered the fft3d_chain work.
- **gen_rader gen_r1**: confirmation that in-place/resident chain structure
  (not arithmetic) is worth ~25% — the residency argument behind volume-major.
- **ice L17_matrixsimd r8 / L23_rader r5** (already in r1): the collision
  model now exposed as the 4b one-call API.

### What I would do next (gen_r3)

1. **Adoption remains the score**: gen_twiddle adopted `gl_arena` in r1;
   gen_dense_prime / gen_pfa_large / gen_pow2 all list layout adoption in
   their next-steps. The 4b audit + pitch picker are written to their asks;
   if a round-3 prompt allows a PR-style diff into an owner's create(), do
   gen_dense_prime's t1/state pitch first (their 1.43 MB working set rides
   the L2 boundary and every 496 B row is line-split — `gl_pick_pitch4k`
   plus a padded-row variant of their chain is a measurable afternoon).
2. **Demo, if idle**: axis-1 lane waste at L≡4 mod 8 (inner=L chunks of 8;
   at L=12 the second chunk carries 4 dead lanes — ~15% of axis-1); a
   two-plane pairing or 4-wide (ymm) tail codelet would fix it. And the
   axis-0 deint could fold into the fold pass (u/v built from interleaved
   loads directly) saving one pb round trip.
3. **Round-3 any-L duty**: supports() already takes 2..128 and the engine is
   size-generic with zero per-size code; keep it as the panel's guaranteed
   floor for surprise draws (it now beats MKL at some of them).

## Round gen_r3

### What changed

**Library (`gl_*` API): nothing.** No adopter asked for a new primitive in
their r2 records, so the API is frozen this round; churn in a layer others
`#include` is its own cost. All work went into the demo entry — which this
round DOGFOODS the layer's own collision model (below), giving adopters a
worked example of section 4b in real kernel code.

**Demo entry, three changes (all r2 next-step items):**

1. **Axis-1 cross-plane lane packing.** The r2 form processed y-pencils per
   x-plane, `ceil(L/8)` vector groups of 8 for L pencils — at L=25 that is
   32 lanes carrying 25 pencils (28% dead), at L=10 60% dead. Now y-pencils
   are indexed globally by g = x·L + z and grouped 8-at-a-time across plane
   boundaries: a boundary group takes lanes 0..cnt−1 from plane x (columns
   z..L−1) and lanes cnt..7 from plane x+1 (columns 0..7−cnt) with a second
   masked load per row (`_mm512_mask_loadu_pd`, complementary masks, second
   base = plane x+1 minus cnt elements so the lane index maps to the right
   column). Group count per volume drops from L·ceil(L/8) to ceil(L²/8) —
   zero dead lanes except the single final tail when L² is odd. Boundary
   groups (≤ 1 per plane) route the kernel through the `ob` staging block and
   pay two masked stores per output row; interior groups store direct,
   exactly as before. For L < 8 (a group would span >2 planes) the old
   per-plane path is kept.

2. **Axis 2 fused behind axis 1 through a 4-plane circular window.** A z-pencil
   of plane x depends only on plane x of the axis-1 output, so the full s2
   scratch volume never needs to exist: axis-1 output goes into a 4-slot
   plane window (slot = x mod 4), and axis-2 trails plane-by-plane on rows
   already complete, in full groups of 8 rows (row blocks freely cross plane
   boundaries — per-lane row pointers into the window). Live span is 3
   consecutive planes (writer touches x, x+1; reader is at ≥ x−1), so 4 slots
   can never collide. The window plane pitch is picked at create() by the
   layer's own `gl_alias_pairs4k` — candidates line-rounded L²·8 B + 0..7
   lines, scored as alias pairs of one slot's L live rows against the other
   three slots' rows (the section-4b pattern, applied to myself). s2
   allocation shrinks from a volume to 4 plane slots (L=100: 8 MB → 640 KB
   per component).

3. **Fold-direct staging on axes 0 and 1.** u/v rows are built straight from
   the source loads (row j and row L−j loaded/deinterleaved, add/sub, store
   to ub/vb); only row 0 still lands in pb (the kernel reads x0 there). Kills
   the full pb staging round trip per chunk (2L vector stores + 2L reloads).
   Axis 2 keeps pb: its pairing partner rows live in different transpose
   blocks.

4. **2-pair tail in the pair kernel** for h mod 4 ∈ {2,3} (was: leftover k's
   one at a time, 6 loads / 4 FMAs per j — load-bound). Helps L=31
   (h=15: tail sweeps 3→2, measured −8.5% on top of the rest); wash at L=15
   (h=7, staging-dominated at that size — number below).

New A/B knob: `-DGL_DEMO_NOFUSE=1` reverts to the r2 per-plane axis-1 +
full-volume axis-2 (used for the attribution runs below).

### Operation count

FMA count unchanged (~4·h·hu vector FMA per 8-pencil group + C-only rows;
h=⌊(L−1)/2⌋, hu=h+even). What changed is group count and traffic: axis-1
groups L·ceil(L/8) → ceil(L²/8) per volume; per-chunk staging on axes 0/1
loses 2L stores + 2L loads (pb round trip); scratch s2 DRAM round trip
(2·V·16 B write+read per volume) replaced by an L2-resident 4-plane window;
boundary groups add 4 masked stores + 2 loads per output row, ≤ 1 group per
plane.

### Measured on the node (a80n0, leased core via tryout.sh, graded chain, min µs/xform)

| L | B | m | gen_r2 | gen_r3 | delta | MKL (same window) |
|---|---|---|---|---|---|---|
| 10 | 64 | 1000 | 5.32 | **4.87** | −8.5% | 4.60 |
| 12 | 64 | 600 | 8.61 | **7.98** | −7.3% | 7.74 |
| 15 | 32 | 600 | 19.92 | **19.07** | −4.3% | — |
| 20 | 32 | 256 | 43.68 | **42.02** | −3.8% | — |
| 25 | 16 | 256 | 100.67 | **94.54** | −6.1% | 121.2 |
| 27 | 16 | 200 | 141.65 | **123.44** | −12.9% | — |
| 31 | 16 | 140 | 264.39 | **227.70** | −13.9% | — |
| 32 | 8 | 250 | 241.44 | **223.95** | −7.2% | 174.5 |
| 40 | 8 | 128 | 551.55 | **499.85** | −9.4% | — |
| 50 | 4 | 128 | 1358.01 | **1241.26** | −8.6% | — |
| 100 | 1 | 64 | 23120.0 | **19508.6** | −15.6% | — |

B=1: L=12 9.13 (r2 10.49), L=31 253.6 (r2 335.4) — path is batch-invariant;
the residual B=1 elevation is the documented core-ramp artifact. Attribution
at L=100 via the knob: `-DGL_DEMO_NOFUSE=1` 23858.9 vs default 19508.6 — the
window+packing is worth **18%** at L=100, i.e. at the memory-pressure size the
locality change finally shows the double-digit effect the r1/r2 A/B ladder
could never produce on this kernel (it was the extra volume round trip, not
the 4K phases, that mattered). The dense floor now beats or matches MKL at
10, 12, 20, 25, 27, 31 (3.8× at 31) and every gen library-layer stub.

Gates: single-call rel L2 2.8–5.6e-16 all 11 suite sizes (tol 1e-12);
two-step map gate 9.44e-16 (tol 3e-14, 30× margin); full chain m=600 at L=12
5.33e-14 vs honest anchor 3.89e-14 (tol 1e-10); chain outputs bit-identical
across independent node runs (manual cmp — tryout's chain-case check/cmp legs
are still broken, see r2 notes; the by-hand invocation from my r2 record
still works verbatim). Correctness spot-checks beyond the suite: L = 7, 8, 9,
13, 17, 64, 127, 128 on the node; scalar (AVX2 wallaby) build PASS at
L = 4, 9, 12, 27. Entry and LIB_ONLY modes compile `-Wall -Wextra`-clean,
native and `-march=icelake-server`. Setup ≤ 10 ms at L=100.

### What did NOT work / attribution notes, with numbers

- **2-pair kernel tail at L=15**: 19.07 → 19.15 µs — wash (h=7 is
  staging-dominated; the tail change only pays when the pair sweep dominates,
  as at L=31). Kept anyway: it never loses, and h mod 4 = 3 draws (L=31, 32,
  40) all benefit.
- **First L=100 default reading was 23,334 µs (+0.9% vs r2)** — a dirty
  window (shared L3 on the dev node; sd line said 1.0% but the median was
  inflated). Re-runs: 19,508 / 19,522 µs at sd 0.02–0.2%. Lesson repeated
  from the ice campaign: never conclude from one tryout window; the NOFUSE
  A/B in the same session is what settled it.
- **6-pair kernel groups: considered, not attempted.** The 4-pair sweep is
  already FMA-port-bound on paper (16 FMA vs 12 load-port µops per j → 8 vs
  6 cycles); widening to 6 pairs only reduces the non-binding load pressure
  (0.67 vs 1.0 data loads per pair per j) and the FMA count is unchanged.
  Recorded so nobody burns a window on it without first showing the kernel
  is NOT FMA-bound (e.g. via the axis-2 port-5 shuffle contention below).

### Borrowed this round, named

- Nothing new from other entries — this round cashed in my own r2 next-steps.
  The plane-window fusion is the standard pencil-pipeline locality trick every
  class engine already uses in some form (gen_pfa_large's fused M-at-out,
  gen_pow2's custody planes); the specific contribution here is picking the
  window pitch with the layer's own `gl_alias_pairs4k` instead of the odd-line
  rule — section 4b used in anger, as a worked example for adopters.

### What I would do next (gen_r4)

1. **Adoption is still the score.** The offer from r2 stands, now with a live
   in-file example: `dm_pick_wpitch` + the 4-slot window is exactly the shape
   gen_dense_prime's t1/state pitch question and gen_rader's padded chain
   arena want. If a round-4 prompt allows cross-entry diffs, that first.
2. **Axis-2 port-5 pressure**: the transpose staging (48 shuffles per 8×8
   block each way) shares port 5 with the second FMA pipe on Ice Lake — at
   L=32 that is ~190 shuffle µops against ~480 port-5 FMA µops per row block.
   A split-stores variant (kernel writes split rows, gl_int8 only at the
   final store) or masked-compress staging could buy the axis-2 pass ~10%;
   measure with the port-utilization counters, not by guessing.
3. **The last lane waste**: axis-0/axis-2 tails when L² mod 8 ≠ 0 (≤ 7 lanes
   per volume — negligible) and the L<8 legacy path (irrelevant to scoring).
   Done with lane geometry; the next real step for the floor would be a
   two-level split of the cos/sin matrices (Winograd-style), which changes
   the class — not this entry's job.

## Round gen_r4

### What changed

**Library (two additions, both answers to adopter records):**

- `gl_alias_drained4k(pairs, uops_between, sbuf_entries)` + constants
  `GL_SBUF_ICX=72 / GL_SBUF_SKX=56` (section 4c) — the in-flight gate the
  collision model was missing. This is gen_dense_prime gen_r3's negative
  result turned into API: their model-clean pitches (4.35 weighted pairs)
  never beat their model-worst r2 layout (240 pairs) because ~450 z-GEMM ops
  sit between each block's store drain and the next block's loads — a store
  only alias-blocks a load while it still occupies the store buffer, and
  `sbuf` stores are gone after roughly 2·sbuf unrelated µops. gen_rader won
  9.5% from the same pitch fix because their chunks run back-to-back. Gate
  every `gl_alias_pairs4k`/`gl_stream_audit4k` score through this with your
  kernel's drain-to-load µop distance BEFORE spending a node window on a
  pitch; if the gate zeroes the score, their table says the fix measures as
  a wash or a loss. First-order model; the A/B stays the law.
- `gl_tr8x8_c2i(re[8], im[8], lo[8], hi[8])` (section 6b) — lane-major split
  block → interleaved complex rows per lane in ONE fused network: 16 unpack +
  32 shuffle_f64x2 = 48 shuffles for the 16 output zmm, where
  `gl_tr8x8`×2 + `gl_int8`×8 spent 64. Port-5 relief on Ice Lake (shuffles
  share port 5 with the second FMA pipe). This is the store side of every
  bl8/soa8-style unpack — gen_batchlane's lane→interleaved exit and
  gen_powp's SoA→natural conversions are this shape. Exactness is now part
  of `gl_selftest()` (bit-compare against the tr8x8+int8 reference path).

**Demo entry (three changes):**

1. **Software prefetch in the fold loads (axes 0 and 1).** The folds walk
   ~L separate row streams per group (rows j and L−j for j=1..h, strides
   16·L² B on axis 0, 8L B on axis 1). At L=100 that is ~100 concurrent
   streams — far past what the L2 streamer tracks — so every group's first
   touch of each row eats a demand miss despite the access being perfectly
   sequential per row. Fix: `_mm_prefetch(T0)` of each row's next-next
   group line(s) at load time (2 groups ahead; 2 lines/row on interleaved
   axis 0, 1 line/row on split axis 1). Knob `-DGL_DEMO_NOPF=1`.
2. **Non-temporal full-line stores on DRAM-resident streams.** When a split
   scratch component is ≥ `GL_NT_MIN_BYTES` (4 MiB; only L=100 in the suite
   qualifies), the axis-0 kernel stores to s1, the axis-2 stores to dst, and
   the map stores all replace their RFO (read-for-ownership) line reads with
   `vmovntpd` — each such store writes a full aligned 64 B line exactly once
   and nothing re-reads it from cache at that size. Eligibility is proven at
   create(): axis-0 rows need L² ≡ 0 (mod 8) (row stride a line multiple),
   dst pencil rows need L ≡ 0 (mod 4); every NT store site also requires a
   full mask (partial groups fall back to masked stores), and each pass ends
   with one `sfence`. Below the threshold the next pass reads the stream
   back from L2/L3, so NT would force a DRAM round trip — that is why the
   default stays off everywhere else. Knob `-DGL_DEMO_NONT=1`.
3. **Axis-2 scatter through `gl_tr8x8_c2i`** (both the window path and the
   NOFUSE/generic path) — the r3 next-step port-5 item, done by fusing
   rather than by restructuring: 64 → 48 shuffles per 8×8-complex exit
   chunk, −14% port-5 traffic per axis-2 row block.

### Operation count

FMA count unchanged (~4·h·hu vector FMA per 8-pencil group per axis). Deltas:
axis-2 exit shuffles 64 → 48 per 8×8-complex chunk; +2 prefetch µops per fold
row load (ports 2/3, which the fold leaves idle); at NT-eligible sizes the
axis-0 s1 stores, axis-2 dst stores, and map stores lose one RFO line READ
each — at L=100 that is 48 MB of eliminated read traffic per chain step
(3 × 16 MB), plus the same stores no longer thrash cache; one sfence per pass.

### Measured on the node (a80n0, leased cores via tryout.sh, graded chain, min µs/xform)

NOTE on windows: this round the node read ~4% hot vs the r3 board (r3 binary
re-measured 237.5 at the L=31 cell whose board number is 227.7); interleaved
same-window A/Bs below are the honest signal, per gen_dense_prime's r3
protocol.

| L | B | r3 board | gen_r4 | delta | note |
|---|---|---|---|---|---|
| 10 | 64 | 4.87 | 4.99 | ~wash | sd 7.2% window |
| 12 | 64 | 7.98 | **7.98** | = | NOPF arm read 8.43 same window |
| 12 | 1 | 9.13 | **8.03** | −12% | |
| 25 | 16 | 94.5 | **93.4–93.7** | −1% | 2×2 interleaved A/B; a first read of 103.4 was a dirty window (sd 4.4%) — r3's "never conclude from one window" lesson, again |
| 31 | 16 | 227.7 | **231.4** | −2.6% vs same-window r3 control 237.5 | odd L: NT ineligible; the delta is prefetch (NOPF arm 238.2) |
| 31 | 1 | 253.6 | **227.5** | −10% | B=1 now ≤ batched (quiet core) |
| 32 | 8 | 224.0 | 232.8 | ~wash | hot window, sd 2.8% |
| 50 | 4 | 1241 | 1290 | ~wash | hot window, sd 3.6%; NT ineligible (1 MB component) |
| 100 | 1 | 19509 | **15606–15845** | **−19%** | NT-eligible; MKL same windows 8283–8832 |

**Attribution at L=100 B=1 (same window, knob knockouts):** default
15,829; `-DGL_DEMO_NOPF=1` (NT+c2i) 15,845; `-DGL_DEMO_NONT=1` (PF+c2i)
19,590. **The entire L=100 win is the NT stores.** The r3 diagnosis stands
corrected in detail: the ~100-stream fold LOADS were never the binder at
L=100 — out-of-order execution rides them — it was the three full-volume
store streams whose RFO reads doubled the write traffic and occupied the
LFBs the loads needed. Prefetch instead pays at the L2/L3-resident middle
(L=31: −2.6%, 3-run stable; L=12 B=64: NOPF +5.6% in-window), where the
fold's row streams miss L2 but the working set is small enough that the
misses are the binder.

Gates, all by hand (tryout's chain legs still die on the unexpanded
'$W/c.bin' — my r2 recipe works verbatim): single rel L2 2.8–5.6e-16 at
10/12/25/31/32/50/100 (tol 1e-12); graded map-chains PASS at 12 (5.33e-14),
31 (2.84e-14), 100 (6.46e-14) vs tol 1e-10; **two-step gate 9.4e-16 /
1.7e-15 / 3.3e-15 at 12/31/100** (tol 3e-14 — the L=100 figure is the NT
path, 9× margin); L=100 single AND chain outputs bit-identical across
independent node runs (NT determinism verified, not assumed). Scalar
(AVX2 wallaby) build PASS end-to-end incl. chains at L=4, 9, 12, 27.
Setup ≤ 7 ms at L=100. All four build modes (entry/LIB_ONLY × AVX-512/
scalar) compile `-Wall -Wextra`-clean.

### What did NOT work / boundaries, with numbers

- **Prefetch at L=100: a wash** (15,845 vs 15,829 — see attribution). Kept
  because it wins where NT is ineligible and never measurably loses.
- **NT below the size gate: not even attempted, by design.** Below
  ~4 MiB/component the next pass reads the stream back from L2/L3; NT would
  trade a saved RFO for a forced DRAM read-back. The gate (plus the
  alignment proofs) is the shipped answer; if a cross-arch round wants a
  different threshold, it is one -D away (`GL_NT_MIN_BYTES`).
- **First L=25 reading (103.4 µs, +9%)**: dirty window, resolved by 2×2
  interleaved A/B at 93.2–94.1. Recorded because it is the second time this
  campaign a single-window number nearly caused a wrong call.

### Borrowed this round, named

- **gen_dense_prime gen_r3**: their falsification of my pure-geometry
  collision model (the "stores still in flight" mechanism) is now section
  4c of the library, credited in the file; also their interleaved-arms
  measurement protocol, used for every A/B above.
- **gen_rader gen_r3 / gen_dense_prime gen_r3**: the huge-page-arena +
  phase-placement recipes they adopted from me came back as confidence that
  the remaining L=100 gap was store-side, not layout-side — which is what
  the NT result confirmed.

### What I would do next (gen_r5)

1. **Adoption of the two new primitives**: `gl_alias_drained4k` is written
   for gen_dense_prime (their own negative result, returned as API) and for
   gen_rader's any-prime pitch planning; `gl_tr8x8_c2i` is the store side
   of gen_batchlane's bl8 exit and gen_powp's SoA→natural conversion —
   both are one-call swaps.
2. **NT-store guidance now has a measured basis** (r1 item 4 closed): full
   aligned lines, nothing re-reads the stream at that size, ≥4 MiB per
   component, sfence per pass — worth −19% at L=100. If adopters want it,
   the dm_st shape (mask fallback + eligibility proof at create()) is the
   thing to copy, not a raw vmovntpd.
3. **Cross-arch caution for the advisory rounds**: the NT win is an
   Ice Lake measurement; CLX (1 MB L2, heavier 512-bit downclock) and SPR
   (larger L2) move both the NT threshold and the prefetch payoff — the
   knobs exist precisely so the race layer can flip them per host.
4. **The floor itself is done evolving inside its class** — remaining ideas
   (Winograd-style matrix splits) change the algorithm class; the demo's
   job stays: any-L guaranteed-correct vectorized floor + the layer's
   living test bench.

## Round gen_r5

### What changed

**Library (one addition, section 7 of the file):**

- `gl_map8(z0, z1, &o0, &o1)` / `gl_map16(z0..z3, &o0..&o3)` — the graded
  chain map w/(1+|w|) as IN-REGISTER primitives on 8 (resp. 16) interleaved
  complex. `gl_map8` is bit-for-bit the campaign-standard ladder this layer
  shipped inside its map pass since r2 (pair-compressed |w|² via unpacklo/hi
  — each lane is one complex's re resp. im, so values never mix across
  complex and the chunking an adopter fuses under is irrelevant to the
  numbers — 1e-300 guard, rsqrt14 + 2 Newton, ONE exact vdivpd per 8
  complex). `gl_map16` is gen_dense_prime gen_r4 next-step 3 BUILT: one
  vdivpd per 16 complex via the reciprocal-product trick (q = 1/(da·db),
  rec_a = q·db, rec_b = q·da; 3 extra vmulpd, ~1–2 ulp on the reciprocal
  against a per-step budget of ~60 ulp). Both are in `gl_selftest()`
  (gl_map8 within 5e-15 of the exact scalar map, gl_map16 within 1e-14 of
  gl_map8). Fuse them at your engine's exit store and the separate map pass
  — and its full state round trip — disappears.

**Demo entry (one structural change):**

- **The graded map is fused into the axis-2 exit and the chain runs IN
  PLACE.** The r2–r4 chain shape was: FFT the state into a full interleaved
  `zt` volume (NT stores at L=100), then a separate map pass reads zt + c
  and writes the state. But the axis-2 exit already holds the final
  interleaved values in registers the moment `gl_tr8x8_c2i` produces them —
  so now the exit adds c, runs `gl_map8` on the registers, and stores the
  chain STATE directly. `zt` is deleted from the plan (16 MB at L=100), and
  the state volume is rewritten in place — legal because axis 0 fully
  consumes the state into split s1 before axis 2 rewrites it. Per chain
  step this deletes a full volume write + read (zt) plus the separate
  pass's loop; map arithmetic is unchanged and moved to ports the scatter
  leaves idle. NT stores now target the state directly (same eligibility
  proofs, plus a runtime 64B-alignment check on the actual dst). Knobs:
  `-DGL_DEMO_NOMAPFUSE=1` (exact r4 shape, for the A/Bs below),
  `-DGL_DEMO_MAP16=1` (gl_map16 in the exit).

### Operation count

FFT FMA count unchanged. Deleted per chain step: V·16 B zt write + V·16 B
zt read + the separate map pass (its loads/stores of state and zt). Map
arithmetic itself unchanged (~14 FMA-class + 2 unpack + 1 vdivpd + 2 mul
per 8 complex) but now runs once per exit CHUNK: pencils issue ceil(L/8)
map calls instead of L/8, so L mod 8 ∈ {1..4} pays dead-lane divides
(+33% map-div count at L=12, +28% at 25, +12% at 50). That waste is the
measured story of the small-L cells below.

### Measured on the node (a80n0, leased cores via tryout.sh, graded chain, min µs/xform)

Window note: this session read ~3–6% hot vs the r4 board at the small sizes
(hot windows, sd up to 10% on first readings); every keep/kill below is an
interleaved same-window A/B (fused vs `-DGL_DEMO_NOMAPFUSE=1` = the r4
shape), per the gen_dense_prime/gen_pfa_large protocol.

| L | B | r4 board | gen_r5 | same-window r4-shape arm | verdict |
|---|---|---|---|---|---|
| 10 | 64 | 4.90 | **5.22** | 5.46 | −4.6% |
| 12 | 64 | 8.08 | **8.49** | 8.68 | −1.5% |
| 12 | 1 | — | 9.97 | 9.64 | **+3.7% (kept anyway, see below)** |
| 15 | 32 | 18.40 | **18.43** | — | wash |
| 20 | 32 | 39.91 | **41.15** | 41.14 | wash (3 pairs: −2.9%, +0.3%, −3.5%; min-of-mins equal) |
| 25 | 16 | 93.36 | **95.30** | 97.49 | −2.3% (2 pairs, both fused) |
| 27 | 16 | 123.8 | **121.6** | — | −1.8% vs board |
| 31 | 16 | 223.5 | **196.6–201.0** | 231.7 | **−13..15%** |
| 31 | 1 | 227.5 (r4) | **222.1** | — | |
| 32 | 8 | 227.7 | **199.0** | — | **−12.6% vs board** |
| 40 | 8 | 507.7 | **461.8** | — | −9.0% vs board |
| 50 | 4 | 1204 | **1147** | — | −4.7% vs board |
| 100 | 1 | 15083 | **14976** | 15088 | −0.7%, ~wash (MKL same windows 7691–7893) |

The shape of the result: the fusion pays exactly where zt lived in L2/L3
and round-tripped it (the 25–50 middle, −2..15%, biggest where the working
set rides the L2/L3 boundary: 31, 32, 40), is a wash at L=100 (after r4's
NT work the chain there is no longer bandwidth-exposed — the deleted 32
MB/step of streaming traffic was fully overlapped; the r4 attribution
stands: it was the RFO reads, not the volume count), and at L1-resident
small-L B=1 the dead-lane divides show up as the only term (+3.7% at 12
B=1). Kept everywhere anyway: every SCORED cell wins or washes, and one
code path beats a size-conditional fusion nobody can maintain.

**gl_map16 verdict (gen_dense_prime's ask): a LOSS in a fused exit.**
Same windows: L=31 220.5 vs 201.0 (+9.7%), L=100 15274 vs 14976 (+2%).
In a fused exit the vdivpd was never the binder — it overlaps the next
block's FMAs in the OoO window — so halving divider occupancy buys nothing,
and pairing two pencils per call lengthens the dependency chain into both
pencils' stores and raises exit register pressure (the compiler spills).
gen_dense_prime's context (map divides inside a z-GEMM phase that is
port-0/5-saturated) is different enough that their item 3 is NOT killed by
this — but measure there, don't adopt on faith. For a REGISTER-fused exit,
use gl_map8.

### Gates (all by hand — tryout's chain legs still die on the unexpanded '$W/c.bin'; r2 recipe verbatim)

Single-call rel L2 2.5e-16–8.7e-16 at 10/12/15/20/25/27/31/32/40/50/100 and
off-suite 7/9/16/17/64/127/128 (tol 1e-12); **two-step gate 9.44e-16 /
1.73e-15 / 3.29e-15 at 12/31/100** (tol 3e-14 — identical to r4, as it must
be: gl_map8 is the same arithmetic the pass ran, and the pair-compress
makes each complex's result independent of chunking); full graded chains
5.33e-14 (12, m=600), 2.88e-14 (31, m=140), 6.46e-14 (100, m=64) vs tol
1e-10; chain outputs bit-identical across independent node runs at 31 and
100; off-suite m=3 chains PASS at 7 (the AVX-512 non-window path), 9, 16,
17, 64, 127, 128; scalar (AVX2 wallaby) build PASS singles + chains at
4/9/12/27 (the scalar exit fuses the map too, exact sqrt). All build modes
(entry/LIB_ONLY × icelake-server/x86-64 × NOMAPFUSE/MAP16/PLAIN) compile
`-Wall -Wextra`-clean. Setup unchanged (≤ 7 ms at L=100).

### What did NOT work / boundaries, with numbers

- **gl_map16 in the exit: +9.7% at 31, +2% at 100** (above) — shipped as a
  library primitive with a measured negative verdict attached, which is the
  point: the next entry that wonders about one-div-per-16 reads a number
  instead of burning a window.
- **L=12 B=1: fused +3.7%** (9.97 vs 9.64) — dead-lane map divides with no
  traffic win to pay for them (28 KB working set never leaves L1). Unscored
  cell; not special-cased. A ymm-width map for kcnt ≤ 4 tail chunks would
  recover it if a scored case ever lands on this regime.
- **L=100 expectation corrected**: I predicted −15..25% from deleting 32
  MB/step; measured −0.7%. The r4 NT work had already removed the exposed
  DRAM time; what remains at L=100 binds inside the FFT passes (fold-load
  L2 misses, port pressure), not on volume count. Recorded so nobody
  re-derives the fusion for bandwidth reasons at DRAM sizes — its money is
  the L2/L3 middle.

### Borrowed this round, named

- **gen_dense_prime gen_r4**: the map-into-z-phase fusion result (their
  −2 MB/step of L2 traffic) is what made this round's shape obviously right
  for the middle sizes; their next-step 3 (one div per 16) is built and
  raced here as gl_map16.
- **gen_pfa_large gen_r4 (ipp) / gen_bluestein gen_r4 (per-block map)**:
  the two other engines that measured map-pass deletion this campaign —
  between the three records the "fuse the map where the state is already
  hot" doctrine is now cross-validated; my register-exit variant is its
  third form (theirs: plane/block granularity; mine: exit-register
  granularity, zero extra passes of any kind).
- **gen_dense_prime gen_r3 / gen_batchlane gen_r4**: the interleaved
  adjacent-pairs A/B protocol, again (the L=20 and L=25 calls above are
  exactly the kind a single window would have gotten wrong).

### What I would do next (gen_r6)

1. **Adoption of gl_map8 at other exits**: gen_rader's r4 list already has
   "gen_layout's NT stores for the map stores at 40/80" — gl_map8 composes
   with that (their exit is interleaved rows too); gen_powp's parity unpack
   and gen_pow2's custody exit are the same shape. The one-call swap is
   written; my record carries the map16 warning so nobody re-races it
   blind.
2. **Round 6 readiness is the priority**: supports() takes any 2..128, the
   engine is size-generic with zero per-size code, gates pass at surprise-
   style sizes (7..128 verified this round), setup ≤ 7 ms. The demo remains
   the panel's guaranteed floor for the three unseen draws.
3. **Cross-arch (XARCH.md due after this round)**: the fusion moves map
   arithmetic into the exit's port mix — on CLX (1 MB L2, heavier downclock)
   the L2/L3-middle win should GROW (more of the suite becomes
   traffic-bound), while the dead-lane div cost is clock-invariant. The
   NOMAPFUSE knob exists precisely so the race layer can flip it per host
   if CLX disagrees.
4. **The ymm tail map** (item from the L=12 B=1 loss) — only if a scored
   cell lands on L mod 8 ∈ {1..4} at L1-resident sizes.

## Round gen_r6

### What changed

**Library (`gl_*` API): frozen again — doctrine updates only.** No adopter r5
record asked for a new primitive, so the round-3 rule holds (churn in a layer
others `#include` is its own cost). Two comment-block updates fold this
round's cross-entry findings into the manual adopters actually read:

- `gl_pad_stride` now records the REVISED mechanism from gen_pow2 gen_r5:
  their L=32 audit shows no store→load pair at equal addr mod 4K in any phase,
  so the odd-line pad's value there is **L1-set uniformity** (gcd(stride_lines,
  64)=1 walks all 64 L1 sets; a power-of-2 line stride hits 16 sets at 4×
  depth). Same rule, second mechanism — both want gcd(stride_lines, 64)=1.
- Section 7 (gl_map8/gl_map16) now carries the post-r5 **adoption map and the
  pair-packing boundary**: gen_powp took the gl_map16 reciprocal-product trick
  into split-complex (divider ops 25→15 at L=25, −1.6%); gen_twiddle REJECTED
  pair-packed ladders inside their tr8x8-bound scatter exit; gen_pfa_large's
  map_step_pair wins −8..−14% on standalone map passes. Boundary: share the
  ladder where the map is standalone/FMA-bound; in a shuffle-bound (port-5)
  fused exit use plain gl_map8. (This round's adoption receipts, from their
  records: gen_rader now includes the layer via `GEN_LAYOUT_LIB_ONLY` for
  `gl_map_huge`/`gl_unmap`; gen_powp ships the map16 idea as above.)

**Demo entry (one structural change): even-L second-level fold.** The r3–r5
kernel folded j ↔ L−j once (conjugate pairs, real C/S matrices over u/v).
For even L there is a second exact symmetry: cos(2πk(L/2−j)/L) = (−1)^k
cos(2πkj/L) and sin(2πk(L/2−j)/L) = −(−1)^k sin(2πkj/L). So outputs split by
k-parity and j folds again over (j, L/2−j):

- fold builds FOUR blocks — ue = u_j + u_{L/2−j}, uo = u_j − u_{L/2−j},
  ve = v_j − v_{L/2−j}, vo = v_j + v_{L/2−j} — for j = 1..⌊h/2⌋, plus a lone
  j = L/4 row when 4 | L (its coefficient is exactly 0 for the parity that
  does not use it, so the loop stays uniform);
- x_{L/2} leaves the table and becomes two base rows: e = x0 + x_{L/2}
  (even k), o = x0 − x_{L/2} (odd k) — free in the fold;
- the kernel (`dm_kfold8e`) runs the SAME 4-pair sweep but over he2 =
  ⌊h/2⌋+(4|L) columns: odd-k pairs read (uo, vo, o-base), even-k pairs
  (ue, ve, e-base). Pairs start at k=1 and step 4, so the parity pattern
  (o,e,o,e) is static; the 1-pair tail is provably always odd. Tables
  Ct2/St2 mirror the Ct/St layout (rows k=1..h, then k=0, then k=L/2), filled
  by the same exact long-double reduction.

Still a dense real-matrix method — no twiddles, no factorization, any even L,
one code path (odd L runs the r5 kernel unchanged; L<8 keeps the legacy
path). All four axis sites (axis-0 direct-load fold, axis-1 split-load fold
incl. boundary groups, both axis-2 stagings and the NOFUSE path) switch on
`p->evenk`. Knob: `-DGL_DEMO_NOEVEN=1` reverts to the r5 kernel everywhere.

### Operation count

Even L, per 8-pencil group per axis: kernel j-sweep 4·h·he2 ≈ **2·h·hu vector
FMAs — HALF of r5**; per j the sweep now loads 8 block rows + 8 broadcasts =
16 loads against 16 FMAs (balanced on paper: 8 cycles each on ICX ports 2/3
vs 0/5), where r5 spent 12 loads/16 FMAs over twice the j's. Fold: 16 add/sub
+ 8 row loads + 8 stores per j over h/2 — same totals as r5's 4/4/4 over h.
Table bytes: +2·(h+2)·hs2 doubles (≈ 26 KB at L=100, cold after create()).
Odd L: identical to r5. Spill audit: zero rsp-relative zmm ops in every hot
function (only the cold gl_selftest spills) — the 16-accumulator + 8-row
budget fits.

### Measured on the node (a80n0, leased cores via tryout.sh, graded chain, min µs/xform)

Same-session control: the r5 binary read 8.80 at L=12 B=64 where the r5 board
says 8.41 — windows ~5% hot; the honest arms are the interleaved
`-DGL_DEMO_NOEVEN=1` runs.

| L | B | r5 board | gen_r6 | same-window NOEVEN arm | MKL same window | verdict |
|---|---|---|---|---|---|---|
| 10 | 64 | 5.26 | **5.17** | — | 4.60 | ~wash (sd 11%, staging-dominated) |
| 12 | 64 | 8.41 | **8.30** | 8.80 (r5 binary) | 7.80–7.87 | −5.7% same-session |
| 12 | 1 | 9.97 | **9.57** | — | — | −4% |
| 20 | 32 | 41.2 | **38.93** | — | 60.7 | −5.5%; 0.64× MKL |
| 25 | 16 | 95.5 | 97.6 | — | — | odd: unchanged (window noise) |
| 27 | 16 | 121.6 | 124.6 | — | — | odd: unchanged |
| 31 | 16 | 196.6–201.0 | 200.7 | — | — | odd: unchanged — no code-layout regression (the gen_twiddle r5 hazard, checked) |
| 32 | 8 | 200.0 | **176.9** | 202.9 | 184.1 | **−12.8%; first MKL win at 32 for this floor** |
| 32 | 1 | — | 201.3 | — | — | B=1 core-ramp signature |
| 40 | 8 | 456.7 | **396.3** | — | 462.2 | −13%; beats MKL |
| 50 | 4 | 1104.9 | **946.2** | — | 1008.5 | −14%; beats MKL |
| 100 | 1 | 14976 | **12357** | 14665 | 7695 | **−15.7% same-window** |

The dense floor now beats or matches MKL at 12(-ish), 20, 25, 27, 31, 32, 40,
50 — everything but 10/15 (PFA territory) and 100.

### Gates (map-chain legs by hand as always — tryout's '$W/c.bin' bug persists; r2 recipe verbatim)

Single-call rel L2 2.8e-16–8.7e-16 at 10/12/20/25/27/31/32/40/50/100 and
off-suite 8/9/14/16/24/33/36/63/64/96/127/128 (tol 1e-12); **two-step gate
9.44e-16 / 1.58e-15 / 3.26e-15 at 12/32/100** (tol 3e-14; the L=12 figure
matches r5 to the printed digits even though the even kernel reassociates the
FFT — the m=2 error is dominated by the unchanged map arithmetic); full
graded chains 4.90e-14 (12, m=600), 3.43e-14 (32, m=250), 7.19e-14 (100,
m=64) vs tol 1e-10; off-suite m=3 chains PASS at all 12 sizes above (even
sizes exercise the new kernel, odd the old, 127/128 the extremes); L=100
chain outputs bit-identical across independent node runs (NT path
determinism, re-verified); scalar (AVX2 wallaby) build PASS singles + m=3
chains at 4/9/12/14/27 (scalar path unchanged, evenk compiled out). Entry and
LIB_ONLY modes compile `-Wall -Wextra`-clean on icelake-server and plain
x86-64. Setup unchanged (≤ 7 ms at L=100; tables fill is still O(L²)
long-double sincos).

### What did NOT work / boundaries, with numbers

- **The FMA halving did not halve wall time** — L=100 gave −15.7%, 32 −12.8%.
  Expected: the kernel sweep is only part of the axis (fold, staging
  transposes, exit, and at 100 the memory system take the rest), and the new
  sweep is load-port-balanced (16 loads/16 FMAs per j) rather than FMA-bound,
  so the halved j-count buys less than 2× on the sweep itself. Recorded so
  the next fold level (see below) is costed honestly.
- **Uniform-loop padding when 4 | L**: the lone j=L/4 column runs through the
  full 4-block loop with two zero coefficients (one wasted j out of he2). At
  L=12 that is 1 of 3 columns; a special-cased epilogue could reclaim it but
  doubles the kernel tail zoo for ≤ a few percent at small even L — skipped
  this round, on the gen_dense_prime r5 finding that exact-tail code-footprint
  costs ~1% where the win is ~0.
- **The ymm tail map (r5 item 4) still not built**: with the even fold the
  L=12 B=1 cell moved −4% anyway (9.97 → 9.57); the dead-lane divides remain
  the residual there. Unscored; still parked.

### Borrowed this round, named

- **gen_pow2 gen_r5**: the L1-set-uniformity mechanism, now in the library's
  gl_pad_stride doctrine comment.
- **gen_twiddle gen_r5**: the code-layout hazard (their +1..4.5% at sizes
  that never execute a new branch) — is why the odd sizes 25/27/31 were
  re-measured before shipping; clean here (branch sits outside the j-loops,
  kernels are separate functions).
- **gen_powp / gen_rader gen_r5**: adoption receipts recorded above; their
  boundary results (pair-packing, RP_PAD_MAX) are folded into section 7's
  adoption map so the next reader gets the verdicts, not the folklore.
- The fold itself extends the conjugate-pair fold adopted from
  gen_dense_prime in r2 (transitively ice L13/L17/L23) one symmetry deeper;
  same real-constant dense class, no twiddles.

### What I would do next (gen_r7 / endgame)

1. **Round 6 readiness stands**: supports() 2..128, zero per-size code, gates
   pass at 24 sizes including 127/128, setup ≤ 7 ms, and the floor is now
   MKL-or-better at most composite draws in 14..127 (even draws hit the new
   kernel). The demo remains the panel's guaranteed correct fallback.
2. **A third fold level for 4 | L** (pair j ↔ L/4−j inside each parity class,
   k mod 4 split) would cut the sweep to ~h/4 columns but mixes cos/sin
   tables per class — that is radix-4 territory and changes the class; if
   anyone builds it, it belongs in gen_powp/gen_pow2, not here.
3. **Adoption**: the even-fold identity applies verbatim to any dense
   real-matrix stage over a length-2m axis — gen_dense_prime's C/S GEMMs at
   even composite lengths and gen_rader's dense blocks at m even are the
   candidates; the dm_kfold8e/dm_fold8e pair is the worked example.
4. **Cross-arch**: the new sweep is load-port-balanced, so on CLX (2 load
   ports but heavier 512-bit downclock) the even-fold win should hold or
   grow; the NOEVEN knob exists for the race layer if SPR disagrees.
