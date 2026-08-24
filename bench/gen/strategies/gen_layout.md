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
