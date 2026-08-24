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
