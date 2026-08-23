# Solution notes — iterated batched 3D complex FFTs, eight fixed cubes

## Shape of the solution

`solution.py` (wrapper skeleton kept verbatim apart from buffer handling and
the marked call) drives ONE merged C file, `implementation.c`
(prebuilt `implementation.so` shipped; the wrapper rebuilds it only if the .so
is missing). Everything is single-threaded AVX-512 fp64; no FFT library is
used anywhere (all DFT arithmetic is generated/hand-written code).

Per-size engine routing (in `_dispatch`):

| L | engine | design |
|---|--------|--------|
| 6, 8 | vendored prior-work engine A (00291a90 final) | lane-major SoA (8 volumes/zmm), PFA(2x3)/radix-2 DFT8, slab+pencil alternation, fused rsqrt14/rcp14+NR map, per-volume tail drivers |
| 13, 17, 23 | engine A | same SoA layout, hand-scheduled symmetric ("Hartley-split") prime kernels, phase-split cos/sin sweeps with register-resident constants |
| 36 | **this round's engine C** (`gen/gen_fg.py`, `MJ_` prefix) | see below |
| 45 | engine A for m>12; engine C for m<=12 (C's per-call setup is ~3 ms cheaper per 6 volumes, A's per-step is ~8% faster; crossover ~m=18, threshold set conservatively) | A: within-volume split re/im, two-stage PFA(5x9), 8x8 register transposes, fused map |
| 64 | vendored prior-work engine B (v6_3f30d81f) | in-place slab/pencil sweeps, digit-transposed 8x8 register tiles, compiled with `-DMAP_STYLE=0` (rsqrt14+NR map — the hardware `vsqrtpd` is a serializing ~19-tsc op on this VM) |

## Engine C (L=36): 6D-PFA digit-state engine (new this round)

36 = 4x9 with gcd(4,9)=1, so by CRT the 1D DFT36 = DFT4 (x) DFT9 with pure
digit permutations (no twiddles), and the 3D DFT36^3 factorizes into
DFT9 along three "v" digit axes and DFT4 along three "u" digit axes of a 6D
grid 9x9x9 x 4x4x4. The iteration x <- map(FFT3(x)+c) is conjugation-invariant
under any fixed state permutation (map is pointwise), so the STATE LIVES
PERMANENTLY on the digit grid: rows = v-grid (729, padded stride), 64 lanes =
u-grid; the CRT input/output relabellings are baked into codelet store
positions and into one gather per volume at call entry/exit (`SPOS` tables).

Per step, in place, one volume at a time (state 850 KB stays L2-resident
across ALL m steps; only c streams from L3):

* **Phase A**: three DFT9 passes over rows (v3, v2, v1) — pure vertical SIMD,
  zero shuffles, 12-op FMA DFT3s inside a 88-op DFT9; row stride 72 doubles
  (64+8 skew) so pencil strides never alias L1/L2 sets (the unpadded version
  ran 2x slower — whole chunk mapped into 4 of 64 L1 sets).
* **Phase B**: per row (64 lanes = 8 zmm re + 8 im, all register-resident):
  DFT4 over u3 (vertical zmm quads), DFT4 over u2 (lane-parity pair trick:
  vertical add/sub + `vpermute_pd 0x55` + masked adds), DFT4 over u1 entirely
  IN-LANE (per-zmm `vshuff64x2`/`vpermutexvar`/`vpermutex2var` butterflies with
  per-lane sign vectors folded into FMAs — replaces a transpose/DFT/transpose
  round and measured ~7% faster), then c-add + map fused, store. The map is
  z/(1+|z|) with rsqrt14 + 2 scale-carried Newton steps (q-form: r=16|z| with
  the 1/16 folded into the final fma) and rcp14 + 2 Newton — 21 port05 ops,
  ~1-2 ulp, verified against longdouble reference.

The same generator emits a 45 = 5x9 variant (L1-tile scratch + 8x8 transposes,
since 5 does not divide the lane count); it reached 3.9 ns/el-step vs the
vendored engine's 3.55-3.65, so 45 stays routed to engine A.

## Measured machine facts this round (rechecked, wall-clock tsc units)

- zmm FMA ceiling ~1.9/cyc at ~2.5 GHz under sustained 512-bit load
  (~1.85 FMA/tsc); ymm 2/cyc at ~3.1 GHz — zmm wins 1.5x in wall time.
- loads <=0.5/FMA are nearly free; embedded-broadcast FMA and memory-operand
  FMA run ~1/cyc (avoid); `vdivpd`/`vsqrtpd` ~11/~19 tsc and do NOT overlap
  (avoid in the map); `vrsqrt14pd`/`vrcp14pd` fine here (~1 tsc amortized);
  `vgatherdpd` slower than scalar gather loops for the conversions.
- GCC hoists broadcast constants out of loops into stack spills (ruinous);
  cured with `asm volatile` broadcasts (`bcastv`) in generated kernels.
- Streaming BW: reads ~28 GB/s, RFO writes ~10 GB/s, NT no better at these
  access shapes; L=64 is memory/latency-bound at ~3.4 ns/el-step — a fused
  one-sweep-per-step variant of engine C for 64 (B(t)+A(t+1) in one pass)
  was built and verified but stayed ~35% behind engine B (64B-granular
  column traffic), so 64 remains on engine B.

## Final self-benchmark (this VM, single core, pinned, fresh processes)

Workload W1 (equal-ish element-steps/size: B=(64,48,24,16,12,8,6,4),
m=(600,400,200,150,100,60,40,30) — see /tmp harness; B*m*L^3 ~ 8-32M/size):

- per-size C time (best-of-7): 6: 9.3ms/1.12ns, 8: 12.4/1.26, 13: 24.1/2.28,
  17: 30.7/2.60, 23: 51.6/3.53, 36: 58.6/2.62, 45: 78.5/3.59, 64: 107.9/3.43
  (ns = per element-step); total C ~373 ms.
- transform() wall (best-of-4 calls per fresh process): quiet-window runs
  0.512-0.525 s; noisy-neighbor epochs push the same build to 0.52-0.58 s and
  move base.py from ~5.3-5.4 s to ~5.7-6.1 s in lockstep (shared-VM ambient
  noise is a +-2-8% band on BOTH sides of the ratio). Input generation via the
  protocol numpy RNG accounts for ~0.136 s of every implementation's wall,
  including the references'.
- engine C phase rates at ship time: L=36 phaseA 2.71 tsc/el, phaseB 3.76
  tsc/el -- both at the measured ~1.9 vector-uop/tsc issue ceiling of this VM,
  i.e. the remaining headroom at 36 is op-count, not scheduling.
- base.py (numpy/pocketfft) same workload: ~5.3-5.4 s  (~10.4x).
- MKL DFTI sequential + numpy map, same interface, same workload: ~2.67 s
  (so ~5.2x vs that reference on this mix).
- Correctness: one-step rel-L2 3.5e-16..8.7e-16 per block (gate 1e-14);
  m-step blocks pass per-size gates with 3+ orders margin (worst m6=3000
  chain: 5.3e-11 vs 1e-4 gate). Long-chain, odd-B, B=0, m=1, big-B stress all
  pass; output deterministic across calls.

## Provenance

`gen/` holds this round's generator (`gen_fg.py` + `genlib.py`) and the merge
script. Sections 1-2 of `implementation.c` are vendored verbatim from the
provided prior-work solutions (permitted by the warm-start rules, audited
FFT-library-free), regenerated/copied on this host; section 3 is generated
here. Compile line (also the wrapper's fallback):

```
gcc -O3 -march=native -funroll-loops -fschedule-insns -fsched-pressure \
    -fno-math-errno -fno-trapping-math -DMAP_STYLE=0 -shared -fPIC \
    implementation.c -o implementation.so -lm
```
