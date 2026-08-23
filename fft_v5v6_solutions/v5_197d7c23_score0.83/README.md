# v5 attempt 197d7c23 — score 0.8266

Reconstruction of the final graded `/workdir` state of attempt `197d7c23` on the
**v5** FFT code-opt problem (`fft3d-fixed-geometry-opt-20260821`, flat accuracy
gates: one-step blocks < 1e-14 relative L2 **per block**, chain blocks < 1e-3,
identical for all eight sizes), reconstructed from
`attempt_197d7c23_score0.83.log`.

## Grading result

| quantity | value |
|---|---|
| score | **0.8266** |
| C_ref (base, best of 3) | 92.974 s |
| C_sota (held-out, best of 3) | 21.498 s |
| C_opt (this attempt, best of 3) | **7.035 s** |
| checks | format ✓ constraint ✓ content ✓ |

C_opt walls: 7.035 / 7.175 / 8.005 s — ≈ 13.2× the base, ≈ 3.06× the held-out
(MKL-DFTI-based) SOTA.

## Files

- `solution.py` — final graded wrapper (written 02:16:36, unchanged after).
  Verbatim skeleton; compiles `implementation.c` at import if
  `implementation.so` is absent, binds `run_size` via ctypes, one-time
  `setup()` at import.
- `implementation.c` — final graded C source (last copied to `/workdir` at
  03:12:42, after the k23 "K3 tail block" patch; `md5sum` in the log confirmed
  the `/workdir` copy tracked the dev copy). **Regenerated** here by
  `dev_generators/run_all.sh`, since the session never printed the file in
  full — it was assembled from heredoc stages plus ~20 in-place Python patch
  scripts, all of which are transcribed under `dev_generators/`.
- `dev_generators/` — the reconstructed generator chain + `NOTE.md`.

Not reconstructable: `implementation.so` (prebuilt x86-64 binary shipped in
`/workdir`; `solution.py` rebuilds it from `implementation.c` when missing) and
the problem-provided `base.py`.

## Algorithm

Single-threaded AVX-512 implementation with **vertical SIMD on split re/im
storage** (128-byte blocks of 8 complex values). Every transform is
hand-written: L=6 is a Good–Thomas PFA(2,3) of DFT2/DFT3; L=8 a hand-coded
split-radix-style DFT8; L=13/17/23 direct symmetric (cos/sin-folded) prime
DFTs, FMA-bound, k-blocked 4+4+3 over conjugate output pairs with fully
unrolled j-loops and embedded-broadcast table loads (`_mm512_set1_pd` from
memory); L=36/45 Good–Thomas PFA(4,9)/PFA(5,9) with DFT9 = 3×3 Cooley–Tukey;
L=64 an 8×8 Cooley–Tukey of two DFT8 layers with 49 twiddles, restructured
load-first for cache-miss overlap (plus `k*d` "direct" no-copy variants for
L1-hot strip buffers). Two batching schemes are dispatched per (L, B): **BL**
(batch-lane, SIMD lanes = 8 independent volumes; small L) and **PV**
(per-volume, lanes = 8 z-samples with in-register 8×8 transposes and
overlap-strips so L∤8 needs no y-padding; L = 36/45/64 and batch remainders),
with a hybrid split of B into groups of 8 (BL) plus a PV tail below per-L
thresholds. Each iteration is two sweeps (x-line FFTs, then per-x-plane y+z
FFTs followed by the `+c`/map pass); the map `z/(1+|z|)` uses
`rsqrt14`+2 Newton and `rcp14`+2 Newton with 4-way ILP (`map_blocks`).
Twiddle/cos/sin tables are built once at import in 80-bit long double with
exact mod-L reduction. Explicit software prefetch, NT stores, software
pipelining and plane-orientation flipping were all tried and measured slower;
the shipped code is the plain-hardware-prefetch variant.

## Compile command

As used by the graded wrapper (and by the final in-`/workdir` rebuild):

```
gcc -O3 -march=native -fno-math-errno -fno-trapping-math -shared -fPIC \
    implementation.c -o implementation.so -lm
```

## Session peculiarity (matters for provenance)

At 02:08:56 a rate-limit storm discarded the container; the attempt was
requeued and the retained transcript covered only the first 53 actions
(through ~01:37). Everything the agent did between ~01:38 and 02:08 (a
different GEN_PV rewrite reading `c` straight from the numpy buffer, a PV2
"orientation-flip" scheme, a k17 no-unroll variant, an earlier hybrid
dispatch) was **lost** and is *not* in the graded file; after the replay the
agent re-derived a simpler architecture, which is what shipped and is what
`dev_generators/` reproduces.

## Verification performed here

- The generator chain replays with **every exact-match assert passing**
  (p14, p16, p21, p25 assert full-text anchors — this cross-checks the
  reconstructed intermediate states against the session's).
- `python3 -m py_compile solution.py` OK.
- `implementation.c`: braces 144/144 and parens 754/754 balanced; 22/22
  structural checks (final macros, dispatch, buffer offsets) present; zero
  stale pre-crash artifacts; `cc -E` (with stub intrinsics header)
  preprocesses with no errors. 685 lines (agent reported "~650 lines").
- **Not compiled** here: AVX-512 source, ARM host.
