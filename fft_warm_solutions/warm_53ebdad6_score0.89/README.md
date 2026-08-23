# warm attempt 53ebdad6 — score 0.89

Warm-start attempt on `fft3d-fixed-geometry-opt-warm` (single-core iterated
batched 3D complex FFTs, 8 fixed cube sizes L = 6,8,13,17,23,36,45,64;
per-size chain tolerances; SOTA = MKL DFTI; roofline r\* = 0.137). This
attempt was given the 16 prior cold-start solutions + `IMPLEMENTATION_NOTES.md`
as background and iterated on the strongest of them.

Run: `53ebdad6-0e0a-48e4-8e5c-316dbceac...` (short `53ebdad6`). Graded on an
`n2-standard-128` (Intel, AVX-512) firecracker container.

## (1) Grading result

| quantity | value |
|---|---|
| **score** | **0.8915023950887556** |
| C_ref (trusted reference / base, wall s) | 47.08170117500049 |
| C_sota (held-out MKL DFTI, wall s) | 13.07060541699866 |
| C_opt (this attempt, best-of-shots wall s) | 3.150501449999865 |
| ratio_vs_sota = C_opt / C_sota | 0.24104 (≈ 4.15× faster than MKL DFTI) |
| ratio_vs_base = C_opt / C_ref | 0.066916 (≈ 14.94× faster than the reference) |
| correctness checks | format ✓  constraint ✓  content ✓ |

Grader `walls` arrays (3 shots each):
- ref:  `[47.2497, 47.0817, 50.0696]`
- sota: `[13.0706, 13.0805, 13.4028]`
- opt:  `[3.1505, 3.1627, 3.1601]`

`C_opt` is the best (min) opt shot, 3.1505 s.

## (2) Files

- **`solution.py`** — the mandated ctypes wrapper (graded). Reconstructed from
  the whole-file heredoc at log L5156 with the compile-flag sed edit at L5395
  applied (adds `-fno-stack-protector -falign-loops=32`). Adds glibc `mallopt`
  M_MMAP_THRESHOLD / M_TRIM_THRESHOLD → 1 GiB so numpy's fresh per-call x0/c
  arrays stay on the reusable heap (the agent measured ~25% wall saved by
  killing mmap/munmap page-fault storms between timed calls); pooled output
  buffers; import-time `_warmup()`. Otherwise the skeleton verbatim.
- **`implementation.c`** — single self-contained C file, all transform
  arithmetic hand-written (includes only `immintrin/stdint/stdlib/string/
  sys/mman`; no FFT-library or thread headers — Rules 1 & 2 clean).
  **Generator-emitted:** produced by replaying the final patch chain
  `patch1 → patch3 → patch5 → patch7 → patch12 → patch13 → patch14` on the
  v6_f40c5e25 (score-0.91) base `implementation.c`, followed by the final
  documentation-header rewrite (log L7286). 6670 lines. All 8 `run_L` entry
  points present, prime kernels `dftp13/17/23_v` spliced in, `buildc_{36,45,64}`
  and the mapcol odd-tail fix present, file ends cleanly at `run_64`.
- **`dev_generators/`** — the generator + patch scripts used to emit
  `implementation.c`, reconstructed at their final in-session state
  (last-write-wins over all in-place edits):
  - `genp.py` — emits `primekerns.h` (register-resident phase-split prime
    DFT kernels for 13/17/23), needs `mpmath` at generation time.
  - `primekerns.h` — the generated prime kernels (1644 lines), regenerated
    here by `genp.py`.
  - `patch1.py` (splice primekerns + redirect calls), `patch3.py` (36/45/64
    swapped-c build + mapcol odd-tail), `patch5.py` (36/45/64 S-pass
    prefetch/strip layout — the most-edited script, ~12 in-place revisions,
    final state = post-L6462), `patch7.py`, `patch12.py`, `patch13.py`
    (P_23 c-stride rework), `patch14.py`.
- **`attempt.log`** — the raw Taiga environment-logs stream (source of truth).

**Compile check (STEP 4):** `gcc -O3 -march=native -c implementation.c` FAILS
on this host — it is an arm64 Mac and the code is x86 AVX-512
(`immintrin.h: "only meant to be used on x86 and x64 architecture"`). This is
a host-architecture mismatch, not a defect; the file compiled and graded
cleanly in the x86 container.

## (3) Approach

Warm start from the best prior attempt (v6_f40c5e25, ratio ≈ 0.27 vs base on
the old curve) and rebuilt its hot paths rather than starting fresh. Kept the
inherited architecture: palindromic axis orders giving one full-data pass per
step (slab pass over the two in-plane axes + pencil pass over the plane axis,
each finishing step *t* and pre-transforming *t*+1 around the fused map);
lane-major SoA batching (8 volumes per zmm group, pure vertical SIMD, no
shuffles) for L ≤ 23; within-volume split re/im planes with padded row strides
(40/56/72) and 8×8 register transposes for L = 36/45/64. **Changed:** the
primes 13/17/23 were re-emitted (via `genp.py`) as register-resident
phase-split folded-DFT sweeps — all (p−1)/2 cos/sin constants held in
registers, j-outer accumulation, partials parked in L1 scratch, zero in-sweep
constant loads (~4h² FMA/pencil); the 36/45/64 map was made maskless and
sequential by storing the transposed *c* copy split-re/im in
strip-consumption order, and the pencil pass consumes *c* from a
column-consumption-ordered copy (one fewer copy); prefetch was retuned
empirically (kept only L=64 next-plane T1 and distance-2 csw strips in the
map); and a latent out-of-bounds read in the reconstructed prior code
(`mapcol_45` processing odd plane counts in pairs) was fixed with an odd-tail
handler. Twiddles baked as extended-precision hex literals; the z/(1+|z|) map
uses rsqrt14+Newton/Heron and rcp14+Newton, full fp64 throughout to clear the
per-size chain gates.

## Reconstruction fidelity

`implementation.c` here is **functionally faithful but not byte-identical** to
the graded artifact. It was regenerated off-host by replaying the session's
generator/patch scripts against this project's local copy of the v6_f40c5e25
base (the log never dumps the final C in full). Consequences: (a) baked
twiddle/trig constants may differ in the last ulp if the local base or `mpmath`
build differs from the container's — the correctness gates are the arbiter, not
byte-equality; (b) the leading documentation-comment block begins at column 0
here vs a single leading blank line in-container (cosmetic); (c) two patch5
edits that the agent applied then reverted (log L6635/L6674) net to identity and
are replayed as such, and one patch5 edit that raised an AssertionError
in-container (L6562) is correctly a no-op. The replay ran the full chain with
zero patch failures. md5 of the reconstructed files:
`implementation.c 0a15aa31…`, `solution.py 38d7150c…`.
