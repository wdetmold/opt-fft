# Generator provenance

`implementation.c` is fully auto-generated, self-written code produced by the
scripts in this directory (run offline during development; NOT needed at
grading time):

- `gencore.py` — expression-DAG builder with CSE/simplification + high-precision
  (mpmath, 60 digits) twiddle constants, correctly rounded to double.
- `dftgen.py`  — DFT plan search: direct-symmetric (odd n), Good-Thomas PFA,
  Cooley-Tukey, and Rader (prime p via length p-1 cyclic convolution), with
  automatic op-count-based plan selection.
- `gen2.py` / `gen2b.py` — staged codelet emitters (straight-line leaf codelets
  through L1 scratch), fused iteration drivers (z,y,x passes + elementwise map
  + next-iteration FFT start fused into single memory sweeps), batch-of-8 lane
  mode, transposes, conversions, prefetch, and the run_size() dispatcher.
- `header_c.h` — hand-written runtime support (8x8 transposes, de/interleave,
  map helpers, allocator).

To regenerate: `python3 gen2b.py` (writes implementation.c).

No FFT library code is used anywhere: all DFT arithmetic is generated from
first principles (the DFT definition + PFA/CT/Rader factorizations).
