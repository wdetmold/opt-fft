# Dev artifacts (generator + tests)

- `ir.py`      — expression DAG builder + DFT construction (PFA / Cooley-Tukey /
                 symmetric-folded prime DFT), constants in long double.
- `emit.py`    — C codelet emitter (stage-structured, AVX-512 intrinsics).
- `prime_gen.py` — register-tiled looped codelets for primes 17/23.
- `gen_impl.py`  — assembles ../implementation.c (codelets + drivers + dispatch).
- `harness.py`, `bench*.py`, `e2e*.py`, `bigval.py`, `ratio_bench.py` — tests.

`implementation.c` in the parent directory is the generated single-file
deliverable; regenerate with `python3 gen_impl.py` (writes ./implementation.c,
then copy up).
