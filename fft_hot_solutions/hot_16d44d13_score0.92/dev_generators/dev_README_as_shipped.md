Development artifacts (provenance for the audit):
- genlib.py, gen_prime.py/gen_prime2.py, gen_vol*.py, vmodel.py: my own engine
  generators + numpy pipeline models (correct engines; benchmarked slower than
  the adopted prior-work engines; not in the graded path).
- patch64.py: produces the run64_alt alternation driver merged into
  implementation.c (applied; in the graded path).
- impl_m3.c: the exact merged source == ../implementation.c.
- ref.py / test_engine.py / persize_merged.py / cmp_*.py / microbench.py:
  extended-precision reference checker + benchmarks used throughout.
implementation.c composition: warm_00291a90 regenerated-on-host engine
(sizes 6-45) + v6_3f30d81f engine (L=64 kernels) + my new L=64 alternation
driver & div-based map (MAP_STYLE 3) + d43251c2 impl_mine L=13 group engine
with my routing. See ../NOTES.md.
