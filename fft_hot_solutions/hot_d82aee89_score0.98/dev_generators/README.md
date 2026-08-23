# dev_generators — hot_d82aee89

The attempt's own generator tree (lived in `/tmp/w/dev/gen` in the container),
extracted verbatim from `attempt.log` heredocs with every logged edit replayed
in chronological order, plus the merge machinery from `/tmp/w/regen`.

## Contents

- `glib.py` — shared emitter utilities: 80-bit-longdouble trig → hex-double
  baking, `Emit` class, AVX-512 prelude (V macros, IDX/PERM2 tables, MAP2
  all-FMA rsqrt14/rcp14+Newton map). Final state (IDX block added, MAP2 tiny
  guard dropped).
- `gen36.py` / `gen36d.py` — L=36 within-volume padded-row engine v3 (PFA 4×9,
  staged A/B with SCRA scratch, TR8 z-pass) + driver. Dev-line prototype;
  superseded by `genwv.py`. `gen36.py` also exports `emit_dft4`/`emit_dft3`/`S3`
  used by the other generators.
- `gensoa.py`, `gen36soa.py` — SoA-8 conversion codelets and the abandoned
  L=36 SoA prototype.
- `genprime.py` — folded symmetric prime DFT codelets p=13/17/23, cos/sin
  split phases, USCR/PSCR scratch, loop-inside-function form (final state).
- `gensmall.py` — DFT6 (PFA 2×3) and DFT8 (DIF) straight-line codelets.
- `gensoadrv.py` — SoA-8 driver. NOTE: final container state is an abandoned,
  buggy rewrite that clobbered the working version AFTER `emitted/implsoa.c`
  was built; the emitted C corresponds to the pre-clobber state (the replay
  reproduces this ordering).
- `gensoadrv2.py` + `build68.py` — padded-XS SoA driver v2 for L=6/8
  (`emitted/impl68.c`, `run_6v2`/`run_8v2`); measured behind b00, unused.
- `genwv.py` + `buildwv.py` + `tr8.inc` — within-volume engines for 36/45/64
  (PFA 4×9 / 9×5, CT 8×8; split-c arenas; prefetch experiment reverted to
  pf=0). Emits `emitted/implwv.c` (`run_36wv/45wv/64wv`); measured behind the
  adopted engines, unused in the final dispatch.
- `buildsoa.py` — emits `emitted/implsoa.c` (SoA engines L=6,8,13,17,23).
- `harness.py` — the dev check/timing harness (numpy reference).
- `merge.py` — FINAL (v3) merge script: parts = [s81, f30, d43, b00-pragma-
  wrapped], compile-error-driven prefix renaming. In-container it read
  `/tmp/w/...` paths; the replay ran a path-remapped copy.
- `dispatch_v3.inc` — the final dispatch layer appended to merged.c (verbatim
  from the log).
- `replay/` — the reconstruction tooling used on this machine: `replay.py`
  (extracts+replays all generator writes/edits from attempt.log), `replay2.py`
  (solution.py / NOTES.md), and `gcc-shim/` (Apple-clang cross-target
  `-fsyntax-only` shim + stub libc headers used to drive merge.py's
  duplicate-symbol probe off-host).

## Rebuilding implementation.c

```
python3 replay/replay.py                      # writes generator tree + parts, runs build*
# stage: regen/mine.c (d43 gen.py), regen/s81.c (s81 gen_impl.py),
#        regen/f30.c (v6_3f30d81f implementation.c verbatim),
#        b00/implementation.c (build_full.py output)
PATH=<gcc-shim>:$PATH python3 merge_exec.py   # merge.py with paths remapped
cat dispatch_v3.inc >> merged.c               # -> implementation.c (3,297,697 B)
```

The four prior-round parts come from the repo copies in
`fft_v5v6_solutions/{v5_8175a973_score0.90, v6_3f30d81f_score0.88}` and
`fft_warm_solutions/{warm_d43251c2_score0.99, warm_00291a90_score0.97}`;
local regeneration of all three generated parts is byte-identical to the repo
pre-emitted copies.
