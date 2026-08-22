# What this repository keeps, and why

The project generates far more material than is worth versioning: twelve implementer
agents rewrite `bench/geom/impl/` every round, each benchmark round writes hundreds of
megabytes of input and output volumes, and the external libraries occupy about 2 GB.
The rule is that **anything reconstructible from a script plus a seed is not tracked**,
and **anything that took judgement to produce is**.

## Tracked

| What | Where | Why |
|---|---|---|
| Reference and textbook FFT | `python/slow_dft.py`, `python/fft3d.py` | the from-scratch code and the correctness ground truth |
| Test suites | `python/test_fft3d.py`, `python/verify_backends.py` | 34 checks against the definition; backend validation |
| Benchmark harness | `bench/geom/{fft3d_api.h,driver.c,gen_input.py,check.py,leaderboard.py,Makefile,sweep.sh,submit.sh}` | the measurement apparatus — every number depends on it |
| Library baselines | `bench/geom/sota/` | FFTW ×3 planner levels, MKL 2022 + 2026, ducc0 |
| **Per-round sources** | `bench/geom/impl_<N>/` | what each generation actually wrote. Round N works in `impl_N`, seeded from `impl_{N-1}`; `impl` is a symlink to the current one |
| Harness floor | `impl_<N>/baseline_matrix.c` | the library-free reference implementation the harness is validated with |
| Panel machinery | `bench/geom/{PANEL_BRIEF.md,panel_round.js,promote.sh}` | the brief, the workflow, the promotion tool |
| **Strategy records** | `bench/geom/strategies/*.md` | the panel's memory: what was tried, what it measured, what failed and why |
| **Promoted exemplars** | `bench/geom/exemplars/<round>/` | the implementations worth showing the next panel |
| Round results | `bench/geom/results/*/leaderboard.txt`, `environment.txt`, `failures.txt`, `build_errors.txt` | the measurements, plus the machine they were taken on |
| Literature corpus | `docs/literature/`, `docs/LITERATURE.md` | ~7,500 cited lines; expensive to reproduce |
| Documentation | `docs/`, `README.md` | survey, textbook-FFT notes, this file |
| External build scripts | `ext/build_*.sh`, `ext/requirements.txt`, `env.sh` | rebuild all of `ext/install` and the venv from scratch |

## Not tracked

* `ext/src/`, `ext/install/`, `venv/` — ~2 GB of third-party source and binaries.
  `ext/build_fftw.sh`, `ext/build_heffte.sh`, `ext/build_venv.sh` and
  `ext/requirements.txt` rebuild all of it. Note `pyvkfft` needs
  `VKFFT_BACKEND=cuda` (no OpenCL headers on this system) and `mpi4py` must be built
  from source against the CUDA-aware OpenMPI — both handled in `build_venv.sh`.
* `*.bin` — input and output volumes, regenerated from `--seed` every round.
* `bench/geom/build/<hostname>/` — binaries and objects, per build host. This directory
  is on a shared filesystem and is built from two machines with `-march=native` (the
  Haswell login node and the Cascade Lake benchmark node), so the artifacts must not
  share a path: a node-built object linked into a login-node binary SIGILLs, which
  silently invalidated an MKL baseline once before the layout was split per host.
* Per-round raw data: the several hundred `t_*.json` / `c_*.json` files each round
  writes, and the verbose logs (`*.log`, `*.err`, `slurm-*.out`). The leaderboard is
  computed from them and a round is re-runnable from its seed, so only the leaderboard
  and the environment it was measured in are kept.
* Login-node smoke rounds (`results/local_smoke/`) — the login node is shared, so its
  timings are not measurements; those runs only prove the pipeline works.

## Why per-round source directories exist

`panel_r1`'s eleven implementations are **gone**. Its monitor died before it could promote
anything, `impl/` was a single shared scratch directory, and round 2's implementers rewrote
all eleven files in place. Only the strategy records and the leaderboard survive for that
round.

So each round now gets its own directory: round N's implementers work in `impl_N`, seeded
from `impl_{N-1}`, and `impl` is a symlink to the current round so the Makefile, `sweep.sh`,
`tryout.sh`, `promote.sh` and `probe_node.sh` keep working unchanged. Every generation's code
is preserved whether or not anyone remembered to promote it, and a regression can be
diffed against the exact source that produced the previous round's numbers.

`exemplars/` remains the *curated* view — the entries judged worth putting in front of the
next panel, with their records and measured numbers alongside. `impl_N/` is the complete
provenance; `exemplars/` is the reading list.

## Promoting exemplars from a panel round

The point of keeping examples is that the next panel starts from real code rather than
from prose. After the monitor reports a round:

```bash
cd bench/geom
./promote.sh panel_r1 L17_rader L8_batchsimd L36_pfa      # copies code + strategy record
git add exemplars/panel_r1 strategies && git commit
```

Promote on these grounds, in order:

1. **The fastest correct entry for each geometry.** One per L, always.
2. **A structurally different runner-up** — if the winner at L=17 is Rader and a dense
   conjugate-symmetric kernel came within ~20%, keep both: the next panel needs to see
   the alternative actually written down, not described.
3. **Instructive failures.** An entry that was slower for a *documented and measured*
   reason is worth keeping, because it stops the next panel spending a round rediscovering
   it. Promote it and say so in the round note.
4. **Anything that beat a library baseline**, regardless of rank.

Do not promote near-duplicates of an already-promoted entry, entries that failed
correctness, or entries whose strategy record is missing — the record is what makes the
code useful later.

Each promoted round gets a `NOTES.md` written at promotion time recording what the round
established and what the next one should attack. `promote.sh` creates the skeleton.
