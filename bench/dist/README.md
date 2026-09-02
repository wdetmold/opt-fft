# bench/dist — the small-node communication fraction of a distributed 3D FFT

Started:  2026-09-02
Owner:    fft project (survey follow-up)

## Why this exists

`docs/literature_dist/00-SURVEY.md` §3 established the Amdahl ceiling on this project's
central claim, and found a hole in it. Every published comm-vs-compute breakdown of a
distributed 3D FFT is at **four nodes or more**:

| configuration | local 1D FFT | pack/unpack | MPI |
|---|---|---|---|
| 1024³, 4 CPU nodes, 168 Power9 cores, Isend/Irecv | 9.3% | 5.6% | 84.3% |
| 1024³, same, MPI_Alltoall | 5.7% | 56.4% | 37.9% |
| 256³, 16 ranks, 4 nodes, five libraries | 11–13% | 29–46% | 41–77% |
| 1024³ fp64, 64 A100 | ~4% | 22% | 89% |

At those fractions a 3.5× faster local batched-1D kernel — the thing this project has —
buys **1.07–1.09× end-to-end**. The survey's conclusion was that our kernel advantage
survives only at small node counts, on CPUs, and in batched/chained regimes. But the
**1–2-node CPU fraction was never measured by anyone**: neither ICL report decomposes
anything below 4 nodes (their Figs. 4.1/4.2 give total time only), and the often-quoted
"communication ~50% at 512 cores" figure turned out to be absent from both reports.

So this directory measures it, on our own hardware, because the entire distributed pitch
rests on a number the literature does not contain.

## How

`ext/build_heffte_trace.sh` builds heFFTe 2.4.1 with `Heffte_ENABLE_TRACING=ON`, which logs
every transform stage with `MPI_Wtime` into one text file per rank. The instrumented events
(`src/heffte_compute_transform.cpp`) give exactly the split we need:

- **`fft-1d`** / `fft-1d x3` — the local batched-1D executor call. *This is the slot our
  kernel would occupy*, so its share is the Amdahl denominator.
- **`reshape`** — the transpose: pack + MPI + unpack **together**. This build cannot
  separate those three the way the ICL Vampir traces did, and no conclusion here may
  pretend otherwise.
- `copy`, `reshape/copy`, `scale` — local data movement outside the transpose.

The rank ladder is the point of the design:

- **1 rank** does no MPI at all, so its `reshape` time is *pure local pack/copy*. That
  pins the local-only composition and lets the multi-rank `reshape` growth be attributed
  to communication.
- **2–64 ranks on one node** keeps MPI inside shared memory.
- **two nodes** puts the transpose on the fabric — the number we are actually after.

Runs use `-nruns5 -no-error` so the trace contains only the timed transforms, and each
configuration runs in its own subdirectory because heFFTe writes its logs to the CWD under
a name that depends only on backend/precision/grid.

## This directory is finished when

`docs/literature_dist/00-SURVEY.md` §3 carries a measured 1-node and 2-node local-FFT
fraction for Ice Lake (`axxxl`), with the derived Amdahl cap for a 3.5× local kernel, and
the survey's "must be measured ourselves" caveat is replaced by that number.

## Where the result goes

`docs/literature_dist/00-SURVEY.md` §3 (the table above gains our own rows), and
`results/<tag>/FRACTIONS.md` here as the raw record.

## Caveats that must travel with any number from here

- `reshape` lumps pack, MPI and unpack. Do not quote an "MPI %" from this harness.
- heFFTe does not runtime-dispatch its AVX-512 kernels: the `avx512` build SIGILLs on the
  Haswell login node and on the `prod`/`devel` nodes. Use `avx2` there, `avx512` on `axxxl`.
- The MPI stack is a first-order term (the survey records 20–40% swings from MPI choice
  alone), so `environment.txt` records the module and the launcher for every run.
