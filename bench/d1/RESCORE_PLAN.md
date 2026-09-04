# Deferred: re-score r1–r4 so the whole campaign is scored one way

**Status:** RUNNING (2026-09-04, chained behind the r7 recovery scoring).

**Deviation from the plan below, deliberate:** no top-up mode was implemented. The 26
once-cells are remeasured **fresh at 9 runs** from each round's own `impl_N`, at ~40 min per
round sharded instead of ~26 — bought with zero new harness mechanism, after three sharding /
resume code paths introduced this week each cost a round or a cell before being fixed. A
fresh remeasure also puts all nine runs in one session instead of splicing 3 old + 6 new.
Chained cells are untouched (the case file is m=1 only; filenames carry _m). The impl
symlink is swapped per round and restored to impl_8 afterwards — the campaign is over, so
the Makefile IMPL parameterization in step 1 is no longer needed and was not done.

## Why this exists

The d1 grading metric changed partway through the campaign, for a measured reason. Run-to-run
spread on the same node, over r3's 52 cells:

| regime | median | p90 | max |
|---|---|---|---|
| chained | 1.1% | 16.2% | 24% |
| single-call (m=1) | **9.0%** | 22.3% | 65% |

The driver is not at fault: an m=1 cell uses ~1.2M inner repetitions and reports 0.5%
*within-process* sd. The variance is **between independent processes** — core placement and
cache state — which 3 runs only partly absorb. So single-call verdicts near parity were not
measurements. r3's L=64 B=512/once was reported as a 1.11× win with ±18% of noise around it.

From r5 onward, m=1 cells get **9 runs** instead of 3, and the ranking statistic is the
**median** over runs rather than the minimum (min-of-N drifts downward as N grows, so it is
not comparable between rounds run with different run counts).

Will's requirement: *"we do want a consistent scoring mechanism in the end."* The
round-over-round trend is what the campaign is judged on, and it is only meaningful if every
round was measured the same way.

## Scope

**r1, r2, r3 AND r4** — four rounds. r4 was scored under the old 3-run setting, so it needs
the same top-up as r1–r3; it is easy to forget because the change landed during it.

The median statistic needs nothing: `leaderboard.py` recomputes it from the stored per-run
JSONs, so it already applies retroactively. Only the **run count** has to be topped up.

## Cost

Adding the 6 missing runs to the 26 once-cells of each round, from those rounds' own measured
per-call and setup times:

| | one node | sharded across both |
|---|---|---|
| per round | ~52 min | ~26 min |
| four rounds | ~3.5 h | ~1.7 h |

A full remeasure instead of a top-up would be ~5.3 h / ~2.7 h, so implement top-up mode.

## What it buys — and what it does not

21 cells are currently unresolved across r1–r3 (7 each). At 9 runs the band shrinks ~√3:
roughly **9 of 21 gain a definite verdict**. The rest persist — but their gaps are 0.97×,
0.99×, 1.01×, 1.02×, 1.04×, i.e. **genuine ties**, and no run count turns a 1% gap into a win.
Demonstrating a tie is the correct result for those, and better than the current "?".

One outlier needs more than 9: r3's L=64 B=512/once, an 11% gap inside an ±18% band, still
±10% at 9 runs. It is a single cell, so a targeted deeper top-up (~27 runs) on whatever
remains borderline costs minutes.

## Why not now

Two build collisions with the live campaign, both needing small additive changes first:

1. `Makefile` hardcodes `impl/` (lines 30, 31, 47) — the symlink the live round uses.
   Needs `IMPL ?= impl` and `$(IMPL)/`, default unchanged, plus a `--impl DIR` passthrough
   in `sweep.sh` so a past round can be built from its own `impl_N`.
2. `build/$(hostname -s)/bin` is shared, so re-scoring would overwrite the binaries the live
   round is scoring with. Needs a build-dir suffix.

Both nodes are busy during scoring now that grading shards across them, so there is no idle
capacity to hide this in while r5/r6 run.

## Steps, once r6 has scored

1. Parameterize `Makefile` (`IMPL`, build suffix) and add `sweep.sh --impl DIR`.
2. Add top-up mode: within a unit, skip run indices whose `t_*_r<N>.json` already parses.
   This also makes ordinary resume cheaper.
3. Per round R in r1..r4: build from `impl_R`, run the 26 once-cells with `--runs-once 9`
   and `--cases cases_once.txt`, sharded across both nodes, writing runs 4–9 alongside the
   existing 1–3.
4. Targeted deeper top-up (27 runs) on any cell still inside its noise band.
5. Regenerate every round's leaderboard and the trend table; state plainly which cells are
   resolved wins, resolved losses, and demonstrated ties.

## This file is finished when

Every d1 round from r1 to r6 reports 9 runs on its single-call cells under the median
statistic, and the campaign trend table is regenerated from that uniform basis. Then delete
this file.
