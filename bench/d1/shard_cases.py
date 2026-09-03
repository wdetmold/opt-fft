#!/usr/bin/env python3
"""Split a case list across grading shards, balanced by ESTIMATED WALL TIME.

Sharding only speeds grading up if the shards actually finish together, and the obvious
proxies do not: dealing by L alone is a stable sort that puts every single-call cell on one
node and every chained cell on the other, and an analytic L*log2(L)*B*m proxy ignores the
fixed floor each cell pays (samples x min-sample-ms x runs, for every backend) so it lumps
40 cheap cells against 12 expensive ones.

So use the previous round's own measurements when they exist.  For each cell, per backend:
    3 runs x setup_seconds            (FFTW patient planning alone is ~58 s at L=65537)
  + runs x samples x max(min_sample_ms, per_call)
which is what the sweep actually spends.  With no prior round, fall back to the analytic
proxy -- balanced by greedy longest-processing-time either way.

usage: shard_cases.py --cases FILE --shards N [--prev-round TAG] [--runs 3] [--samples 12]
prints "<shard> <case>" per line.
"""
import argparse, glob, json, math, os, sys
from collections import defaultdict

p = argparse.ArgumentParser()
p.add_argument("--cases", required=True)
p.add_argument("--shards", type=int, required=True)
p.add_argument("--prev-round", default=None)
p.add_argument("--runs", type=int, default=3)
p.add_argument("--runs-once", type=int, default=None,
               help="runs for m=1 cells (they are measured more often; see sweep.sh)")
p.add_argument("--samples", type=int, default=12)
p.add_argument("--min-sample-ms", type=float, default=20.0)
a = p.parse_args()
RUNS_ONCE = a.runs_once or a.runs * 3

root = os.path.dirname(os.path.abspath(__file__))
measured = defaultdict(float)
if a.prev_round:
    for f in glob.glob(os.path.join(root, "results", a.prev_round, "t_*_r*.json")):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if not d.get("supported", False):
            continue
        m = int(d.get("chain", 1) or 1)
        per = d["per_execute_seconds"]["min"]
        setup = d.get("setup_seconds", 0.0) or 0.0
        # one backend's share of this cell, over all its runs
        nr = RUNS_ONCE if m == 1 else a.runs
        cost = nr * setup + nr * a.samples * max(a.min_sample_ms / 1e3, per)
        # files are per run, so divide by the count that PRODUCED them (the previous round
        # may have used a different run count than the one we are costing for)
        measured[(d["L"], d["batch"], m)] += cost / max(1, len(
            glob.glob(os.path.join(root, "results", a.prev_round,
                                   f"t_{d['name']}_L{d['L']}_B{d['batch']}_m{m}_r*.json"))))

cases = []
for line in open(a.cases):
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    parts = line.split(":")
    L, B = int(parts[0]), int(parts[1])
    m = int(parts[2]) if len(parts) > 2 else 1
    key = (L, B, m)
    if key in measured:
        cost = measured[key]
    else:
        # analytic fallback, floored the same way the driver floors a sample
        per = 5.0 * L * math.log2(max(L, 2)) * B * m / 2e10       # ~20 GFlop/s
        nr = RUNS_ONCE if m == 1 else a.runs
        cost = 15 * nr * a.samples * max(a.min_sample_ms / 1e3, per)
    cases.append((cost, line))

cases.sort(key=lambda x: -x[0])
load = [0.0] * a.shards
out = []
for cost, line in cases:                 # greedy longest-processing-time
    i = min(range(a.shards), key=lambda j: load[j])
    load[i] += cost
    out.append((i + 1, line))
for sh, line in out:
    print(sh, line)
tot = sum(load)
print("# estimated shard wall times (s): "
      + "  ".join(f"{i+1}:{v:.0f}" for i, v in enumerate(load))
      + f"   imbalance {max(load)/ (tot/a.shards):.2f}x"
      + ("   [from measured " + a.prev_round + "]" if measured else "   [analytic fallback]"),
      file=sys.stderr)
