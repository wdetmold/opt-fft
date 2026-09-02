#!/usr/bin/env python3
"""Aggregate heFFTe trace logs into the comm-vs-compute table.

Each traced rank writes lines of `name(w40) start(w20) duration(w20)`, where the name may
itself contain spaces ("fft-1d x3", "reshape/copy") -- so fields are split from the RIGHT.

The reported fraction is of *traced* time, per rank, then averaged over ranks.  We deliberately
do NOT divide by wall time: some of the benchmark's wall time (planning, allocation, the
error check) sits outside any traced region, and mixing the two would silently shrink every
fraction.  The benchmark's own reported per-run time is carried alongside as a cross-check.

What the buckets mean, and what they do not:
  fft     "fft-1d", "fft-1d x3"  -- the local batched-1D executor call.  THE slot our kernel
                                    would occupy, so its share is the Amdahl denominator.
  reshape "reshape"              -- the transpose: pack + MPI + unpack TOGETHER.  This build
                                    cannot separate them; never quote an "MPI %" from here.
  local   "copy", "reshape/copy", "scale" -- local data movement outside the transpose.
At 1 rank there is no MPI at all, so that row's `reshape` is pure local pack/copy: it is the
control that makes the multi-rank reshape growth attributable to communication.
"""
import argparse
import glob
import os
import re
import sys
from collections import defaultdict

FFT = ("fft-1d", "fft-1d x3")
LOCAL = ("copy", "reshape/copy", "scale")

p = argparse.ArgumentParser()
p.add_argument("--tag", required=True)
p.add_argument("--speedup", type=float, default=3.5,
               help="local-kernel speedup whose end-to-end Amdahl cap is reported")
a = p.parse_args()

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", a.tag)
if not os.path.isdir(root):
    sys.exit(f"no such results directory: {root}")

CONF = re.compile(r"^g(?P<grid>\d+)_r(?P<ranks>\d+)_n(?P<nodes>\d+)_(?P<algo>\w+)$")


def parse_rank_log(path):
    """-> {bucket: seconds} for one rank."""
    got = defaultdict(float)
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                dur = float(parts[-1])
            except ValueError:
                continue
            name = " ".join(parts[:-2]).strip()
            if name in FFT:
                got["fft"] += dur
            elif name == "reshape":
                got["reshape"] += dur
            elif name in LOCAL:
                got["local"] += dur
            else:
                got["other:" + name] += dur
    return got


def reported_time(d):
    """The benchmark's own per-run time, for cross-check against the traced total."""
    out = os.path.join(d, "stdout.txt")
    if not os.path.isfile(out):
        return None
    for line in open(out, errors="replace"):
        m = re.search(r"Time per run:\s*([0-9.eE+-]+)", line)
        if m:
            return float(m.group(1))
    return None


rows = []
for d in sorted(glob.glob(os.path.join(root, "g*_r*_*"))):
    m = CONF.match(os.path.basename(d))
    if not m:
        continue
    logs = sorted(glob.glob(os.path.join(d, "*.log")))
    if not logs:
        continue
    per_rank = [parse_rank_log(p_) for p_ in logs]
    n = len(per_rank)
    agg = defaultdict(float)
    for r in per_rank:
        for k, v in r.items():
            agg[k] += v / n          # mean over ranks
    total = sum(agg.values())
    if total <= 0:
        continue
    rows.append(dict(grid=int(m["grid"]), ranks=int(m["ranks"]), nodes=int(m["nodes"]),
                     algo=m["algo"],
                     nlogs=n, fft=agg["fft"], reshape=agg["reshape"], local=agg["local"],
                     other=sum(v for k, v in agg.items() if k.startswith("other:")),
                     total=total, reported=reported_time(d),
                     othernames=sorted(k[6:] for k in agg if k.startswith("other:"))))

if not rows:
    sys.exit(f"no parseable trace directories under {root} -- the runs produced no logs")

rows.sort(key=lambda r: (r["grid"], r["ranks"], r["nodes"], r["algo"]))
nodes_note = ""
env = os.path.join(root, "environment.txt")
if os.path.isfile(env):
    for line in open(env):
        if line.startswith(("date:", "nodelist:", "cpu:", "cores/node:", "mpi:")):
            nodes_note += "  " + line.rstrip() + "\n"

print(f"=== comm-vs-compute fractions: {a.tag} ===")
if nodes_note:
    print(nodes_note.rstrip())
print()
print("  grid ranks nodes algo     fft%  reshape%  local%   traced_s  per_run_s   "
      f"cap@{a.speedup:g}x  cap_if_pack_absorbed")
for r in rows:
    f_fft = r["fft"] / r["total"]
    f_res = r["reshape"] / r["total"]
    f_loc = r["local"] / r["total"]
    # Amdahl: speeding up only the fft-1d slot by `speedup`.
    cap = 1.0 / ((1.0 - f_fft) + f_fft / a.speedup)
    # ...and the design question the survey raised: if a fused SoA kernel also absorbs the
    # LOCAL data movement (copy/scale, and the pack/unpack halves of reshape that we cannot
    # separate here), the addressable fraction grows.  This second column is therefore an
    # UPPER bound: it credits the kernel with all of `local`, which is measurable, and none
    # of `reshape`, which is not separable in this build.
    f_addr = f_fft + f_loc
    cap2 = 1.0 / ((1.0 - f_addr) + f_addr / a.speedup)
    rep = f"{r['reported']:9.4f}" if r["reported"] is not None else "        -"
    print(f"  {r['grid']:4d} {r['ranks']:5d} {r['nodes']:5d} {r['algo']:<5s} "
          f"{100*f_fft:7.2f} {100*f_res:8.2f} {100*f_loc:7.2f} "
          f"{r['total']:10.4f} {rep}   {cap:7.3f}  {cap2:7.3f}")

extra = sorted({n for r in rows for n in r["othernames"]})
if extra:
    print("\n  unbucketed event names present in the traces:", ", ".join(extra))
print("""
  fft%     = local batched-1D executor call -- the slot our kernel would occupy
  reshape% = transpose: pack + MPI + unpack TOGETHER (not separable in this build)
  local%   = copy/scale outside the transpose
  cap      = end-to-end Amdahl limit from making ONLY the fft slot faster
  cap_if_pack_absorbed = same, crediting the kernel with `local` too -- an UPPER bound,
             since the pack/unpack inside `reshape` cannot be separated here
  The 1-rank row does no MPI, so its reshape column is pure local pack/copy.""")
