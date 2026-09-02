#!/usr/bin/env python3
"""Aggregate heFFTe trace logs into the comm-vs-compute table.

Each traced rank writes lines of `name(w40) start(w20) duration(w20)`, where the name may
itself contain spaces ("fft-1d x3", "irecv 32768 from 7") -- so fields split from the RIGHT.

TWO structural facts about these logs decide whether the numbers mean anything:

1. THE EVENTS NEST.  heFFTe logs a `reshape` whose interval CONTAINS its own children --
   `packing`, `all2allv` (or `isend`/`irecv`/`waitany` on the p2p paths), `unpacking`:

       packing    0.006253894  0.001183209
       all2allv   0.007438285  0.001184270
       unpacking  0.008623309  0.001390913
       reshape    0.006247413  0.003767152    <- parent: 1.183+1.184+1.391 = 3.758 ms
       fft-1d     0.010018140  0.001651293

   Summing every line therefore counts the transpose twice and silently shrinks every
   fraction.  Nesting is resolved here STRUCTURALLY, by interval containment, rather than
   from a hand-maintained list of parent names that a heFFTe update could invalidate.

2. THE WARMUP IS IN THE LOG.  `mark warmup begin` at t=0 precedes the untimed warmup
   transform; the measured runs start at the first `mark forward begin`.  Everything before
   that is dropped, or a cold first pass inflates the local-FFT share.

Because the children are logged too, this harness CAN separate pack / MPI / unpack -- which
is the split the ICL Vampir traces report, so the rows here are directly comparable to them.
The `reshape` parent is used only as a consistency check on its children.

Buckets, and what each one is for:
  fft      "fft-1d", "fft-1d x3"     -- the local batched-1D executor call.  THE slot our
                                        kernel would occupy: its share is the Amdahl denominator.
  pack     "packing", "self packing" -- gather into the send buffer
  mpi      "all2allv", "isend ...", "irecv ...", "waitany" -- the wire
  unpack   "unpacking", "unpacking from N", "self unpacking"
  local    "copy", "reshape/copy", "scale" -- data movement outside the transpose
At 1 rank there is no MPI at all, so that row is the pure-local control: it pins what the
pack/unpack cost when nothing crosses a network, making the multi-rank growth attributable.
"""
import argparse
import glob
import os
import re
import sys
from collections import defaultdict

FFT = {"fft-1d", "fft-1d x3"}
PACK = {"packing", "self packing"}
UNPACK = {"unpacking", "self unpacking"}
MPI_EXACT = {"all2allv", "alltoall", "waitany"}
LOCAL = {"copy", "reshape/copy", "scale"}
PARENT = {"reshape"}


def bucket_of(name):
    if name in FFT:
        return "fft"
    if name in PACK:
        return "pack"
    if name in UNPACK or name.startswith("unpacking from"):
        return "unpack"
    if name in MPI_EXACT or name.startswith(("isend ", "irecv ")):
        return "mpi"
    if name in LOCAL:
        return "local"
    if name in PARENT:
        return "_parent"
    return "_unknown:" + name


def parse_rank_log(path):
    """-> ({bucket: seconds}, parent_seconds, unknown_names, dropped_warmup_events)."""
    events = []
    markers = []
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                start = float(parts[-2])
                dur = float(parts[-1])
            except ValueError:
                continue
            name = " ".join(parts[:-2]).strip()
            if name.startswith("mark "):
                markers.append((name, start))
                continue
            events.append((start, dur, name))

    # Drop the warmup: measurement begins at the first "mark forward begin".
    t0 = None
    for name, start in markers:
        if name == "mark forward begin":
            t0 = start
            break
    dropped = 0
    if t0 is not None:
        before = len(events)
        events = [e for e in events if e[0] >= t0]
        dropped = before - len(events)

    # Resolve nesting by containment: sweep in start order, parents before children when
    # starts tie (longer duration first), keeping a stack of still-open intervals.  A parent's
    # start precedes its first child's, so start order puts containers first.
    #
    # Each `reshape` needs its children's total, not just a nested/top-level flag, because two
    # cases would otherwise silently lose time:
    #   * at 1 rank a reshape has NO children at all (nothing is sent) -- it is a leaf local
    #     copy, and discarding it as "a container" wrongly inflated fft% to ~96%;
    #   * on the p2p paths the isend/irecv/waitany children account for only 70-80% of their
    #     reshape, and that residual is real time spent inside the transpose.
    # So a childless reshape counts as local movement, and any residual is reported in its own
    # column rather than folded into mpi% -- we do not know what it is, so we do not claim to.
    events.sort(key=lambda e: (e[0], -e[1]))
    got = defaultdict(float)
    parent_s = 0.0
    unknown = set()
    stack = []          # [end_time, is_reshape, children_sum]
    def close(frame):
        _end, is_reshape, kids, dur = frame
        if not is_reshape:
            return
        if kids <= 0.0:
            got["local"] += dur         # childless reshape == pure local copy (1 rank)
        else:
            residual = dur - kids
            if residual > 0.0:
                got["resh_other"] += residual
    for start, dur, name in events:
        while stack and stack[-1][0] <= start:
            close(stack.pop())
        b = bucket_of(name)
        if stack:
            stack[-1][2] += dur          # this event is part of its parent's cost
        if b == "_parent":
            parent_s += dur
            stack.append([start + dur, True, 0.0, dur])
            continue
        stack.append([start + dur, False, 0.0, dur])
        if b.startswith("_unknown:"):
            unknown.add(b[9:])
            b = "local"          # count it somewhere honest rather than dropping the time
        got[b] += dur
    while stack:
        close(stack.pop())
    return got, parent_s, unknown, dropped


def reported(d):
    """(per-transform seconds, run count) as the benchmark itself reports them.

    This is the independent yardstick for the whole accounting: speed3d times `nruns`
    forward plus `nruns` backward transforms, so a complete, non-double-counted bucketing
    must satisfy traced_total == 2 * nruns * per_run.  That closure is the reason to trust
    the percentages -- without it, a mis-resolved nesting would still produce a tidy table.
    """
    out = os.path.join(d, "stdout.txt")
    if not os.path.isfile(out):
        return None, None
    per, nruns = None, None
    for line in open(out, errors="replace"):
        m = re.search(r"Time per run:\s*([0-9.eE+-]+)", line)
        if m:
            per = float(m.group(1))
        m = re.search(r"Num runs:\s*(\d+)", line)
        if m:
            nruns = int(m.group(1))
    return per, nruns


p = argparse.ArgumentParser()
p.add_argument("--tag", required=True)
p.add_argument("--speedup", type=float, default=3.5,
               help="local-kernel speedup whose end-to-end Amdahl cap is reported")
p.add_argument("--markdown", default=None)
a = p.parse_args()

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", a.tag)
if not os.path.isdir(root):
    sys.exit(f"no such results directory: {root}")

CONF = re.compile(r"^g(?P<grid>\d+)_r(?P<ranks>\d+)_n(?P<nodes>\d+)_(?P<algo>\w+)$")

rows, all_unknown = [], set()
for d in sorted(glob.glob(os.path.join(root, "g*_r*_n*_*"))):
    m = CONF.match(os.path.basename(d))
    if not m:
        continue
    logs = sorted(glob.glob(os.path.join(d, "*.log")))
    if not logs:
        continue
    agg = defaultdict(float)
    parent_s = 0.0
    dropped = 0
    for lp in logs:
        got, ps, unknown, dr = parse_rank_log(lp)
        for k, v in got.items():
            agg[k] += v / len(logs)          # mean over ranks
        parent_s += ps / len(logs)
        all_unknown |= unknown
        dropped += dr
    total = sum(agg.values())
    if total <= 0:
        continue
    rows.append(dict(grid=int(m["grid"]), ranks=int(m["ranks"]), nodes=int(m["nodes"]),
                     algo=m["algo"], nlogs=len(logs), total=total, parent=parent_s,
                     reported=reported(d)[0], nruns=reported(d)[1], dropped=dropped,
                     **{k: agg[k] for k in ("fft", "pack", "mpi", "unpack", "local",
                                             "resh_other")}))

if not rows:
    sys.exit(f"no parseable trace directories under {root} -- the runs produced no logs")

rows.sort(key=lambda r: (r["grid"], r["ranks"], r["nodes"], r["algo"]))

header = ""
env = os.path.join(root, "environment.txt")
if os.path.isfile(env):
    for line in open(env):
        if line.startswith(("date:", "nodelist:", "cpu:", "cores/node:", "mpi:",
                            "probe-summary")):
            header += "  " + line.rstrip() + "\n"

out = []
out.append(f"=== comm-vs-compute fractions: {a.tag} ===")
if header:
    out.append(header.rstrip())
out.append("")
out.append("  grid ranks nds algo     fft%   pack%    mpi%  unpack%  local%  resid%   "
           f"traced_s  per_run_s  cap@{a.speedup:g}x  cap_local")
for r in rows:
    t = r["total"]
    f = {k: r[k] / t for k in ("fft", "pack", "mpi", "unpack", "local", "resh_other")}
    cap = 1.0 / ((1.0 - f["fft"]) + f["fft"] / a.speedup)
    # The survey's design question: if a fused SoA kernel absorbs the transpose pack/unpack
    # as well as the butterflies, the addressable fraction is fft+pack+unpack+local.
    addr = f["fft"] + f["pack"] + f["unpack"] + f["local"]
    cap2 = 1.0 / ((1.0 - addr) + addr / a.speedup)
    rep = f"{r['reported']:9.4f}" if r["reported"] is not None else "        -"
    out.append(f"  {r['grid']:4d} {r['ranks']:5d} {r['nodes']:3d} {r['algo']:<5s} "
               f"{100*f['fft']:7.2f} {100*f['pack']:7.2f} {100*f['mpi']:7.2f} "
               f"{100*f['unpack']:8.2f} {100*f['local']:7.2f} {100*f['resh_other']:7.2f} "
               f"{t:10.4f} {rep}  {cap:8.3f}  {cap2:8.3f}")

# Consistency check that the nesting resolution is right: the reshape parents should account
# for very nearly pack+mpi+unpack.  A large mismatch means the containment sweep mis-grouped
# something, and every fraction above would be wrong -- so it is checked, not assumed.
out.append("")
out.append("  verification -- traced total vs the benchmark's OWN reported time")
out.append("  (speed3d times nruns forward + nruns backward, so closure should be 1.000;")
out.append("   a mis-resolved nesting would still print a tidy table, so this is the check")
out.append("   that the percentages above are actually complete and counted once each)")
worst = 0.0
for r in rows:
    kids = r["pack"] + r["mpi"] + r["unpack"] + r["resh_other"]
    if kids > 0:
        nest = f"nest {r['parent'] / kids:5.3f}"
        nflag = "" if 0.95 <= r["parent"] / kids <= 1.05 else " NEST-MISMATCH"
    else:
        nest = "nest  leaf"          # 1 rank: the reshape sends nothing, so it has no children
        nflag = ""
    if r["reported"] and r["nruns"]:
        expect = 2.0 * r["nruns"] * r["reported"]
        closure = r["total"] / expect
        worst = max(worst, abs(closure - 1.0))
        cflag = "" if 0.97 <= closure <= 1.03 else "   <-- CLOSURE FAIL"
    else:
        closure, cflag = float("nan"), "   <-- no reported time"
    out.append(f"    g{r['grid']:<5d} r{r['ranks']:<4d} n{r['nodes']} {r['algo']:<7s} "
               f"traced {r['total']:9.4f}s  closure {closure:5.3f}  {nest}{nflag}{cflag}")
out.append(f"    worst closure deviation: {100*worst:.2f}%")

if all_unknown:
    out.append("\n  event names not in any bucket (counted as local%): "
               + ", ".join(sorted(all_unknown)))
out.append("""
  fft%     = local batched-1D executor call -- the slot our kernel would occupy
  pack%/unpack% = gather into / scatter out of the communication buffers
  mpi%     = the wire (all2allv, or isend/irecv/waitany on the p2p paths)
  local%   = copy/scale outside the transpose, plus childless reshapes (1-rank local copies)
  resid%   = time inside a reshape not covered by its own child events (large on the p2p
             paths); deliberately NOT folded into mpi%, since what it is was not established
  traced_s = summed over the TIMED runs only; the warmup transform is excluded
  cap      = end-to-end Amdahl limit from speeding up ONLY the fft slot
  cap_local= same if a fused kernel also absorbs pack+unpack+local
  The 1-rank rows do no MPI at all: that is the pure-local control.""")

text = "\n".join(out)
print(text)
if a.markdown:
    with open(a.markdown, "w") as fh:
        fh.write("# Comm-vs-compute fractions: " + a.tag + "\n\n```\n" + text + "\n```\n")
