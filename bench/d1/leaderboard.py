"""Aggregate a benchmark round into a leaderboard.

Each cell is measured in several independent processes; each process reports the minimum of
its own samples.  The statistic across those processes is the MEDIAN, not the minimum.

That is a deliberate change from the original min-of-mins ("the least-disturbed measurement
of the same fixed work"), for a measured reason.  The within-process precision is excellent
-- an m=1 cell uses ~1.2M inner repetitions and reports sd/min of 0.5% -- but the spread
ACROSS processes, which is core placement and cache state rather than the kernel, is 9.0%
median for single-call cells against 1.1% for chained ones.  min-of-N walks downward as N
grows, so it is not comparable between rounds run with different run counts, and single-call
cells are now run 3x more often than chained ones precisely because they are the noisy ones.
The median is stable under changing N, so a round's numbers mean the same thing as the last
round's.  The min is still printed alongside, and so is the inter-run spread.

A cell whose panel-vs-library gap is smaller than the measurement spread is marked
UNRESOLVED rather than being reported as a win or a loss -- at 9% noise, calling a 1.05x a
victory is not a measurement.  A backend that failed correctness is shown but never ranked.
"""
import argparse, glob, json, math, os, re, statistics
from collections import defaultdict

# A CELL is (L, batch, m).  The chain length is part of the identity, not an attribute:
# cases.txt pairs every (L, B) with both a m=1 and a chained variant, so keying cells on
# (L, B) alone pools two measurements that differ by five orders of magnitude.  That is
# not a cosmetic bug -- it took per-call from the m=1 file and divided it by the chained
# file's m, printing "0.000 us", "9658067129 GF/s" and a 5.15e7% run spread, which is
# exactly (13.1ms - 25.8ns)/25.8ns.
CELLRE = re.compile(r"^(?P<name>.+)_L(?P<L>\d+)_B(?P<B>\d+)(?:_m(?P<m>\d+))?$")


def cell_of(basename):
    """'<name>_L<L>_B<B>[_m<M>]' -> (L, B, m, name), or None.  Files predating the _m
    naming fix carry no chain tag and are read as m=1."""
    mo = CELLRE.match(basename)
    if not mo:
        return None
    return (int(mo["L"]), int(mo["B"]), int(mo["m"] or 1), mo["name"])

p = argparse.ArgumentParser()
p.add_argument("--round", required=True)
p.add_argument("--markdown", default=None, help="also write a markdown table here")
a = p.parse_args()

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", a.round)

timings = defaultdict(list)   # (L, batch, m, name) -> [per_execute_min per run]
medians = defaultdict(list)
setups, descriptions = {}, {}
for path in glob.glob(os.path.join(root, "t_*.json")):
    with open(path) as f:
        try:
            d = json.load(f)
        except json.JSONDecodeError:
            continue
    if not d.get("supported", False):
        continue
    # The chain length comes from inside the JSON, which is authoritative; the filename
    # tag only has to agree.  If it does not, the file was mis-named and pooling it would
    # corrupt a cell, so say so loudly rather than guessing.
    m = int(d.get("chain", 1) or 1)
    fromname = cell_of(os.path.basename(path)[2:-5].rsplit("_r", 1)[0])
    if fromname and fromname[2] != m:
        print(f"   WARNING: {os.path.basename(path)} says chain={m} but is named _m{fromname[2]}")
    key = (d["L"], d["batch"], m, d["name"])
    timings[key].append(d["per_execute_seconds"]["min"])
    medians[key].append(d["per_execute_seconds"]["median"])
    setups[key] = d.get("setup_seconds", float("nan"))
    descriptions[d["name"]] = d.get("description", "")

checks = {}
for path in glob.glob(os.path.join(root, "c_*.json")):
    cell = cell_of(os.path.basename(path)[2:-5])
    if cell is None:
        continue
    with open(path) as f:
        try:
            checks[cell] = json.load(f)
        except json.JSONDecodeError:
            pass

onestep = {}
for path in glob.glob(os.path.join(root, "o_*.json")):
    cell = cell_of(os.path.basename(path)[2:-5])
    if cell is None:
        continue
    with open(path) as f:
        try:
            onestep[cell] = json.load(f)
        except json.JSONDecodeError:
            pass

# Final chain gate, per (L, B): the chain is chaotic, so the tolerance is anchored to the
# honest divergence MEASURED ON THIS VERY CHAIN -- the largest drift among the library
# backends (they are independent correct fp64 implementations) and check.py's numpy-pair
# anchor -- times 300 (30x for a solver legally at the 1e-14 one-step ceiling, 10x slop),
# floored at 1e-10, rounded up onto a {1,3}x10^n grid. See docs/GRADER.md. The one-step
# 1e-14 gate (o_*.json, when the sweep produced it) carries the precision contract.
LIB_BACKENDS = ("mkl_dfti", "mkl2026_dfti", "fftw3_estimate", "fftw3_measure", "fftw3_guru",
                "fftw3_patient", "ducc0_c2c")
def chain_gate(L, B, m):
    # Anchored within the cell: a gate must be calibrated on the SAME chain length, since
    # divergence grows with m.  Pooling m=1 and m=200000 anchors would set the short chain's
    # tolerance from the long chain's drift and wave through anything.
    anchors = [checks[k].get("anchor_rel_l2", 0.0) or 0.0
               for k in checks if k[0] == L and k[1] == B and k[2] == m]
    lib = [checks[(L, B, m, n)].get("chain_rel_l2") for n in LIB_BACKENDS
           if (L, B, m, n) in checks and checks[(L, B, m, n)].get("chain_rel_l2") is not None]
    raw = max([300.0 * a for a in anchors] + [300.0 * r for r in lib] + [1e-10])
    exp = math.floor(math.log10(raw))
    mant = raw / 10 ** exp
    return (1.0 if mant <= 1.0 else (3.0 if mant <= 3.0 else 10.0)) * 10 ** exp

def verdict_ok(L, B, m, name):
    chk = checks.get((L, B, m, name))
    if not chk:
        return None
    ok = bool(chk.get("ok"))
    if "chain_rel_l2" in chk:
        rel = chk["chain_rel_l2"]
        ok = bool(chk.get("rel_l2", 1.0) < chk.get("tol", 1e-12)) and \
             bool(rel == rel and rel < chain_gate(L, B, m))
        one = onestep.get((L, B, m, name))
        if one is not None:
            ok = ok and bool(one.get("one_ok"))
    return ok

env = ""
env_path = os.path.join(root, "environment.txt")
if os.path.exists(env_path):
    env = open(env_path).read().strip()

lines = []
def emit(s=""):
    lines.append(s)
    print(s)

emit(f"=== round {a.round} ===")
if env:
    emit(env)
emit()

cases = sorted({(L, B, m) for (L, B, m, _) in timings})
for (L, B, m) in cases:
    # THIS IS A 1D HARNESS: one transform is L points, not L^3.  The 3D formula was carried
    # over when bench/d1 was cloned from the 3D tree and made every derived figure wrong --
    # L=13 was reported as volume 2197 at 5540 GF/s, where the truth is volume 13 at ~11 GF/s.
    # Ranking was by time and so survived, but no GF/s or working-set number printed before
    # this fix should be quoted.
    volume = L
    # One timed unit is a CHAIN of m transforms of B volumes; per-transform figures and
    # the flop yardstick must divide by both, or a chained case reads m times too slow.
    nominal = 5.0 * volume * math.log2(volume) * B * m
    label = "non-batched" if B == 1 else f"batched B={B}"
    label += ", single call" if m == 1 else f", chain m={m}"
    working_set = 2 * 16 * volume * B / 1024**2   # in + out, MiB
    emit(f"-- L={L} ({label}), working set {working_set:.3f} MiB --")
    emit(f"   {'backend':<24} {'median us/xf':>14} {'best us/xf':>12} {'GF/s':>8} "
         f"{'spread':>9} {'runs':>4} {'setup':>8}  correctness   (? = gap inside the noise)")
    rows = []
    for (l, b, mm, name), runs in timings.items():
        if (l, b, mm) != (L, B, m):
            continue
        best = statistics.median(runs)          # stable under changing run counts
        fastest = min(runs)
        spread = (max(runs) - fastest) / fastest if fastest > 0 else float("nan")
        chk = checks.get((L, B, m, name))
        ok = verdict_ok(L, B, m, name)
        rows.append((best, name, spread, ok, chk, fastest, len(runs)))
    rows.sort()
    fastest_ok = next((r[0] for r in rows if r[3]), None)
    best_spread = next((r[2] for r in rows if r[3]), 0.0) or 0.0
    for best, name, spread, ok, chk, fastest, nr in rows:
        if chk and "chain_rel_l2" in chk:
            one = onestep.get((L, B, m, name))
            ostr = (" 1s=%.0e" % one["one_rel_l2"]) if one and "one_rel_l2" in one else ""
            verdict = ("%s ch=%.1e/%.0e%s" % ("ok" if ok else "FAILED",
                       chk["chain_rel_l2"], chain_gate(L, B, m), ostr))
        else:
            verdict = ("ok %.1e" % chk["rel_l2"]) if ok else (
                "FAILED %.1e" % chk["rel_l2"] if chk else "unchecked")
        if fastest_ok and ok:
            ratio = best / fastest_ok
            # Half the spread is a rough 1-sigma on each side; a gap inside that is noise.
            noise = max(spread, best_spread) / 2.0
            rel = f"{ratio:.2f}x" + ("?" if abs(ratio - 1.0) <= noise else "")
        else:
            rel = "--"
        emit(f"   {name:<24} {best/(B*m)*1e6:11.4f} us {fastest/(B*m)*1e6:11.4f} us "
             f"{nominal/best/1e9:8.2f} {spread*100:8.1f}% {nr:3d}r "
             f"{setups[(L,B,m,name)]:7.3f}s  {verdict:<16} {rel}")
    emit()

if descriptions:
    emit("backends:")
    for name in sorted(descriptions):
        emit(f"   {name:<24} {descriptions[name]}")

if a.markdown:
    with open(a.markdown, "w") as f:
        f.write("```\n" + "\n".join(lines) + "\n```\n")
