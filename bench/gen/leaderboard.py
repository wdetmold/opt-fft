"""Aggregate a benchmark round into a leaderboard.

Timing is taken as the MINIMUM over independent processes of each process's minimum
sample: the least-disturbed measurement of the same fixed work.  The spread across
processes is reported alongside, so a suspiciously lucky run is visible rather than
hidden.  A backend that failed correctness is shown but never ranked.
"""
import argparse, glob, json, math, os
from collections import defaultdict

p = argparse.ArgumentParser()
p.add_argument("--round", required=True)
p.add_argument("--markdown", default=None, help="also write a markdown table here")
a = p.parse_args()

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", a.round)

timings = defaultdict(list)   # (L, batch, name) -> [per_execute_min per run]
medians = defaultdict(list)
chains = {}                   # (L, batch) -> chain length m (1 when not a chained case)
setups, descriptions = {}, {}
for path in glob.glob(os.path.join(root, "t_*.json")):
    with open(path) as f:
        try:
            d = json.load(f)
        except json.JSONDecodeError:
            continue
    if not d.get("supported", False):
        continue
    key = (d["L"], d["batch"], d["name"])
    chains[(d["L"], d["batch"])] = int(d.get("chain", 1) or 1)
    timings[key].append(d["per_execute_seconds"]["min"])
    medians[key].append(d["per_execute_seconds"]["median"])
    setups[key] = d.get("setup_seconds", float("nan"))
    descriptions[d["name"]] = d.get("description", "")

checks = {}
for path in glob.glob(os.path.join(root, "c_*.json")):
    base = os.path.basename(path)[2:-5]           # <name>_L<L>_B<B>
    name, _, rest = base.rpartition("_L")
    L, _, B = rest.partition("_B")
    with open(path) as f:
        try:
            checks[(int(L), int(B), name)] = json.load(f)
        except (json.JSONDecodeError, ValueError):
            pass

onestep = {}
for path in glob.glob(os.path.join(root, "o_*.json")):
    base = os.path.basename(path)[2:-5]
    name, _, rest = base.rpartition("_L")
    L, _, B = rest.partition("_B")
    with open(path) as f:
        try:
            onestep[(int(L), int(B), name)] = json.load(f)
        except (json.JSONDecodeError, ValueError):
            pass

# Final chain gate, per (L, B): the chain is chaotic, so the tolerance is anchored to the
# honest divergence MEASURED ON THIS VERY CHAIN -- the largest drift among the library
# backends (they are independent correct fp64 implementations) and check.py's numpy-pair
# anchor -- times 300 (30x for a solver legally at the 1e-14 one-step ceiling, 10x slop),
# floored at 1e-10, rounded up onto a {1,3}x10^n grid. See docs/GRADER.md. The one-step
# 1e-14 gate (o_*.json, when the sweep produced it) carries the precision contract.
LIB_BACKENDS = ("mkl_dfti", "mkl2026_dfti", "fftw3_estimate", "fftw3_measure",
                "fftw3_patient", "ducc0_c2c")
def chain_gate(L, B):
    anchors = [checks[k].get("anchor_rel_l2", 0.0) or 0.0
               for k in checks if k[0] == L and k[1] == B]
    lib = [checks[(L, B, n)].get("chain_rel_l2") for n in LIB_BACKENDS
           if (L, B, n) in checks and checks[(L, B, n)].get("chain_rel_l2") is not None]
    raw = max([300.0 * a for a in anchors] + [300.0 * r for r in lib] + [1e-10])
    exp = math.floor(math.log10(raw))
    mant = raw / 10 ** exp
    return (1.0 if mant <= 1.0 else (3.0 if mant <= 3.0 else 10.0)) * 10 ** exp

def verdict_ok(L, B, name):
    chk = checks.get((L, B, name))
    if not chk:
        return None
    ok = bool(chk.get("ok"))
    if "chain_rel_l2" in chk:
        rel = chk["chain_rel_l2"]
        ok = bool(chk.get("rel_l2", 1.0) < chk.get("tol", 1e-12)) and \
             bool(rel == rel and rel < chain_gate(L, B))
        one = onestep.get((L, B, name))
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

cases = sorted({(L, B) for (L, B, _) in timings})
for (L, B) in cases:
    volume = L ** 3
    m = chains.get((L, B), 1)
    # One timed unit is a CHAIN of m transforms of B volumes; per-transform figures and
    # the flop yardstick must divide by both, or a chained case reads m times too slow.
    nominal = 5.0 * volume * math.log2(volume) * B * m
    label = "non-batched" if B == 1 else f"batched B={B}"
    if m > 1:
        label += f", chain m={m}"
    working_set = 2 * 16 * volume * B / 1024**2   # in + out, MiB
    emit(f"-- L={L} ({label}), volume {volume}, working set {working_set:.2f} MiB --")
    emit(f"   {'backend':<24} {'per-transform':>14} {'per-call':>12} {'GF/s':>8} "
         f"{'run spread':>11} {'setup':>9}  correctness")
    rows = []
    for (l, b, name), runs in timings.items():
        if (l, b) != (L, B):
            continue
        best = min(runs)
        spread = (max(runs) - best) / best if best > 0 else float("nan")
        chk = checks.get((L, B, name))
        ok = verdict_ok(L, B, name)
        rows.append((best, name, spread, ok, chk))
    rows.sort()
    fastest_ok = next((r[0] for r in rows if r[3]), None)
    for best, name, spread, ok, chk in rows:
        if chk and "chain_rel_l2" in chk:
            one = onestep.get((L, B, name))
            ostr = (" 1s=%.0e" % one["one_rel_l2"]) if one and "one_rel_l2" in one else ""
            verdict = ("%s ch=%.1e/%.0e%s" % ("ok" if ok else "FAILED",
                       chk["chain_rel_l2"], chain_gate(L, B), ostr))
        else:
            verdict = ("ok %.1e" % chk["rel_l2"]) if ok else (
                "FAILED %.1e" % chk["rel_l2"] if chk else "unchecked")
        rel = f"{best / fastest_ok:.2f}x" if (fastest_ok and ok) else "--"
        emit(f"   {name:<24} {best/(B*m)*1e6:11.3f} us {best*1e6:9.3f} us "
             f"{nominal/best/1e9:8.2f} {spread*100:9.1f}% "
             f"{setups[(L,B,name)]:8.3f}s  {verdict:<16} {rel}")
    emit()

if descriptions:
    emit("backends:")
    for name in sorted(descriptions):
        emit(f"   {name:<24} {descriptions[name]}")

if a.markdown:
    with open(a.markdown, "w") as f:
        f.write("```\n" + "\n".join(lines) + "\n```\n")
