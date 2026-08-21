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
    nominal = 5.0 * volume * math.log2(volume) * B
    label = "non-batched" if B == 1 else f"batched B={B}"
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
        ok = chk["ok"] if chk else None
        rows.append((best, name, spread, ok, chk))
    rows.sort()
    fastest_ok = next((r[0] for r in rows if r[3]), None)
    for best, name, spread, ok, chk in rows:
        verdict = ("ok %.1e" % chk["rel_l2"]) if ok else (
            "FAILED %.1e" % chk["rel_l2"] if chk else "unchecked")
        rel = f"{best / fastest_ok:.2f}x" if (fastest_ok and ok) else "--"
        emit(f"   {name:<24} {best/B*1e6:11.3f} us {best*1e6:9.3f} us "
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
