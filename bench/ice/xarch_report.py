"""Compare our kernels against MKL across microarchitectures, and separate two hypotheses.

Will measured the consolidated "best" library on Ice Lake over L = 6..64 and batches 1..2048
and found MKL ahead by ~1.65x on a time-weighted average. Two very different explanations:

  (A) PORTABILITY failure -- the kernels are tuned for the Cascade Lake node the competition
      scored on, and lose on a different microarchitecture. The fix is cross-architecture
      scoring and runtime dispatch.

  (B) SELECTION failure -- the library shipped the B=1 winner at every geometry, and at large
      batch a different entry of ours is faster. The fix is batch-aware selection, which
      needs no new kernels at all.

These make different predictions, so the data can tell them apart:

  * under (B), "best of our entries at this (L,B)" beats MKL even where the B=1 winner does
    not, and the shortfall is concentrated at large batch;
  * under (A), even our best entry loses, and it loses at B=1 too.

Both are reported per case, and aggregated three ways -- because a single average can be
made to say almost anything here:

  time-weighted   sum(MKL time) / sum(ours):  what a real workload of these cases would feel
  geometric mean  of the per-case ratios:      scale-free, no case dominates
  worst case      the case where we trail most: what a user hits at their size

usage: python3 xarch_report.py results/xarch_icelake [results/xarch_spr ...]
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict

LIBS = ("mkl_dfti", "mkl2026_dfti", "fftw3_estimate", "fftw3_measure",
        "fftw3_patient", "ducc0_c2c")
FLOOR = ("baseline_matrix",)

# What the consolidated library shipped: the non-batched winner at each geometry.
SHIPPED_B1_WINNER = {
    6: "L6_pfa", 8: "L8_batchsimd", 13: "L13_direct", 17: "L17_matrixsimd",
    23: "L23_rader", 36: "L36_mixedradix", 45: "L45_pfa", 64: "L64_blocked",
}


def load(round_dir):
    """(L, batch, name) -> best per-transform seconds over runs; plus correctness."""
    best = {}
    ok = {}
    for path in glob.glob(os.path.join(round_dir, "t_*.json")):
        try:
            d = json.load(open(path))
        except Exception:
            continue
        if not d.get("supported"):
            continue
        key = (d["L"], d["batch"], d["name"])
        t = d["per_transform_seconds_min"]
        if key not in best or t < best[key]:
            best[key] = t
    for path in glob.glob(os.path.join(round_dir, "c_*.json")):
        base = os.path.basename(path)[2:-5]
        name, _, rest = base.rpartition("_L")
        L, _, B = rest.partition("_B")
        try:
            ok[(int(L), int(B), name)] = json.load(open(path)).get("ok", False)
        except Exception:
            pass
    return best, ok


def machine(round_dir):
    p = os.path.join(round_dir, "environment.txt")
    if not os.path.exists(p):
        return round_dir
    for line in open(p):
        if line.startswith("cpu:"):
            return line.split(":", 1)[1].strip()
    return round_dir


def report(round_dir):
    best, ok = load(round_dir)
    if not best:
        print(f"{round_dir}: no results yet")
        return

    cases = sorted({(L, B) for (L, B, _) in best})
    print(f"\n{'=' * 100}\n{machine(round_dir)}   ({round_dir}, {len(cases)} cases)\n{'=' * 100}")
    print(f"  {'L':>3} {'batch':>6} | {'best library':>14} {'us':>9} | "
          f"{'shipped (B=1 win)':>18} {'us':>9} {'vs lib':>7} | "
          f"{'best of ours':>16} {'us':>9} {'vs lib':>7}")

    agg = defaultdict(lambda: {"lib": 0.0, "ship": 0.0, "mine": 0.0,
                               "ratios_ship": [], "ratios_mine": []})
    for (L, B) in cases:
        # A wrong answer is not a result: an entry whose correctness check failed is dropped
        # here, before anything is chosen from it.
        rows = {n: t for (l, b, n), t in best.items()
                if (l, b) == (L, B) and ok.get((L, B, n), True)}
        lib = min(((n, t) for n, t in rows.items() if n in LIBS), key=lambda x: x[1], default=None)
        mine = min(((n, t) for n, t in rows.items() if n not in LIBS and n not in FLOOR),
                   key=lambda x: x[1], default=None)
        ship_name = SHIPPED_B1_WINNER.get(L)
        ship = (ship_name, rows[ship_name]) if ship_name in rows else None
        if not lib or not mine:
            continue

        r_ship = lib[1] / ship[1] if ship else float("nan")
        r_mine = lib[1] / mine[1]
        print(f"  {L:>3} {B:>6} | {lib[0]:>14} {lib[1]*1e6:9.3f} | "
              f"{(ship[0] if ship else '-'):>18} {(ship[1]*1e6 if ship else 0):9.3f} {r_ship:6.2f}x | "
              f"{mine[0]:>16} {mine[1]*1e6:9.3f} {r_mine:6.2f}x")

        for scope in ("all", f"B={'1' if B == 1 else '>1'}"):
            a = agg[scope]
            a["lib"] += lib[1] * B
            a["mine"] += mine[1] * B
            a["ratios_mine"].append(r_mine)
            if ship:
                a["ship"] += ship[1] * B
                a["ratios_ship"].append(r_ship)

    def geo(v):
        if not v:
            return float("nan")
        p = 1.0
        for x in v:
            p *= x
        return p ** (1.0 / len(v))

    print()
    for scope in ("all", "B=1", "B=>1"):
        a = agg.get(scope)
        if not a or not a["ratios_mine"]:
            continue
        print(f"  [{scope:>5}] time-weighted   shipped {a['lib']/a['ship'] if a['ship'] else float('nan'):5.2f}x   "
              f"best-of-ours {a['lib']/a['mine']:5.2f}x   (>1 means WE are faster)")
        print(f"  [{scope:>5}] geometric mean  shipped {geo(a['ratios_ship']):5.2f}x   "
              f"best-of-ours {geo(a['ratios_mine']):5.2f}x")
        print(f"  [{scope:>5}] worst case      shipped {min(a['ratios_ship']) if a['ratios_ship'] else float('nan'):5.2f}x   "
              f"best-of-ours {min(a['ratios_mine']):5.2f}x")


if __name__ == "__main__":
    dirs = sys.argv[1:] or ["results/xarch_icelake"]
    for d in dirs:
        report(d)
