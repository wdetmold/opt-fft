#!/usr/bin/env python3
"""Replay the hot_d82aee89 attempt's generator-file writes/edits from attempt.log.

Extracts heredoc bodies and `python3 - <<'EOF'` edit scripts verbatim from the
transcript and applies them in chronological order, so the reconstructed
generator tree is byte-faithful to what the attempt had in its container.
"""
import re, os, sys, subprocess

S = "/private/tmp/claude-501/-Users-wdetmold-MIT-Dropbox-William-Detmold-2026-problems/3e8ea67f-c989-4b31-8f6e-316caac734df/scratchpad"
LOG = "/Users/wdetmold/MIT Dropbox/William Detmold/2026_problems/opt-fft/fft_hot_solutions/hot_d82aee89_score0.98/attempt.log"
W = S + "/w"

log = open(LOG).read()

# ---- split into command blocks (tags carry a namespace prefix; build dynamically)
NS = "antml" + ":"
OPEN = "<" + NS + 'parameter name="command">'
CLOSE = "</" + NS + "parameter>"
cmds = []
pos = 0
while True:
    a = log.find(OPEN, pos)
    if a < 0:
        break
    b = log.find(CLOSE, a)
    cmds.append(log[a + len(OPEN):b])
    pos = b
print(f"{len(cmds)} commands parsed")

HD = re.compile(r"cat >(>?) *([^\s]+) <<'EOF'\n(.*?)\n^EOF$", re.S | re.M)
PY = re.compile(r"python3 - <<'EOF'\n(.*?)\n^EOF$", re.S | re.M)

def find_cmd(marker, after_idx=0):
    for i, c in enumerate(cmds):
        if i >= after_idx and marker in c:
            return i, c
    raise KeyError(marker)

def apply_blocks(cmdtext, cwd, only=None):
    """Apply heredoc writes and pyedits in positional order."""
    events = []
    for m in HD.finditer(cmdtext):
        events.append((m.start(), 'cat', m))
    for m in PY.finditer(cmdtext):
        events.append((m.start(), 'py', m))
    events.sort()
    for _, kind, m in events:
        if kind == 'cat':
            append, path, body = m.group(1) == '>', m.group(2), m.group(3)
            if os.path.basename(path) == 'merged.c':
                continue  # dispatch appends handled separately after merge runs
            if only and os.path.basename(path) not in only:
                continue
            # remap absolute /tmp/w paths to scratchpad
            if path.startswith('/tmp/w'):
                path = W + path[len('/tmp/w'):]
            elif not path.startswith('/'):
                path = os.path.join(cwd, path)
            else:
                print(f"    SKIP absolute non-/tmp/w target {path}")
                continue
            mode = 'a' if append else 'w'
            with open(path, mode) as f:
                f.write(body + "\n")
            print(f"    wrote{'(append)' if append else ''} {os.path.relpath(path, W)} ({len(body)}B)")
        else:
            script = m.group(1)
            if only and only != 'py':
                # pyedits always run unless only= restricts to cat targets
                pass
            r = subprocess.run([sys.executable, '-'], input=script, text=True,
                               cwd=cwd, capture_output=True)
            if r.returncode != 0:
                print("    PYEDIT FAILED:\n", r.stdout[-2000:], r.stderr[-2000:])
                raise SystemExit(1)
            if r.stdout.strip():
                print("    pyedit out:", r.stdout.strip()[:200])

DEVGEN = W + "/dev/gen"
DEV = W + "/dev"
REGEN = W + "/regen"
os.makedirs(DEVGEN, exist_ok=True)

steps = [
    # (marker, cwd, note)
    ("cd /tmp/w/dev && cat > harness.py", DEV, "harness.py"),
    ("cd /tmp/w/dev/gen && cat > glib.py", DEVGEN, "glib.py"),
    ('"""L=36 engine: within-volume split re/im, padded rows (36->40)', DEVGEN, "gen36 v1"),
    ('src = src.replace("def emit_dft36_vert(e, addr, stride_expr', DEVGEN, "gen36 v1 fix"),
    ('"""Driver for L=36 engine."""', DEVGEN, "gen36d v1"),
    ('"""L=36 kernels v2: low-register-pressure staged PFA 4x9', DEVGEN, "gen36 v2 (+gen36d v2)"),
    ("# move IDX/PERM2 defs into glib PRELUDE", DEVGEN, "glib IDX edit"),
    ("# make stage functions noinline instead of always_inline", DEVGEN, "noinline edit"),
    ("# glib: drop max guard in MAP2", DEVGEN, "glib MAP2 edit + gen36 v3 + gen36d v3"),
    ('"""    double* p1 = o1 ? o1 : om;', DEVGEN, "gen36d snapshot fix"),
    ('"""Generic SoA-8-volume engine: one-sweep-per-step, all passes vertical."""', DEVGEN, "gensoa.py"),
    ('"""L=36 SoA-8 prototype."""', DEVGEN, "gen36soa.py"),
    ('s = s.replace("""        for (long t = 1; t <= m; t++) {', DEVGEN, "gen36soa snapshot fix"),
    ('"""Folded symmetric prime DFT codelets (p=13,17,23), cos/sin split phases', DEVGEN, "genprime.py final base"),
    ('"""DFT6 (PFA 2x3) and DFT8 codelets, straight-line, SoA vertical, in-place safe."""', DEVGEN, "gensmall.py + gensoadrv.py"),
    ("import gensoa, gensoadrv, genprime, gensmall", DEVGEN, "buildsoa.py"),
    ("# rewrite phase S to store u as well; phase C reads u from scratch.", DEVGEN, "genprime USCR + buildsoa USCR"),
    ('e("VS(USCR + 384, VL(pr)); VS(USCR + 392, VL(pi));', DEVGEN, "genprime x0 save"),
    ("# genprime: wrap body in internal loop over pencils", DEVGEN, "loop-wrap edits (genprime/buildsoa/gensoadrv)"),
    ('"""Within-volume engine for L in {36,45,64}: split re/im padded rows', DEVGEN, "genwv.py base"),
    ("def gen_engine(L):\n    \"\"\"Full within-volume engine for L.\"\"\"", DEVGEN, "genwv append + buildwv.py + tr8.inc"),
    ("# 1) Bm tail path: use masked second load + both stores", DEVGEN, "genwv tail fix"),
    ("# 1) add c split arenas + conversion, used by Bm variants", DEVGEN, "genwv c-split"),
    ("# prefetch next-visit lines in stage A", DEVGEN, "genwv prefetch add"),
    ("src = src.replace('stage_A(stride, pf=RS if tag == \"ps\" else 0)', 'stage_A(stride, pf=0)')", DEVGEN, "genwv prefetch revert"),
]

idx = 0
for marker, cwd, note in steps:
    i, c = find_cmd(marker, idx)
    print(f"[{i}] {note}")
    apply_blocks(c, cwd)
    idx = i  # allow same command index to match later markers (rare); keep monotone

# ---- emit implsoa.c (pre-clobber gensoadrv state) and implwv.c
print("== emit implsoa.c")
r = subprocess.run([sys.executable, "buildsoa.py"], cwd=DEVGEN, capture_output=True, text=True)
print(r.stdout.strip(), r.stderr.strip()[:500])
print("== emit implwv.c")
r = subprocess.run([sys.executable, "buildwv.py"], cwd=DEVGEN, capture_output=True, text=True)
print(r.stdout.strip(), r.stderr.strip()[:500])

# ---- post-implsoa clobber of gensoadrv.py (abandoned rewrite, final container state)
for marker, cwd, note in [
    ('"""Generic SoA-8 driver for one L: arenas, conversions, sweeps, run entry. XS = padded x-block stride (doubles)."""', DEVGEN, "gensoadrv abandoned rewrite (final state)"),
    ('"""SoA-8 driver v2 with padded x-block stride XS (doubles)."""', DEVGEN, "gensoadrv2.py + build68.py"),
]:
    i, c = find_cmd(marker, idx)
    print(f"[{i}] {note}")
    apply_blocks(c, cwd)
    idx = i

print("== emit impl68.c")
r = subprocess.run([sys.executable, "build68.py"], cwd=DEVGEN, capture_output=True, text=True)
print(r.stdout.strip(), r.stderr.strip()[:500])

# ---- merge.py: base + 4 edits (located by content, whole-log search)
def find_one(pred, note):
    hits = [i for i, c in enumerate(cmds) if pred(c)]
    assert len(hits) >= 1, note
    return hits[0], cmds[hits[0]]

merge_steps = [
    (lambda c: '"""Merge adopted engines + my engines into one C file' in c, "merge.py base", "base"),
    (lambda c: "u2018" in c, "merge.py quote-regex fix", "py"),
    (lambda c: "grep -oP '^void \\K\\w+' mine.c" in c, "merge.py expd43 regex", "py"),
    (lambda c: "merged_v2.so" in c and "parts = [" in c, "merge.py v2 edit", "py"),
    (lambda c: "tail -c 2000 merged.c" in c, "merge.py v3 pragma edit", "py"),
]
for pred, note, kind in merge_steps:
    i, c = find_one(pred, note)
    print(f"[{i}] {note}")
    if kind == "py":
        m = PY.search(c)
        r = subprocess.run([sys.executable, '-'], input=m.group(1), text=True,
                           cwd=REGEN, capture_output=True)
        assert r.returncode == 0, r.stderr
    else:
        for m in HD.finditer(c):
            if m.group(2) == 'merge.py':
                open(REGEN + "/merge.py", "w").write(m.group(3) + "\n")
                print("    wrote regen/merge.py", len(m.group(3)))

# save the dispatch heredoc for later append (after merge runs)
i, c = find_one(lambda c: "tail -c 2000 merged.c" in c, "dispatch")
m = HD.search(c)
assert m.group(2) == 'merged.c' and 'dispatch layer' in m.group(3)
open(REGEN + "/dispatch_v3.inc", "w").write(m.group(3) + "\n")
print("dispatch layer saved:", len(m.group(3)), "bytes")

print("replay done")
