#!/usr/bin/env python3
"""Replay solution.py and NOTES.md final states from attempt.log."""
import re, os, sys, subprocess

S = "/private/tmp/claude-501/-Users-wdetmold-MIT-Dropbox-William-Detmold-2026-problems/3e8ea67f-c989-4b31-8f6e-316caac734df/scratchpad"
LOG = "/Users/wdetmold/MIT Dropbox/William Detmold/2026_problems/opt-fft/fft_hot_solutions/hot_d82aee89_score0.98/attempt.log"
WD = S + "/workdir"
os.makedirs(WD, exist_ok=True)

log = open(LOG).read()
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

HD = re.compile(r"cat >(>?) *([^\s]+) <<'EOF'\n(.*?)\n^EOF$", re.S | re.M)
PY = re.compile(r"python3 - <<'EOF'\n(.*?)\n^EOF$", re.S | re.M)

def find(pred):
    return [c for c in cmds if pred(c)]

def run_pyedit(script):
    script = script.replace("/workdir", WD)
    r = subprocess.run([sys.executable, "-"], input=script, text=True,
                       cwd=WD, capture_output=True)
    assert r.returncode == 0, (r.stdout[-1000:], r.stderr[-1000:])
    if r.stdout.strip():
        print("   out:", r.stdout.strip()[:150])

# ---- solution.py base (written to /tmp/w/sol/solution.py)
c = find(lambda c: "mkdir -p /tmp/w/sol" in c and "cat > solution.py" in c)[0]
for m in HD.finditer(c):
    if m.group(2) == "solution.py":
        open(WD + "/solution.py", "w").write(m.group(3) + "\n")
        print("solution.py base written", len(m.group(3)))

# edit 1: add scheduler flags (cmd with 'fix the compile flags line')
c = find(lambda c: "# fix the compile flags line in /workdir/solution.py properly" in c)[0]
run_pyedit(PY.search(c).group(1))
print("solution.py flags -> sched")

# edit 2: revert scheduler flags (cmd with 'flags updated')
c = find(lambda c: 'print("flags updated")' in c)[0]
run_pyedit(PY.search(c).group(1))
print("solution.py flags reverted")

# edit 3: move mallopt into marked region
c = find(lambda c: "# remove mallopt from the top (verbatim region)" in c)[0]
run_pyedit(PY.search(c).group(1))
print("solution.py mallopt moved")

# ---- NOTES.md
c = find(lambda c: "cat > /workdir/NOTES.md <<'EOF'" in c)[0]
m = [m for m in HD.finditer(c) if m.group(2) == "/workdir/NOTES.md"][0]
open(WD + "/NOTES.md", "w").write(m.group(3) + "\n")
print("NOTES.md base written", len(m.group(3)))

c = find(lambda c: "cat >> /workdir/NOTES.md <<'EOF'" in c)[0]
m = [m for m in HD.finditer(c) if m.group(2) == "/workdir/NOTES.md"][0]
open(WD + "/NOTES.md", "a").write(m.group(3) + "\n")
print("NOTES.md robustness ledger appended")

for marker, note in [
    ("## Compile-flag finding (this round)", "flag-finding insert + v2 ledger"),
    ("## Final QA summary (all on the shipped artifact, fresh processes)", "final QA summary"),
    ('s = s.replace("## v2 ledger", "## v2/v3 ledger")', "v3 section"),
    ("## Closing self-benchmark (shipped artifact, 3 fresh processes, best-of-3 each)", "closing benchmark"),
]:
    hits = find(lambda c, mk=marker: mk in c and "NOTES.md" in c)
    assert hits, marker
    run_pyedit(PY.search(hits[0]).group(1))
    print("NOTES.md:", note)

print("replay2 done")
