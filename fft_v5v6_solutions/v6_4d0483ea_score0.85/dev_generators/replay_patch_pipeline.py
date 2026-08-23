#!/usr/bin/env python3
"""Replay the /workdir file-modifying actions of attempt 4d0483ea from its session log.

The log records model actions (never tool outputs). The final container was
restored from the post-setup snapshot and replayed the retained history
(seg-1 actions through the profile1.py action, log lines 236-3252), then
continued live (log lines 12855-16010). We replay, in that exact order, every
operation that writes /workdir/implementation.c or /workdir/solution.py:
  - str_replace_based_edit_tool create / str_replace calls
  - bash `python3 - <<'EOF' ... EOF` heredoc patch scripts that write those files
  - bash `sed -i '...' implementation.c` commands

Log text encoding: original text T was JSON-escaped (\\ -> \\\\, newline -> \\n,
" -> \\") and then a naive unescape re-expanded only \\n -> newline and \\" -> ".
Inverse (left-to-right): '\\\\'->'\\';  '\\'+newline->'\\n' (two chars);
'\\'+'"'->'\\"';  '\\t'->TAB.
"""
import re, subprocess, sys, os

LOG = sys.argv[1]
RECON = sys.argv[2]          # directory acting as /workdir
GENDIR = sys.argv[3]         # where to save extracted patch scripts

os.makedirs(RECON, exist_ok=True)
os.makedirs(GENDIR, exist_ok=True)

raw_lines = open(LOG, encoding='utf-8').read().split('\n')

def window(lo, hi):
    return '\n'.join(raw_lines[lo-1:hi])

def decode(s):
    out = []
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c == '\\' and i + 1 < n:
            d = s[i+1]
            if d == '\\':
                out.append('\\'); i += 2; continue
            if d == '\n':
                out.append('\\n'); i += 2; continue
            if d == '"':
                out.append('\\"'); i += 2; continue
            if d == 't':
                out.append('\t'); i += 2; continue
        out.append(c); i += 1
    return ''.join(out)

ACTION_SPLIT = re.compile(r'^\[2026-08-23T[^\]]+\] \[system\] Processing action from model:.*$', re.M)

def actions_in(text):
    parts = ACTION_SPLIT.split(text)
    return parts[1:] if len(parts) > 1 else []

PARAM = re.compile(r'<(?:antml:)?parameter name="(\w+)">(.*?)</(?:antml:)?parameter>', re.S)
INVOKE = re.compile(r'<(?:antml:)?invoke name="(\w+)">(.*?)</(?:antml:)?invoke>', re.S)

opnum = 0
def save_gen(tag, text):
    global opnum
    opnum += 1
    fn = os.path.join(GENDIR, f'{opnum:03d}_{tag}')
    with open(fn, 'w') as f:
        f.write(text)
    return fn

def run_py(body, tag):
    fn = save_gen(tag + '.py', body)
    r = subprocess.run([sys.executable, fn], cwd=RECON,
                       capture_output=True, text=True)
    status = 'OK' if r.returncode == 0 else f'FAIL rc={r.returncode}'
    print(f'  [{opnum:03d}] py  {tag}: {status}  out={r.stdout.strip()[:60]!r}'
          + (f'  err={r.stderr.strip().splitlines()[-1][:100]!r}' if r.returncode else ''))

def unescape_bre(lhs):
    # our seds only escape '*'; treat rest literally
    return lhs.replace(r'\*', '*')

def run_sed(script, tag):
    fn = os.path.join(RECON, 'implementation.c')
    content = open(fn).read()
    applied = []
    for cmd in script.split(';'):
        cmd = cmd.strip()
        if not cmd:
            continue
        m = re.match(r's/(.*)/(.*)/$', cmd, re.S)
        if not m:
            print(f'  [sed] {tag}: SKIP unparsed {cmd[:60]!r}')
            continue
        lhs, rhs = m.group(1), m.group(2)
        if '\n' in lhs or '\\n' in lhs:
            print(f'  [sed] {tag}: SKIP (pattern spans lines, sed no-op)')
            continue
        lhs_l = unescape_bre(lhs)
        cnt = content.count(lhs_l)
        content = content.replace(lhs_l, rhs)
        applied.append((lhs_l[:50], cnt))
    open(fn, 'w').write(content)
    save_gen(tag + '.sed', script)
    for l, c in applied:
        print(f'  [{opnum:03d}] sed {tag}: {c} hit(s) for {l!r}')

def handle_bash(cmd):
    # walk the command text; execute heredocs and seds in positional order
    lines = cmd.split('\n')
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.rstrip().endswith("python3 - <<'EOF'"):
            j = i + 1
            body = []
            while j < len(lines) and lines[j] != 'EOF':
                body.append(lines[j]); j += 1
            bodytxt = '\n'.join(body) + '\n'
            if ("open('implementation.c','w')" in bodytxt or
                "open('implementation.c', 'w')" in bodytxt or
                "open('solution.py','w')" in bodytxt or
                "open('solution.py', 'w')" in bodytxt):
                run_py(bodytxt, 'patch')
            i = j + 1
            continue
        m = re.search(r"sed -i '([^']*)'\s+implementation\.c", ln)
        if m:
            run_sed(m.group(1), 'sed')
        i += 1

def handle_action(atext):
    for name, body in INVOKE.findall(atext):
        params = dict(PARAM.findall(body))
        if name == 'str_replace_based_edit_tool':
            cmd = params.get('command')
            path = params.get('path', '')
            base = os.path.basename(path)
            if cmd == 'create':
                txt = params['file_text']
                fn = os.path.join(RECON, base)
                open(fn, 'w').write(txt)
                save_gen('create_' + base, txt)
                print(f'  [{opnum:03d}] create {base}: {len(txt)} bytes')
            elif cmd == 'str_replace':
                old, new = params['old_str'], params['new_str']
                fn = os.path.join(RECON, base)
                content = open(fn).read()
                cnt = content.count(old)
                if cnt != 1:
                    print(f'  [!!] str_replace {base}: {cnt} occurrences of old_str')
                content = content.replace(old, new, 1)
                open(fn, 'w').write(content)
                save_gen('strreplace_' + base + '.txt', 'OLD>>>\n' + old + '\n<<<NEW>>>\n' + new)
                print(f'  [   ] str_replace {base}: applied ({cnt} match)')
        elif name == 'bash':
            c = params.get('command', '')
            if ('implementation.c' in c or 'solution.py' in c):
                handle_bash(c)

print('=== window 1 (retained seg-1 history: log lines 236-3252) ===')
for a in actions_in(window(236, 3252)):
    handle_action(decode(a))

print('=== window 2 (final live actions: log lines 12855-16010) ===')
for a in actions_in(window(12855, 16010)):
    handle_action(decode(a))

print('done.')
