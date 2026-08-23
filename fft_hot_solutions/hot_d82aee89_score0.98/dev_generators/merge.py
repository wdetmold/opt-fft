"""Merge adopted engines + my engines into one C file with prefix-renaming via #define blocks."""
import re, subprocess, sys

def wrap(name, text, defines):
    out = [f"// ================= engine: {name} =================\n"]
    # undef common macros that may be redefined
    macros = ["ALIGN64","TR8","VL","VS","VLU","VSU","VADD","VSUB","VMUL","VFMA","VFMS","VFNMA",
              "VSET1","MAP2","PERM2","PERM2Z","VRSQRT","VRCP","HP","LIKELY","UNLIKELY","RESTRICT",
              "BCAST","BCASTV","DEINT","INTER","KSTORE","GEN_WRAPPERS","MAP_PIPE","PF","VHALF","VONE"]
    for m in macros:
        out.append(f"#ifdef {m}\n#undef {m}\n#endif\n")
    for d in defines:
        out.append(f"#define {d} {name}_{d}\n")
    out.append(text)
    out.append("\n")
    for d in defines:
        out.append(f"#undef {d}\n")
    for m in macros:
        out.append(f"#ifdef {m}\n#undef {m}\n#endif\n")
    return "".join(out)

def try_compile(src):
    open('/tmp/w/regen/merged_test.c','w').write(src)
    r = subprocess.run(['gcc','-O0','-march=native','-fsyntax-only','/tmp/w/regen/merged_test.c'],
                       capture_output=True, text=True)
    return r.returncode == 0, r.stderr

def find_redefs(err):
    ids = set()
    for m in re.finditer(r"error: redefinition of [\'\u2018](\w+)[\'\u2019]", err):
        ids.add(m.group(1))
    for m in re.finditer(r"error: conflicting types for [\'\u2018](\w+)[\'\u2019]", err):
        ids.add(m.group(1))
    for m in re.finditer(r"error: redeclaration of [\'\u2018](\w+)[\'\u2019]", err):
        ids.add(m.group(1))
    return ids

def build(parts, out):
    # parts: list of (name, text, set_of_defines)
    for it in range(40):
        src = "".join(wrap(n, t, sorted(d)) for n, t, d in parts)
        ok, err = try_compile(src)
        if ok:
            open(out,'w').write(src)
            print("merged OK ->", out, len(src), "bytes after", it, "iterations")
            return True
        ids = find_redefs(err)
        if not ids:
            print("UNRESOLVED:\n", err[:4000]); return False
        # assign each redefined id to all parts AFTER the first part that defines it
        for i in ids:
            seen = False
            pat = re.compile(r"\b" + re.escape(i) + r"\b")
            for n, t, d in parts:
                if pat.search(t):
                    if seen: d.add(i)
                    seen = True
        print(f"iter {it}: renamed {len(ids)} ids")
    return False

mine_my = open('/tmp/w/dev/implsoa.c').read()
wv_my  = open('/tmp/w/dev/implwv.c').read()
b00    = open('/tmp/w/b00/implementation.c').read()
d43    = open('/tmp/w/regen/mine.c').read()
s81    = open('/tmp/w/regen/s81.c').read()
f30    = open('/tmp/w/regen/f30.c').read()

# exported entry points we must keep callable (rename to eng-specific names deliberately):
# b00: run_6..run_64 -> b00_run_L (prefix via defines)
expb00 = {f"run_{L}" for L in (6,8,13,17,23,36,45,64)}
import re as _re
expd43 = set(_re.findall(r"^void (\w+)\(", d43, _re.M)) | set(_re.findall(r"^void (\w+)\(", d43.replace("\nvoid ","\nvoid "), _re.M))
exps81 = {"run_size"}
expf30 = {"init_tables","run64"}
expmy  = {f"run_{L}" for L in (6,8,13,17,23)}
expwv  = {f"run_{L}wv" for L in (36,45,64)}

b00 = ("#pragma GCC push_options\n"
       "#pragma GCC optimize(\"O3\",\"unroll-loops\",\"schedule-insns\",\"sched-pressure\",\"no-math-errno\",\"no-trapping-math\")\n"
       + b00 +
       "\n#pragma GCC pop_options\n")
parts = [
    ("s81", s81,    set(exps81)),
    ("f30", f30,    set(expf30)),
    ("d43", d43,    set(expd43)),
    ("b00", b00,    set(expb00)),
]
if not build(parts, '/tmp/w/regen/merged.c'):
    sys.exit(1)
