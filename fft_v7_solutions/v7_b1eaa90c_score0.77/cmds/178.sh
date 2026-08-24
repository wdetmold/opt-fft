cd /workdir/dev && python3 - <<'EOF'
src = open('kernels.py').read()
src = src.replace('''    code = pre + f"""#define PWFN {pwfn if pwfn != "pw_alt" else "pw_sqrtnr"}
static __attribute__((noinline)) void {fname}({args}) {{
{decl_tabs}  V S1r[{N}], S1i[{N}];''',
'''    ATTR = "noinline" if not INL else "always_inline"
    kw = "static" if not INL else "static inline"
    code = pre + f"""#define PWFN {pwfn if pwfn != "pw_alt" else "pw_sqrtnr"}
{kw} __attribute__(({ATTR})) void {fname}({args}) {{
{decl_tabs}  V S1r[{N}], S1i[{N}];''')
src = src.replace("def gen_composite_inst(N, P, Q, pfa, fname, IS, OS, pw, PF=0, pwfn='pw_sqrtnr'):",
                  "def gen_composite_inst(N, P, Q, pfa, fname, IS, OS, pw, PF=0, pwfn='pw_sqrtnr', INL=0):")
src = src.replace("def gen_inst(N, fname, IS, OS, pw, PF=0, pwfn='pw_sqrtnr', HU=1):",
                  "def gen_inst(N, fname, IS, OS, pw, PF=0, pwfn='pw_sqrtnr', HU=1, INL=0):")
src = src.replace("if N == 6: return gen_composite_inst(6, 2, 3, True, fname, IS, OS, pw, PF, pwfn)",
                  "if N == 6: return gen_composite_inst(6, 2, 3, True, fname, IS, OS, pw, PF, pwfn, INL)")
src = src.replace("if N == 8: return gen_composite_inst(8, 4, 2, False, fname, IS, OS, pw, PF, pwfn)",
                  "if N == 8: return gen_composite_inst(8, 4, 2, False, fname, IS, OS, pw, PF, pwfn, INL)")
src = src.replace("return gen_composite_inst(36, 4, 9, True, fname, IS, OS, pw, PF, pwfn)",
                  "return gen_composite_inst(36, 4, 9, True, fname, IS, OS, pw, PF, pwfn, INL)")
src = src.replace("return gen_composite_inst(45, 9, 5, True, fname, IS, OS, pw, PF, pwfn)",
                  "return gen_composite_inst(45, 9, 5, True, fname, IS, OS, pw, PF, pwfn, INL)")
src = src.replace("return gen_composite_inst(64, 8, 8, False, fname, IS, OS, pw, PF, pwfn)",
                  "return gen_composite_inst(64, 8, 8, False, fname, IS, OS, pw, PF, pwfn, INL)")
open('kernels.py','w').write(src)
EOF
python3 - <<'EOF'
src = open('gen.py').read()
# INL option for the cb instances in sq and apfa2 paths
src = src.replace('parts.append(kernels.gen_inst(L, f"f{L}_cb", P1V, 1, False))',
                  'parts.append(kernels.gen_inst(L, f"f{L}_cb", P1V, 1, False, INL=int(CONFIG.get("inl",{}).get(str(L),0))))')
src = src.replace('parts.append(kernels.gen_inst(45, "f45_cb", P1V, 1, False))',
                  'parts.append(kernels.gen_inst(45, "f45_cb", P1V, 1, False, INL=int(CONFIG.get("inl",{}).get("45",0))))')
open('gen.py','w').write(src)
EOF
python3 - <<'EOF'
import tune
A = tune.build({'inl':{}}, 'inlA')
B = tune.build({'inl':{'36':'1','45':'1','64':'1'}}, 'inlB')
for i in range(3):
    ra = tune.bench(A, Ls=(36,45,64), reps=4)
    rb = tune.bench(B, Ls=(36,45,64), reps=4)
    print("noinl", {L: round(v,2) for L,v in ra.items()}, " inl", {L: round(v,2) for L,v in rb.items()})
EOF