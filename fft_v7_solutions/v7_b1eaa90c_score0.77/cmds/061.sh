cd /workdir/dev && python3 - <<'EOF'
src = open('gen.py').read()
# 1. alternating pw: pass pwfn list [even_fn, odd_fn] -> kernels pick by k parity? simpler: pw_alt function that we can't do per-lane... Do per-OUTPUT alternation in kernels: replace PWFN with PWFN_A/PWFN_B alternating by store ordinal.
# Implemented in kernels.py instead. Here: scheme config:
src = src.replace("SCHEME = {6:'soa', 8:'soa', 13:'soa', 17:'slab', 23:'slab', 36:'slab', 45:'slab', 64:'slab'}",
"""_SCHEME_DEFAULT = {6:'soa', 8:'soa', 13:'soa', 17:'slab', 23:'slab', 36:'slab', 45:'slab', 64:'slab'}
def cfg_scheme(L):
    return CONFIG.get('scheme', {}).get(str(L), _SCHEME_DEFAULT[L])""")
src = src.replace("""    for L in SIZES:
        if SCHEME[L] == 'soa':
            if L == 13:
                parts.append(kernels.gen_inst(13, "f13_z", 1, 1, False))
                parts.append(kernels.gen_inst(13, "f13_y", 13, 13, False))
                parts.append(kernels.gen_inst(13, "f13_x", 169, 169, True, PF=cfg_pf(13), pwfn="pw_"+cfg_pw(13)))
                parts.append(gen_soa(13, True))
            else:
                code, _ = codelets.gen_core(L)
                parts.append(code)
                parts.append(gen_soa(L, False))
        else:""","""    for L in SIZES:
        if cfg_scheme(L) == 'soa':
            if L in (13, 17, 23, 36, 45, 64):
                parts.append(kernels.gen_inst(L, f"f{L}_z", 1, 1, False))
                parts.append(kernels.gen_inst(L, f"f{L}_y", L, L, False))
                parts.append(kernels.gen_inst(L, f"f{L}_x", L*L, L*L, True, PF=cfg_pf(L), pwfn="pw_"+cfg_pw(L)))
                parts.append(gen_soa(L, True))
            else:
                code, _ = codelets.gen_core(L)
                parts.append(code)
                parts.append(gen_soa(L, False))
        else:""")
open('gen.py','w').write(src)
print("gen.py scheme config ok")
EOF
python3 - <<'EOF'
# kernels.py: alternate pw between two fns by output ordinal
src = open('kernels.py').read()
src = src.replace('def gen_composite_inst(N, P, Q, pfa, fname, IS, OS, pw, PF=0, pwfn=\'pw_sqrtnr\'):',
                  'def gen_composite_inst(N, P, Q, pfa, fname, IS, OS, pw, PF=0, pwfn=\'pw_sqrtnr\'):\n    alt = pwfn == "pw_alt"')
# composite stores: alternate by k2 parity
old = '''        if pw:
            stB.append(f"    {{ long _k = {ko}; V zr = {out2[k2][0]} + cr[_k*{OS}]; V zi = {out2[k2][1]} + ci[_k*{OS}]; "
                       f"V f = PWFN(zr, zi); yr[_k*{OS}] = zr*f; yi[_k*{OS}] = zi*f; }}")'''
new = '''        if pw:
            fn = ("pw_sqrtnr" if k2 % 2 == 0 else "pw_newton") if alt else "PWFN"
            stB.append(f"    {{ long _k = {ko}; V zr = {out2[k2][0]} + cr[_k*{OS}]; V zi = {out2[k2][1]} + ci[_k*{OS}]; "
                       f"V f = {fn}(zr, zi); yr[_k*{OS}] = zr*f; yi[_k*{OS}] = zi*f; }}")'''
assert old in src
src = src.replace(old, new)
# halfmatrix: stfun with alternation by an index
old2 = '''    if pw:
        stfun = lambda k, a, b: (f"{{ long _k = {k}; V zr = {a} + cr[_k*{OS}]; V zi = {b} + ci[_k*{OS}]; "
                                 f"V f = PWFN(zr, zi); yr[_k*{OS}] = zr*f; yi[_k*{OS}] = zi*f; }}")'''
new2 = '''    alt = pwfn == "pw_alt"
    _alt_ctr = [0]
    def stfun_pw(k, a, b):
        if alt:
            fn = "pw_sqrtnr" if _alt_ctr[0] % 2 == 0 else "pw_newton"
            _alt_ctr[0] += 1
        else:
            fn = "PWFN"
        return (f"{{ long _k = {k}; V zr = {a} + cr[_k*{OS}]; V zi = {b} + ci[_k*{OS}]; "
                f"V f = {fn}(zr, zi); yr[_k*{OS}] = zr*f; yi[_k*{OS}] = zi*f; }}")
    if pw:
        stfun = stfun_pw'''
assert old2 in src
src = src.replace(old2, new2)
# PWFN define must handle pw_alt (won't be referenced when alt)
src = src.replace('#define PWFN {pwfn}', '#define PWFN {pwfn if pwfn != "pw_alt" else "pw_sqrtnr"}')
open('kernels.py','w').write(src)
print("kernels alt ok")
EOF
python3 - <<'EOF'
# gen.py: SoA path for 6/8 PW68 define: allow pw_alt -> alternate inside loop? keep simple: map alt->sqrtnr there
src = open('gen.py').read()
src = src.replace("pwdef = f'#define PW68 pw_{cfg_pw(L)}'",
                  "pwfn_ = cfg_pw(L)\n    pwdef = f'#define PW68 pw_{pwfn_ if pwfn_ != \"alt\" else \"sqrtnr\"}'")
open('gen.py','w').write(src)
EOF
GENCFG='{"scheme":{"17":"soa"},"pw":{"17":"alt","23":"alt","36":"alt","45":"alt","64":"alt","13":"alt"}}' python3 gen.py && gcc -fsyntax-only -O2 -march=native ../implementation.c && echo COMPILES