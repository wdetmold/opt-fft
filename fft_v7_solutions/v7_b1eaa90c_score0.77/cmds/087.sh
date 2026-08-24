cd /workdir/dev && python3 - <<'EOF'
src = open('kernels.py').read()
# make halfmatrix inner j-loops fully unrollable via config knob: replace 'unroll 1' with 'unroll {HU}' inside gen_halfmatrix_inst bodies
import re
fn_start = src.index("def gen_halfmatrix_inst(")
fn_end = src.index("def gen_inst(")
block = src[fn_start:fn_end]
block = block.replace("#pragma GCC unroll 1", "#pragma GCC unroll {HU}")
# define HU from CONFIG via parameter
block = block.replace("def gen_halfmatrix_inst(N, fname, IS, OS, pw, PF=0, pwfn='pw_sqrtnr'):",
                      "def gen_halfmatrix_inst(N, fname, IS, OS, pw, PF=0, pwfn='pw_sqrtnr', HU=1):")
src = src[:fn_start] + block + src[fn_end:]
src = src.replace("def gen_inst(N, fname, IS, OS, pw, PF=0, pwfn='pw_sqrtnr'):",
                  "def gen_inst(N, fname, IS, OS, pw, PF=0, pwfn='pw_sqrtnr', HU=1):")
src = src.replace("return gen_halfmatrix_inst(N, fname, IS, OS, pw, PF, pwfn)",
                  "return gen_halfmatrix_inst(N, fname, IS, OS, pw, PF, pwfn, HU)")
open('kernels.py','w').write(src)
EOF
python3 - <<'EOF'
src = open('gen.py').read()
src = src.replace("def cfg_pf(L): return CONFIG['pf'].get(str(L), 256)",
                  "def cfg_pf(L): return CONFIG['pf'].get(str(L), 256)\ndef cfg_hu(L): return int(CONFIG.get('hu', {}).get(str(L), 1))")
src = src.replace('parts.append(kernels.gen_inst(L, f"f{L}_z", 1, 1, False))',
                  'parts.append(kernels.gen_inst(L, f"f{L}_z", 1, 1, False, HU=cfg_hu(L)))')
src = src.replace('parts.append(kernels.gen_inst(L, f"f{L}_y", L, L, False))',
                  'parts.append(kernels.gen_inst(L, f"f{L}_y", L, L, False, HU=cfg_hu(L)))')
src = src.replace('parts.append(kernels.gen_inst(L, f"f{L}_x", L*L, L*L, True, PF=cfg_pf(L), pwfn="pw_"+cfg_pw(L)))',
                  'parts.append(kernels.gen_inst(L, f"f{L}_x", L*L, L*L, True, PF=cfg_pf(L), pwfn="pw_"+cfg_pw(L), HU=cfg_hu(L)))')
src = src.replace('parts.append(kernels.gen_inst(L, f"f{L}_cb", P1//8, 1, False))',
                  'parts.append(kernels.gen_inst(L, f"f{L}_cb", P1//8, 1, False, HU=cfg_hu(L)))')
src = src.replace('parts.append(kernels.gen_inst(L, f"f{L}_k3", P2//8, P2//8, True, PF=cfg_pf(L), pwfn="pw_"+cfg_pw(L)))',
                  'parts.append(kernels.gen_inst(L, f"f{L}_k3", P2//8, P2//8, True, PF=cfg_pf(L), pwfn="pw_"+cfg_pw(L), HU=cfg_hu(L)))')
open('gen.py','w').write(src)
EOF
for HU in 1 16; do
python3 -c "
import tune
cfg={'pw':{'13':'alt','17':'alt','23':'alt','45':'alt'},'hu':{'13':'$HU','17':'$HU','23':'$HU'}}
so=tune.build(cfg,'hu$HU')
r=tune.bench(so, Ls=(13,17,23), reps=5)
print('HU=$HU', {L: round(v,2) for L,v in r.items()})
"
done