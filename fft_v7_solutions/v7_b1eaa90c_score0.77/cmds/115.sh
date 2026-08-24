cd /workdir/dev && python3 - <<'EOF'
src = open('kernels.py').read()
old = """    st = []
    for k2 in range(G):
        fn = "pw_sqrtnr" if k2 % 2 == 0 else "pw_newton"
        st.append(f"    {{ V zr = {out[k2][0]} + cr[{k2}*{P2V} + o]; V zi = {out[k2][1]} + ci[{k2}*{P2V} + o]; "
                  f"V f = {fn}(zr, zi); tr[{k2}*{OUTS} + o] = zr*f; ti[{k2}*{OUTS} + o] = zi*f; }}")"""
new = """    st = []
    for k2 in range(G):
        ch = PWPAT[k2 % len(PWPAT)]
        fn = "pw_sqrtnr" if ch == 's' else "pw_newton"
        st.append(f"    {{ V zr = {out[k2][0]} + cr[{k2}*{P2V} + o]; V zi = {out[k2][1]} + ci[{k2}*{P2V} + o]; "
                  f"V f = {fn}(zr, zi); tr[{k2}*{OUTS} + o] = zr*f; ti[{k2}*{OUTS} + o] = zi*f; }}")"""
assert old in src
src = src.replace(old, new)
src = src.replace("def gen_sq_stage2(N, G, fname, P2V, LP1V, OUTS, PF=0):",
                  "def gen_sq_stage2(N, G, fname, P2V, LP1V, OUTS, PF=0, PWPAT='sn'):")
open('kernels.py','w').write(src)
EOF
python3 - <<'EOF'
src = open('gen.py').read()
src = src.replace('parts.append(kernels.gen_sq_stage2(L, G, f"f{L}_s2", P2V, L*P1V, P2V, PF=cfg_pf(L)))',
                  'parts.append(kernels.gen_sq_stage2(L, G, f"f{L}_s2", P2V, L*P1V, P2V, PF=cfg_pf(L), PWPAT=CONFIG.get("sqpw",{}).get(str(L),"sn")))')
src = src.replace('parts.append(kernels.gen_sq_stage2(L, G, f"f{L}_s2e", P2V, L*P1V, G*P2V, PF=cfg_pf(L)))',
                  'parts.append(kernels.gen_sq_stage2(L, G, f"f{L}_s2e", P2V, L*P1V, G*P2V, PF=cfg_pf(L), PWPAT=CONFIG.get("sqpw",{}).get(str(L),"sn")))')
open('gen.py','w').write(src)
print("ok")
EOF
python3 - <<'EOF'
import tune
pats = ['sn','n','s','ssn','snn']
sos = {p: tune.build({'sqpw':{'36':p,'64':p}}, f'pw{p}') for p in pats}
for i in range(2):
    for p, so in sos.items():
        r = tune.bench(so, Ls=(36,64), reps=3)
        print(f"pat={p:4s}", {L: round(v,2) for L,v in r.items()}, flush=True)
EOF