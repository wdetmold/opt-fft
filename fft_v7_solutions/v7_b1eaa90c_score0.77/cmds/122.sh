cd /workdir/dev && python3 - <<'EOF'
src = open('kernels.py').read()
old = """    st = [f"    dr[0 + o] = {out[0][0]}; di[0 + o] = {out[0][1]};"]
    for k1 in range(1, G):
        st.append(f"    {{ V ar = {out[k1][0]}, ai = {out[k1][1]}; V c_ = VC(twr[{k1}]), s_ = VC(twi[{k1}]); "
                  f"dr[{k1}*{G*P2V} + o] = c_*ar - s_*ai; di[{k1}*{G*P2V} + o] = c_*ai + s_*ar; }}")"""
new = """    if NT:
        st = [f"    _mm512_stream_pd((double*)(dr + o), (__m512d)({out[0][0]})); _mm512_stream_pd((double*)(di + o), (__m512d)({out[0][1]}));"]
        for k1 in range(1, G):
            st.append(f"    {{ V ar = {out[k1][0]}, ai = {out[k1][1]}; V c_ = VC(twr[{k1}]), s_ = VC(twi[{k1}]); "
                      f"_mm512_stream_pd((double*)(dr + {k1}*{G*P2V} + o), (__m512d)(c_*ar - s_*ai)); "
                      f"_mm512_stream_pd((double*)(di + {k1}*{G*P2V} + o), (__m512d)(c_*ai + s_*ar)); }}")
    else:
        st = [f"    dr[0 + o] = {out[0][0]}; di[0 + o] = {out[0][1]};"]
        for k1 in range(1, G):
            st.append(f"    {{ V ar = {out[k1][0]}, ai = {out[k1][1]}; V c_ = VC(twr[{k1}]), s_ = VC(twi[{k1}]); "
                      f"dr[{k1}*{G*P2V} + o] = c_*ar - s_*ai; di[{k1}*{G*P2V} + o] = c_*ai + s_*ar; }}")"""
assert old in src
src = src.replace(old, new)
src = src.replace("def gen_sq_stage1(N, G, fname, INS, P2V, LP1V, PF=0):",
                  "def gen_sq_stage1(N, G, fname, INS, P2V, LP1V, PF=0, NT=0):")
open('kernels.py','w').write(src)
EOF
python3 - <<'EOF'
src = open('gen.py').read()
src = src.replace('parts.append(kernels.gen_sq_stage1(L, G, f"f{L}_s1", P2V, P2V, L*P1V, PF=0))',
                  'parts.append(kernels.gen_sq_stage1(L, G, f"f{L}_s1", P2V, P2V, L*P1V, PF=0, NT=int(CONFIG.get("nt",{}).get(str(L),0))))')
open('gen.py','w').write(src)
EOF
python3 - <<'EOF'
import tune
A = tune.build({'nt':{'64':'0','36':'0'}}, 'ntA')
B = tune.build({'nt':{'64':'1','36':'1'}}, 'ntB')
for i in range(3):
    ra = tune.bench(A, Ls=(36,64), reps=4)
    rb = tune.bench(B, Ls=(36,64), reps=4)
    print("reg", {L: round(v,2) for L,v in ra.items()}, " nt", {L: round(v,2) for L,v in rb.items()})
EOF