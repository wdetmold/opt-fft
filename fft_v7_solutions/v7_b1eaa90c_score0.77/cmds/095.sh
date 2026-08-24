cd /workdir/dev && python3 - <<'EOF'
src = open('gen.py').read()
# emit zline + ycol for 64 and a different yz body
old = """        elif cfg_scheme(L) == 'sq':
            G = {36:6, 64:8}[L]
            P1 = r8(L); P2 = L*P1 + (8 if L == 64 else 0)
            P1V, P2V = P1//8, P2//8
            parts.append(f"static V slabts{L}_r[{P1*P1V}], slabts{L}_i[{P1*P1V}];")
            parts.append(kernels.gen_inst(L, f"f{L}_cb", P1V, 1, False, HU=cfg_hu(L)))"""
new = """        elif cfg_scheme(L) == 'sq':
            G = {36:6, 64:8}[L]
            P1 = r8(L); P2 = L*P1 + (8 if L == 64 else 0)
            P1V, P2V = P1//8, P2//8
            parts.append(f"static V slabts{L}_r[{P1*P1V}], slabts{L}_i[{P1*P1V}];")
            if L == 64:
                parts.append(kernels.gen_zline64("f64_zline"))
                parts.append(kernels.gen_inst(64, "f64_ycol", P1V, P1V, False))
            parts.append(kernels.gen_inst(L, f"f{L}_cb", P1V, 1, False, HU=cfg_hu(L)))"""
assert old in src
src = src.replace(old, new)

# gen_sq: alternate yz body for 64
old2 = """def gen_sq(L, G):
    DPF = int(CONFIG.get('dpf', {}).get(str(L), 0))
    DPF_ON = 1 if DPF else 0
    DPF = DPF if DPF else 64
    P1 = r8(L)"""
new2 = """def gen_sq_yz64(P1V, P2V):
    return f'''static void sq64_yz(V* sr, V* si) {{
  for (long y = 0; y < 64; y++)
    f64_zline(sr + y*{P1V}, si + y*{P1V});
  for (long cc = 0; cc < {P1V}; cc++)
    f64_ycol(sr + cc, si + cc, sr + cc, si + cc);
}}
'''

def gen_sq(L, G):
    DPF = int(CONFIG.get('dpf', {}).get(str(L), 0))
    DPF_ON = 1 if DPF else 0
    DPF = DPF if DPF else 64
    P1 = r8(L)"""
assert old2 in src
src = src.replace(old2, new2)
# in gen_sq body: choose yz definition
old3 = """    return twtabs + f\"\"\"
// ---------------- L={L} : square-CT fused ping-pong (G={G}, P1={P1}, P2={P2}) ----------------"""
new3 = """    yzdef = gen_sq_yz64(P1V, P2V) if L == 64 else None
    return twtabs + f\"\"\"
// ---------------- L={L} : square-CT fused ping-pong (G={G}, P1={P1}, P2={P2}) ----------------"""
assert old3 in src
src = src.replace(old3, new3)
old4 = """static void sq{L}_yz(V* sr, V* si) {{
  for (long cc = 0; cc < {NB}; cc++) {{
    V *br_ = sq{L}_buf_r + (cc&1)*{P1}, *bi_ = sq{L}_buf_i + (cc&1)*{P1};"""
# wrap default yz in conditional: emit yzdef if 64
src = src.replace("""static void sq{L}_yz(V* sr, V* si) {{
  for (long cc = 0; cc < {NB}; cc++) {{
    V *br_ = sq{L}_buf_r + (cc&1)*{P1}, *bi_ = sq{L}_buf_i + (cc&1)*{P1};
    f{L}_cb(sr + cc, si + cc, br_, bi_);
    for (long rb = 0; rb < {P1V}; rb++) {{
      V tb[8];
      tr8x8(br_ + rb*8, tb);
      for (int q = 0; q < 8; q++) slabts{L}_r[(cc*8+q)*{P1V} + rb] = tb[q];
      tr8x8(bi_ + rb*8, tb);
      for (int q = 0; q < 8; q++) slabts{L}_i[(cc*8+q)*{P1V} + rb] = tb[q];
    }}
  }}
  for (long cc = 0; cc < {NB}; cc++) {{
    V *br_ = sq{L}_buf_r + (cc&1)*{P1}, *bi_ = sq{L}_buf_i + (cc&1)*{P1};
    f{L}_cb(slabts{L}_r + cc, slabts{L}_i + cc, br_, bi_);
    int lim = (int)({L} - cc*8); if (lim > 8) lim = 8;
    for (long rb = 0; rb < {P1V}; rb++) {{
      V tb[8];
      tr8x8(br_ + rb*8, tb);
      for (int q = 0; q < lim; q++) sr[(cc*8+q)*{P1V} + rb] = tb[q];
      tr8x8(bi_ + rb*8, tb);
      for (int q = 0; q < lim; q++) si[(cc*8+q)*{P1V} + rb] = tb[q];
    }}
  }}
}}""",
"""{yzdef if yzdef else f'''static void sq{L}_yz(V* sr, V* si) {{{{
  for (long cc = 0; cc < {NB}; cc++) {{{{
    V *br_ = sq{L}_buf_r + (cc&1)*{P1}, *bi_ = sq{L}_buf_i + (cc&1)*{P1};
    f{L}_cb(sr + cc, si + cc, br_, bi_);
    for (long rb = 0; rb < {P1V}; rb++) {{{{
      V tb[8];
      tr8x8(br_ + rb*8, tb);
      for (int q = 0; q < 8; q++) slabts{L}_r[(cc*8+q)*{P1V} + rb] = tb[q];
      tr8x8(bi_ + rb*8, tb);
      for (int q = 0; q < 8; q++) slabts{L}_i[(cc*8+q)*{P1V} + rb] = tb[q];
    }}}}
  }}}}
  for (long cc = 0; cc < {NB}; cc++) {{{{
    V *br_ = sq{L}_buf_r + (cc&1)*{P1}, *bi_ = sq{L}_buf_i + (cc&1)*{P1};
    f{L}_cb(slabts{L}_r + cc, slabts{L}_i + cc, br_, bi_);
    int lim = (int)({L} - cc*8); if (lim > 8) lim = 8;
    for (long rb = 0; rb < {P1V}; rb++) {{{{
      V tb[8];
      tr8x8(br_ + rb*8, tb);
      for (int q = 0; q < lim; q++) sr[(cc*8+q)*{P1V} + rb] = tb[q];
      tr8x8(bi_ + rb*8, tb);
      for (int q = 0; q < lim; q++) si[(cc*8+q)*{P1V} + rb] = tb[q];
    }}}}
  }}}}
}}}}'''}""")
open('gen.py','w').write(src)
print("ok")
EOF
python3 gen.py && gcc -fsyntax-only -O3 -march=native ../implementation.c && echo COMPILES && python3 -c "
import tune
so=tune.build({},'zl')
r=tune.bench(so, Ls=(64,36), reps=5)
print({L: round(v,2) for L,v in r.items()})
"