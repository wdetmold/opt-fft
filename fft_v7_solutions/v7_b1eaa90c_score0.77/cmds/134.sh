cd /workdir/dev && python3 - <<'EOF'
src = open('gen.py').read()
# add parity option to gen_sq: PAR = 1 -> yz flips orientation, c dual copies, parity-extract
old = "def gen_sq(L, G):\n    PFON"
new = "def gen_sq(L, G):\n    PAR = int(CONFIG.get('sqpar', {}).get(str(L), 0))\n    PFON"
assert old in src
src = src.replace(old, new)

# emit alternative yz + c + driver bits when PAR (only for non-64 generic path)
old_yz = """{yzdef if yzdef else f'''static void sq{L}_yz(V* sr, V* si) {{{{"""
assert old_yz in src
# build parity yz generator string: pass1 cb->buf->transpose->TSLAB ; pass2 cb straight TSLAB->slab strided
par_yz = """{yzdef if yzdef else (f'''static void sq{L}_yz(V* sr, V* si) {{{{
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
  for (long cc = 0; cc < {NB}; cc++)
    f{L}_cbs(slabts{L}_r + cc, slabts{L}_i + cc, sr + cc, si + cc);
}}}}''' if PAR else f'''static void sq{L}_yz(V* sr, V* si) {{{{"""
src = src.replace(old_yz, par_yz)
# close the conditional: the original ends with }}}}'''}  -> becomes }}}}''')}
old_end = """  }}}}
}}}}'''}"""
assert old_end in src
src = src.replace(old_end, """  }}}}
}}}}''')}""")
open('gen.py','w').write(src)
print("yz parity emitted")
EOF
python3 - <<'EOF'
src = open('gen.py').read()
# c dual copies + parity-aware s2 calls + extract for PAR in gen_sq body
# 1) buffers: add c odd copy
src = src.replace("static V sq{L}_cr[{L*P2V}], sq{L}_ci[{L*P2V}];",
                  "static V sq{L}_cr[{L*P2V}], sq{L}_ci[{L*P2V}];\n{f'static V sq{L}_c1r[{L*P2V}], sq{L}_c1i[{L*P2V}];' if PAR else ''}")
# 2) ingest: build flipped copy when PAR
src = src.replace("""        for (long z = 0; z < {L}; z++) {{
          xr[row+z] = px[2*z]; xi[row+z] = px[2*z+1];
          cr[rowc+z] = pc[2*z]; ci[rowc+z] = pc[2*z+1];
        }}
        for (long z = {L}; z < {P1}; z++) {{
          xr[row+z] = 0.0; xi[row+z] = 0.0;
          cr[rowc+z] = 1.0; ci[rowc+z] = 0.0;
        }}""",
"""        for (long z = 0; z < {L}; z++) {{
          xr[row+z] = px[2*z]; xi[row+z] = px[2*z+1];
          cr[rowc+z] = pc[2*z]; ci[rowc+z] = pc[2*z+1];
          {f'c1r[xc*{P2} + z*{P1} + y] = pc[2*z]; c1i[xc*{P2} + z*{P1} + y] = pc[2*z+1];' if PAR else ''}
        }}
        for (long z = {L}; z < {P1}; z++) {{
          xr[row+z] = 0.0; xi[row+z] = 0.0;
          cr[rowc+z] = 1.0; ci[rowc+z] = 0.0;
          {f'c1r[row+z] = 1.0; c1i[row+z] = 0.0;' if PAR else ''}
        }}""")
src = src.replace("""  double *xr = (double*)sq{L}_ar, *xi = (double*)sq{L}_ai;
  double *cr = (double*)sq{L}_cr, *ci = (double*)sq{L}_ci;""",
"""  double *xr = (double*)sq{L}_ar, *xi = (double*)sq{L}_ai;
  double *cr = (double*)sq{L}_cr, *ci = (double*)sq{L}_ci;
  {f'double *c1r = (double*)sq{L}_c1r, *c1i = (double*)sq{L}_c1i;' if PAR else ''}""")
# 3) s2/s2e calls pick c by parity
src = src.replace("""        f{L}_s2(srcr + g*{G}*{P2V}, srci + g*{G}*{P2V}, sq{L}_cr + g*{G}*{P2V}, sq{L}_ci + g*{G}*{P2V},
                TAr, TAi, 0, {LP1V});""",
"""        {f'const V *pwcr = ((t+1) & 1) ? sq{L}_c1r : sq{L}_cr;' if PAR else f'const V *pwcr = sq{L}_cr;'}
        {f'const V *pwci = ((t+1) & 1) ? sq{L}_c1i : sq{L}_ci;' if PAR else f'const V *pwci = sq{L}_ci;'}
        f{L}_s2(srcr + g*{G}*{P2V}, srci + g*{G}*{P2V}, pwcr + g*{G}*{P2V}, pwci + g*{G}*{P2V},
                TAr, TAi, 0, {LP1V});""")
src = src.replace("""      f{L}_s2e(srcr + g*{G}*{P2V}, srci + g*{G}*{P2V}, sq{L}_cr + g*{G}*{P2V}, sq{L}_ci + g*{G}*{P2V},
               dstr + g*{P2V}, dsti + g*{P2V}, 0, {LP1V});""",
"""      f{L}_s2e(srcr + g*{G}*{P2V}, srci + g*{G}*{P2V},
               {f'((m & 1) ? sq{L}_c1r : sq{L}_cr)' if PAR else f'sq{L}_cr'} + g*{G}*{P2V},
               {f'((m & 1) ? sq{L}_c1i : sq{L}_ci)' if PAR else f'sq{L}_ci'} + g*{G}*{P2V},
               dstr + g*{P2V}, dsti + g*{P2V}, 0, {LP1V});""")
# 4) extract with parity
src = src.replace("""    slab{L}_extract(m > 1 ? sq{L}_1r : dstr, m > 1 ? sq{L}_1i : dsti, out1 + 2*v*{L3});
    slab{L}_extract(dstr, dsti, outm + 2*v*{L3});""",
"""    slab{L}_extract{'p' if PAR else ''}(m > 1 ? sq{L}_1r : dstr, m > 1 ? sq{L}_1i : dsti, out1 + 2*v*{L3}{', 1' if PAR else ''});
    slab{L}_extract{'p' if PAR else ''}(dstr, dsti, outm + 2*v*{L3}{', (int)(m & 1)' if PAR else ''});""")
open('gen.py','w').write(src)
print("parity c/extract wired")
EOF
echo done