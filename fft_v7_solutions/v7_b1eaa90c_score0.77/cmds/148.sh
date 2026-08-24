cd /workdir/dev && python3 - <<'EOF'
src = open('gen.py').read()
src = src.replace("def gen_apfa45():\n", "def gen_apfa45():\n    PAR = 0\n", 1)
src = src.replace("def gen_apfa2_45():\n    L = 45", "def gen_apfa2_45():\n    PAR = int(CONFIG.get('sqpar', {}).get('45', 0))\n    L = 45", 1)
open('gen.py','w').write(src)
print("ok")
EOF
python3 - <<'EOF'
src = open('gen.py').read()
edits = [
("""  for (long cc = 0; cc < {NB}; cc++) {{
    V *br_ = a245_buf_r + (cc&1)*{P1}, *bi_ = a245_buf_i + (cc&1)*{P1};
    f45_cb(a245ts_r + cc, a245ts_i + cc, br_, bi_);
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
"""{f'''  for (long cc = 0; cc < {NB}; cc++)
    f45_cbs(a245ts_r + cc, a245ts_i + cc, sr + cc, si + cc);
}}''' if PAR else f'''  for (long cc = 0; cc < {NB}; cc++) {{{{
    V *br_ = a245_buf_r + (cc&1)*{P1}, *bi_ = a245_buf_i + (cc&1)*{P1};
    f45_cb(a245ts_r + cc, a245ts_i + cc, br_, bi_);
    int lim = (int)({L} - cc*8); if (lim > 8) lim = 8;
    for (long rb = 0; rb < {P1V}; rb++) {{{{
      V tb[8];
      tr8x8(br_ + rb*8, tb);
      for (int q = 0; q < lim; q++) sr[(cc*8+q)*{P1V} + rb] = tb[q];
      tr8x8(bi_ + rb*8, tb);
      for (int q = 0; q < lim; q++) si[(cc*8+q)*{P1V} + rb] = tb[q];
    }}}}
  }}}}
}}'''}"""),
("static V a245_cer[{L*P2V}], a245_cei[{L*P2V}];\nstatic V a245_cor[{L*P2V}], a245_coi[{L*P2V}];",
 "static V a245_cer[{L*P2V}], a245_cei[{L*P2V}];\nstatic V a245_cor[{L*P2V}], a245_coi[{L*P2V}];\n{f'static V a245_cer1[{L*P2V}], a245_cei1[{L*P2V}]; static V a245_cor1[{L*P2V}], a245_coi1[{L*P2V}];' if PAR else ''}"),
("  double *cer = (double*)a245_cer, *cei = (double*)a245_cei;\n  double *cor = (double*)a245_cor, *coi = (double*)a245_coi;",
 "  double *cer = (double*)a245_cer, *cei = (double*)a245_cei;\n  double *cor = (double*)a245_cor, *coi = (double*)a245_coi;\n  {f'double *cer1 = (double*)a245_cer1, *cei1 = (double*)a245_cei1; double *cor1 = (double*)a245_cor1, *coi1 = (double*)a245_coi1;' if PAR else ''}"),
("""        for (long z = 0; z < {L}; z++) {{
          xr[row+z] = px[2*z]; xi[row+z] = px[2*z+1];
          cer[rowe+z] = pc[2*z]; cei[rowe+z] = pc[2*z+1];
          cor[rowo+z] = pc[2*z]; coi[rowo+z] = pc[2*z+1];
        }}
        for (long z = {L}; z < {P1}; z++) {{
          xr[row+z] = 0.0; xi[row+z] = 0.0;
          cer[rowe+z] = 1.0; cei[rowe+z] = 0.0;
          cor[rowo+z] = 1.0; coi[rowo+z] = 0.0;
        }}""",
"""        for (long z = 0; z < {L}; z++) {{
          xr[row+z] = px[2*z]; xi[row+z] = px[2*z+1];
          cer[rowe+z] = pc[2*z]; cei[rowe+z] = pc[2*z+1];
          cor[rowo+z] = pc[2*z]; coi[rowo+z] = pc[2*z+1];
          {f'cer1[xce*{P2} + z*{P1} + y] = pc[2*z]; cei1[xce*{P2} + z*{P1} + y] = pc[2*z+1]; cor1[xco*{P2} + z*{P1} + y] = pc[2*z]; coi1[xco*{P2} + z*{P1} + y] = pc[2*z+1];' if PAR else ''}
        }}
        for (long z = {L}; z < {P1}; z++) {{
          xr[row+z] = 0.0; xi[row+z] = 0.0;
          cer[rowe+z] = 1.0; cei[rowe+z] = 0.0;
          cor[rowo+z] = 1.0; coi[rowo+z] = 0.0;
          {f'cer1[rowe+z] = 1.0; cei1[rowe+z] = 0.0; cor1[rowo+z] = 1.0; coi1[rowo+z] = 0.0;' if PAR else ''}
        }}"""),
("""          f45e_s2(a245_sr + 5*k1*{P2V}, a245_si + 5*k1*{P2V}, a245_cer + 5*k1*{P2V}, a245_cei + 5*k1*{P2V},
                  a245_tr, a245_ti, 0, {LP1V});""",
"""          {f'const V *ecr = ((t+1) & 1) ? a245_cer1 : a245_cer, *eci = ((t+1) & 1) ? a245_cei1 : a245_cei;' if PAR else 'const V *ecr = a245_cer, *eci = a245_cei;'}
          f45e_s2(a245_sr + 5*k1*{P2V}, a245_si + 5*k1*{P2V}, ecr + 5*k1*{P2V}, eci + 5*k1*{P2V},
                  a245_tr, a245_ti, 0, {LP1V});"""),
("""          f45o_s2(a245_dr + 9*k1p*{P2V}, a245_di + 9*k1p*{P2V}, a245_cor + 9*k1p*{P2V}, a245_coi + 9*k1p*{P2V},
                  a245_tr, a245_ti, 0, {LP1V});
          for (long kq = 0; kq < 9; kq++)
            a245_yz(a245_tr + kq*{P2V}, a245_ti + kq*{P2V});""",
"""          {f'const V *ocr = ((t+1) & 1) ? a245_cor1 : a245_cor, *oci = ((t+1) & 1) ? a245_coi1 : a245_coi;' if PAR else 'const V *ocr = a245_cor, *oci = a245_coi;'}
          f45o_s2(a245_dr + 9*k1p*{P2V}, a245_di + 9*k1p*{P2V}, ocr + 9*k1p*{P2V}, oci + 9*k1p*{P2V},
                  a245_tr, a245_ti, 0, {LP1V});
          for (long kq = 0; kq < 9; kq++)
            a245_yz(a245_tr + kq*{P2V}, a245_ti + kq*{P2V});"""),
("""      for (long k1 = 0; k1 < 9; k1++) {{
        f45e_s2(a245_sr + 5*k1*{P2V}, a245_si + 5*k1*{P2V}, a245_cer + 5*k1*{P2V}, a245_cei + 5*k1*{P2V},
                a245_tr, a245_ti, 0, {LP1V});""",
"""      {f'const V *ecr = (m & 1) ? a245_cer1 : a245_cer, *eci = (m & 1) ? a245_cei1 : a245_cei;' if PAR else 'const V *ecr = a245_cer, *eci = a245_cei;'}
      for (long k1 = 0; k1 < 9; k1++) {{
        f45e_s2(a245_sr + 5*k1*{P2V}, a245_si + 5*k1*{P2V}, ecr + 5*k1*{P2V}, eci + 5*k1*{P2V},
                a245_tr, a245_ti, 0, {LP1V});"""),
("""      for (long k1p = 0; k1p < 5; k1p++) {{
        f45o_s2(a245_dr + 9*k1p*{P2V}, a245_di + 9*k1p*{P2V}, a245_cor + 9*k1p*{P2V}, a245_coi + 9*k1p*{P2V},
                a245_tr, a245_ti, 0, {LP1V});""",
"""      {f'const V *ocr = (m & 1) ? a245_cor1 : a245_cor, *oci = (m & 1) ? a245_coi1 : a245_coi;' if PAR else 'const V *ocr = a245_cor, *oci = a245_coi;'}
      for (long k1p = 0; k1p < 5; k1p++) {{
        f45o_s2(a245_dr + 9*k1p*{P2V}, a245_di + 9*k1p*{P2V}, ocr + 9*k1p*{P2V}, oci + 9*k1p*{P2V},
                a245_tr, a245_ti, 0, {LP1V});"""),
("""    slab45x_extract(m > 1 ? a245_1r : fr, m > 1 ? a245_1i : fi, out1 + 2*v*{L3});
    slab45x_extract(fr, fi, outm + 2*v*{L3});""",
"""    {f'slab45x_extractp(m > 1 ? a245_1r : fr, m > 1 ? a245_1i : fi, out1 + 2*v*{L3}, 1);' if PAR else f'slab45x_extract(m > 1 ? a245_1r : fr, m > 1 ? a245_1i : fi, out1 + 2*v*{L3});'}
    {f'slab45x_extractp(fr, fi, outm + 2*v*{L3}, (int)(m & 1));' if PAR else f'slab45x_extract(fr, fi, outm + 2*v*{L3});'}"""),
]
for old, new in edits:
    assert old in src, old[:60]
    src = src.replace(old, new)
open('gen.py','w').write(src)
print("apfa2 PAR edits applied")
EOF
GENCFG='{"sqpar":{"45":"1"}}' python3 gen.py && gcc -O3 -march=native -shared -fPIC ../implementation.c -o /workdir/implementation.so -lm && python3 /workdir/dev/check.py | grep -E "L=45|ALL"