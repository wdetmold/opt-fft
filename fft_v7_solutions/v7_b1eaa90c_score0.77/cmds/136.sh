cd /workdir/dev && python3 - <<'EOF'
src = open('gen.py').read()
# 1) define PAR=0 inside gen_apfa2_45 (neutralize leaked conditionals)
src = src.replace("def gen_apfa2_45():\n    L = 45", "def gen_apfa2_45():\n    PAR = 0\n    L = 45")
# 2) define PAR in gen_sq
src = src.replace("def gen_sq(L, G):\n    PFON", "def gen_sq(L, G):\n    PAR = int(CONFIG.get('sqpar', {}).get(str(L), 0))\n    PFON")
# 3) parity yz variant: modify the default yz emission (non-64): second cb writes straight via f{L}_cbs when PAR
old = """  for (long cc = 0; cc < {NB}; cc++) {{{{
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
}}}}'''}"""
new = """  {f'for (long cc = 0; cc < {NB}; cc++) f{L}_cbs(slabts{L}_r + cc, slabts{L}_i + cc, sr + cc, si + cc);' if PAR else f'''for (long cc = 0; cc < {NB}; cc++) {{{{
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
  }}}}'''}
}}}}'''}"""
assert old in src, "yz tail not found"
src = src.replace(old, new)
open('gen.py','w').write(src)
print("gen_sq PAR yz ok")
EOF
python3 - <<'EOF'
src = open('gen.py').read()
# 4) emit f{L}_cbs instance + parity extractor for sq sizes when sqpar
old = """            if L == 64:
                parts.append(kernels.gen_zline64("f64_zline"))
                parts.append(kernels.gen_inst(64, "f64_ycol", P1V, P1V, False))
            parts.append(kernels.gen_inst(L, f"f{L}_cb", P1V, 1, False))"""
new = """            if L == 64:
                parts.append(kernels.gen_zline64("f64_zline"))
                parts.append(kernels.gen_inst(64, "f64_ycol", P1V, P1V, False))
            if int(CONFIG.get('sqpar', {}).get(str(L), 0)):
                parts.append(kernels.gen_inst(L, f"f{L}_cbs", P1V, P1V, False))
                parts.append(gen_sq_extractp(L))
            parts.append(kernels.gen_inst(L, f"f{L}_cb", P1V, 1, False))"""
assert old in src
src = src.replace(old, new)
# extractp generator
genexp = '''
def gen_sq_extractp(L):
    P1 = r8(L)
    P2 = L*P1 + (8 if L == 64 else 0)
    return f"""
static void slab{L}_extractp(const V*restrict ar, const V*restrict ai, double*restrict dst, int parity) {{
  const double *r = (const double*)ar, *im = (const double*)ai;
  for (long x = 0; x < {L}; x++)
    for (long y = 0; y < {L}; y++) {{
      const long rowr = x*{P2} + (parity ? y : y*{P1});
      const long step = parity ? {P1} : 1;
      double *d = dst + 2*((x*{L} + y)*{L});
      for (long z = 0; z < {L}; z++) {{
        d[2*z]   = r[rowr + z*step];
        d[2*z+1] = im[rowr + z*step];
      }}
    }}
}}
"""
'''
src = src.replace("def gen_sq_extract(L):", genexp + "\ndef gen_sq_extract(L):")
open('gen.py','w').write(src)
print("cbs/extractp wired")
EOF
# non-parity build must still work:
python3 gen.py && gcc -fsyntax-only -O3 -march=native ../implementation.c && echo BASE-COMPILES
# parity-36 build:
GENCFG='{"sqpar":{"36":"1"}}' python3 gen.py && gcc -fsyntax-only -O3 -march=native ../implementation.c && echo PAR36-COMPILES