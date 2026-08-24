cd /workdir/dev && python3 - <<'EOF'
src = open('gen.py').read()
src = src.replace("""            parts.append(f"static V slabts{L}_r[{P1*P1V}], slabts{L}_i[{P1*P1V}];")
            parts.append(kernels.gen_inst(L, f"f{L}_cb", P1V, 1, False))""",
"""            parts.append(f"static V slabts{L}_r[{P1*P1V}], slabts{L}_i[{P1*P1V}];")
            if L == 64:
                parts.append(kernels.gen_zline64("f64_zline"))
                parts.append(kernels.gen_inst(64, "f64_ycol", P1V, P1V, False))
            parts.append(kernels.gen_inst(L, f"f{L}_cb", P1V, 1, False))""")
# yz body switch for 64: patch the generated string template: add a python-side conditional
src = src.replace("""    twtabs = (f"static const double sq{L}_twr[{G*G}] = {{{','.join(hexf(v) for v in twr)}}};\\n"
              f"static const double sq{L}_twi[{G*G}] = {{{','.join(hexf(v) for v in twi)}}};\\n")""",
"""    twtabs = (f"static const double sq{L}_twr[{G*G}] = {{{','.join(hexf(v) for v in twr)}}};\\n"
              f"static const double sq{L}_twi[{G*G}] = {{{','.join(hexf(v) for v in twi)}}};\\n")
    if L == 64:
        twtabs += f'''static void sq64_yz(V* sr, V* si) {{
  for (long y = 0; y < 64; y++)
    f64_zline(sr + y*{P1V}, si + y*{P1V});
  for (long cc = 0; cc < {P1V}; cc++)
    f64_ycol(sr + cc, si + cc, sr + cc, si + cc);
}}
#define SQ64_YZ_DEFINED
'''""")
# wrap default yz in #ifndef to avoid duplicate for 64
src = src.replace("""static void sq{L}_yz(V* sr, V* si) {{
  for (long cc = 0; cc < {NB}; cc++) {{
    V *br_ = sq{L}_buf_r + (cc&1)*{P1}, *bi_ = sq{L}_buf_i + (cc&1)*{P1};""",
"""#ifndef SQ64_YZ_DEFINED
static void sq{L}_yz(V* sr, V* si) {{
  for (long cc = 0; cc < {NB}; cc++) {{
    V *br_ = sq{L}_buf_r + (cc&1)*{P1}, *bi_ = sq{L}_buf_i + (cc&1)*{P1};""")
src = src.replace("""      tr8x8(bi_ + rb*8, tb);
      for (int q = 0; q < lim; q++) si[(cc*8+q)*{P1V} + rb] = tb[q];
    }}
  }}
}}""","""      tr8x8(bi_ + rb*8, tb);
      for (int q = 0; q < lim; q++) si[(cc*8+q)*{P1V} + rb] = tb[q];
    }}
  }}
}}
#endif
#undef SQ64_YZ_DEFINED""")
open('gen.py','w').write(src)
print("patched")
EOF
python3 gen.py && gcc -fsyntax-only -O3 -march=native ../implementation.c && echo COMPILES && python3 -c "
import tune
so=tune.build({},'zl2')
for i in range(2):
    r=tune.bench(so, Ls=(64,), reps=5)
    print({L: round(v,2) for L,v in r.items()})
"