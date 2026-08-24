cd /workdir/dev && python3 - <<'EOF'
# 1) s2 kernel: add runtime o-range
src = open('kernels.py').read()
old = """    return f\"\"\"static __attribute__((noinline)) void {fname}(const V* xr, const V* xi, const V* cr, const V* ci, V* tr, V* ti) {{
  #pragma GCC unroll 1
  for (long o = 0; o < {LP1V}; o++) {{
{pf}{e.code(indent="    ")}
{chr(10).join(st)}
  }}
}}
\"\"\""""
new = """    return f\"\"\"static __attribute__((noinline)) void {fname}(const V* xr, const V* xi, const V* cr, const V* ci, V* tr, V* ti, long o0, long o1) {{
  #pragma GCC unroll 1
  for (long o = o0; o < o1; o++) {{
{pf}{e.code(indent="    ")}
{chr(10).join(st)}
  }}
}}
\"\"\""""
assert old in src
src = src.replace(old, new)
open('kernels.py','w').write(src)
print("kernels s2 sliced")
EOF
python3 - <<'EOF'
src = open('gen.py').read()
# update all s2/s2e/s2d/s2t call sites with (0, LP1V) and add pipelined sweep for sq
# LP1V per L: 64: 512; 36: 180; 45: 270
src = src.replace("""        f{L}_s2(srcr + g*{G}*{P2V}, srci + g*{G}*{P2V}, sq{L}_cr + g*{G}*{P2V}, sq{L}_ci + g*{G}*{P2V},
                sq{L}_tr, sq{L}_ti);
        if (t == 0) {{
          for (long k2 = 0; k2 < {G}; k2++) {{
            memcpy(sq{L}_1r + ({G}*k2+g)*{P2V}, sq{L}_tr + k2*{P2V}, {P2V}*sizeof(V));
            memcpy(sq{L}_1i + ({G}*k2+g)*{P2V}, sq{L}_ti + k2*{P2V}, {P2V}*sizeof(V));
          }}
        }}
        for (long k2 = 0; k2 < {G}; k2++) {{
          sq{L}_yz(sq{L}_tr + k2*{P2V}, sq{L}_ti + k2*{P2V});
          if ({DPF_ON} && g+1 < {G}) {{
            long gb = (g+1)*{G}*{P2V} + k2*{P2V};
            const char *p0 = (const char*)(srcr + gb), *p1 = (const char*)(srci + gb);
            const char *p2 = (const char*)(sq{L}_cr + gb), *p3 = (const char*)(sq{L}_ci + gb);
            for (long q = 0; q < {P2V}*64; q += {DPF}) {{
              _mm_prefetch(p0 + q, _MM_HINT_T2); _mm_prefetch(p1 + q, _MM_HINT_T2);
              _mm_prefetch(p2 + q, _MM_HINT_T2); _mm_prefetch(p3 + q, _MM_HINT_T2);
            }}
          }}
        }}
        f{L}_s1(sq{L}_tr, sq{L}_ti, dstr + g*{P2V}, dsti + g*{P2V}, sq{L}_twr + g*{G}, sq{L}_twi + g*{G});
      }}""",
"""        if (g == 0)
          f{L}_s2(srcr, srci, sq{L}_cr, sq{L}_ci, TAr, TAi, 0, {LP1V});
        if (t == 0) {{
          for (long k2 = 0; k2 < {G}; k2++) {{
            memcpy(sq{L}_1r + ({G}*k2+g)*{P2V}, TAr + k2*{P2V}, {P2V}*sizeof(V));
            memcpy(sq{L}_1i + ({G}*k2+g)*{P2V}, TAi + k2*{P2V}, {P2V}*sizeof(V));
          }}
        }}
        for (long k2 = 0; k2 < {G}; k2++) {{
          long gb = (g+1)*{G}*{P2V};
          if (g+1 < {G}) {{
            long o0 = k2*{SLICE};
            for (long j = 0; j < {G}; j++) {{
              const char *p0 = (const char*)(srcr + gb + j*{P2V} + o0);
              const char *p1 = (const char*)(srci + gb + j*{P2V} + o0);
              const char *p2 = (const char*)(sq{L}_cr + gb + j*{P2V} + o0);
              const char *p3 = (const char*)(sq{L}_ci + gb + j*{P2V} + o0);
              for (long q = 0; q < {SLICE}*64; q += 64) {{
                _mm_prefetch(p0 + q, _MM_HINT_T0); _mm_prefetch(p1 + q, _MM_HINT_T0);
                _mm_prefetch(p2 + q, _MM_HINT_T0); _mm_prefetch(p3 + q, _MM_HINT_T0);
              }}
            }}
          }}
          sq{L}_yz(TAr + k2*{P2V}, TAi + k2*{P2V});
          if (g+1 < {G})
            f{L}_s2(srcr + gb, srci + gb, sq{L}_cr + gb, sq{L}_ci + gb, TBr, TBi, k2*{SLICE}, (k2+1)*{SLICE});
        }}
        f{L}_s1(TAr, TAi, dstr + g*{P2V}, dsti + g*{P2V}, sq{L}_twr + g*{G}, sq{L}_twi + g*{G});
        {{ V *sw;
           sw = TAr; TAr = TBr; TBr = sw;
           sw = TAi; TAi = TBi; TBi = sw; }}
      }}""")
# T buffers: double
src = src.replace("static V sq{L}_tr[{G*P2V}], sq{L}_ti[{G*P2V}];",
                  "static V sq{L}_tr[{2*G*P2V}], sq{L}_ti[{2*G*P2V}];")
# declare TA/TB at sweep start
src = src.replace("""    V *srcr = sq{L}_br, *srci = sq{L}_bi, *dstr = sq{L}_ar, *dsti = sq{L}_ai;
    for (long t = 0; t < m-1; t++) {{""",
"""    V *srcr = sq{L}_br, *srci = sq{L}_bi, *dstr = sq{L}_ar, *dsti = sq{L}_ai;
    V *TAr = sq{L}_tr, *TAi = sq{L}_ti, *TBr = sq{L}_tr + {G*P2V}, *TBi = sq{L}_ti + {G*P2V};
    for (long t = 0; t < m-1; t++) {{""")
# epilogue call signature
src = src.replace("""      f{L}_s2e(srcr + g*{G}*{P2V}, srci + g*{G}*{P2V}, sq{L}_cr + g*{G}*{P2V}, sq{L}_ci + g*{G}*{P2V},
               dstr + g*{P2V}, dsti + g*{P2V});""",
"""      f{L}_s2e(srcr + g*{G}*{P2V}, srci + g*{G}*{P2V}, sq{L}_cr + g*{G}*{P2V}, sq{L}_ci + g*{G}*{P2V},
               dstr + g*{P2V}, dsti + g*{P2V}, 0, {LP1V});""")
# SLICE definition in gen_sq
src = src.replace("def gen_sq(L, G):\n    DPF", "def gen_sq(L, G):\n    SLICE = (L * r8(L)//8) // G\n    DPF")
open('gen.py','w').write(src)
print("gen sq pipelined")
EOF
python3 - <<'EOF'
# apfa45 call sites also use s2d/s2t with ranges
src = open('gen.py').read()
src = src.replace("""        f45_s2d(ap45_sr + 5*k1*{P2V}, ap45_si + 5*k1*{P2V}, ap45_cr + 5*k1*{P2V}, ap45_ci + 5*k1*{P2V},
                ap45_dr + k1*{P2V}, ap45_di + k1*{P2V});""",
"""        f45_s2d(ap45_sr + 5*k1*{P2V}, ap45_si + 5*k1*{P2V}, ap45_cr + 5*k1*{P2V}, ap45_ci + 5*k1*{P2V},
                ap45_dr + k1*{P2V}, ap45_di + k1*{P2V}, 0, {LP1V});""")
src = src.replace("""      f45_s2t(ap45_sr + 5*k1*{P2V}, ap45_si + 5*k1*{P2V}, ap45_cr + 5*k1*{P2V}, ap45_ci + 5*k1*{P2V},
              ap45_tr, ap45_ti);""",
"""      f45_s2t(ap45_sr + 5*k1*{P2V}, ap45_si + 5*k1*{P2V}, ap45_cr + 5*k1*{P2V}, ap45_ci + 5*k1*{P2V},
              ap45_tr, ap45_ti, 0, {LP1V});""")
# LP1V var exists in gen_apfa45? it computes LP1V ✓
open('gen.py','w').write(src)
print("ok")
EOF
python3 gen.py && gcc -fsyntax-only -O3 -march=native ../implementation.c && echo COMPILES