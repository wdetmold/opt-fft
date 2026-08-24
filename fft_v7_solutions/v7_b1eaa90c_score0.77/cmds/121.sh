cd /workdir/dev && python3 - <<'EOF'
src = open('gen.py').read()
old = """        if (g == 0)
          f{L}_s2(srcr, srci, sq{L}_cr, sq{L}_ci, TAr, TAi, 0, {LP1V});
        if (t == 0) {{
          for (long k2 = 0; k2 < {G}; k2++) {{
            memcpy(sq{L}_1r + ({G}*k2+g)*{P2V}, TAr + k2*{P2V}, {P2V}*sizeof(V));
            memcpy(sq{L}_1i + ({G}*k2+g)*{P2V}, TAi + k2*{P2V}, {P2V}*sizeof(V));
          }}
        }}
        for (long k2 = 0; k2 < {G}; k2++) {{
          long gb = (g+1)*{G}*{P2V};
          if ({PFON} && g+1 < {G}) {{
            long o0 = k2*{SLICE};
            for (long j = 0; j < {G}; j++) {{
              const char *p0 = (const char*)(srcr + gb + j*{P2V} + o0);
              const char *p1 = (const char*)(srci + gb + j*{P2V} + o0);
              const char *p2 = (const char*)(sq{L}_cr + gb + j*{P2V} + o0);
              const char *p3 = (const char*)(sq{L}_ci + gb + j*{P2V} + o0);
              for (long q = 0; q < {SLICE}*64; q += 64) {{
                _mm_prefetch(p0 + q, _MM_HINT_T2); _mm_prefetch(p1 + q, _MM_HINT_T2);
                _mm_prefetch(p2 + q, _MM_HINT_T2); _mm_prefetch(p3 + q, _MM_HINT_T2);
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
      }}"""
new = """        f{L}_s2(srcr + g*{G}*{P2V}, srci + g*{G}*{P2V}, sq{L}_cr + g*{G}*{P2V}, sq{L}_ci + g*{G}*{P2V},
                TAr, TAi, 0, {LP1V});
        if (t == 0) {{
          for (long k2 = 0; k2 < {G}; k2++) {{
            memcpy(sq{L}_1r + ({G}*k2+g)*{P2V}, TAr + k2*{P2V}, {P2V}*sizeof(V));
            memcpy(sq{L}_1i + ({G}*k2+g)*{P2V}, TAi + k2*{P2V}, {P2V}*sizeof(V));
          }}
        }}
        for (long k2 = 0; k2 < {G}; k2++)
          sq{L}_yz(TAr + k2*{P2V}, TAi + k2*{P2V});
        f{L}_s1(TAr, TAi, dstr + g*{P2V}, dsti + g*{P2V}, sq{L}_twr + g*{G}, sq{L}_twi + g*{G});
      }}"""
assert old in src
src = src.replace(old, new)
open('gen.py','w').write(src)
print("reverted to simple sweep")
EOF
python3 gen.py >/dev/null && gcc -O3 -march=native -shared -fPIC ../implementation.c -o /tmp/impl_rev.so -lm && python3 -c "
import tune
r1=tune.bench('/tmp/impl_rev.so', Ls=(36,64), reps=4)
r2=tune.bench('/tmp/impl_rev.so', Ls=(36,64), reps=4)
print('reverted', {L: round(min(r1[L],r2[L]),2) for L in (36,64)})
"