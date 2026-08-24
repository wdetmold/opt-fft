cd /workdir/dev && python3 - <<'EOF'
src = open('gen.py').read()
old = """    V *srcr = sq{L}_br, *srci = sq{L}_bi, *dstr = sq{L}_ar, *dsti = sq{L}_ai;
    for (long t = 0; t < m-1; t++) {{
      for (long g = 0; g < {G}; g++) {{
        f{L}_s2(srcr + g*{G}*{P2V}, srci + g*{G}*{P2V}, sq{L}_cr + g*{G}*{P2V}, sq{L}_ci + g*{G}*{P2V},
                sq{L}_tr, sq{L}_ti);
        if (t == 0) {{
          for (long k2 = 0; k2 < {G}; k2++) {{
            memcpy(sq{L}_1r + ({G}*k2+g)*{P2V}, sq{L}_tr + k2*{P2V}, {P2V}*sizeof(V));
            memcpy(sq{L}_1i + ({G}*k2+g)*{P2V}, sq{L}_ti + k2*{P2V}, {P2V}*sizeof(V));
          }}
        }}
        for (long k2 = 0; k2 < {G}; k2++)
          sq{L}_yz(sq{L}_tr + k2*{P2V}, sq{L}_ti + k2*{P2V});
        f{L}_s1(sq{L}_tr, sq{L}_ti, dstr + g*{P2V}, dsti + g*{P2V}, sq{L}_twr + g*{G}, sq{L}_twi + g*{G});
      }}"""
new = """    V *srcr = sq{L}_br, *srci = sq{L}_bi, *dstr = sq{L}_ar, *dsti = sq{L}_ai;
    for (long t = 0; t < m-1; t++) {{
      for (long g = 0; g < {G}; g++) {{
        f{L}_s2(srcr + g*{G}*{P2V}, srci + g*{G}*{P2V}, sq{L}_cr + g*{G}*{P2V}, sq{L}_ci + g*{G}*{P2V},
                sq{L}_tr, sq{L}_ti);
        if (t == 0) {{
          for (long k2 = 0; k2 < {G}; k2++) {{
            memcpy(sq{L}_1r + ({G}*k2+g)*{P2V}, sq{L}_tr + k2*{P2V}, {P2V}*sizeof(V));
            memcpy(sq{L}_1i + ({G}*k2+g)*{P2V}, sq{L}_ti + k2*{P2V}, {P2V}*sizeof(V));
          }}
        }}
        for (long k2 = 0; k2 < {G}; k2++) {{
          sq{L}_yz(sq{L}_tr + k2*{P2V}, sq{L}_ti + k2*{P2V});
          if (g+1 < {G}) {{
            long gb = (g+1)*{G}*{P2V} + k2*{P2V};
            const char *p0 = (const char*)(srcr + gb), *p1 = (const char*)(srci + gb);
            const char *p2 = (const char*)(sq{L}_cr + gb), *p3 = (const char*)(sq{L}_ci + gb);
            for (long q = 0; q < {P2V}*64; q += 64) {{
              _mm_prefetch(p0 + q, _MM_HINT_T2); _mm_prefetch(p1 + q, _MM_HINT_T2);
              _mm_prefetch(p2 + q, _MM_HINT_T2); _mm_prefetch(p3 + q, _MM_HINT_T2);
            }}
          }}
        }}
        f{L}_s1(sq{L}_tr, sq{L}_ti, dstr + g*{P2V}, dsti + g*{P2V}, sq{L}_twr + g*{G}, sq{L}_twi + g*{G});
      }}"""
assert old in src
src = src.replace(old, new)
open('gen.py','w').write(src)
print("prefetch added")
EOF
python3 -c "
import tune
cfg={'pw':{'13':'alt','17':'alt','23':'alt','45':'alt'}}
so=tune.build(cfg,'sq2')
r=tune.bench(so, Ls=(36,64))
print({L: round(v,2) for L,v in r.items()})
"