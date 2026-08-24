cd /workdir/dev && python3 - <<'EOF'
src = open('gen.py').read()
old = """        }}}} else {{{{
          f{L}_s2(srcr + g*{G}*{P2V}, srci + g*{G}*{P2V}, pwcr + g*{G}*{P2V}, pwci + g*{G}*{P2V},
                  TAr, TAi, 0, {LP1V});
          for (long k2 = 0; k2 < {G}; k2++) {{{{
            memcpy(sq{L}_1r + ({G}*k2+g)*{P2V}, TAr + k2*{P2V}, {P2V}*sizeof(V));
            memcpy(sq{L}_1i + ({G}*k2+g)*{P2V}, TAi + k2*{P2V}, {P2V}*sizeof(V));
          }}}}
          for (long k2 = 0; k2 < {G}; k2++)
            sq{L}_yz(TAr + k2*{P2V}, TAi + k2*{P2V});
        }}}}"""
new = """        }}}} else {{{{
          f{L}_s2(srcr + g*{G}*{P2V}, srci + g*{G}*{P2V}, pwcr + g*{G}*{P2V}, pwci + g*{G}*{P2V},
                  TAr, TAi, 0, {LP1V});
          if (t == 0) {{{{
            for (long k2 = 0; k2 < {G}; k2++) {{{{
              memcpy(sq{L}_1r + ({G}*k2+g)*{P2V}, TAr + k2*{P2V}, {P2V}*sizeof(V));
              memcpy(sq{L}_1i + ({G}*k2+g)*{P2V}, TAi + k2*{P2V}, {P2V}*sizeof(V));
            }}}}
          }}}}
          for (long k2 = 0; k2 < {G}; k2++)
            sq{L}_yz(TAr + k2*{P2V}, TAi + k2*{P2V});
        }}}}"""
assert old in src
src = src.replace(old, new)
open('gen.py','w').write(src)
print("capture guard fixed")
EOF
python3 gen.py && gcc -O3 -march=native -shared -fPIC ../implementation.c -o /workdir/implementation.so -lm && python3 /workdir/dev/check.py | tail -1 && python3 -c "
import tune
allr={}
for i in range(4):
    r=tune.bench('/workdir/implementation.so', Ls=(64,), reps=5, params={64:(2,20)})
    for L,v in r.items(): allr.setdefault(L,[]).append(v)
print('64:', {L: [round(x,2) for x in v] for L,v in allr.items()})
"