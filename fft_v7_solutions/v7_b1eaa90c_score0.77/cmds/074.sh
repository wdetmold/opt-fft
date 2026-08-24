cd /workdir/dev && python3 - <<'EOF'
# parameterize the driver prefetch step via CONFIG 'dpf' (bytes step; 0=off)
src = open('gen.py').read()
old = """            for (long q = 0; q < {P2V}*64; q += 64) {{
              _mm_prefetch(p0 + q, _MM_HINT_T2); _mm_prefetch(p1 + q, _MM_HINT_T2);
              _mm_prefetch(p2 + q, _MM_HINT_T2); _mm_prefetch(p3 + q, _MM_HINT_T2);
            }}"""
new = """            for (long q = 0; q < {P2V}*64; q += {DPF}) {{
              _mm_prefetch(p0 + q, _MM_HINT_T2); _mm_prefetch(p1 + q, _MM_HINT_T2);
              _mm_prefetch(p2 + q, _MM_HINT_T2); _mm_prefetch(p3 + q, _MM_HINT_T2);
            }}"""
assert old in src
src = src.replace(old, new)
old2 = """          if (g+1 < {G}) {{"""
new2 = """          if ({DPF} && g+1 < {G}) {{"""
assert old2 in src
src = src.replace(old2, new2)
# define DPF in gen_sq
src = src.replace("def gen_sq(L, G):\n    P1 = r8(L)",
                  "def gen_sq(L, G):\n    DPF = int(CONFIG.get('dpf', {}).get(str(L), 0))\n    DPF = DPF if DPF else 1 << 30\n    P1 = r8(L)")
# hmm: if DPF==0 we want loop dead: use condition {DPF_ON}
src = src.replace("""          if ({DPF} && g+1 < {G}) {{""", """          if ({DPF_ON} && g+1 < {G}) {{""")
src = src.replace("def gen_sq(L, G):\n    DPF = int(CONFIG.get('dpf', {}).get(str(L), 0))\n    DPF = DPF if DPF else 1 << 30\n    P1 = r8(L)",
                  "def gen_sq(L, G):\n    DPF = int(CONFIG.get('dpf', {}).get(str(L), 0))\n    DPF_ON = 1 if DPF else 0\n    DPF = DPF if DPF else 64\n    P1 = r8(L)")
open('gen.py','w').write(src)
print("ok")
EOF
for d in 0 128 256; do
python3 -c "
import tune
cfg={'pw':{'13':'alt','17':'alt','23':'alt','45':'alt'}, 'dpf':{'36':'$d','64':'$d'}}
so=tune.build(cfg,'sqd$d')
r=tune.bench(so, Ls=(36,64))
print('dpf=$d', {L: round(v,2) for L,v in r.items()})
"
done