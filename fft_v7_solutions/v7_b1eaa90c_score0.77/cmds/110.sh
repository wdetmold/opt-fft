cd /workdir/dev && python3 - <<'EOF'
src = open('gen.py').read()
old = """      for (long u = 0; u < {L2}; u++) {{
        {passx}
      }}"""
new = """      for (long u = 0; u < {L2}; u++) {{
        if ({K3PF} && (u & 7) == 0 && u + 8 < {L2}) {{
          for (long i = 0; i < {L}; i++) {{
            long a = u + 8 + i*{L2};
            __builtin_prefetch(soa{L}_xr + a, 1, 3); __builtin_prefetch(soa{L}_xi + a, 1, 3);
            __builtin_prefetch(soa{L}_cr + a, 0, 3); __builtin_prefetch(soa{L}_ci + a, 0, 3);
          }}
        }}
        {passx}
      }}"""
assert old in src
src = src.replace(old, new)
src = src.replace("def gen_soa(L, use_inst):",
                  "def gen_soa(L, use_inst):\n    K3PF = int(CONFIG.get('k3pf', {}).get(str(L), 0))")
open('gen.py','w').write(src)
print("ok")
EOF
python3 - <<'EOF'
import tune
A = tune.build({'k3pf':{'23':'0','17':'0','13':'0'}}, 'spf0')
B = tune.build({'k3pf':{'23':'1','17':'1','13':'1'}}, 'spf1')
for i in range(3):
    ra = tune.bench(A, Ls=(13,17,23), reps=4)
    rb = tune.bench(B, Ls=(13,17,23), reps=4)
    print("pf0", {L: round(v,2) for L,v in ra.items()}, " pf1", {L: round(v,2) for L,v in rb.items()})
EOF