cd /workdir/dev && python3 - <<'EOF'
src = open('gen.py').read()
old = """    for (long o = 0; o < {L*P1V}; o++)
      f{L}_k3(slab{L}_xr + o, slab{L}_xi + o, slab{L}_xr + o, slab{L}_xi + o,
              slab{L}_cer + o, slab{L}_cei + o);"""
new = """    for (long o = 0; o < {L*P1V}; o++) {{
      if ({K3PF} && (o & 7) == 0 && o + 8 < {L*P1V}) {{
        for (long i = 0; i < {L}; i++) {{
          long a = o + 8 + i*{P2V};
          __builtin_prefetch(slab{L}_xr + a, 1, 3); __builtin_prefetch(slab{L}_xi + a, 1, 3);
          __builtin_prefetch(slab{L}_cer + a, 0, 3); __builtin_prefetch(slab{L}_cei + a, 0, 3);
        }}
      }}
      f{L}_k3(slab{L}_xr + o, slab{L}_xi + o, slab{L}_xr + o, slab{L}_xi + o,
              slab{L}_cer + o, slab{L}_cei + o);
    }}"""
assert old in src
src = src.replace(old, new)
src = src.replace("def gen_slab(L):\n    P1 = r8(L)",
                  "def gen_slab(L):\n    K3PF = int(CONFIG.get('k3pf', {}).get(str(L), 0))\n    P1 = r8(L)")
open('gen.py','w').write(src)
print("ok")
EOF
python3 - <<'EOF'
import tune
A = tune.build({'scheme':{'45':'slab'}, 'k3pf':{'45':'0'}}, 'k3pf0')
B = tune.build({'scheme':{'45':'slab'}, 'k3pf':{'45':'1'}}, 'k3pf1')
for i in range(3):
    ra = tune.bench(A, Ls=(45,), reps=4)
    rb = tune.bench(B, Ls=(45,), reps=4)
    print("pf0", round(ra[45],3), " pf-blk", round(rb[45],3))
EOF