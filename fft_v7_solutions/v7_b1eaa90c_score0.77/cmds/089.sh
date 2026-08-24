cd /workdir/dev && python3 - <<'EOF'
src = open('gen.py').read()
old = """        passx = f\"\"\"{{
          V tr[{L}], ti[{L}];
          fft{L}_core(soa{L}_xr + u, soa{L}_xi + u, tr, ti, {L2}, 1);
          for (long i = 0; i < {L}; i++) {{
            long idx = u + i*{L2};
            V zr = tr[i] + soa{L}_cr[idx];
            V zi = ti[i] + soa{L}_ci[idx];
            V f = PW68(zr, zi);
            soa{L}_xr[idx] = zr*f; soa{L}_xi[idx] = zi*f;
          }}
        }}\"\"\""""
new = """        passx = f\"\"\"{{
          V tr[{L}], ti[{L}];
          fft{L}_core(soa{L}_xr + u, soa{L}_xi + u, tr, ti, {L2}, 1);
          #pragma GCC unroll 2
          for (long i = 0; i < {L}; i++) {{
            long idx = u + i*{L2};
            V zr = tr[i] + soa{L}_cr[idx];
            V zi = ti[i] + soa{L}_ci[idx];
            V f = (i & 1) ? pw_newton(zr, zi) : pw_sqrtnr(zr, zi);
            soa{L}_xr[idx] = zr*f; soa{L}_xi[idx] = zi*f;
          }}
        }}\"\"\""""
assert old in src
src = src.replace(old, new)
src = src.replace("def cfg_pf(L): return CONFIG['pf'].get(str(L), 256)",
                  "def cfg_pf(L): return CONFIG['pf'].get(str(L), 0)")
open('gen.py','w').write(src)
print("ok")
EOF
python3 -c "
import tune
cfg={'pw':{'13':'alt','17':'alt','23':'alt','45':'alt'},'hu':{'13':'16'}}
so=tune.build(cfg,'cons')
import json
for i in range(2):
    r=tune.bench(so, reps=5)
    print({L: round(v,2) for L,v in r.items()})
"