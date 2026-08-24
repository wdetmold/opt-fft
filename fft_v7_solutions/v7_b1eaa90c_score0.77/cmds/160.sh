cd /workdir/dev && python3 - <<'EOF'
src = open('gen.py').read()
old = """        passx = f\"\"\"{{
          V tr[{L}], ti[{L}];
          fft{L}_core(soa{L}_xr + u, soa{L}_xi + u, tr, ti, {L2}, 1);
          #pragma GCC unroll 2
          for (long i = 0; i < {L}; i++) {{
            long idx = u + i*{L2};
            V zr = tr[i] + soa{L}_cr[idx];
            V zi = ti[i] + soa{L}_ci[idx];
            V f = PW68SEL;
            soa{L}_xr[idx] = zr*f; soa{L}_xi[idx] = zi*f;
          }}
        }}\"\"\""""
new = """        pwbody = []
        for i in range(L):
            fn = ('pw_sqrtnr' if i % 2 == 0 else 'pw_newton') if cfg_pw(L) == 'alt' else 'pw_' + cfg_pw(L)
            pwbody.append(f\"\"\"          {{ long idx = u + {i}*{L2};
            V zr = tr[{i}] + soa{L}_cr[idx];
            V zi = ti[{i}] + soa{L}_ci[idx];
            V f = {fn}(zr, zi);
            soa{L}_xr[idx] = zr*f; soa{L}_xi[idx] = zi*f; }}\"\"\")
        pwb = chr(10).join(pwbody)
        passx = f\"\"\"{{
          V tr[{L}], ti[{L}];
          fft{L}_core(soa{L}_xr + u, soa{L}_xi + u, tr, ti, {L2}, 1);
{pwb}
        }}\"\"\""""
assert old in src
src = src.replace(old, new)
open('gen.py','w').write(src)
print("soa pw unrolled")
EOF
python3 - <<'EOF'
import tune
A = tune.build({'pw':{'8':'sqrtnr','6':'alt'}}, 'pw8A')
B = tune.build({'pw':{'8':'alt','6':'alt'}}, 'pw8B')
for i in range(3):
    ra = tune.bench(A, Ls=(6,8), reps=5)
    rb = tune.bench(B, Ls=(6,8), reps=5)
    print("8=sqrtnr", {L: round(v,2) for L,v in ra.items()}, " 8=alt", {L: round(v,2) for L,v in rb.items()})
EOF