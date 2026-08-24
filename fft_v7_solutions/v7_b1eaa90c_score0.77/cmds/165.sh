cd /workdir/dev && python3 - <<'EOF'
# 1) extend gen_inst to handle N=6 and N=8 composite kernels
src = open('kernels.py').read()
src = src.replace("""def gen_inst(N, fname, IS, OS, pw, PF=0, pwfn='pw_sqrtnr', HU=1):
    if N in (13, 17, 23):
        return gen_halfmatrix_inst(N, fname, IS, OS, pw, PF, pwfn, HU)""",
"""def gen_inst(N, fname, IS, OS, pw, PF=0, pwfn='pw_sqrtnr', HU=1):
    if N in (13, 17, 23):
        return gen_halfmatrix_inst(N, fname, IS, OS, pw, PF, pwfn, HU)
    if N == 6: return gen_composite_inst(6, 2, 3, True, fname, IS, OS, pw, PF, pwfn)
    if N == 8: return gen_composite_inst(8, 4, 2, False, fname, IS, OS, pw, PF, pwfn)""")
open('kernels.py','w').write(src)
print("kernels 6/8 inst ok")
EOF
python3 - <<'EOF'
# 2) restructure gen.py: soa -> group function + slab -> one-volume function + dispatcher
src = open('gen.py').read()

# --- gen_soa: wrap body into group function + dispatcher ---
old_sig = """void run{L}(const double*restrict x0, const double*restrict c, long B, long m,
            double*restrict out1, double*restrict outm) {{
  for (long g0 = 0; g0 < B; g0 += 8) {{
    int lanes = (B - g0) < 8 ? (int)(B - g0) : 8;"""
new_sig = """static void soa{L}_grp(const double*restrict x0, const double*restrict c, long g0, int lanes, long m,
            double*restrict out1, double*restrict outm) {{
  {{"""
assert old_sig in src
src = src.replace(old_sig, new_sig)
# close: the body ends with the extract loop then closes both loops + function; adapt
old_end = """    for (int l = 0; l < lanes; l++) {{
      double *d1 = out1 + 2*(g0+l)*{L3}, *dm = outm + 2*(g0+l)*{L3};
      for (long i = 0; i < {L3}; i++) {{
        d1[2*i] = soa{L}_or[i][l]; d1[2*i+1] = soa{L}_oi[i][l];
        dm[2*i] = soa{L}_xr[i][l]; dm[2*i+1] = soa{L}_xi[i][l];
      }}
    }}
  }}
}}"""
new_end = """    for (int l = 0; l < lanes; l++) {{
      double *d1 = out1 + 2*(g0+l)*{L3}, *dm = outm + 2*(g0+l)*{L3};
      for (long i = 0; i < {L3}; i++) {{
        d1[2*i] = soa{L}_or[i][l]; d1[2*i+1] = soa{L}_oi[i][l];
        dm[2*i] = soa{L}_xr[i][l]; dm[2*i+1] = soa{L}_xi[i][l];
      }}
    }}
  }}
}}

void run{L}(const double*restrict x0, const double*restrict c, long B, long m,
            double*restrict out1, double*restrict outm) {{
  long done = 0;
  while (B - done >= 8) {{ soa{L}_grp(x0, c, done, 8, m, out1, outm); done += 8; }}
  long rem = B - done;
  if (rem > 0) {{
    if (rem >= {THRESH})
      soa{L}_grp(x0, c, done, (int)rem, m, out1, outm);
    else
      for (long v = done; v < B; v++) slab{L}_one(x0, c, v, m, out1, outm);
  }}
}}"""
assert old_end in src
src = src.replace(old_end, new_end)
# THRESH param in gen_soa
src = src.replace("def gen_soa(L, use_inst):\n    K3PF",
                  "def gen_soa(L, use_inst):\n    THRESH = int(CONFIG.get('soathresh', {}).get(str(L), 5))\n    K3PF")

# --- gen_slab: rename run{L} -> slab{L}_one(one volume v) ---
old_slab = """void run{L}(const double*restrict x0, const double*restrict c, long B, long m,
            double*restrict out1, double*restrict outm) {{
  double *xr = (double*)slab{L}_xr, *xi = (double*)slab{L}_xi;
  double *cer = (double*)slab{L}_cer, *cei = (double*)slab{L}_cei;
  for (long v = 0; v < B; v++) {{
    const double *sx = x0 + 2*v*{L3}, *sc = c + 2*v*{L3};"""
new_slab = """static void slab{L}_one(const double*restrict x0, const double*restrict c, long v, long m,
            double*restrict out1, double*restrict outm) {{
  double *xr = (double*)slab{L}_xr, *xi = (double*)slab{L}_xi;
  double *cer = (double*)slab{L}_cer, *cei = (double*)slab{L}_cei;
  {{
    const double *sx = x0 + 2*v*{L3}, *sc = c + 2*v*{L3};"""
assert old_slab in src
src = src.replace(old_slab, new_slab)

# --- main dispatch: for soa sizes, also emit slab instances + gen_slab ---
old_main = """        if cfg_scheme(L) == 'soa':
            if L in (13, 17, 23, 36, 45, 64):"""
new_main = """        if cfg_scheme(L) == 'soa':
            P1 = r8(L); P2 = L*P1
            parts.append(kernels.gen_inst(L, f"f{L}_cb", P1//8, 1, False))
            parts.append(kernels.gen_inst(L, f"f{L}_k3", P2//8, P2//8, True, pwfn="pw_"+("sqrtnr" if cfg_pw(L) in ("sqrtnr","newton") else "alt")))
            parts.append(gen_slab(L))
            if L in (13, 17, 23, 36, 45, 64):"""
assert old_main in src
src = src.replace(old_main, new_main)
open('gen.py','w').write(src)
print("gen.py dispatch ok")
EOF
python3 gen.py 2>&1 | tail -2 && gcc -fsyntax-only -O3 -march=native ../implementation.c && echo COMPILES