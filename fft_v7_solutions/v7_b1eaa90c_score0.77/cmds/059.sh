cd /workdir/dev && python3 - <<'EOF'
# parameterize gen.py: CONFIG dict controlling pw impl + PF per size
src = open('gen.py').read()
src = src.replace("""SIZES = (6, 8, 13, 17, 23, 36, 45, 64)""",
"""SIZES = (6, 8, 13, 17, 23, 36, 45, 64)
import os, json
CONFIG = {'pw': {}, 'pf': {}}
if os.environ.get('GENCFG'):
    CONFIG.update(json.loads(os.environ['GENCFG']))
def cfg_pw(L): return CONFIG['pw'].get(str(L), 'sqrtnr')
def cfg_pf(L): return CONFIG['pf'].get(str(L), 256)""")
# pw_factor: emit BOTH, select per instance via macro name passed to kernels? Simplest: two functions pw_newton/pw_sqrtnr and a per-size define
old = """static inline V vsqrtv(V x){ return (V)_mm512_sqrt_pd((__m512d)x); }
static inline __attribute__((always_inline)) V pw_factor(V zr, V zi){
  V s = zr*zr + zi*zi;
  V u = VC(1.0) + vsqrtv(s);
  V v = vrcp14(u);
  v = v + v*(VC(1.0) - u*v);
  v = v + v*(VC(1.0) - u*v);
  return v;
}"""
new = """static inline V vsqrtv(V x){ return (V)_mm512_sqrt_pd((__m512d)x); }
static inline __attribute__((always_inline)) V pw_sqrtnr(V zr, V zi){
  V s = zr*zr + zi*zi;
  V u = VC(1.0) + vsqrtv(s);
  V v = vrcp14(u);
  v = v + v*(VC(1.0) - u*v);
  v = v + v*(VC(1.0) - u*v);
  return v;
}
static inline __attribute__((always_inline)) V pw_newton(V zr, V zi){
  V s = zr*zr + zi*zi;
  s = vmaxv(s, VC(2.2250738585072014e-308));
  V r = vrsqrt14(s);
  V h = s * VC(0.5);
  r = r * (VC(1.5) - h*r*r);
  r = r * (VC(1.5) - h*r*r);
  V t = s * r;
  V u = VC(1.0) + t;
  V v = vrcp14(u);
  v = v + v*(VC(1.0) - u*v);
  v = v + v*(VC(1.0) - u*v);
  return v;
}"""
assert old in src
src = src.replace(old, new)
open('gen.py','w').write(src)
EOF
python3 - <<'EOF'
# kernels.py: pw function name parameter
src = open('kernels.py').read()
src = src.replace("pw_factor(zr, zi)", "PWFN(zr, zi)")
# define PWFN per kernel: add a define before each function and undef after
src = src.replace('''    code = pre + f"""static __attribute__((noinline)) void {fname}({args}) {{
{decl_tabs}  V S1r[{N}], S1i[{N}];''','''    code = pre + f"""#define PWFN {pwfn}
static __attribute__((noinline)) void {fname}({args}) {{
{decl_tabs}  V S1r[{N}], S1i[{N}];''')
    
src = src.replace('''  for (long k1 = 0; k1 < {P}; k1++) {{
{passB}
  }}
}}
"""
    return code''','''  for (long k1 = 0; k1 < {P}; k1++) {{
{passB}
  }}
}}
#undef PWFN
"""
    return code''')
src = src.replace('''    code = pre + f"""static __attribute__((noinline)) void {fname}({args}) {{
  V ur[{h+1}]''','''    code = pre + f"""#define PWFN {pwfn}
static __attribute__((noinline)) void {fname}({args}) {{
  V ur[{h+1}]''')
src = src.replace('''  }}{odd_tail}
}}
"""
    return code''','''  }}{odd_tail}
}}
#undef PWFN
"""
    return code''')
src = src.replace("def gen_composite_inst(N, P, Q, pfa, fname, IS, OS, pw, PF=0):",
                  "def gen_composite_inst(N, P, Q, pfa, fname, IS, OS, pw, PF=0, pwfn='pw_sqrtnr'):")
src = src.replace("def gen_halfmatrix_inst(N, fname, IS, OS, pw, PF=0):",
                  "def gen_halfmatrix_inst(N, fname, IS, OS, pw, PF=0, pwfn='pw_sqrtnr'):")
src = src.replace("def gen_inst(N, fname, IS, OS, pw, PF=0):",
                  "def gen_inst(N, fname, IS, OS, pw, PF=0, pwfn='pw_sqrtnr'):")
src = src.replace("return gen_halfmatrix_inst(N, fname, IS, OS, pw, PF)", "return gen_halfmatrix_inst(N, fname, IS, OS, pw, PF, pwfn)")
src = src.replace("return gen_composite_inst(36, 4, 9, True, fname, IS, OS, pw, PF)", "return gen_composite_inst(36, 4, 9, True, fname, IS, OS, pw, PF, pwfn)")
src = src.replace("return gen_composite_inst(45, 9, 5, True, fname, IS, OS, pw, PF)", "return gen_composite_inst(45, 9, 5, True, fname, IS, OS, pw, PF, pwfn)")
src = src.replace("return gen_composite_inst(64, 8, 8, False, fname, IS, OS, pw, PF)", "return gen_composite_inst(64, 8, 8, False, fname, IS, OS, pw, PF, pwfn)")
open('kernels.py','w').write(src)
EOF
python3 - <<'EOF'
# gen.py: pass pw/pf config into instances
src = open('gen.py').read()
src = src.replace('parts.append(kernels.gen_inst(L, f"f{L}_k3", P2//8, P2//8, True, PF=256))',
                  'parts.append(kernels.gen_inst(L, f"f{L}_k3", P2//8, P2//8, True, PF=cfg_pf(L), pwfn="pw_"+cfg_pw(L)))')
src = src.replace('parts.append(kernels.gen_inst(13, "f13_x", 169, 169, True, PF=256))',
                  'parts.append(kernels.gen_inst(13, "f13_x", 169, 169, True, PF=cfg_pf(13), pwfn="pw_"+cfg_pw(13)))')
# soa non-inst path uses pw_factor symbol: rename to pw config for 6/8
src = src.replace("V f = pw_factor(zr, zi);", "V f = PW68(zr, zi);")
src = src.replace("def gen_soa(L, use_inst):", "def gen_soa(L, use_inst):\n    pwdef = f'#define PW68 pw_{cfg_pw(L)}'")
src = src.replace("""    return f'''
// ---------------- L={L} : SoA across 8 volumes ----------------""","""    return f'''
{pwdef}
// ---------------- L={L} : SoA across 8 volumes ----------------""")
src = src.replace("""    for (int l = 0; l < lanes; l++) {{
      double *d1 = out1 + 2*(g0+l)*{L3}, *dm = outm + 2*(g0+l)*{L3};
      for (long i = 0; i < {L3}; i++) {{
        d1[2*i] = soa{L}_or[i][l]; d1[2*i+1] = soa{L}_oi[i][l];
        dm[2*i] = soa{L}_xr[i][l]; dm[2*i+1] = soa{L}_xi[i][l];
      }}
    }}
  }}
}}
'''""","""    for (int l = 0; l < lanes; l++) {{
      double *d1 = out1 + 2*(g0+l)*{L3}, *dm = outm + 2*(g0+l)*{L3};
      for (long i = 0; i < {L3}; i++) {{
        d1[2*i] = soa{L}_or[i][l]; d1[2*i+1] = soa{L}_oi[i][l];
        dm[2*i] = soa{L}_xr[i][l]; dm[2*i+1] = soa{L}_xi[i][l];
      }}
    }}
  }}
}}
#undef PW68
'''""")
open('gen.py','w').write(src)
print("ok")
EOF
python3 gen.py && head -5 ../implementation.c && gcc -fsyntax-only -O2 -march=native ../implementation.c && echo COMPILES