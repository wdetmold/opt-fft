cd /workdir/dev && python3 - <<'EOF'
src = open('kernels.py').read()

# composite: add prefetch in passA loop (x arrays) and passB (c arrays) when pw
old = """    passA = e.code(indent="    ") + "\\n" + "\\n".join(stA)"""
new = """    pfA = ""
    if PF:
        pfs = []
        for a in range(P):
            if pfa:
                pfs.append(f"    __builtin_prefetch((const char*)(xr + (long)IN[b*{P}+{a}]*{IS}) + {PF}, 1, 3);")
                pfs.append(f"    __builtin_prefetch((const char*)(xi + (long)IN[b*{P}+{a}]*{IS}) + {PF}, 1, 3);")
            else:
                pfs.append(f"    __builtin_prefetch((const char*)(xr + ({Q}*{a})*{IS} + bIS) + {PF}, 1, 3);")
                pfs.append(f"    __builtin_prefetch((const char*)(xi + ({Q}*{a})*{IS} + bIS) + {PF}, 1, 3);")
        pfA = "\\n".join(pfs) + "\\n"
    passA = pfA + e.code(indent="    ") + "\\n" + "\\n".join(stA)"""
assert old in src
src = src.replace(old, new, 1)

old2 = """    passB = e2.code(indent="    ") + "\\n" + "\\n".join(stB)
    args ="""
new2 = """    pfB = ""
    if PF and pw:
        pfs = []
        for k2 in range(Q):
            ko = f"(long)OUT[k1*{Q}+{k2}]" if pfa else f"({P}*{k2}+k1)"
            pfs.append(f"    __builtin_prefetch((const char*)(cr + {ko}*{OS}) + {PF}, 0, 3);")
            pfs.append(f"    __builtin_prefetch((const char*)(ci + {ko}*{OS}) + {PF}, 0, 3);")
        pfB = "\\n".join(pfs) + "\\n"
    passB = pfB + e2.code(indent="    ") + "\\n" + "\\n".join(stB)
    args ="""
assert old2 in src
src = src.replace(old2, new2, 1)

# signatures: add PF params
src = src.replace("def gen_composite_inst(N, P, Q, pfa, fname, IS, OS, pw):",
                  "def gen_composite_inst(N, P, Q, pfa, fname, IS, OS, pw, PF=0):")
src = src.replace("def gen_halfmatrix_inst(N, fname, IS, OS, pw):",
                  "def gen_halfmatrix_inst(N, fname, IS, OS, pw, PF=0):")
# halfmatrix: prefetch in j-loop
old3 = """  #pragma GCC unroll 1
  for (long j = 1; j <= {h}; j++) {{
    V adr = xr[j*{IS}], adi = xi[j*{IS}], bdr = xr[({N}-j)*{IS}], bdi = xi[({N}-j)*{IS}];"""
new3 = """  #pragma GCC unroll 1
  for (long j = 1; j <= {h}; j++) {{
{pf_jloop}    V adr = xr[j*{IS}], adi = xi[j*{IS}], bdr = xr[({N}-j)*{IS}], bdi = xi[({N}-j)*{IS}];"""
assert old3 in src
src = src.replace(old3, new3)
old4 = '''    code = pre + f"""static __attribute__((noinline)) void {fname}({args}) {{'''
new4 = '''    pf_jloop = ""
    if PF:
        pf_jloop = (f"    __builtin_prefetch((const char*)(xr + j*{IS}) + {PF}, 1, 3);\\n"
                    f"    __builtin_prefetch((const char*)(xi + j*{IS}) + {PF}, 1, 3);\\n"
                    f"    __builtin_prefetch((const char*)(xr + ({N}-j)*{IS}) + {PF}, 1, 3);\\n"
                    f"    __builtin_prefetch((const char*)(xi + ({N}-j)*{IS}) + {PF}, 1, 3);\\n")
        if pw:
            pf_jloop += (f"    __builtin_prefetch((const char*)(cr + j*{OS}) + {PF}, 0, 3);\\n"
                         f"    __builtin_prefetch((const char*)(ci + j*{OS}) + {PF}, 0, 3);\\n"
                         f"    __builtin_prefetch((const char*)(cr + ({N}-j)*{OS}) + {PF}, 0, 3);\\n"
                         f"    __builtin_prefetch((const char*)(ci + ({N}-j)*{OS}) + {PF}, 0, 3);\\n")
    code = pre + f"""static __attribute__((noinline)) void {fname}({args}) {{'''
assert old4 in src
src = src.replace(old4, new4)
src = src.replace("def gen_inst(N, fname, IS, OS, pw):", "def gen_inst(N, fname, IS, OS, pw, PF=0):")
src = src.replace("return gen_halfmatrix_inst(N, fname, IS, OS, pw)", "return gen_halfmatrix_inst(N, fname, IS, OS, pw, PF)")
src = src.replace("return gen_composite_inst(36, 4, 9, True, fname, IS, OS, pw)", "return gen_composite_inst(36, 4, 9, True, fname, IS, OS, pw, PF)")
src = src.replace("return gen_composite_inst(45, 9, 5, True, fname, IS, OS, pw)", "return gen_composite_inst(45, 9, 5, True, fname, IS, OS, pw, PF)")
src = src.replace("return gen_composite_inst(64, 8, 8, False, fname, IS, OS, pw)", "return gen_composite_inst(64, 8, 8, False, fname, IS, OS, pw, PF)")
open('kernels.py','w').write(src)
print("kernels.py patched")
EOF
python3 - <<'EOF'
# gen.py: use sqrt+nr pw_factor, and PF on k3 instances
src = open('gen.py').read()
old = """static inline __attribute__((always_inline)) V pw_factor(V zr, V zi){
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
new = """static inline V vsqrtv(V x){ return (V)_mm512_sqrt_pd((__m512d)x); }
static inline __attribute__((always_inline)) V pw_factor(V zr, V zi){
  V s = zr*zr + zi*zi;
  V u = VC(1.0) + vsqrtv(s);
  V v = vrcp14(u);
  v = v + v*(VC(1.0) - u*v);
  v = v + v*(VC(1.0) - u*v);
  return v;
}"""
assert old in src
src = src.replace(old, new)
src = src.replace('parts.append(kernels.gen_inst(L, f"f{L}_k3", P2//8, P2//8, True))',
                  'parts.append(kernels.gen_inst(L, f"f{L}_k3", P2//8, P2//8, True, PF=256))')
src = src.replace('parts.append(kernels.gen_inst(13, "f13_x", 169, 169, True))',
                  'parts.append(kernels.gen_inst(13, "f13_x", 169, 169, True, PF=256))')
open('gen.py','w').write(src)
print("gen.py patched")
EOF
python3 gen.py && cd /workdir && rm -f implementation.so && gcc -O3 -march=native -shared -fPIC implementation.c -o implementation.so -lm && python3 dev/check.py && python3 dev/perf.py