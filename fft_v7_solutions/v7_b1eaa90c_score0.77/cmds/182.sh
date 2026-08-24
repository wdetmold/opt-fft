cd /workdir/dev && cat >> kernels.py <<'PYEOF'

def gen_sq_stage2_fz64(fname, P2V, OUTS, PWPAT='sn'):
    """64-only: stage2 + c + pw, with z-line FFT fused onto completed T rows."""
    G = 8
    e = E()
    xs = [(e.t(f"xr[{j}*{P2V} + o]"), e.t(f"xi[{j}*{P2V} + o]")) for j in range(G)]
    out = fft_any(e, xs, G)
    st = []
    for k2 in range(G):
        ch = PWPAT[k2 % len(PWPAT)]
        fn = "pw_sqrtnr" if ch == 's' else "pw_newton"
        st.append(f"      {{ V zr = {out[k2][0]} + cr[{k2}*{P2V} + o]; V zi = {out[k2][1]} + ci[{k2}*{P2V} + o]; "
                  f"V f = {fn}(zr, zi); tr[{k2}*{OUTS} + o] = zr*f; ti[{k2}*{OUTS} + o] = zi*f; }}")
    zl = "\n".join(f"    f64_zline(tr + {k2}*{OUTS} + y*8, ti + {k2}*{OUTS} + y*8);" for k2 in range(G))
    return f"""static __attribute__((noinline)) void {fname}(const V* xr, const V* xi, const V* cr, const V* ci, V* tr, V* ti) {{
  #pragma GCC unroll 1
  for (long y = 0; y < 64; y++) {{
    #pragma GCC unroll 1
    for (long q = 0; q < 8; q++) {{
      long o = y*8 + q;
{e.code(indent="      ")}
{chr(10).join(st)}
    }}
{zl}
  }}
}}
"""
PYEOF
python3 - <<'EOF'
src = open('gen.py').read()
# emit the fused kernel for 64 and use it in sweeps t>=1
src = src.replace('''            if L == 64:
                parts.append(kernels.gen_zline64("f64_zline"))
                parts.append(kernels.gen_inst(64, "f64_ycol", P1V, P1V, False))''',
'''            if L == 64:
                parts.append(kernels.gen_zline64("f64_zline"))
                parts.append(kernels.gen_inst(64, "f64_ycol", P1V, P1V, False))
                parts.append(kernels.gen_sq_stage2_fz64("f64_s2z", P2V, P2V, PWPAT=CONFIG.get("sqpw",{}).get("64","sn")))''')
# driver: special-case 64 sweeps
old = """        {f'const V *pwcr = ((t+1) & 1) ? sq{L}_c1r : sq{L}_cr;' if PAR else f'const V *pwcr = sq{L}_cr;'}
        {f'const V *pwci = ((t+1) & 1) ? sq{L}_c1i : sq{L}_ci;' if PAR else f'const V *pwci = sq{L}_ci;'}
        f{L}_s2(srcr + g*{G}*{P2V}, srci + g*{G}*{P2V}, pwcr + g*{G}*{P2V}, pwci + g*{G}*{P2V},
                TAr, TAi, 0, {LP1V});
        if (t == 0) {{
          for (long k2 = 0; k2 < {G}; k2++) {{
            memcpy(sq{L}_1r + ({G}*k2+g)*{P2V}, TAr + k2*{P2V}, {P2V}*sizeof(V));
            memcpy(sq{L}_1i + ({G}*k2+g)*{P2V}, TAi + k2*{P2V}, {P2V}*sizeof(V));
          }}
        }}
        for (long k2 = 0; k2 < {G}; k2++)
          sq{L}_yz(TAr + k2*{P2V}, TAi + k2*{P2V});
        f{L}_s1(TAr, TAi, dstr + g*{P2V}, dsti + g*{P2V}, sq{L}_twr + g*{G}, sq{L}_twi + g*{G});
      }}"""
new = """        {f'const V *pwcr = ((t+1) & 1) ? sq{L}_c1r : sq{L}_cr;' if PAR else f'const V *pwcr = sq{L}_cr;'}
        {f'const V *pwci = ((t+1) & 1) ? sq{L}_c1i : sq{L}_ci;' if PAR else f'const V *pwci = sq{L}_ci;'}
        {f'''if (t > 0) {{{{
          f64_s2z(srcr + g*{G}*{P2V}, srci + g*{G}*{P2V}, pwcr + g*{G}*{P2V}, pwci + g*{G}*{P2V}, TAr, TAi);
          for (long k2 = 0; k2 < {G}; k2++)
            for (long cc = 0; cc < {P1V}; cc++)
              f64_ycol(TAr + k2*{P2V} + cc, TAi + k2*{P2V} + cc, TAr + k2*{P2V} + cc, TAi + k2*{P2V} + cc);
        }}}} else {{{{
          f{L}_s2(srcr + g*{G}*{P2V}, srci + g*{G}*{P2V}, pwcr + g*{G}*{P2V}, pwci + g*{G}*{P2V},
                  TAr, TAi, 0, {LP1V});
          for (long k2 = 0; k2 < {G}; k2++) {{{{
            memcpy(sq{L}_1r + ({G}*k2+g)*{P2V}, TAr + k2*{P2V}, {P2V}*sizeof(V));
            memcpy(sq{L}_1i + ({G}*k2+g)*{P2V}, TAi + k2*{P2V}, {P2V}*sizeof(V));
          }}}}
          for (long k2 = 0; k2 < {G}; k2++)
            sq{L}_yz(TAr + k2*{P2V}, TAi + k2*{P2V});
        }}}}''' if L == 64 else f'''f{L}_s2(srcr + g*{G}*{P2V}, srci + g*{G}*{P2V}, pwcr + g*{G}*{P2V}, pwci + g*{G}*{P2V},
                TAr, TAi, 0, {LP1V});
        if (t == 0) {{{{
          for (long k2 = 0; k2 < {G}; k2++) {{{{
            memcpy(sq{L}_1r + ({G}*k2+g)*{P2V}, TAr + k2*{P2V}, {P2V}*sizeof(V));
            memcpy(sq{L}_1i + ({G}*k2+g)*{P2V}, TAi + k2*{P2V}, {P2V}*sizeof(V));
          }}}}
        }}}}
        for (long k2 = 0; k2 < {G}; k2++)
          sq{L}_yz(TAr + k2*{P2V}, TAi + k2*{P2V});'''}
        f{L}_s1(TAr, TAi, dstr + g*{P2V}, dsti + g*{P2V}, sq{L}_twr + g*{G}, sq{L}_twi + g*{G});
      }}"""
assert old in src
src = src.replace(old, new)
open('gen.py','w').write(src)
print("s2z wired")
EOF
python3 gen.py && gcc -O3 -march=native -shared -fPIC ../implementation.c -o /workdir/implementation.so -lm && python3 /workdir/dev/check.py | grep -E "L=64|ALL"