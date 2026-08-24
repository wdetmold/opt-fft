cd /workdir/dev && python3 - <<'EOF'
# add fft6 to fft_any in codelets.py
src = open('codelets.py').read()
src = src.replace("""def fft_any(e, x, N):
    if N == 2: return fft2(e, x)""","""def fft_any(e, x, N):
    if N == 6: return fft_pfa(e, x, 2, 3)
    if N == 2: return fft2(e, x)""")
open('codelets.py','w').write(src)
EOF
cat >> kernels.py <<'PYEOF'

def gen_sq_stage2(N, G, fname, P2V, LP1V, OUTS, PF=0):
    """stage2 (fft-G across slabs) + c + pw. in: SRC group (slab stride P2V);
       out: T (slab stride OUTS); c: permuted group (slab stride P2V)."""
    e = E()
    pf = ""
    if PF:
        pf = "".join(f"    __builtin_prefetch((const char*)(xr + {j}*{P2V} + o) + {PF}, 0, 3);\n"
                     f"    __builtin_prefetch((const char*)(xi + {j}*{P2V} + o) + {PF}, 0, 3);\n"
                     f"    __builtin_prefetch((const char*)(cr + {j}*{P2V} + o) + {PF}, 0, 3);\n"
                     f"    __builtin_prefetch((const char*)(ci + {j}*{P2V} + o) + {PF}, 0, 3);\n" for j in range(G))
    xs = [(e.t(f"xr[{j}*{P2V} + o]"), e.t(f"xi[{j}*{P2V} + o]")) for j in range(G)]
    out = fft_any(e, xs, G)
    st = []
    for k2 in range(G):
        fn = "pw_sqrtnr" if k2 % 2 == 0 else "pw_newton"
        st.append(f"    {{ V zr = {out[k2][0]} + cr[{k2}*{P2V} + o]; V zi = {out[k2][1]} + ci[{k2}*{P2V} + o]; "
                  f"V f = {fn}(zr, zi); tr[{k2}*{OUTS} + o] = zr*f; ti[{k2}*{OUTS} + o] = zi*f; }}")
    return f"""static __attribute__((noinline)) void {fname}(const V* xr, const V* xi, const V* cr, const V* ci, V* tr, V* ti) {{
  #pragma GCC unroll 1
  for (long o = 0; o < {LP1V}; o++) {{
{pf}{e.code(indent="    ")}
{chr(10).join(st)}
  }}
}}
"""

def gen_sq_stage1(N, G, fname, INS, P2V, LP1V, PF=0):
    """stage1: fft-G across slabs + twiddle row (per-call pointer). in slab stride INS;
       out: DST slabs at stride G*P2V."""
    e = E()
    pf = ""
    if PF:
        pf = "".join(f"    __builtin_prefetch((const char*)(tr + {j}*{INS} + o) + {PF}, 0, 3);\n"
                     f"    __builtin_prefetch((const char*)(ti + {j}*{INS} + o) + {PF}, 0, 3);\n" for j in range(G))
    xs = [(e.t(f"tr[{j}*{INS} + o]"), e.t(f"ti[{j}*{INS} + o]")) for j in range(G)]
    out = fft_any(e, xs, G)
    st = [f"    dr[0 + o] = {out[0][0]}; di[0 + o] = {out[0][1]};"]
    for k1 in range(1, G):
        st.append(f"    {{ V ar = {out[k1][0]}, ai = {out[k1][1]}; V c_ = VC(twr[{k1}]), s_ = VC(twi[{k1}]); "
                  f"dr[{k1}*{G*P2V} + o] = c_*ar - s_*ai; di[{k1}*{G*P2V} + o] = c_*ai + s_*ar; }}")
    return f"""static __attribute__((noinline)) void {fname}(const V* tr, const V* ti, V* dr, V* di, const double* twr, const double* twi) {{
  #pragma GCC unroll 1
  for (long o = 0; o < {LP1V}; o++) {{
{pf}{e.code(indent="    ")}
{chr(10).join(st)}
  }}
}}
"""
PYEOF
echo kernels extended