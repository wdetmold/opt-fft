cd /workdir/dev && cat >> kernels.py <<'PYEOF'

def gen_halfmatrix2_inst(N, fname, IS, OS, pw, pwfn='pw_sqrtnr'):
    """halfmatrix with preloaded ±shared constants (2h registers), straight-line k-pairs."""
    h = (N-1)//2
    import numpy as np
    from codelets import tw, hexf
    # distinct values: C[m] = cos(2pi m/N), S[m] = sin(2pi m/N), m=1..h
    Cv = []; Sv = []
    for m_ in range(1, h+1):
        c, s = tw(m_, N)
        Cv.append(c); Sv.append(-s)
    decls = "\n".join(f"  const V C{m_} = VC({hexf(Cv[m_-1])});\n  const V S{m_} = VC({hexf(Sv[m_-1])});"
                      for m_ in range(1, h+1))
    alt = pwfn == "pw_alt"
    _ctr = [0]
    def stfun(k, a, b):
        if pw:
            if alt:
                fn = "pw_sqrtnr" if _ctr[0] % 2 == 0 else "pw_newton"
                _ctr[0] += 1
            else:
                fn = pwfn
            return (f"{{ long _k = {k}; V zr = {a} + cr[_k*{OS}]; V zi = {b} + ci[_k*{OS}]; "
                    f"V f = {fn}(zr, zi); yr[_k*{OS}] = zr*f; yi[_k*{OS}] = zi*f; }}")
        return f"{{ long _k = {k}; yr[_k*{OS}] = {a}; yi[_k*{OS}] = {b}; }}"
    def cs_term(j, k):
        m_ = (j*k) % N
        if m_ <= h:
            return (f"C{m_}", f"S{m_}", "+")
        else:
            return (f"C{N-m_}", f"S{N-m_}", "-")
    blocks = []
    ks = list(range(1, h+1))
    pairs = [ks[i:i+2] for i in range(0, len(ks), 2)]
    for pair in pairs:
        lines = []
        for idx, k in enumerate(pair):
            lines.append(f"    V er{idx} = x0r, ei{idx} = x0i, sr{idx} = VC(0.0), si{idx} = VC(0.0);")
        for j in range(1, h+1):
            for idx, k in enumerate(pair):
                C, S, sg = cs_term(j, k)
                lines.append(f"    er{idx} = er{idx} + {C}*ur[{j}]; ei{idx} = ei{idx} + {C}*ui[{j}];")
                if sg == "+":
                    lines.append(f"    sr{idx} = sr{idx} + {S}*vr[{j}]; si{idx} = si{idx} + {S}*vi[{j}];")
                else:
                    lines.append(f"    sr{idx} = sr{idx} - {S}*vr[{j}]; si{idx} = si{idx} - {S}*vi[{j}];")
        for idx, k in enumerate(pair):
            lines.append("    " + stfun(str(k), f"er{idx} + si{idx}", f"ei{idx} - sr{idx}"))
            lines.append("    " + stfun(str(N-k), f"er{idx} - si{idx}", f"ei{idx} + sr{idx}"))
        blocks.append("  {\n" + "\n".join(lines) + "\n  }")
    args = "const V* xr, const V* xi, V* yr, V* yi" + (", const V* cr, const V* ci" if pw else "")
    return f"""static __attribute__((noinline)) void {fname}({args}) {{
{decls}
  V ur[{h+1}], ui[{h+1}], vr[{h+1}], vi[{h+1}];
  V x0r = xr[0], x0i = xi[0];
  V accr = x0r, acci = x0i;
  #pragma GCC unroll 1
  for (long j = 1; j <= {h}; j++) {{
    V adr = xr[j*{IS}], adi = xi[j*{IS}], bdr = xr[({N}-j)*{IS}], bdi = xi[({N}-j)*{IS}];
    ur[j] = adr + bdr; ui[j] = adi + bdi; vr[j] = adr - bdr; vi[j] = adi - bdi;
    accr += ur[j]; acci += ui[j];
  }}
  {stfun(0, "accr", "acci")}
{chr(10).join(blocks)}
}}
"""
PYEOF
python3 - <<'EOF'
# wire hm2 via config 'hm2' per size
src = open('gen.py').read()
src = src.replace("def cfg_hu(L): return int(CONFIG.get('hu', {}).get(str(L), 1))",
                  "def cfg_hu(L): return int(CONFIG.get('hu', {}).get(str(L), 1))\ndef cfg_hm2(L): return int(CONFIG.get('hm2', {}).get(str(L), 0))")
for pat, rep in [
  ('parts.append(kernels.gen_inst(L, f"f{L}_z", 1, 1, False, HU=cfg_hu(L)))',
   'parts.append(kernels.gen_halfmatrix2_inst(L, f"f{L}_z", 1, 1, False) if cfg_hm2(L) else kernels.gen_inst(L, f"f{L}_z", 1, 1, False, HU=cfg_hu(L)))'),
  ('parts.append(kernels.gen_inst(L, f"f{L}_y", L, L, False, HU=cfg_hu(L)))',
   'parts.append(kernels.gen_halfmatrix2_inst(L, f"f{L}_y", L, L, False) if cfg_hm2(L) else kernels.gen_inst(L, f"f{L}_y", L, L, False, HU=cfg_hu(L)))'),
  ('parts.append(kernels.gen_inst(L, f"f{L}_x", L*L, L*L, True, PF=cfg_pf(L), pwfn="pw_"+cfg_pw(L), HU=cfg_hu(L)))',
   'parts.append(kernels.gen_halfmatrix2_inst(L, f"f{L}_x", L*L, L*L, True, pwfn="pw_"+cfg_pw(L)) if cfg_hm2(L) else kernels.gen_inst(L, f"f{L}_x", L*L, L*L, True, PF=cfg_pf(L), pwfn="pw_"+cfg_pw(L), HU=cfg_hu(L)))'),
]:
    assert pat in src, pat
    src = src.replace(pat, rep)
open('gen.py','w').write(src)
print("hm2 wired")
EOF
python3 - <<'EOF'
import tune
A = tune.build({'hm2':{}}, 'hmA')
B = tune.build({'hm2':{'13':'1','17':'1','23':'1'}}, 'hmB')
for i in range(3):
    ra = tune.bench(A, Ls=(13,17,23), reps=4)
    rb = tune.bench(B, Ls=(13,17,23), reps=4)
    print("hm1", {L: round(v,2) for L,v in ra.items()}, " hm2", {L: round(v,2) for L,v in rb.items()})
EOF