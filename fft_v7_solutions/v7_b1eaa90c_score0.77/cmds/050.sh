cd /workdir/dev && cat > kernels.py <<'PYEOF'
# Looped, I-cache-compact FFT column kernels with BAKED strides.
import numpy as np
from codelets import E, tw, hexf, fft_any

def u8tab(name, rows):
    flat = []
    for r in rows: flat += list(r)
    return f"static const uint8_t {name}[{len(flat)}] = {{{','.join(str(v) for v in flat)}}};\n"

def dtab(name, vals):
    return f"static const double {name}[{len(vals)}] = {{{','.join(hexf(v) for v in vals)}}};\n"

_emitted_tabs = set()

def gen_composite_inst(N, P, Q, pfa, fname, IS, OS, pw):
    """one kernel instance with baked strides. pw: fused c+pointwise on stores."""
    pre = ""
    if pfa:
        inidx = [[(Q*a + P*b) % N for a in range(P)] for b in range(Q)]
        outidx = [[0]*Q for _ in range(P)]
        for k1 in range(P):
            for k2 in range(Q):
                outidx[k1][k2] = [k for k in range(N) if k % P == k1 and k % Q == k2][0]
    else:
        if (N, 'tw') not in _emitted_tabs:
            twr = []; twi = []
            for b in range(Q):
                for k1 in range(P):
                    c, s = tw(b*k1, N)
                    twr.append(c); twi.append(s)
            pre += dtab(f"twr{N}", twr) + dtab(f"twi{N}", twi)
            _emitted_tabs.add((N, 'tw'))
    # passA body
    e = E()
    xs = []
    for a in range(P):
        if pfa:
            pre_in = f"in{N}_{fname}"
            xs.append((e.t(f"xr[(long)IN[b*{P}+{a}]*{IS}]"), e.t(f"xi[(long)IN[b*{P}+{a}]*{IS}]")))
        else:
            xs.append((e.t(f"xr[({Q}*{a})*{IS} + bIS]"), e.t(f"xi[({Q}*{a})*{IS} + bIS]")))
    out1 = fft_any(e, xs, P)
    stA = []
    for k1 in range(P):
        if pfa:
            stA.append(f"    S1r[b*{P}+{k1}] = {out1[k1][0]}; S1i[b*{P}+{k1}] = {out1[k1][1]};")
        else:
            if k1 == 0:
                stA.append(f"    S1r[b*{P}] = {out1[0][0]}; S1i[b*{P}] = {out1[0][1]};")
            else:
                c = f"twr{N}[b*{P}+{k1}]"; s = f"twi{N}[b*{P}+{k1}]"
                rr = e.t(f"{c} * {out1[k1][0]} - {s} * {out1[k1][1]}")
                ii = e.t(f"{c} * {out1[k1][1]} + {s} * {out1[k1][0]}")
                stA.append(f"    S1r[b*{P}+{k1}] = {rr}; S1i[b*{P}+{k1}] = {ii};")
    passA = e.code(indent="    ") + "\n" + "\n".join(stA)
    # passB body
    e2 = E()
    ys = [(e2.t(f"S1r[{b}*{P}+k1]"), e2.t(f"S1i[{b}*{P}+k1]")) for b in range(Q)]
    out2 = fft_any(e2, ys, Q)
    stB = []
    for k2 in range(Q):
        ko = f"(long)OUT[k1*{Q}+{k2}]" if pfa else f"({P}*{k2}+k1)"
        if pw:
            stB.append(f"    {{ long _k = {ko}; V zr = {out2[k2][0]} + cr[_k*{OS}]; V zi = {out2[k2][1]} + ci[_k*{OS}]; "
                       f"V f = pw_factor(zr, zi); yr[_k*{OS}] = zr*f; yi[_k*{OS}] = zi*f; }}")
        else:
            stB.append(f"    yr[{ko}*{OS}] = {out2[k2][0]}; yi[{ko}*{OS}] = {out2[k2][1]};")
    passB = e2.code(indent="    ") + "\n" + "\n".join(stB)
    args = "const V* xr, const V* xi, V* yr, V* yi" + (", const V* cr, const V* ci" if pw else "")
    decl_tabs = ""
    if pfa:
        decl_tabs = (f"  static const uint8_t IN[{Q*P}] = {{{','.join(str(v) for r in inidx for v in r)}}};\n"
                     f"  static const uint8_t OUT[{P*Q}] = {{{','.join(str(v) for r in outidx for v in r)}}};\n")
    bdecl = f"for (long b = 0; b < {Q}; b++)" if pfa else f"for (long b = 0, bIS = 0; b < {Q}; b++, bIS += {IS})"
    code = pre + f"""static __attribute__((noinline)) void {fname}({args}) {{
{decl_tabs}  V S1r[{N}], S1i[{N}];
  #pragma GCC unroll 1
  {bdecl} {{
{passA}
  }}
  #pragma GCC unroll 1
  for (long k1 = 0; k1 < {P}; k1++) {{
{passB}
  }}
}}
"""
    return code

def gen_halfmatrix_inst(N, fname, IS, OS, pw):
    h = (N-1)//2
    pre = ""
    if (N, 'hm') not in _emitted_tabs:
        cos_t = []; sin_t = []
        for k in range(1, h+1):
            for j in range(1, h+1):
                c, s = tw(j*k, N)
                cos_t.append(c); sin_t.append(-s)
        pre = dtab(f"hmc{N}", cos_t) + dtab(f"hms{N}", sin_t)
        _emitted_tabs.add((N, 'hm'))
    args = "const V* xr, const V* xi, V* yr, V* yi" + (", const V* cr, const V* ci" if pw else "")
    if pw:
        stfun = lambda k, a, b: (f"{{ long _k = {k}; V zr = {a} + cr[_k*{OS}]; V zi = {b} + ci[_k*{OS}]; "
                                 f"V f = pw_factor(zr, zi); yr[_k*{OS}] = zr*f; yi[_k*{OS}] = zi*f; }}")
    else:
        stfun = lambda k, a, b: f"{{ long _k = {k}; yr[_k*{OS}] = {a}; yi[_k*{OS}] = {b}; }}"
    kmax_pairs = h-1 if h % 2 == 1 else h
    odd_tail = ""
    if h % 2 == 1:
        odd_tail = f"""
  {{
    V er = x0r, ei = x0i, sr = VC(0.0), si = VC(0.0);
    #pragma GCC unroll 1
    for (long j = 1; j <= {h}; j++) {{
      er += hmc{N}[{(h-1)*h}+j-1] * ur[j]; ei += hmc{N}[{(h-1)*h}+j-1] * ui[j];
      sr += hms{N}[{(h-1)*h}+j-1] * vr[j]; si += hms{N}[{(h-1)*h}+j-1] * vi[j];
    }}
    {stfun(h, "er + si", "ei - sr")}
    {stfun(N-h, "er - si", "ei + sr")}
  }}"""
    code = pre + f"""static __attribute__((noinline)) void {fname}({args}) {{
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
  #pragma GCC unroll 1
  for (long k = 1; k <= {kmax_pairs}; k += 2) {{
    const double *c1 = hmc{N} + (k-1)*{h} - 1, *s1 = hms{N} + (k-1)*{h} - 1;
    const double *c2 = c1 + {h}, *s2 = s1 + {h};
    V er = x0r, ei = x0i, sr = VC(0.0), si = VC(0.0);
    V er2 = x0r, ei2 = x0i, sr2 = VC(0.0), si2 = VC(0.0);
    #pragma GCC unroll 1
    for (long j = 1; j <= {h}; j++) {{
      V u_r = ur[j], u_i = ui[j], v_r = vr[j], v_i = vi[j];
      er += c1[j] * u_r; ei += c1[j] * u_i;
      sr += s1[j] * v_r; si += s1[j] * v_i;
      er2 += c2[j] * u_r; ei2 += c2[j] * u_i;
      sr2 += s2[j] * v_r; si2 += s2[j] * v_i;
    }}
    {stfun("k", "er + si", "ei - sr")}
    {stfun("(long)"+str(N)+"-k", "er - si", "ei + sr")}
    {stfun("k+1", "er2 + si2", "ei2 - sr2")}
    {stfun("(long)"+str(N)+"-k-1", "er2 - si2", "ei2 + sr2")}
  }}{odd_tail}
}}
"""
    return code

def gen_inst(N, fname, IS, OS, pw):
    if N in (13, 17, 23):
        return gen_halfmatrix_inst(N, fname, IS, OS, pw)
    if N == 36: return gen_composite_inst(36, 4, 9, True, fname, IS, OS, pw)
    if N == 45: return gen_composite_inst(45, 9, 5, True, fname, IS, OS, pw)
    if N == 64: return gen_composite_inst(64, 8, 8, False, fname, IS, OS, pw)
    raise ValueError(N)

def reset():
    _emitted_tabs.clear()
PYEOF
echo ok