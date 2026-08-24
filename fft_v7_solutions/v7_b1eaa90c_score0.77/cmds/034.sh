cd /workdir/dev && cat > kernels.py <<'PYEOF'
# Looped, I-cache-compact FFT column kernels.
import numpy as np
from codelets import E, tw, hexf, fft_any

def small_body(e, x, N):
    return fft_any(e, x, N)

def u8tab(name, rows):
    flat = []
    for r in rows: flat += list(r)
    return f"static const uint8_t {name}[{len(flat)}] = {{{','.join(str(v) for v in flat)}}};\n"

def dtab(name, vals):
    return f"static const double {name}[{len(vals)}] = {{{','.join(hexf(v) for v in vals)}}};\n"

def gen_composite(N, P, Q, pfa):
    """emit fft{N}_cols and fft{N}_colspw (fused +c & pointwise)."""
    pre = ""
    # index tables
    if pfa:
        inidx = [[(Q*a + P*b) % N for a in range(P)] for b in range(Q)]
        outidx = [[0]*Q for _ in range(P)]
        for k1 in range(P):
            for k2 in range(Q):
                outidx[k1][k2] = [k for k in range(N) if k % P == k1 and k % Q == k2][0]
        pre += u8tab(f"in{N}", inidx)
        pre += u8tab(f"out{N}", outidx)
        ld_in = lambda: f"(long)in{N}[b*{P}+a]*is_"
        ld_out = lambda: f"(long)out{N}[k1*{Q}+k2]"
    else:
        ld_in = lambda: None
        ld_out = lambda: None
        # twiddle tables (skip b=0 row which is all ones)
        twr = []; twi = []
        for b in range(Q):
            for k1 in range(P):
                c, s = tw(b*k1, N)
                twr.append(c); twi.append(s)
        pre += dtab(f"twr{N}", twr)
        pre += dtab(f"twi{N}", twi)

    # ---- passA body (loop over b) ----
    e = E()
    xs = []
    for a in range(P):
        if pfa:
            xs.append((e.t(f"xr[(long)in{N}[b*{P}+{a}]*is_]"), e.t(f"xi[(long)in{N}[b*{P}+{a}]*is_]")))
        else:
            xs.append((e.t(f"xr[({Q}*{a}+b)*is_]"), e.t(f"xi[({Q}*{a}+b)*is_]")))
    out1 = small_body(e, xs, P)
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

    # ---- passB body (loop over k1) ----
    e2 = E()
    ys = [(e2.t(f"S1r[{b}*{P}+k1]"), e2.t(f"S1i[{b}*{P}+k1]")) for b in range(Q)]
    out2 = small_body(e2, ys, Q)
    stB = []; stBpw = []
    for k2 in range(Q):
        ko = f"(long)out{N}[k1*{Q}+{k2}]" if pfa else f"({P}*{k2}+k1)"
        stB.append(f"    yr[{ko}*os_] = {out2[k2][0]}; yi[{ko}*os_] = {out2[k2][1]};")
        stBpw.append(f"    STPW({ko}, {out2[k2][0]}, {out2[k2][1]});")
    passB = e2.code(indent="    ") + "\n"
    body = lambda stores: f"""  V S1r[{N}], S1i[{N}];
  #pragma GCC unroll 1
  for (long b = 0; b < {Q}; b++) {{
{passA}
  }}
  #pragma GCC unroll 1
  for (long k1 = 0; k1 < {P}; k1++) {{
{passB}{stores}
  }}"""
    code = pre
    code += f"""static __attribute__((noinline)) void fft{N}_cols(const V* xr, const V* xi, V* yr, V* yi, long is_, long os_) {{
{body(chr(10).join(stB))}
}}
static __attribute__((noinline)) void fft{N}_colspw(const V* xr, const V* xi, V* yr, V* yi, long is_, long os_, const V* cr, const V* ci) {{
{body(chr(10).join(stBpw))}
}}
"""
    return code

def gen_halfmatrix(N):
    h = (N-1)//2
    # tables: cos/sin for k=1..h, j=1..h  (positive-angle cos & sin)
    import math
    cos_t = []; sin_t = []
    for k in range(1, h+1):
        for j in range(1, h+1):
            c, s = tw(j*k, N)
            cos_t.append(c); sin_t.append(-s)     # s_pos = -tw_sin
    pre = dtab(f"hmc{N}", cos_t) + dtab(f"hms{N}", sin_t)
    # k-pair loop bodies
    def kbody(koff):  # koff: "k" or "k+1"
        return f"""      V er{koff if koff!='k' else ''} """  # placeholder
    code = pre
    for pw in (False, True):
        name = f"fft{N}_cols{'pw' if pw else ''}"
        args = "const V* xr, const V* xi, V* yr, V* yi, long is_, long os_" + (", const V* cr, const V* ci" if pw else "")
        st0 = f"STPW(0, X0r, X0i);" if pw else f"yr[0] = X0r; yi[0] = X0i;"
        stk = (f"STPW(k, ar, ai); STPW({N}-k, br, bi);" if pw
               else f"yr[k*os_] = ar; yi[k*os_] = ai; yr[({N}-k)*os_] = br; yi[({N}-k)*os_] = bi;")
        stk2 = (f"STPW(k+1, ar2, ai2); STPW({N}-k-1, br2, bi2);" if pw
                else f"yr[(k+1)*os_] = ar2; yi[(k+1)*os_] = ai2; yr[({N}-k-1)*os_] = br2; yi[({N}-k-1)*os_] = bi2;")
        odd_tail = ""
        kmax_pairs = h-1 if h % 2 == 1 else h  # pairs cover k=1..h ; if h odd, last k handled alone
        if h % 2 == 1:
            odd_tail = f"""
  {{
    const long k = {h};
    V er = x0r, ei = x0i, sr = VC(0.0), si = VC(0.0);
    #pragma GCC unroll 1
    for (long j = 1; j <= {h}; j++) {{
      er += hmc{N}[(k-1)*{h}+j-1] * ur[j]; ei += hmc{N}[(k-1)*{h}+j-1] * ui[j];
      sr += hms{N}[(k-1)*{h}+j-1] * vr[j]; si += hms{N}[(k-1)*{h}+j-1] * vi[j];
    }}
    V ar = er - si, ai = ei + sr, br = er + si, bi = ei - sr;
    {stk}
  }}"""
        code += f"""static __attribute__((noinline)) void {name}({args}) {{
  V ur[{h+1}], ui[{h+1}], vr[{h+1}], vi[{h+1}];
  V x0r = xr[0], x0i = xi[0];
  V accr = x0r, acci = x0i;
  #pragma GCC unroll 1
  for (long j = 1; j <= {h}; j++) {{
    V adr = xr[j*is_], adi = xi[j*is_], bdr = xr[({N}-j)*is_], bdi = xi[({N}-j)*is_];
    ur[j] = adr + bdr; ui[j] = adi + bdi; vr[j] = adr - bdr; vi[j] = adi - bdi;
    accr += ur[j]; acci += ui[j];
  }}
  V X0r = accr, X0i = acci;
  {st0}
  #pragma GCC unroll 1
  for (long k = 1; k <= {kmax_pairs}; k += 2) {{
    V er = x0r, ei = x0i, sr = VC(0.0), si = VC(0.0);
    V er2 = x0r, ei2 = x0i, sr2 = VC(0.0), si2 = VC(0.0);
    #pragma GCC unroll 1
    for (long j = 1; j <= {h}; j++) {{
      V u_r = ur[j], u_i = ui[j], v_r = vr[j], v_i = vi[j];
      er += hmc{N}[(k-1)*{h}+j-1] * u_r; ei += hmc{N}[(k-1)*{h}+j-1] * u_i;
      sr += hms{N}[(k-1)*{h}+j-1] * v_r; si += hms{N}[(k-1)*{h}+j-1] * v_i;
      er2 += hmc{N}[k*{h}+j-1] * u_r; ei2 += hmc{N}[k*{h}+j-1] * u_i;
      sr2 += hms{N}[k*{h}+j-1] * v_r; si2 += hms{N}[k*{h}+j-1] * v_i;
    }}
    V ar = er - si, ai = ei + sr, br = er + si, bi = ei - sr;
    V ar2 = er2 - si2, ai2 = ei2 + sr2, br2 = er2 + si2, bi2 = ei2 - sr2;
    {stk}
    {stk2}
  }}{odd_tail}
}}
"""
    return code

def gen_kernels(N):
    if N in (13, 17, 23):
        return gen_halfmatrix(N)
    if N == 36: return gen_composite(36, 4, 9, True)
    if N == 45: return gen_composite(45, 9, 5, True)
    if N == 64: return gen_composite(64, 8, 8, False)
    raise ValueError(N)
PYEOF
echo done