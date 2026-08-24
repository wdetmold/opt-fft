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

def gen_composite_inst(N, P, Q, pfa, fname, IS, OS, pw, PF=0, pwfn='pw_sqrtnr', INL=0):
    alt = pwfn == "pw_alt"
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
    pfA = ""
    if PF:
        pfs = []
        for a in range(P):
            if pfa:
                pfs.append(f"    __builtin_prefetch((const char*)(xr + (long)IN[b*{P}+{a}]*{IS}) + {PF}, 1, 3);")
                pfs.append(f"    __builtin_prefetch((const char*)(xi + (long)IN[b*{P}+{a}]*{IS}) + {PF}, 1, 3);")
            else:
                pfs.append(f"    __builtin_prefetch((const char*)(xr + ({Q}*{a})*{IS} + bIS) + {PF}, 1, 3);")
                pfs.append(f"    __builtin_prefetch((const char*)(xi + ({Q}*{a})*{IS} + bIS) + {PF}, 1, 3);")
        pfA = "\n".join(pfs) + "\n"
    passA = pfA + e.code(indent="    ") + "\n" + "\n".join(stA)
    # passB body
    e2 = E()
    ys = [(e2.t(f"S1r[{b}*{P}+k1]"), e2.t(f"S1i[{b}*{P}+k1]")) for b in range(Q)]
    out2 = fft_any(e2, ys, Q)
    stB = []
    for k2 in range(Q):
        ko = f"(long)OUT[k1*{Q}+{k2}]" if pfa else f"({P}*{k2}+k1)"
        if pw:
            fn = ("pw_sqrtnr" if k2 % 2 == 0 else "pw_newton") if alt else "PWFN"
            stB.append(f"    {{ long _k = {ko}; V zr = {out2[k2][0]} + cr[_k*{OS}]; V zi = {out2[k2][1]} + ci[_k*{OS}]; "
                       f"V f = {fn}(zr, zi); yr[_k*{OS}] = zr*f; yi[_k*{OS}] = zi*f; }}")
        else:
            stB.append(f"    yr[{ko}*{OS}] = {out2[k2][0]}; yi[{ko}*{OS}] = {out2[k2][1]};")
    pfB = ""
    if PF and pw:
        pfs = []
        for k2 in range(Q):
            ko = f"(long)OUT[k1*{Q}+{k2}]" if pfa else f"({P}*{k2}+k1)"
            pfs.append(f"    __builtin_prefetch((const char*)(cr + {ko}*{OS}) + {PF}, 0, 3);")
            pfs.append(f"    __builtin_prefetch((const char*)(ci + {ko}*{OS}) + {PF}, 0, 3);")
        pfB = "\n".join(pfs) + "\n"
    passB = pfB + e2.code(indent="    ") + "\n" + "\n".join(stB)
    args = "const V* xr, const V* xi, V* yr, V* yi" + (", const V* cr, const V* ci" if pw else "")
    decl_tabs = ""
    if pfa:
        decl_tabs = (f"  static const uint8_t IN[{Q*P}] = {{{','.join(str(v) for r in inidx for v in r)}}};\n"
                     f"  static const uint8_t OUT[{P*Q}] = {{{','.join(str(v) for r in outidx for v in r)}}};\n")
    bdecl = f"for (long b = 0; b < {Q}; b++)" if pfa else f"for (long b = 0, bIS = 0; b < {Q}; b++, bIS += {IS})"
    pf_jloop = ""
    if PF:
        pf_jloop = (f"    __builtin_prefetch((const char*)(xr + j*{IS}) + {PF}, 1, 3);\n"
                    f"    __builtin_prefetch((const char*)(xi + j*{IS}) + {PF}, 1, 3);\n"
                    f"    __builtin_prefetch((const char*)(xr + ({N}-j)*{IS}) + {PF}, 1, 3);\n"
                    f"    __builtin_prefetch((const char*)(xi + ({N}-j)*{IS}) + {PF}, 1, 3);\n")
        if pw:
            pf_jloop += (f"    __builtin_prefetch((const char*)(cr + j*{OS}) + {PF}, 0, 3);\n"
                         f"    __builtin_prefetch((const char*)(ci + j*{OS}) + {PF}, 0, 3);\n"
                         f"    __builtin_prefetch((const char*)(cr + ({N}-j)*{OS}) + {PF}, 0, 3);\n"
                         f"    __builtin_prefetch((const char*)(ci + ({N}-j)*{OS}) + {PF}, 0, 3);\n")
    ATTR = "noinline" if not INL else "always_inline"
    kw = "static" if not INL else "static inline"
    code = pre + f"""#define PWFN {pwfn if pwfn != "pw_alt" else "pw_sqrtnr"}
{kw} __attribute__(({ATTR})) void {fname}({args}) {{
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
#undef PWFN
"""
    return code

def gen_halfmatrix_inst(N, fname, IS, OS, pw, PF=0, pwfn='pw_sqrtnr', HU=1):
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
    alt = pwfn == "pw_alt"
    _ctr = [0]
    def stfun(k, a, b):
        if pw:
            if alt:
                fn = "pw_sqrtnr" if _ctr[0] % 2 == 0 else "pw_newton"
                _ctr[0] += 1
            else:
                fn = "PWFN"
            return (f"{{ long _k = {k}; V zr = {a} + cr[_k*{OS}]; V zi = {b} + ci[_k*{OS}]; "
                    f"V f = {fn}(zr, zi); yr[_k*{OS}] = zr*f; yi[_k*{OS}] = zi*f; }}")
        return f"{{ long _k = {k}; yr[_k*{OS}] = {a}; yi[_k*{OS}] = {b}; }}"
    pf_jloop = ""
    if PF:
        pf_jloop = (f"    __builtin_prefetch((const char*)(xr + j*{IS}) + {PF}, 1, 3);\n"
                    f"    __builtin_prefetch((const char*)(xi + j*{IS}) + {PF}, 1, 3);\n"
                    f"    __builtin_prefetch((const char*)(xr + ({N}-j)*{IS}) + {PF}, 1, 3);\n"
                    f"    __builtin_prefetch((const char*)(xi + ({N}-j)*{IS}) + {PF}, 1, 3);\n")
        if pw:
            pf_jloop += (f"    __builtin_prefetch((const char*)(cr + j*{OS}) + {PF}, 0, 3);\n"
                         f"    __builtin_prefetch((const char*)(ci + j*{OS}) + {PF}, 0, 3);\n"
                         f"    __builtin_prefetch((const char*)(cr + ({N}-j)*{OS}) + {PF}, 0, 3);\n"
                         f"    __builtin_prefetch((const char*)(ci + ({N}-j)*{OS}) + {PF}, 0, 3);\n")
    nquad = h // 4
    rem = h - nquad*4
    quad_body = ""
    if nquad:
        quad_body = f"""
  #pragma GCC unroll {HU}
  for (long k = 1; k + 3 <= {h}; k += 4) {{
    const double *c1 = hmc{N} + (k-1)*{h} - 1, *s1 = hms{N} + (k-1)*{h} - 1;
    const double *c2 = c1 + {h}, *s2 = s1 + {h};
    const double *c3 = c2 + {h}, *s3 = s2 + {h};
    const double *c4 = c3 + {h}, *s4 = s3 + {h};
    V er1 = x0r, ei1 = x0i, sr1 = VC(0.0), si1 = VC(0.0);
    V er2 = x0r, ei2 = x0i, sr2 = VC(0.0), si2 = VC(0.0);
    V er3 = x0r, ei3 = x0i, sr3 = VC(0.0), si3 = VC(0.0);
    V er4 = x0r, ei4 = x0i, sr4 = VC(0.0), si4 = VC(0.0);
    #pragma GCC unroll {HU}
    for (long j = 1; j <= {h}; j++) {{
      V u_r = ur[j], u_i = ui[j], v_r = vr[j], v_i = vi[j];
      er1 += c1[j] * u_r; ei1 += c1[j] * u_i; sr1 += s1[j] * v_r; si1 += s1[j] * v_i;
      er2 += c2[j] * u_r; ei2 += c2[j] * u_i; sr2 += s2[j] * v_r; si2 += s2[j] * v_i;
      er3 += c3[j] * u_r; ei3 += c3[j] * u_i; sr3 += s3[j] * v_r; si3 += s3[j] * v_i;
      er4 += c4[j] * u_r; ei4 += c4[j] * u_i; sr4 += s4[j] * v_r; si4 += s4[j] * v_i;
    }}
    {stfun("k",   "er1 + si1", "ei1 - sr1")}
    {stfun("(long)"+str(N)+"-k",   "er1 - si1", "ei1 + sr1")}
    {stfun("k+1", "er2 + si2", "ei2 - sr2")}
    {stfun("(long)"+str(N)+"-k-1", "er2 - si2", "ei2 + sr2")}
    {stfun("k+2", "er3 + si3", "ei3 - sr3")}
    {stfun("(long)"+str(N)+"-k-2", "er3 - si3", "ei3 + sr3")}
    {stfun("k+3", "er4 + si4", "ei4 - sr4")}
    {stfun("(long)"+str(N)+"-k-3", "er4 - si4", "ei4 + sr4")}
  }}"""
    tail = ""
    if rem > 0:
        decls = []; accs = []; sts = []
        for q in range(rem):
            kq = nquad*4 + 1 + q
            decls.append(f"    const double *tc{q} = hmc{N} + {(kq-1)*h} - 1, *ts{q} = hms{N} + {(kq-1)*h} - 1;")
            decls.append(f"    V ter{q} = x0r, tei{q} = x0i, tsr{q} = VC(0.0), tsi{q} = VC(0.0);")
            accs.append(f"      ter{q} += tc{q}[j] * u_r; tei{q} += tc{q}[j] * u_i; tsr{q} += ts{q}[j] * v_r; tsi{q} += ts{q}[j] * v_i;")
            sts.append("    " + stfun(str(kq), f"ter{q} + tsi{q}", f"tei{q} - tsr{q}"))
            sts.append("    " + stfun(str(N-kq), f"ter{q} - tsi{q}", f"tei{q} + tsr{q}"))
        nl = chr(10)
        tail = f"""
  {{
{nl.join(decls)}
    #pragma GCC unroll {HU}
    for (long j = 1; j <= {h}; j++) {{
      V u_r = ur[j], u_i = ui[j], v_r = vr[j], v_i = vi[j];
{nl.join(accs)}
    }}
{nl.join(sts)}
  }}"""
    code = pre + f"""#define PWFN {pwfn if pwfn != "pw_alt" else "pw_sqrtnr"}
static __attribute__((noinline)) void {fname}({args}) {{
  V ur[{h+1}], ui[{h+1}], vr[{h+1}], vi[{h+1}];
  V x0r = xr[0], x0i = xi[0];
  V accr = x0r, acci = x0i;
  #pragma GCC unroll {HU}
  for (long j = 1; j <= {h}; j++) {{
{pf_jloop}    V adr = xr[j*{IS}], adi = xi[j*{IS}], bdr = xr[({N}-j)*{IS}], bdi = xi[({N}-j)*{IS}];
    ur[j] = adr + bdr; ui[j] = adi + bdi; vr[j] = adr - bdr; vi[j] = adi - bdi;
    accr += ur[j]; acci += ui[j];
  }}
  {stfun(0, "accr", "acci")}{quad_body}{tail}
}}
#undef PWFN
"""
    return code

def gen_inst(N, fname, IS, OS, pw, PF=0, pwfn='pw_sqrtnr', HU=1, INL=0):
    if N in (13, 17, 23):
        return gen_halfmatrix_inst(N, fname, IS, OS, pw, PF, pwfn, HU)
    if N == 6: return gen_composite_inst(6, 2, 3, True, fname, IS, OS, pw, PF, pwfn, INL)
    if N == 8: return gen_composite_inst(8, 4, 2, False, fname, IS, OS, pw, PF, pwfn, INL)
    if N == 36: return gen_composite_inst(36, 4, 9, True, fname, IS, OS, pw, PF, pwfn, INL)
    if N == 45: return gen_composite_inst(45, 9, 5, True, fname, IS, OS, pw, PF, pwfn, INL)
    if N == 64: return gen_composite_inst(64, 8, 8, False, fname, IS, OS, pw, PF, pwfn, INL)
    raise ValueError(N)

def reset():
    _emitted_tabs.clear()

def gen_sq_stage2(N, G, fname, P2V, LP1V, OUTS, PF=0, PWPAT='sn'):
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
        ch = PWPAT[k2 % len(PWPAT)]
        fn = "pw_sqrtnr" if ch == 's' else "pw_newton"
        st.append(f"    {{ V zr = {out[k2][0]} + cr[{k2}*{P2V} + o]; V zi = {out[k2][1]} + ci[{k2}*{P2V} + o]; "
                  f"V f = {fn}(zr, zi); tr[{k2}*{OUTS} + o] = zr*f; ti[{k2}*{OUTS} + o] = zi*f; }}")
    return f"""static __attribute__((noinline)) void {fname}(const V* xr, const V* xi, const V* cr, const V* ci, V* tr, V* ti, long o0, long o1) {{
  #pragma GCC unroll 1
  for (long o = o0; o < o1; o++) {{
{pf}{e.code(indent="    ")}
{chr(10).join(st)}
  }}
}}
"""

def gen_sq_stage1(N, G, fname, INS, P2V, LP1V, PF=0, NT=0):
    """stage1: fft-G across slabs + twiddle row (per-call pointer). in slab stride INS;
       out: DST slabs at stride G*P2V."""
    e = E()
    pf = ""
    if PF:
        pf = "".join(f"    __builtin_prefetch((const char*)(tr + {j}*{INS} + o) + {PF}, 0, 3);\n"
                     f"    __builtin_prefetch((const char*)(ti + {j}*{INS} + o) + {PF}, 0, 3);\n" for j in range(G))
    xs = [(e.t(f"tr[{j}*{INS} + o]"), e.t(f"ti[{j}*{INS} + o]")) for j in range(G)]
    out = fft_any(e, xs, G)
    if NT:
        st = [f"    _mm512_stream_pd((double*)(dr + o), (__m512d)({out[0][0]})); _mm512_stream_pd((double*)(di + o), (__m512d)({out[0][1]}));"]
        for k1 in range(1, G):
            st.append(f"    {{ V ar = {out[k1][0]}, ai = {out[k1][1]}; V c_ = VC(twr[{k1}]), s_ = VC(twi[{k1}]); "
                      f"_mm512_stream_pd((double*)(dr + {k1}*{G*P2V} + o), (__m512d)(c_*ar - s_*ai)); "
                      f"_mm512_stream_pd((double*)(di + {k1}*{G*P2V} + o), (__m512d)(c_*ai + s_*ar)); }}")
    else:
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

def gen_zline64(fname):
    """in-place 64-point complex FFT on one contiguous line (8 V re + 8 V im),
       vertical fft8 / lane-twiddle / 8x8 transpose / vertical fft8. natural order."""
    from codelets import tw, hexf
    # lane-twiddle tables: T[k1] lane s = W64^{s*k1}, k1=1..7
    rows_r = []; rows_i = []
    for k1 in range(1, 8):
        rr = []; ii = []
        for s in range(8):
            c, s_ = tw(s*k1, 64)
            rr.append(c); ii.append(s_)
        rows_r.append(rr); rows_i.append(ii)
    tabs = ("static const V zl64_twr[7] = {" +
            ",".join("{" + ",".join(hexf(v) for v in r) + "}" for r in rows_r) + "};\n" +
            "static const V zl64_twi[7] = {" +
            ",".join("{" + ",".join(hexf(v) for v in r) + "}" for r in rows_i) + "};\n")
    e = E()
    x = [(e.t(f"dr[{t}]"), e.t(f"di[{t}]")) for t in range(8)]
    A = fft_any(e, x, 8)
    st1 = []
    st1.append(f"    Br[0] = {A[0][0]}; Bi[0] = {A[0][1]};")
    for k1 in range(1, 8):
        rr = e.t(f"{A[k1][0]} * zl64_twr[{k1-1}] - {A[k1][1]} * zl64_twi[{k1-1}]")
        ii = e.t(f"{A[k1][1]} * zl64_twr[{k1-1}] + {A[k1][0]} * zl64_twi[{k1-1}]")
        st1.append(f"    Br[{k1}] = {rr}; Bi[{k1}] = {ii};")
    e2 = E()
    y = [(e2.t(f"Cr[{s}]"), e2.t(f"Ci[{s}]")) for s in range(8)]
    D = fft_any(e2, y, 8)
    st2 = "\n".join(f"    dr[{k2}] = {D[k2][0]}; di[{k2}] = {D[k2][1]};" for k2 in range(8))
    return tabs + f"""static __attribute__((noinline)) void {fname}(V* dr, V* di) {{
  V Br[8], Bi[8], Cr[8], Ci[8];
  {{
{e.code(indent="    ")}
{chr(10).join(st1)}
  }}
  tr8x8(Br, Cr);
  tr8x8(Bi, Ci);
  {{
{e2.code(indent="    ")}
{st2}
  }}
}}
"""

def gen_pfa_stage1_tab(N, G, fname, P2V, LP1V, OUT_STRIDE):
    """PFA stage1: fft-G across slabs; input slab offsets via runtime u8 table (x P2V);
       outputs at dr[k*OUT_STRIDE + o]. No twiddles, no pw."""
    e = E()
    xs = [(e.t(f"tr[ix{j} + o]"), e.t(f"ti[ix{j} + o]")) for j in range(G)]
    out = fft_any(e, xs, G)
    st = "\n".join(f"    dr[{k}*{OUT_STRIDE} + o] = {out[k][0]}; di[{k}*{OUT_STRIDE} + o] = {out[k][1]};"
                   for k in range(G))
    ixdecl = "\n".join(f"  const long ix{j} = (long)idx[{j}]*{P2V};" for j in range(G))
    return f"""static __attribute__((noinline)) void {fname}(const V* tr, const V* ti, V* dr, V* di, const uint8_t* idx) {{
{ixdecl}
  #pragma GCC unroll 1
  for (long o = 0; o < {LP1V}; o++) {{
{e.code(indent="    ")}
{st}
  }}
}}
"""

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
