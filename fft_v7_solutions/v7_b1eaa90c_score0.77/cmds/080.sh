cd /workdir/dev && python3 - <<'EOF'
src = open('kernels.py').read()
start = src.index("def gen_halfmatrix_inst(")
end = src.index("def gen_inst(")
newfn = '''def gen_halfmatrix_inst(N, fname, IS, OS, pw, PF=0, pwfn='pw_sqrtnr'):
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
        pf_jloop = (f"    __builtin_prefetch((const char*)(xr + j*{IS}) + {PF}, 1, 3);\\n"
                    f"    __builtin_prefetch((const char*)(xi + j*{IS}) + {PF}, 1, 3);\\n"
                    f"    __builtin_prefetch((const char*)(xr + ({N}-j)*{IS}) + {PF}, 1, 3);\\n"
                    f"    __builtin_prefetch((const char*)(xi + ({N}-j)*{IS}) + {PF}, 1, 3);\\n")
        if pw:
            pf_jloop += (f"    __builtin_prefetch((const char*)(cr + j*{OS}) + {PF}, 0, 3);\\n"
                         f"    __builtin_prefetch((const char*)(ci + j*{OS}) + {PF}, 0, 3);\\n"
                         f"    __builtin_prefetch((const char*)(cr + ({N}-j)*{OS}) + {PF}, 0, 3);\\n"
                         f"    __builtin_prefetch((const char*)(ci + ({N}-j)*{OS}) + {PF}, 0, 3);\\n")
    nquad = h // 4
    rem = h - nquad*4
    quad_body = ""
    if nquad:
        quad_body = f"""
  #pragma GCC unroll 1
  for (long k = 1; k + 3 <= {h}; k += 4) {{
    const double *c1 = hmc{N} + (k-1)*{h} - 1, *s1 = hms{N} + (k-1)*{h} - 1;
    const double *c2 = c1 + {h}, *s2 = s1 + {h};
    const double *c3 = c2 + {h}, *s3 = s2 + {h};
    const double *c4 = c3 + {h}, *s4 = s3 + {h};
    V er1 = x0r, ei1 = x0i, sr1 = VC(0.0), si1 = VC(0.0);
    V er2 = x0r, ei2 = x0i, sr2 = VC(0.0), si2 = VC(0.0);
    V er3 = x0r, ei3 = x0i, sr3 = VC(0.0), si3 = VC(0.0);
    V er4 = x0r, ei4 = x0i, sr4 = VC(0.0), si4 = VC(0.0);
    #pragma GCC unroll 1
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
    #pragma GCC unroll 1
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
  #pragma GCC unroll 1
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

'''
src = src[:start] + newfn + src[end:]
open('kernels.py','w').write(src)
print("halfmatrix rewritten")
EOF
python3 -c "
import tune
cfg={'pw':{'13':'alt','17':'alt','23':'alt','45':'alt'}}
so=tune.build(cfg,'kq')
r=tune.bench(so, Ls=(13,17,23), reps=5)
print({L: round(v,2) for L,v in r.items()})
"