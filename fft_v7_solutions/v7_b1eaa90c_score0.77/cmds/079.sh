cd /workdir/dev && python3 - <<'EOF'
src = open('kernels.py').read()
# Rewrite halfmatrix loop: process 4 k's per j-sweep (quad), tail handles remainder (1-3 k's)
old_start = src.index("    kmax_pairs = h-1 if h % 2 == 1 else h")
old_end = src.index('#undef PWFN\n"""\n    return code', old_start)
old_block = src[old_start:old_end]
new_block = '''    nquad = h // 4
    rem = h - nquad*4  # 0..3
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
        decls = []
        accs = []
        sts = []
        for q in range(rem):
            kq = nquad*4 + 1 + q
            decls.append(f"    const double *tc{q} = hmc{N} + {(kq-1)*h} - 1, *ts{q} = hms{N} + {(kq-1)*h} - 1;")
            decls.append(f"    V ter{q} = x0r, tei{q} = x0i, tsr{q} = VC(0.0), tsi{q} = VC(0.0);")
            accs.append(f"      ter{q} += tc{q}[j] * u_r; tei{q} += tc{q}[j] * u_i; tsr{q} += ts{q}[j] * v_r; tsi{q} += ts{q}[j] * v_i;")
            sts.append("    " + stfun(str(kq), f"ter{q} + tsi{q}", f"tei{q} - tsr{q}"))
            sts.append("    " + stfun(str(N-kq), f"ter{q} - tsi{q}", f"tei{q} + tsr{q}"))
        tail = f"""
  {{
{chr(10).join(decls)}
    #pragma GCC unroll 1
    for (long j = 1; j <= {h}; j++) {{
      V u_r = ur[j], u_i = ui[j], v_r = vr[j], v_i = vi[j];
{chr(10).join(accs)}
    }}
{chr(10).join(sts)}
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
  {stfun(0, "accr", "acci")}{quad_body if nquad else ""}{tail}
}}
'''
src = src[:old_start] + new_block + src[old_end:]
# remove the old code tail that built `code` (now replaced); find and strip the old odd_tail construct etc.
# The old block included kmax_pairs..code construction; we cut up to '#undef PWFN...' marker which we must re-add:
idx = src.index("'''\nsrc = src[:old_start]")  # sanity guard against self-reference
EOF
echo "---- that patch approach is too error-prone; rewriting gen_halfmatrix_inst wholesale ----"