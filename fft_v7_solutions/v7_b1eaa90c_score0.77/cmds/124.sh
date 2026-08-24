cd /workdir/dev && python3 - <<'EOF'
src = open('gen.py').read()
apfa2 = '''
def gen_apfa2_45():
    L = 45
    P1 = r8(L); P2 = L*P1
    P1V, P2V = P1//8, P2//8
    NB = P1//8
    L3 = L*L*L
    LP1V = L*P1V
    CRT = [ (10*k1 + 36*k2) % 45 for k1 in range(9) for k2 in range(5) ]   # CRT[k1*5+k2]
    TIN5 = [ (4*b) % 5 for b in range(5) ]
    TIN9 = [ (5*a) % 9 for a in range(9) ]
    IDXP = [ [ (5*a + 9*b) % 45 for a in range(9) ] for b in range(5) ]
    crt_tab = ",".join(str(v) for v in CRT)
    tin5 = ",".join(str(v) for v in TIN5)
    tin9 = ",".join(str(v) for v in TIN9)
    idxp_tab = ",".join(",".join(str(v) for v in row) for row in IDXP)
    return f"""
// ---------------- L=45 : alternating-PFA(9,5) single-sweep ping-pong ----------------
static const uint8_t a245_crt[45] = {{{crt_tab}}};
static const uint8_t a245_tin5[5] = {{{tin5}}};
static const uint8_t a245_tin9[9] = {{{tin9}}};
static const uint8_t a245_idxp[45] = {{{idxp_tab}}};
static V a245_sr[{L*P2V}], a245_si[{L*P2V}];   // A9 layout lives here
static V a245_dr[{L*P2V}], a245_di[{L*P2V}];   // natural / A5 layout
static V a245_cer[{L*P2V}], a245_cei[{L*P2V}];
static V a245_cor[{L*P2V}], a245_coi[{L*P2V}];
static V a245_1r[{L*P2V}], a245_1i[{L*P2V}];
static V a245_tr[{9*P2V}], a245_ti[{9*P2V}];
static V a245_buf_r[2*{P1}], a245_buf_i[2*{P1}];
static V a245ts_r[{P1*P1V}], a245ts_i[{P1*P1V}];

static void a245_yz(V* sr, V* si) {{
  for (long cc = 0; cc < {NB}; cc++) {{
    V *br_ = a245_buf_r + (cc&1)*{P1}, *bi_ = a245_buf_i + (cc&1)*{P1};
    f45_cb(sr + cc, si + cc, br_, bi_);
    for (long rb = 0; rb < {P1V}; rb++) {{
      V tb[8];
      tr8x8(br_ + rb*8, tb);
      for (int q = 0; q < 8; q++) a245ts_r[(cc*8+q)*{P1V} + rb] = tb[q];
      tr8x8(bi_ + rb*8, tb);
      for (int q = 0; q < 8; q++) a245ts_i[(cc*8+q)*{P1V} + rb] = tb[q];
    }}
  }}
  for (long cc = 0; cc < {NB}; cc++) {{
    V *br_ = a245_buf_r + (cc&1)*{P1}, *bi_ = a245_buf_i + (cc&1)*{P1};
    f45_cb(a245ts_r + cc, a245ts_i + cc, br_, bi_);
    int lim = (int)({L} - cc*8); if (lim > 8) lim = 8;
    for (long rb = 0; rb < {P1V}; rb++) {{
      V tb[8];
      tr8x8(br_ + rb*8, tb);
      for (int q = 0; q < lim; q++) sr[(cc*8+q)*{P1V} + rb] = tb[q];
      tr8x8(bi_ + rb*8, tb);
      for (int q = 0; q < lim; q++) si[(cc*8+q)*{P1V} + rb] = tb[q];
    }}
  }}
}}

void run45(const double*restrict x0, const double*restrict c, long B, long m,
           double*restrict out1, double*restrict outm) {{
  for (long j = {L}; j < {P1}; j++) {{ a245_buf_r[j] = VC(0.0); a245_buf_i[j] = VC(0.0);
    a245_buf_r[{P1}+j] = VC(0.0); a245_buf_i[{P1}+j] = VC(0.0); }}
  double *xr = (double*)a245_dr, *xi = (double*)a245_di;
  double *cer = (double*)a245_cer, *cei = (double*)a245_cei;
  double *cor = (double*)a245_cor, *coi = (double*)a245_coi;
  for (long v = 0; v < B; v++) {{
    const double *sx = x0 + 2*v*{L3}, *sc = c + 2*v*{L3};
    for (long x = 0; x < {L}; x++) {{
      long k1 = x % 9, k2 = x % 5;
      long xce = 5*k1 + k2;    // even-sweep c slot for logical slab x
      long xco = 9*k2 + k1;    // odd-sweep  c slot for logical slab x
      for (long y = 0; y < {L}; y++) {{
        long row = x*{P2} + y*{P1};
        long rowe = xce*{P2} + y*{P1}, rowo = xco*{P2} + y*{P1};
        const double *px = sx + 2*((x*{L}+y)*{L});
        const double *pc = sc + 2*((x*{L}+y)*{L});
        for (long z = 0; z < {L}; z++) {{
          xr[row+z] = px[2*z]; xi[row+z] = px[2*z+1];
          cer[rowe+z] = pc[2*z]; cei[rowe+z] = pc[2*z+1];
          cor[rowo+z] = pc[2*z]; coi[rowo+z] = pc[2*z+1];
        }}
        for (long z = {L}; z < {P1}; z++) {{
          xr[row+z] = 0.0; xi[row+z] = 0.0;
          cer[rowe+z] = 1.0; cei[rowe+z] = 0.0;
          cor[rowo+z] = 1.0; coi[rowo+z] = 0.0;
        }}
      }}
    }}
    // prologue
    for (long s = 0; s < {L}; s++)
      a245_yz(a245_dr + s*{P2V}, a245_di + s*{P2V});
    for (long b = 0; b < 5; b++)
      f45o_s1(a245_dr, a245_di, a245_sr + b*{P2V}, a245_si + b*{P2V}, a245_idxp + 9*b);
    for (long t = 0; t < m-1; t++) {{
      if ((t & 1) == 0) {{
        for (long k1 = 0; k1 < 9; k1++) {{
          f45e_s2(a245_sr + 5*k1*{P2V}, a245_si + 5*k1*{P2V}, a245_cer + 5*k1*{P2V}, a245_cei + 5*k1*{P2V},
                  a245_tr, a245_ti, 0, {LP1V});
          if (t == 0) {{
            for (long k2 = 0; k2 < 5; k2++) {{
              long dst = (long)a245_crt[k1*5+k2];
              memcpy(a245_1r + dst*{P2V}, a245_tr + k2*{P2V}, {P2V}*sizeof(V));
              memcpy(a245_1i + dst*{P2V}, a245_ti + k2*{P2V}, {P2V}*sizeof(V));
            }}
          }}
          for (long k2 = 0; k2 < 5; k2++)
            a245_yz(a245_tr + k2*{P2V}, a245_ti + k2*{P2V});
          f45e_s1(a245_tr, a245_ti, a245_dr + ((2*k1) % 9)*{P2V}, a245_di + ((2*k1) % 9)*{P2V}, a245_tin5);
        }}
      }} else {{
        for (long k1p = 0; k1p < 5; k1p++) {{
          f45o_s2(a245_dr + 9*k1p*{P2V}, a245_di + 9*k1p*{P2V}, a245_cor + 9*k1p*{P2V}, a245_coi + 9*k1p*{P2V},
                  a245_tr, a245_ti, 0, {LP1V});
          for (long kq = 0; kq < 9; kq++)
            a245_yz(a245_tr + kq*{P2V}, a245_ti + kq*{P2V});
          f45o_s1(a245_tr, a245_ti, a245_sr + ((4*k1p) % 5)*{P2V}, a245_si + ((4*k1p) % 5)*{P2V}, a245_tin9);
        }}
      }}
    }}
    // epilogue for iteration m-1
    V *fr, *fi;
    if (((m-1) & 1) == 0) {{
      fr = a245_dr; fi = a245_di;
      for (long k1 = 0; k1 < 9; k1++) {{
        f45e_s2(a245_sr + 5*k1*{P2V}, a245_si + 5*k1*{P2V}, a245_cer + 5*k1*{P2V}, a245_cei + 5*k1*{P2V},
                a245_tr, a245_ti, 0, {LP1V});
        for (long k2 = 0; k2 < 5; k2++) {{
          long dst = (long)a245_crt[k1*5+k2];
          memcpy(fr + dst*{P2V}, a245_tr + k2*{P2V}, {P2V}*sizeof(V));
          memcpy(fi + dst*{P2V}, a245_ti + k2*{P2V}, {P2V}*sizeof(V));
        }}
      }}
    }} else {{
      fr = a245_sr; fi = a245_si;
      for (long k1p = 0; k1p < 5; k1p++) {{
        f45o_s2(a245_dr + 9*k1p*{P2V}, a245_di + 9*k1p*{P2V}, a245_cor + 9*k1p*{P2V}, a245_coi + 9*k1p*{P2V},
                a245_tr, a245_ti, 0, {LP1V});
        for (long kq = 0; kq < 9; kq++) {{
          long dst = (long)a245_crt[kq*5+k1p];
          memcpy(fr + dst*{P2V}, a245_tr + kq*{P2V}, {P2V}*sizeof(V));
          memcpy(fi + dst*{P2V}, a245_ti + kq*{P2V}, {P2V}*sizeof(V));
        }}
      }}
    }}
    slab45x_extract(m > 1 ? a245_1r : fr, m > 1 ? a245_1i : fi, out1 + 2*v*{L3});
    slab45x_extract(fr, fi, outm + 2*v*{L3});
  }}
}}
"""
'''
src = src.replace("def gen_apfa45():", apfa2 + "\ndef gen_apfa45():")
src = src.replace("""        elif cfg_scheme(L) == 'apfa':""","""        elif cfg_scheme(L) == 'apfa2':
            assert L == 45
            P1 = r8(45); P2 = 45*P1
            P1V, P2V = P1//8, P2//8
            parts.append(kernels.gen_inst(45, "f45_cb", P1V, 1, False))
            parts.append(kernels.gen_sq_stage2(45, 5, "f45e_s2", P2V, 45*P1V, P2V, PF=0, PWPAT=CONFIG.get("sqpw",{}).get("45","sn")))
            parts.append(kernels.gen_sq_stage2(45, 9, "f45o_s2", P2V, 45*P1V, P2V, PF=0, PWPAT=CONFIG.get("sqpw",{}).get("45","sn")))
            parts.append(kernels.gen_pfa_stage1_tab(45, 5, "f45e_s1", P2V, 45*P1V, 9*P2V))
            parts.append(kernels.gen_pfa_stage1_tab(45, 9, "f45o_s1", P2V, 45*P1V, 5*P2V))
            parts.append(gen_sq_extract(45).replace("slab45_extract", "slab45x_extract"))
            parts.append(gen_apfa2_45())
        elif cfg_scheme(L) == 'apfa':""")
src = src.replace("_SCHEME_DEFAULT = {6:'soa', 8:'soa', 13:'soa', 17:'soa', 23:'soa', 36:'sq', 45:'slab', 64:'sq'}",
                  "_SCHEME_DEFAULT = {6:'soa', 8:'soa', 13:'soa', 17:'soa', 23:'soa', 36:'sq', 45:'apfa2', 64:'sq'}")
open('gen.py','w').write(src)
print("apfa2 wired")
EOF
python3 gen.py && gcc -fsyntax-only -O3 -march=native ../implementation.c && echo COMPILES && cd /workdir && rm -f implementation.so && gcc -O3 -march=native -shared -fPIC implementation.c -o implementation.so -lm && python3 dev/check.py 2>&1 | grep -E "L=45|ALL|FAIL"