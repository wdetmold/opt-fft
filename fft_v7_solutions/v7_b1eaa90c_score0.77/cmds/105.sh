cd /workdir/dev && cat >> kernels.py <<'PYEOF'

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
PYEOF
python3 - <<'EOF'
# add 'apfa' scheme for 45 in gen.py
src = open('gen.py').read()
src = src.replace("_SCHEME_DEFAULT = {6:'soa', 8:'soa', 13:'soa', 17:'soa', 23:'soa', 36:'sq', 45:'slab', 64:'sq'}",
                  "_SCHEME_DEFAULT = {6:'soa', 8:'soa', 13:'soa', 17:'soa', 23:'soa', 36:'sq', 45:'apfa', 64:'sq'}")

apfa = '''
def gen_apfa45():
    L, P, Q = 45, 9, 5          # stage1 = 9pt over a ; stage2 = 5pt over b
    P1 = r8(L); P2 = L*P1
    P1V, P2V = P1//8, P2//8
    NB = P1//8
    L3 = L*L*L
    LP1V = L*P1V
    CRT = [ (10*k1 + 36*k2) % 45 for k1 in range(9) for k2 in range(5) ]   # [k1*5+k2]
    IDXS = [ (5*a) % 9 for a in range(9) ]
    IDXP = [ [ (5*a + 9*b) % 45 for a in range(9) ] for b in range(5) ]
    crt_tab = ",".join(str(v) for v in CRT)
    idxs_tab = ",".join(str(v) for v in IDXS)
    idxp_tab = ",".join(",".join(str(v) for v in row) for row in IDXP)
    return f"""
// ---------------- L=45 : asymmetric PFA(9,5) ping-pong (P1={P1}, P2={P2}) ----------------
static const uint8_t ap45_crt[45] = {{{crt_tab}}};
static const uint8_t ap45_idxs[9] = {{{idxs_tab}}};
static const uint8_t ap45_idxp[45] = {{{idxp_tab}}};
static V ap45_sr[{L*P2V}], ap45_si[{L*P2V}];
static V ap45_dr[{L*P2V}], ap45_di[{L*P2V}];
static V ap45_cr[{L*P2V}], ap45_ci[{L*P2V}];
static V ap45_1r[{L*P2V}], ap45_1i[{L*P2V}];
static V ap45_tr[{5*P2V}], ap45_ti[{5*P2V}];
static V ap45_buf_r[2*{P1}], ap45_buf_i[2*{P1}];
static V ap45ts_r[{P1*P1V}], ap45ts_i[{P1*P1V}];

static void ap45_yz(V* sr, V* si) {{
  for (long cc = 0; cc < {NB}; cc++) {{
    V *br_ = ap45_buf_r + (cc&1)*{P1}, *bi_ = ap45_buf_i + (cc&1)*{P1};
    f45_cb(sr + cc, si + cc, br_, bi_);
    for (long rb = 0; rb < {P1V}; rb++) {{
      V tb[8];
      tr8x8(br_ + rb*8, tb);
      for (int q = 0; q < 8; q++) ap45ts_r[(cc*8+q)*{P1V} + rb] = tb[q];
      tr8x8(bi_ + rb*8, tb);
      for (int q = 0; q < 8; q++) ap45ts_i[(cc*8+q)*{P1V} + rb] = tb[q];
    }}
  }}
  for (long cc = 0; cc < {NB}; cc++) {{
    V *br_ = ap45_buf_r + (cc&1)*{P1}, *bi_ = ap45_buf_i + (cc&1)*{P1};
    f45_cb(ap45ts_r + cc, ap45ts_i + cc, br_, bi_);
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
  for (long j = {L}; j < {P1}; j++) {{ ap45_buf_r[j] = VC(0.0); ap45_buf_i[j] = VC(0.0);
    ap45_buf_r[{P1}+j] = VC(0.0); ap45_buf_i[{P1}+j] = VC(0.0); }}
  double *xr = (double*)ap45_dr, *xi = (double*)ap45_di;
  double *cr = (double*)ap45_cr, *ci = (double*)ap45_ci;
  for (long v = 0; v < B; v++) {{
    const double *sx = x0 + 2*v*{L3}, *sc = c + 2*v*{L3};
    for (long x = 0; x < {L}; x++) {{
      long k1 = x % 9, k2 = x % 5;
      long xc = 5*k1 + k2;             // c slot for logical slab x
      for (long y = 0; y < {L}; y++) {{
        long row = x*{P2} + y*{P1};
        long rowc = xc*{P2} + y*{P1};
        const double *px = sx + 2*((x*{L}+y)*{L});
        const double *pc = sc + 2*((x*{L}+y)*{L});
        for (long z = 0; z < {L}; z++) {{
          xr[row+z] = px[2*z]; xi[row+z] = px[2*z+1];
          cr[rowc+z] = pc[2*z]; ci[rowc+z] = pc[2*z+1];
        }}
        for (long z = {L}; z < {P1}; z++) {{
          xr[row+z] = 0.0; xi[row+z] = 0.0;
          cr[rowc+z] = 1.0; ci[rowc+z] = 0.0;
        }}
      }}
    }}
    // prologue: yz all natural slabs in D, stage1 cosets D -> S
    for (long s = 0; s < {L}; s++)
      ap45_yz(ap45_dr + s*{P2V}, ap45_di + s*{P2V});
    for (long b = 0; b < 5; b++)
      f45_s1(ap45_dr, ap45_di, ap45_sr + b*{P2V}, ap45_si + b*{P2V}, ap45_idxp + 9*b);
    for (long t = 0; t < m-1; t++) {{
      for (long k1 = 0; k1 < 9; k1++) {{
        f45_s2d(ap45_sr + 5*k1*{P2V}, ap45_si + 5*k1*{P2V}, ap45_cr + 5*k1*{P2V}, ap45_ci + 5*k1*{P2V},
                ap45_dr + k1*{P2V}, ap45_di + k1*{P2V});
        if (t == 0) {{
          for (long k2 = 0; k2 < 5; k2++) {{
            long dst = (long)ap45_crt[k1*5+k2];
            memcpy(ap45_1r + dst*{P2V}, ap45_dr + (9*k2+k1)*{P2V}, {P2V}*sizeof(V));
            memcpy(ap45_1i + dst*{P2V}, ap45_di + (9*k2+k1)*{P2V}, {P2V}*sizeof(V));
          }}
        }}
        for (long k2 = 0; k2 < 5; k2++)
          ap45_yz(ap45_dr + (9*k2+k1)*{P2V}, ap45_di + (9*k2+k1)*{P2V});
      }}
      for (long b = 0; b < 5; b++)
        f45_s1(ap45_dr + 9*((4*b) % 5)*{P2V}, ap45_di + 9*((4*b) % 5)*{P2V},
               ap45_sr + b*{P2V}, ap45_si + b*{P2V}, ap45_idxs);
    }}
    // epilogue: stage2+pw -> T, scatter to natural D
    for (long k1 = 0; k1 < 9; k1++) {{
      f45_s2t(ap45_sr + 5*k1*{P2V}, ap45_si + 5*k1*{P2V}, ap45_cr + 5*k1*{P2V}, ap45_ci + 5*k1*{P2V},
              ap45_tr, ap45_ti);
      for (long k2 = 0; k2 < 5; k2++) {{
        long dst = (long)ap45_crt[k1*5+k2];
        memcpy(ap45_dr + dst*{P2V}, ap45_tr + k2*{P2V}, {P2V}*sizeof(V));
        memcpy(ap45_di + dst*{P2V}, ap45_ti + k2*{P2V}, {P2V}*sizeof(V));
      }}
    }}
    slab45x_extract(m > 1 ? ap45_1r : ap45_dr, m > 1 ? ap45_1i : ap45_di, out1 + 2*v*{L3});
    slab45x_extract(ap45_dr, ap45_di, outm + 2*v*{L3});
  }}
}}
"""
'''
src = src.replace("def gen_sq_extract(L):", apfa + "\ndef gen_sq_extract(L):")
# dispatch
src = src.replace("""        else:
            P1 = r8(L); P2 = L*P1 + (8 if L == 64 else 0)
            parts.append(kernels.gen_inst(L, f"f{L}_cb", P1//8, 1, False, HU=cfg_hu(L)))
            parts.append(kernels.gen_inst(L, f"f{L}_k3", P2//8, P2//8, True, PF=cfg_pf(L), pwfn="pw_"+cfg_pw(L), HU=cfg_hu(L)))
            parts.append(gen_slab(L))""","""        elif cfg_scheme(L) == 'apfa':
            assert L == 45
            P1 = r8(45); P2 = 45*P1
            P1V, P2V = P1//8, P2//8
            parts.append(kernels.gen_inst(45, "f45_cb", P1V, 1, False))
            parts.append(kernels.gen_sq_stage2(45, 5, "f45_s2d", P2V, 45*P1V, 9*P2V, PF=cfg_pf(45)))
            parts.append(kernels.gen_sq_stage2(45, 5, "f45_s2t", P2V, 45*P1V, P2V, PF=0))
            parts.append(kernels.gen_pfa_stage1_tab(45, 9, "f45_s1", P2V, 45*P1V, 5*P2V))
            parts.append(gen_sq_extract(45).replace("slab45_extract", "slab45x_extract"))
            parts.append(gen_apfa45())
        else:
            P1 = r8(L); P2 = L*P1 + (8 if L == 64 else 0)
            parts.append(kernels.gen_inst(L, f"f{L}_cb", P1//8, 1, False, HU=cfg_hu(L)))
            parts.append(kernels.gen_inst(L, f"f{L}_k3", P2//8, P2//8, True, PF=cfg_pf(L), pwfn="pw_"+cfg_pw(L), HU=cfg_hu(L)))
            parts.append(gen_slab(L))""")
open('gen.py','w').write(src)
print("apfa45 wired")
EOF
python3 gen.py && gcc -fsyntax-only -O3 -march=native ../implementation.c && echo COMPILES