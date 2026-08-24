cd /workdir/dev && python3 - <<'EOF'
# add gen_sq driver to gen.py and wire scheme 'sq' for 36/64
src = open('gen.py').read()
src = src.replace("_SCHEME_DEFAULT = {6:'soa', 8:'soa', 13:'soa', 17:'slab', 23:'slab', 36:'slab', 45:'slab', 64:'slab'}",
                  "_SCHEME_DEFAULT = {6:'soa', 8:'soa', 13:'soa', 17:'soa', 23:'soa', 36:'sq', 45:'slab', 64:'sq'}")

gensq = '''
def gen_sq(L, G):
    P1 = r8(L)
    P2 = L*P1 + (8 if L == 64 else 0)
    P1V, P2V = P1//8, P2//8
    NB = P1//8
    L3 = L*L*L
    LP1V = L*P1V
    from codelets import tw, hexf
    twr = []; twi = []
    for g in range(G):
        for k1 in range(G):
            c, s = tw(g*k1, L)
            twr.append(c); twi.append(s)
    twtabs = (f"static const double sq{L}_twr[{G*G}] = {{{','.join(hexf(v) for v in twr)}}};\\n"
              f"static const double sq{L}_twi[{G*G}] = {{{','.join(hexf(v) for v in twi)}}};\\n")
    return twtabs + f"""
// ---------------- L={L} : square-CT fused ping-pong (G={G}, P1={P1}, P2={P2}) ----------------
static V sq{L}_ar[{L*P2V}], sq{L}_ai[{L*P2V}];
static V sq{L}_br[{L*P2V}], sq{L}_bi[{L*P2V}];
static V sq{L}_cr[{L*P2V}], sq{L}_ci[{L*P2V}];
static V sq{L}_1r[{L*P2V}], sq{L}_1i[{L*P2V}];
static V sq{L}_tr[{G*P2V}], sq{L}_ti[{G*P2V}];
static V sq{L}_buf_r[{P1}], sq{L}_buf_i[{P1}];

static void sq{L}_yz(V* sr, V* si) {{
  for (long cc = 0; cc < {NB}; cc++) {{
    f{L}_cb(sr + cc, si + cc, sq{L}_buf_r, sq{L}_buf_i);
    for (long rb = 0; rb < {P1V}; rb++) {{
      V tb[8];
      tr8x8(sq{L}_buf_r + rb*8, tb);
      for (int q = 0; q < 8; q++) slabts{L}_r[(cc*8+q)*{P1V} + rb] = tb[q];
      tr8x8(sq{L}_buf_i + rb*8, tb);
      for (int q = 0; q < 8; q++) slabts{L}_i[(cc*8+q)*{P1V} + rb] = tb[q];
    }}
  }}
  for (long cc = 0; cc < {NB}; cc++) {{
    f{L}_cb(slabts{L}_r + cc, slabts{L}_i + cc, sq{L}_buf_r, sq{L}_buf_i);
    int lim = (int)({L} - cc*8); if (lim > 8) lim = 8;
    for (long rb = 0; rb < {P1V}; rb++) {{
      V tb[8];
      tr8x8(sq{L}_buf_r + rb*8, tb);
      for (int q = 0; q < lim; q++) sr[(cc*8+q)*{P1V} + rb] = tb[q];
      tr8x8(sq{L}_buf_i + rb*8, tb);
      for (int q = 0; q < lim; q++) si[(cc*8+q)*{P1V} + rb] = tb[q];
    }}
  }}
}}

void run{L}(const double*restrict x0, const double*restrict c, long B, long m,
            double*restrict out1, double*restrict outm) {{
  for (long j = {L}; j < {P1}; j++) {{ sq{L}_buf_r[j] = VC(0.0); sq{L}_buf_i[j] = VC(0.0); }}
  double *xr = (double*)sq{L}_ar, *xi = (double*)sq{L}_ai;
  double *cr = (double*)sq{L}_cr, *ci = (double*)sq{L}_ci;
  for (long v = 0; v < B; v++) {{
    const double *sx = x0 + 2*v*{L3}, *sc = c + 2*v*{L3};
    for (long x = 0; x < {L}; x++) {{
      long xc = (x % {G})*{G} + (x / {G});   // c permuted: phys G*g+k2 holds logical G*k2+g
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
    // prologue: yz on all slabs (natural), then stage1 cosets (strided) -> B
    for (long s = 0; s < {L}; s++)
      sq{L}_yz(sq{L}_ar + s*{P2V}, sq{L}_ai + s*{P2V});
    for (long s = 0; s < {G}; s++)
      f{L}_s1p(sq{L}_ar + s*{P2V}, sq{L}_ai + s*{P2V}, sq{L}_br + s*{P2V}, sq{L}_bi + s*{P2V},
               sq{L}_twr + s*{G}, sq{L}_twi + s*{G});
    V *srcr = sq{L}_br, *srci = sq{L}_bi, *dstr = sq{L}_ar, *dsti = sq{L}_ai;
    for (long t = 0; t < m-1; t++) {{
      for (long g = 0; g < {G}; g++) {{
        f{L}_s2(srcr + g*{G}*{P2V}, srci + g*{G}*{P2V}, sq{L}_cr + g*{G}*{P2V}, sq{L}_ci + g*{G}*{P2V},
                sq{L}_tr, sq{L}_ti);
        if (t == 0) {{
          for (long k2 = 0; k2 < {G}; k2++) {{
            memcpy(sq{L}_1r + ({G}*k2+g)*{P2V}, sq{L}_tr + k2*{P2V}, {P2V}*sizeof(V));
            memcpy(sq{L}_1i + ({G}*k2+g)*{P2V}, sq{L}_ti + k2*{P2V}, {P2V}*sizeof(V));
          }}
        }}
        for (long k2 = 0; k2 < {G}; k2++)
          sq{L}_yz(sq{L}_tr + k2*{P2V}, sq{L}_ti + k2*{P2V});
        f{L}_s1(sq{L}_tr, sq{L}_ti, dstr + g*{P2V}, dsti + g*{P2V}, sq{L}_twr + g*{G}, sq{L}_twi + g*{G});
      }}
      V *tmp;
      tmp = srcr; srcr = dstr; dstr = tmp;
      tmp = srci; srci = dsti; dsti = tmp;
    }}
    // epilogue: stage2 + pw -> natural into dst
    for (long g = 0; g < {G}; g++)
      f{L}_s2e(srcr + g*{G}*{P2V}, srci + g*{G}*{P2V}, sq{L}_cr + g*{G}*{P2V}, sq{L}_ci + g*{G}*{P2V},
               dstr + g*{P2V}, dsti + g*{P2V});
    slab{L}_extract(m > 1 ? sq{L}_1r : dstr, m > 1 ? sq{L}_1i : dsti, out1 + 2*v*{L3});
    slab{L}_extract(dstr, dsti, outm + 2*v*{L3});
  }}
}}
'''

# insert gen_sq before main()
src = src.replace("def main():", gensq + "\ndef main():")

# wire scheme dispatch in main
src = src.replace("""        else:
            P1 = r8(L); P2 = L*P1 + (8 if L == 64 else 0)
            parts.append(kernels.gen_inst(L, f"f{L}_cb", P1//8, 1, False))
            parts.append(kernels.gen_inst(L, f"f{L}_k3", P2//8, P2//8, True, PF=cfg_pf(L), pwfn="pw_"+cfg_pw(L)))
            parts.append(gen_slab(L))""","""        elif cfg_scheme(L) == 'sq':
            G = {36:6, 64:8}[L]
            P1 = r8(L); P2 = L*P1 + (8 if L == 64 else 0)
            P1V, P2V = P1//8, P2//8
            parts.append(f"static V slabts{L}_r[{P1*P1V}], slabts{L}_i[{P1*P1V}];")
            parts.append(kernels.gen_inst(L, f"f{L}_cb", P1V, 1, False))
            parts.append(kernels.gen_sq_stage2(L, G, f"f{L}_s2", P2V, L*P1V, P2V, PF=cfg_pf(L)))
            parts.append(kernels.gen_sq_stage2(L, G, f"f{L}_s2e", P2V, L*P1V, G*P2V, PF=cfg_pf(L)))
            parts.append(kernels.gen_sq_stage1(L, G, f"f{L}_s1", P2V, P2V, L*P1V, PF=0))
            parts.append(kernels.gen_sq_stage1(L, G, f"f{L}_s1p", G*P2V, P2V, L*P1V, PF=cfg_pf(L)))
            parts.append(gen_sq_extract(L))
            parts.append(gen_sq(L, G))
        else:
            P1 = r8(L); P2 = L*P1 + (8 if L == 64 else 0)
            parts.append(kernels.gen_inst(L, f"f{L}_cb", P1//8, 1, False))
            parts.append(kernels.gen_inst(L, f"f{L}_k3", P2//8, P2//8, True, PF=cfg_pf(L), pwfn="pw_"+cfg_pw(L)))
            parts.append(gen_slab(L))""")

# extractor shared: emit standalone extractor for sq sizes (slab{L}_extract used by gen_sq)
genex = '''
def gen_sq_extract(L):
    P1 = r8(L)
    P2 = L*P1 + (8 if L == 64 else 0)
    L3 = L*L*L
    return f"""
static void slab{L}_extract(const V*restrict ar, const V*restrict ai, double*restrict dst) {{
  const double *r = (const double*)ar, *im = (const double*)ai;
  for (long x = 0; x < {L}; x++)
    for (long y = 0; y < {L}; y++) {{
      const long rowr = x*{P2} + y*{P1};
      double *d = dst + 2*((x*{L} + y)*{L});
      for (long z = 0; z < {L}; z++) {{
        d[2*z]   = r[rowr + z];
        d[2*z+1] = im[rowr + z];
      }}
    }}
}}
"""
'''
src = src.replace("def gen_sq(L, G):", genex + "\ndef gen_sq(L, G):")
open('gen.py','w').write(src)
print("gen.py sq wired")
EOF
GENCFG='{"pw":{"13":"alt","17":"alt","23":"alt","45":"alt"}}' python3 gen.py && gcc -fsyntax-only -O2 -march=native ../implementation.c && echo COMPILES