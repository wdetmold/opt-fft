"""Assembly of implementation.c (v2): rader gen, driver, init, dispatch."""
import mpmath as mp
from gencore import G, wcs
from dftgen import best_plan, primroot
from gen2 import (SIZES, RPAD, cf, emit_block, build_map, make_leaf, LEAFS, CONSTS,
                  gen_line_fn, gen_ct_fn, gen_pfa_fn, gen_dsym_fn, fn_head, itab)

mp.mp.dps = 60

# ---------------- rader staged (L=17: q=16 via ct(4,4)) ----------------
def gen_rader_fn(L, kind, fname):
    q = L - 1
    gr = primroot(L)
    gpow = [pow(gr, l, L) for l in range(q)]
    ginv = [pow(gr, (q - l) % q, L) for l in range(q)]
    # B'_t high precision
    Bc = []
    for t in range(q):
        acc = mp.mpc(0)
        for l in range(q):
            a1 = -2 * mp.pi * mp.mpf(ginv[l]) / L
            a2 = -2 * mp.pi * mp.mpf((t * l) % q) / q
            acc += (mp.cos(a1) + 1j * mp.sin(a1)) * (mp.cos(a2) + 1j * mp.sin(a2))
        acc /= q
        Bc.append((float(mp.re(acc)), float(mp.im(acc))))
    make_leaf(q, f"dft{q}_nt")
    make_leaf(q, f"dft{q}_it", in_tab=True)
    it = itab(f"RAD_II_{L}", [gpow[l] for l in range(q)])

    def dft16(body, sre_, sim_, sstr, via_tab, dst_r, dst_i):
        if via_tab:
            body.append(f"  dft{q}_it({sre_}, {sim_}, {sstr}, {dst_r}, {dst_i}, 8, {it});")
        else:
            body.append(f"  dft{q}_nt({sre_}, {sim_}, {sstr}, {dst_r}, {dst_i}, 8);")

    def half(body, sre_, sim_, sstr, via_tab, out_mode):
        """one rader transform: src -> (out_mode: 'data_plain'|'data_map'|'scratch_map')"""
        dft16(body, sre_, sim_, sstr, via_tab, "SA_R", "SA_I")   # A
        # pointwise P_t = B'_t * A_t -> SP
        g = G()
        st = []
        for t in range(q):
            A = (g.inp(f"VL(SA_R + {t}*8)"), g.inp(f"VL(SA_I + {t}*8)"))
            P = g.cmulk(Bc[t][0], Bc[t][1], A)
            st += [(f"SP_R + {t}*8", P[0]), (f"SP_I + {t}*8", P[1])]
        lines, rnd = emit_block(g, st, prefix="pw")
        body.append("  {")
        body += ["  " + l for l in lines]
        for p, nd in st: body.append(f"    VS({p}, {rnd(nd)});")
        body.append("  }")
        # D = dft16(SP) (affine) -> SD
        dft16(body, "SP_R", "SP_I", "8", False, "SD_R", "SD_I")
        # outputs: X[0] = x0 + A0 ; X[ginv[m]] = x0 + D[(q-m)%q]
        g = G()
        x0 = (g.inp(f"VL({sre_} + 0)"), g.inp(f"VL({sim_} + 0)"))
        outs = [(0, (g.add(x0[0], g.inp("VL(SA_R + 0)")), g.add(x0[1], g.inp("VL(SA_I + 0)"))))]
        for m_ in range(q):
            t = (q - m_) % q
            D = (g.inp(f"VL(SD_R + {t}*8)"), g.inp(f"VL(SD_I + {t}*8)"))
            outs.append((ginv[m_], (g.add(x0[0], D[0]), g.add(x0[1], D[1]))))
        st = []; snapst = []
        for idx, (zr0, zi0) in outs:
            if out_mode == 'data_plain':
                st += [(f"re + {idx}*s", zr0), (f"im + {idx}*s", zi0)]
            else:
                zr = g.add(zr0, g.inp(f"VL(cre + {idx}*s)"))
                zi = g.add(zi0, g.inp(f"VL(cim + {idx}*s)"))
                mr, mi = build_map(g, zr, zi)
                snapst += [(f"sre + {idx}*s", mr), (f"sim + {idx}*s", mi)]
                if out_mode == 'data_map':
                    st += [(f"re + {idx}*s", mr), (f"im + {idx}*s", mi)]
                else:
                    st += [(f"SM_R + {idx}*8", mr), (f"SM_I + {idx}*8", mi)]
        lines, rnd = emit_block(g, st + snapst, prefix="fo")
        body.append("  {")
        body += ["  " + l for l in lines]
        if snapst:
            body.append("    if(__builtin_expect(sre != 0, 0)){")
            for p, nd in snapst: body.append(f"      VS({p}, {rnd(nd)});")
            body.append("    }")
        for p, nd in st: body.append(f"    VS({p}, {rnd(nd)});")
        body.append("  }")

    body = [fn_head(kind, fname)]
    if kind == 'p':
        half(body, "re", "im", "s", True, 'data_plain')
    elif kind == 'm':
        half(body, "re", "im", "s", True, 'data_map')
    else:
        half(body, "re", "im", "s", True, 'scratch_map')
        half(body, "SM_R", "SM_I", "8", True, 'data_plain')
    body.append("}")
    return "\n".join(body)

# patch gen2's dispatcher
import gen2
gen2.gen_rader_fn = gen_rader_fn
from gen2 import gen_fft_fn

STRUCT = {
    6:  ('line',),
    8:  ('line',),
    13: ('dsym', 2),
    17: ('rader',),
    23: ('dsym', 2),
    36: ('pfa', 4, 9),
    45: ('pfa', 9, 5),
    64: ('ct', 8, 8),
}

HEADER = open('header_c.h').read()

DRIVER_L = r"""
/* ================= driver for L={L} ================= */
static void zpass_{L}(double *restrict pr, double *restrict pi){{
  for(int y0 = 0; y0 < {L}; y0 += 8){{
    int rows = {L} - y0; if(rows > 8) rows = 8;
    for(int jb = 0; jb < {NT}; jb++){{
      tr8x8_ld(pr + (long)y0*{R} + jb*8, {R}, ZSR + jb*64);
      tr8x8_ld(pi + (long)y0*{R} + jb*8, {R}, ZSI + jb*64);
    }}
#if {R} > {L}
    for(int j = {L}; j < {R}; j++){{
      VS(ZSR + j*8, VK(0.0));
      VS(ZSI + j*8, VK(0.0));
    }}
#endif
    f{L}_p(ZSR, ZSI, 8);
    for(int jb = 0; jb < {NT}; jb++){{
      tr8x8_st(pr + (long)y0*{R} + jb*8, {R}, ZSR + jb*64, rows);
      tr8x8_st(pi + (long)y0*{R} + jb*8, {R}, ZSI + jb*64, rows);
    }}
  }}
}}

static void zpassm_{L}(double *restrict pr, double *restrict pi,
                       const double *restrict ctr, const double *restrict cti, int cont){{
  for(int y0 = 0; y0 < {L}; y0 += 8){{
    int rows = {L} - y0; if(rows > 8) rows = 8;
    const double *cr = ctr + (long)(y0/8)*{R}*8;
    const double *ci = cti + (long)(y0/8)*{R}*8;
    for(int jb = 0; jb < {NT}; jb++){{
      tr8x8_ld(pr + (long)y0*{R} + jb*8, {R}, ZSR + jb*64);
      tr8x8_ld(pi + (long)y0*{R} + jb*8, {R}, ZSI + jb*64);
    }}
#if {R} > {L}
    for(int j = {L}; j < {R}; j++){{
      VS(ZSR + j*8, VK(0.0));
      VS(ZSI + j*8, VK(0.0));
    }}
#endif
    if(cont) f{L}_f(ZSR, ZSI, 8, cr, ci, 0, 0);
    else     f{L}_m(ZSR, ZSI, 8, cr, ci, 0, 0);
    for(int jb = 0; jb < {NT}; jb++){{
      tr8x8_st(pr + (long)y0*{R} + jb*8, {R}, ZSR + jb*64, rows);
      tr8x8_st(pi + (long)y0*{R} + jb*8, {R}, ZSI + jb*64, rows);
    }}
  }}
}}

static void iter_{L}(double *restrict re, double *restrict im,
                     const double *restrict cre, const double *restrict cim,
                     const double *restrict ctr, const double *restrict cti,
                     double *restrict bsr_, double *restrict bsi_, long m, int part){{
  for(long it = 1; it <= m; it++){{
    int snap = (it == 1), cont = (it < m);
    double *sr0 = snap ? bsr_ : 0, *si0 = snap ? bsi_ : 0;
    if(part == 0){{
      for(int x = 0; x < {L}; x++){{
        long p = (long)x*{PS};
        zpass_{L}(re + p, im + p);
        for(int zc = 0; zc < {NT}; zc++)
          f{L}_p(re + p + zc*8, im + p + zc*8, {R});
      }}
      part = 2;
    }}
    if(part == 2){{
      for(int y = 0; y < {L}; y++){{
        for(int zc = 0; zc < {NT}; zc++){{
          long q = (long)y*{R} + zc*8;
#if {PFV}
          long qn = (zc+2 < {NT}) ? q + 16 : ((long)(y+1)*{R} + (zc+2-{NT})*8);
          for(int j = 0; j < {L}; j++){{
            __builtin_prefetch(re+qn+j*{PS}); __builtin_prefetch(im+qn+j*{PS});
            __builtin_prefetch(cre+qn+j*{PS}); __builtin_prefetch(cim+qn+j*{PS});
          }}
#endif
          if(cont) f{L}_f(re+q, im+q, {PS}, cre+q, cim+q, sr0?sr0+q:0, si0?si0+q:0);
          else     f{L}_m(re+q, im+q, {PS}, cre+q, cim+q, sr0?sr0+q:0, si0?si0+q:0);
        }}
      }}
      part = cont ? 1 : 0;
    }} else {{
      for(int x = 0; x < {L}; x++){{
        long p = (long)x*{PS};
        zpass_{L}(re + p, im + p);
        for(int zc = 0; zc < {NT}; zc++){{
          long q = p + zc*8;
          if(cont) f{L}_f(re+q, im+q, {R}, cre+q, cim+q, sr0?sr0+q:0, si0?si0+q:0);
          else     f{L}_m(re+q, im+q, {R}, cre+q, cim+q, sr0?sr0+q:0, si0?si0+q:0);
        }}
        if(cont) zpass_{L}(re + p, im + p);
      }}
      part = cont ? 2 : 0;
    }}
  }}
}}

/* -------- B8 mode: lanes across 8 volumes -------- */
static void iterB8_{L}(double *restrict re, double *restrict im,
                       const double *restrict cre, const double *restrict cim,
                       double *restrict bsr_, double *restrict bsi_, long m, int part){{
  for(long it = 1; it <= m; it++){{
    int snap = (it == 1), cont = (it < m);
    double *sr0 = snap ? bsr_ : 0, *si0 = snap ? bsi_ : 0;
    if(part == 0){{
      for(int x = 0; x < {L}; x++){{
        long p = (long)x*{PSB};
        for(int z = 0; z < {L}; z++) f{L}_p(re + p + z*8, im + p + z*8, {RB8});
        for(int y = 0; y < {L}; y++) f{L}_p(re + p + y*{RB8}, im + p + y*{RB8}, 8);
      }}
      part = 2;
    }}
    if(part == 2){{
      for(int y = 0; y < {L}; y++) for(int z = 0; z < {L}; z++){{
        long q = (long)y*{RB8} + z*8;
#if {PFB}
        long qn = q + 16;
        for(int j = 0; j < {L}; j++){{
          __builtin_prefetch(re+qn+j*{PSB}); __builtin_prefetch(im+qn+j*{PSB});
          __builtin_prefetch(cre+qn+j*{PSB}); __builtin_prefetch(cim+qn+j*{PSB});
        }}
#endif
        if(cont) f{L}_f(re+q, im+q, {PSB}, cre+q, cim+q, sr0?sr0+q:0, si0?si0+q:0);
        else     f{L}_m(re+q, im+q, {PSB}, cre+q, cim+q, sr0?sr0+q:0, si0?si0+q:0);
      }}
      part = cont ? 1 : 0;
    }} else {{
      for(int x = 0; x < {L}; x++){{
        long p = (long)x*{PSB};
        for(int z = 0; z < {L}; z++) f{L}_p(re + p + z*8, im + p + z*8, {RB8});
        for(int y = 0; y < {L}; y++){{
          long q = p + y*{RB8};
          if(cont) f{L}_f(re+q, im+q, 8, cre+q, cim+q, sr0?sr0+q:0, si0?si0+q:0);
          else     f{L}_m(re+q, im+q, 8, cre+q, cim+q, sr0?sr0+q:0, si0?si0+q:0);
        }}
        if(cont)
          for(int z = 0; z < {L}; z++) f{L}_p(re + p + z*8, im + p + z*8, {RB8});
      }}
      part = cont ? 2 : 0;
    }}
  }}
}}

static void run_{L}(long B, long m, const double *x0, const double *cc, double *out_one, double *out_final){{
  long v = 0;
#if {USEB8}
  while(B - v >= {T8}){{
    int nv = (B - v >= 8) ? 8 : (int)(B - v);
    convB8_in(x0 + 2*v*{L3}L, nv, {L3}L, {L}, {RB8}L, {PSB}L, bre, bim);
    if(m >= 1)
      for(int x = 0; x < {L}; x++){{
        long p = (long)x*{PSB}L;
        for(int y = 0; y < {L}; y++) f{L}_p(bre + p + y*{RB8}L, bim + p + y*{RB8}L, 8);
        for(int z = 0; z < {L}; z++) f{L}_p(bre + p + z*8, bim + p + z*8, {RB8}L);
      }}
    convB8_in(cc + 2*v*{L3}L, nv, {L3}L, {L}, {RB8}L, {PSB}L, bcr, bci);
    if(m >= 1) iterB8_{L}(bre, bim, bcr, bci, bsr, bsi, m, 2);
    else {{ memcpy(bsr, bre, sizeof(double)*{L}*{PSB}L); memcpy(bsi, bim, sizeof(double)*{L}*{PSB}L); }}
    convB8_out(out_one + 2*v*{L3}L, nv, {L3}L, {L}, {RB8}L, {PSB}L, bsr, bsi);
    convB8_out(out_final + 2*v*{L3}L, nv, {L3}L, {L}, {RB8}L, {PSB}L, bre, bim);
    v += nv;
  }}
#endif
  for(; v < B; v++){{
    const double *sx = x0 + 2*v*{L3}L;
    const double *sc = cc + 2*v*{L3}L;
    for(int x = 0; x < {L}; x++){{
      for(int y = 0; y < {L}; y++){{
        long row = (long)(x*{L} + y)*{L};
        long d = (long)x*{PS} + (long)y*{R};
        deint_row(sx + 2*row, bre + d, bim + d, {L}, {R});
      }}
      if(m >= 1){{
        long p = (long)x*{PS};
        zpass_{L}(bre + p, bim + p);
        for(int zc = 0; zc < {NT}; zc++)
          f{L}_p(bre + p + zc*8, bim + p + zc*8, {R});
      }}
    }}
    for(int x = 0; x < {L}; x++){{
      for(int y = 0; y < {L}; y++){{
        long row = (long)(x*{L} + y)*{L};
        long d = (long)x*{PS} + (long)y*{R};
        deint_row(sc + 2*row, bcr + d, bci + d, {L}, {R});
      }}
      for(int t = 0; t < {YT}; t++){{
        long src_b = (long)x*{PS} + (long)(8*t)*{R};
        long dst_b = (long)x*{CTP} + (long)t*{R}*8;
        for(int jb = 0; jb < {NT}; jb++){{
          tr8x8_ld(bcr + src_b + jb*8, {R}, bctr + dst_b + jb*64);
          tr8x8_ld(bci + src_b + jb*8, {R}, bcti + dst_b + jb*64);
        }}
      }}
    }}
    if(m >= 1) iter_{L}(bre, bim, bcr, bci, bctr, bcti, bsr, bsi, m, 2);
    else {{ memcpy(bsr, bre, sizeof(double)*{L}*{PS}); memcpy(bsi, bim, sizeof(double)*{L}*{PS}); }}
    double *d1 = out_one + 2*v*{L3}L, *d2 = out_final + 2*v*{L3}L;
    for(int x = 0; x < {L}; x++)
      for(int y = 0; y < {L}; y++){{
        long row = (long)(x*{L} + y)*{L};
        long d = (long)x*{PS} + (long)y*{R};
        int_row(d1 + 2*row, bsr + d, bsi + d, {L});
        int_row(d2 + 2*row, bre + d, bim + d, {L});
      }}
  }}
}}
"""

SCRATCH = ["SR1","SI1","SR2","SI2","SR3","SI3","SA_R","SA_I","SP_R","SP_I","SD_R","SD_I","SM_R","SM_I","ZSR","ZSI"]
import re as _re
def restrictify(txt):
    """inside each function body, replace scratch array refs with local restrict pointers"""
    out = []
    for fn in txt.split("\n\n"):
        used = [s for s in SCRATCH if _re.search(r'\b' + s + r'\b', fn)]
        if used and fn.lstrip().startswith("static"):
            lines = fn.split("\n")
            # find first line ending with '{' (function head)
            for i, l in enumerate(lines):
                if l.rstrip().endswith("{"):
                    decl = "  " + " ".join(f"double *restrict {s}_p = {s};" for s in used)
                    lines.insert(i + 1, decl)
                    break
            fn = "\n".join(lines)
            for s in used:
                fn = _re.sub(r'\b' + s + r'\b(?!_p| =|;)', s + "_p", fn)
                fn = fn.replace(f"{s}_p = {s}_p;", f"{s}_p = {s};")
    # fix decl line that got mangled
        out.append(fn)
    return "\n\n".join(out)

def gen(path="implementation.c"):
    parts = [HEADER]
    # scratch arrays (staggered sizes to avoid 4K aliasing)
    parts.append("""
static double SR1[64*8+8] __attribute__((aligned(64)));
static double SI1[64*8+8] __attribute__((aligned(64)));
static double SR2[64*8+8] __attribute__((aligned(64)));
static double SI2[64*8+8] __attribute__((aligned(64)));
static double SR3[16*8+8] __attribute__((aligned(64)));
static double SI3[16*8+8] __attribute__((aligned(64)));
static double SA_R[22*8+8] __attribute__((aligned(64)));
static double SA_I[22*8+8] __attribute__((aligned(64)));
static double SP_R[22*8+8] __attribute__((aligned(64)));
static double SP_I[22*8+8] __attribute__((aligned(64)));
static double SD_R[22*8+8] __attribute__((aligned(64)));
static double SD_I[22*8+8] __attribute__((aligned(64)));
static double SM_R[23*8+8] __attribute__((aligned(64)));
static double SM_I[23*8+8] __attribute__((aligned(64)));
static double ZSR[64*8+8] __attribute__((aligned(64)));
static double ZSI[64*8+8] __attribute__((aligned(64)));
static double *bre, *bim, *bcr, *bci, *bsr, *bsi, *bctr, *bcti;
static vd TWR_64[7*7] __attribute__((aligned(64)));
static vd TWI_64[7*7] __attribute__((aligned(64)));
static vd TWR_36[5*5] __attribute__((aligned(64)));
static vd TWI_36[5*5] __attribute__((aligned(64)));
static double TB[64] __attribute__((aligned(64)));
static double TB2[64] __attribute__((aligned(64)));

/* B8 conversions: lanes = volumes; dst row stride RB8, plane stride PSB */
static void convB8_in(const double *restrict src, int nv, long L3, int L, long RB8, long PSB, double *restrict dre, double *restrict dim_){
  for(int x = 0; x < L; x++) for(int y = 0; y < L; y++){
    long base = ((long)x*L + y)*L;        // source point index of row start
    long dbase = (long)x*PSB + (long)y*RB8;
    long p = 0;
    for(; p + 4 <= L; p += 4){
      for(int l = 0; l < 8; l++){
        __m512d r = (l < nv) ? _mm512_loadu_pd(src + l*2*L3 + 2*(base+p)) : _mm512_setzero_pd();
        _mm512_store_pd(TB + l*8, r);
      }
      tr8x8_ld(TB, 8, TB2);
      for(int t = 0; t < 4; t++){
        VS(dre + dbase + (p+t)*8, VL(TB2 + (2*t)*8));
        VS(dim_ + dbase + (p+t)*8, VL(TB2 + (2*t+1)*8));
      }
    }
    for(; p < L; p++){
      for(int l = 0; l < 8; l++){
        dre[dbase + p*8 + l] = (l < nv) ? src[l*2*L3 + 2*(base+p)] : 0.0;
        dim_[dbase + p*8 + l] = (l < nv) ? src[l*2*L3 + 2*(base+p) + 1] : 0.0;
      }
    }
  }
}
static void convB8_out(double *restrict dst, int nv, long L3, int L, long RB8, long PSB, const double *restrict sre_, const double *restrict sim_){
  for(int x = 0; x < L; x++) for(int y = 0; y < L; y++){
    long base = ((long)x*L + y)*L;
    long dbase = (long)x*PSB + (long)y*RB8;
    long p = 0;
    for(; p + 4 <= L; p += 4){
      for(int t = 0; t < 4; t++){
        VS(TB2 + (2*t)*8, VL(sre_ + dbase + (p+t)*8));
        VS(TB2 + (2*t+1)*8, VL(sim_ + dbase + (p+t)*8));
      }
      tr8x8_ld(TB2, 8, TB);
      for(int l = 0; l < nv; l++)
        _mm512_storeu_pd(dst + l*2*L3 + 2*(base+p), _mm512_load_pd(TB + l*8));
    }
    for(; p < L; p++)
      for(int l = 0; l < nv; l++){
        dst[l*2*L3 + 2*(base+p)] = sre_[dbase + p*8 + l];
        dst[l*2*L3 + 2*(base+p) + 1] = sim_[dbase + p*8 + l];
      }
  }
}
""")
    fns = []
    for L in SIZES:
        for kind, suff in (('p', '_p'), ('m', '_m'), ('f', '_f')):
            fns.append(gen_fft_fn(L, STRUCT[L], kind, f"f{L}{suff}"))
    # tables
    tabs = []
    seen = set()
    for name, vals in CONSTS:
        if name in seen: continue
        seen.add(name)
        tabs.append(f"static const int {name}[{len(vals)}] = {{{', '.join(map(str, vals))}}};")
    parts.append("\n".join(tabs))
    parts.append("\n\n".join(LEAFS.values()))
    parts.append(restrictify("\n\n".join(fns)))
    T8D = {6: 5, 8: 5, 13: 6, 17: 6, 23: 8, 36: 8, 45: 8, 64: 8}
    RB8D = {L: L*8 + 8 for L in SIZES}
    PSBD = {L: L*RB8D[L] + 8 for L in SIZES}
    USE = {6: 1, 8: 1, 13: 1, 17: 1, 23: 0, 36: 1, 45: 0, 64: 0}
    PFVD = {6: 0, 8: 0, 13: 0, 17: 0, 23: 0, 36: 1, 45: 1, 64: 1}
    PFBD = {6: 0, 8: 0, 13: 0, 17: 0, 23: 1, 36: 1, 45: 1, 64: 1}
    for L in SIZES:
        R = RPAD[L]; PS = L * R + 8
        YT = (L + 7) // 8
        parts.append(restrictify(DRIVER_L.format(L=L, R=R, PS=PS, NT=R // 8, L3=L**3,
                                     LL8=L*L*8, L8=L*8, LL=L*L, RB8=RB8D[L], PSB=PSBD[L], T8=T8D[L], USEB8=USE[L],
                                     PFV=PFVD[L], PFB=PFBD[L], CTP=YT*R*8, YT=YT)))
    # init + dispatch: one shared buffer set sized to the max (L=64)
    maxn = max(max(L * (L * RPAD[L] + 8) + 16 * RPAD[L] + 64, (L * (L*(L*8+8) + 8) + 64) if USE[L] else 0) for L in SIZES)
    maxv = max(L * ((L + 7) // 8) * RPAD[L] * 8 + 64 for L in SIZES)
    parts.append(f"#define MAXV {maxv}")
    parts.append(f"""
static void init_tw(void){{
  for(int k2 = 1; k2 < 8; k2++)
    for(int t = 1; t < 8; t++){{
      double ang = -2*3.14159265358979323846264338328/64.0 * (double)(k2*t);
      double c = __builtin_cos(ang), s = __builtin_sin(ang);
      for(int l = 0; l < 8; l++){{ TWR_64[(k2-1)*7 + (t-1)][l] = c; TWI_64[(k2-1)*7 + (t-1)][l] = s; }}
    }}
  for(int k2 = 1; k2 < 6; k2++)
    for(int t = 1; t < 6; t++){{
      double ang = -2*3.14159265358979323846264338328/36.0 * (double)(k2*t);
      double c = __builtin_cos(ang), s = __builtin_sin(ang);
      for(int l = 0; l < 8; l++){{ TWR_36[(k2-1)*5 + (t-1)][l] = c; TWI_36[(k2-1)*5 + (t-1)][l] = s; }}
    }}
}}
__attribute__((constructor)) static void init_all(void){{
  init_tw();
  bre = alloc_buf({maxn}); bim = alloc_buf({maxn});
  bcr = alloc_buf({maxn}); bci = alloc_buf({maxn});
  bsr = alloc_buf({maxn}); bsi = alloc_buf({maxn});
  bctr = alloc_buf(MAXV); bcti = alloc_buf(MAXV);
}}
void run_size(long L, long B, long m, const double *x0, const double *cc, double *out_one, double *out_final){{
  switch(L){{""")
    for L in SIZES:
        parts.append(f"    case {L}: run_{L}(B, m, x0, cc, out_one, out_final); break;")
    parts.append("""  }
}
void test_fft(long L, double *re, double *im, long s){
  switch(L){""")
    for L in SIZES:
        parts.append(f"    case {L}: f{L}_p(re, im, s); break;")
    parts.append("""  }
}
""")
    src = "\n".join(parts)
    with open(path, "w") as f:
        f.write(src)
    print("lines:", src.count(chr(10)))

if __name__ == '__main__':
    gen()
