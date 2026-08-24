import sys
sys.path.insert(0, '/tmp/dev')
from gen2 import *
from genloop import symodd_loop, symodd_grouped, symodd_2col
from genloopb import staged_loop
from genmain_prelude import PRELUDE_BASE

SIZES = (6, 8, 13, 17, 23, 36, 45, 64)
CLASS_A = (6, 8, 13, 17, 23)
CLASS_B = (36, 45, 64)
PADB = {36: 40, 45: 48, 64: 72}

def class_a(L):
    out = []
    L3 = L*L*L
    eager = L in (6, 8)
    for W in (8,4,2,1):
        S1 = L*W
        S0 = L*L*W + 8
        if L in (13,17,23):
            GG = 3 if L == 13 else 4
            out.append(symodd_grouped(L, W, 'plain', W,  f"a{L}_z_w{W}", G=GG))
            out.append(symodd_grouped(L, W, 'zmapc', W,  f"a{L}_zm_w{W}", G=3))
            out.append(symodd_grouped(L, W, 'plain', S1, f"a{L}_y_w{W}", G=GG))
            out.append(symodd_grouped(L, W, 'plain', S0, f"a{L}_xp_w{W}", G=GG))
        else:
            out.append(gen_kernel(L, W, 'plain', W,  f"a{L}_z_w{W}"))
            out.append(gen_kernel(L, W, 'plain', S1, f"a{L}_y_w{W}"))
            out.append(gen_kernel_x2(L, W, S0, f"a{L}_x2_w{W}"))
    d = []
    d.append(f"static double *A{L}_xr, *A{L}_xi, *A{L}_cr, *A{L}_ci;")
    d.append(f"static void a{L}_init(void){{")
    d.append(f"  if(!A{L}_xr){{ size_t n = (size_t){L}*({L}*{L}*8+8)+16;")
    d.append(f"    A{L}_xr=xalloc(n*8); A{L}_xi=xalloc(n*8); A{L}_cr=xalloc(n*8); A{L}_ci=xalloc(n*8); }}")
    d.append("}")
    for W in (8,4,2,1):
        S1 = L*W; S0 = L*L*W + 8
        T = TYPE[W]
        d.append(f"""
static void a{L}_conv_in_w{W}(const double*restrict src, double*restrict dr, double*restrict di){{
  for(long v=0; v<{W}; ++v){{
    const double* s = src + v*(long){2*L3};
    for(long j0=0;j0<{L};++j0){{
      double* pr = dr + j0*{S0} + v;
      double* pi = di + j0*{S0} + v;
      const double* ss = s + j0*{2*L*L};
      for(long r=0;r<{L*L};++r){{ pr[r*{W}] = ss[2*r]; pi[r*{W}] = ss[2*r+1]; }}
    }}
  }}
}}
static void a{L}_emit_w{W}(const double*restrict dr, const double*restrict di, double*restrict dst){{
  for(long v=0; v<{W}; ++v){{
    double* s = dst + v*(long){2*L3};
    for(long j0=0;j0<{L};++j0){{
      const double* pr = dr + j0*{S0} + v;
      const double* pi = di + j0*{S0} + v;
      double* ss = s + j0*{2*L*L};
      for(long r=0;r<{L*L};++r){{ ss[2*r] = pr[r*{W}]; ss[2*r+1] = pi[r*{W}]; }}
    }}
  }}
}}""")
        if eager:
            d.append(f"""
static void a{L}_stepe_w{W}(double*restrict xr, double*restrict xi, const double*restrict cr, const double*restrict ci){{
  for(long j0=0;j0<{L};++j0){{
    double* br = xr + j0*{S0}; double* bi = xi + j0*{S0};
    for(long j1=0;j1<{L};++j1) a{L}_z_w{W}(br + j1*{S1}, bi + j1*{S1});
    for(long j2=0;j2<{L};++j2) a{L}_y_w{W}(br + j2*{W}, bi + j2*{W});
  }}
  for(long j1=0;j1<{L};++j1)
    for(long j2=0;j2<{L};j2+=2){{
      long off = j1*{S1} + j2*{W};
      a{L}_x2_w{W}(xr + off, xi + off, cr + off, ci + off);
    }}
}}""")
        else:
            d.append(f"""
static void a{L}_emitmc_w{W}(const double*restrict dr, const double*restrict di,
                             const double*restrict cr, const double*restrict ci, double*restrict dst){{
  double tbr[{W}], tbi[{W}];
  for(long j0=0;j0<{L};++j0){{
    for(long r=0;r<{L*L};++r){{
      long off = j0*{S0} + r*{W};
      {T} zr = *(const {T}*)(dr + off) + *(const {T}*)(cr + off);
      {T} zi = *(const {T}*)(di + off) + *(const {T}*)(ci + off);
      {T} mm = zr*zr + zi*zi;
      {T} uu = rsq{W}(mm);
      {T} wr_ = zr*uu, wi_ = zi*uu;
      {T} vv = rpc{W}(uu);
      *({T}*)(tbr) = wr_*vv; *({T}*)(tbi) = wi_*vv;
      for(long v=0;v<{W};++v){{
        double* s = dst + v*(long){2*L3} + (j0*{L*L} + r)*2;
        s[0] = tbr[v]; s[1] = tbi[v];
      }}
    }}
  }}
}}
static void a{L}_stepf_w{W}(double*restrict xr, double*restrict xi){{
  for(long j0=0;j0<{L};++j0){{
    double* br = xr + j0*{S0}; double* bi = xi + j0*{S0};
    for(long j1=0;j1<{L};++j1) a{L}_z_w{W}(br + j1*{S1}, bi + j1*{S1});
    for(long j2=0;j2<{L};++j2) a{L}_y_w{W}(br + j2*{W}, bi + j2*{W});
  }}
  for(long j1=0;j1<{L};++j1)
    for(long j2=0;j2<{L};++j2){{
      long off = j1*{S1} + j2*{W};
      a{L}_xp_w{W}(xr + off, xi + off);
    }}
}}
static void a{L}_stepr_w{W}(double*restrict xr, double*restrict xi, const double*restrict cr, const double*restrict ci){{
  for(long j0=0;j0<{L};++j0){{
    double* br = xr + j0*{S0}; double* bi = xi + j0*{S0};
    const double* kr = cr + j0*{S0}; const double* ki = ci + j0*{S0};
    for(long j1=0;j1<{L};++j1) a{L}_zm_w{W}(br + j1*{S1}, bi + j1*{S1}, kr + j1*{S1}, ki + j1*{S1});
    for(long j2=0;j2<{L};++j2) a{L}_y_w{W}(br + j2*{W}, bi + j2*{W});
  }}
  for(long j1=0;j1<{L};++j1)
    for(long j2=0;j2<{L};++j2){{
      long off = j1*{S1} + j2*{W};
      a{L}_xp_w{W}(xr + off, xi + off);
    }}
}}""")
    d.append(f"""
static void run{L}(long B, long m, const double* x0, const double* c, double* out1, double* outm){{
  a{L}_init();
  long b=0;
  while(b<B){{
    long rem=B-b; long V = rem>=8?8: rem>=4?4: rem>=2?2:1;
    const double* xs = x0 + b*(long){2*L3};
    const double* cs = c  + b*(long){2*L3};
    switch(V){{""")
    for W in (8,4,2,1):
        if eager:
            d.append(f"""    case {W}:
      a{L}_conv_in_w{W}(xs, A{L}_xr, A{L}_xi);
      a{L}_conv_in_w{W}(cs, A{L}_cr, A{L}_ci);
      for(long it=1; it<=m; ++it){{
        a{L}_stepe_w{W}(A{L}_xr, A{L}_xi, A{L}_cr, A{L}_ci);
        if(it==1) a{L}_emit_w{W}(A{L}_xr, A{L}_xi, out1 + b*(long){2*L3});
      }}
      a{L}_emit_w{W}(A{L}_xr, A{L}_xi, outm + b*(long){2*L3});
      break;""")
        else:
            d.append(f"""    case {W}:
      a{L}_conv_in_w{W}(xs, A{L}_xr, A{L}_xi);
      a{L}_conv_in_w{W}(cs, A{L}_cr, A{L}_ci);
      a{L}_stepf_w{W}(A{L}_xr, A{L}_xi);
      a{L}_emitmc_w{W}(A{L}_xr, A{L}_xi, A{L}_cr, A{L}_ci, out1 + b*(long){2*L3});
      for(long it=2; it<=m; ++it)
        a{L}_stepr_w{W}(A{L}_xr, A{L}_xi, A{L}_cr, A{L}_ci);
      a{L}_emitmc_w{W}(A{L}_xr, A{L}_xi, A{L}_cr, A{L}_ci, outm + b*(long){2*L3});
      break;""")
    d.append("""    }
    b += V;
  }
}""")
    return "\n".join(out) + "\n" + "\n".join(d)

def zgroups(L):
    gs = []
    z = 0
    while z + 8 <= L: gs.append((z, 8)); z += 8
    while z + 4 <= L: gs.append((z, 4)); z += 4
    while z < L: gs.append((z, 1)); z += 1
    return gs

def class_b(L):
    CO = PADB[L]          # component (im) offset within a row
    P = 2*CO              # row pitch in doubles (re segment + im segment)
    LP8 = ((L + 7)//8)*8 if L == 45 else L  # pad z/y lanes to 8-wide groups (only 45 profits)
    PS = LP8*P + 8        # plane stride: room for pad rows written by T2
    L3 = L*L*L
    pos = crt_pos(L)
    widths = (8,) if L in (45, 64) else tuple(sorted(set(w for _, w in zgroups(L)), reverse=True))
    out = []
    for W in widths:
        if L == 64:
            out.append(staged_loop(L, W, 'colT', f"b{L}_t_w{W}", P, PS))
            out.append(staged_loop(L, W, 'colT_mapc', f"b{L}_tm_w{W}", P, PS))
            out.append(staged_loop(L, W, 'xplain', f"b{L}_xp_w{W}", P, PS))
        elif L == 45:
            out.append(gen_staged(L, W, 'colT', f"b{L}_t_w{W}", None, P=P))
            out.append(gen_staged(L, W, 'colT_mapc', f"b{L}_tm_w{W}", None, P=P))
            out.append(gen_staged(L, W, 'xplain', f"b{L}_xp_w{W}", PS))
        else:
            out.append(gen_staged(L, W, 'colT', f"b{L}_t_w{W}", None, P=P))
            out.append(gen_staged(L, W, 'xfused', f"b{L}_xf_w{W}", PS))
    d = []
    pos_c = ",".join(str(p) for p in pos)
    inv = [0]*L
    for k in range(L): inv[pos[k]] = k
    inv_c = ",".join(str(p) for p in inv)
    d.append(f"static const int pos{L}[{L}] = {{{pos_c}}};")
    d.append(f"static const int invpos{L}[{L}] = {{{inv_c}}};")
    d.append(f"static double *B{L}_x, *B{L}_c, *B{L}_t, *B{L}_t2, *B{L}_xb, *B{L}_cb;")
    d.append(f"static void b{L}_init(void){{ if(!B{L}_x){{")
    d.append(f"  size_t n = (size_t){L}*{PS}+64;")
    d.append(f"  B{L}_x=xalloc(n*8); B{L}_c=xalloc(n*8); B{L}_xb=xalloc(n*8); B{L}_cb=xalloc(n*8);")
    d.append(f"  size_t t = (size_t){LP8}*{P}+64; B{L}_t=xalloc(t*8); B{L}_t2=xalloc(t*8); }} }}")
    d.append(f"""
static void b{L}_conv_in(const double*restrict src, double*restrict dx, double padval){{
  for(long j0=0;j0<{L};++j0){{
    for(long j1=0;j1<{L};++j1){{
      const double* s = src + (j0*{L}+j1)*{2*L};
      double* pr = dx + j0*{PS} + pos{L}[j1]*{P};
      double* pi = pr + {CO};
      for(long z={L};z<{LP8};++z){{ pr[z]=padval; pi[z]=0.0; }}
      for(long z=0;z<{L};++z){{ pr[pos{L}[z]]=s[2*z]; pi[pos{L}[z]]=s[2*z+1]; }}
    }}
    for(long j1={L};j1<{LP8};++j1){{
      double* pr = dx + j0*{PS} + j1*{P};
      double* pi = pr + {CO};
      for(long z=0;z<{LP8};++z){{ pr[z]=padval; pi[z]=0.0; }}
    }}
  }}
}}
static void b{L}_emit(const double*restrict dx, double*restrict dst){{
  for(long j0=0;j0<{L};++j0) for(long j1=0;j1<{L};++j1){{
    double* s = dst + (j0*{L}+j1)*{2*L};
    const double* pr = dx + j0*{PS} + pos{L}[j1]*{P};
    const double* pi = pr + {CO};
    for(long z=0;z<{L};++z){{ s[2*z]=pr[pos{L}[z]]; s[2*z+1]=pi[pos{L}[z]]; }}
  }}
}}""")
    g8 = [(z, 8) for z in range(0, LP8, 8)] if L == 45 else zgroups(L)
    eager = (L == 36)
    calls_t1 = "\n".join(f"    b{L}_t_w{w}(br + {zo}, br + {zo} + {CO}, tr + {zo}*{P}, tr + {zo}*{P} + {CO});" for zo, w in g8)
    calls_t1m = "" if eager else "\n".join(f"    b{L}_tm_w{w}(br + {zo}, br + {zo} + {CO}, kr + {zo}, kr + {zo} + {CO}, tr + {zo}*{P}, tr + {zo}*{P} + {CO});" for zo, w in g8)

    calls_t2 = "\n".join(f"    b{L}_t_w{w}(tr + {yo}, tr + {yo} + {CO}, br + {yo}*{P}, br + {yo}*{P} + {CO});" for yo, w in g8)
    calls_x = ("\n".join(f"      b{L}_xf_w{w}(xr + off + {zo}, xr + off + {zo} + {CO}, cx + off + {zo}, cx + off + {zo} + {CO});" for zo, w in g8) if eager
               else "\n".join(f"      b{L}_xp_w{w}(xr + off + {zo}, xr + off + {zo} + {CO});" for zo, w in g8))
    if eager:
        d.append(f"""
static void b{L}_step(double*restrict xr, const double*restrict cx, double*restrict tr){{
  for(long j0=0;j0<{L};++j0){{
    double* br = xr + j0*{PS};
{calls_t1}
{calls_t2}
  }}
  for(long j1=0;j1<{L};++j1){{
    long off = j1*{P};
{calls_x}
  }}
}}
static void run{L}(long B, long m, const double* x0, const double* c, double* out1, double* outm){{
  b{L}_init();
  for(long b=0;b<B;++b){{
    b{L}_conv_in(x0 + b*(long){2*L3}, B{L}_x, 1.0);
    b{L}_conv_in(c  + b*(long){2*L3}, B{L}_c, 0.0);
    for(long it=1; it<=m; ++it){{
      b{L}_step(B{L}_x, B{L}_c, B{L}_t);
      if(it==1) b{L}_emit(B{L}_x, out1 + b*(long){2*L3});
    }}
    b{L}_emit(B{L}_x, outm + b*(long){2*L3});
  }}
}}""")
        return "\n".join(out) + "\n" + "\n".join(d)
    d.append(f"""
static void b{L}_t1_plane(double*restrict br, double*restrict tr){{
{calls_t1}
}}
static void b{L}_t1m_plane(double*restrict br, const double*restrict kr, double*restrict tr){{
{calls_t1m}
}}
static void b{L}_t2_plane(double*restrict br, double*restrict tr){{
{calls_t2}
}}
static void b{L}_xslice(double*restrict xr, long j1){{
  long off = j1*{P};
{calls_x}
}}
static void b{L}_stepf(double*restrict xr, double*restrict tr){{
  for(long j0=0;j0<{L};++j0){{
    b{L}_t1_plane(xr + j0*{PS}, tr);
    b{L}_t2_plane(xr + j0*{PS}, tr);
  }}
  for(long j1=0;j1<{L};++j1) b{L}_xslice(xr, j1);
}}
/* interleave X-phase of volume A (already transformed this step) with the
   T-phase (step `tkind`: 0=plain first-step, 1=map-fused) of volume B.
   Uses two scratch planes so T1(q) can run while T2(q-1) drains. */
static void b{L}_XT(double*restrict xa,
                    double*restrict xb, const double*restrict cb,
                    double*restrict trA, double*restrict trB, int tmap){{
  /* q = 0: X slice 0 + T1(plane 0) -> trA */
  b{L}_xslice(xa, 0);
  if(tmap) b{L}_t1m_plane(xb, cb, trA); else b{L}_t1_plane(xb, trA);
  for(long q=1;q<{L};++q){{
    b{L}_xslice(xa, q);
    b{L}_t2_plane(xb + (q-1)*{PS}, trA);
    if(tmap) b{L}_t1m_plane(xb + q*{PS}, cb + q*{PS}, trB);
    else     b{L}_t1_plane(xb + q*{PS}, trB);
    double* sw = trA; trA = trB; trB = sw;
  }}
  b{L}_t2_plane(xb + ({L}-1)*{PS}, trA);
}}
static void b{L}_stepr(double*restrict xr, const double*restrict cx, double*restrict tr, double*restrict tr_unused){{
  (void)tr_unused;
  for(long j0=0;j0<{L};++j0){{
    b{L}_t1m_plane(xr + j0*{PS}, cx + j0*{PS}, tr);
    b{L}_t2_plane(xr + j0*{PS}, tr);
  }}
  for(long j1=0;j1<{L};++j1) b{L}_xslice(xr, j1);
}}
static void b{L}_emitmc(const double*restrict dx, const double*restrict cx, double*restrict dst){{
  double tbr[8], tbi[8];
  for(long j0=0;j0<{L};++j0) for(long j1=0;j1<{L};++j1){{
    double* s = dst + (j0*{L}+j1)*{2*L};
    const double* pr = dx + j0*{PS} + pos{L}[j1]*{P};
    const double* pi = pr + {CO};
    const double* qr = cx + j0*{PS} + pos{L}[j1]*{P};
    const double* qi = qr + {CO};
    for(long zb=0; zb*8<{L}; ++zb){{
      long z0 = zb*8;
      v8df zr = *(const v8df*)(pr + z0) + *(const v8df*)(qr + z0);
      v8df zi = *(const v8df*)(pi + z0) + *(const v8df*)(qi + z0);
      v8df mm = zr*zr + zi*zi;
      v8df uu = rsq8(mm);
      v8df wr_ = zr*uu, wi_ = zi*uu;
      v8df vv = rpc8(uu);
      *(v8df*)(tbr) = wr_*vv; *(v8df*)(tbi) = wi_*vv;
      long lim = {L} - z0; if(lim>8) lim=8;
      for(long q=0;q<lim;++q){{ long z = z0+q; long zi2 = 0;
        // find logical z for storage position z0+q: positions are permuted, use inverse below
        (void)zi2;
        s[2*invpos{L}[z]] = tbr[q]; s[2*invpos{L}[z]+1] = tbi[q];
      }}
    }}
  }}
}}
static void b{L}_Tphase(double*restrict xr, const double*restrict cx, double*restrict tr, int tmap){{
  for(long j0=0;j0<{L};++j0){{
    if(tmap) b{L}_t1m_plane(xr + j0*{PS}, cx + j0*{PS}, tr);
    else     b{L}_t1_plane(xr + j0*{PS}, tr);
    b{L}_t2_plane(xr + j0*{PS}, tr);
  }}
}}
static void run{L}_one(long m, const double* x0, const double* c, double* out1, double* outm){{
  b{L}_conv_in(x0, B{L}_x, 1.0);
  b{L}_conv_in(c,  B{L}_c, 0.0);
  b{L}_stepf(B{L}_x, B{L}_t);
  b{L}_emitmc(B{L}_x, B{L}_c, out1);
  for(long it=2; it<=m; ++it)
    b{L}_stepr(B{L}_x, B{L}_c, B{L}_t, B{L}_t2);
  b{L}_emitmc(B{L}_x, B{L}_c, outm);
}}
static void run{L}_pair(long m, const double* x0, const double* c, double* out1, double* outm){{
  double *xa = B{L}_x, *ca = B{L}_c, *xb = B{L}_xb, *cb = B{L}_cb;
  const long VN = (long){2*L3};
  b{L}_conv_in(x0,      xa, 1.0);
  b{L}_conv_in(c,       ca, 0.0);
  b{L}_conv_in(x0 + VN, xb, 1.0);
  b{L}_conv_in(c  + VN, cb, 0.0);
  b{L}_Tphase(xa, ca, B{L}_t, 0);
  for(long it=1; it<=m; ++it){{
    b{L}_XT(xa, xb, cb, B{L}_t, B{L}_t2, it>=2);
    if(it==1) b{L}_emitmc(xa, ca, out1);
    if(it<m){{
      b{L}_XT(xb, xa, ca, B{L}_t, B{L}_t2, 1);
    }} else {{
      for(long j1=0;j1<{L};++j1) b{L}_xslice(xb, j1);
    }}
    if(it==1) b{L}_emitmc(xb, cb, out1 + VN);
  }}
  b{L}_emitmc(xa, ca, outm);
  b{L}_emitmc(xb, cb, outm + VN);
}}
static void run{L}(long B, long m, const double* x0, const double* c, double* out1, double* outm){{
  b{L}_init();
  long b = 0;
  if(0){{ /* pairwise interleave: no gain measured (shared L3 bottleneck) */
    for(; b+2<=B; b+=2)
      run{L}_pair(m, x0 + b*(long){2*L3}, c + b*(long){2*L3}, out1 + b*(long){2*L3}, outm + b*(long){2*L3});
  }}
  for(; b<B; ++b)
    run{L}_one(m, x0 + b*(long){2*L3}, c + b*(long){2*L3}, out1 + b*(long){2*L3}, outm + b*(long){2*L3});
}}""")
    return "\n".join(out) + "\n" + "\n".join(d)

PROVENANCE = '/*\n * implementation.c -- iterated batched 3D complex DFT for L in {6,8,13,17,23,36,45,64}\n *\n * All transform arithmetic here is our own, generated by the scripts in ./gen\n * (straight-line / loop DFT kernels built from an expression-level code\n * generator: mixed-radix Cooley-Tukey, Good-Thomas PFA, and symmetric\n * odd-prime kernels; twiddle constants computed in extended precision and\n * rounded to double). No FFT library is called or copied anywhere.\n *\n * Structure:\n *  - sizes 6,8,13,17,23 ("class A"): batches of up to 8 volumes processed in\n *    SIMD lanes (SoA, batch-interleaved); per step: z-pass, y-pass (per\n *    plane), then x-pass; the elementwise map z/(1+|z|) is fused into the\n *    first pass of the following step (sizes 13,17,23) or into the x-pass\n *    (sizes 6,8).\n *  - sizes 36,45,64 ("class B"): one volume at a time, vectors along the\n *    contiguous z axis; per plane the y- and z-transforms are done as two\n *    "transform + transposed store" passes through a scratch plane; the\n *    x-pass runs over plane-strided rows. Output frequencies are kept in a\n *    fixed permuted order internally (CRT / digit-reversal), un-permuted\n *    only during conversion.\n *  - |z| and 1/(1+|z|) are computed with AVX-512 rsqrt14/rcp14 seeds plus\n *    two Newton refinements each (~1-2 ulp, full double precision).\n *  - single-threaded throughout; FTZ/DAZ set only inside run_size (real\n *    data never becomes denormal; this only guards padding lanes).\n */\n'

def main():
    parts = [PROVENANCE, PRELUDE_BASE, EXTRA_HELPERS]
    for L in CLASS_A:
        parts.append(class_a(L))
    for L in CLASS_B:
        parts.append(class_b(L))
    disp = ["void run_size(long L, long B, long m, const double* x0, const double* c, double* out1, double* outm){",
            "  unsigned int _mx = _mm_getcsr();",
            "  _mm_setcsr(_mx | 0x8040u);  /* FTZ|DAZ: pad lanes only; real data never denormal */",
            "  switch(L){"]
    for L in SIZES:
        disp.append(f"    case {L}: run{L}(B,m,x0,c,out1,outm); break;")
    disp.append("  }\n  _mm_setcsr(_mx);\n}")
    parts.append("\n".join(disp))
    src = "\n".join(parts)
    with open('/tmp/dev/implementation.c','w') as f:
        f.write(src)
    print("lines:", src.count("\n"))

main()
