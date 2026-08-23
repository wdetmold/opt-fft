src = open('implementation.c').read()

# 1. extra buffers: X2re/X2im (ping-pong) and C2re/C2im (z-major c) -> 8 planes
src = src.replace("static double *Xre, *Xim, *Cre, *Cim;   // planar volume buffers (max size)",
"""static double *Xre, *Xim, *Cre, *Cim;   // planar volume buffers (max size)
static double *X2re, *X2im, *C2re, *C2im; // ping-pong state + transposed c (small sizes)""")
src = src.replace("""    size_t tot = ((4 * one + (1 << 21) - 1) >> 21) << 21;
    char *blk = (char *)aligned_alloc(1 << 21, tot);
    memset(blk, 0, tot);
    Xre = (double *)blk; Xim = (double *)(blk + one);
    Cre = (double *)(blk + 2 * one); Cim = (double *)(blk + 3 * one);
  }""",
"""    size_t small = 24 * 584 * sizeof(double);   // largest parity-size plane (L=23)
    size_t tot = ((4 * one + 4 * small + (1 << 21) - 1) >> 21) << 21;
    char *blk = (char *)aligned_alloc(1 << 21, tot);
    memset(blk, 0, tot);
    Xre = (double *)blk; Xim = (double *)(blk + one);
    Cre = (double *)(blk + 2 * one); Cim = (double *)(blk + 3 * one);
    X2re = (double *)(blk + 4 * one); X2im = (double *)(blk + 4 * one + small);
    C2re = (double *)(blk + 4 * one + 2 * small); C2im = (double *)(blk + 4 * one + 3 * small);
  }""")

# 2. parity sweepA for 13/17/23: strided pass in place on src, tile pass src->dst w/o inverse transpose
parity_code = '''
// ---- parity engines: tile pass writes transposed (no inverse transpose);
// slice orientation alternates each iteration; sweepB picks matching c copy.
#define DEF_SWEEPA_P(LN, L_, LP_, R_, LPS_, SS_, FFTLINE)                             \\
  static void sweepAp_##LN(double *sre, double *sim, double *dre, double *dim) {      \\
    for (long x = 0; x < L_; x++) {                                                   \\
      double *br = sre + x * (long)SS_, *bi = sim + x * (long)SS_;                    \\
      double *cr = dre + x * (long)SS_, *ci = dim + x * (long)SS_;                    \\
      for (long zc = 0; zc < LP_; zc += 8) FFTLINE(br + zc, bi + zc, R_, 0, 0, 0, 0); \\
      for (long y0 = 0; y0 < LPS_; y0 += 8) {                                         \\
        tile_load(br + y0 * R_, R_, SZA, LP_ / 8);                                    \\
        tile_load(bi + y0 * R_, R_, SZB, LP_ / 8);                                    \\
        FFTLINE(SZA, SZB, 8, 0, 0, 0, 0);                                             \\
        for (long j = 0; j < L_; j++) {                                               \\
          VS(cr + j * R_ + y0, VL(SZA + j * 8));                                      \\
          VS(ci + j * R_ + y0, VL(SZB + j * 8));                                      \\
        }                                                                             \\
      }                                                                               \\
    }                                                                                 \\
  }                                                                                   \\
  static void sweepBp_##LN(double *xr, double *xi, const double *ccr, const double *cci) { \\
    for (long y = 0; y < L_; y++) {                                                   \\
      long base = y * R_;                                                             \\
      for (long zc = 0; zc < LP_; zc += 8) {                                          \\
        long off = base + zc;                                                         \\
        FFTLINE(xr + off, xi + off, (long)SS_, 1, ccr + off, cci + off, 0);           \\
      }                                                                               \\
    }                                                                                 \\
  }

DEF_SWEEPA_P(13, 13, 16, 16, 16, 264, fft13_line)
DEF_SWEEPA_P(17, 17, 24, 24, 24, 584, fft17_line)
DEF_SWEEPA_P(23, 23, 24, 24, 24, 584, fft23_line)

'''
src = src.replace("// ---------------------------------------------------------------- SoA batch-of-8 engine",
                  parity_code + "// ---------------------------------------------------------------- SoA batch-of-8 engine")

# 3. transposed conversions (scalar fine, once per call)
src = src.replace("// ---------------------------------------------------------------- init",
"""static void load_vol_T(long L, long R, long SS, const double *xin, double *dre, double *dim) {
  for (long x = 0; x < L; x++)
    for (long y = 0; y < L; y++) {
      const double *p = xin + (x * L + y) * L * 2;
      double *qr = dre + x * SS + y, *qi = dim + x * SS + y;
      for (long z = 0; z < L; z++) { qr[z * R] = p[2 * z]; qi[z * R] = p[2 * z + 1]; }
    }
}
static void store_vol_T(long L, long R, long SS, const double *sre, const double *sim, double *xout) {
  for (long x = 0; x < L; x++)
    for (long y = 0; y < L; y++) {
      double *p = xout + (x * L + y) * L * 2;
      const double *qr = sre + x * SS + y, *qi = sim + x * SS + y;
      for (long z = 0; z < L; z++) { p[2 * z] = qr[z * R]; p[2 * z + 1] = qi[z * R]; }
    }
}

// ---------------------------------------------------------------- init""")

# 4. parity driver macro for 13/17/23 (with optional SoA prepath)
src = src.replace("""#define RUN_CASE(LN, L_, R_, SS_)""",
"""#define RUN_VOL_P(LN, L_, R_, SS_)                                   \\
      {                                                                \\
        load_vol(L_, R_, SS_, xin + b * 2 * n, Xre, Xim);              \\
        load_vol(L_, R_, SS_, cin + b * 2 * n, Cre, Cim);              \\
        load_vol_T(L_, R_, SS_, cin + b * 2 * n, C2re, C2im);          \\
        double *sr = Xre, *si = Xim, *dr = X2re, *di = X2im, *t;       \\
        for (long it = 0; it < m; it++) {                              \\
          sweepAp_##LN(sr, si, dr, di);                                \\
          if ((it & 1) == 0) sweepBp_##LN(dr, di, C2re, C2im);         \\
          else               sweepBp_##LN(dr, di, Cre, Cim);           \\
          if (it == 0) store_vol_T(L_, R_, SS_, dr, di, out1 + b * 2 * n); \\
          t = sr; sr = dr; dr = t;  t = si; si = di; di = t;           \\
        }                                                              \\
        if (m & 1) store_vol_T(L_, R_, SS_, sr, si, outm + b * 2 * n); \\
        else       store_vol(L_, R_, SS_, sr, si, outm + b * 2 * n);   \\
      }

#define RUN_CASE_P(LN, L_, R_, SS_)                                  \\
  case L_: {                                                         \\
    long n = (long)L_ * L_ * L_;                                     \\
    for (long b = 0; b < B; b++) RUN_VOL_P(LN, L_, R_, SS_)          \\
  } break;

#define RUN_CASE_P_SOA(LN, L_, R_, SS_, SP_)                         \\
  case L_: {                                                         \\
    long n = (long)L_ * L_ * L_;                                     \\
    long b0 = 0;                                                     \\
    while (B - b0 >= 8) {                                            \\
      soa_load(L_, SP_, xin + b0 * 2 * n, Xre, Xim);                 \\
      soa_load(L_, SP_, cin + b0 * 2 * n, Cre, Cim);                 \\
      for (long it = 0; it < m; it++) {                              \\
        soa_iter_##LN();                                             \\
        if (it == 0) soa_store(L_, SP_, Xre, Xim, out1 + b0 * 2 * n);\\
      }                                                              \\
      soa_store(L_, SP_, Xre, Xim, outm + b0 * 2 * n);               \\
      b0 += 8;                                                       \\
    }                                                                \\
    for (long b = b0; b < B; b++) RUN_VOL_P(LN, L_, R_, SS_)         \\
  } break;

#define RUN_CASE(LN, L_, R_, SS_)""")

src = src.replace("    RUN_CASE_SOA(13, 13, 16, 264, 8 * (169 + 1))", "    RUN_CASE_P_SOA(13, 13, 16, 264, 8 * (169 + 1))")
src = src.replace("    RUN_CASE_SOA(17, 17, 24, 584, 8 * (289 + 1))", "    RUN_CASE_P_SOA(17, 17, 24, 584, 8 * (289 + 1))")
src = src.replace("    RUN_CASE(23, 23, 24, 584)", "    RUN_CASE_P(23, 23, 24, 584)")
open('implementation.c','w').write(src)
print("ok")
