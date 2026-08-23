src = open('implementation.c').read()

soa = '''
// ---------------------------------------------------------------- SoA batch-of-8 engine
// For small sizes with B >= 8: lanes = 8 volumes. Layout: plane[x][y*L+z][lane],
// slice stride SP = 8*(L*L+1) (anti-aliasing pad). No transposes, no pad lanes.
#define DEF_SOA(LN, L_, SP_)                                                  \\
  static void soa_iter_##LN(void) {                                          \\
    for (long x = 0; x < L_; x++) {                                          \\
      double *br = Xre + x * SP_, *bi = Xim + x * SP_;                       \\
      for (long y = 0; y < L_; y++)                                          \\
        fft##LN##_line(br + y * (L_ * 8), bi + y * (L_ * 8), 8, 0, 0, 0, 0); \\
      for (long z = 0; z < L_; z++)                                          \\
        fft##LN##_line(br + z * 8, bi + z * 8, L_ * 8, 0, 0, 0, 0);          \\
    }                                                                        \\
    for (long y = 0; y < L_; y++)                                            \\
      for (long z = 0; z < L_; z++) {                                        \\
        long off = (y * L_ + z) * 8;                                         \\
        fft##LN##_line(Xre + off, Xim + off, SP_, 1, Cre + off, Cim + off, 0); \\
      }                                                                      \\
  }

DEF_SOA(6, 6, 8 * (36 + 1))
DEF_SOA(8, 8, 8 * (64 + 1))
DEF_SOA(13, 13, 8 * (169 + 1))
DEF_SOA(17, 17, 8 * (289 + 1))
DEF_SOA(23, 23, 8 * (529 + 1))

static void soa_load(long L, long SP, const double *xin, double *dre, double *dim) {
  long n = L * L * L, n2 = L * L;
  for (long v = 0; v < 8; v++) {
    const double *p = xin + v * 2 * n;
    for (long x = 0; x < L; x++) {
      double *qr = dre + x * SP + v, *qi = dim + x * SP + v;
      const double *px = p + x * n2 * 2;
      for (long i = 0; i < n2; i++) { qr[i * 8] = px[2 * i]; qi[i * 8] = px[2 * i + 1]; }
    }
  }
}
static void soa_store(long L, long SP, const double *sre, const double *sim, double *xout) {
  long n = L * L * L, n2 = L * L;
  for (long v = 0; v < 8; v++) {
    double *p = xout + v * 2 * n;
    for (long x = 0; x < L; x++) {
      const double *qr = sre + x * SP + v, *qi = sim + x * SP + v;
      double *px = p + x * n2 * 2;
      for (long i = 0; i < n2; i++) { px[2 * i] = qr[i * 8]; px[2 * i + 1] = qi[i * 8]; }
    }
  }
}

'''
src = src.replace("// ---------------------------------------------------------------- conversions",
                  soa + "// ---------------------------------------------------------------- conversions")

# driver: SOA-enabled RUN_CASE
src = src.replace("""#define RUN_CASE(LN, L_, R_, SS_)                                  \\
  case L_: {                                                         \\
    long n = (long)L_ * L_ * L_;                                     \\
    for (long b = 0; b < B; b++) {                                   \\""",
"""#define RUN_CASE_SOA(LN, L_, R_, SS_, SP_)                          \\
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
    for (long b = b0; b < B; b++) {                                  \\
      load_vol(L_, R_, SS_, xin + b * 2 * n, Xre, Xim);              \\
      load_vol(L_, R_, SS_, cin + b * 2 * n, Cre, Cim);              \\
      for (long it = 0; it < m; it++) {                              \\
        sweepA_##LN();                                               \\
        sweepB_##LN();                                               \\
        if (it == 0) store_vol(L_, R_, SS_, Xre, Xim, out1 + b * 2 * n); \\
      }                                                              \\
      store_vol(L_, R_, SS_, Xre, Xim, outm + b * 2 * n);            \\
    }                                                                \\
  } break;

#define RUN_CASE(LN, L_, R_, SS_)                                  \\
  case L_: {                                                         \\
    long n = (long)L_ * L_ * L_;                                     \\
    for (long b = 0; b < B; b++) {                                   \\""")
src = src.replace("    RUN_CASE(6, 6, 8, 72)", "    RUN_CASE_SOA(6, 6, 8, 72, 8 * (36 + 1))")
src = src.replace("    RUN_CASE(8, 8, 8, 72)", "    RUN_CASE_SOA(8, 8, 8, 72, 8 * (64 + 1))")
src = src.replace("    RUN_CASE(13, 13, 16, 264)", "    RUN_CASE_SOA(13, 13, 16, 264, 8 * (169 + 1))")
src = src.replace("    RUN_CASE(17, 17, 24, 584)", "    RUN_CASE_SOA(17, 17, 24, 584, 8 * (289 + 1))")
src = src.replace("    RUN_CASE(23, 23, 24, 584)", "    RUN_CASE_SOA(23, 23, 24, 584, 8 * (529 + 1))")
open('implementation.c','w').write(src)
print("ok")
