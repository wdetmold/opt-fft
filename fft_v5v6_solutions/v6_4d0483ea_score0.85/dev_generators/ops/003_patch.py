import re
src = open('implementation.c').read()

# Replace DEF_ENGINE macro and instantiations
old_engine = src[src.index('// ---------------------------------------------------------------- engines'):src.index('// ---------------------------------------------------------------- conversions')]
new_engine = '''// ---------------------------------------------------------------- engines
// layout: element (x,y,z) at  x*SS + y*R + z  (doubles) in each plane.
// LP = vector-padded z extent, LPS = padded rows per slice (junk rows stay 0),
// R = row stride, SS = slice stride; R and SS chosen to spread cache sets.
#define DEF_ENGINE(LN, L_, LP_, R_, LPS_, SS_, FFTLINE)                               \\
  static void sweepA_##LN(void) {                                                     \\
    for (long x = 0; x < L_; x++) {                                                   \\
      double *br = Xre + x * (long)SS_, *bi = Xim + x * (long)SS_;                    \\
      for (long y0 = 0; y0 < LPS_; y0 += 8) {                                         \\
        tile_load(br + y0 * R_, R_, SZA, LP_ / 8);                                    \\
        tile_load(bi + y0 * R_, R_, SZB, LP_ / 8);                                    \\
        FFTLINE(SZA, SZB, 8, 0, 0, 0);                                                \\
        tile_store(br + y0 * R_, R_, SZA, LP_ / 8, L_);                               \\
        tile_store(bi + y0 * R_, R_, SZB, LP_ / 8, L_);                               \\
      }                                                                               \\
      for (long zc = 0; zc < LP_; zc += 8) FFTLINE(br + zc, bi + zc, R_, 0, 0, 0);    \\
    }                                                                                 \\
  }                                                                                   \\
  static void sweepB_##LN(void) {                                                     \\
    for (long y = 0; y < L_; y++) {                                                   \\
      long base = y * R_;                                                             \\
      for (long zc = 0; zc < LP_; zc += 8) {                                          \\
        long off = base + zc;                                                         \\
        FFTLINE(Xre + off, Xim + off, (long)SS_, 1, Cre + off, Cim + off);            \\
      }                                                                               \\
    }                                                                                 \\
  }

DEF_ENGINE(6, 6, 8, 8, 8, 72, fft6_line)
DEF_ENGINE(8, 8, 8, 8, 8, 72, fft8_line)
DEF_ENGINE(13, 13, 16, 16, 16, 264, fft13_line)
DEF_ENGINE(17, 17, 24, 24, 24, 584, fft17_line)
DEF_ENGINE(23, 23, 24, 24, 24, 584, fft23_line)
DEF_ENGINE(36, 36, 40, 40, 40, 1608, fft36_line)
DEF_ENGINE(45, 45, 48, 48, 48, 2312, fft45_line)
DEF_ENGINE(64, 64, 64, 72, 64, 4616, fft64_line)

'''
src = src.replace(old_engine, new_engine)

# conversions
old_conv = src[src.index('// ---------------------------------------------------------------- conversions'):src.index('// ---------------------------------------------------------------- init')]
new_conv = '''// ---------------------------------------------------------------- conversions
static void load_vol(long L, long R, long SS, const double *xin, double *dre, double *dim) {
  for (long x = 0; x < L; x++)
    for (long y = 0; y < L; y++) {
      const double *p = xin + (x * L + y) * L * 2;
      double *qr = dre + x * SS + y * R, *qi = dim + x * SS + y * R;
      for (long z = 0; z < L; z++) { qr[z] = p[2 * z]; qi[z] = p[2 * z + 1]; }
    }
}
static void store_vol(long L, long R, long SS, const double *sre, const double *sim, double *xout) {
  for (long x = 0; x < L; x++)
    for (long y = 0; y < L; y++) {
      double *p = xout + (x * L + y) * L * 2;
      const double *qr = sre + x * SS + y * R, *qi = sim + x * SS + y * R;
      for (long z = 0; z < L; z++) { p[2 * z] = qr[z]; p[2 * z + 1] = qi[z]; }
    }
}

'''
src = src.replace(old_conv, new_conv)

# buffer size in mp_init
src = src.replace('size_t one = 4096 * 64 * sizeof(double);   // 2 MiB, max volume component',
                  'size_t one = 64l * 4616 * sizeof(double);   // max volume component (L=64 padded)')
src = src.replace('char *blk = (char *)aligned_alloc(1 << 21, 4 * one);\n    memset(blk, 0, 4 * one);',
                  'size_t tot = ((4 * one + (1 << 21) - 1) >> 21) << 21;\n    char *blk = (char *)aligned_alloc(1 << 21, tot);\n    memset(blk, 0, tot);')

# RUN_CASE update
old_run = src[src.index('#define RUN_CASE'):src.index('void mp_run')]
new_run = '''#define RUN_CASE(LN, L_, R_, SS_)                                  \\
  case L_: {                                                         \\
    long n = (long)L_ * L_ * L_;                                     \\
    for (long b = 0; b < B; b++) {                                   \\
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

'''
src = src.replace(old_run, new_run)

old_switch = src[src.index('  switch (L) {'):src.index('    default: break;')]
new_switch = '''  switch (L) {
    RUN_CASE(6, 6, 8, 72)
    RUN_CASE(8, 8, 8, 72)
    RUN_CASE(13, 13, 16, 264)
    RUN_CASE(17, 17, 24, 584)
    RUN_CASE(23, 23, 24, 584)
    RUN_CASE(36, 36, 40, 1608)
    RUN_CASE(45, 45, 48, 2312)
    RUN_CASE(64, 64, 72, 4616)
'''
src = src.replace(old_switch, new_switch)
open('implementation.c','w').write(src)
print("done")
