src = open('implementation.c').read()

# helpers: per-slice conversions
src = src.replace("""// ---------------------------------------------------------------- init""",
"""static void load_slice(long L, long R, const double *xin, long x, double *dre, double *dim) {
  for (long y = 0; y < L; y++) {
    const double *p = xin + (x * L + y) * L * 2;
    double *qr = dre + y * R, *qi = dim + y * R;
    for (long z = 0; z < L; z++) { qr[z] = p[2 * z]; qi[z] = p[2 * z + 1]; }
  }
}
static void store_slice(long L, long R, const double *sre, const double *sim, double *xout, long x) {
  for (long y = 0; y < L; y++) {
    double *p = xout + (x * L + y) * L * 2;
    const double *qr = sre + y * R, *qi = sim + y * R;
    for (long z = 0; z < L; z++) { p[2 * z] = qr[z]; p[2 * z + 1] = qi[z]; }
  }
}

// ---------------------------------------------------------------- init""")

# allocation: 8 full planes + 2 staging planes
src = src.replace("""  if (!Xre) {
    size_t one = PLANE * sizeof(double);   // max volume component (L=64 padded)
    size_t tot = ((4 * one + (1 << 21) - 1) >> 21) << 21;
    char *blk = (char *)aligned_alloc(1 << 21, tot);
    memset(blk, 0, tot);
    Xre = (double *)blk; Xim = (double *)(blk + one);
    Cre = (double *)(blk + 2 * one); Cim = (double *)(blk + 3 * one);
  }""",
"""  if (!Xre) {
    size_t one = PLANE * sizeof(double);   // max volume component (L=64 padded)
    size_t stg = 8 * SS64 * sizeof(double);
    size_t tot = ((6 * one + 2 * stg + (1 << 21) - 1) >> 21) << 21;
    char *blk = (char *)aligned_alloc(1 << 21, tot);
    memset(blk, 0, tot);
    Xre = (double *)blk; Xim = (double *)(blk + one);
    Cre = (double *)(blk + 2 * one); Cim = (double *)(blk + 3 * one);
    Tre = (double *)(blk + 4 * one); Tim = (double *)(blk + 5 * one);
    STGre = (double *)(blk + 6 * one); STGim = (double *)(blk + 6 * one + stg);
  }""")
src = src.replace("static double *STGre, *STGim;   // staging: 8 slices",
"static double *STGre, *STGim;   // staging: 8 slices\nstatic double *Tre, *Tim;       // ping-pong partner of Xre/Xim for L=64")

# driver: replace RUN_CASE for 64 with fused pipeline
src = src.replace("""    RUN_CASE(64, 64, 72, 4616)
    default: break;""",
"""    case 64: {
      long n = 64L * 64 * 64;
      for (long b = 0; b < B; b++) {
        load_vol(64, R64, SS64, cin + b * 2 * n, Cre, Cim);
        double *Sr = Xre, *Si = Xim, *Dr = Tre, *Di = Tim;
        for (long g = 0; g < 8; g++) {          // bootstrap: phase1 of iter 1
          for (long j = 0; j < 8; j++)
            load_slice(64, R64, xin + b * 2 * n, g + 8 * j, STGre + j * SS64, STGim + j * SS64);
          phase1_64(g, Sr, Si);
        }
        for (long k = 1; k <= m; k++) {
          for (long g = 0; g < 8; g++) {
            phase2_64(g, Sr, Si);
            if (k == 1)
              for (long j = 0; j < 8; j++)
                store_slice(64, R64, STGre + j * SS64, STGim + j * SS64, out1 + b * 2 * n, 8 * j + g);
            if (k == m)
              for (long j = 0; j < 8; j++)
                store_slice(64, R64, STGre + j * SS64, STGim + j * SS64, outm + b * 2 * n, 8 * j + g);
            if (k < m) phase1_64(g, Dr, Di);
          }
          double *t;
          t = Sr; Sr = Dr; Dr = t;
          t = Si; Si = Di; Di = t;
        }
      }
    } break;
    default: break;""")
open('implementation.c','w').write(src)
print("ok")
