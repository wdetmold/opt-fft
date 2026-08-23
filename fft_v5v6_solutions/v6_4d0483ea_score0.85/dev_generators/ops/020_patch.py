src = open('implementation.c').read()
# revert driver for 64 to standard RUN_CASE; keep fused code (unused) out entirely
start = src.index('// ------------------------------------------------- fused L=64 pipeline')
end = src.index('// ---------------------------------------------------------------- conversions')
src = src[:start] + src[end:]
# remove fused case and restore RUN_CASE(64)
fstart = src.index("""    case 64: {
      long n = 64L * 64 * 64;""")
fend = src.index("""    default: break;""")
src = src[:fstart] + "    RUN_CASE(64, 64, 72, 4616)\n" + src[fend:]
# remove staging/T allocation extras
src = src.replace("""  if (!Xre) {
    size_t one = PLANE * sizeof(double);   // max volume component (L=64 padded)
    size_t stg = 8 * SS64 * sizeof(double);
    size_t tot = ((6 * one + 2 * stg + (1 << 21) - 1) >> 21) << 21;
    char *blk = (char *)aligned_alloc(1 << 21, tot);
    memset(blk, 0, tot);
    Xre = (double *)blk; Xim = (double *)(blk + one);
    Cre = (double *)(blk + 2 * one); Cim = (double *)(blk + 3 * one);
    Tre = (double *)(blk + 4 * one); Tim = (double *)(blk + 5 * one);
    STGre = (double *)(blk + 6 * one); STGim = (double *)(blk + 6 * one + stg);
  }""",
"""  if (!Xre) {
    size_t one = PLANE * sizeof(double);   // max volume component (L=64 padded)
    size_t tot = ((4 * one + (1 << 21) - 1) >> 21) << 21;
    char *blk = (char *)aligned_alloc(1 << 21, tot);
    memset(blk, 0, tot);
    Xre = (double *)blk; Xim = (double *)(blk + one);
    Cre = (double *)(blk + 2 * one); Cim = (double *)(blk + 3 * one);
  }""")
open('implementation.c','w').write(src)
print("ok")
