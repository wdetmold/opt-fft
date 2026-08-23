src = open('implementation.c').read()
# remove unused parity instantiations (keep macro def harmless? remove all parity block)
start = src.index("// ---- parity engines:")
end = src.index("// ---------------------------------------------------------------- SoA batch-of-8 engine")
src = src[:start] + src[end:]
# remove RUN_VOL_P / RUN_CASE_P / RUN_CASE_P_SOA macros & T conversions? store_vol_T/load_vol_T now unused
start = src.index("#define RUN_VOL_P(LN, L_, R_, SS_)")
end = src.index("#define RUN_CASE(LN, L_, R_, SS_)")
src = src[:start] + src[end:]
start = src.index("static void load_vol_T(")
end = src.index("// ---------------------------------------------------------------- init")
src = src[:start] + src[end:]
# remove X2/C2 buffers (unused now)
src = src.replace("""static double *Xre, *Xim, *Cre, *Cim;   // planar volume buffers (max size)
static double *X2re, *X2im, *C2re, *C2im; // ping-pong state + transposed c (small sizes)""",
"""static double *Xre, *Xim, *Cre, *Cim;   // planar volume buffers (max size)""")
src = src.replace("""    size_t small = 24 * 584 * sizeof(double);   // largest parity-size plane (L=23)
    size_t tot = ((4 * one + 4 * small + (1 << 21) - 1) >> 21) << 21;
    char *blk = (char *)aligned_alloc(1 << 21, tot);
    memset(blk, 0, tot);
    Xre = (double *)blk; Xim = (double *)(blk + one);
    Cre = (double *)(blk + 2 * one); Cim = (double *)(blk + 3 * one);
    X2re = (double *)(blk + 4 * one); X2im = (double *)(blk + 4 * one + small);
    C2re = (double *)(blk + 4 * one + 2 * small); C2im = (double *)(blk + 4 * one + 3 * small);
  }""",
"""    size_t tot = ((4 * one + (1 << 21) - 1) >> 21) << 21;
    char *blk = (char *)aligned_alloc(1 << 21, tot);
    memset(blk, 0, tot);
    Xre = (double *)blk; Xim = (double *)(blk + one);
    Cre = (double *)(blk + 2 * one); Cim = (double *)(blk + 3 * one);
  }""")
open('implementation.c','w').write(src)
print("cleaned")
