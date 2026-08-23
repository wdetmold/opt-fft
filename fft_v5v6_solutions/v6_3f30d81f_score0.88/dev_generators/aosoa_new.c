// AoSoA pipeline: sweepY, then sweepX (fused map) per y-row with the next
// iteration's z-pass pipelined behind each completed row (skipped on snapshot
// iterations, where a standalone z sweep restarts the pipeline).
// XS = padded x-slab stride in slots (breaks 4K aliasing for L=8).
#define GEN_RUN_AOSOA(LL, XS, KP, KM)                                           \
static void sweepZ_A##LL(vd *restrict X) {                                      \
    for (int x = 0; x < LL; x++)                                                \
        for (int y = 0; y < LL; y++) {                                          \
            vd *p = X + 2*((ptrdiff_t)x*(XS) + y*LL);                           \
            KP(p, p + 1, 2);                                                    \
        }                                                                       \
}                                                                               \
static void sweepY_A##LL(vd *restrict X) {                                      \
    for (int x = 0; x < LL; x++)                                                \
        for (int z = 0; z < LL; z++) {                                          \
            vd *p = X + 2*((ptrdiff_t)x*(XS) + z);                              \
            KP(p, p + 1, 2*LL);                                                 \
        }                                                                       \
}                                                                               \
static void sweepX_A##LL(vd *restrict X, const vd *restrict C, int fuse_z) {    \
    for (int y = 0; y < LL; y++) {                                              \
        for (int z = 0; z < LL; z++) {                                          \
            ptrdiff_t q = (ptrdiff_t)y*LL + z;                                  \
            KM(X + 2*q, X + 2*q + 1, 2*(XS), C + 2*q, C + 2*q + 1);             \
        }                                                                       \
        if (fuse_z)                                                             \
            for (int x = 0; x < LL; x++) {                                      \
                vd *p = X + 2*((ptrdiff_t)x*(XS) + y*LL);                       \
                KP(p, p + 1, 2);                                                \
            }                                                                   \
    }                                                                           \
}                                                                               \
void run##LL(long long Bll, long long mll, const double *x0, const double *c0, \
             double *out1, double *outm) {                                      \
    const long N = (long)LL*LL*LL; const long B = (long)Bll;                    \
    long m = (long)mll; if (m < 1) m = 1;                                       \
    for (long b0 = 0; b0 < B; b0 += VL) {                                       \
        long nl = B - b0 < VL ? B - b0 : VL;                                    \
        for (int x = 0; x < LL; x++) {                                          \
            conv_in_aosoaC(x0 + (long long)b0*N*2 + (long)x*LL*LL*2, nl, N,     \
                           (long)LL*LL, XP_ + 2*(ptrdiff_t)x*(XS));             \
            conv_in_aosoaC(c0 + (long long)b0*N*2 + (long)x*LL*LL*2, nl, N,     \
                           (long)LL*LL, CP_ + 2*(ptrdiff_t)x*(XS));             \
        }                                                                       \
        sweepZ_A##LL(XP_);                                                      \
        for (long it = 1; it <= m; it++) {                                      \
            int snap = (it == 1) || (it == m);                                  \
            sweepY_A##LL(XP_);                                                  \
            sweepX_A##LL(XP_, CP_, !snap);                                      \
            if (snap) {                                                         \
                for (int x = 0; x < LL; x++) {                                  \
                    if (it == 1)                                                \
                        conv_out_aosoaC(XP_ + 2*(ptrdiff_t)x*(XS),              \
                                        out1 + (long long)b0*N*2 + (long)x*LL*LL*2, \
                                        nl, N, (long)LL*LL);                    \
                    if (it == m)                                                \
                        conv_out_aosoaC(XP_ + 2*(ptrdiff_t)x*(XS),              \
                                        outm + (long long)b0*N*2 + (long)x*LL*LL*2, \
                                        nl, N, (long)LL*LL);                    \
                }                                                               \
                if (it < m) sweepZ_A##LL(XP_);                                  \
            }                                                                   \
        }                                                                       \
    }                                                                           \
}
GEN_RUN_AOSOA(6,  36,  k6_p,  k6_m)
GEN_RUN_AOSOA(8,  68,  k8_p,  k8_m)
GEN_RUN_AOSOA(13, 169, k13_p, k13_m)
GEN_RUN_AOSOA(17, 289, k17_p, k17_m)
