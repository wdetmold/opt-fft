
// ---- in-place x+map variants ----
static void km13_x_map_ip(double* restrict re, double* restrict im,
                       const double* restrict cre, const double* restrict cim, MK m)
{
    const VD Vone=BC(1.0), V1p5=BC(1.5), Vtiny=BC(2.3e-308), Vhalf=BC(0.5);
#define LR_(j) LD(re + 169*(j))
#define LI_(j) LD(im + 169*(j))
#define SP_(k,vr,vi) MAPST(re + 169*(k), im + 169*(k), m, vr, vi, cre + 169*(k), cim + 169*(k))
    KMFOLD_BODY(13, 6, g_C13, g_S13, LR_, LI_, SP_);
#undef LR_
#undef LI_
#undef SP_
}
static void km17_x_map_ip(double* restrict re, double* restrict im,
                       const double* restrict cre, const double* restrict cim, MK m)
{
    const VD Vone=BC(1.0), V1p5=BC(1.5), Vtiny=BC(2.3e-308), Vhalf=BC(0.5);
#define LR_(j) LD(re + 289*(j))
#define LI_(j) LD(im + 289*(j))
#define SP_(k,vr,vi) MAPST(re + 289*(k), im + 289*(k), m, vr, vi, cre + 289*(k), cim + 289*(k))
    KMFOLD_BODY(17, 8, g_C17, g_S17, LR_, LI_, SP_);
#undef LR_
#undef LI_
#undef SP_
}
static void km23_x_map_ip(double* restrict re, double* restrict im,
                       const double* restrict cre, const double* restrict cim, MK m)
{
    const VD Vone=BC(1.0), V1p5=BC(1.5), Vtiny=BC(2.3e-308), Vhalf=BC(0.5);
#define LR_(j) LD(re + 529*(j))
#define LI_(j) LD(im + 529*(j))
#define SP_(k,vr,vi) MAPST(re + 529*(k), im + 529*(k), m, vr, vi, cre + 529*(k), cim + 529*(k))
    KMFOLD_BODY(23, 11, g_C23, g_S23, LR_, LI_, SP_);
#undef LR_
#undef LI_
#undef SP_
}
static void km36_x_map_ip(double* restrict re, double* restrict im,
                       const double* restrict cre, const double* restrict cim)
{
    const VD Vone=BC(1.0), V1p5=BC(1.5), Vtiny=BC(2.3e-308);
    const MK m = 0xFF;
    _Pragma("GCC unroll 36")
    for (int pj = 0; pj < 36; pj++) {
        _mm_prefetch((const char*)(re + 1296*pj + 8), _MM_HINT_T0);
        _mm_prefetch((const char*)(im + 1296*pj + 8), _MM_HINT_T0);
        _mm_prefetch((const char*)(cre + 1296*pj + 8), _MM_HINT_T0);
        _mm_prefetch((const char*)(cim + 1296*pj + 8), _MM_HINT_T0);
    }
#define LR_(j) LD(re + (j))
#define LI_(j) LD(im + (j))
#define SP_(k,vr,vi) MAPST(re + (k), im + (k), m, vr, vi, cre + (k), cim + (k))
    KM36_BODY(LR_, LI_, SP_, P36IN_S1296, P36OUT_S1296);
#undef LR_
#undef LI_
#undef SP_
}
static void km45_x_map_ip(double* restrict re, double* restrict im,
                       const double* restrict cre, const double* restrict cim, MK m)
{
    const VD Vone=BC(1.0), V1p5=BC(1.5), Vtiny=BC(2.3e-308);
    _Pragma("GCC unroll 45")
    for (int pj = 0; pj < 45; pj++) {
        _mm_prefetch((const char*)(re + 2025*pj + 8), _MM_HINT_T0);
        _mm_prefetch((const char*)(im + 2025*pj + 8), _MM_HINT_T0);
        _mm_prefetch((const char*)(cre + 2025*pj + 8), _MM_HINT_T0);
        _mm_prefetch((const char*)(cim + 2025*pj + 8), _MM_HINT_T0);
    }
#define LR_(j) LD(re + (j))
#define LI_(j) LD(im + (j))
#define SP_(k,vr,vi) MAPST(re + (k), im + (k), m, vr, vi, cre + (k), cim + (k))
    KM45_BODY(LR_, LI_, SP_, P45IN_S2025, P45OUT_S2025);
#undef LR_
#undef LI_
#undef SP_
}
