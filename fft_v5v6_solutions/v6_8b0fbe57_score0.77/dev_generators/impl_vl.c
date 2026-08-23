

#define MAPST_NR(pr, pi, mm, yr, yi, pcr, pci) do { \
    VD _zr = VADD(yr, LD(pcr)), _zi = VADD(yi, LD(pci)); \
    VD _s  = VFMA(_zi,_zi, VMUL(_zr,_zr)); \
    _s = VMAX(_s, Vtiny); \
    VD _t = _mm512_rsqrt14_pd(_s); \
    VD _hs = VMUL(Vhalf,_s); \
    _t = VMUL(_t, VFNMA(_hs, VMUL(_t,_t), V1p5)); \
    _t = VMUL(_t, VFNMA(_hs, VMUL(_t,_t), V1p5)); \
    VD _d = VFMA(_s,_t,Vone); \
    VD _q = _mm512_rcp14_pd(_d); \
    _q = VFMA(_q, VFNMA(_d,_q,Vone), _q); \
    _q = VFMA(_q, VFNMA(_d,_q,Vone), _q); \
    MST(pr, mm, VMUL(_zr,_q)); MST(pi, mm, VMUL(_zi,_q)); } while(0)
// ---------------------------------------------------------------------------
// Volume-lane (VL) path for L in {6, 8, 13}: 8 volumes in SIMD lanes.
// Layout: X[i*8 + lane], i = flat element index, lane = volume in group.
// All three axis passes use the same fold/PFA bodies with strides 8, 8L, 8L^2.
// ---------------------------------------------------------------------------
static void vl_import(double* restrict Xre, double* restrict Xim,
                      const double* restrict src, long n, int nl)
{
    // src: nl (<=8) volumes interleaved complex (vol-major), n elements each
    const __m512i idxe = _mm512_set_epi64(14,12,10,8,6,4,2,0);
    const __m512i idxo = _mm512_set_epi64(15,13,11,9,7,5,3,1);
    long i0 = 0;
    for (; i0 + 8 <= n; i0 += 8) {
        VD rr[8], ri[8], o0,o1,o2,o3,o4,o5,o6,o7;
        for (int l = nl; l < 8; l++) { rr[l] = ZERO; ri[l] = ZERO; }
        for (int l = 0; l < nl; l++) {
            VD v0 = LD(src + 2*((size_t)l*n + i0));
            VD v1 = LD(src + 2*((size_t)l*n + i0) + 8);
            rr[l] = _mm512_permutex2var_pd(v0, idxe, v1);
            ri[l] = _mm512_permutex2var_pd(v0, idxo, v1);
        }
        TR8(o0,o1,o2,o3,o4,o5,o6,o7, rr[0],rr[1],rr[2],rr[3],rr[4],rr[5],rr[6],rr[7]);
        ST(Xre+(i0+0)*8,o0); ST(Xre+(i0+1)*8,o1); ST(Xre+(i0+2)*8,o2); ST(Xre+(i0+3)*8,o3);
        ST(Xre+(i0+4)*8,o4); ST(Xre+(i0+5)*8,o5); ST(Xre+(i0+6)*8,o6); ST(Xre+(i0+7)*8,o7);
        TR8(o0,o1,o2,o3,o4,o5,o6,o7, ri[0],ri[1],ri[2],ri[3],ri[4],ri[5],ri[6],ri[7]);
        ST(Xim+(i0+0)*8,o0); ST(Xim+(i0+1)*8,o1); ST(Xim+(i0+2)*8,o2); ST(Xim+(i0+3)*8,o3);
        ST(Xim+(i0+4)*8,o4); ST(Xim+(i0+5)*8,o5); ST(Xim+(i0+6)*8,o6); ST(Xim+(i0+7)*8,o7);
    }
    for (; i0 < n; i0++)
        for (int l = 0; l < 8; l++) {
            Xre[i0*8 + l] = (l < nl) ? src[2*((size_t)l*n + i0)] : 0.0;
            Xim[i0*8 + l] = (l < nl) ? src[2*((size_t)l*n + i0) + 1] : 0.0;
        }
}
static void vl_export(const double* restrict Xre, const double* restrict Xim,
                      double* restrict dst, long n, int nl)
{
    const __m512i idxl = _mm512_set_epi64(11,3,10,2,9,1,8,0);
    const __m512i idxh = _mm512_set_epi64(15,7,14,6,13,5,12,4);
    long i0 = 0;
    for (; i0 + 8 <= n; i0 += 8) {
        VD o0,o1,o2,o3,o4,o5,o6,o7, p0,p1,p2,p3,p4,p5,p6,p7;
        TR8(o0,o1,o2,o3,o4,o5,o6,o7,
            LD(Xre+(i0+0)*8),LD(Xre+(i0+1)*8),LD(Xre+(i0+2)*8),LD(Xre+(i0+3)*8),
            LD(Xre+(i0+4)*8),LD(Xre+(i0+5)*8),LD(Xre+(i0+6)*8),LD(Xre+(i0+7)*8));
        TR8(p0,p1,p2,p3,p4,p5,p6,p7,
            LD(Xim+(i0+0)*8),LD(Xim+(i0+1)*8),LD(Xim+(i0+2)*8),LD(Xim+(i0+3)*8),
            LD(Xim+(i0+4)*8),LD(Xim+(i0+5)*8),LD(Xim+(i0+6)*8),LD(Xim+(i0+7)*8));
#define EXP1(l, vr, vi) do { \
        ST(dst + 2*((size_t)(l)*n + i0),     _mm512_permutex2var_pd(vr, idxl, vi)); \
        ST(dst + 2*((size_t)(l)*n + i0) + 8, _mm512_permutex2var_pd(vr, idxh, vi)); } while(0)
        VD vr_[8] = {o0,o1,o2,o3,o4,o5,o6,o7}, vi_[8] = {p0,p1,p2,p3,p4,p5,p6,p7};
        for (int l = 0; l < nl; l++) EXP1(l, vr_[l], vi_[l]);
#undef EXP1
    }
    for (; i0 < n; i0++)
        for (int l = 0; l < nl; l++) {
            dst[2*((size_t)l*n + i0)]     = Xre[i0*8 + l];
            dst[2*((size_t)l*n + i0) + 1] = Xim[i0*8 + l];
        }
}

static void vlz6(double* restrict re, double* restrict im)
{
#define LR_(j) LD(re + 8*(j))
#define LI_(j) LD(im + 8*(j))
#define SP_(k,vr,vi) do { ST(re + 8*(k), vr); ST(im + 8*(k), vi); } while(0)
    KM6_BODY(LR_, LI_, SP_);
#undef LR_
#undef LI_
#undef SP_
}
static void vly6(double* restrict re, double* restrict im)
{
#define LR_(j) LD(re + 48*(j))
#define LI_(j) LD(im + 48*(j))
#define SP_(k,vr,vi) do { ST(re + 48*(k), vr); ST(im + 48*(k), vi); } while(0)
    KM6_BODY(LR_, LI_, SP_);
#undef LR_
#undef LI_
#undef SP_
}
static void vlx6_map(double* restrict re, double* restrict im,
                       const double* restrict cre, const double* restrict cim)
{
    const VD Vone=BC(1.0), V1p5=BC(1.5), Vtiny=BC(2.3e-308);
    const MK mf = 0xFF;
#define LR_(j) LD(re + 288*(j))
#define LI_(j) LD(im + 288*(j))
#define SP_(k,vr,vi) MAPST(re + 288*(k), im + 288*(k), mf, vr, vi, cre + 288*(k), cim + 288*(k))
    KM6_BODY(LR_, LI_, SP_);
#undef LR_
#undef LI_
#undef SP_
}

static void vlz8(double* restrict re, double* restrict im)
{
#define LR_(j) LD(re + 8*(j))
#define LI_(j) LD(im + 8*(j))
#define SP_(k,vr,vi) do { ST(re + 8*(k), vr); ST(im + 8*(k), vi); } while(0)
    KM8_BODY(LR_, LI_, SP_);
#undef LR_
#undef LI_
#undef SP_
}
static void vly8(double* restrict re, double* restrict im)
{
#define LR_(j) LD(re + 64*(j))
#define LI_(j) LD(im + 64*(j))
#define SP_(k,vr,vi) do { ST(re + 64*(k), vr); ST(im + 64*(k), vi); } while(0)
    KM8_BODY(LR_, LI_, SP_);
#undef LR_
#undef LI_
#undef SP_
}
static void vlx8_map(double* restrict re, double* restrict im,
                       const double* restrict cre, const double* restrict cim)
{
    const VD Vone=BC(1.0), V1p5=BC(1.5), Vtiny=BC(2.3e-308), Vhalf=BC(0.5);
    const MK mf = 0xFF;
#define LR_(j) LD(re + 512*(j))
#define LI_(j) LD(im + 512*(j))
#define SP_(k,vr,vi) MAPST(re + 512*(k), im + 512*(k), mf, vr, vi, cre + 512*(k), cim + 512*(k))
    KM8_BODY(LR_, LI_, SP_);
#undef LR_
#undef LI_
#undef SP_
}

static void vlz13(double* restrict re, double* restrict im)
{
#define LR_(j) LD(re + 8*(j))
#define LI_(j) LD(im + 8*(j))
#define SP_(k,vr,vi) do { ST(re + 8*(k), vr); ST(im + 8*(k), vi); } while(0)
    KMFOLD_BODY(13, 6, g_C13, g_S13, LR_, LI_, SP_);
#undef LR_
#undef LI_
#undef SP_
}
static void vly13(double* restrict re, double* restrict im)
{
#define LR_(j) LD(re + 104*(j))
#define LI_(j) LD(im + 104*(j))
#define SP_(k,vr,vi) do { ST(re + 104*(k), vr); ST(im + 104*(k), vi); } while(0)
    KMFOLD_BODY(13, 6, g_C13, g_S13, LR_, LI_, SP_);
#undef LR_
#undef LI_
#undef SP_
}
static void vlx13_map(double* restrict re, double* restrict im,
                      const double* restrict cre, const double* restrict cim)
{
    const VD Vone=BC(1.0), V1p5=BC(1.5), Vtiny=BC(2.3e-308), Vhalf=BC(0.5);
    const MK mf = 0xFF;
#define LR_(j) LD(re + 1352*(j))
#define LI_(j) LD(im + 1352*(j))
#define SP_(k,vr,vi) MAPST(re + 1352*(k), im + 1352*(k), mf, vr, vi, cre + 1352*(k), cim + 1352*(k))
    KMFOLD_BODY(13, 6, g_C13, g_S13, LR_, LI_, SP_);
#undef LR_
#undef LI_
#undef SP_
}

static void vliter6(double* restrict Xr, double* restrict Xi,
                    const double* restrict Cr, const double* restrict Ci)
{
    for (int x = 0; x < 6; x++) {
        for (int y = 0; y < 6; y++)
            vlz6(Xr + (x*36 + y*6)*8, Xi + (x*36 + y*6)*8);
        for (int z = 0; z < 6; z++)
            vly6(Xr + (x*36 + z)*8, Xi + (x*36 + z)*8);
    }
    for (int p = 0; p < 36; p++)
        vlx6_map(Xr + 8*p, Xi + 8*p, Cr + 8*p, Ci + 8*p);
}
static void vliter8(double* restrict Xr, double* restrict Xi,
                    const double* restrict Cr, const double* restrict Ci)
{
    for (int x = 0; x < 8; x++) {
        for (int y = 0; y < 8; y++)
            vlz8(Xr + (x*64 + y*8)*8, Xi + (x*64 + y*8)*8);
        for (int z = 0; z < 8; z++)
            vly8(Xr + (x*64 + z)*8, Xi + (x*64 + z)*8);
    }
    for (int p = 0; p < 64; p++)
        vlx8_map(Xr + 8*p, Xi + 8*p, Cr + 8*p, Ci + 8*p);
}
static void vliter13(double* restrict Xr, double* restrict Xi,
                     const double* restrict Cr, const double* restrict Ci)
{
    for (int x = 0; x < 13; x++) {
        for (int y = 0; y < 13; y++)
            vlz13(Xr + (x*169 + y*13)*8, Xi + (x*169 + y*13)*8);
        for (int z = 0; z < 13; z++)
            vly13(Xr + (x*169 + z)*8, Xi + (x*169 + z)*8);
    }
    for (int p = 0; p < 169; p++)
        vlx13_map(Xr + 8*p, Xi + 8*p, Cr + 8*p, Ci + 8*p);
}

static void vlz17(double* restrict re, double* restrict im)
{
#define LR_(j) LD(re + 8*(j))
#define LI_(j) LD(im + 8*(j))
#define SP_(k,vr,vi) do { ST(re + 8*(k), vr); ST(im + 8*(k), vi); } while(0)
    KMFOLD_BODY(17, 8, g_C17, g_S17, LR_, LI_, SP_);
#undef LR_
#undef LI_
#undef SP_
}
static void vly17(double* restrict re, double* restrict im)
{
#define LR_(j) LD(re + 136*(j))
#define LI_(j) LD(im + 136*(j))
#define SP_(k,vr,vi) do { ST(re + 136*(k), vr); ST(im + 136*(k), vi); } while(0)
    KMFOLD_BODY(17, 8, g_C17, g_S17, LR_, LI_, SP_);
#undef LR_
#undef LI_
#undef SP_
}
static void vlx17_map(double* restrict re, double* restrict im,
                       const double* restrict cre, const double* restrict cim)
{
    const VD Vone=BC(1.0), V1p5=BC(1.5), Vtiny=BC(2.3e-308), Vhalf=BC(0.5);
    const MK mf = 0xFF;
#define LR_(j) LD(re + 2312*(j))
#define LI_(j) LD(im + 2312*(j))
#define SP_(k,vr,vi) MAPST(re + 2312*(k), im + 2312*(k), mf, vr, vi, cre + 2312*(k), cim + 2312*(k))
    KMFOLD_BODY(17, 8, g_C17, g_S17, LR_, LI_, SP_);
#undef LR_
#undef LI_
#undef SP_
}
static void vliter17(double* restrict Xr, double* restrict Xi,
                      const double* restrict Cr, const double* restrict Ci)
{
    for (int x = 0; x < 17; x++) {
        for (int y = 0; y < 17; y++)
            vlz17(Xr + (x*289 + y*17)*8, Xi + (x*289 + y*17)*8);
        for (int z = 0; z < 17; z++)
            vly17(Xr + (x*289 + z)*8, Xi + (x*289 + z)*8);
    }
    for (int p = 0; p < 289; p++)
        vlx17_map(Xr + 8*p, Xi + 8*p, Cr + 8*p, Ci + 8*p);
}

static void vlz23(double* restrict re, double* restrict im)
{
#define LR_(j) LD(re + 8*(j))
#define LI_(j) LD(im + 8*(j))
#define SP_(k,vr,vi) do { ST(re + 8*(k), vr); ST(im + 8*(k), vi); } while(0)
    KMFOLD_BODY(23, 11, g_C23, g_S23, LR_, LI_, SP_);
#undef LR_
#undef LI_
#undef SP_
}
static void vly23(double* restrict re, double* restrict im)
{
#define LR_(j) LD(re + 184*(j))
#define LI_(j) LD(im + 184*(j))
#define SP_(k,vr,vi) do { ST(re + 184*(k), vr); ST(im + 184*(k), vi); } while(0)
    KMFOLD_BODY(23, 11, g_C23, g_S23, LR_, LI_, SP_);
#undef LR_
#undef LI_
#undef SP_
}
static void vlx23_map(double* restrict re, double* restrict im,
                       const double* restrict cre, const double* restrict cim)
{
    const VD Vone=BC(1.0), V1p5=BC(1.5), Vtiny=BC(2.3e-308), Vhalf=BC(0.5);
    const MK mf = 0xFF;
#define LR_(j) LD(re + 4232*(j))
#define LI_(j) LD(im + 4232*(j))
#define SP_(k,vr,vi) MAPST(re + 4232*(k), im + 4232*(k), mf, vr, vi, cre + 4232*(k), cim + 4232*(k))
    KMFOLD_BODY(23, 11, g_C23, g_S23, LR_, LI_, SP_);
#undef LR_
#undef LI_
#undef SP_
}
static void vliter23(double* restrict Xr, double* restrict Xi,
                      const double* restrict Cr, const double* restrict Ci)
{
    for (int x = 0; x < 23; x++) {
        for (int y = 0; y < 23; y++)
            vlz23(Xr + (x*529 + y*23)*8, Xi + (x*529 + y*23)*8);
        for (int z = 0; z < 23; z++)
            vly23(Xr + (x*529 + z)*8, Xi + (x*529 + z)*8);
    }
    for (int p = 0; p < 529; p++)
        vlx23_map(Xr + 8*p, Xi + 8*p, Cr + 8*p, Ci + 8*p);
}
// run a group of 8 volumes; x0g/cg point at the first of the 8 volumes
static void vl_chain(int L, long m, const double* x0g, const double* cg,
                     double* out1g, double* outmg, int nl)
{
    long n = (long)L*L*L;
    size_t bytes = (((size_t)n*8*2 + 64) * sizeof(double) + 63) & ~(size_t)63;
    static double *VX = 0, *VC = 0; static size_t vcap = 0;
    if (bytes > vcap) {
        if (VX) { free(VX); free(VC); }
        VX = (double*)aligned_alloc(64, bytes);
        VC = (double*)aligned_alloc(64, bytes);
        vcap = bytes;
    }
    double *Xr = VX, *Xi = VX + n*8, *Cr = VC, *Ci = VC + n*8;
    vl_import(Xr, Xi, x0g, n, nl);
    vl_import(Cr, Ci, cg, n, nl);
    void (*vit)(double*, double*, const double*, const double*) =
        (L == 6) ? vliter6 : (L == 8) ? vliter8 : (L == 13) ? vliter13 :
        (L == 17) ? vliter17 : vliter23;
    for (long it = 0; it < m; it++) {
        vit(Xr, Xi, Cr, Ci);
        if (it == 0) vl_export(Xr, Xi, out1g, n, nl);
    }
    vl_export(Xr, Xi, outmg, n, nl);
}
