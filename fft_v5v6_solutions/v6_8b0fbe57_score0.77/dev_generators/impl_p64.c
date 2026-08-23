#pragma GCC push_options
#pragma GCC optimize("O3")
// ---------------------------------------------------------------------------
// L=64 fused pipeline with copy/compute decoupling.
//   x-DFT split as DFT8(a) [stageA w/ twiddle] and DFT8(b) [stageB], x = 8a+b.
//   Per t-step: T(b,t) --copy--> SBT; DFT8_b: SBT->SB; y,z DFTs on SB;
//   c --copy--> SBT; map+DFT8_a+twiddle: SB(+c) -> SB; SB --copy--> T'(t,*).
// ---------------------------------------------------------------------------
#define SBSTRIDE 4104
static double T64re[2*262144 + 16] __attribute__((aligned(64)));
#define T64im (T64re + 262144)
static double SBre[8*SBSTRIDE + 8] __attribute__((aligned(64)));
static double SBim[8*SBSTRIDE + 8] __attribute__((aligned(64)));
static double STre[8*SBSTRIDE + 8] __attribute__((aligned(64)));
static double STim[8*SBSTRIDE + 8] __attribute__((aligned(64)));

// pure streaming copy: n doubles
static inline void pcopy(double* restrict dst, const double* restrict src, long n)
{
    for (long i = 0; i < n; i += 32) {
        VD a = LD(src+i), b = LD(src+i+8), c = LD(src+i+16), d = LD(src+i+24);
        ST(dst+i, a); ST(dst+i+8, b); ST(dst+i+16, c); ST(dst+i+24, d);
    }
}

// stageA from state (prime): read state slabs x=8a+b, DFT8 over a, twiddle -> T
static void stageA_prime64(const double* restrict re, const double* restrict im,
                           double* restrict Tre, double* restrict Tim, int b)
{
    const VD Vr2=BC(g_r2half);
    // stage state slabs into SBT first (pure copy)
    for (int a = 0; a < 8; a++) {
        pcopy(STre + a*SBSTRIDE, re + ((size_t)(8*a+b))*4096, 4096);
        pcopy(STim + a*SBSTRIDE, im + ((size_t)(8*a+b))*4096, 4096);
    }
    double *tr = Tre + (size_t)b*8*4096, *ti = Tim + (size_t)b*8*4096;
    for (long q = 0; q < 4096; q += 8) {
        VD x0r=LD(STre+0*SBSTRIDE+q), x0i=LD(STim+0*SBSTRIDE+q);
        VD x1r=LD(STre+1*SBSTRIDE+q), x1i=LD(STim+1*SBSTRIDE+q);
        VD x2r=LD(STre+2*SBSTRIDE+q), x2i=LD(STim+2*SBSTRIDE+q);
        VD x3r=LD(STre+3*SBSTRIDE+q), x3i=LD(STim+3*SBSTRIDE+q);
        VD x4r=LD(STre+4*SBSTRIDE+q), x4i=LD(STim+4*SBSTRIDE+q);
        VD x5r=LD(STre+5*SBSTRIDE+q), x5i=LD(STim+5*SBSTRIDE+q);
        VD x6r=LD(STre+6*SBSTRIDE+q), x6i=LD(STim+6*SBSTRIDE+q);
        VD x7r=LD(STre+7*SBSTRIDE+q), x7i=LD(STim+7*SBSTRIDE+q);
        VD F0r,F0i,F1r,F1i,F2r,F2i,F3r,F3i,F4r,F4i,F5r,F5i,F6r,F6i,F7r,F7i;
        DFT8V(F0r,F0i,F1r,F1i,F2r,F2i,F3r,F3i,F4r,F4i,F5r,F5i,F6r,F6i,F7r,F7i,
              x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,x4r,x4i,x5r,x5i,x6r,x6i,x7r,x7i);
        ST(tr+q, F0r); ST(ti+q, F0i);
        if (b) {
            VD gr,gi;
            CMULV(gr,gi,F1r,F1i,BC(g_T64r[b][1]),BC(g_T64i[b][1])); ST(tr+1*4096+q,gr); ST(ti+1*4096+q,gi);
            CMULV(gr,gi,F2r,F2i,BC(g_T64r[b][2]),BC(g_T64i[b][2])); ST(tr+2*4096+q,gr); ST(ti+2*4096+q,gi);
            CMULV(gr,gi,F3r,F3i,BC(g_T64r[b][3]),BC(g_T64i[b][3])); ST(tr+3*4096+q,gr); ST(ti+3*4096+q,gi);
            CMULV(gr,gi,F4r,F4i,BC(g_T64r[b][4]),BC(g_T64i[b][4])); ST(tr+4*4096+q,gr); ST(ti+4*4096+q,gi);
            CMULV(gr,gi,F5r,F5i,BC(g_T64r[b][5]),BC(g_T64i[b][5])); ST(tr+5*4096+q,gr); ST(ti+5*4096+q,gi);
            CMULV(gr,gi,F6r,F6i,BC(g_T64r[b][6]),BC(g_T64i[b][6])); ST(tr+6*4096+q,gr); ST(ti+6*4096+q,gi);
            CMULV(gr,gi,F7r,F7i,BC(g_T64r[b][7]),BC(g_T64i[b][7])); ST(tr+7*4096+q,gr); ST(ti+7*4096+q,gi);
        } else {
            ST(tr+1*4096+q,F1r); ST(ti+1*4096+q,F1i);
            ST(tr+2*4096+q,F2r); ST(ti+2*4096+q,F2i);
            ST(tr+3*4096+q,F3r); ST(ti+3*4096+q,F3i);
            ST(tr+4*4096+q,F4r); ST(ti+4*4096+q,F4i);
            ST(tr+5*4096+q,F5r); ST(ti+5*4096+q,F5i);
            ST(tr+6*4096+q,F6r); ST(ti+6*4096+q,F6i);
            ST(tr+7*4096+q,F7r); ST(ti+7*4096+q,F7i);
        }
    }
}

// stageB(t): T col -> DFT8 over b -> SB slabs; prefetches c slabs for stageA.
static void stageB64_compute(const double* restrict Tre, const double* restrict Tim,
                             int t, int par,
                             const double* restrict cre, const double* restrict cim)
{
    const VD Vr2=BC(g_r2half);
    // slot(b,t): even: b*8+t ; odd: t*8+b
    const double *p0r, *p0i; long step;
    if (!par) { p0r = Tre + (size_t)t*4096; step = 8*4096; }
    else      { p0r = Tre + (size_t)t*8*4096; step = 4096; }
    p0i = Tim + (p0r - Tre);
    for (long q = 0; q < 4096; q += 8) {
        _Pragma("GCC unroll 8")
        for (int pb = 0; pb < 8; pb++) {
            _mm_prefetch((const char*)(p0r + pb*step + q + 64), _MM_HINT_T0);
            _mm_prefetch((const char*)(p0i + pb*step + q + 64), _MM_HINT_T0);
        }
        VD x0r=LD(p0r+0*step+q), x0i=LD(p0i+0*step+q);
        VD x1r=LD(p0r+1*step+q), x1i=LD(p0i+1*step+q);
        VD x2r=LD(p0r+2*step+q), x2i=LD(p0i+2*step+q);
        VD x3r=LD(p0r+3*step+q), x3i=LD(p0i+3*step+q);
        VD x4r=LD(p0r+4*step+q), x4i=LD(p0i+4*step+q);
        VD x5r=LD(p0r+5*step+q), x5i=LD(p0i+5*step+q);
        VD x6r=LD(p0r+6*step+q), x6i=LD(p0i+6*step+q);
        VD x7r=LD(p0r+7*step+q), x7i=LD(p0i+7*step+q);
        VD o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i,o4r,o4i,o5r,o5i,o6r,o6i,o7r,o7i;
        DFT8V(o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i,o4r,o4i,o5r,o5i,o6r,o6i,o7r,o7i,
              x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,x4r,x4i,x5r,x5i,x6r,x6i,x7r,x7i);
        ST(SBre+0*SBSTRIDE+q,o0r); ST(SBim+0*SBSTRIDE+q,o0i);
        ST(SBre+1*SBSTRIDE+q,o1r); ST(SBim+1*SBSTRIDE+q,o1i);
        ST(SBre+2*SBSTRIDE+q,o2r); ST(SBim+2*SBSTRIDE+q,o2i);
        ST(SBre+3*SBSTRIDE+q,o3r); ST(SBim+3*SBSTRIDE+q,o3i);
        ST(SBre+4*SBSTRIDE+q,o4r); ST(SBim+4*SBSTRIDE+q,o4i);
        ST(SBre+5*SBSTRIDE+q,o5r); ST(SBim+5*SBSTRIDE+q,o5i);
        ST(SBre+6*SBSTRIDE+q,o6r); ST(SBim+6*SBSTRIDE+q,o6i);
        ST(SBre+7*SBSTRIDE+q,o7r); ST(SBim+7*SBSTRIDE+q,o7i);
    }
}

// stageA-next: SB + c -> map -> (optional state out, NT) -> DFT8 over a + twiddle
// -> write into T slots for next parity (the slots stageB(t) just freed).
static void stageA64_compute(double* restrict Tre, double* restrict Tim,
                             int t, int par, int noT,
                             const double* restrict cre, const double* restrict cim,
                             double* restrict outre, double* restrict outim)
{
    const VD Vr2=BC(g_r2half);
    const VD Vone=BC(1.0), V1p5=BC(1.5), Vtiny=BC(2.3e-308), Vhalf=BC(0.5);
    double *our = outre ? outre + (size_t)t*4096 : 0, *oui = outim ? outim + (size_t)t*4096 : 0;
    const double *crs = cre + (size_t)t*4096, *cis = cim + (size_t)t*4096;
    // next parity slot(t, tp): nextpar = !par: if nextpar odd -> tp*8+t ; even -> t*8+tp
    double *t0r, *t0i; long tstep;
    if (!par) { t0r = Tre + (size_t)t*4096;   tstep = 8*4096; }  // next parity = odd: slot = tp*8+t
    else      { t0r = Tre + (size_t)t*8*4096; tstep = 4096;  }   // next parity = even: slot = t*8+tp
    t0i = Tim + (t0r - Tre);
    for (long q = 0; q < 4096; q += 8) {
        _Pragma("GCC unroll 8")
        for (int pa = 0; pa < 8; pa++) {
            _mm_prefetch((const char*)(crs + pa*8*4096 + q + 128), _MM_HINT_T0);
            _mm_prefetch((const char*)(cis + pa*8*4096 + q + 128), _MM_HINT_T0);
        }
        VD x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,x4r,x4i,x5r,x5i,x6r,x6i,x7r,x7i;
#define MAPLD(dstr, dsti, a) do { \
        VD _yr = LD(SBre+(a)*SBSTRIDE+q), _yi = LD(SBim+(a)*SBSTRIDE+q); \
        VD _zr = VADD(_yr, LD(crs+(a)*8*4096+q)), _zi = VADD(_yi, LD(cis+(a)*8*4096+q)); \
        VD _s  = VFMA(_zi,_zi, VMUL(_zr,_zr)); \
        _s = VMAX(_s, Vtiny); \
        VD _t = _mm512_rsqrt14_pd(_s); \
        VD _hs = VMUL(Vhalf,_s); \
        _t = VMUL(_t, VFNMA(_hs, VMUL(_t,_t), V1p5)); \
        _t = VMUL(_t, VFNMA(_hs, VMUL(_t,_t), V1p5)); \
        VD _d = VFMA(_s,_t,Vone); \
        VD _q = _mm512_div_pd(Vone, _d); \
        dstr = VMUL(_zr,_q); dsti = VMUL(_zi,_q); \
    } while(0)
        MAPLD(x0r,x0i,0); MAPLD(x1r,x1i,1); MAPLD(x2r,x2i,2); MAPLD(x3r,x3i,3);
        MAPLD(x4r,x4i,4); MAPLD(x5r,x5i,5); MAPLD(x6r,x6i,6); MAPLD(x7r,x7i,7);
#undef MAPLD
        if (our) {
            _mm512_stream_pd(our+0*8*4096+q, x0r); _mm512_stream_pd(oui+0*8*4096+q, x0i);
            _mm512_stream_pd(our+1*8*4096+q, x1r); _mm512_stream_pd(oui+1*8*4096+q, x1i);
            _mm512_stream_pd(our+2*8*4096+q, x2r); _mm512_stream_pd(oui+2*8*4096+q, x2i);
            _mm512_stream_pd(our+3*8*4096+q, x3r); _mm512_stream_pd(oui+3*8*4096+q, x3i);
            _mm512_stream_pd(our+4*8*4096+q, x4r); _mm512_stream_pd(oui+4*8*4096+q, x4i);
            _mm512_stream_pd(our+5*8*4096+q, x5r); _mm512_stream_pd(oui+5*8*4096+q, x5i);
            _mm512_stream_pd(our+6*8*4096+q, x6r); _mm512_stream_pd(oui+6*8*4096+q, x6i);
            _mm512_stream_pd(our+7*8*4096+q, x7r); _mm512_stream_pd(oui+7*8*4096+q, x7i);
        }
        if (noT) continue;
        VD F0r,F0i,F1r,F1i,F2r,F2i,F3r,F3i,F4r,F4i,F5r,F5i,F6r,F6i,F7r,F7i;
        DFT8V(F0r,F0i,F1r,F1i,F2r,F2i,F3r,F3i,F4r,F4i,F5r,F5i,F6r,F6i,F7r,F7i,
              x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,x4r,x4i,x5r,x5i,x6r,x6i,x7r,x7i);
        ST(t0r+0*tstep+q, F0r); ST(t0i+0*tstep+q, F0i);
        if (t) {
            VD gr,gi;
            CMULV(gr,gi,F1r,F1i,BC(g_T64r[t][1]),BC(g_T64i[t][1])); ST(t0r+1*tstep+q,gr); ST(t0i+1*tstep+q,gi);
            CMULV(gr,gi,F2r,F2i,BC(g_T64r[t][2]),BC(g_T64i[t][2])); ST(t0r+2*tstep+q,gr); ST(t0i+2*tstep+q,gi);
            CMULV(gr,gi,F3r,F3i,BC(g_T64r[t][3]),BC(g_T64i[t][3])); ST(t0r+3*tstep+q,gr); ST(t0i+3*tstep+q,gi);
            CMULV(gr,gi,F4r,F4i,BC(g_T64r[t][4]),BC(g_T64i[t][4])); ST(t0r+4*tstep+q,gr); ST(t0i+4*tstep+q,gi);
            CMULV(gr,gi,F5r,F5i,BC(g_T64r[t][5]),BC(g_T64i[t][5])); ST(t0r+5*tstep+q,gr); ST(t0i+5*tstep+q,gi);
            CMULV(gr,gi,F6r,F6i,BC(g_T64r[t][6]),BC(g_T64i[t][6])); ST(t0r+6*tstep+q,gr); ST(t0i+6*tstep+q,gi);
            CMULV(gr,gi,F7r,F7i,BC(g_T64r[t][7]),BC(g_T64i[t][7])); ST(t0r+7*tstep+q,gr); ST(t0i+7*tstep+q,gi);
        } else {
            ST(t0r+1*tstep+q,F1r); ST(t0i+1*tstep+q,F1i);
            ST(t0r+2*tstep+q,F2r); ST(t0i+2*tstep+q,F2i);
            ST(t0r+3*tstep+q,F3r); ST(t0i+3*tstep+q,F3i);
            ST(t0r+4*tstep+q,F4r); ST(t0i+4*tstep+q,F4i);
            ST(t0r+5*tstep+q,F5r); ST(t0i+5*tstep+q,F5i);
            ST(t0r+6*tstep+q,F6r); ST(t0i+6*tstep+q,F6i);
            ST(t0r+7*tstep+q,F7r); ST(t0i+7*tstep+q,F7i);
        }
    }
}

static void chain64(double* restrict re, double* restrict im,
                    const double* restrict cre, const double* restrict cim,
                    long m, double* restrict Tre, double* restrict Tim,
                    double* restrict unused1, double* restrict unused2,
                    double* restrict out1)
{
    (void)unused1; (void)unused2;
    for (int b = 0; b < 8; b++) stageA_prime64(re, im, Tre, Tim, b);
    for (long n = 0; n < m; n++) {
        int last = (n == m-1);
        int first = (n == 0);
        int par = (int)(n & 1);
        for (int t = 0; t < 8; t++) {
            stageB64_compute(Tre, Tim, t, par, cre, cim);
            for (int s = 0; s < 8; s++) {
                double *sr = SBre + (size_t)s*SBSTRIDE, *si = SBim + (size_t)s*SBSTRIDE;
                for (int zc = 0; zc < 8; zc++) km64_y(sr + 8*zc, si + 8*zc);
                zslab64(sr, si);
            }
            stageA64_compute(Tre, Tim, t, par, last, cre, cim,
                             (first || last) ? re : 0, (first || last) ? im : 0);
        }
        if (first || last) _mm_sfence();
        if (first && out1) soa_export(re, im, out1, 262144);
    }
}

#pragma GCC pop_options
