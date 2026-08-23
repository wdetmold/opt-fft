// ---------------------------------------------------------------------------
// Kernel instantiations (out-of-place: separate src/dst)
// ---------------------------------------------------------------------------
// ====== L = 6 ======
static inline __attribute__((always_inline)) void km6_y(const double* restrict re, const double* restrict im,
                  double* restrict dre, double* restrict dim, MK m)
{
#define LR_(j) LD(re + 6*(j))
#define LI_(j) LD(im + 6*(j))
#define SP_(k,vr,vi) do { MST(dre + 6*(k), m, vr); MST(dim + 6*(k), m, vi); } while(0)
    KM6_BODY(LR_, LI_, SP_);
#undef LR_
#undef LI_
#undef SP_
}
static inline __attribute__((always_inline)) void km6_x_map(const double* restrict re, const double* restrict im,
                      double* restrict dre, double* restrict dim,
                      const double* restrict cre, const double* restrict cim, MK m)
{
    const VD Vone=BC(1.0), V1p5=BC(1.5), Vtiny=BC(2.3e-308);
#define LR_(j) LD(re + 36*(j))
#define LI_(j) LD(im + 36*(j))
#define SP_(k,vr,vi) MAPST(dre + 36*(k), dim + 36*(k), m, vr, vi, cre + 36*(k), cim + 36*(k))
    KM6_BODY(LR_, LI_, SP_);
#undef LR_
#undef LI_
#undef SP_
}
static inline __attribute__((always_inline)) void zpass6(const double* restrict sre, const double* restrict sim,
                   double* restrict dre, double* restrict dim)
{
    const VD Vhalf=BC(0.5), Vs3=BC(g_s3half);
    for (int g = 0; g < 5; g++) {
        int nrows = (g < 4) ? 8 : 4;
        const double *pr = sre + 48*g, *pi = sim + 48*g;
        double *qr = dre + 48*g, *qi = dim + 48*g;
        VD r0,r1,r2,r3,r4,r5,r6,r7, t0r,t1r,t2r,t3r,t4r,t5r,t6r,t7r, t0i,t1i,t2i,t3i,t4i,t5i,t6i,t7i;
        r0=LD(pr); r1=LD(pr+6); r2=LD(pr+12); r3=LD(pr+18); r4=LD(pr+24); r5=LD(pr+30);
        r6 = (nrows>6)?LD(pr+36):ZERO; r7=(nrows>7)?LD(pr+42):ZERO;
        TR8(t0r,t1r,t2r,t3r,t4r,t5r,t6r,t7r, r0,r1,r2,r3,r4,r5,r6,r7);
        r0=LD(pi); r1=LD(pi+6); r2=LD(pi+12); r3=LD(pi+18); r4=LD(pi+24); r5=LD(pi+30);
        r6 = (nrows>6)?LD(pi+36):ZERO; r7=(nrows>7)?LD(pi+42):ZERO;
        TR8(t0i,t1i,t2i,t3i,t4i,t5i,t6i,t7i, r0,r1,r2,r3,r4,r5,r6,r7);
        VD A00r,A00i,A01r,A01i,A02r,A02i, A10r,A10i,A11r,A11i,A12r,A12i;
        DFT3V(A00r,A00i,A01r,A01i,A02r,A02i, t0r,t0i,t2r,t2i,t4r,t4i);
        DFT3V(A10r,A10i,A11r,A11i,A12r,A12i, t3r,t3i,t5r,t5i,t1r,t1i);
        VD o0r=VADD(A00r,A10r), o0i=VADD(A00i,A10i);
        VD o3r=VSUB(A00r,A10r), o3i=VSUB(A00i,A10i);
        VD o4r=VADD(A01r,A11r), o4i=VADD(A01i,A11i);
        VD o1r=VSUB(A01r,A11r), o1i=VSUB(A01i,A11i);
        VD o2r=VADD(A02r,A12r), o2i=VADD(A02i,A12i);
        VD o5r=VSUB(A02r,A12r), o5i=VSUB(A02i,A12i);
        TR8(r0,r1,r2,r3,r4,r5,r6,r7, o0r,o1r,o2r,o3r,o4r,o5r,o5r,o5r);
        MST(qr, 0x3F, r0); MST(qr+6, 0x3F, r1); MST(qr+12, 0x3F, r2); MST(qr+18, 0x3F, r3);
        if (nrows > 4) { MST(qr+24, 0x3F, r4); MST(qr+30, 0x3F, r5); MST(qr+36, 0x3F, r6); MST(qr+42, 0x3F, r7); }
        TR8(r0,r1,r2,r3,r4,r5,r6,r7, o0i,o1i,o2i,o3i,o4i,o5i,o5i,o5i);
        MST(qi, 0x3F, r0); MST(qi+6, 0x3F, r1); MST(qi+12, 0x3F, r2); MST(qi+18, 0x3F, r3);
        if (nrows > 4) { MST(qi+24, 0x3F, r4); MST(qi+30, 0x3F, r5); MST(qi+36, 0x3F, r6); MST(qi+42, 0x3F, r7); }
    }
}

// ====== L = 8 ======
static inline __attribute__((always_inline)) void km8_y(const double* restrict re, const double* restrict im,
                  double* restrict dre, double* restrict dim)
{
#define LR_(j) LD(re + 8*(j))
#define LI_(j) LD(im + 8*(j))
#define SP_(k,vr,vi) do { ST(dre + 8*(k), vr); ST(dim + 8*(k), vi); } while(0)
    KM8_BODY(LR_, LI_, SP_);
#undef LR_
#undef LI_
#undef SP_
}
static inline __attribute__((always_inline)) void km8_x_map(const double* restrict re, const double* restrict im,
                      double* restrict dre, double* restrict dim,
                      const double* restrict cre, const double* restrict cim)
{
    const VD Vone=BC(1.0), V1p5=BC(1.5), Vtiny=BC(2.3e-308), Vhalf=BC(0.5);
    const MK m = 0xFF;
#define LR_(j) LD(re + 64*(j))
#define LI_(j) LD(im + 64*(j))
#define SP_(k,vr,vi) MAPST(dre + 64*(k), dim + 64*(k), m, vr, vi, cre + 64*(k), cim + 64*(k))
    KM8_BODY(LR_, LI_, SP_);
#undef LR_
#undef LI_
#undef SP_
}
static inline __attribute__((always_inline)) void zpass8(const double* restrict sre, const double* restrict sim,
                   double* restrict dre, double* restrict dim)
{
    const VD Vr2=BC(g_r2half);
    for (int g = 0; g < 8; g++) {
        const double *pr = sre + 64*g, *pi = sim + 64*g;
        double *qr = dre + 64*g, *qi = dim + 64*g;
        VD r0,r1,r2,r3,r4,r5,r6,r7, t0r,t1r,t2r,t3r,t4r,t5r,t6r,t7r, t0i,t1i,t2i,t3i,t4i,t5i,t6i,t7i;
        r0=LD(pr); r1=LD(pr+8); r2=LD(pr+16); r3=LD(pr+24); r4=LD(pr+32); r5=LD(pr+40); r6=LD(pr+48); r7=LD(pr+56);
        TR8(t0r,t1r,t2r,t3r,t4r,t5r,t6r,t7r, r0,r1,r2,r3,r4,r5,r6,r7);
        r0=LD(pi); r1=LD(pi+8); r2=LD(pi+16); r3=LD(pi+24); r4=LD(pi+32); r5=LD(pi+40); r6=LD(pi+48); r7=LD(pi+56);
        TR8(t0i,t1i,t2i,t3i,t4i,t5i,t6i,t7i, r0,r1,r2,r3,r4,r5,r6,r7);
        VD o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i,o4r,o4i,o5r,o5i,o6r,o6i,o7r,o7i;
        DFT8V(o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i,o4r,o4i,o5r,o5i,o6r,o6i,o7r,o7i,
              t0r,t0i,t1r,t1i,t2r,t2i,t3r,t3i,t4r,t4i,t5r,t5i,t6r,t6i,t7r,t7i);
        TR8(r0,r1,r2,r3,r4,r5,r6,r7, o0r,o1r,o2r,o3r,o4r,o5r,o6r,o7r);
        ST(qr,r0); ST(qr+8,r1); ST(qr+16,r2); ST(qr+24,r3); ST(qr+32,r4); ST(qr+40,r5); ST(qr+48,r6); ST(qr+56,r7);
        TR8(r0,r1,r2,r3,r4,r5,r6,r7, o0i,o1i,o2i,o3i,o4i,o5i,o6i,o7i);
        ST(qi,r0); ST(qi+8,r1); ST(qi+16,r2); ST(qi+24,r3); ST(qi+32,r4); ST(qi+40,r5); ST(qi+48,r6); ST(qi+56,r7);
    }
}
