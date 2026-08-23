// ====== L = 36 ======
static void km36_y(const double* restrict re, const double* restrict im,
                   double* restrict dre, double* restrict dim, MK m)
{
#define LR_(j) LD(re + (j))
#define LI_(j) LD(im + (j))
#define SP_(k,vr,vi) do { MST(dre + (k), m, vr); MST(dim + (k), m, vi); } while(0)
    KM36_BODY(LR_, LI_, SP_, P36IN_S36, P36OUT_S36);
#undef LR_
#undef LI_
#undef SP_
}
static void km36_s(double* restrict re, double* restrict im)
{
#define LR_(j) LD(re + (j))
#define LI_(j) LD(im + (j))
#define SP_(k,vr,vi) do { ST(re + (k), vr); ST(im + (k), vi); } while(0)
    KM36_BODY(LR_, LI_, SP_, P36IN_S8, P36OUT_S8);
#undef LR_
#undef LI_
#undef SP_
}

// ====== L = 45 ======
static void km45_y(const double* restrict re, const double* restrict im,
                   double* restrict dre, double* restrict dim, MK m)
{
#define LR_(j) LD(re + (j))
#define LI_(j) LD(im + (j))
#define SP_(k,vr,vi) do { MST(dre + (k), m, vr); MST(dim + (k), m, vi); } while(0)
    KM45_BODY(LR_, LI_, SP_, P45IN_S45, P45OUT_S45);
#undef LR_
#undef LI_
#undef SP_
}
static void km45_s(double* restrict re, double* restrict im)
{
#define LR_(j) LD(re + (j))
#define LI_(j) LD(im + (j))
#define SP_(k,vr,vi) do { ST(re + (k), vr); ST(im + (k), vi); } while(0)
    KM45_BODY(LR_, LI_, SP_, P45IN_S8, P45OUT_S8);
#undef LR_
#undef LI_
#undef SP_
}

// ====== L = 64 ======
static void km64_y(double* restrict re, double* restrict im)
{
#define LR_(j) LD(re + 64*(j))
#define LI_(j) LD(im + 64*(j))
#define SP_(k,vr,vi) do { ST(re + 64*(k), vr); ST(im + 64*(k), vi); } while(0)
    KM64_BODY(LR_, LI_, SP_);
#undef LR_
#undef LI_
#undef SP_
}

// z-pass for a whole 64x64 slab: rows of 64, tables hoisted.
static void zslab64(double* restrict re, double* restrict im)
{
    const VD Vr2=BC(g_r2half);
    VD Tzr[8], Tzi[8];
    for (int t = 1; t < 8; t++) { Tzr[t]=LD(g_T64zr[t]); Tzi[t]=LD(g_T64zi[t]); }
    for (int y = 0; y < 64; y++) {
        double *rr = re + 64*y, *ii = im + 64*y;
        VD xr0=LD(rr+0),  xi0=LD(ii+0),  xr1=LD(rr+8),  xi1=LD(ii+8);
        VD xr2=LD(rr+16), xi2=LD(ii+16), xr3=LD(rr+24), xi3=LD(ii+24);
        VD xr4=LD(rr+32), xi4=LD(ii+32), xr5=LD(rr+40), xi5=LD(ii+40);
        VD xr6=LD(rr+48), xi6=LD(ii+48), xr7=LD(rr+56), xi7=LD(ii+56);
        VD F0r,F0i,F1r,F1i,F2r,F2i,F3r,F3i,F4r,F4i,F5r,F5i,F6r,F6i,F7r,F7i;
        DFT8V(F0r,F0i,F1r,F1i,F2r,F2i,F3r,F3i,F4r,F4i,F5r,F5i,F6r,F6i,F7r,F7i,
              xr0,xi0,xr1,xi1,xr2,xi2,xr3,xi3,xr4,xi4,xr5,xi5,xr6,xi6,xr7,xi7);
        VD G1r,G1i,G2r,G2i,G3r,G3i,G4r,G4i,G5r,G5i,G6r,G6i,G7r,G7i;
        CMULV(G1r,G1i,F1r,F1i,Tzr[1],Tzi[1]);
        CMULV(G2r,G2i,F2r,F2i,Tzr[2],Tzi[2]);
        CMULV(G3r,G3i,F3r,F3i,Tzr[3],Tzi[3]);
        CMULV(G4r,G4i,F4r,F4i,Tzr[4],Tzi[4]);
        CMULV(G5r,G5i,F5r,F5i,Tzr[5],Tzi[5]);
        CMULV(G6r,G6i,F6r,F6i,Tzr[6],Tzi[6]);
        CMULV(G7r,G7i,F7r,F7i,Tzr[7],Tzi[7]);
        VD t0r,t1r,t2r,t3r,t4r,t5r,t6r,t7r, t0i,t1i,t2i,t3i,t4i,t5i,t6i,t7i;
        TR8(t0r,t1r,t2r,t3r,t4r,t5r,t6r,t7r, F0r,G1r,G2r,G3r,G4r,G5r,G6r,G7r);
        TR8(t0i,t1i,t2i,t3i,t4i,t5i,t6i,t7i, F0i,G1i,G2i,G3i,G4i,G5i,G6i,G7i);
        VD H0r,H0i,H1r,H1i,H2r,H2i,H3r,H3i,H4r,H4i,H5r,H5i,H6r,H6i,H7r,H7i;
        DFT8V(H0r,H0i,H1r,H1i,H2r,H2i,H3r,H3i,H4r,H4i,H5r,H5i,H6r,H6i,H7r,H7i,
              t0r,t0i,t1r,t1i,t2r,t2i,t3r,t3i,t4r,t4i,t5r,t5i,t6r,t6i,t7r,t7i);
        ST(rr+0,H0r);  ST(ii+0,H0i);  ST(rr+8,H1r);  ST(ii+8,H1i);
        ST(rr+16,H2r); ST(ii+16,H2i); ST(rr+24,H3r); ST(ii+24,H3i);
        ST(rr+32,H4r); ST(ii+32,H4i); ST(rr+40,H5r); ST(ii+40,H5i);
        ST(rr+48,H6r); ST(ii+48,H6i); ST(rr+56,H7r); ST(ii+56,H7i);
    }
}

// z-row DFT-64 on one contiguous row (64 complex), six-step with one transpose.
static void zrow64(double* restrict re, double* restrict im)
{
    const VD Vr2=BC(g_r2half);
    VD xr0=LD(re+0),  xi0=LD(im+0),  xr1=LD(re+8),  xi1=LD(im+8);
    VD xr2=LD(re+16), xi2=LD(im+16), xr3=LD(re+24), xi3=LD(im+24);
    VD xr4=LD(re+32), xi4=LD(im+32), xr5=LD(re+40), xi5=LD(im+40);
    VD xr6=LD(re+48), xi6=LD(im+48), xr7=LD(re+56), xi7=LD(im+56);
    VD F0r,F0i,F1r,F1i,F2r,F2i,F3r,F3i,F4r,F4i,F5r,F5i,F6r,F6i,F7r,F7i;
    DFT8V(F0r,F0i,F1r,F1i,F2r,F2i,F3r,F3i,F4r,F4i,F5r,F5i,F6r,F6i,F7r,F7i,
          xr0,xi0,xr1,xi1,xr2,xi2,xr3,xi3,xr4,xi4,xr5,xi5,xr6,xi6,xr7,xi7);
    VD G1r,G1i,G2r,G2i,G3r,G3i,G4r,G4i,G5r,G5i,G6r,G6i,G7r,G7i;
    CMULV(G1r,G1i,F1r,F1i,LD(g_T64zr[1]),LD(g_T64zi[1]));
    CMULV(G2r,G2i,F2r,F2i,LD(g_T64zr[2]),LD(g_T64zi[2]));
    CMULV(G3r,G3i,F3r,F3i,LD(g_T64zr[3]),LD(g_T64zi[3]));
    CMULV(G4r,G4i,F4r,F4i,LD(g_T64zr[4]),LD(g_T64zi[4]));
    CMULV(G5r,G5i,F5r,F5i,LD(g_T64zr[5]),LD(g_T64zi[5]));
    CMULV(G6r,G6i,F6r,F6i,LD(g_T64zr[6]),LD(g_T64zi[6]));
    CMULV(G7r,G7i,F7r,F7i,LD(g_T64zr[7]),LD(g_T64zi[7]));
    VD t0r,t1r,t2r,t3r,t4r,t5r,t6r,t7r, t0i,t1i,t2i,t3i,t4i,t5i,t6i,t7i;
    TR8(t0r,t1r,t2r,t3r,t4r,t5r,t6r,t7r, F0r,G1r,G2r,G3r,G4r,G5r,G6r,G7r);
    TR8(t0i,t1i,t2i,t3i,t4i,t5i,t6i,t7i, F0i,G1i,G2i,G3i,G4i,G5i,G6i,G7i);
    VD H0r,H0i,H1r,H1i,H2r,H2i,H3r,H3i,H4r,H4i,H5r,H5i,H6r,H6i,H7r,H7i;
    DFT8V(H0r,H0i,H1r,H1i,H2r,H2i,H3r,H3i,H4r,H4i,H5r,H5i,H6r,H6i,H7r,H7i,
          t0r,t0i,t1r,t1i,t2r,t2i,t3r,t3i,t4r,t4i,t5r,t5i,t6r,t6i,t7r,t7i);
    ST(re+0,H0r);  ST(im+0,H0i);  ST(re+8,H1r);  ST(im+8,H1i);
    ST(re+16,H2r); ST(im+16,H2i); ST(re+24,H3r); ST(im+24,H3i);
    ST(re+32,H4r); ST(im+32,H4i); ST(re+40,H5r); ST(im+40,H5i);
    ST(re+48,H6r); ST(im+48,H6i); ST(re+56,H7r); ST(im+56,H7i);
}

// ---------------------------------------------------------------------------
// sandwich z-pass for 36/45 (OOP): read src slab, write dst slab
// ---------------------------------------------------------------------------
static void spass_gen(const double* restrict sre, const double* restrict sim,
                      double* restrict dstre, double* restrict dstim, int L,
                      int nzc, const MK* zmask, void (*kern)(double*,double*))
{
    VD TMPr[64+8] __attribute__((aligned(64)));
    VD TMPi[64+8] __attribute__((aligned(64)));
    for (int y0 = 0; y0 < L; y0 += 8) {
        int R = L - y0; if (R > 8) R = 8;
        for (int zc = 0; zc < nzc; zc++) {
            VD r0,r1,r2,r3,r4,r5,r6,r7;
            const double *b0 = sre + (size_t)y0*L + 8*zc;
            r0 =           LD(b0 + 0*L);
            r1 = (R > 1) ? LD(b0 + 1*L) : ZERO;
            r2 = (R > 2) ? LD(b0 + 2*L) : ZERO;
            r3 = (R > 3) ? LD(b0 + 3*L) : ZERO;
            r4 = (R > 4) ? LD(b0 + 4*L) : ZERO;
            r5 = (R > 5) ? LD(b0 + 5*L) : ZERO;
            r6 = (R > 6) ? LD(b0 + 6*L) : ZERO;
            r7 = (R > 7) ? LD(b0 + 7*L) : ZERO;
            VD o0,o1,o2,o3,o4,o5,o6,o7;
            TR8(o0,o1,o2,o3,o4,o5,o6,o7, r0,r1,r2,r3,r4,r5,r6,r7);
            TMPr[8*zc+0]=o0; TMPr[8*zc+1]=o1; TMPr[8*zc+2]=o2; TMPr[8*zc+3]=o3;
            TMPr[8*zc+4]=o4; TMPr[8*zc+5]=o5; TMPr[8*zc+6]=o6; TMPr[8*zc+7]=o7;
            const double *b1 = sim + (size_t)y0*L + 8*zc;
            r0 =           LD(b1 + 0*L);
            r1 = (R > 1) ? LD(b1 + 1*L) : ZERO;
            r2 = (R > 2) ? LD(b1 + 2*L) : ZERO;
            r3 = (R > 3) ? LD(b1 + 3*L) : ZERO;
            r4 = (R > 4) ? LD(b1 + 4*L) : ZERO;
            r5 = (R > 5) ? LD(b1 + 5*L) : ZERO;
            r6 = (R > 6) ? LD(b1 + 6*L) : ZERO;
            r7 = (R > 7) ? LD(b1 + 7*L) : ZERO;
            TR8(o0,o1,o2,o3,o4,o5,o6,o7, r0,r1,r2,r3,r4,r5,r6,r7);
            TMPi[8*zc+0]=o0; TMPi[8*zc+1]=o1; TMPi[8*zc+2]=o2; TMPi[8*zc+3]=o3;
            TMPi[8*zc+4]=o4; TMPi[8*zc+5]=o5; TMPi[8*zc+6]=o6; TMPi[8*zc+7]=o7;
        }
        kern((double*)TMPr, (double*)TMPi);
        for (int zc = 0; zc < nzc; zc++) {
            MK zm = zmask[zc];
            VD o0,o1,o2,o3,o4,o5,o6,o7;
            TR8(o0,o1,o2,o3,o4,o5,o6,o7,
                TMPr[8*zc+0],TMPr[8*zc+1],TMPr[8*zc+2],TMPr[8*zc+3],
                TMPr[8*zc+4],TMPr[8*zc+5],TMPr[8*zc+6],TMPr[8*zc+7]);
            double *b0 = dstre + (size_t)y0*L + 8*zc;
            MST(b0 + 0*L, zm, o0);
            if (R > 1) MST(b0 + 1*L, zm, o1);
            if (R > 2) MST(b0 + 2*L, zm, o2);
            if (R > 3) MST(b0 + 3*L, zm, o3);
            if (R > 4) MST(b0 + 4*L, zm, o4);
            if (R > 5) MST(b0 + 5*L, zm, o5);
            if (R > 6) MST(b0 + 6*L, zm, o6);
            if (R > 7) MST(b0 + 7*L, zm, o7);
            TR8(o0,o1,o2,o3,o4,o5,o6,o7,
                TMPi[8*zc+0],TMPi[8*zc+1],TMPi[8*zc+2],TMPi[8*zc+3],
                TMPi[8*zc+4],TMPi[8*zc+5],TMPi[8*zc+6],TMPi[8*zc+7]);
            double *b1 = dstim + (size_t)y0*L + 8*zc;
            MST(b1 + 0*L, zm, o0);
            if (R > 1) MST(b1 + 1*L, zm, o1);
            if (R > 2) MST(b1 + 2*L, zm, o2);
            if (R > 3) MST(b1 + 3*L, zm, o3);
            if (R > 4) MST(b1 + 4*L, zm, o4);
            if (R > 5) MST(b1 + 5*L, zm, o5);
            if (R > 6) MST(b1 + 6*L, zm, o6);
            if (R > 7) MST(b1 + 7*L, zm, o7);
        }
    }
}
