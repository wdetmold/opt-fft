// ====== odd primes 13 / 17 / 23 ======
static void km13_y(const double* restrict re, const double* restrict im,
                   double* restrict dre, double* restrict dim, MK m)
{
#define LR_(j) LD(re + 13*(j))
#define LI_(j) LD(im + 13*(j))
#define SP_(k,vr,vi) do { MST(dre + 13*(k), m, vr); MST(dim + 13*(k), m, vi); } while(0)
    KMFOLD_BODY(13, 6, g_C13, g_S13, LR_, LI_, SP_);
#undef LR_
#undef LI_
#undef SP_
}
static void km13_x_map(const double* restrict re, const double* restrict im,
                       double* restrict dre, double* restrict dim,
                       const double* restrict cre, const double* restrict cim, MK m)
{
    const VD Vone=BC(1.0), V1p5=BC(1.5), Vtiny=BC(2.3e-308), Vhalf=BC(0.5);
#define LR_(j) LD(re + 169*(j))
#define LI_(j) LD(im + 169*(j))
#define SP_(k,vr,vi) MAPST(dre + 169*(k), dim + 169*(k), m, vr, vi, cre + 169*(k), cim + 169*(k))
    KMFOLD_BODY(13, 6, g_C13, g_S13, LR_, LI_, SP_);
#undef LR_
#undef LI_
#undef SP_
}
static void km17_y(const double* restrict re, const double* restrict im,
                   double* restrict dre, double* restrict dim, MK m)
{
#define LR_(j) LD(re + 17*(j))
#define LI_(j) LD(im + 17*(j))
#define SP_(k,vr,vi) do { MST(dre + 17*(k), m, vr); MST(dim + 17*(k), m, vi); } while(0)
    KMFOLD_BODY(17, 8, g_C17, g_S17, LR_, LI_, SP_);
#undef LR_
#undef LI_
#undef SP_
}
static void km17_x_map(const double* restrict re, const double* restrict im,
                       double* restrict dre, double* restrict dim,
                       const double* restrict cre, const double* restrict cim, MK m)
{
    const VD Vone=BC(1.0), V1p5=BC(1.5), Vtiny=BC(2.3e-308), Vhalf=BC(0.5);
#define LR_(j) LD(re + 289*(j))
#define LI_(j) LD(im + 289*(j))
#define SP_(k,vr,vi) MAPST(dre + 289*(k), dim + 289*(k), m, vr, vi, cre + 289*(k), cim + 289*(k))
    KMFOLD_BODY(17, 8, g_C17, g_S17, LR_, LI_, SP_);
#undef LR_
#undef LI_
#undef SP_
}
static void km23_y(const double* restrict re, const double* restrict im,
                   double* restrict dre, double* restrict dim, MK m)
{
#define LR_(j) LD(re + 23*(j))
#define LI_(j) LD(im + 23*(j))
#define SP_(k,vr,vi) do { MST(dre + 23*(k), m, vr); MST(dim + 23*(k), m, vi); } while(0)
    KMFOLD_BODY(23, 11, g_C23, g_S23, LR_, LI_, SP_);
#undef LR_
#undef LI_
#undef SP_
}
static void km23_x_map(const double* restrict re, const double* restrict im,
                       double* restrict dre, double* restrict dim,
                       const double* restrict cre, const double* restrict cim, MK m)
{
    const VD Vone=BC(1.0), V1p5=BC(1.5), Vtiny=BC(2.3e-308), Vhalf=BC(0.5);
#define LR_(j) LD(re + 529*(j))
#define LI_(j) LD(im + 529*(j))
#define SP_(k,vr,vi) MAPST(dre + 529*(k), dim + 529*(k), m, vr, vi, cre + 529*(k), cim + 529*(k))
    KMFOLD_BODY(23, 11, g_C23, g_S23, LR_, LI_, SP_);
#undef LR_
#undef LI_
#undef SP_
}

#define REVIDX _mm512_set_epi64(0,1,2,3,4,5,6,7)
#define VREV(v) _mm512_permutexvar_pd(REVIDX, v)

// z-axis for 13 (OOP): h=6, vector prep into stack, two rows at a time.
static void zpass13(const double* restrict sre, const double* restrict sim,
                    double* restrict dre, double* restrict dim, int nrows)
{
    VD CzR[6], SzR[6];
    for (int j = 0; j < 6; j++) { CzR[j]=LD(g_Cz13[j]); SzR[j]=LD(g_Sz13[j]); }
    double ub[2][4][8] __attribute__((aligned(64)));
#define ZP13PREP(pr, pi, R) do { \
    VD a_r = LD((pr)+1), a_i = LD((pi)+1); \
    VD b_r = VREV(LD((pr)+5)), b_i = VREV(LD((pi)+5)); \
    ST(ub[R][0], VADD(a_r,b_r)); ST(ub[R][1], VADD(a_i,b_i)); \
    ST(ub[R][2], VSUB(a_r,b_r)); ST(ub[R][3], VSUB(a_i,b_i)); \
} while(0)
#define ZP13ROW(pr, pi, qr, qi, R, X) \
    VD rsr##X = VADD(LD(pr), MLD(0x1F,(pr)+8)), rsi##X = VADD(LD(pi), MLD(0x1F,(pi)+8)); \
    VD Pr##X = BC((pr)[0]), Pi##X = BC((pi)[0]), Qr##X = ZERO, Qi##X = ZERO; \
    _Pragma("GCC unroll 8") \
    for (int j = 0; j < 6; j++) { \
        Pr##X = VFMA(BC(ub[R][0][j]), CzR[j], Pr##X); Pi##X = VFMA(BC(ub[R][1][j]), CzR[j], Pi##X); \
        Qr##X = VFMA(BC(ub[R][2][j]), SzR[j], Qr##X); Qi##X = VFMA(BC(ub[R][3][j]), SzR[j], Qi##X); \
    } \
    (qr)[0] = _mm512_reduce_add_pd(rsr##X); (qi)[0] = _mm512_reduce_add_pd(rsi##X); \
    MST((qr)+1, 0x3F, VADD(Pr##X,Qi##X)); MST((qi)+1, 0x3F, VSUB(Pi##X,Qr##X)); \
    MST((qr)+5, 0xFC, VREV(VSUB(Pr##X,Qi##X))); MST((qi)+5, 0xFC, VREV(VADD(Pi##X,Qr##X)));
    int r = 0;
    for (; r + 1 < nrows; r += 2) {
        const double *prA = sre + 13*r, *piA = sim + 13*r;
        const double *prB = prA + 13,  *piB = piA + 13;
        double *qrA = dre + 13*r, *qiA = dim + 13*r, *qrB = qrA + 13, *qiB = qiA + 13;
        ZP13PREP(prA, piA, 0);
        ZP13PREP(prB, piB, 1);
        ZP13ROW(prA, piA, qrA, qiA, 0, A);
        ZP13ROW(prB, piB, qrB, qiB, 1, B);
    }
    if (r < nrows) {
        const double *prA = sre + 13*r, *piA = sim + 13*r;
        double *qrA = dre + 13*r, *qiA = dim + 13*r;
        ZP13PREP(prA, piA, 0);
        ZP13ROW(prA, piA, qrA, qiA, 0, C);
    }
#undef ZP13PREP
#undef ZP13ROW
}
// z-axis for 17 (OOP): h=8 exact, vector prep, two rows at a time.
static void zpass17(const double* restrict sre, const double* restrict sim,
                    double* restrict dre, double* restrict dim, int nrows)
{
    VD CzR[8], SzR[8];
    for (int j = 0; j < 8; j++) { CzR[j]=LD(g_Cz17[j]); SzR[j]=LD(g_Sz17[j]); }
    double ub[2][4][8] __attribute__((aligned(64)));
#define ZP17PREP(pr, pi, R) do { \
    VD a_r = LD((pr)+1), a_i = LD((pi)+1); \
    VD b_r = VREV(LD((pr)+9)), b_i = VREV(LD((pi)+9)); \
    ST(ub[R][0], VADD(a_r,b_r)); ST(ub[R][1], VADD(a_i,b_i)); \
    ST(ub[R][2], VSUB(a_r,b_r)); ST(ub[R][3], VSUB(a_i,b_i)); \
} while(0)
#define ZP17ROW(pr, pi, qr, qi, R, X) \
    VD rsr##X = VADD(LD(pr), LD((pr)+8)), rsi##X = VADD(LD(pi), LD((pi)+8)); \
    VD Pr##X = BC((pr)[0]), Pi##X = BC((pi)[0]), Qr##X = ZERO, Qi##X = ZERO; \
    _Pragma("GCC unroll 8") \
    for (int j = 0; j < 8; j++) { \
        Pr##X = VFMA(BC(ub[R][0][j]), CzR[j], Pr##X); Pi##X = VFMA(BC(ub[R][1][j]), CzR[j], Pi##X); \
        Qr##X = VFMA(BC(ub[R][2][j]), SzR[j], Qr##X); Qi##X = VFMA(BC(ub[R][3][j]), SzR[j], Qi##X); \
    } \
    double dcr##X = _mm512_reduce_add_pd(rsr##X) + (pr)[16]; \
    double dci##X = _mm512_reduce_add_pd(rsi##X) + (pi)[16]; \
    (qr)[0] = dcr##X; (qi)[0] = dci##X; \
    ST((qr)+1, VADD(Pr##X,Qi##X)); ST((qi)+1, VSUB(Pi##X,Qr##X)); \
    ST((qr)+9, VREV(VSUB(Pr##X,Qi##X))); ST((qi)+9, VREV(VADD(Pi##X,Qr##X)));
    int r = 0;
    for (; r + 1 < nrows; r += 2) {
        const double *prA = sre + 17*r, *piA = sim + 17*r;
        const double *prB = prA + 17,  *piB = piA + 17;
        double *qrA = dre + 17*r, *qiA = dim + 17*r, *qrB = qrA + 17, *qiB = qiA + 17;
        ZP17PREP(prA, piA, 0);
        ZP17PREP(prB, piB, 1);
        ZP17ROW(prA, piA, qrA, qiA, 0, A);
        ZP17ROW(prB, piB, qrB, qiB, 1, B);
    }
    if (r < nrows) {
        const double *prA = sre + 17*r, *piA = sim + 17*r;
        double *qrA = dre + 17*r, *qiA = dim + 17*r;
        ZP17PREP(prA, piA, 0);
        ZP17ROW(prA, piA, qrA, qiA, 0, C);
    }
#undef ZP17PREP
#undef ZP17ROW
}
// z-axis for 23 (OOP): h=11, two k-chunks; vector prep into stack buffers.
static void zpass23(const double* restrict sre, const double* restrict sim,
                    double* restrict dre, double* restrict dim, int nrows)
{
    double ub[4][16] __attribute__((aligned(64)));  /* [ur,ui,vr,vi][j-1] */
#define ZP23PREP(pr, pi) do { \
    VD a_r = LD((pr)+1), a_i = LD((pi)+1); \
    VD b_r = VREV(LD((pr)+15)), b_i = VREV(LD((pi)+15)); \
    ST(ub[0], VADD(a_r,b_r)); ST(ub[1], VADD(a_i,b_i)); \
    ST(ub[2], VSUB(a_r,b_r)); ST(ub[3], VSUB(a_i,b_i)); \
    VD a2r = LD((pr)+9), a2i = LD((pi)+9); \
    VD b2r = VREV(LD((pr)+7)), b2i = VREV(LD((pi)+7)); \
    ST(ub[0]+8, VADD(a2r,b2r)); ST(ub[1]+8, VADD(a2i,b2i)); \
    ST(ub[2]+8, VSUB(a2r,b2r)); ST(ub[3]+8, VSUB(a2i,b2i)); \
} while(0)
    for (int r = 0; r < nrows; r++) {
        const double *pr = sre + 23*r, *pi = sim + 23*r;
        double *qr = dre + 23*r, *qi = dim + 23*r;
        VD rsr = VADD(VADD(LD(pr), LD(pr+8)), MLD(0x7F,pr+16));
        VD rsi = VADD(VADD(LD(pi), LD(pi+8)), MLD(0x7F,pi+16));
        ZP23PREP(pr, pi);
        VD P1r = BC(pr[0]), P1i = BC(pi[0]), Q1r = ZERO, Q1i = ZERO;
        VD P2r = P1r, P2i = P1i, Q2r = ZERO, Q2i = ZERO;
        _Pragma("GCC unroll 16")
        for (int j = 0; j < 11; j++) {
            VD c1 = LD(g_Cz23[0][j]), s1 = LD(g_Sz23[0][j]);
            VD c2 = LD(g_Cz23[1][j]), s2 = LD(g_Sz23[1][j]);
            VD ur = BC(ub[0][j]), uv = BC(ub[1][j]);
            VD wr = BC(ub[2][j]), wv = BC(ub[3][j]);
            P1r = VFMA(ur, c1, P1r); P1i = VFMA(uv, c1, P1i);
            Q1r = VFMA(wr, s1, Q1r); Q1i = VFMA(wv, s1, Q1i);
            P2r = VFMA(ur, c2, P2r); P2i = VFMA(uv, c2, P2i);
            Q2r = VFMA(wr, s2, Q2r); Q2i = VFMA(wv, s2, Q2i);
        }
        qr[0] = _mm512_reduce_add_pd(rsr); qi[0] = _mm512_reduce_add_pd(rsi);
        ST(qr+1,  VADD(P1r,Q1i)); ST(qi+1,  VSUB(P1i,Q1r));
        ST(qr+15, VREV(VSUB(P1r,Q1i))); ST(qi+15, VREV(VADD(P1i,Q1r)));
        MST(qr+9, 0x07, VADD(P2r,Q2i)); MST(qi+9, 0x07, VSUB(P2i,Q2r));
        MST(qr+7, 0xE0, VREV(VSUB(P2r,Q2i))); MST(qi+7, 0xE0, VREV(VADD(P2i,Q2r)));
    }
#undef ZP23PREP
}
