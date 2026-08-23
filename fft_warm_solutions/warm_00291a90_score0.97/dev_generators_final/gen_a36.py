"""Batch-lane (SoA, 8 volumes/lane-group) driver for L=36 using table-driven two-stage DFT + fused map."""

def gen_a36():
    L = 36
    L2, L3 = L*L, L*L*L
    PS = L2          # slots per plane
    PSZ = PS*16      # doubles per plane
    CBS = L*16 + 8   # c block stride (doubles) per pencil block
    return f"""
// ---------------- batch-lane driver, L=36 (PS={PS}) ----------------
static DTAB_36 DA36_y, DA36_z, DA36_x;
static double* XA36 = 0;
static double* CGA36 = 0;
static double* CZPA36 = 0;
static double* CPA36 = 0;
static double* CPPA36 = 0;
static double* SNA36 = 0;
static double XS36A[36*16+8] ALIGN64;
static double XS36B[36*16+24] ALIGN64;
static double MM36A[36*16+8] ALIGN64;
static void dtinit36A(void){{
    static const int inm[9][4] = {{ {", ".join("{" + ", ".join(str((9*n1 + 4*n2) % 36) for n1 in range(4)) + "}" for n2 in range(9))} }};
    static const int outm[4][9] = {{ {", ".join("{" + ", ".join(str(__import__('genlib').pfa_maps(4,9)[1][k1][k2]) for k2 in range(9)) + "}" for k1 in range(4))} }};
    long strides[3][2] = {{ {{36*16, 36*16}}, {{16, 16}}, {{{PSZ}, 16}} }};
    DTAB_36* tabs[3] = {{ &DA36_y, &DA36_z, &DA36_x }};
    for(int t=0;t<3;t++){{
        for(int n2=0;n2<9;n2++) for(int n1=0;n1<4;n1++) tabs[t]->in[n2][n1] = (long)inm[n2][n1]*strides[t][0];
        for(int k1=0;k1<4;k1++) for(int k2=0;k2<9;k2++) tabs[t]->out[k1][k2] = (long)outm[k1][k2]*strides[t][1];
    }}
    // x output stride: PSZ (in-place pencils)
    for(int k1=0;k1<4;k1++) for(int k2=0;k2<9;k2++) DA36_x.out[k1][k2] = (long)outm[k1][k2]*{PSZ};
}}
static DTAB_36 DA36_xc;  // x variant with dst stride 16 (to XS)
static DTAB_36 DA36_cx;  // src stride 16 (from MM) dst stride PSZ
static void dtinit36A2(void){{
    static const int inm[9][4] = {{ {", ".join("{" + ", ".join(str((9*n1 + 4*n2) % 36) for n1 in range(4)) + "}" for n2 in range(9))} }};
    static const int outm[4][9] = {{ {", ".join("{" + ", ".join(str(__import__('genlib').pfa_maps(4,9)[1][k1][k2]) for k2 in range(9)) + "}" for k1 in range(4))} }};
    for(int n2=0;n2<9;n2++) for(int n1=0;n1<4;n1++){{ DA36_xc.in[n2][n1] = (long)inm[n2][n1]*{PSZ}; DA36_cx.in[n2][n1] = (long)inm[n2][n1]*16; }}
    for(int k1=0;k1<4;k1++) for(int k2=0;k2<9;k2++){{ DA36_xc.out[k1][k2] = (long)outm[k1][k2]*16; DA36_cx.out[k1][k2] = (long)outm[k1][k2]*{PSZ}; }}
}}
// S visit: mode 0: y,z ; 1: y, z+map(final) ; 2: y, z+map+z', y'
static void SA_36(double* X, const double* CG, const double* CZP, int mode){{
    for(int i=0;i<36;i++){{
        double* sl = X + (long)i*{PSZ};
        _Pragma("GCC unroll 1") for(int k=0;k<36;k++) dft36_g(sl + k*16, sl + k*16, &DA36_y);
        if(mode == 0){{
            _Pragma("GCC unroll 1") for(int j=0;j<36;j++) dft36_g(sl + (long)j*{L*16}, sl + (long)j*{L*16}, &DA36_z);
        }} else if(mode == 1){{
            _Pragma("GCC unroll 1") for(int j=0;j<36;j++){{
                double* p = sl + (long)j*{L*16};
                dft36_g(p, XS36A, &DA36_z);
                mp36_l(XS36A, CG + (long)i*{PSZ} + (long)j*{L*16}, CG, p);
            }}
        }} else {{
            _Pragma("GCC unroll 1") for(int j=0;j<36;j++){{
                double* p = sl + (long)j*{L*16};
                double* xs = (j & 1) ? XS36B : XS36A;
                dft36_g(p, xs, &DA36_z);
                dft36_gmi(xs, CZP + ((long)i*36 + j)*{CBS}, p, &DA36_z);
            }}
            _Pragma("GCC unroll 1") for(int k=0;k<36;k++) dft36_g(sl + k*16, sl + k*16, &DA36_y);
        }}
    }}
}}
// P visit: mode 1: x+map(final); 2: x+map+x'; 3: +snap
static void PA_36(double* X, const double* CPn, const double* CPP, int mode, double* SNAP){{
    _Pragma("GCC unroll 1") for(long e=0;e<{L2};e++){{
        double* p = X + e*16;
        double* XSL = (e & 1) ? XS36B : XS36A;
        dft36_g(p, XSL, &DA36_xc);
        if(mode == 1){{
            mp36_l(XSL, CPn + e*{CBS}, CPn, MM36A);
            for(int q=0;q<36;q++){{
                _mm512_store_pd(p + (long)q*{PSZ}, _mm512_load_pd(MM36A + q*16));
                _mm512_store_pd(p + (long)q*{PSZ} + 8, _mm512_load_pd(MM36A + q*16 + 8));
            }}
        }} else if(mode == 2){{
            dft36_gmi(XSL, CPP + e*{CBS}, p, &DA36_cx);
        }} else {{
            dft36_gmis(XSL, CPP + e*{CBS}, p, &DA36_cx, SNAP + e*16, &DA36_xc);
        }}
    }}
}}
void run_36A(const double* x0, const double* c, double* out1, double* outm, long B, long m){{
    if(!XA36){{
        XA36  = alloc_huge_st((long)36*{PSZ}*8);
        CGA36 = alloc_huge_st((long)36*{PSZ}*8);
        CZPA36 = alloc_huge_st((long)36*36*{CBS}*8);
        CPA36 = alloc_huge_st((long){L2}*{CBS}*8);
        CPPA36 = alloc_huge_st((long){L2}*{CBS}*8);
        SNA36 = alloc_huge_st((long)36*{PSZ}*8);
        dtinit36A(); dtinit36A2();
    }}
    long G = B / 8;
    const double* srcs[8]; double* dsts[8];
    for(long g=0; g<G; g++){{
        long v0 = g*8;
        for(int v=0; v<8; v++) srcs[v] = x0 + (v0 + v)*2*{L3};
        convin_36A(srcs, XA36);
        for(int v=0; v<8; v++) srcs[v] = c + (v0 + v)*2*{L3};
        convin_36A(srcs, CGA36);
        // permuted z-block c: block (i,j): slot t=(n2*4+n1) <- CG[i][j*36 + PERM_36[t]]
        for(long i=0;i<36;i++) for(long j=0;j<36;j++){{
            const double* sc = CGA36 + i*{PSZ} + j*{L*16};
            double* dc = CZPA36 + (i*36 + j)*{CBS};
            for(int t=0;t<36;t++){{
                long q = PERM_36[t];
                _mm512_store_pd(dc + t*16, _mm512_load_pd(sc + q*16));
                _mm512_store_pd(dc + t*16+8, _mm512_load_pd(sc + q*16+8));
            }}
        }}
        // pencil-major c (natural + permuted)
        for(long e=0;e<{L2};e++){{
            double* dn = CPA36 + e*{CBS};
            double* dp = CPPA36 + e*{CBS};
            for(int q=0;q<36;q++){{
                __m512d re = _mm512_load_pd(CGA36 + (long)q*{PSZ} + e*16);
                __m512d im = _mm512_load_pd(CGA36 + (long)q*{PSZ} + e*16 + 8);
                _mm512_store_pd(dn + q*16, re); _mm512_store_pd(dn + q*16+8, im);
            }}
            for(int t=0;t<36;t++){{
                long q = PERM_36[t];
                _mm512_store_pd(dp + t*16, _mm512_load_pd(dn + q*16));
                _mm512_store_pd(dp + t*16+8, _mm512_load_pd(dn + q*16+8));
            }}
        }}
        SA_36(XA36, CGA36, CZPA36, 0);
        if(m == 1){{
            PA_36(XA36, CPA36, CPPA36, 1, 0);
            for(int v=0;v<8;v++) dsts[v] = out1 + (v0+v)*2*{L3};
            convout_36A(XA36, dsts, 8);
            for(int v=0;v<8;v++) dsts[v] = outm + (v0+v)*2*{L3};
            convout_36A(XA36, dsts, 8);
            continue;
        }}
        PA_36(XA36, CPA36, CPPA36, 3, SNA36);
        for(int v=0;v<8;v++) dsts[v] = out1 + (v0+v)*2*{L3};
        convout_36A(SNA36, dsts, 8);
        long t = 2;
        while(1){{
            SA_36(XA36, CGA36, CZPA36, t==m ? 1 : 2);
            if(t == m) break;
            t++;
            PA_36(XA36, CPA36, CPPA36, t==m ? 1 : 2, 0);
            if(t == m) break;
            t++;
        }}
        for(int v=0;v<8;v++) dsts[v] = outm + (v0+v)*2*{L3};
        convout_36A(XA36, dsts, 8);
    }}
}}
"""
