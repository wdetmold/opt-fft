// template: requires LL, N1, N2, NSLOT, YP, RS, PS, SUF
// storage: position q (0..LL-1) holds value index VAL(q) = (N2*n1 + N1*n2) % LL, q = n2*N1+n1
// z: position p at (slot p/8, lane p%8) within row; rows padded to YP; planes LL.

static double* SUF(X);      // state volume: LL planes x YP rows x RS doubles
static double* SUF(CS);     // c slab layout: LL*NYB blocks of CBS doubles
static double* SUF(CP);     // c pencil layout: LL*NSLOT blocks of CBS doubles
static int SUF(VAL)[LL];
static int SUF(IVAL)[LL];
static long SUF(OPOS)[N2];  // ((N1invmodN2)*k2 % N2) * N1
static long SUF(OB1)[N1];   // (N2invmodN1 * k1) % N1
static double SUF(SCP)[LL*16] ALIGN64;
static double SUF(ZS)[NSLOT*8*16 + 8] ALIGN64;

#define NYB (YP/8)
#define CBS (LL*16+16)

static void SUF(init_tabs)(void){
    for(int n2=0;n2<N2;n2++) for(int n1=0;n1<N1;n1++)
        SUF(VAL)[n2*N1+n1] = (N2*n1 + N1*n2) % LL;
    for(int q=0;q<LL;q++) SUF(IVAL)[SUF(VAL)[q]] = q;
    int n1inv=0, n2inv=0;
    for(int a=0;a<N2;a++) if((a*N1)%N2==1) n1inv=a;
    for(int a=0;a<N1;a++) if((a*N2)%N1==1) n2inv=a;
    for(int k2=0;k2<N2;k2++) SUF(OPOS)[k2] = (long)((n1inv*k2)%N2)*N1;
    for(int k1=0;k1<N1;k1++) SUF(OB1)[k1] = (n2inv*k1)%N1;
}

// ---------------- stage passes on a generic column ----------------
#if N1 == 5
#define DFTN1(xr,xi,yr,yi) DFT5(xr##0,xi##0,xr##1,xi##1,xr##2,xi##2,xr##3,xi##3,xr##4,xi##4,\
                                yr##0,yi##0,yr##1,yi##1,yr##2,yi##2,yr##3,yi##3,yr##4,yi##4)
#define DECLN1(p) __m512d p##0r,p##0i,p##1r,p##1i,p##2r,p##2i,p##3r,p##3i,p##4r,p##4i;
#else
#define DFTN1(xr,xi,yr,yi) DFT4(xr##0,xi##0,xr##1,xi##1,xr##2,xi##2,xr##3,xi##3,\
                                yr##0,yi##0,yr##1,yi##1,yr##2,yi##2,yr##3,yi##3)
#endif

// S1: groups n2: read positions n2*N1+n1 (contiguous), DFT_N1 -> SCP[k1*N2+n2]
__attribute__((noinline))
static void SUF(cp_s1)(const double* src, long st){
    for(int n2=0; n2<N2; n2++){
        const double* p = src + (long)n2*N1*st;
#if N1 == 5
        __m512d x0r=_mm512_load_pd(p), x0i=_mm512_load_pd(p+8);
        __m512d x1r=_mm512_load_pd(p+st), x1i=_mm512_load_pd(p+st+8);
        __m512d x2r=_mm512_load_pd(p+2*st), x2i=_mm512_load_pd(p+2*st+8);
        __m512d x3r=_mm512_load_pd(p+3*st), x3i=_mm512_load_pd(p+3*st+8);
        __m512d x4r=_mm512_load_pd(p+4*st), x4i=_mm512_load_pd(p+4*st+8);
        __m512d y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i,y4r,y4i;
        DFT5(x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,x4r,x4i,
             y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i,y4r,y4i);
        double* q = SUF(SCP) + n2*16;
        _mm512_store_pd(q+0*N2*16, y0r); _mm512_store_pd(q+0*N2*16+8, y0i);
        _mm512_store_pd(q+1*N2*16, y1r); _mm512_store_pd(q+1*N2*16+8, y1i);
        _mm512_store_pd(q+2*N2*16, y2r); _mm512_store_pd(q+2*N2*16+8, y2i);
        _mm512_store_pd(q+3*N2*16, y3r); _mm512_store_pd(q+3*N2*16+8, y3i);
        _mm512_store_pd(q+4*N2*16, y4r); _mm512_store_pd(q+4*N2*16+8, y4i);
#else
        __m512d x0r=_mm512_load_pd(p), x0i=_mm512_load_pd(p+8);
        __m512d x1r=_mm512_load_pd(p+st), x1i=_mm512_load_pd(p+st+8);
        __m512d x2r=_mm512_load_pd(p+2*st), x2i=_mm512_load_pd(p+2*st+8);
        __m512d x3r=_mm512_load_pd(p+3*st), x3i=_mm512_load_pd(p+3*st+8);
        __m512d y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i;
        DFT4(x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,
             y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i);
        double* q = SUF(SCP) + n2*16;
        _mm512_store_pd(q+0*N2*16, y0r); _mm512_store_pd(q+0*N2*16+8, y0i);
        _mm512_store_pd(q+1*N2*16, y1r); _mm512_store_pd(q+1*N2*16+8, y1i);
        _mm512_store_pd(q+2*N2*16, y2r); _mm512_store_pd(q+2*N2*16+8, y2i);
        _mm512_store_pd(q+3*N2*16, y3r); _mm512_store_pd(q+3*N2*16+8, y3i);
#endif
    }
}
// S1 with +c and map first (steady-state second half)
__attribute__((noinline))
static void SUF(cp_s1m)(const double* src, long st, const double* c){
    for(int n2=0; n2<N2; n2++){
        const double* p = src + (long)n2*N1*st;
        const double* cc = c + (long)n2*N1*16;
#if N1 == 5
        __m512d x0r=_mm512_add_pd(_mm512_load_pd(p),_mm512_load_pd(cc));
        __m512d x0i=_mm512_add_pd(_mm512_load_pd(p+8),_mm512_load_pd(cc+8));
        __m512d x1r=_mm512_add_pd(_mm512_load_pd(p+st),_mm512_load_pd(cc+16));
        __m512d x1i=_mm512_add_pd(_mm512_load_pd(p+st+8),_mm512_load_pd(cc+24));
        __m512d x2r=_mm512_add_pd(_mm512_load_pd(p+2*st),_mm512_load_pd(cc+32));
        __m512d x2i=_mm512_add_pd(_mm512_load_pd(p+2*st+8),_mm512_load_pd(cc+40));
        __m512d x3r=_mm512_add_pd(_mm512_load_pd(p+3*st),_mm512_load_pd(cc+48));
        __m512d x3i=_mm512_add_pd(_mm512_load_pd(p+3*st+8),_mm512_load_pd(cc+56));
        __m512d x4r=_mm512_add_pd(_mm512_load_pd(p+4*st),_mm512_load_pd(cc+64));
        __m512d x4i=_mm512_add_pd(_mm512_load_pd(p+4*st+8),_mm512_load_pd(cc+72));
        MAP2(x0r,x0i); MAP2(x1r,x1i); MAP2(x2r,x2i); MAP2(x3r,x3i); MAP2(x4r,x4i);
        __m512d y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i,y4r,y4i;
        DFT5(x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,x4r,x4i,
             y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i,y4r,y4i);
        double* q = SUF(SCP) + n2*16;
        _mm512_store_pd(q+0*N2*16, y0r); _mm512_store_pd(q+0*N2*16+8, y0i);
        _mm512_store_pd(q+1*N2*16, y1r); _mm512_store_pd(q+1*N2*16+8, y1i);
        _mm512_store_pd(q+2*N2*16, y2r); _mm512_store_pd(q+2*N2*16+8, y2i);
        _mm512_store_pd(q+3*N2*16, y3r); _mm512_store_pd(q+3*N2*16+8, y3i);
        _mm512_store_pd(q+4*N2*16, y4r); _mm512_store_pd(q+4*N2*16+8, y4i);
#else
        __m512d x0r=_mm512_add_pd(_mm512_load_pd(p),_mm512_load_pd(cc));
        __m512d x0i=_mm512_add_pd(_mm512_load_pd(p+8),_mm512_load_pd(cc+8));
        __m512d x1r=_mm512_add_pd(_mm512_load_pd(p+st),_mm512_load_pd(cc+16));
        __m512d x1i=_mm512_add_pd(_mm512_load_pd(p+st+8),_mm512_load_pd(cc+24));
        __m512d x2r=_mm512_add_pd(_mm512_load_pd(p+2*st),_mm512_load_pd(cc+32));
        __m512d x2i=_mm512_add_pd(_mm512_load_pd(p+2*st+8),_mm512_load_pd(cc+40));
        __m512d x3r=_mm512_add_pd(_mm512_load_pd(p+3*st),_mm512_load_pd(cc+48));
        __m512d x3i=_mm512_add_pd(_mm512_load_pd(p+3*st+8),_mm512_load_pd(cc+56));
        MAP2(x0r,x0i); MAP2(x1r,x1i); MAP2(x2r,x2i); MAP2(x3r,x3i);
        __m512d y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i;
        DFT4(x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,
             y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i);
        double* q = SUF(SCP) + n2*16;
        _mm512_store_pd(q+0*N2*16, y0r); _mm512_store_pd(q+0*N2*16+8, y0i);
        _mm512_store_pd(q+1*N2*16, y1r); _mm512_store_pd(q+1*N2*16+8, y1i);
        _mm512_store_pd(q+2*N2*16, y2r); _mm512_store_pd(q+2*N2*16+8, y2i);
        _mm512_store_pd(q+3*N2*16, y3r); _mm512_store_pd(q+3*N2*16+8, y3i);
#endif
    }
}

static double SUF(SCE)[(N1*N2+2)*16 + 8] ALIGN64;
__attribute__((noinline))
static void SUF(cp_s1m_pipe)(const double* src, long st, const double* c){
    for(int n2=0; n2<=N2; n2++){
        if(n2<N2){
            const double* p = src + (long)n2*N1*st;
            const double* cc = c + (long)n2*N1*16;
            double* qe = SUF(SCE) + (long)n2*N1*16;
            for(int j=0;j<N1;j++){
                __m512d zr=_mm512_add_pd(_mm512_load_pd(p+(long)j*st),   _mm512_load_pd(cc+j*16));
                __m512d zi=_mm512_add_pd(_mm512_load_pd(p+(long)j*st+8), _mm512_load_pd(cc+j*16+8));
                MAP2(zr,zi);
                _mm512_store_pd(qe+j*16, zr); _mm512_store_pd(qe+j*16+8, zi);
            }
        }
        if(n2>0){
            int n0=n2-1;
            const double* pe = SUF(SCE) + (long)n0*N1*16;
#if N1 == 5
            __m512d x0r=_mm512_load_pd(pe), x0i=_mm512_load_pd(pe+8);
            __m512d x1r=_mm512_load_pd(pe+16), x1i=_mm512_load_pd(pe+24);
            __m512d x2r=_mm512_load_pd(pe+32), x2i=_mm512_load_pd(pe+40);
            __m512d x3r=_mm512_load_pd(pe+48), x3i=_mm512_load_pd(pe+56);
            __m512d x4r=_mm512_load_pd(pe+64), x4i=_mm512_load_pd(pe+72);
            __m512d y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i,y4r,y4i;
            DFT5(x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,x4r,x4i,
                 y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i,y4r,y4i);
            double* q = SUF(SCP) + n0*16;
            _mm512_store_pd(q+0*N2*16, y0r); _mm512_store_pd(q+0*N2*16+8, y0i);
            _mm512_store_pd(q+1*N2*16, y1r); _mm512_store_pd(q+1*N2*16+8, y1i);
            _mm512_store_pd(q+2*N2*16, y2r); _mm512_store_pd(q+2*N2*16+8, y2i);
            _mm512_store_pd(q+3*N2*16, y3r); _mm512_store_pd(q+3*N2*16+8, y3i);
            _mm512_store_pd(q+4*N2*16, y4r); _mm512_store_pd(q+4*N2*16+8, y4i);
#else
            __m512d x0r=_mm512_load_pd(pe), x0i=_mm512_load_pd(pe+8);
            __m512d x1r=_mm512_load_pd(pe+16), x1i=_mm512_load_pd(pe+24);
            __m512d x2r=_mm512_load_pd(pe+32), x2i=_mm512_load_pd(pe+40);
            __m512d x3r=_mm512_load_pd(pe+48), x3i=_mm512_load_pd(pe+56);
            __m512d y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i;
            DFT4(x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,
                 y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i);
            double* q = SUF(SCP) + n0*16;
            _mm512_store_pd(q+0*N2*16, y0r); _mm512_store_pd(q+0*N2*16+8, y0i);
            _mm512_store_pd(q+1*N2*16, y1r); _mm512_store_pd(q+1*N2*16+8, y1i);
            _mm512_store_pd(q+2*N2*16, y2r); _mm512_store_pd(q+2*N2*16+8, y2i);
            _mm512_store_pd(q+3*N2*16, y3r); _mm512_store_pd(q+3*N2*16+8, y3i);
#endif
        }
    }
}
// S2: groups k1: read SCP[k1*N2 + n2], DFT_N2, write dst positions OPOS[k2]+OB1[k1]
__attribute__((noinline))
static void SUF(cp_s2)(double* dst, long st){
    for(int k1=0; k1<N1; k1++){
        const double* p = SUF(SCP) + (long)k1*N2*16;
        __m512d xr0=_mm512_load_pd(p+0*16),  xi0=_mm512_load_pd(p+0*16+8);
        __m512d xr1=_mm512_load_pd(p+1*16),  xi1=_mm512_load_pd(p+1*16+8);
        __m512d xr2=_mm512_load_pd(p+2*16),  xi2=_mm512_load_pd(p+2*16+8);
        __m512d xr3=_mm512_load_pd(p+3*16),  xi3=_mm512_load_pd(p+3*16+8);
        __m512d xr4=_mm512_load_pd(p+4*16),  xi4=_mm512_load_pd(p+4*16+8);
        __m512d xr5=_mm512_load_pd(p+5*16),  xi5=_mm512_load_pd(p+5*16+8);
        __m512d xr6=_mm512_load_pd(p+6*16),  xi6=_mm512_load_pd(p+6*16+8);
        __m512d xr7=_mm512_load_pd(p+7*16),  xi7=_mm512_load_pd(p+7*16+8);
        __m512d xr8=_mm512_load_pd(p+8*16),  xi8=_mm512_load_pd(p+8*16+8);
        __m512d yr0,yi0,yr1,yi1,yr2,yi2,yr3,yi3,yr4,yi4,yr5,yi5,yr6,yi6,yr7,yi7,yr8,yi8;
        { __m512d s0r,s0i,s1r,s1i,s2r,s2i, t0r,t0i,t1r,t1i,t2r,t2i, u0r,u0i,u1r,u1i,u2r,u2i;
          DFT3(xr0,xi0,xr3,xi3,xr6,xi6, s0r,s0i,s1r,s1i,s2r,s2i);
          DFT3(xr1,xi1,xr4,xi4,xr7,xi7, t0r,t0i,t1r,t1i,t2r,t2i);
          DFT3(xr2,xi2,xr5,xi5,xr8,xi8, u0r,u0i,u1r,u1i,u2r,u2i);
          CTW2(t1r,t1i, TW9[0][0], TW9[0][1]);
          CTW2(u1r,u1i, TW9[1][0], TW9[1][1]);
          CTW2(t2r,t2i, TW9[1][0], TW9[1][1]);
          CTW2(u2r,u2i, TW9[3][0], TW9[3][1]);
          DFT3(s0r,s0i,t0r,t0i,u0r,u0i, yr0,yi0,yr3,yi3,yr6,yi6);
          DFT3(s1r,s1i,t1r,t1i,u1r,u1i, yr1,yi1,yr4,yi4,yr7,yi7);
          DFT3(s2r,s2i,t2r,t2i,u2r,u2i, yr2,yi2,yr5,yi5,yr8,yi8);
        }
        double* d = dst + SUF(OB1)[k1]*st;
        _mm512_store_pd(d+SUF(OPOS)[0]*st, yr0); _mm512_store_pd(d+SUF(OPOS)[0]*st+8, yi0);
        _mm512_store_pd(d+SUF(OPOS)[1]*st, yr1); _mm512_store_pd(d+SUF(OPOS)[1]*st+8, yi1);
        _mm512_store_pd(d+SUF(OPOS)[2]*st, yr2); _mm512_store_pd(d+SUF(OPOS)[2]*st+8, yi2);
        _mm512_store_pd(d+SUF(OPOS)[3]*st, yr3); _mm512_store_pd(d+SUF(OPOS)[3]*st+8, yi3);
        _mm512_store_pd(d+SUF(OPOS)[4]*st, yr4); _mm512_store_pd(d+SUF(OPOS)[4]*st+8, yi4);
        _mm512_store_pd(d+SUF(OPOS)[5]*st, yr5); _mm512_store_pd(d+SUF(OPOS)[5]*st+8, yi5);
        _mm512_store_pd(d+SUF(OPOS)[6]*st, yr6); _mm512_store_pd(d+SUF(OPOS)[6]*st+8, yi6);
        _mm512_store_pd(d+SUF(OPOS)[7]*st, yr7); _mm512_store_pd(d+SUF(OPOS)[7]*st+8, yi7);
        _mm512_store_pd(d+SUF(OPOS)[8]*st, yr8); _mm512_store_pd(d+SUF(OPOS)[8]*st+8, yi8);
    }
}
// map-only pass over positions (strided): x = map(x + c)
__attribute__((noinline))
static void SUF(cp_map)(double* dst, long st, const double* c){
    for(int q=0; q<LL; q+=(LL%2?1:2)){
        __m512d zr=_mm512_add_pd(_mm512_load_pd(dst+q*st), _mm512_load_pd(c+q*16));
        __m512d zi=_mm512_add_pd(_mm512_load_pd(dst+q*st+8), _mm512_load_pd(c+q*16+8));
        MAP2(zr,zi);
        _mm512_store_pd(dst+q*st, zr); _mm512_store_pd(dst+q*st+8, zi);
        if(!(LL%2)){
            __m512d ar=_mm512_add_pd(_mm512_load_pd(dst+(q+1)*st), _mm512_load_pd(c+(q+1)*16));
            __m512d ai=_mm512_add_pd(_mm512_load_pd(dst+(q+1)*st+8), _mm512_load_pd(c+(q+1)*16+8));
            MAP2(ar,ai);
            _mm512_store_pd(dst+(q+1)*st, ar); _mm512_store_pd(dst+(q+1)*st+8, ai);
        }
    }
}

// ---------------- z axis via trin/trout ----------------
__attribute__((noinline))
static void SUF(trin)(const double* rb){
    for(int v=0; v<NSLOT; v++){
        __m512d a0,a1,a2,a3,a4,a5,a6,a7,b0,b1,b2,b3,b4,b5,b6,b7;
        a0=_mm512_load_pd(rb+0*RS+v*16); b0=_mm512_load_pd(rb+0*RS+v*16+8);
        a1=_mm512_load_pd(rb+1*RS+v*16); b1=_mm512_load_pd(rb+1*RS+v*16+8);
        a2=_mm512_load_pd(rb+2*RS+v*16); b2=_mm512_load_pd(rb+2*RS+v*16+8);
        a3=_mm512_load_pd(rb+3*RS+v*16); b3=_mm512_load_pd(rb+3*RS+v*16+8);
        a4=_mm512_load_pd(rb+4*RS+v*16); b4=_mm512_load_pd(rb+4*RS+v*16+8);
        a5=_mm512_load_pd(rb+5*RS+v*16); b5=_mm512_load_pd(rb+5*RS+v*16+8);
        a6=_mm512_load_pd(rb+6*RS+v*16); b6=_mm512_load_pd(rb+6*RS+v*16+8);
        a7=_mm512_load_pd(rb+7*RS+v*16); b7=_mm512_load_pd(rb+7*RS+v*16+8);
        TR8(a0,a1,a2,a3,a4,a5,a6,a7);
        TR8(b0,b1,b2,b3,b4,b5,b6,b7);
        double* z = SUF(ZS) + v*8*16;
        _mm512_store_pd(z+0*16, a0); _mm512_store_pd(z+0*16+8, b0);
        _mm512_store_pd(z+1*16, a1); _mm512_store_pd(z+1*16+8, b1);
        _mm512_store_pd(z+2*16, a2); _mm512_store_pd(z+2*16+8, b2);
        _mm512_store_pd(z+3*16, a3); _mm512_store_pd(z+3*16+8, b3);
        _mm512_store_pd(z+4*16, a4); _mm512_store_pd(z+4*16+8, b4);
        _mm512_store_pd(z+5*16, a5); _mm512_store_pd(z+5*16+8, b5);
        _mm512_store_pd(z+6*16, a6); _mm512_store_pd(z+6*16+8, b6);
        _mm512_store_pd(z+7*16, a7); _mm512_store_pd(z+7*16+8, b7);
    }
}
__attribute__((noinline))
static void SUF(trout)(double* rb){
    for(int v=0; v<NSLOT; v++){
        const double* z = SUF(ZS) + v*8*16;
        __m512d a0,a1,a2,a3,a4,a5,a6,a7,b0,b1,b2,b3,b4,b5,b6,b7;
        a0=_mm512_load_pd(z+0*16); b0=_mm512_load_pd(z+0*16+8);
        a1=_mm512_load_pd(z+1*16); b1=_mm512_load_pd(z+1*16+8);
        a2=_mm512_load_pd(z+2*16); b2=_mm512_load_pd(z+2*16+8);
        a3=_mm512_load_pd(z+3*16); b3=_mm512_load_pd(z+3*16+8);
        a4=_mm512_load_pd(z+4*16); b4=_mm512_load_pd(z+4*16+8);
        a5=_mm512_load_pd(z+5*16); b5=_mm512_load_pd(z+5*16+8);
        a6=_mm512_load_pd(z+6*16); b6=_mm512_load_pd(z+6*16+8);
        a7=_mm512_load_pd(z+7*16); b7=_mm512_load_pd(z+7*16+8);
        TR8(a0,a1,a2,a3,a4,a5,a6,a7);
        TR8(b0,b1,b2,b3,b4,b5,b6,b7);
        _mm512_store_pd(rb+0*RS+v*16, a0); _mm512_store_pd(rb+0*RS+v*16+8, b0);
        _mm512_store_pd(rb+1*RS+v*16, a1); _mm512_store_pd(rb+1*RS+v*16+8, b1);
        _mm512_store_pd(rb+2*RS+v*16, a2); _mm512_store_pd(rb+2*RS+v*16+8, b2);
        _mm512_store_pd(rb+3*RS+v*16, a3); _mm512_store_pd(rb+3*RS+v*16+8, b3);
        _mm512_store_pd(rb+4*RS+v*16, a4); _mm512_store_pd(rb+4*RS+v*16+8, b4);
        _mm512_store_pd(rb+5*RS+v*16, a5); _mm512_store_pd(rb+5*RS+v*16+8, b5);
        _mm512_store_pd(rb+6*RS+v*16, a6); _mm512_store_pd(rb+6*RS+v*16+8, b6);
        _mm512_store_pd(rb+7*RS+v*16, a7); _mm512_store_pd(rb+7*RS+v*16+8, b7);
    }
}

// ---------------- sweeps ----------------
// SB modes: 0 = FyFz plain; 1 = steady; 2 = final
static void SUF(SB)(int mode){
    for(int i=0;i<LL;i++){
        double* pl = SUF(X) + (long)i*PS;
        for(int v=0; v<NSLOT; v++){ SUF(cp_s1)(pl + v*16, RS); SUF(cp_s2)(pl + v*16, RS); }
        for(int yb=0; yb<NYB; yb++){
            double* rb = pl + (long)yb*8*RS;
            SUF(trin)(rb);
            SUF(cp_s1)(SUF(ZS), 16); SUF(cp_s2)(SUF(ZS), 16);
            if(mode==1){
                SUF(cp_s1m)(SUF(ZS), 16, SUF(CS) + ((long)i*NYB+yb)*CBS);
                SUF(cp_s2)(SUF(ZS), 16);
            } else if(mode==2){
                SUF(cp_map)(SUF(ZS), 16, SUF(CS) + ((long)i*NYB+yb)*CBS);
            }
            SUF(trout)(rb);
        }
        if(mode==1){
            for(int v=0; v<NSLOT; v++){ SUF(cp_s1)(pl + v*16, RS); SUF(cp_s2)(pl + v*16, RS); }
        }
    }
}
// PB modes: 0 = Fx plain; 1 = Fx,+c,map plain; 2 = steady
static void SUF(PB)(int mode){
    for(int y=0;y<LL;y++){
        for(int v=0;v<NSLOT;v++){
            double* p = SUF(X) + (long)y*RS + v*16;
            SUF(cp_s1)(p, PS); SUF(cp_s2)(p, PS);
            if(mode==1){ SUF(cp_map)(p, PS, SUF(CP) + ((long)y*NSLOT+v)*CBS); }
            else if(mode==2){
                SUF(cp_s1m)(p, PS, SUF(CP) + ((long)y*NSLOT+v)*CBS);
                SUF(cp_s2)(p, PS);
            }
        }
    }
}

// ---------------- conversions ----------------
static void SUF(convin)(const double* in, double* X){
    for(int ip=0;ip<LL;ip++){
        int xv=SUF(VAL)[ip];
        for(int yp=0;yp<LL;yp++){
            int yv=SUF(VAL)[yp];
            const double* s = in + 2*(((long)xv*LL+yv)*LL);
            double* d = X + (long)ip*PS + (long)yp*RS;
            for(int zp=0;zp<LL;zp++){
                int zv=SUF(VAL)[zp];
                d[(zp/8)*16 + (zp%8)]     = s[2*zv];
                d[(zp/8)*16 + 8 + (zp%8)] = s[2*zv+1];
            }
        }
    }
}
static void SUF(convout)(const double* X, double* out){
    for(int ip=0;ip<LL;ip++){
        int xv=SUF(VAL)[ip];
        for(int yp=0;yp<LL;yp++){
            int yv=SUF(VAL)[yp];
            double* s = out + 2*(((long)xv*LL+yv)*LL);
            const double* d = X + (long)ip*PS + (long)yp*RS;
            for(int zp=0;zp<LL;zp++){
                int zv=SUF(VAL)[zp];
                s[2*zv]   = d[(zp/8)*16 + (zp%8)];
                s[2*zv+1] = d[(zp/8)*16 + 8 + (zp%8)];
            }
        }
    }
}
static void SUF(build_cs)(const double* c, double* CS){
    for(int i=0;i<LL;i++){
        int xv=SUF(VAL)[i];
        for(int yb=0;yb<NYB;yb++){
            double* blk = CS + ((long)i*NYB+yb)*CBS;
            for(int q=0;q<LL;q++){
                int zv=SUF(VAL)[q];
                for(int l=0;l<8;l++){
                    int yy=8*yb+l;
                    if(yy<LL){
                        int yv=SUF(VAL)[yy];
                        const double* s = c + 2*(((long)xv*LL+yv)*LL+zv);
                        blk[q*16+l]=s[0]; blk[q*16+8+l]=s[1];
                    } else { blk[q*16+l]=0.0; blk[q*16+8+l]=0.0; }
                }
            }
        }
    }
}
static void SUF(build_cp)(const double* c, double* CP){
    for(int y=0;y<LL;y++){
        int yv=SUF(VAL)[y];
        for(int v=0;v<NSLOT;v++){
            double* blk = CP + ((long)y*NSLOT+v)*CBS;
            for(int q=0;q<LL;q++){
                int xv=SUF(VAL)[q];
                for(int l=0;l<8;l++){
                    int zz=8*v+l;
                    if(zz<LL){
                        int zv=SUF(VAL)[zz];
                        const double* s = c + 2*(((long)xv*LL+yv)*LL+zv);
                        blk[q*16+l]=s[0]; blk[q*16+8+l]=s[1];
                    } else { blk[q*16+l]=0.0; blk[q*16+8+l]=0.0; }
                }
            }
        }
    }
}

static void SUF(init)(void){
    if(SUF(X)) return;
    init_tw9();
    SUF(init_tabs)();
    SUF(X)  = alloc_big2((long)LL*PS*8);
    SUF(CS) = alloc_big2((long)LL*NYB*CBS*8);
    SUF(CP) = alloc_big2((long)LL*NSLOT*CBS*8);
}

#define L3SZ ((long)LL*LL*LL)
void SUF(run)(const double* x0, const double* c, double* out1, double* outm, long B, long m){
    SUF(init)();
    if(m<1) m=1;
    for(long b=0;b<B;b++){
        SUF(convin)(x0 + b*2*L3SZ, SUF(X));
        SUF(build_cs)(c + b*2*L3SZ, SUF(CS));
        SUF(build_cp)(c + b*2*L3SZ, SUF(CP));
        SUF(SB)(0);
        SUF(PB)(1);
        SUF(convout)(SUF(X), out1 + b*2*L3SZ);
        if(m==1){ memcpy(outm + b*2*L3SZ, out1 + b*2*L3SZ, L3SZ*16); continue; }
        SUF(PB)(0);
        for(long t=2;t<=m;t++){
            if((t&1)==0){ SUF(SB)(t==m ? 2 : 1); }
            else        { SUF(PB)(t==m ? 1 : 2); }
        }
        SUF(convout)(SUF(X), outm + b*2*L3SZ);
    }
}
void SUF(t_round)(const double* in, double* out){ SUF(init)(); SUF(convin)(in, SUF(X)); SUF(convout)(SUF(X), out); }
void SUF(t_z)(const double* in, double* out){
    SUF(init)(); SUF(convin)(in, SUF(X));
    for(int i=0;i<LL;i++){ double* pl=SUF(X)+(long)i*PS;
        for(int yb=0;yb<NYB;yb++){ double* rb=pl+(long)yb*8*RS;
            SUF(trin)(rb); SUF(cp_s1)(SUF(ZS),16); SUF(cp_s2)(SUF(ZS),16); SUF(trout)(rb); } }
    SUF(convout)(SUF(X), out);
}
void SUF(t_tr)(const double* in, double* out){
    SUF(init)(); SUF(convin)(in, SUF(X));
    for(int i=0;i<LL;i++){ double* pl=SUF(X)+(long)i*PS;
        for(int yb=0;yb<NYB;yb++){ double* rb=pl+(long)yb*8*RS;
            SUF(trin)(rb); SUF(trout)(rb); } }
    SUF(convout)(SUF(X), out);
}
void SUF(t_y)(const double* in, double* out){
    SUF(init)(); SUF(convin)(in, SUF(X));
    for(int i=0;i<LL;i++){ double* pl=SUF(X)+(long)i*PS;
        for(int v=0;v<NSLOT;v++){ SUF(cp_s1)(pl+v*16,RS); SUF(cp_s2)(pl+v*16,RS); } }
    SUF(convout)(SUF(X), out);
}
void SUF(t_x)(const double* in, double* out){
    SUF(init)(); SUF(convin)(in, SUF(X)); SUF(PB)(0); SUF(convout)(SUF(X), out);
}
#undef NYB
#undef CBS
#undef L3SZ
#undef DFTN1
#undef DECLN1
