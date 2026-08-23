// ---------------------------------------------------------------------------
// 8x8 transpose of doubles (r0..r7 -> o0..o7)
// ---------------------------------------------------------------------------
#define TR8(o0,o1,o2,o3,o4,o5,o6,o7, r0,r1,r2,r3,r4,r5,r6,r7) do { \
    VD _t0=_mm512_unpacklo_pd(r0,r1), _t1=_mm512_unpackhi_pd(r0,r1); \
    VD _t2=_mm512_unpacklo_pd(r2,r3), _t3=_mm512_unpackhi_pd(r2,r3); \
    VD _t4=_mm512_unpacklo_pd(r4,r5), _t5=_mm512_unpackhi_pd(r4,r5); \
    VD _t6=_mm512_unpacklo_pd(r6,r7), _t7=_mm512_unpackhi_pd(r6,r7); \
    VD _u0=_mm512_shuffle_f64x2(_t0,_t2,0x88), _u1=_mm512_shuffle_f64x2(_t1,_t3,0x88); \
    VD _u2=_mm512_shuffle_f64x2(_t0,_t2,0xdd), _u3=_mm512_shuffle_f64x2(_t1,_t3,0xdd); \
    VD _u4=_mm512_shuffle_f64x2(_t4,_t6,0x88), _u5=_mm512_shuffle_f64x2(_t5,_t7,0x88); \
    VD _u6=_mm512_shuffle_f64x2(_t4,_t6,0xdd), _u7=_mm512_shuffle_f64x2(_t5,_t7,0xdd); \
    o0=_mm512_shuffle_f64x2(_u0,_u4,0x88); o4=_mm512_shuffle_f64x2(_u0,_u4,0xdd); \
    o1=_mm512_shuffle_f64x2(_u1,_u5,0x88); o5=_mm512_shuffle_f64x2(_u1,_u5,0xdd); \
    o2=_mm512_shuffle_f64x2(_u2,_u6,0x88); o6=_mm512_shuffle_f64x2(_u2,_u6,0xdd); \
    o3=_mm512_shuffle_f64x2(_u3,_u7,0x88); o7=_mm512_shuffle_f64x2(_u3,_u7,0xdd); } while(0)

// ---------------------------------------------------------------------------
// Elementwise map: given y (=DFT result) vectors and c pointers,
// z = y + c ; out = z / (1 + |z|), stored masked at (pr,pi).
// Needs in scope: Vhalf(0.5), V1p5(1.5), Vone(1.0), Vtiny(2.3e-308)
// ---------------------------------------------------------------------------
#define MAPST(pr, pi, mm, yr, yi, pcr, pci) do { \
    VD _zr = VADD(yr, LD(pcr)), _zi = VADD(yi, LD(pci)); \
    VD _s  = VFMA(_zi,_zi, VMUL(_zr,_zr)); \
    _s = VMAX(_s, Vtiny); \
    VD _t = _mm512_rsqrt14_pd(_s); \
    VD _hs = VMUL(Vhalf,_s); \
    _t = VMUL(_t, VFNMA(_hs, VMUL(_t,_t), V1p5)); \
    _t = VMUL(_t, VFNMA(_hs, VMUL(_t,_t), V1p5)); \
    VD _d = VFMA(_s,_t,Vone); \
    VD _q = _mm512_div_pd(Vone, _d); \
    MST(pr, mm, VMUL(_zr,_q)); MST(pi, mm, VMUL(_zi,_q)); } while(0)

// ---------------------------------------------------------------------------
// KM bodies: generic over LOADR/LOADI(j) and STOREP(k, vr, vi)
// ---------------------------------------------------------------------------
#define KM6_BODY(LOADR, LOADI, STOREP) do { \
    const VD Vhalf=BC(0.5), Vs3=BC(g_s3half); \
    VD x0r=LOADR(0),x0i=LOADI(0),x1r=LOADR(1),x1i=LOADI(1),x2r=LOADR(2),x2i=LOADI(2); \
    VD x3r=LOADR(3),x3i=LOADI(3),x4r=LOADR(4),x4i=LOADI(4),x5r=LOADR(5),x5i=LOADI(5); \
    VD A00r,A00i,A01r,A01i,A02r,A02i, A10r,A10i,A11r,A11i,A12r,A12i; \
    DFT3V(A00r,A00i,A01r,A01i,A02r,A02i, x0r,x0i,x2r,x2i,x4r,x4i); \
    DFT3V(A10r,A10i,A11r,A11i,A12r,A12i, x3r,x3i,x5r,x5i,x1r,x1i); \
    STOREP(0, VADD(A00r,A10r), VADD(A00i,A10i)); \
    STOREP(3, VSUB(A00r,A10r), VSUB(A00i,A10i)); \
    STOREP(4, VADD(A01r,A11r), VADD(A01i,A11i)); \
    STOREP(1, VSUB(A01r,A11r), VSUB(A01i,A11i)); \
    STOREP(2, VADD(A02r,A12r), VADD(A02i,A12i)); \
    STOREP(5, VSUB(A02r,A12r), VSUB(A02i,A12i)); } while(0)

#define KM8_BODY(LOADR, LOADI, STOREP) do { \
    const VD Vr2=BC(g_r2half); \
    VD x0r=LOADR(0),x0i=LOADI(0),x1r=LOADR(1),x1i=LOADI(1),x2r=LOADR(2),x2i=LOADI(2),x3r=LOADR(3),x3i=LOADI(3); \
    VD x4r=LOADR(4),x4i=LOADI(4),x5r=LOADR(5),x5i=LOADI(5),x6r=LOADR(6),x6i=LOADI(6),x7r=LOADR(7),x7i=LOADI(7); \
    VD o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i,o4r,o4i,o5r,o5i,o6r,o6i,o7r,o7i; \
    DFT8V(o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i,o4r,o4i,o5r,o5i,o6r,o6i,o7r,o7i, \
          x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,x4r,x4i,x5r,x5i,x6r,x6i,x7r,x7i); \
    STOREP(0,o0r,o0i); STOREP(1,o1r,o1i); STOREP(2,o2r,o2i); STOREP(3,o3r,o3i); \
    STOREP(4,o4r,o4i); STOREP(5,o5r,o5i); STOREP(6,o6r,o6i); STOREP(7,o7r,o7i); } while(0)

static const int P36IN[4][9] = {
 {0,4,8,12,16,20,24,28,32},{9,13,17,21,25,29,33,1,5},
 {18,22,26,30,34,2,6,10,14},{27,31,35,3,7,11,15,19,23}};
static const int P36OUT[4][9] = {
 {0,28,20,12,4,32,24,16,8},{9,1,29,21,13,5,33,25,17},
 {18,10,2,30,22,14,6,34,26},{27,19,11,3,31,23,15,7,35}};

static const int P36IN_S36[4][9] = {
 {0,144,288,432,576,720,864,1008,1152},
 {324,468,612,756,900,1044,1188,36,180},
 {648,792,936,1080,1224,72,216,360,504},
 {972,1116,1260,108,252,396,540,684,828}};
static const int P36OUT_S36[4][9] = {
 {0,1008,720,432,144,1152,864,576,288},
 {324,36,1044,756,468,180,1188,900,612},
 {648,360,72,1080,792,504,216,1224,936},
 {972,684,396,108,1116,828,540,252,1260}};
static const int P36IN_S8[4][9] = {
 {0,32,64,96,128,160,192,224,256},
 {72,104,136,168,200,232,264,8,40},
 {144,176,208,240,272,16,48,80,112},
 {216,248,280,24,56,88,120,152,184}};
static const int P36OUT_S8[4][9] = {
 {0,224,160,96,32,256,192,128,64},
 {72,8,232,168,104,40,264,200,136},
 {144,80,16,240,176,112,48,272,208},
 {216,152,88,24,248,184,120,56,280}};
static const int P36IN_S1296[4][9] = {
 {0,5184,10368,15552,20736,25920,31104,36288,41472},
 {11664,16848,22032,27216,32400,37584,42768,1296,6480},
 {23328,28512,33696,38880,44064,2592,7776,12960,18144},
 {34992,40176,45360,3888,9072,14256,19440,24624,29808}};
static const int P36OUT_S1296[4][9] = {
 {0,36288,25920,15552,5184,41472,31104,20736,10368},
 {11664,1296,37584,27216,16848,6480,42768,32400,22032},
 {23328,12960,2592,38880,28512,18144,7776,44064,33696},
 {34992,24624,14256,3888,40176,29808,19440,9072,45360}};
static const int P45IN_S45[5][9] = {
 {0,225,450,675,900,1125,1350,1575,1800},
 {405,630,855,1080,1305,1530,1755,1980,180},
 {810,1035,1260,1485,1710,1935,135,360,585},
 {1215,1440,1665,1890,90,315,540,765,990},
 {1620,1845,45,270,495,720,945,1170,1395}};
static const int P45OUT_S45[5][9] = {
 {0,450,900,1350,1800,225,675,1125,1575},
 {1620,45,495,945,1395,1845,270,720,1170},
 {1215,1665,90,540,990,1440,1890,315,765},
 {810,1260,1710,135,585,1035,1485,1935,360},
 {405,855,1305,1755,180,630,1080,1530,1980}};
static const int P45IN_S8[5][9] = {
 {0,40,80,120,160,200,240,280,320},
 {72,112,152,192,232,272,312,352,32},
 {144,184,224,264,304,344,24,64,104},
 {216,256,296,336,16,56,96,136,176},
 {288,328,8,48,88,128,168,208,248}};
static const int P45OUT_S8[5][9] = {
 {0,80,160,240,320,40,120,200,280},
 {288,8,88,168,248,328,48,128,208},
 {216,296,16,96,176,256,336,56,136},
 {144,224,304,24,104,184,264,344,64},
 {72,152,232,312,32,112,192,272,352}};
static const int P45IN_S2025[5][9] = {
 {0,10125,20250,30375,40500,50625,60750,70875,81000},
 {18225,28350,38475,48600,58725,68850,78975,89100,8100},
 {36450,46575,56700,66825,76950,87075,6075,16200,26325},
 {54675,64800,74925,85050,4050,14175,24300,34425,44550},
 {72900,83025,2025,12150,22275,32400,42525,52650,62775}};
static const int P45OUT_S2025[5][9] = {
 {0,20250,40500,60750,81000,10125,30375,50625,70875},
 {72900,2025,22275,42525,62775,83025,12150,32400,52650},
 {54675,74925,4050,24300,44550,64800,85050,14175,34425},
 {36450,56700,76950,6075,26325,46575,66825,87075,16200},
 {18225,38475,58725,78975,8100,28350,48600,68850,89100}};

#define KM36_BODY(LOADR, LOADI, STOREP, PIN, POUT) do { \
    const VD Vhalf=BC(0.5), Vs3=BC(g_s3half); \
    const VD Vw91r=BC(g_w9[0]),Vw91i=BC(g_w9[1]),Vw92r=BC(g_w9[2]),Vw92i=BC(g_w9[3]),Vw94r=BC(g_w9[4]),Vw94i=BC(g_w9[5]); \
    VD Ar[4][9], Ai[4][9]; \
    for (int j4 = 0; j4 < 4; j4++) { \
        const int *pin = (PIN)[j4]; \
        VD y0r=LOADR(pin[0]),y0i=LOADI(pin[0]),y1r=LOADR(pin[1]),y1i=LOADI(pin[1]),y2r=LOADR(pin[2]),y2i=LOADI(pin[2]); \
        VD y3r=LOADR(pin[3]),y3i=LOADI(pin[3]),y4r=LOADR(pin[4]),y4i=LOADI(pin[4]),y5r=LOADR(pin[5]),y5i=LOADI(pin[5]); \
        VD y6r=LOADR(pin[6]),y6i=LOADI(pin[6]),y7r=LOADR(pin[7]),y7i=LOADI(pin[7]),y8r=LOADR(pin[8]),y8i=LOADI(pin[8]); \
        DFT9V(Ar[j4][0],Ai[j4][0],Ar[j4][1],Ai[j4][1],Ar[j4][2],Ai[j4][2],Ar[j4][3],Ai[j4][3],Ar[j4][4],Ai[j4][4], \
              Ar[j4][5],Ai[j4][5],Ar[j4][6],Ai[j4][6],Ar[j4][7],Ai[j4][7],Ar[j4][8],Ai[j4][8], \
              y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i,y4r,y4i,y5r,y5i,y6r,y6i,y7r,y7i,y8r,y8i); \
    } \
    for (int k9 = 0; k9 < 9; k9++) { \
        VD o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i; \
        DFT4V(o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i, \
              Ar[0][k9],Ai[0][k9],Ar[1][k9],Ai[1][k9],Ar[2][k9],Ai[2][k9],Ar[3][k9],Ai[3][k9]); \
        STOREP((POUT)[0][k9], o0r, o0i); \
        STOREP((POUT)[1][k9], o1r, o1i); \
        STOREP((POUT)[2][k9], o2r, o2i); \
        STOREP((POUT)[3][k9], o3r, o3i); \
    } } while(0)

static const int P45IN[5][9] = {
 {0,5,10,15,20,25,30,35,40},{9,14,19,24,29,34,39,44,4},
 {18,23,28,33,38,43,3,8,13},{27,32,37,42,2,7,12,17,22},{36,41,1,6,11,16,21,26,31}};
static const int P45OUT[5][9] = {
 {0,10,20,30,40,5,15,25,35},{36,1,11,21,31,41,6,16,26},
 {27,37,2,12,22,32,42,7,17},{18,28,38,3,13,23,33,43,8},{9,19,29,39,4,14,24,34,44}};

#define KM45_BODY(LOADR, LOADI, STOREP, PIN, POUT) do { \
    const VD Vhalf=BC(0.5), Vs3=BC(g_s3half); \
    const VD Vw91r=BC(g_w9[0]),Vw91i=BC(g_w9[1]),Vw92r=BC(g_w9[2]),Vw92i=BC(g_w9[3]),Vw94r=BC(g_w9[4]),Vw94i=BC(g_w9[5]); \
    const VD Vc51=BC(g_c51),Vc52=BC(g_c52),Vs51=BC(g_s51),Vs52=BC(g_s52); \
    VD Ar[5][9], Ai[5][9]; \
    for (int j5 = 0; j5 < 5; j5++) { \
        const int *pin = (PIN)[j5]; \
        VD y0r=LOADR(pin[0]),y0i=LOADI(pin[0]),y1r=LOADR(pin[1]),y1i=LOADI(pin[1]),y2r=LOADR(pin[2]),y2i=LOADI(pin[2]); \
        VD y3r=LOADR(pin[3]),y3i=LOADI(pin[3]),y4r=LOADR(pin[4]),y4i=LOADI(pin[4]),y5r=LOADR(pin[5]),y5i=LOADI(pin[5]); \
        VD y6r=LOADR(pin[6]),y6i=LOADI(pin[6]),y7r=LOADR(pin[7]),y7i=LOADI(pin[7]),y8r=LOADR(pin[8]),y8i=LOADI(pin[8]); \
        DFT9V(Ar[j5][0],Ai[j5][0],Ar[j5][1],Ai[j5][1],Ar[j5][2],Ai[j5][2],Ar[j5][3],Ai[j5][3],Ar[j5][4],Ai[j5][4], \
              Ar[j5][5],Ai[j5][5],Ar[j5][6],Ai[j5][6],Ar[j5][7],Ai[j5][7],Ar[j5][8],Ai[j5][8], \
              y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i,y4r,y4i,y5r,y5i,y6r,y6i,y7r,y7i,y8r,y8i); \
    } \
    for (int k9 = 0; k9 < 9; k9++) { \
        VD o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i,o4r,o4i; \
        DFT5V(o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i,o4r,o4i, \
              Ar[0][k9],Ai[0][k9],Ar[1][k9],Ai[1][k9],Ar[2][k9],Ai[2][k9],Ar[3][k9],Ai[3][k9],Ar[4][k9],Ai[4][k9]); \
        STOREP((POUT)[0][k9], o0r, o0i); \
        STOREP((POUT)[1][k9], o1r, o1i); \
        STOREP((POUT)[2][k9], o2r, o2i); \
        STOREP((POUT)[3][k9], o3r, o3i); \
        STOREP((POUT)[4][k9], o4r, o4i); \
    } } while(0)

#define KM64_BODY(LOADR, LOADI, STOREP) do { \
    const VD Vr2=BC(g_r2half); \
    VD BufR[64], BufI[64]; \
    for (int b = 0; b < 8; b++) { \
        VD x0r=LOADR(b),x0i=LOADI(b),x1r=LOADR(8+b),x1i=LOADI(8+b),x2r=LOADR(16+b),x2i=LOADI(16+b),x3r=LOADR(24+b),x3i=LOADI(24+b); \
        VD x4r=LOADR(32+b),x4i=LOADI(32+b),x5r=LOADR(40+b),x5i=LOADI(40+b),x6r=LOADR(48+b),x6i=LOADI(48+b),x7r=LOADR(56+b),x7i=LOADI(56+b); \
        VD F0r,F0i,F1r,F1i,F2r,F2i,F3r,F3i,F4r,F4i,F5r,F5i,F6r,F6i,F7r,F7i; \
        DFT8V(F0r,F0i,F1r,F1i,F2r,F2i,F3r,F3i,F4r,F4i,F5r,F5i,F6r,F6i,F7r,F7i, \
              x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,x4r,x4i,x5r,x5i,x6r,x6i,x7r,x7i); \
        BufR[0*8+b]=F0r; BufI[0*8+b]=F0i; \
        if (b == 0) { \
            BufR[1*8]=F1r; BufI[1*8]=F1i; BufR[2*8]=F2r; BufI[2*8]=F2i; BufR[3*8]=F3r; BufI[3*8]=F3i; \
            BufR[4*8]=F4r; BufI[4*8]=F4i; BufR[5*8]=F5r; BufI[5*8]=F5i; BufR[6*8]=F6r; BufI[6*8]=F6i; BufR[7*8]=F7r; BufI[7*8]=F7i; \
        } else { \
            VD gr,gi; \
            CMULV(gr,gi,F1r,F1i,BC(g_T64r[b][1]),BC(g_T64i[b][1])); BufR[1*8+b]=gr; BufI[1*8+b]=gi; \
            CMULV(gr,gi,F2r,F2i,BC(g_T64r[b][2]),BC(g_T64i[b][2])); BufR[2*8+b]=gr; BufI[2*8+b]=gi; \
            CMULV(gr,gi,F3r,F3i,BC(g_T64r[b][3]),BC(g_T64i[b][3])); BufR[3*8+b]=gr; BufI[3*8+b]=gi; \
            CMULV(gr,gi,F4r,F4i,BC(g_T64r[b][4]),BC(g_T64i[b][4])); BufR[4*8+b]=gr; BufI[4*8+b]=gi; \
            CMULV(gr,gi,F5r,F5i,BC(g_T64r[b][5]),BC(g_T64i[b][5])); BufR[5*8+b]=gr; BufI[5*8+b]=gi; \
            CMULV(gr,gi,F6r,F6i,BC(g_T64r[b][6]),BC(g_T64i[b][6])); BufR[6*8+b]=gr; BufI[6*8+b]=gi; \
            CMULV(gr,gi,F7r,F7i,BC(g_T64r[b][7]),BC(g_T64i[b][7])); BufR[7*8+b]=gr; BufI[7*8+b]=gi; \
        } \
    } \
    for (int t = 0; t < 8; t++) { \
        VD H0r,H0i,H1r,H1i,H2r,H2i,H3r,H3i,H4r,H4i,H5r,H5i,H6r,H6i,H7r,H7i; \
        DFT8V(H0r,H0i,H1r,H1i,H2r,H2i,H3r,H3i,H4r,H4i,H5r,H5i,H6r,H6i,H7r,H7i, \
              BufR[t*8+0],BufI[t*8+0],BufR[t*8+1],BufI[t*8+1],BufR[t*8+2],BufI[t*8+2],BufR[t*8+3],BufI[t*8+3], \
              BufR[t*8+4],BufI[t*8+4],BufR[t*8+5],BufI[t*8+5],BufR[t*8+6],BufI[t*8+6],BufR[t*8+7],BufI[t*8+7]); \
        STOREP(t,    H0r,H0i); STOREP(t+ 8, H1r,H1i); STOREP(t+16, H2r,H2i); STOREP(t+24, H3r,H3i); \
        STOREP(t+32, H4r,H4i); STOREP(t+40, H5r,H5i); STOREP(t+48, H6r,H6i); STOREP(t+56, H7r,H7i); \
    } } while(0)

// Folded odd-prime body with k-pair unrolling for ILP.
#define KMFOLD_BODY(LL, HH, TC, TS, LOADR, LOADI, STOREP) do { \
    VD ur[HH+1], ui[HH+1], vr[HH+1], vi[HH+1]; \
    VD x0r = LOADR(0), x0i = LOADI(0); \
    VD sur0 = x0r, sui0 = x0i, sur1 = ZERO, sui1 = ZERO; \
    _Pragma("GCC unroll 16") \
    for (int j = 1; j <= HH; j++) { \
        VD ar = LOADR(j), ai = LOADI(j); \
        VD br = LOADR(LL-j), bi = LOADI(LL-j); \
        ur[j]=VADD(ar,br); ui[j]=VADD(ai,bi); \
        vr[j]=VSUB(ar,br); vi[j]=VSUB(ai,bi); \
        if (j & 1) { sur0=VADD(sur0,ur[j]); sui0=VADD(sui0,ui[j]); } \
        else       { sur1=VADD(sur1,ur[j]); sui1=VADD(sui1,ui[j]); } \
    } \
    STOREP(0, VADD(sur0,sur1), VADD(sui0,sui1)); \
    _Pragma("GCC unroll 16") \
    for (int k = 1; k + 1 <= HH; k += 2) { \
        VD P1r = x0r, P1i = x0i, Q1r = ZERO, Q1i = ZERO; \
        VD P2r = x0r, P2i = x0i, Q2r = ZERO, Q2i = ZERO; \
        _Pragma("GCC unroll 16") \
        for (int j = 1; j <= HH; j++) { \
            VD c1 = BC(TC[(k-1)*HH + (j-1)]), s1 = BC(TS[(k-1)*HH + (j-1)]); \
            VD c2 = BC(TC[k*HH + (j-1)]),     s2 = BC(TS[k*HH + (j-1)]); \
            VD uurr = ur[j], uuii = ui[j], vvrr = vr[j], vvii = vi[j]; \
            P1r = VFMA(c1, uurr, P1r); P1i = VFMA(c1, uuii, P1i); \
            Q1r = VFMA(s1, vvrr, Q1r); Q1i = VFMA(s1, vvii, Q1i); \
            P2r = VFMA(c2, uurr, P2r); P2i = VFMA(c2, uuii, P2i); \
            Q2r = VFMA(s2, vvrr, Q2r); Q2i = VFMA(s2, vvii, Q2i); \
        } \
        STOREP(k,      VADD(P1r,Q1i), VSUB(P1i,Q1r)); \
        STOREP(LL-k,   VSUB(P1r,Q1i), VADD(P1i,Q1r)); \
        STOREP(k+1,    VADD(P2r,Q2i), VSUB(P2i,Q2r)); \
        STOREP(LL-k-1, VSUB(P2r,Q2i), VADD(P2i,Q2r)); \
    } \
    if (HH & 1) { \
        const int k = HH; \
        VD P1r = x0r, P1i = x0i, Q1r = ZERO, Q1i = ZERO; \
        VD P2r = ZERO, P2i = ZERO, Q2r = ZERO, Q2i = ZERO; \
        _Pragma("GCC unroll 16") \
        for (int j = 1; j + 1 <= HH; j += 2) { \
            VD c1 = BC(TC[(k-1)*HH + (j-1)]), s1 = BC(TS[(k-1)*HH + (j-1)]); \
            VD c2 = BC(TC[(k-1)*HH + j]),     s2 = BC(TS[(k-1)*HH + j]); \
            P1r = VFMA(c1, ur[j], P1r); P1i = VFMA(c1, ui[j], P1i); \
            Q1r = VFMA(s1, vr[j], Q1r); Q1i = VFMA(s1, vi[j], Q1i); \
            P2r = VFMA(c2, ur[j+1], P2r); P2i = VFMA(c2, ui[j+1], P2i); \
            Q2r = VFMA(s2, vr[j+1], Q2r); Q2i = VFMA(s2, vi[j+1], Q2i); \
        } \
        { \
            VD c1 = BC(TC[(k-1)*HH + (HH-1)]), s1 = BC(TS[(k-1)*HH + (HH-1)]); \
            P1r = VFMA(c1, ur[HH], P1r); P1i = VFMA(c1, ui[HH], P1i); \
            Q1r = VFMA(s1, vr[HH], Q1r); Q1i = VFMA(s1, vi[HH], Q1i); \
        } \
        P1r = VADD(P1r,P2r); P1i = VADD(P1i,P2i); \
        Q1r = VADD(Q1r,Q2r); Q1i = VADD(Q1i,Q2i); \
        STOREP(k,    VADD(P1r,Q1i), VSUB(P1i,Q1r)); \
        STOREP(LL-k, VSUB(P1r,Q1i), VADD(P1i,Q1r)); \
    } } while(0)

// Folded odd-prime body processing TWO lane-chunks (base+0, base+8) at once,
// sharing the C/S broadcasts. k-pair x 2-chunk accumulators.
#define KMFOLD2_BODY(LL, HH, TC, TS, LOADR, LOADI, STOREP, STOREP2) do { \
    VD ur[HH+1], ui[HH+1], vr[HH+1], vi[HH+1]; \
    VD u2r[HH+1], u2i[HH+1], v2r[HH+1], v2i[HH+1]; \
    VD x0r = LOADR(0,0), x0i = LOADI(0,0); \
    VD y0r = LOADR(0,8), y0i = LOADI(0,8); \
    VD sur0 = x0r, sui0 = x0i, sur1 = ZERO, sui1 = ZERO; \
    VD tur0 = y0r, tui0 = y0i, tur1 = ZERO, tui1 = ZERO; \
    _Pragma("GCC unroll 16") \
    for (int j = 1; j <= HH; j++) { \
        VD ar = LOADR(j,0), ai = LOADI(j,0); \
        VD br = LOADR(LL-j,0), bi = LOADI(LL-j,0); \
        ur[j]=VADD(ar,br); ui[j]=VADD(ai,bi); \
        vr[j]=VSUB(ar,br); vi[j]=VSUB(ai,bi); \
        VD cr = LOADR(j,8), ci = LOADI(j,8); \
        VD dr = LOADR(LL-j,8), di = LOADI(LL-j,8); \
        u2r[j]=VADD(cr,dr); u2i[j]=VADD(ci,di); \
        v2r[j]=VSUB(cr,dr); v2i[j]=VSUB(ci,di); \
        if (j & 1) { sur0=VADD(sur0,ur[j]); sui0=VADD(sui0,ui[j]); tur0=VADD(tur0,u2r[j]); tui0=VADD(tui0,u2i[j]); } \
        else       { sur1=VADD(sur1,ur[j]); sui1=VADD(sui1,ui[j]); tur1=VADD(tur1,u2r[j]); tui1=VADD(tui1,u2i[j]); } \
    } \
    STOREP(0, VADD(sur0,sur1), VADD(sui0,sui1)); \
    STOREP2(0, VADD(tur0,tur1), VADD(tui0,tui1)); \
    _Pragma("GCC unroll 16") \
    for (int k = 1; k + 1 <= HH; k += 2) { \
        VD P1r = x0r, P1i = x0i, Q1r = ZERO, Q1i = ZERO; \
        VD P2r = x0r, P2i = x0i, Q2r = ZERO, Q2i = ZERO; \
        VD R1r = y0r, R1i = y0i, S1r = ZERO, S1i = ZERO; \
        VD R2r = y0r, R2i = y0i, S2r = ZERO, S2i = ZERO; \
        _Pragma("GCC unroll 16") \
        for (int j = 1; j <= HH; j++) { \
            VD c1 = BC(TC[(k-1)*HH + (j-1)]), s1 = BC(TS[(k-1)*HH + (j-1)]); \
            VD c2 = BC(TC[k*HH + (j-1)]),     s2 = BC(TS[k*HH + (j-1)]); \
            P1r = VFMA(c1, ur[j], P1r); P1i = VFMA(c1, ui[j], P1i); \
            Q1r = VFMA(s1, vr[j], Q1r); Q1i = VFMA(s1, vi[j], Q1i); \
            P2r = VFMA(c2, ur[j], P2r); P2i = VFMA(c2, ui[j], P2i); \
            Q2r = VFMA(s2, vr[j], Q2r); Q2i = VFMA(s2, vi[j], Q2i); \
            R1r = VFMA(c1, u2r[j], R1r); R1i = VFMA(c1, u2i[j], R1i); \
            S1r = VFMA(s1, v2r[j], S1r); S1i = VFMA(s1, v2i[j], S1i); \
            R2r = VFMA(c2, u2r[j], R2r); R2i = VFMA(c2, u2i[j], R2i); \
            S2r = VFMA(s2, v2r[j], S2r); S2i = VFMA(s2, v2i[j], S2i); \
        } \
        STOREP(k,      VADD(P1r,Q1i), VSUB(P1i,Q1r)); \
        STOREP(LL-k,   VSUB(P1r,Q1i), VADD(P1i,Q1r)); \
        STOREP(k+1,    VADD(P2r,Q2i), VSUB(P2i,Q2r)); \
        STOREP(LL-k-1, VSUB(P2r,Q2i), VADD(P2i,Q2r)); \
        STOREP2(k,      VADD(R1r,S1i), VSUB(R1i,S1r)); \
        STOREP2(LL-k,   VSUB(R1r,S1i), VADD(R1i,S1r)); \
        STOREP2(k+1,    VADD(R2r,S2i), VSUB(R2i,S2r)); \
        STOREP2(LL-k-1, VSUB(R2r,S2i), VADD(R2i,S2r)); \
    } \
    if (HH & 1) { \
        const int k = HH; \
        VD P1r = x0r, P1i = x0i, Q1r = ZERO, Q1i = ZERO; \
        VD R1r = y0r, R1i = y0i, S1r = ZERO, S1i = ZERO; \
        _Pragma("GCC unroll 16") \
        for (int j = 1; j <= HH; j++) { \
            VD c1 = BC(TC[(k-1)*HH + (j-1)]), s1 = BC(TS[(k-1)*HH + (j-1)]); \
            P1r = VFMA(c1, ur[j], P1r); P1i = VFMA(c1, ui[j], P1i); \
            Q1r = VFMA(s1, vr[j], Q1r); Q1i = VFMA(s1, vi[j], Q1i); \
            R1r = VFMA(c1, u2r[j], R1r); R1i = VFMA(c1, u2i[j], R1i); \
            S1r = VFMA(s1, v2r[j], S1r); S1i = VFMA(s1, v2i[j], S1i); \
        } \
        STOREP(k,    VADD(P1r,Q1i), VSUB(P1i,Q1r)); \
        STOREP(LL-k, VSUB(P1r,Q1i), VADD(P1i,Q1r)); \
        STOREP2(k,    VADD(R1r,S1i), VSUB(R1i,S1r)); \
        STOREP2(LL-k, VSUB(R1r,S1i), VADD(R1i,S1r)); \
    } } while(0)
