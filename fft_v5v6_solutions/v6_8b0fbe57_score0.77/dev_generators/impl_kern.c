// ---------------------------------------------------------------------------
// Small DFT macros on (re,im) vector pairs. U = unique suffix for temps.
// ---------------------------------------------------------------------------
#define DFT2V(o0r,o0i,o1r,o1i, a0r,a0i,a1r,a1i) do { \
    VD _t0r=VADD(a0r,a1r), _t0i=VADD(a0i,a1i);       \
    VD _t1r=VSUB(a0r,a1r), _t1i=VSUB(a0i,a1i);       \
    o0r=_t0r; o0i=_t0i; o1r=_t1r; o1i=_t1i; } while(0)

// forward DFT3: needs Vhalf=0.5, Vs3=sqrt(3)/2 in scope
#define DFT3V(o0r,o0i,o1r,o1i,o2r,o2i, x0r,x0i,x1r,x1i,x2r,x2i) do { \
    VD _tr=VADD(x1r,x2r), _ti=VADD(x1i,x2i);   \
    VD _dr=VSUB(x1r,x2r), _di=VSUB(x1i,x2i);   \
    VD _mr=VFNMA(Vhalf,_tr,x0r), _mi=VFNMA(Vhalf,_ti,x0i); \
    VD _sr=VMUL(Vs3,_dr), _si=VMUL(Vs3,_di);   \
    o0r=VADD(x0r,_tr); o0i=VADD(x0i,_ti);      \
    o1r=VADD(_mr,_si); o1i=VSUB(_mi,_sr);      \
    o2r=VSUB(_mr,_si); o2i=VADD(_mi,_sr); } while(0)

// forward DFT4
#define DFT4V(o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i, x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i) do { \
    VD _t0r=VADD(x0r,x2r), _t0i=VADD(x0i,x2i);  \
    VD _t1r=VSUB(x0r,x2r), _t1i=VSUB(x0i,x2i);  \
    VD _t2r=VADD(x1r,x3r), _t2i=VADD(x1i,x3i);  \
    VD _t3r=VSUB(x1r,x3r), _t3i=VSUB(x1i,x3i);  \
    o0r=VADD(_t0r,_t2r); o0i=VADD(_t0i,_t2i);   \
    o2r=VSUB(_t0r,_t2r); o2i=VSUB(_t0i,_t2i);   \
    o1r=VADD(_t1r,_t3i); o1i=VSUB(_t1i,_t3r);   \
    o3r=VSUB(_t1r,_t3i); o3i=VADD(_t1i,_t3r); } while(0)

// forward DFT5 (folded): needs Vc51,Vc52,Vs51,Vs52
#define DFT5V(o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i,o4r,o4i, x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,x4r,x4i) do { \
    VD _u1r=VADD(x1r,x4r), _u1i=VADD(x1i,x4i);  \
    VD _v1r=VSUB(x1r,x4r), _v1i=VSUB(x1i,x4i);  \
    VD _u2r=VADD(x2r,x3r), _u2i=VADD(x2i,x3i);  \
    VD _v2r=VSUB(x2r,x3r), _v2i=VSUB(x2i,x3i);  \
    o0r=VADD(x0r,VADD(_u1r,_u2r)); o0i=VADD(x0i,VADD(_u1i,_u2i)); \
    VD _A1r=VFMA(Vc51,_u1r,VFMA(Vc52,_u2r,x0r)); \
    VD _A1i=VFMA(Vc51,_u1i,VFMA(Vc52,_u2i,x0i)); \
    VD _B1r=VFMA(Vs51,_v1r,VMUL(Vs52,_v2r));     \
    VD _B1i=VFMA(Vs51,_v1i,VMUL(Vs52,_v2i));     \
    VD _A2r=VFMA(Vc52,_u1r,VFMA(Vc51,_u2r,x0r)); \
    VD _A2i=VFMA(Vc52,_u1i,VFMA(Vc51,_u2i,x0i)); \
    VD _B2r=VFMS(Vs52,_v1r,VMUL(Vs51,_v2r));     \
    VD _B2i=VFMS(Vs52,_v1i,VMUL(Vs51,_v2i));     \
    o1r=VADD(_A1r,_B1i); o1i=VSUB(_A1i,_B1r);    \
    o4r=VSUB(_A1r,_B1i); o4i=VADD(_A1i,_B1r);    \
    o2r=VADD(_A2r,_B2i); o2i=VSUB(_A2i,_B2r);    \
    o3r=VSUB(_A2r,_B2i); o3i=VADD(_A2i,_B2r); } while(0)

// forward DFT8: needs Vr2 = sqrt(2)/2
#define DFT8V(o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i,o4r,o4i,o5r,o5i,o6r,o6i,o7r,o7i, \
              x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,x4r,x4i,x5r,x5i,x6r,x6i,x7r,x7i) do { \
    VD _E0r,_E0i,_E1r,_E1i,_E2r,_E2i,_E3r,_E3i; \
    VD _O0r,_O0i,_O1r,_O1i,_O2r,_O2i,_O3r,_O3i; \
    DFT4V(_E0r,_E0i,_E1r,_E1i,_E2r,_E2i,_E3r,_E3i, x0r,x0i,x2r,x2i,x4r,x4i,x6r,x6i); \
    DFT4V(_O0r,_O0i,_O1r,_O1i,_O2r,_O2i,_O3r,_O3i, x1r,x1i,x3r,x3i,x5r,x5i,x7r,x7i); \
    /* T1 = w8 * O1, w8=(s,-s) : re=s*(or+oi), im=s*(oi-or) */ \
    VD _T1r=VMUL(Vr2,VADD(_O1r,_O1i)), _T1i=VMUL(Vr2,VSUB(_O1i,_O1r)); \
    /* T2 = -i*O2: (oi, -or) */ \
    VD _T2r=_O2i, _T2i=_mm512_sub_pd(ZERO,_O2r); \
    /* T3 = w8^3 * O3 = (-s,-s): re=-s*(or-oi)=s*(oi-or), im=-s*(or+oi) */ \
    VD _T3r=VMUL(Vr2,VSUB(_O3i,_O3r)), _T3i=VMUL(Vr2,VSUB(_mm512_sub_pd(ZERO,_O3r),_O3i)); \
    o0r=VADD(_E0r,_O0r); o0i=VADD(_E0i,_O0i);  \
    o4r=VSUB(_E0r,_O0r); o4i=VSUB(_E0i,_O0i);  \
    o1r=VADD(_E1r,_T1r); o1i=VADD(_E1i,_T1i);  \
    o5r=VSUB(_E1r,_T1r); o5i=VSUB(_E1i,_T1i);  \
    o2r=VADD(_E2r,_T2r); o2i=VADD(_E2i,_T2i);  \
    o6r=VSUB(_E2r,_T2r); o6i=VSUB(_E2i,_T2i);  \
    o3r=VADD(_E3r,_T3r); o3i=VADD(_E3i,_T3i);  \
    o7r=VSUB(_E3r,_T3r); o7i=VSUB(_E3i,_T3i); } while(0)

// complex multiply: (or,oi) = (ar,ai)*(br,bi)
#define CMULV(or_,oi_, ar,ai, br,bi) do { \
    VD _pr = VMUL(ar,br), _pi = VMUL(ar,bi); \
    or_ = VFNMA(ai,bi,_pr); oi_ = VFMA(ai,br,_pi); } while(0)

// forward DFT9 via 3x3 CT. Needs Vhalf,Vs3 and Vw91r.. in scope.
// x is 18 VD args x0r..x8i ; o is o0r..o8i
#define DFT9V(o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i,o4r,o4i,o5r,o5i,o6r,o6i,o7r,o7i,o8r,o8i, \
              x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,x4r,x4i,x5r,x5i,x6r,x6i,x7r,x7i,x8r,x8i) do { \
    VD _A00r,_A00i,_A01r,_A01i,_A02r,_A02i;  \
    VD _A10r,_A10i,_A11r,_A11i,_A12r,_A12i;  \
    VD _A20r,_A20i,_A21r,_A21i,_A22r,_A22i;  \
    DFT3V(_A00r,_A00i,_A01r,_A01i,_A02r,_A02i, x0r,x0i,x3r,x3i,x6r,x6i); \
    DFT3V(_A10r,_A10i,_A11r,_A11i,_A12r,_A12i, x1r,x1i,x4r,x4i,x7r,x7i); \
    DFT3V(_A20r,_A20i,_A21r,_A21i,_A22r,_A22i, x2r,x2i,x5r,x5i,x8r,x8i); \
    VD _B11r,_B11i,_B12r,_B12i,_B21r,_B21i,_B22r,_B22i; \
    CMULV(_B11r,_B11i,_A11r,_A11i,Vw91r,Vw91i);  \
    CMULV(_B12r,_B12i,_A12r,_A12i,Vw92r,Vw92i);  \
    CMULV(_B21r,_B21i,_A21r,_A21i,Vw92r,Vw92i);  \
    CMULV(_B22r,_B22i,_A22r,_A22i,Vw94r,Vw94i);  \
    /* out[k3 + 3q] = DFT3 over j2 of Btilde[.][k3] */ \
    DFT3V(o0r,o0i,o3r,o3i,o6r,o6i, _A00r,_A00i,_A10r,_A10i,_A20r,_A20i); \
    DFT3V(o1r,o1i,o4r,o4i,o7r,o7i, _A01r,_A01i,_B11r,_B11i,_B21r,_B21i); \
    DFT3V(o2r,o2i,o5r,o5i,o8r,o8i, _A02r,_A02i,_B12r,_B12i,_B22r,_B22i); } while(0)
