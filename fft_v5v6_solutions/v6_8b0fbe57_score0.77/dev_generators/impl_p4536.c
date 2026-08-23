// ---------------------------------------------------------------------------
// L=45 and L=36 alternating PFA pipelines (see derivation in comments).
// ---------------------------------------------------------------------------
#define PS45 2040
#define PS36 1304
static double T45re[45*PS45 + 8] __attribute__((aligned(64)));
static double T45im[45*PS45 + 8] __attribute__((aligned(64)));
static double T36re[36*PS36 + 8] __attribute__((aligned(64)));
static double T36im[36*PS36 + 8] __attribute__((aligned(64)));
#define SB45S 2040
#define SB36S 1304

#define PMAP(dstr, dsti, sbr, sbi, crp, cip, q) do { \
    VD _yr = LD((sbr)+(q)), _yi = LD((sbi)+(q)); \
    VD _zr = VADD(_yr, LD((crp)+(q))), _zi = VADD(_yi, LD((cip)+(q))); \
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
static void prime45(const double* restrict re, const double* restrict im)
{
    const VD Vhalf=BC(0.5), Vs3=BC(g_s3half);
    const VD Vw91r=BC(g_w9[0]),Vw91i=BC(g_w9[1]),Vw92r=BC(g_w9[2]),Vw92i=BC(g_w9[3]),Vw94r=BC(g_w9[4]),Vw94i=BC(g_w9[5]);
    for (int j5 = 0; j5 < 5; j5++) {
        const double *s_r[9], *s_i[9]; double *d_r[9], *d_i[9];
        for (int j9 = 0; j9 < 9; j9++) {
            int sl = (5*j9 + 9*j5) % 45;
            s_r[j9] = re + (size_t)sl*2025; s_i[j9] = im + (size_t)sl*2025;
            int slot = ((4*j5) % 5)*9 + j9;
            d_r[j9] = T45re + (size_t)slot*PS45; d_i[j9] = T45im + (size_t)slot*PS45;
        }
        for (long q = 0; q < 2025; q += 8) {
            VD y0r=LD(s_r[0]+q), y0i=LD(s_i[0]+q);
            VD y1r=LD(s_r[1]+q), y1i=LD(s_i[1]+q);
            VD y2r=LD(s_r[2]+q), y2i=LD(s_i[2]+q);
            VD y3r=LD(s_r[3]+q), y3i=LD(s_i[3]+q);
            VD y4r=LD(s_r[4]+q), y4i=LD(s_i[4]+q);
            VD y5r=LD(s_r[5]+q), y5i=LD(s_i[5]+q);
            VD y6r=LD(s_r[6]+q), y6i=LD(s_i[6]+q);
            VD y7r=LD(s_r[7]+q), y7i=LD(s_i[7]+q);
            VD y8r=LD(s_r[8]+q), y8i=LD(s_i[8]+q);
            VD o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i,o4r,o4i,o5r,o5i,o6r,o6i,o7r,o7i,o8r,o8i;
            DFT9V(o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i,o4r,o4i,o5r,o5i,o6r,o6i,o7r,o7i,o8r,o8i,
                  y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i,y4r,y4i,y5r,y5i,y6r,y6i,y7r,y7i,y8r,y8i);
            ST(d_r[0]+q,o0r); ST(d_i[0]+q,o0i);
            ST(d_r[1]+q,o1r); ST(d_i[1]+q,o1i);
            ST(d_r[2]+q,o2r); ST(d_i[2]+q,o2i);
            ST(d_r[3]+q,o3r); ST(d_i[3]+q,o3i);
            ST(d_r[4]+q,o4r); ST(d_i[4]+q,o4i);
            ST(d_r[5]+q,o5r); ST(d_i[5]+q,o5i);
            ST(d_r[6]+q,o6r); ST(d_i[6]+q,o6i);
            ST(d_r[7]+q,o7r); ST(d_i[7]+q,o7i);
            ST(d_r[8]+q,o8r); ST(d_i[8]+q,o8i);
        }
    }
}
static void yz45(double* restrict sr, double* restrict si)
{
    static const MK zm[6] = {0xFF,0xFF,0xFF,0xFF,0xFF,0x1F};
    for (int zc = 0; zc < 5; zc++) km45_y(sr + 8*zc, si + 8*zc, Wre_ + 8*zc, Wim_ + 8*zc, 0xFF);
    km45_y(sr + 40, si + 40, Wre_ + 40, Wim_ + 40, 0x1F);
    spass_gen(Wre_, Wim_, sr, si, 45, 6, zm, km45_s);
}
static void iter45_even(const double* restrict cre, const double* restrict cim, int last, double* restrict outre, double* restrict outim)
{
    const VD Vhalf=BC(0.5), Vs3=BC(g_s3half);
    const VD Vc51=BC(g_c51),Vc52=BC(g_c52),Vs51=BC(g_s51),Vs52=BC(g_s52);    const VD Vone=BC(1.0), V1p5=BC(1.5), Vtiny=BC(2.3e-308);
    for (int k9 = 0; k9 < 9; k9++) {
        {
            const double *p_r[5], *p_i[5];
            for (int j5 = 0; j5 < 5; j5++) {
                int slot = ((4*j5) % 5)*9 + k9;
                p_r[j5] = T45re + (size_t)slot*PS45; p_i[j5] = T45im + (size_t)slot*PS45;
            }
            for (long q = 0; q < 2025; q += 8) {
            VD y0r=LD(p_r[0]+q), y0i=LD(p_i[0]+q);
            VD y1r=LD(p_r[1]+q), y1i=LD(p_i[1]+q);
            VD y2r=LD(p_r[2]+q), y2i=LD(p_i[2]+q);
            VD y3r=LD(p_r[3]+q), y3i=LD(p_i[3]+q);
            VD y4r=LD(p_r[4]+q), y4i=LD(p_i[4]+q);
            VD o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i,o4r,o4i;
            DFT5V(o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i,o4r,o4i,
                  y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i,y4r,y4i);
            ST(SBre+0*SB45S+q,o0r); ST(SBim+0*SB45S+q,o0i);
            ST(SBre+1*SB45S+q,o1r); ST(SBim+1*SB45S+q,o1i);
            ST(SBre+2*SB45S+q,o2r); ST(SBim+2*SB45S+q,o2i);
            ST(SBre+3*SB45S+q,o3r); ST(SBim+3*SB45S+q,o3i);
            ST(SBre+4*SB45S+q,o4r); ST(SBim+4*SB45S+q,o4i);
            }
        }
        for (int k5 = 0; k5 < 5; k5++)
            yz45(SBre + (size_t)k5*SB45S, SBim + (size_t)k5*SB45S);
        {
            const double *c_r[5], *c_i[5], *b_r[5], *b_i[5];
            double *o_r[5], *o_i[5], *t_r[5], *t_i[5];
            for (int i = 0; i < 5; i++) {
                int k5 = (4*i) % 5;   /* input j5'=i comes from SB[k5] */
                int sl = (36*k5 + 10*k9) % 45;
                b_r[i] = SBre + (size_t)k5*SB45S; b_i[i] = SBim + (size_t)k5*SB45S;
                c_r[i] = cre + (size_t)sl*2025;   c_i[i] = cim + (size_t)sl*2025;
                o_r[i] = outre ? outre + (size_t)sl*2025 : 0;
                o_i[i] = outim ? outim + (size_t)sl*2025 : 0;
                int slot = i*9 + k9;  /* output k5'=i at slot k5'*9+k9 */
                t_r[i] = T45re + (size_t)slot*PS45; t_i[i] = T45im + (size_t)slot*PS45;
            }
            for (long q = 0; q < 2025; q += 8) {
                MK mm = (q + 8 <= 2025) ? (MK)0xFF : (MK)((1u << (2025 - q)) - 1);
                (void)mm;
                VD x0r,x0i; PMAP(x0r,x0i,b_r[0],b_i[0],c_r[0],c_i[0],q);
                VD x1r,x1i; PMAP(x1r,x1i,b_r[1],b_i[1],c_r[1],c_i[1],q);
                VD x2r,x2i; PMAP(x2r,x2i,b_r[2],b_i[2],c_r[2],c_i[2],q);
                VD x3r,x3i; PMAP(x3r,x3i,b_r[3],b_i[3],c_r[3],c_i[3],q);
                VD x4r,x4i; PMAP(x4r,x4i,b_r[4],b_i[4],c_r[4],c_i[4],q);
                if (outre) {
                    MST(o_r[0]+q,mm,x0r); MST(o_i[0]+q,mm,x0i);
                    MST(o_r[1]+q,mm,x1r); MST(o_i[1]+q,mm,x1i);
                    MST(o_r[2]+q,mm,x2r); MST(o_i[2]+q,mm,x2i);
                    MST(o_r[3]+q,mm,x3r); MST(o_i[3]+q,mm,x3i);
                    MST(o_r[4]+q,mm,x4r); MST(o_i[4]+q,mm,x4i);
                }
                if (last) continue;
                VD o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i,o4r,o4i;
            DFT5V(o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i,o4r,o4i,
                      x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,x4r,x4i);
                ST(t_r[0]+q,o0r); ST(t_i[0]+q,o0i);
                ST(t_r[1]+q,o1r); ST(t_i[1]+q,o1i);
                ST(t_r[2]+q,o2r); ST(t_i[2]+q,o2i);
                ST(t_r[3]+q,o3r); ST(t_i[3]+q,o3i);
                ST(t_r[4]+q,o4r); ST(t_i[4]+q,o4i);
            }
        }
    }
}
static void iter45_odd(const double* restrict cre, const double* restrict cim, int last, double* restrict outre, double* restrict outim)
{
    const VD Vhalf=BC(0.5), Vs3=BC(g_s3half);
    const VD Vw91r=BC(g_w9[0]),Vw91i=BC(g_w9[1]),Vw92r=BC(g_w9[2]),Vw92i=BC(g_w9[3]),Vw94r=BC(g_w9[4]),Vw94i=BC(g_w9[5]);    const VD Vone=BC(1.0), V1p5=BC(1.5), Vtiny=BC(2.3e-308);
    for (int k5 = 0; k5 < 5; k5++) {
        {
            const double *p_r[9], *p_i[9];
            for (int j9 = 0; j9 < 9; j9++) {
                int slot = k5*9 + (5*j9) % 9;
                p_r[j9] = T45re + (size_t)slot*PS45; p_i[j9] = T45im + (size_t)slot*PS45;
            }
            for (long q = 0; q < 2025; q += 8) {
            VD y0r=LD(p_r[0]+q), y0i=LD(p_i[0]+q);
            VD y1r=LD(p_r[1]+q), y1i=LD(p_i[1]+q);
            VD y2r=LD(p_r[2]+q), y2i=LD(p_i[2]+q);
            VD y3r=LD(p_r[3]+q), y3i=LD(p_i[3]+q);
            VD y4r=LD(p_r[4]+q), y4i=LD(p_i[4]+q);
            VD y5r=LD(p_r[5]+q), y5i=LD(p_i[5]+q);
            VD y6r=LD(p_r[6]+q), y6i=LD(p_i[6]+q);
            VD y7r=LD(p_r[7]+q), y7i=LD(p_i[7]+q);
            VD y8r=LD(p_r[8]+q), y8i=LD(p_i[8]+q);
            VD o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i,o4r,o4i,o5r,o5i,o6r,o6i,o7r,o7i,o8r,o8i;
            DFT9V(o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i,o4r,o4i,o5r,o5i,o6r,o6i,o7r,o7i,o8r,o8i,
                  y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i,y4r,y4i,y5r,y5i,y6r,y6i,y7r,y7i,y8r,y8i);
            ST(SBre+0*SB45S+q,o0r); ST(SBim+0*SB45S+q,o0i);
            ST(SBre+1*SB45S+q,o1r); ST(SBim+1*SB45S+q,o1i);
            ST(SBre+2*SB45S+q,o2r); ST(SBim+2*SB45S+q,o2i);
            ST(SBre+3*SB45S+q,o3r); ST(SBim+3*SB45S+q,o3i);
            ST(SBre+4*SB45S+q,o4r); ST(SBim+4*SB45S+q,o4i);
            ST(SBre+5*SB45S+q,o5r); ST(SBim+5*SB45S+q,o5i);
            ST(SBre+6*SB45S+q,o6r); ST(SBim+6*SB45S+q,o6i);
            ST(SBre+7*SB45S+q,o7r); ST(SBim+7*SB45S+q,o7i);
            ST(SBre+8*SB45S+q,o8r); ST(SBim+8*SB45S+q,o8i);
            }
        }
        for (int k9 = 0; k9 < 9; k9++)
            yz45(SBre + (size_t)k9*SB45S, SBim + (size_t)k9*SB45S);
        {
            const double *c_r[9], *c_i[9], *b_r[9], *b_i[9];
            double *o_r[9], *o_i[9], *t_r[9], *t_i[9];
            for (int i = 0; i < 9; i++) {
                int k9 = (5*i) % 9;   /* input j9=i from SB[k9] */
                int sl = (36*k5 + 10*k9) % 45;
                b_r[i] = SBre + (size_t)k9*SB45S; b_i[i] = SBim + (size_t)k9*SB45S;
                c_r[i] = cre + (size_t)sl*2025;   c_i[i] = cim + (size_t)sl*2025;
                o_r[i] = outre ? outre + (size_t)sl*2025 : 0;
                o_i[i] = outim ? outim + (size_t)sl*2025 : 0;
                int slot = k5*9 + i;   /* output k9'=i at slot k5*9+k9' */
                t_r[i] = T45re + (size_t)slot*PS45; t_i[i] = T45im + (size_t)slot*PS45;
            }
            for (long q = 0; q < 2025; q += 8) {
                MK mm = (q + 8 <= 2025) ? (MK)0xFF : (MK)((1u << (2025 - q)) - 1);
                (void)mm;
                VD x0r,x0i; PMAP(x0r,x0i,b_r[0],b_i[0],c_r[0],c_i[0],q);
                VD x1r,x1i; PMAP(x1r,x1i,b_r[1],b_i[1],c_r[1],c_i[1],q);
                VD x2r,x2i; PMAP(x2r,x2i,b_r[2],b_i[2],c_r[2],c_i[2],q);
                VD x3r,x3i; PMAP(x3r,x3i,b_r[3],b_i[3],c_r[3],c_i[3],q);
                VD x4r,x4i; PMAP(x4r,x4i,b_r[4],b_i[4],c_r[4],c_i[4],q);
                VD x5r,x5i; PMAP(x5r,x5i,b_r[5],b_i[5],c_r[5],c_i[5],q);
                VD x6r,x6i; PMAP(x6r,x6i,b_r[6],b_i[6],c_r[6],c_i[6],q);
                VD x7r,x7i; PMAP(x7r,x7i,b_r[7],b_i[7],c_r[7],c_i[7],q);
                VD x8r,x8i; PMAP(x8r,x8i,b_r[8],b_i[8],c_r[8],c_i[8],q);
                if (outre) {
                    MST(o_r[0]+q,mm,x0r); MST(o_i[0]+q,mm,x0i);
                    MST(o_r[1]+q,mm,x1r); MST(o_i[1]+q,mm,x1i);
                    MST(o_r[2]+q,mm,x2r); MST(o_i[2]+q,mm,x2i);
                    MST(o_r[3]+q,mm,x3r); MST(o_i[3]+q,mm,x3i);
                    MST(o_r[4]+q,mm,x4r); MST(o_i[4]+q,mm,x4i);
                    MST(o_r[5]+q,mm,x5r); MST(o_i[5]+q,mm,x5i);
                    MST(o_r[6]+q,mm,x6r); MST(o_i[6]+q,mm,x6i);
                    MST(o_r[7]+q,mm,x7r); MST(o_i[7]+q,mm,x7i);
                    MST(o_r[8]+q,mm,x8r); MST(o_i[8]+q,mm,x8i);
                }
                if (last) continue;
                VD o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i,o4r,o4i,o5r,o5i,o6r,o6i,o7r,o7i,o8r,o8i;
            DFT9V(o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i,o4r,o4i,o5r,o5i,o6r,o6i,o7r,o7i,o8r,o8i,
                      x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,x4r,x4i,x5r,x5i,x6r,x6i,x7r,x7i,x8r,x8i);
                ST(t_r[0]+q,o0r); ST(t_i[0]+q,o0i);
                ST(t_r[1]+q,o1r); ST(t_i[1]+q,o1i);
                ST(t_r[2]+q,o2r); ST(t_i[2]+q,o2i);
                ST(t_r[3]+q,o3r); ST(t_i[3]+q,o3i);
                ST(t_r[4]+q,o4r); ST(t_i[4]+q,o4i);
                ST(t_r[5]+q,o5r); ST(t_i[5]+q,o5i);
                ST(t_r[6]+q,o6r); ST(t_i[6]+q,o6i);
                ST(t_r[7]+q,o7r); ST(t_i[7]+q,o7i);
                ST(t_r[8]+q,o8r); ST(t_i[8]+q,o8i);
            }
        }
    }
}
static void chain45(double* restrict re, double* restrict im,
                    const double* restrict cre, const double* restrict cim,
                    long m, double* restrict out1)
{
    prime45(re, im);
    for (long n = 0; n < m; n++) {
        int last = (n == m-1), first = (n == 0);
        double *outr = (first || last) ? re : 0, *outi = (first || last) ? im : 0;
        if (!(n & 1)) iter45_even(cre, cim, last, outr, outi);
        else          iter45_odd(cre, cim, last, outr, outi);
        if (first && out1) soa_export(re, im, out1, 91125);
    }
}
static void prime36(const double* restrict re, const double* restrict im)
{
    const VD Vhalf=BC(0.5), Vs3=BC(g_s3half);
    const VD Vw91r=BC(g_w9[0]),Vw91i=BC(g_w9[1]),Vw92r=BC(g_w9[2]),Vw92i=BC(g_w9[3]),Vw94r=BC(g_w9[4]),Vw94i=BC(g_w9[5]);
    for (int j4 = 0; j4 < 4; j4++) {
        const double *s_r[9], *s_i[9]; double *d_r[9], *d_i[9];
        for (int j9 = 0; j9 < 9; j9++) {
            int sl = (9*j4 + 4*j9) % 36;
            s_r[j9] = re + (size_t)sl*1296; s_i[j9] = im + (size_t)sl*1296;
            int slot = j4*9 + j9;
            d_r[j9] = T36re + (size_t)slot*PS36; d_i[j9] = T36im + (size_t)slot*PS36;
        }
        for (long q = 0; q < 1296; q += 8) {
            VD y0r=LD(s_r[0]+q), y0i=LD(s_i[0]+q);
            VD y1r=LD(s_r[1]+q), y1i=LD(s_i[1]+q);
            VD y2r=LD(s_r[2]+q), y2i=LD(s_i[2]+q);
            VD y3r=LD(s_r[3]+q), y3i=LD(s_i[3]+q);
            VD y4r=LD(s_r[4]+q), y4i=LD(s_i[4]+q);
            VD y5r=LD(s_r[5]+q), y5i=LD(s_i[5]+q);
            VD y6r=LD(s_r[6]+q), y6i=LD(s_i[6]+q);
            VD y7r=LD(s_r[7]+q), y7i=LD(s_i[7]+q);
            VD y8r=LD(s_r[8]+q), y8i=LD(s_i[8]+q);
            VD o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i,o4r,o4i,o5r,o5i,o6r,o6i,o7r,o7i,o8r,o8i;
            DFT9V(o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i,o4r,o4i,o5r,o5i,o6r,o6i,o7r,o7i,o8r,o8i,
                  y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i,y4r,y4i,y5r,y5i,y6r,y6i,y7r,y7i,y8r,y8i);
            ST(d_r[0]+q,o0r); ST(d_i[0]+q,o0i);
            ST(d_r[1]+q,o1r); ST(d_i[1]+q,o1i);
            ST(d_r[2]+q,o2r); ST(d_i[2]+q,o2i);
            ST(d_r[3]+q,o3r); ST(d_i[3]+q,o3i);
            ST(d_r[4]+q,o4r); ST(d_i[4]+q,o4i);
            ST(d_r[5]+q,o5r); ST(d_i[5]+q,o5i);
            ST(d_r[6]+q,o6r); ST(d_i[6]+q,o6i);
            ST(d_r[7]+q,o7r); ST(d_i[7]+q,o7i);
            ST(d_r[8]+q,o8r); ST(d_i[8]+q,o8i);
        }
    }
}
static void yz36(double* restrict sr, double* restrict si)
{
    static const MK zm[5] = {0xFF,0xFF,0xFF,0xFF,0x0F};
    for (int zc = 0; zc < 4; zc++) km36_y(sr + 8*zc, si + 8*zc, Wre_ + 8*zc, Wim_ + 8*zc, 0xFF);
    km36_y(sr + 32, si + 32, Wre_ + 32, Wim_ + 32, 0x0F);
    spass_gen(Wre_, Wim_, sr, si, 36, 5, zm, km36_s);
}
static void iter36_even(const double* restrict cre, const double* restrict cim, int last, double* restrict outre, double* restrict outim)
{
    const VD Vhalf=BC(0.5);    const VD Vone=BC(1.0), V1p5=BC(1.5), Vtiny=BC(2.3e-308);
    for (int k9 = 0; k9 < 9; k9++) {
        {
            const double *p_r[4], *p_i[4];
            for (int j4 = 0; j4 < 4; j4++) {
                int slot = j4*9 + k9;
                p_r[j4] = T36re + (size_t)slot*PS36; p_i[j4] = T36im + (size_t)slot*PS36;
            }
            for (long q = 0; q < 1296; q += 8) {
            VD y0r=LD(p_r[0]+q), y0i=LD(p_i[0]+q);
            VD y1r=LD(p_r[1]+q), y1i=LD(p_i[1]+q);
            VD y2r=LD(p_r[2]+q), y2i=LD(p_i[2]+q);
            VD y3r=LD(p_r[3]+q), y3i=LD(p_i[3]+q);
            VD o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i;
            DFT4V(o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i,
                  y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i);
            ST(SBre+0*SB36S+q,o0r); ST(SBim+0*SB36S+q,o0i);
            ST(SBre+1*SB36S+q,o1r); ST(SBim+1*SB36S+q,o1i);
            ST(SBre+2*SB36S+q,o2r); ST(SBim+2*SB36S+q,o2i);
            ST(SBre+3*SB36S+q,o3r); ST(SBim+3*SB36S+q,o3i);
            }
        }
        for (int k4 = 0; k4 < 4; k4++)
            yz36(SBre + (size_t)k4*SB36S, SBim + (size_t)k4*SB36S);
        {
            const double *c_r[4], *c_i[4], *b_r[4], *b_i[4];
            double *o_r[4], *o_i[4], *t_r[4], *t_i[4];
            for (int i = 0; i < 4; i++) {
                int k4 = i;  /* j4' = k4 identity */
                int sl = (9*k4 + 28*k9) % 36;
                b_r[i] = SBre + (size_t)k4*SB36S; b_i[i] = SBim + (size_t)k4*SB36S;
                c_r[i] = cre + (size_t)sl*1296;   c_i[i] = cim + (size_t)sl*1296;
                o_r[i] = outre ? outre + (size_t)sl*1296 : 0;
                o_i[i] = outim ? outim + (size_t)sl*1296 : 0;
                int slot = i*9 + k9;  /* slot1(j9',k4'=i) = i*9 + (4*j9')%9, j9'=7k9%9 -> = k9 */
                t_r[i] = T36re + (size_t)slot*PS36; t_i[i] = T36im + (size_t)slot*PS36;
            }
            for (long q = 0; q < 1296; q += 8) {
                VD x0r,x0i; PMAP(x0r,x0i,b_r[0],b_i[0],c_r[0],c_i[0],q);
                VD x1r,x1i; PMAP(x1r,x1i,b_r[1],b_i[1],c_r[1],c_i[1],q);
                VD x2r,x2i; PMAP(x2r,x2i,b_r[2],b_i[2],c_r[2],c_i[2],q);
                VD x3r,x3i; PMAP(x3r,x3i,b_r[3],b_i[3],c_r[3],c_i[3],q);
                if (outre) {
                    ST(o_r[0]+q,x0r); ST(o_i[0]+q,x0i);
                    ST(o_r[1]+q,x1r); ST(o_i[1]+q,x1i);
                    ST(o_r[2]+q,x2r); ST(o_i[2]+q,x2i);
                    ST(o_r[3]+q,x3r); ST(o_i[3]+q,x3i);
                }
                if (last) continue;
                VD o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i;
            DFT4V(o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i,
                      x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i);
                ST(t_r[0]+q,o0r); ST(t_i[0]+q,o0i);
                ST(t_r[1]+q,o1r); ST(t_i[1]+q,o1i);
                ST(t_r[2]+q,o2r); ST(t_i[2]+q,o2i);
                ST(t_r[3]+q,o3r); ST(t_i[3]+q,o3i);
            }
        }
    }
}
static void iter36_odd(const double* restrict cre, const double* restrict cim, int last, double* restrict outre, double* restrict outim)
{
    const VD Vhalf=BC(0.5), Vs3=BC(g_s3half);
    const VD Vw91r=BC(g_w9[0]),Vw91i=BC(g_w9[1]),Vw92r=BC(g_w9[2]),Vw92i=BC(g_w9[3]),Vw94r=BC(g_w9[4]),Vw94i=BC(g_w9[5]);    const VD Vone=BC(1.0), V1p5=BC(1.5), Vtiny=BC(2.3e-308);
    for (int k4 = 0; k4 < 4; k4++) {
        {
            const double *p_r[9], *p_i[9];
            for (int j9 = 0; j9 < 9; j9++) {
                int slot = k4*9 + (4*j9) % 9;
                p_r[j9] = T36re + (size_t)slot*PS36; p_i[j9] = T36im + (size_t)slot*PS36;
            }
            for (long q = 0; q < 1296; q += 8) {
            VD y0r=LD(p_r[0]+q), y0i=LD(p_i[0]+q);
            VD y1r=LD(p_r[1]+q), y1i=LD(p_i[1]+q);
            VD y2r=LD(p_r[2]+q), y2i=LD(p_i[2]+q);
            VD y3r=LD(p_r[3]+q), y3i=LD(p_i[3]+q);
            VD y4r=LD(p_r[4]+q), y4i=LD(p_i[4]+q);
            VD y5r=LD(p_r[5]+q), y5i=LD(p_i[5]+q);
            VD y6r=LD(p_r[6]+q), y6i=LD(p_i[6]+q);
            VD y7r=LD(p_r[7]+q), y7i=LD(p_i[7]+q);
            VD y8r=LD(p_r[8]+q), y8i=LD(p_i[8]+q);
            VD o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i,o4r,o4i,o5r,o5i,o6r,o6i,o7r,o7i,o8r,o8i;
            DFT9V(o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i,o4r,o4i,o5r,o5i,o6r,o6i,o7r,o7i,o8r,o8i,
                  y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i,y4r,y4i,y5r,y5i,y6r,y6i,y7r,y7i,y8r,y8i);
            ST(SBre+0*SB36S+q,o0r); ST(SBim+0*SB36S+q,o0i);
            ST(SBre+1*SB36S+q,o1r); ST(SBim+1*SB36S+q,o1i);
            ST(SBre+2*SB36S+q,o2r); ST(SBim+2*SB36S+q,o2i);
            ST(SBre+3*SB36S+q,o3r); ST(SBim+3*SB36S+q,o3i);
            ST(SBre+4*SB36S+q,o4r); ST(SBim+4*SB36S+q,o4i);
            ST(SBre+5*SB36S+q,o5r); ST(SBim+5*SB36S+q,o5i);
            ST(SBre+6*SB36S+q,o6r); ST(SBim+6*SB36S+q,o6i);
            ST(SBre+7*SB36S+q,o7r); ST(SBim+7*SB36S+q,o7i);
            ST(SBre+8*SB36S+q,o8r); ST(SBim+8*SB36S+q,o8i);
            }
        }
        for (int k9 = 0; k9 < 9; k9++)
            yz36(SBre + (size_t)k9*SB36S, SBim + (size_t)k9*SB36S);
        {
            const double *c_r[9], *c_i[9], *b_r[9], *b_i[9];
            double *o_r[9], *o_i[9], *t_r[9], *t_i[9];
            for (int i = 0; i < 9; i++) {
                int k9 = (4*i) % 9;  /* input j9=i from SB[k9] */
                int sl = (9*k4 + 28*k9) % 36;
                b_r[i] = SBre + (size_t)k9*SB36S; b_i[i] = SBim + (size_t)k9*SB36S;
                c_r[i] = cre + (size_t)sl*1296;   c_i[i] = cim + (size_t)sl*1296;
                o_r[i] = outre ? outre + (size_t)sl*1296 : 0;
                o_i[i] = outim ? outim + (size_t)sl*1296 : 0;
                int slot = k4*9 + i;   /* output k9'=i at slot0 k4*9+k9' */
                t_r[i] = T36re + (size_t)slot*PS36; t_i[i] = T36im + (size_t)slot*PS36;
            }
            for (long q = 0; q < 1296; q += 8) {
                VD x0r,x0i; PMAP(x0r,x0i,b_r[0],b_i[0],c_r[0],c_i[0],q);
                VD x1r,x1i; PMAP(x1r,x1i,b_r[1],b_i[1],c_r[1],c_i[1],q);
                VD x2r,x2i; PMAP(x2r,x2i,b_r[2],b_i[2],c_r[2],c_i[2],q);
                VD x3r,x3i; PMAP(x3r,x3i,b_r[3],b_i[3],c_r[3],c_i[3],q);
                VD x4r,x4i; PMAP(x4r,x4i,b_r[4],b_i[4],c_r[4],c_i[4],q);
                VD x5r,x5i; PMAP(x5r,x5i,b_r[5],b_i[5],c_r[5],c_i[5],q);
                VD x6r,x6i; PMAP(x6r,x6i,b_r[6],b_i[6],c_r[6],c_i[6],q);
                VD x7r,x7i; PMAP(x7r,x7i,b_r[7],b_i[7],c_r[7],c_i[7],q);
                VD x8r,x8i; PMAP(x8r,x8i,b_r[8],b_i[8],c_r[8],c_i[8],q);
                if (outre) {
                    ST(o_r[0]+q,x0r); ST(o_i[0]+q,x0i);
                    ST(o_r[1]+q,x1r); ST(o_i[1]+q,x1i);
                    ST(o_r[2]+q,x2r); ST(o_i[2]+q,x2i);
                    ST(o_r[3]+q,x3r); ST(o_i[3]+q,x3i);
                    ST(o_r[4]+q,x4r); ST(o_i[4]+q,x4i);
                    ST(o_r[5]+q,x5r); ST(o_i[5]+q,x5i);
                    ST(o_r[6]+q,x6r); ST(o_i[6]+q,x6i);
                    ST(o_r[7]+q,x7r); ST(o_i[7]+q,x7i);
                    ST(o_r[8]+q,x8r); ST(o_i[8]+q,x8i);
                }
                if (last) continue;
                VD o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i,o4r,o4i,o5r,o5i,o6r,o6i,o7r,o7i,o8r,o8i;
            DFT9V(o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i,o4r,o4i,o5r,o5i,o6r,o6i,o7r,o7i,o8r,o8i,
                      x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,x4r,x4i,x5r,x5i,x6r,x6i,x7r,x7i,x8r,x8i);
                ST(t_r[0]+q,o0r); ST(t_i[0]+q,o0i);
                ST(t_r[1]+q,o1r); ST(t_i[1]+q,o1i);
                ST(t_r[2]+q,o2r); ST(t_i[2]+q,o2i);
                ST(t_r[3]+q,o3r); ST(t_i[3]+q,o3i);
                ST(t_r[4]+q,o4r); ST(t_i[4]+q,o4i);
                ST(t_r[5]+q,o5r); ST(t_i[5]+q,o5i);
                ST(t_r[6]+q,o6r); ST(t_i[6]+q,o6i);
                ST(t_r[7]+q,o7r); ST(t_i[7]+q,o7i);
                ST(t_r[8]+q,o8r); ST(t_i[8]+q,o8i);
            }
        }
    }
}
static void chain36(double* restrict re, double* restrict im,
                    const double* restrict cre, const double* restrict cim,
                    long m, double* restrict out1)
{
    prime36(re, im);
    for (long n = 0; n < m; n++) {
        int last = (n == m-1), first = (n == 0);
        double *outr = (first || last) ? re : 0, *outi = (first || last) ? im : 0;
        if (!(n & 1)) iter36_even(cre, cim, last, outr, outi);
        else          iter36_odd(cre, cim, last, outr, outi);
        if (first && out1) soa_export(re, im, out1, 46656);
    }
}
