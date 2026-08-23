from glib import Emit
from gen36 import gen as gen_kernels

def gen_driver():
    e = Emit()
    e("static inline __attribute__((always_inline)) void row36_in(const double* s, double* pr, double* pi){")
    e.ind += 1
    for g in range(4):
        e(f"{{ V a=VLU(s+{g*16}), b=VLU(s+{g*16+8});")
        e(f"  VS(pr+{g*8}, PERM2(a, IDXR_, b)); VS(pi+{g*8}, PERM2(a, IDXI_, b)); }}")
    e("{ V a=VLU(s+64);")
    e("  VS(pr+32, PERM2Z(0x0F, a, IDXR_, a)); VS(pi+32, PERM2Z(0x0F, a, IDXI_, a)); }")
    e.ind -= 1
    e("}")
    e("""
static void sweep36_SyX(int y0, const double* cvol, double* o1, double* om, int pre){
    double* br = ARENA36_R + (long)y0*RS36;
    double* bi = ARENA36_I + (long)y0*RS36;
    double* p1 = o1 ? o1 : om;
    double* p2 = (o1 && om) ? om : 0;
    int snap = (p1 != 0);
    for (int ch = 0; ch < 5; ch++) {
        long so = (long)y0*36 + ch*8;
        st36A_ps(br + ch*8, bi + ch*8);
        st36Bmap_ps(br + ch*8, bi + ch*8, cvol + 2*so,
                  p1 ? p1 + 2*so : 0, p2 ? p2 + 2*so : 0, 1296, ch==4, snap);
    }
    if (pre) {
        for (int b = 0; b < 5; b++) zpass36_ps(br + (long)b*8*PS36, bi + (long)b*8*PS36, b<4?8:4);
        for (int ch = 0; ch < 5; ch++) { st36A_ps(br + ch*8, bi + ch*8); st36B_ps(br + ch*8, bi + ch*8); }
    }
}
static void sweep36_PxY(int x0, const double* cvol, double* o1, double* om, int pre){
    double* br = ARENA36_R + (long)x0*PS36;
    double* bi = ARENA36_I + (long)x0*PS36;
    double* p1 = o1 ? o1 : om;
    double* p2 = (o1 && om) ? om : 0;
    int snap = (p1 != 0);
    for (int ch = 0; ch < 5; ch++) {
        long so = (long)x0*1296 + ch*8;
        st36A_rs(br + ch*8, bi + ch*8);
        st36Bmap_rs(br + ch*8, bi + ch*8, cvol + 2*so,
                  p1 ? p1 + 2*so : 0, p2 ? p2 + 2*so : 0, 36, ch==4, snap);
    }
    if (pre) {
        for (int b = 0; b < 5; b++) zpass36_rs(br + (long)b*8*RS36, bi + (long)b*8*RS36, b<4?8:4);
        for (int ch = 0; ch < 5; ch++) { st36A_rs(br + ch*8, bi + ch*8); st36B_rs(br + ch*8, bi + ch*8); }
    }
}

void run_36(const double* x0, const double* c, double* out1, double* outm, long B, long m){
    if (m < 1) m = 1;
    for (long b = 0; b < B; b++) {
        const double* xv = x0 + b*2*46656;
        const double* cv = c + b*2*46656;
        double* o1 = out1 + b*2*46656;
        double* om = outm + b*2*46656;
        for (int x = 0; x < 36; x++)
            for (int y = 0; y < 36; y++)
                row36_in(xv + ((long)x*36+y)*72, ARENA36_R + (long)x*PS36 + y*RS36,
                         ARENA36_I + (long)x*PS36 + y*RS36);
        for (int x = 0; x < 36; x++) {
            double* br = ARENA36_R + (long)x*PS36; double* bi = ARENA36_I + (long)x*PS36;
            for (int bnd = 0; bnd < 5; bnd++) zpass36_rs(br + (long)bnd*8*RS36, bi + (long)bnd*8*RS36, bnd<4?8:4);
            for (int ch = 0; ch < 5; ch++) { st36A_rs(br + ch*8, bi + ch*8); st36B_rs(br + ch*8, bi + ch*8); }
        }
        for (long t = 1; t <= m; t++) {
            double* o1t = (t==1) ? o1 : 0;
            double* omt = (t==m) ? om : 0;
            int pre = (t < m);
            if (t & 1) { for (int y0 = 0; y0 < 36; y0++) sweep36_SyX(y0, cv, o1t, omt, pre); }
            else       { for (int xp = 0; xp < 36; xp++) sweep36_PxY(xp, cv, o1t, omt, pre); }
        }
    }
}
""")
    return e.text()

def gen_all():
    from glib import PRELUDE
    return PRELUDE + gen_kernels() + gen_driver()

if __name__ == "__main__":
    open("../impl36.c","w").write(gen_all())
    print("wrote impl36.c")
