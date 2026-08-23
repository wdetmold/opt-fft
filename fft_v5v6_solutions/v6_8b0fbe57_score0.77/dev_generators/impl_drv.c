// ---------------------------------------------------------------------------
// Per-size one-iteration drivers: A -> (y) -> W -> (z) -> A -> (x+map) -> B
// ---------------------------------------------------------------------------

static void iter6(double* restrict are, double* restrict aim,
                  double* restrict bre, double* restrict bim,
                  const double* restrict cre, const double* restrict cim)
{
    for (int x = 0; x < 6; x++) km6_y(are + 36*x, aim + 36*x, Wre_ + 36*x, Wim_ + 36*x, 0x3F);
    zpass6(Wre_, Wim_, are, aim);
    static const MK pm[5] = {0xFF,0xFF,0xFF,0xFF,0x0F};
    for (int p = 0; p < 5; p++)
        km6_x_map(are + 8*p, aim + 8*p, bre + 8*p, bim + 8*p, cre + 8*p, cim + 8*p, pm[p]);
}
static void iter8(double* restrict are, double* restrict aim,
                  double* restrict bre, double* restrict bim,
                  const double* restrict cre, const double* restrict cim)
{
    for (int x = 0; x < 8; x++) km8_y(are + 64*x, aim + 64*x, Wre_ + 64*x, Wim_ + 64*x);
    zpass8(Wre_, Wim_, are, aim);
    for (int p = 0; p < 8; p++)
        km8_x_map(are + 8*p, aim + 8*p, bre + 8*p, bim + 8*p, cre + 8*p, cim + 8*p);
}
static void iter13(double* restrict are, double* restrict aim,
                   double* restrict bre, double* restrict bim,
                   const double* restrict cre, const double* restrict cim)
{
    for (int x = 0; x < 13; x++) {
        km13_y(are + 169*x, aim + 169*x, Wre_ + 169*x, Wim_ + 169*x, 0xFF);
        km13_y(are + 169*x + 8, aim + 169*x + 8, Wre_ + 169*x + 8, Wim_ + 169*x + 8, 0x1F);
    }
    zpass13(Wre_, Wim_, are, aim, 169);
    for (int p = 0; p < 21; p++)
        km13_x_map(are + 8*p, aim + 8*p, bre + 8*p, bim + 8*p, cre + 8*p, cim + 8*p, 0xFF);
    km13_x_map(are + 168, aim + 168, bre + 168, bim + 168, cre + 168, cim + 168, 0x01);
}
static void iter17(double* restrict are, double* restrict aim,
                   double* restrict bre, double* restrict bim,
                   const double* restrict cre, const double* restrict cim)
{
    for (int x = 0; x < 17; x++) {
        km17_y(are + 289*x, aim + 289*x, Wre_ + 289*x, Wim_ + 289*x, 0xFF);
        km17_y(are + 289*x + 8, aim + 289*x + 8, Wre_ + 289*x + 8, Wim_ + 289*x + 8, 0xFF);
        km17_y(are + 289*x + 16, aim + 289*x + 16, Wre_ + 289*x + 16, Wim_ + 289*x + 16, 0x01);
    }
    zpass17(Wre_, Wim_, are, aim, 289);
    for (int p = 0; p < 36; p++)
        km17_x_map(are + 8*p, aim + 8*p, bre + 8*p, bim + 8*p, cre + 8*p, cim + 8*p, 0xFF);
    km17_x_map(are + 288, aim + 288, bre + 288, bim + 288, cre + 288, cim + 288, 0x01);
}
static void iter23(double* restrict are, double* restrict aim,
                   double* restrict bre, double* restrict bim,
                   const double* restrict cre, const double* restrict cim)
{
    for (int x = 0; x < 23; x++) {
        km23_y(are + 529*x, aim + 529*x, Wre_ + 529*x, Wim_ + 529*x, 0xFF);
        km23_y(are + 529*x + 8, aim + 529*x + 8, Wre_ + 529*x + 8, Wim_ + 529*x + 8, 0xFF);
        km23_y(are + 529*x + 16, aim + 529*x + 16, Wre_ + 529*x + 16, Wim_ + 529*x + 16, 0x7F);
    }
    zpass23(Wre_, Wim_, are, aim, 529);
    for (int p = 0; p < 66; p++)
        km23_x_map(are + 8*p, aim + 8*p, bre + 8*p, bim + 8*p, cre + 8*p, cim + 8*p, 0xFF);
    km23_x_map(are + 528, aim + 528, bre + 528, bim + 528, cre + 528, cim + 528, 0x01);
}


static void iter13ip(double* restrict are, double* restrict aim,
                   double* restrict bre, double* restrict bim,
                   const double* restrict cre, const double* restrict cim)
{
    (void)bre; (void)bim;
    for (int x = 0; x < 13; x++) {
        km13_y(are + 169*x, aim + 169*x, Wre_ + 169*x, Wim_ + 169*x, 0xFF);
        km13_y(are + 169*x + 8, aim + 169*x + 8, Wre_ + 169*x + 8, Wim_ + 169*x + 8, 0x1F);
    }
    zpass13(Wre_, Wim_, are, aim, 169);
    for (int p = 0; p < 21; p++)
        km13_x_map_ip(are + 8*p, aim + 8*p, cre + 8*p, cim + 8*p, 0xFF);
    km13_x_map_ip(are + 168, aim + 168, cre + 168, cim + 168, 0x01);
}
static void iter17ip(double* restrict are, double* restrict aim,
                   double* restrict bre, double* restrict bim,
                   const double* restrict cre, const double* restrict cim)
{
    (void)bre; (void)bim;
    for (int x = 0; x < 17; x++) {
        km17_y(are + 289*x, aim + 289*x, Wre_ + 289*x, Wim_ + 289*x, 0xFF);
        km17_y(are + 289*x + 8, aim + 289*x + 8, Wre_ + 289*x + 8, Wim_ + 289*x + 8, 0xFF);
        km17_y(are + 289*x + 16, aim + 289*x + 16, Wre_ + 289*x + 16, Wim_ + 289*x + 16, 0x01);
    }
    zpass17(Wre_, Wim_, are, aim, 289);
    for (int p = 0; p < 36; p++)
        km17_x_map_ip(are + 8*p, aim + 8*p, cre + 8*p, cim + 8*p, 0xFF);
    km17_x_map_ip(are + 288, aim + 288, cre + 288, cim + 288, 0x01);
}
static void iter23ip(double* restrict are, double* restrict aim,
                   double* restrict bre, double* restrict bim,
                   const double* restrict cre, const double* restrict cim)
{
    (void)bre; (void)bim;
    for (int x = 0; x < 23; x++) {
        km23_y(are + 529*x, aim + 529*x, Wre_ + 529*x, Wim_ + 529*x, 0xFF);
        km23_y(are + 529*x + 8, aim + 529*x + 8, Wre_ + 529*x + 8, Wim_ + 529*x + 8, 0xFF);
        km23_y(are + 529*x + 16, aim + 529*x + 16, Wre_ + 529*x + 16, Wim_ + 529*x + 16, 0x7F);
    }
    zpass23(Wre_, Wim_, are, aim, 529);
    for (int p = 0; p < 66; p++)
        km23_x_map_ip(are + 8*p, aim + 8*p, cre + 8*p, cim + 8*p, 0xFF);
    km23_x_map_ip(are + 528, aim + 528, cre + 528, cim + 528, 0x01);
}
static void iter36ip(double* restrict are, double* restrict aim,
                   double* restrict bre, double* restrict bim,
                   const double* restrict cre, const double* restrict cim)
{
    (void)bre; (void)bim;
    static const MK zm[5] = {0xFF,0xFF,0xFF,0xFF,0x0F};
    for (int x = 0; x < 36; x++) {
        double *sr = are + 1296*x, *si = aim + 1296*x;
        double *wr = Wre_ + 1296*(x & 1), *wi = Wim_ + 1296*(x & 1);
        for (int zc = 0; zc < 4; zc++) km36_y(sr + 8*zc, si + 8*zc, wr + 8*zc, wi + 8*zc, 0xFF);
        km36_y(sr + 32, si + 32, wr + 32, wi + 32, 0x0F);
        spass_gen(wr, wi, sr, si, 36, 5, zm, km36_s);
    }
    for (int p = 0; p < 162; p++)
        km36_x_map_ip(are + 8*p, aim + 8*p, cre + 8*p, cim + 8*p);
}
static void iter45ip(double* restrict are, double* restrict aim,
                   double* restrict bre, double* restrict bim,
                   const double* restrict cre, const double* restrict cim)
{
    (void)bre; (void)bim;
    static const MK zm[6] = {0xFF,0xFF,0xFF,0xFF,0xFF,0x1F};
    for (int x = 0; x < 45; x++) {
        double *sr = are + 2025*x, *si = aim + 2025*x;
        double *wr = Wre_ + 2048*(x & 1), *wi = Wim_ + 2048*(x & 1);
        for (int zc = 0; zc < 5; zc++) km45_y(sr + 8*zc, si + 8*zc, wr + 8*zc, wi + 8*zc, 0xFF);
        km45_y(sr + 40, si + 40, wr + 40, wi + 40, 0x1F);
        spass_gen(wr, wi, sr, si, 45, 6, zm, km45_s);
    }
    for (int p = 0; p < 253; p++)
        km45_x_map_ip(are + 8*p, aim + 8*p, cre + 8*p, cim + 8*p, 0xFF);
    km45_x_map_ip(are + 2024, aim + 2024, cre + 2024, cim + 2024, 0x01);
}


// IP_MODE[L-index]: 1 = in-place x+map (no ping-pong swap)
static int IPMODE(int L){ switch(L){ case 13: case 17: case 23: case 36: case 45: return 1; } return 0; }
// ---------------------------------------------------------------------------
// main entry: run the iteration chain for one size
// ---------------------------------------------------------------------------
void run_chain(int L, long B, long m,
               const double* x0, const double* c,
               double* out1, double* outm)
{
    if (B <= 0) return;
    long n = (long)L*L*L;
    long NP = ((n + 7) & ~7L) + 24;
    long SV = 2*NP + 8;
    size_t bytes = ((size_t)B*SV + 64) * sizeof(double);
    bytes = (bytes + 63) & ~(size_t)63;
    int needB = (L == 6 || L == 8);
    double *stA = (double*)aligned_alloc(64, bytes);
    double *stB = needB ? (double*)aligned_alloc(64, bytes) : 0;
    double *cb  = (double*)aligned_alloc(64, bytes);
    void (*iter)(double*, double*, double*, double*, const double*, const double*) = 0;
    switch (L) {
        case 6:  iter = iter6;  break;
        case 8:  iter = iter8;  break;
        case 13: iter = iter13ip; break;
        case 17: iter = iter17ip; break;
        case 23: iter = iter23ip; break;
        case 36: iter = iter36ip; break;
        case 45: iter = iter45ip; break;
        case 64: break;
        default: free(stA); if (stB) free(stB); free(cb); return;
    }
    long b0 = 0;
    if ((L == 6 || L == 8 || L == 13 || L == 17 || L == 23) && m > 0) {
        int rmin = (L == 6) ? 5 : (L == 13) ? 6 : (L == 23) ? 9 : 7;
        for (; b0 + 8 <= B; b0 += 8)
            vl_chain(L, m, x0 + 2*b0*n, c + 2*b0*n,
                     out1 + 2*b0*n, outm + 2*b0*n, 8);
        if (B - b0 >= rmin) {
            vl_chain(L, m, x0 + 2*b0*n, c + 2*b0*n,
                     out1 + 2*b0*n, outm + 2*b0*n, (int)(B - b0));
            b0 = B;
        }
    }
    for (long b = b0; b < B; b++) {
        double *ar = stA + b*SV, *ai = ar + NP;
        double *br = needB ? stB + b*SV : ar, *bi = needB ? br + NP : ai;
        double *cre = cb + b*SV, *cim = cre + NP;
        soa_import(ar, ai, x0 + 2*b*n, n);
        soa_import(cre, cim, c + 2*b*n, n);
        // zero pads (beyond n up to NP, plus inter-volume gap) to keep dead
        // lanes finite
        for (long i = n; i < NP; i++) { ar[i]=0; ai[i]=0; cre[i]=0; cim[i]=0; br[i]=0; bi[i]=0; }
        for (long i = 0; i < 8; i++) { ai[NP+i]=0; cim[NP+i]=0; bi[NP+i]=0; }
        if (m <= 0) { soa_export(ar, ai, out1 + 2*b*n, n); soa_export(ar, ai, outm + 2*b*n, n); continue; }
        if (L == 64) {
            chain64(ar, ai, cre, cim, m, T64re, T64im, 0, 0, out1 + 2*b*n);
            soa_export(ar, ai, outm + 2*b*n, n);
        } else if (L == 45) {
            chain45(ar, ai, cre, cim, m, out1 + 2*b*n);
            soa_export(ar, ai, outm + 2*b*n, n);
        } else if (L == 36) {
            chain36(ar, ai, cre, cim, m, out1 + 2*b*n);
            soa_export(ar, ai, outm + 2*b*n, n);
        } else {
            int ip = IPMODE(L);
            for (long it = 0; it < m; it++) {
                iter(ar, ai, br, bi, cre, cim);
                if (!ip) { double *t;
                    t = ar; ar = br; br = t;
                    t = ai; ai = bi; bi = t; }
                if (it == 0) soa_export(ar, ai, out1 + 2*b*n, n);
            }
            soa_export(ar, ai, outm + 2*b*n, n);
        }
    }
    free(stA); if (stB) free(stB); free(cb);
}
