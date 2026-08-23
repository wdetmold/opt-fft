"""Patch merged implementation: replace 3f30 run64 iteration with 1-sweep/step alternation."""
import sys
src = open(sys.argv[1]).read()

new_code = r'''
/* ---- alternation-scheduled L=64 driver (1 memory sweep per step) ---- */
static __attribute__((always_inline)) inline
void x64lane_plain(vd *restrict X, ptrdiff_t q) {
    const ptrdiff_t S = 2*SA64;
    vd R[8], I[8];
    vd *p = X + 2*q;
    for (int a = 0; a < 8; a++) { R[a] = p[a*S]; I[a] = p[a*S + 1]; }
    bf8(&R[0],&I[0],&R[1],&I[1],&R[2],&I[2],&R[3],&I[3],
        &R[4],&I[4],&R[5],&I[5],&R[6],&I[6],&R[7],&I[7]);
    for (int k1 = 1; k1 < 8; k1++) {
        vd gr = R[k1], gi = I[k1];
        R[k1] = gr * ZTWR[k1] - gi * ZTWI[k1];
        I[k1] = gr * ZTWI[k1] + gi * ZTWR[k1];
    }
    tr8(R); tr8(I);
    bf8(&R[0],&I[0],&R[1],&I[1],&R[2],&I[2],&R[3],&I[3],
        &R[4],&I[4],&R[5],&I[5],&R[6],&I[6],&R[7],&I[7]);
    for (int k2 = 0; k2 < 8; k2++) { p[k2*S] = R[k2]; p[k2*S + 1] = I[k2]; }
}
static __attribute__((always_inline)) inline
void x64lane_map_pre(vd *restrict X, const vd *restrict C, ptrdiff_t q) {
    const ptrdiff_t S = 2*SA64;
    vd R[8], I[8];
    vd *p = X + 2*q;
    const vd *c = C + 2*q;
    for (int a = 0; a < 8; a++) { R[a] = p[a*S]; I[a] = p[a*S + 1]; }
    bf8(&R[0],&I[0],&R[1],&I[1],&R[2],&I[2],&R[3],&I[3],
        &R[4],&I[4],&R[5],&I[5],&R[6],&I[6],&R[7],&I[7]);
    for (int k1 = 1; k1 < 8; k1++) {
        vd gr = R[k1], gi = I[k1];
        R[k1] = gr * ZTWR[k1] - gi * ZTWI[k1];
        I[k1] = gr * ZTWI[k1] + gi * ZTWR[k1];
    }
    tr8(R); tr8(I);
    bf8(&R[0],&I[0],&R[1],&I[1],&R[2],&I[2],&R[3],&I[3],
        &R[4],&I[4],&R[5],&I[5],&R[6],&I[6],&R[7],&I[7]);
    /* + c, map; keep mapped in regs, then second four-step (x of t+1) */
    for (int k2 = 0; k2 < 8; k2++) {
        vd zr = R[k2] + c[k2*S], zi = I[k2] + c[k2*S + 1];
        mapv(&zr, &zi);
        R[k2] = zr; I[k2] = zi;
    }
    /* second x-DFT: input element a of the lane-pencil lives at (R,I)[a] with
       the SAME (reg,lane) meaning as the loads above, since the first pass
       stored outputs k2 to slot k2 (natural order). */
    bf8(&R[0],&I[0],&R[1],&I[1],&R[2],&I[2],&R[3],&I[3],
        &R[4],&I[4],&R[5],&I[5],&R[6],&I[6],&R[7],&I[7]);
    for (int k1 = 1; k1 < 8; k1++) {
        vd gr = R[k1], gi = I[k1];
        R[k1] = gr * ZTWR[k1] - gi * ZTWI[k1];
        I[k1] = gr * ZTWI[k1] + gi * ZTWR[k1];
    }
    tr8(R); tr8(I);
    bf8(&R[0],&I[0],&R[1],&I[1],&R[2],&I[2],&R[3],&I[3],
        &R[4],&I[4],&R[5],&I[5],&R[6],&I[6],&R[7],&I[7]);
    for (int k2 = 0; k2 < 8; k2++) { p[k2*S] = R[k2]; p[k2*S + 1] = I[k2]; }
}
static void sweepX64_plain(vd *restrict X) {
    for (int y = 0; y < 64; y++)
        for (int z = 0; z < 64; z++)
            x64lane_plain(X, (ptrdiff_t)y*RY64 + z);
}
static void sweepX64_mapfin(vd *restrict X, const vd *restrict C) {
    for (int y = 0; y < 64; y++)
        for (int z = 0; z < 64; z++)
            x64lane(X, C, (ptrdiff_t)y*RY64 + z);
}
static void sweepX64_mappre(vd *restrict X, const vd *restrict C) {
    for (int y = 0; y < 64; y++)
        for (int z = 0; z < 64; z++)
            x64lane_map_pre(X, C, (ptrdiff_t)y*RY64 + z);
}
/* sweep A: per a-slab: Y(t); Z(t)+c+map; [Z(t+1); Y(t+1)] */
static void sweepA64(vd *restrict X, const vd *restrict C, int pre) {
    for (int a = 0; a < 8; a++) {
        vd *slab = X + 2*(ptrdiff_t)a*SA64;
        const vd *cslab = C + 2*(ptrdiff_t)a*SA64;
        for (int z = 0; z < 64; z++) k64_p(slab + 2*z, slab + 2*z + 1, 2*RY64);
        for (int y = 0; y < 64; y++) {
            vd *p = slab + 2*(ptrdiff_t)y*RY64;
            const vd *cp = cslab + 2*(ptrdiff_t)y*RY64;
            k64_m(p, p + 1, 2, cp, cp + 1);
        }
        if (pre) {
            for (int y = 0; y < 64; y++) k64_p(slab + 2*(ptrdiff_t)y*RY64, slab + 2*(ptrdiff_t)y*RY64 + 1, 2);
            for (int z = 0; z < 64; z++) k64_p(slab + 2*z, slab + 2*z + 1, 2*RY64);
        }
    }
}
/* pre-only pass: Z(t+1), Y(t+1) on mapped state (used after captures) */
static void sweepPre64(vd *restrict X) {
    for (int a = 0; a < 8; a++) {
        vd *slab = X + 2*(ptrdiff_t)a*SA64;
        for (int y = 0; y < 64; y++) k64_p(slab + 2*(ptrdiff_t)y*RY64, slab + 2*(ptrdiff_t)y*RY64 + 1, 2);
        for (int z = 0; z < 64; z++) k64_p(slab + 2*z, slab + 2*z + 1, 2*RY64);
    }
}
void run64_alt(long long Bll, long long mll, const double *x0, const double *c0,
               double *out1, double *outm) {
    const long NV = 262144; const long B = (long)Bll;
    long m = (long)mll; if (m < 1) m = 1;
    for (long v = 0; v < B; v++) {
        conv_in_64(x0 + (long long)v*NV*2, XP_);
        conv_in_64(c0 + (long long)v*NV*2, CP_);
        sweepX64_plain(XP_);               /* pre-transform x for step 1 */
        sweepA64(XP_, CP_, 0);             /* step 1 complete; state mapped */
        conv_out_64(XP_, out1 + (long long)v*NV*2);
        if (m == 1) { conv_out_64(XP_, outm + (long long)v*NV*2); continue; }
        sweepPre64(XP_);                   /* z,y of step 2 */
        for (long t = 2; t <= m; t++) {
            if ((t & 1) == 0) {            /* even step: x + map (+x pre) */
                if (t < m) sweepX64_mappre(XP_, CP_);
                else       sweepX64_mapfin(XP_, CP_);
            } else {                       /* odd step: y,z + map (+z,y pre) */
                sweepA64(XP_, CP_, t < m);
            }
        }
        conv_out_64(XP_, outm + (long long)v*NV*2);
    }
}
'''

anchor = 'int probe(void) { return 512; }'
assert anchor in src
src = src.replace(anchor, new_code + '\n' + anchor)
# switch dispatcher to run64_alt
src = src.replace('    run64(B, m, x0, c, out1, outm);\n',
                  '    run64_alt(B, m, x0, c, out1, outm);\n')
open(sys.argv[1], 'w').write(src)
print('patched')
