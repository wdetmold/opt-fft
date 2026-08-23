import numpy as np
from genlib import hexd

PI = np.longdouble('3.14159265358979323846264338327950288')

def emit_prime_axis(p, name, stride, fuse_map=False, cstride=None, dst=None, dstride=None):
    if dst is None: dst, dstride = "x", stride
    """straight-line folded symmetric DFT-p on one vec-pencil, baked stride (doubles).
    phase-split: fold+C-pass (cos accums; d_j spilled), S-pass (sin accums), combine.
    Constants via volatile broadcasts inside each phase (no cross-phase liveness)."""
    h = (p-1)//2
    jj = np.arange(p)
    ang = 2*PI*jj.astype(np.longdouble)/np.longdouble(p)
    Cl = np.cos(ang); Sl = np.sin(ang)
    def cidx(j,k):
        t = (j*k) % p
        return t if t <= h else p-t
    def sidx(j,k):
        t = (j*k) % p
        return (t, 1.0) if t <= h else (p-t, -1.0)
    L = []
    A = L.append
    args = "double* restrict x"
    if dst != "x": args += ", double* restrict " + dst
    if fuse_map: args += ", const double* restrict cb"
    A(f"static void __attribute__((noinline)) {name}({args}){{")
    if h > 6:
        # blocked: fold (spill s,d), then per k-block C and S passes, then combine
        KB = 6
        blocks = [(k0, min(k0+KB, h+1)) for k0 in range(1, h+1, KB)]
        A(f"    double sscr[{2*h}*8] ALIGN64;")
        A(f"    double dscr[{2*h}*8] ALIGN64;")
        A(f"    double pscr[{2*h}*8+16] ALIGN64;")
        A(f"    double qscr[{2*h}*8] ALIGN64;")
        A("    {")
        A(f"    __m512d u0r = _mm512_load_pd(x), u0i = _mm512_load_pd(x+8);")
        A(f"    __m512d x0r = u0r, x0i = u0i;")
        for j in range(1,h+1):
            A("    {")
            A(f"    __m512d ar = _mm512_load_pd(x+{j*stride}), ai = _mm512_load_pd(x+{j*stride}+8);")
            A(f"    __m512d br = _mm512_load_pd(x+{(p-j)*stride}), bi = _mm512_load_pd(x+{(p-j)*stride}+8);")
            A(f"    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);")
            A(f"    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);")
            A(f"    _mm512_store_pd(sscr+{(2*(j-1))*8}, sr); _mm512_store_pd(sscr+{(2*(j-1)+1)*8}, si);")
            A(f"    _mm512_store_pd(dscr+{(2*(j-1))*8}, dr); _mm512_store_pd(dscr+{(2*(j-1)+1)*8}, di);")
            A(f"    x0r = _mm512_add_pd(x0r, sr); x0i = _mm512_add_pd(x0i, si);")
            A("    }")
        if fuse_map:
            A(f"    MAPST(x0r, x0i, {dst}, 0, cb, 0);")
        else:
            A(f"    _mm512_store_pd({dst}, x0r); _mm512_store_pd({dst}+8, x0i);")
        A("    __m512d u0r_s = u0r, u0i_s = u0i;")
        A("    _mm512_store_pd(pscr, u0r_s); _mm512_store_pd(pscr+8, u0i_s);")  # stash u0 in pscr[0..] temporarily? no-keep in loop
        A("    }")
        A('    __asm__ volatile("" ::: "memory");')
        # C passes
        for (k0,k1b) in blocks:
            A("    {")
            A(f"    __m512d u0r = _mm512_load_pd(x+0*0), u0i;")
            A("    u0r = _mm512_load_pd(pscr); u0i = _mm512_load_pd(pscr+8);")
            ts = sorted(set(cidx(j,k) for j in range(1,h+1) for k in range(k0,k1b)))
            for t in ts:
                A(f"    __m512d kc{t}; BCV(kc{t}, KC_{p}[{t-1}]);")
            for k in range(k0,k1b):
                A(f"    __m512d pr{k} = u0r, pi{k} = u0i;")
            for j in range(1,h+1):
                A("    {")
                A(f"    __m512d sr = _mm512_load_pd(sscr+{(2*(j-1))*8}), si = _mm512_load_pd(sscr+{(2*(j-1)+1)*8});")
                for k in range(k0,k1b):
                    t = cidx(j,k)
                    A(f"    pr{k} = _mm512_fmadd_pd(kc{t}, sr, pr{k}); pi{k} = _mm512_fmadd_pd(kc{t}, si, pi{k});")
                A("    }")
            for k in range(k0,k1b):
                A(f"    _mm512_store_pd(pscr+{(2*(k-1))*8+16}, pr{k}); _mm512_store_pd(pscr+{(2*(k-1)+1)*8+16}, pi{k});")
            A("    }")
            A('    __asm__ volatile("" ::: "memory");')
        # S passes
        for (k0,k1b) in blocks:
            A("    {")
            ts = sorted(set(sidx(j,k)[0] for j in range(1,h+1) for k in range(k0,k1b)))
            for t in ts:
                A(f"    __m512d ks{t}; BCV(ks{t}, KS_{p}[{t-1}]);")
            for k in range(k0,k1b):
                A(f"    __m512d qr{k}, qi{k};")
            for j in range(1,h+1):
                A("    {")
                A(f"    __m512d dr = _mm512_load_pd(dscr+{(2*(j-1))*8}), di = _mm512_load_pd(dscr+{(2*(j-1)+1)*8});")
                for k in range(k0,k1b):
                    t, sg = sidx(j,k)
                    op = "_mm512_fmadd_pd" if sg>0 else "_mm512_fnmadd_pd"
                    opn = "_mm512_fnmadd_pd" if sg>0 else "_mm512_fmadd_pd"
                    if j==1:
                        if sg>0:
                            A(f"    qr{k} = _mm512_mul_pd(ks{t}, di); qi{k} = _mm512_mul_pd(ks{t}, dr);")
                        else:
                            A(f"    qr{k} = _mm512_mul_pd(_mm512_sub_pd(_mm512_setzero_pd(),ks{t}), di); qi{k} = _mm512_mul_pd(_mm512_sub_pd(_mm512_setzero_pd(),ks{t}), dr);")
                    else:
                        A(f"    qr{k} = {op}(ks{t}, di, qr{k}); qi{k} = {op}(ks{t}, dr, qi{k});")
                A("    }")
            for k in range(k0,k1b):
                A(f"    _mm512_store_pd(qscr+{(2*(k-1))*8}, qr{k}); _mm512_store_pd(qscr+{(2*(k-1)+1)*8}, qi{k});")
            A("    }")
            A('    __asm__ volatile("" ::: "memory");')
        # combine
        A("    {")
        for k in range(1,h+1):
            A("    {")
            A(f"    __m512d Pr = _mm512_load_pd(pscr+{(2*(k-1))*8+16}), Pi = _mm512_load_pd(pscr+{(2*(k-1)+1)*8+16});")
            A(f"    __m512d Qr = _mm512_load_pd(qscr+{(2*(k-1))*8}), Qi = _mm512_load_pd(qscr+{(2*(k-1)+1)*8});")
            A(f"    __m512d xr = _mm512_add_pd(Pr, Qr), xi = _mm512_sub_pd(Pi, Qi);")
            A(f"    __m512d yr = _mm512_sub_pd(Pr, Qr), yi = _mm512_add_pd(Pi, Qi);")
            if fuse_map:
                A(f"    MAPST(xr, xi, {dst}, {k*dstride}, cb, {k*cstride});")
                A(f"    MAPST(yr, yi, {dst}, {(p-k)*dstride}, cb, {(p-k)*cstride});")
            else:
                A(f"    _mm512_store_pd({dst}+{k*dstride}, xr); _mm512_store_pd({dst}+{k*dstride}+8, xi);")
                A(f"    _mm512_store_pd({dst}+{(p-k)*dstride}, yr); _mm512_store_pd({dst}+{(p-k)*dstride}+8, yi);")
            A("    }")
        A("    }")
        A("}")
        return "\n".join(L)
    A(f"    double dscr[{2*h}*8] ALIGN64;")
    A(f"    double pscr[{2*(h+1)}*8] ALIGN64;")
    A(f"    double sscr[{2*h}*8] ALIGN64;")
    A("    {")
    for t in range(1,h+1):
        A(f"    __m512d kc{t}; BCV(kc{t}, KC_{p}[{t-1}]);")
    A(f"    __m512d u0r = _mm512_load_pd(x), u0i = _mm512_load_pd(x+8);")
    for k in range(1,h+1):
        A(f"    __m512d pr{k} = u0r, pi{k} = u0i;")
    for j in range(1,h+1):
        A("    {")
        A(f"    __m512d ar = _mm512_load_pd(x+{j*stride}), ai = _mm512_load_pd(x+{j*stride}+8);")
        A(f"    __m512d br = _mm512_load_pd(x+{(p-j)*stride}), bi = _mm512_load_pd(x+{(p-j)*stride}+8);")
        A(f"    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);")
        A(f"    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);")
        A(f"    _mm512_store_pd(dscr+{(2*(j-1))*8}, dr); _mm512_store_pd(dscr+{(2*(j-1)+1)*8}, di);")
        A(f"    _mm512_store_pd(sscr+{(2*(j-1))*8}, sr); _mm512_store_pd(sscr+{(2*(j-1)+1)*8}, si);")
        for k in range(1,h+1):
            t = cidx(j,k)
            A(f"    pr{k} = _mm512_fmadd_pd(kc{t}, sr, pr{k}); pi{k} = _mm512_fmadd_pd(kc{t}, si, pi{k});")
        A("    }")
    for k in range(1,h+1):
        A(f"    _mm512_store_pd(pscr+{(2*(k-1))*8}, pr{k}); _mm512_store_pd(pscr+{(2*(k-1)+1)*8}, pi{k});")
    A("    }")
    A('    __asm__ volatile("" ::: "memory");')
    A("    {")
    A(f"    __m512d x0r = _mm512_load_pd(x), x0i = _mm512_load_pd(x+8);")
    for j in range(1,h+1):
        A(f"    x0r = _mm512_add_pd(x0r, _mm512_load_pd(sscr+{(2*(j-1))*8})); x0i = _mm512_add_pd(x0i, _mm512_load_pd(sscr+{(2*(j-1)+1)*8}));")
    A(f"    _mm512_store_pd(pscr+{2*h*8}, x0r); _mm512_store_pd(pscr+{(2*h+1)*8}, x0i);")
    A("    }")
    A('    __asm__ volatile("" ::: "memory");')
    A("    {")
    for t in range(1,h+1):
        A(f"    __m512d ks{t}; BCV(ks{t}, KS_{p}[{t-1}]);")
    for k in range(1,h+1):
        A(f"    __m512d qr{k}, qi{k};")
    for j in range(1,h+1):
        A("    {")
        A(f"    __m512d dr = _mm512_load_pd(dscr+{(2*(j-1))*8}), di = _mm512_load_pd(dscr+{(2*(j-1)+1)*8});")
        for k in range(1,h+1):
            t, sg = sidx(j,k)
            if j == 1:
                if sg > 0:
                    A(f"    qr{k} = _mm512_mul_pd(ks{t}, di); qi{k} = _mm512_mul_pd(ks{t}, dr);")
                else:
                    A(f"    qr{k} = _mm512_mul_pd(_mm512_sub_pd(_mm512_setzero_pd(),ks{t}), di); qi{k} = _mm512_mul_pd(_mm512_sub_pd(_mm512_setzero_pd(),ks{t}), dr);")
            else:
                if sg > 0:
                    A(f"    qr{k} = _mm512_fmadd_pd(ks{t}, di, qr{k}); qi{k} = _mm512_fmadd_pd(ks{t}, dr, qi{k});")
                else:
                    A(f"    qr{k} = _mm512_fnmadd_pd(ks{t}, di, qr{k}); qi{k} = _mm512_fnmadd_pd(ks{t}, dr, qi{k});")
        A("    }")
    # combine + store (qi sign: X_k_im = P_i - Q_i where Q_i = sum ks*dr)
    A(f"    __m512d x0r = _mm512_load_pd(pscr+{2*h*8}), x0i = _mm512_load_pd(pscr+{(2*h+1)*8});")
    if fuse_map:
        A(f"    MAPST(x0r, x0i, {dst}, 0, cb, 0);")
    else:
        A(f"    _mm512_store_pd({dst}, x0r); _mm512_store_pd({dst}+8, x0i);")
    for k in range(1,h+1):
        A("    {")
        A(f"    __m512d Pr = _mm512_load_pd(pscr+{(2*(k-1))*8}), Pi = _mm512_load_pd(pscr+{(2*(k-1)+1)*8});")
        A(f"    __m512d xr = _mm512_add_pd(Pr, qr{k}), xi = _mm512_sub_pd(Pi, qi{k});")
        A(f"    __m512d yr = _mm512_sub_pd(Pr, qr{k}), yi = _mm512_add_pd(Pi, qi{k});")
        if fuse_map:
            A(f"    MAPST2Z(xr, xi, {k*dstride}, {k*cstride}, yr, yi, {(p-k)*dstride}, {(p-k)*cstride}, {dst}, cb);")
        else:
            A(f"    _mm512_store_pd({dst}+{k*dstride}, xr); _mm512_store_pd({dst}+{k*dstride}+8, xi);")
            A(f"    _mm512_store_pd({dst}+{(p-k)*dstride}, yr); _mm512_store_pd({dst}+{(p-k)*dstride}+8, yi);")
        A("    }")
    A("    }")
    A("}")
    return "\n".join(L)

def tables(p):
    h=(p-1)//2
    jj = np.arange(p)
    ang = 2*PI*jj.astype(np.longdouble)/np.longdouble(p)
    Cl = np.cos(ang); Sl = np.sin(ang)
    tc = ", ".join(hexd(Cl[t]) for t in range(1,h+1))
    ts = ", ".join(hexd(Sl[t]) for t in range(1,h+1))
    return f"static const double KC_{p}[{h}] ALIGN64 = {{ {tc} }};\nstatic const double KS_{p}[{h}] ALIGN64 = {{ {ts} }};\n"

MAP_MACRO2 = r'''
#define MAPST2Z(xr1,xi1,off1,coff1, xr2,xi2,off2,coff2, dst, cbase) do{ \
    __m512d zr1 = _mm512_add_pd(xr1, _mm512_load_pd((cbase)+(coff1))); \
    __m512d zr2 = _mm512_add_pd(xr2, _mm512_load_pd((cbase)+(coff2))); \
    __m512d zi1 = _mm512_add_pd(xi1, _mm512_load_pd((cbase)+(coff1)+8)); \
    __m512d zi2 = _mm512_add_pd(xi2, _mm512_load_pd((cbase)+(coff2)+8)); \
    __m512d m1 = _mm512_fmadd_pd(zr1, zr1, _mm512_fmadd_pd(zi1, zi1, V_TINY)); \
    __m512d m2 = _mm512_fmadd_pd(zr2, zr2, _mm512_fmadd_pd(zi2, zi2, V_TINY)); \
    __m512d r01 = _mm512_rsqrt14_pd(m1); \
    __m512d r02 = _mm512_rsqrt14_pd(m2); \
    __m512d mg01= _mm512_mul_pd(m1, r01); \
    __m512d mg02= _mm512_mul_pd(m2, r02); \
    __m512d t1 = _mm512_mul_pd(mg01, r01); \
    __m512d t2 = _mm512_mul_pd(mg02, r02); \
    __m512d e1 = _mm512_fnmadd_pd(t1, V_HALF, V_15); \
    __m512d e2 = _mm512_fnmadd_pd(t2, V_HALF, V_15); \
    __m512d mg11= _mm512_mul_pd(mg01, e1); \
    __m512d mg12= _mm512_mul_pd(mg02, e2); \
    __m512d r11 = _mm512_mul_pd(r01, e1); \
    __m512d r12 = _mm512_mul_pd(r02, e2); \
    __m512d e31 = _mm512_fnmadd_pd(mg11, mg11, m1); \
    __m512d e32 = _mm512_fnmadd_pd(mg12, mg12, m2); \
    __m512d hr1 = _mm512_mul_pd(r11, V_HALF); \
    __m512d hr2 = _mm512_mul_pd(r12, V_HALF); \
    __m512d u1  = _mm512_add_pd(V_ONE, mg11); \
    __m512d u2  = _mm512_add_pd(V_ONE, mg12); \
    u1 = _mm512_fmadd_pd(e31, hr1, u1); \
    u2 = _mm512_fmadd_pd(e32, hr2, u2); \
    __m512d w01 = _mm512_rcp14_pd(u1); \
    __m512d w02 = _mm512_rcp14_pd(u2); \
    __m512d e41 = _mm512_fnmadd_pd(u1, w01, V_ONE); \
    __m512d e42 = _mm512_fnmadd_pd(u2, w02, V_ONE); \
    __m512d w11 = _mm512_fmadd_pd(w01, e41, w01); \
    __m512d w12 = _mm512_fmadd_pd(w02, e42, w02); \
    __m512d ee1 = _mm512_mul_pd(e41, e41); \
    __m512d ee2 = _mm512_mul_pd(e42, e42); \
    __m512d w21 = _mm512_fmadd_pd(w11, ee1, w11); \
    __m512d w22 = _mm512_fmadd_pd(w12, ee2, w12); \
    _mm512_store_pd((dst)+(off1),   _mm512_mul_pd(zr1, w21)); \
    _mm512_store_pd((dst)+(off1)+8, _mm512_mul_pd(zi1, w21)); \
    _mm512_store_pd((dst)+(off2),   _mm512_mul_pd(zr2, w22)); \
    _mm512_store_pd((dst)+(off2)+8, _mm512_mul_pd(zi2, w22)); \
}while(0)

#define BCV(dst, mem) dst = _mm512_set1_pd(*(volatile const double*)&(mem))
// 17-VOP map: z=(xr+c, xi+c); out = z/(1+|z|)
#define MAPST(xr_, xi_, dst, off, cbase, coff) do{ \
    __m512d zr = _mm512_add_pd(xr_, _mm512_load_pd((cbase)+(coff))); \
    __m512d zi = _mm512_add_pd(xi_, _mm512_load_pd((cbase)+(coff)+8)); \
    __m512d mm = _mm512_fmadd_pd(zr, zr, _mm512_fmadd_pd(zi, zi, V_TINY)); \
    __m512d r0 = _mm512_rsqrt14_pd(mm); \
    __m512d mg0= _mm512_mul_pd(mm, r0); \
    __m512d t_ = _mm512_mul_pd(mg0, r0); \
    __m512d e_ = _mm512_fnmadd_pd(t_, V_HALF, V_15); \
    __m512d mg1= _mm512_mul_pd(mg0, e_); \
    __m512d r1 = _mm512_mul_pd(r0, e_); \
    __m512d e3 = _mm512_fnmadd_pd(mg1, mg1, mm); \
    __m512d hr = _mm512_mul_pd(r1, V_HALF); \
    __m512d u  = _mm512_add_pd(V_ONE, mg1); \
    u = _mm512_fmadd_pd(e3, hr, u); \
    __m512d w0 = _mm512_rcp14_pd(u); \
    __m512d e4 = _mm512_fnmadd_pd(u, w0, V_ONE); \
    __m512d w1 = _mm512_fmadd_pd(w0, e4, w0); \
    __m512d ee = _mm512_mul_pd(e4, e4); \
    __m512d w2 = _mm512_fmadd_pd(w1, ee, w1); \
    _mm512_store_pd((dst)+(off),   _mm512_mul_pd(zr, w2)); \
    _mm512_store_pd((dst)+(off)+8, _mm512_mul_pd(zi, w2)); \
}while(0)
'''

def gen_prime_run2(p, mode='pp'):
    L2, L3 = p*p, p*p*p
    s = [tables(p)]
    if mode == 'pp':
        s.append(emit_prime_axis(p, f"dz_{p}", 16))
        s.append(emit_prime_axis(p, f"dy_{p}", p*16, dst="d", dstride=L2*16))
        s.append(emit_prime_axis(p, f"dx_{p}", 16, fuse_map=True, cstride=16, dst="d", dstride=L2*16))
        s.append(f"""
static void step2_{p}(double* restrict G, double* restrict G2, const double* restrict CP){{
    for(int x=0; x<{p}; x++){{
        double* pl = G + (long)x*{L2}*16;
        for(int y=0; y<{p}; y++) dz_{p}(pl + (long)y*{p}*16);
        for(int z=0; z<{p}; z++) dy_{p}(pl + (long)z*16, G2 + ((long)z*{p} + x)*16);
    }}
    for(int e=0; e<{L2}; e++)
        dx_{p}(G2 + (long)e*{p}*16, G + (long)e*16, CP + (long)e*{p}*16);
}}
""")
    else:
        s.append(emit_prime_axis(p, f"dz_{p}", 16))
        s.append(emit_prime_axis(p, f"dy_{p}", p*16))
        s.append(emit_prime_axis(p, f"dx_{p}", L2*16, fuse_map=True, cstride=16))
        s.append(f"""
static void step2_{p}(double* restrict G, double* restrict G2, const double* restrict CP){{
    (void)G2;
    for(int x=0; x<{p}; x++){{
        double* pl = G + (long)x*{L2}*16;
        for(int y=0; y<{p}; y++) dz_{p}(pl + (long)y*{p}*16);
        for(int z=0; z<{p}; z++) dy_{p}(pl + (long)z*16);
    }}
    for(int e=0; e<{L2}; e++)
        dx_{p}(G + (long)e*16, CP + (long)e*{p}*16);
}}
""")
    s.append(f"""
void run2_{p}(const double* x0, const double* c, double* out1, double* outm, long B, long m){{
    if(!G_{p}){{ G_{p} = alloc_arena({L3}*16*8); G2_{p} = alloc_arena({L3}*16*8 + 4096) + 256; CP_{p} = alloc_arena({L3}*16*8 + 65536) + 128; CT_{p} = alloc_arena({L3}*16*8); }}
    long G8 = B/8;
    for(long g=0; g<G8; g++){{
        const double* sx[8]; const double* sc[8]; double* d1[8]; double* dm[8];
        for(int v=0; v<8; v++){{
            long off = (g*8+v)*(long){L3}*2;
            sx[v] = x0+off; sc[v] = c+off; d1[v] = out1+off; dm[v] = outm+off;
        }}
        conv_in_{p}(sx, G_{p});
        conv_in_{p}(sc, CT_{p});
        for(long e=0; e<{L2}; e++)
            for(int j=0; j<{p}; j++)
                memcpy(CP_{p} + (e*(long){p} + j)*16, CT_{p} + ((long)j*{L2} + e)*16, 128);
        for(long t=0; t<m; t++){{
            step2_{p}(G_{p}, G2_{p}, CP_{p});
            if(t==0 && m>1) conv_out_{p}(G_{p}, d1);
        }}
        conv_out_{p}(G_{p}, dm);
        if(m==1) for(int v=0; v<8; v++) memcpy(d1[v], dm[v], {L3}*16);
    }}
}}
""")
    return "\n".join(s)
