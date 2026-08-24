import numpy as np
from genlib import E, trig, hexd

def emit_prime_dft(p, fuse_map=False):
    """emit static inline: dft_p on one pencil-group.
    src: const double* s, stride ss (doubles); dst: double* d, stride ds.
    if fuse_map: also const double* cp (stride cs) and applies z+c then map before store.
    Phase-split: fold -> C-pass (cos accums, k=1..h) -> spill d -> S-pass.
    Constants: h cos + h sin values in registers, index table IDX[j][k] = (t, sign).
    """
    h = (p-1)//2
    cos_t, sin_t = trig(p)   # cos(-2pi t/p) = cos(2pi t/p); sin(-2pi t/p) = -sin(2pi t/p)
    # For forward DFT: X_k = sum_j u_j e^{-2pi i jk/p}
    # folded: X_k = u0 + sum_{j=1..h} [ s_j cos(2pi jk/p) - i d_j sin(2pi jk/p) ]
    # cos(2pi jk/p) = cos(2pi t/p) with t = (j*k) mod p mapped to 1..h with sign +
    #   if t>h: cos(2pi t/p) = cos(2pi (p-t)/p) -> same value (cos even around p)
    # sin(2pi jk/p): t=(j*k)%p; if t<=h: +sin_tab[t]; else: -sin_tab[p-t]
    C = [float(np.cos(2*np.pi*np.longdouble(t)/p)) for t in range(p)]
    # better: exact longdouble
    jj = np.arange(p)
    ang = 2*np.longdouble('3.14159265358979323846264338327950288')*jj.astype(np.longdouble)/np.longdouble(p)
    Cl = np.cos(ang); Sl = np.sin(ang)
    def cidx(j,k):
        t = (j*k) % p
        if t > h: t = p - t
        return t  # cos positive always, value Cl[t]
    def sidx(j,k):
        t = (j*k) % p
        if t <= h: return (t, 1.0)
        return (p-t, -1.0)
    # constant table: KC[t]=cos(2pi t/p) t=1..h ; KS[t]=sin(2pi t/p) t=1..h
    tabc = ", ".join(hexd(Cl[t]) for t in range(1,h+1))
    tabs = ", ".join(hexd(Sl[t]) for t in range(1,h+1))
    lines = []
    A = lines.append
    if not fuse_map:
        A(f"static const double KC_{p}[{h}] __attribute__((aligned(64))) = {{ {tabc} }};")
        A(f"static const double KS_{p}[{h}] __attribute__((aligned(64))) = {{ {tabs} }};")
    args = "const double* restrict s, long ss, double* restrict d, long ds"
    if fuse_map: args += ", const double* restrict cp, long cs"
    A(f"static inline void __attribute__((always_inline)) dft{p}{'m' if fuse_map else ''}(" + args + "){")
    # broadcast constants
    for t in range(1,h+1):
        A(f"    __m512d kc{t} = _mm512_set1_pd(KC_{p}[{t-1}]);")
    # C-pass: load u0, fold pairs, accumulate cos sums, store d_j to scratch
    A(f"    double dscr[{2*h}*8] __attribute__((aligned(64)));")
    A(f"    __m512d u0r = _mm512_load_pd(s);")
    A(f"    __m512d u0i = _mm512_load_pd(s+8);")
    # accumulators: Pre_k, Pim_k for k=1..h, init u0
    for k in range(1,h+1):
        A(f"    __m512d pr{k} = u0r, pi{k} = u0i;")
    A(f"    __m512d x0r = u0r, x0i = u0i;")
    for j in range(1,h+1):
        A(f"    {{")
        A(f"        __m512d ar = _mm512_load_pd(s+{j}*ss), ai = _mm512_load_pd(s+{j}*ss+8);")
        A(f"        __m512d br = _mm512_load_pd(s+{p-j}*ss), bi = _mm512_load_pd(s+{p-j}*ss+8);")
        A(f"        __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);")
        A(f"        __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);")
        A(f"        _mm512_store_pd(dscr+{(2*(j-1))*8}, dr); _mm512_store_pd(dscr+{(2*(j-1)+1)*8}, di);")
        A(f"        x0r = _mm512_add_pd(x0r, sr); x0i = _mm512_add_pd(x0i, si);")
        for k in range(1,h+1):
            t = cidx(j,k)
            A(f"        pr{k} = _mm512_fmadd_pd(kc{t}, sr, pr{k}); pi{k} = _mm512_fmadd_pd(kc{t}, si, pi{k});")
        A(f"    }}")
    # store X0 and spill P to scratch? No: spill P? S-pass needs 12 more accums.
    # Plan: store P_k to scratch, run S-pass with Q accums, combine at end.
    A(f"    double pscr[{2*h}*8] __attribute__((aligned(64)));")
    for k in range(1,h+1):
        A(f"    _mm512_store_pd(pscr+{(2*(k-1))*8}, pr{k}); _mm512_store_pd(pscr+{(2*(k-1)+1)*8}, pi{k});")
    A('    __asm__ volatile("" ::: "memory");')
    if fuse_map:
        A(f"    MAPSTORE(x0r, x0i, d, 0, cp, 0);")
    else:
        A(f"    _mm512_store_pd(d, x0r); _mm512_store_pd(d+8, x0i);")
    # S-pass
    for t in range(1,h+1):
        A(f"    __m512d ks{t} = _mm512_set1_pd(KS_{p}[{t-1}]);")
    for k in range(1,h+1):
        A(f"    __m512d qr{k} = _mm512_setzero_pd(), qi{k} = _mm512_setzero_pd();")
    for j in range(1,h+1):
        A(f"    {{")
        A(f"        __m512d dr = _mm512_load_pd(dscr+{(2*(j-1))*8}), di = _mm512_load_pd(dscr+{(2*(j-1)+1)*8});")
        for k in range(1,h+1):
            t, sg = sidx(j,k)
            if sg > 0:
                A(f"        qr{k} = _mm512_fmadd_pd(ks{t}, di, qr{k}); qi{k} = _mm512_fnmadd_pd(ks{t}, dr, qi{k});")
            else:
                A(f"        qr{k} = _mm512_fnmadd_pd(ks{t}, di, qr{k}); qi{k} = _mm512_fmadd_pd(ks{t}, dr, qi{k});")
        A(f"    }}")
    # combine: X_k = P_k + Q_k ; X_{p-k} = P_k - Q_k
    for k in range(1,h+1):
        A(f"    {{")
        A(f"        __m512d Pr = _mm512_load_pd(pscr+{(2*(k-1))*8}), Pi = _mm512_load_pd(pscr+{(2*(k-1)+1)*8});")
        A(f"        __m512d xr = _mm512_add_pd(Pr, qr{k}), xi = _mm512_add_pd(Pi, qi{k});")
        A(f"        __m512d yr = _mm512_sub_pd(Pr, qr{k}), yi = _mm512_sub_pd(Pi, qi{k});")
        if fuse_map:
            A(f"        MAPSTORE(xr, xi, d, {k}*ds, cp, {k}*cs);")
            A(f"        MAPSTORE(yr, yi, d, {p-k}*ds, cp, {p-k}*cs);")
        else:
            A(f"        _mm512_store_pd(d+{k}*ds, xr); _mm512_store_pd(d+{k}*ds+8, xi);")
            A(f"        _mm512_store_pd(d+{p-k}*ds, yr); _mm512_store_pd(d+{p-k}*ds+8, yi);")
        A(f"    }}")
    A("}")
    return "\n".join(lines)

MAP_MACRO = r'''
// map z/(1+|z|) with z = (xr+cr, xi+ci); writes result to dst+off, dst+off+8
#define MAPSTORE(xr, xi, dst, off, cbase, coff) do{ \
    __m512d zr = _mm512_add_pd(xr, _mm512_load_pd((cbase)+(coff))); \
    __m512d zi = _mm512_add_pd(xi, _mm512_load_pd((cbase)+(coff)+8)); \
    __m512d m  = _mm512_fmadd_pd(zr, zr, _mm512_fmadd_pd(zi, zi, V_TINY)); \
    __m512d r0 = _mm512_rsqrt14_pd(m); \
    __m512d mr = _mm512_mul_pd(m, r0); \
    __m512d e1 = _mm512_fnmadd_pd(mr, r0, V_ONE); \
    __m512d r1 = _mm512_fmadd_pd(_mm512_mul_pd(r0, V_HALF), e1, r0); \
    __m512d mr1= _mm512_mul_pd(m, r1); \
    __m512d e2 = _mm512_fnmadd_pd(mr1, r1, V_ONE); \
    __m512d r2 = _mm512_fmadd_pd(_mm512_mul_pd(r1, V_HALF), e2, r1); \
    /* mag = m*r2 with exact fix */ \
    __m512d mg = _mm512_mul_pd(m, r2); \
    __m512d e3 = _mm512_fnmadd_pd(mg, mg, m); \
    __m512d mag= _mm512_fmadd_pd(e3, _mm512_mul_pd(r2, V_HALF), mg); \
    __m512d u  = _mm512_add_pd(V_ONE, mag); \
    __m512d w0 = _mm512_rcp14_pd(u); \
    __m512d e4 = _mm512_fnmadd_pd(u, w0, V_ONE); \
    __m512d w1 = _mm512_fmadd_pd(w0, e4, w0); \
    __m512d ee = _mm512_mul_pd(e4, e4); \
    __m512d w2 = _mm512_fmadd_pd(w1, ee, w1); \
    _mm512_store_pd((dst)+(off),   _mm512_mul_pd(zr, w2)); \
    _mm512_store_pd((dst)+(off)+8, _mm512_mul_pd(zi, w2)); \
}while(0)
'''
