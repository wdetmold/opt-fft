import numpy as np
from genlib import hexd
from genb import Emit, DFTS, pfa_maps

def dftL(e, L, xs):
    if L == 6:
        inm, outm = pfa_maps(2,3)
        t = {}
        for j2 in range(3):
            a, b = xs[inm[0][j2]], xs[inm[1][j2]]
            t[(0,j2)] = (e.v(f"_mm512_add_pd({a[0]},{b[0]})"), e.v(f"_mm512_add_pd({a[1]},{b[1]})"))
            t[(1,j2)] = (e.v(f"_mm512_sub_pd({a[0]},{b[0]})"), e.v(f"_mm512_sub_pd({a[1]},{b[1]})"))
        outs = [None]*6
        for k1 in range(2):
            ys = DFTS[3](e, [t[(k1,j2)] for j2 in range(3)])
            for k2 in range(3):
                outs[pfa_maps(2,3)[1][k1][k2]] = ys[k2]
        return outs
    else:
        return DFTS[8](e, xs)

def emit_plain_pair(L, name, stride):
    """two pencils zipped, no map (for z/y passes)"""
    e = Emit('t')
    xs1 = []; xs2 = []
    for j in range(L):
        xs1.append((e.v(f"_mm512_load_pd(x+{j*stride})"), e.v(f"_mm512_load_pd(x+{j*stride}+8)")))
        xs2.append((e.v(f"_mm512_load_pd(y+{j*stride})"), e.v(f"_mm512_load_pd(y+{j*stride}+8)")))
    o1 = dftL(e, L, xs1)
    o2 = dftL(e, L, xs2)
    for k in range(L):
        e.raw(f"_mm512_store_pd(x+{k*stride}, {o1[k][0]}); _mm512_store_pd(x+{k*stride}+8, {o1[k][1]});")
        e.raw(f"_mm512_store_pd(y+{k*stride}, {o2[k][0]}); _mm512_store_pd(y+{k*stride}+8, {o2[k][1]});")
    decls = "\n    ".join(e.const_decls())
    return f"""
static void __attribute__((noinline)) {name}(double* restrict x, double* restrict y){{
    {decls}
{e.code()}
}}
"""

def emit_x_pipe(L, name, PSZ):
    """pipelined x-pass codelet variants over pencil stream (stride PSZ doubles between vec-pts):
       {name}_st(nxt, cb): DFT(nxt)+c -> ZA ;
       {name}_ab(dst_prev, nxt, cb): maps(ZA)->dst_prev ZIPPED with DFT(nxt)+c->ZB ; _ba swaps ;
       {name}_fa/fb(dst): maps only."""
    out = []
    def dft_into(e, scw, pfx):
        xs = []
        for j in range(L):
            xs.append((e.v(f"_mm512_load_pd(nxt+{j*PSZ})"), e.v(f"_mm512_load_pd(nxt+{j*PSZ}+8)")))
        os_ = dftL(e, L, xs)
        for k in range(L):
            zr = e.v(f"_mm512_add_pd({os_[k][0]}, _mm512_load_pd(cb+{k*16}))")
            zi = e.v(f"_mm512_add_pd({os_[k][1]}, _mm512_load_pd(cb+{k*16}+8))")
            e.raw(f"_mm512_store_pd({scw}+{k*16}, {zr}); _mm512_store_pd({scw}+{k*16}+8, {zi});")
    def map_unit(e, scr, k):
        e.raw(f"""{{
    __m512d zr = _mm512_load_pd({scr}+{k*16});
    __m512d zi = _mm512_load_pd({scr}+{k*16}+8);
    __m512d mm = _mm512_fmadd_pd(zr, zr, _mm512_fmadd_pd(zi, zi, V_TINY));
    __m512d r0 = _mm512_rsqrt14_pd(mm);
    __m512d mg0= _mm512_mul_pd(mm, r0);
    __m512d t_ = _mm512_mul_pd(mg0, r0);
    __m512d e_ = _mm512_fnmadd_pd(t_, V_HALF, V_15);
    __m512d mg1= _mm512_mul_pd(mg0, e_);
    __m512d r1 = _mm512_mul_pd(r0, e_);
    __m512d e3 = _mm512_fnmadd_pd(mg1, mg1, mm);
    __m512d hr = _mm512_mul_pd(r1, V_HALF);
    __m512d u  = _mm512_add_pd(V_ONE, mg1);
    u = _mm512_fmadd_pd(e3, hr, u);
    __m512d w0 = _mm512_rcp14_pd(u);
    __m512d e4 = _mm512_fnmadd_pd(u, w0, V_ONE);
    __m512d w1 = _mm512_fmadd_pd(w0, e4, w0);
    __m512d ee = _mm512_mul_pd(e4, e4);
    __m512d w2 = _mm512_fmadd_pd(w1, ee, w1);
    _mm512_store_pd(dst+{k*PSZ}, _mm512_mul_pd(zr, w2));
    _mm512_store_pd(dst+{k*PSZ}+8, _mm512_mul_pd(zi, w2));
}}""")
    ZA, ZB = f"ZA_{name}", f"ZB_{name}"
    # st
    e = Emit('u')
    dft_into(e, ZA, 'a')
    out.append(f"static void __attribute__((noinline)) {name}_st(const double* restrict nxt, const double* restrict cb){{\n    " +
               "\n    ".join(e.const_decls()) + "\n" + e.code() + "\n}\n")
    # ab / ba: zip maps (L units) with dft (one big unit split): emit map k, then a slice of DFT ops
    for sfx, scr, scw in (("ab", ZA, ZB), ("ba", ZB, ZA)):
        e = Emit('v')
        # build DFT ops into a temp emitter to slice
        ed = Emit('w')
        dft_into(ed, scw, 'd')
        dops = ed.lines
        per = (len(dops) + L - 1)//L
        di = 0
        for k in range(L):
            map_unit(e, scr, k)
            for _ in range(per):
                if di < len(dops): e.raw(dops[di]); di += 1
        while di < len(dops): e.raw(dops[di]); di += 1
        decls = "\n    ".join(sorted(set(ed.const_decls() + e.const_decls())))
        out.append(f"static void __attribute__((noinline)) {name}_{sfx}(double* restrict dst, const double* restrict nxt, const double* restrict cb){{\n    " +
                   decls + "\n" + e.code() + "\n}\n")
    for sfx, scr in (("fa", ZA), ("fb", ZB)):
        e = Emit('z')
        for k in range(L):
            map_unit(e, scr, k)
        out.append(f"static void __attribute__((noinline)) {name}_{sfx}(double* restrict dst){{\n    " +
                   "\n    ".join(e.const_decls()) + "\n" + e.code() + "\n}\n")
    pre = f"static double ZA_{name}[{L}*16] ALIGN64;\nstatic double ZB_{name}[{L}*16] ALIGN64;\n"
    return pre + "\n".join(out)

def gen_small2(L):
    L2v, L3 = L*L, L*L*L
    PSZ = L2v*16
    s = []
    s.append(emit_plain_pair(L, f"fz_{L}", 16))
    s.append(emit_plain_pair(L, f"fy_{L}", L*16))
    s.append(emit_x_pipe(L, f"fx{L}", PSZ))
    s.append(f"""
static void fstep_{L}(double* restrict G, const double* restrict CP){{
    for(int x=0; x<{L}; x++){{
        double* pl = G + (long)x*{L2v}*16;
        for(int y=0; y+2<={L}; y+=2) fz_{L}(pl + (long)y*{L}*16, pl + (long)(y+1)*{L}*16);
        for(int z=0; z+2<={L}; z+=2) fy_{L}(pl + (long)z*16, pl + (long)(z+1)*16);
    }}
    fx{L}_st(G, CP);
    long e = 0;
    for(; e+2 <= {L2v}-1; e += 2){{
        fx{L}_ab(G + e*16, G + (e+1)*16, CP + (e+1)*{L}*16);
        fx{L}_ba(G + (e+1)*16, G + (e+2)*16, CP + (e+2)*{L}*16);
    }}
    if(e < {L2v}-1){{ fx{L}_ab(G + e*16, G + (e+1)*16, CP + (e+1)*{L}*16); fx{L}_fb(G + ({L2v}-1)*16); }}
    else fx{L}_fa(G + ({L2v}-1)*16);
}}
static double* FG_{L} = 0; static double* FC_{L} = 0; static double* FT_{L} = 0;
void frun_{L}(const double* x0, const double* c, double* out1, double* outm, long B, long m){{
    if(!FG_{L}){{ FG_{L} = alloc_arena({L3}*16*8); FC_{L} = alloc_arena({L3}*16*8+8192)+64; FT_{L} = alloc_arena({L3}*16*8); }}
    long G8 = B/8;
    for(long g=0; g<G8; g++){{
        const double* sx[8]; const double* sc[8]; double* d1[8]; double* dm[8];
        for(int v=0; v<8; v++){{
            long off = (g*8+v)*(long){L3}*2;
            sx[v] = x0+off; sc[v] = c+off; d1[v] = out1+off; dm[v] = outm+off;
        }}
        conv_in_{L}(sx, FG_{L});
        conv_in_{L}(sc, FT_{L});
        for(long e=0; e<{L2v}; e++)
            for(int j=0; j<{L}; j++)
                memcpy(FC_{L} + (e*(long){L} + j)*16, FT_{L} + ((long)j*{L2v} + e)*16, 128);
        for(long t=0; t<m; t++){{
            fstep_{L}(FG_{L}, FC_{L});
            if(t==0 && m>1) conv_out_{L}(FG_{L}, d1);
        }}
        conv_out_{L}(FG_{L}, dm);
        if(m==1) for(int v=0; v<8; v++) memcpy(d1[v], dm[v], {L3}*16);
    }}
}}
""")
    return "\n".join(s)
