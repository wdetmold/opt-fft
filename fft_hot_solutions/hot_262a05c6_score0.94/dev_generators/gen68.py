import numpy as np
from genlib import hexd
from genb import Emit, DFTS, pfa_maps

def emit_small_axis(L, name, stride, fuse_map=False, cstride=None, dst=None, dstride=None):
    """straight-line DFT-L (L=6 PFA(2,3), L=8 radix) on one vec-pencil, baked strides, optional fused map."""
    if dst is None: dst, dstride = "x", stride
    e = Emit('s')
    if L == 6:
        inm, outm = pfa_maps(2,3)
        xs = {}
        for j in range(6):
            xs[j] = (e.v(f"_mm512_load_pd(x+{j*stride})"), e.v(f"_mm512_load_pd(x+{j*stride}+8)"))
        # stage1: 3 DFT2 over j1 for each j2: inputs inm[j1][j2]
        t = {}
        for j2 in range(3):
            a, b = xs[inm[0][j2]], xs[inm[1][j2]]
            t[(0,j2)] = (e.v(f"_mm512_add_pd({a[0]},{b[0]})"), e.v(f"_mm512_add_pd({a[1]},{b[1]})"))
            t[(1,j2)] = (e.v(f"_mm512_sub_pd({a[0]},{b[0]})"), e.v(f"_mm512_sub_pd({a[1]},{b[1]})"))
        outs = {}
        for k1 in range(2):
            ys = DFTS[3](e, [t[(k1,j2)] for j2 in range(3)])
            for k2 in range(3):
                outs[outm[k1][k2]] = ys[k2]
    elif L == 8:
        xs = [ (e.v(f"_mm512_load_pd(x+{j*stride})"), e.v(f"_mm512_load_pd(x+{j*stride}+8)")) for j in range(8) ]
        ys = DFTS[8](e, xs)
        outs = {k: ys[k] for k in range(8)}
    else:
        raise ValueError
    L2 = []
    A = L2.append
    args = "const double* restrict x" if dst != "x" else "double* restrict x"
    if dst != "x": args += ", double* restrict " + dst
    if fuse_map: args += ", const double* restrict cb"
    A(f"static void __attribute__((always_inline)) inline {name}({args}){{")
    decls = "\n    ".join(e.const_decls())
    A("    " + decls if decls else "")
    for l in e.lines:
        A("    " + l)
    for k in range(L):
        vr, vi = outs[k]
        if fuse_map:
            A(f"    MAPST({vr}, {vi}, {dst}, {k*dstride}, cb, {k*cstride});")
        else:
            A(f"    _mm512_store_pd({dst}+{k*dstride}, {vr}); _mm512_store_pd({dst}+{k*dstride}+8, {vi});")
    A("}")
    return "\n".join(L2)

def gen_small_run(L):
    L2v, L3 = L*L, L*L*L
    s = []
    s.append(emit_small_axis(L, f"ez_{L}", 16))
    s.append(emit_small_axis(L, f"ey_{L}", L*16, dst="d", dstride=L2v*16))
    s.append(emit_small_axis(L, f"ex_{L}", 16, fuse_map=True, cstride=16, dst="d", dstride=L2v*16))
    s.append(f"""
static void estep_{L}(double* restrict G, double* restrict G2, const double* restrict CP){{
    for(int x=0; x<{L}; x++){{
        double* pl = G + (long)x*{L2v}*16;
        for(int y=0; y<{L}; y++) ez_{L}(pl + (long)y*{L}*16);
        for(int z=0; z<{L}; z++) ey_{L}(pl + (long)z*16, G2 + ((long)z*{L} + x)*16);
    }}
    for(int e=0; e<{L2v}; e++)
        ex_{L}(G2 + (long)e*{L}*16, G + (long)e*16, CP + (long)e*{L}*16);
}}
static double* EG_{L} = 0; static double* EG2_{L} = 0; static double* EC_{L} = 0; static double* ET_{L} = 0;
void erun_{L}(const double* x0, const double* c, double* out1, double* outm, long B, long m){{
    if(!EG_{L}){{ EG_{L} = alloc_arena({L3}*16*8); EG2_{L} = alloc_arena({L3}*16*8+4096)+128; EC_{L} = alloc_arena({L3}*16*8+8192)+64; ET_{L} = alloc_arena({L3}*16*8); }}
    long G8 = B/8;
    for(long g=0; g<G8; g++){{
        const double* sx[8]; const double* sc[8]; double* d1[8]; double* dm[8];
        for(int v=0; v<8; v++){{
            long off = (g*8+v)*(long){L3}*2;
            sx[v] = x0+off; sc[v] = c+off; d1[v] = out1+off; dm[v] = outm+off;
        }}
        conv_in_{L}(sx, EG_{L});
        conv_in_{L}(sc, ET_{L});
        for(long e=0; e<{L2v}; e++)
            for(int j=0; j<{L}; j++)
                memcpy(EC_{L} + (e*(long){L} + j)*16, ET_{L} + ((long)j*{L2v} + e)*16, 128);
        for(long t=0; t<m; t++){{
            estep_{L}(EG_{L}, EG2_{L}, EC_{L});
            if(t==0 && m>1) conv_out_{L}(EG_{L}, d1);
        }}
        conv_out_{L}(EG_{L}, dm);
        if(m==1) for(int v=0; v<8; v++) memcpy(d1[v], dm[v], {L3}*16);
    }}
}}
""")
    return "\n".join(s)
