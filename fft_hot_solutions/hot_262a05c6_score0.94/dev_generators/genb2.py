import numpy as np
from genlib import hexd
from genb import Emit, DFTS, pfa_maps, cmul_const, W

def facs(L):
    if L == 36: return 4,9,'pfa'
    if L == 45: return 5,9,'pfa'
    if L == 64: return 8,8,'ct'
    raise ValueError

def maps(L):
    N1,N2,mode = facs(L)
    if mode=='pfa': return pfa_maps(N1,N2)
    return ([[8*j1+j2 for j2 in range(N2)] for j1 in range(N1)],
            [[k1+8*k2 for k2 in range(N2)] for k1 in range(N1)])

def stage1_rows(L, name, rw_in, rw_out, nslots):
    """DFT_N1 vertically across rows (row-wise sweep): reads rows inm[.][j2] of src,
       writes rows (j2*N1+k1) of dst. CT: applies W64^{k1*j2} twiddles.
       rw_in/rw_out: row strides in doubles (baked). nslots: vec slots per row."""
    N1,N2,mode = facs(L)
    inm, outm = maps(L)
    e = Emit('a')
    body = []
    for j2 in range(N2):
        unit = Emit(f'u{j2}_')
        for cc in range(nslots):
            xs=[]
            for j1 in range(N1):
                off = inm[j1][j2]*rw_in + cc*16
                xs.append((unit.v(f"_mm512_load_pd(src + {off})"), unit.v(f"_mm512_load_pd(src + {off}+8)")))
            ys = DFTS[N1](unit, xs)
            for k1 in range(N1):
                y = ys[k1]
                if mode=='ct' and (k1*j2)%64 != 0:
                    c,s = W(64, k1*j2)
                    y = cmul_const(unit, y, c, s)
                unit.raw(f"_mm512_store_pd(dst + {(j2*N1+k1)*rw_out + cc*16}, {y[0]});")
                unit.raw(f"_mm512_store_pd(dst + {(j2*N1+k1)*rw_out + cc*16+8}, {y[1]});")
        decls = "\n        ".join(unit.const_decls())
        body.append(f"    {{\n        {decls}\n{unit.code(indent='        ')}\n    }}")
    return f"""
static void __attribute__((noinline)) {name}(const double* restrict src, double* restrict dst){{
{chr(10).join(body)}
}}
"""

def stage2_rows(L, name, rw_in, rw_out, nslots, fuse_map):
    """DFT_N2 across rows: reads rows (j2*N1+k1) j2=0..N2-1 for each k1, writes rows outm[k1][k2].
       fuse_map: c rows in consumption order: c row index (k1*N2+k2)."""
    N1,N2,mode = facs(L)
    inm, outm = maps(L)
    body = []
    for k1 in range(N1):
        unit = Emit(f'v{k1}_')
        for cc in range(nslots):
            xs=[]
            for j2 in range(N2):
                off = (j2*N1+k1)*rw_in + cc*16
                xs.append((unit.v(f"_mm512_load_pd(src + {off})"), unit.v(f"_mm512_load_pd(src + {off}+8)")))
            ys = DFTS[N2](unit, xs)
            for k2 in range(N2):
                k = outm[k1][k2]
                y = ys[k2]
                if fuse_map:
                    unit.raw(f"MAPSTORE2({y[0]}, {y[1]}, dst, {k*rw_out + cc*16}, cb, {(k1*N2+k2)*rw_out + cc*16});")
                else:
                    unit.raw(f"_mm512_store_pd(dst + {k*rw_out + cc*16}, {y[0]});")
                    unit.raw(f"_mm512_store_pd(dst + {k*rw_out + cc*16+8}, {y[1]});")
        decls = "\n        ".join(unit.const_decls())
        body.append(f"    {{\n        {decls}\n{unit.code(indent='        ')}\n    }}")
    args = "const double* restrict src, double* restrict dst"
    if fuse_map: args += ", const double* restrict cb"
    return f"""
static void __attribute__((noinline)) {name}({args}){{
{chr(10).join(body)}
}}
"""
