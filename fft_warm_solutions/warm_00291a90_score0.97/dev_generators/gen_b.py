import numpy as np
from genlib import *
from gen_a import dft_graph
from gen_asm import A
import gen_asm_prime as gap

# ---------------- two-stage codelets (intrinsics) ----------------
def dft9(e, x):
    """3x3 CT DFT9 on list of 9 complex pairs"""
    # stage: rows n1: y[n1][n2'] = DFT3 over n2 of x[n2*3+n1]? use j = 3*j1 + j2
    # X[k] = sum_j x[j] w9^{jk}; j = 3*a + b: X[3*k1+k0]... standard:
    # split j by j mod 3: x_b[a] = x[3a+b]; Y_b = DFT3_a(x_b); X[k] = sum_b w9^{bk} Y_b[k mod 3]
    sub = [dft3(e, [x[3*a+b] for a in range(3)]) for b in range(3)]
    out = []
    for k in range(9):
        k3 = k % 3
        acc = sub[0][k3]
        t1 = cmulw(e, sub[1][k3], 9, k)
        t2 = cmulw(e, sub[2][k3], 9, 2*k)
        acc = cadd(e, acc, t1)
        acc = cadd(e, acc, t2)
        out.append(acc)
    return out

def gen_twostage(L, name):
    """noinline two-stage DFT: (src,ses) -> (dst,des), strides runtime args.
    Uses static scratch SC_{name}."""
    if L == 36: N1, N2, mode = 4, 9, 'pfa'
    elif L == 45: N1, N2, mode = 5, 9, 'pfa'
    elif L == 64: N1, N2, mode = 8, 8, 'ct'
    else: raise ValueError
    decl = f"static double SC_{name}[{L}][16] ALIGN64;\n"
    e1 = E(pfx="a")
    # stage 1
    if mode == 'pfa':
        inm, outm = pfa_maps(N1, N2)
        for n2 in range(N2):
            x = [(load(e1, "src", f"{inm[n1][n2]}*ses"), load(e1, "src", f"{inm[n1][n2]}*ses+8")) for n1 in range(N1)]
            y = dft_graph(e1, x, N1)
            for k1 in range(N1):
                e1.raw(f"_mm512_store_pd(SC_{name}[{k1*N2+n2}], {y[k1][0]});")
                e1.raw(f"_mm512_store_pd(SC_{name}[{k1*N2+n2}]+8, {y[k1][1]});")
    else:  # ct 8x8: j = 8*n1 + n2
        for n2 in range(N2):
            x = [(load(e1, "src", f"{8*n1+n2}*ses"), load(e1, "src", f"{8*n1+n2}*ses+8")) for n1 in range(N1)]
            y = dft_graph(e1, x, N1)
            for k1 in range(N1):
                t = cmulw(e1, y[k1], 64, k1*n2)
                e1.raw(f"_mm512_store_pd(SC_{name}[{k1*N2+n2}], {t[0]});")
                e1.raw(f"_mm512_store_pd(SC_{name}[{k1*N2+n2}]+8, {t[1]});")
    # stage 2
    e2 = E(pfx="b")
    if mode == 'pfa':
        inm, outm = pfa_maps(N1, N2)
        for k1 in range(N1):
            x = [(e2.v(f"_mm512_load_pd(SC_{name}[{k1*N2+n2}])"), e2.v(f"_mm512_load_pd(SC_{name}[{k1*N2+n2}]+8)")) for n2 in range(N2)]
            y = dft_graph(e2, x, N2) if N2 != 9 else dft9(e2, x)
            for k2 in range(N2):
                ko = outm[k1][k2]
                e2.raw(f"_mm512_store_pd(dst + {ko}*des, {y[k2][0]});")
                e2.raw(f"_mm512_store_pd(dst + {ko}*des+8, {y[k2][1]});")
    else:
        for k1 in range(N1):
            x = [(e2.v(f"_mm512_load_pd(SC_{name}[{k1*N2+n2}])"), e2.v(f"_mm512_load_pd(SC_{name}[{k1*N2+n2}]+8)")) for n2 in range(N2)]
            y = dft_graph(e2, x, N2)
            for k2 in range(N2):
                ko = k2*8 + k1
                e2.raw(f"_mm512_store_pd(dst + {ko}*des, {y[k2][0]});")
                e2.raw(f"_mm512_store_pd(dst + {ko}*des+8, {y[k2][1]});")
    return decl + f"""static void __attribute__((noinline)) {name}(const double* src, const long ses, double* dst, const long des){{
{e1.code()}
    __asm__ volatile("" ::: "memory");
{e2.code()}
}}
"""

# ---------------- asm map sweep functions ----------------
def gen_map_fns(L):
    """mp{L}_s16(xs, pc, dst): dst stride 16 (scratch); mp{L}_str(xs, pc, dst, PS) baked strides emitted per use"""
    out = []
    # generic: src XS (16), dst stride param baked at gen time via variants below
    return out

def gen_map_fn(L, name, dst_strides):
    """asm map: src = xs slots (stride 16), c = pc slots (stride 16), dsts = list of (argname, stride_doubles)"""
    a = A()
    gap.emit_map_phase(a, L, ('xs', 16), [(nm, st) for (nm, st) in dst_strides])
    assert not a.live
    body = "\\n\\t".join(a.lines)
    args = ["const double* xs", "const double* pc"] + [f"double* {nm}" for (nm, st) in dst_strides]
    ops = [f'[xs]"r"(xs)', f'[pc]"r"(pc)', f'[tab]"r"(MTB)'] + [f'[{nm}]"r"({nm})' for (nm, st) in dst_strides]
    clob = ", ".join(f'"zmm{i}"' for i in range(32)) + ', "memory"'
    return f"""static void __attribute__((noinline)) {name}({", ".join(args)}){{
    __asm__ volatile("{body}"
    : : {", ".join(ops)}
    : {clob});
}}
"""

def gen_b_driver(L):
    NV = (L + 7) // 8
    RS = NV * 16
    if L == 64: RS = 144
    YP = ((L + 7) // 8) * 8
    NYB = YP // 8
    PS = YP * RS
    if (PS * 8) % 4096 == 0: PS += 16
    L2, L3 = L*L, L*L*L
    NZS = NV * 8   # z slots in ZSCR
    name = f"dft{L}_v"
    # c-layouts: CZT slot base for (i, yb): (i*NYB+yb)*L*16 ; CP for pencil e=(y*NV+v): e*L*16
    return f"""
// ---------------- family B, L={L} (NV={NV} RS={RS} YP={YP} PS={PS}) ----------------
static double ZS_{L}[{NZS}*16] ALIGN64;
static double XSB_{L}[{L}*16] ALIGN64;
static double MMB_{L}[{L}*16] ALIGN64;
static double* XB_{L} = 0;
static double* CGB_{L} = 0;
static double* CZT_{L} = 0;
static double* CPB_{L} = 0;
static double* SNB_{L} = 0;
static inline void trin_{L}(const double* rb){{
    for(int v=0; v<{NV}; v++){{
        __m512d r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7;
        r0=_mm512_load_pd(rb + 0*{RS} + v*16); r1=_mm512_load_pd(rb + 1*{RS} + v*16);
        r2=_mm512_load_pd(rb + 2*{RS} + v*16); r3=_mm512_load_pd(rb + 3*{RS} + v*16);
        r4=_mm512_load_pd(rb + 4*{RS} + v*16); r5=_mm512_load_pd(rb + 5*{RS} + v*16);
        r6=_mm512_load_pd(rb + 6*{RS} + v*16); r7=_mm512_load_pd(rb + 7*{RS} + v*16);
        TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
        _mm512_store_pd(ZS_{L} + (v*8+0)*16, o0); _mm512_store_pd(ZS_{L} + (v*8+1)*16, o1);
        _mm512_store_pd(ZS_{L} + (v*8+2)*16, o2); _mm512_store_pd(ZS_{L} + (v*8+3)*16, o3);
        _mm512_store_pd(ZS_{L} + (v*8+4)*16, o4); _mm512_store_pd(ZS_{L} + (v*8+5)*16, o5);
        _mm512_store_pd(ZS_{L} + (v*8+6)*16, o6); _mm512_store_pd(ZS_{L} + (v*8+7)*16, o7);
        r0=_mm512_load_pd(rb + 0*{RS} + v*16+8); r1=_mm512_load_pd(rb + 1*{RS} + v*16+8);
        r2=_mm512_load_pd(rb + 2*{RS} + v*16+8); r3=_mm512_load_pd(rb + 3*{RS} + v*16+8);
        r4=_mm512_load_pd(rb + 4*{RS} + v*16+8); r5=_mm512_load_pd(rb + 5*{RS} + v*16+8);
        r6=_mm512_load_pd(rb + 6*{RS} + v*16+8); r7=_mm512_load_pd(rb + 7*{RS} + v*16+8);
        TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
        _mm512_store_pd(ZS_{L} + (v*8+0)*16+8, o0); _mm512_store_pd(ZS_{L} + (v*8+1)*16+8, o1);
        _mm512_store_pd(ZS_{L} + (v*8+2)*16+8, o2); _mm512_store_pd(ZS_{L} + (v*8+3)*16+8, o3);
        _mm512_store_pd(ZS_{L} + (v*8+4)*16+8, o4); _mm512_store_pd(ZS_{L} + (v*8+5)*16+8, o5);
        _mm512_store_pd(ZS_{L} + (v*8+6)*16+8, o6); _mm512_store_pd(ZS_{L} + (v*8+7)*16+8, o7);
    }}
}}
static inline void trout_{L}(double* rb){{
    for(int v=0; v<{NV}; v++){{
        __m512d r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7;
        r0=_mm512_load_pd(ZS_{L} + (v*8+0)*16); r1=_mm512_load_pd(ZS_{L} + (v*8+1)*16);
        r2=_mm512_load_pd(ZS_{L} + (v*8+2)*16); r3=_mm512_load_pd(ZS_{L} + (v*8+3)*16);
        r4=_mm512_load_pd(ZS_{L} + (v*8+4)*16); r5=_mm512_load_pd(ZS_{L} + (v*8+5)*16);
        r6=_mm512_load_pd(ZS_{L} + (v*8+6)*16); r7=_mm512_load_pd(ZS_{L} + (v*8+7)*16);
        TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
        _mm512_store_pd(rb + 0*{RS} + v*16, o0); _mm512_store_pd(rb + 1*{RS} + v*16, o1);
        _mm512_store_pd(rb + 2*{RS} + v*16, o2); _mm512_store_pd(rb + 3*{RS} + v*16, o3);
        _mm512_store_pd(rb + 4*{RS} + v*16, o4); _mm512_store_pd(rb + 5*{RS} + v*16, o5);
        _mm512_store_pd(rb + 6*{RS} + v*16, o6); _mm512_store_pd(rb + 7*{RS} + v*16, o7);
        r0=_mm512_load_pd(ZS_{L} + (v*8+0)*16+8); r1=_mm512_load_pd(ZS_{L} + (v*8+1)*16+8);
        r2=_mm512_load_pd(ZS_{L} + (v*8+2)*16+8); r3=_mm512_load_pd(ZS_{L} + (v*8+3)*16+8);
        r4=_mm512_load_pd(ZS_{L} + (v*8+4)*16+8); r5=_mm512_load_pd(ZS_{L} + (v*8+5)*16+8);
        r6=_mm512_load_pd(ZS_{L} + (v*8+6)*16+8); r7=_mm512_load_pd(ZS_{L} + (v*8+7)*16+8);
        TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
        _mm512_store_pd(rb + 0*{RS} + v*16+8, o0); _mm512_store_pd(rb + 1*{RS} + v*16+8, o1);
        _mm512_store_pd(rb + 2*{RS} + v*16+8, o2); _mm512_store_pd(rb + 3*{RS} + v*16+8, o3);
        _mm512_store_pd(rb + 4*{RS} + v*16+8, o4); _mm512_store_pd(rb + 5*{RS} + v*16+8, o5);
        _mm512_store_pd(rb + 6*{RS} + v*16+8, o6); _mm512_store_pd(rb + 7*{RS} + v*16+8, o7);
    }}
}}
static void SB_{L}(double* X, const double* CZT, int mode){{
    for(int i=0;i<{L};i++){{
        double* pl = X + (long)i*{PS};
        _Pragma("GCC unroll 1") for(int v=0;v<{NV};v++) {name}(pl + v*16, {RS}, pl + v*16, {RS});
        for(int yb=0; yb<{NYB}; yb++){{
            double* rb = pl + (long)yb*8*{RS};
            const double* cz = CZT + ((long)i*{NYB} + yb)*{L}*16;
            trin_{L}(rb);
            if(mode == 0){{
                {name}(ZS_{L}, 16, ZS_{L}, 16);
            }} else if(mode == 1){{
                {name}(ZS_{L}, 16, XSB_{L}, 16);
                mp{L}_16(XSB_{L}, cz, ZS_{L});
            }} else {{
                {name}(ZS_{L}, 16, XSB_{L}, 16);
                mp{L}_16(XSB_{L}, cz, MMB_{L});
                {name}(MMB_{L}, 16, ZS_{L}, 16);
            }}
            trout_{L}(rb);
        }}
        if(mode == 2){{
            _Pragma("GCC unroll 1") for(int v=0;v<{NV};v++) {name}(pl + v*16, {RS}, pl + v*16, {RS});
        }}
    }}
}}
static void PB_{L}(double* X, const double* CP, int mode, double* SNAP){{
    _Pragma("GCC unroll 1") for(long e=0;e<{L}*{NV};e++){{
        long y = e / {NV}, v = e % {NV};
        long off = y*{RS} + v*16;
        double* p = X + off;
        const double* pc = CP + e*{L}*16;
        {name}(p, {PS}, XSB_{L}, 16);
        if(mode == 1){{
            mp{L}_ps(XSB_{L}, pc, p);
        }} else if(mode == 2){{
            mp{L}_16(XSB_{L}, pc, MMB_{L});
            {name}(MMB_{L}, 16, p, {PS});
        }} else {{
            mp{L}_16ps(XSB_{L}, pc, MMB_{L}, SNAP + off);
            {name}(MMB_{L}, 16, p, {PS});
        }}
    }}
}}
static void convin_{L}(const double* src, double* X){{
    for(long i=0;i<{L};i++) for(long y=0;y<{L};y++){{
        const double* s = src + 2*((i*{L}+y)*{L});
        double* d = X + i*{PS} + y*{RS};
        long v=0;
        for(; v<{L}/8; v++){{
            __m512d lo=_mm512_loadu_pd(s + v*16), hi=_mm512_loadu_pd(s + v*16 + 8);
            __m512d re, im;
            DEINT(lo, hi, re, im);
            _mm512_store_pd(d + v*16, re); _mm512_store_pd(d + v*16 + 8, im);
        }}
        for(long z=v*8; z<{L}; z++){{ d[(z/8)*16 + (z%8)] = s[2*z]; d[(z/8)*16 + 8 + (z%8)] = s[2*z+1]; }}
    }}
}}
static void convout_{L}(const double* X, double* dst){{
    for(long i=0;i<{L};i++) for(long y=0;y<{L};y++){{
        double* s = dst + 2*((i*{L}+y)*{L});
        const double* d = X + i*{PS} + y*{RS};
        long v=0;
        for(; v<{L}/8; v++){{
            __m512d re=_mm512_load_pd(d + v*16), im=_mm512_load_pd(d + v*16 + 8);
            __m512d lo, hi;
            INTER(re, im, lo, hi);
            _mm512_storeu_pd(s + v*16, lo); _mm512_storeu_pd(s + v*16 + 8, hi);
        }}
        for(long z=v*8; z<{L}; z++){{ s[2*z] = d[(z/8)*16 + (z%8)]; s[2*z+1] = d[(z/8)*16 + 8 + (z%8)]; }}
    }}
}}
void run_{L}(const double* x0, const double* c, double* out1, double* outm, long B, long m){{
    if(m < 1) m = 1;
    if(!XB_{L}){{
        XB_{L}  = alloc_huge_st((long){L}*{PS}*8);
        CGB_{L} = alloc_huge_st((long){L}*{PS}*8);
        CZT_{L} = alloc_huge_st((long){L}*{NYB}*{L}*16*8);
        CPB_{L} = alloc_huge_st((long){L}*{NV}*{L}*16*8);
        SNB_{L} = alloc_huge_st((long){L}*{PS}*8);
    }}
    for(long b=0; b<B; b++){{
        convin_{L}(x0 + b*2*{L3}, XB_{L});
        convin_{L}(c + b*2*{L3}, CGB_{L});
        // CZT: transposed c per (i, yb)
        for(int i=0;i<{L};i++) for(int yb=0; yb<{NYB}; yb++){{
            trin_{L}(CGB_{L} + (long)i*{PS} + (long)yb*8*{RS});
            double* dstc = CZT_{L} + ((long)i*{NYB} + yb)*{L}*16;
            memcpy(dstc, ZS_{L}, {L}*16*8);
        }}
        // CP: pencil-major c
        for(long e=0;e<{L}*{NV};e++){{
            long y = e / {NV}, v = e % {NV};
            for(long q=0;q<{L};q++){{
                _mm512_store_pd(CPB_{L} + e*{L}*16 + q*16,     _mm512_load_pd(CGB_{L} + q*{PS} + y*{RS} + v*16));
                _mm512_store_pd(CPB_{L} + e*{L}*16 + q*16 + 8, _mm512_load_pd(CGB_{L} + q*{PS} + y*{RS} + v*16 + 8));
            }}
        }}
        SB_{L}(XB_{L}, CZT_{L}, 0);
        if(m == 1){{
            PB_{L}(XB_{L}, CPB_{L}, 1, 0);
            convout_{L}(XB_{L}, out1 + b*2*{L3});
            convout_{L}(XB_{L}, outm + b*2*{L3});
            continue;
        }}
        PB_{L}(XB_{L}, CPB_{L}, 3, SNB_{L});
        convout_{L}(SNB_{L}, out1 + b*2*{L3});
        long t = 2;
        while(1){{
            SB_{L}(XB_{L}, CZT_{L}, t==m ? 1 : 2);
            if(t == m) break;
            t++;
            PB_{L}(XB_{L}, CPB_{L}, t==m ? 1 : 2, 0);
            if(t == m) break;
            t++;
        }}
        convout_{L}(XB_{L}, outm + b*2*{L3});
    }}
}}
"""

def build_b(L):
    name = f"dft{L}_v"
    parts = []
    parts.append(gen_twostage(L, name))
    parts.append(gen_map_fn(L, f"mp{L}_16", [("dst", 16)]))
    NV = (L + 7) // 8
    RS = NV * 16
    if L == 64: RS = 144
    YP = ((L + 7) // 8) * 8
    PS = YP * RS
    if (PS * 8) % 4096 == 0: PS += 16
    parts.append(gen_map_fn(L, f"mp{L}_ps", [("dst", PS)]))
    parts.append(gen_map_fn(L, f"mp{L}_16ps", [("dst", 16), ("ps", PS)]))
    parts.append(gen_b_driver(L))
    return "\n".join(parts)
