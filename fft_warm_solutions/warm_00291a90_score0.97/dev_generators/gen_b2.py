"""Family B v2: compact-footprint kernels.
- dft{L}_g(src, dst, tab): looped two-stage DFT, offsets from table (one function per size).
- mp{L}_l*(...): asm looped map over groups of 4 points.
"""
import numpy as np
from genlib import *
from gen_a import dft_graph
from gen_asm import A
import gen_asm_prime as gap
from gen_b import dft9

def fact(L):
    if L == 36: return 4, 9, 'pfa'
    if L == 45: return 5, 9, 'pfa'
    if L == 64: return 8, 8, 'ct'
    raise ValueError

def gen_dft_g(L):
    """one compact two-stage DFT with stride tables.
    table layout (longs): in offsets [N2][N1] (bytes), out offsets [N1][N2] (bytes)."""
    N1, N2, mode = fact(L)
    e1 = E(pfx="a")
    # stage1 body: loads via runtime offsets tin[n1]
    x = []
    for n1 in range(N1):
        e1.raw(f"const double* s{n1} = src + tin[{n1}];")
    for n1 in range(N1):
        x.append((e1.v(f"_mm512_load_pd(s{n1})"), e1.v(f"_mm512_load_pd(s{n1} + 8)")))
    y = dft_graph(e1, x, N1)
    if mode == 'ct':
        # twiddle via table of vector rows: tw base arg twv: row index (k1-1) for this n2
        y2 = [y[0]]
        for k1 in range(1, N1):
            wr = e1.v(f"_mm512_load_pd(twv + {(k1-1)*16})")
            wi = e1.v(f"_mm512_load_pd(twv + {(k1-1)*16+8})")
            xr, xi = y[k1]
            t = mul(e1, xr, wr)
            yr = fnmadd(e1, xi, wi, t)
            u = mul(e1, xi, wr)
            yi = fmadd(e1, xr, wi, u)
            y2.append((yr, yi))
        y = y2
    for k1 in range(N1):
        e1.raw(f"_mm512_store_pd(SCB_{L}[{{}}*{N2}+n2], {y[k1][0]});".format(k1))
        e1.raw(f"_mm512_store_pd(SCB_{L}[{{}}*{N2}+n2]+8, {y[k1][1]});".format(k1))
    e2 = E(pfx="b")
    x2 = [(e2.v(f"_mm512_load_pd(SCB_{L}[k1*{N2}+{n2}])"), e2.v(f"_mm512_load_pd(SCB_{L}[k1*{N2}+{n2}]+8)")) for n2 in range(N2)]
    yy = dft_graph(e2, x2, N2) if N2 != 9 else dft9(e2, x2)
    for k2 in range(N2):
        e2.raw(f"double* d{k2} = dst + tout[{k2}];")
        e2.raw(f"_mm512_store_pd(d{k2}, {yy[k2][0]});")
        e2.raw(f"_mm512_store_pd(d{k2} + 8, {yy[k2][1]});")
    tw_decl = ""
    tw_arg = ""
    if mode == 'ct':
        rows = []
        for n2 in range(N2):
            for k1 in range(1, N1):
                wr, wi = tw(64, k1*n2)
                rows.append("{" + ",".join([hexd(wr)]*8) + "," + ",".join([hexd(wi)]*8) + "}")
        tw_decl = f"static const double TWV_{L}[{N2*(N1-1)}][16] ALIGN64 = {{ {', '.join(rows)} }};\n"
    return f"""
static double SCB_{L}[{L}][16] ALIGN64;
{tw_decl}typedef struct {{ long in[{N2}][{N1}]; long out[{N1}][{N2}]; }} DTAB_{L};
static void __attribute__((noinline)) dft{L}_g(const double* src, double* dst, const DTAB_{L}* T){{
    _Pragma("GCC unroll 1") for(int n2=0; n2<{N2}; n2++){{
        const long* tin = T->in[n2];
        {"const double* twv = &TWV_" + str(L) + "[n2*" + str(N1-1) + "][0];" if mode=='ct' else ""}
{e1.code(indent="        ")}
    }}
    _Pragma("GCC unroll 1") for(int k1=0; k1<{N1}; k1++){{
        const long* tout = T->out[k1];
{e2.code(indent="        ")}
    }}
}}
static DTAB_{L} DT_{L}_rr, DT_{L}_cc, DT_{L}_pc, DT_{L}_cp;
static void dtinit_{L}(long RS, long PS){{
    static const int inm[{N2}][{N1}] = {{ {", ".join("{" + ", ".join(str(((9 if L!=64 else 8)* n1 + (4 if L==36 else 5 if L==45 else 1)*n2) % L if mode=='pfa' else 8*n1+n2) for n1 in range(N1)) + "}" for n2 in range(N2))} }};
    static const int outm[{N1}][{N2}] = {{ {", ".join("{" + ", ".join(str(pfa_out(L, k1, k2)) for k2 in range(N2)) + "}" for k1 in range(N1))} }};
    long strides[4][2] = {{ {{RS, RS}}, {{16, 16}}, {{PS, 16}}, {{16, PS}} }};
    DTAB_{L}* tabs[4] = {{ &DT_{L}_rr, &DT_{L}_cc, &DT_{L}_pc, &DT_{L}_cp }};
    for(int t=0;t<4;t++){{
        for(int n2=0;n2<{N2};n2++) for(int n1=0;n1<{N1};n1++) tabs[t]->in[n2][n1] = (long)inm[n2][n1]*strides[t][0];
        for(int k1=0;k1<{N1};k1++) for(int k2=0;k2<{N2};k2++) tabs[t]->out[k1][k2] = (long)outm[k1][k2]*strides[t][1];
    }}
}}
"""

def pfa_out(L, k1, k2):
    N1, N2, mode = fact(L)
    if mode == 'ct':
        return k2*8 + k1
    inm, outm = pfa_maps(N1, N2)
    return outm[k1][k2]

def gen_map_loop(L, name, dst_strides, with_ps=False):
    """asm looped map: groups of 4, pointer-advancing. dst_strides: [(argname, stride_doubles)]"""
    from gen_asm import A
    import gen_asm_prime as gap
    a = A()
    mc = {n: a.bcast('tab', {'TINY':0,'ONE':8,'HALF':16}[n]) for n in ('TINY','ONE','HALF')}
    ng, tail = divmod(L, 4)
    # loop counter in a GPR via operand; body advances pointers
    a.ins(f"mov ${ng}, %[cnt]")
    a.ins("1:")
    gap.emit_map_points(a, mc, 4, ('xs', 16), dst_strides, gmax=4, delta=12)
    a.ins("add $512, %[xs]")
    a.ins("add $512, %[pc]")
    for (nm, st) in dst_strides:
        a.ins(f"add ${4*st*8}, %[{nm}]")
    a.ins("dec %[cnt]")
    a.ins("jnz 1b")
    if tail:
        gap.emit_map_points(a, mc, tail, ('xs', 16), dst_strides, gmax=4, delta=12)
    for r in mc.values(): a.rel(r)
    assert not a.live
    body = "\\n\\t".join(a.lines)
    args = ["const double* xs", "const double* pc"] + [f"double* {nm}" for (nm, st) in dst_strides]
    ops = [f'[xs]"+r"(xs)', f'[pc]"+r"(pc)', f'[cnt]"=&r"(cnt)'] + [f'[{nm}]"+r"({nm})' for (nm, st) in dst_strides]
    clob = ", ".join(f'"zmm{i}"' for i in range(32)) + ', "memory", "cc"'
    return f"""static void __attribute__((noinline)) {name}({", ".join(args)}){{
    long cnt;
    __asm__ volatile("{body}"
    : {", ".join(ops)}
    : [tab]"r"(MTB)
    : {clob});
}}
"""

def gen_dft_gmi(L, snap=False):
    """map-on-input + two-stage DFT. src slots via T->in (runtime table); c permuted: cz + (n2*N1+n1)*128.
    snap variant stores mapped values to ps at T2->in offsets."""
    from gen_a import emit_map
    N1, N2, mode = fact(L)
    e1 = E(pfx="a")
    for n1 in range(N1):
        e1.raw(f"const double* s{n1} = src + tin[{n1}];")
    if snap:
        for n1 in range(N1):
            e1.raw(f"double* q{n1} = ps + tsn[{n1}];")
    x = []
    for n1 in range(N1):
        xr = e1.v(f"_mm512_load_pd(s{n1})"); xi = e1.v(f"_mm512_load_pd(s{n1} + 8)")
        cr = e1.v(f"_mm512_load_pd(cz + {n1}*16)"); ci = e1.v(f"_mm512_load_pd(cz + {n1}*16 + 8)")
        zr = add(e1, xr, cr); zi = add(e1, xi, ci)
        mr, mi = emit_map(e1, zr, zi)
        if snap:
            e1.raw(f"_mm512_store_pd(q{n1}, {mr});")
            e1.raw(f"_mm512_store_pd(q{n1} + 8, {mi});")
        x.append((mr, mi))
    y = dft_graph(e1, x, N1)
    if mode == 'ct':
        y2 = [y[0]]
        for k1 in range(1, N1):
            wr = e1.v(f"_mm512_load_pd(twv + {(k1-1)*16})")
            wi = e1.v(f"_mm512_load_pd(twv + {(k1-1)*16+8})")
            xr, xi = y[k1]
            t = mul(e1, xr, wr)
            yr = fnmadd(e1, xi, wi, t)
            u = mul(e1, xi, wr)
            yi = fmadd(e1, xr, wi, u)
            y2.append((yr, yi))
        y = y2
    for k1 in range(N1):
        e1.raw(f"_mm512_store_pd(SCB_{L}[{{}}*{N2}+n2], {y[k1][0]});".format(k1))
        e1.raw(f"_mm512_store_pd(SCB_{L}[{{}}*{N2}+n2]+8, {y[k1][1]});".format(k1))
    e2 = E(pfx="b")
    x2 = [(e2.v(f"_mm512_load_pd(SCB_{L}[k1*{N2}+{n2}])"), e2.v(f"_mm512_load_pd(SCB_{L}[k1*{N2}+{n2}]+8)")) for n2 in range(N2)]
    yy = dft_graph(e2, x2, N2) if N2 != 9 else dft9(e2, x2)
    for k2 in range(N2):
        e2.raw(f"double* d{k2} = dst + tout[{k2}];")
        e2.raw(f"_mm512_store_pd(d{k2}, {yy[k2][0]});")
        e2.raw(f"_mm512_store_pd(d{k2} + 8, {yy[k2][1]});")
    nm = f"dft{L}_gmi" + ("s" if snap else "")
    sig = f"const double* src, const double* cz, double* dst, const DTAB_{L}* T" + (f", double* ps, const DTAB_{L}* T2" if snap else "")
    return f"""
static void __attribute__((noinline)) {nm}({sig}){{
    _Pragma("GCC unroll 1") for(int n2=0; n2<{N2}; n2++){{
        const long* tin = T->in[n2];
        {"const long* tsn = T2->in[n2];" if snap else ""}
        {"const double* twv = &TWV_" + str(L) + "[n2*" + str(N1-1) + "][0];" if mode=='ct' else ""}
{e1.code(indent="        ")}
        cz += {N1}*16;
    }}
    _Pragma("GCC unroll 1") for(int k1=0; k1<{N1}; k1++){{
        const long* tout = T->out[k1];
{e2.code(indent="        ")}
    }}
}}
"""
