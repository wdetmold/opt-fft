# Generator v2: loop-structured kernels with constant tables for large sizes.
import math
from kernels import dft, tw, cospi2, sinpi2
from gen import CBE, hexf, PREAMBLE, PS   # reuse emitter pieces

SIZES = (6, 8, 13, 17, 23, 36, 45, 64)
PRIME_KB = {13: 6, 17: 4, 23: 6}
DUAL_KB = {13: 3, 17: 4, 23: 3}

def ssa_kernel(L, ld, st):
    """straight-line kernel (small L)"""
    be = CBE()
    cache = {}
    def ldc(j):
        if j not in cache:
            e_r, e_i = ld(j)
            cache[j] = (be.nv(e_r), be.nv(e_i))
        return cache[j]
    xs = [ldc(j) for j in range(L)]
    out = dft(be, xs, L)
    lines = list(be.lines)
    post = []
    for k in range(L):
        post += st(k, out[k])
    return lines + post

# ---------- loop-structured prime kernel ----------
def prime_tables(p):
    h = (p - 1) // 2
    # pad k-rows up to the largest padded block count used by any kernel variant
    kbs = [PRIME_KB[p], DUAL_KB[p]]
    rows = max(((h + kb - 1) // kb) * kb for kb in kbs)
    ct, stb = [], []
    for k in range(1, rows + 1):
        for j in range(1, h + 1):
            if k <= h:
                ct.append(cospi2(j * k, p))
                stb.append(sinpi2(j * k, p))
            else:
                ct.append(0.0)
                stb.append(0.0)
    return ct, stb, h, rows

def gen_prime_tabs(p):
    ct, stb, h, rows = prime_tables(p)
    ln = []
    ln.append(f"static const double PCT{p}[{rows*h}] __attribute__((aligned(64))) = {{{','.join(hexf(v) for v in ct)}}};")
    ln.append(f"static const double PST{p}[{rows*h}] __attribute__((aligned(64))) = {{{','.join(hexf(v) for v in stb)}}};")
    return ln


PRIME_SL = {13}   # sizes using straight-line prime kernel

def prime_kernel_sl(p, ld, st, uid):
    """Straight-line even/odd prime DFT; constants via laundered table pointer
       (prevents GCC from hoisting/CSEing broadcasts; enables embedded bcast)."""
    h = (p - 1) // 2
    u = uid
    ln = []
    ln.append(f"const double* ct{u} = PCT{p}; const double* stt{u} = PST{p};")
    ln.append(f'__asm__("" : "+r"(ct{u}), "+r"(stt{u}));')
    x0r, x0i = ld(0, 0)
    ln.append(f"V x0r{u} = {x0r}, x0i{u} = {x0i};")
    ln.append(f"V acr{u} = x0r{u}, aci{u} = x0i{u};")
    names = {}
    for j in range(1, h + 1):
        ar, ai = ld(j, 0)
        br, bi = ld(p - j, 0)
        ln.append(f"V sjr{u}{j}, sji{u}{j}, djr{u}{j}, dji{u}{j};")
        ln.append(f"{{ V ar={ar}, ai={ai}, br={br}, bi={bi};")
        ln.append(f"  sjr{u}{j}=ADD(ar,br); sji{u}{j}=ADD(ai,bi); djr{u}{j}=SUB(ar,br); dji{u}{j}=SUB(ai,bi);")
        ln.append(f"  acr{u}=ADD(acr{u},sjr{u}{j}); aci{u}=ADD(aci{u},sji{u}{j}); }}")
    ln += st("0", 0, (f"acr{u}", f"aci{u}"))
    for k in range(1, h + 1):
        ln.append("{")
        for nm, tab, src_r, src_i in (("A", f"ct{u}", "sjr", "sji"), ("B", f"stt{u}", "djr", "dji")):
            base = (k - 1) * h
            ln.append(f" V {nm}r = MUL(SET1({tab}[{base}]), {src_r}{u}1);")
            ln.append(f" V {nm}i = MUL(SET1({tab}[{base}]), {src_i}{u}1);")
            for j in range(2, h + 1):
                ln.append(f" {{ V c = SET1({tab}[{base + j - 1}]);")
                ln.append(f"   {nm}r = FMA(c, {src_r}{u}{j}, {nm}r); {nm}i = FMA(c, {src_i}{u}{j}, {nm}i); }}")
        ln.append(f" V Tr=ADD(x0r{u},Ar), Ti=ADD(x0i{u},Ai);")
        ln.append(f" V Xr=ADD(Tr,Bi), Xi=SUB(Ti,Br);")
        ln.append(f" V Yr=SUB(Tr,Bi), Yi=ADD(Ti,Br);")
        ln += [" " + l for l in st(str(k), 0, ("Xr", "Xi"))]
        ln += [" " + l for l in st(str(p - k), 0, ("Yr", "Yi"))]
        ln.append("}")
    return ln

def prime_kernel(p, ld, st, uid, NV=1, KB=None):
    """Emit loop-blocked prime DFT over NV lane-vectors.
       ld(j, v)->(re,im) exprs; st(k_expr, v, (re,im)) -> lines."""
    h = (p - 1) // 2
    if KB is None:
        KB = PRIME_KB[p] if NV == 1 else DUAL_KB[p]
    NB = (h + KB - 1) // KB
    u = uid
    ln = []
    ln.append(f"V S{u}r[{(h+1)*NV}], S{u}i[{(h+1)*NV}], D{u}r[{(h+1)*NV}], D{u}i[{(h+1)*NV}];")
    for v in range(NV):
        x0r, x0i = ld(0, v)
        ln.append(f"V x0r{u}_{v} = {x0r}, x0i{u}_{v} = {x0i};")
        ln.append(f"V acr{u}_{v} = x0r{u}_{v}, aci{u}_{v} = x0i{u}_{v};")
    for j in range(1, h + 1):
        for v in range(NV):
            ar, ai = ld(j, v)
            br, bi = ld(p - j, v)
            ln.append(f"{{ V ar={ar}, ai={ai}, br={br}, bi={bi};")
            ln.append(f"  S{u}r[{j*NV+v}]=ADD(ar,br); S{u}i[{j*NV+v}]=ADD(ai,bi); D{u}r[{j*NV+v}]=SUB(ar,br); D{u}i[{j*NV+v}]=SUB(ai,bi);")
            ln.append(f"  acr{u}_{v}=ADD(acr{u}_{v},S{u}r[{j*NV+v}]); aci{u}_{v}=ADD(aci{u}_{v},S{u}i[{j*NV+v}]); }}")
    for v in range(NV):
        ln += st("0", v, (f"acr{u}_{v}", f"aci{u}_{v}"))
    ln.append('#pragma GCC unroll 1')
    ln.append(f"for(long kb=0; kb<{NB}; ++kb){{")
    ln.append(f" const double* ct = PCT{p} + kb*{KB*h};")
    ln.append(f" const double* stt = PST{p} + kb*{KB*h};")
    for t in range(KB):
        for v in range(NV):
            ln.append(f" V A{t}r{v}=_mm512_setzero_pd(), A{t}i{v}=_mm512_setzero_pd(), B{t}r{v}=_mm512_setzero_pd(), B{t}i{v}=_mm512_setzero_pd();")
    ln.append('#pragma GCC unroll 1')
    ln.append(f" for(long j=1; j<={h}; ++j){{")
    for v in range(NV):
        ln.append(f"  V sr{v}=S{u}r[j*{NV}+{v}], si{v}=S{u}i[j*{NV}+{v}], dr{v}=D{u}r[j*{NV}+{v}], di{v}=D{u}i[j*{NV}+{v}];")
    for t in range(KB):
        ln.append(f"  {{ V c=SET1(ct[{t*h}+j-1]);")
        for v in range(NV):
            ln.append(f"   A{t}r{v}=FMA(c, sr{v}, A{t}r{v}); A{t}i{v}=FMA(c, si{v}, A{t}i{v});")
        ln.append(f"   c=SET1(stt[{t*h}+j-1]);")
        for v in range(NV):
            ln.append(f"   B{t}r{v}=FMA(c, dr{v}, B{t}r{v}); B{t}i{v}=FMA(c, di{v}, B{t}i{v});")
        ln.append("  }")
    ln.append(" }")
    for t in range(KB):
        ln.append(f" {{ long k = kb*{KB} + {t+1};")
        if NB * KB > h:
            ln.append(f"  if(k <= {h}) {{")
        for v in range(NV):
            ln.append(f"  {{ V Tr=ADD(x0r{u}_{v},A{t}r{v}), Ti=ADD(x0i{u}_{v},A{t}i{v});")
            ln.append(f"  V Xr=ADD(Tr,B{t}i{v}), Xi=SUB(Ti,B{t}r{v});")
            ln.append(f"  V Yr=SUB(Tr,B{t}i{v}), Yi=ADD(Ti,B{t}r{v});")
            ln += ["  " + l for l in st("k", v, ("Xr", "Xi"))]
            ln += ["  " + l for l in st(f"({p}-k)", v, ("Yr", "Yi"))]
            ln.append("  }")
        if NB * KB > h:
            ln.append("  }")
        ln.append(" }")
    ln.append("}")
    return ln

# ---------- loop-structured composite kernels ----------
def crt_maps(N1, N2):
    N = N1 * N2
    inmap = [[(N2 * j1 + N1 * j2) % N for j1 in range(N1)] for j2 in range(N2)]
    outmap = [[None] * N2 for _ in range(N1)]
    for k in range(N):
        outmap[k % N1][k % N2] = k
    return inmap, outmap

def sub_ssa(Lsub, ld, st):
    return ssa_kernel(Lsub, ld, st)

def composite_kernel(L, ld, st, uid, tabsfx, tables):
    """L in (36,45,64). ld(expr)->(re,im) strings with runtime-index expr.
       st(k_expr,(re,im), is_x0) -> lines. tables: list to append static tables."""
    u = uid
    ln = []
    if L == 64:
        N1 = N2 = 8
        # twiddle tables indexed [g][k1], g=1..7 rows (row g at (g-1)*8)
        twr, twi = [], []
        for g in range(1, 8):
            for k1 in range(8):
                wr, wi = tw(g * k1, 64)
                twr.append(wr); twi.append(wi)
        t1 = f"static const double TW64R[{56}] __attribute__((aligned(64))) = {{{','.join(hexf(v) for v in twr)}}};"
        t2 = f"static const double TW64I[{56}] __attribute__((aligned(64))) = {{{','.join(hexf(v) for v in twi)}}};"
        if t1 not in tables:
            tables.append(t1); tables.append(t2)
        ln.append(f"V BUF{u}r[64], BUF{u}i[64];")
        # stage1 g=0 peel (no twiddle)
        def ld1(j1, g):  # input index N2*j1 + g
            return ld(f"({8 * j1}+{g})" if isinstance(g, int) else f"({8*j1}+{g})")
        def mkst_buf(g):
            def stb(k1, v):
                return [f"BUF{u}r[{k1}*8+{g}]={v[0]}; BUF{u}i[{k1}*8+{g}]={v[1]};"]
            return stb
        ln += sub_ssa(8, lambda j1: ld(8 * j1), mkst_buf(0))
        ln.append('#pragma GCC unroll 1')
        ln.append("for(long g=1; g<8; ++g){")
        ln.append(f" const double* twr = TW64R + (g-1)*8;")
        ln.append(f" const double* twi = TW64I + (g-1)*8;")
        stage1 = []
        def stb_tw(k1, v):
            if k1 == 0:
                return [f"BUF{u}r[0*8+g]={v[0]}; BUF{u}i[0*8+g]={v[1]};"]
            return [f"{{ V wr=SET1(twr[{k1}]), wi=SET1(twi[{k1}]);",
                    f"  BUF{u}r[{k1}*8+g]=FNMA(wi,{v[1]},MUL(wr,{v[0]}));",
                    f"  BUF{u}i[{k1}*8+g]=FMA(wr,{v[1]},MUL(wi,{v[0]})); }}"]
        stage1 += sub_ssa(8, lambda j1: ld(f"({8*j1}+g)"), stb_tw)
        ln += [" " + l for l in stage1]
        ln.append("}")
        # stage2: q = k1 runtime loop; outputs at k1 + 8*k2
        ln.append('#pragma GCC unroll 1')
        ln.append("for(long q=0; q<8; ++q){")
        stage2 = []
        def st2(k2, v):
            return st(f"(q+{8 * k2})", v, False)
        stage2 += sub_ssa(8, lambda j2: (f"BUF{u}r[q*8+{j2}]", f"BUF{u}i[q*8+{j2}]"), st2)
        ln += [" " + l for l in stage2]
        ln.append("}")
    else:
        N1, N2 = (4, 9) if L == 36 else (9, 5)
        inmap, outmap = crt_maps(N1, N2)
        # tables of element indices
        tables.append(f"static const int IN{L}_{tabsfx}[{N2}][{N1}] = {{{','.join('{'+','.join(str(inmap[g][j]) for j in range(N1))+'}' for g in range(N2))}}};")
        tables.append(f"static const int OUT{L}_{tabsfx}[{N1}][{N2}] = {{{','.join('{'+','.join(str(outmap[q][k]) for k in range(N2))+'}' for q in range(N1))}}};")
        ln.append(f"V BUF{u}r[{L}], BUF{u}i[{L}];")
        ln.append('#pragma GCC unroll 1')
        ln.append(f"for(long g=0; g<{N2}; ++g){{")
        ln.append(f" const int* inx = IN{L}_{tabsfx}[g];")
        stage1 = []
        def stb(k1, v):
            return [f"BUF{u}r[{k1}*{N2}+g]={v[0]}; BUF{u}i[{k1}*{N2}+g]={v[1]};"]
        stage1 += sub_ssa(N1, lambda j1: ld(f"inx[{j1}]"), stb)
        ln += [" " + l for l in stage1]
        ln.append("}")
        ln.append('#pragma GCC unroll 1')
        ln.append(f"for(long q=0; q<{N1}; ++q){{")
        ln.append(f" const int* outx = OUT{L}_{tabsfx}[q];")
        stage2 = []
        def st2(k2, v):
            return st(f"outx[{k2}]", v, False)
        stage2 += sub_ssa(N2, lambda j2: (f"BUF{u}r[q*{N2}+{j2}]", f"BUF{u}i[q*{N2}+{j2}]"), st2)
        ln += [" " + l for l in stage2]
        ln.append("}")
    return ln

def emit_kernel2(L, ld, st, uid, tabsfx, tables):
    """Dispatch: ld(j int-or-expr)->(re,im) exprs; st(k int-or-expr, (re,im), is_scalar_k) -> list of lines."""
    if L in (13, 17, 23):
        def ld1(j, v): return ld(j)
        def st1(k, v, val): return st(k, val, False)
        if L in PRIME_SL:
            return prime_kernel_sl(L, ld1, st1, uid)
        return prime_kernel(L, ld1, st1, uid, NV=1)
    if L in (36, 45, 64):
        def st2(k, v, flag=False):
            return st(k, v, flag)
        return composite_kernel(L, ld, lambda k, v, f=False: st(k, v, f), uid, tabsfx, tables)
    # small: SSA with constant indices
    return ssa_kernel(L, ld, lambda k, v: st(k, v, False))
