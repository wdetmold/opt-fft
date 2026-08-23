# Generator v3: passes + drivers using loop-structured kernels (gen2).
from gen import PREAMBLE, PS, hexf
from gen2 import emit_kernel2, gen_prime_tabs, SIZES, PRIME_KB, prime_kernel, ssa_kernel

PZ_MODE = {6:'table', 8:'contig', 13:'table', 17:'table', 23:'table', 36:'contig', 45:'table', 64:'plane64'}

def gen_pz(L, tables):
    from gen import PS as _PS
    ps = _PS(L)
    nblk = (L + 7) // 8
    mode = PZ_MODE[L]
    VOL = L * ps
    LD_DUMP = VOL + 16
    ST_DUMP = VOL + 80
    groups = []   # list of (list of (loadoff_expr, storeoff_expr)) per group; address base expr
    ln = [f"static void pz_{L}(double* restrict re, double* restrict im){{"]
    if mode == 'plane64':
        NG = L // 8
        ln += [f" for(long g=0; g<{NG}; ++g){{",
               f"  double* restrict rb = re + g*{8*L};",
               f"  double* restrict ib = im + g*{8*L};"]
        base_r, base_i = "rb", "ib"
        lo = [f"{t*L}" for t in range(8)]
        so = lo
    elif mode == 'contig':
        assert ps == L * L
        NG = (L * L + 7) // 8
        assert NG * 8 == L * L or L*L % 8 != 0
        # rows contiguous across planes; need dump only if L*L % 8
        ln += [f" for(long g=0; g<{NG}; ++g){{",
               f"  double* restrict rb = re + g*{8*L};",
               f"  double* restrict ib = im + g*{8*L};"]
        base_r, base_i = "rb", "ib"
        lo = [f"{t*L}" for t in range(8)]
        so = lo
    elif mode == 'table':
        NG = (L * L + 7) // 8
        loffs, soffs = [], []
        for r in range(NG * 8):
            if r < L * L:
                loffs.append((r // L) * ps + (r % L) * L); soffs.append(loffs[-1])
            else:
                loffs.append(LD_DUMP); soffs.append(ST_DUMP)
        tables.append(f"static const int ROL{L}[{NG*8}] = {{{','.join(map(str,loffs))}}};")
        tables.append(f"static const int ROS{L}[{NG*8}] = {{{','.join(map(str,soffs))}}};")
        ln += [f" for(long g=0; g<{NG}; ++g){{",
               f"  const int* rol = ROL{L} + g*8;",
               f"  const int* ros = ROS{L} + g*8;"]
        base_r, base_i = "re", "im"
        lo = [f"rol[{t}]" for t in range(8)]
        so = [f"ros[{t}]" for t in range(8)]
    elif mode == 'dump':
        # per plane, groups of 8 rows with constant offsets; dump rows at fixed pad
        NGP = (L + 7) // 8
        ln += [f" for(long x=0; x<{L}; ++x){{",
               f"  double* restrict rp = re + x*{ps};",
               f"  double* restrict ip = im + x*{ps};",
               f"  for(long gg=0; gg<{NGP}; ++gg){{"]
        # we will emit one body with a switch... simpler: unroll gg by emitting NGP bodies
        ln = ln[:-1]  # drop the gg loop; unroll below
        base_r, base_i = "rp", "ip"
    # body emission helper
    def emit_group(lo, so, base_r, base_i, indent="  "):
        out = []
        out.append(f"{indent}V sr[{L}], si[{L}], qr[{L}], qi[{L}];")
        for comp, base, arr in (("r", base_r, "sr"), ("i", base_i, "si")):
            for blk in range(nblk):
                c0 = blk * 8
                cnt = min(8, L - c0)
                vs = [f"p{comp}{blk}_{t}" for t in range(8)]
                for t in range(8):
                    out.append(f"{indent}V {vs[t]} = LDU({base} + {lo[t]} + {c0});")
                out.append(f"{indent}TR8({','.join(vs)});")
                for t in range(cnt):
                    out.append(f"{indent}{arr}[{c0+t}] = {vs[t]};")
        def ld(j): return (f"sr[{j}]", f"si[{j}]")
        def st(k, v, flag): return [f"qr[{k}]={v[0]}; qi[{k}]={v[1]};"]
        out += [indent + l for l in emit_kernel2(L, ld, st, "z", f"pz{L}", tables)]
        for comp, base, arr in (("r", base_r, "qr"), ("i", base_i, "qi")):
            for blk in range(nblk):
                c0 = blk * 8
                cnt = min(8, L - c0)
                vs = [f"o{comp}{blk}_{t}" for t in range(8)]
                for t in range(8):
                    srcv = f"{arr}[{c0+t}]" if t < cnt else f"{arr}[{c0}]"
                    out.append(f"{indent}V {vs[t]} = {srcv};")
                out.append(f"{indent}TR8({','.join(vs)});")
                if cnt == 8:
                    for t in range(8):
                        out.append(f"{indent}STU({base} + {so[t]} + {c0}, {vs[t]});")
                else:
                    mk = (1 << cnt) - 1
                    for t in range(8):
                        out.append(f"{indent}MST({base} + {so[t]} + {c0}, {hex(mk)}, {vs[t]});")
        return out
    if mode == 'dump':
        NGP = (L + 7) // 8
        for gg in range(NGP):
            lo, so = [], []
            for t in range(8):
                r = gg * 8 + t
                if r < L:
                    lo.append(f"{r*L}"); so.append(f"{r*L}")
                else:
                    lo.append(f"(re - rp) + {LD_DUMP}"); so.append(f"(re - rp) + {ST_DUMP}")
            # careful: for im component base is ip/im
            lo_i = [x.replace("re - rp", "im - ip") for x in lo]
            so_i = [x.replace("re - rp", "im - ip") for x in so]
            # emit with per-component offsets: hack emit_group to take pairs
            out = ["  {"]
            out.append(f"  V sr[{L}], si[{L}], qr[{L}], qi[{L}];")
            for comp, base, arr, LO in (("r", "rp", "sr", lo), ("i", "ip", "si", lo_i)):
                for blk in range(nblk):
                    c0 = blk * 8
                    cnt = min(8, L - c0)
                    vs = [f"p{comp}{gg}_{blk}_{t}" for t in range(8)]
                    for t in range(8):
                        out.append(f"  V {vs[t]} = LDU({base} + {LO[t]} + {c0});")
                    out.append(f"  TR8({','.join(vs)});")
                    for t in range(cnt):
                        out.append(f"  {arr}[{c0+t}] = {vs[t]};")
            def ld(j): return (f"sr[{j}]", f"si[{j}]")
            def st(k, v, flag): return [f"qr[{k}]={v[0]}; qi[{k}]={v[1]};"]
            out += ["  " + l for l in emit_kernel2(L, ld, st, f"z{gg}", f"pz{L}g{gg}", tables)]
            for comp, base, arr, SO in (("r", "rp", "qr", so), ("i", "ip", "qi", so_i)):
                for blk in range(nblk):
                    c0 = blk * 8
                    cnt = min(8, L - c0)
                    vs = [f"o{comp}{gg}_{blk}_{t}" for t in range(8)]
                    for t in range(8):
                        srcv = f"{arr}[{c0+t}]" if t < cnt else f"{arr}[{c0}]"
                        out.append(f"  V {vs[t]} = {srcv};")
                    out.append(f"  TR8({','.join(vs)});")
                    if cnt == 8:
                        for t in range(8):
                            out.append(f"  STU({base} + {SO[t]} + {c0}, {vs[t]});")
                    else:
                        mk = (1 << cnt) - 1
                        for t in range(8):
                            out.append(f"  MST({base} + {SO[t]} + {c0}, {hex(mk)}, {vs[t]});")
            out.append("  }")
            ln += out
        ln += [" }", "}"]
    else:
        ln += emit_group(lo, so, base_r, base_i)
        ln += [" }", "}"]
    return ln

def gen_py(L, tables):
    L2 = L * L
    perplane = (L == 64)
    nblk = (L + 7) // 8
    rem = L % 8
    masks = [0xFF] * nblk
    if rem: masks[-1] = (1 << rem) - 1
    yms = f"YM_{L}"
    ln = [f"static const uint8_t {yms}[{nblk}] = {{{','.join(hex(m) for m in masks)}}};"]
    nm = f"py_{L}" if perplane else f"pyp_{L}"
    ln += [f"static void {nm}(double* restrict rp, double* restrict ip){{"]
    ln += [f"  for(long zb=0; zb<{nblk}; ++zb){{",
           f"   const __mmask8 mk = (__mmask8){yms}[zb];",
           f"   double* restrict r0 = rp + zb*8;",
           f"   double* restrict i0 = ip + zb*8;"]
    def ld(j): return (f"LDU(r0 + ({j})*{L})", f"LDU(i0 + ({j})*{L})")
    def st(k, v, flag=False):
        return [f"MST(r0 + ({k})*{L}, mk, {v[0]}); MST(i0 + ({k})*{L}, mk, {v[1]});"]
    ln += ["   " + l for l in emit_kernel2(L, ld, st, "y", f"py{L}", tables)]
    ln += ["  }", "}"]
    if not perplane:
        ln += [f"static void py_{L}(double* restrict re, double* restrict im){{",
               f" for(long x=0; x<{L}; ++x) pyp_{L}(re + x*{PS(L)}, im + x*{PS(L)});",
               f"}}"]
    return ln

def gen_px_split(L, tables):
    L2, L3 = L * L, L ** 3
    S = PS(L)
    nub = (L2 + 7) // 8
    rem = L2 % 8
    ln = [f"static void pxs_{L}(double* restrict re, double* restrict im, const double* restrict cre, const double* restrict cim, long ub0, long ub1){{",
          f" for(long ub=ub0; ub<ub1; ++ub){{"]
    if rem:
        ln.append(f"  const __mmask8 mk = (ub=={nub-1}) ? (__mmask8){hex((1<<rem)-1)} : (__mmask8)0xFF;")
    else:
        ln.append(f"  const __mmask8 mk = (__mmask8)0xFF;")
    ln += [f"  double* restrict r0 = re + ub*8;",
           f"  double* restrict i0 = im + ub*8;",
           f"  const double* restrict c0r = cre + ub*8;",
           f"  const double* restrict c0i = cim + ub*8;"]
    def ld(j): return (f"LDU(r0 + ({j})*{S})", f"LDU(i0 + ({j})*{S})")
    def st(k, v, flag=False):
        return [f"{{ V mr, mi; map8({v[0]}, {v[1]}, LDU(c0r + ({k})*{S}), LDU(c0i + ({k})*{S}), &mr, &mi);",
                f"  MST(r0 + ({k})*{S}, mk, mr); MST(i0 + ({k})*{S}, mk, mi); }}"]
    ln += ["  " + l for l in emit_kernel2(L, ld, st, "xs", f"pxs{L}", tables)]
    ln += [" }", "}"]
    return ln

def gen_pxd(L, tables):
    L2, L3 = L * L, L ** 3
    S = PS(L)
    nub = (L2 + 7) // 8
    rem = L2 % 8
    ln = [f"static void pxd_{L}(double* restrict re, double* restrict im, const double* restrict cI, double* restrict dst, long ub0, long ub1){{",
          f" for(long ub=ub0; ub<ub1; ++ub){{"]
    if rem:
        rem2 = 2 * rem
        m1 = (1 << min(rem2, 8)) - 1
        m2 = (1 << max(rem2 - 8, 0)) - 1
        ln.append(f"  const int tail = (ub=={nub-1});")
        ln.append(f"  const __mmask8 mk  = tail ? (__mmask8){hex((1<<rem)-1)} : (__mmask8)0xFF;")
        ln.append(f"  const __mmask8 mlo = tail ? (__mmask8){hex(m1)} : (__mmask8)0xFF;")
        ln.append(f"  const __mmask8 mhi = tail ? (__mmask8){hex(m2)} : (__mmask8)0xFF;")
    else:
        ln.append(f"  const __mmask8 mk = 0xFF; const __mmask8 mlo = 0xFF, mhi = 0xFF;")
    ln += [f"  double* restrict r0 = re + ub*8;",
           f"  double* restrict i0 = im + ub*8;",
           f"  const double* restrict c0 = cI + ub*16;",
           f"  double* restrict d0 = dst + ub*16;"]
    def ld(j): return (f"LDU(r0 + ({j})*{S})", f"LDU(i0 + ({j})*{S})")
    def st(k, v, flag=False):
        return [f"{{ V ca = _mm512_maskz_loadu_pd(mlo, c0 + ({k})*{2*L2});",
                f"  V cb = _mm512_maskz_loadu_pd(mhi, c0 + ({k})*{2*L2} + 8);",
                f"  V cr = _mm512_permutex2var_pd(ca, XRE, cb);",
                f"  V ci = _mm512_permutex2var_pd(ca, XIM, cb);",
                f"  V mr, mi; map8({v[0]}, {v[1]}, cr, ci, &mr, &mi);",
                f"  MST(r0 + ({k})*{S}, mk, mr); MST(i0 + ({k})*{S}, mk, mi);",
                f"  MST(d0 + ({k})*{2*L2}, mlo, _mm512_permutex2var_pd(mr, XLO, mi));",
                f"  MST(d0 + ({k})*{2*L2} + 8, mhi, _mm512_permutex2var_pd(mr, XHI, mi)); }}"]
    ln += ["  " + l for l in emit_kernel2(L, ld, st, "xd", f"pxd{L}", tables)]
    ln += [" }", "}"]
    return ln

def gen_px(L, tables):
    L2, L3 = L * L, L ** 3
    S = PS(L)
    nub = (L2 + 7) // 8
    rem = L2 % 8
    ln = [f"static void px_{L}(double* restrict re, double* restrict im, const double* restrict cI, long ub0, long ub1){{",
          f" for(long ub=ub0; ub<ub1; ++ub){{"]
    if rem:
        rem2 = 2 * rem
        m1 = (1 << min(rem2, 8)) - 1
        m2 = (1 << max(rem2 - 8, 0)) - 1
        ln.append(f"  const int tail = (ub=={nub-1});")
        ln.append(f"  const __mmask8 mk  = tail ? (__mmask8){hex((1<<rem)-1)} : (__mmask8)0xFF;")
        ln.append(f"  const __mmask8 mlo = tail ? (__mmask8){hex(m1)} : (__mmask8)0xFF;")
        ln.append(f"  const __mmask8 mhi = tail ? (__mmask8){hex(m2)} : (__mmask8)0xFF;")
    else:
        ln.append(f"  const __mmask8 mk = 0xFF; const __mmask8 mlo = 0xFF, mhi = 0xFF;")
    ln += [f"  double* restrict r0 = re + ub*8;",
           f"  double* restrict i0 = im + ub*8;",
           f"  const double* restrict c0 = cI + ub*16;"]
    def ld(j): return (f"LDU(r0 + ({j})*{S})", f"LDU(i0 + ({j})*{S})")
    def st(k, v, flag=False):
        return [f"{{ V ca = _mm512_maskz_loadu_pd(mlo, c0 + ({k})*{2*L2});",
                f"  V cb = _mm512_maskz_loadu_pd(mhi, c0 + ({k})*{2*L2} + 8);",
                f"  V cr = _mm512_permutex2var_pd(ca, XRE, cb);",
                f"  V ci = _mm512_permutex2var_pd(ca, XIM, cb);",
                f"  V mr, mi; map8({v[0]}, {v[1]}, cr, ci, &mr, &mi);",
                f"  MST(r0 + ({k})*{S}, mk, mr); MST(i0 + ({k})*{S}, mk, mi); }}"]
    ln += ["  " + l for l in emit_kernel2(L, ld, st, "x", f"px{L}", tables)]
    ln += [" }", "}"]
    return ln


TILE_TS = {36: 32, 45: 16, 64: 16}
TSMAX = 32
TILED_PX = (36, 45, 64)

def gen_px_tiled(L, tables):
    """Two-stage tiled px for composite L, emitted in four flavors:
       pxt  (split-c, in-place), pxti (interleaved-c, in-place),
       pxdt (interleaved-c, in-place + interleaved dst), pxft (interleaved-c, dst only)."""
    from gen2 import crt_maps
    L2, L3 = L * L, L ** 3
    S = PS(L)
    nub = (L2 + 7) // 8
    TS = TILE_TS.get(L, 8)
    ntile = L2 // (8 * TS)
    ln = []
    ln.append(f"static double TBR{L}[{L*8*TSMAX}] __attribute__((aligned(64)));")
    ln.append(f"static double TBI{L}[{L*8*TSMAX}] __attribute__((aligned(64)));")
    if L == 64:
        N1 = N2 = 8
    else:
        N1, N2 = (4, 9) if L == 36 else (9, 5)
        inmap, outmap = crt_maps(N1, N2)
        tables.append(f"static const int TIN{L}[{N2}][{N1}] = {{{','.join('{'+','.join(str(inmap[g][j]) for j in range(N1))+'}' for g in range(N2))}}};")
        tables.append(f"static const int TOUT{L}[{N1}][{N2}] = {{{','.join('{'+','.join(str(outmap[q][k]) for k in range(N2))+'}' for q in range(N1))}}};")
    def emit_flavor(name, has_csplit, write_state, write_dst):
        out = []
        argl = "double* restrict re, double* restrict im"
        if not write_state: argl = "const " + argl.replace("double* restrict im", "double* restrict im")  # keep simple; const-ness not critical
        args = ["double* restrict re, double* restrict im"]
        if has_csplit: args.append("const double* restrict cre, const double* restrict cim")
        else: args.append("const double* restrict cI")
        if write_dst: args.append("double* restrict dst")
        out.append(f"static void {name}_{L}({', '.join(args)}){{")
        out.append(f" for(long tl=0; tl<{ntile}; ++tl){{")
        out.append(f"  double* restrict r0 = re + tl*{8*TS};")
        out.append(f"  double* restrict i0 = im + tl*{8*TS};")
        if has_csplit:
            out.append(f"  const double* restrict c0r = cre + tl*{8*TS};")
            out.append(f"  const double* restrict c0i = cim + tl*{8*TS};")
        else:
            out.append(f"  const double* restrict c0 = cI + tl*{16*TS};")
        if write_dst:
            out.append(f"  double* restrict d0 = dst + tl*{16*TS};")
        # ---- stage A ----
        if L == 64:
            out.append("#pragma GCC unroll 1")
            out.append(f"  for(long g=0; g<8; ++g){{")
            out.append(f"   const double* twr = TW64R + (g-1)*8;")
            out.append(f"   const double* twi = TW64I + (g-1)*8;")
            out.append("#pragma GCC unroll 1")
            out.append(f"   for(long t=0; t<{TS}; ++t){{")
            def ldA(j1): return (f"LDU(r0 + (8*({j1})+g)*{S} + t*8)", f"LDU(i0 + (8*({j1})+g)*{S} + t*8)")
            def stA(k1, v, flag=False):
                if k1 == 0:
                    return [f"STU(TBR{L} + (0*8+g)*{8*TS} + t*8, {v[0]}); STU(TBI{L} + (0*8+g)*{8*TS} + t*8, {v[1]});"]
                return [f"{{ V wr, wi;",
                        f"  if(g){{ wr=SET1(twr[{k1}]); wi=SET1(twi[{k1}]); }} else {{ wr=SET1(1.0); wi=SET1(0.0); }}",
                        f"  STU(TBR{L} + ({k1}*8+g)*{8*TS} + t*8, FNMA(wi,{v[1]},MUL(wr,{v[0]})));",
                        f"  STU(TBI{L} + ({k1}*8+g)*{8*TS} + t*8, FMA(wr,{v[1]},MUL(wi,{v[0]}))); }}"]
            out += ["    " + l for l in ssa_kernel(8, ldA, stA)]
            out += ["   }", "  }"]
        else:
            out.append("#pragma GCC unroll 1")
            out.append(f"  for(long g=0; g<{N2}; ++g){{")
            out.append(f"   const int* inx = TIN{L}[g];")
            out.append("#pragma GCC unroll 1")
            out.append(f"   for(long t=0; t<{TS}; ++t){{")
            def ldA(j1): return (f"LDU(r0 + inx[{j1}]*{S} + t*8)", f"LDU(i0 + inx[{j1}]*{S} + t*8)")
            def stA(k1, v, flag=False):
                return [f"STU(TBR{L} + ({k1}*{N2}+g)*{8*TS} + t*8, {v[0]}); STU(TBI{L} + ({k1}*{N2}+g)*{8*TS} + t*8, {v[1]});"]
            out += ["    " + l for l in ssa_kernel(N1, ldA, stA)]
            out += ["   }", "  }"]
        # ---- stage B ----
        def finisher(kexpr, v):
            lines = []
            if has_csplit:
                lines.append(f"{{ V cr = LDU(c0r + {kexpr}*{S} + t*8), ci = LDU(c0i + {kexpr}*{S} + t*8);")
            else:
                lines.append(f"{{ V ca = LDU(c0 + {kexpr}*{2*L2} + t*16), cb2 = LDU(c0 + {kexpr}*{2*L2} + t*16 + 8);")
                lines.append(f"  V cr = _mm512_permutex2var_pd(ca, XRE, cb2), ci = _mm512_permutex2var_pd(ca, XIM, cb2);")
            lines.append(f"  V mr, mi; map8({v[0]}, {v[1]}, cr, ci, &mr, &mi);")
            if write_state:
                lines.append(f"  STU(r0 + {kexpr}*{S} + t*8, mr); STU(i0 + {kexpr}*{S} + t*8, mi);")
            if write_dst:
                lines.append(f"  STU(d0 + {kexpr}*{2*L2} + t*16, _mm512_permutex2var_pd(mr, XLO, mi));")
                lines.append(f"  STU(d0 + {kexpr}*{2*L2} + t*16 + 8, _mm512_permutex2var_pd(mr, XHI, mi));")
            lines.append("}")
            return lines
        if L == 64:
            out.append("#pragma GCC unroll 1")
            out.append(f"  for(long q=0; q<8; ++q){{")
            out.append("#pragma GCC unroll 1")
            out.append(f"   for(long t=0; t<{TS}; ++t){{")
            def ldB(j2): return (f"LDU(TBR{L} + (q*8+({j2}))*{8*TS} + t*8)", f"LDU(TBI{L} + (q*8+({j2}))*{8*TS} + t*8)")
            def stB(k2, v, flag=False): return finisher(f"(q+{8*k2})", v)
            out += ["    " + l for l in ssa_kernel(8, ldB, stB)]
            out += ["   }", "  }"]
        else:
            out.append("#pragma GCC unroll 1")
            out.append(f"  for(long q=0; q<{N1}; ++q){{")
            out.append(f"   const int* outx = TOUT{L}[q];")
            out.append("#pragma GCC unroll 1")
            out.append(f"   for(long t=0; t<{TS}; ++t){{")
            def ldB(j2): return (f"LDU(TBR{L} + (q*{N2}+({j2}))*{8*TS} + t*8)", f"LDU(TBI{L} + (q*{N2}+({j2}))*{8*TS} + t*8)")
            def stB(k2, v, flag=False): return finisher(f"outx[{k2}]", v)
            out += ["    " + l for l in ssa_kernel(N2, ldB, stB)]
            out += ["   }", "  }"]
        out += [" }", "}"]
        return out
    ln += emit_flavor("pxt", True, True, False)
    ln += emit_flavor("pxti", False, True, False)
    ln += emit_flavor("pxdt", False, True, True)
    ln += emit_flavor("pxft", False, False, True)
    ln.append(f"#define PXS{L}(a,b,c,d) do {{ pxt_{L}(a,b,c,d); pxs_{L}(a,b,c,d,{ntile*TS},{nub}); }} while(0)")
    ln.append(f"#define PXI{L}(a,b,c) do {{ pxti_{L}(a,b,c); px_{L}(a,b,c,{ntile*TS},{nub}); }} while(0)")
    ln.append(f"#define PXD{L}(a,b,c,d) do {{ pxdt_{L}(a,b,c,d); pxd_{L}(a,b,c,d,{ntile*TS},{nub}); }} while(0)")
    ln.append(f"#define PXF{L}(a,b,c,d) do {{ pxft_{L}(a,b,c,d); pxf_{L}(a,b,c,d,{ntile*TS},{nub}); }} while(0)")
    return ln

def gen_pz_first(L, tables):
    """First-iteration pz: loads from interleaved x0 (contiguous rows), stores split padded."""
    from gen import PS as _PS
    ps = _PS(L)
    nblk = (L + 7) // 8
    mode = PZ_MODE[L]
    VOL = L * ps
    ST_DUMP = VOL + 80
    NGALL = (L * L + 7) // 8
    ln = [f"static void pzf_{L}(const double* restrict x0, double* restrict re, double* restrict im){{"]
    # store offset provider per group (runtime g) mirrors pz modes
    if mode == 'plane64':
        # per-plane: caller passes plane pointers; 8 row-groups per plane
        NGALL = L // 8
        ln += [f" for(long g=0; g<{L//8}; ++g){{",
               f"  const double* restrict xb = x0 + g*{16*L};",
               f"  double* restrict rb = re + g*{8*L};",
               f"  double* restrict ib = im + g*{8*L};"]
        so = [f"{t*L}" for t in range(8)]
        base_r, base_i = "rb", "ib"
        ngroups_full = NGALL
        tail_valid = 8
    else:
        # store via table (works for all non-64 modes)
        loffs = []
        for r in range(NGALL * 8):
            loffs.append((r // L) * ps + (r % L) * L if r < L * L else ST_DUMP)
        nm = f"ROF{L}"
        tables.append(f"static const int {nm}[{NGALL*8}] = {{{','.join(map(str,loffs))}}};")
        ln += [f" for(long g=0; g<{NGALL}; ++g){{",
               f"  const double* restrict xb = x0 + g*{16*L};",
               f"  const int* ros = {nm} + g*8;"]
        so = [f"ros[{t}]" for t in range(8)]
        base_r, base_i = "re", "im"
        ngroups_full = NGALL
        tail_valid = (L * L) % 8 or 8
    last_g = NGALL - 1
    ln.append(f"  V sr[{L}], si[{L}], qr[{L}], qi[{L}];")
    ln.append(f"  int tail = (g == {last_g});")
    # loads: deinterleave per col-block; rows contiguous in x0 at t*2L
    for blk in range(nblk):
        c0 = blk * 8
        cnt = min(8, L - c0)
        rem2 = 2 * cnt
        m1 = (1 << min(rem2, 8)) - 1
        m2 = (1 << max(rem2 - 8, 0)) - 1 if rem2 > 8 else 0
        for t in range(8):
            ln.append(f"  {{")
            if tail_valid < 8:
                ln.append(f"   V a, b;")
                ln.append(f"   if(tail && {t} >= {tail_valid}){{ a = _mm512_setzero_pd(); b = _mm512_setzero_pd(); }}")
                ln.append(f"   else {{")
            else:
                ln.append(f"   V a, b; {{")
            if cnt == 8:
                ln.append(f"    a = LDU(xb + {t*2*L + 2*c0}); b = LDU(xb + {t*2*L + 2*c0 + 8});")
            else:
                ln.append(f"    a = _mm512_maskz_loadu_pd({hex(m1)}, xb + {t*2*L + 2*c0});")
                if m2:
                    ln.append(f"    b = _mm512_maskz_loadu_pd({hex(m2)}, xb + {t*2*L + 2*c0 + 8});")
                else:
                    ln.append(f"    b = _mm512_setzero_pd();")
            ln.append(f"   }}")
            ln.append(f"   pr{blk}_{t} = _mm512_permutex2var_pd(a, XRE, b);")
            ln.append(f"   pi{blk}_{t} = _mm512_permutex2var_pd(a, XIM, b);")
            ln.append(f"  }}")
        # declare before use
    # declarations must precede: prepend decls
    decl = []
    for blk in range(nblk):
        for t in range(8):
            decl.append(f"  V pr{blk}_{t}, pi{blk}_{t};")
    # insert decls right after sr/si decl
    idx = ln.index(f"  V sr[{L}], si[{L}], qr[{L}], qi[{L}];") + 1
    ln[idx:idx] = decl
    for blk in range(nblk):
        c0 = blk * 8
        cnt = min(8, L - c0)
        vs = [f"pr{blk}_{t}" for t in range(8)]
        ln.append(f"  TR8({','.join(vs)});")
        for t in range(cnt):
            ln.append(f"  sr[{c0+t}] = {vs[t]};")
        vs = [f"pi{blk}_{t}" for t in range(8)]
        ln.append(f"  TR8({','.join(vs)});")
        for t in range(cnt):
            ln.append(f"  si[{c0+t}] = {vs[t]};")
    def ld(j): return (f"sr[{j}]", f"si[{j}]")
    def st(k, v, flag): return [f"qr[{k}]={v[0]}; qi[{k}]={v[1]};"]
    ln += ["  " + l for l in emit_kernel2(L, ld, st, "zf", f"pzf{L}", tables)]
    for comp, base, arr in (("r", base_r, "qr"), ("i", base_i, "qi")):
        for blk in range(nblk):
            c0 = blk * 8
            cnt = min(8, L - c0)
            vs = [f"o{comp}{blk}_{t}" for t in range(8)]
            for t in range(8):
                srcv = f"{arr}[{c0+t}]" if t < cnt else f"{arr}[{c0}]"
                ln.append(f"  V {vs[t]} = {srcv};")
            ln.append(f"  TR8({','.join(vs)});")
            if cnt == 8:
                for t in range(8):
                    ln.append(f"  STU({base} + {so[t]} + {c0}, {vs[t]});")
            else:
                mk = (1 << cnt) - 1
                for t in range(8):
                    ln.append(f"  MST({base} + {so[t]} + {c0}, {hex(mk)}, {vs[t]});")
    ln += [" }", "}"]
    return ln

def gen_px_fin(L, tables):
    L2, L3 = L * L, L ** 3
    S = PS(L)
    nub = (L2 + 7) // 8
    rem = L2 % 8
    ln = [f"static void pxf_{L}(const double* restrict re, const double* restrict im, const double* restrict cI, double* restrict dst, long ub0, long ub1){{",
          f" for(long ub=ub0; ub<ub1; ++ub){{"]
    if rem:
        rem2 = 2 * rem
        m1 = (1 << min(rem2, 8)) - 1
        m2 = (1 << max(rem2 - 8, 0)) - 1
        ln.append(f"  const int tail = (ub=={nub-1});")
        ln.append(f"  const __mmask8 mlo = tail ? (__mmask8){hex(m1)} : (__mmask8)0xFF;")
        ln.append(f"  const __mmask8 mhi = tail ? (__mmask8){hex(m2)} : (__mmask8)0xFF;")
    else:
        ln.append(f"  const __mmask8 mlo = 0xFF, mhi = 0xFF;")
    ln += [f"  const double* restrict r0 = re + ub*8;",
           f"  const double* restrict i0 = im + ub*8;",
           f"  const double* restrict c0 = cI + ub*16;",
           f"  double* restrict d0 = dst + ub*16;"]
    def ld(j): return (f"LDU(r0 + ({j})*{S})", f"LDU(i0 + ({j})*{S})")
    def st(k, v, flag=False):
        return [f"{{ V ca = _mm512_maskz_loadu_pd(mlo, c0 + ({k})*{2*L2});",
                f"  V cb = _mm512_maskz_loadu_pd(mhi, c0 + ({k})*{2*L2} + 8);",
                f"  V cr = _mm512_permutex2var_pd(ca, XRE, cb);",
                f"  V ci = _mm512_permutex2var_pd(ca, XIM, cb);",
                f"  V mr, mi; map8({v[0]}, {v[1]}, cr, ci, &mr, &mi);",
                f"  MST(d0 + ({k})*{2*L2}, mlo, _mm512_permutex2var_pd(mr, XLO, mi));",
                f"  MST(d0 + ({k})*{2*L2} + 8, mhi, _mm512_permutex2var_pd(mr, XHI, mi)); }}"]
    ln += ["  " + l for l in emit_kernel2(L, ld, st, "xf", f"pxf{L}", tables)]
    ln += [" }", "}"]
    return ln


FUSE_ZY = {}
BATCHED_L = (6, 8, 13, 17, 23)
BUNROLL = {6: 4, 8: 4, 13: 4, 17: 1, 23: 1}
REMTH = {6:5, 8:7, 13:6, 17:6, 23:8}

def gen_batched(L, tables):
    """8-volume lane-major pipeline: element (pos, lane v) at buf[pos*8+v]."""
    L2, L3 = L * L, L ** 3
    ln = []
    # ---- passes ----
    # bzy: per x-slab: z-FFT lines (x,y) then y-FFT lines (x,z); slab stays cache-hot
    UN = BUNROLL.get(L, 1)
    ln.append(f"static void bzy_{L}(double* restrict re, double* restrict im){{")
    ln.append(f" for(long x=0; x<{L}; ++x){{")
    ln.append(f"  double* restrict rs = re + x*{8*L2};")
    ln.append(f"  double* restrict is = im + x*{8*L2};")
    ln.append(f"#pragma GCC unroll {{UN}}".replace("{UN}", str(UN)))
    ln.append(f"  for(long y=0; y<{L}; ++y){{")
    ln.append(f"   double* restrict r0 = rs + y*{8*L};")
    ln.append(f"   double* restrict i0 = is + y*{8*L};")
    def ldz(j): return (f"LDU(r0 + ({j})*8)", f"LDU(i0 + ({j})*8)")
    def stz(k, v, flag=False): return [f"STU(r0 + ({k})*8, {v[0]}); STU(i0 + ({k})*8, {v[1]});"]
    ln += ["   " + l for l in emit_kernel2(L, ldz, stz, "bz", f"bpz{L}", tables)]
    ln += ["  }"]
    ln.append(f"#pragma GCC unroll {{UN}}".replace("{UN}", str(UN)))
    ln.append(f"  for(long z=0; z<{L}; ++z){{")
    ln.append(f"   double* restrict r0 = rs + z*8;")
    ln.append(f"   double* restrict i0 = is + z*8;")
    def ldy(j): return (f"LDU(r0 + ({j})*{8*L})", f"LDU(i0 + ({j})*{8*L})")
    def sty(k, v, flag=False): return [f"STU(r0 + ({k})*{8*L}, {v[0]}); STU(i0 + ({k})*{8*L}, {v[1]});"]
    ln += ["   " + l for l in emit_kernel2(L, ldy, sty, "by", f"bpy{L}", tables)]
    ln += ["  }", " }", "}"]
    # bpx: for u in L2: stride L2; fused c+map
    ln.append(f"static void bpx_{L}(double* restrict re, double* restrict im, const double* restrict cre, const double* restrict cim){{")
    ln.append(f"#pragma GCC unroll {{UN}}".replace("{UN}", str(UN)))
    ln.append(f" for(long u=0; u<{L2}; ++u){{")
    ln.append(f"  double* restrict r0 = re + u*8;")
    ln.append(f"  double* restrict i0 = im + u*8;")
    ln.append(f"  const double* restrict c0r = cre + u*8;")
    ln.append(f"  const double* restrict c0i = cim + u*8;")
    def ldx(j): return (f"LDU(r0 + ({j})*{8*L2})", f"LDU(i0 + ({j})*{8*L2})")
    def stx(k, v, flag=False):
        return [f"{{ V mr, mi; map8({v[0]}, {v[1]}, LDU(c0r + ({k})*{8*L2}), LDU(c0i + ({k})*{8*L2}), &mr, &mi);",
                f"  STU(r0 + ({k})*{8*L2}, mr); STU(i0 + ({k})*{8*L2}, mi); }}"]
    ln += ["  " + l for l in emit_kernel2(L, ldx, stx, "bx", f"bpx{L}", tables)]
    ln += [" }", "}"]
    # ---- conversions ----
    rem = L3 % 8
    nblk = (L3 + 7) // 8
    if rem:
        rem2 = 2 * rem
        m1 = (1 << min(rem2, 8)) - 1
        m2 = (1 << max(rem2 - 8, 0)) - 1
    # in: src interleaved volumes (nv<=8), dst lane-major split
    ln.append(f"""
static void bin_{L}(const double* restrict src, double* restrict re, double* restrict im, long nv){{
 for(long pb=0; pb<{nblk}; ++pb){{
  long p0 = pb*8;
  V rr[8], ii[8];
  {"__mmask8 q1 = 0xFF, q2 = 0xFF; if(pb == %d){ q1 = %s; q2 = %s; }" % (nblk-1, hex(m1), hex(m2)) if rem else "const __mmask8 q1 = 0xFF, q2 = 0xFF;"}
  for(long v=0; v<8; ++v){{
    V a, b;
    if(v < nv){{
      a = _mm512_maskz_loadu_pd(q1, src + v*{2*L3} + p0*2);
      b = _mm512_maskz_loadu_pd(q2, src + v*{2*L3} + p0*2 + 8);
    }} else {{ a = _mm512_setzero_pd(); b = a; }}
    rr[v] = _mm512_permutex2var_pd(a, XRE, b);
    ii[v] = _mm512_permutex2var_pd(a, XIM, b);
  }}
  TR8(rr[0],rr[1],rr[2],rr[3],rr[4],rr[5],rr[6],rr[7]);
  TR8(ii[0],ii[1],ii[2],ii[3],ii[4],ii[5],ii[6],ii[7]);
  for(long t=0; t<8; ++t){{ STU(re + (p0+t)*8, rr[t]); STU(im + (p0+t)*8, ii[t]); }}
 }}
}}
static void bout_{L}(const double* restrict re, const double* restrict im, double* restrict dst, long nv){{
 for(long pb=0; pb<{nblk}; ++pb){{
  long p0 = pb*8;
  V rr[8], ii[8];
  {"__mmask8 q1 = 0xFF, q2 = 0xFF; if(pb == %d){ q1 = %s; q2 = %s; }" % (nblk-1, hex(m1), hex(m2)) if rem else "const __mmask8 q1 = 0xFF, q2 = 0xFF;"}
  for(long t=0; t<8; ++t){{ rr[t] = LDU(re + (p0+t)*8); ii[t] = LDU(im + (p0+t)*8); }}
  TR8(rr[0],rr[1],rr[2],rr[3],rr[4],rr[5],rr[6],rr[7]);
  TR8(ii[0],ii[1],ii[2],ii[3],ii[4],ii[5],ii[6],ii[7]);
  for(long v=0; v<nv; ++v){{
    MST(dst + v*{2*L3} + p0*2,     q1, _mm512_permutex2var_pd(rr[v], XLO, ii[v]));
    MST(dst + v*{2*L3} + p0*2 + 8, q2, _mm512_permutex2var_pd(rr[v], XHI, ii[v]));
  }}
 }}
}}""")
    return ln

def gen_pz_first_plane(L, tables):
    """Per-plane first-iteration pz for 36/45: reads interleaved x0 plane, dump rows zeroed."""
    from gen import PS as _PS
    ps = _PS(L)
    nblk = (L + 7) // 8
    NGP = (L + 7) // 8
    VOL = L * ps
    ST_DUMP = VOL + 80
    ln = [f"static void pzfp_{L}(const double* restrict x0, double* restrict rp, double* restrict ip, double* restrict re, double* restrict im){{"]
    # x0: plane base (interleaved); rp/ip: plane split base; re/im: array bases for dump stores
    for gg in range(NGP):
        ln.append(" {")
        ln.append(f"  V sr[{L}], si[{L}], qr[{L}], qi[{L}];")
        for blk in range(nblk):
            c0 = blk * 8
            cnt = min(8, L - c0)
            rem2 = 2 * cnt
            m1 = (1 << min(rem2, 8)) - 1
            m2 = (1 << max(rem2 - 8, 0)) - 1 if rem2 > 8 else 0
            vsr = [f"pr{gg}_{blk}_{t}" for t in range(8)]
            vsi = [f"pi{gg}_{blk}_{t}" for t in range(8)]
            for t in range(8):
                r = gg * 8 + t
                ln.append(f"  V {vsr[t]}, {vsi[t]};")
                if r >= L:
                    ln.append(f"  {vsr[t]} = _mm512_setzero_pd(); {vsi[t]} = _mm512_setzero_pd();")
                else:
                    ln.append("  {")
                    if cnt == 8:
                        ln.append(f"   V a = LDU(x0 + {r*2*L + 2*c0}); V b = LDU(x0 + {r*2*L + 2*c0 + 8});")
                    else:
                        ln.append(f"   V a = _mm512_maskz_loadu_pd({hex(m1)}, x0 + {r*2*L + 2*c0});")
                        if m2:
                            ln.append(f"   V b = _mm512_maskz_loadu_pd({hex(m2)}, x0 + {r*2*L + 2*c0 + 8});")
                        else:
                            ln.append(f"   V b = _mm512_setzero_pd();")
                    ln.append(f"   {vsr[t]} = _mm512_permutex2var_pd(a, XRE, b);")
                    ln.append(f"   {vsi[t]} = _mm512_permutex2var_pd(a, XIM, b);")
                    ln.append("  }")
            ln.append(f"  TR8({','.join(vsr)});")
            for t in range(cnt):
                ln.append(f"  sr[{c0+t}] = {vsr[t]};")
            ln.append(f"  TR8({','.join(vsi)});")
            for t in range(cnt):
                ln.append(f"  si[{c0+t}] = {vsi[t]};")
        def ld(j): return (f"sr[{j}]", f"si[{j}]")
        def st(k, v, flag): return [f"qr[{k}]={v[0]}; qi[{k}]={v[1]};"]
        ln += ["  " + l for l in emit_kernel2(L, ld, st, f"zp{gg}", f"pzfp{L}g{gg}", tables)]
        for comp, pb, ab, arr in (("r", "rp", "re", "qr"), ("i", "ip", "im", "qi")):
            for blk in range(nblk):
                c0 = blk * 8
                cnt = min(8, L - c0)
                vs = [f"o{comp}{gg}_{blk}_{t}" for t in range(8)]
                for t in range(8):
                    srcv = f"{arr}[{c0+t}]" if t < cnt else f"{arr}[{c0}]"
                    ln.append(f"  V {vs[t]} = {srcv};")
                ln.append(f"  TR8({','.join(vs)});")
                mk = (1 << cnt) - 1
                for t in range(8):
                    r = gg * 8 + t
                    dst = f"{pb} + {r*L} + {c0}" if r < L else f"{ab} + {ST_DUMP} + {c0}"
                    if cnt == 8:
                        ln.append(f"  STU({dst}, {vs[t]});")
                    else:
                        ln.append(f"  MST({dst}, {hex(mk)}, {vs[t]});")
        ln.append(" }")
    ln.append("}")
    return ln

def gen_pz_plane(L, tables):
    """Per-plane pz for 36/45 (constant offsets; dump rows to pad)."""
    from gen import PS as _PS
    ps = _PS(L)
    nblk = (L + 7) // 8
    NGP = (L + 7) // 8
    VOL = L * ps
    LD_DUMP = VOL + 16
    ST_DUMP = VOL + 80
    ln = [f"static void pzp_{L}(double* restrict rp, double* restrict ip, double* restrict re, double* restrict im){{"]
    for gg in range(NGP):
        ln.append(" {")
        ln.append(f"  V sr[{L}], si[{L}], qr[{L}], qi[{L}];")
        for comp, pb, ab, arr in (("r", "rp", "re", "sr"), ("i", "ip", "im", "si")):
            for blk in range(nblk):
                c0 = blk * 8
                cnt = min(8, L - c0)
                vs = [f"p{comp}{gg}_{blk}_{t}" for t in range(8)]
                for t in range(8):
                    r = gg * 8 + t
                    srcp = f"{pb} + {r*L} + {c0}" if r < L else f"{ab} + {LD_DUMP} + {c0}"
                    ln.append(f"  V {vs[t]} = LDU({srcp});")
                ln.append(f"  TR8({','.join(vs)});")
                for t in range(cnt):
                    ln.append(f"  {arr}[{c0+t}] = {vs[t]};")
        def ld(j): return (f"sr[{j}]", f"si[{j}]")
        def st(k, v, flag): return [f"qr[{k}]={v[0]}; qi[{k}]={v[1]};"]
        ln += ["  " + l for l in emit_kernel2(L, ld, st, f"zq{gg}", f"pzp{L}g{gg}", tables)]
        for comp, pb, ab, arr in (("r", "rp", "re", "qr"), ("i", "ip", "im", "qi")):
            for blk in range(nblk):
                c0 = blk * 8
                cnt = min(8, L - c0)
                vs = [f"o{comp}{gg}_{blk}_{t}" for t in range(8)]
                for t in range(8):
                    srcv = f"{arr}[{c0+t}]" if t < cnt else f"{arr}[{c0}]"
                    ln.append(f"  V {vs[t]} = {srcv};")
                ln.append(f"  TR8({','.join(vs)});")
                mk = (1 << cnt) - 1
                for t in range(8):
                    r = gg * 8 + t
                    dst = f"{pb} + {r*L} + {c0}" if r < L else f"{ab} + {ST_DUMP} + {c0}"
                    if cnt == 8:
                        ln.append(f"  STU({dst}, {vs[t]});")
                    else:
                        ln.append(f"  MST({dst}, {hex(mk)}, {vs[t]});")
        ln.append(" }")
    ln.append("}")
    return ln

def gen_dbg(L, tables):
    ln = [f"void dbg1d_{L}(double* re, double* im){{"]
    def ld(j): return (f"LDU(re + ({j})*8)", f"LDU(im + ({j})*8)")
    def st(k, v, flag):
        return [f"STU(re + ({k})*8, {v[0]}); STU(im + ({k})*8, {v[1]});"]
    ln += [" " + l for l in emit_kernel2(L, ld, st, "d", f"db{L}", tables)]
    ln.append("}")
    return ln

def gen_run(L):
    L3 = L ** 3
    L2 = L * L
    ps = PS(L)
    VOL = L * ps
    PAD = 8 * L + 128
    ln = []
    arrb = (VOL + PAD) * 8
    delta = ((arrb + 4095) // 4096) * 4096 + 1024
    dd = delta // 8
    global ARENA_OFF
    base = ARENA_OFF
    for i, nm in enumerate(("SR", "SI", "CR", "CI")):
        ln.append(f"#define {nm}{L} (ARENA + {base + i * dd})")
    ARENA_OFF = base + 4 * dd
    ARENA_OFF = ((ARENA_OFF * 8 + 4095) // 4096 * 4096) // 8 + 512
    if L in BATCHED_L:
        bsz = (L3 + 8) * 8
        bdelta = ((bsz * 8 + 4095) // 4096) * 4096 + 1024
        bdd = bdelta // 8
        base = ARENA_OFF
        for i, nm in enumerate(("SBR", "SBI", "CBR", "CBI")):
            ln.append(f"#define {nm}{L} (ARENA + {base + i * bdd})")
        ARENA_OFF = base + 4 * bdd
        ARENA_OFF = ((ARENA_OFF * 8 + 4095) // 4096 * 4096) // 8 + 512
    if L == 64:
        step = f"""
      for(long x=0; x<{L}; ++x){{
        pz_{L}(SR{L} + x*{ps}, SI{L} + x*{ps});
        py_{L}(SR{L} + x*{ps}, SI{L} + x*{ps});
      }}"""
    elif L in (36, 45) and FUSE_ZY.get(L, False):
        step = f"""
      for(long x=0; x<{L}; ++x){{
        pzp_{L}(SR{L} + x*{ps}, SI{L} + x*{ps}, SR{L}, SI{L});
        pyp_{L}(SR{L} + x*{ps}, SI{L} + x*{ps});
      }}"""
        pyonly = f"""
      for(long x=0; x<{L}; ++x) py_{L}(SR{L} + x*{ps}, SI{L} + x*{ps});"""
    else:
        step = f"""
      pz_{L}(SR{L}, SI{L});
      py_{L}(SR{L}, SI{L});"""
        pyonly = f"""
      py_{L}(SR{L}, SI{L});"""
    dei_c = f"""    for(long x=0; x<{L}; ++x)
      deint(cb + x*{2*L2}, CR{L} + x*{ps}, CI{L} + x*{ps}, {L2});"""
    dei_x = f"""    for(long x=0; x<{L}; ++x)
      deint(xb + x*{2*L2}, SR{L} + x*{ps}, SI{L} + x*{ps}, {L2});"""
    def mk_inter(dst):
        return f"""for(long x=0; x<{L}; ++x) inter(SR{L} + x*{ps}, SI{L} + x*{ps}, {dst} + b*{2*L3} + x*{2*L2}, {L2});"""
    ln.append(f"""
{f"""
static void bgroup_{L}(int64_t m, const double* xb, const double* cb, double* o1, double* om, long nv){{
  bin_{L}(xb, SBR{L}, SBI{L}, nv);
  bin_{L}(cb, CBR{L}, CBI{L}, nv);
  if(m < 1){{
    bout_{L}(SBR{L}, SBI{L}, om, nv);
    bzy_{L}(SBR{L}, SBI{L}); bpx_{L}(SBR{L}, SBI{L}, CBR{L}, CBI{L});
    bout_{L}(SBR{L}, SBI{L}, o1, nv);
    return;
  }}
  for(int64_t it=0; it<m; ++it){{
    bzy_{L}(SBR{L}, SBI{L}); bpx_{L}(SBR{L}, SBI{L}, CBR{L}, CBI{L});
    if(it == 0) bout_{L}(SBR{L}, SBI{L}, o1, nv);
  }}
  if(m != 1 || om != o1) bout_{L}(SBR{L}, SBI{L}, om, nv);
}}""" if L in BATCHED_L else ""}
void run{L}(int64_t B, int64_t m, const double* x0, const double* c, double* out1, double* outm){{
  init_all();
  int64_t b = 0;
{f"""  {{
    int64_t g8 = (B / 8) * 8;
    int64_t rem = B - g8;
    for(; b < g8; b += 8)
      bgroup_{L}(m, x0 + b*{2*L3}, c + b*{2*L3}, out1 + b*{2*L3}, outm + b*{2*L3}, 8);
    if(rem >= {REMTH[L]}){{
      bgroup_{L}(m, x0 + b*{2*L3}, c + b*{2*L3}, out1 + b*{2*L3}, outm + b*{2*L3}, rem);
      b = B;
    }}
  }}""" if L in BATCHED_L else ""}
  for(; b<B; ++b){{
    const double* xb = x0 + b*{2*L3};
    const double* cb = c  + b*{2*L3};
    if(m < 1){{
{dei_x}
      {mk_inter("outm")}
      {step}
      PXI{L}(SR{L}, SI{L}, cb);
      {mk_inter("out1")}
      continue;
    }}
    // iteration 1: fused deinterleave of x0
{f"""    for(long x=0; x<{L}; ++x){{
      pzf_{L}(xb + x*{2*L2}, SR{L} + x*{ps}, SI{L} + x*{ps});
      py_{L}(SR{L} + x*{ps}, SI{L} + x*{ps});
    }}""" if L == 64 else (f"""    for(long x=0; x<{L}; ++x){{
      pzfp_{L}(xb + x*{2*L2}, SR{L} + x*{ps}, SI{L} + x*{ps}, SR{L}, SI{L});
      pyp_{L}(SR{L} + x*{ps}, SI{L} + x*{ps});
    }}""" if L in (36, 45) else f"""    pzf_{L}(xb, SR{L}, SI{L});
    {pyonly}""")}
    if(m == 1){{
      PXF{L}(SR{L}, SI{L}, cb, out1 + b*{2*L3});
      if(outm != out1) memcpy(outm + b*{2*L3}, out1 + b*{2*L3}, {2*L3}*sizeof(double));
      continue;
    }}
    if(m >= {6 if L not in TILED_PX else 32}){{
      for(long x=0; x<{L}; ++x)
        deint(cb + x*{2*L2}, CR{L} + x*{ps}, CI{L} + x*{ps}, {L2});
      PXS{L}(SR{L}, SI{L}, CR{L}, CI{L});
      {mk_inter("out1")}
      for(int64_t it=1; it<m-1; ++it){{
        {step}
        PXS{L}(SR{L}, SI{L}, CR{L}, CI{L});
      }}
    }} else {{
      PXD{L}(SR{L}, SI{L}, cb, out1 + b*{2*L3});
      for(int64_t it=1; it<m-1; ++it){{
        {step}
        PXI{L}(SR{L}, SI{L}, cb);
      }}
    }}
    {step}
    PXF{L}(SR{L}, SI{L}, cb, outm + b*{2*L3});
  }}
}}
static const double* DBGC{L};
void dtr{L}(long reps){{ for(long r=0;r<reps;++r){{ {"for(long x=0;x<%d;++x){ pz_%d(SR%d + x*%d, SI%d + x*%d); py_%d(SR%d + x*%d, SI%d + x*%d);} PXS%d(SR%d, SI%d, CR%d, CI%d);" % (L,L,L,ps,L,ps,L,L,ps,L,ps,L,L,L,L,L) if L==64 else "pz_%d(SR%d, SI%d); py_%d(SR%d, SI%d); PXS%d(SR%d, SI%d, CR%d, CI%d);" % (L,L,L,L,L,L,L,L,L,L,L)} }} }}
void dpz{L}(long reps){{ for(long r=0;r<reps;++r){{ {"for(long x=0;x<%d;++x) pz_%d(SR%d + x*%d, SI%d + x*%d);" % (L,L,L,ps,L,ps) if L==64 else "pz_%d(SR%d, SI%d);" % (L,L,L)} }} }}
void dpy{L}(long reps){{ for(long r=0;r<reps;++r){{ {"for(long x=0;x<%d;++x) py_%d(SR%d + x*%d, SI%d + x*%d);" % (L,L,L,ps,L,ps) if L==64 else "py_%d(SR%d, SI%d);" % (L,L,L)} }} }}
void dpx{L}(long reps){{ for(long r=0;r<reps;++r){{ PXS{L}(SR{L}, SI{L}, CR{L}, CI{L}); }} }}
void dseed{L}(const double* x0, const double* c){{
  init_all();
  DBGC{L} = c;
  for(long x=0; x<{L}; ++x){{
    deint(x0 + x*{2*L2}, SR{L} + x*{ps}, SI{L} + x*{ps}, {L2});
    deint(c + x*{2*L2}, CR{L} + x*{ps}, CI{L} + x*{ps}, {L2});
  }}
}}""")
    return ln

ARENA_OFF = 0

def main():
    global ARENA_OFF
    ARENA_OFF = 0
    tables = ["void init_all(void);"]
    body = []
    for L in SIZES:
        if L in PRIME_KB:
            tables += gen_prime_tabs(L)
        body.append("\n".join(gen_pz(L, tables)))
        body.append("\n".join(gen_pz_first(L, tables)))
        if L in (36, 45):
            body.append("\n".join(gen_pz_first_plane(L, tables)))
            if FUSE_ZY.get(L, False):
                body.append("\n".join(gen_pz_plane(L, tables)))
        body.append("\n".join(gen_py(L, tables)))
        body.append("\n".join(gen_px(L, tables)))
        body.append("\n".join(gen_px_split(L, tables)))
        body.append("\n".join(gen_pxd(L, tables)))
        if L in TILED_PX:
            body.append("\n".join(gen_px_tiled(L, tables)))
        else:
            body.append(f"#define PXS{L}(a,b,c,d) pxs_{L}(a,b,c,d,0,{(L*L+7)//8})")
            body.append(f"#define PXF{L}(a,b,c,d) pxf_{L}(a,b,c,d,0,{(L*L+7)//8})")
            body.append(f"#define PXI{L}(a,b,c) px_{L}(a,b,c,0,{(L*L+7)//8})")
            body.append(f"#define PXD{L}(a,b,c,d) pxd_{L}(a,b,c,d,0,{(L*L+7)//8})")
        body.append("\n".join(gen_px_fin(L, tables)))
        if L in BATCHED_L:
            body.append("\n".join(gen_batched(L, tables)))
        body.append("\n".join(gen_dbg(L, tables)))
        body.append("\n".join(gen_run(L)))
    body.append(f"void init_all(void){{ init0({ARENA_OFF + 64}); }}")
    src = PREAMBLE + "\n" + "\n".join(tables) + "\n" + "\n".join(body) + "\n"
    with open("implementation.c", "w") as f:
        f.write(src)
    print(f"wrote implementation.c: {len(src)} bytes, {src.count(chr(10))} lines")

if __name__ == "__main__":
    main()
