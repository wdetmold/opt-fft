import sys
sys.path.insert(0, '/tmp/dev')
from gen2 import *      # IR + dft + crt_pos + BCFG
from genloop import BC, emit_map_store, MAPC

def subdft_body(n, W, in_exprs, prefix):
    """emit straight-line DFT of length n; in_exprs: list of (re_cexpr, im_cexpr);
       returns (lines, out_names list of (re,im))"""
    g = G()
    inv = {}
    xin = []
    for t in range(n):
        ir = g.inp(('x', t, 0)); ii = g.inp(('x', t, 1))
        inv[ir] = in_exprs[t][0]; inv[ii] = in_exprs[t][1]
        xin.append((ir, ii))
    outs = dft(g, n, xin)
    lines, names = body_lines(g, outs, inv, W)
    import re as _re
    ren = lambda s: _re.sub(r"\bt(\d+)\b", prefix + r"t\1", s)
    lines = [ren(l) for l in lines]
    onames = []
    for (r, i) in outs:
        onames.append((ren(names[r]), ren(names[i])))
    return lines, onames

def staged_loop(L, W, kind, name, P, PS):
    """class B kernel as loops. kind: 'colT' (stride P) or 'xfused' (stride PS)."""
    typ, n1, n2 = BCFG[L]
    pos = crt_pos(L)
    T = TYPE[W]
    o = []
    tabs = []
    S = P if kind in ('colT','colT_mapc') else PS
    # ---- offset tables for stage1 loads ----
    # group index g in 0..n2-1 ; element a in 0..n1-1
    offs = []
    for gi in range(n2):
        if typ == 'pfa':
            rows = [(a*n2 + gi*n1) % L for a in range(n1)]
        else:
            rows = [8*c + gi for c in range(8)]
        offs.append([ (pos[r]*P if kind in ('colT','colT_mapc') else r*PS) for r in rows ])
    # check if affine in a: offs[gi][a] = base_g + a*stride ?
    aff = all(all(offs[gi][a] - offs[gi][0] == a*(offs[gi][1]-offs[gi][0]) for a in range(n1)) for gi in range(n2))
    aff = aff and all(offs[gi][1]-offs[gi][0] == offs[0][1]-offs[0][0] for gi in range(n2))
    gbase_aff = aff and all(offs[gi][0] - offs[0][0] == gi*(offs[1][0]-offs[0][0]) for gi in range(n2))
    if kind == 'colT':
        o.append(f"static void {name}(const double*restrict pr, const double*restrict pi, double*restrict dr, double*restrict di){{")
    elif kind == 'colT_mapc':
        o.append(f"static void {name}(const double*restrict pr, const double*restrict pi, const double*restrict cr, const double*restrict ci, double*restrict dr, double*restrict di){{")
    elif kind == 'xplain':
        o.append(f"static void {name}(double*restrict pr, double*restrict pi){{")
    else:
        o.append(f"static void {name}(double*restrict pr, double*restrict pi, const double*restrict cr, const double*restrict ci){{")
    o.append(f"  {T} Br[{L}] __attribute__((aligned(64)));")
    o.append(f"  {T} Bi[{L}] __attribute__((aligned(64)));")
    # ---------- stage 1 ----------
    if typ == 'ct':
        # twiddle table [d][2b],[2b+1]
        tw_rows = []
        for d in range(n2):
            row = []
            for b in range(n1):
                c, s = tw(L, d*b)
                row += [c, s]
            tw_rows.append(row)
        tabs.append(f"static const double TW_{name}[{n2}][{2*n1}] __attribute__((aligned(64))) = {{\n  "
                    + ",\n  ".join("{"+",".join(hexf(v) for v in r)+"}" for r in tw_rows) + "};")
    if gbase_aff:
        stride_a = offs[0][1]-offs[0][0]
        stride_g = offs[1][0]-offs[0][0]
        base0 = offs[0][0]
        o.append(f"  {{ const double* qr = pr + {base0}; const double* qi = pi + {base0};")
        if kind == 'colT_mapc':
            o.append(f"    const double* qcr = cr + {base0}; const double* qci = ci + {base0};")
        o.append(f"    {T}* bp_r = Br; {T}* bp_i = Bi;")
        if typ == 'ct':
            o.append(f"    const double* twp = &TW_{name}[0][0];")
        o.append(f"    for(int g=0; g<{n2}; ++g){{")
        if kind == 'colT_mapc':
            o.append(f"      {T} mv[{2*n1}] __attribute__((aligned(64)));")
            for a in range(n1):
                o.append(f"      {{ {T} zr = *(const {T}*)(qr + {a*stride_a}) + *(const {T}*)(qcr + {a*stride_a});")
                o.append(f"        {T} zi = *(const {T}*)(qi + {a*stride_a}) + *(const {T}*)(qci + {a*stride_a});")
                o.append(f"        {T} mm = zr*zr + zi*zi;")
                o.append(f"        {T} uu = rsq{W}(mm);")
                o.append(f"        {T} wr_ = zr*uu, wi_ = zi*uu;")
                o.append(f"        {T} vv = rpc{W}(uu);")
                o.append(f"        mv[{2*a}] = wr_*vv; mv[{2*a+1}] = wi_*vv; }}")
            in_exprs = [(f"mv[{2*a}]", f"mv[{2*a+1}]") for a in range(n1)]
        else:
            in_exprs = [(f"(*(const {T}*)(qr + {a*stride_a}))", f"(*(const {T}*)(qi + {a*stride_a}))") for a in range(n1)]
    else:
        tabs.append(f"static const long OFF1_{name}[{n2}][{n1}] = {{\n  "
                    + ",\n  ".join("{"+",".join(str(v) for v in r)+"}" for r in offs) + "};")
        o.append(f"  {{ {T}* bp_r = Br; {T}* bp_i = Bi;")
        o.append(f"    for(int g=0; g<{n2}; ++g){{")
        o.append(f"      const long* ofp = OFF1_{name}[g];")
        in_exprs = [(f"(*(const {T}*)(pr + ofp[{a}]))", f"(*(const {T}*)(pi + ofp[{a}]))") for a in range(n1)]
    lines, onames = subdft_body(n1, W, in_exprs, "s1")
    o += ["    " + l.strip() if l.strip().startswith(TYPE[W]) or True else l for l in lines]
    if typ == 'ct':
        # twiddle multiply b=1..7 (b=0 passthrough)
        newnames = [onames[0]]
        for b in range(1, n1):
            yr, yi = onames[b]
            o.append(f"      {{ const {T} wr = {BC(W, f'twp[{2*b}]')}, wi = {BC(W, f'twp[{2*b+1}]')};")
            o.append(f"        {T} h{b}r = {yr}*wr - {yi}*wi; {T} h{b}i = {yr}*wi + {yi}*wr;")
            o.append(f"      Br[{b}*{n2}] = h{b}r; Bi[{b}*{n2}] = h{b}i; }}")
        o.append(f"      Br[0] = {onames[0][0]}; Bi[0] = {onames[0][1]};")
        # note: writes via bp pointer below instead -- fix: use bp_r indexing
    else:
        for k1 in range(n1):
            o.append(f"      bp_r[{k1*n2}] = {onames[k1][0]}; bp_i[{k1*n2}] = {onames[k1][1]};")
    if typ == 'ct':
        pass
    o.append(f"      bp_r += 1; bp_i += 1;")
    if gbase_aff:
        o.append(f"      qr += {stride_g}; qi += {stride_g};")
        if kind == 'colT_mapc':
            o.append(f"      qcr += {stride_g}; qci += {stride_g};")
    if typ == 'ct':
        o.append(f"      twp += {2*n1};")
    o.append("    }")
    o.append("  }")
    src = "\n".join(o)
    # fix CT buffer writes to use bp_r
    src = src.replace(f"Br[0] = ", f"bp_r[0] = ").replace(f"Bi[0] = ", f"bp_i[0] = ")
    for b in range(1, n1):
        src = src.replace(f"Br[{b}*{n2}] = h{b}r; Bi[{b}*{n2}] = h{b}i;",
                          f"bp_r[{b}*{n2}] = h{b}r; bp_i[{b}*{n2}] = h{b}i;")
    o = src.split("\n")
    # ---------- stage 2 ----------
    if kind == 'xplain':
        outoff = []
        for gi in range(n1):
            row = []
            for t in range(n2):
                if typ == 'pfa':
                    k = next(kk for kk in range(L) if kk % n1 == gi and kk % n2 == t)
                else:
                    k = 8*t + gi
                row.append(k*PS)
            outoff.append(row)
        st_t = outoff[0][1]-outoff[0][0]; st_g = outoff[1][0]-outoff[0][0]
        aff2 = all(all(outoff[gi][t] == gi*st_g + t*st_t for t in range(n2)) for gi in range(n1))
        assert aff2, "staged_loop xplain requires affine output offsets (CT only)"
        o.append(f"  {{ const {T}* bp_r = Br; const {T}* bp_i = Bi;")
        o.append(f"    for(int g=0; g<{n1}; ++g){{")
        in_exprs = [(f"bp_r[{t}]", f"bp_i[{t}]") for t in range(n2)]
        lines, onames = subdft_body(n2, W, in_exprs, "s2")
        o += lines
        o.append(f"      double* xo_r = pr + (long)g*{st_g}; double* xo_i = pi + (long)g*{st_g};")
        for t in range(n2):
            o.append(f"      *({T}*)(xo_r + {t*st_t}) = {onames[t][0]};")
            o.append(f"      *({T}*)(xo_i + {t*st_t}) = {onames[t][1]};")
        o.append(f"      bp_r += {n2}; bp_i += {n2};")
        o.append("    }")
        o.append("  }")
    elif kind == 'xfused':
        # out row offsets
        outoff = []
        for gi in range(n1):
            row = []
            for t in range(n2):
                if typ == 'pfa':
                    k = next(kk for kk in range(L) if kk % n1 == gi and kk % n2 == t)
                else:
                    k = 8*t + gi
                row.append(k*PS)
            outoff.append(row)
        aff2 = all(all(outoff[gi][t] - outoff[gi][0] == t*(outoff[gi][1]-outoff[gi][0]) for t in range(n2)) for gi in range(n1))
        aff2 = aff2 and all(outoff[gi][1]-outoff[gi][0] == outoff[0][1]-outoff[0][0] for gi in range(n1)) \
               and all(outoff[gi][0]-outoff[0][0] == gi*(outoff[1][0]-outoff[0][0]) for gi in range(n1))
        if not aff2:
            tabs.append(f"static const long OUT_{name}[{n1}][{n2}] = {{\n  "
                        + ",\n  ".join("{"+",".join(str(v) for v in r)+"}" for r in outoff) + "};")
        o.append(f"  {{ const {T}* bp_r = Br; const {T}* bp_i = Bi;")
        o.append(f"    for(int g=0; g<{n1}; ++g){{")
        in_exprs = [(f"bp_r[{t}]", f"bp_i[{t}]") for t in range(n2)]
        lines, onames = subdft_body(n2, W, in_exprs, "s2")
        o += lines
        if aff2:
            st_t = outoff[0][1]-outoff[0][0]; st_g = outoff[1][0]-outoff[0][0]
            o.append(f"      double* xo_r = pr + (long)g*{st_g}; double* xo_i = pi + (long)g*{st_g};")
            o.append(f"      const double* co_r = cr + (long)g*{st_g}; const double* co_i = ci + (long)g*{st_g};")
            if W == 8:
                for t in range(n2):
                    o.append(f"      _mm_prefetch((const char*)(co_r + {t*(outoff[0][1]-outoff[0][0])} + 8), _MM_HINT_T0);")
            for t in range(n2):
                emit_map_store(o, W, T, "      ", onames[t][0], onames[t][1],
                               f"(*(const {T}*)(co_r + {t*st_t}))", f"(*(const {T}*)(co_i + {t*st_t}))",
                               f"*({T}*)(xo_r + {t*st_t})", f"*({T}*)(xo_i + {t*st_t})")
        else:
            o.append(f"      const long* oo = OUT_{name}[g];")
            for t in range(n2):
                emit_map_store(o, W, T, "      ", onames[t][0], onames[t][1],
                               f"(*(const {T}*)(cr + oo[{t}]))", f"(*(const {T}*)(ci + oo[{t}]))",
                               f"*({T}*)(pr + oo[{t}])", f"*({T}*)(pi + oo[{t}])")
        o.append(f"      bp_r += {n2}; bp_i += {n2};")
        o.append("    }")
        o.append("  }")
    else:
        # colT: transform results stored transposed into dr/di columns p = g*n2 + t
        if n2 == 8 and W == 8:
            o.append(f"  {{ const {T}* bp_r = Br; const {T}* bp_i = Bi; double* dp_r = dr; double* dp_i = di;")
            o.append(f"    for(int g=0; g<{n1}; ++g){{")
            in_exprs = [(f"bp_r[{t}]", f"bp_i[{t}]") for t in range(n2)]
            lines, onames = subdft_body(n2, W, in_exprs, "s2")
            o += lines
            regs_r = ", ".join(f"(__m512d){rn}" for rn, _ in onames)
            regs_i = ", ".join(f"(__m512d){iN}" for _, iN in onames)
            o.append(f"      tr8x8_store({regs_r}, dp_r, {P});")
            o.append(f"      tr8x8_store({regs_i}, dp_i, {P});")
            o.append(f"      bp_r += 8; bp_i += 8; dp_r += 8; dp_i += 8;")
            o.append("    }")
            o.append("  }")
        elif W == 1:
            o.append(f"  {{ const {T}* bp_r = Br; const {T}* bp_i = Bi; double* dp_r = dr; double* dp_i = di;")
            o.append(f"    for(int g=0; g<{n1}; ++g){{")
            in_exprs = [(f"bp_r[{t}]", f"bp_i[{t}]") for t in range(n2)]
            lines, onames = subdft_body(n2, W, in_exprs, "s2")
            o += lines
            for t in range(n2):
                o.append(f"      dp_r[{t}] = {onames[t][0]}; dp_i[{t}] = {onames[t][1]};")
            o.append(f"      bp_r += {n2}; bp_i += {n2}; dp_r += {n2}; dp_i += {n2};")
            o.append("    }")
            o.append("  }")
        else:
            # generic: write to O buffer then chunk-transpose
            o.append(f"  {T} Or[{L}] __attribute__((aligned(64)));")
            o.append(f"  {T} Oi[{L}] __attribute__((aligned(64)));")
            o.append(f"  {{ const {T}* bp_r = Br; const {T}* bp_i = Bi; {T}* op_r = Or; {T}* op_i = Oi;")
            o.append(f"    for(int g=0; g<{n1}; ++g){{")
            in_exprs = [(f"bp_r[{t}]", f"bp_i[{t}]") for t in range(n2)]
            lines, onames = subdft_body(n2, W, in_exprs, "s2")
            o += lines
            for t in range(n2):
                o.append(f"      op_r[{t}] = {onames[t][0]}; op_i[{t}] = {onames[t][1]};")
            o.append(f"      bp_r += {n2}; bp_i += {n2}; op_r += {n2}; op_i += {n2};")
            o.append("    }")
            o.append("  }")
            # chunk loop
            nfull = L // W
            o.append(f"  {{ double* dp_r = dr; double* dp_i = di; const {T}* op_r = Or; const {T}* op_i = Oi;")
            o.append(f"    for(int q=0; q<{nfull}; ++q){{")
            if W == 8:
                regs_r = ", ".join(f"(__m512d)op_r[{t}]" for t in range(8))
                regs_i = ", ".join(f"(__m512d)op_i[{t}]" for t in range(8))
                o.append(f"      tr8x8_store({regs_r}, dp_r, {P});")
                o.append(f"      tr8x8_store({regs_i}, dp_i, {P});")
            elif W == 4:
                regs_r = ", ".join(f"(__m256d)op_r[{t}]" for t in range(4))
                regs_i = ", ".join(f"(__m256d)op_i[{t}]" for t in range(4))
                o.append(f"      tr4x4_store({regs_r}, dp_r, {P});")
                o.append(f"      tr4x4_store({regs_i}, dp_i, {P});")
            o.append(f"      dp_r += {W}; dp_i += {W}; op_r += {W}; op_i += {W};")
            o.append("    }")
            rem = L - nfull*W
            if rem:
                if W == 8:
                    regs_r = ", ".join(f"(__m512d)op_r[{min(t, rem-1)}]" for t in range(8))
                    regs_i = ", ".join(f"(__m512d)op_i[{min(t, rem-1)}]" for t in range(8))
                    o.append(f"    tr8x8_store_part({regs_r}, dp_r, {P}, {rem});")
                    o.append(f"    tr8x8_store_part({regs_i}, dp_i, {P}, {rem});")
                elif W == 4:
                    for t in range(rem):
                        for lane in range(4):
                            o.append(f"    dp_r[{lane}*{P} + {t}] = ((const double*)&op_r[{t}])[{lane}];")
                            o.append(f"    dp_i[{lane}*{P} + {t}] = ((const double*)&op_i[{t}])[{lane}];")
                elif W == 1:
                    pass
            o.append("  }")
            if W == 1:
                pass
    o.append("}")
    return "\n".join(tabs) + "\n" + "\n".join(o)

print("genloopb ready")
