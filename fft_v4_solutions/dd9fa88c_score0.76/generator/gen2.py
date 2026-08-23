"""Generator v2: staged FFT structure with small leaf codelets through L1 scratch."""
import math
from math import gcd
import mpmath as mp
from gencore import G, wcs
from dftgen import best_plan, primroot, is_prime

mp.mp.dps = 60
SIZES = [6, 8, 13, 17, 23, 36, 45, 64]
RPAD = {6: 8, 8: 8, 13: 16, 17: 24, 23: 24, 36: 40, 45: 48, 64: 64}

def cf(v):
    v = float(v)
    if v == int(v) and abs(v) < 1e15: return f"{int(v)}.0"
    return v.hex()

# ---------------------------------------------------------------- emit core
def emit_block(g, stores, indent="  ", prefix="t", eager=None):
    """emit SSA statements for nodes reachable from stores; returns (lines, names).
       eager: list of (ptr, node) stores to emit immediately after node definition."""
    reach = set(); stack = [n for _, n in stores] + ([n for _, n in eager] if eager else [])
    while stack:
        i = stack.pop()
        if i in reach: continue
        reach.add(i); t = g.ops[i]
        if t[0] in ('+', '-', 'm2'): stack += [t[1], t[2]]
        elif t[0] in ('n', 'rsqrt', 'rcp'): stack.append(t[1])
        elif t[0] == '*': stack.append(t[2])
    emap = {}
    if eager:
        for p, n in eager: emap.setdefault(n, []).append(p)
    names = {}; lines = []; cnt = [0]
    def rnd(i):
        t = g.ops[i]
        if t[0] == 'k': return cf(t[1])
        return names[i]
    for i in sorted(reach):
        t = g.ops[i]; o = t[0]
        if o == 'k':
            if i in emap:
                for p in emap[i]: lines.append(f"{indent}VS({p}, VK({cf(t[1])}));")
            continue
        nm = f"{prefix}{cnt[0]}"; cnt[0] += 1; names[i] = nm
        if o == 'in': lines.append(f"{indent}vd {nm} = {t[1]};")
        elif o == '+': lines.append(f"{indent}vd {nm} = {rnd(t[1])} + {rnd(t[2])};")
        elif o == '-': lines.append(f"{indent}vd {nm} = {rnd(t[1])} - {rnd(t[2])};")
        elif o == 'n': lines.append(f"{indent}vd {nm} = -{rnd(t[1])};")
        elif o == '*': lines.append(f"{indent}vd {nm} = {cf(t[1])} * {rnd(t[2])};")
        elif o == 'm2': lines.append(f"{indent}vd {nm} = {rnd(t[1])} * {rnd(t[2])};")
        elif o == 'rsqrt': lines.append(f"{indent}vd {nm} = VRSQRT({rnd(t[1])});")
        elif o == 'rcp': lines.append(f"{indent}vd {nm} = VRCP({rnd(t[1])});")
        else: raise ValueError(o)
        if i in emap:
            for p in emap[i]: lines.append(f"{indent}VS({p}, {nm});")
    return lines, rnd

def build_map(g, zr, zi):
    r2b = g.add(g.mul2(zr, zr), g.add(g.mul2(zi, zi), g.k(1e-300)))
    u = g.rsqrt(r2b)
    h = g.mul(0.5, r2b)
    for _ in range(2):
        f = g.sub(g.k(1.5), g.mul2(g.mul2(h, u), u))
        u = g.mul2(u, f)
    t = g.add(g.mul2(r2b, u), g.k(1.0))
    y = g.rcp(t)
    for _ in range(2):
        e = g.sub(g.k(1.0), g.mul2(t, y))
        y = g.add(g.mul2(y, e), y)
    return g.mul2(zr, y), g.mul2(zi, y)

# ---------------------------------------------------------------- leaf emitters
LEAFS = {}   # name -> C text

def leaf(name):
    def deco(fn):
        return fn
    return deco

def make_leaf(n, name, *, tw_consts=None, in_tab=False, out_tab=False, tail="store",
              in_name="ir", out_name="or_"):
    """Emit leaf: n-point dft.
       tw_consts: list of (c,s) per k (applied after dft), or None.
       in_tab/out_tab: index tables (const int arrays named {name}_II / {name}_OO baked).
       tail: 'store' | 'map' (adds c loads, map, snap).
    """
    if name in LEAFS: return name
    plan, _, _ = best_plan(n)
    g = G()
    if in_tab:
        xs = [(g.inp(f"VL(ir + II[{j}]*is_)"), g.inp(f"VL(ii + II[{j}]*is_)")) for j in range(n)]
    else:
        xs = [(g.inp(f"VL(ir + {j}*is_)"), g.inp(f"VL(ii + {j}*is_)")) for j in range(n)]
    ys = plan(g, xs)
    if tw_consts is not None:
        out = []
        for k in range(n):
            c, s = tw_consts[k]
            out.append(g.cmulk(c, s, ys[k]))
        ys = out
    stores = []
    lines_extra_pre = []
    if tail == "store":
        for k in range(n):
            o = f"OO[{k}]" if out_tab else f"{k}"
            stores.append((f"or_ + {o}*os", ys[k][0]))
            stores.append((f"oi + {o}*os", ys[k][1]))
        sig = "const double *restrict ir, const double *restrict ii, const long is_, double *restrict or_, double *restrict oi, const long os"
        if in_tab: sig += ", const int *restrict II"
        if out_tab: sig += ", const int *restrict OO"
        body, _ = emit_block(g, stores)
        txt = [f"static inline __attribute__((always_inline)) void {name}({sig}){{"]
        txt += body
        for ptr, nd in stores:
            tname = _rname(g, nd, body)
        # simpler: redo emit with store emission inline
        txt = [f"static inline __attribute__((always_inline)) void {name}({sig}){{"]
        lines, rnd = emit_block(g, stores)
        txt += lines
        for ptr, nd in stores:
            txt.append(f"  VS({ptr}, {rnd(nd)});")
        txt.append("}")
        LEAFS[name] = "\n".join(txt)
        return name
    elif tail == "map":
        mapped = []
        for k in range(n):
            o = f"OO[{k}]" if out_tab else f"{k}"
            zr = g.add(ys[k][0], g.inp(f"VL(cre + {o}*os)"))
            zi = g.add(ys[k][1], g.inp(f"VL(cim + {o}*os)"))
            mapped.append(build_map(g, zr, zi))
        stores = []
        snapst = []
        for k in range(n):
            o = f"OO[{k}]" if out_tab else f"{k}"
            stores.append((f"or_ + {o}*os", mapped[k][0]))
            stores.append((f"oi + {o}*os", mapped[k][1]))
            snapst.append((f"sre + {o}*os", mapped[k][0]))
            snapst.append((f"sim + {o}*os", mapped[k][1]))
        sig = ("const double *restrict ir, const double *restrict ii, const long is_, "
               "double *restrict or_, double *restrict oi, const long os, "
               "const double *restrict cre, const double *restrict cim, "
               "double *restrict sre, double *restrict sim")
        if in_tab: sig += ", const int *restrict II"
        if out_tab: sig += ", const int *restrict OO"
        lines, rnd = emit_block(g, stores + snapst)
        txt = [f"static inline __attribute__((always_inline)) void {name}({sig}){{"]
        txt += lines
        txt.append("  if(__builtin_expect(sre != 0, 0)){")
        for ptr, nd in snapst:
            txt.append(f"    VS({ptr}, {rnd(nd)});")
        txt.append("  }")
        for ptr, nd in stores:
            txt.append(f"  VS({ptr}, {rnd(nd)});")
        txt.append("}")
        LEAFS[name] = "\n".join(txt)
        return name
    raise ValueError(tail)

def _rname(g, nd, body): return None  # placeholder (unused)

# ---------------------------------------------------------------- staged structures
# A staged fft produces C code for: void NAME(double*re, double*im, long s [, c/snap])
# passing data through scratch arrays SR1/SI1 (+SR2/SI2).
# Variants: kind in {'p','m','f'} : plain / fft+map / fft+map+fft.

CONSTS = []   # (name, list_of_ints) int tables

def itab(name, vals):
    CONSTS.append((name, vals))
    return name

class StagedCT:
    """L = n1*n2 Cooley-Tukey. pass1 j1-loop peeled w/ const twiddles, pass2 loop."""
    def __init__(self, L, n1, n2):
        self.L, self.n1, self.n2 = L, n1, n2
    def gen_core(self, L, pfx, src, sstride, dst_expr):
        """emits pass1+pass2 where pass2's final store handled by caller-provided leaf call template"""
        raise NotImplementedError

def gen_fft_fn(L, struct, kind, fname):
    """struct: ('line',) | ('ct',n1,n2) | ('pfa',n1,n2) | ('dsym',kb) | ('rader',)"""
    # returns (C text, leaf names used)
    if struct[0] == 'line':
        return gen_line_fn(L, kind, fname)
    if struct[0] == 'ct':
        return gen_ct_fn(L, struct[1], struct[2], kind, fname)
    if struct[0] == 'pfa':
        return gen_pfa_fn(L, struct[1], struct[2], kind, fname)
    if struct[0] == 'dsym':
        return gen_dsym_fn(L, struct[1], kind, fname)
    if struct[0] == 'dsymr':
        return gen_dsymr_fn(L, kind, fname)
    if struct[0] == 'rader':
        return gen_rader_fn(L, kind, fname)
    raise ValueError(struct)

SIG_P = "(double *restrict re, double *restrict im, const long s)"
SIG_M = ("(double *restrict re, double *restrict im, const long s, "
         "const double *restrict cre, const double *restrict cim, "
         "double *restrict sre, double *restrict sim)")

INLINE_L = {6, 8}
def fn_head(kind, fname):
    L = int(''.join(ch for ch in fname.split('_')[0] if ch.isdigit()))
    attr = "static inline __attribute__((always_inline))" if L in INLINE_L else "static"
    return f"{attr} void {fname}{SIG_P if kind=='p' else SIG_M}{{"

# ---- straight-line whole-L (small L) ----
def gen_line_fn(L, kind, fname):
    plan, _, _ = best_plan(L)
    g = G()
    xs = [(g.inp(f"VL(re + {j}*s)"), g.inp(f"VL(im + {j}*s)")) for j in range(L)]
    ys = plan(g, xs)
    if kind in ('m', 'f'):
        mapped = []
        for j in range(L):
            zr = g.add(ys[j][0], g.inp(f"VL(cre + {j}*s)"))
            zi = g.add(ys[j][1], g.inp(f"VL(cim + {j}*s)"))
            mapped.append(build_map(g, zr, zi))
        snapst = []
        for j in range(L):
            snapst.append((f"sre + {j}*s", mapped[j][0]))
            snapst.append((f"sim + {j}*s", mapped[j][1]))
        final = plan(g, mapped) if kind == 'f' else mapped
    else:
        snapst = []
        final = ys
    stores = []
    for j in range(L):
        stores.append((f"re + {j}*s", final[j][0]))
        stores.append((f"im + {j}*s", final[j][1]))
    lines, rnd = emit_block(g, stores + snapst)
    txt = [fn_head(kind, fname)]
    txt += lines
    if snapst:
        txt.append("  if(__builtin_expect(sre != 0, 0)){")
        for ptr, nd in snapst:
            txt.append(f"    VS({ptr}, {rnd(nd)});")
        txt.append("  }")
    for ptr, nd in stores:
        txt.append(f"  VS({ptr}, {rnd(nd)});")
    txt.append("}")
    return "\n".join(txt)


def make_leaf_ctf(n, name, Lfull):
    """pass2+map+pass1' fused leaf for ct(n,n): in: scratch rows stride is_;
       computes dft_n -> X[k1]; z = X + c[k1*cs]; m = map(z) (snap at sre + k1*cs);
       then dft_n over k1 with VARIABLE twiddles w[t] (vd rows), outputs stored or_ + t*os."""
    if name in LEAFS: return name
    plan, _, _ = best_plan(n)
    g = G()
    xs = [(g.inp(f"VL(ir + {j}*is_)"), g.inp(f"VL(ii + {j}*is_)")) for j in range(n)]
    ys = plan(g, xs)
    mapped = []
    snapst = []
    for k in range(n):
        zr = g.add(ys[k][0], g.inp(f"VL(cre + {k}*cs)"))
        zi = g.add(ys[k][1], g.inp(f"VL(cim + {k}*cs)"))
        mr, mi = build_map(g, zr, zi)
        mapped.append((mr, mi))
        snapst += [(f"sre + {k}*cs", mr), (f"sim + {k}*cs", mi)]
    ys2 = plan(g, mapped)
    out = [ys2[0]]
    for t in range(1, n):
        wr = g.inp(f"wr[{t-1}]"); wi = g.inp(f"wi[{t-1}]")
        zr, zi = ys2[t]
        out.append((g.sub(g.mul2(wr, zr), g.mul2(wi, zi)), g.add(g.mul2(wr, zi), g.mul2(wi, zr))))
    stores = []
    for t in range(n):
        stores.append((f"or_ + {t}*os", out[t][0]))
        stores.append((f"oi + {t}*os", out[t][1]))
    sig = ("const double *restrict ir, const double *restrict ii, const long is_, "
           "double *restrict or_, double *restrict oi, const long os, "
           "const double *restrict cre, const double *restrict cim, const long cs, "
           "double *restrict sre, double *restrict sim, "
           "const vd *restrict wr, const vd *restrict wi")
    lines, rnd = emit_block(g, stores + snapst)
    txt = [f"static inline __attribute__((always_inline)) void {name}({sig}){{"]
    txt += lines
    txt.append("  if(__builtin_expect(sre != 0, 0)){")
    for ptr, nd in snapst:
        txt.append(f"    VS({ptr}, {rnd(nd)});")
    txt.append("  }")
    for ptr, nd in stores:
        txt.append(f"  VS({ptr}, {rnd(nd)});")
    txt.append("}")
    LEAFS[name] = "\n".join(txt)
    return name

def make_leaf_ctf0(n, name):
    """same but j1'=0: no twiddles"""
    if name in LEAFS: return name
    plan, _, _ = best_plan(n)
    g = G()
    xs = [(g.inp(f"VL(ir + {j}*is_)"), g.inp(f"VL(ii + {j}*is_)")) for j in range(n)]
    ys = plan(g, xs)
    mapped = []; snapst = []
    for k in range(n):
        zr = g.add(ys[k][0], g.inp(f"VL(cre + {k}*cs)"))
        zi = g.add(ys[k][1], g.inp(f"VL(cim + {k}*cs)"))
        mr, mi = build_map(g, zr, zi)
        mapped.append((mr, mi))
        snapst += [(f"sre + {k}*cs", mr), (f"sim + {k}*cs", mi)]
    ys2 = plan(g, mapped)
    stores = []
    for t in range(n):
        stores.append((f"or_ + {t}*os", ys2[t][0]))
        stores.append((f"oi + {t}*os", ys2[t][1]))
    sig = ("const double *restrict ir, const double *restrict ii, const long is_, "
           "double *restrict or_, double *restrict oi, const long os, "
           "const double *restrict cre, const double *restrict cim, const long cs, "
           "double *restrict sre, double *restrict sim")
    lines, rnd = emit_block(g, stores + snapst)
    txt = [f"static inline __attribute__((always_inline)) void {name}({sig}){{"]
    txt += lines
    txt.append("  if(__builtin_expect(sre != 0, 0)){")
    for ptr, nd in snapst:
        txt.append(f"    VS({ptr}, {rnd(nd)});")
    txt.append("  }")
    for ptr, nd in stores:
        txt.append(f"  VS({ptr}, {rnd(nd)});")
    txt.append("}")
    LEAFS[name] = "\n".join(txt)
    return name


def gen_dsymr_fn(L, kind, fname):
    """register-resident symmetric DFT: single straight-line, j-major accumulation."""
    h = (L - 1) // 2
    body = [fn_head(kind, fname)]
    g = G()
    xs = [(g.inp(f"VL(re + {j}*s)"), g.inp(f"VL(im + {j}*s)")) for j in range(L)]
    x0 = xs[0]
    sp = {}; sm = {}
    for j in range(1, h + 1):
        sp[j] = g.cadd(xs[j], xs[L - j]); sm[j] = g.csub(xs[j], xs[L - j])
    X0 = g.csum_tree([xs[0]] + [sp[j] for j in range(1, h + 1)])
    acc = {k: [None, None, None, None] for k in range(1, h + 1)}
    for j in range(1, h + 1):
        for k in range(1, h + 1):
            c, ss = wcs(j * k, L)
            cc, sn_ = c, -ss
            t = acc[k]
            ar = g.mul(cc, sp[j][0]); ai = g.mul(cc, sp[j][1])
            br = g.mul(sn_, sm[j][0]); bi = g.mul(sn_, sm[j][1])
            t[0] = ar if t[0] is None else g.add(t[0], ar)
            t[1] = ai if t[1] is None else g.add(t[1], ai)
            t[2] = br if t[2] is None else g.add(t[2], br)
            t[3] = bi if t[3] is None else g.add(t[3], bi)
    pairs = [X0]; idxs = [0]
    for k in range(1, h + 1):
        Ar, Ai, Br, Bi = acc[k]
        pairs.append((g.add(g.add(x0[0], Ar), Bi), g.sub(g.add(x0[1], Ai), Br))); idxs.append(k)
        pairs.append((g.sub(g.add(x0[0], Ar), Bi), g.add(g.add(x0[1], Ai), Br))); idxs.append(L - k)
    stores = []; snapst = []
    if kind == 'p':
        for (zr, zi), kk in zip(pairs, idxs):
            stores += [(f"re + {kk}*s", zr), (f"im + {kk}*s", zi)]
        final2 = None
    else:
        mapped = []
        for (zr0, zi0), kk in zip(pairs, idxs):
            zr = g.add(zr0, g.inp(f"VL(cre + {kk}*s)"))
            zi = g.add(zi0, g.inp(f"VL(cim + {kk}*s)"))
            mr, mi = build_map(g, zr, zi)
            mapped.append(((mr, mi), kk))
            snapst += [(f"sre + {kk}*s", mr), (f"sim + {kk}*s", mi)]
        if kind == 'm':
            for (mr, mi), kk in mapped:
                stores += [(f"re + {kk}*s", mr), (f"im + {kk}*s", mi)]
        else:  # f: second dft on mapped values (in natural order)
            mord = [None] * L
            for (mr, mi), kk in mapped: mord[kk] = (mr, mi)
            x0b = mord[0]
            spb = {}; smb = {}
            for j in range(1, h + 1):
                spb[j] = g.cadd(mord[j], mord[L - j]); smb[j] = g.csub(mord[j], mord[L - j])
            X0b = g.csum_tree([mord[0]] + [spb[j] for j in range(1, h + 1)])
            accb = {k: [None, None, None, None] for k in range(1, h + 1)}
            for j in range(1, h + 1):
                for k in range(1, h + 1):
                    c, ss = wcs(j * k, L)
                    cc, sn_ = c, -ss
                    t = accb[k]
                    ar = g.mul(cc, spb[j][0]); ai = g.mul(cc, spb[j][1])
                    br = g.mul(sn_, smb[j][0]); bi = g.mul(sn_, smb[j][1])
                    t[0] = ar if t[0] is None else g.add(t[0], ar)
                    t[1] = ai if t[1] is None else g.add(t[1], ai)
                    t[2] = br if t[2] is None else g.add(t[2], br)
                    t[3] = bi if t[3] is None else g.add(t[3], bi)
            stores += [(f"re + 0*s", X0b[0]), (f"im + 0*s", X0b[1])]
            for k in range(1, h + 1):
                Ar, Ai, Br, Bi = accb[k]
                stores += [(f"re + {k}*s", g.add(g.add(x0b[0], Ar), Bi)),
                           (f"im + {k}*s", g.sub(g.add(x0b[1], Ai), Br)),
                           (f"re + {L-k}*s", g.sub(g.add(x0b[0], Ar), Bi)),
                           (f"im + {L-k}*s", g.add(g.add(x0b[1], Ai), Br))]
    lines, rnd = emit_block(g, stores + snapst)
    body += lines
    if snapst:
        body.append("  if(__builtin_expect(sre != 0, 0)){")
        for p, nd in snapst: body.append(f"    VS({p}, {rnd(nd)});")
        body.append("  }")
    for p, nd in stores:
        body.append(f"  VS({p}, {rnd(nd)});")
    body.append("}")
    return "\n".join(body)

# ---- CT (for 64 = 8 x 8) ----
def gen_ct_fn(L, n1, n2, kind, fname):
    assert n1 * n2 == L
    # leaves
    lv_nt = make_leaf(n2, f"dft{n2}_nt")
    lv_tw = []
    for j1 in range(1, n1):
        tws = [wcs(j1 * k2, L) for k2 in range(n2)]
        lv_tw.append(make_leaf(n2, f"dft{n2}_twL{L}_{j1}", tw_consts=tws))
    lv_out_store = make_leaf(n1, f"dft{n1}_nt")
    lv_out_map = make_leaf(n1, f"dft{n1}_map", tail="map")
    body = [fn_head(kind, fname)]
    def pass1(src_r, src_i, sstr, dst_r, dst_i):
        # inner dfts over j2 for each j1; input x[(n1*j2 + j1)*stride]; out scratch[(j1 + n1*k2)*8]
        body.append(f"  dft{n2}_nt({src_r} + 0, {src_i} + 0, {n1}*{sstr}, {dst_r}, {dst_i}, {n1}*8);")
        for j1 in range(1, n1):
            body.append(f"  dft{n2}_twL{L}_{j1}({src_r} + {j1}*{sstr}, {src_i} + {j1}*{sstr}, {n1}*{sstr}, {dst_r} + {j1}*8, {dst_i} + {j1}*8, {n1}*8);")
    def pass2(src_r, src_i, dst_r, dst_i, dstr, mapped, last):
        # outer dfts over j1 for each k2: scratch[(j1 + n1*k2)*8] stride 8; out X[(n2*k1 + k2)*dstr]
        body.append(f"  for(int k2 = 0; k2 < {n2}; k2++){{")
        if mapped:
            body.append(f"    dft{n1}_map({src_r} + k2*{n1}*8, {src_i} + k2*{n1}*8, 8, {dst_r} + k2*{dstr}, {dst_i} + k2*{dstr}, {n2}*{dstr}, cre + k2*{dstr}, cim + k2*{dstr}, sre?sre + k2*{dstr}:0, sim?sim + k2*{dstr}:0);")
        else:
            body.append(f"    dft{n1}_nt({src_r} + k2*{n1}*8, {src_i} + k2*{n1}*8, 8, {dst_r} + k2*{dstr}, {dst_i} + k2*{dstr}, {n2}*{dstr});")
        body.append("  }")
    if kind == 'p':
        pass1("re", "im", "s", "SR1", "SI1")
        pass2("SR1", "SI1", "re", "im", "s", False, True)
    elif kind == 'm':
        pass1("re", "im", "s", "SR1", "SI1")
        body.append(f"  for(int k2 = 0; k2 < {n2}; k2++){{")
        body.append(f"    dft{n1}_map(SR1 + k2*{n1}*8, SI1 + k2*{n1}*8, 8, re + k2*s, im + k2*s, {n2}*s, cre + k2*s, cim + k2*s, sre?sre + k2*s:0, sim?sim + k2*s:0);")
        body.append("  }")
    else:
        lvf = make_leaf_ctf(n1, f"dftf{n1}_L{L}", L)
        lvf0 = make_leaf_ctf0(n1, f"dftf{n1}0_L{L}")
        pass1("re", "im", "s", "SR1", "SI1")
        body.append(f"  dftf{n1}0_L{L}(SR1, SI1, 8, SR2, SI2, {n1}*8, cre, cim, {n2}*s, sre, sim);")
        body.append(f"  for(int k2 = 1; k2 < {n2}; k2++){{")
        body.append(f"    dftf{n1}_L{L}(SR1 + k2*{n1}*8, SI1 + k2*{n1}*8, 8, SR2 + k2*8, SI2 + k2*8, {n1}*8, cre + k2*s, cim + k2*s, {n2}*s, sre?sre + k2*s:0, sim?sim + k2*s:0, TWR_{L} + (k2-1)*{n1-1}, TWI_{L} + (k2-1)*{n1-1});")
        body.append("  }")
        pass2("SR2", "SI2", "re", "im", "s", False, True)
    body.append("}")
    return "\n".join(body)

def make_leaf_map2(n, name):
    """dft n + map; input (ir,is_); output dense (or_, os); c and snap at (cre + k*cs*cstr)."""
    if name in LEAFS: return name
    plan, _, _ = best_plan(n)
    g = G()
    xs = [(g.inp(f"VL(ir + {j}*is_)"), g.inp(f"VL(ii + {j}*is_)")) for j in range(n)]
    ys = plan(g, xs)
    mapped = []
    for k in range(n):
        zr = g.add(ys[k][0], g.inp(f"VL(cre + {k}*cn*cs)"))
        zi = g.add(ys[k][1], g.inp(f"VL(cim + {k}*cn*cs)"))
        mapped.append(build_map(g, zr, zi))
    stores = []; snapst = []
    for k in range(n):
        stores.append((f"or_ + {k}*os", mapped[k][0]))
        stores.append((f"oi + {k}*os", mapped[k][1]))
        snapst.append((f"sre + {k}*cn*cs", mapped[k][0]))
        snapst.append((f"sim + {k}*cn*cs", mapped[k][1]))
    sig = ("const double *restrict ir, const double *restrict ii, const long is_, "
           "double *restrict or_, double *restrict oi, const long os, "
           "const double *restrict cre, const double *restrict cim, const long cn, const long cs, "
           "double *restrict sre, double *restrict sim")
    lines, rnd = emit_block(g, stores + snapst)
    txt = [f"static inline __attribute__((always_inline)) void {name}({sig}){{"]
    txt += lines
    txt.append("  if(__builtin_expect(sre != 0, 0)){")
    for ptr, nd in snapst:
        txt.append(f"    VS({ptr}, {rnd(nd)});")
    txt.append("  }")
    for ptr, nd in stores:
        txt.append(f"  VS({ptr}, {rnd(nd)});")
    txt.append("}")
    LEAFS[name] = "\n".join(txt)
    return name

# ---- PFA (36 = 4x9, 45 = 9x5) ----
def gen_pfa_fn(L, n1, n2, kind, fname):
    assert n1 * n2 == L and gcd(n1, n2) == 1
    inv1 = pow(n1, -1, n2); inv2 = pow(n2, -1, n1)
    # input: for j2 fixed: x[(n2*j1 + n1*j2) % L]  -> dft_n1 -> scratch[(j1*n2 + j2)*8]
    # pass2: for k1 fixed: scratch[(k1*n2 + j2)*8] (contiguous) -> dft_n2 -> X[(k1*n2*inv2 + k2*n1*inv1)%L]
    iin = [[(n2 * j1 + n1 * j2) % L for j1 in range(n1)] for j2 in range(n2)]
    oout = [[(k1 * n2 * inv2 + k2 * n1 * inv1) % L for k2 in range(n2)] for k1 in range(n1)]
    # tables flattened; one II per j2 -> table II_{L} of n2*n1; one OO per k1 -> OO_{L} n1*n2
    iit = itab(f"PFA_II_{L}", [v for row in iin for v in row])
    oot = itab(f"PFA_OO_{L}", [v for row in oout for v in row])
    lv1 = make_leaf(n1, f"dft{n1}_it", in_tab=True)
    lv2s = make_leaf(n2, f"dft{n2}_ot", out_tab=True)
    lv2m = make_leaf(n2, f"dft{n2}_ot_map", out_tab=True, tail="map")
    body = [fn_head(kind, fname)]
    def pass1(src_r, src_i, sstr, dst_r, dst_i):
        body.append(f"  for(int j2 = 0; j2 < {n2}; j2++)")
        body.append(f"    dft{n1}_it({src_r}, {src_i}, {sstr}, {dst_r} + j2*8, {dst_i} + j2*8, {n2}*8, {iit} + j2*{n1});")
    def pass2(src_r, src_i, dst_r, dst_i, dstr, mapped):
        body.append(f"  for(int k1 = 0; k1 < {n1}; k1++)")
        if mapped:
            body.append(f"    dft{n2}_ot_map({src_r} + k1*{n2}*8, {src_i} + k1*{n2}*8, 8, {dst_r}, {dst_i}, {dstr}, cre, cim, sre, sim, {oot} + k1*{n2});")
        else:
            body.append(f"    dft{n2}_ot({src_r} + k1*{n2}*8, {src_i} + k1*{n2}*8, 8, {dst_r}, {dst_i}, {dstr}, {oot} + k1*{n2});")
    if kind == 'p':
        pass1("re", "im", "s", "SR1", "SI1")
        pass2("SR1", "SI1", "re", "im", "s", False)
    elif kind == 'm':
        pass1("re", "im", "s", "SR1", "SI1")
        pass2("SR1", "SI1", "re", "im", "s", True)
    else:
        lv2m2 = make_leaf_map2_ot(n2, f"dft{n2}_ot_map2")
        pass1("re", "im", "s", "SR1", "SI1")
        body.append(f"  for(int k1 = 0; k1 < {n1}; k1++)")
        body.append(f"    dft{n2}_ot_map2(SR1 + k1*{n2}*8, SI1 + k1*{n2}*8, 8, SR2, SI2, 8, cre, cim, s, sre, sim, {oot} + k1*{n2});")
        pass1("SR2", "SI2", "8", "SR1", "SI1")
        pass2("SR1", "SI1", "re", "im", "s", False)
    body.append("}")
    return "\n".join(body)

def make_leaf_map2_ot(n, name):
    """dft n + map, output-table addressing used for BOTH scratch-out and c/snap:
       out to or_ + OO[k]*os (dense scratch via os=8), c at cre + OO[k]*cs."""
    if name in LEAFS: return name
    plan, _, _ = best_plan(n)
    g = G()
    xs = [(g.inp(f"VL(ir + {j}*is_)"), g.inp(f"VL(ii + {j}*is_)")) for j in range(n)]
    ys = plan(g, xs)
    mapped = []
    for k in range(n):
        zr = g.add(ys[k][0], g.inp(f"VL(cre + OO[{k}]*cs)"))
        zi = g.add(ys[k][1], g.inp(f"VL(cim + OO[{k}]*cs)"))
        mapped.append(build_map(g, zr, zi))
    stores = []; snapst = []
    for k in range(n):
        stores.append((f"or_ + OO[{k}]*os", mapped[k][0]))
        stores.append((f"oi + OO[{k}]*os", mapped[k][1]))
        snapst.append((f"sre + OO[{k}]*cs", mapped[k][0]))
        snapst.append((f"sim + OO[{k}]*cs", mapped[k][1]))
    sig = ("const double *restrict ir, const double *restrict ii, const long is_, "
           "double *restrict or_, double *restrict oi, const long os, "
           "const double *restrict cre, const double *restrict cim, const long cs, "
           "double *restrict sre, double *restrict sim, const int *restrict OO")
    lines, rnd = emit_block(g, stores + snapst)
    txt = [f"static inline __attribute__((always_inline)) void {name}({sig}){{"]
    txt += lines
    txt.append("  if(__builtin_expect(sre != 0, 0)){")
    for ptr, nd in snapst:
        txt.append(f"    VS({ptr}, {rnd(nd)});")
    txt.append("  }")
    for ptr, nd in stores:
        txt.append(f"  VS({ptr}, {rnd(nd)});")
    txt.append("}")
    LEAFS[name] = "\n".join(txt)
    return name

# ---- dsym staged (13, 23, maybe 17) ----
def gen_dsym_fn(L, kb, kind, fname):
    """symmetric direct; kb = k's per block; blocks emitted as noinline helpers."""
    h = (L - 1) // 2
    helpers = []
    body = [fn_head(kind, fname)]
    body.insert(1, "  const vd x0r = VL(re + 0*s), x0i = VL(im + 0*s);")
    # stage 0: sp/sm into scratch (eager stores), X0 out
    g = G()
    xs = [(g.inp(f"VL(re + {j}*s)"), g.inp(f"VL(im + {j}*s)")) for j in range(L)]
    sp = {}; sm = {}
    st0 = []
    for j in range(1, h + 1):
        sp[j] = g.cadd(xs[j], xs[L - j]); sm[j] = g.csub(xs[j], xs[L - j])
        st0 += [(f"SR1 + {4*(j-1)+0}*8", sp[j][0]), (f"SI1 + {4*(j-1)+0}*8", sp[j][1]),
                (f"SR1 + {4*(j-1)+1}*8", sm[j][0]), (f"SI1 + {4*(j-1)+1}*8", sm[j][1])]
    X0 = g.csum_tree([xs[0]] + [sp[j] for j in range(1, h + 1)])

    def finish_out(g, pairs, idxs):
        stores = []; snapst = []
        for (zr0, zi0), kk in zip(pairs, idxs):
            if kind in ('m', 'f'):
                zr = g.add(zr0, g.inp(f"VL(cre + {kk}*s)"))
                zi = g.add(zi0, g.inp(f"VL(cim + {kk}*s)"))
                mr, mi = build_map(g, zr, zi)
                snapst += [(f"sre + {kk}*s", mr), (f"sim + {kk}*s", mi)]
            else:
                mr, mi = zr0, zi0
            if kind == 'f':
                stores += [(f"SR2 + {kk}*8", mr), (f"SI2 + {kk}*8", mi)]
            else:
                stores += [(f"re + {kk}*s", mr), (f"im + {kk}*s", mi)]
        return stores, snapst

    s0, sn0 = finish_out(g, [X0], [0])
    lines, rnd = emit_block(g, s0 + sn0, prefix="a", eager=st0)
    body += lines
    if sn0:
        body.append("  if(__builtin_expect(sre != 0, 0)){")
        for p, nd in sn0: body.append(f"    VS({p}, {rnd(nd)});")
        body.append("  }")
    for p, nd in s0:
        body.append(f"  VS({p}, {rnd(nd)});")

    # block helpers
    hsig = ("const vd x0r, const vd x0i, double *restrict re, double *restrict im, const long s"
            + (", const double *restrict cre, const double *restrict cim, double *restrict sre, double *restrict sim"
               if kind in ('m', 'f') else ""))
    kl = list(range(1, h + 1))
    bn = 0
    for b0 in range(0, h, kb):
        ks = kl[b0:b0 + kb]
        g = G()
        x0 = (g.inp("x0r"), g.inp("x0i"))
        spn = {}; smn = {}
        for j in range(1, h + 1):
            spn[j] = (g.inp(f"VL(SR1x + {4*(j-1)+0}*8)"), g.inp(f"VL(SI1x + {4*(j-1)+0}*8)"))
            smn[j] = (g.inp(f"VL(SR1x + {4*(j-1)+1}*8)"), g.inp(f"VL(SI1x + {4*(j-1)+1}*8)"))
        pairs = []; idxs = []
        acc = {k: [None, None, None, None] for k in ks}
        for j in range(1, h + 1):
            for k in ks:
                c, ss = wcs(j * k, L)
                cc, sn_ = c, -ss
                t = acc[k]
                ar = g.mul(cc, spn[j][0]); ai = g.mul(cc, spn[j][1])
                br = g.mul(sn_, smn[j][0]); bi_ = g.mul(sn_, smn[j][1])
                t[0] = ar if t[0] is None else g.add(t[0], ar)
                t[1] = ai if t[1] is None else g.add(t[1], ai)
                t[2] = br if t[2] is None else g.add(t[2], br)
                t[3] = bi_ if t[3] is None else g.add(t[3], bi_)
        for k in ks:
            Ar, Ai, Br, Bi = acc[k]
            pairs.append((g.add(g.add(x0[0], Ar), Bi), g.sub(g.add(x0[1], Ai), Br))); idxs.append(k)
            pairs.append((g.sub(g.add(x0[0], Ar), Bi), g.add(g.add(x0[1], Ai), Br))); idxs.append(L - k)
        sk, snk = finish_out(g, pairs, idxs)
        if kind == 'p':
            lines, rnd = emit_block(g, sk + snk, prefix=f"b{bn}_")
            body.append("  {")
            body.append("  const double *restrict SR1x = SR1; const double *restrict SI1x = SI1;")
            body += ["  " + l for l in lines]
            for p, nd in sk:
                body.append(f"    VS({p}, {rnd(nd)});")
            body.append("  }")
        else:
            lines, rnd = emit_block(g, sk + snk, prefix="t")
            htxt = [f"static __attribute__((noinline)) void {fname}_blk{bn}({hsig}){{"]
            htxt.append("  const double *restrict SR1x = SR1; const double *restrict SI1x = SI1;")
            htxt += lines
            if snk:
                htxt.append("  if(__builtin_expect(sre != 0, 0)){")
                for p, nd in snk: htxt.append(f"    VS({p}, {rnd(nd)});")
                htxt.append("  }")
            for p, nd in sk:
                htxt.append(f"  VS({p}, {rnd(nd)});")
            htxt.append("}")
            helpers.append("\n".join(htxt))
            args = "x0r, x0i, re, im, s, cre, cim, sre, sim"
            body.append(f"  {fname}_blk{bn}({args});")
        bn += 1
    if kind == 'f':
        body.append(f"  {fname}_tail(re, im, s);")
    body.append("}")
    txt = "\n\n".join(helpers) + "\n\n" + "\n".join(body)
    if kind == 'f':
        tail = gen_dsym_tail(L, kb, f"{fname}_tail")
        txt = tail + "\n\n" + txt
    return txt

def gen_dsym_tail(L, kb, fname):
    h = (L - 1) // 2
    body = [f"static void {fname}(double *restrict re, double *restrict im, const long s){{"]
    g = G()
    xs = [(g.inp(f"VL(SR2 + {j}*8)"), g.inp(f"VL(SI2 + {j}*8)")) for j in range(L)]
    sp = {}; sm = {}
    st0 = []
    for j in range(1, h + 1):
        sp[j] = g.cadd(xs[j], xs[L - j]); sm[j] = g.csub(xs[j], xs[L - j])
        st0 += [(f"SR1 + {4*(j-1)+0}*8", sp[j][0]), (f"SI1 + {4*(j-1)+0}*8", sp[j][1]),
                (f"SR1 + {4*(j-1)+1}*8", sm[j][0]), (f"SI1 + {4*(j-1)+1}*8", sm[j][1])]
    X0 = g.csum_tree([xs[0]] + [sp[j] for j in range(1, h + 1)])
    st0 += [(f"re + 0*s", X0[0]), (f"im + 0*s", X0[1])]
    lines, rnd = emit_block(g, [], prefix="a", eager=st0)
    body += lines
    bi = 0
    kl = list(range(1, h + 1))
    for b0 in range(0, h, kb):
        ks = kl[b0:b0 + kb]
        g = G()
        x0 = (g.inp("VL(SR2 + 0*8)"), g.inp("VL(SI2 + 0*8)"))
        spn = {}; smn = {}
        for j in range(1, h + 1):
            spn[j] = (g.inp(f"VL(SR1 + {4*(j-1)+0}*8)"), g.inp(f"VL(SI1 + {4*(j-1)+0}*8)"))
            smn[j] = (g.inp(f"VL(SR1 + {4*(j-1)+1}*8)"), g.inp(f"VL(SI1 + {4*(j-1)+1}*8)"))
        st = []
        acc = {k: [None, None, None, None] for k in ks}
        for j in range(1, h + 1):
            for k in ks:
                c, ss = wcs(j * k, L)
                cc, sn_ = c, -ss
                t = acc[k]
                ar = g.mul(cc, spn[j][0]); ai = g.mul(cc, spn[j][1])
                br = g.mul(sn_, smn[j][0]); bi = g.mul(sn_, smn[j][1])
                t[0] = ar if t[0] is None else g.add(t[0], ar)
                t[1] = ai if t[1] is None else g.add(t[1], ai)
                t[2] = br if t[2] is None else g.add(t[2], br)
                t[3] = bi if t[3] is None else g.add(t[3], bi)
        for k in ks:
            Ar, Ai, Br, Bi = acc[k]
            st += [(f"re + {k}*s", g.add(g.add(x0[0], Ar), Bi)),
                   (f"im + {k}*s", g.sub(g.add(x0[1], Ai), Br)),
                   (f"re + {L-k}*s", g.sub(g.add(x0[0], Ar), Bi)),
                   (f"im + {L-k}*s", g.add(g.add(x0[1], Ai), Br))]
        lines, rnd = emit_block(g, st, prefix=f"c{bi}_")
        body.append("  {")
        body += ["  " + l for l in lines]
        for p, nd in st: body.append(f"    VS({p}, {rnd(nd)});")
        body.append("  }")
        bi += 1
    body.append("}")
    return "\n".join(body)
