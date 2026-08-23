RCPMAGIC_I = False
MAGICHEX = '0x1.e6238da3c2118p+1022'
MAPF_COMPOSITE = False
import numpy as np
from genlib import *

def emit_map(e, zr, zi, pfx=""):
    """z/(1+|z|) via rsqrt14+NR2 and rcp14+NR2. returns (xr, xi)"""
    TINY = e.v("_mm512_set1_pd(1e-30)")
    VONE = e.v("_mm512_set1_pd(1.0)")
    VHALF = e.v("_mm512_set1_pd(0.5)")
    m  = fmadd(e, zr, zr, fmadd(e, zi, zi, TINY))
    r0 = e.v(f"_mm512_rsqrt14_pd({m})")
    t  = mul(e, m, r0)
    hr = mul(e, r0, VHALF)
    eh = fnmadd(e, t, hr, VHALF)
    r1 = fmadd(e, r0, eh, r0)
    mg0= mul(e, m, r1)
    hr1= mul(e, r1, VHALF)
    e2 = fnmadd(e, mg0, mg0, m)
    mag= fmadd(e, e2, hr1, mg0)
    u  = add(e, VONE, mag)
    w0 = e.v(f"_mm512_rcp14_pd({u})")
    e3 = fnmadd(e, u, w0, VONE)
    a  = fmadd(e, w0, e3, w0)
    ee = mul(e, e3, e3)
    w2 = fmadd(e, a, ee, a)
    return (mul(e, zr, w2), mul(e, zi, w2))

def emit_map_f(e, zr, zi, pfx=""):
    """z/(1+|z|) float-seeded (no microcoded approx ops). returns (xr, xi)"""
    TINY = e.v("_mm512_set1_pd(1e-30)")
    VONE = e.v("_mm512_set1_pd(1.0)")
    VHALF = e.v("_mm512_set1_pd(0.5)")
    m  = fmadd(e, zr, zr, fmadd(e, zi, zi, TINY))
    r0 = e.v(f"_mm512_cvtps_pd(_mm256_rsqrt_ps(_mm512_cvtpd_ps({m})))")
    t  = mul(e, m, r0)
    hr = mul(e, r0, VHALF)
    eh = fnmadd(e, t, hr, VHALF)
    r1 = fmadd(e, r0, eh, r0)
    t2 = mul(e, m, r1)
    hr2= mul(e, r1, VHALF)
    eh2= fnmadd(e, t2, hr2, VHALF)
    r2 = fmadd(e, r1, eh2, r1)
    mg0= mul(e, m, r2)
    hr3= mul(e, r2, VHALF)
    e2 = fnmadd(e, mg0, mg0, m)
    mag= fmadd(e, e2, hr3, mg0)
    u  = add(e, VONE, mag)
    w0 = e.v(f"_mm512_cvtps_pd(_mm256_rcp_ps(_mm512_cvtpd_ps({u})))")
    e3 = fnmadd(e, u, w0, VONE)
    a  = fmadd(e, w0, e3, w0)
    ee = mul(e, e3, e3)
    w2 = fmadd(e, a, ee, a)
    e4 = mul(e, ee, ee)
    w3 = fmadd(e, w2, e4, w2)
    return (mul(e, zr, w3), mul(e, zi, w3))


def emit_map_staged(e, zs, group=4, fmode=None):
    _f = MAPF_COMPOSITE if fmode is None else fmode
    """map a list of (zr,zi) pairs with stage interleaving; returns mapped list"""
    TINY = e.v("_mm512_set1_pd(1e-30)")
    VONE = e.v("_mm512_set1_pd(1.0)")
    VHALF = e.v("_mm512_set1_pd(0.5)")
    out = [None]*len(zs)
    for g0 in range(0, len(zs), group):
        g = list(range(g0, min(g0+group, len(zs))))
        st = {q: {'zr': zs[q][0], 'zi': zs[q][1]} for q in g}
        for q in g:
            st[q]['m'] = fmadd(e, st[q]['zr'], st[q]['zr'], fmadd(e, st[q]['zi'], st[q]['zi'], TINY))
        for q in g:
            st[q]['r0'] = e.v(f"_mm512_cvtps_pd(_mm256_rsqrt_ps(_mm512_cvtpd_ps({st[q]['m']})))") if _f else e.v(f"_mm512_rsqrt14_pd({st[q]['m']})")
        for q in g:
            s = st[q]
            s['t'] = mul(e, s['m'], s['r0'])
            s['hr'] = mul(e, s['r0'], VHALF)
        for q in g:
            s = st[q]
            s['eh'] = fnmadd(e, s['t'], s['hr'], VHALF)
        for q in g:
            s = st[q]
            r1 = fmadd(e, s['r0'], s['eh'], s['r0'])
            if _f:
                t2 = mul(e, s['m'], r1)
                hr2 = mul(e, r1, VHALF)
                eh2 = fnmadd(e, t2, hr2, VHALF)
                r1 = fmadd(e, r1, eh2, r1)
            s['r1'] = r1
        for q in g:
            s = st[q]
            s['mg0'] = mul(e, s['m'], s['r1'])
            s['hr1'] = mul(e, s['r1'], VHALF)
        for q in g:
            s = st[q]
            s['e2'] = fnmadd(e, s['mg0'], s['mg0'], s['m'])
        for q in g:
            s = st[q]
            s['mag'] = fmadd(e, s['e2'], s['hr1'], s['mg0'])
            s['u'] = add(e, VONE, s['mag'])
        if RCPMAGIC_I:
            MG = e.v(f"_mm512_set1_pd({MAGICHEX})")
            for q in g:
                st[q]['w0'] = e.v(f"_mm512_castsi512_pd(_mm512_sub_epi64(_mm512_castpd_si512({MG}), _mm512_castpd_si512({st[q]['u']})))")
        else:
            for q in g:
                st[q]['w0'] = e.v(f"_mm512_cvtps_pd(_mm256_rcp_ps(_mm512_cvtpd_ps({st[q]['u']})))") if _f else e.v(f"_mm512_rcp14_pd({st[q]['u']})")
        for q in g:
            s = st[q]
            s['e3'] = fnmadd(e, s['u'], s['w0'], VONE)
        for q in g:
            s = st[q]
            s['a'] = fmadd(e, s['w0'], s['e3'], s['w0'])
            s['ee'] = mul(e, s['e3'], s['e3'])
        for q in g:
            s = st[q]
            w2 = fmadd(e, s['a'], s['ee'], s['a'])
            if RCPMAGIC_I:
                e4 = mul(e, s['ee'], s['ee'])
                w2 = fmadd(e, w2, e4, w2)
                e8 = mul(e, e4, e4)
                w2 = fmadd(e, w2, e8, w2)
            elif _f:
                e4 = mul(e, s['ee'], s['ee'])
                w2 = fmadd(e, w2, e4, w2)
            s['w2'] = w2
        for q in g:
            s = st[q]
            out[q] = (mul(e, s['zr'], s['w2']), mul(e, s['zi'], s['w2']))
    return out


def interleave_into(e, ea, eb):
    """merge two emitters' lines into e, alternating proportionally"""
    la, lb = ea.lines, eb.lines
    na, nb = len(la), len(lb)
    ia = ib = 0
    while ia < na or ib < nb:
        if ia < na:
            e.raw(la[ia]); ia += 1
        if ia < na and (ib >= nb or ia * nb <= ib * na):
            e.raw(la[ia]); ia += 1
        if ib < nb:
            e.raw(lb[ib]); ib += 1
    return e

def gen_composite_pair_mn(L, name):
    """two-pencil fused z_mn: dft+map+dft on px and px2 (es runtime)."""
    e = E(pfx="t")
    ea = E(pfx="pa"); eb = E(pfx="pb")
    xa = [(load(ea,"px",f"{q}*es"), load(ea,"px",f"{q}*es+8")) for q in range(L)]
    xb = [(load(eb,"px2",f"{q}*es"), load(eb,"px2",f"{q}*es+8")) for q in range(L)]
    ya = dft_graph(ea, xa, L)
    yb = dft_graph(eb, xb, L)
    interleave_into(e, ea, eb)
    zs = []
    for q in range(L):
        cr = load(e, "pc", f"{q}*16"); ci = load(e, "pc", f"{q}*16+8")
        zs.append((add(e, ya[q][0], cr), add(e, ya[q][1], ci)))
    for q in range(L):
        cr = load(e, "pc2", f"{q}*16"); ci = load(e, "pc2", f"{q}*16+8")
        zs.append((add(e, yb[q][0], cr), add(e, yb[q][1], ci)))
    mapped = emit_map_staged(e, zs)
    ec = E(pfx="qa"); ed = E(pfx="qb")
    y2a = dft_graph(ec, mapped[:L], L)
    y2b = dft_graph(ed, mapped[L:], L)
    for q in range(L):
        store(ec, "px", f"{q}*es", y2a[q][0]); store(ec, "px", f"{q}*es+8", y2a[q][1])
        store(ed, "px2", f"{q}*es", y2b[q][0]); store(ed, "px2", f"{q}*es+8", y2b[q][1])
    interleave_into(e, ec, ed)
    return f"""static __attribute__((always_inline)) inline void {name}(double* px, double* px2, const long es, const double* pc, const double* pc2){{
{e.code()}
}}"""
def emit_map_on(e, x, cbase, q):
    cr = load(e, cbase, f"{q}*16"); ci = load(e, cbase, f"{q}*16+8")
    zr = add(e, x[0], cr); zi = add(e, x[1], ci)
    return emit_map(e, zr, zi)

# ---------- composite codelet (full-register DFT) ----------
def dft_graph(e, x, L):
    if L == 2: return dft2(e, x)
    if L == 3: return dft3(e, x)
    if L == 4: return dft4(e, x)
    if L == 5: return dft5(e, x)
    if L == 8: return dft8(e, x)
    if L == 6:
        # PFA 2x3
        inm, outm = pfa_maps(2, 3)
        # rows: DFT3 over n2 for each n1? decompose N1=2,N2=3: X[k1,k2] = sum_{n1,n2} x[inm] w2^{k1n1} w3^{k2n2}
        # first dft3 across n2 for each n1, then dft2 across n1
        g = [[x[inm[n1][n2]] for n2 in range(3)] for n1 in range(2)]
        r = [dft3(e, g[n1]) for n1 in range(2)]
        out = [None]*6
        for k2 in range(3):
            col = dft2(e, [r[0][k2], r[1][k2]])
            for k1 in range(2):
                out[outm[k1][k2]] = col[k1]
        return out
    raise ValueError(L)

def gen_composite(L, PSZ):
    """emit codelet family for composite L (full-register).
    functions: cd{L}_p(px, es); cd{L}_m(px, es, pc); cd{L}_mn(px, es, pc); cd{L}_mns(px,es,pc,ps)"""
    out = []
    for variant in ("p", "m", "mn", "mns", "pp"):
        if variant == "pp":
            # pair: two plain DFTs at px and px+dq (for ILP in y-pass)
            e = E()
            x1 = [cloadv(e, "px", q) for q in range(0, 0)]  # placeholder
            e = E()
            xa = [(load(e,"px",f"{q}*es"), load(e,"px",f"{q}*es+8")) for q in range(L)]
            xb = [(load(e,"px2",f"{q}*es"), load(e,"px2",f"{q}*es+8")) for q in range(L)]
            ya = dft_graph(e, xa, L)
            yb = dft_graph(e, xb, L)
            for q in range(L):
                store(e, "px", f"{q}*es", ya[q][0]); store(e, "px", f"{q}*es+8", ya[q][1])
                store(e, "px2", f"{q}*es", yb[q][0]); store(e, "px2", f"{q}*es+8", yb[q][1])
            out.append(f"""static __attribute__((always_inline)) inline void cd{L}_pp(double* px, double* px2, const long es){{
{e.code()}
}}""")
            continue
        e = E()
        x = [(load(e,"px",f"{q}*es"), load(e,"px",f"{q}*es+8")) for q in range(L)]
        y = dft_graph(e, x, L)
        if variant != "p":
            zs = []
            for q in range(L):
                cr = load(e, "pc", f"{q}*16"); ci = load(e, "pc", f"{q}*16+8")
                zs.append((add(e, y[q][0], cr), add(e, y[q][1], ci)))
            y = emit_map_staged(e, zs)
            if variant == "mns":
                for q in range(L):
                    store(e, "ps", f"{q}*ss", y[q][0]); store(e, "ps", f"{q}*ss+8", y[q][1])
            if variant in ("mn", "mns"):
                y = dft_graph(e, y, L)
        for q in range(L):
            store(e, "px", f"{q}*es", y[q][0]); store(e, "px", f"{q}*es+8", y[q][1])
        args = {"p": "double* px, const long es",
                "m": "double* px, const long es, const double* pc",
                "mn": "double* px, const long es, const double* pc",
                "mns": "double* px, const long es, const double* pc, double* ps, const long ss"}[variant]
        out.append(f"""static __attribute__((always_inline)) inline void cd{L}_{variant}({args}){{
{e.code()}
}}""")
    out.append(gen_composite_pair_mn(L, f"cd{L}_mn2"))
    return "\n".join(out)

# ---------- prime codelet: phase-split direct symmetric ----------
def prime_tables(L):
    h = (L-1)//2
    cos = [float(np.cos(LD(2)*PI*LD(m)/LD(L))) for m in range(1, h+1)]
    sin = [float(np.sin(LD(2)*PI*LD(m)/LD(L))) for m in range(1, h+1)]
    return h, cos, sin

def fold_idx(t, L):
    h = (L-1)//2
    t %= L
    if t <= h: return t, 1
    return L - t, -1

def gen_prime(L, kb1, kb2):
    """phase-split prime codelet. kb1: k-block size for cos phase, kb2 for sin phase.
    emits cd{L}_p / _m / _mn / _mns using static scratch.
    scratch: A_{L}[2h+2] vec-pairs for cos accums; M_{L}[L] vec-pairs for mapped mid-values."""
    h, cosv, sinv = prime_tables(L)
    kblocks1 = [list(range(s, min(s+kb1, h+1))) for s in range(1, h+1, kb1)]
    kblocks2 = [list(range(s, min(s+kb2, h+1))) for s in range(1, h+1, kb2)]
    decl = f"static double A_{L}[{2*h}+2][16] ALIGN64;\nstatic double M_{L}[{L}][16] ALIGN64;\nstatic double EO_{L}[{4*h}][8] ALIGN64;\n"

    def emit_dft(e, src, ses, dst, des, mapmode, snap):
        """one full DFT from src (stride ses doubles per slot) to dst.
        mapmode: None | 'map' (apply map using pc with slot q at pc + q*16)"""
        need_eo = len(kblocks1) > 1 or len(kblocks2) > 1
        # ---- cos phases ----
        first = True
        for kb in kblocks1:
            e.raw("{")
            C = {}
            for mth in sorted(set(fold_idx(k*j, L)[0] for k in kb for j in range(1, h+1))):
                C[mth] = setc(e, cosv[mth-1])
            if first:
                x0r = load(e, src, f"0*{ses}"); x0i = load(e, src, f"0*{ses}+8")
                if len(kblocks1) > 1:
                    e.raw(f"_mm512_store_pd(A_{L}[{2*h}], {x0r});")
                    e.raw(f"_mm512_store_pd(A_{L}[{2*h+1}], {x0i});")
            else:
                x0r = e.v(f"_mm512_load_pd(A_{L}[{2*h}])"); x0i = e.v(f"_mm512_load_pd(A_{L}[{2*h+1}])")
            acc = {}
            sE = None
            for j in range(1, h+1):
                if first:
                    ar = load(e, src, f"{j}*{ses}"); ai = load(e, src, f"{j}*{ses}+8")
                    br = load(e, src, f"{L-j}*{ses}"); bi = load(e, src, f"{L-j}*{ses}+8")
                    er = add(e, ar, br); ei = add(e, ai, bi)
                    if need_eo:
                        orr = sub(e, ar, br); oi = sub(e, ai, bi)
                        e.raw(f"_mm512_store_pd(EO_{L}[{4*(j-1)}], {er});")
                        e.raw(f"_mm512_store_pd(EO_{L}[{4*(j-1)+1}], {ei});")
                        e.raw(f"_mm512_store_pd(EO_{L}[{4*(j-1)+2}], {orr});")
                        e.raw(f"_mm512_store_pd(EO_{L}[{4*(j-1)+3}], {oi});")
                else:
                    er = e.v(f"_mm512_load_pd(EO_{L}[{4*(j-1)}])"); ei = e.v(f"_mm512_load_pd(EO_{L}[{4*(j-1)+1}])")
                if first:
                    sE = (add(e, sE[0], er), add(e, sE[1], ei)) if sE else (add(e, x0r, er), add(e, x0i, ei))
                for k in kb:
                    mth, sgn = fold_idx(k*j, L)
                    if k in acc:
                        acc[k] = (fmadd(e, C[mth], er, acc[k][0]), fmadd(e, C[mth], ei, acc[k][1]))
                    else:
                        acc[k] = (fmadd(e, C[mth], er, x0r), fmadd(e, C[mth], ei, x0i))
            for k in kb:
                e.raw(f"_mm512_store_pd(A_{L}[{2*(k-1)}], {acc[k][0]});")
                e.raw(f"_mm512_store_pd(A_{L}[{2*(k-1)+1}], {acc[k][1]});")
            if first:
                # X0 = sE ; optionally map
                if mapmode == 'map':
                    X0 = emit_map_on(e, sE, "pc", 0)
                else:
                    X0 = sE
                if snap:
                    store(e, "ps", "0*ss", X0[0]); store(e, "ps", "0*ss+8", X0[1])
                store(e, dst, f"0*{des}", X0[0]); store(e, dst, f"0*{des}+8", X0[1])
            e.raw("}")
            first = False
        # ---- sin phases + combine ----
        for kb in kblocks2:
            e.raw("{")
            S = {}
            for mth in sorted(set(fold_idx(k*j, L)[0] for k in kb for j in range(1, h+1))):
                S[mth] = setc(e, sinv[mth-1])
            acc = {}
            first_j = True
            for j in range(1, h+1):
                if need_eo:
                    orr = e.v(f"_mm512_load_pd(EO_{L}[{4*(j-1)+2}])"); oi = e.v(f"_mm512_load_pd(EO_{L}[{4*(j-1)+3}])")
                else:
                    ar = load(e, src, f"{j}*{ses}"); ai = load(e, src, f"{j}*{ses}+8")
                    br = load(e, src, f"{L-j}*{ses}"); bi = load(e, src, f"{L-j}*{ses}+8")
                    orr = sub(e, ar, br); oi = sub(e, ai, bi)
                for k in kb:
                    mth, sgn = fold_idx(k*j, L)
                    if first_j:
                        if sgn > 0:
                            acc[k] = (mul(e, S[mth], orr), mul(e, S[mth], oi))
                        else:
                            NZ = f"_mm512_setzero_pd()"
                            acc[k] = (fnmadd(e, S[mth], orr, e.v(NZ)), fnmadd(e, S[mth], oi, e.v(NZ)))
                    else:
                        if sgn > 0:
                            acc[k] = (fmadd(e, S[mth], orr, acc[k][0]), fmadd(e, S[mth], oi, acc[k][1]))
                        else:
                            acc[k] = (fnmadd(e, S[mth], orr, acc[k][0]), fnmadd(e, S[mth], oi, acc[k][1]))
                first_j = False
            # combine: X_k = A - iB = (Ar + Bi, Ai - Br) [B=sum s*o, X_k = A_k - i*B_k]
            for k in kb:
                Ar = e.v(f"_mm512_load_pd(A_{L}[{2*(k-1)}])")
                Ai = e.v(f"_mm512_load_pd(A_{L}[{2*(k-1)+1}])")
                Xk  = (add(e, Ar, acc[k][1]), sub(e, Ai, acc[k][0]))
                Xlk = (sub(e, Ar, acc[k][1]), add(e, Ai, acc[k][0]))
                if mapmode == 'map':
                    Xk = emit_map_on(e, Xk, "pc", k)
                    Xlk = emit_map_on(e, Xlk, "pc", L-k)
                if snap:
                    store(e, "ps", f"{k}*ss", Xk[0]); store(e, "ps", f"{k}*ss+8", Xk[1])
                    store(e, "ps", f"{L-k}*ss", Xlk[0]); store(e, "ps", f"{L-k}*ss+8", Xlk[1])
                store(e, dst, f"{k}*{des}", Xk[0]); store(e, dst, f"{k}*{des}+8", Xk[1])
                store(e, dst, f"{L-k}*{des}", Xlk[0]); store(e, dst, f"{L-k}*{des}+8", Xlk[1])
            e.raw("}")

    funcs = []
    # plain: src=px es -> dst=px es
    e = E()
    emit_dft(e, "px", "es", "px", "es", None, False)
    funcs.append(f"static __attribute__((noinline)) void cd{L}_p(double* px, const long es){{\n{e.code()}\n}}")
    # map: dft + map -> store px
    e = E()
    emit_dft(e, "px", "es", "px", "es", 'map', False)
    funcs.append(f"static __attribute__((noinline)) void cd{L}_m(double* px, const long es, const double* pc){{\n{e.code()}\n}}")
    # map+next: dft + map -> M ; dft(M) -> px
    e = E()
    emit_dft(e, "px", "es", f"(double*)M_{L}", "16", 'map', False)
    e.raw('__asm__ volatile("" ::: "memory");')
    emit_dft(e, f"(double*)M_{L}", "16", "px", "es", None, False)
    funcs.append(f"static __attribute__((noinline)) void cd{L}_mn(double* px, const long es, const double* pc){{\n{e.code()}\n}}")
    # map+next+snap
    e = E()
    emit_dft(e, "px", "es", f"(double*)M_{L}", "16", 'map', True)
    e.raw('__asm__ volatile("" ::: "memory");')
    emit_dft(e, f"(double*)M_{L}", "16", "px", "es", None, False)
    funcs.append(f"static __attribute__((noinline)) void cd{L}_mns(double* px, const long es, const double* pc, double* ps, const long ss){{\n{e.code()}\n}}")
    return decl + "\n".join(funcs)
