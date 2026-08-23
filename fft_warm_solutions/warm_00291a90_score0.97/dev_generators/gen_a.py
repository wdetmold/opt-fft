import numpy as np
from genlib import *

def emit_map(e, zr, zi, pfx=""):
    """z/(1+|z|) via rsqrt14+NR2 and rcp14+NR2 (f40-proven). returns (xr, xi)"""
    TINY = e.v("_mm512_set1_pd(1e-300)")
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
            y2 = []
            for q in range(L):
                y2.append(emit_map_on(e, y[q], "pc", q))
            y = y2
            if variant == "mns":
                for q in range(L):
                    store(e, "ps", f"{q}*16", y[q][0]); store(e, "ps", f"{q}*16+8", y[q][1])
            if variant in ("mn", "mns"):
                y = dft_graph(e, y, L)
        for q in range(L):
            store(e, "px", f"{q}*es", y[q][0]); store(e, "px", f"{q}*es+8", y[q][1])
        args = {"p": "double* px, const long es",
                "m": "double* px, const long es, const double* pc",
                "mn": "double* px, const long es, const double* pc",
                "mns": "double* px, const long es, const double* pc, double* ps"}[variant]
        out.append(f"""static __attribute__((always_inline)) inline void cd{L}_{variant}({args}){{
{e.code()}
}}""")
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
    decl = f"static double A_{L}[{2*h}][16] ALIGN64;\nstatic double M_{L}[{L}][16] ALIGN64;\n"

    def emit_dft(e, src, ses, dst, des, mapmode, snap):
        """one full DFT from src (stride ses doubles per slot) to dst.
        mapmode: None | 'map' (apply map using pc with slot q at pc + q*16)"""
        # ---- cos phases ----
        first = True
        for kb in kblocks1:
            e.raw("{")
            C = {}
            for mth in sorted(set(fold_idx(k*j, L)[0] for k in kb for j in range(1, h+1))):
                C[mth] = setc(e, cosv[mth-1])
            x0r = load(e, src, f"0*{ses}"); x0i = load(e, src, f"0*{ses}+8")
            acc = {}
            for k in kb:
                acc[k] = (e.v(x0r), e.v(x0i))
            if first:
                sE = (e.v(x0r), e.v(x0i))
            for j in range(1, h+1):
                ar = load(e, src, f"{j}*{ses}"); ai = load(e, src, f"{j}*{ses}+8")
                br = load(e, src, f"{L-j}*{ses}"); bi = load(e, src, f"{L-j}*{ses}+8")
                er = add(e, ar, br); ei = add(e, ai, bi)
                if first:
                    sE = (add(e, sE[0], er), add(e, sE[1], ei))
                for k in kb:
                    mth, sgn = fold_idx(k*j, L)
                    acc[k] = (fmadd(e, C[mth], er, acc[k][0]), fmadd(e, C[mth], ei, acc[k][1]))
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
                    store(e, "ps", "0*16", X0[0]); store(e, "ps", "0*16+8", X0[1])
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
                    store(e, "ps", f"{k}*16", Xk[0]); store(e, "ps", f"{k}*16+8", Xk[1])
                    store(e, "ps", f"{L-k}*16", Xlk[0]); store(e, "ps", f"{L-k}*16+8", Xlk[1])
                store(e, dst, f"{k}*{des}", Xk[0]); store(e, dst, f"{k}*{des}+8", Xk[1])
                store(e, dst, f"{L-k}*{des}", Xlk[0]); store(e, dst, f"{L-k}*{des}+8", Xlk[1])
            e.raw("}")

    funcs = []
    # plain: src=px es -> dst=px es
    e = E()
    emit_dft(e, "px", "es", "px", "es", None, False)
    funcs.append(f"static __attribute__((always_inline)) inline void cd{L}_p(double* px, const long es){{\n{e.code()}\n}}")
    # map: dft + map -> store px
    e = E()
    emit_dft(e, "px", "es", "px", "es", 'map', False)
    funcs.append(f"static __attribute__((always_inline)) inline void cd{L}_m(double* px, const long es, const double* pc){{\n{e.code()}\n}}")
    # map+next: dft + map -> M ; dft(M) -> px
    e = E()
    emit_dft(e, "px", "es", f"(double*)M_{L}", "16", 'map', False)
    e.raw('__asm__ volatile("" ::: "memory");')
    emit_dft(e, f"(double*)M_{L}", "16", "px", "es", None, False)
    funcs.append(f"static __attribute__((always_inline)) inline void cd{L}_mn(double* px, const long es, const double* pc){{\n{e.code()}\n}}")
    # map+next+snap
    e = E()
    emit_dft(e, "px", "es", f"(double*)M_{L}", "16", 'map', True)
    e.raw('__asm__ volatile("" ::: "memory");')
    emit_dft(e, f"(double*)M_{L}", "16", "px", "es", None, False)
    funcs.append(f"static __attribute__((always_inline)) inline void cd{L}_mns(double* px, const long es, const double* pc, double* ps){{\n{e.code()}\n}}")
    return decl + "\n".join(funcs)
