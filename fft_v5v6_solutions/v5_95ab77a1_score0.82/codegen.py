#!/usr/bin/env python3
# Code generator v2: compact looped table-driven AVX-512 kernels.
import mpmath as mp

mp.mp.prec = 120
SIZES = (6, 8, 13, 17, 23, 36, 45, 64)


def tw(N, e):
    e = e % N
    ang = mp.mpf(2) * mp.pi * e / N
    wr = float(mp.cos(ang))
    wi = float(-mp.sin(ang))
    def snap(x):
        for v in (0.0, 1.0, -1.0, 0.5, -0.5):
            if abs(x - v) < 1e-30:
                return v
        return x
    return snap(wr), snap(wi)


def cs(N, e):
    wr, wi = tw(N, e)
    return wr, -wi


def lit(x):
    x = float(x)
    if x == int(x) and abs(x) < 1e15:
        return f"{x:.1f}"
    return x.hex()


_EM_SEQ = [0]

class Em:
    def __init__(self):
        self.lines = []
        self.cnt = 0
        self.pfx = f"t{_EM_SEQ[0]}_"
        _EM_SEQ[0] += 1

    def raw(self, s):
        self.lines.append(s)

    def mk(self):
        n = f"{self.pfx}{self.cnt}"
        self.cnt += 1
        return n

    def v(self, expr):
        n = f"{self.pfx}{self.cnt}"
        self.cnt += 1
        self.lines.append(f"__m512d {n} = {expr};")
        return n

    def setc(self, x):
        return f"_mm512_set1_pd({lit(x)})"

    def add(self, a, b): return self.v(f"_mm512_add_pd({a},{b})")
    def sub(self, a, b): return self.v(f"_mm512_sub_pd({a},{b})")
    def mul(self, a, b): return self.v(f"_mm512_mul_pd({a},{b})")
    def fmadd(self, a, b, c): return self.v(f"_mm512_fmadd_pd({a},{b},{c})")
    def fmsub(self, a, b, c): return self.v(f"_mm512_fmsub_pd({a},{b},{c})")
    def fnmadd(self, a, b, c): return self.v(f"_mm512_fnmadd_pd({a},{b},{c})")
    def neg(self, a): return self.v(f"_mm512_xor_pd({a},_mm512_set1_pd(-0.0))")
    def load(self, p): return self.v(f"_mm512_loadu_pd({p})")
    def zero(self): return self.v("_mm512_setzero_pd()")

    def code(self):
        return "\n".join(self.lines)


def cadd(e, a, b): return (e.add(a[0], b[0]), e.add(a[1], b[1]))
def csub(e, a, b): return (e.sub(a[0], b[0]), e.sub(a[1], b[1]))
def cneg(e, a): return (e.neg(a[0]), e.neg(a[1]))


def cmulw(e, a, w):
    wr, wi = w
    if wi == 0.0 and wr == 1.0:
        return a
    if wi == 0.0 and wr == -1.0:
        return cneg(e, a)
    if wr == 0.0 and wi == -1.0:
        return (a[1], e.neg(a[0]))
    if wr == 0.0 and wi == 1.0:
        return (e.neg(a[1]), a[0])
    cr = e.setc(wr)
    ci = e.setc(wi)
    t1 = e.mul(a[1], ci)
    re = e.fmsub(a[0], cr, t1)
    t2 = e.mul(a[1], cr)
    im = e.fmadd(a[0], ci, t2)
    return (re, im)


def cmul_tab(e, a, rexpr, iexpr):
    """complex mul by table-driven constant (runtime broadcast)."""
    cr = e.v(f"_mm512_set1_pd({rexpr})")
    ci = e.v(f"_mm512_set1_pd({iexpr})")
    t1 = e.mul(a[1], ci)
    re = e.fmsub(a[0], cr, t1)
    t2 = e.mul(a[1], cr)
    im = e.fmadd(a[0], ci, t2)
    return (re, im)


# ---------------- small DFT networks (SSA, unrolled) -------------------------

def dft(e, xs):
    N = len(xs)
    if N == 1:
        return xs
    if N == 2:
        return [cadd(e, xs[0], xs[1]), csub(e, xs[0], xs[1])]
    if N == 3:
        x0, x1, x2 = xs
        t1 = cadd(e, x1, x2)
        t2 = csub(e, x1, x2)
        X0 = cadd(e, x0, t1)
        half = e.setc(0.5)
        mr = e.fnmadd(t1[0], half, x0[0])
        mi = e.fnmadd(t1[1], half, x0[1])
        s3 = e.setc(cs(3, 1)[1])
        X1 = (e.fmadd(t2[1], s3, mr), e.fnmadd(t2[0], s3, mi))
        X2 = (e.fnmadd(t2[1], s3, mr), e.fmadd(t2[0], s3, mi))
        return [X0, X1, X2]
    if N == 4:
        x0, x1, x2, x3 = xs
        t0 = cadd(e, x0, x2)
        t1 = csub(e, x0, x2)
        t2 = cadd(e, x1, x3)
        t3 = csub(e, x1, x3)
        return [cadd(e, t0, t2),
                (e.add(t1[0], t3[1]), e.sub(t1[1], t3[0])),
                csub(e, t0, t2),
                (e.sub(t1[0], t3[1]), e.add(t1[1], t3[0]))]
    if N == 5:
        return dft_sym_unrolled(e, xs, 5)
    if N == 8:
        return dft_ct_unrolled(e, xs, 2, 4, 8)
    if N == 9:
        return dft_ct_unrolled(e, xs, 3, 3, 9)
    if N == 6:
        return dft_pfa_unrolled(e, xs, 2, 3, 6)
    raise ValueError(N)


def dft_ct_unrolled(e, xs, R, M, N):
    A = [dft(e, [xs[M * j1 + j2] for j1 in range(R)]) for j2 in range(M)]
    B = [[cmulw(e, A[j2][k1], tw(N, j2 * k1)) for k1 in range(R)] for j2 in range(M)]
    Y = [dft(e, [B[j2][k1] for j2 in range(M)]) for k1 in range(R)]
    out = [None] * N
    for k1 in range(R):
        for k2 in range(M):
            out[k1 + R * k2] = Y[k1][k2]
    return out


def dft_pfa_unrolled(e, xs, N1, N2, N):
    T = [dft(e, [xs[(N2 * j1 + N1 * j2) % N] for j2 in range(N2)]) for j1 in range(N1)]
    V = [dft(e, [T[j1][k2] for j1 in range(N1)]) for k2 in range(N2)]
    out = [None] * N
    for k in range(N):
        out[k] = V[k % N2][k % N1]
    return out


def dft_sym_unrolled(e, xs, N):
    h = (N - 1) // 2
    x0 = xs[0]
    u, v = {}, {}
    for j in range(1, h + 1):
        u[j] = cadd(e, xs[j], xs[N - j])
        v[j] = csub(e, xs[j], xs[N - j])
    acc = u[1]
    for j in range(2, h + 1):
        acc = cadd(e, acc, u[j])
    X = [None] * N
    X[0] = cadd(e, x0, acc)
    for k in range(1, h + 1):
        pr, pi = x0
        br = bi = None
        for j in range(1, h + 1):
            c, s = cs(N, (k * j) % N)
            cc, ss = e.setc(c), e.setc(s)
            pr = e.fmadd(u[j][0], cc, pr)
            pi = e.fmadd(u[j][1], cc, pi)
            if br is None:
                br = e.mul(v[j][0], ss)
                bi = e.mul(v[j][1], ss)
            else:
                br = e.fmadd(v[j][0], ss, br)
                bi = e.fmadd(v[j][1], ss, bi)
        X[k] = (e.add(pr, bi), e.sub(pi, br))
        X[N - k] = (e.sub(pr, bi), e.add(pi, br))
    return X


# ---------------- elementwise map ---------------------------------------------

def emit_elementwise(e, zr, zi):
    mkname = "m" + e.mk()
    t = e.mul(zi, zi)
    s = e.fmadd(zr, zr, t)
    e.raw(f"__mmask8 {mkname} = _mm512_cmp_pd_mask({s},_mm512_setzero_pd(),_CMP_GT_OQ);")
    r = e.v(f"_mm512_rsqrt14_pd({s})")
    hs = e.mul(s, e.setc(0.5))
    for _ in range(2):
        rr = e.mul(r, r)
        w = e.fnmadd(hs, rr, e.setc(1.5))
        r = e.mul(r, w)
    mag = e.v(f"_mm512_maskz_mul_pd({mkname},{s},{r})")
    d = e.add(e.setc(1.0), mag)
    q = e.v(f"_mm512_rcp14_pd({d})")
    for _ in range(2):
        er = e.fnmadd(d, q, e.setc(2.0))
        q = e.mul(q, er)
    return (e.mul(zr, q), e.mul(zi, q))


# ---------------- table registry ------------------------------------------------

TABLES = []


def add_table(name, ctype, vals, fmt):
    TABLES.append((name, ctype, [fmt(v) for v in vals]))


def table_decls():
    out = []
    for name, ctype, vals in TABLES:
        out.append(f"static const {ctype} {name}[{len(vals)}] __attribute__((aligned(64))) = {{" +
                   ",".join(vals) + "};")
    return "\n".join(out)


# ---------------- kernel body generator -----------------------------------------
# Generates the body transforming LD(j) -> ST(k).
# ld(j_expr:str or int) returns handle; st(k_expr, handle); both may be called
# inside emitted C loops (expressions reference loop vars).

def gen_body(L, out, ld, st, uid):
    e = Em()

    def flush():
        out.append(e.code())
        e.lines.clear()

    if L in (6, 8):
        xs = [ld(e, j) for j in range(L)]
        X = dft(e, xs)
        for k in range(L):
            st(e, k, X[k])
        flush()
        return

    if L in (13, 17, 23):
        h = (L - 1) // 2
        ct = []
        stab = []
        for k in range(1, h + 1):
            for j in range(1, h + 1):
                c, s = cs(L, (k * j) % L)
                ct.append(c)
                stab.append(s)
        tn_c, tn_s = f"CSY{L}", f"SSY{L}"
        if not any(t[0] == tn_c for t in TABLES):
            add_table(tn_c, "double", ct, lit)
            add_table(tn_s, "double", stab, lit)
        out.append(f"double US_{uid}[{4*h*8}] __attribute__((aligned(64)));")

        import os
        KB = int(os.environ.get('KB', '2'))

        e0 = Em()
        x0 = ld(e0, 0)
        x0r, x0i = x0
        out.append(e0.code())

        import os as _os2
        SPL = int(_os2.environ.get('SPLITACC', '0'))

        def kblock(e2, kexprs, first):
            accs = []
            accs2 = []
            for ke in kexprs:
                accs.append([x0r, x0i, None, None])
                accs2.append([None, None, None, None])
            sumr = sumi = None
            for j in range(h):
                if first:
                    a = ld(e2, j + 1)
                    b = ld(e2, L - 1 - j)
                    ur = e2.add(a[0], b[0])
                    ui = e2.add(a[1], b[1])
                    vr = e2.sub(a[0], b[0])
                    vi = e2.sub(a[1], b[1])
                    e2.raw(f"_mm512_store_pd(US_{uid}+{j*8}, {ur});")
                    e2.raw(f"_mm512_store_pd(US_{uid}+{(h+j)*8}, {ui});")
                    e2.raw(f"_mm512_store_pd(US_{uid}+{(2*h+j)*8}, {vr});")
                    e2.raw(f"_mm512_store_pd(US_{uid}+{(3*h+j)*8}, {vi});")
                    if sumr is None:
                        sumr, sumi = ur, ui
                    else:
                        sumr = e2.add(sumr, ur)
                        sumi = e2.add(sumi, ui)
                else:
                    ur = e2.v(f"_mm512_load_pd(US_{uid}+{j*8})")
                    ui = e2.v(f"_mm512_load_pd(US_{uid}+{(h+j)*8})")
                    vr = e2.v(f"_mm512_load_pd(US_{uid}+{(2*h+j)*8})")
                    vi = e2.v(f"_mm512_load_pd(US_{uid}+{(3*h+j)*8})")
                for t, ke in enumerate(kexprs):
                    cc = f"_mm512_set1_pd(cro{t}[{j}])"
                    ss = f"_mm512_set1_pd(sro{t}[{j}])"
                    a = accs[t] if (not SPL or j % 2 == 0) else accs2[t]
                    if a[0] is None:
                        a[0] = e2.mul(cc, ur)
                        a[1] = e2.mul(cc, ui)
                    else:
                        a[0] = e2.fmadd(cc, ur, a[0])
                        a[1] = e2.fmadd(cc, ui, a[1])
                    if a[2] is None:
                        a[2] = e2.mul(ss, vr)
                        a[3] = e2.mul(ss, vi)
                    else:
                        a[2] = e2.fmadd(ss, vr, a[2])
                        a[3] = e2.fmadd(ss, vi, a[3])
            if SPL:
                for t in range(len(kexprs)):
                    if accs2[t][0] is not None:
                        for q in range(4):
                            accs[t][q] = e2.add(accs[t][q], accs2[t][q])
            if first:
                X0 = (e2.add(x0r, sumr), e2.add(x0i, sumi))
                st(e2, 0, X0)
            for t, ke in enumerate(kexprs):
                pr, pi, br, bi = accs[t]
                Xk = (e2.add(pr, bi), e2.sub(pi, br))
                XLk = (e2.sub(pr, bi), e2.add(pi, br))
                st(e2, ke, Xk)
                st(e2, f"({L}-({ke}))", XLk)

        # first block unrolled with literal k values 1..KB (fused u/v)
        k0 = list(range(1, min(KB, h) + 1))
        out.append("{ const long k=1;")
        for t in range(len(k0)):
            out.append(f"const double* cro{t} = {tn_c} + {t*h};")
            out.append(f"const double* sro{t} = {tn_s} + {t*h};")
        eF = Em()
        kblock(eF, [str(k) for k in k0], True)
        out.append(eF.code())
        out.append("}")
        done = len(k0)
        nfull = (h - done) // KB
        rem = (h - done) % KB
        if nfull:
            out.append(f"for (long k={done+1};k<={done+nfull*KB};k+={KB}){{")
            for t in range(KB):
                out.append(f"const double* cro{t} = {tn_c} + (k-1+{t})*{h};")
                out.append(f"const double* sro{t} = {tn_s} + (k-1+{t})*{h};")
            e2 = Em()
            kblock(e2, ["k"] + [f"(k+{t})" for t in range(1, KB)], False)
            out.append(e2.code())
            out.append("}")
        if rem:
            out.append(f"{{ const long k={done+nfull*KB+1};")
            for t in range(rem):
                out.append(f"const double* cro{t} = {tn_c} + {(done+nfull*KB+t)*h};")
                out.append(f"const double* sro{t} = {tn_s} + {(done+nfull*KB+t)*h};")
            e3 = Em()
            kblock(e3, ["k"] + [f"(k+{t})" for t in range(1, rem)], False)
            out.append(e3.code())
            out.append("}")
        return

    if L == 64:
        # CT 8x8, twiddle tables TW64R/I indexed [j2*8+k1]
        if not any(t[0] == "TW64R" for t in TABLES):
            tr, ti = [], []
            for j2 in range(8):
                for k1 in range(8):
                    wr, wi = tw(64, j2 * k1)
                    tr.append(wr)
                    ti.append(wi)
            add_table("TW64R", "double", tr, lit)
            add_table("TW64I", "double", ti, lit)
        out.append(f"double SS_{uid}[1024] __attribute__((aligned(64)));")
        # stage 1
        out.append("for (long j2=0;j2<8;j2++){")
        e1 = Em()
        xs = [ld(e1, f"(8*{j1}+j2)") for j1 in range(8)]
        A = dft(e1, xs)
        for k1 in range(8):
            if k1 == 0:
                B = A[0]
            else:
                B = cmul_tab(e1, A[k1], f"TW64R[j2*8+{k1}]", f"TW64I[j2*8+{k1}]")
            e1.raw(f"_mm512_store_pd(SS_{uid}+{k1*8}*16+j2*16, {B[0]});")
            e1.raw(f"_mm512_store_pd(SS_{uid}+{k1*8}*16+j2*16+8, {B[1]});")
        out.append(e1.code())
        out.append("}")
        # stage 2
        out.append("for (long k1=0;k1<8;k1++){")
        e2 = Em()
        ys = [(e2.v(f"_mm512_load_pd(SS_{uid}+k1*128+{j2*16})"),
               e2.v(f"_mm512_load_pd(SS_{uid}+k1*128+{j2*16}+8)")) for j2 in range(8)]
        Y = dft(e2, ys)
        for k2 in range(8):
            st(e2, f"(k1+{8*k2})", Y[k2])
        out.append(e2.code())
        out.append("}")
        return

    if L in (36, 45):
        if L == 36:
            N1, N2 = 4, 9   # stage1: N1 blocks of DFT_N2(9); stage2: N2 blocks of DFT_N1(4)
        else:
            N1, N2 = 9, 5
        # input index table: for j1 block: idx[j1*N2+j2] = (N2*j1 + N1*j2) % N
        idx_in = [(N2 * j1 + N1 * j2) % L for j1 in range(N1) for j2 in range(N2)]
        # output index: for k2 block: out element for k1: k with k%N1==k1, k%N2==k2
        idx_out = []
        for k2 in range(N2):
            for k1 in range(N1):
                for k in range(L):
                    if k % N1 == k1 and k % N2 == k2:
                        idx_out.append(k)
                        break
        tn_in, tn_out = f"PIN{L}_{uid}", f"POUT{L}_{uid}"
        add_table(tn_in, "long", idx_in, lambda v: str(v))
        add_table(tn_out, "long", idx_out, lambda v: str(v))
        out.append(f"double SS_{uid}[{N1*N2*16}] __attribute__((aligned(64)));")
        out.append(f"for (long j1=0;j1<{N1};j1++){{")
        out.append(f"const long* ii = {tn_in} + j1*{N2};")
        e1 = Em()
        xs = [ld(e1, f"ii[{j2}]") for j2 in range(N2)]
        T = dft(e1, xs)
        for k2 in range(N2):
            e1.raw(f"_mm512_store_pd(SS_{uid}+(j1*{N2}+{k2})*16, {T[k2][0]});")
            e1.raw(f"_mm512_store_pd(SS_{uid}+(j1*{N2}+{k2})*16+8, {T[k2][1]});")
        out.append(e1.code())
        out.append("}")
        out.append(f"for (long k2=0;k2<{N2};k2++){{")
        out.append(f"const long* oo = {tn_out} + k2*{N1};")
        e2 = Em()
        ys = [(e2.v(f"_mm512_load_pd(SS_{uid}+({j1*N2})*16+k2*16)"),
               e2.v(f"_mm512_load_pd(SS_{uid}+({j1*N2})*16+k2*16+8)")) for j1 in range(N1)]
        V = dft(e2, ys)
        for k1 in range(N1):
            st(e2, f"oo[{k1}]", V[k1])
        out.append(e2.code())
        out.append("}")
        return

    raise ValueError(L)




HAND_SIZES = (36, 45, 64)

def body_hand(L, out, ldc, stc, uid, srd):
    """hand-form looped body; ldc(idx_expr_bytesoff_name, i, Rdst, Idst), stc(kexpr...)"""
    if L == 64:
        if not any(t[0] == "TW64R" for t in TABLES):
            tr, ti = [], []
            for j2 in range(8):
                for k1 in range(8):
                    wr, wi = tw(64, j2 * k1)
                    tr.append(wr)
                    ti.append(wi)
            add_table("TW64R", "double", tr, lit)
            add_table("TW64I", "double", ti, lit)
        out.append(f"double SS_{uid}[1024] __attribute__((aligned(64)));")
        out.append("for (long j2=0;j2<8;j2++){ __m512d R[8], I[8];")
        for j1 in range(8):
            out.append(ldc(f"{8*j1}+j2", f"R[{j1}]", f"I[{j1}]"))
        out.append("dft8v(R,I);")
        out.append("""for (int k1=1;k1<8;k1++){
  __m512d wr=_mm512_set1_pd(TW64R[j2*8+k1]), wi=_mm512_set1_pd(TW64I[j2*8+k1]);
  __m512d nr=_mm512_fmsub_pd(R[k1],wr,_mm512_mul_pd(I[k1],wi));
  I[k1]=_mm512_fmadd_pd(R[k1],wi,_mm512_mul_pd(I[k1],wr)); R[k1]=nr; }""")
        for k1 in range(8):
            out.append(f"_mm512_store_pd(SS_{uid}+{k1*128}+j2*16, R[{k1}]); _mm512_store_pd(SS_{uid}+{k1*128}+j2*16+8, I[{k1}]);")
        out.append("}")
        out.append("for (long k1=0;k1<8;k1++){ __m512d R[8], I[8];")
        for j2 in range(8):
            out.append(f"R[{j2}]=_mm512_load_pd(SS_{uid}+k1*128+{j2*16}); I[{j2}]=_mm512_load_pd(SS_{uid}+k1*128+{j2*16+8});")
        out.append("dft8v(R,I);")
        for k2 in range(8):
            out.append(stc(f"k1+{8*k2}", f"R[{k2}]", f"I[{k2}]"))
        out.append("}")
        return
    if L in (36, 45):
        if L == 36:
            N1, N2, dfn1, dfn2 = 4, 9, "dft4v", "dft9v"
        else:
            N1, N2, dfn1, dfn2 = 9, 5, "dft9v", "dft5v"
        # byte-offset tables scaled by srd
        offin = [((N2*j1 + N1*j2) % L) * srd * 8 for j1 in range(N1) for j2 in range(N2)]
        offout = []
        for k2 in range(N2):
            for k1 in range(N1):
                for k in range(L):
                    if k % N1 == k1 and k % N2 == k2:
                        offout.append(k * srd * 8)
                        break
        tin, tout = f"OIN_{uid}", f"OOUT_{uid}"
        add_table(tin, "long", offin, str)
        add_table(tout, "long", offout, str)
        out.append(f"double SS_{uid}[{N1*N2*16}] __attribute__((aligned(64)));")
        out.append(f"for (long j1=0;j1<{N1};j1++){{ __m512d R[{N2}], I[{N2}];")
        out.append(f"const long* oin = {tin} + j1*{N2};")
        for j2 in range(N2):
            out.append(ldc(("B", f"oin[{j2}]"), f"R[{j2}]", f"I[{j2}]"))
        out.append(f"{dfn2}(R,I);")
        for k2 in range(N2):
            out.append(f"_mm512_store_pd(SS_{uid}+(j1*{N2}+{k2})*16, R[{k2}]); _mm512_store_pd(SS_{uid}+(j1*{N2}+{k2})*16+8, I[{k2}]);")
        out.append("}")
        out.append(f"for (long k2=0;k2<{N2};k2++){{ __m512d R[{N1}], I[{N1}];")
        out.append(f"const long* oo = {tout} + k2*{N1};")
        for j1 in range(N1):
            out.append(f"R[{j1}]=_mm512_load_pd(SS_{uid}+{j1*N2}*16+k2*16); I[{j1}]=_mm512_load_pd(SS_{uid}+{j1*N2}*16+k2*16+8);")
        out.append(f"{dfn1}(R,I);")
        for k1 in range(N1):
            out.append(stc(("B", f"oo[{k1}]"), f"R[{k1}]", f"I[{k1}]"))
        out.append("}")
        return
    raise ValueError(L)

# ---------------- kernel emitters ------------------------------------------------

def fmt_idx(j, srd):
    if isinstance(j, int):
        return str(j * srd)
    return f"({j})*{srd}"


def addr(base, j, srd):
    if isinstance(j, tuple) and j[0] == "B":   # byte offset expression
        return f"(double*)((char*)({base})+({j[1]}))"
    if isinstance(j, int):
        return f"({base})+{j*srd}"
    return f"({base})+({j})*{srd}"


def emit_colk(L, srd, smask, name, uid):
    """in-place pass along stride srd (literal), lanes unit-stride."""
    out = []
    out.append(f"static __attribute__((always_inline)) inline void {name}(double* re, double* im){{")

    if L in HAND_SIZES:
        def ldc(j, Rd, Id):
            return f"{Rd}=_mm512_loadu_pd({addr('re',j,srd)}); {Id}=_mm512_loadu_pd({addr('im',j,srd)});"

        def stc(k, Rs, Is):
            if smask == 0xFF:
                return (f"_mm512_storeu_pd({addr('re',k,srd)},{Rs}); "
                        f"_mm512_storeu_pd({addr('im',k,srd)},{Is});")
            return (f"_mm512_mask_storeu_pd({addr('re',k,srd)},0x{smask:02x},{Rs}); "
                    f"_mm512_mask_storeu_pd({addr('im',k,srd)},0x{smask:02x},{Is});")

        body_hand(L, out, ldc, stc, uid, srd)
        out.append("}")
        return "\n".join(out)

    def ld(e, j):
        o = fmt_idx(j, srd)
        return (e.load(f"re+{o}"), e.load(f"im+{o}"))

    def st(e, k, X):
        o = fmt_idx(k, srd)
        if smask == 0xFF:
            e.raw(f"_mm512_storeu_pd(re+{o},{X[0]});")
            e.raw(f"_mm512_storeu_pd(im+{o},{X[1]});")
        else:
            e.raw(f"_mm512_mask_storeu_pd(re+{o},0x{smask:02x},{X[0]});")
            e.raw(f"_mm512_mask_storeu_pd(im+{o},0x{smask:02x},{X[1]});")

    gen_body(L, out, ld, st, uid)
    out.append("}")
    return "\n".join(out)


def emit_xek(L, smask, name, uid, srd=None):
    """x pass + c + elementwise, stride SS (or given)."""
    if srd is None:
        srd = geom(L)[1]
    out = []
    out.append(f"static __attribute__((always_inline)) inline void {name}"
               f"(double* re, double* im, const double* cre, const double* cim){{")

    if L in HAND_SIZES:
        PF = True

        def ldc(j, Rd, Id):
            s = f"{Rd}=_mm512_loadu_pd({addr('re',j,srd)}); {Id}=_mm512_loadu_pd({addr('im',j,srd)});"
            if PF:
                s += (f" _mm_prefetch((const char*)({addr('re',j,srd)})+PFD,_MM_HINT_T0);"
                      f" _mm_prefetch((const char*)({addr('im',j,srd)})+PFD,_MM_HINT_T0);")
            return s

        ALT = [0]
        def stc(k, Rs, Is):
            ALT[0] ^= 1
            s = ""
            if PF:
                s = f"_mm_prefetch((const char*)({addr('cre',k,srd)})+PFD,_MM_HINT_T0); _mm_prefetch((const char*)({addr('cim',k,srd)})+PFD,_MM_HINT_T0); "
            return s + (f"ew_store_alt({addr('re',k,srd)},{addr('im',k,srd)},"
                    f"{addr('cre',k,srd)},{addr('cim',k,srd)},{Rs},{Is},0x{smask:02x},{ALT[0]});")

        body_hand(L, out, ldc, stc, uid, srd)
        out.append("}")
        return "\n".join(out)

    def ld(e, j):
        o = fmt_idx(j, srd)
        return (e.load(f"re+{o}"), e.load(f"im+{o}"))

    ALT = [0]
    def st(e, k, X):
        o = fmt_idx(k, srd)
        ALT[0] ^= 1
        e.raw(f"ew_store_alt(re+{o},im+{o},cre+{o},cim+{o},{X[0]},{X[1]},0x{smask:02x},{ALT[0]});")

    gen_body(L, out, ld, st, uid)
    out.append("}")
    return "\n".join(out)


def transpose8_c(e, rows):
    t = []
    for i in range(4):
        t.append(e.v(f"_mm512_unpacklo_pd({rows[2*i]},{rows[2*i+1]})"))
        t.append(e.v(f"_mm512_unpackhi_pd({rows[2*i]},{rows[2*i+1]})"))
    u = [None] * 8
    u[0] = e.v(f"_mm512_shuffle_f64x2({t[0]},{t[2]},0x88)")
    u[1] = e.v(f"_mm512_shuffle_f64x2({t[1]},{t[3]},0x88)")
    u[2] = e.v(f"_mm512_shuffle_f64x2({t[0]},{t[2]},0xdd)")
    u[3] = e.v(f"_mm512_shuffle_f64x2({t[1]},{t[3]},0xdd)")
    u[4] = e.v(f"_mm512_shuffle_f64x2({t[4]},{t[6]},0x88)")
    u[5] = e.v(f"_mm512_shuffle_f64x2({t[5]},{t[7]},0x88)")
    u[6] = e.v(f"_mm512_shuffle_f64x2({t[4]},{t[6]},0xdd)")
    u[7] = e.v(f"_mm512_shuffle_f64x2({t[5]},{t[7]},0xdd)")
    c = [None] * 8
    c[0] = e.v(f"_mm512_shuffle_f64x2({u[0]},{u[4]},0x88)")
    c[1] = e.v(f"_mm512_shuffle_f64x2({u[1]},{u[5]},0x88)")
    c[2] = e.v(f"_mm512_shuffle_f64x2({u[2]},{u[6]},0x88)")
    c[3] = e.v(f"_mm512_shuffle_f64x2({u[3]},{u[7]},0x88)")
    c[4] = e.v(f"_mm512_shuffle_f64x2({u[0]},{u[4]},0xdd)")
    c[5] = e.v(f"_mm512_shuffle_f64x2({u[1]},{u[5]},0xdd)")
    c[6] = e.v(f"_mm512_shuffle_f64x2({u[2]},{u[6]},0xdd)")
    c[7] = e.v(f"_mm512_shuffle_f64x2({u[3]},{u[7]},0xdd)")
    return c


def emit_zk(L, nlines, name, uid):
    """z pass on nlines consecutive z-lines (row stride = RS doubles).
    For small/sym L: transposes feed the kernel directly (no VV scratch);
    output side uses WW scratch only when the kernel stores from runtime loops."""
    RS = geom(L)[0]
    nb = (L + 7) // 8
    direct = L in (6, 8, 13, 17, 23)
    out = []
    out.append(f"static __attribute__((always_inline)) inline void {name}(double* re, double* im){{")
    XV = {}
    if direct:
        e = Em()
        for off, base in ((0, "re"), (1, "im")):
            for t in range(nb):
                rows = []
                for l in range(8):
                    if l < nlines:
                        rows.append(e.load(f"{base}+{l*RS}+{t*8}"))
                    else:
                        rows.append(e.zero())
                cols = transpose8_c(e, rows)
                for i in range(8):
                    j = t * 8 + i
                    if j < L:
                        XV.setdefault(j, [None, None])[off] = cols[i]
        out.append(e.code())
    else:
        out.append(f"double VV_{uid}[{nb*128}] __attribute__((aligned(64)));")
        for off, base in ((0, "re"), (8, "im")):
            out.append(f"for (long tb=0;tb<{nb};tb++){{")
            e = Em()
            rows = []
            for l in range(8):
                if l < nlines:
                    rows.append(e.load(f"{base}+{l*RS}+tb*8"))
                else:
                    rows.append(e.zero())
            cols = transpose8_c(e, rows)
            for i in range(8):
                e.raw(f"_mm512_store_pd(VV_{uid}+tb*128+{i*16+off}, {cols[i]});")
            out.append(e.code())
            out.append("}")

    out.append(f"double WW_{uid}[{nb*128}] __attribute__((aligned(64)));")

    if L in HAND_SIZES:
        def ldc(j, Rd, Id):
            return (f"{Rd}=_mm512_load_pd({addr(f'VV_{uid}',j,16)}); "
                    f"{Id}=_mm512_load_pd({addr(f'VV_{uid}',j,16)}+8);")

        def stc(k, Rs, Is):
            return (f"_mm512_store_pd({addr(f'WW_{uid}',k,16)},{Rs}); "
                    f"_mm512_store_pd({addr(f'WW_{uid}',k,16)}+8,{Is});")

        body_hand(L, out, ldc, stc, f"m{uid}", 16)
    else:
        def ld(e, j):
            return (XV[j][0], XV[j][1])

        def st(e, k, X):
            if isinstance(k, int):
                e.raw(f"_mm512_store_pd(WW_{uid}+{k*16},{X[0]});")
                e.raw(f"_mm512_store_pd(WW_{uid}+{k*16+8},{X[1]});")
            else:
                e.raw(f"_mm512_store_pd(WW_{uid}+({k})*16,{X[0]});")
                e.raw(f"_mm512_store_pd(WW_{uid}+({k})*16+8,{X[1]});")

        gen_body(L, out, ld, st, uid)

    # transpose-out from WW
    for off, base in ((0, "re"), (8, "im")):
        out.append(f"for (long tb=0;tb<{nb};tb++){{")
        e = Em()
        rows = [e.v(f"_mm512_load_pd(WW_{uid}+tb*128+{i*16+off})") for i in range(8)]
        cols = transpose8_c(e, rows)
        out.append(e.code())
        e2 = Em()
        if L % 8 == 0:
            for l in range(nlines):
                e2.raw(f"_mm512_storeu_pd({base}+{l*RS}+tb*8, {cols[l]});")
        else:
            kmasks = []
            for t in range(nb):
                m = 0
                for i in range(8):
                    if t * 8 + i < L:
                        m |= 1 << i
                kmasks.append(m)
            tn = f"KM{L}"
            if not any(x[0] == tn for x in TABLES):
                add_table(tn, "unsigned char", kmasks, lambda v: str(v))
            for l in range(nlines):
                e2.raw(f"_mm512_mask_storeu_pd({base}+{l*RS}+tb*8, {tn}[tb], {cols[l]});")
        out.append(e2.code())
        out.append("}")
    out.append("}")
    return "\n".join(out)


# ---------------- drivers -------------------------------------------------------

def geom(L):
    """returns (RS, SS, VS): row stride, slab stride, volume(plane) stride in doubles."""
    if L == 64:
        RS = 72
        SS = 64 * RS + 8
        VS = 64 * SS + 8
        return RS, SS, VS
    import os as _os
    padset = _os.environ.get('PADSET', '6,13,17,23,36,45')
    if str(L) in padset.split(','):
        RS = ((L + 7) // 8) * 8
        SS = L * RS
        VS = L * SS + 8
        return RS, SS, VS
    RS = L
    SS = L * L
    VS = ((L ** 3 + 7) // 8) * 8 + 16
    return RS, SS, VS


def vol_stride(L):
    return geom(L)[2]


def zsweep_lines(L, ind="      "):
    """flat z-pass over the whole volume (dense layouts) or per-slab (padded)."""
    RS, SS, VS = geom(L)
    s = []
    if SS == L * RS:
        nrows = L * L
        nfull = nrows // 8
        tail = nrows % 8
        s.append(f"{ind}for (int g=0; g<{nfull}; g++) zk{L}_8(re + g*{8*RS}, im + g*{8*RS});")
        if tail:
            s.append(f"{ind}zk{L}_t(re + {nfull*8*RS}, im + {nfull*8*RS});")
    else:
        s.append(f"{ind}for (int sl=0; sl<{L}; sl++) for (int y=0; y<{L}; y++) z64f(re + sl*{SS} + y*{RS}, im + sl*{SS} + y*{RS});")
    return s


def zysweep_lines(L, ind="      "):
    """interleaved: flat z-groups, y-pass per slab as soon as its rows are done."""
    RS, SS, VS = geom(L)
    s = []
    if SS == L * RS:
        nrows = L * L
        nfull = nrows // 8
        tail = nrows % 8
        s.append(f"{ind}{{ int sl = 0;")
        s.append(f"{ind}for (int g=0; g<{nfull}; g++){{")
        s.append(f"{ind}  zk{L}_8(re + g*{8*RS}, im + g*{8*RS});")
        s.append(f"{ind}  while ((g+1)*8 >= (sl+1)*{L} && sl < {L}){{")
        s.append(f"{ind}    double* pr = re + sl*{SS}; double* pi = im + sl*{SS};")
        for ln in ysweep_lines(L, ind + "    "):
            s.append(ln)
        s.append(f"{ind}    sl++; }}")
        s.append(f"{ind}}}")
        if tail:
            s.append(f"{ind}zk{L}_t(re + {nfull*8*RS}, im + {nfull*8*RS});")
        s.append(f"{ind}while (sl < {L}){{")
        s.append(f"{ind}  double* pr = re + sl*{SS}; double* pi = im + sl*{SS};")
        for ln in ysweep_lines(L, ind + "  "):
            s.append(ln)
        s.append(f"{ind}  sl++; }}")
        s.append(f"{ind}}}")
    else:
        s.append(f"{ind}for (int sl=0; sl<{L}; sl++){{")
        s.append(f"{ind}  double* pr = re + sl*{SS}; double* pi = im + sl*{SS};")
        s.append(f"{ind}  for (int y=0; y<{L}; y++) z64f(pr + y*{RS}, pi + y*{RS});")
        for ln in ysweep_lines(L, ind + "  "):
            s.append(ln)
        s.append(f"{ind}}}")
    return s


def ysweep_lines(L, ind="        "):
    RS = geom(L)[0]
    s = []
    if RS % 8 == 0 and RS > L:
        ng = (L + 7) // 8
        s.append(f"{ind}for (int g=0; g<{ng}; g++) colk{L}(pr + g*8, pi + g*8);")
        return s
    nfull_y = L // 8
    tail_y = L % 8
    if nfull_y:
        s.append(f"{ind}for (int g=0; g<{nfull_y}; g++) colk{L}(pr + g*8, pi + g*8);")
    if tail_y:
        s.append(f"{ind}colk{L}_t(pr + {nfull_y*8}, pi + {nfull_y*8});")
    return s


def xsweep_lines(L, ind="      "):
    RS, SS, VS = geom(L)
    s = []
    if RS == L:
        L2 = L * L
        npg = L2 // 8
        tail_p = L2 % 8
        s.append(f"{ind}for (int g=0; g<{npg}; g++) xek{L}(re + g*8, im + g*8, cre + g*8, cim + g*8);")
        if tail_p:
            s.append(f"{ind}xek{L}_t(re + {npg*8}, im + {npg*8}, cre + {npg*8}, cim + {npg*8});")
    else:
        nzb = (L + 7) // 8 if L % 8 else L // 8
        s.append(f"{ind}for (int y=0; y<{L}; y++) for (int zb=0; zb<{nzb}; zb++){{ long o = y*{RS} + zb*8; xek{L}(re + o, im + o, cre + o, cim + o); }}")
    return s


def emit_run(L):
    L3 = L ** 3
    RS, SS, VS = geom(L)
    s = [f"""
void run{L}(double* st, double* cst, long B, long m, double* out1, double* outm){{
  for (long vv=0; vv<B; vv++){{
    double* re = st + vv*2L*{VS};
    double* im = re + {VS};
    double* cre = cst + vv*2L*{VS};
    double* cim = cre + {VS};
    for (long it=0; it<m; it++){{"""]
    s += zysweep_lines(L)
    s += xsweep_lines(L)
    s.append(f"""      if (it==0 && m>1) emit_inter{L}(re, im, out1 + vv*2L*{L3});
    }}
    if (m==1) emit_dual{L}(re, im, out1 + vv*2L*{L3}, outm + vv*2L*{L3});
    else emit_inter{L}(re, im, outm + vv*2L*{L3});
  }}
}}""")
    return "\n".join(s)


def emit_bench(L):
    RS, SS, VS = geom(L)
    s = [f"""
void bz{L}(double* st, long reps){{
  double* re = st; double* im = re + {VS};
  for (long r=0;r<reps;r++){{"""]
    s += zsweep_lines(L, "    ")
    s.append("  }\n}")
    s.append(f"""
void by{L}(double* st, long reps){{
  double* re = st; double* im = re + {VS};
  for (long r=0;r<reps;r++) for (int sl=0; sl<{L}; sl++){{
    double* pr = re + sl*{SS}; double* pi = im + sl*{SS};""")
    s += ysweep_lines(L, "    ")
    s.append("  }\n}")
    s.append(f"""
void bx{L}(double* st, double* cst, long reps){{
  double* re = st; double* im = re + {VS};
  double* cre = cst; double* cim = cre + {VS};
  for (long r=0;r<reps;r++){{""")
    s += xsweep_lines(L, "    ")
    s.append("  }\n}")
    return "\n".join(s)


EW_STORE = r"""
#ifndef EWV
#define EWV 2
#endif
static inline __attribute__((always_inline)) void ew_store_alt(double* pr, double* pi,
    const double* pcr, const double* pci, __m512d xr, __m512d xi, __mmask8 m, int alt){
  __m512d zr = _mm512_add_pd(xr, _mm512_loadu_pd(pcr));
  __m512d zi = _mm512_add_pd(xi, _mm512_loadu_pd(pci));
  __m512d s = _mm512_fmadd_pd(zr, zr, _mm512_mul_pd(zi, zi));
  __m512d mag;
  if (alt){
    __mmask8 nz = _mm512_cmp_pd_mask(s, _mm512_setzero_pd(), _CMP_GT_OQ);
    __m512d r = _mm512_rsqrt14_pd(s);
    __m512d hs = _mm512_mul_pd(s, _mm512_set1_pd(0.5));
    r = _mm512_mul_pd(r, _mm512_fnmadd_pd(hs, _mm512_mul_pd(r,r), _mm512_set1_pd(1.5)));
    r = _mm512_mul_pd(r, _mm512_fnmadd_pd(hs, _mm512_mul_pd(r,r), _mm512_set1_pd(1.5)));
    mag = _mm512_maskz_mul_pd(nz, s, r);
  } else {
    mag = _mm512_sqrt_pd(s);
  }
  __m512d d = _mm512_add_pd(_mm512_set1_pd(1.0), mag);
  __m512d q = _mm512_rcp14_pd(d);
  q = _mm512_mul_pd(q, _mm512_fnmadd_pd(d, q, _mm512_set1_pd(2.0)));
  q = _mm512_mul_pd(q, _mm512_fnmadd_pd(d, q, _mm512_set1_pd(2.0)));
  if (m == 0xff){
    _mm512_storeu_pd(pr, _mm512_mul_pd(zr, q));
    _mm512_storeu_pd(pi, _mm512_mul_pd(zi, q));
  } else {
    _mm512_mask_storeu_pd(pr, m, _mm512_mul_pd(zr, q));
    _mm512_mask_storeu_pd(pi, m, _mm512_mul_pd(zi, q));
  }
}

static inline __attribute__((always_inline)) void ew_store(double* pr, double* pi,
    const double* pcr, const double* pci, __m512d xr, __m512d xi, __mmask8 m){
  __m512d zr = _mm512_add_pd(xr, _mm512_loadu_pd(pcr));
  __m512d zi = _mm512_add_pd(xi, _mm512_loadu_pd(pci));
  __m512d s = _mm512_fmadd_pd(zr, zr, _mm512_mul_pd(zi, zi));
#if EWV == 1
  __mmask8 nz = _mm512_cmp_pd_mask(s, _mm512_setzero_pd(), _CMP_GT_OQ);
  __m512d r = _mm512_rsqrt14_pd(s);
  __m512d hs = _mm512_mul_pd(s, _mm512_set1_pd(0.5));
  r = _mm512_mul_pd(r, _mm512_fnmadd_pd(hs, _mm512_mul_pd(r,r), _mm512_set1_pd(1.5)));
  r = _mm512_mul_pd(r, _mm512_fnmadd_pd(hs, _mm512_mul_pd(r,r), _mm512_set1_pd(1.5)));
  __m512d mag = _mm512_maskz_mul_pd(nz, s, r);
#elif EWV == 3
  __m512d mag = s;  /* DIAG ONLY: wrong math, measures kernel+mem cost */
#else
  __m512d mag = _mm512_sqrt_pd(s);
#endif
  __m512d d = _mm512_add_pd(_mm512_set1_pd(1.0), mag);
  __m512d q = _mm512_rcp14_pd(d);
  q = _mm512_mul_pd(q, _mm512_fnmadd_pd(d, q, _mm512_set1_pd(2.0)));
  q = _mm512_mul_pd(q, _mm512_fnmadd_pd(d, q, _mm512_set1_pd(2.0)));
  if (m == 0xff){
    _mm512_storeu_pd(pr, _mm512_mul_pd(zr, q));
    _mm512_storeu_pd(pi, _mm512_mul_pd(zi, q));
  } else {
    _mm512_mask_storeu_pd(pr, m, _mm512_mul_pd(zr, q));
    _mm512_mask_storeu_pd(pi, m, _mm512_mul_pd(zi, q));
  }
}
"""


PREAMBLE = r"""
#include <immintrin.h>
#ifndef PFD
#define PFD 64
#endif
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <sys/mman.h>

static inline __attribute__((always_inline)) void tr8(__m512d* r){
  __m512d t0=_mm512_unpacklo_pd(r[0],r[1]), t1=_mm512_unpackhi_pd(r[0],r[1]);
  __m512d t2=_mm512_unpacklo_pd(r[2],r[3]), t3=_mm512_unpackhi_pd(r[2],r[3]);
  __m512d t4=_mm512_unpacklo_pd(r[4],r[5]), t5=_mm512_unpackhi_pd(r[4],r[5]);
  __m512d t6=_mm512_unpacklo_pd(r[6],r[7]), t7=_mm512_unpackhi_pd(r[6],r[7]);
  __m512d u0=_mm512_shuffle_f64x2(t0,t2,0x88), u1=_mm512_shuffle_f64x2(t1,t3,0x88);
  __m512d u2=_mm512_shuffle_f64x2(t0,t2,0xdd), u3=_mm512_shuffle_f64x2(t1,t3,0xdd);
  __m512d u4=_mm512_shuffle_f64x2(t4,t6,0x88), u5=_mm512_shuffle_f64x2(t5,t7,0x88);
  __m512d u6=_mm512_shuffle_f64x2(t4,t6,0xdd), u7=_mm512_shuffle_f64x2(t5,t7,0xdd);
  r[0]=_mm512_shuffle_f64x2(u0,u4,0x88); r[1]=_mm512_shuffle_f64x2(u1,u5,0x88);
  r[2]=_mm512_shuffle_f64x2(u2,u6,0x88); r[3]=_mm512_shuffle_f64x2(u3,u7,0x88);
  r[4]=_mm512_shuffle_f64x2(u0,u4,0xdd); r[5]=_mm512_shuffle_f64x2(u1,u5,0xdd);
  r[6]=_mm512_shuffle_f64x2(u2,u6,0xdd); r[7]=_mm512_shuffle_f64x2(u3,u7,0xdd);
}

static void emit_inter(const double* re, const double* im, double* out, long n){
  const __m512i IL = _mm512_setr_epi64(0,8,1,9,2,10,3,11);
  const __m512i IH = _mm512_setr_epi64(4,12,5,13,6,14,7,15);
  long i = 0;
  if ((((uintptr_t)out) & 63) == 0){
    for (; i + 8 <= n; i += 8){
      __m512d a = _mm512_loadu_pd(re + i);
      __m512d b = _mm512_loadu_pd(im + i);
      _mm512_stream_pd(out + 2*i,     _mm512_permutex2var_pd(a, IL, b));
      _mm512_stream_pd(out + 2*i + 8, _mm512_permutex2var_pd(a, IH, b));
    }
    _mm_sfence();
  } else {
    for (; i + 8 <= n; i += 8){
      __m512d a = _mm512_loadu_pd(re + i);
      __m512d b = _mm512_loadu_pd(im + i);
      _mm512_storeu_pd(out + 2*i,     _mm512_permutex2var_pd(a, IL, b));
      _mm512_storeu_pd(out + 2*i + 8, _mm512_permutex2var_pd(a, IH, b));
    }
  }
  for (; i < n; i++){ out[2*i] = re[i]; out[2*i+1] = im[i]; }
}

static void emit_inter2(const double* re, const double* im, double* o1, double* o2, long n){
  const __m512i IL = _mm512_setr_epi64(0,8,1,9,2,10,3,11);
  const __m512i IH = _mm512_setr_epi64(4,12,5,13,6,14,7,15);
  long i = 0;
  int nt = ((((uintptr_t)o1) & 63) == 0) && ((((uintptr_t)o2) & 63) == 0);
  if (nt){
    for (; i + 8 <= n; i += 8){
      __m512d a = _mm512_loadu_pd(re + i);
      __m512d b = _mm512_loadu_pd(im + i);
      __m512d lo = _mm512_permutex2var_pd(a, IL, b);
      __m512d hi = _mm512_permutex2var_pd(a, IH, b);
      _mm512_stream_pd(o1 + 2*i, lo); _mm512_stream_pd(o1 + 2*i + 8, hi);
      _mm512_stream_pd(o2 + 2*i, lo); _mm512_stream_pd(o2 + 2*i + 8, hi);
    }
    _mm_sfence();
  } else {
    for (; i + 8 <= n; i += 8){
      __m512d a = _mm512_loadu_pd(re + i);
      __m512d b = _mm512_loadu_pd(im + i);
      __m512d lo = _mm512_permutex2var_pd(a, IL, b);
      __m512d hi = _mm512_permutex2var_pd(a, IH, b);
      _mm512_storeu_pd(o1 + 2*i, lo); _mm512_storeu_pd(o1 + 2*i + 8, hi);
      _mm512_storeu_pd(o2 + 2*i, lo); _mm512_storeu_pd(o2 + 2*i + 8, hi);
    }
  }
  for (; i < n; i++){ o1[2*i] = re[i]; o1[2*i+1] = im[i]; o2[2*i] = re[i]; o2[2*i+1] = im[i]; }
}

static void conv_split(const double* src, double* re, double* im, long n){
  const __m512i IR = _mm512_setr_epi64(0,2,4,6,8,10,12,14);
  const __m512i II = _mm512_setr_epi64(1,3,5,7,9,11,13,15);
  long i = 0;
  for (; i + 8 <= n; i += 8){
    __m512d a = _mm512_loadu_pd(src + 2*i);
    __m512d b = _mm512_loadu_pd(src + 2*i + 8);
    _mm512_storeu_pd(re + i, _mm512_permutex2var_pd(a, IR, b));
    _mm512_storeu_pd(im + i, _mm512_permutex2var_pd(a, II, b));
  }
  for (; i < n; i++){ re[i] = src[2*i]; im[i] = src[2*i+1]; }
}

static void* arena_ptr[2] = {0,0};
static long arena_sz[2] = {0,0};
void* ensure_arena(long bytes, int which){
  if (arena_sz[which] >= bytes) return arena_ptr[which];
  if (arena_ptr[which]) free(arena_ptr[which]);
  long sz = ((bytes + (2L<<20) - 1) >> 21) << 21;
  void* p = aligned_alloc(2L<<20, sz);
  if (!p) p = aligned_alloc(64, sz);
  if (!p) return 0;
  madvise(p, sz, MADV_HUGEPAGE);
  memset(p, 0, sz);
  arena_ptr[which] = p; arena_sz[which] = sz;
  return p;
}
"""




def emit_z64f():
    if not any(t[0] == "TWV64R" for t in TABLES):
        tr, ti = [], []
        for k1 in range(8):
            for j2 in range(8):
                wr, wi = tw(64, k1 * j2)
                tr.append(wr)
                ti.append(wi)
        add_table("TWV64R", "double", tr, lit)
        add_table("TWV64I", "double", ti, lit)
    out = []
    out.append("static __attribute__((always_inline)) inline void z64f(double* re, double* im){")
    out.append("__m512d R[8], I[8];")
    for j1 in range(8):
        out.append(f"R[{j1}]=_mm512_loadu_pd(re+{8*j1}); I[{j1}]=_mm512_loadu_pd(im+{8*j1});")
    out.append("dft8v(R,I);")
    # twiddle with vector constants, k1=1..7
    out.append("""for (int k1=1;k1<8;k1++){
  __m512d wr=_mm512_load_pd(TWV64R+k1*8), wi=_mm512_load_pd(TWV64I+k1*8);
  __m512d nr=_mm512_fmsub_pd(R[k1],wr,_mm512_mul_pd(I[k1],wi));
  I[k1]=_mm512_fmadd_pd(R[k1],wi,_mm512_mul_pd(I[k1],wr)); R[k1]=nr; }""")
    e = Em()
    rows = [f"R[{i}]" for i in range(8)]
    cols = transpose8_c(e, rows)
    for i in range(8):
        e.raw(f"R[{i}]={cols[i]};")
    rows = [f"I[{i}]" for i in range(8)]
    cols = transpose8_c(e, rows)
    for i in range(8):
        e.raw(f"I[{i}]={cols[i]};")
    out.append(e.code())
    out.append("dft8v(R,I);")
    for k2 in range(8):
        out.append(f"_mm512_storeu_pd(re+{8*k2},R[{k2}]); _mm512_storeu_pd(im+{8*k2},I[{k2}]);")
    out.append("}")
    return "\n".join(out)



BATCH_SIZES = (6, 8, 13, 17)


def bps(L):
    """batch-group plane stride (doubles): 8 lanes per element + 64B pad"""
    return L**3 * 8 + 8


def emit_batch(L):
    L2, L3 = L * L, L ** 3
    PS = bps(L)
    parts = []
    parts.append(emit_colk(L, 8, 0xFF, f"bkz{L}", f"bz{L}u"))
    parts.append(emit_colk(L, 8 * L, 0xFF, f"bky{L}", f"by{L}u"))
    parts.append(emit_xek(L, 0xFF, f"bkx{L}", f"bx{L}u", srd=8 * L2))
    # convert: src = 8 consecutive volumes (interleaved complex, dense), dst = lane-planes
    parts.append(f"""
static void convb{L}(const double* src, double* re, double* im){{
  const __m512i IR = _mm512_setr_epi64(0,2,4,6,8,10,12,14);
  const __m512i II = _mm512_setr_epi64(1,3,5,7,9,11,13,15);
  long i = 0;
  for (; i + 8 <= {L3}; i += 8){{
    __m512d R[8], I[8];
    for (int l=0; l<8; l++){{
      __m512d a = _mm512_loadu_pd(src + l*{2*L3} + 2*i);
      __m512d b = _mm512_loadu_pd(src + l*{2*L3} + 2*i + 8);
      R[l] = _mm512_permutex2var_pd(a, IR, b);
      I[l] = _mm512_permutex2var_pd(a, II, b);
    }}
    tr8(R); tr8(I);
    for (int k=0; k<8; k++){{
      _mm512_store_pd(re + (i+k)*8, R[k]);
      _mm512_store_pd(im + (i+k)*8, I[k]);
    }}
  }}
#if {L3} % 8
  for (; i < {L3}; i++)
    for (int l=0; l<8; l++){{
      re[i*8+l] = src[l*{2*L3} + 2*i];
      im[i*8+l] = src[l*{2*L3} + 2*i + 1];
    }}
#endif
}}
static void emitb{L}(const double* re, const double* im, double* out){{
  const __m512i IL = _mm512_setr_epi64(0,8,1,9,2,10,3,11);
  const __m512i IH = _mm512_setr_epi64(4,12,5,13,6,14,7,15);
  long i = 0;
  int nt = ((((uintptr_t)out) & 63) == 0) && ({16 * L3} % 64 == 0);
  for (; i + 8 <= {L3}; i += 8){{
    __m512d R[8], I[8];
    for (int k=0; k<8; k++){{
      R[k] = _mm512_load_pd(re + (i+k)*8);
      I[k] = _mm512_load_pd(im + (i+k)*8);
    }}
    tr8(R); tr8(I);
    if (nt) for (int l=0; l<8; l++){{
      _mm512_stream_pd(out + l*{2*L3} + 2*i,     _mm512_permutex2var_pd(R[l], IL, I[l]));
      _mm512_stream_pd(out + l*{2*L3} + 2*i + 8, _mm512_permutex2var_pd(R[l], IH, I[l]));
    }}
    else for (int l=0; l<8; l++){{
      _mm512_storeu_pd(out + l*{2*L3} + 2*i,     _mm512_permutex2var_pd(R[l], IL, I[l]));
      _mm512_storeu_pd(out + l*{2*L3} + 2*i + 8, _mm512_permutex2var_pd(R[l], IH, I[l]));
    }}
  }}
  if (nt) _mm_sfence();
#if {L3} % 8
  for (; i < {L3}; i++)
    for (int l=0; l<8; l++){{
      out[l*{2*L3} + 2*i]     = re[i*8+l];
      out[l*{2*L3} + 2*i + 1] = im[i*8+l];
    }}
#endif
}}
void convball{L}(const double* src, double* st, long G){{
  for (long g=0; g<G; g++)
    convb{L}(src + g*{16*L3}, st + g*{2*PS}, st + g*{2*PS} + {PS});
}}
void runb{L}(double* st, double* cst, long G, long m, double* out1, double* outm){{
  for (long g=0; g<G; g++){{
    double* re = st + g*{2*PS}; double* im = re + {PS};
    double* cre = cst + g*{2*PS}; double* cim = cre + {PS};
    for (long it=0; it<m; it++){{
      for (long x=0; x<{L}; x++){{
        for (long r=x*{L}; r<(x+1)*{L}; r++) bkz{L}(re + r*{8*L}, im + r*{8*L});
        for (long z=0; z<{L}; z++) bky{L}(re + x*{8*L2} + z*8, im + x*{8*L2} + z*8);
      }}
      for (long s=0; s<{L2}; s++) bkx{L}(re + s*8, im + s*8, cre + s*8, cim + s*8);
      if (it==0 && m>1) emitb{L}(re, im, out1 + g*{16*L3});
    }}
    emitb{L}(re, im, outm + g*{16*L3});
    if (m==1) emitb{L}(re, im, out1 + g*{16*L3});
  }}
}}""")
    return "\n".join(parts)

def emit_convemit(L):
    RS, SS, VS = geom(L)
    L3 = L ** 3
    if RS == L:
        return f"""
static void emit_inter{L}(const double* re, const double* im, double* out){{ emit_inter(re, im, out, {L3}); }}
static void emit_dual{L}(const double* re, const double* im, double* o1, double* o2){{ emit_inter2(re, im, o1, o2, {L3}); }}
void convert_all{L}(const double* src, double* st, long B){{
  for (long v = 0; v < B; v++)
    conv_split(src + v*2*{L3}, st + v*2L*{VS}, st + v*2L*{VS} + {VS}, {L3});
}}"""
    # padded: per-row handling
    return f"""
static void emit_inter{L}(const double* re, const double* im, double* out){{
  for (long x=0; x<{L}; x++) for (long y=0; y<{L}; y++){{
    long o = x*{SS} + y*{RS};
    emit_inter(re + o, im + o, out + (x*{L*L} + y*{L})*2, {L});
  }}
}}
static void emit_dual{L}(const double* re, const double* im, double* o1, double* o2){{
  for (long x=0; x<{L}; x++) for (long y=0; y<{L}; y++){{
    long o = x*{SS} + y*{RS};
    emit_inter2(re + o, im + o, o1 + (x*{L*L} + y*{L})*2, o2 + (x*{L*L} + y*{L})*2, {L});
  }}
}}
void convert_all{L}(const double* src, double* st, long B){{
  for (long v = 0; v < B; v++){{
    double* re = st + v*2L*{VS}; double* im = re + {VS};
    for (long x=0; x<{L}; x++) for (long y=0; y<{L}; y++){{
      long o = x*{SS} + y*{RS};
      conv_split(src + v*2*{L3} + (x*{L*L} + y*{L})*2, re + o, im + o, {L});
    }}
  }}
}}"""


def emit_vs():
    cases = "\n".join(f"    case {L}: return {geom(L)[2]};" for L in SIZES)
    bcases = "\n".join(f"    case {L}: return {bps(L)};" for L in BATCH_SIZES)
    return f"""
long vs_doubles(long L){{
  switch(L){{
{cases}
  }}
  return -1;
}}
long bps_doubles(long L){{
  switch(L){{
{bcases}
  }}
  return -1;
}}"""


def main():
    with open("dftv.h") as f:
        dftv = f.read()
    parts = [PREAMBLE, dftv, EW_STORE, emit_vs()]
    kparts = []
    for L in SIZES:
        tail_y = L % 8
        ymask = (1 << tail_y) - 1
        tail_p = (L * L) % 8
        pmask = (1 << tail_p) - 1
        RS, SS, VS = geom(L)
        padded = (RS % 8 == 0 and RS > L)
        kparts.append(emit_colk(L, RS, 0xFF, f"colk{L}", f"cy{L}"))
        if tail_y and not padded:
            kparts.append(emit_colk(L, RS, ymask, f"colk{L}_t", f"cyt{L}"))
        kparts.append(emit_xek(L, 0xFF, f"xek{L}", f"xe{L}"))
        if tail_p and not padded:
            kparts.append(emit_xek(L, pmask, f"xek{L}_t", f"xet{L}"))
        if L == 64:
            kparts.append(emit_z64f())
        else:
            kparts.append(emit_zk(L, 8, f"zk{L}_8", f"zk{L}"))
            if SS == L * RS and (L * L) % 8:
                kparts.append(emit_zk(L, (L * L) % 8, f"zk{L}_t", f"zkt{L}"))
        kparts.append(emit_convemit(L))
        kparts.append(emit_run(L))
        if L in BATCH_SIZES:
            kparts.append(emit_batch(L))
        kparts.append(emit_bench(L))
    parts.append(table_decls())
    parts.extend(kparts)
    src = "\n".join(parts)
    with open("implementation.c", "w") as f:
        f.write(src)
    print("lines:", src.count("\n"))


if __name__ == "__main__":
    main()
