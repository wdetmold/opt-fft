#!/usr/bin/env python3
"""Generator for implementation.c: specialized batched 3D FFT + map kernels.

Internal state layout (batch-major "group" form): for a group of V volumes,
element (x,y,z) lane l lives at  ((x*L+y)*L+z)*V + l  in separate re/im arrays.
Every 1D FFT is then a straight-line codelet on V-wide vectors with a
compile-time stride; no shuffles anywhere.
"""
import numpy as np

WV = ((23, 24), (36, 36), (45, 48))
PI = np.longdouble('3.14159265358979323846264338327950288')
SIZES = (6, 8, 13, 17, 23, 36, 45, 64)

def hexf(x):
    x = float(x)
    if x == int(x) and abs(x) < 1e15:
        return f"{x:.1f}"
    return float.hex(x)

def omega(N, k):
    """e^{-2*pi*i*k/N} rounded to doubles, with exact special values snapped."""
    kk = k % N
    ang = (-2 * PI) * np.longdouble(kk) / np.longdouble(N)
    c = float(np.cos(ang)); s = float(np.sin(ang))
    # snap values that are mathematically 0/:pm 1
    if (4 * kk) % N == 0:   # multiples of quarter turn are exact
        c = float(round(c)); s = float(round(s))
    return (c, s)

def trig(N, j, k, fn):
    ang = (2 * PI) * np.longdouble((j * k) % N) / np.longdouble(N)
    v = float(np.cos(ang)) if fn == 'c' else float(np.sin(ang))
    return v

class Gen:
    def __init__(self, vt):
        self.vt = vt; self.lines = []; self.cnt = 0
    def t(self):
        self.cnt += 1; return f"t{self.cnt}"
    def raw(self, s):
        self.lines.append(s)
    def emit(self, expr):
        t = self.t(); self.lines.append(f"{self.vt} {t} = {expr};"); return t
    def sp(self, c):
        return f"SP_{self.vt}({hexf(c)})"
    # real vector ops on variable names
    def add(self, a, b):  return self.emit(f"{a} + {b}")
    def sub(self, a, b):  return self.emit(f"{a} - {b}")
    def neg(self, a):     return self.emit(f"-{a}")
    def mulc(self, a, c):
        if c == 1.0: return a
        if c == -1.0: return self.neg(a)
        return self.emit(f"{a} * {self.sp(c)}")
    def fmac(self, a, c, b):   # a*c + b
        if c == 1.0: return self.add(a, b)
        if c == -1.0: return self.sub(b, a)
        return self.emit(f"{a} * {self.sp(c)} + {b}")
    def fmsc(self, a, c, b):   # a*c - b
        if c == 1.0: return self.sub(a, b)
        return self.emit(f"{a} * {self.sp(c)} - {b}")

# ---- complex helpers: values are (re_var, im_var) ----
def cadd(g, a, b): return (g.add(a[0], b[0]), g.add(a[1], b[1]))
def csub(g, a, b): return (g.sub(a[0], b[0]), g.sub(a[1], b[1]))

def cmul_w(g, a, w):
    c, s = w
    if s == 0.0:
        if c == 1.0:  return a
        if c == -1.0: return (g.neg(a[0]), g.neg(a[1]))
        return (g.mulc(a[0], c), g.mulc(a[1], c))
    if c == 0.0:
        if s == -1.0:  # * -i : (x+iy)(-i) = y - ix
            return (a[1], g.neg(a[0]))
        if s == 1.0:   # * i
            return (g.neg(a[1]), a[0])
        return (g.mulc(a[1], -s), g.mulc(a[0], s))
    # generic: re = x*c - y*s ; im = y*c + x*s
    re = g.emit(f"{a[0]} * {g.sp(c)} - {a[1]} * {g.sp(s)}")
    im = g.emit(f"{a[1]} * {g.sp(c)} + {a[0]} * {g.sp(s)}")
    return (re, im)

# ---- DFT codelet builders ----
def dft(g, xs):
    N = len(xs)
    if N == 1: return list(xs)
    if N == 2:
        return [cadd(g, xs[0], xs[1]), csub(g, xs[0], xs[1])]
    if N == 4:
        a = cadd(g, xs[0], xs[2]); b = csub(g, xs[0], xs[2])
        c = cadd(g, xs[1], xs[3]); d = csub(g, xs[1], xs[3])
        y0 = cadd(g, a, c); y2 = csub(g, a, c)
        # y1 = b - i d ; y3 = b + i d
        y1 = (g.add(b[0], d[1]), g.sub(b[1], d[0]))
        y3 = (g.sub(b[0], d[1]), g.add(b[1], d[0]))
        return [y0, y1, y2, y3]
    if N == 8:
        return ct(g, xs, 2, 4)
    if N == 64:
        return ct(g, xs, 8, 8)
    if N % 2 == 1:
        if N == 9 or N == 45:
            if N == 45: return pfa(g, xs, 9, 5)
            return sym_odd(g, xs)
        return sym_odd(g, xs)
    if N == 6:  return pfa(g, xs, 2, 3)
    if N == 36: return pfa(g, xs, 4, 9)
    raise ValueError(N)

def sym_odd(g, xs):
    """DFT of odd length via cos/sin half-matrices; j-major k-blocked FMAs.

    Numerically identical to accumulating P_k = x0 + sum_j c_jk u_j and
    B_k = sum_j s_jk v_j serially in j (same association order)."""
    N = len(xs); h = (N - 1) // 2
    x0 = xs[0]
    u = {}; v = {}
    for j in range(1, h + 1):
        u[j] = cadd(g, xs[j], xs[N - j])
        v[j] = csub(g, xs[j], xs[N - j])
    sr, si = x0
    for j in range(1, h + 1):
        sr = g.add(sr, u[j][0]); si = g.add(si, u[j][1])
    out = [None] * N
    out[0] = (sr, si)
    KB = 4
    ks = list(range(1, h + 1))
    for kb0 in range(0, h, KB):
        kb = ks[kb0:kb0 + KB]
        accP = {k: list(x0) for k in kb}     # P = x0 + sum c u
        accB = {k: [None, None] for k in kb}
        for j in range(1, h + 1):
            for k in kb:
                c = trig(N, j, k, 'c'); s = trig(N, j, k, 's')
                cv = g.emit(g.sp(c))
                accP[k][0] = g.emit(f"{u[j][0]} * {cv} + {accP[k][0]}")
                accP[k][1] = g.emit(f"{u[j][1]} * {cv} + {accP[k][1]}")
                sv = g.emit(g.sp(s))
                if accB[k][0] is None:
                    accB[k][0] = g.emit(f"{v[j][0]} * {sv}")
                    accB[k][1] = g.emit(f"{v[j][1]} * {sv}")
                else:
                    accB[k][0] = g.emit(f"{v[j][0]} * {sv} + {accB[k][0]}")
                    accB[k][1] = g.emit(f"{v[j][1]} * {sv} + {accB[k][1]}")
        for k in kb:
            pr, pi_ = accP[k]; br, bi = accB[k]
            out[k]     = (g.add(pr, bi), g.sub(pi_, br))
            out[N - k] = (g.sub(pr, bi), g.add(pi_, br))
    return out

def ct(g, xs, N1, N2):
    """Cooley-Tukey N=N1*N2: X[N2*k1+k2]=sum_a w_N1^{a k1} (w_N^{a k2} Y_a[k2])."""
    N = N1 * N2
    Y = []
    for a in range(N1):
        Y.append(dft(g, [xs[b * N1 + a] for b in range(N2)]))
    Z = [[None] * N2 for _ in range(N1)]
    for a in range(N1):
        for k2 in range(N2):
            Z[a][k2] = cmul_w(g, Y[a][k2], omega(N, a * k2))
    out = [None] * N
    for k2 in range(N2):
        col = dft(g, [Z[a][k2] for a in range(N1)])
        for k1 in range(N1):
            out[N2 * k1 + k2] = col[k1]
    return out

def pfa(g, xs, N1, N2):
    """Good-Thomas for coprime N1,N2 (no twiddles)."""
    N = N1 * N2
    R = []
    for n1 in range(N1):
        R.append(dft(g, [xs[(n1 * N2 + n2 * N1) % N] for n2 in range(N2)]))
    Y = [[None] * N2 for _ in range(N1)]
    for k2 in range(N2):
        col = dft(g, [R[n1][k2] for n1 in range(N1)])
        for k1 in range(N1):
            Y[k1][k2] = col[k1]
    return [Y[k % N1][k % N2] for k in range(N)]

# ---- kernel emission ----
def vt_of(V): return {8: 'v8', 4: 'v4', 2: 'v2', 1: 'v1'}[V]



def fmt_tbl_d(name, vals):
    body = ",".join(hexf(v) for v in vals)
    return f"static const double {name}[{len(vals)}] = {{{body}}};\n"

def fmt_tbl_l(name, vals):
    body = ",".join(str(v) for v in vals)
    return f"static const long {name}[{len(vals)}] = {{{body}}};\n"

def fmt_tbl_i(name, vals):
    body = ",".join(str(v) for v in vals)
    return f"static const int {name}[{len(vals)}] = {{{body}}};\n"

LOOP_TBLS = {}
def loop_kernel_tables():
    out = []
    # 64 = CT(8,8) twiddles w64^{a*k2}, a=1..7 rows, k2=0..7
    twr = []; twi = []
    for a in range(1, 8):
        for k2 in range(8):
            c, s = omega(64, a * k2)
            twr.append(c); twi.append(s)
    out.append(fmt_tbl_d("TW64R", twr))
    out.append(fmt_tbl_d("TW64I", twi))
    # 45 = PFA(9,5): input index (n1*5 + n2*9) % 45 ; output crt
    IN = [ (n1 * 5 + n2 * 9) % 45 for n1 in range(9) for n2 in range(5) ]
    cm = crt_index(9, 5)
    OUT = [ cm[(k1, k2)] for k2 in range(5) for k1 in range(9) ]
    IN2 = [ (n1 * 9 + n2 * 4) % 36 for n1 in range(4) for n2 in range(9) ]
    cm2 = crt_index(4, 9)
    OUT2 = [ cm2[(k1, k2)] for k2 in range(9) for k1 in range(4) ]
    # premultiplied offset tables for each stride context (doubles)
    def lt(name, vals, mult):
        return fmt_tbl_l(name, [v * mult * 8 for v in vals])   # byte offsets
    for P, tag in ((48*48 + 8, 'PP'), (48, 'P')):
        out.append(lt(f"IN45_{tag}", IN, P))
        out.append(lt(f"OUT45_{tag}", OUT, P))
    for P, tag in ((36*36 + 8, 'PP'), (36, 'P')):
        out.append(lt(f"IN36_{tag}", IN2, P))
        out.append(lt(f"OUT36_{tag}", OUT2, P))
    return "".join(out)

def emit_loop_kernel(L, V, fused):
    """Two-stage kernel with real loops; bounded register pressure."""
    vt = vt_of(V)
    name = f"fft{L}m_{vt}" if fused else f"fft{L}_{vt}"
    if fused and L == 64:
        args = "double* re, double* im"
        args += ", const double* cr, const double* ci"
        args += ", double* dr, double* di"
    else:
        args = "double*restrict re, double*restrict im"
        if fused:
            args += ", const double*restrict cr, const double*restrict ci"
    args += ", const long s"
    if L == 64: N1, N2, mode = 8, 8, 'ct'
    elif L == 36: N1, N2, mode = 4, 9, 'pfa'
    else: N1, N2, mode = 9, 5, 'pfa'
    if mode == 'pfa':
        args += ", const long*restrict tin, const long*restrict tout"
    lines = []
    lines.append(f"static __attribute__((always_inline)) inline void {name}({args}){{")
    lines.append(f"  {vt} bufr_[{L}], bufi_[{L}];")
    lines.append(f"  {vt} *bufr = bufr_, *bufi = bufi_;")
    lines.append('  __asm__("" : "+r"(bufr), "+r"(bufi));')

    # ---- stage A ----
    if mode == 'ct':
        # a = 0: no twiddle
        g = Gen(vt)
        xs = [ (g.emit(f"LD_{vt}(re + {b * N1}*s)"), g.emit(f"LD_{vt}(im + {b * N1}*s)")) for b in range(N2) ]
        Y = dft(g, xs)
        for k2 in range(N2):
            g.raw(f"bufr[{k2 * N1}] = {Y[k2][0]}; bufi[{k2 * N1}] = {Y[k2][1]};")
        lines.append("  {")
        lines += ["    " + l for l in g.lines]
        lines.append("  }")
        # a = 1..7 loop
        g = Gen(vt)
        xs = [ (g.emit(f"LD_{vt}(pa + {b * N1}*s)"), g.emit(f"LD_{vt}(pb + {b * N1}*s)")) for b in range(N2) ]
        Y = dft(g, xs)
        g.raw(f"bufr[{0 * N1}+a] = {Y[0][0]}; bufi[{0 * N1}+a] = {Y[0][1]};")
        for k2 in range(1, N2):
            tr = g.emit(f"SP_{vt}(twr[{k2}])")
            ti = g.emit(f"SP_{vt}(twi[{k2}])")
            zr = g.emit(f"{Y[k2][0]} * {tr} - {Y[k2][1]} * {ti}")
            zi = g.emit(f"{Y[k2][1]} * {tr} + {Y[k2][0]} * {ti}")
            g.raw(f"bufr[{k2 * N1}+a] = {zr}; bufi[{k2 * N1}+a] = {zi};")
        lines.append(f"  for (int a = 1; a < {N1}; ++a){{")
        lines.append(f"    const double* twr = TW64R + (a-1)*{N2};")
        lines.append(f"    const double* twi = TW64I + (a-1)*{N2};")
        lines.append(f"    const double* pa = re + a*s; const double* pb = im + a*s;")
        lines += ["    " + l for l in g.lines]
        lines.append("  }")
    else:
        g = Gen(vt)
        xs = [ (g.emit(f"LD_{vt}(BOFS(re, ix[{n2}]))"), g.emit(f"LD_{vt}(BOFS(im, ix[{n2}]))")) for n2 in range(N2) ]
        R = dft(g, xs)
        for k2 in range(N2):
            g.raw(f"bufr[{k2 * N1}+n1] = {R[k2][0]}; bufi[{k2 * N1}+n1] = {R[k2][1]};")
        lines.append(f"  #pragma GCC unroll {2 if L == 45 else 2}")
        lines.append(f"  for (int n1 = 0; n1 < {N1}; ++n1){{")
        lines.append(f"    const long* ix = tin + n1*{N2};")
        lines += ["    " + l for l in g.lines]
        lines.append("  }")

    # ---- stage B ----
    g = Gen(vt)
    xs = [ (g.emit(f"bufr[{a}]"), g.emit(f"bufi[{a}]")) for a in range(N1) ]
    col = dft(g, xs)
    if mode == 'ct':
        for k1 in range(N1):
            yr, yi = col[k1]
            if fused:
                g.raw(f"MAPST_{vt}(wo + {k1 * N2}*s, vo + {k1 * N2}*s, {yr}, {yi}, co + {k1 * N2}*s, do_ + {k1 * N2}*s);")
            else:
                g.raw(f"ST_{vt}(po + {k1 * N2}*s, {yr}); ST_{vt}(qo + {k1 * N2}*s, {yi});")
        lines.append(f"  for (int k2 = 0; k2 < {N2}; ++k2){{")
        lines.append(f"    bufr += {N1}; bufi += {N1};" if False else f"    ")
        lines.append(f"    double* po = re + k2*s; double* qo = im + k2*s;")
        if fused:
            lines.append(f"    const double* co = cr + k2*s; const double* do_ = ci + k2*s;")
            if L == 64:
                lines.append(f"    double* wo = dr + k2*s; double* vo = di + k2*s;")
            else:
                lines.append(f"    double* wo = po; double* vo = qo;")
        lines.append(f"    {{ {vt} *br_ = bufr + k2*{N1}, *bi_ = bufi + k2*{N1};")
        body = "\n      ".join(g.lines).replace("bufr[", "br_[").replace("bufi[", "bi_[")
        lines.append("      " + body)
        lines.append("    }")
        lines.append("  }")
    else:
        for k1 in range(N1):
            yr, yi = col[k1]
            if fused:
                g.raw(f"MAPST_{vt}(BOFSW(re, ox[{k1}]), BOFSW(im, ox[{k1}]), {yr}, {yi}, BOFS(cr, ox[{k1}]), BOFS(ci, ox[{k1}]));")
            else:
                g.raw(f"ST_{vt}(BOFSW(re, ox[{k1}]), {yr}); ST_{vt}(BOFSW(im, ox[{k1}]), {yi});")
        lines.append(f"  #pragma GCC unroll {3 if L == 36 else 1}")
        lines.append(f"  for (int k2 = 0; k2 < {N2}; ++k2){{")
        lines.append(f"    const long* ox = tout + k2*{N1};")
        lines.append(f"    {{ {vt} *br_ = bufr + k2*{N1}, *bi_ = bufi + k2*{N1};")
        body = "\n      ".join(g.lines).replace("bufr[", "br_[").replace("bufi[", "bi_[")
        lines.append("      " + body)
        lines.append("    }")
        lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"

def emit_sym_kernel(g, L, vt, fused):
    """Odd-prime DFT kernel: u/v into opaque stack buffer, then re-phase and
    im-phase matvecs with k-blocks of 3; constants as literal splats (gcc CSEs
    the ~L distinct values into registers and hoists them out of pass loops).
    j-accumulation order matches sym_odd (x0 + sum_j)."""
    h = (L - 1) // 2
    g.raw(f"{vt} ub_[{4*h}];")
    g.raw(f"{vt} *ub = ub_;")
    g.raw('__asm__("" : "+r"(ub));')
    # phase 0: u/v, X0   (ub layout: [ur h][ui h][vr h][vi h])
    x0r = g.emit(f"LD_{vt}(re + 0*s)"); x0i = g.emit(f"LD_{vt}(im + 0*s)")
    sr, si = x0r, x0i
    for j in range(1, h + 1):
        ar = g.emit(f"LD_{vt}(re + {j}*s)"); ai = g.emit(f"LD_{vt}(im + {j}*s)")
        br = g.emit(f"LD_{vt}(re + {L - j}*s)"); bi = g.emit(f"LD_{vt}(im + {L - j}*s)")
        ur = g.add(ar, br); ui = g.add(ai, bi)
        vr = g.sub(ar, br); vi = g.sub(ai, bi)
        g.raw(f"ub[{j-1}] = {ur}; ub[{h+j-1}] = {ui}; ub[{2*h+j-1}] = {vr}; ub[{3*h+j-1}] = {vi};")
        sr = g.add(sr, ur); si = g.add(si, ui)
    g.raw(f"ST_{vt}(re + 0*s, {sr}); ST_{vt}(im + 0*s, {si});")
    # phase RE: out[k].re = P.re + B.im ; out[L-k].re = P.re - B.im
    KB = 3 if L == 13 else 4
    ks = list(range(1, h + 1))
    for half in ('re', 'im'):
        for kb0 in range(0, h, KB):
            kb = ks[kb0:kb0 + KB]
            accP = {}; accB = {}
            for k in kb:
                accP[k] = x0r if half == 're' else x0i
                accB[k] = None
            for j in range(1, h + 1):
                if half == 're':
                    d = g.emit(f"ub[{j-1}]")        # ur
                    e = g.emit(f"ub[{3*h+j-1}]")    # vi
                else:
                    d = g.emit(f"ub[{h+j-1}]")      # ui
                    e = g.emit(f"ub[{2*h+j-1}]")    # vr
                for k in kb:
                    c = trig(L, j, k, 'c'); s_ = trig(L, j, k, 's')
                    accP[k] = g.emit(f"{d} * {g.sp(c)} + {accP[k]}")
                    if accB[k] is None:
                        accB[k] = g.emit(f"{e} * {g.sp(s_)}")
                    else:
                        accB[k] = g.emit(f"{e} * {g.sp(s_)} + {accB[k]}")
            for k in kb:
                if half == 're':
                    g.raw(f"ST_{vt}(re + {k}*s, {g.add(accP[k], accB[k])});")
                    g.raw(f"ST_{vt}(re + {L-k}*s, {g.sub(accP[k], accB[k])});")
                else:
                    g.raw(f"ST_{vt}(im + {k}*s, {g.sub(accP[k], accB[k])});")
                    g.raw(f"ST_{vt}(im + {L-k}*s, {g.add(accP[k], accB[k])});")
    if fused:
        for j in range(L):
            zr = g.emit(f"LD_{vt}(re + {j}*s)"); zi = g.emit(f"LD_{vt}(im + {j}*s)")
            g.raw(f"MAPST_{vt}(re + {j}*s, im + {j}*s, {zr}, {zi}, cr + {j}*s, ci + {j}*s);")

def crt_index(N1, N2):
    N = N1 * N2
    m = {}
    for k in range(N):
        m[(k % N1, k % N2)] = k
    return m

def emit_kernel(L, V, fused):
    vt = vt_of(V)
    g = Gen(vt)
    name = f"fft{L}m_{vt}" if fused else f"fft{L}_{vt}"
    if fused and L == 64:
        args = "double* re, double* im"
        args += ", const double* cr, const double* ci"
        args += ", double* dr, double* di"
    else:
        args = "double*restrict re, double*restrict im"
        if fused:
            args += ", const double*restrict cr, const double*restrict ci"
    args += ", const long s"
    pre = ""
    def load(j):
        xr = g.emit(f"LD_{vt}(re + {j}*s)")
        xi = g.emit(f"LD_{vt}(im + {j}*s)")
        return (xr, xi)
    def store(j, y):
        if fused:
            return f"MAPST_{vt}(re + {j}*s, im + {j}*s, {y[0]}, {y[1]}, cr + {j}*s, ci + {j}*s);"
        return f"ST_{vt}(re + {j}*s, {y[0]}); ST_{vt}(im + {j}*s, {y[1]});"
    st_lines = []
    if L in (13, 17, 23):
        emit_sym_kernel(g, L, vt, fused)
        body = "\n  ".join(g.lines)
        return (f"static __attribute__((always_inline)) inline void {name}({args}){{\n"
                f"  {body}\n}}\n")
    if L in (36, 45, 64):
        return emit_loop_kernel(L, V, fused)
    if False:
        # (old staged straight-line path, unused)
        if L == 64: N1, N2, mode = 8, 8, 'ct'
        elif L == 36: N1, N2, mode = 4, 9, 'pfa'
        else: N1, N2, mode = 9, 5, 'pfa'
        pre = (f"  {vt} bufr_[{L}], bufi_[{L}];\n"
               f"  {vt} *bufr = bufr_, *bufi = bufi_;\n"
               f"  __asm__(\"\" : \"+r\"(bufr), \"+r\"(bufi));\n")
        if mode == 'ct':
            for a in range(N1):
                xs = [load(b * N1 + a) for b in range(N2)]
                Y = dft(g, xs)
                for k2 in range(N2):
                    Z = cmul_w(g, Y[k2], omega(L, a * k2))
                    g.raw(f"bufr[{k2 * N1 + a}] = {Z[0]}; bufi[{k2 * N1 + a}] = {Z[1]};")
            for k2 in range(N2):
                xs = []
                for a in range(N1):
                    xr = g.emit(f"bufr[{k2 * N1 + a}]"); xi = g.emit(f"bufi[{k2 * N1 + a}]")
                    xs.append((xr, xi))
                col = dft(g, xs)
                for k1 in range(N1):
                    st_lines.append(store(N2 * k1 + k2, col[k1]))
        else:
            cm = crt_index(N1, N2)
            for n1 in range(N1):
                xs = [load((n1 * N2 + n2 * N1) % L) for n2 in range(N2)]
                R = dft(g, xs)
                for k2 in range(N2):
                    g.raw(f"bufr[{k2 * N1 + n1}] = {R[k2][0]}; bufi[{k2 * N1 + n1}] = {R[k2][1]};")
            for k2 in range(N2):
                xs = []
                for n1 in range(N1):
                    xr = g.emit(f"bufr[{k2 * N1 + n1}]"); xi = g.emit(f"bufi[{k2 * N1 + n1}]")
                    xs.append((xr, xi))
                col = dft(g, xs)
                for k1 in range(N1):
                    st_lines.append(store(cm[(k1, k2)], col[k1]))
    else:
        xs = [load(j) for j in range(L)]
        ys = dft(g, xs)
        for j in range(L):
            st_lines.append(store(j, ys[j]))
    body = "\n  ".join(g.lines)
    stores = "\n  ".join(st_lines)
    return (f"static __attribute__((always_inline)) inline void {name}({args}){{\n"
            f"{pre}  {body}\n  {stores}\n}}\n")

def emit_driver(L, V):
    """One full iteration (FFT3 + c + map) on a group of V volumes."""
    vt = vt_of(V)
    L2 = L * L
    return f"""
static void iter{L}_{vt}(double*restrict re, double*restrict im,
                         const double*restrict cr, const double*restrict ci){{
  for (int u = 0; u < {L2}; ++u)
    fft{L}_{vt}(re + u*{V}, im + u*{V}, {L2 * V});
  for (int x = 0; x < {L}; ++x) {{
    double* pr = re + (long)x*{L2 * V};
    double* pi = im + (long)x*{L2 * V};
    const double* qr = cr + (long)x*{L2 * V};
    const double* qi = ci + (long)x*{L2 * V};
    for (int z = 0; z < {L}; ++z)
      fft{L}_{vt}(pr + z*{V}, pi + z*{V}, {L * V});
    for (int y = 0; y < {L}; ++y)
      fft{L}m_{vt}(pr + y*{L * V}, pi + y*{L * V}, qr + y*{L * V}, qi + y*{L * V}, {V});
  }}
}}
"""



def emit_64_stage_kernels():
    out = []
    # twiddle tables for stage B: TW64B[k2][a] = w64^{a*k2}
    twr = []; twi = []
    for k2 in range(8):
        for a in range(8):
            c, s = omega(64, a * k2)
            twr.append(c); twi.append(s)
    out.append(fmt_tbl_d("TW64BR", twr))
    out.append(fmt_tbl_d("TW64BI", twi))
    # stage A kernel: DFT8 over 8 planes; src stride ss, dst stride sd (doubles)
    g = Gen('v8')
    xs = [(g.emit(f"LD_v8(sr + {j}*ss)"), g.emit(f"LD_v8(si + {j}*ss)")) for j in range(8)]
    ys = dft(g, xs)
    sts = []
    for j in range(8):
        sts.append(f"ST_v8(dr + {j}*sd, {ys[j][0]}); ST_v8(di + {j}*sd, {ys[j][1]});")
    body = "\n  ".join(g.lines + sts)
    out.append(
        "static __attribute__((always_inline)) inline void k64a_v8(const double*restrict sr, const double*restrict si, double*restrict dr, double*restrict di, const long ss, const long sd){\n  "
        + body + "\n}\n")
    # stage B kernel: loads 8 consecutive src planes, twiddle row tw (a=1..7), DFT8, store dst stride sd
    g = Gen('v8')
    xs = []
    for a in range(8):
        xr = g.emit(f"LD_v8(sr + {a}*ss)"); xi = g.emit(f"LD_v8(si + {a}*ss)")
        if a == 0:
            xs.append((xr, xi))
        else:
            tr = g.emit(f"SP_v8(twr[{a}])"); ti = g.emit(f"SP_v8(twi[{a}])")
            zr = g.emit(f"{xr} * {tr} - {xi} * {ti}")
            zi = g.emit(f"{xi} * {tr} + {xr} * {ti}")
            xs.append((zr, zi))
    ys = dft(g, xs)
    sts = []
    for k1 in range(8):
        sts.append(f"ST_v8(dr + {k1}*sd, {ys[k1][0]}); ST_v8(di + {k1}*sd, {ys[k1][1]});")
    body = "\n  ".join(g.lines + sts)
    out.append(
        "static __attribute__((always_inline)) inline void k64b_v8(const double*restrict sr, const double*restrict si, double*restrict dr, double*restrict di, const double*restrict twr, const double*restrict twi, const long ss, const long sd){\n  "
        + body + "\n}\n")
    return "".join(out)

def emit_wv64_fused():
    P = 64
    PITCH = 72            # row stride (doubles): avoids L1-set aliasing
    PSZ = 64 * PITCH + 8  # plane stride
    EPSZ = 8 * PSZ
    return f"""
static void transpose_pitch(double* a, long pitch);
static void transpose_copy(double*restrict dst, const double*restrict s_, long n, long pitch);
#define PITCH64 {PITCH}
#define PSZ64 {PSZ}
static double TPBUF[2*{PSZ}] __attribute__((aligned(64)));
// export one plane (64x64) from split re/im (pitched) into interleaved dst.
static void wv64_export_plane(double* d, const double* pr, const double* pi, int parity){{
  if (!parity){{
    for (long y = 0; y < 64; ++y)
      intlv(d + 2*y*64, pr + y*{PITCH}, pi + y*{PITCH}, 64);
    return;
  }}
  __m512i lo = _mm512_loadu_si512(IDX_ILO), hi = _mm512_loadu_si512(IDX_IHI);
  for (long y = 0; y < 64; y += 8)
    for (long z = 0; z < 64; z += 8){{
      v8 A[8], B[8];
      ld8x8(pr + z*{PITCH} + y, {PITCH}, A); tr8x8(A);
      ld8x8(pi + z*{PITCH} + y, {PITCH}, B); tr8x8(B);
      for (int i = 0; i < 8; ++i){{
        double* dz = d + 2*((y + i)*64 + z);
        _mm512_storeu_pd(dz,     _mm512_permutex2var_pd((__m512d)A[i], lo, (__m512d)B[i]));
        _mm512_storeu_pd(dz + 8, _mm512_permutex2var_pd((__m512d)A[i], hi, (__m512d)B[i]));
      }}
    }}
}}
// One iteration, in place on V. Group-plane layout toggles each iteration:
//   zperm=0: group k2 = phys planes (8*k2 + a) (consecutive)
//   zperm=1: group k2 = phys planes (8*a + k2) (stride 8)
// Fused stage A writes back into the consumed group slots, producing the
// opposite layout for the next iteration.
// mode: 0 = mid (fused stage A), 1 = fused + export, 2 = last (no stage A, export only)
static double TPBUF[2*{PSZ}] __attribute__((aligned(64)));
static void wv_iter64f(double*restrict Vr, double*restrict Vi,
                       double*restrict scr, double*restrict sci,
                       const double*restrict cr, const double*restrict ci,
                       int zperm, int mode, double* exdst, int exparity){{
  double* tpr = TPBUF; double* tpi = TPBUF + {PSZ};
  for (int k2 = 0; k2 < 8; ++k2){{
    double* gr = Vr + (zperm ? (long)k2*{PSZ} : (long)k2*8*{PSZ});
    double* gi = Vi + (zperm ? (long)k2*{PSZ} : (long)k2*8*{PSZ});
    const double* twr = TW64BR + k2*8;
    const double* twi = TW64BI + k2*8;
    if (k2 == 0){{
      if (zperm){{
        for (int y = 0; y < 64; ++y)
          for (int q = 0; q < 64; q += 8){{
            long u = (long)y*{PITCH} + q;
            k64a_v8(gr + u, gi + u, scr + u, sci + u, {EPSZ}, {PSZ});
          }}
      }} else {{
        for (int y = 0; y < 64; ++y)
          for (int q = 0; q < 64; q += 8){{
            long u = (long)y*{PITCH} + q;
            k64a_v8(gr + u, gi + u, scr + u, sci + u, {PSZ}, {PSZ});
          }}
      }}
    }} else {{
      if (zperm){{
        for (int y = 0; y < 64; ++y)
          for (int q = 0; q < 64; q += 8){{
            long u = (long)y*{PITCH} + q;
            k64b_v8(gr + u, gi + u, scr + u, sci + u, twr, twi, {EPSZ}, {PSZ});
          }}
      }} else {{
        for (int y = 0; y < 64; ++y)
          for (int q = 0; q < 64; q += 8){{
            long u = (long)y*{PITCH} + q;
            k64b_v8(gr + u, gi + u, scr + u, sci + u, twr, twi, {PSZ}, {PSZ});
          }}
      }}
    }}
    for (int t = 0; t < 8; ++t){{
      double* pr = scr + (long)t*{PSZ};
      double* pi = sci + (long)t*{PSZ};
      long p = 8*t + k2;
      const double* qr = cr + p*{PSZ};
      const double* qi = ci + p*{PSZ};
      for (int q = 0; q < 64; q += 8)
        fft64_v8(pr + q, pi + q, {PITCH});
      transpose_copy(tpr, pr, 64, {PITCH});
      transpose_copy(tpi, pi, 64, {PITCH});
      for (int y = 0; y < 64; y += 8)
        fft64m_v8(tpr + y, tpi + y, qr + y, qi + y, pr + y, pi + y, {PITCH});
      if (mode) wv64_export_plane(exdst + 2*p*64*64, pr, pi, exparity);
    }}
    if (mode != 2){{
      if (zperm){{
        for (int y = 0; y < 64; ++y)
          for (int q = 0; q < 64; q += 8){{
            long u = (long)y*{PITCH} + q;
            k64a_v8(scr + u, sci + u, gr + u, gi + u, {PSZ}, {EPSZ});
          }}
      }} else {{
        for (int y = 0; y < 64; ++y)
          for (int q = 0; q < 64; q += 8){{
            long u = (long)y*{PITCH} + q;
            k64a_v8(scr + u, sci + u, gr + u, gi + u, {PSZ}, {PSZ});
          }}
      }}
    }}
  }}
}}
static void wv64_stageA(double* Vr, double* Vi){{
  for (int a = 0; a < 8; ++a){{
    double* pr = Vr + (long)a*{PSZ};
    double* pi = Vi + (long)a*{PSZ};
    for (int y = 0; y < 64; ++y)
      for (int q = 0; q < 64; q += 8){{
        long u = y*{PITCH} + q;
        k64a_v8(pr + u, pi + u, pr + u, pi + u, {8*PSZ}, {8*PSZ});
      }}
  }}
}}
// 64x64 logical transpose within pitched rows (pitch multiple of 8)
static void transpose_pitch(double* a, long pitch){{
  for (long bi = 0; bi < 64; bi += 8){{
    v8 A[8], B[8];
    ld8x8(a + bi*pitch + bi, pitch, A); tr8x8(A); st8x8(a + bi*pitch + bi, pitch, A);
    for (long bj = bi + 8; bj < 64; bj += 8){{
      ld8x8(a + bi*pitch + bj, pitch, A); ld8x8(a + bj*pitch + bi, pitch, B);
      tr8x8(A); tr8x8(B);
      st8x8(a + bi*pitch + bj, pitch, B); st8x8(a + bj*pitch + bi, pitch, A);
    }}
  }}
}}
// out-of-place transpose: dst[j][i] = src[i][j], n x n, same pitch both
static void transpose_copy(double*restrict dst, const double*restrict s_, long n, long pitch){{
  for (long bi = 0; bi < n; bi += 8)
    for (long bj = 0; bj < n; bj += 8){{
      v8 A[8];
      ld8x8(s_ + bi*pitch + bj, pitch, A); tr8x8(A);
      st8x8(dst + bj*pitch + bi, pitch, A);
    }}
}}
static void wv64_export(double* dst, const double* re, const double* im, int parity){{
  if (!parity){{
    for (long x = 0; x < 64; ++x)
      for (long y = 0; y < 64; ++y)
        intlv(dst + 2*(x*64 + y)*64, re + x*{PSZ} + y*{PITCH}, im + x*{PSZ} + y*{PITCH}, 64);
    return;
  }}
  __m512i lo = _mm512_loadu_si512(IDX_ILO), hi = _mm512_loadu_si512(IDX_IHI);
  for (long x = 0; x < 64; ++x){{
    const double* pr = re + x*{PSZ};
    const double* pi = im + x*{PSZ};
    double* d = dst + 2*x*64*64;
    for (long y = 0; y < 64; y += 8)
      for (long z = 0; z < 64; z += 8){{
        v8 A[8], B[8];
        ld8x8(pr + z*{PITCH} + y, {PITCH}, A); tr8x8(A);
        ld8x8(pi + z*{PITCH} + y, {PITCH}, B); tr8x8(B);
        for (int i = 0; i < 8; ++i){{
          double* dz = d + 2*((y + i)*64 + z);
          _mm512_storeu_pd(dz,     _mm512_permutex2var_pd((__m512d)A[i], lo, (__m512d)B[i]));
          _mm512_storeu_pd(dz + 8, _mm512_permutex2var_pd((__m512d)A[i], hi, (__m512d)B[i]));
        }}
      }}
  }}
}}
static void run_wv64(long B, long m, const double* x0, const double* c,
                     double* out1, double* outm){{
  long n = (long)64*64*64;
  size_t S = (size_t)64*{PSZ} + 64;
  double* Ur = get_arena((8*S) * sizeof(double));
  double* Ui = Ur + S;
  double* c0r = Ur + 2*S; double* c0i = Ur + 3*S;
  double* c1r = Ur + 4*S; double* c1i = Ur + 5*S;
  double* s0r = Ur + 6*S; double* s0i = Ur + 7*S;
  for (long b = 0; b < B; ++b){{
    const double* xv = x0 + b*2*n;
    const double* cv = c  + b*2*n;
    for (int a = 0; a < 8; ++a){{
      for (int j = 0; j < 8; ++j){{
        long x = 8*j + a;
        for (long y = 0; y < 64; ++y)
          deint(xv + 2*(x*64 + y)*64, Ur + x*{PSZ} + y*{PITCH}, Ui + x*{PSZ} + y*{PITCH}, 64);
      }}
      double* pr = Ur + (long)a*{PSZ};
      double* pi = Ui + (long)a*{PSZ};
      for (int y = 0; y < 64; ++y)
        for (int q = 0; q < 64; q += 8){{
          long u = y*{PITCH} + q;
          k64a_v8(pr + u, pi + u, pr + u, pi + u, {EPSZ}, {EPSZ});
        }}
    }}
    for (long x = 0; x < 64; ++x){{
      for (long y = 0; y < 64; ++y)
        deint(cv + 2*(x*64 + y)*64, c0r + x*{PSZ} + y*{PITCH}, c0i + x*{PSZ} + y*{PITCH}, 64);
      transpose_copy(c1r + x*{PSZ}, c0r + x*{PSZ}, 64, {PITCH});
      transpose_copy(c1i + x*{PSZ}, c0i + x*{PSZ}, 64, {PITCH});
    }}
    for (long k = 0; k < m; ++k){{
      int mode = (k == m-1) ? 2 : ((k == 0) ? 1 : 0);
      double* exdst = (k == m-1) ? (outm + b*2*n) : (out1 + b*2*n);
      if (m == 1) exdst = out1 + b*2*n;
      wv_iter64f(Ur, Ui, s0r, s0i,
                 (k % 2 == 0) ? c1r : c0r, (k % 2 == 0) ? c1i : c0i,
                 (int)(k % 2), mode, exdst, (int)((k + 1) % 2));
    }}
    if (m == 1) memcpy(outm + b*2*n, out1 + b*2*n, (size_t)2*n*sizeof(double));
  }}
}}
"""

def emit_wv_driver(L, P):
    """Within-volume iteration: volume stored L x P x P (z,y padded to P)."""
    P2 = P * P + 8   # plane stride (padded to de-alias L1 sets)
    tb_pp = f", IN{L}_PP, OUT{L}_PP" if L in (36, 45) else ""
    tb_p = f", IN{L}_P, OUT{L}_P" if L in (36, 45) else ""
    mid_calls = []
    q = 0
    while q + 8 <= P:
        mid_calls.append((8, q)); q += 8
    if q + 4 <= P:
        mid_calls.append((4, q)); q += 4
    assert q == P, (L, P)
    mids_plain = "\n    ".join(
        f"fft{L}_v{w}(pr + {off}, pi + {off}, {P}{tb_p});" for (w, off) in mid_calls)
    mids_map = "\n    ".join(
        f"fft{L}m_v{w}(pr + {off}, pi + {off}, qr + {off}, qi + {off}, {P}{tb_p});" for (w, off) in mid_calls)
    return f"""
static void wv_iter{L}(double*restrict re, double*restrict im,
                       const double*restrict cr, const double*restrict ci){{
  for (int u = 0; u < {L} * {P}; u += 8)
    fft{L}_v8(re + u, im + u, {P2}{tb_pp});
  for (int x = 0; x < {L}; ++x) {{
    double* pr = re + (long)x*{P2};
    double* pi = im + (long)x*{P2};
    const double* qr = cr + (long)x*{P2};
    const double* qi = ci + (long)x*{P2};
    {mids_plain}
    transpose_sq(pr, {P});
    transpose_sq(pi, {P});
    {mids_map}
  }}
}}
"""

HEADER = r"""
// AUTO-GENERATED by gen.py -- specialized batched 3D FFT + nonlinear map.
#include <immintrin.h>
#include <sys/mman.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

typedef double v8 __attribute__((vector_size(64)));
typedef double v4 __attribute__((vector_size(32)));

static inline v8 ld8(const double* p){ v8 v; __builtin_memcpy(&v, p, 64); return v; }
static inline void st8(double* p, v8 v){ __builtin_memcpy(p, &v, 64); }
static inline v4 ld4(const double* p){ v4 v; __builtin_memcpy(&v, p, 32); return v; }
static inline void st4(double* p, v4 v){ __builtin_memcpy(p, &v, 32); }
#define LD_v8(p) ld8(p)
#define ST_v8(p, v) st8((p), (v))
#define LD_v4(p) ld4(p)
#define ST_v4(p, v) st4((p), (v))
#define BOFS(p, b) ((const double*)((const char*)(p) + (b)))
#define BOFSW(p, b) ((double*)((char*)(p) + (b)))
#define SP_v8(c) ((v8){(c),(c),(c),(c),(c),(c),(c),(c)})
#define SP_v4(c) ((v4){(c),(c),(c),(c)})

// ---- nonlinear map: given z (post-FFT) and c, store (z+c)/(1+|z+c|) ----
// Uses rsqrt14/rcp14 + Newton; ~2 ulp accurate, branch-free.
static __attribute__((always_inline)) inline v8 vsqrt_nr8(v8 s){
  v8 r = (v8)_mm512_rsqrt14_pd((__m512d)s);
  v8 hs = s * SP_v8(-0.5);
  r = r * (hs*r*r + SP_v8(1.5));
  r = r * (hs*r*r + SP_v8(1.5));
  return s * r;  // sqrt(s)
}
static __attribute__((always_inline)) inline v8 vrcp_nr8(v8 d){
  v8 w = (v8)_mm512_rcp14_pd((__m512d)d);
  w = w * (SP_v8(2.0) - d*w);
  w = w * (SP_v8(2.0) - d*w);
  return w;
}
static __attribute__((always_inline)) inline void mapst8(double* pr, double* pi, v8 zr, v8 zi,
                                                         const double* cr, const double* ci){
  zr = zr + ld8(cr); zi = zi + ld8(ci);
  v8 s = zr*zr + zi*zi;
  v8 m = vsqrt_nr8(s);
  v8 w = (v8)_mm512_div_pd((__m512d)SP_v8(1.0), (__m512d)(SP_v8(1.0) + m));
  st8(pr, zr*w); st8(pi, zi*w);
}
static __attribute__((always_inline)) inline v4 vsqrt_nr4(v4 s){
  v4 r = (v4)_mm256_rsqrt14_pd((__m256d)s);
  v4 hs = s * SP_v4(-0.5);
  r = r * (hs*r*r + SP_v4(1.5));
  r = r * (hs*r*r + SP_v4(1.5));
  return s * r;
}
static __attribute__((always_inline)) inline v4 vrcp_nr4(v4 d){
  v4 w = (v4)_mm256_rcp14_pd((__m256d)d);
  w = w * (SP_v4(2.0) - d*w);
  w = w * (SP_v4(2.0) - d*w);
  return w;
}
static __attribute__((always_inline)) inline void mapst4(double* pr, double* pi, v4 zr, v4 zi,
                                                         const double* cr, const double* ci){
  zr = zr + ld4(cr); zi = zi + ld4(ci);
  v4 s = zr*zr + zi*zi;
  v4 m = vsqrt_nr4(s);
  v4 w = (v4)_mm256_div_pd((__m256d)SP_v4(1.0), (__m256d)(SP_v4(1.0) + m));
  st4(pr, zr*w); st4(pi, zi*w);
}
#define MAPST_v8(pr, pi, yr, yi, cr, ci) mapst8((pr), (pi), (yr), (yi), (cr), (ci))
#define MAPST_v4(pr, pi, yr, yi, cr, ci) mapst4((pr), (pi), (yr), (yi), (cr), (ci))
"""

RUNTIME = r"""
// ---------------- runtime glue ----------------
static double* arena = 0;
static size_t arena_sz = 0;
static double* get_arena(size_t bytes){
  if (bytes > arena_sz){
    if (arena) free(arena);
    size_t sz = (bytes + (2u<<20) - 1) & ~(size_t)((2u<<20) - 1);
    arena = (double*)aligned_alloc(2u<<20, sz);
    madvise(arena, sz, MADV_HUGEPAGE);
    memset(arena, 0, sz);
    arena_sz = bytes;
  }
  return arena;
}

// ---- in-register double transpose helpers ----
static __attribute__((always_inline)) inline void t4x4(const double* a, long S, v4* c0, v4* c1, v4* c2, v4* c3){
  v4 r0 = ld4(a), r1 = ld4(a + S), r2 = ld4(a + 2*S), r3 = ld4(a + 3*S);
  v4 u0 = __builtin_shufflevector(r0, r1, 0, 4, 2, 6);
  v4 u1 = __builtin_shufflevector(r0, r1, 1, 5, 3, 7);
  v4 u2 = __builtin_shufflevector(r2, r3, 0, 4, 2, 6);
  v4 u3 = __builtin_shufflevector(r2, r3, 1, 5, 3, 7);
  *c0 = __builtin_shufflevector(u0, u2, 0, 1, 4, 5);
  *c1 = __builtin_shufflevector(u1, u3, 0, 1, 4, 5);
  *c2 = __builtin_shufflevector(u0, u2, 2, 3, 6, 7);
  *c3 = __builtin_shufflevector(u1, u3, 2, 3, 6, 7);
}
static __attribute__((always_inline)) inline void st4x4(double* a, long S, v4 c0, v4 c1, v4 c2, v4 c3){
  st4(a, c0); st4(a + S, c1); st4(a + 2*S, c2); st4(a + 3*S, c3);
}
#define SH64(a,b,imm) ((v8)_mm512_shuffle_f64x2((__m512d)(a),(__m512d)(b),(imm)))
#define UPLO(a,b) ((v8)_mm512_unpacklo_pd((__m512d)(a),(__m512d)(b)))
#define UPHI(a,b) ((v8)_mm512_unpackhi_pd((__m512d)(a),(__m512d)(b)))
static __attribute__((always_inline)) inline void tr8x8(v8 r[8]){
  v8 t0=UPLO(r[0],r[1]), t1=UPHI(r[0],r[1]), t2=UPLO(r[2],r[3]), t3=UPHI(r[2],r[3]);
  v8 t4=UPLO(r[4],r[5]), t5=UPHI(r[4],r[5]), t6=UPLO(r[6],r[7]), t7=UPHI(r[6],r[7]);
  v8 v0=SH64(t0,t2,0x88), v1=SH64(t1,t3,0x88), v2=SH64(t0,t2,0xDD), v3=SH64(t1,t3,0xDD);
  v8 v4_=SH64(t4,t6,0x88), v5=SH64(t5,t7,0x88), v6=SH64(t4,t6,0xDD), v7=SH64(t5,t7,0xDD);
  r[0]=SH64(v0,v4_,0x88); r[4]=SH64(v0,v4_,0xDD);
  r[1]=SH64(v1,v5,0x88);  r[5]=SH64(v1,v5,0xDD);
  r[2]=SH64(v2,v6,0x88);  r[6]=SH64(v2,v6,0xDD);
  r[3]=SH64(v3,v7,0x88);  r[7]=SH64(v3,v7,0xDD);
}
static __attribute__((always_inline)) inline void ld8x8(const double* a, long S, v8 r[8]){
  for (int i = 0; i < 8; ++i) r[i] = ld8(a + i*S);
}
static __attribute__((always_inline)) inline void st8x8(double* a, long S, const v8 r[8]){
  for (int i = 0; i < 8; ++i) st8(a + i*S, r[i]);
}
static void transpose_sq(double* a, long P){
  if ((P & 7) == 0){
    for (long bi = 0; bi < P; bi += 8){
      v8 A[8], B[8];
      ld8x8(a + bi*P + bi, P, A); tr8x8(A); st8x8(a + bi*P + bi, P, A);
      for (long bj = bi + 8; bj < P; bj += 8){
        ld8x8(a + bi*P + bj, P, A); ld8x8(a + bj*P + bi, P, B);
        tr8x8(A); tr8x8(B);
        st8x8(a + bi*P + bj, P, B); st8x8(a + bj*P + bi, P, A);
      }
    }
  } else {
    long P8 = P & ~7L;
    for (long bi = 0; bi < P8; bi += 8){
      v8 A[8], B[8];
      ld8x8(a + bi*P + bi, P, A); tr8x8(A); st8x8(a + bi*P + bi, P, A);
      for (long bj = bi + 8; bj < P8; bj += 8){
        ld8x8(a + bi*P + bj, P, A); ld8x8(a + bj*P + bi, P, B);
        tr8x8(A); tr8x8(B);
        st8x8(a + bi*P + bj, P, B); st8x8(a + bj*P + bi, P, A);
      }
    }
    for (long bi = P8; bi < P; bi += 4){
      double* d = a + bi*P + bi;
      v4 c0,c1,c2,c3; t4x4(d, P, &c0,&c1,&c2,&c3); st4x4(d, P, c0,c1,c2,c3);
      for (long bj = bi + 4; bj < P; bj += 4){
        double* pa = a + bi*P + bj;
        double* pb = a + bj*P + bi;
        v4 a0,a1,a2,a3,b0,b1,b2,b3;
        t4x4(pa, P, &a0,&a1,&a2,&a3);
        t4x4(pb, P, &b0,&b1,&b2,&b3);
        st4x4(pa, P, b0,b1,b2,b3);
        st4x4(pb, P, a0,a1,a2,a3);
      }
    }
    for (long bi = 0; bi < P8; bi += 4)
      for (long bj = P8; bj < P; bj += 4){
        double* pa = a + bi*P + bj;
        double* pb = a + bj*P + bi;
        v4 a0,a1,a2,a3,b0,b1,b2,b3;
        t4x4(pa, P, &a0,&a1,&a2,&a3);
        t4x4(pb, P, &b0,&b1,&b2,&b3);
        st4x4(pa, P, b0,b1,b2,b3);
        st4x4(pb, P, a0,a1,a2,a3);
      }
  }
}
// ---- interleave/deinterleave (complex <-> split) ----
static const long long IDX_EVEN[8] = {0,2,4,6,8,10,12,14};
static const long long IDX_ODD[8]  = {1,3,5,7,9,11,13,15};
static const long long IDX_ILO[8]  = {0,8,1,9,2,10,3,11};
static const long long IDX_IHI[8]  = {4,12,5,13,6,14,7,15};
static void deint(const double* z, double* re, double* im, long n){
  __m512i ev = _mm512_loadu_si512(IDX_EVEN), od = _mm512_loadu_si512(IDX_ODD);
  long i = 0;
  for (; i + 8 <= n; i += 8){
    __m512d z0 = _mm512_loadu_pd(z + 2*i), z1 = _mm512_loadu_pd(z + 2*i + 8);
    _mm512_storeu_pd(re + i, _mm512_permutex2var_pd(z0, ev, z1));
    _mm512_storeu_pd(im + i, _mm512_permutex2var_pd(z0, od, z1));
  }
  if (i < n){
    long r = n - i;                       // 1..7
    __mmask8 mlo = (__mmask8)((2*r >= 8) ? 0xFF : ((1u << (2*r)) - 1));
    __mmask8 mhi = (__mmask8)((2*r <= 8) ? 0 : ((1u << (2*r - 8)) - 1));
    __m512d z0 = _mm512_maskz_loadu_pd(mlo, z + 2*i);
    __m512d z1 = _mm512_maskz_loadu_pd(mhi, z + 2*i + 8);
    __mmask8 mo = (__mmask8)((1u << r) - 1);
    _mm512_mask_storeu_pd(re + i, mo, _mm512_permutex2var_pd(z0, ev, z1));
    _mm512_mask_storeu_pd(im + i, mo, _mm512_permutex2var_pd(z0, od, z1));
  }
}
static void intlv(double* z, const double* re, const double* im, long n){
  __m512i lo = _mm512_loadu_si512(IDX_ILO), hi = _mm512_loadu_si512(IDX_IHI);
  long i = 0;
  for (; i + 8 <= n; i += 8){
    __m512d r = _mm512_loadu_pd(re + i), m = _mm512_loadu_pd(im + i);
    _mm512_storeu_pd(z + 2*i,     _mm512_permutex2var_pd(r, lo, m));
    _mm512_storeu_pd(z + 2*i + 8, _mm512_permutex2var_pd(r, hi, m));
  }
  if (i < n){
    long t = n - i;
    __mmask8 mi = (__mmask8)((1u << t) - 1);
    __m512d r = _mm512_maskz_loadu_pd(mi, re + i), m = _mm512_maskz_loadu_pd(mi, im + i);
    __mmask8 mlo = (__mmask8)((2*t >= 8) ? 0xFF : ((1u << (2*t)) - 1));
    __mmask8 mhi = (__mmask8)((2*t <= 8) ? 0 : ((1u << (2*t - 8)) - 1));
    _mm512_mask_storeu_pd(z + 2*i, mlo, _mm512_permutex2var_pd(r, lo, m));
    _mm512_mask_storeu_pd(z + 2*i + 8, mhi, _mm512_permutex2var_pd(r, hi, m));
  }
}
// out-of-place square transpose, any n multiple of 4, row stride = pitch
static void transpose_copy_sq(double*restrict dst, const double*restrict s_, long n, long pitch){
  long n8 = n & ~7L;
  for (long bi = 0; bi < n8; bi += 8)
    for (long bj = 0; bj < n8; bj += 8){
      v8 A[8];
      ld8x8(s_ + bi*pitch + bj, pitch, A); tr8x8(A);
      st8x8(dst + bj*pitch + bi, pitch, A);
    }
  for (long bi = 0; bi < n; bi += 4)
    for (long bj = (bi < n8 ? n8 : 0); bj < n; bj += 4){
      v4 c0,c1,c2,c3;
      t4x4(s_ + bi*pitch + bj, pitch, &c0,&c1,&c2,&c3);
      st4x4(dst + bj*pitch + bi, pitch, c0,c1,c2,c3);
    }
}

// ---- batch-major group import/export ----
static void import_lane(const double* z, double* re, double* im, long n, int V, int lane){
  for (long i = 0; i < n; ++i){ re[i*V + lane] = z[2*i]; im[i*V + lane] = z[2*i+1]; }
}
static void fill_lane(double* re, double* im, long n, int V, int lane, double vr, double vi){
  for (long i = 0; i < n; ++i){ re[i*V + lane] = vr; im[i*V + lane] = vi; }
}
static void export_lane(double* z, const double* re, const double* im, long n, int V, int lane){
  for (long i = 0; i < n; ++i){ z[2*i] = re[i*V + lane]; z[2*i+1] = im[i*V + lane]; }
}
static void import_group8(const double* const src[8], double* re, double* im, long n){
  long i = 0;
  for (; i + 4 <= n; i += 4){
    v8 R[8];
    for (int l = 0; l < 8; ++l) R[l] = ld8(src[l] + 2*i);
    tr8x8(R);
    for (int j = 0; j < 4; ++j){ st8(re + (i+j)*8, R[2*j]); st8(im + (i+j)*8, R[2*j+1]); }
  }
  for (; i < n; ++i)
    for (int l = 0; l < 8; ++l){ re[i*8 + l] = src[l][2*i]; im[i*8 + l] = src[l][2*i+1]; }
}
static void export_group8(double* const dst[8], const double* re, const double* im, long n, int alive){
  long i = 0;
  for (; i + 4 <= n; i += 4){
    v8 R[8];
    for (int j = 0; j < 4; ++j){ R[2*j] = ld8(re + (i+j)*8); R[2*j+1] = ld8(im + (i+j)*8); }
    tr8x8(R);
    for (int l = 0; l < alive; ++l) st8(dst[l] + 2*i, R[l]);
  }
  for (; i < n; ++i)
    for (int l = 0; l < alive; ++l){ dst[l][2*i] = re[i*8 + l]; dst[l][2*i+1] = im[i*8 + l]; }
}

typedef void (*iter_fn)(double*restrict, double*restrict, const double*restrict, const double*restrict);

static void run_groups(long L, long B, long m, const double* x0, const double* c,
                       double* out1, double* outm, iter_fn f8, iter_fn f4){
  long n = L*L*L;
  long done = 0;
  while (done < B){
    long rem = B - done;
    int V = (rem >= 5) ? 8 : 4;
    iter_fn f = (V == 8) ? f8 : f4;
    int alive = rem < V ? (int)rem : V;
    size_t comp = (size_t)n * V;
    double* re = get_arena(4 * comp * sizeof(double));
    double* im = re + comp;
    double* cr = re + 2 * comp;
    double* ci = re + 3 * comp;
    if (V == 8){
      const double* srcx[8]; const double* srcc[8];
      for (int l = 0; l < 8; ++l){
        srcx[l] = x0 + (done + (l < alive ? l : alive - 1)) * 2 * n;
        srcc[l] = c  + (done + (l < alive ? l : alive - 1)) * 2 * n;
      }
      import_group8(srcx, re, im, n);
      import_group8(srcc, cr, ci, n);
      for (int l = alive; l < 8; ++l){
        fill_lane(re, im, n, 8, l, 0.0, 0.0);
        fill_lane(cr, ci, n, 8, l, 1.0, 0.0);
      }
    } else {
      for (int l = 0; l < V; ++l){
        if (l < alive){
          import_lane(x0 + (done + l) * 2 * n, re, im, n, V, l);
          import_lane(c  + (done + l) * 2 * n, cr, ci, n, V, l);
        } else {
          fill_lane(re, im, n, V, l, 0.0, 0.0);
          fill_lane(cr, ci, n, V, l, 1.0, 0.0);
        }
      }
    }
    f(re, im, cr, ci);
    if (V == 8){
      double* d1[8]; for (int l = 0; l < 8; ++l) d1[l] = out1 + (done + (l < alive ? l : 0)) * 2 * n;
      export_group8(d1, re, im, n, alive);
    } else {
      for (int l = 0; l < alive; ++l)
        export_lane(out1 + (done + l) * 2 * n, re, im, n, V, l);
    }
    for (long k = 1; k < m; ++k) f(re, im, cr, ci);
    if (m > 1){
      if (V == 8){
        double* dm[8]; for (int l = 0; l < 8; ++l) dm[l] = outm + (done + (l < alive ? l : 0)) * 2 * n;
        export_group8(dm, re, im, n, alive);
      } else {
        for (int l = 0; l < alive; ++l)
          export_lane(outm + (done + l) * 2 * n, re, im, n, V, l);
      }
    } else {
      memcpy(outm + done * 2 * n, out1 + done * 2 * n, (size_t)alive * 2 * n * sizeof(double));
    }
    done += alive;
  }
}

// ---- within-volume driver glue ----
typedef void (*wv_fn)(double*restrict, double*restrict, const double*restrict, const double*restrict);

static void wv_export(double* dst, const double* re, const double* im, long L, long P, int parity){
  long PS = P*P + 8;
  if (!parity){
    for (long x = 0; x < L; ++x)
      for (long y = 0; y < L; ++y)
        intlv(dst + 2*(x*L + y)*L, re + x*PS + y*P, im + x*PS + y*P, L);
    return;
  }
  __m512i lo = _mm512_loadu_si512(IDX_ILO), hi = _mm512_loadu_si512(IDX_IHI);
  for (long x = 0; x < L; ++x){
    const double* pr = re + x*PS;
    const double* pi = im + x*PS;
    double* d = dst + 2*x*L*L;
    long y = 0;
    for (; y + 8 <= L; y += 8){
      long z = 0;
      for (; z + 8 <= L; z += 8){
        v8 A[8], B[8];
        ld8x8(pr + z*P + y, P, A); tr8x8(A);
        ld8x8(pi + z*P + y, P, B); tr8x8(B);
        for (int i = 0; i < 8; ++i){
          double* dz = d + 2*((y + i)*L + z);
          _mm512_storeu_pd(dz,     _mm512_permutex2var_pd((__m512d)A[i], lo, (__m512d)B[i]));
          _mm512_storeu_pd(dz + 8, _mm512_permutex2var_pd((__m512d)A[i], hi, (__m512d)B[i]));
        }
      }
      for (; z < L; ++z)
        for (int i = 0; i < 8; ++i){
          d[2*((y+i)*L + z)]     = pr[z*P + y + i];
          d[2*((y+i)*L + z) + 1] = pi[z*P + y + i];
        }
    }
    for (; y < L; ++y)
      for (long z = 0; z < L; ++z){
        d[2*(y*L + z)]     = pr[z*P + y];
        d[2*(y*L + z) + 1] = pi[z*P + y];
      }
  }
}

static void run_wv(long L, long P, long B, long m, const double* x0, const double* c,
                   double* out1, double* outm, wv_fn iter){
  long n = L*L*L;
  long PS = P*P + 8;
  size_t S = (size_t)L*PS;
  double* re  = get_arena(6 * S * sizeof(double));
  double* im  = re + S;
  double* c0r = re + 2*S; double* c0i = re + 3*S;
  double* c1r = re + 4*S; double* c1i = re + 5*S;
  for (long b = 0; b < B; ++b){
    const double* xv = x0 + b*2*n;
    const double* cv = c  + b*2*n;
    if (b == 0){
      for (size_t i = 0; i < S; ++i){ re[i] = 0.0; im[i] = 0.0; c0r[i] = 1.0; c0i[i] = 0.0; }
    }
    for (long x = 0; x < L; ++x){
      for (long y = 0; y < L; ++y){
        deint(xv + 2*(x*L + y)*L, re  + x*PS + y*P, im  + x*PS + y*P, L);
        deint(cv + 2*(x*L + y)*L, c0r + x*PS + y*P, c0i + x*PS + y*P, L);
      }
      transpose_copy_sq(c1r + x*PS, c0r + x*PS, P, P);
      transpose_copy_sq(c1i + x*PS, c0i + x*PS, P, P);
    }
    for (long k = 0; k < m; ++k){
      iter(re, im, (k % 2 == 0) ? c1r : c0r, (k % 2 == 0) ? c1i : c0i);
      if (k == 0) wv_export(out1 + b*2*n, re, im, L, P, 1);
    }
    if (m > 1) wv_export(outm + b*2*n, re, im, L, P, m % 2);
    else memcpy(outm + b*2*n, out1 + b*2*n, (size_t)2*n*sizeof(double));
  }
}
"""

def main():
    parts = [HEADER]
    parts.append(loop_kernel_tables())
    wv_set = {23, 36, 45, 64}
    for L in SIZES:
        for V in (8, 4):
            parts.append(emit_kernel(L, V, fused=False))
            parts.append(emit_kernel(L, V, fused=True))
            if L not in wv_set:
                parts.append(emit_driver(L, V))
    parts.append(RUNTIME)
    for L, P in WV:
        parts.append(emit_wv_driver(L, P))
    parts.append(emit_64_stage_kernels())
    parts.append(emit_wv64_fused())
    # dispatcher
    disp = ["void run(long L, long B, long m, const double* x0, const double* c, double* out1, double* outm){"]
    disp.append("  unsigned int oldcsr = _mm_getcsr(); _mm_setcsr(oldcsr | 0x8040);  // FTZ+DAZ")
    disp.append("  switch(L){")
    wvd = dict(WV)
    for L in SIZES:
        if L == 64:
            disp.append("    case 64: run_wv64(B, m, x0, c, out1, outm); break;")
        elif L in wvd:
            disp.append(f"    case {L}: run_wv({L}, {wvd[L]}, B, m, x0, c, out1, outm, wv_iter{L}); break;")
        else:
            disp.append(f"    case {L}: run_groups({L}, B, m, x0, c, out1, outm, iter{L}_v8, iter{L}_v4); break;")
    disp.append("  }")
    disp.append("  _mm_setcsr(oldcsr);")
    disp.append("}")
    parts.append("\n".join(disp) + "\n")
    # test wrappers (tiny; used by dev harness, harmless in prod)
    tw = []
    for L in SIZES:
        if L in (36, 45):
            tw.append(f"static long TIN{L}[{L}], TOUT{L}[{L}];")
            tw.append(f"void test_fft{L}_v8(double* re, double* im, long s){{ for(int i=0;i<{L};++i){{TIN{L}[i]=IN{L}_P[i]/{36 if L==36 else 48}/8*s*8; TOUT{L}[i]=OUT{L}_P[i]/{36 if L==36 else 48}/8*s*8;}} fft{L}_v8(re, im, s, TIN{L}, TOUT{L}); }}")
            tw.append(f"void test_fft{L}_v4(double* re, double* im, long s){{ for(int i=0;i<{L};++i){{TIN{L}[i]=IN{L}_P[i]/{36 if L==36 else 48}/8*s*8; TOUT{L}[i]=OUT{L}_P[i]/{36 if L==36 else 48}/8*s*8;}} fft{L}_v4(re, im, s, TIN{L}, TOUT{L}); }}")
        else:
            tw.append(f"void test_fft{L}_v8(double* re, double* im, long s){{ fft{L}_v8(re, im, s); }}")
            tw.append(f"void test_fft{L}_v4(double* re, double* im, long s){{ fft{L}_v4(re, im, s); }}")
    parts.append("\n".join(tw) + "\n")
    src = "\n".join(parts)
    with open('/workdir/implementation.c', 'w') as f:
        f.write(src)
    print(f"wrote implementation.c: {len(src)} bytes, {src.count(chr(10))} lines")

if __name__ == '__main__':
    main()
