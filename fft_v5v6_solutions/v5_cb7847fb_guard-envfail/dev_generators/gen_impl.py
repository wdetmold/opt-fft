#!/usr/bin/env python3
"""Generates implementation.c: specialized batched 3D FFT + map iteration."""
import numpy as np, math

SIZES = (6, 8, 13, 17, 23, 36, 45, 64)
LD = np.longdouble
TWO_PI = LD('6.283185307179586476925286766559005768394')

def tw(num, den):
    """exact-ish double (cos, sin) of -2*pi*num/den, num reduced mod den."""
    t = num % den
    if t > den/2: t -= den          # symmetric reduction
    ang = -TWO_PI * LD(t) / LD(den)
    c = float(np.cos(ang)); s = float(np.sin(ang))
    # snap exact values
    def snap(v):
        for e in (0.0, 1.0, -1.0, 0.5, -0.5):
            if abs(v - e) < 1e-17: return e
        return v
    return snap(c), snap(s)

class E:
    """emitter for straight-line width-agnostic vector code"""
    def __init__(self, pfx):
        self.lines = []; self.n = 0; self.consts = {}; self.pfx = pfx
    def v(self, expr):
        self.n += 1; name = f"{self.pfx}t{self.n}"
        self.lines.append(f"VDT {name} = {expr};")
        return name
    def K(self, val):
        val = float(val)
        key = val.hex()
        if key not in self.consts:
            self.consts[key] = (f"{self.pfx}k{len(self.consts)}", val)
        return self.consts[key][0]
    def const_decls(self):
        return [f"const VDT {nm} = VSET1({v.hex()});" for nm, v in self.consts.values()]

# ---------------- DFT emitters (value-level, return list of (re,im) names) ---

def dft2(e, xs):
    (a, b), (c, d) = xs
    return [(e.v(f"{a} + {c}"), e.v(f"{b} + {d}")),
            (e.v(f"{a} - {c}"), e.v(f"{b} - {d}"))]

def dft4(e, xs):
    (x0r,x0i),(x1r,x1i),(x2r,x2i),(x3r,x3i) = xs
    a_r = e.v(f"{x0r} + {x2r}"); a_i = e.v(f"{x0i} + {x2i}")
    b_r = e.v(f"{x0r} - {x2r}"); b_i = e.v(f"{x0i} - {x2i}")
    c_r = e.v(f"{x1r} + {x3r}"); c_i = e.v(f"{x1i} + {x3i}")
    d_r = e.v(f"{x1r} - {x3r}"); d_i = e.v(f"{x1i} - {x3i}")
    return [(e.v(f"{a_r} + {c_r}"), e.v(f"{a_i} + {c_i}")),
            (e.v(f"{b_r} + {d_i}"), e.v(f"{b_i} - {d_r}")),   # X1 = b - i d
            (e.v(f"{a_r} - {c_r}"), e.v(f"{a_i} - {c_i}")),
            (e.v(f"{b_r} - {d_i}"), e.v(f"{b_i} + {d_r}"))]   # X3 = b + i d

def dft8(e, xs):
    Ev = dft4(e, [xs[0], xs[2], xs[4], xs[6]])
    Ov = dft4(e, [xs[1], xs[3], xs[5], xs[7]])
    r = e.K(math.sqrt(0.5))
    X = [None]*8
    # k=0
    X[0] = (e.v(f"{Ev[0][0]} + {Ov[0][0]}"), e.v(f"{Ev[0][1]} + {Ov[0][1]}"))
    X[4] = (e.v(f"{Ev[0][0]} - {Ov[0][0]}"), e.v(f"{Ev[0][1]} - {Ov[0][1]}"))
    # k=1: t = w8 * O1, w8 = (r,-r): tr = r*(or+oi), ti = r*(oi-or)
    tr = e.v(f"{r} * ({Ov[1][0]} + {Ov[1][1]})"); ti = e.v(f"{r} * ({Ov[1][1]} - {Ov[1][0]})")
    X[1] = (e.v(f"{Ev[1][0]} + {tr}"), e.v(f"{Ev[1][1]} + {ti}"))
    X[5] = (e.v(f"{Ev[1][0]} - {tr}"), e.v(f"{Ev[1][1]} - {ti}"))
    # k=2: t = -i*O2 = (oi, -or)
    X[2] = (e.v(f"{Ev[2][0]} + {Ov[2][1]}"), e.v(f"{Ev[2][1]} - {Ov[2][0]}"))
    X[6] = (e.v(f"{Ev[2][0]} - {Ov[2][1]}"), e.v(f"{Ev[2][1]} + {Ov[2][0]}"))
    # k=3: t = w8^3*O3, w8^3=(-r,-r): tr = r*(oi-or), ti = -r*(or+oi)
    u = e.v(f"{r} * ({Ov[3][1]} - {Ov[3][0]})"); w = e.v(f"{r} * ({Ov[3][0]} + {Ov[3][1]})")
    X[3] = (e.v(f"{Ev[3][0]} + {u}"), e.v(f"{Ev[3][1]} - {w}"))
    X[7] = (e.v(f"{Ev[3][0]} - {u}"), e.v(f"{Ev[3][1]} + {w}"))
    return X

def dft_odd(e, xs):
    """symmetric direct DFT for odd n; k-tiled accumulators for ILP"""
    n = len(xs); h = n // 2
    x0r, x0i = xs[0]
    er=[None]*(h+1); ei=[None]*(h+1); orr=[None]*(h+1); oi=[None]*(h+1)
    for j in range(1, h+1):
        er[j] = e.v(f"{xs[j][0]} + {xs[n-j][0]}"); ei[j] = e.v(f"{xs[j][1]} + {xs[n-j][1]}")
        orr[j] = e.v(f"{xs[j][0]} - {xs[n-j][0]}"); oi[j] = e.v(f"{xs[j][1]} - {xs[n-j][1]}")
    # X0 via pairwise tree
    def tree(vals):
        vals = list(vals)
        while len(vals) > 1:
            nxt = []
            for i in range(0, len(vals)-1, 2):
                nxt.append(e.v(f"{vals[i]} + {vals[i+1]}"))
            if len(vals) % 2: nxt.append(vals[-1])
            vals = nxt
        return vals[0]
    X = [None]*n
    X[0] = (tree([x0r] + er[1:]), tree([x0i] + ei[1:]))
    KT = 4
    ks = list(range(1, h+1))
    for t0 in range(0, h, KT):
        tile = ks[t0:t0+KT]
        Ar = {k: x0r for k in tile}; Ai = {k: x0i for k in tile}
        Br = {}; Bi = {}
        for j in range(1, h+1):
            for k in tile:
                c, s = tw(j*k, n)
                kc = e.K(c); ksn = e.K(-s)
                Ar[k] = e.v(f"{Ar[k]} + {kc}*{er[j]}")
                Ai[k] = e.v(f"{Ai[k]} + {kc}*{ei[j]}")
                if j == 1:
                    Br[k] = e.v(f"{ksn}*{orr[j]}"); Bi[k] = e.v(f"{ksn}*{oi[j]}")
                else:
                    Br[k] = e.v(f"{Br[k]} + {ksn}*{orr[j]}")
                    Bi[k] = e.v(f"{Bi[k]} + {ksn}*{oi[j]}")
        for k in tile:
            X[k]   = (e.v(f"{Ar[k]} + {Bi[k]}"), e.v(f"{Ai[k]} - {Br[k]}"))
            X[n-k] = (e.v(f"{Ar[k]} - {Bi[k]}"), e.v(f"{Ai[k]} + {Br[k]}"))
    return X

def gen_prime(L, S, V, axis, st, fused, mapstyle):
    """top-level prime codelet: e/o staged in stack buffer, table-driven k-tile loop"""
    n = L; h = n // 2
    KT = {13: 3, 17: 4, 23: 4}[L]
    # tiles with padding by repeating last k
    ks = list(range(1, h+1))
    tiles = [ks[i:i+KT] for i in range(0, h, KT)]
    tiles[-1] = tiles[-1] + [tiles[-1][-1]]*(KT - len(tiles[-1]))
    NT = len(tiles)
    # constant table: per tile, per j: KT cos then KT (-sin)
    pc_vals = []
    for tile in tiles:
        for j in range(1, h+1):
            row_c = []; row_s = []
            for k in tile:
                c, s = tw(j*k, n)
                row_c.append(float(c).hex()); row_s.append(float(-s).hex())
            pc_vals += row_c + row_s
    # output offset table: per tile, per slot: [k_off, nk_off]
    po_vals = []
    for tile in tiles:
        for k in tile:
            po_vals += [k*st*S, (n-k)*st*S]
    tbl = (f"static const double PC{L}_{axis}_S{S}[{len(pc_vals)}] = {{{','.join(pc_vals)}}};\n"
           f"static const int PO{L}_{axis}_S{S}[{len(po_vals)}] = {{{','.join(map(str,po_vals))}}};\n")
    e = E("h")
    lines = e.lines
    def tree(vals):
        vals = list(vals)
        while len(vals) > 1:
            nxt = []
            for i in range(0, len(vals)-1, 2):
                nxt.append(e.v(f"{vals[i]} + {vals[i+1]}"))
            if len(vals) % 2: nxt.append(vals[-1])
            vals = nxt
        return vals[0]
    pfb = L*S*8 if (axis == 'z' and PFPRIME.get(L)) else 0
    if pfb: e.lines.append(f"PF((const char*)p + {pfb});")
    x0r = e.v(f"T_LD(p + 0)"); x0i = e.v(f"T_LD(p + {V})")
    ers = []; eis = []
    for j in range(1, h+1):
        if pfb:
            e.lines.append(f"PF((const char*)(p + {j*st*S}) + {pfb});")
            e.lines.append(f"PF((const char*)(p + {(n-j)*st*S}) + {pfb});")
        a_r = e.v(f"T_LD(p + {j*st*S})");       a_i = e.v(f"T_LD(p + {j*st*S + V})")
        b_r = e.v(f"T_LD(p + {(n-j)*st*S})");   b_i = e.v(f"T_LD(p + {(n-j)*st*S + V})")
        er = e.v(f"{a_r} + {b_r}"); ei = e.v(f"{a_i} + {b_i}")
        orr = e.v(f"{a_r} - {b_r}"); oi = e.v(f"{a_i} - {b_i}")
        lines.append(f"eob[{4*(j-1)}] = {er};")
        lines.append(f"eob[{4*(j-1)+1}] = {ei};")
        lines.append(f"eob[{4*(j-1)+2}] = {orr};")
        lines.append(f"eob[{4*(j-1)+3}] = {oi};")
        ers.append(er); eis.append(ei)
    X0r = tree([x0r] + ers); X0i = tree([x0i] + eis)
    if not fused:
        lines.append(f"T_ST(p + 0, {X0r});")
        lines.append(f"T_ST(p + {V}, {X0i});")
    else:
        emit_map_store(e, X0r, X0i, "0", "0", mapstyle)
    head = [f"VDA VDT eob[{4*h}];"] + e.const_decls() + e.lines
    # ---- tile loop body ----
    e2 = E("u")
    l2 = e2.lines
    Ar={}; Ai={}; Br={}; Bi={}
    for t in range(KT):
        Ar[t] = x0r; Ai[t] = x0i
    for j in range(1, h+1):
        er = e2.v(f"eob[{4*(j-1)}]"); ei = e2.v(f"eob[{4*(j-1)+1}]")
        orr = e2.v(f"eob[{4*(j-1)+2}]"); oi = e2.v(f"eob[{4*(j-1)+3}]")
        base = (j-1)*2*KT
        for t in range(KT):
            kc = e2.v(f"T_BC(pc + {base + t})")
            Ar[t] = e2.v(f"{Ar[t]} + {kc}*{er}")
            Ai[t] = e2.v(f"{Ai[t]} + {kc}*{ei}")
            ks_ = e2.v(f"T_BC(pc + {base + KT + t})")
            if j == 1:
                Br[t] = e2.v(f"{ks_}*{orr}"); Bi[t] = e2.v(f"{ks_}*{oi}")
            else:
                Br[t] = e2.v(f"{Br[t]} + {ks_}*{orr}")
                Bi[t] = e2.v(f"{Bi[t]} + {ks_}*{oi}")
    for t in range(KT):
        xr = e2.v(f"{Ar[t]} + {Bi[t]}"); xi = e2.v(f"{Ai[t]} - {Br[t]}")
        yr = e2.v(f"{Ar[t]} - {Bi[t]}"); yi = e2.v(f"{Ai[t]} + {Br[t]}")
        if not fused:
            l2.append(f"T_ST(p + po[{2*t}], {xr});")
            l2.append(f"T_ST(p + po[{2*t}] + {V}, {xi});")
            l2.append(f"T_ST(p + po[{2*t+1}], {yr});")
            l2.append(f"T_ST(p + po[{2*t+1}] + {V}, {yi});")
        else:
            emit_map_store(e2, xr, xi, f"po[{2*t}]", f"po[{2*t}]", mapstyle)
            emit_map_store(e2, yr, yi, f"po[{2*t+1}]", f"po[{2*t+1}]", mapstyle)
    args = "double* restrict p" + (", const double* restrict cc" if fused else "")
    hb = "\n  ".join(head)
    bb = "\n    ".join(e2.const_decls() + l2)
    return f"""{tbl}static INLINE void W_d{L}_{axis}({args}){{
  {hb}
  #pragma GCC unroll 1
  for (long t=0; t<{NT}; t++){{
    const double* restrict pc = PC{L}_{axis}_S{S} + t*{h*2*KT};
    const int* restrict po = PO{L}_{axis}_S{S} + t*{2*KT};
    {bb}
  }}
}}
"""

def dft_pfa(e, xs, n1, n2):
    n = len(xs)
    assert n == n1*n2 and math.gcd(n1, n2) == 1
    # input map: in[j1][j2] = x[(n2*j1 + n1*j2) % n]
    cols = []
    for j2 in range(n2):
        col = [xs[(n2*j1 + n1*j2) % n] for j1 in range(n1)]
        cols.append(dft(e, col))           # n1-point along j1 -> indexed k1
    # now rows: for each k1, n2-point over j2
    out2d = {}
    for k1 in range(n1):
        row = [cols[j2][k1] for j2 in range(n2)]
        row_out = dft(e, row)
        for k2 in range(n2):
            out2d[(k1,k2)] = row_out[k2]
    # CRT output map
    inv_n2_mod_n1 = pow(n2, -1, n1); inv_n1_mod_n2 = pow(n1, -1, n2)
    X = [None]*n
    for k1 in range(n1):
        for k2 in range(n2):
            k = (k1*n2*inv_n2_mod_n1 + k2*n1*inv_n1_mod_n2) % n
            X[k] = out2d[(k1,k2)]
    return X

def dft(e, xs):
    n = len(xs)
    if n == 1: return xs
    if n == 2: return dft2(e, xs)
    if n == 4: return dft4(e, xs)
    if n == 8: return dft8(e, xs)
    if n == 6: return dft_pfa(e, xs, 2, 3)
    if n in (3,5,7,9,11,13,17,23): return dft_odd(e, xs)
    raise ValueError(n)

# ------------- validation of straight-line emission --------------------------
def validate(n, builder):
    e = E("v")
    ins = [(f"inr{j}", f"ini{j}") for j in range(n)]
    outs = builder(e, ins)
    rng = np.random.default_rng(7)
    xr = rng.standard_normal(n); xi = rng.standard_normal(n)
    env = {}
    for j in range(n):
        env[f"inr{j}"] = xr[j]; env[f"ini{j}"] = xi[j]
    for nm, v in e.consts.values():
        env[nm] = v
    for ln in e.lines:
        assert ln.startswith("VDT ")
        name, expr = ln[4:-1].split(" = ", 1)
        env[name] = eval(expr, {}, env)
    got = np.array([complex(env[r], env[i]) for r, i in outs])
    j = np.arange(n)
    W = np.exp(-2j*np.pi*np.outer(j,j)/n)
    want = W @ (xr + 1j*xi)
    err = np.linalg.norm(got-want)/np.linalg.norm(want)
    assert err < 1e-14, (n, err)
    return err

for n in (2,3,4,5,6,8,9,13,17,23):
    validate(n, dft)
validate(36, lambda e, xs: dft_pfa(e, xs, 4, 9))
validate(45, lambda e, xs: dft_pfa(e, xs, 9, 5))
print("straight-line codelets validated")


# ===== complex-interleaved (natural layout) emitters =====
# each symbol = vector of VC complex (interleaved re,im per 128-bit lane)
# ops available: +,-, real-const multiply, T_SW(swap re/im), T_FMAS(a,b,c)=a*b[-+]c (even -, odd +),
#                T_FMSA(a,b,c)=a*b[+-]c (even +, odd -)   [matching _mm512_fmaddsub/_fmsubadd]

def dft2_c(e, xs):
    a, b = xs
    return [e.v(f"{a} + {b}"), e.v(f"{a} - {b}")]

def dft4_c(e, xs):
    x0,x1,x2,x3 = xs
    a = e.v(f"{x0} + {x2}"); b = e.v(f"{x0} - {x2}")
    c = e.v(f"{x1} + {x3}"); d = e.v(f"{x1} - {x3}")
    X0 = e.v(f"{a} + {c}"); X2 = e.v(f"{a} - {c}")
    sd = e.v(f"T_SW({d})"); kone = e.K(1.0)
    X1 = e.v(f"T_FMSA({b}, {kone}, {sd})")   # b - i d = (br + di, bi - dr)
    X3 = e.v(f"T_FMAS({b}, {kone}, {sd})")   # b + i d = (br - di, bi + dr)
    return [X0, X1, X2, X3]
# NOTE: b - i*d = (br - di, bi + dr): even lane: br - sd_even (sd_even = di) ok; odd: bi + sd_odd (= dr) ok -> fmaddsub(b,1,sd) = even a*b - c, odd a*b + c -> X1 = T_FMAS. and b + i*d = (br + di, bi - dr) -> fmsubadd? even a*b + c?? fmsubadd: even: a*b+c, odd: a*b-c -> X3 = T_FMSA. OK as written.

def dft8_c(e, xs):
    Ev = dft4_c(e, [xs[0], xs[2], xs[4], xs[6]])
    Ov = dft4_c(e, [xs[1], xs[3], xs[5], xs[7]])
    r = e.K(math.sqrt(0.5)); kone = e.K(1.0)
    X = [None]*8
    X[0] = e.v(f"{Ev[0]} + {Ov[0]}"); X[4] = e.v(f"{Ev[0]} - {Ov[0]}")
    # t = w8 * O1, w8 = (r, -r): t = r*O1 + (-r)*(i*O1) -> use fmaddsub: c=r, s=-r:
    # t = fmaddsub(rvec, O1, s*swap(O1)) with even: r*or - (-r*oi)?? do general cmul helper
    def cmul(z, c, s):
        sz = e.v(f"T_SW({z})")
        kc = e.K(c); ks = e.K(s)
        v = e.v(f"{ks}*{sz}")
        return e.v(f"T_FMAS({z}, {kc}, {v})")   # even: c*zr - s*zi ; odd: c*zi + s*zr
    t1 = cmul(Ov[1], math.sqrt(0.5), -math.sqrt(0.5))
    X[1] = e.v(f"{Ev[1]} + {t1}"); X[5] = e.v(f"{Ev[1]} - {t1}")
    so = e.v(f"T_SW({Ov[2]})")
    X[2] = e.v(f"T_FMSA({Ev[2]}, {kone}, {so})")  # E - i O
    X[6] = e.v(f"T_FMAS({Ev[2]}, {kone}, {so})")  # E + i O
    t3 = cmul(Ov[3], -math.sqrt(0.5), -math.sqrt(0.5))
    X[3] = e.v(f"{Ev[3]} + {t3}"); X[7] = e.v(f"{Ev[3]} - {t3}")
    return X

def dft_odd_c(e, xs):
    n = len(xs); h = n//2
    x0 = xs[0]
    kone = e.K(1.0)
    es = [None]*(h+1); os_ = [None]*(h+1)
    for j in range(1, h+1):
        es[j] = e.v(f"{xs[j]} + {xs[n-j]}")
        os_[j] = e.v(f"{xs[j]} - {xs[n-j]}")
    def tree(vals):
        vals = list(vals)
        while len(vals) > 1:
            nxt = []
            for i in range(0, len(vals)-1, 2):
                nxt.append(e.v(f"{vals[i]} + {vals[i+1]}"))
            if len(vals) % 2: nxt.append(vals[-1])
            vals = nxt
        return vals[0]
    X = [None]*n
    X[0] = tree([x0] + es[1:])
    KT = 4
    ks = list(range(1, h+1))
    for t0 in range(0, h, KT):
        tile = ks[t0:t0+KT]
        A = {k: x0 for k in tile}; Bs = {}
        for j in range(1, h+1):
            sj = e.v(f"T_SW({os_[j]})")
            for k in tile:
                c, s = tw(j*k, n)
                kc = e.K(c); ksn = e.K(-s)
                A[k] = e.v(f"{A[k]} + {kc}*{es[j]}")
                if j == 1: Bs[k] = e.v(f"{ksn}*{sj}")
                else:      Bs[k] = e.v(f"{Bs[k]} + {ksn}*{sj}")
        for k in tile:
            # B symbol holds swap(sum s*o) = (Bi, Br). X_k = A - i*B = (Ar - ... wait:
            # we accumulated Bs = sum ksn * swap(o_j) = (Bim_classic, Bre_classic) pairs swapped.
            # X_k = A - iB: even: Ar + Bi_cl = Ar + Bs_even ; odd: Ai - Br_cl = Ai - Bs_odd -> fmsubadd(A,1,Bs): even a+c, odd a-c
            X[k]   = e.v(f"T_FMSA({A[k]}, {kone}, {Bs[k]})")
            X[n-k] = e.v(f"T_FMAS({A[k]}, {kone}, {Bs[k]})")
        del Bs
    return X

def dft_pfa_c(e, xs, n1, n2):
    n = len(xs)
    inv_n2 = pow(n2, -1, n1); inv_n1 = pow(n1, -1, n2)
    cols = []
    for j2 in range(n2):
        col = [xs[(n2*j1 + n1*j2) % n] for j1 in range(n1)]
        cols.append(dft_c(e, col))
    out2d = {}
    for k1 in range(n1):
        row = [cols[j2][k1] for j2 in range(n2)]
        row_out = dft_c(e, row)
        for k2 in range(n2):
            out2d[(k1,k2)] = row_out[k2]
    X = [None]*n
    for k1 in range(n1):
        for k2 in range(n2):
            k = (k1*n2*inv_n2 + k2*n1*inv_n1) % n
            X[k] = out2d[(k1,k2)]
    return X

def dft_c(e, xs):
    n = len(xs)
    if n == 1: return xs
    if n == 2: return dft2_c(e, xs)
    if n == 4: return dft4_c(e, xs)
    if n == 8: return dft8_c(e, xs)
    if n == 6: return dft_pfa_c(e, xs, 2, 3)
    if n in (3,5,7,9,11,13,17,23): return dft_odd_c(e, xs)
    raise ValueError(n)

def validate_c(n, builder):
    """validate complex-interleaved emission with VC=1-lane complex semantics"""
    import numpy as _np
    e = E("v")
    ins = [f"in{j}" for j in range(n)]
    outs = builder(e, ins)
    rng = _np.random.default_rng(5)
    x = rng.standard_normal(n) + 1j*rng.standard_normal(n)
    env = {}
    class CV:  # pair (even,odd) = (re, im) with lane-parity ops
        def __init__(s, ev, od): s.ev, s.od = ev, od
        def __add__(s, o): return CV(s.ev+o.ev, s.od+o.od)
        def __sub__(s, o): return CV(s.ev-o.ev, s.od-o.od)
        def __mul__(s, o): return CV(s.ev*o.ev, s.od*o.od)
        def __rmul__(s, o): return s.__mul__(o)
    def T_SW(a): return CV(a.od, a.ev)
    def T_FMAS(a,b,c): return CV(a.ev*b.ev - c.ev, a.od*b.od + c.od)
    def T_FMSA(a,b,c): return CV(a.ev*b.ev + c.ev, a.od*b.od - c.od)
    genv = {"T_SW":T_SW, "T_FMAS":T_FMAS, "T_FMSA":T_FMSA}
    for j in range(n):
        env[f"in{j}"] = CV(x[j].real, x[j].imag)
    for nm, v in e.consts.values():
        env[nm] = CV(v, v)
    for ln in e.lines:
        assert ln.startswith("VDT ")
        name, expr = ln[4:-1].split(" = ", 1)
        env[name] = eval(expr, genv, env)
    got = _np.array([complex(env[o].ev, env[o].od) for o in outs])
    j = _np.arange(n)
    W = _np.exp(-2j*_np.pi*_np.outer(j,j)/n)
    want = W @ x
    err = _np.linalg.norm(got-want)/_np.linalg.norm(want)
    assert err < 1e-13, (n, err)
    return err

for _n in (2,3,4,5,8,9,13,23):
    validate_c(_n, dft_c)
validate_c(36, lambda e, xs: dft_pfa_c(e, xs, 4, 9))
validate_c(45, lambda e, xs: dft_pfa_c(e, xs, 9, 5))
print("complex-interleaved codelets validated")

# =============================================================================
#  C code generation
# =============================================================================

MAPSTYLE = {6:'hyb', 8:'hyb', 13:'hyb', 17:'hyb', 23:'hyb', 36:'hyb', 45:'hyb', 64:'hyb'}
# passx: ('direct',) or ('scratch', ZT, use_nt_copyout)
PFPRIME = {}
W2SET = {13, 17, 23}
W1SET = {23}
NATSET = {23, 36, 45, 64}
NATTHRESH = {23: 3, 36: 1<<30, 45: 1<<30, 64: 1<<30}
PREFW = {6:8, 8:8, 13:8, 17:4, 23:4, 36:4, 45:8, 64:4}
PASSX = {6:('direct',), 8:('direct',), 13:('direct',), 17:('direct',), 23:('direct',), 36:('scratch', 12, False), 45:('scratch', 15, False), 64:('scratch', 16, False)}

def SLAB(L): return L*L + 2

class Fn:
    """function text builder with raw lines + emitter"""
    def __init__(self):
        self.e = E("")
        self.raw = []          # list of (position marker) -> we just append in order
    def line(self, s): self.e.lines.append(s)

def emit_loads(e, n, st, S, V, ptr="p"):
    xs = []
    for j in range(n):
        off = j*st*S
        r = e.v(f"T_LD({ptr} + {off})")
        i = e.v(f"T_LD({ptr} + {off+V})")
        xs.append((r, i))
    return xs

def emit_plain_stores(e, outs, st, S, V, ptr="p"):
    for k, (r, i) in enumerate(outs):
        off = k*st*S
        e.lines.append(f"T_ST({ptr} + {off}, {r});")
        e.lines.append(f"T_ST({ptr} + {off+V}, {i});")

def emit_map_store(e, Xr, Xi, poff_expr, coff_expr, style, pptr="p", cptr="cc"):
    """z = X + c ; x = z/(1+|z|); store to pptr at poff"""
    zr = e.v(f"{Xr} + T_LD({cptr} + {coff_expr})")
    zi = e.v(f"{Xi} + T_LD({cptr} + ({coff_expr}) + T_V)")
    k1 = e.K(1.0)
    if style == 'div':
        s = e.v(f"{zr}*{zr} + {zi}*{zi}")
        d = e.v(f"{k1} + T_SQT({s})")
        w = e.v(f"{k1} / {d}")
    else:
        ktiny = e.K(1e-300); khalf = e.K(0.5); k15 = e.K(1.5)
        s  = e.v(f"{ktiny} + {zr}*{zr} + {zi}*{zi}")
        t0 = e.v(f"T_RSQ({s})")
        hs = e.v(f"{khalf}*{s}")
        u  = e.v(f"{t0}*{t0}");  t1 = e.v(f"{t0}*({k15} - {hs}*{u})")
        u2 = e.v(f"{t1}*{t1}");  t2 = e.v(f"{t1}*({k15} - {hs}*{u2})")
        d  = e.v(f"{k1} + {s}*{t2}")
        if style == 'hyb':
            w = e.v(f"{k1} / {d}")
        else:  # nr
            w0 = e.v(f"T_RCP({d})")
            w1 = e.v(f"{w0} + {w0}*({k1} - {d}*{w0})")
            w  = e.v(f"{w1} + {w1}*({k1} - {d}*{w1})")
    e.lines.append(f"T_ST({pptr} + {poff_expr}, {zr}*{w});")
    e.lines.append(f"T_ST({pptr} + ({poff_expr}) + T_V, {zi}*{w});")

def fn_text(name, args, e, extra_body=None):
    body = e.const_decls() + e.lines
    b = "\n  ".join(body)
    return f"static INLINE void {name}({args}){{\n  {b}\n}}\n"

def gen_straight(L, S, V, axis, st, fused, mapstyle):
    e = E("")
    xs = emit_loads(e, L, st, S, V)
    outs = dft(e, xs)
    if not fused:
        emit_plain_stores(e, outs, st, S, V)
        return fn_text(f"W_d{L}_{axis}", "double* restrict p", e)
    else:
        for k, (r, i) in enumerate(outs):
            emit_map_store(e, r, i, str(k*st*S), str(k*st*S), mapstyle)
        return fn_text(f"W_d{L}_{axis}", "double* restrict p, const double* restrict cc", e)

def gen_pfa_looped(L, n1, n2, S, V, axis, st, fused, mapstyle):
    """stage1: n2 iterations of DFT-n1; stage2: n1 iterations of DFT-n2. mid[k1*n2+j2]"""
    n = L
    inv_n2 = pow(n2, -1, n1); inv_n1 = pow(n1, -1, n2)
    IN  = [((n2*j1 + n1*j2) % n)*st*S for j2 in range(n2) for j1 in range(n1)]   # [j2*n1 + j1]
    OUT = [((k1*n2*inv_n2 + k2*n1*inv_n1) % n)*st*S for k1 in range(n1) for k2 in range(n2)]  # [k1*n2+k2]
    tbl = (f"static const int IN{L}_{axis}_S{S}[{n1*n2}] = {{{','.join(map(str,IN))}}};\n"
           f"static const int OUT{L}_{axis}_S{S}[{n1*n2}] = {{{','.join(map(str,OUT))}}};\n")
    # stage1 body
    e1 = E("")
    xs = []
    for j1 in range(n1):
        r = e1.v(f"T_LD(p + off[{j1}])")
        i = e1.v(f"T_LD(p + off[{j1}] + T_V)")
        xs.append((r, i))
    outs1 = dft(e1, xs)
    for k1, (r, i) in enumerate(outs1):
        e1.lines.append(f"mid[{k1*n2} + j2] = {r};")
        e1.lines.append(f"mid[{n + k1*n2} + j2] = {i};")
    # stage2 body
    e2 = E("")
    xs2 = []
    for j2 in range(n2):
        r = e2.v(f"mr[{j2}]"); i = e2.v(f"mi[{j2}]")
        xs2.append((r, i))
    outs2 = dft(e2, xs2)
    if not fused:
        for k2, (r, i) in enumerate(outs2):
            e2.lines.append(f"T_ST(p + oo[{k2}], {r});")
            e2.lines.append(f"T_ST(p + oo[{k2}] + T_V, {i});")
    else:
        for k2, (r, i) in enumerate(outs2):
            emit_map_store(e2, r, i, f"oo[{k2}]", f"oo[{k2}]", mapstyle)
    args = "double* restrict p" + (", const double* restrict cc" if fused else "")
    body1 = "\n      ".join(e1.const_decls() + e1.lines)
    body2 = "\n      ".join(e2.const_decls() + e2.lines)
    txt = f"""{tbl}static void W_d{L}_{axis}({args}){{
  VDA VDT mid[{2*n}];
  {{
    for (long j2=0; j2<{n2}; j2++){{
      const int* restrict off = IN{L}_{axis}_S{S} + j2*{n1};
      {body1}
    }}
  }}
  {{
    for (long k1=0; k1<{n1}; k1++){{
      const VDT* restrict mr = mid + k1*{n2};
      const VDT* restrict mi = mid + {n} + k1*{n2};
      const int* restrict oo = OUT{L}_{axis}_S{S} + k1*{n2};
      {body2}
    }}
  }}
}}
"""
    return txt

def gen_64(S, V, axis, st, fused, mapstyle, prefetch):
    L = 64
    # twiddle tables (shared, emit once globally; name TW64R/TW64I)
    e0 = E("")      # peel b = 0
    xs = emit_loads(e0, 8, 8*st, S, V)          # elements 8a, a=0..7
    outs = dft8(e0, xs)
    for d, (r, i) in enumerate(outs):
        e0.lines.append(f"buf[{d}] = {r};")
        e0.lines.append(f"buf[{64+d}] = {i};")
    # stage1 generic b
    e1 = E("")
    xs = []
    for a in range(8):
        off = (8*a)*st*S
        if prefetch:
            e1.lines.append(f"PF(pb + {off} + {2*S});")
        r = e1.v(f"T_LD(pb + {off})")
        i = e1.v(f"T_LD(pb + {off+V})")
        xs.append((r, i))
    outs = dft8(e1, xs)
    e1.lines.append(f"buf[b8] = {outs[0][0]};")
    e1.lines.append(f"buf[64+b8] = {outs[0][1]};")
    for d in range(1, 8):
        r, i = outs[d]
        wr = e1.v(f"T_BC(twr + {d})"); wi = e1.v(f"T_BC(twi + {d})")
        vr = e1.v(f"{r}*{wr} - {i}*{wi}"); vi = e1.v(f"{r}*{wi} + {i}*{wr}")
        e1.lines.append(f"buf[b8 + {d}] = {vr};")
        e1.lines.append(f"buf[{64}+b8 + {d}] = {vi};")
    # stage2
    e2 = E("")
    xs2 = []
    for b in range(8):
        r = e2.v(f"buf[{8*b} + d]"); i = e2.v(f"buf[{64+8*b} + d]")
        xs2.append((r, i))
    outs2 = dft8(e2, xs2)
    if not fused:
        for c, (r, i) in enumerate(outs2):
            off = (8*c)*st*S
            e2.lines.append(f"T_ST(pd + {off}, {r});")
            e2.lines.append(f"T_ST(pd + {off+V}, {i});")
    else:
        for c, (r, i) in enumerate(outs2):
            off = (8*c)*st*S
            emit_map_store(e2, r, i, str(off), str(off), mapstyle, pptr="pd", cptr="cd")
    args = "double* restrict p" + (", const double* restrict cc" if fused else "")
    body0 = "\n    ".join(e0.const_decls() + e0.lines)
    body1 = "\n    ".join(e1.const_decls() + e1.lines)
    body2 = "\n    ".join(e2.const_decls() + e2.lines)
    cdecl = f"const double* restrict cd = cc + d*(long){st*S};" if fused else ""
    txt = f"""static void W_d64_{axis}({args}){{
  VDA VDT buf[128];
  {{
    {body0}
  }}
  for (long b=1; b<8; b++){{
    double* restrict pb = p + b*(long){st*S};
    const double* restrict twr = TW64R + b*8;
    const double* restrict twi = TW64I + b*8;
    long b8 = b*8;
    {body1}
  }}
  for (long d=0; d<8; d++){{
    double* restrict pd = p + d*(long){st*S};
    {cdecl}
    {body2}
  }}
}}
"""
    return txt


def gen_map_loop(style):
    """map applied elementwise on contiguous run: s (scratch in), cr (c in), d (out), count CNT doubles"""
    e = E("m")
    zr0 = e.v("T_LD(s + u)"); zi0 = e.v("T_LD(s + u + T_V)")
    # add c
    zr = e.v(f"{zr0} + T_LD(cr + u)"); zi = e.v(f"{zi0} + T_LD(cr + u + T_V)")
    k1 = e.K(1.0)
    if style == 'div':
        sq = e.v(f"{zr}*{zr} + {zi}*{zi}")
        d_ = e.v(f"{k1} + T_SQT({sq})")
        w = e.v(f"{k1} / {d_}")
    else:
        ktiny = e.K(1e-300); khalf = e.K(0.5); k15 = e.K(1.5)
        sq  = e.v(f"{ktiny} + {zr}*{zr} + {zi}*{zi}")
        t0 = e.v(f"T_RSQ({sq})")
        hs = e.v(f"{khalf}*{sq}")
        u1 = e.v(f"{t0}*{t0}"); t1 = e.v(f"{t0}*({k15} - {hs}*{u1})")
        u2 = e.v(f"{t1}*{t1}"); t2 = e.v(f"{t1}*({k15} - {hs}*{u2})")
        d_ = e.v(f"{k1} + {sq}*{t2}")
        if style == 'hyb':
            w = e.v(f"{k1} / {d_}")
        else:
            w0 = e.v(f"T_RCP({d_})")
            w1 = e.v(f"{w0} + {w0}*({k1} - {d_}*{w0})")
            w = e.v(f"{w1} + {w1}*({k1} - {d_}*{w1})")
    e.lines.append(f"T_ST(d + u, {zr}*{w});")
    e.lines.append(f"T_ST(d + u + T_V, {zi}*{w});")
    return "\n        ".join(e.const_decls() + e.lines)

def gen_step(L, S, V):
    SL = SLAB(L); VOL = L*SL
    cfg = PASSX[L]
    if cfg[0] == 'scratch':
        ZT = cfg[1]
        NZT = L // ZT
        assert L % ZT == 0
        maploop = gen_map_loop(MAPSTYLE[L])
        passx = f"""static VDA double SCR{L}[{L*ZT*S}];
static void W_passx{L}(double* restrict p, const double* restrict cg){{
  double* restrict sx = SCR{L};
  for (int y=0; y<{L}; y++){{
    for (int zt=0; zt<{NZT}; zt++){{
      long o = (long)y*{L*S} + zt*{ZT*S};
      for (int x=0; x<{L}; x++)
        memcpy(sx + x*{ZT*S}, p + (long)x*{SL*S} + o, {ZT*S}*sizeof(double));
      for (int z=0; z<{ZT}; z++) W_d{L}_xs(sx + z*{S});
      for (int x=0; x<{L}; x++){{
        const double* restrict s = sx + x*{ZT*S};
        const double* restrict cr = cg + (long)x*{SL*S} + o;
        double* restrict d = p + (long)x*{SL*S} + o;
        for (int u=0; u<{ZT*S}; u+={2*V}){{
        {maploop}
      }}
      }}
    }}
  }}
}}
"""
    else:
        passx = f"""static void W_passx{L}(double* restrict p, const double* restrict cg){{
    for (int y=0; y<{L}; y++){{
      long o = (long)y*{L*S};
      for (int z=0; z<{L}; z++){{ W_d{L}_x(p + o, cg + o); o += {S}; }}
    }}
}}
"""
    return f"""static void W_passzy{L}(double* restrict p){{
    for (int i=0; i<{L}; i++){{
      double* restrict q = p + (long)i*{SL*S};
      for (int y=0; y<{L}; y++) W_d{L}_z(q + (long)y*{L*S});
      for (int z=0; z<{L}; z++) W_d{L}_y(q + (long)z*{S});
    }}
}}
{passx}static void W_step{L}(double* restrict xx, const double* restrict cc, long G){{
  for (long g=0; g<G; g++){{
    double* restrict p = xx + g*(long){VOL*S};
    W_passzy{L}(p);
    W_passx{L}(p, cc + g*(long){VOL*S});
  }}
}}
"""


NTOK = [("T_RSQ","vrsqrt8"),("T_RCP","vrcp8"),("T_SQT","vsqrt8"),("T_LDU","vld8u"),("T_STU","vst8u"),
        ("T_LD","vld8"),("T_ST","vst8"),("T_BC","vbc8"),("T_SW","vswp8"),("T_FMAS","vfmas8"),
        ("T_FMSA","vfmsa8"),("VSET1","vset8"),("VDT","vd8"),("VDA","VA64")]

def nat_finish(txt):
    for a,b in NTOK: txt = txt.replace(a,b)
    return txt

def gen_nat_codelet(L, name, estride_doubles, builder):
    """straight/looped complex codelet: element j at p + j*estride_doubles"""
    e = E("n")
    xs = [e.v(f"T_LD(p + {j*estride_doubles})") for j in range(L)]
    outs = builder(e, xs)
    for j, o in enumerate(outs):
        e.lines.append(f"T_ST(p + {j*estride_doubles}, {o});")
    body = "\n  ".join(e.const_decls() + e.lines)
    return f"static INLINE void {name}(double* restrict p){{\n  {body}\n}}\n"

def gen_nat_64(name, estride_doubles):
    ed = estride_doubles
    e0 = E("a")
    xs = [e0.v(f"T_LD(p + {(8*a)*ed})") for a in range(8)]
    outs = dft8_c(e0, xs)
    for d, o in enumerate(outs):
        e0.lines.append(f"buf[{d}] = {o};")
    e1 = E("b")
    xs = [e1.v(f"T_LD(pb + {(8*a)*ed})") for a in range(8)]
    outs = dft8_c(e1, xs)
    e1.lines.append(f"buf[b8] = {outs[0]};")
    for d in range(1, 8):
        o = outs[d]
        sw = e1.v(f"T_SW({o})")
        wr = e1.v(f"T_BC(twr + {d})"); wi = e1.v(f"T_BC(twi + {d})")
        v = e1.v(f"{wi}*{sw}")
        e1.lines.append(f"buf[b8 + {d}] = T_FMAS({o}, {wr}, {v});")
    e2 = E("c")
    xs2 = [e2.v(f"buf[{8*b} + d]") for b in range(8)]
    outs2 = dft8_c(e2, xs2)
    st2 = []
    for c_, o in enumerate(outs2):
        st2.append(f"T_ST(pd + {(8*c_)*ed}, {o});")
    body0 = "\n    ".join(e0.const_decls() + e0.lines)
    body1 = "\n    ".join(e1.const_decls() + e1.lines)
    body2 = "\n    ".join(e2.const_decls() + e2.lines + st2)
    return f"""static void {name}(double* restrict p){{
  VDA VDT buf[64];
  {{
    {body0}
  }}
  for (long b=1; b<8; b++){{
    double* restrict pb = p + b*(long){ed};
    const double* restrict twr = TW64R + b*8;
    const double* restrict twi = TW64I + b*8;
    long b8 = b*8;
    {body1}
  }}
  for (long d=0; d<8; d++){{
    double* restrict pd = p + d*(long){ed};
    {body2}
  }}
}}
"""

def gen_nat_pfa(L, n1, n2, name, estride_doubles):
    ed = estride_doubles
    IN  = [((n2*j1 + n1*j2) % L)*ed for j2 in range(n2) for j1 in range(n1)]
    inv_n2 = pow(n2, -1, n1); inv_n1 = pow(n1, -1, n2)
    OUT = [((k1*n2*inv_n2 + k2*n1*inv_n1) % L)*ed for k1 in range(n1) for k2 in range(n2)]
    tbl = (f"static const int {name}_IN[{L}] = {{{','.join(map(str,IN))}}};\n"
           f"static const int {name}_OUT[{L}] = {{{','.join(map(str,OUT))}}};\n")
    e1 = E("a")
    xs = [e1.v(f"T_LD(p + off[{j1}])") for j1 in range(n1)]
    outs1 = dft_c(e1, xs)
    for k1, o in enumerate(outs1):
        e1.lines.append(f"mid[{k1*n2} + j2] = {o};")
    e2 = E("b")
    xs2 = [e2.v(f"mr[{j2}]") for j2 in range(n2)]
    outs2 = dft_c(e2, xs2)
    for k2, o in enumerate(outs2):
        e2.lines.append(f"T_ST(p + oo[{k2}], {o});")
    body1 = "\n      ".join(e1.const_decls() + e1.lines)
    body2 = "\n      ".join(e2.const_decls() + e2.lines)
    return f"""{tbl}static void {name}(double* restrict p){{
  VDA VDT mid[{L}];
  for (long j2=0; j2<{n2}; j2++){{
    const int* restrict off = {name}_IN + j2*{n1};
    {body1}
  }}
  for (long k1=0; k1<{n1}; k1++){{
    VDT* restrict mr = mid + k1*{n2};
    const int* restrict oo = {name}_OUT + k1*{n2};
    {body2}
  }}
}}
"""

def gen_nat_maploop(style):
    e = E("m")
    z0 = e.v("T_LD(s + u)")
    z = e.v(f"{z0} + T_LDU(cr + u)")
    k1 = e.K(1.0)
    ktiny = e.K(1e-300); khalf = e.K(0.5); k15 = e.K(1.5)
    s2a = e.v(f"{ktiny} + {z}*{z}")
    s2 = e.v(f"{s2a} + T_SW({s2a})")
    t0 = e.v(f"T_RSQ({s2})")
    hs = e.v(f"{khalf}*{s2}")
    u1 = e.v(f"{t0}*{t0}"); t1 = e.v(f"{t0}*({k15} - {hs}*{u1})")
    u2 = e.v(f"{t1}*{t1}"); t2 = e.v(f"{t1}*({k15} - {hs}*{u2})")
    d_ = e.v(f"{k1} + {s2}*{t2}")
    w = e.v(f"{k1} / {d_}")
    e.lines.append(f"T_STU(d + u, {z}*{w});")
    return "\n        ".join(e.const_decls() + e.lines)

def gen_nat_section(L):
    PZ = ((L+3)//4)*4           # padded z-row length (complex)
    ZT = {23: 24, 36: 12, 45: 16, 64: 16}[L]
    NZT = PZ // ZT
    assert PZ % ZT == 0 and ZT % 4 == 0
    L2 = L*L
    SLN = L*PZ + 4              # slab stride (complex)
    R = L % 4
    if L == 64:
        cz = gen_nat_64(f"N_d{L}_z", 8)
        cy = gen_nat_64(f"N_d{L}_y", 2*PZ)
        cx = gen_nat_64(f"N_d{L}_x", 2*ZT)
    elif L == 36:
        cz = gen_nat_pfa(L, 4, 9, f"N_d{L}_z", 8)
        cy = gen_nat_pfa(L, 4, 9, f"N_d{L}_y", 2*PZ)
        cx = gen_nat_pfa(L, 4, 9, f"N_d{L}_x", 2*ZT)
    elif L == 45:
        cz = gen_nat_pfa(L, 9, 5, f"N_d{L}_z", 8)
        cy = gen_nat_pfa(L, 9, 5, f"N_d{L}_y", 2*PZ)
        cx = gen_nat_pfa(L, 9, 5, f"N_d{L}_x", 2*ZT)
    else:
        cz = gen_nat_codelet(L, f"N_d{L}_z", 8, dft_c)
        cy = gen_nat_codelet(L, f"N_d{L}_y", 2*PZ, dft_c)
        cx = gen_nat_codelet(L, f"N_d{L}_x", 2*ZT, dft_c)
    maploop = gen_nat_maploop('hyb')
    # z-pass row-group blocks
    def zgroup(stores):
        st = "\n        ".join(f"_mm512_store_pd(r0 + {k*PZ*2} + jb*8, v[{k}]);" for k in range(stores))
        return f"""{{
      VDA VDT ts[{PZ}];
      for (int jb=0; jb<{PZ//4}; jb++){{
        __m512d v[4];
        PF((const char*)(r0 + {4*PZ*2} + jb*8));
        PF((const char*)(r0 + {5*PZ*2} + jb*8));
        PF((const char*)(r0 + {6*PZ*2} + jb*8));
        PF((const char*)(r0 + {7*PZ*2} + jb*8));
        v[0]=_mm512_load_pd(r0 + jb*8);
        v[1]=_mm512_load_pd(r0 + {PZ*2} + jb*8);
        v[2]=_mm512_load_pd(r0 + {2*PZ*2} + jb*8);
        v[3]=_mm512_load_pd(r0 + {3*PZ*2} + jb*8);
        tr4c(v);
        ts[jb*4]=(VDT)v[0]; ts[jb*4+1]=(VDT)v[1]; ts[jb*4+2]=(VDT)v[2]; ts[jb*4+3]=(VDT)v[3];
      }}
      N_d{L}_z((double*)ts);
      for (int jb=0; jb<{PZ//4}; jb++){{
        __m512d v[4];
        v[0]=(__m512d)ts[jb*4]; v[1]=(__m512d)ts[jb*4+1]; v[2]=(__m512d)ts[jb*4+2]; v[3]=(__m512d)ts[jb*4+3];
        tr4c(v);
        {st}
      }}
    }}"""
    tailblk = ""
    if R:
        tailblk = f"""
    {{
      double* restrict r0 = q + (long){(L-R)*PZ*2};
      {zgroup(R)}
    }}"""
    sec = f"""{cz}{cy}{cx}
static VDA double NSCR{L}[{L*ZT*2}];
static void N_step{L}(double* restrict p, const double* restrict cg){{
  for (int xx=0; xx<{L}; xx++){{
    double* restrict q = p + (long)xx*{SLN*2};
    for (int yg=0; yg+4<={L}; yg+=4){{
      double* restrict r0 = q + (long)yg*{PZ*2};
      {zgroup(4)}
    }}{tailblk}
    for (int zb=0; zb<{PZ//4}; zb++) N_d{L}_y(q + zb*8);
  }}
  {{
    double* restrict sx = NSCR{L};
    for (int y=0; y<{L}; y++){{
      for (int zt=0; zt<{NZT}; zt++){{
        long o = (long)y*{PZ*2} + zt*{ZT*2};
        for (int x=0; x<{L}; x++){{
          memcpy(sx + x*{ZT*2}, p + (long)x*{SLN*2} + o, {ZT*2}*sizeof(double));
          const char* cpf = (const char*)(cg + (long)x*{SLN*2} + o);
          for (int q2=0; q2<{ZT*16}; q2+=64) PF(cpf + q2);
          const char* xpf = (const char*)(p + (long)x*{SLN*2} + o + {ZT*2});
          for (int q2=0; q2<{ZT*16}; q2+=64) PF(xpf + q2);
        }}
        for (int zb=0; zb<{ZT//4}; zb++) N_d{L}_x(sx + zb*8);
        for (int x=0; x<{L}; x++){{
          const double* restrict s = sx + x*{ZT*2};
          const double* restrict cr = cg + (long)x*{SLN*2} + o;
          double* restrict d = p + (long)x*{SLN*2} + o;
          for (int u=0; u<{ZT*2}; u+=8){{
        {maploop}
      }}
        }}
      }}
    }}
  }}
}}
static void N_imp{L}(double* restrict dst, const double* restrict src){{
  for (long xx=0; xx<{L}; xx++)
    for (long y=0; y<{L}; y++)
      memcpy(dst + (xx*{SLN} + y*{PZ})*2, src + (xx*{L2} + y*{L})*2, {L}*16);
}}
static void N_exp{L}(double* restrict dst, const double* restrict src){{
  for (long xx=0; xx<{L}; xx++)
    for (long y=0; y<{L}; y++)
      memcpy(dst + (xx*{L2} + y*{L})*2, src + (xx*{SLN} + y*{PZ})*2, {L}*16);
}}
"""
    return nat_finish(sec)

def gen_size_section(L, w):
    """codelets + step for size L at width w (S=2w, V=w)"""
    S = 2*w; V = w
    global MAPSTYLE
    _saved_ms = dict(MAPSTYLE)
    if w == 1:
        for _k in MAPSTYLE: MAPSTYLE[_k] = 'div' 
    st_z, st_y = 1, L
    use_scratch = PASSX[L][0] == 'scratch'
    st_x = PASSX[L][1] if use_scratch else SLAB(L)
    xname = 'xs' if use_scratch else 'x'
    ms = MAPSTYLE[L]
    parts = []
    if L in (6, 8):
        parts.append(gen_straight(L, S, V, 'z', st_z, False, ms))
        parts.append(gen_straight(L, S, V, 'y', st_y, False, ms))
        parts.append(gen_straight(L, S, V, xname, st_x, not use_scratch, ms))
    elif L in (13, 17, 23):
        parts.append(gen_prime(L, S, V, 'z', st_z, False, ms))
        parts.append(gen_prime(L, S, V, 'y', st_y, False, ms))
        parts.append(gen_prime(L, S, V, xname, st_x, not use_scratch, ms))
    elif L == 36:
        parts.append(gen_pfa_looped(36, 4, 9, S, V, 'z', st_z, False, ms))
        parts.append(gen_pfa_looped(36, 4, 9, S, V, 'y', st_y, False, ms))
        parts.append(gen_pfa_looped(36, 4, 9, S, V, xname, st_x, not use_scratch, ms))
    elif L == 45:
        parts.append(gen_pfa_looped(45, 9, 5, S, V, 'z', st_z, False, ms))
        parts.append(gen_pfa_looped(45, 9, 5, S, V, 'y', st_y, False, ms))
        parts.append(gen_pfa_looped(45, 9, 5, S, V, xname, st_x, not use_scratch, ms))
    elif L == 64:
        parts.append(gen_64(S, V, 'z', st_z, False, ms, False))
        parts.append(gen_64(S, V, 'y', st_y, False, ms, False))
        parts.append(gen_64(S, V, xname, st_x, not use_scratch, ms, False))
    parts.append(gen_step(L, S, V))
    txt = "\n".join(parts)
    # width token substitution
    rep = {8: [("T_RSQ","vrsqrt8"),("T_RCP","vrcp8"),("T_SQT","vsqrt8"),("T_NT","vstream8"),("T_LD","vld8"),
               ("T_ST","vst8"),("T_BC","vbc8"),("T_VD","vd8"),("VSET1","vset8"),
               ("VDT","vd8"),("VDA","VA64"),("T_V","8")],
           4: [("T_RSQ","vrsqrt4"),("T_RCP","vrcp4"),("T_SQT","vsqrt4"),("T_NT","vstream4"),("T_LD","vld4"),
               ("T_ST","vst4"),("T_BC","vbc4"),("T_VD","vd4"),("VSET1","vset4"),
               ("VDT","vd4"),("VDA","VA32"),("T_V","4")],
           2: [("T_RSQ","vrsqrt2"),("T_RCP","vrcp2"),("T_SQT","vsqrt2"),("T_NT","vstream2"),("T_LD","vld2"),
               ("T_ST","vst2"),("T_BC","vbc2"),("T_VD","vd2"),("VSET1","vset2"),
               ("VDT","vd2"),("VDA","VA16"),("T_V","2")],
           1: [("T_RSQ","vrsqrt1"),("T_RCP","vrcp1"),("T_SQT","vsqrt1"),("T_NT","vst1x"),("T_LD","vld1x"),
               ("T_ST","vst1x"),("T_BC","vld1x"),("T_VD","double"),("VSET1","(double)"),
               ("VDT","double"),("VDA","VA16"),("T_V","1")]}[w]
    for a, b in rep:
        txt = txt.replace(a, b)
    txt = txt.replace("W_", f"W{w}_")
    txt = txt.replace("SCR", f"SCR{w}_")
    txt = txt.replace(f"IN{L}_", f"W{w}IN{L}_").replace(f"OUT{L}_", f"W{w}OUT{L}_")
    txt = txt.replace(f"PC{L}_", f"W{w}PC{L}_").replace(f"PO{L}_", f"W{w}PO{L}_")
    MAPSTYLE.clear(); MAPSTYLE.update(_saved_ms)
    return txt

PROLOGUE = r"""
// ============================================================================
// Iterated batched 3D complex FFT + nonlinear map, specialized for
// L in {6, 8, 13, 17, 23, 36, 45, 64}.   (generated by gen_impl.py)
//
// All transform arithmetic in this file is original, generated from first
// principles (no FFTW/MKL/pocketfft code or tables are used anywhere):
//   * composite sizes via prime-factor (Good-Thomas) index maps:
//       6 = 2*3, 36 = 4*9, 45 = 9*5 (no twiddles), 64 = 8*8 Cooley-Tukey
//   * odd primes 13/17/23 via the symmetric half-length direct form
//       X[k] = x0 + sum_j cos(2pi jk/L) e_j - i * sum_j sin(2pi jk/L) o_j,
//       e_j = x_j + x_{L-j}, o_j = x_j - x_{L-j}, k-tiled for ILP
//   * twiddles/cosine tables computed in long double and rounded once.
//
// Data layouts:
//   * batch-SIMD "lane" layout: W widths 8/4/2/1 volumes interleaved per
//     vector lane; split re/im blocks of 2V doubles per element; slab padding
//     to avoid power-of-two strides.
//   * "natural" interleaved-complex layout (single volume, vectorized across
//     the contiguous z axis with swap/fmaddsub tricks) for 36/45/64 and small
//     tails of 23; z rows padded to a multiple of 4 complex.
// Each group of volumes is driven through ALL m iterations while cache
// resident.  Per step: pass A = z-FFT + y-FFT per x-slab; pass B = x-FFT
// (via contiguous scratch tiles) fused with z = FFT3(x) + c,
// x <- z/(1+|z|) using rsqrt14+Newton (guarded) on the store path.
// ============================================================================
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include <sys/mman.h>

#define INLINE inline __attribute__((always_inline))
#define VA64 __attribute__((aligned(64)))
#define VA32 __attribute__((aligned(32)))
#define PF(p) _mm_prefetch((const char*)(p), _MM_HINT_T0)

#if defined(__AVX512F__)
#define HAVE512 1
#else
#define HAVE512 0
#endif

typedef double vd4 __attribute__((vector_size(32), may_alias, aligned(32)));
static INLINE vd4 vld4(const double* p){ return *(const vd4*)p; }
static INLINE void vst4(double* p, vd4 x){ *(vd4*)p = x; }
#define vset4(c) ((vd4){(c),(c),(c),(c)})
static INLINE vd4 vbc4(const double* p){ double c = *p; return vset4(c); }
static INLINE vd4 vsqrt4(vd4 x){ return (vd4)_mm256_sqrt_pd((__m256d)x); }
static INLINE void vstream4(double* p, vd4 x){ _mm256_stream_pd(p, (__m256d)x); }
#if defined(__AVX512VL__)
static INLINE vd4 vrsqrt4(vd4 x){ return (vd4)_mm256_rsqrt14_pd((__m256d)x); }
static INLINE vd4 vrcp4(vd4 x){ return (vd4)_mm256_rcp14_pd((__m256d)x); }
#else
static INLINE vd4 vrsqrt4(vd4 x){ return vset4(1.0)/vsqrt4(x); }
static INLINE vd4 vrcp4(vd4 x){ return vset4(1.0)/x; }
#endif

#define vld1x(p) (*(p))
#define vst1x(p, x) (*(p) = (x))
static INLINE double vsqrt1(double x){ return __builtin_sqrt(x); }
static INLINE double vrsqrt1(double x){ return 1.0/__builtin_sqrt(x); }
static INLINE double vrcp1(double x){ return 1.0/x; }
static void imp1(double* restrict dst, const double* restrict src, long b0, int nl, long L){
  long L2=L*L, SLB=L2+2;
  const double* s = src + ((size_t)b0)*L2*L*2;
  for (long xx=0; xx<L; xx++){
    double* d = dst + (size_t)xx*SLB*2;
    for (long t=0; t<L2; t++){ d[t*2] = s[((size_t)xx*L2+t)*2]; d[t*2+1] = s[((size_t)xx*L2+t)*2+1]; }
  }
  (void)nl;
}
static void exp1(double* restrict dst, const double* restrict src, long b0, int nl, long L){
  long L2=L*L, SLB=L2+2;
  double* d = dst + ((size_t)b0)*L2*L*2;
  for (long xx=0; xx<L; xx++){
    const double* s = src + (size_t)xx*SLB*2;
    for (long t=0; t<L2; t++){ d[((size_t)xx*L2+t)*2] = s[t*2]; d[((size_t)xx*L2+t)*2+1] = s[t*2+1]; }
  }
  (void)nl;
}
typedef double vd2 __attribute__((vector_size(16), may_alias, aligned(16)));
static INLINE vd2 vld2(const double* p){ return *(const vd2*)p; }
static INLINE void vst2(double* p, vd2 x){ *(vd2*)p = x; }
#define vset2(c) ((vd2){(c),(c)})
static INLINE vd2 vbc2(const double* p){ double c = *p; return vset2(c); }
static INLINE vd2 vsqrt2(vd2 x){ return (vd2)_mm_sqrt_pd((__m128d)x); }
static INLINE void vstream2(double* p, vd2 x){ _mm_stream_pd(p, (__m128d)x); }
#if defined(__AVX512VL__)
static INLINE vd2 vrsqrt2(vd2 x){ return (vd2)_mm_rsqrt14_pd((__m128d)x); }
static INLINE vd2 vrcp2(vd2 x){ return (vd2)_mm_rcp14_pd((__m128d)x); }
#else
static INLINE vd2 vrsqrt2(vd2 x){ return vset2(1.0)/vsqrt2(x); }
static INLINE vd2 vrcp2(vd2 x){ return vset2(1.0)/x; }
#endif
#define VA16 __attribute__((aligned(16)))

#if HAVE512
typedef double vd8 __attribute__((vector_size(64), may_alias, aligned(64)));
static INLINE vd8 vld8(const double* p){ return *(const vd8*)p; }
static INLINE void vst8(double* p, vd8 x){ *(vd8*)p = x; }
#define vset8(c) ((vd8){(c),(c),(c),(c),(c),(c),(c),(c)})
static INLINE vd8 vbc8(const double* p){ double c = *p; return vset8(c); }
static INLINE vd8 vsqrt8(vd8 x){ return (vd8)_mm512_sqrt_pd((__m512d)x); }
static INLINE vd8 vld8u(const double* p){ return (vd8)_mm512_loadu_pd(p); }
static INLINE void vst8u(double* p, vd8 x){ _mm512_storeu_pd(p, (__m512d)x); }
static INLINE vd8 vswp8(vd8 x){ return (vd8)_mm512_permute_pd((__m512d)x, 0x55); }
static INLINE vd8 vfmas8(vd8 a, vd8 b, vd8 c){ return (vd8)_mm512_fmaddsub_pd((__m512d)a,(__m512d)b,(__m512d)c); }
static INLINE vd8 vfmsa8(vd8 a, vd8 b, vd8 c){ return (vd8)_mm512_fmsubadd_pd((__m512d)a,(__m512d)b,(__m512d)c); }
static INLINE void tr4c(__m512d v[4]){
  __m512d t0,t1,t2,t3;
  t0=_mm512_shuffle_f64x2(v[0],v[1],0x44); t1=_mm512_shuffle_f64x2(v[0],v[1],0xEE);
  t2=_mm512_shuffle_f64x2(v[2],v[3],0x44); t3=_mm512_shuffle_f64x2(v[2],v[3],0xEE);
  v[0]=_mm512_shuffle_f64x2(t0,t2,0x88); v[1]=_mm512_shuffle_f64x2(t0,t2,0xDD);
  v[2]=_mm512_shuffle_f64x2(t1,t3,0x88); v[3]=_mm512_shuffle_f64x2(t1,t3,0xDD);
}
static INLINE void vstream8(double* p, vd8 x){ _mm512_stream_pd(p, (__m512d)x); }
static INLINE vd8 vrsqrt8(vd8 x){ return (vd8)_mm512_rsqrt14_pd((__m512d)x); }
static INLINE vd8 vrcp8(vd8 x){ return (vd8)_mm512_rcp14_pd((__m512d)x); }
#endif

// ---------------- buffers ----------------
typedef struct { double* p; size_t bytes; } buf_t;
static void ensure(buf_t* b, size_t bytes){
  if (b->bytes >= bytes) return;
  if (b->p) munmap(b->p, b->bytes);
  size_t r = (bytes + ((size_t)2<<20) - 1) & ~(((size_t)2<<20)-1);
  void* q = mmap(NULL, r, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
  if (q == MAP_FAILED){ b->p = 0; b->bytes = 0; return; }
  madvise(q, r, MADV_HUGEPAGE);
  b->p = (double*)q; b->bytes = r;
}

// ---------------- import / export ----------------
#if HAVE512
static INLINE void tr8(__m512d v[8]){
  __m512d t0,t1,t2,t3,t4,t5,t6,t7, ua,ub,uc,ud,ue,uf,ug,uh;
  t0=_mm512_unpacklo_pd(v[0],v[1]); t1=_mm512_unpackhi_pd(v[0],v[1]);
  t2=_mm512_unpacklo_pd(v[2],v[3]); t3=_mm512_unpackhi_pd(v[2],v[3]);
  t4=_mm512_unpacklo_pd(v[4],v[5]); t5=_mm512_unpackhi_pd(v[4],v[5]);
  t6=_mm512_unpacklo_pd(v[6],v[7]); t7=_mm512_unpackhi_pd(v[6],v[7]);
  ua=_mm512_shuffle_f64x2(t0,t2,0x88); ub=_mm512_shuffle_f64x2(t1,t3,0x88);
  uc=_mm512_shuffle_f64x2(t0,t2,0xDD); ud=_mm512_shuffle_f64x2(t1,t3,0xDD);
  ue=_mm512_shuffle_f64x2(t4,t6,0x88); uf=_mm512_shuffle_f64x2(t5,t7,0x88);
  ug=_mm512_shuffle_f64x2(t4,t6,0xDD); uh=_mm512_shuffle_f64x2(t5,t7,0xDD);
  v[0]=_mm512_shuffle_f64x2(ua,ue,0x88); v[1]=_mm512_shuffle_f64x2(ub,uf,0x88);
  v[2]=_mm512_shuffle_f64x2(uc,ug,0x88); v[3]=_mm512_shuffle_f64x2(ud,uh,0x88);
  v[4]=_mm512_shuffle_f64x2(ua,ue,0xDD); v[5]=_mm512_shuffle_f64x2(ub,uf,0xDD);
  v[6]=_mm512_shuffle_f64x2(uc,ug,0xDD); v[7]=_mm512_shuffle_f64x2(ud,uh,0xDD);
}
static void imp8(double* restrict dst, const double* restrict src, long b0, int nl, long L){
  long L2=L*L, SLB=L2+2;
  const double* sp[8];
  for (int v=0; v<8; v++) sp[v] = (v<nl) ? src + ((size_t)(b0+v))*L2*L*2 : 0;
  for (long xx=0; xx<L; xx++){
    double* d = dst + (size_t)xx*SLB*16;
    long t = 0;
    for (; t+4<=L2; t+=4){
      __m512d in[8];
      for (int v=0; v<8; v++) in[v] = sp[v] ? _mm512_loadu_pd(sp[v] + ((size_t)xx*L2 + t)*2) : _mm512_setzero_pd();
      tr8(in);
      for (int k=0; k<4; k++){ _mm512_store_pd(d+(t+k)*16, in[2*k]); _mm512_store_pd(d+(t+k)*16+8, in[2*k+1]); }
    }
    for (; t<L2; t++)
      for (int v=0; v<8; v++){
        d[t*16+v]   = sp[v] ? sp[v][((size_t)xx*L2+t)*2]   : 0.0;
        d[t*16+8+v] = sp[v] ? sp[v][((size_t)xx*L2+t)*2+1] : 0.0;
      }
  }
}
static void exp8(double* restrict dst, const double* restrict src, long b0, int nl, long L){
  long L2=L*L, SLB=L2+2;
  double* dp[8];
  for (int v=0; v<8; v++) dp[v] = (v<nl) ? dst + ((size_t)(b0+v))*L2*L*2 : 0;
  for (long xx=0; xx<L; xx++){
    const double* s = src + (size_t)xx*SLB*16;
    long t = 0;
    for (; t+4<=L2; t+=4){
      __m512d in[8];
      for (int k=0; k<4; k++){ in[2*k]=_mm512_load_pd(s+(t+k)*16); in[2*k+1]=_mm512_load_pd(s+(t+k)*16+8); }
      tr8(in);
      for (int v=0; v<8; v++) if (dp[v]) _mm512_storeu_pd(dp[v] + ((size_t)xx*L2+t)*2, in[v]);
    }
    for (; t<L2; t++)
      for (int v=0; v<8; v++) if (dp[v]){ dp[v][((size_t)xx*L2+t)*2] = s[t*16+v]; dp[v][((size_t)xx*L2+t)*2+1] = s[t*16+8+v]; }
  }
}
#else
static void imp8(double* restrict dst, const double* restrict src, long b0, int nl, long L){ (void)dst;(void)src;(void)b0;(void)nl;(void)L; }
static void exp8(double* restrict dst, const double* restrict src, long b0, int nl, long L){ (void)dst;(void)src;(void)b0;(void)nl;(void)L; }
#endif

static INLINE void tr4(__m256d v[4]){
  __m256d t0,t1,t2,t3;
  t0=_mm256_unpacklo_pd(v[0],v[1]); t1=_mm256_unpackhi_pd(v[0],v[1]);
  t2=_mm256_unpacklo_pd(v[2],v[3]); t3=_mm256_unpackhi_pd(v[2],v[3]);
  v[0]=_mm256_permute2f128_pd(t0,t2,0x20); v[1]=_mm256_permute2f128_pd(t1,t3,0x20);
  v[2]=_mm256_permute2f128_pd(t0,t2,0x31); v[3]=_mm256_permute2f128_pd(t1,t3,0x31);
}
static void imp4(double* restrict dst, const double* restrict src, long b0, int nl, long L){
  long L2=L*L, SLB=L2+2;
  const double* sp[4];
  for (int v=0; v<4; v++) sp[v] = (v<nl) ? src + ((size_t)(b0+v))*L2*L*2 : 0;
  for (long xx=0; xx<L; xx++){
    double* d = dst + (size_t)xx*SLB*8;
    long t = 0;
    for (; t+2<=L2; t+=2){
      __m256d in[4];
      for (int v=0; v<4; v++) in[v] = sp[v] ? _mm256_loadu_pd(sp[v] + ((size_t)xx*L2 + t)*2) : _mm256_setzero_pd();
      tr4(in);
      for (int k=0; k<2; k++){ _mm256_store_pd(d+(t+k)*8, in[2*k]); _mm256_store_pd(d+(t+k)*8+4, in[2*k+1]); }
    }
    for (; t<L2; t++)
      for (int v=0; v<4; v++){
        d[t*8+v]   = sp[v] ? sp[v][((size_t)xx*L2+t)*2]   : 0.0;
        d[t*8+4+v] = sp[v] ? sp[v][((size_t)xx*L2+t)*2+1] : 0.0;
      }
  }
}
static void imp2(double* restrict dst, const double* restrict src, long b0, int nl, long L){
  long L2=L*L, SLB=L2+2;
  const double* sp[2];
  for (int v=0; v<2; v++) sp[v] = (v<nl) ? src + ((size_t)(b0+v))*L2*L*2 : 0;
  for (long xx=0; xx<L; xx++){
    double* d = dst + (size_t)xx*SLB*4;
    for (long t=0; t<L2; t++){
      __m128d a = sp[0] ? _mm_loadu_pd(sp[0] + ((size_t)xx*L2+t)*2) : _mm_setzero_pd();
      __m128d b = sp[1] ? _mm_loadu_pd(sp[1] + ((size_t)xx*L2+t)*2) : _mm_setzero_pd();
      _mm_store_pd(d + t*4,     _mm_unpacklo_pd(a, b));
      _mm_store_pd(d + t*4 + 2, _mm_unpackhi_pd(a, b));
    }
  }
}
static void exp2v(double* restrict dst, const double* restrict src, long b0, int nl, long L){
  long L2=L*L, SLB=L2+2;
  double* dp[2];
  for (int v=0; v<2; v++) dp[v] = (v<nl) ? dst + ((size_t)(b0+v))*L2*L*2 : 0;
  for (long xx=0; xx<L; xx++){
    const double* s = src + (size_t)xx*SLB*4;
    for (long t=0; t<L2; t++){
      __m128d re = _mm_load_pd(s + t*4), im = _mm_load_pd(s + t*4 + 2);
      if (dp[0]) _mm_storeu_pd(dp[0] + ((size_t)xx*L2+t)*2, _mm_unpacklo_pd(re, im));
      if (dp[1]) _mm_storeu_pd(dp[1] + ((size_t)xx*L2+t)*2, _mm_unpackhi_pd(re, im));
    }
  }
}
static void exp4(double* restrict dst, const double* restrict src, long b0, int nl, long L){
  long L2=L*L, SLB=L2+2;
  double* dp[4];
  for (int v=0; v<4; v++) dp[v] = (v<nl) ? dst + ((size_t)(b0+v))*L2*L*2 : 0;
  for (long xx=0; xx<L; xx++){
    const double* s = src + (size_t)xx*SLB*8;
    long t = 0;
    for (; t+2<=L2; t+=2){
      __m256d in[4];
      for (int k=0; k<2; k++){ in[2*k]=_mm256_load_pd(s+(t+k)*8); in[2*k+1]=_mm256_load_pd(s+(t+k)*8+4); }
      tr4(in);
      for (int v=0; v<4; v++) if (dp[v]) _mm256_storeu_pd(dp[v] + ((size_t)xx*L2+t)*2, in[v]);
    }
    for (; t<L2; t++)
      for (int v=0; v<4; v++) if (dp[v]){ dp[v][((size_t)xx*L2+t)*2] = s[t*8+v]; dp[v][((size_t)xx*L2+t)*2+1] = s[t*8+4+v]; }
  }
}
"""

def gen_run(L):
    SL = SLAB(L); VOL = L*SL; L3 = L**3
    prefw = PREFW[L]
    has2 = L in W2SET
    has1 = L in W1SET
    hasnat = L in NATSET
    natalways = hasnat and NATTHRESH.get(L,3) > (1<<20)
    PZv = ((L+3)//4)*4
    SLNv = L*PZv + 4
    BUFSZ = (L*SLNv*2*8 + 4*PZv*16 + 4096) if natalways else (VOL*16*8)
    natblock = ("""#if HAVE512
    if (rem <= {NATTH}){{
      while (b < B){{
        N_imp{L}(X, x0 + (size_t)b*{L3}*2);
        N_imp{L}(C, c0 + (size_t)b*{L3}*2);
        for (long long it=0; it<m; it++){{
          N_step{L}(X, C);
          if (it == 0) N_exp{L}(o1 + (size_t)b*{L3}*2, X);
        }}
        N_exp{L}(om + (size_t)b*{L3}*2, X);
        b++;
      }}
      break;
    }}
#endif
""".format(L3=L3, L=L, NATTH=NATTHRESH.get(L,3)) if hasnat else "")
    # group-width selection: use prefw for full groups; for tail pick smallest available >= rem
    widths_desc = [8, 4] + ([2] if has2 else [])
    # body for one group at width w
    def grp(w):
        impf = {8:'imp8',4:'imp4',2:'imp2',1:'imp1'}[w]
        expf = {8:'exp8',4:'exp4',2:'exp2v',1:'exp1'}[w]
        return f"""{{
      {impf}(X, x0, b, nl, {L}); {impf}(C, c0, b, nl, {L});
      for (long long it=0; it<m; it++){{
        W{w}_step{L}(X, C, 1);
        if (it == 0) {expf}(o1, X, b, nl, {L});
      }}
      {expf}(om, X, b, nl, {L});
    }}"""
    w2sel = ""
    if has2:
        w2sel = "if (rem <= 2) w = 2;\n      "
    if has1:
        w2sel += "if (rem <= 1) w = 1;\n      "
    return f"""
static buf_t xb{L}, cb{L};
void* getxb{L}(void){{ return xb{L}.p; }}
void* getcb{L}(void){{ return cb{L}.p; }}
void bench{L}(long G, long reps, int which){{
#if HAVE512
  double* X = xb{L}.p; double* C = cb{L}.p;
  for (long r=0; r<reps; r++)
    for (long g=0; g<G; g++){{
      double* p = X + (size_t)g*{VOL}*16;
      if (which & 1) W8_passzy{L}(p);
      if (which & 4) W8_passx{L}(p, C + (size_t)g*{VOL}*16);
    }}
#endif
}}
void cbench{L}(long reps, int axis){{
#if HAVE512
  double* X = xb{L}.p; double* C = cb{L}.p;
  if (axis == 0) for (long r=0; r<reps; r++) W8_d{L}_z(X);
  if (axis == 1) for (long r=0; r<reps; r++) W8_d{L}_y(X);
#endif
}}
void run{L}(const double* x0, const double* c0, long long B, long long m, double* o1, double* om){{
  if (B <= 0) return;
  if (m < 1) m = 1;
  unsigned mxcsr0 = _mm_getcsr();
  _mm_setcsr(mxcsr0 | 0x8040u);  /* FTZ|DAZ */
#if HAVE512
  ensure(&xb{L}, (size_t){BUFSZ}); ensure(&cb{L}, (size_t){BUFSZ});
#else
  ensure(&xb{L}, (size_t){VOL}*16*8); ensure(&cb{L}, (size_t){VOL}*16*8);
#endif
  double* X = xb{L}.p; double* C = cb{L}.p;
  long long b = 0;
  int prefw = {prefw};
#if !HAVE512
  if (prefw > 4) prefw = 4;
#endif
  while (b < B){{
    long long rem = B - b;
{natblock}    int w = prefw;
    if (rem < prefw){{
      w = prefw;
      if (rem <= 4 && prefw > 4) w = 4;
      {w2sel}}}
    int nl = (rem >= w) ? w : (int)rem;
#if HAVE512
    if (w == 8){grp(8)} else
#endif
    if (w == 4){grp(4)}"""+ (f""" else if (w == 2){grp(2)}""" if has2 else "") + (f""" else {grp(1)}""" if has1 else "") + f"""
    b += nl;
  }}
  _mm_setcsr(mxcsr0);
}}
"""

def gen_prewarm():
    lines = []
    for L in SIZES:
        SL = SLAB(L); VOL = L*SL
        PZv = ((L+3)//4)*4; SLNv = L*PZv + 4
        natalways = (L in NATSET) and NATTHRESH.get(L,3) > (1<<20)
        BUFSZ = (L*SLNv*2*8 + 4*PZv*16 + 4096) if natalways else (VOL*16*8)
        lines.append(f"  ensure(&xb{L}, (size_t){BUFSZ}); ensure(&cb{L}, (size_t){BUFSZ});")
        lines.append(f"  if (xb{L}.p) memset(xb{L}.p, 0, xb{L}.bytes);")
        lines.append(f"  if (cb{L}.p) memset(cb{L}.p, 0, cb{L}.bytes);")
    return "void prewarm(void){\n" + "\n".join(lines) + "\n}\n"

def main():
    out = [PROLOGUE]
    # TW64 tables
    twr = []; twi = []
    for b in range(8):
        for d in range(8):
            c, s = tw(b*d, 64)
            twr.append(float(c).hex()); twi.append(float(s).hex())
    out.append(f"static const double TW64R[64] = {{{','.join(twr)}}};\n")
    out.append(f"static const double TW64I[64] = {{{','.join(twi)}}};\n")
    for L in SIZES:
        out.append(f"// ================= L = {L} =================\n")
        out.append("#if HAVE512\n")
        out.append(gen_size_section(L, 8))
        out.append("#endif\n")
        out.append(gen_size_section(L, 4))
        if L in W2SET:
            out.append(gen_size_section(L, 2))
        if L in W1SET:
            out.append(gen_size_section(L, 1))
        if L in NATSET:
            out.append("#if HAVE512\n" + gen_nat_section(L) + "#endif\n")
        out.append(gen_run(L))
    out.append(gen_prewarm())
    open("implementation.c", "w").write("\n".join(out))
    print("wrote implementation.c,", sum(len(x.splitlines()) for x in out), "lines")

main()
