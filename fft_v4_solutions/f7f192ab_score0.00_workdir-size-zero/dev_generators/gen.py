#!/usr/bin/env python3
"""Generator for implementation.c: specialized AVX-512 batched 3D FFT + map
for L in (6,8,13,17,23,36,45,64).

Design:
 - All 1D kernels operate on "rows" (transform index r = 0..L-1) each holding a
   contiguous batch of complex128, SIMD vertical across the batch (zmm = 4 complex).
 - y-pass: rows strided by L complex within an (y,z) slice, batch = L (z).
 - x-pass: rows strided by L^2, batch = L (z) for each y; fused +c and z/(1+|z|) map.
 - z-pass: lines are contiguous; groups of 4 lines transposed (4x4 complex blocks)
   into a T[] array of zmm (lane = line), kernel on T, transpose back.
 - composites 6,8: fully in registers. 36=4x9, 45=9x5: PFA (no twiddles), buffered.
   64 = 8x8 Cooley-Tukey with twiddles, buffered. 13,17,23: symmetric direct
   ("half-matrix") with FMA-packed k-block accumulation.
Every emitted body is verified numerically in numpy-mode during generation.
"""
import numpy as np, math

# ----------------------------------------------------------------------------
class Emitter:
    def __init__(self):
        self.lines = []
        self.n = 0
        self.ind = 0
    def w(self, s):
        self.lines.append("    "*self.ind + s)
    def fresh(self, pfx="t"):
        self.n += 1
        return f"{pfx}{self.n}"
    def code(self):
        return "\n".join(self.lines)

class Val:
    __slots__ = ("e","v")
    def __init__(self, e, v):
        self.e = e   # C expression (a variable name)
        self.v = v   # numpy complex value (vector over basis) or None

class Ctx:
    """emission context: wraps an Emitter and provides dual-mode ops"""
    def __init__(self, em, check=True):
        self.em = em
        self.check = check
    def _mk(self, expr, val, pfx="v"):
        name = self.em.fresh(pfx)
        self.em.w(f"V8 {name} = {expr};")
        return Val(name, val)
    def add(self, a, b):
        return self._mk(f"ADD({a.e},{b.e})", None if a.v is None else a.v + b.v)
    def sub(self, a, b):
        return self._mk(f"SUB({a.e},{b.e})", None if a.v is None else a.v - b.v)
    def mulr(self, a, r):  # r = (expr, value) real scalar broadcast
        return self._mk(f"MUL(SET1({r[0]}),{a.e})", None if a.v is None else a.v * r[1])
    def fmar(self, r, a, b):  # r*a + b
        return self._mk(f"FMA(SET1({r[0]}),{a.e},{b.e})", None if a.v is None else r[1]*a.v + b.v)
    def fnmar(self, r, a, b):  # b - r*a
        return self._mk(f"FNMA(SET1({r[0]}),{a.e},{b.e})", None if a.v is None else b.v - r[1]*a.v)
    def muli(self, a):   # i*a   : xor(swap, SGN_E)
        return self._mk(f"XOR(SWAP({a.e}),SGN_E)", None if a.v is None else 1j*a.v)
    def mulmi(self, a):  # -i*a  : xor(swap, SGN_O)
        return self._mk(f"XOR(SWAP({a.e}),SGN_O)", None if a.v is None else -1j*a.v)
    def neg(self, a):
        return self._mk(f"XOR({a.e},SGN_B)", None if a.v is None else -a.v)
    def cmul(self, a, wre, wim, wval):
        # multiply by complex constant w = wre + i*wim (exprs+value)
        # t = swap(a); m = wim*t ; res = fmaddsub(wre, a, m) -> (wre*re - wim*im, wre*im + wim*re)
        t = self._mk(f"SWAP({a.e})", None if a.v is None else 1j*np.conj(a.v))
        m = self._mk(f"MUL(SET1({wim}),{t.e})", None if t.v is None else wim_mul(t.v, 0))
        # note: for verification compute directly:
        name = self.em.fresh("v")
        self.em.w(f"V8 {name} = FMAS(SET1({wre}),{a.e},{m.e});")
        return Val(name, None if a.v is None else wval * a.v)
    def pairpm(self, a, b):
        """return (a + i*b, a - i*b)"""
        ib = self.muli(b)
        return self.add(a, ib), self.sub(a, ib)

def wim_mul(x, dummy):
    return x  # placeholder (cmul verification computed directly)

# ----------------------------------------------------------------------------
# table management
class Tables:
    """Each request gets its own slot (no dedupe) so gcc folds each use as an
    embedded-broadcast memory operand instead of keeping values live."""
    def __init__(self):
        self.arrays = {}   # name -> list of (num, den, fn)
    def trig(self, arr, num, den, fn):
        lst = self.arrays.setdefault(arr, [])
        key = ((num % den), den, fn)
        lst.append(key)
        i = len(lst)-1
        n_,d_,f_ = key
        ang = -2*math.pi*n_/d_
        if f_=='cos': val = math.cos(ang)
        elif f_=='sin': val = math.sin(ang)
        else: val = -math.sin(ang)
        return (f"{arr}[{i}]", val)
    def w(self, arr, num, den):
        cr = self.trig(arr, num, den, 'cos')
        ci = self.trig(arr, num, den, 'sin')
        ang = -2*math.pi*(num % den)/den
        return cr[0], ci[0], complex(math.cos(ang), math.sin(ang))
    def cosp(self, arr, num, den):
        return self.trig(arr, num, den, 'cos')
    def sinp(self, arr, num, den):   # sin(+2 pi num/den)
        return self.trig(arr, num, den, 'nsin')
    def decls(self):
        out = []
        for name, lst in self.arrays.items():
            out.append(f"static double {name}[{max(1,len(lst))}] __attribute__((aligned(64)));")
        return "\n".join(out)
    def init_code(self):
        out = []
        for name, lst in self.arrays.items():
            for i,(n,d,f) in enumerate(lst):
                if f == 'cos':
                    out.append(f"    {name}[{i}] = (double)cosl(PIL2 * {n}.0L / {d}.0L);")
                elif f == 'sin':
                    out.append(f"    {name}[{i}] = (double)sinl(PIL2 * {n}.0L / {d}.0L);")
                else:
                    out.append(f"    {name}[{i}] = (double)(-sinl(PIL2 * {n}.0L / {d}.0L));")
        return "\n".join(out)

TAB = Tables()

# ----------------------------------------------------------------------------
# DFT building blocks. Each takes ctx, list of input Vals, returns output Vals.
def dft2(c, xs):
    return [c.add(xs[0], xs[1]), c.sub(xs[0], xs[1])]

def dft4(c, xs):
    t0 = c.add(xs[0], xs[2]); t1 = c.sub(xs[0], xs[2])
    t2 = c.add(xs[1], xs[3]); t3 = c.sub(xs[1], xs[3])
    y0 = c.add(t0, t2); y2 = c.sub(t0, t2)
    # y1 = t1 - i t3 ; y3 = t1 + i t3
    it3 = c.muli(t3)
    y1 = c.sub(t1, it3); y3 = c.add(t1, it3)
    return [y0, y1, y2, y3]

def dft_odd_direct(c, xs, N, arr):
    """odd prime-ish direct symmetric DFT for small N (3,5)"""
    h = (N-1)//2
    s = {}; dp = {}
    for j in range(1, h+1):
        s[j] = c.add(xs[j], xs[N-j])
        d = c.sub(xs[j], xs[N-j])
        dp[j] = c.mulmi(d)     # -i * d
    # y0
    acc = xs[0]
    for j in range(1, h+1):
        acc = c.add(acc, s[j])
    ys = [None]*N
    ys[0] = acc
    for k in range(1, h+1):
        A = xs[0]; T = None
        for j in range(1, h+1):
            ckj = TAB.cosp(arr, (k*j) % N, N)
            A = c.fmar(ckj, s[j], A)
            skj = TAB.sinp(arr, (k*j) % N, N)
            if T is None:
                T = c.mulr(dp[j], skj)
            else:
                T = c.fmar(skj, dp[j], T)
        ys[k] = c.add(A, T)
        ys[N-k] = c.sub(A, T)
    return ys

def dft3(c, xs): return dft_odd_direct(c, xs, 3, "W3")
def dft5(c, xs): return dft_odd_direct(c, xs, 5, "W5")

def dft8(c, xs):
    E = dft4(c, [xs[0], xs[2], xs[4], xs[6]])
    O = dft4(c, [xs[1], xs[3], xs[5], xs[7]])
    # twiddles on O[k], k=0..3 : w8^k
    RH = ("RSQ2", math.sqrt(0.5))
    # k=1: w=(1-i)/sqrt2: (re+im, im-re)*r  -> fmsubadd(one, v, swap(v)) * r
    v = O[1]
    t = c._mk(f"SWAP({v.e})", None if v.v is None else 1j*np.conj(v.v))
    q = c._mk(f"FMSA(ONE,{v.e},{t.e})", None if v.v is None else (1-1j)*v.v)
    O1 = c.mulr(q, RH)
    # k=2: -i
    O2 = c.mulmi(O[2])
    # k=3: -(1+i)/sqrt2
    v = O[3]
    t = c._mk(f"SWAP({v.e})", None if v.v is None else 1j*np.conj(v.v))
    q = c._mk(f"FMAS(ONE,{v.e},{t.e})", None if v.v is None else (1+1j)*v.v)
    O3 = c.mulr(q, ("MRSQ2", -math.sqrt(0.5)))
    Ot = [O[0], O1, O2, O3]
    ys = [None]*8
    for k in range(4):
        ys[k] = c.add(E[k], Ot[k])
        ys[k+4] = c.sub(E[k], Ot[k])
    return ys

def dft9(c, xs):
    # 3x3 CT: j = j0 + 3 j1 ; k = k0 + 3 k1
    y = {}
    for j0 in range(3):
        sub = dft3(c, [xs[j0], xs[j0+3], xs[j0+6]])
        for k0 in range(3):
            v = sub[k0]
            tw = (j0*k0) % 9
            if tw != 0:
                wre, wim, wv = TAB.w("W9", tw, 9)
                v = c.cmul(v, wre, wim, wv)
            y[(j0,k0)] = v
    ys = [None]*9
    for k0 in range(3):
        sub = dft3(c, [y[(0,k0)], y[(1,k0)], y[(2,k0)]])
        for k1 in range(3):
            ys[k0 + 3*k1] = sub[k1]
    return ys

DFT_PRIM = {2: dft2, 3: dft3, 4: dft4, 5: dft5, 8: dft8, 9: dft9}

def modinv(a, m):
    return pow(a, -1, m)

def pfa_body(c, L, N1, N2, load, store):
    """PFA L = N1*N2 (coprime): stage A: N1 x DFT_N2 -> buf ; stage B: N2 x DFT_N1"""
    # input index n(n1,n2) = (n1*N2 + n2*N1) mod L
    # output index k(k1,k2): k ≡ k1 mod N1, k ≡ k2 mod N2 (CRT)
    crt1 = N2 * modinv(N2, N1)   # ≡1 mod N1, 0 mod N2
    crt2 = N1 * modinv(N1, N2)
    use_buf = L > 10
    buf = {}
    if use_buf:
        c.em.w(f"V8 buf[{L}];")
    for n1 in range(N1):
        xs = [load((n1*N2 + n2*N1) % L) for n2 in range(N2)]
        ys = DFT_PRIM[N2](c, xs)
        for k2 in range(N2):
            if use_buf:
                c.em.w(f"buf[{n1*N2+k2}] = {ys[k2].e};")
                buf[(n1,k2)] = Val(f"buf[{n1*N2+k2}]", ys[k2].v)
            else:
                buf[(n1,k2)] = ys[k2]
    for k2 in range(N2):
        xs = [buf[(n1,k2)] for n1 in range(N1)]
        ys = DFT_PRIM[N1](c, xs)
        for k1 in range(N1):
            store((crt1*k1 + crt2*k2) % L, ys[k1])

def apply_w64(c, v, t):
    """multiply Val v by w64^t (t already reduced mod 64); returns Val"""
    t = t % 64
    if t == 0:
        return v
    if t == 16:
        return c.mulmi(v)
    if t == 32:
        return c.neg(v)
    if t == 48:
        return c.muli(v)
    if t == 8:
        tv = c._mk(f"SWAP({v.e})", None if v.v is None else 1j*np.conj(v.v))
        q = c._mk(f"FMSA(ONE,{v.e},{tv.e})", None if v.v is None else (1-1j)*v.v)
        return c.mulr(q, ("RSQ2", math.sqrt(0.5)))
    if t == 24:
        tv = c._mk(f"SWAP({v.e})", None if v.v is None else 1j*np.conj(v.v))
        q = c._mk(f"FMAS(ONE,{v.e},{tv.e})", None if v.v is None else (1+1j)*v.v)
        return c.mulr(q, ("MRSQ2", -math.sqrt(0.5)))
    if t == 40:
        tv = c._mk(f"SWAP({v.e})", None if v.v is None else 1j*np.conj(v.v))
        q = c._mk(f"FMAS(ONE,{v.e},{tv.e})", None if v.v is None else (1+1j)*v.v)
        return c.mulr(q, ("RSQ2", math.sqrt(0.5)))
    if t == 56:
        tv = c._mk(f"SWAP({v.e})", None if v.v is None else 1j*np.conj(v.v))
        q = c._mk(f"FMSA(ONE,{v.e},{tv.e})", None if v.v is None else (1-1j)*v.v)
        return c.mulr(q, ("MRSQ2", -math.sqrt(0.5)))
    wre, wim, wv = TAB.w("W64", t, 64)
    return c.cmul(v, wre, wim, wv)

def ct64_body(c, load, store):
    """64 = 8 x 8 CT with twiddles; j = j0 + 8 j1, k = k0 + 8 k1"""
    c.em.w("V8 buf[64];")
    buf = {}
    for j0 in range(8):
        xs = [load(j0 + 8*j1) for j1 in range(8)]
        ys = dft8(c, xs)
        for k0 in range(8):
            v = apply_w64(c, ys[k0], j0*k0)
            c.em.w(f"buf[{j0*8+k0}] = {v.e};")
            buf[(j0,k0)] = Val(f"buf[{j0*8+k0}]", v.v)
    for k0 in range(8):
        xs = [buf[(j0,k0)] for j0 in range(8)]
        ys = dft8(c, xs)
        for k1 in range(8):
            store(k0 + 8*k1, ys[k1])

def composite_body(c, L, load, store):
    if L == 6:
        pfa_body(c, L, 2, 3, load, store)
    elif L == 8:
        xs = [load(j) for j in range(8)]
        ys = dft8(c, xs)
        for k in range(8):
            store(k, ys[k])
    elif L == 36:
        pfa_body(c, L, 4, 9, load, store)
    elif L == 45:
        pfa_body(c, L, 9, 5, load, store)
    elif L == 64:
        ct64_body(c, load, store)
    else:
        raise ValueError(L)

# ----------------------------------------------------------------------------
def prime_body(c, L, insts):
    """symmetric direct DFT for prime L; insts = list of (load,store), CU=1 or 2.
    constants shared across instances; k blocked to bound accumulator count."""
    h = (L-1)//2
    arr = f"WP{L}"
    em = c.em
    CU = len(insts)
    for i in range(CU):
        em.w(f"V8 sb{i}[{h+1}], db{i}[{h+1}];")
    x0r = []; acc0 = []
    sv = [dict() for _ in range(CU)]; dv = [dict() for _ in range(CU)]
    for i,(ld,stv) in enumerate(insts):
        x0r.append(ld(0))
    for j in range(1, h+1):
        for i,(ld,stv) in enumerate(insts):
            a = ld(j); b = ld(L-j)
            s = c.add(a, b)
            d = c.sub(a, b)
            dp = c.mulmi(d)
            em.w(f"sb{i}[{j}] = {s.e};")
            em.w(f"db{i}[{j}] = {dp.e};")
            sv[i][j] = s.v; dv[i][j] = dp.v
            if j == 1: acc0.append(s)
            else: acc0[i] = c.add(acc0[i], s)
    for i,(ld,stv) in enumerate(insts):
        stv(0, c.add(x0r[i], acc0[i]))
    KBmax = 6 if CU >= 2 else h
    ks = list(range(1, h+1))
    blocks = [ks[i:i+KBmax] for i in range(0, len(ks), KBmax)]
    for blk in blocks:
        A = {}; T = {}
        for k in blk:
            for i in range(CU):
                A[(k,i)] = Val(em.fresh("A"), x0r[i].v)
                em.w(f"V8 {A[(k,i)].e} = {x0r[i].e};")
                T[(k,i)] = Val(em.fresh("T"), None if x0r[i].v is None else np.zeros_like(x0r[i].v))
                em.w(f"V8 {T[(k,i)].e} = _mm512_setzero_pd();")
        for j in range(1, h+1):
            svars=[]; dvars=[]
            for i in range(CU):
                sn = em.fresh("s"); dn = em.fresh("d")
                em.w(f"V8 {sn} = sb{i}[{j}]; V8 {dn} = db{i}[{j}];")
                svars.append(Val(sn, sv[i][j])); dvars.append(Val(dn, dv[i][j]))
            for k in blk:
                ckj = TAB.cosp(arr, (k*j) % L, L)
                skj = TAB.sinp(arr, (k*j) % L, L)
                if CU == 1:
                    em.w(f"{A[(k,0)].e} = FMA(SET1({ckj[0]}),{svars[0].e},{A[(k,0)].e});")
                    em.w(f"{T[(k,0)].e} = FMA(SET1({skj[0]}),{dvars[0].e},{T[(k,0)].e});")
                else:
                    cb = em.fresh("cc"); sbv = em.fresh("ss")
                    em.w(f"V8 {cb} = SET1({ckj[0]});")
                    em.w(f"V8 {sbv} = SET1({skj[0]});")
                    for i in range(CU):
                        em.w(f"{A[(k,i)].e} = FMA({cb},{svars[i].e},{A[(k,i)].e});")
                        em.w(f"{T[(k,i)].e} = FMA({sbv},{dvars[i].e},{T[(k,i)].e});")
                for i in range(CU):
                    if A[(k,i)].v is not None:
                        A[(k,i)].v = A[(k,i)].v + ckj[1]*sv[i][j]
                        T[(k,i)].v = T[(k,i)].v + skj[1]*dv[i][j]
        for k in blk:
            for i,(ld,stv) in enumerate(insts):
                stv(k, c.add(A[(k,i)], T[(k,i)]))
                stv(L-k, c.sub(A[(k,i)], T[(k,i)]))

# ----------------------------------------------------------------------------
# verification helper: run a body with basis inputs, compare matrix to DFT(L)
def verify_body(L, gen_fn, label):
    em = Emitter()
    c = Ctx(em)
    mat = np.zeros((L, L), dtype=complex)
    ins = {}
    def load(j):
        v = np.zeros(L, dtype=complex); v[j] = 1.0
        name = em.fresh("x")
        em.w(f"V8 {name} = IN[{j}];")
        return Val(name, v)
    def store(k, val):
        mat[k, :] += val.v
    gen_fn(c, load, store)
    W = np.exp(-2j*np.pi*np.outer(np.arange(L), np.arange(L))/L)
    err = np.abs(mat - W).max()
    assert err < 1e-13, f"{label}: verification failed, err={err}"
    return err


# ----------------------------------------------------------------------------
# map emission: z/(1+|z|) elementwise on interleaved-complex zmm vectors.
# paired mode: two vectors -> one Newton chain over 8 packed magnitudes.
def emit_map_pair(em, z1, z2):
    a1 = em.fresh("mq"); a2 = em.fresh("mq")
    em.w(f"V8 {a1} = MUL({z1},{z1});")
    em.w(f"V8 {a2} = MUL({z2},{z2});")
    s = em.fresh("ms")
    em.w(f"V8 {s} = ADD(_mm512_permutex2var_pd({a1}, IDX_EVEN, {a2}), _mm512_permutex2var_pd({a1}, IDX_ODD, {a2}));")
    rc = emit_nr(em, s)
    r1 = em.fresh("mr"); r2 = em.fresh("mr")
    em.w(f"V8 {r1} = _mm512_permutexvar_pd(IDX_LO, {rc});")
    em.w(f"V8 {r2} = _mm512_permutexvar_pd(IDX_HI, {rc});")
    o1 = em.fresh("mo"); o2 = em.fresh("mo")
    em.w(f"V8 {o1} = MUL({z1},{r1});")
    em.w(f"V8 {o2} = MUL({z2},{r2});")
    return o1, o2

def emit_map_single(em, z1):
    a1 = em.fresh("mq")
    em.w(f"V8 {a1} = MUL({z1},{z1});")
    s = em.fresh("ms")
    em.w(f"V8 {s} = ADD({a1}, SWAP({a1}));")
    rc = emit_nr(em, s)
    o1 = em.fresh("mo")
    em.w(f"V8 {o1} = MUL({z1},{rc});")
    return o1

def emit_nr(em, s):
    """given vector s of squared magnitudes, return 1/(1+sqrt(s)) to ~0.5ulp"""
    em.w(f"{s} = MAX({s}, EPS_TINY);")
    a = em.fresh("na")
    em.w(f"V8 {a} = _mm512_rsqrt14_pd({s});")
    em.w(f"{{ V8 e_ = MUL({s},{a}); V8 h_ = MUL({a},CHALF); V8 r_ = FNMA(e_,h_,C15); {a} = MUL({a},r_); }}")
    y = em.fresh("ny")
    em.w(f"V8 {y} = MUL({s},{a});")
    em.w(f"{{ V8 d_ = FNMA({y},{y},{s}); {y} = FMA(MUL(CHALF,{a}), d_, {y}); }}")
    den = em.fresh("nd")
    em.w(f"V8 {den} = ADD(ONE,{y});")
    rc = em.fresh("nr")
    em.w(f"V8 {rc} = _mm512_rcp14_pd({den});")
    em.w(f"{{ V8 w_ = FNMA({den},{rc},C2); {rc} = MUL({rc},w_); }}")
    em.w(f"{{ V8 w_ = FNMA({den},{rc},ONE); {rc} = FMA({rc},w_,{rc}); }}")
    return rc

# ----------------------------------------------------------------------------
# variant emitters
def chunk_list(L):
    """list of (offset_in_doubles, mask_or_None) 4-complex chunks covering batch L"""
    out = []
    n4, rem = divmod(L, 4)
    for t in range(n4):
        out.append((8*t, None))
    if rem:
        out.append((8*n4, (1 << (2*rem)) - 1))
    return out

def mk_ls(em, base, stride2, off, mask, pfx):
    """make load/store callbacks for rows at base + r*stride2 + off (doubles)"""
    def load(r):
        name = em.fresh("x")
        if mask is None:
            em.w(f"V8 {name} = LOADU({base} + {r*stride2 + off if isinstance(stride2,int) else ''}{'' if isinstance(stride2,int) else ''});")
        return Val(name, basis(r))
    return load

def basis_vec(L, j):
    v = np.zeros(L, dtype=complex); v[j] = 1.0
    return v

class RowIO:
    """load/store helpers for one chunk instance"""
    def __init__(self, em, L, baseexpr, stride2, off, mask, storemode="plain",
                 dstexpr=None, cexpr=None):
        self.em=em; self.L=L; self.base=baseexpr; self.s2=stride2; self.off=off
        self.mask=mask; self.storemode=storemode; self.dst=dstexpr; self.c=cexpr; self.pending=[]
    def addr(self, b, r):
        return f"{b} + {r*self.s2 + self.off}"
    def load(self, r):
        name = self.em.fresh("x")
        if self.mask is None:
            self.em.w(f"V8 {name} = LOADU({self.addr(self.base,r)});")
        else:
            self.em.w(f"V8 {name} = MLOADZ(0x{self.mask:02X}, {self.addr(self.base,r)});")
        return Val(name, basis_vec(self.L, r))
    def store(self, r, val):
        em = self.em
        if self.storemode == "plain":
            if self.mask is None:
                em.w(f"STOREU({self.addr(self.base,r)}, {val.e});")
            else:
                em.w(f"MSTORE({self.addr(self.base,r)}, 0x{self.mask:02X}, {val.e});")
        elif self.storemode == "addc":  # store val + c to dst
            cn = em.fresh("cv")
            zn = em.fresh("z")
            if self.mask is None:
                em.w(f"V8 {cn} = LOADU({self.addr(self.c,r)});")
                em.w(f"V8 {zn} = ADD({val.e},{cn});")
                em.w(f"STOREU({self.addr(self.dst,r)}, {zn});")
            else:
                em.w(f"V8 {cn} = MLOADZ(0x{self.mask:02X}, {self.addr(self.c,r)});")
                em.w(f"V8 {zn} = ADD({val.e},{cn});")
                em.w(f"MSTORE({self.addr(self.dst,r)}, 0x{self.mask:02X}, {zn});")
        else:  # mapinline
            cn = em.fresh("cv")
            zn = em.fresh("z")
            if self.mask is None:
                em.w(f"V8 {cn} = LOADU({self.addr(self.c,r)});")
            else:
                em.w(f"V8 {cn} = MLOADZ(0x{self.mask:02X}, {self.addr(self.c,r)});")
            em.w(f"V8 {zn} = ADD({val.e},{cn});")
            self.pending.append((r, zn))
            if len(self.pending) == 2:
                (r1,z1),(r2,z2) = self.pending
                o1, o2 = emit_map_pair(em, z1, z2)
                self._st(r1, o1); self._st(r2, o2)
                self.pending = []
    def finish(self):
        if self.storemode == "mapinline" and self.pending:
            (r1,z1), = self.pending
            o1 = emit_map_single(self.em, z1)
            self._st(r1, o1)
            self.pending = []
    def _st(self, r, name):
        if self.mask is None:
            self.em.w(f"STOREU({self.addr(self.dst,r)}, {name});")
        else:
            self.em.w(f"MSTORE({self.addr(self.dst,r)}, 0x{self.mask:02X}, {name});")

def emit_map_sweep(em, L, dstexpr, s2, off, mask):
    """map pass over rows r=0..L-1 at dst + r*s2 + off (after +c already stored)"""
    def onepair(p0, p1):
        if mask is None:
            em.w(f"V8 z1_ = LOADU({p0}); V8 z2_ = LOADU({p1});")
        else:
            em.w(f"V8 z1_ = MLOADZ(0x{mask:02X}, {p0}); V8 z2_ = MLOADZ(0x{mask:02X}, {p1});")
        o1, o2 = emit_map_pair(em, "z1_", "z2_")
        if mask is None:
            em.w(f"STOREU({p0}, {o1}); STOREU({p1}, {o2});")
        else:
            em.w(f"MSTORE({p0}, 0x{mask:02X}, {o1}); MSTORE({p1}, 0x{mask:02X}, {o2});")
    nquad = L // 4
    pos = 0
    if nquad:
        em.w(f"for (int r_=0; r_<{nquad}; r_++) {{")
        em.ind += 1
        em.w(f"double* mb_ = {dstexpr} + (int64_t)r_*{4*s2} + {off};")
        em.w("{")
        em.ind += 1
        onepair("mb_", f"mb_ + {s2}")
        em.ind -= 1
        em.w("}")
        em.w("{")
        em.ind += 1
        onepair(f"mb_ + {2*s2}", f"mb_ + {3*s2}")
        em.ind -= 1
        em.w("}")
        em.ind -= 1
        em.w("}")
        pos = 4*nquad
    if L - pos >= 2:
        em.w("{")
        em.ind += 1
        onepair(f"{dstexpr} + {pos*s2 + off}", f"{dstexpr} + {(pos+1)*s2 + off}")
        em.ind -= 1
        em.w("}")
        pos += 2
    if L - pos == 1:
        em.w("{")
        em.ind += 1
        p0 = f"{dstexpr} + {pos*s2 + off}"
        if mask is None:
            em.w(f"V8 z1_ = LOADU({p0});")
        else:
            em.w(f"V8 z1_ = MLOADZ(0x{mask:02X}, {p0});")
        o1 = emit_map_single(em, "z1_")
        if mask is None:
            em.w(f"STOREU({p0}, {o1});")
        else:
            em.w(f"MSTORE({p0}, 0x{mask:02X}, {o1});")
        em.ind -= 1
        em.w("}")

PRIMES = (13,17,23)

def emit_body_for(c, L, ios):
    """dispatch: primes use prime_body (multi-instance); composites single."""
    if L in PRIMES:
        insts = [(io.load, io.store) for io in ios]
        prime_body(c, L, insts)
    else:
        assert len(ios) == 1
        composite_body(c, L, ios[0].load, ios[0].store)

def emit_kernel_yx(L, kind, unpadded=False):
    em = Emitter()
    if kind=='y':
        s2 = 2*L if unpadded else ROWST2(L)
    else:
        s2 = SLICEST2(L)
    batchN = L if kind=='y' else L*L
    suf = "u" if unpadded else ""
    if kind == 'y':
        em.w(f"static void kY_{L}{suf}(double* p){{")
    else:
        em.w(f"static void kX_{L}(const double* ps, double* pd, const double* restrict pc){{")
    em.ind += 1
    n4, rem = divmod(batchN, 4)
    tail = [] if rem == 0 else [(8*n4, (1 << (2*rem)) - 1)]
    nfull = n4

    def mkios(basegroups):
        ios = []
        for (bases, off, mask) in basegroups:
            if kind == 'y':
                ios.append(RowIO(em, L, bases[0], s2, off, mask))
            else:
                mode = "mapinline" if MAPFUSE.get(L) else "addc"
                ios.append(RowIO(em, L, bases[0], s2, off, mask, storemode=mode,
                                 dstexpr=bases[1], cexpr=bases[2]))
        return ios

    def emit_bodies(basegroups):
        em.w("{")
        em.ind += 1
        em.w("BARRIER;")
        ios = mkios(basegroups)
        c = Ctx(em)
        emit_body_for(c, L, ios)
        if kind == 'x':
            for io in ios:
                io.finish()
            if not MAPFUSE.get(L):
                for (bases, off, mask) in basegroups:
                    emit_map_sweep(em, L, bases[1], s2, off, mask)
        em.ind -= 1
        em.w("}")

    basesP = ("p",) if kind=='y' else ("ps","pd","pc")
    if L in PRIMES and L in PRIMECU2:
        npl = nfull // 2
        if npl:
            em.w(f"for (int64_t t_=0; t_<{npl}; t_++) {{")
            em.ind += 1
            if kind=='y':
                em.w(f"double* q = p + 16*t_;")
                bq = ("q",)
            else:
                em.w(f"const double* qs = ps + 16*t_;")
                em.w(f"double* qd = pd + 16*t_;")
                em.w(f"const double* restrict qc = pc + 16*t_;")
                bq = ("qs","qd","qc")
            emit_bodies([(bq, 0, None), (bq, 8, None)])
            em.ind -= 1
            em.w("}")
        rem_chunks = []
        if nfull % 2:
            rem_chunks.append((8*(nfull-1), None))
        rem_chunks += tail
        if len(rem_chunks) == 2:
            emit_bodies([(basesP, rem_chunks[0][0], rem_chunks[0][1]),
                         (basesP, rem_chunks[1][0], rem_chunks[1][1])])
        elif len(rem_chunks) == 1:
            emit_bodies([(basesP, rem_chunks[0][0], rem_chunks[0][1])])
    else:
        if nfull > 1:
            em.w(f"for (int64_t t_=0; t_<{nfull}; t_++) {{")
            em.ind += 1
            if kind=='y':
                em.w(f"double* q = p + 8*t_;")
                bq = ("q",)
            else:
                if ROWPAD[L]:
                    nsub = L//4
                    assert nsub & (nsub-1) == 0 and L*L % 4 == 0
                    em.w(f"const int64_t off_ = (t_ >> {nsub.bit_length()-1})*{ROWST2(L)} + 8*(t_ & {nsub-1});")
                else:
                    em.w(f"const int64_t off_ = 8*t_;")
                em.w(f"const double* qs = ps + off_;")
                em.w(f"double* qd = pd + off_;")
                em.w(f"const double* restrict qc = pc + off_;")
                bq = ("qs","qd","qc")
            emit_bodies([(bq, 0, None)])
            em.ind -= 1
            em.w("}")
        elif nfull == 1:
            emit_bodies([(basesP, 0, None)])
        for off, mk in tail:
            emit_bodies([(basesP, off, mk)])
    em.ind -= 1
    em.w("}")
    return em.code()

def emit_kernel_z(L, unpadded=False, copyin=False):
    """kZ_L(double* p): z-FFT of nlines lines (length L each).
       copyin: read from unpadded src, write to padded dst."""
    nlines = L*L if L in ZVOL else L
    em = Emitter()
    if copyin:
        em.w(f"static void kZ_{L}X(const double* ps, double* pd){{")
        em.ind += 1
        ldl = 2*L             # src line stride
        ldlo = ROWST2(L)      # dst line stride
    else:
        suf = "u" if unpadded else ""
        em.w(f"static void kZ_{L}{suf}(double* p){{")
        em.ind += 1
        ldl = 2*L if unpadded else ROWST2(L)   # doubles between lines
        ldlo = ldl
    nblk, erem = divmod(L, 4)
    emask = (1 << (2*erem)) - 1 if erem else None
    ngrp, lrem = divmod(nlines, 4)
    em.w(f"V8 T[{L}];")
    def transpose_in(qexpr, nlines):
        for jb in range(nblk):
            rs = []
            for ln in range(4):
                name = em.fresh("r")
                if ln < nlines:
                    em.w(f"V8 {name} = LOADU({qexpr} + {ln*ldl + jb*8});")
                else:
                    em.w(f"V8 {name} = _mm512_setzero_pd();")
                rs.append(name)
            outs = [em.fresh("c") for _ in range(4)]
            em.w(f"V8 {outs[0]},{outs[1]},{outs[2]},{outs[3]};")
            em.w(f"TRAN4({rs[0]},{rs[1]},{rs[2]},{rs[3]},{outs[0]},{outs[1]},{outs[2]},{outs[3]});")
            for t in range(4):
                em.w(f"T[{4*jb+t}] = {outs[t]};")
        if erem:
            rs = []
            for ln in range(4):
                name = em.fresh("r")
                if ln < nlines:
                    em.w(f"V8 {name} = MLOADZ(0x{emask:02X}, {qexpr} + {ln*ldl + nblk*8});")
                else:
                    em.w(f"V8 {name} = _mm512_setzero_pd();")
                rs.append(name)
            outs = [em.fresh("c") for _ in range(4)]
            em.w(f"V8 {outs[0]},{outs[1]},{outs[2]},{outs[3]};")
            em.w(f"TRAN4({rs[0]},{rs[1]},{rs[2]},{rs[3]},{outs[0]},{outs[1]},{outs[2]},{outs[3]});")
            for t in range(erem):
                em.w(f"T[{4*nblk+t}] = {outs[t]};")
    def transpose_out(qexpr, nlines):
        for jb in range(nblk):
            ins = [f"T[{4*jb+t}]" for t in range(4)]
            outs = [em.fresh("c") for _ in range(4)]
            em.w(f"V8 {outs[0]},{outs[1]},{outs[2]},{outs[3]};")
            em.w(f"TRAN4({ins[0]},{ins[1]},{ins[2]},{ins[3]},{outs[0]},{outs[1]},{outs[2]},{outs[3]});")
            for ln in range(min(4, nlines)):
                em.w(f"STOREU({qexpr} + {ln*ldlo + jb*8}, {outs[ln]});")
        if erem:
            ins = []
            for t in range(4):
                if t < erem:
                    ins.append(f"T[{4*nblk+t}]")
                else:
                    z = em.fresh("zz"); em.w(f"V8 {z} = _mm512_setzero_pd();")
                    ins.append(z)
            outs = [em.fresh("c") for _ in range(4)]
            em.w(f"V8 {outs[0]},{outs[1]},{outs[2]},{outs[3]};")
            em.w(f"TRAN4({ins[0]},{ins[1]},{ins[2]},{ins[3]},{outs[0]},{outs[1]},{outs[2]},{outs[3]});")
            for ln in range(min(4, nlines)):
                em.w(f"MSTORE({qexpr} + {ln*ldl + nblk*8}, 0x{emask:02X}, {outs[ln]});")
    def emit_group(qexpr, nlines, qout=None, nobar=False):
        if not nobar:
            em.w("BARRIER;")
        em.w("{")
        em.ind += 1
        transpose_in(qexpr, nlines)
        class TIO:
            def load(self, r):
                name = em.fresh("x")
                em.w(f"V8 {name} = T[{r}];")
                return Val(name, basis_vec(L, r))
            def store(self, r, val):
                em.w(f"T[{r}] = {val.e};")
        em.w("{")
        c = Ctx(em)
        emit_body_for(c, L, [TIO()])
        em.w("}")
        transpose_out(qout if qout is not None else qexpr, nlines)
        em.ind -= 1
        em.w("}")
    zpair = (L in ZPAIR) and (ngrp % 2 == 0) and not copyin
    if copyin:
        em.w(f"for (int g_=0; g_<{ngrp}; g_++) {{")
        em.ind += 1
        em.w(f"const double* q = ps + g_*{4*ldl};")
        em.w(f"double* qo = pd + g_*{4*ldlo};")
        emit_group("q", 4, qout="qo")
        em.ind -= 1
        em.w("}")
        assert lrem == 0
        em.ind -= 1
        em.w("}")
        return em.code()
    if zpair:
        em.w(f"for (int g_=0; g_<{ngrp//2}; g_++) {{")
        em.ind += 1
        em.w(f"double* q = p + g_*{8*ldl};")
        em.w(f"double* q2 = q + {4*ldl};")
        emit_group("q", 4)
        emit_group("q2", 4, nobar=True)
        em.ind -= 1
        em.w("}")
    elif ngrp > 1:
        em.w(f"for (int g_=0; g_<{ngrp}; g_++) {{")
        em.ind += 1
        em.w(f"double* q = p + g_*{4*ldl};")
        emit_group("q", 4)
        em.ind -= 1
        em.w("}")
    elif ngrp == 1:
        emit_group("p", 4)
    if lrem:
        em.w(f"{{ double* q = p + {ngrp*4*ldl};")
        em.ind += 1
        emit_group("q", lrem)
        em.ind -= 1
        em.w("}")
    em.ind -= 1
    em.w("}")
    return em.code()



def apply_wN(c, v, t, N):
    """multiply Val v by w_N^t"""
    t = t % N
    if t == 0:
        return v
    if N == 64:
        return apply_w64(c, v, t)
    if 4*t % N == 0:
        q = (4*t)//N    # 1,2,3 -> -i, -1, +i
        if q == 1: return c.mulmi(v)
        if q == 2: return c.neg(v)
        if q == 3: return c.muli(v)
    wre, wim, wv = TAB.w(f"W{N}", t, N)
    return c.cmul(v, wre, wim, wv)

XSPLIT = {36: (6,6), 45: (9,5), 64: (8,8)}
DFT_OF = {5: dft5, 6: None, 8: dft8, 9: dft9}

def dft6(c, xs):
    ys = [None]*6
    def ld(j): return xs[j]
    def st(k, v): ys[k] = v
    pfa_body(c, 6, 2, 3, ld, st)
    return ys
DFT_OF[6] = dft6

def emit_xsplit(L):
    """stage kernels for L = R*S: xs1_{L}_{g} (in-place, S slices stride R),
       xs2_{L}[_u] (R consecutive slices in -> R slices stride S out-of-place, +c+map)."""
    R, S = XSPLIT[L]
    out = []
    RS2 = ROWST2(L)
    RS2u = 2*L
    n4, rem = divmod(L*L, 4)
    def chunk_loop_open(em, strides, unp=False):
        em.w(f"for (int64_t t_=0; t_<{n4}; t_++) {{")
        em.ind += 1
        if ROWPAD[L] and not unp:
            nsub = L//4
            em.w(f"const int64_t y_ = t_ >> {nsub.bit_length()-1}; const int64_t u_ = t_ & {nsub-1};")
            for nm, rs in strides:
                if rs == 2*L:
                    em.w(f"const int64_t {nm} = 8*t_;")
                else:
                    em.w(f"const int64_t {nm} = y_*{rs} + 8*u_;")
        else:
            for nm, rs in strides:
                em.w(f"const int64_t {nm} = 8*t_;")
        em.w("BARRIER;")
    def chunk_loop_close(em):
        em.ind -= 1
        em.w("}")
    # ---- stage 1 variants ----
    variants1 = [("", RS2)]
    for suf1, rs1 in variants1:
      for g in range(R):
        em = Emitter()
        em.w(f"static void xs1_{L}_{g}{suf1}(double* base, const int64_t step2){{")
        em.ind += 1
        def body(mask):
            cc = Ctx(em)
            mat = np.zeros((S,S), dtype=complex)
            xs = []
            for j in range(S):
                n = em.fresh("x")
                if mask is None:
                    em.w(f"V8 {n} = LOADU(base + {j}*step2 + off_);")
                else:
                    em.w(f"V8 {n} = MLOADZ(0x{mask:02X}, base + {j}*step2 + off_);")
                v = np.zeros(S, dtype=complex); v[j] = 1
                xs.append(Val(n, v))
            ys = DFT_OF[S](cc, xs)
            for k in range(S):
                v = apply_wN(cc, ys[k], g*k, L)
                if mask is None:
                    em.w(f"STOREU(base + {k}*step2 + off_, {v.e});")
                else:
                    em.w(f"MSTORE(base + {k}*step2 + off_, 0x{mask:02X}, {v.e});")
                mat[k,:] += v.v
            W = np.exp(-2j*np.pi*np.outer(np.arange(S), np.arange(S))/S)
            T = np.exp(-2j*np.pi*g*np.arange(S)/L)[:,None] * W
            assert np.abs(mat - T).max() < 1e-13, (L, g)
        chunk_loop_open(em, [("off_", rs1)], unp=(suf1=="u"))
        body(None)
        chunk_loop_close(em)
        if rem:
            em.w("{")
            em.ind += 1
            em.w(f"const int64_t off_ = {8*n4};")
            body((1 << (2*rem)) - 1)
            em.ind -= 1
            em.w("}")
        em.ind -= 1
        em.w("}")
        out.append(em.code())
    # ---- stage 2 ----
    ntok = (L*L) % 4 == 0
    if not ROWPAD[L]:
        variants = [("", RS2, RS2, RS2, None), ("_d", RS2, RS2, RS2, RS2)]
        if ntok:
            variants += [("_s", RS2, RS2, RS2, None), ("_ds", RS2, RS2, RS2, RS2)]
    else:
        variants = [("", RS2, RS2, RS2u, None), ("_u", RS2, RS2u, RS2u, None), ("_d", RS2, RS2, RS2u, RS2u),
                    ("_us", RS2, RS2u, RS2u, None), ("_ds", RS2, RS2, RS2u, RS2u)]
    for suf, rsin, rsout, rsc, rsout2 in variants:
        ntout = suf in ("_s","_us")
        ntout2 = suf == "_ds"
        em = Emitter()
        if rsout2 is None:
            em.w(f"static void xs2_{L}{suf}(const double* gin, double* gout, const double* restrict gc, const int64_t stin, const int64_t stout, const int64_t stc){{")
        else:
            em.w(f"static void xs2_{L}{suf}(const double* gin, double* gout, double* restrict gout2, const double* restrict gc, const int64_t stin, const int64_t stout, const int64_t stout2, const int64_t stc){{")
        em.ind += 1
        def body2(mask):
            cc = Ctx(em)
            mat = np.zeros((R,R), dtype=complex)
            xs = []
            for j in range(R):
                n = em.fresh("x")
                if mask is None:
                    em.w(f"V8 {n} = LOADU(gin + {j}*stin + offi_);")
                else:
                    em.w(f"V8 {n} = MLOADZ(0x{mask:02X}, gin + {j}*stin + offi_);")
                v = np.zeros(R, dtype=complex); v[j] = 1
                xs.append(Val(n, v))
            ys = DFT_OF[R](cc, xs)
            for k in range(R):
                mat[k,:] += ys[k].v
            W = np.exp(-2j*np.pi*np.outer(np.arange(R), np.arange(R))/R)
            assert np.abs(mat - W).max() < 1e-13, (L,)
            zs = []
            for k in range(R):
                cn = em.fresh("cv")
                if mask is None:
                    em.w(f"V8 {cn} = LOADU(gc + {k}*stc + offc_);")
                else:
                    em.w(f"V8 {cn} = MLOADZ(0x{mask:02X}, gc + {k}*stc + offc_);")
                zn = em.fresh("z")
                em.w(f"V8 {zn} = ADD({ys[k].e},{cn});")
                zs.append(zn)
            k = 0
            def dost(kk, oo):
                if mask is None:
                    st1 = "STREAMU" if ntout else "STOREU"
                    em.w(f"{st1}(gout + {kk}*stout + offo_, {oo});")
                    if rsout2 is not None:
                        st2 = "STREAMU" if ntout2 else "STOREU"
                        em.w(f"{st2}(gout2 + {kk}*stout2 + offo2_, {oo});")
                else:
                    em.w(f"MSTORE(gout + {kk}*stout + offo_, 0x{mask:02X}, {oo});")
                    if rsout2 is not None:
                        em.w(f"MSTORE(gout2 + {kk}*stout2 + offo2_, 0x{mask:02X}, {oo});")
            while k + 2 <= R:
                o1, o2 = emit_map_pair(em, zs[k], zs[k+1])
                dost(k, o1); dost(k+1, o2)
                k += 2
            if k < R:
                o1 = emit_map_single(em, zs[k])
                dost(k, o1)
        offs = [("offi_", rsin), ("offo_", rsout), ("offc_", rsc)]
        if rsout2 is not None:
            offs.append(("offo2_", rsout2))
        chunk_loop_open(em, offs)
        body2(None)
        chunk_loop_close(em)
        if rem:
            em.w("{")
            em.ind += 1
            decls = " ".join(f"const int64_t {nm} = {8*n4};" for nm,_ in offs)
            em.w(decls)
            body2((1 << (2*rem)) - 1)
            em.ind -= 1
            em.w("}")
        em.ind -= 1
        em.w("}")
        out.append(em.code())
    return "\n\n".join(out)

def emit_driver_xsplit(L):
    R, S = XSPLIT[L]
    em = Emitter()
    SS2 = SLICEST2(L)
    SS2u = 2*L*L
    RS2 = ROWST2(L)
    V = 2*L*L*L
    padded = bool(PAD[L] or ROWPAD[L])
    sym = (R == S)
    ntok = (L*L) % 4 == 0
    em.w(f"static void run_{L}(int64_t B, int64_t m, double* restrict x0, const double* restrict c, double* restrict out1, double* restrict outm){{")
    em.ind += 1
    em.w(f"const int64_t VOL = {V};")
    if padded:
        em.w(f"double* restrict w1 = WORK;")
    elif not sym:
        em.w(f"double* restrict wbuf = WORK;")
    em.w(f"for (int64_t b=0;b<B;b++){{")
    em.ind += 1
    em.w(f"const double* restrict cb = c + b*VOL;")
    em.w(f"double* restrict o1 = out1 + b*VOL;")
    em.w(f"double* restrict om = outm + b*VOL;")
    em.w(f"int64_t mm = (m==0)?1:m;")
    em.w(f"if (m==0) memcpy(om, x0 + b*VOL, VOL*8);")
    if sym:
        # ---------------- symmetric in-place parity scheme ----------------
        em.w(f"double* restrict w = {'w1' if padded else '(x0 + b*VOL)'};")
        em.w(f"int par = 0;")
        em.w(f"for (int64_t it=1; it<=mm; it++){{")
        em.ind += 1
        # pass 1
        if padded:
            em.w(f"if (it==1) {{")
            em.ind += 1
            em.w(f"const double* xv = x0 + b*VOL;")
            em.w(f"for (int g=0; g<{R}; g++) {{")
            em.ind += 1
            em.w(f"for (int j1=0;j1<{S};j1++) {{ int s = g + {R}*j1; kZ_{L}X(xv + s*{SS2u}, w + s*{SS2}); kY_{L}(w + s*{SS2}); }}")
            em.w(f"switch(g) {{")
            for g in range(R):
                em.w(f"    case {g}: xs1_{L}_{g}(w + {g}*{SS2}, {R*SS2}); break;")
            em.w(f"}}")
            em.ind -= 1
            em.w(f"}}")
            em.ind -= 1
            em.w(f"}} else {{")
            em.ind += 1
            em.w(f"for (int g=0; g<{R}; g++) {{")
            em.ind += 1
            em.w(f"int64_t gb = (par==0) ? (int64_t)g*{SS2} : (int64_t)g*{R*SS2};")
            em.w(f"int64_t gs = (par==0) ? {R*SS2} : {SS2};")
            em.w(f"for (int j1=0;j1<{S};j1++) {{ double* sl = w + gb + j1*gs; kZ_{L}(sl); kY_{L}(sl); }}")
            em.w(f"switch(g) {{")
            for g in range(R):
                em.w(f"    case {g}: xs1_{L}_{g}(w + gb, gs); break;")
            em.w(f"}}")
            em.ind -= 1
            em.w(f"}}")
            em.ind -= 1
            em.w(f"}}")
        else:
            em.w(f"for (int g=0; g<{R}; g++) {{")
            em.ind += 1
            em.w(f"int64_t gb = (par==0) ? (int64_t)g*{SS2} : (int64_t)g*{R*SS2};")
            em.w(f"int64_t gs = (par==0) ? {R*SS2} : {SS2};")
            em.w(f"for (int j1=0;j1<{S};j1++) {{ double* sl = w + gb + j1*gs; kZ_{L}(sl); kY_{L}(sl); }}")
            em.w(f"switch(g) {{")
            for g in range(R):
                em.w(f"    case {g}: xs1_{L}_{g}(w + gb, gs); break;")
            em.w(f"}}")
            em.ind -= 1
            em.w(f"}}")
        # pass 2: in-place except outputs
        em.w(f"int lastdirect = (it==mm && m!=0);")
        em.w(f"for (int h=0; h<{S}; h++) {{")
        em.ind += 1
        em.w(f"int64_t hb = (par==0) ? (int64_t){R}*h*{SS2} : (int64_t)h*{SS2};")
        em.w(f"int64_t hs = (par==0) ? {SS2} : {R*SS2};")
        em.w(f"const double* restrict gc = cb + h*{SS2u};")
        em.w(f"if (lastdirect) {{")
        if padded:
            if ntok:
                em.w(f"    if (it!=1 && (((uintptr_t)om)&63)==0) {{ xs2_{L}_us(w + hb, om + h*{SS2u}, gc, hs, {R*SS2u}, {R*SS2u}); }}")
                em.w(f"    else xs2_{L}_u(w + hb, om + h*{SS2u}, gc, hs, {R*SS2u}, {R*SS2u});")
            else:
                em.w(f"    xs2_{L}_u(w + hb, om + h*{SS2u}, gc, hs, {R*SS2u}, {R*SS2u});")
        else:
            if ntok:
                em.w(f"    if (it!=1 && (((uintptr_t)om)&63)==0) {{ xs2_{L}_s(w + hb, om + h*{SS2u}, gc, hs, {R*SS2u}, {R*SS2u}); }}")
                em.w(f"    else xs2_{L}(w + hb, om + h*{SS2u}, gc, hs, {R*SS2u}, {R*SS2u});")
            else:
                em.w(f"    xs2_{L}(w + hb, om + h*{SS2u}, gc, hs, {R*SS2u}, {R*SS2u});")
        em.w(f"}} else if (it==1) {{")
        dsuf = "_ds" if ntok else "_d"
        em.w(f"    if ((((uintptr_t)o1)&63)==0) {{ xs2_{L}{dsuf}(w + hb, w + hb, o1 + h*{SS2u}, gc, hs, hs, {R*SS2u}, {R*SS2u}); }}")
        em.w(f"    else xs2_{L}_d(w + hb, w + hb, o1 + h*{SS2u}, gc, hs, hs, {R*SS2u}, {R*SS2u});")
        em.w(f"}} else {{")
        em.w(f"    xs2_{L}(w + hb, w + hb, gc, hs, hs, {R*SS2u});")
        em.w(f"}}")
        em.ind -= 1
        em.w(f"}}")
        em.w(f"if (lastdirect && it==1) {{ _mm_sfence(); ntcopy(o1, om, VOL); }}")
        em.w(f"if (it==1 && !lastdirect) _mm_sfence();")
        em.w(f"if (lastdirect && it!=1) _mm_sfence();")
        em.w(f"par ^= 1;")
        em.ind -= 1
        em.w(f"}}")
    else:
        # ---------------- asymmetric: rotation scheme (45) ----------------
        em.w(f"double* cur = x0 + b*VOL;")
        em.w(f"for (int64_t it=1; it<=mm; it++){{")
        em.ind += 1
        em.w(f"for (int g=0; g<{R}; g++) {{")
        em.ind += 1
        em.w(f"for (int j1=0;j1<{S};j1++) {{ double* sl = cur + (g + {R}*j1)*{SS2}; kZ_{L}(sl); kY_{L}(sl); }}")
        em.w(f"switch(g) {{")
        for g in range(R):
            em.w(f"    case {g}: xs1_{L}_{g}(cur + {g}*{SS2}, {R*SS2}); break;")
        em.w(f"}}")
        em.ind -= 1
        em.w(f"}}")
        em.w(f"double* nxt = (it==mm && m!=0) ? om : ((cur==wbuf)? (x0 + b*VOL) : wbuf);")
        em.w(f"if (it==1) {{")
        em.w(f"    for (int h=0;h<{S};h++) xs2_{L}_d(cur + {R}*h*{SS2}, nxt + h*{SS2}, o1 + h*{SS2}, cb + h*{SS2}, {SS2}, {S*SS2}, {S*SS2}, {S*SS2});")
        em.w(f"}} else {{")
        em.w(f"    for (int h=0;h<{S};h++) xs2_{L}(cur + {R}*h*{SS2}, nxt + h*{SS2}, cb + h*{SS2}, {SS2}, {S*SS2}, {S*SS2});")
        em.w(f"}}")
        em.w(f"cur = nxt;")
        em.ind -= 1
        em.w(f"}}")
    em.ind -= 1
    em.w(f"}}")
    em.ind -= 1
    em.w(f"}}")
    return em.code()


AOS4 = {6, 8, 13, 17, 23}

def emit_kernel_aos(L):
    """AoSoA-4 kernels: layout w4[pos*4 + vol] complex, pos = ((x*L+y)*L+z).
       kZ4(p): L^2 sets (x,y), rows z at stride 4c; kY4(p): sets (x,z); kX4(p, c4): sets (y,z), rows x, fused +c+map."""
    out = []
    P8 = 8  # doubles per position (4 complex)
    # --- kZ4 ---
    em = Emitter()
    em.w(f"static void kZ4_{L}(double* p){{")
    em.ind += 1
    em.w(f"for (int64_t s_=0; s_<{L*L}; s_++) {{")
    em.ind += 1
    em.w(f"double* base = p + s_*{8*L};")
    em.w("BARRIER;")
    em.w("{")
    em.ind += 1
    io = RowIO(em, L, "base", 8, 0, None)
    c = Ctx(em)
    emit_body_for(c, L, [io])
    em.ind -= 1
    em.w("}")
    em.ind -= 1
    em.w("}")
    em.ind -= 1
    em.w("}")
    out.append(em.code())
    # --- kY4 ---
    em = Emitter()
    em.w(f"static void kY4_{L}(double* p){{")
    em.ind += 1
    pair = (L in PRIMES) and (L in AOSPAIR)
    em.w(f"for (int64_t x_=0; x_<{L}; x_++) {{")
    em.ind += 1
    if pair:
        em.w(f"for (int64_t z_=0; z_<{L-1}; z_+=2) {{")
    else:
        em.w(f"for (int64_t z_=0; z_<{L}; z_++) {{")
    em.ind += 1
    em.w(f"double* base = p + x_*{8*L*L} + z_*8;")
    em.w("BARRIER;")
    em.w("{")
    em.ind += 1
    if pair:
        io0 = RowIO(em, L, "base", 8*L, 0, None)
        io1 = RowIO(em, L, "base", 8*L, 8, None)
        c = Ctx(em)
        emit_body_for(c, L, [io0, io1])
    else:
        io = RowIO(em, L, "base", 8*L, 0, None)
        c = Ctx(em)
        emit_body_for(c, L, [io])
    em.ind -= 1
    em.w("}")
    em.ind -= 1
    em.w("}")
    if pair:
        # odd leftover z
        em.w(f"{{ double* base = p + x_*{8*L*L} + {8*(L-1)};")
        em.ind += 1
        em.w("BARRIER;")
        em.w("{")
        em.ind += 1
        io = RowIO(em, L, "base", 8*L, 0, None)
        c = Ctx(em)
        emit_body_for(c, L, [io])
        em.ind -= 1
        em.w("}")
        em.ind -= 1
        em.w("}")
    em.ind -= 1
    em.w("}")
    em.ind -= 1
    em.w("}")
    out.append(em.code())
    # --- kX4 with fused +c+map ---
    em = Emitter()
    em.w(f"static void kX4_{L}(double* p, const double* restrict c4){{")
    em.ind += 1
    em.w(f"for (int64_t s_=0; s_<{L*L}; s_++) {{")
    em.ind += 1
    em.w(f"double* base = p + s_*8;")
    em.w(f"const double* restrict cbase = c4 + s_*8;")
    em.w("BARRIER;")
    em.w("{")
    em.ind += 1
    mode = "addc" if (L in AOSSWEEP) else "mapinline"
    io = RowIO(em, L, "base", 8*L*L, 0, None, storemode=mode,
               dstexpr="base", cexpr="cbase")
    c = Ctx(em)
    emit_body_for(c, L, [io])
    io.finish()
    if L in AOSSWEEP:
        emit_map_sweep(em, L, "base", 8*L*L, 0, None)
    em.ind -= 1
    em.w("}")
    em.ind -= 1
    em.w("}")
    em.ind -= 1
    em.w("}")
    out.append(em.code())
    return "\n\n".join(out)

def emit_driver_aos(L):
    """4-volume AoSoA driver with pack/unpack; falls back to run1_{L} for tail volumes."""
    em = Emitter()
    V = 2*L*L*L           # doubles per volume
    N = L*L*L             # positions
    nb, remp = divmod(N, 4)
    em.w(f"static void pack4_{L}(double* restrict d, const double* s0, const double* s1, const double* s2, const double* s3){{")
    em.ind += 1
    em.w(f"for (int64_t pb=0; pb<{nb}; pb++) {{")
    em.ind += 1
    em.w(f"V8 a0 = LOADU(s0 + pb*8), a1 = LOADU(s1 + pb*8), a2 = LOADU(s2 + pb*8), a3 = LOADU(s3 + pb*8);")
    em.w(f"V8 o0,o1,o2,o3; TRAN4(a0,a1,a2,a3,o0,o1,o2,o3);")
    em.w(f"STOREU(d + pb*32, o0); STOREU(d + pb*32 + 8, o1); STOREU(d + pb*32 + 16, o2); STOREU(d + pb*32 + 24, o3);")
    em.ind -= 1
    em.w(f"}}")
    if remp:
        mk = (1 << (2*remp)) - 1
        em.w(f"{{ V8 a0 = MLOADZ(0x{mk:02X}, s0 + {nb*8}), a1 = MLOADZ(0x{mk:02X}, s1 + {nb*8}), a2 = MLOADZ(0x{mk:02X}, s2 + {nb*8}), a3 = MLOADZ(0x{mk:02X}, s3 + {nb*8});")
        em.w(f"  V8 o0,o1,o2,o3; TRAN4(a0,a1,a2,a3,o0,o1,o2,o3);")
        for t in range(remp):
            em.w(f"  STOREU(d + {nb*32 + 8*t}, o{t});")
        em.w(f"}}")
    em.ind -= 1
    em.w(f"}}")
    em.w(f"static void unpack4_{L}(const double* s, double* d0, double* d1, double* d2, double* d3){{")
    em.ind += 1
    em.w(f"for (int64_t pb=0; pb<{nb}; pb++) {{")
    em.ind += 1
    em.w(f"V8 a0 = LOADU(s + pb*32), a1 = LOADU(s + pb*32 + 8), a2 = LOADU(s + pb*32 + 16), a3 = LOADU(s + pb*32 + 24);")
    em.w(f"V8 o0,o1,o2,o3; TRAN4(a0,a1,a2,a3,o0,o1,o2,o3);")
    em.w(f"STOREU(d0 + pb*8, o0); STOREU(d1 + pb*8, o1); STOREU(d2 + pb*8, o2); STOREU(d3 + pb*8, o3);")
    em.ind -= 1
    em.w(f"}}")
    if remp:
        mk = (1 << (2*remp)) - 1
        em.w(f"{{ V8 a0 = LOADU(s + {nb*32});")
        for t in range(1, remp):
            em.w(f"  V8 a{t} = LOADU(s + {nb*32 + 8*t});")
        for t in range(remp, 4):
            em.w(f"  V8 a{t} = _mm512_setzero_pd();")
        em.w(f"  V8 o0,o1,o2,o3; TRAN4(a0,a1,a2,a3,o0,o1,o2,o3);")
        em.w(f"  MSTORE(d0 + {nb*8}, 0x{mk:02X}, o0); MSTORE(d1 + {nb*8}, 0x{mk:02X}, o1); MSTORE(d2 + {nb*8}, 0x{mk:02X}, o2); MSTORE(d3 + {nb*8}, 0x{mk:02X}, o3);")
        em.w(f"}}")
    em.ind -= 1
    em.w(f"}}")
    return em.code()

def emit_driver(L):
    if L in XSPLIT:
        return emit_driver_xsplit(L)
    em = Emitter()
    fname = f"run1_{L}" if L in AOS4 else f"run_{L}"
    SS2 = SLICEST2(L)
    RS2 = ROWST2(L)
    V = 2*L*L*L
    padded = (PAD[L] or ROWPAD[L])
    em.w(f"static void {fname}(int64_t B, int64_t m, double* restrict x0, const double* restrict c, double* restrict out1, double* restrict outm){{")
    em.ind += 1
    em.w(f"const int64_t VOL = {V};")
    em.w(f"double* w = WORK;")
    if padded:
        em.w(f"double* restrict cw = WORK + {SS2*L};")
    em.w(f"for (int64_t b=0;b<B;b++){{")
    em.ind += 1
    em.w(f"const double* restrict cb = c + b*VOL;")
    em.w(f"double* restrict o1 = out1 + b*VOL;")
    em.w(f"double* restrict om = outm + b*VOL;")
    if padded:
        if ROWPAD[L]:
            em.w(f"for (int s=0;s<{L};s++) for (int y=0;y<{L};y++) {{ memcpy(w + s*{SS2} + y*{RS2}, x0 + b*VOL + (s*{L}+y)*{2*L}, {16*L}); memcpy(cw + s*{SS2} + y*{RS2}, cb + (s*{L}+y)*{2*L}, {16*L}); }}")
        else:
            em.w(f"for (int s=0;s<{L};s++) {{ memcpy(w + s*{SS2}, x0 + b*VOL + s*{2*L*L}, {16*L*L}); memcpy(cw + s*{SS2}, cb + s*{2*L*L}, {16*L*L}); }}")
    else:
        em.w(f"w = x0 + b*VOL;")
        em.w(f"const double* restrict cw = cb;")
    em.w(f"int64_t mm = (m==0)?1:m;")
    em.w(f"if (m==0) memcpy(om, x0 + b*VOL, VOL*8);")
    em.w(f"for (int64_t it=1; it<=mm; it++){{")
    em.ind += 1
    if L in ZVOL:
        em.w(f"kZ_{L}(w);")
        em.w(f"for (int s=0;s<{L};s++) kY_{L}(w + s*{SS2});")
    else:
        em.w(f"for (int s=0;s<{L};s++){{ kZ_{L}(w + s*{SS2}); kY_{L}(w + s*{SS2}); }}")
    if padded:
        em.w(f"kX_{L}(w, w, cw);")
        if ROWPAD[L]:
            em.w(f"if (it==1) for (int s=0;s<{L};s++) for (int y=0;y<{L};y++) ntcopy(o1 + (s*{L}+y)*{2*L}, w + s*{SS2} + y*{RS2}, {2*L});")
            em.w(f"if (it==mm && m!=0) for (int s=0;s<{L};s++) for (int y=0;y<{L};y++) ntcopy(om + (s*{L}+y)*{2*L}, w + s*{SS2} + y*{RS2}, {2*L});")
        else:
            em.w(f"if (it==1) for (int s=0;s<{L};s++) ntcopy(o1 + s*{2*L*L}, w + s*{SS2}, {2*L*L});")
            em.w(f"if (it==mm && m!=0) for (int s=0;s<{L};s++) ntcopy(om + s*{2*L*L}, w + s*{SS2}, {2*L*L});")
    else:
        em.w(f"double* restrict dst_ = (it==mm && m!=0) ? om : w;")
        em.w(f"kX_{L}(w, dst_, cw);")
        em.w(f"if (it==1) ntcopy(o1, dst_, VOL);")
    em.ind -= 1
    em.w(f"}}")
    em.ind -= 1
    em.w(f"}}")
    em.ind -= 1
    em.w(f"}}")
    if L in AOS4:
        V = 2*L*L*L
        em.w(f"static void run_{L}(int64_t B, int64_t m, double* restrict x0, const double* restrict c, double* restrict out1, double* restrict outm){{")
        em.ind += 1
        em.w(f"const int64_t VOL = {V};")
        em.w(f"double* restrict w4 = WORK;")
        em.w(f"double* restrict c4 = WORK + {4*V};")
        em.w(f"int64_t b4 = (B/4)*4;")
        em.w(f"for (int64_t g=0; g<b4; g+=4){{")
        em.ind += 1
        em.w(f"pack4_{L}(w4, x0 + g*VOL, x0 + (g+1)*VOL, x0 + (g+2)*VOL, x0 + (g+3)*VOL);")
        em.w(f"pack4_{L}(c4, c + g*VOL, c + (g+1)*VOL, c + (g+2)*VOL, c + (g+3)*VOL);")
        em.w(f"int64_t mm = (m==0)?1:m;")
        em.w(f"if (m==0) memcpy(outm + g*VOL, x0 + g*VOL, 4*VOL*8);")
        em.w(f"for (int64_t it=1; it<=mm; it++){{")
        em.ind += 1
        em.w(f"kZ4_{L}(w4); kY4_{L}(w4); kX4_{L}(w4, c4);")
        em.w(f"if (it==1) unpack4_{L}(w4, out1 + g*VOL, out1 + (g+1)*VOL, out1 + (g+2)*VOL, out1 + (g+3)*VOL);")
        em.w(f"if (it==mm && m!=0) unpack4_{L}(w4, outm + g*VOL, outm + (g+1)*VOL, outm + (g+2)*VOL, outm + (g+3)*VOL);")
        em.ind -= 1
        em.w(f"}}")
        em.ind -= 1
        em.w(f"}}")
        em.w(f"if (B > b4) run1_{L}(B - b4, m, x0 + b4*VOL, c + b4*VOL, out1 + b4*VOL, outm + b4*VOL);")
        em.ind -= 1
        em.w(f"}}")
    return em.code()

SIZES = (6,8,13,17,23,36,45,64)
PAD = {6:0, 8:0, 13:0, 17:0, 23:0, 36:0, 45:0, 64:16}      # slice pad (complex)
import os as _os2
ZVOL = {int(x) for x in _os2.environ.get("ZVOL","6,8,13,17,23").split(",") if x}
ROWPAD = {6:0, 8:0, 13:0, 17:0, 23:0, 36:0, 45:0, 64:4}    # row pad (complex)
def ROWST2(L):   # doubles between consecutive y-rows (= z-lines) in work layout
    return 2*(L + ROWPAD[L])
def SLICEST2(L): # doubles between consecutive x-slices in work layout
    return 2*(L*(L + ROWPAD[L]) + PAD[L])
import os as _os
_mf = _os.environ.get("MAPFUSE","23,36")
MAPFUSE = {int(x):True for x in _mf.split(",") if x}
_pc = _os.environ.get("PRIMECU2","")
PRIMECU2 = {int(x) for x in _pc.split(",") if x}
_ap = _os.environ.get("AOSPAIR","")
AOSPAIR = {int(x) for x in _ap.split(",") if x}
_asw = _os.environ.get("AOSSWEEP","6,8,13,17,23")
AOSSWEEP = {int(x) for x in _asw.split(",") if x}
_zp = _os.environ.get("ZPAIR","")
ZPAIR = {int(x) for x in _zp.split(",") if x}

PRELUDE = r'''
// auto-generated: specialized batched 3D FFT + z/(1+|z|) iteration
#include <immintrin.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

typedef __m512d V8;
#define LOADU(p) _mm512_loadu_pd(p)
#define STOREU(p,v) _mm512_storeu_pd((p),(v))
#define STREAMU(p,v) _mm512_stream_pd((p),(v))
#define MLOADZ(m,p) _mm512_maskz_loadu_pd((m),(p))
#define MSTORE(p,m,v) _mm512_mask_storeu_pd((p),(m),(v))
#define ADD _mm512_add_pd
#define SUB _mm512_sub_pd
#define MUL _mm512_mul_pd
#define FMA _mm512_fmadd_pd
#define FNMA _mm512_fnmadd_pd
#define FMAS _mm512_fmaddsub_pd
#define FMSA _mm512_fmsubadd_pd
#define SET1 _mm512_set1_pd
#define MAX _mm512_max_pd
#define SWAP(v) _mm512_permute_pd((v), 0x55)
#define XOR(a,b) _mm512_castsi512_pd(_mm512_xor_si512(_mm512_castpd_si512(a), _mm512_castpd_si512(b)))
#define SHUF2(a,b,imm) _mm512_shuffle_f64x2((a),(b),(imm))
#define TRAN4(i0,i1,i2,i3,o0,o1,o2,o3) do { \
    V8 t0_=SHUF2(i0,i1,0x44), t1_=SHUF2(i0,i1,0xEE), t2_=SHUF2(i2,i3,0x44), t3_=SHUF2(i2,i3,0xEE); \
    o0=SHUF2(t0_,t2_,0x88); o1=SHUF2(t0_,t2_,0xDD); o2=SHUF2(t1_,t3_,0x88); o3=SHUF2(t1_,t3_,0xDD); } while(0)

#define SGN_E (_mm512_castsi512_pd(_mm512_set_epi64(0,INT64_MIN,0,INT64_MIN,0,INT64_MIN,0,INT64_MIN)))
#define SGN_O (_mm512_castsi512_pd(_mm512_set_epi64(INT64_MIN,0,INT64_MIN,0,INT64_MIN,0,INT64_MIN,0)))
#define SGN_B (_mm512_castsi512_pd(_mm512_set1_epi64(INT64_MIN)))
#define ONE (SET1(1.0))
#define C2 (SET1(2.0))
#define C15 (SET1(1.5))
#define CHALF (SET1(0.5))
#define EPS_TINY (SET1(1e-300))
#define RSQ2 (0.70710678118654752440)
#define MRSQ2 (-0.70710678118654752440)
#define IDX_EVEN (_mm512_set_epi64(14,12,10,8,6,4,2,0))
#define IDX_ODD  (_mm512_set_epi64(15,13,11,9,7,5,3,1))
#define IDX_LO   (_mm512_set_epi64(3,3,2,2,1,1,0,0))
#define IDX_HI   (_mm512_set_epi64(7,7,6,6,5,5,4,4))

#ifndef UNPAD1
#define UNPAD1 0
#endif
#ifdef NOBARRIER
#define BARRIER
#else
#define BARRIER __asm__ __volatile__("" ::: "memory")
#endif

static double* WORK;
static void ntcopy(double* restrict dst, const double* restrict srcp, size_t ndoubles){
    size_t i = 0;
    if (ndoubles >= 131072 && (((uintptr_t)dst) & 63) == 0){
        for (; i + 8 <= ndoubles; i += 8)
            _mm512_stream_pd(dst + i, _mm512_loadu_pd(srcp + i));
        _mm_sfence();
        for (; i < ndoubles; i++) dst[i] = srcp[i];
        return;
    }
    memcpy(dst, srcp, ndoubles*8);
}
'''

def generate():
    parts = [PRELUDE]
    kernels = []
    for L in SIZES:
        kernels.append(emit_kernel_z(L))
        kernels.append(emit_kernel_yx(L, 'y'))
        if ROWPAD[L]:
            kernels.append(emit_kernel_z(L, copyin=True))
        kernels.append(emit_kernel_yx(L, 'x'))
        if L in XSPLIT:
            kernels.append(emit_xsplit(L))
        if L in AOS4:
            kernels.append(emit_kernel_aos(L))
            kernels.append(emit_driver_aos(L))
        kernels.append(emit_driver(L))
    # tables (TAB filled during emission above)
    parts.append(TAB.decls())
    parts.append(r'''
void initlib(void){
    const long double PIL2 = -6.283185307179586476925286766559005768L;
''' + TAB.init_code() + r'''
    if (!WORK) WORK = (double*)aligned_alloc(64, (size_t)3*((64*68)+16)*64*16 + 8192);
}
''')
    parts.extend(kernels)
    # dispatch
    dis = ["void run_size(int64_t L, int64_t B, int64_t m, double* x0, const double* c, double* out1, double* outm){",
           "    unsigned int mxcsr_ = _mm_getcsr();",
           "    _mm_setcsr(mxcsr_ | 0x8040u);  /* FTZ|DAZ */",
           "    switch(L){"]
    for L in SIZES:
        dis.append(f"        case {L}: run_{L}(B,m,x0,c,out1,outm); break;")
    dis.append("        default: break;")
    dis.append("    }")
    dis.append("    _mm_setcsr(mxcsr_);")
    dis.append("}")
    parts.append("\n".join(dis))
    # debug pass entries (z or y only, on a raw volume, single volume)
    dbg = ["void dbg_pass(int64_t L, int64_t which, double* v){", "    switch(L){"]
    for L in SIZES:
        dbg.append(f"        case {L}:")
        if L in ZVOL:
            dbg.append(f"            if (which==0) {{ kZ_{L}(v); }}")
        else:
            dbg.append(f"            if (which==0) {{ for (int s=0;s<{L};s++) kZ_{L}(v + s*{SLICEST2(L)}); }}")
        dbg.append(f"            else {{ for (int s=0;s<{L};s++) kY_{L}(v + s*{SLICEST2(L)}); }}")
        dbg.append(f"            break;")
    dbg.append("        default: break;")
    dbg.append("    }")
    dbg.append("}")
    parts.append("\n".join(dbg))

    # bench entries
    bn = ["void bench_kernel(int64_t L, int64_t which, double* v, double* v2, int64_t reps){",
          "    unsigned int mxcsr_ = _mm_getcsr();",
          "    _mm_setcsr(mxcsr_ | 0x8040u);",
          "    switch(L){"]
    for L in SIZES:
        bn.append(f"        case {L}:")
        if L in ZVOL:
            bn.append(f"            if (which==0) {{ for(int64_t r=0;r<reps;r++) kZ_{L}(v); }}")
        else:
            bn.append(f"            if (which==0) {{ for(int64_t r=0;r<reps;r++) for (int s=0;s<{L};s++) kZ_{L}(v + s*{SLICEST2(L)}); }}")
        bn.append(f"            if (which==1) {{ for(int64_t r=0;r<reps;r++) for (int s=0;s<{L};s++) kY_{L}(v + s*{SLICEST2(L)}); }}")
        bn.append(f"            if (which==2) {{ for(int64_t r=0;r<reps;r++) kX_{L}(v, v, v2); }}")
        if L in AOS4:
            bn.append(f"            if (which==3) {{ for(int64_t r=0;r<reps;r++) kZ4_{L}(v); }}")
            bn.append(f"            if (which==4) {{ for(int64_t r=0;r<reps;r++) kY4_{L}(v); }}")
            bn.append(f"            if (which==5) {{ for(int64_t r=0;r<reps;r++) kX4_{L}(v, v2); }}")
        if L in XSPLIT:
            R_, S_ = XSPLIT[L]
            SS2_ = SLICEST2(L)
            x1calls = " ".join([f"xs1_{L}_{g}(v + {g}*{SS2_}, {R_*SS2_});" for g in range(R_)])
            bn.append(f"            if (which==6) {{ for(int64_t r=0;r<reps;r++) {{ {x1calls} }} }}")
            bn.append(f"            if (which==7) {{ for(int64_t r=0;r<reps;r++) for (int h=0;h<{S_};h++) xs2_{L}(v + {R_}*h*{SS2_}, v + h*{SS2_}, v2 + h*{SS2_}, {SS2_}, {S_*SS2_}, {S_*SS2_}); }}")
        bn.append(f"            break;")
    bn.append("        default: break;")
    bn.append("    }")
    bn.append("    _mm_setcsr(mxcsr_);")
    bn.append("}")
    parts.append("\n".join(bn))
    return "\n\n".join(parts)


if __name__ == "__main__":
    code = generate()
    with open("/workdir/implementation.c", "w") as f:
        f.write(code)
    print(f"generated {len(code.splitlines())} lines")
