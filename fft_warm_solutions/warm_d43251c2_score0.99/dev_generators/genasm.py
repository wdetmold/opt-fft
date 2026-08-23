# ASM prime codelets: exact register allocation, baked strides.
import numpy as np
LD = np.longdouble
PI = np.longdouble('3.14159265358979323846264338327950288')

def hexd(x):
    return float(np.double(x)).hex()

class A:
    def __init__(self):
        self.lines = []
    def i(self, s):
        self.lines.append('        "' + s + '\\n\\t"')
    def slot(self):
        self.lines.append(('SLOT', len(self.lines)))
    def text(self):
        return "\n".join(l for l in self.lines if not isinstance(l, tuple))
    def riffle(self, chunks):
        slots = [i for i, l in enumerate(self.lines) if isinstance(l, tuple)]
        out = []
        assign = {}
        if slots and chunks:
            step = len(slots)/len(chunks)
            for k in range(len(chunks)):
                s = slots[min(int(k*step), len(slots)-1)]
                assign.setdefault(s, []).append(k)
        for i, l in enumerate(self.lines):
            if isinstance(l, tuple):
                for k in assign.get(i, []):
                    out.extend(chunks[k])
            else:
                out.append(l)
        if not slots:
            for ch in chunks: out.extend(ch)
        return "\n".join(out)


def tblname(N): return f"TBL_{N}"

def gen_tbl(N):
    """constant table: cos[1..h], sin[1..h], then map consts tiny,half,one at fixed slots"""
    h = (N-1)//2
    vals = []
    for n in range(1, h+1): vals.append(np.cos(2*PI*n/LD(N)))
    for n in range(1, h+1): vals.append(np.sin(2*PI*n/LD(N)))
    vals += [1e-30, 0.5, 1.5, 1.0]
    s = f"static const double ALIGN64 {tblname(N)}[{len(vals)}] = {{\n"
    s += ",\n".join(f"    {hexd(v)}" for v in vals)
    s += "\n};\n"
    return s, h

def emit_map2(a, zr, zi, t, TINY, HALF, ONEP5, ONE):
    """map2 on regs zr,zi in place; t = 4 temp regs; consts = byte offsets into [tb].
    Uses 213 forms with mem-broadcast to stay in 4 temps."""
    m, r, t1, t2 = t[0], t[1], t[2], t[3]
    a.i(f"vbroadcastsd {TINY}(%[tb]), %%{m}")
    a.i(f"vfmadd231pd %%{zi}, %%{zi}, %%{m}")
    a.i(f"vfmadd231pd %%{zr}, %%{zr}, %%{m}")
    a.i(f"vrsqrt14pd %%{m}, %%{r}")
    a.i(f"vmulpd {HALF}(%[tb])%{{1to8%}}, %%{m}, %%{t1}")    # h2 = 0.5*m (persist)
    a.i(f"vmulpd %%{r}, %%{r}, %%{t2}")                       # r*r
    a.i(f"vfnmadd213pd {ONEP5}(%[tb])%{{1to8%}}, %%{t1}, %%{t2}")  # t2 = 1.5 - h2*(r*r)
    a.i(f"vmulpd %%{t2}, %%{r}, %%{r}")
    a.i(f"vmulpd %%{r}, %%{r}, %%{t2}")
    a.i(f"vfnmadd213pd {ONEP5}(%[tb])%{{1to8%}}, %%{t1}, %%{t2}")
    a.i(f"vmulpd %%{t2}, %%{r}, %%{r}")
    a.i(f"vmulpd %%{r}, %%{m}, %%{t1}")                       # mg0
    a.i(f"vfnmadd231pd %%{t1}, %%{t1}, %%{m}")                # e2 -> m
    a.i(f"vmulpd {HALF}(%[tb])%{{1to8%}}, %%{r}, %%{t2}")     # hr1
    a.i(f"vfmadd231pd %%{t2}, %%{m}, %%{t1}")                 # mag -> t1
    a.i(f"vaddpd {ONE}(%[tb])%{{1to8%}}, %%{t1}, %%{m}")      # u
    a.i(f"vrcp14pd %%{m}, %%{r}")                             # w0
    a.i(f"vbroadcastsd {ONE}(%[tb]), %%{t2}")
    a.i(f"vfnmadd231pd %%{r}, %%{m}, %%{t2}")                 # e3
    a.i(f"vfmadd231pd %%{t2}, %%{r}, %%{r}")                  # a
    a.i(f"vmulpd %%{t2}, %%{t2}, %%{t2}")                     # ee
    a.i(f"vfmadd231pd %%{t2}, %%{r}, %%{r}")                  # w2
    a.i(f"vmulpd %%{r}, %%{zr}, %%{zr}")
    a.i(f"vmulpd %%{r}, %%{zi}, %%{zi}")

if __name__ == "__main__":
    a = A()
    emit_map2(a, "zmm1", "zmm2", ["zmm3","zmm4","zmm5","zmm6","zmm7"], "zmm8","zmm9","zmm10")
    print(a.text()[:800])

def z(n): return f"zmm{n}"

def hwmap_chunk(eoff, regs, TINY, ONE, onereg=None):
    zr, zi, m = regs
    L = []
    def i(s): L.append('        "' + s + '\\n\\t"')
    i(f"vmovapd {eoff}(%[xm]), %%{zr}")
    i(f"vmovapd {eoff+64}(%[xm]), %%{zi}")
    i(f"vbroadcastsd {TINY}(%[tb]), %%{m}")
    i(f"vfmadd231pd %%{zi}, %%{zi}, %%{m}")
    i(f"vfmadd231pd %%{zr}, %%{zr}, %%{m}")
    i(f"vsqrtpd %%{m}, %%{m}")
    i(f"vaddpd {ONE}(%[tb])%{{1to8%}}, %%{m}, %%{m}")
    if onereg:
        i(f"vdivpd %%{m}, %%{onereg}, %%{m}")
        i(f"vmulpd %%{m}, %%{zr}, %%{zr}")
        i(f"vmulpd %%{m}, %%{zi}, %%{zi}")
    else:
        i(f"vdivpd %%{m}, %%{zr}, %%{zr}")
        i(f"vdivpd %%{m}, %%{zi}, %%{zi}")
    i(f"vmovapd %%{zr}, {eoff}(%[xm])")
    i(f"vmovapd %%{zi}, {eoff+64}(%[xm])")
    return L

def pf_chunk(eoff):
    L = []
    def i(s): L.append('        "' + s + '\\n\\t"')
    i(f"prefetcht0 {eoff}(%[xn])")
    i(f"prefetcht0 {eoff+64}(%[xn])")
    return L

NRSPLIT = {13: 8, 17: 9, 23: 6}
def emit_prime_asm(N, variant):
    """variants: z,y,x (plain), zm (map-on-load), zb (map-block+dft), xm (x + pipelined map of prev pencil), mb (map block only)."""
    h = (N-1)//2
    if variant == 'mb':
        a = A()
        SBM = N*N*128
        MAPT0 = (16*h, 16*h+8, 16*h+16, 16*h+24)
        e = 0
        while e < N:
            cnt = min(4, N-e)
            for q in range(cnt):
                a.i(f"vmovapd {(e+q)*SBM}(%[xm]), %%{z(4*q)}")
                a.i(f"vmovapd {(e+q)*SBM+64}(%[xm]), %%{z(4*q+1)}")
            for q in range(cnt):
                emit_map2(a, z(4*q), z(4*q+1), [z(16+4*q), z(17+4*q), z(18+4*q), z(19+4*q)],
                          MAPT0[0], MAPT0[1], MAPT0[2], MAPT0[3])
            for q in range(cnt):
                a.i(f"vmovapd %%{z(4*q)}, {(e+q)*SBM}(%[xm])")
                a.i(f"vmovapd %%{z(4*q+1)}, {(e+q)*SBM+64}(%[xm])")
            e += cnt
        clob = ", ".join(f'"zmm{i}"' for i in range(32)) + ', "memory"'
        return f"""static void __attribute__((noinline)) adft{N}_mb(double* Xm){{
    __asm__ volatile(
{a.text()}
    : : [xm]"r"(Xm), [tb]"r"({tblname(N)}) : {clob});
}}
"""
    SB = {'zm':128, 'zb':128, 'z':128, 'y':N*128, 'x':N*N*128, 'xm':N*N*128}[variant]
    mapblock = (variant == 'zb')
    pipemap = (variant == 'xm')
    if pipemap: variant = 'x'
    mapin = (variant == 'zm')
    withc = (variant == 'x')
    a = A()
    # register plan
    if N == 13:
        ncost = 6
        cosr = [z(i) for i in range(6)]
        Ea = {k: (z(6+2*(k-1)), z(7+2*(k-1))) for k in range(1,7)}
        Oa = {k: (z(18+2*(k-1)), z(19+2*(k-1))) for k in range(1,7)}
        eblocks = [list(range(1,7))]
        oblocks = [list(range(1,7))]
        regs_free_E = [z(i) for i in range(18,32)]
        regs_free_O = [z(30), z(31)]
        use_escr = False
        sum_in_E = True
        sumr, sumi = z(28), z(29)   # persist through E only (stored at end of E)
    elif N == 17:
        cosr = [z(i) for i in range(8)]
        Ea = {k: (z(8+2*(k-1)), z(9+2*(k-1))) for k in range(1,9)}
        Oa = Ea  # same reg slots reused in O phase
        eblocks = [list(range(1,9))]
        oblocks = [list(range(1,9))]
        regs_free_E = [z(i) for i in range(24,32)]
        regs_free_O = [z(i) for i in range(24,32)]
        use_escr = True
        sum_in_E = False   # sum computed in O phase from stored u
        sumr, sumi = z(24), z(25)  # during O phase persist
    elif N == 23:
        cosr = [z(i) for i in range(11)]
        eblocks = [list(range(1,7)), list(range(7,12))]
        oblocks = [list(range(1,7)), list(range(7,12))]
        Ea = {}
        for blk in eblocks:
            for idx, k in enumerate(blk):
                Ea[k] = (z(12+2*idx), z(13+2*idx))
        Oa = Ea
        regs_free_E = [z(i) for i in range(24,32)]
        regs_free_O = [z(i) for i in range(24,32)]
        use_escr = True
        sum_in_E = 2   # in E block 2
        sumr, sumi = z(22+0), z(23+0)  # during E block2: accs z12..z21, sum z22,z23
    else:
        raise ValueError
    hB = 8*h  # sin table byte start
    MAPT = (hB*2, hB*2+8, hB*2+16, hB*2+24)  # tiny, half, 1.5, 1.0
    def xoff(j): return j*SB
    store_u = (N in (17, 23))  # store u to AB (17: for sum; 23: for E2)
    ABJ = 256 if store_u else 128  # bytes per j in AB
    def ab_u(j): return (j-1)*ABJ
    def ab_v(j): return (j-1)*ABJ + (128 if store_u else 0)
    # ---- build asm ----
    if pipemap and N == 23:
        a.i(f"vbroadcastsd {16*h+24}(%[tb]), %%zmm11")
    if pipemap:
        NNR = NRSPLIT[N]
        MAPT0 = (16*h, 16*h+8, 16*h+16, 16*h+24)
        e = 0
        while e < NNR:
            cnt = min(4, NNR-e)
            for q in range(cnt):
                a.i(f"vmovapd {(e+q)*SB}(%[xm]), %%{z(4*q)}")
                a.i(f"vmovapd {(e+q)*SB+64}(%[xm]), %%{z(4*q+1)}")
            for q in range(cnt):
                emit_map2(a, z(4*q), z(4*q+1), [z(16+4*q), z(17+4*q), z(18+4*q), z(19+4*q)],
                          MAPT0[0], MAPT0[1], MAPT0[2], MAPT0[3])
            for q in range(cnt):
                a.i(f"vmovapd %%{z(4*q)}, {(e+q)*SB}(%[xm])")
                a.i(f"vmovapd %%{z(4*q+1)}, {(e+q)*SB+64}(%[xm])")
            e += cnt
    if mapblock:
        # pipelined in-place map over all N elements, 4 at a time interleaved
        # use regs: data zmm0..zmm15 (4 pairs), temps zmm16..31 (4 per chain)
        MAPT0 = (16*h, 16*h+8, 16*h+16, 16*h+24)
        e = 0
        while e < N:
            cnt = min(4, N-e)
            for q in range(cnt):
                a.i(f"vmovapd {(e+q)*SB}(%[x]), %%{z(4*q)}")
                a.i(f"vmovapd {(e+q)*SB+64}(%[x]), %%{z(4*q+1)}")
            for q in range(cnt):
                emit_map2(a, z(4*q), z(4*q+1), [z(16+4*q), z(17+4*q), z(18+4*q), z(19+4*q)],
                          MAPT0[0], MAPT0[1], MAPT0[2], MAPT0[3])
            for q in range(cnt):
                a.i(f"vmovapd %%{z(4*q)}, {(e+q)*SB}(%[x])")
                a.i(f"vmovapd %%{z(4*q+1)}, {(e+q)*SB+64}(%[x])")
            e += cnt
    # broadcast cos
    for i in range(len(cosr)):
        a.i(f"vbroadcastsd {8*i}(%[tb]), %%{cosr[i]}")
    # x0 load (+map)
    t0r, t0i = regs_free_E[0], regs_free_E[1]
    a.i(f"vmovapd (%[x]), %%{t0r}")
    a.i(f"vmovapd 64(%[x]), %%{t0i}")
    if mapin:
        tm = regs_free_E[2:6]
        emit_map2(a, t0r, t0i, tm, MAPT[0], MAPT[1], MAPT[2], MAPT[3])
    # x0 to stack slot (for later blocks / X0)
    a.i(f"vmovapd %%{t0r}, (%[es])")       # es[0] = x0r (slot -  use first pair for x0)
    a.i(f"vmovapd %%{t0i}, 64(%[es])")
    ES0 = 128  # E results start at es+128
    def es_e(k): return ES0 + (k-1)*128
    # E phase
    first_block = True
    for bi, blk in enumerate(eblocks):
        # init accs from x0
        if not first_block:
            a.i(f"vmovapd (%[es]), %%{regs_free_E[0]}")
            a.i(f"vmovapd 64(%[es]), %%{regs_free_E[1]}")
            t0r, t0i = regs_free_E[0], regs_free_E[1]
        for idx, k in enumerate(blk):
            er, ei = (z(12+2*idx), z(13+2*idx)) if N==23 else Ea[k]
            a.i(f"vmovapd %%{t0r}, %%{er}")
            a.i(f"vmovapd %%{t0i}, %%{ei}")
        do_sum = (sum_in_E is True and first_block) or (sum_in_E == 2 and bi == 1)
        if do_sum:
            a.i(f"vmovapd %%{t0r}, %%{sumr}")
            a.i(f"vmovapd %%{t0i}, %%{sumi}")
        for j in range(1, h+1):
            if first_block:
                pr, pi, qr, qi = regs_free_E[2], regs_free_E[3], regs_free_E[4], regs_free_E[5]
                if N == 13: pr, pi, qr, qi = z(20), z(21), z(22), z(23)
                a.i(f"vmovapd {xoff(j)}(%[x]), %%{pr}")
                a.i(f"vmovapd {xoff(j)+64}(%[x]), %%{pi}")
                if mapin:
                    tm = [r for r in regs_free_E if r not in (pr,pi,qr,qi)][:4]
                    emit_map2(a, pr, pi, tm, MAPT[0], MAPT[1], MAPT[2], MAPT[3])
                a.i(f"vmovapd {xoff(N-j)}(%[x]), %%{qr}")
                a.i(f"vmovapd {xoff(N-j)+64}(%[x]), %%{qi}")
                if mapin:
                    emit_map2(a, qr, qi, tm, MAPT[0], MAPT[1], MAPT[2], MAPT[3])
                ur, ui = pr, pi  # after computing, reuse
                vr, vi = qr, qi
                # u = p+q, v = p-q: need temps: compute v first into spare then u in place
                sp1 = tm[0] if mapin else (regs_free_E[0] if N==23 else regs_free_E[6])
                sp2 = tm[1] if mapin else (regs_free_E[1] if N==23 else regs_free_E[7])
                a.i(f"vsubpd %%{qr}, %%{pr}, %%{sp1}")   # vr
                a.i(f"vaddpd %%{qr}, %%{pr}, %%{ur}")    # ur (overwrites pr)
                a.i(f"vsubpd %%{qi}, %%{pi}, %%{sp2}")   # vi
                a.i(f"vaddpd %%{qi}, %%{pi}, %%{ui}")    # ui
                a.i(f"vmovapd %%{sp1}, {ab_v(j)}(%[ab])")
                a.i(f"vmovapd %%{sp2}, {ab_v(j)+64}(%[ab])")
                if store_u:
                    a.i(f"vmovapd %%{ur}, {ab_u(j)}(%[ab])")
                    a.i(f"vmovapd %%{ui}, {ab_u(j)+64}(%[ab])")
            else:
                ur, ui = regs_free_E[2], regs_free_E[3]
                a.i(f"vmovapd {ab_u(j)}(%[ab]), %%{ur}")
                a.i(f"vmovapd {ab_u(j)+64}(%[ab]), %%{ui}")
            if do_sum:
                a.i(f"vaddpd %%{ur}, %%{sumr}, %%{sumr}")
                a.i(f"vaddpd %%{ui}, %%{sumi}, %%{sumi}")
            for idx, k in enumerate(blk):
                er, ei = (z(12+2*idx), z(13+2*idx)) if N==23 else Ea[k]
                n = (k*j) % N; n = min(n, N-n)
                a.i(f"vfmadd231pd %%{cosr[n-1]}, %%{ur}, %%{er}")
                a.i(f"vfmadd231pd %%{cosr[n-1]}, %%{ui}, %%{ei}")
            a.slot()
        # store E accs (+sum/X0)
        if use_escr or N == 23:
            for idx, k in enumerate(blk):
                er, ei = (z(12+2*idx), z(13+2*idx)) if N==23 else Ea[k]
                a.i(f"vmovapd %%{er}, {es_e(k)}(%[es])")
                a.i(f"vmovapd %%{ei}, {es_e(k)+64}(%[es])")
        if do_sum and sum_in_E:
            # X0 output
            if withc:
                a.i(f"vaddpd (%[c]), %%{sumr}, %%{sumr}")
                a.i(f"vaddpd 64(%[c]), %%{sumi}, %%{sumi}")
            a.i(f"vmovapd %%{sumr}, (%[x])")
            a.i(f"vmovapd %%{sumi}, 64(%[x])")
        first_block = False
    # O phase
    for i in range(len(cosr)):
        a.i(f"vbroadcastsd {hB + 8*i}(%[tb]), %%{cosr[i]}")  # sin now
    osum_pending = not sum_in_E
    for bi, blk in enumerate(oblocks):
        for idx, k in enumerate(blk):
            orr, oi = (z(12+2*idx), z(13+2*idx)) if N==23 else Oa[k]
            a.i(f"vxorpd %%{orr}, %%{orr}, %%{orr}")
            a.i(f"vxorpd %%{oi}, %%{oi}, %%{oi}")
        do_osum = osum_pending and bi == 0
        if do_osum:
            a.i(f"vmovapd (%[es]), %%{sumr}")
            a.i(f"vmovapd 64(%[es]), %%{sumi}")
        for j in range(1, h+1):
            vr0, vi0 = regs_free_O[-2], regs_free_O[-1]
            a.i(f"vmovapd {ab_v(j)}(%[ab]), %%{vr0}")
            a.i(f"vmovapd {ab_v(j)+64}(%[ab]), %%{vi0}")
            if do_osum:
                ur0 = regs_free_O[0] if N != 17 else z(26)
                ui0 = regs_free_O[1] if N != 17 else z(27)
                a.i(f"vmovapd {ab_u(j)}(%[ab]), %%{ur0}")
                a.i(f"vmovapd {ab_u(j)+64}(%[ab]), %%{ui0}")
                a.i(f"vaddpd %%{ur0}, %%{sumr}, %%{sumr}")
                a.i(f"vaddpd %%{ui0}, %%{sumi}, %%{sumi}")
            for idx, k in enumerate(blk):
                orr, oi = (z(12+2*idx), z(13+2*idx)) if N==23 else Oa[k]
                n = (k*j) % N
                if n <= h:
                    a.i(f"vfmadd231pd %%{cosr[n-1]}, %%{vr0}, %%{orr}")
                    a.i(f"vfmadd231pd %%{cosr[n-1]}, %%{vi0}, %%{oi}")
                else:
                    a.i(f"vfnmadd231pd %%{cosr[N-n-1]}, %%{vr0}, %%{orr}")
                    a.i(f"vfnmadd231pd %%{cosr[N-n-1]}, %%{vi0}, %%{oi}")
            if N != 13:
                a.slot()
        if do_osum:
            if withc:
                a.i(f"vaddpd (%[c]), %%{sumr}, %%{sumr}")
                a.i(f"vaddpd 64(%[c]), %%{sumi}, %%{sumi}")
            a.i(f"vmovapd %%{sumr}, (%[x])")
            a.i(f"vmovapd %%{sumi}, 64(%[x])")
            osum_pending = False
        # combine for this block
        for idx, k in enumerate(blk):
            orr, oi = (z(12+2*idx), z(13+2*idx)) if N==23 else Oa[k]
            if use_escr or N == 23:
                er, ei = regs_free_O[0], regs_free_O[1]
                if N == 17: er, ei = z(26), z(27)
                a.i(f"vmovapd {es_e(k)}(%[es]), %%{er}")
                a.i(f"vmovapd {es_e(k)+64}(%[es]), %%{ei}")
            else:
                er, ei = Ea[k]
            w1, w2 = (z(30), z(31)) if N != 13 else (z(30), z(31))
            # X[k]   = (er + oi, ei - or)
            # X[N-k] = (er - oi, ei + or)
            a.i(f"vaddpd %%{oi}, %%{er}, %%{w1}")
            a.i(f"vsubpd %%{orr}, %%{ei}, %%{w2}")
            if withc:
                a.i(f"vaddpd {xoff(k)//SB*128}(%[c]), %%{w1}, %%{w1}")
                a.i(f"vaddpd {xoff(k)//SB*128+64}(%[c]), %%{w2}, %%{w2}")
            a.i(f"vmovapd %%{w1}, {xoff(k)}(%[x])")
            a.i(f"vmovapd %%{w2}, {xoff(k)+64}(%[x])")
            a.i(f"vsubpd %%{oi}, %%{er}, %%{w1}")
            a.i(f"vaddpd %%{orr}, %%{ei}, %%{w2}")
            if withc:
                a.i(f"vaddpd {(N-k)*128}(%[c]), %%{w1}, %%{w1}")
                a.i(f"vaddpd {(N-k)*128+64}(%[c]), %%{w2}, %%{w2}")
            a.i(f"vmovapd %%{w1}, {xoff(N-k)}(%[x])")
            a.i(f"vmovapd %%{w2}, {xoff(N-k)+64}(%[x])")
    # wrap in C function
    clob = ", ".join(f'"zmm{i}"' for i in range(32)) + ', "memory"'
    vname = 'xm' if pipemap else variant
    cargs = "double* X, double* AB, double* ES" + (", const double* Ct" if withc else "")
    if pipemap: cargs += ", double* Xm, const double* Xn"
    ops = '[x]"r"(X), [ab]"r"(AB), [es]"r"(ES), [tb]"r"(' + tblname(N) + ')'
    if withc: ops += ', [c]"r"(Ct)'
    if pipemap: ops += ', [xm]"r"(Xm), [xn]"r"(Xn)'
    if pipemap:
        MAPT0 = (16*h, 16*h+8, 16*h+16, 16*h+24)
        NNR = NRSPLIT[N]
        onereg = 'zmm11' if N == 23 else None
        hwregs = {13: ('zmm30','zmm31','zmm26'), 17: ('zmm30','zmm31','zmm26'), 23: ('zmm28','zmm29','zmm30')}[N]
        chunks = [hwmap_chunk(e*SB, hwregs, MAPT0[0], MAPT0[3], onereg) for e in range(NNR, N)]
        chunks += [pf_chunk(e*SB) for e in range(N)]
        body = a.riffle(chunks)
    else:
        body = a.text()
    fn = f"""static void __attribute__((noinline)) adft{N}_{vname}({cargs}){{
    __asm__ volatile(
{body}
    : : {ops} : {clob});
}}
"""
    return fn

if __name__ == "__main__":
    pass
