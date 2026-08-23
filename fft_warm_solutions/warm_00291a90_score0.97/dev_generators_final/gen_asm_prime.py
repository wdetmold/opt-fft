RCPMAGIC = False
import numpy as np
from genlib import hexd
from gen_a import prime_tables, fold_idx
from gen_asm import A

def table_c(L):
    h, cosv, sinv = prime_tables(L)
    vals = cosv + sinv + [1e-30, 1.0, 0.5, float.fromhex('0x1.e6238da3c2118p+1022')]
    body = ", ".join(hexd(v) for v in vals)
    return (f"static const double TB_{L}[{len(vals)}] ALIGN64 = {{ {body} }};\n"
            f"static double AS_{L}[{(2*h+2)*16}] ALIGN64;\n"
            f"static double EO_{L}[{4*h*8}] ALIGN64;\n"
            f"static double MM_{L}[{L*16}] ALIGN64;\n"
            f"static double XS_{L}[{L*16}] ALIGN64;\n")

def toff(L, kind, m=None):
    h = (L-1)//2
    if kind == 'cos': return (m-1)*8
    if kind == 'sin': return (h + m-1)*8
    return (2*h + {'TINY':0,'ONE':1,'HALF':2,'MAGIC':3}[kind])*8

def emit_dft_core(a, L, kb1, kb2, src, ses, dst, des, force_eo=False):
    """plain DFT src->dst (no map). Writes X0..X_{L-1}."""
    h, cosv, sinv = prime_tables(L)
    kblocks1 = [list(range(s, min(s+kb1, h+1))) for s in range(1, h+1, kb1)]
    kblocks2 = [list(range(s, min(s+kb2, h+1))) for s in range(1, h+1, kb2)]
    need_eo = len(kblocks1) > 1 or len(kblocks2) > 1 or force_eo
    def soff(q): return q*ses*8
    def doff(q): return q*des*8
    first = True
    for kb in kblocks1:
        C = {}
        for mth in sorted(set(fold_idx(k*j, L)[0] for k in kb for j in range(1, h+1))):
            C[mth] = a.bcast('tab', toff(L, 'cos', mth))
        if first:
            x0r = a.ld(src, soff(0)); x0i = a.ld(src, soff(0)+64)
            if need_eo:
                a.st('A', (2*h)*128, x0r); a.st('A', (2*h)*128+64, x0i)
        else:
            x0r = a.ld('A', (2*h)*128); x0i = a.ld('A', (2*h)*128+64)
        acc = {}
        for k in kb:
            acc[k] = (a.mov(x0r), a.mov(x0i))
        sE = (x0r, x0i) if first else None
        if not first:
            a.rel(x0r); a.rel(x0i)
        for j in range(1, h+1):
            if first:
                ar = a.ld(src, soff(j)); br = a.ld(src, soff(L-j))
                er = a.add(ar, br)
                if need_eo:
                    orr = a.sub(ar, br)
                    a.st('EO', (4*(j-1)+2)*64, orr); a.rel(orr)
                a.rel(ar); a.rel(br)
                ai = a.ld(src, soff(j)+64); bi = a.ld(src, soff(L-j)+64)
                ei = a.add(ai, bi)
                if need_eo:
                    oi = a.sub(ai, bi)
                    a.st('EO', (4*(j-1)+3)*64, oi); a.rel(oi)
                a.rel(ai); a.rel(bi)
                if need_eo:
                    a.st('EO', (4*(j-1))*64, er); a.st('EO', (4*(j-1)+1)*64, ei)
                a.ins(f"vaddpd %%zmm{er}, %%zmm{sE[0]}, %%zmm{sE[0]}")
                a.ins(f"vaddpd %%zmm{ei}, %%zmm{sE[1]}, %%zmm{sE[1]}")
            else:
                er = a.ld('EO', (4*(j-1))*64); ei = a.ld('EO', (4*(j-1)+1)*64)
            for k in kb:
                mth, sgn = fold_idx(k*j, L)
                a.fma(acc[k][0], C[mth], er)
                a.fma(acc[k][1], C[mth], ei)
            a.rel(er); a.rel(ei)
        for k in kb:
            a.st('A', (2*(k-1))*128, acc[k][0]); a.st('A', (2*(k-1)+1)*128, acc[k][1])
            a.rel(acc[k][0]); a.rel(acc[k][1])
        for c in C.values(): a.rel(c)
        if first:
            a.st(dst, doff(0), sE[0]); a.st(dst, doff(0)+64, sE[1])
            a.rel(sE[0]); a.rel(sE[1])
        first = False
    for kb in kblocks2:
        S = {}
        for mth in sorted(set(fold_idx(k*j, L)[0] for k in kb for j in range(1, h+1))):
            S[mth] = a.bcast('tab', toff(L, 'sin', mth))
        acc = {}
        for j in range(1, h+1):
            if need_eo:
                orr = a.ld('EO', (4*(j-1)+2)*64); oi = a.ld('EO', (4*(j-1)+3)*64)
            else:
                ar = a.ld(src, soff(j)); br = a.ld(src, soff(L-j))
                orr = a.sub(ar, br, kill=(ar, br))
                ai = a.ld(src, soff(j)+64); bi = a.ld(src, soff(L-j)+64)
                oi = a.sub(ai, bi, kill=(ai, bi))
            for k in kb:
                mth, sgn = fold_idx(k*j, L)
                if k not in acc:
                    assert sgn > 0
                    acc[k] = (a.mul(S[mth], orr), a.mul(S[mth], oi))
                else:
                    if sgn > 0:
                        a.fma(acc[k][0], S[mth], orr); a.fma(acc[k][1], S[mth], oi)
                    else:
                        a.fnma(acc[k][0], S[mth], orr); a.fnma(acc[k][1], S[mth], oi)
            a.rel(orr); a.rel(oi)
        for s in S.values(): a.rel(s)
        for k in kb:
            Br, Bi = acc[k]
            Ar = a.ld('A', (2*(k-1))*128); Ai = a.ld('A', (2*(k-1)+1)*128)
            Xkr  = a.add(Ar, Bi)
            Xlkr = a.sub(Ar, Bi, kill=(Ar, Bi))
            Xki  = a.sub(Ai, Br)
            Xlki = a.add(Ai, Br, kill=(Ai, Br))
            a.st(dst, doff(k), Xkr); a.st(dst, doff(k)+64, Xki)
            a.st(dst, doff(L-k), Xlkr); a.st(dst, doff(L-k)+64, Xlki)
            a.rel(Xkr); a.rel(Xki); a.rel(Xlkr); a.rel(Xlki)

PRIME_MAP_SCHED = {13: (3, 8), 17: (3, 8), 23: (3, 8)}
def emit_map_phase(a, L, src, dsts, gmax=3, delta=8, toffs=None, pf=None):
    if L in PRIME_MAP_SCHED:
        gmax, delta = PRIME_MAP_SCHED[L]
    """software-pipelined map over points 0..L-1."""
    names = ('TINY','ONE','HALF','MAGIC') if RCPMAGIC else ('TINY','ONE','HALF')
    if toffs is None:
        mc = {n: a.bcast('tab', toff(L, n)) for n in names}
    else:
        mc = {n: a.bcast('tab', toffs[n]) for n in names}
    emit_map_points(a, mc, L, src, dsts, gmax, delta, pf, fmode=False)
    for r in mc.values(): a.rel(r)

def emit_map_points(a, mc, L, src, dsts, gmax=3, delta=8, pf=None, fmode=True):
    sbase, ss = src
    pts = list(range(L))
    ng = (L + gmax - 1) // gmax
    base = L // ng; rem = L % ng
    groups = []
    i = 0
    for gi in range(ng):
        sz = base + (1 if gi < rem else 0)
        groups.append(pts[i:i+sz]); i += sz

    def stage(s, q, d):
        if s == 0:
            zr = a.ld(sbase, q*ss*8); cr = a.ld('pc', q*128)
            d['zr'] = a.add(zr, cr, kill=(zr, cr))
            zi = a.ld(sbase, q*ss*8+64); ci = a.ld('pc', q*128+64)
            d['zi'] = a.add(zi, ci, kill=(zi, ci))
        elif s == 1:
            m = a.mul(d['zr'], d['zr'])
            a.fma(m, d['zi'], d['zi'])
            a.ins(f"vaddpd %%zmm{mc['TINY']}, %%zmm{m}, %%zmm{m}")
            d['m'] = m
        elif s == 2:
            if fmode:
                f = a.alloc_low()
                a.ins(f"vcvtpd2ps %%zmm{d['m']}, %%ymm{f}")
                a.ins(f"vrsqrtps %%ymm{f}, %%ymm{f}")
                a.ins(f"vcvtps2pd %%ymm{f}, %%zmm{f}")
                d['r0'] = f
            else:
                d['r0'] = a.rsqrt(d['m'])
        elif s == 3:
            d['t'] = a.mul(d['m'], d['r0'])
            d['hr'] = a.mul(d['r0'], mc['HALF'])
        elif s == 4:
            eh = a.mov(mc['HALF'])
            a.fnma(eh, d['t'], d['hr'])
            a.rel(d['t']); a.rel(d['hr'])
            d['eh'] = eh
        elif s == 5:
            r1 = a.mov(d['r0'])
            a.fma(r1, d['r0'], d['eh'])
            a.rel(d['r0']); a.rel(d['eh'])
            if fmode:
                t2 = a.mul(d['m'], r1)
                hr2 = a.mul(r1, mc['HALF'])
                eh2 = a.mov(mc['HALF'])
                a.fnma(eh2, t2, hr2)
                a.rel(t2); a.rel(hr2)
                r2 = a.mov(r1)
                a.fma(r2, r1, eh2)
                a.rel(r1); a.rel(eh2)
                d['r1'] = r2
            else:
                d['r1'] = r1
        elif s == 6:
            d['mg0'] = a.mul(d['m'], d['r1'])
            d['hr1'] = a.mul(d['r1'], mc['HALF'], kill=(d['r1'],))
        elif s == 7:
            e2 = a.mov(d['m'])
            a.fnma(e2, d['mg0'], d['mg0'])
            a.rel(d['m'])
            a.fma(d['mg0'], e2, d['hr1'])
            a.rel(e2); a.rel(d['hr1'])
            d['u'] = a.add(d['mg0'], mc['ONE'], kill=(d['mg0'],))
        elif s == 8:
            if RCPMAGIC:
                w0 = a.alloc()
                a.ins(f"vpsubq %%zmm{d['u']}, %%zmm{mc['MAGIC']}, %%zmm{w0}")
                d['w0'] = w0
            elif fmode:
                f = a.alloc_low()
                a.ins(f"vcvtpd2ps %%zmm{d['u']}, %%ymm{f}")
                a.ins(f"vrcpps %%ymm{f}, %%ymm{f}")
                a.ins(f"vcvtps2pd %%ymm{f}, %%zmm{f}")
                d['w0'] = f
            else:
                d['w0'] = a.rcp(d['u'])
        elif s == 9:
            e3 = a.mov(mc['ONE'])
            a.fnma(e3, d['u'], d['w0'])
            a.rel(d['u'])
            d['e3'] = e3
        elif s == 10:
            w1 = a.mov(d['w0'])
            a.fma(w1, d['w0'], d['e3'])
            a.rel(d['w0'])
            d['ee'] = a.mul(d['e3'], d['e3'], kill=(d['e3'],))
            d['w1'] = w1
        elif s == 11:
            w2 = a.mov(d['w1'])
            a.fma(w2, d['w1'], d['ee'])
            a.rel(d['w1'])
            if RCPMAGIC:
                e4 = a.mul(d['ee'], d['ee'], kill=(d['ee'],))
                w3 = a.mov(w2)
                a.fma(w3, w2, e4)
                a.rel(w2)
                e8 = a.mul(e4, e4, kill=(e4,))
                w4 = a.mov(w3)
                a.fma(w4, w3, e8)
                a.rel(w3); a.rel(e8)
                w3 = w4
            elif fmode:
                e4 = a.mul(d['ee'], d['ee'], kill=(d['ee'],))
                w3 = a.mov(w2)
                a.fma(w3, w2, e4)
                a.rel(w2); a.rel(e4)
            else:
                a.rel(d['ee'])
                w3 = w2
            xr = a.mul(d['zr'], w3, kill=(d['zr'],))
            xi = a.mul(d['zi'], w3, kill=(d['zi'],))
            a.rel(w3)
            for (db, dstr) in dsts:
                a.st(db, q*dstr*8, xr); a.st(db, q*dstr*8+64, xi)
            a.rel(xr); a.rel(xi)

    NST = 12
    state = [dict() for _ in range(L)]
    total = NST + delta*(len(groups)-1)
    pfq = []
    if pf is not None:
        pfop, nbytes = pf
        pfq = list(range(0, nbytes, 64))
    emitted = 0
    nsteps = sum(1 for t in range(total) for gi in range(len(groups)) if 0 <= t - delta*gi < NST)
    step_i = 0
    for t in range(total):
        for gi, g in enumerate(groups):
            s = t - delta*gi
            if 0 <= s < NST:
                for q in g:
                    stage(s, q, state[q])
                step_i += 1
                if pfq:
                    want = (step_i * len(pfq)) // nsteps
                    while emitted < want:
                        a.ins(f"prefetcht0 {pfq[emitted]}(%[{pfop}])")
                        emitted += 1

def gen_prime_asm_fn(L, kb1, kb2, name, ses, mapmode, nextmode, snap, PSZ, pfn=False):
    a = A()
    if not mapmode:
        emit_dft_core(a, L, kb1, kb2, 'px', ses, 'px', ses)
    else:
        emit_dft_core(a, L, kb1, kb2, 'px', ses, 'XS', 16)
        assert not a.live
        if nextmode:
            dsts = [('M', 16)] + ([('ps', PSZ)] if snap else [])
            emit_map_phase(a, L, ('XS', 16), dsts, pf=(('pcn', L*128) if pfn else None))
            assert not a.live
            emit_dft_core(a, L, kb1, kb2, 'M', 16, 'px', ses)
        else:
            dsts = [('px', ses)] + ([('ps', PSZ)] if snap else [])
            emit_map_phase(a, L, ('XS', 16), dsts)
    assert not a.live, f"live at end: {a.live}"
    body = "\\n\\t".join(a.lines)
    args = ["double* px"]
    ops = [f'[px]"r"(px)', f'[tab]"r"(TB_{L})', f'[A]"r"(AS_{L})', f'[EO]"r"(EO_{L})', f'[M]"r"(MM_{L})', f'[XS]"r"(XS_{L})']
    if mapmode:
        args.append("const double* pc"); ops.append('[pc]"r"(pc)')
    if pfn:
        args.append("const double* pcn"); ops.append('[pcn]"r"(pcn)')
    if snap:
        args.append("double* ps"); ops.append('[ps]"r"(ps)')
    clob = ", ".join(f'"zmm{i}"' for i in range(32)) + ', "memory"'
    return f"""static void __attribute__((noinline)) {name}({", ".join(args)}){{
    __asm__ volatile("{body}"
    : : {", ".join(ops)}
    : {clob});
}}
"""

def gen_prime_asm(L, kb1, kb2, PS, pv=None):
    PSZ = PS*16
    out = [table_c(L)]
    out.append(gen_prime_asm_fn(L, kb1, kb2, f"cd{L}_y_p", L*16, False, False, False, PSZ))
    out.append(gen_prime_asm_fn(L, kb1, kb2, f"cd{L}_z_p", 16, False, False, False, PSZ))
    out.append(gen_prime_asm_fn(L, kb1, kb2, f"cd{L}_z_m", 16, True, False, False, PSZ))
    out.append(gen_prime_asm_fn(L, kb1, kb2, f"cd{L}_z_mn", 16, True, True, False, PSZ))
    out.append(gen_prime_asm_fn(L, kb1, kb2, f"cd{L}_x_m", PSZ, True, False, False, PSZ))
    out.append(gen_prime_asm_fn(L, kb1, kb2, f"cd{L}_x_mn", PSZ, True, True, False, PSZ, pfn=True))
    out.append(gen_prime_asm_fn(L, kb1, kb2, f"cd{L}_x_mns", PSZ, True, True, True, PSZ))
    if pv:
        RSpv, PSpv = pv
        out.append(gen_prime_asm_fn(L, kb1, kb2, f"cd{L}_w_p", RSpv, False, False, False, PSpv))
        out.append(gen_prime_asm_fn(L, kb1, kb2, f"cd{L}_v_m", PSpv, True, False, False, PSpv))
        out.append(gen_prime_asm_fn(L, kb1, kb2, f"cd{L}_v_mn", PSpv, True, True, False, PSpv))
        out.append(gen_prime_asm_fn(L, kb1, kb2, f"cd{L}_v_mns", PSpv, True, True, True, PSpv))
    return "\n".join(out)
