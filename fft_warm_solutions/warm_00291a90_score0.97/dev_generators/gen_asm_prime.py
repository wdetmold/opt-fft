import numpy as np
from genlib import hexd
from gen_a import prime_tables, fold_idx
from gen_asm import A

def table_c(L):
    h, cosv, sinv = prime_tables(L)
    vals = cosv + sinv + [1e-300, 1.0, 0.5]
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
    return (2*h + {'TINY':0,'ONE':1,'HALF':2}[kind])*8

def emit_dft_core(a, L, kb1, kb2, src, ses, dst, des):
    """plain DFT src->dst (no map). Writes X0..X_{L-1}."""
    h, cosv, sinv = prime_tables(L)
    kblocks1 = [list(range(s, min(s+kb1, h+1))) for s in range(1, h+1, kb1)]
    kblocks2 = [list(range(s, min(s+kb2, h+1))) for s in range(1, h+1, kb2)]
    need_eo = len(kblocks1) > 1 or len(kblocks2) > 1
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

def emit_map_phase(a, L, src, dsts, gmax=4):
    """interleaved map over points 0..L-1. src: (base, stride_doubles). dsts: list of (base, stride)."""
    sbase, ss = src
    mc = {n: a.bcast('tab', toff(L, n)) for n in ('TINY','ONE','HALF')}
    pts = list(range(L))
    groups = []
    i = 0
    while i < len(pts):
        g = pts[i:i+gmax]
        # avoid tiny tail group: merge if <=2
        if len(pts) - i - len(g) in (1, 2) and len(g) == gmax:
            pass
        groups.append(g); i += len(g)
    if len(groups) >= 2 and len(groups[-1]) <= 2 and len(groups[-2]) + len(groups[-1]) <= gmax:
        groups[-2] += groups[-1]; groups.pop()
    for g in groups:
        st = {}
        # stage pipeline over points in group
        for q in g:
            zr = a.ld(sbase, q*ss*8); cr = a.ld('pc', q*128)
            zr2 = a.add(zr, cr, kill=(zr, cr))
            zi = a.ld(sbase, q*ss*8+64); ci = a.ld('pc', q*128+64)
            zi2 = a.add(zi, ci, kill=(zi, ci))
            st[q] = dict(zr=zr2, zi=zi2)
        for q in g:
            s = st[q]
            m = a.mul(s['zr'], s['zr'])
            a.fma(m, s['zi'], s['zi'])
            a.ins(f"vaddpd %%zmm{mc['TINY']}, %%zmm{m}, %%zmm{m}")
            s['m'] = m
        for q in g:
            st[q]['r0'] = a.rsqrt(st[q]['m'])
        for q in g:
            s = st[q]
            s['t'] = a.mul(s['m'], s['r0'])
            s['hr'] = a.mul(s['r0'], mc['HALF'])
        for q in g:
            s = st[q]
            eh = a.mov(mc['HALF'])
            a.fnma(eh, s['t'], s['hr'])
            a.rel(s['t']); a.rel(s['hr'])
            s['eh'] = eh
        for q in g:
            s = st[q]
            r1 = a.mov(s['r0'])
            a.fma(r1, s['r0'], s['eh'])
            a.rel(s['r0']); a.rel(s['eh'])
            s['r1'] = r1
        for q in g:
            s = st[q]
            s['mg0'] = a.mul(s['m'], s['r1'])
            s['hr1'] = a.mul(s['r1'], mc['HALF'], kill=(s['r1'],))
        for q in g:
            s = st[q]
            e2 = a.mov(s['m'])
            a.fnma(e2, s['mg0'], s['mg0'])
            a.rel(s['m'])
            a.fma(s['mg0'], e2, s['hr1'])
            a.rel(e2); a.rel(s['hr1'])
            s['u'] = a.add(s['mg0'], mc['ONE'], kill=(s['mg0'],))
        for q in g:
            st[q]['w0'] = a.rcp(st[q]['u'])
        for q in g:
            s = st[q]
            e3 = a.mov(mc['ONE'])
            a.fnma(e3, s['u'], s['w0'])
            a.rel(s['u'])
            s['e3'] = e3
        for q in g:
            s = st[q]
            w1 = a.mov(s['w0'])
            a.fma(w1, s['w0'], s['e3'])
            a.rel(s['w0'])
            ee = a.mul(s['e3'], s['e3'], kill=(s['e3'],))
            s['w1'] = w1; s['ee'] = ee
        for q in g:
            s = st[q]
            w2 = a.mov(s['w1'])
            a.fma(w2, s['w1'], s['ee'])
            a.rel(s['w1']); a.rel(s['ee'])
            xr = a.mul(s['zr'], w2, kill=(s['zr'],))
            xi = a.mul(s['zi'], w2, kill=(s['zi'],))
            a.rel(w2)
            for (db, ds) in dsts:
                a.st(db, q*ds*8, xr); a.st(db, q*ds*8+64, xi)
            a.rel(xr); a.rel(xi)
    for r in mc.values(): a.rel(r)

def gen_prime_asm_fn(L, kb1, kb2, name, ses, mapmode, nextmode, snap, PSZ):
    a = A()
    if not mapmode:
        emit_dft_core(a, L, kb1, kb2, 'px', ses, 'px', ses)
    else:
        emit_dft_core(a, L, kb1, kb2, 'px', ses, 'XS', 16)
        assert not a.live
        if nextmode:
            dsts = [('M', 16)] + ([('ps', PSZ)] if snap else [])
            emit_map_phase(a, L, ('XS', 16), dsts)
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
    if snap:
        args.append("double* ps"); ops.append('[ps]"r"(ps)')
    clob = ", ".join(f'"zmm{i}"' for i in range(32)) + ', "memory"'
    return f"""static void __attribute__((noinline)) {name}({", ".join(args)}){{
    __asm__ volatile("{body}"
    : : {", ".join(ops)}
    : {clob});
}}
"""

def gen_prime_asm(L, kb1, kb2, PS):
    PSZ = PS*16
    out = [table_c(L)]
    out.append(gen_prime_asm_fn(L, kb1, kb2, f"cd{L}_y_p", L*16, False, False, False, PSZ))
    out.append(gen_prime_asm_fn(L, kb1, kb2, f"cd{L}_z_p", 16, False, False, False, PSZ))
    out.append(gen_prime_asm_fn(L, kb1, kb2, f"cd{L}_z_m", 16, True, False, False, PSZ))
    out.append(gen_prime_asm_fn(L, kb1, kb2, f"cd{L}_z_mn", 16, True, True, False, PSZ))
    out.append(gen_prime_asm_fn(L, kb1, kb2, f"cd{L}_x_m", PSZ, True, False, False, PSZ))
    out.append(gen_prime_asm_fn(L, kb1, kb2, f"cd{L}_x_mn", PSZ, True, True, False, PSZ))
    out.append(gen_prime_asm_fn(L, kb1, kb2, f"cd{L}_x_mns", PSZ, True, True, True, PSZ))
    return "\n".join(out)
