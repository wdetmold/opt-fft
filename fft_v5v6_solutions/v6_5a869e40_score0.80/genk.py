"""Loop-structured kernels for primes (13/17/23) and 64, to append into gen.py's namespace."""
import numpy as np
import os as _os
PF = _os.environ.get('PF','1')=='1'
from gen import W, W512, W256, W128, Em, clit, cs, LDPI, emit_map_store, emit_map_pair, emit_map_multi, dup_imm

def prime_tables(L):
    h = (L-1)//2
    tc, ts = [], []
    for k in range(1, h+1):
        for j in range(1, h+1):
            m = (j*k) % L
            cv = float(np.cos(2*LDPI*np.longdouble(m)/np.longdouble(L)))
            sv = float(np.sin(2*LDPI*np.longdouble(m)/np.longdouble(L)))
            tc += [cv]*8
            ts += [sv]*8
    lines = [f'static const double TC{L}[{h*h*8}] __attribute__((aligned(64))) = {{']
    lines.append('  ' + ', '.join(clit(v) for v in tc))
    lines.append('};')
    lines.append(f'static const double TS{L}[{h*h*8}] __attribute__((aligned(64))) = {{')
    lines.append('  ' + ', '.join(clit(v) for v in ts))
    lines.append('};')
    return chr(10).join(lines)

def emit_prime_line(w, L, q, Sd, qo, So, mask=None, map_mode=False, cq=None, cSo=None):
    """Emit code lines: DFT-L along elements at q + j*Sd (doubles), store to qo + k*So.
    If map_mode: store via nonlinear map using c at cq + k*cSo."""
    h = (L-1)//2
    em = Em(w)
    one = em.const(1.0); two = em.const(2.0)
    def LDJ(j):
        ptr = f'{q} + {j*Sd}'
        return em.v(w.MASKZ_LD(mask, ptr)) if mask else em.v(w.LD(ptr))
    if PF:
        for _j in range(L):
            em.raw(f'_mm_prefetch((const char*)({q} + {_j*Sd} + {w.nd}), _MM_HINT_T0);')
            if map_mode:
                em.raw(f'_mm_prefetch((const char*)({cq} + {_j*Sd} + {w.nd}), _MM_HINT_T0);')
    x0 = LDJ(0)
    E, Os = [None]*(h+1), [None]*(h+1)
    for j in range(1, h+1):
        a = LDJ(j); b = LDJ(L-j)
        E[j] = em.v(w.ADD(a, b))
        Os[j] = em.v(w.SWAP(em.v(w.SUB(a, b))))
    # X0 = x0 + sum E (pairwise tree)
    terms = [x0] + [E[j] for j in range(1, h+1)]
    while len(terms) > 1:
        nxt = []
        for i in range(0, len(terms)-1, 2):
            nxt.append(em.v(w.ADD(terms[i], terms[i+1])))
        if len(terms) % 2: nxt.append(terms[-1])
        terms = nxt
    X0 = terms[0]
    lines = list(em.lines)
    if map_mode:
        lines.append(f'{w.V} xb[{L}];')
        lines.append(f'xb[0] = {X0};')
    else:
        if mask:
            lines.append(w.MASK_ST(f'{qo} + 0', mask, X0))
        else:
            lines.append(w.ST(f'{qo} + 0', X0))
    # k loop (unrolled x2 for ILP)
    def kblock(em_outer, kexpr, tcp, tsp):
        em2 = Em(w)
        em2._const = dict(em_outer._const)
        C0 = em2.v(w.FMA(E[1], w.LD(tcp), x0))
        S0 = em2.v(w.MUL(Os[1], w.LD(tsp)))
        C1 = None; S1 = None
        for j in range(2, h+1):
            tc = w.LD(f'{tcp} + {8*(j-1)}'); tsx = w.LD(f'{tsp} + {8*(j-1)}')
            if j % 2 == 0:
                C1 = em2.v(w.MUL(E[j], tc)) if C1 is None else em2.v(w.FMA(E[j], tc, C1))
                S1 = em2.v(w.MUL(Os[j], tsx)) if S1 is None else em2.v(w.FMA(Os[j], tsx, S1))
            else:
                C0 = em2.v(w.FMA(E[j], tc, C0))
                S0 = em2.v(w.FMA(Os[j], tsx, S0))
        C = em2.v(w.ADD(C0, C1)) if C1 is not None else C0
        S = em2.v(w.ADD(S0, S1)) if S1 is not None else S0
        XL = em2.v(w.FMADDSUB(C, one, S))
        Xk = em2.v(w.FMS(two, C, XL))
        out = ['  ' + ln for ln in em2.lines]
        if map_mode:
            out.append(f'  xb[{kexpr}] = {Xk}; xb[{L}-({kexpr})] = {XL};')
        else:
            if mask:
                out.append('  ' + w.MASK_ST(f'{qo} + ({kexpr})*{So}', mask, Xk))
                out.append('  ' + w.MASK_ST(f'{qo} + {L*So} - ({kexpr})*{So}', mask, XL))
            else:
                out.append('  ' + w.ST(f'{qo} + ({kexpr})*{So}', Xk))
                out.append('  ' + w.ST(f'{qo} + {L*So} - ({kexpr})*{So}', XL))
        return out
    def flush(seq):
        outl = []
        pend = []
        for item in seq:
            if isinstance(item, tuple) and item[0] == 'PAIR':
                pend.append(item[1:])
            else:
                outl.append(item)
        if pend:
            em9 = Em(w)
            emit_map_multi(em9, pend, mask)
            outl += ['  ' + ln for ln in em9.lines]
        return outl
    lines.append(f'const double* tcp = TC{L}; const double* tsp = TS{L};')
    lines.append(f'for(long k=1;k<={h};k++){{')
    lines += flush(kblock(em, 'k', 'tcp', 'tsp'))
    lines.append(f'  tcp += {8*h}; tsp += {8*h};')
    lines.append('}')
    if map_mode:
        em7 = Em(w)
        def cload(idx):
            return em7.v(w.MASKZ_LD(mask, f'{cq} + {idx*cSo}')) if mask else em7.v(w.LD(f'{cq} + {idx*cSo}'))
        # pairs: (0, h) then (k, L-k) for k=1..? -> pair X0 with xb[h]? keep (k,L-k) pairs + single X0
        plist = []
        for k in range(1, h+1):
            z0 = em7.v(w.ADD(f'xb[{k}]', cload(k)))
            z1 = em7.v(w.ADD(f'xb[{L-k}]', cload(L-k)))
            plist.append((z0, z1, f'{qo} + {k*So}', f'{qo} + {(L-k)*So}'))
        for i in range(0, len(plist), 4):
            emit_map_multi(em7, plist[i:i+4], mask)
        z = em7.v(w.ADD('xb[0]', cload(0)))
        emit_map_store(em7, z, f'{qo} + 0', mask)
        lines += em7.lines
    return lines

def tw64_table():
    vals = []
    for b in range(1, 8):
        for d in range(1, 8):
            c, s = cs(b*d, 64)
            vals += [c]*8 + [s]*8
    lines = [f'static const double TW64[{49*16}] __attribute__((aligned(64))) = {{']
    lines.append('  ' + ', '.join(clit(v) for v in vals))
    lines.append('};')
    return chr(10).join(lines)

def emit_dft8_vars(em, xs):
    """DFT8 on 8 var names; returns 8 outputs (no DFT[] dependency)."""
    w = em.w
    def d4(x0,x1,x2,x3):
        t0 = em.v(w.ADD(x0, x2)); t1 = em.v(w.SUB(x0, x2))
        t2 = em.v(w.ADD(x1, x3)); t3 = em.v(w.SUB(x1, x3))
        X0 = em.v(w.ADD(t0, t2)); X2 = em.v(w.SUB(t0, t2))
        s = em.v(w.SWAP(t3))
        X3 = em.v(w.FMADDSUB(t1, em.const(1.0), s))
        X1 = em.v(w.FMS(em.const(2.0), t1, X3))
        return [X0, X1, X2, X3]
    r2 = float(np.sqrt(np.longdouble(2))/2)
    E = d4(xs[0], xs[2], xs[4], xs[6])
    O = d4(xs[1], xs[3], xs[5], xs[7])
    X = [None]*8
    X[0] = em.v(w.ADD(E[0], O[0])); X[4] = em.v(w.SUB(E[0], O[0]))
    q2 = em.v(w.SWAP(O[2]))
    X[6] = em.v(w.FMADDSUB(E[2], em.const(1.0), q2))
    X[2] = em.v(w.FMS(em.const(2.0), E[2], X[6]))
    sw1 = em.v(w.SWAP(O[1]))
    nsw1 = em.v(w.SUB(em.const(0.0), sw1))
    d = em.v(w.FMADDSUB(O[1], em.const(1.0), nsw1))
    X[1] = em.v(w.FMA(em.const(r2), d, E[1]))
    X[5] = em.v(w.FNMA(em.const(r2), d, E[1]))
    sw3 = em.v(w.SWAP(O[3]))
    g = em.v(w.FMADDSUB(O[3], em.const(1.0), sw3))
    X[3] = em.v(w.FNMA(em.const(r2), g, E[3]))
    X[7] = em.v(w.FMA(em.const(r2), g, E[3]))
    return X

def emit_c64_line(w, L, q, Sd, qo, So, map_mode=False, cq=None, cSo=None, pf2_off=0):
    """64-pt DFT: elements at q + j*Sd doubles; two staged loops via stack tmp."""
    nd = w.nd
    lines = [f'{w.V} tb[64];']
    # stage 1, b=0 peeled (no twiddle)
    em = Em(w)
    if PF:
        for a in range(8):
            em.raw(f'_mm_prefetch((const char*)({q} + {a*8*Sd} + {w.nd}), _MM_HINT_T0);')
            if map_mode:
                em.raw(f'_mm_prefetch((const char*)({cq} + {a*8*Sd}), _MM_HINT_T0);')
            if pf2_off:
                em.raw(f'_mm_prefetch((const char*)({q} + {a*8*Sd} + {pf2_off}), _MM_HINT_T1);')
    xs = [em.v(w.LD(f'{q} + {a*8*Sd}')) for a in range(8)]
    X = emit_dft8_vars(em, xs)
    for d in range(8):
        em.raw(f'tb[{d}] = {X[d]};')
    lines.append('{')
    lines += ['  '+l for l in em.lines]
    lines.append('}')
    # b loop
    lines.append('for(long b=1;b<8;b++){')
    em = Em(w)
    if PF:
        for a in range(8):
            em.raw(f'_mm_prefetch((const char*)({q} + b*{Sd} + {a*8*Sd} + {w.nd}), _MM_HINT_T0);')
            if map_mode:
                em.raw(f'_mm_prefetch((const char*)({cq} + b*{Sd} + {a*8*Sd}), _MM_HINT_T0);')
            if pf2_off:
                em.raw(f'_mm_prefetch((const char*)({q} + b*{Sd} + {a*8*Sd} + {pf2_off}), _MM_HINT_T1);')
    xs = [em.v(w.LD(f'{q} + b*{Sd} + {a*8*Sd}')) for a in range(8)]
    X = emit_dft8_vars(em, xs)
    em.raw(f'const double* twp = TW64 + (b-1)*112;')
    em.raw(f'tb[b*8+0] = {X[0]};')
    for d in range(1, 8):
        s = em.v(w.SWAP(X[d]))
        r = em.v(w.FMADDSUB(X[d], w.LD(f'twp + {16*(d-1)}'), em.v(w.MUL(s, w.LD(f'twp + {16*(d-1)+8}')))))
        em.raw(f'tb[b*8+{d}] = {r};')
    lines += ['  '+l for l in em.lines]
    lines.append('}')
    # stage 2: d loop
    lines.append('for(long d=0;d<8;d++){')
    em = Em(w)
    xs = [em.v(w.LD(f'(const double*)(tb + {b}*8) + d*{nd}')) for b in range(8)]
    Z = emit_dft8_vars(em, xs)
    if map_mode:
        for c0 in range(8):
            em.raw(f'xb[{c0}*8+d] = {Z[c0]};')
    else:
        for c0 in range(8):
            em.raw(w.ST(f'{qo} + d*{So} + {c0*8*So}', Z[c0]))
    lines += ['  '+l for l in em.lines]
    lines.append('}')
    if map_mode:
        lines.insert(0, f'{w.V} xb[64];')
        lines.append('{ const double* restrict xbd = (const double*)xb;')
        lines.append(f'for(long t=0;t<64;t+=8){{')
        em = Em(w)
        plist = []
        for i in range(0, 8, 2):
            cz0 = em.v(w.LD(f'{cq} + (t+{i})*{cSo}'))
            cz1 = em.v(w.LD(f'{cq} + (t+{i+1})*{cSo}'))
            z0 = em.v(w.ADD(em.v(w.LD(f'xbd + (t+{i})*{nd}')), cz0))
            z1 = em.v(w.ADD(em.v(w.LD(f'xbd + (t+{i+1})*{nd}')), cz1))
            plist.append((z0, z1, f'{qo} + (t+{i})*{So}', f'{qo} + (t+{i+1})*{So}'))
        emit_map_multi(em, plist)
        lines += ['  '+l for l in em.lines]
        lines.append('}')
        lines.append('}')
    return lines


from gen import dft4 as _dft4, dft5 as _dft5, dft9 as _dft9
_DFTB = {4: _dft4, 5: _dft5, 9: _dft9}

def emit_pfa_line(w, L, N1, N2, q, Sd, qo, So, map_mode=False, cq=None, cSo=None):
    """PFA L = N1*N2 (coprime): stage A: N1 x DFT_N2; stage B: N2 x DFT_N1.
    Elements at q + j*Sd; outputs at qo + k*So (+map)."""
    nd = w.nd
    crt = {}
    for k in range(L):
        crt[(k % N1, k % N2)] = k
    lines = [f'{w.V} tb[{L}];']
    if map_mode:
        lines.append(f'{w.V} xb[{L}];')
    if PF and Sd > 16:
        em = Em(w)
        for j in range(L):
            em.raw(f'_mm_prefetch((const char*)({q} + {j*Sd} + {w.nd}), _MM_HINT_T0);')
            if map_mode:
                em.raw(f'_mm_prefetch((const char*)({cq} + {j*cSo}), _MM_HINT_T0);')
        lines += em.lines
    for u in range(N1):
        em = Em(w)
        xs = [em.v(w.LD(f'{q} + {((N2*u + N1*v) % L)*Sd}')) for v in range(N2)]
        R = _DFTB[N2](em, xs)
        for k2 in range(N2):
            em.raw(f'tb[{u*N2+k2}] = {R[k2]};')
        lines.append('{')
        lines += ['  '+l for l in em.lines]
        lines.append('}')
    for k2 in range(N2):
        em = Em(w)
        xs = [f'tb[{u*N2+k2}]' for u in range(N1)]
        C = _DFTB[N1](em, xs)
        for k1 in range(N1):
            k = crt[(k1, k2)]
            if map_mode:
                em.raw(f'xb[{k}] = {C[k1]};')
            else:
                em.raw(w.ST(f'{qo} + {k*So}', C[k1]))
        lines.append('{')
        lines += ['  '+l for l in em.lines]
        lines.append('}')
    if map_mode:
        lines.append(f'const double* restrict xbd = (const double*)xb;')
        nloop = L // 8
        if nloop:
            lines.append(f'for(long t=0;t<{8*nloop};t+=8){{')
            em = Em(w)
            plist = []
            for i in range(0, 8, 2):
                cz0 = em.v(w.LD(f'{cq} + (t+{i})*{cSo}'))
                cz1 = em.v(w.LD(f'{cq} + (t+{i+1})*{cSo}'))
                z0 = em.v(w.ADD(em.v(w.LD(f'xbd + (t+{i})*{nd}')), cz0))
                z1 = em.v(w.ADD(em.v(w.LD(f'xbd + (t+{i+1})*{nd}')), cz1))
                plist.append((z0, z1, f'{qo} + (t+{i})*{So}', f'{qo} + (t+{i+1})*{So}'))
            emit_map_multi(em, plist)
            lines += ['  '+l for l in em.lines]
            lines.append('}')
        rem = L - 8*nloop
        if rem:
            base = 8*nloop
            em = Em(w)
            plist = []
            i = 0
            while i + 1 < rem:
                cz0 = em.v(w.LD(f'{cq} + {(base+i)*cSo}'))
                cz1 = em.v(w.LD(f'{cq} + {(base+i+1)*cSo}'))
                z0 = em.v(w.ADD(f'xb[{base+i}]', cz0))
                z1 = em.v(w.ADD(f'xb[{base+i+1}]', cz1))
                plist.append((z0, z1, f'{qo} + {(base+i)*So}', f'{qo} + {(base+i+1)*So}'))
                i += 2
            if plist:
                emit_map_multi(em, plist)
            if i < rem:
                z = em.v(w.ADD(f'xb[{base+i}]', em.v(w.LD(f'{cq} + {(base+i)*cSo}'))))
                emit_map_store(em, z, f'{qo} + {(base+i)*So}')
            lines += em.lines
    return lines


def prime_tables_scalar(L):
    h = (L-1)//2
    tc, ts = [], []
    for k in range(1, h+1):
        for j in range(1, h+1):
            m = (j*k) % L
            tc.append(float(np.cos(2*LDPI*np.longdouble(m)/np.longdouble(L))))
            ts.append(float(np.sin(2*LDPI*np.longdouble(m)/np.longdouble(L))))
    lines = [f'static const double TCs{L}[{h*h}] __attribute__((aligned(64))) = {{']
    lines.append('  ' + ', '.join(clit(v) for v in tc))
    lines.append('};')
    lines.append(f'static const double TSs{L}[{h*h}] __attribute__((aligned(64))) = {{')
    lines.append('  ' + ', '.join(clit(v) for v in ts))
    lines.append('};')
    return chr(10).join(lines)

def emit_prime_block(w, L, q0, gstride, Sd, NG, map_mode=False, cq0=None):
    """Blocked prime DFT over NG line-groups at q0 + g*gstride (doubles), elements at +j*Sd.
    k-outer with coefficient broadcasts in registers, E/O resident in stack."""
    h = (L-1)//2
    nd = w.nd
    G = max(2, min(12, (10*1024) // (64*(2*h+1 + (L if map_mode else 0)))))
    eosz = 2*h+1
    lines = ['{']
    lines.append(f'{w.V} EO[{G}*{eosz}];')
    if map_mode:
        lines.append(f'{w.V} XB[{G}*{L}];')
    lines.append(f'for(long g0=0; g0<{NG}; g0+={G}){{')
    lines.append(f'  long gn = {NG}-g0 < {G} ? {NG}-g0 : {G};')
    # stage 1
    lines.append(f'  for(long g=0; g<gn; g++){{')
    lines.append(f'    double* restrict q = {q0} + (g0+g)*{gstride};')
    lines.append(f'    {w.V}* restrict eo = EO + g*{eosz};')
    em = Em(w)
    if PF:
        for j in range(L):
            em.raw(f'_mm_prefetch((const char*)(q + {G*gstride} + {j*Sd}), _MM_HINT_T0);')
        if map_mode:
            em.raw(f'const double* restrict cpf = {cq0} + (g0+g)*{gstride};')
            for j in range(L):
                em.raw(f'_mm_prefetch((const char*)(cpf + {j*Sd}), _MM_HINT_T0);')
    x0 = em.v(w.LD('q'))
    em.raw(f'eo[0] = {x0};')
    terms = [x0]
    for j in range(1, h+1):
        a = em.v(w.LD(f'q + {j*Sd}'))
        b = em.v(w.LD(f'q + {(L-j)*Sd}'))
        E = em.v(w.ADD(a, b))
        Os = em.v(w.SWAP(em.v(w.SUB(a, b))))
        em.raw(f'eo[{j}] = {E};')
        em.raw(f'eo[{h+j}] = {Os};')
        terms.append(E)
    while len(terms) > 1:
        nxt = []
        for i in range(0, len(terms)-1, 2):
            nxt.append(em.v(w.ADD(terms[i], terms[i+1])))
        if len(terms) % 2: nxt.append(terms[-1])
        terms = nxt
    if map_mode:
        em.raw(f'XB[g*{L}] = {terms[0]};')
    else:
        em.raw(w.ST('q', terms[0]))
    lines += ['    '+ln for ln in em.lines]
    lines.append('  }')
    # stage 2: k outer, coeffs in registers
    lines.append(f'  const double* tcp = TCs{L}; const double* tsp = TSs{L};')
    lines.append(f'  for(long k=1;k<={h};k++){{')
    em = Em(w)
    one = em.const(1.0); two = em.const(2.0)
    TCv = [em.v(w.SET1(f'tcp[{j-1}]')) for j in range(1, h+1)]
    TSv = [em.v(w.SET1(f'tsp[{j-1}]')) for j in range(1, h+1)]
    lines += ['    '+ln for ln in em.lines]
    if not map_mode:
        lines.append(f'    double* restrict qa = {q0} + g0*{gstride} + k*{Sd};')
        lines.append(f'    double* restrict qb = {q0} + g0*{gstride} + {L*Sd} - k*{Sd};')
    lines.append(f'    for(long g=0; g<gn; g++){{')
    lines.append(f'      const {w.V}* restrict eo = EO + g*{eosz};')
    em2 = Em(w)
    em2._const = dict(em._const)
    x0r = em2.v(w.LD('(const double*)eo'))
    C0 = em2.v(w.FMA(f'eo[1]', TCv[0], x0r))
    S0 = em2.v(w.MUL(f'eo[{h+1}]', TSv[0]))
    C1 = None; S1 = None
    for j in range(2, h+1):
        if j % 2 == 0:
            C1 = em2.v(w.MUL(f'eo[{j}]', TCv[j-1])) if C1 is None else em2.v(w.FMA(f'eo[{j}]', TCv[j-1], C1))
            S1 = em2.v(w.MUL(f'eo[{h+j}]', TSv[j-1])) if S1 is None else em2.v(w.FMA(f'eo[{h+j}]', TSv[j-1], S1))
        else:
            C0 = em2.v(w.FMA(f'eo[{j}]', TCv[j-1], C0))
            S0 = em2.v(w.FMA(f'eo[{h+j}]', TSv[j-1], S0))
    C = em2.v(w.ADD(C0, C1)) if C1 is not None else C0
    S = em2.v(w.ADD(S0, S1)) if S1 is not None else S0
    XL = em2.v(w.FMADDSUB(C, one, S))
    Xk = em2.v(w.FMS(two, C, XL))
    if map_mode:
        em2.raw(f'XB[g*{L}+k] = {Xk};')
        em2.raw(f'XB[g*{L}+{L}-k] = {XL};')
    else:
        em2.raw(w.ST(f'qa + g*{gstride}', Xk))
        em2.raw(w.ST(f'qb + g*{gstride}', XL))
    lines += ['      '+ln for ln in em2.lines]
    lines.append('    }')
    lines.append(f'    tcp += {h}; tsp += {h};')
    lines.append('  }')
    if map_mode:
        lines.append(f'  for(long g=0; g<gn; g++){{')
        lines.append(f'    double* restrict q = {q0} + (g0+g)*{gstride};')
        lines.append(f'    const double* restrict cg = {cq0} + (g0+g)*{gstride};')
        lines.append(f'    const double* restrict xv = (const double*)(XB + g*{L});')
        em3 = Em(w)
        plist = []
        for k in range(1, h+1):
            z0 = em3.v(w.ADD(em3.v(w.LD(f'xv + {k*nd}')), em3.v(w.LD(f'cg + {k*Sd}'))))
            z1 = em3.v(w.ADD(em3.v(w.LD(f'xv + {(L-k)*nd}')), em3.v(w.LD(f'cg + {(L-k)*Sd}'))))
            plist.append((z0, z1, f'q + {k*Sd}', f'q + {(L-k)*Sd}'))
        for i in range(0, len(plist), 4):
            emit_map_multi(em3, plist[i:i+4])
        z = em3.v(w.ADD(em3.v(w.LD('xv')), em3.v(w.LD('cg'))))
        emit_map_store(em3, z, 'q')
        lines += ['    '+ln for ln in em3.lines]
        lines.append('  }')
    lines.append('}')
    lines.append('}')
    return lines
