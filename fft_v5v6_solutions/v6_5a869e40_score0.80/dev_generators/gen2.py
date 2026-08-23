#!/usr/bin/env python3
"""Top-level generator v2: fused per-plane pass12, kernel loops for primes/64."""
import numpy as np
from gen import (W, W512, W256, W128, Em, clit, cs, LDPI, DFT,
                 emit_map_store, emit_map_pair, emit_map_multi, dup_imm)
import genk

SIZES = (6, 8, 13, 17, 23, 36, 45, 64)
PRIMES = (13, 17, 23)

def col_groups(L):
    out = []; k = 0
    while L - k >= 4:
        out.append((k, 4, W512, None)); k += 4
    r = L - k
    if r == 3: out.append((k, 3, W512, '0x3F'))
    elif r == 2: out.append((k, 2, W256, None))
    elif r == 1: out.append((k, 1, W128, None))
    return out

# ---------- generic (small-size) line pass body on one plane/line-set ----------
def lines_generic_axis(L, estride, p, with_map, cp=None):
    """Body (list of lines): transform along axis with element stride estride (complex),
    lanes along contiguous a2, for all L columns of the line-set starting at pointer p."""
    nfull = L // 4
    rem = L % 4
    out = []
    def group_block(w, mask):
        em = Em(w)
        xs = []
        for j in range(L):
            off = 2*(j*estride)
            xs.append(em.v(w.MASKZ_LD(mask, f'q + {off}')) if mask else em.v(w.LD(f'q + {off}')))
        X = DFT[L](em, xs)
        if with_map:
            def ldc(k):
                off = 2*(k*estride)
                return em.v(w.MASKZ_LD(mask, f'cq + {off}')) if mask else em.v(w.LD(f'cq + {off}'))
            pairs = []
            k = 0
            while k + 1 < L:
                z0 = em.v(w.ADD(X[k], ldc(k)))
                z1 = em.v(w.ADD(X[k+1], ldc(k+1)))
                pairs.append((z0, z1, f'q + {2*(k*estride)}', f'q + {2*((k+1)*estride)}'))
                k += 2
            for i in range(0, len(pairs), 4):
                emit_map_multi(em, pairs[i:i+4], mask)
            if k < L:
                z = em.v(w.ADD(X[k], ldc(k)))
                emit_map_store(em, z, f'q + {2*(k*estride)}', mask)
        else:
            for k in range(L):
                off = 2*(k*estride)
                out_ln = w.MASK_ST(f'q + {off}', mask, X[k]) if mask else w.ST(f'q + {off}', X[k])
                em.raw(out_ln)
        return em.lines
    if nfull > 0:
        out.append(f'for(long kk=0;kk<{8*nfull};kk+=8){{')
        out.append(f'  double* restrict q = {p} + kk;')
        if with_map: out.append(f'  const double* restrict cq = {cp} + kk;')
        for ln in group_block(W512, None):
            out.append('  ' + ln)
        out.append('}')
    if rem:
        k0 = 8*nfull
        w, mask = {3:(W512,'0x3F'), 2:(W256,None), 1:(W128,None)}[rem]
        out.append('{')
        out.append(f'  double* restrict q = {p} + {k0};')
        if with_map: out.append(f'  const double* restrict cq = {cp} + {k0};')
        for ln in group_block(w, mask):
            out.append('  ' + ln)
        out.append('}')
    return out

# ---------- prime kernel line pass body ----------
def lines_prime_axis(L, estride, p, with_map, cp=None):
    nfull = L // 4
    rem = L % 4
    Sd = 2*estride
    out = []
    if nfull > 0:
        out.append(f'for(long kk=0;kk<{8*nfull};kk+=8){{')
        out.append(f'  double* restrict q = {p} + kk;')
        if with_map: out.append(f'  const double* restrict cq = {cp} + kk;')
        body = genk.emit_prime_line(W512, L, 'q', Sd, 'q', Sd, None, with_map, 'cq' if with_map else None, Sd)
        out += ['  '+ln for ln in body]
        out.append('}')
    if rem:
        k0 = 8*nfull
        w, mask = {3:(W512,'0x3F'), 2:(W256,None), 1:(W128,None)}[rem]
        out.append('{')
        out.append(f'  double* restrict q = {p} + {k0};')
        if with_map: out.append(f'  const double* restrict cq = {cp} + {k0};')
        body = genk.emit_prime_line(w, L, 'q', Sd, 'q', Sd, mask, with_map, 'cq' if with_map else None, Sd)
        out += ['  '+ln for ln in body]
        out.append('}')
    return out

# ---------- 64 kernel line pass body ----------
def lines_c64_axis(estride, p, with_map, cp=None):
    Sd = 2*estride
    pf2 = 0
    out = [f'for(long kk=0;kk<128;kk+=8){{', f'  double* restrict q = {p} + kk;']
    if with_map: out.append(f'  const double* restrict cq = {cp} + kk;')
    body = genk.emit_c64_line(W512, 64, 'q', Sd, 'q', Sd, with_map, 'cq' if with_map else None, Sd, pf2)
    out += ['  '+ln for ln in body]
    out.append('}')
    return out

PFA_SIZES = {36: (4, 9), 45: (9, 5)}

def lines_pfa_axis(L, estride, p, with_map, cp=None):
    N1, N2 = PFA_SIZES[L]
    nfull = L // 4
    rem = L % 4
    Sd = 2*estride
    out = []
    if nfull > 0:
        out.append(f'for(long kk=0;kk<{8*nfull};kk+=8){{')
        out.append(f'  double* restrict q = {p} + kk;')
        if with_map: out.append(f'  const double* restrict cq = {cp} + kk;')
        body = genk.emit_pfa_line(W512, L, N1, N2, 'q', Sd, 'q', Sd, with_map, 'cq' if with_map else None, Sd)
        out += ['  '+ln for ln in body]
        out.append('}')
    if rem:
        k0 = 8*nfull
        w = {3: W512, 2: W256, 1: W128}[rem]
        assert rem == 1, "PFA tail only r==1 supported"
        out.append('{')
        out.append(f'  double* restrict q = {p} + {k0};')
        if with_map: out.append(f'  const double* restrict cq = {cp} + {k0};')
        body = genk.emit_pfa_line(w, L, N1, N2, 'q', Sd, 'q', Sd, with_map, 'cq' if with_map else None, Sd)
        out += ['  '+ln for ln in body]
        out.append('}')
    return out

def lines_axis(L, estride, p, with_map, cp=None):
    if L in PRIMES:
        return lines_prime_axis(L, estride, p, with_map, cp)
    if L == 64:
        return lines_c64_axis(estride, p, with_map, cp)
    return lines_generic_axis(L, estride, p, with_map, cp)

# ---------- pass2 body on one plane (transform along a2, lanes across a1) ----------
def tp_imm(i):  # shuffle immediates
    return ['0x88','0xDD'][i]

def transpose4(em, t):
    x0 = em.v(f'_mm512_shuffle_f64x2({t[0]}, {t[1]}, 0x88)')
    x1 = em.v(f'_mm512_shuffle_f64x2({t[0]}, {t[1]}, 0xDD)')
    x2 = em.v(f'_mm512_shuffle_f64x2({t[2]}, {t[3]}, 0x88)')
    x3 = em.v(f'_mm512_shuffle_f64x2({t[2]}, {t[3]}, 0xDD)')
    return [em.v(f'_mm512_shuffle_f64x2({x0}, {x2}, 0x88)'),
            em.v(f'_mm512_shuffle_f64x2({x1}, {x3}, 0x88)'),
            em.v(f'_mm512_shuffle_f64x2({x0}, {x2}, 0xDD)'),
            em.v(f'_mm512_shuffle_f64x2({x1}, {x3}, 0xDD)')]

def lines_pass2_plane(L, p, with_map=False, cp=None):
    """Transform along a2 for all rows of plane at pointer p."""
    ct = L % 4; ncb = L // 4
    ngfull = L // 4; rtail = L % 4
    kernelized = (L in PRIMES) or (L == 64)
    out = []
    def zmm_group_lines(nrows, qname, cqname=None):
        em = Em(W512); w = W512
        v = [None]*L
        rows = [f'{qname} + {2*t*L}' for t in range(nrows)]
        lb_mode = kernelized
        if lb_mode:
            em.raw(f'__m512d lb[{L}];')
        for cb in range(ncb + (1 if ct else 0)):
            k0 = cb*4
            tail = (cb == ncb)
            mask = {1:'0x03',2:'0x0F',3:'0x3F'}.get(ct) if tail else None
            t = []
            for tr in range(4):
                if tr < nrows:
                    t.append(em.v(w.MASKZ_LD(mask, f'{rows[tr]} + {2*k0}')) if tail else em.v(w.LD(f'{rows[tr]} + {2*k0}')))
                else:
                    t.append(em.v(w.ZERO()))
            o = transpose4(em, t)
            lim = ct if tail else 4
            for kk2 in range(lim):
                if lb_mode:
                    em.raw(f'lb[{k0+kk2}] = {o[kk2]};')
                else:
                    v[k0+kk2] = o[kk2]
        lines = list(em.lines)
        def emit_ctiles(em2):
            cv = [None]*L
            crows = [f'{cqname} + {2*t*L}' for t in range(nrows)]
            for cb2 in range(ncb + (1 if ct else 0)):
                k0 = cb2*4
                tail = (cb2 == ncb)
                mask = {1:'0x03',2:'0x0F',3:'0x3F'}.get(ct) if tail else None
                t = []
                for tr in range(4):
                    if tr < nrows:
                        t.append(em2.v(W512.MASKZ_LD(mask, f'{crows[tr]} + {2*k0}')) if tail else em2.v(W512.LD(f'{crows[tr]} + {2*k0}')))
                    else:
                        t.append(em2.v(W512.ZERO()))
                o = transpose4(em2, t)
                lim = ct if tail else 4
                for kk2 in range(lim):
                    cv[k0+kk2] = o[kk2]
            return cv
        if lb_mode:
            lines.append('double* restrict lbd = (double*)lb;')
            if with_map:
                lines.append(f'__m512d cbv[{L}];')
                em2 = Em(W512)
                cv = emit_ctiles(em2)
                for k in range(L):
                    em2.raw(f'cbv[{k}] = {cv[k]};')
                lines += em2.lines
                lines.append('const double* restrict cbd = (const double*)cbv;')
                if L == 64:
                    lines += genk.emit_c64_line(W512, 64, 'lbd', 8, 'lbd', 8, True, 'cbd', 8)
                elif L in (36, 45):
                    lines += genk.emit_pfa_line(W512, L, *PFA_SIZES[L], 'lbd', 8, 'lbd', 8, True, 'cbd', 8)
                else:
                    lines += genk.emit_prime_line(W512, L, 'lbd', 8, 'lbd', 8, None, True, 'cbd', 8)
            else:
                if L == 64:
                    lines += genk.emit_c64_line(W512, 64, 'lbd', 8, 'lbd', 8, False)
                elif L in (36, 45):
                    lines += genk.emit_pfa_line(W512, L, *PFA_SIZES[L], 'lbd', 8, 'lbd', 8, False)
                else:
                    lines += genk.emit_prime_line(W512, L, 'lbd', 8, 'lbd', 8, None, False)
            em = Em(W512)
            getv = lambda k: f'lb[{k}]'
        else:
            X = DFT[L](em, v)
            lines = list(em.lines)
            em = Em(W512)
            if with_map:
                cv = emit_ctiles(em)
                from gen import emit_map_multi as _emm
                pairs = []
                newX = [None]*L
                for k in range(0, L-1, 2):
                    z0 = em.v(W512.ADD(X[k], cv[k]))
                    z1 = em.v(W512.ADD(X[k+1], cv[k+1]))
                    # map into fresh vars via emit to lbuf-style: use temp array
                    pairs.append((z0, z1, k, k+1))
                # use a stack buffer for mapped outputs
                em.raw(f'__m512d mb[{L}];')
                plist = [(z0, z1, f'(double*)(mb+{k0})', f'(double*)(mb+{k1})') for (z0,z1,k0,k1) in pairs]
                for i in range(0, len(plist), 4):
                    _emm(em, plist[i:i+4])
                if L % 2:
                    zl = em.v(W512.ADD(X[L-1], cv[L-1]))
                    emit_map_store(em, zl, f'(double*)(mb+{L-1})')
                getv = lambda k: f'mb[{k}]'
            else:
                getv = lambda k: X[k]
        for cb in range(ncb + (1 if ct else 0)):
            k0 = cb*4
            tail = (cb == ncb)
            mask = {1:'0x03',2:'0x0F',3:'0x3F'}.get(ct) if tail else None
            lim = ct if tail else 4
            src4 = [getv(k0+kk2) if kk2 < lim else getv(k0) for kk2 in range(4)]
            o = transpose4(em, src4)
            for tr in range(nrows):
                if tail:
                    em.raw(w.MASK_ST(f'{rows[tr]} + {2*k0}', mask, o[tr]))
                else:
                    em.raw(w.ST(f'{rows[tr]} + {2*k0}', o[tr]))
        lines += em.lines
        return lines
    def ymm_group_lines(qname, cqname=None):
        em = Em(W256); w = W256
        v = [None]*L
        rows = [f'{qname} + {2*t*L}' for t in range(2)]
        nc2 = L//2; ct2 = L%2
        for cb in range(nc2):
            k0 = cb*2
            t0 = em.v(w.LD(f'{rows[0]} + {2*k0}')); t1 = em.v(w.LD(f'{rows[1]} + {2*k0}'))
            v[k0] = em.v(f'_mm256_permute2f128_pd({t0}, {t1}, 0x20)')
            v[k0+1] = em.v(f'_mm256_permute2f128_pd({t0}, {t1}, 0x31)')
        if ct2:
            k0 = nc2*2
            a = em.v(f'_mm_loadu_pd({rows[0]} + {2*k0})'); b = em.v(f'_mm_loadu_pd({rows[1]} + {2*k0})')
            v[k0] = em.v(f'_mm256_insertf128_pd(_mm256_castpd128_pd256({a}), {b}, 1)')
        X = DFT[L](em, v)
        if with_map:
            from gen import emit_map_multi as _emm
            cv = [None]*L
            crows = [f'{cqname} + {2*t*L}' for t in range(2)]
            for cb in range(nc2):
                k0 = cb*2
                t0 = em.v(w.LD(f'{crows[0]} + {2*k0}')); t1 = em.v(w.LD(f'{crows[1]} + {2*k0}'))
                cv[k0] = em.v(f'_mm256_permute2f128_pd({t0}, {t1}, 0x20)')
                cv[k0+1] = em.v(f'_mm256_permute2f128_pd({t0}, {t1}, 0x31)')
            if ct2:
                k0 = nc2*2
                a = em.v(f'_mm_loadu_pd({crows[0]} + {2*k0})'); b = em.v(f'_mm_loadu_pd({crows[1]} + {2*k0})')
                cv[k0] = em.v(f'_mm256_insertf128_pd(_mm256_castpd128_pd256({a}), {b}, 1)')
            em.raw(f'__m256d mb[{L}];')
            plist = []
            for k in range(0, L-1, 2):
                z0 = em.v(w.ADD(X[k], cv[k]))
                z1 = em.v(w.ADD(X[k+1], cv[k+1]))
                plist.append((z0, z1, f'(double*)(mb+{k})', f'(double*)(mb+{k+1})'))
            for i in range(0, len(plist), 4):
                _emm(em, plist[i:i+4])
            if L % 2:
                zl = em.v(w.ADD(X[L-1], cv[L-1]))
                emit_map_store(em, zl, f'(double*)(mb+{L-1})')
            X = [f'mb[{k}]' for k in range(L)]
        for cb in range(nc2):
            k0 = cb*2
            o0 = em.v(f'_mm256_permute2f128_pd({X[k0]}, {X[k0+1]}, 0x20)')
            o1 = em.v(f'_mm256_permute2f128_pd({X[k0]}, {X[k0+1]}, 0x31)')
            em.raw(w.ST(f'{rows[0]} + {2*k0}', o0)); em.raw(w.ST(f'{rows[1]} + {2*k0}', o1))
        if ct2:
            k0 = nc2*2
            em.raw(f'_mm_storeu_pd({rows[0]} + {2*k0}, _mm256_castpd256_pd128({X[k0]}));')
            em.raw(f'_mm_storeu_pd({rows[1]} + {2*k0}, _mm256_extractf128_pd({X[k0]}, 1));')
        return em.lines
    def xmm_group_lines(qname, cqname=None):
        if L in (36, 45):
            return genk.emit_pfa_line(W128, L, *PFA_SIZES[L], qname, 2, qname, 2, with_map, cqname, 2)
        if kernelized:
            return genk.emit_prime_line(W128, L, qname, 2, qname, 2, None, with_map, cqname, 2)
        em = Em(W128); w = W128
        v = [em.v(w.LD(f'{qname} + {2*k}')) for k in range(L)]
        X = DFT[L](em, v)
        if with_map:
            from gen import emit_map_multi as _emm
            plist = []
            for k in range(0, L-1, 2):
                z0 = em.v(w.ADD(X[k], em.v(w.LD(f'{cqname} + {2*k}'))))
                z1 = em.v(w.ADD(X[k+1], em.v(w.LD(f'{cqname} + {2*(k+1)}'))))
                plist.append((z0, z1, f'{qname} + {2*k}', f'{qname} + {2*(k+1)}'))
            for i in range(0, len(plist), 4):
                _emm(em, plist[i:i+4])
            if L % 2:
                zl = em.v(w.ADD(X[L-1], em.v(w.LD(f'{cqname} + {2*(L-1)}'))))
                emit_map_store(em, zl, f'{qname} + {2*(L-1)}')
            return em.lines
        for k in range(L):
            em.raw(w.ST(f'{qname} + {2*k}', X[k]))
        return em.lines
    if ngfull > 0:
        out.append(f'for(long g=0;g<{ngfull};g++){{')
        out.append(f'  double* restrict q2 = {p} + g*{8*L};')
        if with_map: out.append(f'  const double* restrict cq2 = {cp} + g*{8*L};')
        for ln in zmm_group_lines(4, 'q2', 'cq2'):
            out.append('  ' + ln)
        out.append('}')
    if rtail:
        out.append('{')
        out.append(f'  double* restrict q2 = {p} + {8*ngfull*L};')
        if with_map: out.append(f'  const double* restrict cq2 = {cp} + {8*ngfull*L};')
        if rtail == 3:
            for ln in zmm_group_lines(3, 'q2', 'cq2'): out.append('  ' + ln)
        elif rtail == 2:
            for ln in ymm_group_lines('q2', 'cq2'): out.append('  ' + ln)
        else:
            for ln in xmm_group_lines('q2', 'cq2'): out.append('  ' + ln)
        out.append('}')
    return out

# ---------- interleaved-by-4 volumes mode ----------
def lines_line_i4(L, Sd, qexpr, with_map, cqexpr=None, W=W512):
    """Emit one line transform at runtime base q (elements at q + j*Sd doubles, lanes=volumes)."""
    if L in PRIMES:
        return genk.emit_prime_line(W, L, qexpr, Sd, qexpr, Sd, None, with_map, cqexpr, Sd)
    if L in PFA_SIZES:
        return genk.emit_pfa_line(W, L, *PFA_SIZES[L], qexpr, Sd, qexpr, Sd, with_map, cqexpr, Sd)
    if L == 64:
        return genk.emit_c64_line(W, 64, qexpr, Sd, qexpr, Sd, with_map, cqexpr, Sd)
    # generic codelet straight-line
    em = Em(W); w = W
    if Sd > 16:
        for j in range(L):
            em.raw(f'_mm_prefetch((const char*)({qexpr} + {j*Sd} + 8), _MM_HINT_T0);')
            if with_map:
                em.raw(f'_mm_prefetch((const char*)({cqexpr} + {j*Sd}), _MM_HINT_T0);')
    xs = [em.v(w.LD(f'{qexpr} + {j*Sd}')) for j in range(L)]
    X = DFT[L](em, xs)
    if with_map:
        plist = []
        k = 0
        while k + 1 < L:
            z0 = em.v(w.ADD(X[k], em.v(w.LD(f'{cqexpr} + {k*Sd}'))))
            z1 = em.v(w.ADD(X[k+1], em.v(w.LD(f'{cqexpr} + {(k+1)*Sd}'))))
            plist.append((z0, z1, f'{qexpr} + {k*Sd}', f'{qexpr} + {(k+1)*Sd}'))
            k += 2
        for t in range(0, len(plist), 4):
            emit_map_multi(em, plist[t:t+4])
        if k < L:
            z = em.v(w.ADD(X[k], em.v(w.LD(f'{cqexpr} + {k*Sd}'))))
            emit_map_store(em, z, f'{qexpr} + {k*Sd}')
    else:
        for k in range(L):
            em.raw(w.ST(f'{qexpr} + {k*Sd}', X[k]))
    return em.lines

def gen_size_iN(L, NV):
    W = {4: W512, 2: W256}[NV]
    sfx = f'i{NV}'
    parts = []
    A0F = L in (17, 23, 45)
    if A0F:
        # order: a0 plain (flat), then per-plane a1 plain + a2 with map (contiguous c)
        body = [f'static void pass12_{sfx}_{L}(double* restrict x){{',     # now actually a0 pass
                f'  for(long t=0;t<{2*NV*L*L};t+={2*NV}){{',
                f'    double* restrict q = x + t;']
        body += ['    '+ln for ln in lines_line_i4(L, 2*NV*L*L, 'q', False, W=W)]
        body += ['  }', '}']
        parts.append(chr(10).join(body))
        body = [f'static void pass3m_{sfx}_{L}(double* restrict x, const double* restrict cc){{',  # a1 + a2(map)
                f'  for(long u=0;u<{L};u++){{',
                f'    double* restrict pp = x + u*{2*NV*L*L};',
                f'    const double* restrict cpp = cc + u*{2*NV*L*L};',
                f'    for(long kk=0;kk<{2*NV*L};kk+={2*NV}){{',
                f'      double* restrict q = pp + kk;']
        body += ['      '+ln for ln in lines_line_i4(L, 2*NV*L, 'q', False, W=W)]
        body += ['    }',
                f'    for(long j=0;j<{2*NV*L*L};j+={2*NV*L}){{',
                f'      double* restrict q = pp + j;',
                f'      const double* restrict cq = cpp + j;']
        body += ['      '+ln for ln in lines_line_i4(L, 2*NV, 'q', True, 'cq', W=W)]
        body += ['    }', '  }', '}']
        parts.append(chr(10).join(body))
    else:
        body = [f'static void pass12_{sfx}_{L}(double* restrict x){{',
                f'  for(long u=0;u<{L};u++){{',
                f'    double* restrict pp = x + u*{2*NV*L*L};',
                f'    for(long kk=0;kk<{2*NV*L};kk+={2*NV}){{',
                f'      double* restrict q = pp + kk;']
        body += ['      '+ln for ln in lines_line_i4(L, 2*NV*L, 'q', False, W=W)]
        body += ['    }',
                f'    for(long j=0;j<{2*NV*L*L};j+={2*NV*L}){{',
                f'      double* restrict q = pp + j;']
        body += ['      '+ln for ln in lines_line_i4(L, 2*NV, 'q', False, W=W)]
        body += ['    }', '  }', '}']
        parts.append(chr(10).join(body))
        body = [f'static void pass3m_{sfx}_{L}(double* restrict x, const double* restrict cc){{',
                f'  for(long t=0;t<{2*NV*L*L};t+={2*NV}){{',
                f'    double* restrict q = x + t;',
                f'    const double* restrict cq = cc + t;']
        body += ['    '+ln for ln in lines_line_i4(L, 2*NV*L*L, 'q', True, 'cq', W=W)]
        body += ['  }', '}']
        parts.append(chr(10).join(body))
    V = 2*L*L*L
    if NV == 4:
        ildl = f"""
static void il4_{L}(double* restrict dst, const double* restrict src){{
  for(long idx=0;idx<{L*L*L};idx++){{
    __m512d v = _mm512_castpd128_pd512(_mm_loadu_pd(src + idx*2));
    v = _mm512_insertf64x2(v, _mm_loadu_pd(src + {V} + idx*2), 1);
    v = _mm512_insertf64x2(v, _mm_loadu_pd(src + {2*V} + idx*2), 2);
    v = _mm512_insertf64x2(v, _mm_loadu_pd(src + {3*V} + idx*2), 3);
    _mm512_store_pd(dst + idx*8, v);
  }}
}}
static void dl4_{L}(double* restrict dst, const double* restrict src){{
  for(long idx=0;idx<{L*L*L};idx++){{
    __m512d v = _mm512_load_pd(src + idx*8);
    _mm_storeu_pd(dst + idx*2, _mm512_castpd512_pd128(v));
    _mm_storeu_pd(dst + {V} + idx*2, _mm512_extractf64x2_pd(v, 1));
    _mm_storeu_pd(dst + {2*V} + idx*2, _mm512_extractf64x2_pd(v, 2));
    _mm_storeu_pd(dst + {3*V} + idx*2, _mm512_extractf64x2_pd(v, 3));
  }}
}}"""
    else:
        ildl = f"""
static void il2_{L}(double* restrict dst, const double* restrict src){{
  for(long idx=0;idx<{L*L*L};idx++){{
    __m256d v = _mm256_castpd128_pd256(_mm_loadu_pd(src + idx*2));
    v = _mm256_insertf128_pd(v, _mm_loadu_pd(src + {V} + idx*2), 1);
    _mm256_store_pd(dst + idx*4, v);
  }}
}}
static void dl2_{L}(double* restrict dst, const double* restrict src){{
  for(long idx=0;idx<{L*L*L};idx++){{
    __m256d v = _mm256_load_pd(src + idx*4);
    _mm_storeu_pd(dst + idx*2, _mm256_castpd256_pd128(v));
    _mm_storeu_pd(dst + {V} + idx*2, _mm256_extractf128_pd(v, 1));
  }}
}}"""
    parts.append(ildl)
    parts.append(f"""
void run{NV}_{L}(long G, long m, const double* restrict x0, const double* restrict c,
              double* restrict out1, double* restrict outm){{
  for(long g=0;g<G;g++){{
    il{NV}_{L}(STATE, x0 + g*{NV*V});
    il{NV}_{L}(CBUF, c + g*{NV*V});
    for(long s=0;s<m;s++){{
      pass12_{sfx}_{L}(STATE); pass3m_{sfx}_{L}(STATE, CBUF);
      if(s==0) dl{NV}_{L}(out1 + g*{NV*V}, STATE);
    }}
    dl{NV}_{L}(outm + g*{NV*V}, STATE);
  }}
}}""")
    return (chr(10)+chr(10)).join(parts)

# ---------- per-size assembly ----------
def gen_size(L):
    parts = []
    # fused pass12: per-plane a1 then a2 (plain)
    body = [f'void pass12_{L}(double* restrict x){{', f'  for(long u=0;u<{L};u++){{',
            f'    double* restrict pp = x + u*{2*L*L};']
    body += ['    '+ln for ln in lines_axis(L, L, 'pp', False)]
    body += ['    '+ln for ln in lines_pass2_plane(L, 'pp')]
    body += ['  }', '}']
    parts.append(chr(10).join(body))
    import os as _os
    if _os.environ.get('SPLIT12'):
        body = [f'void p1only_{L}(double* restrict x){{', f'  for(long u=0;u<{L};u++){{',
                f'    double* restrict pp = x + u*{2*L*L};']
        body += ['    '+ln for ln in lines_axis(L, L, 'pp', False)]
        body += ['  }', '}']
        parts.append(chr(10).join(body))
        body = [f'void p2only_{L}(double* restrict x){{', f'  for(long u=0;u<{L};u++){{',
                f'    double* restrict pp = x + u*{2*L*L};']
        body += ['    '+ln for ln in lines_pass2_plane(L, 'pp')]
        body += ['  }', '}']
        parts.append(chr(10).join(body))
    # pass3 with map / plain
    for nm, wm in (('pass3m', True), ('pass3p', False)):
        body = [f'void {nm}_{L}(double* restrict x, const double* restrict cc){{',
                f'  for(long u=0;u<{L};u++){{',
                f'    double* restrict pp = x + u*{2*L};',
                f'    const double* restrict cp = cc + u*{2*L};' if wm else '    (void)cc;']
        body += ['    '+ln for ln in lines_axis(L, L*L, 'pp', wm, 'cp' if wm else None)]
        body += ['  }', '}']
        parts.append(chr(10).join(body))
    V = 2*L*L*L
    if L <= 17:
        runbody = f"""
void run_{L}(long B, long m, double* restrict x0, const double* restrict c,
             double* restrict out1, double* restrict outm){{
  for(long b=0;b<B;b++){{
    double* restrict xb = x0 + b*{V};
    const double* restrict cb = c + b*{V};
    for(long s=0;s<m;s++){{
      pass12_{L}(xb); pass3m_{L}(xb, cb);
      if(s==0) memcpy(out1 + b*{V}, xb, {V}*sizeof(double));
    }}
    memcpy(outm + b*{V}, xb, {V}*sizeof(double));
  }}
}}"""
    else:
        runbody = f"""
void run_{L}(long B, long m, double* restrict x0, const double* restrict c,
             double* restrict out1, double* restrict outm){{
  for(long b=0;b<B;b++){{
    memcpy(STATE, x0 + b*{V}, {V}*sizeof(double));
    memcpy(CBUF, c + b*{V}, {V}*sizeof(double));
    for(long s=0;s<m;s++){{
      pass12_{L}(STATE); pass3m_{L}(STATE, CBUF);
      if(s==0) memcpy(out1 + b*{V}, STATE, {V}*sizeof(double));
    }}
    memcpy(outm + b*{V}, STATE, {V}*sizeof(double));
  }}
}}"""
    parts.append(f"""
void fft3_{L}(double* restrict x){{
  pass12_{L}(x); pass3p_{L}(x, 0);
}}
void step_{L}(double* restrict x, const double* restrict c){{
  pass12_{L}(x); pass3m_{L}(x, c);
}}""" + runbody)
    return (chr(10)+chr(10)).join(parts)

HEADER = '''// auto-generated -- specialized 3D FFT + map (v2)
#include <immintrin.h>
#include <string.h>
#include <sys/mman.h>
#include <stdlib.h>
static double* STATE = 0;
static double* CBUF = 0;
static void* halloc(unsigned long sz){
  void* p = mmap(0, sz + (2u<<20), PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
  if(p == MAP_FAILED) return 0;
  unsigned long a = ((unsigned long)p + (2u<<20) - 1) & ~((2ul<<20)-1ul);
  madvise((void*)a, sz, MADV_HUGEPAGE);
  return (void*)a;
}
double* get_state(void){ return STATE; }
double* get_cbuf(void){ return CBUF; }
void init_mem(void){
  if(STATE) return;
  STATE = (double*)halloc(16ul<<20);
  CBUF  = (double*)halloc(16ul<<20);
  if(!STATE) STATE = (double*)aligned_alloc(64, 16ul<<20);
  if(!CBUF)  CBUF  = (double*)aligned_alloc(64, 16ul<<20);
  for(long i=0;i<(16l<<17);i+=512){ STATE[i]=0.0; CBUF[i]=0.0; }
}
'''

def generate(sizes=SIZES, fname='implementation.c'):
    out = [HEADER]
    for L in PRIMES:
        if L in sizes:
            out.append(genk.prime_tables(L))
            out.append(genk.prime_tables_scalar(L))
    if 64 in sizes: out.append(genk.tw64_table())
    for L in sizes:
        out.append(gen_size(L))
        if L <= 45:
            out.append(gen_size_iN(L, 4))
            out.append(gen_size_iN(L, 2))
    with open(fname, 'w') as f:
        f.write('\n\n'.join(out) + '\n')

if __name__ == '__main__':
    import sys
    sizes = tuple(int(a) for a in sys.argv[1:]) or SIZES
    generate(sizes)
    print('generated v2', sizes)
