# software-pipelined two-stage pass emitter for PFA/CT nets (vol layout)
import re
from netlib import E, dft_small, cmul_const

def _stage1(L, plan, qname, stride, RW, scname):
    e = E()
    kind, A, B = plan
    if kind == 'pfa':
        inm = {(n1,n2): ((B*n1 + A*n2) % L) for n1 in range(A) for n2 in range(B)}
        tw = lambda e2, x, k1, n2: x
    else:
        inm = {(n1,n2): (B*n1 + n2) for n1 in range(A) for n2 in range(B)}
        tw = lambda e2, x, k1, n2: cmul_const(e2, x, k1*n2, L)
    for n2 in range(B):
        xs = []
        for n1 in range(A):
            j = inm[(n1,n2)]
            xs.append((e.v(f"LD({qname} + {j*stride})"), e.v(f"LD({qname} + {j*stride} + {RW})")))
        ys = dft_small(e, xs, A)
        for k1 in range(A):
            w = tw(e, ys[k1], k1, n2)
            e.raw(f"ST({scname} + {(n2*A+k1)*16}, {w[0]}); ST({scname} + {(n2*A+k1)*16+8}, {w[1]});")
    return e.lines

def _stage2(L, plan, storer, scname):
    e = E()
    kind, A, B = plan
    if kind == 'pfa':
        Binv = pow(B,-1,A); Ainv = pow(A,-1,B)
        outm = {(k1,k2): ((k1*B*Binv + k2*A*Ainv) % L) for k1 in range(A) for k2 in range(B)}
    else:
        outm = {(k1,k2): (A*k2 + k1) for k1 in range(A) for k2 in range(B)}
    for k1 in range(A):
        xs = [(e.v(f"LD({scname} + {(n2*A+k1)*16})"), e.v(f"LD({scname} + {(n2*A+k1)*16+8})")) for n2 in range(B)]
        ys = dft_small(e, xs, B)
        for k2 in range(B):
            storer(e, outm[(k1,k2)], ys[k2][0], ys[k2][1])
    return e.lines

def _zip(la, lb):
    out=[]; na,nb=len(la),len(lb); ia=ib=0
    while ia<na or ib<nb:
        if ib>=nb or (ia<na and ia*nb<=ib*na):
            out.append(la[ia]); ia+=1
        else:
            out.append(lb[ib]); ib+=1
    return out

def _ren(lines, sfx):
    return [re.sub(r'\bt(\d+)\b', r't\1'+sfx, ln) for ln in lines]

def emit_piped(L, plan, npencils, qfmt, cfmt, stride, RW, sc_base, make_storer):
    """qfmt: format string with {R} placeholder for pencil index expr.
       make_storer(e_unused, qname, cname) -> storer callable(e,k,re,im).
       Returns code string for the full pass."""
    e = E()
    # prologue: stage1 of pencil 0 -> SCA
    e.raw("{")
    e.raw(f"double* restrict q = {qfmt.format(R='0')};")
    for ln in _ren(_stage1(L, plan, "q", stride, RW, sc_base+"A"), "p"): e.raw(ln)
    e.raw("}")
    e.raw(f"for(long r=1;r<{npencils};r++){{")
    e.raw(f"double* restrict q = {qfmt.format(R='r')};")
    e.raw(f"double* restrict qp = {qfmt.format(R='(r-1)')};")
    if cfmt: e.raw(f"const double* restrict cqp = {cfmt.format(R='(r-1)')};")
    e.raw(f"double* restrict sA = (r&1)? {sc_base}B : {sc_base}A;")
    e.raw(f"double* restrict sB = (r&1)? {sc_base}A : {sc_base}B;")
    l1 = _stage1(L, plan, "q", stride, RW, "sA")
    st = make_storer("qp", "cqp")
    l2 = _stage2(L, plan, st, "sB")
    for ln in _zip(_ren(l1,"a"), _ren(l2,"b")): e.raw(ln)
    e.raw("}")
    e.raw("{")
    e.raw(f"double* restrict qp = {qfmt.format(R=f'({npencils}-1)')};")
    if cfmt: e.raw(f"const double* restrict cqp = {cfmt.format(R=f'({npencils}-1)')};")
    par = "B" if (npencils-1)%2==1 else "A"
    e.raw(f"double* restrict sB = {sc_base}{par};")
    st = make_storer("qp", "cqp")
    for ln in _ren(_stage2(L, plan, st, "sB"), "e"): e.raw(ln)
    e.raw("}")
    return e.code()
