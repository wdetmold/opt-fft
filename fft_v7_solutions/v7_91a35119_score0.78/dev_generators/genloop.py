import sys
sys.path.insert(0, '/tmp/dev')
from gen2 import *
import numpy as np

MAPC = {8:'maprec8',4:'maprec4',2:'maprec2',1:'maprec1'}
SET1 = {8:'_mm512_set1_pd',4:'_mm256_set1_pd',2:'_mm_set1_pd'}
def BC(W, expr):
    if W == 1: return f"({expr})"
    return f"(({TYPE[W]}){SET1[W]}({expr}))"

def emit_map_store(o, W, T, ind, zr_expr, zi_expr, cr_expr, ci_expr, dst_r, dst_i):
    """z = y + c; out = z/(1+|z|) using early-mul scheme"""
    o.append(f"{ind}{{ {T} zr = {zr_expr} + {cr_expr};")
    o.append(f"{ind}  {T} zi = {zi_expr} + {ci_expr};")
    o.append(f"{ind}  {T} mm = zr*zr + zi*zi;")
    o.append(f"{ind}  {T} uu = rsq{W}(mm);")
    o.append(f"{ind}  {T} wr_ = zr*uu, wi_ = zi*uu;")
    o.append(f"{ind}  {T} vv = rpc{W}(uu);")
    o.append(f"{ind}  {dst_r} = wr_*vv; {dst_i} = wi_*vv; }}")

def symodd_loop(L, W, kind, S, name):
    h = (L-1)//2
    T = TYPE[W]
    rows = []
    for k in range(1, h+1):
        cs = [tw(L, k*j) for j in range(1, h+1)]
        rows.append([c for c,_ in cs] + [-s for _,s in cs])
    tbl = ",\n  ".join("{" + ",".join(hexf(v) for v in r) + "}" for r in rows)
    o = []
    o.append(f"static const double CT_{name}[{h}][{2*h}] __attribute__((aligned(64))) = {{\n  {tbl}}};")
    if kind == 'plain':
        o.append(f"static void {name}(double*restrict pr, double*restrict pi){{")
    else:
        o.append(f"static void {name}(double*restrict pr, double*restrict pi, const double*restrict cr, const double*restrict ci){{")
    o.append(f"  const {T} x0r = *(const {T}*)(pr);")
    o.append(f"  const {T} x0i = *(const {T}*)(pi);")
    for j in range(1, h+1):
        o.append(f"  {T} e{j}r, e{j}i, o{j}r, o{j}i;")
        o.append(f"  {{ const {T} a = *(const {T}*)(pr + {j*S}), b = *(const {T}*)(pr + {(L-j)*S});")
        o.append(f"    e{j}r = a + b; o{j}r = a - b; }}")
        o.append(f"  {{ const {T} a = *(const {T}*)(pi + {j*S}), b = *(const {T}*)(pi + {(L-j)*S});")
        o.append(f"    e{j}i = a + b; o{j}i = a - b; }}")
    half = (h+1)//2
    t1r = " + ".join(f"e{j}r" for j in range(1, half+1)); t2r = " + ".join(f"e{j}r" for j in range(half+1, h+1))
    t1i = " + ".join(f"e{j}i" for j in range(1, half+1)); t2i = " + ".join(f"e{j}i" for j in range(half+1, h+1))
    o.append(f"  {T} X0r = (x0r + ({t1r})) + ({t2r});")
    o.append(f"  {T} X0i = (x0i + ({t1i})) + ({t2i});")
    if kind != 'xfused':
        o.append(f"  *({T}*)(pr) = X0r; *({T}*)(pi) = X0i;")
    else:
        emit_map_store(o, W, T, "  ", "X0r", "X0i", f"(*(const {T}*)(cr))", f"(*(const {T}*)(ci))",
                       f"*({T}*)(pr)", f"*({T}*)(pi)")
    o.append(f"  double* plo_r = pr + {S}; double* plo_i = pi + {S};")
    o.append(f"  double* phi_r = pr + {(L-1)*S}; double* phi_i = pi + {(L-1)*S};")
    if kind == 'xfused':
        o.append(f"  const double* clo_r = cr + {S}; const double* clo_i = ci + {S};")
        o.append(f"  const double* chi_r = cr + {(L-1)*S}; const double* chi_i = ci + {(L-1)*S};")
    o.append(f"  for(int k=0;k<{h};++k){{")
    o.append(f"    const double* ct = CT_{name}[k];")
    # split chains into two halves for latency
    o.append(f"    {T} Pr0 = x0r + {BC(W,'ct[0]')}*e1r;")
    o.append(f"    {T} Pi0 = x0i + {BC(W,'ct[0]')}*e1i;")
    o.append(f"    {T} Qr0 = {BC(W,f'ct[{h}]')}*o1r;")
    o.append(f"    {T} Qi0 = {BC(W,f'ct[{h}]')}*o1i;")
    o.append(f"    {T} Pr1 = {BC(W,'ct[1]')}*e2r;")
    o.append(f"    {T} Pi1 = {BC(W,'ct[1]')}*e2i;")
    o.append(f"    {T} Qr1 = {BC(W,f'ct[{h+1}]')}*o2r;")
    o.append(f"    {T} Qi1 = {BC(W,f'ct[{h+1}]')}*o2i;")
    for j in range(3, h+1):
        half_id = (j-1) % 2
        o.append(f"    {{ const {T} cj = {BC(W,f'ct[{j-1}]')}; Pr{half_id} += cj*e{j}r; Pi{half_id} += cj*e{j}i; }}")
        o.append(f"    {{ const {T} sj = {BC(W,f'ct[{h+j-1}]')}; Qr{half_id} += sj*o{j}r; Qi{half_id} += sj*o{j}i; }}")
    o.append(f"    {T} Pr = Pr0 + Pr1, Pi = Pi0 + Pi1, Qr = Qr0 + Qr1, Qi = Qi0 + Qi1;")
    o.append(f"    {T} Ar = Pr + Qi, Ai = Pi - Qr, Brx = Pr - Qi, Bix = Pi + Qr;")
    if kind == 'plain':
        o.append(f"    *({T}*)(plo_r) = Ar; *({T}*)(plo_i) = Ai;")
        o.append(f"    *({T}*)(phi_r) = Brx; *({T}*)(phi_i) = Bix;")
    else:
        emit_map_store(o, W, T, "    ", "Ar", "Ai", f"(*(const {T}*)(clo_r))", f"(*(const {T}*)(clo_i))",
                       f"*({T}*)(plo_r)", f"*({T}*)(plo_i)")
        emit_map_store(o, W, T, "    ", "Brx", "Bix", f"(*(const {T}*)(chi_r))", f"(*(const {T}*)(chi_i))",
                       f"*({T}*)(phi_r)", f"*({T}*)(phi_i)")
        o.append(f"    clo_r += {S}; clo_i += {S}; chi_r -= {S}; chi_i -= {S};")
    o.append(f"    plo_r += {S}; plo_i += {S}; phi_r -= {S}; phi_i -= {S};")
    o.append("  }")
    o.append("}")
    return "\n".join(o)

print("genloop ready")

def symodd_grouped(L, W, kind, S, name, G=4):
    h = (L-1)//2
    T = TYPE[W]
    o = []
    if kind == 'plain':
        o.append(f"static void {name}(double*restrict pr, double*restrict pi){{")
    else:
        o.append(f"static void {name}(double*restrict pr, double*restrict pi, const double*restrict cr, const double*restrict ci){{")
    o.append(f"  {T} er[{h}] __attribute__((aligned(64))), ei[{h}] __attribute__((aligned(64)));")
    o.append(f"  {T} odr[{h}] __attribute__((aligned(64))), odi[{h}] __attribute__((aligned(64)));")
    if kind == 'zmapc':
        o.append(f"  {T} x0r, x0i;")
        o.append(f"  {{ {T} zr = *(const {T}*)(pr) + *(const {T}*)(cr);")
        o.append(f"    {T} zi = *(const {T}*)(pi) + *(const {T}*)(ci);")
        o.append(f"    {T} mm = zr*zr + zi*zi;")
        o.append(f"    {T} uu = rsq{W}(mm);")
        o.append(f"    {T} wr_ = zr*uu, wi_ = zi*uu;")
        o.append(f"    {T} vv = rpc{W}(uu);")
        o.append(f"    x0r = wr_*vv; x0i = wi_*vv; }}")
        for j in range(1, h+1):
            o.append(f"  {{ {T} zr1 = *(const {T}*)(pr + {j*S}) + *(const {T}*)(cr + {j*S});")
            o.append(f"    {T} zi1 = *(const {T}*)(pi + {j*S}) + *(const {T}*)(ci + {j*S});")
            o.append(f"    {T} zr2 = *(const {T}*)(pr + {(L-j)*S}) + *(const {T}*)(cr + {(L-j)*S});")
            o.append(f"    {T} zi2 = *(const {T}*)(pi + {(L-j)*S}) + *(const {T}*)(ci + {(L-j)*S});")
            o.append(f"    {T} mm1 = zr1*zr1 + zi1*zi1;")
            o.append(f"    {T} mm2 = zr2*zr2 + zi2*zi2;")
            o.append(f"    {T} uu1 = rsq{W}(mm1);")
            o.append(f"    {T} uu2 = rsq{W}(mm2);")
            o.append(f"    {T} wr1 = zr1*uu1, wi1 = zi1*uu1, wr2 = zr2*uu2, wi2 = zi2*uu2;")
            o.append(f"    {T} vv1 = rpc{W}(uu1);")
            o.append(f"    {T} vv2 = rpc{W}(uu2);")
            o.append(f"    {T} a_r = wr1*vv1, a_i = wi1*vv1, b_r = wr2*vv2, b_i = wi2*vv2;")
            o.append(f"    er[{j-1}] = a_r + b_r; odr[{j-1}] = a_r - b_r;")
            o.append(f"    ei[{j-1}] = a_i + b_i; odi[{j-1}] = a_i - b_i; }}")
    else:
        o.append(f"  const {T} x0r = *(const {T}*)(pr);")
        o.append(f"  const {T} x0i = *(const {T}*)(pi);")
        for j in range(1, h+1):
            o.append(f"  {{ const {T} a = *(const {T}*)(pr + {j*S}), b = *(const {T}*)(pr + {(L-j)*S});")
            o.append(f"    er[{j-1}] = a + b; odr[{j-1}] = a - b; }}")
            o.append(f"  {{ const {T} a = *(const {T}*)(pi + {j*S}), b = *(const {T}*)(pi + {(L-j)*S});")
            o.append(f"    ei[{j-1}] = a + b; odi[{j-1}] = a - b; }}")
    half = (h+1)//2
    t1r = " + ".join(f"er[{j}]" for j in range(half)); t2r = " + ".join(f"er[{j}]" for j in range(half, h))
    t1i = " + ".join(f"ei[{j}]" for j in range(half)); t2i = " + ".join(f"ei[{j}]" for j in range(half, h))
    o.append(f"  {T} X0r = (x0r + ({t1r})) + ({t2r});")
    o.append(f"  {T} X0i = (x0i + ({t1i})) + ({t2i});")
    if kind != 'xfused':
        o.append(f"  *({T}*)(pr) = X0r; *({T}*)(pi) = X0i;")
    else:
        emit_map_store(o, W, T, "  ", "X0r", "X0i", f"(*(const {T}*)(cr))", f"(*(const {T}*)(ci))",
                       f"*({T}*)(pr)", f"*({T}*)(pi)")
    # k groups
    ks = list(range(1, h+1))
    groups = []
    i = 0
    while i < len(ks):
        g = min(G, len(ks)-i)
        groups.append(ks[i:i+g]); i += g
    for grp in groups:
        o.append("  {")
        for k in grp:
            o.append(f"    {T} P{k}r = x0r, P{k}i = x0i, Q{k}r, Q{k}i;")
        for j in range(1, h+1):
            o.append(f"    {{ const {T} vr = er[{j-1}], vi = ei[{j-1}], wr = odr[{j-1}], wi = odi[{j-1}];")
            for k in grp:
                c, sn = tw(L, k*j); s = -sn
                o.append(f"      P{k}r += {KC(W,c)}*vr; P{k}i += {KC(W,c)}*vi;")
                if j == 1:
                    o.append(f"      Q{k}r = {KC(W,s)}*wr; Q{k}i = {KC(W,s)}*wi;")
                else:
                    o.append(f"      Q{k}r += {KC(W,s)}*wr; Q{k}i += {KC(W,s)}*wi;")
            o.append("    }")
        for k in grp:
            o.append(f"    {T} A{k}r = P{k}r + Q{k}i, A{k}i = P{k}i - Q{k}r;")
            o.append(f"    {T} B{k}r = P{k}r - Q{k}i, B{k}i = P{k}i + Q{k}r;")
            if kind != 'xfused':
                o.append(f"    *({T}*)(pr + {k*S}) = A{k}r; *({T}*)(pi + {k*S}) = A{k}i;")
                o.append(f"    *({T}*)(pr + {(L-k)*S}) = B{k}r; *({T}*)(pi + {(L-k)*S}) = B{k}i;")
            else:
                emit_map_store(o, W, T, "    ", f"A{k}r", f"A{k}i",
                               f"(*(const {T}*)(cr + {k*S}))", f"(*(const {T}*)(ci + {k*S}))",
                               f"*({T}*)(pr + {k*S})", f"*({T}*)(pi + {k*S})")
                emit_map_store(o, W, T, "    ", f"B{k}r", f"B{k}i",
                               f"(*(const {T}*)(cr + {(L-k)*S}))", f"(*(const {T}*)(ci + {(L-k)*S}))",
                               f"*({T}*)(pr + {(L-k)*S})", f"*({T}*)(pi + {(L-k)*S})")
        o.append("  }")
    o.append("}")
    return "\n".join(o)

def symodd_2col(L, W, kind, S, COFF, name, G=2):
    """two-column symodd kernel; columns at +0 and +COFF doubles. kind: 'plain' only."""
    h = (L-1)//2
    T = TYPE[W]
    o = []
    o.append(f"static void {name}(double*restrict pr, double*restrict pi){{")
    for c in range(2):
        o.append(f"  {T} er{c}[{h}] __attribute__((aligned(64))), ei{c}[{h}] __attribute__((aligned(64)));")
        o.append(f"  {T} odr{c}[{h}] __attribute__((aligned(64))), odi{c}[{h}] __attribute__((aligned(64)));")
    o.append(f"  const {T} x0r0 = *(const {T}*)(pr);")
    o.append(f"  const {T} x0i0 = *(const {T}*)(pi);")
    o.append(f"  const {T} x0r1 = *(const {T}*)(pr + {COFF});")
    o.append(f"  const {T} x0i1 = *(const {T}*)(pi + {COFF});")
    for j in range(1, h+1):
        for c in range(2):
            off = c*COFF
            o.append(f"  {{ const {T} a = *(const {T}*)(pr + {j*S+off}), b = *(const {T}*)(pr + {(L-j)*S+off});")
            o.append(f"    er{c}[{j-1}] = a + b; odr{c}[{j-1}] = a - b; }}")
            o.append(f"  {{ const {T} a = *(const {T}*)(pi + {j*S+off}), b = *(const {T}*)(pi + {(L-j)*S+off});")
            o.append(f"    ei{c}[{j-1}] = a + b; odi{c}[{j-1}] = a - b; }}")
    half = (h+1)//2
    for c in range(2):
        t1r = " + ".join(f"er{c}[{j}]" for j in range(half)); t2r = " + ".join(f"er{c}[{j}]" for j in range(half, h))
        t1i = " + ".join(f"ei{c}[{j}]" for j in range(half)); t2i = " + ".join(f"ei{c}[{j}]" for j in range(half, h))
        o.append(f"  {{ {T} X0r = (x0r{c} + ({t1r})) + ({t2r});")
        o.append(f"    {T} X0i = (x0i{c} + ({t1i})) + ({t2i});")
        o.append(f"    *({T}*)(pr + {c*COFF}) = X0r; *({T}*)(pi + {c*COFF}) = X0i; }}")
    ks = list(range(1, h+1))
    groups = []
    i = 0
    while i < len(ks):
        g = min(G, len(ks)-i)
        groups.append(ks[i:i+g]); i += g
    for grp in groups:
        o.append("  {")
        for k in grp:
            for c in range(2):
                o.append(f"    {T} P{k}r{c} = x0r{c}, P{k}i{c} = x0i{c}, Q{k}r{c}, Q{k}i{c};")
        for j in range(1, h+1):
            o.append(f"    {{")
            for c in range(2):
                o.append(f"      const {T} vr{c} = er{c}[{j-1}], vi{c} = ei{c}[{j-1}], wr{c} = odr{c}[{j-1}], wi{c} = odi{c}[{j-1}];")
            for k in grp:
                cst, sn = tw(L, k*j); s = -sn
                o.append(f"      {{ const {T} cc_ = {KC(W,cst)}; const {T} ss_ = {KC(W,s)};")
                for c in range(2):
                    o.append(f"        P{k}r{c} += cc_*vr{c}; P{k}i{c} += cc_*vi{c};")
                    if j == 1:
                        o.append(f"        Q{k}r{c} = ss_*wr{c}; Q{k}i{c} = ss_*wi{c};")
                    else:
                        o.append(f"        Q{k}r{c} += ss_*wr{c}; Q{k}i{c} += ss_*wi{c};")
                o.append("      }")
            o.append(f"    }}")
        for k in grp:
            for c in range(2):
                off = c*COFF
                o.append(f"    {{ {T} Ar = P{k}r{c} + Q{k}i{c}, Ai = P{k}i{c} - Q{k}r{c};")
                o.append(f"      {T} Brx = P{k}r{c} - Q{k}i{c}, Bix = P{k}i{c} + Q{k}r{c};")
                o.append(f"      *({T}*)(pr + {k*S+off}) = Ar; *({T}*)(pi + {k*S+off}) = Ai;")
                o.append(f"      *({T}*)(pr + {(L-k)*S+off}) = Brx; *({T}*)(pi + {(L-k)*S+off}) = Bix; }}")
        o.append("  }")
    o.append("}")
    return "\n".join(o)
