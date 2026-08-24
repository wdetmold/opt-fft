import sys
sys.path.insert(0, '/tmp/dev')
from genlib import *

SIZES = (6, 8, 13, 17, 23, 36, 45, 64)
CLASS_A = (6, 8, 13, 17, 23)
CLASS_B = (36, 45, 64)
PADB = {36: 40, 45: 48, 64: 72}       # row stride (doubles) for class B
AW = (8, 4, 2, 1)                      # class A widths

def hexf(c):
    if c == int(c) and abs(c) < 1e15:
        return f"{c!r}"
    return float(c).hex()

TYPE = {8: 'v8df', 4: 'v4df', 2: 'v2df', 1: 'double'}

def KC(W, c):
    h = hexf(c)
    if W == 1: return f"({h})"
    return f"K{W}({h})"

def body_lines(g, outs, invars, W):
    """emit temp lines; returns (lines, names dict node->cexpr)"""
    lines = []
    names = dict(invars)
    ctr = [0]
    def nm(i):
        if i in names: return names[i]
        t = g.nodes[i]
        k = t[0]
        if k == 'add': e = f"{nm(t[1])} + {nm(t[2])}"
        elif k == 'sub': e = f"{nm(t[1])} - {nm(t[2])}"
        elif k == 'neg': e = f"-({nm(t[1])})"
        elif k == 'mul': e = f"{KC(W, t[1])} * {nm(t[2])}"
        elif k == 'zero': e = KC(W, 0.0)
        else: raise Exception(k)
        v = f"t{ctr[0]}"; ctr[0] += 1
        lines.append(f"  {TYPE[W]} {v} = {e};")
        names[i] = v
        return v
    for (r, i) in outs:
        nm(r); nm(i)
    return lines, names

def gen_kernel(L, W, kind, S, name, P=None):
    """kind: 'plain' | 'xfused' | 'zmapc' | 'colT'  ; S stride in doubles"""
    g = G()
    xin = [(g.inp(('x', j, 0)), g.inp(('x', j, 1))) for j in range(L)]
    outs = dft(g, L, xin)
    invars = {}
    T = TYPE[W]
    head = []
    if kind == 'zmapc':
        for j in range(L):
            head.append(f"  {T} x{j}r, x{j}i;")
            head.append(f"  {{ {T} zr = *(const {T}*)(pr + {j*S}) + *(const {T}*)(cr + {j*S});")
            head.append(f"    {T} zi = *(const {T}*)(pi + {j*S}) + *(const {T}*)(ci + {j*S});")
            head.append(f"    {T} mm = zr*zr + zi*zi;")
            head.append(f"    {T} uu = rsq{W}(mm);")
            head.append(f"    {T} wr_ = zr*uu, wi_ = zi*uu;")
            head.append(f"    {T} vv = rpc{W}(uu);")
            head.append(f"    x{j}r = wr_*vv; x{j}i = wi_*vv; }}")
    else:
        for j in range(L):
            head.append(f"  const {T} x{j}r = *(const {T}*)(pr + {j*S});")
            head.append(f"  const {T} x{j}i = *(const {T}*)(pi + {j*S});")
    for j in range(L):
        invars[('in', ('x', j, 0))] = f"x{j}r"
        invars[('in', ('x', j, 1))] = f"x{j}i"
    inv2 = {g.inp(('x', j, c)): v for (_, (x, j, c)), v in
            [((k, kk), vv) for (k, kk), vv in [(('in', key[1]), val) for key, val in invars.items()]]}
    # simpler: rebuild invars keyed by node id
    inv = {}
    for j in range(L):
        inv[g.inp(('x', j, 0))] = f"x{j}r"
        inv[g.inp(('x', j, 1))] = f"x{j}i"
    lines, names = body_lines(g, outs, inv, W)
    out = []
    if kind in ('plain', 'zmapc'):
        if kind == 'plain':
            sig = f"static void {name}(double*restrict pr, double*restrict pi)"
        else:
            sig = (f"static void {name}(double*restrict pr, double*restrict pi,"
                   f" const double*restrict cr, const double*restrict ci)")
        out.append(sig + " {")
        out += head + lines
        for k in range(L):
            out.append(f"  *({T}*)(pr + {k*S}) = {names[outs[k][0]]};")
            out.append(f"  *({T}*)(pi + {k*S}) = {names[outs[k][1]]};")
        out.append("}")
    elif kind == 'xfused':
        sig = (f"static void {name}(double*restrict pr, double*restrict pi,"
               f" const double*restrict cr, const double*restrict ci)")
        out.append(sig + " {")
        out += head + lines
        for k in range(L):
            yr, yi = names[outs[k][0]], names[outs[k][1]]
            out.append(f"  {{ {T} zr = {yr} + *(const {T}*)(cr + {k*S});")
            out.append(f"    {T} zi = {yi} + *(const {T}*)(ci + {k*S});")
            out.append(f"    {T} mm = zr*zr + zi*zi;")
            out.append(f"    {T} rr = maprec{W}(mm);")
            out.append(f"    *({T}*)(pr + {k*S}) = zr*rr;")
            out.append(f"    *({T}*)(pi + {k*S}) = zi*rr; }}")
        out.append("}")
    elif kind == 'colT':
        sig = (f"static void {name}(const double*restrict pr, const double*restrict pi,"
               f" double*restrict dr, double*restrict di)")
        out.append(sig + " {")
        out += head + lines
        # transposed stores in chunks of W
        nch = (L + W - 1) // W
        for ch in range(nch):
            ks = [min(ch*W + t, L-1) for t in range(W)]  # clamp padding chunks
            if W == 8:
                for comp, sel in ((0, 'r'), (1, 'i')):
                    regs = ", ".join(f"(__m512d){names[outs[k][comp]]}" for k in ks)
                    dst = 'dr' if comp == 0 else 'di'
                    out.append(f"  tr8x8_store({regs}, {dst} + {ch*8}, {P});")
            elif W == 4:
                if ch*W + W <= L or True:
                    valid = L - ch*W
                    if valid >= 4 or valid <= 0: valid = 4
                    if ch*W + 4 <= L:
                        for comp in (0, 1):
                            regs = ", ".join(f"(__m256d){names[outs[k][comp]]}" for k in ks[:4])
                            dst = 'dr' if comp == 0 else 'di'
                            out.append(f"  tr4x4_store({regs}, {dst} + {ch*4}, {P});")
                    else:
                        # leftover single columns
                        for k in range(ch*W, L):
                            for comp in (0, 1):
                                dst = 'dr' if comp == 0 else 'di'
                                v = names[outs[k][comp]]
                                for lane in range(4):
                                    out.append(f"  {dst}[{lane}*{P} + {k}] = (({TYPE[1]} const*)&{v})[{lane}];")
            elif W == 1:
                pass
        if W == 1:
            for k in range(L):
                out.append(f"  dr[{k}] = {names[outs[k][0]]};")
                out.append(f"  di[{k}] = {names[outs[k][1]]};")
        if W == 2:
            raise Exception("colT w2 unused")
        out.append("}")
    return "\n".join(out)

print("gen.py lib ready")


def gen_kernel_x2(L, W, S, name):
    """xfused processing two adjacent j2 columns (offset W doubles apart)"""
    T = TYPE[W]
    out = [f"static void {name}(double*restrict pr, double*restrict pi,"
           f" const double*restrict cr, const double*restrict ci) {{"]
    for col in range(2):
        off = col*W
        g = G()
        inv = {}
        head = []
        for j in range(L):
            head.append(f"  const {T} x{col}_{j}r = *(const {T}*)(pr + {j*S+off});")
            head.append(f"  const {T} x{col}_{j}i = *(const {T}*)(pi + {j*S+off});")
            inv[g.inp(('x', j, 0))] = f"x{col}_{j}r"
            inv[g.inp(('x', j, 1))] = f"x{col}_{j}i"
        xin = [(g.inp(('x', j, 0)), g.inp(('x', j, 1))) for j in range(L)]
        outs = dft(g, L, xin)
        lines, names = body_lines(g, outs, inv, W)
        import re as _re
        ren = lambda s: _re.sub(r"\bt(\d+)\b", f"c{col}t" + r"\1", s)
        out += head + [ren(l) for l in lines]
        for k in range(L):
            yr, yi = ren(names[outs[k][0]]), ren(names[outs[k][1]])
            out.append(f"  {{ {T} zr = {yr} + *(const {T}*)(cr + {k*S+off});")
            out.append(f"    {T} zi = {yi} + *(const {T}*)(ci + {k*S+off});")
            out.append(f"    {T} mm = zr*zr + zi*zi;")
            out.append(f"    {T} uu = rsq{W}(mm);")
            out.append(f"    {T} wr_ = zr*uu, wi_ = zi*uu;")
            out.append(f"    {T} vv = rpc{W}(uu);")
            out.append(f"    *({T}*)(pr + {k*S+off}) = wr_*vv;")
            out.append(f"    *({T}*)(pi + {k*S+off}) = wi_*vv; }}")
    out.append("}")
    return "\n".join(out)
