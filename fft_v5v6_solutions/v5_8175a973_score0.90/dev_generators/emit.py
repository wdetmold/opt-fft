import numpy as np
from ir import *

def fmt_const(c):
    return float(c).hex()

MAP_SEQ = '''  {{ __m512d zr = {RE} + {LDCR};
    __m512d zi = {IM} + {LDCI};
    __m512d r2 = zr*zr + zi*zi;
    r2 = _mm512_max_pd(r2, _mm512_set1_pd(1e-300));
    __m512d e = _mm512_rsqrt14_pd(r2);
    __m512d h = r2 * _mm512_set1_pd(0.5);
    e = e * (_mm512_set1_pd(1.5) - h*e*e);
    e = e * (_mm512_set1_pd(1.5) - h*e*e);
    __m512d rc = _mm512_div_pd(vone, vone + r2*e);
    {ST_RE}
    {ST_IM} }}'''

class Emitter:
    def __init__(self):
        self.body = []
        self.names = {}
        self.tmp = 0
        self.emitted = set()
        self.scr = set()
        self.B = Builder()
        self.srcs = {}

    def declare_scratch(self, name, n):
        if name not in self.scr:
            self.scr.add(name)
            self.body.append(f"  __attribute__((aligned(64))) double {name}[{n*8}];")

    def load(self, key, cptr, mask=None):
        i = self.B.var(key)
        if i in self.emitted:
            return i
        v = f"v{self.tmp}"; self.tmp += 1
        if mask:
            self.body.append(f"  __m512d {v} = _mm512_maskz_loadu_pd({mask}, {cptr});")
        elif cptr.startswith('&'):
            self.body.append(f"  __m512d {v} = _mm512_load_pd({cptr});")
        else:
            self.body.append(f"  __m512d {v} = _mm512_loadu_pd({cptr});")
        self.names[i] = v
        self.emitted.add(i)
        return i

    def emit_node(self, i):
        if i in self.emitted: return
        n = self.B.nodes[i]
        if n[0] == 'var':
            raise RuntimeError("unloaded var")
        if n[0] == 'const':
            self.emitted.add(i)
            if n[1] == 0.0: self.names[i] = "_mm512_setzero_pd()"
            else: self.names[i] = f"_mm512_set1_pd({fmt_const(n[1])})"
            return
        for a in n[1:]:
            self.emit_node(a)
        v = f"v{self.tmp}"; self.tmp += 1
        if n[0] == 'neg':
            self.body.append(f"  __m512d {v} = -{self.names[n[1]]};")
        else:
            op = {'add':'+','sub':'-','mul':'*'}[n[0]]
            self.body.append(f"  __m512d {v} = {self.names[n[1]]} {op} {self.names[n[2]]};")
        self.names[i] = v
        self.emitted.add(i)

    def store(self, cptr, node, mask=None, aligned=False):
        self.emit_node(node)
        if mask:
            self.body.append(f"  _mm512_mask_storeu_pd({cptr}, {mask}, {self.names[node]});")
        elif aligned:
            self.body.append(f"  _mm512_store_pd({cptr}, {self.names[node]});")
        else:
            self.body.append(f"  _mm512_storeu_pd({cptr}, {self.names[node]});")

    def store_map(self, k, re, im, mask=None):
        self.emit_node(re); self.emit_node(im)
        m = mask
        ldc_r = (f"_mm512_maskz_loadu_pd({m}, pcr + {k}*sc)" if m else f"_mm512_loadu_pd(pcr + {k}*sc)")
        ldc_i = (f"_mm512_maskz_loadu_pd({m}, pci + {k}*sc)" if m else f"_mm512_loadu_pd(pci + {k}*sc)")
        st_re = (f"_mm512_mask_storeu_pd(pyr + {k}*sy, {m}, zr*rc);" if m
                 else f"_mm512_storeu_pd(pyr + {k}*sy, zr*rc);")
        st_im = (f"_mm512_mask_storeu_pd(pyi + {k}*sy, {m}, zi*rc);" if m
                 else f"_mm512_storeu_pd(pyi + {k}*sy, zi*rc);")
        self.body.append(MAP_SEQ.format(RE=self.names[re], IM=self.names[im],
                                        LDCR=ldc_r, LDCI=ldc_i, ST_RE=st_re, ST_IM=st_im))

def stage_plan(N):
    if N == 64: return ('ct', 8, 8)
    if N == 36: return ('pfa', 9, 4)
    if N == 45: return ('pfa', 9, 5)
    if N in (13, 17, 23): return ('prime',)
    return ('single',)

def emit_codelet(N, name, masked=False, mapped=False, in_perm=None, out_perm=None, tileT=False, pf=0, interleaved_in=False, ret_parts=False):
    """tileT: only for ('ct',8,8): fused transposed tile store; out rows at pyr + u*TROW + d*8.
       in_perm/out_perm: permutation lists applied to load/store indices."""
    E = Emitter()
    B = E.B
    args = "const double* pxr, const double* pxi, double* pyr, double* pyi, long sx, long sy"
    if masked: args += ", __mmask8 ml, __mmask8 ms"
    if mapped: args += ", const double* pcr, const double* pci, long sc"
    head = [f"static void {name}({args}){{"]
    if mapped: head.append("  const __m512d vone = _mm512_set1_pd(1.0);")
    if mapped and pf:
        for k in range(N):
            head.append(f"  _mm_prefetch((const char*)(pcr + {k}*sc), _MM_HINT_T0);")
            head.append(f"  _mm_prefetch((const char*)(pci + {k}*sc), _MM_HINT_T0);")
    ml = "ml" if masked else None
    ms = "ms" if masked else None
    ip = in_perm if in_perm else list(range(N))
    op_ = out_perm if out_perm else list(range(N))

    def ldx(comp, j):
        if interleaved_in:
            key_r = ('x', 'r', j); key_i = ('x', 'i', j)
            ir = E.B.var(key_r); ii = E.B.var(key_i)
            if ir not in E.emitted:
                vr = f"v{E.tmp}"; E.tmp += 1
                vi = f"v{E.tmp}"; E.tmp += 1
                E.body.append(f"  __m512d {vr}, {vi}; DEINT(pxr + 2*{ip[j]}*sx, {vr}, {vi});")
                E.names[ir] = vr; E.names[ii] = vi
                E.emitted.add(ir); E.emitted.add(ii)
            return ir if comp == 'r' else ii
        r = E.load(('x', comp, j), f"px{comp} + {ip[j]}*sx", mask=ml)
        if pf and ('pf', comp, j) not in E.scr:
            E.scr.add(('pf', comp, j))
            E.body.append(f"  _mm_prefetch((const char*)(px{comp} + {ip[j]}*sx + {pf}), _MM_HINT_T0);")
        return r

    def out_store(k, re, im):
        if mapped:
            E.store_map(op_[k], re, im, mask=ml)
        else:
            E.store(f"pyr + {op_[k]}*sy", re, mask=ms)
            E.store(f"pyi + {op_[k]}*sy", im, mask=ms)

    plan = stage_plan(N)
    if plan[0] == 'single':
        xs = [(ldx('r', j), ldx('i', j)) for j in range(N)]
        outs = dft(B, N, xs)
        for k, (r, i) in enumerate(outs):
            out_store(k, r, i)
    elif plan[0] == 'ct':
        r_, m_ = plan[1], plan[2]
        E.declare_scratch("sAr", N); E.declare_scratch("sAi", N)
        for b in range(r_):
            xs = [(ldx('r', r_*a + b), ldx('i', r_*a + b)) for a in range(m_)]
            sub = dft(B, m_, xs)
            for d in range(m_):
                wr, wi = tw(N, b*d)
                t = cmulw(B, sub[d], wr, wi)
                E.store(f"&sAr[{(b*m_+d)*8}]", t[0], aligned=True)
                E.store(f"&sAi[{(b*m_+d)*8}]", t[1], aligned=True)
        E.stageA_end = len(E.body)
        for d in range(m_):
            ts = [(E.load(('s','r',b,d), f"&sAr[{(b*m_+d)*8}]"),
                   E.load(('s','i',b,d), f"&sAi[{(b*m_+d)*8}]")) for b in range(r_)]
            y = dft(B, r_, ts)
            if tileT:
                # outputs k = d + m_*c ; pi(k) = m_*d + c : contiguous block. transpose lanes
                regsR = []; regsI = []
                for c in range(r_):
                    E.emit_node(y[c][0]); E.emit_node(y[c][1])
                    regsR.append(E.names[y[c][0]]); regsI.append(E.names[y[c][1]])
                E.body.append(f"  {{ __m512d R0={regsR[0]},R1={regsR[1]},R2={regsR[2]},R3={regsR[3]},R4={regsR[4]},R5={regsR[5]},R6={regsR[6]},R7={regsR[7]};")
                E.body.append(f"    __m512d I0={regsI[0]},I1={regsI[1]},I2={regsI[2]},I3={regsI[3]},I4={regsI[4]},I5={regsI[5]},I6={regsI[6]},I7={regsI[7]};")
                E.body.append( "    TR8(R0,R1,R2,R3,R4,R5,R6,R7);")
                E.body.append( "    TR8(I0,I1,I2,I3,I4,I5,I6,I7);")
                for u in range(8):
                    E.body.append(f"    _mm512_storeu_pd(pyr + {u}*TROW + {d*8}, R{u});")
                    E.body.append(f"    _mm512_storeu_pd(pyi + {u}*TROW + {d*8}, I{u});")
                E.body.append("  }")
            else:
                for c in range(r_):
                    out_store(d + m_*c, y[c][0], y[c][1])
    elif plan[0] == 'pfa':
        N1, N2 = plan[1], plan[2]
        from math import gcd as _gcd
        assert _gcd(N1, N2) == 1 and N1*N2 == N
        E.declare_scratch("sAr", N); E.declare_scratch("sAi", N)
        for j2 in range(N2):
            col = [(ldx('r', (N2*j1 + N1*j2) % N), ldx('i', (N2*j1 + N1*j2) % N)) for j1 in range(N1)]
            t = dft(B, N1, col)
            for k1 in range(N1):
                E.store(f"&sAr[{(k1*N2+j2)*8}]", t[k1][0], aligned=True)
                E.store(f"&sAi[{(k1*N2+j2)*8}]", t[k1][1], aligned=True)
        E.stageA_end = len(E.body)
        for k1 in range(N1):
            row = [(E.load(('s','r',k1,j2), f"&sAr[{(k1*N2+j2)*8}]"),
                    E.load(('s','i',k1,j2), f"&sAi[{(k1*N2+j2)*8}]")) for j2 in range(N2)]
            y = dft(B, N2, row)
            for k2 in range(N2):
                k = 0
                for kk in range(N):
                    if kk % N1 == k1 and kk % N2 == k2: k = kk; break
                out_store(k, y[k2][0], y[k2][1])
    elif plan[0] == 'prime':
        h = (N-1)//2
        E.declare_scratch("sPr", h+1); E.declare_scratch("sPi", h+1)
        E.declare_scratch("sQr", h+1); E.declare_scratch("sQi", h+1)
        x0 = (ldx('r', 0), ldx('i', 0))
        acc0 = [None, None]   # two-way split for X0 sums (re, im interleaved pairs)
        s0 = {0: [x0[0]], 1: [x0[1]]}
        for j in range(1, h+1):
            xj = (ldx('r', j), ldx('i', j)); xn = (ldx('r', N-j), ldx('i', N-j))
            p = cadd(B, xj, xn); q = csub(B, xj, xn)
            E.store(f"&sPr[{j*8}]", p[0], aligned=True); E.store(f"&sPi[{j*8}]", p[1], aligned=True)
            E.store(f"&sQr[{j*8}]", q[0], aligned=True); E.store(f"&sQi[{j*8}]", q[1], aligned=True)
            s0[0].append(p[0]); s0[1].append(p[1])
        def tree_sum(nodes):
            nodes = list(nodes)
            while len(nodes) > 1:
                nxt = []
                for i in range(0, len(nodes)-1, 2):
                    nxt.append(B.add(nodes[i], nodes[i+1]))
                if len(nodes) & 1: nxt.append(nodes[-1])
                nodes = nxt
            return nodes[0]
        out_store(0, tree_sum(s0[0]), tree_sum(s0[1]))
        NACC = 2
        for k in range(1, h+1):
            # split accumulators
            arA = [None]*NACC; aiA = [None]*NACC; brA = [None]*NACC; biA = [None]*NACC
            arA[0], aiA[0] = x0
            for jj, j in enumerate(range(1, h+1)):
                t = (k*j) % N
                ang = (LD(2)*PI_LD)*LD(t)/LD(N)
                c = float(np.cos(ang)); s = float(np.sin(ang))
                pr = E.load(('p','r',k,j), f"&sPr[{j*8}]"); pi_ = E.load(('p','i',k,j), f"&sPi[{j*8}]")
                qr = E.load(('q','r',k,j), f"&sQr[{j*8}]"); qi = E.load(('q','i',k,j), f"&sQi[{j*8}]")
                w = jj % NACC
                mcr = B.mul(B.const(c), pr); mci = B.mul(B.const(c), pi_)
                msr = B.mul(B.const(s), qr); msi = B.mul(B.const(s), qi)
                arA[w] = mcr if arA[w] is None else B.add(arA[w], mcr)
                aiA[w] = mci if aiA[w] is None else B.add(aiA[w], mci)
                brA[w] = msr if brA[w] is None else B.add(brA[w], msr)
                biA[w] = msi if biA[w] is None else B.add(biA[w], msi)
            ar = tree_sum([a for a in arA if a is not None])
            ai = tree_sum([a for a in aiA if a is not None])
            br = tree_sum([a for a in brA if a is not None])
            bi = tree_sum([a for a in biA if a is not None])
            out_store(k,   B.add(ar, bi), B.sub(ai, br))
            out_store(N-k, B.sub(ar, bi), B.add(ai, br))
    if ret_parts:
        bnd = getattr(E, 'stageA_end', len(E.body))
        args_only = args
        return args_only, head[1:], E.body[:bnd], E.body[bnd:]
    src = head + E.body + ["}"]
    return "\n".join(src)


def _relabel(lines, tag):
    """prefix SSA vars and scratch arrays with tag to avoid collisions"""
    import re
    out = []
    for l in lines:
        l = re.sub(r"\bv(\d+)\b", tag + r"v\1", l)
        l = re.sub(r"\bs(Ar|Ai|Pr|Pi|Qr|Qi)\b", tag + r"s\1", l)
        l = re.sub(r"\b(R|I)([0-7])\b", tag + r"\1\2", l)
        out.append(l)
    return out

def emit_codelet_x2(N, name, **kw):
    """two independent pencils, stage-interleaved: A1 A2 B1 B2 (scratch-decoupled)"""
    args, pre, A1, B1 = emit_codelet(N, "d", ret_parts=True, **kw)
    _, _, A2, B2 = emit_codelet(N, "d", ret_parts=True, **kw)
    A2 = _relabel(A2, "Z"); B2 = _relabel(B2, "Z")
    rep = [("pxr","qxr"),("pxi","qxi"),("pyr","qyr"),("pyi","qyi"),("pcr","qcr"),("pci","qci")]
    def sub(lines):
        out = []
        for l in lines:
            for a,b in rep: l = l.replace(a,b)
            out.append(l)
        return out
    A2 = sub(A2); B2 = sub(B2)
    args2 = args.replace("const double* pxr, const double* pxi, double* pyr, double* pyi",
                         "const double* pxr, const double* pxi, double* pyr, double* pyi, const double* qxr, const double* qxi, double* qyr, double* qyi")
    if "pcr" in args:
        args2 = args2.replace("const double* pcr, const double* pci", "const double* pcr, const double* pci, const double* qcr, const double* qci")
    lines = [f"static void {name}({args2}){{"] + pre + A1 + A2 + B1 + B2 + ["}"]
    return "\n".join(lines)
