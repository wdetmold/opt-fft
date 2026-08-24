import sys
sys.path.insert(0, '/tmp/dev')
from gen import *   # brings genlib + gen_kernel etc.

BCFG = {64: ('ct', 8, 8), 36: ('pfa', 9, 4), 45: ('pfa', 5, 9)}

def crt_pos(L):
    typ, n1, n2 = BCFG[L]
    pos = [0]*L
    if typ == 'ct':
        for a in range(8):
            for b in range(8):
                pos[8*a + b] = 8*b + a
    else:
        for k in range(L):
            pos[k] = (k % n1)*n2 + (k % n2)
    return pos

def gen_staged(L, W, kind, name, S, P=None):
    typ, n1, n2 = BCFG[L]
    pos = crt_pos(L)
    T = TYPE[W]
    out = []
    decl = []
    if kind == 'colT':
        sig = (f"static void {name}(const double*restrict pr, const double*restrict pi,"
               f" double*restrict dr, double*restrict di)")
    elif kind == 'colT_mapc':
        sig = (f"static void {name}(const double*restrict pr, const double*restrict pi,"
               f" const double*restrict cr, const double*restrict ci,"
               f" double*restrict dr, double*restrict di)")
    elif kind == 'xplain':
        sig = f"static void {name}(double*restrict pr, double*restrict pi)"
    else:
        sig = (f"static void {name}(double*restrict pr, double*restrict pi,"
               f" const double*restrict cr, const double*restrict ci)")
    out.append(sig + " {")
    out.append(f"  {T} Br[{L}] __attribute__((aligned(64)));")
    out.append(f"  {T} Bi[{L}] __attribute__((aligned(64)));")
    declmark = len(out)
    rowoff = (lambda j: pos[j]*P) if kind in ('colT','colT_mapc') else (lambda j: j*S)
    # ---- stage 1 ----
    for bidx in range(n2):
        g = G()
        inv = {}
        xin = []
        rows = ([ (a*n2 + bidx*n1) % L for a in range(n1) ] if typ == 'pfa'
                else [ 8*c + bidx for c in range(8) ])
        if kind == 'colT_mapc':
            out.append(f"  {T} m1v_{bidx}[{2*n1}] __attribute__((aligned(64)));")
            out.append(f"  {{")
            for t, j in enumerate(rows):
                o = rowoff(j)
                out.append(f"  {{ {T} zr = *(const {T}*)(pr + {o}) + *(const {T}*)(cr + {o});")
                out.append(f"    {T} zi = *(const {T}*)(pi + {o}) + *(const {T}*)(ci + {o});")
                out.append(f"    {T} mm = zr*zr + zi*zi;")
                out.append(f"    {T} uu = rsq{W}(mm);")
                out.append(f"    {T} wr_ = zr*uu, wi_ = zi*uu;")
                out.append(f"    {T} vv = rpc{W}(uu);")
                out.append(f"    m1v_{bidx}[{2*t}] = wr_*vv; m1v_{bidx}[{2*t+1}] = wi_*vv; }}")
            out.append(f"  }}")
            for t, j in enumerate(rows):
                inv[g.inp(('x', t, 0))] = f"m1v_{bidx}[{2*t}]"
                inv[g.inp(('x', t, 1))] = f"m1v_{bidx}[{2*t+1}]"
                xin.append((g.inp(('x', t, 0)), g.inp(('x', t, 1))))
        else:
            for t, j in enumerate(rows):
                o = rowoff(j)
                inv[g.inp(('x', t, 0))] = f"(*(const {T}*)(pr + {o}))"
                inv[g.inp(('x', t, 1))] = f"(*(const {T}*)(pi + {o}))"
                xin.append((g.inp(('x', t, 0)), g.inp(('x', t, 1))))
        ys = dft(g, n1, xin)
        if typ == 'ct':
            ys = [cmulc(g, ys[b], tw(L, bidx*b)) for b in range(n1)]
        lines, names = body_lines(g, ys, inv, W)
        out.append(f"  {{ // stage1 g{bidx}")
        out += lines
        for k1 in range(n1):
            slot = k1*n2 + bidx
            out.append(f"  Br[{slot}] = {names[ys[k1][0]]}; Bi[{slot}] = {names[ys[k1][1]]};")
        out.append("  }")
    # ---- stage 2 ----
    pending = []
    def flush(force=False):
        while pending and (len(pending) >= W or force):
            chunk = pending[:W]
            del pending[:W]
            p0 = chunk[0][0]
            wch = len(chunk)
            if W == 8:
                regs_r = ", ".join(f"(__m512d){c[1]}" for c in chunk) + (f", (__m512d){chunk[-1][1]}"*(8-wch))
                regs_i = ", ".join(f"(__m512d){c[2]}" for c in chunk) + (f", (__m512d){chunk[-1][2]}"*(8-wch))
                if wch == 8:
                    out.append(f"  tr8x8_store({regs_r}, dr + {p0}, {P});")
                    out.append(f"  tr8x8_store({regs_i}, di + {p0}, {P});")
                else:
                    out.append(f"  tr8x8_store_part({regs_r}, dr + {p0}, {P}, {wch});")
                    out.append(f"  tr8x8_store_part({regs_i}, di + {p0}, {P}, {wch});")
            elif W == 4:
                if wch == 4:
                    regs_r = ", ".join(f"(__m256d){c[1]}" for c in chunk)
                    regs_i = ", ".join(f"(__m256d){c[2]}" for c in chunk)
                    out.append(f"  tr4x4_store({regs_r}, dr + {p0}, {P});")
                    out.append(f"  tr4x4_store({regs_i}, di + {p0}, {P});")
                else:
                    for (pp, rn, iname) in chunk:
                        for lane in range(4):
                            out.append(f"  dr[{lane}*{P} + {pp}] = ((const double*)&{rn})[{lane}];")
                            out.append(f"  di[{lane}*{P} + {pp}] = ((const double*)&{iname})[{lane}];")
            elif W == 1:
                for (pp, rn, iname) in chunk:
                    out.append(f"  dr[{pp}] = {rn}; di[{pp}] = {iname};")
            if not pending: break
    for gidx in range(n1):
        g = G()
        inv = {}
        xin = []
        for t in range(n2):
            slot = gidx*n2 + t
            inv[g.inp(('x', t, 0))] = f"Br[{slot}]"
            inv[g.inp(('x', t, 1))] = f"Bi[{slot}]"
            xin.append((g.inp(('x', t, 0)), g.inp(('x', t, 1))))
        ys = dft(g, n2, xin)
        lines, names = body_lines(g, ys, inv, W)
        out.append(f"  {{ // stage2 g{gidx}")
        out += lines
        if kind == 'xplain':
            for t in range(n2):
                if typ == 'pfa':
                    k = next(kk for kk in range(L) if kk % n1 == gidx and kk % n2 == t)
                else:
                    k = 8*t + gidx
                off = k*S
                out.append(f"  *({T}*)(pr + {off}) = {names[ys[t][0]]};")
                out.append(f"  *({T}*)(pi + {off}) = {names[ys[t][1]]};")
            out.append("  }")
        elif kind == 'xfused':
            for t in range(n2):
                if typ == 'pfa':
                    k = next(kk for kk in range(L) if kk % n1 == gidx and kk % n2 == t)
                else:
                    k = 8*t + gidx
                off = k*S
                yr, yi = names[ys[t][0]], names[ys[t][1]]
                out.append(f"  {{ {T} zr = {yr} + *(const {T}*)(cr + {off});")
                out.append(f"    {T} zi = {yi} + *(const {T}*)(ci + {off});")
                out.append(f"    {T} mm = zr*zr + zi*zi;")
                out.append(f"    {T} rr = maprec{W}(mm);")
                out.append(f"    *({T}*)(pr + {off}) = zr*rr;")
                out.append(f"    *({T}*)(pi + {off}) = zi*rr; }}")
            out.append("  }")
        else:
            for t in range(n2):
                p = gidx*n2 + t
                decl.append(f"  {T} o{p}r, o{p}i;")
                out.append(f"  o{p}r = {names[ys[t][0]]}; o{p}i = {names[ys[t][1]]};")
                pending.append((p, f"o{p}r", f"o{p}i"))
            out.append("  }")
            flush()
    if kind in ('colT','colT_mapc'):
        flush(force=True)
    out.append("}")
    out[declmark:declmark] = decl
    return "\n".join(out)

EXTRA_HELPERS = r'''
static inline void tr8x8_store_part(__m512d r0,__m512d r1,__m512d r2,__m512d r3,
                               __m512d r4,__m512d r5,__m512d r6,__m512d r7,
                               double* dst, long P, int width){
  __m512d t0,t1,t2,t3,t4,t5,t6,t7, u0,u1,u2,u3,u4,u5,u6,u7;
  __mmask8 mk = (__mmask8)((1u<<width)-1u);
  t0=_mm512_unpacklo_pd(r0,r1); t1=_mm512_unpackhi_pd(r0,r1);
  t2=_mm512_unpacklo_pd(r2,r3); t3=_mm512_unpackhi_pd(r2,r3);
  t4=_mm512_unpacklo_pd(r4,r5); t5=_mm512_unpackhi_pd(r4,r5);
  t6=_mm512_unpacklo_pd(r6,r7); t7=_mm512_unpackhi_pd(r6,r7);
  u0=_mm512_shuffle_f64x2(t0,t2,0x88); u1=_mm512_shuffle_f64x2(t1,t3,0x88);
  u2=_mm512_shuffle_f64x2(t0,t2,0xDD); u3=_mm512_shuffle_f64x2(t1,t3,0xDD);
  u4=_mm512_shuffle_f64x2(t4,t6,0x88); u5=_mm512_shuffle_f64x2(t5,t7,0x88);
  u6=_mm512_shuffle_f64x2(t4,t6,0xDD); u7=_mm512_shuffle_f64x2(t5,t7,0xDD);
  _mm512_mask_storeu_pd(dst+0*P, mk, _mm512_shuffle_f64x2(u0,u4,0x88));
  _mm512_mask_storeu_pd(dst+1*P, mk, _mm512_shuffle_f64x2(u1,u5,0x88));
  _mm512_mask_storeu_pd(dst+2*P, mk, _mm512_shuffle_f64x2(u2,u6,0x88));
  _mm512_mask_storeu_pd(dst+3*P, mk, _mm512_shuffle_f64x2(u3,u7,0x88));
  _mm512_mask_storeu_pd(dst+4*P, mk, _mm512_shuffle_f64x2(u0,u4,0xDD));
  _mm512_mask_storeu_pd(dst+5*P, mk, _mm512_shuffle_f64x2(u1,u5,0xDD));
  _mm512_mask_storeu_pd(dst+6*P, mk, _mm512_shuffle_f64x2(u2,u6,0xDD));
  _mm512_mask_storeu_pd(dst+7*P, mk, _mm512_shuffle_f64x2(u3,u7,0xDD));
}
'''
print("gen2 ready")
