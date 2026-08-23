"""L=36 kernels v3: single scratch roundtrip for DFT9 (B merged), baked strides."""
import numpy as np
from glib import Emit, trig, hexd

L = 36
RS = 40
PS = 36*RS

def crt_maps():
    inp = [[(9*n1 + 4*n2) % 36 for n1 in range(4)] for n2 in range(9)]
    out = [[(9*k1 + 28*k2) % 36 for k2 in range(9)] for k1 in range(4)]
    return inp, out

S3 = hexd(np.sqrt(np.longdouble(3))/2)

def emit_dft4(e, xin, xout):
    (a_r,a_i),(b_r,b_i),(c_r,c_i),(d_r,d_i) = xin
    o = xout
    e(f"{{ V t0r=VADD({a_r},{c_r}), t0i=VADD({a_i},{c_i});")
    e(f"  V t1r=VSUB({a_r},{c_r}), t1i=VSUB({a_i},{c_i});")
    e(f"  V t2r=VADD({b_r},{d_r}), t2i=VADD({b_i},{d_i});")
    e(f"  V t3r=VSUB({b_r},{d_r}), t3i=VSUB({b_i},{d_i});")
    e(f"  {o[0][0]}=VADD(t0r,t2r); {o[0][1]}=VADD(t0i,t2i);")
    e(f"  {o[2][0]}=VSUB(t0r,t2r); {o[2][1]}=VSUB(t0i,t2i);")
    e(f"  {o[1][0]}=VADD(t1r,t3i); {o[1][1]}=VSUB(t1i,t3r);")
    e(f"  {o[3][0]}=VSUB(t1r,t3i); {o[3][1]}=VADD(t1i,t3r);")
    e("}")

def emit_dft3(e, xin, xout):
    (a_r,a_i),(b_r,b_i),(c_r,c_i) = xin
    o = xout
    e(f"{{ V ur=VADD({b_r},{c_r}), ui=VADD({b_i},{c_i});")
    e(f"  V vr=VSUB({b_r},{c_r}), vi=VSUB({b_i},{c_i});")
    e(f"  V sr=VFMA(ur,VSET1(-0.5),{a_r}), si=VFMA(ui,VSET1(-0.5),{a_i});")
    e(f"  {o[0][0]}=VADD({a_r},ur); {o[0][1]}=VADD({a_i},ui);")
    e(f"  {o[1][0]}=VFMA(vi,VSET1({S3}),sr);")
    e(f"  {o[1][1]}=VFNMA(vr,VSET1({S3}),si);")
    e(f"  {o[2][0]}=VFNMA(vi,VSET1({S3}),sr);")
    e(f"  {o[2][1]}=VFMA(vr,VSET1({S3}),si);")
    e("}")

def emit_dft9_block(e, k1, loadexpr, store_cb):
    """full DFT9 (CT 3x3) for column k1; loadexpr(g) -> (re,im) exprs; store_cb(e, kout, re, im)."""
    cs9 = trig(9)
    names = []
    for n2 in range(3):
        e("{")
        xin = []
        for n1 in range(3):
            g = 3*n1 + n2
            re_e, im_e = loadexpr(g)
            e(f"V a{n1}r = {re_e}, a{n1}i = {im_e};")
            xin.append((f"a{n1}r", f"a{n1}i"))
        xo = [(f"y{n2}_{q}r", f"y{n2}_{q}i") for q in range(3)]
        e.lines.insert(len(e.lines)-3-2*0, "")  # noop keep
        # declare outs at fn scope: emit declarations outside the block
        e.ind -= 0
        e("}")
        # declarations must be before block; simpler: declare first
        names.append(xo)
    # restart clean: (this function rebuilt below)
    raise SystemExit("unused")

def gen():
    e = Emit()
    inp, out = crt_maps()
    cs9 = trig(9)
    e(f"// ---------------- L=36 engine v3 ----------------")
    e(f"#define RS36 {RS}")
    e(f"#define PS36 {PS}")
    e(f"static double ARENA36_R[{36*PS}] ALIGN64;")
    e(f"static double ARENA36_I[{36*PS}] ALIGN64;")
    e(f"static double SCRA[{36*16}] ALIGN64;")
    e(f"static double ZSCR_R[{40*8}] ALIGN64;")
    e(f"static double ZSCR_I[{40*8}] ALIGN64;")

    def stage_A(stride):
        # 9 x DFT4 from rows -> SCRA[g*4+n1]
        for g in range(9):
            rows = inp[g]
            e(f"{{")
            xin = []
            for t, j in enumerate(rows):
                e(f"V a{t}r = VL(pr + {j*stride}), a{t}i = VL(pi + {j*stride});")
                xin.append((f"a{t}r", f"a{t}i"))
            xo = [(f"o{t}r", f"o{t}i") for t in range(4)]
            e("V " + ", ".join(f"{r}, {i}" for r,i in xo) + ";")
            emit_dft4(e, xin, xo)
            for n1 in range(4):
                e(f"VS(SCRA + {(g*4+n1)*16}, o{n1}r); VS(SCRA + {(g*4+n1)*16 + 8}, o{n1}i);")
            e("}")

    def stage_B(stride, store_cb):
        # full DFT9 per k1 from SCRA -> rows via store_cb
        for k1 in range(4):
            e("{")
            # first layer: 3 DFT3s over n1, for n2=0..2; declare y regs
            ynames = [[(f"y{n2}_{q}r", f"y{n2}_{q}i") for q in range(3)] for n2 in range(3)]
            decl = ", ".join(f"{r}, {i}" for row in ynames for (r,i) in row)
            e(f"V {decl};")
            for n2 in range(3):
                e("{")
                xin = []
                for n1 in range(3):
                    g = 3*n1 + n2
                    e(f"V b{n1}r = VL(SCRA + {(g*4+k1)*16}), b{n1}i = VL(SCRA + {(g*4+k1)*16+8});")
                    xin.append((f"b{n1}r", f"b{n1}i"))
                emit_dft3(e, xin, ynames[n2])
                # twiddle y[n2][kp] *= w9^{n2*kp}
                for kp in range(3):
                    t = (n2*kp) % 9
                    if t:
                        wr, wi = cs9[t]
                        r, i = ynames[n2][kp]
                        e(f"{{ V tr=VFNMA({i},VSET1({hexd(wi)}),VMUL({r},VSET1({hexd(wr)})));")
                        e(f"  {i}=VFMA({r},VSET1({hexd(wi)}),VMUL({i},VSET1({hexd(wr)}))); {r}=tr; }}")
                e("}")
            # second layer: for kp: DFT3 over n2 of y[n2][kp] -> X[3*q2+kp]
            for kp in range(3):
                e("{")
                xin = [ynames[n2][kp] for n2 in range(3)]
                xo = [(f"X{q}r", f"X{q}i") for q in range(3)]
                e("V " + ", ".join(f"{r}, {i}" for r,i in xo) + ";")
                emit_dft3(e, xin, xo)
                for q2 in range(3):
                    k9 = 3*q2 + kp
                    kout = out[k1][k9]
                    store_cb(kout, f"X{q2}r", f"X{q2}i")
                e("}")
            e("}")

    def plain_store(stride):
        def cb(k, r, i):
            e(f"VS(pr + {k*stride}, {r}); VS(pi + {k*stride}, {i});")
        return cb

    def map_store(stride):
        def cb(k, r, i):
            e(f"{{ const double* cs_ = cbase + 2*({k}*sitestep);")
            e(f"  if (!tail) {{ V ca_=VLU(cs_), cb_=VLU(cs_+8);")
            e(f"    {r} = VADD({r}, PERM2(ca_, IDXR_, cb_)); {i} = VADD({i}, PERM2(ca_, IDXI_, cb_)); }}")
            e(f"  else {{ V ca_=VLU(cs_);")
            e(f"    {r} = VADD({r}, PERM2Z(0x0F, ca_, IDXR_, ca_)); {i} = VADD({i}, PERM2Z(0x0F, ca_, IDXI_, ca_)); }}")
            e(f"  MAP2({r}, {i});")
            e(f"  VS(pr + {k*stride}, {r}); VS(pi + {k*stride}, {i});")
            e(f"  if (snap) {{")
            e(f"    double* p_ = o1 + 2*({k}*sitestep);")
            e(f"    VSU(p_, PERM2({r}, IDXLO_, {i})); if(!tail) VSU(p_+8, PERM2({r}, IDXHI_, {i}));")
            e(f"    if (om) {{ double* q_ = om + 2*({k}*sitestep);")
            e(f"      VSU(q_, PERM2({r}, IDXLO_, {i})); if(!tail) VSU(q_+8, PERM2({r}, IDXHI_, {i})); }}")
            e(f"  }}")
            e("}")
        return cb

    # generate per-stride function pairs
    for tag, stride in (("ps", PS), ("rs", RS), ("z", 8)):
        e(f"static __attribute__((noinline)) void st36A_{tag}(const double* pr, const double* pi){{")
        e.ind += 1; stage_A(stride); e.ind -= 1
        e("}")
        e(f"static __attribute__((noinline)) void st36B_{tag}(double* pr, double* pi){{")
        e.ind += 1; stage_B(stride, plain_store(stride)); e.ind -= 1
        e("}")
    for tag, stride in (("ps", PS), ("rs", RS)):
        e(f"static __attribute__((noinline)) void st36Bmap_{tag}(double* pr, double* pi,")
        e(f"        const double* cbase, double* o1, double* om, long sitestep, int tail, int snap){{")
        e.ind += 1; stage_B(stride, map_store(stride)); e.ind -= 1
        e("}")

    e("""
#define TR8(r0,r1,r2,r3,r4,r5,r6,r7) do { \\
    V u0=_mm512_unpacklo_pd(r0,r1), u1=_mm512_unpackhi_pd(r0,r1); \\
    V u2=_mm512_unpacklo_pd(r2,r3), u3=_mm512_unpackhi_pd(r2,r3); \\
    V u4=_mm512_unpacklo_pd(r4,r5), u5=_mm512_unpackhi_pd(r4,r5); \\
    V u6=_mm512_unpacklo_pd(r6,r7), u7=_mm512_unpackhi_pd(r6,r7); \\
    V s0=_mm512_shuffle_f64x2(u0,u2,0x88), s1=_mm512_shuffle_f64x2(u1,u3,0x88); \\
    V s2=_mm512_shuffle_f64x2(u0,u2,0xdd), s3=_mm512_shuffle_f64x2(u1,u3,0xdd); \\
    V s4=_mm512_shuffle_f64x2(u4,u6,0x88), s5=_mm512_shuffle_f64x2(u5,u7,0x88); \\
    V s6=_mm512_shuffle_f64x2(u4,u6,0xdd), s7=_mm512_shuffle_f64x2(u5,u7,0xdd); \\
    r0=_mm512_shuffle_f64x2(s0,s4,0x88); r4=_mm512_shuffle_f64x2(s0,s4,0xdd); \\
    r1=_mm512_shuffle_f64x2(s1,s5,0x88); r5=_mm512_shuffle_f64x2(s1,s5,0xdd); \\
    r2=_mm512_shuffle_f64x2(s2,s6,0x88); r6=_mm512_shuffle_f64x2(s2,s6,0xdd); \\
    r3=_mm512_shuffle_f64x2(s3,s7,0x88); r7=_mm512_shuffle_f64x2(s3,s7,0xdd); \\
} while(0)
""")
    for tag, stride in (("ps", PS), ("rs", RS)):
        e(f"static __attribute__((noinline)) void zpass36_{tag}(double* pr, double* pi, int nrows){{")
        e.ind += 1
        e("for (int t = 0; t < 5; t++) {")
        e.ind += 1
        e("V r0,r1,r2,r3,r4,r5,r6,r7;")
        for ch in "RI":
            arr = "pr" if ch=="R" else "pi"
            dst = "ZSCR_R" if ch=="R" else "ZSCR_I"
            for k in range(8):
                e(f"r{k} = (nrows > {k}) ? VL({arr} + {k*stride} + t*8) : _mm512_setzero_pd();")
            e("TR8(r0,r1,r2,r3,r4,r5,r6,r7);")
            for k in range(8):
                e(f"VS({dst} + (t*8+{k})*8, r{k});")
        e.ind -= 1
        e("}")
        e("st36A_z(ZSCR_R, ZSCR_I); st36B_z(ZSCR_R, ZSCR_I);")
        e("for (int t = 0; t < 5; t++) {")
        e.ind += 1
        e("V r0,r1,r2,r3,r4,r5,r6,r7;")
        for ch in "RI":
            arr = "pr" if ch=="R" else "pi"
            src = "ZSCR_R" if ch=="R" else "ZSCR_I"
            for k in range(8):
                e(f"r{k} = VL({src} + (t*8+{k})*8);")
            e("TR8(r0,r1,r2,r3,r4,r5,r6,r7);")
            for k in range(8):
                e(f"if (nrows > {k}) VS({arr} + {k*stride} + t*8, r{k});")
        e.ind -= 1
        e("}")
        e.ind -= 1
        e("}")
    return e.text()

if __name__ == "__main__":
    print(gen())
