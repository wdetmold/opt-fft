"""Within-volume engine for L in {36,45,64}: split re/im padded rows, one-sweep-per-step.
Factorizations: 36=PFA(4,9), 45=PFA(9,5), 64=CT(8,8 with twiddles).
Stage A: inner DFT over n1-groups -> SCRA ; stage B: outer DFT (+twiddles for CT) -> rows (or map-fused).
"""
import numpy as np
from glib import Emit, trig, hexd
from gen36 import emit_dft4, emit_dft3, S3

def emit_dft5(e, xin, xout):
    # forward DFT5 via folded pairs, FMA-style (36 vops)
    c = trig(5)
    c1, s1 = hexd(c[1][0]), hexd(-c[1][1])   # cos(2pi/5), sin(2pi/5)
    c2, s2 = hexd(c[2][0]), hexd(-c[2][1])
    (x0r,x0i),(x1r,x1i),(x2r,x2i),(x3r,x3i),(x4r,x4i) = xin
    o = xout
    e(f"{{ V u1r=VADD({x1r},{x4r}), u1i=VADD({x1i},{x4i});")
    e(f"  V v1r=VSUB({x1r},{x4r}), v1i=VSUB({x1i},{x4i});")
    e(f"  V u2r=VADD({x2r},{x3r}), u2i=VADD({x2i},{x3i});")
    e(f"  V v2r=VSUB({x2r},{x3r}), v2i=VSUB({x2i},{x3i});")
    e(f"  {o[0][0]}=VADD({x0r},VADD(u1r,u2r)); {o[0][1]}=VADD({x0i},VADD(u1i,u2i));")
    e(f"  V s1r_=VFMA(u1r,VSET1({c1}),VFMA(u2r,VSET1({c2}),{x0r}));")
    e(f"  V s1i_=VFMA(u1i,VSET1({c1}),VFMA(u2i,VSET1({c2}),{x0i}));")
    e(f"  V s2r_=VFMA(u1r,VSET1({c2}),VFMA(u2r,VSET1({c1}),{x0r}));")
    e(f"  V s2i_=VFMA(u1i,VSET1({c2}),VFMA(u2i,VSET1({c1}),{x0i}));")
    e(f"  V t1r_=VFMA(v1r,VSET1({s1}),VMUL(v2r,VSET1({s2})));")
    e(f"  V t1i_=VFMA(v1i,VSET1({s1}),VMUL(v2i,VSET1({s2})));")
    e(f"  V t2r_=VFMS(v1r,VSET1({s2}),VMUL(v2r,VSET1({s1})));")
    e(f"  V t2i_=VFMS(v1i,VSET1({s2}),VMUL(v2i,VSET1({s1})));")
    # X1 = s1 - i t1 ; X4 = s1 + i t1 ; X2 = s2 - i t2 ; X3 = s2 + i t2
    e(f"  {o[1][0]}=VADD(s1r_,t1i_); {o[1][1]}=VSUB(s1i_,t1r_);")
    e(f"  {o[4][0]}=VSUB(s1r_,t1i_); {o[4][1]}=VADD(s1i_,t1r_);")
    e(f"  {o[2][0]}=VADD(s2r_,t2i_); {o[2][1]}=VSUB(s2i_,t2r_);")
    e(f"  {o[3][0]}=VSUB(s2r_,t2i_); {o[3][1]}=VADD(s2i_,t2r_);")
    e("}")

def emit_dft8_regs(e, xin, xout):
    import gensmall
    # reuse DIF DFT8 on named regs: adapt from gensmall by inlining here
    RT2H = hexd(np.sqrt(np.longdouble(2))/2)
    for j in range(4):
        a, b = xin[j], xin[j+4]
        e(f"V s{j}r = VADD({a[0]}, {b[0]}), s{j}i = VADD({a[1]}, {b[1]});")
        e(f"V d{j}r = VSUB({a[0]}, {b[0]}), d{j}i = VSUB({a[1]}, {b[1]});")
    e(f"{{ V c_ = VSET1({RT2H});")
    e(f"  V tr = VMUL(VADD(d1r, d1i), c_); V ti = VMUL(VSUB(d1i, d1r), c_); d1r = tr; d1i = ti; }}")
    e(f"{{ V tr = d2i; V ti = _mm512_sub_pd(_mm512_setzero_pd(), d2r); d2r = tr; d2i = ti; }}")
    e(f"{{ V c_ = VSET1({RT2H});")
    e(f"  V tr = VMUL(VSUB(d3i, d3r), c_); V ti = VMUL(VADD(d3r, d3i), _mm512_sub_pd(_mm512_setzero_pd(), c_)); d3r = tr; d3i = ti; }}")
    for (pfx, outks) in (("s", [0,2,4,6]), ("d", [1,3,5,7])):
        e(f"{{ V t0r=VADD({pfx}0r,{pfx}2r), t0i=VADD({pfx}0i,{pfx}2i);")
        e(f"  V t1r=VSUB({pfx}0r,{pfx}2r), t1i=VSUB({pfx}0i,{pfx}2i);")
        e(f"  V t2r=VADD({pfx}1r,{pfx}3r), t2i=VADD({pfx}1i,{pfx}3i);")
        e(f"  V t3r=VSUB({pfx}1r,{pfx}3r), t3i=VSUB({pfx}1i,{pfx}3i);")
        e(f"  {xout[outks[0]][0]}=VADD(t0r,t2r); {xout[outks[0]][1]}=VADD(t0i,t2i);")
        e(f"  {xout[outks[2]][0]}=VSUB(t0r,t2r); {xout[outks[2]][1]}=VSUB(t0i,t2i);")
        e(f"  {xout[outks[1]][0]}=VADD(t1r,t3i); {xout[outks[1]][1]}=VSUB(t1i,t3r);")
        e(f"  {xout[outks[3]][0]}=VSUB(t1r,t3i); {xout[outks[3]][1]}=VADD(t1i,t3r);")
        e("}")

CONFIGS = {
    36: dict(P=4, Q=9, pfa=True, RS=40, NCH=5, TAILN=4),
    45: dict(P=9, Q=5, pfa=True, RS=48, NCH=6, TAILN=5),
    64: dict(P=8, Q=8, pfa=False, RS=64, NCH=8, TAILN=8),
}
# PFA: n=(Q*n1+P*n2)%L over n1 in Z_P (stage A: DFT_P), n2 in Z_Q (stage B: DFT_Q)
# out k = (alpha*k1 + beta*k2) % L ; alpha = Q*inv(Q,P)... computed below
# CT(8x8): n = 8*n1+n2 ; stage A: DFT8 over n1 fixed n2; twiddle w64^(k1*n2); B: DFT8 over n2; out k=8*k2+k1

def inv(a, m):
    for t in range(1, m):
        if (a*t) % m == 1: return t
    raise ValueError

def maps_for(L):
    cfg = CONFIGS[L]
    P, Q = cfg['P'], cfg['Q']
    if cfg['pfa']:
        inp = [[(Q*n1 + P*n2) % L for n1 in range(P)] for n2 in range(Q)]
        alpha = Q * inv(Q % P, P)
        beta = P * inv(P % Q, Q)
        out = [[(alpha*k1 + beta*k2) % L for k2 in range(Q)] for k1 in range(P)]
    else:
        inp = [[8*n1 + n2 for n1 in range(P)] for n2 in range(Q)]
        out = [[8*k2 + k1 for k2 in range(Q)] for k1 in range(P)]
    return inp, out

def emit_inner(e, L, xin, xout):
    P = CONFIGS[L]['P']
    if L == 36: emit_dft4(e, xin, xout)
    elif L == 45:
        # stage A for 45 is DFT9 (P=9)
        emit_dft9_reg(e, xin, xout)
    else: emit_dft8_regs(e, xin, xout)

def emit_dft9_reg(e, xin, xout):
    cs9 = trig(9)
    ynames = [[(f"w{n2}_{q}r", f"w{n2}_{q}i") for q in range(3)] for n2 in range(3)]
    decl = ", ".join(f"{r}, {i}" for row in ynames for (r,i) in row)
    e(f"V {decl};")
    for n2 in range(3):
        xin3 = [xin[3*n1 + n2] for n1 in range(3)]
        emit_dft3(e, xin3, ynames[n2])
        for kp in range(3):
            t = (n2*kp) % 9
            if t:
                wr, wi = cs9[t]
                r, i = ynames[n2][kp]
                e(f"{{ V tw=VFNMA({i},VSET1({hexd(wi)}),VMUL({r},VSET1({hexd(wr)})));")
                e(f"  {i}=VFMA({r},VSET1({hexd(wi)}),VMUL({i},VSET1({hexd(wr)}))); {r}=tw; }}")
    for kp in range(3):
        xin3 = [ynames[n2][kp] for n2 in range(3)]
        xo = [xout[3*q2 + kp] for q2 in range(3)]
        emit_dft3(e, xin3, xo)

def emit_outer(e, L, k1, xin, xout):
    """stage B for column k1: DFT_Q with (CT) twiddles on inputs."""
    cfg = CONFIGS[L]
    if cfg['pfa']:
        if L == 36: emit_dft9_reg(e, xin, xout)
        else: emit_dft5(e, xin, xout)
    else:
        # twiddle inputs: xin[n2] *= w64^{k1*n2}
        cs = trig(64)
        for n2 in range(1, 8):
            t = (k1*n2) % 64
            if t == 0: continue
            wr, wi = cs[t]
            r, i = xin[n2]
            e(f"{{ V tw=VFNMA({i},VSET1({hexd(wi)}),VMUL({r},VSET1({hexd(wr)})));")
            e(f"  {i}=VFMA({r},VSET1({hexd(wi)}),VMUL({i},VSET1({hexd(wr)}))); {r}=tw; }}")
        emit_dft8_regs(e, xin, xout)

def gen_engine(L):
    """Full within-volume engine for L."""
    cfg = CONFIGS[L]
    P, Q, RS, NCH, TAILN = cfg['P'], cfg['Q'], cfg['RS'], cfg['NCH'], cfg['TAILN']
    PS = L*RS
    N3 = L*L*L
    inp, out = maps_for(L)
    e = Emit()
    e(f"// ============ L={L} within-volume ============")
    e(f"#define RS{L} {RS}")
    e(f"#define PS{L} {PS if L != 64 else L*RS + 8}")
    PSv = PS if L != 64 else L*RS + 8
    e(f"static double A{L}R[{L*PSv}] ALIGN64;")
    e(f"static double A{L}I[{L*PSv}] ALIGN64;")
    e(f"static double SC{L}[{L*16}] ALIGN64;")
    e(f"static double ZR{L}[{RS*8}] ALIGN64;")
    e(f"static double ZI{L}[{RS*8}] ALIGN64;")
    e(f"static double C{L}R[{L*PSv if L==64 else L*PS}] ALIGN64;")
    e(f"static double C{L}I[{L*PSv if L==64 else L*PS}] ALIGN64;")

    def stage_A(stride, pf=0):
        for g in range(Q):
            rows = inp[g]
            e("{")
            xin = []
            for t, j in enumerate(rows):
                e(f"V a{t}r = VL(pr + {j}*{stride}), a{t}i = VL(pi + {j}*{stride});")
                if pf:
                    e(f"_mm_prefetch((const char*)(pr + {j}*{stride} + {pf}), _MM_HINT_T0);")
                    e(f"_mm_prefetch((const char*)(pi + {j}*{stride} + {pf}), _MM_HINT_T0);")
                xin.append((f"a{t}r", f"a{t}i"))
            xo = [(f"o{t}r", f"o{t}i") for t in range(P)]
            e("V " + ", ".join(f"{r}, {i}" for r,i in xo) + ";")
            emit_inner(e, L, xin, xo)
            for n1 in range(P):
                e(f"VS(SC{L} + {(g*P+n1)*16}, o{n1}r); VS(SC{L} + {(g*P+n1)*16+8}, o{n1}i);")
            e("}")

    def stage_B(stride, store_cb):
        for k1 in range(P):
            e("{")
            xin = []
            for g in range(Q):
                e(f"V b{g}r = VL(SC{L} + {(g*P+k1)*16}), b{g}i = VL(SC{L} + {(g*P+k1)*16+8});")
                xin.append((f"b{g}r", f"b{g}i"))
            xo = [(f"X{t}r", f"X{t}i") for t in range(Q)]
            e("V " + ", ".join(f"{r}, {i}" for r,i in xo) + ";")
            emit_outer(e, L, k1, xin, xo)
            for k2 in range(Q):
                store_cb(out[k1][k2], f"X{k2}r", f"X{k2}i")
            e("}")

    # pass functions: loop over chunks inside (n, rowbase handled by caller loop for A/B pair)
    for tag, stride in (("ps", PSv), ("rs", RS)):
        e(f"static __attribute__((noinline)) void wv{L}A_{tag}(const double* pr, const double* pi){{")
        e.ind += 1; stage_A(stride, pf=0); e.ind -= 1
        e("}")
        e(f"static __attribute__((noinline)) void wv{L}B_{tag}(double* pr, double* pi){{")
        e.ind += 1
        def cb(k, r, i, stride=stride):
            e(f"VS(pr + {k}*{stride}, {r}); VS(pi + {k}*{stride}, {i});")
        stage_B(stride, cb)
        e.ind -= 1
        e("}")
    # z-variant on scratch (stride 8)
    e(f"static __attribute__((noinline)) void wv{L}A_z(const double* pr, const double* pi){{")
    e.ind += 1; stage_A(8); e.ind -= 1
    e("}")
    e(f"static __attribute__((noinline)) void wv{L}B_z(double* pr, double* pi){{")
    e.ind += 1
    def cbz(k, r, i):
        e(f"VS(pr + {k}*8, {r}); VS(pi + {k}*8, {i});")
    stage_B(8, cbz)
    e.ind -= 1
    e("}")
    # map-fused B variants, 3 snap modes (0: none, 1: one buffer, 2: two buffers)
    for tag, stride in (("ps", PSv), ("rs", RS)):
        for sm in (0, 1, 2):
            e(f"static __attribute__((noinline)) void wv{L}Bm{sm}_{tag}(double* pr, double* pi,")
            e(f"        const double* ccr, const double* cci, long sitestep{', double* o1' if sm>=1 else ''}{', double* om' if sm>=2 else ''}, int tail){{")
            e.ind += 1
            def cbm(k, r, i, stride=stride, sm=sm):
                e(f"{{")
                e(f"  {r} = VADD({r}, VL(ccr + {k}*{stride})); {i} = VADD({i}, VL(cci + {k}*{stride}));")
                e(f"  MAP2({r}, {i});")
                e(f"  VS(pr + {k}*{stride}, {r}); VS(pi + {k}*{stride}, {i});")
                hmask = hex((1 << max(0, 2*(TAILN-4))) - 1)
                for buf in (["o1"] if sm>=1 else []) + (["om"] if sm>=2 else []):
                    e(f"  {{ double* p_ = {buf} + 2*({k}*sitestep);")
                    e(f"    VSU(p_, PERM2({r}, IDXLO_, {i}));")
                    e(f"    if(!tail) VSU(p_+8, PERM2({r}, IDXHI_, {i}));")
                    if TAILN > 4:
                        e(f"    else _mm512_mask_storeu_pd(p_+8, {hmask}, PERM2({r}, IDXHI_, {i}));")
                    e(f"  }}")
                e("}")
            stage_B(stride, cbm)
            e.ind -= 1
            e("}")
    # zpass bundle: NT tiles of 8 rows x NCH tiles
    for tag, stride in (("ps", PSv), ("rs", RS)):
        e(f"static __attribute__((noinline)) void wvz{L}_{tag}(double* pr, double* pi, int nrows){{")
        e.ind += 1
        e(f"for (int t = 0; t < {NCH}; t++) {{")
        e.ind += 1
        e("V r0,r1,r2,r3,r4,r5,r6,r7;")
        for ch in "RI":
            arr = "pr" if ch=="R" else "pi"
            dst = f"ZR{L}" if ch=="R" else f"ZI{L}"
            for k in range(8):
                e(f"r{k} = (nrows > {k}) ? VL({arr} + {k}*{stride} + t*8) : _mm512_setzero_pd();")
            e("TR8(r0,r1,r2,r3,r4,r5,r6,r7);")
            for k in range(8):
                e(f"VS({dst} + (t*8+{k})*8, r{k});")
        e.ind -= 1
        e("}")
        e(f"wv{L}A_z(ZR{L}, ZI{L}); wv{L}B_z(ZR{L}, ZI{L});")
        e(f"for (int t = 0; t < {NCH}; t++) {{")
        e.ind += 1
        e("V r0,r1,r2,r3,r4,r5,r6,r7;")
        for ch in "RI":
            arr = "pr" if ch=="R" else "pi"
            src = f"ZR{L}" if ch=="R" else f"ZI{L}"
            for k in range(8):
                e(f"r{k} = VL({src} + (t*8+{k})*8);")
            e("TR8(r0,r1,r2,r3,r4,r5,r6,r7);")
            for k in range(8):
                e(f"if (nrows > {k}) VS({arr} + {k}*{stride} + t*8, r{k});")
        e.ind -= 1
        e("}")
        e.ind -= 1
        e("}")
    # conversion row in
    e(f"static inline __attribute__((always_inline)) void row{L}_in(const double* s, double* pr, double* pi){{")
    e.ind += 1
    full = L // 8
    for g in range(full):
        e(f"{{ V a=VLU(s+{g*16}), b=VLU(s+{g*16+8});")
        e(f"  VS(pr+{g*8}, PERM2(a, IDXR_, b)); VS(pi+{g*8}, PERM2(a, IDXI_, b)); }}")
    if L % 8:
        bmask = hex((1 << max(0, 2*(TAILN-4))) - 1)
        e(f"{{ V a=VLU(s+{full*16}); V b=_mm512_maskz_loadu_pd({bmask}, s+{full*16+8});")
        e(f"  VS(pr+{full*8}, PERM2Z({hex((1<<TAILN)-1)}, a, IDXR_, b)); VS(pi+{full*8}, PERM2Z({hex((1<<TAILN)-1)}, a, IDXI_, b)); }}")
    e.ind -= 1
    e("}")
    # sweeps + driver
    nb = (L + 7)//8
    lastn = L - 8*(nb-1)
    e(f"""
static void sw{L}_SyX(int y0, const double* cvol, double* o1, double* om, int pre){{
    double* br = A{L}R + (long)y0*RS{L};
    double* bi = A{L}I + (long)y0*RS{L};
    for (int ch = 0; ch < {NCH}; ch++) {{
        long so = (long)y0*{L} + ch*8;
        int tail = (ch == {NCH-1}) && {1 if L % 8 else 0};
        const double* ccr = C{L}R + (long)y0*RS{L} + ch*8;
        const double* cci = C{L}I + (long)y0*RS{L} + ch*8;
        wv{L}A_ps(br + ch*8, bi + ch*8);
        if (o1 && om) wv{L}Bm2_ps(br + ch*8, bi + ch*8, ccr, cci, {L*L}, o1 + 2*so, om + 2*so, tail);
        else if (o1)  wv{L}Bm1_ps(br + ch*8, bi + ch*8, ccr, cci, {L*L}, o1 + 2*so, tail);
        else if (om)  wv{L}Bm1_ps(br + ch*8, bi + ch*8, ccr, cci, {L*L}, om + 2*so, tail);
        else          wv{L}Bm0_ps(br + ch*8, bi + ch*8, ccr, cci, {L*L}, tail);
    }}
    if (pre) {{
        for (int b = 0; b < {nb}; b++) wvz{L}_ps(br + (long)b*8*PS{L}, bi + (long)b*8*PS{L}, b<{nb-1}?8:{lastn});
        for (int ch = 0; ch < {NCH}; ch++) {{ wv{L}A_ps(br + ch*8, bi + ch*8); wv{L}B_ps(br + ch*8, bi + ch*8); }}
    }}
}}
static void sw{L}_PxY(int x0, const double* cvol, double* o1, double* om, int pre){{
    double* br = A{L}R + (long)x0*PS{L};
    double* bi = A{L}I + (long)x0*PS{L};
    for (int ch = 0; ch < {NCH}; ch++) {{
        long so = (long)x0*{L*L} + ch*8;
        int tail = (ch == {NCH-1}) && {1 if L % 8 else 0};
        const double* ccr = C{L}R + (long)x0*PS{L} + ch*8;
        const double* cci = C{L}I + (long)x0*PS{L} + ch*8;
        wv{L}A_rs(br + ch*8, bi + ch*8);
        if (o1 && om) wv{L}Bm2_rs(br + ch*8, bi + ch*8, ccr, cci, {L}, o1 + 2*so, om + 2*so, tail);
        else if (o1)  wv{L}Bm1_rs(br + ch*8, bi + ch*8, ccr, cci, {L}, o1 + 2*so, tail);
        else if (om)  wv{L}Bm1_rs(br + ch*8, bi + ch*8, ccr, cci, {L}, om + 2*so, tail);
        else          wv{L}Bm0_rs(br + ch*8, bi + ch*8, ccr, cci, {L}, tail);
    }}
    if (pre) {{
        for (int b = 0; b < {nb}; b++) wvz{L}_rs(br + (long)b*8*RS{L}, bi + (long)b*8*RS{L}, b<{nb-1}?8:{lastn});
        for (int ch = 0; ch < {NCH}; ch++) {{ wv{L}A_rs(br + ch*8, bi + ch*8); wv{L}B_rs(br + ch*8, bi + ch*8); }}
    }}
}}
void run_{L}wv(const double* x0, const double* c, double* out1, double* outm, long B, long m){{
    if (m < 1) m = 1;
    for (long b = 0; b < B; b++) {{
        const double* xv = x0 + b*2*{N3};
        const double* cv = c + b*2*{N3};
        double* o1 = out1 + b*2*{N3};
        double* om = outm + b*2*{N3};
        for (int x = 0; x < {L}; x++)
            for (int y = 0; y < {L}; y++) {{
                row{L}_in(xv + ((long)x*{L}+y)*{2*L}, A{L}R + (long)x*PS{L} + y*RS{L},
                         A{L}I + (long)x*PS{L} + y*RS{L});
                row{L}_in(cv + ((long)x*{L}+y)*{2*L}, C{L}R + (long)x*PS{L} + y*RS{L},
                         C{L}I + (long)x*PS{L} + y*RS{L});
            }}
        for (int x = 0; x < {L}; x++) {{
            double* br = A{L}R + (long)x*PS{L}; double* bi = A{L}I + (long)x*PS{L};
            for (int bnd = 0; bnd < {nb}; bnd++) wvz{L}_rs(br + (long)bnd*8*RS{L}, bi + (long)bnd*8*RS{L}, bnd<{nb-1}?8:{lastn});
            for (int ch = 0; ch < {NCH}; ch++) {{ wv{L}A_rs(br + ch*8, bi + ch*8); wv{L}B_rs(br + ch*8, bi + ch*8); }}
        }}
        for (long t = 1; t <= m; t++) {{
            double* o1t = (t==1) ? o1 : 0;
            double* omt = (t==m) ? om : 0;
            int pre = (t < m);
            if (t & 1) {{ for (int y0 = 0; y0 < {L}; y0++) sw{L}_SyX(y0, cv, o1t, omt, pre); }}
            else       {{ for (int xp = 0; xp < {L}; xp++) sw{L}_PxY(xp, cv, o1t, omt, pre); }}
        }}
    }}
}}
""")
    return e.text()
