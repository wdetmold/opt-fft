"""L=36 SoA-8 prototype."""
import numpy as np
from glib import Emit, trig, hexd, PRELUDE
from gen36 import emit_dft4, emit_dft3, crt_maps, S3
import gensoa

L = 36
N3 = L*L*L
cs9 = trig(9)

def stage_A(e, stride):
    inp, out = crt_maps()
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

def stage_B(e, stride, store_cb):
    inp, out = crt_maps()
    for k1 in range(4):
        e("{")
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
            for kp in range(3):
                t = (n2*kp) % 9
                if t:
                    wr, wi = cs9[t]
                    r, i = ynames[n2][kp]
                    e(f"{{ V tr=VFNMA({i},VSET1({hexd(wi)}),VMUL({r},VSET1({hexd(wr)})));")
                    e(f"  {i}=VFMA({r},VSET1({hexd(wi)}),VMUL({i},VSET1({hexd(wr)}))); {r}=tr; }}")
            e("}")
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

def gen():
    e = Emit()
    e("// ============ L=36 SoA-8 ============")
    e(f"static double S36RE[{N3*8}] ALIGN64;")
    e(f"static double S36IM[{N3*8}] ALIGN64;")
    e(f"static double C36RE[{N3*8}] ALIGN64;")
    e(f"static double C36IM[{N3*8}] ALIGN64;")
    e(f"static double SCRA[{36*16}] ALIGN64;")
    gensoa.emit_conv(e)
    # strides in doubles (x: 36*36*8, y: 36*8, z: 8)
    for tag, stride in (("x", 36*36*8), ("y", 36*8), ("z", 8)):
        e(f"static __attribute__((noinline)) void sA36{tag}(const double* pr, const double* pi){{")
        e.ind += 1; stage_A(e, stride); e.ind -= 1
        e("}")
        e(f"static __attribute__((noinline)) void sB36{tag}(double* pr, double* pi){{")
        e.ind += 1
        def cb(k, r, i, stride=stride):
            e(f"VS(pr + {k*stride}, {r}); VS(pi + {k*stride}, {i});")
        stage_B(e, stride, cb)
        e.ind -= 1
        e("}")
    # Bmap variants (fused +c, map) for completing axes x and y
    for tag, stride in (("x", 36*36*8), ("y", 36*8)):
        e(f"static __attribute__((noinline)) void sBmap36{tag}(double* pr, double* pi, const double* cr, const double* ci){{")
        e.ind += 1
        def cbm(k, r, i, stride=stride):
            e(f"{{ {r} = VADD({r}, VL(cr + {k*stride})); {i} = VADD({i}, VL(ci + {k*stride}));")
            e(f"  MAP2({r}, {i});")
            e(f"  VS(pr + {k*stride}, {r}); VS(pi + {k*stride}, {i}); }}")
        stage_B(e, stride, cbm)
        e.ind -= 1
        e("}")
    e("""
// sweep SyX: slab y0: pencils (x) for each z; complete X_t + map; pre: Z,X
static void sw36_SyX(int y0, int pre){
    for (int z = 0; z < 36; z++) {
        long off = ((long)y0*36 + z)*8;
        sA36x(S36RE + off, S36IM + off);
        sBmap36x(S36RE + off, S36IM + off, C36RE + off, C36IM + off);
    }
    if (pre) {
        for (int x = 0; x < 36; x++) {
            long off = ((long)x*36 + y0)*36*8;
            sA36z(S36RE + off, S36IM + off); sB36z(S36RE + off, S36IM + off);
        }
        for (int z = 0; z < 36; z++) {
            long off = ((long)y0*36 + z)*8;
            sA36x(S36RE + off, S36IM + off); sB36x(S36RE + off, S36IM + off);
        }
    }
}
// sweep PxY: plane x0: pencils (y) for each z; complete Y_t + map; pre: Z,Y
static void sw36_PxY(int x0, int pre){
    for (int z = 0; z < 36; z++) {
        long off = ((long)x0*36*36 + z)*8;
        sA36y(S36RE + off, S36IM + off);
        sBmap36y(S36RE + off, S36IM + off, C36RE + off, C36IM + off);
    }
    if (pre) {
        for (int y = 0; y < 36; y++) {
            long off = ((long)x0*36 + y)*36*8;
            sA36z(S36RE + off, S36IM + off); sB36z(S36RE + off, S36IM + off);
        }
        for (int z = 0; z < 36; z++) {
            long off = ((long)x0*36*36 + z)*8;
            sA36y(S36RE + off, S36IM + off); sB36y(S36RE + off, S36IM + off);
        }
    }
}
void run_36soa(const double* x0, const double* c, double* out1, double* outm, long B, long m){
    if (m < 1) m = 1;
    long vs = 2*(long)46656;
    for (long g0 = 0; g0 < B; g0 += 8) {
        int nv = (B - g0) < 8 ? (int)(B - g0) : 8;
        const double* xg = x0 + g0*vs;
        const double* cg = c + g0*vs;
        // convert in (state + c)
        if (nv == 8) {
            for (long s = 0; s < 46656; s += 8) soa_in8(xg, vs, s, S36RE, S36IM);
            for (long s = 0; s < 46656; s += 8) soa_in8(cg, vs, s, C36RE, C36IM);
        } else {
            for (long s = 0; s < 46656; s += 8) soa_in8_nv(xg, vs, s, S36RE, S36IM, nv);
            for (long s = 0; s < 46656; s += 8) soa_in8_nv(cg, vs, s, C36RE, C36IM, nv);
        }
        // prologue: Z,Y per plane x
        for (int x = 0; x < 36; x++) {
            for (int y = 0; y < 36; y++) {
                long off = ((long)x*36 + y)*36*8;
                sA36z(S36RE + off, S36IM + off); sB36z(S36RE + off, S36IM + off);
            }
            for (int z = 0; z < 36; z++) {
                long off = ((long)x*36*36 + z)*8;
                sA36y(S36RE + off, S36IM + off); sB36y(S36RE + off, S36IM + off);
            }
        }
        for (long t = 1; t <= m; t++) {
            int snap = (t == 1) || (t == m);
            int pre = (t < m);
            int dopre = pre && !snap;
            if (t & 1) { for (int y0 = 0; y0 < 36; y0++) sw36_SyX(y0, dopre); }
            else       { for (int xp = 0; xp < 36; xp++) sw36_PxY(xp, dopre); }
            if (t == 1) for (long s = 0; s < 46656; s += 8)
                soa_out8(S36RE, S36IM, s, out1 + g0*vs, vs, nv);
            if (t == m) for (long s = 0; s < 46656; s += 8)
                soa_out8(S36RE, S36IM, s, outm + g0*vs, vs, nv);
            if (pre && !dopre) {  // catch-up pre after snapshot
                for (long p = 0; p < 1296; p++) {
                    long off = p*36*8;
                    sA36z(S36RE + off, S36IM + off); sB36z(S36RE + off, S36IM + off);
                }
                if (t & 1) {  // next sweep is PxY -> needs Y... wait next step t+1 even: completes with Y; pre for t+1 = Z (done) + Y? NO: SyX pre does Z,X; PxY pre does Z,Y.
                    for (int y0 = 0; y0 < 36; y0++) for (int z = 0; z < 36; z++) {
                        long off = ((long)y0*36 + z)*8;
                        sA36x(S36RE + off, S36IM + off); sB36x(S36RE + off, S36IM + off);
                    }
                } else {
                    for (int x0 = 0; x0 < 36; x0++) for (int z = 0; z < 36; z++) {
                        long off = ((long)x0*36*36 + z)*8;
                        sA36y(S36RE + off, S36IM + off); sB36y(S36RE + off, S36IM + off);
                    }
                }
            }
        }
    }
}
""")
    return e.text()

if __name__ == "__main__":
    src = PRELUDE + """
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
""" + gen()
    open("../impl36soa.c","w").write(src)
    print("wrote impl36soa.c")
