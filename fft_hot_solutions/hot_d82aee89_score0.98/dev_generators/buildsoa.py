import numpy as np
from glib import Emit, PRELUDE, hexd
import gensoa, gensoadrv, genprime, gensmall

TR8MACRO = """
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
"""

def prime_passes(p):
    def f(e, L):
        assert L == p
        genprime.emit_tables(e, p)
        kb = genprime.KBLOCKS[p]
        genprime.emit_prime_func(e, p, "zz", 8, kb, False)
        genprime.emit_prime_func(e, p, "yy", L*8, kb, False)
        genprime.emit_prime_func(e, p, "xx", L*L*8, kb, False)
        genprime.emit_prime_func(e, p, "xx", L*L*8, kb, True, cstride=L*L*8)
        genprime.emit_prime_func(e, p, "yy", L*8, kb, True, cstride=L*8)
    return f

def small_passes(L):
    emitfn = gensmall.emit_dft6 if L == 6 else gensmall.emit_dft8
    p = L
    def f(e, LL):
        for tag, stride in (("zz", 8), ("yy", L*8), ("xx", L*L*8)):
            e(f"static __attribute__((noinline)) void p{p}_{tag}(double* PR, double* PI, long n, long pstep){{")
            e.ind += 1
            e("#pragma GCC unroll 1")
            e("for (long q_ = 0; q_ < n; q_++) {")
            e.ind += 1
            e("double* pr = PR + q_*pstep; double* pi = PI + q_*pstep;")
            def cb(k, r, i, stride=stride):
                e(f"VS(pr + {k*stride}, {r}); VS(pi + {k*stride}, {i});")
            emitfn(e, stride, cb)
            e.ind -= 1
            e("}")
            e.ind -= 1
            e("}")
        for tag, stride in (("xx", L*L*8), ("yy", L*8)):
            e(f"static __attribute__((noinline)) void p{p}_{tag}m(double* PR, double* PI, const double* CR, const double* CI, long n, long pstep){{")
            e.ind += 1
            e("#pragma GCC unroll 1")
            e("for (long q_ = 0; q_ < n; q_++) {")
            e.ind += 1
            e("double* pr = PR + q_*pstep; double* pi = PI + q_*pstep;")
            e("const double* cr = CR + q_*pstep; const double* ci = CI + q_*pstep;")
            def cbm(k, r, i, stride=stride):
                e(f"{{ V zr_ = VADD({r}, VL(cr + {k*stride})), zi_ = VADD({i}, VL(ci + {k*stride}));")
                e(f"  MAP2(zr_, zi_);")
                e(f"  VS(pr + {k*stride}, zr_); VS(pi + {k*stride}, zi_); }}")
            emitfn(e, stride, cbm)
            e.ind -= 1
            e("}")
            e.ind -= 1
            e("}")
    return f

def build():
    e = Emit()
    e(PRELUDE)
    e(TR8MACRO)
    e("static double PSCR[400] ALIGN64;")
    e("static double USCR[400] ALIGN64;")
    gensoa.emit_conv(e)
    for p in (13, 17, 23):
        gensoadrv.emit_soa_engine(e, p, prime_passes(p))
    for L in (6, 8):
        gensoadrv.emit_soa_engine(e, L, small_passes(L))
    return e.text()

if __name__ == "__main__":
    open("../implsoa.c","w").write(build())
    print("wrote implsoa.c")
