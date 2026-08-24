from glib import Emit, PRELUDE
import gensoa, gensoadrv2, gensmall
TR8MACRO = open('tr8.inc').read()
XSMAP = {6: 36*8, 8: 576}
def small_passes(L):
    emitfn = gensmall.emit_dft6 if L == 6 else gensmall.emit_dft8
    p = L
    XS = XSMAP[L]
    def f(e, LL):
        for tag, stride in (("zz", 8), ("yy", L*8), ("xx", XS)):
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
        for tag, stride in (("xx", XS), ("yy", L*8)):
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
e = Emit()
e(PRELUDE)
e(TR8MACRO)
gensoa.emit_conv(e)
for L in (6, 8):
    gensoadrv2.emit_soa_engine(e, L, small_passes(L), XS=XSMAP[L])
open("../impl68.c","w").write(e.text())
print("wrote impl68.c")
