"""Folded symmetric prime DFT codelets (p=13,17,23), cos/sin split phases, SoA vertical.
In-place safe: phase S (sin, accumulates D,E -> PSCR), phase C (cos, A,B + recombine + stores).
  A_k = x0r + sum_j C_kj ur_j ; B_k = x0i + sum_j C_kj ui_j
  D_k = sum_j S_kj vi_j       ; E_k = sum_j S_kj vr_j
  X_k=(A+D, B-E) ; X_{p-k}=(A-D, B+E) ; X0 = x0 + sum u
"""
import numpy as np
from glib import Emit, trig, hexd

def tables(p):
    h = (p-1)//2
    cs = trig(p)
    ct = {t: cs[t][0] for t in range(1, h+1)}          # cos(2pi t/p) (cs = angle -2pi t/p, cos even)
    st = {t: -cs[t][1] for t in range(1, h+1)}         # sin(2pi t/p)
    def fold(kj):
        t = kj % p
        if t <= h: return (t, 1)
        return (p - t, -1)   # sin sign flips
    return h, ct, st, fold

def emit_tables(e, p):
    h, ct, st, fold = tables(p)
    e(f"static const double CT{p}[{h}] ALIGN64 = {{ {', '.join(hexd(ct[t]) for t in range(1,h+1))} }};")
    e(f"static const double ST{p}[{h}] ALIGN64 = {{ {', '.join(hexd(st[t]) for t in range(1,h+1))} }};")

def emit_prime_func(e, p, tag, stride, kblocks, mapfused, cstride=8):
    """kblocks: list of lists of k values covering 1..h (for both phases). Loops n pencils spaced pstep."""
    h, ct, st, fold = tables(p)
    fname = f"p{p}_{tag}" + ("m" if mapfused else "")
    args = "double* PR, double* PI"
    if mapfused: args += ", const double* CR, const double* CI"
    args += ", long n, long pstep"
    e(f"static __attribute__((noinline)) void {fname}({args}){{")
    e.ind += 1
    e("#pragma GCC unroll 1")
    e("for (long q_ = 0; q_ < n; q_++) {")
    e.ind += 1
    e("double* pr = PR + q_*pstep; double* pi = PI + q_*pstep;")
    if mapfused:
        e("const double* cr = CR + q_*pstep; const double* ci = CI + q_*pstep;")
    # ---- phase S: accumulate D_k, E_k -> PSCR[(k-1)*16 ..]; also store u_j -> USCR[(j-1)*16 ..] (block 0)
    for bi_, blk in enumerate(kblocks):
        e("{")
        for t in range(1, h+1):
            e(f"V S{t} = VSET1({hexd(st[t])});")
        for k in blk:
            e(f"V D{k} = _mm512_setzero_pd(), E{k} = _mm512_setzero_pd();")
        for j in range(1, h+1):
            e(f"{{ V ar = VL(pr + {j*stride}), ai = VL(pi + {j*stride});")
            e(f"  V br = VL(pr + {(p-j)*stride}), bi = VL(pi + {(p-j)*stride});")
            e(f"  V vr = VSUB(ar,br), vi = VSUB(ai,bi);")
            if bi_ == 0:
                e(f"  VS(USCR + {(j-1)*16}, VADD(ar,br)); VS(USCR + {(j-1)*16+8}, VADD(ai,bi));")
            for k in blk:
                t, sgn = fold(k*j)
                op = "VFMA" if sgn > 0 else "VFNMA"
                e(f"  D{k} = {op}(vi, S{t}, D{k}); E{k} = {op}(vr, S{t}, E{k});")
            e("}")
        for k in blk:
            e(f"VS(PSCR + {(k-1)*16}, D{k}); VS(PSCR + {(k-1)*16+8}, E{k});")
        e("}")
    # ---- phase C: A,B + X0 + recombine with D,E; stores (maybe map-fused)
    e("VS(USCR + 384, VL(pr)); VS(USCR + 392, VL(pi));  // save x0 (row 0 gets overwritten)")
    for bi_, blk in enumerate(kblocks):
        e("{")
        for t in range(1, h+1):
            e(f"V C{t} = VSET1({hexd(ct[t])});")
        e("V x0r = VL(USCR + 384), x0i = VL(USCR + 392);")
        for k in blk:
            e(f"V A{k} = x0r, B{k} = x0i;")
        if bi_ == 0:
            e("V s0r = x0r, s0i = x0i;")
        for j in range(1, h+1):
            e(f"{{ V ur = VL(USCR + {(j-1)*16}), ui = VL(USCR + {(j-1)*16+8});")
            for k in blk:
                t, sgn = fold(k*j)
                e(f"  A{k} = VFMA(ur, C{t}, A{k}); B{k} = VFMA(ui, C{t}, B{k});")
            if bi_ == 0:
                e(f"  s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);")
            e("}")
        # stores: X0 first (block 0)
        def store(k, rexpr, iexpr):
            if mapfused:
                e(f"{{ V zr_ = VADD({rexpr}, VL(cr + {k*cstride})), zi_ = VADD({iexpr}, VL(ci + {k*cstride}));")
                e(f"  MAP2(zr_, zi_);")
                e(f"  VS(pr + {k*stride}, zr_); VS(pi + {k*stride}, zi_); }}")
            else:
                e(f"VS(pr + {k*stride}, {rexpr}); VS(pi + {k*stride}, {iexpr});")
        if bi_ == 0:
            store(0, "s0r", "s0i")
        for k in blk:
            e(f"{{ V Dk = VL(PSCR + {(k-1)*16}), Ek = VL(PSCR + {(k-1)*16+8});")
            e(f"  V Xr = VADD(A{k}, Dk), Xi = VSUB(B{k}, Ek);")
            e(f"  V Yr = VSUB(A{k}, Dk), Yi = VADD(B{k}, Ek);")
            store(k, "Xr", "Xi")
            store(p-k, "Yr", "Yi")
            e("}")
        e("}")
    e.ind -= 1
    e("}")
    e.ind -= 1
    e("}")

KBLOCKS = {13: [[1,2,3,4,5,6]], 17: [[1,2,3,4,5,6,7,8]], 23: [[1,2,3,4,5,6],[7,8,9,10,11]]}
