"""DFT6 (PFA 2x3) and DFT8 codelets, straight-line, SoA vertical, in-place safe."""
import numpy as np
from glib import Emit, hexd
from gen36 import emit_dft3

RT2H = hexd(np.sqrt(np.longdouble(2))/2)

def emit_dft6(e, stride, store_cb):
    # input map n=(3n1+2n2)%6 ; output k=(3k1+4k2)%6
    e("{")
    for n in range(6):
        e(f"V x{n}r = VL(pr + {n*stride}), x{n}i = VL(pi + {n*stride});")
    # stage A: DFT2 over n1 for n2=0,1,2: pair (2n2 mod ... n=(3n1+2n2)%6
    a = {}
    for n2 in range(3):
        j0 = (3*0 + 2*n2) % 6
        j1 = (3*1 + 2*n2) % 6
        e(f"V a{n2}0r = VADD(x{j0}r, x{j1}r), a{n2}0i = VADD(x{j0}i, x{j1}i);")
        e(f"V a{n2}1r = VSUB(x{j0}r, x{j1}r), a{n2}1i = VSUB(x{j0}i, x{j1}i);")
    # stage B: DFT3 over n2 for k1=0,1 ; outputs X[(3k1+4k2)%6]
    for k1 in range(2):
        xin = [(f"a{n2}{k1}r", f"a{n2}{k1}i") for n2 in range(3)]
        xo = [(f"o{k1}{k2}r", f"o{k1}{k2}i") for k2 in range(3)]
        e("V " + ", ".join(f"{r}, {i}" for r,i in xo) + ";")
        emit_dft3(e, xin, xo)
    for k1 in range(2):
        for k2 in range(3):
            k = (3*k1 + 4*k2) % 6
            store_cb(k, f"o{k1}{k2}r", f"o{k1}{k2}i")
    e("}")

def emit_dft8(e, stride, store_cb):
    # DIT radix-2: bit-reversed input grouping; forward DFT (w = e^{-2pi i/8})
    e("{")
    for n in range(8):
        e(f"V x{n}r = VL(pr + {n*stride}), x{n}i = VL(pi + {n*stride});")
    # level 1: pairs (j, j+4)
    for j in range(4):
        e(f"V s{j}r = VADD(x{j}r, x{j+4}r), s{j}i = VADD(x{j}i, x{j+4}i);")
        e(f"V d{j}r = VSUB(x{j}r, x{j+4}r), d{j}i = VSUB(x{j}i, x{j+4}i);")
    # level 2 on s: pairs (0,2),(1,3); on d: twiddles w8^0, w8^1... classic DIF actually.
    # Use DIF: X[k] via decimation in frequency:
    # E[j] = s[j] (even outputs = DFT4 of s), O[j] = d[j]*w8^j (odd outputs = DFT4 of that)
    # d1 *= w8 = (c,-c) ; d2 *= w8^2 = -i ; d3 *= w8^3 = (-c,-c), c = sqrt2/2
    e(f"V c_ = VSET1({RT2H});")
    e(f"{{ V tr = VMUL(VADD(d1r, d1i), c_); V ti = VMUL(VSUB(d1i, d1r), c_); d1r = tr; d1i = ti; }}")
    e(f"{{ V tr = d2i; V ti = _mm512_sub_pd(_mm512_setzero_pd(), d2r); d2r = tr; d2i = ti; }}")
    e(f"{{ V tr = VMUL(VSUB(d3i, d3r), c_); V ti = VMUL(VADD(d3r, d3i), _mm512_sub_pd(_mm512_setzero_pd(), c_)); d3r = tr; d3i = ti; }}")
    # DFT4 of s -> X[0,2,4,6]; DFT4 of d -> X[1,3,5,7]
    for (pfx, outks) in (("s", [0,2,4,6]), ("d", [1,3,5,7])):
        e(f"{{ V t0r=VADD({pfx}0r,{pfx}2r), t0i=VADD({pfx}0i,{pfx}2i);")
        e(f"  V t1r=VSUB({pfx}0r,{pfx}2r), t1i=VSUB({pfx}0i,{pfx}2i);")
        e(f"  V t2r=VADD({pfx}1r,{pfx}3r), t2i=VADD({pfx}1i,{pfx}3i);")
        e(f"  V t3r=VSUB({pfx}1r,{pfx}3r), t3i=VSUB({pfx}1i,{pfx}3i);")
        e(f"  V y0r=VADD(t0r,t2r), y0i=VADD(t0i,t2i);")
        e(f"  V y2r=VSUB(t0r,t2r), y2i=VSUB(t0i,t2i);")
        e(f"  V y1r=VADD(t1r,t3i), y1i=VSUB(t1i,t3r);")
        e(f"  V y3r=VSUB(t1r,t3i), y3i=VADD(t1i,t3r);")
        for q, k in enumerate(outks):
            store_cb(k, f"y{q}r", f"y{q}i")
        e("}")
    e("}")
