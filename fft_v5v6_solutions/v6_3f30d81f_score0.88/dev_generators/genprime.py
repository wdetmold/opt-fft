# Generates sign-folded, register-twiddle prime kernels for p in {13,17,23}
def gen(p, kb):
    h = (p-1)//2
    L = []
    A = L.append
    A(f"// generated: {p}-point DFT, twiddle magnitudes in registers, k-block {kb}")
    A(f"static __attribute__((always_inline)) inline")
    A(f"void k{p}(vd *restrict xr, vd *restrict xi, ptrdiff_t s,")
    A(f"          const vd *restrict cr, const vd *restrict ci, const int domap) {{")
    A(f"    vd ur[{h+1}], ui[{h+1}], vr[{h+1}], vi[{h+1}];")
    A(f"    vd x0r = xr[0], x0i = xi[0];")
    A(f"    vd s0r = x0r, s0i = x0i, s1r = (vd){{0}}, s1i = (vd){{0}};")
    A(f"    _Pragma(\"GCC unroll 16\")")
    A(f"    for (int j = 1; j <= {h}; j++) {{")
    A(f"        vd ar = xr[(ptrdiff_t)j*s],      ai = xi[(ptrdiff_t)j*s];")
    A(f"        vd br = xr[(ptrdiff_t)({p}-j)*s], bi = xi[(ptrdiff_t)({p}-j)*s];")
    A(f"        ur[j] = ar + br; ui[j] = ai + bi;")
    A(f"        vr[j] = ar - br; vi[j] = ai - bi;")
    A(f"        if (j & 1) {{ s0r += ur[j]; s0i += ui[j]; }}")
    A(f"        else       {{ s1r += ur[j]; s1i += ui[j]; }}")
    A(f"    }}")
    A(f"    KSTORE(xr[0], xi[0], s0r + s1r, s0i + s1i, cr[0], ci[0], domap);")
    # preload twiddle magnitude registers
    for m in range(1, h+1):
        A(f"    const vd cw{m} = SPLAT(CW{p}[{m}]), sw{m} = SPLAT(SW{p}[{m}]);")
    k = 1
    while k <= h:
        nb = min(kb, h - k + 1)
        for t in range(nb):
            A(f"    {{}}" if False else f"    // k = {k+t}")
        A("    {")
        for t in range(nb):
            A(f"        vd A{t} = x0r, C{t} = x0i, B{t} = (vd){{0}}, D{t} = (vd){{0}};")
        for j in range(1, h+1):
            A(f"        {{ vd u = ur[{j}], iu = ui[{j}], v = vr[{j}], iv = vi[{j}];")
            for t in range(nb):
                m = (j*(k+t)) % p
                sgn = '+' if m <= h else '-'
                mt = m if m <= h else p - m
                A(f"          A{t} += u*cw{mt}; C{t} += iu*cw{mt}; B{t} {sgn}= iv*sw{mt}; D{t} {sgn}= v*sw{mt};")
            A(f"        }}")
        for t in range(nb):
            A(f"        {{ ptrdiff_t ka = (ptrdiff_t){k+t}*s, kb2 = (ptrdiff_t){p-k-t}*s;")
            A(f"          KSTORE(xr[ka], xi[ka], A{t} + B{t}, C{t} - D{t}, cr[ka], ci[ka], domap);")
            A(f"          KSTORE(xr[kb2], xi[kb2], A{t} - B{t}, C{t} + D{t}, cr[kb2], ci[kb2], domap); }}")
        A("    }")
        k += nb
    A("}")
    return "\n".join(L)

import sys
out = []
out.append(gen(13, 2))
out.append(gen(17, 2))
out.append(gen(23, int(sys.argv[1]) if len(sys.argv)>1 else 1))
open('primegen.c','w').write("\n\n".join(out) + "\n")
print("generated")
