# k-tiled looped prime DFT for 23 (register-budget-proof): fold loop into scratch,
# then k-tiles with runtime j-loop and table-broadcast constants.
from netlib import fmt, trigc, trigs

def prime23_tables():
    N=23; h=11; W=h+2
    ct=[]; st=[]
    for k in range(1,h+1):
        crow=['0.0']; srow=['0.0']
        for j in range(1,h+1):
            crow.append(fmt(trigc((k*j)%N, N)))
            srow.append(fmt(trigs((k*j)%N, N)))
        crow.append('0.0'); srow.append('0.0')
        ct.append("{"+",".join(crow)+"}")
        st.append("{"+",".join(srow)+"}")
    return (f"static const double CT23[{h}][{W}] __attribute__((aligned(64))) = {{{','.join(ct)}}};\n"
            f"static const double ST23[{h}][{W}] __attribute__((aligned(64))) = {{{','.join(st)}}};\n")

def emit_prime23_loop(e, qname, stride, storer, KB=4):
    """qname: base ptr (re at +0, im at +8 per 16-block, stride per point).
       storer(e, k, re_expr, im_expr)."""
    N=23; h=11
    e.raw(f"__attribute__((aligned(64))) double PR23[{(h+2)*8}], PI23[{(h+2)*8}], QR23[{(h+2)*8}], QI23[{(h+2)*8}];")
    e.raw(f"vd x0r = LD({qname}), x0i = LD({qname} + 8);")
    e.raw(f"vd s0r = x0r, s0i = x0i, s1r = K(0.0), s1i = K(0.0);")
    e.raw(f"for(long j=1;j<={h};++j){{")
    e.raw(f"  vd a = LD({qname} + j*{stride}), b = LD({qname} + ({N}-j)*{stride});")
    e.raw(f"  vd cc = LD({qname} + j*{stride} + 8), d = LD({qname} + ({N}-j)*{stride} + 8);")
    e.raw(f"  vd pr = a+b, qr = a-b, pi = cc+d, qi = cc-d;")
    e.raw(f"  ST(PR23+j*8, pr); ST(QR23+j*8, qr); ST(PI23+j*8, pi); ST(QI23+j*8, qi);")
    e.raw(f"  if(j & 1){{ s0r = s0r + pr; s0i = s0i + pi; }} else {{ s1r = s1r + pr; s1i = s1i + pi; }}")
    e.raw(f"}}")
    storer(e, 0, "(s0r+s1r)", "(s0i+s1i)")
    k=1
    while k<=h:
        kb=min(KB, h-k+1)
        ks=list(range(k,k+kb))
        decl=", ".join(f"ar{t} = x0r, ai{t} = x0i, br{t} = K(0.0), bi{t} = K(0.0)" for t in ks)
        e.raw(f"{{ vd {decl};")
        e.raw(f"  for(long j=1;j<={h};++j){{")
        e.raw(f"    vd pr = LD(PR23+j*8), pi = LD(PI23+j*8);")
        e.raw(f"    vd qr = LD(QR23+j*8), qi = LD(QI23+j*8);")
        for t in ks:
            e.raw(f"    {{ vd c_ = K2(CT23[{t-1}][j]), s_ = K2(ST23[{t-1}][j]);")
            e.raw(f"      ar{t} = FMA(c_, pr, ar{t}); ai{t} = FMA(c_, pi, ai{t});")
            e.raw(f"      br{t} = FMA(s_, qr, br{t}); bi{t} = FMA(s_, qi, bi{t}); }}")
        e.raw(f"  }}")
        # X_k = C - iS  where C=(ar,ai), S=(br,bi): re = ar + bi? derive: X_k = C + i*S_neg...
        # From my reim: X_k.re = Cr - Si ; X_k.im = Ci + Sr with S = sum sin_kj * o_j, sin = sin(-2pi..)<0 baked.
        # Here ST23 = sin(+2pi kj/N) (positive angle) like 8175: their combine: yr = ar + bi, yi = ai - br; y(N-t): ar-bi, ai+br
        for t in ks:
            storer(e, t,   f"(ar{t} + bi{t})", f"(ai{t} - br{t})")
            storer(e, N-t, f"(ar{t} - bi{t})", f"(ai{t} + br{t})")
        e.raw("}")
        k += kb
