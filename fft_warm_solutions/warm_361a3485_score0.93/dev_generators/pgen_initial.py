import numpy as np
LD = np.longdouble
PI2 = LD('6.28318530717958647692528676655900577')

def hexd(v): return float(v).hex()

def prime_tables(p):
    h = (p-1)//2
    C = [None]*(h+1); S = [None]*(h+1)
    for m in range(1, h+1):
        a = PI2*LD(m)/LD(p)
        C[m] = float(np.cos(a)); S[m] = float(np.sin(a))
    return C, S

def csidx(p, k, j):
    """return (m, sgn_sin): cos(2pi kj/p)=C[m], sin(2pi kj/p)=sgn*S[m]"""
    h = (p-1)//2
    r = (k*j) % p
    if r <= h: return r, +1
    return p-r, -1

def gen_prime(p, KB, style='split'):
    """emit codelet kNEWp: phase-split folded DFT with register constants.
       KB: k-block size for accumulation phases."""
    h = (p-1)//2
    C, S = prime_tables(p)
    L = []
    A = L.append
    A(f"static __attribute__((always_inline)) inline")
    A(f"void kg{p}(vd *restrict xr, vd *restrict xi, ptrdiff_t s,")
    A(f"         const vd *restrict cr, const vd *restrict ci, const int domap) {{")
    # fold phase
    A(f"  vd sr[{h+1}], si[{h+1}], dr[{h+1}], di[{h+1}];")
    A(f"  vd x0r = xr[0], x0i = xi[0];")
    A(f"  vd y0r = x0r, y0i = x0i;")
    for j in range(1, h+1):
        A(f"  {{ vd ar = xr[{j}*s], ai = xi[{j}*s], br = xr[{p-j}*s], bi = xi[{p-j}*s];")
        A(f"    sr[{j}] = ar + br; si[{j}] = ai + bi; dr[{j}] = ar - br; di[{j}] = ai - bi;")
        A(f"    y0r += sr[{j}]; y0i += si[{j}]; }}")
    A(f"  vd Ar[{h+1}], Ai[{h+1}], Br[{h+1}], Bi[{h+1}];")
    # cos phase, k blocks
    kb = []
    k = 1
    while k <= h:
        kb.append(list(range(k, min(k+KB, h+1)))); k += KB
    A("  {")
    for m in range(1, h+1):
        A(f"    const vd KC{m} = VC({hexd(C[m])});")
    for blk in kb:
        A("    {")
        for k in blk:
            A(f"      vd a{k}r = x0r, a{k}i = x0i;")
        for j in range(1, h+1):
            for k in blk:
                m, sg = csidx(p, k, j)
                A(f"      a{k}r += KC{m} * sr[{j}]; a{k}i += KC{m} * si[{j}];")
        for k in blk:
            A(f"      Ar[{k}] = a{k}r; Ai[{k}] = a{k}i;")
        A("    }")
    A("  }")
    # sin phase
    A("  {")
    for m in range(1, h+1):
        A(f"    const vd KS{m} = VC({hexd(S[m])});")
    for blk in kb:
        A("    {")
        for k in blk:
            j0 = 1
            m, sg = csidx(p, k, j0)
            op = "" if sg > 0 else "-"
            A(f"      vd b{k}r = {op}(KS{m} * dr[1]), b{k}i = {op}(KS{m} * di[1]);")
        for j in range(2, h+1):
            for k in blk:
                m, sg = csidx(p, k, j)
                op = "+=" if sg > 0 else "-="
                A(f"      b{k}r {op} KS{m} * dr[{j}]; b{k}i {op} KS{m} * di[{j}];")
        for k in blk:
            A(f"      Br[{k}] = b{k}r; Bi[{k}] = b{k}i;")
        A("    }")
    A("  }")
    # combine + store
    A(f"  KSTORE(xr[0], xi[0], y0r, y0i, cr[0], ci[0], domap);")
    for k in range(1, h+1):
        A(f"  {{ vd ur = Ar[{k}] + Bi[{k}], ui = Ai[{k}] - Br[{k}];")
        A(f"    vd vr = Ar[{k}] - Bi[{k}], vi = Ai[{k}] + Br[{k}];")
        A(f"    KSTORE(xr[{k}*s], xi[{k}*s], ur, ui, cr[{k}*s], ci[{k}*s], domap);")
        A(f"    KSTORE(xr[{p-k}*s], xi[{p-k}*s], vr, vi, cr[{p-k}*s], ci[{p-k}*s], domap); }}")
    A("}")
    return "\n".join(L)

if __name__ == '__main__':
    import sys
    p = int(sys.argv[1]); kb = int(sys.argv[2])
    print(gen_prime(p, kb))
