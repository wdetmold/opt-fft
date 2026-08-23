# row-wise z-pass for L=64: per row: stageA DFT8 (vertical over 8 row-vecs),
# lane-twiddles W64^{j0*k0} (vector constants), transpose, stageB DFT8 (vertical),
# transpose back, +c, map, store. All in registers; no VS2/SC.
import numpy as np
from netlib import E, dft_small, fmt, trigc, trigs

def gen_tw_tables():
    # TWR64[k0][lane j0] = cos(-2pi j0 k0/64), TWI = sin
    rows_r = []
    rows_i = []
    for k0 in range(8):
        rr = ", ".join(fmt(trigc(j0*k0, 64)) for j0 in range(8))
        ii = ", ".join(fmt(trigs(j0*k0, 64)) for j0 in range(8))
        rows_r.append(f"{{{rr}}}")
        rows_i.append(f"{{{ii}}}")
    return (f"static const double TWR64[8][8] __attribute__((aligned(64))) = {{{', '.join(rows_r)}}};\n"
            f"static const double TWI64[8][8] __attribute__((aligned(64))) = {{{', '.join(rows_i)}}};\n")

def emit_z64_row(e, RW, mappat="HR"):
    """q: row base (re at +0..63, im at +RW..RW+63); cq: c row base"""
    # load 8 complex vector-points: vec j1 = (LD(q+8*j1), LD(q+RW+8*j1)) -- these are
    # z-groups: vec v holds lanes z=8v+l : stageA is DFT8 over v (j1=v? CHECK):
    # z = 8*v + lane. For DIT with j = 8*j1 + j0: j1 = v? NO: j = z: j1 = z/8 = v, j0 = lane. OK
    # X[8k1+k0] = sum_j0 W64^{j0 k0} W8^{j0 k1} [ sum_j1 x[8j1+j0] W8^{j1 k0} ]
    # stage A: for each lane j0 (vertical): A[k0] = DFT8 over vecs.
    xs = [(e.v(f"LD(q + {8*j1})"), e.v(f"LD(q + {RW} + {8*j1})")) for j1 in range(8)]
    A = dft_small(e, xs, 8)     # A[k0], lanes j0
    # twiddle: A[k0] *= W64^{j0*k0} per lane: vector constants
    B_in = []
    for k0 in range(8):
        if k0 == 0:
            B_in.append(A[0])
        else:
            ar, ai = A[k0]
            tr = e.v(f"LD(TWR64[{k0}])")
            ti = e.v(f"LD(TWI64[{k0}])")
            nr = e.v(f"FMS({tr}, {ar}, {ti} * {ai})")
            ni = e.v(f"FMA({tr}, {ai}, {ti} * {ar})")
            B_in.append((nr, ni))
    # transpose re and im across the 8 vectors: after transpose vec j0, lanes k0
    rn = [B_in[k0][0] for k0 in range(8)]
    im = [B_in[k0][1] for k0 in range(8)]
    e.raw(f"TR8({','.join(rn)});")
    e.raw(f"TR8({','.join(im)});")
    xs2 = [(rn[j0], im[j0]) for j0 in range(8)]
    Bv = dft_small(e, xs2, 8)   # B[k1], lanes k0
    # transpose back to z-major: vec t, lanes l -> z = 8t + l: currently vec k1 lanes k0: k = 8k1+k0
    # want store vec v' holding k = 8v' + lane: i.e., vec k1 lane k0 holds k=8k1+k0: TRANSPOSE AGAIN ->
    # vec a lane b = original vec b lane a = k = 8b + a... hmm: after TR8: vec a holds lanes b where value(k1=b,k0=a): k = 8b+a: NOT contiguous.
    # Instead store in "digit-swapped" order? We need natural. Do: transpose -> vec k0, lanes k1: k = 8*lane + vec -> still swapped.
    # Natural requires gathering k=8v+l: value at (k1=v, k0=l): source vec v lane l -> no permute needed!! vec k1 IS the natural vector index (k1 = k/8) IF lanes are k0 = k%8. YES: vec k1 lanes k0: element (vec v, lane l) = X[8v + l] -> NATURAL ALREADY.
    out = Bv
    return out

if __name__ == "__main__":
    # quick numeric test of the row codelet emitted standalone
    import subprocess, ctypes
    RW = 64
    e = E()
    out = emit_z64_row(e, RW)
    for k1 in range(8):
        e.raw(f"ST(o + {8*k1}, {out[k1][0]}); ST(o + {RW} + {8*k1}, {out[k1][1]});")
    full = open('impl1.c').read()
    hdr = full[:full.index('static void* xalloc')]
    trh = full[full.index('static const long long IDX_TAB'):full.index('#define IX(i)')]
    ix = full[full.index('#define IX(i)'):full.index('static double SC_6')]
    src = hdr + trh + ix + gen_tw_tables() + f"""
void zrow(const double* q, double* o){{
  {e.code()}
}}
"""
    open('z64test.c','w').write(src)
    subprocess.run(['gcc','-O2','-march=native','-shared','-fPIC','z64test.c','-o','z64test.so'],check=True)
    lib=ctypes.CDLL('./z64test.so')
    x=np.zeros(2*RW); rng=np.random.default_rng(1)
    x[:64]=rng.standard_normal(64); x[RW:RW+64]=rng.standard_normal(64)
    o=np.zeros(2*RW)
    lib.zrow.argtypes=[ctypes.c_void_p]*2
    xa=np.ascontiguousarray(x); oa=np.ascontiguousarray(o)
    lib.zrow(xa.ctypes.data, oa.ctypes.data)
    ref=np.fft.fft(x[:64]+1j*x[RW:RW+64])
    got=oa[:64]+1j*oa[RW:RW+64]
    print("z64 row err:", np.abs(ref-got).max()/np.abs(ref).max())
