import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import prelude_c

import gen_main
from gen_main import gen_conv, gen_driver, PRELUDE as MYPRE
from gen_a import gen_composite
from gen_asm_prime import gen_prime_asm

def build_full():
    parts = [prelude_c.PRELUDE]   # includes alloc_huge, TR8, map2, map_range, DEINT/INTER, FMA_BC...
    # my A sizes
    KB = {13:(6,6),17:(8,8),23:(6,6)}
    from gen_pv import gen_pv
    from gen_a import gen_prime
    for L in (6,8,13,17,23):
        PS = L*L if (L*L) % 2 == 1 else L*L+1
        if L in (6,8):
            parts.append(gen_composite(L, PS*16))
            parts.append(gen_pv(L))
            parts.append(gen_conv(L, PS))
            parts.append(gen_driver(L, PS, pair_y=True))
        else:
            NVp = (L + 7) // 8
            RSp = NVp * 16
            YPp = NVp * 8
            PSp = YPp * RSp
            if (PSp * 8) % 4096 == 0: PSp += 16
            parts.append(gen_prime_asm(L, KB[L][0], KB[L][1], PS, pv=(RSp, PSp)))
            parts.append(gen_pv(L, asmcd=True))
            parts.append(gen_conv(L, PS))
            parts.append(gen_driver(L, PS, prime_asm=True))
    # my B sizes
    parts.append('static const double MTB[4] ALIGN64 = { 1e-30, 1.0, 0.5, 0x1.e6238da3c2118p+1022 };')
    from gen_b import build_b
    for L in (36,45,64):
        code = build_b(L)
        if L == 36:
            code = code.replace("void run_36(const double*", "static void run_36B(const double*")
        parts.append(code)
    # batch-lane 36
    from gen_a36 import gen_a36
    conv36A = gen_conv(36, 36*36).replace("convin_36", "convin_36A").replace("convout_36", "convout_36A")
    parts.append(conv36A)
    parts.append(gen_a36())
    parts.append("""
void run_36(const double* x0, const double* c, double* out1, double* outm, long B, long m){
    if(m < 1) m = 1;
    long rem = B % 8, G8 = B - rem;
    if(G8) run_36A(x0, c, out1, outm, G8, m);
    if(rem) run_36B(x0 + G8*2*46656, c + G8*2*46656, out1 + G8*2*46656, outm + G8*2*46656, rem, m);
}
""")
    return "\n".join(parts)

if __name__ == "__main__":
    HEADER = '/*\n * Iterated batched 3D complex DFTs (z = FFT3(x) + c ; x <- z/(1+|z|)) for the\n * eight fixed cube sizes L in {6,8,13,17,23,36,45,64}. All transform\n * arithmetic is generated, hand-scheduled AVX-512 (no FFT library code):\n *  - sizes 6/8/13/17/23: batch-lane SoA, 8 volumes per zmm lane group (pure\n *    vertical SIMD). 6 = PFA(2x3); 8 = radix-2 DFT8; 13/17/23 = direct\n *    symmetric ("Hartley-split") prime DFTs as hand-allocated inline asm:\n *    phase-split cos/sin accumulator sweeps, register-resident constants,\n *    software-pipelined elementwise map. Per-volume fallback drivers handle\n *    batch remainders without padded-lane waste.\n *  - size 36: batch-lane SoA with table-driven two-stage PFA(4x9) codelets,\n *    map fused into the next step\'s first DFT stage (c pre-permuted to PFA\n *    input order); within-volume fallback for remainders.\n *  - sizes 45/64 (+36 fallback): within-volume split re/im rows, 8x8\n *    in-register transposes on the contiguous axis, two-stage PFA(5x9) /\n *    CT(8x8, baked twiddles) codelets, fused map, L2-resident planes.\n *  - steady state alternates a slab visit (y,z; completes odd steps) and a\n *    pencil visit (x; completes even steps), each pre-transforming the next\n *    step.\n *  - map z/(1+|z|): rsqrt14/rcp14 + Newton/Heron to ~1 ulp, pipelined 3-4\n *    wide (float-seeded cvtpd2ps/vrsqrtps where approx ops are microcoded).\n *  - twiddles computed in long double, baked as hex literals; hugepage\n *    arenas; padded/staggered strides; single-threaded.\n */\n'
    src = HEADER + build_full()
    open("implementation.c","w").write(src)
    print("wrote implementation.c", len(src))
