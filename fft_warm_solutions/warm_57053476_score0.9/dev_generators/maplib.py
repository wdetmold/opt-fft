# batched stage-interleaved map with integer-magic seeds (no microcoded/convert ops)
# x = z/(1+|z|), z = y + c.  H: hw sqrt for |z| ; R: magic rsqrt + 4 NR.
# w = 1/(1+s): magic rcp + 3 NR + exact residual step.
def emit_map_batch(e, offs, buf, cb, pattern="HHR", start_idx=0, outbuf=None):
    n = len(offs)
    ob = outbuf or buf
    zr = {}; zi = {}; tv = {}; sv = {}; wv = {}
    for i, o in enumerate(offs):
        zr[i] = e.v(f"LD({buf} + {o}) + LD({cb} + {o})")
        zi[i] = e.v(f"LD({buf} + {o+8}) + LD({cb} + {o+8})")
        tv[i] = e.v(f"FMA({zr[i]},{zr[i]}, FMA({zi[i]},{zi[i]}, K(1e-300)))")
    kinds = [pattern[(start_idx + i) % len(pattern)] for i in range(n)]
    rr = {}
    for i in range(n):
        if kinds[i] == 'H':
            sv[i] = e.v(f"VD(_mm512_sqrt_pd(MD({tv[i]})))")
        else:
            rr[i] = e.v(f"IRSQ({tv[i]})")
    hv = {}
    for i in range(n):
        if kinds[i] == 'R':
            hv[i] = e.v(f"K(0.5) * {tv[i]}")
    for it in range(4):
        for i in range(n):
            if kinds[i] == 'R':
                ee = e.v(f"FNMA({hv[i]} * {rr[i]}, {rr[i]}, K(1.5))")
                rr[i] = e.v(f"{rr[i]} * {ee}")
    for i in range(n):
        if kinds[i] == 'R':
            sv[i] = e.v(f"{tv[i]} * {rr[i]}")
    dv = {}
    for i in range(n):
        dv[i] = e.v(f"K(1.0) + {sv[i]}")
    for i in range(n):
        wv[i] = e.v(f"IRCP({dv[i]})")
    for it in range(3):
        for i in range(n):
            ee = e.v(f"FNMA({dv[i]}, {wv[i]}, K(2.0))")
            wv[i] = e.v(f"{wv[i]} * {ee}")
    for i in range(n):
        ee = e.v(f"FNMA({dv[i]}, {wv[i]}, K(1.0))")
        wv[i] = e.v(f"FMA({wv[i]}, {ee}, {wv[i]})")
    for i, o in enumerate(offs):
        e.raw(f"ST({ob} + {o}, {zr[i]} * {wv[i]});")
        e.raw(f"ST({ob} + {o+8}, {zi[i]} * {wv[i]});")

MAP_HELPERS = r'''
#define IRCP(d) VD(_mm512_castsi512_pd(_mm512_sub_epi64(_mm512_set1_epi64(0x7FDE6238502484BAll), _mm512_castpd_si512(MD(d)))))
#define IRSQ(t) VD(_mm512_castsi512_pd(_mm512_sub_epi64(_mm512_set1_epi64(0x5FE6EB50C7B537A9ll), _mm512_srli_epi64(_mm512_castpd_si512(MD(t)), 1))))
'''
