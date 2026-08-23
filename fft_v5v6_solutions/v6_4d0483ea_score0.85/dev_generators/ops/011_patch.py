src = open('implementation.c').read()
# drop the zero-guard in map (add tiny bias instead) and slim it
src = src.replace("""    v8 _t = _zr * _zr + _zi * _zi;                                          \\
    v8 _y = (v8)_mm512_rsqrt14_pd((__m512d)_t);                             \\""",
"""    v8 _t = _zr * _zr + _zi * _zi + vtiny;                                  \\
    v8 _y = (v8)_mm512_rsqrt14_pd((__m512d)_t);                             \\""")
src = src.replace("""    _r = _r * (vtwo - _d * _r);                                             \\
    _r = _r * (vtwo - _d * _r);                                             \\
    _r = (v8)_mm512_mask_mov_pd((__m512d)_r, _mm512_cmp_pd_mask((__m512d)_t, (__m512d)vzero, _CMP_EQ_OQ), (__m512d)vone); \\
    VS((pr), _zr * _r); VS((pi), _zi * _r);                                 \\""",
"""    _r = _r * (vtwo - _d * _r);                                             \\
    _r = _r * (vtwo - _d * _r);                                             \\
    VS((pr), _zr * _r); VS((pi), _zi * _r);                                 \\""")
src = src.replace("""  const v8 vc32 = vbc(1.5), vtwo = vbc(2.0);                          \\
  (void)vc32; (void)vtwo;                                             \\""",
"""  const v8 vc32 = vbc(1.5), vtwo = vbc(2.0), vtiny = vbc(1e-300);     \\
  (void)vc32; (void)vtwo; (void)vtiny;                                \\""")
# roll the core j-loop (remove unroll pragma) - keep phase1 unrolled
src = src.replace("""    _Pragma("GCC unroll 16") for (int j = 0; j < H; j++) {                        \\
      v8 par = VL(SAR + j * 8), pai = VL(SAI + j * 8);                            \\""",
"""    for (int j = 0; j < H; j++) {                                                 \\
      v8 par = VL(SAR + j * 8), pai = VL(SAI + j * 8);                            \\""")
open('implementation.c','w').write(src)
