src = open('implementation.c').read()
src = src.replace("""#define STORE_MAP(pr, pi, vr, vi, pcr, pci)                    \\
  {                                                            \\
    v8 _zr = (vr) + VL(pcr), _zi = (vi) + VL(pci);             \\
    v8 _mg = vsqrt8(_zr * _zr + _zi * _zi);                    \\
    v8 _sc = vone / (vone + _mg);                              \\
    VS((pr), _zr * _sc); VS((pi), _zi * _sc);                  \\
  }""",
"""#define STORE_MAP(pr, pi, vr, vi, pcr, pci)                                 \\
  {                                                                         \\
    v8 _zr = (vr) + VL(pcr), _zi = (vi) + VL(pci);                          \\
    v8 _t = _zr * _zr + _zi * _zi;                                          \\
    v8 _y = (v8)_mm512_rsqrt14_pd((__m512d)_t);                             \\
    _y = _y * (vc32 - vhalf * _t * _y * _y);                                \\
    _y = _y * (vc32 - vhalf * _t * _y * _y);                                \\
    v8 _d = vone + _t * _y;          /* 1 + |z|  (t*rsqrt(t)) */            \\
    v8 _r = (v8)_mm512_rcp14_pd((__m512d)_d);                               \\
    _r = _r * (vtwo - _d * _r);                                             \\
    _r = _r * (vtwo - _d * _r);                                             \\
    _r = (v8)_mm512_mask_mov_pd((__m512d)_r, _mm512_cmp_pd_mask((__m512d)_t, (__m512d)vzero, _CMP_EQ_OQ), (__m512d)vone); \\
    VS((pr), _zr * _r); VS((pi), _zi * _r);                                 \\
  }""")
src = src.replace("""  const v8 vone = vbc(1.0), vzero = vbc(0.0);                         \\""",
"""  const v8 vone = vbc(1.0), vzero = vbc(0.0);                         \\
  const v8 vc32 = vbc(1.5), vtwo = vbc(2.0);                          \\
  (void)vc32; (void)vtwo;                                             \\""")
open('implementation.c','w').write(src)
print("ok")
