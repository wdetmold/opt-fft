src = open('implementation.c').read()
old = """#define STORE_MAP(pr, pi, vr, vi, pcr, pci)                                 \\
  {                                                                         \\
    v8 _zr = (vr) + VL(pcr), _zi = (vi) + VL(pci);                          \\
    v8 _t = _zr * _zr + _zi * _zi + vtiny;                                  \\
    v8 _y = (v8)_mm512_rsqrt14_pd((__m512d)_t);                             \\
    _y = _y * (vc32 - vhalf * _t * _y * _y);                                \\
    _y = _y * (vc32 - vhalf * _t * _y * _y);                                \\
    v8 _d = vone + _t * _y;          /* 1 + |z|  (t*rsqrt(t)) */            \\
    v8 _r = (v8)_mm512_rcp14_pd((__m512d)_d);                               \\
    _r = _r * (vtwo - _d * _r);                                             \\
    _r = _r * (vtwo - _d * _r);                                             \\
    VS((pr), _zr * _r); VS((pi), _zi * _r);                                 \\
  }"""
new = """#ifdef MAP_SQRT
#define STORE_MAP(pr, pi, vr, vi, pcr, pci)                                 \\
  {                                                                         \\
    v8 _zr = (vr) + VL(pcr), _zi = (vi) + VL(pci);                          \\
    v8 _t = _zr * _zr + _zi * _zi;                                          \\
    v8 _d = vone + vsqrt8(_t);                                              \\
    v8 _r = (v8)_mm512_rcp14_pd((__m512d)_d);                               \\
    _r = _r * (vtwo - _d * _r);                                             \\
    _r = _r * (vtwo - _d * _r);                                             \\
    VS((pr), _zr * _r); VS((pi), _zi * _r);                                 \\
  }
#else
#define STORE_MAP(pr, pi, vr, vi, pcr, pci)                                 \\
  {                                                                         \\
    v8 _zr = (vr) + VL(pcr), _zi = (vi) + VL(pci);                          \\
    v8 _t = _zr * _zr + _zi * _zi + vtiny;                                  \\
    v8 _y = (v8)_mm512_rsqrt14_pd((__m512d)_t);                             \\
    _y = _y * (vc32 - vhalf * _t * _y * _y);                                \\
    _y = _y * (vc32 - vhalf * _t * _y * _y);                                \\
    v8 _d = vone + _t * _y;          /* 1 + |z|  (t*rsqrt(t)) */            \\
    v8 _r = (v8)_mm512_rcp14_pd((__m512d)_d);                               \\
    _r = _r * (vtwo - _d * _r);                                             \\
    _r = _r * (vtwo - _d * _r);                                             \\
    VS((pr), _zr * _r); VS((pi), _zi * _r);                                 \\
  }
#endif"""
src = src.replace(old, new)
src = src.replace("#ifndef BLK17\n#define BLK17 4\n#endif", "#ifndef BLK17\n#define BLK17 6\n#endif")
src = src.replace("#ifndef BLK23\n#define BLK23 4\n#endif", "#ifndef BLK23\n#define BLK23 6\n#endif")
open('implementation.c','w').write(src)
print("ok")
