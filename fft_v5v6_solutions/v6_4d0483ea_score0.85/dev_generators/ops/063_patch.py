src = open('implementation.c').read()
# STORE_MAP: two variants selected by compile-time domap value (1=sqrt, 2=NR)
old = src[src.index("#if 1\n#define STORE_MAP"):src.index("#define STORE_X")]
new = """#define STORE_MAP(pr, pi, vr, vi, pcr, pci, MV)                             \\
  {                                                                         \\
    v8 _zr = (vr) + VL(pcr), _zi = (vi) + VL(pci);                          \\
    v8 _t = _zr * _zr + _zi * _zi;                                          \\
    v8 _d, _r;                                                              \\
    if ((MV) == 1) {                                                        \\
      _d = vone + vsqrt8(_t);                                               \\
      _r = (v8)_mm512_rcp14_pd((__m512d)_d);                                \\
      _r = _r * (vtwo - _d * _r);                                           \\
      _r = _r * (vtwo - _d * _r);                                           \\
    } else {                                                                \\
      v8 _t2 = _t + vtiny;                                                  \\
      v8 _y = (v8)_mm512_rsqrt14_pd((__m512d)_t2);                          \\
      _y = _y * (vc32 - vhalf * _t2 * _y * _y);                             \\
      _y = _y * (vc32 - vhalf * _t2 * _y * _y);                             \\
      _d = vone + _t2 * _y;                                                 \\
      _r = (v8)_mm512_rcp14_pd((__m512d)_d);                                \\
      _r = _r * (vtwo - _d * _r);                                           \\
      _r = _r * (vtwo - _d * _r);                                           \\
    }                                                                       \\
    VS((pr), _zr * _r); VS((pi), _zi * _r);                                 \\
  }
"""
src = src.replace(old, new)
src = src.replace("""#define STORE_X(pr, pi, vr, vi, off)                           \\
  {                                                            \\
    if (domap) { STORE_MAP((pr) + (off), (pi) + (off), vr, vi, cre + (off), cim + (off)) } \\
    else       { STORE_PLAIN((pr) + (off), (pi) + (off), vr, vi) }                          \\
  }""",
"""#define STORE_X(pr, pi, vr, vi, off)                           \\
  {                                                            \\
    if (domap) { STORE_MAP((pr) + (off), (pi) + (off), vr, vi, cre + (off), cim + (off), domap) } \\
    else       { STORE_PLAIN((pr) + (off), (pi) + (off), vr, vi) }                          \\
  }""")
# remove old #else/#endif leftovers from previous #if 1 block
src = src.replace("""#else
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
#endif""", "")
# DEF_SWEEPB: MAPV parameter
src = src.replace("#define DEF_SWEEPB(LN, L_, LP_, R_, LPS_, SS_, FFTLINE)",
                  "#define DEF_SWEEPB(LN, L_, LP_, R_, LPS_, SS_, FFTLINE, MAPV)")
src = src.replace("""        FFTLINE(Xre + off, Xim + off, (long)SS_, 1, Cre + off, Cim + off,             \\""",
                  """        FFTLINE(Xre + off, Xim + off, (long)SS_, MAPV, Cre + off, Cim + off,          \\""")
for (args, mv) in [("6, 6, 8, 8, 8, 72, fft6_line", "2"), ("8, 8, 8, 8, 8, 72, fft8_line", "2"),
                   ("13, 13, 16, 16, 16, 208, fft13_line", "1"), ("17, 17, 24, 24, 24, 408, fft17_line", "1"),
                   ("23, 23, 24, 24, 24, 552, fft23_line", "1"), ("36, 36, 40, 40, 40, 1440, fft36_line", "1"),
                   ("45, 45, 48, 48, 48, 2312, fft45_line", "1"), ("64, 64, 64, 72, 64, 4616, fft64_line", "1")]:
    src = src.replace(f"DEF_SWEEPB({args})", f"DEF_SWEEPB({args}, {mv})")
# SoA iter: map variant 2 for 6/8, 1 for 13/17
src = src.replace("#define DEF_SOA(LN, L_, SP_)", "#define DEF_SOA(LN, L_, SP_, MAPV)")
src = src.replace("""        fft##LN##_line(Xre + off, Xim + off, SP_, 1, Cre + off, Cim + off, 0); \\""",
                  """        fft##LN##_line(Xre + off, Xim + off, SP_, MAPV, Cre + off, Cim + off, 0); \\""")
src = src.replace("DEF_SOA(6, 6, 8 * (36 + 1))", "DEF_SOA(6, 6, 8 * (36 + 1), 2)")
src = src.replace("DEF_SOA(8, 8, 8 * (64 + 1))", "DEF_SOA(8, 8, 8 * (64 + 1), 2)")
src = src.replace("DEF_SOA(13, 13, 8 * (169 + 1))", "DEF_SOA(13, 13, 8 * (169 + 1), 1)")
src = src.replace("DEF_SOA(17, 17, 8 * (289 + 1))", "DEF_SOA(17, 17, 8 * (289 + 1), 1)")
open('implementation.c','w').write(src)
print("ok")
