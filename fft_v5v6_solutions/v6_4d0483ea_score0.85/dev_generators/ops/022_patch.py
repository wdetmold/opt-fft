src = open('implementation.c').read()
old = src[src.index("static inline __attribute__((always_inline)) void\nfft13_line"):src.index("static inline __attribute__((always_inline)) void\nfft17_line")]
new = '''static inline __attribute__((always_inline)) void
fft13_line(double *re, double *im, const long s, const int domap,
           const double *cre, const double *cim, const double *pf) { (void)pf;
  DEFCONSTS
  v8 x0r = VL(re), x0i = VL(im);
#define LDAB(j, a, b)                                              \\
  v8 a##r, a##i, b##r, b##i;                                       \\
  {                                                                \\
    v8 _ur = VL(re + (j) * s), _ui = VL(im + (j) * s);             \\
    v8 _vr = VL(re + (13 - (j)) * s), _vi = VL(im + (13 - (j)) * s); \\
    a##r = _ur + _vr; a##i = _ui + _vi;                            \\
    b##r = _ur - _vr; b##i = _ui - _vi;                            \\
  }
  LDAB(1, a1, b1) LDAB(2, a2, b2) LDAB(3, a3, b3)
  LDAB(4, a4, b4) LDAB(5, a5, b5) LDAB(6, a6, b6)
#undef LDAB
  {
    v8 sr = x0r + a1r + a2r + a3r + a4r + a5r + a6r;
    v8 si = x0i + a1i + a2i + a3i + a4i + a5i + a6i;
    STORE_X(re, im, sr, si, 0);
  }
#define KROW(k)                                                               \\
  {                                                                           \\
    const double *ct = C13 + ((k) - 1) * 6, *st = S13 + ((k) - 1) * 6;        \\
    v8 crr = x0r, cii = x0i, srr, sii;                                        \\
    crr += vbc(ct[0]) * a1r; cii += vbc(ct[0]) * a1i;                         \\
    srr = vbc(st[0]) * b1r; sii = vbc(st[0]) * b1i;                           \\
    crr += vbc(ct[1]) * a2r; cii += vbc(ct[1]) * a2i;                         \\
    srr += vbc(st[1]) * b2r; sii += vbc(st[1]) * b2i;                         \\
    crr += vbc(ct[2]) * a3r; cii += vbc(ct[2]) * a3i;                         \\
    srr += vbc(st[2]) * b3r; sii += vbc(st[2]) * b3i;                         \\
    crr += vbc(ct[3]) * a4r; cii += vbc(ct[3]) * a4i;                         \\
    srr += vbc(st[3]) * b4r; sii += vbc(st[3]) * b4i;                         \\
    crr += vbc(ct[4]) * a5r; cii += vbc(ct[4]) * a5i;                         \\
    srr += vbc(st[4]) * b5r; sii += vbc(st[4]) * b5i;                         \\
    crr += vbc(ct[5]) * a6r; cii += vbc(ct[5]) * a6i;                         \\
    srr += vbc(st[5]) * b6r; sii += vbc(st[5]) * b6i;                         \\
    v8 xr1 = crr + sii, xi1 = cii - srr;                                      \\
    v8 xr2 = crr - sii, xi2 = cii + srr;                                      \\
    STORE_X(re, im, xr1, xi1, (k) * s);                                       \\
    STORE_X(re, im, xr2, xi2, (13 - (k)) * s);                                \\
  }
  KROW(1) KROW(2) KROW(3) KROW(4) KROW(5) KROW(6)
#undef KROW
}
'''
src = src.replace(old, new)
open('implementation.c','w').write(src)
print("ok")
