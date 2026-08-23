src = open('implementation.c').read()
for L in (17,23):
    src = src.replace(f"""fft{L}_line(double *re, double *im, const long s, const int domap,
           const double *cre, const double *cim, const double *pf) {{ (void)pf;""",
f"""fft{L}_line(double *re, double *im, const long s, const int domap,
           const double *cre, const double *cim, const double *pf) {{""")
open('implementation.c','w').write(src)
