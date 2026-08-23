src = open('implementation.c').read()
src = src.replace("""    double *ro = re + k2 * s, *io = im + k2 * s;
    const double *cro = cre + k2 * s, *cio = cim + k2 * s;
    (void)cro; (void)cio;
    STORE_X(ro, io, x0r, x0i, 0 * 8 * s - k2 * s + (0 * 8 + k2) * s);
    STORE_X(ro, io, x1r, x1i, (1 * 8 + k2) * s - k2 * s);
    STORE_X(ro, io, x2r, x2i, (2 * 8 + k2) * s - k2 * s);
    STORE_X(ro, io, x3r, x3i, (3 * 8 + k2) * s - k2 * s);
    STORE_X(ro, io, x4r, x4i, (4 * 8 + k2) * s - k2 * s);
    STORE_X(ro, io, x5r, x5i, (5 * 8 + k2) * s - k2 * s);
    STORE_X(ro, io, x6r, x6i, (6 * 8 + k2) * s - k2 * s);
    STORE_X(ro, io, x7r, x7i, (7 * 8 + k2) * s - k2 * s);""",
"""    STORE_X(re, im, x0r, x0i, (0 * 8 + k2) * s);
    STORE_X(re, im, x1r, x1i, (1 * 8 + k2) * s);
    STORE_X(re, im, x2r, x2i, (2 * 8 + k2) * s);
    STORE_X(re, im, x3r, x3i, (3 * 8 + k2) * s);
    STORE_X(re, im, x4r, x4i, (4 * 8 + k2) * s);
    STORE_X(re, im, x5r, x5i, (5 * 8 + k2) * s);
    STORE_X(re, im, x6r, x6i, (6 * 8 + k2) * s);
    STORE_X(re, im, x7r, x7i, (7 * 8 + k2) * s);""")
open('implementation.c','w').write(src)
print("fixed")
