src = open('implementation.c').read()
src = src.replace("""  PRAGMA_UNROLL(UTILE) for (int cb = 0; cb < ncb; cb++) {
    PF1(b + (PFD + 0) * ld + cb * 8); PF1(b + (PFD + 1) * ld + cb * 8);
    PF1(b + (PFD + 2) * ld + cb * 8); PF1(b + (PFD + 3) * ld + cb * 8);
    PF1(b + (PFD + 4) * ld + cb * 8); PF1(b + (PFD + 5) * ld + cb * 8);
    PF1(b + (PFD + 6) * ld + cb * 8); PF1(b + (PFD + 7) * ld + cb * 8);""",
"""  PRAGMA_UNROLL(UTILE) for (int cb = 0; cb < ncb; cb++) {
    PF1(b + (PFD + 0) * ld + cb * 8); PF1(b + (PFD + 1) * ld + cb * 8);
    PF1(b + (PFD + 2) * ld + cb * 8); PF1(b + (PFD + 3) * ld + cb * 8);
    PF1(b + (PFD + 4) * ld + cb * 8); PF1(b + (PFD + 5) * ld + cb * 8);
    PF1(b + (PFD + 6) * ld + cb * 8); PF1(b + (PFD + 7) * ld + cb * 8);
#if PFD2
    _mm_prefetch((const char *)(b + (PFD2 + 0) * ld + cb * 8), _MM_HINT_T1);
    _mm_prefetch((const char *)(b + (PFD2 + 1) * ld + cb * 8), _MM_HINT_T1);
    _mm_prefetch((const char *)(b + (PFD2 + 2) * ld + cb * 8), _MM_HINT_T1);
    _mm_prefetch((const char *)(b + (PFD2 + 3) * ld + cb * 8), _MM_HINT_T1);
    _mm_prefetch((const char *)(b + (PFD2 + 4) * ld + cb * 8), _MM_HINT_T1);
    _mm_prefetch((const char *)(b + (PFD2 + 5) * ld + cb * 8), _MM_HINT_T1);
    _mm_prefetch((const char *)(b + (PFD2 + 6) * ld + cb * 8), _MM_HINT_T1);
    _mm_prefetch((const char *)(b + (PFD2 + 7) * ld + cb * 8), _MM_HINT_T1);
#endif""")
src = src.replace("#ifndef PFD\n#define PFD 8\n#endif", "#ifndef PFD\n#define PFD 8\n#endif\n#ifndef PFD2\n#define PFD2 0\n#endif")
open('implementation.c','w').write(src)
print("ok")
