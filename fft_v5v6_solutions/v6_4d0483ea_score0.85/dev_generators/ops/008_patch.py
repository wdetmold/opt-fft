src = open('implementation.c').read()
src = src.replace("""static inline __attribute__((always_inline)) void tile_load(const double *b, long ld,
                                                            double *s, int ncb) {
  for (int cb = 0; cb < ncb; cb++) {
    v8 r0 = VL(b + 0 * ld + cb * 8), r1 = VL(b + 1 * ld + cb * 8);""",
"""static inline __attribute__((always_inline)) void tile_load(const double *b, long ld,
                                                            double *s, int ncb) {
  for (int cb = 0; cb < ncb; cb++) {
    PF1(b + (8 + 0) * ld + cb * 8); PF1(b + (8 + 1) * ld + cb * 8);
    PF1(b + (8 + 2) * ld + cb * 8); PF1(b + (8 + 3) * ld + cb * 8);
    PF1(b + (8 + 4) * ld + cb * 8); PF1(b + (8 + 5) * ld + cb * 8);
    PF1(b + (8 + 6) * ld + cb * 8); PF1(b + (8 + 7) * ld + cb * 8);
    v8 r0 = VL(b + 0 * ld + cb * 8), r1 = VL(b + 1 * ld + cb * 8);""")
open('implementation.c','w').write(src)
print("ok")
