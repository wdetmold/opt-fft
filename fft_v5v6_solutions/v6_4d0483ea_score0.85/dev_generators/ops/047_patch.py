src = open('implementation.c').read()
src = src.replace("#ifndef U36S1\n#define U36S1 1\n#endif", "#ifndef U36S1\n#define U36S1 4\n#endif")
src = src.replace("#ifndef U36S2\n#define U36S2 1\n#endif", "#ifndef U36S2\n#define U36S2 9\n#endif")
src = src.replace("""static inline __attribute__((always_inline)) void tile_load(const double *b, long ld,
                                                            double *s, int ncb) {
  for (int cb = 0; cb < ncb; cb++) {""",
"""#ifndef UTILE
#define UTILE 16
#endif
static inline __attribute__((always_inline)) void tile_load(const double *b, long ld,
                                                            double *s, int ncb) {
  PRAGMA_UNROLL(UTILE) for (int cb = 0; cb < ncb; cb++) {""")
src = src.replace("""static inline __attribute__((always_inline)) void tile_store(double *b, long ld,
                                                             const double *s, int ncb, int L) {
  for (int cb = 0; cb < ncb; cb++) {""",
"""static inline __attribute__((always_inline)) void tile_store(double *b, long ld,
                                                             const double *s, int ncb, int L) {
  PRAGMA_UNROLL(UTILE) for (int cb = 0; cb < ncb; cb++) {""")
# PRAGMA_UNROLL must be defined before tile_load: move the PSTR/PRAGMA_UNROLL defs up
src = src.replace("""#define PSTR(x) #x
#define PRAGMA_UNROLL(n) _Pragma(PSTR(GCC unroll n))
// ---------------------------------------------------------------- codelet cores""",
"// ---------------------------------------------------------------- codelet cores")
src = src.replace("// ---------------------------------------------------------------- 8x8 double transpose",
"""#define PSTR(x) #x
#define PRAGMA_UNROLL(n) _Pragma(PSTR(GCC unroll n))
// ---------------------------------------------------------------- 8x8 double transpose""")
open('implementation.c','w').write(src)
print("ok")
