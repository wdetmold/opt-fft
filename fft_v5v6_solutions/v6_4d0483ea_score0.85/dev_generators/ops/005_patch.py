src = open('implementation.c').read()
src = src.replace('''  static void sweepA_##LN(void) {''', '''  SWEEPVIS void sweepA_##LN(void) {''')
src = src.replace('''  static void sweepB_##LN(void) {''', '''  SWEEPVIS void sweepB_##LN(void) {''')
src = src.replace('// ---------------------------------------------------------------- engines',
'''// ---------------------------------------------------------------- engines
#ifdef EXPOSE_SWEEPS
#define SWEEPVIS
#else
#define SWEEPVIS static
#endif''')
open('implementation.c','w').write(src)
