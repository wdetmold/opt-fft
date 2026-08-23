src = open('implementation.c').read()
src = src.replace("""#define PF1(p) _mm_prefetch((const char *)(p), _MM_HINT_T0)
#ifndef CPFHINT""","""#ifndef XPFHINT
#define XPFHINT _MM_HINT_T0
#endif
#define PF1(p) _mm_prefetch((const char *)(p), XPFHINT)
#ifndef CPFHINT""")
open('implementation.c','w').write(src)
