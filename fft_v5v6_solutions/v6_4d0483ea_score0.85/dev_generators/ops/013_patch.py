src = open('implementation.c').read()
src = src.replace("#define PF1(p) _mm_prefetch((const char *)(p), _MM_HINT_T0)",
"""#define PF1(p) _mm_prefetch((const char *)(p), _MM_HINT_T0)
#ifndef CPFHINT
#define CPFHINT _MM_HINT_T0
#endif
#define PFC(p) _mm_prefetch((const char *)(p), CPFHINT)""")
# c prefetches are the ones with 2*PLANE / 3*PLANE offsets
import re
src = re.sub(r'PF1\(pf \+ 2 \* PLANE', 'PFC(pf + 2 * PLANE', src)
src = re.sub(r'PF1\(pf \+ 3 \* PLANE', 'PFC(pf + 3 * PLANE', src)
open('implementation.c','w').write(src)
