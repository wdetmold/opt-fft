# splice primekerns into the f40 implementation, renaming calls
src = open('base_impl.c').read()
kern = open('primekerns.h').read()
# insert kernels right after the map_range definition (before dft6_v)
anchor = "static const __m512i IDX_EVEN_"
i = src.index(anchor)
src = src[:i] + kern + "\n" + src[i:]
# redirect calls: replace call sites dft13_v( -> dftp13_v( etc. but NOT the definitions
import re
for N in (13,17,23):
    # keep the original definitions (unused); just swap call sites in S_/P_ functions and anywhere else
    src = src.replace(f"dft{N}_v(", f"dftp{N}_v(")
    # restore the definition line
    src = src.replace(f"static void dftp{N}_v(double* re, double* im, long es){{", f"static void dft{N}_v(double* re, double* im, long es){{", 1)
open('implementation.c','w').write(src)
print("patched")