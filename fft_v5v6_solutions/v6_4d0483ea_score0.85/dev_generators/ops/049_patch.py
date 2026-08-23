src = open('implementation.c').read()
old = """  KROW(1) KROW(2) KROW(3) KROW(4) KROW(5) KROW(6)
#undef KROW"""
new = """#if JROLL13
  for (int k = 1; k <= 6; k++) KROW(k)
#else
  KROW(1) KROW(2) KROW(3) KROW(4) KROW(5) KROW(6)
#endif
#undef KROW"""
src = src.replace(old, new)
src = src.replace("#ifndef JP17", "#ifndef JROLL13\n#define JROLL13 0\n#endif\n#ifndef JP17")
open('implementation.c','w').write(src)
print("ok")
