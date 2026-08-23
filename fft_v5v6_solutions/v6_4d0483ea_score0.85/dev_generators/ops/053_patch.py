src = open('implementation.c').read()
src = src.replace("#define PH17 8", "#define PH17 12")
src = src.replace("DEF_SOA(23, 23, 8 * (529 + 1))\n", "")
open('implementation.c','w').write(src)
