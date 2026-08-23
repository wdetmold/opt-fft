src = open('implementation.c').read()
src = src.replace("#ifdef MAP_SQRT\n", "#if 1\n")
open('implementation.c','w').write(src)
