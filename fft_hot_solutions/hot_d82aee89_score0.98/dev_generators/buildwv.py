from glib import Emit, PRELUDE
import genwv
TR8MACRO = open('tr8.inc').read()
src = PRELUDE + TR8MACRO
for L in (36, 45, 64):
    src += genwv.gen_engine(L)
open("../implwv.c","w").write(src)
print("wrote implwv.c")
