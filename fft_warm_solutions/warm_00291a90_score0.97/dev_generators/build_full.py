import sys, importlib.util
spec = importlib.util.spec_from_file_location("f40gen", "/tmp/w/f40/dev_generators/gen.py")
f40gen = importlib.util.module_from_spec(spec)
import numpy as np
spec.loader.exec_module(f40gen)

import gen_main
from gen_main import gen_conv, gen_driver, PRELUDE as MYPRE
from gen_a import gen_composite
from gen_asm_prime import gen_prime_asm

def build_full():
    parts = [f40gen.PRELUDE]   # includes alloc_huge, TR8, map2, map_range, DEINT/INTER, FMA_BC...
    # my A sizes
    KB = {13:(6,6),17:(8,8),23:(6,6)}
    for L in (6,8,13,17,23):
        PS = L*L if (L*L) % 2 == 1 else L*L+1
        if L in (6,8):
            parts.append(gen_composite(L, PS*16))
            parts.append(gen_conv(L, PS))
            parts.append(gen_driver(L, PS, pair_y=True))
        else:
            parts.append(gen_prime_asm(L, KB[L][0], KB[L][1], PS))
            parts.append(gen_conv(L, PS))
            parts.append(gen_driver(L, PS, prime_asm=True))
    # f40 B sizes
    parts.append(f40gen.build_B_sizes())
    return "\n".join(parts)

if __name__ == "__main__":
    src = build_full()
    open("implementation.c","w").write(src)
    print("wrote implementation.c", len(src))
