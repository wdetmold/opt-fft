#!/usr/bin/env python3
"""Assemble implementation.c from template + generated prime kernels."""
import sys, re
import pgen

tpl = open('implementation.c').read()

# config via env/args
import os
KB13 = int(os.environ.get('KB13', 3))
KB17 = int(os.environ.get('KB17', 4))
KB23 = int(os.environ.get('KB23', 4))
USE = set(os.environ.get('USE', '13,17,23').split(','))

gen = ["#define VC(x) ((vd)_mm512_set1_pd(x))"]
for p, kb in ((13,KB13),(17,KB17),(23,KB23)):
    if str(p) in USE:
        gen.append(pgen.gen_prime(p, kb))
gencode = "\n".join(gen)

# insert generated kernels just before GEN_WRAPPERS definition
anchor = "#define GEN_WRAPPERS(KK, LL, PF)"
assert anchor in tpl
tpl = tpl.replace(anchor, gencode + "\n\n" + anchor)

# add wrappers for generated kernels
wr = "GEN_WRAPPERS(k23, 23, 0)"
assert wr in tpl
add = ""
for p in (13,17,23):
    if str(p) in USE:
        add += f" GEN_WRAPPERS(kg{p}, {p}, 0)"
tpl = tpl.replace(wr, wr + add)

# rebind AoSoA drivers for 13/17
if '13' in USE:
    tpl = tpl.replace("GEN_RUN_AOSOA(13, 169, k13_p, k13_m)",
                      "GEN_RUN_AOSOA(13, 169, kg13_p, kg13_m)")
if '17' in USE:
    tpl = tpl.replace("GEN_RUN_AOSOA(17, 289, k17_p, k17_m)",
                      "GEN_RUN_AOSOA(17, 289, kg17_p, kg17_m)")
# move 23 to AoSoA (lane-major)
if '23' in USE:
    tpl = tpl.replace("GEN_RUN_AOSOA(17, 289, kg17_p, kg17_m)",
                      "GEN_RUN_AOSOA(17, 289, kg17_p, kg17_m)\nGEN_RUN_AOSOA(23, 529, kg23_p, kg23_m)")
    # disable old run23 (GEN_VOL instantiation)
    tpl = tpl.replace("GEN_VOL(23, 3, zpass23, k23_p, k23_m, xgroup_23)",
                      "/* GEN_VOL 23 disabled: lane-major */")

open('impl_gen.c','w').write(tpl)
print("wrote impl_gen.c", len(tpl))
