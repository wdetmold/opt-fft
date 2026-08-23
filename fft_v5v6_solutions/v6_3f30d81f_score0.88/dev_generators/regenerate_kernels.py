#!/usr/bin/env python3
"""Regenerate the prime-DFT kernel section of ../implementation.c.

The graded implementation.c embeds generated kernels for the odd primes
13/17/23.  The agent produced them with genprime2.py and spliced them in
between the marker comment "// generated: 13-point" and the start of the
hand-written k36 kernel.  This script reproduces that splice exactly
(config "13:2:both,17:3:cos,23:2:cos" -- the shipped v5 variant).

Verified during reconstruction: the regenerated text is byte-identical to
the section already present in ../implementation.c.
"""
import subprocess
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
IMPL = os.path.join(HERE, "..", "implementation.c")
CFG = "13:2:both,17:3:cos,23:2:cos"

subprocess.run([sys.executable, os.path.join(HERE, "genprime2.py"), CFG],
               cwd=HERE, check=True)
gen = open(os.path.join(HERE, "primegen2.c")).read()

src = open(IMPL).read()
start = src.index("// generated: 13-point")
end = src.index("static __attribute__((always_inline)) inline\nvoid k36(")
new = src[:start] + gen + "\n" + src[end:]
if new == src:
    print("implementation.c already contains the exact generated section")
else:
    open(IMPL, "w").write(new)
    print("implementation.c kernel section regenerated")
