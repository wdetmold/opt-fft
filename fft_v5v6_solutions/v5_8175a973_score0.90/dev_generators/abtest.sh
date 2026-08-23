#!/bin/bash
# usage: abtest.sh NACC INLINE
python3 - << PYEOF
src = open('emit.py').read()
import re
src = re.sub(r'NACC = \d+', 'NACC = $1', src)
if '$2' == '1':
    src = src.replace('head = [f"static void {name}({args}){{"]',
                      'head = [f"static inline __attribute__((always_inline)) void {name}({args}){{"]')
else:
    src = src.replace('head = [f"static inline __attribute__((always_inline)) void {name}({args}){{"]',
                      'head = [f"static void {name}({args}){{"]')
open('emit.py','w').write(src)
PYEOF
python3 gen_impl.py > /dev/null && gcc -O3 -march=native -ffp-contract=fast -fno-math-errno codebench.c -o codebench -lm 2>/dev/null && taskset -c 0 ./codebench | grep -E "fft(13|17|23|36|45)"
