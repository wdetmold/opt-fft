#!/usr/bin/env python3
"""Merged generator: v2 (split planes) for most sizes, v1 (interleaved) for 17/36."""
import codegen as g1
import codegen_split as g2

V1_SIZES = (17, 36)
V2_SIZES = (6, 8, 13, 23, 45, 64)
import os
_mapmode = os.environ.get('MAPMODE', 'newton')
NEWTON = {L: (_mapmode != 'sqrt') for L in (6,8,13,17,23,36,45,64)}
if ',' in _mapmode:   # e.g. "sqrt:64,45"
    pass

def main(path='implementation.c'):
    parts = [g2.HEADER]
    # v1 header extras (ld4s etc) without duplicate includes/hp_alloc
    h1 = g1.HEADER
    a = h1.index('static inline __attribute__((always_inline)) __m512d ld4s')
    b = h1.index('#include <sys/mman.h>')
    parts.append(h1[a:b])
    for L in (6, 8, 13, 17, 23, 36, 45, 64):
        parts.append('// ======================= L = %d =======================' % L)
        if L in V1_SIZES:
            parts.append(g1.gen_size(L, newton=NEWTON[L]))
        else:
            parts.append(g2.gen_size(L, newton=(False if L == 64 and _mapmode == 'sqrt64' else NEWTON[L])))
    src = '\n'.join(parts)
    with open(path, 'w') as f:
        f.write(src)
    print('wrote %s: %d lines' % (path, src.count('\n')))

if __name__ == '__main__':
    main()
