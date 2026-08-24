import sys; sys.path.insert(0,'/tmp/g')
from genlib import *
from gen_fg import emit_map
import numpy as np, subprocess, ctypes, os

def build(unroll, fn, nm, em):
    b = Emitter(); CC = Consts()
    b(f'uint64_t bench_{nm}(long reps){{')
    b('uint64_t t0=__rdtsc();')
    b('for(long rp=0;rp<reps;rp++){ __asm__ volatile(""::: "memory");')
    b(f'for(int i=0;i<128;i+={unroll}){{')
    # emit loads for all unroll lanes first, then interleave via separate emitters
    subs = []
    for u in range(unroll):
        bu = Emitter()
        zr = T(bu, f'_mm512_add_pd(_mm512_load_pd(R+(i+{u})*8), _mm512_load_pd(CR+(i+{u})*8))')
        zi = T(bu, f'_mm512_add_pd(_mm512_load_pd(I+(i+{u})*8), _mm512_load_pd(CI+(i+{u})*8))')
        bu(f'__m512d or{u}, oi{u};')
        fn(bu, CC, zr, zi, f'or{u}', f'oi{u}')
        bu(f'_mm512_store_pd(OR_+(i+{u})*8, or{u});')
        bu(f'_mm512_store_pd(OI_+(i+{u})*8, oi{u});')
        subs.append(bu.out().split('\n'))
    # round-robin interleave
    maxlen = max(len(s) for s in subs)
    for k in range(maxlen):
        for s in subs:
            if k < len(s): b(s[k])
    b('}}')
    b('return __rdtsc()-t0;}')
    lines = b.out().split('\n')
    out = [lines[0]] + CC.loads().split('\n') + lines[1:]
    em('\n'.join(out))

em = Emitter()
em('#include <immintrin.h>\n#include <x86intrin.h>\n#include <stdint.h>')
for arr in ('R','I','CR','CI','OR_','OI_'):
    em(f'static double {arr}[1024+64] __attribute__((aligned(64)));')
for unroll in (1,2,4):
    build(unroll, emit_map, f'm14_u{unroll}', em)
    build(unroll, emit_map_fs, f'mfs_u{unroll}', em)
open('/tmp/g/tm2.c','w').write(em.out())
subprocess.run(['gcc','-O3','-march=native','-shared','-fPIC','/tmp/g/tm2.c','-o','/tmp/g/tm2.so'],check=True)
os.sched_setaffinity(0,{0})
lib = ctypes.CDLL('/tmp/g/tm2.so')
for nm in ('m14_u1','m14_u2','m14_u4','mfs_u1','mfs_u2','mfs_u4'):
    f = getattr(lib, f'bench_{nm}'); f.restype=ctypes.c_uint64; f.argtypes=[ctypes.c_long]
    f(2000); t = f(20000)
    print(f'{nm}: {t/20000/128:.2f} tsc per 8-el map')
