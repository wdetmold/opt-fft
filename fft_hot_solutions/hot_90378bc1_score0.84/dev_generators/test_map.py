import sys; sys.path.insert(0,'/tmp/g')
from genlib import *
from gen_fg import emit_map
import numpy as np, subprocess, ctypes, os
em = Emitter(); C = Consts()
em('#include <immintrin.h>\n#include <x86intrin.h>\n#include <stdint.h>')
em('static double R[1024+8] __attribute__((aligned(64)));')
em('static double OR_[1024+8] __attribute__((aligned(64)));')
em('static double OI_[1024+8] __attribute__((aligned(64)));')
em('static double I[1024+8] __attribute__((aligned(64)));')
em('static double CR[1024] __attribute__((aligned(64)));')
em('static double CI[1024] __attribute__((aligned(64)));')
for nm, fn in (('m14', emit_map), ('mfs', emit_map_fs)):
    b = Emitter(); CC = Consts()
    b(f'uint64_t bench_{nm}(long reps){{')
    b('uint64_t t0=__rdtsc();')
    b('for(long rp=0;rp<reps;rp++){ __asm__ volatile(""::: "memory");')
    b('for(int i=0;i<128;i++){')
    zr = T(b, '_mm512_add_pd(_mm512_load_pd(R+i*8), _mm512_load_pd(CR+i*8))')
    zi = T(b, '_mm512_add_pd(_mm512_load_pd(I+i*8), _mm512_load_pd(CI+i*8))')
    b('__m512d or_, oi_;')
    fn(b, CC, zr, zi, 'or_', 'oi_')
    b('_mm512_store_pd(OR_+i*8, or_);')   # offset store to avoid exact overwrite dependence
    b('_mm512_store_pd(OI_+i*8, oi_);')
    b('}}')
    b('return __rdtsc()-t0;}')
    em(CC.loads().replace('const __m512d','static const __m512d') if False else '')
    # put const loads inside fn
    lines = b.out().split('\n')
    # insert const loads after function opening
    out = [lines[0]] + CC.loads().split('\n') + lines[1:]
    em('\n'.join(out))
src = em.out()
open('/tmp/g/tm.c','w').write(src)
subprocess.run(['gcc','-O3','-march=native','-shared','-fPIC','/tmp/g/tm.c','-o','/tmp/g/tm.so'],check=True)
os.sched_setaffinity(0,{0})
lib = ctypes.CDLL('/tmp/g/tm.so')
for nm in ('m14','mfs'):
    f = getattr(lib, f'bench_{nm}'); f.restype=ctypes.c_uint64; f.argtypes=[ctypes.c_long]
    f(2000)
    t = f(20000)
    print(f'{nm}: {t/20000/128:.2f} tsc per 8-el map')
