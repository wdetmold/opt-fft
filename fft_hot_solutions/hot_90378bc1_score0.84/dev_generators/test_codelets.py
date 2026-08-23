import sys; sys.path.insert(0,'/tmp/g')
from genlib import *
import numpy as np, subprocess, ctypes

em = Emitter(); C = Consts()
body = Emitter()
for N, fn in ((3,dft3),(9,dft9),(5,dft5),(4,dft4)):
    be = Emitter()
    x = []
    for j in range(N):
        r = T(be, f'_mm512_loadu_pd(re+{j}*8)')
        i = T(be, f'_mm512_loadu_pd(im+{j}*8)')
        x.append((r,i))
    X = fn(be, C, x)
    for k in range(N):
        be(f'_mm512_storeu_pd(ore+{k}*8, {X[k][0]});')
        be(f'_mm512_storeu_pd(oim+{k}*8, {X[k][1]});')
    body(f'void test{N}(const double* re, const double* im, double* ore, double* oim){{')
    body(C.loads())
    body(be.out())
    body('}')
src = '#include <immintrin.h>\n' 
srcC = Emitter(); C.decl(srcC)
src += srcC.out() + body.out()
open('/tmp/g/tc.c','w').write(src)
subprocess.run(['gcc','-O2','-march=native','-shared','-fPIC','/tmp/g/tc.c','-o','/tmp/g/tc.so'],check=True)
lib = ctypes.CDLL('/tmp/g/tc.so')
rng = np.random.default_rng(0)
for N in (3,9,5,4):
    f = getattr(lib,f'test{N}')
    f.argtypes=[ctypes.c_void_p]*4
    x = rng.standard_normal((N,8)) + 1j*rng.standard_normal((N,8))
    re = np.ascontiguousarray(x.real); im = np.ascontiguousarray(x.imag)
    ore = np.zeros((N,8)); oim = np.zeros((N,8))
    f(re.ctypes.data, im.ctypes.data, ore.ctypes.data, oim.ctypes.data)
    ref = np.fft.fft(x, axis=0)
    err = np.abs(ore+1j*oim - ref).max()/np.abs(ref).max()
    print(f'N={N}: rel err {err:.2e}')
