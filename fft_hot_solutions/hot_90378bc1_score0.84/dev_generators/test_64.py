import ctypes, numpy as np, os, time
os.sched_setaffinity(0,{0})
lib = ctypes.CDLL('/tmp/g/impl_64.so')
lib.init_64()
f = lib.run_64; f.argtypes=[ctypes.c_void_p]*4+[ctypes.c_long]*2
B,m = 2,3
rng = np.random.default_rng(7)
x0 = rng.standard_normal((B,64,64,64)) + 1j*rng.standard_normal((B,64,64,64))
c = 0.1*(rng.standard_normal((B,64,64,64)) + 1j*rng.standard_normal((B,64,64,64)))
one = np.zeros(B*64**3, complex); fin = np.zeros(B*64**3, complex)
f(x0.ctypes.data, c.ctypes.data, one.ctypes.data, fin.ctypes.data, B, m)
x = x0.copy(); on=None
for k in range(m):
    z = np.fft.fftn(x, axes=(1,2,3)) + c
    x = z/(1+np.abs(z))
    if k==0: on=x.copy()
print('one rel', np.linalg.norm(one-on.ravel())/np.linalg.norm(on))
print('fin rel', np.linalg.norm(fin-x.ravel())/np.linalg.norm(x))
