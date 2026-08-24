import ctypes, numpy as np
lib = ctypes.CDLL('./codelets_test.so')
for fn in (lib.run_test, lib.run_test1):
    fn.argtypes = [ctypes.c_int] + [ctypes.POINTER(ctypes.c_double)]*4
rng = np.random.default_rng(0)
for n in (2,3,4,5,6,8,13,17,23,36,45,64):
    for which, fn, W in (('w8', lib.run_test, 8), ('w1', lib.run_test1, 1)):
        x = rng.standard_normal((n, W)) + 1j*rng.standard_normal((n, W))
        ri = np.ascontiguousarray(x.real); ii = np.ascontiguousarray(x.imag)
        ro = np.zeros((n, W)); io = np.zeros((n, W))
        pp = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        fn(n, pp(ri), pp(ii), pp(ro), pp(io))
        ref = np.fft.fft(x, axis=0)
        got = ro + 1j*io
        err = np.linalg.norm(got - ref)/np.linalg.norm(ref)
        stat = "OK " if err < 1e-14 else "BAD"
        print(f"n={n:3d} {which}: rel err {err:.3e} {stat}")
