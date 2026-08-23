import ctypes, numpy as np
lib = ctypes.CDLL("/workdir/dev/implementation.so")
rng = np.random.default_rng(1)
for L in (6, 8, 13, 17, 23, 36, 45, 64):
    x = rng.standard_normal((L, 8)) + 1j*rng.standard_normal((L, 8))
    re = np.ascontiguousarray(x.real); im = np.ascontiguousarray(x.imag)
    fn = getattr(lib, f"dbg1d_{L}")
    fn(ctypes.c_void_p(re.ctypes.data), ctypes.c_void_p(im.ctypes.data))
    got = re + 1j*im
    ref = np.fft.fft(x, axis=0)
    err = np.linalg.norm(got-ref)/np.linalg.norm(ref)
    print(f"L={L:3d} 1D kernel rel_err={err:.3e}", "OK" if err < 1e-14 else "FAIL")
