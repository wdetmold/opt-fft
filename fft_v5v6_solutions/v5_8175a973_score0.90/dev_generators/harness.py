import os, ctypes
import numpy as np
_here = os.path.dirname(os.path.abspath(__file__))
_lib = ctypes.CDLL(os.path.join(_here, 'implementation.so'))
_lib.run_size.argtypes = [ctypes.c_long]*3 + [ctypes.c_void_p]*4 + [ctypes.c_long]
_lib.run_size.restype = None

# default mode per L: 0=batched (lanes=volumes), 1=per-volume
MODE = {6:0, 8:0, 13:0, 17:0, 23:0, 36:0, 45:1, 64:1}

def run(L, B, m, x0, c, mode=None):
    x0 = np.ascontiguousarray(x0, dtype=np.complex128)
    c = np.ascontiguousarray(c, dtype=np.complex128)
    n = B * L**3
    one = np.empty(n, dtype=np.complex128)
    fin = np.empty(n, dtype=np.complex128)
    md = MODE[L] if mode is None else mode
    _lib.run_size(L, B, m, x0.ctypes.data, c.ctypes.data,
                  one.ctypes.data, fin.ctypes.data, md)
    return one, fin

def ref_np(L, B, m, x0, c):
    x = x0.copy()
    one = None
    for k in range(m):
        z = np.fft.fftn(x, axes=(1,2,3)) + c
        x = z / (1.0 + np.abs(z))
        if k == 0: one = x.copy()
    return one.ravel(), x.ravel()

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for L in (6, 8, 13, 17, 23, 36, 45, 64):
        for B in (1, 2, 3, 8, 9):
            for m in (1, 2, 5):
                x0 = rng.standard_normal((B,L,L,L)) + 1j*rng.standard_normal((B,L,L,L))
                c = 0.1*(rng.standard_normal((B,L,L,L)) + 1j*rng.standard_normal((B,L,L,L)))
                modes = [0,1] if L in (13,17,23,36,45) else [None]
                for md in modes:
                    one, fin = run(L, B, m, x0, c, mode=md)
                    r1, rm = ref_np(L, B, m, x0, c)
                    e1 = np.linalg.norm(one-r1)/np.linalg.norm(r1)
                    em = np.linalg.norm(fin-rm)/np.linalg.norm(rm)
                    status = "OK" if e1 < 1e-14 and em < 1e-10 else "FAIL"
                    if status=="FAIL" or (B==1 and m==1):
                        print(f"L={L:2d} B={B} m={m} mode={md} e1={e1:.2e} em={em:.2e} {status}")
    print("done")
