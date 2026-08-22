# Numeric mirror backend: values are (re,im) numpy arrays; verifies algorithm structure.
import numpy as np
from kernels import dft

class NumBE:
    def cadd(self, a, b): return (a[0] + b[0], a[1] + b[1])
    def csub(self, a, b): return (a[0] - b[0], a[1] - b[1])
    def csum(self, vs):
        # pairwise tree
        vs = list(vs)
        while len(vs) > 1:
            nxt = []
            for i in range(0, len(vs) - 1, 2):
                nxt.append(self.cadd(vs[i], vs[i + 1]))
            if len(vs) % 2: nxt.append(vs[-1])
            vs = nxt
        return vs[0]
    def racc(self, terms):
        # sum of real_const * value  (chained fma)
        r = terms[0][0] * terms[0][1][0]
        i = terms[0][0] * terms[0][1][1]
        for c, v in terms[1:]:
            r = r + c * v[0]; i = i + c * v[1]
        return (r, i)
    def cmix(self, a, b, sign):
        # a + sign*i*b  -> (a.re - sign*b.im, a.im + sign*b.re)
        if sign < 0: return (a[0] + b[1], a[1] - b[0])
        else:        return (a[0] - b[1], a[1] + b[0])
    def cmulw(self, a, w):
        wr, wi = w
        return (wr * a[0] - wi * a[1], wr * a[1] + wi * a[0])

def run_dft(L, x):
    be = NumBE()
    xs = [(x[j].real.copy(), x[j].imag.copy()) for j in range(L)]
    out = dft(be, xs, L)
    return np.array([o[0] + 1j * o[1] for o in out])

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for L in (2, 3, 4, 5, 6, 8, 9, 13, 17, 23, 36, 45, 64):
        x = rng.standard_normal((L, 7)) + 1j * rng.standard_normal((L, 7))
        got = run_dft(L, x)
        ref = np.fft.fft(x, axis=0)
        err = np.linalg.norm(got - ref) / np.linalg.norm(ref)
        print(f"L={L:3d} rel_err={err:.3e}")
        assert err < 1e-14, L
    print("ALL MIRROR KERNELS OK")
