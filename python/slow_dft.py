"""Reference (deliberately slow) complex DFTs written from scratch.

These are the correctness ground truth for everything else in this project: no
library calls beyond basic numpy array arithmetic, no FFT factorization.  Three
levels of naivety are provided, in increasing speed and decreasing obviousness:

  dft3d_loops      definitional 6-fold sum,   O(V^2)          -- L <= 8 only
  dft3d_matrix     dense V x V matrix apply,  O(V^2)          -- L <= 16 only
  dft3d_separable  row-column, 1D matrix per axis, O(V*sum(n)) -- L <= 64 fine

Conventions (chosen to match numpy.fft / FFTW so validation is a direct compare):

  forward   X[k] = sum_j x[j] exp(-2 pi i k.j / n)          (unnormalized)
  backward  x[j] = (1/V) sum_k X[k] exp(+2 pi i k.j / n)    (1/V on the inverse)

with k.j/n meaning sum_d k_d j_d / n_d, and V = prod(n_d).  Momentum index k_d
runs 0..n_d-1 in the usual wrap-around ordering (k_d > n_d/2 is negative
momentum); use numpy.fft.fftfreq-style shifts if you want it centered.

Batched layout is (B, n0, n1, n2): the batch index is the slowest-varying, which
is what the target geometry (many small L^3 transforms over time slices /
spin-color components) wants.
"""

import numpy as np

FORWARD = -1  # sign in the exponent
BACKWARD = +1


def dft_matrix(n, sign=FORWARD, dtype=np.complex128):
    """The n x n DFT matrix W[k, j] = exp(sign * 2 pi i k j / n)."""
    idx = np.arange(n)
    return np.exp(sign * 2j * np.pi * np.outer(idx, idx) / n).astype(dtype)


# ---------------------------------------------------------------- 1D

def dft1d_loops(x, sign=FORWARD):
    """Definitional 1D DFT with explicit loops.  O(n^2) scalar operations."""
    x = np.asarray(x, dtype=np.complex128)
    n = x.shape[0]
    out = np.zeros(n, dtype=np.complex128)
    for k in range(n):
        acc = 0.0 + 0.0j
        for j in range(n):
            acc += x[j] * np.exp(sign * 2j * np.pi * k * j / n)
        out[k] = acc
    return out


# ---------------------------------------------------------------- 3D

def dft3d_loops(x, sign=FORWARD):
    """Definitional 3D DFT as a 6-fold sum.  O(V^2); keep L <= 8.

    This is the equation itself, with nothing hidden: every output mode is an
    independent sum over the whole volume.
    """
    x = np.asarray(x, dtype=np.complex128)
    n0, n1, n2 = x.shape
    out = np.zeros_like(x)
    for k0 in range(n0):
        for k1 in range(n1):
            for k2 in range(n2):
                acc = 0.0 + 0.0j
                for j0 in range(n0):
                    for j1 in range(n1):
                        for j2 in range(n2):
                            phase = k0 * j0 / n0 + k1 * j1 / n1 + k2 * j2 / n2
                            acc += x[j0, j1, j2] * np.exp(sign * 2j * np.pi * phase)
                out[k0, k1, k2] = acc
    return out


def dft3d_matrix(x, sign=FORWARD):
    """Dense V x V matrix-vector product.  Same O(V^2) work as dft3d_loops,
    but handed to BLAS, so it is usable a little further out in L."""
    x = np.asarray(x, dtype=np.complex128)
    n0, n1, n2 = x.shape
    j = np.array(np.meshgrid(np.arange(n0), np.arange(n1), np.arange(n2),
                             indexing="ij")).reshape(3, -1)
    n = np.array([n0, n1, n2])[:, None]
    phase = (j / n).T @ j  # V x V matrix of k.j/n
    return (np.exp(sign * 2j * np.pi * phase) @ x.reshape(-1)).reshape(n0, n1, n2)


def _apply_axis_matrix(x, W, axis):
    """Contract W (m x n) against `axis` of x, leaving the result on that axis."""
    return np.moveaxis(np.tensordot(W, x, axes=([1], [axis])), 0, axis)


def dft3d_separable(x, sign=FORWARD, axes=(-3, -2, -1)):
    """Row-column algorithm: a 1D DFT matrix applied along each axis in turn.

    Still no FFT factorization -- each 1D transform is the dense O(n^2) matrix --
    but exploiting separability drops the total from O(V^2) to O(V * sum n_d),
    i.e. 3 L^4 instead of L^6.  Works unchanged on batched input, since only the
    three named axes are touched.
    """
    x = np.asarray(x, dtype=np.complex128)
    out = x
    for axis in axes:
        n = x.shape[axis]
        out = _apply_axis_matrix(out, dft_matrix(n, sign), axis)
    return out


def idft3d_separable(x, axes=(-3, -2, -1)):
    """Inverse of dft3d_separable, with the 1/V normalization."""
    x = np.asarray(x, dtype=np.complex128)
    volume = int(np.prod([x.shape[a] for a in axes]))
    return dft3d_separable(x, sign=BACKWARD, axes=axes) / volume


def dft3d_batched(x, sign=FORWARD):
    """Batched 3D DFT for input shaped (B, n0, n1, n2) (or plain (n0, n1, n2)).

    The whole batch rides along in the same matrix contractions, which is how the
    library backends will be asked to do it too, so this is the reference that
    the batched benchmarks compare against.
    """
    x = np.asarray(x, dtype=np.complex128)
    if x.ndim == 3:
        return dft3d_separable(x, sign)
    if x.ndim != 4:
        raise ValueError(f"expected (B, n0, n1, n2) or (n0, n1, n2), got {x.shape}")
    return dft3d_separable(x, sign, axes=(-3, -2, -1))


# ---------------------------------------------------------------- helpers

def flop_count(shape, batch=1):
    """Textbook 5 N log2 N complex-FFT flop count, for reporting GFLOP/s.

    This is the *FFT* cost model, not the cost of the slow routines above; it is
    the standard yardstick every FFT library quotes, so benchmarks stay
    comparable to published numbers.
    """
    volume = int(np.prod(shape))
    return 5.0 * batch * volume * np.log2(volume)


def random_field(shape, batch=None, seed=0, dtype=np.complex128):
    """Reproducible complex test field."""
    rng = np.random.default_rng(seed)
    full = tuple(shape) if batch is None else (batch,) + tuple(shape)
    return (rng.standard_normal(full) + 1j * rng.standard_normal(full)).astype(dtype)


def rel_error(a, b):
    """Relative L2 difference, the accuracy metric used throughout."""
    a, b = np.asarray(a), np.asarray(b)
    denom = np.linalg.norm(b.reshape(-1))
    return float(np.linalg.norm((a - b).reshape(-1)) / (denom if denom else 1.0))


if __name__ == "__main__":
    import time

    print("validating against numpy.fft (which is *not* used inside the routines)")
    L = 6
    x = random_field((L, L, L))
    ref = np.fft.fftn(x)
    for name, fn in [("dft3d_loops", dft3d_loops),
                     ("dft3d_matrix", dft3d_matrix),
                     ("dft3d_separable", dft3d_separable)]:
        t0 = time.perf_counter()
        got = fn(x)
        dt = time.perf_counter() - t0
        print(f"  {name:16s} rel_err={rel_error(got, ref):.3e}  {dt*1e3:8.2f} ms")

    print(f"  {'roundtrip':16s} rel_err="
          f"{rel_error(idft3d_separable(dft3d_separable(x)), x):.3e}")

    B, L = 4, 8
    xb = random_field((L, L, L), batch=B)
    print(f"  {'batched':16s} rel_err="
          f"{rel_error(dft3d_batched(xb), np.fft.fftn(xb, axes=(-3, -2, -1))):.3e}"
          f"   (B={B}, L={L})")

    print("\nscaling of the separable reference (forward, double):")
    for L in (4, 8, 12, 16, 24, 32):
        x = random_field((L, L, L))
        t0 = time.perf_counter()
        dft3d_separable(x)
        dt = time.perf_counter() - t0
        print(f"  L={L:3d}  V={L**3:8d}  {dt*1e3:9.2f} ms"
              f"   (3L^4 = {3*L**4:.3g} complex mults)")
