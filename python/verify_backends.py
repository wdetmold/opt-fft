"""Check every installed FFT backend against the from-scratch reference.

`slow_dft.dft3d_batched` is the ground truth: no library FFT is involved in it, so
agreement here means a backend is correctly installed *and* uses the conventions we
assume (forward = exp(-2 pi i k.j/n), unnormalized, batch index slowest).

Run:  source env.sh && python python/verify_backends.py
"""

import numpy as np

from slow_dft import dft3d_batched, random_field, rel_error

L, B = 8, 3
AXES = (-3, -2, -1)


def backends():
    """Yield (name, callable) for each backend that imports on this machine."""
    yield "numpy.fft", lambda x: np.fft.fftn(x, axes=AXES)

    try:
        import scipy.fft
        yield "scipy.fft", lambda x: scipy.fft.fftn(x, axes=AXES, workers=-1)
    except ImportError:
        pass

    try:
        import ducc0.fft
        yield "ducc0", lambda x: ducc0.fft.c2c(x, axes=AXES, forward=True, nthreads=0)
    except ImportError:
        pass

    try:
        import pyfftw
        yield "pyfftw (FFTW3)", lambda x: pyfftw.interfaces.numpy_fft.fftn(
            x, axes=AXES, threads=8)
    except ImportError:
        pass

    try:
        import mkl_fft
        yield "mkl_fft (DFTI)", lambda x: mkl_fft.fftn(x, axes=AXES)
    except ImportError:
        pass

    try:
        import cupy
        yield "cupy (cuFFT)", lambda x: cupy.asnumpy(
            cupy.fft.fftn(cupy.asarray(x), axes=AXES))
    except ImportError:
        pass

    try:
        import cupy
        from pyvkfft.fft import fftn as vk_fftn
        # VkFFT is single-precision-friendly; feed it complex64 and compare loosely.
        yield "pyvkfft (VkFFT)", lambda x: cupy.asnumpy(
            vk_fftn(cupy.asarray(x.astype(np.complex64)), ndim=3)).astype(np.complex128)
    except ImportError:
        pass


if __name__ == "__main__":
    x = random_field((L, L, L), batch=B, seed=1)
    ref = dft3d_batched(x)
    print(f"reference: slow_dft.dft3d_batched, B={B}, L={L}  "
          f"(||X|| = {np.linalg.norm(ref):.4g})\n")
    print(f"  {'backend':18s} {'rel L2 error':>13s}   verdict")
    for name, fn in backends():
        try:
            err = rel_error(fn(x), ref)
            # complex64 paths land near 1e-7, complex128 paths near 1e-15
            ok = err < 1e-5
            print(f"  {name:18s} {err:13.3e}   {'ok' if ok else 'MISMATCH'}")
        except Exception as exc:
            print(f"  {name:18s} {'--':>13s}   FAILED: {type(exc).__name__}: {exc}")
