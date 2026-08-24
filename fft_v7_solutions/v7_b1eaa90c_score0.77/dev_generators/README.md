# dev/ -- generator and tooling for implementation.c

- `gen.py`        : generates ../implementation.c (run from this directory: `python3 gen.py`).
                    Config via GENCFG env (JSON) overrides baked defaults.
- `codelets.py`   : straight-line small-DFT emitters (2,3,4,5,6,8,9 + halfmatrix + CT/PFA combiners),
                    exact twiddles from numpy longdouble.
- `kernels.py`    : looped/instanced column-FFT kernels with baked strides, fused pointwise stores,
                    square-CT stage kernels, in-register 64-point line FFT.
- `check.py`      : correctness harness vs /workdir/base.py (per-block gates).
- `tune.py`       : per-size ns/element-iteration benchmark via ctypes.
- `test_cores.py` : unit tests of DFT cores vs numpy.

implementation.c is committed; grading does not need to run the generator.
