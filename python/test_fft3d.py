"""Validation of the textbook FFT (`fft3d.py`) against the slow DFT (`slow_dft.py`).

The slow reference is the authority: it is the definition, transcribed, with no
factorization anywhere in it.  `numpy.fft` appears only where the slow reference
is too expensive to reach the size being checked, and never as the sole judge of
a result.

Run:  python3 python/test_fft3d.py          (a few minutes; pure-Python FFT)
"""

import time
import traceback

import numpy as np

import fft3d
import slow_dft
from slow_dft import rel_error

TOL = 1e-12  # double-precision row-column FFT vs the definition, at these sizes

# Geometries chosen to cover every route through the code: powers of two, smooth
# composites, prime powers, bare primes, primes whose (p-1) is itself awkward,
# unit extents, and strongly anisotropic mixtures of all of those.
GEOMETRIES = [
    (1, 1, 1), (2, 2, 2), (4, 4, 4), (8, 8, 8), (16, 16, 16), (32, 8, 4),
    (3, 3, 3), (5, 5, 5), (7, 7, 7), (11, 11, 11), (13, 13, 13),
    (6, 6, 6), (9, 9, 9), (10, 10, 10), (12, 12, 12), (15, 15, 15),
    (49, 2, 2), (25, 4, 3), (27, 2, 2),          # prime powers
    (17, 2, 2), (19, 3, 2), (23, 2, 2), (47, 3, 2),  # primes -> Rader
    (83, 2, 1), (97, 2, 1),                      # p-1 with a large prime factor
    (1, 8, 8), (8, 1, 8), (8, 8, 1), (1, 1, 16), (1, 7, 1),
    (2, 3, 5), (3, 5, 7), (4, 6, 10), (5, 8, 12), (7, 11, 13),
    (13, 5, 8), (16, 3, 5), (24, 18, 12), (30, 30, 30), (2, 2, 64), (64, 2, 2),
    (9, 25, 7), (20, 12, 6), (11, 4, 9),
]

CHECKS = []


def check(fn):
    """Register a check; each returns a short detail string or raises."""
    CHECKS.append(fn)
    return fn


# ------------------------------------------------------------------ 1D

@check
def one_d_exhaustive_vs_definition():
    """Every length 1..40 against the explicit-loop 1D DFT."""
    worst, worst_n = 0.0, None
    for n in range(1, 41):
        x = slow_dft.random_field((n,), seed=n)
        err = rel_error(fft3d.fft1d(x), slow_dft.dft1d_loops(x))
        assert err < TOL, f"n={n}: rel err {err:.3e}"
        if err > worst:
            worst, worst_n = err, n
    return f"n=1..40 all < {TOL:g}, worst {worst:.2e} at n={worst_n}"


@check
def one_d_algorithms_agree():
    """`direct`, `mixed-radix`, `rader`, `bluestein` must all give one answer.

    Each is only legal for some lengths, so this also pins down the guard rails:
    Rader needs a prime, mixed-radix needs a composite, Bluestein takes anything.
    """
    worst, detail = 0.0, []
    for n in [2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 16, 17, 18, 19, 23, 24, 25,
              27, 31, 32, 36, 47, 49, 64, 83, 97, 121, 128]:
        x = slow_dft.random_field((n,), seed=n)
        ref = fft3d.fft1d(x, algorithm="direct")
        routes = ["auto", "bluestein"]
        routes.append("rader" if fft3d.is_prime(n) else "mixed-radix")
        for route in routes:
            err = rel_error(fft3d.fft1d(x, algorithm=route), ref)
            assert err < TOL, f"n={n} {route}: rel err {err:.3e}"
            worst = max(worst, err)
        detail.append(n)
    return f"{len(detail)} lengths x 3 routes agree with the definition, worst {worst:.2e}"


@check
def one_d_inverse_roundtrip():
    """`ifft1d(fft1d(x)) == x` across the same spread of lengths."""
    worst = 0.0
    for n in [1, 2, 5, 7, 12, 16, 17, 30, 47, 64, 83, 97, 100]:
        x = slow_dft.random_field((n,), seed=n)
        worst = max(worst, rel_error(fft3d.ifft1d(fft3d.fft1d(x)), x))
    assert worst < TOL, f"worst roundtrip {worst:.3e}"
    return f"worst {worst:.2e}"


@check
def one_d_bluestein_at_larger_primes():
    """Bluestein and Rader must agree at primes big enough to matter.

    Bluestein's chirp is the accuracy-sensitive part of the whole module (it
    evaluates phases from `t^2`), so it is checked separately at larger `n`
    against Rader, which shares none of its machinery.
    """
    worst, sizes = 0.0, [101, 127, 199, 251]
    for n in sizes:
        x = slow_dft.random_field((n,), seed=n)
        err = rel_error(fft3d.fft1d(x, algorithm="bluestein"),
                        fft3d.fft1d(x, algorithm="rader"))
        assert err < 1e-11, f"n={n}: Bluestein vs Rader {err:.3e}"
        worst = max(worst, err)
    return f"primes {sizes}, worst Bluestein-vs-Rader {worst:.2e}"


# ------------------------------------------------------------------ 3D

@check
def three_d_vs_definitional_sum():
    """Small volumes against `dft3d_loops`, the literal 6-fold sum."""
    worst, cases = 0.0, [(2, 3, 4), (3, 3, 3), (4, 5, 6), (5, 1, 7), (2, 7, 3)]
    for shape in cases:
        x = slow_dft.random_field(shape, seed=sum(shape))
        err = rel_error(fft3d.fftn(x), slow_dft.dft3d_loops(x))
        assert err < TOL, f"{shape}: rel err {err:.3e}"
        worst = max(worst, err)
    return f"{len(cases)} shapes vs the 6-fold sum, worst {worst:.2e}"


@check
def three_d_geometry_sweep():
    """The full geometry list against the slow separable reference."""
    worst, worst_shape = 0.0, None
    for shape in GEOMETRIES:
        x = slow_dft.random_field(shape, seed=abs(hash(shape)) % 2**31)
        err = rel_error(fft3d.fftn(x), slow_dft.dft3d_separable(x))
        assert err < TOL, f"{shape}: rel err {err:.3e}"
        if err > worst:
            worst, worst_shape = err, shape
    return f"{len(GEOMETRIES)} geometries, worst {worst:.2e} at {worst_shape}"


@check
def three_d_inverse_roundtrip():
    """`ifft3d(fft3d(x)) == x`, including anisotropic and prime extents."""
    worst, worst_shape = 0.0, None
    for shape in [(8, 8, 8), (12, 18, 24), (7, 11, 13), (1, 8, 8), (47, 3, 2),
                  (2, 2, 64), (9, 25, 7)]:
        x = slow_dft.random_field(shape, seed=7)
        err = rel_error(fft3d.ifft3d(fft3d.fft3d(x)), x)
        assert err < TOL, f"{shape}: roundtrip {err:.3e}"
        if err > worst:
            worst, worst_shape = err, shape
    return f"worst {worst:.2e} at {worst_shape}"


@check
def three_d_backward_matches_reference():
    """The backward sign must match the slow reference, not just self-invert."""
    worst = 0.0
    for shape in [(6, 4, 8), (5, 7, 3), (16, 2, 9)]:
        x = slow_dft.random_field(shape, seed=3)
        got = fft3d.fftn(x, sign=fft3d.BACKWARD)
        ref = slow_dft.dft3d_separable(x, sign=slow_dft.BACKWARD)
        worst = max(worst, rel_error(got, ref))
    assert worst < TOL, f"backward mismatch {worst:.3e}"
    return f"worst {worst:.2e}"


@check
def batched_transforms():
    """A batch axis must ride along untouched, and match per-slice transforms."""
    worst = 0.0
    for batch, shape in [(3, (8, 8, 8)), (4, (5, 7, 3)), (2, (12, 6, 4))]:
        x = slow_dft.random_field(shape, batch=batch, seed=11)
        got = fft3d.fft3d(x)
        worst = max(worst, rel_error(got, slow_dft.dft3d_batched(x)))
        # and each batch element must equal the transform of that element alone
        for b in range(batch):
            worst = max(worst, rel_error(got[b], fft3d.fftn(x[b])))
    assert worst < TOL, f"batched mismatch {worst:.3e}"
    return f"worst {worst:.2e}"


@check
def arbitrary_axes_and_dimensions():
    """`fftn` over any axes of a higher-rank array, in any order."""
    rng = np.random.default_rng(5)
    x = (rng.standard_normal((3, 4, 2, 5)) + 1j * rng.standard_normal((3, 4, 2, 5)))
    worst = 0.0
    for axes in [(0, 1, 2), (1, 2, 3), (0, 2), (3,), (-1, -3), (2, 0, 3, 1),
                 (3, 1, 0)]:
        got = fft3d.fftn(x, axes=axes)
        worst = max(worst, rel_error(got, np.fft.fftn(x, axes=axes)))
        back = fft3d.ifftn(got, axes=axes)
        worst = max(worst, rel_error(back, x))
    assert worst < TOL, f"axes handling {worst:.3e}"
    return f"7 axis selections on a 4D array, worst {worst:.2e}"


# ------------------------------------------------------------------ properties

@check
def analytic_special_cases():
    """Cases whose transform is known in closed form."""
    shape = (4, 6, 5)
    volume = int(np.prod(shape))

    delta = np.zeros(shape, dtype=complex)
    delta[0, 0, 0] = 1.0
    err_delta = rel_error(fft3d.fftn(delta), np.ones(shape, dtype=complex))

    ones = np.ones(shape, dtype=complex)
    expect = np.zeros(shape, dtype=complex)
    expect[0, 0, 0] = volume
    err_ones = rel_error(fft3d.fftn(ones), expect)

    # A single Fourier mode transforms to a single spike of weight V.  With the
    # forward kernel exp(-2 pi i k.j/n), the mode carrying exp(+2 pi i k0.j/n) is
    # the one that lands on k0; the opposite sign would land on -k0 mod n.
    k0 = (1, 2, 3)
    j = np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")
    mode = np.exp(+2j * np.pi * sum(kk * jj / n
                                    for kk, jj, n in zip(k0, j, shape)))
    spike = np.zeros(shape, dtype=complex)
    spike[k0] = volume
    err_mode = rel_error(fft3d.fftn(mode), spike)

    for name, err in [("delta", err_delta), ("ones", err_ones), ("mode", err_mode)]:
        assert err < TOL, f"{name}: {err:.3e}"
    return (f"delta->ones {err_delta:.1e}, ones->spike {err_ones:.1e}, "
            f"mode->spike {err_mode:.1e}")


@check
def linearity_and_parseval():
    """Linearity, and `sum|X|^2 = V sum|x|^2` for this normalization."""
    shape = (6, 4, 10)
    x = slow_dft.random_field(shape, seed=1)
    y = slow_dft.random_field(shape, seed=2)
    a, b = 0.3 - 1.7j, 2.1 + 0.4j
    lin = rel_error(fft3d.fftn(a * x + b * y),
                    a * fft3d.fftn(x) + b * fft3d.fftn(y))
    assert lin < TOL, f"linearity {lin:.3e}"

    volume = int(np.prod(shape))
    lhs = float(np.sum(np.abs(fft3d.fftn(x))**2))
    rhs = volume * float(np.sum(np.abs(x)**2))
    par = abs(lhs - rhs) / rhs
    assert par < TOL, f"Parseval {par:.3e}"
    return f"linearity {lin:.1e}, Parseval {par:.1e}"


@check
def shift_theorem():
    """Shifting the input multiplies the transform by a pure phase."""
    shape = (5, 8, 6)
    shift = (2, 3, 1)
    x = slow_dft.random_field(shape, seed=4)
    shifted = np.roll(x, shift, axis=(0, 1, 2))
    k = np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")
    phase = np.exp(-2j * np.pi * sum(kk * d / n
                                     for kk, d, n in zip(k, shift, shape)))
    err = rel_error(fft3d.fftn(shifted), phase * fft3d.fftn(x))
    assert err < TOL, f"shift theorem {err:.3e}"
    return f"rel err {err:.2e}"


@check
def conjugate_symmetry_of_real_input():
    """A real input must produce `X[-k] = conj(X[k])` on the periodic lattice."""
    shape = (6, 8, 4)
    rng = np.random.default_rng(9)
    x = rng.standard_normal(shape).astype(complex)
    X = fft3d.fftn(x)
    mirrored = np.conj(X[np.ix_(*[(-np.arange(n)) % n for n in shape])])
    err = rel_error(X, mirrored)
    assert err < TOL, f"hermitian symmetry {err:.3e}"
    return f"rel err {err:.2e}"


# ------------------------------------------------------------------ interface

@check
def input_is_not_modified():
    """No aliasing: the caller's array must come back untouched."""
    x = slow_dft.random_field((6, 4, 8), seed=8)
    before = x.copy()
    fft3d.fftn(x)
    fft3d.ifftn(x)
    fft3d.fft1d(x[0, 0])
    assert np.array_equal(x, before), "input array was mutated"
    return "fftn / ifftn / fft1d all leave the input alone"


@check
def non_contiguous_and_odd_layouts():
    """Views, transposes, strided slices and real input must all work."""
    rng = np.random.default_rng(6)
    base = (rng.standard_normal((8, 6, 10)) + 1j * rng.standard_normal((8, 6, 10)))
    worst = 0.0
    for name, view in [("transpose", base.transpose(2, 0, 1)),
                       ("strided", base[::2, ::3, ::5]),
                       ("reversed", base[::-1, :, :]),
                       ("real input", base.real)]:
        got = fft3d.fftn(view)
        worst = max(worst, rel_error(got, slow_dft.dft3d_separable(view)))
    assert worst < TOL, f"layout handling {worst:.3e}"
    return f"transpose / strided / reversed / real, worst {worst:.2e}"


@check
def complex64_dtype_is_preserved():
    """A `complex64` input returns `complex64`, computed in double internally."""
    x = slow_dft.random_field((6, 4, 8), seed=2).astype(np.complex64)
    got = fft3d.fftn(x)
    assert got.dtype == np.complex64, f"dtype {got.dtype}"
    err = rel_error(got.astype(np.complex128),
                    slow_dft.dft3d_separable(x.astype(np.complex128)))
    assert err < 1e-6, f"complex64 rel err {err:.3e}"
    also = fft3d.fftn(x.astype(np.complex128))
    assert also.dtype == np.complex128
    return f"complex64 preserved, rel err {err:.2e} (single-precision roundoff)"


@check
def invalid_input_is_rejected():
    """Guard rails: each of these must raise, not silently do something wrong."""
    cases = [
        ("empty array", lambda: fft3d.fftn(np.zeros((0, 4, 4), dtype=complex))),
        ("zero-length 1D", lambda: fft3d.fft1d(np.zeros(0, dtype=complex))),
        ("duplicate axes", lambda: fft3d.fftn(np.zeros((4, 4, 4), dtype=complex),
                                              axes=(0, 0))),
        ("axis out of range", lambda: fft3d.fftn(np.zeros((4, 4, 4), dtype=complex),
                                                axes=(0, 5))),
        ("unknown algorithm", lambda: fft3d.fft1d(np.ones(4, dtype=complex),
                                                  algorithm="radix2")),
        ("bad sign", lambda: fft3d.fft1d(np.ones(4, dtype=complex), sign=0)),
        ("rader on composite", lambda: fft3d.fft1d(np.ones(8, dtype=complex),
                                                   algorithm="rader")),
        ("mixed-radix on prime", lambda: fft3d.fft1d(np.ones(7, dtype=complex),
                                                     algorithm="mixed-radix")),
        ("2D into fft3d", lambda: fft3d.fft3d(np.ones((4, 4), dtype=complex))),
        ("2D array into fft1d", lambda: fft3d.fft1d(np.ones((4, 4), dtype=complex))),
    ]
    for name, fn in cases:
        try:
            fn()
        except Exception:
            continue
        raise AssertionError(f"{name}: no exception raised")
    return f"{len(cases)} invalid inputs all rejected"


@check
def number_theory_helpers():
    """The supporting number theory, checked independently of the FFT."""
    for n in range(1, 200):
        assert int(np.prod(fft3d.factorize(n))) == n, f"factorize({n})"
        assert all(fft3d.is_prime(p) for p in fft3d.factorize(n)), f"factorize({n})"
    primes = [p for p in range(2, 200) if fft3d.is_prime(p)]
    assert primes[:6] == [2, 3, 5, 7, 11, 13], primes[:6]
    for p in primes:
        g = fft3d.primitive_root(p)
        if p > 2:  # g must generate every nonzero residue exactly once
            seen = {pow(g, q, p) for q in range(p - 1)}
            assert seen == set(range(1, p)), f"primitive root {g} of {p}"
    return f"factorize/is_prime on 1..199, primitive roots for {len(primes)} primes"


# ------------------------------------------------------------------ larger sizes

@check
def larger_sizes_against_numpy():
    """Sizes past the slow reference's reach, cross-checked with numpy.fft.

    numpy is not the authority here -- the checks above establish that against
    the definition -- but it is the only affordable witness at these sizes.
    """
    worst, worst_case = 0.0, None
    for shape in [(64, 4, 4), (36, 36, 4), (101, 3, 2), (128, 8, 2), (60, 15, 2)]:
        x = slow_dft.random_field(shape, seed=13)
        err = rel_error(fft3d.fftn(x), np.fft.fftn(x))
        assert err < 1e-11, f"{shape}: {err:.3e}"
        if err > worst:
            worst, worst_case = err, shape
    return f"worst {worst:.2e} at {worst_case}"


# ------------------------------------------------------------------ hardening
#
# The checks below were added after an adversarial audit of fft3d.py: each one
# corresponds to a code path or mathematical property that nothing exercised, and
# five of them are regression tests for defects the audit found (a zero-length batch
# axis rejected, the 1D dtype contract, forced-route guards skipped at n=1, `axes`
# given as an iterator silently transforming nothing, and a cost model whose
# documented direction was backwards).

@check
def zero_length_batch_axis():
    """An empty batch is a well-defined transform; an empty *transformed* axis is not."""
    empty = np.zeros((0, 4, 4, 4), dtype=complex)
    got = fft3d.fftn(empty)
    assert got.shape == np.fft.fftn(empty, axes=(-3, -2, -1)).shape, got.shape
    assert got.dtype == np.complex128
    try:
        fft3d.fftn(np.zeros((4, 0, 4), dtype=complex))
    except ValueError:
        pass
    else:
        raise AssertionError("transforming a length-0 axis should raise")
    return "empty batch -> empty result; length-0 transformed axis rejected"


@check
def dtype_contract_across_entry_points():
    """`complex64` in, `complex64` out -- at every entry point, not just fftn."""
    x32 = slow_dft.random_field((8, 4, 4), seed=1).astype(np.complex64)
    for name, got in [("fftn", fft3d.fftn(x32)), ("ifftn", fft3d.ifftn(x32)),
                      ("fft3d", fft3d.fft3d(x32)), ("ifft3d", fft3d.ifft3d(x32)),
                      ("fft1d", fft3d.fft1d(x32[0, 0])),
                      ("ifft1d", fft3d.ifft1d(x32[0, 0]))]:
        assert got.dtype == np.complex64, f"{name} returned {got.dtype}"
    for name, got in [("float", fft3d.fftn(np.ones((4, 4, 4)))),
                      ("int", fft3d.fftn(np.ones((4, 4, 4), dtype=int))),
                      ("complex128", fft3d.fftn(np.ones((4, 4, 4), dtype=complex)))]:
        assert got.dtype == np.complex128, f"{name} returned {got.dtype}"
    return "complex64 preserved by all 6 entry points; float/int/complex128 -> complex128"


@check
def forced_routes_validate_at_length_one():
    """`n == 1` is the identity but must not bypass a route's precondition."""
    one = np.ones(1, dtype=complex)
    for algorithm in ("rader", "mixed-radix"):
        try:
            fft3d.fft1d(one, algorithm=algorithm)
        except ValueError:
            continue
        raise AssertionError(f"algorithm={algorithm!r} on length 1 should raise")
    for algorithm in ("auto", "direct", "bluestein"):
        got = fft3d.fft1d(one, algorithm=algorithm)
        assert rel_error(got, one) < TOL, f"{algorithm} on length 1"
    return "rader/mixed-radix rejected at n=1; auto/direct/bluestein give the identity"


@check
def axes_may_be_any_iterable():
    """`axes` as a list, tuple, array or generator must all mean the same thing.

    A generator is single-pass: consuming it twice would leave the second pass with
    nothing and silently transform no axes at all.
    """
    x = slow_dft.random_field((4, 5, 3), seed=2)
    reference = slow_dft.dft3d_separable(x)
    worst = 0.0
    for label, make in [("tuple", lambda: (-3, -2, -1)), ("list", lambda: [0, 1, 2]),
                        ("array", lambda: np.array([0, 1, 2])),
                        ("generator", lambda: (a for a in (0, 1, 2)))]:
        worst = max(worst, rel_error(fft3d.fftn(x, axes=make()), reference))
        back = fft3d.ifftn(fft3d.fftn(x, axes=make()), axes=make())
        worst = max(worst, rel_error(back, x))
    assert worst < TOL, f"axes iterable handling {worst:.3e}"
    return f"tuple/list/array/generator all agree, worst {worst:.2e}"


@check
def convolution_theorem():
    """`ifftn(fftn(x) * fftn(y))` is the cyclic convolution of x and y.

    This is the identity Rader's algorithm is built on, and nothing else in the
    suite tests it through the public API.  The convolution is computed here by
    brute-force shifting, independently of any transform.
    """
    shape = (3, 4, 5)
    x = slow_dft.random_field(shape, seed=3)
    y = slow_dft.random_field(shape, seed=4)
    direct = np.zeros(shape, dtype=complex)
    for a in range(shape[0]):
        for b in range(shape[1]):
            for c in range(shape[2]):
                shifted = np.roll(np.roll(np.roll(y, a, 0), b, 1), c, 2)
                direct += x[a, b, c] * shifted
    got = fft3d.ifftn(fft3d.fftn(x) * fft3d.fftn(y))
    err = rel_error(got, direct)
    assert err < TOL, f"convolution theorem {err:.3e}"
    return f"rel err {err:.2e} against brute-force cyclic convolution"


@check
def dft_fourth_power_is_identity():
    """Four forward transforms return `V^2 * x`, which pins sign and index reversal."""
    worst = 0.0
    for shape in [(4, 6, 5), (7, 3, 2), (1, 8, 3)]:
        volume = int(np.prod(shape))
        x = slow_dft.random_field(shape, seed=5)
        four = fft3d.fftn(fft3d.fftn(fft3d.fftn(fft3d.fftn(x))))
        worst = max(worst, rel_error(four, volume ** 2 * x))
    assert worst < TOL, f"DFT^4 {worst:.3e}"
    return f"DFT^4 = V^2 I on 3 shapes, worst {worst:.2e}"


@check
def properties_across_many_geometries():
    """Parseval, the shift theorem and Hermitian symmetry at *every* kind of extent.

    Previously each of these was asserted at one hard-coded smooth shape, so a
    prime-extent or Rader-path bug could not have shown up in them.
    """
    shapes = [(6, 4, 10), (7, 5, 3), (17, 2, 2), (13, 4, 6), (1, 11, 3), (9, 9, 4),
              (23, 2, 1)]
    worst = {"parseval": 0.0, "shift": 0.0, "hermitian": 0.0}
    for shape in shapes:
        volume = int(np.prod(shape))
        x = slow_dft.random_field(shape, seed=sum(shape))
        spectrum = fft3d.fftn(x)

        lhs = float(np.sum(np.abs(spectrum) ** 2))
        rhs = volume * float(np.sum(np.abs(x) ** 2))
        worst["parseval"] = max(worst["parseval"], abs(lhs - rhs) / rhs)

        shift = tuple(min(2, n - 1) if n > 1 else 0 for n in shape)
        k = np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")
        phase = np.exp(-2j * np.pi * sum(kk * d / n
                                         for kk, d, n in zip(k, shift, shape)))
        rolled = fft3d.fftn(np.roll(x, shift, axis=(0, 1, 2)))
        worst["shift"] = max(worst["shift"], rel_error(rolled, phase * spectrum))

        real = np.asarray(np.random.default_rng(sum(shape)).standard_normal(shape),
                          dtype=complex)
        real_spectrum = fft3d.fftn(real)
        mirrored = np.conj(real_spectrum[np.ix_(*[(-np.arange(n)) % n for n in shape])])
        worst["hermitian"] = max(worst["hermitian"], rel_error(real_spectrum, mirrored))

    for name, err in worst.items():
        assert err < TOL, f"{name}: {err:.3e}"
    return (f"{len(shapes)} shapes: Parseval {worst['parseval']:.1e}, "
            f"shift {worst['shift']:.1e}, hermitian {worst['hermitian']:.1e}")


@check
def accuracy_against_extended_precision():
    """Every route judged against an 80-bit reference, not merely against each other.

    Mutual agreement cannot detect a shared accuracy problem -- e.g. Bluestein's
    chirp and Rader's convolution both losing digits at the same size.  This builds
    the dense DFT matrix in `clongdouble` and measures each route against it.
    """
    def extended_reference(x):
        # The exponent is reduced mod n *before* the angle is formed.  Without that
        # reduction the reference itself loses precision in argument reduction (j*k
        # reaches n^2), and it would be the reference, not the code, being measured.
        n = len(x)
        idx = np.arange(n)
        reduced = np.outer(idx, idx) % n
        phase = -2.0 * np.pi * reduced.astype(np.longdouble) / np.longdouble(n)
        matrix = (np.cos(phase) + 1j * np.sin(phase)).astype(np.clongdouble)
        return matrix @ x.astype(np.clongdouble)

    worst, worst_where = 0.0, None
    for n in (7, 11, 13, 16, 17, 31, 36, 64, 97, 128):
        x = slow_dft.random_field((n,), seed=n)
        reference = extended_reference(x)
        routes = ["auto", "direct", "bluestein"]
        routes.append("rader" if fft3d.is_prime(n) else "mixed-radix")
        for route in routes:
            got = fft3d.fft1d(x, algorithm=route).astype(np.clongdouble)
            err = float(np.linalg.norm(got - reference) / np.linalg.norm(reference))
            assert err < 1e-14, f"n={n} {route}: {err:.3e} vs extended precision"
            if err > worst:
                worst, worst_where = err, f"{route} at n={n}"
    return f"10 lengths x 4 routes vs clongdouble, worst {worst:.2e} ({worst_where})"


@check
def one_d_and_nd_entry_points_agree():
    """`fftn` over one axis is `fft1d`; `fftn` over three is three passes of it."""
    x1 = slow_dft.random_field((12,), seed=6)
    worst = rel_error(fft3d.fftn(x1, axes=(0,)), fft3d.fft1d(x1))

    x3 = slow_dft.random_field((4, 6, 5), seed=7)
    manual = x3.astype(np.complex128)
    for axis in (0, 1, 2):
        manual = np.apply_along_axis(lambda line: fft3d.fft1d(line), axis, manual)
    worst = max(worst, rel_error(fft3d.fftn(x3), manual))
    assert worst < TOL, f"1D/nD consistency {worst:.3e}"
    return f"worst {worst:.2e}"


@check
def deep_rader_nesting():
    """Primes whose Rader recursion nests three levels or more.

    167 -> 166 = 2*83 -> 82 = 2*41 -> 40, and 359 -> 358 = 2*179 -> 178 = 2*89 -> 88:
    each level is a fresh Rader call inside a convolution inside a Rader call.
    """
    worst = 0.0
    err = rel_error(fft3d.fft1d(slow_dft.random_field((167,), seed=167)),
                    slow_dft.dft1d_loops(slow_dft.random_field((167,), seed=167)))
    assert err < TOL, f"n=167 vs the definition: {err:.3e}"
    worst = max(worst, err)
    for n in (359, 719):
        x = slow_dft.random_field((n,), seed=n)
        err = rel_error(fft3d.fft1d(x), np.fft.fft(x))
        assert err < 1e-11, f"n={n}: {err:.3e}"
        worst = max(worst, err)
    depth = []
    n = 167
    while fft3d.is_prime(n) and n > fft3d.DIRECT_MAX_PRIME:
        depth.append(n)
        n = max(fft3d.factorize(n - 1))
    return f"167 (vs definition) / 359 / 719 all ok, worst {worst:.2e}; 167 chain {depth}"


@check
def prime_powers_and_large_extents():
    """Prime squares and cubes, and extents past everything else in the suite."""
    worst, worst_where = 0.0, None
    for n in (169, 243, 289, 343, 257, 512):
        x = slow_dft.random_field((n,), seed=n)
        err = rel_error(fft3d.fft1d(x), np.fft.fft(x))
        assert err < 1e-11, f"n={n}: {err:.3e}"
        if err > worst:
            worst, worst_where = err, f"n={n}"
    for shape in [(167, 3, 2), (257, 2, 1), (1, 1, 512), (512, 1, 1)]:
        x = slow_dft.random_field(shape, seed=abs(hash(shape)) % 2**31)
        err = rel_error(fft3d.fftn(x), np.fft.fftn(x))
        assert err < 1e-11, f"{shape}: {err:.3e}"
        if err > worst:
            worst, worst_where = err, str(shape)
    return f"169/243/257/289/343/512 and 4 lopsided 3D shapes, worst {worst:.2e} at {worst_where}"


@check
def ranks_one_two_and_five():
    """`fftn` on rank 1, 2 and 5 arrays, and the degenerate `axes=()`."""
    worst = 0.0
    for shape, axes in [((6,), (0,)), ((5, 7), (0, 1)), ((4, 3, 2, 5, 3), (0, 2, 4)),
                        ((4, 3, 2, 5, 3), (1, 3))]:
        x = slow_dft.random_field(shape, seed=len(shape))
        worst = max(worst, rel_error(fft3d.fftn(x, axes=axes),
                                     np.fft.fftn(x, axes=axes)))
    x = slow_dft.random_field((4, 4, 4), seed=8)
    untouched = fft3d.fftn(x, axes=())
    assert rel_error(untouched, x) < TOL, "axes=() should transform nothing"
    assert rel_error(fft3d.ifftn(x, axes=()), x) < TOL, "ifftn with axes=()"
    assert worst < TOL, f"rank handling {worst:.3e}"
    return f"ranks 1/2/5 match numpy (worst {worst:.2e}); axes=() is the identity"


@check
def outputs_are_independent_and_writable():
    """Results must be fresh writable arrays, never views of the cached twiddle table."""
    x = slow_dft.random_field((4, 4, 4), seed=9)
    first = fft3d.fftn(x)
    assert first.flags.writeable, "returned array is read-only"
    first[:] = 12345.0                        # scribble on it
    second = fft3d.fftn(x)
    assert rel_error(second, slow_dft.dft3d_separable(x)) < TOL, \
        "a later transform was corrupted by mutating an earlier result"

    table = fft3d._twiddle_table(8, fft3d.FORWARD)
    assert not table.flags.writeable, "the cached twiddle table must be read-only"
    line = fft3d.fft1d(np.ones(8, dtype=complex))
    assert line.base is None or not np.shares_memory(line, table), \
        "returned array shares memory with the twiddle cache"
    return "outputs writable and independent; twiddle cache read-only and unshared"


@check
def cost_model_matches_instrumentation():
    """`line_cost` must equal the multiply-adds the code actually performs.

    The previous cost model claimed to be a lower bound for prime extents while
    actually over-charging them by up to ~50x (it billed `p^2` for a length routed
    through Rader).  This pins the model to instrumented counts of the real calls,
    so the claim cannot drift again.
    """
    counted = {"total": 0}
    original = (fft3d._dft_direct, fft3d._fft_mixed_radix, fft3d._fft_rader)

    def direct(x, sign):
        counted["total"] += len(x) ** 2
        return original[0](x, sign)

    def mixed(x, sign):
        counted["total"] += len(x) * fft3d.smallest_prime_factor(len(x))
        return original[1](x, sign)

    def rader(x, sign):
        if len(x) > 2:
            counted["total"] += 2 * (len(x) - 1)
        return original[2](x, sign)

    try:
        fft3d._dft_direct, fft3d._fft_mixed_radix, fft3d._fft_rader = direct, mixed, rader
        for n in (1, 2, 6, 8, 12, 13, 17, 36, 49, 64, 97, 121, 127, 169, 257, 1009):
            counted["total"] = 0
            fft3d.fft1d(np.ones(n, dtype=complex))
            assert counted["total"] == fft3d.line_cost(n), \
                f"n={n}: instrumented {counted['total']} != line_cost {fft3d.line_cost(n)}"
    finally:
        fft3d._dft_direct, fft3d._fft_mixed_radix, fft3d._fft_rader = original

    # and the shorthand it replaced is exact only for DIRECT_MAX_PRIME-smooth n
    smooth = all(p <= fft3d.DIRECT_MAX_PRIME for p in fft3d.factorize(36))
    assert smooth and fft3d.line_cost(36) == 36 * sum(fft3d.factorize(36))
    assert fft3d.line_cost(97) < 97 * sum(fft3d.factorize(97)), \
        "Rader must cost less than the p^2 shorthand"
    return ("16 lengths match instrumented counts exactly; "
            f"line_cost(97)={fft3d.line_cost(97)} vs p^2 shorthand {97*97}")


if __name__ == "__main__":
    print(f"textbook FFT validation -- {len(CHECKS)} checks, "
          f"tolerance {TOL:g}\n")
    failures = 0
    total = time.perf_counter()
    for fn in CHECKS:
        t0 = time.perf_counter()
        try:
            detail = fn()
            status = "PASS"
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            status = "FAIL"
            failures += 1
        dt = time.perf_counter() - t0
        print(f"  [{status}] {fn.__name__:36s} {dt:6.2f}s  {detail}")
        if status == "FAIL":
            traceback.print_exc()
    elapsed = time.perf_counter() - total
    print(f"\n{len(CHECKS) - failures}/{len(CHECKS)} checks passed in {elapsed:.1f}s")
    raise SystemExit(1 if failures else 0)
