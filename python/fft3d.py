"""A textbook complex FFT for arbitrary 3D geometries, written from scratch.

This is the *clear* implementation, not the fast one.  It exists to (a) state the
algorithms explicitly, with their derivations in the docstrings, and (b) be the
thing we later optimize against, with `slow_dft.py` as the correctness anchor and
the installed libraries (FFTW, MKL, cuFFT, ...) as the performance yardstick.

Nothing here is tuned: no SIMD, no cache blocking, no in-place butterflies, no
plan reuse, no vectorization over lines.  Complexity is expressed the way a
textbook writes it, one output element at a time.

Geometry coverage
-----------------
Any `n0 x n1 x n2` complex volume, anisotropic included, with each extent an
arbitrary positive integer -- powers of two, smooth composites, primes, and
primes with awkward factorizations of `n-1`.  Three algorithms cover that:

  * **Cooley-Tukey**, mixed-radix decimation in time, for composite `n`.
    With `n = r*m` and `r` the smallest prime factor, split `x` into `r`
    subsequences decimated by `r`, transform each of length `m`, recombine.
    Writing `j = a + r*t` (`a = 0..r-1`, `t = 0..m-1`):

        X[k] = sum_j x[j] W_n^{jk}
             = sum_a W_n^{ak} sum_t x[a+rt] W_n^{rtk}
             = sum_a W_n^{ak} Y_a[k mod m]          (since W_n^r = W_m)

    where `Y_a` is the length-`m` transform of the `a`-th subsequence.  For
    `n = 2^p` this recursion *is* the classic radix-2 butterfly, so no separate
    radix-2 routine is needed.

  * **Rader**, for a prime `n = p`: the `p-1` nonzero-index outputs become a
    cyclic correlation of length `p-1`, which is composite and so handled by
    Cooley-Tukey.  See `_fft_rader`.

  * **Bluestein** (chirp-z), for any `n` at all: the transform is rewritten as a
    linear convolution and evaluated with power-of-two FFTs.  See
    `_fft_bluestein`.  It needs no factorization of `n` whatsoever.

    Worth being explicit, because it is a genuinely interesting consequence: the
    `auto` route never actually selects Bluestein.  Rader reduces a prime `p` to
    a transform of length `p-1`, which is composite for every prime `p > 3`, and
    `p=2,3` are handled directly -- so Cooley-Tukey plus Rader already covers
    every length.  Bluestein is here because it is the algorithm that makes
    "any geometry" true without relying on that argument, it is what a library
    reaches for when the Rader recursion would be too deep, and forcing
    `algorithm="bluestein"` gives the other routes an independent witness to be
    checked against.

The 3D transform is the row-column (separable) algorithm: a 1D FFT along every
line of each axis in turn.  Axes are independent, which is exactly why
anisotropic shapes need no special handling.

Conventions (identical to `slow_dft.py` and to `numpy.fft`)
----------------------------------------------------------
    forward   X[k] = sum_j x[j] exp(-2 pi i k.j / n)        (unnormalized)
    backward  x[j] = (1/V) sum_k X[k] exp(+2 pi i k.j / n)

with `V = prod(n_d)` and the batch index slowest-varying in `(B, n0, n1, n2)`.

The single concession to practicality is `functools.lru_cache` on the twiddle
table: it memoizes `exp(sign*2 pi i j/n)` instead of rebuilding it inside every
recursive call.  That changes no arithmetic and no algorithm -- it only stops the
reference from being unusably slow -- and a table lookup is what a textbook
recommends for accuracy anyway.
"""

import functools

import numpy as np

FORWARD = -1  # sign of the exponent
BACKWARD = +1

# Below this, a prime costs fewer multiply-adds done straight from the definition
# than through Rader's three convolution transforms (p=11: 121 versus 230).
# Accuracy is *not* a reason for the threshold -- measured against an extended
# precision reference the three routes are indistinguishable at these sizes
# (~1e-16 for all of direct/Rader/Bluestein at n = 7..31).
DIRECT_MAX_PRIME = 11

ALGORITHMS = ("auto", "direct", "mixed-radix", "rader", "bluestein")


# ------------------------------------------------------------------ number theory

def factorize(n):
    """Prime factorization of `n >= 1` by trial division, smallest factor first."""
    if n < 1:
        raise ValueError(f"factorize needs n >= 1, got {n}")
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1 if d == 2 else 2
    if n > 1:
        factors.append(n)
    return factors


def smallest_prime_factor(n):
    """Smallest prime factor of `n >= 2`; returns `n` itself when `n` is prime."""
    if n % 2 == 0:
        return 2
    d = 3
    while d * d <= n:
        if n % d == 0:
            return d
        d += 2
    return n


def is_prime(n):
    """True when `n` is prime.  Trial division: clarity over speed."""
    return n >= 2 and smallest_prime_factor(n) == n


def primitive_root(p):
    """Smallest primitive root modulo the prime `p`.

    `g` is a primitive root when its order is exactly `p-1`, which holds iff
    `g^((p-1)/q) != 1 (mod p)` for every prime `q` dividing `p-1`.
    """
    if not is_prime(p):
        raise ValueError(f"primitive_root needs a prime, got {p}")
    if p == 2:
        return 1
    qs = set(factorize(p - 1))
    for g in range(2, p):
        if all(pow(g, (p - 1) // q, p) != 1 for q in qs):
            return g
    raise ArithmeticError(f"no primitive root found modulo {p}")  # unreachable


# ------------------------------------------------------------------ twiddles

@functools.lru_cache(maxsize=None)
def _twiddle_table(n, sign):
    """W[j] = exp(sign * 2 pi i j / n) for j = 0..n-1, as a read-only array.

    Every routine below indexes this table with an exponent reduced mod `n`,
    which keeps the phase error at one rounding of `exp` rather than letting it
    accumulate through repeated complex multiplication.
    """
    table = np.exp(sign * 2j * np.pi * np.arange(n) / n)
    table.setflags(write=False)
    return table


# ------------------------------------------------------------------ 1D kernels

def _dft_direct(x, sign):
    """The definition itself, `O(n^2)`: X[k] = sum_j x[j] W^{jk}."""
    n = len(x)
    w = _twiddle_table(n, sign)
    out = np.empty(n, dtype=np.complex128)
    for k in range(n):
        acc = 0.0 + 0.0j
        for j in range(n):
            acc += x[j] * w[(j * k) % n]
        out[k] = acc
    return out


def _fft_mixed_radix(x, sign):
    """Cooley-Tukey, decimation in time, radix `r` = smallest prime factor of `n`.

    `X[k] = sum_a W_n^{ak} Y_a[k mod m]`, with `Y_a` the length-`m` transform of
    the subsequence `x[a::r]`.  Cost is `O(n * sum of prime factors of n)`.
    """
    n = len(x)
    r = smallest_prime_factor(n)
    if r == n:
        raise ValueError(f"mixed-radix needs a composite length, got prime {n}")
    m = n // r

    # The r decimated subsequences, each transformed at length m.
    sub = [_fft_auto(np.ascontiguousarray(x[a::r]), sign) for a in range(r)]

    w = _twiddle_table(n, sign)
    out = np.empty(n, dtype=np.complex128)
    for k in range(n):
        acc = 0.0 + 0.0j
        for a in range(r):
            acc += w[(a * k) % n] * sub[a][k % m]
        out[k] = acc
    return out


def _cyclic_convolution(u, v):
    """Cyclic convolution `(u * v)[s] = sum_q u[q] v[(s-q) mod m]`.

    Straight from the convolution theorem: multiply the transforms, invert.  The
    inner transforms use the `auto` dispatch at length `m`, which is always
    strictly smaller than the length that asked for the convolution, so the
    recursion terminates.
    """
    m = len(u)
    spectrum = _fft_auto(u, FORWARD) * _fft_auto(v, FORWARD)
    return _fft_auto(spectrum, BACKWARD) / m


def _fft_rader(x, sign):
    """Rader's algorithm: a prime-length DFT as a cyclic correlation.

    For prime `p`, the nonzero indices `1..p-1` form a cyclic group under
    multiplication mod `p`, generated by a primitive root `g`.  Substituting
    `j = g^q` and `k = g^{-s}` turns the exponent `jk` into `g^{q-s}`:

        X[g^{-s}] = x[0] + sum_{q=0}^{p-2} x[g^q] W_p^{g^{q-s}}
                  = x[0] + sum_q a[q] b[(q-s) mod (p-1)]

    with `a[q] = x[g^q]` and `b[q] = W_p^{g^q}`.  That sum is a cyclic
    *correlation* of length `p-1`, i.e. the cyclic convolution of `a` with `b`
    reversed, and `p-1` is composite for every prime `p > 3`.  The remaining
    output is just `X[0] = sum_j x[j]`.
    """
    p = len(x)
    if not is_prime(p):
        raise ValueError(f"Rader needs a prime length, got {p}")
    if p == 2:  # p-1 = 1 leaves no group to work with; the definition is trivial
        return _dft_direct(x, sign)

    m = p - 1
    g = primitive_root(p)

    powers = np.empty(m, dtype=np.int64)  # powers[q] = g^q mod p
    value = 1
    for q in range(m):
        powers[q] = value
        value = (value * g) % p

    w = _twiddle_table(p, sign)
    a = np.array([x[powers[q]] for q in range(m)], dtype=np.complex128)
    b = np.array([w[powers[q]] for q in range(m)], dtype=np.complex128)
    b_reversed = np.array([b[(-t) % m] for t in range(m)], dtype=np.complex128)

    c = _cyclic_convolution(a, b_reversed)

    out = np.empty(p, dtype=np.complex128)
    out[0] = x.sum()
    for s in range(m):
        out[pow(g, -s, p)] = x[0] + c[s]
    return out


def _fft_bluestein(x, sign):
    """Bluestein's chirp-z algorithm: any length, via one linear convolution.

    With `h[t] = W^{t^2/2}` the identity `jk = (j^2 + k^2 - (k-j)^2)/2` gives
    `W^{jk} = h[j] h[k] / h[k-j]`, so

        X[k] = h[k] * sum_j (x[j] h[j]) * conj(h[k-j]),

    a linear convolution of a length-`n` sequence with the two-sided chirp
    kernel.  Embedding it in a cyclic convolution of length `M >= 2n-1` (taken
    to be a power of two, so the inner transforms are pure radix-2) makes the
    wraparound harmless.  `h[-t] = h[t]`, since the chirp depends on `t^2`.

    The exponent is evaluated as `t^2 mod 2n`, exact in integer arithmetic:
    `h[t]` has period `2n` in `t^2`, and reducing first stops the phase from
    being computed from a huge argument.
    """
    n = len(x)
    t = np.arange(n, dtype=np.int64)
    h = np.exp(sign * 1j * np.pi * ((t * t) % (2 * n)) / n)

    size = 1
    while size < 2 * n - 1:
        size *= 2

    padded = np.zeros(size, dtype=np.complex128)
    padded[:n] = x * h

    kernel = np.zeros(size, dtype=np.complex128)
    kernel[:n] = h.conj()          # lags t = 0 .. n-1
    if n > 1:
        kernel[size - n + 1:] = h.conj()[1:][::-1]  # lags t = -(n-1) .. -1

    spectrum = _fft_auto(padded, FORWARD) * _fft_auto(kernel, FORWARD)
    convolved = _fft_auto(spectrum, BACKWARD) / size
    return h * convolved[:n]


def _fft_auto(x, sign):
    """Pick an algorithm by length: recursion base, Cooley-Tukey, or a prime rule."""
    n = len(x)
    if n == 1:
        return x.astype(np.complex128, copy=True)
    if not is_prime(n):
        return _fft_mixed_radix(x, sign)
    if n <= DIRECT_MAX_PRIME:
        return _dft_direct(x, sign)
    return _fft_rader(x, sign)


# ------------------------------------------------------------------ public 1D API

def fft1d(x, sign=FORWARD, algorithm="auto"):
    """1D complex FFT of any length `n >= 1`.

    `algorithm` selects the route, which is what makes the pieces separately
    testable: `auto` (Cooley-Tukey for composites, direct for tiny primes, Rader
    for larger primes), or one of `direct`, `mixed-radix`, `rader`, `bluestein`
    forced.  Every route computes the same transform, to roundoff.
    """
    if algorithm not in ALGORITHMS:
        raise ValueError(f"algorithm must be one of {ALGORITHMS}, got {algorithm!r}")
    if sign not in (FORWARD, BACKWARD):
        raise ValueError(f"sign must be {FORWARD} (forward) or {BACKWARD}, got {sign}")

    original = np.asarray(x)
    x = np.ascontiguousarray(original, dtype=np.complex128)
    if x.ndim != 1:
        raise ValueError(f"fft1d takes a 1D sequence, got shape {x.shape}")
    n = x.shape[0]
    if n == 0:
        raise ValueError("fft1d needs at least one point")

    # n == 1 is the identity, but it is NOT a licence to skip the route's own
    # precondition: algorithm="rader" on length 1 is a caller error (1 is not
    # prime) and must say so rather than quietly returning the input.
    if algorithm == "auto":
        out = _fft_auto(x, sign)
    elif algorithm == "direct":
        out = _dft_direct(x, sign)
    elif algorithm == "mixed-radix":
        out = _fft_mixed_radix(x, sign)
    elif algorithm == "rader":
        out = _fft_rader(x, sign)
    else:
        out = _fft_bluestein(x, sign)
    return out.astype(np.complex64) if original.dtype == np.complex64 else out


def ifft1d(x, algorithm="auto"):
    """Inverse of `fft1d`, including the `1/n` normalization.

    Like `fft1d`, a `complex64` input comes back as `complex64`.
    """
    x = np.asarray(x)
    out = fft1d(x, BACKWARD, algorithm) / x.shape[0]
    return out.astype(np.complex64) if x.dtype == np.complex64 else out


# ------------------------------------------------------------------ public nD API

def _transform_axis(x, axis, sign, algorithm):
    """Apply the 1D FFT to every line of `x` along `axis` (the row-column step)."""
    moved = np.moveaxis(x, axis, -1)
    shape = moved.shape
    lines = np.ascontiguousarray(moved).reshape(-1, shape[-1])
    out = np.empty_like(lines)
    for i in range(lines.shape[0]):
        out[i] = fft1d(lines[i], sign, algorithm)
    return np.moveaxis(out.reshape(shape), -1, axis)


def fftn(x, axes=(-3, -2, -1), sign=FORWARD, algorithm="auto"):
    """Multidimensional complex FFT over `axes`, by the row-column algorithm.

    Axes are transformed one after another and are entirely independent, so
    anisotropic extents -- `48 x 48 x 96`, `13 x 5 x 8`, an extent of 1, a prime
    next to a power of two -- need no special case.  Any axes not listed ride
    along untouched, which is how a batch dimension is handled.

    A `complex64` input comes back as `complex64`; everything else is computed
    and returned in `complex128`.  The arithmetic is always double precision:
    mixed-precision behaviour is a performance question for later, not something
    to bake into the reference.
    """
    x = np.asarray(x)

    # Bounds first: `a % x.ndim` would quietly wrap an out-of-range axis into a
    # valid one (and divide by zero for a 0-d input), so it must not run first.
    # tuple() also materializes an iterator: consuming `axes` twice would silently
    # transform nothing.
    axes = tuple(int(a) for a in axes)
    for a in axes:
        if not -x.ndim <= a < x.ndim:
            raise np.exceptions.AxisError(a, x.ndim)
    normalized = tuple(a % x.ndim for a in axes)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"axes must be distinct, got {axes}")

    # A transformed axis of length 0 has no transform; an *untransformed* axis of
    # length 0 (an empty batch) is fine and gives an empty result, as in numpy.
    for a in axes:
        if x.shape[a] == 0:
            raise ValueError(f"cannot transform axis {a} of length 0 (shape {x.shape})")
    if x.size == 0:
        return x.astype(np.complex64 if x.dtype == np.complex64 else np.complex128)

    single = x.dtype == np.complex64
    out = x.astype(np.complex128, copy=True)
    for axis in axes:
        out = _transform_axis(out, axis, sign, algorithm)
    return out.astype(np.complex64) if single else out


def ifftn(x, axes=(-3, -2, -1), algorithm="auto"):
    """Inverse of `fftn`, including the `1/prod(n_d)` normalization."""
    x = np.asarray(x)
    axes = tuple(int(a) for a in axes)   # must not be a single-pass iterator: it is used twice
    volume = int(np.prod([x.shape[a] for a in axes])) if axes else 1
    return fftn(x, axes, BACKWARD, algorithm) / volume


def fft3d(x, sign=FORWARD, algorithm="auto"):
    """3D complex FFT over the last three axes; leading axes are batch."""
    x = np.asarray(x)
    if x.ndim < 3:
        raise ValueError(f"fft3d needs at least 3 dimensions, got shape {x.shape}")
    return fftn(x, (-3, -2, -1), sign, algorithm)


def ifft3d(x, algorithm="auto"):
    """Inverse of `fft3d`, including the `1/(n0*n1*n2)` normalization."""
    x = np.asarray(x)
    if x.ndim < 3:
        raise ValueError(f"ifft3d needs at least 3 dimensions, got shape {x.shape}")
    return ifftn(x, (-3, -2, -1), algorithm)


# ------------------------------------------------------------------ cost model

@functools.lru_cache(maxsize=None)
def line_cost(n):
    """Complex multiply-adds one length-`n` transform costs on the `auto` route.

    This walks the same dispatch tree `_fft_auto` walks, so it reports what the
    code actually does rather than an idealized model:

      * `n = 1`                    0
      * composite `n = r*m`        `n*r` for the combine, plus `r` sub-transforms of
                                   length `m`
      * prime `n <= DIRECT_MAX_PRIME`   `n^2`, the definition
      * larger prime `p`           Rader: three transforms of length `p-1` (two
                                   forward, one inverse, inside the convolution),
                                   plus `p-1` pointwise products and `p-1` additions

    The textbook shorthand `n * sum(prime factors of n)` agrees with this only when
    every prime factor of `n` is at most `DIRECT_MAX_PRIME`.  For a bare prime it is
    wildly pessimistic -- it charges `p^2`, the very cost Rader exists to avoid -- and
    for a composite with a large prime factor it is wrong in the other direction.
    `test_fft3d.py` pins this function against instrumented counts of the real calls.
    """
    if n < 1:
        raise ValueError(f"line_cost needs n >= 1, got {n}")
    if n == 1:
        return 0
    if not is_prime(n):
        r = smallest_prime_factor(n)
        return n * r + r * line_cost(n // r)
    if n <= DIRECT_MAX_PRIME:
        return n * n
    m = n - 1
    return 3 * line_cost(m) + 2 * m


def operation_count(shape, batch=1):
    """Complex multiply-adds this implementation performs for a whole transform.

    Row-column: each axis of extent `n` costs `line_cost(n)` per line, and there are
    `V/n` lines along it, times the batch.  Useful for seeing which geometries are
    cheap -- `36^3` costs 4.0x what `8^3` does per point, while `17^3` pays the Rader
    premium -- and for sanity-checking a future optimized implementation's own count.

    It counts arithmetic on the `auto` route only; forcing `direct` or `bluestein`
    changes the work, and none of this accounts for Python interpreter overhead,
    which dominates the actual runtime of this reference.
    """
    volume = int(np.prod(shape))
    total = 0
    for n in shape:
        total += (volume // n) * line_cost(int(n)) * batch
    return total


if __name__ == "__main__":
    import time

    print("textbook FFT vs numpy.fft on a spread of geometries\n")
    print(f"  {'shape':>14s} {'rel err':>10s} {'roundtrip':>10s} {'time':>9s}"
          f"   factorization")
    cases = [(8, 8, 8), (16, 16, 16), (12, 18, 24), (7, 11, 13),
             (1, 8, 8), (2, 2, 64), (5, 8, 12), (47, 3, 2), (30, 30, 30)]
    for shape in cases:
        rng = np.random.default_rng(0)
        x = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        t0 = time.perf_counter()
        got = fftn(x)
        dt = time.perf_counter() - t0
        ref = np.fft.fftn(x)
        err = np.linalg.norm(got - ref) / np.linalg.norm(ref)
        back = np.linalg.norm(ifftn(got) - x) / np.linalg.norm(x)
        fac = " x ".join("*".join(map(str, factorize(n))) or "1" for n in shape)
        print(f"  {str(shape):>14s} {err:10.2e} {back:10.2e} {dt:8.2f}s   {fac}")
