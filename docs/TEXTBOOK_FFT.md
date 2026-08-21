# The textbook FFT — `python/fft3d.py`

A from-scratch complex 3D FFT for arbitrary geometries, written for clarity rather than
speed. It is the thing we will later optimize; `python/slow_dft.py` is the correctness
anchor and the installed libraries are the performance yardstick.

## How a length gets transformed

One 1D engine handles every extent, choosing among three classical algorithms:

| length `n` | route | why |
|---|---|---|
| 1 | identity | nothing to do |
| composite | **mixed-radix Cooley-Tukey**, decimation in time, radix = smallest prime factor | `X[k] = Σ_a W_n^{ak} Y_a[k mod m]` for `n = r·m`, recursing on the `r` subsequences `x[a::r]`. For `n = 2^p` this *is* the classic radix-2 butterfly |
| prime ≤ 11 | **direct `O(n²)`** | the definition is cheaper and more accurate than Rader's machinery at this size |
| prime > 11 | **Rader** | the `p-1` nonzero-index outputs are a cyclic correlation of length `p-1`, which is composite for every prime `p > 3`, so Cooley-Tukey finishes the job |
| anything, on request | **Bluestein** (chirp-z) | `jk = (j²+k²-(k-j)²)/2` turns the transform into one linear convolution, evaluated at a power-of-two length ≥ `2n-1`. The universal fallback: never needs `n` to factor at all |

The 3D transform is row-column: a 1D FFT along every line of each axis in turn. Axes are
independent, which is precisely why **anisotropic shapes need no special case** — a
`13 × 5 × 8` volume is a Rader axis, a direct-prime axis and a radix-2 axis, composed.

`algorithm=` on `fft1d`/`fftn` forces any single route, which is what makes the pieces
separately testable: the same input through `direct`, `mixed-radix`/`rader` and
`bluestein` must agree to roundoff.

## Route taken by extents we care about

Every realistic lattice extent is smooth — `2^a·3^b` — so production geometries never
leave the Cooley-Tukey path. Rader and Bluestein exist so that "any geometry" is true
rather than nearly true:

| n | factorization | route |
|---|---|---|
| 16, 32, 64, 128, 256 | `2^4 … 2^8` | Cooley-Tukey, radix 2 throughout |
| 24, 48, 96, 192 | `2^3·3`, `2^4·3`, `2^5·3`, `2^6·3` | Cooley-Tukey, radix 2 then 3 |
| 12, 18, 20, 36, 40, 72, 144 | mixed 2/3/5 | Cooley-Tukey |
| 28, 44, 56, 112 | `2^k·7`, `2^2·11` | Cooley-Tukey with one large final radix |
| 7, 11 | prime | direct `O(n²)` |
| 13, 17, 19, 31, 37, 41, 97, 127 | prime | Rader; convolution length `n-1` is smooth |
| 23, 47, 53 | prime | Rader, and `n-1` contains a large prime (`22 = 2·11`, `46 = 2·23`, `52 = 2²·13`) so Rader nests inside itself |

## Validation

`python/test_fft3d.py` — 20 checks, all passing, ~1.6 s:

- **against the definition**: every 1D length 1…40 vs `dft1d_loops`; five small volumes vs
  `dft3d_loops` (the literal 6-fold sum); 44 geometries vs `dft3d_separable`
- **route cross-checks**: 29 lengths × {direct, auto, rader-or-mixed-radix, bluestein} agree;
  Bluestein vs Rader at primes 101…251
- **properties**: forward/backward against the reference's own sign convention, inverse
  roundtrip, linearity, Parseval, shift theorem, Hermitian symmetry of real input,
  δ → ones, ones → spike, single mode → single spike
- **interface**: batched `(B, n0, n1, n2)`, arbitrary `axes` on a 4D array (7 selections),
  non-contiguous / transposed / strided / reversed / real inputs, `complex64` preserved,
  input never mutated, 10 invalid inputs rejected with the right exception
- **helpers**: `factorize`/`is_prime` on 1…199, primitive roots verified to generate the
  full multiplicative group for all 46 primes below 200

Worst relative error anywhere in the sweep: **1.7e-14**, at `(97, 2, 1)` — a Rader axis
whose convolution length is `96`. Typical error is `~1e-15`.

Geometry classes covered by the sweep: powers of two; smooth composites; prime powers
(`25`, `27`, `49`); bare primes on the direct path (`3`, `5`, `7`, `11`) and the Rader path
(`13`, `17`, `19`, `23`, `47`, `83`, `97`); primes whose `p-1` has a large prime factor,
forcing nested Rader; unit extents in every position; extreme aspect ratios (`1×1×16`,
`2×2×64`, `64×2×2`); and anisotropic mixtures (`7×11×13`, `9×25×7`, `13×5×8`, `24×18×12`).

## Deliberately not done

No SIMD, no cache blocking, no in-place butterflies, no plan reuse, no vectorization over
lines, no split-radix, no real-input specialization. The one concession is
`functools.lru_cache` on the twiddle table, which memoizes `exp(sign·2πi j/n)` rather than
rebuilding it in every recursive call: it changes no arithmetic and no algorithm, and
table lookup with an exponent reduced mod `n` is what a textbook recommends for accuracy
anyway.
