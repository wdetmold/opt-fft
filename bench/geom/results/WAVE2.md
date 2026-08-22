# Wave 2: the geometry pool widened after panel_r5

Added: **L = 13, 23, 45, 64**, two competing entries each, alongside the original
L = 6, 8, 17, 36 (which continue to be revised).

| L | why it is here | what it settles |
|---|---|---|
| 13 | prime, and FFTW's generator switches to Rader at exactly rader_min = 13 | whether the dense conjugate-symmetric kernel that beat Rader at 17 still wins where the arithmetic gap is smaller |
| 23 | prime with p-1 = 2*11, not a power of two | whether the 4.98x win at L=17 was about primes or about 17 being a lucky prime |
| 45 | 9*5, coprime, no factor of two anywhere | how to fill SIMD lanes when no radix-2 is available and 4 complex doubles divides neither factor |
| 64 | 2^6, 4.19 MB volume, power-of-two strides throughout | the cache-set conflict / padding question sections 04 and 05 of the corpus disagree about, at the first size where the volume cannot hide in cache |

Rounds panel_r6 through panel_r11 run this wider pool. The new entries start as
stubs whose header comments carry their strategy brief; `fft3d_supports()` returns 0 until
an implementer writes the real kernel, so the harness skips them cleanly in the meantime.

Not included, and why: **anisotropic geometries** (32x32x64, 48x48x96 -- the shapes the real
workload has) need an ABI change, since `fft3d_api.h` takes a single cube side L. That is a
deliberate separate step, not a stub away.
