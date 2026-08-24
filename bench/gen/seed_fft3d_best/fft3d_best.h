/* fft3d_best -- the best single-core kernel found for each fixed geometry.
 *
 * One entry point covering L = 6, 8, 13, 17, 23, 36, 45, 64. Each geometry is served by the
 * kernel that won the single-threaded competition at that size (eleven measured rounds, on
 * an isolated Xeon Gold 5218); see README.md for the technique and the measured time per L.
 *
 * Forward, unnormalized, complex-double 3D DFT of a cube L^3 over a batch of B volumes,
 * out-of-place:
 *
 *     out[b][k0][k1][k2] = sum_j in[b][j0][j1][j2]
 *                          * exp(-2*pi*i * (k0*j0 + k1*j1 + k2*j2) / L)
 *
 * Layout: contiguous C order, interleaved complex, batch slowest-varying --
 *     element (b, x, y, z)  at  ((b*L + x)*L + y)*L + z
 * Both buffers must be distinct and 64-byte aligned. `in` is not modified.
 *
 * No library is used: the arithmetic is all in kernels/.
 *
 *     fft3d_best_plan *p = fft3d_best_create(17, 256);
 *     for (...) fft3d_best_execute(p, in, out);      // repeatable, out-of-place
 *     fft3d_best_destroy(p);
 */
#ifndef FFT3D_BEST_H
#define FFT3D_BEST_H

#include <complex.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct fft3d_best_plan fft3d_best_plan;

/* Nonzero if a kernel exists for this cube size. */
int fft3d_best_supports(int L);

/* Which kernel serves this L, and how it works (NULL if unsupported). */
const char *fft3d_best_kernel_name(int L);
const char *fft3d_best_kernel_description(int L);

/* Which kernel a given batch size selects -- the choice is batch-dependent at geometries
 * where the measurements showed a real crossover (L=64 today). */
const char *fft3d_best_kernel_name_for(int L, int batch);

/* What a live plan actually selected. The choice is measured in create(), so this is the
 * only authoritative answer. */
const char *fft3d_best_selected_name(const fft3d_best_plan *plan);

/* The second kernel this library carries at this geometry, if any: the runner-up from the
 * competition, kept so a caller can measure the choice on its own machine instead of
 * trusting ours. NULL where there is no alternate. */
const char *fft3d_best_alternate_name(int L);

/* All setup: twiddle tables, permutations, tuning. May be expensive. NULL if unsupported. */
fft3d_best_plan *fft3d_best_create(int L, int batch);

/* The transform. Repeatable: same plan and input give the same output every call. */
void fft3d_best_execute(fft3d_best_plan *plan, const double _Complex *in, double _Complex *out);

void fft3d_best_destroy(fft3d_best_plan *plan);

#ifdef __cplusplus
}
#endif

#endif /* FFT3D_BEST_H */
