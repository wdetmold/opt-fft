/* The contract every FFT implementation in this benchmark must satisfy.
 *
 * THE FIXED TASK
 * --------------
 *   Forward, unnormalized, complex-double 3D DFT of a cube L^3, over a batch of
 *   B independent volumes:
 *
 *       out[b][k0][k1][k2] = sum_{j0,j1,j2} in[b][j0][j1][j2]
 *                            * exp(-2*pi*i * (k0*j0 + k1*j1 + k2*j2) / L)
 *
 *   Target sizes are L = 6, 8, 17, 36.  An implementation may specialize as hard
 *   as it likes for one L and refuse the others via fft3d_supports().
 *
 * LAYOUT (identical for every implementation, so the comparison is like for like)
 * ------
 *   Contiguous C order, interleaved complex (i.e. C99 double _Complex):
 *       element (b, x, y, z)  lives at index  ((b*L + x)*L + y)*L + z
 *   Both buffers are allocated by the driver, 64-byte aligned, distinct.
 *   OUT-OF-PLACE: `in` must not be modified; `out` may be written in any order.
 *   An implementation may keep its own internal scratch of any layout, allocated
 *   in fft3d_create().
 *
 * RULES
 * -----
 *   * No FFT library calls anywhere inside fft3d_execute() -- the arithmetic must
 *     be the implementation's own.  Any technique is fair game: unrolled
 *     codelets, PFA/Good-Thomas, Rader, vector radix, SIMD intrinsics, blocking,
 *     precomputed twiddles, generated code.
 *   * MULTICORE.  OpenMP is expected; pthreads are allowed.  The harness fixes the thread
 *     count at OMP_NUM_THREADS=32 with OMP_PROC_BIND=close and OMP_PLACES=cores, and
 *     every backend is measured under identical settings.  Do NOT call omp_set_num_threads()
 *     to grab more than you were given -- the comparison depends on everyone getting the
 *     same cores.  Creating your own thread pool in fft3d_create() is fine and is the point:
 *     thread creation is setup, not transform time.
 *   * All setup, precomputation, table building, and self-tuning belongs in
 *     fft3d_create(), which is timed separately and excluded from the reported
 *     transform time.  It may be arbitrarily expensive.
 *   * fft3d_execute() must be callable repeatedly with the same plan and give
 *     the same answer every time.
 *
 * TIMING (what the driver does, so you know what you are being measured on)
 * ------
 *   Compilation and fft3d_create() are excluded.  Warmup calls are excluded.
 *   The driver auto-calibrates an inner repeat count so each timed sample is
 *   long enough to clear timer resolution, then takes many samples and reports
 *   the distribution; the whole thing is repeated in several independent
 *   processes.  Batched (B>1) and non-batched (B=1) are separate measurements.
 */
#ifndef FFT3D_API_H
#define FFT3D_API_H

#include <complex.h>

typedef struct fft3d_plan fft3d_plan;

/* Short identifier, no spaces -- used as the results key. */
const char *fft3d_name(void);

/* One line describing the technique, for the leaderboard. */
const char *fft3d_description(void);

/* Nonzero if this build can transform a cube of side L. */
int fft3d_supports(int L);

/* All setup and precomputation.  Returns NULL if unsupported. */
fft3d_plan *fft3d_create(int L, int batch);

/* The measured operation: forward transform of the whole batch, out-of-place. */
void fft3d_execute(fft3d_plan *plan, const double _Complex *in, double _Complex *out);

void fft3d_destroy(fft3d_plan *plan);

#endif /* FFT3D_API_H */
