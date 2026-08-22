/* The contract every GPU FFT implementation in this competition must satisfy.
 *
 * THE FIXED TASK  (same mathematics as the CPU phases)
 * --------------
 *   Forward, unnormalized, complex-double 3D DFT of a cube L^3, over a batch of B
 *   independent volumes:
 *
 *       out[b][k0][k1][k2] = sum_j in[b][j0][j1][j2]
 *                            * exp(-2*pi*i * (k0*j0 + k1*j1 + k2*j2) / L)
 *
 * LAYOUT
 * ------
 *   Contiguous C order, interleaved complex (double2 = {re, im}):
 *       element (b, x, y, z)  at index  ((b*L + x)*L + y)*L + z
 *   Both buffers are DEVICE allocations made by the driver with cudaMalloc (so they are
 *   256-byte aligned), distinct, out-of-place. `in` must not be modified.
 *
 * WHAT IS TIMED
 * -------------
 *   The transform of device-resident data. Host-to-device and device-to-host copies are
 *   NOT part of your time -- they are measured separately and reported alongside, because
 *   the target workload keeps the field on the GPU across many operations. Your execute
 *   may launch as many kernels as it likes; the driver synchronizes once per timed sample,
 *   so launch overhead inside a sample is yours to pay.
 *
 *   fft3d_gpu_create() is excluded: plan setup, twiddle tables in device memory, kernel
 *   selection, occupancy probing, and any autotuning belong there.
 *
 * RULES
 * -----
 *   * No FFT library calls anywhere inside fft3d_gpu_execute(). Not cuFFT, not cuFFTDx,
 *     not VkFFT, not any vendor DFT. The kernels must be yours. cuFFTDx in particular is
 *     banned as a dependency even though it is header-only and device-side -- read its
 *     design, do not link it.
 *   * cuBLAS/cuTENSOR are also banned inside execute. If a matrix formulation is the right
 *     answer (it was, at the prime size, on the CPU) you write the mma/dmma yourself.
 *   * One GPU. No multi-GPU, no NVSHMEM, no MPI.
 *   * Any CUDA technique is fair game: shared memory, registers, warp shuffles, cp.async,
 *     persistent kernels, __ldg, vectorized double2 loads, tensor-core DMMA, inline PTX,
 *     launch bounds, occupancy tuning, generated code.
 *   * fft3d_gpu_execute() must be repeatable: same plan, same input, same answer.
 *   * Accuracy gate: relative L2 error against numpy below 1e-12. Double precision
 *     throughout -- half or single precision tensor-core tricks cannot meet this gate and
 *     are not a shortcut around it.
 *
 * ERRORS
 * ------
 *   Return NULL from create() for an unsupported size. The driver checks
 *   cudaGetLastError() after your execute and fails the entry loudly if a kernel faulted,
 *   so do not swallow errors.
 */
#ifndef FFT3D_GPU_API_H
#define FFT3D_GPU_API_H

#include <vector_types.h>   /* double2 */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct fft3d_gpu_plan fft3d_gpu_plan;

/* Short identifier, no spaces -- the results key. Must match the source file's stem. */
const char *fft3d_gpu_name(void);

/* One line describing the technique, for the leaderboard. */
const char *fft3d_gpu_description(void);

/* Nonzero if this build can transform a cube of side L. */
int fft3d_gpu_supports(int L);

/* All setup: plans, device twiddle tables, tuning. Returns NULL if unsupported. */
fft3d_gpu_plan *fft3d_gpu_create(int L, int batch);

/* The measured operation. `in` and `out` are DEVICE pointers, distinct, out-of-place.
   Asynchronous work is fine: the driver synchronizes before stopping the clock. */
void fft3d_gpu_execute(fft3d_gpu_plan *plan, const double2 *in, double2 *out);

void fft3d_gpu_destroy(fft3d_gpu_plan *plan);

#ifdef __cplusplus
}
#endif

#endif /* FFT3D_GPU_API_H */
