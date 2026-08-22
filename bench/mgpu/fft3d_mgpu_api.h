/* The contract for the MULTI-GPU competition (phase 4).
 *
 * THE FIXED TASK
 * --------------
 *   Forward, unnormalized, complex-double 3D DFT of a cube L^3 over a batch of B volumes,
 *   distributed across `ngpus` GPUs of one node:
 *
 *       out[b][k0][k1][k2] = sum_j in[b][j0][j1][j2]
 *                            * exp(-2*pi*i * (k0*j0 + k1*j1 + k2*j2) / L)
 *
 *   Geometries here are LARGER than the single-GPU phase, because distribution is only
 *   interesting when there is something to distribute: L = 64, 96, 128, 192, 256. L=64 is
 *   deliberately shared with phase 3 so strong scaling can be measured on the same data.
 *
 * WHY THIS ABI LOOKS DIFFERENT FROM THE SINGLE-GPU ONE
 * ----------------------------------------------------
 *   A distributed transform has no single natural pointer to hand you: the winning
 *   decomposition might be slabs, pencils, or something else, and cuFFT's own multi-GPU
 *   path (cufftXt) insists on its own descriptor layout. So instead of the driver dictating
 *   the distribution, YOU own your device memory:
 *
 *     create()    allocate whatever you like on each GPU; build plans and tables
 *     upload()    take the host buffer and distribute it however you want   (NOT timed)
 *     execute()   transform what was uploaded into your own output buffer   (TIMED)
 *     download()  gather your result back into the host buffer in canonical order
 *                                                                          (NOT timed)
 *
 *   That keeps the measurement on the transform and the communication -- which is the whole
 *   question at this scale -- while letting cufftXt and a hand-written slab code be
 *   compared honestly.
 *
 * HOST LAYOUT (what upload receives and download must produce)
 * -----------
 *   Contiguous C order, interleaved complex, exactly as in the CPU and single-GPU phases:
 *       element (b, x, y, z)  at index  ((b*L + x)*L + y)*L + z
 *
 * REQUIREMENTS
 * ------------
 *   * execute() must be IDEMPOTENT: it reads the uploaded input and writes your output
 *     buffer, so calling it a hundred times gives the same answer a hundred times. Do NOT
 *     transform in place over the input -- the driver times many back-to-back calls, and an
 *     in-place kernel would be transforming its own output from the second call onwards.
 *   * execute() may be asynchronous. The driver synchronizes every device before it stops
 *     the clock, so nothing is counted as free.
 *   * All setup in create(): plans, twiddles, peer-access enabling, IPC handles, NCCL
 *     communicators, pinned staging buffers, autotuning. Excluded from your time.
 *   * Accuracy gate: relative L2 against numpy below 1e-12. Double precision throughout.
 *
 * RULES
 * -----
 *   * No FFT library anywhere inside execute(): not cuFFT, cufftXt, cuFFTMp, cuFFTDx,
 *     VkFFT, heFFTe, cuBLAS or cuTENSOR. The transform is yours.
 *   * COMMUNICATION IS EXPLICITLY ALLOWED and is not "the FFT": cudaMemcpyPeer, peer-mapped
 *     pointers, CUDA IPC, NCCL, and streams/events for overlap are all fair. The interesting
 *     question at 8 GPUs is which communication pattern wins, not whether you may move
 *     bytes. What you may not do is call somebody else's transform.
 *   * One node. No MPI, no NVSHMEM, no multi-node.
 *   * Any CUDA technique: shared memory, registers, warp shuffles, cp.async, tensor cores,
 *     persistent kernels, inline PTX, generated code.
 */
#ifndef FFT3D_MGPU_API_H
#define FFT3D_MGPU_API_H

#include <vector_types.h>   /* double2 */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct fft3d_mgpu_plan fft3d_mgpu_plan;

/* Short identifier, no spaces -- the results key; must match the source file's stem. */
const char *fft3d_mgpu_name(void);

/* One line describing the technique and the decomposition, for the leaderboard. */
const char *fft3d_mgpu_description(void);

/* Nonzero if this build can transform a cube of side L across exactly `ngpus` GPUs. */
int fft3d_mgpu_supports(int L, int ngpus);

/* All setup. `devices` lists the CUDA device ordinals to use, length `ngpus`.
   Returns NULL if unsupported or if allocation fails. */
fft3d_mgpu_plan *fft3d_mgpu_create(int L, int batch, int ngpus, const int *devices);

/* Distribute the host buffer into your device memory. Not timed; called once. */
void fft3d_mgpu_upload(fft3d_mgpu_plan *plan, const double2 *host);

/* THE MEASURED OPERATION. Transform the uploaded input into your output buffer.
   Must be idempotent and must not consume its own output. */
void fft3d_mgpu_execute(fft3d_mgpu_plan *plan);

/* Gather your result into the host buffer in canonical C order. Not timed. */
void fft3d_mgpu_download(fft3d_mgpu_plan *plan, double2 *host);

void fft3d_mgpu_destroy(fft3d_mgpu_plan *plan);

#ifdef __cplusplus
}
#endif

#endif /* FFT3D_MGPU_API_H */
